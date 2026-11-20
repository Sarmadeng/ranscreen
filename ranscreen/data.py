"""ranscreen.data — loading, quality control and the trace-driven environment.

Everything downstream reads the environment from here, so the data path and the quality filter
are defined once and are identical across every experiment in the paper.
"""
from __future__ import annotations
import os, glob, json
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# --- LTE CQI table (TS 36.213, Table 7.2.3-1): spectral efficiency, CQI index 0..15
CQI_EFFICIENCY_LTE = np.array([0.0, .1523, .2344, .3770, .6016, .8770, 1.1758, 1.4766,
                               1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547])
# --- NR CQI table 2 (TS 38.214, Table 5.2.2.1-3), which reaches 256QAM. The fifth-generation
#     corpus is non-standalone and 82.9% of its retained rows report NR, so this is the table
#     that applies there. Used for the sensitivity analysis reported in the paper.
CQI_EFFICIENCY_NR = np.array([0.0, .1523, .3770, .8770, 1.4766, 1.9141, 2.4063, 2.7305,
                              3.3223, 3.9023, 4.5234, 5.1152, 5.5547, 6.2266, 6.9141, 7.4063])
# --- SINR [dB] at which each CQI index becomes reportable. The efficiency values above are
#     normative; this threshold vector is NOT: CQI reporting is implementation-defined in both
#     releases, and this is the usual engineering mapping. The NR vector keeps the LTE
#     thresholds for the indices whose modulation and code rate coincide and extends the three
#     256QAM entries in 2 dB steps. Both are released with the code so the choice is auditable.
CQI_SINR_DB_LTE = np.array([-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3,
                            18.7, 21.0, 22.7])
CQI_SINR_DB_NR = np.array([-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3,
                           18.3, 20.3, 22.3])

# the active table; build_environment() switches it per corpus
CQI_EFFICIENCY = CQI_EFFICIENCY_LTE
CQI_SINR_DB = CQI_SINR_DB_LTE
SE_MAX = CQI_EFFICIENCY.max()


def use_cqi_table(which: str = 'lte') -> str:
    """Select the CQI table used by sinr_to_se(). 'lte' = TS 36.213, 'nr' = TS 38.214 table 2."""
    global CQI_EFFICIENCY, CQI_SINR_DB, SE_MAX
    if which == 'nr':
        CQI_EFFICIENCY, CQI_SINR_DB = CQI_EFFICIENCY_NR, CQI_SINR_DB_NR
    else:
        CQI_EFFICIENCY, CQI_SINR_DB = CQI_EFFICIENCY_LTE, CQI_SINR_DB_LTE
    SE_MAX = CQI_EFFICIENCY.max()
    return which

NUMCOLS = ['RSRP', 'RSRQ', 'SNR', 'CQI', 'RSSI', 'NRxRSRP', 'NRxRSRQ',
           'DL_bitrate', 'UL_bitrate', 'Speed']


# Dataset registry: both corpora share the same column schema, so one pipeline serves both.
DATASETS = {
    '4g': dict(root=os.environ.get('RANSCREEN_4G_ROOT', 'data/4G_Dataset'), mob_level=-2, label='4G LTE (135 traces)'),
    '5g': dict(root=os.environ.get('RANSCREEN_5G_ROOT', 'data/5G-production-dataset'), mob_level=1,
               label='5G production (83 traces)'),
}


@dataclass
class Config:
    dataset: str = '4g'
    data_dir: str = ''                # filled from DATASETS[dataset] when left empty
    rsrp_min: float = -140.0          # 3GPP reporting range
    rsrp_max: float = -44.0
    margin_max: float = 25.0          # |neighbour - serving| plausibility limit [dB]
    ho_cost: float = 0.045            # replaced by the value calibrated from real cell changes
    speed_scale: float = 0.0015       # extra interruption cost per m/s
    pingpong_pen: float = 0.03
    mc_draws: int = 400               # Monte Carlo draws for the oracle
    seed: int = 0
    sinr_folds: int = 5


def sinr_to_se(sinr_db) -> np.ndarray:
    """SINR [dB] -> CQI index -> spectral efficiency, normalised to [0, 1]."""
    cqi = np.searchsorted(CQI_SINR_DB, np.asarray(sinr_db), side='right')
    return CQI_EFFICIENCY[cqi]/SE_MAX


def load_raw(cfg: Config) -> pd.DataFrame:
    root = cfg.data_dir or DATASETS[cfg.dataset]['root']
    files = sorted(glob.glob(os.path.join(root, '**', '*.csv'), recursive=True))
    files = [f for f in files if not os.path.basename(f).startswith('._')]
    if not files:
        raise FileNotFoundError(f'no CSV files under {root}')
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d['trace'] = os.path.splitext(os.path.basename(f))[0]
        rel = os.path.relpath(f, root).split(os.sep)
        lvl = DATASETS[cfg.dataset]['mob_level']
        d['mob'] = rel[lvl] if abs(lvl) <= len(rel) else 'all'
        d['app'] = rel[0] if len(rel) > 1 else 'all'
        frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    D.columns = [c.strip() for c in D.columns]
    for c in NUMCOLS:
        if c in D.columns:
            D[c] = pd.to_numeric(D[c].astype(str).str.strip().replace({'-': np.nan, '': np.nan}),
                                 errors='coerce')
    return D


def quality_control(D: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Pre-registered filter. Returns the retained frame and a provenance record."""
    n0 = len(D)
    D = D.dropna(subset=['RSRP', 'NRxRSRP', 'DL_bitrate'])
    D = D[D.RSRP.between(cfg.rsrp_min, cfg.rsrp_max) &
          D.NRxRSRP.between(cfg.rsrp_min, cfg.rsrp_max) & (D.DL_bitrate >= 0)]
    D = D[(D.NRxRSRP - D.RSRP).abs() <= cfg.margin_max].reset_index(drop=True)
    imputed = {c: float(1 - D[c].notna().mean()) for c in ['SNR', 'CQI', 'RSRQ', 'NRxRSRQ']}
    D['SNR'] = D['SNR'].fillna(pd.Series((D.RSRP + 112)/3.2, index=D.index))
    D['CQI'] = D['CQI'].fillna(np.clip(D.SNR/2.1 + 3, 0, 15))
    D['RSRQ'] = D['RSRQ'].fillna(D['RSRQ'].median())
    D['NRxRSRQ'] = D['NRxRSRQ'].fillna(D['NRxRSRQ'].median())
    D['Speed'] = D['Speed'].fillna(D['Speed'].median())
    if D['Speed'].max() > 60:                      # the dataset reports km/h
        D['Speed'] = D['Speed']/3.6
    D['a3_margin'] = D.NRxRSRP - D.RSRP
    prov = dict(rows_raw=int(n0), rows_kept=int(len(D)), retained=float(len(D)/n0),
                traces=int(D.trace.nunique()), imputed_fraction=imputed,
                neighbour_stronger=float((D.a3_margin > 0).mean()))
    return D, prov


def calibrate_ho_cost(D_raw: pd.DataFrame) -> tuple[float, int, float]:
    """Interruption cost measured from real serving-cell changes in the traces."""
    ratios = []
    for _, g in D_raw.groupby('trace'):
        bits = pd.to_numeric(g.DL_bitrate, errors='coerce').values
        cid = g.CellID.values
        for i in np.where(cid[1:] != cid[:-1])[0] + 1:
            pre, post = np.nanmean(bits[max(0, i-6):i]), np.nanmean(bits[i:i+2])
            if np.isfinite(pre) and np.isfinite(post) and pre > 50:
                ratios.append(post/pre)
    ratios = np.asarray(ratios)
    med = float(np.median(ratios))
    return float(np.clip(1 - med, .005, .3)), int(len(ratios)), med


def fit_sinr_model(D: pd.DataFrame, cfg: Config):
    """(RSRP, RSRQ) -> SINR, validated across traces so no trace is in both train and test."""
    X, y, g = D[['RSRP', 'RSRQ']].values, D.SNR.values, D.trace.values
    oof = np.zeros(len(D)); folds = []
    for a, b in GroupKFold(n_splits=cfg.sinr_folds).split(X, y, g):
        m = HistGradientBoostingRegressor(max_depth=5, max_iter=250, learning_rate=.07,
                                          random_state=cfg.seed).fit(X[a], y[a])
        oof[b] = m.predict(X[b])
        folds.append(1 - ((y[b]-oof[b])**2).sum()/((y[b]-y[b].mean())**2).sum())
    model = HistGradientBoostingRegressor(max_depth=5, max_iter=250, learning_rate=.07,
                                          random_state=cfg.seed).fit(X, y)
    return model, y - oof, float(np.mean(folds)), float(np.std(folds))


@dataclass
class Environment:
    """Trace-driven contextual bandit with an exactly known oracle."""
    D: pd.DataFrame
    sinr: np.ndarray            # (n, 2) estimated SINR of serving and neighbour cell
    resid: np.ndarray           # SINR prediction residuals, resampled as measurement noise
    q_base: np.ndarray          # (n, 2) cell values without the interruption cost
    q_oracle: np.ndarray        # (n, 2) E[reward actually drawn], cost and clipping inside
    ho_penalty: np.ndarray
    prov: dict = field(default_factory=dict)

    # ---- context accessors
    @property
    def rsrp_s(self): return self.D.RSRP.values
    @property
    def rsrp_n(self): return self.D.NRxRSRP.values
    @property
    def rsrq_s(self): return self.D.RSRQ.values
    @property
    def rsrq_n(self): return self.D.NRxRSRQ.values
    @property
    def speed(self): return self.D.Speed.values
    @property
    def mob(self): return self.D.mob.values
    @property
    def trace(self): return self.D.trace.values

    def __len__(self): return len(self.D)

    def draw_reward(self, rng, idx, a):
        """Noise enters in the SINR domain and is mapped through the CQI table."""
        s = self.sinr[idx, a] + rng.choice(self.resid, size=len(idx))
        return np.clip(sinr_to_se(s) - a*self.ho_penalty[idx], 0, 1)

    def value(self, P, idx) -> float:
        """Exact value of a policy on a set of contexts."""
        return float((P*self.q_oracle[idx]).sum(1).mean())

    def gain(self, P, P0, idx) -> float:
        return float(((P - P0)*self.q_oracle[idx]).sum(1).mean())


def build_environment(cfg: Config | None = None, verbose: bool = True) -> Environment:
    cfg = cfg or Config()
    raw = load_raw(cfg)
    D, prov = quality_control(raw, cfg)
    cost, n_events, med = calibrate_ho_cost(raw)
    cfg.ho_cost = cost
    prov.update(ho_cost_calibrated=cost, ho_events=n_events, ho_median_ratio=med)

    model, resid, r2, r2sd = fit_sinr_model(D, cfg)
    prov.update(sinr_r2=r2, sinr_r2_sd=r2sd, sinr_resid_sd=float(resid.std()))

    sinr = np.column_stack([model.predict(D[['RSRP', 'RSRQ']].values),
                            model.predict(D[['NRxRSRP', 'NRxRSRQ']].values)])
    ho_pen = cfg.ho_cost + cfg.speed_scale*D.Speed.values

    # The oracle is the expectation of exactly the reward that draw_reward produces: the cost and
    # the clipping are inside the expectation, and the nonlinear CQI map is averaged, not evaluated
    # at the mean. Doing either outside the expectation biases the oracle by an amount larger than
    # the policy differences under test (see the paper, Section on benchmark construction).
    rng = np.random.default_rng(cfg.seed)
    q_base = np.zeros_like(sinr); q_oracle = np.zeros_like(sinr)
    for _ in range(cfg.mc_draws):
        e = rng.choice(resid, size=sinr.shape)
        q_base += sinr_to_se(sinr + e)
        q_oracle[:, 0] += np.clip(sinr_to_se(sinr[:, 0] + e[:, 0]), 0, 1)
        q_oracle[:, 1] += np.clip(sinr_to_se(sinr[:, 1] + e[:, 1]) - ho_pen, 0, 1)
    q_base /= cfg.mc_draws; q_oracle /= cfg.mc_draws
    prov['oracle_shift_vs_point_map'] = float(np.abs(q_base - sinr_to_se(sinr)).mean())
    prov['config'] = asdict(cfg)

    if verbose:
        print(f"{prov['rows_kept']:,} samples ({prov['retained']:.1%} retained) "
              f"from {prov['traces']} traces")
        print(f"handover cost calibrated on {n_events} real cell changes: {cost:.3f}")
        print(f"SINR model grouped-CV R2 = {r2:.3f} +- {r2sd:.3f}")
    return Environment(D=D, sinr=sinr, resid=resid, q_base=q_base, q_oracle=q_oracle,
                       ho_penalty=ho_pen, prov=prov)
