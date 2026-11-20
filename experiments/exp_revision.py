"""Revision experiments requested by Reviewer 1.

R1-a  Operational failure rate: the fraction of logs in which the gate ACCEPTS a candidate
      whose true gain is non-positive. Computed from the per-replicate records already
      written by audit_main.py, so no resampling is needed.
R1-b  Diagnostics with an out-of-sample threshold: the threshold is fitted on one corpus and
      tested on the other, which removes the in-sample bias of the original Table 9. The
      raw-weight (Kish) ESS is included alongside the truncated one.
R1-c  Retention bias: does the plausibility filter remove rows systematically by mobility
      class, and does that bear on the regime-shift results?

Writes results/revision/*.csv and prints every number that goes into the manuscript.
"""
from __future__ import annotations
import sys, os, json
import pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd

RES = str(RESULTS) + ''
OUT = f'{RES}/revision'
os.makedirs(OUT, exist_ok=True)
NOMINAL = 0.95
summary = {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return float(c - h), float(c + h)


# =============================================================== R1-a operational failure
print('=' * 78)
print('R1-a  OPERATIONAL FAILURE RATE  (gate accepts while the true gain is <= 0)')
print('=' * 78)
op_rows = []
for ds in ('4g', '5g'):
    raw = pd.read_csv(f'{RES}/{ds}/audit_sweep_raw.csv')
    for charged in (False, True):
        d = raw[(raw.kind == 'dr') & (raw.charged == charged)].copy()
        d['accept'] = d.lb_t > 0
        d['harmful'] = d.d_true <= 0
        d['op_fail'] = d.accept & d.harmful
        # per operating point
        g = d.groupby(['temp0', 'n', 'policy'], as_index=False).agg(
            reps=('op_fail', 'size'), fails=('op_fail', 'sum'),
            harmful=('harmful', 'first'), d_true=('d_true', 'first'))
        g['rate'] = g.fails/g.reps
        g['dataset'] = ds
        g['charged'] = charged
        op_rows.append(g)
        harmful_cells = g[g.harmful]
        overall = d.op_fail.mean()
        worst = harmful_cells.rate.max() if len(harmful_cells) else 0.0
        lo, hi = wilson(int(d.op_fail.sum()), len(d))
        tag = 'charged  ' if charged else 'standard '
        print(f'  {ds.upper()} {tag} overall {overall:.4f} [{lo:.4f},{hi:.4f}]   '
              f'worst harmful cell {worst:.3f}   harmful cells {len(harmful_cells)}')
        summary[f'op_fail_{ds}_{"charged" if charged else "standard"}'] = dict(
            overall=float(overall), lo=lo, hi=hi, worst_cell=float(worst),
            n_harmful_cells=int(len(harmful_cells)))
pd.concat(op_rows).to_csv(f'{OUT}/operational_failure.csv', index=False)


# =============================================================== R1-b out-of-sample thresholds
print()
print('=' * 78)
print('R1-b  DIAGNOSTICS, THRESHOLD FITTED ON ONE CORPUS AND TESTED ON THE OTHER')
print('=' * 78)


def balanced_accuracy(pred, y):
    tp = np.sum(pred & y); tn = np.sum(~pred & ~y)
    fn = np.sum(~pred & y); fp = np.sum(pred & ~y)
    sens = tp/(tp + fn) if (tp + fn) else 0.0
    spec = tn/(tn + fp) if (tn + fp) else 0.0
    return 0.5*(sens + spec)


def fit_threshold(x, y):
    best = (-1.0, None, None)
    for thr in np.unique(x):
        for direction in (+1, -1):
            pred = (x <= thr) if direction > 0 else (x >= thr)
            ba = balanced_accuracy(pred, y)
            if ba > best[0]:
                best = (ba, float(thr), direction)
    return best


cells = {}
for ds in ('4g', '5g'):
    ess = pd.read_csv(f'{RES}/{ds}/ess_raw_check.csv')
    sw = pd.read_csv(f'{RES}/{ds}/audit_sweep_summary.csv')
    std = sw[(sw.kind == 'dr') & (~sw.charged)][['temp0', 'n', 'policy', 'cov_t',
                                                 'cov_t_lo', 'cov_t_hi']]
    std = std.rename(columns={'n': 'n_log'})
    m = ess.merge(std, on=['temp0', 'n_log', 'policy'], how='inner')
    # a cell FAILS when its whole Wilson interval lies below the nominal level; it PASSES when
    # the interval contains or exceeds it. Cells straddling the line are excluded, so the
    # label is not a function of sampling noise.
    m['fails'] = m.cov_t_hi < NOMINAL
    m['passes'] = m.cov_t_lo >= NOMINAL
    m = m[m.fails | m.passes].copy()
    cells[ds] = m
    print(f'  {ds.upper()}: {len(m)} unambiguous cells  ({int(m.fails.sum())} failing, '
          f'{int(m.passes.sum())} passing)')

DIAGS = [('ess_trunc_frac', 'ESS on truncated weights (fraction of n)'),
         ('ess_raw_frac',   'ESS on raw weights, Kish (fraction of n)'),
         ('wmax',           'maximum raw importance weight'),
         ('charge',         'charge c-bar')]
rows = []
for col, label in DIAGS:
    line = {'diagnostic': label, 'column': col}
    for fit_ds, test_ds in (('4g', '5g'), ('5g', '4g')):
        f, t = cells[fit_ds], cells[test_ds]
        ba_in, thr, direction = fit_threshold(f[col].values, f.fails.values)
        pred = (t[col].values <= thr) if direction > 0 else (t[col].values >= thr)
        ba_out = balanced_accuracy(pred, t.fails.values)
        line[f'in_sample_{fit_ds}'] = round(float(ba_in), 3)
        line[f'threshold_{fit_ds}'] = float(thr)
        line[f'out_of_sample_{fit_ds}_to_{test_ds}'] = round(float(ba_out), 3)
    rows.append(line)
    print(f'  {label:<42} in-sample 4G {line["in_sample_4g"]:.3f} -> out-of-sample 5G '
          f'{line["out_of_sample_4g_to_5g"]:.3f} | in-sample 5G {line["in_sample_5g"]:.3f} -> '
          f'out-of-sample 4G {line["out_of_sample_5g_to_4g"]:.3f}')
diag = pd.DataFrame(rows)
diag.to_csv(f'{OUT}/diagnostics_out_of_sample.csv', index=False)
summary['diagnostics'] = rows


# =============================================================== R1-c retention bias
print()
print('=' * 78)
print('R1-c  DOES THE PLAUSIBILITY FILTER REMOVE ROWS SYSTEMATICALLY BY MOBILITY CLASS?')
print('=' * 78)
from ranscreen.data import Config, load_raw, quality_control   # noqa: E402
from ranscreen.paths import RESULTS

ret_rows = []
for ds in ('4g', '5g'):
    cfg = Config(dataset=ds)
    D0 = load_raw(cfg)
    D1, prov = quality_control(D0.copy(), cfg)
    before = D0.mob.value_counts(normalize=True)
    after = D1.mob.value_counts(normalize=True)
    kept = D1.mob.value_counts()
    tot = D0.mob.value_counts()
    print(f'  {ds.upper()}  overall retention {prov["retained"]:.3f}')
    for k in before.index:
        r = float(kept.get(k, 0))/float(tot.get(k, 1))
        ret_rows.append(dict(dataset=ds, mob=str(k), share_before=float(before.get(k, 0)),
                             share_after=float(after.get(k, 0)), retention=r,
                             n_before=int(tot.get(k, 0)), n_after=int(kept.get(k, 0))))
        print(f'    {str(k):<16} share before {before.get(k, 0):.3f} -> after '
              f'{after.get(k, 0):.3f}   retained {r:.3f}')
    summary[f'retention_{ds}'] = {str(k): float(kept.get(k, 0))/float(tot.get(k, 1))
                                  for k in before.index}

pd.DataFrame(ret_rows).to_csv(f'{OUT}/retention_by_mobility.csv', index=False)

with open(f'{OUT}/revision_numbers.json', 'w') as fh:
    json.dump(summary, fh, indent=2, default=float)
print(f'\nwrote {OUT}/')
