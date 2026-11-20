"""Reviewer 1, point on Contribution 2: the ESS in the paper is computed on the TRUNCATED
weights, which removes exactly the tail that causes the failure, so its blindness may be
circular. This script recomputes, per operating point, Kish's effective sample size on the
RAW importance weights and asks whether that version separates failing from passing cells.

Output: results/<ds>/ess_raw_check.csv  and a printed separation summary.
"""
from __future__ import annotations
import sys, os
import pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from ranscreen import build_environment, Config, probs, sample_log, CANDIDATES, estimators as E
from ranscreen.paths import RESULTS, read_records

DS     = sys.argv[1] if len(sys.argv) > 1 else '4g'
N_REP  = int(sys.argv[2]) if len(sys.argv) > 2 else 20   # ESS is a stable per-cell statistic
TEMPS  = (0.2, 0.4, 0.8, 1.2, 2.0)
SIZES  = (1_000, 5_000, 20_000)
OUT    = f'{RESULTS}/{DS}'

CACHE = f'{OUT}/ess_raw_check.csv'
if os.path.exists(CACHE) and os.environ.get('REUSE') == '1':
    cell = pd.read_csv(CACHE)
else:
  env = build_environment(Config(dataset=DS))
  rows = []
  for temp0 in TEMPS:
      for n_log in SIZES:
          for s in range(N_REP):
              rng = np.random.default_rng(900_000 + s)
              log = sample_log(env, rng, n_log, temp0)
              for name, (h, t) in CANDIDATES.items():
                  P  = probs(env, log['idx'], h, t)
                  ar = np.arange(log['n'])
                  wr = P[ar, log['a']]/np.maximum(log['prop'], 1e-300)      # raw weight
                  wr = np.minimum(wr, E.WCAP)
                  wM = np.minimum(wr, E.M_DEFAULT)                          # truncated weight
                  rows.append(dict(
                      temp0=temp0, n_log=n_log, policy=name, seed=s,
                      ess_raw=float(wr.sum()**2/np.sum(wr**2)),             # Kish, raw
                      ess_trunc=float(wM.sum()**2/np.sum(wM**2)),           # as printed in the paper
                      ess_raw_frac=float(wr.sum()**2/np.sum(wr**2))/n_log,
                      ess_trunc_frac=float(wM.sum()**2/np.sum(wM**2))/n_log,
                      wmax=float(wr.max()),
                      charge=float(E.charge(P, log['P0'], 'dr').mean())))
  df = pd.DataFrame(rows)
  cell = df.groupby(['temp0', 'n_log', 'policy'], as_index=False).mean(numeric_only=True)
os.makedirs(OUT, exist_ok=True)
cell.to_csv(f'{OUT}/ess_raw_check.csv', index=False)

# ---- join with the measured coverage of the standard certificate at each cell -------------
sw = read_records(f'{DS}/audit_sweep_summary.csv')
std = sw[(sw.kind == 'dr') & (~sw.charged)][['temp0', 'n', 'policy', 'cov_t']]
std = std.rename(columns={'n': 'n_log', 'cov_t': 'coverage'})
m = cell.merge(std, on=['temp0', 'n_log', 'policy'], how='inner')
m['fails'] = m.coverage < 0.95
print(f'[{DS}] cells={len(m)}  failing={int(m.fails.sum())}')


def best_threshold(x, y):
    """Largest balanced accuracy over all thresholds of a single statistic."""
    best = (0.0, None, None)
    for thr in np.unique(x):
        for direction in (+1, -1):
            pred = (x <= thr) if direction > 0 else (x >= thr)
            tp = np.sum(pred & y); tn = np.sum(~pred & ~y)
            fn = np.sum(~pred & y); fp = np.sum(pred & ~y)
            sens = tp/(tp + fn) if (tp + fn) else 0.0
            spec = tn/(tn + fp) if (tn + fp) else 0.0
            ba = 0.5*(sens + spec)
            if ba > best[0]:
                best = (ba, float(thr), direction)
    return best


y = m.fails.values
for col in ('ess_trunc', 'ess_trunc_frac', 'ess_raw', 'ess_raw_frac', 'wmax', 'charge'):
    ba, thr, d = best_threshold(m[col].values, y)
    sep = 'SEPARATES' if ba == 1.0 else ''
    print(f'  {col:<16} best balanced accuracy = {ba:.3f}  at threshold {thr:.4g} '
          f'({"<=" if d > 0 else ">="})  {sep}')

print('\n  mean raw ESS fraction   failing cells: '
      f'{m.loc[y, "ess_raw_frac"].mean():.3f}   passing: {m.loc[~y, "ess_raw_frac"].mean():.3f}')
print('  mean trunc ESS fraction failing cells: '
      f'{m.loc[y, "ess_trunc_frac"].mean():.3f}   passing: {m.loc[~y, "ess_trunc_frac"].mean():.3f}')
