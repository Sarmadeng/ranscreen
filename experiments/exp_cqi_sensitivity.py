"""Reviewer 1: "Wrong CQI table for the 5G corpus."

The reviewer is right. The 5G corpus is non-standalone and 82.9% of its retained rows report
NR, for which TS 38.214 table 2 applies rather than the LTE table of TS 36.213 used
throughout. This script rebuilds the 5G environment with the NR table and repeats the main
coverage audit, so that the paper can report whether any conclusion depends on the choice.

usage: exp_cqi_sensitivity.py [n_rep]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, '/home/claude/paper')
import numpy as np, pandas as pd
from ranscreen import data as DATA

DATA.use_cqi_table('nr')                      # must precede build_environment

from ranscreen import build_environment, Config, probs, sample_log, CANDIDATES, estimators as E

N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TEMPS = (0.2, 0.4, 0.8, 1.2, 2.0)
SIZES = (1_000, 5_000, 20_000)
OUT = '/home/claude/paper/results/revision'
os.makedirs(OUT, exist_ok=True)


def wilson(k, n, z=1.96):
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return float(c - h), float(c + h)


env = build_environment(Config(dataset='5g'))
rows = []
for temp0 in TEMPS:
    for n_log in SIZES:
        for s in range(N_REP):
            rng = np.random.default_rng(100_000 + s)
            log = sample_log(env, rng, n_log, temp0)
            Q = E.crossfit_q(env, log, seed=s)
            for name, (h, t) in CANDIDATES.items():
                P = probs(env, log['idx'], h, t)
                gt = env.gain(P, log['P0'], log['idx'])
                for charged in (False, True):
                    tm, d = E.terms(env, log, P, Q, 'dr', charged=charged)
                    lb = E.t_lower(tm)
                    rows.append(dict(temp0=temp0, n=n_log, policy=name, charged=charged,
                                     seed=s, d_true=gt, lb=lb, cov=lb <= gt, acc=lb > 0,
                                     harmful=gt <= 0, charge=d['charge'], wmax=d['wmax']))

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/cqi_nr_raw_5g.csv', index=False)
g = df.groupby(['temp0', 'n', 'policy', 'charged'], as_index=False).agg(
    reps=('cov', 'size'), cov=('cov', 'mean'), acc=('acc', 'mean'),
    d_true=('d_true', 'first'), charge=('charge', 'mean'), wmax=('wmax', 'mean'))
g.to_csv(f'{OUT}/cqi_nr_summary_5g.csv', index=False)

print('=== 5G audit repeated with the NR CQI table (TS 38.214 table 2) ===')
for charged in (False, True):
    h = g[g.charged == charged]
    k = int(df[df.charged == charged].cov.sum()); n = int((df.charged == charged).sum())
    lo, hi = wilson(k, n)
    below = int((h.cov < 0.95).sum())
    print(f'  {"charged " if charged else "standard"}  worst-cell coverage {h.cov.min():.3f}   '
          f'cells below nominal {below} of {len(h)}   pooled {k/n:.3f} [{lo:.3f},{hi:.3f}]')
    good = h[h.d_true > 0]
    print(f'            acceptance where the candidate is better: {good.acc.mean():.3f}')
print('\nFor comparison, the LTE-table figures reported in the paper are: standard worst-cell '
      '0.000, 21 of 60 cells below nominal; charged worst-cell 0.936 with power 1.000.')
