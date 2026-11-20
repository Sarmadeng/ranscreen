"""Experiment: the charge for an arbitrary weight transform (Theorem 3).

The published estimators differ in how they tame the importance weight. Proposition 2 was derived
for hard truncation; Theorem 3 gives the charge for any transform phi, so one correction covers
the family. This script screens the same candidates with each estimator, with and without it.
"""
from __future__ import annotations
import sys
import pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from common import *

KINDS = ['ips', 'snips', 'dr', 'switch', 'shrink']


def run(env, n_rep=N_REP, n_log=20_000, temps0=(0.2, 0.8, 2.0)) -> pd.DataFrame:
    rows = []
    for t0 in temps0:
        for s in range(n_rep):
            rng = np.random.default_rng(45_000 + s)
            log = sample_log(env, rng, n_log, t0)
            Q = E.crossfit_q(env, log, seed=s)
            for name, (h, t) in CANDIDATES.items():
                P = probs(env, log['idx'], h, t)
                gt = env.gain(P, log['P0'], log['idx'])
                for k in KINDS:
                    for ch in (False, True):
                        tm, d = E.terms(env, log, P, Q, k, charged=ch)
                        rows.append(dict(temp0=t0, seed=s, policy=name, kind=k, charged=ch,
                                         d_true=gt, d_hat=float(tm.mean()), lb=E.t_lower(tm),
                                         charge=d['charge'], wmax=d['wmax']))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else '4g'
    env = env_for(ds); od = out_dir(ds)
    with timed('estimator family'):
        R = run(env); R.to_csv(f'{od}/estimator_family_raw.csv', index=False)
    S = coverage_table(R, ['temp0', 'policy', 'kind', 'charged'])
    S = S.merge(R.groupby(['temp0', 'policy', 'kind', 'charged']).charge.mean().reset_index(),
                on=['temp0', 'policy', 'kind', 'charged'])
    S.to_csv(f'{od}/estimator_family_summary.csv', index=False)
    pd.set_option('display.width', 230)
    print('\nworst-case coverage over all cells, per estimator:')
    print(S.pivot_table(index='kind', columns='charged', values='coverage', aggfunc='min').round(3).to_string())
    good = S[S.d_true > 0]
    print('\nmean acceptance where the candidate is genuinely better:')
    print(good.pivot_table(index='kind', columns='charged', values='acceptance').round(3).to_string())
    print('\nmean charge per estimator and incumbent exploration (smaller certifies more):')
    print(S[S.charged].pivot_table(index='temp0', columns='kind', values='charge').round(4).to_string())
