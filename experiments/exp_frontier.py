"""Experiment: the certifiability frontier.

The charge cbar = E[c(X)] depends only on the two policies, so it is computable before any log is
collected. Delta - 2 r_max cbar <= E[charged contribution] <= Delta, hence a candidate whose gain
exceeds 2 r_max cbar is certifiable at a large enough sample size, and one whose gain is below it
is not, no matter how much data is collected. This script computes the frontier and checks it
against what the gate actually does.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/claude/paper/experiments')
import numpy as np, pandas as pd
from common import *

HYSTS = (-3., -1., 0., 1., 2., 3., 4., 5., 6.)
TEMPS_C = (0.3, 0.6, 1.0, 1.6)


def frontier(env, n_ctx=60_000, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(env), min(n_ctx, len(env)), replace=False)
    rows = []
    for t0 in TEMPS:
        P0 = probs(env, idx, 3.0, t0)
        for h in HYSTS:
            for t in TEMPS_C:
                P = probs(env, idx, h, t)
                cbar = float(E.charge(P, P0, 'dr').mean())
                d = env.gain(P, P0, idx)
                rows.append(dict(temp0=t0, hyst=h, temp=t, cbar=cbar, delta=d,
                                 certifiable=bool(d > 2*E.R_MAX*cbar), positive=bool(d > 0)))
    return pd.DataFrame(rows)


def validate(env, F: pd.DataFrame, n_rep=60, n_log=20_000, n_settings=40) -> pd.DataFrame:
    rows = []
    sub = F.sample(n=min(n_settings, len(F)), random_state=0)
    for rec in sub.itertuples():
        acc = cov = 0
        for s in range(n_rep):
            rng = np.random.default_rng(12_000 + s)
            log = sample_log(env, rng, n_log, rec.temp0)
            Q = E.crossfit_q(env, log, seed=s)
            P = probs(env, log['idx'], rec.hyst, rec.temp)
            tm, _ = E.terms(env, log, P, Q, 'dr', charged=True)
            lb = E.t_lower(tm); gt = env.gain(P, log['P0'], log['idx'])
            acc += lb > 0; cov += lb <= gt
        rows.append(dict(temp0=rec.temp0, hyst=rec.hyst, temp=rec.temp, cbar=rec.cbar,
                         delta=rec.delta, predicted=rec.certifiable,
                         acceptance=acc/n_rep, coverage=cov/n_rep))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else '4g'
    env = env_for(ds); od = out_dir(ds)
    with timed('frontier'):
        F = frontier(env); F.to_csv(f'{od}/frontier.csv', index=False)
    with timed('validation'):
        V = validate(env, F); V.to_csv(f'{od}/frontier_validation.csv', index=False)
    agree = float(((V.predicted) == (V.acceptance > 0)).mean())
    pd.set_option('display.width', 220)
    print(f'\nprediction agrees with the realised gate in {agree:.0%} of settings')
    print('\nfraction of candidate settings that are certifiable, by incumbent exploration:')
    print(F.groupby('temp0')[['positive', 'certifiable']].mean().round(3).to_string())
    print('\nlargest certifiable distance |h - 3| per incumbent temperature:')
    print(F[F.certifiable].assign(d=lambda x: (x.hyst - 3).abs()).groupby('temp0').d.max().to_string())
    pd.DataFrame([dict(dataset=ds, agreement=agree)]).to_csv(f'{od}/frontier_agreement.csv', index=False)
