"""Experiment: transfer across regimes, and the diagnostic question.

Two questions a reviewer will ask and this script answers.

(1) An operator tunes on one part of the network and deploys on another. We therefore audit the
    certificate against the truth in the DEPLOYMENT regime, not the logging regime, over every
    pair of mobility classes.

(2) Which diagnostic actually predicts failure? We compare the effective sample size, the maximum
    importance weight and the charge, scoring each by how well a single threshold separates the
    cells whose coverage is below nominal from those clearly above it.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/claude/paper/experiments')
import numpy as np, pandas as pd
from common import *


def transfer(env, n_rep=N_REP//3, n_log=20_000, t0=1.2, min_rows=3_000) -> pd.DataFrame:
    mob = env.mob
    classes = [c for c in pd.unique(mob) if (mob == c).sum() > min_rows]
    pools = {c: np.where(mob == c)[0] for c in classes}
    rows = []
    for src in classes:
        for s in range(n_rep):
            rng = np.random.default_rng(73_000 + s)
            log = sample_log(env, rng, min(n_log, len(pools[src])), t0, pool=pools[src])
            Q = E.crossfit_q(env, log, seed=s)
            for name, (h, t) in CANDIDATES.items():
                P = probs(env, log['idx'], h, t)
                for ch in (False, True):
                    tm, _ = E.terms(env, log, P, Q, 'dr', charged=ch)
                    lb = E.t_lower(tm)
                    for dst in classes:
                        j = pools[dst]
                        gd = env.gain(probs(env, j, h, t), probs(env, j, 3.0, t0), j)
                        rows.append(dict(src=src, dst=dst, policy=name, charged=ch, seed=s,
                                         lb=lb, d_true=gd, same=src == dst))
    return pd.DataFrame(rows)


def diagnostics(cells: pd.DataFrame) -> pd.DataFrame:
    """Which single quantity separates failing from clearly-passing cells?"""
    U = cells[~cells.charged].copy()
    ch = cells[cells.charged][['temp0', 'n', 'policy', 'charge']].rename(columns={'charge': 'cbar'})
    U = U.merge(ch, on=['temp0', 'n', 'policy'])
    U['ess_frac'] = U.ess/U.n
    fail = U.cov_hi < 0.90
    ok = U.cov_lo > 0.90
    out = []
    for col, higher_is_worse in [('ess_frac', False), ('wmax_mean', True), ('cbar', True)]:
        best = (0.0, np.nan)
        for th in np.unique(U[col]):
            pred = U[col] > th if higher_is_worse else U[col] < th
            tp = (pred & fail).sum(); tn = ((~pred) & ok).sum()
            ba = .5*(tp/max(fail.sum(), 1) + tn/max(ok.sum(), 1))
            if ba > best[0]: best = (ba, th)
        out.append(dict(diagnostic=col, best_threshold=best[1], balanced_accuracy=best[0],
                        separates=bool(U[fail][col].min() > U[ok][col].max()) if higher_is_worse
                        else bool(U[fail][col].max() < U[ok][col].min()),
                        fail_range=f'[{U[fail][col].min():.4g},{U[fail][col].max():.4g}]',
                        pass_range=f'[{U[ok][col].min():.4g},{U[ok][col].max():.4g}]'))
    return pd.DataFrame(out)


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else '4g'
    env = env_for(ds); od = out_dir(ds)
    with timed('transfer'):
        T = transfer(env); T.to_csv(f'{od}/transfer_raw.csv', index=False)
    S = coverage_table(T, ['src', 'dst', 'policy', 'charged'])
    S.to_csv(f'{od}/transfer_summary.csv', index=False)
    pd.set_option('display.width', 240)
    for ch in (False, True):
        print(f'\ncoverage against the deployment regime, charged={ch}:')
        print(S[S.charged == ch].pivot_table(index='src', columns='dst', values='coverage').round(3).to_string())
    S['same'] = S.src == S.dst
    print('\npooled, same regime vs across regimes:')
    print(S.groupby(['charged', 'same']).coverage.mean().round(3).to_string())

    cells_path = f'{od}/audit_cells.csv'
    import os
    if os.path.exists(cells_path):
        D = diagnostics(pd.read_csv(cells_path))
        D.to_csv(f'{od}/diagnostics.csv', index=False)
        print('\nwhich diagnostic predicts failure:')
        print(D.round(3).to_string(index=False))
