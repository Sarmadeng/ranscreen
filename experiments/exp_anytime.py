"""Experiment: anytime-valid screening under continuous monitoring.

A fixed-sample certificate loses its guarantee when the operator watches the evidence accumulate
and stops when it looks sufficient. The betting confidence sequence is valid at all times and under
any stopping rule, but its width scales with the a priori bound on the contribution, which is set
by the truncation level M. This script measures both effects and sweeps M, since the level that is
best for a fixed-n certificate is not the level that is best for an anytime-valid one.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/claude/paper/experiments')
import numpy as np, pandas as pd
from common import *

CHECKS = np.unique(np.round(np.logspace(np.log10(500), np.log10(20_000), 25)).astype(int))
M_GRID = (2.0, 3.0, 5.0, 10.0, 20.0)


def run(env, n_rep=max(N_REP//8,25), n_log=20_000, temps0=(0.2, 1.2, 2.0), m_grid=(2.0, 5.0, 20.0)) -> pd.DataFrame:
    rows = []
    for t0 in temps0:
        for s in range(n_rep):
            rng = np.random.default_rng(57_000 + s)
            log = sample_log(env, rng, n_log, t0)
            Q = E.crossfit_q(env, log, seed=s)
            for name, (h, t) in CANDIDATES.items():
                P = probs(env, log['idx'], h, t)
                gt = env.gain(P, log['P0'], log['idx'])
                for M in m_grid:
                    tm, _ = E.terms(env, log, P, Q, 'dr', charged=True, M=M)
                    lb_cs = E.betting_lower(tm, CHECKS, B=M + 1.0)
                    lb_t = np.array([E.t_lower(tm[:k]) for k in CHECKS])
                    rows.append(dict(temp0=t0, seed=s, policy=name, M=M, d_true=gt,
                                     cov_t_fixed=bool(lb_t[-1] <= gt),
                                     cov_cs_fixed=bool(lb_cs[-1] <= gt),
                                     cov_t_peek=bool((lb_t <= gt).all()),
                                     cov_cs_peek=bool((lb_cs <= gt).all()),
                                     acc_t=bool(lb_t[-1] > 0), acc_cs=bool(lb_cs[-1] > 0),
                                     stop_cs=int(CHECKS[np.argmax(lb_cs > 0)]) if (lb_cs > 0).any() else -1,
                                     stop_t=int(CHECKS[np.argmax(lb_t > 0)]) if (lb_t > 0).any() else -1))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else '4g'
    env = env_for(ds); od = out_dir(ds)
    with timed('anytime'):
        R = run(env); R.to_csv(f'{od}/anytime_raw.csv', index=False)
    S = R.groupby(['temp0', 'policy', 'M']).agg(
        d_true=('d_true', 'mean'),
        cov_t_fixed=('cov_t_fixed', 'mean'), cov_t_peek=('cov_t_peek', 'mean'),
        cov_cs_fixed=('cov_cs_fixed', 'mean'), cov_cs_peek=('cov_cs_peek', 'mean'),
        acc_t=('acc_t', 'mean'), acc_cs=('acc_cs', 'mean'),
        stop_cs=('stop_cs', lambda v: v[v > 0].mean())).reset_index()
    S.to_csv(f'{od}/anytime_summary.csv', index=False)
    pd.set_option('display.width', 240)
    print('\nworst-case coverage over all cells:')
    print(S.groupby('M')[['cov_t_fixed', 'cov_t_peek', 'cov_cs_fixed', 'cov_cs_peek']].min().round(3).to_string())
    good = S[S.d_true > 0]
    print('\nacceptance where the candidate is genuinely better, by truncation level:')
    print(good.pivot_table(index='M', values=['acc_t', 'acc_cs']).round(3).to_string())
    print('\nmean decisions needed by the confidence sequence:')
    print(good.pivot_table(index='M', values='stop_cs').round(0).to_string())
