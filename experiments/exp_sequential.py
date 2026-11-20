"""Experiment: sequential control with windowed per-decision importance sampling.

Full per-decision importance sampling weights r_t by the product of all weights up to t, which
collapses exponentially with the horizon. The k-step windowed weight keeps only the last k single
step ratios: variance stops growing with the horizon, at the cost of a bias whose sign and size
are measured here rather than assumed.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/claude/paper/experiments')
import numpy as np, pandas as pd
from common import *
from ranscreen.data import sinr_to_se

HMAX, PP_WIN = 40, 3


def episodes(env, hmax=HMAX) -> np.ndarray:
    df = pd.DataFrame({'trace': env.trace}).reset_index().rename(columns={'index': 'row'})
    blocks = []
    for _, g in df.groupby('trace'):
        rows = g.row.values
        if len(rows) >= hmax:
            blocks.append(np.lib.stride_tricks.sliding_window_view(rows, hmax)[::hmax])
    return np.concatenate(blocks, axis=0)


def _margin(env, rows, cur):
    s, n = env.rsrp_s[rows], env.rsrp_n[rows]
    return np.where(cur == 0, n - s, s - n)


def rollout(env, rng, ep, hyst, temp, record=False, pingpong=0.03):
    E_, H = ep.shape
    cur = np.zeros(E_, int); last = np.full(E_, -99)
    R = np.zeros((E_, H)); A = np.zeros((E_, H), int); PR = np.zeros((E_, H))
    for t in range(H):
        rows = ep[:, t]
        p = 1/(1 + np.exp(-(_margin(env, rows, cur) - hyst)/temp))
        a = (rng.random(E_) < p).astype(int)
        PR[:, t] = np.where(a == 1, p, 1 - p); A[:, t] = a
        new = np.where(a == 1, 1 - cur, cur)
        sinr = np.where(new == 0, env.sinr[rows, 0], env.sinr[rows, 1]) + rng.choice(env.resid, size=E_)
        cost = a*env.ho_penalty[rows] + a*(t - last <= PP_WIN)*pingpong
        R[:, t] = np.clip(sinr_to_se(sinr) - cost, 0, 1)
        last = np.where(a == 1, t, last); cur = new
    return dict(ep=ep, A=A, PR=PR, R=R) if record else R.sum(1)


def replay_propensities(env, log, hyst, temp):
    E_, H = log['A'].shape
    cur = np.zeros(E_, int); P = np.zeros((E_, H))
    for t in range(H):
        p = 1/(1 + np.exp(-(_margin(env, log['ep'][:, t], cur) - hyst)/temp))
        a = log['A'][:, t]
        P[:, t] = np.where(a == 1, p, 1 - p)
        cur = np.where(a == 1, 1 - cur, cur)
    return P


def windowed(ratio, k, M):
    E_, H = ratio.shape
    cum = np.cumsum(np.log(np.clip(ratio, 1e-12, None)), axis=1)
    pad = np.concatenate([np.zeros((E_, 1)), cum], axis=1)
    return np.minimum(np.exp(cum - pad[:, np.maximum(np.arange(H) - k + 1, 0)]), M)


def run(env, horizons=(10, 20, 40), temps0=(0.4, 1.2), ks=(1, 2, 3, 5, 10, None),
        n_rep=max(N_REP//5,60), n_ep=3_000, M=E.M_DEFAULT):
    EP = episodes(env); rows = []
    for t0 in temps0:
        for H in horizons:
            base = EP[:, :H]
            for s in range(n_rep):
                rng = np.random.default_rng(42_000 + s)
                ep = base[rng.choice(len(base), min(n_ep, len(base)), replace=False)]
                log = rollout(env, rng, ep, 3.0, t0, record=True)
                for name, (h, t) in CANDIDATES.items():
                    if name == 'extreme': continue
                    ratio = replay_propensities(env, log, h, t)/log['PR']
                    gt = rollout(env, np.random.default_rng(7_000+s), ep, h, t).mean()
                    gb = rollout(env, np.random.default_rng(7_000+s), ep, 3.0, t0).mean()
                    for k in ks:
                        kk = H if k is None else k
                        W = windowed(ratio, kk, M); Wraw = windowed(ratio, kk, np.inf)
                        d = ((W - 1)*log['R']).sum(1)
                        dp = d - (np.maximum(Wraw - M, 0.)*log['R']).sum(1)
                        rows.append(dict(temp0=t0, H=H, policy=name, k=kk, full=(k is None),
                                         seed=s, d_true=gt - gb, d_hat=float(d.mean()),
                                         lb=E.t_lower(dp),
                                         ess=float(W[:, -1].sum()**2/np.sum(W[:, -1]**2))))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else '4g'
    env = env_for(ds); od = out_dir(ds)
    with timed('sequential'):
        R = run(env); R.to_csv(f'{od}/sequential_raw.csv', index=False)
    S = coverage_table(R, ['temp0', 'H', 'policy', 'k'])
    S = S.merge(R.groupby(['temp0', 'H', 'policy', 'k']).agg(
        bias=('d_hat', 'mean'), ess=('ess', 'mean'),
        truth=('d_true', 'mean')).reset_index(), on=['temp0', 'H', 'policy', 'k'])
    S['bias'] = S.bias - S.truth
    S.to_csv(f'{od}/sequential_summary.csv', index=False)
    pd.set_option('display.width', 240)
    print('\ncoverage by window k (nominal 0.95):')
    print(S.pivot_table(index=['temp0', 'H', 'policy'], columns='k', values='coverage').round(3).to_string())
    print('\nacceptance by window k:')
    print(S.pivot_table(index=['temp0', 'H', 'policy'], columns='k', values='acceptance').round(3).to_string())
    unsafe = S[S.bias > 0].groupby('k').size()
    print('\ncells with upward (unsafe) bias, by k:'); print(unsafe.to_string())
