"""Reviewer 1: "Propositions 7 and 8 are not evaluated."

This script evaluates Proposition 7 (post-selection validity). An autonomous controller
searches the policy class with its own SNIPS estimate and submits the argmax. Two certificates
are then computed for that winner:

  naive  : the single-policy charged bound at level delta
  union  : the same bound at level delta / |G|, which is what Proposition 7 licenses

Both are audited against the true gain of the selected policy, so the experiment measures
exactly the quantity the proposition is about: whether the bound reported for a policy that
was chosen because its estimate looked good is still valid.

usage: exp_postselection.py <4g|5g> [n_rep]
"""
from __future__ import annotations
import sys, os
import pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from scipy import stats
from ranscreen import build_environment, Config, probs, sample_log, estimators as E
from ranscreen.policies import AGENT_GRID, snips, INCUMBENT_HYST
from ranscreen.paths import RESULTS

DS    = sys.argv[1] if len(sys.argv) > 1 else '4g'
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 200
TEMPS = (0.4, 0.8, 1.2, 2.0)
SIZES = (5_000, 20_000)
DELTA = 0.05
G     = len(AGENT_GRID)
OUT   = str(RESULTS) + '/revision'
os.makedirs(OUT, exist_ok=True)


def charged_lb(env, log, P, Q, delta):
    """Lower confidence bound on the mean of the charged contributions, Student-t."""
    t, _ = E.terms(env, log, P, Q, 'dr', charged=True)
    n = len(t)
    return float(t.mean() - stats.t.ppf(1 - delta, n - 1)*t.std(ddof=1)/np.sqrt(n))


env = build_environment(Config(dataset=DS))
rows = []
for temp0 in TEMPS:
    for n_log in SIZES:
        for s in range(N_REP):
            rng = np.random.default_rng(700_000 + s)
            log = sample_log(env, rng, n_log, temp0)
            Q = E.crossfit_q(env, log, seed=s)

            # the agent searches the class with its own estimate and submits the argmax
            v0 = snips(env, log, log['P0'])
            best, best_v = None, -np.inf
            for (h, t) in AGENT_GRID:
                v = snips(env, log, probs(env, log['idx'], h, t))
                if v > best_v:
                    best_v, best = v, (h, t)
            Pw = probs(env, log['idx'], *best)
            d_true = env.gain(Pw, log['P0'], log['idx'])

            lb_naive = charged_lb(env, log, Pw, Q, DELTA)
            lb_union = charged_lb(env, log, Pw, Q, DELTA/G)
            rows.append(dict(dataset=DS, temp0=temp0, n=n_log, seed=s,
                             hyst=best[0], temp=best[1], d_true=d_true,
                             lb_naive=lb_naive, lb_union=lb_union,
                             cov_naive=lb_naive <= d_true, cov_union=lb_union <= d_true,
                             acc_naive=lb_naive > 0, acc_union=lb_union > 0,
                             harmful=d_true <= 0))

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/postselection_raw_{DS}.csv', index=False)


def wilson(k, n, z=1.96):
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return float(c - h), float(c + h)


g = df.groupby(['temp0', 'n'], as_index=False).agg(
    reps=('cov_naive', 'size'),
    cov_naive=('cov_naive', 'mean'), cov_union=('cov_union', 'mean'),
    acc_naive=('acc_naive', 'mean'), acc_union=('acc_union', 'mean'),
    opfail_naive=('harmful', lambda h: np.mean(h & df.loc[h.index, 'acc_naive'])),
    opfail_union=('harmful', lambda h: np.mean(h & df.loc[h.index, 'acc_union'])),
    d_true=('d_true', 'mean'))
g.to_csv(f'{OUT}/postselection_summary_{DS}.csv', index=False)

print(f'=== Proposition 7, post-selection validity, {DS.upper()} '
      f'(|G| = {G}, delta = {DELTA}) ===')
print(f'{"T0":>5} {"n":>7} | {"cov naive":>10} {"cov union":>10} | '
      f'{"acc naive":>10} {"acc union":>10}')
for _, r in g.iterrows():
    print(f'{r.temp0:5.1f} {int(r.n):7d} | {r.cov_naive:10.3f} {r.cov_union:10.3f} | '
          f'{r.acc_naive:10.3f} {r.acc_union:10.3f}')
k_n = int(df.cov_naive.sum()); k_u = int(df.cov_union.sum()); N = len(df)
print(f'\npooled coverage  naive {k_n/N:.3f} {wilson(k_n, N)}   '
      f'union {k_u/N:.3f} {wilson(k_u, N)}   (nominal 0.95)')
print(f'pooled acceptance naive {df.acc_naive.mean():.3f}   union {df.acc_union.mean():.3f}')
print(f'worst cell naive {g.cov_naive.min():.3f}   worst cell union {g.cov_union.min():.3f}')
