"""Reviewer 1: Table 7 reports the 5G figure only in the caption; give the full 5G columns.

This script repeats the certificate comparison on BOTH corpora with the same sweep, so that
every row of the table carries a 4G and a 5G figure. Seven combinations are compared: the
uncorrected bound, the same bound with the naive SAMPLED charge under four concentration
inequalities, and the closed-form charge under two.

usage: exp_bound_compare.py <4g|5g> [n_rep]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, '/home/claude/paper')
import numpy as np, pandas as pd
from ranscreen import build_environment, Config, probs, sample_log, CANDIDATES, estimators as E

DS    = sys.argv[1] if len(sys.argv) > 1 else '4g'
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 100
TEMPS = (0.2, 0.4, 0.8, 1.2, 2.0)
SIZES = (1_000, 5_000, 20_000)
OUT   = '/home/claude/paper/results/revision'
os.makedirs(OUT, exist_ok=True)

BOUNDS = {'t': E.t_lower, 'eb': E.eb_lower, 'mom': E.mom_lower, 'catoni': E.catoni_lower}
COMBOS = [('none', 't'),
          ('sampled', 't'), ('sampled', 'catoni'), ('sampled', 'mom'), ('sampled', 'eb'),
          ('closed', 't'), ('closed', 'eb')]

env = build_environment(Config(dataset=DS))
rows = []
for temp0 in TEMPS:
    for n_log in SIZES:
        for s in range(N_REP):
            rng = np.random.default_rng(300_000 + s)
            log = sample_log(env, rng, n_log, temp0)
            Q = E.crossfit_q(env, log, seed=s)
            ar = np.arange(log['n'])
            for name, (h, t) in CANDIDATES.items():
                P = probs(env, log['idx'], h, t)
                gt = env.gain(P, log['P0'], log['idx'])
                base, _ = E.terms(env, log, P, Q, 'dr', charged=False)
                wr = P[ar, log['a']]/np.maximum(log['prop'], 1e-300)
                samp = E.sampled_charge(np.minimum(wr, E.WCAP))
                clos = E.charge(P, log['P0'], 'dr')
                variants = {'none': base,
                            'sampled': base - E.R_MAX*samp,
                            'closed': base - E.R_MAX*clos}
                for corr, bnd in COMBOS:
                    lb = BOUNDS[bnd](variants[corr])
                    rows.append(dict(dataset=DS, temp0=temp0, n=n_log, policy=name, seed=s,
                                     correction=corr, bound=bnd, d_true=gt, lb=lb,
                                     cov=lb <= gt, acc=lb > 0, better=gt > 0))

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/bound_compare_raw_{DS}.csv', index=False)
cell = df.groupby(['correction', 'bound', 'temp0', 'n', 'policy'], as_index=False).agg(
    cov=('cov', 'mean'), acc=('acc', 'mean'), better=('better', 'first'))
summ = cell.groupby(['correction', 'bound'], as_index=False).agg(worst_cov=('cov', 'min'))
pw = cell[cell.better].groupby(['correction', 'bound'], as_index=False).agg(power=('acc', 'mean'))
out = summ.merge(pw, on=['correction', 'bound'])
out.to_csv(f'{OUT}/bound_compare_summary_{DS}.csv', index=False)
print(f'=== certificate comparison, {DS.upper()} ({N_REP} logs per cell) ===')
print(f'{"correction":>10} {"bound":>8} | {"worst-case coverage":>20} {"power":>8}')
for _, r in out.iterrows():
    print(f'{r.correction:>10} {r.bound:>8} | {r.worst_cov:20.3f} {r.power:8.3f}')
