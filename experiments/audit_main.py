"""Full audit: every headline claim re-run with Monte Carlo error bars and an explicit check.

Each claim in the paper is reduced here to a predicate that either holds or fails, so the audit
is a pass/fail table rather than a narrative. Coverage is reported with a Wilson interval, so a
claim of "0.92" is never quoted without the uncertainty attached to it.
"""
from __future__ import annotations
import sys, json, time
sys.path.insert(0, '/home/claude/paper')
import numpy as np, pandas as pd
from scipy import stats
from ranscreen import build_environment, Config, probs, sample_log, CANDIDATES, estimators as E
from ranscreen.policies import INCUMBENT_HYST, agent_propose

DATASET = sys.argv[1] if len(sys.argv) > 1 else '4g'
OUT = f'/home/claude/paper/results/{DATASET}'
import os; os.makedirs(OUT, exist_ok=True)
N_REP = 500                      # raised from 200: Wilson half-width ~0.02 at coverage 0.95
TEMPS = (0.2, 0.4, 0.8, 1.2, 2.0)
SIZES = (1_000, 5_000, 20_000)


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return float(c - h), float(c + h)


def sweep(env, n_rep=N_REP, kinds=('dr',), charged=(False, True), bounds=('t', 'eb')):
    rows = []
    t0 = time.time()
    for temp0 in TEMPS:
        for n_log in SIZES:
            for s in range(n_rep):
                rng = np.random.default_rng(100_000 + s)
                log = sample_log(env, rng, n_log, temp0)
                Q = E.crossfit_q(env, log, seed=s)
                for name, (h, t) in CANDIDATES.items():
                    P = probs(env, log['idx'], h, t)
                    gt = env.gain(P, log['P0'], log['idx'])
                    for kind in kinds:
                        for ch in charged:
                            tm, d = E.terms(env, log, P, Q, kind, charged=ch)
                            rec = dict(temp0=temp0, n=n_log, seed=s, policy=name, kind=kind,
                                       charged=ch, d_true=gt, d_hat=float(tm.mean()), **d)
                            for b in bounds:
                                rec[f'lb_{b}'] = E.BOUNDS[b](tm)
                            rows.append(rec)
            print(f'  temp0={temp0} n={n_log} done  ({time.time()-t0:.0f}s)', flush=True)
    return pd.DataFrame(rows)


def summarise(R, bounds=('t', 'eb')):
    out = []
    for keys, g in R.groupby(['temp0', 'n', 'policy', 'kind', 'charged']):
        rec = dict(zip(['temp0', 'n', 'policy', 'kind', 'charged'], keys))
        rec.update(d_true=g.d_true.mean(), bias=g.d_hat.mean() - g.d_true.mean(),
                   wmax=g.wmax.mean(), ess=g.ess.mean(), charge=g.charge.mean(), reps=len(g))
        for b in bounds:
            k = int((g[f'lb_{b}'] <= g.d_true).sum())
            lo, hi = wilson(k, len(g))
            rec[f'cov_{b}'] = k/len(g); rec[f'cov_{b}_lo'] = lo; rec[f'cov_{b}_hi'] = hi
            rec[f'acc_{b}'] = float((g[f'lb_{b}'] > 0).mean())
        out.append(rec)
    return pd.DataFrame(out)


if __name__ == '__main__':
    env = build_environment(Config(dataset=DATASET))
    json.dump(env.prov, open(f'{OUT}/provenance.json', 'w'), indent=1, default=float)

    print(f'\n=== {DATASET}: main sweep, uncharged vs charged, {N_REP} replicates per cell ===', flush=True)
    R = sweep(env)
    R.to_csv(f'{OUT}/audit_sweep_raw.csv', index=False)
    A = summarise(R)
    A.to_csv(f'{OUT}/audit_sweep_summary.csv', index=False)

    pd.set_option('display.width', 250)
    for ch in (False, True):
        sub = A[(A.charged == ch)]
        print(f"\n--- coverage, Student-t, charged={ch} ---")
        print(sub.pivot_table(index=['temp0', 'n'], columns='policy', values='cov_t').round(3).to_string())
    print('\n--- worst cells, with Wilson 95% intervals ---')
    w = A.sort_values('cov_t').head(8)
    print(w[['temp0', 'n', 'policy', 'charged', 'cov_t', 'cov_t_lo', 'cov_t_hi',
             'bias', 'wmax', 'ess']].round(4).to_string(index=False))

    claims = []
    unch, ch = A[~A.charged], A[A.charged]
    claims.append(dict(claim='standard certificate falls below nominal somewhere',
                       value=unch.cov_t.min(), ci=f"[{unch.sort_values('cov_t').cov_t_lo.iloc[0]:.3f},"
                                                  f"{unch.sort_values('cov_t').cov_t_hi.iloc[0]:.3f}]",
                       passes=bool(unch.cov_t.min() < 0.90)))
    claims.append(dict(claim='charged certificate worst-case coverage',
                       value=ch.cov_t.min(), ci=f"[{ch.sort_values('cov_t').cov_t_lo.iloc[0]:.3f},"
                                                f"{ch.sort_values('cov_t').cov_t_hi.iloc[0]:.3f}]",
                       passes=bool(ch.cov_t.min() > 0.85)))
    good_u = unch[(unch.d_true > 0) & (unch.temp0 >= 1.2) & (unch.n == 20_000)].acc_t.mean()
    good_c = ch[(ch.d_true > 0) & (ch.temp0 >= 1.2) & (ch.n == 20_000)].acc_t.mean()
    claims.append(dict(claim='power preserved by the charge (temp>=1.2, n=20k)',
                       value=f'{good_u:.3f} -> {good_c:.3f}', ci='', passes=bool(good_c >= good_u - .02)))
    mono = unch[(unch.temp0 == 0.2) & (unch.policy == 'aggressive')].sort_values('n').cov_t.values
    claims.append(dict(claim='coverage degrades with more data (temp0=0.2, aggressive)',
                       value=' -> '.join(f'{v:.3f}' for v in mono), ci='',
                       passes=bool(mono[0] > mono[-1])))
    worst = unch.sort_values('cov_t').iloc[0]
    claims.append(dict(claim='ESS is high at the worst cell', value=f"ESS={worst.ess:.0f}/{worst.n}",
                       ci='', passes=bool(worst.ess > 0.5*worst.n)))
    # Proposition 1 predicts zero bias only where no weight is truncated; the charged statistic is
    # pessimistic BY CONSTRUCTION, so its 'bias' must equal minus the charge and is checked as such.
    nz = R[(R.clipped == 0) & (~R.charged)].groupby(['temp0', 'n', 'policy']).apply(
        lambda g: abs(g.d_hat.mean() - g.d_true.mean()))
    claims.append(dict(claim='Prop. 1: zero bias where no weight is truncated',
                       value=float(nz.max()), ci='', passes=bool(nz.max() < 1e-3)))
    chk = R[R.charged].groupby(['temp0', 'n', 'policy']).apply(
        lambda g: abs((g.d_hat.mean() - g.d_true.mean()) + g.charge.mean()))
    claims.append(dict(claim='charged statistic is pessimistic by exactly the charge',
                       value=float(chk.max()), ci='', passes=bool(chk.max() < 5e-3)))
    one_below = int((ch.cov_t_hi < 0.95).sum())
    claims.append(dict(claim='charged cells statistically below nominal',
                       value=f'{one_below} of {len(ch)}', ci='', passes=bool(one_below <= 3)))
    CL = pd.DataFrame(claims); CL.to_csv(f'{OUT}/audit_claims.csv', index=False)
    print('\n=== CLAIM AUDIT ===')
    print(CL.to_string(index=False))
