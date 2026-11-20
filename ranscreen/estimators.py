"""ranscreen.estimators — off-policy estimators, charges, confidence bounds and the gate.

Naming follows the paper:
  Proposition 1  bias identity for a truncated doubly-robust contribution
  Proposition 2  closed-form conditional charge for hard truncation
  Theorem 3      the same charge for an arbitrary weight transform
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

M_DEFAULT, R_MAX, DELTA = 20.0, 1.0, 0.05
TAU_SWITCH, LAMBDA_SHRINK, WCAP = 5.0, 10.0, 1e12


# --------------------------------------------------------------------- outcome model
def _feat(env, idx, a):
    m = env.rsrp_n[idx] - env.rsrp_s[idx]
    a = np.asarray(a, dtype=float)
    return np.column_stack([np.ones(len(idx)), env.rsrp_s[idx]/100, env.rsrp_n[idx]/100,
                            m/10, (m/10)**2, env.rsrq_s[idx]/10, env.rsrq_n[idx]/10,
                            env.speed[idx]/30, a, a*m/10, a*env.speed[idx]/30])


def crossfit_q(env, log, folds: int = 2, seed: int = 0, alpha: float = 1.0) -> np.ndarray:
    """Cross-fitted qhat(x, a) for both actions, trained on logged (x, a, r) only."""
    n = log['n']
    parts = np.array_split(np.random.default_rng(seed).permutation(n), folds)
    Q = np.zeros((n, 2))
    for i, te in enumerate(parts):
        tr = np.concatenate([p for j, p in enumerate(parts) if j != i])
        mdl = Ridge(alpha=alpha).fit(_feat(env, log['idx'][tr], log['a'][tr]), log['r'][tr])
        for a in (0, 1):
            Q[te, a] = mdl.predict(_feat(env, log['idx'][te], np.full(len(te), a)))
    return np.clip(Q, 0, 1)


# --------------------------------------------------------------------- weight transforms
TRANSFORMS = {
    'ips':    lambda w: np.minimum(w, M_DEFAULT),
    'dr':     lambda w: np.minimum(w, M_DEFAULT),
    'snips':  lambda w: np.minimum(w, M_DEFAULT),
    'switch': lambda w: np.where(w <= TAU_SWITCH, w, 0.0),
    'shrink': lambda w: (LAMBDA_SHRINK/(np.minimum(w, WCAP)**2 + LAMBDA_SHRINK))*np.minimum(w, WCAP),
}

# pi0 * phi(pi/pi0) in closed form: no division by a vanishing propensity ever occurs.
PHI_TIMES_P0 = {
    'ips':    lambda P, P0: np.minimum(P, M_DEFAULT*P0),
    'dr':     lambda P, P0: np.minimum(P, M_DEFAULT*P0),
    'snips':  lambda P, P0: np.minimum(P, M_DEFAULT*P0),
    'switch': lambda P, P0: np.where(P <= TAU_SWITCH*P0, P, 0.0),
    'shrink': lambda P, P0: LAMBDA_SHRINK*P*P0**2/(P**2 + LAMBDA_SHRINK*P0**2 + 1e-300),
}


def charge(P, P0, kind: str = 'dr') -> np.ndarray:
    """c_phi(x) = sum_a | pi(a|x) - pi0(a|x) phi(pi/pi0) |  (Theorem 3).

    For hard truncation this reduces to sum_a (pi - M pi0)_+ (Proposition 2). The value lies in
    [0, 1] for any policy pair, which is what makes bounded-range finite-sample bounds applicable.
    """
    return np.abs(P - PHI_TIMES_P0[kind](P, P0)).sum(1)


def sampled_charge(wr, M: float = M_DEFAULT) -> np.ndarray:
    """The same correction evaluated at the logged action only — heavy-tailed, kept for comparison."""
    return np.maximum(wr - M, 0.0)


# --------------------------------------------------------------------- contributions
def terms(env, log, P, Q, kind: str = 'dr', charged: bool = True, M: float = M_DEFAULT):
    """Per-sample contributions to Delta = V(pi) - V(pi0), optionally charged (Theorem 3)."""
    ar = np.arange(log['n']); a = log['a']
    wr = P[ar, a]/np.maximum(log['prop'], 1e-300)
    w = TRANSFORMS[kind](np.minimum(wr, WCAP))
    dm = ((P - log['P0'])*Q).sum(1)
    if kind == 'ips':
        t = (w - 1)*log['r']
    elif kind == 'snips':
        t = (w/max(w.mean(), 1e-9) - 1)*log['r']
    else:
        t = dm + (w - 1)*(log['r'] - Q[ar, a])
    if charged:
        t = t - R_MAX*charge(P, log['P0'], kind)
    return t, dict(wmax=float(wr.max()), ess=float(w.sum()**2/np.sum(w**2)),
                   clipped=float((wr > M).mean()), charge=float(charge(P, log['P0'], kind).mean()))


# --------------------------------------------------------------------- bounds
def t_lower(x, delta=DELTA):
    n = len(x)
    return float(x.mean() - stats.t.ppf(1-delta, n-1)*x.std(ddof=1)/np.sqrt(n))


def eb_lower(x, delta=DELTA, rng_range=M_DEFAULT+2.0):
    n = len(x)
    return float(x.mean() - np.sqrt(2*x.var(ddof=1)*np.log(2/delta)/n)
                 - 7*rng_range*np.log(2/delta)/(3*(n-1)))


def mom_lower(x, delta=DELTA):
    n = len(x); k = int(min(max(np.ceil(8*np.log(1/delta)), 2), n//8))
    b = np.array_split(x[np.random.default_rng(0).permutation(n)], k)
    return float(np.median([bb.mean() for bb in b]) - x.std(ddof=1)*np.sqrt(32*np.log(1/delta)/n))


def catoni_lower(x, delta=DELTA, iters=60):
    n = len(x); v = x.var(ddof=1)
    alpha = np.sqrt(2*np.log(1/delta)/(n*max(v, 1e-18)))
    psi = lambda u: np.where(u >= 0, np.log1p(u + u**2/2), -np.log1p(-u + u**2/2))
    lo, hi = x.mean() - 12*x.std(ddof=1), x.mean() + 12*x.std(ddof=1)
    for _ in range(iters):
        mid = (lo + hi)/2
        if psi(alpha*(x - mid)).sum() > 0: lo = mid
        else: hi = mid
    return float((lo + hi)/2 - np.sqrt(2*v*np.log(1/delta)/n))


def bca_lower(x, delta=DELTA, B=300, seed=0):
    rng = np.random.default_rng(seed); n = len(x); th = x.mean()
    bs = x[rng.integers(0, n, (B, n))].mean(1)
    z0 = stats.norm.ppf(np.clip((bs < th).mean(), 1e-6, 1-1e-6))
    jk = (x.sum() - x)/(n - 1); jm = jk.mean()
    acc = ((jm - jk)**3).sum()/(6*(((jm - jk)**2).sum())**1.5 + 1e-18)
    zl = stats.norm.ppf(delta)
    alpha = stats.norm.cdf(z0 + (z0 + zl)/(1 - acc*(z0 + zl)))
    return float(np.quantile(bs, np.clip(alpha, 1e-4, .5)))


BOUNDS = {'t': t_lower, 'eb': eb_lower, 'mom': mom_lower, 'catoni': catoni_lower, 'bca': bca_lower}


def betting_lower(x, checks, delta=DELTA, B=M_DEFAULT+1.0, lo=-0.08, hi=0.08, step=1e-4,
                  c=0.5, chunk=256):
    """Lower confidence SEQUENCE (hedged capital, Waudby-Smith & Ramdas), valid at all times.

    Requires |x| <= B, which Proposition 2 supplies: with the closed-form charge the contribution
    is bounded, whereas with the sampled charge it is not.
    """
    z = np.clip((np.asarray(x, dtype=np.float64) + B)/(2*B), 0, 1)
    n = len(z); i = np.arange(1, n+1); k = np.arange(1, n)
    csum, csq = np.cumsum(z), np.cumsum(z**2)
    var_hat = np.concatenate([[0.25], (0.25 + csq[:-1] - csum[:-1]**2/k)/(1 + k)])
    lam_base = np.sqrt(2*np.log(1/delta)/(np.clip(var_hat, 1e-4, None)*i*np.log(1 + i)))
    grid = np.arange(lo, hi + step, step); mz = (grid + B)/(2*B)
    thresh = np.log(1/delta); checks = np.asarray(checks)
    alive = np.zeros((len(grid), len(checks)), dtype=bool)
    for s in range(0, len(grid), chunk):
        mm = mz[s:s+chunk][:, None]
        lam = np.minimum(lam_base[None, :], c/np.maximum(mm, 1e-9))
        logK = np.cumsum(np.log1p(np.clip(lam*(z[None, :] - mm), -0.999999, None)), axis=1)
        alive[s:s+chunk] = np.maximum.accumulate(logK, axis=1)[:, checks-1] <= thresh
    out = np.full(len(checks), hi)
    for j in range(len(checks)):
        w = np.flatnonzero(alive[:, j])
        out[j] = grid[w[0]] - step if len(w) else hi
    return out


# --------------------------------------------------------------------- gate
TAU_DEFAULT = 6e-3      # charge budget for abstention, calibrated out of sample (Section 6.6)


def gate(lb: float, diag: dict, tau: float = TAU_DEFAULT) -> str:
    """Three-way decision, matching Algorithm 1 of the paper.

    The abstention branch is keyed to the charge, which is the bias bound itself and is
    computable from the two policies and the contexts alone. Earlier revisions keyed it to the
    effective sample size and the maximum weight; Section 6.6 shows that neither transfers
    across corpora, which is why the charge is used instead. Abstention is NOT applied in the
    coverage and power audit, so no ABSTAIN outcome enters any reported number.
    """
    if diag['charge'] > tau:
        return 'abstain'
    return 'accept' if lb > 0 else 'reject'


def certifiable(P, P0, delta_true: float, kind: str = 'dr') -> bool:
    """Certifiability frontier.

    Delta > 2 r_max E[c(X)] is SUFFICIENT for eventual acceptance and is not necessary: the
    asymptotic acceptance condition is E[D] - r_max E[c(X)] > 0, which can hold below the
    frontier when the realised bias runs in the favourable direction.
    """
    return delta_true > 2*R_MAX*float(charge(P, P0, kind).mean())
