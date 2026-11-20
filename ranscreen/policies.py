"""ranscreen.policies — the policy family, logging, and the offline agent."""
from __future__ import annotations
import numpy as np

# Every policy in the paper is logistic in the A3 margin, parameterised by a hysteresis h [dB]
# and a temperature T. The incumbent is the same family, which keeps importance weights analytic.
CANDIDATES = {'conservative': (5.0, 1.0),
              'neutral':      (3.0, 1.0),
              'aggressive':   (0.0, 0.6),
              'extreme':      (-3.0, 0.3)}
INCUMBENT_HYST = 3.0


def probs(env, idx, hyst: float, temp: float) -> np.ndarray:
    """pi(a | x) for a = (stay, handover)."""
    m = (env.rsrp_n[idx] - env.rsrp_s[idx] - hyst)/temp
    p_ho = 1/(1 + np.exp(-m))
    return np.column_stack([1 - p_ho, p_ho])


def sample_log(env, rng, n: int, temp0: float, hyst0: float = INCUMBENT_HYST, pool=None) -> dict:
    """Draw a logged dataset: contexts from real rows, actions from the incumbent policy."""
    pool = np.arange(len(env)) if pool is None else pool
    idx = rng.choice(pool, size=min(n, len(pool)), replace=False)
    P0 = probs(env, idx, hyst0, temp0)
    a = (rng.random(len(idx)) < P0[:, 1]).astype(int)
    return dict(idx=idx, P0=P0, a=a, prop=P0[np.arange(len(idx)), a],
                r=env.draw_reward(rng, idx, a), n=len(idx), temp0=temp0, hyst0=hyst0)


AGENT_GRID = [(h, t) for h in (-1., 0., 1., 2., 3., 4., 5.) for t in (0.6, 1.0, 1.6)]


def snips(env, log, P, M=20.0) -> float:
    w = np.minimum(P[np.arange(log['n']), log['a']]/np.maximum(log['prop'], 1e-300), M)
    return float((w*log['r']).sum()/max(w.sum(), 1e-9))


def agent_propose(env, log, grid=AGENT_GRID):
    """An autonomous controller searching the class and submitting the argmax of its own estimate.

    This is the realistic object to certify: the winner of a search is selected because its
    estimate is favourable, so a single-policy bound applied to it is invalid.
    """
    best, best_v = None, -np.inf
    for (h, t) in grid:
        v = snips(env, log, probs(env, log['idx'], h, t))
        if v > best_v:
            best_v, best = v, (h, t)
    return best, best_v, best_v - snips(env, log, log['P0'])
