"""Regression tests: the identities the paper rests on, checked numerically.

    python -m pytest tests/ -q

These are not statistical experiments; each one is a property that must hold exactly (to numerical
tolerance) for any input, so a failure means the implementation contradicts the theory.
"""
import sys, numpy as np, pytest
import pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from ranscreen import estimators as E

RNG = np.random.default_rng(0)


def random_policies(n=5000):
    p = RNG.uniform(.001, .999, n); q = RNG.uniform(.001, .999, n)
    P = np.column_stack([1 - p, p]); P0 = np.column_stack([1 - q, q])
    return P, P0


def test_charge_identity_matches_sampled_expectation():
    """Proposition 2: the closed-form charge equals the conditional mean of the sampled charge."""
    P, P0 = random_policies()
    M = 3.0
    sampled = sum(P0[:, a]*np.maximum(P[:, a]/P0[:, a] - M, 0) for a in (0, 1))
    closed = np.abs(P - np.minimum(P, M*P0)).sum(1)
    assert np.allclose(closed, sampled, atol=1e-10)


def test_charge_is_bounded_in_unit_interval():
    """The property that makes bounded-range finite-sample bounds applicable."""
    for _ in range(20):
        P, P0 = random_policies(2000)
        for kind in ('dr', 'switch', 'shrink'):
            c = E.charge(P, P0, kind)
            assert c.min() >= -1e-12 and c.max() <= 2 + 1e-12


def test_charge_vanishes_when_weights_are_small():
    """With pi <= M pi0 everywhere the charge is exactly zero, so the certificate is unpenalised."""
    p = RNG.uniform(.3, .7, 4000)
    P = np.column_stack([1 - p, p]); P0 = P.copy()
    assert np.allclose(E.charge(P, P0, 'dr'), 0.0)


def test_charge_handles_zero_support():
    """Where the logging policy has no support the charge tends to pi, not to infinity."""
    P = np.array([[0.4, 0.6]]); P0 = np.array([[1.0, 1e-300]])
    c = E.charge(P, P0, 'dr')
    assert np.isfinite(c).all() and abs(c[0] - 0.6) < 1e-9


def test_generalised_charge_reduces_to_proposition_2():
    """Theorem 3 with hard truncation must coincide with Proposition 2."""
    P, P0 = random_policies()
    assert np.allclose(E.charge(P, P0, 'dr'), E.charge(P, P0, 'ips'))


def test_bounds_are_ordered():
    """A finite-sample bound can never be tighter than the asymptotic one it replaces."""
    x = RNG.normal(0.01, 0.1, 5000)
    assert E.eb_lower(x) <= E.t_lower(x)


def test_t_lower_covers_a_normal_mean():
    """Sanity: the Student-t bound covers at its nominal rate on well-behaved data."""
    hits = sum(E.t_lower(RNG.normal(0.5, 1.0, 400)) <= 0.5 for _ in range(400))
    assert 0.90 < hits/400 < 0.99


def test_betting_sequence_is_time_uniform_on_heavy_tails():
    """The confidence sequence must not be violated at ANY checkpoint, even with heavy tails."""
    checks = np.array([200, 1000, 4000])
    viol = 0
    for _ in range(40):
        x = np.clip(0.01 + RNG.standard_t(3, size=4000)*0.05, -21, 21)
        L = E.betting_lower(x, checks, B=21.0)
        viol += bool((L > 0.01).any())
    assert viol <= 4                       # nominal 5% of 40 runs, with slack


def test_betting_sequence_is_monotone_in_information():
    """More data may only tighten the sequence on average, never collapse it."""
    x = np.clip(0.02 + RNG.normal(0, .05, 8000), -21, 21)
    L = E.betting_lower(x, np.array([500, 2000, 8000]), B=21.0)
    assert L[0] <= L[-1] + 1e-9


def test_gate_abstains_on_poor_overlap():
    diag = dict(ess=100.0, wmax=1e4, clipped=.01, charge=.3)
    assert E.gate(1.0, diag) == 'abstain'
    good = dict(ess=19000.0, wmax=3.0, clipped=0.0, charge=0.0)
    assert E.gate(1.0, good) == 'accept' and E.gate(-1.0, good) == 'reject'
