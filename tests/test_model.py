"""Tests for the simulation core, including the invariants the paper relies on."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from marriage.affinity import make_affinity
from marriage.model import simulate, simulate_pair


def test_both_backends_agree_in_distribution():
    a = make_affinity(2000, "N", seed=1, backend="lazy")
    b = make_affinity(2000, "N", seed=1, backend="dense")
    for aff in (a, b):
        v = aff.pair(np.arange(1000), np.arange(1000, 2000))
        assert abs(v.mean()) < 0.1 and abs(v.std() - 1) < 0.05


def test_lazy_affinity_is_a_fixed_matrix():
    """The same (i, j) must give the same value every time it is looked up."""
    aff = make_affinity(500, "N", seed=3)
    i, j = np.array([7, 11, 499]), np.array([3, 3, 0])
    assert np.array_equal(aff.pair(i, j), aff.pair(i, j))
    assert not np.allclose(aff.pair(i, j), aff.pair(j, i))   # A is not symmetric


def test_partner_map_stays_an_involution():
    """partner[partner[k]] == k for every coupled agent, at every step."""
    for lam in (np.inf, 1.0, -2.0):
        res = simulate(n=2000, steps=40, dist="N", lam=lam, seed=4)
        taken = res.partner >= 0
        assert np.array_equal(res.partner[res.partner[taken]], np.flatnonzero(taken))
        assert not np.any(res.partner[taken] == np.flatnonzero(taken))   # nobody self-paired


def test_married_agents_are_coupled_and_above_threshold():
    lam = 1.0
    res = simulate(n=4000, steps=60, dist="N", lam=lam, seed=5)
    married = np.flatnonzero(res.married)
    assert np.all(res.partner[married] >= 0)
    assert np.all(res.affinity.pair(married, res.partner[married]) > lam)
    assert np.all(res.married[res.partner[married]])          # marriage is mutual


def test_married_share_never_decreases():
    """Married couples never break, so the share is monotone."""
    res = simulate(n=4000, steps=80, dist="N", lam=1.0, seed=6)
    assert np.all(np.diff(res.married_share) >= -1e-12)


def test_mean_utility_never_decreases():
    """No agent's utility can fall below A_kk, and couples only trade up."""
    for dist in ("N", "U"):
        res = simulate(n=4000, steps=80, dist=dist, lam=np.inf, seed=7)
        assert np.all(np.diff(res.mean_utility) > -0.02)


def test_infinite_lambda_never_marries():
    res = simulate(n=2000, steps=50, dist="N", lam=np.inf, seed=8)
    assert not res.married.any()
    assert res.married_share.max() == 0.0


def test_coupled_share_after_one_step_is_one_quarter():
    """b_1 = 1/4 exactly in the large-N limit (Eq. 16 of the paper)."""
    res = simulate(n=200_000, steps=1, dist="U", lam=np.inf, seed=9)
    assert res.coupled_share[1] == pytest.approx(0.25, abs=0.005)


def test_marriage_beats_no_marriage_at_lambda_sigma():
    """The paper's headline claim, on identical affinity matrices."""
    for dist in ("N", "U"):
        free, wed = simulate_pair(n=10_000, steps=100, dist=dist, lam=1.0, seed=10)
        assert wed.mean_utility[-1] > free.mean_utility[-1] + 0.1
        assert wed.std_utility[-1] < free.std_utility[-1]      # and lowers inequality


def test_marrying_the_first_person_you_like_is_worst():
    free, wed = simulate_pair(n=10_000, steps=100, dist="N", lam=-2.0, seed=11)
    assert wed.mean_utility[-1] < free.mean_utility[-1]


def test_two_sided_society_matches_the_one_sided_one():
    """Section 2.3: partitioning into two groups changes nothing visible."""
    one = np.mean([simulate(n=10_000, steps=100, dist="N", lam=1.0, seed=20 + s).mean_utility[-1]
                   for s in range(4)])
    two = np.mean([simulate(n=10_000, steps=100, dist="N", lam=1.0, seed=20 + s,
                            side="half").mean_utility[-1] for s in range(4)])
    assert abs(one - two) < 0.03


def test_shift_and_rescale_symmetry():
    """Eq. (11)/(14): (mu, sigma, sigma*L + mu) maps u to sigma*u + mu."""
    mu, sigma, lam = 3.0, 2.5, 1.0
    base = simulate(n=8000, steps=50, dist="N", lam=lam, seed=12)
    scaled = simulate(n=8000, steps=50, dist="N", lam=sigma * lam + mu, seed=12,
                      mu=mu, sigma=sigma)
    expected = sigma * base.mean_utility + mu
    assert np.allclose(scaled.mean_utility, expected, atol=0.08)


def test_per_agent_thresholds():
    """A lam vector lets a subgroup deviate from the society's convention."""
    n = 4000
    lam = np.full(n, 1.0)
    lam[:200] = np.inf                       # these agents never marry
    res = simulate(n=n, steps=60, dist="N", lam=lam, seed=30)
    assert not res.married[:200].any()
    assert res.married[200:].any()
    with pytest.raises(ValueError):
        simulate(n=n, steps=2, dist="N", lam=np.ones(7), seed=30)


def test_commitment_norm_is_not_self_enforcing_at_the_top():
    """The most desirable agents gain by defecting from the welfare-optimal norm.

    Society plays Lambda = 1; the most desirable 5% deviate to Lambda' = 2.
    """
    from marriage.affinity import make_affinity

    n, gains = 10_000, []
    for seed in range(3):
        aff = make_affinity(n, "N", sigma=0.8, sigma_q=0.6, seed=seed)
        top = np.argsort(aff.column_means())[-500:]
        base = np.full(n, 1.0)
        conform = simulate(steps=100, dist="N", lam=base, affinity=aff,
                           seed=seed, match_seed=seed + 7, track_utility=True)
        lam = base.copy()
        lam[top] = 2.0
        defect = simulate(steps=100, dist="N", lam=lam, affinity=aff,
                          seed=seed, match_seed=seed + 7, track_utility=True)
        gains.append(defect.utility[-1, top].mean() - conform.utility[-1, top].mean())
    assert np.mean(gains) > 0.1
