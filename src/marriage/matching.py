"""Benchmarks the decentralised model is compared against: Gale-Shapley stable
matching and the utilitarian (Becker) optimum.

Both split the society into two equal halves and match across them, which is
the setting Gale-Shapley is defined for.  The paper notes that the stable
matching reaches a far higher average utility than the dynamics ever do
(``u_GS ~ 4`` at N ~ 1000 in the Gaussian model), which is its argument that
the decentralised relationship market is badly sub-optimal.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["gale_shapley", "stable_matching_utility", "utilitarian_optimum", "random_matching_utility"]


def gale_shapley(prop_util: np.ndarray, recv_util: np.ndarray) -> np.ndarray:
    """Deferred acceptance.  ``prop_util[i, j]`` is proposer i's utility for
    receiver j, ``recv_util[j, i]`` receiver j's utility for proposer i.

    Returns ``match`` with ``match[i] = j``, the proposer-optimal stable matching.
    """
    n = prop_util.shape[0]
    # each proposer's receivers, best first
    order = np.argsort(-prop_util, axis=1)
    next_choice = np.zeros(n, dtype=np.int64)
    held = np.full(n, -1, dtype=np.int64)      # receiver -> proposer, -1 if free
    free = list(range(n))

    while free:
        i = free.pop()
        j = int(order[i, next_choice[i]])
        next_choice[i] += 1
        current = held[j]
        if current == -1:
            held[j] = i
        elif recv_util[j, i] > recv_util[j, current]:
            held[j] = i
            free.append(int(current))
        else:
            free.append(i)

    match = np.empty(n, dtype=np.int64)
    match[held] = np.arange(n)
    return match


def _halves(affinity, n):
    """Utility blocks for the two halves of the society."""
    left = np.arange(n // 2, dtype=np.int64)
    right = np.arange(n // 2, 2 * (n // 2), dtype=np.int64)
    L = np.stack([affinity.pair(np.full(len(right), i), right) for i in left])
    R = np.stack([affinity.pair(np.full(len(left), j), left) for j in right])
    return left, right, L, R


def stable_matching_utility(affinity, n=None):
    """Average utility of the proposer-optimal stable matching, ``u_GS``."""
    n = affinity.n if n is None else n
    left, right, L, R = _halves(affinity, n)
    match = gale_shapley(L, R)
    rows = np.arange(len(left))
    return float(np.mean(np.concatenate([L[rows, match], R[match, rows]])))


def utilitarian_optimum(affinity, n=None):
    """Average utility of the matching maximising the total surplus (Becker).

    The paper calls this computationally impossible because it counts all ``N!``
    matchings; it is in fact a linear assignment problem, so the Hungarian
    algorithm solves it in ``O(N^3)``.  Practical up to a few thousand agents.
    """
    n = affinity.n if n is None else n
    left, right, L, R = _halves(affinity, n)
    surplus = L + R.T
    rows, cols = linear_sum_assignment(-surplus)
    return float(surplus[rows, cols].mean() / 2)


def random_matching_utility(affinity, n=None, rng=None):
    """Average utility if everyone is paired uniformly at random: the floor."""
    n = affinity.n if n is None else n
    rng = np.random.default_rng() if rng is None else rng
    order = rng.permutation(n)
    a, b = order[: n // 2], order[n // 2: 2 * (n // 2)]
    return float(np.mean(np.concatenate([affinity.pair(a, b), affinity.pair(b, a)])))
