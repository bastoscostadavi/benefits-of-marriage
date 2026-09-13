"""The mate-search and marriage models of Costa, "Benefits of marriage as a
search strategy" (arXiv:2108.04885).

Dynamics, per discrete step ``t -> t+1``:

1. All agents that are not married are matched into random pairs (a random
   perfect matching; one agent sits out if their number is odd).
2. A matched pair ``(a, b)`` forms a new couple iff *both* strictly prefer the
   other to their current situation: ``A[a,b] > u_t[a]`` and ``A[b,a] > u_t[b]``.
   Both comparisons use the utilities as of the *start* of the step.
3. Every agent whose partner left them becomes single again, and their utility
   reverts to ``A[k,k]``.
4. A couple ``(a, b)`` is married once ``A[a,b] > Lambda[a]`` and
   ``A[b,a] > Lambda[b]``.  Married agents drop out of step 1 forever, so a
   married couple never breaks: ``Lambda = inf`` is the model without marriage.

This mirrors the ``DatingMarriageModel`` function in the paper's Mathematica
notebook (``submissions/wolfram/anc/DatingMarriageModel.nb``), vectorised with numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .affinity import _BaseAffinity, make_affinity

__all__ = ["SimulationResult", "simulate", "simulate_pair"]


@dataclass
class SimulationResult:
    """Everything the figures need, recorded at every step ``t = 0..T``."""

    utility: np.ndarray            # (T+1, N) utility of each agent, if tracked
    mean_utility: np.ndarray       # (T+1,)  u_t
    std_utility: np.ndarray        # (T+1,)  sigma_t
    coupled_share: np.ndarray      # (T+1,)  fraction of agents in any couple
    married_share: np.ndarray      # (T+1,)  fraction of agents married
    partner: np.ndarray            # (N,)    final partner, -1 if single
    married: np.ndarray            # (N,)    final married flag
    n_partners: np.ndarray         # (N,)    distinct partners over the run
    affinity: _BaseAffinity = field(repr=False, default=None)
    lam: float = np.inf
    params: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.partner)

    @property
    def steps(self) -> int:
        return len(self.mean_utility) - 1


def _random_matching(eligible: np.ndarray, rng: np.random.Generator):
    """Split a shuffled list of agents down the middle and zip the halves.

    This is exactly the notebook's construction.  With an odd number of
    eligible agents the last one sits the step out.
    """
    order = rng.permutation(eligible)
    half = len(order) // 2
    return order[:half], order[half:2 * half]


def _bipartite_matching(eligible: np.ndarray, side: np.ndarray,
                        rng: np.random.Generator):
    """Match only across the two sides of a partitioned society.

    Section 2.3 of the paper: separating agents into two groups that only match
    each other recovers the classic two-sided marriage problem.  When the sides
    differ in size the matching is a random injection from the smaller into the
    larger, so a random subset of the larger side sits the step out.
    """
    left = rng.permutation(eligible[side[eligible] == 0])
    right = rng.permutation(eligible[side[eligible] == 1])
    k = min(len(left), len(right))
    return left[:k], right[:k]


def simulate(n=10_000, steps=100, dist="N", lam=np.inf, *, mu=0.0, sigma=1.0,
             mu_s=None, sigma_s=None, sigma_lam=0.0, sigma_q=0.0, seed=0, backend="lazy",
             affinity=None, match_seed=None, track_utility=False,
             track_partners=False, side=None):
    """Run the model for ``steps`` steps and return a :class:`SimulationResult`.

    Parameters
    ----------
    lam
        The proposal threshold ``Lambda``.  ``np.inf`` gives the mate-search
        model with no marriage.  May also be a length-``n`` array, giving each
        agent its own threshold -- used to let a small group deviate from the
        society's convention.
    sigma_lam
        Spread of the per-agent thresholds ``Lambda_k ~ N(lam, sigma_lam)``.
        The paper takes the ``sigma_lam -> 0`` limit, which is the default.
    affinity
        Reuse an affinity matrix built elsewhere (see :func:`simulate_pair`),
        instead of building one from ``seed``.
    track_utility
        Also keep the full ``(steps+1, n)`` utility history.  Needed for the
        histogram and single-agent-trajectory figures; off by default because
        it is the only part of the run whose memory grows with ``steps * n``.
    track_partners
        Count how many distinct partners each agent has over the run.
    side
        Optional 0/1 array partitioning the society into two groups that only
        match across the partition (the two-sided "men and women" variant of
        section 2.3).  ``side="half"`` splits it evenly.  ``None``, the default,
        is the paper's baseline where anyone can match anyone.
    """
    if affinity is None:
        affinity = make_affinity(n, dist, mu=mu, sigma=sigma, mu_s=mu_s,
                                 sigma_s=sigma_s, seed=seed, backend=backend,
                                 sigma_q=sigma_q)
    n = affinity.n
    rng = np.random.default_rng(seed if match_seed is None else match_seed)

    if np.ndim(lam) > 0:
        # a per-agent threshold vector: lets a sub-population deviate from the
        # society's convention, which is what the invasion analysis needs
        thresholds = np.asarray(lam, dtype=float)
        if thresholds.shape != (n,):
            raise ValueError(f"lam vector must have length {n}, got {thresholds.shape}")
    elif np.isinf(lam):
        thresholds = np.full(n, np.inf)
    elif sigma_lam > 0:
        thresholds = rng.normal(lam, sigma_lam, n)
    else:
        thresholds = np.full(n, float(lam))

    if isinstance(side, str):
        if side != "half":
            raise ValueError("side must be None, 'half', or a 0/1 array")
        side = (np.arange(n) >= n // 2).astype(np.int8)
    elif side is not None:
        side = np.asarray(side, dtype=np.int8)

    partner = np.full(n, -1, dtype=np.int64)
    married = np.zeros(n, dtype=bool)
    u = affinity.diag.copy()
    all_idx = np.arange(n, dtype=np.int64)

    history = np.empty((steps + 1, n)) if track_utility else None
    if track_utility:
        history[0] = u
    mean_u = np.empty(steps + 1)
    std_u = np.empty(steps + 1)
    coupled = np.empty(steps + 1)
    marr = np.empty(steps + 1)
    mean_u[0], std_u[0], coupled[0], marr[0] = u.mean(), u.std(), 0.0, 0.0

    seen = [set() for _ in range(n)] if track_partners else None

    for t in range(1, steps + 1):
        eligible = all_idx if not married.any() else all_idx[~married]
        if len(eligible) >= 2:
            if side is None:
                a, b = _random_matching(eligible, rng)
            else:
                a, b = _bipartite_matching(eligible, side, rng)
            u_ab = affinity.pair(a, b)
            u_ba = affinity.pair(b, a)
            love = (u_ab > u[a]) & (u_ba > u[b])
            new_a, new_b = a[love], b[love]

            if len(new_a):
                # everyone abandoned by a partner who just re-coupled goes
                # single *before* the new pairings are written down
                jilted = np.concatenate((partner[new_a], partner[new_b]))
                partner[jilted[jilted >= 0]] = -1
                partner[new_a] = new_b
                partner[new_b] = new_a

                if track_partners:
                    for x, y in zip(new_a.tolist(), new_b.tolist()):
                        seen[x].add(y)
                        seen[y].add(x)

                # a couple marries the moment both affinities clear Lambda
                keep = (u_ab[love] > thresholds[new_a]) & (u_ba[love] > thresholds[new_b])
                married[new_a[keep]] = True
                married[new_b[keep]] = True

        # utilities: A[k, partner[k]] when coupled, A[k, k] when single
        u = affinity.diag.copy()
        taken = partner >= 0
        if taken.any():
            idx = all_idx[taken]
            u[idx] = affinity.pair(idx, partner[idx])

        if track_utility:
            history[t] = u
        mean_u[t], std_u[t] = u.mean(), u.std()
        coupled[t] = taken.mean()
        marr[t] = married.mean()

    if track_partners:
        n_partners = np.fromiter((len(s) for s in seen), dtype=np.int64, count=n)
    else:
        n_partners = np.zeros(n, dtype=np.int64)

    return SimulationResult(
        utility=history, mean_utility=mean_u, std_utility=std_u,
        coupled_share=coupled, married_share=marr, partner=partner,
        married=married, n_partners=n_partners, affinity=affinity,
        lam=float(lam) if np.ndim(lam) == 0 else np.nan,
        params=dict(n=n, steps=steps, dist=dist, mu=mu, sigma=sigma, mu_s=mu_s,
                    sigma_s=sigma_s, sigma_lam=sigma_lam, sigma_q=sigma_q, seed=seed,
                    two_sided=side is not None),
    )


def simulate_pair(n=10_000, steps=100, dist="N", lam=1.0, *, seed=0, **kwargs):
    """Run the model with and without marriage **on the same affinity matrix**.

    The paper's notebook always compares the two systems this way: identical
    preferences, independent random matchings.  Pairing removes the sampling
    noise in ``A`` from the difference between the two curves, which is the
    quantity of interest.

    Returns ``(without_marriage, with_marriage)``.
    """
    backend = kwargs.pop("backend", "lazy")
    affinity = kwargs.pop("affinity", None)
    if affinity is None:
        affinity = make_affinity(
            n, dist, mu=kwargs.get("mu", 0.0), sigma=kwargs.get("sigma", 1.0),
            mu_s=kwargs.get("mu_s"), sigma_s=kwargs.get("sigma_s"),
            seed=seed, backend=backend, sigma_q=kwargs.get("sigma_q", 0.0),
        )
    free = simulate(steps=steps, dist=dist, lam=np.inf, affinity=affinity,
                    seed=seed, match_seed=seed + 1, **kwargs)
    wed = simulate(steps=steps, dist=dist, lam=lam, affinity=affinity,
                   seed=seed, match_seed=seed + 2, **kwargs)
    return free, wed


def replicate(fn, seeds, reducer=None):
    """Run ``fn(seed)`` over ``seeds`` and stack the results.

    The paper reports single runs at N = 10,000.  Averaging a handful of
    independent societies instead costs little and turns every curve into a
    mean with a standard error, which is what a referee will ask for.

    Returns ``(mean, stderr)`` of whatever ``reducer`` extracts (by default
    ``result.mean_utility``).
    """
    reducer = (lambda r: r.mean_utility) if reducer is None else reducer
    stacked = np.stack([np.asarray(reducer(fn(int(s)))) for s in seeds])
    mean = stacked.mean(axis=0)
    stderr = stacked.std(axis=0, ddof=1) / np.sqrt(len(stacked)) if len(stacked) > 1 else np.zeros_like(mean)
    return mean, stderr
