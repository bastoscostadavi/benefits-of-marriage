"""Affinity matrices for the mate-search model.

The affinity matrix ``A`` holds, in ``A[i, j]`` with ``i != j``, the utility
agent ``i`` gets from being in a couple with agent ``j``.  It is not symmetric.
The diagonal ``A[k, k]`` is the utility of agent ``k`` while single.

Two backends are provided:

``DenseAffinity``
    Materialises the whole ``N x N`` matrix.  Exact, simple, and the right
    choice for validation, but costs ``8 N^2`` bytes (800 MB at N = 10,000).

``LazyAffinity``
    Never stores the off-diagonal entries.  ``A[i, j]`` is derived on demand
    from a counter-based hash of ``(seed, i, j)``, so repeated lookups of the
    same pair always return the same number and the matrix is still a fixed
    object, just an implicit one.  Memory is ``O(N)``, which is what lets the
    scaling study in ``scripts/fig12_scaling.py`` reach N = 10^6.

Both expose the same interface: ``diag``, ``pair(a, b)`` and ``row(k)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

__all__ = ["Distribution", "GAUSSIAN", "UNIFORM", "DenseAffinity", "LazyAffinity", "make_affinity"]


class Distribution:
    """A location/scale family parameterised the way the paper does it.

    Both families have mean ``mu`` and standard deviation ``sigma``; the uniform
    one is ``U(mu - 2 sigma, mu + 2 sigma)``, whose standard deviation is
    ``2 sigma / sqrt(3)``.  The paper calls that "variance sigma", treating
    ``2 sigma`` as the half-width; we keep the paper's convention so that
    ``UNIFORM`` with ``sigma=1`` is ``U(-2, 2)``.
    """

    def __init__(self, kind: str):
        if kind not in ("N", "U"):
            raise ValueError(f"unknown distribution {kind!r}, expected 'N' or 'U'")
        self.kind = kind

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Distribution({self.kind!r})"

    @property
    def name(self) -> str:
        return "Gaussian" if self.kind == "N" else "Uniform"

    def label(self, mu: float = 0.0, sigma: float = 1.0) -> str:
        if self.kind == "N":
            return rf"$\mathcal{{N}}({mu:g},{sigma:g})$"
        return rf"$U({mu - 2 * sigma:g},{mu + 2 * sigma:g})$"

    def sample(self, rng: np.random.Generator, mu: float, sigma: float, size) -> np.ndarray:
        if self.kind == "N":
            return rng.normal(mu, sigma, size)
        return rng.uniform(mu - 2 * sigma, mu + 2 * sigma, size)

    def ppf(self, q: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Inverse CDF, used to turn uniform hashes into affinity values."""
        if self.kind == "N":
            return mu + sigma * ndtri(q)
        return mu - 2 * sigma + 4 * sigma * q

    def cdf(self, u, mu: float = 0.0, sigma: float = 1.0):
        from scipy.special import ndtr

        u = np.asarray(u, dtype=float)
        if self.kind == "N":
            return ndtr((u - mu) / sigma)
        return np.clip((u - mu + 2 * sigma) / (4 * sigma), 0.0, 1.0)

    def pdf(self, u, mu: float = 0.0, sigma: float = 1.0):
        u = np.asarray(u, dtype=float)
        if self.kind == "N":
            return np.exp(-0.5 * ((u - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        inside = (u >= mu - 2 * sigma) & (u <= mu + 2 * sigma)
        return np.where(inside, 1.0 / (4 * sigma), 0.0)

    def support(self, mu: float = 0.0, sigma: float = 1.0, tail: float = 5.0):
        if self.kind == "N":
            return mu - tail * sigma, mu + tail * sigma
        return mu - 2 * sigma, mu + 2 * sigma


GAUSSIAN = Distribution("N")
UNIFORM = Distribution("U")


# --------------------------------------------------------------------------
# counter-based hashing (splitmix64), for the lazy backend
# --------------------------------------------------------------------------

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_S1, _S2, _S3 = np.uint64(30), np.uint64(27), np.uint64(31)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser.  Input and output are uint64 arrays."""
    with np.errstate(over="ignore"):
        z = x + _GAMMA
        z = (z ^ (z >> _S1)) * _M1
        z = (z ^ (z >> _S2)) * _M2
        return z ^ (z >> _S3)


def _uniform01(seed: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Deterministic U(0,1) draw attached to the ordered pair ``(i, j)``."""
    i = np.asarray(i, dtype=np.uint64)
    j = np.asarray(j, dtype=np.uint64)
    with np.errstate(over="ignore"):
        key = _splitmix64(i * _GAMMA + np.uint64(seed))
        key = _splitmix64(key ^ (j + _GAMMA))
    # top 53 bits -> a double in [0, 1)
    return (key >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)


class _BaseAffinity:
    """Common interface.  ``diag`` is always materialised (it is only O(N)).

    ``quality`` is an optional per-agent desirability offset added to every
    entry of that agent's *column*: ``A[i, j] = draw + quality[j]``.  The paper
    has no such term -- there, all agents are exchangeable and the only
    heterogeneity is the sampling noise of a finite matrix.  With
    ``quality != 0`` some agents really are more widely liked than others, which
    is what lets ``scripts/fig07_liked_disliked.py`` test the paper's claim
    about who marriage helps.
    """

    def __init__(self, n: int, diag: np.ndarray, quality: np.ndarray = None):
        self.n = n
        self.diag = diag
        self.quality = quality

    def pair(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def row(self, k: int) -> np.ndarray:
        raise NotImplementedError

    def column_means(self, chunk: int = 512) -> np.ndarray:
        """Mean of ``A[:, k]`` over ``j != k``: how much society likes agent k."""
        raise NotImplementedError


class DenseAffinity(_BaseAffinity):
    """The full matrix, held in memory."""

    def __init__(self, matrix: np.ndarray, quality: np.ndarray = None):
        super().__init__(matrix.shape[0], np.diag(matrix).copy(), quality)
        self.A = matrix

    def pair(self, a, b):
        return self.A[a, b]

    def row(self, k: int) -> np.ndarray:
        return self.A[k]

    def column_means(self, chunk: int = 512) -> np.ndarray:
        n = self.n
        total = self.A.sum(axis=0) - self.diag
        return total / (n - 1)


class LazyAffinity(_BaseAffinity):
    """Off-diagonal entries generated on demand from a hash of ``(i, j)``.

    Statistically indistinguishable from an i.i.d. draw, but reproducible and
    memory-free: the same ``(seed, i, j)`` always yields the same value, so the
    affinity matrix is just as fixed as in the dense case.
    """

    def __init__(self, n: int, dist: Distribution, mu: float, sigma: float,
                 diag: np.ndarray, seed: int, quality: np.ndarray = None):
        super().__init__(n, diag, quality)
        self.dist = dist
        self.mu = mu
        self.sigma = sigma
        self.seed = int(seed) & ((1 << 64) - 1)

    def pair(self, a, b):
        a = np.atleast_1d(np.asarray(a, dtype=np.int64))
        b = np.atleast_1d(np.asarray(b, dtype=np.int64))
        out = self.dist.ppf(_uniform01(self.seed, a, b), self.mu, self.sigma)
        if self.quality is not None:
            out = out + self.quality[b]
        same = a == b
        if same.any():
            out = np.where(same, self.diag[a], out)
        return out

    def row(self, k: int) -> np.ndarray:
        idx = np.arange(self.n, dtype=np.int64)
        return self.pair(np.full(self.n, k, dtype=np.int64), idx)

    def column_means(self, chunk: int = 512) -> np.ndarray:
        n = self.n
        totals = np.zeros(n)
        cols = np.arange(n, dtype=np.int64)
        for start in range(0, n, chunk):
            rows = np.arange(start, min(start + chunk, n), dtype=np.int64)
            block = self.dist.ppf(
                _uniform01(self.seed, rows[:, None], cols[None, :]), self.mu, self.sigma
            )
            if self.quality is not None:
                block = block + self.quality[None, :]
            np.fill_diagonal(block[:, start:start + len(rows)], 0.0)
            totals += block.sum(axis=0)
        return totals / (n - 1)


def make_affinity(n, dist, mu=0.0, sigma=1.0, mu_s=None, sigma_s=None,
                  seed=0, backend="lazy", sigma_q=0.0):
    """Build an affinity matrix for ``n`` agents.

    ``mu, sigma`` parameterise the off-diagonal entries (how much agents like
    each other); ``mu_s, sigma_s`` the diagonal (how much they like being
    single).  They default to the same values, which is the paper's baseline.
    Pass ``sigma_s=0`` for the ``A_kk = 0`` variant of section 2.3.

    ``sigma_q > 0`` adds a per-agent desirability ``q_k ~ N(0, sigma_q)`` to
    every entry of column ``k``, giving vertical heterogeneity that the paper's
    model does not have.  The marginal distribution of ``A[i, j]`` then has
    variance ``sigma^2 + sigma_q^2``, so compare runs at fixed total variance.
    """
    dist = _as_dist(dist)
    mu_s = mu if mu_s is None else mu_s
    sigma_s = sigma if sigma_s is None else sigma_s
    rng = np.random.default_rng(seed)

    if sigma_s == 0:
        diag = np.full(n, float(mu_s))
    else:
        diag = dist.sample(rng, mu_s, sigma_s, n)

    quality = rng.normal(0.0, sigma_q, n) if sigma_q > 0 else None

    if backend == "dense":
        A = dist.sample(rng, mu, sigma, (n, n))
        if quality is not None:
            A += quality[None, :]
        np.fill_diagonal(A, diag)
        return DenseAffinity(A, quality)
    if backend == "lazy":
        # a second, independent 64-bit seed for the off-diagonal hash stream
        off_seed = int(rng.integers(1, 2 ** 63 - 1))
        return LazyAffinity(n, dist, mu, sigma, diag, off_seed, quality)
    raise ValueError(f"unknown backend {backend!r}, expected 'lazy' or 'dense'")


def _as_dist(dist) -> Distribution:
    if isinstance(dist, Distribution):
        return dist
    if isinstance(dist, str):
        key = dist.strip().lower()
        if key in ("n", "gaussian", "normal"):
            return GAUSSIAN
        if key in ("u", "uniform"):
            return UNIFORM
    raise ValueError(f"unknown distribution {dist!r}")
