"""Fig. 17 -- is there a simple law for the equilibrium threshold per agent?

For a society constrained to one threshold, the welfare optimum is
Lambda* = sigma: marry if you like your partner a sigma above average.
Figure 16 showed that when each desirability type picks its own threshold, the
equilibrium profile rises monotonically with desirability. This script asks
whether that profile has an equally simple form.

Method: sort agents into deciles by desirability q, run the best-response
iteration over per-decile thresholds, and regress the equilibrium Lambda_d on
the decile's mean q_d. A law should hold across different amounts of
desirability spread, so the whole thing is repeated at sigma_q = 0.3, 0.6, 0.8.

Note that q enters the two sides of the model differently. Agent k's own
affinities A_kj = eps_kj + q_j have the same distribution for every k, so the
scale on which Lambda_k is measured does not depend on k. What changes with
q_k is only how often k is accepted -- which is exactly what should set how
picky k can afford to be.
"""

from _common import N, SEED, T, banner
import numpy as np

from marriage.affinity import make_affinity
from marriage.model import simulate
from marriage.style import LAMBDA_COLORS, new_fig, save

GROUPS = 10
SEEDS = 4
ROUNDS = 3
GRID = np.arange(-0.5, 2.51, 0.25)
SIGMA_QS = [0.3, 0.6, 0.8]

_cache = {}


def society(seed, sigma_q):
    key = (seed, sigma_q)
    if key not in _cache:
        sigma = float(np.sqrt(max(1e-12, 1 - sigma_q ** 2)))
        aff = make_affinity(N, "N", sigma=sigma, sigma_q=sigma_q, seed=seed)
        order = np.argsort(aff.column_means())
        label = np.empty(N, dtype=np.int64)
        for g in range(GROUPS):
            label[order[g * N // GROUPS:(g + 1) * N // GROUPS]] = g
        q = np.array([aff.quality[label == g].mean() for g in range(GROUPS)])
        _cache[key] = (aff, label, q)
    return _cache[key]


def group_payoff(profile, g, seed, sigma_q):
    aff, label, _ = society(seed, sigma_q)
    res = simulate(steps=T, dist="N", lam=np.take(profile, label), affinity=aff,
                   seed=seed, match_seed=seed + 7, track_utility=True)
    return res.utility[-1][label == g].mean()


def equilibrium(sigma_q):
    profile = np.full(GROUPS, 1.0)
    for _ in range(ROUNDS):
        for g in range(GROUPS):
            pay = [np.mean([group_payoff(np.where(np.arange(GROUPS) == g, l, profile),
                                         g, SEED + s, sigma_q)
                            for s in range(SEEDS)]) for l in GRID]
            profile[g] = float(GRID[int(np.argmax(pay))])
    return profile


banner("Fig. 17: a law for the equilibrium threshold")
fig, axes = new_fig(ncols=2, width=4.3)
rows = []

for sigma_q, color in zip(SIGMA_QS, LAMBDA_COLORS):
    prof = equilibrium(sigma_q)
    q = np.mean([society(SEED + s, sigma_q)[2] for s in range(SEEDS)], axis=0)
    slope, intercept = np.polyfit(q, prof, 1)
    resid = prof - (slope * q + intercept)
    rows.append((sigma_q, prof, q, slope, intercept))
    print(f"  sigma_q={sigma_q:g}")
    print(f"    q_d      : " + " ".join(f"{v:+.2f}" for v in q))
    print(f"    Lambda_d : " + " ".join(f"{v:+.2f}" for v in prof))
    print(f"    fit Lambda = {slope:.3f} q + {intercept:.3f}   "
          f"(rms residual {np.sqrt((resid**2).mean()):.3f})")

    axes[0].plot(q, prof, "o-", color=color, label=rf"$\sigma_q={sigma_q:g}$")
    axes[1].plot(np.arange(1, GROUPS + 1) * 10 - 5, prof, "o-", color=color,
                 label=rf"$\sigma_q={sigma_q:g}$")

qq = np.linspace(-1.6, 1.6, 50)
axes[0].plot(qq, 1.0 + qq, "k--", lw=1.2, label=r"$\Lambda = \sigma + q$")
axes[0].set_xlabel(r"group mean desirability $q_d$")
axes[0].set_ylabel(r"equilibrium $\Lambda_d$")
axes[0].set_title("threshold against desirability")
axes[0].legend(fontsize=8)

axes[1].axhline(1.0, color="k", ls=":", lw=1, label=r"uniform optimum $\Lambda=\sigma$")
axes[1].set_xlabel("desirability percentile")
axes[1].set_ylabel(r"equilibrium $\Lambda_d$")
axes[1].set_title("threshold against percentile")
axes[1].legend(fontsize=8)

print("\n  pooled fit across all sigma_q:")
allq = np.concatenate([r[2] for r in rows]); allp = np.concatenate([r[1] for r in rows])
s, i = np.polyfit(allq, allp, 1)
r2 = 1 - ((allp - (s * allq + i)) ** 2).sum() / ((allp - allp.mean()) ** 2).sum()
print(f"    Lambda = {s:.3f} q + {i:.3f}    R^2 = {r2:.4f}")
print(f"    against the guess Lambda = sigma + q (slope 1, intercept 1): "
      f"rms {np.sqrt(((allp - (1.0 + allq))**2).mean()):.3f}")

save(fig, "fig17_threshold_law")
