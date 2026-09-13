"""Fig. 15 -- is the welfare-optimal commitment threshold self-enforcing?

Sections 5.1 and 5.5 leave a question hanging. The threshold Lambda is handed to
the agents exogenously, as a social convention, and at Lambda ~ sigma it
maximises utilitarian welfare. But nothing makes an individual agent want to
obey it. This script asks whether it would.

The experiment is a standard invasion analysis. The society plays the convention
Lambda; a small group (5% of agents) deviates to Lambda'; we measure the
deviators' payoff at t = 100 as a function of Lambda'. If the best response is
the convention itself, the norm is self-enforcing.

Three findings:

* In the homogeneous model the best response sits just above the welfare
  optimum -- agents want to search slightly longer than is socially optimal --
  but the gain is small and the resulting welfare loss is not measurable.
* With desirability spread the private optimum separates from the social one and
  the norm becomes strictly inefficient: a measurable price of anarchy.
* The pressure is concentrated entirely at the top. The most desirable agents
  gain ~20% by defecting to near-zero commitment; the least desirable are
  indifferent, because their threshold barely affects an outcome that is
  determined by whether anyone accepts them at all.

So the benefit of marriage documented in Section 5 is a real social optimum, but
it is not an equilibrium of the game in which agents pick their own thresholds,
and it fails in the direction that hurts the agents it was helping most.
"""

from _common import N, SEED, T, banner
import numpy as np

from marriage.affinity import make_affinity
from marriage.model import simulate
from marriage.style import BLUE, LAMBDA_COLORS, ORANGE, new_fig, save

MUT = 500                    # size of the deviating group (5%)
SEEDS = 6
GRID = np.arange(0.5, 3.01, 0.25)
DEV_GRID = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CONVENTION = 1.0

_cache = {}


def society(seed, sigma_q):
    """Affinity matrix, desirability ranking and a fixed random mutant set."""
    key = (seed, sigma_q)
    if key not in _cache:
        sigma = float(np.sqrt(max(1e-12, 1 - sigma_q ** 2)))
        aff = make_affinity(N, "N", sigma=sigma, sigma_q=sigma_q, seed=seed)
        order = np.argsort(aff.column_means())
        rnd = np.random.default_rng(seed).choice(N, MUT, replace=False)
        _cache[key] = (aff, order, rnd)
    return _cache[key]


def welfare(lam, seed, sigma_q):
    aff, _, _ = society(seed, sigma_q)
    return simulate(steps=T, dist="N", lam=np.full(N, lam), affinity=aff,
                    seed=seed, match_seed=seed + 7).mean_utility[-1]


def deviate(resident, dev, seed, sigma_q, group="random"):
    """Payoff at t = T to a group deviating to `dev` while the rest play `resident`."""
    aff, order, rnd = society(seed, sigma_q)
    idx = {"random": rnd, "top": order[-MUT:], "bottom": order[:MUT]}[group]
    lam = np.full(N, resident)
    lam[idx] = dev
    res = simulate(steps=T, dist="N", lam=lam, affinity=aff, seed=seed,
                   match_seed=seed + 7, track_utility=True)
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True
    return res.utility[-1, mask].mean()


def mean_se(vals):
    vals = np.asarray(vals)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))


banner("Fig. 15: is the commitment norm self-enforcing?")
fig, axes = new_fig(ncols=3, width=3.6)

# --- panel A: welfare against Lambda, with and without desirability spread ---
print("  social optimum")
optima = {}
for sq, color in ((0.0, BLUE), (0.6, ORANGE)):
    w = np.array([[welfare(l, SEED + s, sq) for l in GRID] for s in range(SEEDS)])
    m, e = w.mean(axis=0), w.std(axis=0, ddof=1) / np.sqrt(SEEDS)
    star = float(GRID[int(np.argmax(m))])
    optima[sq] = star
    axes[0].plot(GRID, m, "o-", color=color, label=rf"$\sigma_q={sq:g}$")
    axes[0].fill_between(GRID, m - e, m + e, color=color, alpha=0.25, lw=0)
    axes[0].axvline(star, color=color, ls=":", lw=1)
    print(f"    sigma_q={sq:g}: Lambda* = {star:.2f}, welfare {m.max():.4f}")
axes[0].set_xlabel(r"society-wide $\Lambda$")
axes[0].set_ylabel(f"welfare $u_{{{T}}}$")
axes[0].set_title("the welfare optimum")
axes[0].legend(fontsize=8)

# --- panel B: payoff to a deviating group, residents at the convention -------
print(f"  deviation payoffs, residents at Lambda = {CONVENTION:g}")
for (group, sq, color, style) in (("random", 0.0, BLUE, "o-"),
                                  ("top", 0.6, ORANGE, "s-"),
                                  ("bottom", 0.6, LAMBDA_COLORS[3], "^-")):
    vals = np.array([[deviate(CONVENTION, d, SEED + s, sq, group) for d in DEV_GRID]
                     for s in range(SEEDS)])
    m, e = vals.mean(axis=0), vals.std(axis=0, ddof=1) / np.sqrt(SEEDS)
    label = {"random": r"random 5%, $\sigma_q=0$",
             "top": r"most desirable 5%, $\sigma_q=0.6$",
             "bottom": r"least desirable 5%, $\sigma_q=0.6$"}[group]
    axes[1].errorbar(DEV_GRID, m, yerr=e, fmt=style, color=color, capsize=2,
                     ms=3.5, label=label)
    at_conv = m[DEV_GRID.index(CONVENTION)]
    best = int(np.argmax(m))
    print(f"    {label:38s} best response {DEV_GRID[best]:+.2f}, "
          f"payoff {m[best]:.3f} vs {at_conv:.3f} conforming "
          f"(gain {m[best] - at_conv:+.3f})")
axes[1].axvline(CONVENTION, color="k", ls=":", lw=1)
axes[1].annotate("convention", (CONVENTION, axes[1].get_ylim()[0]), fontsize=7,
                 rotation=90, va="bottom", ha="right")
axes[1].set_xlabel(r"deviator's threshold $\Lambda'$")
axes[1].set_ylabel(f"deviator's payoff at $t={T}$")
axes[1].set_title("payoff to defecting")
axes[1].legend(fontsize=7)

# --- panel C: who wants to defect, by desirability decile -------------------
print(f"  gain from defecting to Lambda' = 2, by desirability decile "
      f"(sigma_q = 0.6)")
DECILES = 10
gains, errs = [], []
for d in range(DECILES):
    per_seed = []
    for s in range(SEEDS):
        aff, order, _ = society(SEED + s, 0.6)
        idx = order[d * N // DECILES:(d + 1) * N // DECILES][:MUT]
        base = np.full(N, CONVENTION)
        conform = simulate(steps=T, dist="N", lam=base, affinity=aff, seed=SEED + s,
                           match_seed=SEED + s + 7, track_utility=True)
        lam = base.copy()
        lam[idx] = 2.0
        defect = simulate(steps=T, dist="N", lam=lam, affinity=aff, seed=SEED + s,
                          match_seed=SEED + s + 7, track_utility=True)
        per_seed.append(defect.utility[-1, idx].mean() - conform.utility[-1, idx].mean())
    m, e = mean_se(per_seed)
    gains.append(m)
    errs.append(e)
    print(f"    decile {d + 1:2d} (least liked = 1): {m:+.4f} +/- {e:.4f}")

axes[2].bar(np.arange(1, DECILES + 1), gains, yerr=errs, capsize=2,
            color=[ORANGE if g > 0 else BLUE for g in gains])
axes[2].axhline(0, color="k", lw=0.8)
axes[2].set_xlabel("desirability decile (10 = most liked)")
axes[2].set_ylabel(r"gain from defecting to $\Lambda'=2$")
axes[2].set_title(r"who wants to defect ($\sigma_q=0.6$)")

save(fig, "fig15_norm_stability")
