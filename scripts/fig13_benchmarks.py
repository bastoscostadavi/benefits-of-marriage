"""Fig. 13 -- how far is the decentralised outcome from a planner's?

Three reference points, all on the same affinity matrix as the dynamics:

* random pairing -- the floor, average utility ~ mu;
* Gale-Shapley -- the proposer-optimal stable matching;
* the utilitarian optimum -- the matching maximising total surplus (Becker).

Two notes on the paper's version of this comparison.

The paper reports u_GS ~ 4 for the Gaussian model at N ~ 1000.  We get ~1.8,
and the gap is structural: in a proposer-optimal stable matching proposers do
very well (~2.2 sigma here) but receivers do badly (~0.9 sigma), and u_GS is
the average of the two sides.  4 sigma is above even the best-possible partner
averaged over proposers, so no stable matching can reach it at this N.

The paper also calls the utilitarian optimum computationally impossible,
counting N! matchings.  It is a linear assignment problem, so the Hungarian
algorithm solves it exactly in O(N^3) -- included here up to N = 2000.

The qualitative conclusion survives both corrections: a planner does far better
than the dynamics ever do, so the decentralised market is genuinely
sub-optimal.
"""

from _common import SEED, T, banner
import numpy as np

from marriage.affinity import make_affinity
from marriage.matching import (gale_shapley, random_matching_utility,
                               stable_matching_utility, utilitarian_optimum)
from marriage.model import simulate
from marriage.style import BLUE, LAMBDA_COLORS, ORANGE, new_fig, save

SIZES = [100, 200, 500, 1000, 2000]
banner("Fig. 13: stable and optimal matchings vs the dynamics")
fig, (ax_n, ax_side) = new_fig(ncols=2, width=4.3)

rows = {k: [] for k in ("random", "gs", "opt", "dyn_free", "dyn_wed")}
for n in SIZES:
    aff = make_affinity(n, "N", seed=SEED)
    rows["random"].append(random_matching_utility(aff, rng=np.random.default_rng(SEED)))
    rows["gs"].append(stable_matching_utility(aff))
    rows["opt"].append(utilitarian_optimum(aff))
    rows["dyn_free"].append(simulate(steps=T, lam=np.inf, affinity=aff, seed=SEED).mean_utility[-1])
    rows["dyn_wed"].append(simulate(steps=T, lam=1.0, affinity=aff, seed=SEED).mean_utility[-1])
    print(f"  N={n:5d}: random {rows['random'][-1]:+.3f}  dynamics(inf) {rows['dyn_free'][-1]:.3f}  "
          f"dynamics(L=1) {rows['dyn_wed'][-1]:.3f}  Gale-Shapley {rows['gs'][-1]:.3f}  "
          f"utilitarian {rows['opt'][-1]:.3f}")

labels = [("opt", "utilitarian optimum", "o-"), ("gs", "Gale-Shapley", "s-"),
          ("dyn_wed", r"dynamics, $\Lambda=1$", "^-"),
          ("dyn_free", r"dynamics, $\Lambda=\infty$", "v-"),
          ("random", "random pairing", "d-")]
for (key, label, fmt), color in zip(labels, LAMBDA_COLORS):
    ax_n.plot(SIZES, rows[key], fmt, color=color, label=label)
ax_n.set_xscale("log")
ax_n.set_xlabel("$N$"); ax_n.set_ylabel(f"average utility at $t={T}$")
ax_n.set_title(r"Gaussian model, $\mathcal{N}(0,1)$")
ax_n.legend(fontsize=8, loc="upper left")

# the two sides of a stable matching are not alike
n = 1000
aff = make_affinity(n, "N", seed=SEED)
left = np.arange(n // 2, dtype=np.int64)
right = np.arange(n // 2, n, dtype=np.int64)
L = np.stack([aff.pair(np.full(len(right), i), right) for i in left])
R = np.stack([aff.pair(np.full(len(left), j), left) for j in right])
m = gale_shapley(L, R)
inv = np.empty(len(m), dtype=np.int64); inv[m] = np.arange(len(m))
prop = L[np.arange(len(m)), m]
recv = R[np.arange(len(m)), inv]
bins = np.linspace(-3, 4, 61)
ax_side.hist(prop, bins=bins, color=ORANGE, alpha=0.6, edgecolor="none",
             label=f"proposers, mean {prop.mean():.2f}")
ax_side.hist(recv, bins=bins, color=BLUE, alpha=0.6, edgecolor="none",
             label=f"receivers, mean {recv.mean():.2f}")
ax_side.axvline(L.max(axis=1).mean(), color="k", ls=":", lw=1)
ax_side.annotate("best possible\nper proposer", (L.max(axis=1).mean(), 1), fontsize=7,
                 ha="center", va="bottom")
ax_side.set_xlabel("utility"); ax_side.set_ylabel("number of agents")
ax_side.set_title(rf"stable matching at $N={n}$, by side")
ax_side.legend(fontsize=8)
print(f"  N={n}: proposer side {prop.mean():.3f}, receiver side {recv.mean():.3f}, "
      f"u_GS {0.5 * (prop.mean() + recv.mean()):.3f}; best possible per proposer "
      f"{L.max(axis=1).mean():.3f} -- the paper's u_GS ~ 4 is above this")

save(fig, "fig13_benchmarks")
