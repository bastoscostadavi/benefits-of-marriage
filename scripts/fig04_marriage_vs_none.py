"""Fig. 4 -- with marriage (Lambda = sigma) against without, mean and spread.

Same affinity matrix for both systems, so the gap is not a sampling artefact.
Marriage raises the mean *and* lowers the variance: if agents care about
equality of outcomes, that is a second benefit on top of the first.
"""

from _common import N, SEED, T, banner
import numpy as np

from marriage.model import simulate_pair
from marriage.style import BLUE, ORANGE, band, new_fig, save

banner("Fig. 4: marriage (Lambda=1) vs no marriage")
t = np.arange(T + 1)
fig, axes = new_fig(ncols=2)

for ax, dist, title in zip(axes, ("N", "U"),
                           (r"Gaussian $\mathcal{N}(0,1)$", "Uniform $U(-2,2)$")):
    free, wed = simulate_pair(n=N, steps=T, dist=dist, lam=1.0, seed=SEED)
    band(ax, t, free.mean_utility, free.std_utility, BLUE, r"$\Lambda=\infty$")
    band(ax, t, wed.mean_utility, wed.std_utility, ORANGE, r"$\Lambda=1$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$u_t$" if ax is axes[0] else "")
    ax.set_ylim(-1.2, 2.2)
    ax.set_title(title)
    print(f"  {dist}: u_{T} = {free.mean_utility[-1]:.4f} (sd {free.std_utility[-1]:.4f}) "
          f"-> {wed.mean_utility[-1]:.4f} (sd {wed.std_utility[-1]:.4f}); "
          f"married share {wed.married_share[-1]:.1%}")

axes[0].legend(loc="lower right")
save(fig, "fig04_marriage_vs_none")
