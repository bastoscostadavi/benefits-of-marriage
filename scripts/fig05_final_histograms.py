"""Fig. 5 -- the full utility distribution at t = 100, not just its mean.

With marriage the distribution is shifted right and has a hard left edge: every
married agent is above Lambda by construction, and the singles pile is much
smaller.  Without marriage a large mass is still sitting at its A_kk value.
"""

from _common import N, SEED, T, banner
import numpy as np

from marriage.model import simulate_pair
from marriage.style import BLUE, ORANGE, new_fig, save

banner("Fig. 5: utility histograms at t = 100")
fig, axes = new_fig(ncols=2)
bins = np.arange(-4, 4.01, 0.1)

for ax, dist, title in zip(axes, ("N", "U"),
                           (r"Gaussian $\mathcal{N}(0,1)$", "Uniform $U(-2,2)$")):
    free, wed = simulate_pair(n=N, steps=T, dist=dist, lam=1.0, seed=SEED,
                              track_utility=True)
    ax.hist(free.utility[-1], bins=bins, color=BLUE, alpha=0.65,
            edgecolor="none", label=r"$\Lambda=\infty$")
    ax.hist(wed.utility[-1], bins=bins, color=ORANGE, alpha=0.65,
            edgecolor="none", label=r"$\Lambda=1$")
    ax.axvline(1.0, color="k", lw=0.9, ls=":")
    ax.set_xlabel("$u$")
    ax.set_ylabel("number of agents" if ax is axes[0] else "")
    ax.set_title(title)
    below = (wed.utility[-1] < 1.0).mean()
    print(f"  {dist}: with marriage, {below:.1%} of agents still below Lambda=1 at t={T}")

axes[0].legend()
save(fig, "fig05_final_histograms")
