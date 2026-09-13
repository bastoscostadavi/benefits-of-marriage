"""Fig. 14 -- how many partners does an agent go through?

The cost side of the search ledger.  Without marriage an agent keeps cycling
through partners for the whole run; with marriage the count stops as soon as
the agent marries.  The paper shows this histogram but does not connect it to
the utility result -- it is the natural place to note that the benefit of
marriage is bought with a much shorter search.
"""

from _common import N, SEED, T, banner
import numpy as np

from marriage.model import simulate_pair
from marriage.style import BLUE, ORANGE, new_fig, save

banner("Fig. 14: distinct partners per agent over 100 steps")
fig, axes = new_fig(ncols=2)
bins = np.arange(-0.5, 18.5, 1.0)

for ax, dist, title in zip(axes, ("N", "U"),
                           (r"Gaussian $\mathcal{N}(0,1)$", "Uniform $U(-2,2)$")):
    free, wed = simulate_pair(n=N, steps=T, dist=dist, lam=1.0, seed=SEED,
                              track_partners=True)
    ax.hist(free.n_partners, bins=bins, color=BLUE, alpha=0.65, edgecolor="none",
            label=rf"$\Lambda=\infty$, mean {free.n_partners.mean():.2f}")
    ax.hist(wed.n_partners, bins=bins, color=ORANGE, alpha=0.65, edgecolor="none",
            label=rf"$\Lambda=1$, mean {wed.n_partners.mean():.2f}")
    ax.set_xlabel("number of distinct partners")
    ax.set_ylabel("number of agents" if ax is axes[0] else "")
    ax.set_title(title)
    ax.legend(fontsize=8)
    gain = wed.mean_utility[-1] - free.mean_utility[-1]
    saved = free.n_partners.mean() - wed.n_partners.mean()
    print(f"  {dist}: {free.n_partners.mean():.2f} -> {wed.n_partners.mean():.2f} partners "
          f"({saved:.2f} fewer) for {gain:+.3f} utility "
          f"-- {gain / saved:+.3f} utility per partner forgone")

save(fig, "fig14_number_of_partners")
