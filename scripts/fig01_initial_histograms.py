"""Fig. 1 -- how one agent's preferences are distributed.

Histogram of row k of the affinity matrix, i.e. what agent k thinks of everyone
else, in the Gaussian and Uniform models.  This is the model's only primitive:
everything downstream follows from these draws plus the matching rule.
"""

from _common import N, SEED, banner
import numpy as np

from marriage.affinity import GAUSSIAN, UNIFORM, make_affinity
from marriage.style import BLUE, ORANGE, new_fig, save

banner("Fig. 1: initial affinity histograms")
fig, axes = new_fig(ncols=2)
bins = np.arange(-4, 4.01, 0.1)

for ax, dist, color in zip(axes, (GAUSSIAN, UNIFORM), (BLUE, ORANGE)):
    aff = make_affinity(N, dist, seed=SEED)
    row = np.delete(aff.row(0), 0)
    ax.hist(row, bins=bins, color=color, alpha=0.85, edgecolor="none")
    grid = np.linspace(-4, 4, 600)
    ax.plot(grid, dist.pdf(grid) * len(row) * 0.1, color="k", lw=1.2, ls="--",
            label="exact density")
    ax.set_xlabel(r"$A_{kj}$")
    ax.set_ylabel("number of agents" if ax is axes[0] else "")
    ax.set_title(f"{dist.name} model, {dist.label()}")
    ax.legend()
    print(f"  {dist.name}: mean {row.mean():+.4f}, sd {row.std():.4f}")

save(fig, "fig01_initial_histograms")
