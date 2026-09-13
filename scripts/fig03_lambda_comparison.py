"""Fig. 3 -- the paper's main result: average utility for different Lambda.

Marry too readily (Lambda = -2, i.e. below almost every affinity) and you lock
in the first person you like: the curve saturates low.  Never marry
(Lambda = inf) and you keep searching but keep getting left: the curve rises
slowly.  Around Lambda ~ sigma the two effects trade off and the society beats
both extremes -- "get married if you like your partner more than a sigma above
average".

Averaged over independent societies, with standard-error bands, which the
paper's single-run figures did not have.
"""

from _common import LAMBDAS, N, SEED, T, banner, lam_label
import numpy as np

from marriage.model import replicate, simulate
from marriage.style import LAMBDA_COLORS, new_fig, save

REPLICATES = 8
banner(f"Fig. 3: utility vs Lambda ({REPLICATES} replicates of N={N})")
t = np.arange(T + 1)
fig, axes = new_fig(ncols=2, width=4.5)

for ax, dist in zip(axes, ("N", "U")):
    print(f"  {dist} model")
    for lam, color in zip(LAMBDAS, LAMBDA_COLORS):
        mean, err = replicate(
            lambda s, lam=lam, dist=dist: simulate(
                n=N, steps=T, dist=dist, lam=lam, seed=SEED + 100 * s),
            range(REPLICATES),
        )
        ax.plot(t, mean, color=color, label=lam_label(lam))
        ax.fill_between(t, mean - err, mean + err, color=color, alpha=0.25, lw=0)
        print(f"    {lam_label(lam):16s} u_{T} = {mean[-1]:.4f} +/- {err[-1]:.4f}")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$u_t$" if ax is axes[0] else "")
    ax.set_title("Gaussian model $\\mathcal{N}(0,1)$" if dist == "N" else "Uniform model $U(-2,2)$")

axes[0].legend(ncol=2, loc="lower right", fontsize=8.5)
save(fig, "fig03_lambda_comparison")
