"""Fig. 16 -- the same comparison as Fig. 3, run to T = 1000.

Fig. 3 stops at T = 100, so both the social optimum and the Nash threshold are
measured at one arbitrary horizon. This runs the same sweep ten times longer to
see whether Lambda = sigma stays on top.

Lambda = 1.25 is included because that is roughly where the iterated best
response of nash-equilibrium/ lands; without it the figure cannot say whether
the equilibrium threshold overtakes the convention at long horizons.
"""

from _common import N, SEED, banner, lam_label
import numpy as np

from marriage.model import replicate, simulate
from marriage.style import new_fig, save

T_LONG = 1000
REPLICATES = 8
LAMBDAS = [np.inf, 2.0, 1.5, 1.25, 1.0, 0.5, 0.0, -2.0]
COLORS = ["#2b6cb0", "#718096", "#805ad5", "#e53e3e", "#dd6b20",
          "#38a169", "#d53f8c", "#975a16"]

banner(f"Fig. 16: utility vs Lambda at T = {T_LONG}")
t = np.arange(T_LONG + 1)
fig, axes = new_fig(ncols=2, width=4.5)
final = {}

for ax, dist in zip(axes, ("N", "U")):
    print(f"  {dist} model")
    final[dist] = {}
    for lam, color in zip(LAMBDAS, COLORS):
        mean, err = replicate(
            lambda s, lam=lam, dist=dist: simulate(
                n=N, steps=T_LONG, dist=dist, lam=lam, seed=SEED + 100 * s),
            range(REPLICATES),
        )
        ax.plot(t, mean, color=color, label=lam_label(lam))
        ax.fill_between(t, mean - err, mean + err, color=color, alpha=0.25, lw=0)
        final[dist][lam] = (mean, err)
        print(f"    {lam_label(lam):16s} u_100 = {mean[100]:.4f}   "
              f"u_{T_LONG} = {mean[-1]:.4f} +/- {err[-1]:.4f}")
    ax.set_xscale("log")
    ax.set_xlim(1, T_LONG)
    ax.set_xlabel("$t$ (log scale)")
    ax.set_ylabel("$u_t$" if ax is axes[0] else "")
    ax.set_title(r"Gaussian $\mathcal{N}(0,1)$" if dist == "N" else "Uniform $U(-2,2)$")

axes[0].legend(ncol=2, loc="upper left", fontsize=8)
save(fig, "fig16_long_horizon")

print("\n  ranking at each horizon (best first):")
for dist in ("N", "U"):
    for horizon in (100, 300, T_LONG):
        order = sorted(LAMBDAS, key=lambda l: -final[dist][l][0][horizon])
        best = order[0]
        print(f"    {dist}, t={horizon:4d}: " +
              " > ".join(("inf" if np.isinf(l) else f"{l:g}") for l in order) +
              f"   (best {('inf' if np.isinf(best) else f'{best:g}')}"
              f" at {final[dist][best][0][horizon]:.4f})")
