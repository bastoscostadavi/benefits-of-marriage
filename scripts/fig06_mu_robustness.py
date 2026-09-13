"""Fig. 6 -- does the benefit survive when being single is not the same draw?

Section 2.3's robustness check.  Fix A_kk = 0 for everyone (being single is
worth exactly nothing) and let A_ij ~ N(mu, 1), sweeping mu: negative mu means
most people are worse than solitude, positive mu means almost anyone beats it.
Lambda tracks the distribution at mu + sigma.

The gap f(mu) = u_marriage - u_no_marriage stays positive throughout and widens
with mu, so the result is not an artefact of the two distributions coinciding.
"""

from _common import N, SEED, T, banner
import numpy as np

from marriage.model import simulate_pair
from marriage.style import BLUE, ORANGE, band, new_fig, save

MUS = [-2.0, -1.0, 0.0, 1.0, 2.0]
banner("Fig. 6: robustness to the single-utility distribution (A_kk = 0)")
t = np.arange(T + 1)
fig, axes = new_fig(ncols=len(MUS), width=2.6, height=2.7, sharey=True)
gaps = []

for ax, mu in zip(axes, MUS):
    free, wed = simulate_pair(n=N, steps=T, dist="N", lam=mu + 1.0, seed=SEED,
                              mu=mu, sigma=1.0, mu_s=0.0, sigma_s=0.0)
    band(ax, t, free.mean_utility, free.std_utility, BLUE, r"$\Lambda=\infty$")
    band(ax, t, wed.mean_utility, wed.std_utility, ORANGE, rf"$\Lambda={mu + 1:g}$")
    ax.set_title(rf"$\mu={mu:g}$")
    ax.set_xlabel("$t$")
    ax.legend(fontsize=7.5, loc="lower right")
    gap = wed.mean_utility[-1] - free.mean_utility[-1]
    gaps.append(gap)
    print(f"  mu={mu:+.0f}: u_no_marriage={free.mean_utility[-1]:.3f}  "
          f"u_marriage={wed.mean_utility[-1]:.3f}  f(mu)={gap:+.3f}")

axes[0].set_ylabel("$u_t$")
save(fig, "fig06_mu_robustness")

print(f"  f(mu) positive everywhere: {all(g > 0 for g in gaps)}; "
      f"non-decreasing in mu: {all(b >= a - 1e-9 for a, b in zip(gaps, gaps[1:]))}")

fig2, ax = new_fig(width=4.3)
ax.plot(MUS, gaps, "o-", color=ORANGE)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$f(\mu)=u_M(\mu)-u(\mu)$")
ax.set_title(r"marriage premium at $t=100$, $\Lambda=\mu+\sigma$")
save(fig2, "fig06b_marriage_premium")
