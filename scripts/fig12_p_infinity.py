"""Fig. 12 -- the asymptotic distribution of married couples (section 3.3).

Married couples never break, so once an agent is married its utility is frozen.
That makes the married-couple density the same at every step, and equal to the
t -> inf distribution of the whole system.  Eq. (61) builds it directly, with no
recursion: an agent sitting below Lambda marries the first partner it likes
enough (so its final utility is p_0 conditioned on exceeding Lambda); an agent
already at x > Lambda has to clear x instead.

Two corrections to the paper come out of this panel.

1. The integral in Eq. (61) is elementary for *any* p_0, not just the uniform
   one: since p_0 = -dGbar/dx, it is just ln(Gbar(Lambda)/Gbar(u)).  So the
   Gaussian model has the same closed form, which the paper does not report.

2. Eq. (64) at n = 1 gives E[U_inf] = 3L^2/(16s) + 3L/4 + 7s/4, maximal at
   4 sigma.  Integrating u p_inf(u) actually gives L^2/(16s) + L/4 + 5s/4,
   maximal at 2 sigma.  The two agree only at L = -2 sigma -- the one value the
   paper checked.  Long simulations land on the corrected value.
"""

from _common import SEED, banner
import numpy as np

from marriage.analytic import (p_infinity, p_infinity_mean, p_infinity_mean_paper,
                               p_infinity_uniform_closed_form)
from marriage.model import simulate
from marriage.style import BLUE, LAMBDA_COLORS, ORANGE, new_fig, save

LAMS = [1.0, 0.0, -2.0]
CHECK = [-2.0, -1.0, 0.0, 1.0, 1.5]
N_LONG, T_LONG = 20_000, 800
banner("Fig. 12: asymptotic distribution of married couples")
fig, axes = new_fig(ncols=3, width=3.6)
grid = np.linspace(-2, 2 - 1e-9, 200_001)

for lam, color in zip(LAMS, LAMBDA_COLORS):
    theory = p_infinity(grid, lam, "U", sig=1.0)
    closed = p_infinity_uniform_closed_form(grid, lam, sig=1.0)
    axes[0].plot(grid, theory, color=color, label=rf"$\Lambda={lam:g}$")
    axes[0].plot(grid[::400], closed[::400], "k.", ms=2)
    print(f"  Lambda={lam:+.1f}: |Eq.61 - Eq.62|_max = {np.nanmax(np.abs(theory - closed)):.2e}, "
          f"mass = {np.trapezoid(theory, grid):.6f}")
axes[0].set_xlabel("$u$"); axes[0].set_ylabel(r"$p_\infty(u)$")
axes[0].set_ylim(0, 1.6)
axes[0].set_title("Uniform model (dots: closed form)")
axes[0].legend(fontsize=8)

# the same density, now for the Gaussian model, where the paper gives none
for lam, color in zip(LAMS, LAMBDA_COLORS):
    gg = np.linspace(-4, 6, 4001)
    axes[1].plot(gg, p_infinity(gg, lam, "N", sig=1.0), color=color,
                 label=rf"$\Lambda={lam:g}$")
    r = simulate(n=N_LONG, steps=T_LONG, dist="N", lam=lam, seed=SEED, track_utility=True)
    axes[1].hist(r.utility[-1][r.married], bins=np.linspace(-4, 6, 81), density=True,
                 color=color, alpha=0.3, edgecolor="none")
axes[1].set_xlabel("$u$"); axes[1].set_title(r"Gaussian model, $t=800$")
axes[1].legend(fontsize=8)

lam_grid = np.linspace(-2, 2, 201)
axes[2].plot(lam_grid, p_infinity_mean(lam_grid), color=ORANGE, label="corrected")
axes[2].plot(lam_grid, p_infinity_mean_paper(lam_grid), color=BLUE, ls="--",
             label="paper, Eq. (64)")
print("  Lambda   corrected   paper    simulation")
sims = []
for lam in CHECK:
    r = simulate(n=N_LONG, steps=T_LONG, dist="U", lam=lam, seed=SEED, track_utility=True)
    sim = float(r.utility[-1][r.married].mean())
    sims.append(sim)
    print(f"  {lam:+6.1f}   {p_infinity_mean(lam):9.4f}   {p_infinity_mean_paper(lam):6.4f}  "
          f"{sim:10.4f}   (married {r.married.mean():.1%})")
axes[2].plot(CHECK, sims, "ko", ms=4.5, label=rf"simulation, $t={T_LONG}$")
axes[2].set_xlabel(r"$\Lambda$"); axes[2].set_ylabel(r"$E[U_\infty]$")
axes[2].set_title("asymptotic mean utility")
axes[2].legend(fontsize=8)

save(fig, "fig12_p_infinity")
