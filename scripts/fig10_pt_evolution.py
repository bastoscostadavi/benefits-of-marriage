"""Fig. 10 -- the mean-field recursion over several steps.

Left: the exact Uniform-model densities p_0..p_4 from the symbolic recursion,
with p_2(u) against the simulated histogram.  The recursion reproduces the
paper's b_2 = 1861/5184 and r_2 = 3323/5184 exactly.

Right: the same recursion run numerically for 100 steps, for both the Uniform
and the Gaussian model (which has no closed form), compared to the simulated
mean utility.  This is the check that the N -> inf equations really are the
limit of the agent-based model.
"""

from _common import N, SEED, T, banner
import numpy as np
import sympy as sp

from marriage.analytic import numeric_steps, uniform_steps, u as u_sym, sigma as sig_sym
from marriage.model import simulate
from marriage.style import BLUE, LAMBDA_COLORS, ORANGE, new_fig, save

banner("Fig. 10: mean-field evolution")
exact = uniform_steps(4)
for t, d in enumerate(exact):
    print(f"  t={t}: b={d['b']}  r={d['r']}")
print(f"  paper: b_2 = 1861/5184 -> match {exact[2]['b'] == sp.Rational(1861, 5184)}, "
      f"r_2 = 3323/5184 -> match {exact[2]['r'] == sp.Rational(3323, 5184)}")

grid = np.linspace(-2, 2, 400)
bins = np.linspace(-2, 2, 41)
fig, axes = new_fig(ncols=3, width=3.6)

for t, d in enumerate(exact):
    f = sp.lambdify(u_sym, d["p"].subs({sig_sym: 1}), "numpy")
    axes[0].plot(grid, np.broadcast_to(f(grid), grid.shape), color=LAMBDA_COLORS[t % 6],
                 label=f"$p_{t}(u)$")
axes[0].set_xlabel("$u$"); axes[0].set_ylabel("$p_t(u)$")
axes[0].set_title("exact densities, Uniform model")
axes[0].legend(fontsize=8)

res2 = simulate(n=N, steps=2, dist="U", lam=np.inf, seed=SEED, track_utility=True)
f2 = sp.lambdify(u_sym, exact[2]["p"].subs({sig_sym: 1}), "numpy")
axes[1].hist(res2.utility[2], bins=bins, density=True, color=BLUE, alpha=0.55,
             edgecolor="none", label=f"simulation, $N=10^4$")
axes[1].plot(grid, f2(grid), color=ORANGE, label="$p_2(u)$, exact")
axes[1].set_xlabel("$u$"); axes[1].set_title("$t=2$")
axes[1].legend(fontsize=8)
print(f"  simulated b_2 = {(res2.partner >= 0).mean():.4f} vs exact "
      f"{float(exact[2]['b']):.4f}")

for dist, color, name in (("U", ORANGE, "Uniform"), ("N", BLUE, "Gaussian")):
    mf = numeric_steps(n_steps=T, dist=dist)
    sim = simulate(n=N, steps=T, dist=dist, lam=np.inf, seed=SEED)
    axes[2].plot(np.arange(T + 1), mf["mean"], color=color, label=f"{name}, mean field")
    axes[2].plot(np.arange(T + 1), sim.mean_utility, color=color, ls="none",
                 marker="o", ms=2.2, alpha=0.55, label=f"{name}, simulation")
    err = np.abs(mf["mean"] - sim.mean_utility).max()
    print(f"  {name}: max |mean field - simulation| over t<= {T} = {err:.4f}")
axes[2].set_xlabel("$t$"); axes[2].set_ylabel("$u_t$")
axes[2].set_title(r"$N\to\infty$ vs $N=10^4$")
axes[2].legend(fontsize=7.5, loc="lower right")

save(fig, "fig10_pt_evolution")
