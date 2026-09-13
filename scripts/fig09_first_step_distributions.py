"""Fig. 9 -- the large-N theory of section 3, checked at the first step.

For the Uniform model every density stays a polynomial, and the recursion of
`marriage.analytic` returns them exactly:

    b_1 = 1/4     p_c1(u) = (2s+u)/(2s) p_0(u)
    r_1 = 3/4     p_s1(u) = (6s+u)/(6s) p_0(u)      p_1(u) = (4s+u)/(4s) p_0(u)

all matching the paper.  Plotted against the simulated histograms at N = 10,000.
"""

from _common import N, SEED, banner
import numpy as np
import sympy as sp

from marriage.analytic import u as u_sym, sigma as sig_sym, uniform_steps
from marriage.model import simulate
from marriage.style import BLUE, ORANGE, new_fig, save

banner("Fig. 9: exact distributions after one step (Uniform model)")
steps = uniform_steps(1)
subs = {sig_sym: 1}
p0 = sp.lambdify(u_sym, steps[0]["p"].subs(subs), "numpy")
p1 = sp.lambdify(u_sym, steps[1]["p"].subs(subs), "numpy")
pc1 = sp.lambdify(u_sym, steps[1]["pc"].subs(subs), "numpy")
ps1 = sp.lambdify(u_sym, steps[1]["ps"].subs(subs), "numpy")
print(f"  b_1 = {steps[1]['b']} (paper 1/4), r_1 = {steps[1]['r']} (paper 3/4)")
print(f"  p_c1 = {sp.simplify(steps[1]['pc'])}")
print(f"  p_s1 = {sp.simplify(steps[1]['ps'])}")

res = simulate(n=N, steps=1, dist="U", lam=np.inf, seed=SEED, track_utility=True)
u1 = res.utility[1]
coupled = res.partner >= 0
print(f"  simulated b_1 = {coupled.mean():.4f}, r_1 = {(~coupled).mean():.4f}")

grid = np.linspace(-2, 2, 400)
bins = np.linspace(-2, 2, 41)
fig, (ax_a, ax_b) = new_fig(ncols=2)

ax_a.hist(res.utility[0], bins=bins, density=True, color=BLUE, alpha=0.5,
          edgecolor="none", label="simulation, $t=0$")
ax_a.hist(u1, bins=bins, density=True, color=ORANGE, alpha=0.5,
          edgecolor="none", label="simulation, $t=1$")
ax_a.plot(grid, np.full_like(grid, float(p0(0.0))), color=BLUE, label="$p_0(u)$")
ax_a.plot(grid, p1(grid), color=ORANGE, label="$p_1(u)$")
ax_a.set_xlabel("$u$"); ax_a.set_ylabel("$p_t(u)$")
ax_a.set_title("all agents"); ax_a.legend(fontsize=8)

ax_b.hist(u1[coupled], bins=bins, density=True, color=ORANGE, alpha=0.5,
          edgecolor="none", label="simulation, couples")
ax_b.hist(u1[~coupled], bins=bins, density=True, color=BLUE, alpha=0.5,
          edgecolor="none", label="simulation, singles")
ax_b.plot(grid, pc1(grid), color=ORANGE, label="$p_{c1}(u)$")
ax_b.plot(grid, ps1(grid), color=BLUE, label="$p_{s1}(u)$")
ax_b.set_xlabel("$u$"); ax_b.set_title("couples and singles at $t=1$")
ax_b.legend(fontsize=8)

save(fig, "fig09_first_step_distributions")
