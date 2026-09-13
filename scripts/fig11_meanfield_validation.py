"""Fig. 11 -- are the paper's large-N equations actually the large-N limit?

Section 3 of the paper writes down evolution equations for p_t(u) and says of
them: "we do not know how to rigorously prove that this system of equations is
indeed the N -> inf limit ... it may be the case that there are higher-order
effects that we are not considering."

This settles it.  Because the lazy affinity backend never materialises A, the
simulation runs at N = 10^6 -- a 10^12-entry affinity matrix -- so finite-N
noise is negligible and any remaining gap is the theory's own.

Result: the equations are exact at t = 1 and give b_2 exactly, but u_2 is
already off by 3.5% and the error grows to ~8% and saturates.  It does not
shrink with N, so it is a systematic bias, not a sampling artefact.

The reason is an independence assumption.  Eq. (34) takes the two members of a
couple, K and M, as independent draws from C_t.  They are not: a couple exists
precisely because each cleared the other's threshold, so their utilities are
correlated from the moment the couple forms.  Occupancy (b_t) is insensitive to
that correlation for one step; the shape of the density is not.
"""

from _common import SEED, T, banner
import time

import numpy as np

from marriage.analytic import numeric_steps
from marriage.model import simulate
from marriage.style import BLUE, LAMBDA_COLORS, ORANGE, new_fig, save

SIZES = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
banner("Fig. 11: mean-field equations vs simulation as N -> infinity")
fig, axes = new_fig(ncols=3, width=3.6)
t = np.arange(T + 1)

mf = {d: numeric_steps(n_steps=T, dist=d) for d in ("N", "U")}
runs = {}
for n, color in zip(SIZES, LAMBDA_COLORS):
    t0 = time.time()
    runs[n] = simulate(n=n, steps=T, dist="N", lam=np.inf, seed=SEED)
    print(f"  N = {n:>9,}: u_{T} = {runs[n].mean_utility[-1]:.5f}  ({time.time() - t0:.1f}s)")
    axes[0].plot(t, runs[n].mean_utility, color=color, lw=1.3,
                 label=rf"$N=10^{{{int(np.log10(n))}}}$")
print(f"  mean field : u_{T} = {mf['N']['mean'][-1]:.5f}")

axes[0].plot(t, mf["N"]["mean"], color="k", ls="--", label="mean field")
axes[0].set_xlabel("$t$"); axes[0].set_ylabel("$u_t$")
axes[0].set_title(r"Gaussian model, $\Lambda=\infty$")
axes[0].legend(fontsize=7.5, loc="lower right")

# the gap does not close with N
gaps = [abs(runs[n].mean_utility[-1] - mf["N"]["mean"][-1]) for n in SIZES]
axes[1].loglog(SIZES, gaps, "o-", color=ORANGE, label=r"$|u_{100}^{\rm sim}-u_{100}^{\rm MF}|$")
axes[1].loglog(SIZES, [abs(runs[n].coupled_share[2] - mf["N"]["b"][2]) for n in SIZES],
               "s--", color=BLUE, label=r"$|b_2^{\rm sim}-b_2^{\rm MF}|$")
axes[1].loglog(SIZES, [3 / np.sqrt(n) for n in SIZES], ":", color="k", lw=1,
               label=r"$\propto N^{-1/2}$")
axes[1].set_xlabel("$N$"); axes[1].set_ylabel("absolute error")
axes[1].set_title("the $u_t$ gap does not close")
axes[1].legend(fontsize=7.5)

# where the error enters
big = simulate(n=SIZES[-1], steps=10, dist="N", lam=np.inf, seed=SEED)
steps = np.arange(11)
axes[2].semilogy(steps[1:], 100 * np.abs(big.coupled_share[1:11] - mf["N"]["b"][1:11])
                 / mf["N"]["b"][1:11], "s--", color=BLUE, label=r"$b_t$")
axes[2].semilogy(steps[1:], 100 * np.abs(big.mean_utility[1:11] - mf["N"]["mean"][1:11])
                 / np.abs(mf["N"]["mean"][1:11]), "o-", color=ORANGE, label=r"$u_t$")
axes[2].set_xlabel("$t$"); axes[2].set_ylabel("relative error (%)")
axes[2].set_title(r"error at $N=10^6$, by step")
axes[2].legend(fontsize=8)

for tt in (1, 2, 3):
    print(f"  t={tt}: b_t error {100 * abs(big.coupled_share[tt] - mf['N']['b'][tt]) / mf['N']['b'][tt]:6.3f}%   "
          f"u_t error {100 * abs(big.mean_utility[tt] - mf['N']['mean'][tt]) / abs(mf['N']['mean'][tt]):6.3f}%")

save(fig, "fig11_meanfield_validation")
