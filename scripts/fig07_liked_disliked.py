"""Fig. 7 -- who benefits from marriage?  (A claim in the paper that needs care.)

Section 2.3 argues that marriage helps the less-liked agents more, on the
grounds that at finite N the society is not exactly homogeneous: every agent
gets a slightly different column of A.

Left panel reproduces that setup and finds *no* effect.  The reason is a matter
of scale: with N = 10,000 i.i.d. entries, column means are spread by only
1/sqrt(N) ~ 0.01 sigma, so "the 1% most liked" are more liked by about
0.03 sigma -- far too little to move a curve.  The paper's single run at ~50
agents per group was reading sampling noise.

Right panels give the claim a fair test by adding genuine vertical
heterogeneity: A[i,j] = draw + q_j with q_j ~ N(0, sigma_q), holding the total
variance of A[i,j] fixed at 1.  Now the claim is not just true but stronger
than stated -- past sigma_q ~ 0.4 the marriage premium for the most desirable
agents turns *negative*.  Marriage is insurance, and the agents who are never
at risk of being left pay for it without needing it.
"""

from _common import N, SEED, T, banner
import numpy as np

from marriage.model import simulate_pair
from marriage.style import BLUE, ORANGE, band, new_fig, save

REPLICATES = 6
PCT = 10
SIGMA_QS = [0.0, 0.2, 0.4, 0.6, 0.8]
banner(f"Fig. 7: marriage premium for the most vs least liked {PCT}%")


def premiums(sigma_q, seed):
    """(least-liked gain, most-liked gain) at t = T, total Var(A_ij) = 1."""
    sigma = float(np.sqrt(max(1e-12, 1.0 - sigma_q ** 2)))
    free, wed = simulate_pair(n=N, steps=T, dist="N", lam=1.0, seed=seed,
                              sigma=sigma, sigma_q=sigma_q, track_utility=True)
    order = np.argsort(free.affinity.column_means())
    k = N * PCT // 100
    out = []
    for idx in (order[:k], order[-k:]):
        out.append(wed.utility[-1, idx].mean() - free.utility[-1, idx].mean())
    return out, free, wed, order, k


# --- panel A: the paper's own homogeneous model, with error bars -------------
fig, axes = new_fig(ncols=3, width=3.6)
_, free, wed, order, k = premiums(0.0, SEED)
t = np.arange(T + 1)
for idx, name, ls in ((order[:k], f"least liked {PCT}%", "-"),
                      (order[-k:], f"most liked {PCT}%", "--")):
    axes[0].plot(t, free.utility[:, idx].mean(axis=1), color=BLUE, ls=ls, label=f"{name}, $\\Lambda=\\infty$")
    axes[0].plot(t, wed.utility[:, idx].mean(axis=1), color=ORANGE, ls=ls, label=f"{name}, $\\Lambda=1$")
axes[0].set_title(r"paper's model ($\sigma_q=0$)")
axes[0].set_xlabel("$t$")
axes[0].set_ylabel("$u_t$")
axes[0].legend(fontsize=6.5, loc="lower right")

# --- panel B: same split, but with real desirability spread ------------------
_, free_h, wed_h, order_h, k_h = premiums(0.6, SEED)
for idx, name, ls in ((order_h[:k_h], f"least liked {PCT}%", "-"),
                      (order_h[-k_h:], f"most liked {PCT}%", "--")):
    axes[1].plot(t, free_h.utility[:, idx].mean(axis=1), color=BLUE, ls=ls)
    axes[1].plot(t, wed_h.utility[:, idx].mean(axis=1), color=ORANGE, ls=ls)
axes[1].set_title(r"with desirability spread ($\sigma_q=0.6$)")
axes[1].set_xlabel("$t$")

# --- panel C: the premium as a function of sigma_q --------------------------
lo_m, lo_e, hi_m, hi_e = [], [], [], []
for sq in SIGMA_QS:
    vals = np.array([premiums(sq, SEED + 31 * s)[0] for s in range(REPLICATES)])
    m, e = vals.mean(axis=0), vals.std(axis=0, ddof=1) / np.sqrt(REPLICATES)
    lo_m.append(m[0]); lo_e.append(e[0]); hi_m.append(m[1]); hi_e.append(e[1])
    print(f"  sigma_q={sq:.1f}: least liked {m[0]:+.3f}+/-{e[0]:.3f}   "
          f"most liked {m[1]:+.3f}+/-{e[1]:.3f}")

axes[2].errorbar(SIGMA_QS, lo_m, yerr=lo_e, fmt="o-", color=BLUE, capsize=2,
                 label=f"least liked {PCT}%")
axes[2].errorbar(SIGMA_QS, hi_m, yerr=hi_e, fmt="s--", color=ORANGE, capsize=2,
                 label=f"most liked {PCT}%")
axes[2].axhline(0, color="k", lw=0.8)
axes[2].set_xlabel(r"desirability spread $\sigma_q$")
axes[2].set_ylabel(r"marriage premium at $t=100$")
axes[2].set_title("who gains from marriage")
axes[2].legend(fontsize=8)

save(fig, "fig07_liked_disliked")
