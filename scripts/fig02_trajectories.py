"""Fig. 2 -- individual vs aggregate utility, Gaussian model, no marriage.

Left: four agents, whose utility jumps up when they find someone better and
crashes back to A_kk when they are left.  Right: the mean over all N agents
with a one-standard-deviation band.  The point is that the aggregate rises
smoothly and monotonically while no individual's path does.
"""

from _common import N, SEED, T, banner
import numpy as np

from marriage.model import simulate
from marriage.style import BLUE, new_fig, save

banner("Fig. 2: agent trajectories and average utility")
res = simulate(n=N, steps=T, dist="N", lam=np.inf, seed=SEED, track_utility=True)
t = np.arange(T + 1)

fig, (ax_k, ax_u) = new_fig(ncols=2)
for k, color in zip((0, 1, 2, 3), ["#2b6cb0", "#dd6b20", "#38a169", "#d53f8c"]):
    ax_k.step(t, res.utility[:, k], where="post", color=color, lw=1.3, alpha=0.9)
ax_k.set_xlabel("$t$")
ax_k.set_ylabel("$u_t^k$")
ax_k.set_ylim(-1.2, 2.2)
ax_k.set_title("four individual agents")

ax_u.plot(t, res.mean_utility, color=BLUE)
ax_u.fill_between(t, res.mean_utility - res.std_utility,
                  res.mean_utility + res.std_utility, color=BLUE, alpha=0.18, lw=0)
ax_u.set_xlabel("$t$")
ax_u.set_ylabel("$u_t$")
ax_u.set_ylim(-1.2, 2.2)
ax_u.set_title(r"society average, band $=\pm\sigma_t$")

print(f"  u_0 = {res.mean_utility[0]:+.4f}, u_1 = {res.mean_utility[1]:+.4f} "
      f"(large-N: 1/(2 sqrt(pi)) = {1 / (2 * np.sqrt(np.pi)):.4f}), "
      f"u_{T} = {res.mean_utility[-1]:.4f}")
save(fig, "fig02_trajectories")
