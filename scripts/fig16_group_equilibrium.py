"""Fig. 16 -- where self-interest settles when each type picks its own threshold.

Figure 15 showed that the welfare-optimal norm Lambda* is not a best response
once agents differ in desirability. This script asks what replaces it.

Two best-response dynamics, both run to a fixed point:

*Uniform.* The whole society sits at one Lambda. A small group deviates; we
find the deviation that pays best and move the society there. Repeat.

*Per group.* Agents are sorted into G groups by desirability, and each group
gets its own threshold. In turn, each group picks the Lambda that maximises its
own payoff, holding the others fixed. Repeat until nobody moves. This is a Nash
equilibrium in group strategies, not an invasion: a group is 1/G of the
population, so its choice really does change the environment.

The comparison that matters is welfare at the fixed point against welfare at the
uniform optimum. A planner free to tailor Lambda per group can always match or
beat the uniform optimum, since uniform is a special case. Self-interest has no
such guarantee -- each group maximises its own payoff, not the total -- so the
fixed point can, and does, land below a single number imposed on everyone.
"""

from _common import N, SEED, T, banner
import numpy as np

from marriage.affinity import make_affinity
from marriage.model import simulate
from marriage.style import BLUE, LAMBDA_COLORS, ORANGE, new_fig, save

GROUPS = 5
SEEDS = 6
GRID = np.array([0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
UNIFORM_GRID = np.arange(0.5, 3.01, 0.25)
MUT = 500
UNIFORM_OPT = 1.0
SIGMA_Q = 0.6
MAX_ROUNDS = 6

_cache = {}


def society(seed, sigma_q):
    """Affinity matrix, desirability-sorted group labels, and a mutant set."""
    key = (seed, sigma_q)
    if key not in _cache:
        sigma = float(np.sqrt(max(1e-12, 1 - sigma_q ** 2)))
        aff = make_affinity(N, "N", sigma=sigma, sigma_q=sigma_q, seed=seed)
        order = np.argsort(aff.column_means())
        label = np.empty(N, dtype=np.int64)
        for g in range(GROUPS):
            label[order[g * N // GROUPS:(g + 1) * N // GROUPS]] = g
        rnd = np.random.default_rng(seed).choice(N, MUT, replace=False)
        _cache[key] = (aff, label, rnd)
    return _cache[key]


def run(lam_vec, seed, sigma_q):
    aff, _, _ = society(seed, sigma_q)
    return simulate(steps=T, dist="N", lam=lam_vec, affinity=aff, seed=seed,
                    match_seed=seed + 7, track_utility=True)


def group_payoffs(profile, seed, sigma_q):
    """Mean utility at t = T of each group, under a per-group threshold profile."""
    aff, label, _ = society(seed, sigma_q)
    lam = np.take(profile, label)
    u = run(lam, seed, sigma_q).utility[-1]
    return np.array([u[label == g].mean() for g in range(GROUPS)]), u.mean()


def welfare_uniform(lam, seed, sigma_q):
    return run(np.full(N, lam), seed, sigma_q).mean_utility[-1]


def mse(a, axis=0):
    a = np.asarray(a)
    return a.mean(axis=axis), a.std(axis=axis, ddof=1) / np.sqrt(a.shape[axis])


# ---------------------------------------------------------------------------
banner("Fig. 16: best-response fixed points")
fig, axes = new_fig(ncols=4, width=3.2)

# --- A: the uniform best-response iteration --------------------------------
print("  uniform best-response iteration")
paths = {}
for sigma_q, color in ((0.0, BLUE), (SIGMA_Q, ORANGE)):
    w = np.array([[welfare_uniform(l, SEED + s, sigma_q) for l in UNIFORM_GRID]
                  for s in range(SEEDS)])
    wm, wse = mse(w)
    axes[0].plot(UNIFORM_GRID, wm, "-", color=color, label=rf"$\sigma_q={sigma_q:g}$")
    axes[0].fill_between(UNIFORM_GRID, wm - wse, wm + wse, color=color, alpha=0.2, lw=0)

    lam, path = UNIFORM_GRID[int(np.argmax(wm))], []
    for _ in range(MAX_ROUNDS):
        path.append(float(lam))
        pay = []
        for l in UNIFORM_GRID:
            v = []
            for s in range(SEEDS):
                aff, _, rnd = society(SEED + s, sigma_q)
                prof = np.full(N, lam)
                prof[rnd] = l
                m = np.zeros(N, bool); m[rnd] = True
                v.append(run(prof, SEED + s, sigma_q).utility[-1, m].mean())
            pay.append(np.mean(v))
        br = float(UNIFORM_GRID[int(np.argmax(pay))])
        if br == lam or br in path[:-1]:
            path.append(br)
            break
        lam = br
    paths[sigma_q] = path
    cyc = "cycle" if len(path) != len(set(path)) else "fixed point"
    print(f"    sigma_q={sigma_q:g}: optimum {path[0]:.2f} -> "
          f"{' -> '.join(f'{p:.2f}' for p in path[1:])}  ({cyc})")
    for a, b in zip(path, path[1:]):
        if a != b:
            axes[0].annotate("", xy=(b, np.interp(b, UNIFORM_GRID, wm)),
                             xytext=(a, np.interp(a, UNIFORM_GRID, wm)),
                             arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
axes[0].set_xlabel(r"society-wide $\Lambda$")
axes[0].set_ylabel(f"welfare $u_{{{T}}}$")
axes[0].set_title("uniform: optimum $\\to$ equilibrium")
axes[0].legend(fontsize=8)

# --- B/C/D: the per-group best-response iteration --------------------------
print(f"\n  per-group best-response iteration, sigma_q={SIGMA_Q:g}, {GROUPS} groups")
profile = np.full(GROUPS, UNIFORM_OPT)
history = [profile.copy()]
for rnd_i in range(MAX_ROUNDS):
    moved = False
    for g in range(GROUPS):
        pay = []
        for l in GRID:
            trial = profile.copy(); trial[g] = l
            pay.append(np.mean([group_payoffs(trial, SEED + s, SIGMA_Q)[0][g]
                                for s in range(SEEDS)]))
        br = float(GRID[int(np.argmax(pay))])
        if br != profile[g]:
            moved = True
        profile[g] = br
    print(f"    round {rnd_i + 1}: " + "  ".join(f"g{g+1}={profile[g]:.2f}"
                                                 for g in range(GROUPS)))
    if any(np.array_equal(profile, h) for h in history):
        print("    repeated a profile -> settled (or cycling)")
        break
    history.append(profile.copy())
    if not moved:
        print("    fixed point")
        break

uni = np.full(GROUPS, UNIFORM_OPT)
pay_u = np.array([group_payoffs(uni, SEED + s, SIGMA_Q) for s in range(SEEDS)], dtype=object)
pay_e = np.array([group_payoffs(profile, SEED + s, SIGMA_Q) for s in range(SEEDS)], dtype=object)
gu = np.stack([p[0] for p in pay_u]); wu = np.array([p[1] for p in pay_u])
ge = np.stack([p[0] for p in pay_e]); we = np.array([p[1] for p in pay_e])
gum, guse = mse(gu); gem, gese = mse(ge)
d = wu - we

axes[1].plot(np.arange(1, GROUPS + 1), profile, "s-", color=ORANGE, label="equilibrium")
axes[1].axhline(UNIFORM_OPT, color=BLUE, ls="--", label=f"uniform optimum")
axes[1].set_xlabel(f"desirability group ({GROUPS} = most liked)")
axes[1].set_ylabel(r"$\Lambda_d$")
axes[1].set_title("equilibrium threshold by type")
axes[1].legend(fontsize=8)

x = np.arange(1, GROUPS + 1)
axes[2].bar(x - 0.2, gum, 0.4, yerr=guse, capsize=2, color=BLUE, label="uniform $\\Lambda=1$")
axes[2].bar(x + 0.2, gem, 0.4, yerr=gese, capsize=2, color=ORANGE, label="equilibrium")
axes[2].set_xlabel(f"desirability group ({GROUPS} = most liked)")
axes[2].set_ylabel(f"payoff at $t={T}$")
axes[2].set_title("who wins, who loses")
axes[2].legend(fontsize=7.5)

names = ["uniform\noptimum", "uniform\nequilibrium", "per-group\nequilibrium"]
uw = np.array([welfare_uniform(UNIFORM_OPT, SEED + s, SIGMA_Q) for s in range(SEEDS)])
ue = np.array([welfare_uniform(paths[SIGMA_Q][-1], SEED + s, SIGMA_Q) for s in range(SEEDS)])
vals = [mse(uw), mse(ue), mse(we)]
axes[3].bar(names, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=3,
            color=[BLUE, LAMBDA_COLORS[3], ORANGE])
axes[3].set_ylabel(f"welfare $u_{{{T}}}$")
axes[3].set_ylim(min(v[0] for v in vals) * 0.96, max(v[0] for v in vals) * 1.02)
axes[3].set_title(rf"welfare, $\sigma_q={SIGMA_Q:g}$")

print(f"\n    equilibrium profile: " + ", ".join(f"{p:.2f}" for p in profile))
print(f"    welfare  uniform optimum      {uw.mean():.4f} +/- {uw.std(ddof=1)/np.sqrt(SEEDS):.4f}")
print(f"    welfare  uniform equilibrium  {ue.mean():.4f} +/- {ue.std(ddof=1)/np.sqrt(SEEDS):.4f}")
print(f"    welfare  per-group equilibrium {we.mean():.4f} +/- {we.std(ddof=1)/np.sqrt(SEEDS):.4f}")
print(f"    price of anarchy (uniform opt - group eq) = {d.mean():+.4f} "
      f"+/- {d.std(ddof=1)/np.sqrt(SEEDS):.4f}")
print("    per-group payoff change (equilibrium - uniform):")
for g in range(GROUPS):
    diff = ge[:, g] - gu[:, g]
    print(f"      group {g+1}: {diff.mean():+.4f} +/- {diff.std(ddof=1)/np.sqrt(SEEDS):.4f}")

save(fig, "fig16_group_equilibrium")
