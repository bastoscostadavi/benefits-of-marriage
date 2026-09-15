"""How does the welfare-optimal threshold depend on the horizon?

The paper reports Lambda* = sigma, but measures it at T = 100. That horizon is
arbitrary, and Fig. 3 already shows the curves for different Lambda crossing, so
Lambda* cannot be horizon-independent. This measures Lambda*(T) directly.

The trick that makes it cheap: one run at a given Lambda produces u_t for every
t along the way, so a single sweep to T_MAX yields the whole family of horizons
at once. We sweep Lambda on a fine grid, average over independent societies, and
read off the argmax at each T, refining it to sub-grid resolution with a
parabola through the top three points. Error bars on Lambda*(T) come from
bootstrapping over societies.

    python best-lambda-vs-horizon/sweep.py [--dist N|U] [--seeds 12]

Writes results/sweep_<dist>.npz and results/best_lambda_<dist>.pdf/png.
"""

import argparse
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marriage.model import simulate                      # noqa: E402
from marriage.style import BLUE, ORANGE, new_fig         # noqa: E402

N = 10_000
T_MAX = 10_000
LAMBDAS = np.round(np.arange(0.40, 3.501, 0.05), 3)
HORIZONS = np.unique(np.round(np.logspace(1, 4, 40)).astype(int))
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
WORKERS = 13


def one(job):
    lam, seed, dist = job
    return simulate(n=N, steps=T_MAX, dist=dist, lam=float(lam), seed=seed).mean_utility


def peak(y, grid):
    """Argmax of y over grid, refined by a parabola through the top 3 points."""
    i = int(np.argmax(y))
    if 0 < i < len(y) - 1:
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            return float(grid[i] + 0.5 * (y0 - y2) / denom * (grid[1] - grid[0]))
    return float(grid[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dist", default="N", choices=["N", "U"])
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    jobs = [(lam, 7000 + s, args.dist) for s in range(args.seeds) for lam in LAMBDAS]
    print(f"{len(jobs)} runs of N={N:,}, T={T_MAX:,} on {WORKERS} workers")
    t0 = time.time()
    with Pool(WORKERS) as pool:
        out = pool.map(one, jobs, chunksize=1)
    print(f"  done in {time.time() - t0:.0f}s")

    # (seeds, lambdas, T_MAX+1)
    u = np.array(out).reshape(args.seeds, len(LAMBDAS), T_MAX + 1)
    mean = u.mean(axis=0)

    lam_star = np.array([peak(mean[:, T], LAMBDAS) for T in HORIZONS])
    u_star = np.array([mean[:, T].max() for T in HORIZONS])

    rng = np.random.default_rng(0)
    boot = np.empty((args.boot, len(HORIZONS)))
    for b in range(args.boot):
        idx = rng.integers(0, args.seeds, args.seeds)
        m = u[idx].mean(axis=0)
        boot[b] = [peak(m[:, T], LAMBDAS) for T in HORIZONS]
    lam_se = boot.std(axis=0, ddof=1)

    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / f"sweep_{args.dist}.npz", lambdas=LAMBDAS, horizons=HORIZONS,
             mean=mean, lam_star=lam_star, lam_se=lam_se, u_star=u_star,
             seeds=args.seeds, dist=args.dist, n=N, t_max=T_MAX)

    print(f"\n  {'T':>6}  {'Lambda*(T)':>16}  {'u_T':>8}")
    for T, l, e, uu in zip(HORIZONS, lam_star, lam_se, u_star):
        print(f"  {T:6d}   {l:6.3f} +/- {e:5.3f}     {uu:.4f}")

    # log-linear fit over the range where the grid is not clipping
    keep = (HORIZONS >= 20) & (lam_star < LAMBDAS[-1] - 0.1)
    slope, icpt = np.polyfit(np.log(HORIZONS[keep]), lam_star[keep], 1)
    print(f"\n  fit  Lambda*(T) = {icpt:.3f} + {slope:.3f} ln T   "
          f"over T in [{HORIZONS[keep].min()}, {HORIZONS[keep].max()}]")
    print(f"  at T=100 the fit gives {icpt + slope * np.log(100):.3f}")

    fig, (ax, ax2) = new_fig(ncols=2, width=4.2)
    ax.errorbar(HORIZONS, lam_star, yerr=lam_se, fmt="o-", color=ORANGE, capsize=2, ms=3)
    tt = np.logspace(np.log10(HORIZONS[keep].min()), np.log10(HORIZONS[keep].max()), 100)
    ax.plot(tt, icpt + slope * np.log(tt), "k--", lw=1,
            label=f"${icpt:.2f} + {slope:.2f}\\,\\ln T$")
    ax.axhline(1.0, color=BLUE, ls=":", lw=1)
    ax.axvline(100, color=BLUE, ls=":", lw=1)
    ax.annotate(r"the paper's $T=100$, $\Lambda^*=\sigma$", (100, 1.0), fontsize=7,
                xytext=(4, 4), textcoords="offset points", color=BLUE)
    ax.set_xscale("log")
    ax.set_xlabel("horizon $T$")
    ax.set_ylabel(r"$\Lambda^*(T)$")
    ax.set_title("the optimal threshold rises with the horizon")
    ax.legend(fontsize=8)

    ax2.plot(HORIZONS, u_star, "o-", color=ORANGE, ms=3, label=r"$u_T$ at $\Lambda^*(T)$")
    ax2.plot(HORIZONS, [mean[np.argmin(np.abs(LAMBDAS - 1.0)), T] for T in HORIZONS],
             "s--", color=BLUE, ms=3, label=r"$u_T$ at $\Lambda=\sigma$")
    ax2.set_xscale("log")
    ax2.set_xlabel("horizon $T$")
    ax2.set_ylabel(f"$u_T$")
    ax2.set_title("what the fixed convention gives up")
    ax2.legend(fontsize=8)

    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"best_lambda_{args.dist}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote {RESULTS}/best_lambda_{args.dist}.pdf")


if __name__ == "__main__":
    main()
