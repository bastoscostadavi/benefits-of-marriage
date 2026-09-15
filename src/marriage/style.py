"""Shared matplotlib styling so every figure in results/ reads as one set."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results"

# Paired categorical colours; the first two are the paper's blue/orange for
# "without marriage" / "with marriage".
BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
LAMBDA_COLORS = ["#2b6cb0", "#805ad5", "#dd6b20", "#38a169", "#d53f8c", "#975a16"]

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 3.0,
})


def new_fig(ncols=1, nrows=1, width=4.1, height=3.1, **kw):
    return plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), **kw)


def save(fig, name, subdir=None):
    """Write ``name`` as a PDF.  PDF only -- see CLAUDE.md."""
    out = RESULTS if subdir is None else RESULTS / subdir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote results/{name}.pdf")
    return out / f"{name}.pdf"


def band(ax, x, mean, spread, color, label=None, alpha=0.18):
    """Mean line with a +/- one standard deviation band, the paper's style."""
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - spread, mean + spread, color=color, alpha=alpha, lw=0)
