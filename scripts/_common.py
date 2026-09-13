"""Path setup and shared defaults for the figure scripts."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The paper's baseline: N = 10,000 agents, T = 100 steps, mu = 0, sigma = 1.
N = 10_000
T = 100
SEED = 20210809          # arXiv submission date of the paper, for luck
LAMBDAS = [float("inf"), 1.5, 1.0, 0.5, 0.0, -2.0]   # the legend of Fig. 3


def lam_label(lam):
    return r"$\Lambda=\infty$" if lam == float("inf") else rf"$\Lambda={lam:g}$"


def banner(title):
    print(f"\n=== {title}")
