# Benefits of marriage as a search strategy — reproduction code

Python reproduction of the agent-based marriage model in
**Davi B. Costa, _Benefits of marriage as a search strategy_**
([arXiv:2108.04885](https://arxiv.org/abs/2108.04885)), rebuilt from the paper
and its original Mathematica notebook, as the basis for an AAMAS submission.

The model: `N` agents hold private, asymmetric affinities for one another. Each
step, everyone is matched at random with someone new; a pair becomes a couple
when both strictly prefer the other to their current partner, and whoever is
left behind goes back to being single. **Marriage** is the rule "stop searching
once you like your partner more than `Lambda`" — married couples drop out of the
matching forever, so `Lambda = inf` is the model with no marriage at all.

The result: average utility under marriage can beat average utility without it,
maximised around `Lambda = sigma`. Don't marry the first person you like, don't
hold out for the love of your life, but marry if you like your partner more than
a sigma above average.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_all.py      # regenerates results/, ~70s
python -m pytest tests/ -q     # 22 tests, ~2s
```

```python
import sys; sys.path.insert(0, "src")
from marriage.model import simulate_pair

free, wed = simulate_pair(n=10_000, steps=100, dist="N", lam=1.0, seed=0)
print(free.mean_utility[-1], wed.mean_utility[-1])   # 1.194  1.440
```

`simulate_pair` runs both systems on the *same* affinity matrix, which is how
the paper compares them: identical preferences, independent random matchings, so
the gap between the curves is not sampling noise in `A`.

## Layout

```
src/marriage/
  affinity.py    affinity matrices: dense, or hashed on demand (see below)
  model.py       the simulation core; mirrors DatingMarriageModel in the notebook
  analytic.py    the large-N recursion, exact (sympy) and numeric
  matching.py    Gale-Shapley and the utilitarian optimum, as benchmarks
  style.py       shared plotting style
scripts/         one script per figure, each regenerating results/figNN_*.pdf/png
data/            the England & Wales cohort marriage data used in Fig. 8
tests/           22 tests, most of them checking the paper's own numbers
paper/           the original LaTeX, figures and Mathematica notebook
FINDINGS.md      what reproduced, what did not, and what turned up
```

### Figures

| | |
| --- | --- |
| `fig01` | the model's only primitive: one agent's affinities |
| `fig02` | individual vs aggregate utility |
| `fig03` | **the main result** — average utility for each `Lambda` |
| `fig04` | `Lambda = sigma` against no marriage, mean and spread |
| `fig05` | the full utility distribution at `t = 100` |
| `fig06` | robustness: what if being single is not the same draw? |
| `fig07` | who gains from marriage — and who loses |
| `fig08` | married share, model against real cohort data |
| `fig09` | the large-`N` theory checked at the first step |
| `fig10` | the mean-field recursion over several steps |
| `fig11` | are the large-`N` equations really the large-`N` limit? |
| `fig12` | the asymptotic distribution of married couples |
| `fig13` | how far the decentralised outcome is from a planner's |
| `fig14` | how many partners an agent goes through |

## Two things this code does that the notebook could not

**It runs at `N = 10^6`.** The affinity matrix is `N x N`, which is 800 MB at
`N = 10^4` and hopeless beyond that. `LazyAffinity` never stores it: `A[i, j]` is
derived on demand from a counter-based hash of `(seed, i, j)`, so repeated
lookups of a pair always return the same number and the matrix is just as fixed
as a stored one, at `O(N)` memory. A `10^6`-agent run — a `10^12`-entry matrix —
takes about seven seconds. `DenseAffinity` materialises the matrix instead, and
the two are cross-checked in the tests.

**It derives the mean-field equations from scratch.** `analytic.py` implements
the section 3 recursion in density form rather than moment form, symbolically for
the Uniform model (where every density stays a polynomial) and numerically for
the Gaussian one, for which the paper reports no closed form. It reproduces
`b_1 = 1/4`, `p_c1`, `p_s1`, `b_2 = 1861/5184`, `p_c2` and `p_s2` exactly.

Together these settle a question the paper leaves open — whether its evolution
equations really are the `N -> inf` limit. They are not; see `FINDINGS.md`.

## What reproduces

Everything substantive. The headline result, the optimal `Lambda ~ sigma`, the
variance reduction, the robustness check, the two-sided variant, the scaling
symmetry, and every closed form in section 3 up to `t = 2`. The paper's quoted
`u_100 ~ 1.27` for the Uniform model comes out at `1.2685 ± 0.0034`.

Five things do not, and they are set out in **[FINDINGS.md](FINDINGS.md)**:

1. `E[U_inf]` (Eq. 64) is wrong except at `Lambda = -2 sigma`; the correct
   asymptotic mean is `L^2/(16 s) + L/4 + 5 s/4`, with ceiling `2 sigma` rather
   than `4 sigma`.
2. The large-`N` evolution equations are exact at `t = 1` but carry a ~8%
   systematic bias in `u_t` that does not shrink with `N`, traceable to treating
   the two members of a couple as independent.
3. "Marriage helps the less-liked more" is not detectable in the paper's own
   homogeneous model — but becomes true, and sharper, with genuine desirability
   spread, where marriage actively *hurts* the most desirable agents.
4. `u_GS ~ 4` should be `~1.8`; the planner's advantage is real but about half
   the stated size.
5. The utilitarian optimum is a linear assignment problem, not an `N!` search.

The first two are corrections; the third is arguably a better result than the
one it replaces.

## Citation

```bibtex
@article{costa2021marriage,
  title  = {Benefits of marriage as a search strategy},
  author = {Costa, Davi B.},
  year   = {2021},
  eprint = {2108.04885},
  archivePrefix = {arXiv},
  primaryClass  = {physics.soc-ph}
}
```
