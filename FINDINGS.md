# Reproduction notes

What reproduced, what did not, and what turned up along the way. Every claim
below is produced by a script in `scripts/` and asserted in `tests/`.

Baseline throughout: `N = 10,000`, `T = 100`, `mu = 0`, `sigma = 1`, matching
the paper's figures.

## Reproduces

| Paper claim | Where | Result |
| --- | --- | --- |
| `u_100 ~ 1.27`, Uniform model, no marriage | fig03 | **1.2685 ± 0.0034** |
| `E[U_1] = sigma / (2 sqrt(pi))` Gaussian, `sigma/3` Uniform | fig02, fig09 | 0.2785 vs 0.2821; matches |
| `b_1 = 1/4`, `r_1 = 3/4` | fig09 | exact symbolically; 0.2450 simulated |
| `p_c1`, `p_s1`, `p_1` (Eqs. 43–45) | fig09 | exact |
| `b_2 = 1861/5184`, `r_2 = 3323/5184`, `p_c2`, `p_s2` (Eqs. 48–49) | fig10 | exact |
| Marriage at `Lambda = sigma` beats no marriage | fig03, fig04 | 1.447 vs 1.191 (Gaussian) |
| `Lambda ~ sigma` is the optimum over `{inf, 1.5, 1, 0.5, 0, -2}` | fig03 | yes, both models |
| Marrying the first person you like (`Lambda = -2`) is worst | fig03 | 0.868, well below no marriage |
| Marriage lowers the variance as well as raising the mean | fig04 | sd 0.914 → 0.645 |
| `f(mu) = u_M - u > 0` and non-decreasing, for `A_kk = 0` | fig06 | +0.001 → +0.739 over `mu ∈ [-2, 2]` |
| Two-sided ("men and women") society is indistinguishable | test_model | within 0.03 |
| Shift/rescale symmetry, Eqs. (11) and (14) | test_model | holds to 0.08 |
| Married-share curves resemble real cohort data | fig08 | shapes agree |
| A planner beats the dynamics by a wide margin | fig13 | yes, though see below |

The reproduction was written from the paper text and checked against the
original Mathematica notebook (`submissions/wolfram/anc/DatingMarriageModel.nb`), whose
`DatingMarriageModel` function the Python core mirrors step for step.

## Does not reproduce

### 1. `E[U_inf]` is wrong (Eq. 64 at `n = 1`)

The paper reports, for the Uniform model,

    E[U_inf] = 3 L^2 / (16 s) + 3 L / 4 + 7 s / 4,     max 4 sigma at L = 2 sigma.

Integrating `u p_inf(u)` over the paper's own ansatz Eq. (61) gives

    E[U_inf] = L^2 / (16 s) + L / 4 + 5 s / 4,         max 2 sigma at L = 2 sigma.

The two agree at `L = -2 sigma`, where both give `sigma` — the single value the
paper checks. Everywhere else they differ, and by a lot: at `L = sigma` the
paper gives 2.6875 against 1.5625. A simulation run to `t = 800` gives **1.5667**.
Checked at five values of `Lambda` in `fig12`; asserted in
`tests/test_analytic.py::test_corrected_asymptotic_mean`.

This matters for the interpretation. The paper reads the `4 sigma` ceiling as a
prize that is unreachable only because it takes forever. The true ceiling is
`2 sigma`, and the dynamics already reach `~1.5 sigma` by `t = 100` at
`Lambda = sigma`, so much less is being left on the table than the paper suggests.

### 2. The large-`N` equations are not the large-`N` limit

Section 3 hedges: "it may be the case that there are higher-order effects that
we are not considering." They are. Because the lazy affinity backend never
materialises `A`, the model runs at `N = 10^6` — a `10^12`-entry matrix — so
finite-`N` noise is negligible and any residual is the theory's own:

| | `t = 1` | `t = 2` | `t = 3` | `t = 100` |
| --- | --- | --- | --- | --- |
| error in `b_t` | 0.25% | 0.12% | 0.69% | — |
| error in `u_t` | 0.13% | **3.57%** | 5.29% | ~8% |

`b_2` converges to the predicted `1861/5184` as `N^{-1/2}` (so that equation is
right), but the `u_t` gap is flat in `N` from `10^3` to `10^6`. It is a
systematic bias, not a sampling artefact. See `fig11`.

The culprit is an independence assumption. Eq. (34) draws the two members of a
couple, `K` and `M`, independently from `C_t`. They are not independent: a
couple exists precisely because each cleared the other's threshold, so their
utilities are correlated from the moment it forms. Occupancy is insensitive to
that for one step; the shape of the density is not. Fixing it means tracking the
joint density of a couple rather than its marginal.

### 3. "Marriage is more beneficial to those that are not that liked"

Not detectable in the paper's own model. With `N = 10,000` i.i.d. entries, column
means are spread by only `1/sqrt(N) ~ 0.01 sigma`, so the "1% most liked" are more
liked by about `0.03 sigma`. Over six replicates the marriage premium is
`+0.219 ± 0.045` for the least liked and `+0.254 ± 0.043` for the most liked —
indistinguishable. The paper's single run at ~50 agents per group was reading noise.

The claim becomes true, and stronger than stated, once the model is given real
vertical heterogeneity (`A[i,j] = draw + q_j`, `q_j ~ N(0, sigma_q)`, holding
`Var(A_ij) = 1`):

| `sigma_q` | least liked 10% | most liked 10% |
| --- | --- | --- |
| 0.0 | +0.250 ± 0.015 | +0.281 ± 0.014 |
| 0.2 | +0.249 ± 0.014 | +0.108 ± 0.013 |
| 0.4 | +0.169 ± 0.009 | −0.051 ± 0.013 |
| 0.6 | +0.075 ± 0.008 | −0.184 ± 0.009 |
| 0.8 | −0.000 ± 0.003 | −0.251 ± 0.008 |

Past `sigma_q ~ 0.4` the premium for the most desirable agents turns **negative**:
marriage is insurance, and agents who are never at risk of being left pay for it
without needing it. The paper's "benefit of marriage" is therefore not
society-wide once agents differ in desirability — there is a distributional
conflict the paper does not mention. See `fig07`.

### 4. `u_GS ~ 4` for the Gaussian model at `N ~ 1000`

We get **1.79**. The number is structural, not a tuning difference: in a
proposer-optimal stable matching the proposing side does well (2.45 here) and
the receiving side badly (1.13), and `u_GS` averages the two. `4 sigma` is above
the mean best-possible partner for a proposer (3.02 at this `N`), so no matching
of any kind reaches it. See `fig13`.

The paper's qualitative point survives intact — a planner does far better than
the dynamics (1.79 and 2.20 against 1.19) — but the size of the gap is overstated
by roughly a factor of two, and the asymmetry between the two sides of a stable
matching is worth reporting in its own right.

### 5. The utilitarian optimum is computable

The paper calls it "computationally impossible" on the grounds that there are
`N!` matchings. It is a linear assignment problem; the Hungarian algorithm
solves it exactly in `O(N^3)`. `fig13` includes it up to `N = 2000`.

## Smaller notes

- The paper reports `b_2^exp = 1737/5000 = 0.347` differing from `1861/5184 = 0.359`
  "in the 0.01% level". That is a 3% relative difference.
- Fig. 8's model curve is concave from `t = 0` while the cohort data is S-shaped,
  because nobody marries before their late teens. Any quantitative cohort-to-`Lambda`
  mapping needs an explicit step-to-age map, which the paper does not have.
- The notebook's matching splits an odd-sized eligible set with a non-integer
  index. Here the leftover agent simply sits the step out.
