# AAMAS draft — working notes

`marriage.tex` is the paper ported into the AAMAS 2026 template. It compiles to
13 pages, which is over the 8-page main-track limit; cutting comes later.

```bash
make            # build marriage.pdf
make figures    # re-sync figures/ from ../../results/ after rerunning the sims
make zip        # anonymous source zip for OpenReview
```

## Template

Official AAMAS 2026 package, downloaded from
<https://cyprusconferences.org/aamas2026/wp-content/uploads/2025/06/AAMAS-2026-Formatting-Instructions-pre-CR.zip>
and unmodified. `aamas.cls` is acmart with the IFAAMAS copyright block; the
untouched sample and its bibliography are kept in `template/` for reference.
The draft is in `anonymous` mode — switch the `\documentclass` line for
camera-ready. One local addition: `\let\Bbbk\relax` before `amssymb`, which
otherwise clashes with the Libertine math font the class mandates.

Submission requirements: 8 pages plus unlimited references, double-blind, no
pointers to external supplementary material (so the code repository cannot be
linked in the submitted version — it goes in the supplementary zip instead,
which must itself be anonymous).

## Changes from the original paper

Marked in the source with `>>> CHANGED`; `grep -n '>>> CHANGED' marriage.tex`
finds all six. Each is a claim the reproduction could not confirm, documented in
`../../FINDINGS.md`.

| § | What changed |
| --- | --- |
| 5.5 Who gains | The original claimed marriage helps the less-liked more, from a single run at ~50 agents per group. At `N = 10^4` the two premia are statistically indistinguishable. Replaced with the `sigma_q` sweep, where the claim holds and strengthens: past `sigma_q ~ 0.4` the most desirable agents are made strictly worse off. |
| 6 Benchmarks | `u_GS ~ 4` → `1.79`; added the proposer/receiver asymmetry that explains the discrepancy. |
| 6 Benchmarks | The utilitarian optimum is a linear assignment problem, `O(N^3)`, not an `N!` search. |
| 7.4 Validity | New subsection. The original left the validity of the large-`N` equations open; runs at `N = 10^6` show they are exact at `t = 1` but carry a ~8% bias in `u_t` thereafter that does not vanish with `N`. |
| 7.5 `p_inf` | The integral in Eq. (61) is elementary for any `p_0`, which extends the closed form to the Gaussian model. |
| 7.5 `p_inf` | `E[U_inf]` corrected from `3L²/16s + 3L/4 + 7s/4` to `L²/16s + L/4 + 5s/4`. Ceiling `2σ`, not `4σ`. Confirmed by simulation at five values of `Λ`. |

Other edits that are not corrections: the matching bijection is `π_t` rather
than `μ_t`, which clashed with the mean `μ`; Related Work is its own section;
a contributions list was added to the introduction; every number is now a mean
over replicates with a standard error.

## Open questions for the rewrite

1. **What is the AAMAS pitch?** As it stands the paper is a social-physics
   model. The multi-agent-systems angle that seems strongest is
   Section 5.5: a society-level convention `Λ` that benefits most agents but
   which the most desirable agents have an incentive to defect from. That is a
   mechanism-design question, and it is unanswered.
2. **Cut to 8 pages.** The large-`N` material (Section 7) is roughly half the
   paper and the least AAMAS-shaped part. Candidate: keep 7.4 and 7.5, move the
   recursion derivation to an appendix.
3. **Endogenous `Λ`.** Agents currently receive `Λ` exogenously. Letting them
   learn or best-respond to it is the obvious next model, and would connect to
   the AAMAS literature directly.
4. **The step-to-age map** (Section 8) is the gap between the model and the
   data. Fitting it would turn Figure 8 from a visual analogy into a result.
