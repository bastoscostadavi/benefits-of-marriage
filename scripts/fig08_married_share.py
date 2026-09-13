"""Fig. 8 -- the model's one contact with real data.

Left: the share of agents married at each step, for each Lambda.  Right: the
share of men in England and Wales ever married by a given age, by birth cohort.

The shapes agree without anything in the model having been fitted to the data,
which is the paper's argument that the mechanism is not unreasonable.  Read
through the model, the decline across cohorts is a society-wide rise in Lambda:
people became more selective about whom they would marry.

Caveat worth keeping in the AAMAS version: steps are not years.  The model's
curve is concave from t = 0 while the data is S-shaped, because nobody marries
before their late teens.  Any quantitative cohort-to-Lambda claim needs an
explicit step-to-age map, which the paper does not have.
"""

from _common import LAMBDAS, N, SEED, T, banner, lam_label
import csv
import pathlib

import numpy as np

from marriage.model import simulate
from marriage.style import LAMBDA_COLORS, new_fig, save

banner("Fig. 8: married share, model vs England & Wales cohorts")
fig, (ax_m, ax_d) = new_fig(ncols=2, width=4.3)

for lam, color in zip(LAMBDAS, LAMBDA_COLORS):
    if np.isinf(lam):
        continue          # no marriage: the share is identically zero
    res = simulate(n=N, steps=T, dist="N", lam=lam, seed=SEED)
    ax_m.plot(np.arange(T + 1), 100 * res.married_share, color=color, label=lam_label(lam))
    print(f"  {lam_label(lam):14s} married at t={T}: {res.married_share[-1]:.1%}")

ax_m.set_xlabel("$t$")
ax_m.set_ylabel("share married")
ax_m.set_ylim(0, 100)
ax_m.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax_m.set_title(r"model, $\mathcal{N}(0,1)$, $N=10^4$")
ax_m.legend(fontsize=8, loc="lower right")

path = pathlib.Path(__file__).resolve().parents[1] / "data" / "marriage_by_birth_cohort_england_wales.csv"
rows = list(csv.DictReader(path.open()))
cohorts = [1940, 1950, 1960, 1970, 1980]
for cohort, color in zip(cohorts, LAMBDA_COLORS[1:]):
    sel = [(int(r["age"]), float(r["share_men_married_pct"])) for r in rows
           if int(r["birth_year"]) == cohort]
    sel.sort()
    ax_d.plot([a for a, _ in sel], [s for _, s in sel], color=color, label=str(cohort))

ax_d.set_xlabel("age (years)")
ax_d.set_ylabel("share ever married")
ax_d.set_ylim(0, 100)
ax_d.set_xlim(17, 50)
ax_d.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax_d.set_title("men, England & Wales, by birth cohort")
ax_d.legend(fontsize=8, loc="lower right")

save(fig, "fig08_married_share")
