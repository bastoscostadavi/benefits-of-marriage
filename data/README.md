# Data

## `marriage_by_birth_cohort_england_wales.csv`

Share of men in England and Wales who had ever married, by age, for each
birth cohort from 1900 to 1990.

| column | meaning |
| --- | --- |
| `birth_year` | birth cohort (1900–1990, decadal) |
| `age` | age in years |
| `share_men_married_pct` | percent of that cohort married by that age |

Source: Our World in Data, "Marriages and Divorces"
(<https://ourworldindata.org/marriages-and-divorces>), after the UK Office for
National Statistics. Transcribed from the paper's Mathematica notebook
(`paper/anc/DatingMarriageModel.nb`, cell `In[32]`), which is the version the
published figure was drawn from.

Cohorts are truncated at the age they had reached when the data was compiled,
so the 1970, 1980 and 1990 series are shorter than the earlier ones.
