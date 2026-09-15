# Working notes for Claude sessions

## Figures: PDF only

Write figures as **PDF and nothing else**. No PNG, no SVG, no duplicate
rasterised copy alongside the vector one.

The papers are LaTeX, so PDF is the only format that gets used; a PNG beside
every PDF doubles the file count in `results/` for nothing and bloats the repo.

Use `marriage.style.save(fig, name)`, which writes a single PDF to `results/`.
If a script writes its own files (as the self-contained experiment folders do),
call `fig.savefig(path.with_suffix(".pdf"))` once and stop there.

If you need to *look* at a figure yourself, render it to a temp directory
outside the repo (`pdftoppm -png -r 60 fig.pdf /tmp/...`) rather than committing
a PNG.

## Experiment folders

Each experiment that is more than a single figure lives in its own top-level
folder, self-contained, named for the **question it asks**, with dashes and not
underscores: `nash-equilibrium/`, `best-lambda-vs-horizon/`,
`desirability-nash-equilibrium/`. Each carries its own `README.md` (the question
and the method) and `FINDINGS.md` (the answer, with numbers).

## Submissions

`submissions/<venue>/` must build standalone: sources, `figures/`, class files
and bibliography all present, no path reaching outside the directory. Do not add
scripts that copy figures in from `results/` -- commit the figure files
themselves.

`submissions/wolfram/` is the original paper and its Mathematica notebook. It is
a historical record: do not regenerate its figures or edit its sources.

## Numbers in the paper

Every number quoted in a paper must come from a committed script, with an error
bar, averaged over independent societies -- never from a single run. The unit of
replication is the society, never the agent.
