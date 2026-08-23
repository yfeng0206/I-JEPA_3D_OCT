# Built paper bundles

Download these directly rather than rebuilding.

## `OCT_JEPA_GenAI4Health2026_Overleaf.zip` - CURRENT

GenAI4Health @ NeurIPS 2026, Research Paper track. Rebuilt 2026-08-23.

**For colleague review, not for submission.** Three placeholders remain and
render in red with a dagger, so what is unfinished is visible on the page rather
than hidden:

| placeholder | why |
|---|---|
| `cover` epoch 75 AUC | that arm was deliberately halted at epoch 73 |
| `cover` epoch 100 AUC | same |
| fp32 re-probe appendix table | probes still running at build time |

### Validation at build (all pass)

| check | result |
|---|---|
| compiles standalone from a clean extract | pass |
| main content within the 9-page limit | pass, exactly 9 |
| no undefined citations or references | pass |
| anonymous (double-blind) | pass |
| all referenced graphics present | pass |

Total 19 pages: 9 main, references from page 10, then appendices.
11.3 MB, well under OpenReview's 50 MB single-PDF limit.

### How to use

Upload to Overleaf, set `main.tex` as the root document, compile with XeLaTeX
and BibTeX.

### What changed in this rebuild

Every quantity in the manuscript is now emitted as a LaTeX macro by
`autopilot/p8_make_assets.py` from stored per-case predictions
(`auto/auto_numbers.tex`, 185 macros). Nothing is typed by hand, so prose,
tables and figures cannot drift apart.

Substantive corrections applied after an independent numerical audit and a
simulated review panel:

- the claim that the racial AUC gap widens as models improve was **withdrawn**:
  it fails Benjamini-Hochberg correction across the seven attributes tested and
  disappears entirely when pseudo-replication is removed by aggregating within
  training branches
- the claim that better masking "did not help early disease" was **corrected**:
  mild disease improves most of all three severity strata
- the subgroup analysis had silently re-admitted two probes the paper declares
  excluded; it now joins the authoritative inventory (21 probes to 19)
- a table that mixed epoch-100 and epoch-50 AUCs in one column was rebuilt at a
  single matched epoch
- the `oracle` arm was renamed `intensity`, because it uses no ground truth and
  the old name implied an upper bound
- fp16/fp32 probe precision is now disclosed, partitioned, and measured rather
  than assumed uniform

### Provenance

`paper/genai4health2026/SOURCES.md` documents the origin of every figure, table
and number, including which runs are excluded or retracted and why.

## `OCT_JEPA_3D_CVPR_2027.zip`

Earlier, separate submission bundle. Unrelated to the above.

## `archive/`

Superseded bundles kept for reference.
