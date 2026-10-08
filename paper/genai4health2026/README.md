# GenAI4Health 2026 paper package

Working directory for the NeurIPS 2026 GenAI4Health workshop submission
(Research Paper track, double-blind, 9 pages excluding references and appendix).

The canonical source is `main_submission.tex`. The September 10 revision is
titled **Where to Predict in Retinal OCT? Anatomy-Guided Target Selection for
I-JEPA**. It presents an empirical investigation, not a new validated model
or a completed causal explanation.

**Accepted** to the workshop. The latest release is v9 (v8 plus one Introduction
sentence); `main_submission.pdf` here is the validated v9 PDF. Camera-ready and
poster work continue on branch `poster-ready`.

Final PDF and validated source ZIP are in Downloads with stem
`OCT_JEPA_GenAI4Health2026_20260910_v9` (v8 remains alongside). This compact version includes an annotated
side-by-side OCT framework figure, a number-free abstract and the original
attribution motivation. Separate low-label, subgroup, fine-tuning and broad
background analyses are retained in the research record, not this submission.
The PDF has eight main-content pages
(15 total). Selected archived heatmap panels accompany the feature-occlusion
delta-logit definition, with their original numerical and clinical headings
omitted. They are illustrative, not a quantitative localization result.
The final formulation describes context and target sampling jointly and
identifies the tissue measurements as a MIRAGE-derived proxy.
The exact reviewed source and internally generated Word attachment are synchronized
to Overleaf; the PDF has not been submitted to OpenReview by this workflow.

## Central claim

Tissue-directed rectangle placement gives higher frozen-probe AUC than uniform
masking in the recorded continuations. More specific anatomical targeting does
not consistently give further gains, but guide products, target sizes and
visible context also differ. These are strategy comparisons, not controlled
ablations of anatomical coverage. The highest epoch-100 point estimate is
from `centroid`, which uses a per-column intensity-weighted row centroid
without segmentation or ground-truth anatomical labels.

One pretraining continuation per policy and repeated inspection of the same
test split limit the strength of the conclusions. An interval containing zero
does not establish equivalence. The corrected masking policies have not been
pretrained.

Two results are reported as negative or withdrawn rather than buried:

* The `cover` arm carries an implementation defect. Predictor targets are
  truncated after placement, so its delivered masks hide less anatomy than the
  `envelope` arm rather than more. Its decline is reported; the over-coverage
  interpretation is withdrawn. See `autopilot/COVER_AUDIT.md`.
* Anatomy-shaped targets use a different fixed-length target policy
  (`pred_target_k`), so their comparison with rectangles does not isolate shape.
  The early anatomy-v1 result is positive but has incomplete launch provenance.
  Anatomy-v2 has no epoch-100 result.

## Files

| path | role |
|---|---|
| `main_submission.tex` | the submission. This is the file to compile. |
| `auto/` | generated macros, tables and figures; preserve source evidence. |
| `figures/` | reviewed figures with source-bound release metadata. |
| `neurips_2026.sty` | official style file. |
| `references.bib` | bibliography. |
| `SOURCES.md` | generated provenance for every quoted quantity. |

Primary result macros are generated from retained predictions. Other numeric
literals and figures require their own evidence; macros alone do not verify
claims or protect against contradictory wording. `numeric_reviews.json` and
the release checks track the reviewed manuscript and figure inputs.

## Build

The managed Overleaf project compiles **`main.tex`**, mapped from local
`main_submission.tex` by `scripts\sync_overleaf.py`. Publish only a validated
release manifest. Remote-only historical assets are preserved; exporting the
whole Overleaf project is not equivalent to the allowlisted submission ZIP.

Locally, this project builds with Tectonic (self-contained; neither `latexmk`
nor `pdflatex` is installed on the development machine):

```powershell
& "D:\jepa_phase0\tools\tectonic\tectonic.exe" -X compile main_submission.tex --keep-intermediates
```

The `Fontconfig error: Cannot load default config file` message is benign.

## Validated release

Use `autopilot\p13_build_zip.py` with the reviewed numeric file, verified
citation record and statistics directory. It stages the exact source tree,
runs manuscript, numeric, citation, figure, PDF and Word checks, and promotes
the PDF, editable DOCX and source ZIP only after all gates pass.
Failed gates preserve the previous deliverables. Run `--help` for explicit
output and evidence arguments.

Then use `scripts\sync_overleaf.py --release-manifest PATH`. The sync checks
for collaborator changes before pushing. A standalone Tectonic compile is
useful for previewing layout, but is not a validated release.

## Adding a result

Verify the experiment's protocol and artifact identity before adding a result.
Update its inventory, numerical bindings, plots and prose together, then repeat
the release checks. New exploratory diagnostics are not automatically primary
results. The September 10 submission rewrite does not replace historical AUCs
with the subsequent probe-budget investigation.

## Anonymity

`\author{}` is empty and the ZIP validator scans the compiled PDF for
identifying terms and fails the build if any appear. Note that the wider
repository is NOT anonymous: commit authorship and some paths identify the
authors, so the repository must not be linked from the submission.
