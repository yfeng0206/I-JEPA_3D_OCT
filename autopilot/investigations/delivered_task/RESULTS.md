# Delivered-task investigation: completed workshop track

The reviewed source release is `6f4d62e` on
`fix/jepa-delivered-task-audit`. No merge to `main` or OpenReview submission
was performed. The original release and all historical checkpoints/predictions
remain preserved.

## What explains the apparent contradiction

The broad training machinery was not inert: bounded production-size tests
showed finite online/predictor gradients and updates, no teacher gradients,
correct EMA updates, and no masked-online hidden-pixel access under fixed
preprocessing. That does not establish universal training health or a mechanism
for the AUC ranking.

The delivered masking task nevertheless differed materially from its intent.
Scoring full candidate rectangles and then truncating them changes which targets
reach the loss. Keeping tissue outside the target union does not ensure that it
survives in the final encoder context. Random fill is conditional and is neither
a guaranteed quota nor synonymous with background-only supervision.

On the strict production-size subset of the fixed Training replay:

| Quantity | Historical COVER | Exact-prefix v2 | v2 with context guard |
|---|---:|---:|---:|
| Views in full batches | 576 | 576 | 576 |
| Valid guides | 573 | 573 | 573 |
| Views losing tissue targets through truncation | 300 | 0 | 0 |
| Views with no guide-positive encoder context | 55 | 38 | 1 |
| Valid-guide misses of the defined final occupancy-cell criterion | 358 | 320 | 0 |

The guarded zero-context case has an invalid guide and is not certified.
The occupancy-cell criterion is explicitly defined and is not silently equated
with the historical soft-mass constraint. The replay uses fixed new crops in
the existing 600-view scope; these are not reconstructed historical crops.
The short 24-view tail is excluded from the table.

The target-only correction did not settle the context issue. The guard is a
separate, guide-aware intervention that can select outside the candidate
context rectangle. Old and corrected policies are not a location-only causal
comparison. No corrected-policy AUC was measured.

## Implemented changes

- Opt-in `cover_algorithm: delivered_v2` scores exact delivered prefixes.
  `cover_context_guard: true` is separately declared. Legacy masks remain
  replayable, with explicit invalid/infeasible statuses.
- Corrected multi-context rectangularization, unsafe impossible-context
  fallback, nonprefix RNG epoch behavior, and per-target source bookkeeping.
- Fixed short accumulation-window normalization and scheduler/EMA advancement
  after skipped AMP updates.
- Added versioned continuation state and explicit `exact` versus `fork` resume
  semantics. Exact replay is bounded by topology/worker conditions; old
  checkpoints cannot supply missing historical RNG/overflow state.
- Strengthened evaluation cache identity and future row-order manifests.
  Filename-derived identifiers are not claimed to establish independent patient
  identity or provide anonymization.
- Made release validation fail closed, staged outputs immutable, promotion and
  rollback conflict-aware, and Git authentication non-persistent.
- Corrected Word appendix references, headings and ordered table checks.
  Added explicit nonempty numerical coverage, retained citation-authority
  records and independent plotted-value checks.
- Repaired the trajectory plot's cross-precision null selection without
  changing the underlying AUCs, tables or macros.

Current opt-in configurations and diagnostics are described in `mask_report.md`
and `training_report.md`. The corrected multi-arm training campaign is
subsequent work requiring separate authorization.

## Paper revisions

Title: **Anatomy-Guided Masking for I-JEPA Representation Learning on Retinal OCT**.

The paper retains the positive, run-specific CENTROID/ENVELOPE results, including
the frozen-probe 0.8855 endpoint, while removing equivalence and isolated-mechanism
claims. It now explains the 2D B-scan encoder, volume pooling, target/context
relationship, separate background diagnostics, actual subgroup eligibility
rules and incomplete historical provenance.

The enlarged pipeline schematic and new source-verified token-map/precision
figures replace cramped or stale displays. Five quantitative attribution
displays lacking retained numeric inputs were excluded with an explicit
limitation; their original files and reports were not deleted. This did not
remove adverse primary AUC results.

Six further reporting errors were resolved from exact evidence: net AUC
pair-equivalents were not counts of pair-order disagreements; a geometry
agreement count was stale; subgroup workflows used different eligibility rules;
an initial audit count was unretained; a rounded spread was not an exact bound;
and local software metadata did not describe every historical training host.

## Evidence and execution boundaries

- All 43 historical prediction files had their recorded AUCs reproduced in the
  baseline audit; excluded/retracted status was preserved.
- Parent engineering integration: 149 targeted tests.
- Parent release/numeric/Word/sync integration: 160 targeted tests, plus 25
  additional literal/scatter cases and seven staging-path/input-preservation
  cases. These are separate, potentially overlapping test sets, not a summed
  unique-test claim.
- Eight initial independent one-update GPU checks and three real guided-mask
  one-update checks ran. Each reset from the ancestor. The real guided checks
  used the first two predeclared Training views; the context guard was nonbinding
  on those views. No sustained pretraining or test-set tuning ran.
- The final numeric gate has no unresolved entries: eight programmatically
  checked plots and two explicitly source-reviewed illustrations. Protocol,
  formula and citation reviews are distinguished from mathematical verification.
- Independent critics found additional bugs in the proposed fixes; these were
  repaired rather than dismissed. Generated reviews are author aids, not venue
  decisions or acceptance predictions.

## Delivered version

The actual release build passed all required gates and produced:

- PDF: nine body pages, 34 total pages.
- Word: 129 references, 17 data tables, 11 figures/images, 45 bibliography entries.
- Validated source ZIP: 24 members, with no withdrawn composites, private
  fixtures or full third-party papers.

Overleaf was updated from the exact release manifest, then independently
re-cloned in a dry run: all 23 managed source/Word items matched, with nothing
left to push. Historical remote-only files were preserved. **A whole Overleaf
project export is therefore not the certified anonymous source ZIP.**

Downloads:

- `OCT_JEPA_GenAI4Health2026_FINAL.zip`
- `OCT_JEPA_GenAI4Health2026_6f4d62e.pdf`
- `OCT_JEPA_GenAI4Health2026_editable.docx`
- `OCT_JEPA_GenAI4Health2026_FINAL.release.json`
- `OCT_JEPA_GenAI4Health2026_6f4d62e_DELIVERY.json`

Use the validated PDF/ZIP for submission. The original loose `_files` mirror
belongs to the older workflow and is not updated or certified by this release.

Full source papers used for citation review remain local-only; public acquisition
manifests retain their URLs/hashes. No raw OCT volumes, real-image fixture tensors,
private case maps, or credentials were newly committed by this investigation.

## Remaining author decisions

Submit through OpenReview and confirm the author list before the stated deadline:
September 6, 2026, 04:59 PDT. Merging the branch, changing repository visibility,
rotating the previously exposed Overleaf token, and launching sustained
corrected-model training are separate actions, not performed here.
