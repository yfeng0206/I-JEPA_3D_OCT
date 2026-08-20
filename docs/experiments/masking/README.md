# Masking experiment documentation

Start here if you are preparing the CVPR/NeurIPS-workshop submission. Generic MIRAGE-use wording is avoided because it hides the key distinction: `mirage_envelope` uses MIRAGE to place rectangles, while `mirage_anatomy` uses MIRAGE to shape connected anatomy targets.

## Recommended reading order

1. [`method_setup.md`](method_setup.md) — masking modes, sampler pipeline, config pointers, and historical method context.
2. [`comparison.md`](comparison.md) — arm-vs-arm evidence, including the ep30 AUC result and mask-budget decomposition.
3. [`sampler_ablations.md`](sampler_ablations.md) — mass-cap, collation, coverage, region-growth, integration, and rejected sampler designs.
4. [`adapter_ablations.md`](adapter_ablations.md) — cfg-7 sweep, guardrails, saturation, and AMP-vs-fp32 guide generation.
5. [`crop_and_precision_audit.md`](crop_and_precision_audit.md) — audit of the `amp_target` teacher-precision and `enc_truncate: window` confounds that retract the COVER-then-RANDOM campaign result.

## Method

- [`method_setup.md`](method_setup.md) — defines `random_default`, `random_matched`, `mirage_envelope`, and `mirage_anatomy`; states that shape is the contribution.

## Comparison

- [`comparison.md`](comparison.md) — consolidated comparison page for AUC, mask budgets, one-epoch diagnostics, and legacy rectangle-placement results.
- [`anatomy_vs_rectangle_ep30.md`](anatomy_vs_rectangle_ep30.md) — focused ep30 head-to-head between MIRAGE-placed rectangles and anatomy-shaped targets.

## Ablations

- [`sampler_ablations.md`](sampler_ablations.md) — sampler knobs and failure modes.
- [`adapter_ablations.md`](adapter_ablations.md) — adapter sweep and guide-generation ablations.
- [`structural_loss.md`](structural_loss.md) — structural-loss objective investigation.

## Adapter investigation

- [`adapter_placement.md`](adapter_placement.md) — encoder/mid/H0 placement ablation.
- [`class_relations.md`](class_relations.md) — MIRAGE-vs-I-JEPA tissue-class relationship probe.

## Confound audit

- [`crop_and_precision_audit.md`](crop_and_precision_audit.md) — `amp_target` teacher-precision defect, the `enc_truncate: window` confound, a retraction of the "crop is a local bug" claim, oracle-fallback feasibility, and the resulting retraction of the COVER-then-RANDOM campaign result.
- [`cover_random_campaign.md`](cover_random_campaign.md) — the COVER-then-RANDOM run plan and completed result, now flagged non-comparable; see its RESULTS AND RETRACTION section.

## Compatibility stubs

- [`ablations.md`](ablations.md) — old mixed ablation record, now an index to focused pages.
- [`findings.md`](findings.md) — old mixed findings record, now an index to focused pages.
