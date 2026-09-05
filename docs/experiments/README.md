# Experiments

This directory is the index for self-supervised pretraining, downstream evaluation, masking-method ablations, and interpretability work for the OCT I-JEPA project.

## Current study and historical method statements

The current workshop study examines **anatomy-guided target selection and the
task actually delivered to I-JEPA**. `mirage_envelope` places rectangular targets
using a segmenter; `anatomical_prior` is the segmentation-free CENTROID policy
(historically `oracle`); `mirage_anatomy` constructs anatomy-shaped targets.
Target shape is not isolated from context, area, collation and historical
provenance in the existing comparisons. See the root `VERSION_BOARD.md` and
`autopilot\investigations\delivered_task\BOARD.md`.

The older documents below preserve early sampler and repeated-probe summaries.
They do not establish that shape alone caused the epoch-30 AUC difference, and
probe-seed repeats are not independent pretraining runs. Use the current
manuscript's matched prediction inventory for reported comparisons, not the
older narrative that anatomy-shaped masking was already an isolated improvement.

## Recommended reading order

1. [`masking/`](masking/) — current masking-method record, including sampler design, ablations, guardrails, and findings.
2. [`mirage_guided_masking.md`](mirage_guided_masking.md) — disambiguation page for `mirage_envelope` versus `mirage_anatomy`.
3. [`pretraining/`](pretraining/) — self-supervised run records and pretraining diagnostics.
4. [`frozen/`](frozen/) — frozen-encoder probe evaluations.
5. [`finetune/`](finetune/) — unfrozen encoder fine-tuning evaluations.
6. [`interpretability.md`](interpretability.md) — occlusion attribution results used to check what downstream probes use.
7. [`curriculum_masking.md`](curriculum_masking.md) — historical oracle → self-guided → curriculum plan; use it for context, not as the current implementation record.

## Directory map

| Path | Contents |
|---|---|
| [`masking/`](masking/) | Current masking documentation. This is the active home for anatomy-shaped masking, rectangle-envelope baselines, sampler ablations, adapter ablations, findings, and reproduction notes. |
| [`pretraining/`](pretraining/) | Self-supervised I-JEPA pretraining runs, including random, oracle/anatomical-prior, and MIRAGE-envelope records. |
| [`frozen/`](frozen/) | Frozen encoder evaluations with MeanPool, CrossAttnPool, and AttentiveProbe variants. |
| [`finetune/`](finetune/) | Fine-tuning evaluations with the encoder unfrozen and layer-wise learning-rate decay. |

## Top-level documents

| Document | Purpose |
|---|---|
| [`mirage_guided_masking.md`](mirage_guided_masking.md) | Defines the two MIRAGE-based masking modes and prevents the rectangle-vs-anatomy ambiguity. |
| [`curriculum_masking.md`](curriculum_masking.md) | Historical detailed plan for oracle, self-guided, and curriculum masking modes. |
| [`interpretability.md`](interpretability.md) | Occlusion attribution analysis for the downstream fine-tune probes. |

## Related documentation

- [`../architecture/`](../architecture/) — model and adapter architecture.
- [`../research_log.md`](../research_log.md) — chronological research log and bibliography pointers.
- [`../lessons_learned.md`](../lessons_learned.md) — debug traps and protocol invariants.
- [`../../archive/`](../../archive/) — parked or superseded experiment directions retained for provenance.
