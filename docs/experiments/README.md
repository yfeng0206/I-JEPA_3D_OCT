# Experiments

This directory is the index for self-supervised pretraining, downstream evaluation, masking-method ablations, and interpretability work for the OCT I-JEPA project.

## Method statement

The masking contribution is **anatomy-shaped masking**. The `mirage_envelope` baseline uses MIRAGE to place ordinary rectangular I-JEPA targets on the retina. The `mirage_anatomy` method uses MIRAGE to shape connected, irregular targets to tissue. MIRAGE guidance is therefore not the novelty; target shape is.

Measured sampler diagnostics support that distinction: `mirage_envelope` puts 30.7% of masked cells on anatomy with 3.57% dead targets, while `mirage_anatomy` puts 72.1% of masked cells on anatomy with 2.05% dead targets. The matched ep30 comparison is envelope 0.8528 ± 0.0018 test AUC versus anatomy 0.8582 ± 0.0003 test AUC, a +0.0054 difference with Welch p=0.00219 and Cohen's d=4.20. This does not establish that every MIRAGE-based variant is better; it isolates the shape change in that matched comparison.

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
