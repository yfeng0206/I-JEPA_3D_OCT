# Documentation Index

Navigation hub for the I-JEPA 3D OCT project documentation.

## Contents

| Directory | Description |
|-----------|-------------|
| [`architecture/`](architecture/) | System design: patch-level I-JEPA, MIRAGE adapter, downstream probes |
| [`experiments/`](experiments/) | All empirical work: pretraining, frozen probes, fine-tuning, masking, interpretability |
| [`reference/`](reference/) | Citations, related work, and conventions |

## Top-level documents

| Document | Description |
|----------|-------------|
| [`research_log.md`](research_log.md) | Chronological problem/solution log, bibliography, backlog |
| [`lessons_learned.md`](lessons_learned.md) | Mistakes, debug traps, and invariants paid to learn |

## Architecture

| Document | Description |
|----------|-------------|
| [`architecture/README.md`](architecture/README.md) | Full model spec: ViT-B/16 encoder, predictor, EMA target, masking, downstream pipeline |
| [`architecture/mirage_adapter.md`](architecture/mirage_adapter.md) | MIRAGE model structure, cfg-7 adapter, freeze map, data path |

## Experiments

| Document | Description |
|----------|-------------|
| [`experiments/README.md`](experiments/README.md) | Experiment index with results summary table |
| [`experiments/pretraining/`](experiments/pretraining/) | SSL run specs: random, oracle, MIRAGE-envelope, anatomy-guided |
| [`experiments/frozen/`](experiments/frozen/) | Frozen-encoder probe evaluations (d=1, CrossAttnPool, MeanPool) |
| [`experiments/finetune/`](experiments/finetune/) | LLRD fine-tune evaluations |
| [`experiments/masking/`](experiments/masking/) | MIRAGE-guided anatomy masking: ablations, findings, collation fixes |
| [`experiments/interpretability.md`](experiments/interpretability.md) | Occlusion attribution across 3 fine-tune probes |
| [`experiments/curriculum_masking.md`](experiments/curriculum_masking.md) | Historical oracle → self-guided → curriculum plan |

## Reference

| Document | Description |
|----------|-------------|
| [`reference/citations.md`](reference/citations.md) | Full bibliography with BibTeX, usage context, and related-work positioning |
