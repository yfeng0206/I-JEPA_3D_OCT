# Experiments

Three sections: pretraining (the SSL runs), frozen (probe-only evaluation with encoder frozen), finetune (encoder unfrozen).

Research direction plans:

- [`masking/README.md`](masking/README.md) — masking-documentation index.
- [`masking/ablations.md`](masking/ablations.md) — **authoritative active
  record:** MIRAGE-guided anatomy sampler, mask-budget ablations, adapter sweep,
  guardrails, bug fixes, rejected designs, blockers, and reproduction.
- [`masking/findings.md`](masking/findings.md) — consolidated quantitative
  findings from all masking experiments.
- [`../architecture/mirage_adapter.md`](../architecture/mirage_adapter.md) —
  MIRAGE + cfg-7 adapter architecture (verified model structure, freeze map,
  data path, losses).
- [`curriculum_masking.md`](curriculum_masking.md) — retained historical plan
  for the distinct oracle → self-guided → curriculum experiment program.

Archived direction:

- [`archive/semantic_teacher_phase0/`](../../archive/semantic_teacher_phase0/) — the frozen semantic-teacher screen (ImageNet-50 I-JEPA / DINOv3 / Qwen3-VL / Molmo / SAM3 / TokenCut maps, CNN stage atlas). Parked 2026-07-31: VLM-derived maps are object-centric, prompt-unstable and not anatomically addressable, so they could not serve as a target-block prior. A medical segmentation model answers the same need directly, which became the MIRAGE direction. Phase-0 evidence and findings are preserved there.

All results use FairVision glaucoma held-out Test split (3000 volumes). Encoder: ViT-B/16.

## Results summary

| Stage | Probe | Params (trainable) | Test AUC | Detail |
|---|---|---|---|---|
| **Finetune** | AttentiveProbe d=1 + Linear, LLRD γ=0.5 | 7.17M + 86M encoder | **0.8878** | [finetune/llrd.md](finetune/llrd.md) |
| **Finetune** | CrossAttnPool + Linear, LLRD γ=0.5 | 277K + 86M encoder | **0.8872** | [finetune/llrd.md](finetune/llrd.md) |
| **Finetune** | MeanPool + Linear, LLRD γ=0.5 | 2.3K + 86M encoder | **0.8868** | [finetune/llrd.md](finetune/llrd.md) |
| Frozen | CrossAttnPool + Linear | 277K | 0.8791 | [frozen/cross_attn_pool.md](frozen/cross_attn_pool.md) |
| Frozen | MeanPool + Linear | 2.3K | 0.8746 | [frozen/mean_pool.md](frozen/mean_pool.md) |
| Frozen | AttentiveProbe d=1 + Linear | 7.17M | 0.8706 | [frozen/d1_sweep.md](frozen/d1_sweep.md) |

Masking ladder, frozen MeanPool on ep100 (same probe, three pretraining arms):
random 0.8746 → oracle **0.8855** ([frozen/oracle_meanpool_sweep.md](frozen/oracle_meanpool_sweep.md))
→ MIRAGE 0.8807 ([frozen/mirage_meanpool_sweep.md](frozen/mirage_meanpool_sweep.md)).
**Rung 1b falsified the program's working assumption**: MIRAGE masks the retina
more purely than the hand-crafted oracle band and still yields the weaker
encoder, so masking purity does not predict downstream AUC.

Best overall: **fine-tune with MAE-style LLRD at Test AUC 0.8878**. +0.017 over the frozen d=1 baseline.

Primary ablation finding: **under fine-tune, the probe architecture is irrelevant.** All three probes land within 0.001 AUC of each other (pairwise p > 0.6). MeanPool (0 probe params) matches AttentiveProbe d=1 (7.17M probe params) when the encoder is unfrozen.

Secondary finding (frozen regime only): **CrossAttnPool beats d=1 at 26× fewer params** (+0.009 AUC, p=0.002) — the self-attn + FFN in the I-JEPA-style attentive probe is over-parameterized for frozen-probe protocols.

## Structure

```
docs/experiments/
  pretraining/
    README.md
    random_100ep.md      random-init ViT-B/16, 100 ep SSL
    oracle_100ep.md      warm-start ep25, anatomical_prior masking (Rung 1)
    mirage_100ep.md      warm-start ep25, mirage_envelope masking (Rung 1b)
  frozen/
    README.md
    d1_sweep.md          AttentiveProbe d=1 sweep across ep25/50/75/100
    cross_attn_pool.md   minimal cross-attention (277K params) on ep100
    mean_pool.md         mean-pool + linear (ablation floor) on ep100
    ablation_analysis.md paired-bootstrap stats on all 6 runs
    oracle_meanpool_sweep.md  frozen MeanPool, oracle vs random (Rung 1)
    mirage_meanpool_sweep.md  frozen MeanPool, MIRAGE vs oracle vs random (Rung 1b)
  finetune/
    README.md
    llrd.md              unfrozen encoder + LLRD γ=0.5 on ep100 (3 probes)
  masking/
    README.md            masking-documentation index
    ablations.md         authoritative MIRAGE-guided anatomy-masking record
  interpretability.md    occlusion attribution: 3 probes converge on disc-rim
  curriculum_masking.md  historical oracle/self-guided/curriculum plan
  mirage_guided_masking.md  compatibility pointer to masking/ablations.md
```

The semantic-teacher research plan and its Phase-0 docs moved to
`archive/semantic_teacher_phase0/docs/` when that direction was parked.

## Reference

- [research_log.md](../research_log.md) — chronological problem/solution log + paper bibliography + backlog
- [lessons_learned.md](../lessons_learned.md) — mistakes, fixes, invariants
- [architecture/](../architecture/) — model architecture spec
