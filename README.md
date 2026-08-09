# I-JEPA for FairVision OCT Glaucoma Classification

Self-supervised pretraining using [I-JEPA](https://github.com/facebookresearch/ijepa) (Assran et al., CVPR 2023) on [Harvard FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision) OCT data, evaluated via frozen probe + fine-tune on binary glaucoma classification.

## Headline result — anatomy-guided masking

Best downstream glaucoma classifier: **0.8947 Test AUC** (FairVision, 3000-volume held-out) — fine-tuned MeanPool on an encoder pretrained with anatomy-guided ("oracle") masking, which biases I-JEPA's prediction targets onto the retinal band. Beats random-masking I-JEPA (0.8878) at every probe and regime.

![Best downstream Test AUC: anatomy-guided masking (ours) 0.8947 vs random-masking I-JEPA 0.8878](results/summary/oracle_headline.png)

Paired bootstrap, B=2000, on the 3000-volume Test split:

| Regime | Probe | Random | Oracle | Δ | p |
|---|---|---|---|---|---|
| Frozen | MeanPool | 0.8746 | 0.8855 | +0.0109 | <0.0005 |
| Fine-tune | MeanPool | 0.8868 | **0.8947** | +0.0079 | 0.001 |
| Fine-tune | CrossAttnPool | 0.8872 | 0.8937 | +0.0065 | 0.009 |
| Fine-tune | AttentiveProbe d=1 | 0.8878 | 0.8901 | +0.0023 | 0.26 (ns) |

Source: [`docs/experiments/frozen/oracle_meanpool_sweep.md`](docs/experiments/frozen/oracle_meanpool_sweep.md), [`docs/experiments/finetune/oracle_finetune.md`](docs/experiments/finetune/oracle_finetune.md).

## Probe-architecture ablation (random-init baseline)

Random-init I-JEPA ViT-B/16, 100 epochs SSL on 600K OCT slices. Full 2×3 matrix on the 3000-volume Test split.

| Method | Probe | Params (trainable) | **Test AUC** |
|---|---|---|---|
| **Fine-tune + LLRD γ=0.5** | AttentiveProbe d=1 + Linear | 7.17M + 86M encoder | **0.8878** |
| **Fine-tune + LLRD γ=0.5** | CrossAttnPool + Linear | 277K + 86M encoder | **0.8872** |
| **Fine-tune + LLRD γ=0.5** | **MeanPool + Linear (0 probe params)** | **2.3K + 86M encoder** | **0.8868** |
| Frozen probe | CrossAttnPool + Linear | 277K | 0.8791 |
| Frozen probe | MeanPool + Linear | 2.3K | 0.8746 |
| Frozen probe | AttentiveProbe d=1 + Linear | 7.17M | 0.8706 |

**Key finding:** Under fine-tune, probe architecture is irrelevant (all within 0.001 AUC, p>0.6 pairwise). MeanPool (0 probe params) is Pareto-optimal. Full analysis: [`docs/experiments/frozen/ablation_analysis.md`](docs/experiments/frozen/ablation_analysis.md).

## Method

- **Pretraining**: I-JEPA on 256×256 OCT slices (FairVision Training split, 600K slices). ViT-B/16, 100 epochs, peak LR 0.00025, EMA 0.996→1.0, effective batch 512.
- **Downstream input**: Frozen ViT encodes each slice → per-slice 768-dim token. 100 slices per volume.
- **Probes**: MeanPool / CrossAttnPool / AttentiveProbe d=1, all with linear binary head.
- **Fine-tune**: LLRD γ=0.5 with base LR 2e-4, 50 epochs planned / early-stopped by patience=15.

See [`docs/architecture/`](docs/architecture/) for the full spec.

## Dataset

Harvard FairVision Glaucoma subset: 10,000 subjects (6K Train / 1K Val / 3K Test), each with a 200×200×200 OCT B-scan volume. Binary label glaucoma/not. ~48.5% positive prevalence — balanced. Available on [HuggingFace](https://huggingface.co/datasets/ming0100/Harvard_FairVision).

## Roadmap

- Phase 1 (done): Random-init I-JEPA SSL → frozen probe + fine-tune evaluation
- Phase 2 (done): Probe architecture ablations — full 2×3 matrix (3 probes × frozen/fine-tune)
- Phase 3 (done): Interpretability — occlusion attribution, patch aggregate, bootstrap CI
- Masking strategy (Rung 1, done): anatomy-guided "oracle" masking beats random — frozen +0.010 (p<0.0005), fine-tune +0.008
- Phase 4 (in progress): Foundation-model baselines on same Test split (DINOv3, OCTCube)
- Phase 5 (planned): 3D-aware SSL extension (multi-view / axial)

Details and backlog: [`docs/research_log.md`](docs/research_log.md).

## Documentation

| | |
|---|---|
| **Full index** | [`docs/README.md`](docs/README.md) |
| Architecture | [`docs/architecture/`](docs/architecture/) |
| Experiments | [`docs/experiments/`](docs/experiments/) |
| Citations & related work | [`docs/reference/citations.md`](docs/reference/citations.md) |
| Interpretability | [`docs/experiments/interpretability.md`](docs/experiments/interpretability.md) |
| Lessons learned | [`docs/lessons_learned.md`](docs/lessons_learned.md) |
| Research log | [`docs/research_log.md`](docs/research_log.md) |

## References

Key citations (full bibliography with BibTeX: [`docs/reference/citations.md`](docs/reference/citations.md)):

- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA), CVPR 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- Luo et al., *FairVision: Equitable Deep Learning for Eye Disease Screening*, 2024. [arXiv:2310.02492](https://arxiv.org/abs/2310.02492)
- Park et al., *Relational Knowledge Distillation*, CVPR 2019. [arXiv:1904.05068](https://arxiv.org/abs/1904.05068)

Full bibliography with context: [`docs/reference/citations.md`](docs/reference/citations.md).
