# Adapter and guide ablations

This page contains the original cfg-7 adapter sweep, guardrails, saturation checks, and precision checks. The newer focused analyses are linked rather than duplicated: [`adapter_placement.md`](adapter_placement.md), [`class_relations.md`](class_relations.md), and [`structural_loss.md`](structural_loss.md).

FairVision has no anatomy labels. Adapter-induced changes are therefore reported as drift, relocation, frozen-output agreement, or representation alignment, not segmentation improvement.

### Adapter architecture: why cfg-7

![Adapter architecture sweep: 12 configurations](../../../results/masking/adapter_sweep/adapter_sweep.png)

*Twelve configurations swept over one pass of 6,000 FairVision images
(375 optimizer steps). cfg-7 sits at the depth-2 knee.*

| cfg | depth | width | alpha | peak LR | params | L_rel ↓ | drift | s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 64 | 0.50 | 1e-3 | 86,528 | 20.6% | 0.192 | 52 |
| **7** | **2** | **128** | **0.50** | **1e-3** | **689,664** | **29.9%** | **0.185** | **60** |
| 11 | 4 | 128 | 0.50 | 1e-3 | 1,280,512 | 30.3% | 0.186 | 62 |

Depth 0→2: +9.3 pp. Depth 2→4: +0.4 pp. The knee is at depth 2.
Alpha is the largest lever; LR second; depth third. OneCycle peak 1e-3 beats
1e-4 in every matched pair. AdamW with gradient clipping at 1.0, dropout 0.

**Selected:** cfg-7 — depth 2, width 128, alpha 0.5, OneCycle peak 1e-3,
dropout 0. 689,664 params (46% fewer than depth 4). Held-out generalisation:
29.79% vs 29.59% train (no gap, §Guardrails T1).

**Memorisation warning:** an earlier probe on **24 slices × 400 steps** reported
76.8% L_rel reduction. That was memorisation. The honest single-pass number is
29.9% (cfg-7) / 30.3% (sweep max) over 6,000 distinct images.

## Adapter Architecture Sweep

Full 12-configuration table from the **6,000-image** one-pass sweep (commit
778791b, `results/masking/adapter_sweep/sweep.json`):

| cfg | depth | width | peak LR | alpha | params | L_rel ↓ | feature drift | seg agreement | mask Jaccard | s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 64 | 1e-4 | 0.25 | 86,528 | 8.6% | 0.093 | 0.988 | 0.867 | 50 |
| 1 | 0 | 64 | 1e-4 | 0.50 | 86,528 | 17.3% | 0.187 | 0.975 | 0.745 | 49 |
| 2 | 0 | 64 | 1e-3 | 0.25 | 86,528 | 10.8% | 0.095 | 0.987 | 0.845 | 51 |
| 3 | 0 | 64 | 1e-3 | 0.50 | 86,528 | 20.6% | 0.192 | 0.973 | 0.730 | 52 |
| 4 | 2 | 128 | 1e-4 | 0.25 | 689,664 | 9.3% | 0.054 | 0.988 | 0.846 | 57 |
| 5 | 2 | 128 | 1e-4 | 0.50 | 689,664 | 18.9% | 0.116 | 0.976 | 0.724 | 57 |
| 6 | 2 | 128 | 1e-3 | 0.25 | 689,664 | 14.8% | 0.092 | 0.985 | 0.856 | 56 |
| **7** | **2** | **128** | **1e-3** | **0.50** | **689,664** | **29.9%** | **0.185** | **0.971** | **0.745** | **60** |
| 8 | 4 | 128 | 1e-4 | 0.25 | 1,280,512 | 9.9% | 0.060 | 0.988 | 0.863 | 61 |
| 9 | 4 | 128 | 1e-4 | 0.50 | 1,280,512 | 20.1% | 0.131 | 0.976 | 0.786 | 62 |
| 10 | 4 | 128 | 1e-3 | 0.25 | 1,280,512 | 15.0% | 0.093 | 0.985 | 0.866 | 62 |
| 11 | 4 | 128 | 1e-3 | 0.50 | 1,280,512 | 30.3% | 0.186 | 0.972 | 0.736 | 62 |

![Adapter feasibility: logit-scale analysis](../../../results/masking/logit_scale/adapter_feasibility.png)

*Feasibility analysis: adapter residual magnitude vs MIRAGE logit scale. The
tanh-bounded residual (alpha=0.5) cannot overwhelm the frozen head's confident
predictions.*

![MIRAGE logit scale distribution](../../../results/masking/logit_scale/mirage_logit_scale.png)

*Distribution of MIRAGE logit magnitudes. Most cells are saturated (margin
≥0.9), limiting the adapter's effective influence to boundaries.*

---

## Guardrail Tests

![Guardrail summary: loss, generalisation, budget lock, confidence localization](../../../results/masking/adapter_guardrails/backprop_effect.png)

*Cfg-7 trained on 4,800 images, evaluated on disjoint 1,200. Reported metrics
use 192 sampled images per split.*

### T1 — Held-out generalisation

| Split | L_rel before | L_rel after | Reduction | Drift | Seg agreement |
|---|---:|---:|---:|---:|---:|
| Train (192) | 0.33636 | 0.23684 | 29.59% | 0.181 | 0.971 |
| Held-out (192) | 0.33719 | 0.23673 | 29.79% | 0.182 | 0.971 |

No generalisation gap. The adapter learned a repeatable transformation.

### T2 — Budget lock

| Guide | Mean cells | Jaccard vs frozen |
|---|---:|---:|
| Frozen | 50.6 | 1.000 |
| Adapted, free | 59.8 | 0.805 |
| Adapted, locked | **47.2** | **0.737** |

Locking removes expansion; ~26% of cells still relocate.

### T3 — Where the adapter acts

| Statistic | Value |
|---|---:|
| corr(\|Δscore\|, margin) | **−0.7159** |
| Uncertain cells (margin <0.5) | 0.2% |
| Mean \|Δ\| at uncertain cells | 0.14307 |
| Sure cells (margin ≥0.9) | 90.2% |
| Mean \|Δ\| at sure cells | 0.01399 |
| Uncertain/sure change ratio | **10.2×** |

![20 held-out slices: before/after masks under frozen budget](../../../results/masking/adapter_guardrails/before_after_20.png)

*Budget-locked mask Jaccard averages 0.773 (range 0.52–0.96) across 20
held-out slices.*

![Frozen-head outputs before and after adaptation](../../../results/masking/adapter_guardrails/seg_before_after.png)

*Demonstrates change, not improvement — no FairVision anatomy labels exist.*

![Element-wise gate probe: gradient cannot reach masked pixels](../../../results/masking/gate_real/elementwise_gate_probe.png)

*The element-wise gate probe confirms `dL/ds = 0` for all target-excluded
patches (1,360 measured). Gradient does not flow through the hard mask.*

---

## Adapter Saturation Curve

Measured against a **fixed** JEPA EMA teacher (cfg-7 adapter, `adapter_sweep.py`,
middle-slice cache at `d = len(vol)//2`):

| Images | Fraction epoch | L_rel reduction | Drift |
|---:|---:|---:|---:|
| 480 | 0.08% | 3.1% | 0.028 |
| 1,200 | 0.20% | 18.1% | 0.122 |
| 2,400 | 0.40% | 27.7% | 0.179 |
| 4,800 | 0.80% | 30.7% | 0.185 |
| 9,600 | 1.60% | 31.9% | 0.188 |
| 19,200 | 3.20% | 32.5% | 0.189 |

**Conclusion:** saturates by ~2,400 images (0.4% of one epoch). Quadrupling
data past 4,800 buys only +1.8 pp. This justifies training the adapter ONCE
and generating the guide once, rather than rerunning MIRAGE every JEPA epoch.

> **Comparability note.** These numbers come from the middle-slice cache and are
> NOT directly comparable to the 26.18% stratified figure (§F), which uses a
> depth-balanced sample. Middle-slice reports 29.90% at convergence vs 26.18%
> stratified — the difference is the depth-selection bias documented above.

**Figure needed:** L_rel reduction and drift vs images seen (does NOT exist on
disk — needs generating from `scripts\adapter_sweep.py`).

**Source:** user-provided specification (CVPR ablation plan); numbers from
`adapter_sweep.py` run against frozen teacher.

---

## AMP vs FP32 Guide Generation

Implementation ablation: is `autocast` safe for the one-time guide precompute?

| Metric | Value |
|---|---:|
| Pixel argmax agreement | 0.999960 |
| 16×16 score \|diff\| mean | 1.9e-05 |
| 16×16 score \|diff\| max | 2.8e-03 |
| Final mask Jaccard | 0.9984 (min 0.609) |
| Identical masks | 254/256 |
| Mean cells (AMP / FP32) | 50.41 / 50.42 |
| Speed (ms/img) | 3.61 (AMP) vs 7.72 (FP32) |
| Speedup | **2.14×** |

**Conclusion:** the one-time preprocessing speedup does not meaningfully change
the masking policy. AMP is safe for guide generation.

**Separate note:** fp32 is retained for DOWNSTREAM evaluation by user
requirement. Bug: `precompute_features()` previously called `autocast()`
unconditionally, so all earlier downstream evaluations were computed in fp16,
not fp32 (commit 376665d).

**Figure needed:** AMP vs FP32 rare-disagreement examples (does NOT exist on
disk — needs generating).

**Source:** user-provided specification (CVPR ablation plan); guide cache
verification in `results/masking/precompute/precompute_verification.json`
(max 1 uint8 level diff); commit 7d0d7f9.

---

### v1 adapter pipeline

![v1 adapter pipeline](../../../results/masking/v1_demo/v1_adapter_pipeline.png)

*The original v1 adapter routing (before the dead-head finding). Retained for
historical reference.*

![Guide equivalence verification](../../../results/masking/v1_demo/guide_equivalence.png)

*Verified that the v1 guide matches the production envelope within rounding.*
