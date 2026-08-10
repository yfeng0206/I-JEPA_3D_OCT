# Pretraining: MIRAGE envelope rectangles (warm-start ep25, 100ep)

This document describes the `mirage_envelope` arm. MIRAGE placed ordinary rectangular I-JEPA targets on the repaired retinal envelope. The target shape stayed rectangular. This is a baseline for MIRAGE placement, not the anatomy-shaped contribution.

The anatomy-shaped arm is documented in [anatomy_ep30.md](anatomy_ep30.md). In that arm, `mirage_anatomy` changes the target shape to connected irregular tissue blobs.

## Status

| Item | Value |
|---|---|
| Run directory | `D:\jepa_phase0\runs\patch_mirage_envelope\` |
| Status | Completed 100 epochs |
| Checkpoints on disk | ep30,35,40,45,...,100 plus `resume-ep27.pth.tar` and `best` |
| Downstream historic ep100 AUC | 0.8807 |
| Matched ep30 AUC | 0.8528 ± 0.0018 over 5 probe seeds |

## Lineage

```
random-init run:  ep0 ───────────────────────────► ep100   (all random masking)
                       │
                       ├─ ep25 fork
                       ├──► oracle run:          ep25 ──[curriculum ep26–30]──► ep100  (anatomical_prior rectangles)
                       └──► mirage_envelope:     ep25 ──[curriculum ep26–30]──► ep100  (MIRAGE-placed rectangles)
```

All three 100-epoch arms share the same random ep25 starting point. The variable versus the oracle arm is where the rectangular target blocks land. Shape is not changed in this arm.

## Config

| Parameter | Value | vs oracle |
|---|---|---|
| Architecture | ViT-B/16 | same |
| Initialization | Warm-start from random ep25 | same |
| Masking mode | `mirage_envelope` | oracle: `anatomical_prior` |
| Target shape | Rectangles | same rectangular I-JEPA targets, different placement |
| Curriculum | T_warm=25, T_total=30, r_max=1.0, linear ramp | same |
| Guide | MIRAGE repaired envelope, occupancy ≥ 0.25, dilation 0 | oracle: hand-crafted band |
| Block fill / retina visible | ≥ 0.40 / ≥ 0.25, best-effort | n/a |
| I-JEPA masking | enc 0.85–1.0, pred 0.15–0.2, nenc=1, npred=4 | same |
| Learning rate | 0.00025 cosine to 1e-6; start 1e-4; warmup 5 | same |
| EMA | 0.996→1.0 cosine | same |
| Weight decay | 0.04→0.4 cosine | same |
| Batch | 64 × 1 GPU × 8 accum = 512 effective | oracle: 64 × 4 GPUs × 2 accum = 512 |
| Num slices | 100 | same |
| Patience | 9999, disabled | same |
| Total epochs | 100 | same |

## Diagnostic plots

![All Diagnostics](../../../results/pretraining/pretrain_mirage_envelope/diagnostics_all.png)

| Plot | Description |
|---|---|
| ![Loss](../../../results/pretraining/pretrain_mirage_envelope/train_val_loss.png) | Train and validation loss. Same broad shape as oracle and random: rises into a mid-run peak around ep70–75 then eases. |
| ![Rep Diversity](../../../results/pretraining/pretrain_mirage_envelope/rep_diversity.png) | Representation diversity 0.12–0.34; ep100 = 0.236. This metric is background-dominated for OCT and cannot rule out retina-specific collapse. |
| ![Cos Sim](../../../results/pretraining/pretrain_mirage_envelope/cos_sim.png) | Predictor-target cosine 0.755–0.901; the mid-run dip is milder than the oracle's 0.69. |

## Training summary

| Epoch | r_t | train_loss | val_loss | cos_sim | rep_div | l2_dist |
|---:|---:|---:|---:|---:|---:|---:|
| 26 | 0.00 | 0.1182 | 0.1191 | 0.874 | 0.337 | 12.34 |
| 30 | 1.00 | 0.1198 | 0.1200 | 0.854 | 0.272 | 12.69 |
| 35 | 1.00 | 0.1184 | 0.1305 | 0.844 | 0.238 | 13.53 |
| 50 | 1.00 | 0.1216 | 0.1401 | 0.843 | 0.270 | 14.07 |
| 60 | 1.00 | 0.1276 | 0.1463 | 0.852 | 0.220 | 13.67 |
| 75 | 1.00 | 0.1332 | 0.1514 | 0.852 | 0.229 | 14.30 |
| 88 | 1.00 | 0.1269 | 0.1476 | 0.821 | 0.266 | 15.45 |
| 95 | 1.00 | 0.1242 | 0.1457 | 0.782 | 0.199 | 15.87 |
| 100 | 1.00 | 0.1234 | 0.1448 | 0.811 | 0.236 | 15.37 |

Full per-epoch data: [`../../../logs/pretraining/mirage_epoch_summary.csv`](../../../logs/pretraining/mirage_epoch_summary.csv).

## Envelope rectangles vs oracle rectangles

| Epoch | `mirage_envelope` train / val | Oracle train / val | Δ train | Δ val |
|---:|---|---|---:|---:|
| 26 | 0.1182 / 0.1191 | 0.1186 / 0.1202 | −0.0004 | −0.0011 |
| 30 | 0.1198 / 0.1200 | 0.1197 / 0.1242 | +0.0001 | −0.0042 |
| 35 | 0.1184 / 0.1305 | 0.1232 / 0.1310 | −0.0048 | −0.0005 |
| 50 | 0.1216 / 0.1401 | 0.1316 / 0.1400 | −0.0100 | +0.0001 |
| 60 | 0.1276 / 0.1463 | 0.1388 / 0.1489 | −0.0112 | −0.0026 |
| 75 | 0.1332 / 0.1514 | 0.1404 / 0.1507 | −0.0072 | +0.0007 |
| 88 | 0.1269 / 0.1476 | 0.1335 / 0.1454 | −0.0066 | +0.0022 |
| 100 | 0.1234 / 0.1448 | 0.1303 / 0.1432 | −0.0069 | +0.0016 |

Validation is measured under uniform rectangular masking in both arms, so it is like-for-like across runs. Training loss is measured under each arm's own masking policy, so lower training loss means the pretext task is easier, not that the encoder is better.

## Validation-loss caveat

`src/train_patch.py` hard-codes validation to a plain `MaskCollator` with 42-cell rectangles and the comment "ALWAYS uniform — keeps val loss comparable". This is useful for comparing validation loss across runs. It is not a train-vs-validation generalization gap for curriculum runs when the training target geometry differs from validation.

This caveat matters more for [anatomy_ep30.md](anatomy_ep30.md), where training uses 16-cell anatomy blobs. The ep30–32 validation rise in that run is a ramp artifact, not evidence of degradation.

## Downstream

Frozen mean-pool probe, 100 slices/volume, true fp32, test n=3000:

| Checkpoint | AUC | Notes |
|---|---:|---|
| ep30 | 0.8528 ± 0.0018 | 5 probe seeds; fair comparator for anatomy ep30 |
| ep100 | 0.8807 | Historic 100-epoch number |

The historic ep100 comparison was random 0.8746, `mirage_envelope` 0.8807, oracle 0.8855. It does not answer whether anatomy-shaped targets help, because the anatomy arm currently has only ep30.

Earlier downstream evaluations ran mostly in fp16 because `data.use_amp: false` was ignored at 5 of 6 `autocast()` call sites in `src/eval_downstream.py`. The measured impact on the current anatomy probe was 0.8582 fp16 vs 0.8583 fp32, so earlier results are not invalidated. The fix is in. The feature-cache key now includes precision as `{split}_s{n}_{amp|fp32}` to avoid fp16 cache reuse in fp32 runs.
