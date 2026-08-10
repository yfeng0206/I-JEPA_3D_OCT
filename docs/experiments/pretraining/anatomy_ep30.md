# Pretraining: MIRAGE anatomy-shaped targets (stopped at ep30)

This document describes the current `mirage_anatomy` arm. MIRAGE supplies the tissue guide, and the target shape changes from rectangles to connected irregular anatomy-shaped blobs. This is the contribution being tested.

This is not the same as [envelope_100ep.md](envelope_100ep.md). The `mirage_envelope` arm uses MIRAGE only to place ordinary rectangles and is the baseline for MIRAGE placement.

## Status

| Item | Value |
|---|---|
| Config | `configs/patch_mirage_anatomy.yaml` |
| Run directory | `D:\jepa_phase0\runs\patch_mirage_anatomy\` |
| Status | Stopped at epoch 30 |
| Checkpoints on disk | `ep30.pth.tar` and `best.pth.tar` only |
| Warm start | `D:\jepa_phase0\runs\patch_mirage_envelope\resume-ep27.pth.tar` |
| Current fair comparator | `mirage_envelope` ep30 |

The anatomy arm was warm-started from the envelope run's `resume-ep27.pth.tar`. It therefore tests what anatomy-shaped targets add after the random/envelope bootstrap, not learning from scratch.

## What is trained

| Module | Parameters | Trainable |
|---|---:|---|
| JEPA context encoder, ViT-B/16 | 85,843,200 | yes |
| JEPA predictor, depth 6, dim 384 | 11,337,216 | yes |
| JEPA EMA target encoder | 85,843,200 | no; EMA only |
| MIRAGE | 95,571,460 | no; not resident during pretraining |
| cfg-7 adapter | 689,664 | frozen for this run |

MIRAGE outputs were precomputed into a guide cache. MIRAGE is not loaded during pretraining.

## Data and optimization

| Setting | Value |
|---|---|
| Dataset | FairVision glaucoma Training split only |
| Volumes | 6,000 |
| Slices per volume | 100 at `np.linspace(0, 199, 100)` |
| Samples per epoch | 600,000 |
| Crop | `PairedRandomResizedCrop`, `crop_size 256`, `crop_scale [0.3, 1.0]` |
| Normalization | ImageNet mean/std, grayscale replicated to 3 channels |
| Augmentation | none; no flip, color jitter, or blur |
| Micro-batch | 64 |
| Grad accumulation | 8; effective batch 512 |
| Iterations per epoch | 9,375 |
| Optimizer | AdamW |
| LR schedule | start_lr 1e-4 → peak 2.5e-4 → final_lr 1e-6, cosine |
| Warmup | 5 epochs |
| Weight decay | 0.04→0.4 |
| EMA momentum | 0.996→1.0 |
| Intended precision | fp32 for downstream evaluation; pretraining config has `use_bfloat16: false` |
| Loss | `smooth_l1(predictor_output, EMA_target_features)` |

Horizontal flip is off because an OCT B-scan has fixed anatomical orientation and the guide is orientation-specific.

## Masking

| Setting | Value |
|---|---|
| Mode | `mirage_anatomy` |
| Target shape | Connected irregular anatomy-shaped blobs |
| Token grid | 16×16 = 256 |
| Target sets | 4 |
| `pred_target_k` | 16; every target contributes exactly 16 indices |
| Encoder mask scale | [0.85, 1.0] |
| `anatomy_mass_cap` | 0.90 |
| `anatomy_tau` | 0.10 |
| Fallback | Random rectangles when the guide cannot fill 4 targets, about 1.7–6% |

`pred_target_k` is mandatory for this mode. Without it, the collator front-slices every target in the batch to the smallest target, retaining 7.2% of target cells and collapsing to K=1 in 99.8% of micro-batch-64 batches.

### Curriculum ramp

| Epochs | `r_t` | Masking actually used |
|---|---:|---|
| 0–25 | 0.00 | Random rectangles, already present in the warm-start checkpoint |
| 25–30 | 0→1.0 linear | Mixed by image; this run starts inside this window at epoch 27 |
| 30 onward | 1.00 | Anatomy-shaped targets |

The ramp is per image, not per block. An image receives four anatomy targets or four random rectangles, never a mixture.

## Guide cache

| Item | Value |
|---|---|
| Path | `mirage_soft_guides\base512_cfg7_3186b1fa278bc97f` |
| Schema | 2 |
| Contents | `P_inner`, `P_choroid`, uint8, native 200×200, post-softmax |
| Shape per volume | `(100, 2, 200, 200)` |
| Size | 3.85 GiB for 600,000 slices |
| Build time | 3,941 s |
| MIRAGE sha | `82e5a0dd09b6bd58` |
| Adapter sha | `3186b1fa278bc97f`, taught by `jepa_patch_mirage-ep100` |

The cache is post-softmax because the guide is cropped with the image and then pooled to 16×16. Pooling before softmaxing is a different function and was measured to change masks badly: Jaccard 0.587, 0/200 identical, −40% cells.

## Result

Frozen mean-pool probe, 100 slices/volume, true fp32, test n=3000.

| Arm | Checkpoint | AUC | Seeds | Notes |
|---|---|---:|---:|---|
| `mirage_envelope` | ep30 | 0.8528 ± 0.0018 | 5 | MIRAGE-placed rectangles |
| `mirage_anatomy` | ep30 | 0.8582 ± 0.0003 | 5 | Connected anatomy-shaped targets |
| anatomy − envelope | ep30 | +0.0054 | 5 per arm | Welch t p=0.00219; Mann-Whitney p=0.0079; Cohen's d 4.20; arms fully separated |
| anatomy − envelope paired bootstrap | ep30 | +0.0044 | volume bootstrap | 95% CI [+0.0010,+0.0077]; p=0.012 |

This is evidence that anatomy-shaped targets improve the ep30 encoder over the matched envelope-rectangle baseline. It does not establish the ep100 outcome because the anatomy run is stopped at ep30.

Historic ep100 AUCs are random 0.8746, `mirage_envelope` 0.8807, and oracle 0.8855. Those are useful context but are not a fair head-to-head for the stopped anatomy arm.

## Validation-loss caveat

`src/train_patch.py` hard-codes validation to a plain `MaskCollator` with 42-cell rectangles and the comment "ALWAYS uniform — keeps val loss comparable". This keeps validation comparable across runs but makes train loss and validation loss incomparable for curriculum runs.

The anatomy run trains on 16-cell anatomy blobs. Validation uses 42-cell rectangles. Anatomy and `random_default` had near-identical train loss, 0.0247 vs 0.0246, but validation differed by 1.5×. Rising validation loss at ep30–32 is a ramp artifact; the baseline diverges identically when its guidance ramp completes. Do not interpret that rise as degradation. Use downstream AUC.

## Downstream precision and cache correction

A bug in `src/eval_downstream.py` meant `data.use_amp: false` was ignored at 5 of 6 `autocast()` call sites. Earlier downstream evaluations therefore ran mostly in fp16 despite the config. The measured impact was immaterial for this result: 0.8582 fp16 vs 0.8583 fp32. No earlier result is invalidated, and the fix is in.

A second latent bug omitted precision from the feature-cache key, allowing an fp16 cache to be silently reused by an fp32 run. The cache key is now `{split}_s{n}_{amp|fp32}`.

## What this run can and cannot establish

Can establish:

- At matched ep30, `mirage_anatomy` outperformed `mirage_envelope` by +0.0054 AUC, with Welch t p=0.00219 and paired-bootstrap p=0.012.
- The current comparison isolates target shape better than the historic ep100 table because both ep30 arms are matched in training length.

Cannot establish:

- The ep100 anatomy result. The run has only `ep30.pth.tar` and `best.pth.tar` on disk.
- That MIRAGE guidance is novel. MIRAGE guidance is already used by the `mirage_envelope` baseline.
- That segmentation improved. FairVision has no anatomy ground truth.

## Planned replacement run

A fresh `mirage_anatomy` run is planned, not done. It will warm-start from the random ep25 checkpoint and use a new adapter design: encoder-tap placement with a class-conditioned structural loss. See [structural_loss.md](../masking/structural_loss.md) and [adapter_placement.md](../masking/adapter_placement.md).

The adapter and guide cache will refresh at epochs 25/50/75/100 to match checkpoint boundaries. A guide-cache rebuild costs 66 minutes, measured. Caches are sha-tagged by adapter, so a new adapter writes a new cache directory and old results stay reproducible.
