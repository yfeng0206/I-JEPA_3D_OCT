# Anatomy-Guided Pretraining — Run Specification

**Config:** `configs/patch_mirage_anatomy.yaml`
**Run dir:** `D:\jepa_phase0\runs\patch_mirage_anatomy`
**Started:** 2026-08-09

This is the full anatomy-guided pretraining run. It is the first run to use
the two-class soft guide end to end.

---

## 1. What is trained, and what is not

| module | params | trainable |
|---|---|---|
| JEPA context encoder (ViT-B/16) | 85,843,200 | **yes** |
| JEPA predictor (depth 6, dim 384) | 11,337,216 | **yes** |
| JEPA EMA target encoder | 85,843,200 | no — EMA only, no gradient |
| MIRAGE (all of it) | 95,571,460 | **no — 0 trainable** |
| cfg-7 adapter | 689,664 | frozen for this run (trained once, offline) |

MIRAGE is **never loaded during this run**. Its output was precomputed into a
guide cache, so the 95.6M parameters are not resident. See §5.

## 2. Data

| | |
|---|---|
| dataset | FairVision glaucoma, **Training split only** |
| volumes | 6,000 |
| slices per volume | 100, at `np.linspace(0, 199, 100)` |
| **samples per epoch** | **600,000** |
| crop | `PairedRandomResizedCrop`, `crop_size 256`, `crop_scale [0.3, 1.0]` |
| normalisation | ImageNet mean/std, 3-channel (grayscale replicated) |
| augmentation | **none** — no flip, no colour jitter, no blur |
| validation | Validation split, **unguided** dataset (no guide needed) |

Horizontal flip is off deliberately: an OCT B-scan has a fixed anatomical
orientation (vitreous above, choroid below), and the guide is orientation-
specific.

## 3. Optimisation

| setting | value |
|---|---|
| micro-batch | 64 |
| grad accumulation | 8 → **effective batch 512** |
| iterations/epoch | 9,375 |
| epochs | **100** |
| optimiser | AdamW |
| lr schedule | `start_lr 1e-4` → `lr 2.5e-4` (peak) → `final_lr 1e-6`, cosine |
| warmup | **5 epochs** |
| weight decay | `0.04` → `0.4` (ramped) |
| EMA momentum | `0.996` → `1.0` (ramped) |
| precision | **fp32** (`use_bfloat16: false`) |
| dropout | **0** — none anywhere: not in the ViT, not in the adapter |
| stochastic depth | **0** — MIRAGE's `drop_path_rate=0.1` is disabled at source |
| loss | `smooth_l1(predictor_output, EMA_target_features)` |

**Warm start:** resume from `resume-ep27.pth.tar` — the checkpoint produced by
the random-masked bootstrap. Training therefore begins at **epoch 27**, not
from random init, and the encoder has already had ~27 epochs of standard
random-block I-JEPA before any anatomy guidance is applied.

This is deliberate and matters for reading the result: the anatomy arm is
**not** learning from scratch. It inherits a random-masked representation and
the experiment measures what anatomy guidance adds **on top of** that. An
earlier attempt at this run set `read_checkpoint: null` and trained 100 epochs
from random init, which is a different experiment and was aborted.

## 4. Masking

| setting | value |
|---|---|
| mode | `mirage_anatomy` |
| token grid | 16×16 = 256 (patch 16 at crop 256) |
| target sets | 4 |
| **`pred_target_k`** | **16** — every target contributes exactly 16 indices |
| encoder mask scale | `[0.85, 1.0]` |
| `anatomy_mass_cap` | 0.90 |
| `anatomy_tau` | 0.10 |
| fallback | random rectangles when the guide cannot fill 4 targets (~1.7–6%) |

`pred_target_k` is **mandatory** for this mode and the code refuses to start
without it. Without it the collator front-slices every target in the batch to
the smallest one, which retains 7.2% of target cells and collapses to K=1 in
99.8% of batches at micro-batch 64.

### Curriculum ramp

| epochs | `r_t` | masking actually used |
|---|---|---|
| 0–25 | 0.00 | random rectangles — **already done in the bootstrap checkpoint** |
| 25–30 | 0 → 1.0 linear | mixed; this run starts inside this window at epoch 27 |
| 30–100 | 1.00 | **anatomy targets** |

`T_warm 25`, `T_total 30`, `r_max 1.0`, `ramp_shape linear`.

The ramp is per **image**, not per block: an image either gets four anatomy
targets or four random rectangles, never a mixture, because a mixed target set
would not be a coherent partition of the retina.

Resuming at epoch 27 means the run enters partway up the ramp (`r_t = 0.40`)
and reaches full guidance at epoch 30. So **epochs 30–100 are the anatomy
experiment**, roughly 70 epochs of fully-guided training on top of the
bootstrap.

## 5. Guide cache

| | |
|---|---|
| path | `mirage_soft_guides\base512_cfg7_3186b1fa278bc97f` |
| schema | 2 |
| contents | `P_inner`, `P_choroid`, uint8, native 200×200, **post-softmax** |
| shape/volume | `(100, 2, 200, 200)` |
| size | 3.85 GiB for 600,000 slices |
| build time | 3,941 s, once |
| MIRAGE sha | `82e5a0dd09b6bd58` |
| adapter sha | `3186b1fa278bc97f` (taught by `jepa_patch_mirage-ep100`) |

Stored **post-softmax** because the guide is cropped with the image and pooled
to 16×16 afterwards. Pooling before softmaxing is a different function and was
measured to change the masks badly (Jaccard 0.587, 0/200 identical, −40% cells).

## 6. Checkpoints

`save_every: 5` → checkpoints at epochs 5, 10, …, **50**, …, **75**, …, **100**.
Each saves encoder, predictor, EMA target encoder, optimiser, scaler, epoch,
loss and the mask generator state.

## 7. Expected cost

| | |
|---|---|
| measured | ~5,796 s/epoch (anatomy, one full epoch) |
| 50 epochs | ~80 h |
| 75 epochs | ~121 h |
| **100 epochs** | **~161 h (6.7 days)** |
| VRAM | ~17.6 GB of 24 GB |

## 8. What this run can and cannot establish

**Can:** whether anatomy-guided masking produces a better encoder than the
prior rectangle-masked ep100 baseline, measured by downstream glaucoma AUC at
epochs 50/75/100.

**Cannot, on its own:** attribute any gain to target *shape*. This run hides
~23% of the image; the shipped rectangle baseline hides ~46%. Those differ in
both *where* and *how much*. An **area-matched random control** is required
before claiming the shape is what helped — see `findings.md` §1.

**Never:** FairVision has no anatomy ground truth. This run can show the mask
changed and that downstream AUC changed. It cannot show the segmentation
improved.
