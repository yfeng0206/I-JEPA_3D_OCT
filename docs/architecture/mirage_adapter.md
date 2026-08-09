# MIRAGE Adapter Architecture

> Architecture, parameter counts and tensor shapes verified by model instantiation
> and forward passes via `scripts/jepa_to_mirage_probe.py::build_mirage()` (2026-08-09).
> Training metrics (L_rel reduction percentages) are from the cited adapter probes
> (`scripts/adapter_stage.py`, `scripts/adapter_sweep.py`).
>
> **Scope:** This document covers only MIRAGE↔JEPA representation adaptation.
> Target collation and batching are handled by the anatomy masking pipeline and
> documented separately (`docs/experiments/masking/findings.md` section 4).

---

## A. MIRAGE Model Structure

| Property | Value |
|----------|-------|
| Total parameters | **95,571,460** |
| Trainable parameters | **0** (all frozen) |
| Input adapter (`PatchedInputAdapter`) | 983,808 params |
| ViT encoder (`Sequential` of `Block`) | 85,054,464 params |
| Output adapter (`ConvNeXtAdapter`, semseg) | 9,532,420 params |

### Input Adapter

```
PatchedInputAdapter
  proj: Conv2d(1, 768, kernel_size=32×32, stride=32)   # grayscale OCT
  + learnable_pos_emb=False (sinusoidal)
```

Patch size 32 on 512×512 input → **16×16 = 256 spatial tokens** + 1 global token.

### ViT Encoder

| Property | Value |
|----------|-------|
| Type | Sequential of `Block` |
| Depth | 12 |
| Embed dim | 768 |
| Attention heads | 12 |
| Global tokens | 1 (shape `[1, 1, 768]`) |
| Drop path rate | 0.1 at build time, **disabled to 0.0 post-load** |

### Semseg Output Adapter (ConvNeXtAdapter)

| Property | Value |
|----------|-------|
| `depth` | 4 |
| `preds_per_patch` | 16 |
| `embed_dim` | 6144 |
| `num_classes` | 4 |
| `patch_size` | [32, 32] |
| `image_size` | [512, 512] |

Children:

```
proj_dec:    Linear(768 → 6144, bias=True)              4,724,736 params
blocks:      Sequential of 4 × ConvNeXtBlock            4,806,144 params
  each block:
    dwconv:  Conv2d(384, 384, 7×7, groups=384, pad=3)   # depthwise
    norm:    LayerNorm(384)
    pwconv1: Linear(384, 1536)
    act:     GELU
    pwconv2: Linear(1536, 384)
    drop_path: Identity (disabled)
final_layer: Conv2d(384, 4, kernel_size=1×1)            1,540 params
```

### Spatial geometry

```
proj_dec reshapes 256 tokens × 6144 → (B, 6144, 16, 16)
  → view as (B, 384, 16×4, 16×4) = (B, 384, 64, 64)      # preds_per_patch=16=4×4
  → ConvNeXtBlocks operate at 64×64
  → final_layer: (B, 384, 64, 64) → (B, 4, 64, 64)       # logits
```

H0 is (B, 384, 64, 64). Pooling 64→16 maps exactly onto the ViT patch grid.

---

## B. Data Path and Freeze Map

```
    +--------- ORIGINAL MIRAGE — ALL FROZEN -----------+
    |  95,571,460 params, 0 trainable                  |
    |                                                  |
    |  image 512×512                                   |
    |    → PatchedInputAdapter  → (B, 256+1, 768)     |
    |    → ViT encoder ×12     → (B, 257, 768)        |
    |    → proj_dec + reshape  → (B, 384, 64, 64)     |
    |    → ConvNeXtBlocks ×4   → H0 (B, 384, 64, 64) |
    +----------------------+---------------------------+
                           |
                           v
              +-------------------------+
              |  cfg-7 ADAPTER          |
              |  TRAINABLE  689,664 p   |
              |  H = H0 + 0.5·tanh(A)  |
              +------------+------------+
                           |
                           v  H
    +--------- ORIGINAL MIRAGE — FROZEN ---------------+
    |  final_layer  Conv2d(384, 4, 1×1)  1,540 p       |
    |  (grad=False, reads the ADAPTED feature H)       |
    +----------------------+---------------------------+
                           |
                           v
              logits (B, 4, 64, 64)
                           |
              softmax → P_inner + P_choroid
                           |
              pool 64×64 → 16×16 → masking targets
```

**Key wiring fact:** The FROZEN `final_layer` reads the ADAPTED feature `H`.
The adapter's output conv is zero-initialised, so at step 0 `H = H0` exactly
(identity property). After training, `H ≠ H0`, and the frozen head's output
changes — this is the signal path that makes `L_rel` effective without any
segmentation labels.

### Dead-end: separate residual head

Routing the adapter into a SEPARATE logit head while the frozen head still
reads unadapted `H0` is a structural dead end:
- The frozen head receives exactly **0.000e+00 gradient** without a labelled segmentation loss
- Segmentation agreement = **1.000000** (guide never changes)
- Mask Jaccard = **1.000000**

The frozen head **must** read the adapted feature.

### Frozen-ness guarantees

1. `build_mirage()` sets `p.requires_grad_(False)` on every MIRAGE parameter (0 of 95,571,460 trainable)
2. `drop_path_rate=0.1` was making the model stochastic: two identical forwards differed by **1.428e+01** in logits after `.train()`. Now disabled at source (`mod.drop_prob = 0.0` for all modules with that attribute). Train and eval modes are bitwise identical (diff = 0.000e+00).

---

## C. The cfg-7 Adapter

```python
class Adapter(nn.Module):
    """cfg 7: depth 2, width 128, alpha 0.5. Zero-init => identity at step 0."""
```

| Property | Value |
|----------|-------|
| Depth (ResBlocks) | 2 |
| Width | 128 |
| Alpha | 0.5 |
| Dropout | 0 |
| Output init | zeros (weight and bias) |
| Total params | **689,664** |

### Module structure (verified)

```
Adapter(
  trunk: Sequential(
    Conv2d(384, 128, 1×1)
    GELU
    ResBlock(
      b: Conv2d(128,128,3×3,pad=1) → GELU → Conv2d(128,128,3×3,pad=1)
      n: GroupNorm(8, 128)
    )
    ResBlock(
      b: Conv2d(128,128,3×3,pad=1) → GELU → Conv2d(128,128,3×3,pad=1)
      n: GroupNorm(8, 128)
    )
  )
  out: Conv2d(128, 384, 1×1)       # zero-initialised
)
```

### Step-0 identity property

```
H = H0 + alpha * tanh(out(trunk(H0)))
    = H0 + 0.5 * tanh(0)       # because out.weight=0, out.bias=0
    = H0
```

The pipeline asserts this: feature drift = 0, seg agreement = 1.0 exactly.

### Training

- Loss: `L_rel = MSE(Gram(pool(H)), sg(Gram(Z_ema)))` — only the adapter's 689,664 params receive gradient
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4, OneCycleLR
- Run ONCE for ~300 steps (4,800 images)
- **Saturation results** (two different evaluation caches — percentages are NOT directly
  comparable across them, but both support the same conclusion: adaptation saturates quickly):
  - *Slice-stratified* cache (`D:\jepa_phase0\mirage-goals\outputs\slice_pos`, one random
    depth per volume via `np.linspace(0,199,100)`; used by `scripts/adapter_stage.py`):
    **26.18% L_rel reduction** at 4,800 images (see `results/masking/slice_pos/slice_pos.json`)
  - *Middle-slice* cache (`d = len(vol)//2`, single most favourable B-scan per volume;
    used by `scripts/adapter_sweep.py`):
    **27.7%** at 2.4k images, **30.7%** at 4.8k, **32.5%** at 19.2k
  - Note: the 6,000-image sweep (`ablations.md`) reports **29.9%** for cfg-7, also from
    the middle-slice cache but at different N (6,000 vs 4,800); the small difference is
    not explained and the two should not be quoted as if identical.
  - The middle-vs-stratified gap is a consistent ~3.7 pp (29.90% middle vs 26.18% stratified;
    depth-band spread only 3.52 pp — see `docs/experiments/masking/findings.md` section 8
    and `results/masking/slice_pos/*.json`)

---

## D. The Losses

### L_JEPA (pretraining)

```
L_JEPA = smooth_l1(predictor_output, sg(EMA_target_features))
```

Trains: JEPA encoder + predictor. The EMA target encoder is updated by momentum, not gradient.

### L_rel (adapter stage)

```
H       = Adapter(H0)                           # (B, 384, 64, 64)
U       = pool(H, 16×16).flatten(2).T           # (B, 256, 384)
R_M     = Gram(L2_normalize(U))                 # (B, 256, 256)

Z_ema   = JEPA_target_encoder(image_256)        # (B, 256, 768)
Z_ema   = LayerNorm(Z_ema)
R_J     = Gram(L2_normalize(Z_ema))             # (B, 256, 256)

L_rel   = MSE(R_M, sg(R_J))
```

Trains: ONLY the cfg-7 adapter (689,664 params).

### Verified shapes (from forward pass)

| Tensor | Shape | Notes |
|--------|-------|-------|
| H0 | (B, 384, 64, 64) | MIRAGE decoder feature before final_layer |
| Logits | (B, 4, 64, 64) | Output of final_layer(H) |
| Pooled MIRAGE U | (B, 256, 384) | pool 64→16, flatten, transpose |
| JEPA EMA Z | (B, 256, 768) | ViT-B/16 on 256×256, 16×16=256 tokens |
| Gram R_M | (B, 256, 256) | Normalised self-similarity |
| Gram R_J | (B, 256, 256) | Normalised self-similarity |

The Gram matrices are in the same (B, 256, 256) space despite the feature
dimensions differing (384 vs 768), because Gram computes token-to-token
similarity after L2 normalisation.

---

## E. Final Training Schedule

The cfg-7 adapter does **not** train continuously alongside JEPA. The full
pipeline is strictly sequential:

```
1. JEPA warmup  →  pick an EMA checkpoint
2. Train cfg-7 ONCE with L_rel
      ~300 steps, ~4,800 FairVision images  (scripts/adapter_stage.py)
3. FREEZE cfg-7 permanently
4. ONE MIRAGE + cfg-7 forward pass over all 600,000 slices
      (scripts/precompute_soft_guides.py)
5. Cache P_inner, P_choroid to disk
6. All remaining JEPA epochs read the cached guide
      — no MIRAGE forward per epoch
      — no further adapter updates
```

**Cost:** The cache build (step 4-5) takes **3,941 s once** for 6,000 volumes /
600,000 slices, producing **3.85 GiB compressed** (`np.savez_compressed`).
This replaces live MIRAGE inference across 75 epochs, which would cost
~75 × 36 min ≈ **45 h** and require MIRAGE's 95.6 M parameters to be GPU-resident
every epoch.

---

## Source files

| File | Role |
|------|------|
| `scripts/jepa_to_mirage_probe.py` | `build_mirage()`, `build_jepa()` |
| `scripts/adapter_stage.py` | `Adapter` class, training loop, L_rel |
| `scripts/precompute_soft_guides.py` | Guide cache builder |
| `src/masks/anatomy.py` | Production anatomy sampler |
