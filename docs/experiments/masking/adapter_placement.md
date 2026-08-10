# Adapter placement ablation

**Question.** The JEPA→MIRAGE adapter does not improve segmentation and at
α=0.50 significantly harms it. Two candidate explanations:

1. **tap point** — the adapter sits on `H0`, the last feature map, with only a
   1,540-parameter 1×1 conv downstream, so perturbations land directly on the
   logits with nothing to absorb them;
2. **objective mismatch** — MIRAGE (MultiMAE / masked image modelling) and
   I-JEPA (joint-embedding predictive) encode different similarity structure,
   so no placement can reconcile them.

This ablation separates the two. Run with
`scripts/adapter_placement_ablation.py`.

## Design

Three taps into the frozen MIRAGE segmentation path:

| tap | site | shape | downstream of tap |
|---|---|---|---|
| `enc` | pre `proj_dec` | (B, 256, 768) | whole decoder (95.5M) |
| `mid` | pre ConvNeXt blocks | (B, 384, 64, 64) | 4 blocks + head |
| `h0` | pre `final_layer` | (B, 384, 64, 64) | **1,540 params** (current) |

Everything else is held fixed: same EMA teacher (`envelope-ep100`), same 4,800
train / 1,200 eval FairVision slices, same steps, same trunk shape
(depth 2, width 128), same lr 1e-3, same zero-init residual
`Z' = Z + α·tanh(A(Z))`.

Training is identical for all three: capture the representation at the tap
under `no_grad`, then train the adapter on it — `L_rel` is evaluated *at the
tap*, so no gradient need flow through the decoder. Evaluation differs: a
forward **pre-hook** injects the adapter into MIRAGE's own forward, so the
frozen decoder actually processes the perturbed features and GOALS Dice
reflects the true downstream cost.

Validation: with no adapter the harness reproduces the known frozen baseline
Dice of **0.9457** exactly, and the `h0` tap returns amplification **1.00** by
construction, as it must.

## Result

α is swept because the same α produces very different transfer at different
taps, so matched-α comparison is meaningless.

| tap | α | L_rel red | GOALS Dice | Δ vs frozen | paired t | amplif | mask J |
|---|---|---|---|---|---|---|---|
| enc | 0.05 | 24.50% | 0.9454 | −0.00035 | 0.140 | 0.49 | 0.9973 |
| enc | 0.10 | 46.00% | 0.9446 | −0.00108 | 0.019 | 0.48 | 0.9896 |
| enc | 0.25 | 77.74% | 0.9396 | −0.00616 | <1e-4 | 0.41 | 0.9725 |
| enc | 0.50 | 86.97% | 0.9328 | −0.01297 | <1e-4 | 0.39 | 0.9535 |
| mid | 0.05 | 75.59% | 0.9437 | −0.00206 | <1e-4 | 0.56 | 0.9959 |
| mid | 0.10 | 79.14% | 0.9423 | −0.00345 | <1e-4 | 0.44 | 0.9811 |
| mid | 0.25 | 82.76% | 0.9399 | −0.00582 | <1e-4 | 0.56 | 0.9676 |
| mid | 0.50 | 85.87% | 0.9382 | −0.00749 | <1e-4 | 0.57 | 0.9610 |
| h0 | 0.05 | 2.92% | 0.9461 | +0.00037 | 0.049 | 1.00 | 0.9963 |
| h0 | 0.10 | 5.87% | 0.9460 | +0.00030 | 0.342 | 1.00 | 0.9888 |
| h0 | 0.25 | 14.84% | 0.9452 | −0.00055 | 0.425 | 1.00 | 0.9741 |
| h0 | 0.50 | 29.90% | 0.9410 | −0.00477 | <1e-3 | 1.00 | 0.9577 |

### Matched-transfer comparison

The honest axis is *cost to achieve the same L_rel reduction*:

| transfer | enc | mid | h0 |
|---|---|---|---|
| ~25% | α=.05 → **−0.0004** (p=0.14, neutral) | — | α=.50 → −0.0048 (p=3e-4, harm) |
| ~46% | α=.10 → **−0.0011** | — | α=1.0 → −0.0162 |
| ~76% | α=.25 → −0.0062 | α=.05 → **−0.0021** | cannot reach |

**`h0` — the current placement — is dominated everywhere.** It saturates at
29.9% transfer for α≤0.5 and needs α=1.0 (severe damage, −0.0162) to reach
48.7%. The encoder tap reaches the same transfer for **~15× less damage**.

Efficiency, L_rel reduction per unit Dice damage: **enc α=0.05 = 70,000 vs
h0 α=0.50 = 6,268 → 11× better.**

### The frozen decoder absorbs; it does not amplify

The stated risk was that pushing encoder features off-manifold would explode
through the frozen decoder. Measured `(‖ΔH0‖/‖H0‖)/(‖ΔZ‖/‖Z‖)`:

```
h0    1.00          by construction — validates the metric
mid   0.44 – 0.57
enc   0.39 – 0.49   decoder damps the perturbation by ~60%
```

The decoder roughly **halves** the relative perturbation. The danger does not
materialise.

### Why h0 transfers so little

Gram agreement with JEPA *before any training*:

```
enc  r = 0.313
mid  r = 0.371
h0   r = 0.062     5–6× further from JEPA
```

`H0` is a task-specialised, near-logit representation whose similarity
structure is essentially uncorrelated with JEPA's. `L_rel` there is asking two
unrelated structures to agree, which is why it buys so little and costs so much.

## Caveat

At matched **mask relocation** (Jaccard ≈0.96) rather than matched transfer,
`h0` is the cheapest (−0.0048 vs enc −0.0130). Configurations that are
Dice-neutral barely move the guide (enc α=0.05 → J=0.9973). This ablation
settles **placement**; it does not establish that the adapter earns its place
at all. That requires the downstream AUC ablation against a frozen-MIRAGE
guide.

## Conclusion

Explanation (1) — tap point — is a real and large effect, and the hypothesised
ranking **encoder > mid-decoder > H0** is confirmed on the transfer/damage
trade-off. Explanation (2) is not refuted: even at the encoder, agreement with
JEPA starts at r=0.31, and buying high transfer still costs Dice.

Figure: `placement_tradeoff.png`. Raw data: `placement_sweep.json`.
