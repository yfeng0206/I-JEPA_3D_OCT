# Anatomy-shaped vs rectangular masking — matched ep30 comparison

**Status:** complete. First fair head-to-head between anatomy-shaped prediction
targets and the rectangular I-JEPA baseline.

## Why this comparison and not the earlier ones

Every downstream AUC reported before this point came from an **ep100**
checkpoint (random 0.8746, envelope 0.8807, oracle 0.8855). The anatomy run
only reaches ep30, so comparing it against those numbers measured *training
length*, not *masking strategy*.

This experiment removes that confound.

## Design

Both arms start from **identical weights**. The anatomy run was warm-started
from the envelope run's `resume-ep27.pth.tar`, so the two arms share every
weight up to epoch 27 and differ **only in the masking strategy applied during
epochs 28–30**.

| | envelope (baseline) | anatomy (ours) |
|---|---|---|
| checkpoint | `patch_mirage_envelope/…-ep30.pth.tar` | `patch_mirage_anatomy/…-ep30.pth.tar` |
| epochs 1–27 | rectangles | rectangles (shared weights) |
| epochs 28–30 | rectangles | anatomy-shaped blobs |
| eval config | `configs/frozen_meanpool_envelope_ep30.yaml` | `configs/frozen_meanpool_anatomy_ep30.yaml` |

The two eval configs were diffed programmatically and differ in exactly two
keys: `model.encoder_checkpoint` and `logging.output_dir`. Everything else —
probe type, learning rates, dropout, epochs, patience, slice count, precision —
is identical.

Protocol: frozen encoder, mean-pool probe + linear head, 100 slices/volume,
50 epochs with patience 15, early stop on val AUC, evaluated on the held-out
test split (n=3000; 1466 positive, 1534 negative).

## Precision

Both arms ran with `use_amp: false`. This required first fixing a bug in which
`autocast()` was called unconditionally at five sites, so the flag had no
effect outside feature precompute (see *Precision audit* below).

The cached features were then **verified** to be fp32-derived rather than
inherited from an earlier fp16 run, by recomputing four test volumes in both
precisions and comparing:

| arm | \|cache − fp32\| | \|cache − fp16\| | conclusion |
|---|---|---|---|
| anatomy | 1.907e-06 | 2.260e-03 | fp32 |
| envelope | 2.384e-06 | 3.710e-03 | fp32 |

## Result

### Test AUC, seed 42

| arm | val AUC | **test AUC** |
|---|---|---|
| envelope ep30 | 0.8467 | 0.8539 |
| anatomy ep30 | 0.8461 | **0.8583** |

Paired bootstrap over test volumes (10,000 resamples, identical label order
verified):

```
diff  +0.0044   95% CI [+0.0010, +0.0077]   p = 0.012
```

### Probe-seed control (5 seeds each)

The bootstrap above captures test-set sampling noise only. Because probe
training is stochastic, the probe was re-trained with five seeds per arm on the
same verified feature caches.

| seed | anatomy | envelope |
|---|---|---|
| 42 | 0.8583 | 0.8540 |
| 43 | 0.8580 | 0.8530 |
| 44 | 0.8583 | 0.8533 |
| 45 | 0.8578 | 0.8497 |
| 46 | 0.8586 | 0.8542 |
| **mean** | **0.8582** | **0.8528** |
| sd | 0.0003 | 0.0018 |

```
delta          +0.0054
Welch t        p = 0.00219
Mann-Whitney   p = 0.0079
Cohen's d      4.20
separation     min(anatomy) 0.8578  >  max(envelope) 0.8542   FULLY SEPARATED
```

The worst anatomy seed beats the best envelope seed. The arms do not overlap.

### Secondary observation: stability

Anatomy seed-to-seed sd is **0.0003 vs 0.0018**, a 6× reduction. The
anatomy-pretrained features give a more reproducible probe, not just a better
one.

## What this does and does not establish

### Correction: the arms are matched on anatomy, not on raw cells

An earlier version of this document treated the arms as confounded, on the
grounds that anatomy hides fewer cells (54.3 vs 117.5) and sees more context
(158.2 vs 71.4 tokens). Those are raw cell counts, and they are the wrong
denominator: background is trivially predictable, so masking it is close to
free. Decomposing the budget (`scripts/fair_compare.py`, 1,000 slices through
the production collator):

| arm | total hidden | anatomy hidden | background hidden | anatomy context | % retina hidden |
|---|---|---|---|---|---|
| envelope_default | 117.5 | 36.2 | **81.3 (69%)** | 6.0 | 77.3% |
| anatomy | 54.3 | **39.3** | 15.0 (28%) | 5.7 | **84.5%** |

On the axis that matters the two arms are closely matched, and where they
differ it is **against** us: anatomy hides 8.6% *more* retina, leaves 5% *less*
retinal context, and covers a larger fraction of the retina. Tissue-context per
tissue-cell predicted is **0.145 (anatomy) vs 0.166 (envelope)** — anatomy's
task is marginally harder.

The earlier "4.3x more context per predicted token" figure counted all tokens
including background and should not be used.

The real difference is budget efficiency: to hide a comparable amount of
retina, rectangles must waste **5.4x more budget on background** (81.3 vs 15.0
cells). Concentrating the mask on tissue is what the anatomy shape buys.

Note this also means `envelope_matched` (rectangles shrunk to anatomy's *total*
cell count) is **not** a valid control: it hides only 21.8 tissue cells (48% of
the retina) and leaves 18.4 tissue context, making it a substantially easier
task than either arm above.

**Does establish:** three epochs of anatomy-shaped masking, applied to weights
otherwise identical to the baseline and at matched anatomical difficulty,
produce a measurably better frozen representation for glaucoma classification.
The effect survives both test-set resampling and probe-seed variation, and the
arms are fully separated.

**Does not:**

- The effect size is small in absolute terms (+0.0054 AUC).
- Only three epochs of divergence. Whether the gap widens, holds or closes by
  ep100 is untested; ep50 is the next scheduled measurement.
- Single pretraining run per arm. Pretraining-seed variance is not measured and
  is the largest remaining uncertainty — it cannot be estimated from probe
  reruns.
- The anatomy guide used the α=0.50 adapter. This experiment does **not**
  isolate the adapter's contribution from the anatomy shape's; see
  `mirage_adapter.md` for the GOALS ground-truth evaluation showing the adapter
  does not improve segmentation. The frozen-MIRAGE-guide ablation is still
  outstanding.

## Precision audit (bug found and fixed)

`src/eval_downstream.py` called `autocast()` unconditionally at six sites.
Only feature precompute consulted `data.use_amp`, so every previously reported
downstream number ran the probe, the evaluation and the finetune paths in fp16
regardless of config.

Fixed by routing all six sites through a module-level `amp_ctx()` set once from
the config. The active precision is now printed at startup.

Measured impact on this checkpoint: **0.8582 (fp16) vs 0.8583 (fp32)** —
immaterial, so no previously reported result is invalidated by precision. The
config is now truthful, which matters for the fp32 requirement.

A second latent bug was fixed alongside it: the feature cache key was
`{split}_s{num_slices}` with no precision component, so an fp16 cache would be
silently reused by an fp32 run. The key is now
`{split}_s{num_slices}_{amp|fp32}`.

## Reproduction

```bash
python -m src.eval_downstream --config configs/frozen_meanpool_envelope_ep30.yaml
python -m src.eval_downstream --config configs/frozen_meanpool_anatomy_ep30.yaml
```

Predictions are written to `test_predictions.npz` in each output directory in a
common label order, so the paired bootstrap can be recomputed directly.
