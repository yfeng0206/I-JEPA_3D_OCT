# Anatomy-Guided Masking for I-JEPA — Findings

Branch `vlm-guided-masking`. All numbers measured on real data; every table below
is reproducible from the scripts named beside it. Nothing here is in `src/` yet —
training still uses random rectangles.

---

## 1. Headline

Replacing I-JEPA's random rectangular prediction targets with connected,
anatomy-shaped targets derived from a frozen MIRAGE segmentation model produces
**two** effects, not one. The second was not anticipated.

**(a) The targets land on anatomy instead of background.**

A target is *dead* if it contains no cell MIRAGE calls anatomy (score > 0.10) —
the encoder is asked to predict pure background. Over 1000 slices × 4 targets:

| | RANDOM rect | ANATOMY (+fallback) |
|---|---|---|
| dead targets | **14.12%** (565/4000) | **2.12%** (85/4000) |
| images where all 4 targets dead | 2.40% | 1.70% |

Random discards roughly one target in seven on background. The anatomy arm —
*including* the 2.3% of images where it gives up and falls back to random —
still fails 6.7× less often.

**(b) It also leaves the encoder MORE context, not less.**

This is the counter-intuitive result. Anatomy occupies a minority of an OCT
B-scan, so an anatomy-shaped target union is far smaller than four random
rectangles. Under the identical context policy and identical context block:

| | RANDOM rect | ANATOMY |
|---|---|---|
| context block before removal | 222.8 | 222.8 |
| target union removed | 122.6 | **43.8** |
| **context surviving** | 107.4 | **183.3** |

The two effects share one cause: spending the mask budget on anatomy means not
spending it on background. Concentration buys both precision *and* context.

The advantage survives batch-min truncation, and widens:

| batch | rect pre → post | anatomy pre → post | ratio |
|---|---|---|---|
| 1 | 105.0 → 105.0 | 177.1 → 177.1 | 1.69× |
| 8 | 108.1 → 80.2 | 181.9 → 156.7 | 1.95× |
| 16 | 107.4 → 72.3 | 182.2 → 145.0 | 2.01× |
| 32 | 109.5 → 73.1 | 181.7 → 124.4 | 1.70× |
| 64 | 107.7 → **66.1** | 182.1 → **124.1** | **1.88×** |

Variable-size targets are hit harder by truncation in principle, but rectangles
start from a much smaller surviving context, so the anatomy arm still ends up
ahead at every batch size.

`scripts/ctx_anatomy_probe.py`

---

## 2. Method

```
MIRAGE-Base@512
  image (B,1,512,512)  ──►  ViT-B + ConvNeXt decoder  ──►  H (B,384,64,64)
  H ──► Conv2d(384,4,1×1) ──► logits (B,4,64,64)
      ──► softmax ──► S = P_InnerRetina + P_Choroid
      ──► AvgPool 4×4 ──► (B,1,16,16) ──► DETACH ──► sampler ──► 4 masks
```

Softmax **before** pooling. The head is applied at 64×64, not to pooled features.

Sampler (`scripts/anatomy_target_sampler_v2.py`), per class independently:

1. `grow_region` — 8-connected growth confined to support `S = {score > τ}`,
   stopping at `mass_cap` of the class's probability mass.
2. `fill_small_holes` — cavities of ≤ 2 cells only.
3. `geodesic_partition` — farthest-point seeding + multi-source BFS.
   **Geometry only; the anatomy score is not consulted again.**
4. `rebalance` — adjacent-pair relaxation to `max_ratio = 1.25`.

Anatomy decides *which* cells are in the region; geometry decides *how* the
region is divided. Defaults: `mass_cap=0.80`, `τ=0.10`, `overlap=0.0`, n=4.

**The context policy is unchanged.** MIRAGE replaces `masks_pred` only. The
I-JEPA collator still draws its own context block and subtracts the targets.

Verified invariants over 1000 slices: 0 disconnected parts, 0 cells lost
(per-class part union == region exactly), 0 within-class overlaps, always
exactly 4 masks, `expand_overlap(0.0)` a true no-op.

---

## 3. Why `mass_cap = 0.80`

The budget is quoted in probability **mass**, and the confident core carries
most of the mass, so the cap controls how much *confident* anatomy the encoder
can still see. Recalibrated on that (300 slices, production collator, confident
:= score > 0.5):

| cap | union | context | confident cells in context | slices keeping ≥1 |
|---|---|---|---|---|
| 0.70 | 36.9 | 188.5 | 10.2 | 97.7% |
| 0.75 | 40.0 | 185.8 | 7.8 | 96.0% |
| **0.80** | **43.5** | **182.8** | **5.5** | **92.0%** |
| 0.85 | 47.3 | 179.4 | 3.5 | 69.7% |
| 0.90 | 52.1 | 175.3 | 2.4 | 35.7% |
| 0.95 | 58.4 | 169.6 | 2.2 | 22.3% |

The knee is a cliff between 0.80 and 0.825. Going 0.90 → 0.80 costs 8.6 target
cells and 7.5 context tokens but lifts the safe fraction from 35.7% to 92.0%.

**The previous 0.90 default was calibrated on the wrong criterion** — "context
never falls below `min_keep=10`, even at cap 0.99". That counts *tokens*, and is
satisfied by 176 tokens of black vitreous. Budgets must be calibrated on
anatomy **content** surviving in context.

The headline is threshold-sensitive and must be quoted with its cutoff. % of
slices whose context has zero tokens above threshold:

| cap | >0.3 | >0.4 | >0.5 | >0.6 | >0.7 |
|---|---|---|---|---|---|
| 0.80 | 1.3% | 2.3% | 8.0% | 34.7% | 60.3% |
| 0.90 | 6.0% | 31.3% | 64.3% | 75.7% | 79.0% |

The ordering across caps is stable at every cutoff; the magnitude is not.

---

## 4. Context quality at cap 0.80

| | RANDOM rect | ANATOMY |
|---|---|---|
| anatomy mass retained | 0.358 | 0.215 |
| ctx tokens with anatomy > 0.10 | 25.6 | **25.7** |
| ctx tokens with anatomy > 0.50 | 15.9 | 5.3 |
| inner-retina mass kept | 0.386 | 0.225 |
| choroid mass kept | 0.345 | 0.214 |
| zero anatomy tokens | 0.90% | 0.90% |
| < 5 anatomy tokens | 2.90% | **2.20%** |
| < 5% anatomy mass | 1.60% | **0.50%** |
| zero *confident* anatomy | 2.00% | 9.30% |
| zero inner retina | 0.20% | **0.00%** |
| zero choroid | 0.30% | **0.00%** |

Retina-token count matches random almost exactly (25.7 vs 25.6) while the
context is 70% larger. Tail risk is equal or better on every measure except
confident-anatomy, and the anatomy arm never strips a whole class.

Independently, on plain image statistics (`scripts/ctx_informative_probe.py`):

| | rect | anatomy |
|---|---|---|
| context tokens | 105.6 | 169.2 |
| informative (mean > 0.15) | 61.1 (57.9%) | 99.4 (**58.7%**) |
| variance retained | 38.1% | **58.4%** |

The extra context is *not* diluted background — it is marginally more
informative per token than the baseline's.

---

## 5. Failure modes and the fallback

Empty targets crash the collator: `torch.stack([t[:min_len] ...])` in
`multiblock.py` and `curriculum.py:1212-1218` raises
`stack expects each tensor to be equal size, but got [1] and [0]`.

Empty-target causes, 1000 slices:

| cause | rate | nature |
|---|---|---|
| `allocate` exceeded class capacity | 1.00% | **bug — fixed** |
| MIRAGE found no anatomy at all | 0.40% | real |

The bug: `allocate` used `present = mass > 1e-6`, and softmax mass is positive
almost everywhere, so a class with **zero** support cells still claimed a target
that was empty by construction. Slice 417: inner retina mass 4.88 / 10 support
cells, choroid 0.04 / **0** support cells → old rule allocated `[3,1]`.
`allocate` is now capacity-aware and gives `[4,0]`. Empties: 1.40% → 0.90%.

The residual 0.90% is images whose **total** grown capacity is < 4 cells — four
targets cannot exist. Gate on capacity, not on presence:

| `min_cells` | fallback rate | empties reaching collator |
|---|---|---|
| 2 | 1.20% | 0 |
| 3 | 1.90% | 0 |
| **4** | **2.30%** | **0** |
| 5 | 2.60% | 0 |

**Fall back to random rectangles, not to the geometric "oracle" ribbon.**
Tested on a real degenerate crop (`data_08569/slice_199`, near-black scan whose
only retina is a faint sliver at the left edge):

| | oracle ribbon | anatomy |
|---|---|---|
| cells masked | 70 | 18 |
| mean brightness inside | 0.2343 | **0.2847** |
| anatomy score captured | 46.2% | **63.4%** |

MIRAGE localised the sliver correctly; the anatomy mask was 1.52× brighter than
its surroundings. The oracle is confined to the central `oracle_lateral_frac` of
columns (3..12 of 16) and the retina was at columns 0..5 — degenerate crops put
anatomy at the **edge**, exactly where a centred band cannot look.

Random rectangles are also the honest fallback: they are the baseline arm, so a
fallback image is scored under the control condition rather than a third,
untested mask distribution.

---

## 6. Blockers before training

**B-1 `global_min_pred` truncation — fatal.** `curriculum.py:1212-1218`
truncates every target in the batch to the single smallest. With variable-size
anatomy targets at batch 64, `global_min_pred == 1` in **69.2%** of batches and
only **15.3%** of intended target area survives — when it does not crash first.
Fix: fixed target cardinality per microbatch. Padding plus a loss mask is
**incorrect** — padded tokens still participate in self-attention and the
attention implementation has no padding mask.

**B-2 In-loop cost.** Sampler 5.30 ms/image = 339 ms per batch of 64, a 3.0
steps/s ceiling from masking alone. MIRAGE-Base@512 adds ~0.43 s/batch
(148.7 img/s). At batch 64 × accum 8 that is ~6.1 s of added overhead per
optimizer step. **Mitigation: generate masks from frozen H₀ and precompute
them**, which also removes the adapter-drift feedback loop.

**B-3 Paired views.** The dataset returns one `(3,256,256)` ImageNet-normalized
tensor. MIRAGE needs the *same crop* at 512×512, 1-channel, raw min-max — no
ImageNet normalization. Needs a paired-view dataset branching before normalize.

**B-4 Residual target-size risk.** The capacity gate removes empty targets but
does not guarantee a per-target minimum; a 1-cell target is still reachable from
a thin region tail, and one such target drags the whole batch down under B-1.

**B-5 Confounds.** Anatomy targets average 43.8 union cells vs ~122.6 for
rectangles, and context differs 183.3 vs 107.4. Any result could be caused by
target area, context amount, or shape. Required controls: random *connected*
blobs matched for cardinality, and anatomy masks matched for context size.

---

## 7. Rejected designs

**D1 pixel-level gate `s ⊙ x`.** `apply_masks` gathers before any transformer
block and the target branch runs under `no_grad`. Measured `∂L/∂s = 0.0` on
0 of 1360 patches; after 200 Adam steps `max|s−0.5| = 0.00e+00`. Independently,
gating both branches makes a blank image *cheaper* (L=0.2143) than the real one
(L=0.2893).

**D2 token gate + straight-through top-K.** `∂L/∂q = e·h(1−h)/T ≥ 0` because
smooth-L1 error `e ≥ 0`, so the gate is always pushed toward hiding whatever is
already easiest. Measured `corr(∂L/∂q, error) = +0.24 … +0.93` at every
temperature. Triple bind: low T gives faithful STE but only 26% live gates;
high T gives degeneracy +0.93 and a 29% STE gap.

**D3 detached hard mask — ACCEPTED.** `∇_MIRAGE L_JEPA = 0` by construction.

**Extent-based budget — inert.** Three formulations moved coverage by
0.001–0.011; the mass cap always binds first. Unconfined extent-greedy growth
consumed 255 of 256 cells.

Also rejected: rank-cut geodesic ordering (29.7% disconnected), largest→smallest
rebalancing (spread 3.54; chunks form a chain), constrained projector for
`L_sem` (233% feature churn, 0.0% gain).

---

## 8. Bidirectional distillation — NOT validated

`L_rel = ‖ĤĤᵀ − sg(ẐẐᵀ)‖²_F / N²` on 256×256 Gram matrices removes the
384→768 projector that absorbed 97% of the pointwise `L_sem` gradient.

But it **cannot be claimed to improve segmentation**. The frozen 1×1 head maps
384 channels to 4 directions; the adapter is free to move the other ~380:

| λ | L_rel reduced | feature drift | argmax agreement |
|---|---|---|---|
| 0.1 | 57.6% | 0.814 | 0.9997 |
| 10 | 89.3% | 1.866 | 0.9991 |
| no anchor | 93.7% | 1.028 | **0.3754** |

Features move 81–187% while segmentation output is essentially unchanged — the
signature of learning entirely in the head's nullspace. The anchor is
load-bearing: without it, agreement collapses to 0.375.

Confirmed on the masks themselves (real MIRAGE features, head at 64×64):

| adapter LR | feature drift | argmax agreement | mask Jaccard |
|---|---|---|---|
| 1e-5 | 0.339 | 0.9986 | 0.9942 |
| 1e-4 | 1.375 | 0.9999 | 0.9990 |
| 1e-3 | 2.833 | 0.9998 | 0.9980 |

A 100× LR range moves features 0.34 → 2.83 while masks stay ≥ 0.994 identical.

**Optimizer settings implied by measurement:**

- Adapter LR **1e-4** — 1e-5 is not safer (worse Jaccard, barely learns).
- Segmentation head LR **0, frozen** — under `L_rel` the head receives
  **exactly 0.000e+00** gradient, and FairVision has no anatomy GT so there is
  no `L_seg` either. With a logit anchor added, a *trainable* head is actively
  harmful: it can satisfy the anchor by rotating itself while features drift.
- Backbone LR **0**, and keep MIRAGE in `.eval()` — it is constructed with
  `drop_path_rate=0.1` and freezing parameters does not disable stochastic depth.
- Adapter dropout **0.0** — note the reason is *not* mask instability. That was
  predicted and measured false: mask Jaccard was 1.0000 across dropout
  0.0/0.1/0.3 in both `eval()` and `train()`, because the zero-init residual is
  small relative to H₀. The reason is simply that there is nothing to regularise.

Before any claim here, decide whether the target is better *segmentation*
(needs a head-sensitive/logit-space objective plus labelled GOALS replay) or
better *MIRAGE features* (needs a downstream eval, and the segmentation claim
retracted).

**Claim discipline:** with `L_seg` only, say *"MIRAGE continues improving via
segmentation training while guiding JEPA"* — never *"JEPA improved MIRAGE."*

---

## 9. Corrections to earlier claims

Recorded so they are not repeated.

| claim | status |
|---|---|
| "I-JEPA scattered ablation 54.2 → 19.2" | **wrong**; 19.2 is not in that table. Real: random patches 17.6 vs multi-block 54.2 |
| "`L_sem` is harmful" | **wrong**; `cos(∇L_anchor, ∇L_sem) = −0.076`, segmentation bit-identical at λ=1.0 |
| "ViT-L interpolates pos_emb at its own resolution" | **wrong**; neither checkpoint does |
| "57.6–89.3% gradient share reaches MIRAGE" | **wrong**; that JSON key is `L_rel_reduction_pct`. Only the adapter was trainable, so share was trivially 100% |
| "informative context: anatomy's extra tokens are diluted (48.7% vs 58.3%)" | **wrong**; produced by an `np.argwhere(...).flatten()` index bug. Corrected: 58.7% vs 57.9% — *not* diluted |
| "84.2 context tokens is what the collator gives" | **incomplete**; that is post-truncation at batch 8. Per-image it is ~100–105 |
| C3 verified by `p.grad is None` | proves nothing on frozen params; now asserts `H.grad_fn is None` |
| "dropout would make masks stochastic" | **predicted, measured false** (Jaccard 1.0000) |
| void class 3 invalidates the anatomy score | **overstated**; P(void) mean 1.1e-5, corr 0.99999997, 1 cell of 25 600 crosses τ. Worth fixing, does not move any measurement |

Bugs found in probe code and fixed this session: flat-index corruption;
mismatched context definitions between arms; unseeded crops (torch RNG, not
numpy); `try/except ImportError` swallowing sampler failures; `torch.manual_seed`
used to control a collator that draws from Python's `random`.

---

## 10. Reproduce

```
scripts/anatomy_target_sampler_v2.py   the sampler (build_targets, is_viable)
scripts/ctx_anatomy_probe.py           §1, §4  anatomy content in context
scripts/ctx_informative_probe.py       §4      image-statistics context quality
scripts/relational_distill_probe.py    §8      L_rel, anchor, λ sweep
scripts/b2_predictor_probe.py          plumbing gate — see caveat below
scripts/demo_v2_masking.py             figures
```

Grids: `D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz` — `per`
`(1000,2,16,16)` = `[P_inner, P_choroid]` on the 16×16 JEPA grid.

Environment: `D:\jepa_phase0\.venv` (scipy, matplotlib, torch+CUDA). MIRAGE
builds and forwards inside this venv; `timm` is not required. Needs
`sys.path.insert(0, MIRAGE_WS/'MIRAGE')` and `os.chdir(MIRAGE_WS)` before
`fm_seg_config`.

**Caveat on `b2_predictor_probe.py`:** it overfits a single fixed batch for 600
steps (loss 0.463 → 0.008). That is a plumbing/shape gate only. A degenerate
blurry-average predictor would show the same curve, so it is **not** evidence
that the pretraining task is learnable. Testing that needs held-out slices and a
mean-predictor control.

Two known probe-vs-production discrepancies: cached `feats.npz` is pre-pooled to
16×16, so `relational_distill_probe.py` applies the head *after* pooling
(production applies it at 64×64 before softmax); and it uses MIRAGE-Large@1024
with a JEPA checkpoint that was itself trained with MIRAGE-guided masks, which is
circular. Re-run on Base@512 with an independent frozen JEPA teacher before
drawing conclusions.
