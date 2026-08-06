# Phase 0-1: MIRAGE-guided semantic multi-block masking

**Status:** investigation complete, no pretraining launched. The proposed A'
design is REJECTED on measured grounds; a cheaper and better alternative
(block anisotropy) was found. See TL;DR.

Companion figures: `results/masking/phase1/`. Full artifact bundle (layer
dumps, matrix dumps, occupancy npz) is attached to the `phase1-masking`
release, not committed.

Session of 2026-08-05/06. Everything below is measured; every number traces to a
script that was run, and the script is named. Nothing was committed to git and
no training was launched.

---

## TL;DR — read these five

1. **MIRAGE-512 fine-tune works.** GOALS fg Dice 0.9603 vs 0.9688 at 1024
   (−0.0084 all-class, −0.0140 choroid). It passes your "within 0.01" gate on
   the aggregate but not on choroid, and it wins on only 1/30 images.
2. **512 passes the sampler-equivalence gate — but ONLY after recalibrating the
   occupancy threshold from 0.25 to 0.0868.** At the inherited 0.25 the feasible
   rate collapses 0.9950 → 0.8600, i.e. 14% of slices silently fall back to
   uniform random placement. After recalibration every geometry metric matches
   1024 within noise.
3. **A′ (target = block ∩ anatomy) must not be run as specified.** Adversarial
   review found, and I verified, that it **crashes the collator** (empty
   intersections occur; `torch.stack` raises; ~19% of batches at r_t=1.0) and
   that when it does not crash the batch-min truncation at
   `curriculum.py:1210` delivers only **~45 target cells** against the
   comparison arms' ~102–112. It also hands the predictor the segmentation for
   free via target positional embeddings. Root cause is geometric: **target
   blocks (height ~6.8 cells) are taller than the retinal band (~4.9–5.6)**, so
   A′ and "whole-retina masking" are the same thing by identity. Full detail
   and revised plan in §5b.
4. **Cheapest next run is not a new arm — it is a second seed of an existing
   one.** Every ladder conclusion rests on n=1 run per arm and the decisive
   effect is 0.0047.
5. **A better lever than A′ was found and measured: block SHAPE.** The retina is
   a horizontal band ~4.9 cells tall; I-JEPA's default `aspect_ratio
   [0.75, 1.5]` draws blocks ~7.0 cells tall, so most target blocks are taller
   than the entire band. Making blocks wide and flat **at identical block area**
   improves every metric at once — see §5c. This is a one-line config change
   that preserves I-JEPA's block prior, its target area, and comparability.

---

## 1. V1 → V2 → V3, the segmentation arms

Terminology: **V1** = out-of-the-box MIRAGE (GOALS-only, 4-class RNFL/GCIPL/
Choroid). **V2** = merged 3-class fine-tune (GOALS+Duke+AROI, real labels only).
**V3** = V2 + a bounded synthetic choroid band below Bruch's membrane.

### 1a. What V2 changed, and what it cost

Training data, measured from the built dataset:

| source | images | share | choroid px | sub-BM region |
|---|---|---|---|---|
| AROI | 1,105 | 88.5% | 0 | IGNORE |
| Duke_DME | 88 | 7.1% | 0 | IGNORE |
| GOALS | 55 | 4.4% | 4.97% | choroid |

100% of choroid pixels come from 55 GOALS images; choroid is 0.68% of all
pixels. Sampler was on (`balance_sources=True, source_share=['GOALS=0.25']`,
read from the checkpoint args).

Result on FairVision transfer (n=5 slices, raw argmax, no repair):

| | V1 | V2 | V3 |
|---|---|---|---|
| inner n_components | 7.00 | **5.40** | 5.80 |
| inner largest share | 0.7595 | **0.8667** | 0.7682 |
| inner col coverage | 0.9479 | 0.9486 | **0.9699** |
| choroid area | 0.0799 | **0.0086** | 0.1180 |
| choroid col coverage | 0.9613 | **0.2611** | 0.9500 |

**V2 improved the inner band and destroyed the choroid on transfer** (−89.2%
area, coverage 0.9613 → 0.2611).

The cause is *not* under-supervision: on GOALS test (real ground truth) V2's
choroid Dice is **0.9365, better than V1's 0.9278**. It fails only out of
domain. Mechanism: in 75% of sampled batches (AROI/Duke) the sub-BM region is
IGNORE — zero gradient — while everything else above BM is `Elsewhere`. So
three of every four steps teach a strong "not-inner ⇒ Elsewhere" prior with no
choroid counterexample, and 100% of the positive choroid evidence comes from one
device/protocol. The detector memorises GOALS appearance.

Verified the pixels really do become `Elsewhere`, not the void class: raw argmax
over all four logits gives `Background/void = 0.0000` on every FairVision slice
tested. The void class is never emitted.

**Lesson worth keeping: ignore-masking prevents a wrong label but does not
prevent a competing prior in the surrounding context.**

### 1b. What V3 changed

One flag: `--choroid-band`, calling `add_choroid_band()` in
`scripts/build_seg_merged_v2.py`. It relabels the top slice of the sub-BM IGNORE
region as choroid, bounded to the top `frac` of image height, and only where the
pixel is *currently* IGNORE (so cyst/PED/SRF labels are never overwritten).

| training labels | V2 | V3 |
|---|---|---|
| images with any choroid label | 55 / 1248 | **1248 / 1248** |
| choroid pixel share | 0.68% | 5.38% |
| ignore share | 43.88% | 39.18% |

Synthetic band is 5.44% for AROI/Duke vs GOALS' real 4.97% — well calibrated.

Result: choroid transfer fully repaired (coverage 0.2611 → 0.9500) and V3 is the
best arm on GOALS ground truth (all-class 0.9683 vs baseline 0.9634, winning
25/30 images).

Residual defect: the band runs too deep. Top edge is fenced by Bruch's membrane
in all three sources and lands within 0.011 of ground truth; the bottom edge is
fenced only by GOALS' 25% sampler share, so it over-extends (thickness 0.1406 vs
GT 0.0500). This is the V4 target and remains parked at your instruction.

**Artifacts:** `outputs\mergedv2-eval\v1_vs_v2.png`,
`outputs\mergedv3-eval\v1_v2_v3.png`, `outputs\mergedv3-eval\raw_native.png`
(pure argmax, no fusion/suppression/repair — V1 reproduction verified identical
to the published preview to 4 decimals).

---

## 2. What is actually trainable in MIRAGE

Compared every weight between V1 and V2:

| group | identical | changed | params |
|---|---|---|---|
| **encoder (ViT-L, 24 blocks)** | **288 / 288** | **0** | 302,309,376 |
| decoder / ConvNeXt head | 1 | 38 | 13,203,460 |
| global_tokens | 0 | 1 | 1,024 |

**The encoder is bit-for-bit unchanged. We trained 3.85% of the model.** The
frozen MIRAGE features are good enough to reach 0.9683 Dice through a
1,540-parameter classifier — which is the strongest argument that the encoder,
not the segmentation head, is the valuable asset.

### Verified layer-by-layer trace (forward hooks, not config reading)

```
input                (1,1,1024,1024)
 patchify conv 32x32 -> (1,1024,1024)      1,049,600  trained
 + fixed sin-cos pos_emb (1,1024,32,32)    1,048,576  FROZEN
 + 1 global token    -> (1,1025,1024)
 ENCODER ViT-L       -> (1,1025,1024)    302,309,376  FROZEN
 drop global token   -> (1,1024,1024)
 proj_dec Linear     -> (1,1024,6144)      6,297,600  trained
 reshape 6144=16x384 -> (1,384,128,128)
 4x ConvNeXtBlock    -> (1,384,128,128)    4,806,144  trained
 final_layer 1x1conv -> (1,4,128,128)          1,540  trained   <- REAL resolution
 F.interpolate x8    -> (1,4,1024,1024)            0
 [model returns here]
 argmax(dim=1)       -> (1,1024,1024)
```

Each ConvNeXt block: `dwconv 7x7 groups=384 → permute → LayerNorm → Linear 384→1536
→ GELU → Linear 1536→384 → permute → + residual`. Layer-scale disabled
(`gamma=None`), drop_path = Identity, **no ReLU anywhere, no BatchNorm, and no
softmax or sigmoid in the entire model** (`any Softmax/Sigmoid module: False`).

Two consequences that matter for the masking plan:

* **True prediction resolution is 128×128, not 1024×1024.** The ×8 upsample adds
  no information: computing argmax before vs after it changes only **1.088%** of
  pixels (11,405 / 1,048,576) and class areas by <0.0005. At the 16×16 mask grid
  that difference cannot survive pooling. The ×8 can be skipped entirely.
* `argmax` is standard MIRAGE practice (`run_seg_tuning.py:1157`), not something
  we invented. What *we* added is the void-suppression line immediately before
  it, which stopped the ignore channel winning the argmax and corrupting
  best-checkpoint selection.

**Artifacts:** `outputs\layer-dump\{V1,V3}_data_07266_slice199\` (19 PNGs each +
manifest), `outputs\matrix-dump\{V1,V3}_data_07266_slice199\` (CSV/TXT matrices
of the final three stages, plus `trace_column*.txt` showing all four logits and
the winner for every pixel down one column).

---

## 3. MIRAGE-512 — the resolution question

**Finding that made this worth doing:** MIRAGE-Large's *native* `pos_emb` is
`(1,1024,16,16)` — a 16×16 token grid, i.e. **512×512 at patch 32. MIRAGE was
pretrained at 512.** Our GOALS/V2/V3 fine-tunes all ran at 1024, so the
pretrained embedding was interpolated *up* 2×.

**Hazard, learned the hard way:** the checkpoint's `pos_emb` is NOT regenerable
sin-cos despite `learnable_pos_emb=False` — it differs from a fresh table by max
2.003. My first 512 measurement dropped it and regenerated; the 1024 control
then scored Inner 0.606 against a known 0.9686, which is how I caught it. It
must be **interpolated from the checkpoint**. Bicubic vs bilinear made no
difference (0.8847 vs 0.8849).

Zero-shot 1024→512 transfer: Inner −0.084, Choroid −0.146. Fine-tuned at 512
(200-epoch run, same recipe, `--input_size 512`):

| ep35, GOALS GT, n=30 | Inner | Choroid | all |
|---|---|---|---|
| V3 @1024 | 0.9691 | 0.9420 | 0.9688 |
| V3 @512 | 0.9591 | 0.9280 | 0.9603 |
| delta | −0.0100 | **−0.0140** | −0.0084 |

512 wins on 1/30 images. Fine-tuning recovers most of the zero-shot gap but not
all of it. The 512 run peaked at ep34 — the same epoch as V2 and V3 at 1024,
which is now three independent confirmations.

Speed: 167.0 img/s vs 41.7 (4.0×) inference; ~63 s/epoch vs ~108 (1.7×)
fine-tune.

---

## 4. Sampler-equivalence check (your requested gate)

`scripts/sampler_equivalence_dump.py` + `scripts/sampler_equivalence_replay.py`.
200 held-out FairVision Test slices, driving the **production sampler**
`CurriculumMaskGenerator._sample_mirage_blocks` with identical block sizes,
flags and RNG seeds per arm. Purity/visibility are scored against an
arm-independent reference (I first wrote it circular — measuring purity against
each arm's own threshold — which inflated 512's purity for free; fixed).

Mean occupancy differs a lot: **1024 → 0.1851, 512 → 0.1415** (−24% relative).
So the inherited threshold is wrong for 512.

| metric | 1024 | 512 @0.25 | 512 @cal | cal − 1024 |
|---|---|---|---|---|
| accept rate | 0.3350 | 0.3200 | 0.2600 | −0.0750 |
| **feasible rate** | 0.9950 | **0.8600** | **0.9950** | **+0.0000** |
| mean block fill | 0.4818 | 0.4544 | 0.4878 | +0.0060 |
| retina visible | 0.2276 | 0.2448 | 0.2217 | −0.0058 |
| unique target cells | 116.14 | 105.60 | 116.90 | +0.76 |
| unique cells (A′) | 50.12 | 48.96 | 50.36 | +0.24 |
| target purity | 0.4316 | 0.4651 | 0.4330 | +0.0013 |
| guided blocks /4 | 3.99 | 3.74 | 3.99 | −0.01 |

Calibrated threshold: **0.0868** (area-matched to 1024@0.25).

**Verdict: 512 passes, conditional on recalibration.** At the inherited 0.25 the
feasible rate drops 13.5 points — 14% of slices would silently fall back to
uniform random placement, which would quietly contaminate the arm. After
recalibration every geometry metric matches within noise.

Two caveats:
* Per-slice block *positions* rarely coincide (0.5%), and target-union IoU is
  0.61. Much of this is an RNG artifact — the sampler picks uniformly among
  admissible windows, so any change in the candidate count shifts the index. The
  distributions match even though individual draws do not.
* The 0.335 accept rate in this table is measured on the **uncropped** full
  slice. Under the real pipeline, with `PairedRandomResizedCrop` applied, the
  accept rate is much higher — see §5, where the raw soft guide reaches 0.6167.
  Anatomy occupies a larger fraction of a crop than of the whole slice, so the
  four blocks leave proportionally more of it visible.

---

## 5. Phase-1 implementation and the A′ problem

### What was built (all CPU-verified, nothing launched)

* `scripts/mirage_soft_guide_dump.py` — raw MIRAGE anatomy probability at native
  200×200, uint8, **no envelope repair**.
* `scripts/demo_guided_masking.py` — full one-shot pipeline using the production
  sampler and production paired crop.
* `scripts/ap_intersection_cost.py` — measures what A′ does to target geometry.

**Justification for dropping `repair_union`:** the mid-retina taxonomy hole is
present in 100% of GOALS columns with mean height 0.0504 of the image, but a
16×16 mask cell spans 0.0625. Measured on real ground truth: of 1,695 cells
lying between anatomy, exactly **1 (0.06%)** is empty. The hole closes by itself
at mask-grid resolution, so the repair — the only non-differentiable step in the
guide path — is unnecessary. This is a genuine simplification and it also
removes the obstacle to a future differentiable phase.

**Verified end-to-end, 300 masks, against the production guide.** Dropping the
repair costs nothing once the threshold is recalibrated:

| guide | accept | feasible | retina visible | admissible area |
|---|---|---|---|---|
| repaired envelope (production) | 0.5800 | 1.0000 | 0.2498 | 0.3311 |
| **raw soft @0.0868 (new)** | **0.6167** | 0.9967 | 0.2499 | 0.3081 |
| raw soft @0.25 (uncalibrated) | 0.4800 | 0.9900 | 0.2327 | 0.2621 |

The raw soft guide at the calibrated threshold **accepts more often** than the
repaired envelope (0.6167 vs 0.5800) with identical retina visibility, and only
a slightly smaller admissible region. At the uncalibrated 0.25 it is clearly
worse — one more reason the threshold recalibration is mandatory, not cosmetic.

**Architecture simplification worth flagging:** because MIRAGE is frozen, the
guide can stay *precomputed and then cropped* with the same paired rectangle,
exactly as the current pipeline already does. In-loop MIRAGE is only required
once MIRAGE becomes trainable. That means **Phase 1 costs zero extra per-epoch
time** — not the 1.0–4.5 h/epoch estimated earlier — and it is arguably more
faithful, since MIRAGE then sees the full slice it was trained on rather than a
30%-scale crop. The existing mask cache stores only `hard_masks`, so a one-time
probability precompute is needed (~1 h at 512, ~4 h at 1024, over 600k slices).

### The A′ measurement — 300 masks, 12 slices × 25 crops

| metric | rectangles (plain A) | A′ targets |
|---|---|---|
| target cells (of 256) | 124.30 | **58.88** |
| fraction of frame | 0.4855 | **0.2300** |
| connected components | 1.08 | 1.53 |
| largest-component share | 0.9677 | 0.8865 |
| accept rate | 0.6167 | — |
| feasible rate | 0.9967 | — |

* A′ keeps **47.4%** of the rectangle area. Consistent across slices: 48.7%,
  48.9%, 50.0%, 54.7% on the four demo cases.
* Fragmentation is real but **mild** — 1.53 components, 88.7% of area in the
  largest blob. This is not "scattered patches"; the intersection of a rectangle
  with a thick band is mostly still one piece.
* **Anatomy is only 78.9 / 256 cells (30.8%) of a crop, and A′ already masks
  75% of it.**
* Compensating with bigger blocks **fails**: `pred_mask_scale` 0.15–0.2 →
  0.317–0.422 grew rectangles 124.3 → 152.7 but left A′ at 58.9 → 59.1, and
  feasible rate fell 0.997 → 0.807. **A′ area is capped by anatomy area.**

### Why I think this is a problem, not a detail

The existing random/oracle/MIRAGE arms use **~102–112 unique target cells**. A′
tops out at 78.9 and sits at 58.9. So an A′ arm differs from its own baselines in
**target area by roughly a factor of two** — and target area is precisely the
**B2** confound that already makes the existing MIRAGE-vs-oracle result
unattributable, compounded by the newly-found **spread** confound.

Concretely: if an A′ run comes back worse, we cannot tell whether semantic
targeting hurt or whether halving the masked area hurt. I-JEPA's own ablation
says target size matters enormously (54.2% → 19.2% when small blocks are
allowed). If it comes back better, same problem inverted.

There is also a definitional issue: A′ masking 75% of all anatomy is close to
option B (whole-retina masking), which you explicitly rejected in favour of four
blocks. Pushing area up pushes A′ further toward B.

Finally — and this deserves scrutiny — the **"zero background target patches"
requirement appears to have originated in the design discussion rather than from
you or from any measurement.** The production design already enforces a softer
version (`mirage_min_block_fill: 0.40`: each block must be ≥40% on-region). I-JEPA
targets are *representations*, not pixels, so a block containing some vitreous is
not obviously harmful.

### Options, with the numbers

| option | target cells | background targets | area-comparable to existing arms? |
|---|---|---|---|
| **A (rectangles)** | 124.3 | ~53% of cells | **yes** (~102–112) |
| **A′ (∩ anatomy)** | 58.9 | 0 | **no — ~2× smaller** |
| A′ + bigger blocks | 59.1 | 0 | no, and feasible 0.997→0.807 |

**My recommendation: DO NOT run A′ as specified.** See §5b — adversarial review
found a verified crash and a verified target-collapse mechanism, plus three
confounds I had missed. Revised recommendation at the end of §5b.

---

## 5b. Adversarial review — what a code review and a rubber-duck found

Both were run against this design tonight. I verified their central claims
rather than taking them on trust; the important ones are true.

### VERIFIED BLOCKER — A′ crashes the collator, and collapses targets when it doesn't

`src/masks/curriculum.py:1210-1211`:

```python
global_min_pred = max(1, min(t.numel() for group in masks_pred for t in group))
collated_masks_pred.append(torch.stack([t[:global_min_pred] for t in group], dim=0))
```

The target length is truncated to the **minimum over all 4 blocks × all 64
images in the microbatch**. Rectangles have an implicit floor (`num_target ∈
[38,51]`), so this is survivable today. A′ removes that floor. Measured over
1,200 guided blocks:

| per-block target size | mean | min | p1 | p10 |
|---|---|---|---|---|
| rectangles | 44.6 | 35 | 35 | 40 |
| **A′** | **22.9** | **0** | **15** | **17** |

* **Empty A′ blocks occur** (1/1200 = 0.08% at r_t = 1.0). Reproduced the
  consequence directly:
  `RuntimeError: stack expects each tensor to be equal size, but got [1] at entry 0 and [0] at entry 1`.
  With 64×4 = 256 draws per batch, P(≥1 empty) ≈ **19% per batch**. During the
  curriculum ramp, where `biased_flags` is Bernoulli(r_t) and non-guided blocks
  are placed uniformly — frequently missing a band only ~5 cells tall — it is
  far worse.
* **When it does not crash, it silently collapses.** Simulating the batch
  minimum over 256 draws: expected `global_min_pred ≈ 11.3` against a mean
  per-block A′ of 22.9. So A′ delivers **~49% of its already-halved targets**:
  ~45 cells actually reaching the predictor, versus ~102–112 in the comparison
  arms. That is a **2.3× area deficit**, not the 1.7× I reported in §5.
* `_block_to_indices` returns row-major-sorted indices, so truncation deletes
  the **last rows** — for an A′ intersection, systematically the *deepest*
  anatomy, i.e. the choroid. That is precisely the tissue §6 identifies as 96.8%
  of MIRAGE's purity gain.

My `run_once` never exposed this: it hardcodes `[True] * npred` and runs one
image, exercising neither the ramp path nor the batch-min path.

### Three confounds I had missed

* **A′ is not a location-only change.** Against the shipped arms it changes the
  guide on five axes at once: model (GOALS-tuned → merged V3), taxonomy
  (RNFL/GCIPL/Choroid → InnerRetina/Choroid), representation (bit-packed binary
  → soft probability), repair (frozen fingerprint `9a25a2cdb36f9cba` → none),
  and threshold (0.25, chosen by a 1,000-volume sweep → **0.0868, which I
  derived tonight and which has no sweep, no test and no config provenance**).
  Even a clean AUC number could not be attributed to the intersection.
* **A′ leaks the segmentation into the predictor.** `vision_transformer.py:552`
  does `pos_embs = apply_masks(pos_embs, masks)` — the predictor is conditioned
  on the positional embeddings of exactly the target indices. Under rectangles
  those carry no anatomical information. Under A′ **the shape of the target
  index set is the segmentation**, handed over free every step. Since the
  predictor is discarded at evaluation, this *reduces* pressure on the encoder —
  the same "easier pretext task" mechanism already blamed for MIRAGE's loss.
* **No seed-variance estimate exists.** Every ladder conclusion rests on n=1
  pretraining run per arm, and the decisive result is Δ = −0.0047 with a CI that
  clears zero by 0.0004. That CI is test-sample uncertainty with the encoder
  held fixed; it contains no run-to-run term.

### The geometric reason all of this happens

`_sample_block_size`: `num_target ∈ [38,51]`, `ar ∈ [0.75,1.5]` → **block height
5.4–8.2 cells, mean ≈ 6.8**. The retinal band is 78.9 cells over ~14–16 columns
→ **height ≈ 4.9–5.6 cells**.

**Almost every target block is taller than the entire retinal band.** Therefore:

* The purity ceiling for a perfectly placed block is ≈ `4.9/6.8 ≈ 0.72`. Every
  arm measured sits at 0.458–0.506, i.e. 63–70% of that ceiling, and they differ
  from each other by 0.02–0.05. **The intervention is close to saturated.**
  (Independent check: my plain-A on-region rate 58.75/123.81 = 0.474 vs the
  production training log's 0.462 — the new pipeline reproduces production.)
* Intersecting a block taller than the band with the band leaves a strip of band
  height; do that four times with `spread` and you have reconstructed the band.
  **A′ ≡ option B is a geometric identity, not a slippery slope.**

### Also worth recording

* The "zero background targets" requirement is **nominal at this grid
  resolution**: at threshold 0.0868 a cell counts as anatomy when its *mean*
  probability is 8.7%, so an admitted cell can still be >90% background pixels.
  It also contradicts the swept `mirage_min_block_fill: 0.40`, which explicitly
  permits 60% background per block. And the arm ordering (purity 0.4530 →
  0.8746, 0.5602 → **0.8855**, 0.6320 → 0.8807) is **non-monotone and already
  turning down**; A′ extrapolates along the negative segment.
* Domain argument against it: glaucoma is RNFL *thinning* — a shift in the
  tissue/void boundary. A′ deletes every boundary-straddling target cell, which
  may remove the most glaucoma-relevant supervision in the pretext task.
* **Dead patches.** Context is the complement of the *rectangle* union while
  targets are the *intersection*, so 123.8 − 58.8 = **65 cells (25% of the
  frame)** are in neither the encoder input nor the loss — an uncontrolled third
  category no existing arm has.

### Defects in my own measurement code (from the code review)

* **Crop RNG and block-size RNG were seeded identically**, so crop scale/aspect
  correlated with block scale/aspect. Fixed; re-ran; the effect was negligible
  (A′ retention 47.4% → 47.5%, area 124.3 → 123.8), so §5's conclusions stand.
* **The §4 equivalence check pooled the full slice and never applied a crop**,
  whereas production crops first and pools second. The 512-vs-1024 conclusion is
  therefore about the *guide*, not about training-time placement. The later
  repair-vs-soft comparison in §5 *does* use the production crop, and 512 held
  up there too — but a crop-aware redo of §4 is owed before launch.
* In the replay, A′ cells were constructed from the arm-independent reference
  rather than each arm's own thresholded grid, so `unique cells (A′)` slightly
  overstates agreement.
* `decoder_stage_dump.py` builds its union with `hard > 0`, which for merged
  models includes class 3 (void). Harmless in practice — void is predicted
  0.0000 everywhere — but wrong in principle.
* Same file drops `enc[0]` as the global token; MIRAGE appends it **last**, so
  it should be `enc[:-1]`. Affects only the token-norm visualisation panel.

### Revised recommendation

1. **Do not run A′.** It crashes ~19% of batches, and where it does not it
   delivers ~45 target cells against the arms' ~102–112 while handing the
   predictor the segmentation for free.
2. **Cheapest decisive run: a second seed of an existing arm.** Re-fork from the
   same ep25 checkpoint, change only the RNG seed, 100 epochs (~3 days). One
   arm's cost makes all three arms already paid for interpretable. Five-minute
   precursor: re-run the frozen MeanPool probe on the *same* encoder with 5 head
   seeds to bound probe noise.
3. **Then the control this program already prescribed for itself:** the shipped
   MIRAGE arm re-run with `mirage_spread: false`, the one intervention measured
   to restore area parity (100.9 vs 100.6, +0.4% instead of +7.8%).
4. **If zero-background must be tested,** do it as a paired 2-arm design at
   matched delivered area — A′ vs plain rectangles from the *same* guide with
   `pred_mask_scale` reduced to match — so that X−Y isolates zero-background and
   Y−(existing arm) isolates area. Do not shorten these runs: the existing table
   inverts sign between ep50 (+0.0020) and ep100 (−0.0047).
5. **Before any of it, one cheap calculation worth doing:** the geometric purity
   ceiling as a function of `pred_mask_scale` × `aspect_ratio` × per-slice band
   height. Given the ≈0.72 estimate above and arms clustered at 0.46–0.51, it may
   show the masking ladder has ~1 bit of headroom left and is finished.

---

## 5c. The anisotropy finding — a better lever than A′

Chasing down *why* A′ degenerates produced something more useful than A′ itself.

The retinal band is **4.94 cells tall** (p10 4.06, p90 6.00) on the 16×16 grid.
I-JEPA's default `aspect_ratio: [0.75, 1.5]` — inherited from the ImageNet
recipe, where objects are roughly isotropic — draws blocks **7.04 cells tall on
average**. Most target blocks are therefore taller than the entire structure
they are supposed to land on, and the excess height can only ever be background.

Best-case purity for a perfectly placed block, computed on the real occupancy
grids (upper bound for *any* placement policy):

| block height | purity ceiling |
|---|---|
| 5 | 0.9612 |
| 6 | 0.8662 |
| 7 | 0.7596 |
| 8 | 0.6671 |

So block height, not block placement, is the dominant term. Measured effect of
changing only the aspect ratio — **`pred_mask_scale` is untouched, so block AREA
is identical in every row**:

| aspect_ratio | on-region | target cells | block h | accept | feasible | retina visible |
|---|---|---|---|---|---|---|
| **(0.75, 1.5) — current** | 0.4752 | 123.8 | 7.04 | 0.6167 | 0.9967 | 0.2526 |
| (0.50, 1.00) | 0.5036 | 111.6 | 5.73 | 0.9067 | 1.0000 | 0.2888 |
| **(0.30, 0.60) — wide** | **0.5395** | **103.1** | 4.47 | **0.9833** | 1.0000 | 0.2964 |
| (0.20, 0.45) | 0.5600 | 101.4 | 3.74 | 0.8967 | 1.0000 | 0.2831 |

Wide blocks are better on **every axis simultaneously**:

* **on-region purity 0.4752 → 0.5395 (+0.064).** For scale: the entire measured
  span of the existing ladder — random 0.4530 → oracle 0.5602 → MIRAGE 0.6320 in
  the sweep's purity metric, and 0.458 → 0.506 in targets-on-retina — is of the
  same order. Block shape alone moves purity about as much as the whole guide
  did.
* **target cells 123.8 → 103.1**, i.e. *closer* to the comparison arms'
  ~102–112 than the current setting is. Area comparability improves.
* **accept rate 0.6167 → 0.9833.** Nearly every slice now satisfies the
  retina-visibility rule instead of falling back to a best-effort attempt.
* **feasible rate 1.0000**, and retina visible rises 0.2526 → 0.2964, so the
  encoder keeps *more* context, not less.

Crucially it violates none of I-JEPA's ablation constraints: block area is
unchanged (so "sufficiently large targets" holds), blocks stay connected
rectangles, and there are still four of them. It is a one-line config change:

```yaml
aspect_ratio: [0.3, 0.6]     # was [0.75, 1.5]
```

This is a domain adaptation of the masking prior — I-JEPA's isotropic default is
simply wrong for a strongly anisotropic structure — and unlike A′ it is cheap,
safe, area-comparable, and does not touch the loss, the collator or the model.

**Caveats before it becomes a claim:** measured on 12 slices × 25 crops (n≈12 at
slice level) with the new soft guide, not the shipped envelope; on-region purity
is a proxy, not AUC; and it has not been run through the collator's batch-min
truncation. It should be re-measured on ≥500 slices and against the shipped
guide before any run. But as a candidate arm it dominates A′ on every measured
axis, and it costs one config line.

---

## 6. Status of everything else

* **Committed:** `3907ff0` "Document the MIRAGE frozen MeanPool sweep: purity
  did not buy AUC" on `vlm-guided-masking`, 21 files, **not pushed**. Contains
  `docs/experiments/frozen/mirage_meanpool_sweep.md` (fixes a link that pointed
  at a non-existent file), the three frozen configs, the pretraining doc, the
  epoch CSV, the sweep predictions, and `scripts/mirage_vs_oracle_region_split.py`.
  34 tests pass.
* **Your inner-vs-choroid hypothesis, measured:** *refuted* on inner retina —
  coverage is identical (0.1171 vs 0.1164, ratio 1.007) and MIRAGE's admissible
  region covers 42% *more* inner retina; *confirmed* on choroid — +15.2%, on
  67.1% of slices. The accurate statement: MIRAGE does not take inner retina
  away, it **adds choroid on top**, and **96.8% of its extra on-tissue masking is
  choroid**. Purity rises, but the added tissue is diagnostically inert.
* **Disk:** C: 22.8 → 47.1 GB free. One incident: a checkpoints folder on the
  Desktop is an **NTFS junction to D:**, so deleting the apparent duplicate
  destroyed the real 11.7 GB file on D:. It was restored from the documented
  source and verified by SHA256 + exact byte size. Biggest remaining win left
  untouched: `C:\Riot Games`, 37.7 GB.
* **Not done / deliberately not started:** V4 (still parked), any pretraining
  launch, any gradient into MIRAGE (Phase 3 — the straight-through design was
  shown to be degenerate: `corr(∂L/∂w, error) = 1.0000`, so it deselects hard
  targets).

## 7. Artifacts

```
D:\jepa_phase0\mirage-goals\outputs\
  phase1-demo\masking_demo.png            <- START HERE: 8-panel one-shot pipeline
  phase1-demo\masking_demo_{3,6,9}.png       three more examples
  phase1-demo\soft_guides.npz                raw MIRAGE probability, no repair
  mergedv3-eval\v1_v2_v3.png                 V1/V2/V3 raw output, 5 slices
  mergedv3-eval\raw_native.png               pure argmax, nothing applied
  mergedv3-eval\sampler_equiv.npz            200-slice 16x16 occupancy, both res
  mergedv2-eval\v1_vs_v2.png                 V1 vs V2, inner-band win / choroid loss
  layer-dump\{V1,V3}_data_07266_slice199\    per-stage PNGs + manifest
  matrix-dump\{V1,V3}_data_07266_slice199\   CSV/TXT matrices, final 3 stages
  mergedv3-512\...\checkpoint-34.pth         the 512 fine-tune (ep35 equivalent)
```

New scripts (untracked, not committed): `mirage_soft_guide_dump.py`,
`demo_guided_masking.py`, `ap_intersection_cost.py`,
`sampler_equivalence_dump.py`, `sampler_equivalence_replay.py`,
`fairvision_v1_v2.py`, `fairvision_raw_native.py`, `decoder_stage_dump.py`,
`dump_decoder_matrices.py`.
