# Masking Experiment Findings

> Consolidated record of all established results from the MIRAGE-guided anatomy
> masking programme. Every number is sourced from committed results or commit
> messages. Cross-references `docs/experiments/masking/ablations.md` for method
> details; this document focuses on quantitative findings.

---

## 1. Three Matched Arms (one full epoch each)

600,000 slices (FairVision Training), 9,375 iterations, identical data/seeds.

| Arm | Hidden | Context | On-region | Fallback | rep_div | val_loss | s/epoch |
|-----|--------|---------|-----------|----------|---------|----------|---------|
| random_default | 116.8 | 112.2 | 0.349 | 100% | 0.6727 | 0.0293 | 3893 |
| random_matched | 57.4 | 167.8 | 0.327 | 100% | 0.9404 | 0.0041 | 4920 |
| anatomy | 58.6 | 169.6 | 0.983 | 1.7% | 0.9521 | 0.0445 | 5796 |

**Source:** `results/masking/arms/arms.json`, commit 144b938.

### Why random_matched exists

It isolates TARGET SHAPE from MASKED AREA. By lowering `pred_mask_scale` to
0.055–0.075, it matches anatomy's hidden count (57.4 vs 58.6) and context
(167.8 vs 169.6) to within 2%. The only remaining difference is where the
targets land: 32.7% on retina (random) vs 98.3% (anatomy) — a 3.0× improvement.

### rep_diversity confound

Without the matched control, anatomy's 0.9521 vs baseline's 0.6727 looks like
anatomy causes collapse. But random_matched sits at 0.9404. The effect tracks
the **masking ratio** (21% vs 46%), not target shape. Anatomy adds only +0.012
over its proper control.

---

## 2. rep_diversity Is Not a Collapse Metric for OCT

The retina occupies only **17.6%** of tiles. 97% of tile-pairs involve at least
one background tile.

| Partition | Pairwise cosine similarity |
|-----------|---------------------------|
| All pairs | 0.3297 |
| Background–background | 0.3293 |
| Retina–retina | 0.4886 |
| Retina–background | 0.3131 |

**Source:** `results/masking/arms/arms.json` (arm diagnostics), commit 5ed437d.

The untrained encoder has the LARGEST retina/background gap (0.0851) versus
trained (0.0155–0.0287), because raw brightness separates retina from vitreous
before any learning occurs.

**Conclusion:** Drop the metric or restrict it to retina-only tiles.

---

## 3. Validation Loss Is Inverted vs Targeting Quality

| Arm | val_loss | on-region |
|-----|----------|-----------|
| random_matched | **0.0041** (best) | 0.327 (worst) |
| anatomy | **0.0445** (worst) | 0.983 (best) |

A method that predicts mostly-background patches gets a trivially low loss.
Retina is the high-variance, informative region — harder to predict, higher
loss. **Validation loss cannot rank masking strategies.** Only downstream AUC
can.

**Source:** commit 144b938.

---

## 4. The Collation Bug

### Diagnosis

`global_min_pred` truncation in `CurriculumMaskGenerator`:

| Metric | Measured |
|--------|----------|
| Target area retained (batch 64) | **7.2%** |
| K=1 in batch (batch 64) | **99.8%** |
| Mean union per slice | 55.7 cells |
| 1st-percentile target size | 1 cell |
| P(1-cell target in batch of 64) | 99.7% |

**Source:** `results/masking/collation/collation.json`, commit 765efbf.

### Front-slicing spatial bias

The truncation kept the FIRST K indices in raster order, biasing survivors to
the top-left edge:

| Metric | All targets | Old (truncated) | New (resampled) |
|--------|-------------|-----------------|-----------------|
| Mean row | 6.18 | 4.50 | 6.13 |
| Leftmost-3-columns share | 18.6% | 36.0% | 18.6% |

**Source:** `results/masking/collation/why_starved.json`.

### Fix: fixed-K resampling

| Policy | Distinct cells | % of ideal |
|--------|---------------|------------|
| Ideal (no truncation) | 55.7 | 100% |
| global_min (batch 64) | 4.0 | 7.2% |
| Bucketing | 36.8 | 66.0% |
| **Fixed-K K=16** | **51.3** | **92.0%** |
| Fixed-K K=20 | 53.9 | 96.8% |

K=16 chosen: 92% of ideal at 19.9% repeated slots (the knee).

### shrink_to_k connectivity

Uniform subsampling broke target connectivity: 231/256 single-component.
Breadth-first shrink_to_k: **256/256 connected**.

**Source:** commit 2b167f7.

---

## 5. Timing Decomposition

| Transition | Δ seconds | Cause |
|------------|-----------|-------|
| random_default → random_matched | +1,027 | Context grows 112→168 tokens |
| random_matched → anatomy | +876 | The anatomy sampler |

The sampler costs ~15% of total epoch time, not 49%. The majority is the price
of leaving more context visible to the encoder.

Additional cost signal: `is_viable` costs **18.89 ms/img** vs `build_targets`
14.24 ms and duplicates its work.

**Source:** commit 144b938.

---

## 6. Guide Cache

| Property | Value |
|----------|-------|
| Volumes | 6,000 |
| Slices | 600,000 |
| Build time | 3,941 s (66 min) |
| Size (compressed) | 3.85 GiB |
| Schema | v2: `[100, 2, 200, 200]` uint8, channels P_inner + P_choroid |
| Post-softmax at native 200×200 | Yes — pool-then-softmax measured at Jaccard 0.587, 0/200 identical, −40% cells |
| Max diff vs live recomputation | 1 uint8 level (AMP rounding) |
| Adapter SHA | 3186b1fa278bc97f |
| MIRAGE SHA | 82e5a0dd09b6bd58 |

**Source:** `results/masking/precompute/precompute_verification.json`, commit 7d0d7f9.

### Why NOT cache H0

H0 is (384, 64, 64) fp16 = 3.00 MiB/sample. Full dataset: **1.72 TiB**. And
features cannot be validly cropped after the fact (the crop must happen before
the encoder, not after).

---

## 7. Downstream VRAM (fp32)

**Bug found:** `precompute_features()` called `autocast()` unconditionally. All
previous downstream evaluations were fp16, not fp32.

| Mode | Config | Peak MB | Speed |
|------|--------|---------|-------|
| Frozen probe | chunk=100, 1 vol | 1,653 | 0.32 s/vol |
| Full fine-tune | batch=1, 100 slices, accum=4 | 18,975 | 0.87 s/vol |
| Partial FT (freeze 6/12) | batch=1, 100 slices | 9,961 | 0.59 s/vol |

Full fine-tune at >1 volume on 24 GB: Windows silently spills to host RAM
(no OOM raised, crawls instead).

Gradient checkpointing: **not available** in `src/models/vision_transformer.py`.

**Source:** `results/masking/vram/fp32_downstream.json`, commit 376665d.

---

## 8. Slice-Depth Validation (cfg-7 Adapter)

| Arm | L_rel reduction | Drift |
|-----|-----------------|-------|
| Middle-slice only | 29.90% | 0.1811 |
| Stratified (full depth) | 26.18% | 0.1745 |
| Δ | −3.72 pp | |

Per-depth-band spread: only **3.52 pp** (min 24.54%, max 28.06%).

**Source:** `results/masking/slice_pos/slice_pos.json`, `depth_profile.json`, commit c6d33c0.

---

## 9. Coverage Audit

### Issue 1: Block masking centre-vs-edge bias

I-JEPA's `_sample_uniform_location` gives a 10.68× centre-vs-edge coverage
ratio. Anatomy masking is the most spatially uniform arm at 2.55×.

### Issue 2: Guided-rectangle overlap

Overlap 40.5% → 36.1% after fix A (pick least-overlapping window). Residual
is geometrically forced: 4 targets × 41 cells × 0.40 fill = 65.7 anatomy cells
needed, only 45.6 available.

### Issue 3: Deep-layer under-coverage

Anatomy under-covers choroid (0.48×) and sclera (0.39×) vs random, because
MIRAGE labels inner retina and choroid but not sclera. Acceptable for glaucoma
(RNFL + choroid/lamina are the structures of interest).

### Issue 4: Ramp gate (T_warm)

Every probe omitted `set_epoch`, so `r_t=0` → pure random masking while
claiming to be guided. Production used T_warm=25 of 100 epochs (first 25
epochs trained random masking, by design).

**Source:** `results/masking/coverage/`, commit fc49f61.

---

## 10. Region-Growth Bug Fixes

1. **Bug A:** `grow_components` only grew ONE connected component per class.
   The ONH splits the retina. 5.8% of mass stranded. Fixed: grow all components
   under shared budget.
2. **Bug B:** `build_targets` orphaned surplus regions when >4 components existed
   (8.4% of slices, up to 93.1% capacity lost). Fixed: per-class component cap.

After fix: dead targets 14.12% (random) vs **2.12%** (anatomy), 6.7× fewer.

**Source:** commit 8ef247d.

---

## 11. Integration Smoke Test

| Arm | Hidden | On-anatomy | Context | To-loss | Fallback | s/step | VRAM MB |
|-----|--------|------------|---------|---------|----------|--------|---------|
| rect_default | 113.4 | 22.0% | 77.7 | 113.4 | 0% | 0.20 | 4667 |
| rect_matched | 51.3 | 20.6% | 156.8 | 51.3 | 0% | 0.26 | 6107 |
| anatomy | 55.6 | 80.9% | 200.4 | 51.3 | 1.9% | 0.51 | 6413 |

Found `is_viable()` is required: without it, empty targets raise within 40 steps.

**Source:** `results/masking/integration/integration.json`, commit 595938a.

---

## Open Questions / Not Yet Established

1. **No downstream AUC has been measured** for any of the three arms. The
   existing AUC numbers (0.8746 random, 0.8807 MIRAGE-envelope, 0.8855 oracle)
   come from the PRIOR rectangle-based masking programme, not the anatomy sampler.

2. **No checkpoints were saved** from the three matched arms — `save_every`
   defaulted to 25, and only 1 epoch was run. The arms cannot be evaluated
   downstream without retraining.

3. **One epoch is too early** to judge representation quality. The arms
   demonstrate targeting and mechanics, not that anatomy masking produces a
   better encoder.

4. **FairVision has no anatomy ground truth**, so "the guide changed the mask"
   is established but "the guide improved the mask" is not. Only downstream AUC
   against glaucoma labels can establish improvement.

5. **rep_diversity** is confirmed unreliable for OCT (§2) but no replacement
   collapse metric has been validated.

6. **The schema-2 soft guide is now wired in** (commit `a904bfd`), so anatomy
   mode can run on two-class continuous scores. Measured end to end on real
   FairVision volumes: two-class reaches on-region 0.925 with a 50.1-cell union
   versus the merged single-class 0.907 with 55.4 cells, both 256/256 connected.
   What is NOT yet done is a training run that USES it — every arm reported in
   this document ran on the merged single-class envelope.

### On-region values across this document (disambiguation)

Three different on-region scores appear; they are NOT from the same measurement:

| on-region | Guide | Collator version | Source |
|-----------|-------|------------------|--------|
| **0.983** | merged single-class (adapted) | pre-connectivity-fix, three-arm comparison | commit `144b938`, `results/masking/arms/arms.json` |
| **0.907** | merged single-class (schema 1) | post-connectivity-fix, r_t=1, batch 64 | commit `a904bfd` end-to-end measurement |
| **0.925** | two-class soft (schema 2) | post-connectivity-fix, r_t=1, batch 64 | commit `a904bfd` end-to-end measurement |

The drop from 0.983 → 0.907 for the merged guide is due to the collator
connectivity fix in `a904bfd` (fallback targets previously boosted on-region
because `shrink_to_k` was only applied in the anatomy branch; after the fix,
oversized fallback rectangles are also shrunk, changing the cell allocation).
The 0.925 vs 0.907 comparison (same collator, same run) shows two-class is
more efficient: slightly higher on-region with fewer cells (50.1 vs 55.4).
