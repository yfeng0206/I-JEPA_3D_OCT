# Engineering notes, limitations, and paper inventory

This page keeps engineering results and bookkeeping separate from the comparison and ablation pages.

## E. Efficiency

### Timing decomposition (commit 144b938)

| Transition | Δ seconds | Cause |
|---|---:|---|
| random_default → random_matched | +1,027 | Context grows 112→168 tokens (encoder work) |
| random_matched → anatomy | +876 | The anatomy sampler |

The sampler costs ~15% of total epoch time, not the naive 49%
((5796−3893)/3893). The majority is the price of larger context.

`is_viable` costs **18.89 ms/img** and duplicates `build_targets`'
**14.24 ms/img** — a redundant check that should be merged.

### Guide cache (commit 7d0d7f9)

| Property | Value |
|---|---|
| Volumes / slices | 6,000 / 600,000 |
| Build time | 3,941 s (66 min) |
| Size (compressed) | 3.85 GiB |
| Schema | v2: `[100, 2, 200, 200]` uint8, P_inner + P_choroid |
| Max diff vs live | 1 uint8 level (AMP rounding) |
| Adapter SHA | 3186b1fa278bc97f |
| MIRAGE SHA | 82e5a0dd09b6bd58 |

**Source:** `results/masking/precompute/precompute_verification.json`.

**Post-softmax at native 200×200** is mandatory. Pool-then-softmax was measured
at Jaccard 0.587, 0/200 identical masks, −40% cells. The guide is cropped with
the image before pooling to 16×16.

**Why NOT cache H0:** (384, 64, 64) fp16 = 3.00 MiB/sample → **1.72 TiB** for
the full dataset, and features cannot be validly cropped after encoding (crop
must precede the encoder).

### fp32 downstream VRAM (commit 376665d)

![fp32 downstream memory and throughput](../../../results/masking/vram/fp32_downstream.png)

*Peak VRAM by evaluation mode. Full fine-tune exceeds 24 GB for >1 volume.*

**Bug found:** `precompute_features()` called `autocast()` unconditionally. All
prior downstream evaluations were fp16, not fp32.

| Mode | Peak MB | Speed |
|---|---:|---:|
| Frozen probe (chunk=100, 1 vol) | 1,653 | 0.32 s/vol |
| Full fine-tune (batch=1, 100 slices, accum=4) | 18,975 | 0.87 s/vol |
| Partial FT (freeze 6/12 blocks) | 9,961 | 0.59 s/vol |

Full fine-tune >1 volume on 24 GB: Windows silently spills to host RAM (no OOM,
crawls). Gradient checkpointing not available in `src/models/vision_transformer.py`.

**Source:** `results/masking/vram/fp32_downstream.json`.

---

## F. Generality Across Slice Depth

Cfg-7 validated on slices spanning the full volume depth, not just the
middle-of-ONH B-scan (commit c6d33c0):

| Arm | L_rel reduction | Drift |
|---|---:|---:|
| Middle-slice only | 29.90% | 0.1811 |
| Stratified (full depth) | 26.18% | 0.1745 |
| Δ | −3.72 pp | |

Per-depth-band spread: only **3.52 pp** (min 24.54% at band 160–200, max
28.06% at band 80–120). No peripheral cliff.

**Source:** `results/masking/slice_pos/slice_pos.json`, `depth_profile.json`.

**Caveat:** the 27.7/30.7/32.5% saturation figures reported elsewhere come from
the middle-slice cache and are not directly comparable to the 26.18% stratified
figure.

---

## G. Limitations and Open Questions

| Limitation | Status |
|---|---|
| **No downstream AUC measured** for any anatomy arm | Existing AUCs (0.8746/0.8807/0.8855) come from the PRIOR rectangle-placement programme |
| **No checkpoints saved** from the three matched arms | `save_every` defaulted to 25; only 1 epoch was run (fixed to 5 in production config) |
| **One epoch is too early** to judge representation quality | Arms demonstrate mechanics, not encoder improvement |
| **FairVision has no anatomy ground truth** | "Changed" ≠ "improved"; only downstream AUC can establish improvement |
| **Anatomy arm warm-starts from ep27** | Measures anatomy ON TOP of a random-masked bootstrap, not anatomy from scratch |
| **rep_diversity is unreliable** for OCT (§C1) | No validated replacement collapse metric exists |
| **The two-class soft guide is wired** (commit a904bfd) but not trained | All reported arms used the merged single-class envelope |

---

## Corrections to Earlier Claims

| Earlier statement | Correction | Source |
|---|---|---|
| default `mass_cap=0.80` | **Superseded.** 500-slice sweep selects 0.90. | 8ef247d |
| old cap-sweep tables | **Superseded.** Predated multi-component growth. | 8ef247d |
| "residual head gets zero gradient → guide is static" | **Wrong wiring.** Frozen head must read adapted H; 192-image held-out: agreement 0.9705, Jaccard 0.7365. | 778791b |
| "76.8% L_rel reduction" | **Memorisation.** 24 slices × 400 steps. Honest: 29.9% over 6,000 images. | c6d33c0 |
| "I-JEPA scattered 54.2→19.2" | **Wrong table reading.** Correct: random patches 17.6 vs multi-block 54.2. | 778791b |
| "L_sem is harmful" | **Wrong.** 40-slice probe: gradient cosine −0.076, agreement 0.9999. Ineffective, not harmful. | 778791b |
| "ViT-L interpolates positional embeddings" | **Wrong.** Neither checkpoint did so. | 778791b |
| "57.6–89.3% of gradient reaches MIRAGE" | **Wrong label.** Those were L_rel reduction, not gradient share. | 778791b |
| "anatomy context diluted: 48.7% vs 58.3%" | **Indexing bug.** Fixed: 58.7% vs 57.9%. | 778791b |
| "84.2 context tokens" | **Incomplete.** Post-truncation batch-8; per-image is ~100–105. | 778791b |
| `p.grad is None` proves MIRAGE detached | **Wrong test.** Check `H.grad_fn is None`. | c6d33c0 |
| dropout makes masks stochastic | **Measured false.** Jaccard 1.0000 in train/eval (with drop_path=0.1 but EVAL mode). | c6d33c0 |
| void class invalidates anatomy score | **Overstated.** 25,600 cells: void prob 1.1e-5, corr 0.99999997. | 778791b |
| anatomy is choroid-biased | **Wrong for current sampler.** Random is worst (0.899); anatomy is 0.966. | 778791b |
| changed MIRAGE outputs = better segmentation | **Not supported.** No FairVision anatomy labels. Report change only. | 778791b |
| `jepa_to_mirage` shows JEPA improved MIRAGE | **Circular.** Teachers came from MIRAGE-derived-mask run. | 778791b |
| B-1: "15.3% retained, K=1 in 69.2%" | **Revised.** Re-measured after growth fixes: 7.2% retained, K=1 in 99.8%. | c6d33c0 |
| MIRAGE was frozen | **Not actually.** 99.8% of params had requires_grad=True; safe only via no_grad wrapping. | c6d33c0 |

---

## Blockers and Recommendations

**Recommended implementation order:**

1. **Bucketing / fixed-K collation** — the global-min truncation destroys
   anatomy targets (7.2% retained at batch 64). Bucketing recovers 66.0%;
   fixed-K K=16 resampling recovers 92.0%. This is the top priority because
   without it the entire anatomy-masking benefit is nullified in production.

2. **Post-softmax two-channel guide caching** — the one-time precompute
   (66 min, 3.85 GiB compressed) replaces live MIRAGE inference (~45 h across
   the schedule) and removes 95.6M frozen params from JEPA training VRAM.

**Rejected forward plan:** precomputing from frozen H0 features is NOT
recommended. H0 at (384,64,64) fp16 = 1.72 TiB for 600K slices, features
cannot be validly cropped post-encoding, and the pool-before-softmax
approximation measured Jaccard 0.587 / 0 of 200 identical masks / −40% cells.
This belongs exclusively in §Rejected Designs.

---

## Main Paper vs Appendix Split

### Main paper (8-page body)

| ID | Ablation | Key result |
|---|---|---|
| A1 | Random vs Oracle vs Anatomy targets | 3.7× efficiency per cell |
| A2 | Target efficiency + context preservation | 53.4 vs 122.6 cells; 175.0 vs 107.4 context |
| A3 | Mass-cap sweep | 0.90 operating point, 0.99 context collapse |
| A4 | Adapter architecture (cfg-7 knee) | Depth 0→2: +9.3 pp; 2→4: +0.4 pp |
| A5 | Held-out + slice-depth generalisation | 29.79% vs 29.59% (no gap); 3.52 pp spread |
| A7 | Budget lock | 26% cells relocate under fixed budget |
| A8 | Uncertainty localisation | 10.2× change ratio at uncertain vs sure cells |
| — | Final downstream results | Pending — no AUC yet |

### Appendix

| Ablation | Section |
|---|---|
| Pixel gate (D1) | §Rejected |
| STE / soft top-K (D2) | §Rejected |
| JEPA-error confound (D3) | §Historical |
| Projector absorption (D4) | §Rejected |
| Multi-component growth bugs | §D2–D3 |
| Direct intersection failure | §Rejected |
| Rectangle aspect ratio | §Historical |
| Adapter saturation curve | §Adapter Saturation |
| AMP vs FP32 guide generation | §AMP vs FP32 |
| Feature-cache rejection | §E Guide cache |
| Variable-K batching / bucketing | §D1 |
| Determinism / freezing | §D6 |
| Separate residual head | §Rejected |

---

## Comparison-Image Inventory

### Main-paper figures

| Figure | Purpose | Exists on disk? | Path |
|---|---|---|---|
| Random vs Oracle vs Anatomy (same crop) | A1 method comparison | ✓ | `results/masking/explain/three_methods.png` |
| Adapter behaviour: OCT \| frozen \| adapted \| delta \| uncertainty \| masks | A4 + A8 | ✓ | `results/masking/adapter_guardrails/seg_before_after.png` |
| Sampler topology: single- vs multi-component | S1 | ✓ | `results/masking/split_fix/split_fix.png` |
| Edge case: oracle miss vs anatomy target | Motivation | ✓ | `results/masking/oracle_failure_cases.png` |
| 20-slice budget-locked before/after grid | A7 | ✓ | `results/masking/adapter_guardrails/before_after_20.png` |
| Adapter sweep scatter | A4 | ✓ | `results/masking/adapter_sweep/adapter_sweep.png` |
| Mass-cap trade-off plot | A3 | ✗ | NEEDS GENERATING |

### Appendix figures

| Figure | Purpose | Exists on disk? | Path |
|---|---|---|---|
| Depth profile (reduction vs B-scan depth) | A5 slice robustness | ✗ | NEEDS GENERATING from `results/masking/slice_pos/depth_profile.json` |
| Adapter saturation curve | Schedule justification | ✗ | NEEDS GENERATING from `adapter_sweep.py` |
| AMP vs FP32 rare disagreement | Implementation safety | ✗ | NEEDS GENERATING |
| Wrong-cache vs correct-cache (pool-before-softmax) | Cache rejection | ✗ | NEEDS GENERATING |
| Global-min vs bucketing comparison | Collation fix | ✓ | `results/masking/collation/collation_fix.png` |
| Full pipeline trace | Method overview | ✓ | `results/masking/pipeline/full_pipeline.png` |
| Class balance (random/oracle/anatomy) | A1 | ✓ | `results/masking/pipeline/class_balance.png` |
| Coverage heatmaps | Spatial uniformity | ✓ | `results/masking/coverage/coverage.png` |
| Element-wise gate probe | D1 rejection | ✓ | `results/masking/gate_real/elementwise_gate_probe.png` |
| Error vs anatomy | D3 rejection | ✓ | `results/masking/error_vs_anatomy/error_vs_anatomy.png` |
| Edge cases (demo) | Sampler robustness | ✓ | `results/masking/demo/edge_cases.png` |

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
