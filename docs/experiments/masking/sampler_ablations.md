# Sampler and mask-budget ablations

This page contains ablations of the target sampler, mask budget, coverage, collation, and rejected sampler designs. It is about mechanics and controls, not downstream AUC.

## Mask Budget Sweep

### Full table (500 slices, commit 8ef247d)

| cap | target cells | anatomy mass | dead diag | inner masked | context | retina-in-ctx | zero-retina ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| RANDOM | 122.0 | 0.556 | 2.40% | 0.516 | 107.7 | 25.1 | 1.00% |
| 0.80 | 46.1 | 0.758 | 0.46% | 0.725 | 181.3 | 25.6 | 0.80% |
| 0.85 | 50.5 | 0.808 | 0.46% | 0.774 | 177.4 | 21.7 | 0.80% |
| **0.90** | **55.9** | **0.858** | **0.46%** | **0.825** | **172.7** | **17.0** | **0.80%** |
| 0.95 | 63.3 | 0.909 | 0.46% | 0.875 | 166.1 | 10.4 | 1.00% |
| 0.99 | 69.2 | 0.907 | 0.46% | 0.885 | 160.7 | 5.0 | 21.00% |

### Class balance (1,000 slices, commit 778791b)

![Class balance: random vs oracle vs anatomy](../../../results/masking/pipeline/class_balance.png)

*Anatomy is near-balanced (0.966 inner/choroid); random is the most
choroid-biased (0.899).*

| Method | Target cells | Inner eff. | Choroid eff. |
|---|---:|---:|---:|
| RANDOM | 122.6 | 0.42 | 0.47 |
| ORACLE | 70.0 | 0.89 | 0.86 |
| ANATOMY | 46.6 | **1.56** | **1.61** |

Efficiency = fraction of class mass hidden per cell spent. Anatomy is 3.7×
more efficient than random for inner retina.

### Dead targets (1,000 slices × 4 = 4,000 targets)

| | RANDOM | ANATOMY+fallback |
|---|---:|---:|
| Dead targets | 14.12% (565) | **2.12%** (85) |

6.7× fewer dead targets.

---

## Coverage Audit

![Per-epoch spatial coverage, all arms](../../../results/masking/coverage/coverage.png)

*Coverage heatmaps for each arm over 1,000 slices. Random block masking has a
10.68× centre-vs-edge bias (inherited from I-JEPA's `_sample_uniform_location`).
Anatomy masking is the most spatially uniform arm (2.55×).*

**Source:** `results/masking/coverage/`, commit fc49f61.

| Issue | Finding | Status |
|---|---|---|
| Centre-vs-edge bias | 10.68× for blocks, 2.55× for anatomy | Design (inherited from I-JEPA) |
| Guided-rect overlap | 40.5% → 36.1% → 8.0% | Fixed |
| Deep-layer under-coverage | Choroid 0.48×, sclera 0.39× vs random | Acceptable (glaucoma targets RNFL + choroid) |
| Ramp gate (T_warm) | Probes measured random while claiming guided | Fixed (assert r_t > 0) |

Spatial correlation with anatomy frequency: random 0.664, envelope 0.840,
envelope-matched 0.903, **anatomy 0.996** (commit fc49f61).

---

## D. Bugs Found and Fixed

### D1. global_min_pred collation (commit 765efbf, 0f1ab43)

The collator's `global_min_pred` truncation keeps the minimum target size
across the entire microbatch, destroying ragged anatomy targets.

| Metric (batch 64, 981 slices) | Value |
|---|---:|
| Target area retained | **7.2%** |
| K=1 in batch | **99.8%** |
| P(1-cell target in batch) | 99.7% |

![Collation fix: target area before and after](../../../results/masking/collation/collation_fix.png)

*Left: truncated targets retain only 4 of 55.7 cells. Right: fixed-K K=16
resampling retains 51.3 cells (92.0% of ideal).*

**Front-slicing spatial bias:** truncation kept the FIRST K indices in raster
order, biasing survivors to the top-left:

![Why the predictor was starved: front-slicing bias](../../../results/masking/collation/why_starved.png)

*The survivor distribution concentrates at the ILM/vitreous interface — a
high-contrast edge trivially predictable from context.*

| Metric | All targets | Old (truncated) | New (resampled) |
|--------|---:|---:|---:|
| Mean row | 6.18 | 4.50 | 6.13 |
| Leftmost-3-column share | 18.6% | 36.0% | 18.6% |

**Source:** `results/masking/collation/why_starved.json`, commit 0f1ab43.

**Fix:** fixed-K resampling (K=16 chosen at 92.0% retention, 19.9% repeated
slots — the knee). Connectivity preserved by `shrink_to_k` (breadth-first
connected subset): 231/256 → **256/256** single-component. The later
fallback-path variant achieved 233/256 → 256/256.

| Policy | Distinct cells | % of ideal |
|--------|---:|---:|
| Ideal | 55.7 | 100% |
| global_min (batch 64) | 4.0 | 7.2% |
| Bucketing | 36.8 | 66.0% |
| **Fixed-K K=16** | **51.3** | **92.0%** |

The earlier findings record also measured **Fixed-K K=20** at **53.9**
distinct cells (**96.8%** of ideal). K=16 was retained because it is the knee:
92.0% of ideal with 19.9% repeated slots. Source:
`results/masking/collation/collation.json`, commit 765efbf.

### D2. Sampler bug A — single-component growth (commit 8ef247d)

`grow_region()` seeded at the global maximum and could only fill that seed's
connected component. Over **1,982 class-regions**: 5.8% of mass stranded in
unreachable components; cap bound on only 18.9% of regions (81.1% stopped by
component exhaustion).

![Split-support slices before and after fix](../../../results/masking/split_fix/split_fix.png)

*Four ONH slices where the retinal band splits. Before: one fragment captured.
After: all fragments grown under shared budget.*

### D3. Sampler bug B — orphaned components (commit 8ef247d)

8.4% of slices had >4 connected components; surplus regions were orphaned after
growth. Mean capacity discarded 28.6% (max 93.1%). Fix: per-class component cap
at n=4, budget redistributed. After: 0.27% mean / 15.4% max loss.

### D4. Guided-rectangle overlap (commit fc49f61)

Overlap 40.5% against a declared tolerance of 0.25; random was only 28.9%.
Fix A (least-overlap window): 40.5% → 36.1%. Residual is geometrically forced:
4 targets × 41 cells × 0.40 fill needs 65.7 anatomy cells but only 45.6 exist.
Fix B (matched scale): → **8.0%**.

![Coverage fixes: overlap before and after](../../../results/masking/coverage/fixes_before_after.png)

*Overlap progression: 40.5% (original) → 36.1% (least-overlap) → 8.0%
(matched scale). The anatomy arm has 0% overlap by construction.*

### D5. The ramp trap (commit fc49f61)

`T_warm=25` means `r_t=0` for the first 25 epochs, so every probe that omitted
`set_epoch` silently measured **random masking** while reporting anatomy.
The production config's first 25/100 epochs trained random masking by design.

### D6. MIRAGE was not actually frozen (commit c6d33c0)

`build_mirage()` returned `.eval()` but left 95,374,852 / 95,571,460 params
(99.8%) with `requires_grad=True`. Safe only via `no_grad()` wrapping.
Additionally, `drop_path_rate=0.1` made it stochastic: train-vs-eval logit
swing of **1.545e+01**; two identical forwards differed by **1.428e+01**.
Fix: explicit freeze (0 trainable) + drop_path disabled at source (bitwise
identical in train and eval: 0.000e+00 diff).

---

## Rejected Designs

| Design | Measurement | Decision |
|---|---|---|
| **D1 pixel-level gate `s*x`** | 1,360 patches: `dL/ds=0` for all; 200 Adam steps: `max(\|s−0.5\|)=0` | Rejected: no gradient where selection is needed |
| **D2 token gate + ST top-k** | 20 slices, 4 temps, 600 steps: always favors easy targets | Rejected |
| **D3 detached hard mask** | `grad_MIRAGE(L_JEPA)=0` by construction | Accepted |
| **Separate residual logit head** | 0.000e+00 gradient; seg agreement and Jaccard both 1.000000 | Dead end |
| **Extent-based budget** | Coverage changed by 0.001–0.011; unconstrained consumed 255/256 cells | Rejected |
| **Rank-cut geodesic** | 29.7% of 300 slices disconnected | Rejected |
| **Largest-to-smallest rebalance** | Size spread 3.54, chain formation (300 slices) | Rejected |
| **Constrained projector for L_sem** | 233% feature churn, 0.0% gain (40 slices, 400 steps) | Rejected |
| **Direct A' intersection** | 58.9 vs 123.8 cells; ~19% empty-batch risk | Rejected |
| **JEPA-error scorer** | Distance correlation +0.57 dominates anatomy −0.27 (20 slices) | Rejected |

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
