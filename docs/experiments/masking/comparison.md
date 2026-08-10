# Masking comparisons

This page collects arm-vs-arm evidence: downstream AUC where available, mask-budget decomposition, and diagnostics that explain which comparisons are valid.

## Ep30 envelope-vs-anatomy head-to-head

The clean downstream comparison is [`anatomy_vs_rectangle_ep30.md`](anatomy_vs_rectangle_ep30.md). Both arms share identical weights through epoch 27 and differ only in the mask used during epochs 28--30. The baseline is `mirage_envelope` (MIRAGE-placed rectangles); the contribution arm is `mirage_anatomy` (MIRAGE-shaped connected anatomy blobs).

| arm | test AUC, 5 probe seeds |
|---|---:|
| envelope ep30 | 0.8528 ± 0.0018 |
| anatomy ep30 | **0.8582 ± 0.0003** |

Delta is **+0.0054**. Welch t p=**0.00219**, Mann-Whitney p=**0.0079**, Cohen's d=**4.20**. The arms are fully separated: worst anatomy seed **0.8578** > best envelope seed **0.8542**. Paired bootstrap over test volumes gives **+0.0044**, 95% CI **[+0.0010, +0.0077]**, p=**0.012**.

## Mask-budget decomposition for the ep30 comparison

Raw cell counts mislead because background is trivially predictable. The relevant axis is retinal/anatomy budget. Measured on 1,000 slices through the production collator:

| arm | context | total hidden | anatomy hidden | background hidden | anat context | % retina hidden | on-anatomy | dead tgt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_default | 63.2 | 114.1 | 24.7 | 89.4 | 12.5 | 52.5% | 21.8% | 28.68% |
| random_matched | 157.3 | 48.9 | 10.1 | 38.8 | 30.2 | 21.6% | 20.6% | 46.35% |
| envelope_default | 71.4 | 117.5 | 36.2 | 81.3 (69%) | 6.0 | 77.3% | 30.7% | 3.57% |
| envelope_matched | 157.3 | 51.6 | 21.8 | 29.7 | 18.4 | 48.2% | 42.3% | 2.02% |
| anatomy | 158.2 | 54.3 | 39.3 | 15.0 (28%) | 5.7 | 84.5% | 72.1% | 2.05% |

On the anatomy axis, the ep30 arms are closely matched and the residual difference is against anatomy: anatomy hides **8.6% more retina** and leaves **5% less retinal context**. Tissue-context per tissue-cell predicted is **0.145** for anatomy vs **0.166** for envelope, so the anatomy task is marginally harder. The real finding is budget efficiency: rectangles must spend **5.4×** more hidden budget on background to hide comparable retina (**81.3** vs **15.0** cells). The earlier **4.3× more context per predicted token** claim counted background tokens and is retracted. `envelope_matched` is not a valid control because matching total cells leaves it hiding only **48.2%** of retina.

## B. Controlled Three-Arm Experiment

One full FairVision Training epoch each: **600,000 slices, 9,375 iterations**,
identical data and seeds. Only the mask strategy differs.

![Three-arm matched comparison](../../../results/masking/arms/arms.png)

*The headline result. Left: random_default (shipped I-JEPA). Centre:
random_matched (area-matched random). Right: anatomy (MIRAGE-anatomy shapes).
random_matched isolates target shape from masked area.*

**Source:** `results/masking/arms/arms.json`, commit 144b938.

| Metric | random_default | random_matched | anatomy |
|--------|---:|---:|---:|
| Hidden cells | 116.8 | 57.4 | 58.6 |
| Context tokens | 112.2 | 167.8 | 169.6 |
| On-region fraction | 0.349 | 0.327 | **0.983** |
| Fallback rate | 100% | 100% | 1.7% |
| rep_diversity | 0.6727 | 0.9404 | 0.9521 |
| val_loss | 0.0293 | 0.0041 | 0.0445 |
| Seconds/epoch | 3,893 | 4,920 | 5,796 |

### Why random_matched exists

Comparing anatomy to the shipped baseline changes **both** WHERE targets land
and HOW MUCH is masked (58.6 vs 116.8 cells). Shape cannot be isolated without
an area-matched control. `random_matched` lowers `pred_mask_scale` to
0.055–0.075, matching anatomy's hidden count (57.4 vs 58.6) and context
(167.8 vs 169.6) to within 2%. The only remaining variable is target
placement.

**Result:** at matched budget, anatomy places **3.0×** more targets on retina
(98.3% vs 32.7%) with a 1.7% fallback rate.

### The anatomy run's starting point

The anatomy arm warm-starts from `resume-ep27.pth.tar` (epoch 27, loss 0.11894,
random-masked bootstrap). It therefore measures what anatomy guidance adds
**on top of** a random-masked foundation, not anatomy from random init.

---

## C. Negative and Cautionary Results

### C1. rep_diversity is not a collapse metric for OCT

Retina occupies only **17.6%** of grid cells. 97% of tile-pairs involve at least
one background tile, so the all-pairs mean is dominated by background.

| Partition | Cosine similarity |
|-----------|---:|
| All pairs | 0.3297 |
| Background–background | 0.3293 |
| Retina–retina | 0.4886 |
| Retina–background | 0.3131 |

**Source:** `results/masking/arms/arms.json`, commit 5ed437d.

The **untrained** encoder has the LARGEST retina/background gap (0.0851) versus
trained encoders (0.0155–0.0287), because raw pixel brightness already separates
vitreous from tissue before any learning.

The matched control overturns the naive reading: anatomy (0.9521) vs baseline
(0.6727) looks like anatomy causes collapse, but random_matched is 0.9404.
The effect tracks the **masking ratio** (masking 21% vs 46% of the image), not
target shape. Anatomy adds only +0.012 over its proper control.

### C2. Validation loss is inverted versus targeting quality

| Arm | val_loss | on-region |
|-----|---:|---:|
| random_matched | **0.0041** (best) | 0.327 (worst) |
| anatomy | **0.0445** (worst) | 0.983 (best) |

A method predicting mostly-background patches gets trivially low loss. Retina
is high-variance and harder to predict. **Validation loss cannot rank masking
strategies.** Only downstream AUC can.

### C3. One epoch from random init is too early

The three arms ran one epoch each. This demonstrates targeting mechanics and
engineering correctness, not that anatomy masking produces a better encoder.
No representation-quality conclusion is drawn.

---

### Legacy MIRAGE-envelope arm (rectangle placement)

![Legacy MIRAGE guide construction](../../../results/masking/mirage_guide_pipeline.png)

![Legacy arms comparison](../../../results/masking/mirage_masking_arms.png)

Policy sweep over **1,000 volumes / 19,987 slices**:

| Method | Target purity | Unique cells | Context |
|---|---:|---:|---:|
| RANDOM | 0.4530 | 112.4 | 107.6 |
| ORACLE | 0.5602 | 101.9 | 116.6 |
| MIRAGE envelope | **0.6320** | 101.7 | 117.2 |

Downstream AUC (FairVision Test, **3,000 volumes**, frozen MeanPool ep100):

| Arm | AUC |
|---|---:|
| random | 0.8746 |
| MIRAGE envelope | 0.8807 |
| **oracle** | **0.8855** |

Mask purity was not a validated proxy for downstream AUC.

![Threshold wiring bug](../../../results/masking/mirage_threshold_bug.png)

![Oracle failure cases](../../../results/masking/oracle_failure_cases.png)
