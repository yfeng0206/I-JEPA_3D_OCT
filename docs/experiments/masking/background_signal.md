# Background signal in patch I-JEPA

**Status:** investigation complete; **no pretraining or downstream training run
was launched for a new masking arm**. The model experiments below are
forward-only measurements of frozen checkpoints, except for lightweight
downstream linear heads.

**Date:** 2026-08-14
**Companion method note:** [COVER-then-RANDOM](cover_random.md)
**Primary artifacts:** `D:\jepa_phase0\reports\background_signal\`,
`target_composition\`, `downstream_region_auc\`, `patch_attribution\`,
`anatomy_mask_calib\`, `cover_random\`, and `edge_cases_random_legal\`

## Executive answer

The motivating idea was:

> The predictor needs anatomical context to reconstruct masked anatomy, but
> black/background cells also have positional embeddings. It may therefore
> need to learn that “black here means there is nothing informative around
> me,” so background itself is part of the training signal.

That statement contains three different claims:

| claim | verdict | strongest evidence |
|---|---|---|
| **A. Background as context is informative.** | **Partly supported.** Background tokens are attended and removing them raises error, but a healthy model values one anatomy token about 4–6× more. The ablation does not isolate a positive background-specific effect beyond generic token removal. | Architecture plus paired context-removal probe |
| **B. Background as a target is a real learning signal.** | **Confirmed relative to a strong position-only baseline.** Healthy predictors remove 58–68% of the background error left by a per-position, no-context reference. | Frozen predictor skill score |
| **C. An arm that rarely predicts background cannot represent it.** | **Refuted in the narrow sense tested.** All 15 target encoders linearly separate anatomy from background at AUC 0.979–0.988. | Frozen-token linear probe |

The most actionable unplanned finding is that the anatomy-blob arm's
**predictor deteriorated sharply after ep30**. Its full-context error rose
2.76× by ep56, anatomy-context value became statistically indistinguishable
from a count-matched random removal, and background context became more useful
than anatomy context. This is a plausible mechanism for its downstream
plateau, but the frozen-checkpoint analysis is diagnostic rather than causal.

Downstream, anatomy-only pooling improved test AUC for all four ep50 encoders,
while background-position tokens still supported high AUC. Because ViT tokens
mix information globally and the anatomy mask misses about 11% of true anatomy,
this shows that disease signal is **readable from background positions**; it
does not show that black pixels themselves contain glaucoma signal.

## Evidence language used here

- **PROVEN:** follows from code or arithmetic and does not depend on learned
  weights.
- **MEASURED:** observed on the frozen checkpoints or stored downstream
  artifacts under the stated protocol.
- **SUGGESTIVE:** consistent with a mechanism but confounded or lacking
  experimental replication.
- **UNTESTED:** requires a new controlled experiment. In particular,
  COVER-then-RANDOM has been implemented and gated but has not been trained.

---

## Part 1 — what background does during I-JEPA pretraining

### Layer 0: settled by architecture and arithmetic

#### The target query contains position, not image content

`src/models/vision_transformer.py` constructs:

```text
predictor_pos_embed : (1, 256, 384), requires_grad=False
mask_token          : (1,   1, 384), requires_grad=True
```

For target cell \(j\), the initial predictor query is:

```text
mask_token + predictor_pos_embed[j]
```

The learned mask token is shared by every target. Therefore the only
target-specific input in the query is the fixed sinusoidal position code; all
image-specific content must arrive through attention to the encoded context.

The position codes are distinct but smooth:

| pair | cosine similarity |
|---|---:|
| `(0,0)` vs `(0,1)` | 0.9857 |
| `(0,0)` vs `(8,8)` | 0.7182 |
| `(0,0)` vs `(15,15)` | 0.6383 |

`VisionTransformerPredictor.forward` concatenates context and target tokens and
runs ordinary transformer blocks with no attention mask between them.
Background context tokens are consequently first-class, ungated inputs.

**PROVEN:** background positions are addressable and background context tokens
can be attended. Architecture alone does not prove that their pixel content is
useful.

#### The loss does not know which targets are anatomy

`src/train_patch.py` uses:

```python
F.smooth_l1_loss(z, h_rep)
```

with the default `reduction="mean"`. Every scalar in every target slot is
weighted equally, regardless of whether the slot is anatomy or background.
Thus the fraction of target slots on background is also the fraction of
elementwise loss terms assigned to background.

This has two immediate consequences:

1. **Training loss is not comparable across masking arms.** The arms predict
   different target populations and, for the blob arm, a different number of
   target observations.
2. **Validation loss is comparable.** `src/train_patch.py` deliberately uses
   the same stock uniform collator for every validation loader.

The mean reduction keeps nominal loss scale from shrinking merely because an
arm has fewer slots. Fewer slots instead mean fewer distinct supervised
constraints per image, not automatically a 2.4× smaller gradient norm.

#### Where each arm spends its target and context budget

`scripts/target_composition.py` evaluated 500 slices from 20 volumes on CPU.
Mean anatomy was 68.256 of 256 cells.

| arm | target slots | target background | unique target cells | context tokens | context background | repeated slots across 4 targets |
|---|---:|---:|---:|---:|---:|---:|
| random | 154.624 | 65.21% | 111.008 | 80.416 | 75.00% | 28.21% |
| oracle | 154.624 | 55.33% | 100.674 | 91.704 | 81.95% | 34.89% |
| envelope | 154.624 | 52.09% | 118.314 | 84.768 | 88.54% | 23.48% |
| anatomy blob | 64.000 | 1.55% | 54.744 | 163.016 | 92.57% | 14.46% |
| COVER-transition | 154.624 | 46.14% | 105.980 | 92.872 | 91.62% | 31.46% |

The blob arm is unusual on two independent axes:

- only **1.55%** of its target slots are background;
- it supplies only 64 target slots versus 154.624, or 2.42× fewer target
  observations.

The final column measures repeated indices after concatenating all four target
groups. It includes inter-block overlap for every rectangular arm. Separately,
`src/masks/utils.py::resample_to_k` pads a short target **with replacement**;
only the blob arm sets `pred_target_k`, so only it can receive within-target
replacement duplicates. The aggregate 14.46% must not be attributed wholly to
that padding mechanism.

### Layer 1: measurements requiring frozen encoders

#### Protocol

`scripts/background_signal_probe.py` evaluated 15 checkpoints from four arms.
Random and oracle checkpoints came from
`yfeng0206/ijepa-3d-oct-checkpoints`; envelope and blob checkpoints were local.
Every checkpoint saw the same 108 slices, crops, stock uniform mask draws, and
context/target indices. Each context-removal comparison was paired within the
same draw. No weights were updated.

#### Claim A: marginal value of context tokens

For each draw, the probe removed the same number \(k\) of background, anatomy,
or uniformly random context tokens and measured anatomy-target error. The
reported token value is the raw rise from full context divided by \(k\):

| checkpoint | background token | anatomy token | anatomy/background |
|---|---:|---:|---:|
| fork ep25 | 0.000169 | 0.001027 | 6.09× |
| random ep50 | 0.000278 | 0.001806 | 6.51× |
| random ep75 | 0.000391 | 0.001611 | 4.12× |
| random ep100 | 0.000405 | 0.001594 | 3.94× |
| oracle ep50 | 0.000277 | 0.001557 | 5.63× |
| oracle ep75 | 0.000357 | 0.001943 | 5.45× |
| oracle ep100 | 0.000373 | 0.001750 | 4.69× |
| envelope ep30 | 0.000185 | 0.001222 | 6.60× |
| envelope ep50 | 0.000265 | 0.001242 | 4.68× |
| envelope ep75 | 0.000357 | 0.001565 | 4.39× |
| envelope ep100 | 0.000309 | 0.001655 | 5.35× |
| blob ep30 | 0.000319 | 0.001126 | 3.53× |
| blob ep40 | 0.000981 | 0.000979 | 1.00× |
| blob ep50 | 0.001085 | 0.000901 | 0.83× |
| blob ep56 | 0.001397 | 0.001030 | 0.74× |

For healthy checkpoints, one anatomy context token is worth roughly 4–6.6
background tokens under this intervention. Along the random lineage,
background-token value rose from 0.000169 at ep25 to 0.000405 at ep100.
Envelope also ended higher at ep100 than ep30, although it peaked at ep75; the
trend is not universally monotonic.

There is an important limit to claim A. Removing any token reduces context.
Against the count-matched **random-removal** control, the background-specific
excess in healthy checkpoints was negative and within about two standard
errors, whereas anatomy removal produced a positive 5.5–8.4σ excess.
Therefore:

- **MEASURED:** background tokens are operationally non-inert under removal,
  and the model can exploit them as part of its context set.
- **NOT PROVEN:** the semantic fact “this location is black” adds unique
  information beyond generic context count, positions, and attention
  normalization. A content-shuffle or matched replacement test would isolate
  that stronger claim.

#### Claim B: skill over a no-context positional reference

The reference predicts the mean target vector at each grid cell and receives
no context. The score is:

```text
skill = 1 - predictor_error / position_only_error
```

| checkpoint | background targets | anatomy targets |
|---|---:|---:|
| fork ep25 | +0.585 | +0.569 |
| random ep100 | **+0.680** | +0.633 |
| oracle ep100 | +0.632 | +0.649 |
| envelope ep100 | +0.604 | +0.629 |
| blob ep50 | +0.132 | +0.218 |

Healthy predictors remove 58–68% of the background error left by the
position-only reference. For random ep100, background skill exceeds anatomy
skill. This is the strongest evidence for the original hypothesis:
background targets are not merely filled from a fixed positional lookup.

The reference is deliberately strong because its per-cell means are estimated
on the same 108 slices used for scoring. That makes the comparison conservative
with respect to an independently fitted positional baseline. It is still a
per-cell mean rather than a formal proof of the exact Smooth-L1 optimum, and it
does not separately ablate image content from context-mask layout.

Background target representations are also not collapsed to a constant:

| checkpoint | cosine within background | cosine within anatomy | effective rank background | effective rank anatomy |
|---|---:|---:|---:|---:|
| random ep100 | 0.246 | 0.323 | 22.5 | 12.8 |
| envelope ep100 | 0.231 | 0.551 | 24.0 | 13.6 |

High diversity alone could reflect nuisance variation or speckle. Combined
with positive context skill, it establishes that a substantial predictable
component exists.

#### Claim C: can every arm represent anatomy versus background?

A linear probe on frozen `target_encoder` patch tokens achieved AUC
**0.979–0.988 for all 15 checkpoints**, with no masking arm separating from the
others.

This refutes only the narrow claim that the blob arm cannot encode the
anatomy/background distinction. The probe used a random cell split rather than
held-out volumes, and anatomy/background is an easy visual distinction; it
does not establish equal disease information or equal representation quality.

### The blob predictor deteriorated after ep30

| blob checkpoint | full-context error | anatomy/background token value | anatomy excess over random removal |
|---|---:|---:|---:|
| ep30 | 0.1049 | 3.53× | +0.00777 ± 0.00121 (6.42σ) |
| ep40 | 0.2244 | 1.00× | +0.00191 ± 0.00182 (1.05σ) |
| ep50 | 0.2712 | 0.83× | +0.00057 ± 0.00210 (0.27σ) |
| ep56 | 0.2895 | 0.74× | +0.00062 ± 0.00226 (0.28σ) |

At ep50 its skill over the position-only reference was only +0.218 on anatomy
and +0.132 on background. By ep56, removing anatomy tokens was no worse than
removing a count-matched random set, while the raw background-token value
exceeded anatomy-token value.

The observations are consistent with a predictor drifting toward positional
shortcutting after being trained on small, almost entirely anatomy-only target
sets. They also provide a concrete diagnostic consistent with the blob arm's
0.8654 downstream plateau versus 0.8807 for envelope. They do **not** prove
that target composition caused the plateau: the blob arm also differs in
target geometry, target count, replacement padding, and training trajectory.

---

## Part 2 — where the downstream glaucoma signal is read out

### Why region-specific pooling was needed

The published frozen protocol uses:

```python
out.mean(dim=1)
```

over all 256 patch cells. In the cached downstream masks, anatomy occupies
about 23.4% of cells, so background-position tokens receive about 76.6% of the
pooling weight by construction. That arithmetic does not say whether those
tokens are useful.

`scripts/downstream_region_auc.py` kept the frozen encoder and dataset fixed
and retrained the same lightweight probe on three pooled feature vectors:

- uniform mean over all cells;
- mean over anatomy cells only;
- mean over background cells only.

The paired sample was 2,000 training, 600 validation, and 1,000 test volumes.
Each volume contributed 25 stratified random slices, one from each equal depth
bin. The same volumes and slice indices were used for every arm.

### Anatomy-mask quality and provenance

Val/Test do not have MIRAGE guides. A small
`HistGradientBoostingClassifier`, trained on cheap pixel/position features
against MIRAGE masks from Training, supplies an encoder-independent mask at
downstream inference.

| calibration metric | value |
|---|---:|
| held-out-volume patch AUC | 0.97949 |
| threshold | 0.45 |
| Dice | 0.87181 |
| precision | 0.85123 |
| recall | 0.89340 |

The model is byte-identical across encoder arms. No MIRAGE guide is needed at
Val/Test inference, although MIRAGE masks were used as Training supervision.

The current stored mask cache has anatomy rates of 23.35% on Training, 23.41%
on Validation, and 23.35% on Test; the paired 1,000-volume/25-slice test sample
used for attribution is 23.37%. These artifacts do **not** support the earlier
quoted 23.85% prediction rate, so this report uses the cache values. The
separately quoted 22.90% MIRAGE reference rate is not present in the listed
JSON/CSV artifacts and is not used as a primary result.

A development bug materially changed this rate: `OCTVolumeDataset` returns
unnormalized slices, while the mask model was fitted on ImageNet-normalized
images. Applying the mask model before `imagenet_normalize` under-predicted
anatomy (about 0.145 rather than the expected roughly 0.23–0.24). The cache
builder now normalizes before feature extraction.

### Region-pooled test AUC

| frozen encoder | all cells | anatomy only | background only | anatomy − all |
|---|---:|---:|---:|---:|
| random ep50 | 0.8608 | **0.8747** | 0.8544 | +0.0138 |
| oracle ep50 | 0.8683 | **0.8746** | 0.8652 | +0.0064 |
| envelope ep50 | 0.8730 | **0.8784** | 0.8701 | +0.0055 |
| blob ep50 | 0.8593 | **0.8606** | 0.8590 | +0.0013 |

**MEASURED:** anatomy-only pooling improved AUC for every encoder in this
paired run. The largest gain was random ep50 (+0.0138). This supports replacing
uniform all-cell pooling with anatomy-weighted or anatomy-only pooling before
spending compute on another encoder.

Background-only pooling remained high (0.854–0.870), but this result has four
important caveats:

1. Mask recall is 0.893, so about 10.7% of true anatomy cells can leak into the
   nominal background pool.
2. Every ViT patch token has globally attended the image. A token whose
   **position** lies in background can already contain retinal information.
3. A separate head was fitted for each pooling region. This asks whether signal
   is readable, not whether the original all-cell head uses it.
4. This was a 1,000-volume, 25-slice test subset with one probe seed and no
   confidence interval, not a new full-protocol benchmark.

The clean outstanding test is to erode the background mask by several patch
rows away from the retina and rerun both pooling and attribution.

### Exact attribution under one fixed head

`scripts/patch_attribution.py` answers the third caveat. For mean pooling
followed by `LinearHead = LayerNorm -> Linear`, the logit can be decomposed
exactly into patch contributions.

Let \(f\) be the mean of patch vectors \(h_{sj}\), and define:

```text
a = w * gamma / (sigma(f) + eps)
A = sum(a)
```

Then:

```text
contrib(s,j) =
    [a · h_sj - A * mean_d(h_sj)] / (S * 256)

logit = sum_sj contrib(s,j) + (w · beta + b)
```

The maximum absolute reconstruction residual was
`5.07e-07`, so the decomposition is numerically exact.

| arm | one anatomy patch | one background patch | background/anatomy per patch | background/anatomy total absolute mass | AUC full / anatomy / background contribution |
|---|---:|---:|---:|---:|---:|
| random ep50 | 0.000268 | 0.000340 | **1.27×** | 4.16× | 0.8552 / 0.8639 / 0.8470 |
| oracle ep50 | 0.000328 | 0.000382 | **1.17×** | 3.82× | 0.8632 / 0.8643 / 0.8594 |
| envelope ep50 | 0.000337 | 0.000358 | **1.06×** | 3.49× | 0.8652 / 0.8654 / 0.8586 |
| blob ep50 | 0.000415 | 0.000360 | **0.87×** | 2.84× | 0.8566 / 0.8464 / 0.8557 |

“Total mass” is the sum of absolute contributions over all evaluated cells,
not signed evidence for one class. Background dominates it because there are
roughly 3.3× more background cells and, in three arms, each background-position
cell is also at least as influential.

The per-patch background/anatomy ratio falls monotonically as masking becomes
more anatomy-focused: 1.27 → 1.17 → 1.06 → 0.87. Blob is the only arm that
weights anatomy positions more strongly per patch, yet its background
contribution discriminates better than its anatomy contribution
(0.8557 versus 0.8464). It appears to emphasize anatomy positions while
extracting relatively little class separation from them.

The attribution heads are refitted published-format `LayerNorm -> Linear`
heads, whereas the region-pooling table retrains standardized linear probes
separately per region. Their “full” AUCs therefore differ slightly; comparisons
should be made within each table.

**Interpretation boundary:** these results concern token positions after global
self-attention. They do not establish a disease signal in optically black
pixels.

---

## Part 3 — COVER-then-RANDOM is implemented, not trained

The requested policy greedily places rectangles until anatomy coverage reaches
the hard visibility floor, then spends every leftover slot as a plain uniform
I-JEPA rectangle while still respecting that floor.

`src/masks/cover.py` now exposes:

- `fill="transition"` — shipped boundary-straddling behavior;
- `fill="random"` — unconstrained uniform leftovers;
- `fill="random_legal"` — uniform over windows that preserve the anatomy
  visibility floor.

Each slot records `info["slot_kind"]` for provenance. Integration is through
`curriculum.cover_fill`; `configs/patch_cover_random_ep25.yaml` selects
`random_legal`.

The stored 600-slice artifact was generated with `anatomy_tau=0.30`:

| variant | anatomy hidden | anatomy visible | cover blocks | random/transition blocks | floor OK | usable |
|---|---:|---:|---:|---:|---:|---:|
| unconstrained random | 86.37% | 13.63% | 2.948 | 1.052 random | 60.0% | 59.83% |
| random legal | 84.42% | 15.58% | 2.948 | 1.052 random | 100% | 99.83% |
| transition | 84.42% | 15.58% | 2.948 | 1.045 transition + 0.007 random | 100% | 99.83% |

Thus legal random placement and transition have identical coverage at the same
threshold. Unconstrained random does **not**: it hides an extra 1.95 percentage
points by violating the floor on 40% of slices.

The production config uses `anatomy_tau=0.10`, while
`scripts/cover_random_probe.py` hard-codes 0.30. A documentation-time parity
rerun on the same 600 cases at 0.10 found 84.58% hidden, 15.42% visible, 3.215
cover plus 0.785 legal-random blocks, 100% floor compliance, and 99.83% usable
samples. The conclusion is unchanged, but the original block-split statistics
must not be presented as exact production-config numbers.

Because `cover_leave_frac == cover_min_visible_frac == 0.15`, the soft target
and hard floor meet at the same boundary. Discrete cells and soft mass mean
exactly 85% hidden is normally unreachable without crossing the floor; both
legal variants report about 84.4–84.6%. A true 85% target requires separating
the soft stop from the hard floor, for example by lowering the latter and
re-auditing.

The 50-slice edge-case gate passed with no fallbacks, no zero-visible slice,
and no assertion failures. Mean strict-occupancy anatomy hidden was 83.39%
(57.14–88.10%), with at least three strict-occupancy anatomy cells visible.
Thirty-one of 50 cases used three cover plus one random block.

The gate's strict occupancy reference and COVER's soft support are not
interchangeable. Thirteen of 50 slices left less than 15% visible under the
strict occupancy definition even though the sampler's own soft-mass floor
passed. One sparse case also crossed the stricter four-cell interpretation:
`data_08569/slice_199` has seven anatomy cells under the audit's strict
occupancy threshold and leaves three visible, while COVER's soft-score support
contains nine cells and leaves four visible. `floor_violation=False` is correct
under the sampler's own definition, but the stricter audit can report less than
the nominal fractional floor and, in this one case, less than the nominal cell
floor. No slice reached zero visible anatomy.

See [cover_random.md](cover_random.md) for the implementation and gate details.
No COVER-then-RANDOM checkpoint or downstream AUC exists yet.

---

## Reusable bugs and traps found during the investigation

| issue | consequence | durable rule |
|---|---|---|
| `anatomical_prior` called without `imgs_cpu` | `bias_active=False`; oracle silently becomes uniform random | Pass `imgs_cpu=images`, as `scripts/five_arm_audit.py` and the corrected target-composition probe do |
| Anatomy mask applied to unnormalized `OCTVolumeDataset` slices | Anatomy rate collapsed to about 0.145 | Call `imagenet_normalize` before the mask model, just as before the encoder |
| NumPy 2 removed `ndarray.ptp()` | Plot scripts fail at runtime | Use `np.ptp(x)` |
| `resample_to_k` pads with replacement | Short blob targets contain repeated, zero-new-information slots | Treat `pred_target_k` as a supervision-policy change; only the blob arm enables it |

The first failure was caught because oracle masks were byte-identical to random.
It is particularly dangerous because no exception is raised.

---

## What is proven, suggestive, and still untested

### Settled without training

- Target queries contain fixed position plus one shared mask token.
- Background context tokens are attended without a gate.
- Smooth-L1 weights target slots without anatomy awareness.
- Target/context composition differs sharply across arms.
- Uniform validation masking makes validation loss comparable; training loss is
  task-dependent.

### Measured on frozen checkpoints

- Healthy predictors beat a position-only reference by 58–68% on background
  targets.
- Anatomy context tokens are much more valuable than background tokens.
- All target encoders easily encode the anatomy/background distinction.
- The blob predictor deteriorates sharply after ep30.
- Anatomy-only pooling improves AUC in all four tested ep50 encoders.
- Disease discrimination is readable from tokens located at background
  positions.

### Suggestive, not causal

- The blob arm's target starvation and replacement padding caused its predictor
  collapse or downstream plateau.
- A 52–55% background target share is optimal because oracle/envelope were the
  strongest historical arms.
- The monotonic downstream attribution shift was caused by the masking policy
  rather than other arm differences.

### Untested

- Whether intentionally allocating target slots to background improves
  downstream AUC under a single-variable masking ablation.
- Whether background-only AUC survives erosion several patches away from the
  retina.
- Whether the context result survives cross-image content shuffling or matched
  token replacement.
- Whether the anatomy-pooling gain persists under the full-volume protocol,
  multiple probe seeds, and an untouched evaluation split.
- Whether COVER-then-RANDOM improves representations. It has not been trained.

## Decisions this supports

1. **Use anatomy-weighted pooling first.** It is a low-cost downstream change
   that improved every tested encoder without modifying encoder weights.
2. **Do not resume the blob arm blindly.** Resume only with predictor-health
   monitoring and a corrected supervision design, or restart from the shared
   ep25 fork.
3. **COVER-then-RANDOM is implementation-ready pending approval.** The legal
   placement policy and edge gate pass; the production-threshold parity check
   preserves the result. There is still no training evidence.
4. **Run eroded-background evaluation next.** It is the key missing control
   before making a publishable claim about background-region disease signal.

## Artifact and script index

| question | artifacts | scripts |
|---|---|---|
| target/context composition | `reports\target_composition\summary.csv`, `per_image.csv` | `scripts\target_composition.py` |
| frozen background signal | `reports\background_signal\background_signal.json`, `marginal_token_value.csv`, `skill_scores.json` | `background_signal_probe.py`, `background_skill_score.py` |
| region-pooled AUC | `reports\downstream_region_auc\region_auc_summary.csv` and per-arm JSON | `downstream_region_auc.py`, `merge_region_auc.py` |
| exact patch attribution | `reports\patch_attribution\attribution_summary.csv`, `*_attrib.json` | `patch_attribution.py`, `fit_head_from_features.py` |
| anatomy-mask calibration | `reports\anatomy_mask_calib\mask_model_report.json` | `fit_anatomy_mask.py` |
| COVER audit | `reports\cover_random\summary.csv`, `per_slice.csv` | `cover_random_probe.py` |
| COVER edge gate | `reports\edge_cases_random_legal\cover_edge_cases.json` | `mask_edge_case_test.py` |
