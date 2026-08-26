# BACKGROUND_SIGNAL.md — testing the PI's background/position theory

Date: 2026-08-26. Branch: `docs/background-signal-findings`.
GPU was in use throughout; every number below is either read from a pre-existing
artifact or computed CPU-only. No training, no probe training on GPU, no GPU
memory allocated.

The hypothesis under test, in the PI's words:

> "Sometimes we learned from random frozen AUC probabilities that even the
> background matters. I assume this is due to position embedding, which might
> indicate that certain areas are not important, like a black background."

Split into:

- **H-a** Background (non-tissue, dark vitreous) regions carry information that
  helps the downstream glaucoma classifier, rather than being pure noise.
- **H-b** The mechanism is positional: because I-JEPA adds position embeddings,
  the encoder can learn "this location is empty background", and that location
  information is itself diagnostic.

Evidence labels: **[MEASURED]** read from a file or computed here;
**[INFERRED]** a reading of measured numbers that is not itself measured;
**[PENDING]** requires an experiment that has not been run.

---

## 0. Verdicts up front

| claim | verdict | one-line basis |
|---|---|---|
| H-a for **pretraining** (background is a real learning signal) | **SUPPORTED** | random ep100 predictor beats a per-position no-context reference by 0.680 on background targets, higher than its 0.633 on anatomy targets |
| H-a for **downstream classification** (background helps the glaucoma classifier) | **CONTRADICTED in its strong form** | background pooled features are 95.2 percent linearly reconstructible from anatomy pooled features; after residualising, background predicts glaucoma at test AUC 0.5515; adding background to anatomy changes test AUC by -0.0076 (95 percent CI -0.0139 to -0.0012) for the random arm and by less than +0.002 for every other arm |
| H-b (the positional mechanism) | **INSUFFICIENT EVIDENCE** for the full causal chain; its **first link is now MEASURED and strong** | 90.8 percent of the across-position variance of the layer-0 encoder input at background cells is contributed by `pos_embed`, versus 40.8 percent at tissue cells; but nothing measures that this positional content is what drives downstream AUC, and the H-a downstream result argues the downstream payoff is near zero |

The single most surprising thing found: **the PI's premise and its proposed
mechanism point in opposite directions.** The mechanism (H-b link 1) is real and
larger than expected — background tokens really are almost pure position at the
encoder input. But the payoff (H-a downstream) is essentially zero once
redundancy with tissue is removed. Background-position tokens score 0.867 AUC on
their own because they are near-copies of the tissue tokens after global
self-attention, not because background positions add anything.

---

## 1. Inventory of pre-existing evidence

Everything in this section already existed before this session. Paths are exact.

### 1.1 A prior investigation of this exact question already exists

`C:\Users\Gary\Desktop\jepa\docs\experiments\masking\background_signal.md`
(dated 2026-08-14, 26.3 KB) already decomposes the PI's idea into three claims
and answers them. Its own summary table:

| its claim | its verdict | its evidence |
|---|---|---|
| A. Background as context is informative | Partly supported | one anatomy context token is worth 4-6 background tokens |
| B. Background as a target is a real learning signal | Confirmed relative to a strong position-only baseline | predictors remove 58-68 percent of the background error left by a per-position no-context reference |
| C. An arm that rarely predicts background cannot represent it | Refuted | all 15 target encoders separate anatomy from background at AUC 0.979-0.988 |

[MEASURED] Its stated outstanding control, still not run, is: "erode the
background mask by several patch rows away from the retina and rerun both
pooling and attribution."

### 1.2 Masking composition — how much background each arm actually masks

`C:\Users\Gary\Desktop\jepa\results\masking\fair\arms_with_connectivity.json`
[MEASURED], per-slice cell counts out of 256:

| arm | hidden | anat_hidden | bg_hidden | anat_ctx | on_anatomy_pct |
|---|---:|---:|---:|---:|---:|
| random_default | 110.438 | 24.032 | 86.406 | 13.632 | 21.881 |
| random_matched | 49.419 | 10.124 | 39.295 | 30.528 | 20.551 |
| envelope_default | 113.260 | 35.889 | 77.371 | 6.513 | 31.760 |
| anatomy_4conn | 51.527 | 34.935 | 16.592 | 9.615 | 67.485 |
| anatomy_8conn | 54.242 | 39.292 | 14.950 | 5.887 | 72.111 |

`D:\jepa_phase0\reports\target_composition\summary.csv` (quoted in
`background_signal.md`) [MEASURED], 500 slices from 20 volumes, mean anatomy
68.256 of 256 cells:

| arm | target slots | percent of target slots on background | context tokens | percent of context on background |
|---|---:|---:|---:|---:|
| random | 154.624 | 65.21 | 80.416 | 75.00 |
| oracle | 154.624 | 55.33 | 91.704 | 81.95 |
| envelope | 154.624 | 52.09 | 84.768 | 88.54 |
| anatomy blob | 64.000 | 1.55 | 163.016 | 92.57 |
| COVER-transition | 154.624 | 46.14 | 92.872 | 91.62 |

[INFERRED] The PI's framing is correct: the strong random baseline spends about
two thirds of its predictor targets on background.

### 1.3 Frozen-probe test AUCs the PI supplied as ground truth

MeanPool + Linear, seed 42, not recomputed here [MEASURED, PI-supplied]:
ep25 ancestor 0.8487; random ep50 0.8641, ep75 0.8723, ep100 0.8746;
envelope ep100 0.8807; oracle (intensity) ep100 0.8855.

Cross-checked against
`D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep100.json`
[MEASURED], which stores the same three ep100 AUCs (random 0.8746, oracle
0.8855, envelope 0.8807) alongside composition:

| arm | AUC | pct_anat_hid | pct_tgt_anat | pct_ctx_anat |
|---|---:|---:|---:|---:|
| random | 0.8746 | 53.044 | 31.583 | 26.378 |
| oracle | 0.8855 | 61.585 | 39.688 | 19.344 |
| envelope | 0.8807 | 77.582 | 43.193 | 11.384 |

[INFERRED] With n = 3 arms this file's Spearman values (`pct_anat_hid` 0.5,
`pct_tgt_anat` 0.5, `ctx` 1.0) carry no statistical weight and should not be
cited as a trend.

### 1.4 The predictor's skill on background targets — the strongest pre-existing H-a evidence

`D:\jepa_phase0\reports\background_signal\skill_scores.json` [MEASURED], 108
slices, `skill_vs_pos = 1 - err_predictor / err_position_only_reference`:

| checkpoint | background skill | anatomy skill | bg err_pred | bg err_pos |
|---|---:|---:|---:|---:|
| fork ep25 | 0.5850 | 0.5691 | 0.12650 | 0.30481 |
| random ep100 | **0.6798** | 0.6334 | 0.10976 | 0.34279 |
| oracle ep100 | 0.6321 | 0.6486 | 0.12299 | 0.33430 |
| envelope ep100 | 0.6043 | 0.6287 | 0.13110 | 0.33127 |
| blob ep50 | 0.1324 | 0.2175 | 0.31436 | 0.36235 |

[MEASURED] For the random arm at ep100 the predictor's background skill (0.6798)
exceeds its anatomy skill (0.6334). The reference is a per-grid-cell mean fitted
on the same 108 slices, so it is a strong, position-only baseline.

[INFERRED] This is the load-bearing evidence that background targets are a real
pretraining signal, and it also constrains H-b: a purely positional account
would predict skill near zero against a per-position reference. The predictor is
doing something with image content at background positions that a positional
lookup cannot do.

### 1.5 Marginal value of background context tokens

`D:\jepa_phase0\reports\background_signal\marginal_token_value.csv` (table
reproduced in `background_signal.md`) [MEASURED]. Raw error rise per removed
token:

| checkpoint | background token | anatomy token | anatomy / background |
|---|---:|---:|---:|
| fork ep25 | 0.000169 | 0.001027 | 6.09 |
| random ep50 | 0.000278 | 0.001806 | 6.51 |
| random ep75 | 0.000391 | 0.001611 | 4.12 |
| random ep100 | 0.000405 | 0.001594 | 3.94 |
| envelope ep100 | 0.000309 | 0.001655 | 5.35 |
| blob ep56 | 0.001397 | 0.001030 | 0.74 |

[MEASURED] Background-token value along the random lineage rises 0.000169
(ep25) to 0.000405 (ep100), a 2.4x increase.
[MEASURED] Against a count-matched random-removal control the background-specific
excess was negative and within about two standard errors, whereas anatomy
removal produced a +5.5 to +8.4 sigma excess. So background tokens are not inert,
but their removal effect is not distinguishable from generic token removal.

### 1.6 Background representations are not collapsed

`background_signal.md` [MEASURED]:

| checkpoint | cosine within background | cosine within anatomy | eff. rank background | eff. rank anatomy |
|---|---:|---:|---:|---:|
| random ep100 | 0.246 | 0.323 | 22.5 | 12.8 |
| envelope ep100 | 0.231 | 0.551 | 24.0 | 13.6 |

`C:\Users\Gary\Desktop\jepa\results\masking\class_relations\class_relations.json`
[MEASURED], mean cosine within/between anatomical classes:

| encoder | bg_bg | within_inner | inner_choroid | contrast inner vs choroid | contrast tissue vs bg | cohens_d | discrim_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| JEPA untrained (control) | 0.7842 | 0.6957 | 0.5260 | 0.1939 | 0.4350 | 1.3803 | 0.8288 |
| JEPA ep30 (anatomy) | 0.4435 | 0.8973 | 0.8031 | 0.0785 | 0.5166 | 0.8361 | 0.7714 |
| JEPA ep100 (envelope) | 0.3460 | 0.8365 | 0.7185 | 0.0921 | 0.5058 | 0.6543 | 0.6945 |
| MIRAGE encoder | 0.4608 | 0.6637 | 0.3041 | 0.3621 | 0.3330 | 2.8149 | 0.9773 |

[MEASURED] Two things happen during I-JEPA pretraining, in opposite directions.
Tissue self-similarity rises (`within_inner` 0.6957 untrained to 0.8365 at
ep100) while tissue-vs-tissue contrast collapses (0.1939 to 0.0921), which is
the known finding that pretraining makes tissue separability worse. At the same
time background self-similarity **falls** from 0.7842 to 0.3460, i.e. background
tokens become far more differentiated from each other.

[INFERRED] Training does the opposite of collapsing background to a constant.
Combined with the effective-rank numbers (background 22.5, anatomy 12.8) the
representation devotes more dimensions to background than to tissue. This is
consistent with H-b but does not establish it — differentiation could come from
speckle content rather than from position. See section 3.

### 1.7 Predictor error is better explained by position than by anatomy

`C:\Users\Gary\Desktop\jepa\results\masking\error_vs_anatomy\error_confound_check.json`
[MEASURED], epoch 100, 20 slices, 400 mask draws:

| quantity | value |
|---|---:|
| corr(error, distance to context centroid) | **+0.5687** |
| corr(error, distance to nearest visible) | +0.3162 |
| corr(error, patch intensity) | -0.3647 |
| corr(error, patch variance) | -0.3243 |
| corr(error, anatomy occupancy) | -0.2675 |
| partial corr(error, anatomy \| intensity) | -0.0367 |
| partial corr(error, anatomy \| all + centroid) | +0.0425 |
| err on anatomy / err off anatomy | 0.15869 / 0.19485 |

[MEASURED] The strongest single predictor of predictor error is a purely
geometric quantity — distance to the context centroid (+0.5687) — and the
anatomy effect essentially vanishes under partial correlation (-0.0367 given
intensity, +0.0425 given the full set).

[INFERRED] This is the best pre-existing indirect support for a positional
account of what the predictor is solving. It concerns the pretraining task, not
the downstream classifier.

Companion file
`results\masking\error_vs_anatomy\error_vs_anatomy.json` [MEASURED]:
`err_on_anatomy` 0.15640, `err_off_anatomy` 0.20184, ratio 0.77489. Binned by
anatomy occupancy, the `[0.00,0.05)` bin holds 64.96 percent of cells at error
0.21255 while the `[0.95,1.01)` bin holds 5.45 percent at error 0.15931.

### 1.8 Where the downstream head reads from — region-pooled AUC

`D:\jepa_phase0\reports\downstream_region_auc\region_auc_summary.csv`
[MEASURED], 2000/600/1000 paired volumes, 25 stratified slices each, ep50
checkpoints, a separate probe refitted per region:

| arm | all | anatomy | background | anatomy − all |
|---|---:|---:|---:|---:|
| random_ep50 | 0.86083 | **0.87466** | 0.85436 | +0.01383 |
| oracle_ep50 | 0.86826 | **0.87465** | 0.86523 | +0.00639 |
| envelope_ep50 | 0.87299 | **0.87844** | 0.87008 | +0.00545 |
| blob_ep50 | 0.85935 | **0.86060** | 0.85905 | +0.00125 |

[MEASURED] Anatomy-only pooling beats all-cell pooling for every encoder, and
background-only pooling is worse than all-cell pooling for every encoder.
Background-only nevertheless stays at 0.854-0.870.

Mask quality, `D:\jepa_phase0\reports\anatomy_mask_calib\mask_model_report.json`
[MEASURED]: held-out-volume patch AUC 0.97949, threshold 0.45, Dice 0.87181,
precision 0.85123, recall 0.89340. Cache anatomy rates 23.35 percent Training,
23.41 percent Validation, 23.35 percent Test.

[INFERRED] Recall 0.89340 means about 10.7 percent of true anatomy cells leak
into the nominal background pool, so "background-only" AUC is an upper bound on
what background positions contribute.

### 1.9 Exact per-patch attribution under one fixed head

`D:\jepa_phase0\reports\patch_attribution\*_attrib.json` [MEASURED], exact
LayerNorm-then-Linear decomposition, max reconstruction residual 5.066e-07:

| arm | per-patch anatomy | per-patch background | bg/anat per patch | AUC full / anatomy contrib / background contrib | corr(anat contrib, bg contrib) | total mass bg/anat |
|---|---:|---:|---:|---:|---:|---:|
| random_ep50 | 0.00026793 | 0.00034029 | **1.2701** | 0.85525 / 0.86385 / 0.84696 | **0.90635** | 4.1646 |
| oracle_ep50 | 0.00032791 | 0.00038204 | 1.1651 | 0.86321 / 0.86425 / 0.85944 | 0.95394 | 3.8202 |
| envelope_ep50 | 0.00033675 | 0.00035846 | 1.0644 | 0.86523 / 0.86540 / 0.85859 | 0.91030 | 3.4903 |
| blob_ep50 | 0.00041550 | 0.00035991 | 0.8662 | 0.85665 / 0.84642 / 0.85568 | 0.94827 | 2.8403 |

Also in `random_ep50_attrib.json`: `mean_abs_share_anatomy` 0.21367 against
`anatomy_cells_frac` 0.23368; `std_anat` 0.26614, `std_bg` 1.15749;
`total_abs_anatomy` 400.735, `total_abs_background` 1668.884.

[MEASURED] For the random arm, a background-position patch is 1.27x as
influential as an anatomy-position patch, and background carries 4.16x the total
absolute mass. The per-patch ratio falls monotonically as masking gets more
anatomy-focused: 1.2701, 1.1651, 1.0644, 0.8662.

[MEASURED] The anatomy and background contributions are correlated at 0.90635 to
0.95394 across the four arms. This is the key pre-existing hint that background
is largely redundant, and it motivated the new test in section 2.2.

### 1.10 The encoder only sees context tokens

`C:\Users\Gary\Desktop\jepa\results\masking\gate_real\elementwise_gate_probe.json`
[MEASURED]: `grad_sum_inside_context` 0.10347, `grad_sum_outside_context` 0.0,
`grad_sum_on_target_patches` 0.0, `nonzero_patches_outside_context` 0 of 1480.

[MEASURED] Background influences pretraining only through being in the context
set or in the target set; there is no third channel. This is architectural.

### 1.11 Occlusion attribution on the fine-tune probes

`C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md` [MEASURED]:
slice-level MeanPool-CrossAttnPool agreement r = 0.94; patch-level r = 0.35-0.48
after the fp16 fix; 84-91 percent of patches have a bootstrap CI excluding zero;
"per-patch attribution concentrates on the B-scan center"; fp32 global max
per-patch |delta logit| is 0.003. The population two-peak slice structure is an
OD/OS storage mirror artefact (corr(c1, flip(c2)) = +0.971 MeanPool, +0.988
CrossAttnPool), not bilateral anatomy.

[MEASURED] The underlying occlusion `.npz` arrays are on blob at
`ijepa-interpretability/`; the search of `C:` and `D:` found no local copy, so
the share-of-attribution-on-background test could not be run against the
occlusion maps. It was run instead against the exact patch-attribution maps in
section 2.1, which are local.

### 1.12 The position-embedding bug and its fix

Commits `721cd26` (code) and `5b9f20a` (docs), both authored 2026-04-10
[MEASURED]:

- In `_get_2d_sincos_pos_embed_from_grid_proper` the meshgrid indices were
  swapped. `np.meshgrid(w, h)` with default `'xy'` indexing puts W in `grid[0]`
  and H in `grid[1]`; the code read `grid[0, 0, :, 0]` for H (all zeros) and
  `grid[1, 0, 0, :]` for W (all zeros).
- Result: **all 256 patch positions received identical positional embeddings —
  unique rows = 1.**
- `pos_embed` is `nn.Parameter(..., requires_grad=False)`
  (`src\models\vision_transformer.py:404-407`), so it was never corrected by
  training.
- ImageNet-init runs were affected too because
  `scripts/download_imagenet_vit.py` skips `pos_embed` during conversion.
- Fix: `grid[1, 0, :, 0]` for H and `grid[0, 0, 0, :]` for W. Verified unique
  rows = 256.

[MEASURED] Confirmed still fixed in the working tree:
`src\models\vision_transformer.py:114-115`, and re-verified numerically in
section 2.3 (unique rows 256 of 256).

Best pre-posfix downstream result, recovered from
`git show eae19c3~1:docs/experiments/README.md` [MEASURED]: run F1,
"Random to SSL ep11", Val AUC 0.828, Test AUC **0.834**, d=3 MLP probe with
ImageNet normalisation.

[INFERRED] This is the closest thing the project has to a pos_embed ablation,
and it is **not usable as one**. The confounds are severe: 11 pretraining epochs
against 100; a d=3 MLP probe against MeanPool + Linear; a different evaluation
era. The 0.834-versus-0.8746 gap cannot be attributed to `pos_embed`.

### 1.13 Files checked and found not to bear on H-a or H-b

- `results\masking\slice_pos\slice_pos.json` and `depth_profile.json`
  [MEASURED]: relative slice-position error reduced 29.90 percent (middle) and
  26.18 percent (stratified); depth-band spread only 3.52 pp across five bands.
  This is about **axial slice index**, a different position axis from the
  in-plane patch grid that H-b concerns.
- `results\masking\region_split.json` [MEASURED]: mask-placement region shares
  (oracle `target_inner_share_of_tissue` 0.43270, mirage 0.39999). Describes
  where masks land, not what the encoder does with background.
- `results\masking\latent_probe\latent_anatomy.json` [MEASURED]: JEPA
  anatomy-vs-background token probe AUC 1.0 with silhouette 0.17900 (against
  MIRAGE H0 silhouette 0.82069); CKA MIRAGE-encoder vs JEPA 0.47978. Confirms
  separability, adds nothing on background usefulness.
- `results\masking\b2_probe\b2_predictor_probe.json` [MEASURED]: collator
  truncation loses 36.95 percent of target cells. A supervision bug shared by
  all arms; not background-specific.

---

## 2. New CPU-only analyses run in this session

All three scripts are in `autopilot\bgsig\` with their JSON outputs beside them.
All read only artifacts that already existed. No GPU.

### 2.1 Attribution mass by grid position versus anatomy frequency

Script `autopilot\bgsig\a1_position_attribution.py`, output
`a1_position_attribution.json`.
Inputs: the 256-cell `patch_absmean` / `patch_mean` maps in
`D:\jepa_phase0\reports\patch_attribution\*_attrib.npz`, and the bit-packed
anatomy masks in `D:\jepa_phase0\reports\anatomy_mask_cache\Test_s100_r256.npz`
(3000 volumes x 100 slices = 300,000 slice instances, unpacked to per-cell
anatomy frequency).

[MEASURED] Global anatomy rate 0.23350, matching the documented cache value.
Of 256 cells, 23 are anatomy in at most 1 percent of slices, 65 in at most 5
percent, 87 in at most 10 percent. No cell is background in 100 percent of
slices.

| arm | Pearson(\|attr\|, anatomy freq) | Spearman | mass share on cells with anat freq <= 0.01 (9.0 percent of cells) | enrichment | mean \|attr\| near-always-background / mostly-anatomy |
|---|---:|---:|---:|---:|---:|
| random_ep50 | **-0.3945** (p=5.8e-11) | **-0.4645** (p=4.2e-15) | 0.0982 | **1.093** | **1.291** |
| oracle_ep50 | +0.1364 (p=0.029) | +0.1381 (p=0.027) | 0.0779 | 0.867 | 0.842 |
| envelope_ep50 | +0.3005 (p=9.7e-07) | +0.4161 (p=3.8e-12) | 0.0872 | 0.971 | 0.895 |
| blob_ep50 | +0.4877 (p=1.1e-16) | +0.4360 (p=2.7e-13) | 0.0873 | 0.972 | 0.808 |

[MEASURED] For the random arm — the arm the PI's observation is about — the
head's per-cell attribution magnitude is **negatively** correlated with how often
that cell contains tissue, and near-always-background cells are 1.291x as
influential per cell as mostly-anatomy cells.

[MEASURED] The correlation is ordered by how anatomy-focused the masking policy
is: random -0.39, oracle +0.14, envelope +0.30, blob +0.49.

[MEASURED] Enrichment (mass share divided by cell share) is 0.87 to 1.09 in
every arm at every threshold tested. Background positions are neither favoured
nor suppressed relative to their count; they are used roughly proportionally.

[INFERRED] This is genuine support for the weak form of H-a: background
positions are not discounted by the trained readout, and for the random arm they
are actively favoured. It says nothing about whether the information at those
positions originated in background pixels.

### 2.2 Does background add anything the anatomy pool does not already have

Script `autopilot\bgsig\a2_region_incremental.py`, output
`a2_region_incremental.json`.
Input: `D:\jepa_phase0\reports\region_features\{arm}_ep50_s100.pt`, which caches
mean-pooled 768-d vectors for the all / anatomy / background regions over the
paired 2000 / 600 / 1000 volume splits.
Method: standardise on Training; logistic regression with C selected on
Validation from {0.001, 0.01, 0.1, 1, 10}; report Test AUC with a 2000-sample
bootstrap CI. Then residualise: fit Ridge(alpha=10) from the anatomy pool to the
background pool on Training only, and probe the residual.

This is the incremental-information test that the existing
`background_signal.md` does not contain — it only probes each region in
isolation.

| arm | all | anatomy | background | anatomy+background | **background residualised on anatomy** | anatomy residualised on background | Ridge R2 (Test) background from anatomy |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0.8712 | 0.8811 | 0.8667 | 0.8735 | **0.5515** [0.5165, 0.5893] | 0.6370 [0.6037, 0.6715] | **0.9522** |
| oracle | 0.8706 | 0.8749 | 0.8674 | 0.8752 | **0.5310** [0.4949, 0.5664] | 0.5791 [0.5430, 0.6149] | 0.9424 |
| envelope | 0.8776 | 0.8797 | 0.8767 | 0.8812 | **0.5371** [0.5026, 0.5718] | 0.5859 [0.5500, 0.6197] | 0.9474 |
| blob | 0.8727 | 0.8730 | 0.8739 | 0.8727 | **0.6008** [0.5659, 0.6378] | 0.5701 [0.5330, 0.6074] | 0.9746 |

Paired bootstrap difference, (anatomy + background) minus anatomy alone
[MEASURED]:

| arm | mean delta | 95 percent CI |
|---|---:|---|
| random | **-0.0076** | [-0.0139, -0.0012] |
| oracle | +0.0003 | [-0.0034, +0.0040] |
| envelope | +0.0016 | [-0.0015, +0.0047] |
| blob | -0.0003 | [-0.0040, +0.0032] |

[MEASURED] The background pool is 94.2 to 97.5 percent linearly reconstructible
from the anatomy pool on held-out Test volumes. Once that shared component is
removed, background predicts glaucoma at 0.531 to 0.601. Concatenating
background onto anatomy changes Test AUC by less than 0.002 for three arms and
by a **statistically significant -0.0076** for the random arm.

[MEASURED] The symmetric control is not symmetric: anatomy residualised on
background reaches 0.637 for the random arm, clearly above the 0.5515 the
background residual reaches. Anatomy carries more unique information than
background does.

[INFERRED] The 0.8667 background-only AUC is redundancy, not independent
background signal. Global self-attention has already copied tissue information
into tokens whose positions lie in background, exactly as caveat 2 of
`background_signal.md` warned. This kills the strong reading of H-a for the
downstream classifier.

Caveats [MEASURED]: mask recall is 0.89340, so about 10.7 percent of true
anatomy cells sit in the nominal background pool; residualisation is linear, so
a nonlinear background-only signal would be missed; single seed; ep50
checkpoints only, because region features were cached only at ep50.

### 2.3 Layer-0 decomposition — how much of a background token is position

Scripts `autopilot\bgsig\a3_layer0_position_content.py` and
`a3b_threshold_sweep.py`, outputs `a3_layer0_position_content.json` and
`a3b_threshold_sweep.json`.

The encoder forward is `x = patch_embed(img) + pos_embed`
(`src\models\vision_transformer.py:451`). Only the `patch_embed` Conv2d was
evaluated — no transformer blocks, no GPU. Data: 12 Test volumes x 8 slices = 96
slices, resized 256x256 and ImageNet-normalised exactly as
`src\eval_downstream.py:68` does. Cells are classed by mean raw luminance.

For each slice, the across-position variance of the layer-0 token, summed over
768 dims and taken across the selected cells, decomposes as
Var(content) + Var(position) + 2Cov.

[MEASURED] `pos_embed` unique rows: **256 of 256** (the fix holds).
Row norm 19.5959 with standard deviation 1.88e-06, i.e. constant by construction.

| checkpoint | region | position share of layer-0 across-position variance | content share |
|---|---|---:|---:|
| random_ep50 | background (lum <= 0.06) | **0.8525** | 0.1474 |
| random_ep50 | anatomy (lum >= 0.30) | 0.1520 | 0.8484 |
| random_ep100 | background | **0.9479** | 0.0521 |
| random_ep100 | anatomy | 0.3677 | 0.6334 |
| oracle_ep100 | background | **0.9455** | 0.0542 |
| oracle_ep100 | anatomy | 0.3556 | 0.6455 |

Threshold robustness, random_ep100 [MEASURED]:

| background threshold | fraction of cells | position share |
|---:|---:|---:|
| <= 0.04 | 0.015 | 0.9894 |
| <= 0.06 | 0.021 | 0.9479 |
| <= 0.08 | 0.035 | 0.9203 |
| <= 0.10 | 0.276 | 0.9082 |
| <= 0.15 | 0.722 | 0.8966 |
| <= 0.20 | 0.799 | 0.8509 |

| anatomy threshold | fraction of cells | position share |
|---:|---:|---:|
| >= 0.20 | 0.201 | 0.4080 |
| >= 0.30 | 0.064 | 0.3677 |
| >= 0.40 | 0.005 | 0.3719 |

oracle_ep100 tracks within 0.008 of random_ep100 at every threshold.

[MEASURED] The project's own convention puts anatomy at 23.35 percent of cells,
so background is about 76.7 percent. The closest threshold in the sweep is
lum <= 0.15 at 72.2 percent of cells, where position accounts for 0.897 of the
across-position variance of the encoder input; at the stricter lum <= 0.10
(27.6 percent of cells) it is 0.908. Tissue cells sit at 0.408.

[INFERRED] This is the mechanism half of H-b, and it is quantitatively strong.
At the encoder input, two background patches at different grid locations differ
almost entirely because of `pos_embed`; two tissue patches differ mostly because
of their pixels. Whatever the encoder learns to say about a background location,
it must be reading it off the position code, because there is very little else
there. This also explains section 1.6: background token diversity in the trained
encoder (cosine 0.246, effective rank 22.5) is exactly what a position-dominated
input would produce.

[MEASURED] One caveat on the raw norms: `||content||` is larger for background
(24.955) than anatomy (11.967) at random_ep100, because an ImageNet-normalised
black patch is a large constant offset ((0 - 0.485) / 0.229 = -2.118) and the
conv projects it to a large but nearly **constant** vector. The variance
decomposition above is the correct quantity precisely because it removes that
constant.

---

## 3. Verdict on H-a

**H-a as stated — "background regions carry information that helps the
downstream glaucoma classifier" — is CONTRADICTED in its strong form and
SUPPORTED for pretraining. The two need to be stated separately.**

### Background as a pretraining signal: SUPPORTED

- [MEASURED] Random masking spends 65.21 percent of its target slots on
  background (`target_composition\summary.csv`) and still reaches 0.8746 frozen
  test AUC.
- [MEASURED] The random ep100 predictor removes 67.98 percent of the background
  error left by a per-cell, no-context positional reference, which is **more**
  than the 63.34 percent it removes on anatomy targets
  (`background_signal\skill_scores.json`).
- [MEASURED] Background-token context value rises 2.4x along the random lineage
  (0.000169 at ep25 to 0.000405 at ep100).
- [MEASURED] Background representations do not collapse: cosine within
  background 0.246, effective rank 22.5 against 12.8 for anatomy; background
  self-similarity falls from 0.7842 untrained to 0.3460 at ep100.

Strongest single piece: the +0.6798 background skill score against a
position-only reference. A background target that were pure noise would score
about zero, and one that were pure position would also score about zero, because
the reference already knows the position.

### Background as a downstream contributor: CONTRADICTED in its strong form

- [MEASURED, new] Background pooled features are 95.22 percent linearly
  reconstructible from anatomy pooled features on held-out Test volumes (random
  arm; 94.2 to 97.5 percent across arms).
- [MEASURED, new] After residualising out the anatomy pool, background predicts
  glaucoma at Test AUC 0.5515 [0.5165, 0.5893] for random, 0.531 to 0.601 across
  arms.
- [MEASURED, new] Concatenating background onto anatomy changes Test AUC by
  -0.0076 [-0.0139, -0.0012] for random, and by within +/-0.002 for the other
  three arms.
- [MEASURED] Anatomy-only pooling beats all-cell pooling for all four encoders
  (`region_auc_summary.csv`), largest gain random +0.01383.

The weak reading — "background positions are not pure noise, they are used, and
they are readable" — remains SUPPORTED:

- [MEASURED, new] The random arm's per-cell attribution magnitude is negatively
  correlated with anatomy frequency (Spearman -0.4645, p = 4.2e-15), and
  near-always-background cells are 1.291x as influential as mostly-anatomy
  cells.
- [MEASURED] Background-only pooling still reaches 0.854 to 0.877.

[INFERRED] The reconciliation: background-position tokens are useful and heavily
weighted, but what they carry is a near-copy of tissue information delivered by
global self-attention, not information that originated in background pixels. The
PI's observation ("even the background matters") is correct as an observation
about attribution weight and about the pretraining task, and incorrect as a
claim that background adds downstream discriminative information.

---

## 4. Verdict on H-b

**INSUFFICIENT EVIDENCE for the causal claim. The first link is now MEASURED and
is stronger than expected; the last link is unmeasured, and one measured result
argues against it.**

H-b is a three-link chain. Status of each link:

| link | claim | status |
|---|---|---|
| 1 | Position embeddings dominate what distinguishes one background token from another | **SUPPORTED [MEASURED, new]** — 0.897 to 0.989 of layer-0 across-position variance at background cells is `pos_embed`, against 0.356 to 0.408 at tissue cells; robust across six thresholds and two checkpoints |
| 2 | The encoder therefore learns a location-specific "this is empty" representation | **SUGGESTIVE [INFERRED]** — background self-similarity falls 0.7842 untrained to 0.3460 at ep100 and effective rank is 22.5 against 12.8 for anatomy, which is what a position-dominated input would produce, but nothing rules out speckle content as the driver |
| 3 | That location information is itself diagnostic for glaucoma | **NOT SUPPORTED [MEASURED, new]** — background residualised on anatomy predicts glaucoma at 0.5515; adding background to anatomy changes Test AUC by -0.0076 to +0.0016 |

Strongest single piece for H-b: at background cells, 90.8 percent of the
across-position variance of the encoder input is contributed by the frozen
sincos position table, against 40.8 percent at tissue cells (random_ep100,
threshold 0.10). Mechanistically the PI is right about what a background token
*is*.

Reasons not to accept H-b as established:

1. [MEASURED] The predictor's background skill is measured **against a
   per-position reference** (`skill_vs_pos` 0.6798). A purely positional
   representation cannot beat a positional reference. So the pretraining
   background signal is demonstrably not just position.
2. [MEASURED, new] Link 3 fails on the only direct downstream measurement
   available. Even if the encoder does encode "location X is empty", that fact
   contributes no measurable incremental AUC.
3. [MEASURED] No `pos_embed` ablation exists. The apparent natural experiment
   (pre-posfix Test AUC 0.834 versus posfix random ep100 0.8746) is confounded by
   11 versus 100 pretraining epochs, a d=3 MLP probe versus MeanPool + Linear,
   and a different evaluation pipeline generation. It must not be cited as an
   ablation.
4. [MEASURED] The positional evidence in `error_confound_check.json`
   (corr(error, distance to context centroid) = +0.5687, the strongest single
   correlate) concerns the **pretraining predictor's error**, not the classifier.
   It supports "position drives the pretext task", which is a different and
   weaker statement than H-b.

[INFERRED] The honest current picture is: position dominates background token
identity (link 1, strong); the encoder builds a rich, non-collapsed background
representation (link 2, suggestive); and none of it shows up as incremental
downstream signal (link 3, contradicted). H-b is therefore a plausible and now
partly measured account of *what the encoder represents*, and not an account of
*why the downstream AUC is 0.8746*.

---

## 5. GPU experiments that would settle H-b, in priority order

Cost basis [MEASURED]: the campaign notes record a 68 minute (4,500 s) epoch
baseline (`docs\experiments\masking\cover_random_campaign.md:271,310`) and about
1 hour per frozen mean-pool probe
(`docs\experiments\masking\mask_composition_report.md:199`).

### P1. Inference-time `pos_embed` ablation on the frozen encoder — decisive for link 3, cheap

Take the existing `random-posfix-100ep` ep100 encoder. Extract mean-pooled
features three ways, refitting the MeanPool + Linear head each time on the same
splits:

- control: `pos_embed` as shipped (must reproduce 0.8746);
- **zeroed**: `self.pos_embed` set to zeros;
- **shuffled**: rows of `pos_embed` permuted with a fixed seed, identical norm
  and spectrum, only the position-to-code assignment destroyed.

The shuffled arm is the real control — it holds the input norm and statistics
fixed and varies only the positional *assignment*. If zeroed and shuffled both
drop AUC sharply, positional information is load-bearing at readout. If neither
moves, H-b link 3 is dead.

Cost: 3 feature extraction passes over 3600 volumes plus 3 head fits.
**Estimate 4 to 6 GPU hours.** Highest value per hour of anything on this list.

### P2. Background content substitution with positions held fixed — separates content from position

Same frozen encoder, same head, four inference-time treatments of the pixels in
cells the anatomy mask calls background:

- control: unmodified;
- constant fill at the background mean luminance (removes all background content
  variation, keeps positions);
- background patches shuffled **across volumes** at the same grid position
  (keeps background content statistics and position, destroys per-volume
  background content);
- Gaussian noise matched to per-position background mean and variance.

If AUC is unchanged under constant fill, then no information originates in
background pixels and everything the background pool carries is attention
leakage plus position. That is the clean version of the "background is
redundant" result in section 2.2.

Cost: 4 feature extraction passes plus head refits. **Estimate 5 to 8 GPU
hours.**

### P3. Position decodability from background tokens — settles link 2

Forward 500 Test slices through the frozen encoder with no masking. For each
output token, fit a linear head predicting its (row, col) grid index. Report
decoding R2 separately for background-position and tissue-position tokens, and
repeat with `pos_embed` zeroed as the control.

If position is decodable at high R2 from background tokens and collapses when
`pos_embed` is zeroed, link 2 is established: the encoder really does carry
"which empty location am I" in its background representation.

Cost: one forward pass plus ridge regressions. **Estimate 1 to 2 GPU hours.**
Cheapest item on the list.

### P4. Eroded-background pooling and attribution — the control `background_signal.md` already asked for

Erode the background mask by 2, 4 and 6 patch rows away from the retina, then
rerun both region-pooled AUC and exact patch attribution. This bounds the
10.7 percent anatomy leakage (recall 0.89340) that limits every background-only
number in this document, including section 2.2.

Cost: 3 erosion levels x 4 arms, feature extraction plus head fits.
**Estimate 8 to 12 GPU hours.**

### P5. Retrain the random arm from the ep25 fork with `pos_embed` disabled — the only fully causal test

Resume from the shared ep25 fork, 100 epochs of the random-masking arm, with
`pos_embed` and `predictor_pos_embed` zeroed (and a second run with them
shuffled per sample). Control is the existing `random-posfix-100ep` ep100 at
0.8746. Then the standard MeanPool + Linear frozen probe at ep50, ep75, ep100.

This is the only design that answers H-b end to end, because it removes position
during representation learning rather than only at readout. Note that zeroing
`predictor_pos_embed` makes every target query identical, which changes the
pretext task rather than merely ablating a feature; the shuffled variant is the
better-controlled of the two.

Cost: at 68 minutes per epoch, 75 epochs from the ep25 fork is about
**85 GPU hours per variant**, plus about 3 hours of probes. Two variants is
roughly **175 GPU hours, about 7.3 days.** Run this only if P1 and P3 come back
positive.

### Recommended order

P3, then P1 (both cheap, together under 8 GPU hours, and between them they
settle links 2 and 3). If both are negative, H-b is refuted at the downstream
level and P5 is not worth 7 days. If P1 shows a large drop under shuffled
`pos_embed`, run P2 and P4 next, and only then consider P5.

---

## 6. Provenance summary

Pre-existing artifacts used, none recomputed:
`results\masking\fair\arms_with_connectivity.json`,
`results\masking\class_relations\class_relations.json`,
`results\masking\error_vs_anatomy\error_vs_anatomy.json`,
`results\masking\error_vs_anatomy\error_confound_check.json`,
`results\masking\gate_real\elementwise_gate_probe.json`,
`results\masking\latent_probe\latent_anatomy.json`,
`results\masking\slice_pos\*.json`,
`results\masking\region_split.json`,
`results\masking\b2_probe\b2_predictor_probe.json`,
`docs\experiments\masking\background_signal.md`,
`docs\experiments\interpretability.md`,
`docs\lessons_learned.md`,
git commits `721cd26`, `5b9f20a`, `eae19c3`,
`D:\jepa_phase0\reports\background_signal\skill_scores.json`,
`D:\jepa_phase0\reports\downstream_region_auc\region_auc_summary.csv`,
`D:\jepa_phase0\reports\patch_attribution\*_attrib.json`,
`D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep100.json`,
`D:\jepa_phase0\reports\anatomy_mask_calib\mask_model_report.json`.

Fresh computation, CPU only, in `autopilot\bgsig\`:

| script | output | what is new |
|---|---|---|
| `a1_position_attribution.py` | `a1_position_attribution.json` | per-cell anatomy frequency from the 300,000-slice mask cache, crossed with the 256-cell attribution maps; attribution enrichment on near-always-background cells |
| `a2_region_incremental.py` | `a2_region_incremental.json` | incremental-information test: background residualised on anatomy, concatenation delta with paired bootstrap CIs |
| `a3_layer0_position_content.py` | `a3_layer0_position_content.json` | layer-0 position-versus-content variance decomposition at background and tissue cells |
| `a3b_threshold_sweep.py` | `a3b_threshold_sweep.json` | robustness of the above across six background and three anatomy thresholds |

Inputs those scripts read were all pre-existing:
`D:\jepa_phase0\reports\anatomy_mask_cache\Test_s100_r256.npz`,
`D:\jepa_phase0\reports\patch_attribution\*_attrib.npz`,
`D:\jepa_phase0\reports\region_features\*_ep50_s100.pt`,
`D:\jepa_phase0\checkpoints_hf\{random-posfix-100ep,oracle-anatomical-100ep}\*.pth.tar`
(only `patch_embed.proj.*` read from these),
`D:\jepa_phase0\fairvision-glaucoma\data\Test\*.npz` (12 volumes).
