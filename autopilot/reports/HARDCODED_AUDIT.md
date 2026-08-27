# Hardcoded Number Audit

[MEASURED] Verification-only audit of `paper\genai4health2026\main_submission.tex`; the paper was not edited.
[MEASURED] Audited snapshot: 1,424 lines; SHA-256 `99891543105105D4151682BB1A92C3161AD105CB8D6E6B16DF03DF49D902910D`.

## Result

- [MEASURED] Hardcoded numeric occurrences in scope: **310**
- [MEASURED] WRONG: **1**
- [MEASURED] UNBACKED: **75**
- [MEASURED] CONFIRMED: **234**

[MEASURED] Counting is by rendered occurrence, not distinct value. A row containing five displayed values contributes five to the count.

## Scope and method

- [MEASURED] Included every rendered Arabic numeric quantity typed directly in the TeX: results, percentages, sample sizes, epochs, seeds, hyperparameters, thresholds, literature values, and numeric table/figure text.
- [MEASURED] Excluded macro expansions, citation-key years, filenames, LaTeX command indices, pure layout values (widths, column spans, `tabcolsep`, table column specifications), and non-quantity identifiers such as `H1`, `v2`, `3D`, `fp16`, `fp32`, and `SHA-256`.
- [MEASURED] Table 2 was searched by exact row sequences and individual values throughout the repository, the research inventory, Git history, `results\masking`, `autopilot`, and read-only `D:\jepa_phase0\reports`.
- [MEASURED] The explicitly warned-about `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json` was treated only as a non-source comparison. It does not back Table 2.
- [INFERRED] CONFIRMED means a stored artifact contains the exact value or enough exact inputs to reproduce the displayed rounding. WRONG means the stored source/config gives a different value in the same context. UNBACKED means no producing machine-readable artifact was found.

## Table 2 verdict

[PENDING] No artifact backs Table 2 as a common 600-slice, five-policy production-sampler measurement. Git history shows the block first entered as hardcoded prose/table text in commit `df1bd6d`; it did not add a geometry JSON/CSV. The only 600-slice JSON found is an anatomy-only K-sweep. Consequently 24 of the 25 hardcoded geometry cells are UNBACKED. The sole independently backed cell is anatomy-v2 loss slots `64.0`, fixed by `npred=4` and `pred_target_k=16`.

## WRONG

| line | the number | context | class | artifact path | stored value |
|---:|---|---|---|---|---|
| 1060 | 7.17M | [MEASURED] Fine-tuned attentive-probe parameter count. | WRONG | C:\Users\Gary\Desktop\jepa\results\downstream\finetune_oracle\d1_results.json; C:\Users\Gary\Desktop\jepa\src\eval_downstream.py | The fine-tuned config stores num_slices=64, probe_depth=1, probe_num_heads=12. Instantiating that stored architecture gives 7,140,096 parameters, i.e. 7.14M. 7.17M is the 100-slice probe (7,167,744), not the 64-slice fine-tuned probe. |

## UNBACKED

| line | the number | context | class | artifact path | stored value |
|---:|---|---|---|---|---|
| 56 | 97% | [PENDING] Abstract purity claim propagated from Table 2. | UNBACKED | No producing artifact found; nearest different sweep: D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json | Different sweep stores anatomy-v2 pct_tgt_anat=97.5011057324625 on n=1,534, not a 600-slice 97% source. |
| 69 | 600 | [PENDING] Claimed sample size for the five-policy geometry measurement. | UNBACKED | No producing artifact found. The only JSON under D:\jepa_phase0\reports with slices=600 is D:\jepa_phase0\reports\budget_masks\budget_mask_audit_fairvision.json | The 600-slice artifact is a K-sweep for anatomy masks only; it does not contain the five Table 2 policies or Table 2's rows. |
| 131 | 600 | [PENDING] Claimed sample size for the five-policy geometry measurement. | UNBACKED | No producing artifact found. The only JSON under D:\jepa_phase0\reports with slices=600 is D:\jepa_phase0\reports\budget_masks\budget_mask_audit_fairvision.json | The 600-slice artifact is a K-sweep for anatomy masks only; it does not contain the five Table 2 policies or Table 2's rows. |
| 388 | 43.5% | [PENDING] Envelope purity claim propagated from Table 2. | UNBACKED | No producing artifact found; nearest different sweep: D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json | Different sweep stores envelope pct_tgt_anat=43.192692001977136 on n=6,137. |
| 465 | 600 | [PENDING] Claimed sample size for the five-policy geometry measurement. | UNBACKED | No producing artifact found. The only JSON under D:\jepa_phase0\reports with slices=600 is D:\jepa_phase0\reports\budget_masks\budget_mask_audit_fairvision.json | The 600-slice artifact is a K-sweep for anatomy masks only; it does not contain the five Table 2 policies or Table 2's rows. |
| 476 | 52.2%; 31.6%; 43.7%; 42.1%; 157.7 | [PENDING] Table 2 random geometry cells. | UNBACKED | No producing artifact found; explicitly non-source near-match: D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json | No stored 600-slice five-policy row. pct_anat_hid=53.04446982807447; pct_tgt_anat=31.583217099488156 in the different 6,137-slice sweep. |
| 477 | 62.2%; 41.1%; 40.0%; 45.6%; 158.4 | [PENDING] Table 2 centroid geometry cells. | UNBACKED | No producing artifact found; explicitly non-source near-match: D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json | No stored 600-slice five-policy row. pct_anat_hid=61.58470785250753; pct_tgt_anat=39.68775691115843 in the different 6,137-slice sweep. |
| 478 | 76.9%; 43.5%; 46.4%; 40.5%; 159.9 | [PENDING] Table 2 envelope geometry cells. | UNBACKED | No producing artifact found; explicitly non-source near-match: D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json | No stored 600-slice five-policy row. pct_anat_hid=77.58224558657956; pct_tgt_anat=43.192692001977136 in the different 6,137-slice sweep. |
| 479 | 74.1%; 45.3%; 43.3%; 43.5%; 160.0 | [PENDING] Table 2 cover f=0.21 geometry cells. | UNBACKED | No producing artifact found; explicitly non-source near-match: D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json | No stored 600-slice five-policy row. pct_anat_hid=73.0945185679789; pct_tgt_anat=40.877618298930145 in the different 6,137-slice sweep. |
| 480 | 80.3%; 97.3%; 21.4%; 67.9% | [PENDING] Table 2 anatomy-v2 (all cells except independently backed loss slots) geometry cells. | UNBACKED | No producing artifact found; explicitly non-source near-match: D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json | No stored 600-slice five-policy row. pct_anat_hid=82.0662494138952; pct_tgt_anat=97.5011057324625 in the different n=1,534 sweep. |
| 485 | 600 | [PENDING] Claimed sample size for the five-policy geometry measurement. | UNBACKED | No producing artifact found. The only JSON under D:\jepa_phase0\reports with slices=600 is D:\jepa_phase0\reports\budget_masks\budget_mask_audit_fairvision.json | The 600-slice artifact is a K-sweep for anatomy masks only; it does not contain the five Table 2 policies or Table 2's rows. |
| 496--497 | 97.3%; 41.1% | [PENDING] Purities repeated from Table 2 in the direct-comparison prose. | UNBACKED | No producing artifact found; nearest different sweep: D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json | Different sweep: anatomy-v2=97.5011057324625; centroid=39.68775691115843. |
| 502--505 | 21.4%; 40%; 46%; 67.9%; 40%; 46%; 158; 160 | [PENDING] Geometry ranges repeated from Table 2. | UNBACKED | No producing artifact found; nearest different sweep: D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json | No artifact stores these Table 2 ranges for a common 600-slice run. |
| 519 | 0.400; 0.316; 0.338 | [PENDING] Claimed per-cell hit-map Gini values. | UNBACKED | No producing artifact found. A different diagnostic exists at C:\Users\Gary\Desktop\jepa\results\masking\coverage\coverage.json | Different diagnostic stores random_default=0.4111302367507499, random_matched=0.28943627281526546, envelope_default=0.40376528615375246, and no centroid value. |
| 656 | 0.0003; 0.0018 | [PENDING] Five probe-seed standard deviations. | UNBACKED | C:\Users\Gary\Desktop\jepa\paper\genai4health2026\research\verify_sections_4_6.md | The prior audit found only seed-42 result files; the other four per-seed outputs needed to regenerate either SD were not found. |
| 915--916 | 78.6%; 73.9% | [PENDING] COVER pre/post-truncation fresh CPU audit. | UNBACKED | No raw output found. Narrative only: C:\Users\Gary\Desktop\jepa\autopilot\COVER_AUDIT.md | Narrative says 78.62% and 73.88% over 194 accepted slices but names no persisted producing artifact. |
| 919 | 32.5%; 2.51 | [PENDING] COVER retained-target frequency and mean. | UNBACKED | No raw output found. Narrative only: C:\Users\Gary\Desktop\jepa\autopilot\COVER_AUDIT.md | Narrative says 32.47% and 2.51 but names no persisted producing artifact. |
| 995 | 0.8811 | [PENDING] Superseded full-batch label-efficiency null AUC. | UNBACKED | No result artifact found; literal appears only in C:\Users\Gary\Desktop\jepa\autopilot\p5_label_efficiency.py | No stored output for the superseded fit. |
| 1061 | 0.94 | [PENDING] Interpretability slice-level cross-probe correlation. | UNBACKED | No raw local source found. C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md states the NPZ inputs live on an external blob/local archive that is absent from this checkout. | No machine-readable local stored value. |
| 1074 | 25 | [PENDING] Interpretability window/single-slice amplification. | UNBACKED | No raw local source found. C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md states the NPZ inputs live on an external blob/local archive that is absent from this checkout. | No machine-readable local stored value. |
| 1079--1080 | 63; 137; 95 | [PENDING] Interpretability population peak/dip positions. | UNBACKED | No raw local source found. C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md states the NPZ inputs live on an external blob/local archive that is absent from this checkout. | No machine-readable local stored value. |
| 1086 | -0.22; -0.07; -0.14 | [PENDING] Interpretability per-volume peak correlations. | UNBACKED | No raw local source found. C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md states the NPZ inputs live on an external blob/local archive that is absent from this checkout. | No machine-readable local stored value. |
| 1089 | 0.971; 0.988 | [PENDING] Interpretability mirror correlations. | UNBACKED | No raw local source found. C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md states the NPZ inputs live on an external blob/local archive that is absent from this checkout. | No machine-readable local stored value. |
| 1090 | -0.124; -0.478 | [PENDING] Interpretability raw cluster correlations. | UNBACKED | No raw local source found. C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md states the NPZ inputs live on an external blob/local archive that is absent from this checkout. | No machine-readable local stored value. |
| 1112 | 0.25 | [PENDING] Interpretability confidence-attribution correlation bound. | UNBACKED | No raw local source found. C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md states the NPZ inputs live on an external blob/local archive that is absent from this checkout. | No machine-readable local stored value. |
| 1131 | 84%; 91%; 95% | [PENDING] Interpretability patch-significance range and CI level. | UNBACKED | No raw local source found. C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md states the NPZ inputs live on an external blob/local archive that is absent from this checkout. | No machine-readable local stored value. |
| 1134 | 0.94 | [PENDING] Interpretability repeated slice-level correlation. | UNBACKED | No raw local source found. C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md states the NPZ inputs live on an external blob/local archive that is absent from this checkout. | No machine-readable local stored value. |
| 1135 | 0.35; 0.48 | [PENDING] Interpretability patch-level correlation range. | UNBACKED | No raw local source found. C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md states the NPZ inputs live on an external blob/local archive that is absent from this checkout. | No machine-readable local stored value. |
| 1343--1344 | 97%; 41%; 44% | [PENDING] Figure-caption purity summary propagated from Table 2. | UNBACKED | No producing artifact found; nearest different sweep: D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json | Different sweep purities are random=31.583217099488156, centroid=39.68775691115843, envelope=43.192692001977136, anatomy-v2=97.5011057324625. |
| 1362 | 600 | [PENDING] Claimed sample size for the five-policy geometry measurement. | UNBACKED | No producing artifact found. The only JSON under D:\jepa_phase0\reports with slices=600 is D:\jepa_phase0\reports\budget_masks\budget_mask_audit_fairvision.json | The 600-slice artifact is a K-sweep for anatomy masks only; it does not contain the five Table 2 policies or Table 2's rows. |
| 1365 | 1.6; 0.4 | [PENDING] Context and loss-slot ratios derived from unbacked Table 2 cells. | UNBACKED | No producing artifact found. | The arithmetic is compatible with the printed cells, but the input cells have no producing artifact. |

## CONFIRMED

| line | the number | context | class | artifact path | stored value |
|---:|---|---|---|---|---|
| 52, 450, 1319 | 95%; 95%; 95% | [MEASURED] Confidence-interval level. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p1c_stats.json | All interval fields are ci95/boot_ci95; n_bootstrap=10000. |
| 54, 58--64, 115--119 | 100; 50; 75; 50; 100; 50; 75; 30 | [MEASURED] Epochs in the abstract/introduction. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json | Primary records store the cited arms at epochs 30, 50, 75, and 100. |
| 136 | 10^-5 | [INFERRED] Order of the independent reproduction difference. | CONFIRMED | Recomputed from the inputs at lines 617--618; see those rows. | abs(0.8854753820 - 0.8854851648)=9.7828e-6. |
| 167 | 75; 73.5; 75; 63.9 | [MEASURED] MAE random/block masking ablation quoted in prose. | CONFIRMED | C:\Users\Gary\Desktop\jepa\autopilot\reports\P2-02_related_work.md | Literature extraction stores random-75=73.5 linear and block-75=63.9 linear. |
| 169 | 0.2 | [INFERRED] Maximum AttMask DINO-only margin over random. | CONFIRMED | C:\Users\Gary\Desktop\jepa\autopilot\reports\P2-02_related_work.md | Stored extraction: random=43.4, high=43.5, hint=43.6; max gap=0.2. |
| 217 | 16; 16; 256 | [MEASURED] Token-grid dimensions and cell count. | CONFIRMED | D:\jepa_phase0\reports\mask_stats_fairvision.json | Artifact metadata stores grid=16x16 and total_patches=256; the count is 16*16=256. |
| 222, 511, 920 | 4; 4; 16; 64; 4 | [MEASURED/INFERRED] Four targets and 4x16=64 anatomy loss slots. | CONFIRMED | C:\Users\Gary\Desktop\jepa\configs\patch_cover_f021_ep25.yaml; C:\Users\Gary\Desktop\jepa\configs\patch_anatomy_v2.yaml; D:\jepa_phase0\reports\budget_masks\budget_mask_audit_fairvision.json | Config stores npred=4/pred_target_k=16; 600-slice K=16 artifact stores target_union=64. Exact product 4*16=64. |
| 247, 311, 373, 667 | 0.21; 0.21; .21; 0.21 | [MEASURED] COVER visible-anatomy floor. | CONFIRMED | C:\Users\Gary\Desktop\jepa\configs\patch_cover_f021_ep25.yaml | cover_leave_frac=0.21 and cover_min_visible_frac=0.21. |
| 283, 293 | 100; 256; 256; 256; 256 | [MEASURED] OCT resampling and crop dimensions. | CONFIRMED | C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_random\ep100_results.json | config.data.num_slices=100; config model crop_size/slice_size=256. |
| 286 | 3000; 1466; 1534; 42 | [MEASURED] Test size, class counts, and evaluation seed. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p1c_stats.json; C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_random\ep100_results.json | n_test=3000; n_pos=1466; n_neg=1534; config.training.seed=42. |
| 294 | 16; 6 | [MEASURED] Patch size and predictor depth. | CONFIRMED | C:\Users\Gary\Desktop\jepa\configs\patch_oracle_anatomical.yaml | patch_size=16; pred_depth=6. |
| 295 | 25 | [MEASURED] Shared ancestor epoch. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json | ancestor primary record epoch=25; all arm descriptions trace to this fork. |
| 296 | 512 | [INFERRED] Effective pretraining batch size. | CONFIRMED | C:\Users\Gary\Desktop\jepa\configs\patch_mirage_envelope.yaml | batch_size=64 and accum_steps=8; 64*8=512. |
| 297 | 25; 30 | [MEASURED] Guidance ramp endpoints. | CONFIRMED | C:\Users\Gary\Desktop\jepa\configs\patch_oracle_anatomical.yaml | curriculum_cfg stores T_warm=25 and T_total=30. |
| 304--312 | 100; 75; 75; 92; 75; 100; 30; 100; 73; 50 | [MEASURED] Arm horizons, exclusions, COVER peak, and shared comparison epoch. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json | Primary/excluded records store exactly these arm/epoch combinations; COVER records include ep73 and ep100. |
| 353, 364, 386--387, 397, 409, 412, 423, 429, 433, 437--438, 449, 453, 455--456, 468, 474, 498, 559, 565, 590, 614, 628, 669, 672, 693, 767, 812, 835, 868, 896, 903, 957, 962, 1163, 1225, 1234, 1342, 1354 | 25; 50; 75; 100; 50; 100; 25; 50; 30; 50; 100; 50; 50; 75; 100; 100; 100; 50; 75; 50; 50; 100; 25; 100; 100; 100; 100; 100; 50; 50; 75; 100; 100; 100; 100; 75; 92; 100; 100; 100; 100; 100; 100; 50; 50 | [MEASURED] Repeated checkpoint/analysis epoch literals. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json; D:\jepa_phase0\autopilot_out\p1_stats\p1c_stats.json | Inventory and statistical table store the cited epochs (25, 30, 50, 73, 75, 92 excluded, and 100) in their stated contexts. |
| 358, 609--610, 1384 | 2x10^-4; 2x10^-4; 2x10^-4 | [MEASURED] Maximum fp16/fp32 re-probe shift. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p3b_fp32.json | Largest abs(delta_fp32_minus_fp16)=0.00019209869604108754, which rounds upward to 2e-4. |
| 443--444, 732--733, 917 | 73.1%; 77.6%; 73.1%; 77.6%; 73.1%; 77.6% | [MEASURED] Delivered COVER versus envelope anatomy hidden. | CONFIRMED | D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json | f=0.21 pct_anat_hid=73.0945185679789; envelope pct_anat_hid=77.58224558657956. |
| 480, 504, 939 | 64.0; 64; 16 | [MEASURED/INFERRED] Anatomy-v2 fixed target size and loss slots. | CONFIRMED | C:\Users\Gary\Desktop\jepa\configs\patch_anatomy_v2.yaml; D:\jepa_phase0\reports\budget_masks\budget_mask_audit_fairvision.json | pred_target_k=16, npred=4; K=16 600-slice artifact stores target_union=64. |
| 491--494 | +0.80; 0.20; +0.40; 0.60; +0.50; +0.20; 4; 5 | [INFERRED] Corrected Spearman coefficients, p-values, and arm counts. | CONFIRMED | C:\Users\Gary\Desktop\jepa\autopilot\reports\SPEARMAN_CORRECTION.md | Recomputed inputs: four rectangles hidden=[52.2,62.2,76.9,74.1], purity=[31.6,41.1,43.5,45.3], AUC=[0.8641,0.8740,0.8761,0.8643]; all-five adds hidden=80.3, purity=97.3, AUC=0.8654. scipy.stats.spearmanr returns hidden rho=+0.80,p=0.20; purity rho=+0.40,p=0.60; all-five rho=+0.50 and +0.20. |
| 531, 997 | 0.0003; 0.0003 | [INFERRED] Matched label-efficiency/full-protocol agreement bound. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p5_label_efficiency.json; D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json | Full-data label-efficiency AUCs random=0.8748054556029675 and centroid=0.8856443577233458; primary AUCs are 0.8745808957846787 and 0.88548516482246. Both absolute differences are <0.0003. |
| 533, 536, 1004, 1006, 1012 | 5%; 1%; 5%; 1%; 1% | [MEASURED] Label-efficiency fractions. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p5_label_efficiency.json | fractions=[0.01,0.05,0.1,0.25,1.0]. |
| 618--619 | 9.8x10^-6; 0.8854754; 0.8854852; 22; 2,248,844 | [MEASURED/INFERRED] Independent fixed-head reproduction. | CONFIRMED | D:\jepa_phase0\runs\frozen_meanpool_oracle_ep100_fp32\feature_cache\Test_s100_r256_fp32_52d1a1812356.pt; D:\jepa_phase0\checkpoints_hf\downstream-heads\frozen-meanpool\oracle-ep100-head.pt; C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_oracle\ep100_test_predictions.npz | Fresh recomputation: AUC=0.8854753820, reference=0.8854851648, delta=-0.0000097828. With 1466*1534=2,248,844 pairs, the AUC delta is exactly 22 discordant pairs. |
| 650 | 1 | [MEASURED] One pretraining continuation per policy. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json | Inventory has one continuation lineage for each policy; checkpoint rows within a lineage are not independent retrainings. |
| 714, 801 | 42; 0.4821 | [MEASURED] All-probes seed and branch-level race trend p-value. | CONFIRMED | C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_random\ep100_results.json; D:\jepa_phase0\autopilot_out\p1_stats\p7b_gap_trend.json | seed=42; race.branch_spearman_p=0.4820720382996778. |
| 730--731, 916 | 6,137; 1,534; 6,137 | [MEASURED] Large sampler-sweep sample sizes. | CONFIRMED | D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json; D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json | Rectangle/floor rows store n=6137; anatomy-v2 row states src=arm_stats(n=1534). |
| 827, 844--846 | -2; -12; 334; -12; -6; 460; -6; -2; 672 | [MEASURED] Severity thresholds and positive counts. | CONFIRMED | D:\jepa_phase0\autopilot_out\subgroup\subgroup_auc.json | Every run stores severity levels severe (<=-12), n=334; moderate (-12,-6], n=460; mild (-6,-2], n=672, each against 1534 negatives. |
| 920 | 24,000; 73.4% | [MEASURED] Emitted-target rectangle audit. | CONFIRMED | D:\jepa_phase0\reports\cover_random_scale\scale_validation.json | blocks_checked=24000; perfect_rectangles_pct=73.4. |
| 958--959 | 0.680; 0.633 | [MEASURED] Random ep100 predictor skill versus position-only reference. | CONFIRMED | D:\jepa_phase0\reports\background_signal\skill_scores.json | random_ep100 bg.skill_vs_pos=0.6798094511032104; anat.skill_vs_pos=0.6334153413772583. |
| 961--962 | 0.784; 0.346 | [MEASURED] Background self-similarity before and after pretraining. | CONFIRMED | C:\Users\Gary\Desktop\jepa\results\masking\class_relations\class_relations.json | JEPA untrained bg_bg=0.7842189359236829; stored ep100 comparison is JEPA envelope bg_bg=0.3460093365644443. The artifact does not label the latter as random. |
| 965--969 | 95.2%; 0.5515; 0.5165; 0.5893; 0.0076; -0.0139; -0.0012; 0.002; 0.867 | [MEASURED/INFERRED] Background regional-probe and incremental-value results. | CONFIRMED | C:\Users\Gary\Desktop\jepa\autopilot\bgsig\a2_region_incremental.json | random ridge R2=0.9522224497054176; residual AUC=0.5515071507150715, CI=[0.5164548313700296,0.5893034399036036]; cat-minus-anatomy mean=-0.007608015169676156, CI=[-0.013850635858963197,-0.0011915449892519023]; other-arm absolute means <=0.0016082824831453722; random background-only AUC=0.8666506650665066. |
| 974--975 | 0; 90.8%; 40.8% | [MEASURED] Layer-0 position-variance shares. | CONFIRMED | C:\Users\Gary\Desktop\jepa\autopilot\bgsig\a3b_threshold_sweep.json | At threshold <=0.10 random_ep100 background position_share=0.9082097946887898; at >=0.20 anatomy position_share=0.40799012607939. |
| 992--993 | 256; 4x10^-4; 0.05; 50; 5 | [MEASURED] Matched label-efficiency probe protocol. | CONFIRMED | C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_random\ep100_results.json | config stores batch_size=256, lr_probe=0.0004, weight_decay=0.05, epochs=50, warmup_epochs=5. |
| 1032 | 0.027; 5%; 0.085 | [INFERRED] Across-arm AUC spreads at full data and 5% labels. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p5_label_efficiency.json | Full spread=0.8856443577233458-0.858561554291894=0.0270828034314518; 5% spread=0.8335294489079722-0.748360935662945=0.0851685132450272. |
| 1040--1041 | 1; 0.887 | [MEASURED] Fine-tuned probe depth and approximate random-family AUC. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json | Random fine-tune AUCs are 0.8867558176556489, 0.8871778122448689, 0.88776344646405; attentive config stores probe_depth=1. |
| 1053 | 256 | [MEASURED] Patch tokens per target slice. | CONFIRMED | C:\Users\Gary\Desktop\jepa\scripts\deeper_interpretability_analysis.py | Stored analysis code reads patch_contrib with shape (3000,256). |
| 1071, 1113, 1132 | 7; 0.5; 500 | [MEASURED] Interpretability procedure inputs (window, threshold, bootstrap draws). | CONFIRMED | C:\Users\Gary\Desktop\jepa\scripts\deeper_interpretability_analysis.py | Code reads window W7, sets pred=(probs>=0.5), and sets B=500. |
| 1166, 1174, 1179, 1183 | 15; 0.90; 0.90; 0.85 | [MEASURED] ECE bins and operating-point targets. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p8b_operating_points.json | ece_15bin fields and target_specificities=[0.85,0.9]. |
| 1189 | +0.011 | [INFERRED] Rounded epoch-100 centroid-minus-random AUC difference. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p1c_stats.json | Exact primary contrast=0.0109042690377812 (rounded to +0.011). |
| 1288--1296 | 75; 84.9; 73.5; 75; 82.8; 63.9; 75; 84.0; 66.0; 83.0; 1; 82.7; 82.6; 66.8; 66.5; 52.9; 68.7; 63.7; 63.6; 65.0; 83.26; 83.32; 82.49; 82.95; 81.40; 43.4; 43.5; 43.6; 78.5; 10%; 78.1; 79.48; 25%; 77.53; 50%; 79.97 | [MEASURED] Published masking ablations and their mask/label percentages. | CONFIRMED | C:\Users\Gary\Desktop\jepa\autopilot\reports\P2-02_related_work.md | Local literature extraction records every displayed value verbatim in its comparison table and study notes (MAE, SimMIM, SemMAE, AutoMAE, HPM, AttMask, AttG-MAE, SSiT). |
| 1321--1331 | 50; 75; 100; 50; 75; 100; 50; 75; 100 | [MEASURED] Full paired-contrast table epochs. | CONFIRMED | D:\jepa_phase0\autopilot_out\p1_stats\p1c_stats.json | Confirmatory family stores three contrasts at each of epochs 50, 75, 100. |
| 1414 | 0.894668 | [MEASURED] Hardcoded fine-tuned test AUC. | CONFIRMED | C:\Users\Gary\Desktop\jepa\results\downstream\finetune_oracle\meanpool_results.json | test_auc=0.8946676603623906; rounds to the displayed six decimals. |
| 1415 | 0.893717 | [MEASURED] Hardcoded fine-tuned test AUC. | CONFIRMED | C:\Users\Gary\Desktop\jepa\results\downstream\finetune_oracle\crossattn_results.json | test_auc=0.8937171720226036; rounds to the displayed six decimals. |
| 1416 | 0.890100 | [MEASURED] Hardcoded fine-tuned test AUC. | CONFIRMED | C:\Users\Gary\Desktop\jepa\results\downstream\finetune_oracle\d1_results.json | test_auc=0.8900997579200691; rounds to the displayed six decimals. |
| 1417 | 0.887763 | [MEASURED] Hardcoded fine-tuned test AUC. | CONFIRMED | C:\Users\Gary\Desktop\jepa\results\downstream\finetune_random\attentive_results.json | test_auc=0.88776344646405; rounds to the displayed six decimals. |
| 1418 | 0.887178 | [MEASURED] Hardcoded fine-tuned test AUC. | CONFIRMED | C:\Users\Gary\Desktop\jepa\results\downstream\finetune_random\cross_attn_pool_results.json | test_auc=0.8871778122448689; rounds to the displayed six decimals. |
| 1419 | 0.886756 | [MEASURED] Hardcoded fine-tuned test AUC. | CONFIRMED | C:\Users\Gary\Desktop\jepa\results\downstream\finetune_random\mean_pool_results.json | test_auc=0.8867558176556489; rounds to the displayed six decimals. |

## Recomputations

### Geometry rank correlations

[INFERRED] Inputs used exactly as printed in Table 2:

- rectangle anatomy hidden: `[52.2, 62.2, 76.9, 74.1]`
- rectangle purity: `[31.6, 41.1, 43.5, 45.3]`
- rectangle epoch-50 AUC: `[0.8641, 0.8740, 0.8761, 0.8643]`
- all-five extension: hidden `80.3`, purity `97.3`, AUC `0.8654`

[MEASURED] `scipy.stats.spearmanr` returns rectangle hidden `rho=+0.80, p=0.20`, rectangle purity `rho=+0.40, p=0.60`, and all-five coefficients `+0.50` and `+0.20`. This confirms the corrected arithmetic, not the provenance of Table 2's input cells.

### Fixed-head reproduction

[MEASURED] Re-ran `scripts\score_head_on_cache.py` with:

- features: `D:\jepa_phase0\runs\frozen_meanpool_oracle_ep100_fp32\feature_cache\Test_s100_r256_fp32_52d1a1812356.pt`
- head: `D:\jepa_phase0\checkpoints_hf\downstream-heads\frozen-meanpool\oracle-ep100-head.pt`
- reference predictions: `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_oracle\ep100_test_predictions.npz`

[MEASURED] Output: `0.8854753820` versus `0.8854851648`, `delta=-0.0000097828`; `1466*1534=2,248,844`, so the AUC difference is exactly 22 discordant positive-negative pairs.

## Exclusions audit

[MEASURED] Excluded layout examples: lines 17, 361, 364--365, 750, 1127, 1284, 1377, and figure width/height literals. Excluded identifier examples: `anatomy-v2`, `H3`, `fp32`, and `SHA-256`. Citation years occur only inside non-rendered citation keys and were excluded.
