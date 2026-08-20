# Verification of Sections 3, 7, and 8

## Summary verdict (counts)

**68 claim rows checked: 49 CONFIRMED, 10 WRONG, 7 MISLEADING, 2 UNVERIFIABLE.**

The dataset identity, split, array shape, class counts, license, encoder setup, masking-policy descriptions, composition metrics, COVER trajectory AUCs, regional AUCs, and long-horizon AUCs are artifact-backed. There is no Harvard-GF conflation: the data are the 10,000-subject **Harvard-FairVision glaucoma** cohort, not the 3,300-patient Harvard-GF dataset.

The largest problems are in Section 7. Six valid random/oracle prediction files were overlooked. After de-duplicating the three byte-identical MIRAGE copies and excluding the four retracted COVER probes, the available evidence contains **18 unique valid probes from six pretraining trajectories**, not 12 from five. Across all 18, Black patients have the lowest point-estimate AUC in 18/18; the racial-gap range remains 0.0475--0.0935 (1.97x), but the overall-AUC range is 0.0371, the nominal Spearman result is rho=0.4881, p=0.03986, and the severity-gap range is 0.1286--0.1394 (spread 0.0108). The checkpoint observations are non-independent, so the nominal Spearman p-value must not be used to claim a tradeoff.

The statement that no individual racial gap is significant is also false. For blob epoch 50, the Asian-minus-Black AUC difference is 0.09354; independent-sample DeLong 95% CI [0.04027, 0.14680], p=0.000577. It remains significant after Holm correction over 18 probes (adjusted p=0.0104) and even Bonferroni correction over all 54 probe-by-race-pair tests (p=0.0312).

## Table of every claim checked

| # | Claim quoted verbatim from `main.tex` | Raw artifact / field checked | Independently computed value | Verdict |
|---|---|---|---|---|
| S3-01 | L114: “We use the released FairVision glaucoma cohort.” | Official FairVision README (`Harvard-Ophthalmology-AI-Lab/FairVision`), Dataset section; local `D:\jepa_phase0\fairvision-glaucoma\metadata\data_summary_glaucoma.csv` | Official name is Harvard-FairVision; local glaucoma metadata contain 10,000 unique subject files. No Harvard-GF reference or 3,300-patient count occurs here. | **CONFIRMED** |
| S3-02 | L114--115: “Its split contains 6,000 training, 1,000 validation, and 3,000 test OCT volumes.” | Metadata CSV fields `filename`, `use`; filesystem counts under `data\Training`, `data\Validation`, `data\Test` | `use`: training=6000, validation=1000, test=3000; filesystem NPZ counts match exactly. | **CONFIRMED** |
| S3-03 | L115--117: “Each `oct_bscans` array is a \(200\times200\times200\) uint8 B-scan stack.” | NPY headers for `oct_bscans.npy` inside all 10,000 NPZ files | Shape `(200,200,200)` and dtype `uint8` in 10,000/10,000 files. This is one 3-D OCT volume made from a stack of B-scans, not “one 3-D B-scan.” | **CONFIRMED** |
| S3-04 | L117--118: “Evaluation samples 100 of the 200 B-scans per volume.” | `src\datasets\oct_volumes.py`, `slice_indices`; standard result JSONs, `config.data.num_slices` | Loader uses `np.linspace(0,199,num=100)`; all standard scoped probes use `num_slices=100`. | **CONFIRMED** |
| S3-05 | L118: “The test set has 1,466 positive and 1,534 negative volumes.” | Test metadata `glaucoma`; all test NPZ `glaucoma` fields | yes/1=1466, no/0=1534. | **CONFIRMED** |
| S3-06 | L119--120: “The data license is CC BY-NC-ND 4.0 and permits non-commercial research, not clinical use.” | Official FairVision README, Dataset paragraph and license link | README states non-commercial research only, no clinical decisions/patient care, CC BY-NC-ND 4.0. | **CONFIRMED** |
| S3-07 | L123--124: “We use patch-level I-JEPA with a ViT-B/16 encoder.” | `D:\jepa_phase0\fairvision-glaucoma\checkpoint-ep25\README.md`; result configs `encoder_name`, `patch_size` | Architecture is I-JEPA ViT-B/16, patch size 16, embedding dimension 768. | **CONFIRMED** |
| S3-08 | L124: “Slices are processed on a \(16\times16\) patch grid.” | Result configs `crop_size=256`, `patch_size=16`; `src\datasets\oct_volumes.py` | \(256/16=16\) patches per axis, 256 patch tokens. | **CONFIRMED** |
| S3-09 | L125--126: “Unless stated otherwise, evaluation freezes the encoder, mean-pools patch features, and fits a linear glaucoma head with probe seed 42.” | Standard result JSON fields `freeze_encoder`, `probe_type`, `head_type`, `training.seed`, `probe_params`, `head_params` | `true`, `mean_pool`, `linear`, seed 42, zero probe parameters, 2305 head parameters. | **CONFIRMED** |
| S3-10 | L126--127: “Every masking arm descends from the same epoch-25 checkpoint” | `checkpoint-ep25\README.md`; oracle/envelope/anatomy/COVER configs `read_checkpoint`; checkpoint paths in probe JSONs | Random fork is the shared epoch-25 checkpoint; oracle, envelope, anatomy/blob, and clean COVER descend directly or transitively from it. | **CONFIRMED** |
| S3-11 | L127: “whose test AUC is 0.8487.” | `D:\jepa_phase0\runs\frozen_meanpool_fork_ep25\test_predictions.npz`, keys `labels`, `probs` | Rank-based AUC=0.848680032941369, rounding to 0.8487. | **CONFIRMED** |
| S3-12 | L127--129: “This shared ancestor improves comparability but does not replace pretraining-seed replication.” | Shared checkpoint SHA/provenance; all relevant pretraining configs use one trajectory/seed per policy | Common initialization controls one source of variation; there is no pretraining-seed replication. | **CONFIRMED** |
| S3-13 | L132: “Random uses standard multiblock rectangles” | `src\masks\multiblock.py`; `scripts\arm_stats_table.py`, `random (stock JEPA)` | Stock I-JEPA target masks are axis-aligned multiblock rectangles. | **CONFIRMED** |
| S3-14 | L132--133: “oracle places rectangles using a hand-crafted retinal band” | `configs\patch_oracle_anatomical.yaml`; `src\masks\curriculum.py`, `anatomical_prior` | Fixed-size rectangles are positioned using an intensity-derived retinal band. | **CONFIRMED** |
| S3-15 | L133--134: “envelope places rectangles using frozen MIRAGE segmentation” | `configs\patch_mirage_envelope.yaml`, `mirage_guide_dir`; `src\masks\curriculum.py`, `mirage_envelope` | Precomputed MIRAGE guides direct rectangular placement; the guide model is not trained with I-JEPA. | **CONFIRMED** |
| S3-16 | L134--135: “COVER greedily places rectangles while enforcing an anatomy-visibility floor” | `src\masks\cover.py`, `build_targets`; clean config `cover_min_visible_frac`, `cover_min_visible_cells` | Exhaustive greedy rectangle placement enforces mass and cell visibility floors. | **CONFIRMED** |
| S3-17 | L135--136: “blob uses connected, near-pure anatomy targets.” | `src\masks\anatomy.py`; `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json`, blob `pct_tgt_anat` | Targets are connected by construction; measured target purity=97.5011%. | **CONFIRMED** |
| S3-18 | L136--138: “For each slice we measure anatomy recall by targets, target purity, anatomy retained in delivered context, target/context budgets, and the zero-anatomy context rate.” | `composition_vs_auc_ep50.json` fields `pct_anat_hid`, `pct_tgt_anat`, `ctx_anat`, `pct_ctx_anat`, `tgt`, `ctx`, `zero_pct` | All listed quantities are present in the raw report outputs. | **CONFIRMED** |
| S7-01 | L321: “Saved test predictions contain no identifiers” | All scoped `test_predictions.npz` files | Prediction NPZ keys are only `labels` and `probs`; no filenames or subject IDs. | **CONFIRMED** |
| S7-02 | L321--322: “we join them to demographics in deterministic test-loader order.” | `src\datasets\oct_volumes.py` sorted file discovery; `src\eval_downstream.py` test loaders `shuffle=False` | Prediction order is sorted test filename order with no test shuffling. | **CONFIRMED** |
| S7-03 | L322--324: “The join is validated by exact label reconstruction, filename order, in-volume race/gender fields, and in-volume labels: all 3,000 labels agree in all 16 probe directories.” | Test metadata; 3,000 test NPZ scalar fields; 16 D-run prediction NPZs | Metadata↔in-volume agreement: labels 3000/3000, race 3000/3000, gender 3000/3000. Prediction-label agreement is 3000/3000 in each of the 16 D-run directories. | **CONFIRMED** |
| S7-04 | L324--325: “Four contaminated probes are excluded below.” | `subgroup_auc.json` status; `subgroup_analysis.py` `ARM_STATUS`; retracted run logs/config | Exactly four `frozen_cover_random_ep*` probes are marked RETRACTED and filtered from Fig. 6. | **CONFIRMED** |
| S7-05 | L325--326: “The aggregate random and oracle epoch-50 mean-pool artifacts contain no per-sample predictions, so neither enters this analysis.” | `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_random\ep50_test_predictions.npz`; corresponding oracle file | Both files exist, contain 3,000 `labels` and `probs`, and reproduce metadata labels 3000/3000. Valid ep75 and ep100 prediction files also exist for both policies. | **WRONG** — six random/oracle probes were omitted without the stated reason. |
| S7-06 | L331: “Subgroup results for 12 valid probes from five pretraining runs.” | All D-run and local downstream prediction NPZs; SHA-256 de-duplication of MIRAGE copies | 18 unique valid saved probes from six trajectories after excluding four retracted arms and de-duplicating the byte-identical MIRAGE copies. | **WRONG** |
| S7-07 | L332: “Black patients have the lowest AUC in every probe.” | Race-stratified AUC recomputed from all 18 unique valid prediction NPZs | Black is the lowest point estimate in 18/18. | **CONFIRMED** |
| S7-08 | L333--334: “each positive severity stratum is compared with the shared negative pool because mean deviation defines the label.” | Metadata fields `md`, `glaucoma`; subgroup calculation | `glaucoma == yes` iff `md <= -2` for 3000/3000 test subjects. Positive strata: severe=334, moderate=460, mild=672; shared negatives=1534. | **CONFIRMED** |
| S7-09 | L334--335: “Individual racial gaps are not statistically significant.” | Raw prediction NPZs; independent-sample DeLong variance for disjoint race groups | Blob ep50 Asian−Black difference=0.0935356, 95% CI [0.0402729, 0.1467983], p=0.000577; still significant after Holm-18 and Bonferroni-54 correction. | **WRONG** |
| S7-10 | L339: “The test split contains 2,318 white, 431 Black, and 251 Asian participants.” | Test metadata `race` | white=2318, black=431, asian=251. | **CONFIRMED** |
| S7-11 | L340: “Black participants have the lowest AUC in 12/12 valid probes” | All unique valid saved prediction NPZs | The complete available count is 18/18, not 12/12. | **WRONG** |
| S7-12 | L341: “The max--min racial gap varies 1.97x” | All 18 unique valid probes | \(0.0935356297/0.0475226247=1.9682337\), rounding to 1.97x. | **CONFIRMED** |
| S7-13 | L341--342: “from 0.0475 for COVER epoch 34 to 0.0935 for blob epoch 50” | COVER ep34 and blob ep50 prediction NPZs | 0.0475226247 and 0.0935356297. These remain the extrema over all 18 probes. | **CONFIRMED** |
| S7-14 | L342--343: “while overall AUCs differ by 0.032.” | All 18 unique valid prediction NPZs | Range=0.8854851648−0.8483465283=0.0371386366, rounding to 0.037. The 0.0324 value applies only to the incomplete 12-probe subset. | **WRONG** |
| S7-15 | L343--345: “the correlation between overall AUC and racial gap is not significant (Spearman \(\rho=0.427\), \(p=0.167\)); we do not claim an accuracy--fairness tradeoff.” | Overall and race-gap AUCs recomputed from all valid saved probes; `scipy.stats.spearmanr` | Complete-set nominal rho=0.4881321, p=0.0398595. However, repeated checkpoints are non-independent, so this nominal p-value is not a valid policy-level test and no tradeoff claim is warranted. | **WRONG** numeric claim; the refusal to claim a tradeoff is appropriate. |
| S7-16 | L345--346: “No individual subgroup gap is statistically significant, so the supported result is the consistency of ordering.” | Same DeLong calculation as S7-09 | At least blob ep50 has a multiplicity-robust significant Asian−Black gap. Ordering consistency is true, but it is not the only supported result. | **WRONG** |
| S7-17 | L348--349: “Mean deviation defines the FairVision glaucoma label, making within-bin AUC undefined.” | Test metadata `md`, `glaucoma` | Exact rule agreement 3000/3000; each MD bin is label-pure, so ordinary within-bin AUC has only one class. | **CONFIRMED** |
| S7-18 | L349--350: “We instead compare positives in each severity stratum with the same 1,534 negatives.” | Raw metadata and all prediction NPZs | Every severity AUC uses the same 1534 negative subjects. | **CONFIRMED** |
| S7-19 | L350: “The severe-to-mild AUC gap is 0.1306--0.1394 in every valid probe” | Severity AUCs recomputed over all 18 unique valid probes | Complete range=0.1286300 (oracle ep100) to 0.1394349 (fork ep25). The stated range is for the incomplete 12-probe set. | **WRONG** |
| S7-20 | L350--351: “a spread of 0.009.” | Same complete severity calculation | Spread=0.0108049, rounding to 0.011. | **WRONG** |
| S7-21 | L351--352: “Thus none of the tested mask policies materially alleviates the mild-disease penalty.” | 18 probe estimates from six non-replicated trajectories | All point estimates show a large penalty, but repeated checkpoints and one pretraining seed per policy do not establish a policy effect or equivalence. | **MISLEADING** |
| S8-01 | L357--358: “The \(B=1\rightarrow64\) intervention ... isolates the collation mechanism” | `arm_stats_b1\arm_stats.json`; `arm_stats\arm_stats.json`; inventory note; `scripts\arm_stats_table.py` | B=1 used n=256 and B=64 used n=1534 under separate sampling protocols, not a paired same-slice intervention. The contrast is diagnostic but does not strictly isolate batch size. | **MISLEADING** |
| S8-02 | L358--360: “zero-anatomy context rises 0.00\(\rightarrow\)4.63% for random” | Raw keys `random (stock JEPA).zero` in B1/B64 JSONs | 0.000000% → 4.628422%. | **CONFIRMED** |
| S8-03 | L360: “1.56\(\rightarrow\)10.10% for envelope” | Raw keys `envelope(mirage_envelope).zero` | 1.562500% → 10.104302%. | **CONFIRMED** |
| S8-04 | L361: “2.34\(\rightarrow\)11.02% for COVER.” | Raw key `COVER floor 0.15 prefix.zero` | 2.343750% → 11.016949%, but these values are specifically for COVER **f=0.15**, not the clean f=0.21 trajectory discussed next. | **MISLEADING** |
| S8-05 | L361--363: “Table ... is the post-collation composition ablation, and Figure ... is the completed four-point purity sweep.” | `composition_vs_auc_ep50.json`; masking configs for the four arms | Four completed points exist, but policy, shape, target count, context budget, guide source, and precision co-vary. This is an observational comparison, not an isolated composition/purity ablation. | **MISLEADING** |
| S8-06 | L363--364: “A clean COVER \(f=.21\) trajectory reaches AUC 0.8483 ... at epoch 27” | `frozen_meanpool_cover_f021_ep27\test_predictions.npz`; clean config | AUC=0.8483465283. Config uses stock `prefix` truncation and `amp_target=false`. | **CONFIRMED** |
| S8-07 | L364: “0.8522 ... at epoch 30” | `frozen_meanpool_cover_f021_ep30\test_predictions.npz` | AUC=0.8522489777. | **CONFIRMED** |
| S8-08 | L364: “0.8571 at ... epoch 34” | `frozen_meanpool_cover_f021_ep34\test_predictions.npz` | AUC=0.8570825722. | **CONFIRMED** |
| S8-09 | L364--365: “these are checkpoints within one run, not a cross-floor dose response.” | All three result JSON `encoder_checkpoint` paths | All point to checkpoints in `D:\jepa_phase0\runs\cover_f021_ep25`; floor is fixed at 0.21. | **CONFIRMED** |
| S8-10 | L365--367: “The matching epoch-50 probe for this trajectory was still pretraining at the time of writing and is therefore not reported.” | Live clean-run log; run directory contents | At audit time the log is in epoch 38 and no ep50 checkpoint/probe artifact exists. | **CONFIRMED** |
| S8-11 | L374--378: “The regional probes are separately trained on a 1,000-volume subset ... [versus] the 3,000-volume standard probes.” | Region run logs; feature-cache tensor shapes | Regional probes train on 2000 Training volumes, select on 600 Validation volumes, and test on a 1000-volume Test subset. Standard probes test on 3000. | **WRONG** training-set description. |
| S8-12 | L377--378: “The \(d=1\) and cross-attention values are retained in experiment documentation; their local result JSONs were not recovered.” | Exhaustive JSON/CSV/NPZ/log search; `linear_sweep_random_posfix_d1\summary.csv` | No per-run JSON was found for either row, but d=1 has a raw summary CSV with the exact value. Cross-attention has no matching raw artifact. Treating both as equally DOC-only is inaccurate. | **MISLEADING** |
| S8-13 | L384: “Probe head, random ep100 ... attentive \(d=1\) ... 0.8706” | `C:\Users\Gary\Desktop\jepa\results\downstream\linear_sweep_random_posfix_d1\summary.csv`, row `ep=100`, field `test_auc` | 0.8705706131683657, rounding to 0.8706. No per-sample predictions survive. | **CONFIRMED** |
| S8-14 | L385: “mean-pool ... 0.8746” | `meanpool_sweep_random\ep100_test_predictions.npz` | Recomputed AUC=0.8745808957846787, rounding to 0.8746. | **CONFIRMED** |
| S8-15 | L386: “cross-attention ... 0.8791” | Searched `C:\Users\Gary\Desktop\jepa\results`, `D:\jepa_phase0\reports`, and `D:\jepa_phase0\runs` for matching raw result/prediction artifacts | No frozen random-ep100 cross-attention JSON, CSV, NPZ, or log was found. Existing cross-attention artifacts are fine-tuned runs with different AUCs and `freeze_encoder=false`. | **UNVERIFIABLE** |
| S8-16 | L388: “Region, random ep50 ... all positions ... 0.8608341” | `region_features\random_ep50_s100.pt`; rerun `train_probe` on CUDA with seed 42 | 0.8608340834083408. | **CONFIRMED** |
| S8-17 | L389: “anatomy positions ... 0.8746555” | Same raw feature cache and rerun | 0.8746554655465547. | **CONFIRMED** |
| S8-18 | L390: “background positions ... 0.8543614” | Same raw feature cache and rerun | 0.8543614361436144. | **CONFIRMED** |
| S8-19 | L391: “Region, oracle ep50 ... all positions ... 0.8682588” | `region_features\oracle_ep50_s100.pt`; rerun on CUDA | 0.8682588258825882. | **CONFIRMED** |
| S8-20 | L392: “anatomy positions ... 0.8746475” | Same raw feature cache and rerun | 0.8746474647464747. | **CONFIRMED** |
| S8-21 | L393: “background positions ... 0.8652265” | Same raw feature cache and rerun | 0.8652265226522653. | **CONFIRMED** |
| S8-22 | L395: “Long-horizon placement ... random ep100 ... 0.8745809” | `meanpool_sweep_random\ep100_test_predictions.npz` | Recomputed AUC=0.8745808957846787. | **CONFIRMED** |
| S8-23 | L396: “oracle ep100 ... 0.8854852” | `meanpool_sweep_oracle\ep100_test_predictions.npz` | Recomputed AUC=0.8854851648224599. | **CONFIRMED** |
| S8-24 | L401--402: “The head ablation shows that mean pooling is not uniquely privileged, although cross-attention changes the point estimate.” | d=1 summary CSV; mean-pool predictions; absent cross-attention artifact | Artifact-backed rows only show mean-pool 0.8746 versus d=1 0.8706. The cross-attention premise cannot be checked. | **UNVERIFIABLE** |
| S8-25 | L402--405: “anatomy positions outperform background positions; these are arm-specific controls, not a global regional estimate.” | Exact regional reruns; no seed replication or paired uncertainty calculation | Anatomy has higher point estimates by 0.0202940 (random) and 0.00942094 (oracle). “Outperform” implies more certainty than was tested; the arm-specific caveat is correct. | **MISLEADING** |
| S8-26 | L405--407: “random/oracle all-position AUCs are 0.8608341/0.8682588 with early stopping at epochs 43/42” | CUDA rerun from raw feature caches | AUCs and epochs reproduce exactly: random 0.8608340834 at 43; oracle 0.8682588259 at 42. | **CONFIRMED** |
| S8-27 | L407--408: “the composition table uses standard mean-pool AUCs 0.8640971/0.8740299 stopped at epochs 46/47.” | Random/oracle ep50 result JSONs and prediction NPZs | 0.8640970650 at epoch 46; 0.8740299461 at epoch 47. | **CONFIRMED** |
| S8-28 | L408--409: “Absolute AUCs across the two protocols are not interchangeable.” | Protocol configs and sample counts | Regional: train/val/test=2000/600/1000 and separately fitted heads; standard: 6000/1000/3000. | **CONFIRMED** |
| S8-29 | L409--411: “The epoch-100 oracle/random difference shows that anatomical placement can remain useful, but with one trajectory per policy it does not identify a causal policy effect.” | Ep100 prediction NPZs; pretraining provenance | Observed difference=0.8854851648−0.8745808958=+0.0109042690. One trajectory per policy supports only a descriptive contrast, not the “useful” interpretation. | **MISLEADING** |

## WRONG and MISLEADING claims with exact corrected replacement text

1. **Lines 325--326 — missing random/oracle predictions (WRONG).** Replace with:
   > Valid per-sample predictions are available for the random and oracle epoch-50, epoch-75, and epoch-100 mean-pool probes; all six are included after exact label-order validation.

2. **Lines 331--335 — Fig. 6 scope and significance (WRONG).** Replace the caption with:
   > Subgroup results for 18 unique valid saved probes from six pretraining trajectories. Left: racial AUC gap by probe; Black patients have the lowest point-estimate AUC in all 18. Right: each positive severity stratum is compared with the shared negative pool because mean deviation defines the label. Probe checkpoints from the same trajectory are non-independent.

3. **Lines 334--335 and 345--346 — “no individual racial gap is significant” (WRONG).** Replace with:
   > Black patients have the lowest point estimate in every valid probe. At least the blob epoch-50 Asian--Black contrast is statistically significant (AUC difference 0.0935; DeLong 95% CI 0.0403--0.1468; multiplicity-adjusted \(p<0.05\)).

4. **Line 340 — 12/12 (WRONG).** Replace with:
   > Black participants have the lowest point-estimate AUC in 18/18 unique valid saved probes.

5. **Lines 342--345 — overall range and Spearman result (WRONG).** Replace with:
   > The max--min racial gap ranges from 0.0475 for COVER epoch 34 to 0.0935 for blob epoch 50 (1.97x), while overall AUC spans 0.0371. Across all 18 probes, the nominal Spearman correlation is \(\rho=0.488\) (\(p=0.0399\)); because repeated checkpoints from the same trajectory are non-independent, this p-value is not a valid policy-level test and we do not infer an accuracy--fairness tradeoff.

6. **Lines 350--352 — severity range and policy claim (WRONG/MISLEADING).** Replace with:
   > Across all 18 unique valid probes, the severe-to-mild AUC gap ranges from 0.1286 to 0.1394 (spread 0.0108). These non-independent, single-seed trajectories consistently show a mild-disease penalty but do not identify masking-policy effects.

7. **Lines 357--361 — “isolates” and unlabeled COVER setting (MISLEADING).** Replace with:
   > Separate \(B=1\) (\(n=256\)) and \(B=64\) (\(n=1{,}534\)) audits are consistent with batch-minimum collation increasing zero-anatomy context: 0.00\(\rightarrow\)4.63% for random, 1.56\(\rightarrow\)10.10% for envelope, and 2.34\(\rightarrow\)11.02% for COVER \(f=.15\).

8. **Lines 361--363 — “composition ablation” / “purity sweep” (MISLEADING).** Replace with:
   > Table~\ref{tab:composition} reports post-collation composition, and Figure~\ref{fig:composition} compares four completed, confounded policy points; it is not an isolated purity ablation or dose response.

9. **Lines 374--378 — regional subset size (WRONG).** Replace with:
   > The regional probes are separately trained on 2,000 training volumes, selected on 600 validation volumes, and evaluated on a 1,000-volume test subset, so they are not directly interchangeable with the 3,000-volume standard test probes.

10. **Lines 377--378 — d=1 and cross-attention provenance (MISLEADING).** Replace with:
    > The \(d=1\) value is backed by `results/downstream/linear_sweep_random_posfix_d1/summary.csv`, although its per-run JSON and predictions were not recovered. No raw artifact backs the cross-attention value, so that row is omitted.

11. **Lines 402--405 — regional “outperform” wording (MISLEADING).** Replace with:
    > In separately trained random- and oracle-epoch-50 regional probes, anatomy-position pooling has higher test-AUC point estimates than background-position pooling; these are arm-specific descriptive controls, not a global regional estimate.

12. **Lines 409--411 — “can remain useful” (MISLEADING).** Replace with:
    > At epoch 100, the single oracle trajectory has a test-AUC point estimate 0.0109 above the single random trajectory; with one trajectory per policy, this is descriptive and does not identify a causal policy effect.

## Verdict on the two DOC-ONLY ablation rows (keep with citation, or delete)

### `d=1`, AUC 0.8706 — **KEEP, with a direct raw-CSV citation and limitation**

Backing artifact found:

`C:\Users\Gary\Desktop\jepa\results\downstream\linear_sweep_random_posfix_d1\summary.csv`

Row `ep=100`, field `test_auc` is `0.8705706131683657`. The directory also contains the sweep plots. No per-run result JSON, config, or prediction NPZ survives, so the paper should cite the CSV directly and should not imply per-sample re-analysis is possible.

### Cross-attention, AUC 0.8791 — **DELETE**

No matching frozen random-epoch-100 cross-attention JSON, CSV, NPZ, or log was found in:

- `C:\Users\Gary\Desktop\jepa\results`
- `D:\jepa_phase0\reports`
- `D:\jepa_phase0\runs`

Exact-string and rounded numeric scans found no test-AUC artifact. Existing cross-attention files are **fine-tuning** runs (`freeze_encoder=false`) with different AUCs (for example, random 0.8871778 and oracle 0.8937172), so they cannot support 0.8791. Delete the row and the sentence that interprets it.

## Retracted-arm contamination check

- Retracted probes: `frozen_cover_random_ep30`, `ep50`, `ep75`, `ep100`.
- Their recomputed AUCs are 0.8557668, 0.8589675, 0.8611851, and 0.8607336.
- Run logs record `amp_target: True`; the archived campaign configuration uses `enc_truncate: window`. These are the stated contamination mechanisms.
- `main.tex` contains none of the four retracted names and none of their rounded AUCs.
- `subgroup_analysis.py` marks all four `RETRACTED`; `make_fairness_figure.py` requires `status == "OK"` and has no retracted arm in `FAMILY`.

**Verdict: no retracted arm appears in a reported result in Sections 3, 7, or 8.**

The naming alias trap is also handled correctly in the current generator: `frozen_meanpool_envelope_ep30` and `frozen_meanpool_mirage_ep50/75/100` all map to the single `envelope` pretraining family. The C- and D-drive MIRAGE prediction files for epochs 50/75/100 are byte-identical by SHA-256 and were de-duplicated in the 18-probe recount.

## Figure-vs-prose consistency

`scripts\make_fairness_figure.py` currently sets:

- Panel (a), lines 61--63: “Black lowest point estimate in 12/12; no individual gap significant.”
- Panel (b), lines 88--89: “Severity-gap estimates are similar across tested probes (gap \(\approx0.13\) in every arm).”

Panel (a) agrees with the current prose but is factually wrong twice: the complete saved valid set is 18/18, and blob epoch 50 has a multiplicity-robust significant Asian--Black gap. It also explicitly asserts a significance conclusion, contrary to the requested overclaim removal.

Panel (b) is descriptively compatible with the raw estimates and does not assert statistical significance. Its scope is nevertheless incomplete because the generator hard-codes only 12 probes. The prose goes further than the panel by claiming that no policy materially alleviates the penalty; that causal/comparative framing is not supported by single-seed, non-independent trajectories.

## Artifacts consulted

- `C:\Users\Gary\Desktop\jepa\paper\genai4health2026\main.tex`
- Official Harvard-FairVision GitHub README and CC BY-NC-ND 4.0 license link
- `D:\jepa_phase0\fairvision-glaucoma\metadata\data_summary_glaucoma.csv`
- All 10,000 FairVision glaucoma NPZ headers; all 3,000 test NPZ demographic/label scalar fields
- `C:\Users\Gary\Desktop\jepa\src\datasets\oct_volumes.py`
- `C:\Users\Gary\Desktop\jepa\src\eval_downstream.py`
- `C:\Users\Gary\Desktop\jepa\src\masks\{multiblock,curriculum,cover,anatomy}.py`
- Relevant pretraining and probe configs under `configs\`
- `D:\jepa_phase0\fairvision-glaucoma\checkpoint-ep25\README.md`
- All scoped prediction/result artifacts under `D:\jepa_phase0\runs\`
- Random/oracle/mirage downstream result and prediction artifacts under `C:\Users\Gary\Desktop\jepa\results\downstream\`
- `D:\jepa_phase0\reports\subgroup\subgroup_auc.json` and `.csv`
- `scripts\subgroup_analysis.py`
- `scripts\make_fairness_figure.py`
- `D:\jepa_phase0\reports\arm_stats_b1\arm_stats.json`
- `D:\jepa_phase0\reports\arm_stats\arm_stats.json`
- `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json`
- `D:\jepa_phase0\reports\target_composition\summary.json`
- `D:\jepa_phase0\reports\region_features\random_ep50_s100.pt`
- `D:\jepa_phase0\reports\region_features\oracle_ep50_s100.pt`
- `D:\jepa_phase0\reports\downstream_region_auc\*\region_auc.json` and run logs
- `C:\Users\Gary\Desktop\jepa\results\downstream\linear_sweep_random_posfix_d1\summary.csv`
- `paper\genai4health2026\research\ablation_inventory.json` and `numbers_master.csv`, queried programmatically only

Methods: rank-based Mann--Whitney AUC with tie averaging; exact metadata label conversion (`yes`/`no` to 1/0); SciPy Spearman correlation; independent-sample DeLong variance for disjoint race groups; CUDA rerun of the regional linear probes from saved feature tensors with seed 42.
