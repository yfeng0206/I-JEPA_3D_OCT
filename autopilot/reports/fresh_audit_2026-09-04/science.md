# Fresh scientific manuscript audit — author-side working draft

**Audit date:** 2026-09-04  
**Authorized role:** author-requested local scientific audit; not a conference review or acceptance decision  
**Baseline:** `de145d7005f57e871bc0181bf58b271775d1d25d`  
**Manuscript read:** `paper\genai4health2026\main_submission.tex`, all 2,059 source lines, including appendices and captions  
**Handling:** local files only; no manuscript content sent externally; no training, manuscript edits, rendering, commits, pushes, credential access, or GUI/PDF inspection

The accountable author should verify every location and judgment before acting on this working draft.

# Comments to authors

## Executive conclusion

The narrow supported central claim is:

> On one repeatedly inspected FairVision test split, for one completed continuation per masking policy from a shared epoch-25 ancestor, the implemented **envelope** and **centroid** rectangle-placement policies produced higher frozen-probe glaucoma-classification AUC than the implemented unguided random policy at matched epochs and precision. Greater measured tissue coverage or target purity was not monotonically associated with higher AUC in these runs.

The evidence does **not** identify target placement alone as the cause, because placement changes delivered context and other task geometry. It does not establish an expected policy ranking across retraining seeds. The centroid and segmenter-guided envelope arms are not statistically demonstrated to be equivalent: their paired difference intervals include zero at epochs 50 and 75, and centroid is higher at epoch 100. “Matches a segmenter” is therefore an informal similarity claim, not an equivalence result, and the title can also be misread as a segmentation-task result even though the outcome is glaucoma classification.

No P0 issue was identified that requires new training or invalidates the reported saved-run AUC values. The principal risks are claim scope, construct naming, stale replication status, and secondary analyses being described more strongly than their designs support. Most can be resolved by bounded wording and status corrections.

## Version and status board

| Item | Fresh observation |
|---|---|
| Git HEAD | `de145d7005f57e871bc0181bf58b271775d1d25d`, matching the requested baseline |
| Working tree at intake | Pre-existing modified `autopilot\RESOURCE_MONITOR.csv`; audit-report directory untracked |
| Manuscript length | 2,059 source lines |
| Primary analyzed pretraining seeds | **One completed continuation per policy** |
| Primary probe seed | **One**, seed 42 |
| Mask-geometry redraw seeds | Three measurement draws: 42, 1234, 2026; these are not pretraining replications |
| Label-efficiency subset repeats | Five subsets per fraction below 100%; these are probe refits, not pretraining replications |
| Planned replication | Six additional legs in configs: 3 policies × seeds 1234/5678 |
| Locally observed replication result | No completed epoch-50 replication result; only `rep_random_s1234` exists locally and its latest records are partial epoch 27 |
| Active local replication process | None observed at audit time |
| Prediction AUC recomputation | Independent parent-side local check: all 43 HANDOFF prediction files are present and match their six-decimal AUC records (31 primary, 2 excluded, 4 retracted, 6 supplementary) |
| Prediction-array scope | Every file has `n=3000`, labels 1466/1534, and one common label sequence; equal label arrays do **not** establish subject-identity pairing |
| Prediction snapshot | `autopilot\reports\fresh_audit_2026-09-04\prediction_snapshot.json` |

## Neutral study map

| Element | Current design |
|---|---|
| Research question | Does steering JEPA predictor targets toward retinal tissue improve learned representations for glaucoma classification, and does more anatomically precise steering add further benefit? |
| Data/system | FairVision glaucoma OCT; fixed patient-disjoint split; test `N=3000` with 1,466 positive and 1,534 negative volumes |
| Representation model | A 2D ViT-B/16 JEPA encoder applied to individual B-scans; downstream slice features are averaged across 100 B-scans per volume |
| Experimental unit for pretraining-policy inference | One post-fork continuation per policy; no completed replication across pretraining stochasticity |
| Test-set unit | Subject/volume; all arms evaluated on the same cases |
| Intervention | Six target-mask policies differing in location, shape, guide, coverage logic, collation path, and realized geometry |
| Main comparators | Unguided random rectangles; segmenter-guided envelope rectangles; segmentation-free intensity-centroid rectangles |
| Primary outcome | Frozen MeanPool linear-probe test ROC AUC for glaucoma classification |
| Main timing | Epochs 50, 75, and 100 for the three long rectangle arms; epoch 50 is the broadest cross-policy endpoint |
| Main uncertainty | Subject-level paired bootstrap intervals and correlated-ROC DeLong tests on the fixed test set |
| Generalization target actually supported | These trained continuations and this test split, not an expected ranking over retraining seeds or external cohorts |

## Current arm table reconstructed from the manuscript

Sources: `paper\genai4health2026\main_submission.tex:217-242,339-383,397-417,459-487`; `paper\genai4health2026\auto\auto_numbers.tex:1-139`; `paper\genai4health2026\auto\table_allprobes.tex:1-39`.

| Policy | Scientific manipulation | Valid reported endpoints | Matched values central to the paper | Interpretation boundary |
|---|---|---|---|---|
| **random** | Unguided axis-aligned rectangles | ep50, 75, 100 | AUC 0.8641, 0.8723, 0.8746 | Null policy, one continuation |
| **envelope** | Same rectangle family, rejection-sampled toward a MIRAGE retinal envelope | ep30, 50, 75, 100 | ep50/75/100 AUC 0.8761/0.8803/0.8807; versus random +0.0120/+0.0080/+0.0062 | Policy improves in these runs; delivered context also changes |
| **centroid** (`oracle` in artifacts) | Segmentation-free intensity-centroid band used to place ordinary rectangles | ep50, 75, 100 | AUC 0.8740/0.8836/0.8855; versus random +0.0099/+0.0113/+0.0109 | Highest ep100 point estimate; mechanism not isolated |
| **anatomy-v1** | MIRAGE-guided ragged shape, early implementation | ep30 only | AUC 0.8583; versus envelope at ep30 +0.0044, CI [+0.0009, +0.0078] | Early positive shaped-policy result; different implementation and epoch |
| **anatomy-v2** | MIRAGE-guided ragged connected targets, fixed 4×16 loss slots | valid ep35, 40, 50, 75 | ep50 AUC 0.8654; versus fp32 random +0.0013, CI [-0.0055, +0.0082]; versus envelope -0.0107, CI [-0.0167, -0.0046]. ep75 versus random -0.0111 | No ep100 value; differs in area, context, slots, collation, and possibly guide provenance |
| **cover, f=0.21** | Segmenter-driven greedy coverage floor | ep27, 30, 34, 50, 73, 75, 100 | ep50/75/100 AUC 0.8643/0.8639/0.8577; versus fp32 random +0.0002/-0.0084/-0.0168 | Implemented arm has a target-truncation defect and does not test aggressive coverage |

### Precision and epoch assessment

The main table’s compared deltas are now matched by epoch and probe precision. The long random/envelope/centroid contrasts are fp16-versus-fp16. Anatomy-v2 and cover contrasts use fp32 re-probed nulls at matching epochs. This is a verified strength, not a fresh defect. The anatomy-v1 positive result is a valid matched ep30 comparison but cannot be pooled with anatomy-v2 ep50 or used as a replication of it.

The fp32 robustness table contains eight re-probes and omits centroid ep75, but that omission does not create a cross-precision headline contrast because all three ep75 long-arm values compared in the primary table are fp16.

## Prioritized findings

### P0 — submission-threatening or factually invalid

**No P0 finding identified.** The audit found no basis to invalidate the saved-run AUC values or require a new six-day training campaign. The minimum adequate remedies below are predominantly narrowing, relabeling, and stale-status correction.

### P1 — important inference or reproducibility issues

#### P1-01 — “Matches a segmenter” is not an equivalence result and is task-ambiguous

- **Location:** `paper\genai4health2026\main_submission.tex:32-35,43-60,754-769`; `paper\genai4health2026\auto\auto_numbers.tex:111-122`
- **Observation:** The title says segmentation-free guidance “Matches a Segmenter.” Centroid-minus-envelope is -0.0020 with CI [-0.0069, +0.0029] at ep50 and +0.0033 with CI [-0.0013, +0.0079] at ep75; these intervals crossing zero do not establish equivalence. At ep100 centroid is higher by +0.0047, CI [+0.0004, +0.0091]. The phrase can also be read as matching a segmenter on a segmentation task, but the measured outcome is glaucoma-classification AUC.
- **Evidence:** No equivalence margin, non-inferiority margin, or equivalence test is defined. Failure to detect a difference is not proof of equality.
- **Impact:** The title turns a bounded three-epoch comparison into a stronger statistical and task-level claim than was tested.
- **Minimum bounded requested action:** Retitle around segmentation-free **target placement for glaucoma classification**. In prose, say centroid was not resolved from envelope at ep50/75 and was higher at ep100; do not call the first two equivalence.
- **Confidence:** High.

#### P1-02 — “Region matters” and “H1 holds” over-identify target placement as the cause

- **Location:** `paper\genai4health2026\main_submission.tex:217-242,388-439,687-708,979-1002,1904-1913`
- **Observation:** The manuscript says the rectangle arms differ only in placement and that “H1 holds,” but it also measures delivered context changing from 24.7% for random to 30.7% for envelope and 32.9% for centroid. The geometry caption later calls the four rectangle policies “near single-variable” and says they “separate only” on anatomy hidden.
- **Evidence:** Placement/rejection changes rectangle overlap and therefore delivered context. Cover also has different coverage logic and a known post-placement truncation defect. The manuscript itself recognizes that mechanism and retained context are not identified.
- **Impact:** A defensible policy-package comparison is presented as an isolated causal effect of region/placement.
- **Minimum bounded requested action:** Replace “region matters,” “where you aim beats,” and “H1 holds” with “the implemented tissue-directed placement policies outperformed the implemented random policy in these runs.” Remove “separate only”/“near single-variable” from the geometry caption or explicitly name delivered-context differences.
- **Confidence:** High.

#### P1-03 — “Anatomical precision does not add” is broader than the mixed, confounded evidence

- **Location:** `paper\genai4health2026\main_submission.tex:57-64,103-125,459-470,527-554,687-708`; `paper\genai4health2026\auto\auto_numbers.tex:59-69`
- **Observation:** The abstract gives the categorical headline “Anatomical precision does not add.” Yet anatomy-v1 is positive versus envelope at ep30 (+0.0044, CI [+0.0009, +0.0078]), while anatomy-v2 is negative versus envelope at ep50 and unresolved versus random. Anatomy-v2 differs in mask ratio, delivered context, loss slots, collation, implementation, and guide provenance; cover is defective.
- **Evidence:** These results reject a simple monotonic “more anatomy is always better” story. They do not isolate “precision,” target shape, or coverage as a causal variable, and one shaped implementation has an early positive result.
- **Impact:** The strongest negative headline suppresses the study’s own mixed result and can be read as a general conclusion about anatomical precision.
- **Minimum bounded requested action:** Use “greater measured anatomical specificity was not monotonically beneficial” or “did not consistently add in the tested, confounded arms.” Keep the anatomy-v1 ep30 positive result visible in the abstract-level interpretation.
- **Confidence:** High.

#### P1-04 — “3D retinal OCT” language obscures a 2D encoder with late volume pooling

- **Location:** `paper\genai4health2026\main_submission.tex:43-44,259-284,320-325,360-362`
- **Observation:** The abstract opens “In 3D retinal OCT,” while the pipeline tokenizes one B-scan at a time with a 2D 16×16 grid. Per-slice features are then mean-pooled across 100 B-scans; there is no learned cross-slice interaction in the described encoder.
- **Evidence:** The input data are volumetric, but the representation model is a slice encoder plus permutation-invariant late aggregation.
- **Impact:** Readers may infer a 3D JEPA or volume encoder and overgeneralize the masking result to genuinely 3D architectures.
- **Minimum bounded requested action:** State at first mention: “on volumetric OCT, using a 2D B-scan JEPA encoder and mean pooling across 100 slices.” Avoid “3D JEPA/3D representation” implications.
- **Confidence:** High.

#### P1-05 — The released statistics artifact calls a post-inspection family “confirmatory”

- **Location:** `paper\genai4health2026\main_submission.tex:326-331,744-747,1857-1865`; `autopilot\p1c_stats.py:129-176`; `D:\jepa_phase0\autopilot_out\p1_stats\p1c_stats.json:14-16,7785,7898,8479,8592,9060,9195,9578,10002,10203`
- **Observation:** The manuscript correctly says policies, checkpoints, and analyses followed repeated test inspection and that intervals are descriptive. The generating code and JSON nevertheless label the nine long-arm contrasts “confirmatory.”
- **Evidence:** Benjamini–Hochberg within the declared nine-comparison family is computationally coherent, but a family fixed after test inspection is not made confirmatory by naming or correction.
- **Impact:** Downstream consumers may quote the structured artifact rather than the manuscript caveat and misstate the inferential status.
- **Minimum bounded requested action:** Rename the code/JSON family to `primary_descriptive` or `headline_descriptive`. Preserve the q-values but state they organize multiplicity within an adaptively selected analysis, not confirmatory error control.
- **Confidence:** High.

#### P1-06 — Background claims combine different arms, epochs, and evaluation protocols

- **Location:** `paper\genai4health2026\main_submission.tex:556-576,1374-1414`; `autopilot\bgsig\a2_region_incremental.py:1-6,59-102`
- **Observation:** The “background matters: for pretraining, not for the classifier” narrative combines: random ep100 predictor skill; envelope ep100 self-similarity; and downstream regional probes produced from ep50 checkpoints using 25 stratified slices per volume and Training/Validation/Test sizes 2000/600/1000. The main probe uses 100 slices and the full 6000/1000/3000 split.
- **Evidence:** The regional analysis does support high redundancy and weak incremental signal under its reduced ep50 protocol. It does not directly show why the ep100 random baseline is strong, nor that background is generally irrelevant to classifiers. A background-only probe remains high because tissue information can mix into background tokens, and the two causal controls were not run.
- **Impact:** Heterogeneous diagnostic analyses are stitched into a single mechanism claim.
- **Minimum bounded requested action:** Name the arm, epoch, slice count, and subset size wherever each background number is interpreted. Replace “why unguided masking is such a strong baseline” and “not for the classifier” with “consistent with a redundant downstream contribution under an ep50 reduced-slice regional probe; mechanism unresolved.”
- **Confidence:** High.

#### P1-07 — Label-efficiency section title and conclusion outrun five descriptive subset repeats

- **Location:** `paper\genai4health2026\main_submission.tex:578-590,1417-1464`; `autopilot\p5_label_efficiency.py:162-216`; `D:\jepa_phase0\autopilot_out\p1_stats\p5_label_efficiency.json:2-25,49-57`
- **Observation:** “The advantage concentrates where labels are scarce” is stated as a result. The analysis uses five shared training subsets per fraction below 100%, reports arm-wise means and standard deviations, and performs no paired-delta interval or test of gap-by-label-fraction interaction. The prose later admits that widening was not tested.
- **Evidence:** At 5%, the observed mean gap is +0.0496 versus +0.0108 at full supervision, but five subset draws do not establish a trend or concentration effect.
- **Impact:** A useful exploratory observation is promoted to a comparative label-efficiency conclusion.
- **Minimum bounded requested action:** Rename the section “The observed gap is larger in low-label probe refits.” State the five-repeat descriptive basis in the first sentence and retain “whether the gap widens was not tested.”
- **Confidence:** High.

#### P1-08 — The broad fairness summary is factually false and exceeds the tested paired families

- **Location:** `paper\genai4health2026\main_submission.tex:1187-1201,1278-1293,1758-1778`; `autopilot\p7c_paired_subgroup.py:43-44,138-158`; `D:\jepa_phase0\autopilot_out\p1_stats\p7_fairness.json:4038-4053,4809-4824`
- **Observation:** The ethics appendix says “every subgroup point estimate rises at the matched epoch.” In the retained fairness JSON, the adequately sized divorced group falls from AUC 0.88280 under random to 0.87346 under centroid. Moreover, paired differential-benefit inference is implemented only for race and sex, with severity strata compared separately; it does not support “narrows none of them reliably” across all seven attributes.
- **Evidence:** Race, sex, and severity point estimates do rise in the highlighted ep100 comparisons, and all six race-by-sex cells rise at ep100. That narrower statement is supportable; the all-subgroup statement is not.
- **Impact:** The broad fairness reassurance is factually inaccurate and overextends the scope of adjusted paired analysis.
- **Minimum bounded requested action:** Restrict the statement to the explicitly verified race, sex, severity, and ep100 race-by-sex cells. State that other attributes were used for descriptive worst-group/gap consistency, not paired policy-benefit inference; note the divorced-group exception if retaining an all-attribute summary.
- **Confidence:** High.

#### P1-09 — Dataset “gender” is relabeled as biological “sex” without construct justification

- **Location:** `paper\genai4health2026\main_submission.tex:332-334,1204-1213`
- **Observation:** The manuscript acknowledges that the source artifact uses FairVision “gender” coding but says the paper reports it as “sex throughout.”
- **Evidence:** Gender and sex are not interchangeable constructs. No local evidence shows that the released variable is a biological-sex measure.
- **Impact:** This weakens fairness construct validity and can mischaracterize the subgroup attribute.
- **Minimum bounded requested action:** Use the dataset’s reported term “gender,” or provide a source-backed variable definition justifying “sex.” Do not silently recode the construct.
- **Confidence:** High.

#### P1-10 — Operating-point language claims usable benefit and calibration superiority despite a threshold trade-off

- **Location:** `paper\genai4health2026\main_submission.tex:1601-1647,1650-1695`; `paper\genai4health2026\auto\table_operating.tex:1-12`
- **Observation:** The manuscript says the AUC advantage “survives as a usable sensitivity gain” and calls centroid “the best-calibrated arm.” At the nominal 0.90 target, test specificity is 0.8794 for random and 0.8696 for centroid, so part of the sensitivity increase is purchased by more false positives. ECE is a single 15-bin point estimate with no uncertainty or reliability curve; race-stratified ECE worsens for the black group.
- **Evidence:** The manuscript later recognizes the shared-threshold, unequal-specificity trade-off and lack of untouched deployment evaluation.
- **Impact:** “Usable” and “best-calibrated” can be read as clinical utility claims that are not supported by this retrospective exploratory comparison.
- **Minimum bounded requested action:** Say “at one shared validation-selected threshold, centroid had higher sensitivity and lower specificity.” Replace “best-calibrated” with “lower measured overall Brier score and 15-bin ECE on this split.” Keep clinical utility explicitly unclaimed.
- **Confidence:** High.

#### P1-11 — Interpretability captions and synthesis are more causal/definitive than the evidence

- **Location:** `paper\genai4health2026\main_submission.tex:1472-1485,1519-1551,1573-1594`
- **Observation:** The appendix clearly says the attribution analyses use three fine-tuned probes rather than the frozen probes used for arm comparisons, and that archived arrays are not released. Yet a caption states the two-peak structure “is an OD/OS storage artefact” even though laterality labels are unavailable, and the final paragraph says diffuse attribution “supports the main claim.”
- **Evidence:** The mirror pattern is consistent with an orientation/laterality mixture, not ground-truth validated as OD/OS. Hand-picked patch heat maps, no intervals, different fine-tuned models, and non-recomputable arrays cannot explain the pretraining-policy effect.
- **Impact:** A useful cautionary appendix becomes an apparent mechanistic validation.
- **Minimum bounded requested action:** Change the caption to “consistent with an unverified orientation/laterality mixture.” Present diffuse attribution as contextual consistency only, not support for the causal masking claim. Retain the null/negative interpretability observations.
- **Confidence:** High.

### P2 — clarity, stale state, or secondary presentation issues

#### P2-01 — Replication is not currently “running”; actual analyzed seed count remains one

- **Location:** `paper\genai4health2026\main_submission.tex:710-733,812-874`; `autopilot\PROCESS_REGISTRY.csv:5-6`; `D:\jepa_phase0\runs\rep_random_s1234\train_20260826_180614_a0.log:21-23`; `D:\jepa_phase0\runs\rep_random_s1234\train_20260826_181255_a0.log:21-23`; `D:\jepa_phase0\runs\rep_random_s1234\jepa_patch_rep_random_s1234-log.csv:11027-11038`
- **Observation:** The manuscript twice says replication is in progress/running. Fresh local inspection found no matching active Python process, no completed epoch-50 replication, and only one replication directory. Its latest logs repeatedly resume from epoch 26 into epoch 27, and the CSV ends with partial/restarted epoch-27 records.
- **Evidence:** Six configs exist, but configuration is not execution. The historical process registry has no completion/exit record.
- **Impact:** “Running” is a stale factual status and can imply stronger near-term evidence than exists.
- **Minimum bounded requested action:** Replace with “a replication was planned and partially attempted; no completed result is reported.” State explicitly that the analyzed evidence remains one pretraining continuation per policy and one probe seed. No new run is required for a wording-only submission.
- **Confidence:** High for local status; remote/cloud activity was not inspected.

#### P2-02 — “Anatomical specificity” is encoded with two different quantities

- **Location:** `paper\genai4health2026\main_submission.tex:57-66,99-112,503-519,883-890,1884-1897`; `autopilot\make_fig_specificity_ladder.py:37-44,68-77`; `autopilot\make_fig_geometry_panel.py:30-34`
- **Observation:** In the abstract, anatomical precision is described using **purity**: the fraction of masked cells on tissue (97.1% for anatomy-v2). The “specificity ladder” is sorted by **anatomy hidden**: the fraction of all tissue cells hidden. These are different constructs and produce different envelope/cover orderings.
- **Evidence:** At ep50, purity is envelope 43.3% versus cover 44.2%, while anatomy hidden is envelope 77.6% versus cover 73.5%. The generator explicitly sorts `hidden_share_of_all_anat` and comments that this is increasing specificity.
- **Impact:** The figure can make an apparent dose-response axis out of a construct whose definition changes between prose and plots.
- **Minimum bounded requested action:** Rename the ladder “observed fraction of anatomy hidden” and reserve “target purity” for the fraction of masked cells on tissue. If “specificity” is retained, define one metric and use it consistently.
- **Confidence:** High.

#### P2-03 — Main-table visual emphasis bolds unresolved lower-block deltas

- **Location:** `paper\genai4health2026\main_submission.tex:397-421`; `paper\genai4health2026\auto\auto_numbers.tex:395-398`
- **Observation:** The lower-block delta macros are bold even when the intervals contain zero: anatomy-v2 versus random at ep50 (+0.0013) and cover versus random at ep50 (+0.0002).
- **Evidence:** The table caption does not define boldface as “reported contrast” rather than statistical separation, while elsewhere filled markers encode intervals excluding zero.
- **Impact:** Readers may interpret boldface as significance, creating a presentation contradiction with the text’s preserved null results.
- **Minimum bounded requested action:** Bold only estimates whose stated interval excludes zero, or remove boldface from all lower-block deltas and explain any remaining emphasis.
- **Confidence:** High.

## Direct answers to the requested scientific questions

### What is the narrow supported central claim?

The implemented envelope and centroid placement policies outperform implemented random masking on this fixed test split for these single continuations under matched frozen-probe comparisons. The ordering of target purity/anatomy coverage does not track AUC monotonically. This is a run-level comparative result, not a policy-level causal or retraining expectation.

### Does “matches a segmenter” mean equivalence when CIs cross zero?

No. The ep50 and ep75 centroid-envelope intervals cross zero, which supports “difference unresolved within these data,” not equivalence. Equivalence would require a justified margin and an equivalence/non-inferiority analysis. Ep100 instead favors centroid.

### Target placement versus causal mechanism?

The treatment is an implemented sampler policy. Placement changes delivered context through overlap/rejection and may change prediction difficulty. The study does not isolate location consistency, visible context, mask ratio, or task difficulty. “Placement policy outperformed random” is supported; “placement caused the gain” is not.

### 2D B-scan encoder versus 3D volume language?

The data are 3D volumes, but the described encoder is 2D per B-scan and volume aggregation is mean pooling. The manuscript should say this prominently.

### Precision matching and comparison epochs?

The current headline comparisons are matched on epoch and probe precision. Ep50 is the only broad five-policy endpoint. Anatomy-v1 ep30 is a separate implementation and cannot be treated as a seed replicate or direct temporal continuation of anatomy-v2.

### Adaptive test reuse and multiplicity?

Repeated test inspection is candidly disclosed and correctly makes the intervals descriptive. BH and max-\(|t|\) calculations are reasonable within their declared computational families, but they do not restore confirmatory status after adaptive analysis. The JSON label “confirmatory” should be corrected. No untouched-cohort demand is necessary if language remains explicitly descriptive and run-specific.

### Actual seed count and replication status?

One completed pretraining continuation per policy and one probe seed underpin the main arm table. Geometry has three measurement redraw seeds; label efficiency has five subset refits. The planned two additional pretraining seeds per random/envelope/centroid policy have no completed local result, and the local chain was not active at audit time.

### Background claims?

Supported: background targets are nontrivially predictable beyond a per-position reference in one random ep100 analysis; regional pooled features are highly redundant in an ep50 reduced-slice analysis. Not supported: that this establishes why random masking is strong, that background is generally useless to classifiers, or that position is the causal mechanism.

### Fairness claims?

Supported: worst-group identity is stable for five of seven attributes across the correlated probe inventory; adjusted paired gains are resolved for mild severity, white race, and both gender/sex strata in the specified family; race differential benefit is unresolved. Overclaimed: every subgroup point estimate rises, benefit/gap claims over all seven axes, and biological “sex” terminology for a source variable described as gender. The analysis is explicitly incomplete and is not a fairness intervention.

### Label efficiency?

Supported as descriptive: mean gaps are larger in five repeated low-label subset refits. Not established: a statistically reliable widening trend or concentration effect as labels decrease.

### Interpretability?

Supported as caution: attribution is diffuse in the examined fine-tuned probes, and apparent bilateral structure may reflect orientation mixture. Overclaimed: verified OD/OS causation and mechanistic support for the frozen-pretraining arm comparison.

### Bounds of “anatomical precision does not add”?

The strongest defensible statement is that greater measured specificity did not yield monotonic or consistent improvement across the tested arms. Target shape and coverage are not identified because policies differ on area, retained context, loss slots, collation, implementation, guide provenance, and—cover specifically—a defect. The ep30 anatomy-v1 positive result must remain visible.

### Is the title misleading about a segmentation task?

Yes. “Matches a Segmenter” can suggest segmentation accuracy, but no segmentation outcome is evaluated. The segmenter supplies a pretraining mask-placement guide; the downstream task is glaucoma classification.

## Verified strengths

1. **Complete, unusually transparent probe inventory.** Valid, excluded, and retracted probes are all tabulated with reasons (`main_submission.tex:774-800,1304-1318`).
2. **Matched epoch and precision for headline arm contrasts.** The current main table avoids the earlier cross-precision comparison problem (`main_submission.tex:372-383,397-421`).
3. **Correct subject-level pairing in the primary saved-prediction statistics.** Shared cases are used for paired bootstrap deltas and correlated-ROC tests (`main_submission.tex:363-371`; `autopilot\p1c_stats.py:65-115`).
4. **Multiplicity is addressed rather than ignored.** The subgroup AUC and sensitivity families use simultaneous max-\(|t|\) intervals, and the primary nine contrasts receive BH organization (`results\p17_subgroup_multiplicity.json:3-22,143-160`).
5. **Null and adverse results are retained.** Anatomy-v2 ep50 versus random remains unresolved, moderate/severe subgroup intervals cross zero after adjustment, and the defective cover trajectory is not hidden.
6. **The cover defect is disclosed with bounded consequences.** The manuscript explicitly says the arm does not test aggressive coverage and that a corrected arm would require retraining (`main_submission.tex:1323-1368`).
7. **Adaptive test reuse and single-continuation limits are substantially disclosed.** The paper repeatedly states that intervals are descriptive and policy-level rankings are not established (`main_submission.tex:710-747,812-874,1857-1865`).
8. **The specificity ladder uses a zero-centered paired-difference axis.** This is scientifically preferable to a truncated raw-AUC bar chart; the remaining issue is construct naming, not axis deception.
9. **Fairness analysis acknowledges pseudo-replication and incomplete coverage.** Branch aggregation, adjusted paired contrasts, and missing fairness components are stated (`main_submission.tex:1088-1201`).
10. **Clinical deployment is explicitly disclaimed.** The ethics appendix distinguishes retrospective research measurements from clinical utility (`main_submission.tex:1720-1788`).
11. **Independent stored-prediction arithmetic check is complete.** The parent-side snapshot finds all 43 HANDOFF prediction files present and all recomputed AUCs matching the six-decimal record, while correctly preserving excluded/retracted status and declining to infer subject identity from equal label arrays (`autopilot\reports\fresh_audit_2026-09-04\prediction_snapshot.json:1-20`).

## Known limitations that are already properly disclosed

These should not be presented as fresh discoveries, although central claim wording must remain consistent with them:

- One completed pretraining continuation per policy and one probe seed.
- Repeated adaptive inspection of one test split; descriptive rather than confirmatory inference.
- Selective stopping of anatomy-v2 after observing its ep75 deficit; ep100 is unmeasured.
- Cross-family differences in collation, mask ratio, retained context, loss slots, and guide provenance.
- Cover-arm post-placement truncation defect and absent corrected arm.
- No external cohort, clinical validation, deployment claim, or complete fairness evaluation.
- Correlated checkpoints from seven pretraining branches rather than 23 independent models.
- No paired per-cell intervals for the intersectional table.
- Laterality unavailable for validating the inferred orientation mixture.
- Background causal controls were not run.

## What this audit did not reproduce

- This science pass did not itself recompute the saved-NPZ AUC inventory; the independent parent-side snapshot subsequently verified 43/43 files and six-decimal AUC records.
- Did not fit any probe, re-encode any volume, train any model, or resume replication.
- Did not recompute bootstrap, DeLong, BH, or max-\(|t|\) draws; inspected code and retained summaries.
- Did not verify subject identity from case identifiers. The independent snapshot establishes one common label sequence only, not subject-identity pairing.
- Did not render or visually inspect PDFs, PNGs, or GUI figures; reviewed captions and generator encodings only.
- Did not audit `src` training/downstream implementation, as assigned to a separate agent.
- Did not assess citation existence or appropriateness through external services; no manuscript text left the local environment.
- Did not verify remote/cloud replication jobs.

## Minimal revision order

1. Correct the title/equivalence/task wording.
2. Narrow the causal “region/H1/precision” claims to implemented policy comparisons.
3. Correct the replication status and seed accounting.
4. Correct the broad fairness sentence and gender/sex terminology.
5. Downgrade label-efficiency, background-mechanism, operating-point, and interpretability headings/captions.
6. Make “purity,” “anatomy hidden,” and “specificity” distinct throughout figures and prose.
7. Rename the statistics artifact’s `confirmatory` family without changing numeric results.

# Confidential comments to editor

Not applicable to this author-side audit. No confidential editorial channel, editorial recommendation, conflict note, or integrity allegation is being supplied.
