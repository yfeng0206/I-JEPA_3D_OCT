# Inventory of computed-but-unreported results

Scope: `results/` (~390 files), `D:\jepa_phase0\reports\` (read-only), `docs/experiments/`,
`autopilot/reports/`. Method: every `.json`/`.csv`/`.npz` summary was diffed against the
provenance inventory `paper/genai4health2026/research/numbers_master.csv` (46,586 rows,
183 distinct source paths) and against the macro file `paper/.../auto/auto_numbers.tex` (402 macros).

Excluded by instruction: `D:\jepa_phase0\reports\patch_attribution\*` (closed, rejected) and
`results/masking/adapter_*` (owned by another agent). `results/phase0_local_examples/*` excluded
as off-project (ImageNet/Qwen/SAM demos, not OCT).

Counts (MEASURED): 54 artifact files are absent from `numbers_master.csv`. Four of those groups
(`p5_label_efficiency.json`, `p16_subgroup_operating.json`, `region_auc_summary.csv`,
`results/masking/table2_geometry/*`) do reach the paper by another route (LE*/SOP* macros,
Table geom prose), leaving **48 files / 19 distinct result groups genuinely unreported**.

---

## HIGH (3)

| # | Artifact | What it measures | Paper uses it | Rank | One-line judgement |
|---|---|---|---|---|---|
| 1 | `D:\jepa_phase0\reports\subgroup\intersectional_auc.json` + `.csv` (producer: `paper/genai4health2026/scripts/intersectional_analysis.py`, claim reducer `intersectional_claims.py`) | Race x gender AUC with bootstrap CIs for all 22 probes (18 non-retracted), incl. every main-table arm | NO - and the paper twice states the opposite (L1079-1082 "we cannot speak to intersectional groups... no intersectional breakdown"; L1497 "no intersectional breakdown") | HIGH | A control the paper declares absent was in fact run, is unanimous on its ordering claims, and shows the reported marginal race gap understates the worst cell by 60.1 percent. |
| 2 | `docs/experiments/masking/anatomy_vs_rectangle_ep30_seeds.json` (5 probe seeds x 2 arms) | Probe-seed dispersion of test AUC on two fixed encoders | NO - carried in `numbers_master` as 8 rows flagged DOC-ONLY; paper L665-667 says the multi-seed probe check "is not reproducible from retained artifacts, so this paper states no bound on probe noise", and L1669 records the bound as deliberately deleted | HIGH | The raw per-seed values plus both fp32 feature caches survive, so the deleted probe-noise bound is likely recoverable on CPU - which would remove a self-inflicted "we cannot bound this" from the limitations. |
| 3 | `results/downstream/frozen_random_crossattn/ep100_results.json` + `results/downstream/linear_sweep_random_posfix_d1/{ep25,ep50,ep75,ep100}_results.json`, `summary.csv` | Same frozen RANDOM ep100 encoder read out by three probe heads | NO (only the fine-tuned head comparison at L1250 and App. table are reported; the frozen head sweep is not) | HIGH | Head choice alone moves the null arm by about +0.0045 / -0.0040 AUC, the same magnitude as the policy effects the paper reports, which complicates rather than strengthens the headline. |

### Skeptical audit of the HIGH items

**1. Intersectional fairness.** MEASURED, recomputed by me from the artifact:
`sweep_oracle_ep100` minus `sweep_random_ep100`, per cell:
asian x female +0.0144, asian x male +0.0164, black x female +0.0114, black x male +0.0228,
white x female +0.0121, white x male +0.0099 (n = 123, 128, 250, 181, 1343, 975).
Reducer output (MEASURED, 18 non-retracted arms): black x female is the worst cell 18/18;
asian x male best 18/18; female below male within race 54/54; black below white within gender 36/36;
mean gap gender 0.0340, race 0.0653, race x gender 0.1046; intersectional exceeds marginal race 18/18;
worst-cell understatement 60.1 percent; additive prediction 0.0993 vs observed 0.1046 (ratio 1.053).
*Wrong if:* (a) the per-cell arm deltas were not robust across epochs - CHECKED, and they are NOT:
at ep50 `oracle - random` is -0.0020 for asian x female, and `envelope - random` is negative in
3 of 6 cells at both ep75 and ep100. So the "every group improves" sentence does NOT extend to
intersections and must not be written that way. (b) the deltas carried intervals - CHECKED, they do
not; the JSON stores per-arm CIs only, no paired per-cell bootstrap. Per-cell delta significance is
therefore **PENDING** (computable on CPU from the retained `test_predictions.npz` per arm).
(c) unanimity being 18 independent tests - it is not; the 18 probes share one test split and one
encoder lineage, so 18/18 is a consistency check, not 18 replications.
Net: the *descriptive* intersectional table and the marginal-understates-intersectional
result are MEASURED and appendix-ready; any causal or per-cell-significance wording is PENDING.

**2. Probe-seed bound.** MEASURED values in the file: anatomy 0.8583/0.8580/0.8583/0.8578/0.8586
(mean 0.8582, sd 0.0003); envelope 0.8540/0.8530/0.8533/0.8497/0.8542 (mean 0.8528, sd 0.0018);
delta +0.0054, Welch p 0.00219, Cohen d 4.20 (all as recorded in
`docs/experiments/masking/anatomy_vs_rectangle_ep30.md`, which itself labels them technical, not
experimental, replicates).
*Wrong if:* (a) the values cannot be regenerated - PARTLY TRUE and this is the real risk. Git shows
the file entered in commit `a3e04f0` alongside the ep30 comparison, and **no script in the repo
references it** (`git grep` returns nothing), so its producer is a hand-run. (b) the caches are gone -
CHECKED, they are not: `D:\jepa_phase0\runs\frozen_meanpool_{anatomy,envelope}_ep30\feature_cache\`
each hold Training/Validation/Test fp32 tensors. BUT the cache key format is the old
`Test_s100_fp32.pt`, whereas current runs use `Test_s100_r256_fp32_<hash>.pt`, so today's eval path
may not accept them without a shim. Recompute status: **PENDING**.
(c) It bounds the wrong variance component - TRUE and it must be said: this is probe noise on two
*fixed* encoders (anatomy-v1 ep30, envelope ep30), not pretraining-seed noise, and neither arm is a
main-table row at a main-table epoch. It repairs the "no bound on probe noise" sentence; it does
**not** touch n=1.

**3. Frozen probe-head sensitivity.** MEASURED: RANDOM ep100 frozen, mean-pool 0.8746 (paper);
cross-attention pool 0.8791; depth-1 linear 0.8706 (with ep25 0.8558, ep50 0.8611, ep75 0.8691).
Spread across heads about 0.0085 on a single arm, against reported
envelope-random +0.0062 and oracle-random +0.0109.
*Wrong if:* (a) the heads are not like-for-like - PARTLY TRUE: the cross-attention probe carries
276,672 probe params vs a mean-pool linear head, so a higher AUC is partly capacity, not readout
robustness. (b) the runs are fp16-era - TRUE, the crossattn config predates the amp fix; the paper's
own fp16-vs-fp32 table shows differences of about 0.0001, so this is immaterial but should be stated.
(c) it reorders the arms - UNKNOWN and probably not demonstrable: only RANDOM has all three frozen
heads, so no ordering claim can be made. Significance of the head spread: **PENDING** (paired
bootstrap computable from the retained `*_test_predictions.npz`).

---

## Bearing on the n=1 weakness

**Nothing closes it.** The replication named in the paper is genuinely still running, not forgotten:
`D:\jepa_phase0\runs\rep_random_s1234\` contains only pretraining checkpoints and logs written
today (last write 2026-08-26 18:13), no downstream probe, and no second or third seed directory
exists for any arm. Producers `scripts/chain_replication.py`, `scripts/make_replication_configs.py`
and `scripts/smoke_replication.py` are present and consistent with a live chain. Status: PENDING,
correctly described by the paper.
The only multi-seed evidence anywhere in the project is HIGH item 2, and it is *probe* seeds on
fixed encoders - it bounds readout noise, not pretraining noise. Per the brief's rule it is at best
MEDIUM as evidence about n=1; it is ranked HIGH only because it repairs an explicit
"we can state no bound" sentence at CPU cost.

---

## MEDIUM (6)

| # | Artifact | What it measures | Paper uses it | Rank | One-line judgement |
|---|---|---|---|---|---|
| 4 | `D:\jepa_phase0\reports\subgroup\subgroup_auc.json` + `.csv` | Marginal gender/race AUC with CIs for all 22 probes, incl. arms absent from the fairness figure (cover f021 ep27/30/34, bridge ep35/40/50, anatomy ep30, fork ep25) | Partly - the reported fairness figure covers a subset of probes; these files are not in `numbers_master` at all | MEDIUM | Same-format fairness numbers for every arm the paper omits from Figure fairness; cheap appendix completion, no new claim. |
| 5 | `results/masking/table2_geometry/mask_geometry_600slices_bs1_coverf021_seed{42,1234,2026}.json` and `bs64_coverf015_seed{42,1234}.json` | Mask-geometry statistics under batch-size 1 and under a different cover floor, three seeds each | Only the bs64 / f=0.21 / 3-seed set is described in prose (L862-881); these variants are not | MEDIUM | Shows the geometry table is stable to collation batch size and floor setting - a robustness footnote for the "geometry does not explain the ordering" claim. |
| 6 | `D:\jepa_phase0\reports\downstream_region_auc\{blob,envelope,oracle,random}_ep50\region_auc.json` | Per-arm anatomy-only vs background-only region AUC at ep50 | Only the merged `region_auc_summary.json` is in `numbers_master`; the per-arm files are not | MEDIUM | Per-arm decomposition behind the "background barely reaches the classifier" claim; useful as an appendix table, but it is the same evidence family that failed under scrutiny in ATTRIBUTION_ANALYSIS.md, so treat with suspicion. |
| 7 | `D:\jepa_phase0\reports\loss_curves_full.csv` (+ `.xlsx`, producer `scripts/collect_loss_curves.py`) | Pretraining train/val loss trajectories per arm, 125 rows | NO | MEDIUM | Would let the appendix show the arms trained comparably (a reviewer's first "did you just undertrain the baseline" question), but the `fidelity` column says `sampled` and the `source` column points back to markdown, so it is transcription, not raw log - verify against `results/pretraining/.../-log.csv` before using. |
| 8 | `D:\jepa_phase0\reports\cover_audit_mv{0.1,0.15,0.2}.json`, `cover_coverage_audit.json`, `cover_mask_audit.json`, `budget_mask_audit.json` | Mask statistics at three visible-tissue floors | NO | MEDIUM | The closest thing to the dose-response over the cover floor that the paper says it cannot report - but it is mask geometry only, with no pretraining or downstream leg, so it cannot fill that gap, only describe it. |
| 9 | `results/masking/split_fix/split_fix.json`, `results/masking/collation/collation_union.json` | Split-leak fix verification and union-collation audit | NO | MEDIUM | Integrity controls that would answer reviewer questions about the split and the collation defect; no new scientific claim. |

## LOW (10 groups)

| Artifact | What it measures | Rank | Judgement |
|---|---|---|---|
| `results/downstream/**/ *_train_log.csv` (9 files) | Per-epoch probe training logs | LOW | Raw backing for reported AUCs; nothing new. |
| `results/pretraining/pretrain_mirage_envelope/jepa_patch_mirage-log.csv` | Raw envelope pretraining log | LOW | Raw backing; relevant only as the check on item 7. |
| `D:\jepa_phase0\reports\arm_coverage\{per_image.csv,summary.csv}`, `five_arm_audit\per_image_rows.csv` | Per-image rows behind summaries already in `numbers_master` | LOW | Raw backing. |
| `D:\jepa_phase0\reports\background_signal_smoke\background_signal.json`, `sweep_smoke\cover_floor_sweep.json`, `results/masking/placement/smoke.json` | Smoke tests | LOW | Superseded by full runs already reported. |
| `results/masking/placement/{enc_saved,placement_a050}.json` | Adapter placement variants | LOW | Adjacent to the adapter sweep another agent owns; out of scope here. |
| `results/masking/diagnostics/worst_idx.json` | Indices of worst slices | LOW | Debug artifact. |
| `results/downstream/ARTIFACT_MAP.json` | Index of artifacts | LOW | Metadata. |
| `D:\jepa_phase0\reports\patch_attribution\attribution_summary.csv` | Rolled-up patch attribution | LOW / CLOSED | Same lead already rejected in ATTRIBUTION_ANALYSIS.md. |
| `D:\jepa_phase0\reports\downstream_region_auc\region_auc_summary.csv` | CSV twin of the reported JSON | LOW | Duplicate of a reported artifact. |
| `results/p5_label_efficiency.json`, `results/p16_subgroup_operating.json` | Label-efficiency curve, subgroup operating point | REPORTED | Absent from `numbers_master` but their values are the LE* and SOP* macros; not unreported. |

## Controls the paper names as not run - checked one by one

| Control named in the paper | Line | Was it actually run | Evidence |
|---|---|---|---|
| Intersectional fairness breakdown | 1079-1082, 1497 | YES - contradicts the paper | `D:\jepa_phase0\reports\subgroup\intersectional_auc.json`, 22 probes (MEASURED) |
| Probe-seed noise bound | 665-667, 1669 | YES, values retained; recompute PENDING | `docs/experiments/masking/anatomy_vs_rectangle_ep30_seeds.json` |
| Three-seed replication of the ordering | 659-676 | NO - in progress, correctly PENDING | `D:\jepa_phase0\runs\rep_random_s1234\` only, no probe, no other seed |
| Band-position randomisation arm (H3 mechanism) | 655-657 | NO | No matching run dir or result file |
| Shape-isolating anatomy-v2 at K approx 40 | 945-956 | NO | No matching artifact |
| Two background-mechanism controls | 529 | NO | `background_signal*` covers the descriptive result only |
| Cover floor dose-response (only f=0.21 pretrained) | 680-686 | NO pretraining; mask-stat sweeps only | `cover_audit_mv*.json`, `arm_stats_sweep\cover_floor_sweep.json` |
| Subgroup calibration | 1497 | NO | Only aggregate Brier/ECE macros exist |

## Recommended order of work

1. Item 1: add an intersectional appendix table plus the 60.1 percent understatement statistic, and
   fix the two sentences that say it does not exist. Compute paired per-cell deltas with bootstrap CIs
   from the retained per-arm `test_predictions.npz` before claiming any per-cell improvement, and do
   not say "every group improves" - it is false at ep50 and for envelope.
2. Item 2: attempt the CPU recompute of the five probe seeds from the retained fp32 caches. If the
   old cache key blocks it, leave the limitation sentence as written.
3. Item 3: state the frozen head-sensitivity range for RANDOM as a caveat, not a result.
4. Items 4-9 as appendix filler if space in the (unconstrained) appendix is being used anyway.

Labels used: MEASURED = read or recomputed from an artifact during this audit;
INFERRED = arithmetic on measured values; PENDING = not computable from retained artifacts today.
