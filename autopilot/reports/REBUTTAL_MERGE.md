# REBUTTAL merge record

Applied 2026-08-26 to `paper\genai4health2026\main_submission.tex`. Source of instructions:
`autopilot\reports\REBUTTAL.md` (26 items) plus the merge brief. Nothing was committed.

## Gate results after the merge

| gate | result |
|---|---|
| `autopilot\p13_build_zip.py` | all 6 checks PASS, total 32 pages, references start page 10 (heading at page top), **main content 9 pages** (limit 9), `ALL_PASS = True` |
| `autopilot\check_manuscript.py` | **RESULT: PASS** (0 hard failures; 3 pre-existing warnings, see below) |
| `autopilot\p15_verify_numbers.py` | **RESULT: PASS** (20 AUC macros verified, no cross-arm attribution) |

Pre-existing warnings, unchanged by this merge: 212 unused generated macros; the two
hand-typed re-encoding literals `0.8854754` / `0.8854852` in Section 5.5; and the literal
"92 probes" which is the substring of "epoch-75 and epoch-92 probes" in Section 4.

Body-line ledger: the four CUTs removed more than the three body ADDs added. The first
build after the ADDs overflowed to 10 pages; the Section 6 replication text and the
Section 5.4 sentence were then compressed until the body returned to exactly 9 pages,
with the References heading again at the top of page 10 (y = 72.79, identical to the
pre-merge build).

---

## CUTs applied (unsupported claims removed)

### C1. Section 5.2, the "Third, \ArmBest{} is the most spatially consistent policy" paragraph (R12)
Deleted in full, including the Gini values `0.400` and `0.316`--`0.338` and the
"consistency of the predictive task" mechanism suggestion. The lead-in was changed from
"Three observations follow" to "Two observations follow", and a pointer to the new
provenance appendix was folded into the existing sentence.
Basis: `autopilot\reports\HARDCODED_AUDIT.md` line 519 marks the Gini values UNBACKED; the
only stored Gini artifact, `results\masking\coverage\coverage.json`, has no CENTROID entry
at all and gives `random_default` 0.4111 and `envelope_default` 0.4038, i.e. it points the
other way. No number from that file was imported into the paper.

### C2. Conclusion, "We suggest the operative variable is the consistency of the predictive task..." (R12)
Deleted. Because it was the paper's closing interpretive sentence, it was replaced (not
merely removed) by a supported closing statement: "What makes the best policy work is not
identified here: this design cannot separate consistent target placement from mask ratio
and task difficulty." That restates the already-present Section 6 statement that two
explanations remain live, so it introduces no new claim and the Conclusion still ends on
an interpretive note rather than stopping mid-argument.

### C3. Section 5.3, the 1 percent label-fraction sentence (R7)
Deleted the sentence using `\LEOracleOne` against `\LERandomOne` with the
`\LESDRandomOne`--`\LESDCoverOne` repeat-spread hedge. The following sentence, "Whether the
gap genuinely widens as labels are withdrawn we did not test", is kept verbatim. The same
1 percent numbers remain in Appendix I (label efficiency), where the standard deviations
are also printed, so no measured value was lost from the paper.

### C4. Section 6, the five-probe-seed standard deviations (R13)
Deleted "with five probe seeds on two fixed encoders the probe-seed standard deviation is
0.0003 and 0.0018 ... so probe noise is three to seven times below the effects in
Table 1". Replaced with an explicit statement of non-quantification: "Probe-seed variance
is not quantified here either: an earlier multi-seed probe check is not reproducible from
retained artifacts, so this paper states no bound on probe noise."
Basis: `HARDCODED_AUDIT.md` line 656 marks these UNBACKED, and
`paper\genai4health2026\research\verify_sections_4_6.md` findings C25/C27 record that only
two of the six per-seed `results.json` files still exist. This removes a bound the Area
Chair cited in our favour; it was removed because it is not reproducible.

---

## ADDs applied to the body

### A1. Section 6, replication paragraph, PENDING (R1)
The "One pretraining run per policy" paragraph is now "One pretraining run per policy, and
a replication in progress". It states the design only: six continuations (RANDOM,
ENVELOPE, CENTROID at two further seeds) from the same epoch-25 ancestor, SHA-256 verified
before each leg, configurations asserted to differ only in the seed, each run to epoch 50,
giving three continuations per policy at a matched endpoint; the fixed-data-order caveat;
and "every result of it is PENDING". No replication result, partial result or expected
direction is stated anywhere in the paper. The closing clause is symmetric by design: "an
ordering that does not reproduce establishes that Table 1 describes these runs rather than
the policies", i.e. a null reads as the measurement the design was built to deliver.
Artifacts: ancestor hash and byte size verified by me directly with `Get-FileHash` on
`D:\jepa_phase0\checkpoints_hf\random-posfix-100ep\jepa_patch-ep025.pth.tar` (SHA-256
`E5AD5B0C2AADFA15449409786AFBFA39D8B5405B699BE8F02F2E540195E97E7B`, 1,507,519,602 bytes);
leg list, seeds and endpoint from `autopilot\reports\G1_REPLICATION.md` sections 2-4 and
the queue order in section 8.

### A2. Section 5.4, subgroup specificity at the deployed threshold (R2)
Added one sentence: "the validation-selected threshold aimed at specificity 0.90 realises
0.895 for white and 0.736 for black patients under the null, and 0.879 and 0.761 under
CENTROID, so the smaller stratum absorbs a higher false-positive rate under *both* arms".
Presented as a property of threshold transfer, not of masking policy.
Artifact, read by me from `results\p16_subgroup_operating.json` (threshold 0.55517578125,
selected on the RANDOM validation split, epoch 100):
`arms.random.groups.race.White.specificity` = 0.8949919224555735,
`arms.random.groups.race.Black.specificity` = 0.7361963190184049,
`arms.intensity.groups.race.White.specificity` = 0.8788368336025848,
`arms.intensity.groups.race.Black.specificity` = 0.7607361963190185. The `intensity` arm
key is CENTROID; the RANDOM pair is reported under the null and the CENTROID pair under
CENTROID, never mixed.

### A3. Section 4 "Data", model-selection narrowing (R15)
"No test volume is seen during pretraining, probe fitting, or model selection" became "No
test volume is used to fit or select the probe head, which is selected on a separate
validation split; the study's choices of policy, checkpoint, analysis and stopping horizon
were made after repeated inspection of that same test split (Section 6)." This removes the
direct contradiction with the Section 6 adaptive-reuse disclosure. Net length neutral.

---

## ADDs applied to the appendix (unconstrained space)

### B1. New Appendix B, "Continuation-level replication" (R1, R16, R17)
Status line "PENDING. No result of this replication is reported anywhere in this paper",
then: the design (ancestor SHA-256 in full, 1,507,519,602 bytes, re-hash before each leg,
generated configs asserted identical outside the mask curriculum except seed/logging,
epoch-50 endpoint with schedules still sized for 100 epochs, encoder hashed around each
probe); the four pre-committed analysis rules copied from REBUTTAL R1 (all nine AUCs
reported, per-policy mean and range, paired per-seed differences, no test or interval at
n=3); what the seed does and does not randomise, including that data visitation order is
not randomised; the failure case written in advance; and "What this replication does not
do", which states that no untouched cohort was available (R16), what an untouched
evaluation would require, and that the number of test-split inspections was not logged and
cannot be reconstructed (R17).
Numbers used: hash and byte size verified directly (see A1); no result value appears.

### B2. New Appendix D, "Mask-geometry provenance" (R3, R11, R4)
Contains:
- how the 600-slice measurement was made (24 volumes x 25 slices, one image at a time,
  COVER floor f = 0.21, 16x16 grid, MIRAGE guide occupancy threshold 0.25, seeds 42/1234/2026),
  and that Training slices are forced because the guide cache has a Training split only;
- cell-level agreement: one cell exact, 21 preserve the printed ordering, 3 do not, 18 of
  25 inside the three-seed range, largest absolute difference 2.21 points, and the
  statement that the loss-slot ordering carries no information;
- Table 4: printed vs regenerated vs three-seed sd for anatomy hidden and loss slots;
- Table 5: visible context per image vs delivered at the production batch size of 64,
  with the note that the anatomy/rectangle context confound is roughly a doubling rather
  than the 1.6x ratio the per-image column implies;
- the COVER-floor sensitivity: at floor 0.15 COVER hides 79.5 percent of anatomy instead
  of 73.4 percent at the same batch size, overtakes ENVELOPE, and the anatomy-hidden/AUC
  Spearman moves from +0.80 to +0.40;
- "What would identify target shape" (R4): pred_target_k about 40 to match 158-160 loss
  slots at four targets, delivered mask ratio 40-46 percent instead of 21 percent,
  rectangle collation path, same replicated continuation design and endpoint. The
  K-about-40 figure is labelled inferred arithmetic.
Artifacts, all read by me:
`results\masking\table2_geometry\mask_geometry_600slices_bs1_coverf021_seed{42,1234,2026}.json`
(per-arm `hidden_share_of_all_anat`, `n_slots_mean`, `ctx_frac_of_grid` and the sd across
the three seeds), `...bs64_coverf021_seed42.json` and `...bs64_coverf015_seed42.json`
(delivered context and the floor-0.15 COVER value 79.5428), the meta blocks of those files
(600 slices, 24 volumes, floor, threshold, grid), `configs\patch_anatomy_v2.yaml`
(`num_pred_masks: 4`, `pred_target_k: 16`), and `C:\jepa_data\mirage_soft_guides\
base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy\cache_meta.json` (`"split": "Training"`,
Training directory only). Verdict counts (1 MATCHES / 21 CLOSE / 3 DIFFERS, 18/25 inside
the seed range, max difference 2.21) were reproduced by running
`autopilot\compare_table2_geometry.py`. Both Spearman coefficients were recomputed by me
with `scipy.stats.spearmanr` on the measured anatomy-hidden values and the epoch-50 AUC
macros: +0.80 (p = 0.20) at floor 0.21 and +0.40 (p = 0.60) at floor 0.15, at matched
batch size.

### B3. New Appendix P, "Reproducibility and numeric provenance" (R10, R11)
Describes the three build gates, the checkpoint-identity guarantees (ancestor SHA-256,
per-probe encoder hashing, asserted config invariants), and the hand-typed-number audit:
310 numeric occurrences typed into the source, 234 confirmed, 75 with no locatable
producing artifact, one wrong and corrected. It also records that the two unbacked
quantities carrying interpretive weight - the per-cell concentration statistic and the
probe-seed noise bound - were removed in this version rather than restated, and states the
rule adopted.
Artifact: `autopilot\reports\HARDCODED_AUDIT.md` summary lines 8-11 (310 / 234 / 75 / 1)
and line 31 (the 7.17M vs 7.14M probe parameter count).

### B4. Appendix A opening, probe-count reconciliation (R5)
Added: the table has 37 rows, being `\Nprobes` valid probes plus two excluded and four
retracted; `\NprobesSub` carry the metadata used by the subgroup audit and `\NprobesRace`
a usable race summary. Only existing macros are used, no new literal count except "37
rows".
Artifacts: `D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json` (43 records:
31 primary, 6 supplementary, 4 retracted, 2 excluded, so the printed table's 37 rows =
31 + 4 + 2), `paper\genai4health2026\auto\table_allprobes.tex` (38 row terminators, one of
which is the header), `p7_fairness.json` `n_probes_with_race_summary` = 19 =
`\NprobesRace`, `p7b_gap_trend.json` `n_probes` = 23 = `\NprobesSub`.

### B5. Appendix K (operating points), two new paragraphs (R2)
"The threshold itself does not transfer evenly across strata": the 0.895 / 0.736 pair under
RANDOM and the 0.879 / 0.761 pair under CENTROID, with the shortfall named as a property of
threshold transfer present under both arms.
"The sensitivity gain is not a uniform trade": white +0.034 sensitivity against -0.016
specificity; black +0.025 and asian +0.008 specificity alongside sensitivity changes
(+0.004, +0.008) whose intervals contain zero; female +0.026 and male +0.027 sensitivity
against -0.008 and -0.013 specificity; subgroup calibration improving overall
(`\ECERandom` to `\ECEIntensity`) and in the white (0.0382 to 0.0321) and asian (0.0832 to
0.0729) strata but worsening in the black stratum (0.0525 to 0.0569). The existing
shared-threshold-not-shared-FPR limitation is restated, not weakened.
Artifact: every value read by me from `results\p16_subgroup_operating.json`
(`delta_random_to_intensity.race.*.d_sensitivity` / `.d_specificity`,
`delta_random_to_intensity.sex.*`, and the per-arm `ece` fields under `arms.random.*` and
`arms.intensity.*`). The two overall ECE values are quoted through the existing generated
macros rather than as literals.

### B6. Appendix N (full paired-contrast table), multiplicity clause (R9)
Added: Benjamini-Hochberg is applied within a declared confirmatory family of exactly these
nine contrasts, with the 25 exploratory contrasts corrected as a separate family, and the
two families never pooled.
Artifact: `D:\jepa_phase0\autopilot_out\p1_stats\p1c_stats.json`, `multiplicity`:
`confirmatory_family_size` = 9, `exploratory_family_size` = 25, with the stored note.

### B7. Appendix G (COVER collation defect), provenance caveat (R14)
Added a "Provenance" clause stating that the pre/post-truncation coverage figures and the
retained-target statistics in that paragraph come from a one-off CPU audit of 194 accepted
slices whose raw output was not persisted, and that the 73.1 / 77.6 comparison the body
relies on is backed by the stored floor sweep. No number was changed.
Artifact: `autopilot\COVER_AUDIT.md` lines 31-36 ("fresh CPU audit, 194 accepted slices"),
and `HARDCODED_AUDIT.md` lines 915-919 marking those four values UNBACKED.

### B8. Appendix H (why unguided masking is strong), the two unrun controls (R25)
Added a paragraph naming both settling controls - an eroded-background regional probe, and
a background-content shuffle with positions and token count fixed - and marking both
PENDING and not run.

### B9. Appendix J (occlusion attribution), laterality wording and provenance (R24)
"The signature is OD/OS axial storage" became "What these three tests support is an
orientation or storage mixture consistent with OD/OS", with an added sentence that the
released metadata carries no eye-laterality label so the reading is inferred and not
validated. The appendix opening now states that its numbers come from archived per-volume
attribution arrays outside the released artifact set and cannot be recomputed from the
release.

### B10. Appendix L (broader impact and ethics), consistency repair
The ethics appendix said the subgroup audit "contains no subgroup calibration". Since B5
adds subgroup calibration, that clause was corrected to "beyond AUC, the
transferred-threshold sensitivity, specificity and calibration of Appendix K, it contains
no predictive-value analysis and no intersectional breakdown". This is a contradiction
repair forced by B5, not a weakening: the remaining caveats are untouched.

---

## Non-paper action taken

### R26. The promised DeLong validation artifact now exists
`SOURCES.md` names `autopilot/p1_validate_delong.py`, and REBUTTAL R26 records that its
output `delong_validation.json` was absent. I ran the script; it wrote
`D:\jepa_phase0\autopilot_out\p1_stats\delong_validation.json` (1,680 bytes) with
`ALL_PASS = true` (AUC agreement, variance agreement, null calibration with empirical
type-I error 0.0417, self-comparison). The promise in the source list is now satisfiable.

---

## Items in REBUTTAL.md NOT applied, and why

| item | target | why not applied |
|---|---|---|
| R6 (move the direct H2 contrasts into Table 1) | body, Table 1 | Body-targeted and outside the brief's ADD list. The body has zero slack: the References heading currently sits exactly at the top of page 10, so a table block plus prose deletion is a page-layout gamble that the brief's priority order does not authorise. The contrasts remain printed in Section 5.1 prose, unchanged. |
| R7 (per-repeat paired label-efficiency differences in Appendix I) | appendix | Requires first re-running `autopilot\p5_label_efficiency.py` with per-repeat AUCs persisted; the current artifact `results\p5_label_efficiency.json` stores only per-arm mean, SD and `n_repeats`, so the paired quantity does not exist yet. REBUTTAL itself says the appendix change stays PENDING until that rerun. Writing it now would mean inventing numbers. |
| R8 (severity difference-of-differences) | appendix | The mild-minus-moderate and mild-minus-severe bootstrap intervals are not stored anywhere. REBUTTAL's own instruction is that if they are not computed before freeze, leave the existing "we did not test" sentence exactly as it is and add nothing. Done: nothing added, sentence untouched. |
| R17 (four words in the Section 6 body) | body | The substance was added, in Appendix B instead of the body, for the page-budget reason above: the inspection count was not logged and cannot be reconstructed is now stated in "What this replication does not do". |
| R16 (add the no-untouched-cohort clause inside the Section 6 body sentence) | body | Same treatment: stated in Appendix B rather than the body. The existing Section 6 adaptive-reuse sentence is untouched and not softened. |
| R20 (replace containment verbs for ENVELOPE) | body | Net-neutral in principle but not in practice: the recommended replacements ("rectangle centres rejection-sampled onto the predicted retinal envelope") are longer than the phrases they replace, and the body has zero slack. Section 5.1 already concedes in print that only 43.5 percent of ENVELOPE's masked cells land on tissue, so the claim is disclosed. Not in the brief's ADD list. |
| R21 (fixed phrasing for COVER) | body | Same reason. The paper already states that the arm never realised its configured coverage and that its trajectory is not evidence about aggressive coverage; Appendix G now also carries the provenance caveat (B7). |
| R23 (reorder the contributions list) | body | Body-targeted reordering, outside the brief's ADD list, and any reflow of that list risks the 9-page limit for no factual gain. |
| R2, table-column half (add `d_spec` / `d_ece` columns to Table 11) | appendix | Table 11 is generated by `autopilot\p16_subgroup_operating.py` into `auto\table_subgroup_operating.tex`. Editing generated output by hand is forbidden by the repository's own rule, and regenerating it would change an artifact rather than the paper. All the quantities those columns would carry are instead reported in prose in B5, from the same JSON. |
| R12 optional (extend `scripts\coverage_probe.py` to all five arms) | analysis | Not run. It would produce a new measurement, not a paper edit; the claim it would support has been deleted, which is the safe state. |
| R18, R19, R22 | body | Explicitly "no change" items in REBUTTAL: the relevant disclosures are already present and correct, and were left untouched. |
| R24 (move the laterality and confidence-invariance material out of Appendix J) | appendix | Only the wording fix and the provenance statement were applied. Removing the material would delete cautionary content, which the brief forbids beyond the four named CUTs. |

## Rules honoured

- No digit or numeric value already in the paper was altered. Every number added was read
  by me from the artifact named beside it, and two (the Spearman pair, the ancestor hash)
  were independently recomputed rather than copied from REBUTTAL.md.
- No limitation or caveat was removed other than the four CUTs, all of which are removals
  of unsupported claims.
- No arm's numbers appear under another arm's name; `p15_verify_numbers.py` enforces this
  and passes.
- No emoji or tick/cross symbols. Added claims are labelled measured, inferred or PENDING
  where natural.
- Nothing was committed.

