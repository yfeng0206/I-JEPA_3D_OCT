# AC required changes, applied

Source: `autopilot/reports/AC_FINAL.md`, section 5, "Weaknesses fixable by writing", items 1-3.
File edited: `paper/genai4health2026/main_submission.tex`. Nothing committed.

---

## 1. Title and causal wording

### Title applied

**Segmentation-Free Anatomy Guidance Matches Segmenter-Driven Target Placement for
Masked Predictive Pretraining on Retinal OCT**

It asserts two things only: (a) the segmentation-free policy is not worse than the
segmenter-driven one, and (b) what the study varied is target placement. It asserts
nothing about how much anatomy a mask covers, nothing about target shape, and no
mechanism.

### The sentence of the paper that supports it

Section 5.1, last paragraph, unchanged by this pass:

> **The strongest policy consults no segmentation model.** CENTROID finds its band from a
> per-column intensity centroid: no model and no annotation. Its margin over the null is the
> largest at epoch 100 and does not decay with training, whereas ENVELOPE's does
> (Table 1), and it exceeds the segmenter-guided ENVELOPE arm at epoch 100 by +0.0047
> (CI [+0.0004, +0.0091]), though their paired intervals span zero at epochs 50 and 75.

Checked sentence by sentence against the two objects the AC named:

- **Table 2 (`tab:geom`)**, whose AUC column is matched epoch 50: ENVELOPE 0.8761 sits
  *above* CENTROID 0.8740. A title saying the band *beats* the segmenter would contradict
  that column. "Matches" does not: the paired epoch-50 contrast is -0.0020,
  CI [-0.0069, +0.0029], and epoch 75 is +0.0033, CI [-0.0013, +0.0079]; only epoch 100
  excludes zero. Matches at 50 and 75, exceeds at 100.
- **Section 5.2**: "**H3 is therefore not identified by this design**, and we do *not* claim
  that irregular target shape is harmful." The title now makes no coverage or shape claim,
  so nothing in the title is disavowed three pages later.
- Target placement as the varied axis is stated in Section 3 ("the design choice we vary is
  which of the 256 cells become targets") and Section 3.2 ("The rectangle arms differ only in
  placement").

### Titles rejected

| Title | Why rejected |
|---|---|
| "Where to Aim, Not How Much to Cover: Segmentation-Free Anatomy Guidance for Masked Predictive Pretraining on Retinal OCT" (the title being replaced) | "Not how much to cover" asserts coverage is inert. The COVER arm is defective - the collator truncates predictor targets *after* placement, so it never delivered the configured coverage (Appendix G) - and the anatomy arms differ from the rectangle family in mask ratio (21.3% vs 40-47%), retained context (67.7% vs 40-46%), loss slots (64 vs ~159), collation and guide provenance. Section 5.2 says H3 is not identified. The paper disavowed its own title. |
| "What Should a Joint-Embedding Predictor Predict? Target Composition, Context Loss, and Equity in Medical Image SSL" (the title before that, commit b9b2130) | "Context loss" names an axis the paper explicitly cannot separate ("the arms that differ in retained context also differ in mask ratio and loss slots"), and "Equity" overstates a subgroup audit that Section 5.5 and Appendix L both say is not a fairness intervention. |
| "Where to Aim: Segmentation-Free Anatomy Guidance Matches Segmenter-Driven Target Placement ..." | Supported, but typeset to four title lines instead of three (first line 470pt against a 396pt text block), which cost 20pt on page 1 and pushed the body over nine pages. Dropped for layout, not for content. |
| "Segmentation-Free Guidance Beats Segmenter-Driven Masking ..." | Contradicts Table 2: ENVELOPE leads CENTROID at the matched epoch 50, and two of the three paired intervals span zero. |
| "Anatomy Guidance Improves Masked Pretraining on Retinal OCT" | A policy-expectation claim from one continuation per arm. The paper is scoped to "these runs rather than an expected ranking over retrainings". |

### "Identical in every other respect" / "the only moving variable"

Every instance found and replaced with what was actually held fixed - ancestor checkpoint,
optimiser, schedule, effective batch size, probe protocol - plus what the policy also moves.

| Location | Before | After |
|---|---|---|
| Section 3 opener | "of increasing anatomical specificity, identical in every other respect." | "... sharing one ancestor checkpoint, optimiser, schedule, effective batch size and probe protocol." |
| Section 6, single-run paragraph | "... fixed across arms, leaving masking policy the only moving variable" | "... fixed across arms; the mask sampler is the intended difference, though it also moves delivered mask ratio, retained context and loss slots, and across families collation and guide provenance (Section 5.2)." |
| Introduction | "and hold everything else fixed" | "All arms continue from one shared pretrained checkpoint under one optimiser, schedule, effective batch size and frozen-probe protocol; the mask geometry each policy delivers is measured, not assumed equal." |
| Abstract opener | "We test that intuition under tightly controlled conditions." | "We test that intuition from one shared ancestor checkpoint under a matched schedule and probe protocol." |
| Contribution bullet 1 | "Masking policy is isolated in medical SSL by ..." | "A controlled six-arm study of masking policy in medical SSL, with ..." |

Coverage wording narrowed in the four other places the AC named, with no result or caveat removed:

- Abstract close: "anatomy says where to aim, not how much to cover" -> "how much anatomy a
  policy hides is confounded here with mask ratio and retained context, so this design does
  not identify it".
- Section 5 opener: "Region matters and coverage does not." -> "Region matters; how much
  anatomy a policy covers is not identified here."
- Section 5.2 heading: "Aim, not coverage" -> "Aim, and the coverage this design cannot
  isolate"; its lead sentence "Anatomy tells you where to aim, not how much to cover" ->
  "The highest epoch-100 endpoint hides the least anatomy" (which is what Tables 1 and 2
  jointly show; the old lead also mis-scoped "best performer", since ENVELOPE leads at the
  epoch-50 column of Table 2).
- Discussion: "Anatomical *coverage* does not:" -> "... whether *coverage* caused that is not
  separable here, any more than whether what the mask *leaves visible* matters".
- Conclusion: "anatomy mattered for *where* ... and not for *how much*" -> "... ; *how much*
  anatomy they cover is not identified by this design", and the Conclusion now states the
  title's claim explicitly ("matching the segmenter-guided arm at epochs 50 and 75 and
  exceeding it at 100"). Abstract opening and Conclusion both checked against the new title:
  neither contradicts it.

Every COVER and ANATOMY-V2 negative result, the epoch-75 deficit, the selective stopping
horizon, the collation defect and the adaptive-test-reuse limitation are untouched.

---

## 2. The fairness scope contradiction

**Verified: Appendices E and L were wrong, Appendix K was right.**

Appendix K, "Clinical operating points and calibration", reports race-stratified expected
calibration error explicitly: ECE improves in the white (0.0382 -> 0.0321) and asian
(0.0832 -> 0.0729) strata and *worsens* in the black stratum (0.0525 -> 0.0569). Those are
subgroup calibration numbers, and they are the paper's most negative fairness finding.

Meanwhile Appendix E's Limitations said "we report no subgroup calibration and no subgroup
predictive values", and Appendix L said the audit "contains no predictive-value analysis and
no subgroup calibration" - the latter in the same sentence that had already cited the
calibration of Appendix K, so it contradicted itself within one sentence.

No calibration result was deleted. The two scope statements were corrected to say what is
genuinely absent:

- Appendix E: "Subgroup sensitivity, specificity and race-stratified expected calibration
  error at a transferred threshold are reported in Appendix K; what is absent is subgroup
  predictive values, reliability curves, and calibration stratified by any attribute other
  than race, so this is not a complete fairness evaluation."
- Appendix L: "... it contains no predictive-value analysis, no reliability curves, and no
  calibration stratified by sex, ethnicity or severity ..."
- Appendix K now says so from its own side: "race is the only attribute for which we report
  stratified calibration."

**Related overreach, also corrected.** Section 5.5 said the unanimity of the worst-served
group shows "the disparity is not introduced by any policy we tried". Unchanged worst-group
identity across non-independent probes cannot establish where a disparity comes from. Now:
"no policy we tried removes or reorders the disparity, not that policy has no part in its
origin."

---

## 3. Artifact terminology and subset documentation

All three counts are now traceable to the artifacts, not inferred.

**Where the counts come from** (`D:\jepa_phase0\autopilot_out\p1_stats\`,
`D:\jepa_phase0\reports\subgroup\`):

- 37 table rows = 31 valid + 2 excluded + 4 retracted (`p1b_full_inventory.json`).
- 31 -> 23: `p7b_gap_trend.py` collapses eight fp32 re-probes of an encoder already probed at
  fp16 (`collapsed_duplicates` in `p7b_gap_trend.json`: random ep50/75/100, oracle ep50/100,
  envelope ep50/75/100), leaving 23 distinct encoder-epoch units.
- 23 -> 19: `p7_fairness.json` holds per-group race summaries for 19 arms. The four without
  one are COVER at epochs 73, 75 and 100 and ANATOMY-V2 at epoch 75. That artifact was
  generated 2026-08-22 19:03; those four probes' `test_predictions.npz` are dated 2026-08-23
  04:06, 2026-08-23 07:29, 2026-08-24 14:08 and 2026-08-26 08:12 - all after it, and it was
  not recomputed. Their race *gaps* exist, so they do enter the 23-probe trend test.
- 22 -> 18 intersectional: `intersectional_auc.json` (generated 2026-08-19 21:50) holds 22
  arms, 4 of them status RETRACTED. The five of the 23 that are absent are COVER at epochs
  50, 73, 75 and 100 and the ANATOMY-V2 epoch-75 probe; their predictions are dated
  2026-08-20 09:14 and later, again all after the artifact was generated.

**Text added.** Appendix A now states the eight-duplicate collapse, names the four probes
missing from the 19-probe race scatter, and says the omission is artifact chronology rather
than a property of the runs. Appendix E.1 now names the five probes missing from the 18-arm
intersectional artifact, gives the same reason, and notes that the four retracted arms the
artifact does contain are dropped under the Appendix F rules.

**BLOB defined.** The residual `\textsc{blob}` in Appendix E.1 is now `\texttt{blob}`, defined
in place: "whose run directory carries the internal name `blob` - an artifact label for the
anatomy-shaped family that this paper calls ANATOMY-V2, and the only place that label
survives here". Verified true: the inventory tag is `frozen_meanpool_blob_fp32_ep75`, arm
`anatomy-v2`, epoch 75, fp32, status primary. Two incidental lowercase uses of "blob" as a
common noun were removed so the label is unambiguous ("blob geometry" -> "target shape";
"more or larger blobs" -> "more or larger connected components").

**PENDING language** (third element of AC item 3) replaced where no result belongs in the
submission: "every result of it is PENDING" -> "with no result of it reported here"; "Status
at submission: PENDING" -> "Status at submission: no result reported"; "per-cell significance
is PENDING" -> "is not reported"; "per-cell delta significance is PENDING and is computable"
-> "is not computed here, though it is computable"; "Both are PENDING; neither has been run"
-> "Neither has been run". No PENDING remains in the manuscript.

---

## Page budget

The body was already at the hard nine-page limit with the references heading at the exact top
of page 10 (y=72.79). The narrowings cost about 163pt, which put the body on ten pages. That
was recovered without deleting a limitation, caveat, result or citation:

- title rebalanced to three typeset lines (20pt);
- Figure 1 0.53 -> 0.47 linewidth, Figure 2 0.78 -> 0.71 linewidth;
- Figure 1, Table 1 and Figure 2 captions compressed (redundant clauses only);
- four prose redundancies removed where the same statement already appears elsewhere: the
  ENVELOPE rectangle description repeated from Section 3.2, the label-subsampling seed
  sentence repeated from Appendix I, "the most surprising result in the study", and the
  duplicated "Every number here is generated by one script" (Appendix P states it).

A numeric-literal diff of the whole file confirms the only numbers changed anywhere are the
two `\includegraphics` widths. No result digit was altered.

---

## Verification (from `C:\Users\Gary\Desktop\jepa`)

| Gate | Result |
|---|---|
| `autopilot\p13_build_zip.py` | 6/6 checks PASS, main content pages 9 (limit 9), references start page 10 at y=72.79, ALL_PASS = True |
| `autopilot\check_manuscript.py` | RESULT: PASS - macros 400 defined / 0 duplicate / 0 undefined, citations 47 cited 0 missing, labels 56, refs 56, **dangling 0** |
| `autopilot\p15_verify_numbers.py` | RESULT: PASS - 20 AUC macros verified against the inventory, no cross-arm attribution |

Two warnings from `check_manuscript.py` are pre-existing and unrelated: 232 unused generated
macros, and a regex false positive on "epoch-92 probes" in Section 4.

Not committed, as instructed.
