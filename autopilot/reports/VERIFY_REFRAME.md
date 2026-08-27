# Adversarial verification of the Results reframe

Target: `paper/genai4health2026/main_submission.tex` and the rendered
`main_submission.pdf`.

Verdict counts:

| category | findings |
|---|---:|
| WRONG NUMBER | 2 forbidden literal alterations, both layout dimensions; 0 wrong added scientific values |
| OVERCLAIM | 3 |
| DROPPED CAVEAT | 0 |
| INCONSISTENCY | 5 |
| CONFIRMED | 8 groups of checks |

## WRONG NUMBER

1. **[MEASURED] `main_submission.tex:249`: `0.66` became `0.53`.** HEAD line
   255 has `\includegraphics[width=0.66\linewidth]`; the working tree has
   `0.53\linewidth`. This is a layout parameter, not a scientific result, but
   the requested invariant was “NO DIGIT CHANGED” and explicitly disallowed
   alteration. There is no measurement artifact backing either width.

2. **[MEASURED] `main_submission.tex:398`: `0.88` became `0.78`.** HEAD line
   396 has `\includegraphics[width=0.88\linewidth]`; the working tree has
   `0.78\linewidth`. This is likewise a layout-only alteration with no backing
   measurement artifact.

No added scientific value failed its backing artifact; the complete added-value
ledger is under CONFIRMED.

## OVERCLAIM

1. **[MEASURED] Background is repeatedly said to contribute nothing to the
   classifier, although the paper's own residual probe is above chance.**
   The categorical statements occur at lines 65, 509, 521, 525 and 706:
   “contributes nothing,” “not for the classifier,” “none of it reaches,”
   “useless classification feature,” and “never reaches.” The backing artifact
   `autopilot/bgsig/a2_region_incremental.json`,
   `random.bg_residual_on_anatomy`, gives test AUC
   `0.5515071507150715` with interval
   `[0.5164548313700296, 0.5893034399036036]`, not chance AUC `0.5`.
   The same artifact gives
   `random.delta_cat_minus_anatomy_ci95_and_mean =
   [-0.013850635858963197, -0.0011915449892519023,
   -0.007608015169676156]`. Thus the supported claim is narrower:
   background has weak residual predictive signal but gives no incremental
   benefit when appended to anatomy under this probe.

2. **[INFERRED] Lines 378 and 388 promote run-level evidence to a policy-level
   conclusion.** “H1 holds” and “Region is a real design variable” use
   `+0.0120` for ENVELOPE-minus-RANDOM at epoch 50 and `+0.0109` for
   CENTROID-minus-RANDOM at epoch 100
   (`paper/genai4health2026/auto/auto_numbers.tex`:
   `\DEnvelopeRandomEpFifty=+0.0120`,
   `\DOracleRandomEpHundred=+0.0109`). But lines 657--659 say that with
   `n=1` continuation per arm the ranking is not statistically established
   even when the fixed-test-set paired intervals exclude zero. The findings can
   describe these runs; they do not yet establish an expected policy effect.

3. **[INFERRED] Lines 630--638 and 699--710 claim that visible context
   “matters” and anatomical coverage “does not,” although the design explicitly
   does not identify either variable.** The production geometry artifact
   `results/masking/table2_geometry/mask_geometry_600slices_bs1_coverf021_seed42.json`
   gives CENTROID anatomy hidden `62.132389842416934`, mask ratio
   `0.4029231770833333`, context kept `0.45495442708333333`, and loss slots
   `158.98666666666668`; ANATOMY-V2 gives `79.88557663354412`,
   `0.21347005208333333`, `0.6774283854166666`, and `64.0`, respectively.
   COVER gives anatomy hidden `73.54963364448459` but has the documented
   post-placement collation defect. Lines 497--499 themselves say H3 is not
   identified, and lines 709--710 say the design cannot separate placement from
   mask ratio and task difficulty. The observed ordering rejects “more anatomy
   is always better”; it cannot establish that context matters causally or that
   coverage does not.

## DROPPED CAVEAT

**[MEASURED] None of the six requested caveat families was dropped.**

- `n=1` continuation per arm and the unestablished ranking remain forceful at
  lines 655--663 and render on PDF page 9.
- **[PENDING]** The replication is labelled PENDING at lines 664--671 and
  745--748; the appendix says, “No result of this replication is reported
  anywhere in this paper.” It renders on PDF pages 9 and 14.
- The anatomy-family confounds remain together at lines 492--507: retained
  context, mask ratio, `64` versus about `159` loss slots, and different
  collation. The stronger delivered-context accounting remains at lines
  896--905.
- The COVER collation defect and its invalidation of an aggressive-coverage
  reading remain at lines 436--442, 681--685 and 1096--1145.
- Selective stopping remains explicit at lines 301--306; repeated adaptive use
  of the test split and the “descriptive rather than confirmatory” consequence
  remain at lines 685--689. They render on PDF pages 4 and 9.
- “Not identified by this design” remains at lines 497--499 and 709--710 and
  renders on PDF pages 7 and 9.

## INCONSISTENCY

1. **[MEASURED] Line 429 calls COVER the study's “one negative result,” but
   ANATOMY-V2 is also significantly below the null at epoch 75.**
   `paper/genai4health2026/auto/auto_numbers.tex` gives
   `\DAnatomyTwoRandomEpSeventyFive=-0.0111` and
   `\DAnatomyTwoRandomEpSeventyFiveCI=[-0.0181,-0.0042]`. COVER is negative
   at epoch 75 (`-0.0084`) and epoch 100 (`-0.0168`) in the same artifact, so
   the COVER result is not unique.

2. **[MEASURED] Lines 445--446 say CENTROID's null margin “does not decay,”
   but Table 1 shows a small late decay.**
   `paper/genai4health2026/auto/auto_numbers.tex` gives the three margins as
   `+0.0099`, `+0.0113`, and `+0.0109` at epochs 50, 75, and 100. The margin
   decreases by `0.0004` from epoch 75 to 100. It remains the largest margin
   among the compared arms at epoch 100, but “does not decay” is false
   literally.

3. **[MEASURED] Lines 481--484 are locally inconsistent with Table 2's own
   matched-epoch caption.** The sentence says “the best performer hides the
   least anatomy.” Table 2 explicitly reports AUC at epoch 50, where
   `paper/genai4health2026/auto/auto_numbers.tex` gives ENVELOPE `0.8761`
   and CENTROID `0.8740`. The geometry artifact above gives ENVELOPE anatomy
   hidden `77.57703502960955` and CENTROID `62.132389842416934`. CENTROID is
   the long-horizon winner at epoch 100 (`0.8855` versus ENVELOPE `0.8807` in
   `auto_numbers.tex`), but the sentence must say “long-horizon winner” to
   avoid contradicting the adjacent epoch-50 table.

4. **[MEASURED] New Section 5.3 silently combines different arms and
   checkpoints as one narrative.** The `0.680`/`0.633` skills are RANDOM
   epoch 100
   (`D:\jepa_phase0\reports\background_signal\skill_scores.json`,
   `random_ep100.bg.skill_vs_pos=0.6798094511032104`,
   `random_ep100.anat.skill_vs_pos=0.6334153413772583`).
   The `0.784` to `0.346` self-similarity comparison is untrained versus
   ENVELOPE epoch 100
   (`results/masking/class_relations/class_relations.json`,
   `JEPA untrained (control).bg_bg=0.7842189359236829`,
   `JEPA ep100 (envelope).bg_bg=0.3460093365644443`).
   The `95.2%`, `0.5515`, and append-to-anatomy result are RANDOM epoch 50
   (`autopilot/bgsig/a2_region_incremental.json`, whose `source`/`note`
   identify ep50; values `0.9522224497054176` and
   `0.5515071507150715`). Lines 513--524 do not disclose either switch, so a
   reader can reasonably take the paragraph as a coherent RANDOM-epoch-100
   analysis.

5. **[MEASURED] The change report's deletion accounting is incomplete, and
   the stated “number lives in a table” rationale is not universally true.**
   Scientific literals `0.20` (two occurrences), `0.50`, `0.60`,
   `0.8854754`, and `0.8854852` are removed and have no surviving exact
   occurrence in a result table or appendix. The first four correlation
   values came from old lines 494--496; the last two AUC values came from old
   line 616. The inferential caveats remain in prose, so this is not counted as
   a dropped caveat, but `REFRAME.md` does not account for these removals.

## CONFIRMED

### 1. Complete numeric-literal diff ledger

**[MEASURED]** Numeric tokens were extracted from zero-context diff lines with
alphanumeric boundaries, then identical added/removed multisets were cancelled.
Gross extraction, including unchanged literals on rewritten lines:

```text
ADDED:
0.0009 x1; 0.346 x1; 0.53 x1; 0.5515 x1; 0.633 x1; 0.680 x1;
0.78 x1; 0.784 x1; 4 x1; 5 x2; 6 x1; 9.8 x1; 16 x2; 22 x1;
30 x1; 40.8 x1; 50 x7; 64 x1; 75 x3; 90.8 x1; 95.2 x1; 97 x1;
97.1 x1; 100 x7; 159 x1; 256 x1; 600 x2.

REMOVED:
0.0009 x1; 0.20 x2; 0.40 x1; 0.50 x1; 0.60 x1; 0.66 x1;
0.736 x1; 0.761 x1; 0.80 x1; 0.879 x1; 0.88 x1;
0.8854754 x1; 0.8854852 x1; 0.895 x1; 0.90 x1; 4 x1; 5 x2;
6 x1; 9.8 x1; 16 x2; 21.3 x1; 22 x1; 30 x1; 40.0 x1;
41 x2; 43.3 x1; 45 x2; 50 x10; 64 x1; 67.7 x1; 73.1 x1;
75 x4; 77.6 x1; 95 x2; 97 x1; 97.1 x1; 100 x9; 159 x1;
160 x1; 256 x1; 600 x2.
```

After cancellation, the net additions are:

```text
0.346, 0.53, 0.5515, 0.633, 0.680, 0.78, 0.784, 40.8, 90.8, 95.2
```

The net removals are:

```text
0.20 x2, 0.40, 0.50, 0.60, 0.66, 0.736, 0.761, 0.80, 0.879,
0.88, 0.8854754, 0.8854852, 0.895, 0.90, 21.3, 40.0, 41 x2,
43.3, 45 x2, 50 x3, 67.7, 73.1, 75, 77.6, 95 x2, 100 x2, 160
```

The only net additions not in the promoted background result are the two
layout alterations already reported under WRONG NUMBER.

### 2. Added scientific literals and backing artifacts

**[MEASURED]** Each promoted value occurs once in the body and once in Appendix
H, with the same printed string:

| body / appendix lines | printed value | backing artifact and stored value |
|---|---:|---|
| 514 / 1156 | `0.680` | `D:\jepa_phase0\reports\background_signal\skill_scores.json`, `random_ep100.bg.skill_vs_pos = 0.6798094511032104` |
| 515 / 1157 | `0.633` | same artifact, `random_ep100.anat.skill_vs_pos = 0.6334153413772583` |
| 519 / 1159 | `0.784` | `results/masking/class_relations/class_relations.json`, `JEPA untrained (control).bg_bg = 0.7842189359236829` |
| 520 / 1160 | `0.346` | same artifact, `JEPA ep100 (envelope).bg_bg = 0.3460093365644443` |
| 517 / 1172 | `90.8%` | `autopilot/bgsig/a3b_threshold_sweep.json`, `random_ep100.background["<=0.10"].position_share = 0.9082097946887898` |
| 518 / 1173 | `40.8%` | same artifact, `random_ep100.anatomy[">=0.20"].position_share = 0.40799012607939` |
| 521 / 1163 | `95.2%` | `autopilot/bgsig/a2_region_incremental.json`, `random.bg_residual_on_anatomy.ridge_test_R2_bg_from_anatomy = 0.9522224497054176` |
| 523 / 1164 | `0.5515` | same artifact, `random.bg_residual_on_anatomy.test_auc = 0.5515071507150715` |

The rounding is consistent in all eight cases. The undisclosed arm/epoch
switches are reported separately under INCONSISTENCY.

### 3. Direction of the anatomy-coverage claim

**[MEASURED]** No sentence says that hiding more anatomy is better or that the
winner hides the most. The paper states the opposite at lines 62, 481--484 and
636--638. The production geometry artifact
`results/masking/table2_geometry/mask_geometry_600slices_bs1_coverf021_seed42.json`
stores CENTROID anatomy hidden `62.132389842416934`, the least of the guided
arms, and ANATOMY-V2 anatomy hidden `79.88557663354412` with purity
`97.0935374668334`, the most and purest. The local epoch-50 “best performer”
ambiguity is reported above rather than silently accepted.

### 4. Table 1 and Figure 2 caption checks

**[MEASURED]** Apart from the two narrative inconsistencies already reported:

- `paper/genai4health2026/auto/auto_numbers.tex` stores all six
  ENVELOPE/CENTROID-versus-null intervals as excluding zero; their largest
  adjusted value is ENVELOPE epoch 100 `q=0.0299`, matching lines 384--388.
- The same artifact stores COVER-versus-null as `+0.0002` with interval
  `[-0.0050,+0.0053]` at epoch 50, then `-0.0084` and `-0.0168` at epochs 75
  and 100. This matches Figure 2's caption: straddling zero at 50 and negative
  thereafter.
- The artifact stores ANATOMY-V2-versus-null at epoch 50 as `+0.0013` with
  interval `[-0.0055,+0.0082]`, supporting “indistinguishable rather than
  worse” at the matched epoch.
- COVER's peak-to-epoch-100 change is `-0.0071` in the same artifact, and its
  null gap widens over the three matched epochs.

### 5. Table 2 geometry checks

**[MEASURED]** The body values and relational claims match
`results/masking/table2_geometry/mask_geometry_600slices_bs1_coverf021_seed42.json`:
CENTROID context kept is `0.45495442708333333`, above the other rectangle arms;
ANATOMY-V2 context kept is `0.6774283854166666`, mask ratio
`0.21347005208333333`, and loss slots `64.0`; rectangle-arm loss slots are
`158.98666666666668` to `159.90666666666667`. These support the stated
confound accounting, not a causal ranking of the confounds.

### 6. Appendix H promotion and count words

**[MEASURED]** Appendix H is neither orphaned nor copied wholesale. Section
5.3 references it at line 528, and Appendix H remains at lines 1150--1187 with
the fuller intervals, background-only caveat and two unrun controls. After TeX
normalisation, the body subsection has no complete sentence exactly duplicated
in Appendix H; the longest exact shared fragment is 60 characters.

Count words match their rows/design:

- “six policies” means RANDOM, ENVELOPE, CENTROID, ANATOMY-V1, ANATOMY-V2 and
  COVER; the combined anatomy item accounts for two policies.
- The full contrast table has nine rows: three pair types at three epochs. Six
  are ENVELOPE/CENTROID versus the null.
- The replication has three policies times two new continuations, hence six new
  continuations; adding the reported continuation gives three per policy and
  nine epoch-50 AUCs.

### 7. PDF-level verification

**[MEASURED]** The PDF is newer than the TeX source and contains the rewritten
text. It was read with PyMuPDF, normalised with
`unicodedata.normalize("NFKD", text)`, and end-of-line word hyphens were
rejoined before searching.

- The PDF has 32 pages total.
- Main content occupies pages 1--9; page 9 ends with the Conclusion.
- “References” begins at the top of page 10.
- Results begins on page 5; the new Section 5.3 renders on page 7.
- The n=1 ranking caveat, PENDING replication, H3 non-identification, COVER
  defect, selective stopping, adaptive reuse and unrun-controls caveats are all
  present in rendered text.
- Appendix H's heading renders on page 20.

### 8. Existing automated gates

**[MEASURED]** Independent reruns completed successfully:

- `D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p15_verify_numbers.py`:
  `RESULT: PASS`, 20 AUC macros checked.
- `D:\jepa_phase0\.venv\Scripts\python.exe autopilot\check_manuscript.py`:
  `RESULT: PASS`, 53 labels, 52 references, 0 dangling references.

These gates do not detect the literal layout alterations, arm/epoch
provenance switches, or narrative overclaims reported above.
