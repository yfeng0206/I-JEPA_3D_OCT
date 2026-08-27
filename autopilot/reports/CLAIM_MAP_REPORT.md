# Claim map: report

Artifact: `paper/genai4health2026/research/claim_evidence.csv`
Documentation: `paper/genai4health2026/research/CLAIM_MAP.md`
Generator: `paper/genai4health2026/scripts/make_claim_evidence.py`

`main_submission.tex` was **read only**. No manuscript file was modified. The
manuscript was being edited concurrently while this map was built (it changed
size twice during the pass), so claims are located by section, macro name and
JSON key path rather than by line number.

---

## Counts

| status | claims |
|---|---:|
| SUPPORTED | 103 |
| SCOPED-OK | 13 |
| PENDING | 9 |
| UNSUPPORTED | 4 |
| **total** | **129** |

| claim_type | claims |
|---|---:|
| MEASURED | 97 |
| DERIVED | 16 |
| LIMITATION | 7 |
| INTERPRETIVE | 6 |
| SCOPED | 3 |

Coverage by location: title 1, abstract 19, introduction and contributions 8,
experimental setup 10, results 20, subgroups 11, precision controls 3,
discussion and limitations 5, conclusion 4, appendices 45, figure and table
captions 3.

Every `evidence_path` in the file was checked to exist on disk; the generator
refuses to write otherwise.

---

## UNSUPPORTED (4)

### R-14 - an arm swap inside the background-signal paragraph

**Claim** (Sec. Results, background; repeated in App. background):
"Pretraining spends real capacity on them, and background self-similarity falls
from **0.784** untrained to **0.346**." The appendix form adds "at epoch 100",
inside a paragraph whose first sentence is "At epoch 100 the **random arm's**
predictor beats a per-position, no-context reference...".

**Evidence**: `results/masking/class_relations/class_relations.json`.

**What the artifact actually contains**: seven keys. `'JEPA untrained
(control)'.bg_bg = 0.7842` and `'JEPA ep100 (envelope)'.bg_bg = 0.3460`. There
is **no random-arm entry at all** - the only JEPA encoders measured are the
untrained control, anatomy at epoch 30, and **envelope** at epoch 100.

**Why this matters**: both numbers are real and correctly transcribed, so the
number gate cannot see anything wrong. But the value is measured on the
*envelope* encoder and both passages place it inside a paragraph explicitly
about the *random/null* arm. This is precisely the arm-substitution the AUC
provenance gate exists to prevent, occurring in a hand-typed quantity that the
gate does not cover.

**What would close it**: attribute the number to envelope; or measure `bg_bg` on
the random epoch-100 encoder; or restate the sentence as a claim about
pretraining in general rather than about the null arm.

### A-15, I-07, R-24 - "the same worst-served group every time", unqualified

**Claims**, three surfaces plus the ethics appendix:

- Abstract: "A subgroup audit over 23 probes finds **the same worst-served group
  every time**."
- Contributions bullet: "The subgroup audit is a caution: **every policy leaves
  the same groups worst-served**."
- Sec. Subgroups opener: "Across **seven stratifications** and 23 probes ... **no
  masking policy reorders the subgroups**." Section heading: "Subgroups: the
  ordering never changes." Ethics appendix: "Every masking policy we tested
  leaves the same groups worst-served."

**Evidence**: `D:\jepa_phase0\autopilot_out\p1_stats\p7b_gap_trend.json`
`:: worst_group_consistency`.

**What the artifact actually contains**: unanimous 23/23 for sex (female), race
(black), ethnicity (hispanic) and disease severity (mild). **Not** unanimous for
language (`other languages` 19, `spanish` 4) or age (`<60` 22, `60-69` 1).

**Why this matters**: the paper's own Table 5 prints `other languages (19/23)`
and `<60 (22/23)`, so the unqualified sentences contradict a table in the same
document. The body already contains the correct form - "The worst-performing
group is the same in every one of the 23 probes for sex, race, ethnicity and
disease severity" (catalogued as R-25, SUPPORTED).

**What would close it**: carry the four-attribute scope into the abstract, the
contributions bullet, the section opener, the section heading and the ethics
appendix. This is a wording decision, which is the operator's to make.

---

## PENDING (9)

### R-15, R-16, R-17, R-18, A-14 - the background-classifier numbers are epoch 50, in an epoch-100 passage

**Claims**: `95.2%` linear reconstructibility of background from anatomy;
residual background AUC `0.5515` (CI `[0.5165, 0.5893]`); appending background
changes AUC by `-0.0076` (CI `[-0.0139, -0.0012]`) for the null and under
`+/-0.002` for every other arm; a background-only probe scores `0.867`. The
abstract's "background contributes almost nothing to the classifier" rests on
the same measurements.

**Evidence**: `autopilot/bgsig/a2_region_incremental.json`. Every value matches
exactly.

**The gap**: the artifact's own `source` field is
`D:\jepa_phase0\reports\region_features\{}_ep50_s100.pt` and its `note` reads
"**ep50 checkpoints**, 25 stratified slices/volume, paired splits";
`autopilot/BACKGROUND_SIGNAL.md` records the same. The manuscript paragraph
opens "At epoch 100 the null's predictor beats..." and never restates an epoch,
so a reader takes the whole passage as epoch 100. The pretraining-side numbers
in the same paragraph (`0.680`, `0.633`) *are* epoch 100 and are SUPPORTED
(R-12).

**What would close it**: label these four values epoch 50, or re-run the region
analysis on epoch-100 features.

### I-01 - "effective batch size (512 for all arms)" is unverified for the random arm

**Claim**: Introduction and Sec. Experimental setup - all arms share "effective
batch size (512 for all arms)".

**Evidence**: `src/train_patch.py` prints
`batch_size * world_size * accum_steps`. Verified from configs for five arms:
`patch_oracle_anatomical.yaml` 64 x 4 x 2, and
`patch_mirage_envelope.yaml` / `patch_anatomy_v2.yaml` /
`patch_mirage_anatomy.yaml` / `patch_cover_f021_ep25.yaml` 64 x 1 x 8. Both give
512.

**The gap**: the **random** arm's producing config is not in this checkout.
`configs/patch_vitb16_ep100.yaml` is an older remote config (batch 32, peak LR
0.0005, not the shared 0.00025) and is not it. Its checkpoint stores
`batch_size` and `world_size` but not `accum_steps`.
`docs/experiments/pretraining/README.md` asserts effective batch 512 for all
completed runs, which is an assertion in a repository README rather than a
producing artifact.

**Related, and worth recording**: an earlier session flagged a "4x
effective-batch confound" between the remote arms (`world_size` 4) and the local
arms (`world_size` 1). That calculation used `batch_size x world_size` and
omitted `accum_steps`. Including `accum_steps` (2 remote against 8 local) makes
both 512 and dissolves the alarm. The paper's claim is very likely right; it is
one artifact short of proven.

**What would close it**: recover the random run's config, or a training log line
`Effective batch size: 512` for that run.

### P-23, P-24 - the occlusion-attribution appendix has no local artifact

**Claims**: Sec. Results cites "Occlusion attribution on the fine-tuned probes is
correspondingly diffuse"; the appendix additionally claims three tests reject a
bilateral-anatomy reading of the bimodal curve, with clustering returning
near-perfect mirror images.

**Evidence**: `docs/experiments/interpretability.md`, which states that "All
.npz outputs and per-slice contribution tables are on blob at
`ijepa-interpretability/`". Those arrays are absent from this checkout. The only
local evidence is the rendered figures.

**Mitigating**: the appendix discloses this itself - "Every number in this
appendix is computed from archived per-volume attribution arrays that are not
part of the released artifact set ... these values cannot be recomputed from the
release." The specific correlations that once supported the mirror argument
(`0.971`/`0.988`, `-0.124`/`-0.478`) were removed from this version rather than
restated, and `interp_heatmap_grid.png`'s own caption says the volumes are
hand-picked illustrative examples with no colour bar.

**What would close it**: restore the attribution arrays into the artifact set,
or re-run occlusion on the released heads and store the output.

### P-38 - the hand-typed-number audit counts describe a superseded snapshot

**Claim**: App. reproducibility - "we audited every numeric quantity typed into
the source: **310** occurrences, of which **234** were confirmed against a stored
artifact, **75** had no producing artifact that we could locate, and **one** was
wrong".

**Evidence**: `autopilot/reports/HARDCODED_AUDIT.md`. The four counts are exact.

**The gap**: that audit records its own scope as a **1,424-line** snapshot with
SHA-256 `99891543...`. `main_submission.tex` is now roughly **2,000** lines. At
least 25 of the 75 unbacked items have since been closed: the 24 Table 2
geometry cells were regenerated against a stored artifact
(`autopilot/reports/TABLE2_PROVENANCE.md`, verified this cycle against
`results/masking/table2_geometry/...seed42.json`), and the interpretability
numbers were removed. The manuscript does not say which snapshot the counts
describe, so the figures read as current when they are not.

**What would close it**: re-run the audit against the current source, or name
the audited snapshot in the sentence.

---

## SCOPED-OK (13) - correct, and the scope is load-bearing

These are not defects. They are listed so a later edit does not drop the scope.

| id | claim | why the scope matters |
|---|---|---|
| T-01, A-07, C-01 | the band "matches" the segmenter-guided arm at epochs 50 and 75 | "Matches" rests on non-rejection, not on an equivalence test. No equivalence margin or TOST is stated anywhere; the widest bound against ArmBest is `-0.0069` at epoch 50. At epoch 100 ArmBest exceeds envelope, so the title is conservative there. |
| R-21, R-22 | reproduction bound `0.0009` (four arms) and `0.0003` (two arms) | Different quantities, both correct. Recomputed: the four-arm bound is set by cover (0.858562 against 0.8577); the two-arm bound is 2.25e-4 (random) and 1.59e-4 (ArmBest). The paper explicitly distinguishes them. |
| P-03, F-02 | `19` probes carry a race summary, `23` enter the trend test | Correct by construction and stated in App. A. `p7_fairness.json :: n_probes_with_race_summary = 19`; `p7b_gap_trend.json :: n_probes = 23`. |
| R-06 | ArmBest's "margin over the null is the largest at epoch 100" | True as "largest of any arm's margin at epoch 100" (+0.0109 against envelope's +0.0062). It is **not** a within-arm maximum: ArmBest peaks at epoch 75 (+0.0113). The non-decay half of the sentence holds either way. |
| R-35 | fp32 re-fit "shifts every arm by less than 2e-4" | 8 of 9 planned re-probes exist; `p3b_fp32.json :: pending = ["frozen_meanpool_oracle_ep75_fp32"]`. True per arm, not per arm-epoch. The table caption's "eight DeLong p-values" discloses the count. |
| A-03 | one continuation per policy | The seed replication is running and explicitly unreported. |
| I-04 | anatomy-v1 exceeded envelope at epoch 30 | Different implementation, non-matched epoch, fp32. The paper says it does not read this as settling H2. |
| S-03 | test split inspected repeatedly while choosing policies | Self-declared, not independently verifiable; the paper says the inspection count was never logged. |
| S-10 | probe precision "inferred from stored prediction dtype" | Precision is inferred, not read from each config; disclosed in the App. A caption. |

---

## Two further observations, both minor, neither status-changing

1. **A stale internal attribution.** App. mask-geometry provenance says "The
   `+0.80` anatomy-hidden/AUC Spearman coefficient **of Section 5.2**". Section
   5.2 no longer prints any coefficient - it now says only that the rank
   correlations "are positive". The value itself is backed
   (`autopilot/reports/TABLE2_PROVENANCE.md` recomputes `+0.80` at cover floor
   0.21 and `+0.40` at 0.15). Catalogued in the note on R-09.

2. **An unnamed arm.** App. occlusion attribution says "the three fine-tuned
   probes ... that tie at test AUC approximately 0.887" without naming the arm.
   The three heads at that AUC are the **random** arm's (0.887763, 0.887178,
   0.886756 in `results/downstream/finetune_random/`), which
   `docs/experiments/interpretability.md` also states. Catalogued in the note on
   P-22.

---

## What the map confirms is fixed

Four of the five failures that motivated this exercise are closed in the current
source, and the map records each with its artifact:

- **R-32** - severity: after adjustment over the declared ten-contrast family
  only mild excludes zero, and the paper says so
  (`results/p17_subgroup_multiplicity.json :: severity:moderate.conclusion_changed = true`).
- **R-34** - the intersectional analysis is now reported, and every one of its
  claims re-ran exactly this cycle
  (`paper/genai4health2026/scripts/intersectional_claims.py`: 18/18, 54/54,
  36/36, gaps 0.0340 / 0.0653 / 0.1046, 60.1%, ratio 1.053).
- **R-29** - the race trend is now correctly described as **not** passing
  correction at `q = 0.0668` (`p7b_gap_trend.json :: trends.race.q_bh_across_attributes = 0.066756`).
- **I-07** - the contributions bullet now reads "The subgroup audit is a
  caution", matching the ethics appendix. Its residual problem is the
  over-generalisation shared with A-15, not the contribution/caution mismatch.

The title no longer asserts an unidentified comparison: ArmBest against envelope
is a within-rectangle-family contrast that the design does identify. What it
asserts instead is an equivalence inferred from non-rejection, which is the
SCOPED-OK caveat at T-01.

Also worth recording as fixed: the earlier audit found 24 of 25 Table 2 geometry
cells unbacked. All 25 now resolve to
`results/masking/table2_geometry/mask_geometry_600slices_bs1_coverf021_seed42.json`,
and the derived provenance tables (three-seed standard deviations, delivered
batch-64 context, the 0.15-floor re-measurement) were recomputed cell by cell
this cycle and match.

---

## Regenerating this map

```
D:\jepa_phase0\.venv\Scripts\python.exe paper\genai4health2026\scripts\make_claim_evidence.py
```

CPU-only, no GPU, no network, under a second. It fails if any `evidence_path`
does not exist, so no row can cite an artifact that is not there.
