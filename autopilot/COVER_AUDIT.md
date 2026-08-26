# COVER f=0.21 audit

Audit run 2026-08-26 (gpt-5.6-sol, xhigh) with independent verification by the
coordinator. Triggered by the principal investigator's hypothesis that
"something went wrong" with the COVER arm.

## VERDICT: REAL BUG FOUND - high confidence

The principal investigator's instinct was correct. The coordinator's competing
hypothesis (that COVER was a genuine over-coverage dose-response result) is
WITHDRAWN and contradicted by measurement.

---

## 1. COVERAGE TARGET DISCREPANCY

COVER greedily places rectangles to satisfy a configured coverage target, and
then the collator discards part of what it placed. The masks the model trains on
are not the masks COVER computed.

**Mechanism.** `src\masks\curriculum.py:1785-1788` truncates every predictor
target to `t[:global_min_pred]`, where `global_min_pred` is the smallest target
length anywhere in the microbatch. The in-code comment at `curriculum.py:1763-1766`
already describes this path as "destructive" and names `pred_target_k` as the
remedy.

**Measured effect.**

| quantity | value | source |
|---|---|---|
| COVER anatomy mass hidden, pre-truncation | 78.62% | fresh CPU audit, 194 accepted slices |
| COVER anatomy mass hidden, post-truncation | 73.88% | same |
| COVER vs envelope, 200-slice end-to-end | 73.28% vs 76.96% | same |
| COVER vs envelope, 6,137-slice sweep | 73.09% vs 77.58% | `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json` |
| images retaining all four rectangular targets | 32.47% | fresh CPU audit |
| mean rectangular targets delivered, of 4 | 2.51 | fresh CPU audit |

[MEASURED] So COVER hides LESS anatomy than envelope, inverting the entire design
intent of the arm. COVER exists specifically to hide MORE anatomy than envelope
while keeping envelope's target geometry.

**The logs concealed this.** [MEASURED] `cover_hidden_sum` is recorded before
truncation (`curriculum.py:1480-1482`, consumed at `:1811`), so the training log
reports approximately 78.5% - the intent - while the model receives approximately
73.9%. Any monitoring based on the training log would show the arm working
correctly.

**Config semantics are sound.** [MEASURED] `target_mass = (1-leave_frac)*total_mass`
and `floor_mass = min_visible_frac*total_mass` are enforced cumulatively
(`cover.py:311-312,364-374,395`). With both set to 0.21 the soft stop and hard
floor coincide as documented (`cover.py:148-152`). Pre-truncation, 89.69% of
slices finished within one percentage point of the mass floor. The bug is in
collation, not in the COVER sampler.

---

## 2. CROSS-ARM COLLATION ASYMMETRY

Discovered during verification of finding 1. Potentially more consequential,
because it affects the study's central design claim.

[MEASURED] The `pred_target_k` remedy is applied to some arms and not others:

| mode | configs | pred_target_k | collation path |
|---|---|---|---|
| `mirage_anatomy` | arm_anatomy, arm_random_default, arm_random_matched, patch_anatomy_v2, patch_blob_fp32, patch_blob_resume, patch_blob_resume_resume, patch_mirage_anatomy | **16** | safe `resample_to_k` (`curriculum.py:1768-1776`) |
| `mirage_envelope` | patch_mirage_envelope | **absent** | destructive prefix truncation (`curriculum.py:1778-1783`) |
| `mirage_cover` | patch_cover_ep25, patch_cover_f021_ep25(+resume), patch_cover_random_ep25(+resume) | **absent** | destructive prefix truncation |

[MEASURED] The YAML absence is decisive: `train_patch.py:123-124` loads the YAML
directly and passes `mask_cfg.get('pred_target_k')` at `:267` and `:415`; the
constructor default is `None` (`curriculum.py:206,223`). No base config, argparse
default or code default supplies it. Anatomy mode refuses to run without it
(`curriculum.py:473-482`).

[INFERRED] The paper claims all arms hold everything but masking policy fixed.
They do not hold collation fixed. The confound runs between the anatomy arms and
the rectangle arms, not between COVER and envelope (those two share a path).

### 2b. The "64 loss slots" attribution is wrong

The paper states of the anatomy arms: "Because their targets are compact tissue
blobs rather than large rectangles, they hide only 21.4% of the grid ... and
supply 64 predictor loss slots against 158-160."

[MEASURED] 64 = 4 targets x K=16. Anatomy masks are shrunk toward K
(`curriculum.py:1305-1334`) and short targets padded with replacement to K
(`masks\utils.py:6-31`). A stored 600-slice budget experiment varying only K
gives K=16 -> union 62.610 / context 193.390 and K=30 -> union 117.077 /
context 138.923 (`D:\jepa_phase0\reports\budget_masks\budget_mask_audit_fairvision.json`).

[INFERRED] The loss-slot count and grid coverage are imposed by the
`pred_target_k: 16` configuration knob, not caused by blob geometry. Blob
geometry determines placement, overlap and occasional padding. The paper's
causal attribution must be corrected.

### 2c. Prefix truncation is a directional spatial bias

[MEASURED] Rectangle cell indices are generated row-first then column-first as
`r * width + c`, then sorted (`curriculum.py:882-889`; `multiblock.py:97-104`).
Therefore `t[:global_min_pred]` retains the TOP rows of each rectangle and, in
the final retained row, its left portion.

[MEASURED] Only 73.4% of 24,000 emitted COVER targets remained perfect rectangles
in the existing production-path validation
(`D:\jepa_phase0\reports\cover_random_scale\scale_validation.json:20-27`;
measurement code `scripts\cover_random_scale_validation.py:131-153`).

[INFERRED] This is a systematic top-of-rectangle bias, not a neutral size
normalisation. It affects envelope and COVER alike.

---

## 3. Checks that came back CLEAN

**Bernoulli granularity.** [MEASURED] COVER draws one Bernoulli per predictor
block per image (`curriculum.py:1420-1430`), matching envelope
(`curriculum.py:1549-1551`) and the stock sampler. Correct.

**Summed-area arithmetic.** [MEASURED] Exhaustive comparison of 18,496 window
configurations against naive sums found no off-by-one failures (`cover.py:52-69`).

**Degenerate inputs.** [MEASURED] Empty and four-cell supports return four
non-empty random rectangles flagged `fallback=True, ok=False`; the caller routes
these to uniform masking (`cover.py:292-305`; `curriculum.py:1469-1514`). Four of
198 viable attempts were infeasible and counted, not silently swallowed. No empty
target observed in 200 slices.

**Floors.** [MEASURED] The four-cell occupancy floor was saturated on 58.25% of
slices, but removing it changed none of 198 samples; removing the fractional
floor changed every sample. The mass floor is the binding constraint.

**Fill and transition.** [MEASURED] `cover_fill: random_legal` supersedes
`cover_transition: true` (`cover.py:207`; `curriculum.py:464-470`). Accepted
slices averaged 3.04 greedy blocks and 0.96 uniform legal blocks, zero transition
blocks. No hidden boundary policy.

**The epoch-73 decline is NOT mechanical.** [MEASURED] All ten COVER training
process logs report `Target-encoder autocast (amp_target): False`, including the
uninterrupted epoch-76-to-100 process (`train_20260823_073016_a0.log:19`). The
resume config sets `amp_target: false`
(`patch_cover_f021_ep25_resume.yaml:55`). The historical hardcoded fp16-target
override is confined to `make_blob_cfg` in the separate blob stage
(`scripts\campaign_chain.py:169-186,245`); COVER used
`scripts\chain_cover_f021.py:30,112-118`. Restarts at epochs 73 and 75 used the
same base config and rolling checkpoint (`supervisor.log:259-272`). [INFERRED]
No precision or configuration switch coincides with the decline.

---

## 4. What this means for the paper's claims

### Statements that must change

1. **"COVER hides approximately 79% of anatomy"** - false for delivered masks.
   It hides approximately 73%. [MEASURED]
2. **"COVER hides more anatomy than envelope"** - false. It hides less
   (73.09% vs 77.58%). [MEASURED]
3. **Any claim that COVER demonstrates harm from over-covering anatomy** - not
   supported. The arm never achieved the coverage it was built to test. [INFERRED]
4. **"All arms hold everything except masking policy fixed"** - false. Anatomy
   arms use `resample_to_k` with K=16; rectangle arms use destructive prefix
   truncation. [MEASURED]
5. **"Because their targets are compact tissue blobs ... they supply 64 predictor
   loss slots"** - wrong causal attribution. 64 is imposed by
   `pred_target_k: 16`. [MEASURED]
6. **Implicit claim that emitted targets are rectangles** - only 73.4% of COVER
   targets remain perfect rectangles after collation. [MEASURED]

### What remains safe to cite

[INFERRED] The ep27/30/34/50/73/75/100 COVER probes are legitimate measurements
of the *implemented* truncated-COVER policy. They are valid as "a coverage-seeking
policy whose realised masks hide approximately 73% of anatomy", and its
trajectory (peak ep73, decline to below the null at ep100) is a real measured
behaviour of that policy. They are NOT valid as evidence about hiding 79%.

[MEASURED] The paper's Table `tab:geom` figure of 74.1% anatomy hidden for COVER
is CORRECT - it measured post-collation reality. The stale claim is in the
`cover.py` docstring (approximately 85%), which describes pre-truncation intent.

### Correcting the experiment

[INFERRED] A corrected COVER run must restart from the shared epoch-25 ancestor
and rerun through epoch 100. Patching after epoch 73 or 75 would splice two
different masking policies into one trajectory. At the measured rate this is
roughly 115 GPU-hours, which is not reachable before the 2026-09-05 deadline on
one RTX 3090.

[INFERRED] The fix is truncation-aware placement: compute the effective global
target length before COVER placement, and score and enforce coverage against the
exact prefix-truncated shapes that will reach the model. Log coverage again after
collation so the discrepancy cannot recur silently.

[INFERRED] Setting `pred_target_k=16` on the rectangle arms is NOT an appropriate
correction - it would cut their approximately 158-160 loss slots to 64 and change
what those arms measure. Either COVER gets truncation-aware placement at the
existing rectangle budget, or all arms are rerun under one common collation policy.

---

## 5. Process note

The single most valuable check was the one the principal investigator's standing
rule forced: search for existing results before computing. Three pre-existing
artifacts independently confirmed the finding and cost nothing to read -
`cover_floor_sweep.json` (6,137 slices), `scale_validation.json` (24,000 targets),
and `budget_mask_audit_fairvision.json` (600-slice K sweep). The fresh CPU audit
agreed with all three.

The failure was invisible for so long because the instrument agreed with the
intent: `cover_hidden_sum` was logged pre-truncation, so every log said the arm
was hiding 78.5% of anatomy exactly as configured.
