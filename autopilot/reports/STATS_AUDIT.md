# Statistical Reporting Audit

**Paper:** `paper/genai4health2026/main_submission.tex`  
**Checklist:** `.agents/skills/statistical-analysis/SKILL.md` and the requested reporting, effect-size/power, and assumptions references  
**Finding count:** 2 critical, 5 high, 6 medium, 1 low

The audit was textual and artifact-backed. No statistic was recomputed, no result
estimate or test digit was changed, and no limitation was removed.

## Findings, worst first

### Critical

1. **A surviving false multiplicity claim said the race trend passed correction.**
   The fairness figure caption said the checkpoint result passed multiplicity
   correction, although the paper reports `q=0.0668` against `0.05`. This directly
   reverses the decision. **Fixed:** the caption now says it fails checkpoint-level
   correction and also fails to persist after branch aggregation.

2. **The bootstrap estimator was over-generalised and one resample count was
   wrong.** `\Nboot` is 10,000, but the marginal subgroup artifact records 3,000
   resamples; the intersectional producer uses 2,000. The Methods wording also
   implied that every interval used one class-stratified draw shared across arms,
   which is false for those per-arm subgroup analyses and for the background
   analysis. **Fixed:** Methods now scopes the 10,000-draw shared bootstrap to
   primary AUCs and paired deltas; the fairness and intersectional intervals now
   state their own resampling units, counts, percentile method, and fixed heads.

### High

3. **Seven branch-level Spearman tests were an uncorrected second family, with
   only seven branches.** Raw branch `p` values were used to call sex “consistent
   across both views,” even though the same seven attributes were tested again.
   At `n=7`, asymptotic Spearman `p` values are also weak evidence. **Fixed:** the
   family is now explicit, branch `p` values are labelled unadjusted and
   descriptive, and only the checkpoint-level BH result is called corrected.
   **Operator:** use an exact/permutation analysis and multiplicity adjustment, or
   omit branch-level significance language.

4. **Ten reported subgroup AUC contrasts had no declared family and unadjusted
   intervals were interpreted as confirmed improvements.** These are eight
   stratum deltas plus two gain-difference contrasts. **Fixed:** one unadjusted
   exploratory family is declared, simultaneous coverage is disclaimed, and
   prose now reports positive point estimates with unadjusted intervals.
   **Operator:** choose Holm/BH or simultaneous paired-bootstrap intervals before
   making stratum-level inferential claims.

5. **Five subgroup sensitivity contrasts were interpreted without multiplicity
   control.** White, female, and male intervals excluded zero, but race and sex
   tests overlap and were silently read together. **Fixed:** the five are now one
   unadjusted exploratory family; estimator details are complete. **Operator:**
   adjust the family before describing any subgroup as demonstrably benefiting.

6. **Non-significance was repeatedly written as equivalence.** “Matches” and
   “indistinguishable” appeared where intervals span zero, without TOST, ROPE, or
   a smallest effect of interest. **Fixed:** replaced with “does not separate,”
   “intervals span zero,” or “not resolved as worse.”

7. **The fine-tuning claim selected the best head by test AUC and inferred a
   persistent gap/mechanism from point estimates.** There is no interval,
   validation-selected head comparison, or correction across the six
   arm-by-head results. **Fixed:** title and prose now say only that the observed
   gap narrows; head comparisons are descriptive and do not distinguish
   representation from optimisation. **Operator:** validation-select the head and
   estimate a paired interval if this result is to support a substantive claim.

### Medium

8. **Four operating-point arm-by-target intervals formed an undeclared,
   unadjusted family.** **Fixed:** the family and its unadjusted exploratory status
   are stated, with 10,000 class-stratified percentile subject resamples and
   fixed heads/thresholds. **Operator:** adjust before inferential use.

9. **Two background-analysis intervals lacked the estimator, resample count,
   held-fixed quantities, and analysis `n`.** **Fixed:** the appendix now gives
   `n=1000`, 2,000 case-percentile resamples, paired resampling for the delta, and
   states that fitted transforms/probes were fixed.

10. **The fp32 diagnostic reports eight unadjusted DeLong `p` values and prints
    `p=0.000`.** The family was undefined, and non-significance would not establish
    equivalence. **Fixed:** the caption calls the eight values unadjusted
    diagnostics, not a family-wise test, and removes “remove any doubt.”
    **Operator:** render the zero as `p<.001`; preferably report paired difference
    intervals against a pre-specified negligible precision effect.

11. **The 25 exploratory matched-epoch contrasts are correctly separated from
    the nine primary contrasts, but their adjusted `q` values are not printed
    beside claims about anatomy and coverage.** A reader cannot verify from the
    paper which exploratory claims survive BH. **Operator:** print the relevant
    adjusted values or explicitly label those statements descriptive.

12. **Practical effect-size interpretation is only partial.** Delta AUC is
    consistently identified in AUC units, so it is an unstandardised effect size.
    No standardised effect or pre-specified clinically meaningful AUC difference
    is given. The operating-point appendix does help: it calls `+0.011` AUC hard
    to interpret, shows sensitivity/specificity changes, notes threshold
    non-comparability, and disclaims clinical utility. **Operator:** define a
    clinically anchored smallest effect of interest or explain why no accepted
    anchor exists.

13. **Assumption reporting is candid but incomplete.** Subject independence is
    plausible because volumes are distinct subjects; fixed-model bootstraps are
    correctly conditional on trained heads. Geometry correlations at `n=4--5`
    are explicitly dismissed, and checkpoint pseudo-replication is acknowledged.
    Remaining weaknesses are percentile rather than BCa intervals in small cells,
    no bootstrap stability/sensitivity check, and no monotonicity/tie diagnostic
    or correlation CI. **Operator:** add diagnostics or retain strictly
    descriptive language. No Welch test appears in the paper, so Welch
    assumptions are not applicable.

### Low

14. **Reproducibility metadata is incomplete under the skill checklist.**
    Software/library versions, one- versus two-sided test declarations, and
    bootstrap RNG seeds are not in the manuscript. These omissions do not change
    the reported numbers but limit exact replication.

## Multiplicity family disposition

| Family | Disposition after textual fixes |
|---|---|
| Nine precision-matched contrasts | BH named; exact nine defined; separate from 25 exploratory contrasts; internally consistent |
| 25 exploratory matched contrasts | BH and family size defined; adjusted values not displayed for cited claims |
| Seven checkpoint subgroup attributes | BH named; exact seven defined; `q=0.0668` now correctly described as failing `0.05` |
| Seven branch subgroup attributes | Explicitly unadjusted/descriptive at `n=7`; no family-wise claim |
| Six intersectional cells | No paired cell-delta tests; all six reported; per-cell significance explicitly absent, so no correction is claimed |
| Ten subgroup AUC delta contrasts | One explicit unadjusted exploratory family; no simultaneous coverage claim |
| Five subgroup sensitivity contrasts | One explicit unadjusted exploratory family |
| Four arm-by-target sensitivity contrasts | One explicit unadjusted exploratory family |
| Eight fp32 diagnostics | Explicitly unadjusted diagnostics, not a family-wise test |

## Interval specification after fixes

| Interval set | Resampling and what is fixed |
|---|---|
| Primary per-arm AUCs and paired AUC deltas | 10,000 class-stratified subject resamples; percentile; fitted heads fixed; draw shared across arms |
| Marginal race AUCs | 3,000 class-stratified resamples within group; percentile; fitted heads fixed |
| Paired subgroup AUC deltas | 10,000 class-stratified subject resamples; percentile; fitted heads fixed; draw shared across arms/strata |
| Intersectional per-arm AUCs | 2,000 class-stratified resamples within cell; percentile; fitted heads fixed; marginal, not simultaneous |
| Background AUC and paired delta | `n=1000`; 2,000 case resamples; percentile; fitted transforms/probes fixed |
| Overall sensitivity deltas | 10,000 class-stratified subject resamples; percentile; fitted heads/thresholds fixed |
| Subgroup sensitivity deltas | 10,000 positive-subject resamples within stratum; percentile; paired arms; fitted heads/threshold fixed |

## Verification

- `autopilot/p13_build_zip.py`: **6/6 PASS**, body **9 pages**, `ALL_PASS=True`
- `autopilot/check_manuscript.py`: **RESULT: PASS**, dangling references **0**
- `autopilot/p15_verify_numbers.py`: **RESULT: PASS**
