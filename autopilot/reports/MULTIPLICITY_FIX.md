# Subgroup Multiplicity Fix

## Decision and method

Adjustment was computable from the retained artifacts. Neither
`D:\jepa_phase0\autopilot_out\p1_stats\p7c_paired_subgroup.json` nor
`results\p16_subgroup_operating.json` retained per-contrast p-values; they
retained marginal percentile intervals. Following
`.agents\skills\statistical-analysis\references\reporting_standards.md`, the
declared families were therefore re-analysed from the paired per-case
predictions rather than treating marginal intervals as simultaneous.

`autopilot\p17_adjust_subgroup_multiplicity.py` performs a single-step
max-absolute-t paired bootstrap with 10,000 draws and family-wise alpha 0.05.
For contrast \(j\), its bootstrap standard error is \(s_j\), and the critical
value is the 95th percentile of
\(\max_j |(\hat\theta_j^*-\hat\theta_j)/s_j|\). The simultaneous interval is
\(\hat\theta_j \mathbin{+/-} c_{0.95}s_j\). The adjusted p-value is the
finite-bootstrap proportion of maximum absolute t statistics at least as large
as the observed absolute t, using the `(count + 1) / (10000 + 1)` correction.

- AUC family: one outcome-stratified subject draw is shared across both fixed
  heads and all strata. The family contains eight stratum AUC deltas and the
  Black-minus-Asian and Female-minus-Male gain differences. Seed 20260826;
  max-absolute-t critical value 2.735046.
- Sensitivity family: one positive-subject draw is shared across both fixed
  heads and all five overlapping race and sex strata; the externally selected
  threshold 0.55517578125 remains fixed. Seed 20260827;
  max-absolute-t critical value 2.528359.

The full 15-contrast analysis took 7.2 seconds on CPU, within the operator's
two-minute limit. Its machine-readable result is
`results\p17_subgroup_multiplicity.json`.

## Ten-contrast subgroup AUC family

The “unadjusted” column reproduces the retained 95% percentile CI. The
“simultaneous” column is the new family-wise 95% max-absolute-t CI.

| Contrast | Estimate | Unadjusted 95% CI | Simultaneous 95% CI | Adjusted p | Excludes 0 before / after |
|---|---:|---:|---:|---:|---|
| Severity, mild | +0.01372 | [+0.00541, +0.02192] | [+0.00204, +0.02541] | 0.0127 | yes / yes |
| Severity, moderate | +0.01016 | [+0.00262, +0.01757] | [-0.00029, +0.02061] | 0.0607 | yes / no |
| Severity, severe | +0.00626 | [+0.00104, +0.01187] | [-0.00135, +0.01386] | 0.1737 | yes / no |
| Race, White | +0.01129 | [+0.00518, +0.01731] | [+0.00286, +0.01972] | 0.0025 | yes / yes |
| Race, Black | +0.01467 | [-0.00055, +0.03060] | [-0.00742, +0.03676] | 0.4033 | no / no |
| Race, Asian | +0.01558 | [-0.00079, +0.03283] | [-0.00766, +0.03882] | 0.3918 | no / no |
| Sex, Female | +0.01152 | [+0.00412, +0.01890] | [+0.00100, +0.02204] | 0.0243 | yes / yes |
| Sex, Male | +0.01006 | [+0.00335, +0.01678] | [+0.00067, +0.01945] | 0.0295 | yes / yes |
| Race gain difference, Black minus Asian | -0.00091 | [-0.02379, +0.02139] | [-0.03285, +0.03104] | 1.0000 | no / no |
| Sex gain difference, Female minus Male | +0.00146 | [-0.00853, +0.01156] | [-0.01270, +0.01562] | 0.9999 | no / no |

Four of ten AUC contrasts survive: mild severity, White race, Female sex, and
Male sex.

## Five-contrast subgroup sensitivity family

| Contrast | Estimate | Unadjusted 95% CI | Simultaneous 95% CI | Adjusted p | Excludes 0 before / after |
|---|---:|---:|---:|---:|---|
| Race, Asian | +0.0085 | [-0.0339, +0.0508] | [-0.0485, +0.0654] | 0.9947 | no / no |
| Race, Black | +0.0037 | [-0.0261, +0.0336] | [-0.0375, +0.0449] | 0.9993 | no / no |
| Race, White | +0.0343 | [+0.0194, +0.0491] | [+0.0149, +0.0536] | 0.0002 | yes / yes |
| Sex, Female | +0.0259 | [+0.0074, +0.0444] | [+0.0020, +0.0498] | 0.0285 | yes / yes |
| Sex, Male | +0.0275 | [+0.0092, +0.0458] | [+0.0043, +0.0507] | 0.0158 | yes / yes |

Three of five sensitivity contrasts survive: White race, Female sex, and Male
sex. Adjustment removes no sensitivity conclusion.

## Claim removed

**Adjustment removes the paper's claim that all three disease-severity AUC
intervals exclude zero.** Moderate and severe no longer exclude zero after
family-wise adjustment; only mild does. The paper retains all three positive
point estimates but now states this loss of inferential support plainly in the
main results, conclusion, appendix text, and adjusted table.

The White race-group AUC conclusion and both sex-stratum AUC conclusions remain.
The White, Female, and Male sensitivity conclusions also remain.

## Paper changes

`paper\genai4health2026\main_submission.tex` now names the single-step
max-absolute-t method, family sizes 10 and 5, 10,000 resamples, family-wise alpha
0.05, resampling units, fixed heads/threshold, and every surviving contrast.
The two generated tables and their numeric macros now report simultaneous
intervals. `autopilot\p8_make_assets.py` consumes the adjusted artifact, and
`autopilot\refresh_all.py` regenerates it before paper assets.

## Verification

- `autopilot\p13_build_zip.py`: 6/6 checks passed; body 9 pages; total PDF 34
  pages; `ALL_PASS = True`.
- `autopilot\check_manuscript.py`: `RESULT: PASS`; citations missing 0;
  dangling references 0.
- `autopilot\p15_verify_numbers.py`: `RESULT: PASS`.
