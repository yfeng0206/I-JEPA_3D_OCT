# P-VERIFY: Independent numerical audit

## VERDICT: MATERIAL ERRORS FOUND

The primary AUC and paired-inference results are reproducible from the saved
case-level predictions. In particular, the epoch-100 oracle-minus-random result
is **+0.0109043**, with my independent 10,000-draw CI
**[+0.0058574, +0.0159795]** and DeLong \(p=2.8841\times10^{-5}\).

The material errors are outside that headline calculation:

1. `p7b_gap_trend.json` includes two probes that the evidence inventory and
   manuscript explicitly exclude (`frozen_meanpool_bridge_ep75` and `ep92`).
2. The appendix contains a stale 12-probe subgroup analysis that conflicts with
   both the 21-row `p7b` analysis and the exclusion-consistent 19-probe analysis.
3. The abstract says all nine contrasts are against the null and exclude zero.
   There are only six null contrasts; seven of all nine intervals exclude zero.
4. The all-probes appendix table's precision labels disagree with the stored
   `probs` dtypes in 11 rows under the precision definition specified for this
   audit.

Independent scratch code and machine-readable output are at:

- `D:\jepa_phase0\autopilot_out\verify\independent_numerical_audit.py`
- `D:\jepa_phase0\autopilot_out\verify\independent_audit_results.json`

## A. Independent AUC recomputation

I discovered prediction files independently from the specified repo directories
and `D:\jepa_phase0\runs\frozen_*`; I did not use the inventory to choose them.
Every AUC below is `sklearn.metrics.roc_auc_score(labels,
probs.astype(np.float64))`.

| Prediction file/directory | Arm | Epoch | Stored dtype | Recomputed AUC | \(|\Delta|\) vs reported `test_auc` |
|---|---:|---:|---:|---:|---:|
| `meanpool_sweep_random/ep50` | random | 50 | float16 | 0.864097065 | 0 |
| `meanpool_sweep_random/ep75` | random | 75 | float16 | 0.872302169 | 0 |
| `meanpool_sweep_random/ep100` | random | 100 | float16 | 0.874580896 | 0 |
| `meanpool_sweep_oracle/ep50` | oracle | 50 | float16 | 0.874029946 | 0 |
| `meanpool_sweep_oracle/ep75` | oracle | 75 | float16 | 0.883635548 | 0 |
| `meanpool_sweep_oracle/ep100` | oracle | 100 | float16 | 0.885485165 | 0 |
| `meanpool_sweep_mirage/ep50` | envelope | 50 | float16 | 0.876064102 | no sibling JSON |
| `meanpool_sweep_mirage/ep75` | envelope | 75 | float16 | 0.880306726 | no sibling JSON |
| `meanpool_sweep_mirage/ep100` | envelope | 100 | float16 | 0.880743395 | no sibling JSON |
| `frozen_cover_random_ep30` | retracted random | 30 | float32 | 0.855766785 | 0 |
| `frozen_cover_random_ep50` | retracted random | 50 | float32 | 0.858967541 | 0 |
| `frozen_cover_random_ep75` | retracted random | 75 | float32 | 0.861185124 | 0 |
| `frozen_cover_random_ep100` | retracted random | 100 | float32 | 0.860733559 | 0 |
| `frozen_meanpool_anatomy_ep30` | anatomy-v1 | 30 | float32 | 0.858274296 | 0 |
| `frozen_meanpool_bridge_ep35` | anatomy-v2 | 35 | float32 | 0.866128553 | 0 |
| `frozen_meanpool_bridge_ep40` | anatomy-v2 | 40 | float32 | 0.868251422 | 0 |
| `frozen_meanpool_bridge_ep50` | anatomy-v2 | 50 | float32 | 0.865385505 | 0 |
| `frozen_meanpool_bridge_ep75` | anatomy-v2, excluded | 75 | float32 | 0.862492463 | 0 |
| `frozen_meanpool_bridge_ep92` | anatomy-v2, excluded | 92 | float32 | 0.860364036 | 0 |
| `frozen_meanpool_cover_f021_ep27` | cover-f021 | 27 | float32 | 0.848346528 | 0 |
| `frozen_meanpool_cover_f021_ep30` | cover-f021 | 30 | float32 | 0.852248978 | 0 |
| `frozen_meanpool_cover_f021_ep34` | cover-f021 | 34 | float32 | 0.857082572 | 0 |
| `frozen_meanpool_cover_f021_ep50` | cover-f021 | 50 | float32 | 0.864281382 | 0 |
| `frozen_meanpool_envelope_ep30` | envelope | 30 | float32 | 0.853916946 | 0 |
| `frozen_meanpool_fork_ep25` | ancestor | 25 | float32 | 0.848680033 | 0 |
| `frozen_meanpool_mirage_ep50` | envelope copy | 50 | **float16** | 0.876064102 | 0 |
| `frozen_meanpool_mirage_ep75` | envelope copy | 75 | **float16** | 0.880306726 | 0 |
| `frozen_meanpool_mirage_ep100` | envelope copy | 100 | **float16** | 0.880743395 | 0 |

**Maximum absolute discrepancy: 0.0** among the 25 files with a directly
available `test_auc`. Nothing exceeds \(10^{-9}\). The three repo envelope
files have no sibling JSON, but each is byte-identical (native `probs` SHA-256)
to the corresponding `D:\...\frozen_meanpool_mirage_ep*` file, whose
`results.json` agrees exactly.

The premise that every `D:\...\frozen_*` prediction is fp32 is false for these
three `frozen_meanpool_mirage_ep*` copies: their stored dtype is float16.

### `p1b_full_inventory.json`

All retained records have exactly correct arm, epoch, status, dtype-derived
precision and AUC. However, a script calling itself the full inventory says
that every prediction set is enumerated and de-duplicated, while its patterns
never enumerate the three `D:\...\frozen_meanpool_mirage_ep*` physical copies.
Consequently it reports `n_records_total = n_records_after_dedup = 31`.
Enumerating those copies would give 34 physical records and 31 after
de-duplication. This does not change any scientific value because the omitted
files are byte-identical copies.

## B. Test-split identity

**PASS — PAIRED STATISTICS ARE VALID FOR THESE FILES.**

- Files checked: **28**
- Length of every label vector: **3,000**
- Positive: **1,466**
- Negative: **1,534**
- Prevalence: **0.4886667**
- Element-wise mismatches: **none**
- Every label vector had the same SHA-256:
  `dc4f823946abda8848e5765b36f9e8563dd093913b6b9e90878b66dd264db8d0`.

## C. Independent DeLong verification

### Reference implementation

I implemented DeLong independently using the original placement-value
definition, not the audited midrank code:

\[
V_{10,i}={1\over n}\sum_j
\left[I(X_i>Y_j)+\tfrac12 I(X_i=Y_j)\right],\qquad
V_{01,j}={1\over m}\sum_i
\left[I(X_i>Y_j)+\tfrac12 I(X_i=Y_j)\right].
\]

The covariance is
\(\operatorname{cov}(V_{10})/m+\operatorname{cov}(V_{01})/n\).
Across the nine real headline contrasts, the independent and pipeline results
agreed to:

- maximum AUC-difference error: \(1.11\times10^{-16}\);
- maximum standard-error error: \(2.60\times10^{-18}\);
- maximum p-value error: \(2.55\times10^{-14}\).

### Equal-true-AUC null simulation

I simulated **5,000** datasets with 200 positives and 200 negatives. The two
scores had identical marginal signal distributions (therefore identical true
AUCs) and correlated errors through a shared Gaussian component.

| Diagnostic | Result |
|---|---:|
| \(P(p<0.05)\) | **0.0480** |
| Exact binomial 95% band under 0.05 | **[0.0440, 0.0562]** |
| Mean p-value | **0.49818** |
| p-value quantiles (5%, 25%, 50%, 75%, 95%) | **0.05160, 0.25354, 0.49262, 0.74449, 0.95177** |
| Uniform KS statistic | **0.00887** |
| Uniform KS p-value | **0.82311** |

The type-I rate is inside its binomial band, and uniformity is not rejected.

### DeLong SE versus paired bootstrap SD

For the real paired contrasts, the largest relative discrepancy between
DeLong's difference SE and the empirical SD from my independent 10,000-draw
stratified paired bootstrap was **1.167%**, far below the requested 15%.
For the headline epoch-100 oracle-minus-random contrast specifically:

- DeLong SE: **0.002607161**
- bootstrap SD: **0.002606185**
- relative difference: **0.0374%**

**DeLong verdict: PASS.**

## D. Headline contrasts recomputed from scratch

The bootstrap used seed 44017931, independent of the pipeline's seed. Every
draw resampled 1,466 positive and 1,534 negative case indices with replacement,
and the same indices were applied to both arms.

| Epoch | Contrast | Independent \(\Delta\) | Independent paired-bootstrap 95% CI | Independent DeLong \(p\) | Max CI-endpoint difference vs `p1c` |
|---:|---|---:|---:|---:|---:|
| 50 | oracle - random | +0.009932881 | [+0.005083829, +0.014819353] | 0.0000619092 | 0.0001145 |
| 50 | envelope - random | +0.011967037 | [+0.006783024, +0.017389279] | 0.0000070095 | 0.0000951 |
| 50 | oracle - envelope | -0.002034156 | [-0.006882514, +0.002789539] | 0.4114220 | 0.0001002 |
| 75 | oracle - random | +0.011333378 | [+0.006277837, +0.016317061] | 0.0000089514 | 0.0000679 |
| 75 | envelope - random | +0.008004557 | [+0.002851576, +0.013181044] | 0.00250995 | 0.0000545 |
| 75 | oracle - envelope | +0.003328821 | [-0.001270686, +0.007912321] | 0.1490703 | 0.0000573 |
| 100 | oracle - random | **+0.010904269** | **[+0.005857421, +0.015979538]** | **0.0000288409** | 0.0001706 |
| 100 | envelope - random | +0.006162499 | [+0.001001353, +0.011341849] | 0.0199491 | 0.0000420 |
| 100 | oracle - envelope | +0.004741770 | [+0.000509795, +0.009056970] | 0.0294061 | 0.0001490 |

Point estimates and DeLong p-values match `p1c_stats.json` to floating-point
round-off. The largest independently seeded CI-endpoint shift is 0.000171,
ordinary Monte-Carlo variation for 10,000 draws.

**The paper's epoch-100 oracle-minus-random claim is confirmed:** the exact
point difference is +0.0109043. The pipeline-seed CI is
[+0.0056868, +0.0160383], which correctly renders as
**[+0.0057, +0.0160]**. My independent-seed interval rounds to
[+0.0059, +0.0160].

One manuscript sentence is nevertheless false: abstract lines 46--47 say
“All nine pre-specified paired contrasts against the null exclude zero.”
Only six contrasts are against the null. All six do exclude zero, but only
seven of all nine contrasts do; oracle-minus-envelope crosses zero at epochs 50
and 75.

## E. Macro consistency and hand-typed values

### `auto_numbers.tex`

I reconstructed all expected macro strings directly from the corresponding
fields/formulas in:

- `p1c_stats.json`
- `p7_fairness.json`
- `p7_gap_correlation.json`
- `p7b_gap_trend.json`

Result: **136/136 macro definitions match exactly; zero transcription
differences.** This verifies JSON-to-TeX consistency, not whether each JSON's
inclusion criteria are scientifically valid. In particular, the exact macros
faithfully propagate the `p7b` inclusion error discussed in G.

### Hand-typed result literals

The requested scan found **seven** values of the form `0.8xxx...` or `+0.0xxx...`
in the Abstract, Results, or Conclusion that do not resolve through macros:

| Line | Section | Literal(s) |
|---:|---|---|
| 380 | Results | `+0.0013` |
| 492 | Results | `0.8947`, `0.8868` |
| 493 | Results | `+0.0079`, `+0.0109` |
| 552 | Results | `0.8854754`, `0.8854852` |

There are no matches of the requested forms in the Abstract or Conclusion.
The first five result values round consistently with available raw artifacts;
the two line-552 rescore values are not among the prediction files specified
for this audit. They remain transcription risks and should be macros.

## F. Epoch and precision mismatch hunt

Precision here means the stored `probs` dtype, exactly as requested.

| Manuscript location | Compared values | Epoch alignment | Precision alignment | Disclosed? | Audit |
|---|---|---|---|---|---|
| 302--352, 357--371, 401--410 | random/oracle/envelope headline contrasts | matched at 50, 75 or 100 | all fp16 | yes | valid |
| 320--321, 379--396 | anatomy-v2 and cover versus random at epoch 50 | matched | fp32 versus fp16 | **yes**: dagger and prose caveat | deliberately confounded, not a clean arm contrast |
| 418--465 | five-arm epoch-50 geometry/AUC ordering and correlations | matched | three fp16, two fp32 | **yes** in table caption/daggers; less explicit in correlation prose | descriptive only |
| 489--494 | oracle versus random fine-tuning and frozen-gap comparison | both pretraining epoch 100 | saved predictions all fp16 | protocol change is stated | no epoch/precision mismatch |
| 501--539 | subgroup gap/AUC trends across 21 probes | **mixed epochs 25--100** | **mixed fp16/fp32** | dependence is disclosed at 512--514; epoch and precision mixing are **not** disclosed in the main subsection | invalid as an ordinary independent 21-point Spearman analysis |
| 549--553 | fixed-head oracle reproduction | matched at epoch 100 | intentionally fp16 versus fp32 | **yes**, explicitly the purpose | valid robustness check, not an arm effect |
| 655--685 | all-probes appendix table | mixed epochs 25--100 | mixed fp16/fp32 | caption instead says “Protocol identical throughout” | misleading inventory caption/column |
| 727--760 | stale appendix race trend | mixed epochs | mixed fp16/fp32 | unmatched epochs disclosed at 757--759; precision not disclosed | stale and confounded |
| 771--788 | stale appendix severity ranges | mixed epochs | mixed fp16/fp32 | inherited epoch caveat only; precision not disclosed | stale and confounded |
| 842--851 | fine-tuned result table | all pretraining epoch 100 | all stored float16 | separate fine-tune table | valid |

There is **no undisclosed epoch mismatch in the primary pairwise arm
contrasts**. The main unacknowledged mismatch is the subgroup correlation,
which pools checkpoints from different epochs and precisions.

The appendix all-probes table also misstates stored prediction precision under
this audit's definition:

- Lines 666--671 and 673--675 label nine float16 long-horizon predictions
  `fp32`.
- Lines 680--681 label two float32 anatomy-v2 predictions `fp16`.

The heading says “target precision,” apparently mixing pretraining EMA-target
precision with probe/evaluation precision. That is a different quantity from
the manuscript's numerical-precision discussion and from the task's
dtype-defined precision. The paper must use two separately named columns if it
intends to report both.

## G. Subgroup claims

### Exact recomputation from `p7b`'s underlying rows

Using the 21 rows currently embedded in `p7b_gap_trend.json`:

| Attribute | Worst group counts | Spearman \(\rho\) vs overall AUC | \(p\) | Gap range |
|---|---:|---:|---:|---:|
| sex/gender | female 21/21 | -0.6766234 | 0.0007567 | [0.0272394, 0.0459424] |
| race | black 21/21 | +0.4818182 | 0.0269872 | [0.0475226, 0.0935356] |
| ethnicity | hispanic 21/21 | +0.4181818 | 0.0592258 | [0.0223014, 0.0660529] |
| severity | mild \((-6,-2]\) 21/21 | -0.4103896 | 0.0646235 | [0.1286300, 0.1399821] |

Thus the main-text claims about unanimous worst groups, the displayed 21-row
race and sex correlations, and severity gap spread are arithmetically faithful
to `p7b`. The severity spread is:

\[
0.1399820842-0.1286300153=\mathbf{0.0113520689},
\]

which correctly renders as 0.0114.

### Inclusion failure

`p7b_gap_trend.py` excludes only rows whose subgroup JSON status is literally
`RETRACTED`. It never joins the authoritative inventory status. The subgroup
input marks the following as `OK`, so `p7b` includes them:

- `frozen_meanpool_bridge_ep75`
- `frozen_meanpool_bridge_ep92`

Yet `p1b_full_inventory.json` marks both `excluded`, and manuscript lines
800--807 say these probes are excluded from analysis because of the
EMA-target-precision splice. The reported `NprobesSub = 21` is therefore
inconsistent with the paper's own exclusion rule. Removing those two rows gives
19 valid probes:

| Attribute | 21-row `p7b` result | Exclusion-consistent 19-row result |
|---|---:|---:|
| race | \(\rho=+0.481818,\ p=0.026987\) | **\(\rho=+0.510526,\ p=0.025516\)** |
| sex/gender | \(\rho=-0.676623,\ p=0.000757\) | **\(\rho=-0.714035,\ p=0.000595\)** |
| ethnicity | \(\rho=+0.418182,\ p=0.059226\) | **\(\rho=+0.350877,\ p=0.140775\)** |
| severity | \(\rho=-0.410390,\ p=0.064623\) | **\(\rho=-0.521053,\ p=0.022161\)** |

The worst group remains unanimous for all four requested attributes, and the
severity range/spread is unchanged because the excluded rows are not its
extrema. However, the severity trend changes from nominally non-significant to
nominally significant, so the inclusion bug changes an inferential conclusion.

### Manuscript inconsistency

The appendix subgroup section (lines 727--788) is stale:

- it says 12 probes/arms rather than current 21 or valid 19;
- it reports race \(\rho=0.427,\ p=0.167\), conflicting with both current and
  exclusion-consistent calculations;
- it reports severity gap range 0.1306--0.1394 and spread 0.009 rather than
  0.1286300--0.1399821 and 0.0113521;
- it says excluded probes are excluded “here as everywhere,” which is false for
  the current `p7b`-driven main text.

### Dependence and hedging

The probes are **not independent**. They reuse the same 3,000 cases, share an
epoch-25 ancestor and seed, and repeatedly probe checkpoints from the same arm
trajectories. Standard Spearman p-values treat rows as independent and are
therefore not calibrated here and may be anti-conservative.

The manuscript appropriately acknowledges non-independence at lines 512--514
for the worst-group ordering and calls that result descriptive. The appendix
also discusses shared encoders for its obsolete 12-row subset. It does **not**
adequately hedge the main race/sex trend inference at lines 522--529: it reports
ordinary p-values and says the gap “widens as the masking policy gets better”
without explaining the mixed epochs/precisions or invalid row independence.
These trends should be labelled descriptive, or recomputed at one matched epoch
with arm/trajectory-level uncertainty.

## Discrepancy table

This audit counts **13 distinct discrepancies or transcription risks**.

| # | Claim/artifact | Manuscript/artifact value | Independent result | Magnitude | Severity |
|---:|---|---|---|---:|---|
| 1 | All `D:\...\frozen_*` predictions are fp32 | fp32 | three `frozen_meanpool_mirage_ep*` files are float16 | 3 files | minor metadata |
| 2 | `p1b` enumerates then de-duplicates every prediction set | total 31, after de-dup 31 | physical total 34, unique total 31 when omitted D copies are included | 3 omitted paths | minor inventory |
| 3 | Abstract: all nine contrasts are against null and exclude zero (46--47) | 9/9 null contrasts | 6 null contrasts; 6/6 exclude zero; 7/9 total exclude zero | 3 mislabeled; 2 of 9 cross zero | **major** |
| 4 | Every result number is macro-generated | seven hand-typed literals at 380, 492--493, 552 | 7 transcription-risk literals | 7 values | minor risk |
| 5 | `p7b` obeys exclusions | 21 probes | 19 after removing explicitly excluded ep75/ep92 bridge probes | 2 probes | **major** |
| 6 | Race trend from valid probe set | +0.481818, p=0.026987 | +0.510526, p=0.025516 | \(\Delta\rho=0.028708\) | moderate |
| 7 | Sex trend from valid probe set | -0.676623, p=0.000757 | -0.714035, p=0.000595 | \(\Delta\rho=0.037412\) | moderate |
| 8 | Severity has no significant trend in `p7b` | -0.410390, p=0.064623 | -0.521053, p=0.022161 after exclusions | \(\Delta\rho=0.110663\); nominal conclusion flips | **major** |
| 9 | Appendix subgroup sample count (727--788) | 12 probes/arms | current 21; exclusion-consistent 19 | 7 versus valid set | **major/stale** |
| 10 | Appendix race correlation (755--760) | rho=0.427, p=0.167 | rho=0.510526, p=0.025516 | \(\Delta\rho=0.083526,\ \Delta p=0.141484\) | **major/stale** |
| 11 | Appendix severity range/spread (786--788) | 0.1306--0.1394; spread 0.009 | 0.128630--0.139982; spread 0.011352 | spread +0.002352 | moderate/stale |
| 12 | All-probes appendix precision/protocol (655--685) | 11 dtype-opposite labels; “protocol identical” | nine listed fp32 are float16; two listed fp16 are float32 | 11 rows | **major metadata** |
| 13 | Main subgroup trend is precision/epoch comparable (501--539) | no mismatch caveat | epochs 25--100 and both float16/float32 are pooled | two confounded axes | **major inference** |

## Must fix before submission, in priority order

1. **Fix `p7b_gap_trend.py` to join `p1b_full_inventory.json` and include only
   `status == primary`; regenerate `p7b`, `auto_numbers.tex`, and the paper.**
   The valid count is 19, not 21. Do not rely on the subgroup JSON's `OK`
   status.
2. **Replace the stale appendix subgroup section** with the same generated,
   exclusion-consistent data used by the main text. There must be one probe set,
   one count and one correlation everywhere.
3. **Treat subgroup Spearman p-values as descriptive** unless inference is
   redone at a matched epoch with arm/trajectory-level dependence handled.
4. **Correct abstract lines 46--47** to “all six contrasts against the null
   exclude zero.” Do not say all nine.
5. **Separate probe/evaluation dtype from pretraining EMA-target precision** in
   the all-probes appendix and remove “protocol identical throughout.”
6. Convert the seven hand-typed Results literals into generated macros.
7. Extend `p1b` to enumerate the three D-drive envelope copies and mark them as
   byte-identical duplicates, so its physical and de-duplicated totals are
   auditable.

No correction is required to the primary AUCs, DeLong implementation, paired
bootstrap method, or epoch-100 headline contrast.
