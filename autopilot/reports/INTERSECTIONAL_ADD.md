# Intersectional fairness: false statements corrected, analysis added

Target: `paper/genai4health2026/main_submission.tex` (only file edited).
Artifacts read (READ ONLY): `D:\jepa_phase0\reports\subgroup\intersectional_auc.json` and `.csv`,
`D:\jepa_phase0\reports\subgroup\subgroup_auc.csv`,
producer `paper/genai4health2026/scripts/intersectional_analysis.py`,
reducer `paper/genai4health2026/scripts/intersectional_claims.py`,
background `autopilot/reports/UNREPORTED_INVENTORY.md`.
Nothing was written under `D:\jepa_phase0`. Nothing was committed.

## 1. Re-verification of every fact before use

The reducer was re-run (`D:\jepa_phase0\.venv\Scripts\python.exe intersectional_claims.py`) and every
per-cell number was independently recomputed from the JSON at full precision. All MEASURED unless noted.

| Claim | Re-verified value | Status |
|---|---|---|
| Non-retracted arms in the artifact | 18 (of 22 arms present) | MEASURED |
| black x female worst cell | 18/18 | MEASURED |
| asian x male best cell | 18/18 | MEASURED |
| female below male within race | 54/54 | MEASURED |
| black below white within gender | 36/36 | MEASURED |
| Mean max-min gap: gender / race / race x gender | 0.0340 / 0.0653 / 0.1046 | MEASURED |
| Intersectional gap exceeds marginal race gap | 18/18 | MEASURED |
| Worst-cell understatement vs marginal race gap | 60.1% | INFERRED (arithmetic on the two means) |
| Additive prediction vs observed | 0.0993 vs 0.1046, ratio 1.053 | INFERRED |
| ep100 oracle-minus-random per cell | asian x female +0.0144, asian x male +0.0164, black x female +0.0114, black x male +0.0228, white x female +0.0121, white x male +0.0099 | MEASURED |
| Cell n | 123, 128, 250, 181, 1343, 975 | MEASURED |
| ep50 exception | oracle-random asian x female = -0.0020 | MEASURED |
| envelope-random negatives | 3 of 6 at ep75; **2 of 6 at ep100** (asian x female is +0.0001, not negative) | MEASURED |
| Per-cell paired intervals | none in the artifact (per-arm CIs only) | PENDING |

Two corrections to the brief, both made against the artifact:

1. The brief said envelope-minus-random is negative in 3 of 6 cells at **both** ep75 and ep100.
   At ep100 it is negative in **2** of 6 (asian x male -0.0100, black x female -0.0040); asian x
   female is +0.000135. The paper states 3 of 6 at ep75 and 2 of 6 at ep100, with the +0.0001 shown.
2. The paper's "envelope" arm is `frozen_meanpool_mirage_*` and its "\ArmBest" (centroid) arm is
   `sweep_oracle_*`; the mapping was confirmed against `auto/auto_numbers.tex`
   (`\AUCEnvelopeEpHundred` = 0.8807 = `frozen_meanpool_mirage_ep100`, `\AUCOracleEpHundred` = 0.8855
   = `sweep_oracle_ep100`). The paper text uses `\ArmBest` throughout, never the raw key `oracle`.

Also verified and stated in the paper: the 18 arms are a **strict subset** of the paper's
`\NprobesSub{}` = 23 subgroup probes; the five absent are `frozen_meanpool_blob_fp32_ep75` and
`frozen_meanpool_cover_f021_ep{50,73,75,100}`. Every count in the new subsection is out of 18, not 23.
The `skipped` list is empty for all arms, so no cell was dropped for size or single class.

## 2. The three false statements, corrected

| Line (before) | Was | Now |
|---|---|---|
| L1080 (App. E Limitations) | "at these subgroup sizes we cannot speak to intersectional groups" | "A race-by-sex intersectional breakdown *is* reported, in Appendix~\ref{app:intersectional}; its limits are that two of its six cells hold fewer than 130 cases and that it carries no paired per-cell intervals, so it describes the disparity and tests no per-cell change." |
| L1083 (same paragraph) | "we report no subgroup calibration and no intersectional breakdown, so this is not a complete fairness evaluation" | "we report no subgroup calibration and no subgroup predictive values, so this is not a complete fairness evaluation" |
| L1498 (App. Ethics) | "it contains no predictive-value analysis and no intersectional breakdown" | "...and the race-by-sex breakdown of Appendix~\ref{app:intersectional}, it contains no predictive-value analysis and no subgroup calibration, and the intersectional cells carry no paired per-cell intervals, so no per-cell change there is tested against zero." |

No existing limitation was deleted. The "no subgroup calibration" limitation was preserved in both
places (it had been bundled into the same sentence as the false claim), and "no predictive-value
analysis" was preserved. Two *new* limitations were added: small cells, and no per-cell intervals.

One further precision fix in the Ethics paragraph: "every subgroup point estimate rises" ->
"every subgroup point estimate rises **at the matched epoch**", because the rise is not universal
across epochs once the cells are examined (see below).

## 3. What was added

New appendix subsection **E.1 "Intersectional breakdown: race by sex"**
(`\label{app:intersectional}`), placed at the end of Appendix E, containing:

- **Table 9** (`\label{tab:intersectional}`): all six race-by-sex cells at matched epoch 100, with
  n, n_+, and AUC plus per-arm bootstrap percentile interval for RANDOM, ENVELOPE and CENTROID, the
  delta CENTROID-RANDOM, and the max-min gap row (0.1050 / 0.0990 / 0.1100). Every value quoted from
  `intersectional_auc.json`.
- **"The ordering compounds, and it never reorders."** The 18/18, 54/54 and 36/36 counts, explicitly
  framed as a consistency check across checkpoints sharing one test split, one epoch-25 ancestor and
  one probe seed - "not 18 replications" - with no p-value attached.
- **"The marginal race gap understates the worst cell."** Mean gaps 0.0340 / 0.0653 / 0.1046,
  18/18 exceedance, 60.1% understatement labelled as arithmetic on the two means rather than a
  separate measurement; the matched-epoch instance (0.0717 marginal race vs 0.1100 race-by-sex under
  CENTROID; worst cell 0.8309 sits 0.0163 below the marginal black figure `\RaceOracleBlack`=0.8472).
  Additivity stated as close to additive: 0.0993 predicted vs 0.1046 observed, ratio 1.053,
  explicitly "not evidence of a super-additive interaction".
- **"Not every cell improves."** States the exception plainly: all six cells rise at ep100, but at
  ep50 asian x female falls by 0.0020, and envelope-minus-random is negative in three of six cells at
  ep75 and two of six at ep100 (with asian x female at +0.0001). Concludes no policy raises every
  intersectional cell at every epoch.
- **"What this does not establish."** No paired per-cell bootstrap exists; every delta is a
  difference of point estimates and none is separated from zero; per-cell delta significance is
  labelled **PENDING** and noted as computable from retained per-arm predictions. Small cells
  (n=123, n=128) named. No fairness-intervention claim.

**Body change (net zero lines).** One clause in Section 5.5, replacing
"Appendix~\ref{app:subgroup} gives every stratum." with
"Appendix~\ref{app:subgroup} gives every stratum, Appendix~\ref{app:intersectional} the race-by-sex
cells, whose gap exceeds the race margin in every arm."
A first, longer version of this clause pushed the body to 10 pages (References heading y went
72.79 -> 112.21); it was trimmed until the References heading returned to y=72.789, byte-identical to
the pre-edit baseline, i.e. the body addition costs **zero** net lines. The 60.1% figure itself stays
in the appendix; it did not fit in the body.

## 4. Gate results (from `C:\Users\Gary\Desktop\jepa`)

```
p13_build_zip.py      6/6 PASS, total 33 pages, main content 9 (limit 9), ALL_PASS = True
check_manuscript.py   RESULT: PASS - labels 55, refs 55, dangling 0, undefined macros 0, duplicates 0
p15_verify_numbers.py RESULT: PASS - 20 AUC macros verified, no cross-arm attribution
```

Both `check_manuscript.py` warnings are pre-existing and unrelated (233 unused generated macros;
the literal "epoch-92 probes" on L301). Two new labels were added, `app:intersectional` and
`tab:intersectional`; both are referenced (`app:intersectional` from the body Section 5.5, the
Appendix E Limitations paragraph and the Ethics appendix; `tab:intersectional` twice within E.1).
Dangling refs remain 0. Total page count rose 32 -> 33; the extra page is appendix, which is
unconstrained. Table 9 was checked visually in the rendered PDF (page 19) and fits the text width.

## 5. Explicitly not claimed

- No per-cell delta is called significant, and no per-cell interval is quoted, because none exists.
  The paper says so twice and marks it PENDING.
- The 18/18 unanimity is called a consistency check, never replication.
- The interaction is called close to additive; no super-additivity is claimed.
- "Every group improves" is not extended to intersections; the ep50 and envelope exceptions are
  stated with their signed values.
- No emoji, tick or cross symbols were used. `x` in cell names renders as `$\times$`.
