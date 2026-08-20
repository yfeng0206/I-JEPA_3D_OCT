# Subgroup / fairness findings (zero-GPU, post-hoc)

Status labels: **[MEASURED]** observed from a stored artifact; **[INFERRED]**
arithmetic or interpretation; **[ASSUMED]** premise not identified by the data.

Produced by `scripts/subgroup_analysis.py` →
`D:\jepa_phase0\reports\subgroup\subgroup_auc.{json,csv}`.
Figure: `scripts/make_fairness_figure.py` → `figures/fig6_subgroup_disparity.{pdf,png}`.

## 1. How the join was established, and why it is trustworthy

No probe stored a subject identifier — `test_predictions.npz` contains only
`labels` and `probs` (n=3000). The join to demographics is therefore
**order-based**:

- `src/eval_downstream.py:372` and `:584` build the test loader with
  `shuffle=False`, and features are cached in dataset order.
- Test volumes are `D:\jepa_phase0\fairvision-glaucoma\data\Test\data_*.npz`,
  enumerated in sorted filename order. Each holds `oct_bscans` of shape
  **200x200x200** (uint8); the probe samples `num_slices: 100` of those 200
  B-scans (all 26 configs agree on 100).
- Demographics come from
  `D:\jepa_phase0\fairvision-glaucoma\metadata\data_summary_glaucoma.csv`
  (columns `filename, age, gender, race, ethnicity, language, maritalstatus,
  md, glaucoma, use`; 10,000 rows, 3,000 of them `use == test`).

**[MEASURED] The join is verified four independent ways, not assumed.**

1. *Label reconstruction.* For every arm the script rebuilds the label vector
   from the CSV `glaucoma` column and requires exact agreement with the stored
   `labels`. Agreement: **3000/3000 on all 16 probe directories**. An arm
   failing this check is refused, not reported. Prevalence is ~49%, so any
   misalignment would break agreement almost everywhere.
2. *Filename ordering.* The sorted `data_*.npz` listing of `Test/` is
   **element-for-element identical** to the `use == test` rows of the CSV
   (3000 vs 3000, first entries `data_07001.npz`, `data_07002.npz`, ...).
3. *In-volume demographics.* Each FairVision `.npz` **carries its own**
   `race`, `male`, `hispanic`, `language`, `maritalstatus`, `glaucoma` fields,
   entirely independent of the CSV. Decoding those and comparing to the CSV in
   sorted-filename order gives **race 3000/3000** and **gender 3000/3000**.
4. *Label agreement against the volumes themselves.* npz `glaucoma` vs CSV
   `glaucoma` agrees **3000/3000**, with class counts matching exactly
   (1,534 negative / 1,466 positive).

**Gotcha worth recording:** the CSV stores `glaucoma` as the strings
`'yes'/'no'` while the npz stores `1/0`. A naive comparison returns *zero*
agreement and looks exactly like a catastrophic label inversion. It is purely
an encoding difference; after mapping, agreement is perfect. Anyone re-running
this check should map before comparing.

## 2. What can and cannot be treated as an independent replicate

**[MEASURED]** 16 probe directories carry saved predictions. Four of them —
`frozen_cover_random_ep30/ep50/ep75/ep100` (AUC 0.8558 / 0.8590 / 0.8612 /
0.8607) — are the **RETRACTED** COVER campaign, pretrained with
`enc_truncate: window` and `amp_target: true`
(`docs/experiments/masking/cover_random_campaign.md#L22`). They are tagged
`RETRACTED` in the JSON and excluded from every claim below.

**[INFERRED]** The remaining 12 probes come from only **5 pretraining runs**:
envelope (ep30/50/75/100), blob (ep35/40/50), COVER f0.21 (ep27/30/34),
anatomy (ep30), shared fork (ep25). Probes within a family share one encoder
and one pretraining seed, so **n = 5 independent runs, not 12**. Every claim
below is limited by that.

## 3. Finding A — a consistent racial performance ordering

**[MEASURED]** Black patients are the worst-served race group in **12 of 12**
non-retracted probes, and in 16 of 16 including the retracted ones.

Race composition of the test split: white n=2318, Black n=431, Asian n=251.

Representative arm (blob ep50), 2000-sample bootstrap CI:

| group | n | n_pos | AUC | 95% CI |
|---|---:|---:|---:|---|
| Asian | 251 | 118 | 0.9058 | [0.8708, 0.9424] |
| white | 2318 | 1080 | 0.8672 | [0.8515, 0.8809] |
| Black | 431 | 268 | **0.8123** | [0.7711, 0.8507] |

**[LIMITATION]** With n=431 the Black CI spans ±0.04 and overlaps the white CI
in every arm. **No single arm's gap is individually significant.** The
defensible claim is the *consistency of the ordering across 12 independently
fitted probes spanning 5 encoders*, not the magnitude of any one gap.

## 4. Finding B — mask policy modulates the racial gap ~2x

**[MEASURED]** Racial AUC gap (max − min across race groups), non-retracted:

| arm | overall AUC | race gap |
|---|---:|---:|
| COVER f0.21 ep34 | 0.8571 | **0.0475** (smallest) |
| COVER f0.21 ep30 | 0.8522 | 0.0499 |
| envelope ep30 | 0.8539 | 0.0528 |
| COVER f0.21 ep27 | 0.8483 | 0.0565 |
| envelope ep50 | 0.8761 | 0.0598 |
| envelope ep100 | 0.8807 | 0.0632 |
| shared fork ep25 | 0.8487 | 0.0634 |
| envelope ep75 | 0.8803 | 0.0638 |
| anatomy ep30 | 0.8583 | 0.0652 |
| blob ep35 | 0.8661 | 0.0749 |
| blob ep40 | 0.8683 | 0.0772 |
| blob ep50 | 0.8654 | **0.0935** (largest) |

**[MEASURED]** The gap varies by a factor of **1.97** (0.0475 → 0.0935) across
policies whose overall AUCs differ by only 0.032.

**[MEASURED]** Spearman(overall AUC, race gap) over the 12 valid probes is
**rho = 0.427, p = 0.167** — *not* significant. **Do not claim "more accurate
models are less fair"; the data do not support it.**

**[INFERRED — moderate confidence]** Two orderings are suggestive and mutually
consistent: the blob family (near-pure anatomy targets, the arm that also
collapses on overall AUC) has the three largest gaps, and its gap *widens*
monotonically with training (0.0749 → 0.0772 → 0.0935 at ep35/40/50); the
COVER f0.21 family has the smallest gaps. **[LIMITATION]** Family and policy
are perfectly confounded with pretraining seed (one run each), and epoch is not
matched across families, so this is an association, not a causal effect of
target composition.

## 5. Finding C — the mild-disease penalty is large and policy-invariant

`md` (visual-field mean deviation) **defines** the FairVision label: all 1,466
test volumes with md <= -2 are positive and all 1,534 with md > -2 are
negative (corr(md, label) = -0.726; bins are perfectly class-pure). **[MEASURED]**
Therefore md **cannot** be used as an ordinary subgroup axis — within-bin AUC is
undefined. This is a real trap: naive stratification silently yields nothing.

Instead each positive severity stratum is scored against the **shared pool of
all 1,534 negatives**:

| stratum | n positives | AUC range across the 12 valid arms |
|---|---:|---|
| severe (md <= -12) | 334 | 0.928 – 0.956 |
| moderate (-12 < md <= -6) | 460 | 0.870 – 0.910 |
| mild (-6 < md <= -2) | 672 | 0.791 – 0.823 |

**[MEASURED]** The severe→mild gap is **0.1306 – 0.1394 in every one of the 12
arms** — a spread of just 0.009 across policies that differ by 0.032 in overall
AUC and 2x in racial gap.

**[INFERRED — high confidence]** Mask policy does **not** address the
mild-disease penalty. Early/moderate disease — the clinically actionable
window, and the majority of positives (1,132 of 1,466) — is where every encoder
is weakest, and no target-placement change tested here moves it.

## 6. Why this matters for the paper

**[INFERRED]** These results are *orthogonal* to the anatomy-vs-rectangle claim
that the ep50 reversal undermined. They stand on their own: they use only
already-saved predictions, cost zero GPU, and are exactly what the venue's
disparities/fairness topics ask for. Finding C in particular is the most
robust result in the whole programme — a tight, monotone, fully consistent
effect across every arm measured.

## 7. Honest limitations to state in the paper

1. One pretraining seed per policy; 5 encoders total. Policy is confounded with
   seed.
2. Epoch is not matched across families (blob ep35/40/50 vs envelope ep30–100).
3. Black n=431, Asian n=251 — no individual gap is significant.
4. Single dataset, single test split, reused across the programme's history.
5. Self-reported race/ethnicity categories from FairVision; the analysis
   inherits their coding and cannot speak to intersectional subgroups at these
   sample sizes.
6. Frozen mean-pool linear probe only; disparities could differ under
   fine-tuning or a stronger head.
