# Attribution analysis

## Decision

- **Claim A, “anatomy-only beats the full image”: CORROBORATIVE-ONLY.**
  [MEASURED] The point estimates have the stated direction for the centroid
  (`oracle`) and random arms, but paired case-bootstrap intervals include zero
  for both. The result is therefore not evidence of an improvement for either
  fixed model. The random direction is consistent with the paper's stronger,
  already-reported region-pooling result, so it is corroboration rather than a
  new claim.
- **Claim B, “a background patch carries more attribution than an anatomy
  patch”: ARTIFACT.** [MEASURED] The raw mean-absolute quantity is larger for
  background in three arms, not four. [INFERRED] Its sign reverses after the
  requested scale normalisation, while a second standardised-magnitude check is
  essentially equal between groups. The raw comparison is driven by scale and
  is not evidence that a background patch carries more information.
- **Paper action: add nothing.** [INFERRED] Neither claim warrants a new body or
  appendix assertion. In particular, these results should not be inserted into
  the existing occlusion appendix: the current artifacts are an exact additive
  decomposition of fixed linear heads, not the occlusion experiment described
  there.

## What was actually measured

[MEASURED] `scripts\patch_attribution.py` decomposes a
LayerNorm-then-linear logit exactly into signed patch contributions. For each
case, `logit = C_an + C_bg + constant`; the maximum reported reconstruction
residual is `5.066394805908203e-07`.
Sources:
`C:\Users\Gary\Desktop\jepa\scripts\patch_attribution.py`;
`D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.json`;
`D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.json`.

[MEASURED] The four NPZ files retain `C_an`, `C_bg`, `logit`, and `label` for
each case. Their label vectors are identical: `n=1000`, with `495` positive and
`505` negative cases. Sources:
`D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.npz`;
`D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.npz`;
`D:\jepa_phase0\reports\patch_attribution\envelope_ep50_attrib.npz`;
`D:\jepa_phase0\reports\patch_attribution\blob_ep50_attrib.npz`.

[INFERRED] Calling these files “occlusion attribution” would conflate two
different analyses. The manuscript's occlusion appendix zeroes slice, window,
or patch tokens for three fine-tuned probes. The present files analytically
partition the score of four fixed epoch-50 linear heads. Source for the
manuscript method:
`C:\Users\Gary\Desktop\jepa\paper\genai4health2026\main_submission.tex`
(section `Occlusion attribution: where the classifier looks`). Source for the
present method:
`C:\Users\Gary\Desktop\jepa\scripts\patch_attribution.py`.

## Claim A: paired test of anatomy-only versus full

[MEASURED] I applied the paper's paired, class-stratified case bootstrap: the
same resampled indices were used for `C_an` and `logit`, with `10,000`
resamples and RNG seed `20260822`. The interval is the percentile interval on
the paired AUC difference. Sources for the established method and constants:
`C:\Users\Gary\Desktop\jepa\autopilot\p1_paired_stats.py`;
`C:\Users\Gary\Desktop\jepa\autopilot\p1c_stats.py`;
`C:\Users\Gary\Desktop\jepa\paper\genai4health2026\auto\auto_numbers.tex`.

| arm | anatomy-only AUC | full AUC | anatomy minus full | paired bootstrap interval | two-sided bootstrap p | source arrays |
|---|---:|---:|---:|---:|---:|---|
| centroid (`oracle`) | 0.864250 | 0.863214 | +0.001036 | [-0.006733, +0.008861] | 0.7744 | `D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.npz` |
| random | 0.863850 | 0.855250 | +0.008601 | [-0.003013, +0.020114] | 0.1438 | `D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.npz` |
| envelope | 0.865403 | 0.865235 | +0.000168 | [-0.010365, +0.010797] | 0.9648 | `D:\jepa_phase0\reports\patch_attribution\envelope_ep50_attrib.npz` |
| blob | 0.846425 | 0.856650 | -0.010225 | [-0.017658, -0.002768] | 0.0064 | `D:\jepa_phase0\reports\patch_attribution\blob_ep50_attrib.npz` |

[MEASURED] Correlated-ROC DeLong checks agree with the bootstrap: centroid
`p=0.7946`, random `p=0.1431`, envelope `p=0.9753`, and blob `p=0.00655`.
The DeLong implementation is
`C:\Users\Gary\Desktop\jepa\autopilot\p1_paired_stats.py`; the score and label
arrays are the four NPZ paths in the table.

[INFERRED] The centroid margin is plainly inside case-sampling noise. The
larger random margin is also inside case-sampling noise. Moreover, the
four-arm version of Claim A is false: blob provides statistically resolved
evidence in the opposite direction for this fixed checkpoint and case set.

### Why the random point margin is larger

[MEASURED] In random, anatomy-only AUC exceeds background-only AUC by
`0.016890` (`0.863850 - 0.846961`) and the anatomy/background contribution
correlation is `0.906351`. Source:
`D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.json`.

[MEASURED] In centroid, the corresponding AUC gap is only `0.004812`
(`0.864250 - 0.859438`) and the correlation is higher at `0.953939`. Source:
`D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.json`.

[INFERRED] Because the full score is the sum of the two contribution streams,
adding a less discriminative and less rank-aligned background stream harms the
random point estimate more. This explains the observed asymmetry
descriptively; it does not establish a causal property of the masking policy,
and the paired interval shows that the random degradation is not resolved in
these `1000` cases. Source for `1000`:
`D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.json`.

## Claim B: scale normalisation

[MEASURED] The raw statistic is the mean absolute cell contribution in each
mask group. It is not below one in every arm: blob's anatomy/background ratio
is `1.154439`, meaning anatomy is larger there. Source:
`D:\jepa_phase0\reports\patch_attribution\blob_ep50_attrib.json`.

[MEASURED] For centroid, the raw anatomy/background ratio is
`0.858314`, while the aggregate contribution standard deviations are
`0.305190` for anatomy and `1.224306` for background. Source:
`D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.json`.

[MEASURED] For random, the raw ratio is `0.787351`, while the standard
deviations are `0.266142` for anatomy and `1.157493` for background. Source:
`D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.json`.

[INFERRED] The requested scale-adjusted ratio is

`R_SD = (per_patch_anatomy / std_anatomy) /
        (per_patch_background / std_background)
      = R_raw * std_background / std_anatomy`.

| arm | raw anatomy/background | background/anatomy SD | `R_SD` | source |
|---|---:|---:|---:|---|
| centroid (`oracle`) | 0.858314 | 4.011618 | 3.443226 | `D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.json` |
| random | 0.787351 | 4.349156 | 3.424313 | `D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.json` |
| envelope | 0.939454 | 3.626358 | 3.406797 | `D:\jepa_phase0\reports\patch_attribution\envelope_ep50_attrib.json` |
| blob | 1.154439 | 2.599862 | 3.001382 | `D:\jepa_phase0\reports\patch_attribution\blob_ep50_attrib.json` |

[INFERRED] Thus the background advantage does not survive the specified
normalisation; the direction reverses in every arm.

[MEASURED] A second, less cell-count-dependent diagnostic uses the saved
case-level group sums: compute
`mean(abs(C_group - mean(C_group))) / std(C_group)` and compare anatomy with
background. The anatomy/background ratios are `0.995540` for centroid,
`0.996655` for random, `1.008878` for envelope, and `0.999778` for blob.
Sources, respectively:
`D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.npz`;
`D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.npz`;
`D:\jepa_phase0\reports\patch_attribution\envelope_ep50_attrib.npz`;
`D:\jepa_phase0\reports\patch_attribution\blob_ep50_attrib.npz`.

[INFERRED] After centring and standardising, the two group distributions have
essentially the same absolute spread. The unstandardised mean absolute value
therefore reflects their different scales, not evidence that a typical
background patch contains more decision information.

[PENDING] A direct per-cell or per-patch-position variance normalisation cannot
be computed from the surviving files. They save case-level group sums and
already-averaged patch maps, not the case-by-cell contribution tensor. Source
for the fields that were saved:
`C:\Users\Gary\Desktop\jepa\scripts\patch_attribution.py`; surviving files:
`D:\jepa_phase0\reports\patch_attribution\*_attrib.npz`.
This missing tensor does not rescue Claim B: both defensible normalisations
available from the archived data eliminate its advertised direction.

[MEASURED] The reported cell totals are `1,495,687` anatomy cells and
`4,904,313` background cells per arm. Source:
`D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.json`.
[INFERRED] These cells are repeated measurements nested within the same
`1000` cases, not millions of independent observations. Source for `1000`:
`D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.json`.

## Is contribution correlation the same as feature reconstructibility?

[MEASURED] No. The visually similar values compare different arms and
different statistics:

- `0.953939` is centroid's Pearson correlation between two scalar signed
  contribution sums, `C_an` and `C_bg`, across test cases. Its square is
  `0.910000`. Source:
  `D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.json`.
- `0.952222` is random's held-out multi-output Ridge `R2` when reconstructing a
  pooled background feature vector from a pooled anatomy feature vector.
  Source:
  `C:\Users\Gary\Desktop\jepa\autopilot\bgsig\a2_region_incremental.json`.
- Random's attribution correlation is `0.906351`, not `0.953939`. Source:
  `D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.json`.

[INFERRED] The near match between centroid `r=0.953939` and random
`R2=0.952222` is numerical coincidence, not the same result reported twice.
Qualitatively, however, both analyses say that anatomy-position and
background-position representations carry a large shared component. The
attribution correlation therefore corroborates the existing redundancy result;
it does not establish a new mechanism.

## What the paper already contains

[MEASURED] The section `Occlusion attribution: where the classifier looks`
reports slice, window, and patch leave-one-out occlusion for three fine-tuned
probes. It reports diffuse attribution, an orientation/storage artifact, and
error curves with lower amplitude. It does not report the four-arm
anatomy/full AUC decomposition, per-group patch means, per-group standard
deviations, or anatomy/background contribution correlations. Exact searches
for the rounded values and field names found no occurrence. Source:
`C:\Users\Gary\Desktop\jepa\paper\genai4health2026\main_submission.tex`.

[MEASURED] The separate background appendix already reports that pooled
background features are `95.2%` linearly reconstructible from pooled anatomy
features and that appending background lowers random-arm AUC by `0.0076`, with
interval `[-0.0139, -0.0012]`. Source:
`C:\Users\Gary\Desktop\jepa\paper\genai4health2026\main_submission.tex`.
The unrounded source values are in
`C:\Users\Gary\Desktop\jepa\autopilot\bgsig\a2_region_incremental.json`.

[INFERRED] The exact attribution numbers are absent, but their only defensible
scientific reading is already represented in the background appendix. Adding
them would duplicate that conclusion while mixing an additive decomposition
into an appendix explicitly framed as occlusion.

## Scope and publication recommendation

[MEASURED] The analysis has one epoch-50 checkpoint and one fitted head per arm,
all evaluated on the same `1000` cases. Sources:
`D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.json`;
`D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.json`;
`D:\jepa_phase0\reports\patch_attribution\envelope_ep50_attrib.json`;
`D:\jepa_phase0\reports\patch_attribution\blob_ep50_attrib.json`.

[INFERRED] Paired resampling supports statements about case-sampling
uncertainty conditional on these fixed checkpoints and heads. It does not
measure pretraining-seed variation, head-fitting variation, checkpoint
selection variation, stability across epochs, expected policy performance, or
transport to another cohort. Patch and cell counts do not increase the
independent sample size.

[INFERRED] **No sentence should be added to the body or appendix.** Claim A is
unresolved by its paired intervals and is already qualitatively covered by a
stronger background-pooling analysis. Claim B is a scale artifact and its
four-arm premise is false. The correlation is useful as an internal
cross-check, but only corroborates the paper's existing redundancy finding.
