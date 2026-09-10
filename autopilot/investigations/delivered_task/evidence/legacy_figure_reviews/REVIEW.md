# Seven legacy figure review decisions

**Decision: do not mark these seven original whole assets release-verified.**
This review completed source, label, geometry, and pixel checks; it did not merely
inventory hashes. It found one reproducible quantitative mismatch and one
caption/figure-variant mismatch. Four attribution plots lack the retained numeric
inputs needed for verification. The remaining mask grid mixes historical images
with empirical AUC headings.

The candidate is deliberately valid version-1 JSON with **zero figure approvals**.
Issuing `reviewed_historical_illustration` for a quantitative curve/scatter, or for
the unsupported empirical annotations in a mixed figure, would conceal rather
than resolve a blocker. No current whole asset qualifies for that receipt.
This is a bounded finding, not a claim that the original analyses were fabricated.

## Deliverables and audit scope

- `candidate_numeric_reviews.json`: mergeable, schema-valid candidate; source
  fingerprints only, no blanket approval.
- `inspection.json`: exact current asset hashes, exact caption text and
  `numeric_bindings.digest_text` hashes, source keys/pointers/hashes, reviewer,
  rationale, limitations, scientific claim still unresolved, and recommended
  action for **each** asset. Includes source-level numeric/geometry results.
- `review_decisions.json`: authored review decisions.
- `*.ocr.json`: actual local Windows OCR transcripts with word rectangles and
  input hashes. OCR is an observation, not independent empirical source data.
- `audit_legacy_figures.py`, `read_image_text.ps1`, `test_legacy_review.py`:
  independent inspection and duplicate-rejecting merge helper, local OCR helper,
  and validation tests.

All work is inside this evidence directory. No production, manuscript, released
figure, or parent review files were edited. No commits, GPU, network uploads,
new raw-case image exports, or desktop viewers were used. Inspection used local
Windows OCR, PyMuPDF vector/text extraction, Pillow, and NumPy/SciPy pixel
components. No new dependencies were installed. `MPLBACKEND=Agg`.

The five interpretation PNGs are byte-identical to their corresponding existing
files under `results\summary`. That establishes historical asset identity,
**not** the correctness of their curves or numeric annotations. Bounded checks
found no matching attribution arrays or numeric summary tables in `results` or
`archive`, and no interpretation PDF companions in the repository. The documented
`archive\04_interpretability`, `archive\04_interpretability_v2_fp32`, and
`results_presentation` directories are absent. External blob arrays were not
downloaded. The AUC inventory was read only at its exact path documented in
`autopilot\p8_make_assets.py`, not by a broad external search.

## 1. `fig_masking_policies.png`

**Classification:** selected mask images plus empirical AUC annotations. The epoch
and selection index are contextual metadata; AUC remains an empirical result.

Actual OCR reads six headings. Each four-decimal AUC agrees with a semantically
selected historical inventory record:

| Heading | Epoch | Precision in retained inventory | Displayed AUC | Inventory pointer |
|---|---:|---|---:|---|
| random | 100 | fp16 | 0.8746 | `/records/37/auc` |
| oracle | 100 | fp16 | 0.8855 | `/records/31/auc` |
| envelope | 100 | fp16 | 0.8807 | `/records/26/auc` |
| anatomy-v1 | 30 | fp32 | 0.8583 | `/records/6/auc` |
| anatomy-v2 | 40 | fp32 | 0.8683 | `/records/8/auc` |
| cover-f021 | 50 | fp32 | 0.8643 | `/records/17/auc` |

Source key: `legacy_auc_inventory`. The helper pairs image heading columns with
their AUC labels by OCR coordinates **before** selecting inventory records.
Documented OCR confusions (`eplOO`, `anatomy-vl`) are explicitly normalized; this is
not a claim of infallible OCR.

This checks annotations, **not** the historical target-mask coordinates, sampler
execution, selected volume/seed, or the claim that these are exact production
draws. Different epochs must not be compared as matched policy results.

**Minimal resolution:** omit this redundant appendix grid and refer to the
already-reviewed delivered-mask illustration. Alternatively remove the empirical
headings and review a new exact historical-image-only asset, with unrecovered
draw provenance disclosed. A qualifying caption would say:

> Selected historical mask overlays on B-scans from one volume. These examples
> illustrate the displayed shapes only; the original draw seed and execution
> receipt were not retained, and no quantitative performance or coverage
> comparison is inferred.

That is a proposed condition, **not approval** of the present asset or of its
production-sampler claim.

## 2. `fig_precision_paradox.png`

**Classification:** quantitative scatter and fitted/reference lines.

### Independently reproduced mismatch

The retained PDF has no embedded raster images. The helper reads numeric tick
labels and actual tick positions, checks their affine mapping against **every
tick**, associates the five vector circle centers with arm labels, and decodes
their coordinates. Independent dark-spine and colored-component inspection of
the PNG agrees with the PDF marker centers within 1.5 pixels. No chart data were
supplied to that extraction.

| Arm | Plotted purity (%) | Current Table 2 source (%) | Plotted minus source (percentage points) |
|---|---:|---:|---:|
| random | 31.6 | 31.46603448 | +0.13396561 |
| oracle | 41.1 | 40.00872530 | +1.09127484 |
| envelope | 43.5 | 43.30093280 | +0.19906714 |
| cover-f0.21 | 45.3 | 44.18566088 | +1.11433858 |
| anatomy-v2 | 97.3 | 97.09353747 | +0.20646170 |

**All five fail** a one-decimal rounding tolerance of 0.050001 percentage points.
Thus the old scatter cannot be signed off as the current Table 2 geometry.
The historical y-coordinates are 0.864097, 0.874030, 0.876064, 0.864281,
and 0.865386; they agree with the appropriate retained historical AUCs at
six-decimal rounding tolerance. This is a stale **x-source** issue, not an
invented AUC discrepancy.

The dashed line agrees with a least-squares fit through the **old plotted
points**; the null line is the plotted random AUC. These checks do not validate
the plot's scientific interpretation. Raster text is not fully certified:
overlapping labels are incompletely recognized by OCR, although PDF text is
extractable.

### Exact replacement sources

Geometry:
`results\masking\table2_geometry\mask_geometry_600slices_bs1_coverf021_seed42.json`
(`legacy_table2_geometry_seed42`), pointer `/{arm}/hidden_pct_on_anat`.
Map `anatomy-v2` to `anatomy`, and `cover-f0.21` to `cover`.
This is the field independently bound to Table 2 by `numeric_bindings.py`.

AUC:
`D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json`
(`legacy_auc_inventory`).
Original figure rows are `/records/33/auc` (random fp16),
`/records/28/auc` (oracle fp16), `/records/22/auc` (envelope fp16),
`/records/17/auc` (cover fp32), `/records/9/auc` (anatomy-v2 fp32).
For an explicitly all-fp32 epoch-50 replacement, semantically select random
`/records/34/auc`, oracle `/records/29/auc`, envelope `/records/23/auc`, plus the
same cover/anatomy rows. **Do not silently mix protocols.** Pointers are valid for
the source hash in `inspection.json`; select by semantic record fields and reject
duplicates if that source changes.

**Minimal resolution:** regenerate from these real summaries, then independently
validate the new asset. Prefer a descriptive title such as *Delivered target
purity and frozen-probe AUC in the evaluated runs*. State exactly five arms, the
target-union purity definition, epoch/precision, and the lack of retraining
replicates. “Does not separate from the null” requires actual paired uncertainty;
the present no-interval scatter alone cannot establish that. “More accurate
masking does not help” is too general/causal for these confounded implementations.
Alternatively omit the scatter and refer to the verified tables.

## 3. `interp_04_window_occlusion_W7.png`

**Classification:** six quantitative class-mean curves, not an illustration.
The actual title says *“Amplifies single-slice signal 7x while smoothing noise.”*
Legends show the three fine-tuned probes and the glaucoma/healthy counts.

Seven slices is a **protocol setting**; sevenfold signal gain is an **empirical
claim**. The historical PNG and documentation cannot verify the latter, the means,
or the claimed noise reduction. No window/slice attribution arrays were found.
The surviving analysis script documents expected input locations, not retained
numeric values sufficient to validate this figure.

**Minimal resolution:** omit the figure and empirical amplification/completeness
claims unless original series are recovered. A methods-only sentence can say
that contiguous-window deletion is a different perturbation from single-slice
deletion. Caption disclosure alone cannot neutralize the quantitative title and
curves embedded in the image.

## 4. `interp_14_odos_mirror_test.png`

**Classification:** six quantitative cluster-mean panels, correlations, and
cluster counts. Actual OCR reads mirrored correlations **+0.971, +0.988, +0.237**;
the last does not support a universal “near-perfect mirror” claim.

`scripts\odos_mirror_test.py` preserves the L2-normalization, k-means, and
curve-reversal procedure, but not its input arrays. Copying correlations from
the documentation is not numerical verification. Eye laterality is not
ground-truth validated, and clustering/flipping derived from the same curves
cannot establish actual OD/OS storage orientation.

**Minimal resolution:** omit the empirical mirror/ODOS result without retained
series. Keep, at most, the general methodological caution that unknown orientation
can complicate population-averaged attribution. Do not say that the current data
prove an OD/OS artifact or causal anatomy.

## 5. `interp_heatmap_grid.png`

**Classification:** selected B-scan overlays **plus** empirical probabilities and
signed slice contributions, with clinical/descriptive text.

### Confirmed variant/caption discrepancy

- The released image is byte-identical to `results\summary\heatmap_grid.png`.
- It is **not** `results\summary\heatmap_grid_BC.png`; the latter has a distinct
  hash, size, and locally OCR-read title explicitly naming shared scale and
  slice-mean subtraction, with alternative transformation columns.
- The released image's six row annotations correspond to
  `scripts\assemble_heatmap_grid.py`, including prediction values, signed slice
  contributions, and text such as *“RNFL thinning visible”* and *“Localized
  pathology.”*
- The corresponding overlay path in `scripts\interpretability.py` uses
  `vmax = abs(heatmap).max() + 1e-9` separately for each image, linear
  interpolation (`zoom(..., order=1)`), and alpha blending. It does **not**
  subtract the slice mean. `heatmap_BC_comparison.py` is the separate
  shared-scale/mean-subtracted comparison.

This establishes a documented mismatch between the caption's named
transformation and the retained figure variant. It does **not** claim that the
historical executable run or normalization can be uniquely reconstructed from
RGB pixels.

The missing colorbar limits readable magnitude; it does not turn printed
probabilities/contributions into nonnumeric metadata. The selected images
cannot establish population-wide diffuse attribution, slice-versus-patch
agreement, disease localization, or the frozen-policy mechanism. Clinical labels
were not independently reviewed.

**Minimal resolution:** omit this grid and any main-text claim whose support is
only this grid. Conditional historical-illustration retention would require
removing empirical/clinical headings, correcting normalization language, and
reviewing the newly hashed asset/caption. A qualifying caption would be:

> Selected historical per-patch occlusion overlays from fine-tuned classifiers,
> shown only as examples of the visualization. The documented overlay uses
> per-image scaling, interpolation and alpha blending; cross-image magnitudes
> and sub-patch detail must not be inferred. The original attribution arrays are
> not retained, and these examples do not establish disease localization,
> population-level attribution, or a masking-policy mechanism.

That conditional option is **not a receipt for the existing whole asset**.
The different B+C figure should not be substituted without its own quantitative
source review.

## 6. `interp_slice_contribution_by_outcome.png`

**Classification:** twelve quantitative outcome-mean curves, empirical accuracy
percentages, and TP/FN/TN/FP counts across three panels.

Actual labels, not just the caption, encode numerical results. The asset equals
the historical summary PNG, but the attribution/probability arrays needed to
verify outcome stratification and the plotted means are unavailable. A visually
similar mean curve is not proof that errors arise from the same anatomy at lower
signal amplitude; uncertainty and possible heterogeneity matter.

**Minimal resolution:** omit the result and its weaker-signal/error-mechanism
interpretation unless the exact stratified series and threshold metadata are
recovered. Do not substitute frozen-policy AUC data: these are different,
fine-tuned models.

## 7. `interp_slice_contribution_curves.png`

**Classification:** six quantitative class-mean signed curves with counts and
native-position/subset-index axes. Actual labels identify 64 sampled slices
against the native 200-position axis, and class counts 1466/1534.

The retained PNG and method code do not validate the curve values. In addition,
“only slices near zero are genuinely unused” is scientifically unjustified:
a near-zero **signed class mean** can result from cancellation across cases.
Neither a diffuse-looking population curve nor hand-picked patch maps establish
which masking policy the frozen encoder needs.

**Minimal resolution:** omit the quantitative convergence/diffuse-attribution
result without recoverable series. Methods prose may define signed delta-logit,
but should not equate a near-zero class mean with nonuse.

The manuscript's historical **7.14M** and documentation's **7.17M** probe counts
were not overwritten. They refer to configuration-dependent counts; this review
does not provide a configuration-specific parameter-count receipt.

## Is historical-illustration retention defensible?

Yes, for a genuinely selected-image/mask visualization with a narrow and truthful
caption, adequate provenance disclosure, and no unsupported empirical or
clinical annotations. No, as a way to pass quantitative AUC/scatter/attribution
curves whose sources are unavailable.

The existing caveat about unretained arrays is valuable transparency, but does
not validate numbers in the image or remove their scientific claims. A statement
that fine-tuned diagnostics do not establish the frozen-policy mechanism is
necessary and correct; it is not sufficient to make every quantitative
attribution result release-verified.

## Verification and parent application

Run the independent checks:

```powershell
$env:MPLBACKEND='Agg'
& D:\jepa_phase0\.venv\Scripts\python.exe autopilot\investigations\delivered_task\evidence\legacy_figure_reviews\audit_legacy_figures.py
& D:\jepa_phase0\.venv\Scripts\python.exe -m unittest discover -s autopilot\investigations\delivered_task\evidence\legacy_figure_reviews -p test_legacy_review.py -v
```

The tests check real PDF/source mismatch detection, PNG/PDF marker concordance,
failure after an in-memory marker displacement, every caption/input hash,
annotation source checks, schema validation, and duplicate rejection. No test
changes a released image.

The helper can merge a candidate into a **new file inside this owned directory**:

```powershell
& D:\jepa_phase0\.venv\Scripts\python.exe autopilot\investigations\delivered_task\evidence\legacy_figure_reviews\audit_legacy_figures.py --merge-base autopilot\investigations\delivered_task\evidence\delivered_release_reviews.json --merge-output merged_numeric_reviews.json
```

It rejects duplicate source/macro keys and duplicate figure/literal identities,
including Windows path-separator/case variants, and refuses to overwrite an
existing output. The parent can use that new file as the explicit review input;
the original review file is untouched. **Merging the present candidate cannot
clear the seven blockers**, because it intentionally grants no figure approvals.

A parent-approved removal or genuinely source-backed regeneration is required
for closure. Any future asset or caption edit invalidates the old hashes;
regenerate and re-review the relevant evidence rather than carrying a receipt
forward blindly.
