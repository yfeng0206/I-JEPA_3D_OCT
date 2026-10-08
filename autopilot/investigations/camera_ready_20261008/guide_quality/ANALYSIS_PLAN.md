# I6 guide-quality analysis plan (pre-declared)

Written 2026-10-08, **before any metric in this analysis was computed**. Owner: analyst I6.
Script: `autopilot\cr_guide_quality.py`. Outputs: this directory. CPU only, headless
(`MPLBACKEND=Agg`), no image pixels saved or plotted (FairVision CC BY-NC-ND): statistics only.

Prior knowledge disclosed: A4's 32-volume/160-slice Training pilot
(`camera_ready\a4_scratch\cpu_audit.json:411-500`) reported ribbon-envelope grid IoU median .41
(5th pct .26) and ribbon zero-envelope cell fraction median .36 (95th pct .52). The thresholds
below were chosen with that pilot in view; this analysis does not re-tune them after seeing its
own results.

## 0. What is measured

These are **guide-to-guide agreement** statistics between two automatic guides: the CENTROID
intensity-centroid ribbon and the hard MIRAGE repaired envelope. **Neither guide is anatomical
ground truth**; FairVision has no layer annotations. "Failure" below means *disagreement with a
second automatic guide* (or an envelope property suggesting an implausible guide), not a
verified anatomical error.

## 1. Sampling frame and sample

* Frame: **Training split only for all envelope-based analyses.** Production hard envelopes
  (`D:\jepa_phase0\fairvision-glaucoma\mirage_guides\Training`, schema 1, fingerprint
  `9a25a2cdb36f9cba`, MIRAGE-Large GOALS checkpoint SHA-256 `589b0ef2...`) exist for all 6,000
  Training volumes and for **no** Validation/Test volume (`mirage_masks\Validation` is empty).
  Running MIRAGE-Large on CPU to create new Validation envelopes is out of scope (and would not
  be the guide used in pretraining). The Test split is never read.
* Validation (1,000 volumes) contributes a **CENTROID-only descriptive supplement** (no
  envelope): 150 volumes, 25 per stratum, same B-scans, same descriptors (clamping, fallback
  columns, centroid range/slope), stratified with the Training tertile cut-points.
* Image source: the bit-equality-tested slice cache
  (`fairvision-glaucoma\slice_cache\{Training,Validation}\slice_cache.u8`, shape
  `(N,100,200,200)` uint8, slice indices `linspace(0,199,100).astype(int)`); volume order is
  checked against the cache JSON and guide `source_filename`/`slice_indices`.
* B-scans per volume: 10 evenly spaced positions within the 100-slice set,
  `k = linspace(0,99,10)` = {0,11,...,99}, i.e. volume B-scan indices
  {0,22,44,66,88,110,132,154,176,199}.
* Signal-quality proxy (primary), computed for **every** Training and Validation volume on those
  10 native 200x200 uint8 B-scans:
  * per B-scan: background = pixels at or below the B-scan's 50th percentile;
    `mu_b`, `sd_b` = their mean and SD (SD floored at 1.0 grey level);
    `S` = 99th percentile; `SNR_dB = 20*log10(max(S - mu_b, 1) / sd_b)`;
  * per volume: median over the 10 B-scans.
  This is a simple contrast-to-background-noise proxy, **not** the device signal-strength index
  (none is in the metadata).
* Secondary proxy (sensitivity stratifier only): per-volume mean intensity of the same 10
  B-scans (0-255).
* Tertile cut-points: 33.33/66.67 percentiles of the per-volume proxy over all 6,000 Training
  volumes. Strata = glaucoma label (0/1) x SNR tertile (low/mid/high) = 6 strata.
* Primary sample: **50 Training volumes per stratum = 300 volumes x 10 B-scans = 3,000
  B-scans**, drawn without replacement with `numpy.random.default_rng(620261008)`.
* Census sensitivity (decision based on runtime only, made before looking at any metric): if the
  measured primary run time x 20 < 45 min and no `train_patch.py` process is running, the same
  per-B-scan metrics are also computed on all 6,000 Training volumes x 10 B-scans and reported as
  a secondary census check of the stratified estimates.

## 2. Guide construction (center view, exactly as recorded)

* Image: native uint8 B-scan -> `PIL.Image.fromarray(..., "L").resize((256,256), BILINEAR)` ->
  RGB replicate -> float /255 -> tensor (3,256,256). No crop, no flip: identical to the frozen
  probe's view and to `OCTSliceGuidedDataset` with `transform=None`.
* CENTROID ribbon: the production method `CurriculumMaskGenerator._anatomical_prior_weight_grid_for_image`
  extracted **read-only by AST** from `git show fc4527a:src/masks/curriculum.py` (the revision
  bound to the trained CENTROID run by A1/A2) and compared (AST, docstring removed) with
  `HEAD` and the working tree; the digest of each is recorded. Parameters from the producing
  config `results\pretraining\pretrain_oracle_anatomical\config.yaml:37-40`:
  patch 16, grid 16x16, `oracle_region_frac=.28`, `oracle_lateral_frac=.6`,
  `oracle_row_offset=0`, `oracle_min_band_rows=3` -> 10 central columns x 7 rows = 70 cells.
  A copy of the function (attributed) additionally returns its intermediates (smoothed column
  centroids, fallback columns); its grid is asserted equal to the extracted original on every
  B-scan. Affine invariance check: grid on ImageNet-normalized input must equal grid on raw input.
* Hard ENVELOPE: unpack native 200x200 `packed_envelopes[k]`; nearest resize to 256 exactly as
  the dataset (`uint8*255`, `Image.NEAREST`, `>127`); `patch_occupancy` with 16-px patches;
  `E = occupancy >= 0.25` (production `mirage_occupancy_threshold`). Stored `valid` flag and
  center-view `occupancy_is_valid` (min area .05, min span .40) recorded.

## 3. Per-B-scan metrics

Agreement (16x16 patch grid, R = ribbon cells, E = envelope cells):
* `iou = |R and E| / |R or E|`.
* `ribbon_zero_env_frac` = fraction of R cells with envelope occupancy exactly 0.
* `ribbon_outside_frac` = fraction of R cells with occupancy < .25.
* `env_covered` = |R and E| / |E|; `env_covered_central` = same within the 10 ribbon columns.
* `ribbon_purity` = mean occupancy over R (pixel-level envelope fraction inside ribbon).
* Per ribbon column: envelope centroid row (occupancy-weighted, cell-centre coordinates) minus
  ribbon centre row (`top + 3.5`) = signed offset in patches; `col_miss` = envelope present in the
  column but zero occupancy in all 7 ribbon cells; `col_noenv` = no envelope in the column.
  Per B-scan: median and max |offset|, number of columns with |offset| > 3.5 (envelope centroid
  outside the ribbon window), `col_miss` count, `col_noenv` count.
* CENTROID internals: number of clamped ribbon columns (top = 0 or top = 16-7), zero-mass
  fallback columns, globally flat flag.
* Retinal tilt/curvature from the smoothed centroids over the 10 ribbon columns:
  **`centroid_range` = max - min (patch rows) (primary tilt/curvature stratifier)**;
  secondary `abs_slope` (OLS, rows/column) and `abs_quad` (|quadratic coefficient|).
  Tertiles of `centroid_range` over the analysed Training B-scans.

ENVELOPE characterization:
* `env_empty` (0 native pixels); stored `valid`; center-view validity; native area and span.
* Fragmentation: 8-connected components of the native envelope (`n_comp_native`) and of `E`
  (`n_comp_grid`). Report the distribution {0,1,2,>=3}; **`fragmented` = n_comp_native >= 3**
  (two components can be the legitimate optic-nerve-head split).
* Columns without envelope: native fraction of 200 columns, native fraction of the central 60%
  columns, patch-grid columns (of 16) with no E cell.
* Thickness: native per-column pixel count / 12.5 (native px per patch) for envelope-present
  columns -> per-B-scan median, P10, P90, max (patches); patch-grid E cells per column (median,
  max). Outlier column = native thickness > 2x or < .5x the B-scan median;
  **`thickness_irregular` = outlier columns > 20% of envelope-present columns.**
  `multi_run_frac` = fraction of envelope-present native columns with >1 vertical run.
* Vertical plausibility: **`position_implausible` = median |offset| over ribbon columns with
  envelope > 3.5 patches**; `edge_touch_frac` = fraction of envelope-present native columns whose
  envelope touches row 0 or row 199.
* MIRAGE hard-mask QC covariates from `mirage_masks\Training` (descriptive only):
  `mean_confidence`, `mean_entropy`, `topology_violation_fraction`, `all_classes_present`,
  `constant_input`.

## 4. Failure rules (fixed now)

* **Reference available** = envelope non-empty AND stored `valid` AND center-view
  `occupancy_is_valid`. B-scans without a usable reference are reported as ENVELOPE failures and
  excluded from CENTROID-failure denominators (count reported).
* **F1 (primary CENTROID guide failure): `iou < 0.20` OR `ribbon_zero_env_frac > 0.50`.**
  "Outside the envelope" is defined as *zero* envelope pixels in the cell because a 7-row ribbon
  cannot be more than ~50% inside a 3-4-patch-thick envelope even when perfectly centred.
* F2 (centering sensitivity): >= 3 of 10 ribbon columns whose envelope centroid lies outside the
  ribbon window (|offset| > 3.5) or are `col_miss`.
* F3 (literal strict sensitivity): `iou < 0.20` OR `ribbon_outside_frac > 0.50`. Expected to be
  inflated by the ribbon/envelope thickness mismatch; reported only to show the rule dependence.
* Volume level: fraction of volumes with >= 1 and with >= 3 of 10 F1-failing B-scans.
* **Envelope flags (part 2)**: `env_empty`, `env_invalid` (stored or center-view), `fragmented`,
  `thickness_irregular`, `position_implausible`, any-flag.

## 5. Reporting and statistics

* Rates overall, by glaucoma label, by SNR tertile, by label x tertile, and (F1/F2/F3 only) by
  `centroid_range` tertile; secondary by mean-intensity tertile.
* Overall rates: unweighted sample rate and population-weighted rate (stratum weight = Training
  stratum share / sample share).
* 95% CIs: volume-cluster bootstrap, resampling volumes with replacement within each of the 6
  strata, 2,000 replicates, `default_rng(20261008)`; B-scans stay with their volume.
  Contrasts (glaucoma - normal; low - high SNR tertile) as bootstrap risk differences. No
  hypothesis tests or p-values; descriptive only.
* Continuous metrics: mean, median, P5, P95.
* MIRAGE guide-model context: cite documented GOALS metrics with path:line; no new segmentation
  scoring.
* Outputs: `per_bscan_metrics.csv.gz` (hashed case IDs, no pixels), `volume_proxy.csv.gz`,
  `validation_centroid_descriptors.csv.gz`, `summary.json`, `appendix_table.md/.csv`,
  at most two headless histograms (`hist_iou.png`, `hist_env_thickness.png`), `run_log.txt`.

## 6. Interpretation limits (fixed now)

Agreement is not accuracy; the envelope is a repaired MIRAGE prediction (fills the unlabeled
middle retina), produced by a GOALS-trained model evaluated only in-domain. Center view is not the
random-resized-crop view seen during pretraining, so rates describe the guides on full B-scans,
not delivered target masks. The SNR proxy is heuristic. Ten B-scans per volume are clustered;
inference is at volume level.
