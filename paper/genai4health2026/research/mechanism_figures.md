# Figure specifications for the mechanism argument

## Main Figure A — the non-monotonic composition curve

- **[MEASURED axes]** x-axis: `% of masked cells on anatomy` (`pct_tgt_anat`); y-axis: frozen mean-pool test AUC at ep50.
- **[MEASURED encodings]** point area: total context tokens (`ctx`); point color: `% of anatomy hidden` (`pct_anat_hid`); label each arm. Show COVER f0.21 as an open point at x=40.88 with “AUC pending.”
- **[DATA]** `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json`, `rows[*]`.
- **[ANNOTATION]** Add a bracket over blob: “simultaneously changes purity, geometry, target count, and context budget.”
- **[INFERRED conclusion]** Moderate anatomy targeting is compatible with improving AUC; near-pure targeting is an outlier, not evidence for a smooth optimum.
- **[LIMITATION]** Caption must state four measured AUC points, one pretraining seed each, and blob composition from a separate 1,534-slice pass.
- **[EXISTING PNG]** `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.png` already shows six composition panels, but it should be redesigned with context budget encoded directly.

## Main Figure B — predictor health over the blob trajectory

- **[MEASURED panel 1]** x-axis: blob epoch 30/40/50/56; y-axis: full-context prediction error.
- **[MEASURED panel 2]** x-axis: epoch; y-axis: marginal value ratio `anatomy/background`; horizontal line at 1.
- **[MEASURED panel 3]** x-axis: epoch; y-axis: anatomy excess over count-matched random removal with ±SEM.
- **[DATA]** `D:\jepa_phase0\reports\background_signal\background_signal.json`, entries `tag=blob_ep30|blob_ep40|blob_ep50|blob_ep56`; or `marginal_token_value.csv`.
- **[INFERRED conclusion]** The blob predictor does not merely plateau downstream; its dependence on anatomical context deteriorates and inverts after ep30.
- **[EXISTING PNG]** `D:\jepa_phase0\reports\background_signal\background_signal.png`, top-left and top-right panels. A paper version should isolate blob and add uncertainty for the matched-removal comparison.

## Main Figure C — where glaucoma is readable

- **[MEASURED axes]** grouped bars by encoder arm; y-axis: test AUC from 0.5; bars: all, anatomy-only, background-only pooling.
- **[MEASURED callouts]** mark the top three arm-region cells: envelope-anatomy 0.87844, random-anatomy 0.87466, oracle-anatomy 0.87465.
- **[DATA]** `D:\jepa_phase0\reports\downstream_region_auc\region_auc_summary.json` and per-arm `region_auc.json`.
- **[INFERRED conclusion]** Anatomy positions are the most efficient readout, but disease information remains readable from background positions.
- **[LIMITATION]** Use “background-position tokens,” never “black pixels”; state global attention, 0.8934 mask recall, separately fitted heads, and the 1,000-volume subset.
- **[EXISTING PNG]** `D:\jepa_phase0\reports\downstream_region_auc\region_auc_explained.png` is the preferred existing visualization. `region_auc_visual.png` is a useful mask sanity-check companion.

## Main Figure D — objective focus versus discriminative use

- **[MEASURED panel 1]** x-axis: arm ordered by target anatomy purity; y-axis: background/anatomy absolute contribution per patch.
- **[MEASURED panel 2]** paired points per arm for AUC from anatomy contribution and background contribution under the same exact decomposed head.
- **[DATA]** `D:\jepa_phase0\reports\patch_attribution\attribution_summary.csv`; exact values in each `*_attrib.json`.
- **[MEASURED annotation]** Blob is the only arm below ratio 1, yet its anatomy-contribution AUC is 0.84642 versus 0.85568 for background contribution.
- **[INFERRED conclusion]** More anatomy-focused pretraining makes the head emphasize anatomy positions, but blob fails to convert that emphasis into glaucoma separation.
- **[EXISTING PNG]** None in `D:\jepa_phase0\reports\patch_attribution\`; generate from the CSV/JSON.

## Supplement Figure S1 — background is a learning target, not proven pixel biomarker

- **[MEASURED panel 1]** grouped bars of background versus anatomy skill over a per-position no-context reference for fork ep25, random ep100, oracle ep100, envelope ep100, and blob ep50.
- **[MEASURED panel 2]** marginal error rise per removed background/anatomy context token across checkpoints.
- **[DATA]** `skill_scores.json`, `background_signal.json`, and `marginal_token_value.csv` in `D:\jepa_phase0\reports\background_signal\`.
- **[INFERRED conclusion]** Healthy models predict background targets using context, but anatomy context is more valuable and semantic black-pixel value remains unisolated.
- **[EXISTING PNG]** `D:\jepa_phase0\reports\background_signal\background_signal.png`.

## Supplement Figure S2 — the intervention is multiaxial

- **[MEASURED axes]** parallel-coordinate or small-multiple plot for target slots, unique targets, target purity, context tokens, absolute anatomy context, and zero-anatomy rate.
- **[DATA]** `D:\jepa_phase0\reports\target_composition\summary.json` and `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json`.
- **[INFERRED conclusion]** Blob is not a one-variable “anatomy placement” test.
- **[EXISTING PNG]** `D:\jepa_phase0\reports\arm_coverage\arm_coverage.png` shows related coverage distributions; `D:\jepa_phase0\reports\budget_masks\five_way_masking.png` shows spatial hiding patterns.

## Supplement Figure S3 — absolute versus percentage anatomy context

- **[MEASURED axes]** x-axis: total context tokens; y-axis: anatomy cells in context; iso-percentage rays at 5%, 10%, 20%, and 30%.
- **[DATA]** `composition_vs_auc_ep50.json`, keys `ctx` and `ctx_anat`.
- **[MEASURED annotation]** Blob: 159.997 total and 9.967 anatomy; envelope: 76.411 total and 8.635 anatomy.
- **[INFERRED conclusion]** Blob’s 6.3% anatomy fraction does not mean it has the fewest anatomy cells; this visually falsifies the percentage-only starvation narrative.
- **[EXISTING PNG]** None; generate directly from the JSON.

## Optional conceptual schematic — claim boundary

- **[ASSUMED/INFERRED diagram]** Draw: mask policy → `{target composition, target count, geometry, context budget}` → predictor health → frozen representation → regional readout.
- **[MEASURED overlays]** Attach measured boxes: background target skill, blob error trajectory, anatomy/background regional AUC, patch attribution.
- **[LIMITATION]** Put a dashed, unresolved arrow from “black pixel content” to “glaucoma”; label “not identified: global token mixing + mask leakage.”

