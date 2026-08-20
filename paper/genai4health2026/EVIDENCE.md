# Evidence map

This map covers the original core defect/composition draft. The expanded
mechanism, equity, and ablation claims are mapped in
`research/draft_notes.md`. JSON citations use `file :: key.path`; derived
values state their formula. Bibliographic years and venue names come from the
corresponding entries in `references.bib`.

| ID | Paper claim | Source |
|---|---|---|
| C01 | ViT-B/16; 256×256 slices; 16×16 patch grid; 256 patches; four target blocks | `configs/patch_cover_f021_ep25.yaml:1-27,95`; `src/masks/multiblock.py:1-4,31-47` |
| C02 | Frozen mean-pool + linear protocol; seed 42; 100 slices/volume | `configs/frozen_meanpool_mirage_ep50.yaml:13-47`; the exact output protocol is also recorded in `D:\jepa_phase0\runs\frozen_meanpool_mirage_ep50\results.json` |
| C03 | Test n=3,000 with 1,466 positive and 1,534 negative volumes | `docs/experiments/masking/anatomy_vs_rectangle_ep30.md:57-62`; `docs/experiments/frozen/mirage_meanpool_sweep.md:43-50` |
| C04 | Shared epoch-25 ancestor AUC 0.8487 | `D:\jepa_phase0\runs\frozen_meanpool_fork_ep25\results.json :: test_auc = 0.8486800329413691` |
| C05 | Arm definitions: random, oracle, envelope, anatomy/blob, COVER | `docs/experiments/masking/method_setup.md:5-12`; `src/masks/cover.py:1-35`; `configs/patch_oracle_anatomical.yaml:24-42`; `configs/patch_mirage_envelope.yaml:65-96`; `configs/patch_anatomy_v2.yaml:19-47`; `configs/patch_cover_f021_ep25.yaml:19-95` |
| C06 | Anatomy occupancy threshold 0.25 and the n=6,137 matched sweep | `scripts/cover_floor_sweep.py:35-40,66-90,129-160`; `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json :: *.n = 6137` |
| C07 | Encoder crop chain: `height`/`width`, row-major `r * width + c`, ascending `sorted(best_indices)`, then per-group `min_len` and `t[:min_len]` | `src/masks/multiblock.py:51-53,103,187-189,216-229`; upstream commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`, `src/masks/multiblock.py:145-175` |
| C08 | The encoder prefix retains top-row/small indices and discards bottom-row/large indices; predictor masks instead use unsorted storage and a separate global-minimum truncation, so no predictor spatial claim is made | Encoder: C07 and `docs/experiments/masking/crop_and_precision_audit.md:102-126`; predictor: `src/masks/multiblock.py:193-210` |
| C09 | B=1 uses n=256; B=64 uses n=1,534 | `D:\jepa_phase0\reports\arm_stats_b1\arm_stats.json :: *.n`; `D:\jepa_phase0\reports\arm_stats\arm_stats.json :: *.n`; `docs/experiments/masking/crop_and_precision_audit.md:128-140` |
| C10 | Random context 108.5→69.0, oracle 116.8→78.3, envelope 108.4→75.2, blob 174.8→160.0, COVER .15 103.0→65.7 | Exact means are in `D:\jepa_phase0\reports\arm_stats_b1\arm_stats.json :: <arm>.ctx` and `D:\jepa_phase0\reports\arm_stats\arm_stats.json :: <arm>.ctx`; rounded presentation is independently tabulated at `docs/experiments/masking/crop_and_precision_audit.md:133-140` |
| C11 | Context losses 36.4%, 33.0%, 30.7%, 8.5%, and 36.2% | Derived in `scripts/make_figures.py::figure_crop_defect` as `100*(ctx_B1-ctx_B64)/ctx_B1` from the two JSON files in C10; unrounded results are 36.413%, 32.976%, 30.691%, 8.458%, and 36.179% |
| C12 | Zero-anatomy rates: random 0.00→4.63%, oracle 0.39→4.56%, envelope 1.56→10.10%, blob 2.34→1.24%, COVER .15 2.34→11.02% | `D:\jepa_phase0\reports\arm_stats_b1\arm_stats.json :: <arm>.zero`; `D:\jepa_phase0\reports\arm_stats\arm_stats.json :: <arm>.zero`; `docs/experiments/masking/crop_and_precision_audit.md:133-149` |
| C13 | Example patch accounting: 256 = 90 target + 123 withheld + 43 delivered; anatomy 65 = 50 + 15 + 0 | `docs/experiments/masking/crop_and_precision_audit.md:382-403`; source image `D:\jepa_phase0\reports\arm_stats\zero_anatomy_floor20.png` |
| C14 | Blanking draw dependence: 256 slices, 12 draws, 0% always, none above 50%, 52.0% never versus 44.8% chance | `D:\jepa_phase0\reports\arm_stats\blank_proneness.json :: always_pct, never_pct, overall_blank_pct`; `docs/experiments/masking/crop_and_precision_audit.md:261-299` |
| C15 | Epoch-50 AUC/composition points: random 0.8641/31.6%, oracle 0.8740/39.7%, envelope 0.8761/43.2%, blob 0.8654/97.5% | Random AUC: `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_random\ep50_results.json :: test_auc`; oracle AUC: `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_oracle\oracle_ep50.json :: test_auc`. Composition and envelope/blob AUCs: `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json :: rows[*]`; blob composition originates in `D:\jepa_phase0\reports\arm_stats\arm_stats.json` |
| C16 | “40–43%” is descriptive, not a causal optimum | Rounded interval from C15 (39.6878% to 43.1927%); the observational warning is explicit in `scripts/composition_vs_auc.py:1-18,135-153` |
| C17 | COVER floor .15→.30: anatomy hidden 78.84→64.25%, anatomy reaching context 10.41→20.69%, purity 43.70→37.51%, blank 10.74±0.40→5.51±0.29% | `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json :: ["0.15"|"0.3"].pct_anat_hid, pct_anat_vis, pct_tgt_anat, zero_pct, se` |
| C18 | Floor .21 blank 7.84±0.34% versus envelope 8.07±0.35%; paired McNemar Δ=-0.23 pp, z=-0.6 | Rates and SE: `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json :: ["0.21"|"envelope"].zero_pct,se`; paired result: `configs/patch_cover_f021_ep25.yaml:84-87` |
| C19 | Clean COVER .21 AUCs: ep27 0.8483, ep30 0.8522, ep34 0.8571 | `D:\jepa_phase0\runs\frozen_meanpool_cover_f021_ep27\results.json :: test_auc`; corresponding ep30 and ep34 `results.json` files |
| C20 | Random AUCs .8641/.8723/.8746 and oracle .8740/.8836/.8855 at ep50/75/100 | Ep50 uses the two `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_*` JSONs in C15. Ep75/100: `docs/experiments/frozen/oracle_meanpool_sweep.md:25-40`; mirrored in `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep{75,100}.json :: rows` |
| C21 | Envelope AUCs .8539/.8761/.8803/.8807 | `D:\jepa_phase0\runs\frozen_meanpool_envelope_ep30\results.json :: test_auc`; `D:\jepa_phase0\runs\frozen_meanpool_mirage_ep{50,75,100}\results.json :: test_auc` |
| C22 | Anatomy v1 ep30 .8583; blob v2 ep35 .8661, ep40 .8683, ep50 .8654 | `D:\jepa_phase0\runs\frozen_meanpool_anatomy_ep30\results.json :: test_auc`; `D:\jepa_phase0\runs\frozen_meanpool_bridge_ep{35,40,50}\results.json :: test_auc` |
| C23 | Ep30 five-probe-seed means and SDs: anatomy .8582±.0003, envelope .8528±.0018; Δ .0054; Welch p .00219; Mann–Whitney p .0079; Cohen d 4.20 | `docs/experiments/masking/anatomy_vs_rectangle_ep30.md:95-127` |
| C24 | Ep30 paired bootstrap: +.0044, 95% CI [.0010,.0077], p=.012 | `docs/experiments/masking/anatomy_vs_rectangle_ep30.md:88-94` |
| C25 | Probe seeds are technical replicates of fixed encoders; one pretraining trajectory per arm | `docs/experiments/masking/anatomy_vs_rectangle_ep30.md:95-110,178-189`; `docs/experiments/masking/mask_composition_report.md:241-246` |
| C26 | Ep30 guide mismatch: envelope hard `.npz`, anatomy v1 different soft cache | `docs/experiments/masking/crop_and_precision_audit.md:303-319` |
| C27 | Ep50 reversal: blob .8654 versus envelope .8761, Δ=-.0107 | AUC sources C21-C22; subtraction is exact after four-decimal reporting; confounded interpretation at `docs/experiments/masking/mask_composition_report.md:161-167` |
| C28 | Ep50 hidden fraction approximately 21% vs 46%, target balance approximately 97% vs 43% | `docs/experiments/masking/mask_composition_report.md:81-96,161-167`; newer exact purity values in `D:\jepa_phase0\reports\arm_stats\arm_stats.json :: blob/envelope.pct_tgt_anat` |
| C29 | Retracted old COVER AUCs .8558/.8590/.8612/.8607 and causes (window truncation plus fp16 target) | Exact AUCs: `D:\jepa_phase0\runs\frozen_cover_random_ep{30,50,75,100}\results.json :: test_auc`; retraction and causes: `docs/experiments/masking/crop_and_precision_audit.md:1-7,46-80,89-100,342-363` |
| C30 | Clean COVER is incomplete at ep34 in the package; ep50 is pending | `D:\jepa_phase0\campaign\chain_f021_status.json :: stage = train_ep50, aucs.ep30`; existing ep34 probe artifact in C19; absence of `D:\jepa_phase0\runs\frozen_meanpool_cover_f021_ep50\results.json` at package creation |
| C31 | Qualitative render is not bit-stable; the measured PNG is reused | `docs/experiments/masking/crop_and_precision_audit.md:373-410`; `scripts/show_zero_anatomy_slices.py`; source PNG path in C13 |
| C32 | Current training code seeds Python, NumPy, CPU Torch, and CUDA from the configured pretraining seed | `src/train_patch.py:147-154`; this does not create pretraining-seed replication because only one trajectory per arm was run |
| C33 | Envelope validation loss 0.1191 at ep26, 0.1200 at ep30, 0.1326 at ep34 | `D:\jepa_phase0\campaign\val_baseline_envelope.json :: ["26"], ["30"], ["34"]` |

## Figure provenance

| Figure | Generator | Inputs |
|---|---|---|
| F1 `fig1_crop_defect` | `scripts/make_figures.py::figure_crop_defect` | B=1 and B=64 `arm_stats.json` files |
| F1b `fig1b_context_excision` | `scripts/make_figures.py::figure_context_excision` | Crop of stored `arm_stats/zero_anatomy_floor20.png`; no new sampling |
| F2 `fig2_composition_vs_auc` | `scripts/make_figures.py::figure_composition_auc` | `composition_vs_auc_ep50.json` |
| F3 `fig3_cover_floor_dose_response` | `scripts/make_figures.py::figure_cover_dose_response` | `cover_floor_sweep.json` and clean COVER probe `results.json` files |
| F4 `fig4_auc_trajectories` | `scripts/make_figures.py::figure_auc_trajectories` | composition reports plus all listed probe `results.json` files |
| F5 `fig5_zero_anatomy_example` | `scripts/make_figures.py::figure_qualitative` | measured `zero_anatomy_floor20.png`; no resampling |

## Numeric audit note

The numeric families in the original defect/composition draft are represented
above:
architecture/grid (`16`, `256`), sample sizes (`256`, `1,534`, `3,000`,
`6,137`), epochs (`25`, `27`, `30`, `34`, `35`, `40`, `50`, `75`, `100`),
composition and blanking percentages, all AUC milestones, the ep30 inferential
statistics, and the struck retracted COVER values. Section/table/figure
numbers and bibliographic years are document structure or citation metadata,
not experimental measurements.

Literal LaTeX spellings checked by the audit include `1{,}466`, `1{,}534`,
`3{,}000`, `6{,}137`, `216--229`, `0.00219`, `0.0044`, `0.0054`, `0.0079`,
`0.0107`, `0.8528`, and `0.8582`; their sources are C03, C07, C23, C24, and
C27. The remaining literal tokens `0.20`, `0.22`, `0.30`, `0.70`, and `3.2`
are respectively a measured floor (C13/C31), LaTeX table widths, a measured
floor (C17), a LaTeX table width, and table spacing. `2026` identifies the
downloaded style and bibliography metadata rather than a measured result.
