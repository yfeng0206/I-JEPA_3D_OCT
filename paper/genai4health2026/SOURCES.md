# SOURCES AND PROVENANCE

**This file is generated** by `autopilot/gen_sources.py` from the artifacts and
the manuscript that actually exist, and is rebuilt on every refresh. Do not edit
it by hand: it will be overwritten, and a hand-edited copy is exactly how it
drifted from the paper before.

Generated 2026-08-24T14:33:43-07:00

## 1. What the built PDF contains

| item | count |
|---|---|
| auto-generated numeric macros | 296 |
| generated tables `\input` into the paper | 6 |
| figures included | 12 |
| distinct citation keys | 48 |

Every numeric quantity in the manuscript resolves through
`auto/auto_numbers.tex`. No number is typed by hand, so prose, tables and
figures cannot disagree.

Generated tables:

- `auto/auto_numbers.tex`
- `auto/table_allprobes.tex`
- `auto/table_fp32.tex`
- `auto/table_operating.tex`
- `auto/table_paired_subgroup.tex`
- `auto/table_subgroup_trends.tex`

Figures:

- `fig1_policies_compact.png`
- `fig_fairness.png`
- `fig_geometry_panel.png`
- `fig_masking_policies.png`
- `fig_precision_paradox.png`
- `fig_specificity_ladder.png`
- `fig_trajectories_ci.png`
- `interp_04_window_occlusion_W7.png`
- `interp_14_odos_mirror_test.png`
- `interp_heatmap_grid.png`
- `interp_slice_contribution_by_outcome.png`
- `interp_slice_contribution_curves.png`

## 2. Statistical base

- test set: N=3000, 1466 positive, 1534 negative, identical across every probe
- bootstrap: 10,000 resamples, seed 20260822, stratified by class, the same resampled
  index set applied to every arm so all differences are paired
- primary frozen probes analysed: 30
- multiplicity: Benjamini-Hochberg within families, confirmatory family size 9, exploratory 22

### Probes in the analysis

| arm | epoch | precision | test AUC |
|---|---|---|---|
| anatomy-v1 | 30 | fp32 | 0.858274 |
| anatomy-v2 | 35 | fp32 | 0.866129 |
| anatomy-v2 | 40 | fp32 | 0.868251 |
| anatomy-v2 | 50 | fp32 | 0.865386 |
| ancestor | 25 | fp32 | 0.848680 |
| cover-f021 | 27 | fp32 | 0.848347 |
| cover-f021 | 30 | fp32 | 0.852249 |
| cover-f021 | 34 | fp32 | 0.857083 |
| cover-f021 | 50 | fp32 | 0.864281 |
| cover-f021 | 73 | fp32 | 0.864717 |
| cover-f021 | 75 | fp32 | 0.863858 |
| cover-f021 | 100 | fp32 | 0.857664 |
| envelope | 30 | fp32 | 0.853917 |
| envelope | 50 | fp16 | 0.876064 |
| envelope | 50 | fp32 | 0.876063 |
| envelope | 75 | fp16 | 0.880307 |
| envelope | 75 | fp32 | 0.880305 |
| envelope | 100 | fp16 | 0.880743 |
| envelope | 100 | fp32 | 0.880761 |
| oracle | 50 | fp16 | 0.874030 |
| oracle | 50 | fp32 | 0.874015 |
| oracle | 75 | fp16 | 0.883636 |
| oracle | 100 | fp16 | 0.885485 |
| oracle | 100 | fp32 | 0.885293 |
| random | 50 | fp16 | 0.864097 |
| random | 50 | fp32 | 0.864121 |
| random | 75 | fp16 | 0.872302 |
| random | 75 | fp32 | 0.872302 |
| random | 100 | fp16 | 0.874581 |
| random | 100 | fp32 | 0.874485 |

### Excluded and retracted runs, never cited as evidence

| run | arm | epoch | AUC | status |
|---|---|---|---|---|
| `frozen_meanpool_bridge_ep75` | anatomy-v2 | 75 | 0.862492 | excluded |
| `frozen_meanpool_bridge_ep92` | anatomy-v2 | 92 | 0.860364 | excluded |
| `frozen_cover_random_ep30` | random-RETRACTED | 30 | 0.855767 | retracted |
| `frozen_cover_random_ep50` | random-RETRACTED | 50 | 0.858968 | retracted |
| `frozen_cover_random_ep75` | random-RETRACTED | 75 | 0.861185 | retracted |
| `frozen_cover_random_ep100` | random-RETRACTED | 100 | 0.860734 | retracted |

- **random-RETRACTED**: SOURCES.md 5.1 - half-precision EMA targets and enc_truncate=window
- **anatomy-v2 ep75/ep92**: SOURCES.md 5.2 - EMA-target precision splice at ep56 (scripts/campaign_chain.py:179 hardcodes amp_target=True)

## 3. Precision

`src/eval_downstream.py:541` reads `use_amp = data_cfg.get('use_amp', True)`,
so the harness default is fp16. Configs that omit the key ran fp16; those
setting it false ran fp32. Rather than assume this immaterial, it is measured:

| arm | epoch | fp16 | fp32 | difference |
|---|---|---|---|---|
| envelope | 50 | 0.876064 | 0.876063 | -0.000001 |
| envelope | 75 | 0.880307 | 0.880305 | -0.000002 |
| envelope | 100 | 0.880743 | 0.880761 | +0.000018 |
| oracle | 50 | 0.874030 | 0.874015 | -0.000015 |
| oracle | 100 | 0.885485 | 0.885293 | -0.000192 |
| random | 50 | 0.864097 | 0.864121 | +0.000024 |
| random | 75 | 0.872302 | 0.872302 | -0.000001 |
| random | 100 | 0.874581 | 0.874485 | -0.000096 |

Largest observed effect 1.92e-04, orders of magnitude below the reported
differences. Each re-probe is hash-guarded: the encoder checkpoint is
SHA-256 hashed before and after and the run is invalidated if it changed.

## 4. Subgroup analysis

- probes used: 22, drawn from 7 pretraining branches
- exclusion status is taken from the evidence inventory, not from the
  subgroup script's own tag, so runs the paper declares excluded are
  excluded here too
- 8 technical duplicates collapsed (an fp32 re-probe and its fp16
  original are the same frozen encoder scored twice):
  - `frozen_meanpool_random_ep50_fp32` same encoder as sweep_random_ep50
  - `frozen_meanpool_random_ep75_fp32` same encoder as sweep_random_ep75
  - `frozen_meanpool_oracle_ep50_fp32` same encoder as sweep_oracle_ep50
  - `frozen_meanpool_random_ep100_fp32` same encoder as sweep_random_ep100
  - `frozen_meanpool_envelope_fp32_ep50` same encoder as frozen_meanpool_mirage_ep50
  - `frozen_meanpool_envelope_fp32_ep75` same encoder as frozen_meanpool_mirage_ep75
  - `frozen_meanpool_envelope_fp32_ep100` same encoder as frozen_meanpool_mirage_ep100
  - `frozen_meanpool_oracle_ep100_fp32` same encoder as sweep_oracle_ep100
- these probes share one test set, one epoch-25 ancestor and one probe seed, and several are checkpoints of the same arm. They are NOT independent, so the Spearman p-values below are not calibrated and must be reported as descriptive.

## 5. Paired subgroup changes

one subject resample per draw, applied to BOTH arms and every subgroup, so all differences below are paired

## 6. Clinical operating points

threshold selected on the validation split at a fixed target specificity, then transferred unchanged to the test split. Cohort prevalence 0.4887.

## 7. Generators

| artifact | generator |
|---|---|
| evidence inventory | `autopilot/p1b_full_inventory.py` |
| paired statistics | `autopilot/p1c_stats.py` |
| DeLong validation | `autopilot/p1_validate_delong.py` |
| demographic join and alignment proof | `autopilot/p1_test_metadata.py` |
| subgroup and severity | `paper/genai4health2026/scripts/subgroup_analysis.py` |
| subgroup trends | `autopilot/p7b_gap_trend.py` |
| paired subgroup changes | `autopilot/p7c_paired_subgroup.py` |
| clinical operating points | `autopilot/p8b_operating_points.py` |
| fp32 integration | `autopilot/p3b_integrate_fp32.py` |
| macros, tables, figures | `autopilot/p8_make_assets.py` |
| this file | `autopilot/gen_sources.py` |
| build and validate the archive | `autopilot/p13_build_zip.py` |

## 8. Pending

No unresolved placeholders remain.

