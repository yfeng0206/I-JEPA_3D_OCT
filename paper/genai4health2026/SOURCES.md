# SOURCES AND PROVENANCE

Every figure, table, macro and headline number in `main_submission.tex`, with
where it came from. Regenerated 2026-08-22 during the autonomous run.

Provenance labels:

| label | meaning |
|---|---|
| `[AUTOPILOT]` | produced by `autopilot/` scripts during the 2026-08-22 autonomous run |
| `[LOCAL-RESULTS]` | `C:\Users\Gary\Desktop\jepa\results\` (tracked in the repo) |
| `[LOCAL-RUNS]` | `D:\jepa_phase0\runs\` (large artifacts, not in git) |
| `[LOCAL-DATA]` | `D:\jepa_phase0\fairvision-glaucoma\` (dataset + released metadata) |
| `[GITHUB]` | public: `yfeng0206/I-JEPA_3D_OCT` |
| `[HUGGINGFACE]` | public: `yfeng0206/ijepa-3d-oct-checkpoints` |

**Rule adopted this run:** no number is typed by hand into the manuscript.
All quantities are emitted as LaTeX macros into `auto/auto_numbers.tex` by
`autopilot/p8_make_assets.py`, which reads only verified JSON artifacts. Prose,
tables and figures therefore cannot disagree, and a correction propagates
everywhere with one re-run.

---

## 0. The evidence base

34 saved per-case prediction files were located; 31 survive de-duplication.
All carry a byte-identical 3000-case label vector (verified by array equality),
so paired bootstrap and DeLong tests are valid.

| family | location | precision | count | status |
|---|---|---|---|---|
| random / oracle / envelope, ep50/75/100 | `results/downstream/meanpool_sweep_*` `[LOCAL-RESULTS]` | fp16 | 9 | primary |
| anatomy-v1/v2, cover-f021, ancestor, envelope ep30 | `D:/jepa_phase0/runs/frozen_*` `[LOCAL-RUNS]` | fp32 | 10 | primary |
| anatomy-v2 ep75/ep92 | `[LOCAL-RUNS]` | fp32 | 2 | excluded (precision splice at ep56) |
| `frozen_cover_random_ep*` | `[LOCAL-RUNS]` | fp32 | 4 | **retracted** (see 5.1) |
| fine-tuned heads | `results/downstream/finetune_*` `[LOCAL-RESULTS]` | fp16 | 6 | supplementary, never pooled with frozen probes |
| envelope/oracle/random fp32 re-probes | `[AUTOPILOT]` `[LOCAL-RUNS]` | fp32 | up to 9 | robustness replication |

**Arm identity is taken from `results.json -> config.model.encoder_checkpoint`,
never from the directory name.** The probe directories named
`frozen_meanpool_mirage_*` and `meanpool_sweep_mirage` are the **envelope** arm
(`patch_mirage_envelope`). Mis-transcribing them as a "mirage" arm would be an
error.

---

## 1. Statistical artifacts `[AUTOPILOT]`

| artifact | generator | contents |
|---|---|---|
| `p1b_full_inventory.json` | `autopilot/p1b_full_inventory.py` | labelled, de-duplicated inventory with arm, epoch, precision, family, status |
| `p1c_stats.json` | `autopilot/p1c_stats.py` | AUC + 10,000-resample paired bootstrap CI per probe; every pairwise contrast with DeLong p, BH q, and a family tag (`A_primary_matched`, `B_CONFOUNDED_cross_precision`, `C_precision_robustness`) |
| `delong_validation.json` | `autopilot/p1_validate_delong.py` | four-way validation of the DeLong estimator: vs sklearn AUC, vs bootstrap SD, null p-value calibration, self-comparison |
| `test_metadata.csv`, `test_metadata_summary.json` | `autopilot/p1_test_metadata.py` | demographic join and the index-alignment proof |
| `p7_fairness.json` | `autopilot/p7_fairness.py` | subgroup AUCs with bootstrap CIs, validation-selected thresholds |
| `subgroup_auc.json` | `paper/genai4health2026/scripts/subgroup_analysis.py` | **authoritative** subgroup source: adds age bins and severity strata from the released metadata CSV |
| `p7b_gap_trend.json` | `autopilot/p7b_gap_trend.py` | gap-vs-AUC trends over 21 probes, all seven stratifications |
| `p3b_fp32.json` | `autopilot/p3b_integrate_fp32.py` | fp16 vs fp32 per arm, and a fully-fp32 replication of the primary contrasts |
| `battery_600.json` | `files/mask_metric_battery.py` | mask geometry over 600 held-out slices using the production samplers |

### Index-alignment proof
The downstream test loader is `shuffle=False` over a dataset whose file list is
`sorted(glob.glob('*.npz'))` (`src/datasets/oct_volumes.py:57-58`,
`oct_slices.py:62-63`). Prediction index *i* therefore corresponds to the *i*-th
sorted file in `Test/`. This is not assumed: `p1_test_metadata.py` rebuilds the
label vector from the sorted file order and asserts equality against the stored
labels of **all 19** local prediction files. Result: 19/19 match, 0 mismatched.
`subgroup_analysis.py` performs the same check against the metadata CSV and
reports `join verified 3000/3000`.

### Subject-level inference
`Test/` holds 3000 `.npz` volumes with 3000 distinct subject identifiers, i.e.
one volume per subject. Case-level resampling is therefore already
subject-level, and no cluster correction is required.

---

## 2. Figures

| figure | file | provenance | generator |
|---|---|---|---|
| 1 | `fig1_policies_compact.png` | `[LOCAL-RUNS]` + session | `files/viz_arm_masking.py`, using the **production** classes `src.masks.multiblock.MaskCollator` and `src.masks.curriculum.CurriculumMaskGenerator` verbatim |
| 2 | `auto/fig_trajectories_ci.png` | `[AUTOPILOT]` | `p8_make_assets.py`; panel (a) trajectories, panel (b) paired deltas with bootstrap CIs |
| 3 | `fig_precision_paradox.png`, `fig_specificity_ladder.png` | `[LOCAL-RUNS]` geometry + `[AUTOPILOT]` AUCs | geometry from `battery_600.json` |
| appendix | `auto/fig_fairness.png` | `[AUTOPILOT]` | `p8_make_assets.py` from `p7_fairness.json` + `p7b_gap_trend.json` |
| appendix | `fig_geometry_panel.png` | 600-slice battery | `files/mask_metric_battery.py` |
| appendix | `fig_masking_policies.png` | production samplers | `files/viz_arm_masking.py` |

Panel (b) of Figure 2 plots **paired** differences rather than per-arm error
bars. Marginal intervals overlap heavily because they carry between-case
variance that cancels in a paired comparison; plotting them alone would
understate the evidence.

## 3. Tables

| table | file | source |
|---|---|---|
| 1 main AUC | inline, macro-driven | `p1c_stats.json` via `auto_numbers.tex` |
| 2 contrasts (CI, p, q) | inline, macro-driven | `p1c_stats.json`, BH over the nine pre-specified contrasts |
| 3 mask geometry | inline | `battery_600.json`; AUC column quoted at **matched epoch 50** so the geometry-to-AUC association is not read across mixed epochs |
| appendix all probes | `auto/table_allprobes.tex` | `p1c_stats.json` + `p1b_full_inventory.json` |
| appendix fairness | `auto/table_fairness.tex` | `p7_fairness.json` |
| appendix fp32 | `auto/table_fp32.tex` | `p3b_fp32.json` |

---

## 4. Precision: the confound, and how it is handled

`src/eval_downstream.py:541` reads `use_amp = data_cfg.get('use_amp', True)`,
so the default is **fp16 autocast**. The `random`, `oracle` and `envelope`
long-horizon probe configs omit the key and therefore ran fp16. Every other
probe sets `use_amp: false` and ran fp32. The earlier draft asserted a single
shared protocol; that assertion was false and has been removed.

Handling:
1. Contrasts are partitioned. Headline claims use only arms matched on **both**
   epoch and precision.
2. Cross-precision contrasts are reported but marked with a dagger and excluded
   from headline claims.
3. The precision effect is measured, not assumed: an existing check re-scores a
   fixed head on fp32-re-encoded features and reproduces the epoch-100 oracle AUC
   to `9.8e-6`. `[AUTOPILOT]` adds the stronger check - a full re-fit of the
   entire probe pipeline at fp32 for all three arms, reported in the appendix
   "Full fp32 re-probe".

Each fp32 re-probe runs through `autopilot/run_guarded_probe.py`, which SHA-256
hashes the encoder checkpoint before and after and marks the run invalid if it
changed. Guard records: `D:/jepa_phase0/autopilot_out/probe_guards/`.

---

## 5. EXCLUDED - never cited as evidence

### 5.1 Retracted coverage probes
`frozen_cover_random_ep{30,50,75,100}` = 0.855767 / 0.858968 / 0.861185 /
0.860734. Pretrained with half-precision EMA targets **and**
`enc_truncate: window`, unlike the arms they would be compared against, so their
deficit is not attributable to masking. **These are not the paper's null.** The
null is `meanpool_sweep_random`.

### 5.2 Precision-spliced anatomy-v2 probes
`anatomy-v2` epochs 75 (0.862492) and 92 (0.860364) follow a change of EMA-target
precision at epoch 56, caused by `scripts/campaign_chain.py:179` hardcoding
`amp_target = True` and overriding the YAML. Listed in the appendix table,
marked, and excluded from every matched-epoch comparison.

**This is a live bug and is still unfixed.** Any re-run through that chain
silently re-breaks fp32.

---

## 6. Pending

`cover f=0.21` epochs 75 and 100 are marked `\ph{}` and render red with a
dagger. That arm was at epoch 73 of 100 when pretraining was deliberately
stopped; per operator instruction it is **not** resumed in this run. Its
epoch-73 checkpoint is pinned at
`D:/jepa_phase0/runs/cover_f021_ep25/jepa_patch_cover_f021-ep73-pinned.pth.tar`
(sha256 `1554ad0e0686fa97d80a34dd3e6e1eefdc6eb5d6c8aab4e04cf4eee573ec54e8`) and
probed as evidence, but the arm's endpoint is not claimed.
