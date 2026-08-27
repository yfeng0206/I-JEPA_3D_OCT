# Table 2 (`tab:geom`) provenance: regenerated from the production samplers

Status: REGENERATION SUCCEEDED. All 25 geometry cells now have a stored artifact
behind them. The rank order of arms by anatomy-hidden and by purity is unchanged,
and the Spearman coefficients recompute to exactly the printed `+0.80` and `+0.40`.

Labels used throughout: MEASURED = read from an artifact produced in this session;
INFERRED = arithmetic or interpretation on top of measured values; PENDING = not
established.

---

## 1. Script chosen

**`scripts/mask_composition_probe.py`** (MEASURED choice, by inspection of all five
candidates).

It is the only candidate whose aggregation step emits all five printed columns
directly, for all five arms, from one pass over one slice set:

| printed column | key emitted by `aggregate()` (`mask_composition_probe.py:182-202`) |
|---|---|
| anatomy hidden | `hidden_share_of_all_anat` |
| purity | `hidden_pct_on_anat` |
| mask ratio | `hidden_frac_of_grid` (x100) |
| context kept | `ctx_frac_of_grid` (x100) |
| loss slots | `n_slots_mean` |

Why the others were rejected (MEASURED, by reading each file):

- `scripts/arm_stats_table.py` - emits `ctx`, `tgt`, `pct_tgt_anat`,
  `pct_anat_hid`, but has no slot count (loss slots), and its COVER entries are a
  floor sweep rather than one production arm.
- `scripts/target_composition.py` - emits slots, unique cells, context tokens and
  background fractions for all five arms, but never computes anatomy-hidden
  (share of all tissue cells masked). Its stored artifact
  `D:\jepa_phase0\reports\target_composition\summary.json` is the one that
  independently supports the single `64.0` cell.
- `scripts/five_arm_audit.py` - close second; emits slots, union, ctx tokens and
  anatomy-hidden for all five arms, but purity is only derivable, not emitted,
  and it does not print the five columns as such.
- `scripts/composition_vs_auc.py` - not a measurement script at all. It re-reads
  two pre-existing artifacts (`arm_stats_sweep/cover_floor_sweep.json` n=6137 and
  `arm_stats/arm_stats.json` n=1534) and joins them to AUC. This is the source of
  `composition_vs_auc_ep50.json`, the related-but-different artifact: it mixes two
  sampling passes of different sizes, which is why its numbers are near but not
  equal to Table 2.

## 2. Two configuration facts had to be recovered before the numbers reproduced

Both were found by comparing the script's hard-coded arm definitions against the
production YAML configs.

**(a) COVER floor. MEASURED.** `mask_composition_probe.py` hard-coded
`cover_leave_frac = cover_min_visible_frac = 0.15`. The trained COVER arm - the
one whose AUC 0.8643 sits in Table 2 - uses `0.21`
(`configs/patch_cover_f021_ep25.yaml:55-56`). At floor 0.15 COVER hides 79.5% of
anatomy and overtakes ENVELOPE; at the production floor 0.21 it hides 73.6% and
sits below ENVELOPE, as printed. The paper's own prose independently quotes
`73.1%` for COVER (`main_submission.tex:440`), which agrees with the 0.21 run.

**(b) The `context kept` column is per-image geometry, not batch-collated
geometry. MEASURED, with INFERRED interpretation.** `MaskCollator._truncate_and_stack`
(`src/masks/multiblock.py:217-228`) truncates every context mask in a batch to the
batch minimum, so the delivered context shrinks as batch size grows. Measured on
the same 600 slices:

| arm | context kept, per-image (batch_size=1) | context kept, delivered at production batch_size=64 | printed |
|---|---|---|---|
| random | 41.87% | 24.72% | 42.1% |
| centroid | 45.50% | 32.94% | 45.6% |
| envelope | 40.63% | 30.66% | 40.5% |
| cover | 43.18% | 30.06% | 43.5% |
| anatomy-v2 | 67.74% | 63.46% | 67.9% |

The printed column tracks the per-image figure to within 0.32 points on all five
rows, and is 10-17 points above the delivered figure on the four rectangle arms.
INFERRED: Table 2 reports the geometry the sampler produces per image, before the
collator's batch-minimum truncation. This is a defensible choice - it isolates the
masking policy from the batch - but the caption's phrase "production samplers"
does not say so, and a reader may take "context kept" to be what the encoder
actually receives. Recommended caption addition (not applied - `main_submission.tex`
was left untouched as instructed): state that geometry is measured per image,
before batch collation.

## 3. Exact commands

Primary artifact (the one Table 2 should cite):

```
D:\jepa_phase0\.venv\Scripts\python.exe scripts\mask_composition_probe.py ^
  --split Training --volumes 24 --num_slices 100 --slices_per_volume 25 ^
  --batch_size 1 --seed 42 --cover_floor 0.21 ^
  --out C:\Users\Gary\Desktop\jepa\results\masking\table2_geometry\mask_geometry_600slices_bs1_coverf021_seed42.json
```

Replicates (sampling-noise bound): the same command with `--seed 1234` and
`--seed 2026`.

Delivered-to-encoder variant (production batch size): the same command with
`--batch_size 64`.

Environment: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `CUDA_VISIBLE_DEVICES=""`.
The script loads no model and uses no GPU and no dataloader workers. Nothing under
`C:\jepa_data` or `D:\jepa_phase0` was written, moved or deleted.

Three small backwards-compatible additions were made to
`scripts/mask_composition_probe.py` to make the run expressible on the command
line; defaults preserve the previous behaviour exactly:

- `--slices_per_volume` (default 0 = old behaviour) - stride-subsample slices per
  volume, so 600 slices can span 24 volumes instead of 6.
- `--cover_floor` (default 0.15 = old behaviour) - expose the COVER floor.
- `_meta` now records `slices_per_volume`, `batch_size` and `cover_floor`.

## 4. Artifacts written

All under `C:\Users\Gary\Desktop\jepa\results\masking\table2_geometry\`:

| file | role |
|---|---|
| `mask_geometry_600slices_bs1_coverf021_seed42.json` | PRIMARY - the artifact Table 2 should cite |
| `mask_geometry_600slices_bs1_coverf021_seed1234.json` | replicate |
| `mask_geometry_600slices_bs1_coverf021_seed2026.json` | replicate |
| `mask_geometry_600slices_bs64_coverf021_seed42.json` | delivered-to-encoder variant at production batch size |
| `mask_geometry_600slices_bs64_coverf015_seed42.json` | exploratory, wrong COVER floor, retained for audit |
| `mask_geometry_600slices_bs64_coverf015_seed1234.json` | exploratory, wrong COVER floor, retained for audit |
| `table2_comparison.md` | machine-generated comparison, regenerated by `autopilot/compare_table2_geometry.py` |

## 5. Comparison table (all MEASURED except the `printed` column, which is
transcribed from `main_submission.tex:473-477`)

`regenerated` is the primary artifact, seed 42. `seed range` is the spread of that
cell across the three 600-slice draws, i.e. the measurement's own sampling noise.
Verdicts follow the requested rule: MATCHES = within rounding (|diff| <= 0.05);
CLOSE = beyond rounding but the arm keeps its position in that metric's ordering;
DIFFERS = ordering position changed.

| arm | metric | printed | regenerated | abs diff | seed range (n=3) | verdict |
|---|---|---|---|---|---|---|
| random | anatomy hidden | 52.2 | 53.95 | 1.75 | 1.33 | CLOSE |
| random | purity | 31.6 | 31.47 | 0.13 | 1.18 | CLOSE |
| random | mask ratio | 43.7 | 44.49 | 0.79 | 0.82 | CLOSE |
| random | context kept | 42.1 | 41.87 | 0.23 | 0.38 | CLOSE |
| random | loss slots | 157.7 | 159.91 | 2.21 | 2.30 | DIFFERS |
| centroid | anatomy hidden | 62.2 | 62.13 | 0.07 | 1.15 | CLOSE |
| centroid | purity | 41.1 | 40.01 | 1.09 | 1.92 | CLOSE |
| centroid | mask ratio | 40.0 | 40.29 | 0.29 | 0.28 | CLOSE |
| centroid | context kept | 45.6 | 45.50 | 0.10 | 0.36 | CLOSE |
| centroid | loss slots | 158.4 | 158.99 | 0.59 | 0.82 | DIFFERS |
| envelope | anatomy hidden | 76.9 | 77.58 | 0.68 | 0.74 | CLOSE |
| envelope | purity | 43.5 | 43.30 | 0.20 | 0.47 | CLOSE |
| envelope | mask ratio | 46.4 | 46.48 | 0.08 | 1.18 | CLOSE |
| envelope | context kept | 40.5 | 40.63 | 0.13 | 1.10 | CLOSE |
| envelope | loss slots | 159.9 | 159.68 | 0.22 | 1.75 | CLOSE |
| cover | anatomy hidden | 74.1 | 73.55 | 0.55 | 0.54 | CLOSE |
| cover | purity | 45.3 | 44.19 | 1.11 | 1.09 | CLOSE |
| cover | mask ratio | 43.3 | 43.19 | 0.11 | 0.44 | CLOSE |
| cover | context kept | 43.5 | 43.18 | 0.32 | 0.24 | CLOSE |
| cover | loss slots | 160.0 | 159.09 | 0.91 | 0.89 | DIFFERS |
| anatomy-v2 | anatomy hidden | 80.3 | 79.89 | 0.41 | 0.73 | CLOSE |
| anatomy-v2 | purity | 97.3 | 97.09 | 0.21 | 0.19 | CLOSE |
| anatomy-v2 | mask ratio | 21.4 | 21.35 | 0.05 | 0.32 | CLOSE |
| anatomy-v2 | context kept | 67.9 | 67.74 | 0.16 | 0.48 | CLOSE |
| anatomy-v2 | loss slots | 64.0 | 64.00 | 0.00 | 0.00 | MATCHES |

Counts: **MATCHES 1, CLOSE 21, DIFFERS 3.**

Supporting facts (MEASURED):

- Largest absolute difference over all 25 cells: **2.21** (random / loss slots).
- 18 of 25 cells fall inside the three-seed range of the measurement itself, i.e.
  they are indistinguishable from a re-draw of 600 slices.
- The strict MATCHES count is 1 only because a fresh 600-slice draw cannot be
  expected to reproduce a one-decimal value exactly; the only cell that can is
  `anatomy-v2 / loss slots = 64.0`, which is deterministic (`pred_target_k=16` x
  `npred=4`) and does reproduce exactly.

**The three DIFFERS cells are all `loss slots` on rectangle arms, and are not a
substantive disagreement (INFERRED).** The four rectangle arms all draw four
rectangles from the same `pred_mask_scale` range, so their slot counts are equal
to within noise: regenerated 158.99 / 159.09 / 159.68 / 159.91 with a per-arm
seed-to-seed standard deviation of 0.43-1.15. Their printed ordering
(157.7 < 158.4 < 159.9 < 160.0) is inside that noise band and therefore carries no
information; the ordering rule flags them mechanically. No claim in the paper
depends on the ordering of `loss slots`. The one contentful `loss slots` fact -
that anatomy-v2 supervises 64 slots against roughly 160 for every other arm - is
reproduced exactly.

## 6. Rank-order verdict: UNCHANGED (MEASURED)

**Anatomy hidden**, low to high:
- printed, 5 arms: random, centroid, cover, envelope, anatomy-v2
- regenerated, 5 arms: random, centroid, cover, envelope, anatomy-v2 - IDENTICAL
- printed, 4 rectangles: random, centroid, cover, envelope
- regenerated, 4 rectangles: random, centroid, cover, envelope - IDENTICAL
- all three independent 600-slice draws give the same 4-rectangle order.

**Purity**, low to high:
- printed, 5 arms: random, centroid, envelope, cover, anatomy-v2
- regenerated, 5 arms: random, centroid, envelope, cover, anatomy-v2 - IDENTICAL
- printed, 4 rectangles: random, centroid, envelope, cover
- regenerated, 4 rectangles: random, centroid, envelope, cover - IDENTICAL
- all three independent 600-slice draws give the same 4-rectangle order.

Caveat (MEASURED): the anatomy-hidden order is only robust at the production COVER
floor. With the script's old hard-coded floor of 0.15, COVER hides 79.5% and
overtakes ENVELOPE, which would drop the 4-rectangle Spearman from +0.80 to +0.40.
The floor is therefore load-bearing for that coefficient and is worth stating
wherever COVER's geometry is quoted.

## 7. Recomputed Spearman (scipy.stats.spearmanr, MEASURED)

AUC at the matched epoch 50, transcribed from
`paper/genai4health2026/auto/auto_numbers.tex`: random 0.8641, centroid 0.8740,
envelope 0.8761, cover 0.8643, anatomy-v2 0.8654.

| metric | arm set | printed values | regenerated values | agree |
|---|---|---|---|---|
| anatomy hidden | 4 rectangles | rho = +0.8000, p = 0.2000 | rho = +0.8000, p = 0.2000 | yes |
| purity | 4 rectangles | rho = +0.4000, p = 0.6000 | rho = +0.4000, p = 0.6000 | yes |
| anatomy hidden | 5 arms | rho = +0.5000, p = 0.3910 | rho = +0.5000, p = 0.3910 | yes |
| purity | 5 arms | rho = +0.2000, p = 0.7471 | rho = +0.2000, p = 0.7471 | yes |

All three replicate draws give identical coefficients to the primary. **The
corrected values `+0.80` (anatomy hidden) and `+0.40` (purity) over the four
rectangle arms are confirmed by regeneration and require no further change.** The
"including the anatomy arm these fall to +0.50 and +0.20" sentence is also
confirmed.

## 8. What could NOT be reproduced

**The word "held-out" in the caption. PENDING - and, as far as this repository
goes, not reproducible.**

The three segmenter-guided samplers (envelope, cover, anatomy-v2) require the
MIRAGE soft guide cache. That cache
(`C:\jepa_data\mirage_soft_guides\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy`)
contains a `Training` directory only, covering volumes `data_00001` to
`data_06000`; its `cache_meta.json` records `"split": "Training"`,
`"n_volumes": 6000`. The FairVision Validation split is `data_06001`-`data_07000`
(`configs/patch_mirage_envelope.yaml:44-47`) and has no guide files - checked
directly, and no other guide directory exists on either drive. Three of the five
production samplers therefore cannot be run on held-out slices at all.

The regeneration was consequently run on 600 Training-split slices (24 volumes x
25 slices, stride 4). Every stored geometry artifact in the repository that I
inspected - `arm_stats`, `arm_stats_sweep`, `target_composition`,
`five_arm_audit` - was also produced on the Training split, so this is the
established convention rather than a deviation.

Recommendation (not applied): the caption's "600 held-out slices" should become
"600 slices" or "600 pretraining-split slices". No model is involved in this
measurement, so the phrase buys nothing and is not currently supportable.

## 9. Reproducing this report

```
D:\jepa_phase0\.venv\Scripts\python.exe autopilot\compare_table2_geometry.py
```

Regenerates `results\masking\table2_geometry\table2_comparison.md` from the
artifacts. Only two values in that script are typed by hand - the printed Table 2
cells and the epoch-50 AUCs - and both are transcriptions from the paper sources
cited above.
