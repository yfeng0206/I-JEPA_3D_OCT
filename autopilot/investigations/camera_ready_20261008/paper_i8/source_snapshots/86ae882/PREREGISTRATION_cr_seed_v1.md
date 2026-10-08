# Pre-registered analysis: camera-ready replication campaign `cr_seed_v1`

Committed before any new pretraining run produced a checkpoint and before any new probe.
The commit hash and timestamp of this file serve as the registration record.
Companion plan: `CAMERA_READY_PLAN.md` (same folder).

## 1. Runs

| ID | Arm | Seed | Notes |
|---|---|---|---|
| R1 | RANDOM (stock I-JEPA multiblock) | 1234 | |
| C1 | CENTROID (intensity-centroid ribbon, `oracle_lateral_frac` 0.6, `oracle_region_frac` 0.28) | 1234 | |
| E1 | ENVELOPE (hard MIRAGE envelope guides, sampler `legacy_uniform_v1` = trained behaviour) | 1234 | |
| R2 | RANDOM | 5678 | |
| C2 | CENTROID | 5678 | |
| CB | RANDOM-CB: uniform placement with CENTROID's exact per-image target and context budgets (CENTROID s1234 shadow masks) | 1234 | runs only if implemented and verified in time |
| E2 | ENVELOPE (`legacy_uniform_v1`) | 5678 | dropped first if time runs short |

Common settings: continuation from the shared RANDOM epoch-25 checkpoint
(SHA-256 `e5ad5b0c...e97e7b`), `resume_policy: fork`, schedule horizon `epochs: 100`,
stop after the saved epoch-50 checkpoint, microbatch 64 x accumulation 8 on one GPU
(effective batch 512), fp32 teacher targets (`amp_target: false`), prefix context
truncation, no fixed target length, guidance ramp T_warm 25 -> T_total 30.
The seed sets the data order (training sampler), crops, mask draws and worker streams.
Runs with the same seed index share data order only; crops and masks are not paired.
These are repeated continuations from a shared epoch-25 checkpoint, not independent
initializations.

## 2. Evaluation protocol

- Endpoint: epoch-50 target encoder.
- Probe: frozen encoder; 100 B-scans per volume; patch-token mean per B-scan; mean over
  B-scans; LayerNorm + linear head (2,305 parameters); AdamW lr 4e-4, wd 0.05, batch 256,
  50 epochs, warmup 5, patience 15, best epoch by validation AUC; fp32 throughout.
- Head seeds 42, 43, 44, 45, 46, seeded after feature caches are created or loaded.
  The per-encoder value is the mean AUC over head seeds; the head-seed SD is reported.
- Split: the existing FairVision glaucoma split (6,000 / 1,000 / 3,000). The test split is
  the same one used for the original paper.
- Test predictions are written sealed (hashed, no metrics computed) until the results
  freeze. All campaign decisions use validation AUC only.
- Original anchors: the original RANDOM, CENTROID and ENVELOPE epoch-50 encoders are
  re-evaluated with the same protocol (RANDOM from its retained fp32 per-slice feature
  cache because its weights are no longer stored locally; CENTROID and ENVELOPE freshly
  extracted).

## 3. Primary analysis

- New seeds only (n = 2 per arm). For each seed s and guided arm g in {CENTROID, ENVELOPE}:
  delta(g, s) = AUC(g, s) - AUC(RANDOM, s), with a paired test-case bootstrap 95% CI
  (10,000 resamples, stratified by label). No seed-level bootstrap at n = 2.
- Outcome label per guided arm:
  - **Consistent direction:** both new-seed deltas > 0 and the mean delta exceeds the
    range of RANDOM's three realizations (original re-evaluated, R1, R2).
  - **Within run-to-run variation:** mean delta > 0 but either condition fails.
  - **Reversed:** mean delta <= 0.
  Two of two positive deltas occur 25% of the time under a symmetric null; the first
  label is not described as a confirmed replication.
- Sensitivity (descriptive only): the same contrasts with the original runs included
  (n = 3; different code version and hardware).

## 4. Matched-budget control (if CB completes)

- Placement at equal budgets: AUC(C1) - AUC(CB).
- Budget-conditioned shift: AUC(CB) - AUC(R1).
- Both judged against the spread of RANDOM's realizations. One continuation per arm;
  reported as such.
- Delivered-budget audit: per-image unique targets, loss slots, duplicates and context
  counts for CB versus CENTROID shadow masks, and target tissue purity.

## 5. Secondary analyses

- Validation-split deltas for every contrast above.
- Pooling variants on every new encoder and anchor: patch-max within B-scan with mean
  across B-scans; across-B-scan max of patch-means; concatenated mean and max.
  The primary pooling stays the patch mean with the across-B-scan mean.
- Visual-field mean deviation regression within glaucoma cases (ridge; alpha selected on
  validation; MAE, RMSE, Spearman), reported as a severity endpoint on the same cohort,
  not an independent task. Specified here before any regressor is fitted.

## 6. Reproduction gates (engineering checks, not outcomes)

- G1 (loss): absolute deviation from the reference loss curve of the same arm; warn above
  0.003, hold above 0.005 on two consecutive epochs; from epoch 31 the run must track its
  own arm's reference more closely than the other arms'.
- G2 (validation AUC at epoch 50): compared with the same arm's original under this
  protocol; differences above 0.01 trigger investigation, not reruns. A shift common to all
  arms with clean loss and mask statistics is reported as is; an arm whose loss or mask
  statistics also mismatch is stopped, fixed and rerun.
- No run is repeated or discarded because of its AUC.

## 7. Reporting

Every started run is reported, including runs stopped by the deadline (as not completed).
No checkpoint selection beyond the fixed epoch-50 endpoint. Results freeze: Thursday
October 22 2026, 08:00 PDT; sealed test predictions are opened once after the freeze.
