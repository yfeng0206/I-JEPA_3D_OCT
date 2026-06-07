# Frozen MeanPool Sweep — Oracle anatomical (ep50/75/100)

MeanPool + Linear frozen probe across the oracle anatomical checkpoints (warm-start from random ep25, `oracle_100ep.md`). Same probe config as the random ep100 MeanPool run (Test 0.8746) for apples-to-apples. Probe is the zero-parameter ablation floor (mean over slices → LinearHead), so this reads encoder quality directly.

**Oracle sweep completed** 2026-06-07 (job `tough_malanga_dyyldrr5tn`). **Random backfill running** (job `good_dog_d8lx1wg14t`) — comparison + paired bootstrap finalized when it lands.

## Config (matches `mean_pool.md` / `d1_sweep.md`)

| Parameter | Value |
|---|---|
| Probe | MeanPool + LinearHead (0 probe params, 2.3K head) |
| Encoder | Frozen ViT-B/16 (oracle anatomical) |
| Num slices | 100 |
| Batch size | 256 |
| Epochs / patience | 50 / 15 |
| Warmup | 5 |
| LR (head) | 4e-4 |
| Weight decay | 0.05 |
| Dropout | 0.2 |
| Seed | 42 |

## Oracle results

| Checkpoint | Best epoch | Train AUC | Val AUC | Test AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| ep50 | 47 | 0.8790 | 0.8544 | 0.8740 | 0.751 | 0.836 |
| ep75 | 41 | 0.8925 | 0.8624 | 0.8836 | 0.769 | 0.838 |
| **ep100** | 47 | 0.8936 | 0.8636 | **0.8855** | 0.778 | 0.842 |

Monotonic with pretraining length. Train > Val ~0.03 (mild, expected for a 2.3K-param head); Test > Val ~0.02 is a consistent FairVision val(1000)/test(3000) split property seen in every run.

## Oracle vs Random — preliminary (random ep75 still running)

| Epoch | Random MeanPool Test AUC | Oracle MeanPool Test AUC | Δ (oracle − random) |
|---|---|---|---|
| ep50 | 0.8641 | 0.8740 | +0.0099 |
| ep75 | _running_ | 0.8836 | _pending_ |
| ep100 | 0.8746 (from `mean_pool.md`) | 0.8855 | +0.0109 |

Two preliminary reads (both need the sanity gate + bootstrap below before they are claims):

1. **Consistent quality gain.** Oracle is ~+0.010 over random at both ep50 and ep100. A consistent offset across independent epochs is stronger than any single point.
2. **Pretraining efficiency.** Oracle ep50 (0.8740) ≈ random ep100 (0.8746) — oracle reaches random's final MeanPool performance at half the pretraining epochs.

Cross-check (no harness bug): random ep50 MeanPool 0.8641 = random ep50 d=1 0.8611 (`d1_sweep.md`) + the ~0.003 MeanPool offset — lands exactly on the existing baseline.

## Pending before these become claims

- **Sanity gate**: random ep100 re-run must reproduce ~0.8746. If it drifts, the harness changed — stop and investigate.
- **Paired bootstrap** (B=2000, stratified, `ablation_analysis.md` method) on each run's `test_predictions.npz`: 95% CI + p-value on the +0.010 deltas. Expected significant given the ablation's ~±0.005 paired-delta half-width, but run, not assumed.
- **Fine-tune comparison**: oracle ep100 fine-tuned (d=1 / MeanPool / CrossAttnPool) vs the random fine-tunes (0.8878 / 0.8868 / 0.8872). Note frozen oracle MeanPool (0.8855) already nearly matches fine-tuned random MeanPool (0.8868).
