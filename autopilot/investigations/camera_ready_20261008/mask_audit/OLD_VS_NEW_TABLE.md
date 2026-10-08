# Delivered-mask audit (E12): old vs new

Primary: `autopilot\investigations\camera_ready_20261008\mask_audit\delivered_mask_audit_trained_3seeds_100draws.json` (sha256 1523354f143c9ab9); samplers 86ae8823f3afc9fa2c9ed6a3cb0c459b513ba1c5; seeds [42, 1234, 5678]; 100 budget draws x 64 views per seed; 6400 views from 256 volumes; batch order shuffled.

## Trained configuration (mean ± SD across audit seeds)

| Arm | Context % grid | per seed | U | L | D | Mask ratio % | Purity % (soft proxy) | Purity % (hard proxy) | Tissue hidden % | Context share of tissue % |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RANDOM | 27.52 ± 0.14 | 27.38 / 27.54 / 27.66 | 113.1 ± 0.3 | 160.2 ± 0.5 | 47.1 ± 0.2 | 44.2 ± 0.1 | 30.9 ± 0.0 | 34.7 ± 0.1 | 53.2 ± 0.0 | 29.0 ± 0.5 |
| CENTROID | 31.55 ± 0.42 | 31.07 / 31.70 / 31.86 | 102.5 ± 0.2 | 160.2 ± 0.5 | 57.7 ± 0.5 | 40.1 ± 0.1 | 38.2 ± 0.2 | 43.7 ± 0.3 | 59.7 ± 0.2 | 25.3 ± 0.5 |
| ENVELOPE | 29.16 ± 0.49 | 28.67 / 29.15 / 29.65 | 110.2 ± 0.5 | 160.2 ± 0.5 | 50.0 ± 0.9 | 43.1 ± 0.2 | 42.1 ± 0.2 | 48.8 ± 0.2 | 70.7 ± 0.2 | 17.8 ± 0.4 |
| ANATOMY-V2 | 63.32 ± 0.44 | 62.81 / 63.55 / 63.59 | 54.0 ± 0.0 | 64.0 ± 0.0 | 10.0 ± 0.0 | 21.1 ± 0.0 | 96.9 ± 0.2 | 85.4 ± 0.2 | 79.7 ± 0.0 | 17.5 ± 0.1 |
| COVER | 25.16 ± 0.41 | 24.74 / 25.56 / 25.18 | 120.1 ± 0.2 | 160.2 ± 0.5 | 40.2 ± 0.3 | 46.9 ± 0.1 | 40.1 ± 0.2 | 46.4 ± 0.2 | 73.3 ± 0.3 | 14.3 ± 0.3 |

## Paired context gaps vs RANDOM (percentage points of the 256-cell grid)

| Arm − RANDOM | per seed | mean ± seed SD | pooled ± batch SE (95% CI) | batches arm>R / = / < |
|---|---|---:|---:|---:|
| CENTROID | +3.70 / +4.16 / +4.20 | +4.02 ± 0.28 | +4.02 ± 0.20 (+3.63, +4.41) | 259 / 11 / 30 |
| ENVELOPE | +1.30 / +1.61 / +2.00 | +1.63 ± 0.35 | +1.63 ± 0.26 (+1.13, +2.13) | 181 / 18 / 101 |
| ANATOMY-V2 | +35.43 / +36.01 / +35.93 | +35.79 ± 0.31 | +35.79 ± 0.20 (+35.41, +36.17) | 300 / 0 / 0 |
| COVER | -2.63 / -1.98 / -2.48 | -2.36 ± 0.34 | -2.36 ± 0.20 (-2.76, -1.97) | 69 / 10 / 221 |
| CENTROID, published audit lateral 0.8 (code default) | +4.18 / +4.15 / +3.83 | +4.05 ± 0.19 | +4.05 ± 0.21 (+3.64, +4.47) | 257 / 8 / 35 |
| ENVELOPE, published audit (soft guide, least-overlap HEAD fallback) | +2.54 / +2.38 / +2.54 | +2.48 ± 0.09 | +2.48 ± 0.20 (+2.08, +2.89) | 219 / 14 / 67 |
| ENVELOPE, hard guide, least-overlap HEAD fallback | +0.01 / -0.09 / +0.17 | +0.03 ± 0.13 | +0.03 ± 0.21 (-0.39, +0.45) | 151 / 11 / 138 |
| ENVELOPE, soft guide, legacy fallback | +4.92 / +4.98 / +5.25 | +5.05 ± 0.17 | +5.05 ± 0.25 (+4.57, +5.53) | 259 / 8 / 33 |

## Old vs new: delivered context, % of grid (decomposition ladder)

| Step | RANDOM | CENTROID | ENVELOPE | C − R | E − R | note |
|---|---:|---:|---:|---:|---:|---|
| Published v9 (548f4d6; unpaired; 600 views, seed 42, 10 draws; C lateral 0.8; E soft guide + HEAD fallback) | 24.72 | 32.94 | 30.66 | +8.22 | +5.94 |  |
| Pre-camera-ready probe (HEAD; paired crops+sizes; same 600 views/seed/settings) | 28.41 | 30.76 | 28.40 | +2.35 | -0.01 |  |
| A3-scope replay, trained settings (600 views x 3 seeds, 10 draws/seed, sequential) | 27.76 | 31.44 | 28.89 | +3.68 | +1.12 | SD 1.87 / SD 2.74 / SD 1.53 |
| New protocol, OLD settings (C lateral 0.8; E soft guide + least-overlap) | 27.52 | 31.58 | 30.01 | +4.05 | +2.48 |  |
| New protocol, C lateral 0.6; E hard guide + least-overlap (HEAD) | 27.52 | 31.55 | 27.55 | +4.02 | +0.03 |  |
| **New protocol, TRAINED settings (C 0.6; E hard guide + legacy fallback)** | 27.52 | 31.55 | 29.16 | +4.02 | +1.63 | SD 0.14 / SD 0.42 / SD 0.49 |

ENVELOPE 2x2 (guide product x fallback), context % grid: ENVELOPE, published audit (soft guide, least-overlap HEAD fallback): 30.01, ENVELOPE, soft guide, legacy fallback: 32.57, ENVELOPE, hard guide, least-overlap HEAD fallback: 27.55, ENVELOPE: 29.16
