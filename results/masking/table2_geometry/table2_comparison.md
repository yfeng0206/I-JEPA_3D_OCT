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

verdict counts: {'MATCHES': 1, 'CLOSE': 21, 'DIFFERS': 3}

cells within the 3-seed range of the measurement: 18/25
largest absolute difference over all 25 cells: 2.21 (random / loss slots)

### Delivered-to-encoder variant (batch_size=64, same slices)
| arm | metric | printed | delivered (bs=64) | abs diff |
|---|---|---|---|---|
| random | anatomy hidden | 52.2 | 53.54 | 1.34 |
| random | purity | 31.6 | 32.04 | 0.44 |
| random | mask ratio | 43.7 | 43.97 | 0.27 |
| random | context kept | 42.1 | 24.72 | 17.38 |
| random | loss slots | 157.7 | 161.76 | 4.06 |
| centroid | anatomy hidden | 62.2 | 62.97 | 0.77 |
| centroid | purity | 41.1 | 40.44 | 0.66 |
| centroid | mask ratio | 40.0 | 40.97 | 0.97 |
| centroid | context kept | 45.6 | 32.94 | 12.66 |
| centroid | loss slots | 158.4 | 162.13 | 3.73 |
| envelope | anatomy hidden | 76.9 | 78.20 | 1.30 |
| envelope | purity | 43.5 | 43.13 | 0.37 |
| envelope | mask ratio | 46.4 | 47.71 | 1.31 |
| envelope | context kept | 40.5 | 30.66 | 9.84 |
| envelope | loss slots | 159.9 | 164.16 | 4.26 |
| cover | anatomy hidden | 74.1 | 73.43 | 0.67 |
| cover | purity | 45.3 | 44.68 | 0.62 |
| cover | mask ratio | 43.3 | 43.24 | 0.06 |
| cover | context kept | 43.5 | 30.06 | 13.44 |
| cover | loss slots | 160.0 | 159.04 | 0.96 |
| anatomy-v2 | anatomy hidden | 80.3 | 79.82 | 0.48 |
| anatomy-v2 | purity | 97.3 | 97.89 | 0.59 |
| anatomy-v2 | mask ratio | 21.4 | 21.46 | 0.06 |
| anatomy-v2 | context kept | 67.9 | 63.46 | 4.44 |
| anatomy-v2 | loss slots | 64.0 | 64.00 | 0.00 |

### Rank order
- anatomy hidden, 5 arms, printed (low to high): ['random', 'centroid', 'cover', 'envelope', 'anatomy-v2']
- anatomy hidden, 5 arms, regen   (low to high): ['random', 'centroid', 'cover', 'envelope', 'anatomy-v2']
- anatomy hidden, 5 arms unchanged: True
- anatomy hidden, 4 rectangles, printed: ['random', 'centroid', 'cover', 'envelope']
- anatomy hidden, 4 rectangles, regen  : ['random', 'centroid', 'cover', 'envelope']
- anatomy hidden, 4 rectangles unchanged: True
  - replicate 0 4-rectangle order: ['random', 'centroid', 'cover', 'envelope'] (same as printed: True)
  - replicate 1 4-rectangle order: ['random', 'centroid', 'cover', 'envelope'] (same as printed: True)
  - replicate 2 4-rectangle order: ['random', 'centroid', 'cover', 'envelope'] (same as printed: True)

- purity, 5 arms, printed (low to high): ['random', 'centroid', 'envelope', 'cover', 'anatomy-v2']
- purity, 5 arms, regen   (low to high): ['random', 'centroid', 'envelope', 'cover', 'anatomy-v2']
- purity, 5 arms unchanged: True
- purity, 4 rectangles, printed: ['random', 'centroid', 'envelope', 'cover']
- purity, 4 rectangles, regen  : ['random', 'centroid', 'envelope', 'cover']
- purity, 4 rectangles unchanged: True
  - replicate 0 4-rectangle order: ['random', 'centroid', 'envelope', 'cover'] (same as printed: True)
  - replicate 1 4-rectangle order: ['random', 'centroid', 'envelope', 'cover'] (same as printed: True)
  - replicate 2 4-rectangle order: ['random', 'centroid', 'envelope', 'cover'] (same as printed: True)

### Spearman rho against AUC @ ep50
| metric | arm set | source | rho | p |
|---|---|---|---|---|
| anatomy hidden | 4 rectangles | printed | +0.8000 | 0.2000 |
| anatomy hidden | 4 rectangles | regenerated | +0.8000 | 0.2000 |
| anatomy hidden | 4 rectangles | replicate 0 | +0.8000 | 0.2000 |
| anatomy hidden | 4 rectangles | replicate 1 | +0.8000 | 0.2000 |
| anatomy hidden | 4 rectangles | replicate 2 | +0.8000 | 0.2000 |
| anatomy hidden | 5 arms | printed | +0.5000 | 0.3910 |
| anatomy hidden | 5 arms | regenerated | +0.5000 | 0.3910 |
| anatomy hidden | 5 arms | replicate 0 | +0.5000 | 0.3910 |
| anatomy hidden | 5 arms | replicate 1 | +0.5000 | 0.3910 |
| anatomy hidden | 5 arms | replicate 2 | +0.5000 | 0.3910 |
| purity | 4 rectangles | printed | +0.4000 | 0.6000 |
| purity | 4 rectangles | regenerated | +0.4000 | 0.6000 |
| purity | 4 rectangles | replicate 0 | +0.4000 | 0.6000 |
| purity | 4 rectangles | replicate 1 | +0.4000 | 0.6000 |
| purity | 4 rectangles | replicate 2 | +0.4000 | 0.6000 |
| purity | 5 arms | printed | +0.2000 | 0.7471 |
| purity | 5 arms | regenerated | +0.2000 | 0.7471 |
| purity | 5 arms | replicate 0 | +0.2000 | 0.7471 |
| purity | 5 arms | replicate 1 | +0.2000 | 0.7471 |
| purity | 5 arms | replicate 2 | +0.2000 | 0.7471 |

### Replicate values (n=3 draws of 600 slices)
| arm | metric | seed 42 | seed 1234 | seed 2026 | mean | stdev |
|---|---|---|---|---|---|---|
| random | anatomy hidden | 53.95 | 53.73 | 52.63 | 53.44 | 0.71 |
| random | purity | 31.47 | 31.31 | 30.29 | 31.02 | 0.64 |
| random | mask ratio | 44.49 | 43.67 | 43.96 | 44.04 | 0.41 |
| random | context kept | 41.87 | 42.25 | 42.25 | 42.12 | 0.22 |
| random | loss slots | 159.91 | 158.76 | 161.06 | 159.91 | 1.15 |
| centroid | anatomy hidden | 62.13 | 62.24 | 61.08 | 61.82 | 0.64 |
| centroid | purity | 40.01 | 39.13 | 38.09 | 39.08 | 0.96 |
| centroid | mask ratio | 40.29 | 40.48 | 40.57 | 40.45 | 0.14 |
| centroid | context kept | 45.50 | 45.48 | 45.14 | 45.37 | 0.20 |
| centroid | loss slots | 158.99 | 159.59 | 159.81 | 159.46 | 0.43 |
| envelope | anatomy hidden | 77.58 | 77.37 | 76.84 | 77.26 | 0.38 |
| envelope | purity | 43.30 | 42.83 | 42.91 | 43.01 | 0.25 |
| envelope | mask ratio | 46.48 | 45.97 | 45.30 | 45.92 | 0.59 |
| envelope | context kept | 40.63 | 40.67 | 41.73 | 41.01 | 0.62 |
| envelope | loss slots | 159.68 | 159.65 | 157.93 | 159.09 | 1.00 |
| cover | anatomy hidden | 73.55 | 73.61 | 73.07 | 73.41 | 0.30 |
| cover | purity | 44.19 | 43.23 | 43.09 | 43.50 | 0.60 |
| cover | mask ratio | 43.19 | 43.34 | 42.89 | 43.14 | 0.22 |
| cover | context kept | 43.18 | 43.42 | 43.21 | 43.27 | 0.13 |
| cover | loss slots | 159.09 | 159.68 | 158.79 | 159.19 | 0.45 |
| anatomy-v2 | anatomy hidden | 79.89 | 80.47 | 80.61 | 80.32 | 0.39 |
| anatomy-v2 | purity | 97.09 | 97.16 | 96.97 | 97.08 | 0.10 |
| anatomy-v2 | mask ratio | 21.35 | 21.08 | 21.03 | 21.15 | 0.17 |
| anatomy-v2 | context kept | 67.74 | 68.23 | 67.74 | 67.90 | 0.28 |
| anatomy-v2 | loss slots | 64.00 | 64.00 | 64.00 | 64.00 | 0.00 |

primary meta: {"dataset": "FairVision-glaucoma", "split": "Training", "volumes": 24, "slices": 600, "seed": 42, "slices_per_volume": 25, "batch_size": 1, "cover_floor": 0.21, "occupancy_threshold": 0.25, "guide_dir": "C:\\jepa_data\\mirage_soft_guides\\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy", "anatomy_reference": "MIRAGE guide occupancy channel 0 >= 0.25", "grid": "16x16", "total_patches": 256}
delivered meta: {"dataset": "FairVision-glaucoma", "split": "Training", "volumes": 24, "slices": 600, "seed": 42, "slices_per_volume": 25, "batch_size": 64, "cover_floor": 0.21, "occupancy_threshold": 0.25, "guide_dir": "C:\\jepa_data\\mirage_soft_guides\\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy", "anatomy_reference": "MIRAGE guide occupancy channel 0 >= 0.25", "grid": "16x16", "total_patches": 256}