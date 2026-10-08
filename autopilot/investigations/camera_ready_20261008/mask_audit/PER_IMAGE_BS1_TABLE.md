# Per-image geometry (main Table 2 analogue): published v9 vs camera-ready

Published: archived `results/masking/table2_geometry/mask_geometry_600slices_bs1_coverf021_seed42.json` (600 views, seed 42, CENTROID lateral 0.8, ENVELOPE soft guide + least-overlap, COVER cover_fill=transition). New: `runs/per_image_bs1_trained_3seeds.json` (sha256 7d1d6e6fb84511ea), trained configs, batch size 1, 6,400 views x 3 audit seeds, paired crops/sizes/placement seeds; mean +- SD over seeds. Tissue proxy: soft MIRAGE envelope occupancy >= 0.25 (as published).

| Policy | anatomy hidden % (old -> new) | purity % (old -> new) | mask ratio % (old -> new) | context kept % (old -> new) | loss slots (old -> new) |
|---|---|---|---|---|---|
| RANDOM | 54.0 -> 53.0 +- 0.1 | 31.5 -> 30.9 +- 0.1 | 44.5 -> 44.0 +- 0.0 | 41.9 -> 42.1 +- 0.0 | 159.9 -> 159.3 +- 0.3 |
| CENTROID | 62.1 -> 59.7 +- 0.2 | 40.0 -> 38.3 +- 0.2 | 40.3 -> 39.9 +- 0.1 | 45.5 -> 45.7 +- 0.0 | 159.0 -> 159.3 +- 0.3 |
| ENVELOPE | 77.6 -> 70.5 +- 0.2 | 43.3 -> 42.1 +- 0.1 | 46.5 -> 42.9 +- 0.1 | 40.6 -> 43.7 +- 0.1 | 159.7 -> 159.3 +- 0.3 |
| COVER | 73.5 -> 73.2 +- 0.0 | 44.2 -> 40.1 +- 0.0 | 43.2 -> 46.7 +- 0.1 | 43.2 -> 39.9 +- 0.1 | 159.1 -> 159.3 +- 0.3 |
| ANATOMY-V2 | 79.9 -> 79.7 +- 0.0 | 97.1 -> 96.8 +- 0.2 | 21.3 -> 21.1 +- 0.0 | 67.7 -> 68.0 +- 0.0 | 64.0 -> 64.0 +- 0.0 |

Per-image paired context-kept gaps vs RANDOM (points of grid): centroid +3.60 (seed SD 0.06; per seed +3.55 / +3.60 / +3.66); envelope +1.61 (seed SD 0.12; per seed +1.52 / +1.56 / +1.74); anatomy +25.93 (seed SD 0.03; per seed +25.93 / +25.91 / +25.96); cover -2.24 (seed SD 0.13; per seed -2.11 / -2.26 / -2.36); centroid_lat08_published +3.60 (seed SD 0.07; per seed +3.52 / +3.64 / +3.64); envelope_soft_leastoverlap_published -0.88 (seed SD 0.02; per seed -0.89 / -0.88 / -0.85); envelope_hard_leastoverlap -2.94 (seed SD 0.09; per seed -2.98 / -3.01 / -2.85); envelope_soft_legacy +3.88 (seed SD 0.10; per seed +3.83 / +3.82 / +4.00)

Old-setting arms at bs1 (same crops/sizes): CENTROID 0.8 45.72; ENVELOPE soft+least-overlap 41.24; ENVELOPE hard+least-overlap 39.17; ENVELOPE soft+legacy 46.00 (context kept %).
