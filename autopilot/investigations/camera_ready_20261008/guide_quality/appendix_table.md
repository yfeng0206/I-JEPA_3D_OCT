| Rule | Overall (pop.-wt.) | Normal | Glaucoma | SNR low | SNR mid | SNR high |
|---|---:|---:|---:|---:|---:|---:|
| CENTROID F1: IoU<.20 or >50% ribbon cells with zero envelope | 4.8% [3.7, 5.9] | 2.6% [1.7, 3.6] | 6.5% [4.8, 8.3] | 7.5% [5.2, 9.9] | 3.8% [2.4, 5.4] | 2.3% [1.2, 3.5] |
| CENTROID F2: >=3/10 ribbon columns off-centre/missed | 0.9% [0.4, 1.4] | 0.3% [0.1, 0.7] | 1.3% [0.5, 2.1] | 1.7% [0.7, 2.9] | 0.3% [0.0, 0.8] | 0.4% [0.0, 1.1] |
| CENTROID F3 (strict): IoU<.20 or >50% ribbon cells occ<.25 | 24.9% [22.5, 27.4] | 18.3% [15.9, 20.6] | 31.6% [27.8, 35.6] | 33.7% [29.0, 38.5] | 24.6% [20.4, 28.5] | 16.5% [13.4, 19.9] |
| Envelope empty | 0.0% [0.0, 0.1] | 0.1% [0.0, 0.2] | 0.0% [0.0, 0.0] | 0.1% [0.0, 0.3] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Envelope invalid (stored or centre view) | 0.0% [0.0, 0.1] | 0.1% [0.0, 0.2] | 0.1% [0.0, 0.2] | 0.1% [0.0, 0.3] | 0.0% [0.0, 0.0] | 0.1% [0.0, 0.3] |
| Envelope >=3 components | 1.5% [0.9, 2.0] | 1.3% [0.7, 1.9] | 1.4% [0.7, 2.2] | 1.3% [0.4, 2.4] | 1.2% [0.6, 2.0] | 1.5% [0.8, 2.3] |
| Envelope thickness irregular (>20% outlier cols) | 2.5% [1.6, 3.4] | 1.7% [0.8, 2.6] | 3.3% [1.9, 5.0] | 3.2% [1.6, 5.2] | 2.1% [0.9, 3.5] | 2.2% [0.8, 4.0] |
| Envelope centroid outside ribbon (median |offset|>3.5) | 0.4% [0.1, 0.7] | 0.0% [0.0, 0.0] | 0.6% [0.1, 1.2] | 0.7% [0.1, 1.6] | 0.1% [0.0, 0.3] | 0.1% [0.0, 0.3] |
| Any envelope flag | 3.8% [2.7, 5.0] | 2.7% [1.7, 3.9] | 4.7% [3.1, 6.8] | 4.6% [2.4, 7.1] | 3.3% [2.0, 4.8] | 3.2% [1.7, 5.1] |

| CENTROID rule by tilt/curvature (centroid-range tertile) | low | mid | high |
|---|---:|---:|---:|
| F1 | 7.9% [5.8, 10.4] | 2.9% [1.8, 4.3] | 2.8% [1.8, 4.0] |
| F2 | 1.8% [0.7, 3.0] | 0.6% [0.1, 1.3] | 0.0% [0.0, 0.0] |
| F3 | 27.0% [23.0, 30.8] | 20.0% [16.6, 23.5] | 27.9% [24.4, 31.8] |

| Continuous (median [P5, P95]) | All | Normal | Glaucoma |
|---|---:|---:|---:|
| Ribbon-envelope IoU | 0.406 [0.3, 0.484] | 0.415 [0.317, 0.484] | 0.398 [0.276, 0.474] |
| Ribbon cells with zero envelope | 0.357 [0.243, 0.5] | 0.343 [0.243, 0.471] | 0.371 [0.257, 0.529] |
| Envelope covered by ribbon | 0.621 [0.532, 0.667] | 0.621 [0.547, 0.667] | 0.619 [0.511, 0.667] |
| Median |envelope centroid - ribbon centre| (patches) | 0.661 [0.218, 2.05] | 0.587 [0.208, 1.91] | 0.757 [0.22, 2.18] |
| Envelope median thickness (patches) | 3.36 [2.8, 3.92] | 3.44 [2.96, 3.92] | 3.28 [2.72, 3.84] |
| Native columns without envelope | 0 [0, 0.01] | 0 [0, 0.005] | 0 [0, 0.015] |
| Envelope components (native) | 1 [1, 2] | 1 [1, 2] | 1 [1, 2] |

| Census check (all 6000 Training volumes x 10 B-scans) | Overall | Normal | Glaucoma |
|---|---:|---:|---:|
| CENTROID F1: IoU<.20 or >50% ribbon cells with zero envelope | 5.3% [5.1, 5.6] | 2.7% [2.4, 2.9] | 8.1% [7.7, 8.6] |
| CENTROID F2: >=3/10 ribbon columns off-centre/missed | 1.0% [0.9, 1.2] | 0.4% [0.4, 0.5] | 1.7% [1.5, 1.9] |
| CENTROID F3 (strict): IoU<.20 or >50% ribbon cells occ<.25 | 25.8% [25.3, 26.3] | 16.6% [16.1, 17.2] | 35.5% [34.5, 36.4] |
| Envelope empty | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Envelope invalid (stored or centre view) | 0.1% [0.0, 0.1] | 0.0% [0.0, 0.0] | 0.1% [0.1, 0.2] |
| Envelope >=3 components | 1.5% [1.4, 1.7] | 1.0% [0.9, 1.1] | 2.1% [1.9, 2.3] |
| Envelope thickness irregular (>20% outlier cols) | 2.7% [2.5, 2.9] | 1.5% [1.3, 1.7] | 3.9% [3.6, 4.3] |
| Envelope centroid outside ribbon (median |offset|>3.5) | 0.5% [0.4, 0.6] | 0.2% [0.1, 0.2] | 0.9% [0.7, 1.0] |
| Any envelope flag | 4.0% [3.7, 4.2] | 2.4% [2.2, 2.7] | 5.6% [5.2, 6.1] |
