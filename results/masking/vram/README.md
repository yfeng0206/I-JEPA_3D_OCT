# FP32 Downstream VRAM & Throughput Results

GPU: RTX 3090 24 GB | Precision: strict fp32 (TF32 disabled) | Safety threshold: 20,000 MB

## TF32 Numerical Impact
| Metric | Value |
|--------|-------|
| Max absolute diff | 0.305 |
| Max relative diff | 3.1% |
| **Verdict** | **Unacceptable for strict fp32 — disabled** |

## Frozen Probe Encoding (per-volume, no_grad)
| chunk_size | slices/sec | peak MB |
|-----------:|-----------:|--------:|
| 25 | 299.7 | 658 |
| 50 | 305.5 | 994 |
| **100** | **306.5** | **1,653** |

Recommendation: chunk_size=100. Probe training batch_size=256 on cached features adds only ~75 MB.

## Partial Freeze (batch=1 vol, num_slices=100)
| Frozen blocks | Trainable blocks | Peak MB | slices/sec | sec/vol (100sl) | Status |
|--------------:|-----------------:|--------:|-----------:|----------------:|--------|
| 12 (all) | 0 | 1,690 | 309 | 0.32 | ✅ frozen probe |
| 9 | 3 | 5,453 | 219 | 0.46 | ✅ safe |
| 6 | 6 | 9,961 | 168 | 0.59 | ✅ **sweet spot** |
| 3 | 9 | 14,468 | 137 | 0.73 | ✅ safe |
| 0 (all) | 12 | 18,975 | 115 | 0.87 | ✅ fits (5 GB headroom) |

## Recommendations
- **Frozen probe**: encode_chunk_size=100, batch_size=256 for probe training. ~0.32 sec/vol.
- **Full fine-tune**: batch_size=1, num_slices=100, accum_steps=4. ~0.87 sec/vol. 18.9 GB peak.
- **Partial fine-tune (fast)**: freeze first 6 blocks, same settings. ~0.59 sec/vol. 9.9 GB peak.

## Existing Config Safety
`frozen_meanpool_mirage_ep{50,75,100}.yaml` with batch_size=256, num_slices=100:
- **Will NOT OOM.** batch_size applies to cached-feature probe training, not encoding.
- **BUT**: `precompute_features()` uses `autocast()` on line ~362 of eval_downstream.py.
  For strict fp32, this autocast must be removed or disabled.

## Gradient Checkpointing
Not supported in `src/models/vision_transformer.py` — no flag exists.
