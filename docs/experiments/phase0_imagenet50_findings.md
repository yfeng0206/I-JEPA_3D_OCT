# Phase-0 ImageNet-50 Preliminary Findings

**Status:** implementation complete; full experiment blocked on ImageNet and
DINOv3 access.

This document separates completed research/tooling evidence from results that
cannot yet be produced. The locked plan remains
[`semantic_teacher_guided_masking.md`](semantic_teacher_guided_masking.md).

## Research findings

- No canonical ImageNet-50 benchmark was found. The project uses the explicitly
  noncanonical first 50 WNIDs from CMC's published ImageNet-100 order.
- Meta published no I-JEPA ImageNet-50/100 kNN or linear result. Its official
  ViT-H/14 ImageNet-1K frozen-linear result is 79.3%; it is context, not a local
  baseline.
- The local comparison uses official standard checkpoints rather than inventing
  parameter-matched variants:

| Model | Frozen parameters relevant to comparison |
|---|---:|
| I-JEPA ViT-H/14 | 630,762,240 |
| DINOv3 ViT-L/16 | 303,129,600 artifact / 300M official |
| Qwen3-VL visual tower | 576,388,336 |
| Molmo vision backbone + connector | 469,115,216 |

- The three literature-backed visualization paths are final CLS attention
  adapted to DINOv3, all-patch three-component PCA adapted from the official
  DINOv3 notebook, and TokenCut-style NCut using the pinned official TokenCut
  implementation.

Full citations and protocol details are in
[`phase0_imagenet50_evidence.md`](phase0_imagenet50_evidence.md).

## Completed implementation

- Explicit-WNID ImageNet subset loading and content-addressed dataset snapshots.
- Resumable, atomic, SHA-256-verified FP32 feature caches.
- Weighted cosine kNN (`k=20`, temperature `0.07`) and a deterministic
  encoder-frozen multinomial linear probe.
- Frozen I-JEPA, DINOv3, Qwen3-VL, and MolmoPoint adapters.
- Separate VLM generation and feature-only paths; no class label is supplied to
  either VLM.
- Generated captions, native boxes/points, shaded grounding maps, PCA,
  TokenCut-style NCut, difference maps, and explicitly non-training target-block
  illustrations.
- Molmo pooled tokens are marked nonspatial and are never reshaped into a fake
  image grid.

## Real local integration results

These one-image checks validate APIs and memory only. They do not satisfy the
locked 20-image gate.

| Model | Observed output | Shape | Latency | Peak reserved |
|---|---|---:|---:|---:|
| Qwen3-VL-8B | Factual caption, valid dog box, valid dog point | 256 x 4096, 16 x 16 grid | 8.26 s | 16.66 GiB |
| MolmoPoint-8B | Factual marina caption, one native point, no box | 392 x 4096 pooled nonspatial tokens | 13.30 s | 17.64 GiB |

On the Qwen demo image, Qwen's box and point cover the dog while its caption
also describes the woman and interaction. The I-JEPA global-cosine diagnostic
places several highest illustrative blocks in background regions. This single
example demonstrates that the two readouts can differ; it does not establish
which masking strategy is better.

On the marina image, Molmo captions the multi-boat scene but returns one point
for one prominent boat. This preserves the official point-only contract but
also shows why multi-object coverage must be measured before the VLM arm can
advance.

The real I-JEPA feature-cache smoke also completed extraction, cache resume,
weighted kNN, and linear-probe execution end to end. Its synthetic two-class
accuracy is only a pipeline check and is not a benchmark result.

Local artifacts:

```text
D:\jepa_phase0\results\atlases\ijepa_qwen_smoke
D:\jepa_phase0\results\atlases\molmo_smoke_visiononly
D:\jepa_phase0\results\manifests\qwen_smoke.json
D:\jepa_phase0\results\manifests\molmo_smoke.json
```

## Validation and independent review

- Full test discovery: 66 tests passed.
- Real pinned TokenCut function completed on a 14 x 14 feature grid.
- Real I-JEPA, Qwen3-VL, and MolmoPoint paths completed on the RTX 3090.
- Claude Opus 4.8 at max effort performed the final implementation review and a
  second post-fix regression review.
- Both reviews found no P0/P1 issue. The first review's two P2 findings were
  fixed: cache identity no longer includes unrelated runtime settings, and
  Molmo feature caching bypasses the text decoder through the pinned official
  ViT and connector modules.
- The second review found no remaining blocking or non-blocking bug and retained
  a **GO** verdict.

## Blocked results and decision

The following are not available and must not be inferred:

- 20-image ImageNet grounding-gate rates;
- 100-200-image ImageNet atlas comparisons;
- frozen ImageNet-50 kNN or linear-probe values;
- DINOv3 attention, PCA, TokenCut, or DINO-VLM difference results;
- a Phase-1 guide selection.

No official ImageNet archive/credentials are configured, and DINOv3 still
requires approved access. Unauthorized mirrors were not used.

**Current decision:** the implementation is ready, but the scientific decision
is **WAIT FOR ACCESS**. Qwen and Molmo both pass one-image API/memory checks;
neither has passed the reliability gate, and DINOv3 has not yet been compared.
