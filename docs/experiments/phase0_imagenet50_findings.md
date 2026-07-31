# Phase-0 ImageNet-50 Preliminary Findings

**Status:** implementation and ImageNet-50 materialization complete; DINOv3-7B
access is approved and the size-matched checkpoint is being validated locally.

This document separates completed research/tooling evidence from results that
cannot yet be produced. The locked plan remains
[`semantic_teacher_guided_masking.md`](semantic_teacher_guided_masking.md).

## Research findings

- No canonical ImageNet-50 benchmark was found. The project uses the explicitly
  noncanonical first 50 WNIDs from CMC's published ImageNet-100 order.
- Meta published no I-JEPA ImageNet-50/100 kNN or linear result. Its official
  ViT-H/14 ImageNet-1K frozen-linear result is 79.3%; it is context, not a local
  baseline.
- The local comparison uses official standard checkpoints and matches the
  primary deployed guide systems at the 7B/8B scale:

| Model | Total parameters | Visual-stack parameters |
|---|---:|---:|
| I-JEPA ViT-H/14 | 630,762,240 | 630,762,240 |
| DINOv3 ViT-7B/16 | 6.716B | 6.716B |
| Qwen3-VL-8B | 8.767B | 576,388,336 |
| MolmoPoint-8B | 8.678B | 469,115,216 |

- The three literature-backed visualization paths are final CLS attention
  adapted to DINOv3, all-patch three-component PCA adapted from the official
  DINOv3 notebook, and TokenCut-style NCut using the pinned official TokenCut
  implementation.

Full citations and protocol details are in
[`phase0_imagenet50_evidence.md`](phase0_imagenet50_evidence.md).

## Locked-plan reconciliation

The answer to “is Phase 0 complete?” is **no**. Research and implementation are
ready, but the required ImageNet experiment has not run.

| Locked requirement | State | Evidence or missing output |
|---|---|---|
| Audit official/credible I-JEPA checkpoints and results | Complete | Official ImageNet-1K tables, transfer tables, checkpoint, architecture, protocol, hashes, and third-party ImageNet-100 context are recorded |
| Find an official I-JEPA ImageNet-50/100 frozen kNN/linear table | Not found | No directly comparable public result exists as of the evidence cutoff |
| Obtain an I-JEPA checkpoint usable for ImageNet-50 | Complete | Official ImageNet-1K-pretrained ViT-H/14 `target_encoder`; no ImageNet-50-specific checkpoint is needed for frozen evaluation |
| Pin a reproducible ImageNet-50 definition | Research complete | CMC-derived 50-WNID manifest and expected 63,747/2,500 counts are pinned |
| Materialize and verify ImageNet-50 files | Complete | 63,747 train + 2,500 validation images; every file SHA-256-recorded and decode-verified |
| Download all four checkpoints | In progress | I-JEPA, Qwen3-VL, and MolmoPoint are local; approved DINOv3-7B is downloading |
| Run 20-image Qwen/Molmo grounding gate | Ready, not run | Deterministic first-20-WNID manifest exists; GPU is currently reserved by the user |
| Run frozen ImageNet-50 kNN/linear for all encoders | Ready, not run | Data/code and access are ready; size-matched DINOv3-7B BF16 smoke runs first |
| Generate 100-200-image atlas and manually review 30-50 | Ready, not run | Deterministic 150-image manifest exists; DINOv3-7B and VLM panels remain |
| Report grounding/stability/coverage/hallucination diagnostics | Blocked | Requires the manifest-locked ImageNet sample and annotations/manual review |
| Lock one readout/guide and decide Phase 1 | Blocked | DINO versus VLM comparison has not run |

The existing official checkpoint is:

```text
Meta I-JEPA ViT-H/14, ImageNet-1K, 300 epochs
state: target_encoder
parameters: 630,762,240
input/patch: 224 / 14
raw SHA-256:
0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22
extracted safetensors SHA-256:
03fdafbd89f4e20184a83e5ce71a605fe0ff2ce69aee9eb4b6dabc8c8f5cd899
```

There is therefore a valid checkpoint but **no I-JEPA ImageNet-50 benchmark
number yet**. The locked plan explicitly requires computing that number locally
when no published table exists.

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
D:\Users\Gary\Desktop\ImageNet_Data\
  phase0-cmc-in100-prefix50-v1
  manifests\phase0-grounding-20.json
  manifests\phase0-atlas-150.json
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

Official ImageNet-50 files and approved DINOv3-7B access are now present.
Unauthorized mirrors were not used.

**Current decision:** the implementation is ready, but the scientific decision
is **RUN THE SIZE-MATCHED PHASE-0 TESTS**. Qwen and Molmo both pass one-image
API/memory checks; neither has passed the reliability gate, and DINOv3-7B has
not yet been compared.
