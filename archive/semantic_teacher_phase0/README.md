# Semantic-teacher guided masking — Phase 0 (archived)

**Status: parked 2026-07-31. Superseded by
[`docs/experiments/mirage_guided_masking.md`](../../docs/experiments/mirage_guided_masking.md).**

This directory holds a complete, self-contained research thread that is no
longer on the active path. Nothing here is imported by training, evaluation or
the MIRAGE pipeline — the boundary was verified before archiving: the live
training path touches `src/guides/` only through `mirage_envelope.py`.

## What this was

The question was whether a **frozen foundation model** could supply the
target-block prior for I-JEPA masking, instead of a hand-crafted anatomical
band. Rather than guess, Phase 0 screened candidate teachers on ImageNet-50 —
where ground truth is cheap and the failure modes are visible — before spending
GPU time on OCT.

Teachers screened: I-JEPA (self-attention rollout), DINOv3 (PCA / attention),
Qwen3-VL and Molmo (phrase grounding and pointing), SAM3, TokenCut, and a CNN
stage-channel atlas as a depth analogy.

## Why it was parked

The screen did its job: it showed that VLM-derived maps were not a usable
target-block prior for this problem.

- Grounding-based guides are **object-centric**. They localise "the object",
  which on a retinal B-scan is close to "the whole bright band" — not a
  discriminative sub-region.
- Their outputs are **soft, unstable across prompts, and not anatomically
  addressable**. There is no way to say "the RNFL" and get the RNFL.
- Every candidate needed a per-image forward pass through a large model, which
  is far more expensive than a precomputed segmentation cache.

A medical segmentation model answers the same need directly: **MIRAGE-Large**
(GOALS-trained) returns RNFL / GCIPL / choroid, which is exactly the anatomy the
oracle band was hand-approximating. That became the active direction.

The detailed Phase-0 evidence, protocol and findings are in `docs/` here.

## Layout

```
scripts/    phase0 evaluation, semantic map atlas, CNN stage atlas,
            DINOv3 PCA atlas, Qwen phrase grounding, grounding gate
src/guides/ the guide registry's implementations: base, maps, ijepa, tokencut,
            hf_guides (DINOv3/SigLIP2/CLIP), vlm_guides (Qwen3-VL/Molmo),
            dino_pca, cnn_stages
src/evaluation/  grounding metrics, feature cache, ImageNet frozen probe
tests/      the thread's test suite (moved out of the active suite with it)
docs/       phase0 evidence, findings, Qwen/SAM3 screen, and the original
            semantic-teacher research plan
configs/    guide manifests and model pins
```

## Reviving it

The code is intact and its suite was green when parked. It does **not** run
from here as-is: the tests import `src.guides.*` and `src.evaluation.*`, which
now live under this directory, so they fail at collection. `pytest.ini`
excludes `archive/` from collection for exactly that reason — a bare `pytest`
from the repo root must stay green.

To bring the thread back:

1. move `src/guides/*` and `src/evaluation/*` back to their original paths
2. restore the entries in `src/guides/__init__.py` (`_GUIDE_CLASSES`), which
   were emptied so the active registry only advertises what is reachable;
   `build_guide` currently raises a pointer to this README for archived names
3. move `tests/*` back so they rejoin the active suite
4. re-pin models from `configs/semantic_maps/manifests/phase0_models.yaml`

The model weights themselves live outside the repo under `D:\jepa_phase0\models`.
