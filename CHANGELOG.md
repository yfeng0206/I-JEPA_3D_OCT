# Changelog

## Unreleased

### Current direction

- The active workshop paper studies anatomy-guided I-JEPA target selection on
  retinal OCT. The delivered-task engineering investigation and its versioned
  corrections are tracked in `autopilot\investigations\delivered_task\BOARD.md`.
- The semantic-teacher work listed below is historical and archived, not the
  current pretraining or submission plan.

### Changed

- Narrowed the active semantic-teacher plan to local ImageNet-50 frozen I-JEPA/DINOv3/Qwen3-VL/Molmo evaluation and map review; guided training, adaptation, and OCT transfer are deferred until a positive screen.

### Added

- Phase 0 frozen semantic-guide API with lazy DINOv3, SigLIP 2, CLIP, and local I-JEPA adapters.
- Deterministic manifest-driven semantic-map atlas CLI with generated captions, native box/point overlays, token PCA, TokenCut-style NCut, illustrative target blocks, anonymized artifacts, and atomic output handling.
- Isolated Phase 0 dependency manifest, guide configuration, and download-free unit tests.
- Locked ImageNet-50 class, model, checkpoint, license, and frozen-evaluation evidence manifests for local Phase-0 execution.
- Resumable integrity-checked frozen-feature caches with weighted kNN and encoder-frozen linear-probe evaluation.
- Phase-0 preliminary findings with real I-JEPA/Qwen3-VL/MolmoPoint integration evidence and explicit ImageNet/DINOv3 access blockers.
- Ten-image Qwen3-VL concept-selection and SAM3 continuous-segmentation screen for candidate I-JEPA importance maps.
