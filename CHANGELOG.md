# Changelog

## Unreleased

### Changed

- Narrowed the active semantic-teacher plan to local ImageNet-50 frozen I-JEPA/DINOv3/Qwen3-VL/Molmo evaluation and map review; guided training, adaptation, and OCT transfer are deferred until a positive screen.

### Added

- Phase 0 frozen semantic-guide API with lazy DINOv3, SigLIP 2, CLIP, and local I-JEPA adapters.
- Deterministic semantic-map atlas CLI with anonymized artifacts, model-native maps, token PCA, label-free diagnostics, and atomic output handling.
- Isolated Phase 0 dependency manifest, guide configuration, and download-free unit tests.
