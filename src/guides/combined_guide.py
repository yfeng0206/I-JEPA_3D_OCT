"""Combined anatomical guide: MIRAGE segmentation unioned with image salience.

MIRAGE is trained on GOALS and applied to FairVision, and the transfer is
imperfect -- it reliably finds the major layers but drops tissue at the retinal
margins and in dim choroid.  Image intensity finds that missing tissue but has
no notion of anatomy and cannot bridge an optic-nerve-head gap or reject a
bright artefact.  The two failure modes are largely disjoint, so the union is a
more complete retinal prior than either alone.

Pipeline, all at native resolution before any crop:

    raw MIRAGE union  \\
                       >--- OR ---> repair_union ---> combined envelope
    salience band     /

Repair runs *after* the union so the combined region is closed and made
coherent as one structure, rather than two structures stitched together.

Naming note: the result is no longer "the MIRAGE envelope".  It is a
MIRAGE-plus-intensity retinal envelope, and experiments using it must be
described that way -- the guide is partly derived from the image itself.

Evaluation warning: any tissue reference built from intensity is *inside* this
guide, so purity measured against it is circular and saturates near 1.0.  Score
this guide against a reference it does not contain -- ``pixel_otsu`` is the
stricter bright-tissue rule kept for exactly this purpose.
"""

from __future__ import annotations

import numpy as np

from src.guides.mirage_envelope import (
    DEFAULT_REPAIR,
    RepairParams,
    build_union,
    repair_union,
)
from src.guides.tissue_truth import DEFAULT_K, tissue_pixels_noise_band

__all__ = [
    "salience_band",
    "build_combined_union",
    "combined_envelope",
    "combined_fingerprint",
]


def salience_band(
    image: np.ndarray, k: float = DEFAULT_K, min_run: int = 6
) -> np.ndarray:
    """Intensity-derived retinal band at the image's native resolution.

    Thin wrapper over the calibrated tissue reference so callers do not have to
    know which knobs matter; ``k`` is the sigma multiplier above the vitreous
    noise floor and is the only one that materially changes the result.
    """
    return tissue_pixels_noise_band(image, k=k, min_run=min_run)


def build_combined_union(
    hard_masks: np.ndarray, image: np.ndarray, k: float = DEFAULT_K
) -> np.ndarray:
    """Union the raw MIRAGE prediction with the salience band."""
    if hard_masks.shape[-2:] != image.shape[-2:]:
        raise ValueError(
            f"shape mismatch: masks {hard_masks.shape[-2:]} vs image "
            f"{image.shape[-2:]}; both must be native resolution"
        )
    return build_union(hard_masks) | salience_band(image, k=k)


def combined_envelope(
    hard_masks: np.ndarray,
    image: np.ndarray,
    params: RepairParams = DEFAULT_REPAIR,
    k: float = DEFAULT_K,
):
    """Repaired envelope over the MIRAGE-plus-salience union.

    Returns ``(envelope, is_valid, stats)`` exactly as ``repair_union`` does, so
    it is a drop-in replacement for the MIRAGE-only path.
    """
    return repair_union(build_combined_union(hard_masks, image, k=k), params=params)


def combined_fingerprint(params: RepairParams = DEFAULT_REPAIR,
                         k: float = DEFAULT_K) -> str:
    """Cache key covering both the repair parameters and the salience setting."""
    from src.guides.mirage_envelope import params_fingerprint

    return f"{params_fingerprint(params)}+sal{k:g}"
