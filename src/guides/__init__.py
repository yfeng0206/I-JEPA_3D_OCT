"""Guide adapters for target-allocation research.

The active path is the MIRAGE retinal envelope (:mod:`src.guides.mirage_envelope`),
consumed by :class:`src.datasets.oct_slices_guided.GuidedOCTSliceDataset`.

The frozen semantic-teacher adapters that used to be registered here — DINOv3,
SigLIP2, CLIP, I-JEPA rollout, Qwen3-VL and Molmo — were parked on 2026-07-31
under ``archive/semantic_teacher_phase0/``.  See that directory's README for
what the Phase-0 screen concluded and how to revive them.  The registry is left
in place, and deliberately empty, so a caller gets an explicit pointer rather
than an ImportError.
"""

from importlib import import_module


# name -> (module, class).  Empty while the semantic-teacher thread is archived.
_GUIDE_CLASSES = {}

_ARCHIVED_GUIDES = ("clip", "dinov3", "ijepa", "molmo", "qwen3_vl", "siglip2")


def build_guide(name, **kwargs):
    """Build a registered guide without importing unrelated model packages."""
    key = str(name).lower()
    if key in _ARCHIVED_GUIDES:
        raise ValueError(
            "semantic guide %r was archived to archive/semantic_teacher_phase0/ "
            "when the project moved to MIRAGE-guided masking. See that "
            "directory's README to revive it." % name
        )
    if key not in _GUIDE_CLASSES:
        raise ValueError(
            "unknown semantic guide %r. Known guides: %s"
            % (name, sorted(_GUIDE_CLASSES))
        )
    module_name, class_name = _GUIDE_CLASSES[key]
    cls = getattr(import_module(module_name), class_name)
    return cls(**kwargs)


def available_guides():
    return tuple(sorted(_GUIDE_CLASSES))
