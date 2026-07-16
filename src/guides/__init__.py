"""Frozen semantic-guide adapters for target-allocation research.

Guide implementations are imported lazily so the active I-JEPA training path
does not require transformers or external model packages.
"""

from importlib import import_module


_GUIDE_CLASSES = {
    "dinov3": ("src.guides.hf_guides", "DINOv3Guide"),
    "siglip2": ("src.guides.hf_guides", "SigLIP2Guide"),
    "clip": ("src.guides.hf_guides", "CLIPGuide"),
    "ijepa": ("src.guides.ijepa", "IJEPAGuide"),
    "qwen3_vl": ("src.guides.vlm_guides", "Qwen3VLGuide"),
    "molmo": ("src.guides.vlm_guides", "MolmoPointGuide"),
}


def build_guide(name, **kwargs):
    """Build a registered guide without importing unrelated model packages."""
    key = str(name).lower()
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


from .base import (  # noqa: E402,F401
    GroundingBox,
    GroundingPoint,
    GuideOutput,
    SemanticGuide,
)
