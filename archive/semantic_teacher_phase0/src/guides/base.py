"""Common tensor contract for frozen semantic vision guides."""

from dataclasses import dataclass, field
import gc
import math
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class GroundingBox:
    """A model-native box, without rasterization or inferred extent."""

    label: str
    bbox_2d: Tuple[float, float, float, float]
    coordinate_space: str = "normalized_1000"
    image_index: int = 0
    image_size: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("grounding box label must be a nonempty string")
        if not isinstance(self.bbox_2d, (list, tuple)) or len(self.bbox_2d) != 4:
            raise ValueError("bbox_2d must contain four coordinates")
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in self.bbox_2d
        ):
            raise ValueError("bbox_2d coordinates must be numeric")
        coordinates = tuple(float(value) for value in self.bbox_2d)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("bbox_2d coordinates must be finite")
        x1, y1, x2, y2 = coordinates
        upper_x, upper_y = _coordinate_bounds(
            self.coordinate_space, self.image_size
        )
        if (
            x1 < 0
            or y1 < 0
            or x2 > upper_x
            or y2 > upper_y
            or x1 >= x2
            or y1 >= y2
        ):
            raise ValueError("bbox_2d must be ordered and in bounds")
        if (
            isinstance(self.image_index, bool)
            or int(self.image_index) != self.image_index
            or int(self.image_index) < 0
        ):
            raise ValueError("image_index must be nonnegative")
        self.bbox_2d = coordinates
        self.image_index = int(self.image_index)


@dataclass
class GroundingPoint:
    """A model-native point in normalized-1000 or original-pixel space."""

    label: str
    point_2d: Tuple[float, float]
    coordinate_space: str
    image_index: int = 0
    image_size: Optional[Tuple[int, int]] = None
    object_id: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("grounding point label must be a nonempty string")
        if not isinstance(self.point_2d, (list, tuple)) or len(self.point_2d) != 2:
            raise ValueError("point_2d must contain two coordinates")
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in self.point_2d
        ):
            raise ValueError("point_2d coordinates must be numeric")
        coordinates = tuple(float(value) for value in self.point_2d)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("point_2d coordinates must be finite")
        upper_x, upper_y = _coordinate_bounds(
            self.coordinate_space, self.image_size
        )
        if (
            coordinates[0] < 0
            or coordinates[1] < 0
            or coordinates[0] > upper_x
            or coordinates[1] > upper_y
        ):
            raise ValueError("point_2d must be in bounds")
        if (
            isinstance(self.image_index, bool)
            or int(self.image_index) != self.image_index
            or int(self.image_index) < 0
        ):
            raise ValueError("image_index must be nonnegative")
        self.point_2d = coordinates
        self.image_index = int(self.image_index)
        if self.object_id is not None:
            if (
                isinstance(self.object_id, bool)
                or int(self.object_id) != self.object_id
                or int(self.object_id) < 0
            ):
                raise ValueError("object_id must be a nonnegative integer")
            self.object_id = int(self.object_id)


def _coordinate_bounds(coordinate_space, image_size):
    if coordinate_space == "normalized_1000":
        return 1000.0, 1000.0
    if coordinate_space != "pixels":
        raise ValueError("unsupported coordinate space %r" % coordinate_space)
    if image_size is None or len(image_size) != 2:
        raise ValueError("pixel coordinates require image_size=(width, height)")
    width, height = int(image_size[0]), int(image_size[1])
    if (
        width != image_size[0]
        or height != image_size[1]
        or width <= 0
        or height <= 0
    ):
        raise ValueError("image_size must contain positive integers")
    return float(width), float(height)


@dataclass
class GuideOutput:
    """Frozen features plus optional generated and model-native grounding data."""

    patch_tokens: torch.Tensor
    grid_size: Tuple[int, int]
    global_token: Optional[torch.Tensor] = None
    native_map: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_text: Optional[List[str]] = None
    grounding_regions: Optional[List[List[GroundingBox]]] = None
    grounding_points: Optional[List[List[GroundingPoint]]] = None
    raw_generation: Optional[List[Dict[str, Any]]] = None
    original_image_sizes: Optional[List[Tuple[int, int]]] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    latency_seconds: Optional[List[float]] = None
    memory_telemetry: Optional[List[Dict[str, int]]] = None
    spatial_metadata: Optional[List[Dict[str, Any]]] = None
    failures: Optional[List[List[Dict[str, str]]]] = None

    def __post_init__(self):
        if self.patch_tokens.dim() != 3:
            raise ValueError(
                "patch_tokens must have shape (B, N, D); got %s"
                % (tuple(self.patch_tokens.shape),)
            )
        height, width = self.grid_size
        if height <= 0 or width <= 0:
            raise ValueError("grid_size must contain positive integers")
        if self.patch_tokens.size(1) != height * width:
            raise ValueError(
                "patch token count %d does not match grid %dx%d"
                % (self.patch_tokens.size(1), height, width)
            )
        if self.global_token is not None:
            expected = (self.patch_tokens.size(0), self.patch_tokens.size(2))
            if tuple(self.global_token.shape) != expected:
                raise ValueError(
                    "global_token must have shape %s; got %s"
                    % (expected, tuple(self.global_token.shape))
                )
        if self.native_map is not None:
            expected = (
                self.patch_tokens.size(0),
                height,
                width,
            )
            if tuple(self.native_map.shape) != expected:
                raise ValueError(
                    "native_map must have shape %s; got %s"
                    % (expected, tuple(self.native_map.shape))
                )
        batch_fields = (
            "generated_text",
            "grounding_regions",
            "grounding_points",
            "raw_generation",
            "original_image_sizes",
            "latency_seconds",
            "memory_telemetry",
            "spatial_metadata",
            "failures",
        )
        for field_name in batch_fields:
            value = getattr(self, field_name)
            if value is not None and len(value) != self.batch_size:
                raise ValueError(
                    "%s must contain one item per batch element; got %d for "
                    "batch size %d"
                    % (field_name, len(value), self.batch_size)
                )
        if self.grounding_regions is not None:
            for regions in self.grounding_regions:
                if not all(isinstance(item, GroundingBox) for item in regions):
                    raise TypeError(
                        "grounding_regions entries must be GroundingBox values"
                    )
        if self.grounding_points is not None:
            for points in self.grounding_points:
                if not all(isinstance(item, GroundingPoint) for item in points):
                    raise TypeError(
                        "grounding_points entries must be GroundingPoint values"
                    )
        if self.original_image_sizes is not None:
            for image_size in self.original_image_sizes:
                if (
                    len(image_size) != 2
                    or int(image_size[0]) <= 0
                    or int(image_size[1]) <= 0
                ):
                    raise ValueError(
                        "original_image_sizes must contain positive "
                        "(width, height) pairs"
                    )

    @property
    def batch_size(self):
        return int(self.patch_tokens.size(0))

    @property
    def embed_dim(self):
        return int(self.patch_tokens.size(2))


class SemanticGuide:
    """Base class for frozen dense semantic guides."""

    name = ""
    default_model_id = ""

    def __init__(
        self,
        model_id=None,
        device="auto",
        dtype="auto",
        cache_dir=None,
        local_files_only=False,
        **kwargs
    ):
        self.model_id = model_id or self.default_model_id
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self.kwargs = dict(kwargs)
        self.model = None
        self.processor = None
        self._load_model()

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _resolve_dtype(self, dtype):
        if dtype == "auto":
            return torch.float16 if self.device.type == "cuda" else torch.float32
        aliases = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if isinstance(dtype, str):
            if dtype.lower() not in aliases:
                raise ValueError("unsupported guide dtype %r" % dtype)
            return aliases[dtype.lower()]
        return dtype

    def _pretrained_kwargs(self):
        kwargs = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        revision = self.kwargs.get("revision")
        if revision:
            kwargs["revision"] = revision
        if self.dtype != torch.float32:
            kwargs["dtype"] = self.dtype
        return kwargs

    @staticmethod
    def validate_images(images):
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.dim() != 4 or images.size(1) != 3:
            raise ValueError(
                "images must have shape (B, 3, H, W); got %s"
                % (tuple(images.shape),)
            )
        if not torch.is_floating_point(images):
            raise TypeError("images must be floating point in [0, 1]")
        if not bool(torch.isfinite(images).all().item()):
            raise ValueError("images contain NaN or Inf")
        minimum = float(images.min().item())
        maximum = float(images.max().item())
        if minimum < -1e-5 or maximum > 1.0 + 1e-5:
            raise ValueError(
                "images must be in [0, 1]; observed [%.4f, %.4f]"
                % (minimum, maximum)
            )

    def _load_model(self):
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, images):
        raise NotImplementedError

    @torch.no_grad()
    def encode_features(self, images):
        """Return frozen visual features without optional generative analysis."""
        return self.encode(images)

    def cleanup(self):
        self.processor = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self):
        return "<%s name=%s model_id=%s device=%s dtype=%s>" % (
            self.__class__.__name__,
            self.name,
            self.model_id,
            self.device,
            self.dtype,
        )
