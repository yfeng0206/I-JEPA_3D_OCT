"""Common tensor contract for frozen semantic vision guides."""

from dataclasses import dataclass, field
import gc
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class GuideOutput:
    """Dense guide features and an optional model-native importance map."""

    patch_tokens: torch.Tensor
    grid_size: Tuple[int, int]
    global_token: Optional[torch.Tensor] = None
    native_map: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
            kwargs["torch_dtype"] = self.dtype
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
