"""Local I-JEPA target-encoder guide for endogenous map baselines."""

import hashlib

import torch
import torch.nn.functional as F

from src.models.vision_transformer import VisionTransformer

from .base import GuideOutput, SemanticGuide
class IJEPAGuide(SemanticGuide):
    """Load a local I-JEPA checkpoint and expose its full patch grid."""

    name = "ijepa"
    default_model_id = "IN1K-vit.h.14-300e"

    def _load_model(self):
        weights_path = self.kwargs.get("weights_path")
        if not weights_path:
            raise ValueError("I-JEPA guide requires weights_path")
        expected_sha256 = self.kwargs.get("weights_sha256")
        if expected_sha256:
            digest = hashlib.sha256()
            with open(weights_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            observed_sha256 = digest.hexdigest()
            if observed_sha256 != str(expected_sha256).lower():
                raise RuntimeError(
                    "I-JEPA weights SHA-256 mismatch: expected %s, observed %s"
                    % (expected_sha256, observed_sha256)
                )
            self._weights_sha256 = observed_sha256
        else:
            self._weights_sha256 = None
        self.input_size = int(self.kwargs.get("input_size", 224))
        self.patch_size = int(self.kwargs.get("patch_size", 16))
        embed_dim = int(self.kwargs.get("embed_dim", 768))
        depth = int(self.kwargs.get("depth", 12))
        num_heads = int(self.kwargs.get("num_heads", 12))
        self.model = VisionTransformer(
            img_size=self.input_size,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        ).to(self.device, dtype=self.dtype).eval()
        checkpoint_key = self.kwargs.get("checkpoint_key", "target_encoder")
        if str(weights_path).lower().endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise RuntimeError(
                    "safetensors is required for extracted I-JEPA weights"
                ) from exc
            state = load_file(weights_path, device="cpu")
        else:
            checkpoint = torch.load(
                weights_path, map_location="cpu", weights_only=True, mmap=True
            )
            if not isinstance(checkpoint, dict):
                raise TypeError("I-JEPA checkpoint must be a dictionary")
            if checkpoint_key not in checkpoint:
                raise KeyError(
                    "configured checkpoint has no %r state" % checkpoint_key
                )
            if not isinstance(checkpoint[checkpoint_key], dict):
                raise TypeError(
                    "%r checkpoint state must be a dictionary"
                    % checkpoint_key
                )
            state = checkpoint[checkpoint_key]
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise TypeError(
                "%r checkpoint state must contain only tensors"
                % checkpoint_key
            )
        state = {
            key.replace("module.", ""): value for key, value in state.items()
        }
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "I-JEPA guide state mismatch: %d missing, %d unexpected"
                % (len(missing), len(unexpected))
            )

    def _prepare_images(self, images):
        self.validate_images(images)
        images = images.to(self.device, dtype=torch.float32)
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )
        mean = torch.tensor(
            (0.485, 0.456, 0.406),
            dtype=images.dtype,
            device=images.device,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            (0.229, 0.224, 0.225),
            dtype=images.dtype,
            device=images.device,
        ).view(1, 3, 1, 1)
        return ((images - mean) / std).to(dtype=self.dtype)

    @torch.no_grad()
    def encode(self, images):
        images = self._prepare_images(images)
        patches = self.model(images).float()
        height = int(self.input_size // self.patch_size)
        width = int(self.input_size // self.patch_size)
        global_token = patches.mean(dim=1)
        output = GuideOutput(
            patch_tokens=patches,
            grid_size=(height, width),
            global_token=global_token,
            metadata={
                "model_id": "local_ijepa",
                "checkpoint_key": self.kwargs.get(
                    "checkpoint_key", "target_encoder"
                ),
                "weights_format": (
                    "safetensors"
                    if str(self.kwargs.get("weights_path", "")).lower().endswith(
                        ".safetensors"
                    )
                    else "pytorch_checkpoint"
                ),
                "weights_sha256_verified": self._weights_sha256 is not None,
                "readout": "no_official_native_saliency",
                "spatial_token_grid": True,
            },
            model_metadata={
                "guide": self.name,
                "official_model_id": self.model_id,
                "dtype": str(self.dtype).replace("torch.", ""),
                "device": str(self.device),
                "frozen": True,
            },
        )
        output.metadata["global_cosine_available"] = True
        return output
