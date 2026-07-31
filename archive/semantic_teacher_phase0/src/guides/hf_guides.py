"""Hugging Face adapters for frozen DINOv3, SigLIP 2, and CLIP guides."""

from typing import Tuple

import torch
import torch.nn.functional as F

from .base import GuideOutput, SemanticGuide
from .maps import global_patch_cosine


def _grid_from_image(images, patch_size):
    height = int(images.size(-2) // patch_size)
    width = int(images.size(-1) // patch_size)
    if height <= 0 or width <= 0:
        raise ValueError("image is smaller than guide patch size")
    return height, width


class _FixedImageGuide(SemanticGuide):
    input_size = 224
    image_mean = (0.485, 0.456, 0.406)
    image_std = (0.229, 0.224, 0.225)

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
            self.image_mean, device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.image_std, device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)
        images = (images - mean) / std
        return images.to(dtype=self.dtype)


class DINOv3Guide(_FixedImageGuide):
    """Frozen DINOv3 dense-feature guide."""

    name = "dinov3"
    default_model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def _load_model(self):
        try:
            from transformers import AutoModel

            pretrained_kwargs = self._pretrained_kwargs()
            pretrained_kwargs["attn_implementation"] = self.kwargs.get(
                "attn_implementation", "eager"
            )
            self.model = AutoModel.from_pretrained(
                self.model_id, **pretrained_kwargs
            ).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "failed to load DINOv3 guide %r. The official checkpoint may "
                "require access approval or a local cached path. Original "
                "error: %s" % (self.model_id, exc)
            ) from exc
        config = self.model.config
        self.patch_size = int(getattr(config, "patch_size", 16))
        self.input_size = int(
            self.kwargs.get("input_size", getattr(config, "image_size", 224))
        )

    @torch.no_grad()
    def encode(self, images):
        images = self._prepare_images(images)
        output = self.model(
            pixel_values=images,
            output_attentions=True,
            return_dict=True,
        )
        tokens = output.last_hidden_state
        grid_size = _grid_from_image(images, self.patch_size)
        patch_count = grid_size[0] * grid_size[1]
        prefix_count = int(tokens.size(1) - patch_count)
        if prefix_count < 1:
            raise ValueError(
                "DINOv3 output has %d tokens, fewer than expected patch count %d"
                % (tokens.size(1), patch_count)
            )
        patches = tokens[:, prefix_count:]
        global_token = tokens[:, 0]
        if not output.attentions:
            raise RuntimeError(
                "DINOv3 did not return attention weights; eager attention "
                "is required for the native attention diagnostic"
            )
        last_attention = output.attentions[-1]
        native_map = last_attention[:, :, 0, prefix_count:].mean(dim=1)
        native_map = native_map.view(tokens.size(0), *grid_size)
        result = GuideOutput(
            patch_tokens=patches,
            grid_size=grid_size,
            global_token=global_token,
            native_map=native_map,
            metadata={
                "model_id": self.model_id,
                "readout": "final_cls_attention_mean_adapted",
                "special_token_count": prefix_count,
                "attention_heads": int(last_attention.size(1)),
                "spatial_token_grid": True,
            },
            model_metadata={
                "guide": self.name,
                "official_model_id": self.kwargs.get(
                    "official_model_id", self.model_id
                ),
                "dtype": str(self.dtype).replace("torch.", ""),
                "device": str(self.device),
                "frozen": True,
            },
        )
        return result

    @torch.no_grad()
    def encode_features(self, images):
        """Return frozen CLS/patch features without attention materialization."""
        images = self._prepare_images(images)
        output = self.model(
            pixel_values=images,
            output_attentions=False,
            return_dict=True,
        )
        tokens = output.last_hidden_state
        grid_size = _grid_from_image(images, self.patch_size)
        patch_count = grid_size[0] * grid_size[1]
        prefix_count = int(tokens.size(1) - patch_count)
        if prefix_count < 1:
            raise ValueError(
                "DINOv3 output has fewer tokens than the expected patch grid"
            )
        return GuideOutput(
            patch_tokens=tokens[:, prefix_count:],
            grid_size=grid_size,
            global_token=tokens[:, 0],
            metadata={
                "model_id": self.model_id,
                "readout": "final_normalized_cls_token",
                "special_token_count": prefix_count,
                "features_only": True,
                "spatial_token_grid": True,
            },
            model_metadata={
                "guide": self.name,
                "official_model_id": self.kwargs.get(
                    "official_model_id", self.model_id
                ),
                "dtype": str(self.dtype).replace("torch.", ""),
                "device": str(self.device),
                "frozen": True,
            },
        )


class SigLIP2Guide(SemanticGuide):
    """Frozen SigLIP 2 vision-tower guide with pooling-query attention."""

    name = "siglip2"
    default_model_id = "google/siglip2-base-patch16-224"

    def _load_model(self):
        try:
            from transformers import AutoImageProcessor, SiglipVisionModel

            self.processor = AutoImageProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                use_fast=False,
            )
            self.model = SiglipVisionModel.from_pretrained(
                self.model_id, **self._pretrained_kwargs()
            ).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "failed to load SigLIP 2 vision guide %r: %s"
                % (self.model_id, exc)
            ) from exc
        config = self.model.config
        self.patch_size = int(getattr(config, "patch_size", 16))
        self.input_size = int(
            self.kwargs.get("input_size", getattr(config, "image_size", 224))
        )

    def _prepare_images(self, images):
        self.validate_images(images)
        images = images.float().cpu()
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )
        processed = self.processor(
            images=list(images),
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
        )
        return processed["pixel_values"].to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode(self, images):
        pixel_values = self._prepare_images(images)
        output = self.model(
            pixel_values=pixel_values,
            output_attentions=False,
            return_dict=True,
        )
        patches = output.last_hidden_state
        grid_size = _grid_from_image(
            torch.empty(
                patches.size(0),
                3,
                self.input_size,
                self.input_size,
                device=patches.device,
            ),
            self.patch_size,
        )
        head = self.model.vision_model.head
        probe = head.probe.repeat(patches.size(0), 1, 1)
        _, attention = head.attention(
            probe,
            patches,
            patches,
            need_weights=True,
            average_attn_weights=True,
        )
        native_map = attention[:, 0].view(patches.size(0), *grid_size)
        return GuideOutput(
            patch_tokens=patches,
            grid_size=grid_size,
            global_token=None,
            native_map=native_map,
            metadata={
                "model_id": self.model_id,
                "readout": "pooling_query_attention",
            },
        )


class CLIPGuide(SemanticGuide):
    """Frozen CLIP vision guide using final-layer CLS attention."""

    name = "clip"
    default_model_id = "openai/clip-vit-base-patch16"

    def _load_model(self):
        try:
            from transformers import (
                AutoImageProcessor,
                CLIPVisionModel,
            )

            self.processor = AutoImageProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                use_fast=False,
            )
            pretrained_kwargs = self._pretrained_kwargs()
            pretrained_kwargs["attn_implementation"] = "eager"
            self.model = CLIPVisionModel.from_pretrained(
                self.model_id, **pretrained_kwargs
            ).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "failed to load CLIP vision guide %r: %s"
                % (self.model_id, exc)
            ) from exc
        config = self.model.config
        self.patch_size = int(getattr(config, "patch_size", 16))
        self.input_size = int(
            self.kwargs.get("input_size", getattr(config, "image_size", 224))
        )

    def _prepare_images(self, images):
        self.validate_images(images)
        images = images.float().cpu()
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )
        processed = self.processor(
            images=list(images),
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
        )
        return processed["pixel_values"].to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode(self, images):
        pixel_values = self._prepare_images(images)
        output = self.model(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True,
        )
        tokens = output.last_hidden_state
        patches = tokens[:, 1:]
        grid_size = _grid_from_image(
            torch.empty(
                patches.size(0),
                3,
                self.input_size,
                self.input_size,
                device=patches.device,
            ),
            self.patch_size,
        )
        if not output.attentions:
            raise RuntimeError(
                "CLIP model did not return attention weights; use a model "
                "implementation with output_attentions support"
            )
        attention = output.attentions[-1].mean(dim=1)[:, 0, 1:]
        native_map = attention.view(patches.size(0), *grid_size)
        return GuideOutput(
            patch_tokens=patches,
            grid_size=grid_size,
            global_token=tokens[:, 0],
            native_map=native_map,
            metadata={
                "model_id": self.model_id,
                "readout": "final_cls_attention",
            },
        )
