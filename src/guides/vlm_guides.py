"""Frozen Qwen3-VL and MolmoPoint adapters for the Phase-0 sidecar."""

from contextlib import nullcontext
import json
import math
import re
import time

import torch

from .base import (
    GroundingBox,
    GroundingPoint,
    GuideOutput,
    SemanticGuide,
)


CAPTION_PROMPT = (
    "Describe this image in one factual sentence of at most 20 words."
)
QWEN_BOX_PROMPT = (
    "Return one JSON item with bbox_2d and label for the most prominent "
    "named class instance."
)
QWEN_POINT_PROMPT = (
    "Return one JSON item with point_2d and label for the most prominent "
    "named class instance."
)
MOLMO_POINT_PROMPT = "Point to the most prominent object in the image."

_JSON_FENCE = re.compile(
    r"\A```(?:json)?\s*(.*?)\s*```\Z", flags=re.IGNORECASE | re.DOTALL
)


def build_vlm_prompt(guide, task, class_name=None):
    """Build exactly the deterministic prompts fixed by the evidence lock."""

    guide = str(guide).lower()
    task = str(task).lower()
    if task == "caption":
        return CAPTION_PROMPT
    if guide == "qwen3_vl":
        if task == "box":
            return QWEN_BOX_PROMPT
        if task == "point":
            return QWEN_POINT_PROMPT
    if guide in ("molmo", "molmopoint") and task == "point":
        if class_name is not None:
            raise ValueError(
                "Phase-0 MolmoPoint grounding is image-only and does not "
                "accept class-conditioned prompts"
            )
        return MOLMO_POINT_PROMPT
    raise ValueError("unsupported prompt request %r/%r" % (guide, task))


def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant %r is not allowed" % value)


def _load_one_json_item(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("grounding output is empty")
    payload = text.strip()
    if payload.startswith("```"):
        match = _JSON_FENCE.fullmatch(payload)
        if match is None:
            raise ValueError("malformed JSON code fence")
        payload = match.group(1).strip()
    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
    except (TypeError, ValueError) as exc:
        raise ValueError("grounding output is not one strict JSON item") from exc
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("grounding JSON list must contain exactly one item")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("grounding JSON item must be an object")
    return value


def _strict_coordinates(value, count):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("grounding coordinates must be a %d-item list" % count)
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise ValueError("grounding coordinates must be numeric")
    coordinates = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coordinates):
        raise ValueError("grounding coordinates must be finite")
    return coordinates


def parse_qwen_grounding(text, grounding_type, image_size=None):
    """Strictly parse one Qwen normalized-1000 box or point JSON item."""

    grounding_type = str(grounding_type).lower()
    item = _load_one_json_item(text)
    if grounding_type == "box":
        if set(item) != {"bbox_2d", "label"}:
            raise ValueError("box JSON must contain only bbox_2d and label")
        coordinates = _strict_coordinates(item["bbox_2d"], 4)
        return GroundingBox(
            label=item["label"],
            bbox_2d=coordinates,
            coordinate_space="normalized_1000",
            image_size=image_size,
        )
    if grounding_type == "point":
        if set(item) != {"point_2d", "label"}:
            raise ValueError("point JSON must contain only point_2d and label")
        coordinates = _strict_coordinates(item["point_2d"], 2)
        return GroundingPoint(
            label=item["label"],
            point_2d=coordinates,
            coordinate_space="normalized_1000",
            image_size=image_size,
        )
    raise ValueError("grounding_type must be 'box' or 'point'")


def try_parse_qwen_grounding(text, grounding_type, image_size=None):
    """Return a parsed value or explicit failure metadata, never a fake region."""

    try:
        return (
            parse_qwen_grounding(text, grounding_type, image_size=image_size),
            None,
        )
    except (TypeError, ValueError) as exc:
        return None, {
            "code": "invalid_%s" % grounding_type,
            "reason": str(exc),
        }


def molmo_points_from_official(
    extracted_points, label, image_sizes
):
    """Validate the pixel-space rows returned by ``extract_image_points``."""

    try:
        points = []
        for row in extracted_points:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                raise ValueError(
                    "MolmoPoint extraction must return four-value rows"
                )
            object_id, image_index, x, y = row
            if (
                isinstance(object_id, bool)
                or isinstance(image_index, bool)
                or int(object_id) != object_id
                or int(image_index) != image_index
            ):
                raise ValueError("MolmoPoint identifiers must be integers")
            image_index = int(image_index)
            if image_index < 0 or image_index >= len(image_sizes):
                raise ValueError("MolmoPoint image index is out of bounds")
            points.append(
                GroundingPoint(
                    label=label,
                    point_2d=(x, y),
                    coordinate_space="pixels",
                    image_index=image_index,
                    image_size=tuple(image_sizes[image_index]),
                    object_id=int(object_id),
                )
            )
        if not points:
            raise ValueError("MolmoPoint returned no native points")
        return points, None
    except (TypeError, ValueError, OverflowError) as exc:
        return [], {"code": "invalid_point", "reason": str(exc)}


def _image_to_pil(image):
    from PIL import Image

    array = (
        image.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _resize_pil_square(image, input_size):
    from PIL import Image

    input_size = int(input_size)
    if input_size <= 0:
        raise ValueError("VLM input_size must be positive")
    if image.size == (input_size, input_size):
        return image
    return image.resize(
        (input_size, input_size), resample=Image.Resampling.BICUBIC
    )


def _tensor_inputs_to_device(inputs, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _plain_metadata(value):
    """Convert processor metadata to JSON-safe Python values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {
            str(key): _plain_metadata(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain_metadata(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _generation_sequences(output):
    return output.sequences if hasattr(output, "sequences") else output


class _FrozenVLMGuide(SemanticGuide):
    expected_revision = ""

    def __init__(
        self,
        model_id=None,
        device="auto",
        dtype="bfloat16",
        cache_dir=None,
        local_files_only=True,
        **kwargs
    ):
        kwargs.setdefault("revision", self.expected_revision)
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **kwargs
        )

    def _require_bfloat16(self):
        if self.dtype != torch.bfloat16:
            raise ValueError("%s Phase-0 inference requires bfloat16" % self.name)

    def _freeze_model(self):
        if hasattr(self.model, "requires_grad_"):
            self.model.requires_grad_(False)
        self.model.eval()

    def _autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _start_telemetry(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return time.perf_counter()

    def _finish_telemetry(self, started):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            memory = {
                "max_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(self.device)
                ),
                "max_cuda_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(self.device)
                ),
            }
        else:
            memory = {
                "max_cuda_allocated_bytes": 0,
                "max_cuda_reserved_bytes": 0,
            }
        return time.perf_counter() - started, memory

    def _metadata(self):
        return {
            "guide": self.name,
            "official_model_id": self.official_model_id,
            "model_id": self.model_id,
            "revision": self.kwargs.get("revision"),
            "transformers_version": self.transformers_version,
            "model_class": self.model.__class__.__name__,
            "processor_class": self.processor.__class__.__name__,
            "dtype": "bfloat16",
            "device": str(self.device),
            "batch_size": 1,
            "frozen": True,
        }

    def _validate_batch_one(self, images):
        self.validate_images(images)
        if images.size(0) != 1:
            raise ValueError("%s Phase-0 adapter requires batch size 1" % self.name)

    @staticmethod
    def _task_failure(task, failure):
        failure = dict(failure)
        failure["task"] = task
        return failure


class Qwen3VLGuide(_FrozenVLMGuide):
    """Frozen Qwen3-VL caption, JSON grounding, and vision-feature guide."""

    name = "qwen3_vl"
    official_model_id = "Qwen/Qwen3-VL-8B-Instruct"
    default_model_id = r"D:\jepa_phase0\models\qwen3_vl"
    expected_revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

    def _load_model(self):
        self._require_bfloat16()
        try:
            import transformers
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                revision=self.kwargs.get("revision"),
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, **self._pretrained_kwargs()
            ).to(self.device)
            self.transformers_version = transformers.__version__
            self._freeze_model()
        except Exception as exc:
            raise RuntimeError(
                "failed to load frozen Qwen3-VL guide %r at revision %s: %s"
                % (self.model_id, self.kwargs.get("revision"), exc)
            ) from exc

    def _prepare(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return _tensor_inputs_to_device(inputs, self.device)

    def _generate_task(self, image, task, max_new_tokens):
        prompt = build_vlm_prompt(self.name, task)
        inputs = self._prepare(image, prompt)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        output = self.model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        sequences = _generation_sequences(output)
        generated = sequences[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        raw = {
            "prompt": prompt,
            "text": text,
            "token_ids": generated[0].detach().cpu().tolist(),
            "latency_seconds": elapsed,
        }
        return text, raw, inputs

    def _feature_output(self, images):
        self._validate_batch_one(images)
        original_size = (int(images.size(-1)), int(images.size(-2)))
        input_size = int(self.kwargs.get("input_size", 512))
        image = _resize_pil_square(_image_to_pil(images[0]), input_size)
        inputs = self._prepare(image, CAPTION_PROMPT)
        image_embeds, deepstack_embeds = self.model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        patches = image_embeds[0].detach().unsqueeze(0)
        grid_thw = inputs["image_grid_thw"][0].detach().cpu()
        temporal, grid_height, grid_width = [
            int(value) for value in grid_thw.tolist()
        ]
        merge_size = int(self.model.config.vision_config.spatial_merge_size)
        if temporal != 1:
            raise ValueError("Qwen3-VL image preprocessing returned a video grid")
        grid_size = (
            grid_height // merge_size,
            grid_width // merge_size,
        )
        if patches.size(1) != grid_size[0] * grid_size[1]:
            raise ValueError(
                "Qwen3-VL post-merger token count does not match image grid"
            )
        return (
            image,
            inputs,
            patches,
            grid_thw,
            grid_size,
            merge_size,
            len(deepstack_embeds),
            original_size,
        )

    @torch.inference_mode()
    def encode_features(self, images):
        started = self._start_telemetry()
        with self._autocast():
            (
                _,
                _,
                patches,
                grid_thw,
                grid_size,
                merge_size,
                deepstack_count,
                original_size,
            ) = self._feature_output(images)
        elapsed, memory = self._finish_telemetry(started)
        return GuideOutput(
            patch_tokens=patches,
            grid_size=grid_size,
            global_token=patches.float().mean(dim=1),
            metadata={
                "readout": "mean_final_post_merger_image_tokens",
                "deepstack_feature_count": deepstack_count,
                "features_only": True,
                "spatial_token_grid": True,
            },
            original_image_sizes=[original_size],
            model_metadata=self._metadata(),
            latency_seconds=[elapsed],
            memory_telemetry=[memory],
            spatial_metadata=[
                {
                    "image_grid_thw": tuple(int(value) for value in grid_thw),
                    "post_merger_grid": grid_size,
                    "spatial_merge_size": merge_size,
                    "model_input_size": (
                        int(self.kwargs.get("input_size", 512)),
                        int(self.kwargs.get("input_size", 512)),
                    ),
                    "original_image_size": original_size,
                }
            ],
        )

    @torch.inference_mode()
    def encode(self, images):
        failures = []
        started = self._start_telemetry()
        with self._autocast():
            (
                image,
                caption_inputs,
                patches,
                grid_thw,
                grid_size,
                merge_size,
                deepstack_count,
                image_size,
            ) = self._feature_output(images)
            caption, caption_raw, _ = self._generate_task(
                image,
                "caption",
                int(self.kwargs.get("caption_max_new_tokens", 48)),
            )

            box_text, box_raw, _ = self._generate_task(
                image,
                "box",
                int(self.kwargs.get("grounding_max_new_tokens", 96)),
            )
            point_text, point_raw, _ = self._generate_task(
                image,
                "point",
                int(self.kwargs.get("grounding_max_new_tokens", 96)),
            )

        box, box_failure = try_parse_qwen_grounding(
            box_text, "box", image_size=image_size
        )
        point, point_failure = try_parse_qwen_grounding(
            point_text, "point", image_size=image_size
        )
        if not caption:
            failures.append(
                {"task": "caption", "code": "empty_caption", "reason": "empty text"}
            )
        if box_failure is not None:
            failures.append(self._task_failure("box", box_failure))
        if point_failure is not None:
            failures.append(self._task_failure("point", point_failure))
        elapsed, memory = self._finish_telemetry(started)
        global_token = patches.float().mean(dim=1)
        return GuideOutput(
            patch_tokens=patches,
            grid_size=grid_size,
            global_token=global_token,
            metadata={
                "readout": "mean_final_post_merger_image_tokens",
                "grounding_coordinate_space": "normalized_1000",
                "deepstack_feature_count": deepstack_count,
                "spatial_token_grid": True,
            },
            generated_text=[caption],
            grounding_regions=[[box] if box is not None else []],
            grounding_points=[[point] if point is not None else []],
            raw_generation=[
                {
                    "caption": caption_raw,
                    "box": box_raw,
                    "point": point_raw,
                }
            ],
            original_image_sizes=[image_size],
            model_metadata=self._metadata(),
            latency_seconds=[elapsed],
            memory_telemetry=[memory],
            spatial_metadata=[
                {
                    "image_grid_thw": tuple(int(value) for value in grid_thw),
                    "post_merger_grid": grid_size,
                    "spatial_merge_size": merge_size,
                    "model_input_size": (
                        int(self.kwargs.get("input_size", 512)),
                        int(self.kwargs.get("input_size", 512)),
                    ),
                    "original_image_size": image_size,
                }
            ],
            failures=[failures],
        )


class MolmoPointGuide(_FrozenVLMGuide):
    """Frozen MolmoPoint caption, native-point, and vision-feature guide."""

    name = "molmo"
    official_model_id = "allenai/MolmoPoint-8B"
    default_model_id = r"D:\jepa_phase0\models\molmo"
    expected_revision = "188130f961c8e0888a34e11121a1423c461a01ba"

    def _load_model(self):
        self._require_bfloat16()
        try:
            import transformers
            from transformers import AutoModelForImageTextToText, AutoProcessor

            common = {
                "cache_dir": self.cache_dir,
                "local_files_only": self.local_files_only,
                "revision": self.kwargs.get("revision"),
                "trust_remote_code": True,
            }
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, **common
            )
            model_kwargs = dict(common)
            model_kwargs["dtype"] = self.dtype
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, **model_kwargs
            ).to(self.device)
            self.transformers_version = transformers.__version__
            self._freeze_model()
        except Exception as exc:
            raise RuntimeError(
                "failed to load reviewed MolmoPoint guide %r at revision %s: %s"
                % (self.model_id, self.kwargs.get("revision"), exc)
            ) from exc

    def _prepare(self, image, prompt, pointing):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            return_pointing_metadata=pointing,
        )
        metadata = inputs.pop("metadata", None)
        return _tensor_inputs_to_device(inputs, self.device), metadata

    def _generate(
        self, inputs, max_new_tokens, logits_processor=None, keep_special=False
    ):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        kwargs = {}
        if logits_processor is not None:
            kwargs["logits_processor"] = logits_processor
        output = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **kwargs
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        sequences = _generation_sequences(output)
        generated = sequences[:, inputs["input_ids"].shape[1] :]
        text = self.processor.post_process_image_text_to_text(
            generated,
            skip_special_tokens=not keep_special,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return text, generated, elapsed

    @staticmethod
    def _feature_tokens(image_hidden_states):
        if not isinstance(image_hidden_states, torch.Tensor):
            raise RuntimeError("MolmoPoint did not return image_hidden_states")
        tokens = image_hidden_states.detach()
        if tokens.dim() == 3 and tokens.size(1) == 1:
            tokens = tokens[:, 0, :]
        elif tokens.dim() == 3 and tokens.size(0) == 1:
            tokens = tokens[0]
        if tokens.dim() != 2:
            raise ValueError(
                "unexpected MolmoPoint image feature shape %s"
                % (tuple(tokens.shape),)
            )
        return tokens.unsqueeze(0)

    def _vision_only_features(self, inputs):
        """Run the pinned official ViT and connector without the text decoder."""
        core = getattr(self.model, "model", None)
        required = ("merge_visual_inputs", "vit", "vit_layers", "connector")
        if core is None or any(not hasattr(core, name) for name in required):
            raise RuntimeError(
                "pinned MolmoPoint model does not expose the reviewed "
                "vision-only modules"
            )
        images, token_pooling = core.merge_visual_inputs(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_token_pooling=inputs["image_token_pooling"],
            image_grids=inputs["image_grids"],
            image_num_crops=inputs["image_num_crops"],
        )
        if images is None or token_pooling is None:
            raise RuntimeError("MolmoPoint produced no merged visual inputs")

        images = images.to(device=self.device, dtype=self.dtype)
        batch_size, crop_count, patch_count, pixel_dim = images.shape
        vit_outputs = core.vit(
            images.reshape(
                batch_size * crop_count, patch_count, pixel_dim
            )
        )
        selected_layers = [vit_outputs[layer] for layer in core.vit_layers]
        if not selected_layers:
            raise RuntimeError("MolmoPoint configured no visual feature layers")
        vit_features = torch.cat(selected_layers, dim=-1).to(self.device)
        feature_dim = vit_features.size(-1)
        token_pooling = token_pooling.to(self.device)
        batch_indices = torch.arange(
            batch_size, device=self.device
        )[:, None, None]
        pooled_inputs = vit_features.reshape(
            batch_size, -1, feature_dim
        )[
            batch_indices,
            token_pooling.clamp(min=0),
        ]
        pooling_mask = token_pooling >= 0
        pooled_inputs = pooled_inputs * pooling_mask.float()[..., None]
        valid_tokens = pooling_mask.any(dim=-1)
        flat_inputs = pooled_inputs.reshape(
            -1, token_pooling.size(-1), feature_dim
        )[valid_tokens.reshape(-1)]
        flat_mask = pooling_mask.reshape(
            -1, token_pooling.size(-1)
        )[valid_tokens.reshape(-1)]
        if flat_inputs.size(0) == 0:
            raise RuntimeError("MolmoPoint produced no valid pooled tokens")
        return core.connector(flat_inputs, flat_mask).to(self.device)

    def _feature_output(self, image):
        caption_inputs, _ = self._prepare(
            image, CAPTION_PROMPT, pointing=False
        )
        patches = self._feature_tokens(
            self._vision_only_features(caption_inputs)
        )
        return caption_inputs, patches

    @torch.inference_mode()
    def encode_features(self, images):
        self._validate_batch_one(images)
        image_size = (int(images.size(-1)), int(images.size(-2)))
        image = _image_to_pil(images[0])
        started = self._start_telemetry()
        with self._autocast():
            caption_inputs, patches = self._feature_output(image)
        token_pooling = caption_inputs["image_token_pooling"].detach().cpu()
        valid_count = int((token_pooling >= 0).any(dim=-1).sum().item())
        if patches.size(1) != valid_count:
            raise ValueError(
                "MolmoPoint post-projector feature count %d does not match "
                "%d valid pooled tokens" % (patches.size(1), valid_count)
            )
        elapsed, memory = self._finish_telemetry(started)
        image_grid = caption_inputs["image_grids"][0].detach().cpu().tolist()
        image_num_crops = int(
            caption_inputs["image_num_crops"][0].detach().cpu().item()
        )
        return GuideOutput(
            patch_tokens=patches,
            grid_size=(1, patches.size(1)),
            global_token=patches.float().mean(dim=1),
            metadata={
                "readout": "mean_valid_post_pooling_post_projector_tokens",
                "token_layout": "official_pooled_token_sequence",
                "feature_extraction_path": "official_vit_plus_connector",
                "features_only": True,
                "spatial_token_grid": False,
            },
            original_image_sizes=[image_size],
            model_metadata=self._metadata(),
            latency_seconds=[elapsed],
            memory_telemetry=[memory],
            spatial_metadata=[
                {
                    "image_grids": image_grid,
                    "image_num_crops": image_num_crops,
                    "valid_pooled_tokens": valid_count,
                    "original_image_size": image_size,
                }
            ],
        )

    @torch.inference_mode()
    def encode(self, images, class_names=None):
        self._validate_batch_one(images)
        if class_names is not None or self.kwargs.get("class_name") is not None:
            raise ValueError(
                "Phase-0 MolmoPoint grounding must not receive class labels"
            )
        grounding_label = "prominent object"
        image_size = (int(images.size(-1)), int(images.size(-2)))
        image = _image_to_pil(images[0])
        failures = []
        started = self._start_telemetry()
        caption_prompt = build_vlm_prompt(self.name, "caption")
        point_prompt = build_vlm_prompt(
            self.name, "point"
        )
        with self._autocast():
            caption_inputs, patches = self._feature_output(image)
            caption, caption_ids, caption_latency = self._generate(
                caption_inputs,
                int(self.kwargs.get("caption_max_new_tokens", 48)),
            )

            point_inputs, pointing_metadata = self._prepare(
                image, point_prompt, pointing=True
            )
            if pointing_metadata is None:
                raise RuntimeError(
                    "MolmoPoint processor omitted return_pointing_metadata"
                )
            # MolmoPoint constrains generated patch/location tokens with this
            # model-owned processor; decoding those tokens as ordinary text loses
            # the native pointing contract.
            logit_processor = self.model.build_logit_processor_from_inputs(
                point_inputs
            )
            raw_point_text, point_ids, point_latency = self._generate(
                point_inputs,
                int(self.kwargs.get("point_max_new_tokens", 64)),
                logits_processor=logit_processor,
                keep_special=True,
            )
            try:
                extracted = self.model.extract_image_points(
                    raw_point_text,
                    pointing_metadata["token_pooling"],
                    pointing_metadata["subpatch_mapping"],
                    pointing_metadata["image_sizes"],
                )
                extraction_failure = None
            except (IndexError, TypeError, ValueError) as exc:
                extracted = []
                extraction_failure = {
                    "code": "invalid_point",
                    "reason": "official point extraction failed: %s" % exc,
                }

        if extraction_failure is None:
            points, point_failure = molmo_points_from_official(
                extracted,
                grounding_label,
                pointing_metadata["image_sizes"],
            )
        else:
            points, point_failure = [], extraction_failure
        if not caption:
            failures.append(
                {"task": "caption", "code": "empty_caption", "reason": "empty text"}
            )
        if point_failure is not None:
            failures.append(self._task_failure("point", point_failure))
        token_pooling = point_inputs["image_token_pooling"].detach().cpu()
        valid_count = int((token_pooling >= 0).any(dim=-1).sum().item())
        if patches.size(1) != valid_count:
            raise ValueError(
                "MolmoPoint post-projector feature count %d does not match "
                "%d valid pooled tokens" % (patches.size(1), valid_count)
            )
        elapsed, memory = self._finish_telemetry(started)
        global_token = patches.float().mean(dim=1)
        image_grid = point_inputs["image_grids"][0].detach().cpu().tolist()
        image_num_crops = int(
            point_inputs["image_num_crops"][0].detach().cpu().item()
        )
        return GuideOutput(
            patch_tokens=patches,
            grid_size=(1, patches.size(1)),
            global_token=global_token,
            metadata={
                "readout": "mean_valid_post_pooling_post_projector_tokens",
                "grounding_contract": "native_points_only",
                "token_layout": "official_pooled_token_sequence",
                "feature_extraction_path": "official_vit_plus_connector",
                "spatial_token_grid": False,
            },
            generated_text=[caption],
            grounding_regions=[[]],
            grounding_points=[points],
            raw_generation=[
                {
                    "caption": {
                        "prompt": caption_prompt,
                        "text": caption,
                        "token_ids": caption_ids[0].detach().cpu().tolist(),
                        "latency_seconds": caption_latency,
                    },
                    "point": {
                        "prompt": point_prompt,
                        "text": raw_point_text,
                        "token_ids": point_ids[0].detach().cpu().tolist(),
                        "latency_seconds": point_latency,
                    },
                }
            ],
            original_image_sizes=[image_size],
            model_metadata=self._metadata(),
            latency_seconds=[elapsed],
            memory_telemetry=[memory],
            spatial_metadata=[
                {
                    "image_grid": image_grid,
                    "image_num_crops": image_num_crops,
                    "image_token_pooling": token_pooling.tolist(),
                    "subpatch_mapping": _plain_metadata(
                        pointing_metadata["subpatch_mapping"]
                    ),
                    "processor_image_sizes": _plain_metadata(
                        pointing_metadata["image_sizes"]
                    ),
                    "valid_image_token_count": valid_count,
                }
            ],
            failures=[failures],
        )
