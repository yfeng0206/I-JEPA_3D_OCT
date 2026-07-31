import subprocess
import sys
from types import SimpleNamespace
import unittest
import json

import torch

from src.guides.base import GroundingBox, GroundingPoint, GuideOutput
from src.guides.vlm_guides import (
    MolmoPointGuide,
    Qwen3VLGuide,
    build_vlm_prompt,
    molmo_points_from_official,
    parse_qwen_grounding,
    try_parse_qwen_grounding,
)


class _FakeQwenProcessor:
    def __init__(self, responses=None):
        self.image_sizes = []
        self.prompts = []
        self.responses = {
            11: "A dog.",
            12: (
                '[{"bbox_2d": [10, 20, 450, 950], "label": "dog"}, '
                '{"bbox_2d": [500, 100, 900, 800], "label": "ball"}]'
            ),
            13: '{"point_2d": [500, 250], "label": "dog"}',
            14: (
                '[{"point_2d": [250, 500], "label": "dog"}, '
                '{"point_2d": [750, 500], "label": "ball"}]'
            ),
        }
        if responses:
            self.responses.update(responses)

    def apply_chat_template(self, messages, **kwargs):
        image = messages[0]["content"][0]["image"]
        prompt = messages[0]["content"][1]["text"]
        self.image_sizes.append(image.size)
        self.prompts.append(prompt)
        if prompt == "Describe this image.":
            task_id = 1
        elif "bounding boxes" in prompt:
            task_id = 2
        elif "up to 10 distinct" in prompt:
            task_id = 4
        else:
            task_id = 3
        grid_height = image.height // 2
        grid_width = image.width // 2
        return {
            "input_ids": torch.tensor([[task_id]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "pixel_values": torch.zeros(grid_height * grid_width, 3),
            "image_grid_thw": torch.tensor(
                [[1, grid_height, grid_width]]
            ),
        }

    def batch_decode(self, generated, **kwargs):
        return [self.responses[int(generated[0, 0])]]


class _FakeQwenModel:
    def __init__(self):
        vision_config = SimpleNamespace(spatial_merge_size=2, patch_size=2)
        self.config = SimpleNamespace(vision_config=vision_config)
        self.generation_calls = []

    def generate(self, input_ids, **kwargs):
        self.generation_calls.append(dict(kwargs))
        new_id = input_ids[:, -1:] + 10
        return torch.cat([input_ids, new_id], dim=1)

    def get_image_features(self, pixel_values, image_grid_thw):
        del pixel_values
        _, height, width = image_grid_thw[0].tolist()
        count = (height // 2) * (width // 2)
        return (torch.ones(count, 3),), [torch.ones(count, 3)]


class _FakeMolmoProcessor:
    def __init__(self):
        self.image_sizes = []
        self.prompts = []

    def apply_chat_template(
        self, messages, return_pointing_metadata=False, **kwargs
    ):
        prompt = messages[0]["content"][0]["text"]
        image = messages[0]["content"][1]["image"]
        self.image_sizes.append(image.size)
        self.prompts.append(prompt)
        task_id = 2 if prompt.startswith("Point to") else 1
        result = {
            "input_ids": torch.tensor([[task_id]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "token_type_ids": torch.zeros(1, 1, dtype=torch.long),
            "pixel_values": torch.zeros(1, 4, 3),
            "image_token_pooling": torch.tensor(
                [[0, 1], [2, 3], [4, 5], [6, 7]]
            ),
            "image_grids": torch.tensor([[1, 2, 1, 2]]),
            "image_num_crops": torch.tensor([1]),
        }
        if return_pointing_metadata:
            result["metadata"] = {
                "token_pooling": result["image_token_pooling"].numpy(),
                "subpatch_mapping": [[[0, 1], [2, 3]]],
                "image_sizes": [image.size],
            }
        return result

    def post_process_image_text_to_text(self, generated, **kwargs):
        responses = {11: "A dog.", 12: "<POINT_0><POINT_5><POINT_7> 0"}
        return [responses[int(generated[0, 0])]]


class _FakeMolmoModel:
    def __init__(self):
        self.used_native_logit_processor = False
        self.used_native_extractor = False
        self.generation_calls = []
        self.extracted_points = [
            [0, 0, 4.0, 5.0],
            [1, 0, 6.0, 2.0],
        ]
        self.model = _FakeMolmoCore()

    def __call__(self, **kwargs):
        raise AssertionError("feature extraction called the text decoder")

    def generate(self, input_ids, **kwargs):
        self.generation_calls.append(dict(kwargs))
        new_id = input_ids[:, -1:] + 10
        return torch.cat([input_ids, new_id], dim=1)

    def build_logit_processor_from_inputs(self, inputs):
        self.used_native_logit_processor = True
        return ["native-point-processor"]

    def extract_image_points(
        self, output_text, pooling, subpatch_mapping, image_sizes
    ):
        self.used_native_extractor = True
        return self.extracted_points


class _FakeMolmoVision(torch.nn.Module):
    def forward(self, images):
        return [torch.ones(images.size(0), images.size(1), 3)]


class _FakeMolmoConnector(torch.nn.Module):
    def forward(self, values, mask):
        del mask
        return values.mean(dim=1, keepdim=True)


class _FakeMolmoCore:
    def __init__(self):
        self.vit = _FakeMolmoVision()
        self.vit_layers = [0]
        self.connector = _FakeMolmoConnector()

    def merge_visual_inputs(
        self,
        input_ids,
        pixel_values,
        image_token_pooling,
        image_grids,
        image_num_crops,
    ):
        del input_ids, pixel_values, image_grids, image_num_crops
        images = torch.zeros(1, 1, 8, 3)
        return images, image_token_pooling.unsqueeze(0)


def _fake_guide(guide_class, model, processor, **kwargs):
    guide = object.__new__(guide_class)
    guide.model = model
    guide.processor = processor
    guide.device = torch.device("cpu")
    guide.dtype = torch.bfloat16
    guide.model_id = "fake-local-model"
    guide.kwargs = {"revision": "fake-revision", **kwargs}
    guide.transformers_version = "fake"
    return guide


class LazyRegistrationTests(unittest.TestCase):
    def test_package_import_does_not_import_vlm_dependencies(self):
        command = (
            "import sys; from src.guides import available_guides; "
            "assert 'qwen3_vl' in available_guides(); "
            "assert 'molmo' in available_guides(); "
            "assert 'src.guides.vlm_guides' not in sys.modules; "
            "assert 'transformers' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", command], check=True)


class PromptTests(unittest.TestCase):
    def test_prompts_are_mode_specific_and_image_only(self):
        self.assertEqual(
            build_vlm_prompt("qwen3_vl", "caption"),
            "Describe this image.",
        )
        self.assertEqual(
            build_vlm_prompt("qwen3_vl", "single_point"),
            "Identify and locate the single most visually prominent visible "
            "object. Return only one JSON object like "
            '{"point_2d": [x, y], "label": "specific noun phrase"}. '
            "The label must name the object (for example, person, hat, or dog) "
            "and must not repeat the instruction.",
        )
        self.assertEqual(
            build_vlm_prompt("qwen3_vl", "plural_points"),
            "Identify and locate up to 10 distinct, whole, visually prominent "
            "objects. Do not list object parts or duplicate objects. Return "
            "only a JSON list of items like "
            '{"point_2d": [x, y], "label": "specific noun phrase"}. '
            "Each label must name the object and must not be "
            '"visible object" or "visually prominent object".',
        )
        self.assertEqual(
            build_vlm_prompt("qwen3_vl", "boxes"),
            "Identify and locate up to 10 distinct, whole, visually prominent "
            "objects using bounding boxes. Do not list object parts or "
            "duplicate objects. Return only a JSON list of items like "
            '{"bbox_2d": [x1, y1, x2, y2], '
            '"label": "specific noun phrase"}. Each label must name the object '
            'and must not be "visible object" or "visually prominent object".',
        )
        self.assertEqual(
            build_vlm_prompt("molmo", "single_point"),
            "Point to the single most visually prominent object.",
        )
        self.assertEqual(
            build_vlm_prompt("molmo", "plural_points"),
            "Point to all visually prominent objects.",
        )

    def test_prompts_reject_class_conditioning(self):
        for guide in ("qwen3_vl", "molmo"):
            with self.subTest(guide=guide), self.assertRaises(ValueError):
                build_vlm_prompt(guide, "single_point", "golden retriever")

    def test_legacy_point_and_box_task_aliases_remain_supported(self):
        self.assertEqual(
            build_vlm_prompt("qwen3_vl", "point"),
            build_vlm_prompt("qwen3_vl", "single_point"),
        )
        self.assertEqual(
            build_vlm_prompt("qwen3_vl", "box"),
            build_vlm_prompt("qwen3_vl", "boxes"),
        )

    def test_invalid_model_mode_combinations_fail_loudly(self):
        self.assertEqual(
            Qwen3VLGuide._validate_grounding_mode("box"), "boxes"
        )
        with self.assertRaises(ValueError):
            MolmoPointGuide._validate_grounding_mode("boxes")
        with self.assertRaises(ValueError):
            Qwen3VLGuide._validate_grounding_mode("segmentation")


class QwenParsingTests(unittest.TestCase):
    def test_strict_single_dict_and_plural_list_json(self):
        boxes = parse_qwen_grounding(
            (
                '[{"bbox_2d": [10, 20, 400, 950], "label": "dog"}, '
                '{"bbox_2d": [500, 30, 900, 700], "label": "ball"}]'
            ),
            "boxes",
            image_size=(640, 480),
        )
        points = parse_qwen_grounding(
            '```json\n{"point_2d": [500, 250], "label": "dog"}\n```',
            "single_point",
            image_size=(640, 480),
        )
        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(points), 1)
        self.assertTrue(all(isinstance(item, GroundingBox) for item in boxes))
        self.assertIsInstance(points[0], GroundingPoint)
        self.assertEqual(boxes[1].label, "ball")
        self.assertEqual(points[0].coordinate_space, "normalized_1000")

    def test_single_point_rejects_multiple_items(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_qwen_grounding(
                (
                    '[{"point_2d": [10, 20], "label": "dog"}, '
                    '{"point_2d": [30, 40], "label": "ball"}]'
                ),
                "single_point",
            )

    def test_malformed_mixed_or_non_strict_json_is_rejected(self):
        invalid = (
            "",
            "result: {\"point_2d\": [1, 2], \"label\": \"dog\"}",
            "[]",
            (
                '[{"point_2d": [1, 2], "label": "dog"}, '
                '{"bbox_2d": [1, 2, 3, 4], "label": "cat"}]'
            ),
            '[{"point_2d": [1, 2], "label": "dog"}, 7]',
            "{\"point_2d\": [1, 2], \"label\": \"dog\", \"score\": 1}",
            "{\"point_2d\": [NaN, 2], \"label\": \"dog\"}",
            "```json\n{\"point_2d\": [1, 2], \"label\": \"dog\"}",
        )
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_qwen_grounding(text, "point")

    def test_coordinate_validation(self):
        invalid = (
            '{"point_2d": [-1, 10], "label": "dog"}',
            '{"point_2d": [1001, 10], "label": "dog"}',
            '{"point_2d": [true, 10], "label": "dog"}',
            '{"bbox_2d": [100, 100, 50, 200], "label": "dog"}',
            '{"bbox_2d": [0, 0, 1001, 200], "label": "dog"}',
        )
        for text in invalid[:3]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_qwen_grounding(text, "point")
        for text in invalid[3:]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_qwen_grounding(text, "box")

    def test_parse_failure_does_not_fabricate_grounding(self):
        parsed, failure = try_parse_qwen_grounding(
            "not JSON", "box", image_size=(224, 224)
        )
        self.assertIsNone(parsed)
        self.assertEqual(failure["code"], "invalid_box")
        self.assertTrue(failure["reason"])
        self.assertEqual(failure["raw_output"], "not JSON")


class MolmoPointContractTests(unittest.TestCase):
    def test_official_pixel_points_are_preserved_without_boxes(self):
        points, failure = molmo_points_from_official(
            [[7, 0, 100.5, 75.25]], "dog", [(224, 160)]
        )
        self.assertIsNone(failure)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].coordinate_space, "pixels")
        self.assertEqual(points[0].object_id, 7)
        output = GuideOutput(
            patch_tokens=torch.randn(1, 4, 8),
            grid_size=(1, 4),
            grounding_regions=[[]],
            grounding_points=[points],
        )
        self.assertEqual(output.grounding_regions, [[]])

    def test_invalid_or_empty_official_points_are_failures(self):
        cases = (
            [],
            [[0, 0, 225, 10]],
            [[0, 1, 10, 10]],
            [[0, 0, 10]],
        )
        for extracted in cases:
            with self.subTest(extracted=extracted):
                points, failure = molmo_points_from_official(
                    extracted, "dog", [(224, 160)]
                )
                self.assertEqual(points, [])
                self.assertEqual(failure["code"], "invalid_point")


class ExtendedGuideOutputTests(unittest.TestCase):
    def test_batch_metadata_lengths_are_checked(self):
        with self.assertRaises(ValueError):
            GuideOutput(
                patch_tokens=torch.randn(2, 4, 8),
                grid_size=(2, 2),
                generated_text=["only one"],
            )
        with self.assertRaises(ValueError):
            GuideOutput(
                patch_tokens=torch.randn(1, 4, 8),
                grid_size=(2, 2),
                failures=[],
            )

    def test_legacy_constructor_remains_compatible(self):
        output = GuideOutput(
            patch_tokens=torch.randn(1, 4, 8),
            grid_size=(2, 2),
            global_token=torch.randn(1, 8),
        )
        self.assertEqual(output.batch_size, 1)
        self.assertIsNone(output.generated_text)
        self.assertEqual(output.model_metadata, {})

    def test_point_and_box_lists_enforce_contract_types(self):
        with self.assertRaises(TypeError):
            GuideOutput(
                patch_tokens=torch.randn(1, 4, 8),
                grid_size=(2, 2),
                grounding_regions=[[{"bbox_2d": [0, 0, 1, 1]}]],
            )
        with self.assertRaises(TypeError):
            GuideOutput(
                patch_tokens=torch.randn(1, 4, 8),
                grid_size=(2, 2),
                grounding_points=[[{"point_2d": [0, 0]}]],
            )


class DownloadFreeAdapterTests(unittest.TestCase):
    def test_qwen_features_use_full_source_and_dynamic_processor_metadata(self):
        model = _FakeQwenModel()
        processor = _FakeQwenProcessor()
        guide = _fake_guide(Qwen3VLGuide, model, processor, input_size=512)
        model.generate = lambda *args, **kwargs: self.fail(
            "features-only path called generation"
        )
        output = guide.encode_features(torch.zeros(1, 3, 8, 12))
        self.assertIsNone(output.generated_text)
        self.assertTrue(output.metadata["features_only"])
        self.assertEqual(processor.image_sizes, [(12, 8)])
        self.assertEqual(output.original_image_sizes, [(12, 8)])
        self.assertEqual(output.grid_size, (2, 3))
        self.assertEqual(
            output.spatial_metadata[0]["model_input_size"], (12, 8)
        )
        self.assertEqual(
            output.spatial_metadata[0]["processor_resize"],
            "aspect_preserving_dynamic_resolution",
        )

    def test_molmo_features_use_full_source_and_skip_generation(self):
        model = _FakeMolmoModel()
        processor = _FakeMolmoProcessor()
        guide = _fake_guide(MolmoPointGuide, model, processor)
        model.generate = lambda *args, **kwargs: self.fail(
            "features-only path called generation"
        )
        output = guide.encode_features(torch.zeros(1, 3, 8, 12))
        self.assertIsNone(output.generated_text)
        self.assertTrue(output.metadata["features_only"])
        self.assertEqual(processor.image_sizes, [(12, 8)])
        self.assertEqual(output.original_image_sizes, [(12, 8)])
        self.assertEqual(
            output.spatial_metadata[0]["processor_crop_strategy"],
            "official_global_plus_up_to_24_local_crops",
        )

    def test_molmo_spatial_metadata_is_json_serializable(self):
        guide = _fake_guide(
            MolmoPointGuide, _FakeMolmoModel(), _FakeMolmoProcessor()
        )
        output = guide.encode(
            torch.zeros(1, 3, 8, 8)
        )
        json.dumps(output.spatial_metadata)

    def test_qwen_single_point_returns_only_selected_grounding(self):
        model = _FakeQwenModel()
        processor = _FakeQwenProcessor()
        guide = _fake_guide(
            Qwen3VLGuide, model, processor
        )
        output = guide.encode(torch.zeros(1, 3, 8, 8))
        self.assertEqual(output.generated_text, ["A dog."])
        self.assertEqual(output.grounding_regions, [[]])
        self.assertEqual(len(output.grounding_points[0]), 1)
        self.assertEqual(output.metadata["grounding_mode"], "single_point")
        self.assertEqual(set(output.raw_generation[0]), {
            "caption",
            "grounding_mode",
            "grounding",
        })
        self.assertFalse(output.failures[0])
        self.assertEqual(
            output.raw_generation[0]["caption"]["prompt"],
            "Describe this image.",
        )
        self.assertEqual(
            output.raw_generation[0]["grounding"]["decoding"]["strategy"],
            "greedy",
        )
        self.assertEqual(tuple(output.patch_tokens.shape), (1, 4, 3))
        self.assertEqual(output.grid_size, (2, 2))

    def test_qwen_plural_points_preserve_every_valid_item(self):
        guide = _fake_guide(
            Qwen3VLGuide,
            _FakeQwenModel(),
            _FakeQwenProcessor(),
            grounding_mode="plural_points",
        )
        output = guide.encode(torch.zeros(1, 3, 8, 8))
        self.assertEqual(output.grounding_regions, [[]])
        self.assertEqual(
            [point.label for point in output.grounding_points[0]],
            ["dog", "ball"],
        )
        self.assertEqual(
            output.raw_generation[0]["grounding"]["prompt"],
            build_vlm_prompt("qwen3_vl", "plural_points"),
        )

    def test_qwen_boxes_are_a_separate_capability_mode(self):
        guide = _fake_guide(
            Qwen3VLGuide,
            _FakeQwenModel(),
            _FakeQwenProcessor(),
            grounding_mode="boxes",
        )
        output = guide.encode(torch.zeros(1, 3, 8, 8))
        self.assertEqual(len(output.grounding_regions[0]), 2)
        self.assertEqual(output.grounding_points, [[]])
        self.assertNotIn("point", output.raw_generation[0])

    def test_qwen_parse_failure_preserves_raw_without_fake_grounding(self):
        guide = _fake_guide(
            Qwen3VLGuide,
            _FakeQwenModel(),
            _FakeQwenProcessor({13: "not JSON"}),
        )
        output = guide.encode(torch.zeros(1, 3, 8, 8))
        self.assertEqual(output.grounding_regions, [[]])
        self.assertEqual(output.grounding_points, [[]])
        self.assertEqual(output.failures[0][0]["task"], "single_point")
        self.assertEqual(output.failures[0][0]["raw_output"], "not JSON")
        self.assertEqual(
            output.raw_generation[0]["grounding"]["text"], "not JSON"
        )

    def test_molmo_adapter_uses_native_point_apis_and_never_boxes(self):
        model = _FakeMolmoModel()
        processor = _FakeMolmoProcessor()
        guide = _fake_guide(
            MolmoPointGuide, model, processor
        )
        output = guide.encode(
            torch.zeros(1, 3, 8, 8)
        )
        self.assertTrue(model.used_native_logit_processor)
        self.assertTrue(model.used_native_extractor)
        self.assertEqual(output.grounding_regions, [[]])
        self.assertEqual(output.grounding_points[0][0].point_2d, (4.0, 5.0))
        self.assertEqual(len(output.grounding_points[0]), 2)
        self.assertEqual(
            output.grounding_points[0][0].label, "visually prominent object"
        )
        self.assertEqual(tuple(output.global_token.shape), (1, 3))
        self.assertEqual(
            output.metadata["grounding_contract"], "native_points_only"
        )
        self.assertEqual(
            output.raw_generation[0]["grounding"]["prompt"],
            "Point to the single most visually prominent object.",
        )

    def test_molmo_plural_mode_uses_200_tokens_and_keeps_all_points(self):
        model = _FakeMolmoModel()
        guide = _fake_guide(
            MolmoPointGuide,
            model,
            _FakeMolmoProcessor(),
            grounding_mode="plural_points",
            point_max_new_tokens=64,
        )
        output = guide.encode(torch.zeros(1, 3, 8, 8))
        self.assertEqual(len(output.grounding_points[0]), 2)
        self.assertEqual(model.generation_calls[-1]["max_new_tokens"], 200)
        self.assertEqual(
            output.raw_generation[0]["grounding"]["prompt"],
            "Point to all visually prominent objects.",
        )
        self.assertEqual(
            output.raw_generation[0]["grounding"]["decoding"][
                "max_new_tokens"
            ],
            200,
        )

    def test_adapters_require_batch_one_and_reject_label_leakage(self):
        for guide in (
            _fake_guide(
                Qwen3VLGuide, _FakeQwenModel(), _FakeQwenProcessor()
            ),
            _fake_guide(
                MolmoPointGuide, _FakeMolmoModel(), _FakeMolmoProcessor()
            ),
        ):
            with self.subTest(guide=guide.name), self.assertRaises(ValueError):
                guide.encode(torch.zeros(2, 3, 8, 8))
            with self.subTest(guide=guide.name), self.assertRaises(ValueError):
                guide.encode(
                    torch.zeros(1, 3, 8, 8),
                    class_names=["dog"],
                )


if __name__ == "__main__":
    unittest.main()
