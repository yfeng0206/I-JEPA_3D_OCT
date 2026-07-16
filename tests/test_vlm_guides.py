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
    def apply_chat_template(self, messages, **kwargs):
        prompt = messages[0]["content"][1]["text"]
        if "Describe" in prompt:
            task_id = 1
        elif "bbox_2d" in prompt:
            task_id = 2
        else:
            task_id = 3
        return {
            "input_ids": torch.tensor([[task_id]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "pixel_values": torch.zeros(16, 3),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }

    def batch_decode(self, generated, **kwargs):
        responses = {
            11: "A dog.",
            12: '{"bbox_2d": [10, 20, 900, 950], "label": "dog"}',
            13: "not JSON",
        }
        return [responses[int(generated[0, 0])]]


class _FakeQwenModel:
    def __init__(self):
        vision_config = SimpleNamespace(spatial_merge_size=2)
        self.config = SimpleNamespace(vision_config=vision_config)

    def generate(self, input_ids, **kwargs):
        new_id = input_ids[:, -1:] + 10
        return torch.cat([input_ids, new_id], dim=1)

    def get_image_features(self, pixel_values, image_grid_thw):
        return (torch.ones(4, 3),), [torch.ones(4, 3)]


class _FakeMolmoProcessor:
    def apply_chat_template(
        self, messages, return_pointing_metadata=False, **kwargs
    ):
        prompt = messages[0]["content"][0]["text"]
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
                "image_sizes": [(8, 8)],
            }
        return result

    def post_process_image_text_to_text(self, generated, **kwargs):
        responses = {11: "A dog.", 12: "<POINT_0><POINT_5><POINT_7> 0"}
        return [responses[int(generated[0, 0])]]


class _FakeMolmoModel:
    def __init__(self):
        self.used_native_logit_processor = False
        self.used_native_extractor = False
        self.model = _FakeMolmoCore()

    def __call__(self, **kwargs):
        raise AssertionError("feature extraction called the text decoder")

    def generate(self, input_ids, **kwargs):
        new_id = input_ids[:, -1:] + 10
        return torch.cat([input_ids, new_id], dim=1)

    def build_logit_processor_from_inputs(self, inputs):
        self.used_native_logit_processor = True
        return ["native-point-processor"]

    def extract_image_points(
        self, output_text, pooling, subpatch_mapping, image_sizes
    ):
        self.used_native_extractor = True
        return [[0, 0, 4.0, 5.0]]


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


def _fake_guide(guide_class, model, processor):
    guide = object.__new__(guide_class)
    guide.model = model
    guide.processor = processor
    guide.device = torch.device("cpu")
    guide.dtype = torch.bfloat16
    guide.model_id = "fake-local-model"
    guide.kwargs = {"revision": "fake-revision"}
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
    def test_locked_prompts_are_deterministic(self):
        self.assertEqual(
            build_vlm_prompt("qwen3_vl", "caption"),
            "Describe this image in one factual sentence of at most 20 words.",
        )
        self.assertEqual(
            build_vlm_prompt("qwen3_vl", "box"),
            "Return one JSON item with bbox_2d and label for the most "
            "prominent named class instance.",
        )
        self.assertEqual(
            build_vlm_prompt("qwen3_vl", "point"),
            "Return one JSON item with point_2d and label for the most "
            "prominent named class instance.",
        )
        self.assertEqual(
            build_vlm_prompt("molmo", "point"),
            "Point to the most prominent object in the image.",
        )

    def test_molmo_prompt_rejects_class_conditioning(self):
        with self.assertRaises(ValueError):
            build_vlm_prompt("molmo", "point", "golden retriever")


class QwenParsingTests(unittest.TestCase):
    def test_strict_box_and_point_json(self):
        box = parse_qwen_grounding(
            '[{"bbox_2d": [10, 20, 900, 950], "label": "dog"}]',
            "box",
            image_size=(640, 480),
        )
        point = parse_qwen_grounding(
            '```json\n{"point_2d": [500, 250], "label": "dog"}\n```',
            "point",
            image_size=(640, 480),
        )
        self.assertIsInstance(box, GroundingBox)
        self.assertIsInstance(point, GroundingPoint)
        self.assertEqual(box.coordinate_space, "normalized_1000")
        self.assertEqual(point.coordinate_space, "normalized_1000")

    def test_malformed_or_non_strict_json_is_rejected(self):
        invalid = (
            "",
            "result: {\"point_2d\": [1, 2], \"label\": \"dog\"}",
            "[{\"point_2d\": [1, 2], \"label\": \"dog\"}, "
            "{\"point_2d\": [3, 4], \"label\": \"cat\"}]",
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
    def test_qwen_features_only_skips_generation(self):
        model = _FakeQwenModel()
        guide = _fake_guide(Qwen3VLGuide, model, _FakeQwenProcessor())
        model.generate = lambda *args, **kwargs: self.fail(
            "features-only path called generation"
        )
        output = guide.encode_features(torch.zeros(1, 3, 8, 8))
        self.assertIsNone(output.generated_text)
        self.assertTrue(output.metadata["features_only"])

    def test_molmo_features_only_skips_generation(self):
        model = _FakeMolmoModel()
        guide = _fake_guide(MolmoPointGuide, model, _FakeMolmoProcessor())
        model.generate = lambda *args, **kwargs: self.fail(
            "features-only path called generation"
        )
        output = guide.encode_features(torch.zeros(1, 3, 8, 8))
        self.assertIsNone(output.generated_text)
        self.assertTrue(output.metadata["features_only"])

    def test_molmo_spatial_metadata_is_json_serializable(self):
        guide = _fake_guide(
            MolmoPointGuide, _FakeMolmoModel(), _FakeMolmoProcessor()
        )
        output = guide.encode(
            torch.zeros(1, 3, 8, 8)
        )
        json.dumps(output.spatial_metadata)

    def test_qwen_adapter_preserves_parse_failure_without_fake_point(self):
        guide = _fake_guide(
            Qwen3VLGuide, _FakeQwenModel(), _FakeQwenProcessor()
        )
        output = guide.encode(torch.zeros(1, 3, 8, 8))
        self.assertEqual(output.generated_text, ["A dog."])
        self.assertEqual(len(output.grounding_regions[0]), 1)
        self.assertEqual(output.grounding_points, [[]])
        self.assertEqual(
            output.spatial_metadata[0]["model_input_size"], (512, 512)
        )
        self.assertEqual(output.failures[0][0]["task"], "point")
        self.assertEqual(tuple(output.patch_tokens.shape), (1, 4, 3))
        self.assertEqual(output.grid_size, (2, 2))

    def test_molmo_adapter_uses_native_point_apis_and_never_boxes(self):
        model = _FakeMolmoModel()
        guide = _fake_guide(
            MolmoPointGuide, model, _FakeMolmoProcessor()
        )
        output = guide.encode(
            torch.zeros(1, 3, 8, 8)
        )
        self.assertTrue(model.used_native_logit_processor)
        self.assertTrue(model.used_native_extractor)
        self.assertEqual(output.grounding_regions, [[]])
        self.assertEqual(output.grounding_points[0][0].point_2d, (4.0, 5.0))
        self.assertEqual(
            output.grounding_points[0][0].label, "prominent object"
        )
        self.assertEqual(tuple(output.global_token.shape), (1, 3))
        self.assertEqual(
            output.metadata["grounding_contract"], "native_points_only"
        )


if __name__ == "__main__":
    unittest.main()
