import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from scripts.semantic_map_atlas import (
    commit_output_dir,
    guide_input_tensors,
    guide_input_transform_profile,
    image_id,
    load_images,
    load_input_manifest,
    prepare_output_dir,
    safe_model_id,
    save_guide_npz,
    sanitize_error,
    sanitize_metadata,
    select_primary_map,
    validate_atlas_batch_size,
)


class AtlasUtilityTests(unittest.TestCase):
    def test_content_hash_prevents_same_stem_collision(self):
        with tempfile.TemporaryDirectory() as root:
            left = os.path.join(root, "left", "image.png")
            right = os.path.join(root, "right", "image.png")
            os.makedirs(os.path.dirname(left))
            os.makedirs(os.path.dirname(right))
            with open(left, "wb") as handle:
                handle.write(b"left")
            with open(right, "wb") as handle:
                handle.write(b"right")
            self.assertNotEqual(image_id(left), image_id(right))

    def test_existing_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "run")
            os.makedirs(output)
            with self.assertRaises(FileExistsError):
                prepare_output_dir(output, overwrite=False)

    def test_failed_replacement_can_restore_old_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "run")
            run_dir = os.path.join(root, "new")
            os.makedirs(output)
            os.makedirs(run_dir)
            with open(os.path.join(output, "old.txt"), "w") as handle:
                handle.write("old")
            with open(os.path.join(run_dir, "new.txt"), "w") as handle:
                handle.write("new")
            commit_output_dir(run_dir, output, overwrite=True)
            self.assertFalse(os.path.exists(os.path.join(output, "old.txt")))
            self.assertTrue(os.path.exists(os.path.join(output, "new.txt")))

    def test_error_sanitization_hides_input_path(self):
        with tempfile.TemporaryDirectory() as root:
            input_path = os.path.join(root, "subject_001.png")
            text = sanitize_error(
                RuntimeError("failed to read %s" % input_path),
                [input_path],
            )
            self.assertNotIn(input_path, text)
            self.assertIn("<input>", text)

    def test_local_model_id_is_anonymized(self):
        with tempfile.TemporaryDirectory() as root:
            model_path = os.path.join(root, "model")
            os.makedirs(model_path)
            identifier = safe_model_id(model_path)
            self.assertTrue(identifier.startswith("local_model_"))
            self.assertNotIn(root, identifier)

    def test_manifest_checks_image_content(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "image.png")
            with open(image_path, "wb") as handle:
                handle.write(b"content")
            expected = image_id(image_path).removeprefix("img_")
            import hashlib

            full_hash = hashlib.sha256(b"content").hexdigest()
            manifest_path = os.path.join(root, "inputs.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset_id": "tiny",
                        "images": [
                            {
                                "path": image_path,
                                "image_sha256": full_hash,
                                "class_name": "object",
                            }
                        ],
                    },
                    handle,
                )
            entries, metadata = load_input_manifest(manifest_path, 1)
            self.assertEqual(entries[0]["image_id"], "img_" + expected)
            self.assertEqual(metadata["dataset_id"], "tiny")

            with open(image_path, "wb") as handle:
                handle.write(b"changed")
            with self.assertRaises(RuntimeError):
                load_input_manifest(manifest_path, 1)

    def test_metadata_sanitization_converts_tensors(self):
        value = sanitize_metadata({"tensor": torch.tensor([1, 2])})
        self.assertEqual(value, {"tensor": [1, 2]})

    def test_full_source_is_preserved_for_display_and_vlm_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "wide.png")
            Image.new("RGB", (12, 4), color="red").save(image_path)
            originals, tensors = load_images(
                [{"path": image_path}], crop_size=224
            )
            self.assertEqual(originals[0].size, (12, 4))
            self.assertEqual(tuple(tensors[0].shape), (3, 4, 12))
            for guide_name in ("qwen3_vl", "molmo"):
                prepared = guide_input_tensors(
                    tensors, guide_name, crop_size=224
                )
                self.assertEqual(tuple(prepared[0].shape), (3, 4, 12))
                self.assertTrue(
                    guide_input_transform_profile(guide_name)[
                        "preserves_source_aspect_ratio"
                    ]
                )

    def test_patch_guides_keep_deterministic_256_center_crop_224(self):
        tensor = torch.zeros(3, 100, 300)
        prepared = guide_input_tensors([tensor], "dinov3", crop_size=224)
        self.assertEqual(tuple(prepared[0].shape), (3, 224, 224))
        profile = guide_input_transform_profile("ijepa", crop_size=224)
        self.assertEqual(profile["resize_short_side"], 256)
        self.assertEqual(profile["center_crop"], 224)
        self.assertEqual(validate_atlas_batch_size("dinov3", 4), 4)
        for guide_name in ("qwen3_vl", "molmo"):
            with self.subTest(guide=guide_name), self.assertRaises(ValueError):
                validate_atlas_batch_size(guide_name, 2)

    def test_grounding_raster_is_explicitly_derived_in_maps_and_npz(self):
        raster = torch.ones(2, 3)
        name, selected = select_primary_map({}, raster)
        self.assertEqual(name, "derived_grounding_raster")
        self.assertIs(selected, raster)
        with tempfile.TemporaryDirectory() as root:
            save_guide_npz(
                root,
                "image",
                "qwen3_vl",
                {},
                None,
                raster,
                None,
                None,
            )
            path = os.path.join(root, "maps", "image__qwen3_vl.npz")
            with np.load(path) as payload:
                self.assertIn("derived_grounding_raster", payload.files)
                self.assertNotIn("grounded_native_raster", payload.files)


if __name__ == "__main__":
    unittest.main()
