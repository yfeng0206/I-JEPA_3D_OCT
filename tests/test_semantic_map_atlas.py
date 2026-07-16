import json
import os
import tempfile
import unittest

from scripts.semantic_map_atlas import (
    commit_output_dir,
    image_id,
    load_input_manifest,
    prepare_output_dir,
    safe_model_id,
    sanitize_error,
    sanitize_metadata,
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
        import torch

        value = sanitize_metadata({"tensor": torch.tensor([1, 2])})
        self.assertEqual(value, {"tensor": [1, 2]})


if __name__ == "__main__":
    unittest.main()
