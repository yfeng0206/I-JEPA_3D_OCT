import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.datasets.imagenet_subset import (
    ImageNetSubsetDataset,
    atlas_selection_score,
    canonical_wnid_bytes,
    file_sha256,
    load_class_names,
    load_wnids,
    wnid_manifest_sha256,
)


class ImageNetManifestTests(unittest.TestCase):
    def test_manifest_order_defines_labels(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            manifest = root_path / "classes.txt"
            manifest.write_text(
                "n00000002\nn00000001\n", encoding="utf-8"
            )
            for wnid, color in (
                ("n00000001", "red"),
                ("n00000002", "blue"),
            ):
                class_dir = root_path / "train" / wnid
                class_dir.mkdir(parents=True)
                Image.new("RGB", (4, 4), color).save(
                    class_dir / (wnid + ".JPEG")
                )

            dataset = ImageNetSubsetDataset(
                root_path, "train", manifest
            )
            labels = {
                record["wnid"]: record["label"]
                for record in dataset.records()
            }
            self.assertEqual(labels["n00000002"], 0)
            self.assertEqual(labels["n00000001"], 1)

    def test_manifest_hash_normalizes_line_endings(self):
        with tempfile.TemporaryDirectory() as root:
            left = Path(root) / "left.txt"
            right = Path(root) / "right.txt"
            left.write_bytes(b"n00000001\nn00000002\n")
            right.write_bytes(b"n00000001\r\nn00000002\r\n")
            self.assertEqual(
                wnid_manifest_sha256(left),
                wnid_manifest_sha256(right),
            )
            expected = hashlib.sha256(
                canonical_wnid_bytes(("n00000001", "n00000002"))
            ).hexdigest()
            self.assertEqual(wnid_manifest_sha256(left), expected)

    def test_duplicate_and_invalid_wnids_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            duplicate = Path(root) / "duplicate.txt"
            duplicate.write_text(
                "n00000001\nn00000001\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_wnids(duplicate)

            invalid = Path(root) / "invalid.txt"
            invalid.write_text("dog\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_wnids(invalid)

    def test_class_names_require_exact_manifest_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "classes.json"
            path.write_text(
                json.dumps(
                    {
                        "classes": {
                            "n00000001": {
                                "imagenet_index": 1,
                                "label": "one",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_class_names(path, ("n00000001",)),
                {"n00000001": "one"},
            )
            with self.assertRaises(ValueError):
                load_class_names(path, ("n00000001", "n00000002"))

    def test_missing_class_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "train").mkdir()
            manifest = root_path / "classes.txt"
            manifest.write_text("n00000001\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                ImageNetSubsetDataset(root_path, "train", manifest)

    def test_sample_id_and_file_hash_are_stable(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            manifest = root_path / "classes.txt"
            manifest.write_text("n00000001\n", encoding="utf-8")
            class_dir = root_path / "val" / "n00000001"
            class_dir.mkdir(parents=True)
            image_path = class_dir / "sample.JPEG"
            Image.new("RGB", (4, 4), "green").save(image_path)

            dataset = ImageNetSubsetDataset(root_path, "val", manifest)
            _, label, sample_id = dataset[0]
            self.assertEqual(label, 0)
            self.assertEqual(
                sample_id, "val/n00000001/sample.JPEG"
            )
            self.assertEqual(file_sha256(image_path), file_sha256(image_path))

    def test_atlas_selection_score_is_namespaced(self):
        image_hash = "a" * 64
        first = atlas_selection_score("n00000001", image_hash, "v1")
        second = atlas_selection_score("n00000001", image_hash, "v2")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
