import json
import gc
import os
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from scripts.phase0_imagenet50_eval import (
    evaluation_transform_profile,
    feature_guide_identity,
    guide_transform,
    prepare_writers,
    validate_evaluation_batch_size,
)
from src.datasets.imagenet_subset import ImageNetSubsetDataset
from src.evaluation.feature_cache import (
    FeatureCacheWriter,
    dataset_snapshot,
    load_feature_cache,
)


class FeatureCacheTests(unittest.TestCase):
    def test_cache_resumes_and_finalizes_with_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "cache")
            provenance = {"dataset": "tiny", "model": "fake"}
            sample_ids = ["a", "b", "c"]
            labels = [0, 1, 1]
            writer = FeatureCacheWriter(
                output, provenance, sample_ids, labels
            )
            writer.write_batch(
                [[1.0, 2.0], [3.0, 4.0]],
                start=0,
                elapsed_seconds=1.0,
                feature_metadata={"readout": "fake"},
            )
            del writer
            gc.collect()

            resumed = FeatureCacheWriter(
                output, provenance, sample_ids, labels
            )
            self.assertEqual(resumed.completed, 2)
            resumed.write_batch(
                [[5.0, 6.0]],
                start=2,
                elapsed_seconds=0.5,
            )
            manifest = resumed.finalize()
            self.assertTrue(manifest["completed"])
            self.assertEqual(manifest["feature_dim"], 2)
            self.assertAlmostEqual(manifest["images_per_second"], 2.0)

            loaded = load_feature_cache(output)
            np.testing.assert_allclose(
                loaded["features"],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            )
            self.assertEqual(loaded["sample_ids"], sample_ids)
            del loaded
            gc.collect()

    def test_cache_rejects_identity_change(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "cache")
            FeatureCacheWriter(output, {"model": "a"}, ["x"], [0])
            with self.assertRaises(RuntimeError):
                FeatureCacheWriter(output, {"model": "b"}, ["x"], [0])

    def test_noncontiguous_write_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            writer = FeatureCacheWriter(
                os.path.join(root, "cache"),
                {"model": "a"},
                ["x", "y"],
                [0, 1],
            )
            with self.assertRaises(ValueError):
                writer.write_batch([[1.0]], 1, 0.1)

    def test_corrupted_feature_file_fails_verification(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "cache")
            writer = FeatureCacheWriter(
                output, {"model": "a"}, ["x"], [0]
            )
            writer.write_batch([[1.0]], 0, 0.1)
            writer.finalize()
            with open(os.path.join(output, "features.npy"), "r+b") as handle:
                handle.seek(-1, os.SEEK_END)
                value = handle.read(1)
                handle.seek(-1, os.SEEK_END)
                handle.write(bytes([value[0] ^ 0xFF]))
            with self.assertRaises(RuntimeError):
                load_feature_cache(output)

    def test_frozen_transforms_are_guide_specific_and_aspect_preserving(self):
        image = Image.new("RGB", (12, 4), color="blue")
        qwen = guide_transform("qwen3_vl")(image)
        molmo = guide_transform("molmo")(image)
        ijepa = guide_transform("ijepa")(image)
        dinov3 = guide_transform("dinov3")(image)
        self.assertEqual(tuple(qwen.shape), (3, 4, 12))
        self.assertEqual(tuple(molmo.shape), (3, 4, 12))
        self.assertEqual(tuple(ijepa.shape), (3, 224, 224))
        self.assertEqual(tuple(dinov3.shape), (3, 224, 224))
        self.assertTrue(
            evaluation_transform_profile("qwen3_vl")[
                "preserves_source_aspect_ratio"
            ]
        )
        self.assertEqual(
            evaluation_transform_profile("molmo")["official_processor"],
            "global_plus_up_to_24_local_crops",
        )
        self.assertEqual(
            evaluation_transform_profile("dinov3")["name"],
            "resize_256_center_crop_224",
        )

    def test_vlm_feature_identity_ignores_obsolete_square_input_size(self):
        values = {
            "model_id": "official/model",
            "revision": "abc",
            "input_size": 512,
            "dtype": "bfloat16",
        }
        self.assertNotIn(
            "input_size",
            feature_guide_identity(values, guide_name="qwen3_vl"),
        )
        self.assertIn(
            "input_size",
            feature_guide_identity(values, guide_name="ijepa"),
        )

    def test_vlm_evaluation_requires_batch_one(self):
        self.assertEqual(validate_evaluation_batch_size("qwen3_vl", 1), 1)
        self.assertEqual(validate_evaluation_batch_size("ijepa", 8), 8)
        for guide_name in ("qwen3_vl", "molmo"):
            with self.subTest(guide=guide_name), self.assertRaises(ValueError):
                validate_evaluation_batch_size(guide_name, 2)

    def test_cache_provenance_records_full_source_vlm_profile(self):
        with tempfile.TemporaryDirectory() as root:
            wnids = os.path.join(root, "wnids.txt")
            with open(wnids, "w", encoding="utf-8") as handle:
                handle.write("n00000001\n")
            for split in ("train", "val"):
                class_dir = os.path.join(root, "data", split, "n00000001")
                os.makedirs(class_dir)
                Image.new("RGB", (12, 4), color="red").save(
                    os.path.join(class_dir, "one.JPEG")
                )
            args = SimpleNamespace(
                data_root=os.path.join(root, "data"),
                wnid_manifest=wnids,
                dataset_manifest_dir=os.path.join(root, "manifests"),
                cache_dir=os.path.join(root, "cache"),
                refresh_dataset_snapshot=False,
                overwrite=False,
            )
            prepared = prepare_writers(
                args,
                {"dataset": {"id": "tiny"}},
                "qwen3_vl",
                {
                    "model_id": "official/model",
                    "revision": "abc",
                    "input_size": 512,
                },
            )
            for item in prepared.values():
                provenance = item["writer"].provenance
                profile = provenance["input_transform_profile"]
                self.assertEqual(
                    profile["name"],
                    "full_source_to_tensor_qwen_dynamic_resolution",
                )
                self.assertEqual(
                    profile["official_processor"],
                    "aspect_preserving_dynamic_resolution",
                )
                self.assertTrue(profile["preserves_source_aspect_ratio"])
                self.assertNotIn(
                    "shared_input_transform", provenance
                )
                self.assertNotIn(
                    "input_size", provenance["guide_config"]
                )


class DatasetSnapshotTests(unittest.TestCase):
    def test_snapshot_hashes_image_content_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as root:
            manifest_path = os.path.join(root, "wnids.txt")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write("n00000001\n")
            class_dir = os.path.join(root, "train", "n00000001")
            os.makedirs(class_dir)
            image_path = os.path.join(class_dir, "one.JPEG")
            Image.new("RGB", (4, 4), color="red").save(image_path)
            dataset = ImageNetSubsetDataset(
                root, "train", manifest_path
            )
            output = os.path.join(root, "snapshot.json")
            snapshot = dataset_snapshot(dataset, output)
            self.assertEqual(snapshot["count"], 1)
            self.assertEqual(len(snapshot["files"][0]["sha256"]), 64)
            json.dumps(snapshot)

            Image.new("RGB", (5, 5), color="blue").save(image_path)
            with self.assertRaises(RuntimeError):
                dataset_snapshot(dataset, output)


if __name__ == "__main__":
    unittest.main()
