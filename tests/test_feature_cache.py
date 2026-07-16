import json
import gc
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

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
