import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.phase0_atlas_manifest import build_selection
from src.datasets.imagenet_subset import ImageNetSubsetDataset
from src.evaluation.feature_cache import dataset_snapshot


class AtlasManifestTests(unittest.TestCase):
    def test_selection_is_model_blind_and_stable(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "wnids.txt"
            manifest.write_text("n00000001\nn00000002\n", encoding="utf-8")
            for class_offset, wnid in enumerate(
                ("n00000001", "n00000002")
            ):
                class_dir = Path(root) / "train" / wnid
                class_dir.mkdir(parents=True)
                for index in range(3):
                    Image.new(
                        "RGB",
                        (4, 4),
                        color=(index * 20, class_offset * 30, 0),
                    ).save(class_dir / ("%d.JPEG" % index))
            dataset = ImageNetSubsetDataset(root, "train", manifest)
            snapshot = dataset_snapshot(
                dataset, os.path.join(root, "snapshot.json")
            )
            names = {"n00000001": "one", "n00000002": "two"}
            first = build_selection(
                dataset, snapshot, names, 2, 4, "test-v1"
            )
            second = build_selection(
                dataset, snapshot, names, 2, 4, "test-v1"
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)
            self.assertEqual(
                {item["class_name"] for item in first}, {"one", "two"}
            )
            self.assertTrue(
                all(len(item["selection_score"]) == 64 for item in first)
            )


if __name__ == "__main__":
    unittest.main()
