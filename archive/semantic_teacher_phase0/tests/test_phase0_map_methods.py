import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.guides.base import GroundingBox, GroundingPoint, GuideOutput
from src.guides.maps import (
    grounding_score_map,
    illustrative_target_rectangles,
    token_pca_rgb,
)
from src.guides.tokencut import (
    load_official_ncut,
    tokencut_partition,
)


class PCAVisualizationTests(unittest.TestCase):
    def test_rgb_shape_and_determinism(self):
        torch.manual_seed(3)
        output = GuideOutput(
            patch_tokens=torch.randn(2, 16, 8),
            grid_size=(4, 4),
        )
        first = token_pca_rgb(output)
        second = token_pca_rgb(output)
        self.assertEqual(tuple(first.shape), (2, 4, 4, 3))
        self.assertTrue(torch.allclose(first, second))

    def test_foreground_shape_is_validated(self):
        output = GuideOutput(
            patch_tokens=torch.randn(1, 16, 8),
            grid_size=(4, 4),
        )
        with self.assertRaises(ValueError):
            token_pca_rgb(output, torch.ones(1, 3, 3))


class TokenCutWrapperTests(unittest.TestCase):
    @staticmethod
    def fake_ncut(
        feats,
        dims,
        scales,
        init_image_size,
        tau=0.2,
        eps=1e-5,
    ):
        del feats, scales, init_image_size, tau, eps
        height, width = dims
        mask = np.zeros((height, width), dtype=np.float32)
        mask[:, : max(1, width // 2)] = 1.0
        eigenvector = np.arange(height * width).reshape(height, width)
        return [0, 0, width // 2, height], None, mask, 0, None, eigenvector

    def test_batch_contract_and_metadata(self):
        result = tokencut_partition(
            torch.randn(2, 16, 8),
            (4, 4),
            (224, 224),
            ncut_fn=self.fake_ncut,
        )
        self.assertEqual(tuple(result.masks.shape), (2, 4, 4))
        self.assertEqual(tuple(result.eigenvectors.shape), (2, 4, 4))
        self.assertEqual(len(result.boxes), 2)
        self.assertEqual(
            result.metadata["source_commit"],
            "fed52cd5b60891baefd8ec7110dafa73be816ee1",
        )

    def test_missing_official_checkout_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                load_official_ncut(Path(root))

    def test_grid_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            tokencut_partition(
                torch.randn(1, 15, 8),
                (4, 4),
                (224, 224),
                ncut_fn=self.fake_ncut,
            )


class GroundingMapTests(unittest.TestCase):
    def test_box_and_point_are_rasterized_in_native_coordinates(self):
        box = GroundingBox(
            label="dog",
            bbox_2d=(0, 0, 500, 500),
        )
        point = GroundingPoint(
            label="dog",
            point_2d=(625, 625),
            coordinate_space="normalized_1000",
        )
        score = grounding_score_map(
            [[box]], [[point]], grid_size=(4, 4), point_sigma_fraction=0.1
        )
        self.assertEqual(tuple(score.shape), (1, 4, 4))
        self.assertAlmostEqual(float(score[0, 0, 0]), 1.0)
        self.assertGreater(float(score[0, 2, 2]), 0.9)

    def test_pixel_point_uses_processor_image_size(self):
        point = GroundingPoint(
            label="boat",
            point_2d=(150, 50),
            coordinate_space="pixels",
            image_size=(200, 100),
        )
        score = grounding_score_map(
            [[]], [[point]], grid_size=(4, 4), point_sigma_fraction=0.1
        )
        self.assertEqual(
            int(score[0].flatten().argmax()),
            1 * 4 + 2,
        )

    def test_illustrative_rectangles_are_nonoverlapping(self):
        score = torch.arange(64, dtype=torch.float32).view(1, 8, 8)
        rectangles = illustrative_target_rectangles(
            score, block_size=(2, 2), count=4
        )[0]
        self.assertEqual(len(rectangles), 4)
        occupied = set()
        for row, column, height, width, _ in rectangles:
            cells = {
                (y, x)
                for y in range(row, row + height)
                for x in range(column, column + width)
            }
            self.assertTrue(occupied.isdisjoint(cells))
            occupied.update(cells)


if __name__ == "__main__":
    unittest.main()
