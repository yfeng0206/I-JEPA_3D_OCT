import unittest

import numpy as np
from PIL import Image

from src.guides.dino_pca import (
    dinov2_two_stage_pca,
    median_foreground_mask,
    paper_style_pca,
    pca_orientation_variants,
    quantize_foreground_mask,
    resize_aspect_to_patch_grid,
)


class DINOAspectResizeTests(unittest.TestCase):
    def test_resize_preserves_aspect_and_patch_alignment(self):
        image = Image.new("RGB", (800, 600), "red")
        tensor, grid = resize_aspect_to_patch_grid(
            image, image_height=768, patch_size=16
        )
        self.assertEqual(grid, (48, 64))
        self.assertEqual(tuple(tensor.shape), (3, 768, 1024))

    def test_quantized_mask_matches_grid(self):
        mask = Image.new("L", (80, 60), 255)
        values = quantize_foreground_mask(mask, (3, 4), patch_size=16)
        self.assertEqual(tuple(values.shape), (12,))
        self.assertTrue(np.allclose(values.numpy(), 1.0))


class DINOPCAProtocolTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.features = rng.normal(size=(16, 8)).astype(np.float32)
        self.mask = np.zeros((4, 4), dtype=bool)
        self.mask[1:3, 1:3] = True

    def test_paper_style_background_is_black(self):
        result = paper_style_pca(
            self.features, (4, 4), self.mask, whiten=True
        )
        self.assertEqual(result["rgb"].shape, (4, 4, 3))
        self.assertTrue(np.all(result["rgb"][~self.mask] == 0))
        self.assertTrue(
            np.all(
                (result["rgb"][self.mask] > 0)
                & (result["rgb"][self.mask] < 1)
            )
        )

    def test_orientation_count_and_metadata(self):
        result = paper_style_pca(self.features, (4, 4), self.mask)
        variants = pca_orientation_variants(
            result["projected"], self.mask
        )
        self.assertEqual(len(variants), 48)
        self.assertEqual(
            len({(item["order"], item["signs"]) for item in variants}),
            48,
        )
        self.assertTrue(
            all(np.all(item["rgb"][~self.mask] == 0) for item in variants)
        )

    def test_two_stage_pca_preserves_both_polarities(self):
        result = dinov2_two_stage_pca(
            self.features, (4, 4), threshold=0.5
        )
        self.assertEqual(result["pc1"].shape, (4, 4))
        self.assertEqual(
            [item["polarity"] for item in result["polarities"]],
            ["high_pc1", "low_pc1"],
        )
        masks = [item["mask"] for item in result["polarities"]]
        self.assertTrue(np.array_equal(~masks[0], masks[1]))

    def test_median_mask_shape(self):
        probability = np.linspace(0, 1, 25, dtype=np.float32)
        filtered, mask = median_foreground_mask(
            probability, (5, 5), threshold=0.5
        )
        self.assertEqual(filtered.shape, (5, 5))
        self.assertEqual(mask.shape, (5, 5))


if __name__ == "__main__":
    unittest.main()
