import unittest
import subprocess
import sys

import torch

from src.guides import available_guides
from src.guides.base import GuideOutput
from src.guides.maps import (
    border_occupancy,
    effective_support,
    extract_maps,
    map_entropy,
    minmax_normalize,
    rank_normalize,
    token_pca_map,
    top_fraction_mask,
)


class GuideOutputTests(unittest.TestCase):
    def test_package_import_is_dependency_isolated(self):
        self.assertEqual(
            set(available_guides()), {"clip", "dinov3", "ijepa", "siglip2"}
        )
        command = (
            "import sys; import src.guides; "
            "assert 'transformers' not in sys.modules; "
            "assert 'matplotlib' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", command], check=True)

    def test_valid_output(self):
        output = GuideOutput(
            patch_tokens=torch.randn(2, 16, 8),
            grid_size=(4, 4),
            global_token=torch.randn(2, 8),
            native_map=torch.randn(2, 4, 4),
        )
        self.assertEqual(output.batch_size, 2)
        self.assertEqual(output.embed_dim, 8)

    def test_rejects_bad_grid(self):
        with self.assertRaises(ValueError):
            GuideOutput(
                patch_tokens=torch.randn(1, 15, 8),
                grid_size=(4, 4),
            )


class MapTests(unittest.TestCase):
    def test_minmax_normalization(self):
        score_map = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
        normalized = minmax_normalize(score_map)
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_rank_normalization_is_order_preserving(self):
        score_map = torch.tensor([[[4.0, 1.0], [3.0, 2.0]]])
        ranked = rank_normalize(score_map).flatten()
        order = torch.argsort(score_map.flatten())
        self.assertTrue(torch.equal(torch.argsort(ranked), order))

    def test_rank_normalization_uses_average_rank_for_ties(self):
        score_map = torch.ones(1, 2, 3)
        ranked = rank_normalize(score_map)
        self.assertTrue(
            torch.allclose(ranked, torch.full_like(ranked, 0.5))
        )

    def test_top_fraction_selects_exact_count(self):
        score_map = torch.arange(16.0).view(1, 4, 4)
        mask = top_fraction_mask(score_map, 0.25)
        self.assertEqual(int(mask.sum()), 4)

    def test_uniform_map_has_full_effective_support(self):
        score_map = torch.zeros(2, 4, 4)
        support = effective_support(score_map)
        self.assertTrue(torch.allclose(support, torch.ones_like(support)))

    def test_temperature_controls_nonuniform_support(self):
        score_map = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
        cold = effective_support(score_map, temperature=0.1)
        warm = effective_support(score_map, temperature=1.0)
        self.assertLess(float(cold), float(warm))
        self.assertLess(float(map_entropy(score_map, 0.1)), 1.0)

    def test_border_occupancy(self):
        score_map = torch.zeros(1, 4, 4)
        score_map[:, 0, :] = 10.0
        occupancy = border_occupancy(score_map, fraction=0.25)
        self.assertAlmostEqual(float(occupancy), 1.0)

    def test_extract_maps(self):
        output = GuideOutput(
            patch_tokens=torch.randn(2, 16, 8),
            grid_size=(4, 4),
            global_token=torch.randn(2, 8),
            native_map=torch.randn(2, 4, 4),
        )
        maps = extract_maps(output)
        self.assertEqual(
            set(maps), {"native", "global_cosine", "token_pca"}
        )
        for score_map in maps.values():
            self.assertEqual(tuple(score_map.shape), (2, 4, 4))

    def test_pca_sign_does_not_depend_on_native_map(self):
        tokens = torch.randn(1, 16, 8)
        output_a = GuideOutput(
            patch_tokens=tokens,
            grid_size=(4, 4),
            native_map=torch.randn(1, 4, 4),
        )
        output_b = GuideOutput(
            patch_tokens=tokens,
            grid_size=(4, 4),
            native_map=-output_a.native_map,
        )
        self.assertTrue(
            torch.allclose(token_pca_map(output_a), token_pca_map(output_b))
        )


if __name__ == "__main__":
    unittest.main()
