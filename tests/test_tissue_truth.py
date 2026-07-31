"""Tests for the tissue-truth references used to score masking policies.

The masking sweep is only as trustworthy as its notion of "real tissue", and
the first reference was wrong in a way that silently inflated every reported
purity.  These tests pin the properties that failure violated.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.guides.tissue_truth import (
    TRUTH_MODES,
    build_truth,
    otsu_threshold,
    patch_coverage,
    tissue_pixels_noise_band,
    truth_patchmean_otsu,
    truth_pixel_otsu,
)

PATCH = 16
GRID = 16
SIZE = PATCH * GRID


def synthetic_slice(
    top: int = 60,
    bright_thickness: int = 24,
    dim_thickness: int = 60,
    bright: int = 190,
    dim: int = 45,
    background: int = 18,
    noise: float = 6.0,
    seed: int = 0,
) -> np.ndarray:
    """A B-scan caricature: bright retina over dim choroid over dark vitreous.

    ``dim`` sits far below ``bright`` deliberately.  Otsu maximises between-class
    variance, so with a large dark background it splits between "bright bands"
    and "everything else" and discards the whole choroid -- which is exactly how
    the retired patch-mean reference lost roughly half the tissue on real data.
    """
    rng = np.random.default_rng(seed)
    image = np.full((SIZE, SIZE), background, dtype=np.float32)
    image[top:top + bright_thickness] = bright
    image[top + bright_thickness:top + bright_thickness + dim_thickness] = dim
    image += rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_otsu_threshold_separates_two_modes():
    values = np.concatenate([np.full(500, 20.0), np.full(500, 200.0)])
    assert 20.0 < otsu_threshold(values) < 200.0


def test_otsu_threshold_handles_degenerate_input():
    constant = np.full(64, 42.0)
    assert np.isfinite(otsu_threshold(constant))


def test_patch_coverage_is_a_fraction():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[:PATCH // 2, :PATCH] = True          # half of the first patch
    coverage = patch_coverage(mask, PATCH)
    assert coverage.shape == (GRID, GRID)
    assert coverage[0, 0] == pytest.approx(0.5)
    assert coverage[0, 1] == pytest.approx(0.0)


def test_patch_coverage_full_patch_is_one():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[:PATCH, :PATCH] = True
    assert patch_coverage(mask, PATCH)[0, 0] == pytest.approx(1.0)


def test_noise_band_recovers_dim_tissue_that_patchmean_otsu_drops():
    """The defect that invalidated the first sweep: dim tissue read as void."""
    image = synthetic_slice()
    old = truth_patchmean_otsu(image, patch=PATCH)
    new = build_truth("noise_band", image, patch=PATCH)
    assert new.sum() > old.sum() * 1.4


def test_noise_band_covers_the_dim_band_rows():
    image = synthetic_slice(top=64, bright_thickness=32, dim_thickness=64)
    truth = build_truth("noise_band", image, patch=PATCH)
    dim_rows = slice((64 + 32) // PATCH, (64 + 32 + 64) // PATCH)
    assert truth[dim_rows].mean() > 0.9


def test_noise_band_excludes_vitreous():
    image = synthetic_slice(top=96, bright_thickness=32, dim_thickness=48)
    truth = build_truth("noise_band", image, patch=PATCH)
    assert truth[:96 // PATCH - 1].sum() == 0


def test_pixel_otsu_recovers_a_half_covered_boundary_patch():
    """Pixel-level coverage makes a half-tissue patch recoverable.

    Patch-mean thresholding cannot express this: the patch has one number, so
    it is all-or-nothing.  Coverage keeps the fraction, so the same patch is
    tissue at a 0.4 requirement and background at 0.6.
    """
    edge_row = 4
    column = GRID // 2
    image = synthetic_slice(
        top=PATCH * edge_row + PATCH // 2,   # tissue starts mid-patch
        bright_thickness=PATCH * 3,
        dim_thickness=0,
        noise=0.0,
    )
    means = (
        image.astype(np.float32)
        .reshape(GRID, PATCH, GRID, PATCH)
        .mean(axis=(1, 3))
    )
    assert 18 < means[edge_row, column] < 190      # genuinely a mixed patch

    assert truth_pixel_otsu(image, patch=PATCH, coverage=0.4)[edge_row, column]
    assert not truth_pixel_otsu(
        image, patch=PATCH, coverage=0.6
    )[edge_row, column]


def test_column_band_bridges_speckle_holes():
    """Speckle punches holes through the retina; the band must close them."""
    image = synthetic_slice(noise=0.0)
    holed = image.copy()
    holed[70:74, ::3] = 18                    # thin dropouts inside the retina
    pixels = tissue_pixels_noise_band(holed)
    assert pixels[70:74, ::3].mean() > 0.8


def test_column_band_does_not_span_across_a_large_dark_gap():
    """A bright artefact near the floor must not drag the band through void."""
    image = synthetic_slice(top=32, bright_thickness=24, dim_thickness=8,
                            noise=0.0)
    image[220:228] = 200                      # detached bright speck at bottom
    pixels = tissue_pixels_noise_band(image)
    middle = pixels[120:200]
    assert middle.mean() < 0.15


def test_union_mirage_is_a_superset_of_noise_band():
    image = synthetic_slice()
    mirage = np.zeros((SIZE, SIZE), dtype=bool)
    mirage[150:180, 40:200] = True
    band = build_truth("noise_band", image, patch=PATCH)
    union = build_truth("union_mirage", image, patch=PATCH, mirage_raw=mirage)
    assert np.all(union | band == union)
    assert union.sum() >= band.sum()


def test_union_mirage_without_a_guide_equals_noise_band():
    image = synthetic_slice()
    assert np.array_equal(
        build_truth("union_mirage", image, patch=PATCH),
        build_truth("noise_band", image, patch=PATCH),
    )


def test_all_modes_return_a_boolean_grid():
    image = synthetic_slice()
    for name in TRUTH_MODES:
        truth = build_truth(name, image, patch=PATCH)
        assert truth.shape == (GRID, GRID), name
        assert truth.dtype == bool, name


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown truth mode"):
        build_truth("nope", synthetic_slice())


def test_truth_is_deterministic():
    image = synthetic_slice(seed=3)
    first = build_truth("noise_band", image, patch=PATCH)
    second = build_truth("noise_band", image, patch=PATCH)
    assert np.array_equal(first, second)


def test_higher_k_never_grows_the_region():
    """k scales the threshold above the noise floor, so it must be monotone."""
    image = synthetic_slice()
    counts = [
        build_truth("noise_band", image, patch=PATCH, k=k).sum()
        for k in (2.0, 3.0, 4.0, 6.0)
    ]
    assert counts == sorted(counts, reverse=True)


def test_blank_slice_yields_no_tissue():
    rng = np.random.default_rng(0)
    void = np.clip(rng.normal(18, 6, (SIZE, SIZE)), 0, 255).astype(np.uint8)
    assert build_truth("noise_band", void, patch=PATCH).sum() == 0
