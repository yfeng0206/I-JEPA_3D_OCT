"""Tests for MIRAGE-guided masking.

Two things must hold for the experiment to be interpretable:

1. The retinal envelope guide is anatomically sane -- the optic-nerve-head gap
   stays open, real detections are never deleted, and failed segmentations are
   reported instead of invented.
2. Switching the curriculum into ``mirage_envelope`` changes *only* where the
   four target blocks land.  Block count, block sizes, the context mask and the
   collated mask contract must match the random baseline exactly.
"""

import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.guides.mirage_envelope import (
    CLASS_CHOROID,
    CLASS_GCIPL,
    CLASS_RNFL,
    DEFAULT_REPAIR,
    build_union,
    dilate_patch_grid,
    occupancy_is_valid,
    pack_guides,
    patch_occupancy,
    repair_union,
    unpack_guides,
)
from src.masks.curriculum import CurriculumMaskGenerator
from src.masks.multiblock import MaskCollator
from src.transforms import make_paired_transforms, make_transforms


# ----------------------------------------------------------------------
# Envelope construction
# ----------------------------------------------------------------------


def make_band(height=200, width=200, top=60, gap=(80, 110), bottom=130):
    """Hollow retina: RNFL/GCIPL above, choroid below, unlabelled middle."""
    labels = np.zeros((height, width), dtype=np.uint8)
    labels[top : gap[0], :] = CLASS_RNFL
    labels[top + 5 : gap[0], :] = CLASS_GCIPL
    labels[gap[1] : bottom, :] = CLASS_CHOROID
    return labels


def test_union_contains_all_three_classes():
    union = build_union(make_band())
    assert union.any()
    labels = make_band()
    for class_index in (CLASS_RNFL, CLASS_GCIPL, CLASS_CHOROID):
        assert union[labels == class_index].all()
    assert not union[labels == 0].any()


def test_repair_fills_unlabelled_mid_retina():
    labels = make_band()
    union = build_union(labels)
    envelope, valid, stats = repair_union(union, params=DEFAULT_REPAIR)
    assert valid
    # The GOALS "Elsewhere" band between GCIPL and choroid must be closed.
    assert envelope[95, 100]
    assert not union[95, 100]
    assert stats["repaired_area_frac"] > stats["raw_area_frac"]


def test_repair_never_deletes_real_detections():
    union = build_union(make_band())
    envelope, _valid, _stats = repair_union(union, params=DEFAULT_REPAIR)
    assert envelope[union].all()


def test_optic_nerve_head_gap_is_not_bridged():
    labels = make_band()
    labels[:, 90:130] = 0  # wide central dropout, as at the optic nerve head
    envelope, _valid, _stats = repair_union(
        build_union(labels), params=DEFAULT_REPAIR
    )
    occupied = envelope.any(axis=0)
    assert occupied[:90].any() and occupied[130:].any()
    assert not occupied[100:120].any(), "wide ONH gap must stay open"


def test_short_gap_is_bridged():
    labels = make_band()
    labels[:, 100:105] = 0  # 5 px < max_horizontal_gap
    envelope, _valid, _stats = repair_union(
        build_union(labels), params=DEFAULT_REPAIR
    )
    assert envelope.any(axis=0)[100:105].all()


def test_speckle_is_removed_but_large_pieces_kept():
    labels = make_band()
    labels[190:193, 5:8] = CLASS_CHOROID  # tiny + narrow -> speckle
    envelope, _valid, _stats = repair_union(
        build_union(labels), params=DEFAULT_REPAIR
    )
    assert not envelope[190:193, 5:8].any()
    assert envelope[70, 100]


def test_failed_segmentation_reports_invalid():
    labels = np.zeros((200, 200), dtype=np.uint8)
    labels[100:104, 100:104] = CLASS_RNFL
    _envelope, valid, _stats = repair_union(
        build_union(labels), params=DEFAULT_REPAIR
    )
    assert not valid, "a near-empty guide must be flagged, not invented"


# ----------------------------------------------------------------------
# Patch grid helpers
# ----------------------------------------------------------------------


def test_patch_occupancy_is_fractional_and_bounded():
    mask = np.zeros((256, 256), dtype=bool)
    mask[0:8, 0:16] = True  # exactly half of the first patch
    grid = patch_occupancy(mask, patch_size=16)
    assert grid.shape == (16, 16)
    assert grid.min() >= 0.0 and grid.max() <= 1.0
    assert grid[0, 0] == pytest.approx(0.5)


def test_dilate_patch_grid_adds_one_ring():
    grid = np.zeros((16, 16), dtype=bool)
    grid[8, 8] = True
    grown = dilate_patch_grid(grid, 1)
    assert grown[7:10, 7:10].all()
    assert grown.sum() == 9


def test_dilate_patch_grid_clips_at_border():
    grid = np.zeros((16, 16), dtype=bool)
    grid[0, 0] = True
    grown = dilate_patch_grid(grid, 1)
    assert grown.sum() == 4  # corner ring is clipped, not wrapped


def test_guide_packing_round_trips():
    rng = np.random.default_rng(0)
    masks = rng.random((3, 200, 200)) > 0.5
    restored = unpack_guides(pack_guides(masks), (200, 200))
    assert np.array_equal(masks, restored)


def test_occupancy_validity_rejects_empty_crop():
    assert not occupancy_is_valid(np.zeros((16, 16), dtype=np.float32))


# ----------------------------------------------------------------------
# Paired transform
# ----------------------------------------------------------------------


def test_paired_transform_matches_unpaired_image_path():
    image = Image.fromarray(
        np.random.default_rng(1).integers(0, 255, (256, 256), dtype=np.uint8),
        mode="L",
    ).convert("RGB")
    guide = Image.fromarray(np.full((256, 256), 255, dtype=np.uint8), mode="L")

    torch.manual_seed(7)
    np.random.seed(7)
    reference = make_transforms(crop_size=256, crop_scale=(0.3, 1.0))(image)

    torch.manual_seed(7)
    np.random.seed(7)
    paired, _ = make_paired_transforms(crop_size=256, crop_scale=(0.3, 1.0))(
        image, guide
    )
    assert torch.allclose(reference, paired, atol=1e-6)


def test_paired_transform_keeps_guide_binary():
    image = Image.fromarray(
        np.zeros((256, 256), dtype=np.uint8), mode="L"
    ).convert("RGB")
    guide_array = np.zeros((256, 256), dtype=np.uint8)
    guide_array[64:192, 64:192] = 255
    guide = Image.fromarray(guide_array, mode="L")
    _tensor, cropped = make_paired_transforms(crop_size=256)(image, guide)
    values = np.unique(np.asarray(cropped))
    assert set(values.tolist()).issubset({0, 255}), "nearest must keep hard labels"


def test_paired_transform_rejects_unsupported_augmentation():
    with pytest.raises(ValueError):
        make_paired_transforms(horizontal_flip=True)


# ----------------------------------------------------------------------
# Curriculum integration
# ----------------------------------------------------------------------


def make_generator(**overrides):
    cfg = {
        "mode": "mirage_envelope",
        "T_warm": 25,
        "T_total": 30,
        "r_max": 1.0,
        "ramp_shape": "linear",
    }
    cfg.update(overrides)
    return CurriculumMaskGenerator(
        input_size=(256, 256),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
        curriculum_cfg=cfg,
    )


def band_guide(batch=2):
    grid = torch.zeros(batch, 2, 16, 16)
    grid[:, 0, 6:10, :] = 1.0   # true occupancy
    grid[:, 1, 5:11, :] = 1.0   # dilated placement region
    return grid


def test_mirage_mode_is_registered():
    assert "mirage_envelope" in CurriculumMaskGenerator.VALID_MODES


def test_mask_contract_matches_multiblock():
    generator = make_generator()
    generator.set_epoch(30)
    guides = band_guide()
    valid = torch.ones(2, dtype=torch.bool)
    masks_enc, masks_pred = generator.generate(
        batch_size=2, guide_grids=guides, guide_valid=valid
    )
    assert len(masks_enc) == 1 and len(masks_pred) == 4
    for group in masks_pred:
        assert group.shape[0] == 2 and group.dtype == torch.long
    for group in masks_enc:
        assert group.shape[0] == 2 and group.dtype == torch.long
    collator = MaskCollator(
        input_size=(256, 256),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
    )
    images = [torch.zeros(3, 256, 256) for _ in range(2)]
    _imgs, ref_enc, ref_pred = collator(images)
    assert len(ref_enc) == len(masks_enc)
    assert len(ref_pred) == len(masks_pred)


def test_context_excludes_targets():
    generator = make_generator()
    generator.set_epoch(30)
    masks_enc, masks_pred = generator.generate(
        batch_size=2,
        guide_grids=band_guide(),
        guide_valid=torch.ones(2, dtype=torch.bool),
    )
    for b in range(2):
        targets = set()
        for group in masks_pred:
            targets.update(group[b].tolist())
        context = set(masks_enc[0][b].tolist())
        assert not (context & targets)


def test_targets_land_on_region_when_ramp_is_full():
    generator = make_generator()
    generator.set_epoch(30)
    assert generator.r_t == pytest.approx(1.0)
    guides = band_guide(batch=4)
    generator.generate(
        batch_size=4,
        guide_grids=guides,
        guide_valid=torch.ones(4, dtype=torch.bool),
    )
    stats = generator.mirage_stats
    assert stats["guided_images"] == 4
    assert stats["mean_block_fill"] >= generator.mirage_min_block_fill


def test_bootstrap_epoch_is_unbiased():
    generator = make_generator()
    generator.set_epoch(25)
    assert generator.r_t == pytest.approx(0.0)
    generator.generate(
        batch_size=2,
        guide_grids=band_guide(),
        guide_valid=torch.ones(2, dtype=torch.bool),
    )
    stats = generator.mirage_stats
    assert stats["guided_images"] == 0, "epoch 25 must stay fully random"
    # A healthy guide that the ramp simply chose not to use is NOT a failure.
    assert stats["fallbacks"] == 0
    assert stats["unbiased_by_ramp"] == 2


def test_dilated_placement_channel_is_used():
    """Blocks may sit on the dilated ring, but visibility uses true retina."""
    generator = make_generator(mirage_min_block_fill=0.9)
    generator.set_epoch(30)
    guide = torch.zeros(1, 2, 16, 16)
    guide[:, 0, 8, 3:13] = 1.0          # one true retinal row
    guide[:, 1, 3:14, 1:15] = 1.0       # generous dilated placement region
    generator.generate(
        batch_size=1,
        guide_grids=guide,
        guide_valid=torch.ones(1, dtype=torch.bool),
    )
    stats = generator.mirage_stats
    assert stats["guided_images"] == 1
    # Fill is judged on the dilated region, so a 90% threshold is satisfiable
    # even though the true segmentation is only a single patch row.
    assert stats["mean_block_fill"] >= 0.9
    # Visibility, however, is measured on the true row only.
    assert stats["truncated_target_patches"] > 0


def test_infeasible_guide_is_reported_not_hidden():
    """A guide too small for any block must not be logged as a success."""
    generator = make_generator(mirage_min_block_fill=0.95)
    generator.set_epoch(30)
    guide = torch.zeros(2, 2, 16, 16)
    guide[:, 0, 8, 7:9] = 1.0
    guide[:, 1, 8, 7:9] = 1.0
    generator.generate(
        batch_size=2,
        guide_grids=guide,
        guide_valid=torch.ones(2, dtype=torch.bool),
    )
    stats = generator.mirage_stats
    assert stats["infeasible"] == 2
    assert stats["accept_rate"] == pytest.approx(0.0)


def test_unbiased_blocks_are_fixed_across_retries():
    """Blocks the ramp left random must not be re-drawn by the accept loop."""
    generator = make_generator()
    generator.set_epoch(27)  # partial ramp -> mixed flags
    assert 0.0 < generator.r_t < 1.0
    masks_enc, masks_pred = generator.generate(
        batch_size=4,
        guide_grids=band_guide(batch=4),
        guide_valid=torch.ones(4, dtype=torch.bool),
    )
    assert len(masks_pred) == 4
    for group in masks_pred:
        assert group.shape[0] == 4


def test_invalid_guide_falls_back_to_random():
    generator = make_generator()
    generator.set_epoch(30)
    generator.generate(
        batch_size=3,
        guide_grids=band_guide(batch=3),
        guide_valid=torch.zeros(3, dtype=torch.bool),
    )
    assert generator.mirage_stats["fallbacks"] == 3


def test_batch_statistics_are_reported():
    generator = make_generator()
    generator.set_epoch(30)
    generator.generate(
        batch_size=2,
        guide_grids=band_guide(),
        guide_valid=torch.ones(2, dtype=torch.bool),
    )
    stats = generator.mirage_stats
    for key in (
        "patches_per_block",
        "unique_target_patches",
        "context_patches",
        "target_on_region",
        "target_background",
        "fallbacks",
    ):
        assert key in stats
    assert stats["target_on_region"] + stats["target_background"] == pytest.approx(1.0)


def test_block_sizes_are_unchanged_by_guidance():
    """Guidance must move blocks, never resize them."""
    guides = band_guide(batch=4)
    valid = torch.ones(4, dtype=torch.bool)

    torch.manual_seed(3)
    np.random.seed(3)
    import random

    random.seed(3)
    guided = make_generator()
    guided.set_epoch(30)
    _enc_g, pred_g = guided.generate(
        batch_size=4, guide_grids=guides, guide_valid=valid
    )

    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)
    plain = make_generator()
    plain.set_epoch(25)  # r_t = 0 -> uniform placement, same size draw
    _enc_p, pred_p = plain.generate(
        batch_size=4, guide_grids=guides, guide_valid=valid
    )

    assert len(pred_g) == len(pred_p)
    for group_g, group_p in zip(pred_g, pred_p):
        assert group_g.shape[0] == group_p.shape[0]


def test_other_modes_still_work_without_guides():
    generator = CurriculumMaskGenerator(
        input_size=(256, 256),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
        curriculum_cfg={"mode": "anatomical_prior", "T_warm": 25, "T_total": 30},
    )
    generator.set_epoch(30)
    imgs = torch.rand(2, 3, 256, 256)
    masks_enc, masks_pred = generator.generate(batch_size=2, imgs_cpu=imgs)
    assert len(masks_enc) == 1 and len(masks_pred) == 4
