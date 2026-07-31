#!/usr/bin/env python
"""Preview region-only MIRAGE masking: every target patch inside the envelope.

The repaired retinal envelope is expanded outward by 5%, then four target
blocks are shrunk until a window exists that lies *entirely* within that region.
Several shrink levels are rendered side by side so the trade-off between target
purity and total masked area can be judged before training.

Example:
    python scripts/mirage_region_only_preview.py --volumes 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.guides.mirage_envelope import (  # noqa: E402
    DEFAULT_REPAIR,
    build_union,
    expand_envelope,
    patch_occupancy,
    repair_union,
)
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402

DATA_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\data\Training")
MASK_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_masks\Training")
OUTPUT_DIR = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_region_only")

CROP, PATCH = 256, 16
GRID = CROP // PATCH
EXPANSION = 0.05
MIN_OCCUPANCY = 0.5
TILE, HEADER, ROW_LABEL = 256, 20, 112

# (label, pred_mask_scale) -- the baseline keeps the current I-JEPA geometry.
VARIANTS = (
    ("current 0.15-0.20", (0.15, 0.20), "unconstrained"),
    ("CENTER 0.15-0.20 (masks 99% retina)", (0.15, 0.20), "center"),
    ("RECOMMENDED region-only x4", (0.04, 0.05), "region"),
    ("region-only x8 (harder)", (0.04, 0.05), "region8"),
)


def make_generator(pred_scale) -> CurriculumMaskGenerator:
    return CurriculumMaskGenerator(
        input_size=(CROP, CROP),
        patch_size=PATCH,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=pred_scale,
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
        curriculum_cfg={"mode": "anatomical_prior", "T_warm": 25, "T_total": 30,
                        "r_max": 1.0},
    )


def window_sums(grid: np.ndarray, block_h: int, block_w: int):
    """Summed-area-table block sums for every (block_h, block_w) window."""
    height, width = grid.shape
    n_top, n_left = height - block_h + 1, width - block_w + 1
    if n_top <= 0 or n_left <= 0:
        return None
    padded = np.zeros((height + 1, width + 1), dtype=np.float64)
    padded[1:, 1:] = grid
    sat = padded.cumsum(axis=0).cumsum(axis=1)
    return (
        sat[block_h:, block_w:]
        - sat[:n_top, block_w:]
        - sat[block_h:, :n_left]
        + sat[:n_top, :n_left]
    )


def sample_region_only(
    generator,
    occupancy,
    seed,
    avoid_overlap=True,
    spread=True,
):
    """Shrink each block until it fits wholly inside the region, then place it.

    Every admissible window is already 100% inside the region, so weighting
    candidates by occupancy would only bias placement toward the thickest part
    of the band and pack the blocks into one clump.  Candidates are therefore
    drawn uniformly, and with ``spread`` a distance-based repulsion pushes each
    new block away from the ones already placed so the targets sample the whole
    retina rather than a single spot.
    """
    in_region = (occupancy >= MIN_OCCUPANCY).astype(np.float64)
    generator._size_gen.manual_seed(seed)
    sizes = [
        generator._sample_block_size(generator.pred_mask_scale, generator._size_gen)
        for _ in range(generator.npred)
    ]
    torch.manual_seed(seed)
    random.seed(seed)

    claimed = np.zeros_like(in_region)
    centres: list = []
    blocks, union, shrink_steps, failures = [], set(), 0, 0

    # Stratify the band horizontally: block i is confined to the i-th slice of
    # the retina's lateral extent.  Repulsion alone still let blocks clump on
    # one part of the band, which left whole stretches of retina unsampled.
    occupied_cols = np.flatnonzero(in_region.any(axis=0))
    segments: list = []
    if spread and occupied_cols.size:
        edges = np.linspace(
            occupied_cols.min(), occupied_cols.max() + 1, len(sizes) + 1
        )
        segments = [
            (int(np.floor(edges[i])), int(np.ceil(edges[i + 1])))
            for i in range(len(sizes))
        ]
        order = np.random.default_rng(seed).permutation(len(segments))
        segments = [segments[i] for i in order]

    for block_index, (block_h, block_w) in enumerate(sizes):
        height, width = block_h, block_w
        placement = None
        segment = segments[block_index] if segments else None
        while height >= 1 and width >= 1:
            counts = window_sums(in_region, height, width)
            candidates = None
            if counts is not None:
                valid = np.isclose(counts, height * width)
                if avoid_overlap and valid.any():
                    overlap = window_sums(claimed, height, width)
                    free = valid & np.isclose(overlap, 0.0)
                    if free.any():
                        valid = free
                if segment is not None and valid.any():
                    # Keep shrinking until this block fits inside ITS OWN
                    # segment.  Steep parts of the band admit no wide rectangle,
                    # and abandoning the segment here is what let every block
                    # collapse onto the flat portion of the retina.
                    low, high = segment
                    banded = np.zeros_like(valid)
                    centre_cols = np.arange(valid.shape[1]) + width / 2.0
                    inside = (centre_cols >= low) & (centre_cols < high)
                    banded[:, inside] = valid[:, inside]
                    candidates = banded if banded.any() else None
                elif valid.any():
                    candidates = valid
            if candidates is not None and candidates.any():
                rows, cols = np.nonzero(candidates)
                if spread and centres:
                    centre_rows = rows + height / 2.0
                    centre_cols = cols + width / 2.0
                    distances = np.min(
                        [
                            np.hypot(centre_rows - cr, centre_cols - cc)
                            for cr, cc in centres
                        ],
                        axis=0,
                    )
                    probabilities = distances**2 + 1e-6
                else:
                    probabilities = np.ones(rows.shape, dtype=np.float64)
                probabilities = probabilities / probabilities.sum()
                picked = int(
                    np.random.default_rng(
                        seed + block_index * 7919 + height * 31 + width
                    ).choice(rows.size, p=probabilities)
                )
                placement = (int(rows[picked]), int(cols[picked]))
                break
            # Shrink the longer side first so blocks stay as square as possible.
            if height >= width and height > 1:
                height -= 1
            elif width > 1:
                width -= 1
            elif segment is not None:
                # Exhausted every size inside this segment: release the
                # constraint once, at the smallest size, rather than clumping.
                segment = None
                height, width = block_h, block_w
                continue
            else:
                break
            shrink_steps += 1
        if placement is None:
            failures += 1
            continue
        top, left = placement
        indices = generator._block_to_indices(top, left, height, width)
        blocks.append({"top": top, "left": left, "h": height, "w": width})
        union.update(indices)
        claimed[top : top + height, left : left + width] = 1.0
        centres.append((top + height / 2.0, left + width / 2.0))

    flat = occupancy.reshape(-1)
    target = flat[sorted(union)] if union else np.zeros(0, dtype=np.float32)
    return {
        "blocks": blocks,
        "union": union,
        "unique": len(union),
        "on_region": float((target >= MIN_OCCUPANCY).mean()) if target.size else 0.0,
        "mean_occupancy": float(target.mean()) if target.size else 0.0,
        "shrink_steps": shrink_steps,
        "failures": failures,
    }


def sample_center_anchored(
    generator,
    occupancy,
    seed,
    avoid_overlap=True,
    lateral_frac=1.0,
):
    """Keep the original block size, but anchor each block centre on the region.

    A patch is drawn from the MIRAGE envelope with probability proportional to
    its occupancy, and the block is centred there.  Blocks therefore always sit
    *on* retina even though they are far larger than the band, so target size,
    count and total masked area stay identical to the random and oracle arms.

    ``lateral_frac`` restricts admissible centres to that fraction of the
    lateral extent, mirroring the oracle's ``oracle_lateral_frac``.  Without it
    four full-size blocks centred on a thin band cover essentially the whole
    retina, leaving the encoder no retinal context to reason from.
    """
    generator._size_gen.manual_seed(seed)
    sizes = [
        generator._sample_block_size(generator.pred_mask_scale, generator._size_gen)
        for _ in range(generator.npred)
    ]
    torch.manual_seed(seed)
    random.seed(seed)

    height_grid, width_grid = generator.height, generator.width
    weights = np.asarray(occupancy, dtype=np.float64).copy()
    if lateral_frac < 1.0:
        keep = max(1, int(round(lateral_frac * width_grid)))
        start = (width_grid - keep) // 2
        lateral = np.zeros(width_grid, dtype=bool)
        lateral[start : start + keep] = True
        weights[:, ~lateral] = 0.0
    weights = weights.reshape(-1)
    if weights.sum() <= 0:
        weights = np.asarray(occupancy, dtype=np.float64).reshape(-1).copy()
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    rng = np.random.default_rng(seed)

    blocks, union, centres_on_region = [], set(), 0
    claimed = np.zeros(weights.shape, dtype=bool)
    for block_index, (block_h, block_w) in enumerate(sizes):
        probabilities = weights.copy()
        if avoid_overlap and not claimed.all():
            probabilities[claimed] = 0.0
            if probabilities.sum() <= 0:
                probabilities = weights.copy()
        probabilities = probabilities / probabilities.sum()
        centre = int(rng.choice(probabilities.size, p=probabilities))
        centre_row, centre_col = centre // width_grid, centre % width_grid
        if occupancy.reshape(-1)[centre] > 0:
            centres_on_region += 1
        top = int(np.clip(centre_row - block_h // 2, 0, height_grid - block_h))
        left = int(np.clip(centre_col - block_w // 2, 0, width_grid - block_w))
        indices = generator._block_to_indices(top, left, block_h, block_w)
        blocks.append(
            {
                "top": top,
                "left": left,
                "h": block_h,
                "w": block_w,
                "centre": (centre_row, centre_col),
            }
        )
        union.update(indices)
        claimed[indices] = True

    flat = occupancy.reshape(-1)
    target = flat[sorted(union)]
    return {
        "blocks": blocks,
        "union": union,
        "unique": len(union),
        "on_region": float((target >= MIN_OCCUPANCY).mean()),
        "mean_occupancy": float(target.mean()),
        "shrink_steps": 0,
        "failures": 0,
        "centres_on_region": centres_on_region,
    }


def _place_centres(generator, occupancy, seed, sizes, avoid_overlap, lateral_frac):
    """Draw one centre-anchored placement.  Returns a list of block dicts."""
    height_grid, width_grid = generator.height, generator.width
    weights = np.asarray(occupancy, dtype=np.float64).copy()
    if lateral_frac < 1.0:
        keep = max(1, int(round(lateral_frac * width_grid)))
        start = (width_grid - keep) // 2
        lateral = np.zeros(width_grid, dtype=bool)
        lateral[start : start + keep] = True
        weights[:, ~lateral] = 0.0
    weights = weights.reshape(-1)
    if weights.sum() <= 0:
        weights = np.asarray(occupancy, dtype=np.float64).reshape(-1).copy()
    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    rng = np.random.default_rng(seed)
    blocks, claimed = [], np.zeros(weights.shape, dtype=bool)
    for block_h, block_w in sizes:
        probabilities = weights.copy()
        if avoid_overlap and not claimed.all():
            probabilities[claimed] = 0.0
            if probabilities.sum() <= 0:
                probabilities = weights.copy()
        probabilities = probabilities / probabilities.sum()
        centre = int(rng.choice(probabilities.size, p=probabilities))
        centre_row, centre_col = centre // width_grid, centre % width_grid
        top = int(np.clip(centre_row - block_h // 2, 0, height_grid - block_h))
        left = int(np.clip(centre_col - block_w // 2, 0, width_grid - block_w))
        blocks.append({"top": top, "left": left, "h": block_h, "w": block_w,
                       "centre": (centre_row, centre_col)})
        claimed[generator._block_to_indices(top, left, block_h, block_w)] = True
    return blocks


def _coverage(blocks, shape):
    """How many blocks cover each grid cell."""
    count = np.zeros(shape, dtype=np.int16)
    for block in blocks:
        count[block["top"]:block["top"] + block["h"],
              block["left"]:block["left"] + block["w"]] += 1
    return count


def _trim_to_visibility(blocks, retina, shape, min_visible, min_side,
                        min_patches):
    """Peel outer rows/columns off blocks until enough retina is visible.

    Greedy: each step removes the single edge line that frees the most retina
    cells.  Only cells covered by exactly one block can be freed, so the
    coverage count decides what a trim actually buys.  Rectangularity is
    preserved, which keeps the targets shaped like every other arm's.

    ``min_patches`` is the hard floor on block area and it matters far more
    than it looks.  The I-JEPA collator truncates *every* target block in the
    batch to the smallest one, so a single over-trimmed block silently discards
    content from all 256 blocks in a batch of 64.  Measured without this floor:
    one block fell to 16 patches and dragged the batch from 35 down to 16, a
    61% loss.  The floor is therefore set at the smallest block the unguided
    sampler could produce, so trimming can never make the batch worse than the
    baseline already is.
    """
    retina_total = int(retina.sum())
    if retina_total == 0:
        return blocks, 0
    steps = 0
    while True:
        count = _coverage(blocks, shape)
        visible = int((retina & (count == 0)).sum())
        if visible / retina_total >= min_visible:
            return blocks, steps

        best = None
        for index, block in enumerate(blocks):
            top, left, h, w = block["top"], block["left"], block["h"], block["w"]
            edges = []
            if h > min_side and (h - 1) * w >= min_patches:
                edges.append(("top", (slice(top, top + 1), slice(left, left + w))))
                edges.append(("bottom",
                              (slice(top + h - 1, top + h), slice(left, left + w))))
            if w > min_side and h * (w - 1) >= min_patches:
                edges.append(("left", (slice(top, top + h), slice(left, left + 1))))
                edges.append(("right",
                              (slice(top, top + h), slice(left + w - 1, left + w))))
            for side, window in edges:
                sole = (count[window] == 1) & retina[window]
                gain = int(sole.sum())
                cost = int(count[window].size)
                key = (gain, gain / max(cost, 1))
                if best is None or key > best[0]:
                    best = (key, index, side)

        if best is None:      # every block is at its floor; accept what we have
            return blocks, steps
        _, index, side = best
        block = blocks[index]
        if side == "top":
            block["top"] += 1
            block["h"] -= 1
        elif side == "bottom":
            block["h"] -= 1
        elif side == "left":
            block["left"] += 1
            block["w"] -= 1
        else:
            block["w"] -= 1
        steps += 1


def sample_center_anchored_constrained(
    generator,
    occupancy,
    seed,
    min_retina_visible=0.25,
    max_attempts=30,
    min_side=2,
    min_patches=None,
    avoid_overlap=True,
    lateral_frac=1.0,
):
    """Centre-anchored placement with a hard floor on retinal context.

    Plain centre-anchoring puts every block centre on retina, which gives the
    best purity of any full-size policy but masks ~90% of the retina and leaves
    the encoder nothing anatomical to predict from.  This variant enforces a
    floor in two stages:

    1. **Retry.**  Redraw the whole placement up to ``max_attempts`` times and
       accept the first that leaves at least ``min_retina_visible`` of the
       retina visible.  Sizes are drawn once and held fixed so retries change
       only location, never block size.
    2. **Forced trim.**  If every draw fails, peel outer rows and columns off
       the blocks until the floor is met, never letting a block fall below
       ``min_patches``.

    ``min_patches`` defaults to the smallest block the unguided sampler could
    produce, ``pred_mask_scale[0] * num_patches``.  The floor exists because the
    collator truncates every block in the batch to the smallest one; see
    ``_trim_to_visibility``.

    Where the floor cannot be reached without breaching ``min_patches``, the
    result is returned with ``floor_met=False`` rather than being forced.
    """
    generator._size_gen.manual_seed(seed)
    sizes = [
        generator._sample_block_size(generator.pred_mask_scale, generator._size_gen)
        for _ in range(generator.npred)
    ]
    torch.manual_seed(seed)
    random.seed(seed)

    if min_patches is None:
        min_patches = int(
            round(generator.pred_mask_scale[0] * generator.height * generator.width)
        )

    shape = (generator.height, generator.width)
    retina = np.asarray(occupancy) >= MIN_OCCUPANCY
    retina_total = int(retina.sum())

    attempts_used, chosen, trim_steps = 0, None, 0
    for attempt in range(max_attempts):
        attempts_used = attempt + 1
        blocks = _place_centres(
            generator, occupancy, seed + attempt * 7919, sizes,
            avoid_overlap, lateral_frac,
        )
        if retina_total == 0:
            chosen = blocks
            break
        visible = int((retina & (_coverage(blocks, shape) == 0)).sum())
        if visible / retina_total >= min_retina_visible:
            chosen = blocks
            break
    else:
        chosen, trim_steps = _trim_to_visibility(
            blocks, retina, shape, min_retina_visible, min_side, min_patches
        )

    union = set()
    centres_on_region = 0
    flat_occ = np.asarray(occupancy).reshape(-1)
    for block in chosen:
        union.update(
            generator._block_to_indices(
                block["top"], block["left"], block["h"], block["w"]
            )
        )
        centre_row, centre_col = block["centre"]
        if flat_occ[centre_row * generator.width + centre_col] > 0:
            centres_on_region += 1

    target = flat_occ[sorted(union)] if union else np.zeros(0)
    visible = int((retina & (_coverage(chosen, shape) == 0)).sum())
    visible_frac = (visible / retina_total) if retina_total else 1.0
    return {
        "blocks": chosen,
        "union": union,
        "unique": len(union),
        "on_region": float((target >= MIN_OCCUPANCY).mean()) if target.size else 0.0,
        "mean_occupancy": float(target.mean()) if target.size else 0.0,
        "shrink_steps": trim_steps,
        "trim_steps": trim_steps,
        "trimmed": bool(trim_steps),
        "attempts": attempts_used,
        "retina_visible": visible_frac,
        "floor_met": bool(retina_total == 0 or visible_frac >= min_retina_visible),
        "smallest_block": min(b["h"] * b["w"] for b in chosen),
        "failures": 0,
        "centres_on_region": centres_on_region,
    }


def _context_from_block(generator, block, union):
    """Context indices for a fixed raw context block after target removal."""
    indices = generator._block_to_indices(
        block["top"], block["left"], block["h"], block["w"]
    )
    return [i for i in indices if i not in union]


def sample_mixed_guided(
    generator,
    occupancy,
    seed,
    n_guided=2,
    min_keep_frac=0.20,
    max_attempts=30,
    avoid_overlap=True,
):
    """Guide only some target blocks; leave the rest unbiased.

    Rationale: four full-size blocks centred on retina cannot avoid blanketing
    it -- four blocks of ~44 patches against a retina of ~55-70 leaves nothing
    over.  Guiding only ``n_guided`` of them keeps some targets on anatomy while
    the unbiased ones are free to land off it, which is what actually preserves
    retinal context.

    Two details make this a genuine location-only change:

    * block sizes are drawn once and never altered -- no trimming;
    * the raw context block is drawn once and held fixed across retries, so
      retries cannot shop for a favourable context crop.

    Acceptance uses the *true* visible-retina fraction

        K = |G and C_raw minus T| / |G|

    not ``|G minus T| / |G|``.  The context is itself a sampled block covering
    85-100% of the frame, so retina lying outside it is invisible to the encoder
    no matter what the targets do; scoring without the intersection overstates
    preserved context.

    A candidate is chosen uniformly at random among those meeting the floor,
    rather than taking the maximum, so the policy does not drift toward
    degenerate placements that maximise K by stacking the blocks.
    """
    generator._size_gen.manual_seed(seed)
    sizes = [
        generator._sample_block_size(generator.pred_mask_scale, generator._size_gen)
        for _ in range(generator.npred)
    ]
    generator._size_gen.manual_seed(seed + 999)
    context_h, context_w = generator._sample_block_size(
        generator.enc_mask_scale, generator._size_gen
    )
    random.seed(seed + 999)
    context_top, context_left = generator._sample_uniform_location(
        context_h, context_w, generator.height, generator.width
    )
    context_block = {"top": context_top, "left": context_left,
                     "h": context_h, "w": context_w}
    context_all = set(
        generator._block_to_indices(context_top, context_left, context_h, context_w)
    )

    torch.manual_seed(seed)
    random.seed(seed)

    height_grid, width_grid = generator.height, generator.width
    retina = np.asarray(occupancy) >= MIN_OCCUPANCY
    retina_flat = retina.reshape(-1)
    retina_total = int(retina_flat.sum())
    weights = np.asarray(occupancy, dtype=np.float64).reshape(-1).copy()
    guided_possible = weights.sum() > 0

    candidates = []
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt * 7919)
        random.seed(seed + attempt * 7919)
        blocks, union, claimed = [], set(), np.zeros(weights.size, dtype=bool)
        for index, (block_h, block_w) in enumerate(sizes):
            if index < n_guided and guided_possible:
                probabilities = weights.copy()
                if avoid_overlap and not claimed.all():
                    probabilities[claimed] = 0.0
                    if probabilities.sum() <= 0:
                        probabilities = weights.copy()
                probabilities = probabilities / probabilities.sum()
                centre = int(rng.choice(probabilities.size, p=probabilities))
                top = int(np.clip(centre // width_grid - block_h // 2,
                                  0, height_grid - block_h))
                left = int(np.clip(centre % width_grid - block_w // 2,
                                   0, width_grid - block_w))
            else:
                top, left = generator._sample_uniform_location(
                    block_h, block_w, height_grid, width_grid
                )
            indices = generator._block_to_indices(top, left, block_h, block_w)
            blocks.append({"top": top, "left": left, "h": block_h, "w": block_w,
                           "centre": (top + block_h // 2, left + block_w // 2),
                           "guided": index < n_guided and guided_possible})
            union.update(indices)
            claimed[indices] = True

        kept = context_all - union
        if len(kept) < generator.min_keep:
            continue
        keep_frac = (
            len(retina_visible_set(retina_flat, kept)) / retina_total
            if retina_total else 1.0
        )
        candidates.append((keep_frac, blocks, union, kept))

    if not candidates:
        return None

    passing = [c for c in candidates if c[0] >= min_keep_frac]
    picker = random.Random(seed)
    if passing:
        keep_frac, blocks, union, kept = picker.choice(passing)
        floor_met = True
    else:
        keep_frac, blocks, union, kept = max(candidates, key=lambda c: c[0])
        floor_met = retina_total == 0

    flat_occ = np.asarray(occupancy).reshape(-1)
    target = flat_occ[sorted(union)] if union else np.zeros(0)
    naive = (
        float((retina_flat & ~_as_mask(union, retina_flat.size)).sum()) / retina_total
        if retina_total else 1.0
    )
    return {
        "blocks": blocks,
        "union": union,
        "context": kept,
        "context_block": context_block,
        "unique": len(union),
        "on_region": float((target >= MIN_OCCUPANCY).mean()) if target.size else 0.0,
        "mean_occupancy": float(target.mean()) if target.size else 0.0,
        "retina_visible": keep_frac,          # true: intersected with context
        "retina_unmasked": naive,             # naive: ignores the context crop
        "floor_met": floor_met,
        "candidates": len(candidates),
        "passing": len(passing),
        "smallest_block": min(b["h"] * b["w"] for b in blocks),
        "shrink_steps": 0,
        "trim_steps": 0,
        "trimmed": False,
        "attempts": len(candidates),
        "failures": 0,
    }


def _as_mask(indices, size):
    mask = np.zeros(size, dtype=bool)
    if indices:
        mask[sorted(indices)] = True
    return mask


def retina_visible_set(retina_flat, kept):
    """Retina indices that survive into the context."""
    return {i for i in kept if retina_flat[i]}


def sample_unconstrained(generator, occupancy, seed):
    generator._size_gen.manual_seed(seed)
    sizes = [
        generator._sample_block_size(generator.pred_mask_scale, generator._size_gen)
        for _ in range(generator.npred)
    ]
    torch.manual_seed(seed)
    random.seed(seed)
    grid = torch.from_numpy(occupancy.astype(np.float32))
    blocks, union = [], set()
    for block_h, block_w in sizes:
        top, left = generator._sample_biased_location(block_h, block_w, grid)
        blocks.append({"top": top, "left": left, "h": block_h, "w": block_w})
        union.update(generator._block_to_indices(top, left, block_h, block_w))
    flat = occupancy.reshape(-1)
    target = flat[sorted(union)]
    return {
        "blocks": blocks,
        "union": union,
        "unique": len(union),
        "on_region": float((target >= MIN_OCCUPANCY).mean()),
        "mean_occupancy": float(target.mean()),
        "shrink_steps": 0,
        "failures": 0,
    }


def context_patches(generator, union, seed):
    generator._size_gen.manual_seed(seed + 999)
    block_h, block_w = generator._sample_block_size(
        generator.enc_mask_scale, generator._size_gen
    )
    random.seed(seed + 999)
    for _attempt in range(50):
        top, left = generator._sample_uniform_location(
            block_h, block_w, generator.height, generator.width
        )
        indices = generator._block_to_indices(top, left, block_h, block_w)
        kept = [i for i in indices if i not in union]
        if len(kept) >= generator.min_keep:
            return kept, {"top": top, "left": left, "h": block_h, "w": block_w}
    kept = [i for i in range(generator.num_patches) if i not in union]
    return kept, {"top": 0, "left": 0, "h": generator.height, "w": generator.width}


def render(slice_256, region_grid, sample, context, context_block=None):
    base = np.repeat(slice_256[..., None].astype(np.float32), 3, axis=2)
    region_pixels = np.kron(
        region_grid >= MIN_OCCUPANCY, np.ones((PATCH, PATCH), dtype=bool)
    )
    tint = np.array([40.0, 220.0, 90.0], dtype=np.float32)
    alpha = 0.28 * region_pixels[..., None].astype(np.float32)
    blended = base * (1.0 - alpha) + tint * alpha

    visible = np.zeros((GRID, GRID), dtype=bool)
    if context_block is not None:
        visible[
            context_block["top"] : context_block["top"] + context_block["h"],
            context_block["left"] : context_block["left"] + context_block["w"],
        ] = True
    else:
        for index in context:
            visible[index // GRID, index % GRID] = True
    # Dim only what lies OUTSIDE the encoder's context block.  Target patches
    # are excluded from the context by design, so dimming them too would make
    # every target look like it landed in dead space.
    blended[~np.kron(visible, np.ones((PATCH, PATCH), dtype=bool))] *= 0.5

    image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    if context_block is not None:
        x0, y0 = context_block["left"] * PATCH, context_block["top"] * PATCH
        draw.rectangle(
            [x0, y0, x0 + context_block["w"] * PATCH - 1,
             y0 + context_block["h"] * PATCH - 1],
            outline=(0, 220, 255),
            width=2,
        )
        draw.text((x0 + 4, y0 + 2), "encoder context block", fill=(0, 220, 255))
    for block in sample["blocks"]:
        x0, y0 = block["left"] * PATCH, block["top"] * PATCH
        draw.rectangle(
            [x0, y0, x0 + block["w"] * PATCH - 1, y0 + block["h"] * PATCH - 1],
            fill=(255, 60, 60, 110),
            outline=(255, 60, 60),
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", type=int, default=3)
    parser.add_argument("--slices", type=int, nargs="+", default=[0, 50, 99])
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    generators = {name: make_generator(scale) for name, scale, _ in VARIANTS}
    retina_stats: dict = {name: [] for name, _s, _m in VARIANTS}
    records = []

    for mask_path in sorted(MASK_DIR.glob("data_*.npz"))[: args.volumes]:
        with np.load(mask_path, allow_pickle=False) as cache:
            masks = cache["hard_masks"]
            slice_indices = cache["slice_indices"].astype(int)
            label = int(cache["glaucoma"])
        with np.load(DATA_DIR / mask_path.name, allow_pickle=False) as data:
            volume = data["oct_bscans"]

        rows = []
        for cache_index in args.slices:
            slice_index = int(slice_indices[cache_index])
            slice_256 = np.asarray(
                Image.fromarray(volume[slice_index], mode="L").resize(
                    (CROP, CROP), Image.BILINEAR
                )
            )
            envelope, _v, _s = repair_union(
                build_union(masks[cache_index]), params=DEFAULT_REPAIR
            )
            grown, achieved = expand_envelope(envelope, EXPANSION)
            grown_256 = (
                np.asarray(
                    Image.fromarray(grown.astype(np.uint8) * 255, mode="L").resize(
                        (CROP, CROP), Image.Resampling.NEAREST
                    )
                )
                > 127
            )
            occupancy = patch_occupancy(grown_256, patch_size=PATCH)
            seed = args.seed + slice_index

            columns = [Image.fromarray(slice_256, mode="L").convert("RGB")]
            stats = []
            for name, _scale, mode in VARIANTS:
                generator = generators[name]
                if mode in ("region", "region8"):
                    if mode == "region8":
                        generator.npred = 8
                    else:
                        generator.npred = 4
                    sample = sample_region_only(generator, occupancy, seed)
                elif mode == "center":
                    sample = sample_center_anchored(generator, occupancy, seed)
                else:
                    sample = sample_unconstrained(generator, occupancy, seed)
                context, context_block = context_patches(
                    generator, sample["union"], seed
                )
                columns.append(
                    render(slice_256, occupancy, sample, context, context_block)
                )
                entry = {
                    "variant": name,
                    "unique_target_patches": sample["unique"],
                    "on_region_frac": round(sample["on_region"], 4),
                    "mean_target_occupancy": round(sample["mean_occupancy"], 4),
                    "context_patches": len(context),
                    "blocks_placed": len(sample["blocks"]),
                    "shrink_steps": sample["shrink_steps"],
                    "failures": sample["failures"],
                }
                retina = set(
                    np.flatnonzero(occupancy.reshape(-1) >= MIN_OCCUPANCY).tolist()
                )
                if retina:
                    entry["retina_visible"] = round(
                        len(retina & set(context)) / len(retina), 4
                    )
                    entry["retina_masked"] = round(
                        len(retina & sample["union"]) / len(retina), 4
                    )
                    retina_stats[name].append(entry["retina_visible"])
                stats.append(entry)
                records.append(
                    {
                        "volume": mask_path.stem,
                        "slice": slice_index,
                        "glaucoma": label,
                        "achieved_expansion": round(achieved, 4),
                        "region_cells": int((occupancy >= MIN_OCCUPANCY).sum()),
                        **entry,
                    }
                )
            rows.append({"slice": slice_index, "columns": columns, "stats": stats})

        titles = ["Original OCT"] + [name for name, _s, _c in VARIANTS]
        panel = Image.new(
            "RGB",
            (ROW_LABEL + TILE * len(titles), HEADER * 2 + TILE * len(rows)),
            "white",
        )
        draw = ImageDraw.Draw(panel)
        draw.text(
            (6, 5),
            f"{mask_path.stem} glaucoma={label}   envelope +5%;  green=MIRAGE region, "
            "red=target blocks, cyan=encoder context block, "
            "dark=patches the encoder never sees (standard I-JEPA)",
            fill="black",
        )
        for column, name in enumerate(titles):
            draw.text((ROW_LABEL + column * TILE + 4, HEADER + 3), name, fill="black")
        for index, row in enumerate(rows):
            top = HEADER * 2 + index * TILE
            draw.text((4, top + 6), f"slice {row['slice']}", fill="black")
            for column, image in enumerate(row["columns"]):
                panel.paste(image, (ROW_LABEL + column * TILE, top))
            for column, entry in enumerate(row["stats"]):
                x = ROW_LABEL + (column + 1) * TILE + 4
                draw.text(
                    (x, top + TILE - 32),
                    f"in-region {entry['on_region_frac'] * 100:.0f}%",
                    fill=(255, 240, 120),
                )
                draw.text(
                    (x, top + TILE - 18),
                    f"retina left for encoder {entry.get('retina_visible', 0) * 100:.0f}%",
                    fill=(120, 255, 160)
                    if entry.get("retina_visible", 0) >= 0.25
                    else (255, 110, 110),
                )
        panel.save(args.output / f"{mask_path.stem}_region_only.png", optimize=True)

    summary = {}
    for name, _scale, _c in VARIANTS:
        subset = [r for r in records if r["variant"] == name]
        summary[name] = {
            "mean_unique_target_patches": round(
                float(np.mean([r["unique_target_patches"] for r in subset])), 2
            ),
            "mean_target_area_frac": round(
                float(np.mean([r["unique_target_patches"] for r in subset]) / 256), 4
            ),
            "mean_in_region_frac": round(
                float(np.mean([r["on_region_frac"] for r in subset])), 4
            ),
            "mean_target_occupancy": round(
                float(np.mean([r["mean_target_occupancy"] for r in subset])), 4
            ),
            "mean_context_patches": round(
                float(np.mean([r["context_patches"] for r in subset])), 2
            ),
            "blocks_placed": round(
                float(np.mean([r["blocks_placed"] for r in subset])), 2
            ),
            "placement_failures": int(sum(r["failures"] for r in subset)),
        }
    (args.output / "summary.json").write_text(
        json.dumps({"expansion": EXPANSION, "min_occupancy": MIN_OCCUPANCY,
                    "summary": summary, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
