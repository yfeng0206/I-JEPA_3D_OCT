"""MIRAGE-derived repaired retinal envelope for I-JEPA target-block biasing.

The MIRAGE-Large GOALS segmentation head labels every OCT B-scan pixel as one
of four classes::

    0 = Elsewhere   1 = RNFL   2 = GCIPL   3 = Choroid

The *union* of RNFL, GCIPL and choroid is the retinal-importance signal used to
bias I-JEPA target-block placement.  Taken raw, that union is **hollow**: GOALS
does not label the middle retina (INL/OPL/ONL/photoreceptors/RPE), so those rows
fall into ``Elsewhere`` and split the band into an upper (RNFL+GCIPL) and a lower
(choroid) ribbon.  ``repair_union`` closes that gap so a sampled target rectangle
lands on retina rather than straddling a labelling artefact.

Because the repair fills the tissue lying *between* MIRAGE's upper and lower
retinal boundaries, the result is strictly larger than the three predicted
layers.  The scientifically accurate name for the output is therefore the
**MIRAGE-derived repaired retinal envelope**, not simply "the MIRAGE union";
``build_union`` returns the raw union and ``repair_union`` returns the envelope.

Design constraints (see docs/experiments/curriculum_masking.md):

* Repair runs in the **native 200x200 label space**, before any resize or crop,
  so it is deterministic per slice and can be precomputed once.
* All *substantial* components are retained.  Around the optic nerve head the
  band legitimately separates into two large pieces; keeping only the largest
  would delete real anatomy in the most glaucoma-relevant region.  Left and
  right retinal sections are repaired independently and never joined.
* Only short, interior, boundary-compatible horizontal gaps are bridged.
  Distant isolated detections are never connected.
* The module never invents anatomy when the segmentation has failed: it reports
  ``valid=False`` and the caller falls back to uniform random placement.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

CLASS_ELSEWHERE = 0
CLASS_RNFL = 1
CLASS_GCIPL = 2
CLASS_CHOROID = 3

UNION_CLASSES: Tuple[int, ...] = (CLASS_RNFL, CLASS_GCIPL, CLASS_CHOROID)

CLASS_NAMES: Tuple[str, ...] = ("Elsewhere", "RNFL", "GCIPL", "Choroid")


@dataclass(frozen=True)
class RepairParams:
    """Frozen repair thresholds.

    Components are kept when their area clears both an absolute floor and a
    fraction of the largest component, so the two large pieces either side of
    the optic nerve head both survive while scattered speckle -- which can
    chain diagonally into a wide but nearly massless blob -- is discarded.

    ``component_rule`` selects which detections survive:

    ``"small_and_short"``
        Drop a component only when it is *both* under ``min_component_area``
        and under ``min_component_span`` columns wide.  Conservative: keeps
        genuine but fragmented pieces of the band, so the envelope still has
        an upper and a lower boundary to fill between.
    ``"area_ratio"``
        Additionally drop anything below ``min_component_area_ratio`` of the
        largest component.  Aggressive: removes speckle, but can delete real
        band fragments and leave the envelope unfilled.

    ``envelope_mode`` selects how each column is closed:

    ``"column_gap"``
        Close only gaps up to ``max_column_gap`` rows.  Conservative: a
        column never reaches down to a far-away detection.
    ``"section"``
        Close every column from its uppermost to its lowermost retained pixel.
        Yields the fullest retinal envelope and relies on component filtering
        to have already removed stray detections.

    Every threshold is recorded in the run config so the guide is reproducible.
    """

    min_component_area: int = 50
    min_component_span: int = 20
    min_component_area_ratio: float = 0.10
    component_rule: str = "small_and_short"
    envelope_mode: str = "column_gap"
    max_column_gap: int = 45
    max_hole_area: int = 200
    min_fill_width: int = 5
    max_horizontal_gap: int = 12
    max_boundary_jump: int = 12
    min_valid_area: float = 0.05
    min_valid_span: float = 0.40

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_REPAIR = RepairParams()


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected component labelling, preferring scipy when available."""
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - exercised only without scipy
        return _label_components_numpy(mask)
    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndimage.label(mask, structure=structure)
    return labels.astype(np.int32), int(count)


def _label_components_numpy(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Union-find fallback so the guide never hard-depends on scipy."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent: list = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for row in range(height):
        for col in range(width):
            if not mask[row, col]:
                continue
            neighbours = []
            for d_row, d_col in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
                n_row, n_col = row + d_row, col + d_col
                if 0 <= n_row < height and 0 <= n_col < width:
                    if labels[n_row, n_col]:
                        neighbours.append(int(labels[n_row, n_col]))
            if not neighbours:
                parent.append(len(parent))
                labels[row, col] = len(parent) - 1
            else:
                smallest = min(neighbours)
                labels[row, col] = smallest
                for other in neighbours:
                    union(smallest, other)

    remap: Dict[int, int] = {}
    for row in range(height):
        for col in range(width):
            if labels[row, col]:
                root = find(int(labels[row, col]))
                if root not in remap:
                    remap[root] = len(remap) + 1
                labels[row, col] = remap[root]
    return labels, len(remap)


def _horizontal_opening(mask: np.ndarray, width: int) -> np.ndarray:
    """Binary opening with a 1 x ``width`` horizontal structuring element.

    Used to strip thin vertical spikes out of *filled* regions: a genuine
    retinal fill is horizontally extensive, whereas a column that reached down
    to an isolated detection leaves a one- or two-pixel-wide bar.
    """
    if width <= 1:
        return mask
    try:
        from scipy import ndimage

        structure = np.ones((1, width), dtype=bool)
        return ndimage.binary_opening(mask, structure=structure)
    except ImportError:  # pragma: no cover - exercised only without scipy
        eroded = mask.copy()
        for shift in range(1, width):
            shifted = np.zeros_like(mask)
            shifted[:, : mask.shape[1] - shift] = mask[:, shift:]
            eroded &= shifted
        dilated = eroded.copy()
        for shift in range(1, width):
            shifted = np.zeros_like(eroded)
            shifted[:, shift:] = eroded[:, : eroded.shape[1] - shift]
            dilated |= shifted
        return dilated


def _fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    """Fill enclosed background pockets no larger than ``max_hole_area``."""
    if max_hole_area <= 0:
        return mask
    background_labels, count = _label_components(~mask)
    if count == 0:
        return mask
    filled = mask.copy()
    border = set(background_labels[0, :].tolist())
    border.update(background_labels[-1, :].tolist())
    border.update(background_labels[:, 0].tolist())
    border.update(background_labels[:, -1].tolist())
    areas = np.bincount(background_labels.ravel(), minlength=count + 1)
    for component in range(1, count + 1):
        if component in border:
            continue
        if areas[component] <= max_hole_area:
            filled[background_labels == component] = True
    return filled


def _column_bounds(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-column occupancy plus first/last foreground row indices."""
    occupied = mask.any(axis=0)
    rows = np.arange(mask.shape[0])[:, None]
    masked_rows = np.where(mask, rows, np.iinfo(np.int32).max)
    top = masked_rows.min(axis=0)
    masked_rows_low = np.where(mask, rows, -1)
    bottom = masked_rows_low.max(axis=0)
    return occupied, top, bottom


def build_union(labels: np.ndarray) -> np.ndarray:
    """Binary RNFL+GCIPL+choroid union from a hard MIRAGE label map."""
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label map, got shape {labels.shape}")
    return np.isin(labels, UNION_CLASSES)


def repair_union(
    union: np.ndarray,
    params: RepairParams = DEFAULT_REPAIR,
) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    """Repair a raw retinal union into a coherent envelope.

    Args:
        union: (H, W) boolean raw union.
        params: frozen repair thresholds.

    Returns:
        ``(repaired, valid, stats)`` where ``repaired`` is a boolean (H, W)
        envelope, ``valid`` reports whether the guide passed quality control,
        and ``stats`` carries bounded diagnostics for logging.
    """
    if union.ndim != 2:
        raise ValueError(f"Expected a 2D union, got shape {union.shape}")
    if params.envelope_mode not in ("column_gap", "section"):
        raise ValueError(
            f"envelope_mode must be 'column_gap' or 'section', "
            f"got {params.envelope_mode!r}"
        )
    if params.component_rule not in ("small_and_short", "area_ratio"):
        raise ValueError(
            f"component_rule must be 'small_and_short' or 'area_ratio', "
            f"got {params.component_rule!r}"
        )
    union = np.asarray(union, dtype=bool)
    height, width = union.shape
    raw_area_frac = float(union.mean())

    labels, count = _label_components(union)
    kept = np.zeros_like(union)
    dropped_components = 0
    kept_components = 0
    component_areas = np.bincount(labels.ravel(), minlength=count + 1)
    largest_area = int(component_areas[1:].max()) if count else 0
    ratio_threshold = int(round(params.min_component_area_ratio * largest_area))
    for component in range(1, count + 1):
        component_mask = labels == component
        area = int(component_areas[component])
        columns = np.flatnonzero(component_mask.any(axis=0))
        span = int(columns[-1] - columns[0] + 1) if columns.size else 0
        # Drop only speckle: too small AND too narrow to be part of the band.
        drop = area < params.min_component_area and span < params.min_component_span
        if params.component_rule == "area_ratio":
            drop = drop or area < ratio_threshold
        if drop:
            dropped_components += 1
        else:
            kept |= component_mask
            kept_components += 1

    # Vertical envelope: GOALS leaves the middle retina unlabelled, so each
    # column is closed between its own retained pixels.  Columns inside the
    # optic-nerve-head gap hold no foreground at all, so the left and right
    # retinal sections are never joined by this step.
    envelope = kept.copy()
    long_gaps_skipped = 0
    column_gap_limit = (
        height if params.envelope_mode == "section" else params.max_column_gap
    )
    for col in np.flatnonzero(kept.any(axis=0)):
        rows = np.flatnonzero(kept[:, col])
        gaps = np.diff(rows) - 1
        for position, gap in enumerate(gaps):
            if gap <= 0:
                continue
            if gap > column_gap_limit:
                long_gaps_skipped += 1
                continue
            envelope[rows[position] + 1 : rows[position + 1], col] = True

    # Vertical fills must be band-like.  Opening removes narrow spikes from the
    # filled region; unioning the retained detections back guarantees the step
    # only ever deletes synthesised pixels, never MIRAGE's own output.
    spike_pixels = 0
    if params.min_fill_width > 1:
        opened = _horizontal_opening(envelope, params.min_fill_width) | kept
        spike_pixels = int(envelope.sum() - opened.sum())
        envelope = opened

    envelope = _fill_small_holes(envelope, params.max_hole_area)

    # Horizontal bridging: only short, interior gaps whose two sides agree on
    # where the retina is. Everything else stays disconnected on purpose.
    occupied, top, bottom = _column_bounds(envelope)
    bridged_gaps = 0
    skipped_gaps = 0
    empty = ~occupied
    col = 0
    while col < width:
        if not empty[col]:
            col += 1
            continue
        start = col
        while col < width and empty[col]:
            col += 1
        end = col - 1
        gap_length = end - start + 1
        if start == 0 or end == width - 1:
            skipped_gaps += 1
            continue
        if gap_length > params.max_horizontal_gap:
            skipped_gaps += 1
            continue
        left, right = start - 1, end + 1
        top_jump = abs(int(top[left]) - int(top[right]))
        bottom_jump = abs(int(bottom[left]) - int(bottom[right]))
        if (
            top_jump > params.max_boundary_jump
            or bottom_jump > params.max_boundary_jump
        ):
            skipped_gaps += 1
            continue
        for offset in range(1, gap_length + 1):
            weight = offset / (gap_length + 1)
            upper = int(round(top[left] + weight * (top[right] - top[left])))
            lower = int(round(bottom[left] + weight * (bottom[right] - bottom[left])))
            upper = max(0, min(upper, height - 1))
            lower = max(0, min(lower, height - 1))
            if lower < upper:
                upper, lower = lower, upper
            envelope[upper : lower + 1, start + offset - 1] = True
        bridged_gaps += 1

    repaired_area_frac = float(envelope.mean())
    span_frac = float(envelope.any(axis=0).mean())
    valid = bool(
        repaired_area_frac >= params.min_valid_area
        and span_frac >= params.min_valid_span
    )
    stats = {
        "raw_area_frac": raw_area_frac,
        "repaired_area_frac": repaired_area_frac,
        "added_area_frac": repaired_area_frac - raw_area_frac,
        "span_frac": span_frac,
        "components_kept": float(kept_components),
        "components_dropped": float(dropped_components),
        "gaps_bridged": float(bridged_gaps),
        "gaps_skipped": float(skipped_gaps),
        "column_gaps_skipped": float(long_gaps_skipped),
        "spike_pixels_removed": float(spike_pixels),
        "valid": float(valid),
    }
    return envelope, valid, stats


def repair_labels(
    labels: np.ndarray,
    params: RepairParams = DEFAULT_REPAIR,
) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    """Convenience wrapper: hard MIRAGE labels -> repaired retinal envelope."""
    return repair_union(build_union(labels), params=params)


def patch_occupancy(mask: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """Fraction of each ``patch_size`` cell covered by ``mask``.

    Returns a (H // patch_size, W // patch_size) float32 grid in [0, 1].  This
    is the fractional-occupancy weight grid consumed by the curriculum's
    biased-location sampler -- never a class argmax.
    """
    mask = np.asarray(mask)
    height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Mask {mask.shape} is not divisible by patch size {patch_size}"
        )
    grid_h, grid_w = height // patch_size, width // patch_size
    return (
        mask.astype(np.float32)
        .reshape(grid_h, patch_size, grid_w, patch_size)
        .mean(axis=(1, 3))
    )


def occupancy_is_valid(
    grid: np.ndarray,
    params: RepairParams = DEFAULT_REPAIR,
) -> bool:
    """Quality control on the post-crop grid the model actually sees.

    A guide can be healthy in native coordinates yet nearly empty after a
    RandomResizedCrop lands off the retina, so validity is re-checked here.
    """
    grid = np.asarray(grid, dtype=np.float32)
    area_frac = float(grid.mean())
    span_frac = float((grid.max(axis=0) > 0.0).mean())
    return bool(
        area_frac >= params.min_valid_area and span_frac >= params.min_valid_span
    )


def dilate_patch_grid(grid: np.ndarray, patches: int) -> np.ndarray:
    """Grow a binary patch-grid mask outward by ``patches`` cells.

    Expansion happens on the 16x16 patch grid rather than in pixel space, so the
    tolerance is expressed in the same units the mask sampler works in: one
    patch of slack all the way around the region.  Growth that would fall off
    the grid is simply clipped.

    Unlike area-proportional dilation this adds a constant ring regardless of
    how big the region already is, so a thin retinal band and a thick one get
    the same absolute tolerance for MIRAGE's boundary error.
    """
    grid = np.asarray(grid, dtype=bool)
    if patches <= 0:
        return grid.copy()
    try:
        from scipy import ndimage

        structure = np.ones((3, 3), dtype=bool)
        return ndimage.binary_dilation(grid, structure=structure, iterations=patches)
    except ImportError:  # pragma: no cover - exercised only without scipy
        grown = grid.copy()
        for _step in range(patches):
            padded = np.zeros(
                (grown.shape[0] + 2, grown.shape[1] + 2), dtype=bool
            )
            padded[1:-1, 1:-1] = grown
            neighbourhood = np.zeros_like(grown)
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    neighbourhood |= padded[
                        1 + d_row : 1 + d_row + grown.shape[0],
                        1 + d_col : 1 + d_col + grown.shape[1],
                    ]
            grown = neighbourhood
        return grown


def expand_envelope(
    mask: np.ndarray,
    area_frac: float,
) -> Tuple[np.ndarray, float]:
    """Grow the envelope border outward by ``area_frac`` of its own area.

    The boundary is pushed outward isotropically -- pixels are added in order
    of their euclidean distance to the envelope -- so the retinal band keeps its
    shape and the optic-nerve-head gap stays open until the expansion is large
    enough to close it naturally.

    Args:
        mask: (H, W) boolean repaired envelope.
        area_frac: fractional area increase (``0.10`` grows the mask by 10%).

    Returns:
        ``(expanded, achieved_frac)`` where ``achieved_frac`` is the realised
        fractional increase, which can exceed ``area_frac`` slightly when many
        candidate pixels sit at the same distance.
    """
    mask = np.asarray(mask, dtype=bool)
    if area_frac <= 0.0:
        return mask.copy(), 0.0
    base_area = int(mask.sum())
    if base_area == 0:
        return mask.copy(), 0.0
    wanted = int(round(area_frac * base_area))
    if wanted <= 0:
        return mask.copy(), 0.0

    distance = _distance_outside(mask)
    candidates = distance[distance > 0.0]
    if candidates.size == 0:
        return mask.copy(), 0.0
    if wanted >= candidates.size:
        expanded = np.ones_like(mask)
    else:
        threshold = np.partition(candidates, wanted - 1)[wanted - 1]
        expanded = mask | ((distance > 0.0) & (distance <= threshold))
    achieved = (int(expanded.sum()) - base_area) / float(base_area)
    return expanded, achieved


def _distance_outside(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from every background pixel to the nearest True."""
    try:
        from scipy import ndimage

        return ndimage.distance_transform_edt(~mask)
    except ImportError:  # pragma: no cover - exercised only without scipy
        distance = np.full(mask.shape, np.inf, dtype=np.float32)
        distance[mask] = 0.0
        frontier = mask.copy()
        step = 0
        while not frontier.all():
            step += 1
            grown = frontier.copy()
            grown[1:, :] |= frontier[:-1, :]
            grown[:-1, :] |= frontier[1:, :]
            grown[:, 1:] |= frontier[:, :-1]
            grown[:, :-1] |= frontier[:, 1:]
            newly = grown & ~frontier
            if not newly.any():
                break
            distance[newly] = float(step)
            frontier = grown
        distance[np.isinf(distance)] = float(max(mask.shape))
        return distance


GUIDE_SCHEMA_VERSION = 1


def pack_guides(masks: np.ndarray) -> np.ndarray:
    """Bit-pack a stack of boolean guides for compact on-disk caching."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"Expected (N, H, W) guides, got shape {masks.shape}")
    return np.packbits(masks.reshape(masks.shape[0], -1), axis=1)


def unpack_guides(packed: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`pack_guides`."""
    height, width = shape
    count = packed.shape[0]
    flat = np.unpackbits(packed, axis=1, count=height * width)
    return flat.reshape(count, height, width).astype(bool)


def params_fingerprint(params: RepairParams) -> str:
    """Stable digest of the repair thresholds, stored beside every guide."""
    import hashlib
    import json

    payload = json.dumps(params.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
