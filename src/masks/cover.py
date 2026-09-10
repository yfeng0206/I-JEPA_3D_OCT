"""COVER masking: stock I-JEPA rectangles, greedily placed to hide the anatomy.

Design goal — change **only placement** relative to the ``mirage_envelope``
baseline, so the comparison isolates one variable:

  * shape  : axis-aligned rectangles (NOT the irregular blobs of
             ``mirage_anatomy``)
  * count  : ``npred`` blocks, unchanged
  * size   : drawn from ``pred_mask_scale`` / ``aspect_ratio`` by the caller,
             **once per batch and shared across images**, exactly as
             ``multiblock.MaskCollator`` does

What changes is *where* the blocks go: they are placed greedily so that nearly
all anatomy is hidden, leaving only a small visible remainder.  Overlap between
blocks is allowed, and blocks are allowed to spill onto background.

Rationale.  ``mirage_envelope`` aims rectangles at the retina; ``mirage_anatomy``
hides the right cells but pays for it with irregular, much smaller targets.
COVER was intended to keep the envelope's target geometry and supervision volume
while pushing the hidden fraction of anatomy higher, which is the variable we
actually want to test.

Historical default (``delivered_k=None``): this intent is not realised.
The opt-in ``delivered_k`` path scores and returns the exact prefix-shaped
targets; ``curriculum.cover_algorithm=delivered_v2`` supplies the common K.
Neither placement mode alone guarantees tissue in the FINAL encoder context:
the separately labeled ``cover_context_guard`` checks/repairs that after
collation, or reports invalid/infeasible status.

Historical failure:
The collator truncates every predictor target to the shortest target in the
microbatch (curriculum.py, the ``global_min_pred`` branch), and it does so AFTER
this module has chosen placement.  The greedy optimisation below is therefore
computed on full rectangles and then defeated.  Measured on delivered masks:

    policy            anatomy hidden
    random                 53.1%
    oracle                 61.6%
    cover  f=0.21          73.1%     <- placement achieved 78.6% before truncation
    envelope               77.6%

So COVER as shipped hides LESS anatomy than envelope, inverting its purpose.  Do
not cite this arm as an over-coverage condition.  Note also that realised
coverage is logged BEFORE truncation, so the training logs report the intent
(~78.5%) rather than what the model received.

The opt-in fix scores coverage against the prefix-truncated shapes that
will actually reach the model, and re-logging coverage after collation.  Setting
``pred_target_k`` on this arm is not a fix: it would cut the rectangle arms from
~158 loss slots to 64 and change what they measure.

Greedy placement is *exact* here rather than a heuristic: a 16x16 grid admits at
most 256 candidate windows per block, so every one is evaluated via a
summed-area table instead of rejection-sampled.

Two floors keep the encoder from being handed an impossible task:

  ``min_visible_frac``   fraction of anatomy MASS that must remain visible.
  ``min_visible_cells``  number of anatomy CELLS that must remain visible.

The mass floor alone is not sufficient on sparse edge slices: a slice holding
7 anatomy cells satisfies a 15% mass floor with a single cell, which is too
weak an anchor.  This mirrors the existing ``mirage_min_retina_visible`` guard —
the predictor must not be asked to reconstruct the retina from pure background,
where it can only fall back on a positional prior.
"""
from __future__ import annotations

import random as _random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["build_targets", "is_viable", "anatomy_support"]


# ----------------------------------------------------------------------
# Summed-area helpers
# ----------------------------------------------------------------------
def _integral(grid: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero row/column prepended."""
    padded = np.zeros((grid.shape[0] + 1, grid.shape[1] + 1), dtype=np.float64)
    padded[1:, 1:] = grid
    return padded.cumsum(axis=0).cumsum(axis=1)


def _window_sums(sat: np.ndarray, block_h: int, block_w: int) -> Optional[np.ndarray]:
    """Sum over every ``block_h x block_w`` window, from a summed-area table."""
    n_top = sat.shape[0] - block_h
    n_left = sat.shape[1] - block_w
    if n_top <= 0 or n_left <= 0:
        return None
    return (
        sat[block_h:, block_w:]
        - sat[:n_top, block_w:]
        - sat[block_h:, :n_left]
        + sat[:n_top, :n_left]
    )


def anatomy_support(
    class_scores: Sequence[np.ndarray], tau: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(score, support)`` — summed class scores and the ``> tau`` mask.

    Matches ``src.masks.anatomy``: the per-class soft scores are SUMMED, then
    thresholded at ``tau`` to decide which cells meaningfully carry anatomy.
    """
    score = np.asarray(class_scores[0], dtype=np.float64)
    for extra in class_scores[1:]:
        score = score + np.asarray(extra, dtype=np.float64)
    return score, score > tau


def is_viable(
    class_scores: Sequence[np.ndarray],
    *,
    tau: float = 0.10,
    min_visible_cells: int = 4,
) -> bool:
    """Whether COVER can do anything meaningful on this slice.

    A slice whose anatomy is no larger than the visible-cell floor cannot have
    any of it hidden without breaching the floor, so greedy placement would
    degenerate into random rectangles.  Such slices are routed to the uniform
    fallback explicitly, so they show up in the stats rather than silently
    masquerading as guided images.
    """
    score, support = anatomy_support(class_scores, tau)
    n_anat = int(support.sum())
    if n_anat <= 0 or float(score[support].sum()) <= 0.0:
        return False
    return n_anat > int(min_visible_cells)


# ----------------------------------------------------------------------
# Placement
# ----------------------------------------------------------------------
def build_targets(
    class_scores: Sequence[np.ndarray],
    block_sizes: Sequence[Tuple[int, int]],
    *,
    guided: Optional[Sequence[bool]] = None,
    fixed: Optional[Sequence[Optional[np.ndarray]]] = None,
    leave_frac: float = 0.15,
    min_visible_frac: float = 0.15,
    min_visible_cells: int = 4,
    tau: float = 0.10,
    transition: bool = True,
    fill: Optional[str] = None,
    occupancy: Optional[np.ndarray] = None,
    delivered_k: Optional[int] = None,
    rng=None,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """Greedily place ``len(block_sizes)`` rectangles to hide the anatomy.

    Args:
        class_scores: per-class soft anatomy scores, each ``(H, W)``.  Summed
            before use, matching ``src.masks.anatomy``.
        block_sizes: ``(block_h, block_w)`` per target, **already sampled by
            the caller once per batch** so every image in the batch shares
            them.  Sampling them here instead would break comparability with
            the envelope / stock arms.
        guided: per-block flags from the curriculum ramp.  ``mirage_envelope``
            draws Bernoulli(r_t) **per block**, so COVER must too — a per-image
            draw would make guidance all-or-none and change the correlation
            structure during the ramp, which is a second difference on top of
            placement and would break the comparison.  ``None`` means all
            blocks are guided.
        fixed: the uniformly-placed mask for each non-guided slot, drawn by the
            caller BEFORE this call so retries cannot quietly bias a block the
            ramp designated random.  Non-guided blocks are still folded into
            the coverage bookkeeping, because the floors are properties of the
            final union, not of the guided blocks alone.
        leave_frac: SOFT target — stop spending blocks on coverage once
            ``1 - leave_frac`` of the anatomy mass is hidden.
            NOTE: when ``leave_frac == min_visible_frac`` (the shipped default,
            both 0.15) the soft stop and the hard floor coincide, and the two
            loop conditions ``covered_mass < target_mass`` and
            ``remaining.sum() > floor_mass`` become algebraically identical.
            The policy is then exactly "hide as much anatomy as possible
            without dropping below ``min_visible_frac`` visible", which is the
            intended reading — not an accident.
        min_visible_frac: HARD floor on anatomy mass that must stay visible.
        min_visible_cells: HARD floor on the number of anatomy cells that must
            stay visible.
        transition: place blocks left over after coverage so they straddle the
            tissue/background boundary, instead of dropping them on empty
            vitreous where they carry no signal.
        rng: ``random.Random``-like source.  Defaults to the ``random`` module
            so COVER shares the global seeding of the rest of the sampler.
        delivered_k: Opt-in delivered-v2 geometry. Score and return exactly
            the row-major prefix of each candidate rectangle that collation
            will deliver. Candidate top-left bounds still use the full drawn
            rectangle; ``None`` preserves the historical full-rectangle policy.

    Returns:
        ``(masks, info)`` — ``masks`` is a list of ``(H, W)`` bool arrays, one
        per block, in the same order as ``block_sizes``.  ``info["ok"]`` is
        False when no legal placement existed and a floor had to be breached,
        or when the greedy hid no anatomy at all; the caller must then discard
        the result and use the stock uniform fallback.
    """
    rng = rng if rng is not None else _random
    n_blocks = len(block_sizes)
    if guided is None:
        guided = [True] * n_blocks
    guided = [bool(g) for g in guided]
    if fixed is None:
        fixed = [None] * n_blocks

    score, support = anatomy_support(class_scores, tau)
    height, width = score.shape
    if delivered_k is not None:
        delivered_k = int(delivered_k)
        if delivered_k < 1 or any(
            delivered_k > min(h, height) * min(w, width) for h, w in block_sizes
        ):
            raise ValueError("delivered_k must fit every drawn rectangle")
    n_anat = int(support.sum())
    total_mass = float(score[support].sum()) if n_anat else 0.0

    info: Dict[str, float] = {
        "anat_cells": n_anat,
        "fallback": False,
        "ok": True,
        "floor_violation": False,
        "coverage_needed": True,
        "n_cover": 0,
        "n_transition": 0,
        "n_random": 0,
        "n_unguided": 0,
        "n_boundary_fallback": 0,
        "floor_blocked": 0,
        # Per-slot provenance, so callers can visualise and audit which blocks
        # did coverage work and which were just uniform JEPA blocks.
        "slot_kind": [None] * n_blocks,
    }

    # Backwards compatibility: ``transition`` is the original boolean switch.
    # ``fill`` supersedes it and names the three ways a leftover block (one not
    # needed for coverage) can be spent:
    #   "transition"   straddle the anatomy/background boundary (shipped)
    #   "random"       plain uniform JEPA block, floor NOT enforced
    #   "random_legal" plain uniform block restricted to windows that keep the
    #                  hard anatomy-visibility floor intact
    fill_mode = fill if fill is not None else ("transition" if transition else "random")
    if fill_mode not in ("transition", "random", "random_legal"):
        raise ValueError(
            "fill must be one of 'transition', 'random', 'random_legal'; "
            "got %r" % (fill_mode,)
        )

    # The floors above are defined on the SOFT support (score > tau).  Every
    # audit in this repo, and the arm's stated "leave 15% of the anatomy"
    # guarantee, use the OCCUPANCY mask instead, and the soft support is a
    # strict superset of it.  Enforcing only the soft floor therefore lets the
    # occupancy-visible fraction fall below the promised floor -- measured at
    # 27% of slices (min 10.0%) at the production tau=0.10.  When an occupancy
    # mask is supplied the same two floors are enforced against it as well, so
    # the guarantee holds under the definition the audits actually check.
    occ = None
    if occupancy is not None:
        occ = np.asarray(occupancy, dtype=bool)
        if occ.shape != (height, width):
            raise ValueError(
                "occupancy must be (H, W) matching the guide grid; got %r vs %r"
                % (occ.shape, (height, width))
            )
    n_occ = int(occ.sum()) if occ is not None else 0
    occ_floor_cells = (
        max(int(np.ceil(min_visible_frac * n_occ)), min(int(min_visible_cells), n_occ))
        if n_occ > 0
        else 0
    )

    def _slot_rng():
        """Placement RNG derived from a FIXED amount of shared-RNG consumption.

        Python's ``randrange(n)`` is rejection-based and consumes a variable
        number of PRNG words depending on ``n``.  Calling it directly with
        branch-dependent bounds would desynchronise the shared generator
        between fill modes, changing every later draw -- including the
        per-batch block sizes seeded at ``curriculum.py`` -- so two arms that
        share a seed would silently diverge.  Every placement path instead
        draws exactly ONE fixed-width seed here and samples from a slot-local
        generator, keeping shared-RNG consumption identical across branches.
        """
        return _random.Random(rng.getrandbits(64))

    def _uniform_loc(block_h: int, block_w: int) -> Tuple[int, int]:
        top = rng.randrange(0, max(1, height - block_h + 1))
        left = rng.randrange(0, max(1, width - block_w + 1))
        return top, left

    def _rect(top: int, left: int, block_h: int, block_w: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=bool)
        mask[top:top + block_h, left:left + block_w] = True
        if delivered_k is not None:
            indices = np.flatnonzero(mask)
            mask.ravel()[indices[delivered_k:]] = False
        return mask

    def _sums(grid: np.ndarray, block_h: int, block_w: int) -> np.ndarray:
        if delivered_k is None:
            return _window_sums(_integral(grid), block_h, block_w)
        rows, tail = divmod(delivered_k, block_w)
        n_top, n_left = height - block_h + 1, width - block_w + 1
        sums = np.zeros((n_top, n_left), dtype=np.float64)
        if rows:
            sums += _window_sums(_integral(grid), rows, block_w)[:n_top, :n_left]
        if tail:
            sums += _window_sums(_integral(grid), 1, tail)[
                rows:rows + n_top, :n_left
            ]
        return sums

    def _finish(masks: List[np.ndarray], floor_cells: int) -> Tuple[List[np.ndarray], Dict]:
        union = np.logical_or.reduce(masks)
        hidden = float(score[support & union].sum())
        info["covered_frac"] = (hidden / total_mass) if total_mass > 0 else 0.0
        info["visible_frac"] = 1.0 - info["covered_frac"]
        info["covered_cells"] = int((support & union).sum())
        info["visible_cells"] = int((support & ~union).sum())
        info["hit_target"] = bool(
            info["covered_frac"] >= 1.0 - leave_frac - 1e-9
        )
        info["floor_ok"] = bool(
            info["visible_frac"] >= min_visible_frac - 1e-9
            and info["visible_cells"] >= floor_cells
        )
        if occ is not None and n_occ > 0:
            occ_vis = int((occ & ~union).sum())
            info["occ_visible_cells"] = occ_vis
            info["occ_visible_frac"] = occ_vis / n_occ
            info["occ_floor_ok"] = bool(occ_vis >= occ_floor_cells)
            info["floor_ok"] = bool(info["floor_ok"] and info["occ_floor_ok"])
        info["union"] = int(union.sum())
        info["slots"] = int(sum(int(m.sum()) for m in masks))
        # The caller must fall back to stock uniform placement unless the hard
        # floors held AND the greedy actually hid something.  Reporting a
        # breach as a success is what would silently corrupt the arm.
        info["ok"] = bool(
            info["floor_ok"]
            and not info["floor_violation"]
            and (
                info["n_cover"] > 0
                or not info["coverage_needed"]
                or not any(guided)
            )
        )
        return masks, info

    # -- Degenerate slice: no anatomy, or too little to hide legally --------
    if n_anat <= 0 or total_mass <= 0.0 or n_anat <= int(min_visible_cells):
        info["fallback"] = True
        masks = []
        for block_h, block_w in block_sizes:
            block_h, block_w = min(block_h, height), min(block_w, width)
            top, left = _uniform_loc(block_h, block_w)
            mask = _rect(top, left, block_h, block_w)
            masks.append(mask)
            info["n_random"] += 1
            info["slot_kind"][len(masks) - 1] = "fallback"
        masks, info = _finish(masks, min(int(min_visible_cells), n_anat))
        info["ok"] = False
        return masks, info

    # ``remaining`` is anatomy mass not yet hidden by an already-placed block.
    remaining = np.where(support, score, 0.0)
    covered = np.zeros((height, width), dtype=bool)
    target_mass = (1.0 - leave_frac) * total_mass
    floor_mass = min_visible_frac * total_mass
    # Never demand more visible cells than the slice actually has.  The
    # degenerate guard above already routed away the cases where this would
    # forbid hiding anything at all.
    floor_cells = min(int(min_visible_cells), n_anat)
    covered_mass = 0.0
    masks: List[Optional[np.ndarray]] = [None] * n_blocks

    def _place(slot: int, mask: np.ndarray) -> None:
        """Record a mask and fold it into the coverage bookkeeping."""
        nonlocal covered_mass
        covered_mass += float(remaining[mask].sum())
        remaining[mask] = 0.0
        covered[mask] = True
        masks[slot] = mask

    # Blocks the ramp left unguided are placed uniformly by the caller.  They
    # are folded in FIRST so the floors are evaluated against the true final
    # union rather than the guided subset alone.
    for slot in range(n_blocks):
        if not guided[slot]:
            block_h = min(block_sizes[slot][0], height)
            block_w = min(block_sizes[slot][1], width)
            mask = fixed[slot]
            if mask is None:
                top, left = _uniform_loc(block_h, block_w)
                mask = _rect(top, left, block_h, block_w)
            elif delivered_k is not None:
                mask = np.asarray(mask, dtype=bool).copy()
                mask.ravel()[np.flatnonzero(mask)[delivered_k:]] = False
            _place(slot, np.asarray(mask, dtype=bool))
            info["n_unguided"] += 1
            info["slot_kind"][slot] = "unguided"

    # Whether any coverage work remained once the ramp's uniform blocks were
    # folded in.  If they already met the target, the guided blocks correctly
    # become transitions and n_cover==0 is a valid outcome, not a failure.
    info["coverage_needed"] = bool(covered_mass < target_mass)

    def _legality(block_h: int, block_w: int):
        """``(gains, legal)`` for every window of this size.

        ``gains`` is the anatomy mass a window would newly hide.  ``legal``
        marks windows that breach neither the mass floor nor the cell floor,
        both evaluated against the CUMULATIVE union rather than this block
        alone.
        """
        gains = _sums(remaining, block_h, block_w)
        if gains is None:
            return None, None
        legal = gains <= float(remaining.sum() - floor_mass) + 1e-9
        uncovered = (support & ~covered).astype(np.float64)
        newly = _sums(uncovered, block_h, block_w)
        visible_after = float(uncovered.sum()) - newly
        legal = legal & (visible_after >= floor_cells - 1e-9)
        if occ is not None and n_occ > 0:
            # Same floor, evaluated on the occupancy mask the audits use.
            occ_unc = (occ & ~covered).astype(np.float64)
            occ_newly = _sums(occ_unc, block_h, block_w)
            occ_after = float(occ_unc.sum()) - occ_newly
            legal = legal & (occ_after >= occ_floor_cells - 1e-9)
        return gains, legal

    def _choose(weights: np.ndarray) -> Tuple[int, int]:
        """Pick uniformly at random among the arg-maxima of ``weights``."""
        best = float(weights.max())
        candidates = np.argwhere(weights >= best - 1e-9)
        top, left = candidates[_slot_rng().randrange(len(candidates))]
        return int(top), int(left)

    def _commit(slot: int, top: int, left: int, block_h: int, block_w: int) -> None:
        _place(slot, _rect(top, left, block_h, block_w))

    for slot in range(n_blocks):
        if not guided[slot]:
            continue
        block_h = min(block_sizes[slot][0], height)
        block_w = min(block_sizes[slot][1], width)

        # ---- Phase 1: hide anatomy -------------------------------------
        placed = False
        if covered_mass < target_mass and remaining.sum() > floor_mass:
            gains, legal = _legality(block_h, block_w)
            if gains is not None and legal.any():
                scored = np.where(legal, gains, -np.inf)
                if float(scored.max()) > 0.0:
                    top, left = _choose(scored)
                    _commit(slot, top, left, block_h, block_w)
                    info["n_cover"] += 1
                    info["slot_kind"][slot] = "cover"
                    placed = True
            elif gains is not None:
                info["floor_blocked"] += 1
        if placed:
            continue

        # ---- Phase 2: spend the leftover block on the boundary ----------
        gains, legal = _legality(block_h, block_w)

        if fill_mode == "random_legal" and gains is not None:
            # Plain uniform placement -- no boundary shaping, no greedy scoring
            # -- but restricted to windows that keep the hard anatomy-visibility
            # floor intact.  This is the "rest of the blocks are random like
            # normal JEPA, but 0.15 anatomy stays visible" variant.
            if legal.any():
                cand = np.argwhere(legal)
                top, left = (int(v) for v in cand[_slot_rng().randrange(len(cand))])
                info["n_random"] += 1
                info["slot_kind"][slot] = "random_legal"
            else:
                # Nothing can be placed without breaching the floor.  Take the
                # least-damaging window and FLAG it so the caller can discard
                # this image rather than pass off a floor breach as guided.
                top, left = _choose(-gains)
                info["floor_violation"] = True
                info["n_random"] += 1
                info["slot_kind"][slot] = "random_violation"
            _commit(slot, top, left, block_h, block_w)
            continue

        if fill_mode == "random" or gains is None:
            # No boundary work requested: this leftover block is a plain
            # uniform JEPA block, exactly as in the stock sampler.  Note this
            # ignores the floor by construction.  Drawn through _slot_rng so
            # shared-RNG consumption matches the other fill modes.
            r = _slot_rng()
            top = r.randrange(0, max(1, height - block_h + 1))
            left = r.randrange(0, max(1, width - block_w + 1))
            _commit(slot, top, left, block_h, block_w)
            info["n_random"] += 1
            info["slot_kind"][slot] = "random"
            continue

        n_tissue = _sums(support.astype(np.float64), block_h, block_w)
        n_background = float(delivered_k or block_h * block_w) - n_tissue
        balance = np.minimum(n_tissue, n_background)
        scored = np.where(legal, balance, -np.inf)
        if float(scored.max()) > 0.0:
            top, left = _choose(scored)
            info["n_transition"] += 1
            info["slot_kind"][slot] = "transition"
        else:
            # Nothing legal straddles the boundary.  Prefer a legal window if
            # one exists at all, and otherwise minimise the floor breach by
            # hiding as little *remaining* anatomy as possible — minimising
            # against the full support instead would ignore what has already
            # been covered and can pick a strictly worse window.
            if legal.any():
                top, left = _choose(np.where(legal, -n_tissue, -np.inf))
            else:
                # No legal window exists for this block at all.  Take the
                # least-damaging one, but FLAG it: the hard floors are the
                # arm's stated guarantee, so an image that breaches them is
                # handed back to the caller to replace with the stock uniform
                # fallback rather than being passed off as guided.
                top, left = _choose(-gains)
                info["floor_violation"] = True
            # Random tie-breaking among guide-selected least-tissue windows
            # does not make this an unguided uniform fill.
            info["n_boundary_fallback"] += 1
            info["slot_kind"][slot] = "boundary_fallback"
        _commit(slot, top, left, block_h, block_w)

    return _finish([m for m in masks], floor_cells)
