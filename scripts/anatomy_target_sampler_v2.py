"""Class-aware anatomy target sampler (v2).

Rewritten after two measured defects in v1:

  D1  CONTEXT.  v1's demo used  context = all_patches - union,  giving the
      encoder ~214 of 256 tokens.  The real I-JEPA collator hands it far less.
      Two different figures are both correct and must not be confused:
      per-image, before collation, masks_enc holds ~100 tokens (rect arm);
      after _truncate_and_stack cuts every row to the batch minimum it is
      84.3 at batch 8, 80.9 at 16, 67.9 at 64 (measured, 40 batches each).
      Either way v1's 214 was 2.5x too generous and the task was not
      comparable.  The context policy now stays with the original collator and
      MIRAGE only replaces masks_pred.

  D2  MASS vs EXTENT.  v1 grew the union until it held rho of the anatomy
      PROBABILITY MASS.  Because the confident core of the band carries most of
      the mass, 70% of mass covered only 52-56% of the cells MIRAGE calls
      anatomy (measured, tau=0.05/0.10).  The union therefore collapsed onto the
      bright ridge and systematically skipped uncertain boundaries and thin
      structures.  Growth now satisfies a mass budget AND an extent budget.

Four further changes, all from the same review:

  * CLASS-AWARE.  InnerRetina and Choroid are separate structures with a real
    unlabelled gap between them (GOALS does not annotate the mid-retina).
    Forcing P1+P2 into one connected object made the sampler build bridges
    across tissue that is not labelled as either.  Each class now grows its own
    region and receives its own share of the four targets.
  * GEOMETRY-ONLY PARTITION.  v1 used the anatomy score again when splitting,
    so all four chunks kept chasing the same bright ridge.  Anatomy decides
    WHICH cells are in the region; geometry decides HOW the region is divided.
    Partition is farthest-point seeding + multi-source BFS, which yields
    geodesic Voronoi cells -- provably connected, since every cell on a
    shortest path to its own seed is also assigned to that seed.
  * OVERLAP OFF BY DEFAULT.  v1 dilated targets back into each other to imitate
    the 23.9% overlap that I-JEPA rectangles happen to have, partly undoing the
    split it had just computed.  Distinct connected targets matter more; keep
    overlap as an ablation.
  * SMALL HOLES ONLY.  binary_fill_holes closed every cavity, including genuine
    low-anatomy tissue, and the trim step then removed real boundary cells to
    pay for it.  Only cavities up to `max_hole` cells are filled.

Balance is relaxed from "spread <= 1" to a size ratio, because forcing a cell
across a border to turn 13/15 into 14/14 can make the anatomical shape worse.
Priority order: anatomically sensible > connected > distinct > roughly balanced.

SCOPE OF GUARANTEES (measured, 1000 slices, cap 0.80).  Both the disjointness
and the balance guarantee are PER CLASS, because grow_region / geodesic_partition
/ rebalance each run once per class:

  * Disjointness holds strictly within a class (0/1000 within-class overlaps).
    Across classes 2.8% of slices have at least one shared cell, because MIRAGE
    gives 0.37 cells per image a probability above tau for BOTH inner retina and
    choroid at their mutual boundary.  This is benign -- a cell the segmenter
    calls ambiguous is reasonably predicted by both targets, and I-JEPA's own
    rectangles overlap each other on 65.4% of slices.  Raising the cap makes it
    worse (35/1000 at 0.90), not better.
  * `max_ratio` likewise bounds sizes only within a class.  The global max/min
    over all four targets has mean 1.64 / p95 2.29, and reaches 13.0 when a
    2-cell choroid region is paired with a 26-cell inner-retina one.

Targets average 11.0 cells (p05 6, max 22) versus 38-51 for an I-JEPA
rectangle; the scale gap is a known open risk, not a defect of this file.
1.4% of slices yield at least one EMPTY target, driven by crops where MIRAGE
finds essentially no anatomy.  That rate is independent of `mass_cap`
(14/1000 at both 0.80 and 0.90) and needs a caller-side fallback.
"""
from __future__ import annotations

from collections import deque

import numpy as np

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbours(r, c, h, w):
    for dr, dc in NB8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def n_components(m):
    from scipy import ndimage
    if m.sum() == 0:
        return 0
    return int(ndimage.label(m, structure=np.ones((3, 3)))[1])


# ------------------------------------------------------------ region ------
def grow_region(score, mass_cap=0.80, tau=0.10, rho_extent=None):
    """One connected region inside the anatomy support S = {score > tau}.

    Growth is CONFINED to S.  Letting it leave S was measured to be a real bug:
    with an extent-driven stopping rule the region grew to 208-255 of 256 cells,
    because near-zero-score background cells cost almost no mass, so the mass
    cap never tripped and ~145 of those cells were outside the anatomy entirely.

    `mass_cap` is the single stopping rule; 0.80 is the calibrated default.

    The previous 0.90 default was calibrated on the WRONG criterion: "the
    context never starves -- 0.0% of slices fall below the collator's
    min_keep=10, even at cap 0.99".  That test counts TOKENS, not CONTENT, and
    is satisfied by 176 tokens of black vitreous.  Recalibrated on what the
    encoder can actually SEE (300 slices, production MaskCollator context
    policy, confident cell := score > 0.5):

        cap   union    ctx   confident cells   slices whose CONTEXT
                             left in context   keeps >=1 confident cell
        0.70   36.9   188.5        10.2                97.7%
        0.75   40.0   185.8         7.8                96.0%
        0.80   43.5   182.8         5.5                92.0%   <- default
        0.85   47.3   179.4         3.5                69.7%
        0.90   52.1   175.3         2.4                35.7%
        0.95   58.4   169.6         2.2                22.3%

    The knee is a cliff between 0.80 and 0.825.  Past it the encoder is
    increasingly asked to predict the confident core of the retina from a view
    holding none of it -- the K=105 failure reached from the other side.
    Going 0.90 -> 0.80 costs only 8.6 target cells and 7.5 context tokens but
    lifts the safe fraction from 35.7% to 92.0%; the trade is very lopsided.
    Random rectangles keep >=1 confident cell in 98.3% of slices at any cap.

    The headline is threshold-sensitive and should be quoted with its cutoff:
    at cap 0.90 the unsafe fraction is 6.0% at score>0.3, 31.3% at >0.4,
    64.3% at >0.5, 75.7% at >0.6.  The ORDERING across caps is stable at every
    cutoff; the magnitude is not.

    A per-slice floor (stop early if fewer than N confident cells would remain
    visible) dominates any fixed cap, since failures concentrate on slices with
    little anatomy.  Not yet implemented.

    An additional extent budget was tried in three formulations and is INERT:
    raising the extent goal from 0.70 to 0.90 moved coverage by 0.001-0.011,
    because the mass cap always binds first.  `rho_extent` is kept for
    ablations only and defaults to off.

    Note the budget is quoted in MASS, not area.  At cap 0.80 the region holds
    80% of the anatomy probability mass but covers a smaller share of the
    support cells, because the confident core carries most of the mass.
    """
    h, w = score.shape
    S = score > tau
    total_mass = float(score.sum())
    U = np.zeros((h, w), bool)
    if total_mass <= 0 or not S.any():
        return U

    idx = [tuple(x) for x in np.argwhere(S)]
    seed = max(idx, key=lambda rc: score[rc])
    U[seed] = True
    mass = float(score[seed])
    ext, total_ext = 1, int(S.sum())
    while mass < mass_cap * total_mass:
        if rho_extent is not None and ext >= rho_extent * total_ext:
            break
        frontier = {}
        for (rr, cc) in np.argwhere(U):
            for nr, nc in _neighbours(rr, cc, h, w):
                if S[nr, nc] and not U[nr, nc]:
                    frontier[(nr, nc)] = score[nr, nc]
        if not frontier:
            break
        best = max(frontier, key=frontier.get)
        if mass + float(score[best]) > mass_cap * total_mass:
            break
        U[best] = True
        mass += float(score[best])
        ext += 1
    return U


def fill_small_holes(U, score, max_hole=2):
    """Close only tiny cavities.  A large hole may be genuine unlabelled tissue."""
    from scipy import ndimage
    st = np.ones((3, 3))
    holes = ndimage.binary_fill_holes(U) & ~U
    if not holes.any():
        return U
    lab, n = ndimage.label(holes, structure=st)
    out = U.copy()
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() <= max_hole:
            out |= comp
    return out


# --------------------------------------------------------- partition -----
def _bfs(U, sources):
    """Multi-source BFS inside U.  Returns (distance, owner)."""
    h, w = U.shape
    dist = np.full((h, w), np.inf)
    own = np.full((h, w), -1, int)
    q = deque()
    for i, rc in enumerate(sources):
        dist[rc] = 0
        own[rc] = i
        q.append(rc)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbours(r, c, h, w):
            if U[nr, nc] and dist[nr, nc] == np.inf:
                dist[nr, nc] = dist[r, c] + 1
                own[nr, nc] = own[r, c]
                q.append((nr, nc))
    return dist, own


def geodesic_partition(U, k):
    """Split U into k connected pieces using GEOMETRY ONLY -- no anatomy score.

    Farthest-point seeding then multi-source BFS.  The resulting geodesic
    Voronoi cells are connected: if x is assigned to seed s, every cell on a
    shortest path x->s is at least as close to s, so it belongs to the same
    cell.
    """
    cells = [tuple(x) for x in np.argwhere(U)]
    if not cells:
        return [np.zeros_like(U) for _ in range(k)]
    if len(cells) <= k:
        parts = [np.zeros_like(U) for _ in range(k)]
        for i, rc in enumerate(cells):
            parts[i % k][rc] = True
        return parts

    # seed 1: one end of the region's diameter (double BFS)
    d0, _ = _bfs(U, [cells[0]])
    seeds = [max(cells, key=lambda rc: d0[rc])]
    # remaining seeds: farthest from every seed chosen so far
    while len(seeds) < k:
        d, _ = _bfs(U, seeds)
        nxt = max(cells, key=lambda rc: d[rc])
        if d[nxt] == 0:                       # degenerate: no cell left apart
            free = [rc for rc in cells if rc not in seeds]
            if not free:
                break
            nxt = free[0]
        seeds.append(nxt)

    _, own = _bfs(U, seeds)
    parts = [np.zeros_like(U) for _ in range(k)]
    for rc in cells:
        o = own[rc]
        parts[o if o >= 0 else 0][rc] = True
    return parts


def rebalance(parts, max_ratio=1.25, rounds=200):
    """Relax size imbalance across ADJACENT borders, connectivity-preserving.

    Deliberately weaker than v1's spread<=1: forcing a cell across a border to
    turn 13/15 into 14/14 can worsen the anatomical shape, and I-JEPA's own
    targets are not equal either.
    """
    parts = [p.copy() for p in parts]
    if len(parts) < 2:
        return parts
    h, w = parts[0].shape
    n = len(parts)

    def try_move(src, dst):
        for (r, c) in np.argwhere(parts[src]):
            if not any(parts[dst][nr, nc] for nr, nc in _neighbours(r, c, h, w)):
                continue
            donor = parts[src].copy()
            donor[r, c] = False
            if donor.sum() == 0 or n_components(donor) != 1:
                continue
            recv = parts[dst].copy()
            recv[r, c] = True
            if n_components(recv) != 1:
                continue
            parts[src], parts[dst] = donor, recv
            return True
        return False

    for _ in range(rounds):
        sizes = [max(int(p.sum()), 1) for p in parts]
        if max(sizes) / min(sizes) <= max_ratio:
            break
        order = sorted(range(n), key=lambda i: -sizes[i])
        moved = False
        for i in order:
            for j in sorted(range(n), key=lambda x: sizes[x]):
                if i == j or sizes[i] - sizes[j] < 2:
                    continue
                if try_move(i, j):
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    return parts


def expand_overlap(parts, U, score, frac):
    """Optional: dilate targets within U to a target pairwise overlap. OFF by default."""
    if frac <= 0 or len(parts) < 2:
        return parts
    parts = [p.copy() for p in parts]
    h, w = U.shape
    n = len(parts)
    for _ in range(400):
        base = float(np.mean([p.sum() for p in parts]))
        pairs = n * (n - 1) / 2
        ov = sum(int((parts[i] & parts[j]).sum())
                 for i in range(n) for j in range(i + 1, n))
        if base > 0 and ov / pairs / base >= frac:
            break
        grew = False
        for i, p in enumerate(parts):
            cand = {}
            for (r, c) in np.argwhere(p):
                for nr, nc in _neighbours(r, c, h, w):
                    if U[nr, nc] and not p[nr, nc]:
                        cand[(nr, nc)] = score[nr, nc]
            if cand:
                parts[i][max(cand, key=cand.get)] = True
                grew = True
        if not grew:
            break
    return parts


# ------------------------------------------------------------ driver -----
def allocate(masses, n=4, capacities=None):
    """Split n targets across classes proportionally to mass.

    `capacities` is the number of GROWN CELLS available to each class.  It is
    what makes the split safe: a class can never receive more targets than it
    has cells, so a target can never come back empty.

    Allocating on mass alone was measured to be the dominant cause of empty
    targets -- 1.0% of 1000 slices, versus 0.4% where MIRAGE genuinely finds no
    anatomy.  `present = mass > 1e-6` is almost always True for both classes
    because softmax probabilities are positive everywhere, so a class with ZERO
    cells above tau still claimed a target and that target was empty by
    construction.  Real example (slice 417): inner retina mass 4.88 / 10 support
    cells, choroid mass 0.04 / 0 support cells; the old rule allocated [3, 1]
    and the choroid target could not be filled.  Capacity-aware allocation
    gives [4, 0] instead.
    """
    m = np.asarray(masses, float)
    cap = (np.full(len(m), n, int) if capacities is None
           else np.asarray(capacities, int))
    present = cap > 0
    if not present.any():
        return [0] * len(m)

    k = np.zeros(len(m), int)
    k[present] = 1
    k = np.minimum(k, cap)

    # Distribute the remainder by mass share, never exceeding capacity.
    while k.sum() < n:
        room = present & (k < cap)
        if not room.any():
            break
        # largest mass-per-target-so-far wins the next target
        pref = np.where(room, m / np.maximum(k, 1), -np.inf)
        k[int(np.argmax(pref))] += 1

    while k.sum() > n:
        k[int(np.argmax(k))] -= 1
    return k.tolist()


def build_targets(class_scores, n=4, mass_cap=0.80, tau=0.10,
                  overlap=0.0, max_hole=2, max_ratio=1.25, rho_extent=None):
    """Class-aware anatomy targets.

    class_scores : list of per-class soft occupancy grids, e.g.
                   [P_inner, P_choroid] on the 16x16 JEPA grid.
    Returns (parts, regions) with exactly `n` masks.

    Regions are grown BEFORE allocation so the split can be capacity-aware;
    growth does not depend on how many targets a class receives, so this
    reorder changes no region, only how targets are shared out.

    A class with no support above `tau` is dropped entirely.  When every class
    is empty -- MIRAGE found no anatomy at all, 0.4% of slices -- this returns
    `n` empty masks and the CALLER must fall back (see `has_anatomy`).
    """
    shape = class_scores[0].shape
    grown, masses, caps = [], [], []
    for score in class_scores:
        if not (score > tau).any():
            grown.append(None)
            masses.append(0.0)
            caps.append(0)
            continue
        U = grow_region(score, mass_cap, tau, rho_extent)
        U = fill_small_holes(U, score, max_hole)
        grown.append(U)
        masses.append(float(score.sum()))
        caps.append(int(U.sum()))

    ks = allocate(masses, n, caps)

    parts, regions = [], []
    for score, U, k in zip(class_scores, grown, ks):
        if k == 0 or U is None:
            continue
        regions.append(U)
        cp = geodesic_partition(U, k)
        cp = rebalance(cp, max_ratio)
        cp = expand_overlap(cp, U, score, overlap)
        parts.extend(cp)

    while len(parts) < n:
        parts.append(np.zeros(shape, bool))
    return parts[:n], regions


def region_capacity(class_scores, mass_cap=0.80, tau=0.10, max_hole=2,
                    rho_extent=None):
    """Cells the sampler could actually place targets in, per class.

    This is the real feasibility quantity.  Support alone is not enough: the
    grown region is what gets partitioned, so capacity is what limits how many
    non-empty targets exist.
    """
    caps = []
    for score in class_scores:
        score = np.asarray(score)
        if not (score > tau).any():
            caps.append(0)
            continue
        U = grow_region(score, mass_cap, tau, rho_extent)
        U = fill_small_holes(U, score, max_hole)
        caps.append(int(U.sum()))
    return caps


def is_viable(class_scores, n=4, min_cells=4, mass_cap=0.80, tau=0.10,
              max_hole=2, rho_extent=None):
    """Can this image support `n` targets of at least `min_cells` each?

    Callers MUST gate on this and fall back to a MIRAGE-independent mask when
    it is False.  Checking "is there any anatomy" is NOT sufficient: measured
    over 1000 slices, only 0.4% have zero anatomy but 0.9% cannot fill four
    targets at all, because a 1-3 cell region still partitions into fewer than
    four non-empty parts and `build_targets` then pads with empty masks.

    An empty mask reaching the collator is fatal, not merely degenerate --
    `torch.stack([t[:min_len] ...])` in multiblock.py / curriculum.py raises
    "stack expects each tensor to be equal size, but got [1] and [0]".

    `min_cells` also guards a subtler batch-level hazard: curriculum.py
    truncates every target in the batch to the SMALLEST one, so a single
    2-cell target drags all 4*B targets down to 2 cells.

    Measured fallback rate over 1000 slices:

        min_cells   1 -> 0.90%     4 -> 2.30%
        min_cells   2 -> 1.20%     5 -> 2.60%
        min_cells   3 -> 1.90%

    Default 4 costs 2.3% of images and keeps every target at a usable size
    (normal targets average 11.0 cells, p05 6).

    FALL BACK TO RANDOM RECTANGLES, not to the geometric "oracle" ribbon.
    Measured on a real degenerate crop (data_08569/slice_199, a near-black scan
    whose only retina is a faint sliver at the left edge): MIRAGE localised it
    correctly and the 18-cell anatomy mask was 1.52x brighter than its
    surroundings, capturing 63.4% of the anatomy score.  The oracle ribbon
    masked 70 cells and captured only 46.2%, because it is confined to the
    central `oracle_lateral_frac` of columns (3..12 of 16) and the retina was at
    columns 0..5.  Degenerate crops put anatomy at the EDGE, which is exactly
    where a centred band cannot look.

    Random rectangles are also the honest fallback because they are the
    baseline arm, so a fallback image is scored under the control condition
    rather than under a third, untested mask distribution.  They are not a
    downgrade either -- measured over 1000 slices x 4 targets, using
    "target contains no cell above tau" as a common definition of failure:

        dead targets          random 14.12%   anatomy+fallback 2.12%
        all 4 targets dead    random  2.40%   anatomy+fallback 1.70%

    i.e. the anatomy arm WITH its 2.3% fallback still fails 6.7x less often
    than the random arm it falls back to.

    CAVEAT: this gate bounds TOTAL capacity, so it removes empty targets
    entirely but does not guarantee a per-target minimum -- a 1-cell target is
    still reachable when a region has a thin tail.  That matters because
    curriculum.py truncates the whole batch to its smallest target.
    """
    caps = region_capacity(class_scores, mass_cap, tau, max_hole, rho_extent)
    return sum(caps) >= n * min_cells


def has_anatomy(class_scores, tau=0.10):
    """True if ANY class has support above `tau`.

    NOTE: this is a necessary but NOT sufficient condition for building `n`
    targets -- prefer `is_viable`.  Kept because it is the cheap check (no
    region growth) and is the right test when you only care whether MIRAGE saw
    anything at all.
    """
    return any(bool((np.asarray(s) > tau).any()) for s in class_scores)


def topology_of(m):
    from scipy import ndimage
    if m.sum() == 0:
        return dict(components=0, holes=0, hole_area=0, n_cells=0)
    st = np.ones((3, 3))
    _, k = ndimage.label(m, structure=st)
    holes = ndimage.binary_fill_holes(m) & ~m
    _, nh = ndimage.label(holes, structure=st)
    return dict(components=int(k), holes=int(nh),
                hole_area=int(holes.sum()), n_cells=int(m.sum()))
