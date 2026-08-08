"""Union-first, split-second anatomy target sampler.

Builds ONE connected anatomy-budgeted union, then partitions it into four
contiguous, near-equal, individually-connected targets.  This replaces growing
four independent regions, which produced a union of 2-5 disjoint islands and
targets that looked like arbitrary worms rather than a meaningful partition of
one anatomical band.

    a          MIRAGE anatomy occupancy on the 16x16 JEPA grid
    rho        fraction of anatomy MASS the union may hide (0.70 measured)
    overlap    optional per-target dilation WITHIN the union, so targets
               overlap like I-JEPA's rectangles do (measured 23.9% of a block);
               set to 0.0 for a strict disjoint partition

Pipeline
    1  grow_union      seed at max score, 8-connected region growing to budget
    2  fill_holes      close cavities, then prune the lowest-score boundary
                       cells back to the budget so filling cannot inflate it
    3  order_geodesic  double-BFS to find the band's two extreme ends, then
                       order every union cell by geodesic distance from one end
                       -- this follows a curved band, unlike a principal axis
    4  split4          cut the ordered sequence into four near-equal chunks
    5  repair          reassign stray disconnected cells to the neighbouring
                       chunk they touch most
    6  expand          optionally dilate each target within the union to reach
                       the requested overlap fraction

Everything is integer/boolean and runs under no_grad by construction: the mask
is a discrete decision, deliberately outside the autograd graph.
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


def grow_union(a, rho):
    """One connected region, grown until it hides `rho` of the anatomy mass."""
    h, w = a.shape
    budget = rho * float(a.sum())
    U = np.zeros((h, w), bool)
    if budget <= 0 or not np.isfinite(a).any():
        return U
    r, c = np.unravel_index(np.argmax(a), a.shape)
    U[r, c] = True
    got = a[r, c]
    frontier = {}
    while got < budget:
        for (rr, cc) in np.argwhere(U):
            for nr, nc in _neighbours(rr, cc, h, w):
                if not U[nr, nc]:
                    frontier[(nr, nc)] = a[nr, nc]
        if not frontier:
            break
        best = max(frontier, key=frontier.get)
        del frontier[best]
        U[best] = True
        got += a[best]
    return U


def fill_and_trim(U, a, rho):
    """Close cavities, then trim back to the budget from the lowest-score edge."""
    from scipy import ndimage
    U = ndimage.binary_fill_holes(U)
    budget = rho * float(a.sum())
    while (a * U).sum() > budget and U.sum() > 4:
        # only remove cells whose deletion cannot disconnect the region
        edge = [(r, c) for r, c in np.argwhere(U)
                if any(not U[nr, nc] for nr, nc in _neighbours(r, c, *U.shape))]
        if not edge:
            break
        edge.sort(key=lambda rc: a[rc])
        removed = False
        for rc in edge:
            trial = U.copy()
            trial[rc] = False
            lab, n = ndimage.label(trial, structure=np.ones((3, 3)))
            if n == 1 and trial.sum() > 0:
                U = trial
                removed = True
                break
        if not removed:
            break
    return U


def _bfs_dist(U, start):
    h, w = U.shape
    d = np.full((h, w), -1, int)
    d[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbours(r, c, h, w):
            if U[nr, nc] and d[nr, nc] < 0:
                d[nr, nc] = d[r, c] + 1
                q.append((nr, nc))
    return d


def order_geodesic(U):
    """Order union cells along the band, from one extreme end to the other.

    Double BFS: from an arbitrary cell find the farthest (one end of the band's
    diameter), then order everything by geodesic distance from that end.  This
    follows a curved or diagonal band, which a principal-axis projection does
    not.
    """
    cells = np.argwhere(U)
    if len(cells) == 0:
        return []
    start = tuple(cells[0])
    d0 = _bfs_dist(U, start)
    end_a = tuple(cells[np.argmax([d0[tuple(c)] for c in cells])])
    d1 = _bfs_dist(U, end_a)
    order = sorted((tuple(c) for c in cells), key=lambda rc: (d1[rc], rc))
    return order


def split4(U, a, n=4):
    """Partition the union into n connected, near-equal chunks.

    Seeds are placed evenly along the geodesic ordering, then the chunks are
    grown ROUND-ROBIN by multi-source BFS inside the union.  Both properties
    come for free:

      connectivity  each chunk only ever expands into its own 8-neighbourhood
                    from a single seed, so it cannot fragment
      balance       round-robin means chunk sizes differ by at most one until a
                    chunk runs out of frontier

    Cutting the geodesic ORDER by rank instead (the obvious approach) splits
    level sets of the BFS distance and produced disconnected chunks on 29.7% of
    slices, with a size spread of 4.0 cells.
    """
    order = order_geodesic(U)
    if not order:
        return [np.zeros_like(U) for _ in range(n)]
    if len(order) <= n:
        parts = [np.zeros_like(U) for _ in range(n)]
        for i, rc in enumerate(order):
            parts[i % n][rc] = True
        return parts

    h, w = U.shape
    # seeds at 1/2n, 3/2n, ... along the band so the chunks start evenly spread
    idx = [int((2 * i + 1) * len(order) / (2 * n)) for i in range(n)]
    parts = [np.zeros_like(U) for _ in range(n)]
    claimed = np.zeros_like(U)
    for i, k in enumerate(idx):
        rc = order[min(k, len(order) - 1)]
        if claimed[rc]:                       # duplicate seed on a tiny union
            free = [x for x in order if not claimed[x]]
            if not free:
                continue
            rc = free[0]
        parts[i][rc] = True
        claimed[rc] = True

    stalled = [not p.any() for p in parts]
    while not claimed[U].all() and not all(stalled):
        for i in range(n):
            if stalled[i]:
                continue
            cand = {}
            for (r, c) in np.argwhere(parts[i]):
                for nr, nc in _neighbours(r, c, h, w):
                    if U[nr, nc] and not claimed[nr, nc]:
                        cand[(nr, nc)] = a[nr, nc]
            if not cand:
                stalled[i] = True
                continue
            best = max(cand, key=cand.get)
            parts[i][best] = True
            claimed[best] = True
    return parts


def repair(parts):
    """Reassign stray disconnected cells to the neighbouring chunk they touch most."""
    from scipy import ndimage
    parts = [p.copy() for p in parts]
    h, w = parts[0].shape
    for i, p in enumerate(parts):
        if p.sum() == 0:
            continue
        lab, n = ndimage.label(p, structure=np.ones((3, 3)))
        if n <= 1:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        keep = int(np.argmax(sizes)) + 1
        for comp in range(1, n + 1):
            if comp == keep:
                continue
            for (r, c) in np.argwhere(lab == comp):
                votes = np.zeros(len(parts), int)
                for nr, nc in _neighbours(r, c, h, w):
                    for j, q in enumerate(parts):
                        if j != i and q[nr, nc]:
                            votes[j] += 1
                if votes.sum() > 0:
                    parts[i][r, c] = False
                    parts[int(np.argmax(votes))][r, c] = True
    return parts


def expand_overlap(parts, U, a, frac):
    """Dilate each target within the union until pairwise overlap reaches `frac`.

    I-JEPA's rectangular targets overlap by ~23.9% of a block (measured on this
    repo's own collator).  A strict partition has 0% overlap and quarter-sized
    targets, which moves further from I-JEPA's target-scale regime; this step
    restores both without changing the union.

    Growth is round-robin and each chunk adds at most one cell per pass, so the
    size balance established by ``split4`` is preserved.
    """
    if frac <= 0:
        return parts
    parts = [p.copy() for p in parts]
    h, w = U.shape
    n = len(parts)
    for _ in range(400):
        base = float(np.mean([p.sum() for p in parts]))
        ov = sum(int((parts[i] & parts[j]).sum())
                 for i in range(n) for j in range(i + 1, n))
        if base > 0 and ov / 6.0 / base >= frac:
            break
        grew = False
        for i, p in enumerate(parts):
            cand = {}
            for (r, c) in np.argwhere(p):
                for nr, nc in _neighbours(r, c, h, w):
                    if U[nr, nc] and not p[nr, nc]:
                        cand[(nr, nc)] = a[nr, nc]
            if cand:
                parts[i][max(cand, key=cand.get)] = True
                grew = True
        if not grew:
            break
    return parts


def rebalance(parts, U, max_spread=1, rounds=400):
    """Equalise chunk sizes by moving cells across ADJACENT chunk borders.

    Round-robin growth alone does not equalise sizes: in a band the two END
    chunks run out of frontier early while the middle chunks absorb everything
    left, giving a measured spread of ~4.6 cells.

    Moving directly from the global largest to the global smallest fails when
    those two are not adjacent (chunks form a chain along the band, so chunk 1
    and chunk 4 never touch) -- that only reached 3.5.  Instead this relaxes
    every ADJACENT pair, which propagates cells along the chain, and accepts a
    move only when BOTH donor and receiver stay connected.
    """
    from scipy import ndimage
    parts = [p.copy() for p in parts]
    h, w = U.shape
    st = np.ones((3, 3))
    n = len(parts)

    def touching(i, j):
        for (r, c) in np.argwhere(parts[i]):
            for nr, nc in _neighbours(r, c, h, w):
                if parts[j][nr, nc]:
                    return True
        return False

    def try_move(src, dst):
        for (r, c) in np.argwhere(parts[src]):
            if not any(parts[dst][nr, nc] for nr, nc in _neighbours(r, c, h, w)):
                continue
            donor = parts[src].copy()
            donor[r, c] = False
            if donor.sum() == 0 or ndimage.label(donor, structure=st)[1] != 1:
                continue
            recv = parts[dst].copy()
            recv[r, c] = True
            if ndimage.label(recv, structure=st)[1] != 1:
                continue
            parts[src], parts[dst] = donor, recv
            return True
        return False

    for _ in range(rounds):
        sizes = [int(p.sum()) for p in parts]
        if max(sizes) - min(sizes) <= max_spread:
            break
        moved = False
        # relax every adjacent pair, biggest imbalance first
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        pairs.sort(key=lambda ij: parts[ij[1]].sum() - parts[ij[0]].sum())
        for i, j in pairs:
            if int(parts[j].sum()) - int(parts[i].sum()) < 2:
                continue
            if not touching(i, j):
                continue
            if try_move(j, i):
                moved = True
                break
        if not moved:
            break
    return parts


def build_targets(a, rho=0.70, overlap=0.24, n=4, do_fill=True):
    """Full pipeline: one connected union, split into n balanced targets."""
    U = grow_union(a, rho)
    if U.sum() == 0:
        return [np.zeros_like(U) for _ in range(n)], U
    if do_fill:
        U = fill_and_trim(U, a, rho)
    parts = split4(U, a, n)
    parts = rebalance(parts, U)
    parts = expand_overlap(parts, U, a, overlap)
    return parts, U


def topology_of(m):
    """Components, holes and area for a boolean mask (8-connectivity)."""
    from scipy import ndimage
    if m.sum() == 0:
        return dict(components=0, holes=0, hole_area=0, n_cells=0)
    st = np.ones((3, 3))
    _, n = ndimage.label(m, structure=st)
    holes = ndimage.binary_fill_holes(m) & ~m
    _, nh = ndimage.label(holes, structure=st)
    return dict(components=int(n), holes=int(nh),
                hole_area=int(holes.sum()), n_cells=int(m.sum()))


# --------------------------------------------------------------- audit ----
def audit(grids, rho=0.70, overlap=0.24, do_fill=True):
    from scipy import ndimage
    rows = []
    for a in grids:
        A = float(a.sum())
        if A <= 1e-6:
            continue
        parts, U = build_targets(a, rho, overlap, do_fill=do_fill)
        union = np.zeros_like(U)
        for p in parts:
            union |= p
        _, uc = ndimage.label(union, structure=np.ones((3, 3)))
        holes = ndimage.binary_fill_holes(union) & ~union
        _, nh = ndimage.label(holes, structure=np.ones((3, 3)))
        sizes = [int(p.sum()) for p in parts]
        pc = [ndimage.label(p, structure=np.ones((3, 3)))[1] for p in parts if p.sum()]
        ov = sum(int((parts[i] & parts[j]).sum())
                 for i in range(4) for j in range(i + 1, 4))
        rows.append(dict(
            union_components=uc, union_cells=int(union.sum()),
            holes=nh, hole_area=int(holes.sum()),
            target_min=min(sizes), target_max=max(sizes),
            target_spread=max(sizes) - min(sizes),
            target_mean=float(np.mean(sizes)),
            all_targets_connected=all(c == 1 for c in pc),
            visible=float(1 - (a * union).sum() / A),
            purity=float((a * union).sum() / max(union.sum(), 1)),
            pairwise_overlap_pct=100.0 * ov / 6 / max(np.mean(sizes), 1),
        ))
    return rows


def main():
    import argparse
    import json
    import pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', type=pathlib.Path,
                    default=pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_base.npz'))
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--rho', type=float, nargs='+', default=[0.70])
    ap.add_argument('--overlap', type=float, nargs='+', default=[0.0, 0.24])
    ap.add_argument('--out', type=pathlib.Path,
                    default=pathlib.Path('results/masking/union_split'))
    a = ap.parse_args()

    G = np.load(a.grids, allow_pickle=True)['grids'][:a.n]
    rep = {'n_slices': int(len(G)), 'runs': {}}
    print('=== union-first, split-second sampler: mask quality audit ===')
    print('%6s %8s %8s %9s %8s %9s %10s %9s %9s %9s'
          % ('rho', 'overlap', 'union=1', 'targets=1', 'holes', 'holefree',
             'tgt spread', 'tgt mean', 'visible', 'ov%'))
    for rho in a.rho:
        for ovf in a.overlap:
            rows = audit(G, rho, ovf)
            d = {
                'union_connected_frac': float(np.mean([r['union_components'] == 1 for r in rows])),
                'all_targets_connected_frac': float(np.mean([r['all_targets_connected'] for r in rows])),
                'holes_mean': float(np.mean([r['holes'] for r in rows])),
                'holefree_frac': float(np.mean([r['holes'] == 0 for r in rows])),
                'target_spread_mean': float(np.mean([r['target_spread'] for r in rows])),
                'target_spread_p95': float(np.percentile([r['target_spread'] for r in rows], 95)),
                'target_mean': float(np.mean([r['target_mean'] for r in rows])),
                'visible_mean': float(np.mean([r['visible'] for r in rows])),
                'visible_p05': float(np.percentile([r['visible'] for r in rows], 5)),
                'visible_in_25_35_frac': float(np.mean(
                    [0.25 <= r['visible'] <= 0.35 for r in rows])),
                'union_cells_mean': float(np.mean([r['union_cells'] for r in rows])),
                'purity_mean': float(np.mean([r['purity'] for r in rows])),
                'pairwise_overlap_pct': float(np.mean([r['pairwise_overlap_pct'] for r in rows])),
            }
            rep['runs']['rho%.2f_ov%.2f' % (rho, ovf)] = d
            print('%6.2f %8.2f %8.3f %9.3f %8.2f %9.3f %10.2f %9.1f %9.3f %9.1f'
                  % (rho, ovf, d['union_connected_frac'], d['all_targets_connected_frac'],
                     d['holes_mean'], d['holefree_frac'], d['target_spread_mean'],
                     d['target_mean'], d['visible_mean'], d['pairwise_overlap_pct']))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'union_split_audit.json').write_text(json.dumps(rep, indent=2))
    print('\npass criteria: union=1 ~1.0 | targets=1 = 1.0 | holes ~0 |'
          ' spread <=2 | visible 0.25-0.35')
    print('reference: I-JEPA rectangles overlap 23.9%% of a block')
    print('wrote %s' % (a.out / 'union_split_audit.json'))


if __name__ == '__main__':
    main()
