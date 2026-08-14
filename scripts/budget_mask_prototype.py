#!/usr/bin/env python3
"""Prototype + audit of BUDGET masking: hide all anatomy, grow into background.

Motivation (measured in docs/experiments/masking/mask_composition_report.md):
the current mirage_anatomy arm hides only 21% of the grid and leaves a 162-token
context that is 93% background.  Every arm that works hides 40-46%.

This sampler fixes the hidden budget directly:

  1. support   S = {a > tau}, a = P_inner + P_choroid           (all anatomy)
  2. partition S into n connected parts (geodesic_partition + rebalance)
  3. GROW each part outward into background by competitive multi-source BFS,
     smallest-part-first, until every part holds exactly k cells
  4. bridge diagonals so each part is 4-connected

Step 3 is the new bit: the production `_grow_cells_within` is confined to the
anatomy component (`if comp[nr, nc]`) and `grow_components_fixed_cells` clamps
n_cells to the support size, so neither can exceed the anatomy.  Growing past it
is what buys the hidden budget, and it gives every target a background collar --
the model has to learn where the retina ENDS, not just what its interior looks
like.

Nothing here touches src/.  This is a measurement prototype.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from src.masks.anatomy import (  # noqa: E402
    geodesic_partition, make_connected, n_components4, rebalance,
)

GRID = 16
NPATCH = GRID * GRID


def _nbrs4(r, c, h=GRID, w=GRID):
    if r > 0:
        yield r - 1, c
    if r < h - 1:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c < w - 1:
        yield r, c + 1


def grow_parts_to_budget(parts, k, score=None, rng=None):
    """Competitive region growing: expand disjoint parts to exactly k cells.

    Smallest part moves first, so the parts stay balanced and none is starved
    by a neighbour that happened to be seeded in open space.

    Returns (parts, n_short) where n_short counts parts that could not reach k
    because the grid ran out (all cells claimed).
    """
    rng = rng or random
    n = len(parts)
    owner = -np.ones((GRID, GRID), dtype=int)
    for i, p in enumerate(parts):
        owner[p] = i
    counts = [int(p.sum()) for p in parts]

    def frontier(i):
        out = []
        for (r, c) in np.argwhere(owner == i):
            for nr, nc in _nbrs4(r, c):
                if owner[nr, nc] == -1:
                    out.append((nr, nc))
        return out

    while True:
        hungry = [i for i in range(n) if counts[i] < k]
        if not hungry:
            break
        hungry.sort(key=lambda i: counts[i])
        progressed = False
        for i in hungry:
            f = frontier(i)
            if not f:
                continue
            if score is not None:
                # hug the anatomy first, then spill outward
                best = max(f, key=lambda rc: (score[rc], rng.random()))
            else:
                best = f[rng.randrange(len(f))]
            owner[best] = i
            counts[i] += 1
            progressed = True
            if counts[i] >= k:
                continue
        if not progressed:
            break

    grown = [(owner == i) for i in range(n)]
    n_short = sum(1 for i in range(n) if counts[i] < k)
    return grown, n_short


def build_budget_targets(class_scores, n=4, k=57, tau=0.10, rng=None):
    """Full sampler.  Returns (parts, info-dict)."""
    rng = rng or random
    a = np.asarray(class_scores[0], float)
    for s in class_scores[1:]:
        a = a + np.asarray(s, float)
    S = a > tau

    info = dict(anat_cells=int(S.sum()), fallback=False, reason="",
                n_short=0, parts_before=0, conn_ok=True, union=0,
                grew_into_bg=0)

    # --- fallback 1: no usable anatomy -> random rectangles -----------------
    if not S.any():
        info.update(fallback=True, reason="no_anatomy")
        return _random_rect_targets(n, k, rng), info

    # --- partition the support ---------------------------------------------
    try:
        parts = rebalance(geodesic_partition(S, n), 1.25)
    except Exception:                                    # noqa: BLE001
        parts = []
    parts = [p for p in parts if p.any()]
    info["parts_before"] = len(parts)

    # --- fallback 2: cannot split into n connected seeds --------------------
    if len(parts) == 0:
        info.update(fallback=True, reason="partition_empty")
        return _random_rect_targets(n, k, rng), info

    # Too few seeds: seed the missing ones in background, away from the rest.
    while len(parts) < n:
        taken = np.logical_or.reduce(parts)
        free = np.argwhere(~taken)
        if not len(free):
            break
        r, c = free[rng.randrange(len(free))]
        m = np.zeros((GRID, GRID), bool)
        m[r, c] = True
        parts.append(m)
        info["reason"] = (info["reason"] + "|seeded_bg").strip("|")

    # --- shrink case: anatomy already exceeds the budget --------------------
    over = [i for i, p in enumerate(parts) if int(p.sum()) > k]
    for i in over:
        from src.masks.anatomy import shrink_to_k_connected
        parts[i] = shrink_to_k_connected(parts[i], k, a)

    before = sum(int(p.sum()) for p in parts)
    parts, n_short = grow_parts_to_budget(parts, k, score=a, rng=rng)
    info["n_short"] = n_short
    info["grew_into_bg"] = int(sum(int(p.sum()) for p in parts) - before)

    # --- 4-connectivity -----------------------------------------------------
    union = np.logical_or.reduce(parts)
    fixed = []
    for p in parts:
        if not p.any():
            fixed.append(p)
            continue
        if n_components4(p) > 1:
            # make_connected preserves the cell count (bridge then trim back),
            # so the budget survives the connectivity fix.
            p = make_connected(p, a, forbid=union & ~p)
        fixed.append(p)
    parts = fixed
    info["conn_ok"] = all((not p.any()) or n_components4(p) == 1 for p in parts)
    info["union"] = int(np.logical_or.reduce(parts).sum())
    return parts, info


def _random_rect_targets(n, k, rng):
    """Fallback: n random near-square blocks of ~k cells, disjoint-ish."""
    side = max(1, int(round(k ** 0.5)))
    out, taken = [], np.zeros((GRID, GRID), bool)
    for _ in range(n):
        bh = min(GRID, side)
        bw = min(GRID, max(1, int(round(k / bh))))
        top = rng.randrange(0, max(1, GRID - bh + 1))
        left = rng.randrange(0, max(1, GRID - bw + 1))
        m = np.zeros((GRID, GRID), bool)
        m[top:top + bh, left:left + bw] = True
        m &= ~taken
        taken |= m
        out.append(m)
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default=(
        r"D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz"))
    ap.add_argument("--k", type=int, nargs="+", default=[16, 30, 40, 57],
                    help="cells per target block to sweep")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\budget_mask_audit.json")
    args = ap.parse_args()

    per = np.load(args.grids)["per"]          # (N, 2, 16, 16)
    N = min(args.limit, per.shape[0])
    print(f"slices: {N}", flush=True)

    results = {}
    for k in args.k:
        rng = random.Random(args.seed)
        rows = []
        for i in range(N):
            parts, info = build_budget_targets(
                [per[i, 0], per[i, 1]], n=args.n, k=k, tau=args.tau, rng=rng)
            sizes = [int(p.sum()) for p in parts]
            info["sizes_min"] = min(sizes) if sizes else 0
            info["sizes_mean"] = float(np.mean(sizes)) if sizes else 0.0
            info["hit_budget"] = all(s == k for s in sizes)
            rows.append(info)
        nf = sum(r["fallback"] for r in rows)
        short = sum(r["n_short"] > 0 for r in rows)
        hit = sum(r["hit_budget"] for r in rows)
        conn = sum(not r["conn_ok"] for r in rows)
        seeded = sum("seeded_bg" in r["reason"] for r in rows)
        union = np.array([r["union"] for r in rows], float)
        anat = np.array([r["anat_cells"] for r in rows], float)
        bg = np.array([r["grew_into_bg"] for r in rows], float)
        results[str(k)] = dict(
            k=k, n=args.n, slices=N,
            pct_fallback=100.0 * nf / N,
            pct_seeded_bg=100.0 * seeded / N,
            pct_short=100.0 * short / N,
            pct_hit_budget=100.0 * hit / N,
            pct_not_4conn=100.0 * conn / N,
            union_mean=float(union.mean()),
            union_frac_of_grid=float(union.mean() / NPATCH),
            context_est=float(NPATCH - union.mean()),
            context_frac_est=float((NPATCH - union.mean()) / NPATCH),
            anat_cells_mean=float(anat.mean()),
            grown_into_bg_mean=float(bg.mean()),
        )
        r = results[str(k)]
        print(f"k={k:3d} | union {r['union_mean']:6.1f} "
              f"({r['union_frac_of_grid']*100:4.1f}%) | ctx~{r['context_est']:6.1f} "
              f"({r['context_frac_est']*100:4.1f}%) | hit {r['pct_hit_budget']:5.1f}% "
              f"| fallback {r['pct_fallback']:4.1f}% | short {r['pct_short']:5.1f}% "
              f"| not4conn {r['pct_not_4conn']:4.1f}%", flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
