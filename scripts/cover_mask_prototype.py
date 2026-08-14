#!/usr/bin/env python3
"""COVER masking: stock I-JEPA rectangles, greedily placed to hide the anatomy.

Design (per user spec):
  * keep the standard shape          -> axis-aligned rectangles, not blobs
  * keep the standard count          -> npred = 4
  * keep the standard size           -> pred_mask_scale (0.15, 0.2), aspect (0.75, 1.5)
  * overlap between blocks ALLOWED
  * covering background ALLOWED
  * place blocks greedily so that nearly all anatomy is hidden, leaving only a
    small visible remainder (``leave_frac``, default 0.10 - the same 10% idea as
    ``anatomy_mass_cap = 0.90``)
  * once the anatomy is covered, any remaining blocks are placed as TRANSITION
    blocks straddling the tissue/background boundary, so they still carry signal
    instead of landing on empty vitreous.

Only PLACEMENT changes versus the envelope baseline.  Shape, count and size are
identical to stock I-JEPA, which is what makes the comparison clean.

Greedy is exact here: a 16x16 grid has <=256 candidate positions per block, so
we can evaluate every one rather than rejection-sample.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

GRID = 16
NPATCH = GRID * GRID
PRED_SCALE = (0.15, 0.20)
ASPECT = (0.75, 1.5)
TAU = 0.10          # "meaningfully anatomy" on the MIRAGE soft score


def sample_block_size(gen, scale=PRED_SCALE, aspect=ASPECT):
    """Identical policy to src/masks/multiblock.MaskCollator._sample_block_size."""
    _rand = torch.rand(1, generator=gen).item()
    min_s, max_s = scale
    mask_scale = min_s + _rand * (max_s - min_s)
    max_keep = int(NPATCH * mask_scale)
    _rand = torch.rand(1, generator=gen).item()
    min_ar, max_ar = aspect
    ar = min_ar + _rand * (max_ar - min_ar)
    h = int(round(math.sqrt(max_keep * ar)))
    w = int(round(math.sqrt(max_keep / ar)))
    h = min(max(h, 1), GRID)
    w = min(max(w, 1), GRID)
    return h, w


def _integral(x):
    return np.pad(np.cumsum(np.cumsum(x, 0), 1), ((1, 0), (1, 0)))


def _window_sums(ii, bh, bw):
    """Sum over every bh x bw window, via the integral image."""
    return (ii[bh:, bw:] - ii[:-bh, bw:] - ii[bh:, :-bw] + ii[:-bh, :-bw])


def cover_targets(class_scores, n=4, leave_frac=0.10, min_visible_frac=0.15,
                  min_visible_cells=4, tau=TAU, gen=None, rng=None,
                  transition=True):
    """Return (list_of_bool_masks, info).

    Three knobs control how much anatomy is hidden:

      leave_frac         SOFT target.  Stop spending blocks on coverage once
                         (1 - leave_frac) of the anatomy mass is hidden.
      min_visible_frac   HARD floor on anatomy MASS that must stay visible.
      min_visible_cells  HARD floor on the NUMBER of anatomy cells that must
                         stay visible.  The mass floor alone is not enough on
                         sparse edge slices: a slice with 7 anatomy cells
                         satisfies a 10% mass floor with a single cell, which
                         is too weak an anchor for the encoder.  Capped at the
                         anatomy size so slices smaller than the floor are not
                         made unmaskable.

    Mirrors the existing ``mirage_min_retina_visible`` guard: the encoder must
    always retain some tissue, otherwise the predictor is reconstructing the
    retina from pure background and can only fall back on a positional prior.

    When the floors block every candidate, the block becomes a TRANSITION block
    straddling the tissue/background boundary instead.
    """
    rng = rng or random
    gen = gen or torch.Generator()

    a = np.asarray(class_scores[0], float)
    for s in class_scores[1:]:
        a = a + np.asarray(s, float)
    S = a > tau                                  # anatomy support
    total_mass = float(a[S].sum())
    n_anat = int(S.sum())
    info = dict(anat_cells=n_anat, fallback=False,
                n_cover=0, n_transition=0, n_random=0,
                covered_frac=0.0, covered_cells=0, floor_blocked=0)

    sizes = [sample_block_size(gen) for _ in range(n)]

    if not S.any() or total_mass <= 0:
        info["fallback"] = True
        masks = []
        for (bh, bw) in sizes:
            top = rng.randrange(0, GRID - bh + 1)
            left = rng.randrange(0, GRID - bw + 1)
            m = np.zeros((GRID, GRID), bool)
            m[top:top + bh, left:left + bw] = True
            masks.append(m)
            info["n_random"] += 1
        union = np.logical_or.reduce(masks)
        info.update(covered_frac=0.0, covered_cells=0, hit_target=False,
                    visible_frac=1.0, visible_cells=int(S.sum()),
                    floor_ok=True,
                    union=int(union.sum()),
                    slots=int(sum(int(m.sum()) for m in masks)))
        return masks, info

    remaining = a * S                            # uncovered anatomy mass
    target_mass = (1.0 - leave_frac) * total_mass
    floor_mass = min_visible_frac * total_mass   # mass that must stay visible
    # never demand more visible cells than the slice actually has
    floor_cells = min(int(min_visible_cells), n_anat)
    covered_mass = 0.0
    covered = np.zeros((GRID, GRID), bool)
    masks = []

    def legal_mask(bh, bw):
        """Windows that respect BOTH the mass floor and the cell floor."""
        gains = _window_sums(_integral(remaining), bh, bw)
        ok = gains <= float(remaining.sum() - floor_mass) + 1e-9
        # cell floor: anatomy cells still visible after adding this window
        uncov = (S & ~covered).astype(float)
        newly = _window_sums(_integral(uncov), bh, bw)
        vis_after = float(uncov.sum()) - newly
        ok &= vis_after >= floor_cells - 1e-9
        return gains, ok

    for (bh, bw) in sizes:
        placed = False
        if covered_mass < target_mass and remaining.sum() > floor_mass:
            gains, ok = legal_mask(bh, bw)
            if ok.any():
                g = np.where(ok, gains, -np.inf)
                best = float(g.max())
                if best > 0:
                    cand = np.argwhere(g >= best - 1e-9)
                    top, left = cand[rng.randrange(len(cand))]
                    m = np.zeros((GRID, GRID), bool)
                    m[top:top + bh, left:left + bw] = True
                    covered_mass += float(remaining[m].sum())
                    remaining[m] = 0.0
                    covered |= m
                    masks.append(m)
                    info["n_cover"] += 1
                    placed = True
            else:
                info["floor_blocked"] += 1
        if placed:
            continue

        if transition:
            # ---- TRANSITION: straddle the tissue/background boundary --------
            ii_a = _integral(S.astype(float))
            n_a = _window_sums(ii_a, bh, bw)
            n_bg = (bh * bw) - n_a
            balance = np.minimum(n_a, n_bg).astype(float)
            _, ok = legal_mask(bh, bw)
            bal = np.where(ok, balance, -np.inf)
            best = float(bal.max())
            if best > 0:
                cand = np.argwhere(bal >= best - 1e-9)
                top, left = cand[rng.randrange(len(cand))]
                info["n_transition"] += 1
            else:
                # nothing legal touches tissue: place clear of the anatomy
                free = _window_sums(_integral(S.astype(float)), bh, bw)
                cand = np.argwhere(free <= float(free.min()) + 1e-9)
                top, left = cand[rng.randrange(len(cand))]
                info["n_random"] += 1
        else:
            top = rng.randrange(0, GRID - bh + 1)
            left = rng.randrange(0, GRID - bw + 1)
            info["n_random"] += 1

        m = np.zeros((GRID, GRID), bool)
        m[top:top + bh, left:left + bw] = True
        covered_mass += float(remaining[m].sum())
        remaining[m] = 0.0
        covered |= m
        masks.append(m)

    union = np.logical_or.reduce(masks)
    info["covered_frac"] = float(a[S & union].sum() / total_mass)
    info["visible_frac"] = 1.0 - info["covered_frac"]
    info["visible_cells"] = int((S & ~union).sum())
    info["covered_cells"] = int((S & union).sum())
    info["hit_target"] = bool(info["covered_frac"] >= 1.0 - leave_frac - 1e-9)
    info["floor_ok"] = bool(
        info["visible_frac"] >= min_visible_frac - 1e-9
        and info["visible_cells"] >= floor_cells)
    info["union"] = int(union.sum())
    info["slots"] = int(sum(int(m.sum()) for m in masks))
    return masks, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default=(
        r"D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz"))
    ap.add_argument("--leave", type=float, nargs="+", default=[0.10])
    ap.add_argument("--min_visible", type=float, default=0.15)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\cover_mask_audit.json")
    args = ap.parse_args()

    per = np.load(args.grids)["per"]
    N = min(args.limit, per.shape[0])
    res = {}
    for lf in args.leave:
        rng = random.Random(args.seed)
        gen = torch.Generator()
        gen.manual_seed(args.seed)
        rows = []
        for i in range(N):
            _, info = cover_targets([per[i, 0], per[i, 1]], n=args.n,
                                    leave_frac=lf,
                                    min_visible_frac=args.min_visible,
                                    gen=gen, rng=rng)
            rows.append(info)
        cf = np.array([r["covered_frac"] for r in rows])
        vf = np.array([r["visible_frac"] for r in rows])
        un = np.array([r["union"] for r in rows], float)
        sl = np.array([r["slots"] for r in rows], float)
        res[str(lf)] = dict(
            leave_frac=lf, min_visible_frac=args.min_visible, slices=N,
            pct_hit=100 * float(np.mean([r["hit_target"] for r in rows])),
            pct_floor_ok=100 * float(np.mean([r["floor_ok"] for r in rows])),
            pct_zero_visible=100 * float(np.mean(vf <= 1e-9)),
            pct_fallback=100 * float(np.mean([r["fallback"] for r in rows])),
            covered_frac_mean=float(cf.mean()),
            covered_frac_p95=float(np.percentile(cf, 95)),
            covered_frac_max=float(cf.max()),
            visible_frac_mean=float(vf.mean()),
            visible_frac_p05=float(np.percentile(vf, 5)),
            visible_frac_min=float(vf.min()),
            visible_cells_mean=float(np.mean([r["visible_cells"] for r in rows])),
            union_mean=float(un.mean()), union_frac=float(un.mean() / NPATCH),
            slots_mean=float(sl.mean()),
            overlap_mean=float((sl - un).mean()),
            n_cover_mean=float(np.mean([r["n_cover"] for r in rows])),
            n_transition_mean=float(np.mean([r["n_transition"] for r in rows])),
            n_random_mean=float(np.mean([r["n_random"] for r in rows])),
            anat_cells_mean=float(np.mean([r["anat_cells"] for r in rows])),
        )
        r = res[str(lf)]
        print(f"leave={lf:.2f} floor={args.min_visible:.2f} | hidden "
              f"{r['covered_frac_mean']*100:5.1f}% (max {r['covered_frac_max']*100:5.1f}%) "
              f"| visible {r['visible_frac_mean']*100:5.1f}% "
              f"(min {r['visible_frac_min']*100:4.1f}%, {r['visible_cells_mean']:4.1f} cells) "
              f"| zero-visible {r['pct_zero_visible']:4.1f}% "
              f"| union {r['union_mean']:6.1f} ({r['union_frac']*100:4.1f}%) "
              f"| c/t/r {r['n_cover_mean']:.2f}/{r['n_transition_mean']:.2f}/"
              f"{r['n_random_mean']:.2f}", flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
