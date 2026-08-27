"""Measure what supervision the merged dataset actually gives BELOW the choroid.

Question this answers: after merging, why did the predicted choroid get THICKER,
when GOALS was the only source with real choroid annotation?

A model only learns an upper bound on choroid thickness if some pixel below the
choroid is labelled "not choroid". If that region is `ignore` instead, the loss
is masked there and the gradient is exactly zero -- the band tells the model
where the choroid STARTS but never where it must STOP.

So for every column that carries choroid, we walk below the lowest choroid pixel
and ask what the label says: Elsewhere (a real negative, bounds the layer) or
ignore (no signal at all).

Usage:
  python scripts/choroid_supervision_probe.py --root D:\\jepa_phase0\\mirage-datasets\\MergedV3
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
from PIL import Image

ELSEWHERE, IGNORE, INNER, CHOROID = 0, 1, 128, 255


def probe_mask(arr: np.ndarray) -> dict | None:
    """Per-column stats for one semseg mask. None if the mask has no choroid."""
    ch = arr == CHOROID
    cols = np.where(ch.any(axis=0))[0]
    if cols.size == 0:
        return None

    h = arr.shape[0]
    below_ignore = below_elsewhere = below_total = 0
    thick = np.zeros(cols.size, dtype=np.float64)

    for i, c in enumerate(cols):
        rows = np.where(ch[:, c])[0]
        top, bot = rows[0], rows[-1]
        thick[i] = (bot - top + 1) / h
        under = arr[bot + 1 :, c]
        below_total += under.size
        below_ignore += int((under == IGNORE).sum())
        below_elsewhere += int((under == ELSEWHERE).sum())

    return {
        "cols": int(cols.size),
        "col_cov": cols.size / arr.shape[1],
        "thick": float(thick.mean()),
        "below_total": below_total,
        "below_ignore": below_ignore,
        "below_elsewhere": below_elsewhere,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    seg_dir = args.root / args.split / "semseg"
    files = sorted(seg_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"no masks under {seg_dir}")

    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    thick_by_src: dict[str, list[float]] = collections.defaultdict(list)
    cov_by_src: dict[str, list[float]] = collections.defaultdict(list)

    for f in files:
        src = f.name.split("__", 1)[0]
        arr = np.array(Image.open(f))
        agg[src]["images"] += 1
        st = probe_mask(arr)
        if st is None:
            agg[src]["no_choroid"] += 1
            continue
        agg[src]["with_choroid"] += 1
        agg[src]["below_total"] += st["below_total"]
        agg[src]["below_ignore"] += st["below_ignore"]
        agg[src]["below_elsewhere"] += st["below_elsewhere"]
        thick_by_src[src].append(st["thick"])
        cov_by_src[src].append(st["col_cov"])

    print(f"\ndataset: {args.root.name}  split={args.split}  masks={len(files)}\n")
    hdr = f"{'source':<10} {'imgs':>5} {'w/chor':>7} {'thick':>7} {'colcov':>7} {'below=ignore':>13} {'below=elsewh':>13}"
    print(hdr)
    print("-" * len(hdr))

    tot = collections.Counter()
    for src in sorted(agg):
        a = agg[src]
        bt = a["below_total"]
        ig = a["below_ignore"] / bt if bt else float("nan")
        el = a["below_elsewhere"] / bt if bt else float("nan")
        th = float(np.mean(thick_by_src[src])) if thick_by_src[src] else float("nan")
        cv = float(np.mean(cov_by_src[src])) if cov_by_src[src] else float("nan")
        print(
            f"{src:<10} {a['images']:>5} {a['with_choroid']:>7} {th:>7.4f} {cv:>7.4f} "
            f"{ig:>12.1%} {el:>12.1%}"
        )
        tot.update(a)

    bt = tot["below_total"]
    print("-" * len(hdr))
    print(
        f"{'ALL':<10} {tot['images']:>5} {tot['with_choroid']:>7} {'':>7} {'':>7} "
        f"{tot['below_ignore'] / bt:>12.1%} {tot['below_elsewhere'] / bt:>12.1%}"
    )

    print(
        "\nbelow=elsewhere is the only column that bounds choroid thickness from below.\n"
        "Where below=ignore dominates, the loss is masked and the gradient is zero,\n"
        "so nothing stops the model from extending the choroid downward."
    )


if __name__ == "__main__":
    main()
