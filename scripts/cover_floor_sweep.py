"""Full COVER floor sweep under the stock `prefix` crop.

Two things the earlier sweep got wrong:

1. Too few slices. At n=1534 and p~0.08 the SE is 0.7%, which swamped the
   ~0.35-point-per-step effect and produced a non-monotonic column.
2. Independent CIs. Every floor is evaluated on the SAME slices with the SAME
   batch composition, so the right comparison is PAIRED (McNemar), which
   removes the between-slice variance entirely.

Monotonicity is the built-in sanity check: raising the floor protects more
anatomy, so the blank rate must not rise. Any rise is noise.
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

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator          # noqa: E402
from src.masks.multiblock import MaskCollator                     # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402

CROP, PATCH, GRID = 256, 16, 16
NP = GRID * GRID
BASE = dict(input_size=(CROP, CROP), patch_size=PATCH,
            enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
            allow_overlap=False)


def cover_cfg(floor):
    return dict(mode="mirage_cover", T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", mirage_occupancy_threshold=0.25,
                anatomy_tau=0.10, cover_leave_frac=floor,
                cover_min_visible_frac=floor, cover_min_visible_cells=4,
                cover_fill="random_legal", enc_truncate="prefix")


def ref_cfg(mode):
    return dict(mode=mode, T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", mirage_occupancy_threshold=0.25,
                anatomy_tau=0.10, enc_truncate="prefix")


# reference arms shown alongside the sweep, all on the SAME slices
REFS = ["random", "oracle", "envelope"]
REF_MODE = {"oracle": "anatomical_prior", "envelope": "mirage_envelope"}


K = ["ctx", "ctx_anat", "pct_ctx_anat", "pct_anat_vis",
     "tgt", "tgt_anat", "pct_tgt_anat", "pct_anat_hid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--start", type=float, default=0.15)
    ap.add_argument("--stop", type=float, default=0.30)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\arm_stats_sweep")
    args = ap.parse_args()

    n_f = int(round((args.stop - args.start) / args.step)) + 1
    floors = [round(args.start + i * args.step, 4) for i in range(n_f)]
    print(f"floors: {floors}")

    paired = make_paired_transforms(
        crop_size=CROP, crop_scale=(0.3, 1.0), gaussian_blur=False,
        horizontal_flip=False, color_distortion=False, color_jitter=0.0)
    ds = GuidedOCTSliceDataset(
        data_dir=r"D:\jepa_phase0\fairvision-glaucoma\data\Training",
        guide_dir=(r"C:\jepa_data\mirage_soft_guides"
                   r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy\Training"),
        num_slices=100, slice_size=CROP, transform=paired, patch_size=PATCH,
        dilate_patches=0, occupancy_threshold=0.25,
        slice_cache=r"C:\jepa_data\slice_cache\Training")

    gens = {}
    for f in floors:
        g = CurriculumMaskGenerator(**BASE, curriculum_cfg=cover_cfg(f))
        g.set_epoch(args.epoch, 100)
        gens[f] = g
    stock = MaskCollator(**BASE)
    for name in REFS:
        if name == "random":
            continue
        g = CurriculumMaskGenerator(**BASE, curriculum_cfg=ref_cfg(REF_MODE[name]))
        g.set_epoch(args.epoch, 100)
        gens[name] = g

    keys = floors + REFS
    agg = {f: {k: 0.0 for k in K} for f in keys}
    blank = {f: [] for f in keys}      # per-slice 0/1, aligned across arms
    B = args.batch_size

    for bi in range(args.batches):
        start = bi * B
        items = [ds[i] for i in range(start, start + B)]
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        anat = (guides[:, 0].reshape(B, -1).numpy() >= 0.25)

        outs = {}
        for f in keys:
            random.seed(11 + start); np.random.seed(11 + start)
            torch.manual_seed(11 + start)
            if f == "random":
                _, me, mp = stock([imgs[i] for i in range(B)])
                outs[f] = (me, mp)
            else:
                outs[f] = gens[f].generate(batch_size=B, imgs_cpu=imgs,
                                           guide_grids=guides, guide_valid=valid)

        for b in range(B):
            a = anat[b]; na = int(a.sum())
            if na == 0:
                continue
            for f in keys:
                me, mp = outs[f]
                c = np.zeros(NP, bool); c[me[0][b].numpy()] = True
                t = np.zeros(NP, bool)
                for m in mp:
                    t[m[b].numpy()] = True
                ca, ta = int((a & c).sum()), int((a & t).sum())
                d = agg[f]
                d["ctx"] += int(c.sum()); d["ctx_anat"] += ca
                d["pct_ctx_anat"] += 100.0 * ca / max(int(c.sum()), 1)
                d["pct_anat_vis"] += 100.0 * ca / na
                d["tgt"] += int(t.sum()); d["tgt_anat"] += ta
                d["pct_tgt_anat"] += 100.0 * ta / max(int(t.sum()), 1)
                d["pct_anat_hid"] += 100.0 * ta / na
                blank[f].append(1 if ca == 0 else 0)
        if (bi + 1) % 8 == 0:
            print(f"  batch {bi + 1}/{args.batches}", flush=True)

    bl = {f: np.array(blank[f], dtype=np.int8) for f in keys}
    n = len(bl[keys[0]])
    se_ind = {f: 100.0 * math.sqrt(max(bl[f].mean(), 1e-9)
                                   * (1 - bl[f].mean()) / n) for f in keys}

    def line(label, f):
        d = agg[f]
        r = {k: d[k] / n for k in K}
        z = 100.0 * bl[f].mean()
        rows[str(f)] = dict(r, zero_pct=z, se=se_ind[f], n=n)
        print(f"{label:>10s} {r['ctx']:6.1f} {r['ctx_anat']:8.1f} "
              f"{r['pct_ctx_anat']:8.1f}% {r['pct_anat_vis']:8.1f}% "
              f"{z:6.2f}% {se_ind[f]:5.2f} {r['tgt']:6.1f} "
              f"{r['tgt_anat']:8.1f} {r['pct_tgt_anat']:8.1f}% "
              f"{r['pct_anat_hid']:8.1f}%")

    hdr = (f"{'arm':>10s} {'ctx':>6s} {'ctxAnat':>8s} {'ctx%anat':>9s} "
           f"{'anatVis%':>9s} {'ZERO%':>7s} {'+-SE':>6s} {'tgt':>6s} "
           f"{'tgtAnat':>8s} {'tgt%anat':>9s} {'anatHid%':>9s}")
    rows = {}
    print("\n" + hdr); print("-" * len(hdr))
    print("--- REFERENCE ARMS (same slices, same prefix crop) ---")
    for name in REFS:
        line(name, name)
    print("--- COVER floor sweep ---")
    for f in floors:
        line(f"{f:.2f}", f)

    ref = floors[0]
    print(f"\nPAIRED comparison vs floor {ref:.2f} (McNemar, same slices)")
    print(f"{'floor':>6s} {'fixed':>7s} {'broke':>7s} {'net delta':>11s} "
           f"{'paired SE':>10s} {'z':>7s}")
    for f in floors[1:]:
        a, b = bl[ref], bl[f]
        fixed = int(((a == 1) & (b == 0)).sum())    # ref blanked, this did not
        broke = int(((a == 0) & (b == 1)).sum())
        disc = fixed + broke
        delta = 100.0 * (b.mean() - a.mean())
        se = 100.0 * math.sqrt(max(disc, 1)) / n
        z = delta / se if se > 0 else 0.0
        rows[str(f)]["paired"] = dict(fixed=fixed, broke=broke,
                                      delta=delta, se=se, z=z)
        print(f"{f:6.2f} {fixed:7d} {broke:7d} {delta:10.2f}% "
              f"{se:9.2f}% {z:7.1f}")

    print("\nPAIRED vs ENVELOPE (is this floor better or worse than the "
          "best-AUC baseline?)")
    print(f"{'floor':>6s} {'net delta':>11s} {'paired SE':>10s} {'z':>7s}  verdict")
    for f in floors:
        a, b = bl["envelope"], bl[f]
        disc = int(((a == 1) & (b == 0)).sum() + ((a == 0) & (b == 1)).sum())
        delta = 100.0 * (b.mean() - a.mean())
        se = 100.0 * math.sqrt(max(disc, 1)) / n
        z = delta / se if se > 0 else 0.0
        verdict = ("BETTER than envelope" if z < -2 else
                   "worse than envelope" if z > 2 else "same as envelope")
        print(f"{f:6.2f} {delta:10.2f}% {se:9.2f}% {z:7.1f}  {verdict}")

    zs = [100.0 * bl[f].mean() for f in floors]
    rises = [(floors[i], floors[i + 1]) for i in range(len(floors) - 1)
             if zs[i + 1] > zs[i] + 1e-9]
    print(f"\nmonotonicity: {len(rises)} rises out of {len(floors) - 1} steps "
          f"(0 expected if noise is beaten)")
    if rises:
        print("  rises at:", ", ".join(f"{a:.2f}->{b:.2f}" for a, b in rises))
    print(f"slices per floor: {n}")

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "cover_floor_sweep.json").write_text(json.dumps(rows, indent=2))
    print("wrote", out / "cover_floor_sweep.json")


if __name__ == "__main__":
    main()
