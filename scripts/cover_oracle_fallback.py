"""Does an oracle fallback rescue COVER's zero-anatomy encoder views?

Rule under test: generate COVER normally; if a slice ends up with NO anatomy in
the encoder context after the stock `prefix` crop, use that slice's ORACLE
(`anatomical_prior`) mask instead.

The decisive question is not COVER's blank rate or oracle's blank rate
separately, but the CONDITIONAL one: of the slices COVER blanks, how many does
oracle ALSO blank? Both fail when the retina sits low in the frame, so the
failures may be correlated and the fallback may rescue nothing.

CAVEAT: the crop length `min_len` is a batch-level quantity. Swapping one
slice's mask for a different-length one would change `min_len` for everyone.
This script therefore measures each arm's masks under its OWN min_len and then
substitutes per slice. That is an upper bound on how well the fallback can do,
not a production-exact simulation.
"""
from __future__ import annotations

import argparse
import json
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
from src.transforms import make_paired_transforms                 # noqa: E402

CROP, PATCH, GRID = 256, 16, 16
NP = GRID * GRID
BASE = dict(input_size=(CROP, CROP), patch_size=PATCH,
            enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
            allow_overlap=False)
FLOORS = [0.15, 0.20, 0.25, 0.30, 0.35]   # default; override with --floors


def cover_cfg(floor):
    return dict(mode="mirage_cover", T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", mirage_occupancy_threshold=0.25,
                anatomy_tau=0.10, cover_leave_frac=floor,
                cover_min_visible_frac=floor, cover_min_visible_cells=4,
                cover_fill="random_legal", enc_truncate="prefix")


def oracle_cfg():
    return dict(mode="anatomical_prior", T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", mirage_occupancy_threshold=0.25,
                anatomy_tau=0.10, enc_truncate="prefix")


def sets(me, mp, b):
    c = np.zeros(NP, bool); c[me[0][b].numpy()] = True
    t = np.zeros(NP, bool)
    for m in mp:
        t[m[b].numpy()] = True
    return c, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--floors", nargs="+", type=float, default=None)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\arm_stats")
    args = ap.parse_args()

    global FLOORS
    if args.floors:
        FLOORS = list(args.floors)

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
    for f in FLOORS:
        g = CurriculumMaskGenerator(**BASE, curriculum_cfg=cover_cfg(f))
        g.set_epoch(args.epoch, 100); gens[f] = g
    orc = CurriculumMaskGenerator(**BASE, curriculum_cfg=oracle_cfg())
    orc.set_epoch(args.epoch, 100)

    K = ["ctx", "ctx_anat", "pct_ctx_anat", "pct_anat_vis",
         "tgt", "tgt_anat", "pct_tgt_anat", "pct_anat_hid"]
    agg = {f: {k: 0.0 for k in K} for f in FLOORS}
    cnt = {f: dict(n=0, cover_zero=0, both_zero=0, rescued=0) for f in FLOORS}
    B = args.batch_size

    for bi in range(args.batches):
        start = bi * B
        items = [ds[i] for i in range(start, start + B)]
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        anat = (guides[:, 0].reshape(B, -1).numpy() >= 0.25)

        random.seed(11 + start); np.random.seed(11 + start)
        torch.manual_seed(11 + start)
        o_me, o_mp = orc.generate(batch_size=B, imgs_cpu=imgs,
                                  guide_grids=guides, guide_valid=valid)
        cov = {}
        for f in FLOORS:
            random.seed(11 + start); np.random.seed(11 + start)
            torch.manual_seed(11 + start)
            cov[f] = gens[f].generate(batch_size=B, imgs_cpu=imgs,
                                      guide_grids=guides, guide_valid=valid)

        for b in range(B):
            a = anat[b]; na = int(a.sum())
            if na == 0:
                continue
            oc, ot = sets(o_me, o_mp, b)
            o_blank = int((a & oc).sum()) == 0
            for f in FLOORS:
                c, t = sets(cov[f][0], cov[f][1], b)
                blank = int((a & c).sum()) == 0
                d, k = agg[f], cnt[f]
                k["n"] += 1
                if blank:
                    k["cover_zero"] += 1
                    if o_blank:
                        k["both_zero"] += 1
                    else:
                        k["rescued"] += 1
                    c, t = oc, ot          # apply the fallback
                ca, ta = int((a & c).sum()), int((a & t).sum())
                d["ctx"] += int(c.sum()); d["ctx_anat"] += ca
                d["pct_ctx_anat"] += 100.0 * ca / max(int(c.sum()), 1)
                d["pct_anat_vis"] += 100.0 * ca / na
                d["tgt"] += int(t.sum()); d["tgt_anat"] += ta
                d["pct_tgt_anat"] += 100.0 * ta / max(int(t.sum()), 1)
                d["pct_anat_hid"] += 100.0 * ta / na
        print(f"  batch {bi + 1}/{args.batches}", flush=True)

    print("\nFALLBACK DIAGNOSTIC (does oracle rescue the slices COVER blanks?)")
    print(f"{'floor':>6s} {'COVER zero%':>12s} {'oracle ALSO':>12s} "
          f"{'RESCUED':>9s} {'residual zero%':>15s}")
    rows = {}
    for f in FLOORS:
        k = cnt[f]; n = max(k["n"], 1)
        resid = 100.0 * k["both_zero"] / n
        bz = 100.0 * k["both_zero"] / max(k["cover_zero"], 1)
        print(f"{f:6.2f} {100.0*k['cover_zero']/n:11.2f}% {bz:11.1f}% "
              f"{100.0*k['rescued']/n:8.2f}% {resid:14.2f}%")
        rows[str(f)] = dict(cover_zero_pct=100.0*k["cover_zero"]/n,
                            oracle_also_blanks_pct=bz,
                            rescued_pct=100.0*k["rescued"]/n,
                            residual_zero_pct=resid, n=k["n"])

    print("\nMETRICS AFTER FALLBACK")
    hdr = (f"{'floor':>6s} {'ctx':>6s} {'ctxAnat':>8s} {'ctx%anat':>9s} "
           f"{'anatVis%':>9s} {'ZERO%':>7s} {'tgt':>6s} {'tgtAnat':>8s} "
           f"{'tgt%anat':>9s} {'anatHid%':>9s}")
    print(hdr); print("-" * len(hdr))
    for f in FLOORS:
        d = agg[f]; n = max(cnt[f]["n"], 1)
        r = {k: d[k] / n for k in K}
        r["residual_zero"] = rows[str(f)]["residual_zero_pct"]
        rows[str(f)].update(r)
        print(f"{f:6.2f} {r['ctx']:6.1f} {r['ctx_anat']:8.1f} "
              f"{r['pct_ctx_anat']:8.1f}% {r['pct_anat_vis']:8.1f}% "
              f"{r['residual_zero']:6.2f}% {r['tgt']:6.1f} {r['tgt_anat']:8.1f} "
              f"{r['pct_tgt_anat']:8.1f}% {r['pct_anat_hid']:8.1f}%")

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "cover_oracle_fallback.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote", out / "cover_oracle_fallback.json")


if __name__ == "__main__":
    main()
