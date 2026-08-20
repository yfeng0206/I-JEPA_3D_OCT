"""Is zero-anatomy failure a property of the SLICE, or of the random draw?

If it is the slice, we can tag a fixed list offline and route those to oracle.
If it is the draw, no tag list can work -- the same slice blanks in one epoch
and not the next, because RandomResizedCrop, target placement, and the batch
composition that sets `min_len` all change every epoch.

Method: hold a fixed pool of slices; repeat R times, reshuffling the pool into
fresh batches each time so both the crop draw and the batch companions change.
Record how often each individual slice blanks.
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


def cover_cfg(floor):
    return dict(mode="mirage_cover", T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", mirage_occupancy_threshold=0.25,
                anatomy_tau=0.10, cover_leave_frac=floor,
                cover_min_visible_frac=floor, cover_min_visible_cells=4,
                cover_fill="random_legal", enc_truncate="prefix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.15)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\arm_stats")
    args = ap.parse_args()

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

    gen = CurriculumMaskGenerator(**BASE, curriculum_cfg=cover_cfg(args.floor))
    gen.set_epoch(50, 100)

    ids = list(range(args.pool))
    blanks = {i: 0 for i in ids}
    trials = {i: 0 for i in ids}
    B = args.batch_size

    for rep in range(args.repeats):
        order = ids[:]
        random.Random(1000 + rep).shuffle(order)
        for s in range(0, len(order) - B + 1, B):
            chunk = order[s:s + B]
            items = [ds[i] for i in chunk]
            imgs = torch.stack([it[0] for it in items], 0)
            guides = torch.stack([it[1] for it in items], 0)
            valid = torch.stack([it[2] for it in items], 0)
            anat = (guides[:, 0].reshape(B, -1).numpy() >= 0.25)
            me, _ = gen.generate(batch_size=B, imgs_cpu=imgs,
                                 guide_grids=guides, guide_valid=valid)
            for b, sid in enumerate(chunk):
                a = anat[b]
                if a.sum() == 0:
                    continue
                c = np.zeros(NP, bool); c[me[0][b].numpy()] = True
                trials[sid] += 1
                if int((a & c).sum()) == 0:
                    blanks[sid] += 1
        print(f"  repeat {rep + 1}/{args.repeats}", flush=True)

    rate = {i: blanks[i] / trials[i] for i in ids if trials[i] > 0}
    tot_tr = sum(trials[i] for i in rate)
    tot_bl = sum(blanks[i] for i in rate)
    r = np.array(sorted(rate.values(), reverse=True))
    n = len(r)
    print(f"\nslices={n}  trials/slice~{tot_tr / max(n,1):.1f}  "
          f"overall blank rate={100.0 * tot_bl / max(tot_tr,1):.2f}%")

    print("\nPER-SLICE BLANK FREQUENCY")
    print(f"  never blanks            : {100.0 * (r == 0).sum() / n:5.1f}% of slices")
    print(f"  blanks 1-25% of draws   : {100.0 * ((r > 0) & (r <= .25)).sum() / n:5.1f}%")
    print(f"  blanks 25-50%           : {100.0 * ((r > .25) & (r <= .5)).sum() / n:5.1f}%")
    print(f"  blanks 50-75%           : {100.0 * ((r > .5) & (r <= .75)).sum() / n:5.1f}%")
    print(f"  blanks >75%             : {100.0 * (r > .75).sum() / n:5.1f}%")
    print(f"  ALWAYS blanks           : {100.0 * (r == 1).sum() / n:5.1f}%")

    print("\nCONCENTRATION — if we TAG the worst K% of slices and route to oracle")
    print(f"{'tag K%':>8s} {'blanks covered':>15s} {'residual rate':>14s}")
    order = np.argsort(-np.array([rate[i] for i in ids if trials[i] > 0]))
    vals = np.array([rate[i] for i in ids if trials[i] > 0])
    tr = np.array([trials[i] for i in ids if trials[i] > 0])
    bl = np.array([blanks[i] for i in ids if trials[i] > 0])
    rows = {}
    for K in (5, 10, 15, 20, 30, 50):
        k = max(int(round(n * K / 100.0)), 1)
        idx = order[:k]
        covered = bl[idx].sum()
        resid = (tot_bl - covered) / max(tot_tr, 1)
        print(f"{K:7d}% {100.0 * covered / max(tot_bl,1):14.1f}% "
              f"{100.0 * resid:13.2f}%")
        rows[f"tag_top_{K}pct"] = dict(
            blanks_covered_pct=100.0 * covered / max(tot_bl, 1),
            residual_blank_pct=100.0 * resid)

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "blank_proneness.json").write_text(json.dumps(
        dict(overall_blank_pct=100.0 * tot_bl / max(tot_tr, 1),
             never_pct=100.0 * float((r == 0).sum()) / n,
             always_pct=100.0 * float((r == 1).sum()) / n,
             tagging=rows), indent=2))
    print("\nwrote", out / "blank_proneness.json")


if __name__ == "__main__":
    main()
