"""Full mask statistics for every arm, measured through the production path.

Every column below is generated on the SAME slices with the SAME batch size, so
the batch-minimum encoder crop (`t[:min_len]`) bites identically. Numbers are
POST-crop, i.e. what the encoder actually receives.

The point of the sweep is `cover_min_visible_frac` under the stock `prefix`
crop: does raising COVER's own visibility floor remove the zero-anatomy encoder
views without touching shared collation? The `window` column is the arm we
actually trained, kept for reference only.
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
from src.masks.multiblock import MaskCollator                     # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402

CROP, PATCH, GRID = 256, 16, 16
NP = GRID * GRID
BASE = dict(input_size=(CROP, CROP), patch_size=PATCH,
            enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
            allow_overlap=False)


def cover_cfg(floor, trunc):
    """COVER. Both the soft leave target and the hard floor move together,
    matching plot_window_vs_floor.py."""
    return dict(mode="mirage_cover", T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", mirage_occupancy_threshold=0.25,
                anatomy_tau=0.10, cover_leave_frac=floor,
                cover_min_visible_frac=floor, cover_min_visible_cells=4,
                cover_fill="random_legal", enc_truncate=trunc)


def mode_cfg(mode, trunc="prefix"):
    return dict(mode=mode, T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", mirage_occupancy_threshold=0.25,
                anatomy_tau=0.10, enc_truncate=trunc)


# Constructor kwargs (siblings of BASE), NOT curriculum_cfg entries.
# blob's config carries pred_target_k in its `mask:` block; without it the
# generator refuses to build.
EXTRA = {"blob    (mirage_anatomy)": dict(pred_target_k=16)}


# name -> (kind, payload).  kind: "stock" | "curr"
ARMS = [
    ("random  (stock JEPA)",        "stock", None),
    ("oracle  (anatomical_prior)",  "curr",  mode_cfg("anatomical_prior")),
    ("envelope(mirage_envelope)",   "curr",  mode_cfg("mirage_envelope")),
    ("blob    (mirage_anatomy)",    "curr",  mode_cfg("mirage_anatomy")),
    ("COVER floor 0.15  prefix",    "curr",  cover_cfg(0.15, "prefix")),
    ("COVER floor 0.20  prefix",    "curr",  cover_cfg(0.20, "prefix")),
    ("COVER floor 0.25  prefix",    "curr",  cover_cfg(0.25, "prefix")),
    ("COVER floor 0.30  prefix",    "curr",  cover_cfg(0.30, "prefix")),
    ("COVER floor 0.35  prefix",    "curr",  cover_cfg(0.35, "prefix")),
    ("COVER floor 0.15  window *",  "curr",  cover_cfg(0.15, "window")),
]

KEYS = ["ctx", "ctx_anat", "pct_ctx_anat", "pct_anat_vis", "zero",
        "tgt", "tgt_anat", "pct_tgt_anat", "pct_anat_hid", "n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epoch", type=int, default=50)
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

    gens, stock = {}, None
    for name, kind, cfg in ARMS:
        if kind == "stock":
            stock = MaskCollator(**BASE)
        else:
            g = CurriculumMaskGenerator(**BASE, **EXTRA.get(name, {}),
                                        curriculum_cfg=cfg)
            g.set_epoch(args.epoch, 100)
            gens[name] = g

    agg = {name: {k: 0.0 for k in KEYS} for name, _, _ in ARMS}
    B = args.batch_size

    for bi in range(args.batches):
        start = bi * B
        items = [ds[i] for i in range(start, start + B)]
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        anat = (guides[:, 0].reshape(B, -1).numpy() >= 0.25)

        outs = {}
        for name, kind, _ in ARMS:
            # identical RNG state for every arm on every batch
            random.seed(11 + start); np.random.seed(11 + start)
            torch.manual_seed(11 + start)
            if kind == "stock":
                _, me, mp = stock([imgs[i] for i in range(B)])
            else:
                me, mp = gens[name].generate(
                    batch_size=B, imgs_cpu=imgs,
                    guide_grids=guides, guide_valid=valid)
            outs[name] = (me, mp)

        for b in range(B):
            a = anat[b]
            na = int(a.sum())
            if na == 0:
                continue
            for name, _, _ in ARMS:
                me, mp = outs[name]
                c = np.zeros(NP, bool); c[me[0][b].numpy()] = True
                t = np.zeros(NP, bool)
                for m in mp:
                    t[m[b].numpy()] = True
                ca = int((a & c).sum())
                ta = int((a & t).sum())
                d = agg[name]
                d["ctx"] += int(c.sum())
                d["ctx_anat"] += ca
                d["pct_ctx_anat"] += 100.0 * ca / max(int(c.sum()), 1)
                d["pct_anat_vis"] += 100.0 * ca / na
                d["zero"] += (ca == 0)
                d["tgt"] += int(t.sum())
                d["tgt_anat"] += ta
                d["pct_tgt_anat"] += 100.0 * ta / max(int(t.sum()), 1)
                d["pct_anat_hid"] += 100.0 * ta / na
                d["n"] += 1
        print(f"  batch {bi + 1}/{args.batches}", flush=True)

    hdr = (f"{'arm':28s} {'ctx':>6s} {'ctxAnat':>8s} {'ctx%anat':>9s} "
           f"{'anatVis%':>9s} {'ZERO%':>7s} {'tgt':>6s} {'tgtAnat':>8s} "
           f"{'tgt%anat':>9s} {'anatHid%':>9s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    rows = {}
    for name, _, _ in ARMS:
        d = agg[name]; n = max(d["n"], 1)
        r = {k: d[k] / n for k in KEYS if k != "n"}
        r["zero"] = 100.0 * d["zero"] / n
        r["n"] = int(d["n"])
        rows[name] = r
        print(f"{name:28s} {r['ctx']:6.1f} {r['ctx_anat']:8.1f} "
              f"{r['pct_ctx_anat']:8.1f}% {r['pct_anat_vis']:8.1f}% "
              f"{r['zero']:6.2f}% {r['tgt']:6.1f} {r['tgt_anat']:8.1f} "
              f"{r['pct_tgt_anat']:8.1f}% {r['pct_anat_hid']:8.1f}%")
    print(f"\nslices measured per arm: {rows[ARMS[0][0]]['n']}")
    print("* window = the crop actually used by the trained COVER arm")

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "arm_stats.json").write_text(json.dumps(rows, indent=2))
    print("wrote", out / "arm_stats.json")


if __name__ == "__main__":
    main()
