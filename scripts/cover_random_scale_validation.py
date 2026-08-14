"""Large-scale validation of the COVER-then-RANDOM arm through the REAL sampler.

The earlier 600-slice probe called ``cover.build_targets`` directly with its own
block-size draw and ``tau=0.30``, which an adversarial review correctly flagged
as not reproducing the launched configuration.  This script instead drives the
production ``CurriculumMaskGenerator`` with the exact curriculum config from the
run YAML, so what is measured is what will train.

Checks the guarantees that matter before committing GPU time:
  * the anatomy-visibility floor, evaluated on the OCCUPANCY mask the audits use
    (not the sampler's softer internal support),
  * anatomy actually hidden,
  * how the 4 blocks are split between coverage and plain random placement,
  * fallback / infeasible rates,
  * target cardinality parity with the envelope arm (42 cells per block).

Runs the shipped ``transition`` fill on the SAME slices for a paired contrast.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator          # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402

CROP, PATCH = 256, 16
NPATCH = (CROP // PATCH) ** 2


def build_gen(curr_cfg, mask_cfg, fill, epoch, total):
    cfg = dict(curr_cfg)
    cfg.pop("enabled", None)
    cfg["cover_fill"] = fill
    g = CurriculumMaskGenerator(
        input_size=(CROP, CROP), patch_size=PATCH,
        enc_mask_scale=tuple(mask_cfg["enc_mask_scale"]),
        pred_mask_scale=tuple(mask_cfg["pred_mask_scale"]),
        aspect_ratio=tuple(mask_cfg["aspect_ratio"]),
        nenc=int(mask_cfg["num_enc_masks"]), npred=int(mask_cfg["num_pred_masks"]),
        min_keep=int(mask_cfg["min_keep"]),
        allow_overlap=bool(mask_cfg["allow_overlap"]),
        curriculum_cfg=cfg,
    )
    g.set_epoch(epoch, total)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=r"configs\patch_cover_random_ep25.yaml")
    ap.add_argument("--volumes", type=int, default=60)
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--slices_per_volume", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--total", type=int, default=100)
    ap.add_argument("--fills", nargs="+", default=["random_legal", "transition"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\cover_random_scale")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    mask_cfg = cfg["mask"]
    curr_cfg = mask_cfg["curriculum"]
    occ_t = float(curr_cfg.get("mirage_occupancy_threshold", 0.25))
    floor = float(curr_cfg.get("cover_min_visible_frac", 0.15))
    print(f"config {args.config}  tau={curr_cfg.get('anatomy_tau')} "
          f"leave={curr_cfg.get('cover_leave_frac')} floor={floor} occ_t={occ_t}",
          flush=True)

    paired = make_paired_transforms(
        crop_size=CROP, crop_scale=tuple(cfg["data"]["crop_scale"]),
        gaussian_blur=False, horizontal_flip=False, color_distortion=False,
        color_jitter=0.0)
    sc = os.path.join(cfg["data"]["slice_cache_dir"], "Training")
    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(cfg["data"]["data_dir"], "Training"),
        guide_dir=os.path.join(curr_cfg["mirage_guide_dir"], "Training"),
        num_slices=args.num_slices, slice_size=CROP, transform=paired,
        patch_size=PATCH, dilate_patches=int(curr_cfg.get("mirage_dilate_patches", 0)),
        occupancy_threshold=occ_t, slice_cache=sc if os.path.isdir(sc) else None)

    rng0 = random.Random(args.seed)
    vols = sorted(rng0.sample(range(len(ds.file_paths)),
                              min(args.volumes, len(ds.file_paths))))
    step = max(1, args.num_slices // args.slices_per_volume)
    idxs = [v * args.num_slices + s
            for v in vols for s in range(0, args.num_slices, step)]
    print(f"{len(idxs)} slices from {len(vols)} volumes", flush=True)

    results = {}
    for fill in args.fills:
        gen = build_gen(curr_cfg, mask_cfg, fill, args.epoch, args.total)
        vis, hid, tgt_cells, ctx_cells = [], [], [], []
        zero = below = 0
        n_invalid = zero_invalid = 0
        rect_perfect = rect_total = 0
        rect_fill = []
        block_hw = []
        t0 = time.time()
        for start in range(0, len(idxs), args.batch_size):
            chunk = idxs[start:start + args.batch_size]
            items = [ds[i] for i in chunk]
            imgs = torch.stack([it[0] for it in items], 0)
            guides = torch.stack([it[1] for it in items], 0)
            valid = torch.stack([it[2] for it in items], 0)
            B = imgs.size(0)
            anat = (guides[:, 0].reshape(B, -1).numpy() >= occ_t)
            gvalid = valid.numpy().astype(bool).reshape(B)
            random.seed(args.seed + start)
            np.random.seed(args.seed + start)
            torch.manual_seed(args.seed + start)
            m_enc, m_pred = gen.generate(batch_size=B, guide_grids=guides,
                                         guide_valid=valid)
            tgt_cells.append(m_pred[0].shape[1])
            ctx_cells.append(m_enc[0].shape[1])
            # Shape audit on what the PREDICTOR actually receives, i.e. AFTER
            # the stock global-min truncation.  COVER places solid rectangles,
            # but truncation keeps a row-major prefix of the indices and can
            # clip a rectangle into a partial one, so this must be measured on
            # the emitted indices rather than assumed from the placement code.
            for m in m_pred:
                idx = m.numpy()
                for b in range(idx.shape[0]):
                    cells = idx[b]
                    rr, cc = np.divmod(cells, CROP // PATCH)
                    h = rr.max() - rr.min() + 1
                    w = cc.max() - cc.min() + 1
                    rect_total += 1
                    frac = len(cells) / float(h * w)
                    rect_fill.append(frac)
                    if len(np.unique(cells)) == len(cells) and abs(frac - 1.0) < 1e-9:
                        rect_perfect += 1
                    block_hw.append((int(h), int(w)))
            for b in range(B):
                u = np.zeros(NPATCH, bool)
                for m in m_pred:
                    u[m[b].numpy()] = True
                a = anat[b]
                if a.sum() == 0:
                    continue
                v = (a & ~u).sum() / a.sum()
                if not gvalid[b]:
                    # The guide failed QC, so COVER never ran and the slice was
                    # masked by the stock uniform sampler, which makes no
                    # anatomy promise.  Counting it as a floor breach would
                    # blame COVER for the documented fallback path.
                    n_invalid += 1
                    zero_invalid += (v == 0)
                    continue
                vis.append(v); hid.append(1 - v)
                zero += (v == 0)
                below += (v < floor - 1e-9)
            if start % (args.batch_size * 20) == 0:
                print(f"  {fill}: {start + B}/{len(idxs)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        st = getattr(gen, "_mirage_stats", {})
        st = st() if callable(st) else st
        vis = np.array(vis); hid = np.array(hid)
        r = dict(
            fill=fill, n_slices=int(len(vis)),
            occ_visible_mean=float(vis.mean()), occ_visible_min=float(vis.min()),
            occ_visible_p1=float(np.percentile(vis, 1)),
            occ_visible_p5=float(np.percentile(vis, 5)),
            below_floor=int(below), below_floor_pct=float(100 * below / len(vis)),
            zero_visible=int(zero),
            # Slices COVER never saw because the guide failed QC.  Masked by
            # the stock uniform sampler; reported, not counted as failures.
            invalid_guide_slices=int(n_invalid),
            invalid_guide_pct=float(100 * n_invalid / max(len(vis) + n_invalid, 1)),
            invalid_guide_zero_visible=int(zero_invalid),
            anatomy_hidden_mean=float(hid.mean()),
            anatomy_hidden_p95=float(np.percentile(hid, 95)),
            target_cells_per_block=float(np.mean(tgt_cells)),
            context_tokens=float(np.mean(ctx_cells)),
            # Shape parity with the stock/envelope arms: the predictor must be
            # reconstructing plain rectangles, not irregular blobs.
            blocks_checked=int(rect_total),
            perfect_rectangles_pct=float(100 * rect_perfect / max(rect_total, 1)),
            bbox_fill_mean=float(np.mean(rect_fill)),
            bbox_fill_min=float(np.min(rect_fill)),
            block_h_mean=float(np.mean([h for h, _ in block_hw])),
            block_w_mean=float(np.mean([w for _, w in block_hw])),
            **{k: float(v) for k, v in st.items()
               if isinstance(v, (int, float)) and ("cover" in k or "fallback" in k
                                                   or "infeasible" in k)},
        )
        results[fill] = r
        print(f"\n=== {fill} ===")
        for k, v in r.items():
            if k != "fill":
                print(f"  {k:28s} {v}")
        print(flush=True)

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "scale_validation.json").write_text(json.dumps(results, indent=2))

    print("\n================ VERDICT ================")
    ok = True
    for fill, r in results.items():
        bad = (r["below_floor"] > 0) or (r["zero_visible"] > 0)
        ok &= not bad
        print(f"{fill:14s} n={r['n_slices']:5d}  hidden {100*r['anatomy_hidden_mean']:.1f}%  "
              f"visible min {100*r['occ_visible_min']:.1f}%  "
              f"below-floor {r['below_floor']}  zero {r['zero_visible']}  "
              f"rect {r['perfect_rectangles_pct']:.1f}%  "
              f"[invalid-guide fallbacks {r['invalid_guide_slices']}, "
              f"of which zero-visible {r['invalid_guide_zero_visible']}]  "
              f"{'OK' if not bad else 'FAIL'}")
    print(f"\nwrote {out / 'scale_validation.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
