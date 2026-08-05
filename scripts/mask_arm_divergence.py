#!/usr/bin/env python
"""How different are the ORACLE and MIRAGE target masks, really?

The MIRAGE run reached essentially the same validation loss as the oracle arm.
One explanation is that the two masking policies place their target blocks in
nearly the same cells, so the two arms trained on nearly the same task.  This
script tests that directly instead of arguing about it.

The measurement is the Jaccard overlap (IoU) between the four-block target
unions produced by each arm on the SAME slice with the SAME RNG seed, so block
sizes are identical and only placement differs.

The number that matters is the comparison against a within-arm control:

    IoU(oracle, MIRAGE)     how similar the two arms are
    IoU(MIRAGE_a, MIRAGE_b) how similar one arm is to ITSELF on two seeds

If the between-arm IoU is as high as the within-arm IoU, the arms are no more
different from each other than two draws of the same arm, and identical val
loss would be unsurprising.  If it is clearly lower, the arms are genuinely
sampling different regions and the similar loss means something else.

    python scripts/mask_arm_divergence.py --volumes 300
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402
from src.transforms import make_paired_transforms  # noqa: E402

CROP, PATCH, GRID = 256, 16, 16


def _read(ds, idx):
    """Deterministic read; the paired transform uses Python's random too."""
    random.seed(idx)
    torch.manual_seed(idx)
    np.random.seed(idx % (2 ** 31))
    return ds[idx]


def _union(generator, kind, img_t, guide, seed):
    """Union of the four target blocks, as a flat boolean over the 16x16 grid."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    if kind == "oracle":
        _, pred = generator.generate(batch_size=1, imgs_cpu=img_t.unsqueeze(0))
    elif kind == "mirage":
        _, pred = generator.generate(
            batch_size=1, guide_grids=guide,
            guide_valid=torch.ones(1, dtype=torch.bool))
    else:
        _, pred = generator.generate(batch_size=1)
    out = np.zeros(GRID * GRID, dtype=bool)
    for blk in pred:
        out[blk[0].numpy()] = True
    return out


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/patch_mirage_envelope.yaml")
    ap.add_argument("--oracle-config", default="configs/patch_oracle_anatomical.yaml")
    ap.add_argument("--volumes", type=int, default=300)
    ap.add_argument("--slice-stride", type=int, default=23)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    oracle_curr = yaml.safe_load(open(args.oracle_config))["mask"]["curriculum"]
    d, m, curr = cfg["data"], cfg["mask"], cfg["mask"]["curriculum"]

    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(d["data_dir"], "Training"),
        guide_dir=os.path.join(curr["mirage_guide_dir"], "Training"),
        num_slices=d["num_slices"], slice_size=CROP,
        transform=make_paired_transforms(
            crop_size=CROP, crop_scale=tuple(d["crop_scale"]),
            horizontal_flip=False, gaussian_blur=False,
            color_distortion=False, color_jitter=0.0),
        patch_size=PATCH,
        dilate_patches=int(curr["mirage_dilate_patches"]),
        occupancy_threshold=float(curr["mirage_occupancy_threshold"]),
        slice_cache=os.path.join(d["slice_cache_dir"], "Training"),
    )

    def build(extra):
        c = dict(curr)
        c.update(extra)
        g = CurriculumMaskGenerator(
            input_size=(CROP, CROP), patch_size=PATCH,
            enc_mask_scale=tuple(m["enc_mask_scale"]),
            pred_mask_scale=tuple(m["pred_mask_scale"]),
            aspect_ratio=tuple(m["aspect_ratio"]),
            nenc=m["num_enc_masks"], npred=m["num_pred_masks"],
            min_keep=m["min_keep"], allow_overlap=m["allow_overlap"],
            curriculum_cfg=c)
        g.set_epoch(60, cfg["optimization"]["epochs"])
        return g

    g_mir = build({})
    g_ora = build({k: v for k, v in oracle_curr.items()
                   if k == "mode" or k.startswith("oracle_")})
    g_rnd = build({"enabled": False})

    rng = np.random.default_rng(args.seed)
    vol_ids = rng.choice(len(ds.file_paths),
                         size=min(args.volumes, len(ds.file_paths)),
                         replace=False)

    acc = {k: [] for k in ("ora_mir", "mir_mir", "ora_ora", "rnd_ora",
                           "rnd_mir", "rnd_rnd", "ret_ora", "ret_mir",
                           "area_ora", "area_mir")}
    n = 0
    for vi in vol_ids:
        for sl in range(0, ds.num_slices, args.slice_stride):
            idx = int(vi) * ds.num_slices + sl
            img_t, guide_t, valid_t = _read(ds, idx)
            if not bool(valid_t):
                continue
            occ = guide_t[0].numpy()
            retina = occ >= 0.25
            if retina.sum() < 24:
                continue
            guide = torch.from_numpy(
                np.stack([occ, retina.astype(np.float32)], 0)).unsqueeze(0)
            sa, sb = 1000 + idx, 5000 + idx

            o_a = _union(g_ora, "oracle", img_t, guide, sa)
            o_b = _union(g_ora, "oracle", img_t, guide, sb)
            m_a = _union(g_mir, "mirage", img_t, guide, sa)
            m_b = _union(g_mir, "mirage", img_t, guide, sb)
            r_a = _union(g_rnd, "random", img_t, guide, sa)
            r_b = _union(g_rnd, "random", img_t, guide, sb)

            acc["ora_mir"].append(_iou(o_a, m_a))   # between arms, same seed
            acc["mir_mir"].append(_iou(m_a, m_b))   # within MIRAGE, two seeds
            acc["ora_ora"].append(_iou(o_a, o_b))   # within oracle, two seeds
            acc["rnd_ora"].append(_iou(r_a, o_a))
            acc["rnd_mir"].append(_iou(r_a, m_a))
            acc["rnd_rnd"].append(_iou(r_a, r_b))
            flat = retina.reshape(-1)
            acc["ret_ora"].append(float(flat[o_a].mean()) if o_a.any() else 0.0)
            acc["ret_mir"].append(float(flat[m_a].mean()) if m_a.any() else 0.0)
            acc["area_ora"].append(o_a.sum() / 256.0)
            acc["area_mir"].append(m_a.sum() / 256.0)
            n += 1

    def s(key):
        return float(np.nanmean(acc[key]))

    print("slices measured: %d" % n)
    print()
    print("Target-union overlap (IoU), same slice, same RNG seed")
    print("  %-34s %.3f" % ("oracle  vs MIRAGE   [between arms]", s("ora_mir")))
    print("  %-34s %.3f" % ("MIRAGE  vs MIRAGE   [within, 2 seeds]", s("mir_mir")))
    print("  %-34s %.3f" % ("oracle  vs oracle   [within, 2 seeds]", s("ora_ora")))
    print("  %-34s %.3f" % ("random  vs random   [within, 2 seeds]", s("rnd_rnd")))
    print("  %-34s %.3f" % ("random  vs oracle", s("rnd_ora")))
    print("  %-34s %.3f" % ("random  vs MIRAGE", s("rnd_mir")))
    print()
    print("Targets landing on retina")
    print("  %-34s %.3f" % ("oracle", s("ret_ora")))
    print("  %-34s %.3f" % ("MIRAGE", s("ret_mir")))
    print()
    print("Masked area (fraction of the 256-cell grid)")
    print("  %-34s %.3f" % ("oracle", s("area_ora")))
    print("  %-34s %.3f" % ("MIRAGE", s("area_mir")))
    print()

    between = s("ora_mir")
    within = 0.5 * (s("mir_mir") + s("ora_ora"))
    print("VERDICT")
    print("  between-arm IoU %.3f vs within-arm IoU %.3f  -> ratio %.2f"
          % (between, within, between / within if within else float("nan")))
    if between >= within:
        print("  The arms overlap as much as one arm overlaps itself.")
        print("  The two policies are NOT meaningfully different in practice.")
    else:
        print("  The arms overlap LESS than one arm overlaps itself, so they")
        print("  genuinely sample different regions; similar val loss is not")
        print("  explained by the masks being the same.")


if __name__ == "__main__":
    main()
