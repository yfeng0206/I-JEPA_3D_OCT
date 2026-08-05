#!/usr/bin/env python
"""Which retinal layers do the ORACLE and MIRAGE target blocks actually cover?

Glaucoma damages the RNFL first, then the ganglion-cell / inner-plexiform
complex (GCIPL).  The choroid is not a glaucoma-diagnostic layer.

The oracle prior is a fixed-height ribbon centred on the per-column intensity
centroid, sized to ~28% of the frame.  The MIRAGE guide is the repaired
envelope of RNFL + GCIPL + choroid, i.e. the WHOLE retina.  Both can therefore
score well on "targets land on tissue" while distributing their target budget
very differently across layers.  If MIRAGE spends more of its budget on
choroid, its targets are anatomically correct but diagnostically more diffuse,
which is a candidate explanation for it not beating the oracle downstream.

This measures that directly.  Image, guide and the three layer masks all pass
through ONE RandomResizedCrop draw, and each arm samples with the same RNG
seed, so only target placement differs.

    python scripts/layer_target_composition.py --volumes 250
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.guides.mirage_envelope import (  # noqa: E402
    CLASS_CHOROID,
    CLASS_GCIPL,
    CLASS_RNFL,
    patch_occupancy,
    unpack_guides,
)
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402

CROP, PATCH, GRID, NATIVE = 256, 16, 16, 200
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LAYERS = (("RNFL", CLASS_RNFL), ("GCIPL", CLASS_GCIPL), ("choroid", CLASS_CHOROID))


def paired_crop(image, masks, rng):
    """One RandomResizedCrop draw applied to the image and every mask."""
    height, width = image.shape
    for _ in range(10):
        area = height * width * rng.uniform(0.3, 1.0)
        ratio = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
        crop_w = int(round(np.sqrt(area * ratio)))
        crop_h = int(round(np.sqrt(area / ratio)))
        if crop_w <= width and crop_h <= height:
            top = int(rng.integers(0, height - crop_h + 1))
            left = int(rng.integers(0, width - crop_w + 1))
            break
    else:
        top, left, crop_h, crop_w = 0, 0, height, width
    img = np.asarray(
        Image.fromarray(image[top:top + crop_h, left:left + crop_w], mode="L")
        .resize((CROP, CROP), Image.BICUBIC))
    out = [
        np.asarray(
            Image.fromarray(
                mk[top:top + crop_h, left:left + crop_w].astype(np.uint8) * 255,
                mode="L").resize((CROP, CROP), Image.NEAREST)) > 127
        for mk in masks
    ]
    return img, out


def to_tensor(image_crop):
    """Match the training transform: replicate to RGB and ImageNet-normalise."""
    arr = np.repeat(image_crop[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def union_of_targets(gen, arm, img_t, guide, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    if arm == "oracle":
        _, pred = gen.generate(batch_size=1, imgs_cpu=img_t.unsqueeze(0))
    elif arm == "mirage":
        _, pred = gen.generate(batch_size=1, guide_grids=guide,
                               guide_valid=torch.ones(1, dtype=torch.bool))
    else:
        _, pred = gen.generate(batch_size=1)
    out = np.zeros(GRID * GRID, dtype=bool)
    for blk in pred:
        out[blk[0].numpy()] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/patch_mirage_envelope.yaml")
    ap.add_argument("--oracle-config", default="configs/patch_oracle_anatomical.yaml")
    ap.add_argument("--volumes", type=int, default=250)
    ap.add_argument("--slice-stride", type=int, default=25)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ocurr = yaml.safe_load(open(args.oracle_config))["mask"]["curriculum"]
    d, m, curr = cfg["data"], cfg["mask"], cfg["mask"]["curriculum"]
    root = os.path.dirname(d["data_dir"])
    data_dir = os.path.join(d["data_dir"], "Training")
    mask_dir = os.path.join(root, "mirage_masks", "Training")
    guide_dir = os.path.join(curr["mirage_guide_dir"], "Training")

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

    gens = {
        "random": build({"enabled": False}),
        "oracle": build({k: v for k, v in ocurr.items()
                         if k == "mode" or k.startswith("oracle_")}),
        "mirage": build({}),
    }

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
    rng0 = np.random.default_rng(args.seed)
    vols = rng0.choice(len(files), size=min(args.volumes, len(files)), replace=False)

    arms = ("random", "oracle", "mirage")
    keys = [n for n, _ in LAYERS] + ["any_retina", "background"]
    frac = {a: {k: [] for k in keys} for a in arms}
    avail = {n: [] for n, _ in LAYERS}
    n_slices = 0
    slice_indices = np.linspace(0, 199, num=d["num_slices"], dtype=np.int64)

    for vi in vols:
        name = files[vi]
        gp, mp = os.path.join(guide_dir, name), os.path.join(mask_dir, name)
        if not (os.path.isfile(gp) and os.path.isfile(mp)):
            continue
        with np.load(os.path.join(data_dir, name), allow_pickle=True) as z:
            volume = z["oct_bscans"]
        with np.load(mp, allow_pickle=False) as z:
            hard_all = z["hard_masks"]
        with np.load(gp, allow_pickle=False) as z:
            packed, valid_all = z["packed_envelopes"], z["valid"]

        for sl in range(0, d["num_slices"], args.slice_stride):
            if not bool(valid_all[sl]):
                continue
            image = np.array(volume[int(slice_indices[sl])])
            hard = hard_all[sl]
            envelope = unpack_guides(packed[sl:sl + 1], (NATIVE, NATIVE))[0]

            rng = np.random.default_rng(1000 + int(vi) * 1000 + sl)
            image_crop, cropped = paired_crop(
                image, [envelope] + [hard == cls for _, cls in LAYERS], rng)
            env_c, layers_c = cropped[0], cropped[1:]

            occ = patch_occupancy(env_c, patch_size=PATCH)
            retina = occ >= 0.25
            if retina.sum() < 24:
                continue

            layer_grids = {
                nm: patch_occupancy(mk, patch_size=PATCH) >= 0.25
                for (nm, _), mk in zip(LAYERS, layers_c)
            }
            for nm, grid in layer_grids.items():
                avail[nm].append(grid.sum() / 256.0)

            img_t = to_tensor(image_crop)
            guide = torch.from_numpy(
                np.stack([occ, retina.astype(np.float32)], 0)).unsqueeze(0)
            seed = 4242 + (int(vi) * 131 + sl) % 9973

            for arm in arms:
                u = union_of_targets(gens[arm], arm, img_t, guide, seed)
                if not u.any():
                    continue
                k = float(u.sum())
                for nm, grid in layer_grids.items():
                    frac[arm][nm].append(float(grid.reshape(-1)[u].sum()) / k)
                r = float(retina.reshape(-1)[u].sum()) / k
                frac[arm]["any_retina"].append(r)
                frac[arm]["background"].append(1.0 - r)
            n_slices += 1

    print("slices measured: %d" % n_slices)
    print()
    print("Layer availability (share of the 256-cell grid, mean per slice)")
    for nm, _ in LAYERS:
        print("  %-9s %.4f" % (nm, float(np.mean(avail[nm]))))
    print()
    print("Composition of each arm's TARGET patches")
    print("  %-8s %8s %8s %8s %9s %9s"
          % ("arm", "RNFL", "GCIPL", "choroid", "retina", "backgrnd"))
    for arm in arms:
        print("  %-8s %8.4f %8.4f %8.4f %9.4f %9.4f"
              % (arm, float(np.mean(frac[arm]["RNFL"])),
                 float(np.mean(frac[arm]["GCIPL"])),
                 float(np.mean(frac[arm]["choroid"])),
                 float(np.mean(frac[arm]["any_retina"])),
                 float(np.mean(frac[arm]["background"]))))
    print()

    def g(arm):
        return (float(np.mean(frac[arm]["RNFL"]))
                + float(np.mean(frac[arm]["GCIPL"])))

    print("Glaucoma-relevant layers (RNFL + GCIPL) as a share of targets")
    print("  random %.4f   oracle %.4f   mirage %.4f"
          % (g("random"), g("oracle"), g("mirage")))
    print("  mirage - oracle  %+.4f" % (g("mirage") - g("oracle")))
    print("  mirage - random  %+.4f" % (g("mirage") - g("random")))
    print()
    c_o = float(np.mean(frac["oracle"]["choroid"]))
    c_m = float(np.mean(frac["mirage"]["choroid"]))
    print("Choroid share of targets: oracle %.4f  mirage %.4f  diff %+.4f"
          % (c_o, c_m, c_m - c_o))


if __name__ == "__main__":
    main()
