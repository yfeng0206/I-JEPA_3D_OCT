#!/usr/bin/env python
"""Render the MIRAGE guide pipeline for docs/experiments/mirage_guided_masking.md.

Two committed figures, sized for GitHub rendering:

``mirage_guide_pipeline.png``
    How the placement region is built, left to right: the B-scan, MIRAGE's raw
    three-class segmentation, the raw union (three disconnected bands), the
    repaired envelope (one connected structure), the fractional 16x16 patch
    occupancy, and the boolean placement region at the shipped threshold.

``mirage_masking_arms.png``
    The three experimental arms side by side on the same slice with the same
    RNG seed, so only target-block LOCATION differs: RANDOM, ORACLE band,
    MIRAGE thr 0.25.

Run from the repo root:
    python scripts/mirage_doc_figures.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.guides.mirage_envelope import (  # noqa: E402
    CLASS_CHOROID,
    CLASS_GCIPL,
    CLASS_RNFL,
    build_union,
    patch_occupancy,
    repair_union,
    unpack_guides,
)
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402
from src.transforms import make_paired_transforms  # noqa: E402

NATIVE = 200
PATCH = 16
GRID = 16
CROP = 256

# Layer colours are distinct hues so the three bands read separately even where
# they touch; the envelope is cyan throughout the docs.
COL_RNFL = (240, 92, 92)
COL_GCIPL = (250, 190, 70)
COL_CHOROID = (120, 150, 255)
COL_UNION = (170, 170, 178)
COL_ENVELOPE = (0, 210, 190)
BLOCKS = [(228, 62, 58), (66, 133, 220), (233, 168, 47), (128, 194, 100)]


def _font(size, bold=False):
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _gray_rgb(slice_2d, size):
    img = Image.fromarray(slice_2d, mode="L").resize((size, size), Image.BILINEAR)
    return np.asarray(img.convert("RGB")).astype(np.uint8)


def _overlay(base, mask, colour, alpha=0.55):
    out = base.astype(np.float32)
    for c in range(3):
        out[..., c] = np.where(mask, out[..., c] * (1 - alpha) + colour[c] * alpha,
                               out[..., c])
    return out.astype(np.uint8)


def _upscale(arr, size, grid=False):
    img = Image.fromarray(arr).resize((size, size), Image.NEAREST)
    if grid:
        draw = ImageDraw.Draw(img)
        step = size / GRID
        for k in range(GRID + 1):
            p = k * step
            draw.line([(p, 0), (p, size)], fill=(255, 255, 255, 40))
            draw.line([(0, p), (size, p)], fill=(255, 255, 255, 40))
    return img


def _occupancy_heat(occ, base, size):
    """Fractional occupancy as a cyan ramp over the dimmed B-scan."""
    big = np.repeat(np.repeat(occ, PATCH, 0), PATCH, 1)
    if big.shape[0] != base.shape[0]:
        big = np.asarray(
            Image.fromarray((big * 255).astype(np.uint8)).resize(
                (base.shape[1], base.shape[0]), Image.NEAREST)
        ).astype(np.float32) / 255.0
    out = base.astype(np.float32) * 0.45
    for c in range(3):
        out[..., c] += big * COL_ENVELOPE[c] * 0.75
    return np.clip(out, 0, 255).astype(np.uint8)


def _load(volume_name, slice_idx, data_dir, mask_dir, guide_dir):
    with np.load(Path(data_dir) / volume_name, allow_pickle=True) as z:
        volume = z["oct_bscans"]
        with np.load(Path(mask_dir) / volume_name, allow_pickle=False) as m:
            slice_indices = m["slice_indices"]
            hard = m["hard_masks"][slice_idx]
        depth = int(slice_indices[slice_idx])
        image = np.array(volume[depth])
    with np.load(Path(guide_dir) / volume_name, allow_pickle=False) as g:
        envelope = unpack_guides(
            g["packed_envelopes"][slice_idx:slice_idx + 1], (NATIVE, NATIVE))[0]
        valid = bool(g["valid"][slice_idx])
    return image, hard, envelope, valid


def figure_pipeline(rows, out_path, size=224):
    cols = [
        "B-scan",
        "MIRAGE raw (3 classes)",
        "raw union",
        "repaired envelope",
        "patch occupancy",
        "placement region >= 0.25",
    ]
    lab, head, gap, foot = 120, 76, 10, 74
    W = lab + len(cols) * (size + gap)
    H = head + len(rows) * (size + gap) + foot
    fig = Image.new("RGB", (W, H), (14, 14, 16))
    d = ImageDraw.Draw(fig)
    d.text((16, 12), "Building the MIRAGE placement region",
           font=_font(20, True), fill=(245, 245, 245))
    d.text((16, 40),
           "MIRAGE segments three disconnected bands. The unlabelled mid-retina between them is filled in so the guide is ONE "
           "connected structure, then reduced to a 16x16 patch grid.",
           font=_font(13), fill=(178, 178, 184))
    for c, t in enumerate(cols):
        d.text((lab + c * (size + gap) + 3, head - 19), t, font=_font(12, True),
               fill=(228, 228, 233))

    for r, (name, sl, image, hard, envelope) in enumerate(rows):
        y = head + r * (size + gap)
        base = _gray_rgb(image, size)

        raw = base.copy()
        for cls, colour in ((CLASS_RNFL, COL_RNFL), (CLASS_GCIPL, COL_GCIPL),
                            (CLASS_CHOROID, COL_CHOROID)):
            m = np.asarray(Image.fromarray((hard == cls).astype(np.uint8) * 255)
                           .resize((size, size), Image.NEAREST)) > 127
            raw = _overlay(raw, m, colour, 0.62)

        union = build_union(hard)
        union_r = np.asarray(Image.fromarray(union.astype(np.uint8) * 255)
                             .resize((size, size), Image.NEAREST)) > 127
        env_r = np.asarray(Image.fromarray(envelope.astype(np.uint8) * 255)
                           .resize((size, size), Image.NEAREST)) > 127

        occ = _occ_from_envelope(envelope)
        region = occ >= 0.25

        cell = size // GRID
        region_big = np.repeat(np.repeat(region, cell, 0), cell, 1)
        panes = [
            Image.fromarray(base),
            Image.fromarray(raw),
            Image.fromarray(_overlay(base, union_r, COL_UNION, 0.6)),
            Image.fromarray(_overlay(base, env_r, COL_ENVELOPE, 0.55)),
            Image.fromarray(_occupancy_heat(occ, base, size)),
            _upscale(_overlay(base, region_big, COL_ENVELOPE, 0.5), size, grid=True),
        ]
        for c, pane in enumerate(panes):
            fig.paste(pane.convert("RGB"), (lab + c * (size + gap), y))

        # Row label: the connectivity number is the whole point of the repair.
        runs_raw = _mean_runs(union)
        runs_env = _mean_runs(envelope)
        d.text((14, y + 4), name, font=_font(12, True), fill=(238, 238, 238))
        d.text((14, y + 22), "slice %d" % sl, font=_font(11), fill=(150, 150, 158))
        d.text((14, y + 42), "runs/col", font=_font(11), fill=(150, 150, 158))
        d.text((14, y + 58), "raw  %.2f" % runs_raw, font=_font(11),
               fill=COL_UNION)
        d.text((14, y + 74), "env  %.2f" % runs_env, font=_font(11),
               fill=COL_ENVELOPE)
        d.text((14, y + 96), "region", font=_font(11), fill=(150, 150, 158))
        d.text((14, y + 112), "%d/256" % int(region.sum()), font=_font(11),
               fill=COL_ENVELOPE)

    fy = head + len(rows) * (size + gap) + 6
    d.text((16, fy),
           "runs/col = mean number of separate vertical runs per column. 1.00 means every column is a single unbroken band, "
           "i.e. the guide is genuinely one connected structure.",
           font=_font(12), fill=(178, 178, 184))
    d.text((16, fy + 20),
           "Red RNFL   Amber GCIPL   Blue choroid   Grey raw union   Cyan repaired envelope",
           font=_font(12), fill=(150, 150, 158))
    d.text((16, fy + 42),
           "Dilation is 0: measured over 1,000 volumes it costs 6-8 purity points at every threshold. "
           "Threshold 0.25 keeps masking geometry within 0.5% of the oracle arm.",
           font=_font(12), fill=(150, 150, 158))
    fig.save(out_path, optimize=True)
    return fig.size


def _occ_from_envelope(envelope):
    """Patch occupancy exactly as training computes it.

    The guide is resized to the 256 crop with NEAREST first (hard stays hard),
    then reduced to the 16x16 grid -- 200 is not divisible by 16.
    """
    big = np.asarray(
        Image.fromarray(envelope.astype(np.uint8) * 255).resize(
            (CROP, CROP), Image.NEAREST)
    ) > 127
    return patch_occupancy(big, patch_size=PATCH)


def _mean_runs(mask):
    """Mean number of separate vertical runs per occupied column."""
    cols = []
    for c in range(mask.shape[1]):
        col = mask[:, c]
        if not col.any():
            continue
        cols.append(int(np.sum(col[1:] & ~col[:-1])) + int(col[0]))
    return float(np.mean(cols)) if cols else 0.0


def figure_arms(picks, cfg, oracle_cfg, out_path, size=224):
    """RANDOM vs ORACLE vs MIRAGE on identical slices and seeds.

    Uses the real GuidedOCTSliceDataset so the image and its guide go through
    the SAME paired crop training applies -- rendering a mask computed on the
    cropped guide over an uncropped B-scan would misalign them.
    """
    m, curr = cfg["mask"], cfg["mask"]["curriculum"]

    def gen(extra):
        c = dict(curr)
        c.update(extra)
        g = CurriculumMaskGenerator(
            input_size=(CROP, CROP), patch_size=PATCH,
            enc_mask_scale=tuple(m["enc_mask_scale"]),
            pred_mask_scale=tuple(m["pred_mask_scale"]),
            aspect_ratio=tuple(m["aspect_ratio"]),
            nenc=m["num_enc_masks"], npred=m["num_pred_masks"],
            min_keep=m["min_keep"], allow_overlap=m["allow_overlap"],
            curriculum_cfg=c,
        )
        g.set_epoch(60, cfg["optimization"]["epochs"])
        return g

    # The oracle arm must use its OWN band parameters, not the MIRAGE config's
    # defaults, or it is not the arm we are comparing against.
    arms = [
        ("RANDOM (baseline)", gen({"enabled": False}), "random"),
        ("ORACLE anatomical band", gen(oracle_cfg), "oracle"),
        ("MIRAGE thr 0.25 (this run)", gen({}), "mirage"),
    ]
    cols = ["B-scan + retina"] + [a[0] for a in arms]
    lab, head, gap, foot = 128, 78, 10, 76
    W = lab + len(cols) * (size + gap)
    H = head + len(picks) * (size + gap) + foot
    fig = Image.new("RGB", (W, H), (14, 14, 16))
    d = ImageDraw.Draw(fig)
    d.text((16, 12), "The three arms: only target-block LOCATION differs",
           font=_font(20, True), fill=(245, 245, 245))
    d.text((16, 40),
           "Same slice, same crop, same RNG seed, so the four block sizes are identical across arms. "
           "Block count, scales, aspect ratio, context construction, predictor and loss are unchanged.",
           font=_font(13), fill=(178, 178, 184))
    for c, t in enumerate(cols):
        d.text((lab + c * (size + gap) + 3, head - 19), t, font=_font(12, True),
               fill=(228, 228, 233))

    cell = size // GRID
    for r, (name, sl, img_t, occ, valid) in enumerate(picks):
        y = head + r * (size + gap)
        base = _denorm(img_t, size)
        region = occ >= 0.25
        fig.paste(
            _upscale(_overlay(base, np.repeat(np.repeat(region, cell, 0), cell, 1),
                              COL_ENVELOPE, 0.35), size, grid=True).convert("RGB"),
            (lab, y))

        guide = torch.from_numpy(
            np.stack([occ.astype(np.float32), region.astype(np.float32)], 0)
        ).unsqueeze(0)
        seed = 909 + r * 31
        on_region = []

        for a, (label, generator, kind) in enumerate(arms):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if kind == "mirage":
                enc, pred = generator.generate(
                    batch_size=1, guide_grids=guide,
                    guide_valid=torch.ones(1, dtype=torch.bool))
            elif kind == "oracle":
                # anatomical_prior derives its band from the image intensity
                # profile, so bias_active is False without imgs_cpu and the
                # arm silently degrades to uniform random (curriculum.py:1008).
                enc, pred = generator.generate(
                    batch_size=1, imgs_cpu=img_t.unsqueeze(0))
            else:
                enc, pred = generator.generate(batch_size=1)
            ctx = np.zeros(GRID * GRID, dtype=bool)
            ctx[enc[0][0].numpy()] = True
            pane = _overlay(base.copy(),
                            np.repeat(np.repeat(~ctx.reshape(GRID, GRID), cell, 0),
                                      cell, 1), (0, 0, 0), 0.8)
            union = np.zeros(GRID * GRID, dtype=bool)
            for i, blk in enumerate(pred):
                flat = np.zeros(GRID * GRID, dtype=bool)
                flat[blk[0].numpy()] = True
                union |= flat
                pane = _overlay(pane,
                                np.repeat(np.repeat(flat.reshape(GRID, GRID), cell, 0),
                                          cell, 1), BLOCKS[i % 4], 0.64)
            fig.paste(_upscale(pane, size, grid=True).convert("RGB"),
                      (lab + (a + 1) * (size + gap), y))
            hit = region.reshape(-1)[union]
            on_region.append(float(hit.mean()) if union.any() else 0.0)

        d.text((14, y + 4), name, font=_font(12, True), fill=(238, 238, 238))
        d.text((14, y + 22), "slice %d" % sl, font=_font(11), fill=(150, 150, 158))
        d.text((14, y + 44), "retina %d/256" % int(region.sum()), font=_font(11),
               fill=COL_ENVELOPE)
        d.text((14, y + 66), "on-retina", font=_font(11), fill=(150, 150, 158))
        for i, (lbl, val) in enumerate(zip(("rand", "orac", "mir"), on_region)):
            d.text((14, y + 82 + i * 16), "%-5s %.2f" % (lbl, val), font=_font(11),
                   fill=(150, 230, 150) if i == 2 else (170, 170, 176))

    fy = head + len(picks) * (size + gap) + 6
    d.text((16, fy),
           "Dimmed = withheld from the encoder.  Red/blue/amber/green = the four target blocks the predictor must reconstruct.  "
           "Cyan in column 1 = the retina.",
           font=_font(12), fill=(178, 178, 184))
    d.text((16, fy + 20),
           "Over 1,000 volumes MIRAGE raises target-on-retina purity from 0.4530 (random) and 0.5602 (oracle) to 0.6320, "
           "while masked area stays within 0.2% of the oracle arm.",
           font=_font(12), fill=(150, 150, 158))
    d.text((16, fy + 38),
           "The per-slice on-retina numbers on the left are a 3-slice sample and are noisy -- oracle sits below random on two "
           "of these three. Only the 1,000-volume aggregate above is meaningful.",
           font=_font(11), fill=(132, 132, 140))
    fig.save(out_path, optimize=True)
    return fig.size


def _denorm(tensor, size):
    """Undo ImageNet normalisation and upscale for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr = (tensor * std + mean).clamp(0, 1).numpy().transpose(1, 2, 0)
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR)
    return np.asarray(img).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/patch_mirage_envelope.yaml")
    ap.add_argument("--oracle-config", default="configs/patch_oracle_anatomical.yaml")
    ap.add_argument("--out", default="results/masking")
    ap.add_argument("--volumes", nargs="+",
                    default=["data_00001.npz", "data_01640.npz", "data_03120.npz"])
    ap.add_argument("--slices", nargs="+", type=int, default=[50, 48, 36])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    oracle_curr = yaml.safe_load(open(args.oracle_config))["mask"]["curriculum"]
    oracle_cfg = {k: v for k, v in oracle_curr.items()
                  if k == "mode" or k.startswith("oracle_")}

    d = cfg["data"]
    data_dir = os.path.join(d["data_dir"], "Training")
    guide_dir = os.path.join(cfg["mask"]["curriculum"]["mirage_guide_dir"], "Training")
    mask_dir = os.path.join(os.path.dirname(d["data_dir"]), "mirage_masks", "Training")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for name, sl in zip(args.volumes, args.slices):
        image, hard, envelope, valid = _load(name, sl, data_dir, mask_dir, guide_dir)
        if not valid:
            print("skipping %s s%d (guide invalid)" % (name, sl))
            continue
        rows.append((name.replace(".npz", ""), sl, image, hard, envelope))

    # For the arms figure, go through the real dataset so image and guide share
    # the same paired crop that training applies.
    ds = GuidedOCTSliceDataset(
        data_dir=data_dir, guide_dir=guide_dir,
        num_slices=d["num_slices"], slice_size=CROP,
        transform=make_paired_transforms(
            crop_size=CROP, crop_scale=tuple(d["crop_scale"]),
            horizontal_flip=False, gaussian_blur=False,
            color_distortion=False, color_jitter=0.0),
        patch_size=PATCH,
        dilate_patches=int(cfg["mask"]["curriculum"]["mirage_dilate_patches"]),
        occupancy_threshold=float(
            cfg["mask"]["curriculum"]["mirage_occupancy_threshold"]),
        slice_cache=os.path.join(d["slice_cache_dir"], "Training"),
    )
    names = [os.path.basename(p) for p in ds.file_paths]
    picks = []
    for name, sl in zip(args.volumes, args.slices):
        idx = names.index(name) * ds.num_slices + sl
        torch.manual_seed(4242 + sl)
        np.random.seed(4242 + sl)
        random.seed(4242 + sl)
        img_t, guide_t, valid_t = ds[idx]
        if not bool(valid_t):
            continue
        picks.append((name.replace(".npz", ""), sl, img_t,
                      guide_t[0].numpy(), bool(valid_t)))

    p1 = os.path.join(args.out, "mirage_guide_pipeline.png")
    p2 = os.path.join(args.out, "mirage_masking_arms.png")
    s1 = figure_pipeline(rows, p1)
    s2 = figure_arms(picks, cfg, oracle_cfg, p2)
    for p, s in ((p1, s1), (p2, s2)):
        print("%s  %dx%d  %.0f KB" % (p, s[0], s[1], os.path.getsize(p) / 1024))


if __name__ == "__main__":
    main()
