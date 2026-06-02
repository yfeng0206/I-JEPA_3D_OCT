"""ORACLE build-check — verify the v2 retina-following mask hits the retina.

The oracle (`anatomical_prior` mode) localizes the bright retinal band per
slice from the row-intensity profile and masks a fixed-size window centred on
it across most of the lateral extent.  This script renders that mask on a
sample of real OCT slices (and synthetic fallbacks incl. tilted/atypical
cases) so we can eyeball that the band-finder hit the retina at ~25% region
size BEFORE committing compute to a pretraining run.

This is the oracle's only pre-flight check (G2a is for the self-guided rung,
not the oracle).

Usage:
    python scripts/oracle_build_check.py [--data_dir DIR] [--n 12]

Outputs: results/summary/oracle_build_check.png
"""
import argparse
import glob
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.masks.curriculum import CurriculumMaskGenerator


def make_oracle(slice_size=256, patch_size=16, **oracle_kw):
    cfg = {
        "enabled": True,
        "mode": "anatomical_prior",
        "T_warm": 0,
        "T_total": 1,
        "r_max": 1.0,
    }
    cfg.update(oracle_kw)
    gen = CurriculumMaskGenerator(
        input_size=(slice_size, slice_size),
        patch_size=patch_size,
        nenc=1,
        npred=4,
        curriculum_cfg=cfg,
        world_size=1,
        rank=0,
    )
    gen.set_epoch(50, 100)  # ramp fully engaged
    return gen


def load_real_slices(data_dir, n, slice_size):
    """Return up to n (3, S, S) float tensors sampled across volumes/depths."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        return []
    from PIL import Image
    out, labels = [], []
    # Spread picks across files and a range of depths (incl. off-centre slices
    # which tend to be the tilted/atypical ones).
    depths = [40, 70, 100, 130, 160]
    fi = 0
    while len(out) < n and fi < len(files):
        data = np.load(files[fi], allow_pickle=True)
        vol = data["oct_bscans"]  # (D, H, W) uint8
        d = depths[len(out) % len(depths)] % vol.shape[0]
        sl = vol[d]
        pil = Image.fromarray(sl, mode="L").resize(
            (slice_size, slice_size), Image.BILINEAR
        ).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        out.append(torch.from_numpy(arr).permute(2, 0, 1))
        labels.append("vol%d d%d" % (fi, d))
        fi += 1
    return list(zip(out, labels))


def make_synthetic_slices(slice_size):
    """Synthetic OCT-like slices to exercise edge cases when no data is local."""
    S = slice_size
    cases = []

    def band(y0, y1, tilt=0, rnfl=True):
        img = torch.zeros(3, S, S)
        for x in range(S):
            shift = int(tilt * (x - S / 2) / (S / 2))
            a = max(0, y0 + shift)
            b = min(S, y1 + shift)
            img[:, a:b, x] = 0.75
            if rnfl:
                img[:, a:min(b, a + 12), x] += 0.2  # brighter RNFL near top
        return img.clamp(max=1.0)

    cases.append((band(96, 160), "centred band"))
    cases.append((band(60, 120), "band shifted UP"))
    cases.append((band(150, 210), "band shifted DOWN"))
    cases.append((band(96, 160, tilt=40), "tilted band"))
    cases.append((band(110, 150, rnfl=False), "thin/faint band"))
    return cases


def render(samples, gen, patch_size, out_path):
    n = len(samples)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    W = gen.width
    for i, (img, label) in enumerate(samples):
        ax = axes[i]
        ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1)
        grid = gen._anatomical_prior_weight_grid_for_image(img)  # (H, W)
        region = float(grid.mean())
        # Faint red = the BIAS region (where target blocks are encouraged to
        # land), not a pixel mask.
        ys, xs = torch.nonzero(grid, as_tuple=True)
        for y, x in zip(ys.tolist(), xs.tolist()):
            ax.add_patch(Rectangle(
                (x * patch_size, y * patch_size), patch_size, patch_size,
                linewidth=0, facecolor="red", alpha=0.13,
            ))
        # Solid yellow = ONE sampled set of the 4 actual I-JEPA target blocks
        # at the current r_t — this is what the encoder truly loses.
        masks_enc, masks_pred = gen.generate(
            batch_size=1, imgs_cpu=img.unsqueeze(0), h_for_cluster=None
        )
        masked = set()
        for m in masks_pred:
            masked.update(m[0].tolist())
        for idx in masked:
            y, x = idx // W, idx % W
            ax.add_patch(Rectangle(
                (x * patch_size, y * patch_size), patch_size, patch_size,
                linewidth=0, facecolor="yellow", alpha=0.45,
            ))
        ax.set_title(
            "%s\nbias band=%.0f%%  masked=%.0f%%"
            % (label, region * 100, 100 * len(masked) / (gen.height * W)),
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(
        "ORACLE v2 build-check — faint red = bias band, yellow = actual 4 "
        "target blocks (what the encoder loses)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="", help="dir with FairVision .npz")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--slice_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--out", default="results/summary/oracle_build_check.png")
    ap.add_argument("--oracle_region_frac", type=float, default=0.28)
    ap.add_argument("--oracle_lateral_frac", type=float, default=0.6)
    ap.add_argument("--oracle_row_offset", type=float, default=0.0)
    args = ap.parse_args()

    gen = make_oracle(
        slice_size=args.slice_size,
        patch_size=args.patch_size,
        oracle_region_frac=args.oracle_region_frac,
        oracle_lateral_frac=args.oracle_lateral_frac,
        oracle_row_offset=args.oracle_row_offset,
    )

    samples = []
    if args.data_dir and os.path.isdir(args.data_dir):
        samples = load_real_slices(args.data_dir, args.n, args.slice_size)
        print("Loaded %d real slices from %s" % (len(samples), args.data_dir))
    if not samples:
        print("No real data — using synthetic OCT-like slices (incl. edge cases).")
        samples = make_synthetic_slices(args.slice_size)

    # Report region-size stats so we catch "too little / too much" numerically.
    regions = [float(gen._anatomical_prior_weight_grid_for_image(im).mean())
               for im, _ in samples]
    print("region size: min=%.3f mean=%.3f max=%.3f (target ~%.2f, band 0.20-0.40)"
          % (min(regions), sum(regions) / len(regions), max(regions),
             args.oracle_region_frac))
    if min(regions) < 0.20 or max(regions) > 0.40:
        print("  WARN: some slices fall outside the 0.20-0.40 region band.")

    # GUARD: training feeds ImageNet-normalized tensors into generate(), this
    # build-check uses raw [0,1].  The detector is affine-invariant, so the two
    # MUST produce identical masks — assert it so a future change can't silently
    # break the match (the original clamp(min=0) bug).
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    mismatches = 0
    for im, label in samples:
        g_raw = gen._anatomical_prior_weight_grid_for_image(im)
        g_norm = gen._anatomical_prior_weight_grid_for_image((im - mean) / std)
        if not torch.equal(g_raw, g_norm):
            mismatches += 1
            print("  WARN: raw vs normalized mask differ on '%s'" % label)
    if mismatches == 0:
        print("raw vs ImageNet-normalized: identical masks on all %d slices "
              "(detector is normalization-invariant, matches training input)."
              % len(samples))

    render(samples, gen, args.patch_size, args.out)


if __name__ == "__main__":
    main()
