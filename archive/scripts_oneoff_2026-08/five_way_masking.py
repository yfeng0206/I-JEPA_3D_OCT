#!/usr/bin/env python3
"""Five-way masking comparison on identical slices: distributions + spatial maps.

Arms
    random    MaskCollator, uniform block placement
    oracle    CurriculumMaskGenerator(anatomical_prior), intensity retina band
    envelope  CurriculumMaskGenerator(mirage_envelope), MIRAGE rejection sampling
    cover     prototype: greedy exact coverage with a visible-anatomy floor
    blobs     current arm: irregular anatomy targets, pred_target_k = 16

Everything is scored against the same anatomy reference (MIRAGE occupancy >= 0.25)
and every arm sees the exact same crop.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from budget_mask_prototype import build_budget_targets  # noqa: E402
from budget_mask_visualize import OCC_T, load_slices  # noqa: E402
from cover_mask_prototype import cover_targets, GRID, NPATCH  # noqa: E402
from mask_composition_probe import build_arms  # noqa: E402

GUIDE_DIR = (r"C:\jepa_data\mirage_soft_guides"
             r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy")

ARMS = ["random", "oracle", "envelope", "cover", "blobs"]
COL = {"random": "#888888", "oracle": "#2ca02c", "envelope": "#4363d8",
       "cover": "#e6194b", "blobs": "#ff7f0e"}
# measured ep100 downstream AUC where it exists
AUC = {"random": 0.8746, "oracle": 0.8855, "envelope": 0.8807,
       "cover": None, "blobs": None}


def union_from_masks(m_pred, b=0):
    u = np.zeros(NPATCH, bool)
    for g in m_pred:
        u[g[b].numpy()] = True
    return u.reshape(GRID, GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=800)
    ap.add_argument("--floor", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=r"D:\jepa_phase0\reports\budget_masks")
    args = ap.parse_args()
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ds, idxs = load_slices(args.slices, args.seed, args.slices)
    prod = build_arms(GUIDE_DIR)          # random / oracle / envelope / anatomy
    rng = random.Random(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed)

    rec = {a: [] for a in ARMS}
    heat = {a: np.zeros((GRID, GRID)) for a in ARMS}
    n = 0

    for i in idxs:
        img_t, guide, valid = ds[i]
        occ = guide[0].numpy()
        anat = occ >= OCC_T
        n_anat = int(anat.sum())
        if n_anat == 0:
            continue
        cs = ([guide[2].numpy(), guide[3].numpy()] if guide.shape[0] >= 4
              else [occ, np.zeros_like(occ)])
        imgs = img_t[None]
        gg, vv = guide[None], valid[None]

        unions = {}
        _, _, mp = prod["random"]([img_t])
        unions["random"] = union_from_masks(mp)
        _, mp = prod["oracle"].generate(batch_size=1, imgs_cpu=imgs)
        unions["oracle"] = union_from_masks(mp)
        _, mp = prod["envelope"].generate(batch_size=1, guide_grids=gg,
                                          guide_valid=vv)
        unions["envelope"] = union_from_masks(mp)
        rects, _ = cover_targets(cs, n=4, leave_frac=args.floor,
                                 min_visible_frac=args.floor, gen=gen, rng=rng)
        unions["cover"] = np.logical_or.reduce(rects)
        blobs, _ = build_budget_targets(cs, n=4, k=16, rng=rng)
        unions["blobs"] = np.logical_or.reduce(blobs)

        for a in ARMS:
            u = unions[a]
            hid = int((anat & u).sum())
            rec[a].append(dict(
                anat_hidden_frac=hid / n_anat,
                anat_hidden=hid,
                anat_visible=n_anat - hid,
                union=int(u.sum()),
                bg_hidden=int((~anat & u).sum()),
                pct_hidden_is_anat=100.0 * hid / max(int(u.sum()), 1),
            ))
            heat[a] += u
        n += 1

    summary = {}
    for a in ARMS:
        arr = {k: np.array([r[k] for r in rec[a]], float) for k in rec[a][0]}
        summary[a] = {k: dict(mean=float(v.mean()), sd=float(v.std()),
                              p05=float(np.percentile(v, 5)),
                              p95=float(np.percentile(v, 95)),
                              min=float(v.min()), max=float(v.max()))
                      for k, v in arr.items()}
        summary[a]["pct_zero_anat_visible"] = float(
            100.0 * np.mean(arr["anat_visible"] <= 0))
        summary[a]["ep100_auc"] = AUC[a]
        summary[a]["slices"] = n
    (out / "five_way_masking.json").write_text(json.dumps(summary, indent=2))

    keys = [("anat_hidden_frac", "anatomy hidden (frac)"),
            ("anat_visible", "anatomy visible (cells)"),
            ("union", "union hidden (cells)"),
            ("bg_hidden", "background hidden (cells)"),
            ("pct_hidden_is_anat", "% of hidden that is anatomy")]
    print(f"\n{'metric':30s}" + "".join(f"{a:>18s}" for a in ARMS))
    print("-" * (30 + 18 * len(ARMS)))
    for k, lbl in keys:
        print(f"{lbl:30s}" + "".join(
            f"{summary[a][k]['mean']:11.2f}+-{summary[a][k]['sd']:5.2f}"
            for a in ARMS))
    print(f"{'zero anatomy visible %':30s}" + "".join(
        f"{summary[a]['pct_zero_anat_visible']:18.1f}" for a in ARMS))
    print(f"{'ep100 test AUC':30s}" + "".join(
        f"{(('%.4f' % AUC[a]) if AUC[a] else '-'):>18s}" for a in ARMS))
    print(f"\nslices: {n}")

    # ---------------- figure ------------------------------------------------
    fig = plt.figure(figsize=(17, 9.5))
    gs = gridspec.GridSpec(2, len(ARMS), height_ratios=[1.05, 1.0],
                           hspace=0.32, wspace=0.25)

    for c, (k, lbl, rng_) in enumerate([
            ("anat_hidden_frac", "fraction of ANATOMY hidden", (0, 1)),
            ("anat_visible", "anatomy cells left VISIBLE", (0, 45)),
            ("union", "union hidden (cells / 256)", (40, 200)),
            ("bg_hidden", "BACKGROUND cells hidden", (0, 140)),
            ("pct_hidden_is_anat", "% of hidden that is anatomy", (0, 100))]):
        ax = fig.add_subplot(gs[0, c])
        for a in ARMS:
            v = np.array([r[k] for r in rec[a]], float)
            ax.hist(v, bins=38, range=rng_, histtype="step", lw=1.8,
                    label=a, color=COL[a], density=True)
        ax.set_title(lbl, fontsize=9.5)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.legend(fontsize=7.5)

    for c, a in enumerate(ARMS):
        ax = fig.add_subplot(gs[1, c])
        im = ax.imshow(heat[a] / max(n, 1), cmap="magma", vmin=0, vmax=1)
        auc = f"  ep100 AUC {AUC[a]:.4f}" if AUC[a] else "  (not run)"
        ax.set_title(f"{a}: P(hidden){auc}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Five masking policies on the SAME {n} FairVision slices — "
                 "distributions (top) and where they hide (bottom)", fontsize=13)
    p = out / "five_way_masking.png"
    fig.savefig(p, dpi=118, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
