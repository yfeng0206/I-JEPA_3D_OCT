#!/usr/bin/env python3
"""How does COVER actually differ from mirage_envelope?  Distributions, not prose.

Both use 4 stock rectangles placed with the MIRAGE guide, so the difference is
entirely in the PLACEMENT RULE:

  mirage_envelope   rejection sampling.  Draw a random location, accept if it
                    passes min_block_fill / min_retina_visible / overlap
                    tolerance / spread.  Up to mirage_max_attempts tries, then
                    fall back to a uniform random block.  Blocks are drawn
                    INDEPENDENTLY -- block 2 does not know what block 1 covered.

  cover             exhaustive greedy.  Score all <=256 positions by the
                    anatomy mass still UNCOVERED, take the best, subtract it,
                    repeat.  Blocks are therefore COMPLEMENTARY, and a hard
                    floor keeps some tissue visible.

This script runs both on identical slices and compares the distributions.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from budget_mask_visualize import OCC_T, load_slices  # noqa: E402
from cover_mask_prototype import cover_targets, GRID, NPATCH  # noqa: E402
from mask_composition_probe import BASE_KW, CURR_COMMON  # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402

GUIDE_DIR = (r"C:\jepa_data\mirage_soft_guides"
             r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy")


def make_envelope():
    g = CurriculumMaskGenerator(
        **BASE_KW,
        curriculum_cfg=dict(
            mode="mirage_envelope", mirage_guide_dir=GUIDE_DIR,
            mirage_occupancy_threshold=OCC_T,
            mirage_min_block_fill=0.4, mirage_min_retina_visible=0.25,
            mirage_max_attempts=30, mirage_spread=True,
            mirage_overlap_tolerance=0.25, **CURR_COMMON),
    )
    g.set_epoch(50, 100)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=1500)
    ap.add_argument("--floor", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=r"D:\jepa_phase0\reports\budget_masks")
    args = ap.parse_args()
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ds, idxs = load_slices(args.slices, args.seed, args.slices)
    env = make_envelope()
    rng = random.Random(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed)

    rec = {"envelope": [], "cover": []}
    heat = {"envelope": np.zeros((GRID, GRID)), "cover": np.zeros((GRID, GRID))}
    heat_ctx = {"envelope": np.zeros((GRID, GRID)),
                "cover": np.zeros((GRID, GRID))}
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

        # ---- envelope: batch of 1 so no global-min truncation ------------
        m_enc, m_pred = env.generate(
            batch_size=1, guide_grids=guide[None], guide_valid=valid[None])
        eu = np.zeros(NPATCH, bool)
        for g_ in m_pred:
            eu[g_[0].numpy()] = True
        eu = eu.reshape(GRID, GRID)

        # ---- cover -------------------------------------------------------
        rects, info = cover_targets(cs, n=4, leave_frac=args.floor,
                                    min_visible_frac=args.floor,
                                    gen=gen, rng=rng)
        cu = np.logical_or.reduce(rects)

        for name, u in (("envelope", eu), ("cover", cu)):
            hid = int((anat & u).sum())
            rec[name].append(dict(
                anat_cells=n_anat,
                union=int(u.sum()),
                anat_hidden=hid,
                anat_hidden_frac=hid / n_anat,
                anat_visible=n_anat - hid,
                bg_hidden=int((~anat & u).sum()),
                union_pct_anat=100.0 * hid / max(int(u.sum()), 1),
            ))
            heat[name] += u
            heat_ctx[name] += (~u)
        n += 1

    # ---------------- summary ----------------------------------------------
    summary = {}
    for name in ("envelope", "cover"):
        arr = {k: np.array([r[k] for r in rec[name]], float)
               for k in rec[name][0]}
        summary[name] = {k: dict(mean=float(v.mean()), sd=float(v.std()),
                                 p05=float(np.percentile(v, 5)),
                                 p50=float(np.percentile(v, 50)),
                                 p95=float(np.percentile(v, 95)),
                                 min=float(v.min()), max=float(v.max()))
                         for k, v in arr.items()}
        summary[name]["pct_zero_anat_visible"] = float(
            100.0 * np.mean(arr["anat_visible"] <= 0))
        summary[name]["slices"] = n
    (out / "cover_vs_envelope.json").write_text(json.dumps(summary, indent=2))

    hdr = f"{'metric':26s} {'envelope':>22s} {'cover':>22s}"
    print("\n" + hdr); print("-" * len(hdr))
    for k, lbl in (("anat_hidden_frac", "anatomy hidden (frac)"),
                   ("anat_hidden", "anatomy hidden (cells)"),
                   ("anat_visible", "anatomy visible (cells)"),
                   ("union", "union hidden (cells)"),
                   ("bg_hidden", "background hidden"),
                   ("union_pct_anat", "% of hidden that is anat")):
        e, c = summary["envelope"][k], summary["cover"][k]
        print(f"{lbl:26s} {e['mean']:8.2f} +-{e['sd']:5.2f} [{e['p05']:6.2f},{e['p95']:6.2f}]"
              f" {c['mean']:8.2f} +-{c['sd']:5.2f} [{c['p05']:6.2f},{c['p95']:6.2f}]")
    print(f"{'zero anatomy visible':26s} "
          f"{summary['envelope']['pct_zero_anat_visible']:21.1f}% "
          f"{summary['cover']['pct_zero_anat_visible']:21.1f}%")
    print(f"\nslices compared: {n}")

    # ---------------- figures ----------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    specs = [("anat_hidden_frac", "fraction of ANATOMY hidden", (0, 1)),
             ("union", "union hidden (cells / 256)", None),
             ("anat_visible", "anatomy cells left VISIBLE", None)]
    for c, (key, title, rng_) in enumerate(specs):
        ax = axes[0, c]
        for name, col in (("envelope", "#4363d8"), ("cover", "#e6194b")):
            v = np.array([r[key] for r in rec[name]], float)
            ax.hist(v, bins=40, range=rng_, alpha=0.55, label=name, color=col,
                    density=True)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylabel("density", fontsize=8)

    for c, name in enumerate(("envelope", "cover")):
        ax = axes[1, c]
        im = ax.imshow(heat[name] / max(n, 1), cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{name}: P(cell is HIDDEN)", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1, 2]
    d = (heat["cover"] - heat["envelope"]) / max(n, 1)
    im = ax.imshow(d, cmap="bwr", vmin=-0.5, vmax=0.5)
    ax.set_title("cover − envelope  (red = cover hides more)", fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"COVER vs mirage_envelope — same {n} FairVision slices, "
                 f"same guide, same block shape/size/count", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out / "cover_vs_envelope.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print("wrote", p)
    print("wrote", out / "cover_vs_envelope.json")


if __name__ == "__main__":
    main()
