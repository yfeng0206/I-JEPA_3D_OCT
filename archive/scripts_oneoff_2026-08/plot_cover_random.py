"""Visualise COVER-then-RANDOM masks and the block-role distribution."""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = pathlib.Path(r"D:\jepa_phase0\reports\cover_random")
PATCH = 16

ex = np.load(OUT / "examples.npy", allow_pickle=True)
df = pd.read_csv(OUT / "per_slice.csv")
cr = df[df.variant == "cover_random_legal"]

KIND_COL = {"cover": "#2b5fd9", "random_legal": "#ff8c00",
            "random": "#ff8c00", "random_violation": "#d62728",
            "transition": "#7d3c98", "unguided": "#888888",
            "fallback": "#d62728"}

n_show = min(6, len(ex))
fig = plt.figure(figsize=(17, 9.6))
gs = fig.add_gridspec(3, n_show, height_ratios=[1, 1, 0.95], hspace=0.28)

for k in range(n_show):
    idx, img, g, masks, kinds, meta = ex[k]
    im = img.mean(0)
    im = (im - im.min()) / (np.ptp(im) + 1e-6)
    anat = (g[0] >= 0.25)

    # row 0: the slice with the anatomy band outlined
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(im, cmap="gray")
    ax.contour(np.kron(anat, np.ones((PATCH, PATCH))), levels=[0.5],
               colors="#00d000", linewidths=1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice {idx}\nanatomy outlined", fontsize=9)

    # row 1: what the predictor is asked to reconstruct
    ax = fig.add_subplot(gs[1, k])
    shown = im.copy()
    union = np.zeros_like(anat, dtype=bool)
    for m in masks:
        union |= m
    big = np.kron(union, np.ones((PATCH, PATCH), dtype=bool))
    shown = np.where(big, 0.0, shown)
    ax.imshow(shown, cmap="gray")
    for m, kind in zip(masks, kinds):
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        ax.add_patch(mpatches.Rectangle(
            (xs.min() * PATCH, ys.min() * PATCH),
            (xs.max() - xs.min() + 1) * PATCH, (ys.max() - ys.min() + 1) * PATCH,
            fill=False, lw=2.2, edgecolor=KIND_COL.get(kind, "#888888")))
    ax.contour(np.kron(anat, np.ones((PATCH, PATCH))), levels=[0.5],
               colors="#00d000", linewidths=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    vis = 100 - meta["hidden"]
    ax.set_title(f"hidden {meta['hidden']:.0f}%  |  {vis:.0f}% anatomy left",
                 fontsize=9)

handles = [mpatches.Patch(color="#2b5fd9", label="COVER block (greedy, hides anatomy)"),
           mpatches.Patch(color="#ff8c00", label="RANDOM block (uniform, floor-constrained)"),
           mpatches.Patch(color="#00d000", label="anatomy boundary")]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, 0.985))

# ---- row 2: distributions --------------------------------------------------
ax = fig.add_subplot(gs[2, :2])
vc = cr.n_cover.value_counts().sort_index()
vr = cr.n_random.value_counts().sort_index()
allk = sorted(set(vc.index) | set(vr.index))
x = np.arange(len(allk)); w = 0.38
ax.bar(x - w / 2, [100 * vc.get(k, 0) / len(cr) for k in allk], w,
       label="COVER blocks", color="#2b5fd9", edgecolor="k")
ax.bar(x + w / 2, [100 * vr.get(k, 0) / len(cr) for k in allk], w,
       label="RANDOM blocks", color="#ff8c00", edgecolor="k")
ax.set_xticks(x); ax.set_xticklabels(allk)
ax.set_xlabel("blocks per image (of 4)"); ax.set_ylabel("% of slices")
ax.set_title(f"How the 4 blocks get spent\nmean {cr.n_cover.mean():.2f} cover + "
             f"{cr.n_random.mean():.2f} random", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")

ax = fig.add_subplot(gs[2, 2:4])
ax.hist(cr.hidden_pct, bins=40, color="#2b5fd9", edgecolor="k")
ax.axvline(85, color="r", ls="--", lw=2, label="85% target")
ax.set_xlabel("% anatomy hidden"); ax.set_ylabel("slices")
ax.set_title(f"Anatomy hidden\nmean {cr.hidden_pct.mean():.1f}%  "
             f"p10 {cr.hidden_pct.quantile(.1):.1f}  p90 {cr.hidden_pct.quantile(.9):.1f}",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")

ax = fig.add_subplot(gs[2, 4:])
comp = df.groupby("variant").agg(
    floor_ok=("floor_ok", lambda s: 100 * s.mean()),
    usable=("ok", lambda s: 100 * s.mean()),
    violation=("floor_violation", lambda s: 100 * s.mean())).reindex(
    ["cover_random_legal", "cover_random_free", "cover_transition"])
xx = np.arange(len(comp)); w = 0.27
ax.bar(xx - w, comp.floor_ok, w, label="floor respected %", color="#2ca02c", edgecolor="k")
ax.bar(xx, comp.usable, w, label="usable %", color="#2b5fd9", edgecolor="k")
ax.bar(xx + w, comp.violation, w, label="FAIL: floor breach %", color="#d62728", edgecolor="k")
ax.set_xticks(xx)
ax.set_xticklabels(["random\n(floor-constrained)", "random\n(unconstrained)",
                    "transition\n(shipped)"], fontsize=8)
ax.set_ylim(0, 108); ax.set_ylabel("%")
ax.set_title("Does it hold the 15% floor?", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")

fig.suptitle("COVER-then-RANDOM: greedily hide anatomy to 85%, keep 15% visible, "
             "spend leftover blocks as plain JEPA rectangles",
             fontsize=13, fontweight="bold", y=0.999)
fig.tight_layout(rect=[0, 0, 1, 0.945])
dst = OUT / "cover_random_visual.png"
fig.savefig(dst, dpi=135)
print("wrote", dst)
