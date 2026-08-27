"""Visuals for the region-importance experiment.

Panel A shows what the anatomy/background split actually looks like on real Test
slices, so the numbers can be sanity-checked by eye rather than trusted blind.
Panel B shows the downstream AUC obtained from each region's pooled features.

Panel A deliberately draws the mask the pooling ACTUALLY used -- the cached
per-slice mask from scripts/fit_anatomy_mask.py -- not a fresh recomputation.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.oct_volumes import OCTVolumeDataset          # noqa: E402
from scripts.downstream_region_auc import load_mask, GRID, NPATCH  # noqa: E402

OUT = pathlib.Path(r"D:\jepa_phase0\reports\downstream_region_auc")
MASKC = r"D:\jepa_phase0\reports\anatomy_mask_cache"
DATA = r"D:\jepa_phase0\fairvision-glaucoma\data"

# ---- gather whatever arms have finished -----------------------------------
rows = []
for shard in sorted(OUT.glob("*/region_auc.json")):
    for r in json.loads(shard.read_text()):
        rows.append(dict(tag=r["tag"], all=r["all"]["test"],
                         anatomy=r["anatomy"]["test"],
                         background=r["background"]["test"]))
rows = {r["tag"]: r for r in rows}
order = [t for t in ("random_ep50", "oracle_ep50", "envelope_ep50", "blob_ep50")
         if t in rows]
print("arms available:", order)

ds = OCTVolumeDataset(f"{DATA}\\Test", num_slices=100, slice_size=256,
                      return_label=True)
M = load_mask(MASKC, "Test", 100, len(ds))

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1])

# ---- Panel A: the split, on real slices -----------------------------------
picks = [(3, 50), (17, 30), (42, 70), (61, 45)]
for k, (vi, si) in enumerate(picks):
    vol, lab = ds[vi]
    img = vol[si].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    m = M[vi][si].reshape(GRID, GRID)
    big = np.kron(m, np.ones((16, 16), dtype=bool))

    ov = img.copy()
    ov[..., 1] = np.where(big, np.clip(ov[..., 1] + 0.35, 0, 1), ov[..., 1])   # anatomy -> green
    ov[..., 0] = np.where(~big, np.clip(ov[..., 0] + 0.28, 0, 1), ov[..., 0])  # background -> red

    ax = fig.add_subplot(gs[0, k])
    ax.imshow(ov)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"vol {vi} slice {si} — anatomy {100*m.mean():.0f}% of cells",
                 fontsize=9)
    if k == 0:
        ax.set_ylabel("green = anatomy pool\nred = background pool", fontsize=9)

# ---- Panel B: AUC by pooled region ----------------------------------------
ax = fig.add_subplot(gs[1, :])
x = np.arange(len(order)); w = 0.26
cols = {"all": "#888888", "anatomy": "#e6194b", "background": "#4363d8"}
for i, reg in enumerate(("all", "anatomy", "background")):
    v = [rows[t][reg] for t in order]
    b = ax.bar(x + (i - 1) * w, v, w, label=reg, color=cols[reg], edgecolor="k")
    for xi, vi_ in zip(x + (i - 1) * w, v):
        ax.text(xi, vi_ + 0.0012, f"{vi_:.4f}", ha="center", fontsize=8)

vals = [rows[t][r] for t in order for r in ("all", "anatomy", "background")]
ax.set_ylim(min(vals) - 0.012, max(vals) + 0.010)
ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_ylabel("downstream TEST AUC (glaucoma)")
ax.set_title("Which cells carry the downstream signal?  Same frozen encoder, "
             "same volumes/slices — only the POOLED REGION differs\n"
             "(1000 test volumes x 25 stratified slices)", fontsize=11)
ax.legend(); ax.grid(alpha=.3, axis="y")

fig.tight_layout()
dst = OUT / "region_auc_visual.png"
fig.savefig(dst, dpi=140)
print("wrote", dst)
