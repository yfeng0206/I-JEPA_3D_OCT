"""A readable version of the region-importance result.

The first attempt zoomed the y-axis to 0.845-0.88, which made 0.6-point gaps look
enormous and hid the actual finding: all three poolings land close together and
far above chance.  This version

  * shows what each probe literally SEES (the background probe never sees a
    single retina patch), and
  * plots AUC against the 0.5 chance floor, plus the share of above-chance
    signal each region retains, which is the number that answers "is background
    important too".
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.oct_volumes import OCTVolumeDataset            # noqa: E402
from scripts.downstream_region_auc import load_mask, GRID        # noqa: E402

OUT = pathlib.Path(r"D:\jepa_phase0\reports\downstream_region_auc")
MASKC = r"D:\jepa_phase0\reports\anatomy_mask_cache"
DATA = r"D:\jepa_phase0\fairvision-glaucoma\data"

rows = {}
for shard in sorted(OUT.glob("*/region_auc.json")):
    for r in json.loads(shard.read_text()):
        rows[r["tag"]] = dict(all=r["all"]["test"], anatomy=r["anatomy"]["test"],
                              background=r["background"]["test"])
order = [t for t in ("random_ep50", "oracle_ep50", "envelope_ep50", "blob_ep50")
         if t in rows]
print("arms:", order)

ds = OCTVolumeDataset(f"{DATA}\\Test", num_slices=100, slice_size=256,
                      return_label=True)
M = load_mask(MASKC, "Test", 100, len(ds))

fig = plt.figure(figsize=(17, 10.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.30, wspace=0.22)

# ---------- row 1: what each probe actually sees ---------------------------
vi, si = 17, 30
vol, _ = ds[vi]
img = vol[si].permute(1, 2, 0).numpy()
img = (img - img.min()) / (np.ptp(img) + 1e-6)
m = M[vi][si].reshape(GRID, GRID)
big = np.kron(m, np.ones((16, 16), dtype=bool))[..., None]

views = [
    (img, "1. The B-scan the encoder sees", "all 256 patches"),
    (img * big, "2. What the ANATOMY probe pools", f"only the {100*m.mean():.0f}% retina patches"),
    (img * (~big), "3. What the BACKGROUND probe pools",
     f"only the {100*(1-m.mean()):.0f}% dark patches — NO retina at all"),
]
for k, (im, title, sub) in enumerate(views):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(sub, fontsize=10)

# ---------- row 2a: AUC against the chance floor ---------------------------
ax = fig.add_subplot(gs[1, :2])
x = np.arange(len(order)); w = 0.26
cols = {"all": "#9e9e9e", "anatomy": "#e6194b", "background": "#2b5fd9"}
lbl = {"all": "all 256 cells (published protocol)",
       "anatomy": "anatomy cells only",
       "background": "background cells only"}
for i, reg in enumerate(("all", "anatomy", "background")):
    v = [rows[t][reg] for t in order]
    ax.bar(x + (i - 1) * w, v, w, label=lbl[reg], color=cols[reg], edgecolor="k")
    for xi, vv in zip(x + (i - 1) * w, v):
        ax.text(xi, vv + 0.004, f"{vv:.3f}", ha="center", fontsize=9,
                fontweight="bold")
ax.axhline(0.5, color="k", ls="--", lw=1.5)
ax.text(len(order) - 0.55, 0.513, "chance = 0.50", fontsize=10)
ax.set_ylim(0.5, 0.95)
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=11)
ax.set_ylabel("downstream TEST AUC (glaucoma)", fontsize=11)
ax.set_title("Plotted against chance, all three are nearly the same height:\n"
             "background alone is almost as predictive as the retina alone",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=.3, axis="y")

# ---------- row 2b: share of above-chance signal retained ------------------
ax2 = fig.add_subplot(gs[1, 2])
for i, reg in enumerate(("anatomy", "background")):
    v = [100 * (rows[t][reg] - 0.5) / (rows[t]["all"] - 0.5) for t in order]
    ax2.bar(x + (i - 0.5) * 0.34, v, 0.34, label=lbl[reg], color=cols[reg],
            edgecolor="k")
    for xi, vv in zip(x + (i - 0.5) * 0.34, v):
        ax2.text(xi, vv + 1.2, f"{vv:.0f}%", ha="center", fontsize=9,
                 fontweight="bold")
ax2.axhline(100, color="k", ls="--", lw=1.5)
ax2.text(-0.45, 102, "= all-cell pooling", fontsize=9)
ax2.set_ylim(0, 125)
ax2.set_xticks(x); ax2.set_xticklabels(order, rotation=20, fontsize=9)
ax2.set_ylabel("% of above-chance signal kept", fontsize=10)
ax2.set_title("How much signal survives\nwhen you keep ONLY that region",
              fontsize=11, fontweight="bold")
ax2.grid(alpha=.3, axis="y")

msg = ("READ THIS:  Background is NOT dead space — pooling only dark cells still classifies glaucoma at "
       f"{rows[order[0]]['background']:.3f} AUC ({100*(rows[order[0]]['background']-0.5)/(rows[order[0]]['all']-0.5):.0f}% of the full signal).\n"
       "BUT this is not evidence that black pixels contain disease: in a ViT every patch token attends to the whole image, so a token sitting at a\n"
       "background POSITION has already absorbed retinal information.  The honest reading is that the signal is READABLE from background positions.\n"
       "Practical note: anatomy-only BEATS the standard all-cell pooling, so the published protocol loses AUC by diluting the retina with 77% background.")
fig.text(0.5, 0.005, msg, ha="center", fontsize=10.5, family="monospace",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff6d5", edgecolor="#b0a060"))

fig.suptitle("Does background matter for the downstream glaucoma decision?  "
             "Same frozen encoder, same volumes and slices — only the pooled region changes",
             fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=[0, 0.085, 1, 0.965])
dst = OUT / "region_auc_explained.png"
fig.savefig(dst, dpi=135)
print("wrote", dst)
