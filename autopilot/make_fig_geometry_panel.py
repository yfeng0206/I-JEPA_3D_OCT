"""Regenerate fig_geometry_panel from the same artifact that backs Table 2.

The published figure was carried forward from an earlier draft.  Table 2 was
later regenerated from the production samplers, so the two disagreed while
both captions claimed the same 600-slice measurement: RANDOM anatomy hidden
read 52.2 percent in the figure against 54.0 in the table, and 157.7 loss
slots against 159.9.  A blind reviewer caught it.

FIGURE_REGEN.md records that this figure had no generator, which is why it
was not refreshed with the table.  This script is that generator.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")           # must precede pyplot; never render to screen
import matplotlib.pyplot as plt

ART = ("results/masking/table2_geometry/"
       "mask_geometry_600slices_bs1_coverf021_seed42.json")
OUT = "paper/genai4health2026/figures/fig_geometry_panel"

# Display name -> artifact key.  The stored JSON still uses the old arm keys;
# the figure keeps the artifact labels that the caption already decodes.
POLICIES = [("random", "random"), ("oracle", "oracle"), ("envelope", "envelope"),
            ("cover-f0.21", "cover"), ("anatomy-v2", "anatomy")]

# Panel title, artifact field, scale, axis label.
PANELS = [
    ("Masking ratio", "hidden_frac_of_grid", 100.0, "% of patch grid"),
    ("Context kept", "ctx_frac_of_grid", 100.0, "% of patch grid"),
    ("Predictor loss slots", "n_slots_mean", 1.0, "slots per image"),
    ("Anatomy hidden", "hidden_share_of_all_anat", 1.0, "% of all tissue cells"),
]

# Colourblind-safe (Okabe-Ito), with the anatomy arm set apart because it is
# the one that diverges on every axis.
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00"]


def main():
    art = json.load(open(ART, encoding="utf-8"))
    meta = art["_meta"]
    assert meta["slices"] == 600 and meta["batch_size"] == 1, meta
    assert meta["cover_floor"] == 0.21, meta

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.15))
    names = [n for n, _ in POLICIES]
    xs = range(len(POLICIES))

    for ax, (title, field, scale, ylab) in zip(axes, PANELS):
        vals = [art[k][field] * scale for _, k in POLICIES]
        bars = ax.bar(xs, vals, color=COLOURS, width=0.68,
                      edgecolor="black", linewidth=0.4)
        # Hatch the anatomy arm so the panels survive greyscale printing.
        bars[-1].set_hatch("///")
        for x, v in zip(xs, vals):
            ax.text(x, v + max(vals) * 0.025, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=8.2)
        ax.set_title(title, fontsize=10.5, pad=7)
        ax.set_ylabel(ylab, fontsize=8.8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8.2)
        ax.set_ylim(0, max(vals) * 1.20)      # zero baseline, never truncated
        ax.tick_params(axis="y", labelsize=8.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.35, alpha=0.35)

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.text(0.5, 0.008,
             "600 FairVision Training slices (24 volumes x 25), 16x16 grid, "
             "seed 42, COVER floor f=0.21; means, no interval drawn.",
             ha="center", fontsize=7.6, color="#333333")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext, kw in (("png", {"dpi": 320}), ("pdf", {})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)

    print("regenerated from", ART)
    for name, key in POLICIES:
        r = art[key]
        print(f"  {name:12s} ratio {r['hidden_frac_of_grid']*100:5.1f}%  "
              f"ctx {r['ctx_frac_of_grid']*100:5.1f}%  "
              f"slots {r['n_slots_mean']:6.1f}  "
              f"anat-hidden {r['hidden_share_of_all_anat']:5.1f}%")


if __name__ == "__main__":
    main()
