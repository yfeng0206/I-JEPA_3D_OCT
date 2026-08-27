"""Redraw the specificity ladder as a zero-centred forest plot.

The published figure was a bar chart on an AUC axis truncated at 0.8600.  Our
own figure audit called it "the worst encoding in the submission": arms whose
AUCs differ by about 1.4 percent relative were drawn with a 3.9x apparent
height ratio.  The caption disclosed the truncation and told the reader to
consult the table instead, which is an admission that the figure was doing no
work.

Plotting the paired difference against the unguided null removes the problem
rather than disclosing it, because zero is a meaningful baseline for a
difference, so no axis choice can exaggerate anything.  It also plots the
quantity the paper's inference actually rests on: per-arm error bars are not
shown anywhere in this paper, since between-case variance cancels in a pairing
and would understate the evidence.

Ordering the arms by how much anatomy they hide keeps the figure's original
job, which Figure 2(b) does not do; that one is organised by epoch.

Values are read from the generated macro file, so this figure cannot drift
from the tables the way the bar chart did.
"""

import os
import re

import matplotlib
matplotlib.use("Agg")           # must precede pyplot; never render to screen
import matplotlib.pyplot as plt

AUTO = "paper/genai4health2026/auto/auto_numbers.tex"
GEOM = ("results/masking/table2_geometry/"
        "mask_geometry_600slices_bs1_coverf021_seed42.json")
OUT = "paper/genai4health2026/figures/fig_specificity_ladder"

# display label, macro stem, artifact key for the anatomy-hidden ordering
ARMS = [
    ("random (null)", None, "random"),
    ("centroid", "DOracleRandomEpFifty", "oracle"),
    ("cover f=0.21", "DCoverRandomEpFifty", "cover"),
    ("envelope", "DEnvelopeRandomEpFifty", "envelope"),
    ("anatomy-v2", "DAnatomyTwoRandomEpFifty", "anatomy"),
]


def macros(path):
    txt = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\s*$",
                         txt, re.M):
        out[m.group(1)] = m.group(2)
    return out


def num(s):
    return float(s.replace("+", "").replace("$", "").strip())


def ci(s):
    lo, hi = re.findall(r"[-+]?\d*\.\d+", s.replace("\\,", " "))
    return float(lo), float(hi)


def main():
    import json
    mac = macros(AUTO)
    geo = json.load(open(GEOM, encoding="utf-8"))

    rows = []
    for label, stem, key in ARMS:
        hidden = geo[key]["hidden_share_of_all_anat"]
        if stem is None:
            rows.append((label, hidden, 0.0, None, None))
        else:
            lo, hi = ci(mac[stem + "CI"])
            rows.append((label, hidden, num(mac[stem]), lo, hi))
    rows.sort(key=lambda r: r[1])           # increasing anatomical specificity

    fig, ax = plt.subplots(figsize=(6.4, 3.05))
    ys = range(len(rows))
    ax.axvline(0, color="black", linewidth=0.9, zorder=1)

    for y, (label, hidden, d, lo, hi) in zip(ys, rows):
        if lo is None:                       # the null itself, no difference
            ax.plot([0], [y], marker="|", markersize=13, color="black",
                    zorder=3)
            continue
        excludes = lo > 0 or hi < 0
        colour = "#0072B2" if excludes else "#888888"
        ax.plot([lo, hi], [y, y], color=colour, linewidth=1.9,
                solid_capstyle="butt", zorder=2)
        for x in (lo, hi):                   # explicit interval caps
            ax.plot([x, x], [y - 0.14, y + 0.14], color=colour, linewidth=1.4,
                    zorder=2)
        ax.plot([d], [y], marker="o" if excludes else "o", markersize=6.4,
                color=colour, markerfacecolor=colour if excludes else "white",
                markeredgewidth=1.5, zorder=4)
        ax.text(hi + 0.0007, y, f"{d:+.4f}", va="center", fontsize=8.1,
                color="#222222")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r[0]}\n{r[1]:.1f}% anatomy hidden" for r in rows],
                       fontsize=8.4)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("paired difference in test AUC against the unguided null, "
                  "epoch 50", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.35, alpha=0.35)
    ax.margins(x=0.16)

    filled = plt.Line2D([], [], marker="o", color="#0072B2", linestyle="-",
                        markersize=6.4, label="interval excludes zero")
    hollow = plt.Line2D([], [], marker="o", color="#888888", linestyle="-",
                        markersize=6.4, markerfacecolor="white",
                        markeredgewidth=1.5, label="interval contains zero")
    ax.legend(handles=[filled, hollow], fontsize=8, frameon=False,
              loc="lower right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext, kw in (("png", {"dpi": 320}), ("pdf", {})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)

    print("regenerated", OUT)
    for label, hidden, d, lo, hi in rows:
        if lo is None:
            print(f"  {label:16s} {hidden:5.1f}% hidden   (null)")
        else:
            flag = "excludes 0" if (lo > 0 or hi < 0) else "contains 0"
            print(f"  {label:16s} {hidden:5.1f}% hidden   {d:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  {flag}")


if __name__ == "__main__":
    main()
