"""Regenerate the masking-narrative figures from measured artifacts only.

Story arc:
  S1  anatomy tokens carry disproportionate predictive value  -> motivates guiding
  S2  pushing purity to the extreme collapses downstream AUC  -> inverted U
  S3  the collapse mechanism (token-value inversion, error blow-up, skill loss)
  S4  background/context carries substantial independent signal
  S5  the coverage floor as the balance knob (dose-response)
  S6  full masking-statistics panel across arms

No training, probing, or mask sampling happens here.  Every value is read from
an existing artifact under D:\\jepa_phase0\\reports.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
FIGURES = PAPER / "figures"
REPORTS = Path(r"D:\jepa_phase0\reports")

COMPOSITION = REPORTS / "composition_vs_auc" / "composition_vs_auc_ep50.json"
TOKEN_VALUE = REPORTS / "background_signal" / "marginal_token_value.csv"
SKILL = REPORTS / "background_signal" / "skill_scores.json"
REGION_AUC = REPORTS / "downstream_region_auc" / "region_auc_summary.json"
ATTRIB = REPORTS / "patch_attribution" / "attribution_summary.csv"

COLORS = {
    # Mirrors autopilot/p8_make_assets.py COL exactly, so one arm carries one
    # colour across the whole manuscript. Audited with the bundled skill tool
    # .agents/skills/scientific-visualization/scripts/palette_audit.py
    # (background FFFFFF, role graphical); every entry clears 3:1 on white.
    "random": "#000000",
    "oracle": "#882255",
    "envelope": "#0072B2",
    "anatomy": "#CC79A7",
    "blob": "#666666",
    "cover": "#009E73",
    "cover_f021": "#009E73",
    "ancestor": "#333333",
}
LABEL = {
    "random": "random",
    # Display names must match the paper, which resolves them through \ArmBest.
    # The stored artifacts keep their historical keys (oracle, blob) so the JSON
    # is not rewritten; only the rendered label changes.
    "oracle": "centroid",
    "envelope": "envelope",
    "cover_f021": "COVER $f{=}.21$",
    "blob": "anatomy-v2",
}
ANAT_C = "#c53030"
BG_C = "#4a5568"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


# ----------------------------------------------------------------- S1 + S4
def fig_signal():
    """Anatomy tokens are worth more per token, but background is not inert."""
    tv = load_csv(TOKEN_VALUE)
    reg = load_json(REGION_AUC)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

    # (a) marginal token value, healthy arms only
    healthy = [r for r in tv if r["fam"] != "anatomy"]
    x = np.arange(len(healthy))
    va = [float(r["v_anat"]) * 1e3 for r in healthy]
    vb = [float(r["v_bg"]) * 1e3 for r in healthy]
    ax = axes[0]
    ax.bar(x - 0.2, va, 0.4, color=ANAT_C, label="anatomy token")
    ax.bar(x + 0.2, vb, 0.4, color=BG_C, label="background token")
    ax.set_xticks(x)
    abbr = {"fork": "fork", "random": "rand", "oracle": "cent", "envelope": "envl",
            "blob": "an-v2", "cover": "covr", "cover_f021": "covr"}
    ax.set_xticklabels(
        [f"{abbr.get(r['tag'].split('_ep')[0], r['tag'].split('_ep')[0])}·{r['ep']}"
         for r in healthy], rotation=45, ha="right", fontsize=5.8)
    ax.set_ylabel(r"marginal token value ($\times10^{-3}$)")
    ratios = [float(r["ratio"]) for r in healthy]
    ax.set_title(f"(a) Anatomy tokens are worth\n"
                 f"{min(ratios):.1f}--{max(ratios):.1f}$\\times$ more than background")
    ax.set_ylim(0, 2.6)
    ax.legend(loc="upper left", frameon=False, fontsize=6.2)

    # (b) regional probe AUC: background alone is nearly as good
    order = ["random_ep50", "oracle_ep50", "envelope_ep50", "blob_ep50"]
    rows = {r["tag"]: r for r in reg}
    x = np.arange(len(order))
    ax = axes[1]
    ax.bar(x - 0.22, [rows[t]["anatomy"] for t in order], 0.44, color=ANAT_C,
           label="anatomy patches only")
    ax.bar(x + 0.22, [rows[t]["background"] for t in order], 0.44, color=BG_C,
           label="background patches only")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(t.replace("_ep50", ""), t.replace("_ep50", ""))
                        for t in order], fontsize=7)
    ax.set_ylim(0.80, 0.90)
    ax.set_ylabel("glaucoma AUC")
    ax.set_title("(b) Background patches alone\nstill reach AUC 0.85--0.87")
    ax.legend(loc="upper left", frameon=False, fontsize=6.2)
    for i, t in enumerate(order):
        ax.text(i, 0.807, f"$\\Delta${abs(rows[t]['bg_minus_anat']):.3f}",
                ha="center", fontsize=6, color="0.25")

    # (c) total attribution mass
    at = load_csv(ATTRIB)
    tags = [r for r in at if r.get("arm") or r.get("tag")]
    key = "arm" if "arm" in at[0] else "tag"
    ratio_key = None
    for cand in ("bg_over_anat_TOTAL", "background_over_anatomy_total_abs",
                 "total_abs_ratio"):
        if cand in at[0]:
            ratio_key = cand
            break
    ax = axes[2]
    if ratio_key:
        names = [r[key].replace("_ep50", "") for r in tags]
        vals = [float(r[ratio_key]) for r in tags]
    else:  # derive from totals
        names, vals = [], []
        for r in tags:
            names.append(r[key].replace("_ep50", ""))
            vals.append(float(r["total_abs_background"]) / float(r["total_abs_anatomy"]))
    cols = [COLORS.get(n, "#6b7280") for n in names]
    ax.bar(np.arange(len(names)), vals, 0.6, color=cols)
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([LABEL.get(n, n) for n in names], fontsize=7)
    ax.set_ylabel("background / anatomy\ntotal |attribution|")
    ax.set_title("(c) Background carries most of the\ntotal attribution mass")

    fig.tight_layout()
    save(fig, "figS1_background_matters")


# ----------------------------------------------------------------- S2
def fig_inverted_u():
    """Downstream AUC is non-monotonic in how much anatomy is hidden."""
    comp = load_json(COMPOSITION)
    rows = comp["rows"]
    scored = [r for r in rows if r["auc"] is not None]
    pending = [r for r in rows if r["auc"] is None]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, xk, xlabel, title in (
        (axes[0], "pct_anat_hid", "% of anatomy covered by targets",
         "(a) AUC vs. anatomy put in targets"),
        (axes[1], "pct_tgt_anat", "% of target cells that are anatomy",
         "(b) AUC vs. target purity"),
    ):
        pts = sorted(scored, key=lambda r: r[xk])
        ax.plot([r[xk] for r in pts], [r["auc"] for r in pts],
                "-", color="0.6", lw=1.0, zorder=1)
        for r in pts:
            ax.scatter(r[xk], r["auc"], s=44, zorder=3,
                       color=COLORS.get(r["arm"], "#333"),
                       edgecolor="white", linewidth=0.7)
            ax.annotate(LABEL.get(r["arm"], r["arm"]), (r[xk], r["auc"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=6.8,
                        color=COLORS.get(r["arm"], "#333"))
        for r in pending:
            ax.axvline(r[xk], color=COLORS["cover"], lw=1.0, ls="--", alpha=0.8)
            ax.annotate(LABEL.get(r["arm"], r["arm"]) + " (pending)",
                        (r[xk], 0.8616), fontsize=6.2, ha="center",
                        color=COLORS["cover"],
                        bbox=dict(fc="white", ec="none", pad=0.8))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("glaucoma AUC (mean-pool, ep50)")
        ax.set_title(title)
        ax.set_ylim(0.860, 0.879)
        ax.margins(x=0.14)
    fig.suptitle("More anatomy hiding helps -- until it does not "
                 "(5 arms, 1 seed each, observational)", fontsize=8, y=1.04)
    fig.tight_layout()
    save(fig, "figS2_inverted_u")


# ----------------------------------------------------------------- S3
def fig_collapse():
    """What breaks in the near-pure-anatomy arm."""
    tv = load_csv(TOKEN_VALUE)
    skill = load_json(SKILL)

    fams = {}
    for r in tv:
        fams.setdefault(r["fam"], []).append(r)
    for v in fams.values():
        v.sort(key=lambda r: int(r["ep"]))

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

    ax = axes[0]
    for fam, rs in fams.items():
        c = COLORS["blob"] if fam == "anatomy" else COLORS.get(fam, "#666")
        ax.plot([int(r["ep"]) for r in rs], [float(r["ratio"]) for r in rs],
                "o-", ms=3.4, lw=1.3, color=c,
                label="anatomy-v2 (near-pure)" if fam == "anatomy" else LABEL.get(fam, fam))
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("pretraining epoch")
    ax.set_ylabel("anatomy / background\ntoken value")
    ax.set_title("(a) Token-value ratio inverts")
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")

    ax = axes[1]
    for fam, rs in fams.items():
        c = COLORS["blob"] if fam == "anatomy" else COLORS.get(fam, "#666")
        ax.plot([int(r["ep"]) for r in rs], [float(r["err_full"]) for r in rs],
                "o-", ms=3.4, lw=1.3, color=c)
    ax.set_xlabel("pretraining epoch")
    ax.set_ylabel("full-context prediction error")
    ax.set_title("(b) Prediction error blows up")

    ax = axes[2]
    tags = [s["tag"] for s in skill]
    x = np.arange(len(tags))
    ax.bar(x - 0.2, [s["anat"]["skill_vs_pos"] for s in skill], 0.4,
           color=ANAT_C, label="anatomy")
    ax.bar(x + 0.2, [s["bg"]["skill_vs_pos"] for s in skill], 0.4,
           color=BG_C, label="background")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(t.split("_ep")[0], t.split("_ep")[0]) + "\n"
                        + t.split("_ep")[-1] for t in tags], fontsize=5.6)
    ax.set_ylabel("skill vs. position-only baseline")
    ax.set_title("(c) Predictor skill collapses")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    ax.annotate("anatomy-v2", xy=(4, 0.22), xytext=(2.7, 0.42), fontsize=6.5,
                color=COLORS["blob"],
                arrowprops=dict(arrowstyle="->", color=COLORS["blob"], lw=0.8))

    fig.tight_layout()
    save(fig, "figS3_collapse_mechanism")


# ----------------------------------------------------------------- S5
def fig_floor():
    """The coverage floor is the balance knob."""
    comp = load_json(COMPOSITION)
    curve = comp["floor_curve"]
    fs = sorted(float(k) for k in curve)
    get = lambda k, f: curve[f"{f:.2f}"][k]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

    ax = axes[0]
    ax.plot(fs, [get("pct_anat_hid", f) for f in fs], "o-", ms=3.2, lw=1.3,
            color=ANAT_C, label="% of anatomy in targets")
    ax.plot(fs, [get("pct_anat_vis", f) for f in fs], "s-", ms=3.2, lw=1.3,
            color=COLORS["envelope"], label="% of anatomy reaching context")
    ax.set_xlabel("coverage floor $f$")
    ax.set_ylabel("percent of slice anatomy")
    ax.set_title("(a) The floor trades hiding\nagainst visible anatomy")
    ax.set_ylim(0, 94)
    ax.legend(frameon=False, fontsize=6.2, loc="center left")
    ax.axvline(0.21, color=COLORS["cover"], lw=1.0, ls="--")
    ax.text(0.21, 86, "$f{=}.21$", fontsize=6.4, color=COLORS["cover"], ha="center")

    ax = axes[1]
    ax.plot(fs, [get("zero_pct", f) for f in fs], "o-", ms=3.2, lw=1.3,
            color=COLORS["cover"])
    ax.set_xlabel("coverage floor $f$")
    ax.set_ylabel("% slices with zero anatomy in context")
    ax.set_title("(b) Raising the floor removes\nblank-context slices")
    ax.axvline(0.21, color=COLORS["cover"], lw=1.0, ls="--", alpha=0.6)

    ax = axes[2]
    fp = [f for f in fs if "paired" in curve[f"{f:.2f}"]]
    ax.bar(np.array(fp) - 0.002, [curve[f"{f:.2f}"]["paired"]["fixed"] for f in fp],
           0.004, color=COLORS["envelope"], label="slices fixed")
    ax.bar(np.array(fp) + 0.002, [curve[f"{f:.2f}"]["paired"]["broke"] for f in fp],
           0.004, color=ANAT_C, label="slices broken")
    ax.set_xlabel("coverage floor $f$")
    ax.set_ylabel("slices (paired vs. $f{=}.15$)")
    ax.set_title("(c) Every floor fixes more\nthan it breaks")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")

    fig.suptitle("COVER coverage-floor dose-response "
                 f"($n={curve['0.21']['n']}$ slices per floor)".replace("6137", "6{,}137"),
                 fontsize=8, y=1.04)
    fig.tight_layout()
    save(fig, "figS4_coverage_floor")


# ----------------------------------------------------------------- S6
def fig_mask_stats():
    """Full masking statistics across arms."""
    comp = load_json(COMPOSITION)
    rows = sorted(comp["rows"], key=lambda r: r["pct_anat_hid"])
    names = [LABEL.get(r["arm"], r["arm"]) for r in rows]
    cols = [COLORS.get(r["arm"], "#333") for r in rows]
    x = np.arange(len(rows))

    panels = [
        ("pct_anat_hid", "% of anatomy in targets", "(a) Anatomy covered by targets"),
        ("pct_tgt_anat", "% of target that is anatomy", "(b) Target purity"),
        ("pct_ctx_anat", "% of context that is anatomy", "(c) Context anatomy share"),
        ("ctx_anat", "anatomy cells in context", "(d) Anatomy cells kept"),
        ("ctx", "context tokens", "(e) Context budget"),
        ("zero_pct", "% slices, zero anatomy in context", "(f) Blank-context rate"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.2))
    for ax, (key, ylab, title) in zip(axes.ravel(), panels):
        ax.bar(x, [r[key] for r in rows], 0.62, color=cols)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=28, ha="right", fontsize=6.4)
        ax.set_ylabel(ylab, fontsize=7)
        ax.set_title(title)
        for i, r in enumerate(rows):
            ax.text(i, r[key], f"{r[key]:.1f}", ha="center", va="bottom", fontsize=5.8)
        ax.margins(y=0.18)
    fig.suptitle("Masking statistics by arm, ordered by anatomy placed in targets "
                 "(anatomy-v2 measured on $n{=}1{,}534$; others on the 6,137-slice sweep)",
                 fontsize=7.6, y=1.01)
    fig.tight_layout()
    save(fig, "figS5_mask_statistics")


if __name__ == "__main__":
    fig_signal()
    fig_inverted_u()
    fig_collapse()
    fig_floor()
    fig_mask_stats()
