"""Map mask composition -> downstream AUC across every measured arm.

The question this answers: does *how much anatomy the mask hides*, *how much
anatomy survives in the context*, and *how much of the mask lands on anatomy
rather than background* predict downstream AUC?  If so, there is an optimal
ratio and the floor knob can be tuned to hit it.

Composition numbers come from measurements already in the repo, all produced by
one identical sampling pass:

  reports/arm_stats_sweep/cover_floor_sweep.json   n=6137 slices
      random, oracle, envelope + COVER floors 0.15..0.30
  reports/arm_stats/arm_stats.json                 n=1534 slices
      blob (mirage_anatomy) only -- not present in the sweep

AUCs are frozen MeanPool+Linear, probe seed 42, identical harness.

IMPORTANT: this is an observational fit over a handful of arms that differ in
more than one variable at a time.  It is hypothesis-generating.  The confirmatory
design is the within-arm floor sweep (one knob, everything else identical),
which is what `--emit-plan` prints.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional

import numpy as np

SWEEP = pathlib.Path(r"D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json")
ARMS = pathlib.Path(r"D:\jepa_phase0\reports\arm_stats\arm_stats.json")
OUT = pathlib.Path(r"D:\jepa_phase0\reports\composition_vs_auc")

# Frozen MeanPool+Linear, seed 42.  None = not measured.
# OLD COVER is deliberately absent: enc_truncate=window + amp_target=true make
# it unattributable (see docs/experiments/masking/crop_and_precision_audit.md).
AUC: Dict[str, Dict[str, Optional[float]]] = {
    "random":   {"ep30": None,   "ep50": 0.8641, "ep75": 0.8723, "ep100": 0.8746},
    "oracle":   {"ep30": None,   "ep50": 0.8740, "ep75": 0.8836, "ep100": 0.8855},
    "envelope": {"ep30": 0.8540, "ep50": 0.8761, "ep75": 0.8803, "ep100": 0.8807},
    "blob":     {"ep30": 0.8583, "ep50": 0.8654, "ep75": None,   "ep100": None},
    "cover_f021": {"ep30": 0.8522, "ep50": None, "ep75": None,   "ep100": None},
}

# Metrics to test, with the direction the project intuitively expected.
METRICS = [
    ("pct_anat_hid",  "% of anatomy hidden by targets"),
    ("ctx_anat",      "anatomy cells surviving in context"),
    ("pct_ctx_anat",  "% of context that is anatomy"),
    ("pct_tgt_anat",  "% of masked cells on anatomy (mask purity)"),
    ("zero_pct",      "% slices with zero anatomy in context"),
    ("ctx",           "context patches delivered to encoder"),
]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without a scipy dependency."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def load_composition() -> Dict[str, dict]:
    sweep = json.loads(SWEEP.read_text())
    arms = json.loads(ARMS.read_text())

    comp: Dict[str, dict] = {}
    for name in ("random", "oracle", "envelope"):
        comp[name] = dict(sweep[name], _n=sweep[name].get("n"), _src="sweep")

    # COVER at the floor actually being trained.
    comp["cover_f021"] = dict(sweep["0.21"], _n=sweep["0.21"].get("n"), _src="sweep")

    # blob only exists in the 1534-slice pass; zero_pct is named `zero` there.
    for key in arms:
        if key.startswith("blob"):
            b = dict(arms[key])
            b["zero_pct"] = b.pop("zero", None)
            comp["blob"] = dict(b, _n=b.get("n"), _src="arm_stats(n=1534)")
            break
    return comp


def floor_curve() -> Dict[str, dict]:
    """Composition of every COVER floor -- the confirmatory design's x-axis."""
    sweep = json.loads(SWEEP.read_text())
    out = {}
    for k, v in sweep.items():
        try:
            f = float(k)
        except ValueError:
            continue
        out[f"{f:.2f}"] = v
    return dict(sorted(out.items(), key=lambda kv: float(kv[0])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="ep50",
                    choices=["ep30", "ep50", "ep75", "ep100"],
                    help="which milestone's AUC to correlate against")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--emit-plan", action="store_true",
                    help="print the confirmatory floor-sweep design")
    args = ap.parse_args()

    comp = load_composition()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm, c in comp.items():
        a = AUC.get(arm, {}).get(args.epoch)
        rows.append({
            "arm": arm,
            "auc": a,
            "src": c.get("_src"),
            **{m: c.get(m) for m, _ in METRICS},
        })

    measured = [r for r in rows if r["auc"] is not None]

    print(f"\n=== composition vs AUC @ {args.epoch} "
          f"({len(measured)} arms with a measured AUC) ===\n")
    hdr = f"{'arm':<12} {'AUC':>7} " + " ".join(f"{m:>14}" for m, _ in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["auc"] is None, -(r["auc"] or 0))):
        auc = f"{r['auc']:.4f}" if r["auc"] is not None else "   --"
        vals = " ".join(
            f"{r[m]:14.2f}" if r[m] is not None else f"{'--':>14}"
            for m, _ in METRICS)
        print(f"{r['arm']:<12} {auc:>7} {vals}")

    print(f"\n--- rank correlation with AUC @ {args.epoch} (n={len(measured)}) ---")
    corrs = {}
    for m, label in METRICS:
        xs = np.array([r[m] for r in measured if r[m] is not None], float)
        ys = np.array([r["auc"] for r in measured if r[m] is not None], float)
        rho = spearman(xs, ys)
        corrs[m] = rho
        print(f"  {label:<42} rho = {rho:+.3f}")

    print("\n  NOTE: n is tiny and arms differ in several variables at once.")
    print("  Treat sign and shape as hypotheses, not estimates.")

    result = {
        "epoch": args.epoch,
        "rows": rows,
        "spearman": corrs,
        "floor_curve": floor_curve(),
    }
    p = OUT / f"composition_vs_auc_{args.epoch}.json"
    p.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {p}")

    if args.emit_plan:
        emit_plan()
    if args.plot:
        make_plot(rows, args.epoch)


def emit_plan() -> None:
    fc = floor_curve()
    print("\n=== confirmatory design: within-arm floor sweep ===")
    print("One knob (cover floor), everything else byte-identical.  Each floor")
    print("is a different point on the composition axes below.\n")
    hdr = f"{'floor':>6} " + " ".join(f"{m:>14}" for m, _ in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for f, v in fc.items():
        vals = " ".join(f"{v.get(m, float('nan')):14.2f}" for m, _ in METRICS)
        print(f"{f:>6} {vals}")


def make_plot(rows, epoch: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured = [r for r in rows if r["auc"] is not None]
    fc = floor_curve()

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
    fig.suptitle(
        f"Mask composition vs downstream AUC @ {epoch}  "
        f"(frozen MeanPool+Linear, seed 42)",
        fontsize=14, fontweight="bold")

    colours = {
        "random": "#888888", "oracle": "#2ca02c", "envelope": "#1f77b4",
        "blob": "#d62728", "cover_f021": "#ff7f0e",
    }

    for ax, (m, label) in zip(axes.ravel(), METRICS):
        # COVER floor family as a light trajectory: composition is known for
        # every floor, AUC is not -- this is the axis the sweep would fill in.
        fx = [v.get(m) for v in fc.values() if v.get(m) is not None]
        if fx:
            ax.axvspan(min(fx), max(fx), color="#ffe9d6", zorder=0,
                       label="COVER floor range (AUC unmeasured)")

        for r in measured:
            if r[m] is None:
                continue
            c = colours.get(r["arm"], "#333333")
            ax.scatter(r[m], r["auc"], s=140, color=c, zorder=3,
                       edgecolor="black", linewidth=0.8)
            ax.annotate(r["arm"], (r[m], r["auc"]), fontsize=9,
                        xytext=(6, 5), textcoords="offset points")

        xs = np.array([r[m] for r in measured if r[m] is not None], float)
        ys = np.array([r["auc"] for r in measured if r[m] is not None], float)
        if len(xs) >= 3:
            o = np.argsort(xs)
            ax.plot(xs[o], ys[o], color="#999999", lw=1.0, ls="--", zorder=2)
            ax.set_title(f"{label}\nSpearman rho = {spearman(xs, ys):+.2f}",
                         fontsize=10)
        else:
            ax.set_title(label, fontsize=10)

        ax.set_xlabel(m)
        ax.set_ylabel(f"AUC @ {epoch}")
        ax.grid(alpha=0.3)

    axes.ravel()[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = OUT / f"composition_vs_auc_{epoch}.png"
    fig.savefig(p, dpi=130)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
