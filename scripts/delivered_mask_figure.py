"""Render audited token masks, never raw OCT pixels, with cohort-level metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.delivered_mask_audit import BASELINE, FixedCropDataset
from src.datasets.oct_slices_guided import GuidedOCTSliceDataset
from src.transforms import make_paired_transforms

EVIDENCE = ROOT / "autopilot" / "investigations" / "delivered_task" / "evidence"
CONTEXT, TARGET, UNUSED = "#56B4E9", "#E69F00", "#F2F2F2"
EXAMPLE_ORDINAL = 94  # Previously documented failure, not a new case search.


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def cohort_metrics(legacy, corrected):
    output = []
    for name, rows in [("Legacy", legacy), ("V2 + guard", corrected)]:
        full = [r for r in rows if r["batch_size"] == 64]
        valid = [r for r in full if r["guide_valid"] and r.get("policy_info")]
        for stage, key in [("Scored", "scored_hidden_mass_fraction"),
                           ("Delivered", "delivered_hidden_mass_fraction")]:
            output.append(dict(policy=name, metric="hidden_guide_mass_percent",
                               stage=stage, n=len(valid),
                               mean=100 * float(np.mean([r[key] for r in valid]))))
        for stage, key in [("Before", "context_tissue_before_collation"),
                           ("Final", "context_tissue")]:
            output.append(dict(policy=name, metric="context_tissue_cells",
                               stage=stage, n=len(full),
                               mean=float(np.mean([r[key] for r in full]))))
    return output


def load_fixed_guide():
    meta = json.loads(BASELINE.read_text())["_meta"]
    cfg = yaml.safe_load((ROOT / "configs" / "patch_cover_f021_ep25.yaml").read_text())["data"]
    ds = GuidedOCTSliceDataset(
        data_dir=str(Path(cfg["data_dir"]) / "Training"),
        guide_dir=str(Path(meta["guide_dir"]) / "Training"),
        num_slices=100, slice_size=256, patch_size=16, dilate_patches=0,
        occupancy_threshold=.25, transform=make_paired_transforms(),
        slice_cache=str(Path(cfg["slice_cache_dir"]) / "Training"))
    volumes = sorted(random.Random(42).sample(range(len(ds.file_paths)), 24))
    indices = [v * 100 + s for v in volumes for s in range(0, 100, 4)]
    image, guide, valid, _ = FixedCropDataset(ds, indices)[EXAMPLE_ORDINAL]
    return image, guide, valid


def mask_panel(ax, tissue, targets, context, title):
    target = {i for group in targets for i in group}
    visible = set(context)
    if target & visible:
        raise ValueError("Figure input has context-target overlap")
    if not target or not visible or min(target | visible) < 0 or max(target | visible) >= 256:
        raise ValueError("Invalid figure mask indices")
    for i in range(256):
        y, x = divmod(i, 16)
        color = TARGET if i in target else CONTEXT if i in visible else UNUSED
        ax.add_patch(Rectangle(
            (x, y), 1, 1, facecolor=color, edgecolor="#444444",
            linewidth=.12, hatch="///" if i in visible else None))
    ys, xs = np.nonzero(tissue)
    ax.scatter(xs + .5, ys + .5, s=8, facecolors="none",
               edgecolors="black", linewidths=.55, zorder=3)
    ax.set(xlim=(0, 16), ylim=(16, 0), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, loc="left", pad=5)


def aggregate_panel(ax, metrics, metric, title, color, limit):
    rows = [r for r in metrics if r["metric"] == metric]
    labels = [("Legacy" if r["policy"] == "Legacy" else "V2 + guard")
              + "\n" + r["stage"].lower() for r in rows]
    positions = np.arange(len(rows))[::-1]
    bars = ax.barh(positions, [r["mean"] for r in rows],
                   height=.62, color=color, edgecolor="black", linewidth=.4)
    for index, (bar, row) in enumerate(zip(bars, rows)):
        if index % 2 == 0:
            bar.set_hatch("//")
        ax.text(row["mean"] + limit * .025, bar.get_y() + bar.get_height() / 2,
                f"{row['mean']:.1f}", va="center", fontsize=7)
    ax.set_yticks(positions, labels, fontsize=7)
    ax.set_xlim(0, limit)
    ax.set_ylim(-.7, len(rows) - .3)
    ax.set_title(title, fontsize=8, loc="left", pad=8)
    ax.set_xlabel("Percent of supported guide mass" if metric.endswith("percent")
                  else "Guide-positive cells in encoder context", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(tissue, legacy, corrected, metrics):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "hatch.linewidth": .3}):
        fig = plt.figure(figsize=(7.05, 5.75))
        grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.12],
                               left=.035, right=.965, bottom=.19, top=.86,
                               wspace=.43, hspace=.58)
        mask_panel(fig.add_subplot(grid[0, 0]), tissue, legacy["intended_targets"],
                   legacy["context_before_collation"][0], "(a) Legacy: before collation")
        mask_panel(fig.add_subplot(grid[0, 1]), tissue, legacy["targets"],
                   legacy["context"][0], "(b) Legacy: exact final indices")
        mask_panel(fig.add_subplot(grid[1, 0]), tissue, corrected["intended_targets"],
                   corrected["context_before_collation"][0], "(c) V2: scored / sampled indices")
        mask_panel(fig.add_subplot(grid[1, 1]), tissue, corrected["targets"],
                   corrected["context"][0], "(d) V2 + guard: exact final indices")
        aggregate_panel(
            fig.add_subplot(grid[0, 2]), metrics, "hidden_guide_mass_percent",
            "(e) Mean hidden guide mass\n573 valid views, full-size batches", TARGET, 100)
        aggregate_panel(
            fig.add_subplot(grid[1, 2]), metrics, "context_tissue_cells",
            "(f) Mean visible tissue\n576 views, full-size batches", CONTEXT, 20)
        fig.suptitle("COVER: intended masks versus the delivered task", fontsize=11, y=.975)
        handles = [
            Patch(facecolor=TARGET, edgecolor="black", label="Target union"),
            Patch(facecolor=CONTEXT, edgecolor="black", hatch="///", label="Encoder context"),
            Line2D([], [], marker="o", markersize=4, markerfacecolor="none",
                   markeredgecolor="black", linestyle="none", label="Guide-positive tissue"),
            Patch(facecolor=UNUSED, edgecolor="black", label="Neither"),
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .08),
                   ncol=4, fontsize=7, frameon=False, handlelength=1.4)
        fig.text(.5, .044,
                 "Maps: one fixed Training view, 16 × 16 token grid; no raw OCT pixels. Charts: cohort means.",
                 ha="center", fontsize=7)
        fig.text(.5, .019,
                 "V2 scores delivered prefixes; its final context guard is a separate guide-aware intervention.",
                 ha="center", fontsize=7)
        return fig


def make_map_figure(tissue, legacy, corrected):
    """Keep the manuscript illustration separate from aggregate statistics."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "hatch.linewidth": .3}):
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.6))
        panels = [
            (legacy["intended_targets"], legacy["context_before_collation"][0],
             "(a) Historical COVER\nBefore collation"),
            (legacy["targets"], legacy["context"][0],
             "(b) Historical COVER\nDelivered indices"),
            (corrected["intended_targets"], corrected["context_before_collation"][0],
             "(c) Exact-prefix correction\nBefore final context selection"),
            (corrected["targets"], corrected["context"][0],
             "(d) Correction with context guard\nDelivered indices"),
        ]
        for ax, (targets, context, title) in zip(axes.flat, panels):
            mask_panel(ax, tissue, targets, context, title)
            ax.title.set_fontsize(9)
        fig.subplots_adjust(left=.035, right=.965, top=.92, bottom=.14,
                            wspace=.18, hspace=.34)
        fig.legend(handles=[
            Patch(facecolor=TARGET, edgecolor="black", label="Target union"),
            Patch(facecolor=CONTEXT, edgecolor="black", hatch="///", label="Encoder context"),
            Line2D([], [], marker="o", markersize=4, markerfacecolor="none",
                   markeredgecolor="black", linestyle="none", label="Guide-positive tissue"),
            Patch(facecolor=UNUSED, edgecolor="black", label="Neither"),
        ], loc="lower center", bbox_to_anchor=(.5, .025), ncol=2,
           fontsize=8, frameon=False)
        return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=EVIDENCE / "mask_replay600_v2")
    parser.add_argument("--out", type=Path, default=EVIDENCE / "mask_figure_v1")
    parser.add_argument("--maps-only", action="store_true",
                        help="Export a manuscript-sized illustration without quantitative charts")
    args = parser.parse_args()
    paths = {name: args.evidence / f"{name}_bs64.jsonl"
             for name in ("cover_legacy", "cover_v2_guard")}
    all_rows = {name: read_rows(path) for name, path in paths.items()}
    legacy = next(r for r in all_rows["cover_legacy"] if r["ordinal"] == EXAMPLE_ORDINAL)
    corrected = next(r for r in all_rows["cover_v2_guard"] if r["ordinal"] == EXAMPLE_ORDINAL)
    image, guide, valid = load_fixed_guide()
    for row in (legacy, corrected):
        assert bool(valid) == row["guide_valid"]
        assert hashlib.sha256(image.numpy().tobytes()).hexdigest() == row["crop_tensor_sha256"]
        assert hashlib.sha256(guide.numpy().tobytes()).hexdigest() == row["guide_sha256"]
    tissue = guide[0].numpy() >= .25
    assert legacy["context_tissue"] == 0 and corrected["context_floor"]["status"] == "satisfied"
    metrics = cohort_metrics(all_rows["cover_legacy"], all_rows["cover_v2_guard"])
    args.out.mkdir(parents=True, exist_ok=True)
    fig = (make_map_figure(tissue, legacy, corrected) if args.maps_only
           else make_figure(tissue, legacy, corrected, metrics))
    stem = "delivered_token_maps" if args.maps_only else "delivered_masks"
    png, pdf = args.out / (stem + ".png"), args.out / (stem + ".pdf")
    with plt.rc_context({"pdf.fonttype": 42, "hatch.linewidth": .3}):
        fig.savefig(png, dpi=300, facecolor="white", transparent=False)
        fig.savefig(pdf, facecolor="white", transparent=False,
                    metadata={"Title": "COVER intended versus delivered token masks",
                              "Author": "", "Creator": "Matplotlib"})
        if args.maps_only:
            fig.savefig(args.out / (stem + ".svg"), facecolor="white",
                        metadata={"Creator": "Matplotlib"},
                        transparent=False)
    plt.close(fig)
    with Image.open(png) as rendered:
        assert rendered.getchannel("A").getextrema() == (255, 255)
        rendered.convert("RGB").save(png, dpi=(300, 300))
    with (args.out / "aggregate_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy", "metric", "stage", "n", "mean"])
        writer.writeheader()
        writer.writerows(metrics)
    caption = (
        "COVER intended versus delivered masks. Token maps use one fixed, previously audited "
        "Training view; orange shows target unions, hatched blue the encoder-visible indices, "
        "and hollow circles occupancy-guide tissue. No raw OCT pixels or subject identifiers "
        "are displayed. Historical full-rectangle scoring precedes prefix collation, which "
        "can discard targets and all guide-positive context. Opt-in v2 scores exact delivered "
        "prefix targets; its separately enabled guide-aware context guard may reach outside "
        "the sampled context rectangle. Policies use the same cropped input and drawn sizes, "
        "not exactly paired placements. All numbers are cohort means: hidden soft mass over "
        "573 valid views and context tissue over 576 views in nine complete batches of 64. "
        "The 24-view short tail is excluded; three invalid guides remain in the context "
        "denominator and are not certified. Views are clustered within the fixed Training "
        "scope, not independent training replications. Bars start at zero and show descriptive "
        "means without inferential intervals. Target maps show unions, not duplicate loss "
        "weights. No downstream AUC result is implied."
    )
    if args.maps_only:
        caption = (
            "Token masks from one fixed Training view used in the engineering replay. "
            "Orange cells show target unions, hatched blue cells encoder context, and "
            "hollow circles guide-positive tissue; raw OCT pixels are not displayed. "
            "Historical collation removes tissue targets and all tissue from the "
            "context in this illustrative failure case. Exact-prefix scoring and a "
            "separate guide-aware context guard change the delivered task. The guard "
            "may select tissue outside the original context rectangle. Inputs and "
            "drawn sizes are shared, but placements are not exactly paired between "
            "historical and corrected policies. This is not a random example or "
            "evidence of improved downstream AUC; aggregate counts are reported separately."
        )
    (args.out / "caption.txt").write_text(caption, encoding="utf-8")
    alt_text = (
        "Four token-grid maps compare historical COVER before and after collation with the "
        "opt-in correction. Historical final context contains no tissue-marked cells in the "
        "illustrative view; corrected guarded context retains them. Raw OCT pixels are absent."
    )
    if not args.maps_only:
        alt_text += (
            " Two zero-based aggregate bar charts show the historical gap between scored "
            "and delivered coverage and between pre-collation and final visible tissue."
        )
    (args.out / "alt_text.txt").write_text(alt_text, encoding="utf-8")
    manifest = {
        "source_script": r"scripts\delivered_mask_figure.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_evidence_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for name, path in paths.items()},
        "raw_oct_pixels_rendered": False, "subject_identifiers_rendered": False,
        "numeric_evidence": ("illustrative mask sets only; no aggregate chart" if args.maps_only
                             else "aggregated only; descriptive fixed-scope means"),
        "guide_definition": "occupancy channel 0 >= 0.25",
        "width_inches": float(fig.get_figwidth()),
        "height_inches": float(fig.get_figheight()), "dpi": 300,
        "publisher_compliance": "not asserted; parent verifies final manuscript placement",
        "matplotlib_version": matplotlib.__version__, "torch_version": torch.__version__,
        "metrics": metrics,
        "rendered_metrics": [] if args.maps_only else metrics,
        "outputs_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ([png, pdf, args.out / (stem + ".svg")] if args.maps_only else [png, pdf])
        },
    }
    (args.out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote deidentified token-grid figure, aggregate metrics, caption, alt text and manifest.")
    print(png)


if __name__ == "__main__":
    main()
