"""Regenerate every paper figure from measured artifacts.

No training, probing, or mask sampling is performed.  The script only reads
existing JSON/PNG evidence and writes figures beside the paper.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PAPER = Path(__file__).resolve().parents[1]
FIGURES = PAPER / "figures"
REPORTS = Path(r"D:\jepa_phase0\reports")
RUNS = Path(r"D:\jepa_phase0\runs")

SWEEP = REPORTS / "arm_stats_sweep" / "cover_floor_sweep.json"
ARM_B64 = REPORTS / "arm_stats" / "arm_stats.json"
ARM_B1 = REPORTS / "arm_stats_b1" / "arm_stats.json"
COMPOSITION = REPORTS / "composition_vs_auc" / "composition_vs_auc_ep50.json"
QUALITATIVE = REPORTS / "arm_stats" / "zero_anatomy_floor20.png"

COLORS = {
    "random": "#6b7280",
    "oracle": "#2f855a",
    "envelope": "#2b6cb0",
    "anatomy": "#c53030",
    "blob": "#9b2c2c",
    "cover": "#dd6b20",
    "fork": "#111827",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_auc(run_name: str) -> float:
    return float(load_json(RUNS / run_name / "results.json")["test_auc"])


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def clean_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def find_arm(data: dict, prefix: str) -> dict:
    for key, value in data.items():
        if key.startswith(prefix):
            return value
    raise KeyError(prefix)


def figure_crop_defect() -> None:
    """F1: measured B=1 versus B=64 crop effect."""
    b1 = load_json(ARM_B1)
    b64 = load_json(ARM_B64)
    arms = [
        ("random", "random"),
        ("oracle", "oracle"),
        ("envelope", "envelope"),
        ("blob", "blob"),
        ("COVER .15", "COVER floor 0.15  prefix"),
    ]
    labels, losses, zero1, zero64 = [], [], [], []
    for label, prefix in arms:
        one = find_arm(b1, prefix)
        many = find_arm(b64, prefix)
        labels.append(label)
        losses.append(100.0 * (one["ctx"] - many["ctx"]) / one["ctx"])
        zero1.append(one["zero"])
        zero64.append(many["zero"])

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))

    colors = [
        COLORS["random"],
        COLORS["oracle"],
        COLORS["envelope"],
        COLORS["blob"],
        COLORS["cover"],
    ]
    axes[0].bar(x, losses, color=colors, edgecolor="white")
    axes[0].set_ylabel("Context removed at B=64 (%)")
    axes[0].set_title("(a) Batch-minimum truncation")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    for i, value in enumerate(losses):
        axes[0].text(i, value + 0.8, f"{value:.1f}", ha="center", fontsize=7)
    clean_axes(axes[0])

    width = 0.36
    axes[1].bar(
        x - width / 2,
        zero1,
        width,
        color="#d1d5db",
        edgecolor="#4b5563",
        label="B=1 (n=256)",
    )
    axes[1].bar(
        x + width / 2,
        zero64,
        width,
        color=colors,
        edgecolor="white",
        label="B=64 (n=1,534)",
    )
    axes[1].set_ylabel("Slices with zero anatomy\nin encoder context (%)")
    axes[1].set_title("(b) Anatomical blanking")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].legend(frameon=False, fontsize=7)
    clean_axes(axes[1])

    fig.suptitle(
        "Row-major prefix truncation removes context and can erase all anatomy",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "fig1_crop_defect")


def figure_context_excision() -> None:
    """F1b: crop the measured retained-versus-discarded encoder example."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    image = Image.open(QUALITATIVE)
    crop = image.crop((900, 1335, 1880, 1715))
    crop.save(FIGURES / "fig1b_context_excision.png")

    width, height = crop.size
    fig = plt.figure(figsize=(7.2, 7.2 * height / width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(crop)
    ax.axis("off")
    fig.savefig(
        FIGURES / "fig1b_context_excision.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)


def figure_composition_auc() -> None:
    """F2: observational composition-versus-AUC panels at epoch 50."""
    rows = {
        row["arm"]: row
        for row in load_json(COMPOSITION)["rows"]
        if row["auc"] is not None
    }
    order = ["random", "oracle", "envelope", "blob"]
    panels = [
        ("pct_tgt_anat", "Target patches on anatomy (%)"),
        ("pct_anat_hid", "Anatomy hidden by targets (%)"),
        ("ctx_anat", "Anatomy patches in context"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)

    for ax, (metric, xlabel) in zip(axes, panels):
        for arm in order:
            row = rows[arm]
            ax.scatter(
                row[metric],
                row["auc"],
                s=48,
                color=COLORS[arm],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            offset = {
                "random": (3, -10),
                "oracle": (3, 5),
                "envelope": (3, 5),
                "blob": (3, -10),
            }[arm]
            ax.annotate(
                arm,
                (row[metric], row["auc"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.5,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.8615, 0.8782)
        ax.set_title(f"({chr(97 + panels.index((metric, xlabel)))})")
        clean_axes(ax)
    axes[0].set_ylabel("Frozen mean-pool test AUC at epoch 50")
    fig.suptitle(
        "Mask composition and downstream AUC (observational; four arms)",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "fig2_composition_vs_auc")


def figure_cover_dose_response() -> None:
    """F3: measured COVER composition sweep and explicitly sparse AUC panel."""
    sweep = load_json(SWEEP)
    floors = sorted(float(key) for key in sweep if _is_float(key))
    data = [sweep[str(f)] for f in floors]

    panels = [
        ("pct_anat_hid", "Anatomy hidden (%)"),
        ("pct_anat_vis", "Anatomy reaching context (%)"),
        ("pct_tgt_anat", "Target purity (%)"),
        ("zero_pct", "Zero-anatomy slices (%)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55))
    axes = axes.ravel()

    for idx, (metric, ylabel) in enumerate(panels):
        ax = axes[idx]
        values = [row[metric] for row in data]
        ax.plot(floors, values, color=COLORS["cover"], marker="o", markersize=3)
        ax.axvline(0.21, color="#374151", linestyle=":", linewidth=0.9)
        ax.set_xlabel("COVER visibility floor")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97 + idx)})")
        clean_axes(ax)
        if metric == "zero_pct":
            envelope = sweep["envelope"]["zero_pct"]
            ax.axhline(
                envelope,
                color=COLORS["envelope"],
                linestyle="--",
                linewidth=1,
                label=f"envelope: {envelope:.2f}%",
            )
            ax.fill_between(
                floors,
                [value - row["se"] for value, row in zip(values, data)],
                [value + row["se"] for value, row in zip(values, data)],
                color=COLORS["cover"],
                alpha=0.16,
                linewidth=0,
                label="COVER ±1 SE",
            )
            ax.annotate(
                "f=.21 vs envelope:\nΔ=-0.23 pp, z=-0.6\npaired McNemar: same",
                xy=(0.21, sweep["0.21"]["zero_pct"]),
                xytext=(0.225, 10.2),
                arrowprops={"arrowstyle": "-", "color": "#4b5563", "lw": 0.7},
                fontsize=6.2,
            )
            ax.legend(frameon=False, fontsize=6.2, loc="lower left")

    ax = axes[4]
    clean_auc = [
        (27, result_auc("frozen_meanpool_cover_f021_ep27")),
        (30, result_auc("frozen_meanpool_cover_f021_ep30")),
        (34, result_auc("frozen_meanpool_cover_f021_ep34")),
    ]
    ax.scatter(
        [0.21] * len(clean_auc),
        [auc for _, auc in clean_auc],
        color=COLORS["cover"],
        s=30,
        zorder=3,
    )
    for epoch, auc in clean_auc:
        ax.annotate(
            f"ep{epoch}: {auc:.4f}",
            (0.21, auc),
            xytext=(6, -2),
            textcoords="offset points",
            fontsize=6.2,
        )
    ax.set_xlim(0.145, 0.305)
    ax.set_ylim(0.8465, 0.8590)
    ax.set_xlabel("COVER visibility floor")
    ax.set_ylabel("Frozen test AUC")
    ax.set_title("(e) AUC evidence is not a floor sweep")
    ax.text(
        0.15,
        0.8470,
        "No matched ep50 AUC at any floor;\n"
        "only f=.21 has interim checkpoints.",
        fontsize=6.5,
        bbox={"facecolor": "#fff7ed", "edgecolor": "#fdba74", "pad": 2},
    )
    clean_axes(ax)

    ax = axes[5]
    ax.axis("off")
    ax.text(
        0.02,
        0.96,
        "Paired blank-rate comparisons\n"
        "stored in the JSON are against\n"
        "f=.15, not the preceding floor.\n\n"
        "All composition points:\n"
        "n=6,137 slices.\n\n"
        "AUC across floors remains\n"
        "unmeasured.",
        va="top",
        fontsize=7.2,
    )

    fig.suptitle(
        "COVER visibility-floor dose response",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "fig3_cover_floor_dose_response")


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def figure_auc_trajectories() -> None:
    """F4: all clean frozen-probe milestones, plus an explicit pending point."""
    ep50 = load_json(
        REPORTS / "composition_vs_auc" / "composition_vs_auc_ep50.json"
    )
    ep75 = load_json(
        REPORTS / "composition_vs_auc" / "composition_vs_auc_ep75.json"
    )
    ep100 = load_json(
        REPORTS / "composition_vs_auc" / "composition_vs_auc_ep100.json"
    )

    def comp_auc(report: dict, arm: str) -> float:
        for row in report["rows"]:
            if row["arm"] == arm:
                return float(row["auc"])
        raise KeyError(arm)

    fork = result_auc("frozen_meanpool_fork_ep25")
    series = {
        "random rectangles": (
            [25, 50, 75, 100],
            [fork, comp_auc(ep50, "random"), comp_auc(ep75, "random"), comp_auc(ep100, "random")],
            COLORS["random"],
            "-",
        ),
        "oracle band rectangles": (
            [25, 50, 75, 100],
            [fork, comp_auc(ep50, "oracle"), comp_auc(ep75, "oracle"), comp_auc(ep100, "oracle")],
            COLORS["oracle"],
            "-",
        ),
        "MIRAGE envelope rectangles": (
            [25, 30, 50, 75, 100],
            [
                fork,
                result_auc("frozen_meanpool_envelope_ep30"),
                result_auc("frozen_meanpool_mirage_ep50"),
                result_auc("frozen_meanpool_mirage_ep75"),
                result_auc("frozen_meanpool_mirage_ep100"),
            ],
            COLORS["envelope"],
            "-",
        ),
        "anatomy-shaped v1": (
            [25, 30],
            [fork, result_auc("frozen_meanpool_anatomy_ep30")],
            COLORS["anatomy"],
            "--",
        ),
        "blob v2 (separate continuation)": (
            [25, 35, 40, 50],
            [
                fork,
                result_auc("frozen_meanpool_bridge_ep35"),
                result_auc("frozen_meanpool_bridge_ep40"),
                result_auc("frozen_meanpool_bridge_ep50"),
            ],
            COLORS["blob"],
            "-.",
        ),
        "COVER f=.21 (clean, incomplete)": (
            [25, 27, 30, 34],
            [
                fork,
                result_auc("frozen_meanpool_cover_f021_ep27"),
                result_auc("frozen_meanpool_cover_f021_ep30"),
                result_auc("frozen_meanpool_cover_f021_ep34"),
            ],
            COLORS["cover"],
            "-",
        ),
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    for label, (epochs, aucs, color, style) in series.items():
        ax.plot(
            epochs,
            aucs,
            marker="o",
            markersize=4,
            linewidth=1.5,
            linestyle=style,
            color=color,
            label=label,
        )

    ax.text(
        0.03,
        0.94,
        "COVER ep50 unmeasured",
        transform=ax.transAxes,
        fontsize=6.5,
        color=COLORS["cover"],
        va="top",
        bbox={"facecolor": "white", "edgecolor": COLORS["cover"], "pad": 1.5},
    )
    ax.annotate(
        "ep30 anatomy v1 > envelope",
        (30, result_auc("frozen_meanpool_anatomy_ep30")),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=6.5,
        color=COLORS["anatomy"],
    )
    ax.annotate(
        "ep50 blob v2 < envelope",
        (50, result_auc("frozen_meanpool_bridge_ep50")),
        xytext=(7, -15),
        textcoords="offset points",
        fontsize=6.5,
        color=COLORS["blob"],
    )
    ax.set_xlabel("Pretraining epoch")
    ax.set_ylabel("Frozen mean-pool test AUC")
    ax.set_xlim(23, 102)
    ax.set_ylim(0.846, 0.889)
    ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="lower right")
    clean_axes(ax)
    ax.set_title(
        "Matched frozen-probe milestones from the common epoch-25 ancestor",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "fig4_auc_trajectories")


def figure_qualitative() -> None:
    """F5: preserve the measured diagnostic PNG and create a PDF wrapper."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    png_out = FIGURES / "fig5_zero_anatomy_example.png"
    shutil.copyfile(QUALITATIVE, png_out)

    image = mpimg.imread(QUALITATIVE)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(7.2, 7.2 * height / width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    fig.savefig(FIGURES / "fig5_zero_anatomy_example.pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure_crop_defect()
    figure_context_excision()
    figure_composition_auc()
    figure_cover_dose_response()
    figure_auc_trajectories()
    figure_qualitative()
    for path in sorted(FIGURES.glob("fig*")):
        print(path)


if __name__ == "__main__":
    main()
