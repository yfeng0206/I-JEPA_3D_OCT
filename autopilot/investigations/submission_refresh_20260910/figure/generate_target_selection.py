"""Export an honest token-map fallback; never export clinical image pixels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PRIVATE = ROOT / ".audit" / "masking_family_20260910" / "fixed_inputs.pt"
SOURCES = ROOT / "autopilot" / "investigations" / "delivered_task" / "evidence"
DEST = ROOT / "paper" / "genai4health2026" / "figures"
STEM = "fig_oct_target_selection"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    fixture = torch.load(PRIVATE, map_location="cpu", weights_only=False)
    index = fixture["ordinals"].index(0)
    guide = fixture["guides"][index].numpy()
    image_hash = hashlib.sha256(fixture["images"][index].numpy().tobytes()).hexdigest()
    guide_hash = hashlib.sha256(guide.tobytes()).hexdigest()
    tissue = guide[0] >= .25
    rows, source_hashes = {}, {}
    for arm in ("random", "envelope"):
        source = SOURCES / "mask_replay600_v2" / f"{arm}_bs64.jsonl"
        rows[arm] = next(json.loads(line) for line in source.read_text().splitlines()
                         if json.loads(line)["ordinal"] == 0)
        row = rows[arm]
        assert row["crop_tensor_sha256"] == image_hash
        assert row["guide_sha256"] == guide_hash
        targets = {t for group in row["targets"] for t in group}
        assert targets and all(0 <= t < 256 for t in targets)
        assert not targets.intersection(row["context"][0])
        assert int(tissue.sum()) == row["tissue_cells"]
        source_hashes[source.name] = sha(source)
    styles = {"font.family": "DejaVu Sans", "font.size": 8,
              "pdf.fonttype": 42, "ps.fonttype": 42,
              "svg.fonttype": "none", "svg.hashsalt": STEM}
    with plt.rc_context(styles):
        fig, axes = plt.subplots(1, 4, figsize=(6.4, 2.0))
        fig.subplots_adjust(left=.017, right=.985, bottom=.25, top=.86, wspace=.15)
        titles = ["(a) OCT input", "(b) RANDOM", "(c) Anatomical guide", "(d) ENVELOPE"]
        subtitles = ["Token coordinates only", "Unguided targets",
                     "Guide-positive cells", "Anatomy-guided targets"]
        for panel, ax in enumerate(axes):
            ax.set(xlim=(0, 16), ylim=(16, 0), aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[panel], fontsize=8, loc="left", pad=7, fontweight="bold")
            ax.text(.5, -.08, subtitles[panel], ha="center", va="top",
                    transform=ax.transAxes, fontsize=7.5)
            for y in range(16):
                for x in range(16):
                    face = "#F7F7F7"
                    if panel and tissue[y, x]:
                        face = "#D6D6D6" if panel != 2 else "#397D8A"
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face,
                                           edgecolor="#DADADA", linewidth=.16))
            if panel == 0:
                ax.text(8, 8, "Clinical pixels\nnot reproduced", ha="center",
                        va="center", fontsize=8, color="#333333",
                        bbox=dict(facecolor="white", edgecolor="#777777", pad=5))
            if panel in (1, 3):
                arm = "random" if panel == 1 else "envelope"
                for target in sorted({t for group in rows[arm]["targets"] for t in group}):
                    y, x = divmod(target, 16)
                    ax.add_patch(Rectangle((x, y), 1, 1,
                                           facecolor=(.90, .56, .10, .58),
                                           edgecolor="#995000", linewidth=.35))
                ys, xs = np.nonzero(tissue)
                ax.scatter(xs + .5, ys + .5, s=3, facecolors="none",
                           edgecolors="#252525", linewidths=.35, zorder=4)
            for spine in ax.spines.values():
                spine.set_linewidth(.6)
                spine.set_color("#555555")
        fig.legend(handles=[
            Patch(facecolor="#EAB763", edgecolor="#995000", label="Final target union"),
            Line2D([], [], marker="o", markersize=3, markerfacecolor="none",
                   markeredgecolor="#252525", linestyle="none", label="Guide-positive cell"),
        ], loc="lower center", bbox_to_anchor=(.5, .05), ncol=2,
                   fontsize=7.5, frameon=False)
        fig.text(.5, .015, "Fixed-view shared-guide replay; no clinical pixels; encoder context omitted.",
                 ha="center", fontsize=7)
        DEST.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf", "svg"):
            metadata = {"Creator": "Matplotlib", "Title": "Fixed-view target-selection token maps"}
            if suffix == "pdf":
                metadata.update(Author="", CreationDate=None, ModDate=None)
            if suffix == "svg":
                metadata["Date"] = None
            fig.savefig(DEST / f"{STEM}.{suffix}", dpi=300, facecolor="white",
                        transparent=False, metadata=metadata)
        plt.close(fig)
    png = DEST / f"{STEM}.png"
    with Image.open(png) as im:
        assert im.size == (1920, 600)
        assert np.asarray(im)[..., :3].std() > 10
        png_metadata = {"pixels": list(im.size), "dpi": list(im.info["dpi"]), "mode": im.mode}
    caption = (
        "Target-selection intuition on one fixed view. (a) Input token coordinates; "
        "clinical OCT pixels are not reproduced. (b) RANDOM final target union. "
        "(c) Guide-positive cells (occupancy at least 0.25). (d) ENVELOPE final target union. "
        "All panels use the same 16-by-16 coordinates; target overlays show exact saved "
        "delivered indices, not idealized rectangles. The first predeclared scoped Training "
        "view is used (ordinal 0). This fixed-view replay uses a shared adapted guide and "
        "does not reconstruct historical training batches. Encoder context is omitted. "
        "The illustration is not a clinical annotation, cohort summary, or performance claim."
    )
    evidence = {
        "status": "usable_token_map_fallback_no_clinical_pixels",
        "permission_decision": (
            "Primary FairVision dataset card declares CC BY-NC-ND 4.0. Explicit permission "
            "for public distribution of a cropped and overlaid clinical image was not verified. "
            "No clinical raster, source tensor, patient identifier, or private path is exported."
        ),
        "license_primary_url": "https://huggingface.co/datasets/harvardairobotics/FairVision/raw/0c795428de3455d8cd9af4c244b85d4f347f4907/README.md",
        "license_deed_url": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
        "license_checked_date": "2026-09-10",
        "license_source_sha256": sha(SOURCES / "literal_sources" / "public" / "fairvision_readme.txt"),
        "selection": {"ordinal": 0, "scope_seed": 42, "crop_seed": 91009,
                      "scope": "predeclared 24 Training volumes by 25 views"},
        "source_file_sha256": source_hashes,
        "private_fixture_sha256": sha(PRIVATE),
        "crop_tensor_sha256": image_hash,
        "guide_tensor_sha256": guide_hash,
        "guide_semantics": "shared adapted guide, occupancy channel 0 >= 0.25",
        "source_arrays_preserved_in_private_fixture": True,
        "clinical_pixels_exported": False,
        "transformations": ["threshold guide occupancy at 0.25",
                            "display union of exact final targets without smoothing",
                            "omit encoder context; retain same full token coordinates"],
        "public_data": {"guide_positive_indices": np.flatnonzero(tissue).tolist(),
                        "targets": {arm: row["targets"] for arm, row in rows.items()}},
        "figure_inches": [6.4, 2.0],
        "png": png_metadata,
        "outputs": {suffix: {"file": f"{STEM}.{suffix}", "sha256": sha(DEST / f"{STEM}.{suffix}")}
                    for suffix in ("png", "pdf", "svg")},
        "caption": caption,
        "alt_text": "Four aligned token maps show the OCT input coordinate grid with clinical "
                    "pixels withheld, RANDOM targets, guide-positive cells, and ENVELOPE targets. "
                    "Orange bordered cells are final target unions; hollow dots locate guide-positive "
                    "cells in target panels. This is a fixed-view shared-guide replay, not a performance comparison.",
        "latex": "\\begin{figure}[t]\n\\centering\n"
                 "\\includegraphics[width=\\linewidth]{figures/fig_oct_target_selection.pdf}\n"
                 "\\caption{" + caption + "}\n\\label{fig:target-selection}\n\\end{figure}",
        "validation": {"exact_source_hash_matches": True,
                       "target_context_disjoint": True,
                       "guide_cell_counts_match": True,
                       "publisher_specific_compliance": "not certified; requested 6.4-inch width used"},
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence["outputs"], indent=2))


if __name__ == "__main__":
    main()
