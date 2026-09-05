"""Render two source-linked replacements, without sampling or raw OCT export."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OUT = HERE / "replacements"
GEOMETRY = ROOT / "results" / "masking" / "table2_geometry" / "mask_geometry_600slices_bs1_coverf021_seed42.json"
INVENTORY = Path(r"D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json")
FIXTURE = ROOT / ".audit" / "delivered_task" / "private_real_fixtures" / "real_b1_b2_final_masks_v2.pt"
FIXTURE_SHA256 = "269f1c143af8e91daa179796fb0bfbd40583d91fafebbc0dbd45cba9c6c4692e"
AUDIT_ROWS = ROOT / "autopilot" / "investigations" / "delivered_task" / "evidence" / "mask_final64_v2"
SCATTER_STEM = "fig_purity_auc_ep50_fp32"
MAP_STEM = "fig_policy_family_token_maps"
POLICIES = (
    ("random", "random", "random", "Random"),
    ("oracle", "oracle", "oracle", "Centroid"),
    ("envelope", "envelope", "envelope", "Envelope"),
    ("anatomy", "anatomy-v2", "anatomy", "ANATOMY-v2"),
    ("cover", "cover-f021", "cover_legacy", "COVER"),
)
COLORS = ("#595959", "#0072B2", "#009E73", "#CC79A7", "#D55E00")
MARKERS = ("o", "s", "^", "D", "P")
TARGET, CONTEXT, UNUSED = "#E69F00", "#56B4E9", "#F2F2F2"
STYLE = {"font.family": "DejaVu Sans", "font.size": 9,
         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
         "svg.hashsalt": "jepa-source-reviewed-replacements-v1", "hatch.linewidth": .35}
PNG_OPTIONS = {"dpi": 300, "facecolor": "white", "transparent": False}
SCATTER_CAPTION = (
    "Delivered target-union tissue purity and frozen-probe test AUC for five policy "
    "families. Purity uses the current Table 2 geometry audit; all AUC points use "
    "epoch-50 fp32 probes. Centroid is the artifact arm named oracle, and ANATOMY-v2 "
    "is the shaped-policy representative; ANATOMY-v1 is not shown. Points are "
    "descriptive estimates from the evaluated continuations, with no fitted line "
    "or uncertainty intervals. The AUC axis does not start at zero. These "
    "confounded implementations do not isolate the effect of tissue purity or "
    "establish a policy ranking over retrainings."
)
MAP_CAPTION = (
    "Delivered token sets for five policy families on the first predeclared "
    "Training view in the frozen two-view engineering-audit fixture. Orange "
    "cells are the union of final predictor targets, hatched blue cells are "
    "final encoder context, and hollow circles mark occupancy-guide-positive "
    "cells; other cells are neither target nor context. The guide is a tissue "
    "proxy, not clinical ground truth. All panels use the same input view and "
    "the already stored masks, without redrawing seeds or displaying raw OCT "
    "pixels. ANATOMY-v2 represents the shaped family, not ANATOMY-v1; COVER "
    "shows the historical implementation, not the proposed correction. "
    "Placements are not exactly paired across policies, and unions do not "
    "show repeated loss slots. This selected engineering example is not a "
    "population summary or evidence of downstream performance."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def scatter_data(geometry_path=GEOMETRY, inventory_path=INVENTORY):
    geometry = json.loads(Path(geometry_path).read_text(encoding="utf-8"))
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    rows = []
    for geometry_arm, auc_arm, _, display in POLICIES:
        selected = [(index, row) for index, row in enumerate(inventory["records"])
                    if row.get("arm") == auc_arm and row.get("epoch") == 50
                    and row.get("precision") == "fp32" and row.get("family") == "frozen_probe"
                    and row.get("status") == "primary"]
        if len(selected) != 1:
            raise ValueError("Missing or duplicate matched-fp32 AUC source: " + auc_arm)
        index, record = selected[0]
        rows.append({"arm": auc_arm, "geometry_arm": geometry_arm, "display": display,
                     "x": geometry[geometry_arm]["hidden_pct_on_anat"], "y": record["auc"],
                     "x_expression": {"source": "geometry42", "pointer": f"/{geometry_arm}/hidden_pct_on_anat"},
                     "y_expression": {"source": "p1b_full_inventory.json", "pointer": f"/records/{index}/auc"},
                     "inventory_row": index, "epoch": record["epoch"], "precision": record["precision"]})
    return rows


def build_scatter(geometry_path=GEOMETRY, inventory_path=INVENTORY):
    rows = scatter_data(geometry_path, inventory_path)
    offsets = ((8, 9), (8, -16), (8, 9), (-8, 10), (8, 9))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 3.9))
        fig.subplots_adjust(left=.13, right=.985, top=.865, bottom=.195)
        ax.set_gid("purity_auc_axes")
        for row, color, marker, offset in zip(rows, COLORS, MARKERS, offsets):
            point = ax.scatter([row["x"]], [row["y"]], s=60, marker=marker,
                               facecolor=color, edgecolor="black", linewidth=.6, zorder=3)
            point.set_gid("purity_auc__" + row["arm"])
            annotation = ax.annotate(row["display"], (row["x"], row["y"]),
                                     xytext=offset, textcoords="offset points",
                                     ha="right" if offset[0] < 0 else "left",
                                     fontsize=9, annotation_clip=False)
            annotation.set_gid("policy_label__" + row["arm"])
        ax.set(xlim=(20, 105), ylim=(.8615, .879),
               xlabel="Delivered target-union tissue purity (%)",
               ylabel="Frozen-probe test AUC")
        ax.set_title("Target purity and frozen-probe performance", pad=25, fontsize=11)
        fig.text(.557, .878, "Matched epoch 50, fp32 point estimates; no intervals shown.",
                 ha="center", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        return fig


def frozen_token_data(fixture_path=FIXTURE):
    import torch
    if sha(fixture_path) != FIXTURE_SHA256:
        raise ValueError("Frozen private fixture identity changed")
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    batch = fixture["batches"]["bs2"]
    if batch["ordinals"] != [0, 1] or fixture["metadata"]["split"] != "Training":
        raise ValueError("Predeclared Training selection changed")
    if not batch["guide_valid"].all() or tuple(batch["guides"].shape) != (2, 4, 16, 16):
        raise ValueError("Guide shape/validity changed")
    tissue = batch["tissue_labels"][0].numpy()
    if not np.array_equal(tissue, batch["guides"][0, 0].numpy() >= .25):
        raise ValueError("Stored tissue labels disagree with occupancy definition")
    crop_sha = array_sha(batch["images"][0].numpy())
    guide_sha = array_sha(batch["guides"][0].numpy())
    tokens, provenance = [], []
    for _, auc_arm, fixture_arm, display in POLICIES:
        policy = batch["policies"][fixture_arm]
        if len(policy["masks_enc"]) != 1 or len(policy["masks_pred"]) != 4:
            raise ValueError("Unexpected frozen target/context group count")
        targets = [v[0].numpy().copy() for v in policy["masks_pred"]]
        context = policy["masks_enc"][0][0].numpy().copy()
        target_union = np.unique(np.concatenate(targets))
        if np.intersect1d(target_union, context).size:
            raise ValueError("Target/context overlap in frozen source")
        if min(target_union.min(), context.min()) < 0 or max(target_union.max(), context.max()) >= 256:
            raise ValueError("Out-of-grid frozen mask")
        audit_path = AUDIT_ROWS / (fixture_arm + "_bs2.jsonl")
        audit = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
        if audit["ordinal"] != 0 or audit["batch_size"] != 2:
            raise ValueError("Historical audit row is not the predeclared first BS2 view")
        if audit["crop_tensor_sha256"] != crop_sha or audit["guide_sha256"] != guide_sha:
            raise ValueError("Stored view differs from the audited crop/guide")
        if not np.array_equal(audit["context"][0], context):
            raise ValueError("Frozen context differs from audited final row")
        if any(not np.array_equal(a, b) for a, b in zip(audit["targets"], targets)):
            raise ValueError("Frozen targets differ from audited final row")
        tokens.append({"arm": auc_arm, "fixture_arm": fixture_arm, "display": display,
                       "target_union": target_union, "context": context, "tissue": tissue.copy()})
        provenance.append({
            "arm": auc_arm, "fixture_policy_key": fixture_arm, "anonymous_ordinal": 0,
            "fixture_target_pointer": f"/batches/bs2/policies/{fixture_arm}/masks_pred",
            "fixture_context_pointer": f"/batches/bs2/policies/{fixture_arm}/masks_enc/0/0",
            "tissue_pointer": "/batches/bs2/tissue_labels/0",
            "target_union_int64_sha256": array_sha(target_union.astype(np.int64)),
            "context_int64_sha256": array_sha(context.astype(np.int64)),
            "tissue_bool_sha256": array_sha(tissue.astype(bool)),
            "audit_source": {"root": "repo", "path": str(audit_path.relative_to(ROOT)), "sha256": sha(audit_path)},
            "audit_row_ordinal": 0,
        })
    del fixture, batch
    return tokens, {"private_fixture_sha256": FIXTURE_SHA256, "private_fixture_redistributed": False,
                    "split": "Training", "anonymous_ordinal": 0, "frozen_batch_key": "bs2",
                    "selection": "first of the already predeclared two audited views; no redraw or search",
                    "crop_tensor_sha256": crop_sha, "guide_sha256": guide_sha,
                    "tissue_definition": "occupancy channel 0 >= 0.25; tissue proxy, not clinical ground truth",
                    "raw_oct_pixels_rendered": False, "per_case_identifiers_exported": False,
                    "policies": provenance}


def build_token_maps(tokens):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 5, figsize=(7.3, 2.55))
        fig.subplots_adjust(left=.02, right=.99, top=.88, bottom=.31, wspace=.10)
        for letter, ax, row in zip("abcde", axes, tokens):
            target, context = set(row["target_union"]), set(row["context"])
            ax.set_gid("token_map__" + row["arm"])
            for index in range(256):
                y, x = divmod(index, 16)
                cell = Rectangle((x, y), 1, 1, facecolor=TARGET if index in target else CONTEXT if index in context else UNUSED,
                                 edgecolor="#444444", linewidth=.15, hatch="///" if index in context else None)
                cell.set_gid(f"token__{row['arm']}__{index}")
                ax.add_patch(cell)
            y, x = np.nonzero(row["tissue"])
            tissue = ax.scatter(x + .5, y + .5, s=7, marker="o", facecolors="none",
                                edgecolors="black", linewidths=.55, zorder=3)
            tissue.set_gid("tissue__" + row["arm"])
            ax.set(xlim=(0, 16), ylim=(16, 0), aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"({letter}) {row['display']}", fontsize=8.5, pad=7)
        fig.legend(handles=[
            Patch(facecolor=TARGET, edgecolor="black", label="Final target union"),
            Patch(facecolor=CONTEXT, edgecolor="black", hatch="///", label="Final encoder context"),
            Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="black",
                   linestyle="none", markersize=4, label="Guide-positive tissue proxy"),
            Patch(facecolor=UNUSED, edgecolor="black", label="Neither"),
        ], loc="lower center", bbox_to_anchor=(.5, .055), ncol=2, fontsize=8,
           frameon=False, handlelength=1.5, columnspacing=2.2)
        fig.text(.5, .015, "Frozen delivered token sets; one predeclared Training view; no raw OCT pixels.",
                 ha="center", fontsize=7.5)
        return fig


def png_bytes(fig):
    stream = io.BytesIO()
    with plt.rc_context(STYLE):
        fig.savefig(stream, format="png", **PNG_OPTIONS)
    return stream.getvalue()


def export(fig, out, stem):
    (out / (stem + ".png")).write_bytes(png_bytes(fig))
    with plt.rc_context(STYLE):
        fig.savefig(out / (stem + ".pdf"), facecolor="white", transparent=False,
                    metadata={"Author": "", "CreationDate": None, "ModDate": None})
        fig.savefig(out / (stem + ".svg"), facecolor="white", transparent=False,
                    metadata={"Date": None, "Creator": "Source-reviewed JEPA replacement generator"})
    return {ext: {"path": stem + "." + ext, "sha256": sha(out / (stem + "." + ext))}
            for ext in ("png", "pdf", "svg")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(HERE):
        raise ValueError("Replacement output must stay in the owned investigation directory")
    out.mkdir(parents=True, exist_ok=True)
    if not args.force and any(out.glob("fig_*")):
        raise FileExistsError("Replacement assets exist; pass --force only to rebuild this owned output")
    import verify_replacements as verifier
    scatter = build_scatter()
    token_data, token_sources = frozen_token_data()
    maps = build_token_maps(token_data)
    scatter_checks = verifier.verify_scatter_artists(scatter)
    map_checks = verifier.verify_token_artists(maps)
    exports = {SCATTER_STEM: export(scatter, out, SCATTER_STEM),
               MAP_STEM: export(maps, out, MAP_STEM)}
    validation = {
        "scatter": scatter_checks, "token_maps": map_checks,
        "scatter_png": verifier.verify_png(scatter, out / (SCATTER_STEM + ".png")),
        "token_maps_png": verifier.verify_png(maps, out / (MAP_STEM + ".png")),
        "vector_files": verifier.verify_vectors(out),
        "scatter_text_layout": verifier.verify_text_layout(scatter),
        "token_text_layout": verifier.verify_text_layout(maps),
        "no_training_gpu_network_or_resampling": True,
    }
    plt.close(scatter)
    plt.close(maps)
    for stem, caption in ((SCATTER_STEM, SCATTER_CAPTION), (MAP_STEM, MAP_CAPTION)):
        (out / (stem + ".caption.txt")).write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "version": 1,
        "generator": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha(__file__)},
        "independent_validator": {"path": str(Path(verifier.__file__).relative_to(ROOT)), "sha256": sha(verifier.__file__)},
        "public_numeric_sources": {
            "geometry42": {"root": "repo", "path": str(GEOMETRY.relative_to(ROOT)), "sha256": sha(GEOMETRY)},
            "p1b_full_inventory.json": {"root": "stats", "path": INVENTORY.name, "sha256": sha(INVENTORY)},
        },
        "scatter": {"series": scatter_data(), "auc_selection": "primary frozen_probe epoch=50 precision=fp32",
                    "no_fitted_or_reference_lines": True, "no_uncertainty_claim": True},
        "token_maps": token_sources,
        "outputs": exports,
        "limits": "Verification checks source-to-artist/pixel correspondence. The mask panel is one illustrative engineering example; neither replacement is causal or retraining evidence.",
    }
    write_json(out / "source_manifest.json", manifest)
    write_json(out / "independent_validation.json", validation)
    verifier.write_registration(out, manifest, validation)
    print(json.dumps({"out": str(out.relative_to(ROOT)), "scatter_points": 5,
                      "token_panels": 5, "png_replays_exact": True, "raw_image_exports": 0}))


if __name__ == "__main__":
    main()
