"""Independent source-field, frozen-mask, artist, and delivered-byte checks."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
GEOMETRY = ROOT / "results" / "masking" / "table2_geometry" / "mask_geometry_600slices_bs1_coverf021_seed42.json"
INVENTORY = Path(r"D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json")
FIXTURE = ROOT / ".audit" / "delivered_task" / "private_real_fixtures" / "real_b1_b2_final_masks_v2.pt"
RECONCILIATION = ROOT / "autopilot" / "investigations" / "delivered_task" / "evidence" / "mask_real_loss_handoff_v1" / "guided_loss_reconciliation.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(actual, expected, description):
    observed, source = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if observed.shape != source.shape or not np.allclose(observed, source, atol=1e-12, rtol=0):
        raise ValueError("Source/artist mismatch: " + description)


def require(condition, description):
    if not condition:
        raise ValueError(description)


def common_figure(fig, axes):
    require(len(fig.axes) == axes, "Unexpected axes count")
    require(not fig.images and not fig.artists, "Uncovered figure-level artist or raster image")
    for ax in fig.axes:
        require(ax.name == "rectilinear", "Only two-dimensional Cartesian axes permitted")
        require(not ax.images and not ax.artists, "Raw/raster or uncovered axes artist")
        require(ax._left_title.get_text() == ax._right_title.get_text() == "", "Uncovered extra title")


def verify_scatter_artists(fig, geometry_path=GEOMETRY, inventory_path=INVENTORY):
    common_figure(fig, 1)
    geometry = json.loads(Path(geometry_path).read_text(encoding="utf-8"))
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    expected = [
        ("random", "random", "Random", "#595959", "o"),
        ("oracle", "oracle", "Centroid", "#0072B2", "s"),
        ("envelope", "envelope", "Envelope", "#009E73", "^"),
        ("anatomy", "anatomy-v2", "ANATOMY-v2", "#CC79A7", "D"),
        ("cover", "cover-f021", "COVER", "#D55E00", "P"),
    ]
    ax = fig.axes[0]
    require(len(ax.collections) == len(ax.texts) == 5, "Exactly five markers and semantic labels required")
    require(not ax.lines and not ax.patches and not ax.containers and not fig.legends,
            "Fitted/reference lines, intervals, or uncovered scatter artists are forbidden")
    require(ax.get_gid() == "purity_auc_axes", "Scatter semantic axis changed")
    records = []
    for (geom_arm, auc_arm, display, color, shape), point, text in zip(expected, ax.collections, ax.texts):
        selected = [(i, r) for i, r in enumerate(inventory["records"])
                    if (r.get("family"), r.get("status"), r.get("arm"), r.get("epoch"), r.get("precision"))
                    == ("frozen_probe", "primary", auc_arm, 50, "fp32")]
        require(len(selected) == 1, "Missing/duplicate all-fp32 semantic inventory row")
        index, row = selected[0]
        xy = [geometry[geom_arm]["hidden_pct_on_anat"], row["auc"]]
        equal(point.get_offsets(), [xy], auc_arm + " exact fp32 point")
        require(point.get_offset_transform() == ax.transData, "Point data transform changed")
        require(point.get_gid() == "purity_auc__" + auc_arm, "Marker semantic arm changed")
        equal(point.get_facecolors(), [to_rgba(color)], auc_arm + " marker color")
        equal(point.get_sizes(), [60], auc_arm + " marker area")
        marker = MarkerStyle(shape)
        equal(point.get_paths()[0].vertices,
              marker.get_path().transformed(marker.get_transform()).vertices, auc_arm + " redundant marker shape")
        require(point.get_visible() and point.get_alpha() in (None, 1), "Hidden or transparent data marker")
        require(text.get_text() == display and text.get_gid() == "policy_label__" + auc_arm,
                "Scatter semantic label changed")
        equal(text.xy, xy, auc_arm + " label data anchor")
        require(text.xycoords == "data" and text.arrow_patch is None and text.get_visible(),
                "Uncovered label coordinate mapping")
        records.append({"arm": auc_arm, "display_label": display,
                        "observed_xy": np.asarray(point.get_offsets()).tolist()[0],
                        "x_expression": {"source": "geometry42", "pointer": f"/{geom_arm}/hidden_pct_on_anat"},
                        "y_expression": {"source": "p1b_full_inventory.json", "pointer": f"/records/{index}/auc"},
                        "epoch": row["epoch"], "precision": row["precision"]})
    equal(ax.get_xlim(), [20, 105], "scatter x limits")
    equal(ax.get_ylim(), [.8615, .879], "scatter y limits")
    require(ax.get_xscale() == ax.get_yscale() == "linear", "Unexpected scatter scale")
    require(ax.get_xlabel() == "Delivered target-union tissue purity (%)" and
            ax.get_ylabel() == "Frozen-probe test AUC", "Scatter quantity labels changed")
    require(ax.get_title() == "Target purity and frozen-probe performance", "Unreviewed scatter title")
    require([t.get_text() for t in fig.texts] ==
            ["Matched epoch 50, fp32 point estimates; no intervals shown."], "Unreviewed scatter annotation")
    return {"status": "source_fields_and_all_data_artists_verified",
            "geometry_sha256": sha(geometry_path), "inventory_sha256": sha(inventory_path),
            "series": records, "line_or_interval_count": 0,
            "scope": "Five exact source-valued markers and their semantic annotations; not statistical/causal validation."}


def verify_token_artists(fig, fixture_path=FIXTURE):
    import torch
    common_figure(fig, 5)
    reconciliation = json.loads(RECONCILIATION.read_text(encoding="utf-8"))
    expected_fixture_hash = reconciliation["gpu_used_original_container_sha256"]
    require(sha(fixture_path) == expected_fixture_hash, "Original frozen private fixture hash mismatch")
    data = torch.load(fixture_path, map_location="cpu", weights_only=True)
    batch = data["batches"]["bs2"]
    require(data["metadata"]["split"] == "Training" and batch["ordinals"] == [0, 1],
            "Not the first predeclared audited Training view")
    require(tuple(batch["images"].shape) == (2, 3, 256, 256), "Unexpected private fixture shape")
    require(tuple(batch["guides"].shape) == (2, 4, 16, 16) and bool(batch["guide_valid"][0]),
            "Invalid guide source")
    expected_tissue = batch["guides"][0, 0].numpy() >= .25
    equal(batch["tissue_labels"][0].numpy(), expected_tissue, "tissue predicate vs stored array")
    ys, xs = np.where(expected_tissue)
    circle_positions = np.column_stack([xs + .5, ys + .5])
    rows = [
        ("random", "random", "(a) Random"), ("oracle", "oracle", "(b) Centroid"),
        ("envelope", "envelope", "(c) Envelope"), ("anatomy", "anatomy-v2", "(d) ANATOMY-v2"),
        ("cover_legacy", "cover-f021", "(e) COVER"),
    ]
    evidence = []
    for ax, (key, arm, title) in zip(fig.axes, rows):
        policy = batch["policies"][key]
        require(len(policy["masks_enc"]) == 1 and len(policy["masks_pred"]) == 4,
                "Changed final mask group contract")
        for array in policy["masks_enc"] + policy["masks_pred"]:
            require(array.dtype == torch.int64 and array.ndim == 2 and array.shape[0] == 2,
                    "Invalid frozen index tensor")
        targets = torch.unique(torch.cat([m[0] for m in policy["masks_pred"]])).numpy()
        context = policy["masks_enc"][0][0].numpy()
        require(not np.intersect1d(targets, context).size, "Source target/context overlap")
        require(min(targets.min(), context.min()) >= 0 and max(targets.max(), context.max()) < 256,
                "Source mask outside token grid")
        require(len(ax.patches) == 256 and len(ax.collections) == 1 and not ax.lines and not ax.texts,
                "Uncovered/missing token cells, circles, or extra annotations")
        require(not ax.containers and ax.get_gid() == "token_map__" + arm, "Uncovered token-map artist")
        target_set, context_set = set(targets), set(context)
        for index, cell in enumerate(ax.patches):
            require(isinstance(cell, Rectangle), "Token cell is not a unit rectangle")
            equal([*cell.get_xy(), cell.get_width(), cell.get_height()],
                  [index % 16, index // 16, 1, 1], "token-grid coordinates")
            require(cell.get_data_transform() == ax.transData, "Token data transform changed")
            category = "target" if index in target_set else "context" if index in context_set else "neither"
            color = {"target": "#E69F00", "context": "#56B4E9", "neither": "#F2F2F2"}[category]
            equal(cell.get_facecolor(), to_rgba(color), arm + " source token class")
            equal(cell.get_edgecolor(), to_rgba("#444444"), arm + " grid edge")
            require(cell.get_hatch() == ("///" if category == "context" else None),
                    "Visible-context hatch differs from exact delivered context")
            require(cell.get_gid() == f"token__{arm}__{index}" and cell.get_visible(),
                    "Token identity/visibility changed")
        circles = ax.collections[0]
        equal(circles.get_offsets(), circle_positions, arm + " exact guide-positive circles")
        require(circles.get_offset_transform() == ax.transData, "Tissue-circle data transform changed")
        equal(circles.get_sizes(), [7], arm + " tissue marker size")
        require(circles.get_gid() == "tissue__" + arm and len(circles.get_facecolors()) == 0,
                "Tissue circles must be hollow")
        equal(circles.get_edgecolors(), [to_rgba("black")], arm + " tissue-circle edges")
        circle = MarkerStyle("o")
        equal(circles.get_paths()[0].vertices,
              circle.get_path().transformed(circle.get_transform()).vertices, "tissue-circle path")
        equal(ax.get_xlim(), [0, 16], "token grid x-axis")
        equal(ax.get_ylim(), [16, 0], "token grid y-axis")
        require(ax.get_aspect() == 1 and not len(ax.get_xticks()) and not len(ax.get_yticks()),
                "Grid aspect/ticks changed")
        require(ax.get_title() == title and ax.get_xlabel() == ax.get_ylabel() == "",
                "Unreviewed token-map text/clinical annotation")
        evidence.append({"arm": arm, "fixture_policy_key": key, "anonymous_ordinal": 0,
                         "cells_checked": 256, "tissue_circles_checked": len(circle_positions),
                         "target_union_sha256": hashlib.sha256(targets.astype(np.int64).tobytes()).hexdigest(),
                         "context_sha256": hashlib.sha256(context.astype(np.int64).tobytes()).hexdigest(),
                         "exact_source_membership_checked": True})
    require(len(fig.legends) == 1, "Missing or additional token legend")
    require([t.get_text() for t in fig.legends[0].get_texts()] ==
            ["Final target union", "Final encoder context", "Guide-positive tissue proxy", "Neither"],
            "Unreviewed legend annotations")
    handles = fig.legends[0].legend_handles
    require(len(handles) == 4, "Uncovered legend glyph")
    equal(handles[0].get_facecolor(), to_rgba("#E69F00"), "legend target color")
    equal(handles[1].get_facecolor(), to_rgba("#56B4E9"), "legend context color")
    require(handles[1].get_hatch() == "///", "Legend context hatch changed")
    require(handles[2].get_marker() == "o" and handles[2].get_markerfacecolor() == "none",
            "Legend tissue glyph is not a hollow circle")
    equal(to_rgba(handles[2].get_markeredgecolor()), to_rgba("black"), "legend tissue edge")
    equal(handles[3].get_facecolor(), to_rgba("#F2F2F2"), "legend unused color")
    require([t.get_text() for t in fig.texts] ==
            ["Frozen delivered token sets; one predeclared Training view; no raw OCT pixels."],
            "Unreviewed figure text or hidden clinical annotations")
    require(sha(fixture_path) == expected_fixture_hash, "Private fixture changed during verification")
    return {"status": "exact_frozen_source_arrays_and_artists_verified",
            "private_fixture_sha256": expected_fixture_hash, "anonymous_ordinal": 0,
            "raw_image_artists": 0, "axes_dimension": 2, "policies": evidence,
            "scope": "Selected source-reviewed engineering token-map illustration; not performance or population verification."}


def verify_png(fig, path):
    before = sha(path)
    stream = io.BytesIO()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "jepa-source-reviewed-replacements-v1", "hatch.linewidth": .35}):
        fig.savefig(stream, format="png", dpi=300, facecolor="white", transparent=False)
    require(stream.getvalue() == Path(path).read_bytes(), "Delivered PNG differs from independently checked artist replay")
    with Image.open(path) as image:
        require(image.mode == "RGBA" and image.getchannel("A").getextrema() == (255, 255),
                "Unreviewed transparency in PNG")
        dimensions = list(image.size)
    require(sha(path) == before, "PNG changed during replay")
    return {"exact_byte_equality": True, "sha256": before, "dimensions_pixels": dimensions,
            "fully_opaque": True}


def verify_text_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        texts.extend(ax.texts)
        texts.extend(t for t in ax.get_xticklabels()
                     if min(ax.get_xlim()) <= t.get_position()[0] <= max(ax.get_xlim()))
        texts.extend(t for t in ax.get_yticklabels()
                     if min(ax.get_ylim()) <= t.get_position()[1] <= max(ax.get_ylim()))
    for legend in fig.legends:
        texts.extend(legend.get_texts())
    boxes = []
    for text in texts:
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1
                and box.y1 <= fig.bbox.height + 1, "Text extends outside figure: " + text.get_text())
        for prior_text, prior in boxes:
            overlap_x = min(prior.x1, box.x1) - max(prior.x0, box.x0)
            overlap_y = min(prior.y1, box.y1) - max(prior.y0, box.y0)
            require(overlap_x <= .5 or overlap_y <= .5,
                    "Text rectangles overlap: " + prior_text + " / " + text.get_text())
        boxes.append((text.get_text(), box))
    return {"visible_text_rectangles_checked": len(boxes),
            "outside_figure": 0, "text_rectangle_overlaps": 0,
            "scope": "Headless renderer geometry; not human readability or accessibility certification."}


def verify_vectors(out):
    import fitz
    result = {}
    for stem in ("fig_purity_auc_ep50_fp32", "fig_policy_family_token_maps"):
        svg = ET.parse(out / (stem + ".svg")).getroot()
        require(not any(node.tag.rsplit("}", 1)[-1] == "image" for node in svg.iter()),
                "Raster image unexpectedly embedded in SVG")
        with fitz.open(out / (stem + ".pdf")) as doc:
            require(len(doc) == 1 and not doc[0].get_images(full=True), "Raster image or extra page embedded in PDF")
            text = doc[0].get_text()
            if stem == "fig_policy_family_token_maps":
                require(not re.search(r"\bAUC\b|pred\s*=|probability|RNFL|pathology|vol\s+\d", text, re.I),
                        "Unexpected clinical or prediction annotation in token-map PDF")
            result[stem] = {"pdf_embedded_images": 0, "svg_image_elements": 0,
                            "pdf_pages": 1, "pdf_text": text}
    return result


def write_registration(out, manifest, validation):
    sys.path.insert(0, str(ROOT))
    from autopilot.numeric_bindings import digest_text
    import jsonschema
    generator = manifest["generator"]["path"]
    registration = {
        "version": 1, "producer": generator, "producer_sha256": manifest["generator"]["sha256"],
        "new_paper_path": "figures/fig_purity_auc_ep50_fp32.png",
        "staged_path": str((out / "fig_purity_auc_ep50_fp32.png").relative_to(ROOT)),
        "build_function": "build_scatter(geometry_path, inventory_path)",
        "independent_validator": "verify_replacements.verify_scatter_artists(fig, geometry_path, inventory_path)",
        "png_serializer": "generate_replacements.png_bytes(fig)",
        "public_source_hashes": manifest["public_numeric_sources"],
        "expected_semantic_series": validation["scatter"]["series"],
        "expected_point_gids": ["purity_auc__" + row["arm"] for row in validation["scatter"]["series"]],
        "expected_counts": {"axes": 1, "scatter_collections": 5, "annotations": 5, "lines": 0, "images": 0},
        "required_checks": ["strict primary frozen_probe epoch50 fp32 semantic row selection; reject duplicates",
                            "source expressions vs every marker offset and annotation anchor; no fit, reference, or interval artists",
                            "exact PNG bytes after independent artist verification; do not trust producer success or manifest values alone"],
        "caption": (out / "fig_purity_auc_ep50_fp32.caption.txt").read_text().strip(),
        "scope": "Registration handoff, not a numeric_reviews schema method or claim of existing gate integration.",
    }
    (out / "numeric_validator_registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    sources = {}
    for key, filename in (("replacement_policy_token_manifest", "source_manifest.json"),
                          ("replacement_policy_token_validation", "independent_validation.json")):
        path = out / filename
        sources[key] = {"root": "repo", "path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    caption = (out / "fig_policy_family_token_maps.caption.txt").read_text().strip()
    candidate = {
        "version": 1,
        "scope": "One new source-reviewed token illustration only. Scatter needs strict registered artist validation. Parent must install the new exact asset and unchanged caption before applying.",
        "sources": sources,
        "figures": [{
            "path": "figures/fig_policy_family_token_maps.png",
            "sha256": sha(out / "fig_policy_family_token_maps.png"),
            "caption_sha256": digest_text(caption),
            "inputs": [{"source": key, "pointer": "/token_maps", "sha256": value["sha256"]}
                       for key, value in sources.items()],
            "validation": {
                "method": "reviewed_source_illustration", "reviewer": "GitHub Copilot source-figure review, 2026-09-04",
                "quantitative_scope": "illustrative_only",
                "limitations": "Exact stored BS2 ordinal-zero target unions, context memberships and guide-positive circles were independently checked against the frozen private fixture and byte-replayed to this PNG. This is one predeclared engineering view with a tissue proxy, not raw OCT, clinical interpretation, population geometry, downstream AUC or retraining evidence. Anatomy is the v2 representative and COVER is historical; placements are not exactly paired and unions omit duplicate loss weights. Private image/guide tensors are not distributed.",
            },
        }],
    }
    jsonschema.validate(candidate, json.loads((ROOT / "autopilot" / "numeric_reviews.schema.json").read_text()))
    (out / "candidate_replacement_reviews.json").write_text(json.dumps(candidate, indent=2) + "\n")


if __name__ == "__main__":
    import generate_replacements as producer
    out = producer.OUT
    figures = [producer.build_scatter(), producer.build_token_maps(producer.frozen_token_data()[0])]
    try:
        verify_scatter_artists(figures[0])
        verify_token_artists(figures[1])
        for fig, stem in zip(figures, ("fig_purity_auc_ep50_fp32", "fig_policy_family_token_maps")):
            verify_png(fig, out / (stem + ".png"))
        verify_vectors(out)
        print("Source fields, frozen masks, two exact PNG replays, and vector-only exports verified.")
    finally:
        for fig in figures:
            plt.close(fig)
