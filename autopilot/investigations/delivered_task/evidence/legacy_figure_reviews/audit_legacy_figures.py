"""Headless, read-only legacy-asset inspection; never issues an empirical approval."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ["MPLBACKEND"] = "Agg"
import fitz
import numpy as np
from PIL import Image
from scipy.ndimage import label

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PAPER = REPO / "paper" / "genai4health2026"
STATS = Path(r"D:\jepa_phase0\autopilot_out\p1_stats")
sys.path.insert(0, str(REPO))
from autopilot.numeric_bindings import digest_text, figure_context

NAMES = (
    "fig_masking_policies", "fig_precision_paradox",
    "interp_04_window_occlusion_W7", "interp_14_odos_mirror_test",
    "interp_heatmap_grid", "interp_slice_contribution_by_outcome",
    "interp_slice_contribution_curves",
)
GEOMETRY = REPO / "results" / "masking" / "table2_geometry" / "mask_geometry_600slices_bs1_coverf021_seed42.json"
INVENTORY = STATS / "p1b_full_inventory.json"
ARM_MAP = {"random": "random", "oracle": "oracle", "envelope": "envelope",
           "cover-f0.21": "cover", "anatomy-v2": "anatomy"}
SOURCE_PRECISION = {"random": "fp16", "oracle": "fp16", "envelope": "fp16",
                    "cover-f0.21": "fp32", "anatomy-v2": "fp32"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, data):
    path = HERE / name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def source(path, root="repo"):
    roots = {"repo": REPO, "paper": PAPER, "stats": STATS}
    return {"root": root, "path": str(path.relative_to(roots[root])), "sha256": sha(path)}


def affine(values, positions):
    values, positions = np.asarray(values), np.asarray(positions)
    if len(values) < 3 or len(set(values)) != len(values):
        raise ValueError("At least three independent numeric ticks required")
    scale, offset = np.polyfit(values, positions, 1)
    residual = float(np.max(np.abs(scale * values + offset - positions)))
    if residual > 0.0001:
        raise ValueError("PDF numeric ticks do not define one affine axis")
    return {"scale": float(scale), "offset": float(offset),
            "max_residual_pdf_points": residual,
            "tick_values": values.tolist(), "tick_positions": positions.tolist()}


def decode(position, mapping):
    return (position - mapping["offset"]) / mapping["scale"]


def groups(indices):
    return [v for v in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1) if len(v)]


def raster_marks(image, markers, axes):
    rgb = np.asarray(image.convert("RGB")).astype(int)
    dark = rgb.max(axis=2) < 50
    row_counts, col_counts = dark.sum(axis=1), dark.sum(axis=0)
    rows = groups(np.flatnonzero(row_counts > row_counts.max() * 0.8))
    cols = groups(np.flatnonzero(col_counts > col_counts.max() * 0.8))
    if len(rows) != 2 or len(cols) != 2:
        raise ValueError("Cannot independently resolve the four PNG axis spines")
    bounds = [float(cols[0].mean()), float(rows[0].mean()),
              float(cols[1].mean()), float(rows[1].mean())]
    out = []
    for mark in markers:
        color = np.rint(np.asarray(mark["fill"]) * 255).astype(int)
        components, _ = label(np.max(np.abs(rgb - color), axis=2) <= 1)
        counts = np.bincount(components.ravel())
        counts[0] = 0
        ys, xs = np.nonzero(components == counts.argmax())
        if len(xs) < 1000:
            raise ValueError("No sufficiently large colored PNG marker")
        actual = [(float(xs.min()) + float(xs.max())) / 2,
                  (float(ys.min()) + float(ys.max())) / 2]
        expected = [
            bounds[i] + (mark["center_pdf"][i] - axes[i]) /
            (axes[i + 2] - axes[i]) * (bounds[i + 2] - bounds[i])
            for i in (0, 1)
        ]
        error = float(np.max(np.abs(np.asarray(actual) - expected)))
        out.append({"arm": mark["arm"], "rgb": color.tolist(),
                    "center_px": actual, "expected_from_pdf_px": expected,
                    "max_abs_error_px": error, "concordant_within_1_5_px": error <= 1.5,
                    "fill_pixel_count": len(xs)})
    return {"axes_spines_px": bounds, "markers": out,
            "all_marker_centers_concordant": all(x["concordant_within_1_5_px"] for x in out),
            "scope": "Independent PNG colored-marker/spine geometry only; not full raster text verification."}


def precision_audit():
    pdf_path = PAPER / "figures" / "fig_precision_paradox.pdf"
    png_path = pdf_path.with_suffix(".png")
    doc = fitz.open(pdf_path)
    page = doc[0]
    drawings = page.get_drawings()
    spans = [s for b in page.get_text("dict")["blocks"] if b["type"] == 0
             for ln in b["lines"] for s in ln["spans"]]
    black_lines = [d for d in drawings if d["type"] == "s" and
                   d["color"] == (0.0, 0.0, 0.0) and len(d["items"]) == 1]
    vertical = [d["rect"] for d in black_lines if d["rect"].width == 0 and d["rect"].height > 100]
    horizontal = [d["rect"] for d in black_lines if d["rect"].height == 0 and d["rect"].width > 100]
    if len(vertical) != 2 or len(horizontal) != 2:
        raise ValueError("Unexpected PDF axes layout")
    axes = [min(r.x0 for r in vertical), min(r.y0 for r in horizontal),
            max(r.x0 for r in vertical), max(r.y0 for r in horizontal)]
    ticks = [d["rect"] for d in drawings if d["type"] == "fs" and len(d["items"]) == 1]
    numeric = [s for s in spans if re.fullmatch(r"\d+(?:\.\d+)?", s["text"])]
    xt = sorted(r.x0 for r in ticks if r.height > 0 and abs(r.y0 - axes[3]) < 0.01)
    yt = sorted((r.y0 for r in ticks if r.width > 0 and abs(r.x1 - axes[0]) < 0.01), reverse=True)
    xlabels = [s for s in numeric if s["bbox"][1] > axes[3]]
    ylabels = [s for s in numeric if s["bbox"][2] < axes[0]]

    def nearest_label(pos, labels, dim):
        selected = min(labels, key=lambda s: abs((s["bbox"][dim] + s["bbox"][dim + 2]) / 2 - pos))
        if abs((selected["bbox"][dim] + selected["bbox"][dim + 2]) / 2 - pos) > 0.2:
            raise ValueError("PDF tick and numeric label are not aligned")
        return float(selected["text"])

    xmap = affine([nearest_label(p, xlabels, 0) for p in xt], xt)
    ymap = affine([nearest_label(p, ylabels, 1) for p in yt], yt)
    circles = [d for d in drawings if d["type"] == "fs" and len(d["items"]) == 8
               and all(item[0] == "c" for item in d["items"])]
    if len(circles) != 5:
        raise ValueError("Unexpected number or shape of PDF data markers")
    geometry = json.loads(GEOMETRY.read_text())
    inventory = json.loads(INVENTORY.read_text())
    marks = []
    for arm, geometry_arm in ARM_MAP.items():
        arm_labels = [s for s in spans if s["text"] == arm]
        if len(arm_labels) != 1:
            raise ValueError("Ambiguous PDF arm label")
        label_x = (arm_labels[0]["bbox"][0] + arm_labels[0]["bbox"][2]) / 2
        mark = min(circles, key=lambda d: abs((d["rect"].x0 + d["rect"].x1) / 2 - label_x))
        center = [(mark["rect"].x0 + mark["rect"].x1) / 2,
                  (mark["rect"].y0 + mark["rect"].y1) / 2]
        if abs(center[0] - label_x) > 0.02:
            raise ValueError("PDF marker and arm label are not aligned")
        x, y = decode(center[0], xmap), decode(center[1], ymap)
        source_arm = "cover-f021" if arm == "cover-f0.21" else arm
        records = [(i, r) for i, r in enumerate(inventory["records"])
                   if r["arm"] == source_arm and r.get("epoch") == 50
                   and r.get("precision") == SOURCE_PRECISION[arm]
                   and r.get("family") == "frozen_probe" and r.get("status") == "primary"]
        if len(records) != 1:
            raise ValueError("Ambiguous historical AUC source selection")
        index, record = records[0]
        expected_x, expected_y = geometry[geometry_arm]["hidden_pct_on_anat"], record["auc"]
        marks.append({
            "arm": arm, "fill": list(mark["fill"]), "center_pdf": center,
            "observed_purity_percent": x, "observed_auc": y,
            "source_purity_percent": expected_x, "source_auc": expected_y,
            "purity_difference_percentage_points": x - expected_x,
            "purity_matches_rounding_to_1_decimal": abs(x - expected_x) <= 0.050001,
            "auc_matches_rounding_to_6_decimals": abs(y - expected_y) <= 0.000000501,
            "geometry_pointer": "/" + geometry_arm + "/hidden_pct_on_anat",
            "inventory_pointer": "/records/%d/auc" % index,
            "source_precision": SOURCE_PRECISION[arm], "source_tag": record["tag"],
        })
    xs = np.array([m["observed_purity_percent"] for m in marks])
    ys = np.array([m["observed_auc"] for m in marks])
    slope, intercept = np.polyfit(xs, ys, 1)
    dashed = [d for d in drawings if d["type"] == "s" and d["dashes"] not in (None, "[] 0")]
    regression = [d for d in dashed if len(d["items"]) > 1]
    null = [d for d in dashed if d["rect"].height == 0]
    if len(regression) != 1 or len(null) != 1:
        raise ValueError("Unexpected reference/fit lines")
    regression_residuals = [
        abs(decode(point.y, ymap) - (slope * decode(point.x, xmap) + intercept))
        for item in regression[0]["items"] for point in item[1:]
    ]
    raster = raster_marks(Image.open(png_path), marks, axes)
    result = {
        "status": "mismatch_not_releasable",
        "method": "PDF vector marker/tick extraction plus independent PNG color-component/spine concordance",
        "pdf_sha256": sha(pdf_path), "png_sha256": sha(png_path),
        "geometry_source": source(GEOMETRY),
        "auc_source": source(INVENTORY, "stats"),
        "pdf_page_size_points": list(page.rect), "embedded_pdf_images": len(page.get_images()),
        "pdf_axis_bounds": axes, "x_axis_mapping": xmap, "y_axis_mapping": ymap,
        "points": marks, "png_geometry": raster,
        "source_rounding_policy": "Purity tolerance 0.050001 percentage points (one decimal); AUC tolerance 0.000000501 (six decimals).",
        "pdf_text": page.get_text(),
        "regression": {"slope_from_extracted_points": float(slope),
                       "intercept_from_extracted_points": float(intercept),
                       "max_abs_pdf_line_residual_auc": max(regression_residuals)},
        "null_reference_auc": decode(null[0]["rect"].y0, ymap),
        "limitations": "Checks original figure against retained summary artifacts, not a rerun of masks or prediction analysis. The historical AUCs mix fp16 and fp32 and the purity coordinates are stale. No whole-raster label verification or scientific-mechanism certification.",
    }
    doc.close()
    return result


def mask_annotation_checks():
    ocr = json.loads((HERE / "fig_masking_policies.ocr.json").read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text())
    lines = ocr["lines"]
    headings = [line for line in lines if line["text"].startswith("AUC ")]
    aliases = {"random": "random", "oracle": "oracle", "envelope": "envelope",
               "anatomy-vl": "anatomy-v1", "anatomy-v2": "anatomy-v2",
               "cover-f021": "cover-f021"}
    if len(headings) != 6:
        raise ValueError("Expected six OCR AUC headings")

    def center(line):
        return (min(w["x"] for w in line["words"]) +
                max(w["x"] + w["width"] for w in line["words"])) / 2

    checks = []
    for ocr_name, arm in aliases.items():
        names = [line for line in lines if line["text"] == ocr_name]
        if len(names) != 1:
            raise ValueError("Ambiguous mask-grid arm heading")
        heading = min(headings, key=lambda line: abs(center(line) - center(names[0])))
        if abs(center(heading) - center(names[0])) > 3:
            raise ValueError("Mask-grid heading and AUC are misaligned")
        match = re.fullmatch(r"AUC (\d\.\d+) \(ep([0-9lOI]+)\)", heading["text"])
        if not match:
            raise ValueError("Unrecognized OCR AUC/epoch text")
        epoch = int(match[2].translate(str.maketrans({"l": "1", "I": "1", "O": "0"})))
        precision = "fp16" if arm in ("random", "oracle", "envelope") else "fp32"
        records = [(i, r) for i, r in enumerate(inventory["records"])
                   if r.get("arm") == arm and r.get("epoch") == epoch
                   and r.get("precision") == precision and r.get("family") == "frozen_probe"
                   and r.get("status") == "primary"]
        if len(records) != 1:
            raise ValueError("Ambiguous mask-header source")
        index, record = records[0]
        checks.append({"arm": arm, "epoch": epoch, "precision": precision,
                       "ocr_arm_text": ocr_name, "ocr_annotation_text": heading["text"],
                       "observed_auc_label": match[1], "source_auc": record["auc"],
                       "expected_auc_label": "%.4f" % record["auc"],
                       "label_matches": match[1] == "%.4f" % record["auc"],
                       "source_pointer": "/records/%d/auc" % index,
                       "scope": "OCR-based annotation comparison only; not a mask-pixel or provenance check.",
                       "ocr_normalization": "Only documented l/I-to-1 and O-to-0 epoch confusions; anatomy-vl denotes anatomy-v1."})
    return checks


def inspect():
    manuscript = (PAPER / "main_submission.tex").read_text(encoding="utf-8")
    assets = []
    for name in NAMES:
        path = PAPER / "figures" / (name + ".png")
        caption = figure_context(manuscript, str(path))
        ocr_path = HERE / (name + ".ocr.json")
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        if ocr["image_sha256"] != sha(path):
            raise ValueError("OCR asset changed: " + name)
        with Image.open(path) as image:
            metadata = {"width": image.width, "height": image.height, "mode": image.mode,
                        "dpi": image.info.get("dpi"), "software": image.info.get("Software")}
        assets.append({
            "path": str(path.relative_to(PAPER)), "sha256": sha(path),
            "caption": caption, "caption_sha256": digest_text(caption),
            "metadata": metadata, "ocr_source": source(ocr_path),
            "ocr_text": [row["text"] for row in ocr["lines"]],
            "reviewer": "GitHub Copilot legacy-figure evidence review, 2026-09-04",
            "decision": "blocked_no_figure_receipt",
            "mathematically_verified": False,
        })
    sources = {
        "legacy_table2_geometry_seed42": source(GEOMETRY),
        "legacy_auc_inventory": source(INVENTORY, "stats"),
        "legacy_precision_pdf": source(PAPER / "figures" / "fig_precision_paradox.pdf", "paper"),
    }
    for name in NAMES:
        sources["legacy_" + name + "_ocr"] = source(HERE / (name + ".ocr.json"))
        if name.startswith("interp_"):
            original = REPO / "results" / "summary" / (name.removeprefix("interp_") + ".png")
            sources["legacy_summary_" + name] = source(original)
    sources["legacy_summary_heatmap_grid_BC"] = source(REPO / "results" / "summary" / "heatmap_grid_BC.png")
    sources["legacy_heatmap_grid_BC_ocr"] = source(HERE / "heatmap_grid_BC.ocr.json")
    for name in ("interpretability", "assemble_heatmap_grid", "heatmap_BC_comparison",
                 "odos_mirror_test", "deeper_interpretability_analysis"):
        sources["legacy_producer_" + name] = source(REPO / "scripts" / (name + ".py"))
    sources["legacy_interpretability_history"] = source(REPO / "docs" / "experiments" / "interpretability.md")
    sources["legacy_inspection_helper"] = source(Path(__file__).resolve())
    sources["legacy_review_decisions"] = source(HERE / "review_decisions.json")
    decisions = json.loads((HERE / "review_decisions.json").read_text(encoding="utf-8"))
    for asset in assets:
        name = Path(asset["path"]).stem
        asset.update(decisions[name])
        asset["inputs"] = [
            {"source": key,
             "pointer": "/sha256" if Path(sources[key]["path"]).suffix.lower() in (".png", ".pdf") else "",
             "sha256": sources[key]["sha256"]}
            for key in decisions[name]["source_keys"]
        ]
        if name.startswith("interp_"):
            asset["identical_to_retained_summary_png"] = (
                sources["legacy_summary_" + name]["sha256"] == asset["sha256"])
    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "review_scope": "Seven fixed legacy assets; no manuscript, production, figure, GPU, or network changes.",
        "classification_rule": "Image dimensions and OCR transcripts are metadata/observations, not empirical source truth. Lines, scatter points, AUCs, correlations, accuracies, predictions, and attributions remain quantitative.",
        "sources": sources, "assets": assets, "precision": precision_audit(),
        "mask_grid_auc_annotation_checks": mask_annotation_checks(),
        "bounded_input_search": {
            "searched": ["repository archive subtree for named attribution arrays/images",
                         "repository PDF companions", "documented results and results_presentation locations"],
            "found_interpretability_vector_companions_or_arrays": False,
            "not_searched": "No broad search of D: or user profile; no blob downloads or case exports.",
        },
    }
    dump("inspection.json", audit)
    candidate = {
        "version": 1,
        "scope": "Legacy figure evidence candidate: no approvals. Original seven assets remain blocked; see inspection.json and REVIEW.md. Sources pin observations and retained summaries; they must not be treated as extracted empirical truth merely because their hashes match.",
        "sources": sources,
        "figures": [],
    }
    import jsonschema
    schema = json.loads((REPO / "autopilot" / "numeric_reviews.schema.json").read_text())
    jsonschema.validate(candidate, schema)
    dump("candidate_numeric_reviews.json", candidate)
    print(json.dumps({"assets_inspected": len(assets), "approved_figures": 0,
                      "stale_purity_points": sum(not x["purity_matches_rounding_to_1_decimal"] for x in audit["precision"]["points"]),
                      "png_pdf_marker_concordance": audit["precision"]["png_geometry"]["all_marker_centers_concordant"],
                      "candidate_schema_valid": True}))


def merge_reviews(base, candidate):
    result = dict(base)
    if base.get("version") != 1 or candidate.get("version") != 1:
        raise ValueError("Only version-one reviews can be merged")
    for key in ("sources", "macros"):
        old, new = base.get(key, {}), candidate.get(key, {})
        duplicates = set(old) & set(new)
        if duplicates:
            raise ValueError("Duplicate %s keys: %s" % (key, sorted(duplicates)))
        result[key] = {**old, **new}
    for key in ("figures", "literals"):
        combined = list(base.get(key, [])) + list(candidate.get(key, []))
        identities = [
            row["path"].replace("\\", "/").casefold() if key == "figures"
            else (row["file"].replace("\\", "/").casefold(), row["context_sha256"], row["token_index"])
            for row in combined
        ]
        if any(n > 1 for n in Counter(identities).values()):
            raise ValueError("Duplicate %s receipts; refusing overwrite" % key)
        result[key] = combined
    result["scope"] = "\n".join(x for x in (base.get("scope"), candidate.get("scope")) if x)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-base", type=Path)
    parser.add_argument("--merge-candidate", type=Path, default=HERE / "candidate_numeric_reviews.json")
    parser.add_argument("--merge-output", default="merged_numeric_reviews.json")
    args = parser.parse_args()
    if args.merge_base:
        if Path(args.merge_output).name != args.merge_output:
            raise ValueError("Merged output must be a filename within the owned evidence directory")
        if (HERE / args.merge_output).exists():
            raise FileExistsError("Refusing to overwrite existing merged output")
        base = json.loads(args.merge_base.read_text(encoding="utf-8-sig"))
        candidate = json.loads(args.merge_candidate.read_text(encoding="utf-8-sig"))
        merged = merge_reviews(base, candidate)
        import jsonschema
        jsonschema.validate(merged, json.loads((REPO / "autopilot" / "numeric_reviews.schema.json").read_text()))
        print(dump(args.merge_output, merged))
    else:
        inspect()


if __name__ == "__main__":
    main()
