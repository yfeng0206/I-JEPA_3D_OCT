"""I13 registration of the camera-ready v2 figures (idempotent, fail-closed).

1. Source snapshots: the src/config files whose constants the new captions and the
   appendix paragraph state are copied byte-exactly from git (SNAP_REV) into
   figures_v2/source_snapshots/<rev>/..., after checking the work tree still matches
   (receipts then survive later src edits, as I8's paper_i8/source_snapshots do).
2. autopilot/release_assets.json: producer entries for figures/fig_oct_strategies_v2.png
   and figures/fig_oct_jepa_volume_v2.png (script hash, inputs, evidence manifest + hash,
   expected PNG hash, PDF/SVG companions), inserted surgically; the rest of the
   hand-formatted file stays byte-identical. Old producer entries are kept (the poster
   still uses the old PDFs).
3. paper/genai4health2026/numeric_reviews.json: the three retired figure receipts are
   replaced by two reviewed_source_illustration receipts (PNG hash + caption hash), and
   the literal receipts of every paragraph this change touched are re-keyed to the new
   context hashes (same value, same evidence) or added (new literals, protocol reviews
   against hash-pinned sources). Each token is checked by value and position first.

Run: python register_i13.py [--check]   (--check: verify only, write nothing)

Re-run after any edit of the two captions, the Data/Evaluation paragraphs or the appendix
paragraph (and after regenerating a figure): re-keyed receipts are always taken from the
pre-I13 review file (git PRE_I13_REV), new ones are rebuilt from NEW_LITERALS, and receipts
this script wrote before (register_i13_state.json) are replaced, so p15 sees no stale entry.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "autopilot"))
import numeric_bindings as numeric  # noqa: E402
import release_assets as assets  # noqa: E402

PAPER = REPO / "paper" / "genai4health2026"
REVIEWS = PAPER / "numeric_reviews.json"
REGISTRY = REPO / "autopilot" / "release_assets.json"
REL_V2 = "autopilot/investigations/camera_ready_20261008/figures_v2"
SNAP_REV = "110bd18"
PRE_I13_REV = "110bd18"   # review file before this change (unchanged in becc645)
STATE = HERE / "register_i13_state.json"
SNAP_FILES = ["src/datasets/oct_slices.py", "src/datasets/oct_volumes.py", "src/helper.py",
              "configs/patch_vitb16_ep100.yaml", "configs/patch_oracle_anatomical.yaml"]
REVIEWER = "GitHub Copilot I13 camera-ready figure and source review, 2026-10-11"
REKEY_NOTE = "; context re-reviewed after the v2 figure redesign (GitHub Copilot I13, 2026-10-11)"

FIGURES = {
    "strategies": {"png": "figures/fig_oct_strategies_v2.png", "stem": "fig_oct_strategies_v2",
                   "script": "generate_strategies_v2.py", "arguments": ["--replay"]},
    "volume": {"png": "figures/fig_oct_jepa_volume_v2.png", "stem": "fig_oct_jepa_volume_v2",
               "script": "generate_framework_v2.py", "arguments": []},
}
RETIRED_FIGURES = ["figures/fig_oct_target_selection_cr.png", "figures/fig_oct_jepa_comparison_final_cr.png",
                   "figures/fig_oct_all_strategies.png"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snap_rel(rel):
    return "%s/source_snapshots/%s/%s" % (REL_V2, SNAP_REV, rel)


def snapshots(write):
    out = {}
    for rel in SNAP_FILES:
        blob = subprocess.check_output(["git", "show", "%s:%s" % (SNAP_REV, rel)], cwd=REPO)
        work = (REPO / rel).read_bytes()
        if blob.replace(b"\r\n", b"\n") != work.replace(b"\r\n", b"\n"):
            raise SystemExit("work tree %s differs from %s; refusing to snapshot" % (rel, SNAP_REV))
        target = REPO / snap_rel(rel)
        if target.exists() and target.read_bytes() != blob:
            raise SystemExit("existing snapshot differs: %s" % target)
        if write and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        out[rel] = hashlib.sha256(blob).hexdigest()
    return out


def win(rel):
    return rel.replace("/", "\\")


def new_sources(snap_hashes):
    names = {"src/datasets/oct_slices.py": "i13_snap_oct_slices", "src/datasets/oct_volumes.py": "i13_snap_oct_volumes",
             "src/helper.py": "i13_snap_helper", "configs/patch_vitb16_ep100.yaml": "i13_snap_cfg_uniform",
             "configs/patch_oracle_anatomical.yaml": "i13_snap_cfg_centroid"}
    sources = {names[rel]: {"root": "repo", "path": win(snap_rel(rel)), "sha256": digest}
               for rel, digest in snap_hashes.items()}
    for key, name in (("i13_fig_strategies_evidence", "fig_oct_strategies_v2.evidence.json"),
                      ("i13_fig_volume_evidence", "fig_oct_jepa_volume_v2.evidence.json"),
                      ("i13_fig_strategies_generator", "generate_strategies_v2.py"),
                      ("i13_fig_volume_generator", "generate_framework_v2.py"),
                      ("i13_figstyle", "figstyle_v2.py")):
        sources[key] = {"root": "repo", "path": win("%s/%s" % (REL_V2, name)), "sha256": sha(HERE / name)}
    return sources


def protocol(rationale, *evidence):
    return {"review": {"kind": "protocol", "reviewer": REVIEWER, "rationale": rationale, "evidence": list(evidence)}}


EV_200 = {"source": "i13_snap_oct_slices", "excerpt": "(200, 200, 200) (uint8)"}
EV_LINSPACE = {"source": "i13_snap_oct_slices", "excerpt": "0, 199, num=num_slices, dtype=np.int64"}
EV_RESIZE = {"source": "i13_snap_oct_slices", "excerpt": "(self.slice_size, self.slice_size), Image.BILINEAR"}
EV_SLICES = {"source": "lr_primary_result", "pointer": "/config/data/num_slices"}
EV_SLICES_PRE = {"source": "i13_snap_cfg_uniform", "excerpt": "num_slices: 100"}
EV_SIZE = {"source": "lr_primary_result", "pointer": "/config/data/slice_size"}
EV_CROP_PRE = {"source": "i13_snap_cfg_uniform", "excerpt": "crop_size: 256"}
EV_PATCH = {"source": "lr_primary_result", "pointer": "/config/model/patch_size"}
EV_PATCH_PRE = {"source": "i13_snap_cfg_uniform", "excerpt": "patch_size: 16"}
EV_768 = {"source": "i13_snap_helper", "excerpt": "'vit_base':  dict(embed_dim=768,  depth=12, num_heads=12),"}

R_200_SCANS = protocol("FairVision volumes are stored as oct_bscans arrays of shape (200, 200, 200); the first "
                       "axis indexes the 200 B-scans of a volume (loader docstring; one Training file was "
                       "opened read-only on 2026-10-11 and has this shape).", EV_200)
R_200_PX = protocol("Each stored B-scan is 200 by 200 pixels (last two axes of the (200, 200, 200) array) "
                    "before the bilinear resize to slice_size.", EV_200, EV_RESIZE)
R_100 = protocol("Pretraining and evaluation configurations both use num_slices 100: 100 B-scans per volume.",
                 EV_SLICES, EV_SLICES_PRE)
R_256 = protocol("Every selected B-scan is resized bilinearly to slice_size 256 (evaluation) / crop_size 256 "
                 "(pretraining view) pixels.", EV_SIZE, EV_CROP_PRE, EV_RESIZE)
R_TOKENS = protocol("A 256 by 256 input with patch size 16 gives a 16 by 16 grid, i.e. 256 patch tokens.",
                    EV_SIZE, EV_PATCH, EV_PATCH_PRE)
R_16 = protocol("Patch size 16 pixels (ViT-Base/16), in both the probe and the pretraining configuration.",
                EV_PATCH, EV_PATCH_PRE)
R_INDEX = protocol("Both loaders select np.linspace(0, 199, num_slices) cast to int64; with num_slices 100 this "
                   "is floor(199 k / 99) for k = 0..99, i.e. 0, 2, ..., 196, 199 (100 distinct indices; "
                   "checked numerically by generate_framework_v2.slice_indices).", EV_LINSPACE, EV_SLICES)
R_768 = protocol("ViT-Base feature dimension 768: each B-scan vector and the volume vector are 768-dimensional.",
                 EV_768)

# (context marker, token index, value, receipt) for literals that are new in this change.
NEW_LITERALS = [
    ("main_submission.tex", r"\paragraph{Data.}", 4, "200", R_200_SCANS),
    ("main_submission.tex", "fig_oct_jepa_volume_v2.png", 1, "200", R_200_SCANS),
    ("main_submission.tex", "fig_oct_jepa_volume_v2.png", 2, "100", R_100),
    ("main_submission.tex", "fig_oct_jepa_volume_v2.png", 3, "256", R_256),
    ("main_submission.tex", "fig_oct_jepa_volume_v2.png", 4, "256", R_256),
    ("main_submission.tex", "fig_oct_jepa_volume_v2.png", 5, "256", R_TOKENS),
    ("main_submission.tex", "fig_oct_jepa_volume_v2.png", 6, "16", R_16),
    ("main_submission.tex", "fig_oct_jepa_volume_v2.png", 7, "16", R_16),
    ("compact_protocol.tex", r"\paragraph{B-scan selection and pooling.}", 0, "200", R_200_SCANS),
    ("compact_protocol.tex", r"\paragraph{B-scan selection and pooling.}", 1, "200", R_200_PX),
    ("compact_protocol.tex", r"\paragraph{B-scan selection and pooling.}", 2, "200", R_200_PX),
    ("compact_protocol.tex", r"\paragraph{B-scan selection and pooling.}", 3, "100", R_100),
] + [("compact_protocol.tex", r"\paragraph{B-scan selection and pooling.}", i, v, R_INDEX)
     for i, v in ((4, "199"), (5, "99"), (6, "0"), (7, "99"), (8, "0"), (9, "2"), (10, "196"), (11, "199"))] + [
    ("compact_protocol.tex", r"\paragraph{B-scan selection and pooling.}", 12, "256", R_256),
    ("compact_protocol.tex", r"\paragraph{B-scan selection and pooling.}", 13, "256", R_256),
    ("compact_protocol.tex", r"\paragraph{B-scan selection and pooling.}", 14, "256", R_TOKENS),
    ("compact_protocol.tex", r"\paragraph{B-scan selection and pooling.}", 15, "100", R_100),
    ("compact_protocol.tex", r"\paragraph{B-scan selection and pooling.}", 16, "768", R_768),
]

# (old context sha prefix, old index) -> (context marker, new index): same value, same evidence.
REKEY = [
    ("a3c634619446", 0, "fig_oct_strategies_v2.png", 1), ("a3c634619446", 1, "fig_oct_strategies_v2.png", 2),
    ("a3c634619446", 2, "fig_oct_strategies_v2.png", 3), ("a3c634619446", 3, "fig_oct_strategies_v2.png", 4),
    ("a3c634619446", 4, "fig_oct_strategies_v2.png", 5), ("a3c634619446", 5, "fig_oct_strategies_v2.png", 6),
    ("a3c634619446", 6, "fig_oct_strategies_v2.png", 7),
    ("be8cd6099684", 1, r"\paragraph{Data.}", 1), ("be8cd6099684", 2, r"\paragraph{Data.}", 2),
    ("be8cd6099684", 3, r"\paragraph{Data.}", 3), ("be8cd6099684", 4, r"\paragraph{Data.}", 5),
    ("be8cd6099684", 5, r"\paragraph{Data.}", 6),
    ("bfb8099a8747", 0, r"\paragraph{Evaluation.}", 0),
    ("398bde8b3745", 0, "All policies request", 0),
    ("11fa2f455fb1", 0, "fig_oct_jepa_volume_v2.png", 8),
]


def literal_rows(rel):
    source = (PAPER / rel).read_text(encoding="utf-8")
    return list(numeric.literals(source, rel))


def find(rows, marker, index):
    hits = {r["context_sha256"] for r in rows if marker in r["context"]}
    if len(hits) != 1:
        raise SystemExit("context marker %r matches %d contexts" % (marker, len(hits)))
    context = hits.pop()
    row = [r for r in rows if r["context_sha256"] == context and r["token_index"] == index]
    if len(row) != 1:
        raise SystemExit("token %d missing in context %r" % (index, marker))
    return row[0]


def figure_receipt(key, png_rel, caption, ev):
    staged = "autopilot/investigations/camera_ready_20261008/figures_staging/fig_oct_all_strategies.evidence.json"
    common_tail = [{"source": "external_oct_source_image", "pointer": "/sha256"},
                   {"source": "external_oct_license", "pointer": "/"},
                   {"source": "i13_figstyle", "pointer": "/"}]
    if key == "strategies":
        inputs = [{"source": "i13_fig_strategies_evidence", "pointer": "/plotted/masks"},
                  {"source": "i13_fig_strategies_evidence", "pointer": "/plotted/guide_overlays"},
                  {"source": "i13_fig_strategies_evidence", "pointer": "/replay"},
                  {"source": "i13_fig_strategies_evidence", "pointer": "/guide_products"},
                  {"source": "cr_figall_evidence", "pointer": "/masks"},
                  {"source": "cr_external_guides", "pointer": "/"},
                  {"source": "i13_fig_strategies_generator", "pointer": "/"}] + common_tail
        limitations = (
            "Attributed CC BY 4.0 OCTDL Figure 1 crop (arXiv:2312.08255v3), not FairVision. A single fixed-seed "
            "replay for one image with matched block sizes (batch size 1: no additional cross-image shortening; "
            "target-group collation still applies, so each rectangle group is truncated to 40 indices and "
            "ANATOMY-V2 groups are resampled to 16); masks are the "
            "staged camera-ready replay indices (" + staged + "), re-derived from the samplers pinned at commit "
            "86ae882 by the producer's --replay check. MIRAGE envelope and soft scores were produced on this "
            "crop by the training guide pipelines; MIRAGE was not trained on OCTDL, so they are not a validated "
            "segmentation. Not a segmentation annotation, historical training batch, evaluation sample, cohort "
            "statistic or performance comparison. Hash and evidence checks are not mathematical verification of "
            "every rendered pixel.")
        checks = [
            "Final PNG rendered and inspected at paper scale: two rows (guide A_s(x); I-JEPA view with context, "
            "four Okabe-Ito target groups each also identified by a numeral 1-4 with white halo at the top-left "
            "2x2 square of its own cells, grey hidden-not-target cells, repeated-index dots left uncovered) over "
            "RANDOM, CENTROID, ENVELOPE, ANATOMY-V2 and COVER; a target-rule line per column whose words equal "
            "the target-shape and placement cells of Table tab:strategies (checked by the producer); legend with "
            "numbered swatches and attribution line; no identifiers or performance readouts. Layout guard "
            "(producer assertion, lane B): all 61 drawn texts, numerals and legend digits included, pairwise "
            "collision-free and inside the canvas; guard self-test regression cases recorded in the evidence. "
            "Smallest PDF text span 6.08 pt (math subscripts at 0.8 x base; ordinary labels >= 6.5 pt).",
            "Soft guides drawn for every cell with score > 0 (opacity 0.8 x min(score, 1)); the first export's "
            "0.02 display cutoff (23 cells, 0.24% of the summed score) was removed and is recorded in the "
            "evidence (plotted.display_transformations).",
            "Producer --replay: masks recomputed from the pinned sampler sources equal the staged indices "
            "(masks_equal_staged true) and the CENTROID guide equals the staged guide.",
            "Guide grids rebuilt through GuidedOCTSliceDataset hash-match the staged replay "
            "(grid_sha256_equal_staged true).",
            "PNG/PDF/SVG hashes equal the evidence outputs and the release_assets.json pins; generator and "
            "style-module hashes equal the evidence.",
            "Caption states the external crop, the trained configurations, MIRAGE not trained on OCTDL, and "
            "disclaims segmentation, historical batch, evaluation sample and performance use."]
    else:
        inputs = [{"source": "i13_fig_volume_evidence", "pointer": "/plotted/a_volume"},
                  {"source": "i13_fig_volume_evidence", "pointer": "/plotted/b_pretraining"},
                  {"source": "i13_fig_volume_evidence", "pointer": "/plotted/c_evaluation"},
                  {"source": "i13_fig_volume_evidence", "pointer": "/source_facts"},
                  {"source": "cr_figall_evidence", "pointer": "/masks/centroid"},
                  {"source": "i13_snap_oct_slices", "pointer": "/"},
                  {"source": "i13_snap_oct_volumes", "pointer": "/"},
                  {"source": "i13_snap_helper", "pointer": "/"},
                  {"source": "i13_fig_volume_generator", "pointer": "/"}] + common_tail
        limitations = (
            "Schematic of the implemented protocol, drawn from source facts recorded with file, line and sha256 "
            "in the evidence (loaders, transforms, train_patch.jepa_forward_loss, eval_downstream pooling and "
            "LinearHead, registered probe config). Volume stacks are drawn schematics, not data; the example "
            "B-scan and CENTROID masks are the attributed CC BY 4.0 OCTDL crop and staged replay indices of the "
            "strategy figure. Encoder shapes and token/vector icons are illustrative, not measured features. "
            "No cohort statistic, result or performance claim.")
        checks = [
            "V12 typography revision and author polish: three full-width bands; every label >= 6.6 pt and math "
            ">= 7.6 pt at the 5.5 in print width (single-level sub/superscripts drawn at 0.8 x base, >= 6.08 pt, "
            "none nested; measured from the PDF, evidence typography); target identity carried by numbered "
            "tokens 1-4; pooling steps drawn as operator boxes on the arrows. Layout guard (producer assertion, "
            "lane B): all 69 drawn texts, token digits included, pairwise collision-free and inside the canvas; "
            "guard self-test regression cases recorded in the evidence.",
            "Final PNG rendered and inspected at paper scale: (a) 200 B-scans with alternate slices marked "
            "used, indices 0, 2, ..., 196, 199, resize to 256 x 256, 16 x 16 patch grid (N = 256); (b) view x, "
            "guide, sampler, context view, f_theta, g_phi with mask tokens at T positions, EMA f_theta_bar on the "
            "full view, LayerNorm and select T_m with stop-gradient, Smooth-L1 per T_m; (c) frozen EMA encoder, "
            "256 x 768 tokens, patch mean, 100 x 768, B-scan mean, 768, LayerNorm + linear, sigmoid.",
            "Every drawn protocol fact is located in the evidence source_facts (path, line, sha256) and was "
            "checked against the code: oct_slices.py/oct_volumes.py linspace(0, 199, 100) and bilinear resize, "
            "transforms.RandomResizedCrop, train_patch.jepa_forward_loss, vision_transformer context masking, "
            "helper vit_base 768, eval_downstream target_encoder/patch mean/LinearHead/sigmoid, MeanPool.",
            "PNG/PDF/SVG hashes equal the evidence outputs and the release_assets.json pins; generator and "
            "style-module hashes equal the evidence.",
            "Caption states the B-scan selection, per-B-scan pretraining and the pooling path; numbers in the "
            "caption carry literal receipts."]
    resolved = []
    for item in inputs:
        item = dict(item)
        item["sha256"] = ev.hashes[item["source"]] if item["source"] in ev.hashes else None
        resolved.append(item)
    return {"path": png_rel, "sha256": sha(PAPER / png_rel), "caption_sha256": numeric.digest_text(caption),
            "inputs": resolved,
            "validation": {"method": "reviewed_source_illustration", "reviewer": REVIEWER,
                           "quantitative_scope": "illustrative_only", "limitations": limitations,
                           "review_checks": checks}}


def producer_entry(key, outputs_dir=HERE):
    fig = FIGURES[key]
    evidence_path = HERE / (fig["stem"] + ".evidence.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    paper_out = evidence["outputs"]["paper"]
    for suffix in ("png", "pdf", "svg"):
        produced = HERE / paper_out[suffix]["file"]
        shipped = PAPER / "figures" / paper_out[suffix]["file"]
        if sha(produced) != paper_out[suffix]["sha256"] or sha(shipped) != paper_out[suffix]["sha256"]:
            raise SystemExit("%s: output %s differs from its evidence" % (key, suffix))
    if evidence["generator_sha256"] != sha(HERE / fig["script"]):
        raise SystemExit("%s: generator changed after export; regenerate" % key)
    if evidence["style_module_sha256"] != sha(HERE / "figstyle_v2.py"):
        raise SystemExit("%s: style module changed after export; regenerate" % key)
    common = [REL_V2 + "/figstyle_v2.py",
              "autopilot/investigations/camera_ready_20261008/figures_staging/fig_oct_all_strategies.evidence.json",
              "autopilot/investigations/submission_refresh_20260910/figure/octdl_figure1_original.png",
              "autopilot/investigations/submission_refresh_20260910/figure/octdl_primary_license.html"]
    if key == "strategies":
        inputs = common + [
            "autopilot/investigations/camera_ready_20261008/figures_staging/external_guides/octdl_crop_guides.npz",
            "autopilot/investigations/camera_ready_20261008/figures_staging/external_guides/octdl_crop_guides.json",
            "src/datasets/oct_slices_guided.py", "scripts/mask_composition_probe.py",
            "git:86ae8823f3afc9fa2c9ed6a3cb0c459b513ba1c5:src/masks (pinned sampler sources, --replay)",
            "configs/patch_oracle_anatomical.yaml", "configs/patch_mirage_envelope.yaml",
            "configs/patch_anatomy_v2.yaml", "configs/patch_cover_f021_ep25.yaml",
            "paper/genai4health2026/main_submission.tex"]
        scope = ("All five strategies on the attributed CC BY 4.0 OCTDL crop, not FairVision: row (a) each placement "
                 "guide A_s(x) on the patch grid (none, CENTROID band, hard MIRAGE envelope cells, summed soft "
                 "MIRAGE scores); row (b) the delivered context C, target groups T1-T4 in Okabe-Ito colours, "
                 "hidden-not-target cells and repeated indices. Exact staged replay indices (trained sampler "
                 "configurations, block-size seed 3107, seed 42, batch size 1), re-derived by --replay from the "
                 "samplers pinned at 86ae882; guide grids rebuilt through GuidedOCTSliceDataset and hash-matched. "
                 "Target-rule words equal the tab:strategies cells of the declared manuscript input (LF-normalised "
                 "table-block SHA-256 in the evidence). "
                 "Replaces the target-selection and all-strategies figures in the paper. No cohort statistic or "
                 "performance claim.")
        policy = "camera-ready v2 strategy figure (I13); changed outputs or caption require renewed review"
    else:
        inputs = common + ["src/datasets/oct_slices.py", "src/datasets/oct_volumes.py", "src/transforms.py",
                           "src/train_patch.py", "src/models/vision_transformer.py", "src/helper.py",
                           "src/eval_downstream.py", "src/models/attentive_pool_minimal.py",
                           "configs/patch_vitb16_ep100.yaml", "configs/patch_oracle_anatomical.yaml",
                           "configs/downstream_cr_seed_v1.yaml"]
        scope = ("Schematic I-JEPA framework with 3-D volume handling: 200 B-scans per volume, the 100 used "
                 "(linspace indices 0, 2, ..., 196, 199), resize to 256 x 256 and 16 x 16 patches (256 tokens); "
                 "per-B-scan pretraining with uniform or guided sampler, context encoder, predictor with mask "
                 "tokens, EMA target encoder, LayerNorm target selection with stop-gradient and Smooth-L1; frozen "
                 "EMA evaluation with patch mean per B-scan, mean over B-scans and LayerNorm-linear head. Facts "
                 "recorded with source lines and hashes; CENTROID masks from the staged replay on the OCTDL crop; "
                 "volume stacks schematic. Replaces the uniform-versus-guided comparison figure in the paper. "
                 "No measured feature or performance claim.")
        policy = "camera-ready v2 framework/volume figure (I13); changed outputs or caption require renewed review"
    entry = {
        "script": REL_V2 + "/" + fig["script"],
        "script_sha256_at_export": evidence["generator_sha256"],
    }
    if fig["arguments"]:
        entry["arguments"] = fig["arguments"]
    entry.update({
        "inputs": inputs,
        "evidence_manifest": REL_V2 + "/" + evidence_path.name,
        "evidence_manifest_sha256": sha(evidence_path),
        "expected_output_sha256": paper_out["png"]["sha256"],
        "refresh_policy": policy,
        "evidence_scope": scope,
        "license": "CC BY 4.0",
        "license_source": "https://arxiv.org/abs/2312.08255v3",
        "rendered_metrics": [],
        "raw_oct_pixels_rendered": True,
        "clinical_image_source": "Publicly published external OCTDL figure crop; no private FairVision image",
        "subject_identifiers_rendered": False,
        "poster_variant": {REL_V2 + "/poster/" + evidence["outputs"]["poster"][s]["file"]:
                           evidence["outputs"]["poster"][s]["sha256"] for s in ("png", "pdf", "svg")},
        "companions": {"figures/" + paper_out[s]["file"]: paper_out[s]["sha256"] for s in ("pdf", "svg")},
    })
    return entry


def entry_text(key, entry):
    """Registry style: one key per line, short lists inline, indent 4/6."""
    lines = ['    "%s": {' % key]
    items = list(entry.items())
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if isinstance(v, list) and len(json.dumps(v)) > 90:
            body = ",\n".join("        " + json.dumps(x, ensure_ascii=False) for x in v)
            lines.append('      "%s": [\n%s\n      ]%s' % (k, body, comma))
        elif isinstance(v, dict) and v:
            body = ",\n".join("        %s: %s" % (json.dumps(a), json.dumps(b, ensure_ascii=False))
                              for a, b in v.items())
            lines.append('      "%s": {\n%s\n      }%s' % (k, body, comma))
        else:
            lines.append('      "%s": %s%s' % (k, json.dumps(v, ensure_ascii=False), comma))
    lines.append("    }")
    return "\n".join(lines)


def update_registry(write):
    raw = REGISTRY.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    data = json.loads(text)
    expected = json.loads(text)
    entries = {FIGURES[k]["png"]: producer_entry(k) for k in FIGURES}
    for key, entry in entries.items():
        expected["producers"][key] = entry
        if key in data["producers"]:
            # replace an existing entry block (idempotent re-run)
            start = text.index('    "%s": {' % key)
            end = text.index("\n    }", start) + len("\n    }")
            text = text[:start] + entry_text(key, entry) + text[end:]
        else:
            anchor = '\n  },\n  "fixed_inputs"'
            if text.count(anchor) != 1:
                raise SystemExit("registry: producers block end not found once")
            text = text.replace(anchor, ",\n" + entry_text(key, entry) + anchor)
    if json.loads(text) != expected:
        raise SystemExit("registry: surgical edit does not reproduce the intended JSON")
    out = text.replace("\n", "\r\n") if crlf else text
    changed = out != raw
    if write and changed:
        REGISTRY.write_bytes(out.encode("utf-8"))
    return changed


def update_reviews(write, snap_hashes):
    raw = REVIEWS.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    # Receipts re-keyed by this script come from the pre-I13 review file (fixed revision), so a
    # re-run after a caption edit still finds them; keys this script wrote are tracked in STATE.
    pristine = json.loads(subprocess.check_output(
        ["git", "show", "%s:paper/genai4health2026/numeric_reviews.json" % PRE_I13_REV], cwd=REPO).decode("utf-8-sig"))
    pristine_index = {(e["file"], e["context_sha256"][:12], e["token_index"]): e for e in pristine["literals"]}
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {"written": []}
    written_before = {tuple(k) for k in state["written"]}
    sources = new_sources(snap_hashes)
    for name, spec in sources.items():
        if name in data["sources"] and data["sources"][name] != spec:
            print("updating source", name)
        data["sources"][name] = spec
    ev = numeric.Evidence(PAPER, REPO, sources={**data["sources"]})
    for name, spec in sources.items():
        target = REPO / spec["path"].replace("\\", "/")
        if not target.exists() and name.startswith("i13_snap_"):
            rel = spec["path"].replace("\\", "/").split("/%s/" % SNAP_REV, 1)[1]
            blob = subprocess.check_output(["git", "show", "%s:%s" % (SNAP_REV, rel)], cwd=REPO)
            ev.cache[name], ev.hashes[name], ev.paths[name] = blob.decode("utf-8-sig"), spec["sha256"], target
            continue
        ev.load(name)
    for name in ("cr_figall_evidence", "cr_external_guides", "external_oct_source_image", "external_oct_license"):
        ev.load(name)
    rows = {rel: literal_rows(rel) for rel in ("main_submission.tex", "compact_protocol.tex")}
    current_keys = {(rel, r["context_sha256"], r["token_index"]) for rel, rs in rows.items() for r in rs}
    rekey_prefixes = {rk[0] for rk in REKEY}

    def managed(entry):
        review = entry.get("review") or {}
        return ((entry["file"], entry["context_sha256"], entry["token_index"]) in written_before
                or "I13" in review.get("reviewer", "")
                or (entry["file"] == "main_submission.tex" and entry["context_sha256"][:12] in rekey_prefixes))

    keep = []
    for entry in data["literals"]:
        key = (entry["file"], entry["context_sha256"], entry["token_index"])
        if entry["file"] not in rows or key in current_keys:
            keep.append(entry)
        elif not managed(entry):
            raise SystemExit("unexpected stale literal receipt %s" % (key,))
    present = {(e["file"], e["context_sha256"], e["token_index"]) for e in keep}
    additions, written = [], []
    for prefix, index, marker, new_index in REKEY:
        row = find(rows["main_submission.tex"], marker, new_index)
        key = ("main_submission.tex", row["context_sha256"], new_index)
        written.append(list(key))
        if key in present:
            continue
        source_entry = pristine_index.get(("main_submission.tex", prefix, index))
        if source_entry is None:
            raise SystemExit("missing pre-I13 receipt for %s:%d" % (prefix, index))
        if source_entry["value"] != row["value"]:
            raise SystemExit("re-key value mismatch %s:%d -> %r" % (prefix, index, row["value"]))
        new = json.loads(json.dumps(source_entry))
        new["context_sha256"], new["token_index"] = row["context_sha256"], new_index
        if "review" in new and REKEY_NOTE not in new["review"]["reviewer"]:
            new["review"]["reviewer"] += REKEY_NOTE
        additions.append(new)
    for rel, marker, index, value, receipt in NEW_LITERALS:
        row = find(rows[rel], marker, index)
        if row["value"] != value:
            raise SystemExit("%s %r token %d is %r, expected %r" % (rel, marker, index, row["value"], value))
        key = (rel, row["context_sha256"], index)
        written.append(list(key))
        if key in present:
            continue
        additions.append({"file": rel, "context_sha256": row["context_sha256"], "token_index": index,
                          "value": value, **json.loads(json.dumps(receipt))})
    data["literals"] = keep + additions
    text, _ = assets.source_tree(PAPER)
    figures = [f for f in data["figures"] if f["path"] not in RETIRED_FIGURES
               and f["path"] not in {v["png"] for v in FIGURES.values()}]
    for key, fig in FIGURES.items():
        caption = numeric.figure_context(text, fig["png"])
        figures.append(figure_receipt(key, fig["png"], caption, ev))
    data["figures"] = figures
    for f in figures:
        for item in f.get("inputs", []):
            if item.get("sha256") is None:
                ev.load(item["source"])
                item["sha256"] = ev.hashes[item["source"]]
    out = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if b"\r\n" in raw:
        out = out.replace("\n", "\r\n")
    changed = out.encode("utf-8") != raw
    state_text = json.dumps({"pre_i13_review_revision": PRE_I13_REV, "written": sorted(written)}, indent=1) + "\n"
    state_changed = not STATE.is_file() or STATE.read_text(encoding="utf-8") != state_text
    if write and changed:
        REVIEWS.write_bytes(out.encode("utf-8"))
    if write and state_changed:
        STATE.write_text(state_text, encoding="utf-8")
    return changed or state_changed, len(additions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write = not args.check
    snap_hashes = snapshots(write)
    reg = update_registry(write)
    rev, added = update_reviews(write, snap_hashes)
    print(json.dumps({"snapshots": snap_hashes, "registry_changed": reg, "reviews_changed": rev,
                      "literal_receipts_added_or_rekeyed": added, "written": write}, indent=1))
    return 1 if args.check and (reg or rev) else 0


if __name__ == "__main__":
    sys.exit(main())
