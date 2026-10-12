"""Lane-B registration (GitHub Copilot I13, 2026-10-12).

1. Fig. 3 (auto/fig_trajectories_ci.png, produced by autopilot/p8_make_assets.py:draw_trajectories)
   was redrawn at print size. This script replays p8 main() exactly as p15 does (outputs
   captured in a work directory, never written to the paper), requires the delivered PNG bytes,
   runs the independent p15 artist validator (numeric_plot_review.p8_artists) on the captured
   figure, audits every text artist (size at the 1:1 print width, collisions, canvas
   containment), checks that every other P8 output is byte-identical to paper/auto, and writes
   the evidence manifest fig_trajectories_ci.evidence.json next to this script.
2. release_assets.json: producer entry for auto/fig_trajectories_ci.png (script hash, input,
   evidence manifest hash, expected output hash); the archived occlusion figure, whose panels are
   FairVision B-scans (CC BY-NC-ND 4.0: adapted overlays may not be shared), is removed from the
   paper, its producer entry is dropped and its PNG hash is kept under retired_inputs so it
   cannot be silently re-included.
3. numeric_reviews.json: the occlusion figure receipt and the five sources only it used are
   removed; lr_p8_code is re-pinned to the new p8 bytes after checking that every excerpt cited
   from it still occurs; the literal receipts of the edited Fig. 3 caption are re-keyed (same
   evidence) or added (epochs 50/75/100 joined in panel (a); the fp16/fp32 bound 2x10^-4 with
   the assertion of the Table 1 caption).

Idempotent. --check writes nothing and exits 1 if anything would change. Run after editing the
Fig. 3 caption or p8's trajectory code, and before figures_v2/register_i13.py (which refuses
stale receipts it does not manage).
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))

from autopilot import numeric_bindings as numeric  # noqa: E402
from autopilot import numeric_plot_review as plots  # noqa: E402
from autopilot import release_assets as assets  # noqa: E402

PAPER = REPO / "paper" / "genai4health2026"
REVIEWS = PAPER / "numeric_reviews.json"
REGISTRY = REPO / "autopilot" / "release_assets.json"
P8 = REPO / "autopilot" / "p8_make_assets.py"
VALIDATOR = REPO / "autopilot" / "numeric_plot_review.py"
STATS = Path(r"D:\jepa_phase0\autopilot_out\p1_stats")
REL_HERE = "autopilot/investigations/camera_ready_20261008/figures_v2/lane_b"
EVIDENCE = HERE / "fig_trajectories_ci.evidence.json"
FIG3 = "auto/fig_trajectories_ci.png"
P8_OUTPUTS = ("fig_trajectories_ci.png", "fig_fairness.png", "fig_roc.png", "fig_labeleff.png")
PRINT_WIDTH_IN = 5.5            # \linewidth of the NeurIPS 2026 text block; Fig. 3 is included 1:1

OCCLUSION = "figures/fig_archived_occlusion_examples.png"
OCCLUSION_SHA256 = "9792883349d0c3b173f41be67e3ab9cbc3a7c621c3d51934fe341ee724fae391"
OCCLUSION_SOURCES = ("archived_overlay_manifest", "archived_overlay_generator", "archived_overlay_source",
                     "archived_overlay_assembly", "archived_overlay_prior_review")
OLD_FIG3_CONTEXT = "56b70901b9bee988162a8b71978652117a071662ad95140d696d5a91cffbb20d"
TABLE1_CONTEXT = "60ad6b8d28c7f88780870086b4db46619b764c03a79a72c239063960df7c2cb9"
PRE_LANE_B_REV = "30a8c53"   # review file before lane B (old Fig. 3 caption receipts)
REVIEWER = "GitHub Copilot I13 lane-B figure review, 2026-10-12"
REKEY_NOTE = "; context re-keyed after the lane-B Fig. 3 caption edit (GitHub Copilot I13, 2026-10-12)"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats_table():
    return json.loads((STATS / "p1c_stats.json").read_text(encoding="utf-8"))["table"]


def row_index(key):
    keys = [r["key"] for r in stats_table()]
    assert keys.count(key) == 1, key
    return keys.index(key)


# ---------------------------------------------------------------- Fig. 3 replay and audit
def replay_fig3():
    """Run p8 main() into a work directory with savefig captured (as p15 does)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import figstyle_v2 as st

    evidence = numeric.Evidence(PAPER, STATS)
    delivered = sha(PAPER / FIG3)
    work = assets.unique_work(prefix="laneb-fig3")
    save = Figure.savefig
    captured = {}

    def capture(fig, destination, *args, **kwargs):
        name = Path(destination).name
        stream = io.BytesIO()
        save(fig, stream, *args, format="png", **kwargs)
        captured[name] = hashlib.sha256(stream.getvalue()).hexdigest()
        if name != "fig_trajectories_ci.png":
            return
        records = plots.p8_artists(fig, name, evidence)
        guard = st.layout_guard(fig)
        texts = st.visible_texts(fig)
        scale = PRINT_WIDTH_IN / fig.get_figwidth()
        sizes = sorted({round(t.get_fontsize() * scale, 2) for t in texts})
        captured["audit"] = {
            "inches": [fig.get_figwidth(), fig.get_figheight()], "dpi": kwargs.get("dpi"),
            "print_scale": scale, "text_sizes_pt": sizes, "min_text_pt": sizes[0],
            "texts": sorted(t.get_text() for t in texts), "layout_guard": guard,
            "legend": [t.get_text() for t in fig.legends[0].get_texts()],
            "titles": [a.get_title(loc="left") for a in fig.axes],
            "axis_labels": [[a.get_xlabel(), a.get_ylabel()] for a in fig.axes],
            "observed_series": records}

    try:
        spec = importlib.util.spec_from_file_location("laneb_p8_candidate", P8)
        module = importlib.util.module_from_spec(spec)
        with matplotlib.rc_context(), contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
            module.AUTO = str(work)
            module.STATS = str(STATS)
            with patch.object(Figure, "savefig", capture):
                module.main()
        bundle = {}
        for path in sorted(work.iterdir()):
            bundle[path.name] = sha(path)
    finally:
        plt.close("all")
        shutil.rmtree(work)
    for name in P8_OUTPUTS:
        bundle[name] = captured[name]
    paper_auto = {name: sha(PAPER / "auto" / name) for name in bundle}
    differs = sorted(n for n in bundle if bundle[n] != paper_auto[n])
    assert captured["fig_trajectories_ci.png"] == delivered, "delivered Fig. 3 differs from the p8 replay"
    assert differs == [], "P8 outputs differ from paper/auto: %s" % differs
    audit = captured["audit"]
    assert audit["min_text_pt"] >= 7.0, audit["text_sizes_pt"]
    assert not audit["layout_guard"]["overlaps"] and not audit["layout_guard"]["outside_canvas"], audit
    return delivered, bundle, audit, evidence


def plotted_content():
    st = json.loads((STATS / "p1c_stats.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("laneb_p8_constants", P8)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    table = st["table"]
    anc = next(i for i, r in enumerate(table) if r["key"] == "ancestor@ep25@fp32")
    series = []
    for arm, precision, style in module.TRAJ_SERIES:
        rows = sorted([(i, r) for i, r in enumerate(table) if r["arm"] == arm and r["precision"] == precision],
                      key=lambda pair: pair[1]["epoch"])
        series.append({
            "arm": arm, "label": module.ARM_FIG[arm], "precision": precision,
            "line_style": "solid" if style == "-" else "dashed",
            "joined": [{"key": table[i]["key"], "table_index": i, "epoch": table[i]["epoch"],
                        "auc": table[i]["auc"]} for i in [anc] + [i for i, r in rows
                                                                  if r["epoch"] in module.TRAJ_COMMON_EPOCHS]],
            "unjoined_open_markers": [{"key": r["key"], "table_index": i, "epoch": r["epoch"], "auc": r["auc"]}
                                      for i, r in rows if r["epoch"] not in module.TRAJ_COMMON_EPOCHS]})
    deltas = []
    for arm in ("envelope", "oracle", "cover-f021"):
        for ep in module.TRAJ_COMMON_EPOCHS:
            precision = "fp32" if arm == "cover-f021" else "fp16"
            a, b = "%s@ep%d@%s" % (arm, ep, precision), "random@ep%d@%s" % (ep, precision)
            (i, c), = [(i, c) for i, c in enumerate(st["contrasts"]) if c["a"] == a and c["b"] == b]
            deltas.append({"label": module.ARM_FIG[arm], "contrast_index": i, "a": a, "b": b,
                           "delta_a_minus_b": c["delta_a_minus_b"], "boot_ci95_lo": c["boot_ci95_lo"],
                           "boot_ci95_hi": c["boot_ci95_hi"], "x_offset": module.TRAJ_DELTA_OFFSET[arm]})
    drawn = {s["joined"][j]["key"] for s in series for j in range(len(s["joined"]))}
    drawn |= {m["key"] for s in series for m in s["unjoined_open_markers"]}
    omitted = sorted(r["key"] for r in table if r["key"] not in drawn)
    return {"ancestor": {"key": table[anc]["key"], "table_index": anc, "epoch": table[anc]["epoch"],
                         "auc": table[anc]["auc"], "marker": "filled star, legend entry 'ancestor'"},
            "panel_a_series": series, "panel_b_paired_deltas": deltas,
            "n_bootstrap": st["n_bootstrap"],
            "not_drawn": {"rows": omitted,
                          "reason": ("anatomy-v1@ep30 is the excluded continuation reported only in the appendix "
                                     "(main text: 'The early ANATOMY-V1 continuation is reported in Appendix'); "
                                     "the fp32 re-probes of RANDOM, CENTROID and ENVELOPE are Table 1's lower "
                                     "block and Appendix fp32, not trajectories (the previous Fig. 3 omitted "
                                     "them as well)")}}


def build_evidence(delivered, bundle, audit):
    plotted = plotted_content()
    return {
        "status": "programmatically_verified_trajectory_figure",
        "figure": FIG3,
        "producer": {"file": "autopilot/p8_make_assets.py", "sha256": sha(P8),
                     "function": "draw_trajectories (called from main())"},
        "independent_validator": {"file": "autopilot/numeric_plot_review.py", "sha256": sha(VALIDATOR),
                                  "function": "p8_artists, fig_trajectories_ci.png branch (run by p15 on every "
                                              "audit; requires the delivered PNG bytes)"},
        "inputs": [{"root": "stats", "path": "p1c_stats.json", "sha256": sha(STATS / "p1c_stats.json"),
                    "note": "original-run aggregate statistics (published results), read-only"}],
        "output": {"path": FIG3, "sha256": delivered, "inches": audit["inches"], "dpi": audit["dpi"],
                   "inclusion": "\\includegraphics[width=\\linewidth]: 5.5 in, scale %.3f" % audit["print_scale"]},
        "data_identity": ("Every plotted coordinate equals p1c_stats.json (table rows and matched-precision "
                          "contrasts), checked artist by artist by numeric_plot_review.p8_artists; the same "
                          "values as the previous Fig. 3 for every point still drawn."),
        "p8_bundle_identity": {"outputs_sha256": bundle,
                               "all_equal_paper_auto": True,
                               "note": "only draw_trajectories changed; every table, macro and other figure of "
                                       "the P8 bundle replays byte-identical"},
        "plotted": plotted,
        "design": {
            "review": "R1 polish review F1/H5 (readability, encoding) and M4 (caption wording)",
            "print_size": "drawn at 5.5 x 2.05 in and included at \\linewidth (1:1); every text 7 pt",
            "epoch_axis": ("ticks only at the measured common epochs 25, 50, 75, 100; lines join the ancestor and "
                           "epochs 50/75/100; other checkpoints (ANATOMY-V2 35, 40; COVER 27, 30, 34, 73) are "
                           "unjoined open markers, so no interpolation crosses uneven epoch grids"),
            "precision": "solid = fp16 family, dashed = fp32 arms (ANATOMY-V2, COVER); panel (b) deltas use the "
                         "precision-matched RANDOM (p8 matched_trajectory_keys)",
            "colours": "the paper's single arm mapping (p8 COL): RANDOM #000000, CENTROID #882255, ENVELOPE "
                       "#0072B2, ANATOMY-V2 #666666, COVER #009E73, ancestor #333333; unique marker per arm",
            "legend": "one row above both panels, upper-case arm names, ancestor as a legend entry (no arrow)",
            "anatomy_v1": "not drawn (excluded continuation, appendix only)",
            "panel_b": "x = epoch 50 / 75 / 100, y = Delta AUC vs RANDOM, zero line, 95% bootstrap intervals"},
        "typography": {"text_sizes_pt_at_print": audit["text_sizes_pt"], "min_text_pt": audit["min_text_pt"]},
        "layout_guard": audit["layout_guard"],
        "texts": audit["texts"], "legend": audit["legend"], "titles": audit["titles"],
        "axis_labels": audit["axis_labels"],
        "artist_replay": {"status": "programmatically_verified_plotted_values",
                          "observed_series": audit["observed_series"]},
        "limits": ("Original single runs; intervals are test-case bootstrap intervals, not seed variation. "
                   "No new result; no FairVision pixels."),
    }


# ---------------------------------------------------------------- registry
def registry_entry(evidence_bytes, delivered):
    return {
        "script": "autopilot/p8_make_assets.py",
        "script_sha256_at_export": sha(P8),
        "inputs": ["p1c_stats.json"],
        "evidence_manifest": REL_HERE + "/" + EVIDENCE.name,
        "evidence_manifest_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "expected_output_sha256": delivered,
        "refresh_policy": ("p15 replays p8 main() and requires these PNG bytes (numeric_plot_review.p8_artists); "
                           "any change of the trajectory code, the statistics or the caption requires a renewed "
                           "lane-B registration"),
        "evidence_scope": ("Fig. 3 at print size (5.5 x 2.05 in, 7 pt, included 1:1): (a) frozen-probe test AUC of "
                           "RANDOM, CENTROID, ENVELOPE (fp16) and ANATOMY-V2, COVER (fp32) from the shared epoch-25 "
                           "ancestor, lines joining epochs 50/75/100 only, other checkpoints as open markers, "
                           "ANATOMY-V1 (excluded continuation) not drawn; (b) matched-precision paired differences "
                           "from RANDOM with 95% percentile bootstrap intervals at epochs 50/75/100. Every "
                           "coordinate equals p1c_stats.json (p15 artist replay). Original single runs; no seed "
                           "variation."),
        "rendered_metrics": ["frozen-probe test AUC (original runs)",
                             "paired test-AUC difference from RANDOM with 95% percentile bootstrap interval"],
        "raw_oct_pixels_rendered": False,
        "subject_identifiers_rendered": False,
    }


def entry_text(key, entry):
    lines = ['    "%s": {' % key]
    items = list(entry.items())
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if isinstance(v, list) and len(json.dumps(v)) > 90:
            body = ",\n".join("        " + json.dumps(x, ensure_ascii=False) for x in v)
            lines.append('      "%s": [\n%s\n      ]%s' % (k, body, comma))
        else:
            lines.append('      "%s": %s%s' % (k, json.dumps(v, ensure_ascii=False), comma))
    lines.append("    }")
    return "\n".join(lines)


def update_registry(evidence_bytes, delivered):
    raw = REGISTRY.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    expected = json.loads(text)
    entry = registry_entry(evidence_bytes, delivered)
    expected["producers"][FIG3] = entry
    expected["producers"].pop(OCCLUSION, None)
    expected["retired_inputs"][OCCLUSION] = OCCLUSION_SHA256
    # Fig. 3 entry: one line before, a block after the first run.
    one_line = re.search(r'\n    "%s": \{[^\n]*\},?(?=\n)' % re.escape(FIG3), text)
    if one_line:
        comma = "," if one_line.group(0).endswith(",") else ""
        text = text[:one_line.start()] + "\n" + entry_text(FIG3, entry) + comma + text[one_line.end():]
    else:
        start = text.index('    "%s": {' % FIG3)
        end = text.index("\n    }", start) + len("\n    }")
        text = text[:start] + entry_text(FIG3, entry) + text[end:]
    if '    "%s": {' % OCCLUSION in text:
        start = text.index('    "%s": {' % OCCLUSION)
        end = text.index("\n    }", start) + len("\n    }")
        assert text[end:end + 2] == ",\n", "occlusion entry is not followed by another producer"
        text = text[:start] + text[end + 2:]
    if '"%s"' % OCCLUSION not in text.split('"retired_inputs"', 1)[1]:
        anchor = '\n  "retired_inputs": {\n'
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor + '    "%s": "%s",\n' % (OCCLUSION, OCCLUSION_SHA256))
    if json.loads(text) != expected:
        raise SystemExit("registry: surgical edit does not reproduce the intended JSON")
    out = text.replace("\n", "\r\n") if crlf else text
    return out, out != raw


# ---------------------------------------------------------------- numeric reviews
def fig3_rows():
    text = (PAPER / "main_submission.tex").read_text(encoding="utf-8")
    env = [m for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", text, re.S) if r"\label{fig:traj}" in m[0]]
    assert len(env) == 1
    context = numeric.digest_text(env[0][0])
    rows = [r for r in numeric.literals(text, "main_submission.tex") if r["context_sha256"] == context]
    return context, [r for r in rows if r["status"] != "structural"]


def fig3_receipts(data, context, rows):
    by_key = {(e["file"], e["context_sha256"], e["token_index"]): e for e in data["literals"]}
    pristine = {}
    for e in json.loads(subprocess_show(PRE_LANE_B_REV + ":paper/genai4health2026/numeric_reviews.json"))["literals"]:
        pristine[(e["context_sha256"], e["token_index"])] = e

    def rekey(old_context, old_index, row):
        src = pristine.get((old_context, old_index)) or by_key.get(("main_submission.tex", old_context, old_index))
        assert src is not None and src["value"] == row["value"], (old_context[:12], old_index, row["value"])
        new = json.loads(json.dumps(src))
        new["context_sha256"], new["token_index"] = row["context_sha256"], row["token_index"]
        if "review" in new and REKEY_NOTE not in new["review"]["reviewer"]:
            new["review"]["reviewer"] += REKEY_NOTE
        return new

    def epoch_receipt(row, key):
        return {"file": "main_submission.tex", "context_sha256": row["context_sha256"],
                "token_index": row["token_index"], "value": row["value"],
                "review": {"kind": "protocol", "reviewer": REVIEWER,
                           "rationale": ("Measured checkpoint epoch %s, one of the epochs 50, 75 and 100 that "
                                         "every arm drawn in Fig. 3(a) shares (ANATOMY-V2 has no epoch-100 "
                                         "checkpoint); lines join only these and the epoch-25 ancestor."
                                         % row["value"]),
                           "evidence": [{"source": "lr_stats", "pointer": "/table/%d" % row_index(key)},
                                        {"source": "lr_p8_code",
                                         "excerpt": "TRAJ_COMMON_EPOCHS = (50, 75, 100)"}]}}

    values = [r["value"] for r in rows]
    assert values == ["-25", "50", "75", "100", "2", "10", "-4", "50"], values
    out = [rekey(OLD_FIG3_CONTEXT, 1, rows[0]),
           epoch_receipt(rows[1], "random@ep50@fp16"),
           epoch_receipt(rows[2], "random@ep75@fp16"),
           epoch_receipt(rows[3], "random@ep100@fp16")]
    for row, table_index in zip(rows[4:7], (6, 7, 8)):
        out.append(rekey(TABLE1_CONTEXT, table_index, row))
    out.append(rekey(OLD_FIG3_CONTEXT, 4, rows[7]))
    return out


def subprocess_show(spec):
    import subprocess
    return subprocess.check_output(["git", "show", spec], cwd=REPO).decode("utf-8-sig")


def update_reviews():
    raw = REVIEWS.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    # occlusion figure: receipt and the sources only it cited
    data["figures"] = [f for f in data["figures"] if f["path"] != OCCLUSION]
    cited = json.dumps({k: data[k] for k in ("macros", "literals", "figures")})
    for name in OCCLUSION_SOURCES:
        assert '"%s"' % name not in cited, "source still cited: " + name
        data["sources"].pop(name, None)
    # p8 source pin: every excerpt cited from it must still occur in the new bytes
    p8_text = P8.read_bytes().decode("utf-8-sig")
    context, rows = fig3_rows()
    receipts = fig3_receipts(data, context, rows)
    excerpts = [ev["excerpt"] for e in data["literals"] + receipts for ev in (e.get("review") or {}).get("evidence", [])
                if ev.get("source") == "lr_p8_code"]
    assert excerpts and all(x in p8_text for x in excerpts), excerpts
    data["sources"]["lr_p8_code"]["sha256"] = sha(P8)
    # Fig. 3 caption literals: drop the old-caption and current-caption keys, add the receipts
    data["literals"] = [e for e in data["literals"]
                        if not (e["file"] == "main_submission.tex" and e["context_sha256"] in (OLD_FIG3_CONTEXT, context))]
    data["literals"] += receipts
    out = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if b"\r\n" in raw:
        out = out.replace("\n", "\r\n")
    out = out.encode("utf-8")
    return out, out != raw, {"fig3_context": context, "fig3_receipts": len(receipts),
                             "p8_excerpts_checked": sorted(set(excerpts))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    delivered, bundle, audit, _ = replay_fig3()
    evidence_text = json.dumps(build_evidence(delivered, bundle, audit), indent=1, ensure_ascii=False) + "\n"
    evidence_bytes = evidence_text.encode("utf-8")
    evidence_changed = not EVIDENCE.is_file() or EVIDENCE.read_bytes() != evidence_bytes
    registry_text, registry_changed = update_registry(evidence_bytes, delivered)
    reviews_bytes, reviews_changed, info = update_reviews()
    if not args.check:
        if evidence_changed:
            EVIDENCE.write_bytes(evidence_bytes)
        if registry_changed:
            REGISTRY.write_bytes(registry_text.encode("utf-8"))
        if reviews_changed:
            REVIEWS.write_bytes(reviews_bytes)
    print(json.dumps({"fig3_sha256": delivered, "min_text_pt": audit["min_text_pt"],
                      "texts_checked": audit["layout_guard"]["texts_checked"],
                      "evidence_changed": evidence_changed, "registry_changed": registry_changed,
                      "reviews_changed": reviews_changed, "written": not args.check, **info}, indent=1))
    return 1 if args.check and (evidence_changed or registry_changed or reviews_changed) else 0


if __name__ == "__main__":
    sys.exit(main())
