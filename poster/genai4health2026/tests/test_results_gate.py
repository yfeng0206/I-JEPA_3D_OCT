"""Regression tests for the poster's results gate (make_preview.py --precheck).

Usage:
  python tests/test_results_gate.py --bundle DIR [--other-bundle DIR2] --work WORKDIR

DIR (and DIR2, a different scenario) are synthetic result bundles written by
``autopilot/make_cr_results.py build --allow-synthetic`` (the files the generator puts in
paper/genai4health2026/auto/). Each case copies DIR to WORKDIR/<case>/auto, applies one
tampering, and runs the gate exactly as build.ps1 does
(``make_preview.py --precheck --allow-synthetic --results-dir ...``) from a copy of the tool
in WORKDIR/tool, so no state is written into the repository. The baseline must pass and every
tampered case must fail. The registered-RANDOM-reference cases (V10-2) depend on the bundle:
with a complete reference (e.g. all runs on time) they mark it late, move the range or drop
\\RepRandomRange; without it (original RANDOM missing or late, or fewer than two complete seed
indices; generator label M_incomplete) they forge the registered label a partial range would
give, clear the recorded reasons, add a range macro or a registered replication label.
Synthetic data only; files are written only under WORKDIR.
Set POSTER_GENERATOR_DIR to autopilot/ to include the generator's own metadata check.
Relative paths are resolved first. Exit code: number of failing expectations.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TOOL = HERE.parent / "make_preview.py"
NAMES = {"M0": "placement difference not detected", "M-": "placement difference reversed",
         "M+": "placement accounts for most of the gap", "M_partial": "both contribute"}


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def edit_tex(auto, name, new_body=None, delete=False):
    path = auto / "cr_results.tex"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^\\newcommand\{\\%s\}\{.*\}\n" % name, re.M)
    assert len(pattern.findall(text)) == 1, name
    replacement = "" if delete else "\\newcommand{\\%s}{%s}\n" % (name, new_body)
    text = pattern.sub(lambda m: replacement, text)
    path.write_bytes(text.encode("utf-8"))


def load_side(auto):
    return json.loads((auto / "cr_results.values.json").read_text(encoding="utf-8"))


def save_side(auto, side):
    (auto / "cr_results.values.json").write_text(json.dumps(side, indent=1) + "\n",
                                                 encoding="utf-8")


def rehash(auto, name):
    side = load_side(auto)
    side["outputs"][name] = sha(auto / name)
    save_side(auto, side)


def pick(names, present):
    return next(n for n in names if n in present)


def cases(bundle):
    """(name, expected exit, mutation(auto, other bundle), substring the output must contain).
    V7's macros (RepAUCCentroidSeedA, RepPoolSliceMaxDeltaEnvelope) unless the bundle lacks them."""
    side = load_side(bundle)
    present = {m["name"] for m in side["macros"]}
    num = pick(("RepAUCCentroidSeedA", "RepAUCCentroidSeedB", "RepAUCRandomOrig"), present)
    gone = pick(("RepPoolSliceMaxDeltaEnvelope", "RepPoolPatchMaxDeltaEnvelope",
                 "RepMeanAUCRandom"), present)
    shown = next(m["display"] for m in side["macros"] if m["name"] == num)
    fake = "0.1234" if shown != "0.1234" else "0.4321"

    def v7_number(a, o):
        edit_tex(a, num, fake)

    def v7_label(a, o):
        edit_tex(a, "RepLabelCentroid",
                 "reversed" if side["scenario"]["labels"]["centroid"] != "reversed"
                 else "consistent direction")

    def v7_deleted(a, o):
        edit_tex(a, gone, delete=True)

    def v7_all_three(a, o):
        v7_number(a, o)
        v7_label(a, o)
        v7_deleted(a, o)

    def v7_figure_swap(a, o):
        src = (o / "fig_cr_seed_spread.pdf") if o else None
        if src and src.is_file() and sha(src) != sha(a / "fig_cr_seed_spread.pdf"):
            shutil.copyfile(src, a / "fig_cr_seed_spread.pdf")
        else:
            with open(a / "fig_cr_seed_spread.pdf", "ab") as stream:
                stream.write(b"\n% swapped\n")

    def v7_minimal_sidecar(a, o):
        (a / "cr_results.values.json").write_text('{"synthetic": true}\n', encoding="utf-8")

    def number_rehashed(a, o):
        v7_number(a, o)
        rehash(a, "cr_results.tex")

    def label_rehashed(a, o):
        v7_label(a, o)
        rehash(a, "cr_results.tex")

    def deleted_rehashed(a, o):
        v7_deleted(a, o)
        rehash(a, "cr_results.tex")

    def extra_redefinition_rehashed(a, o):
        with open(a / "cr_results.tex", "a", encoding="utf-8", newline="\n") as stream:
            stream.write("\\renewcommand{\\RepAUCCentroidSeedA}{0.1234}\n")
        rehash(a, "cr_results.tex")

    def table_tampered(a, o):
        path = a / "cr_table_replication.tex"
        path.write_text(path.read_text(encoding="utf-8").replace("\\midrule", "\\midrule ", 1),
                        encoding="utf-8")

    def evidence_point(a, o):
        path = a / "fig_cr_seed_spread.evidence.json"
        ev = json.loads(path.read_text(encoding="utf-8"))
        ev["points"][1]["test_auc"] += 0.01
        path.write_text(json.dumps(ev, indent=1) + "\n", encoding="utf-8")

    def evidence_missing(a, o):
        (a / "fig_cr_seed_spread.evidence.json").unlink()

    def figure_not_in_sidecar(a, o):
        side = load_side(a)
        side["figure"] = None
        save_side(a, side)

    def sidecar_flag_flipped(a, o):
        side = load_side(a)
        side["synthetic"] = False
        save_side(a, side)

    def sidecar_label_changed(a, o):
        side = load_side(a)
        side["scenario"]["labels"]["centroid"] = "reversed"
        save_side(a, side)

    def sidecar_p_changed(a, o):
        side = load_side(a)
        mc = side["scenario"]["matched_control"]
        if mc.get("P") is None:
            mc["P"], mc["P_descriptive"] = 0.5, False
        else:
            mc["P"], mc["P_descriptive"] = None, None
        save_side(a, side)

    def control_label_unknown(a, o):
        side = load_side(a)
        side["scenario"]["matched_control"]["label"] = "M_bogus"
        save_side(a, side)

    def edit_evidence(a, change):
        path = a / "fig_cr_seed_spread.evidence.json"
        ev = json.loads(path.read_text(encoding="utf-8"))
        change(ev)
        path.write_text(json.dumps(ev, indent=1) + "\n", encoding="utf-8")

    def v9_extra_mean(a, o):
        edit_evidence(a, lambda ev: ev["means"].__setitem__(
            "random_cb", {"x": 3, "mean": 0.95, "n": 999}))

    def evidence_missing_mean(a, o):
        edit_evidence(a, lambda ev: ev["means"].pop("envelope"))

    def evidence_mean_extra_key(a, o):
        edit_evidence(a, lambda ev: ev["means"]["random"].__setitem__("label", "x"))

    def evidence_extra_point(a, o):
        def add(ev):
            extra = dict(ev["points"][-1])
            extra.update({"run": "invented", "tag": "SeedC" if extra["tag"] != "SeedC"
                          else "SeedB", "test_auc": 0.95})
            ev["points"].append(extra)
        edit_evidence(a, add)

    def evidence_duplicate_point(a, o):
        edit_evidence(a, lambda ev: ev["points"].append(dict(ev["points"][0])))

    def evidence_missing_point(a, o):
        edit_evidence(a, lambda ev: ev["points"].pop())

    def v9_display_forged(a, o):
        # V9 probe: TeX value, its output hash AND the sidecar display changed; full
        # precision (and the figure) unchanged.
        v7_number(a, o)
        rehash(a, "cr_results.tex")
        side = load_side(a)
        next(m for m in side["macros"] if m["name"] == num)["display"] = fake
        save_side(a, side)

    def v9_control_relabelled(a, o):
        # V9 probe: matched-control label/name changed, P/B/range unchanged.
        side = load_side(a)
        mc = side["scenario"]["matched_control"]
        new = "M-" if mc["label"] != "M-" else "M+"
        mc["label"] = new
        mc["name"] = {"M-": "placement difference reversed",
                      "M+": "placement accounts for most of the gap"}[new]
        save_side(a, side)

    def marker_removed_rehashed(a, o):
        path = a / "cr_results.tex"
        path.write_text(path.read_text(encoding="utf-8").replace("SYNTHETIC DRY RUN", "DRY RUN"),
                        encoding="utf-8")
        rehash(a, "cr_results.tex")

    # V10-2: the registered RANDOM reference (on-time original RANDOM + the new RANDOM runs of
    # >= 2 complete seed indices). Mutations keep every hash consistent (sidecar = root of trust).
    mc0 = side["scenario"]["matched_control"]
    missing = side["scenario"].get("reference_missing") or []
    anchor = "RepAUCRandomOrig" in present

    def set_reference(a, reasons, scenario_too=True):
        s = load_side(a)
        s["scenario"]["matched_control"]["reference_missing"] = list(reasons)
        if scenario_too:
            s["scenario"]["reference_missing"] = list(reasons)
        save_side(a, s)

    def forged_label(p, b, b_desc):
        # What the label would read with a range of the remaining RANDOM runs (abs(P)/2).
        if p < 0:
            return "M-"
        if b is None or b_desc:
            return "M_unassessed"
        return "M+" if p >= 0.5 * (p + b) else "M_partial"

    def v10_registered_without_reference(a, o):
        s = load_side(a)
        mc = s["scenario"]["matched_control"]
        mc["spread"] = abs(mc["P"]) / 2
        mc["label"] = forged_label(mc["P"], mc["B"], mc["B_descriptive"])
        mc["name"] = NAMES.get(mc["label"])
        save_side(a, s)

    def v10_reference_cleared(a, o):
        v10_registered_without_reference(a, o)
        set_reference(a, [])

    def add_range_macro(a, o):
        path = a / "cr_results.tex"
        text = path.read_text(encoding="utf-8")
        at = re.search(r"^\\newcommand\{\\Rep", text, re.M).start()
        path.write_bytes((text[:at] + "\\newcommand{\\RepRandomRange}{0.0100}\n" + text[at:])
                         .encode("utf-8"))
        s = load_side(a)
        s["macros"].append({"name": "RepRandomRange", "display": "0.0100",
                            "full_precision": [0.01],
                            "expression": {"op": "format", "format": "%.4f", "args": []}})
        s["outputs"]["cr_results.tex"] = sha(path)
        save_side(a, s)

    def labels_registered_without_reference(a, o):
        edit_tex(a, "RepLabelCentroid", "consistent direction")
        s = load_side(a)
        s["scenario"]["labels"]["centroid"] = "consistent_direction"
        next(t for t in s["text_macros"] if t["name"] == "RepLabelCentroid")["text"] = \
            "consistent direction"
        s["outputs"]["cr_results.tex"] = sha(a / "cr_results.tex")
        save_side(a, s)

    def reference_lists_disagree(a, o):
        set_reference(a, [] if missing else ["anchor_late"], scenario_too=False)

    def reference_reason_swapped(a, o):
        set_reference(a, ["anchor_missing"] if anchor else ["anchor_late"])

    def reference_marked_late(a, o):
        set_reference(a, ["anchor_late"])

    def range_not_from_random_runs(a, o):
        s = load_side(a)
        s["scenario"]["matched_control"]["spread"] += 0.001
        save_side(a, s)

    def range_macro_removed(a, o):
        edit_tex(a, "RepRandomRange", delete=True)
        s = load_side(a)
        s["macros"] = [m for m in s["macros"] if m["name"] != "RepRandomRange"]
        s["outputs"]["cr_results.tex"] = sha(a / "cr_results.tex")
        save_side(a, s)

    def baseline(a, o):
        pass

    reference_cases = [
        ("v10_reference_lists_disagree", 1, reference_lists_disagree,
         "!= scenario.reference_missing"),
        ("v10_reference_reason_swapped", 1, reference_reason_swapped,
         "disagrees with the sidecar" if anchor else "finished late, but it has no result"),
    ]
    if missing:
        has_p = isinstance(mc0.get("P"), (int, float)) and not mc0.get("P_descriptive")
        if has_p:
            reference_cases += [
                ("v10_registered_label_without_reference", 1, v10_registered_without_reference,
                 "pre-registered rule"),
                ("v10_reference_cleared", 1, v10_reference_cleared,
                 "missing although the registered RANDOM reference is complete"
                 if missing == ["anchor_late"] else "disagrees with the sidecar"),
            ]
        reference_cases += [
            ("v10_range_macro_added", 1, add_range_macro,
             "RepRandomRange defined without the registered RANDOM reference"),
        ]
        if set(missing) & {"indices", "anchor_missing", "anchor_late"}:
            reference_cases.append(("v10_label_registered_without_reference", 1,
                                    labels_registered_without_reference,
                                    "for centroid without the registered RANDOM reference"))
    else:
        reference_cases += [
            ("v10_reference_marked_late", 1, reference_marked_late, "pre-registered rule"),
            ("v10_range_not_from_random_runs", 1, range_not_from_random_runs,
             "is not the range of the registered RANDOM runs"),
            ("v10_range_macro_removed", 1, range_macro_removed,
             "RepRandomRange missing although the registered RANDOM reference is complete"),
        ]

    return [
        ("baseline", 0, baseline, "state written"),
        ("baseline_without_allow_synthetic", 1, baseline, "synthetic dry-run results"),
        ("v7_altered_number", 1, v7_number, "does not match its recorded SHA-256"),
        ("v7_altered_label", 1, v7_label, "RepLabelCentroid"),
        ("v7_deleted_macro", 1, v7_deleted, "does not match its recorded SHA-256"),
        ("v7_all_three", 1, v7_all_three, "does not match its recorded SHA-256"),
        ("v7_swapped_figure", 1, v7_figure_swap, "fig_cr_seed_spread.pdf does not match"),
        ("v7_minimal_sidecar", 1, v7_minimal_sidecar, "schema"),
        ("number_with_rehashed_output", 1, number_rehashed, num + " = '" + fake),
        ("label_with_rehashed_output", 1, label_rehashed, "RepLabelCentroid"),
        ("deleted_with_rehashed_output", 1, deleted_rehashed,
         gone + " is in the sidecar but not defined in cr_results.tex"),
        ("extra_redefinition_rehashed", 1, extra_redefinition_rehashed, "not a single"),
        ("replication_table_tampered", 1, table_tampered, "cr_table_replication.tex"),
        ("evidence_point_tampered", 1, evidence_point, "figure point"),
        ("evidence_missing", 1, evidence_missing, "evidence"),
        ("figure_not_in_sidecar", 1, figure_not_in_sidecar, "records no figure"),
        ("sidecar_synthetic_flag_flipped", 1, sidecar_flag_flipped, "synthetic marker"),
        ("sidecar_label_changed", 1, sidecar_label_changed, "RepLabelCentroid"),
        ("sidecar_P_changed", 1, sidecar_p_changed, "RepPlacementDelta"),
        ("control_label_unknown", 1, control_label_unknown, "unknown to the poster"),
        ("marker_removed_rehashed", 1, marker_removed_rehashed, "synthetic marker"),
        ("v9_extra_figure_mean", 1, v9_extra_mean, "extra or missing mean"),
        ("evidence_missing_mean", 1, evidence_missing_mean, "extra or missing mean"),
        ("evidence_mean_extra_key", 1, evidence_mean_extra_key, "figure mean for random"),
        ("evidence_extra_point", 1, evidence_extra_point, "extra point"),
        ("evidence_duplicate_point", 1, evidence_duplicate_point, "appears twice"),
        ("evidence_missing_point", 1, evidence_missing_point, "lacks points"),
        ("v9_sidecar_display_forged", 1, v9_display_forged, "recorded format"),
        ("v9_sidecar_control_relabelled", 1, v9_control_relabelled, "pre-registered rule"),
    ] + reference_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--other-bundle")
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    bundle = pathlib.Path(a.bundle).resolve()
    other = pathlib.Path(a.other_bundle).resolve() if a.other_bundle else None
    work = pathlib.Path(a.work).resolve()
    if not (bundle / "cr_results.tex").is_file():
        sys.exit("--bundle needs a generated results directory (cr_results.tex missing)")
    if "SYNTHETIC DRY RUN" not in (bundle / "cr_results.tex").read_text(encoding="utf-8"):
        sys.exit("refusing to use non-synthetic results for tests")
    shutil.rmtree(work, ignore_errors=True)
    tool = work / "tool"
    tool.mkdir(parents=True)
    shutil.copyfile(TOOL, tool / "make_preview.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8")
    failures = 0
    for name, want, mutate, needle in cases(bundle):
        auto = work / name / "auto"
        shutil.copytree(bundle, auto)
        mutate(auto, other)
        args = [sys.executable, str(tool / "make_preview.py"), "--precheck",
                "--results-dir", str(auto)]
        if name != "baseline_without_allow_synthetic":
            args.append("--allow-synthetic")
        proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", env=env,
                              cwd=str(tool))
        out = proc.stdout + proc.stderr
        ok = proc.returncode == want and needle in out
        if ok and name == "baseline":
            state = json.loads((tool / "generated" / "results_state.json").read_text(
                encoding="utf-8"))
            macros = {m["name"] for m in load_side(bundle)["macros"]}
            auc = {n for n in macros if re.match(r"^RepAUC(RandomCB|Random|Centroid|Envelope)"
                                                 r"(Orig|Seed[ABC])$", n)}
            means = sorted(a.lower() for a in ("Random", "Centroid", "Envelope")
                           if "RepMeanAUC" + a in macros)
            series = state.get("figure_series") or {}
            ok = len(series.get("points", [])) == len(auc) and series.get("means") == means
            needle = "plotted series = sidecar AUC/mean macros" if ok else "SERIES MISMATCH"
            side = load_side(bundle)
            missing = side["scenario"]["reference_missing"]
            mc = side["scenario"]["matched_control"]
            ref_ok = state.get("reference_missing") == missing and \
                ("RepRandomRange" in state["expected_macros"]) == (not missing) and \
                (state["control"] == "M_incomplete") == (bool(missing) and mc["P"] is not None
                                                         and not mc["P_descriptive"])
            ok = ok and ref_ok
            needle += ("; reference %s, control %s, range %s" % (
                ", ".join(missing) + " missing" if missing else "complete", state["control"],
                "shown" if "RepRandomRange" in state["expected_macros"] else "not shown")
                if ref_ok else "; REFERENCE STATE MISMATCH")
        failures += not ok
        first = next((ln[15:] for ln in out.splitlines() if ln.startswith("PROBLEMS")), "")
        if name == "baseline":
            first = needle
        print("%-34s exit %d (want %d) %s %s" % (name, proc.returncode, want,
                                                 "ok  " if ok else "FAIL", first[:160]))
        if not ok:
            print(out)
    print("%d case(s) failed" % failures if failures else "all cases behave as expected")
    return failures


if __name__ == "__main__":
    sys.exit(main())
