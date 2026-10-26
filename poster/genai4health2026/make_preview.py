"""Results gate, headless PNG preview and build report for the poster (no viewer is opened).

Usage:
  python make_preview.py --precheck [--results-dir DIR] [--allow-synthetic]
  python make_preview.py PRINT_PDF --design-log poster.log --png out.png [--dpi 50]
                         [--results-dir DIR] [--allow-synthetic]

--precheck (build.ps1 runs it before compiling) validates the registered-replication
bundle that autopilot/make_cr_results.py wrote to ../../paper/genai4health2026/auto/:
  * cr_results.values.json: schema cr_results_v1, boolean synthetic flag, scenario, matched
    control, numeric and text macro records, output and figure hashes; every display is its
    recorded format of its full-precision value, and the matched-control label follows from
    P, B, the registered RANDOM reference and eligibility (pre-registration amendment 2);
    the registered RANDOM reference (on-time original RANDOM + the new RANDOM runs of >= 2
    complete seed indices) agrees with the recorded reference_missing reasons; with it the
    range and \\RepRandomRange/Min/Max are those of exactly these AUCs, without it there is no
    range, no range macro and no registered label (M_incomplete);
  * every output it lists (cr_results.tex, cr_table_replication.tex, ...) and the figure PNG/PDF
    match their SHA-256; fig_cr_seed_spread.evidence.json is required with a figure, records the
    same figure hashes, and its arms/points/means equal exactly (no extra, missing or duplicate
    entries) the series built from the sidecar's AUC, s.d. and mean macros; its panel (b), if
    present, equals the paired differences, mean differences and registered-range band built
    from the sidecar; its encoding (contract cr_seed_spread_encoding_v2) is validated;
  * cr_results.tex defines exactly the sidecar's macros with the same values; the label macros
    equal the sidecar labels; P/B macros exist exactly when the sidecar has P/B;
  * if autopilot/make_cr_results.py is importable, its own verified_results_metadata() passes.
On success it writes generated/results_state.tex|json (validated control status, late-run
flags, seed status, expected macros, MD5s that poster.tex re-checks) and renders
generated/fig_cr_seed_spread_poster.pdf from that sidecar-built series (never from the evidence
file; only the validated encoding is read from it) with poster-size labels.
Any failure removes the generated state and exits 1, so nothing unvalidated is compiled.
Synthetic results exit 1 unless --allow-synthetic is given.

The report mode renders page 1 of the print PDF with PyMuPDF and reports page count, size,
fonts, Overfull/Underfull boxes, the fit check, placeholders, figures (allow-list; the two paper
figures must be the registered poster variants whose SHA-256 autopilot/release_assets.json
registers), the KeyBox/Answer wording against the paper macros, and re-validates the
results: the bundle again, the state the poster used, the log, and every result cell, label
and late-run mark extracted from the PDF against the sidecar. Exit 1 on any problem.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / ".." / ".." / "paper" / "genai4health2026" / "auto"
GENERATED = HERE / "generated"
STATE_TEX = GENERATED / "results_state.tex"
STATE_JSON = GENERATED / "results_state.json"
PP_TEX = GENERATED / "poster_units.tex"
POSTER_FIG_NAME = "fig_cr_seed_spread_poster.pdf"
POSTER_FIG = GENERATED / POSTER_FIG_NAME
STATE_SCHEMA = "poster_results_state_v1"

# Conventions of autopilot/make_cr_results.py (read, not imported, so the gate also runs
# where the generator is absent; the generator's own check is added when importable).
SCHEMA = "cr_results_v1"
FIGURE_SCHEMA = "cr_fig_seed_spread_v1"
SYNTHETIC_MARKER = "SYNTHETIC DRY RUN"
REQUIRED_OUTPUTS = ("cr_results.tex", "cr_table_replication.tex")
SENTINELS = ("RepEndpoint", "RepNSeedsWord", "RepLabelCentroid", "RepLabelEnvelope")
LABEL_TEX = {"consistent_direction": "consistent direction",
             "within_run_to_run_variation": "within run-to-run variation",
             "reversed": "reversed", "incomplete": "incomplete", "not_available": "not available"}
SEED_LETTERS = {1234: "A", 5678: "B", 9012: "C"}
ARM_WORD = {"random": "Random", "centroid": "Centroid", "envelope": "Envelope",
            "random_cb": "RandomCB"}
ARM_PLAIN = {"random": "RANDOM", "centroid": "CENTROID", "envelope": "ENVELOPE",
             "random_cb": "RANDOM-CB"}
FIG_ARMS = ("random", "centroid", "envelope", "random_cb")
GUIDED = ("centroid", "envelope")
# Figure encoding contract of the generator (make_cr_results.figure_encoding, recorded in the
# evidence as "encoding"): the poster render reads it from the validated evidence; these are
# the defaults for evidence that predates the contract (same values as contract v2: the
# paper's single arm-to-colour mapping, p8 arm markers, fill by status, seed letters).
FIG_ENCODING = "cr_seed_spread_encoding_v2"
FIG_COLORS = {"random": "#000000", "centroid": "#882255", "envelope": "#0072B2",
              "random_cb": "#E69F00"}
FIG_ARM_MARKERS = {"random": "o", "centroid": "s", "envelope": "^", "random_cb": "h"}
FIG_OFFSETS = {"Orig": -0.24, "SeedA": -0.08, "SeedB": 0.08, "SeedC": 0.24}
FIG_B_OFFSETS = {"SeedA": -0.18, "SeedB": 0.0, "SeedC": 0.18}
FIG_SEED_LABELS = {"Orig": "O", "SeedA": "A", "SeedB": "B", "SeedC": "C"}
FIG_FILL = {"registered": "full", "control": "full", "original": "left", "descriptive": "none"}
FIG_LATE_ORIGINAL = {"fill": "none", "label": "O"}  # V14-1: late original drawn open
FIG_BAND_GREY = "0.86"
DEFAULT_ENCODING = {"contract": FIG_ENCODING, "colors": FIG_COLORS, "markers": FIG_ARM_MARKERS,
                    "fill_by_status": FIG_FILL, "seed_labels": FIG_SEED_LABELS,
                    "x_offsets_panel_a": FIG_OFFSETS, "x_offsets_panel_b": FIG_B_OFFSETS,
                    "band_color": FIG_BAND_GREY, "late_original": FIG_LATE_ORIGINAL}
ALLOWED_MARKERS = set("os^vh<>dDpPH8X*")
CONTROL_TEXT = {
    "M+": "placement accounts for most of the gap (registered description)",
    "M0": "placement difference not detected (registered description)",
    "M-": "placement difference reversed (registered description)",
    "M_partial": "both contribute (registered description)",
    "M_desc": "descriptive only (a run finished after the freeze)",
    "M_unassessed": "not assessed (no on-time budget shift)",
    "M_noP": "not assessed (no CENTROID result at seed A)",
    "M_na": "not assessed (no RANDOM-CB result)",
    "M_incomplete": "not assessed (no registered RANDOM range)",
}
KNOWN_CONTROL = set(CONTROL_TEXT)
# Registered description names (pre-registration amendment 2); other labels have none.
CONTROL_NAMES = {"M0": "placement difference not detected", "M-": "placement difference reversed",
                 "M+": "placement accounts for most of the gap", "M_partial": "both contribute"}
# Why the registered RANDOM reference is missing (make_cr_results.registered_reference).
REFERENCE_REASONS = ("indices", "anchor_missing", "anchor_late", "range")
ABSTAIN_LABELS = ("incomplete", "not_available")
RANGE_MACROS = ("RepRandomRange", "RepRandomMin", "RepRandomMax")


def control_rule(mc, registered):
    """Matched-control label from P, B, the registered RANDOM range and eligibility (amendment
    2, as make_cr_results.matched_outcome): only on-time runs give a registered description,
    and only with the registered RANDOM reference (``registered``: the on-time re-probed
    original RANDOM plus the new RANDOM runs of at least two complete seed indices)."""
    p, b, r = mc.get("P"), mc.get("B"), mc.get("spread")
    if not mc.get("cb_present") and p is None and b is None:
        return "M_na"
    if p is None:
        return "M_noP"
    if mc.get("P_descriptive"):
        return "M_desc"
    if not registered:
        return "M_incomplete"
    if not (number(p) and number(r)):
        return None
    if abs(p) <= r:
        return "M0"
    if p < 0:
        return "M-"
    if b is None or mc.get("B_descriptive"):
        return "M_unassessed"
    return "M+" if p >= 0.5 * (p + b) else "M_partial"


def reference_state(side, scen, mc, complete, numeric, fullp, labels, problems):
    """Whether the registered RANDOM reference of pre-registration section 3 (reused by
    amendment 2 for the matched control) is complete: the on-time re-probed original RANDOM
    plus the new RANDOM runs of at least two complete seed indices (V10-2, as
    make_cr_results.registered_reference). The generator's recorded reasons must agree with
    what the sidecar shows (complete indices, an original RANDOM result; the original's
    lateness is recorded only there). With the reference, the matched-control range and
    \\RepRandomRange/Min/Max must be the range/min/max of exactly those RANDOM AUCs; without it
    no range is shown, no range is used and neither guided arm has a registered label."""
    recorded = scen.get("reference_missing")
    if not (isinstance(recorded, list) and all(r in REFERENCE_REASONS for r in recorded)
            and len(set(recorded)) == len(recorded)):
        problems.append(f"scenario.reference_missing {recorded!r} missing or invalid "
                        f"(expected a list of {', '.join(REFERENCE_REASONS)})")
        return False
    if mc.get("reference_missing") != recorded:
        problems.append(f"matched_control.reference_missing {mc.get('reference_missing')!r} "
                        f"!= scenario.reference_missing {recorded!r}")
    anchor = "RepAUCRandomOrig" in numeric
    runs = side.get("probe_runs")
    if not isinstance(runs, dict):
        problems.append("cr_results.values.json has no 'probe_runs' object")
    else:
        listed = any(isinstance(r, dict) and isinstance(r.get("identity"), dict)
                     and r["identity"].get("role") == "anchor"
                     and r["identity"].get("arm") == "random" for r in runs.values())
        if listed != anchor:
            problems.append("probe_runs " + ("lists" if listed else "lacks") +
                            " the original RANDOM run but \\RepAUCRandomOrig is " +
                            ("missing" if listed else "defined"))
    got, want = set(recorded), set()
    if len(complete) < 2:
        want.add("indices")
    if not anchor:
        want.add("anchor_missing")
    if got - {"anchor_late", "range"} != want:
        problems.append(f"reference_missing {sorted(got)} disagrees with the sidecar "
                        f"({len(complete)} complete seed indices; original RANDOM "
                        f"{'present' if anchor else 'missing'})")
    if "anchor_late" in got and not anchor:
        problems.append("reference_missing says the original RANDOM finished late, but it has "
                        "no result")
    if "range" in got and len(got) > 1:
        problems.append("reference_missing 'range' is recorded only when nothing else is missing")
    registered = not recorded
    if registered:
        names = ["RepAUCRandomOrig"] + ["RepAUCRandomSeed" + SEED_LETTERS[s] for s in complete]
        values = [fullp.get(n) for n in names]
        if not all(number(v) for v in values):
            problems.append("registered RANDOM reference without the AUCs of the original and "
                            "every complete seed index")
        else:
            want_range = {"RepRandomRange": max(values) - min(values),
                          "RepRandomMin": min(values), "RepRandomMax": max(values)}
            if not close(mc.get("spread"), want_range["RepRandomRange"]):
                problems.append(f"matched-control range {mc.get('spread')!r} is not the range "
                                f"of the registered RANDOM runs "
                                f"({want_range['RepRandomRange']!r})")
            for name, value in want_range.items():
                if name not in fullp:
                    problems.append(f"\\{name} missing although the registered RANDOM "
                                    f"reference is complete")
                elif not close(fullp[name], value):
                    problems.append(f"\\{name} {fullp[name]!r} is not the value from the "
                                    f"registered RANDOM runs ({value!r})")
    else:
        if mc.get("spread") is not None:
            problems.append(f"matched-control range {mc.get('spread')!r} without the registered "
                            f"RANDOM reference ({', '.join(recorded)})")
        for name in RANGE_MACROS:
            if name in numeric:
                problems.append(f"\\{name} defined without the registered RANDOM reference "
                                f"({', '.join(recorded)})")
        if got & {"indices", "anchor_missing", "anchor_late"}:
            for arm in ("centroid", "envelope"):
                if labels.get(arm) not in ABSTAIN_LABELS:
                    problems.append(f"registered label {labels.get(arm)!r} for {arm} without "
                                    f"the registered RANDOM reference ({', '.join(recorded)})")
    return registered


HEX64 = re.compile(r"^[0-9a-f]{64}$")
AUC_MACRO = re.compile(r"^RepAUC(RandomCB|Random|Centroid|Envelope)(Orig|Seed[ABC])$")
WORD_ARM = {v: k for k, v in ARM_WORD.items()}
POSTER_ROWS = ("random", "envelope", "centroid")

# Figures: poster-only annotated OCT and clean proposal schematics, plus the I13 framework
# (CC BY 4.0 OCTDL crop; no FairVision pixels), read from figures_v2/poster and checked
# against the SHA-256 that autopilot/release_assets.json records for them (poster_variant);
# the generated seed-spread plot and its poster render (no images).
REPO_ROOT = HERE / ".." / ".."
POSTER_VARIANT_DIR = "autopilot/investigations/camera_ready_20261008/figures_v2/poster"
REGISTRY = "autopilot/release_assets.json"
PAPER_FIGURES = ("fig_oct_proposals_poster.pdf", "fig_oct_structure_poster.pdf",
                 "fig_oct_jepa_volume_v2_poster.pdf")
ALLOWED_FIGURES = set(PAPER_FIGURES) | {"fig_cr_seed_spread.pdf", POSTER_FIG_NAME}


class GateError(Exception):
    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def md5_file(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest().upper()


def read_json(path, problems, what):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problems.append(f"{what} missing or unreadable ({exc})")
        return None


def parse_results_tex(text, problems):
    """Every non-comment line must be one \\newcommand{\\Rep...}{...}; returns name -> body."""
    defs = {}
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        m = re.match(r"^\\newcommand\{\\(Rep[A-Za-z]+)\}\{(.*)\}$", stripped)
        depth = 0
        for ch in (m.group(2) if m else ""):
            depth += {"{": 1, "}": -1}.get(ch, 0)
            if depth < 0:
                break
        if not m or depth != 0:
            problems.append(f"cr_results.tex line {number} is not a single \\Rep* definition")
            continue
        if m.group(1) in defs:
            problems.append(f"cr_results.tex defines \\{m.group(1)} twice")
        defs[m.group(1)] = m.group(2)
    return defs


def close(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)


def generator_check(results_dir):
    """Run the generator's own verified_results_metadata() if the generator is importable."""
    places = [os.environ.get("POSTER_GENERATOR_DIR"), str(HERE.parent.parent / "autopilot")]
    for place in places:
        if not place or not (pathlib.Path(place) / "make_cr_results.py").is_file():
            continue
        if place not in sys.path:
            sys.path.insert(0, place)
        try:
            import make_cr_results as mk  # noqa: PLC0415 (read-only use)
        except Exception as exc:  # missing dependency in this Python
            return f"unavailable (import failed: {type(exc).__name__})", None
        try:
            mk.verified_results_metadata(str(results_dir))
        except Exception as exc:
            return "failed", f"generator verified_results_metadata: {exc}"
        return "passed", None
    return "unavailable (autopilot/make_cr_results.py not found)", None


def validate_bundle(results_dir, use_generator=True):
    """None if there are no results; else the validated facts. Raises GateError."""
    rdir = pathlib.Path(results_dir)
    tex_path = rdir / "cr_results.tex"
    if not tex_path.is_file():
        return None
    problems = []
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    sidecar_path = rdir / "cr_results.values.json"
    side = read_json(sidecar_path, problems, "cr_results.values.json")
    if not isinstance(side, dict):
        problems.append("cr_results.values.json is not a JSON object")
        raise GateError(problems)
    if side.get("schema") != SCHEMA:
        problems.append(f"cr_results.values.json schema {side.get('schema')!r} != {SCHEMA}")
    synthetic = side.get("synthetic")
    if not isinstance(synthetic, bool):
        problems.append("cr_results.values.json has no boolean 'synthetic' flag")
    marked = SYNTHETIC_MARKER in tex
    if isinstance(synthetic, bool) and marked != synthetic:
        problems.append("synthetic marker in cr_results.tex disagrees with the sidecar flag")

    # Output and figure hashes.
    outputs = side.get("outputs")
    hashes = {}
    if not isinstance(outputs, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                and HEX64.match(v) for k, v in outputs.items()):
        problems.append("cr_results.values.json has no valid 'outputs' hash map")
    else:
        missing = [n for n in REQUIRED_OUTPUTS if n not in outputs]
        if missing:
            problems.append(f"sidecar outputs lack {missing}")
        hashes.update(outputs)
    figure = side.get("figure", "missing")
    if figure == "missing":
        problems.append("cr_results.values.json has no 'figure' field")
        figure = None
    elif figure is not None:
        if not (isinstance(figure, dict) and set(figure) == {"png", "pdf"}
                and all(isinstance(v, str) and HEX64.match(v) for v in figure.values())):
            problems.append("cr_results.values.json has malformed figure hashes")
            figure = None
        else:
            hashes["fig_cr_seed_spread.png"] = figure["png"]
            hashes["fig_cr_seed_spread.pdf"] = figure["pdf"]
    if figure is None and (rdir / "fig_cr_seed_spread.pdf").exists():
        problems.append("fig_cr_seed_spread.pdf is present but the sidecar records no figure")
    for name, digest in sorted(hashes.items()):
        path = rdir / name
        if not path.is_file():
            problems.append(f"{name} listed in the sidecar is missing")
        elif sha256_file(path) != digest:
            problems.append(f"{name} does not match its recorded SHA-256")

    # Scenario and matched control.
    scen = side.get("scenario")
    labels, complete, mc = {}, [], {}
    if not isinstance(scen, dict):
        problems.append("cr_results.values.json has no 'scenario' object")
    else:
        labels = scen.get("labels") if isinstance(scen.get("labels"), dict) else {}
        for arm in ("centroid", "envelope"):
            if labels.get(arm) not in LABEL_TEX:
                problems.append(f"scenario label for {arm} is {labels.get(arm)!r}")
        complete = scen.get("complete_seed_indices")
        if not (isinstance(complete, list) and all(s in SEED_LETTERS for s in complete)):
            problems.append("scenario.complete_seed_indices missing or invalid")
            complete = []
        mc = scen.get("matched_control") if isinstance(scen.get("matched_control"), dict) else {}
        if mc.get("label") not in KNOWN_CONTROL:
            problems.append(f"matched-control label {mc.get('label')!r} unknown to the poster")
        for key in ("cb_present", "cb_eligible"):
            if not isinstance(mc.get(key), bool):
                problems.append(f"matched_control.{key} missing or not boolean")
        for value, flag in (("P", "P_descriptive"), ("B", "B_descriptive")):
            if mc.get(value) is None:
                if mc.get(flag) is not None:
                    problems.append(f"matched_control.{flag} set without {value}")
            elif not isinstance(mc.get(value), (int, float)) or not isinstance(mc.get(flag), bool):
                problems.append(f"matched_control.{value}/{flag} malformed")

    # Macro records versus cr_results.tex.
    numeric, fullp, texts = {}, {}, {}
    records = side.get("macros")
    if not isinstance(records, list):
        problems.append("cr_results.values.json has no 'macros' list")
        records = []
    for rec in records:
        if not (isinstance(rec, dict) and isinstance(rec.get("name"), str)
                and isinstance(rec.get("display"), str)):
            problems.append("malformed macro record in the sidecar")
            continue
        numeric[rec["name"]] = rec["display"]
        fp = rec.get("full_precision")
        if isinstance(fp, list) and fp and all(number(v) for v in fp):
            fullp[rec["name"]] = fp[0]
            expr = rec.get("expression")
            if isinstance(expr, dict) and expr.get("op") == "format":
                try:
                    shown = expr["format"] % tuple(fp)
                except (KeyError, TypeError, ValueError):
                    shown = None
                if shown is None or rec["display"].replace("{,}", "") != shown:
                    problems.append(f"sidecar display of \\{rec['name']} {rec['display']!r} is "
                                    f"not its recorded format of the full-precision value "
                                    f"({shown!r})")
            else:
                problems.append(f"sidecar macro \\{rec['name']} has no format expression")
        else:
            problems.append(f"sidecar macro \\{rec['name']} has no finite full precision")
    trecs = side.get("text_macros")
    if not isinstance(trecs, list):
        problems.append("cr_results.values.json has no 'text_macros' list")
        trecs = []
    for rec in trecs:
        if not (isinstance(rec, dict) and isinstance(rec.get("name"), str)
                and isinstance(rec.get("text"), str)):
            problems.append("malformed text-macro record in the sidecar")
            continue
        texts[rec["name"]] = rec["text"]
    defs = parse_results_tex(tex, problems)
    expected = dict(numeric, **texts)
    macro_problems = []
    for name in sorted(set(expected) - set(defs)):
        macro_problems.append(f"\\{name} is in the sidecar but not defined in cr_results.tex")
    for name in sorted(set(defs) - set(expected)):
        macro_problems.append(f"\\{name} is defined in cr_results.tex but not in the sidecar")
    for name in sorted(set(defs) & set(expected)):
        if defs[name] != expected[name]:
            macro_problems.append(f"\\{name} = {defs[name]!r} disagrees with the sidecar "
                                  f"{expected[name]!r}")
    problems += macro_problems[:12]
    if len(macro_problems) > 12:
        problems.append(f"... and {len(macro_problems) - 12} more macro disagreements")
    for name in SENTINELS:
        if name not in defs:
            problems.append(f"sentinel \\{name} missing")
    for arm in ("centroid", "envelope"):
        name = "RepLabel" + ARM_WORD[arm]
        want = LABEL_TEX.get(labels.get(arm))
        if want is not None and texts.get(name) != want:
            problems.append(f"\\{name} {texts.get(name)!r} != sidecar label {want!r}")
    for value, name in (("P", "RepPlacementDelta"), ("B", "RepBudgetShift")):
        have = mc.get(value) is not None
        if have != (name in numeric):
            problems.append(f"\\{name} {'missing' if have else 'present'} but sidecar "
                            f"{value} is {mc.get(value)!r}")
        elif have and name in fullp and not close(fullp[name], mc.get(value)):
            problems.append(f"\\{name} full precision disagrees with sidecar {value}")
    registered = reference_state(side, scen, mc, complete, numeric, fullp, labels, problems) \
        if isinstance(scen, dict) else False
    rule = control_rule(mc, registered)
    if rule != mc.get("label"):
        problems.append(f"matched-control label {mc.get('label')!r} != {rule!r} from the "
                        f"pre-registered rule applied to the sidecar's P, B, registered RANDOM "
                        f"reference and eligibility")
    if mc.get("name") != CONTROL_NAMES.get(mc.get("label")):
        problems.append(f"matched-control name {mc.get('name')!r} does not belong to "
                        f"{mc.get('label')!r}")

    # Seeds.
    runs = {}
    for name in numeric:
        m = AUC_MACRO.match(name)
        if m:
            runs[(WORD_ARM[m.group(1)], m.group(2))] = name
    present = {tag[-1] for (arm, tag) in runs if arm in POSTER_ROWS and tag != "Orig"}
    complete_l = {SEED_LETTERS[s] for s in complete}
    if not complete_l <= present:
        problems.append("a complete seed index has no AUC macros")
    desc = present - complete_l
    late_orig = late_originals(rdir, hashes, runs, scen, problems)

    # Figure evidence.
    evidence, series = None, None
    if figure is not None:
        evidence = read_json(rdir / "fig_cr_seed_spread.evidence.json", problems,
                             "fig_cr_seed_spread.evidence.json (required with a figure)")
        if evidence is not None:
            series = figure_series(runs, fullp, numeric, complete, mc, problems)
            series["late_originals"] = late_orig
            evidence = check_evidence(evidence, side, figure, series, problems)

    gen_status, gen_problem = ("skipped", None)
    if use_generator and not problems:
        gen_status, gen_problem = generator_check(rdir)
        if gen_problem:
            problems.append(gen_problem)
    if problems:
        raise GateError(problems)
    return {
        "present": True, "synthetic": synthetic, "dir": str(rdir.resolve()),
        "tex_sha256": sha256_file(tex_path), "tex_md5": md5_file(tex_path),
        "sidecar_sha256": sha256_file(sidecar_path), "numeric": numeric, "texts": texts,
        "full_precision": fullp,
        "labels": labels, "complete": sorted(complete_l), "present_seeds": sorted(present),
        "desc_seeds": sorted(desc), "control": mc, "figure": figure, "evidence": evidence,
        "late_originals": late_orig,
        "series": series if evidence is not None else None,
        "evidence_sha256": sha256_file(rdir / "fig_cr_seed_spread.evidence.json")
        if evidence is not None else None,
        "n_hashes": len(hashes), "generator_check": gen_status,
    }


TAG_ORDER = ("Orig", "SeedA", "SeedB", "SeedC")
SEEDS_TABLE = "cr_table_seeds.tex"
ORIG_STATUS = {"original": False, "original, late": True}
ORIG_ROW = re.compile(r"\\RepAUC(RandomCB|Random|Centroid|Envelope)Orig(?![A-Za-z])")
LATE_ORIGINAL_NOTE = "\u2020original re-probed after the freeze"


def late_originals(rdir, hashes, runs, scen, problems):
    """Re-probed originals that finished after the freeze (generator V14-1), read from the
    hash-checked per-run table (tab:replication_runs, analysis column "original" or
    "original, late"); every original AUC macro needs exactly one such row, and the original
    RANDOM is late exactly when the sidecar's reference_missing says anchor_late."""
    origs = {arm for (arm, tag) in runs if tag == "Orig"}
    seen = {}
    if SEEDS_TABLE in hashes and (rdir / SEEDS_TABLE).is_file():
        text = (rdir / SEEDS_TABLE).read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            m = ORIG_ROW.search(line)
            if not m:
                continue
            arm = WORD_ARM[m.group(1)]
            status = line.rstrip().rstrip("\\").split("&")[-1].strip()
            if status not in ORIG_STATUS:
                problems.append(f"{SEEDS_TABLE}: original {ARM_PLAIN[arm]} has analysis status "
                                f"{status!r} (expected 'original' or 'original, late')")
            elif arm in seen:
                problems.append(f"{SEEDS_TABLE}: original {ARM_PLAIN[arm]} listed twice")
            else:
                seen[arm] = ORIG_STATUS[status]
        if set(seen) != origs:
            problems.append(f"{SEEDS_TABLE} lists originals {sorted(seen)} but the sidecar has "
                            f"original AUC macros for {sorted(origs)}")
    elif origs:
        problems.append(f"{SEEDS_TABLE} (hash-checked per-run status) is required to tell late "
                        f"originals")
    late = {arm for arm, is_late in seen.items() if is_late}
    if isinstance(scen, dict) and "random" in origs:
        anchor_late = "anchor_late" in (scen.get("reference_missing") or [])
        if anchor_late != ("random" in late):
            problems.append(f"{SEEDS_TABLE} marks the original RANDOM "
                            f"{'late' if 'random' in late else 'on time'} but reference_missing "
                            f"says {'anchor_late' if anchor_late else 'not late'}")
    return [arm for arm in FIG_ARMS if arm in late]
LETTER_SEED = {v: k for k, v in SEED_LETTERS.items()}
POINT_KEYS = {"run", "arm", "tag", "seed", "x", "test_auc", "head_sd", "status"}
MEAN_KEYS = {"x", "mean", "n"}
PANEL_B_KEYS = {"arms", "points", "means", "band"}
B_POINT_KEYS = {"arm", "tag", "seed", "x", "delta", "status", "contrast"}
B_MEAN_KEYS = {"x", "mean_delta", "n"}
BAND_KEYS = {"drawn", "r", "missing"}
DELTA_MACRO = re.compile(r"^RepDelta(Centroid|Envelope)(Seed[ABC])$")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(value)


def figure_series(runs, fullp, numeric, complete, mc, problems):
    """The seed-spread series built from the validated sidecar alone: one point per AUC macro,
    one mean per mean macro (the generator's figure_points rules). The poster render plots
    exactly this; the evidence file must equal it."""
    arms = [a for a in FIG_ARMS if any(arm == a for (arm, _) in runs)]
    points = []
    for arm, tag in sorted(runs, key=lambda k: (FIG_ARMS.index(k[0]), TAG_ORDER.index(k[1]))):
        name = runs[(arm, tag)]
        sd_name = "RepHeadSD" + name[len("RepAUC"):]
        if not number(fullp.get(name)) or (sd_name in numeric and not number(fullp.get(sd_name))):
            problems.append(f"no finite full-precision value for \\{name} or its s.d.")
            continue
        if tag == "Orig":
            status = "original"
        elif arm == "random_cb":
            status = "control" if mc.get("label") not in ("M_desc", "M_na") and \
                mc.get("cb_eligible") else "descriptive"
        else:
            status = "registered" if LETTER_SEED[tag[-1]] in complete else "descriptive"
        points.append({"arm": arm, "tag": tag,
                       "seed": None if tag == "Orig" else LETTER_SEED[tag[-1]],
                       "x": arms.index(arm) + FIG_OFFSETS[tag], "test_auc": fullp[name],
                       "head_sd": fullp.get(sd_name) if sd_name in numeric else None,
                       "status": status})
    means = {}
    for arm in ("random", "centroid", "envelope"):
        name = "RepMeanAUC" + ARM_WORD[arm]
        if name in numeric:
            if arm not in arms or not number(fullp.get(name)):
                problems.append(f"\\{name} has no plottable arm or full precision")
                continue
            means[arm] = {"x": arms.index(arm), "mean": fullp[name], "n": len(complete)}
    # Panel (b), the generator's figure_panel_b rules: one point per paired difference
    # \RepDelta<Arm>Seed<X> (registered iff the seed index is complete), one mean per
    # \RepDeltaMean<Arm> over the complete indices, the band +/- \RepRandomRange only with the
    # registered RANDOM reference (the macro exists only then).
    deltas = {}
    for name in numeric:
        m = DELTA_MACRO.match(name)
        if m:
            deltas[(WORD_ARM[m.group(1)], m.group(2))] = name
    bpoints = []
    for arm, tag in sorted(deltas, key=lambda k: (GUIDED.index(k[0]), TAG_ORDER.index(k[1]))):
        name = deltas[(arm, tag)]
        if not number(fullp.get(name)):
            problems.append(f"no finite full-precision value for \\{name}")
            continue
        seed = LETTER_SEED[tag[-1]]
        bpoints.append({"arm": arm, "tag": tag, "seed": seed,
                        "x": GUIDED.index(arm) + FIG_B_OFFSETS[tag], "delta": fullp[name],
                        "status": "registered" if seed in complete else "descriptive"})
    bmeans = {}
    for arm in GUIDED:
        name = "RepDeltaMean" + ARM_WORD[arm]
        if name in numeric:
            if not number(fullp.get(name)):
                problems.append(f"no finite full-precision value for \\{name}")
                continue
            bmeans[arm] = {"x": GUIDED.index(arm), "mean_delta": fullp[name], "n": len(complete)}
    r = fullp.get("RepRandomRange") if "RepRandomRange" in numeric else None
    panel_b = {"arms": list(GUIDED), "points": bpoints, "means": bmeans,
               "band": {"drawn": r is not None, "r": r,
                        "missing": list(mc.get("reference_missing") or [])}} if bpoints else None
    return {"arms": arms, "points": points, "means": means, "panel_b": panel_b}


def check_panel_b(evb, want, problems):
    """Evidence panel (b) must equal the sidecar-built panel exactly (or both be absent)."""
    if want is None or evb is None:
        if (want is None) != (evb is None):
            problems.append("figure evidence panel (b) " + ("present without" if want is None
                            else "missing despite") + " paired-difference macros in the sidecar")
        return
    if not isinstance(evb, dict) or set(evb) != PANEL_B_KEYS:
        problems.append(f"figure evidence panel (b) keys "
                        f"{sorted(evb) if isinstance(evb, dict) else evb!r} != "
                        f"{sorted(PANEL_B_KEYS)}")
        return
    if evb["arms"] != want["arms"]:
        problems.append(f"figure panel (b) arms {evb['arms']} != {want['arms']}")
    wp = {(q["arm"], q["tag"]): q for q in want["points"]}
    seen = set()
    for p in evb["points"] if isinstance(evb["points"], list) else [None]:
        if not isinstance(p, dict) or set(p) != B_POINT_KEYS:
            problems.append(f"figure panel (b) point {p!r} has keys other than "
                            f"{sorted(B_POINT_KEYS)}")
            continue
        key = (p["arm"], p["tag"])
        if key not in wp:
            problems.append(f"figure panel (b) point {key} has no paired-difference macro in "
                            f"the sidecar (extra point)")
            continue
        if key in seen:
            problems.append(f"figure panel (b) point {key} appears twice")
            continue
        seen.add(key)
        q = wp[key]
        if not (number(p["delta"]) and close(p["delta"], q["delta"])):
            problems.append(f"figure panel (b) point {key} difference disagrees with the sidecar")
        if not close(p["x"], q["x"]) or p["status"] != q["status"] or p["seed"] != q["seed"]:
            problems.append(f"figure panel (b) point {key} position, status or seed disagrees")
        c = p["contrast"]
        if not (isinstance(c, list) and len(c) == 2 and isinstance(c[0], str)
                and type(c[1]) is int):
            problems.append(f"figure panel (b) point {key} has a malformed contrast reference")
    missing = sorted(set(wp) - seen)
    if missing:
        problems.append(f"figure panel (b) lacks points {missing}")
    means = evb["means"] if isinstance(evb["means"], dict) else {}
    if set(means) != set(want["means"]):
        problems.append(f"figure panel (b) means {sorted(means)} != sidecar mean-difference "
                        f"macros {sorted(want['means'])} (extra or missing mean)")
    for arm, q in want["means"].items():
        m = means.get(arm)
        if m is None:
            continue
        if not isinstance(m, dict) or set(m) != B_MEAN_KEYS or not number(m["mean_delta"]) \
                or not close(m["mean_delta"], q["mean_delta"]) or not close(m["x"], q["x"]) \
                or type(m["n"]) is not int or m["n"] != q["n"]:
            problems.append(f"figure panel (b) mean for {arm} {m!r} disagrees with the sidecar")
    band, wb = evb["band"], want["band"]
    if not isinstance(band, dict) or set(band) != BAND_KEYS or band["drawn"] is not wb["drawn"] \
            or band["missing"] != wb["missing"] or (wb["r"] is None) != (band["r"] is None) \
            or (wb["r"] is not None and not close(band["r"], wb["r"])):
        problems.append(f"figure panel (b) band {band!r} disagrees with the registered RANDOM "
                        f"reference of the sidecar {wb!r}")


HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
# Encoding fields of contract cr_seed_spread_encoding_v2 the poster knows: the ones it draws
# with (required), late_original (optional, V14-1) and descriptive text/size fields it does not
# use. Any other field is refused: it could change how something must be drawn.
ENCODING_DRAWN = {"contract", "colors", "markers", "fill_by_status", "seed_labels",
                  "x_offsets_panel_a", "x_offsets_panel_b", "band_color"}
ENCODING_OPTIONAL = {"late_original"}
ENCODING_DESCRIPTIVE = {"colors_source", "markers_source", "fill_meaning", "seed_encoding",
                        "mean_bar", "band", "figsize_in", "include_width", "include_width_in",
                        "min_font_pt"}
LATE_ORIGINAL_KEYS = {"runs", "fill", "label", "meaning"}


def check_encoding(enc, series, problems):
    """The generator's figure encoding (colours, markers, fill by status, seed letters,
    offsets, late originals), validated so the poster render can follow it; defaults if the
    evidence predates the contract. Presentation only: no value is read from it."""
    if enc is None:
        return dict(DEFAULT_ENCODING, source="poster default (evidence has no encoding)")
    n0 = len(problems)
    if not isinstance(enc, dict) or enc.get("contract") != FIG_ENCODING:
        problems.append(f"figure encoding contract "
                        f"{enc.get('contract') if isinstance(enc, dict) else enc!r} != "
                        f"{FIG_ENCODING} (update the poster render)")
        return None
    unknown = sorted(set(enc) - ENCODING_DRAWN - ENCODING_OPTIONAL - ENCODING_DESCRIPTIVE)
    if unknown:
        problems.append(f"figure encoding has unknown fields {unknown} (update the poster "
                        f"render)")
    missing = sorted(ENCODING_DRAWN - set(enc))
    if missing:
        problems.append(f"figure encoding lacks {missing}")
    late = enc.get("late_original")
    if late is not None and not (
            isinstance(late, dict) and set(late) <= LATE_ORIGINAL_KEYS
            and late.get("fill") in ("full", "left", "none")
            and isinstance(late.get("label"), str) and 1 <= len(late["label"]) <= 2):
        problems.append(f"figure encoding late_original {late!r} is malformed")
    if series.get("late_originals") and late is None:
        problems.append("figure encoding has no late_original entry although an original "
                        "finished after the freeze (cannot draw it)")
    arms = set(series["arms"]) | set((series.get("panel_b") or {}).get("arms", []))
    points = series["points"] + list((series.get("panel_b") or {}).get("points", []))
    statuses = {p["status"] for p in points}
    tags = {p["tag"] for p in points}
    colors, markers = enc.get("colors"), enc.get("markers")
    fills, labels = enc.get("fill_by_status"), enc.get("seed_labels")
    if not (isinstance(colors, dict) and arms <= set(colors)
            and all(isinstance(colors[a], str) and HEX_COLOR.match(colors[a]) for a in arms)):
        problems.append("figure encoding colours missing or not #RRGGBB for every arm")
    if not (isinstance(markers, dict) and arms <= set(markers)
            and all(markers[a] in ALLOWED_MARKERS for a in arms)):
        problems.append("figure encoding markers missing or unknown for an arm")
    if not (isinstance(fills, dict) and statuses <= set(fills)
            and all(fills[s] in ("full", "left", "none") for s in statuses)):
        problems.append("figure encoding fill_by_status missing or unknown for a status")
    if not (isinstance(labels, dict) and tags <= set(labels)
            and all(isinstance(labels[t], str) and 1 <= len(labels[t]) <= 2 for t in tags)):
        problems.append("figure encoding seed_labels missing or malformed")
    if enc.get("x_offsets_panel_a") != FIG_OFFSETS or enc.get("x_offsets_panel_b") != FIG_B_OFFSETS:
        problems.append("figure encoding x offsets differ from the poster's (update the render)")
    band = enc.get("band_color")
    try:
        grey_ok = isinstance(band, str) and (HEX_COLOR.match(band) or 0.0 <= float(band) <= 1.0)
    except ValueError:
        grey_ok = False
    if not grey_ok:
        problems.append(f"figure encoding band colour {band!r} is not a grey level or #RRGGBB")
    if len(problems) != n0:
        return None
    return {"contract": FIG_ENCODING, "colors": dict(colors), "markers": dict(markers),
            "fill_by_status": dict(fills), "seed_labels": dict(labels),
            "x_offsets_panel_a": dict(FIG_OFFSETS), "x_offsets_panel_b": dict(FIG_B_OFFSETS),
            "band_color": band,
            "late_original": {"fill": late["fill"], "label": late["label"]} if late
            else dict(FIG_LATE_ORIGINAL),
            "source": "figure evidence (generator contract)"}


def check_evidence(ev, side, figure, series, problems):
    """The evidence must equal the constructed series exactly: same arms, the same set of
    points and means (nothing extra, nothing missing), same keys and values."""
    n0 = len(problems)
    if not isinstance(ev, dict):
        problems.append("figure evidence is not a JSON object")
        return None
    if ev.get("schema") != FIGURE_SCHEMA:
        problems.append(f"figure evidence schema {ev.get('schema')!r} != {FIGURE_SCHEMA}")
    if ev.get("synthetic") is not side.get("synthetic"):
        problems.append("figure evidence synthetic flag disagrees with the sidecar")
    if ev.get("outputs") != figure:
        problems.append("figure evidence outputs disagree with the sidecar figure hashes")
    points, means, arms = ev.get("points"), ev.get("means"), ev.get("arms")
    if not isinstance(points, list) or not isinstance(means, dict) or not isinstance(arms, list):
        problems.append("figure evidence lacks points/means/arms")
        return None
    if arms != series["arms"]:
        problems.append(f"figure evidence arms {arms} != {series['arms']}")
    want = {(q["arm"], q["tag"]): q for q in series["points"]}
    seen = set()
    for p in points:
        if not isinstance(p, dict) or set(p) != POINT_KEYS:
            problems.append(f"figure point with keys {sorted(p) if isinstance(p, dict) else p!r} "
                            f"!= {sorted(POINT_KEYS)}")
            continue
        key = (p["arm"], p["tag"])
        if key not in want:
            problems.append(f"figure point {key} has no AUC macro in the sidecar (extra point)")
            continue
        if key in seen:
            problems.append(f"figure point {key} appears twice")
            continue
        seen.add(key)
        q = want[key]
        if not isinstance(p["run"], str):
            problems.append(f"figure point {key} has no run name")
        if not close(p["test_auc"], q["test_auc"]) or not number(p["test_auc"]):
            problems.append(f"figure point {key} AUC disagrees with the sidecar")
        if (p["head_sd"] is None) != (q["head_sd"] is None) or \
                (q["head_sd"] is not None and not close(p["head_sd"], q["head_sd"])):
            problems.append(f"figure point {key} s.d. disagrees with the sidecar")
        if not close(p["x"], q["x"]):
            problems.append(f"figure point {key} x position disagrees")
        if p["status"] != q["status"]:
            problems.append(f"figure point {key} status {p['status']!r} != {q['status']!r}")
        if p["seed"] != q["seed"]:
            problems.append(f"figure point {key} seed disagrees with its tag")
    missing = sorted(set(want) - seen)
    if missing:
        problems.append(f"figure evidence lacks points {missing}")
    if set(means) != set(series["means"]):
        problems.append(f"figure evidence means {sorted(means)} != sidecar mean macros "
                        f"{sorted(series['means'])} (extra or missing mean)")
    for arm, q in series["means"].items():
        m = means.get(arm)
        if m is None:
            continue
        if not isinstance(m, dict) or set(m) != MEAN_KEYS or not number(m.get("mean")) \
                or not close(m["mean"], q["mean"]) or not close(m["x"], q["x"]) \
                or type(m["n"]) is not int or m["n"] != q["n"]:
            problems.append(f"figure mean for {arm} {m!r} disagrees with the sidecar {q!r}")
    if "panel_b" in ev:
        check_panel_b(ev["panel_b"], series["panel_b"], problems)
    else:
        series["panel_b"] = None  # evidence before panel (b): the poster draws panel (a) only
    check_late_originals(ev, side, series, problems)
    series["encoding"] = check_encoding(ev.get("encoding"), series, problems)
    return ev if len(problems) == n0 else None


def check_late_originals(ev, side, series, problems):
    """Evidence late_originals (run names) must name exactly the originals that the
    hash-checked per-run table marks late; absent only when none is late."""
    want = set(series.get("late_originals") or [])
    names = ev.get("late_originals")
    if names is None:
        if want:
            problems.append(f"figure evidence has no late_originals although the per-run table "
                            f"marks {sorted(want)} late")
        return
    runs = side.get("probe_runs") if isinstance(side.get("probe_runs"), dict) else {}
    origin = {q["arm"] for q in series["points"] if q["tag"] == "Orig"}
    got = []
    if not isinstance(names, list):
        problems.append(f"figure evidence late_originals {names!r} is not a list")
        return
    for name in names:
        ident = (runs.get(name) or {}).get("identity") if isinstance(name, str) else None
        if not (isinstance(ident, dict) and ident.get("role") == "anchor"
                and ident.get("arm") in origin):
            problems.append(f"figure evidence late original {name!r} is not a re-probed "
                            f"original run of the sidecar")
            continue
        got.append(ident["arm"])
    if len(got) != len(set(got)):
        problems.append("figure evidence late_originals lists an original twice")
    if set(got) != want:
        problems.append(f"figure evidence late originals {sorted(set(got))} != per-run table "
                        f"{sorted(want)}")


def render_poster_figure(bundle, out_pdf):
    """Poster-size render of the series built from the validated sidecar (equal to the checked
    evidence), drawn with the generator's figure encoding (validated contract
    cr_seed_spread_encoding_v2: arm colours and markers, fill by status, seed letters under
    the points): (a) test AUC of every encoder with head-seed s.d. and the registered means,
    (b) the paired differences guided minus RANDOM of the same seed index, their registered
    means and the band +/- the registered RANDOM range (only with that reference). A
    re-probed original that finished after the freeze (hash-checked per-run table, equal to
    the evidence late_originals) is drawn with the encoding's late_original fill (open). Nothing
    from the evidence file is plotted; only the encoding (presentation) is read from it."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter, MaxNLocator
    except ImportError:
        return None
    series, enc = bundle["series"], bundle["series"]["encoding"]
    arms, points, means, pb = (series["arms"], series["points"], series["means"],
                               series.get("panel_b"))
    colors, markers = enc["colors"], enc["markers"]
    fills, letters = enc["fill_by_status"], enc["seed_labels"]
    late = set(series.get("late_originals") or [])

    def fill_of(p):  # V14-1: a late original is drawn with the late_original fill (open)
        if p["status"] == "original" and p["arm"] in late:
            return enc["late_original"]["fill"]
        return fills[p["status"]]
    fs = 14
    plt.rcParams.update({"font.size": fs, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "DejaVu Sans", "axes.linewidth": 1.0,
                         "axes.titlesize": fs, "axes.labelsize": fs, "xtick.labelsize": fs,
                         "ytick.labelsize": fs, "legend.fontsize": fs})
    if pb:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.1, 3.4), gridspec_kw={
            "width_ratios": [len(arms) + 0.15, len(pb["arms"]) + 0.15]})
    else:
        fig, ax = plt.subplots(figsize=(8.1, 3.4))
        bx = None

    def draw(axis, x, y, arm, fill, yerr=None):
        color = colors[arm]
        if yerr is not None:
            axis.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.3,
                          capsize=3, zorder=2)
        axis.plot(x, y, marker=markers[arm], markersize=9.5, linestyle="none",
                  fillstyle="full" if fill == "none" else fill,
                  markerfacecolor="white" if fill == "none" else color,
                  markerfacecoloralt="white", markeredgecolor=color, markeredgewidth=1.5,
                  zorder=3)

    def x_labels(axis, groups, seeds):
        axis.xaxis.remove_overlapping_locs = False  # keep the letter at a group centre
        axis.set_xticks(range(len(groups)))
        axis.set_xticklabels([ARM_PLAIN[g].replace("-", "-\n") for g in groups],
                             linespacing=0.95, fontsize=fs - 1)
        axis.set_xticks([x for x, _ in seeds], minor=True)
        axis.set_xticklabels([letter for _, letter in seeds], minor=True)
        axis.tick_params(axis="x", which="minor", length=0, pad=2, labelcolor="0.3",
                         labelsize=fs - 2)
        axis.tick_params(axis="x", which="major", length=0, pad=fs + 4)
        axis.set_xlim(-0.55, len(groups) - 0.45)
        axis.grid(axis="y", linewidth=0.6, alpha=0.5)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    for p in points:
        draw(ax, p["x"], p["test_auc"], p["arm"], fill_of(p), p["head_sd"])
    for arm, mean in means.items():
        ax.plot([mean["x"] - 0.34, mean["x"] + 0.34], [mean["mean"]] * 2, color=colors[arm],
                linewidth=2.2, zorder=1)
    x_labels(ax, arms, [(p["x"], letters[p["tag"]]) for p in points])
    ax.set_ylabel("test AUC (epoch %s)" % bundle["numeric"]["RepEndpoint"])
    ax.set_title("(a) every encoder", loc="left")
    if bx is not None:
        band = pb["band"]
        if band["drawn"]:
            bx.axhspan(-band["r"], band["r"], color=enc["band_color"], linewidth=0, zorder=0)
        bx.axhline(0.0, color="0.45", linewidth=1.0, zorder=1)
        for q in pb["points"]:
            draw(bx, q["x"], q["delta"], q["arm"], fills[q["status"]])
        for arm, mean in pb["means"].items():
            bx.plot([mean["x"] - 0.3, mean["x"] + 0.3], [mean["mean_delta"]] * 2,
                    color=colors[arm], linewidth=2.2, zorder=2)
        top = max([abs(q["delta"]) for q in pb["points"]] +
                  [band["r"] if band["drawn"] else 0.0])
        bx.set_ylim(-1.25 * top, 1.25 * top)
        x_labels(bx, pb["arms"], [(q["x"], letters[q["tag"]]) for q in pb["points"]])
        bx.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-12 else ("%+.3f" % v).replace("-", "\u2212")))

        bx.set_title("(b) \u0394 vs RANDOM", loc="left")
    shown = {fill_of(p) for p in points} | {fills[q["status"]] for q in (pb or {}).get("points", [])}
    handles = [Line2D([], [], marker="o", color="0.3", fillstyle="full", linestyle="none",
                      markersize=9, label="registered")]
    if "left" in shown:
        handles.append(Line2D([], [], marker="o", color="0.3", fillstyle="left",
                              markerfacecoloralt="white", linestyle="none", markersize=9,
                              label="original (re-probed)"))
    if "none" in shown:
        only_late = late and not any(p["status"] == "descriptive" for p in points)
        handles.append(Line2D([], [], marker="o", color="0.3", markerfacecolor="white",
                              linestyle="none", markersize=9,
                              label="descriptive (late)" if only_late else "descriptive"))
    handles.append(Line2D([], [], color="0.3", linewidth=2.2, label="mean (registered)"))
    if pb and pb["band"]["drawn"]:
        handles.append(Patch(facecolor=enc["band_color"], edgecolor="none",
                             label="\u00b1 RANDOM range"))
    rows = 2 if len(handles) > 3 else 1
    fig.legend(handles=handles, frameon=False, loc="lower center",
               ncol=math.ceil(len(handles) / rows), handletextpad=0.3, columnspacing=1.0,
               borderaxespad=0.1)
    fig.tight_layout(pad=0.3, rect=(0, 0.12 * rows, 1, 1), w_pad=0.6)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)
    return {"matplotlib": matplotlib.__version__, "panels": "a+b" if bx is not None else "a",
            "encoding": enc["source"], "late_originals_open": sorted(late)}


def clear_state():
    for path in (STATE_TEX, STATE_JSON, POSTER_FIG, PP_TEX):
        if path.exists():
            path.unlink()


def write_state(bundle):
    """generated/results_state.json|tex from the validated bundle (+ the poster figure)."""
    GENERATED.mkdir(exist_ok=True)
    pp = poster_pp_values(bundle)
    PP_TEX.write_text("% Display-only AUC differences in percentage points; source values unchanged.\n" +
                      "\n".join(r"\newcommand{\PosterPP%s}{%s}" % (key, value)
                                for key, value in sorted(pp.items())) + "\n", encoding="ascii")
    figure = None
    if bundle["figure"] is not None and bundle["evidence"] is not None:
        info = render_poster_figure(bundle, POSTER_FIG)
        if info is not None:
            figure = {"kind": "poster render of the validated evidence",
                      "tex_path": "generated/" + POSTER_FIG_NAME, "file": str(POSTER_FIG),
                      "sha256": sha256_file(POSTER_FIG), "md5": md5_file(POSTER_FIG),
                      "evidence_sha256": bundle["evidence_sha256"], **info}
        else:
            src = pathlib.Path(bundle["dir"]) / "fig_cr_seed_spread.pdf"
            figure = {"kind": "generator figure (matplotlib unavailable for the poster render)",
                      "tex_path": "../../paper/genai4health2026/auto/fig_cr_seed_spread.pdf",
                      "file": str(src), "sha256": sha256_file(src), "md5": md5_file(src)}
    mc = bundle["control"]

    def flag(value):
        return "n" if value is None else ("1" if value else "0")

    state = {"schema": STATE_SCHEMA, "results_dir": bundle["dir"],
             "difference_units": "percentage_points", "pp_values": pp,
             "units_sha256": sha256_file(PP_TEX), "units_md5": md5_file(PP_TEX),
             "cr_results_sha256": bundle["tex_sha256"], "cr_results_md5": bundle["tex_md5"],
             "sidecar_sha256": bundle["sidecar_sha256"], "synthetic": bundle["synthetic"],
             "control": mc["label"], "cb_present": mc["cb_present"],
             "cb_eligible": mc["cb_eligible"], "P_descriptive": mc.get("P_descriptive"),
             "B_descriptive": mc.get("B_descriptive"),
             "reference_missing": mc.get("reference_missing"),
             "late_originals": bundle["late_originals"],
             "seeds_present": bundle["present_seeds"],
             "seeds_descriptive": bundle["desc_seeds"], "figure": figure,
             "figure_series": {"points": [[q["arm"], q["tag"]] for q in bundle["series"]["points"]],
                               "means": sorted(bundle["series"]["means"]),
                               "panel_b_points": [[q["arm"], q["tag"]] for q in
                                                  (bundle["series"]["panel_b"] or {})
                                                  .get("points", [])],
                               "panel_b_means": sorted((bundle["series"]["panel_b"] or {})
                                                       .get("means", {})),
                               "panel_b_band": ((bundle["series"]["panel_b"] or {})
                                                .get("band") or {}).get("drawn"),
                               "encoding": bundle["series"]["encoding"]}
             if bundle["series"] else None,
             "expected_macros": sorted(bundle["numeric"]) + sorted(bundle["texts"])}
    STATE_JSON.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
    lines = ["% Generated by make_preview.py --precheck from the validated results bundle;",
             "% do not edit. poster.tex re-checks the MD5s below.",
             r"\PosterStateDef{ResultsMDfive}{%s}" % bundle["tex_md5"],
             r"\PosterStateDef{UnitsMDfive}{%s}" % md5_file(PP_TEX),
             r"\PosterStateDef{Synthetic}{%s}" % flag(bundle["synthetic"]),
             r"\PosterStateDef{Control}{%s}" % mc["label"],
             r"\PosterStateDef{CBPresent}{%s}" % flag(mc["cb_present"]),
             r"\PosterStateDef{CBEligible}{%s}" % flag(mc["cb_eligible"]),
             r"\PosterStateDef{PDesc}{%s}" % flag(mc.get("P_descriptive")),
             r"\PosterStateDef{BDesc}{%s}" % flag(mc.get("B_descriptive")),
             r"\PosterStateDef{SeedsPresent}{%s}" % "".join(bundle["present_seeds"]),
             r"\PosterStateDef{SeedsDesc}{%s}" % "".join(bundle["desc_seeds"]),
             r"\PosterStateDef{LateOrig}{,%s,}" % ",".join(ARM_WORD[a] for a in
                                                         bundle["late_originals"]),
             r"\PosterStateDef{Figure}{%s}" % (figure["tex_path"] if figure else ""),
             r"\PosterStateDef{FigureMDfive}{%s}" % (figure["md5"] if figure else "")]
    lines += [r"\PosterExpectMacro{%s}" % name for name in state["expected_macros"]]
    STATE_TEX.write_text("\n".join(lines) + "\n", encoding="ascii")
    return state


def gate(results_dir, allow_synthetic, label):
    """Validate; print the results line(s). Returns (bundle or None, problems)."""
    head = f"{label:<15}"
    try:
        bundle = validate_bundle(results_dir)
    except GateError as exc:
        print(f"{head}auto/cr_results.tex: INVALID results bundle")
        return None, exc.problems
    if bundle is None:
        print(f"{head}none: auto/cr_results.tex not found (placeholders)")
        return None, []
    kind = "SYNTHETIC DRY RUN" if bundle["synthetic"] else "real"
    fig = "figure + evidence" if bundle["evidence"] is not None else "no figure"
    print(f"{head}auto/cr_results.tex: {kind}; validated {len(bundle['numeric'])} numeric + "
          f"{len(bundle['texts'])} text macros, {bundle['n_hashes']} file hashes, {fig}; "
          f"generator check {bundle['generator_check']}")
    problems = []
    if bundle["synthetic"]:
        if allow_synthetic:
            print("               SYNTHETIC DRY RUN allowed (-AllowSynthetic): test output only, "
                  "never print, share or commit it")
        else:
            problems.append("synthetic dry-run results in auto/ (build.ps1 -AllowSynthetic only "
                            "in a scratch copy)")
    return bundle, problems


# ---------------------------------------------------------------------------
# PDF checks (report mode)
# ---------------------------------------------------------------------------

def norm(text):
    return (text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "---")
            .replace("\u2217", "*"))


def bottom_words(page):
    words = [w for w in page.get_text("words") if w[1] > page.rect.height * 0.70]
    return [(w[0], w[1], w[2], w[3], norm(w[4])) for w in words]


def row_tokens(words, anchor, x_lo, x_hi):
    """Tokens of the table row whose first word is ``anchor`` (sorted by x; superscripts and
    the following value glued when they touch)."""
    y0, y1 = anchor[1], anchor[3]
    h = y1 - y0
    row = sorted((w for w in words if x_lo <= w[0] < x_hi and y0 < (w[1] + w[3]) / 2 < y1
                  and w is not anchor), key=lambda w: w[0])
    tokens = []
    for w in row:
        if tokens and w[0] - tokens[-1][1] < 0.12 * h:
            tokens[-1] = (tokens[-1][0] + w[4], w[2])
        else:
            tokens.append((w[4], w[2]))
    return [t[0] for t in tokens]


def squash(text):
    """Whitespace- and hyphen-free form, robust to line breaks inside phrases."""
    return re.sub(r"[\s\-]", "", norm(text))


def pdf_results_check(page, bundle):
    """Compare every result cell, label and late-run mark in the PDF with the sidecar."""
    problems, slots = [], 0
    words = bottom_words(page)
    heads = {}
    for key, word in (("rep", "Repeated"), ("ctl", "Matched-budget"), ("pool", "Pooling"),
                      ("cap", "Error")):
        hits = [w for w in words if w[4] == word]
        if hits:
            heads[key] = min(hits, key=lambda w: w[1])
    if set(heads) != {"rep", "ctl", "pool", "cap"}:
        return 0, [f"PDF check: replication panels not found ({sorted(heads)})"]
    xr, xc, xp, xf = (heads[k][0] for k in ("rep", "ctl", "pool", "cap"))
    strip = [w[1] for w in words if w[4] == "labels:" and w[1] > heads["rep"][3]]
    y_end = min(strip) if strip else page.rect.height
    num, mc = bundle["numeric"], bundle["control"]

    pp = poster_pp_values(bundle)

    def val(name):
        return norm(pp.get(name, num[name])) if name in num else "---"

    def anchors(x_lo, head, names):
        found = {}
        for w in sorted(words, key=lambda w: w[1]):
            if abs(w[0] - x_lo) < 2 and head[3] < w[1] < y_end and w[4] in names \
                    and w[4] not in found:
                found[w[4]] = w
        return found

    seeds = bundle["present_seeds"] or []
    rep_rows = anchors(xr, heads["rep"], {"RANDOM", "ENVELOPE", "CENTROID"})
    for arm in POSTER_ROWS:
        w = ARM_WORD[arm]
        orig = "RepAUC%sOrig" % w
        late_mark = "\u2020" if arm in bundle["late_originals"] and orig in num else ""
        want = [val(orig) + late_mark] + [val("RepAUC%sSeed%s" % (w, s)) for s in seeds] + \
               [val("RepMeanAUC" + w), "---" if arm == "random" else val("RepDeltaMean" + w)]
        anchor = rep_rows.get(ARM_PLAIN[arm])
        got = row_tokens(words, anchor, xr, xc) if anchor else None
        slots += len(want)
        if got != want:
            problems.append(f"PDF replication row {ARM_PLAIN[arm]}: {got} != sidecar {want}")
    ctl_rows = anchors(xc, heads["ctl"], {"CENTROID", "RANDOM-CB"})
    for name, first, desc in (("RepPlacementDelta", "CENTROID", mc.get("P_descriptive")),
                              ("RepBudgetShift", "RANDOM-CB", mc.get("B_descriptive"))):
        anchor = ctl_rows.get(first)
        got = row_tokens(words, anchor, xc, xp)[-1:] if anchor else None
        want = [val(name) + ("\u2020" if desc and name in num else "")]
        slots += 1
        if got != want:
            problems.append(f"PDF control {name}: {got} != sidecar {want}")
    pool_rows = anchors(xp, heads["pool"], {"ENVELOPE", "CENTROID"})
    for arm in ("envelope", "centroid"):
        w = ARM_WORD[arm]
        want = [val("RepPoolPatchMaxDelta" + w), val("RepPoolSliceMaxDelta" + w)]
        anchor = pool_rows.get(ARM_PLAIN[arm])
        got = row_tokens(words, anchor, xp, xf) if anchor else None
        slots += 2
        if got != want:
            problems.append(f"PDF pooling row {ARM_PLAIN[arm]}: {got} != sidecar {want}")
    text = squash(page.get_text())
    lab = {a: LABEL_TEX[bundle["labels"][a]] for a in ("centroid", "envelope")}
    phrases = [
        ("outcome strip labels",
         "Registered labels: CENTROID %s, ENVELOPE %s" % (lab["centroid"], lab["envelope"])),
        ("takeaway labels", "Registered replication labels: CENTROID %s, ENVELOPE %s."
         % (lab["centroid"], lab["envelope"])),
        ("answer box labels", "Registered replication: CENTROID %s, ENVELOPE %s."
         % (lab["centroid"], lab["envelope"])),
        ("control description", "Matched-budget control: %s." % CONTROL_TEXT[mc["label"]]),
        ("seed-spread caption", "Error bars: s.d. over probe-head seeds."),
        ("label rule", "Label rule: consistent direction if every registered"),
    ]
    if "RepRandomRange" in num:
        phrases.append(("RANDOM range", "(RANDOM range %s pp)" % val("RepRandomRange")))
    elif squash("(RANDOM range") in text:
        problems.append("PDF shows a RANDOM range although the registered RANDOM reference "
                        "is incomplete")
    for what, phrase in phrases:
        if squash(phrase) not in text:
            problems.append(f"PDF {what}: '{phrase}' not found")
    late = squash("\u2020From a run after the freeze: descriptive.")
    has_late = bool(mc.get("P_descriptive") and "RepPlacementDelta" in num) or \
        bool(mc.get("B_descriptive") and "RepBudgetShift" in num)
    if (late in text) != has_late:
        problems.append("PDF late-run note " + ("missing" if has_late else "unexpected"))
    desc, late_o = bool(bundle["desc_seeds"]), bool(bundle["late_originals"])
    if desc != (squash("*Incomplete at the freeze") in text):
        problems.append("PDF descriptive-seed note disagrees with the sidecar")
    if late_o != (squash(LATE_ORIGINAL_NOTE).lower() in text.lower()):
        problems.append("PDF late-original note disagrees with the per-run table")
    note = ("*Incomplete at the freeze, \u2020original re-probed after the freeze: descriptive."
            if desc and late_o else "*Incomplete at the freeze: descriptive." if desc else
            "\u2020Original re-probed after the freeze: descriptive." if late_o else None)
    if note and squash(note) not in text:
        problems.append(f"PDF strip note '{note}' not found")
    return slots, problems


def pdf_static_results_check(page):
    """Original-run absolute AUCs, pp differences and key intervals in the rendered PDF."""
    problems = []
    raw = paper_macros(problems)
    # No registered bundle is needed to verify the original-run table.
    pp = poster_pp_values({"numeric": {}, "full_precision": {}})
    words = [(w[0], w[1], w[2], w[3], norm(w[4])) for w in page.get_text("words")]
    candidates = [w for w in words if w[4] == "RANDOM" and
                  w[0] > page.rect.width * .50 and w[1] < page.rect.height * .50]
    if not candidates:
        return 0, ["PDF original-run table not found"]
    first = min(candidates, key=lambda w: w[1])
    specs = [("RANDOM", "Random", None), ("ENVELOPE", "Envelope", "EnvelopeRandom"),
             ("CENTROID", "Oracle", "OracleRandom"),
             ("ANATOMY-V2", "AnatomyTwo", "AnatomyTwoRandom"),
             ("COVER", "Cover", "CoverRandom")]
    for label, auc, delta in specs:
        found = [w for w in words if w[4] == label and abs(w[0] - first[0]) < 2 and
                 first[1] - 1 <= w[1] < first[1] + 8 * (first[3] - first[1])]
        anchor = min(found, key=lambda w: w[1]) if found else None
        expected = []
        for epoch in ("Fifty", "SeventyFive", "Hundred"):
            expected += [norm(raw["AUC" + auc + "Ep" + epoch]),
                         norm(pp["D" + delta + "Ep" + epoch]) if delta else "---"]
        got = row_tokens(words, anchor, first[0], page.rect.width) if anchor else None
        if got != expected:
            problems.append(f"PDF original-run row {label}: {got} != AUC / pp values {expected}")
    text = squash(page.get_text())
    for name in ("DEnvelopeRandomEpFiftyCI", "DOracleRandomEpFiftyCI"):
        if squash(pp[name].replace(r"\,", "")) not in text:
            problems.append(f"PDF key interval missing or not in pp: {name}")
    return 32, problems


def check_log_and_state(log, bundle, problems):
    """The poster must have used the state written for exactly this bundle."""
    loaded = re.search(r"POSTER-RESULTS: ([^\n]+)", log)
    loaded = loaded.group(1).strip() if loaded else ""
    if (bundle is not None) != loaded.startswith("loaded"):
        problems.append(f"poster log says '{loaded or 'no POSTER-RESULTS line'}' but the "
                        f"results bundle is {'valid' if bundle else 'absent'}")
    if bundle is None:
        return
    if ("SYNTHETIC" in loaded) != bundle["synthetic"]:
        problems.append("synthetic state in the poster log disagrees with the bundle")
    try:
        state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        problems.append("generated/results_state.json missing (run build.ps1)")
        return
    for key, want in (("schema", STATE_SCHEMA), ("cr_results_sha256", bundle["tex_sha256"]),
                      ("sidecar_sha256", bundle["sidecar_sha256"])):
        if state.get(key) != want:
            problems.append(f"generated state {key} is stale")
    expected_pp = poster_pp_values(bundle)
    expected_tex = "% Display-only AUC differences in percentage points; source values unchanged.\n" + \
        "\n".join(r"\newcommand{\PosterPP%s}{%s}" % (key, value)
                  for key, value in sorted(expected_pp.items())) + "\n"
    if (state.get("difference_units") != "percentage_points" or
            state.get("pp_values") != expected_pp or not PP_TEX.is_file() or
            PP_TEX.read_text(encoding="ascii") != expected_tex or
            state.get("units_sha256") != sha256_file(PP_TEX)):
        problems.append("percentage-point display state is missing, stale or tampered")
    seeds = re.search(r"POSTER-RESULTS-SEEDS: present=(\S*) descriptive=(\S+)", log)
    if not seeds or seeds.group(1) != "".join(bundle["present_seeds"]) or \
            seeds.group(2) != ("".join(bundle["desc_seeds"]) or "none"):
        problems.append("poster seed columns/descriptive seeds disagree with the sidecar")
    else:
        print(f"               seed indices shown: {seeds.group(1)}; descriptive: "
              f"{seeds.group(2)}; late originals: "
              f"{', '.join(ARM_PLAIN[a] for a in bundle['late_originals']) or 'none'}")
    mc = bundle["control"]
    control = re.search(r"POSTER-RESULTS-CONTROL: (\S+) P=(\S+) B=(\S+)", log)
    want = (mc["label"], {None: "none", True: "late", False: "ontime"}[mc.get("P_descriptive")],
            {None: "none", True: "late", False: "ontime"}[mc.get("B_descriptive")])
    if not control or control.groups() != want:
        problems.append(f"poster control status {control.groups() if control else None} != "
                        f"sidecar {want}")
    else:
        print(f"               matched control: {want[0]}; P {want[1]}, B {want[2]}; "
              f"registered RANDOM reference: "
              f"{', '.join(mc.get('reference_missing') or []) or 'complete'}"
              f"{' missing' if mc.get('reference_missing') else ''}")
    figure = re.search(r"POSTER-RESULTS-FIGURE: (\S+)(?: (\S+))?", log)
    fig_state = state.get("figure")
    if fig_state is None:
        if not figure or figure.group(1) != "missing":
            problems.append("poster included a figure the bundle does not validate")
        print("               seed-spread figure: none in the bundle (placeholder)")
    else:
        if not figure or figure.group(1) != "included" or figure.group(2) != fig_state["tex_path"]:
            problems.append("poster did not include the validated seed-spread figure")
        elif sha256_file(fig_state["file"]) != fig_state["sha256"]:
            problems.append("included seed-spread figure changed after validation")
        else:
            print(f"               seed-spread figure: {fig_state['kind']}")


def registered_variants(problems):
    """Every poster_variant entry (repository path -> SHA-256) in the release registry."""
    reg = read_json(REPO_ROOT / REGISTRY, problems, REGISTRY + " (figure registry)")
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "poster_variant" and isinstance(value, dict):
                    found.update(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    if reg is not None:
        walk(reg)
    return found


def check_paper_figures(log, problems):
    """Each paper figure must be loaded from the committed poster-variant folder and match the
    SHA-256 that the release registry records for it (poster_variant)."""
    registered = registered_variants(problems)
    used = [u.strip().strip('"') for u in re.findall(r"<use ([^<>]+?)>", log)]
    for name in PAPER_FIGURES:
        rel = POSTER_VARIANT_DIR + "/" + name
        want = (REPO_ROOT / rel).resolve()
        paths = sorted({u for u in used if u.endswith(name)})
        if not paths:
            problems.append(f"{name} was not included")
        for path in paths:
            got = (HERE / path).resolve()
            if got != want:
                problems.append(f"{name} loaded from {path}, not from the committed {rel}")
                continue
            digest = sha256_file(got)
            if registered.get(rel) != digest:
                problems.append(f"{name} (SHA-256 {digest[:12]}) is not the poster variant "
                                f"registered in {REGISTRY}")
            else:
                print(f"               {name}: registered poster variant "
                      f"(SHA-256 {digest[:12]})")


def paper_macros(problems):
    values = {}
    for name in ("auto_numbers.tex", "cr_numbers.tex"):
        path = REPO_ROOT / "paper" / "genai4health2026" / "auto" / name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{name} unreadable ({exc})")
            continue
        for m in re.finditer(r"^\\newcommand\{\\(\w+)\}\{([^{}]*)\}", text, re.M):
            values[m.group(1)] = m.group(2)
    return values


def percentage_points(value, signed=True):
    """Display an AUC difference as pp; absolute AUCs never pass through this helper."""
    if not number(value):
        raise ValueError("non-finite AUC difference")
    return format(100.0 * float(value), "+.2f" if signed else ".2f")


def poster_pp_values(bundle):
    """Display aliases only: static paper differences and verified full-precision replication differences."""
    errors = []
    raw = paper_macros(errors)
    if errors:
        raise ValueError("; ".join(errors))
    result = {}
    pattern = r"D(?:OracleRandom|EnvelopeRandom|AnatomyTwoRandom|AnatomyTwoEnvelope|CoverRandom)Ep(?:Fifty|SeventyFive|Hundred)(?:CI)?"
    for key, value in raw.items():
        if not re.fullmatch(pattern, key):
            continue
        if value == "---":
            result[key] = "---"
        elif key.endswith("CI"):
            parts = re.findall(r"[+-]?\d+\.\d+", value)
            if len(parts) != 2:
                raise ValueError("unrecognized interval: " + key)
            result[key] = "[" + r",\,".join(percentage_points(float(x)) for x in parts) + "]"
        else:
            result[key] = percentage_points(float(value))
    for key in bundle["numeric"]:
        if re.fullmatch(r"Rep(?:DeltaMean\w+|PlacementDelta|BudgetShift|Pool(?:PatchMax|SliceMax)Delta\w+|RandomRange)", key):
            result[key] = percentage_points(bundle["full_precision"][key], signed=key != "RepRandomRange")
    return result


def check_static_claims(problems):
    """Words in the KeyBox, the Answer box and the delivered-mask bullet that depend on the
    original-run and mask macros: the signs (scoped to the arms named on the poster), and which
    epochs give the end points of the stated ranges."""
    v = paper_macros(problems)
    try:
        env = [float(v["DEnvelopeRandomEp" + e]) for e in ("Fifty", "SeventyFive", "Hundred")]
        cen = [float(v["DOracleRandomEp" + e]) for e in ("Fifty", "SeventyFive", "Hundred")]
        ana, cov = float(v["DAnatomyTwoEnvelopeEpFifty"]), float(v["DCoverRandomEpHundred"])
        ctx = [float(v["CRCtxGapCentroid"]), float(v["CRCtxGapEnvelope"])]
        cover_ctx, random_ctx = float(v["CRCtxCover"]), float(v["CRCtxRandom"])
        cover50 = float(v["DCoverRandomEpFifty"])
    except (KeyError, ValueError) as exc:
        problems.append(f"KeyBox/Answer macros missing or not numeric ({exc})")
        return
    claims = [
        ("CENTROID and ENVELOPE minus RANDOM positive at epochs 50, 75 and 100",
         min(env + cen) > 0),
        ("ENVELOPE range runs from epoch 100 to epoch 50", min(env) == env[2] and
         max(env) == env[0]),
        ("CENTROID range runs from epoch 50 to epoch 75", min(cen) == cen[0] and
         max(cen) == cen[1]),
        ("more anatomy did not add gains (ANATOMY-V2 - ENVELOPE < 0, COVER - RANDOM < 0 at "
         "epoch 100 and below ENVELOPE - RANDOM at epoch 50)",
         ana < 0 and cov < 0 and cover50 < env[0]),
        ("CENTROID and ENVELOPE leave more context visible (context gaps > 0)", min(ctx) > 0),
        ("COVER leaves less context than RANDOM", cover_ctx < random_ctx),
    ]
    bad = [text for text, ok in claims if not ok]
    for text in bad:
        problems.append(f"poster wording no longer matches the paper macros: {text}")
    if not bad:
        print(f"Claims         {len(claims)} KeyBox/Answer wording checks agree with the paper "
              f"macros")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--design-log")
    ap.add_argument("--png")
    ap.add_argument("--dpi", type=int, default=50)
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="accept SYNTHETIC DRY RUN results (scratch copies only)")
    ap.add_argument("--precheck", action="store_true",
                    help="validate auto/ and write the generated state before compiling")
    a = ap.parse_args()

    if a.precheck:
        clear_state()
        bundle, problems = gate(a.results_dir, a.allow_synthetic, "Precheck")
        if problems:
            for p in problems:
                print(f"PROBLEMS       {p}")
            return 1
        if bundle is not None:
            state = write_state(bundle)
            fig = state["figure"]["kind"] if state["figure"] else "none (placeholder)"
            print(f"               state written: control {state['control']}, seeds "
                  f"{''.join(state['seeds_present'])} (descriptive "
                  f"{''.join(state['seeds_descriptive']) or 'none'}), figure: {fig}")
        else:
            # Original-run pp displays also work in the pre-results placeholder build.
            pp = poster_pp_values({"numeric": {}, "full_precision": {}})
            GENERATED.mkdir(exist_ok=True)
            PP_TEX.write_text("% Display-only AUC differences in percentage points; source values unchanged.\n" +
                              "\n".join(r"\newcommand{\PosterPP%s}{%s}" % (key, value)
                                        for key, value in sorted(pp.items())) + "\n", encoding="ascii")
        return 0
    if not (a.pdf and a.design_log and a.png):
        ap.error("PRINT_PDF, --design-log and --png are required unless --precheck is given")

    import pymupdf as fitz  # PyMuPDF

    problems = []
    pdf_path = pathlib.Path(a.pdf)
    doc = fitz.open(pdf_path)
    page = doc[0]
    w_in, h_in = page.rect.width / 72.0, page.rect.height / 72.0
    pix = page.get_pixmap(dpi=a.dpi, alpha=False)
    pix.save(a.png)
    print(f"PDF            {pdf_path.name}: {doc.page_count} page(s), "
          f"{w_in:.2f} x {h_in:.2f} in ({w_in * 25.4:.1f} x {h_in * 25.4:.1f} mm), "
          f"{pdf_path.stat().st_size / 1e6:.2f} MB")
    print(f"PNG preview    {a.png}: {pix.width} x {pix.height} px at {a.dpi} dpi, "
          f"{pathlib.Path(a.png).stat().st_size / 1e6:.2f} MB")
    if doc.page_count != 1:
        problems.append(f"expected 1 page, found {doc.page_count}")

    fonts = sorted({f[3] for f in page.get_fonts(full=True)})
    print(f"Fonts          {len(fonts)}: {', '.join(fonts)}")
    if any(f[2] == "Type3" for f in page.get_fonts(full=True)):
        print("               note: Type3 fonts present")

    raw = pathlib.Path(a.design_log).read_text(encoding="utf-8", errors="replace")
    # TeX hard-wraps log lines at 79 characters; re-join them before matching.
    log = raw
    while True:
        joined = re.sub(r"^(.{79})\n", r"\1", log, flags=re.M)
        if joined == log:
            break
        log = joined
    overfull = re.findall(r"^Overfull \\[hv]box .*$", log, re.M)
    underfull = re.findall(r"^Underfull \\[hv]box .*$", log, re.M)
    print(f"Overfull boxes {len(overfull)}")
    for line in overfull:
        print(f"               {line}")
    print(f"Underfull      {len(underfull)}")
    for line in re.findall(r"POSTER-SIZE: .*", log):
        print("Size           " + " ".join(line.split()))
    for line in re.findall(r"POSTER-FIT .*", log):
        print(f"Fit            {line}")
    overflow = re.findall(r"OVERFLOW \(.*", log)
    for line in overflow:
        print(f"OVERFLOW       {line}")
        problems.append(line)
    holders = collections.Counter(re.findall(r"POSTER-PLACEHOLDER: ([^\n]+)", log))
    print("Placeholders   " + (", ".join(f"{k} x{v}" for k, v in sorted(holders.items()))
                               or "none"))
    figs = sorted(set(re.findall(r"(fig_[A-Za-z0-9_]+\.(?:pdf|png|jpe?g))", log)))
    print(f"Figures        {', '.join(figs) or 'none found in log'}")
    bad = [f for f in figs if f not in ALLOWED_FIGURES]
    if bad:
        problems.append(f"figures outside the allow-list: {bad}")
    check_paper_figures(log, problems)
    check_static_claims(problems)
    static_slots, static_problems = pdf_static_results_check(page)
    problems += static_problems
    if not static_problems:
        print(f"Original cells {static_slots} original-run AUC / pp cells and key intervals match the paper macros")
    problems += re.findall(r"POSTER-PROBLEM: ([^\n]+)", log)

    bundle, gate_problems = gate(a.results_dir, a.allow_synthetic, "Results")
    problems += gate_problems
    if not gate_problems or bundle is not None:
        check_log_and_state(log, bundle, problems)
    text = page.get_text()
    synthetic_shown = SYNTHETIC_MARKER in text
    if bundle is not None and synthetic_shown != bundle["synthetic"]:
        problems.append("SYNTHETIC banner " + ("missing from" if bundle["synthetic"] else
                                               "present in") + " the PDF")
    if bundle is None and synthetic_shown:
        problems.append("SYNTHETIC banner present without validated results")
    if bundle is not None:
        slots, cell_problems = pdf_results_check(page, bundle)
        problems += cell_problems
        if not cell_problems:
            print(f"Cells          {slots} result slots, labels, control text and late-run "
                  f"marks in the PDF match the sidecar")
    if problems:
        for p in problems:
            print(f"PROBLEMS       {p}")
        return 1
    print("Checks         OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
