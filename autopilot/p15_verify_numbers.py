"""Audit numeric coverage; unresolved rendered values are release blockers.

Known AUC/CI/contrast/count macros are bound to named aggregate statistics, not
to whichever arm happens to round to the same value. All other macros, literals,
and figures enter an explicit coverage record. This is intentionally not a
claim that all manuscript quantities have already been independently verified.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

try:
    from . import release_assets as assets
    from . import numeric_bindings as numeric
    from . import numeric_plot_review as plots
except ImportError:
    import release_assets as assets
    import numeric_bindings as numeric
    import numeric_plot_review as plots

PAPER = str(assets.PAPER)
INV = r"D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json"
ARM = {"Random": "random", "Oracle": "oracle", "Envelope": "envelope",
       "Cover": "cover-f021", "AnatomyTwo": "anatomy-v2", "AnatomyOne": "anatomy-v1"}
EPOCH = {"EpTwentyFive": 25, "EpTwentySeven": 27, "EpThirty": 30, "EpThirtyFour": 34,
         "EpThirtyFive": 35, "EpForty": 40, "EpFifty": 50, "EpSeventyThree": 73,
         "EpSeventyFive": 75, "EpNinetyTwo": 92, "EpHundred": 100}
EP_PATTERN = "|".join(sorted(EPOCH, key=len, reverse=True))
ARM_PATTERN = "|".join(sorted(ARM, key=len, reverse=True))


def canonical(value):
    """Remove only recognised presentation wrappers, never silently skip a value."""
    value = str(value).strip()
    value = value.replace(r"{,}", "").replace(r"\,", "").replace("$", "")
    value = value.replace(r"\%", "%")
    for _ in range(6):
        new = re.sub(r"\\(?:mathbf|mathrm|textbf|textnormal)\{([^{}]*)\}", r"\1", value)
        if new == value:
            break
        value = new
    return re.sub(r"\s+", "", value)


def rendered_equal(actual, expected):
    return canonical(actual) == canonical(expected)


def pfmt(value):
    return "$<$0.0001" if value < 1e-4 else "%.4f" % value


def known_bindings(stats, inventory):
    """Mirror only understood generator lookups; unknowns remain unresolved."""
    bindings = {}
    table = {row["key"]: (i, row) for i, row in enumerate(stats.get("table", []))}
    contrasts = {(row["a"], row["b"]): (i, row)
                 for i, row in enumerate(stats.get("contrasts", []))}

    def put(name, shown, source, pointer, operation=None):
        bindings[name] = {"expected": shown, "source": source, "pointer": pointer,
                          "operation": operation or "format"}

    def key(tag, ep, fp32=False):
        precision = "fp32" if fp32 or tag in ("AnatomyOne", "AnatomyTwo", "Cover") else "fp16"
        return "%s@ep%d@%s" % (ARM[tag], ep, precision)

    def contrast(name, a, b):
        found = contrasts.get((a, b)) or contrasts.get((b, a))
        if not found:
            return
        index, record = found
        direct = record["a"] == a
        sign = 1 if direct else -1
        pointer = "/contrasts/%d" % index
        put(name, "%+.4f" % (sign * record["delta_a_minus_b"]), "p1c_stats.json",
            pointer + "/delta_a_minus_b", "sign=%d; signed 4 decimals" % sign)
        lo = record["boot_ci95_lo"] if direct else -record["boot_ci95_hi"]
        hi = record["boot_ci95_hi"] if direct else -record["boot_ci95_lo"]
        put(name + "CI", "[%+.4f,\\,%+.4f]" % (lo, hi), "p1c_stats.json", pointer,
            "paired bootstrap CI; swap/negate endpoints when reversed")
        put(name + "P", pfmt(record["delong_p"]), "p1c_stats.json", pointer + "/delong_p")
        if "delong_q_bh" in record:
            put(name + "Q", pfmt(record["delong_q_bh"]), "p1c_stats.json", pointer + "/delong_q_bh")

    for name, field in (("Ntest", "n_test"), ("Npos", "n_pos"), ("Nboot", "n_bootstrap")):
        if field in stats:
            put(name, str(stats[field]), "p1c_stats.json", "/" + field)
    if "n_test" in stats and "n_pos" in stats:
        put("Nneg", str(stats["n_test"] - stats["n_pos"]), "p1c_stats.json", "/",
            "n_test - n_pos")
    if stats.get("table"):
        put("Nprobes", str(len(stats["table"])), "p1c_stats.json", "/table", "length")

    for tag in ARM:
        for word, ep in EPOCH.items():
            for suffix, use_fp32 in (("", False), ("FPthirtytwo", True)):
                record = table.get(key(tag, ep, use_fp32))
                if record:
                    index, row = record
                    for prefix, field in (("AUC", "auc"), ("CIlo", "ci95_lo"), ("CIhi", "ci95_hi")):
                        if row.get(field) is not None:
                            if not math.isfinite(row[field]) or not 0 <= row[field] <= 1:
                                raise ValueError("invalid AUC/CI statistic at " + row["key"] + "/" + field)
                            put(prefix + tag + word + suffix, "%.4f" % row[field],
                                "p1c_stats.json", "/table/%d/%s" % (index, field))
                elif not suffix:
                    # Inventory fallback is precision-specific and requires one
                    # primary record. It never accepts an arbitrary closest arm.
                    expected_precision = key(tag, ep).split("@")[-1]
                    matches = [(i, row) for i, row in enumerate(inventory.get("records", []))
                               if row.get("status") == "primary" and row.get("arm") == ARM[tag]
                               and row.get("epoch") == ep and row.get("precision") == expected_precision
                               and row.get("auc") is not None]
                    if len(matches) == 1:
                        index, row = matches[0]
                        if not math.isfinite(row["auc"]) or not 0 <= row["auc"] <= 1:
                            raise ValueError("invalid primary AUC statistic")
                        put("AUC" + tag + word, "%.4f" % row["auc"], "p1b_full_inventory.json",
                            "/records/%d/auc" % index)
            for other in ARM:
                if other == tag:
                    continue
                a = key(tag, ep)
                b = key(other, ep)
                if tag in ("Cover", "AnatomyTwo") and key(other, ep, True) in table:
                    b = key(other, ep, True)
                contrast("D" + tag + other + word, a, b)
    for tag in ("Cover", "Random"):
        a = key(tag, 75)
        b = key(tag, 50)
        contrast("D" + tag + "SelfFiftyToSeventyFive", a, b)
    cover = [(key_, index, row) for key_, (index, row) in table.items()
             if key_.startswith("cover-f021@")]
    if cover:
        peak, index, row = max(cover, key=lambda item: item[2]["auc"])
        put("AUCCoverPeak", "%.4f" % row["auc"], "p1c_stats.json", "/table/%d/auc" % index)
        put("CoverPeakEpoch", str(row["epoch"]), "p1c_stats.json", "/table/%d/epoch" % index)
        contrast("DCoverPeakToHundred", key("Cover", 100), peak)
    ancestor = [(i, row) for i, row in enumerate(stats.get("table", []))
                if row.get("arm") == "ancestor"]
    if len(ancestor) == 1:
        index, row = ancestor[0]
        put("AUCAncestor", "%.4f" % row["auc"], "p1c_stats.json", "/table/%d/auc" % index)
    # The direct anatomy-envelope paragraph deliberately uses DeLong intervals,
    # unlike the bootstrap intervals in the trajectory comparisons.
    for tag, ep, word in (("AnatomyOne", 30, "EpThirty"), ("AnatomyTwo", 50, "EpFifty")):
        pair = (key(tag, ep, True), key("Envelope", ep, True))
        contrast("D" + tag + "Envelope" + word, *pair)
        if pair in contrasts:
            index, row = contrasts[pair]
            put("D" + tag + "Envelope" + word + "CI",
                "[%+.4f,\\,%+.4f]" % tuple(row["delong_ci95"]), "p1c_stats.json",
                "/contrasts/%d/delong_ci95" % index, "DeLong CI; signed 4 decimals")
    return bindings


def no_result(name, value, stats, inventory):
    if canonical(value) not in ("---", "--", "notrun"):
        return False
    match = re.fullmatch(r"(?:T?AUC|D|T)(%s)(?:Random)?(%s)" % (ARM_PATTERN, EP_PATTERN), name)
    if not match:
        return False
    arm, ep = ARM[match[1]], EPOCH[match[2]]
    return not any(row.get("arm") == arm and row.get("epoch") == ep
                   and row.get("auc") is not None
                   for row in stats.get("table", []) + inventory.get("records", []))


def main_table_bindings(source, bindings):
    """Bind Table 1's data cells by arm/epoch/column, never by value coincidence."""
    cells, errors, seen = {}, [], []
    names = {r"\textsc{random}": "Random", r"\textsc{envelope}": "Envelope", r"\ArmBest{}": "Oracle"}
    offset = 0
    for line in source.splitlines(keepends=True):
        parts = line.split("&")
        tag = names.get(parts[0].strip())
        if tag:
            seen.append(tag)
            if len(parts) != 7:
                errors.append("table_main row has unexpected number of columns: " + tag)
            column_offset = len(parts[0]) + 1
            for column, cell in enumerate(parts[1:7], 1):
                word = ("EpFifty", "EpSeventyFive", "EpHundred")[(column - 1) // 2]
                name = "AUC" + tag + word if column % 2 else "D" + tag + "Random" + word
                binding = bindings.get(name)
                if binding:
                    matches = list(assets.NUMBER.finditer(cell))
                    if len(matches) == 1:
                        cells[offset + column_offset + matches[0].start()] = binding
                column_offset += len(cell) + 1
        offset += len(line)
    if sorted(seen) != sorted(names.values()):
        errors.append("table_main required policy rows are missing or duplicated")
    return cells, errors


def audit(paper, stats_dir, review_file=None):
    paper, stats_dir = Path(paper), Path(stats_dir)
    review_path = Path(review_file) if review_file else paper / "numeric_reviews.json"
    review_hash = assets.sha256(review_path) if review_path.exists() else None
    snapshot = assets.input_hashes(paper)
    text, files = assets.source_tree(paper)
    # The publisher renames only the active root. Keep immutable receipt
    # identities logical while retaining physical filenames in input_hashes.
    source_aliases = ({"main.tex": "main_submission.tex"}
                      if "main.tex" in files and "main_submission.tex" not in files else {})
    data, source_hashes, errors = {}, {}, []
    for name in ("p1c_stats.json", "p1b_full_inventory.json"):
        path = stats_dir / name
        if path.is_file():
            raw = path.read_bytes()
            data[name] = json.loads(raw.decode("utf-8-sig"))
            source_hashes[name] = hashlib.sha256(raw).hexdigest()
        else:
            data[name] = {}
            errors.append("required evidence missing: " + name)
    stats, inventory = data["p1c_stats.json"], data["p1b_full_inventory.json"]
    keys = [row["key"] for row in stats.get("table", [])]
    if len(keys) != len(set(keys)):
        errors.append("ambiguous duplicate primary statistic keys")
    pairs = [tuple(sorted((row["a"], row["b"]))) for row in stats.get("contrasts", [])]
    if len(pairs) != len(set(pairs)):
        errors.append("ambiguous duplicate paired contrasts")
    evidence = numeric.Evidence(paper, stats_dir, numeric.BUILTIN_SOURCES)
    for name in data:
        evidence.load(name)
    bindings = numeric.extended_bindings(evidence, known_bindings(stats, inventory))
    reviews = numeric.read_reviews(paper, evidence, review_file)
    for name, spec in reviews.get("macros", {}).items():
        if name in bindings:
            errors.append("custom binding cannot override known macro: " + name)
            continue
        try:
            bindings[name] = evidence.binding(spec["expression"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            bindings[name] = {"binding_error": str(exc)}
    macro_records = assets.macros(text)
    definitions = {name: value for name, value, _, _ in macro_records}
    if len(definitions) != len(macro_records):
        errors.append("duplicate macro definitions cannot be assigned one numeric meaning")
    body = assets.without_definitions(text)
    used = set(re.findall(r"\\([A-Za-z]+)\b", body))
    for _ in range(len(definitions) + 1):
        expanded = used | {child for name in used if name in definitions
                           for child in re.findall(r"\\([A-Za-z]+)\b", definitions[name])}
        if expanded == used:
            break
        used = expanded
    rows = []
    for name, value in sorted(definitions.items()):
        numeric_value = bool(assets.NUMBER.search(value))
        required = name in used and (numeric_value or name in bindings or
                                     name.startswith(("AUC", "D", "CI", "N", "TAUC", "TAnatomy", "TCover"))
                                     or name in ("CoverFixedStatus", "CoverFixedHidden") or r"\ph" in value)
        row = {"kind": "macro", "id": name, "value": value, "rendered": name in used}
        binding = bindings.get(name)
        if not required:
            row["status"] = "unused" if name not in used else "nonnumeric_definition"
        elif binding:
            row.update(binding)
            if "binding_error" in binding:
                row["status"] = "unresolved"
                row["action"] = binding["binding_error"]
            else:
                row["status"] = "verified" if rendered_equal(value, binding["expected"]) else "mismatch"
                if "source" in binding and binding["source"] in evidence.hashes:
                    row["source_hashes"] = {binding["source"]: evidence.hashes[binding["source"]]}
        elif no_result(name, value, stats, inventory):
            row["status"] = "explicit_no_result"
            row["source"] = "absence in p1c_stats.json table and p1b_full_inventory.json records"
            row["source_hashes"] = {n: evidence.hashes[n] for n in data}
        elif name.startswith(("AUCCoverFixed", "CoverFixed")) and canonical(value) in ("---", "notrun") and not (stats_dir / "p19_cover_fixed.json").exists():
            row["status"] = "explicit_no_result"
            row["source"] = str(stats_dir / "p19_cover_fixed.json")
            row["operation"] = "explicit P8 no-run slot; artifact absent"
        else:
            row["status"] = "unresolved"
            row["action"] = "Bind this rendered macro to its exact source statistic/operation."
        rows.append(row)
    for name in sorted(used - set(definitions)):
        if name in bindings or name.startswith(("AUC", "CIlo", "CIhi", "Ntest", "Nprobes", "TAUC", "TCover",
                            "TAnatomy", "DOracle", "DEnvelope", "DAnatomy", "DCover",
                            "SOP", "Sev", "SubRace", "LEGap", "PDSev", "PDDiff", "DSens")):
            rows.append({"kind": "macro", "id": name, "status": "undefined", "rendered": True})

    review_index = {}
    for entry in reviews.get("literals", []):
        logical_file = source_aliases.get(entry["file"], entry["file"])
        key = (logical_file, entry["context_sha256"], entry["token_index"])
        if key in review_index:
            errors.append("duplicate literal review: " + str(key))
        review_index[key] = entry
    applied_reviews = set()
    for rel, source in sorted(files.items()):
        logical_rel = source_aliases.get(rel, rel)
        literal_bindings, table_errors = numeric.table_bindings(logical_rel, source, evidence)
        errors.extend(table_errors)
        prose_cells, prose_errors = numeric.prose_bindings(source, logical_rel, evidence)
        literal_bindings.update(prose_cells)
        errors.extend(prose_errors)
        if rel == "auto/table_main.tex":
            main_cells, table_errors = main_table_bindings(source, bindings)
            literal_bindings.update(main_cells)
            errors.extend(table_errors)
        for row in numeric.literals(source, logical_rel):
            row["source_file"] = rel
            binding = literal_bindings.get(row.pop("offset"))
            key = (logical_rel, row["context_sha256"], row["token_index"])
            if binding:
                row.update(binding)
                if "status" not in binding:
                    row["status"] = "verified" if rendered_equal(row["value"], binding["expected"]) else "mismatch"
            elif key in review_index and row["status"] != "structural":
                applied_reviews.add(key)
                try:
                    row.update(numeric.review_literal(row, review_index[key], evidence))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row.update(status="unresolved", action=str(exc))
            elif row["status"] == "unresolved":
                row["action"] = ("Bind exact empirical field/operation, or supply a context-hashed "
                                 "protocol/formula/citation review with immutable source evidence.")
            rows.append(row)
    for key in set(review_index) - applied_reviews:
        errors.append("stale or inapplicable literal review (context/token changed): " + str(key))
    asset_manifest = assets.asset_inventory(paper)
    plotted = plots.verify_local_plots(paper, evidence, bindings, asset_manifest["items"])
    figure_reviews = {entry["path"]: entry for entry in reviews.get("figures", [])}
    if len(figure_reviews) != len(reviews.get("figures", [])):
        errors.append("duplicate figure receipts")
    for item in asset_manifest["items"]:
        caption = numeric.figure_context(text, item["path"])
        row = {"kind": "figure", "id": item["path"], "sha256": item["sha256"],
               "caption_sha256": numeric.digest_text(caption), "caption": caption,
               "identity_status": item["identity_status"], "production": item["kind"],
               "producer": item.get("script"), "status": "unresolved",
               "action": "Independent plotted-value validator or explicit historical-illustration review required; identity is not numeric proof."}
        row.update(plotted.get(item["path"], {}))
        if item["path"] in figure_reviews:
            try:
                if row["status"] == "mismatch":
                    raise ValueError("known plotted-value mismatch cannot be discharged by an illustration receipt")
                row.update(numeric.figure_receipt(item, figure_reviews.pop(item["path"]), evidence, caption))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row["action"] = str(exc)
        if row["status"] not in ("unresolved", "mismatch"):
            row.pop("action", None)
        rows.append(row)
    if figure_reviews:
        errors.append("stale figure receipts: " + ", ".join(figure_reviews))
    counts = dict(Counter(row["status"] for row in rows))
    checked_auc = sum(row["kind"] == "macro" and row["id"].startswith("AUC")
                      and row["status"] == "verified" for row in rows)
    if checked_auc == 0:
        errors.append("no required AUC measurement was verified; empty coverage cannot pass")
    if assets.input_hashes(paper) != snapshot:
        errors.append("source inputs changed during coverage audit; rerun on a stable snapshot")
    if (assets.sha256(review_path) if review_path.exists() else None) != review_hash:
        errors.append("numeric review receipts changed during coverage audit")
    for name, digest in evidence.hashes.items():
        if assets.sha256(evidence.paths[name]) != digest:
            errors.append("statistical evidence changed during coverage audit: " + name)
    unresolved = sum(counts.get(status, 0) for status in ("unresolved", "mismatch", "undefined"))
    if not asset_manifest["ALL_PASS"]:
        errors.append("asset identity declaration failed")
    validators = [Path(__file__), Path(numeric.__file__), Path(plots.__file__),
                  Path(__file__).with_name("numeric_svg_review.py"),
                  Path(__file__).with_name("numeric_reviews.schema.json")]
    blockers = [row for row in rows if row["status"] in ("unresolved", "mismatch", "undefined")]
    return {"version": 2, "scope": "independent source-field formatting and explicit context-bound reviews; not reanalysis or universal correctness",
            "asset_identity": asset_manifest,
            "source_hashes": evidence.hashes, "source_paths": {k: str(v) for k, v in evidence.paths.items()},
            "review_path": str(review_path), "review_sha256": review_hash,
            "input_hashes": snapshot,
            "source_aliases": source_aliases,
            "validator_hashes": {path.name: assets.sha256(path) for path in validators},
            "checked_auc": checked_auc, "counts": counts, "errors": errors,
            "unresolved_by_class": dict(Counter(row.get("claim_class", row["kind"]) for row in blockers)),
            "unresolved_ids": [row["id"] for row in blockers],
            "items": rows, "ALL_PASS": not errors and unresolved == 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", default=PAPER)
    parser.add_argument("--stats-dir", default=str(Path(INV).parent))
    parser.add_argument("--report")
    parser.add_argument("--markdown-report")
    parser.add_argument("--review-file", help="Version-1 evidence receipt file; default paper/numeric_reviews.json")
    args = parser.parse_args(argv)
    report_path = Path(args.report) if args.report else assets.unique_work(prefix="numbers") / "coverage.json"
    try:
        report = audit(args.paper_dir, args.stats_dir, args.review_file)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"ALL_PASS": False, "errors": [str(exc)], "items": []}
    assets.write_json(report_path, report)
    if args.markdown_report:
        numeric.write_report(args.markdown_report, report)
    print("Required AUCs verified:", report.get("checked_auc", 0))
    print("Coverage statuses:", report.get("counts", {}))
    print("Errors:", report.get("errors", []))
    print("Coverage/action manifest:", report_path)
    print("RESULT:", "PASS" if report["ALL_PASS"] else "FAIL")
    return 0 if report["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
