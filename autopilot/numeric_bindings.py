"""Independent numeric bindings and version-1 evidence/review schema.

References are {source, pointer}; pointers are RFC6901, never value searches.
Expressions are {op, args, ...}: format (Python %-format), subtract, add,
multiply, divide, min, max, length, mean, stdev (sample), abs, and literal.
Literal expressions are allowed only inside a documented protocol declaration,
not as empirical evidence. Custom macros use {"expression": expression}.

An optional paper/numeric_reviews.json contains version=1, sources, macros,
literals, figures. Sources specify root (repo/stats/paper), path, sha256.
Each literal specifies file, context_sha256, token_index, value and either an
expression, an assertion, or a review. An expression may select one numeric
component of its independently formatted display. Assertions compare an exact
source expression (lt/le/gt/ge) with a decimal or explicit power-of-ten bound;
their display component must match the printed token. Rounding never repairs
a failed inequality. Context is a whitespace-normalised paragraph or table
row, not a line number. Reviews require kind (protocol/formula/citation),
reviewer, rationale and evidence [{source, pointer} or {source, excerpt}].
P15 treats a staged active root main.tex as logical main_submission.tex for
binding/review lookups only; input_hashes and each literal's source_file retain
the physical paths, and approved receipt bytes are never rewritten.
Citation reviews additionally require immutable_locator (versioned DOI/arXiv
or a retained publication hash) and locator (page/table/section).
Retained PDF/image inputs expose /sha256 and /byte_length identity metadata
for HUMAN reviews only; this is not extraction or mathematical verification.
No reviews are inferred or written by the auditor.

Figure receipts require path, sha256, caption_sha256, inputs and validation.
Illustrations require method=reviewed_source_illustration (or the existing
reviewed_historical_illustration category),
reviewer, limitations and quantitative_scope=illustrative_only. Such receipts
are explicitly NOT mathematical verification. Programmatic receipts use
method=svg_coordinates, series=[{element_id, x, y, x_scale, x_offset, y_scale,
y_offset}]. The x/y expressions must resolve to source-derived arrays; the
actual SVG polyline/polygon/M-L-z coordinates are read and compared. Stable
Matplotlib group gids and source-valued bars are supported by method=svg_bars;
see numeric_svg_review.py for complete axis/decoration/label receipt fields.
Numeric ticks independently check bar affine mappings. Mixed token-map/bar
figures keep reviewed illustrations separate from mathematically checked bars.
Raster plots
need a real independent validator, or honest historical review; matching
their identity or attaching asserted numeric payloads does not verify pixels.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

try:
    from . import release_assets as assets
except ImportError:
    import release_assets as assets


def digest_text(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def pointer(data, path):
    if path in ("", "/"):
        return data
    if not path.startswith("/"):
        raise ValueError("JSON pointer must begin with /")
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def ptr(*parts):
    return "/" + "/".join(str(x).replace("~", "~0").replace("/", "~1") for x in parts)


def ref(source, *parts):
    return {"source": source, "pointer": ptr(*parts)}


def operation(op, *args, **kwargs):
    return {"op": op, "args": list(args), **kwargs}


def fmt(pattern, *args):
    return operation("format", *args, format=pattern)


class Evidence:
    def __init__(self, paper, stats_dir, sources=None):
        self.roots = {"paper": Path(paper), "stats": Path(stats_dir), "repo": assets.REPO}
        self.specs = dict(sources or {})
        self.cache, self.hashes, self.paths = {}, {}, {}

    def load(self, name):
        if name in self.cache:
            return self.cache[name]
        spec = self.specs.get(name, {"root": "stats", "path": name})
        if spec["root"] in ("stats_parent", "phase0"):
            # Only built-in bindings use this root; external reviews cannot.
            root = self.roots["stats"].parent if spec["root"] == "stats_parent" else self.roots["stats"].parent.parent
            path = assets.safe_path(root, spec["path"])
        else:
            path = assets.safe_path(self.roots[spec["root"]], spec["path"])
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if spec.get("sha256") and sha != spec["sha256"]:
            raise ValueError("evidence hash mismatch: " + name)
        self.hashes[name], self.paths[name] = sha, path
        if path.suffix.lower() == ".json":
            self.cache[name] = json.loads(raw.decode("utf-8-sig"))
        elif path.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg", ".npz"):
            # Human citation/illustration receipts may pin retained binary
            # publications. Identity metadata is not extracted numeric proof.
            self.cache[name] = {"sha256": sha, "byte_length": len(raw), "kind": "binary_identity_only"}
        else:
            self.cache[name] = raw.decode("utf-8-sig")
        return self.cache[name]

    def evaluate(self, expr, allow_literal=False):
        if not isinstance(expr, dict):
            raise ValueError("typed expression required")
        if "source" in expr and "op" in expr:
            raise ValueError("expression cannot be both a reference and an operation")
        if "source" in expr:
            value = pointer(self.load(expr["source"]), expr["pointer"])
        else:
            op = expr["op"]
            if op == "literal":
                if not allow_literal:
                    raise ValueError("literal is not empirical evidence")
                return expr["value"]
            args = [self.evaluate(x, allow_literal) for x in expr.get("args", [])]
            if op == "format":
                fields = re.findall(r"%\+?(?:\.\d+)?[dfg]", expr["format"])
                remainder = re.sub(r"%\+?(?:\.\d+)?[dfg]", "", expr["format"])
                if len(fields) != len(args) or "%" in remainder or re.search(r"\d", remainder):
                    raise ValueError("numeric formatter may not inject literal numbers or hide evidence")
                if any(type(a) not in (int, float) or not math.isfinite(a) for a in args):
                    raise ValueError("numeric formatter requires finite numbers, not booleans or text")
                if any(field.endswith("d") and type(a) is not int for field, a in zip(fields, args)):
                    raise ValueError("integer display requires an integer source, not truncation")
                return expr["format"] % tuple(args)
            if op == "subtract":
                value = args[0] - args[1]
            elif op == "divide":
                value = args[0] / args[1]
            elif op == "multiply":
                value = math.prod(args)
            elif op == "add":
                value = sum(args)
            elif op in ("min", "max", "mean", "stdev"):
                values = args[0] if len(args) == 1 and isinstance(args[0], list) else args
                value = {"min": min, "max": max, "mean": statistics.mean,
                         "stdev": statistics.stdev}[op](values)
            elif op == "length":
                value = len(args[0])
            elif op == "abs":
                value = abs(args[0])
            elif op == "array":
                value = args
            elif op == "percent":
                value = args[0] * 100
            elif op == "pvalue":
                if not 0 <= args[0] <= 1:
                    raise ValueError("p-value outside [0, 1]")
                value = "$<$0.0001" if args[0] < 1e-4 else "%.4f" % args[0]
            elif op == "tex_scientific":
                digits = expr.get("digits", 2)
                if type(digits) is not int or not 1 <= digits <= 8 or len(args) != 1:
                    raise ValueError("scientific display needs one value and 1-8 significant digits")
                mantissa, exponent = ("%.*e" % (digits - 1, args[0])).split("e")
                value = r"%s\times10^{%d}" % (mantissa.rstrip("0").rstrip(".") if "." in mantissa else mantissa, int(exponent))
            elif op == "yesno":
                if not isinstance(args[0], bool):
                    raise ValueError("yesno requires a boolean source")
                value = "yes" if args[0] else "no"
            elif op == "severity_label":
                value = (args[0].replace("<", "$<$").replace(">", "$>$")
                         .replace("-6", "$-6$").replace("-2", "$-2$")
                         .replace("-12", "$-12$").replace("_", r"\_"))
            else:
                raise ValueError("unsupported numeric operation: " + op)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite evidence value")
        return value

    def binding(self, expr, **metadata):
        value = self.evaluate(expr)
        names = set()

        def walk(x):
            if isinstance(x, dict):
                if "source" in x:
                    names.add(x["source"])
                for a in x.values():
                    walk(a)
            elif isinstance(x, list):
                for a in x:
                    walk(a)
        walk(expr)
        if not names:
            raise ValueError("empirical binding has no evidence source")
        return {"expected": str(value), "expression": expr,
                "source_hashes": {n: self.hashes[n] for n in sorted(names)}, **metadata}


BUILTIN_SOURCES = {
    "subgroup_auc.json": {"root": "stats_parent", "path": r"subgroup\subgroup_auc.json"},
    "p17_subgroup_multiplicity.json": {"root": "repo", "path": r"results\p17_subgroup_multiplicity.json"},
    "intersectional": {"root": "phase0", "path": r"reports\subgroup\intersectional_auc.json"},
    "background_skill": {"root": "phase0", "path": r"reports\background_signal\skill_scores.json"},
    "composition": {"root": "phase0", "path": r"reports\composition_vs_auc\composition_vs_auc_ep50.json"},
}
for _seed in (42, 1234, 2026):
    BUILTIN_SOURCES["geometry%d" % _seed] = {
        "root": "repo", "path": r"results\masking\table2_geometry\mask_geometry_600slices_bs1_coverf021_seed%d.json" % _seed}
BUILTIN_SOURCES["geometry_batch64"] = {
    "root": "repo", "path": r"results\masking\table2_geometry\mask_geometry_600slices_bs64_coverf021_seed42.json"}


def extended_bindings(evidence, base):
    """Reconstruct P8 formatting from exact source fields, without importing P8."""
    out = dict(base)

    def put(name, expression):
        try:
            out[name] = evidence.binding(expression)
        except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
            out[name] = {"binding_error": str(exc), "expression": expression}

    def scalar(name, source, *path, pattern="%.4f"):
        put(name, fmt(pattern, ref(source, *path)))

    def ci(name, source, path, lo, hi, digits=4):
        put(name, fmt("[%+." + str(digits) + r"f,\,%+." + str(digits) + "f]",
                      ref(source, *path, lo), ref(source, *path, hi)))

    fair = "p7_fairness.json"
    pg = ("arms", "oracle@ep100@fp16", "groups", "race", "per_group")
    for g in ("Black", "White", "Asian"):
        scalar("N" + g, fair, *pg, g, "n", pattern="%d")
        for arm, word in (("random", "Random"), ("oracle", "Oracle")):
            scalar("Race" + word + g, fair, "arms", arm + "@ep100@fp16", "groups", "race", "per_group", g, "auc")
    scalar("NprobesRace", fair, "n_probes_with_race_summary", pattern="%d")
    scalar("WorstRaceCount", fair, "worst_race_group_across_probes", "Black", pattern="%d")
    for g in ("Black", "White"):
        scalar(g + "AUCOracle", fair, *pg, g, "auc")
    put("BlackCIOracle", fmt(r"[%.4f,\,%.4f]", ref(fair, *pg, "Black", "auc_ci95_lo"), ref(fair, *pg, "Black", "auc_ci95_hi")))
    put("RaceGapOracle", fmt("%.4f", operation("subtract", ref(fair, *pg, "White", "auc"), ref(fair, *pg, "Black", "auc"))))
    put("BlackGainOracle", fmt("%+.4f", operation("subtract", ref(fair, *pg, "Black", "auc"),
        ref(fair, "arms", "random@ep100@fp16", "groups", "race", "per_group", "Black", "auc"))))
    for pre, field in (("Gap", "racegap"), ("SexGap", "sexgap")):
        scalar(pre + "Rho", "p7_gap_correlation.json", "spearman_auc_vs_" + field, 0, pattern="%+.3f")
        put(pre + "RhoP", operation("pvalue", ref("p7_gap_correlation.json", "spearman_auc_vs_" + field, 1)))
    trend = "p7b_gap_trend.json"
    scalar("NprobesSub", trend, "n_probes", pattern="%d")
    scalar("NbranchesSub", trend, "trends", "race", "n_branches", pattern="%d")
    for attr, word in (("gender", "Gender"), ("race", "Race"), ("ethnicity", "Ethnicity"),
                       ("language", "Language"), ("maritalstatus", "Marital"), ("age", "Age"), ("severity", "Severity")):
        for suffix, field, pattern in (("Rho", "spearman_rho", "%+.3f"),
            ("BranchRho", "branch_spearman_rho", "%+.3f"), ("GapMin", "gap_min", "%.4f"), ("GapMax", "gap_max", "%.4f")):
            scalar("Sub" + word + suffix, trend, "trends", attr, field, pattern=pattern)
        for suffix, field in (("RhoP", "spearman_p"), ("Q", "q_bh_across_attributes"), ("BranchP", "branch_spearman_p")):
            put("Sub" + word + suffix, operation("pvalue", ref(trend, "trends", attr, field)))
        try:
            counts = evidence.load(trend)["worst_group_consistency"][attr]["counts"]
            top = max(counts, key=counts.get)
            scalar("Sub" + word + "WorstN", trend, "worst_group_consistency", attr, "counts", top, pattern="%d")
            # Labels themselves are categorical; no measured number is inferred.
            out["Sub" + word + "Worst"] = {
                "expected": top.replace("<", "$<$").replace(">", "$>$").replace("-6", "$-6$").replace("-2", "$-2$").replace("-12", "$-12$").replace("_", r"\_"),
                "source": trend, "pointer": ptr("worst_group_consistency", attr, "counts"),
                "operation": "argmax key; severity label escaping", "source_hashes": {trend: evidence.hashes[trend]}}
        except (OSError, KeyError, ValueError):
            pass
    put("SeverityGapSpread", fmt("%.4f", operation("subtract", ref(trend, "trends", "severity", "gap_max"), ref(trend, "trends", "severity", "gap_min"))))
    try:
        rows = evidence.load(trend)["rows"]
        for suffix, op in (("Min", "min"), ("Max", "max")):
            put("SubAUC" + suffix, fmt("%.4f", operation(op, *[ref(trend, "rows", i, "overall_auc") for i in range(len(rows))])))
    except (OSError, KeyError, ValueError):
        pass
    sub = "subgroup_auc.json"
    severity = {"mild (-6,-2]": "Mild", "moderate (-12,-6]": "Moderate", "severe (<=-12)": "Severe"}
    sevrefs = {}
    try:
        data = evidence.load(sub)
        for probe, word in (("sweep_random_ep100", "Random"), ("sweep_oracle_ep100", "Oracle"), ("frozen_meanpool_mirage_ep100", "Envelope")):
            for i, row in enumerate(data[probe]["subgroups"]["severity"]["levels"]):
                if row["subgroup"] in severity:
                    level = severity[row["subgroup"]]
                    r = ref(sub, probe, "subgroups", "severity", "levels", i, "auc")
                    sevrefs[word, level] = r
                    put("Sev" + word + level, fmt("%.4f", r))
        for level in severity.values():
            put("SevDelta" + level, fmt("%+.4f", operation("subtract", sevrefs["Oracle", level], sevrefs["Random", level])))
        for word in ("Oracle", "Random"):
            put("SevGap" + word, fmt("%.4f", operation("subtract", sevrefs[word, "Severe"], sevrefs[word, "Mild"])))
    except (OSError, KeyError, ValueError):
        pass
    try:
        inv = evidence.load("p1b_full_inventory.json")
        ft = {}
        for i, row in enumerate(inv["records"]):
            if row.get("family") == "finetune":
                key = row["arm"], row["tag"].split("/")[-1]
                if key in ft:
                    raise ValueError("ambiguous fine-tuned head: " + str(key))
                ft[key] = ref("p1b_full_inventory.json", "records", i, "auc")
        fo, fr = ft["oracle", "meanpool"], ft["random", "mean_pool"]
        put("FTOracleMeanpool", fmt("%.4f", fo))
        put("FTRandomMeanpool", fmt("%.4f", fr))
        put("FTDelta", fmt("%+.4f", operation("subtract", fo, fr)))
        best = {arm: operation("max", *[r for (a, _), r in ft.items() if a == arm]) for arm in ("oracle", "random")}
        for arm, word in (("oracle", "Oracle"), ("random", "Random")):
            put("FT" + word + "Best", fmt("%.4f", best[arm]))
        put("FTDeltaBest", fmt("%+.4f", operation("subtract", best["oracle"], best["random"])))
    except (OSError, KeyError, ValueError):
        pass
    op = "p8b_operating_points.json"
    for arm, word in (("random", "Random"), ("envelope", "Envelope"), ("intensity", "Intensity")):
        scalar("Brier" + word, op, "arms", arm, "brier")
        scalar("ECE" + word, op, "arms", arm, "ece_15bin")
        for key, kw in (("spec85", "SpecEightyFive"), ("spec90", "SpecNinety")):
            for pre, field in (("Sens", "sensitivity"), ("SpecAch", "specificity"), ("PPV", "ppv"), ("NPV", "npv")):
                scalar(pre + word + kw, op, "arms", arm, "at", key, field)
    for pair, word in (("intensity_minus_random", "IntRand"), ("envelope_minus_random", "EnvRand")):
        for key, kw in (("spec85", "SpecEightyFive"), ("spec90", "SpecNinety")):
            path = ("contrasts", pair, key)
            scalar("DSens" + word + kw, op, *path, "delta_sensitivity", pattern="%+.4f")
            ci("DSens" + word + kw + "CI", op, path, "ci95_lo", "ci95_hi")
    scalar("Prevalence", op, "prevalence")
    pc, adj = "p7c_paired_subgroup.json", "p17_subgroup_multiplicity.json"
    groups = {"severity:mild": "SevMild", "severity:moderate": "SevModerate", "severity:severe": "SevSevere",
              "race:White": "RaceWhite", "race:Black": "RaceBlack", "race:Asian": "RaceAsian",
              "sex:Female": "SexFemale", "sex:Male": "SexMale"}
    for gk, word in groups.items():
        scalar("PD" + word + "Delta", pc, "contrasts", "intensity_minus_random", "per_group", gk, "delta_auc", pattern="%+.5f")
        ci("PD" + word + "CI", adj, ("auc_family", "contrasts", gk), "simultaneous_ci95_lo", "simultaneous_ci95_hi", 5)
        put("PD" + word + "Sig", operation("yesno", ref(adj, "auc_family", "contrasts", gk, "simultaneous_excludes_zero")))
    for group, word, gk in (("race", "Race", "race:Black-minus-Asian"), ("sex", "Sex", "sex:Female-minus-Male")):
        scalar("PDDiff" + word, pc, "contrasts", "intensity_minus_random", group + "_differential_benefit", "delta_worst_minus_delta_best", pattern="%+.5f")
        ci("PDDiff" + word + "CI", adj, ("auc_family", "contrasts", gk), "simultaneous_ci95_lo", "simultaneous_ci95_hi", 5)
        put("PDDiff" + word + "Sig", operation("yesno", ref(adj, "auc_family", "contrasts", gk, "simultaneous_excludes_zero")))
    le = "p5_label_efficiency.json"
    for key, word in (("0.01", "One"), ("0.05", "Five"), ("0.10", "Ten"), ("0.25", "TwentyFive"), ("1.00", "Hundred")):
        for arm, aw in (("random", "Random"), ("intensity", "Oracle"), ("envelope", "Envelope"), ("cover", "Cover")):
            scalar("LE" + aw + word, le, "arms", arm, key, "auc_mean")
            scalar("LESD" + aw + word, le, "arms", arm, key, "auc_sd")
        scalar("LEN" + word, le, "arms", "cover", key, "n_train", pattern="%d")
    for field in ("repeats", "epoch"):
        scalar("LE" + field.capitalize(), le, field, pattern="%d")
    for key, word in (("0.05", "Five"), ("1.00", "Hundred")):
        put("LEGap" + word, fmt("%+.4f", operation("subtract", ref(le, "arms", "intensity", key, "auc_mean"), ref(le, "arms", "random", key, "auc_mean"))))
    so = "p16_subgroup_operating.json"
    scalar("SOPSpecTarget", so, "target_specificity", pattern="%.2f")
    scalar("SOPThreshold", so, "threshold")
    for arm, word in (("random", "Random"), ("intensity", "Oracle"), ("envelope", "Envelope")):
        scalar("SOPSens" + word, so, "arms", arm, "overall", "sensitivity")
        scalar("SOPSpec" + word, so, "arms", arm, "overall", "specificity")
    for g in ("White", "Black", "Asian", "Female", "Male"):
        group = "race" if g in ("White", "Black", "Asian") else "sex"
        scalar("SOPD" + g, so, "delta_random_to_intensity", group, g, "d_sensitivity", pattern="%+.4f")
        scalar("SOPN" + g + "Pos", so, "delta_random_to_intensity", group, g, "n_pos", pattern="%d")
        ci("SOPD" + g + "CI", adj, ("sensitivity_family", "contrasts", group + ":" + g), "simultaneous_ci95_lo", "simultaneous_ci95_hi")
        put("SOPD" + g + "Sig", operation("yesno", ref(adj, "sensitivity_family", "contrasts", group + ":" + g, "simultaneous_excludes_zero")))
    for word in ("EpFifty", "EpSeventyFive", "EpHundred"):
        for prefix, source in (("TAUCCover", "AUCCover"), ("TAUCAnatomyTwo", "AUCAnatomyTwo"),
                               ("TCoverRandom", "DCoverRandom"), ("TAnatomyTwoRandom", "DAnatomyTwoRandom")):
            if source + word in out:
                out[prefix + word] = {**out[source + word], "display_of": source + word}
    return out


NUMBER = re.compile(r"(?<![\d.])[-+]?(?:\d+(?:(?:\{,\}|,)\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def scalar_text(text):
    text = text.replace("{,}", "").replace(",", "").replace(r"\,", "").replace("$", "").replace(r"\%", "%")
    for _ in range(8):
        text = re.sub(r"\\(?:textbf|mathbf|mathrm|textnormal|textsc|emph)\{([^{}]*)\}", r"\1", text)
    return re.sub(r"\s+", "", text)


def structural_spans(source):
    """Constrained TeX grammar: never discard a whole TikZ node or table."""
    spans = []

    def add(start, end, reason):
        spans.append((start, end, reason))

    for _, _, start, end in assets.macros(source):
        add(start, end, "macro_definition_audited_separately")
    if r"\begin{document}" in source:
        add(0, source.index(r"\begin{document}") + len(r"\begin{document}"), "preamble")
    patterns = {
        "identifier": r"\\[A-Za-z]+|(?<![A-Za-z0-9])(?:fp(?:16|32)|[23]D|H[123]|v[12]|anatomy-v[12]|SHA-256|d1|data2vec|top-1)(?![A-Za-z0-9])",
        "file_or_hash": r"\\(?:texttt|url|path)\{(?:[^{}]*[\\/][^{}]*\.[A-Za-z]{1,8}|[0-9a-f]{32,64})\}",
        "reference_identifier": r"\\(?:cite[a-zA-Z]*|ref|autoref|eqref|label|input|include|includegraphics|bibliography)\*?\s*(?:\[[^\]]*\]\s*)*\{[^}]*\}",
        "layout": r"\\(?:vspace|hspace)\*?\{[^{}]*\}|\\setlength\{\\[A-Za-z]+\}\{[^{}]*\}|\\(?:cline|cmidrule)(?:\([^)]*\))?\{[^{}]*\}",
        "table_layout": r"\\begin\{(?:tabular|array)\}(?:\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}|\\multicolumn\{\d+\}\{[^{}]*\}",
        "figure_layout": r"\\begin\{(?:figure|table)\}\[[^\]]*\]",
    }
    for reason, pattern in patterns.items():
        for m in re.finditer(pattern, source):
            add(m.start(), m.end(), reason)
    for tikz in re.finditer(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", source, re.S):
        start = tikz.start()
        # Coordinates, styling and foreach ranges are conceptual layout, but
        # node text ("16 x 16", "4 targets", or an inserted AUC) is still audited.
        for pattern in (r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]",
                        r"\([-+]?\d[^()]*\)", r"\\foreach\s+\\\w+\s+in\s+\{[^{}]*\}",
                        r"\\(?:draw|fill)\s*\[[^\]]*\]"):
            for m in re.finditer(pattern, tikz[0]):
                add(start + m.start(), start + m.end(), "conceptual_tikz_coordinate_or_style")
    return spans


def literals(source, rel):
    spans = structural_spans(source)
    blocks = list(re.finditer(r"(?:[^\n]|\n(?!\s*\n))+", source))
    block_index, indices = 0, Counter()
    citation_tables = [(m.start(), m.end()) for m in re.finditer(
        r"\\begin\{table\}.*?\\end\{table\}", source, re.S) if r"\label{tab:counter}" in m[0]]
    arithmetic = []
    for m in re.finditer(r"(?<!\\)\$([^$]+)\$", source):
        plain = re.sub(r"\\(?:times|cdot|,)", "", m[1])
        if "=" in plain and not re.search(r"[A-Za-z\\]", plain):
            arithmetic.append((m.start(), m.end()))
    for match in NUMBER.finditer(source):
        while block_index + 1 < len(blocks) and blocks[block_index].end() <= match.start():
            block_index += 1
        block = blocks[block_index][0] if blocks else source
        # Table rows have stable identities even if surrounding rows move.
        line_start = source.rfind("\n", 0, match.start()) + 1
        line_end = source.find("\n", match.end())
        if line_end == -1:
            line_end = len(source)
        line = source[line_start:line_end]
        if "&" in line:
            block = line
        reason = next((r for a, b, r in spans if a <= match.start() and match.end() <= b), None)
        context_hash = digest_text(block)
        index = indices[context_hash]
        indices[context_hash] += 1
        before, after = source[max(0, match.start() - 45):match.start()], source[match.end():match.end() + 45]
        claim_class = "empirical_or_unclassified"
        if re.search(r"(?:epochs?[- ~]*(?:\d+\s*(?:,|and|to)\s*)*|seeds?\s+|batch(?: size)?(?: of)?\s+|layer\s+|depth[- ]*)$", before, re.I):
            claim_class = "protocol_descriptor"
        elif re.search(r"(?:M|K|f|W)\{?=\}?\s*$", before):
            claim_class = "protocol_descriptor"
        elif any(a <= match.start() < b for a, b in arithmetic):
            claim_class = "arithmetic_expression"
        elif re.match(r"(?:\\%|%)?\s*(?:CI|confidence|single-step)", after) or re.search(r"(?:alpha|alpha=|excludes\s*)$", before):
            claim_class = "statistical_convention"
        elif any(a <= match.start() < b for a, b in citation_tables):
            claim_class = "published_citation_claim"
        elif re.search(r"(?:Python|PyTorch|CUDA|cuDNN|NumPy|SciPy|scikit-learn|Tectonic|driver|pytest|statsmodels|seaborn|build)\s+[\d.]*$", before):
            claim_class = "software_environment_metadata"
        yield {"kind": "literal", "id": "%s:%s:%d" % (rel, context_hash[:16], index),
               "file": rel, "value": match[0], "line": source[:match.start()].count("\n") + 1,
               "context": ("TeX structure: " + reason) if reason else re.sub(r"\s+", " ", block).strip(),
               "context_sha256": context_hash,
               "token_index": index, "offset": match.start(),
               "claim_class": claim_class,
               "status": "structural" if reason else "unresolved",
               **({"classification": reason} if reason else {})}


def prose_bindings(source, rel, evidence):
    """Small explicit semantic clauses, never a global value allow-list."""
    result, errors = {}, []
    n = "(" + NUMBER.pattern + ")"

    def clause(pattern, expressions, label, flags=0):
        for match in re.finditer(pattern, source, flags):
            for i, expr in enumerate(expressions, 1):
                try:
                    bound = evidence.binding(expr, semantic_key=label)
                    value = match[i]
                    bound["status"] = "verified" if scalar_text(value) == scalar_text(bound["expected"]) else "mismatch"
                    result[match.start(i)] = bound
                except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
                    errors.append(label + ": " + str(exc))

    if rel == "auto/table_subgroup_trends.tex":
        clause(r"per checkpoint \(\$n\{=\}" + n + r"\$\)", [fmt("%d", ref("p7b_gap_trend.json", "n_probes"))], "trend checkpoint count")
        clause(r"per branch \(\$n\{=\}" + n + r"\$\)", [fmt("%d", ref("p7b_gap_trend.json", "trends", "race", "n_branches"))], "trend branch count")
    if rel != "main_submission.tex":
        return result, errors
    clause(r"\$N\{=\}" + n + r"\$ volumes \(" + n + r" positive / " + n + r" negative\)",
           [fmt("%d", ref("p1c_stats.json", "n_test")), fmt("%d", ref("p1c_stats.json", "n_pos")),
            fmt("%d", operation("subtract", ref("p1c_stats.json", "n_test"), ref("p1c_stats.json", "n_pos")))],
           "held-out split counts")
    clause(r"Delivered context also changes\s*\(\$" + n + r"\\%\$ to \$" + n + r"\\%\$",
           [fmt("%.1f", operation("percent", ref("geometry_batch64", a, "ctx_frac_of_grid"))) for a in ("random", "envelope")],
           "random versus envelope delivered context")
    clause(r"(?<![\w.])" + n + r"\s+(?:FairVision \\emph\{Training\}|training) slices",
           [fmt("%d", ref("geometry42", "_meta", "slices"))], "geometry measurement slice count")
    clause(r"Training\} slices \(" + n + r" volumes, " + n + r" slices each\)",
           [fmt("%d", ref("geometry42", "_meta", field)) for field in ("volumes", "slices_per_volume")],
           "geometry sampling layout")
    clause(r"seeds " + n + r", " + n + r" and " + n,
           [fmt("%d", ref("geometry%d" % seed, "_meta", "seed")) for seed in (42, 1234, 2026)],
           "geometry independent redraw seeds")
    clause(r"(?<![\w.])" + n + r"-slice sweep",
           [fmt("%d", ref("composition", "floor_curve", "0.21", "n"))], "composition sweep slice count")
    clause(r"sweep over \$" + n + r"\$ slices",
           [fmt("%d", ref("composition", "floor_curve", "0.21", "n"))], "composition sweep slice count")
    try:
        comp = evidence.load("composition")["rows"]
        ids = {r["arm"]: i for i, r in enumerate(comp)}
        clause(r"\$" + n + r"\\%\$ against \$" + n + r"\\%\$ for \\textsc\{envelope\}",
               [fmt("%.1f", ref("composition", "rows", ids[a], "pct_anat_hid")) for a in ("cover_f021", "envelope")],
               "composition cover versus envelope anatomy coverage")
    except (OSError, KeyError, ValueError):
        pass
    clause(r"(?:background|\\emph\{background\}) targets, (?:above|\\emph\{above\}) its \$" + n + r"\$ on anatomy",
           [fmt("%.3f", ref("background_skill", 1, "anat", "skill_vs_pos"))], "anatomy predictor skill against positional reference")
    clause(r"reference by \$" + n + r"\$ on\s*(?:\\emph\{background\}|background) targets",
           [fmt("%.3f", ref("background_skill", 1, "bg", "skill_vs_pos"))], "background predictor skill against positional reference")
    # Metadata counts: each clause names the exact category whose n is read.
    meta = "test_metadata_summary.json"
    for text, group, key in (("female", "sex", "Female"), ("male", "sex", "Male"),
                            ("non-hispanic", "ethnicity", "Non-Hispanic"), ("hispanic", "ethnicity", "Hispanic"),
                            ("english", "language", "English"), ("other", "language", "Other"), ("spanish", "language", "Spanish"),
                            ("married or partnered", "marital", "Married or partnered"), ("single", "marital", "Single"),
                            ("widowed", "marital", "Widowed"), ("divorced", "marital", "Divorced"),
                            ("unknown", "marital", "code_-1"), ("legally separated", "marital", "Legally separated")):
        clause(r"(?<![\w.])" + n + r"\s+" + re.escape(text) + r"(?=[;, .])",
               [fmt("%d", ref(meta, "subgroup_counts", group, key, "n"))], "test metadata " + group + "/" + key)
    # These are explicit named race calibration/specificity clauses.
    so = "p16_subgroup_operating.json"
    clause(r"under \\textsc\{random\} it realises \$" + n + r"\$ specificity in the\s*white stratum but \$" + n + r"\$ in the black stratum",
           [fmt("%.3f", ref(so, "arms", "random", "groups", "race", g, "specificity")) for g in ("White", "Black")],
           "random race-specific operating specificity")
    clause(r"under \\ArmBest\{\} the same threshold realises \$" + n + r"\$ and \$" + n + r"\$",
           [fmt("%.3f", ref(so, "arms", "intensity", "groups", "race", g, "specificity")) for g in ("White", "Black")],
           "centroid race-specific operating specificity")
    for group in ("white", "asian", "black"):
        clause(re.escape(group) + r"(?:\s+stratum)?\s*\(\$" + n + r"\$ to \$" + n + r"\$\)",
               [fmt("%.4f", ref(so, "arms", arm, "groups", "race", group.capitalize(), "ece")) for arm in ("random", "intensity")],
               group + " ECE change")
    for field, word in (("sensitivity", "sensitivity"), ("specificity", "specificity")):
        sign = "%+.3f"
        expression = lambda group: fmt(sign, operation("subtract",
            ref(so, "arms", "intensity", "groups", "race", group, field),
            ref(so, "arms", "random", "groups", "race", group, field)))
        if field == "sensitivity":
            clause(r"In the white stratum sensitivity rises by\s*\$" + n + r"\$",
                   [expression("White")], "white sensitivity change")
        else:
            clause(r"while specificity falls by \$" + n + r"\$", [expression("White")], "white specificity change")
            clause(r"In the black stratum specificity\s*\\emph\{rises\} by \$" + n + r"\$ and in the asian stratum by \$" + n + r"\$",
                   [expression("Black"), expression("Asian")], "black and Asian specificity changes")
    clause(r"sensitivity changes \(\$" + n + r"\$ and \$" + n + r"\$\)",
           [fmt("%+.3f", ref(so, "delta_random_to_intensity", "race", g, "d_sensitivity")) for g in ("Black", "Asian")],
           "black and Asian sensitivity changes")
    clause(r"both strata gain sensitivity \(\$" + n + r"\$ female, \$" + n + r"\$ male\)",
           [fmt("%+.3f", ref(so, "delta_random_to_intensity", "sex", g, "d_sensitivity")) for g in ("Female", "Male")],
           "female and male sensitivity changes")
    clause(r"and lose\s*specificity \(\$" + n + r"\$ and \$" + n + r"\$\)",
           [fmt("%+.3f", operation("subtract", ref(so, "arms", "intensity", "groups", "sex", g, "specificity"),
                                  ref(so, "arms", "random", "groups", "sex", g, "specificity"))) for g in ("Female", "Male")],
           "female and male specificity changes")
    return result, errors


ARM_LABELS = {r"\textsc{random}": "random", r"\textsc{envelope}": "envelope",
              r"\ArmBest{}": "oracle", r"\textsc{cover}": "cover-f021",
              r"\textsc{anatomy-v1}": "anatomy-v1", r"\textsc{anatomy-v2}": "anatomy-v2",
              "ancestor": "ancestor", "random": "random", "anatomy-v2": "anatomy-v2"}


def table_bindings(rel, source, evidence):
    """Map complete numeric table cells by semantic row keys and column roles."""
    results, errors = {}, []

    def cell(offset, text, expression, label):
        try:
            binding = evidence.binding(expression, semantic_key=label)
            # Check the complete cell, so mutated endpoint order/punctuation,
            # missing values, and additional quantities cannot be hidden.
            numeric_actual = [scalar_text(m[0]) for m in NUMBER.finditer(text)]
            numeric_expected = [scalar_text(m[0]) for m in NUMBER.finditer(binding["expected"])]
            matches = list(NUMBER.finditer(text))
            status = "verified" if numeric_actual == numeric_expected else "mismatch"
            if len(matches) != len(numeric_expected):
                errors.append("numeric cell shape differs: " + label)
            if not matches:
                errors.append("required numeric table cell missing: " + label)
            for m in matches:
                results[offset + m.start()] = {**binding, "status": status}
        except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
            errors.append(label + ": " + str(exc))

    table_label = None
    seen, expected_rows = [], None
    offset = 0
    for line in source.splitlines(keepends=True):
        label = re.search(r"\\label\{(tab:[^}]+)\}", line)
        if label:
            table_label = label[1]
        if "&" not in line or not line.rstrip().endswith(r"\\"):
            offset += len(line)
            continue
        parts = line.split("&")
        positions, pos = [], offset
        for part in parts:
            positions.append(pos)
            pos += len(part) + 1
        arm = ARM_LABELS.get(parts[0].strip())
        specs = {}
        key = parts[0].strip()
        try:
            if rel == "auto/table_allprobes.tex" and arm:
                epoch, precision = int(parts[1]), parts[2].strip()
                excluded = re.search(r"\\emph\{(excluded|retracted)\}", parts[-1])
                src = "p1b_full_inventory.json" if excluded else "p1c_stats.json"
                container = "records" if excluded else "table"
                data = evidence.load(src)[container]
                matches = [(i, r) for i, r in enumerate(data) if r.get("arm", "").replace("-RETRACTED", "") == arm
                           and r.get("epoch") == epoch and r.get("precision") == precision
                           and (not excluded or r.get("status") == excluded[1])]
                if len(matches) != 1:
                    raise ValueError("ambiguous/missing arm-epoch-precision/status")
                i, row = matches[0]
                specs = {1: fmt("%d", ref(src, container, i, "epoch")), 3: fmt("%.4f", ref(src, container, i, "auc"))}
                if not excluded:
                    specs[4] = fmt(r"[%.4f,\,%.4f]", ref(src, container, i, "ci95_lo"), ref(src, container, i, "ci95_hi"))
                seen.append((arm, epoch, precision, excluded[1] if excluded else "valid"))
            elif rel == "auto/table_fp32.tex" and arm:
                ep = int(parts[1])
                src = "p3b_fp32.json"
                matches = [i for i, r in enumerate(evidence.load(src)["rows"]) if r["arm"] == arm and r["epoch"] == ep]
                if len(matches) != 1:
                    raise ValueError("ambiguous/missing fp32 row")
                i = matches[0]
                for col, field, pattern in ((1, "epoch", "%d"), (2, "auc_fp16", "%.6f"), (3, "auc_fp32", "%.6f"),
                                            (4, "delta_fp32_minus_fp16", "%+.6f"), (5, "delong_p", "%.3f")):
                    specs[col] = fmt(pattern, ref(src, "rows", i, field))
                seen.append((arm, ep))
            elif rel == "auto/table_labeleff.tex" and re.fullmatch(r"\d+\\%", parts[0].strip()):
                frac = float(parts[0].strip().replace(r"\%", "")) / 100
                key = "%.2f" % frac
                src = "p5_label_efficiency.json"
                i = evidence.load(src)["fractions"].index(frac)
                specs[0] = fmt("%g", operation("percent", ref(src, "fractions", i)))
                specs[1] = fmt("%d", ref(src, "arms", "cover", key, "n_train"))
                for col, a in enumerate(("random", "intensity", "envelope", "cover"), 2):
                    specs[col] = fmt("%.4f", ref(src, "arms", a, key, "auc_mean"))
                seen.append(key)
            elif rel == "auto/table_operating.tex" and arm:
                src = "p8b_operating_points.json"
                key = {"0.85": "spec85", "0.90": "spec90"}[parts[1].strip()]
                a = "intensity" if arm == "oracle" else arm
                specs[1] = fmt("%.2f", ref(src, "target_specificities", 0 if key == "spec85" else 1))
                for col, field in enumerate(("sensitivity", "specificity", "ppv", "npv"), 2):
                    specs[col] = fmt("%.4f", ref(src, "arms", a, "at", key, field))
                specs[6] = fmt("%.4f", ref(src, "arms", a, "brier"))
                specs[7] = fmt("%.4f", ref(src, "arms", a, "ece_15bin"))
                seen.append((a, key))
            elif rel == "auto/table_paired_subgroup.tex" and "," in key:
                gk = key.replace(", ", ":")
                pc, adj = "p7c_paired_subgroup.json", "p17_subgroup_multiplicity.json"
                path = ("contrasts", "intensity_minus_random", "per_group", gk)
                specs[1] = fmt("%d", ref(pc, *path, "n"))
                specs[2] = fmt("%+.5f", ref(pc, *path, "delta_auc"))
                specs[3] = fmt(r"[%+.5f,\,%+.5f]", ref(adj, "auc_family", "contrasts", gk, "simultaneous_ci95_lo"),
                               ref(adj, "auc_family", "contrasts", gk, "simultaneous_ci95_hi"))
                seen.append(gk)
            elif rel == "auto/table_subgroup_operating.tex" and key in ("Asian", "Black", "White", "Female", "Male"):
                src, adj = "p16_subgroup_operating.json", "p17_subgroup_multiplicity.json"
                group = "sex" if key in ("Female", "Male") else "race"
                path = ("delta_random_to_intensity", group, key)
                specs[1] = fmt("%d", ref(src, *path, "n_pos"))
                specs[2] = fmt("%+.4f", ref(src, *path, "d_sensitivity"))
                specs[3] = fmt(r"[%+.4f,\,%+.4f]", ref(adj, "sensitivity_family", "contrasts", group + ":" + key, "simultaneous_ci95_lo"),
                               ref(adj, "sensitivity_family", "contrasts", group + ":" + key, "simultaneous_ci95_hi"))
                seen.append(key)
            elif rel == "auto/table_subgroup_trends.tex" and key in ("sex", "race", "ethnicity", "language", "marital status", "age", "disease severity"):
                src = "p7b_gap_trend.json"
                attr = {"sex": "gender", "marital status": "maritalstatus", "disease severity": "severity"}.get(key, key)
                path = ("trends", attr)
                specs[2] = fmt("%.4f--%.4f", ref(src, *path, "gap_min"), ref(src, *path, "gap_max"))
                for col, field, pattern in ((3, "spearman_rho", "%+.3f"), (6, "branch_spearman_rho", "%+.3f")):
                    specs[col] = fmt(pattern, ref(src, *path, field))
                for col, field in ((4, "spearman_p"), (5, "q_bh_across_attributes"), (7, "branch_spearman_p")):
                    specs[col] = operation("pvalue", ref(src, *path, field))
                # Worst-group label contains protocol cutpoints. Bind all
                # numeric pieces, including those in the source category key.
                counts = evidence.load(src)["worst_group_consistency"][attr]["counts"]
                top = max(counts, key=counts.get)
                shown = top.replace("<", "$<$").replace(">", "$>$").replace("-6", "$-6$").replace("-2", "$-2$")
                n = counts[top]
                for m in NUMBER.finditer(parts[1]):
                    results[positions[1] + m.start()] = {
                        "status": "verified" if [x[0] for x in NUMBER.finditer(parts[1])] ==
                            [x[0] for x in NUMBER.finditer(shown + " (%d/%d)" % (n, evidence.load(src)["n_probes"]))] else "mismatch",
                        "source": src, "pointer": ptr("worst_group_consistency", attr, "counts"),
                        "operation": "argmax category and count / n_probes",
                        "source_hashes": {src: evidence.hashes[src]}}
                seen.append(attr)
            elif rel == "main_submission.tex" and arm and table_label in ("tab:geom", "tab:geomprov", "tab:delivered"):
                key = arm
                arm = {"cover-f021": "cover", "anatomy-v2": "anatomy"}.get(arm, arm)
                if table_label == "tab:geom":
                    for col, field, percent in ((1, "hidden_share_of_all_anat", False), (2, "hidden_pct_on_anat", False),
                                                (3, "hidden_frac_of_grid", True), (4, "ctx_frac_of_grid", True),
                                                (5, "n_slots_mean", False)):
                        r = ref("geometry42", arm, field)
                        specs[col] = fmt("%.1f", operation("percent", r) if percent else r)
                elif table_label == "tab:geomprov":
                    for col, field in ((1, "hidden_share_of_all_anat"), (3, "n_slots_mean")):
                        specs[col] = fmt("%.2f", ref("geometry42", arm, field))
                        specs[col + 1] = fmt("%.2f", operation("stdev", *[ref("geometry%d" % seed, arm, field) for seed in (42, 1234, 2026)]))
                else:
                    specs[1] = fmt("%.1f", operation("percent", ref("geometry42", arm, "ctx_frac_of_grid")))
                    specs[2] = fmt("%.1f", operation("percent", ref("geometry_batch64", arm, "ctx_frac_of_grid")))
            elif rel == "main_submission.tex" and table_label == "tab:intersectional" and (r"$\times$" in key or key == "max$-$min gap"):
                src = "intersectional"
                probes = ("sweep_random_ep100", "frozen_meanpool_mirage_ep100", "sweep_oracle_ep100")
                data = evidence.load(src)
                if key == "max$-$min gap":
                    for col, probe in enumerate(probes, 3):
                        specs[col] = fmt("%.4f", ref(src, probe, "gap"))
                else:
                    group = re.sub(r"\s+", " ", key.replace(r"$\times$", "x")).strip()
                    selected = {}
                    for probe in probes:
                        matches = [i for i, row in enumerate(data[probe]["cells"]) if row["subgroup"] == group]
                        if len(matches) != 1:
                            raise ValueError("ambiguous/missing intersectional group")
                        selected[probe] = matches[0]
                    specs[1] = fmt("%d", ref(src, probes[0], "cells", selected[probes[0]], "n"))
                    specs[2] = fmt("%d", ref(src, probes[0], "cells", selected[probes[0]], "n_pos"))
                    for col, probe in enumerate(probes, 3):
                        i = selected[probe]
                        specs[col] = fmt("%.4f [%.4f, %.4f]", *[ref(src, probe, "cells", i, field) for field in ("auc", "ci_lo", "ci_hi")])
                    specs[6] = fmt("%+.4f", operation("subtract", ref(src, probes[2], "cells", selected[probes[2]], "auc"),
                                                     ref(src, probes[0], "cells", selected[probes[0]], "auc")))
            elif rel == "main_submission.tex" and table_label == "tab:severity" and key.split(" ")[0] in ("mild", "moderate", "severe"):
                severity = key.split(" ")[0]
                src, probe = "subgroup_auc.json", "sweep_random_ep100"
                rows = evidence.load(src)[probe]["subgroups"]["severity"]["levels"]
                ids = [i for i, r in enumerate(rows) if r["subgroup"].startswith(severity + " ")]
                if len(ids) != 1:
                    raise ValueError("ambiguous severity stratum")
                specs[0] = ref(src, probe, "subgroups", "severity", "levels", ids[0], "subgroup")
                specs[1] = fmt("%d", ref(src, probe, "subgroups", "severity", "levels", ids[0], "n_pos"))
            elif rel == "main_submission.tex" and table_label == "tab:finetuned" and arm:
                src = "p1b_full_inventory.json"
                head = {("oracle", "mean-pool"): "meanpool", ("oracle", "cross-attention"): "crossattn", ("oracle", "attentive"): "d1",
                        ("random", "mean-pool"): "mean_pool", ("random", "cross-attention"): "cross_attn_pool", ("random", "attentive"): "attentive"}[(arm, parts[1].strip())]
                ids = [i for i, r in enumerate(evidence.load(src)["records"]) if r.get("family") == "finetune" and r["arm"] == arm and r["tag"].split("/")[-1] == head]
                if len(ids) != 1:
                    raise ValueError("ambiguous fine-tuned head")
                specs[2] = fmt("%.6f", ref(src, "records", ids[0], "auc"))
            elif rel == "main_submission.tex" and table_label == "tab:contrasts" and parts[2].strip().startswith(r"\D"):
                match = re.fullmatch(r"\\D(EnvelopeRandom|OracleRandom|OracleEnvelope)(EpFifty|EpSeventyFive|EpHundred)", parts[2].strip())
                if not match:
                    raise ValueError("unrecognised primary contrast macro")
                arm = "envelope" if match[1] == "EnvelopeRandom" else "oracle"
                epoch = {"EpFifty": 50, "EpSeventyFive": 75, "EpHundred": 100}[match[2]]
                src = "p1c_stats.json"
                ids = [i for i, r in enumerate(evidence.load(src)["table"]) if r["key"] == "%s@ep%d@fp16" % (arm, epoch)]
                if len(ids) != 1:
                    raise ValueError("ambiguous primary contrast epoch")
                specs[1] = fmt("%d", ref(src, "table", ids[0], "epoch"))
        except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
            errors.append("%s row %s: %s" % (rel, key, exc))
        for col, expression in specs.items():
            if col >= len(parts):
                errors.append("%s required column missing: %d" % (rel, col))
            else:
                cell(positions[col], parts[col], expression, "%s/%s/%s" % (rel, table_label or "", key + "/column" + str(col)))
        offset += len(line)
    if rel == "auto/table_allprobes.tex":
        expected_rows = [(r["arm"], r["epoch"], r["precision"], "valid") for r in evidence.load("p1c_stats.json")["table"]]
        expected_rows += [(r["arm"].replace("-RETRACTED", ""), r["epoch"], r["precision"], r["status"])
                         for r in evidence.load("p1b_full_inventory.json")["records"] if r.get("status") in ("excluded", "retracted")]
    elif rel == "auto/table_fp32.tex":
        expected_rows = [(r["arm"], r["epoch"]) for r in evidence.load("p3b_fp32.json")["rows"]]
    elif rel == "auto/table_labeleff.tex":
        expected_rows = ["%.2f" % x for x in evidence.load("p5_label_efficiency.json")["fractions"]]
    elif rel == "auto/table_operating.tex":
        expected_rows = [(a, k) for a in ("random", "intensity", "envelope") for k in ("spec85", "spec90")]
    elif rel == "auto/table_paired_subgroup.tex":
        expected_rows = ["severity:mild", "severity:moderate", "severity:severe", "race:White", "race:Black", "race:Asian", "sex:Female", "sex:Male"]
    elif rel == "auto/table_subgroup_operating.tex":
        expected_rows = ["Asian", "Black", "White", "Female", "Male"]
    elif rel == "auto/table_subgroup_trends.tex":
        expected_rows = ["gender", "race", "ethnicity", "language", "maritalstatus", "age", "severity"]
    if expected_rows is not None and Counter(seen) != Counter(expected_rows):
        errors.append(rel + ": missing, duplicated or unrecognised required table rows")
    return results, errors


def read_reviews(paper, evidence, review_file=None):
    path = Path(review_file) if review_file else Path(paper) / "numeric_reviews.json"
    if not path.exists():
        if review_file:
            raise FileNotFoundError("explicit numeric review file is missing: " + str(path))
        return {"version": 1, "macros": {}, "literals": [], "figures": []}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if type(data.get("version")) is not int or data.get("version") != 1:
        raise ValueError("numeric review schema version must be 1")
    if set(data) - {"version", "sources", "macros", "literals", "figures", "scope"}:
        raise ValueError("unknown numeric review schema fields")
    if not isinstance(data.get("macros", {}), dict) or not isinstance(data.get("sources", {}), dict):
        raise ValueError("sources and macros must be objects")
    if not isinstance(data.get("literals", []), list) or not isinstance(data.get("figures", []), list):
        raise ValueError("literal and figure receipts must be arrays")
    for name, spec in data.get("sources", {}).items():
        if name in evidence.specs or name in evidence.cache:
            raise ValueError("review cannot override built-in evidence: " + name)
        if spec.get("root") not in ("repo", "paper", "stats") or not re.fullmatch("[0-9a-f]{64}", spec.get("sha256", "")):
            raise ValueError("custom sources require an allowed root and immutable sha256")
        if Path(spec["path"]).name == "auto_numbers.tex":
            raise ValueError("rendered numbers are not source evidence")
        evidence.specs[name] = spec
    return data


def figure_context(text, path):
    name = Path(path).name
    found = []
    for figure in re.finditer(r"\\begin\{figure\}(.*?)\\end\{figure\}", text, re.S):
        if any(Path(g).name in (name, Path(name).stem) for g in assets.graphics(figure[0])):
            match = re.search(r"\\caption\s*(?:\[[^\]]*\])?\s*", figure[0])
            if match:
                found.append(assets.group(figure[0], match.end())[0])
    if len(found) != 1:
        raise ValueError("figure receipt requires exactly one unambiguous caption: " + path)
    return found[0]


def write_report(path, report):
    lines = ["# Numeric coverage report", "",
             "Status: **%s**. No manuscript quantities were changed." % ("PASS" if report["ALL_PASS"] else "BLOCKED"),
             "Verification here means independently reformatting exact stored source fields, not rerunning scientific analyses.",
             "Historical illustration review is kept distinct from programmatic plotted-value verification.", "",
             "Binding/review input: `%s` (SHA-256 `%s`)." % (report.get("review_path", "none"), report.get("review_sha256", "none")), "",
             "## Coverage", ""]
    lines += ["- %s: %s" % (key, n) for key, n in sorted(report.get("counts", {}).items())]
    lines += ["", "## Gate errors", ""] + ["- " + e for e in report.get("errors", [])]
    unresolved = [r for r in report.get("items", []) if r["status"] in ("unresolved", "mismatch", "undefined")]
    lines += ["", "## Exact unresolved claims", "",
              "Structural TeX numbers have been removed from this list. Protocol/formula/citation literals below still require an explicit review; they are not automatically labelled empirical measurements."]
    classes = Counter(r.get("claim_class", r["kind"]) for r in unresolved)
    lines += ["", "Unresolved classification (none are silently approved):"]
    lines += ["- %s: %d" % pair for pair in sorted(classes.items())]
    for row in (r for r in unresolved if r["kind"] != "literal"):
        lines += ["", "### `%s` (%s; %s)" % (row["id"], row["kind"], row["status"])]
        if "value" in row:
            lines += ["- Rendered token: `%s`" % row["value"]]
        if "expected" in row:
            lines += ["- Source-derived expected: `%s`" % row["expected"]]
        if "context" in row:
            lines += ["- Context: " + row["context"]]
        if "action" in row:
            lines += ["- Action: " + row["action"]]
    contexts = {}
    for row in unresolved:
        if row["kind"] == "literal":
            contexts.setdefault((row["file"], row["context_sha256"]), []).append(row)
    lines += ["", "## Literal contexts", "",
              "Each listed token is independently unresolved. Token indices address the normalized context hash, not manuscript line numbers."]
    for (file, context_hash), tokens in contexts.items():
        lines += ["", "### `%s` — context `%s`" % (file, context_hash),
                  "- Tokens: " + "; ".join("`%s` (index %d; %s)" % (r["value"], r["token_index"], r["claim_class"]) for r in tokens),
                  "- Context: " + tokens[0]["context"]]
        for row in tokens:
            if row["status"] == "mismatch":
                lines.append("- Mismatch at index %d: expected `%s`." % (row["token_index"], row.get("expected", "see machine report")))
    lines += ["", "## Review mechanism", "", __doc__.split("\n\n", 1)[1].strip(), "",
              "Known local raster producers are also executed CPU-only into memory/project-local scratch. Independent artist validators compare source-derived values, intervals and numeric annotations before requiring exact equality to the delivered PNG. The temporary producer outputs are removed; no manuscript asset is replaced.",
              "No blanket literal approval, closest-value matching, self-validation against auto_numbers, or raster hash-as-numeric-proof is accepted.",
              "Source-writing can move lines without invalidating a review; changing its paragraph/table-row content invalidates its context hash.",
              "Run `python autopilot\\p15_verify_numbers.py --report <coverage.json> --markdown-report <report.md>` after source stabilization.",
              "The machine-readable report includes every token, semantic table key, source pointer/executable expression, and evidence hash."]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def review_literal(row, entry, evidence):
    if len(set(entry) & {"expression", "review", "assertion"}) != 1:
        raise ValueError("literal receipt needs exactly one expression, assertion or review")
    if entry.get("value") != row["value"]:
        raise ValueError("literal review value mismatch")
    if "assertion" in entry:
        assertion = entry["assertion"]
        display = assertion["bound"]
        text = scalar_text(display)
        scientific = re.fullmatch(r"(?:(%s)\\times)?10\^\{([-+]?\d+)\}" % NUMBER.pattern, text)
        if scientific:
            exponent = int(scientific[2])
            if not -308 <= exponent <= 308:
                raise ValueError("assertion exponent is outside supported finite precision")
            bound = (float(scientific[1]) if scientific[1] else 1) * 10.0 ** exponent
        elif re.fullmatch(NUMBER, text):
            bound = float(text)
        else:
            raise ValueError("assertion bound must be a decimal or explicit TeX power of ten")
        tokens = list(NUMBER.finditer(display))
        index = assertion.get("component", 0)
        if type(index) is not int or not 0 <= index < len(tokens):
            raise ValueError("invalid assertion display component")
        if scalar_text(row["value"]) != scalar_text(tokens[index][0]):
            raise ValueError("assertion display does not match the literal token")
        value = evidence.evaluate(assertion["expression"])
        if type(value) not in (int, float) or not math.isfinite(value) or not math.isfinite(bound):
            raise ValueError("assertion requires finite numeric evidence")
        relation = assertion["relation"]
        if relation not in ("lt", "le", "gt", "ge"):
            raise ValueError("unsupported literal inequality")
        passed = {"lt": value < bound, "le": value <= bound, "gt": value > bound, "ge": value >= bound}[relation]
        metadata = evidence.binding(fmt("%.17g", assertion["expression"]))
        return {**metadata, "status": "verified" if passed else "mismatch",
                "verification": "exact_source_inequality", "assertion": assertion,
                "computed_value": value, "bound_value": bound,
                "expected": tokens[index][0],
                **({} if passed else {"action": "Source value %.17g does not satisfy %s %.17g" % (value, relation, bound)})}
    if "expression" in entry:
        bound = evidence.binding(entry["expression"])
        if "component" in entry:
            index = entry["component"]
            tokens = list(NUMBER.finditer(bound["expected"]))
            if type(index) is not int or not 0 <= index < len(tokens):
                raise ValueError("invalid source-formatted display component")
            bound["expected_display"] = bound["expected"]
            bound["expected"] = tokens[index][0]
        return {**bound, "status": "verified" if scalar_text(row["value"]) == scalar_text(bound["expected"]) else "mismatch"}
    review = entry["review"]
    kind = review["kind"]
    if kind not in ("protocol", "formula", "citation"):
        raise ValueError("empirical claims require executable source bindings")
    if not review.get("reviewer") or not review.get("rationale"):
        raise ValueError("review requires reviewer and rationale")
    refs = review.get("evidence", [])
    if not refs:
        raise ValueError("review requires source evidence")
    hashes = {}
    for item in refs:
        data = evidence.load(item["source"])
        hashes[item["source"]] = evidence.hashes[item["source"]]
        pinned = item.get("sha256") or evidence.specs.get(item["source"], {}).get("sha256")
        if pinned != hashes[item["source"]]:
            raise ValueError("review evidence must pin its source hash")
        if "pointer" in item:
            pointer(data, item["pointer"])
        elif "excerpt" in item:
            if not isinstance(data, str) or not item["excerpt"] or item["excerpt"] not in data:
                raise ValueError("review evidence excerpt does not match")
        else:
            raise ValueError("review needs an exact pointer or excerpt")
    if kind == "citation" and (not review.get("immutable_locator") or not review.get("locator")):
        raise ValueError("citation review requires immutable publication and page/table locator")
    return {"status": "reviewed_" + kind, "review": review, "source_hashes": hashes}


def figure_receipt(item, entry, evidence, caption):
    if entry.get("sha256") != item["sha256"] or entry.get("caption_sha256") != digest_text(caption):
        raise ValueError("figure or caption changed since evidence receipt")
    inputs = entry.get("inputs", [])
    if not inputs:
        raise ValueError("figure receipt requires evidence inputs")
    for inp in inputs:
        evidence.evaluate(inp)
        if inp.get("sha256") != evidence.hashes[inp["source"]]:
            raise ValueError("figure source input hash missing or changed")
    v = entry["validation"]
    if v.get("method") in ("reviewed_historical_illustration", "reviewed_source_illustration"):
        if not v.get("reviewer") or not v.get("limitations") or v.get("quantitative_scope") != "illustrative_only":
            raise ValueError("historical illustration needs reviewer, limitations and illustrative-only scope")
        return {"status": v["method"], "mathematically_verified": False,
                "validation": v, "inputs": inputs}
    if v.get("method") not in ("svg_coordinates", "svg_bars"):
        raise ValueError("unsupported independent figure validation method")
    path = assets.safe_path(evidence.roots["paper"], item["path"])
    if path.suffix.lower() != ".svg" or not v.get("series"):
        raise ValueError("coordinate validation requires SVG series")
    if assets.sha256(path) != item["sha256"]:
        raise ValueError("SVG changed since the asset inventory")
    try:
        from .numeric_svg_review import verify
    except ImportError:
        from numeric_svg_review import verify
    result = verify(path, v, evidence, inputs)
    if assets.sha256(path) != item["sha256"]:
        raise ValueError("SVG changed during coordinate verification")
    return {**result, "validation": v, "inputs": inputs}
