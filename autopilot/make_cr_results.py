"""Camera-ready replication results: macros, tables, scenario text and figure (cr_seed_v1).

Turns the post-freeze outputs of the registered analysis
(paper/genai4health2026/PREREGISTRATION_cr_seed_v1.md, incl. amendment 1) into paper inputs:

  * ``autopilot/cr_stats.py final`` JSON (primary analysis, outcome labels, matched control,
    pooling variants),
  * ``autopilot/cr_md_regression.py --split test`` JSON (MD severity endpoint; optional),
  * the probe directories (results.json, sealed manifest, unseal receipt), and
  * delivered-budget evidence for the matched control: the RANDOM-CB run's own audit
    (schema ``cr_cb_budget_audit_v1``) if available, else the I5 delivered-mask audit
    (budgets) plus the I7 matcher bench (per-image match counts, target purity).

Outputs (``--out-dir``, default paper/genai4health2026/auto):
  cr_results.tex             numeric macros ``\\Rep*`` (typed expressions, re-evaluated by p15)
                             and scenario text macros (digit-free; every number is a macro)
  cr_results.values.json     sidecar: full precision, expressions, source hashes, scenario rule
  cr_table_replication.tex   seed-replication table (tab:replication, main text)
  cr_table_control.tex       matched-budget control table (tab:control)
  cr_table_pooling.tex       pooling variants x arms (tab:pooling)
  cr_table_seeds.tex         per-run table with paired CIs (tab:replication_runs, appendix)
  cr_table_md.tex            MD severity table (tab:md, appendix; only with --md)
  cr_figure_seed_spread.tex  figure float for the seed-spread figure (fig:seed_spread)
  fig_cr_seed_spread.pdf/.png + fig_cr_seed_spread.evidence.json
  cr_tldr.txt                OpenReview TL;DR (plain text, <= 250 characters)

Every input is copied byte-identically into ``--evidence-dir`` (inside the repository for the
real run) and every numeric macro is a typed expression over those copies, evaluated with
autopilot/numeric_bindings.py exactly as p15_verify_numbers.py re-evaluates it.
``--update-reviews`` writes the bindings into numeric_reviews.json (sources ``rep_*``, macros
``Rep*``; make_cr_numbers.py owns ``CR*`` and is not affected).

Scenario selection (per guided arm, primary pooling) uses the ``final`` labels unchanged:
consistent_direction -> A, within_run_to_run_variation -> B, reversed -> C,
incomplete/not_available -> I; both arms equal -> that scenario, else "mixed".
Matched control (pre-registration section 4 and amendment 2): RANDOM-CB is registered at seed
1234 only (other seeds are refused). P = AUC(C1) - AUC(CB), B = AUC(CB) - AUC(R1), r = the RANDOM
range of section 3 (cr_stats ``random_range``); descriptions only from runs finished before
the freeze:
  M0  |P| <= r                     (placement difference not detected)
  M-  P < -r                       (placement difference reversed)
  M+  P > r and P >= (P + B) / 2   (placement accounts for most of the gap)
  M_partial  P > r and P < (P + B) / 2 (both contribute)
  without a description: M_unassessed (P > r but no on-time B), M_desc (P uses a late run),
  M_noP (no CENTROID result at seed 1234), M_na (no RANDOM-CB result), M_incomplete (no r).
Scenario prose is literal for its label: B wording names the failing condition (a
non-positive continuation and/or a mean not above r), never "smaller" or "comparable".

Usage (repo root, after the freeze):
  python autopilot/make_cr_results.py build --final <final.json> --md <md_test.json> ^
      --inventory <inventory.json> --budget-i7 <real_bench.json> [--update-reviews]
  python autopilot/make_cr_results.py build ... --check
  python autopilot/make_cr_results.py cb-audit --log <RANDOM-CB train log> --out cb_audit.json
  python autopilot/make_cr_results.py integrate --paper-dir <scratch copy> --results-dir <out>
Synthetic dry runs need ``--allow-synthetic`` and an --out-dir/--evidence-dir outside the paper
and repository.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

try:
    from . import numeric_bindings as numeric
    from . import release_assets as assets
    from . import cr_stats
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numeric_bindings as numeric
    import release_assets as assets
    import cr_stats

REPO = assets.REPO
PAPER = assets.PAPER
DEFAULT_OUT = PAPER / "auto"
CR_DIR = "autopilot/investigations/camera_ready_20261008"
DEFAULT_EVIDENCE = REPO / CR_DIR / "results_cr_seed_v1"
DEFAULT_STATS = Path(r"D:\jepa_phase0\autopilot_out\p1_stats")
DEFAULT_I5 = REPO / CR_DIR / "mask_audit" / "delivered_mask_audit_trained_3seeds_100draws.json"
REVIEWS = PAPER / "numeric_reviews.json"
SCHEMA = "cr_results_v1"

PRIMARY = cr_stats.PRIMARY_VARIANT
GUIDED = ("centroid", "envelope")
SEED_ARMS = ("random", "centroid", "envelope")
SEED_LETTERS = {1234: "A", 5678: "B", 9012: "C"}
# Registered runs in pre-registration order (section 1 and amendment 1).
REGISTERED_RUNS = (("random", 1234), ("centroid", 1234), ("envelope", 1234),
                   ("random", 5678), ("centroid", 5678), ("random_cb", 1234),
                   ("envelope", 5678), ("random", 9012), ("centroid", 9012),
                   ("envelope", 9012))
ARM_WORD = {"random": "Random", "centroid": "Centroid", "envelope": "Envelope",
            "random_cb": "RandomCB"}
ARM_TEX = {"random": r"\textsc{random}", "centroid": r"\ArmBest{}",
           "envelope": r"\textsc{envelope}", "random_cb": r"\textsc{random-cb}"}
ARM_PLAIN = {"random": "RANDOM", "centroid": "CENTROID", "envelope": "ENVELOPE",
             "random_cb": "RANDOM-CB"}
VARIANTS = ("patchmean_slicemean", "patchmax_slicemean", "patchmean_slicemax",
            "patchmean_slicemeanmax")
VARIANT_WORD = {"patchmean_slicemean": "Mean", "patchmax_slicemean": "PatchMax",
                "patchmean_slicemax": "SliceMax", "patchmean_slicemeanmax": "MeanMax"}
VARIANT_TEX = {"patchmean_slicemean": "mean, mean (primary)",
               "patchmax_slicemean": "max, mean",
               "patchmean_slicemax": "mean, max",
               "patchmean_slicemeanmax": "mean, mean and max"}
VARIANT_SHORT = {"patchmax_slicemean": "patch maxima within B-scans",
                 "patchmean_slicemax": "maxima across B-scans",
                 "patchmean_slicemeanmax": "concatenated means and maxima"}
LABEL_CODE = {"consistent_direction": "A", "within_run_to_run_variation": "B",
              "reversed": "C", "incomplete": "I", "not_available": "I"}
LABEL_TEX = {"consistent_direction": "consistent direction",
             "within_run_to_run_variation": "within run-to-run variation",
             "reversed": "reversed", "incomplete": "incomplete", "not_available": "not available"}
LABEL_SHORT = {"consistent_direction": "consistent", "within_run_to_run_variation": "within",
               "reversed": "reversed", "incomplete": "incomplete", "not_available": "---"}
LABEL_RANK = {"A": 0, "B": 1, "C": 2, "I": 3}
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve")
NULL_CHANCE = {2: "one in four", 3: "one in eight", 4: "one in sixteen"}
CI4 = r"[%+.4f,\,%+.4f]"
CI2 = r"[%+.2f,\,%+.2f]"
TLDR_LIMIT = 250
OUTPUT_FILES = ("cr_results.tex", "cr_table_replication.tex", "cr_table_control.tex",
                "cr_table_pooling.tex", "cr_table_seeds.tex", "cr_table_md.tex",
                "cr_figure_seed_spread.tex", "cr_tldr.txt")

ref, fmt, op = numeric.ref, numeric.fmt, numeric.operation


class InputError(RuntimeError):
    """The inputs are not the registered, unsealed, identity-complete results."""


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8-sig")), sha256_bytes(raw)


def word(n, capital=False):
    text = NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else None
    if text is None:
        raise InputError("no number word for %r" % n)
    return text.capitalize() if capital else text


def join_and(items):
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def thousands(text):
    """Display grouping only; p15 normalises {,} before comparing (as make_cr_numbers)."""
    return "{,}".join(reversed([text[::-1][i:i + 3][::-1] for i in range(0, len(text), 3)]))


# ---------------------------------------------------------------------------
# Input verification (before any number is used)
# ---------------------------------------------------------------------------

def _ident(identity):
    ident = cr_stats._identity({"identity": identity or {}})
    return {key: ident.get(key) for key in cr_stats.IDENTITY_KEYS}


def check_identity_complete(name, ident):
    """Registered, complete identity; anchors may lack a run UUID (cr_stats derives one)."""
    problems = []
    if not ident.get("run_uuid") and ident.get("role") != "anchor":
        problems.append("run_uuid")
    sha = ident.get("checkpoint_sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        problems.append("checkpoint_sha256")
    if ident.get("arm") not in cr_stats.ARMS:
        problems.append("arm")
    if ident.get("role") not in ("new", "anchor"):
        problems.append("role")
    if ident.get("epoch") != cr_stats.ENDPOINT_EPOCH:
        problems.append("epoch")
    if ident.get("role") == "new" and ident.get("train_seed") not in cr_stats.REGISTERED_SEEDS:
        problems.append("train_seed")
    if ident.get("role") == "anchor" and ident.get("arm") not in SEED_ARMS:
        problems.append("anchor arm")
    if problems:
        raise InputError("%s: incomplete or unregistered identity (%s)" % (name, ", ".join(problems)))


def run_dirs_from(runs=None, inventory=None):
    dirs = []
    if inventory:
        inv, _ = read_json(inventory)
        dirs += sorted(set(row["run_dir"] for row in inv["rows"]))
    dirs += cr_stats._expand(runs or [])
    if not dirs:
        raise InputError("no probe directories (--runs or --inventory)")
    return dirs


def verify_final(final, allow_synthetic):
    analysis = final.get("analysis") or {}
    if final.get("primary_variant") != PRIMARY or PRIMARY not in analysis:
        raise InputError("final JSON has no primary-variant analysis (%s)" % PRIMARY)
    if final.get("before_freeze") is not False and not allow_synthetic:
        raise InputError("final JSON was computed before the results freeze (before_freeze=%r)"
                         % final.get("before_freeze"))
    if final.get("n_boot") != cr_stats.N_BOOT and not allow_synthetic:
        raise InputError("final JSON uses %r bootstrap resamples; %d are registered"
                         % (final.get("n_boot"), cr_stats.N_BOOT))
    if not final.get("runs"):
        raise InputError("final JSON lists no runs")
    for variant, block in analysis.items():
        if variant not in VARIANTS:
            raise InputError("unknown pooling variant %s" % variant)
        outcomes = block.get("outcomes") or {}
        if any(arm not in outcomes for arm in GUIDED):
            raise InputError("%s: outcome labels missing" % variant)
        for arm in GUIDED:
            if outcomes[arm].get("label") not in LABEL_CODE:
                raise InputError("%s/%s: unknown outcome label %r" % (variant, arm,
                                                                      outcomes[arm].get("label")))
        if outcomes["centroid"]["random_range"] != outcomes["envelope"]["random_range"]:
            raise InputError("%s: RANDOM range differs between arms" % variant)


def verify_runs(final, run_dirs, allow_synthetic):
    """Every final run has a probe dir with a complete identity, a seal and its unseal receipt."""
    by_name = {}
    for run_dir in run_dirs:
        real = os.path.realpath(run_dir)
        name = os.path.basename(real)
        if name in by_name and by_name[name] != real:
            raise InputError("two probe directories named %s" % name)
        by_name[name] = real
    final_runs = final["runs"]
    missing = sorted(set(final_runs) - set(by_name))
    extra = sorted(set(by_name) - set(final_runs))
    if missing:
        raise InputError("final JSON runs without a probe directory: %s" % missing)
    if extra:
        raise InputError("probe directories that are not in the final analysis: %s" % extra)
    records, head_sets = {}, {}
    for name, run_dir in sorted(by_name.items()):
        try:
            res = cr_stats.load_run(run_dir)
            ident = cr_stats._identity(res)
            check_identity_complete(name, ident)
            manifest, manifest_sha = cr_stats.read_seal_manifest(run_dir)
            cr_stats.check_receipt(run_dir, manifest_sha)
            cr_stats.check_seal_identity(run_dir, ident, manifest)
        except (cr_stats.IntegrityError, OSError, ValueError, KeyError) as exc:
            raise InputError(str(exc))
        receipt_path = os.path.join(run_dir, "unsealed_results.json")
        receipt, receipt_sha = read_json(receipt_path)
        if _ident(receipt.get("identity")) != _ident(ident):
            raise InputError("%s: unseal receipt identity differs from results.json" % name)
        if receipt.get("unsealed_before_freeze") is not False and not allow_synthetic:
            raise InputError("%s: test predictions were unsealed before the freeze" % name)
        entry = final_runs[name]
        if _ident(entry.get("identity")) != _ident(ident):
            raise InputError("%s: final JSON identity differs from the probe" % name)
        if "eligible" not in (entry.get("eligibility") or {}):
            raise InputError("%s: final JSON has no completion eligibility" % name)
        variants = receipt.get("variants") or {}
        for variant, block in final["analysis"].items():
            value = block["per_run_test_auc"].get(name)
            if value is None:
                continue
            mean = (variants.get(variant) or {}).get("test_auc_mean")
            if mean is None or abs(mean - value) > 1e-9:
                raise InputError("%s/%s: final test AUC is not the unsealed head-seed mean"
                                 % (name, variant))
        if name not in final["analysis"][PRIMARY]["per_run_test_auc"] or PRIMARY not in variants:
            raise InputError("%s: primary pooling missing" % name)
        head_sets[name] = tuple(sorted(int(r["head_seed"]) for r in variants[PRIMARY]["per_seed"]))
        records[name] = {"run_dir": run_dir, "identity": ident, "receipt": receipt,
                         "receipt_path": receipt_path, "receipt_sha256": receipt_sha,
                         "results_sha256": sha256_file(os.path.join(run_dir, "results.json")),
                         "sealed_manifest_sha256": manifest_sha}
    if len(set(head_sets.values())) != 1:
        raise InputError("probe-head seeds differ across runs")
    head_seeds = next(iter(head_sets.values()))
    if head_seeds != tuple(cr_stats.REGISTERED_HEAD_SEEDS) and not allow_synthetic:
        raise InputError("probe-head seeds %s are not the registered %s"
                         % (list(head_seeds), list(cr_stats.REGISTERED_HEAD_SEEDS)))
    return records, head_seeds


def verify_md(md, final, allow_synthetic):
    if md.get("split") != "Test":
        raise InputError("MD JSON is for split %r; the paper reports the test split only"
                         % md.get("split"))
    if md.get("before_freeze") is not False and not allow_synthetic:
        raise InputError("MD JSON was computed before the results freeze")
    runs = md.get("runs") or {}
    if not runs:
        raise InputError("MD JSON has no runs")
    baselines, sizes = set(), set()
    for name, entry in runs.items():
        if name not in final["runs"]:
            raise InputError("MD run %s is not in the final analysis" % name)
        if _ident(entry.get("identity")) != _ident(final["runs"][name]["identity"]):
            raise InputError("MD run %s: identity differs from the final analysis" % name)
        test = ((entry.get("subsets") or {}).get("glaucoma") or {}).get("test")
        if not test:
            raise InputError("MD run %s has no glaucoma-subset test metrics" % name)
        baselines.add((test["baseline_train_mean"]["mae"], test["baseline_train_mean"]["rmse"]))
        sizes.add(test["ridge"]["n"])
    if len(baselines) != 1 or len(sizes) != 1:
        raise InputError("MD runs were not evaluated on identical cases")
    for c in md.get("contrasts") or []:
        if c["a"] not in runs or c["b"] not in runs:
            raise InputError("MD contrast %s refers to a run without metrics" % c.get("kind"))


def budget_kind(data):
    if data.get("schema") == "cr_cb_budget_audit_v1":
        return "cb"
    summary = data.get("summary")
    if isinstance(summary, dict) and "random" in summary and "centroid" in summary:
        return "i5"
    results = data.get("results")
    if isinstance(results, dict) and any(isinstance(v, dict) and "purity" in v
                                         for v in results.values()):
        return "i7"
    raise InputError("unrecognised budget evidence (expected cr_cb_budget_audit_v1, the I5 "
                     "delivered-mask audit or the I7 matcher bench)")


# ---------------------------------------------------------------------------
# Evidence staging and typed numeric macros
# ---------------------------------------------------------------------------

def receipt_source(name):
    return "rep_receipt_" + re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class Stage:
    """Byte-identical copies of every input; numeric sources point at the copies."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.in_repo = inside(self.root, REPO)
        self.sources, self.inputs = {}, {}

    def add(self, name, src_path, rel):
        src = Path(src_path)
        raw = src.read_bytes()
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.read_bytes() != raw:
            dst.write_bytes(raw)
        sha = sha256_bytes(raw)
        if sha256_file(dst) != sha:
            raise InputError("staged copy differs from its source: %s" % dst)
        if self.in_repo:
            spec = {"root": "repo", "path": dst.relative_to(REPO.resolve()).as_posix(), "sha256": sha}
        else:
            spec = {"root": "stage", "path": dst.relative_to(self.root).as_posix(), "sha256": sha}
        self.sources[name] = spec
        self.inputs[name] = {"original": str(src.resolve()), "staged": str(dst), "sha256": sha,
                             "bytes": len(raw)}
        return spec

    def evidence(self, stats_dir):
        evidence = numeric.Evidence(PAPER, stats_dir or self.root, self.sources)
        evidence.roots["stage"] = self.root
        return evidence


class Builder:
    """Ordered typed macro specs, evaluated with numeric_bindings (as p15 does)."""

    def __init__(self):
        self.specs, self.order = {}, []

    def add(self, name, expression, group=False):
        if not re.fullmatch(r"Rep[A-Za-z]+", name):
            raise ValueError("macro names are Rep + letters: " + name)
        if name in self.specs:
            raise ValueError("duplicate macro " + name)
        self.specs[name] = (expression, group)
        self.order.append(name)

    def has(self, name):
        return name in self.specs

    def evaluate(self, evidence):
        values, records = {}, []
        for name in self.order:
            expression, group = self.specs[name]
            bound = evidence.binding(expression)
            display = thousands(bound["expected"]) if group else bound["expected"]
            if expression.get("op") == "format":
                full = [evidence.evaluate(arg) for arg in expression["args"]]
            else:
                full = [evidence.evaluate(expression)]
            values[name] = display
            records.append({"name": name, "display": display, "full_precision": full,
                            "expression": expression, "source_hashes": bound["source_hashes"]})
        return values, records


def run_table(final):
    """name -> run facts; (arm, tag) -> name. Tags: Orig, SeedA, SeedB, SeedC."""
    runs, slots = {}, {}
    for name, entry in final["runs"].items():
        ident = _ident(entry["identity"])
        role = ident["role"]
        seed = int(ident["train_seed"]) if role == "new" else None
        tag = "Orig" if role == "anchor" else "Seed" + SEED_LETTERS[seed]
        runs[name] = {"name": name, "arm": ident["arm"], "role": role, "seed": seed, "tag": tag,
                      "eligible": bool(entry["eligibility"]["eligible"]),
                      "reasons": list(entry["eligibility"].get("reasons") or [])}
        if (ident["arm"], tag) in slots:
            raise InputError("two runs for %s %s" % (ident["arm"], tag))
        slots[(ident["arm"], tag)] = name
    return runs, slots


def contrast_index(block):
    """(kind, seed) -> (list name, index, contrast, descriptive)."""
    out = {}
    for listname in ("primary_contrasts", "matched_control", "descriptive_incomplete_contrasts"):
        for i, c in enumerate(block.get(listname) or []):
            key = (c["kind"], int(c["train_seed"]))
            if key in out:
                raise InputError("duplicate contrast %s" % (key,))
            out[key] = (listname, i, c, listname == "descriptive_incomplete_contrasts")
    return out


def seed_tag(seed):
    return "Seed" + SEED_LETTERS[seed]


def numeric_specs(ctx):
    b = Builder()
    F = "rep_final"
    final, runs, slots, block = ctx.final, ctx.runs, ctx.slots, ctx.block
    A = ("analysis", PRIMARY)
    b.add("RepEndpoint", fmt("%d", ref(F, "runs", sorted(runs)[0], "identity", "epoch")))
    b.add("RepNBoot", fmt("%d", ref(F, "n_boot")), group=True)
    head = []
    for name, r in sorted(runs.items()):
        w = ARM_WORD[r["arm"]] + r["tag"]
        b.add("RepAUC" + w, fmt("%.4f", ref(F, *A, "per_run_test_auc", name)))
        if block["per_run_val_auc"].get(name) is not None:
            b.add("RepValAUC" + w, fmt("%.4f", ref(F, *A, "per_run_val_auc", name)))
        if ctx.records[name]["receipt"]["variants"][PRIMARY].get("test_auc_sd") is not None:
            sd = ref(receipt_source(name), "variants", PRIMARY, "test_auc_sd")
            b.add("RepHeadSD" + w, fmt("%.4f", sd))
            head.append(sd)
    if head:
        b.add("RepHeadSDMax", fmt("%.4f", op("max", *head)))
    for (kind, seed), (listname, i, c, _) in sorted(ctx.cidx.items()):
        if kind in ("centroid-random", "envelope-random"):
            base = "Delta" + ARM_WORD[kind.split("-")[0]] + seed_tag(seed)
        elif kind == "centroid-random_cb":
            base = "PlacementDelta"
        elif kind == "random_cb-random":
            base = "BudgetShift"
        else:
            continue
        p = A + (listname, i)
        b.add("Rep" + base, fmt("%+.4f", ref(F, *p, "delta")))
        b.add("Rep" + base + "CI", fmt(CI4, ref(F, *p, "ci95", 0), ref(F, *p, "ci95", 1)))
        if c.get("val_delta") is not None:
            b.add("RepVal" + base, fmt("%+.4f", ref(F, *p, "val_delta")))
    matched = ctx.matched
    if matched.get("P_ref") and matched.get("B_ref") and matched["G"] and matched["G"] > 0:
        b.add("RepBudgetShareOfGap", fmt("%.0f", op("percent", op(
            "divide", matched["B_ref"], op("add", matched["P_ref"], matched["B_ref"])))))
    for arm in GUIDED:
        o, w, p = block["outcomes"][arm], ARM_WORD[arm], A + ("outcomes", arm)
        if o["mean_delta"] is not None:
            b.add("RepDeltaMean" + w, fmt("%+.4f", ref(F, *p, "mean_delta")))
        sens = o["sensitivity_with_original"]
        if sens.get("original_delta") is not None:
            b.add("RepDeltaOrig" + w, fmt("%+.4f", ref(F, *p, "sensitivity_with_original",
                                                     "original_delta")))
            if sens.get("n") == len(o["new_seed_deltas"]) + 1 and o["new_seed_deltas"]:
                b.add("RepSensDeltaMean" + w, fmt("%+.4f", ref(F, *p, "sensitivity_with_original",
                                                             "pooled_mean_delta")))
        if (arm, "Orig") in slots and ("random", "Orig") in slots and \
                block["per_run_val_auc"].get(slots[(arm, "Orig")]) is not None and \
                block["per_run_val_auc"].get(slots[("random", "Orig")]) is not None:
            b.add("RepValDeltaOrig" + w, fmt("%+.4f", op(
                "subtract", ref(F, *A, "per_run_val_auc", slots[(arm, "Orig")]),
                ref(F, *A, "per_run_val_auc", slots[("random", "Orig")]))))
    o, p = block["outcomes"]["centroid"], A + ("outcomes", "centroid")
    if o["random_range"] is not None:
        b.add("RepRandomRange", fmt("%.4f", ref(F, *p, "random_range")))
        b.add("RepRandomMin", fmt("%.4f", op("min", ref(F, *p, "random_realizations"))))
        b.add("RepRandomMax", fmt("%.4f", op("max", ref(F, *p, "random_realizations"))))
    if o.get("random_range_all_completed_runs") is not None:
        b.add("RepRandomRangeAll", fmt("%.4f", ref(F, *p, "random_range_all_completed_runs")))
    for arm in SEED_ARMS:
        refs = [ref(F, *A, "per_run_test_auc", slots[(arm, seed_tag(s))]) for s in ctx.complete]
        if refs:
            b.add("RepMeanAUC" + ARM_WORD[arm], fmt("%.4f", op("mean", *refs)))
        if len(refs) >= 2:
            b.add("RepSDAUC" + ARM_WORD[arm], fmt("%.4f", op("stdev", *refs)))
    if ctx.p1c is not None:
        diffs = []
        for arm, key in (("random", "random"), ("centroid", "oracle"), ("envelope", "envelope")):
            rows = [i for i, row in enumerate(ctx.p1c.get("table", []))
                    if row.get("key") == "%s@ep%d@fp16" % (key, cr_stats.ENDPOINT_EPOCH)]
            if (arm, "Orig") in slots and len(rows) == 1:
                diffs.append(op("abs", op("subtract", ref(F, *A, "per_run_test_auc",
                                                          slots[(arm, "Orig")]),
                                          ref("p1c_stats.json", "table", rows[0], "auc"))))
        if diffs:
            b.add("RepReprobeMaxDiff", fmt("%.4f", op("max", *diffs)))
    # Pooling variants (secondary): means over each variant's complete indices.
    for variant in VARIANTS[1:]:
        vb = final["analysis"].get(variant)
        if vb is None:
            continue
        vw = VARIANT_WORD[variant]
        for arm in SEED_ARMS:
            refs = [ref(F, "analysis", variant, "per_run_test_auc", slots[(arm, seed_tag(s))])
                    for s in vb["complete_seed_indices"]]
            if refs:
                b.add("RepPool%sAUC%s" % (vw, ARM_WORD[arm]), fmt("%.4f", op("mean", *refs)))
        for arm in GUIDED:
            if vb["outcomes"][arm]["mean_delta"] is not None:
                b.add("RepPool%sDelta%s" % (vw, ARM_WORD[arm]),
                      fmt("%+.4f", ref(F, "analysis", variant, "outcomes", arm, "mean_delta")))
    for arm in GUIDED:
        refs = [ref(F, "analysis", v, "outcomes", arm, "mean_delta") for v in VARIANTS[1:]
                if v in final["analysis"]
                and final["analysis"][v]["outcomes"][arm]["mean_delta"] is not None]
        if refs:
            b.add("RepPoolDeltaMin" + ARM_WORD[arm], fmt("%+.4f", op("min", *refs)))
            b.add("RepPoolDeltaMax" + ARM_WORD[arm], fmt("%+.4f", op("max", *refs)))
    budget_specs(b, ctx)
    md_specs(b, ctx)
    return b


BUDGET_KEYS = (("unique_targets_U", "U"), ("loss_slots_L", "L"), ("duplicate_slots_D", "D"),
               ("context_tokens", "C"))


def budget_specs(b, ctx):
    """Budget-audit macros. Precedence: the RANDOM-CB run's own audit (cr_cb_budget_audit_v1),
    then the I5 delivered-mask audit (RANDOM, CENTROID), then the I7 bench (purity, per-image
    match counts). ctx.budget_provenance records where each table block came from."""
    data, prov = ctx.budget, {}
    ctx.budget_provenance = prov

    def add(name, expression, origin, key, group=False):
        if not b.has(name):
            b.add(name, expression, group)
            prov.setdefault(key, origin)

    if "cb" in data:
        src, audit = "rep_budget_cb", data["cb"]
        for arm in ("random", "random_cb", "centroid"):
            row = (audit.get("rows") or {}).get(arm) or {}
            for key, letter in BUDGET_KEYS:
                if row.get(key) is not None:
                    add("RepBudget%s%s" % (letter, ARM_WORD[arm]),
                        fmt("%.1f", ref(src, "rows", arm, key)), "cb", "budget_" + arm)
            if row.get("target_purity_pct") is not None:
                add("RepPurity" + ARM_WORD[arm], fmt("%.1f", ref(src, "rows", arm,
                                                                 "target_purity_pct")),
                    "cb", "purity")
        if audit.get("images") is not None:
            add("RepCBMatchImages", fmt("%d", ref(src, "images")), "cb", "match", group=True)
        if audit.get("mismatches"):
            add("RepCBMismatches", fmt("%d", op("add", *[ref(src, "mismatches", k)
                                                        for k in sorted(audit["mismatches"])])),
                "cb", "match")
        if audit.get("batches") is not None:
            add("RepCBBatches", fmt("%d", ref(src, "batches")), "cb", "batches", group=True)
        if audit.get("mean_abs_hidden_diff") is not None:
            add("RepCBHiddenDiff", fmt("%.2f", ref(src, "mean_abs_hidden_diff")), "cb", "hidden")
    if "i5" in data:
        src, summary = "rep_budget_i5", data["i5"]["summary"]
        for arm in ("random", "centroid"):
            for key, letter in BUDGET_KEYS:
                if (summary[arm].get(key) or {}).get("mean") is not None:
                    add("RepBudget%s%s" % (letter, ARM_WORD[arm]),
                        fmt("%.1f", ref(src, "summary", arm, key, "mean")), "i5", "budget_" + arm)
    if "i7" in data:
        src, cell = "rep_budget_i7", ctx.i7_cell
        result = data["i7"]["results"].get(cell)
        if result is None:
            raise InputError("I7 bench has no cell %s" % cell)
        for arm, key in (("centroid", "centroid"), ("random_cb", "cb"), ("random", "random")):
            if (result["purity"].get(key) or {}).get("target_purity_pct") is not None:
                add("RepPurity" + ARM_WORD[arm], fmt("%.1f", ref(src, "results", cell, "purity",
                                                                 key, "target_purity_pct")),
                    "i7", "purity")
        if result["purity"].get("n_valid_views_per_draw") is not None and prov.get("purity") == "i7":
            add("RepPurityViews", fmt("%d", ref(src, "results", cell, "purity",
                                                "n_valid_views_per_draw")), "i7", "purity_views")
        if not b.has("RepCBMatchImages") and not b.has("RepCBMismatches"):
            mism = result["per_image_mismatches_vs_shadow"]
            add("RepCBMatchImages", fmt("%d", ref(src, "results", cell, "images")), "i7", "match",
                group=True)
            add("RepCBMismatches", fmt("%d", op("add", *[ref(src, "results", cell,
                                                             "per_image_mismatches_vs_shadow", k)
                                                         for k in sorted(mism)])), "i7", "match")


CB_LINE = re.compile(r"\[CB\] matching=(?P<matching>\S+) r_t=(?P<r_t>[\d.]+) "
                     r"bypass_ramp0=(?P<bypass>\d+) K=(?P<K>\d+) context=(?P<context>\d+) "
                     r"loss_slots=(?P<L>\d+) unique_targets=(?P<U>[\d.]+) "
                     r"duplicates=(?P<D>[\d.]+) hidden_shadow=(?P<Hs>[\d.]+) "
                     r"hidden_matched=(?P<Hm>[\d.]+)")


def cb_audit_from_log(log_paths, min_ramp=1.0):
    """cr_cb_budget_audit_v1 from the RANDOM-CB trainer's [CB] lines (src/train_patch.py
    format_cb_stats): means over logged batches at full guidance, bypass batches excluded.
    Budgets equal the CENTROID shadow by construction and are re-verified in the worker; the
    log adds the hidden-area residual (strict matching: zero)."""
    rows, sources = [], []
    for path in log_paths:
        raw = Path(path).read_bytes()
        sources.append({"path": str(Path(path).resolve()), "sha256": sha256_bytes(raw)})
        for m in CB_LINE.finditer(raw.decode("utf-8", errors="replace")):
            if float(m["r_t"]) >= min_ramp - 1e-9 and int(m["bypass"]) == 0:
                rows.append(m)
    if not rows:
        raise InputError("no full-ramp [CB] lines in %s" % [s["path"] for s in sources])
    matching = sorted(set(m["matching"] for m in rows))

    def mean(key):
        return sum(float(m[key]) for m in rows) / len(rows)

    return {"schema": "cr_cb_budget_audit_v1", "source": "RANDOM-CB trainer [CB] log lines",
            "logs": sources, "matching": matching, "min_ramp": min_ramp, "batches": len(rows),
            "mean_abs_hidden_diff": sum(abs(float(m["Hm"]) - float(m["Hs"])) for m in rows)
            / len(rows),
            "rows": {"random_cb": {"unique_targets_U": mean("U"), "loss_slots_L": mean("L"),
                                   "duplicate_slots_D": mean("D"),
                                   "context_tokens": mean("context")}}}


def md_specs(b, ctx):
    md = ctx.md
    if md is None:
        return
    M = "rep_md"
    names = ctx.md_names
    maes = []
    for name in names:
        r = ctx.runs[name]
        w = ARM_WORD[r["arm"]] + r["tag"]
        p = ("runs", name, "subsets", "glaucoma", "test", "ridge")
        b.add("RepMDMAE" + w, fmt("%.2f", ref(M, *p, "mae")))
        b.add("RepMDRMSE" + w, fmt("%.2f", ref(M, *p, "rmse")))
        if md["runs"][name]["subsets"]["glaucoma"]["test"]["ridge"]["spearman"] is not None:
            b.add("RepMDSpearman" + w, fmt("%.2f", ref(M, *p, "spearman")))
        maes.append(ref(M, *p, "mae"))
    first = ("runs", names[0], "subsets", "glaucoma", "test")
    b.add("RepMDBaseMAE", fmt("%.2f", ref(M, *first, "baseline_train_mean", "mae")))
    b.add("RepMDBaseRMSE", fmt("%.2f", ref(M, *first, "baseline_train_mean", "rmse")))
    b.add("RepMDN", fmt("%d", ref(M, *first, "ridge", "n")), group=True)
    b.add("RepMDMAEMin", fmt("%.2f", op("min", *maes)))
    b.add("RepMDMAEMax", fmt("%.2f", op("max", *maes)))
    for i, c in enumerate(md.get("contrasts") or []):
        if c["subset"] != "glaucoma":
            continue
        arm = c["kind"].split("-")[0]
        w = ARM_WORD[arm] + seed_tag(int(c["train_seed"]))
        p = ("contrasts", i, "metrics")
        b.add("RepMDDeltaMAE" + w, fmt("%+.2f", ref(M, *p, "mae", "delta")))
        b.add("RepMDDeltaMAE" + w + "CI", fmt(CI2, ref(M, *p, "mae", "ci95", 0),
                                              ref(M, *p, "mae", "ci95", 1)))
        if c["metrics"]["spearman"]["delta"] is not None:
            b.add("RepMDDeltaSpearman" + w, fmt("%+.2f", ref(M, *p, "spearman", "delta")))


# ---------------------------------------------------------------------------
# Scenario selection (pre-registration labels; matched-control rule in the docstring)
# ---------------------------------------------------------------------------

def arm_labels(block):
    return {arm: block["outcomes"][arm]["label"] for arm in GUIDED}


def scenario_of(labels):
    codes = {arm: LABEL_CODE[label] for arm, label in labels.items()}
    return codes["centroid"] if codes["centroid"] == codes["envelope"] else "mixed", codes


CB_SEED = 1234
CONTROL_LABELS = ("M+", "M0", "M-", "M_partial")
CONTROL_NAMES = {"M0": "placement difference not detected",
                 "M-": "placement difference reversed",
                 "M+": "placement accounts for most of the gap",
                 "M_partial": "both contribute"}


def matched_outcome(block, runs, cidx=None):
    """Matched-control description (pre-registration section 4 and amendment 2).

    P = AUC(C1) - AUC(CB) and B = AUC(CB) - AUC(R1) at the registered RANDOM-CB seed 1234,
    r = the RANDOM range of section 3 (``random_range``). Labels only from on-time runs:
    M0 |P| <= r; M- P < -r; M+ P > r and P >= (P + B) / 2; M_partial (both contribute)
    P > r and P < (P + B) / 2. Otherwise a status without a label: M_na (no RANDOM-CB result),
    M_noP (no CENTROID result at that seed), M_desc (placement contrast uses a late run),
    M_unassessed (P > r but no on-time B), M_incomplete (no RANDOM range).
    """
    cidx = cidx if cidx is not None else contrast_index(block)
    spread = block["outcomes"]["centroid"]["random_range"]
    keys = [k for k in cidx if k[0] in ("centroid-random_cb", "random_cb-random")]
    cb_runs = [r for r in runs.values() if r.get("arm") == "random_cb"]
    seeds = set(k[1] for k in keys) | set(r.get("seed") for r in cb_runs)
    if seeds - {CB_SEED}:
        raise InputError("RANDOM-CB is registered at seed %d only; found %s"
                         % (CB_SEED, sorted(seeds)))
    info = {"seed": CB_SEED if seeds else None, "spread": spread, "P": None, "B": None,
            "G": None, "P_ref": None, "B_ref": None, "P_descriptive": None,
            "B_descriptive": None, "cb_present": bool(cb_runs),
            "cb_eligible": bool(cb_runs) and all(r.get("eligible", True) for r in cb_runs),
            "label": None, "name": None, "reason": None}

    def pointer(key):
        listname, i, _, _ = cidx[key]
        return ref("rep_final", "analysis", PRIMARY, listname, i, "delta")

    pk, bk = ("centroid-random_cb", CB_SEED), ("random_cb-random", CB_SEED)
    if bk in cidx:
        info["B"], info["B_ref"], info["B_descriptive"] = cidx[bk][2]["delta"], pointer(bk), \
            cidx[bk][3]
    if pk in cidx:
        info["P"], info["P_ref"], info["P_descriptive"] = cidx[pk][2]["delta"], pointer(pk), \
            cidx[pk][3]
    if info["P"] is not None and info["B"] is not None:
        info["G"] = info["P"] + info["B"]
    if not info["cb_present"] and info["P"] is None and info["B"] is None:
        info["label"], info["reason"] = "M_na", "no RANDOM-CB result"
        return info
    if info["P"] is None:
        info["label"], info["reason"] = "M_noP", "no CENTROID result at the RANDOM-CB seed"
        return info
    if info["P_descriptive"]:
        info["label"], info["reason"] = "M_desc", "a run of the placement contrast finished late"
        return info
    if spread is None:
        info["label"], info["reason"] = "M_incomplete", "no RANDOM range"
        return info
    exceeds = abs(info["P"]) > spread
    flag = cidx[pk][2].get("abs_delta_exceeds_random_range")
    if flag is not None and bool(flag) != exceeds:
        raise InputError("matched-control flag disagrees with the RANDOM range")
    if not exceeds:
        info["label"] = "M0"
    elif info["P"] < 0:
        info["label"] = "M-"
    elif info["B"] is None or info["B_descriptive"]:
        info["label"] = "M_unassessed"
        info["reason"] = "P > r but no on-time budget shift B; half-gap condition not assessed"
    elif info["P"] >= 0.5 * info["G"]:
        info["label"] = "M+"
    else:
        info["label"] = "M_partial"
    info["name"] = CONTROL_NAMES.get(info["label"])
    return info


# ---------------------------------------------------------------------------
# Scenario text (pre-written in A6 sections 0.5 and 4; labels from the pre-registration)
# ---------------------------------------------------------------------------

PLACEMENT = {"centroid": "intensity-guided placement", "envelope": "segmentation-guided placement"}
PLACEMENT_RECT = {"centroid": "intensity-guided rectangle placement",
                  "envelope": "segmentation-guided rectangle placement"}
GUIDE = {"centroid": "segmentation-free intensity guidance", "envelope": "segmentation guidance"}
CB_INTRO = (r" When uniform placement receives \ArmBest{}'s per-image target and context "
            r"budgets, ")
ABSTRACT_CB = (" When uniform placement receives the intensity-guided strategy's target and "
               "context budgets, ")


class Text:
    """Text macros; every number is a reference to a defined numeric macro."""

    def __init__(self, ctx):
        self.ctx, self.values = ctx, ctx.values
        self.macros, self.notes = {}, {}

    def num(self, name):
        if name not in self.values:
            raise KeyError("text refers to an undefined numeric macro: " + name)
        return "$\\%s$" % name

    def bare(self, name):
        if name not in self.values and name not in self.macros:
            raise KeyError("text refers to an undefined macro: " + name)
        return "\\%s{}" % name

    def put(self, name, text, note=None):
        text = re.sub(r"[ \t]+", " ", text).strip()
        if re.search(r"\d", text):
            raise ValueError("digits in text macro %s (numbers must come from macros)" % name)
        if not text.isascii():
            raise ValueError("non-ASCII text in " + name)
        self.macros[name] = text
        if note:
            self.notes[name] = note


def ordered_arms(codes):
    return sorted(GUIDED, key=lambda arm: (LABEL_RANK[codes[arm]], GUIDED.index(arm)))


def build_text(ctx):
    t = Text(ctx)
    block, labels, codes, scen = ctx.block, ctx.labels, ctx.codes, ctx.scenario
    m = ctx.matched["label"]
    n_seeds = len(ctx.complete)
    random_vals = block["outcomes"]["centroid"]["random_realizations"]
    t.put("RepNSeedsWord", word(n_seeds), "number of complete seed indices")
    if n_seeds >= 2:
        t.put("RepAllSeedsWords", "both" if n_seeds == 2 else "all " + word(n_seeds),
              "both / all n complete seed indices")
    t.put("RepNRandomWord", word(len(random_vals)), "number of RANDOM realizations in the range")
    t.put("RepNHeadSeedsWord", word(len(ctx.head_seeds)), "number of probe-head seeds")
    if n_seeds in NULL_CHANCE:
        t.put("RepNullChanceWords", NULL_CHANCE[n_seeds],
              "probability that all n deltas are positive under a symmetric null (0.5^n)")
    for arm in GUIDED:
        t.put("RepLabel" + ARM_WORD[arm], LABEL_TEX[labels[arm]], "final label, primary pooling")
    name = ARM_TEX
    have_range = "RepRandomRange" in t.values

    def mean(arm):
        return t.num("RepDeltaMean" + ARM_WORD[arm])

    def per_seed(arm):
        return join_and(t.num("RepDelta%s%s" % (ARM_WORD[arm], seed_tag(s))) for s in ctx.complete)

    def facts(arm):
        o = block["outcomes"][arm]
        deltas = o["new_seed_deltas"]
        return {"all_positive": bool(deltas) and all(d > 0 for d in deltas),
                "mean_exceeds": (o["mean_delta"] is not None and o["random_range"] is not None
                                 and o["mean_delta"] > o["random_range"]),
                "below_range": (o["mean_delta"] is not None and o["random_range"] is not None
                                and o["mean_delta"] < -o["random_range"])}

    def b_phrase(arm, versus, bound):
        """Literal wording for one arm's within-run-to-run-variation label (failing condition)."""
        f = facts(arm)
        mean_positive = (block["outcomes"][arm]["mean_delta"] or 0) > 0
        if not mean_positive or (f["all_positive"] and f["mean_exceeds"]):
            return "falls within run-to-run variation under the registered rule"
        if not f["all_positive"] and not f["mean_exceeds"]:
            return ("is ahead of %s on average, but not in every continuation and by no more "
                    "than %s" % (versus, bound))
        if not f["all_positive"]:
            return "is ahead of %s on average but not in every continuation" % versus
        return "is ahead of %s in every continuation, but on average by no more than %s" % (
            versus, bound)

    abstract_bound = "the variation between uniform-masking runs"
    intro_bound = (r"the range of \textsc{random}'s realizations (%s)" % t.num("RepRandomRange")
                   if "RepRandomRange" in t.values else abstract_bound)
    t.put("RepNRegisteredSeedsWord", word(len(cr_stats.REGISTERED_SEEDS)),
          "registered seed indices (amendment 1 adds the third)")

    # ---- abstract (number-free, A6 section 4.3) ------------------------------------------
    more = " More anatomically specific strategies do not consistently provide further gains."
    coupled = (" Analysis of the delivered masks shows that target selection, target size and "
               "visible context are coupled")
    ctrl = {"M+": ABSTRACT_CB + "it stays below that strategy in a single continuation.",
            "M_partial": ABSTRACT_CB + "part of the difference remains in a single continuation.",
            "M0": ABSTRACT_CB + "the difference falls within run-to-run variation.",
            "M-": ABSTRACT_CB + "it scores higher than that strategy in a single continuation."
            }.get(m, "")
    candidate = (" We therefore present tissue-directed placement as a candidate rather than an "
                 "established improvement, and identify the controls needed to test it.")
    if scen == "A" and m in ("M0", "M-"):
        budget = ("the difference between the two falls within run-to-run variation"
                  if m == "M0" else
                  "uniform placement scores higher than the guided strategy by more than "
                  "run-to-run variation")
        abstract = ("Across repeated pretraining continuations, tissue-directed rectangle "
                    "placement gives higher frozen-probe AUC than uniform masking, including with "
                    "segmentation-free intensity guidance. These strategies also leave more of the "
                    "image visible to the encoder." + ABSTRACT_CB + budget + "." + more)
    elif scen == "A":
        abstract = ("Across repeated pretraining continuations, tissue-directed rectangle "
                    "placement gives higher frozen-probe AUC than uniform masking, including with "
                    "segmentation-free intensity guidance")
        if m == "M+":
            abstract += (", and the advantage remains when uniform placement receives the "
                         "intensity-guided strategy's target and context budgets")
        abstract += "."
        if m == "M_partial":
            abstract += ABSTRACT_CB + "part of the advantage remains."
        abstract += (more + coupled + " unless they are controlled explicitly. These results "
                     "support tissue-directed target selection as a simple, segmentation-free "
                     "option for retinal OCT pretraining.")
    elif scen == "B":
        abstract = ("In the original single runs, tissue-directed rectangle placement gave higher "
                    "frozen-probe AUC than uniform masking, including with segmentation-free "
                    "intensity guidance. Across repeated pretraining continuations, however, "
                    "intensity-guided placement %s, and segmentation-guided placement %s, so "
                    "neither meets the registered criterion for a consistent direction."
                    % (b_phrase("centroid", "uniform masking", abstract_bound),
                       b_phrase("envelope", "uniform masking", abstract_bound)) + more +
                    coupled + "." + ctrl + candidate)
    elif scen == "C":
        below = [facts(arm)["below_range"] for arm in GUIDED]
        how = ("below" if all(below) else "similarly to or below" if any(below)
               else "similarly to")
        abstract = ("Repeated pretraining continuations do not reproduce the advantage of "
                    "tissue-directed placement seen in the original single runs. Averaged across "
                    "continuations, guided rectangles perform %s uniform masking. Analysis of the "
                    "delivered masks shows that the strategies also differ in target size and "
                    "visible context.%s At this effect size, single-run comparisons of masking "
                    "strategies can be misleading, and target placement, target budget and visible "
                    "context need to be controlled jointly." % (how, ctrl))
    elif scen == "I":
        abstract = ("In the original single runs, tissue-directed rectangle placement gave higher "
                    "frozen-probe AUC than uniform masking, including with segmentation-free "
                    "intensity guidance. The registered repeated continuations did not all finish "
                    "before the results freeze, so they are reported descriptively and do not "
                    "establish the direction of the effect." + more + coupled + "." + ctrl +
                    " Our findings motivate anatomy-guided predictive learning for retinal OCT and "
                    "better-controlled comparisons of what to predict and what to keep visible.")
    else:
        def clause(arm, subject):
            code = codes[arm]
            if code == "A":
                return ("%s gives higher frozen-probe AUC than uniform masking in every "
                        "continuation" % subject)
            if code == "B":
                return "%s %s" % (subject, b_phrase(arm, "uniform masking", abstract_bound))
            if code == "C":
                return "%s does not improve on uniform masking on average" % subject
            return "%s could not be assessed because its registered runs did not all finish" % \
                subject

        arms = ordered_arms(codes)
        first = clause(arms[0], PLACEMENT_RECT[arms[0]])
        second = clause(arms[1], PLACEMENT[arms[1]].capitalize())
        closing = (" Tissue-directed target selection therefore remains promising for retinal OCT "
                   "pretraining, with a consistent direction only for %s." % GUIDE[arms[0]]
                   if codes[arms[0]] == "A" else candidate)
        abstract = ("Across repeated pretraining continuations, " + first + ". " + second + "." +
                    more + coupled + "." + ctrl + closing)
    t.put("RepAbstractResults", abstract, "replaces the abstract's result sentences")

    # ---- introduction --------------------------------------------------------------------
    lead = "Across %s further continuations per strategy from the same checkpoint, " % \
        t.bare("RepNSeedsWord")
    if scen == "A":
        intro = (lead + r"both guided strategies stay above \textsc{random} in every continuation, "
                 r"with mean differences of %s for \ArmBest{} and %s for \textsc{envelope}, larger "
                 r"than the range of \textsc{random}'s own runs (%s). With this few continuations, "
                 r"this is a consistent direction rather than a confirmed replication."
                 % (mean("centroid"), mean("envelope"), t.num("RepRandomRange")))
    elif scen == "B":
        intro = (lead + r"\ArmBest{} (mean difference %s) %s, and \textsc{envelope} (mean "
                 r"difference %s) %s. Neither meets the registered criterion for a consistent "
                 r"direction." % (mean("centroid"), b_phrase("centroid", r"\textsc{random}",
                                                             intro_bound),
                                  mean("envelope"), b_phrase("envelope", r"\textsc{random}",
                                                             intro_bound)))
    elif scen == "C":
        intro = (lead + r"the original advantage does not replicate: the mean differences from "
                 r"\textsc{random} are %s for \ArmBest{} and %s for \textsc{envelope}."
                 % (mean("centroid"), mean("envelope")))
    elif scen == "I":
        intro = ("The registered repeated continuations did not all finish before the results "
                 "freeze, so they are reported descriptively.")
    else:
        clause = {"A": r"%s stays above \textsc{random} in every continuation (mean difference %s)",
                  "B": r"%s (mean difference %s) %s",
                  "C": r"%s does not exceed \textsc{random} on average (mean difference %s)",
                  "I": r"%s has no registered label because its runs did not all finish%s"}
        parts = []
        for arm in ordered_arms(codes):
            value = "" if codes[arm] == "I" else mean(arm)
            if codes[arm] == "B":
                parts.append(clause["B"] % (name[arm], value, b_phrase(
                    arm, r"\textsc{random}", intro_bound)))
            else:
                parts.append(clause[codes[arm]] % (name[arm], value))
        intro = lead + parts[0] + ", whereas " + parts[1] + "."
        if "A" in codes.values():
            intro += (" With this few continuations, a consistent direction is not a confirmed "
                      "replication.")
    intro += {"M+": CB_INTRO + r"it stays below \ArmBest{} by more than the variation between "
                               r"\textsc{random} runs.",
              "M_partial": CB_INTRO + r"part of the difference from \ArmBest{} remains.",
              "M0": CB_INTRO + r"it comes within run-to-run variation of \ArmBest{}.",
              "M-": CB_INTRO + r"it scores above \ArmBest{} by more than the variation between "
                               r"\textsc{random} runs."}.get(m, "")
    t.put("RepIntroResults", intro, "end of the introduction's results paragraph")

    # ---- contributions -------------------------------------------------------------------
    if scen == "A":
        contrib = ("Frozen-probe results, repeated across pretraining continuations, showing gains "
                   "from tissue-directed rectangle placement in a consistent direction, alongside "
                   "mixed results for more specific anatomical targeting.")
    elif scen == "B":
        contrib = ("Frozen-probe results with repeated continuations, in which the gains seen in "
                   "single runs fall within run-to-run variation under the registered rule.")
    elif scen == "C":
        contrib = ("Evidence that single-run gains of guided masking strategies on retinal OCT do "
                   "not replicate across repeated continuations.")
    elif scen == "I":
        contrib = ("Frozen-probe results showing gains from tissue-directed rectangle placement in "
                   "the original runs, with the registered repeated continuations reported "
                   "descriptively.")
    else:
        verb = {"A": "keeps a consistent direction",
                "B": "falls within run-to-run variation under the registered rule",
                "C": "reverses", "I": "is incomplete"}
        arms = ordered_arms(codes)
        contrib = ("Frozen-probe results with repeated continuations, in which %s %s and %s %s."
                   % (PLACEMENT[arms[0]], verb[codes[arms[0]]], PLACEMENT[arms[1]],
                      verb[codes[arms[1]]]))
    t.put("RepContribResults", contrib, "replaces contribution 2")
    effect = {"M+": "placement accounts for most of the same-seed gap",
              "M_partial": "placement and budget both contribute to the same-seed gap",
              "M0": "the remaining placement difference is within run-to-run variation",
              "M-": r"uniform placement scores above \ArmBest{} by more than run-to-run "
                    r"variation"}.get(m)
    t.put("RepContribControlItem",
          (r"\item A matched-budget control in which uniform placement receives \ArmBest{}'s "
           r"per-image target and context budgets. In a single continuation, %s." % effect)
          if effect else "",
          "new contribution bullet (empty without a matched result)")

    # ---- setup: completion of the registered runs -----------------------------------------
    present = {(r["arm"], r["seed"]): r for r in ctx.runs.values() if r["role"] == "new"}
    done = [k for k in REGISTERED_RUNS if k in present and present[k]["eligible"]]
    late = [k for k in REGISTERED_RUNS if k in present and not present[k]["eligible"]]
    absent = [k for k in REGISTERED_RUNS if k not in present]

    def run_list(keys):
        items = []
        for seed in sorted(set(s for _, s in keys), key=lambda x: SEED_LETTERS[x]):
            arms = [a for a, s in keys if s == seed]
            registered = [a for a, s in REGISTERED_RUNS if s == seed]
            if len(arms) > 1 and set(arms) == set(registered):
                items.append("the seed-%s runs" % SEED_LETTERS[seed])
            else:
                items += ["%s seed %s" % (ARM_TEX[a], SEED_LETTERS[seed]) for a in arms]
        text = join_and(items)
        return text[0].upper() + text[1:] if text.startswith("the") else text

    if not late and not absent:
        setup = ("All %s registered continuations finished before the results freeze."
                 % word(len(REGISTERED_RUNS)))
    else:
        setup = ("%s of the %s registered continuations finished before the results freeze."
                 % (word(len(done), True), word(len(REGISTERED_RUNS))))
        if late:
            setup += (" %s finished after the freeze and %s reported descriptively."
                      % (run_list(late), "is" if len(late) == 1 else "are"))
        if absent:
            setup += (r" %s did not produce an epoch-%s result before the freeze and %s reported "
                      r"as not completed." % (run_list(absent), t.bare("RepEndpoint"),
                                              "is" if len(absent) == 1 else "are"))
    setup = ("The third continuation of each strategy was added by a registered amendment before "
             "any new result. " + setup)
    t.put("RepSetupResults", setup, "registered runs: finished / late / not completed")

    # ---- results: repeated continuations ---------------------------------------------------
    sentences = []
    if scen == "I" or all(codes[arm] == "I" for arm in GUIDED):
        reason = ("fewer than two seed indices finished all three strategies before the results "
                  "freeze" if n_seeds < 2 else
                  r"the re-probed original \textsc{random} encoder is not available")
        sentences.append("No registered label is assigned because %s. The completed runs are "
                         r"reported descriptively in Table~\ref{tab:replication}." % reason)
    else:
        both = codes["centroid"] if codes["centroid"] == codes["envelope"] else None
        detail = (r"%s for \ArmBest{} (per seed %s) and %s for \textsc{envelope} (per seed %s)"
                  % (mean("centroid"), per_seed("centroid"), mean("envelope"),
                     per_seed("envelope"))) if both in ("A", "C") else None
        if both == "A":
            sentences.append(
                r"Both guided strategies exceed \textsc{random} in %s new continuations: the "
                r"mean differences are %s, both larger than the range of \textsc{random}'s %s "
                r"realizations (%s). Both registered labels are \emph{consistent direction}."
                % (t.bare("RepAllSeedsWords"), detail, t.bare("RepNRandomWord"),
                   t.num("RepRandomRange")))
        elif both == "C":
            sentences.append(
                r"Neither guided strategy exceeds \textsc{random} on average: the mean differences "
                r"are %s. Both registered labels are \emph{reversed}." % detail)
        for arm in GUIDED if both not in ("A", "C") else ():
            f, code = facts(arm), codes[arm]
            if code == "A":
                sentences.append(
                    r"%s exceeds \textsc{random} in %s new continuations (%s), and its mean "
                    r"difference of %s is larger than the range of \textsc{random}'s %s "
                    r"realizations (%s). Its registered label is \emph{consistent direction}."
                    % (name[arm], t.bare("RepAllSeedsWords"), per_seed(arm), mean(arm),
                       t.bare("RepNRandomWord"), t.num("RepRandomRange")))
            elif code == "B":
                why = []
                if not f["all_positive"]:
                    why.append("not every new continuation is positive")
                if not f["mean_exceeds"]:
                    why.append(r"the mean does not exceed the range of \textsc{random}'s %s "
                               r"realizations (%s)" % (t.bare("RepNRandomWord"),
                                                       t.num("RepRandomRange")))
                sentences.append(
                    r"%s has a positive mean difference from \textsc{random} of %s (per seed %s), "
                    r"but %s. Its registered label is \emph{within run-to-run variation}."
                    % (name[arm], mean(arm), per_seed(arm), " and ".join(why)))
            elif code == "C":
                sentences.append(
                    r"%s does not exceed \textsc{random} on average (mean difference %s, per seed "
                    r"%s). Its registered label is \emph{reversed}."
                    % (name[arm], mean(arm), per_seed(arm)))
            else:
                sentences.append("%s has no registered label." % name[arm])
        if "A" in codes.values() and "RepNullChanceWords" in t.macros:
            sentences.append(
                "Under a symmetric null, %s differences would be positive with probability "
                "%s, so this label does not amount to a confirmed replication."
                % (t.bare("RepAllSeedsWords"), t.bare("RepNullChanceWords")))
    if all("RepSensDeltaMean" + ARM_WORD[arm] in t.values for arm in GUIDED):
        sentences.append(
            r"Including the original runs, which used earlier code and partly different hardware, "
            r"the mean differences are %s for \ArmBest{} and %s for \textsc{envelope} "
            r"(descriptive)." % (t.num("RepSensDeltaMeanCentroid"),
                                 t.num("RepSensDeltaMeanEnvelope")))
    incomplete = block.get("incomplete_seed_indices") or []
    if incomplete:
        letters = join_and(SEED_LETTERS[s] for s in incomplete)
        sentences.append("Seed index%s %s did not finish all three strategies before the freeze, "
                         "so %s runs are descriptive."
                         % ("es" if len(incomplete) > 1 else "", letters,
                            "their" if len(incomplete) > 1 else "its"))
    t.put("RepResultsSeeds", " ".join(sentences), "results paragraph 'Repeated continuations'")

    # ---- results: matched-budget control ---------------------------------------------------
    t.put("RepResultsControl", control_text(t, ctx), "results paragraph 'Matched-budget control'")

    # ---- results: pooling ------------------------------------------------------------------
    t.put("RepResultsPooling", pooling_text(t, ctx), "results paragraph 'Pooling'")

    # ---- discussion, limitations, conclusion ----------------------------------------------
    if scen == "A":
        disc = ("Across repeated continuations, the rectangle-policy gains keep their direction, "
                "which supports tissue-directed target selection without a new encoder "
                "architecture. The segmentation-free centroid strategy is a useful simple baseline.")
    elif scen == "B":
        disc = ("The rectangle-policy gains do not meet the registered criterion for a consistent "
                "direction across continuations, so they motivate tissue-directed target "
                "selection as a candidate rather than an established improvement. The "
                "segmentation-free centroid strategy remains a useful simple baseline.")
    elif scen == "C":
        disc = ("The rectangle-policy gains of the original runs do not hold across repeated "
                "continuations. For masking studies, single-run differences of this size need "
                "repeated runs before they are interpreted.")
    elif scen == "I":
        disc = ("The rectangle-policy results of the original runs motivate anatomy-guided target "
                "selection, but the registered repeated continuations did not all finish, so the "
                "direction of these gains across runs is not established.")
    else:
        arms = ordered_arms(codes)
        state = {"A": "keeps its direction",
                 "B": "falls within run-to-run variation under the registered rule",
                 "C": "reverses", "I": "could not be assessed"}
        disc = ("Across repeated continuations, the evidence differs between the two guides: %s "
                "%s, while %s %s. The support for tissue-directed target selection therefore "
                "depends on the guide." % (PLACEMENT[arms[0]], state[codes[arms[0]]],
                                           PLACEMENT[arms[1]], state[codes[arms[1]]]))
    disc += {"M+": " In the matched-budget control, placement accounts for most of the same-seed "
                   "gap.",
             "M_partial": " In the matched-budget control, placement and budget both contribute "
                          "to the same-seed gap.",
             "M0": " In the matched-budget control, the placement difference at equal budgets is "
                   "within run-to-run variation.",
             "M-": " In the matched-budget control, uniform placement with equal budgets scores "
                   "above the guided strategy by more than run-to-run variation."}.get(m, "")
    t.put("RepDiscussionLead", disc, "replaces the first two discussion sentences")

    limits = []
    if n_seeds:
        limits.append(r"Only %s further continuation%s per strategy entered the registered "
                      r"analysis, the endpoint is epoch~%s only, and the per-seed intervals "
                      r"describe test cases, not retraining."
                      % (t.bare("RepNSeedsWord"), "" if n_seeds == 1 else "s", t.bare("RepEndpoint")))
    if scen == "B" or (scen == "mixed" and "B" in codes.values()):
        limits.append("With this few continuations the comparison has little power to detect "
                      "differences of the size seen in the original runs.")
    if scen == "C" or (scen == "mixed" and "C" in codes.values()):
        limits.append("Single-run differences of this size between masking strategies should "
                      "therefore be read as preliminary.")
    if scen == "I" or "I" in codes.values():
        limits.append("The registered replication did not finish before the results freeze, so "
                      "its completed runs are descriptive.")
    if m in CONTROL_LABELS or m in ("M_unassessed", "M_incomplete"):
        limits.append("The matched-budget control is a single continuation per arm.")
    elif m in ("M_desc", "M_noP"):
        limits.append("The matched-budget control is incomplete and reported descriptively.")
    t.put("RepLimitations", " ".join(limits), "limitations sentence after the continuation caveat")

    if scen == "A":
        concl = ("Across repeated continuations from the same checkpoint, the gain from "
                 "tissue-directed placement keeps its direction for both guides.")
    elif scen == "B":
        concl = ("Across repeated continuations, the gains do not meet the registered criterion "
                 "for a consistent direction, so their size remains uncertain.")
    elif scen == "C":
        concl = ("Across repeated continuations, tissue-directed placement does not reliably "
                 "improve on uniform masking.")
    elif scen == "I":
        concl = ("The registered repeated continuations did not all finish before the results "
                 "freeze.")
    else:
        arms = ordered_arms(codes)
        concl = ("Across repeated continuations, the gain keeps its direction for %s but not for "
                 "%s." % (PLACEMENT[arms[0]], PLACEMENT[arms[1]])
                 if codes[arms[0]] == "A" else
                 "Across repeated continuations, neither guide meets the registered criterion "
                 "for a consistent direction.")
    concl += {"M+": " With matched target and context budgets, placement accounts for most of the "
                    "same-seed gap in a single continuation.",
              "M_partial": " With matched target and context budgets, placement and budget both "
                           "contribute in a single continuation.",
              "M0": " With matched target and context budgets, the placement difference is within "
                    "run-to-run variation in a single continuation.",
              "M-": " With matched budgets, uniform placement scores higher by more than "
                    "run-to-run variation in a single continuation."}.get(m, "")
    t.put("RepConclusion", concl, "conclusion sentence")

    # ---- appendix and figure ---------------------------------------------------------------
    appendix = (r"Table~\ref{tab:replication_runs} lists every evaluated encoder with its paired "
                r"interval and Table~\ref{tab:pooling} the pooling variants%s."
                % (r", and Figure~\ref{fig:seed_spread} shows the spread across runs"
                   if getattr(ctx, "figure", True) else ""))
    if ctx.md is not None:
        appendix += (r" In glaucoma cases of the test split ($n=%s$), ridge regression on the "
                     r"frozen features predicts visual-field mean deviation with a mean absolute "
                     r"error from %s to %s dB across encoders, against %s dB for the training "
                     r"mean (Table~\ref{tab:md}). Mean deviation partly defines the glaucoma "
                     r"label, so this is a severity endpoint on the same cohort, not an "
                     r"independent task." % ("\\RepMDN", t.num("RepMDMAEMin"),
                                             t.num("RepMDMAEMax"), t.num("RepMDBaseMAE")))
    t.put("RepAppendixResults", appendix, "appendix results paragraph")
    if n_seeds >= 2:
        rule = (r"A seed index enters the primary analysis only if its \textsc{random}, \ArmBest{} "
                r"and \textsc{envelope} runs all finished before the results freeze (%s did). A "
                r"guided strategy is labelled consistent in direction only if %s new-seed "
                r"differences are positive and their mean exceeds the range of the %s "
                r"\textsc{random} realizations (the original and the new runs of those indices)."
                % (t.bare("RepNSeedsWord"), t.bare("RepAllSeedsWords"), t.bare("RepNRandomWord")))
    else:
        rule = (r"A seed index enters the primary analysis only if its \textsc{random}, \ArmBest{} "
                r"and \textsc{envelope} runs all finished before the results freeze (%s did, so "
                r"no label is assigned). With at least two such indices, a guided strategy is "
                r"labelled consistent in direction only if every new-seed difference is positive "
                r"and their mean exceeds the range of the \textsc{random} realizations (the "
                r"original and the new runs of those indices)." % t.bare("RepNSeedsWord"))
    rule += (r" The matched control is described by the placement difference P (\ArmBest{} minus "
             r"\textsc{random-cb}) and the budget shift B (\textsc{random-cb} minus \textsc{random}) "
             r"at the first seed, judged against the same range: placement difference not "
             r"detected if the absolute value of P is within it, reversed if P is below its "
             r"negative, placement accounting for most of the gap if P exceeds it and is at least "
             r"half of P plus B, and both contributing otherwise.")
    t.put("RepAppendixRule", rule, "appendix decision rule (amendments one and two)")
    t.put("RepFigSeedSpreadCaption",
          r"Test AUC at epoch~%s of every evaluated encoder, averaged over %s probe-head seeds "
          r"(error bars: head-seed s.d.). Filled markers are new continuations in the registered "
          r"analysis, open markers are descriptive runs and stars are the re-probed original "
          r"encoders. Horizontal bars are means over the registered new seeds. Continuations share "
          r"an ancestor checkpoint, so the spread is across continuations, not independent "
          r"initializations." % (t.bare("RepEndpoint"), t.bare("RepNHeadSeedsWord")),
          "figure caption")
    return t


def control_text(t, ctx):
    """Matched-control paragraph: only available contrasts, labels only from on-time runs."""
    info, m = ctx.matched, ctx.matched["label"]
    if m == "M_na":
        return (r"The control did not produce an epoch-%s result before the results freeze, so "
                r"no matched comparison is reported." % t.bare("RepEndpoint"))
    seed = SEED_LETTERS[info["seed"]]
    out = []
    rng = (r"the range of \textsc{random}'s realizations (%s)" % t.num("RepRandomRange")
           if "RepRandomRange" in t.values else None)
    if info["cb_present"] and not info["cb_eligible"]:
        out.append(r"\textsc{random-cb} finished after the results freeze, so its comparisons are "
                   r"descriptive only.")
    if info["P"] is None:
        out.append(r"The \ArmBest{} run of seed %s did not finish before the results freeze, so "
                   r"the placement difference is not available." % seed)
    elif info["P_descriptive"] and info["cb_eligible"]:
        out.append(r"The \ArmBest{} run of seed %s finished after the results freeze, so the "
                   r"placement difference is descriptive only." % seed)
    if info["B"] is None:
        out.append(r"The \textsc{random} run of seed %s did not finish before the results freeze, "
                   r"so the budget shift is not available." % seed)
    rng_ref = rng
    if info["B"] is not None and "RepBudgetShift" in t.values:
        if info["B_descriptive"]:
            out.append(r"With these budgets, \textsc{random-cb} differs from \textsc{random} of the "
                       r"same seed by %s (descriptive)." % t.num("RepBudgetShift"))
        else:
            beyond = rng and abs(info["B"]) > info["spread"]
            out.append(r"With these budgets, \textsc{random-cb} differs from \textsc{random} of "
                       r"the same seed by %s%s." % (
                           t.num("RepBudgetShift"),
                           (", %s %s" % ("beyond" if beyond else "within", rng)) if rng else ""))
            rng_ref = "that range" if rng else None
    if info["P"] is not None and "RepPlacementDelta" in t.values:
        place = (r"The placement difference, \ArmBest{} minus \textsc{random-cb}, is %s"
                 % t.num("RepPlacementDelta"))
        gap = "RepDeltaCentroid" + seed_tag(info["seed"])
        gap_text = t.num(gap) if gap in t.values else None
        named = r" Its registered description is \emph{%s}." % info["name"] if info["name"] else ""
        if m == "M+":
            out.append(place + r". It exceeds %s and is at least half of the same-seed difference "
                       r"between \ArmBest{} and \textsc{random}%s.%s" % (
                           rng_ref, " (%s)" % gap_text if gap_text else "", named))
        elif m == "M_partial":
            share = ((r" Matching the budgets accounts for %s\%% of that difference."
                      % t.num("RepBudgetShareOfGap")) if "RepBudgetShareOfGap" in t.values else "")
            out.append(place + r". It exceeds %s but is less than half of the same-seed difference "
                       r"between \ArmBest{} and \textsc{random}%s.%s%s" % (
                           rng_ref, " (%s)" % gap_text if gap_text else "", named, share))
        elif m == "M0":
            out.append(place + r", within %s.%s" % (rng_ref, named))
        elif m == "M-":
            out.append(place + r", negative and beyond %s, so uniform placement with matched "
                       r"budgets scores above \ArmBest{}.%s" % (rng_ref, named))
        elif m == "M_unassessed":
            out.append(place + r", beyond %s. Without an on-time same-seed \textsc{random} run, "
                       r"the registered description is not assigned." % rng)
        elif m == "M_incomplete":
            out.append(place + r". The range of \textsc{random}'s realizations is not available, "
                       r"so no registered description is assigned.")
        else:
            out.append(place + " (descriptive).")
        if "RepPurityCentroid" in t.values and "RepPurityRandomCB" in t.values:
            out.append(r"The comparison is a single continuation per arm, and the two policies "
                       r"differ only modestly in target tissue purity (%s\%% versus %s\%%), so a "
                       r"null placement difference would be weak evidence that placement does not "
                       r"matter." % (t.num("RepPurityCentroid"), t.num("RepPurityRandomCB")))
        else:
            out.append("The comparison is a single continuation per arm, so its own run-to-run "
                       "variation is not estimated.")
    return " ".join(out)


def pooling_text(t, ctx):
    final, labels = ctx.final, ctx.labels
    variants = [v for v in VARIANTS[1:] if v in final["analysis"]]
    if not variants:
        return "The alternative poolings were not available."
    out = []
    ranges = []
    for arm in GUIDED:
        lo, hi = "RepPoolDeltaMin" + ARM_WORD[arm], "RepPoolDeltaMax" + ARM_WORD[arm]
        if lo in t.values:
            ranges.append(r"from %s to %s for %s" % (t.num(lo), t.num(hi), ARM_TEX[arm]))
    if ranges:
        out.append(r"Across the alternative poolings, the mean difference from \textsc{random} "
                   r"over the new continuations ranges %s (Table~\ref{tab:pooling})."
                   % join_and(ranges))
    changed, unlabeled = [], []
    random_anchor = ctx.slots.get(("random", "Orig"))
    for arm in GUIDED:
        diff = [VARIANT_SHORT[v] for v in variants
                if LABEL_CODE[final["analysis"][v]["outcomes"][arm]["label"]] != "I"
                and final["analysis"][v]["outcomes"][arm]["label"] != labels[arm]]
        if diff:
            changed.append("for %s with %s" % (ARM_TEX[arm], join_and(diff)))
    for v in variants:
        if any(LABEL_CODE[final["analysis"][v]["outcomes"][arm]["label"]] == "I"
               and LABEL_CODE[labels[arm]] != "I" for arm in GUIDED):
            unlabeled.append(v)
    if changed:
        out.append("The registered label differs from the primary pooling %s." % join_and(changed))
    elif len(unlabeled) < len(variants):
        out.append("Where a label can be assigned, it is the same as with the primary pooling.")
    if unlabeled:
        missing_anchor = random_anchor is not None and all(
            random_anchor not in final["analysis"][v]["per_run_test_auc"] for v in unlabeled)
        out.append(r"No label is assigned with %s, because %s." % (
            join_and(VARIANT_SHORT[v] for v in unlabeled),
            r"the original \textsc{random} encoder was not evaluated with %s" % (
                "it" if len(unlabeled) == 1 else "them")
            if missing_anchor else "the variant is missing for a run the rule needs"))
    out.append("These analyses are secondary and descriptive.")
    return " ".join(out)


def tldr_text(ctx):
    scen, codes, m = ctx.scenario, ctx.codes, ctx.matched["label"]
    if scen == "A" and m in ("M0", "M-"):
        text = ("Tissue-directed I-JEPA targets on retinal OCT give higher frozen glaucoma-probe "
                "AUC across repeated continuations; with matched target and context budgets, "
                + ("uniform placement comes within run-to-run variation of the intensity guide."
                   if m == "M0" else
                   "uniform placement scores above the intensity guide in a single continuation."))
    elif scen == "A":
        text = ("Placing I-JEPA prediction targets on retinal tissue, even with a "
                "segmentation-free intensity guide, gives higher frozen glaucoma-probe AUC than "
                "uniform masking in every repeated continuation; more anatomical detail does not "
                "help further.")
    elif scen == "B":
        text = ("Tissue-directed I-JEPA targets on retinal OCT give positive mean gains that fall "
                "within run-to-run variation under a pre-registered rule; mask audits show target "
                "placement, size and visible context are coupled.")
    elif scen == "C":
        text = ("Single-run gains from anatomy-guided I-JEPA targets on retinal OCT do not "
                "replicate across repeated continuations; mask audits show why masking comparisons "
                "need matched context and target budgets.")
    elif scen == "I":
        text = ("Anatomy-guided I-JEPA target placement gave higher frozen glaucoma-probe AUC on "
                "retinal OCT in single runs; mask audits show that target placement, target size "
                "and visible context are coupled.")
    else:
        verb = {"A": "keeps its gain", "B": "is within run-to-run variation",
                "C": "does not beat uniform masking", "I": "is incomplete"}
        guide = {"centroid": "the intensity guide", "envelope": "the segmentation guide"}
        arms = ordered_arms(codes)
        text = ("Tissue-directed I-JEPA targets on retinal OCT over repeated continuations: %s "
                "%s, %s %s; mask audits show placement, size and visible context are coupled."
                % (guide[arms[0]], verb[codes[arms[0]]], guide[arms[1]], verb[codes[arms[1]]]))
    if len(text) > TLDR_LIMIT or not text.isascii():
        raise ValueError("TL;DR exceeds %d characters" % TLDR_LIMIT)
    return text


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def cell(values, name, math=False):
    if name not in values:
        return "---"
    return ("$\\%s$" if math else "\\%s") % name


def delta_cell(values, name, interval=True):
    if name not in values:
        return "---"
    text = "$\\%s$" % name
    if interval and name + "CI" in values:
        text += " $\\%sCI$" % name
    return text


def float_table(label, caption, colspec, header, rows, size=r"\footnotesize", sep=None):
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption, r"\label{%s}" % label,
             size]
    if sep:
        lines.append(r"\setlength{\tabcolsep}{%s}" % sep)
    lines += [r"\begin{tabular}{%s}" % colspec, r"\toprule", header + r" \\", r"\midrule"]
    lines += rows
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def row_text(cells):
    return " & ".join(cells) + r" \\"


def table_replication(ctx):
    v = ctx.values

    def row(label, tag):
        cells = [label] + [cell(v, "RepAUC%s%s" % (ARM_WORD[a], tag)) for a in SEED_ARMS]
        for arm in GUIDED:
            name = ("RepDeltaOrig" + ARM_WORD[arm]) if tag == "Orig" else \
                "RepDelta%s%s" % (ARM_WORD[arm], tag)
            cells.append(cell(v, name, math=True))
        return row_text(cells)

    rows = []
    if any((a, "Orig") in ctx.slots for a in SEED_ARMS):
        rows.append(row("original (re-probed)", "Orig"))
    for s in ctx.new_seeds:
        rows.append(row("seed %s%s" % (SEED_LETTERS[s], "" if s in ctx.complete else "$^{*}$"),
                        seed_tag(s)))
    rows.append(r"\midrule")
    rows.append(row_text(["mean (new seeds)"] +
                         [cell(v, "RepMeanAUC" + ARM_WORD[a]) for a in SEED_ARMS] +
                         [cell(v, "RepDeltaMean" + ARM_WORD[g], math=True) for g in GUIDED]))
    rows.append(row_text(["s.d. (new seeds)"] +
                         [cell(v, "RepSDAUC" + ARM_WORD[a]) for a in SEED_ARMS] + ["", ""]))
    caption = (r"Registered replication at epoch~\RepEndpoint{}: test AUC, mean over "
               r"\RepNHeadSeedsWord{} probe-head seeds (fp32 probes")
    if "RepHeadSDMax" in v:
        caption += r", head-seed s.d.\ at most $\RepHeadSDMax$"
    caption += "). Original encoders are re-probed with the same protocol"
    if "RepReprobeMaxDiff" in v:
        caption += r" (within $\RepReprobeMaxDiff$ of Table~\ref{tab:main})"
    caption += (r". Mean and s.d.\ are over the new seeds in the registered analysis. $\Delta$ is "
                r"the paired difference from \textsc{random} of the same seed, descriptive for the "
                r"original row.")
    if any(s not in ctx.complete for s in ctx.new_seeds):
        caption += r" $^{*}$Seed index incomplete at the freeze, not in the analysis."
    caption += r" Intervals: Table~\ref{tab:replication_runs}."
    header = (r" & \textsc{random} & \ArmBest{} & \textsc{envelope} & $\Delta$ \ArmBest{} & "
              r"$\Delta$ \textsc{envelope}")
    return float_table("tab:replication", caption, "lccccc", header, rows, sep="4pt")


def run_status(ctx, r):
    if r["role"] == "anchor":
        return "original"
    if r["arm"] == "random_cb":
        return "control" if ctx.matched["label"] not in ("M_desc", "M_na") and r["eligible"] \
            else "descriptive"
    return "registered" if r["seed"] in ctx.complete else "descriptive"


def ordered_runs(ctx, names=None):
    arms = SEED_ARMS + ("random_cb",)
    tags = ("Orig", "SeedA", "SeedB", "SeedC")
    names = list(ctx.runs) if names is None else list(names)
    return sorted(names, key=lambda n: (arms.index(ctx.runs[n]["arm"]),
                                        tags.index(ctx.runs[n]["tag"])))


def table_seeds(ctx):
    v = ctx.values
    rows = []
    for name in ordered_runs(ctx):
        r = ctx.runs[name]
        w = ARM_WORD[r["arm"]] + r["tag"]
        if r["arm"] == "random":
            delta, val = "---", "---"
        elif r["arm"] == "random_cb":
            delta = delta_cell(v, "RepBudgetShift")
            val = cell(v, "RepValBudgetShift", math=True)
        elif r["tag"] == "Orig":
            delta = cell(v, "RepDeltaOrig" + ARM_WORD[r["arm"]], math=True)
            val = cell(v, "RepValDeltaOrig" + ARM_WORD[r["arm"]], math=True)
        else:
            delta = delta_cell(v, "RepDelta" + w)
            val = cell(v, "RepValDelta" + w, math=True)
        run = "original" if r["tag"] == "Orig" else "seed " + SEED_LETTERS[r["seed"]]
        rows.append(row_text([ARM_TEX[r["arm"]], run, cell(v, "RepAUC" + w),
                              cell(v, "RepHeadSD" + w), cell(v, "RepValAUC" + w), delta, val,
                              run_status(ctx, r)]))
    caption = (r"Every evaluated encoder at epoch~\RepEndpoint{}. AUC is the mean over "
               r"\RepNHeadSeedsWord{} probe-head seeds, with their s.d. $\Delta$ is the difference "
               r"from \textsc{random} of the same seed with a paired, label-stratified test-case "
               r"bootstrap interval (\RepNBoot{} resamples), and the validation difference is "
               r"given for comparison. Intervals describe test-case sampling, not retraining. "
               r"Original encoders are compared descriptively. Registered: in the primary "
               r"analysis. Descriptive: seed index incomplete or run finished after the freeze.")
    header = (r"policy & run & test AUC & head s.d. & val.\ AUC & $\Delta$ test (interval) & "
              r"$\Delta$ val. & analysis")
    return float_table("tab:replication_runs", caption, "llccclcl", header, rows,
                       size=r"\scriptsize", sep="3pt")


def table_control(ctx):
    v, info = ctx.values, ctx.matched
    if info["label"] == "M_na" or info["seed"] is None:
        return "% No matched-budget control result (" + str(info["reason"]) + ").\n"
    tag = seed_tag(info["seed"])
    budget_cb = "RepBudgetURandomCB" in v

    def budgets(arm):
        word_ = ARM_WORD[arm]
        if arm == "random_cb" and not budget_cb:
            word_ = "Centroid" if "RepCBMismatches" in v else None
        cells = [cell(v, "RepBudget%s%s" % (k, word_)) if word_ else "---"
                 for _, k in BUDGET_KEYS]
        return cells + [cell(v, "RepPurity" + ARM_WORD[arm])]

    rows = [row_text([r"\textsc{random}", cell(v, "RepAUCRandom" + tag), "---"] +
                     budgets("random")),
            row_text([r"\textsc{random-cb}", cell(v, "RepAUCRandomCB" + tag),
                      delta_cell(v, "RepBudgetShift")] + budgets("random_cb")),
            row_text([r"\ArmBest{}", cell(v, "RepAUCCentroid" + tag),
                      delta_cell(v, "RepDeltaCentroid" + tag)] + budgets("centroid")),
            r"\midrule",
            row_text([r"\multicolumn{2}{l}{\ArmBest{} $-$ \textsc{random-cb}}",
                      delta_cell(v, "RepPlacementDelta"),
                      r"\multicolumn{5}{l}{placement at matched budgets}"])]
    prov = getattr(ctx, "budget_provenance", {})
    caption = (r"Matched-budget control at epoch~\RepEndpoint{}, seed %s, one continuation per "
               r"arm. $\Delta$ is the paired difference from \textsc{random} with a test-case "
               r"interval. Budgets are per image at the training batch size: unique targets (U), "
               r"loss slots (L), duplicates (D) and context tokens (C)." % SEED_LETTERS[info["seed"]])
    sources = []
    if prov.get("budget_random") == "i5" or prov.get("budget_centroid") == "i5":
        sources.append(r"\textsc{random} and \ArmBest{} come from the audit of "
                       r"Table~\ref{tab:geometry}")
    elif prov.get("budget_random") == "cb":
        sources.append(r"\textsc{random} and \ArmBest{} come from the control audit")
    if budget_cb:
        text = r"\textsc{random-cb} comes from its run's mask statistics"
        if "RepCBBatches" in v:
            text += r" ($\RepCBBatches$ batches"
            text += (r", mean hidden-area residual $\RepCBHiddenDiff$)" if "RepCBHiddenDiff" in v
                     else ")")
        sources.append(text)
    if "RepCBMismatches" in v:
        sources.append(r"\textsc{random-cb} equals its \ArmBest{} shadow per image "
                       r"($\RepCBMismatches$ mismatches in $\RepCBMatchImages$ image draws)")
    if sources:
        text = join_and(sources)
        caption += " " + text[0].upper() + text[1:] + "."
    if "RepPurityCentroid" in v:
        caption += r" Purity uses a hard MIRAGE-envelope tissue proxy"
        if "RepPurityViews" in v:
            caption += r" ($\RepPurityViews$ views)"
        if prov.get("purity") == "i7":
            caption += r", not comparable with Table~\ref{tab:geometry}"
        caption += "."
    header = (r"policy & test AUC & $\Delta$ (interval) & U & L & D & C & purity (\%)")
    return float_table("tab:control", caption, "lccccccc", header, rows,
                       size=r"\scriptsize", sep="3pt")


def table_pooling(ctx):
    v, final = ctx.values, ctx.final
    rows = []
    for variant in VARIANTS:
        if variant not in final["analysis"]:
            continue
        labels = arm_labels(final["analysis"][variant])
        if variant == PRIMARY:
            aucs = [cell(v, "RepMeanAUC" + ARM_WORD[a]) for a in SEED_ARMS]
            deltas = [cell(v, "RepDeltaMean" + ARM_WORD[g], math=True) for g in GUIDED]
        else:
            vw = VARIANT_WORD[variant]
            aucs = [cell(v, "RepPool%sAUC%s" % (vw, ARM_WORD[a])) for a in SEED_ARMS]
            deltas = [cell(v, "RepPool%sDelta%s" % (vw, ARM_WORD[g]), math=True) for g in GUIDED]
        rows.append(row_text([VARIANT_TEX[variant]] + aucs + deltas +
                             [LABEL_SHORT[labels[g]] for g in GUIDED]))
    caption = (r"Pooling variants at epoch~\RepEndpoint{} (secondary, descriptive), given as "
               r"pooling of patch tokens within a B-scan, then across B-scans: mean test AUC "
               r"over the new seeds in the registered analysis, mean paired difference from "
               r"\textsc{random}, and the label from the registered rule (consistent: consistent "
               r"direction, within: within run-to-run variation). Runs lacking a variant are "
               r"omitted for that variant.")
    header = (r"pooling & \textsc{random} & \ArmBest{} & \textsc{envelope} & $\Delta$ \ArmBest{} & "
              r"$\Delta$ \textsc{envelope} & label \ArmBest{} & label \textsc{envelope}")
    return float_table("tab:pooling", caption, "lccccccc", header, rows, size=r"\scriptsize",
                       sep="3pt")


def table_md(ctx):
    if ctx.md is None:
        return "% No MD severity result (cr_md_regression.py --split test not supplied).\n"
    v = ctx.values
    rows = []
    for name in ctx.md_names:
        r = ctx.runs[name]
        w = ARM_WORD[r["arm"]] + r["tag"]
        dw = "RepMDDeltaMAE" + w
        rows.append(row_text([ARM_TEX[r["arm"]],
                              "original" if r["tag"] == "Orig" else "seed " + SEED_LETTERS[r["seed"]],
                              cell(v, "RepMDMAE" + w), cell(v, "RepMDRMSE" + w),
                              cell(v, "RepMDSpearman" + w), delta_cell(v, dw),
                              cell(v, "RepMDDeltaSpearman" + w, math=True)]))
    rows.append(r"\midrule")
    rows.append(row_text([r"\multicolumn{2}{l}{training-mean baseline}", cell(v, "RepMDBaseMAE"),
                          cell(v, "RepMDBaseRMSE"), "---", "", ""]))
    caption = (r"Visual-field mean deviation (MD, dB) regressed from frozen epoch-\RepEndpoint{} "
               r"features in glaucoma cases of the test split ($n=\RepMDN$), ridge regression with "
               r"the penalty chosen on validation. Lower MAE and RMSE and higher Spearman "
               r"correlation are better. $\Delta$ is the difference from \textsc{random} of the "
               r"same seed with a paired case-bootstrap interval. MD partly defines the glaucoma "
               r"label, so this is a severity endpoint on the same cohort, not an independent "
               r"task.")
    excluded = [n for n in (ctx.md.get("excluded_runs") or {}) if n in ctx.runs]
    if excluded:
        listed = join_and("%s %s" % (ARM_TEX[ctx.runs[n]["arm"]], "original"
                                     if ctx.runs[n]["tag"] == "Orig"
                                     else "seed " + SEED_LETTERS[ctx.runs[n]["seed"]])
                          for n in ordered_runs(ctx, excluded))
        caption += (r" Encoders without a per-case feature-cache identity (%s) are not included."
                    % listed)
    header = (r"policy & run & MAE & RMSE & Spearman & $\Delta$MAE (interval) & "
              r"$\Delta$Spearman")
    return float_table("tab:md", caption, "llcccll", header, rows, size=r"\scriptsize",
                       sep="3pt")


def figure_float():
    return "\n".join([r"\begin{figure}[t]", r"\centering",
                      r"\includegraphics[width=0.62\linewidth]{fig_cr_seed_spread.png}",
                      r"\caption{\RepFigSeedSpreadCaption}", r"\label{fig:seed_spread}",
                      r"\end{figure}"]) + "\n"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

FIG_COLORS = {"random": "#555555", "centroid": "#0072B2", "envelope": "#D55E00",
              "random_cb": "#009E73"}
FIG_OFFSETS = {"Orig": -0.24, "SeedA": -0.08, "SeedB": 0.08, "SeedC": 0.24}
FIG_MARKERS = {"SeedA": "o", "SeedB": "s", "SeedC": "^"}


def figure_points(ctx):
    block, final = ctx.block, ctx.final
    arms = [a for a in SEED_ARMS + ("random_cb",) if any(r["arm"] == a for r in ctx.runs.values())]
    points = []
    for name in ordered_runs(ctx):
        r = ctx.runs[name]
        sd = ctx.records[name]["receipt"]["variants"][PRIMARY].get("test_auc_sd")
        points.append({"run": name, "arm": r["arm"], "tag": r["tag"], "seed": r["seed"],
                       "x": arms.index(r["arm"]) + FIG_OFFSETS[r["tag"]],
                       "test_auc": block["per_run_test_auc"][name], "head_sd": sd,
                       "status": run_status(ctx, r)})
    means = {}
    for arm in SEED_ARMS:
        vals = [block["per_run_test_auc"][ctx.slots[(arm, seed_tag(s))]] for s in ctx.complete]
        if vals:
            means[arm] = {"x": arms.index(arm), "mean": sum(vals) / len(vals), "n": len(vals)}
    return arms, points, means


FIGURE_SCHEMA = "cr_fig_seed_spread_v1"


def figure_provenance(ctx, stage):
    """Schema, generator identity and input hashes of the figure (shared by make_figure and
    --check, so a corrupted provenance field is detected)."""
    return {"schema": FIGURE_SCHEMA, "synthetic": ctx.synthetic,
            "script": "autopilot/make_cr_results.py", "script_sha256": sha256_file(__file__),
            "inputs": {name: {"path": spec["path"], "sha256": spec["sha256"]}
                       for name, spec in stage.sources.items()
                       if name == "rep_final" or name.startswith("rep_receipt_")}}


def make_figure(ctx, out_dir, stage):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    arms, points, means = figure_points(ctx)
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    for p in points:
        color = FIG_COLORS[p["arm"]]
        if p["tag"] == "Orig":
            marker, face, size = "*", color, 9
        else:
            marker, size = FIG_MARKERS[p["tag"]], 5
            face = color if p["status"] in ("registered", "control") else "none"
        if p["head_sd"] is not None:
            ax.errorbar(p["x"], p["test_auc"], yerr=p["head_sd"], fmt="none", ecolor=color,
                        elinewidth=0.6, capsize=1.5, zorder=2)
        ax.plot(p["x"], p["test_auc"], marker=marker, markersize=size, markerfacecolor=face,
                markeredgecolor="black" if p["tag"] == "Orig" else color, markeredgewidth=0.6,
                linestyle="none", zorder=3)
    for arm, mean in means.items():
        ax.plot([mean["x"] - 0.34, mean["x"] + 0.34], [mean["mean"]] * 2, color="black",
                linewidth=1.0, zorder=1)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_PLAIN[a] for a in arms], fontsize=6)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylabel("test AUC (epoch %d)" % cr_stats.ENDPOINT_EPOCH)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker="*", color="grey", markeredgecolor="black", linestyle="none",
                      markersize=7, label="original (re-probed)")]
    for tag in ("SeedA", "SeedB", "SeedC"):
        if any(p["tag"] == tag for p in points):
            handles.append(Line2D([], [], marker=FIG_MARKERS[tag], color="grey", linestyle="none",
                                  markersize=4, label="seed " + tag[-1]))
    if any(p["status"] == "descriptive" for p in points):
        handles.append(Line2D([], [], marker="o", markerfacecolor="none", color="grey",
                              linestyle="none", markersize=4, label="descriptive"))
    handles.append(Line2D([], [], color="black", linewidth=1.0, label="mean (registered seeds)"))
    ax.legend(handles=handles, fontsize=5.5, frameon=False, loc="upper left",
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    png, pdf = out_dir / "fig_cr_seed_spread.png", out_dir / "fig_cr_seed_spread.pdf"
    fig.savefig(png, dpi=300, metadata={"Software": None})
    fig.savefig(pdf, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)
    evidence = dict(figure_provenance(ctx, stage), matplotlib=matplotlib.__version__)
    evidence.update({
        "values": "test_auc = final analysis/<primary>/per_run_test_auc; head_sd = unseal "
                  "receipt variants/<primary>/test_auc_sd; mean over complete seed indices",
        "arms": arms, "points": points, "means": means,
        "outputs": {"png": sha256_file(png), "pdf": sha256_file(pdf)}})
    path = out_dir / "fig_cr_seed_spread.evidence.json"
    path.write_text(json.dumps(evidence, indent=1) + "\n", encoding="utf-8", newline="\n")
    return evidence


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

class Ctx(object):
    pass


def display_warnings(ctx):
    """Rounded displays that could read against the label they support."""
    v, out = ctx.values, []

    def num(name):
        return float(v[name].replace("{,}", "")) if name in v else None

    for name, value in v.items():
        if re.fullmatch(r"-0\.0+", value):
            out.append("%s displays %s" % (name, value))
    rng = num("RepRandomRange")
    for arm in GUIDED:
        o, w = ctx.block["outcomes"][arm], ARM_WORD[arm]
        mean = num("RepDeltaMean" + w)
        if o["label"] == "consistent_direction":
            if rng is not None and mean is not None and not mean > rng:
                out.append("%s: label consistent_direction but displayed mean %s <= range %s"
                           % (arm, v["RepDeltaMean" + w], v["RepRandomRange"]))
            for s in ctx.complete:
                d = num("RepDelta%s%s" % (w, seed_tag(s)))
                if d is not None and not d > 0:
                    out.append("%s seed %s: positive delta displays as %s"
                               % (arm, SEED_LETTERS[s], v["RepDelta%s%s" % (w, seed_tag(s))]))
        if o["label"] == "reversed" and mean is not None and mean > 0:
            out.append("%s: label reversed but displayed mean %s" % (arm, v["RepDeltaMean" + w]))
    p = num("RepPlacementDelta")
    if p is not None and rng is not None:
        within = abs(p) <= rng
        if within != (ctx.matched["label"] in ("M0",)) and ctx.matched["label"] in ("M0", "M+",
                                                                                     "M-", "M_partial"):
            out.append("matched label %s but displayed |P| %s vs range %s"
                       % (ctx.matched["label"], v["RepPlacementDelta"], v["RepRandomRange"]))
    return out


def render_results_tex(ctx, text, stage):
    lines = ["% Generated by autopilot/make_cr_results.py; do not edit values.",
             "% Registered replication results (cr_seed_v1). Numeric macros are typed expressions",
             "% over the staged evidence below (see cr_results.values.json); text macros contain",
             "% no digits and take every number from a numeric macro."]
    if ctx.synthetic:
        lines.append("% SYNTHETIC DRY RUN: these are not results.")
    lines.append("%% Scenario: %s; labels centroid=%s envelope=%s; matched control %s"
                 % (ctx.scenario, ctx.labels["centroid"], ctx.labels["envelope"],
                    ctx.matched["label"]))
    lines.append("% Sources:")
    lines += ["%%   %s  %s" % (name, spec["sha256"]) for name, spec in sorted(stage.sources.items())]
    if ctx.p1c_sha:
        lines.append("%%   p1c_stats.json  %s" % ctx.p1c_sha)
    lines += [r"\newcommand{\%s}{%s}" % (name, ctx.values[name]) for name in ctx.order]
    lines.append("% Scenario text")
    lines += [r"\newcommand{\%s}{%s}" % (name, value) for name, value in text.macros.items()]
    return "\n".join(lines) + "\n"


def prepare(final_path, run_dirs, md_path=None, budget_paths=(), stats_dir=DEFAULT_STATS,
            evidence_dir=DEFAULT_EVIDENCE, allow_synthetic=False, i7_cell="strict_ep30",
            check=False):
    """Verify the inputs, stage them and evaluate every numeric macro. Returns the context."""
    evidence_dir = Path(evidence_dir).resolve()
    if allow_synthetic and inside(evidence_dir, REPO):
        raise InputError("synthetic evidence must stay outside the repository")
    final, _ = read_json(final_path)
    verify_final(final, allow_synthetic)
    records, head_seeds = verify_runs(final, run_dirs, allow_synthetic)
    md = None
    if md_path:
        md, _ = read_json(md_path)
        verify_md(md, final, allow_synthetic)
    budget, budget_files = {}, {}
    for path in budget_paths or ():
        data, _ = read_json(path)
        kind = budget_kind(data)
        if kind in budget:
            raise InputError("two %s budget files" % kind)
        budget[kind], budget_files[kind] = data, path
    p1c, p1c_sha = None, None
    if stats_dir and (Path(stats_dir) / "p1c_stats.json").is_file():
        p1c, p1c_sha = read_json(Path(stats_dir) / "p1c_stats.json")

    ctx = Ctx()
    ctx.final, ctx.block, ctx.records, ctx.head_seeds = final, final["analysis"][PRIMARY], \
        records, head_seeds
    ctx.synthetic, ctx.md, ctx.budget, ctx.i7_cell = allow_synthetic, md, budget, i7_cell
    ctx.p1c, ctx.p1c_sha, ctx.stats_dir = p1c, p1c_sha, stats_dir
    ctx.runs, ctx.slots = run_table(final)
    ctx.cidx = contrast_index(ctx.block)
    ctx.complete = list(ctx.block["complete_seed_indices"])
    ctx.new_seeds = sorted(set(r["seed"] for r in ctx.runs.values()
                               if r["role"] == "new" and r["arm"] in SEED_ARMS))
    ctx.md_names = ordered_runs(ctx, md["runs"]) if md else []
    ctx.labels = arm_labels(ctx.block)
    ctx.scenario, ctx.codes = scenario_of(ctx.labels)
    ctx.matched = matched_outcome(ctx.block, ctx.runs, ctx.cidx)

    stage = Stage(evidence_dir)
    staged = {"rep_final": (final_path, "final.json")}
    if md is not None:
        staged["rep_md"] = (md_path, "md_test.json")
    for kind, path in budget_files.items():
        staged["rep_budget_" + kind] = (path, "budget_%s.json" % kind)
    for name, rec in records.items():
        staged[receipt_source(name)] = (rec["receipt_path"],
                                        "receipts/%s.unsealed_results.json" % name)
    if check:
        for source, (path, rel) in staged.items():
            dst = stage.root / rel
            if not dst.is_file() or dst.read_bytes() != Path(path).read_bytes():
                raise InputError("staged evidence is stale or missing: %s" % dst)
    for source, (path, rel) in staged.items():
        stage.add(source, path, rel)

    builder = numeric_specs(ctx)
    evidence = stage.evidence(stats_dir if p1c is not None else None)
    ctx.values, macro_records = builder.evaluate(evidence)
    ctx.order = builder.order
    return ctx, stage, macro_records


def build(final_path, run_dirs, md_path=None, budget_paths=(), stats_dir=DEFAULT_STATS,
          out_dir=DEFAULT_OUT, evidence_dir=DEFAULT_EVIDENCE, allow_synthetic=False,
          i7_cell="strict_ep30", figure=True, check=False, update_reviews=False):
    out_dir, evidence_dir = Path(out_dir).resolve(), Path(evidence_dir).resolve()
    if allow_synthetic and (inside(out_dir, PAPER) or inside(evidence_dir, REPO)):
        raise InputError("synthetic outputs must stay outside the paper and the repository")
    if update_reviews and (allow_synthetic or not inside(evidence_dir, REPO)):
        raise InputError("--update-reviews needs real inputs staged inside the repository")
    ctx, stage, macro_records = prepare(final_path, run_dirs, md_path, budget_paths, stats_dir,
                                        evidence_dir, allow_synthetic, i7_cell, check)
    records, head_seeds, p1c, p1c_sha = ctx.records, ctx.head_seeds, ctx.p1c, ctx.p1c_sha
    ctx.figure = bool(figure)
    text = build_text(ctx)
    tldr = tldr_text(ctx)
    outputs = {
        "cr_results.tex": render_results_tex(ctx, text, stage),
        "cr_table_replication.tex": table_replication(ctx),
        "cr_table_control.tex": table_control(ctx),
        "cr_table_pooling.tex": table_pooling(ctx),
        "cr_table_seeds.tex": table_seeds(ctx),
        "cr_table_md.tex": table_md(ctx),
        "cr_figure_seed_spread.tex": figure_float(),
        "cr_tldr.txt": tldr + "\n",
    }
    for name, content in outputs.items():
        if not content.isascii():
            raise ValueError("non-ASCII output: " + name)
        if name.endswith(".tex") and name != "cr_results.tex":
            body = assets.uncomment(content)
            if re.search(r"\d", re.sub(r"\\(?:begin\{tabular\}|setlength\{\\tabcolsep\})\{[^}]*\}|"
                                       r"\\multicolumn\{\d+\}|\\includegraphics\[[^\]]*\]|"
                                       r"fp(?:16|32)", "", body)):
                raise ValueError("literal digit in generated table/figure file " + name)
    warnings = display_warnings(ctx)
    if check:
        stale = [name for name, content in outputs.items()
                 if not (out_dir / name).is_file()
                 or (out_dir / name).read_text(encoding="utf-8") != content]
        stale += check_sidecar_and_figure(ctx, out_dir, outputs, figure, stage)
        return {"stale": stale, "warnings": warnings}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (out_dir / name).write_text(content, encoding="utf-8", newline="\n")
    fig_evidence = make_figure(ctx, out_dir, stage) if figure else None
    manifest = {
        "schema": SCHEMA, "synthetic": allow_synthetic,
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": {"path": "autopilot/make_cr_results.py", "sha256": sha256_file(__file__)},
        "inputs": stage.inputs,
        "probe_runs": {name: {"run_dir": rec["run_dir"], "identity": rec["identity"],
                              "results_sha256": rec["results_sha256"],
                              "sealed_manifest_sha256": rec["sealed_manifest_sha256"],
                              "unseal_receipt_sha256": rec["receipt_sha256"]}
                       for name, rec in sorted(records.items())},
        "head_seeds": list(head_seeds),
        "p1c_stats": {"path": str(Path(stats_dir) / "p1c_stats.json"), "sha256": p1c_sha}
        if p1c is not None else None,
        "scenario": {
            "rule": "per guided arm, primary pooling, final labels unchanged: "
                    "consistent_direction->A, within_run_to_run_variation->B, reversed->C, "
                    "incomplete/not_available->I; equal codes -> that scenario, else mixed",
            "labels": ctx.labels, "codes": ctx.codes, "scenario": ctx.scenario,
            "complete_seed_indices": ctx.complete,
            "incomplete_seed_indices": ctx.block.get("incomplete_seed_indices") or [],
            "matched_control": {k: val for k, val in ctx.matched.items()
                                if not k.endswith("_ref")},
            "matched_rule": "s = RANDOM range; M0 |P|<=s; M- P<-s; M+ P>s and P>=(P+B)/2; "
                            "M_partial P>s and P<(P+B)/2; M_desc late run; M_na no result"},
        "macros": macro_records,
        "text_macros": [{"name": n, "text": val, "note": text.notes.get(n)}
                        for n, val in text.macros.items()],
        "tldr": tldr, "warnings": warnings,
        "outputs": {name: sha256_bytes(content.encode("utf-8")) for name, content in outputs.items()},
        "figure": fig_evidence["outputs"] if fig_evidence else None}
    (out_dir / "cr_results.values.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                                    encoding="utf-8", newline="\n")
    (stage.root / "inputs_manifest.json").write_text(
        json.dumps({"schema": SCHEMA + "_inputs", "synthetic": allow_synthetic,
                    "inputs": stage.inputs, "probe_runs": manifest["probe_runs"]}, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    if update_reviews:
        write_reviews(stage, macro_records)
    return manifest


def check_sidecar_and_figure(ctx, out_dir, outputs, figure, stage):
    """--check: the sidecar must describe these outputs; the figure files must match their
    recorded hashes and the evidence must hold the values recomputed from the inputs."""
    stale = []
    sidecar_path = out_dir / "cr_results.values.json"
    try:
        sidecar = read_json(sidecar_path)[0]
    except (OSError, ValueError):
        return ["cr_results.values.json"] + (list(FIGURE_FILES) if figure else [])
    expected = {name: sha256_bytes(content.encode("utf-8")) for name, content in outputs.items()}
    inputs = {name: (spec or {}).get("sha256")
              for name, spec in (sidecar.get("inputs") or {}).items()}
    if sidecar.get("schema") != SCHEMA or sidecar.get("synthetic") is not ctx.synthetic \
            or sidecar.get("outputs") != expected \
            or (sidecar.get("generator") or {}).get("sha256") != sha256_file(__file__) \
            or inputs != {name: spec["sha256"] for name, spec in stage.sources.items()}:
        stale.append("cr_results.values.json")
    if not figure:
        return stale
    evidence_path = out_dir / "fig_cr_seed_spread.evidence.json"
    try:
        evidence = read_json(evidence_path)[0]
    except (OSError, ValueError):
        return stale + list(FIGURE_FILES)
    arms, points, means = figure_points(ctx)
    recomputed = json.loads(json.dumps({"arms": arms, "points": points, "means": means}))
    provenance = json.loads(json.dumps(figure_provenance(ctx, stage)))
    if any(evidence.get(k) != recomputed[k] for k in recomputed) or \
            any(evidence.get(k) != provenance[k] for k in provenance):
        stale.append("fig_cr_seed_spread.evidence.json")
    recorded = evidence.get("outputs") or {}
    for kind, name in (("png", "fig_cr_seed_spread.png"), ("pdf", "fig_cr_seed_spread.pdf")):
        path = out_dir / name
        if not path.is_file() or sha256_file(path) != recorded.get(kind) or \
                (sidecar.get("figure") or {}).get(kind) != recorded.get(kind):
            stale.append(name)
    return stale


FIGURE_FILES = ("fig_cr_seed_spread.png", "fig_cr_seed_spread.pdf",
                "fig_cr_seed_spread.evidence.json")


def write_reviews(stage, macro_records, path=REVIEWS):
    """Own sources rep_* and macros Rep* in numeric_reviews.json (make_cr_numbers owns CR*)."""
    raw = Path(path).read_bytes()
    crlf = b"\r\n" in raw
    data = json.loads(raw.decode("utf-8"))
    data["sources"] = {k: val for k, val in data.get("sources", {}).items()
                       if not k.startswith("rep_")}
    for name, spec in stage.sources.items():
        if spec["root"] != "repo":
            raise InputError("review sources must live in the repository: " + name)
        data["sources"][name] = dict(spec)
    data["macros"] = {k: val for k, val in data.get("macros", {}).items() if not k.startswith("Rep")}
    for record in macro_records:
        data["macros"][record["name"]] = {"expression": record["expression"]}
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if crlf:
        text = text.replace("\n", "\r\n")
    Path(path).write_bytes(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# Integration into a paper copy (scratch dry runs; the real insertion after the freeze)
# ---------------------------------------------------------------------------

def _ws(text):
    """Whitespace-insensitive pattern for an exact passage of the source."""
    return r"\s+".join(re.escape(token) for token in text.split())


INTEGRATION = (
    ("main_submission.tex", r"\input{auto/cr_numbers}",
     "\\input{auto/cr_numbers}\n\\input{auto/cr_results}"),
    ("main_submission.tex", r"\newcommand{\CRSeedResultsPending}{\ph{seed-replication result pending}}", ""),
    ("main_submission.tex", r"\newcommand{\CRControlResultsPending}{\ph{matched-budget control result pending}}", ""),
    ("main_submission.tex", r"\newcommand{\CRPoolingResultsPending}{\ph{pooling-variant result pending}}", ""),
    ("main_submission.tex", r"\newcommand{\CRCell}{\ph{--}}", ""),
    ("main_submission.tex",
     "Tissue-directed rectangle placement improves the observed classification results, including "
     "with segmentation-free intensity guidance, while more anatomically specific strategies do not "
     "consistently provide further gains. Analysis of the actual masks shows that target selection, "
     "target size and visible context are coupled. Our findings motivate anatomy-guided predictive "
     "learning for retinal OCT and better-controlled comparisons of what to predict and what to keep "
     "visible.", r"\RepAbstractResults{}"),
    ("main_submission.tex", r"It adds two further continuations each of",
     r"It adds \RepNRegisteredSeedsWord{} further continuations each of"),
    ("main_submission.tex", r"controlled ablations of target location. \CRSeedResultsPending",
     r"controlled ablations of target location. \RepIntroResults{}"),
    ("main_submission.tex",
     r"\item Frozen-probe results showing gains from tissue-directed rectangle placement, "
     r"alongside mixed results for more specific anatomical targeting.",
     "\\item \\RepContribResults\n\\RepContribControlItem"),
    ("main_submission.tex", r"and all run decisions use validation AUC. \CRSeedResultsPending",
     r"and all run decisions use validation AUC. \RepSetupResults{}"),
    ("main_submission.tex", r"\paragraph{Repeated continuations.} \CRSeedResultsPending",
     r"\paragraph{Repeated continuations.} \RepResultsSeeds{}"),
    ("main_submission.tex", r"\CRControlResultsPending", r"\RepResultsControl{}"),
    ("main_submission.tex", r"\CRPoolingResultsPending", r"\RepResultsPooling{}"),
    ("main_submission.tex", r"not across independent initializations. \CRSeedResultsPending",
     r"not across independent initializations. \RepLimitations{}"),
    ("main_submission.tex", r"including the context they leave visible. \CRSeedResultsPending",
     r"including the context they leave visible. \RepConclusion{}"),
    ("main_submission.tex",
     "The positive rectangle-policy results motivate anatomy-guided target selection without "
     "requiring a new encoder architecture. The segmentation-free centroid strategy is "
     "particularly useful as a simple baseline.", r"\RepDiscussionLead{}"),
    ("compact_protocol.tex", r"epoch-25 checkpoint with two new seeds, which set",
     r"epoch-25 checkpoint with \RepNRegisteredSeedsWord{} new seeds (the third added by a "
     r"registered amendment), which set"),
    ("compact_protocol.tex",
     r"A guided strategy is labelled consistent in direction only if both new-seed differences "
     r"are positive and their mean exceeds the range of the three \textsc{random} realizations.",
     r"\RepAppendixRule{}"),
    ("compact_protocol.tex", "variants are secondary. Every started run is reported.",
     "variants are secondary. Every started run is reported.\n\n\\RepAppendixResults\n\n"
     "\\input{auto/cr_table_seeds}\n\\input{auto/cr_table_pooling}\n\\input{auto/cr_table_md}\n"
     "\\input{auto/cr_figure_seed_spread}"),
)
REPLICATION_TABLE = re.compile(r"\\begin\{table\}\[t\](?:(?!\\end\{table\}).)*?"
                               r"\\label\{tab:replication\}.*?\\end\{table\}", re.S)
RESULT_FILES = ("cr_results.tex", "cr_table_replication.tex", "cr_table_control.tex",
                "cr_table_pooling.tex", "cr_table_seeds.tex", "cr_table_md.tex",
                "cr_figure_seed_spread.tex", "fig_cr_seed_spread.png", "fig_cr_seed_spread.pdf")


OBSOLETE_PHRASES = ("two further continuations", "two new seeds", "both new-seed differences",
                    r"three \textsc{random} realizations")


def verified_results_metadata(results_dir):
    """Fail closed: known schema, explicit synthetic flag, every output matching its hash."""
    path = Path(results_dir) / "cr_results.values.json"
    try:
        sidecar = read_json(path)[0]
    except (OSError, ValueError) as exc:
        raise InputError("results metadata missing or unreadable: %s (%s)" % (path, exc))
    if not isinstance(sidecar, dict) or sidecar.get("schema") != SCHEMA:
        raise InputError("results metadata has an unknown schema: %s" % path)
    if not isinstance(sidecar.get("synthetic"), bool):
        raise InputError("results metadata has no explicit synthetic flag: %s" % path)
    outputs = sidecar.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(OUTPUT_FILES):
        raise InputError("results metadata does not list the generated outputs: %s" % path)
    files = dict(outputs)
    figure = sidecar.get("figure")
    if figure is not None:
        if not isinstance(figure, dict) or set(figure) != {"png", "pdf"}:
            raise InputError("results metadata has malformed figure hashes: %s" % path)
        files.update({"fig_cr_seed_spread.png": figure["png"],
                      "fig_cr_seed_spread.pdf": figure["pdf"]})
    for name, digest in files.items():
        target = Path(results_dir) / name
        if not target.is_file() or sha256_file(target) != digest:
            raise InputError("%s does not match its recorded hash" % target)
    marked = "SYNTHETIC DRY RUN" in (Path(results_dir) / "cr_results.tex").read_text(
        encoding="utf-8")
    if marked != sidecar["synthetic"]:
        raise InputError("synthetic marker in cr_results.tex disagrees with the metadata")
    return sidecar


def integrate(paper_dir, results_dir, allow_real_paper=False):
    """Replace the result placeholders in a paper copy with the generated inputs.

    Each passage is matched whitespace-insensitively and must occur exactly once (plain
    replacement, no shell regex on the source). The real paper directory is refused unless
    allow_real_paper is set and the results are not synthetic.
    """
    paper_dir, results_dir = Path(paper_dir).resolve(), Path(results_dir).resolve()
    sidecar = verified_results_metadata(results_dir)
    if paper_dir == PAPER.resolve() and (not allow_real_paper or sidecar["synthetic"] is not False):
        raise InputError("refusing to edit the real paper (synthetic results or no "
                         "--allow-real-paper)")
    has_figure = bool(sidecar.get("figure"))
    auto = paper_dir / "auto"
    if auto.resolve() != results_dir:
        for name in RESULT_FILES:
            if name.startswith("fig_") and not has_figure:
                continue
            if (results_dir / name).is_file():
                shutil.copyfile(results_dir / name, auto / name)
    edits = {}
    for rel, old, new in INTEGRATION:
        if rel not in edits:
            raw = (paper_dir / rel).read_bytes().decode("utf-8")
            edits[rel] = {"crlf": "\r\n" in raw, "text": raw.replace("\r\n", "\n")}
        text = edits[rel]["text"]
        matches = list(re.finditer(_ws(old), text))
        if len(matches) != 1:
            raise InputError("%s: expected one occurrence of %r, found %d"
                             % (rel, old[:60], len(matches)))
        m = matches[0]
        edits[rel]["text"] = text[:m.start()] + new + text[m.end():]
    if not has_figure:
        appendix = edits["compact_protocol.tex"]
        line = "\n\\input{auto/cr_figure_seed_spread}"
        if appendix["text"].count(line) != 1:
            raise InputError("figure input not found once in the appendix")
        appendix["text"] = appendix["text"].replace(line, "")
    main = edits["main_submission.tex"]
    tables = REPLICATION_TABLE.findall(main["text"])
    if len(tables) != 1:
        raise InputError("expected one tab:replication skeleton, found %d" % len(tables))
    main["text"] = REPLICATION_TABLE.sub(lambda _: "\\input{auto/cr_table_replication}\n"
                                         "\\input{auto/cr_table_control}", main["text"], count=1)
    for rel, edit in edits.items():
        left = re.findall(r"\\CR(?:SeedResultsPending|ControlResultsPending|PoolingResultsPending|"
                          r"Cell)\b", edit["text"])
        if left:
            raise InputError("%s: placeholders left after integration: %s" % (rel, left))
        flat = re.sub(r"\s+", " ", edit["text"])
        stale = [phrase for phrase in OBSOLETE_PHRASES if phrase in flat]
        if stale:
            raise InputError("%s: superseded seed-count text left: %s" % (rel, stale))
        text = edit["text"].replace("\n", "\r\n") if edit["crlf"] else edit["text"]
        (paper_dir / rel).write_bytes(text.encode("utf-8"))
    return sorted(edits)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="generate macros, tables, text, TL;DR and figure")
    p.add_argument("--final", required=True, help="cr_stats.py final JSON")
    p.add_argument("--runs", nargs="+", help="probe directories or globs")
    p.add_argument("--inventory", help="cr_stats.py inventory JSON (alternative to --runs)")
    p.add_argument("--md", help="cr_md_regression.py --split test JSON")
    p.add_argument("--budget-cb", help="RANDOM-CB run audit (schema cr_cb_budget_audit_v1)")
    p.add_argument("--budget-i5", default=None,
                   help="I5 delivered-mask audit (default: the committed audit, unless --budget-cb)")
    p.add_argument("--budget-i7", help="I7 matcher bench (real_bench.json)")
    p.add_argument("--budget-i7-cell", default="strict_ep30")
    p.add_argument("--stats-dir", default=str(DEFAULT_STATS),
                   help="directory with p1c_stats.json (Table 1 values for the re-probe check)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE))
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--check", action="store_true", help="fail if the outputs are stale")
    p.add_argument("--update-reviews", action="store_true",
                   help="write Rep* bindings into numeric_reviews.json (real results only)")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="dry runs: accept pre-freeze inputs; outputs must stay outside the repo")
    p = sub.add_parser("cb-audit", help="RANDOM-CB budget audit JSON from its [CB] log lines")
    p.add_argument("--log", nargs="+", required=True, help="RANDOM-CB trainer log file(s)")
    p.add_argument("--out", required=True)
    p.add_argument("--min-ramp", type=float, default=1.0)
    p = sub.add_parser("integrate", help="insert the generated inputs into a paper copy")
    p.add_argument("--paper-dir", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--allow-real-paper", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "cb-audit":
            audit = cb_audit_from_log(args.log, args.min_ramp)
            Path(args.out).write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8",
                                      newline="\n")
            print("RANDOM-CB audit: %d full-ramp batches -> %s" % (audit["batches"], args.out))
            return 0
        if args.command == "integrate":
            changed = integrate(args.paper_dir, args.results_dir, args.allow_real_paper)
            print("integrated:", ", ".join(changed))
            return 0
        budgets = [p for p in (args.budget_cb, args.budget_i7) if p]
        i5 = args.budget_i5 or (str(DEFAULT_I5) if not args.budget_cb and DEFAULT_I5.is_file()
                                else None)
        if i5:
            budgets.append(i5)
        result = build(args.final, run_dirs_from(args.runs, args.inventory), args.md, budgets,
                       args.stats_dir, args.out_dir, args.evidence_dir, args.allow_synthetic,
                       args.budget_i7_cell, not args.no_figure, args.check, args.update_reviews)
    except (InputError, cr_stats.IntegrityError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    if args.check:
        print("RESULT:", "PASS" if not result["stale"] else "FAIL (stale: %s)" % result["stale"])
        for line in result["warnings"]:
            print("WARNING:", line)
        return 0 if not result["stale"] else 1
    s = result["scenario"]
    print("scenario %s (centroid=%s, envelope=%s); matched control %s"
          % (s["scenario"], s["labels"]["centroid"], s["labels"]["envelope"],
             s["matched_control"]["label"]))
    print("wrote %d numeric and %d text macros to %s" % (len(result["macros"]),
                                                          len(result["text_macros"]), args.out_dir))
    for line in result["warnings"]:
        print("WARNING:", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

