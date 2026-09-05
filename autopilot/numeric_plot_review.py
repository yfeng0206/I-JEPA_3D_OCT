"""CPU-only artist audit of known local producers, without publishing assets.

The renderer is NOT the numerical oracle: artist coordinates and annotations
are compared with independently selected source fields first. Its PNG is then
rendered in memory and required to equal the delivered PNG byte for byte.
No hash-only receipt or asserted producer-success flag can discharge a figure.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
from unittest.mock import patch

try:
    from . import numeric_bindings as n
    from . import release_assets as assets
except ImportError:
    import numeric_bindings as n
    import release_assets as assets


def equal(actual, expected, label):
    import numpy as np
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != e.shape or not np.allclose(a, e, rtol=0, atol=1e-12):
        raise ValueError("plotted values differ from source: %s; observed=%s; expected=%s" %
                         (label, a.reshape(-1)[:8].tolist(), e.reshape(-1)[:8].tolist()))


def geometry_artists(fig, evidence):
    fields = ("hidden_frac_of_grid", "ctx_frac_of_grid", "n_slots_mean", "hidden_share_of_all_anat")
    arms = ("random", "oracle", "envelope", "cover", "anatomy")
    labels = ("random", "oracle", "envelope", "cover-f0.21", "anatomy-v2")
    if len(fig.axes) != 4:
        raise ValueError("geometry panel coverage requires four axes")
    records = []
    for ax, field in zip(fig.axes, fields):
        expressions = [n.ref("geometry42", arm, field) for arm in arms]
        if field.endswith("frac_of_grid"):
            expressions = [n.operation("percent", x) for x in expressions]
        expected = [evidence.evaluate(x) for x in expressions]
        actual = [bar.get_height() for bar in ax.patches]
        equal(actual, expected, field)
        equal([bar.get_x() + bar.get_width() / 2 for bar in ax.patches], range(5), field + " arm placement")
        if [t.get_text() for t in ax.get_xticklabels()] != list(labels):
            raise ValueError("geometry arm labels changed")
        if [t.get_text() for t in ax.texts] != ["%.1f" % v for v in expected]:
            raise ValueError("geometry numeric annotations differ from source")
        if ax.get_ylim()[0] != 0 or ax.lines or ax.collections:
            raise ValueError("uncovered geometry artist or nonzero bar baseline")
        records.append({"field": field, "expressions": expressions, "observed": actual})
    meta = evidence.load("geometry42")["_meta"]
    footer = ("600 FairVision Training slices (24 volumes x 25), 16x16 grid, "
              "seed 42, COVER floor f=0.21; means, no interval drawn.")
    if [t.get_text() for t in fig.texts] != [footer]:
        raise ValueError("geometry protocol footer requires renewed review")
    if (meta["slices"] != 600 or meta["seed"] != 42 or meta["cover_floor"] != .21 or meta["batch_size"] != 1
            or meta["volumes"] != 24 or meta["slices_per_volume"] != 25 or meta["grid"] != "16x16"):
        raise ValueError("geometry protocol footer differs from source configuration")
    return records


def mask_statistics_artists(fig, evidence):
    source = "composition"
    comp = evidence.load(source)
    rows = sorted(enumerate(comp["rows"]), key=lambda item: item[1]["pct_anat_hid"])
    names = {"random": "random", "oracle": "centroid", "envelope": "envelope",
             "cover_f021": "COVER $f{=}.21$", "blob": "anatomy-v2"}
    fields = ("pct_anat_hid", "pct_tgt_anat", "pct_ctx_anat", "ctx_anat", "ctx", "zero_pct")
    if len(fig.axes) != len(fields):
        raise ValueError("mask-statistics panel coverage differs")
    records = []
    for ax, field in zip(fig.axes, fields):
        expressions = [n.ref(source, "rows", i, field) for i, _ in rows]
        expected = [evidence.evaluate(x) for x in expressions]
        actual = [bar.get_height() for bar in ax.patches]
        equal(actual, expected, "mask statistics " + field)
        if ax.get_ylim()[0] != 0 or ax.collections or ax.lines:
            raise ValueError("mask-statistics baseline/artist coverage differs")
        if [t.get_text() for t in ax.get_xticklabels()] != [names[r["arm"]] for _, r in rows]:
            raise ValueError("mask-statistics arm labels differ")
        if [t.get_text() for t in ax.texts] != ["%.1f" % x for x in expected]:
            raise ValueError("mask-statistics annotations differ")
        records.append({"field": field, "expressions": expressions, "observed": actual})
    if comp["floor_curve"]["0.21"]["n"] != 6137 or next(r for _, r in rows if r["arm"] == "blob")["src"] != "arm_stats(n=1534)":
        raise ValueError("mask-statistics sample-size annotation differs from source")
    expected = ("Masking statistics by arm, ordered by anatomy placed in targets "
                "(anatomy-v2 measured on $n{=}1{,}534$; others on the 6,137-slice sweep)")
    if [t.get_text() for t in fig.texts] != [expected]:
        raise ValueError("mask-statistics sample-size annotation changed")
    return records


def p8_artists(fig, name, evidence):
    import numpy as np
    from matplotlib.container import ErrorbarContainer
    records = []

    def values(expression):
        return evidence.evaluate(expression)

    def record(expressions, observed, label):
        expected = [values(x) for x in expressions]
        equal(observed, expected, label)
        records.append({"semantic_key": label, "expressions": expressions, "observed": np.asarray(observed).tolist()})
        return expected

    def errorbar(container, x, y, lo, hi, label):
        line, caps, bars = container.lines
        equal(line.get_xdata(), x, label + " x")
        equal(line.get_ydata(), y, label + " y")
        if len(bars) != 1 or len(caps) != 2:
            raise ValueError("missing/uncovered error bars: " + label)
        equal(bars[0].get_segments(), [[[a, b], [a, c]] for a, b, c in zip(x, lo, hi)], label + " CI")
        equal(caps[0].get_ydata(), lo, label + " lower caps")
        equal(caps[1].get_ydata(), hi, label + " upper caps")

    stats = evidence.load("p1c_stats.json")
    if name == "fig_labeleff.png":
        if len(fig.axes) != 1:
            raise ValueError("label-efficiency axis missing")
        ax = fig.axes[0]
        if len(ax.lines) != 4 or len(ax.collections) != 4 or ax.patches or ax.get_xscale() != "log":
            raise ValueError("label-efficiency series coverage differs")
        for i, (arm, label) in enumerate((("random", "random"), ("intensity", "centroid"), ("envelope", "envelope"), ("cover", "cover"))):
            source = "p5_label_efficiency.json"
            rows = evidence.load(source)["arms"][arm]
            keys = sorted(rows, key=float)
            fractions = evidence.load(source)["fractions"]
            expressions = [n.operation("percent", n.ref(source, "fractions", fractions.index(float(k)))) for k in keys]
            x = record(expressions, ax.lines[i].get_xdata(), arm + " label fraction")
            y = record([n.ref(source, "arms", arm, k, "auc_mean") for k in keys], ax.lines[i].get_ydata(), arm + " mean AUC")
            sd = np.array([values(n.ref(source, "arms", arm, k, "auc_sd")) for k in keys])
            lo, hi = np.asarray(y) - sd, np.asarray(y) + sd
            expected = [[x[0], hi[0]]] + list(map(list, zip(x, lo))) + [[x[-1], hi[-1]]] + list(map(list, zip(x[::-1], hi[::-1]))) + [[x[0], hi[0]]]
            paths = ax.collections[i].get_paths()
            if len(paths) != 1:
                raise ValueError("label-efficiency band coverage differs")
            equal(paths[0].vertices, expected, arm + " SD band")
            records.append({"semantic_key": arm + " SD", "expressions": [n.ref(source, "arms", arm, k, "auc_sd") for k in keys],
                            "observed": sd.tolist()})
            if ax.lines[i].get_label() != label:
                raise ValueError("label-efficiency arm label changed")
        equal([float(t.get_text()) for t in ax.get_xticklabels()], [1, 5, 10, 25, 100], "label fraction ticks")
        if ax.texts or fig.texts:
            raise ValueError("uncovered label-efficiency annotation")
    elif name == "fig_trajectories_ci.png":
        if len(fig.axes) != 2:
            raise ValueError("trajectory axes missing")
        table = stats["table"]
        ix = {r["key"]: i for i, r in enumerate(table)}
        anc = ix["ancestor@ep25@fp32"]
        ax = fig.axes[0]
        if len(ax.lines) != 7 or ax.collections or ax.patches:
            raise ValueError("trajectory series coverage differs")
        record([n.ref("p1c_stats.json", "table", anc, "auc")], ax.lines[0].get_ydata(), "ancestor AUC")
        record([n.ref("p1c_stats.json", "table", anc, "epoch")], ax.lines[0].get_xdata(), "ancestor epoch")
        if len(ax.texts) != 1 or ax.texts[0].get_text() != "shared ancestor (ep25)":
            raise ValueError("uncovered trajectory annotation")
        equal(ax.texts[0].xy, [25, table[anc]["auc"]], "ancestor annotation position")
        for line, arm in zip(ax.lines[1:], ("random", "oracle", "envelope", "anatomy-v2", "cover-f021", "anatomy-v1")):
            rows = sorted([(i, r) for i, r in enumerate(table) if r["arm"] == arm
                           and (arm not in ("random", "oracle", "envelope") or r["precision"] == "fp16")], key=lambda pair: pair[1]["epoch"])
            ids = [anc] + [i for i, r in rows]
            record([n.ref("p1c_stats.json", "table", i, "epoch") for i in ids], line.get_xdata(), arm + " epochs")
            record([n.ref("p1c_stats.json", "table", i, "auc") for i in ids], line.get_ydata(), arm + " AUC")
        ax = fig.axes[1]
        if ax.get_title() != "(b) Paired difference, 95%% bootstrap CI\n(%s draws, same test cases)" % stats["n_bootstrap"]:
            raise ValueError("trajectory resampling annotation differs")
        if ax.texts or fig.texts:
            raise ValueError("uncovered paired-trajectory annotation")
        containers = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
        if len(containers) != 3:
            raise ValueError("trajectory paired-delta series missing")
        for c, arm, off in zip(containers, ("envelope", "oracle", "cover-f021"), (-.30, 0, .30)):
            expected, lower, upper, x = [], [], [], []
            for j, ep in enumerate((50, 75, 100)):
                precision = "fp32" if arm == "cover-f021" else "fp16"
                a, b = "%s@ep%d@%s" % (arm, ep, precision), "random@ep%d@%s" % (ep, precision)
                matches = [(i, r) for i, r in enumerate(stats["contrasts"]) if r["a"] == a and r["b"] == b]
                if len(matches) != 1:
                    raise ValueError("matched-precision paired source unavailable: " + a)
                i, row = matches[0]
                expected.append(row["delta_a_minus_b"])
                lower.append(row["boot_ci95_lo"])
                upper.append(row["boot_ci95_hi"])
                x.append(j + off)
                records.append({"semantic_key": a + " minus " + b, "source": "p1c_stats.json", "pointer": n.ptr("contrasts", i)})
            errorbar(c, x, expected, lower, upper, arm + " matched-precision delta")
    elif name == "fig_fairness.png":
        if len(fig.axes) != 2:
            raise ValueError("fairness axes missing")
        src, fair = "p7_fairness.json", evidence.load("p7_fairness.json")
        ax = fig.axes[0]
        if ax.get_title() != ("Race-stratified AUC at epoch 100\n"
                              "vertical bars: 95%% percentile bootstrap CI (%s resamples)") % "{:,}".format(fair["n_bootstrap"]):
            raise ValueError("fairness sample-size annotation differs")
        containers = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
        if len(containers) != 3:
            raise ValueError("fairness subgroup series missing")
        for i, (g, c) in enumerate(zip(("White", "Black", "Asian"), containers)):
            keys = ("random@ep100@fp16", "envelope@ep100@fp16", "oracle@ep100@fp16")
            expressions = [n.ref(src, "arms", k, "groups", "race", "per_group", g, "auc") for k in keys]
            y = record(expressions, c.lines[0].get_ydata(), g + " subgroup AUC")
            lower = [values(n.ref(src, "arms", k, "groups", "race", "per_group", g, "auc_ci95_lo")) for k in keys]
            upper = [values(n.ref(src, "arms", k, "groups", "race", "per_group", g, "auc_ci95_hi")) for k in keys]
            errorbar(c, np.arange(3) + (i - 1) * .25, y, lower, upper, g)
            expected_label = "%s (n=%d)" % (g, values(n.ref(src, "arms", keys[0], "groups", "race", "per_group", g, "n")))
            if c.get_label() != expected_label:
                raise ValueError("fairness group count label changed")
            records.append({"semantic_key": g + " intervals and count", "source": src,
                            "pointers": [n.ptr("arms", k, "groups", "race", "per_group", g) for k in keys]})
        ax = fig.axes[1]
        keys = [k for k, r in fair["arms"].items() if r["groups"]["race"]["summary"]]
        if len(ax.collections) != len(keys) or len(ax.lines) != 1:
            raise ValueError("uncovered fairness scatter/fit")
        xs, ys = [], []
        for artist, k in zip(ax.collections, keys):
            expressions = [n.ref(src, "arms", k, "overall_auc"), n.ref(src, "arms", k, "groups", "race", "summary", "auc_gap")]
            expected = record(expressions, artist.get_offsets()[0], k + " race gap scatter")
            xs.append(expected[0])
            ys.append(expected[1])
        x = np.linspace(min(xs), max(xs), 50)
        b = np.sum((np.array(xs) - np.mean(xs)) * (np.array(ys) - np.mean(ys))) / np.sum((np.array(xs) - np.mean(xs)) ** 2)
        a = np.mean(ys) - b * np.mean(xs)
        equal(ax.lines[0].get_xdata(), x, "fairness fit x")
        equal(ax.lines[0].get_ydata(), a + b * x, "fairness least-squares fit")
        rho, p = evidence.load("p7_gap_correlation.json")["spearman_auc_vs_racegap"]
        if ax.get_title() != r"Gap widens as AUC improves ($\rho=%+.2f$, $p=%.3f$)" % (rho, p):
            raise ValueError("fairness correlation annotation changed")
        if any(a.texts for a in fig.axes) or fig.texts:
            raise ValueError("uncovered fairness annotation")
    elif name == "fig_roc.png":
        from sklearn.metrics import roc_curve
        if len(fig.axes) != 1 or len(fig.axes[0].lines) != 4:
            raise ValueError("ROC series coverage differs")
        ax = fig.axes[0]
        if ax.get_title() != "ROC at epoch 100 (N=%d)" % stats["n_test"] or ax.texts or fig.texts:
            raise ValueError("ROC sample-size annotation differs or uncovered text")
        for line, arm, label in zip(ax.lines[:3], ("random", "envelope", "oracle"), ("random", "envelope", "centroid")):
            candidates = [(i, r) for i, r in enumerate(evidence.load("p1b_full_inventory.json")["records"])
                          if r["arm"] == arm and r["epoch"] == 100 and r["family"] == "frozen_probe" and r["precision"] == "fp16"]
            if len(candidates) != 1:
                raise ValueError("ambiguous ROC source")
            i, row = candidates[0]
            path = Path(row["path"])
            name = "prediction:" + arm
            evidence.paths[name], evidence.hashes[name] = path, assets.sha256(path)
            with np.load(path, allow_pickle=False) as data:
                fpr, tpr, _ = roc_curve(data["labels"].astype(int), data["probs"].astype(float))
            equal(line.get_xdata(), fpr, arm + " ROC false-positive rates")
            equal(line.get_ydata(), tpr, arm + " ROC true-positive rates")
            auc = next(r["auc"] for r in stats["table"] if r["key"] == arm + "@ep100@fp16")
            if line.get_label() != "%s  AUC %.3f" % (label, auc):
                raise ValueError("ROC AUC label differs from source")
            records.append({"semantic_key": arm + " ROC", "inventory_pointer": n.ptr("records", i, "path"),
                            "prediction_sha256": evidence.hashes[name], "operation": "ROC from labels/probs; sklearn drop_intermediate=True",
                            "points": len(fpr)})
        equal(ax.lines[-1].get_xdata(), [0, 1], "ROC identity reference x")
        equal(ax.lines[-1].get_ydata(), [0, 1], "ROC identity reference y")
    else:
        raise ValueError("unsupported P8 figure")
    return records


def ladder_artists(fig, evidence, bindings):
    if len(fig.axes) != 1:
        raise ValueError("specificity ladder requires one axis")
    ax = fig.axes[0]
    rows = []
    for arm, label, macro in (
        ("random", "random (null)", None), ("oracle", "centroid", "DOracleRandomEpFifty"),
        ("cover", "cover f=0.21", "DCoverRandomEpFifty"), ("envelope", "envelope", "DEnvelopeRandomEpFifty"),
        ("anatomy", "anatomy-v2", "DAnatomyTwoRandomEpFifty")):
        hidden = evidence.evaluate(n.ref("geometry42", arm, "hidden_share_of_all_anat"))
        if macro:
            d = float(bindings[macro]["expected"])
            lo, hi = [float(m[0]) for m in n.NUMBER.finditer(bindings[macro + "CI"]["expected"])]
        else:
            d, lo, hi = 0, None, None
        rows.append((hidden, arm, label, macro, d, lo, hi))
    rows.sort()
    expected_lines = [([0, 0], [0, 1])]
    labels, annotations, records = [], [], []
    for y, (hidden, arm, label, macro, d, lo, hi) in enumerate(rows):
        labels.append("%s\n%.1f%% anatomy hidden" % (label, hidden))
        if macro is None:
            expected_lines.append(([0], [y]))
        else:
            expected_lines.extend([([lo, hi], [y, y]), ([lo, lo], [y - .14, y + .14]),
                                   ([hi, hi], [y - .14, y + .14]), ([d], [y])])
            annotations.append("%+.4f" % d)
            records.append({"arm": arm, "delta_binding": bindings[macro],
                            "interval_binding": bindings[macro + "CI"], "observed": [d, lo, hi],
                            "geometry": n.ref("geometry42", arm, "hidden_share_of_all_anat")})
    if len(ax.lines) != len(expected_lines) or ax.collections or ax.patches:
        raise ValueError("uncovered or missing specificity ladder artists")
    for i, (line, (x, y)) in enumerate(zip(ax.lines, expected_lines)):
        equal(line.get_xdata(), x, "ladder line %d x" % i)
        equal(line.get_ydata(), y, "ladder line %d y" % i)
    if [t.get_text() for t in ax.get_yticklabels()] != labels or [t.get_text() for t in ax.texts] != annotations:
        raise ValueError("ladder numeric or arm annotations differ from source")
    return records


def _load_scatter_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_replacement_scatter(paper, evidence, item):
    """Replay only the registered public-source scatter, never private masks."""
    import matplotlib
    import matplotlib.pyplot as plt

    base = assets.REPO / "autopilot" / "investigations" / "delivered_task" / "evidence" / "legacy_figure_reviews"
    manifest_path = base / "replacements" / "source_manifest.json"
    fig = None
    try:
        manifest_hash = assets.sha256(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        producer_path = base / "generate_replacements.py"
        validator_path = base / "verify_replacements.py"
        dependencies = {
            "replacement_scatter_manifest": (manifest_path, manifest_hash),
            "replacement_scatter_producer": (producer_path, manifest["generator"]["sha256"]),
            "replacement_scatter_validator": (validator_path, manifest["independent_validator"]["sha256"]),
        }
        if (assets.safe_path(assets.REPO, manifest["generator"]["path"]) != producer_path.resolve()
                or assets.safe_path(assets.REPO, manifest["independent_validator"]["path"]) != validator_path.resolve()):
            raise ValueError("Unregistered replacement scatter implementation path")
        for name, (path, digest) in dependencies.items():
            spec = {"root": "repo", "path": str(path.relative_to(assets.REPO)), "sha256": digest}
            if name in evidence.specs and evidence.specs[name] != spec:
                raise ValueError("Replacement replay dependency conflicts with approved evidence")
            evidence.specs[name] = spec
            evidence.load(name)
        for name in ("geometry42", "p1b_full_inventory.json"):
            evidence.load(name)
            declared = manifest["public_numeric_sources"][name]
            if evidence.hashes[name] != declared["sha256"]:
                raise ValueError("Replacement scatter source changed: " + name)
            path = assets.safe_path(evidence.roots[declared["root"]], declared["path"])
            if path != evidence.paths[name].resolve():
                raise ValueError("Replacement scatter source locator differs: " + name)
        target = assets.safe_path(paper, item["path"])
        expected_png = manifest["outputs"]["fig_purity_auc_ep50_fp32"]["png"]["sha256"]
        if assets.sha256(target) != item["sha256"] or item["sha256"] != expected_png:
            raise ValueError("Delivered replacement PNG is not the registered source-linked export")
        with matplotlib.rc_context(), contextlib.redirect_stdout(io.StringIO()):
            producer = _load_scatter_module(producer_path, "numeric_replacement_scatter_producer")
            verifier = _load_scatter_module(validator_path, "numeric_replacement_scatter_validator")
            geometry, inventory = evidence.paths["geometry42"], evidence.paths["p1b_full_inventory.json"]
            fig = producer.build_scatter(geometry, inventory)
            # The separate verifier reads named JSON records independently;
            # neither its expected values nor these checks use manifest means.
            details = verifier.verify_scatter_artists(fig, geometry, inventory)
            if details.get("status") != "source_fields_and_all_data_artists_verified" or len(details.get("series", [])) != 5:
                raise ValueError("Replacement scatter did not produce complete artist coverage")
            if (details["geometry_sha256"] != evidence.hashes["geometry42"]
                    or details["inventory_sha256"] != evidence.hashes["p1b_full_inventory.json"]):
                raise ValueError("Scatter verifier consumed different source bytes")
            for row in details["series"]:
                equal(row["observed_xy"], [evidence.evaluate(row["x_expression"]), evidence.evaluate(row["y_expression"])],
                      "replacement source expressions: " + row["arm"])
            # Independent PNG rendering also enforces opacity and exact bytes.
            png = verifier.verify_png(fig, target)
            if not png["exact_byte_equality"] or png["sha256"] != item["sha256"]:
                raise ValueError("Replacement PNG is not the independently checked render")
        for name, (path, digest) in dependencies.items():
            if assets.sha256(path) != digest:
                raise ValueError("Replacement replay implementation changed during audit: " + name)
        return {
            "status": "programmatically_verified_plotted_values", "mathematically_verified": True,
            "validation": {
                "method": "registered_independent_scatter_artist_and_png_replay",
                "producer": str(producer_path.relative_to(assets.REPO)),
                "producer_sha256": dependencies["replacement_scatter_producer"][1],
                "independent_validator": str(validator_path.relative_to(assets.REPO)),
                "independent_validator_sha256": dependencies["replacement_scatter_validator"][1],
                "manifest_sha256": manifest_hash, "rendered_sha256": png["sha256"],
                "observed_series": details["series"], "artist_checks": details, "png_checks": png,
            },
            "inputs": [{"source": name, "path": str(evidence.paths[name]), "sha256": evidence.hashes[name]}
                       for name in ("geometry42", "p1b_full_inventory.json")],
            "scope": "Five exact primary frozen-probe epoch50 fp32 points, semantic labels and delivered PNG; no causal or retraining validation.",
        }
    except (OSError, ValueError, KeyError, TypeError, AssertionError, ImportError, AttributeError, SyntaxError) as exc:
        message = str(exc)
        return {"status": "mismatch" if message.startswith(("Source/artist mismatch", "plotted values differ")) else "unresolved",
                "action": "Registered replacement scatter replay: " + message}
    finally:
        if fig is not None:
            plt.close(fig)


def verify_local_plots(paper, evidence, bindings, items):
    """Registered safe producers only; all output bytes remain in memory."""
    selected = {i["path"]: i for i in items if i["path"] in {
        "figures/fig_geometry_panel.png", "figures/fig_specificity_ladder.png", "figures/figS5_mask_statistics.png"}}
    p8_selected = {i["path"]: i for i in items if i["path"] in {
        "auto/fig_labeleff.png", "auto/fig_trajectories_ci.png", "auto/fig_fairness.png", "auto/fig_roc.png"}}
    replacements = {i["path"]: i for i in items if i["path"] == "figures/fig_purity_auc_ep50_fp32.png"}
    if not selected and not p8_selected and not replacements:
        return {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    result = {}
    for rel, item in replacements.items():
        result[rel] = verify_replacement_scatter(paper, evidence, item)
    if p8_selected:
        script = assets.REPO / "autopilot" / "p8_make_assets.py"
        script_hash = assets.sha256(script)
        work = assets.unique_work(prefix="numeric-p8")
        save = Figure.savefig

        def capture_p8(fig, destination, *args, **kwargs):
            rel = "auto/" + Path(destination).name
            if rel not in p8_selected:
                raise ValueError("unregistered P8 output figure: " + rel)
            try:
                records = p8_artists(fig, Path(destination).name, evidence)
                stream = io.BytesIO()
                save(fig, stream, *args, format="png", **kwargs)
                rendered = hashlib.sha256(stream.getvalue()).hexdigest()
                if rendered != p8_selected[rel]["sha256"]:
                    raise ValueError("artist values verified but delivered raster differs from audited replay")
                result[rel] = {"status": "programmatically_verified_plotted_values", "mathematically_verified": True,
                    "validation": {"method": "independent_matplotlib_artist_replay", "producer": "autopilot/p8_make_assets.py",
                                   "producer_sha256": script_hash, "rendered_sha256": rendered, "observed_series": records},
                    "scope": "enumerated data artists, intervals and labels; not causal/statistical validity"}
            except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
                result[rel] = {"status": "mismatch" if str(exc).startswith("plotted values differ") else "unresolved", "action": str(exc)}
        try:
            spec = importlib.util.spec_from_file_location("numeric_p8_candidate", script)
            module = importlib.util.module_from_spec(spec)
            with matplotlib.rc_context(), contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(module)
                module.AUTO = str(work)
                module.STATS = str(evidence.roots["stats"])
                module.SUBGROUP_ADJUSTED = str(evidence.paths["p17_subgroup_multiplicity.json"])
                with patch.object(Figure, "savefig", capture_p8):
                    module.main()
            if assets.sha256(script) != script_hash:
                raise ValueError("P8 producer changed during numeric audit")
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
            for rel in p8_selected:
                result[rel] = {"status": "unresolved", "action": "independent P8 replay failed: " + str(exc)}
        finally:
            plt.close("all")
            shutil.rmtree(work)
        for rel in p8_selected:
            result.setdefault(rel, {"status": "unresolved", "action": "registered P8 output not produced"})
    for rel, item in selected.items():
        stem = Path(rel).stem
        script = (assets.REPO / "paper" / "genai4health2026" / "scripts" / "make_story_figures.py"
                  if stem == "figS5_mask_statistics" else assets.REPO / "autopilot" / ("make_" + stem + ".py"))
        script_hash = assets.sha256(script)
        work = assets.unique_work(prefix="numeric-plot")
        captured = {}
        save = Figure.savefig

        def intercept(fig, destination, *args, **kwargs):
            if Path(destination).suffix != ".png":
                return
            if stem == "fig_geometry_panel":
                records = geometry_artists(fig, evidence)
            elif stem == "figS5_mask_statistics":
                records = mask_statistics_artists(fig, evidence)
            else:
                records = ladder_artists(fig, evidence, bindings)
            stream = io.BytesIO()
            save(fig, stream, *args, format="png", **kwargs)
            rendered_hash = hashlib.sha256(stream.getvalue()).hexdigest()
            if rendered_hash != item["sha256"]:
                raise ValueError("artist values verified but current raster is not the independently audited render")
            captured.update(status="programmatically_verified_plotted_values", mathematically_verified=True,
                            validation={"method": "independent_matplotlib_artist_replay",
                                        "producer": str(script.relative_to(assets.REPO)), "producer_sha256": script_hash,
                                        "rendered_sha256": rendered_hash, "observed_series": records},
                            scope="all registered producer data artists and numeric annotations; not causal or statistical validity")
        try:
            spec = importlib.util.spec_from_file_location("numeric_figure_candidate", script)
            module = importlib.util.module_from_spec(spec)
            with matplotlib.rc_context(), contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(module)
                module.OUT = str(work / stem)
                evidence.load("geometry42")
                if stem == "fig_geometry_panel":
                    module.ART = str(evidence.paths["geometry42"])
                elif stem == "figS5_mask_statistics":
                    evidence.load("composition")
                    module.COMPOSITION = evidence.paths["composition"]
                    module.FIGURES = work
                else:
                    module.GEOM = str(evidence.paths["geometry42"])
                    module.AUTO = str(Path(paper) / "auto" / "auto_numbers.tex")
                with patch.object(Figure, "savefig", intercept):
                    if stem == "figS5_mask_statistics":
                        module.fig_mask_stats()
                    else:
                        module.main()
            if not captured:
                raise ValueError("producer did not deliver the required registered figure")
            if assets.sha256(script) != script_hash:
                raise ValueError("figure producer changed during numeric audit")
            result[rel] = {**captured, "inputs": [
                {"source": name, "sha256": digest, "path": str(evidence.paths[name])}
                for name, digest in evidence.hashes.items() if name in ("geometry42", "p1c_stats.json", "composition")]}
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
            result[rel] = {"status": "unresolved", "action": str(exc)}
        finally:
            plt.close("all")
            work.rmdir()
    return result
