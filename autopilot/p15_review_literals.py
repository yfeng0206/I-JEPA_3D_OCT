"""Build explicit, source-reviewed literal receipts; never edit the manuscript.

The queue is the retained 61d audited context inventory. Every assignment below
is an explicit reviewed context/token, exact source field, or checked inequality.
Unassigned/contradicted claims remain blockers. No number-wide approval exists.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import yaml

try:
    from . import numeric_bindings as n, release_assets as assets
except ImportError:
    import numeric_bindings as n
    import release_assets as assets

ROOT = assets.REPO
PHASE = Path(r"D:\jepa_phase0")
STATS = PHASE / "autopilot_out" / "p1_stats"
OUT = ROOT / "autopilot" / "investigations" / "delivered_task" / "evidence"
SOURCES = OUT / "literal_sources"
REVIEWER = "Numeric evidence agent; direct source/primary-publication review, 2026-09-04"


class Reviews:
    def __init__(self):
        self.queue = json.loads((OUT / "literal_review_queue.json").read_text())
        self.document = json.loads((OUT / "delivered_release_reviews.json").read_text())
        self.document.setdefault("literals", [])
        self.sources = self.document["sources"]
        self.origins, self.data, self.assignments, self.findings = {}, {}, {}, []
        self.evidence = n.Evidence(assets.PAPER, STATS, self.sources)

    def source(self, name, path):
        path = Path(path)
        digest = assets.sha256(path)
        original = str(path)
        if path.is_relative_to(ROOT):
            spec = {"root": "repo", "path": str(path.relative_to(ROOT)), "sha256": digest}
        elif path.is_relative_to(STATS):
            spec = {"root": "stats", "path": str(path.relative_to(STATS)), "sha256": digest}
        else:
            target = SOURCES / ("retained_" + name + path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if assets.sha256(target) != digest:
                raise ValueError("retained source copy differs")
            spec = {"root": "repo", "path": str(target.relative_to(ROOT)), "sha256": digest}
        self.sources[name] = spec
        self.evidence.specs[name] = spec
        self.origins[name] = {"original": original, "sha256": digest, "retained": spec}
        if path.suffix == ".json":
            self.data[name] = json.loads(path.read_text(encoding="utf-8-sig"))
        elif path.suffix in (".yaml", ".yml"):
            self.data[name] = yaml.safe_load(path.read_text())
        return name

    def token(self, group, index):
        matches = [t for t in self.queue[group]["tokens"] if t["index"] == index]
        if len(matches) != 1:
            raise ValueError("unlisted/ambiguous reviewed token: %d/%d" % (group, index))
        return matches[0]

    def entry(self, group, index):
        context, token = self.queue[group], self.token(group, index)
        return {"file": context["file"], "context_sha256": context["context_sha256"],
                "token_index": index, "value": token["value"]}

    def put(self, group, index, **fields):
        key = (group, index)
        if key in self.assignments:
            raise ValueError("duplicate reviewer assignment: " + str(key))
        entry = {**self.entry(group, index), **fields}
        self.assignments[key] = entry
        try:
            result = n.review_literal({"value": entry["value"]}, entry, self.evidence)
            if result["status"] == "mismatch":
                self.findings.append({"group": group, "index": index, "claim": self.queue[group]["context"],
                                      "value": entry["value"], "result": result})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.findings.append({"group": group, "index": index, "value": entry["value"], "error": str(exc)})
        return entry

    def field(self, group, index, source, *path, pattern="%.4f"):
        return self.put(group, index, expression=n.fmt(pattern, n.ref(source, *path)))

    def expression(self, group, index, expression, component=None):
        return self.put(group, index, expression=expression, **({} if component is None else {"component": component}))

    def assertion(self, group, index, expression, relation, bound, component=0):
        return self.put(group, index, assertion={"expression": expression, "relation": relation,
                                                "bound": bound, "component": component})

    def review(self, group, indices, expected, evidence, rationale, kind="protocol", **extra):
        if len(indices) != len(expected):
            raise ValueError("review declaration length mismatch")
        for index, value in zip(indices, expected):
            if self.token(group, index)["value"] != str(value):
                raise ValueError("reviewed declaration changed: %d/%d" % (group, index))
            self.put(group, index, review={"kind": kind, "reviewer": REVIEWER, "rationale": rationale,
                                           "evidence": evidence, **extra})

    def epoch(self, group, index, arm, epoch, precision=None, status=None):
        if status:
            rows = self.data["lr_inventory"]["records"]
            selected = [i for i, r in enumerate(rows) if r.get("arm") == arm and r.get("epoch") == epoch and r.get("status") == status]
            source, path = "lr_inventory", ("records",)
        else:
            rows = self.data["lr_stats"]["table"]
            precision = precision or ("fp32" if arm in ("ancestor", "anatomy-v1", "anatomy-v2", "cover-f021") else "fp16")
            selected = [i for i, r in enumerate(rows) if r["key"] == f"{arm}@ep{epoch}@{precision}"]
            source, path = "lr_stats", ("table",)
        if len(selected) != 1:
            raise ValueError("epoch lookup is ambiguous or missing: " + str((arm, epoch, precision, status)))
        value = self.token(group, index)["value"]
        if abs(float(value)) != epoch:
            raise ValueError("epoch token disagrees with named source")
        self.review(group, [index], [value], [n.ref(source, *path, selected[0])],
                    "Reviewed checkpoint/analysis epoch descriptor for %s, epoch %d%s. A leading hyphen belongs to 'epoch-N', not a negative measurement." %
                    (arm, epoch, "; " + status if status else ""))

    def hold(self, group, indices, reason):
        for index in indices:
            self.findings.append({"group": group, "index": index, "value": self.token(group, index)["value"],
                                  "claim": self.queue[group]["context"], "blocked_reason": reason})


def register(r):
    repo = {
        "lr_cover_cfg": r"configs\patch_cover_f021_ep25.yaml",
        "lr_anatomy_cfg": r"configs\patch_anatomy_v2.yaml",
        "lr_v1_cfg": r"configs\patch_mirage_anatomy.yaml",
        "lr_replication_code": r"scripts\make_replication_configs.py",
        "lr_primary_result": r"results\downstream\meanpool_sweep_random\ep100_results.json",
        "lr_geom": r"results\masking\table2_geometry\mask_geometry_600slices_bs1_coverf021_seed42.json",
        "lr_geom64": r"results\masking\table2_geometry\mask_geometry_600slices_bs64_coverf021_seed42.json",
        "lr_geom015": r"results\masking\table2_geometry\mask_geometry_600slices_bs64_coverf015_seed42.json",
        "lr_geometry_derived": r"autopilot\investigations\delivered_task\evidence\literal_sources\geometry_reduction.json",
        "lr_intersection_derived": r"autopilot\investigations\delivered_task\evidence\literal_sources\intersectional_reduction.json",
        "lr_probability": r"autopilot\investigations\delivered_task\evidence\literal_sources\probability_diagnostics.json",
        "lr_environment": r"autopilot\investigations\delivered_task\evidence\literal_sources\environment_snapshot.json",
        "lr_metadata": r"autopilot\investigations\delivered_task\evidence\literal_sources\metadata_reduction.json",
        "lr_hardware": r"autopilot\investigations\delivered_task\evidence\literal_sources\hardware_snapshot.json",
        "lr_fixed_head": r"autopilot\investigations\delivered_task\evidence\literal_sources\fixed_head_reproduction.json",
        "lr_parameters": r"autopilot\investigations\delivered_task\evidence\literal_sources\probe_parameter_count.json",
        "lr_p1_code": r"autopilot\p1c_stats.py",
        "lr_p17_code": r"autopilot\p17_adjust_subgroup_multiplicity.py",
        "lr_p8_code": r"autopilot\p8_make_assets.py",
        "lr_p5_code": r"autopilot\p5_label_efficiency.py",
        "lr_subgroup_code": r"paper\genai4health2026\scripts\subgroup_analysis.py",
        "lr_intersection_code": r"paper\genai4health2026\scripts\intersectional_analysis.py",
        "lr_fairness_code": r"autopilot\p7_fairness.py",
        "lr_eval_code": r"src\eval_downstream.py",
        "lr_interpretability_code": r"scripts\deeper_interpretability_analysis.py",
        "lr_incremental": r"autopilot\bgsig\a2_region_incremental.json",
        "lr_incremental_code": r"autopilot\bgsig\a2_region_incremental.py",
        "lr_position": r"autopilot\bgsig\a3b_threshold_sweep.json",
        "lr_position_code": r"autopilot\bgsig\a3b_threshold_sweep.py",
        "lr_class_relations": r"results\masking\class_relations\class_relations.json",
        "lr_training": r"src\train_patch.py",
        "lr_lock": r"requirements-phase0.lock.txt",
        "lr_mask_probe": r"scripts\mask_composition_probe.py",
        "lr_geometry_figure": r"autopilot\make_fig_geometry_panel.py",
    }
    for name, path in repo.items():
        r.source(name, ROOT / path)
    for name, file in (("lr_stats", "p1c_stats.json"), ("lr_inventory", "p1b_full_inventory.json"),
                       ("lr_fp32", "p3b_fp32.json"), ("lr_le", "p5_label_efficiency.json"),
                       ("lr_trend", "p7b_gap_trend.json"), ("lr_fairness", "p7_fairness.json"),
                       ("lr_operating", "p8b_operating_points.json"), ("lr_suboperating", "p16_subgroup_operating.json")):
        r.source(name, STATS / file)
    for name, path in (
        ("lr_intersection", r"reports\subgroup\intersectional_auc.json"),
        ("lr_marginal", r"reports\subgroup\subgroup_auc.json"),
        ("lr_arm_stats", r"reports\arm_stats\arm_stats.json"),
        ("lr_floor", r"reports\arm_stats_sweep\cover_floor_sweep.json"),
        ("lr_rectangles", r"reports\cover_random_scale\scale_validation.json"),
    ):
        r.source(name, PHASE / path)


def methods(r):
    # Each tuple identifies one reviewed token and one named checkpoint record.
    checkpoints = {
        3: [(1, "envelope", 50), (2, "oracle", 100), (3, "oracle", 50), (4, "oracle", 75), (5, "oracle", 100)],
        4: [(0, "oracle", 100)],
        12: [(4, "ancestor", 25), (8, "ancestor", 25)],
        13: [(0, "random", 100), (2, "anatomy-v2", 75), (6, "anatomy-v2", 75), (9, "anatomy-v1", 30),
             (11, "cover-f021", 100), (12, "cover-f021", 73), (13, "envelope", 50)],
        14: [(0, "ancestor", 25)], 15: [(1, "random", 50), (3, "random", 75), (5, "random", 100)],
        17: [(3, "envelope", 50), (4, "oracle", 100)], 18: [(1, "ancestor", 25), (4, "cover-f021", 50)],
        19: [(1, "anatomy-v2", 50), (3, "anatomy-v1", 30)],
        20: [(0, "cover-f021", 100), (1, "cover-f021", 50)],
        21: [(0, "oracle", 100), (1, "oracle", 100), (2, "oracle", 50), (3, "oracle", 75)],
        22: [(1, "envelope", 50)], 23: [(0, "envelope", 50)], 24: [(0, "oracle", 100)],
        27: [(0, "random", 100), (7, "random", 50), (14, "random", 100)],
        29: [(0, "oracle", 100)], 30: [(5, "oracle", 100)],
        31: [(0, "envelope", 50), (1, "oracle", 100)],
        32: [(2, "random", 50), (3, "ancestor", 25)],
        33: [(1, "cover-f021", 100), (2, "cover-f021", 50)],
        34: [(3, "cover-f021", 73), (4, "cover-f021", 75), (5, "cover-f021", 100), (7, "anatomy-v2", 75)],
        37: [(0, "ancestor", 25)],
        49: [(0, "oracle", 100)], 54: [(0, "oracle", 100)], 56: [(0, "oracle", 100)],
        57: [(0, "oracle", 100)], 59: [(4, "cover-f021", 50), (5, "cover-f021", 73), (6, "cover-f021", 75),
                                       (7, "cover-f021", 100), (9, "anatomy-v2", 75)],
        60: [(0, "oracle", 100)], 61: [(5, "ancestor", 25)],
        63: [(0, "oracle", 100), (1, "oracle", 50), (3, "envelope", 75), (7, "envelope", 100)],
        66: [(0, "cover-f021", 100)], 72: [(0, "random", 100), (5, "envelope", 100)],
        73: [(0, "random", 50)], 84: [(0, "oracle", 100)], 88: [(2, "oracle", 100)],
        89: [(1, "oracle", 100)], 101: [(1, "random", 50), (2, "random", 75), (3, "random", 100)],
        102: [(1, "oracle", 50)], 103: [(1, "oracle", 50)], 108: [(0, "ancestor", 25)],
    }
    for group, entries in checkpoints.items():
        for index, arm, epoch in entries:
            r.epoch(group, index, arm, epoch)
    for group, index, epoch in ((13, 3, 75), (13, 4, 92), (65, 1, 75), (65, 2, 92)):
        r.epoch(group, index, "anatomy-v2", epoch, status="excluded")
    r.review(13, [7], ["-100"], [{"source": "lr_p8_code", "excerpt": 'NOT_RUN = {"EpHundred"}'}],
             "Epoch100 is explicitly an unmeasured anatomy-v2 horizon, not a measured checkpoint.")
    for group, indices, expected in ((36, [2], ["-50"]), (37, [16, 17], ["50", "100"]),
                                      (38, [0], ["-50"]), (39, [0], ["-50"])):
        r.review(group, indices, expected, [{"source": "lr_replication_code", "excerpt": "optimization.epochs`` stays 100"},
                                           {"source": "lr_replication_code", "excerpt": "--stop_after_epoch 50"}],
                 "These are the predeclared replication endpoint and schedule horizon, not completed replication results.")
    r.review(36, [0, 1], ["-26", "27"], [n.ref("lr_metadata", "replication_iterations")],
             "Retained log has a full9375-iteration epoch26 and only939 distinct epoch27 iterations; no epoch50 probe is implied.")
    # CI levels and the null zero are explicit statistical definitions.
    for group, indices, expected in ((0, [0], ["95"]), (1, [0, 1], ["95", "0"]),
                                     (2, [0, 1], ["95", "0"]), (100, [0], ["95"])):
        source = "lr_p1_code" if group in (0, 100) else "lr_p17_code"
        r.review(group, indices, expected, [{"source": source, "excerpt": "np.percentile"}],
                 "95% interval level and zero as the null of a difference are statistical conventions; source code computes the named percentile/simultaneous intervals.",
                 kind="formula")
    for group, indices in ((57, [1, 2]), (88, [0, 1])):
        r.review(group, indices, ["95", "0.05"],
                 [{"source": "lr_p17_code", "excerpt": "np.percentile(max_abs_t, 95)"},
                  {"source": "lr_p17_code", "excerpt": '"alpha": 0.05'}],
                 "Declared simultaneous interval coverage95% and family alpha0.05; not empirical estimates.", kind="formula")
    r.review(53, [1], ["0.05"], [{"source": "lr_p17_code", "excerpt": '"alpha": 0.05'}],
             "The text explicitly calls0.05 a conventional interpretation threshold, not a measured p-value.", kind="formula")
    # Geometry/architecture declarations are supported by measured metadata,
    # saved probe configuration, and the actual pretraining YAML.
    r.review(5, [0, 1, 2], ["16", "16", "256"], [n.ref("lr_geom", "_meta")],
             "Measured sampler metadata declares grid16x16 and total_patches256.")
    r.review(6, [0], ["4"], [{"source": "lr_cover_cfg", "excerpt": "num_pred_masks: 4"}],
             "Declared target-block count M=4.")
    r.review(8, [15, 16, 17], ["16", "16", "4"], [n.ref("lr_geom", "_meta"),
              {"source": "lr_cover_cfg", "excerpt": "num_pred_masks: 4"}],
             "Pipeline node text states the token grid and requested targets; conceptual coordinates are separately excluded.")
    r.review(9, [46, 47, 51, 55], ["6", "384", "256", "1"],
             [{"source": "lr_cover_cfg", "excerpt": "pred_depth: 6"},
              {"source": "lr_cover_cfg", "excerpt": "pred_emb_dim: 384"}, n.ref("lr_geom", "_meta", "total_patches"),
              {"source": "lr_eval_code", "excerpt": "def __init__(self, in_dim, out_dim=1):"}],
             "Predictor depth/width, full-grid tokens, and one-logit binary head are declared architecture, not performance measurements.")
    r.review(10, [17], ["100"], [n.ref("lr_primary_result", "config", "data", "num_slices")],
             "Frozen primary probe pools100 B-scans per volume.")
    for group, indices, expected in ((11, [2, 3, 4, 8], ["100", "256", "256", "42"]),
                                     (35, [0], ["42"])):
        r.review(group, indices, expected, [n.ref("lr_primary_result", "config")],
                 "Saved primary probe configuration: num_slices100, slice/crop size256, evaluation/probe seed42; not the pretraining seed.")
    r.review(12, [0, 1, 2, 3, 5, 9, 10], ["256", "256", "16", "6", "512", "25", "30"],
             [{"source": "lr_cover_cfg", "excerpt": "batch_size: 64"},
              {"source": "lr_cover_cfg", "excerpt": "accum_steps: 8"},
              {"source": "lr_cover_cfg", "excerpt": "patch_size: 16"},
              {"source": "lr_cover_cfg", "excerpt": "pred_depth: 6"},
              {"source": "lr_cover_cfg", "excerpt": "T_warm: 25"},
              {"source": "lr_cover_cfg", "excerpt": "T_total: 30"},
              {"source": "lr_replication_code", "excerpt": "64 x 4 T4s x 2"}],
             "Reviewed nominal settings: crop256, patch16, depth6, effective64x8=512 (archived equivalent64x4x2), guidance ramp25--30.")
    r.review(12, [7], ["-27"], [{"source": "lr_v1_cfg", "excerpt": r"resume-ep27.pth.tar"}],
             "Epoch27 is the checked-in anatomy-v1 resume filename. This does not establish the missing historical launch, as the text explicitly qualifies.")
    for group, index in ((7, 0), (13, 10), (16, 0), (33, 0), (104, 3)):
        r.review(group, [index], [r.token(group, index)["value"]],
                 [{"source": "lr_cover_cfg", "excerpt": "cover_leave_frac: 0.21"}],
                 "COVER f=0.21 is the declared parameter/artifact identifier, not a measured coverage percentage.")
    r.review(26, [0, 1, 2, 3], ["16", "4", "16", "64"],
             [{"source": "lr_anatomy_cfg", "excerpt": "pred_target_k: 16"},
              {"source": "lr_anatomy_cfg", "excerpt": "num_pred_masks: 4"}],
             "Fixed K=16 and M=4 imply4x16=64 target-loss slots by construction.", kind="formula")
    for group in (68,):
        r.review(group, [0], ["16"], [{"source": "lr_anatomy_cfg", "excerpt": "pred_target_k: 16"}],
                 "Anatomy-family target resampling constant.")
    r.review(41, [3, 4, 5, 6], ["0.21", "16", "16", "0.25"], [n.ref("lr_geom", "_meta")],
             "Measurement metadata explicitly records floor0.21, grid16x16 and occupancy threshold0.25.")
    r.review(45, [1], ["64"], [n.ref("lr_geom64", "_meta", "batch_size")], "Production-batch geometry measurement uses batch64.")
    r.review(46, [0], ["64"], [n.ref("lr_geom64", "_meta", "batch_size")], "Column label names the measured batch64 protocol.")
    r.review(47, [2, 3], ["0.21", "0.15"], [n.ref("lr_geom64", "_meta", "cover_floor"), n.ref("lr_geom015", "_meta"),
             {"source": "lr_mask_probe", "excerpt": 'ap.add_argument("--cover_floor", type=float, default=0.15,'}],
             "Named f021 and f015 measurement settings. The older f015 artifact omits an inline floor field; its retained filename and the producer's explicit default identify the nominal0.15 setting, not a new inferred biological quantity.")
    r.review(48, [1, 2, 5, 6, 7, 8], ["16", "64", "40", "40", "46", "21"],
             [{"source": "lr_anatomy_cfg", "excerpt": "pred_target_k: 16"}, n.ref("lr_geometry_derived", "proposed_target_k_from_mean_slots"),
              n.ref("lr_geometry_derived", "rectangle_mask_percent_min"), n.ref("lr_geometry_derived", "rectangle_mask_percent_max"),
              n.ref("lr_geom", "anatomy", "hidden_frac_of_grid")],
             "This paragraph specifies an unrun control. Current K16 gives64 slots; mean rectangle budget/4 is39.854=>K approximately40. Target area40--46% and anatomy21% are rounded design magnitudes, not an assertion that each rectangle pair is exactly matched.",
             kind="formula")
    r.review(55, [0], ["-2"], [n.ref("lr_metadata", "md_threshold_labels_agree"),
              {"source": "lr_subgroup_code", "excerpt": '(-6, -2, "mild (-6,-2]")'}],
             "Metadata threshold md<=-2 reconstructs every positive label; this is the label/stratum definition.")
    r.review(59, [0], ["2{,}000"], [{"source": "lr_subgroup_code", "excerpt": "def auc_ci(y, p, n_boot=2000, seed=0):"}],
             "Stored intersectional per-arm intervals use the imported2000-draw class-stratified routine.")
    r.review(64, [2], ["40"], [{"source": "lr_intersection_code", "excerpt": 'ap.add_argument("--min-n", type=int, default=40)'}],
             "Explicit intersectional inclusion threshold; different from the separate p7_fairness50/10 rule.")
    r.review(69, [0, 1, 2, 3], ["600", "24", "25", "64"],
             [n.ref("lr_geom", "_meta"), n.ref("lr_geom64", "_meta")],
             "Declared fixed replay scope24x25=600 views and full production batch64; partial tail is separately disclosed.",
             kind="formula")
    r.review(70, [0, 1, 2], ["0.25", "0.21", "4"],
             [{"source": "lr_cover_cfg", "excerpt": "mirage_occupancy_threshold: 0.25"},
              {"source": "lr_cover_cfg", "excerpt": "cover_min_visible_frac: 0.21"},
              {"source": "lr_cover_cfg", "excerpt": "cover_min_visible_cells: 4"}],
             "Guide threshold and requested final-context floor parameters;4 is the minimum visible-cell floor, not a matched empirical outcome.")
    r.review(71, [1], ["1"], [{"source": "lr_training", "excerpt": "smooth_l1_loss"}],
             "The1 belongs to the SmoothL1 objective name, not a measured result.", kind="formula")
    r.review(74, [0], ["0"], [{"source": "lr_position_code", "excerpt": "pos_embed"}],
             "Layer0 denotes the encoder input before transformer blocks, whose content/position variance is explicitly decomposed.", kind="formula")
    r.review(75, [0, 1, 2, 3, 4, 5, 6], ["256", "4", "10", "-4", "0.05", "50", "5"],
             [n.ref("lr_primary_result", "config"), {"source": "lr_p5_code", "excerpt": "epochs=50, lr=4e-4, wd=0.05, batch_size=256, warmup_epochs=5"}],
             "Probe protocol: batch256, LR4x10^-4, decay0.05,50 epochs and5 warmup epochs. These are configured methods, not fitted outputs.",
             kind="formula")
    for group, indices, expected in ((76, [0, 1], ["5", "1"]), (77, [0], ["1"]), (28, [0], ["5"]), (78, [2], ["5"])):
        r.review(group, indices, expected, [n.ref("lr_le", "fractions")],
                 "Explicit label-efficiency design fractions0.05 and0.01, displayed as percentages; corresponding named LE macros identify the same fraction.")
    r.review(79, [0], ["-1"], [{"source": "lr_interpretability_code", "excerpt": "AttentiveProbe d=1"}],
             "Attentive-probe depth1 is an architecture identifier, not an attribution measurement.")
    r.review(80, [0], ["256"], [{"source": "lr_interpretability_code", "excerpt": "(3000, 256)"}],
             "Recorded attribution layout is256 patch tokens per selected slice.")
    r.review(82, [3], ["7"], [{"source": "lr_interpretability_code", "excerpt": "_W7.npz"}],
             "The historical interpretation procedure uses the W7 window artifact; no amplification estimate is being certified.")
    r.review(83, [0], ["0.5"], [{"source": "lr_interpretability_code", "excerpt": "probs >= 0.5"}],
             "Declared classification threshold used to form outcome strata, not a calibrated operating-point estimate.")
    r.review(84, [1], ["15"], [n.ref("lr_operating", "arms", "random", "ece_15bin")],
             "The source field explicitly identifies the15-bin ECE estimator.")
    for group, indices, expected in ((85, [0, 1, 2], ["0.90", "0.90", "0.85"]), (87, [0], ["0.90"])):
        r.review(group, indices, expected, [n.ref("lr_operating", "target_specificities")],
                 "Prespecified validation specificity targets0.90/0.85, not achieved test specificity.")


def empirical(r):
    ref, op, fmt = n.ref, n.operation, n.fmt
    r.field(24, 1, "lr_geom", "anatomy", "hidden_pct_on_anat", pattern="%.1f")
    r.expression(24, 3, fmt("%d", op("length", ref("lr_geometry_derived", "rectangle_arms"))))
    r.expression(24, 4, fmt("%d", op("length", ref("lr_geometry_derived", "all_arms"))))
    r.field(25, 1, "lr_geom", "anatomy", "n_slots_mean", pattern="%.0f")
    rect = ("random", "oracle", "envelope", "cover")
    r.expression(25, 2, fmt("%.0f", op("mean", *[ref("lr_geom", arm, "n_slots_mean") for arm in rect])))
    for group, positions in ((27, (3, 4)), (74, (1, 2))):
        for index, region, threshold in zip(positions, ("background", "anatomy"), ("<=0.10", ">=0.20")):
            r.expression(group, index, fmt("%.1f", op("percent", ref("lr_position", "ckpts", "random_ep100", region, threshold, "position_share"))))
    relations = r.data["lr_class_relations"]
    untrained, trained = ["JEPA untrained (control)"], ["JEPA ep100 (envelope)"]
    if untrained[0] not in relations or trained[0] not in relations:
        raise ValueError("named class-relation checkpoints missing")
    for group, indices in ((27, (5, 6)), (72, (3, 4))):
        for index, key in zip(indices, (untrained[0], trained[0])):
            r.field(group, index, "lr_class_relations", key, "bg_bg", pattern="%.3f")
    # Regional protocol and independent incremental-feature results.
    for group, indices in ((27, (8, 9, 10, 11)), (73, (1, 2, 3, 4))):
        r.review(group, [indices[0]], ["25"], [ref("lr_incremental", "note")],
                 "The stored incremental-analysis artifact explicitly records25 stratified slices per volume.")
        for index, split in zip(indices[1:], ("Training", "Validation", "Test")):
            r.field(group, index, "lr_incremental", "random", "n", split, pattern="%d")
    for group, r2index, aucindex in ((27, 12, 13), (73, 6, 8)):
        r.expression(group, r2index, fmt("%.1f", op("percent", ref("lr_incremental", "random", "bg_residual_on_anatomy", "ridge_test_R2_bg_from_anatomy"))))
        r.field(group, aucindex, "lr_incremental", "random", "bg_residual_on_anatomy", "test_auc")
    r.review(73, [5], ["100"], [ref("lr_primary_result", "num_slices")], "This is the primary protocol used only as contrast to the25-slice diagnostic.")
    r.field(73, 7, "lr_incremental", "random", "n", "Test", pattern="%d")
    r.review(73, [9], ["2{,}000"], [{"source": "lr_incremental_code", "excerpt": "def boot_ci(y, s, n=2000):"}],
             "Regional fixed-head confidence intervals use2000 case-percentile draws.")
    for index, endpoint in ((10, 0), (11, 1)):
        r.field(73, index, "lr_incremental", "random", "bg_residual_on_anatomy", "test_auc_ci95", endpoint)
    for index, position in ((12, 2), (13, 0), (14, 1)):
        r.field(73, index, "lr_incremental", "random", "delta_cat_minus_anatomy_ci95_and_mean", position)
    r.assertion(73, 15, op("max", *[op("abs", ref("lr_incremental", arm, "delta_cat_minus_anatomy_ci95_and_mean", 2))
                                  for arm in ("oracle", "envelope", "blob")]), "lt", "0.002")
    r.field(73, 16, "lr_incremental", "random", "background", "test_auc", pattern="%.3f")
    # Exact source-bound inequalities, never rounded into a passing bound.
    table = r.data["lr_stats"]["table"]
    rows = {row["key"]: i for i, row in enumerate(table)}
    full_diff = []
    for arm, source_arm, precision in (("random", "random", "fp16"), ("intensity", "oracle", "fp16"),
                                       ("envelope", "envelope", "fp16"), ("cover", "cover-f021", "fp32")):
        full_diff.append(op("abs", op("subtract", ref("lr_le", "arms", arm, "1.00", "auc_mean"),
                                      ref("lr_stats", "table", rows[f"{source_arm}@ep100@{precision}"], "auc"))))
    r.assertion(28, 1, op("max", *full_diff), "le", "0.0009")
    r.assertion(75, 7, op("max", *full_diff[:2]), "le", "0.0003")
    fp = op("max", *[op("abs", ref("lr_fp32", "rows", i, "delta_fp32_minus_fp16"))
                    for i in range(len(r.data["lr_fp32"]["rows"]))])
    for group, indices, relation in ((14, [6, 7, 8], "le"), (30, [1, 2, 3], "lt")):
        for part, index in enumerate(indices):
            r.assertion(group, index, fp, relation, r"2\times10^{-4}", part)
    for part, index in enumerate((2, 3, 4)):
        r.expression(105, index, op("tex_scientific", fp, digits=1), component=part)
    for part, index in enumerate((8, 9, 10)):
        r.expression(30, index, op("tex_scientific", op("abs", ref("lr_fixed_head", "delta")), digits=2), component=part)
    r.field(30, 11, "lr_fixed_head", "all_pair_order_disagreements", pattern="%d")
    r.field(30, 12, "lr_fixed_head", "pair_count", pattern="%d")
    r.field(37, 15, "lr_fixed_head", "ancestor_bytes", pattern="%d")
    excluded = [i for i, row in enumerate(r.data["lr_inventory"]["records"]) if row["status"] in ("excluded", "retracted")]
    r.expression(34, 0, fmt("%d", op("add", op("length", ref("lr_stats", "table")),
                                      op("length", op("array", *[ref("lr_inventory", "records", i) for i in excluded])))))
    r.field(40, 3, "lr_arm_stats", "blob    (mirage_anatomy)", "n", pattern="%d")
    floor = r.data["lr_floor"]
    # The original floor-sweep layout is checked by named keys below.
    for group, indices in ((40, (4, 5)), (67, (6, 7))):
        r.field(group, indices[0], "lr_floor", "0.21", "pct_anat_hid", pattern="%.1f")
        r.field(group, indices[1], "lr_floor", "envelope", "pct_anat_hid", pattern="%.1f")
    r.field(42, 0, "lr_geometry_derived", "inside_unrounded_range", pattern="%d")
    r.field(42, 1, "lr_geometry_derived", "cell_count", pattern="%d")
    r.field(43, 0, "lr_geom", "_meta", "slices", pattern="%d")
    r.field(44, 1, "lr_geometry_derived", "anatomy_context_over_rectangle_mean", pattern="%.1f")
    r.field(45, 0, "lr_geom", "_meta", "slices", pattern="%d")
    # Both per-image columns are independently bound to the same primary field.
    r.assertion(45, 2, op("subtract", ref("lr_geom", "random", "ctx_frac_of_grid"),
                         ref("lr_geom", "random", "ctx_frac_of_grid")), "le", "0.4")
    for index, field, pattern in ((0, ("spearman", "floor021", "rho"), "%+.2f"),
                                  (4, ("cover_hidden_floor015_bs64",), "%.1f"),
                                  (5, ("cover_hidden_floor021_bs64",), "%.1f"),
                                  (6, ("spearman", "floor015", "rho"), "%+.2f")):
        r.field(47, index, "lr_geometry_derived", *field, pattern=pattern)
    r.assertion(48, 3, ref("lr_geometry_derived", "rectangle_slots_min"), "ge", "158")
    r.assertion(48, 4, ref("lr_geometry_derived", "rectangle_slots_max"), "le", "160")
    r.field(49, 1, "lr_fairness", "n_bootstrap", pattern="%d")
    for index, field in ((13, "age_min"), (14, "age_max")):
        r.field(50, index, "lr_metadata", field, pattern="%.1f")
    # The counts are factual; their claimed retention rule is separately blocked.
    r.field(51, 0, "lr_metadata", "categorical_counts", "maritalstatus", "unknown", pattern="%d")
    r.field(51, 1, "lr_metadata", "marital_unknown_percent", pattern="%.1f")
    for index, level in ((2, "severe"), (3, "moderate"), (4, "mild")):
        r.field(51, index, "lr_metadata", "severity_n", level, pattern="%d")
    r.field(51, 5, "lr_metadata", "md_sentinel", pattern="%d")
    for index in (6, 7):
        r.field(51, index, "lr_metadata", "md_sentinel_count", pattern="%d")
    r.hold(52, [0, 1], "The50/10 rule belongs to p7_fairness, not the seven-attribute trend producer: subgroup_analysis uses min_n40 and skips unknown/blank/na. Stored Spanish n44 is included; marital unknown71 and legally-separated22 are omitted. Scope/reword the methods paragraph.")
    r.field(53, 0, "lr_trend", "trends", "race", "n_branches", pattern="%d")
    r.field(53, 2, "lr_trend", "trends", "race", "branch_spearman_p")
    # Intersectional reductions use exactly the18 retained keys, not all23 newer probes.
    for group, indices in ((59, (2, 3, 11)), (60, (1,)), (61, (0, 1, 4, 6)), (62, (0, 4))):
        for index in indices:
            r.field(group, index, "lr_intersection_derived", "counts", "retained_records", pattern="%d")
    r.field(59, 1, "lr_intersection_derived", "counts", "all_records", pattern="%d")
    for index, field in ((2, "female_below_male"), (3, "black_below_white")):
        r.field(61, index, "lr_intersection_derived", "counts", field, pattern="%d")
    for group, indices in ((60, (2, 3)), (64, (0, 1))):
        for index, cell in zip(indices, ("asian x female", "asian x male")):
            cells = r.data["lr_intersection"]["sweep_random_ep100"]["cells"]
            ix = next(i for i, row in enumerate(cells) if row["subgroup"] == cell)
            r.field(group, index, "lr_intersection", "sweep_random_ep100", "cells", ix, "n", pattern="%d")
    r.assertion(58, 0, op("max", *[ref("lr_intersection", "sweep_random_ep100", "cells", i, "n")
                                 for i, row in enumerate(r.data["lr_intersection"]["sweep_random_ep100"]["cells"])
                                 if row["subgroup"].startswith("asian x")]), "lt", "130")
    for index, path, pattern in ((1, ("mean_gaps", "gender"), "%.4f"), (2, ("mean_gaps", "race"), "%.4f"),
        (3, ("mean_gaps", "intersection"), "%.4f"), (5, ("understatement_percent",), "%.1f"),
        (6, ("oracle_race_gap",), "%.4f"), (7, ("oracle_intersection_gap",), "%.4f"),
        (8, ("oracle_black_female_auc",), "%.4f"), (9, ("oracle_black_margin_minus_female",), "%.4f"),
        (10, ("additive_gap",), "%.4f"), (11, ("mean_gaps", "intersection"), "%.4f"), (12, ("additive_ratio",), "%.3f")):
        r.field(62, index, "lr_intersection_derived", *path, pattern=pattern)
    r.expression(63, 2, fmt("%.4f", op("abs", ref("lr_intersection_derived", "deltas", "oracle_minus_random_ep50", "asian x female"))))
    for index, key, cell in ((4, "envelope_minus_random_ep75", "asian x female"),
                            (5, "envelope_minus_random_ep75", "asian x male"),
                            (6, "envelope_minus_random_ep75", "black x female"),
                            (8, "envelope_minus_random_ep100", "asian x male"),
                            (9, "envelope_minus_random_ep100", "black x female"),
                            (10, "envelope_minus_random_ep100", "asian x female")):
        r.field(63, index, "lr_intersection_derived", "deltas", key, cell, pattern="%+.4f")
    r.field(67, 3, "lr_rectangles", "random_legal", "blocks_checked", pattern="%d")
    r.field(67, 4, "lr_rectangles", "random_legal", "perfect_rectangles_pct", pattern="%.1f")
    r.hold(67, [5], "The194-slice first audit was not persisted; its sample count survives only in narrative COVER_AUDIT.md, which is not an independent producing artifact. Remove the number or explicitly label it a historical unverified recollection.")
    spread = lambda fraction: op("subtract", op("max", *[ref("lr_le", "arms", arm, fraction, "auc_mean") for arm in ("random", "intensity", "envelope", "cover")]),
                                 op("min", *[ref("lr_le", "arms", arm, fraction, "auc_mean") for arm in ("random", "intensity", "envelope", "cover")]))
    r.assertion(78, 1, spread("1.00"), "le", "0.027")
    r.expression(78, 3, fmt("%.3f", spread("0.05")))
    ft = [ref("lr_inventory", "records", i, "auc") for i, row in enumerate(r.data["lr_inventory"]["records"])
          if row["family"] == "finetune" and row["arm"] == "random"]
    r.expression(79, 1, fmt("%.3f", op("mean", *ft)))
    r.field(81, 1, "lr_parameters", "parameters_millions", pattern="%.2f")
    contrasts = r.data["lr_stats"]["contrasts"]
    ci = next(i for i, row in enumerate(contrasts) if row["a"] == "oracle@ep100@fp16" and row["b"] == "random@ep100@fp16")
    r.field(86, 0, "lr_stats", "contrasts", ci, "delta_a_minus_b", pattern="%+.3f")
    r.expression(90, 1, fmt("%d", op("add", *[ref("lr_metadata", "split_sizes", split) for split in ("Training", "Validation", "Test")])))
    r.field(90, 2, "lr_metadata", "split_sizes", "Training", pattern="%d")
    r.field(90, 3, "lr_metadata", "split_sizes", "Validation", pattern="%d")
    r.field(101, 4, "lr_stats", "multiplicity", "exploratory_family_size", pattern="%d")
    r.field(102, 2, "lr_geom", "anatomy", "hidden_pct_on_anat", pattern="%.0f")
    r.field(102, 3, "lr_geom", "oracle", "hidden_pct_on_anat", pattern="%.0f")
    r.field(102, 4, "lr_geom", "envelope", "hidden_pct_on_anat", pattern="%.0f")
    r.field(104, 1, "lr_geometry_derived", "anatomy_context_over_rectangle_mean", pattern="%.1f")
    r.field(104, 2, "lr_geometry_derived", "anatomy_slots_over_rectangle_mean", pattern="%.1f")
    r.field(107, 0, "lr_probability", "paired_deltas", "envelope_minus_random", "W", pattern="%.3f")
    r.field(107, 1, "lr_probability", "paired_deltas", "oracle_minus_random", "W", pattern="%.3f")
    maxp = op("max", *[ref("lr_probability", "paired_deltas", key, "p") for key in r.data["lr_probability"]["paired_deltas"]])
    for part, index in enumerate((2, 3)):
        r.assertion(107, index, maxp, "lt", r"10^{-10}", part)
    rawmax = op("max", *[ref("lr_probability", "raw_scores", arm, "p") for arm in ("random", "oracle", "envelope")])
    for part, index in enumerate((4, 5), 1):
        r.expression(107, index, op("tex_scientific", rawmax, digits=1), component=part)
    r.field(107, 6, "lr_probability", "levene", "statistic", pattern="%.1f")
    r.assertion(107, 7, ref("lr_probability", "levene", "p"), "lt", "0.001")
    r.field(107, 8, "lr_probability", "variance_ratio", pattern="%.2f")
    r.field(107, 9, "lr_probability", "n_stacked_scores", pattern="%d")


def software(r):
    environment = r.data["lr_environment"]
    expected = {0: "3.11", 1: "64", 2: "10.0", 3: "2.7", 4: "12.8", 5: "9.7",
                6: "3090", 7: "24", 8: "610.62", 9: "2.4", 10: "1.17", 11: "1.9",
                12: "0.17", 15: "9.1", 16: "0.14", 17: "0.13"}
    assert environment["python"] == "3.11.9" and environment["windows_version"] == "10.0.26200"
    assert environment["versions"] == {"torch": "2.7.1+cu128", "numpy": "2.4.4", "scipy": "1.17.1",
                                       "scikit-learn": "1.9.0", "pytest": "9.1.1", "statsmodels": "0.14.6", "seaborn": "0.13.2"}
    assert environment["cudnn_library_version"] == 90701 and environment["cuda_compiled"] == "12.8"
    hardware = r.data["lr_hardware"]
    assert len(hardware["devices"]) == 1
    assert hardware["devices"][0]["name"] == "NVIDIA GeForce RTX 3090"
    assert hardware["devices"][0]["vram_GiB"] == 24 and hardware["devices"][0]["nvidia_driver_version"] == "610.62"
    assert hardware["tectonic_version_output"].lower() == "tectonic 0.17.0"
    r.review(106, list(expected), list(expected.values()), [n.ref("lr_environment"), n.ref("lr_hardware")],
             "Directly inspected complete version/model strings, not merely numeric fragments. Snapshot verifies the local analysis environment, not every historical pretraining environment. CUDA is a compiled/library declaration; no GPU work was performed.")
    r.field(106, 13, "lr_environment", "lock_count", pattern="%d")
    r.field(106, 14, "lr_environment", "extra_count", pattern="%d")


def citations(r):
    manifest = json.loads((SOURCES / "public" / "download_manifest.json").read_text())
    entries = [
        (91, "mae", [1,2,3,4,5,6,7,8,9], ["-75","84.9","73.5","-75","82.8","63.9","-75","84.0","66.0"],
         "Published CVPR2022 PDF p5, Table1(f): random75=84.9FT/73.5linear; block75=82.8/63.9; grid75=84.0/66.0."),
        (92, "simmim", [1,3,4], ["83.0","82.7","82.6"],
         "Published CVPR2022 PDF p5, Table1: random32px/ratio0.5=83.0; maximum block-wise=82.7; maximum square=82.6."),
        (93, "semmae", [1,2,3,4], ["66.8","66.5","52.9","68.7"],
         "NeurIPS2022 PDF p7, Table3: random66.8, mask75% patches66.5, mask75% parts52.9, adaptive alpha0->1/gamma2=68.7."),
        (94, "semmae", [1,2,3], ["63.7","63.6","65.0"],
         "NeurIPS2022 PDF p8, Table4: baseline without parts63.7, iBOT-initialized parts63.6, learned parts65.0."),
        (95, "automae", [1,2], ["83.26","83.32"],
         "arXiv2303.06583v1 PDF p7, Table5,100% ratio: MAE83.26 and AutoMAE83.32 ImageNet fine-tuning."),
        (96, "hpm", [1,2,3], ["82.49","82.95","81.40"],
         "Published CVPR2023 PDF p6, Table2,mask ratio75: random alpha0=alphaT=0 gives82.49; alpha0=0,alphaT=.5 gives82.95; alpha0=alphaT=1 gives81.40."),
        (97, "attmask", [1,2,3], ["43.4","43.5","43.6"],
         "arXiv2203.12719v2 PDF p11, Table3 (DINO,20% IN1K,k-NN): random43.4, AttMask-High43.5, AttMask-Hint43.6."),
        (98, "attg", [1,2,3], ["78.5","10","78.1"],
         "Published WACV2025 PDF p5, Table2,10% labelled IN1K/ViT-L1600 epochs: MAE78.5 and AttG78.1."),
        (99, "ssit", [1,2,3,4,5], ["79.48","25","77.53","50","79.97"],
         "arXiv2210.10969v2 PDF p7, TableIII,Messidor-2 kappa means:0%mask79.48,25%mask77.53,50%mask79.97; SDs are not copied as means."),
    ]
    registered = set()
    for group, name, indices, values, locator in entries:
        source = "lr_public_" + name
        if name not in registered:
            r.source(source, SOURCES / "public" / (name + ".pdf"))
            registered.add(name)
        item = manifest[name]
        r.review(group, indices, values, [n.ref(source, "sha256")],
                 "Inspected the original paper table and matched row, metric and protocol, not just number occurrence. " + locator,
                 kind="citation", immutable_locator=item["resolved_url"] + "#sha256=" + item["sha256"], locator=locator)
    r.source("lr_fairvision_card", SOURCES / "public" / "fairvision_readme.txt")
    card = json.loads((SOURCES / "public" / "fairvision_source.json").read_text())
    r.review(90, [4], ["4.0"], [{"source": "lr_fairvision_card", "excerpt": "license: cc-by-nc-nd-4.0"}],
             "Licence version from the official dataset card, pinned to an immutable repository commit; not the code repository licence.",
             kind="citation", immutable_locator=card["url"], locator="YAML licence declaration and Dataset Details/Uses.")


def current_candidate(r):
    """Reuse exact matches and only the individually re-reviewed revisions."""
    _, files = assets.source_tree(assets.PAPER)
    rows = {(x["file"], x["context_sha256"], x["token_index"]): x
            for file, source in files.items() for x in n.literals(source, file)}
    candidate = {**r.document, "literals": []}
    withdrawn = []
    for entry in r.document["literals"]:
        key = (entry["file"], entry["context_sha256"], entry["token_index"])
        if key in rows and rows[key]["value"] == entry["value"]:
            candidate["literals"].append(entry)
        else:
            withdrawn.append(key)
    # Read in full during this review: only the pipeline example-reference
    # sentence and the attribution caveat changed; their numerical meanings did not.
    revisions = [
        (10, [17], "5362ff287c76afc5ae4d6037be6ae92f5f4cb3eeb555bbacaf09adb11b600bf8"),
        (24, [0, 1, 3, 4], "95481ec4b0811261c5aa3ec869540c5d3ece1bb87dd2df8d04c12fcb9632ed1c"),
        (30, [1, 2, 3, 5], "50034426a6d5f0180b67b080cfbe865b41d87f656d3dfe9a2ceb47a19a3d184a"),
        (43, [0], "9e881e4cec4d798bf558dd51bf2a889f7e7e347cafca9141d59182a269eade96"),
        (50, [13, 14], "090892b763f5e0b86491bed7f2f63ec1c878f4b1310a820cfa57773d0367e44e"),
        (51, list(range(8)), "6cf56d8ef4833253f87cb4c041e2887f2f3051dc37ec1612076a3df6e404207e"),
        (67, [3, 4], "72b84b26c0b5654c07d1b45eb0c9b6996e34dcf518267cbfdd47a5d9ccae61ed"),
        (78, [2, 3], "5b99c4f344ca82565669d8bf5dcb397157c811e4a86b1154faa18041376b8a17"),
        (90, [1, 2, 3, 4], "3f620b52369cebe00b18f4e3a594a43f917609d854200c7a398bf2a56663d4e2"),
        (106, list(range(14)), "fe96d432eb37acccadea5bc7d6231bc13d9a6e6d4a93038fe1476ed3080a918a"),
    ]
    for group, indices, digest in revisions:
        for index in indices:
            key = ("main_submission.tex", digest, index)
            if key not in rows:
                continue
            original = r.assignments[group, index]
            if rows[key]["value"] != original["value"]:
                raise ValueError("re-reviewed context token changed")
            candidate["literals"].append({**original, "context_sha256": digest})
    # Explicitly rereviewed corrections: the cache-only reproduction removes
    # the misleading22-pair assertion; the unretained194 count is withdrawn.
    remapped = [
        (30, 8, 7, "50034426a6d5f0180b67b080cfbe865b41d87f656d3dfe9a2ceb47a19a3d184a"),
        (30, 9, 8, "50034426a6d5f0180b67b080cfbe865b41d87f656d3dfe9a2ceb47a19a3d184a"),
        (30, 10, 9, "50034426a6d5f0180b67b080cfbe865b41d87f656d3dfe9a2ceb47a19a3d184a"),
        (67, 6, 5, "72b84b26c0b5654c07d1b45eb0c9b6996e34dcf518267cbfdd47a5d9ccae61ed"),
        (67, 7, 6, "72b84b26c0b5654c07d1b45eb0c9b6996e34dcf518267cbfdd47a5d9ccae61ed"),
    ]
    for group, old_index, index, digest in remapped:
        key = ("main_submission.tex", digest, index)
        if key in rows:
            original = r.assignments[group, old_index]
            if rows[key]["value"] != original["value"]:
                raise ValueError("explicitly remapped reviewed token changed")
            candidate["literals"].append({**original, "context_sha256": digest, "token_index": index})
    fresh = [
        ("c11ecbba379b77409b5a84ae2ab6c99241d8eba23790f18694a1ffe07631dbb9", 0, "-42",
         {"review": {"kind": "protocol", "reviewer": REVIEWER,
                     "rationale": "The corrected text identifies the retained seed42 draw and distinguishes sampler variation from encoder-retraining uncertainty. The hyphen is lexical.",
                     "evidence": [n.ref("lr_geom", "_meta", "seed")]}}),
        ("5b99c4f344ca82565669d8bf5dcb397157c811e4a86b1154faa18041376b8a17", 1, "0.027",
         {"expression": n.fmt("%.3f", r.assignments[78, 1]["assertion"]["expression"])}),
    ]
    for index, value in enumerate(("50", "10", "40")):
        fresh.append(("5e918676666235cf5ced5d3ddaed288ff2584ea9299ae449662255c33a7130e5", index, value,
                      {"review": {"kind": "protocol", "reviewer": REVIEWER,
                                  "rationale": "The corrected paragraph now names two different workflows:50/10 is p7_fairness; the seven-attribute trend uses40 and omits unknown/blank/na. This matches both code and retained eligible-level outputs; no statistics were changed.",
                                  "evidence": [
                                      {"source": "lr_fairness_code", "excerpt": "MIN_N, MIN_CLASS = 50, 10"},
                                      {"source": "lr_subgroup_code", "excerpt": 'ap.add_argument("--min-n", type=int, default=40)'},
                                      {"source": "lr_subgroup_code", "excerpt": 'if v in ("", "unknown", "na"):'},
                                  ]}}))
    for digest, index, value, fields in fresh:
        key = ("main_submission.tex", digest, index)
        if key in rows:
            if rows[key]["value"] != value:
                raise ValueError("newly reviewed correction changed")
            candidate["literals"].append({"file": key[0], "context_sha256": digest, "token_index": index, "value": value, **fields})
    # Independently reviewed replacement purity caption: Table2 is an index,
    # and all five displayed policy endpoints are explicitly epoch50/fp32.
    digest = "2172561ed4d6885b9b91c7940e6cf53e638244c9668c879cea64a38c465a96dd"
    if ("main_submission.tex", digest, 2) in rows:
        current_text = files["main_submission.tex"]
        table_labels = []
        for table in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", current_text, re.S):
            label = re.search(r"\\label\{([^}]+)\}", table[0])
            if label:
                table_labels.append(label[1])
        if table_labels.index("tab:geom") + 1 != 2:
            raise ValueError("geometry table is no longer Table2")
        for index, value, evidence, rationale in [
            (2, "2", [{"source": "lr_geometry_figure", "excerpt": "same artifact that backs Table 2"}],
             "Document reference to the second table, independently checked against current table order and the geometry producer's documented Table2 source; not a scientific measurement."),
            (3, "-50", [n.ref("lr_stats", "table", i) for i, row in enumerate(r.data["lr_stats"]["table"])
                        if row["epoch"] == 50 and row["precision"] == "fp32" and row["arm"] in ("random", "oracle", "envelope", "cover-f021", "anatomy-v2")],
             "All five named endpoints exist at epoch50/fp32. The hyphen is part of 'epoch-50', not a negative quantity."),
        ]:
            row = rows["main_submission.tex", digest, index]
            if row["value"] != value:
                raise ValueError("replacement caption changed")
            candidate["literals"].append({"file": row["file"], "context_sha256": digest,
                "token_index": index, "value": value,
                "review": {"kind": "formula" if index == 2 else "protocol", "reviewer": REVIEWER,
                           "rationale": rationale, "evidence": evidence}})
    candidate["scope"] = ("Exact surviving61d reviews plus individually re-reviewed changed-context hashes; "
                          "unlisted new contexts remain unresolved. No parent-owned review file is modified.")
    assets.write_json(OUT / "literal_review_candidate_current.json", candidate)
    assets.write_json(OUT / "literal_context_migration.json", {
        "original_context_source_sha256": "61d8d2cceb397f58313b166d481a03a3f94c2c0c914fb2ee8caca25cef7e27e6",
        "current_source_sha256": assets.sha256(assets.PAPER / "main_submission.tex"),
        "withdrawn_or_changed_not_blindly_reused": withdrawn,
        "manually_reviewed_context_hashes": sorted(set([x[2] for x in revisions] + [x[3] for x in remapped] + [x[0] for x in fresh] + [digest]))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT / "literal_review_candidate_61d.json"))
    args = parser.parse_args()
    r = Reviews()
    register(r)
    methods(r)
    empirical(r)
    software(r)
    citations(r)
    r.document["literals"] += list(r.assignments.values())
    r.document["scope"] = ("Explicit per-context literal reviews/field bindings for audited source61d; "
                           "contradicted assertions remain mismatches, unassigned tokens remain blockers. "
                           "Includes existing DT and selected-map receipts; no parent numeric_reviews.json write.")
    assets.write_json(args.out, r.document)
    pending = [{"group": g["group"], "index": token["index"], "value": token["value"], "context": g["context"]}
               for g in r.queue for token in g["tokens"] if (g["group"], token["index"]) not in r.assignments]
    assets.write_json(OUT / "literal_review_findings.json", {
        "source_snapshot_sha256": "61d8d2cceb397f58313b166d481a03a3f94c2c0c914fb2ee8caca25cef7e27e6",
        "scope": "Historical frozen61d findings; use current validation for revised-source blockers.",
        "assigned": len(r.assignments), "pending": pending, "findings": r.findings, "source_origins": r.origins})
    current_candidate(r)
    print("Historical61d assigned:", len(r.assignments), "pending:", len(pending), "findings:", len(r.findings))
    print("Current-source candidate:", OUT / "literal_review_candidate_current.json")


if __name__ == "__main__":
    main()
