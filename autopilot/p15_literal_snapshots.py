"""Independent CPU-only reductions for the remaining literal evidence review.

No training, encoder execution, source-data writes, or manuscript edits. The
optional fixed-head check applies retained weights to already cached features.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import subprocess

import numpy as np
from scipy import stats

try:
    from . import release_assets as assets
except ImportError:
    import release_assets as assets

ROOT = assets.REPO
PHASE = Path(r"D:\jepa_phase0")
STATS = PHASE / "autopilot_out" / "p1_stats"
OUT = ROOT / "autopilot" / "investigations" / "delivered_task" / "evidence" / "literal_sources"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Snapshot:
    def __init__(self):
        self.inputs = {}

    def load(self, path):
        path = Path(path)
        self.inputs[str(path)] = file_hash(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def save(self, name, values, operations):
        for path, digest in self.inputs.items():
            if file_hash(path) != digest:
                raise ValueError("evidence changed during reduction: " + path)
        payload = {"producer": str(Path(__file__).relative_to(ROOT)),
                   "producer_sha256": file_hash(__file__), "inputs_sha256": self.inputs,
                   "operations": operations, **values}
        assets.write_json(OUT / name, payload)
        print(name)


def subgroup_snapshot():
    s = Snapshot()
    inter = s.load(PHASE / "reports" / "subgroup" / "intersectional_auc.json")
    marginal = s.load(PHASE / "reports" / "subgroup" / "subgroup_auc.json")
    retained = {key: row for key, row in inter.items() if row["status"] == "OK"}
    cells = {key: {r["subgroup"]: r["auc"] for r in row["cells"]} for key, row in retained.items()}
    gap = {key: max(values.values()) - min(values.values()) for key, values in cells.items()}
    race, gender = {}, {}
    for key in retained:
        pg = marginal[key]["subgroups"]
        for attr, output in (("race", race), ("gender", gender)):
            values = [row["auc"] for row in pg[attr]["levels"]]
            output[key] = max(values) - min(values)
    means = {"intersection": float(np.mean(list(gap.values()))),
             "race": float(np.mean(list(race.values()))), "gender": float(np.mean(list(gender.values())))}
    counts = {
        "all_records": len(inter), "retained_records": len(retained),
        "worst_black_female": sum(min(v, key=v.get) == "black x female" for v in cells.values()),
        "best_asian_male": sum(max(v, key=v.get) == "asian x male" for v in cells.values()),
        "female_below_male": sum(v[f"{race} x female"] < v[f"{race} x male"]
                                  for v in cells.values() for race in ("asian", "black", "white")),
        "black_below_white": sum(v[f"black x {sex}"] < v[f"white x {sex}"]
                                  for v in cells.values() for sex in ("female", "male")),
        "intersection_exceeds_race": sum(gap[k] > race[k] for k in retained),
    }
    deltas = {}
    for epoch in (50, 75, 100):
        for arm, probe in (("oracle", "sweep_oracle"), ("envelope", "frozen_meanpool_mirage")):
            a, b = cells[f"{probe}_ep{epoch}"], cells[f"sweep_random_ep{epoch}"]
            deltas[f"{arm}_minus_random_ep{epoch}"] = {k: a[k] - b[k] for k in a}
    oracle = "sweep_oracle_ep100"
    s.save("intersectional_reduction.json", {
        "retained_keys": sorted(retained), "counts": counts, "mean_gaps": means,
        "understatement_percent": 100 * (means["intersection"] - means["race"]) / means["race"],
        "additive_gap": means["race"] + means["gender"],
        "additive_ratio": means["intersection"] / (means["race"] + means["gender"]),
        "oracle_race_gap": race[oracle], "oracle_intersection_gap": gap[oracle],
        "oracle_black_female_auc": cells[oracle]["black x female"],
        "oracle_black_margin_minus_female": next(r["auc"] for r in marginal[oracle]["subgroups"]["race"]["levels"]
                                                if r["subgroup"] == "black") - cells[oracle]["black x female"],
        "deltas": deltas,
    }, ["Select status == OK from the intersectional artifact; use exactly those keys in the marginal artifact.",
        "Per-arm gaps = max(group AUC)-min(group AUC); arithmetic mean across retained keys.",
        "Order counts and matched-epoch cell deltas use named race/gender cells, never nearest values."])


def geometry_snapshot():
    s = Snapshot()
    folder = ROOT / "results" / "masking" / "table2_geometry"
    seeds = {seed: s.load(folder / ("mask_geometry_600slices_bs1_coverf021_seed%d.json" % seed))
             for seed in (42, 1234, 2026)}
    full = s.load(folder / "mask_geometry_600slices_bs64_coverf021_seed42.json")
    alternative = s.load(folder / "mask_geometry_600slices_bs64_coverf015_seed42.json")
    table = s.load(STATS / "p1c_stats.json")["table"]
    auc = {r["key"]: r["auc"] for r in table}
    rect = ["random", "oracle", "envelope", "cover"]
    all_arms = rect + ["anatomy"]
    fields = {"hidden_share_of_all_anat": 1, "hidden_pct_on_anat": 1, "hidden_frac_of_grid": 100,
              "ctx_frac_of_grid": 100, "n_slots_mean": 1}
    cells = []
    for arm in all_arms:
        for field, factor in fields.items():
            measured = [seeds[seed][arm][field] * factor for seed in (42, 1234, 2026)]
            displayed = float("%.1f" % measured[0])
            cells.append({"arm": arm, "field": field, "seed_values": measured, "source_formatted": displayed,
                          "inside_unrounded_range": min(measured) <= displayed <= max(measured),
                          "within_redraw_range_or_rounding": abs(displayed - measured[0]) <= max(max(measured) - min(measured), .05 + 1e-12)})
    y = [auc[f"{a if a != 'cover' else 'cover-f021'}@ep50@{'fp32' if a == 'cover' else 'fp16'}"] for a in rect]
    spearman = {}
    for label, geometry in (("floor021", full), ("floor015", alternative)):
        value = stats.spearmanr([geometry[a]["hidden_share_of_all_anat"] for a in rect], y)
        spearman[label] = {"rho": float(value.statistic), "p": float(value.pvalue)}
    s.save("geometry_reduction.json", {
        "rectangle_arms": rect, "all_arms": all_arms, "cells": cells,
        "inside_unrounded_range": sum(c["inside_unrounded_range"] for c in cells),
        "within_redraw_range_or_rounding": sum(c["within_redraw_range_or_rounding"] for c in cells),
        "cell_count": len(cells), "spearman": spearman,
        "anatomy_context_over_rectangle_mean": seeds[42]["anatomy"]["ctx_frac_of_grid"] / np.mean([seeds[42][a]["ctx_frac_of_grid"] for a in rect]),
        "delivered_anatomy_context_over_rectangle_mean": full["anatomy"]["ctx_frac_of_grid"] / np.mean([full[a]["ctx_frac_of_grid"] for a in rect]),
        "anatomy_slots_over_rectangle_mean": seeds[42]["anatomy"]["n_slots_mean"] / np.mean([seeds[42][a]["n_slots_mean"] for a in rect]),
        "rectangle_slots_min": min(seeds[42][a]["n_slots_mean"] for a in rect),
        "rectangle_slots_max": max(seeds[42][a]["n_slots_mean"] for a in rect),
        "rectangle_mask_percent_min": min(seeds[42][a]["hidden_frac_of_grid"] * 100 for a in rect),
        "rectangle_mask_percent_max": max(seeds[42][a]["hidden_frac_of_grid"] * 100 for a in rect),
        "proposed_target_k_from_mean_slots": np.mean([seeds[42][a]["n_slots_mean"] for a in rect]) / 4,
        "cover_hidden_floor021_bs64": full["cover"]["hidden_share_of_all_anat"],
        "cover_hidden_floor015_bs64": alternative["cover"]["hidden_share_of_all_anat"],
    }, ["Independently reconstruct the one-decimal source formatting of Table geometry from seed42 fields.",
        "Compare each source-formatted cell with the min/max of the three retained seed values and separately with redraw spread/rounding.",
        "Ratios use the arithmetic mean of the four named rectangle arms; no manuscript numbers are inputs.",
        "Spearman uses exact primary ep50 AUC fields and named bs64 geometry records."])


def probability_snapshot():
    s = Snapshot()
    paths = {
        "random": ROOT / "results" / "downstream" / "meanpool_sweep_random" / "ep100_test_predictions.npz",
        "oracle": ROOT / "results" / "downstream" / "meanpool_sweep_oracle" / "ep100_test_predictions.npz",
        "envelope": ROOT / "results" / "downstream" / "meanpool_sweep_mirage" / "ep100_test_predictions.npz",
    }
    scores, labels = {}, {}
    for key, path in paths.items():
        s.inputs[str(path)] = file_hash(path)
        with np.load(path, allow_pickle=False) as data:
            scores[key] = data["probs"].astype(np.float64)
            labels[key] = data["labels"].astype(np.int64)
    assert all(np.array_equal(labels["random"], v) for v in labels.values())
    pairs = {}
    for a, b in (("oracle", "random"), ("envelope", "random"), ("oracle", "envelope")):
        value = stats.shapiro(scores[a] - scores[b])
        pairs[a + "_minus_" + b] = {"W": float(value.statistic), "p": float(value.pvalue)}
    raw = {}
    for arm, x in scores.items():
        value = stats.shapiro(x)
        raw[arm] = {"W": float(value.statistic), "p": float(value.pvalue), "variance": float(np.var(x, ddof=1))}
    levene = stats.levene(*scores.values(), center="median")
    s.save("probability_diagnostics.json", {
        "n_subjects": len(labels["random"]), "n_stacked_scores": sum(len(x) for x in scores.values()),
        "paired_deltas": pairs, "raw_scores": raw,
        "levene": {"statistic": float(levene.statistic), "p": float(levene.pvalue)},
        "variance_ratio": max(v["variance"] for v in raw.values()) / min(v["variance"] for v in raw.values()),
    }, ["Read only retained ep100 prediction NPZ labels/probs; confirm shared labels.",
        "Subtract float64 probabilities before Shapiro-Wilk; raw-score tests use the same unmodified probabilities.",
        "Levene median-centred statistic is descriptive only: the three vectors are paired, not independent samples."])


def environment_snapshot():
    s = Snapshot()
    lock = ROOT / "requirements-phase0.lock.txt"
    s.inputs[str(lock)] = file_hash(lock)
    pins = {}
    for line in lock.read_text().splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.strip().split("==", 1)
            pins[name.lower().replace("_", "-")] = version
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    mismatches = {name: {"expected": version, "actual": installed.get(name)} for name, version in pins.items() if installed.get(name) != version}
    import torch
    s.save("environment_snapshot.json", {
        "python": platform.python_version(), "windows_version": platform.version(), "pointer_bits": struct.calcsize("P") * 8,
        "versions": {name: installed.get(name) for name in ("torch", "numpy", "scipy", "scikit-learn", "pytest", "statsmodels", "seaborn")},
        "cuda_compiled": torch.version.cuda, "cudnn_library_version": torch.backends.cudnn.version(),
        "lock_pins": pins, "lock_count": len(pins), "pin_mismatches": mismatches,
        "extra_packages": {name: version for name, version in installed.items() if name not in pins},
        "extra_count": len(set(installed) - set(pins)),
    }, ["Read current interpreter/distribution metadata and installed cuDNN library version; no CUDA tensors or model execution.",
        "Parse exact == pins and compare names/versions to importlib.metadata; enumerate rather than infer extra-package count."])


def metadata_snapshot():
    s = Snapshot()
    metadata_path = PHASE / "fairvision-glaucoma" / "metadata" / "data_summary_glaucoma.csv"
    s.inputs[str(metadata_path)] = file_hash(metadata_path)
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        metadata = {r["filename"]: r for r in csv.DictReader(stream)}
    data_root = PHASE / "fairvision-glaucoma" / "data"
    filenames = sorted(p.name for p in (data_root / "Test").glob("*.npz"))
    selected = [metadata[name] for name in filenames]
    y = np.array([r["glaucoma"].lower() == "yes" for r in selected], dtype=int)
    reference = ROOT / "results" / "downstream" / "meanpool_sweep_random" / "ep100_test_predictions.npz"
    s.inputs[str(reference)] = file_hash(reference)
    with np.load(reference, allow_pickle=False) as values:
        assert np.array_equal(y, values["labels"].astype(int))
    ages = np.array([float(r["age"]) for r in selected])
    md = np.array([float(r["md"]) for r in selected])
    assert np.isfinite(ages).all() and np.isfinite(md).all()
    counts = {key: dict(Counter(r[key].lower() for r in selected))
              for key in ("gender", "race", "ethnicity", "language", "maritalstatus")}
    log = PHASE / "runs" / "rep_random_s1234" / "jepa_patch_rep_random_s1234-log.csv"
    s.inputs[str(log)] = file_hash(log)
    epochs = {}
    with log.open(newline="") as stream:
        for row in csv.DictReader(stream):
            epochs.setdefault(int(row["epoch"]), set()).add(int(row["iteration"]))
    s.save("metadata_reduction.json", {
        "test_n": len(selected), "test_filename_order_sha256": hashlib.sha256("\n".join(filenames).encode()).hexdigest(),
        "split_sizes": {split: len(list((data_root / split).glob("*.npz"))) for split in ("Training", "Validation", "Test")},
        "age_min": float(ages.min()), "age_max": float(ages.max()), "categorical_counts": counts,
        "marital_unknown_percent": 100 * counts["maritalstatus"].get("unknown", 0) / len(selected),
        "severity_n": {"severe": int(np.sum((md <= -12) & (y == 1))),
                       "moderate": int(np.sum((md > -12) & (md <= -6) & (y == 1))),
                       "mild": int(np.sum((md > -6) & (md <= -2) & (y == 1)))},
        "md_sentinel": -1, "md_sentinel_count": int(sum(md == -1)), "md_sentinel_positive_count": int(sum((md == -1) & (y == 1))),
        "md_threshold_labels_agree": bool(np.array_equal(md <= -2, y == 1)),
        "replication_iterations": {str(epoch): {"distinct_iterations": len(values), "maximum_iteration": max(values)}
                                   for epoch, values in epochs.items()},
    }, ["Join immutable metadata CSV by exact sorted Test filenames; compare reconstructed labels to saved primary predictions.",
        "Read only metadata, never OCT pixels. The selected filename-order digest records the scope without publishing identifiers.",
        "Count complete versus incomplete replication iterations from its retained CSV; do not infer a completed probe."])


def probe_parameter_snapshot():
    s = Snapshot()
    path = ROOT / "results" / "downstream" / "finetune_random" / "attentive_results.json"
    record = s.load(path)
    source = ROOT / "src" / "eval_downstream.py"
    blocks = ROOT / "src" / "models" / "vision_transformer.py"
    s.inputs[str(source)], s.inputs[str(blocks)] = file_hash(source), file_hash(blocks)
    dim, slices, depth = 768, record["num_slices"], record["probe_depth"]
    # QKV/projection/MLP weights and biases, two LayerNorms, then CLS,
    # positional embeddings and final norm. This is parameter arithmetic only.
    per_block = 12 * dim * dim + 13 * dim
    total = depth * per_block + dim + (slices + 1) * dim + 2 * dim
    s.save("probe_parameter_count.json", {"embed_dim": dim, "num_slices": slices, "depth": depth,
                                        "per_block": per_block, "parameters": total, "parameters_millions": total / 1e6},
           ["Use the stored finetuned configuration (64 slices, one block), not the 100-slice frozen configuration.",
            "AttentiveProbe/Block arithmetic: depth*(12*d*d+13*d)+d+(slices+1)*d+2*d, with d=768 for ViT-Base.",
            "No model instantiated, trained or executed."])


def hardware_snapshot():
    import winreg
    s = Snapshot()
    root = r"SYSTEM\CurrentControlSet\Control\Video"
    devices = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root) as key:
        for i in range(winreg.QueryInfoKey(key)[0]):
            subkey = winreg.EnumKey(key, i)
            try:
                with winreg.OpenKey(key, subkey + r"\0000") as device:
                    name = winreg.QueryValueEx(device, "DriverDesc")[0]
                    if "NVIDIA" not in name:
                        continue
                    memory = int(winreg.QueryValueEx(device, "HardwareInformation.qwMemorySize")[0])
                    version = winreg.QueryValueEx(device, "DriverVersion")[0]
                    major, tail = version.split(".")[-2:]
                    driver = "%d.%02d" % (int(major[-1] + tail) // 100, int(major[-1] + tail) % 100)
                    devices.append({"name": name, "vram_bytes": memory, "vram_GiB": memory / 2**30,
                                    "windows_driver_version": version, "nvidia_driver_version": driver})
            except OSError:
                continue
    executable = PHASE / "tools" / "tectonic" / "tectonic.exe"
    s.inputs[str(executable)] = file_hash(executable)
    version = subprocess.run([str(executable), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    s.save("hardware_snapshot.json", {"devices": devices, "tectonic_version_output": version},
           ["Read cached Windows display-driver registry values only; no CUDA initialization or GPU work.",
            "NVIDIA display version = last five Windows driver digits with decimal before final two digits.",
            "VRAM GiB = registry qwMemorySize / 2**30. Query only Tectonic --version, not a manuscript build."])


def fixed_head_snapshot():
    import torch
    import torch.nn.functional as functional
    from sklearn.metrics import roc_auc_score
    s = Snapshot()
    cache_path = PHASE / "runs" / "frozen_meanpool_oracle_ep100_fp32" / "feature_cache" / "Test_s100_r256_fp32_52d1a1812356.pt"
    head_path = PHASE / "checkpoints_hf" / "downstream-heads" / "frozen-meanpool" / "oracle-ep100-head.pt"
    pred_path = ROOT / "results" / "downstream" / "meanpool_sweep_oracle" / "ep100_test_predictions.npz"
    for path in (cache_path, head_path, pred_path):
        s.inputs[str(path)] = file_hash(path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    # The retained val_auc metadata is a NumPy scalar. Permit only those
    # numerical constructors, never unrestricted pickle execution.
    safe = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.dtype, type(np.dtype("float64"))]
    with torch.serialization.safe_globals(safe):
        head = torch.load(head_path, map_location="cpu", weights_only=True)
    weights = head["head"]
    output = []
    with torch.no_grad():
        for batch in cache["features"].split(64):
            x = batch.float().mean(1)
            x = functional.layer_norm(x, (x.shape[-1],), weights["norm.weight"], weights["norm.bias"], 1e-5)
            output.append(torch.sigmoid(functional.linear(x, weights["linear.weight"], weights["linear.bias"])).squeeze(-1).numpy())
    probabilities = np.concatenate(output).astype(np.float64)
    y = cache["labels"].numpy().astype(int)
    with np.load(pred_path, allow_pickle=False) as stored:
        assert np.array_equal(stored["labels"].astype(int), y)
        reference = stored["probs"].astype(np.float64)
    old_order = np.sign(reference[y == 1, None] - reference[y == 0]).astype(np.int8)
    new_order = np.sign(probabilities[y == 1, None] - probabilities[y == 0]).astype(np.int8)
    auc, ref_auc = roc_auc_score(y, probabilities), roc_auc_score(y, reference)
    ancestor = PHASE / "fairvision-glaucoma" / "checkpoint-ep25" / "jepa_patch-random_posfix-ep25.pth.tar"
    s.inputs[str(ancestor)] = file_hash(ancestor)
    s.save("fixed_head_reproduction.json", {
        "auc": float(auc), "reference_auc": float(ref_auc), "delta": float(auc - ref_auc),
        "n_pos": int(sum(y == 1)), "n_neg": int(sum(y == 0)), "pair_count": int(sum(y == 1) * sum(y == 0)),
        "net_concordance_pair_equivalents": float(np.sum(new_order.astype(int) - old_order.astype(int)) / 2),
        "strict_order_flips": int(np.sum(old_order * new_order < 0)),
        "tie_status_changes": int(np.sum((old_order == 0) != (new_order == 0))),
        "all_pair_order_disagreements": int(np.sum(old_order != new_order)),
        "ancestor_bytes": ancestor.stat().st_size, "ancestor_sha256": s.inputs[str(ancestor)],
    }, ["Apply the saved head (no fitting) to memory-mapped cached fp32 features on CPU: mean over slices, LayerNorm eps1e-5, Linear, sigmoid.",
        "Compare positive-negative pair ordering, distinguishing strict flips, tie changes and net AUC-equivalent pair changes.",
        "Hash checkpoint bytes by streaming; no encoder loading/forward pass."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-head", action="store_true")
    args = parser.parse_args()
    subgroup_snapshot()
    geometry_snapshot()
    probability_snapshot()
    environment_snapshot()
    metadata_snapshot()
    probe_parameter_snapshot()
    hardware_snapshot()
    if args.fixed_head:
        fixed_head_snapshot()


if __name__ == "__main__":
    main()
