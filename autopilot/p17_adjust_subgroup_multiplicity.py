"""Simultaneous inference for the paper's two subgroup contrast families.

The retained artifacts contain marginal percentile intervals but no
per-contrast p-values.  This script therefore reuses the retained paired
per-case predictions and computes single-step max-|t| bootstrap intervals:

* 10 AUC contrasts: eight stratum deltas and two gain differences, using one
  outcome-stratified subject resample shared across arms and strata.
* 5 sensitivity contrasts: one positive-subject resample shared across arms
  and all overlapping race and sex strata.

Fitted heads and the validation-selected threshold remain fixed.  No GPU is
used.  The output is consumed by ``p8_make_assets.py``.
"""
import argparse
import csv
import json
import os
import time

import numpy as np


REPO = r"C:\Users\Gary\Desktop\jepa"
STATS = r"D:\jepa_phase0\autopilot_out\p1_stats"
META = os.path.join(STATS, "test_metadata.csv")
FAIRVISION = r"D:\jepa_phase0\fairvision-glaucoma\metadata\data_summary_glaucoma.csv"
P7C = os.path.join(STATS, "p7c_paired_subgroup.json")
P16 = os.path.join(REPO, r"results\p16_subgroup_operating.json")
RANDOM = os.path.join(
    REPO, r"results\downstream\meanpool_sweep_random\ep100_test_predictions.npz"
)
INTENSITY = os.path.join(
    REPO, r"results\downstream\meanpool_sweep_oracle\ep100_test_predictions.npz"
)
DEFAULT_OUTPUT = os.path.join(REPO, r"results\p17_subgroup_multiplicity.json")
SEED = 20260826

AUC_GROUPS = (
    "severity:mild",
    "severity:moderate",
    "severity:severe",
    "race:White",
    "race:Black",
    "race:Asian",
    "sex:Female",
    "sex:Male",
)
AUC_CONTRASTS = AUC_GROUPS + ("race:Black-minus-Asian", "sex:Female-minus-Male")
SENSITIVITY_CONTRASTS = (
    "race:Asian",
    "race:Black",
    "race:White",
    "sex:Female",
    "sex:Male",
)


def weighted_auc(weights, y, scores, mask):
    """Tie-correct weighted AUC for every bootstrap row in ``weights``."""
    idx = np.flatnonzero(mask)
    order = idx[np.argsort(scores[idx], kind="stable")]
    ys = y[order]
    ss = scores[order]
    _, starts, counts = np.unique(ss, return_index=True, return_counts=True)
    group = np.repeat(np.arange(len(starts)), counts)
    pos_at = np.flatnonzero(ys == 1)
    pos_group = group[pos_at]
    group_start = starts[pos_group]
    group_end = group_start + counts[pos_group] - 1

    ordered_weights = weights[:, order]
    neg_weights = ordered_weights * (ys == 0)[None, :]
    cumulative_neg = np.cumsum(neg_weights, axis=1, dtype=np.float64)
    through_group = cumulative_neg[:, group_end]
    below_group = np.zeros_like(through_group)
    has_lower = group_start > 0
    below_group[:, has_lower] = cumulative_neg[:, group_start[has_lower] - 1]
    negative_credit = below_group + 0.5 * (through_group - below_group)
    positive_weights = ordered_weights[:, pos_at]
    numerator = np.sum(positive_weights * negative_credit, axis=1)
    denominator = np.sum(positive_weights, axis=1) * cumulative_neg[:, -1]
    return np.divide(
        numerator,
        denominator,
        out=np.full(weights.shape[0], np.nan),
        where=denominator > 0,
    )


def max_t_summary(point, boot, names, unadjusted):
    """Return single-step max-|t| intervals and adjusted bootstrap p-values."""
    if np.isnan(boot).any():
        raise RuntimeError("A bootstrap contrast is undefined")
    se = np.std(boot, axis=0, ddof=1)
    if np.any(se <= 0):
        raise RuntimeError("A bootstrap contrast has zero standard error")
    centered_t = (boot - point[None, :]) / se[None, :]
    max_abs_t = np.max(np.abs(centered_t), axis=1)
    critical = float(np.percentile(max_abs_t, 95))
    n_boot = boot.shape[0]
    out = {}
    for j, name in enumerate(names):
        observed_t = abs(point[j] / se[j])
        raw_p = float(
            (1 + np.count_nonzero(np.abs(centered_t[:, j]) >= observed_t))
            / (n_boot + 1)
        )
        adjusted_p = float(
            (1 + np.count_nonzero(max_abs_t >= observed_t)) / (n_boot + 1)
        )
        lo = float(point[j] - critical * se[j])
        hi = float(point[j] + critical * se[j])
        before = unadjusted[name]
        out[name] = {
            "estimate": float(point[j]),
            "unadjusted_ci95_lo": float(before["lo"]),
            "unadjusted_ci95_hi": float(before["hi"]),
            "unadjusted_excludes_zero": bool(before["excludes_zero"]),
            "bootstrap_se": float(se[j]),
            "unadjusted_bootstrap_p": raw_p,
            "max_t_adjusted_p": adjusted_p,
            "simultaneous_ci95_lo": lo,
            "simultaneous_ci95_hi": hi,
            "simultaneous_excludes_zero": bool(lo > 0 or hi < 0),
            "conclusion_changed": bool(before["excludes_zero"] != (lo > 0 or hi < 0)),
        }
    return critical, out


def load_inputs():
    with open(META, newline="", encoding="utf-8") as f:
        meta = sorted(csv.DictReader(f), key=lambda r: int(r["index"]))
    y = np.array([int(r["glaucoma"]) for r in meta], dtype=np.int8)

    zr = np.load(RANDOM)
    zi = np.load(INTENSITY)
    yr = zr["labels"].astype(np.int8)
    yi = zi["labels"].astype(np.int8)
    if not np.array_equal(y, yr) or not np.array_equal(y, yi):
        raise RuntimeError("Retained prediction labels do not match the metadata join")
    random_scores = zr["probs"].astype(np.float64)
    intensity_scores = zi["probs"].astype(np.float64)

    md_by_file = {}
    with open(FAIRVISION, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            md_by_file[row["filename"]] = float(row["md"])
    md = np.array([md_by_file[row["file"]] for row in meta])
    return meta, y, random_scores, intensity_scores, md


def auc_family(meta, y, random_scores, intensity_scores, md, n_boot, rng, batch_size):
    neg = y == 0
    masks = {
        "severity:mild": ((md > -6) & (md <= -2) & (y == 1)) | neg,
        "severity:moderate": ((md > -12) & (md <= -6) & (y == 1)) | neg,
        "severity:severe": ((md > -100) & (md <= -12) & (y == 1)) | neg,
    }
    for value in ("White", "Black", "Asian"):
        masks["race:" + value] = np.array([r["race_label"] == value for r in meta])
    for value in ("Female", "Male"):
        masks["sex:" + value] = np.array([r["sex_label"] == value for r in meta])

    all_weights = np.ones((1, len(y)), dtype=np.uint16)
    point_group = {}
    for name in AUC_GROUPS:
        point_group[name] = float(
            weighted_auc(all_weights, y, intensity_scores, masks[name])[0]
            - weighted_auc(all_weights, y, random_scores, masks[name])[0]
        )
    point = np.array(
        [point_group[name] for name in AUC_GROUPS]
        + [
            point_group["race:Black"] - point_group["race:Asian"],
            point_group["sex:Female"] - point_group["sex:Male"],
        ]
    )

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    pos_prob = np.full(len(pos_idx), 1.0 / len(pos_idx))
    neg_prob = np.full(len(neg_idx), 1.0 / len(neg_idx))
    boot = np.empty((n_boot, len(AUC_CONTRASTS)))
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        size = stop - start
        weights = np.zeros((size, len(y)), dtype=np.uint16)
        weights[:, pos_idx] = rng.multinomial(len(pos_idx), pos_prob, size=size)
        weights[:, neg_idx] = rng.multinomial(len(neg_idx), neg_prob, size=size)
        group_boot = {}
        for name in AUC_GROUPS:
            group_boot[name] = (
                weighted_auc(weights, y, intensity_scores, masks[name])
                - weighted_auc(weights, y, random_scores, masks[name])
            )
        boot[start:stop, : len(AUC_GROUPS)] = np.column_stack(
            [group_boot[name] for name in AUC_GROUPS]
        )
        boot[start:stop, 8] = group_boot["race:Black"] - group_boot["race:Asian"]
        boot[start:stop, 9] = group_boot["sex:Female"] - group_boot["sex:Male"]

    baseline = json.load(open(P7C, encoding="utf-8"))["contrasts"]["intensity_minus_random"]
    unadjusted = {}
    for name in AUC_GROUPS:
        value = baseline["per_group"][name]
        if abs(value["delta_auc"] - point[AUC_CONTRASTS.index(name)]) > 1e-12:
            raise RuntimeError("AUC point estimate does not reproduce p7c for " + name)
        unadjusted[name] = {
            "lo": value["ci95_lo"],
            "hi": value["ci95_hi"],
            "excludes_zero": value["excludes_zero"],
        }
    for axis, name in (
        ("race", "race:Black-minus-Asian"),
        ("sex", "sex:Female-minus-Male"),
    ):
        value = baseline[axis + "_differential_benefit"]
        if abs(value["delta_worst_minus_delta_best"] - point[AUC_CONTRASTS.index(name)]) > 1e-12:
            raise RuntimeError("AUC gain difference does not reproduce p7c for " + axis)
        unadjusted[name] = {
            "lo": value["ci95_lo"],
            "hi": value["ci95_hi"],
            "excludes_zero": value["excludes_zero"],
        }
    critical, results = max_t_summary(point, boot, AUC_CONTRASTS, unadjusted)
    return critical, results


def sensitivity_family(
    meta, y, random_scores, intensity_scores, n_boot, rng, threshold, batch_size
):
    positive = y == 1
    individual_delta = (
        (intensity_scores >= threshold).astype(np.int8)
        - (random_scores >= threshold).astype(np.int8)
    )
    masks = {}
    for value in ("Asian", "Black", "White"):
        masks["race:" + value] = positive & np.array(
            [r["race_label"] == value for r in meta]
        )
    for value in ("Female", "Male"):
        masks["sex:" + value] = positive & np.array(
            [r["sex_label"] == value for r in meta]
        )
    point = np.array([individual_delta[masks[name]].mean() for name in SENSITIVITY_CONTRASTS])

    positive_idx = np.flatnonzero(positive)
    positive_delta = individual_delta[positive_idx]
    positive_masks = {
        name: masks[name][positive_idx] for name in SENSITIVITY_CONTRASTS
    }
    probabilities = np.full(len(positive_idx), 1.0 / len(positive_idx))
    boot = np.empty((n_boot, len(SENSITIVITY_CONTRASTS)))
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        weights = rng.multinomial(
            len(positive_idx), probabilities, size=stop - start
        )
        for j, name in enumerate(SENSITIVITY_CONTRASTS):
            mask = positive_masks[name]
            denominator = np.sum(weights[:, mask], axis=1)
            boot[start:stop, j] = (
                np.sum(weights[:, mask] * positive_delta[mask], axis=1)
                / denominator
            )

    baseline = json.load(open(P16, encoding="utf-8"))["delta_random_to_intensity"]
    unadjusted = {}
    for name in SENSITIVITY_CONTRASTS:
        axis, value_name = name.split(":")
        value = baseline[axis][value_name]
        if abs(value["d_sensitivity"] - point[SENSITIVITY_CONTRASTS.index(name)]) > 1e-12:
            raise RuntimeError("Sensitivity point estimate does not reproduce p16 for " + name)
        unadjusted[name] = {
            "lo": value["ci95_lo"],
            "hi": value["ci95_hi"],
            "excludes_zero": value["excludes_zero"],
        }
    critical, results = max_t_summary(point, boot, SENSITIVITY_CONTRASTS, unadjusted)
    return critical, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        parser.error("--n-bootstrap must be at least 100")

    started = time.perf_counter()
    meta, y, random_scores, intensity_scores, md = load_inputs()
    p16 = json.load(open(P16, encoding="utf-8"))
    rng_auc = np.random.default_rng(SEED)
    rng_sensitivity = np.random.default_rng(SEED + 1)
    auc_critical, auc_results = auc_family(
        meta,
        y,
        random_scores,
        intensity_scores,
        md,
        args.n_bootstrap,
        rng_auc,
        args.batch_size,
    )
    sensitivity_critical, sensitivity_results = sensitivity_family(
        meta,
        y,
        random_scores,
        intensity_scores,
        args.n_bootstrap,
        rng_sensitivity,
        p16["threshold"],
        args.batch_size,
    )
    elapsed = time.perf_counter() - started
    output = {
        "generated": __import__("datetime").datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "method": "single-step max-|t| simultaneous paired bootstrap intervals",
        "alpha": 0.05,
        "n_bootstrap": args.n_bootstrap,
        "auc_family": {
            "family_size": 10,
            "seed": SEED,
            "resampling": "one outcome-stratified subject resample shared across both fixed heads and every stratum",
            "max_abs_t_critical": auc_critical,
            "contrasts": auc_results,
            "n_survive": sum(
                value["simultaneous_excludes_zero"] for value in auc_results.values()
            ),
        },
        "sensitivity_family": {
            "family_size": 5,
            "seed": SEED + 1,
            "threshold": p16["threshold"],
            "resampling": "one positive-subject resample shared across both fixed heads and all overlapping race and sex strata",
            "max_abs_t_critical": sensitivity_critical,
            "contrasts": sensitivity_results,
            "n_survive": sum(
                value["simultaneous_excludes_zero"]
                for value in sensitivity_results.values()
            ),
        },
        "runtime_seconds": elapsed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1)
    print(
        "AUC: %d/10 survive; sensitivity: %d/5 survive; %.1f seconds"
        % (
            output["auc_family"]["n_survive"],
            output["sensitivity_family"]["n_survive"],
            elapsed,
        )
    )
    print("wrote", args.output)


if __name__ == "__main__":
    main()
