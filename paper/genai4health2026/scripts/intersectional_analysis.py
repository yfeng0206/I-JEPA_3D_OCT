"""Race-by-gender intersectional subgroup AUC for frozen-probe predictions.

Companion to ``subgroup_analysis.py``, which reports only *marginal* subgroups
(race alone, gender alone, ...). The 2024 GenAI4Health oral "Demographic Bias of
Expert-Level Vision-Language Foundation Models in Medical Imaging" reported
intersectional cells (e.g. Black female) rather than marginal ones only, and
that granularity is what this script adds.

Nothing in ``subgroup_analysis.py`` is modified or re-run: the verified section-8
pipeline stays byte-identical. This script imports its helpers (join proof,
AUC, bootstrap CI, arm discovery) and only changes how rows are bucketed.

Feasibility was checked before writing this: all six race x gender cells in the
3000-volume test split contain both classes, the smallest being asian x female
at n=123. Cells are still reported with bootstrap CIs so the small ones can be
read with appropriate caution rather than silently dropped.

Zero GPU. Outputs JSON + CSV to the report directory.
"""

import argparse
import csv
import json
import os
from collections import OrderedDict

import numpy as np

import subgroup_analysis as S

# Attributes crossed to form cells. Order fixes the printed label "race x gender".
AXES = ("race", "gender")
# Levels that carry no demographic meaning and are dropped from either axis.
_SKIP = ("", "unknown", "na")


def intersectional_cells(rows):
    """Yield (label, boolean index array) for each populated race x gender cell."""
    vals = [np.array([r[a].strip().lower() for r in rows]) for a in AXES]
    levels = [sorted(v for v in set(col) if v not in _SKIP) for col in vals]
    for a in levels[0]:
        for b in levels[1]:
            idx = (vals[0] == a) & (vals[1] == b)
            if idx.any():
                yield "%s x %s" % (a, b), idx


def analyse_arm(pred_path, files, rows, min_n, n_boot):
    """Verify the order-based join, then compute per-cell AUC."""
    if not os.path.exists(pred_path):
        return None
    z = np.load(pred_path, allow_pickle=True)
    labels = np.asarray(z["labels"]).reshape(-1).astype(int)
    probs = np.asarray(z["probs"]).reshape(-1).astype(float)
    if len(labels) != len(files):
        return {"error": "length mismatch: %d preds vs %d test files"
                         % (len(labels), len(files))}

    # Same join-integrity proof as the marginal analysis: the metadata label
    # column must reproduce the stored labels exactly, or the arm is refused.
    csv_lab = np.array([1 if r["glaucoma"].strip().lower() == "yes" else 0
                        for r in rows])
    agree = int((csv_lab == labels).sum())
    if agree != len(labels):
        return {"error": "join unverified: metadata labels match only %d/%d"
                         % (agree, len(labels)),
                "label_agreement": agree / len(labels)}

    entries, skipped = [], []
    for name, idx in intersectional_cells(rows):
        n = int(idx.sum())
        y, p = labels[idx], probs[idx]
        if n < min_n or len(set(y.tolist())) < 2:
            skipped.append({"subgroup": name, "n": n,
                            "reason": "below min_n" if n < min_n
                                      else "single class"})
            continue
        lo, hi = S.auc_ci(y, p, n_boot=n_boot)
        entries.append({"subgroup": name, "n": n,
                        "n_pos": int((y == 1).sum()),
                        "n_neg": int((y == 0).sum()),
                        "auc": S.auc_score(y, p), "ci_lo": lo, "ci_hi": hi})

    out = {"n": len(labels),
           "status": S.arm_status(os.path.basename(os.path.dirname(pred_path))),
           "label_agreement": 1.0,
           "overall_auc": S.auc_score(labels, probs),
           "prevalence": float(labels.mean()),
           "axes": " x ".join(AXES),
           "cells": entries,
           "skipped": skipped}
    if entries:
        aucs = [e["auc"] for e in entries]
        out["gap"] = max(aucs) - min(aucs)
        out["worst"] = min(aucs)
        out["worst_group"] = entries[int(np.argmin(aucs))]["subgroup"]
        out["best_group"] = entries[int(np.argmax(aucs))]["subgroup"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="probe dir names under D:/jepa_phase0/runs")
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\subgroup")
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    files, rows = S.build_test_table()
    print("test split: %d volumes; metadata joined" % len(files))
    counts = [(name, int(idx.sum())) for name, idx in intersectional_cells(rows)]
    print("cells: " + ", ".join("%s n=%d" % c for c in counts))

    if args.runs:
        names = args.runs
    else:
        names = sorted(d for d in os.listdir(S.RUNS_ROOT)
                       if d.startswith(("frozen_meanpool_", "frozen_cover_"))
                       and os.path.exists(os.path.join(S.RUNS_ROOT, d,
                                                       "test_predictions.npz")))
        names += list(S.EXTRA_PROBES)

    def resolve(name):
        if name in S.EXTRA_PROBES:
            return S.EXTRA_PROBES[name]
        return os.path.join(S.RUNS_ROOT, name, "test_predictions.npz")

    os.makedirs(args.out, exist_ok=True)
    results, flat = OrderedDict(), []
    for name in names:
        res = analyse_arm(resolve(name), files, rows, args.min_n, args.n_boot)
        if res is None:
            continue
        results[name] = res
        if "error" in res:
            print("  %-42s SKIPPED  %s" % (name, res["error"]))
            continue
        print("  %-42s overall AUC %.4f  [%s]  gap %.4f  worst=%s %.4f"
              % (name, res["overall_auc"], res["status"], res["gap"],
                 res["worst_group"], res["worst"]))
        for e in res["cells"]:
            print("      %-18s n=%4d pos=%4d  AUC %.4f  [%.4f, %.4f]"
                  % (e["subgroup"], e["n"], e["n_pos"], e["auc"],
                     e["ci_lo"], e["ci_hi"]))
            flat.append({"arm": name, "status": res["status"],
                         "axes": res["axes"], "subgroup": e["subgroup"],
                         "n": e["n"], "n_pos": e["n_pos"], "n_neg": e["n_neg"],
                         "auc": "%.4f" % e["auc"],
                         "ci_lo": "%.4f" % e["ci_lo"],
                         "ci_hi": "%.4f" % e["ci_hi"],
                         "overall_auc": "%.4f" % res["overall_auc"]})

    with open(os.path.join(args.out, "intersectional_auc.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    if flat:
        with open(os.path.join(args.out, "intersectional_auc.csv"), "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
