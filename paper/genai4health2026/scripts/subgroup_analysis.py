"""Post-hoc subgroup / fairness analysis of frozen-probe predictions.

Zero GPU. Joins each probe's saved ``test_predictions.npz`` (labels, probs) to
the FairVision demographic table by *sorted test filename order*, then reports
per-subgroup AUC.

The join is order-based, so it is verified before use: the ``glaucoma`` column
of the metadata CSV must reproduce the stored ``labels`` vector exactly. If it
does not, the arm is refused rather than reported.

Outputs JSON + CSV to the report directory.
"""

import argparse
import csv
import json
import os
from collections import OrderedDict

import numpy as np

DATA_ROOT = r"D:\jepa_phase0\fairvision-glaucoma"
RUNS_ROOT = r"D:\jepa_phase0\runs"
CSV_PATH = os.path.join(DATA_ROOT, "metadata", "data_summary_glaucoma.csv")
TEST_DIR = os.path.join(DATA_ROOT, "data", "Test")

# Probes whose predictions live in the repo tree rather than under RUNS_ROOT,
# stored as `ep{N}_test_predictions.npz` inside a shared sweep directory. The
# auto-discovery glob below only matches `<dir>/test_predictions.npz`, so these
# were silently skipped; they are the six random/oracle long-horizon probes.
#
# The repo also holds `meanpool_sweep_mirage/ep{50,75,100}`, which are
# byte-equivalent duplicates of `frozen_meanpool_mirage_ep{50,75,100}` under
# RUNS_ROOT (verified: identical test AUC to 8 dp). They are deliberately NOT
# listed here, so each unique probe is counted exactly once.
_SWEEP_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "results", "downstream")
EXTRA_PROBES = OrderedDict(
    (("sweep_%s_ep%d" % (arm, ep),
      os.path.join(_SWEEP_ROOT, "meanpool_sweep_%s" % arm,
                   "ep%d_test_predictions.npz" % ep))
     for arm in ("random", "oracle") for ep in (50, 75, 100)))

# Subgroup definitions: column -> how to bucket it.
CATEGORICAL = ["gender", "race", "ethnicity", "language", "maritalstatus"]
AGE_BINS = [(0, 60, "<60"), (60, 70, "60-69"), (70, 80, "70-79"), (80, 200, "80+")]
# Visual-field mean deviation (dB); less negative = milder loss.
#
# NOTE: in FairVision `md` *defines* the glaucoma label -- every volume with
# md <= -2 is positive and every md > -2 is negative (corr = -0.73, and the
# bins are perfectly class-pure). So md cannot be used as an ordinary subgroup
# axis: within-bin AUC is undefined. Instead we use it for severity-stratified
# detection, scoring each positive stratum against the SHARED pool of all
# negatives. That measures whether mild/early disease is detected as reliably
# as advanced disease -- clinically the question that matters most.
MD_BINS = [(-100, -12, "severe (<=-12)"), (-12, -6, "moderate (-12,-6]"),
           (-6, -2, "mild (-6,-2]")]

# Provenance status per probe directory. The `frozen_cover_random_*` family is
# the RETRACTED COVER campaign: those encoders were pretrained with
# `enc_truncate: window` and `amp_target: true`, so their representations are
# contaminated and must not be cited as evidence. They are still analysed here
# (the join is valid) but are tagged so downstream consumers can exclude them.
# See docs/experiments/masking/cover_random_campaign.md#L22 (retraction).
ARM_STATUS = {
    "frozen_cover_random_ep30": "RETRACTED",
    "frozen_cover_random_ep50": "RETRACTED",
    "frozen_cover_random_ep75": "RETRACTED",
    "frozen_cover_random_ep100": "RETRACTED",
}


def arm_status(name):
    return ARM_STATUS.get(name, "OK")


def _avg_ranks(p):
    """Average (tie-corrected) ranks, fully vectorised."""
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    # Start index of each run of equal values, plus a terminating sentinel.
    bounds = np.flatnonzero(np.concatenate(
        ([True], sp[1:] != sp[:-1], [True])))
    starts, ends = bounds[:-1], bounds[1:]
    avg = 0.5 * (starts + ends - 1) + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def auc_score(y, p):
    """Rank-based AUC (Mann-Whitney U), ties averaged. No sklearn dependency."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_ranks(p)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_ci(y, p, n_boot=2000, seed=0):
    """Percentile bootstrap CI for AUC, resampling cases and controls separately."""
    y = np.asarray(y)
    p = np.asarray(p)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        vals[b] = auc_score(y[idx], p[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_metadata():
    """Return test-split rows keyed by filename, in CSV order."""
    rows = {}
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rows[r["filename"]] = r
    return rows


def build_test_table():
    """Align sorted Test/*.npz filenames with their metadata rows."""
    files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".npz"))
    meta = load_metadata()
    missing = [f for f in files if f not in meta]
    if missing:
        raise SystemExit("%d test files absent from metadata CSV (e.g. %s)"
                         % (len(missing), missing[:3]))
    return files, [meta[f] for f in files]


def bucket(rows, key):
    """Yield (subgroup_label, boolean index array) for one attribute."""
    n = len(rows)
    if key == "age":
        age = np.array([float(r["age"]) for r in rows])
        for lo, hi, name in AGE_BINS:
            yield name, (age >= lo) & (age < hi)
    else:
        vals = np.array([r[key].strip().lower() for r in rows])
        for v in sorted(set(vals)):
            if v in ("", "unknown", "na"):
                continue
            yield v, (vals == v)
    assert n == len(rows)


def analyse_arm(pred_path, files, rows, min_n, n_boot):
    """Verify the order-based join, then compute per-subgroup AUC."""
    if not os.path.exists(pred_path):
        return None
    z = np.load(pred_path, allow_pickle=True)
    labels = np.asarray(z["labels"]).reshape(-1).astype(int)
    probs = np.asarray(z["probs"]).reshape(-1).astype(float)
    if len(labels) != len(files):
        return {"error": "length mismatch: %d preds vs %d test files"
                         % (len(labels), len(files))}

    # --- join integrity proof -------------------------------------------------
    csv_lab = np.array([1 if r["glaucoma"].strip().lower() == "yes" else 0
                        for r in rows])
    agree = int((csv_lab == labels).sum())
    if agree != len(labels):
        return {"error": "join unverified: metadata labels match only %d/%d"
                         % (agree, len(labels)),
                "label_agreement": agree / len(labels)}

    out = {"n": len(labels),
           "status": arm_status(os.path.basename(os.path.dirname(pred_path))),
           "label_agreement": 1.0,
           "overall_auc": auc_score(labels, probs),
           "prevalence": float(labels.mean()),
           "subgroups": OrderedDict()}
    for key in CATEGORICAL + ["age"]:
        entries = []
        for name, idx in bucket(rows, key):
            n = int(idx.sum())
            if n < min_n:
                continue
            y, p = labels[idx], probs[idx]
            if len(set(y.tolist())) < 2:
                continue
            lo, hi = auc_ci(y, p, n_boot=n_boot)
            entries.append({"subgroup": name, "n": n,
                            "n_pos": int((y == 1).sum()),
                            "auc": auc_score(y, p), "ci_lo": lo, "ci_hi": hi})
        if entries:
            aucs = [e["auc"] for e in entries]
            out["subgroups"][key] = {
                "levels": entries,
                "gap": max(aucs) - min(aucs),
                "worst": min(aucs),
                "worst_group": entries[int(np.argmin(aucs))]["subgroup"],
            }
    # Severity-stratified detection: positives of each stratum vs ALL negatives.
    md = np.array([float(r["md"]) for r in rows])
    neg = labels == 0
    sev = []
    for lo, hi, name in MD_BINS:
        sel = (md > lo) & (md <= hi) & (labels == 1)
        n = int(sel.sum())
        if n < min_n:
            continue
        idx = sel | neg
        lo_ci, hi_ci = auc_ci(labels[idx], probs[idx], n_boot=n_boot)
        sev.append({"subgroup": name, "n": n, "n_pos": n,
                    "n_neg": int(neg.sum()),
                    "auc": auc_score(labels[idx], probs[idx]),
                    "ci_lo": lo_ci, "ci_hi": hi_ci})
    if sev:
        aucs = [e["auc"] for e in sev]
        out["subgroups"]["severity"] = {
            "levels": sev,
            "gap": max(aucs) - min(aucs),
            "worst": min(aucs),
            "worst_group": sev[int(np.argmin(aucs))]["subgroup"],
            "note": "positives of each stratum scored against all negatives",
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="probe dir names under D:/jepa_phase0/runs")
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\subgroup")
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    files, rows = build_test_table()
    print("test split: %d volumes; metadata joined" % len(files))

    if args.runs:
        names = args.runs
    else:
        names = sorted(d for d in os.listdir(RUNS_ROOT)
                       if d.startswith(("frozen_meanpool_", "frozen_cover_"))
                       and os.path.exists(os.path.join(RUNS_ROOT, d,
                                                       "test_predictions.npz")))
        names += list(EXTRA_PROBES)

    def resolve(name):
        if name in EXTRA_PROBES:
            return EXTRA_PROBES[name]
        return os.path.join(RUNS_ROOT, name, "test_predictions.npz")

    os.makedirs(args.out, exist_ok=True)
    results, flat = {}, []
    for name in names:
        res = analyse_arm(resolve(name), files, rows,
                          args.min_n, args.n_boot)
        if res is None:
            continue
        results[name] = res
        if "error" in res:
            print("  %-42s SKIPPED  %s" % (name, res["error"]))
            continue
        print("  %-42s overall AUC %.4f  [%s]  (join verified %d/%d)"
              % (name, res["overall_auc"], res["status"], res["n"], res["n"]))
        for key, blk in res["subgroups"].items():
            print("      %-14s gap %.4f  worst=%s %.4f"
                  % (key, blk["gap"], blk["worst_group"], blk["worst"]))
            for e in blk["levels"]:
                flat.append({"arm": name, "status": res["status"],
                             "attribute": key,
                             "subgroup": e["subgroup"], "n": e["n"],
                             "n_pos": e["n_pos"], "auc": "%.4f" % e["auc"],
                             "ci_lo": "%.4f" % e["ci_lo"],
                             "ci_hi": "%.4f" % e["ci_hi"],
                             "overall_auc": "%.4f" % res["overall_auc"]})

    with open(os.path.join(args.out, "subgroup_auc.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    if flat:
        with open(os.path.join(args.out, "subgroup_auc.csv"), "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
