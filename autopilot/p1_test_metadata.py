"""P1-04 / P7 prerequisite: build the test-set metadata table.

The downstream test DataLoader uses shuffle=False over
OCTVolumeDataset/OCTSliceDataset, whose file list is `sorted(glob.glob('*.npz'))`
(src/datasets/oct_volumes.py:57-58, oct_slices.py:62-63). Therefore prediction
index i in test_predictions.npz corresponds to the i-th sorted .npz in Test/.

This script materialises that join so subgroup and clustered analyses become
possible, and independently re-derives the label vector so the index alignment
is PROVEN rather than assumed.

Output -> D:/jepa_phase0/autopilot_out/p1_stats/test_metadata.csv
"""
import csv
import glob
import json
import os
import numpy as np

DATA = r"D:\jepa_phase0\fairvision-glaucoma\data\Test"
OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"
os.makedirs(OUT, exist_ok=True)

ATTRS = ["race", "male", "hispanic", "maritalstatus", "language", "glaucoma"]

# Harvard-FairVision categorical codings, as distributed with the dataset.
RACE = {0: "Asian", 1: "Black", 2: "White"}
MALE = {0: "Female", 1: "Male"}
HISP = {0: "Non-Hispanic", 1: "Hispanic"}
MARITAL = {0: "Married or partnered", 1: "Single", 2: "Divorced",
           3: "Widowed", 4: "Legally separated", 5: "Unknown"}
LANG = {0: "English", 1: "Spanish", 2: "Other", 3: "Unknown"}


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*.npz")))
    print("test volumes:", len(files))

    rows = []
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)          # lazy: does not decode oct_bscans
        r = {"index": i, "file": os.path.basename(f),
             "subject_id": os.path.splitext(os.path.basename(f))[0]}
        for a in ATTRS:
            r[a] = int(z[a])
        r["race_label"] = RACE.get(r["race"], "code_%d" % r["race"])
        r["sex_label"] = MALE.get(r["male"], "code_%d" % r["male"])
        r["hispanic_label"] = HISP.get(r["hispanic"], "code_%d" % r["hispanic"])
        r["marital_label"] = MARITAL.get(r["maritalstatus"], "code_%d" % r["maritalstatus"])
        r["language_label"] = LANG.get(r["language"], "code_%d" % r["language"])
        rows.append(r)
        if (i + 1) % 500 == 0:
            print("  %d/%d" % (i + 1, len(files)), flush=True)

    cols = list(rows[0].keys())
    with open(os.path.join(OUT, "test_metadata.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- PROOF of index alignment: the label vector reconstructed from the
    # sorted file order must equal the labels stored in every predictions file.
    y = np.array([r["glaucoma"] for r in rows], dtype=int)
    checked = 0
    mismatch = []
    for p in sorted(glob.glob(r"D:\jepa_phase0\runs\*\test_predictions.npz")):
        lab = np.load(p)["labels"].astype(int)
        if len(lab) != len(y) or not np.array_equal(lab, y):
            mismatch.append(os.path.basename(os.path.dirname(p)))
        checked += 1

    summary = {
        "n_test_volumes": len(files),
        "n_unique_subject_ids": len(set(r["subject_id"] for r in rows)),
        "prevalence_glaucoma": float(y.mean()),
        "index_alignment_proof": {
            "method": "reconstruct label vector from sorted(glob) file order and "
                      "compare to labels stored in every test_predictions.npz",
            "runs_checked": checked,
            "runs_mismatched": mismatch,
            "aligned": len(mismatch) == 0,
        },
        "subgroup_counts": {},
    }
    for a, lbl in [("race_label", "race"), ("sex_label", "sex"),
                   ("hispanic_label", "ethnicity"), ("marital_label", "marital"),
                   ("language_label", "language")]:
        d = {}
        for r in rows:
            k = r[a]
            d.setdefault(k, {"n": 0, "pos": 0})
            d[k]["n"] += 1
            d[k]["pos"] += r["glaucoma"]
        for k in d:
            d[k]["prevalence"] = d[k]["pos"] / d[k]["n"]
        summary["subgroup_counts"][lbl] = d

    with open(os.path.join(OUT, "test_metadata_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    print("\nINDEX ALIGNMENT PROVEN:", summary["index_alignment_proof"]["aligned"],
          "(%d runs checked, %d mismatched)" % (checked, len(mismatch)))
    print("unique subject ids: %d / %d volumes" %
          (summary["n_unique_subject_ids"], len(files)))
    print("glaucoma prevalence: %.4f" % summary["prevalence_glaucoma"])
    for grp, d in summary["subgroup_counts"].items():
        print("\n%s:" % grp)
        for k, v in sorted(d.items(), key=lambda kv: -kv[1]["n"]):
            print("   %-24s n=%-5d pos=%-5d prev=%.3f" % (k, v["n"], v["pos"], v["prevalence"]))


if __name__ == "__main__":
    main()
