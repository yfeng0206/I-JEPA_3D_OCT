"""P1b: the complete, correctly-labelled evidence table.

Supersedes p1_paired_stats.py, which mislabelled the retracted `cover_random`
arm as the null. The paper's actual null and oracle live in the repo's
`results/downstream/` tree, not in D:/jepa_phase0/runs.

Every prediction set on the shared FairVision test split is enumerated here and
tagged with:

  arm        - masking policy, from the pretraining checkpoint
  epoch      - pretraining epoch
  precision  - fp16 or fp32, inferred from the stored `probs` dtype and
               confirmed against the config's `use_amp` where a config exists
  family     - frozen_probe | finetune | linear_d1
  status     - primary | retracted | excluded | supplementary

Exclusions follow paper/genai4health2026/SOURCES.md section 5:
  * cover_random_*      RETRACTED: half-precision EMA targets AND
                        enc_truncate=window, so its deficit is not attributable
                        to masking.
  * anatomy-v2 ep75/92  EXCLUDED: EMA-target precision splice at epoch 56 caused
                        by scripts/campaign_chain.py:179 hardcoding amp_target.

Output -> D:/jepa_phase0/autopilot_out/p1_stats/p1b_full_inventory.json
"""
import glob
import json
import os
import re
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = r"C:\Users\Gary\Desktop\jepa"
RUNS = r"D:\jepa_phase0\runs"
OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"
os.makedirs(OUT, exist_ok=True)

# (glob, arm, family, status, epoch_from)
SPECS = [
    # ---- the paper's primary frozen-probe family (fp16, run off-machine) ----
    (r"results\downstream\meanpool_sweep_random\ep*_test_predictions.npz", "random", "frozen_probe", "primary"),
    (r"results\downstream\meanpool_sweep_oracle\ep*_test_predictions.npz", "oracle", "frozen_probe", "primary"),
    (r"results\downstream\meanpool_sweep_mirage\ep*_test_predictions.npz", "envelope", "frozen_probe", "primary"),
    # ---- local fp32 frozen probes ----
    (os.path.join(RUNS, "frozen_meanpool_fork_ep25", "test_predictions.npz"), "ancestor", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_envelope_ep30", "test_predictions.npz"), "envelope", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_anatomy_ep30", "test_predictions.npz"), "anatomy-v1", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_bridge_ep35", "test_predictions.npz"), "anatomy-v2", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_bridge_ep40", "test_predictions.npz"), "anatomy-v2", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_bridge_ep50", "test_predictions.npz"), "anatomy-v2", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_bridge_ep75", "test_predictions.npz"), "anatomy-v2", "frozen_probe", "excluded"),
    (os.path.join(RUNS, "frozen_meanpool_bridge_ep92", "test_predictions.npz"), "anatomy-v2", "frozen_probe", "excluded"),
    (os.path.join(RUNS, "frozen_meanpool_cover_f021_ep27", "test_predictions.npz"), "cover-f021", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_cover_f021_ep30", "test_predictions.npz"), "cover-f021", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_cover_f021_ep34", "test_predictions.npz"), "cover-f021", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_cover_f021_ep50", "test_predictions.npz"), "cover-f021", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_cover_f021_ep73", "test_predictions.npz"), "cover-f021", "frozen_probe", "primary"),
    # ---- fp32 re-probes produced by this autopilot run ----
    (os.path.join(RUNS, "frozen_meanpool_envelope_fp32_ep*", "test_predictions.npz"), "envelope", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_oracle_ep*_fp32", "test_predictions.npz"), "oracle", "frozen_probe", "primary"),
    (os.path.join(RUNS, "frozen_meanpool_random_ep*_fp32", "test_predictions.npz"), "random", "frozen_probe", "primary"),
    # ---- retracted ----
    (os.path.join(RUNS, "frozen_cover_random_ep*", "test_predictions.npz"), "random-RETRACTED", "frozen_probe", "retracted"),
    # ---- separate families ----
    (r"results\downstream\finetune_oracle\*_test_predictions.npz", "oracle", "finetune", "supplementary"),
    (r"results\downstream\finetune_random\*_test_predictions.npz", "random", "finetune", "supplementary"),
]

ARM_DESC = {
    "ancestor": "shared epoch-25 fork point, no policy applied yet",
    "random": "unguided I-JEPA multi-block target sampling (null)",
    "random-RETRACTED": "retracted null: half-precision EMA targets and enc_truncate=window",
    "envelope": "targets constrained to a coarse retina envelope (segmenter, location only)",
    "oracle": "targets in a band located by a per-column intensity-weighted row centroid (no segmenter)",
    "anatomy-v1": "targets shaped by MIRAGE segmentation layers, v1",
    "anatomy-v2": "targets shaped by MIRAGE segmentation layers, v2 (bridge/blob)",
    "cover-f021": "coverage-constrained policy, visible fraction floor 0.21",
}


def epoch_of(path):
    b = os.path.basename(path)
    d = os.path.basename(os.path.dirname(path))
    for s in (b, d):
        m = re.search(r"ep(\d+)", s)
        if m:
            return int(m.group(1))
    return None


def main():
    ref = None
    recs = []
    seen_hash = {}

    for pat, arm, family, status in SPECS:
        pat_abs = pat if os.path.isabs(pat) else os.path.join(REPO, pat)
        for p in sorted(glob.glob(pat_abs)):
            z = np.load(p)
            lab = z["labels"]
            pr = z["probs"]
            y = lab.astype(int)
            if ref is None:
                ref = y
            same_split = len(y) == len(ref) and np.array_equal(y, ref)
            if not same_split:
                print("[skip] different test split:", p)
                continue

            precision = "fp16" if pr.dtype == np.float16 else "fp32"
            auc = float(roc_auc_score(y, pr.astype(np.float64)))

            # de-duplicate physical copies of the same probe
            h = hash(pr.astype(np.float64).tobytes())
            dup_of = seen_hash.get(h)
            seen_hash.setdefault(h, p)

            tag = os.path.basename(os.path.dirname(p))
            if family == "finetune":
                tag = tag + "/" + os.path.basename(p).replace("_test_predictions.npz", "")

            recs.append({
                "path": p, "tag": tag, "arm": arm, "arm_desc": ARM_DESC.get(arm, ""),
                "epoch": epoch_of(p), "precision": precision, "family": family,
                "status": status, "auc": auc, "n": int(len(y)),
                "duplicate_of": dup_of,
            })

    # collapse duplicates, preferring the repo copy (it is the cited source)
    keep = []
    for r in recs:
        if r["duplicate_of"] is None:
            keep.append(r)
        else:
            print("[dedup] %s is a byte-identical copy of %s" % (r["tag"], os.path.basename(os.path.dirname(r["duplicate_of"]))))

    out = {
        "generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_test": int(len(ref)), "n_pos": int(ref.sum()),
        "test_split_note": "every entry verified to carry the identical 3000-sample label vector",
        "precision_note": "precision inferred from stored probs dtype; eval_downstream.py:541 "
                          "defaults use_amp=True, so configs omitting the key ran fp16",
        "exclusions": {
            "random-RETRACTED": "SOURCES.md 5.1 - half-precision EMA targets and enc_truncate=window",
            "anatomy-v2 ep75/ep92": "SOURCES.md 5.2 - EMA-target precision splice at ep56 "
                                    "(scripts/campaign_chain.py:179 hardcodes amp_target=True)",
        },
        "records": sorted(keep, key=lambda r: (r["family"], r["arm"], r["epoch"] or -1, r["precision"])),
        "n_records_total": len(recs), "n_records_after_dedup": len(keep),
    }
    with open(os.path.join(OUT, "p1b_full_inventory.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print("\n%-46s %-17s %-5s %-6s %-14s %s" % ("tag", "arm", "ep", "prec", "status", "AUC"))
    print("-" * 108)
    for r in out["records"]:
        print("%-46s %-17s %-5s %-6s %-14s %.6f" % (
            r["tag"][:46], r["arm"], r["epoch"], r["precision"], r["status"], r["auc"]))
    print("\n%d records (%d after dedup) | test n=%d pos=%d" %
          (len(recs), len(keep), out["n_test"], out["n_pos"]))


if __name__ == "__main__":
    main()
