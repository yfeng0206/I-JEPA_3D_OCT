"""P1: protocol-matched paired statistics from saved test predictions.

Zero GPU. Consumes the 19 `test_predictions.npz` files written by
src/eval_downstream.py and produces every number the manuscript needs:

  - master table of arms x epochs with AUC
  - bootstrap CIs per arm (stratified, paired resampling across arms)
  - DeLong test for every pairwise contrast (correlated ROC, same cases)
  - operating-point metrics: sensitivity, specificity, balanced accuracy, F1
  - PR-AUC, Brier score, calibration (ECE)

Everything is written as JSON so downstream figure/table code and the
verification agent can re-derive rather than trust prose.

Outputs -> D:/jepa_phase0/autopilot_out/p1_stats/
"""
import json
import os
import glob
import numpy as np
from scipy import stats as sps
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_curve, brier_score_loss)

RUNS = r"D:\jepa_phase0\runs"
OUT = r"D:\jepa_phase0\autopilot_out\p1_stats"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260822)
N_BOOT = 10000

# Arm identity is derived from the pretraining checkpoint recorded in
# results.json -> config.model.encoder_checkpoint, NEVER from the probe dir name.
# The probe dirs named `frozen_meanpool_mirage_*` are the ENVELOPE arm.
ARM_BY_PRETRAIN_DIR = {
    "checkpoint-ep25":        ("ancestor", "Shared ancestor (random_posfix ep25 fork point)"),
    "cover_random_ep25":      ("random", "Unguided I-JEPA multi-block random target sampling"),
    "patch_mirage_envelope":  ("envelope", "Targets constrained to coarse retina envelope"),
    "patch_mirage_anatomy":   ("anatomy-v1", "Targets from MIRAGE segmentation layers (v1)"),
    "anatomy_v2_ep25":        ("anatomy-v2", "Targets from MIRAGE segmentation layers (v2/bridge)"),
    "blob_resume_ep56":       ("anatomy-v2", "Targets from MIRAGE segmentation layers (v2/bridge), resumed"),
    "cover_f021_ep25":        ("cover-f021", "Coverage-controlled policy, fraction 0.21"),
}


def parse_epoch(ckpt_path):
    """Pull the pretraining epoch out of the checkpoint filename."""
    base = os.path.basename(ckpt_path)
    import re
    m = re.search(r"ep(\d+)", base)
    return int(m.group(1)) if m else None


def load_runs():
    recs = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "test_predictions.npz"))):
        d = os.path.dirname(p)
        rj = os.path.join(d, "results.json")
        if not os.path.exists(rj):
            continue
        res = json.load(open(rj))
        ck = res["config"]["model"]["encoder_checkpoint"]
        norm = ck.replace("/", "\\")
        pretrain_dir = None
        for key in ARM_BY_PRETRAIN_DIR:
            if "\\%s\\" % key in norm:
                pretrain_dir = key
                break
        if pretrain_dir is None:
            raise RuntimeError("unmapped pretraining dir for %s -> %s" % (d, ck))
        arm, desc = ARM_BY_PRETRAIN_DIR[pretrain_dir]
        z = np.load(p)
        recs.append({
            "probe_dir": os.path.basename(d),
            "arm": arm,
            "arm_desc": desc,
            "pretrain_dir": pretrain_dir,
            "encoder_checkpoint": ck,
            "epoch": parse_epoch(ck),
            "labels": z["labels"].astype(int),
            "probs": z["probs"].astype(float),
            "reported_test_auc": res["test_auc"],
            "best_val_auc": res.get("best_val_auc"),
            "best_epoch": res.get("best_epoch"),
            "protocol": (res["config"]["model"].get("probe_type"),
                         res["config"]["model"].get("head_type"),
                         res.get("probe_depth"),
                         res["config"]["data"].get("num_slices"),
                         res["config"]["training"].get("seed"),
                         res["config"]["model"].get("freeze_encoder"),
                         res["config"]["training"].get("lr_head"),
                         res["config"]["training"].get("epochs")),
        })
    return recs


# ---------------------------------------------------------------- DeLong
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_cov(labels, list_of_scores):
    """Fast DeLong (Sun & Xu 2014). Returns (aucs, covariance matrix)."""
    order = np.argsort(-labels, kind="mergesort")
    lab = labels[order]
    preds = np.vstack([s[order] for s in list_of_scores])
    m = int(lab.sum())          # positives first
    n = len(lab) - m
    k = preds.shape[0]

    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - float(m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]]); sy = np.array([[float(sy)]])
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(labels, s1, s2):
    """Two-sided p-value for AUC(s1) - AUC(s2) on the SAME cases."""
    aucs, cov = delong_cov(labels, [s1, s2])
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(diff), float("nan"), 1.0, (float("nan"), float("nan"))
    se = float(np.sqrt(var))
    z = diff / se
    p = 2.0 * sps.norm.sf(abs(z))
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    return float(diff), se, float(p), (float(ci[0]), float(ci[1]))


# ------------------------------------------------- bootstrap (paired)
def paired_bootstrap(labels, score_dict, n_boot=N_BOOT, rng=RNG):
    """Stratified case-resampling bootstrap.

    The SAME resampled index set is applied to every arm, which preserves the
    pairing and is what makes the difference-CIs valid.
    """
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    names = list(score_dict.keys())
    boot = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])
        y = labels[idx]
        for k in names:
            boot[k][b] = roc_auc_score(y, score_dict[k][idx])
    return boot


def ci_from_boot(v, alpha=0.05):
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ------------------------------------------------- operating points
def op_metrics(labels, probs):
    auc = roc_auc_score(labels, probs)
    fpr, tpr, thr = roc_curve(labels, probs)
    # Youden J on the test set itself is optimistic; we report BOTH the
    # fixed 0.5 threshold and the Youden point, and label them as such.
    j = np.argmax(tpr - fpr)
    thr_j = float(thr[j])

    def at(t):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else float("nan")
        return {"threshold": float(t), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "sensitivity": sens, "specificity": spec, "precision": prec,
                "f1": f1, "balanced_accuracy": (sens + spec) / 2,
                "accuracy": (tp + tn) / len(labels)}

    # expected calibration error, 15 equal-width bins
    bins = np.linspace(0, 1, 16)
    which = np.digitize(probs, bins[1:-1])
    ece = 0.0
    for b in range(15):
        msk = which == b
        if msk.sum() == 0:
            continue
        ece += msk.sum() / len(probs) * abs(probs[msk].mean() - labels[msk].mean())

    return {"auc": float(auc),
            "pr_auc": float(average_precision_score(labels, probs)),
            "brier": float(brier_score_loss(labels, probs)),
            "ece_15bin": float(ece),
            "at_0.5": at(0.5),
            "at_youden": at(thr_j)}


def main():
    recs = load_runs()
    protos = {r["protocol"] for r in recs}
    labels = recs[0]["labels"]
    assert all(np.array_equal(r["labels"], labels) for r in recs), "test sets differ"
    assert len(protos) == 1, "multiple probe protocols: %s" % protos

    print("loaded %d runs | protocol=%s | n=%d pos=%d" %
          (len(recs), protos.pop(), len(labels), int(labels.sum())))

    key = {r["probe_dir"]: r for r in recs}
    scores = {k: v["probs"] for k, v in key.items()}

    # ---- per-run metrics + bootstrap CI
    print("bootstrapping %d resamples x %d runs ..." % (N_BOOT, len(recs)))
    boot = paired_bootstrap(labels, scores)

    table = []
    for r in recs:
        m = op_metrics(labels, r["probs"])
        lo, hi = ci_from_boot(boot[r["probe_dir"]])
        assert abs(m["auc"] - r["reported_test_auc"]) < 1e-9, \
            "recomputed AUC disagrees with results.json for %s" % r["probe_dir"]
        table.append({
            "probe_dir": r["probe_dir"], "arm": r["arm"], "arm_desc": r["arm_desc"],
            "epoch": r["epoch"], "encoder_checkpoint": r["encoder_checkpoint"],
            "auc": m["auc"], "auc_ci95_lo": lo, "auc_ci95_hi": hi,
            "auc_boot_sd": float(boot[r["probe_dir"]].std(ddof=1)),
            "best_val_auc": r["best_val_auc"], "probe_best_epoch": r["best_epoch"],
            **{k: v for k, v in m.items() if k != "auc"},
        })
    table.sort(key=lambda t: (t["arm"], t["epoch"] if t["epoch"] is not None else -1))

    # ---- all pairwise contrasts: DeLong + paired bootstrap difference CI
    contrasts = []
    dirs = [r["probe_dir"] for r in recs]
    for i in range(len(dirs)):
        for j in range(len(dirs)):
            if i >= j:
                continue
            a, b = dirs[i], dirs[j]
            diff, se, p, ci = delong_test(labels, scores[a], scores[b])
            db = boot[a] - boot[b]
            blo, bhi = ci_from_boot(db)
            contrasts.append({
                "a": a, "b": b,
                "arm_a": key[a]["arm"], "epoch_a": key[a]["epoch"],
                "arm_b": key[b]["arm"], "epoch_b": key[b]["epoch"],
                "auc_a": float(roc_auc_score(labels, scores[a])),
                "auc_b": float(roc_auc_score(labels, scores[b])),
                "delta": diff, "delong_se": se, "delong_p": p,
                "delong_ci95": list(ci),
                "boot_delta_mean": float(db.mean()),
                "boot_ci95_lo": blo, "boot_ci95_hi": bhi,
                "boot_p_two_sided": float(2 * min((db <= 0).mean(), (db >= 0).mean())),
                "matched_epoch": key[a]["epoch"] == key[b]["epoch"],
            })

    # Benjamini-Hochberg over the matched-epoch family only (the pre-specified
    # comparisons); reporting FDR over all 171 pairs would be misleading.
    fam = [c for c in contrasts if c["matched_epoch"]]
    if fam:
        ps = np.array([c["delong_p"] for c in fam])
        order = np.argsort(ps)
        n = len(ps)
        q = np.empty(n)
        prev = 1.0
        for rank in range(n - 1, -1, -1):
            idx = order[rank]
            val = ps[idx] * n / (rank + 1)
            prev = min(prev, val)
            q[idx] = prev
        for c, qq in zip(fam, q):
            c["delong_q_bh_matched_epoch_family"] = float(qq)

    out = {
        "generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_test": int(len(labels)), "n_pos": int(labels.sum()), "n_neg": int((labels == 0).sum()),
        "n_runs": len(recs), "n_bootstrap": N_BOOT, "bootstrap_seed": 20260822,
        "protocol_note": "all runs share one probe protocol: mean_pool / linear head / "
                         "probe_depth 2 / 100 slices / seed 42 / frozen encoder / lr_head 4e-4 / 50 epochs",
        "pairing_note": "identical test label vector across all runs, verified by array equality; "
                        "DeLong and paired bootstrap are therefore valid",
        "table": table, "contrasts": contrasts,
    }
    with open(os.path.join(OUT, "p1_master_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(OUT, "bootstrap_draws.npz"), **boot)

    # ---- console summary
    print("\n%-32s %-11s %-5s %8s %-18s %7s %7s" %
          ("probe_dir", "arm", "ep", "AUC", "95% CI", "sens", "spec"))
    for t in table:
        print("%-32s %-11s %-5s %8.4f [%.4f,%.4f] %7.3f %7.3f" %
              (t["probe_dir"], t["arm"], t["epoch"], t["auc"],
               t["auc_ci95_lo"], t["auc_ci95_hi"],
               t["at_0.5"]["sensitivity"], t["at_0.5"]["specificity"]))

    print("\nMatched-epoch contrasts (DeLong):")
    for c in sorted(fam, key=lambda c: (c["epoch_a"], c["arm_a"], c["arm_b"])):
        print("  ep%-4s %-11s vs %-11s  d=%+.4f  p=%.4f  q=%.4f  bootCI[%+.4f,%+.4f]" %
              (c["epoch_a"], c["arm_a"], c["arm_b"], c["delta"], c["delong_p"],
               c.get("delong_q_bh_matched_epoch_family", float("nan")),
               c["boot_ci95_lo"], c["boot_ci95_hi"]))
    print("\nwrote", os.path.join(OUT, "p1_master_stats.json"))


if __name__ == "__main__":
    main()
