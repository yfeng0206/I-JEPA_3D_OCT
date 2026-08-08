"""Paired stratified bootstrap between ANY two arms' saved test predictions.

Generalises scripts/bootstrap_frozen_meanpool.py (which hard-codes oracle vs
random at fixed epochs) so a new arm -- e.g. MIRAGE -- can be compared on the
same 3000-volume FairVision Test split.

The method is identical to the published oracle-vs-random result:
  B=2000 resamples, stratified by class, and the SAME resample indices applied
  to BOTH arms within each replicate. Pairing cancels the shared "which volumes
  got drawn" variance, so the CI is on the DIFFERENCE, not on either AUC.

Both .npz files must come from src/eval_downstream.py (keys: labels, probs) and
must have been produced on the same split in the same order -- the script
asserts this and refuses to run otherwise.

Usage:
    python scripts/bootstrap_paired_arms.py \
        --a results/downstream/meanpool_sweep_mirage/ep100_test_predictions.npz \
        --b results/downstream/meanpool_sweep_oracle/ep100_test_predictions.npz \
        --name-a MIRAGE --name-b oracle

    # several comparisons at once, all against the same reference:
    python scripts/bootstrap_paired_arms.py \
        --a .../mirage.npz --b .../oracle.npz --b .../random.npz
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score


def load(path):
    d = np.load(path)
    missing = {"labels", "probs"} - set(d.files)
    if missing:
        raise KeyError(f"{path}: missing key(s) {sorted(missing)}; expected an "
                       f"eval_downstream.py test_predictions.npz")
    return d["probs"].astype(np.float64), d["labels"].astype(np.int32)


def paired_bootstrap(labels, probs_a, probs_b, b_resamples, seed):
    """Return (deltas, auc_a, auc_b). Same indices scored on both arms."""
    rng = np.random.RandomState(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    deltas = np.empty(b_resamples)
    for i in range(b_resamples):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        y = labels[idx]
        deltas[i] = roc_auc_score(y, probs_a[idx]) - roc_auc_score(y, probs_b[idx])
    return deltas, roc_auc_score(labels, probs_a), roc_auc_score(labels, probs_b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="test_predictions.npz for arm A")
    ap.add_argument("--b", required=True, action="append",
                    help="test_predictions.npz for a comparison arm (repeatable)")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", action="append", default=None,
                    help="label for each --b, in the same order")
    ap.add_argument("--B", type=int, default=2000, help="resamples (default 2000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    names_b = args.name_b or [f"B{i+1}" for i in range(len(args.b))]
    if len(names_b) != len(args.b):
        ap.error(f"{len(args.b)} --b paths but {len(names_b)} --name-b labels")

    pa, la = load(args.a)

    print(f"Paired stratified bootstrap  (B={args.B}, seed={args.seed})")
    print(f"reference arm A = {args.name_a}  <-  {args.a}")
    print(f"n={len(la)}  positives={int(la.sum())}  negatives={int((la == 0).sum())}\n")
    print(f"{'comparison':>22} | {'A':>7} | {'B':>7} | {'delta':>8} | "
          f"{'95% CI on delta':>20} | {'p':>9}")
    print("-" * 92)

    for path_b, name_b in zip(args.b, names_b):
        pb, lb = load(path_b)

        # Pairing is only valid if both arms scored the identical rows in the
        # identical order. Anything else silently invalidates the CI.
        if len(lb) != len(la):
            raise SystemExit(f"ABORT {name_b}: {len(lb)} rows vs {len(la)} in arm A")
        if not np.array_equal(la, lb):
            raise SystemExit(
                f"ABORT {name_b}: label vectors differ from arm A. The two runs "
                f"did not evaluate the same volumes in the same order, so a "
                f"paired test is invalid. Re-run the probe with the same split, "
                f"shuffle=False and the same seed.")

        deltas, auc_a, auc_b = paired_bootstrap(la, pa, pb, args.B, args.seed)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
        p_str = f"<{1/args.B:.4f}" if p == 0 else f"{p:.4f}"
        flag = "ns" if p > 0.05 else ("*" if p > 0.01 else ("**" if p > 0.001 else "***"))

        print(f"{args.name_a + ' - ' + name_b:>22} | {auc_a:.4f} | {auc_b:.4f} | "
              f"{auc_a - auc_b:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p_str:>7} {flag}")

    print("\nThe CI is on the DIFFERENCE. Do not compare it to a marginal "
          "per-model CI:\na marginal CI keeps the shared cohort variance that "
          "pairing cancels, so it is\nroughly 2.5x wider and cannot resolve gaps "
          "this method resolves at p<0.0005.")
    print("\nScope: this measures TEST-SAMPLE uncertainty with both models held "
          "fixed.\nIt says nothing about pretraining-seed variance.")


if __name__ == "__main__":
    main()
