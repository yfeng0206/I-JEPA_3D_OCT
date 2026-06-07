"""Paired stratified bootstrap: oracle vs random frozen MeanPool, per epoch.

Same method as docs/experiments/frozen/ablation_analysis.md — B=2000 resamples,
stratified by class, SAME resample indices for both models each iteration
(correlated-AUC / paired test on the shared 3000-volume FairVision Test split).

Reads test_predictions.npz ({labels, probs}) from each run.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

ORACLE = "results/downstream/meanpool_sweep_oracle/ep{ep}_test_predictions.npz"
RANDOM = "results/downstream/meanpool_sweep_random/ep{ep}_test_predictions.npz"
B = 2000
SEED = 42


def load(path):
    d = np.load(path)
    return d["probs"].astype(np.float64), d["labels"].astype(np.int32)


print(f"Paired stratified bootstrap (B={B}, seed={SEED}) — oracle - random\n")
print(f"{'epoch':>5} | {'oracle':>7} | {'random':>7} | {'delta':>8} | {'95% CI':>20} | {'p(2-sided)':>10}")
print("-" * 78)
for ep in (50, 75, 100):
    op, ol = load(ORACLE.format(ep=ep))
    rp, rl = load(RANDOM.format(ep=ep))
    assert len(ol) == len(rl) == 3000, "size mismatch"
    aligned = np.array_equal(ol, rl)  # same samples, same order?

    auc_o = roc_auc_score(ol, op)
    auc_r = roc_auc_score(rl, rp)

    rng = np.random.RandomState(SEED)
    pos = np.where(ol == 1)[0]
    neg = np.where(ol == 0)[0]
    deltas = np.empty(B)
    for b in range(B):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        deltas[b] = roc_auc_score(ol[idx], op[idx]) - roc_auc_score(ol[idx], rp[idx])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    p_str = f"{p:.4f}" if p > 0 else f"<{1/B:.4f}"
    flag = "ns" if p > 0.05 else ("*" if p > 0.01 else ("**" if p > 0.001 else "***"))
    print(f"{ep:>5} | {auc_o:.4f} | {auc_r:.4f} | {auc_o-auc_r:+.4f} | "
          f"[{lo:+.4f}, {hi:+.4f}] | {p_str:>8} {flag}   aligned={aligned}")
