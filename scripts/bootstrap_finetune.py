"""Paired stratified bootstrap: oracle vs random FINE-TUNE, per probe.

Same method as bootstrap_frozen_meanpool.py / ablation_analysis.md.
Pairs oracle FT predictions against the matched random FT run (same probe).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

# (label, oracle_path, random_path)
PAIRS = [
    ("MeanPool",   "results/downstream/finetune_oracle/meanpool_test_predictions.npz",
                   "results/downstream/finetune_random/mean_pool_test_predictions.npz"),
    ("d=1",        "results/downstream/finetune_oracle/d1_test_predictions.npz",
                   "results/downstream/finetune_random/attentive_test_predictions.npz"),
    ("CrossAttn",  "results/downstream/finetune_oracle/crossattn_test_predictions.npz",
                   "results/downstream/finetune_random/cross_attn_pool_test_predictions.npz"),
]
B, SEED = 2000, 42


def load(p):
    d = np.load(p)
    return d["probs"].astype(np.float64), d["labels"].astype(np.int32)


print(f"Paired stratified bootstrap (B={B}, seed={SEED}) — oracle FT - random FT\n")
print(f"{'probe':>9} | {'oracleFT':>8} | {'randomFT':>8} | {'delta':>8} | {'95% CI':>20} | {'p':>9} | aligned")
print("-" * 86)
for name, op_path, rp_path in PAIRS:
    op, ol = load(op_path)
    rp, rl = load(rp_path)
    aligned = (len(ol) == len(rl)) and np.array_equal(ol, rl)
    auc_o, auc_r = roc_auc_score(ol, op), roc_auc_score(rl, rp)
    if not aligned:
        print(f"{name:>9} | {auc_o:.4f} | {auc_r:.4f} | {auc_o-auc_r:+.4f} | "
              f"{'NOT ALIGNED — unpaired':>20} | {'n/a':>9} | {aligned}")
        continue
    rng = np.random.RandomState(SEED)
    pos, neg = np.where(ol == 1)[0], np.where(ol == 0)[0]
    deltas = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        deltas[b] = roc_auc_score(ol[idx], op[idx]) - roc_auc_score(ol[idx], rp[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    p_str = f"{p:.4f}" if p > 0 else f"<{1/B:.4f}"
    flag = "ns" if p > 0.05 else ("*" if p > 0.01 else ("**" if p > 0.001 else "***"))
    print(f"{name:>9} | {auc_o:.4f} | {auc_r:.4f} | {auc_o-auc_r:+.4f} | "
          f"[{lo:+.4f}, {hi:+.4f}] | {p_str:>7} {flag} | {aligned}")
