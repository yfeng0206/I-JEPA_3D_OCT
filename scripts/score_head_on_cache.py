"""Score a saved MeanPool classifier head against a cached feature tensor.

`eval_downstream.py` caches encoder features as
``{'features': (N, num_slices, embed_dim), 'labels': (N,)}``. A frozen MeanPool
classifier is just ``mean over slices -> LayerNorm -> Linear -> sigmoid``, so
once the cache exists the head can be applied on CPU with no GPU work at all.

That makes it cheap to ask a question the probe itself cannot answer: the probe
retrains its head on every run, so an fp32-vs-fp16 comparison of probe outputs
confounds the precision change with head-retraining noise. Holding the ORIGINAL
head fixed and swapping only the features isolates the precision effect.

Examples
--------
Score the original oracle ep100 head against locally recomputed fp32 features::

    python scripts/score_head_on_cache.py \
        --head  D:/jepa_phase0/checkpoints_hf/downstream-heads/frozen-meanpool/oracle-ep100-head.pt \
        --cache D:/jepa_phase0/runs/frozen_meanpool_oracle_ep100_fp32/feature_cache/Test_s100_r256_fp32_<hash>.pt \
        --expect-npz results/downstream/meanpool_sweep_oracle/ep100_test_predictions.npz
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class LinearHead(nn.Module):
    """Mirrors eval_downstream.LinearHead so state dicts load unchanged."""

    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(self.norm(x))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--head', required=True, help='best_model.pt to apply')
    ap.add_argument('--cache', required=True, help='cached feature .pt')
    ap.add_argument('--expect-npz', help='reference predictions to compare with')
    ap.add_argument('--save-npz', help='write labels/probs here')
    args = ap.parse_args()

    ckpt = torch.load(args.head, map_location='cpu', weights_only=False)
    if sum(v.numel() for v in ckpt.get('probe', {}).values()):
        raise SystemExit('head has a parameterised probe; only MeanPool is supported')

    print('head  : %s' % args.head)
    print('  saved epoch %s   val_auc %s' % (ckpt.get('epoch'), ckpt.get('val_auc')))

    data = torch.load(args.cache, map_location='cpu')
    feats, labels = data['features'], data['labels']
    print('cache : %s' % args.cache)
    print('  features %s  %s   labels %s'
          % (tuple(feats.shape), feats.dtype, tuple(labels.shape)))

    embed_dim = ckpt['head']['norm.weight'].shape[0]
    if feats.shape[-1] != embed_dim:
        raise SystemExit('embed dim mismatch: cache %d vs head %d'
                         % (feats.shape[-1], embed_dim))

    head = LinearHead(embed_dim)
    head.load_state_dict(ckpt['head'])
    head.eval()

    # MeanPool: average the per-slice features, exactly as MeanPoolProbe does.
    pooled = feats.float().mean(dim=1)
    with torch.no_grad():
        probs = torch.sigmoid(head(pooled)).squeeze(-1).numpy().astype(np.float64)
    y = labels.numpy()

    auc = roc_auc_score(y, probs)
    print('\nAUC = %.10f   (n=%d, %d positive)' % (auc, len(y), int(y.sum())))

    if args.expect_npz:
        d = np.load(args.expect_npz)
        ref = roc_auc_score(d['labels'], d['probs'].astype(np.float64))
        print('reference   = %.10f' % ref)
        print('delta       = %+.10f' % (auc - ref))
        if len(d['probs']) == len(probs):
            if not np.array_equal(d['labels'].astype(np.int64), y.astype(np.int64)):
                print('WARNING: label vectors differ - volume order is not aligned')
            else:
                md = np.abs(d['probs'].astype(np.float64) - probs).max()
                print('max per-volume |dprob| = %.3e' % md)

    if args.save_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)), exist_ok=True)
        np.savez(args.save_npz, labels=y.astype(np.float32),
                 probs=probs.astype(np.float16))
        print('wrote %s' % args.save_npz)


if __name__ == '__main__':
    main()
