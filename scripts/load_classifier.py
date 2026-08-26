"""Reconstruct a trained downstream classifier from a saved head + an encoder.

`src/eval_downstream.py` saves `best_model.pt` next to every run's results, but
nothing consumed it afterwards, so the trained classifier heads sat unused and a
published AUC could not be reproduced without re-training the probe.

This script rebuilds the full classifier — encoder + probe + head — and scores
volumes end to end.

For the frozen arms the head is tiny (a MeanPool probe has zero parameters, so
the whole trained classifier is a LayerNorm + Linear over 768 dims = 2,305
parameters, ~11 KB). Pair it with the matching pretrained encoder and you have a
complete glaucoma classifier.

Inference path, matching eval_downstream.py exactly:

    volume (S,3,H,W)
      -> imagenet_normalize
      -> encoder                      (S, patches, 768)
      -> mean over patch tokens       (S, 768)
      -> probe (MeanPool: mean over slices)   (768,)
      -> head: LayerNorm -> Linear    (1,)
      -> sigmoid                      probability of glaucoma

Examples
--------
Inspect a saved head without touching any data::

    python scripts/load_classifier.py \
        --head results/downstream/meanpool_sweep_oracle/ep100_best_model.pt

Score volumes end to end with the matching encoder::

    python scripts/load_classifier.py \
        --head    results/downstream/meanpool_sweep_oracle/ep100_best_model.pt \
        --encoder results/pretraining/pretrain_oracle_anatomical/jepa_patch_oracle-ep100.pth.tar \
        --data-dir D:/fairvision/data --split Test

Verify a head reproduces its published AUC::

    python scripts/load_classifier.py \
        --head    results/downstream/meanpool_sweep_oracle/ep100_best_model.pt \
        --encoder results/pretraining/pretrain_oracle_anatomical/jepa_patch_oracle-ep100.pth.tar \
        --data-dir D:/fairvision/data --split Test \
        --expect-npz results/downstream/meanpool_sweep_oracle/ep100_test_predictions.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LinearHead(nn.Module):
    """Mirrors eval_downstream.LinearHead so state dicts load unchanged."""

    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(self.norm(x))


class MLPHead(nn.Module):
    """Mirrors eval_downstream.MLPHead."""

    def __init__(self, in_dim, hidden_dim=256, out_dim=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def describe(ckpt, path):
    """Print what a saved best_model.pt actually contains."""
    print(f"head file: {path}")
    print(f"  size          : {os.path.getsize(path):,} bytes")
    print(f"  keys          : {list(ckpt.keys())}")
    print(f"  best epoch    : {ckpt.get('epoch')}")
    if ckpt.get("val_auc") is not None:
        print(f"  val AUC       : {ckpt['val_auc']:.4f}")

    probe, head = ckpt.get("probe", {}), ckpt.get("head", {})
    n_probe = sum(v.numel() for v in probe.values())
    n_head = sum(v.numel() for v in head.values())
    print(f"  probe params  : {n_probe:,}"
          + ("   (MeanPool is parameter-free)" if n_probe == 0 else ""))
    print(f"  head params   : {n_head:,}")
    if "encoder" in ckpt:
        n_enc = sum(v.numel() for v in ckpt["encoder"].values())
        print(f"  encoder params: {n_enc:,}   (fine-tuned encoder is bundled)")
    else:
        print("  encoder       : not bundled - this is a FROZEN run, pair it "
              "with the matching pretrained checkpoint")
    print("  head tensors  :")
    for n, t in head.items():
        print(f"     {n:<28} {tuple(t.shape)}")
    return n_probe, n_head


def build_head(head_state, embed_dim):
    """Infer head type from the saved tensor names and load it."""
    if any(k.startswith("net.") for k in head_state):
        hidden = head_state["net.1.weight"].shape[0]
        head = MLPHead(embed_dim, hidden_dim=hidden)
    else:
        head = LinearHead(embed_dim)
    head.load_state_dict(head_state)
    head.eval()
    return head


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True, help="best_model.pt from a downstream run")
    ap.add_argument("--encoder", help="pretraining .pth.tar (frozen runs only)")
    ap.add_argument("--data-dir", help="FairVision data root")
    ap.add_argument("--split", default="Test")
    ap.add_argument("--num-slices", type=int, default=100)
    ap.add_argument("--slice-size", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--limit", type=int, help="score only the first N volumes")
    ap.add_argument("--expect-npz", help="test_predictions.npz to check against")
    ap.add_argument("--save-npz", help="write labels/probs here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    n_probe, _ = describe(ckpt, args.head)

    if not args.data_dir:
        print("\nNo --data-dir given, so stopping after inspection.")
        print("Pass --encoder and --data-dir to score volumes end to end.")
        return

    if n_probe:
        raise SystemExit(
            "\nThis head uses a probe with trainable parameters "
            "(cross_attn_pool or attentive). Only the parameter-free MeanPool "
            "path is supported here; use src/eval_downstream.py for the others.")

    # ---- encoder ----------------------------------------------------------
    from src.helper import _VIT_CONFIGS
    import src.models.vision_transformer as vit

    if "encoder" in ckpt:
        enc_state = ckpt["encoder"]
        print("\nusing the fine-tuned encoder bundled in the head file")
    else:
        if not args.encoder:
            raise SystemExit("This is a frozen run: --encoder is required.")
        pre = torch.load(args.encoder, map_location="cpu", weights_only=False)
        enc_state = pre["target_encoder"]      # EMA teacher, as eval_downstream uses
        print(f"\nusing target_encoder from {args.encoder} "
              f"(pretraining epoch {pre.get('epoch')})")

    cfg = _VIT_CONFIGS["vit_base"]
    encoder = vit.VisionTransformer(
        img_size=[args.slice_size], patch_size=16, **cfg)
    enc_state = {k.replace("module.", ""): v for k, v in enc_state.items()}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing or unexpected:
        print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    encoder.to(args.device).eval()

    embed_dim = ckpt["head"]["norm.weight"].shape[0]
    head = build_head(ckpt["head"], embed_dim).to(args.device)

    # ---- data -------------------------------------------------------------
    from torch.utils.data import DataLoader
    from src.datasets.oct_volumes import OCTVolumeDataset
    from src.eval_downstream import imagenet_normalize

    ds = OCTVolumeDataset(
        os.path.join(args.data_dir, args.split),
        num_slices=args.num_slices, slice_size=args.slice_size, return_label=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    print(f"scoring {args.limit or len(ds)} volumes from {args.split} "
          f"on {args.device}")

    probs, labels = [], []
    with torch.no_grad():
        for i, (volume, label) in enumerate(loader):
            if args.limit and i >= args.limit:
                break
            flat = volume.squeeze(0).to(args.device)     # (S,3,H,W)
            feats = []
            for j in range(0, flat.size(0), args.chunk_size):
                chunk = imagenet_normalize(flat[j:j + args.chunk_size])
                feats.append(encoder(chunk).mean(dim=1))  # mean over patches
            f = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)  # MeanPool
            probs.append(torch.sigmoid(head(f)).item())
            labels.append(int(label.squeeze()))
            if (i + 1) % 500 == 0:
                print(f"  {i + 1} volumes")

    probs, labels = np.array(probs), np.array(labels)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, probs)
    print(f"\n{args.split} AUC = {auc:.4f}  (n={len(labels)}, "
          f"{labels.sum()} positive)")

    if args.expect_npz:
        d = np.load(args.expect_npz)
        ref_auc = roc_auc_score(d["labels"], d["probs"].astype(np.float64))
        print(f"published AUC = {ref_auc:.4f}   delta = {auc - ref_auc:+.4f}")
        if len(d["probs"]) == len(probs):
            md = np.abs(d["probs"].astype(np.float64) - probs).max()
            print(f"max per-volume |Δprob| = {md:.2e}"
                  + ("   MATCH" if md < 1e-2 else "   MISMATCH"))

    if args.save_npz:
        np.savez(args.save_npz, labels=labels.astype(np.float32),
                 probs=probs.astype(np.float16))
        print(f"wrote {args.save_npz}")


if __name__ == "__main__":
    main()
