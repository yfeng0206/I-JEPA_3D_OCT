"""Download pretrained model weights from HuggingFace Hub.

Usage:
    python scripts/download_weights.py --all          # Download everything
    python scripts/download_weights.py --encoder      # Just the best encoder (ep32)
    python scripts/download_weights.py --list         # List available weights
    python scripts/download_weights.py --ancestor-ep25 --output-dir D:\...

`--ancestor-ep25` pulls the epoch-25 fork point that every masking arm
continues from.  It lives in a DIFFERENT repo from the three weights above
(`yfeng0206/ijepa-3d-oct-checkpoints`, path `random-posfix-100ep/
jepa_patch-ep025.pth.tar`) and is verified against the SHA-256 published with
the standalone mirror `yfeng0206/I-JEPA-OCT-random-posfix-ep25`.  A replication
in which one continuation starts from a different byte sequence is not a
replication, so the hash check is fatal, not advisory.

Requires: huggingface_hub
    pip install huggingface_hub
"""
import argparse
import hashlib
import os
import sys

REPO_ID = "yfeng0206/ijepa-oct-glaucoma"

# Epoch-25 ancestor: the locked fork point shared by RANDOM, ENVELOPE,
# CENTROID (oracle), ANATOMY and COVER.
ANCESTOR_REPO_ID = "yfeng0206/ijepa-3d-oct-checkpoints"
ANCESTOR_PATH = "random-posfix-100ep/jepa_patch-ep025.pth.tar"
ANCESTOR_SHA256 = (
    "e5ad5b0c2aadfa15449409786afbfa39d8b5405b699be8f02f2e540195e97e7b"
)


def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

WEIGHTS = {
    "jepa_patch-imagenet-init-ep32-best.pth.tar": {
        "desc": "Best encoder: ViT-B/16, ImageNet-init -> I-JEPA on 600K OCT slices, epoch 32",
        "size": "1.5 GB",
        "auc": "0.829 (fine-tuned)",
    },
    "jepa_patch-run3-ep11.pth.tar": {
        "desc": "Random-init encoder: ViT-B/16, I-JEPA on 600K OCT slices, epoch 11",
        "size": "1.5 GB",
        "auc": "0.819 val (fine-tuned)",
    },
    "vit_b16_imagenet_timm.pth": {
        "desc": "ImageNet supervised ViT-B/16 (timm). Base initialization for I-JEPA pretraining.",
        "size": "327 MB",
        "auc": "N/A (base init)",
    },
}


def download_ancestor(output_dir):
    """Fetch and verify the epoch-25 ancestor.  Returns 0 on success."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return 1

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {ANCESTOR_PATH} from {ANCESTOR_REPO_ID} ...")
    path = hf_hub_download(ANCESTOR_REPO_ID, ANCESTOR_PATH, local_dir=output_dir)
    print(f"  Saved to {path}")
    got = sha256_of(path)
    print(f"  sha256   {got}")
    print(f"  expected {ANCESTOR_SHA256}")
    if got != ANCESTOR_SHA256:
        print("  SHA-256 MISMATCH -- do not train from this file")
        return 1
    print("  SHA-256 OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Download model weights from HuggingFace")
    parser.add_argument("--all", action="store_true", help="Download all weights")
    parser.add_argument("--encoder", action="store_true", help="Download best encoder only")
    parser.add_argument("--list", action="store_true", help="List available weights")
    parser.add_argument("--ancestor-ep25", action="store_true",
                        help="Download the locked epoch-25 fork point and verify its SHA-256")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    args = parser.parse_args()

    if args.ancestor_ep25:
        return download_ancestor(args.output_dir)

    if args.list or not (args.all or args.encoder):
        print(f"Available weights from {REPO_ID}:\n")
        for fname, info in WEIGHTS.items():
            print(f"  {fname}")
            print(f"    {info['desc']}")
            print(f"    Size: {info['size']}  |  AUC: {info['auc']}")
            print()
        if not (args.all or args.encoder):
            print("Use --encoder for best encoder, --all for everything")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.encoder:
        files = ["jepa_patch-imagenet-init-ep32-best.pth.tar"]
    else:
        files = list(WEIGHTS.keys())

    for fname in files:
        info = WEIGHTS[fname]
        local_path = os.path.join(args.output_dir, fname)
        if os.path.exists(local_path):
            print(f"Already exists: {local_path}")
            continue
        print(f"Downloading {fname} ({info['size']})...")
        path = hf_hub_download(REPO_ID, fname, local_dir=args.output_dir)
        print(f"  Saved to {path}")

    print("\nDone!")


if __name__ == "__main__":
    sys.exit(main() or 0)
