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


# The six masking-policy arms the paper compares. Every one is now published;
# previously only RANDOM and CENTROID were, which left the paper's release
# statement unsupported for the other four -- including ENVELOPE, which carries
# the headline result. AUCs are the frozen MeanPool test values reported in the
# paper, quoted here so a reader can confirm they fetched the right weights.
PAPER_ARMS = {
    "random": {
        "hub": "random-posfix-100ep",
        "desc": "Unguided masking. The null every other arm is compared against.",
        "files": ["jepa_patch-ep050.pth.tar", "jepa_patch-ep075.pth.tar",
                  "jepa_patch-ep100.pth.tar"],
        "epochs": [("ep50", "0.8641"), ("ep75", "0.8723"), ("ep100", "0.8746")],
    },
    "centroid": {
        "hub": "oracle-anatomical-100ep",
        "desc": "Band located by a per-column intensity centroid. Best arm; uses no segmentation model.",
        "files": ["jepa_patch_oracle-ep050.pth.tar", "jepa_patch_oracle-ep075.pth.tar",
                  "jepa_patch_oracle-ep100.pth.tar"],
        "epochs": [("ep50", "0.8740"), ("ep75", "0.8836"), ("ep100", "0.8855")],
    },
    "envelope": {
        "hub": "envelope-mirage-100ep",
        "desc": "Rectangles restricted to retinal tissue. Largest gain at the matched epoch.",
        "files": ["jepa_patch_mirage-ep50.pth.tar", "jepa_patch_mirage-ep75.pth.tar",
                  "jepa_patch_mirage-ep100.pth.tar"],
        "epochs": [("ep50", "0.8761"), ("ep75", "0.8803"), ("ep100", "0.8807")],
    },
    "cover": {
        "hub": "cover-f021-100ep",
        "desc": "Coverage-constrained placement, floor f=0.21. See the collation-defect appendix.",
        "files": ["jepa_patch_cover_f021-ep50.pth.tar", "jepa_patch_cover_f021-ep75.pth.tar",
                  "jepa_patch_cover_f021-ep100.pth.tar"],
        "epochs": [("ep50", "0.8643"), ("ep75", "0.8639"), ("ep100", "0.8577")],
    },
    "anatomy-v2": {
        "hub": "anatomy-v2-100ep",
        "desc": "Targets shaped to the segmented retina. ep75 is the clean fp32 continuation, "
                "not the superseded fp16 splice.",
        "files": ["jepa_patch_anatomy_v2-ep50.pth.tar",
                  "jepa_patch_anatomy_v2-ep75-fp32.pth.tar"],
        "epochs": [("ep50", "0.8654"), ("ep75 fp32", "0.8612")],
    },
    "anatomy-v1": {
        "hub": "anatomy-v1-30ep",
        "desc": "First anatomy-shaped sampler. Only epoch 30 exists.",
        "files": ["jepa_patch_mirage-ep30.pth.tar"],
        # No AUC is quoted here. Unlike every other arm this one has no
        # generated macro, and the nearest artifact -- region_auc_summary
        # anatomy_val -- is a VALIDATION figure, so quoting it would report a
        # validation number under a test label. See the paper for this arm.
        "epochs": [("ep30", "see paper")],
    },
}


def download_arm(arm, output_dir):
    """Fetch every published checkpoint for one masking-policy arm."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return 1
    spec = PAPER_ARMS[arm]
    print("arm  : %s" % arm)
    print("desc : %s" % spec["desc"])
    print("repo : %s/%s\n" % (ANCESTOR_REPO_ID, spec["hub"]))
    os.makedirs(output_dir, exist_ok=True)
    for fn in spec["files"]:
        rel = "%s/%s" % (spec["hub"], fn)
        print("  downloading %s ..." % rel)
        p = hf_hub_download(ANCESTOR_REPO_ID, rel, local_dir=output_dir)
        print("    -> %s" % p)
    print("\nAll arms fork from the same epoch-25 ancestor; fetch it with --ancestor-ep25.")
    return 0


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
    parser.add_argument("--arm", choices=sorted(PAPER_ARMS),
                        help="Download one masking-policy arm from the paper")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    args = parser.parse_args()

    if args.ancestor_ep25:
        return download_ancestor(args.output_dir)

    if args.arm:
        return download_arm(args.arm, args.output_dir)

    if args.list or not (args.all or args.encoder):
        print("Masking-policy arms from the paper (repo %s):\n" % ANCESTOR_REPO_ID)
        for name in sorted(PAPER_ARMS):
            a = PAPER_ARMS[name]
            print("  --arm %-11s %s" % (name, a["desc"]))
            print("      %-13s %s" % ("hub folder:", a["hub"]))
            print("      %-13s %s" % ("epochs:", ", ".join(
                "%s AUC %s" % (e, v) for e, v in a["epochs"])))
            print()
        print("  --ancestor-ep25   the locked epoch-25 fork point shared by every arm\n")
        print(f"Legacy weights from {REPO_ID}:\n")
        for fname, info in WEIGHTS.items():
            print(f"  {fname}")
            print(f"    {info['desc']}")
            print(f"    Size: {info['size']}  |  AUC: {info['auc']}")
            print()
        if not (args.all or args.encoder):
            print("Use --arm NAME for a paper arm, --encoder for best encoder, --all for everything")
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
