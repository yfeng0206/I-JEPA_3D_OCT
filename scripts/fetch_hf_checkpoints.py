"""Fetch the published random/oracle pretraining checkpoints from Hugging Face.

The two 100-epoch arms live at yfeng0206/ijepa-3d-oct-checkpoints but were never
mirrored locally, so any cross-arm probe could only see the envelope and blob
runs that happen to sit on D:.  This pulls the epoch checkpoints (not the
-lowest-pretrain-loss- ones, which the repo README explicitly warns are selected
by an anti-signal) and verifies each against MANIFEST.json.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "yfeng0206/ijepa-3d-oct-checkpoints"
DEST = pathlib.Path(r"D:\jepa_phase0\checkpoints_hf")

WANTED = [
    "random-posfix-100ep/jepa_patch-ep050.pth.tar",
    "random-posfix-100ep/jepa_patch-ep075.pth.tar",
    "random-posfix-100ep/jepa_patch-ep100.pth.tar",
    "oracle-anatomical-100ep/jepa_patch_oracle-ep050.pth.tar",
    "oracle-anatomical-100ep/jepa_patch_oracle-ep075.pth.tar",
    "oracle-anatomical-100ep/jepa_patch_oracle-ep100.pth.tar",
]


def sha256(path: pathlib.Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)

    man_path = hf_hub_download(REPO_ID, "MANIFEST.json", local_dir=str(DEST))
    manifest = json.loads(pathlib.Path(man_path).read_text())
    # Manifest shape is not contractual; index whatever maps path -> sha256.
    expected: dict[str, str] = {}

    def walk(node):
        if isinstance(node, dict):
            p, s = node.get("path"), node.get("sha256")
            if isinstance(p, str) and isinstance(s, str):
                expected[p.replace("\\", "/")] = s.lower()
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(manifest)
    print(f"manifest indexed {len(expected)} hashed entries", flush=True)

    ok = True
    for rel in WANTED:
        print(f"\n=== {rel} ===", flush=True)
        local = hf_hub_download(REPO_ID, rel, local_dir=str(DEST))
        lp = pathlib.Path(local)
        size_gb = lp.stat().st_size / 1024**3
        want = expected.get(rel) or expected.get(pathlib.PurePosixPath(rel).name)
        if want:
            got = sha256(lp)
            match = got == want
            ok &= match
            print(f"  {size_gb:.2f} GB  sha256 {'OK' if match else 'MISMATCH'}", flush=True)
            if not match:
                print(f"    expected {want}\n    got      {got}", flush=True)
        else:
            print(f"  {size_gb:.2f} GB  (no manifest hash to check)", flush=True)

    print("\nALL VERIFIED" if ok else "\nSOME FILES FAILED VERIFICATION", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
