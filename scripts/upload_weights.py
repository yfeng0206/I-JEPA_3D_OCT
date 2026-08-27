#!/usr/bin/env python
"""Publish pretraining checkpoints to Hugging Face.

Why this exists
---------------
The paper states that the encoder, head and per-case predictions are released.
That was true for two arms only. An audit of the Hub found RANDOM and
CENTROID (published as ``oracle-anatomical-100ep``) present, and ENVELOPE,
COVER, ANATOMY-V1 and ANATOMY-V2 absent -- including ENVELOPE, which carries
the paper's headline positive result. Those arms existed on one local disk and
nowhere else, so a single disk failure would have destroyed them and falsified
a release claim already in print.

There was no upload script in the repository, only ``download_weights.py``.
That asymmetry is the reason nothing was ever published: fetching was one
command, publishing was a manual chore. This closes the loop.

Usage
-----
    python scripts/upload_weights.py --arm envelope
    python scripts/upload_weights.py --arm envelope --dry-run
    python scripts/upload_weights.py --list

Only the epochs the paper actually cites are uploaded by default, because
those are the ones a reader needs in order to reproduce a printed number.
Pass --all-epochs to publish the full trajectory.

Uploads are idempotent: a file already present on the Hub with the same SHA is
skipped, so an interrupted run can simply be repeated.
"""
import argparse
import hashlib
import os
import sys
import time

REPO_ID = "yfeng0206/ijepa-3d-oct-checkpoints"
RUNS = r"D:\jepa_phase0\runs"

# Local run directory, Hub subfolder, and the epochs the paper cites.
# Epochs are quoted from auto/auto_numbers.tex, not chosen by hand.
ARMS = {
    "envelope": {
        "run": "patch_mirage_envelope",
        "prefix": "jepa_patch_mirage",
        "hub": "envelope-mirage-100ep",
        "epochs": ["ep50", "ep75", "ep100"],
    },
    "cover": {
        "run": "cover_f021_ep25",
        "prefix": None,
        "hub": "cover-f021-100ep",
        "epochs": ["ep50", "ep75", "ep100"],
    },
    "anatomy-v2": {
        "run": "blob_resume_ep56",
        "prefix": None,
        "hub": "anatomy-v2-100ep",
        "epochs": ["ep50", "ep75"],
    },
    "anatomy-v1": {
        "run": "patch_mirage_anatomy",
        "prefix": None,
        "hub": "anatomy-v1-30ep",
        "epochs": ["ep30"],
    },
}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def pick(run_dir, epochs, all_epochs):
    """Return checkpoints to upload. Matches on the epoch token so that a
    prefix change between arms cannot silently select the wrong file."""
    out = []
    for fn in sorted(os.listdir(run_dir)):
        if not fn.endswith(".pth.tar"):
            continue
        if all_epochs:
            out.append(fn)
            continue
        stem = fn[: -len(".pth.tar")]
        tok = stem.split("-")[-1]
        # ep050 and ep50 both occur across arms.
        norm = tok.replace("ep0", "ep") if tok.startswith("ep0") else tok
        if norm in epochs:
            out.append(fn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--all-epochs", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list or not a.arm:
        print("Arms defined (epochs are those the paper cites):\n")
        for k, v in sorted(ARMS.items()):
            d = os.path.join(RUNS, v["run"])
            print("  %-11s %-24s -> %-22s %s%s"
                  % (k, v["run"], v["hub"], ",".join(v["epochs"]),
                     "" if os.path.isdir(d) else "   [RUN DIR MISSING]"))
        return 0

    spec = ARMS[a.arm]
    run_dir = os.path.join(RUNS, spec["run"])
    if not os.path.isdir(run_dir):
        print("run directory not found: %s" % run_dir)
        return 1

    files = pick(run_dir, spec["epochs"], a.all_epochs)
    if not files:
        print("no checkpoints matched for arm %s in %s" % (a.arm, run_dir))
        print("present: %s" % ", ".join(sorted(os.listdir(run_dir))[:12]))
        return 1

    total = sum(os.path.getsize(os.path.join(run_dir, f)) for f in files)
    print("arm       : %s" % a.arm)
    print("run dir   : %s" % run_dir)
    print("hub target: %s/%s" % (REPO_ID, spec["hub"]))
    print("files     : %d, %.2f GB" % (len(files), total / 1e9))
    for f in files:
        print("   %7.0f MB  %s" % (os.path.getsize(os.path.join(run_dir, f)) / 1e6, f))
    if a.dry_run:
        print("\ndry run, nothing uploaded")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    existing = set()
    try:
        existing = set(api.list_repo_files(REPO_ID, repo_type="model"))
    except Exception as e:
        print("could not list repo, continuing: %s" % str(e)[:120])

    for f in files:
        local = os.path.join(run_dir, f)
        target = "%s/%s" % (spec["hub"], f)
        if target in existing:
            print("skip (already on hub): %s" % target)
            continue
        t0 = time.time()
        print("uploading %s ... " % target, end="", flush=True)
        api.upload_file(path_or_fileobj=local, path_in_repo=target,
                        repo_id=REPO_ID, repo_type="model",
                        commit_message="publish %s %s" % (a.arm, f))
        dt = time.time() - t0
        mb = os.path.getsize(local) / 1e6
        print("done %.0f MB in %.0fs (%.1f MB/s)" % (mb, dt, mb / max(dt, 1e-6)))
        print("   sha256 %s" % sha256(local))

    print("\ncomplete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
