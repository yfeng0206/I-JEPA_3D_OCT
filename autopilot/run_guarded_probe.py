"""Guarded frozen-probe runner.

Runs src/eval_downstream.py in frozen-probe mode ONLY, with the safety
assertions the operator directive requires:

  * config must set model.freeze_encoder = true
  * the pretrained encoder checkpoint is SHA-256 hashed before and after the run
  * if the hash changes, the run is marked INVALID
  * protocol is pinned to the fp32 protocol shared by the 16 comparator runs
    (use_amp: false, encode_chunk_size: 100, batch 256, seed 42, 50 epochs)

This script NEVER trains a representation encoder. It only trains a linear head
on top of cached frozen features.

Usage:
  python run_guarded_probe.py <name> <encoder_checkpoint>
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import yaml

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
REPO = r"C:\Users\Gary\Desktop\jepa"
RUNS = r"D:\jepa_phase0\runs"
CFGDIR = r"D:\jepa_phase0\autopilot_out\probe_cfgs"
GUARD = r"D:\jepa_phase0\autopilot_out\probe_guards"
os.makedirs(CFGDIR, exist_ok=True)
os.makedirs(GUARD, exist_ok=True)

# Byte-identical to the protocol used by the 16 fp32 comparator runs.
PROTOCOL = {
    "data": {"data_dir": r"D:\jepa_phase0\fairvision-glaucoma\data",
             "num_slices": 100, "slice_size": 256, "batch_size": 256,
             "num_workers": 4, "encode_chunk_size": 100, "use_amp": False},
    "model": {"encoder_name": "vit_base", "patch_size": 16, "crop_size": 256,
              "freeze_encoder": True, "probe_type": "mean_pool", "head_type": "linear"},
    "training": {"lr_probe": 0.0004, "lr_head": 0.0004, "weight_decay": 0.05,
                 "dropout": 0.2, "epochs": 50, "patience": 15,
                 "warmup_epochs": 5, "seed": 42},
}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    name, ckpt = sys.argv[1], sys.argv[2]
    out_dir = os.path.join(RUNS, "frozen_" + name)
    res_path = os.path.join(out_dir, "results.json")

    if os.path.exists(res_path):
        r = json.load(open(res_path))
        print("[skip] %s already has results.json (test_auc=%s)" % (name, r.get("test_auc")))
        return 0

    if not os.path.exists(ckpt):
        print("[fail] checkpoint missing: %s" % ckpt)
        return 2

    cfg = {"mode": "patch",
           "data": dict(PROTOCOL["data"]),
           "model": dict(PROTOCOL["model"], encoder_checkpoint=ckpt),
           "training": dict(PROTOCOL["training"]),
           "logging": {"output_dir": out_dir}}

    # ---- pre-flight assertions -------------------------------------------
    assert cfg["model"]["freeze_encoder"] is True, "freeze_encoder must be True"
    assert cfg["data"]["use_amp"] is False, "probe protocol requires fp32"

    print("[guard] hashing encoder checkpoint before run ...", flush=True)
    t0 = time.time()
    h_before = sha256(ckpt)
    size_before = os.path.getsize(ckpt)
    mtime_before = os.path.getmtime(ckpt)
    print("[guard] sha256(before) = %s  (%.1fs)" % (h_before, time.time() - t0), flush=True)

    cfg_path = os.path.join(CFGDIR, "probe_%s.yaml" % name)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    os.makedirs(out_dir, exist_ok=True)

    print("[run] %s -> %s" % (name, out_dir), flush=True)
    started = time.time()
    with open(os.path.join(out_dir, "eval.log"), "w") as lf:
        rc = subprocess.call([PY, "-u", "src/eval_downstream.py", "--config", cfg_path],
                             cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONPATH": REPO})
    elapsed = time.time() - started

    # ---- post-flight assertions ------------------------------------------
    print("[guard] hashing encoder checkpoint after run ...", flush=True)
    h_after = sha256(ckpt)
    unchanged = (h_after == h_before
                 and os.path.getsize(ckpt) == size_before
                 and os.path.getmtime(ckpt) == mtime_before)

    auc = None
    if os.path.exists(res_path):
        auc = json.load(open(res_path)).get("test_auc")

    guard = {
        "name": name, "encoder_checkpoint": ckpt,
        "sha256_before": h_before, "sha256_after": h_after,
        "size_before": size_before, "size_after": os.path.getsize(ckpt),
        "mtime_before": mtime_before, "mtime_after": os.path.getmtime(ckpt),
        "encoder_unchanged": unchanged,
        "freeze_encoder": cfg["model"]["freeze_encoder"],
        "use_amp": cfg["data"]["use_amp"],
        "return_code": rc, "elapsed_sec": elapsed,
        "test_auc": auc, "output_dir": out_dir,
        "valid": bool(unchanged and rc == 0 and auc is not None),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(GUARD, "guard_%s.json" % name), "w") as f:
        json.dump(guard, f, indent=1)

    if not unchanged:
        print("[INVALID] ENCODER HASH CHANGED. Run is invalid; restore checkpoint.")
        return 3
    print("[guard] encoder hash unchanged - frozen confirmed")
    print("[done] %s rc=%d auc=%s elapsed=%.1f min" % (name, rc, auc, elapsed / 60.0))
    return 0 if guard["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
