"""Frozen mean-pool AUC probes for the bridged-anatomy (blob) arm at ep75 and ep92.

Protocol is byte-identical to make_probe_cfg() in campaign_chain.py, which is the
same harness that produced bridge ep35/40/50 and the COVER numbers. Do not edit
the config below without re-running every arm.

ep92 comes from the rolling -last checkpoint, which a resume would overwrite, so
it is copied to a pinned path before probing.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import yaml

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
REPO = pathlib.Path(__file__).resolve().parents[1]
CAMP = pathlib.Path(r"D:\jepa_phase0\campaign")
RUNS = pathlib.Path(r"D:\jepa_phase0\runs")
BLOB = RUNS / "blob_resume_ep56"
LOG = CAMP / "probe_blob.log"


def say(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def make_probe_cfg(ckpt: pathlib.Path, out_dir: pathlib.Path, name: str) -> pathlib.Path:
    cfg = {
        "mode": "patch",
        "data": {
            "data_dir": r"D:\jepa_phase0\fairvision-glaucoma\data",
            "num_slices": 100, "slice_size": 256, "batch_size": 256,
            "num_workers": 4, "encode_chunk_size": 100, "use_amp": False,
        },
        "model": {
            "encoder_checkpoint": str(ckpt), "encoder_name": "vit_base",
            "patch_size": 16, "crop_size": 256, "freeze_encoder": True,
            "probe_type": "mean_pool", "head_type": "linear",
        },
        "training": {
            "lr_probe": 0.0004, "lr_head": 0.0004, "weight_decay": 0.05,
            "dropout": 0.2, "epochs": 50, "patience": 15,
            "warmup_epochs": 5, "seed": 42,
        },
        "logging": {"output_dir": str(out_dir)},
    }
    CAMP.mkdir(parents=True, exist_ok=True)
    p = CAMP / f"probe_{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def run_probe(ckpt: pathlib.Path, name: str):
    out_dir = RUNS / f"frozen_{name}"
    res = out_dir / "results.json"
    if res.exists():
        try:
            auc = json.loads(res.read_text()).get("test_auc")
            say(f"  probe {name}: already done (test_auc={auc})")
            return auc
        except Exception:  # noqa: BLE001
            pass
    cfg = make_probe_cfg(ckpt, out_dir, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    say(f"  probe {name}: launching -> {out_dir}")
    with open(out_dir / "eval.log", "w") as lf:
        rc = subprocess.call(
            [PY, "-u", "src/eval_downstream.py", "--config", str(cfg)],
            cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(REPO)})
    if rc != 0 or not res.exists():
        say(f"  probe {name}: FAILED rc={rc} (see {out_dir / 'eval.log'})")
        return None
    auc = json.loads(res.read_text()).get("test_auc")
    say(f"  probe {name}: test_auc={auc}")
    return auc


def ckpt_epoch(p: pathlib.Path):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False).get("epoch")


def main() -> int:
    say("=" * 70)
    say("blob (bridged anatomy) frozen mean-pool probes: ep75 + latest")

    targets = []

    ep75 = BLOB / "jepa_patch_blob_resume-ep75.pth.tar"
    if ep75.exists():
        targets.append((ep75, "meanpool_bridge_ep75"))
    else:
        say(f"  MISSING {ep75}")

    last = BLOB / "jepa_patch_blob_resume-last.pth.tar"
    if last.exists():
        ep = ckpt_epoch(last)
        pinned = BLOB / f"jepa_patch_blob_resume-ep{ep}-pinned.pth.tar"
        if not pinned.exists():
            say(f"  pinning rolling -last (epoch {ep}) -> {pinned.name}")
            shutil.copy2(last, pinned)
        targets.append((pinned, f"meanpool_bridge_ep{ep}"))
    else:
        say(f"  MISSING {last}")

    results = {}
    for ckpt, name in targets:
        results[name] = run_probe(ckpt, name)
        say(f"  results so far: {results}")

    (CAMP / "probe_blob_results.json").write_text(json.dumps(results, indent=2))
    say(f"done: {results}")
    say("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
