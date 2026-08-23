"""Clean fp32 continuation of the blob / anatomy-v2 arm, epoch 56 -> 100.

Why this exists
---------------
The original continuation (`runs/blob_resume_ep56`) was launched through
`scripts/campaign_chain.py`, whose `make_blob_cfg()` hardcoded
`meta.amp_target = True` and silently overrode the YAML. That computed EMA
targets in fp16 from epoch 56 onward, which is why the anatomy-v2 epoch-75 and
epoch-92 probes are excluded from every matched-epoch comparison in the paper.

That line is now fixed. This script re-runs the continuation from the SAME clean
pre-splice seed with fp32 targets throughout, writing to a NEW run directory so
the contaminated artifacts are preserved for the record and never overwritten.

Seed: runs/blob_resume_ep56/blob_resume_seed.pth.tar  (verified epoch 56,
loss 0.079249; byte-identical to runs/anatomy_v2_ep25/jepa_patch_mirage-last).

Everything except target precision matches the original: same config file, same
schedules sized for 100 epochs, same masking.

Idempotent: probes whose results.json exists are skipped, and training resumes
from the rolling `-last` checkpoint.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import yaml

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
REPO = pathlib.Path(__file__).resolve().parents[1]
CAMP = pathlib.Path(r"D:\jepa_phase0\campaign")
RUNS = pathlib.Path(r"D:\jepa_phase0\runs")

SEED = RUNS / "blob_resume_ep56" / "blob_resume_seed.pth.tar"
RUN_DIR = RUNS / "blob_fp32_ep56"
TAG = "jepa_patch_blob_fp32"
MILESTONES = [75, 100]
LOG = CAMP / "chain_blob_fp32.log"
STATUS = CAMP / "chain_blob_fp32_status.json"


def say(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    CAMP.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def set_status(**kw):
    cur = {}
    if STATUS.exists():
        try:
            cur = json.loads(STATUS.read_text())
        except Exception:  # noqa: BLE001
            pass
    cur.update(kw)
    cur["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(cur, indent=2))


def make_train_cfg() -> pathlib.Path:
    cfg = yaml.safe_load((REPO / "configs" / "patch_anatomy_v2.yaml").read_text())
    cfg.setdefault("meta", {})
    cfg["meta"]["read_checkpoint"] = str(SEED)
    cfg["meta"]["load_checkpoint"] = True
    cfg["meta"]["amp_target"] = False        # fp32 targets - the whole point
    cfg["optimization"]["epochs"] = 100
    cfg["logging"]["folder"] = str(RUN_DIR)
    cfg["logging"]["write_tag"] = TAG
    p = CAMP / "patch_blob_fp32.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def make_probe_cfg(ckpt: pathlib.Path, out_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Byte-identical protocol to every fp32 frozen probe in this programme."""
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
    p = CAMP / f"probe_{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def run_probe(ckpt: pathlib.Path, name: str):
    out_dir = RUNS / f"frozen_{name}"
    res = out_dir / "results.json"
    if res.exists():
        auc = json.loads(res.read_text()).get("test_auc")
        say(f"  probe {name}: already done (test_auc={auc})")
        return auc
    cfg = make_probe_cfg(ckpt, out_dir, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    say(f"  probe {name}: launching -> {out_dir}")
    with open(out_dir / "eval.log", "w") as lf:
        rc = subprocess.call([PY, "-u", "src/eval_downstream.py", "--config", str(cfg)],
                             cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONPATH": str(REPO)})
    if rc != 0 or not res.exists():
        say(f"  probe {name}: FAILED rc={rc}")
        return None
    auc = json.loads(res.read_text()).get("test_auc")
    say(f"  probe {name}: test_auc={auc}")
    return auc


def ckpt_for(epoch: int):
    p = RUN_DIR / f"{TAG}-ep{epoch}.pth.tar"
    return p if p.exists() else None


def train_to(epoch: int, cfg: pathlib.Path) -> bool:
    say(f"  training -> epoch {epoch} (fp32 targets)")
    rc = subprocess.call(
        [PY, "-u", "scripts/campaign_supervisor.py",
         "--config", str(cfg),
         "--stop_after_epoch", str(epoch),
         "--max_restarts", "8"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)})
    say(f"  supervisor for ep{epoch} exited rc={rc}")
    return rc == 0


def main() -> int:
    say("=" * 70)
    say("BLOB fp32 continuation start  (amp_target=false, seed epoch 56)")
    if not SEED.exists():
        say(f"ABORT: seed missing {SEED}")
        return 2
    cfg = make_train_cfg()
    say(f"  config: {cfg}")
    say(f"  run dir: {RUN_DIR}")

    aucs = {}
    for ep in MILESTONES:
        set_status(stage=f"train_ep{ep}", aucs=aucs)
        if ckpt_for(ep) is None:
            if not train_to(ep, cfg):
                say(f"ABORT: training to ep{ep} did not complete cleanly")
                set_status(stage=f"failed_train_ep{ep}", aucs=aucs)
                return 1
        c = ckpt_for(ep)
        if c is None:
            say(f"ABORT: no checkpoint for ep{ep}")
            set_status(stage=f"failed_ckpt_ep{ep}", aucs=aucs)
            return 1
        set_status(stage=f"probe_ep{ep}", aucs=aucs)
        aucs[f"ep{ep}"] = run_probe(c, f"meanpool_blob_fp32_ep{ep}")
        say(f"  AUC so far: {aucs}")
        set_status(aucs=aucs)

    set_status(stage="done", aucs=aucs)
    say(f"BLOB fp32 chain complete: {aucs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
