"""Chain for the COVER floor-0.21 arm: train to a milestone, probe, repeat.

Config is `configs/patch_cover_f021_ep25.yaml`, which differs from the archived
envelope/oracle/blob arms ONLY in the masking method:

    mode: mirage_cover, cover_min_visible_frac/leave_frac 0.21,
    cover_fill: random_legal, cover_min_visible_cells 4, anatomy_tau 0.1

`enc_truncate` is stock `prefix` and `amp_target` is stock false, so every other
knob matches the baselines. Verified with scripts/config_diff_arms.py.

Idempotent: probes whose results.json already exists are skipped, and the
supervisor resumes training from the rolling `-last` checkpoint.
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
CFG = REPO / "configs" / "patch_cover_f021_ep25.yaml"
RUN_DIR = RUNS / "cover_f021_ep25"
TAG = "jepa_patch_cover_f021"
MILESTONES = [30, 50, 75, 100]
LOG = CAMP / "chain_f021.log"
STATUS = CAMP / "chain_f021_status.json"


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


def make_probe_cfg(ckpt: pathlib.Path, out_dir: pathlib.Path, name: str):
    """Byte-identical protocol to every prior frozen probe in this programme."""
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


def ckpt_for(epoch: int):
    p = RUN_DIR / f"{TAG}-ep{epoch}.pth.tar"
    return p if p.exists() else None


def train_to(epoch: int) -> bool:
    say(f"  training -> epoch {epoch}")
    rc = subprocess.call(
        [PY, "-u", "scripts/campaign_supervisor.py",
         "--config", str(CFG),
         "--stop_after_epoch", str(epoch),
         "--val_baseline_json", str(CAMP / "val_baseline_envelope.json"),
         "--baseline_epoch_s", "6000",
         "--max_restarts", "8"],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)})
    say(f"  supervisor for ep{epoch} exited rc={rc}")
    return rc == 0


def main() -> int:
    say("=" * 70)
    say("COVER floor-0.21 chain start  (enc_truncate=prefix, amp_target=false)")
    aucs = {}
    for ep in MILESTONES:
        set_status(stage=f"train_ep{ep}", aucs=aucs)
        if ckpt_for(ep) is None:
            if not train_to(ep):
                say(f"ABORT: training to ep{ep} did not complete cleanly")
                set_status(stage=f"failed_train_ep{ep}", aucs=aucs)
                return 1
        c = ckpt_for(ep)
        if c is None:
            say(f"ABORT: no checkpoint for ep{ep} after training")
            set_status(stage=f"failed_ckpt_ep{ep}", aucs=aucs)
            return 1
        set_status(stage=f"probe_ep{ep}", aucs=aucs)
        auc = run_probe(c, f"meanpool_cover_f021_ep{ep}")
        aucs[f"ep{ep}"] = auc
        say(f"  AUC so far: {aucs}")
        set_status(aucs=aucs)
    set_status(stage="done", aucs=aucs)
    say(f"COVER f0.21 chain complete: {aucs}")
    say("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
