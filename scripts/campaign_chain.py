"""Unattended campaign chain: COVER training -> AUC probes -> blob -> AUC probes.

Runs for days with nobody watching, so it is deliberately conservative:

  * every stage is idempotent -- re-running after a reboot skips work whose
    output already exists;
  * it never touches the frozen blob run, only the copied seed checkpoint;
  * a stage failure is recorded and the chain continues to the next INDEPENDENT
    stage rather than dying silently;
  * a JSON status file is written after every state change so progress is
    readable at a glance.

The COVER training stage is expected to already be running under
``campaign_supervisor.py``; this chain waits for its checkpoints rather than
launching a second trainer.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

import yaml

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
REPO = pathlib.Path(__file__).resolve().parents[1]
CAMP = pathlib.Path(r"D:\jepa_phase0\campaign")
COVER_RUN = pathlib.Path(r"D:\jepa_phase0\runs\cover_random_ep25")
COVER_TAG = "jepa_patch_cover_random"
BLOB_RUN = pathlib.Path(r"D:\jepa_phase0\runs\blob_resume_ep56")
BLOB_TAG = "jepa_patch_blob_resume"
BLOB_SEED = BLOB_RUN / "blob_resume_seed.pth.tar"
STATUS = CAMP / "chain_status.json"
LOG = CAMP / "chain.log"


def say(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    CAMP.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def status(**kw):
    s = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    s.update(kw)
    s["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(s, indent=2))


def trainer_running() -> bool:
    """True while a train_patch.py process is alive.

    The AUC probes and the trainer cannot share this GPU: training already sits
    at ~97% of the 24 GB card, so launching a probe alongside it would OOM both.
    """
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return False
    return "train_patch.py" in out


def wait_for_training_done(timeout_h=200):
    """Block until no trainer is alive, so the GPU is exclusively ours."""
    deadline = time.time() + timeout_h * 3600
    announced = False
    while time.time() < deadline:
        if not trainer_running():
            # Confirm across two polls -- the supervisor may be between restarts.
            time.sleep(120)
            if not trainer_running():
                return True
        if not announced:
            say("  training in progress; probes deferred until the GPU is free")
            announced = True
        time.sleep(300)
    say("  TIMEOUT waiting for training to finish")
    return False


def ckpt_for(run: pathlib.Path, tag: str, epoch: int):
    for name in (f"{tag}-ep{epoch}.pth.tar", f"{tag}-ep{epoch:03d}.pth.tar"):
        p = run / name
        if p.exists():
            return p
    return None


def wait_for_ckpt(run, tag, epoch, timeout_h=96):
    """Block until the epoch checkpoint appears, or the trainer clearly died."""
    deadline = time.time() + timeout_h * 3600
    announced = False
    while time.time() < deadline:
        p = ckpt_for(run, tag, epoch)
        if p:
            return p
        if not announced:
            say(f"  waiting for {tag}-ep{epoch} ...")
            announced = True
        time.sleep(300)
    say(f"  TIMEOUT waiting for {tag}-ep{epoch}")
    return None


def make_probe_cfg(ckpt: pathlib.Path, out_dir: pathlib.Path, name: str):
    """Frozen mean_pool probe config, byte-identical protocol to prior probes."""
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
    out_dir = pathlib.Path(r"D:\jepa_phase0\runs") / f"frozen_{name}"
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
    logf = out_dir / "eval.log"
    with open(logf, "w") as lf:
        rc = subprocess.call(
            [PY, "-u", "src/eval_downstream.py", "--config", str(cfg)],
            cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT,
            env=dict(**{**dict(__import__("os").environ), "PYTHONPATH": str(REPO)}))
    if rc != 0:
        say(f"  probe {name}: FAILED rc={rc} (see {logf})")
        return None
    auc = None
    if res.exists():
        auc = json.loads(res.read_text()).get("test_auc")
    say(f"  probe {name}: test_auc={auc}")
    return auc


def make_blob_cfg():
    """Blob resume config -- reads the COPIED seed, writes to a NEW run dir.

    The original run is frozen (FROZEN_MANIFEST.json says do not modify or
    resume) and best_val_loss resets to +inf on every launch, so resuming in
    place would overwrite the frozen -best checkpoint on the first epoch.
    """
    base = yaml.safe_load((REPO / "configs" / "patch_anatomy_v2.yaml").read_text())
    base["meta"]["read_checkpoint"] = str(BLOB_SEED)
    base["meta"]["load_checkpoint"] = True
    base["meta"]["amp_target"] = True
    base["optimization"]["epochs"] = 100
    base["logging"]["folder"] = str(BLOB_RUN)
    base["logging"]["write_tag"] = BLOB_TAG
    p = CAMP / "patch_blob_resume.yaml"
    p.write_text(yaml.safe_dump(base, sort_keys=False))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_blob", action="store_true")
    args = ap.parse_args()

    say("=" * 70)
    say("campaign chain start")
    status(stage="start")

    # ---- Stage A: COVER, stopping at each milestone to probe ---------------
    # Sequential by design: training holds ~97% of the 24 GB card, so a probe
    # cannot run alongside it.  optimization.epochs stays at 100 throughout --
    # it sizes the cosine LR/WD/EMA schedules, and shortening it per milestone
    # would anneal the LR to final_lr at the milestone instead of at epoch 100.
    # The supervisor stops the trainer at the milestone instead, then resumes
    # from the rolling checkpoint for the next leg.
    aucs = {}
    for ep in (30, 50, 75, 100):
        existing = ckpt_for(COVER_RUN, COVER_TAG, ep)
        if existing is None:
            status(stage=f"cover_train_to_{ep}")
            say(f"COVER: training to epoch {ep}")
            rc = subprocess.call(
                [PY, "-u", "scripts/campaign_supervisor.py",
                 "--config", str(REPO / "configs" / "patch_cover_random_ep25.yaml"),
                 "--val_baseline_json", str(CAMP / "val_baseline_envelope.json"),
                 "--baseline_epoch_s", "2600", "--max_restarts", "8",
                 "--stop_after_epoch", str(ep)],
                cwd=str(REPO),
                env=dict(**{**dict(__import__("os").environ), "PYTHONPATH": str(REPO)}))
            say(f"  supervisor for ep{ep} exited rc={rc}")
            if rc == 4:
                say("  collapse gate stopped training; probing what exists and "
                    "skipping the remaining COVER legs")
                p = ckpt_for(COVER_RUN, COVER_TAG, ep)
                if p:
                    aucs[f"cover_ep{ep}"] = run_probe(p, f"cover_random_ep{ep}")
                    status(cover_aucs=aucs)
                break
        p = ckpt_for(COVER_RUN, COVER_TAG, ep)
        if p is None:
            say(f"  COVER ep{ep} checkpoint absent after training; stopping stage A")
            break
        status(stage=f"cover_ep{ep}_probe")
        wait_for_training_done()
        aucs[f"cover_ep{ep}"] = run_probe(p, f"cover_random_ep{ep}")
        status(cover_aucs=aucs)
        say(f"  AUC so far: {aucs}")
    say(f"COVER AUCs: {aucs}")

    # ---- Stage B: blob resume from the COPY --------------------------------
    if not args.skip_blob:
        if not BLOB_SEED.exists():
            say(f"blob seed missing at {BLOB_SEED}; skipping blob stage")
        else:
            status(stage="blob_train")
            cfg = make_blob_cfg()
            say(f"blob: supervising resume {BLOB_SEED.name} -> ep100 in {BLOB_RUN}")
            # Blob MUST be gated against its own trajectory, not envelope's.
            # It sits at ~0.33 val loss versus envelope's ~0.14 -- that gap IS
            # the predictor collapse this run exists to document, and it
            # predates the campaign. Gating it on the envelope curve stopped it
            # at ep57 on a 2.33x ratio that was really a 1.01x continuation of
            # its own history.
            blob_baseline = CAMP / "val_baseline_blob.json"
            if not blob_baseline.exists():
                blob_baseline = CAMP / "val_baseline_envelope.json"
                say("  WARNING: blob baseline missing, falling back to envelope "
                    "(will over-trigger)")
            rc = subprocess.call(
                [PY, "-u", "scripts/campaign_supervisor.py", "--config", str(cfg),
                 "--val_baseline_json", str(blob_baseline),
                 "--baseline_epoch_s", "5400", "--max_restarts", "8"],
                cwd=str(REPO),
                env=dict(**{**dict(__import__("os").environ), "PYTHONPATH": str(REPO)}))
            say(f"blob training supervisor exited rc={rc}")
            bl = {}
            for ep in (75, 100):
                p = ckpt_for(BLOB_RUN, BLOB_TAG, ep)
                if p:
                    status(stage=f"blob_ep{ep}_probe")
                    bl[f"blob_ep{ep}"] = run_probe(p, f"blob_resume_ep{ep}")
                    status(blob_aucs=bl)
            say(f"BLOB AUCs: {bl}")

    status(stage="done")
    say("campaign chain complete")
    say("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
