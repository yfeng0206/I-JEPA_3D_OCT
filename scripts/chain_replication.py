#!/usr/bin/env python
"""G1 REPLICATION chain: six paired post-fork continuations, strictly serial.

The Area Chair's decisive objection is that every masking-policy contrast rests
on ONE post-fork continuation per policy, so policy is confounded with
post-fork optimisation noise.  This chain adds TWO independently randomised
continuations for each of RANDOM, ENVELOPE and CENTROID, all from the same
locked epoch-25 ancestor to the same locked endpoint (epoch 50), and probes
each with the frozen fp32 MeanPool protocol shared by the existing comparators.

Order is SEED-MAJOR, not policy-major:

    seed 1234: RANDOM, ENVELOPE, CENTROID
    seed 5678: RANDOM, ENVELOPE, CENTROID

so that a complete paired triple exists after roughly a third of the wall time
rather than only at the very end.  With a paper deadline this matters: a
partial chain that stops after leg 3 still supports a paired three-policy
comparison at n=2 continuations per policy.

Safety properties, all of which have bitten this programme before:

  * ONE GPU JOB AT A TIME.  A PID lock file refuses to start a second chain.
  * The ancestor's SHA-256 is re-verified immediately before EVERY launch.  Six
    continuations that did not all start from the same bytes are not a
    replication, and a silently re-downloaded or truncated file would be
    invisible otherwise.
  * Idempotent.  A leg whose epoch-50 checkpoint exists is not retrained; a
    probe whose results.json exists is not rerun; an interrupted leg resumes
    from the rolling per-epoch checkpoint instead of restarting from epoch 25.
  * Training is delegated to scripts/campaign_supervisor.py, which rewrites
    read_checkpoint to the rolling checkpoint on resume and restarts after a
    transient crash.
  * Probing is delegated to autopilot/run_guarded_probe.py, which pins the fp32
    protocol used by the existing comparator probes and hashes the encoder
    before and after to prove it stayed frozen.
  * The probe reads a PINNED copy of the epoch-50 checkpoint, never a rolling
    file that a later leg could mutate.

Usage:
    python scripts/chain_replication.py            # run the chain
    python scripts/chain_replication.py --status   # print progress, run nothing
    python scripts/chain_replication.py --plan     # print the queue, run nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
CFG_DIR = REPO / "configs" / "replication"
CAMP = pathlib.Path(r"D:\jepa_phase0\campaign\replication")
LOG = CAMP / "chain_replication.log"
STATUS = CAMP / "chain_replication_status.json"
LOCK = CAMP / "chain_replication.lock"

ANCESTOR = pathlib.Path(
    r"D:\jepa_phase0\fairvision-glaucoma\checkpoint-ep25"
    r"\jepa_patch-random_posfix-ep25.pth.tar")
ANCESTOR_SHA256 = "e5ad5b0c2aadfa15449409786afbfa39d8b5405b699be8f02f2e540195e97e7b"

ENDPOINT_EPOCH = 50
# Measured 57.7 min/epoch on this box; the supervisor warns above 1.5x this.
BASELINE_EPOCH_S = 3462.0
MAX_RESTARTS = 8

SEEDS = [1234, 5678]
POLICIES = ["random", "envelope", "centroid"]

QUEUE = [(p, s) for s in SEEDS for p in POLICIES]


def say(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    CAMP.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256(path: pathlib.Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_ancestor() -> bool:
    if not ANCESTOR.exists():
        say(f"ANCESTOR MISSING: {ANCESTOR}")
        return False
    got = sha256(ANCESTOR)
    if got != ANCESTOR_SHA256:
        say(f"ANCESTOR SHA MISMATCH: got {got} expected {ANCESTOR_SHA256}")
        return False
    return True


def leg_name(policy: str, seed: int) -> str:
    return f"rep_{policy}_s{seed}"


def leg_paths(policy: str, seed: int):
    name = leg_name(policy, seed)
    cfg_path = CFG_DIR / f"{name}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    run_dir = pathlib.Path(cfg["logging"]["folder"])
    tag = cfg["logging"]["write_tag"]
    return name, cfg_path, run_dir, tag


def endpoint_ckpt(run_dir: pathlib.Path, tag: str):
    p = run_dir / f"{tag}-ep{ENDPOINT_EPOCH}.pth.tar"
    return p if p.exists() else None


def rolling_epoch(run_dir: pathlib.Path, tag: str) -> int:
    p = run_dir / f"{tag}-last.pth.tar"
    if not p.exists():
        return -1
    import torch
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        return int(d.get("epoch", -1))
    except Exception as e:  # noqa: BLE001
        say(f"  could not read {p.name}: {e}")
        return -1


def pinned_ckpt(run_dir: pathlib.Path, tag: str) -> pathlib.Path:
    return run_dir / f"{tag}-ep{ENDPOINT_EPOCH}-pinned.pth.tar"


def set_status(**kw) -> None:
    cur = {}
    if STATUS.exists():
        try:
            cur = json.loads(STATUS.read_text())
        except Exception:  # noqa: BLE001
            pass
    cur.update(kw)
    cur["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    CAMP.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(cur, indent=2))


def train_leg(name: str, cfg_path: pathlib.Path) -> int:
    state_dir = CAMP / name
    state_dir.mkdir(parents=True, exist_ok=True)
    say(f"  supervisor -> epoch {ENDPOINT_EPOCH}  ({cfg_path.name})")
    rc = subprocess.call(
        [PY, "-u", "scripts/campaign_supervisor.py",
         "--config", str(cfg_path),
         "--stop_after_epoch", str(ENDPOINT_EPOCH),
         "--baseline_epoch_s", str(BASELINE_EPOCH_S),
         "--max_restarts", str(MAX_RESTARTS),
         "--state_dir", str(state_dir)],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)})
    say(f"  supervisor exited rc={rc}")
    return rc


def ensure_endpoint(name: str, cfg_path: pathlib.Path, run_dir: pathlib.Path,
                    tag: str, attempts: int = 3):
    """Train until the epoch-50 checkpoint exists.  Returns its path or None.

    The supervisor stops the trainer as soon as the epoch line appears in the
    log, but that line is printed just BEFORE the checkpoint is written, so a
    poll landing inside that ~15 s window can cut the save short.  Re-invoking
    the supervisor resumes from the rolling checkpoint and finishes the epoch;
    if the rolling checkpoint is already AT the endpoint, it is pinned instead.
    """
    for attempt in range(attempts):
        ck = endpoint_ckpt(run_dir, tag)
        if ck is not None:
            return ck
        ep = rolling_epoch(run_dir, tag)
        if ep >= ENDPOINT_EPOCH:
            src = run_dir / f"{tag}-last.pth.tar"
            dst = run_dir / f"{tag}-ep{ENDPOINT_EPOCH}.pth.tar"
            say(f"  rolling checkpoint is at epoch {ep}; pinning it as {dst.name}")
            shutil.copy2(src, dst)
            return dst
        if attempt:
            say(f"  retry {attempt}: no epoch-{ENDPOINT_EPOCH} checkpoint yet "
                f"(rolling at {ep})")
        train_leg(name, cfg_path)
    return endpoint_ckpt(run_dir, tag)


def probe_leg(name: str, ckpt: pathlib.Path, run_dir: pathlib.Path, tag: str):
    """Frozen fp32 MeanPool probe on a pinned copy of the endpoint checkpoint."""
    probe_name = f"meanpool_{name}_ep{ENDPOINT_EPOCH}"
    res = pathlib.Path(r"D:\jepa_phase0\runs") / f"frozen_{probe_name}" / "results.json"
    if res.exists():
        auc = json.loads(res.read_text()).get("test_auc")
        say(f"  probe {probe_name}: already done (test_auc={auc})")
        return auc
    pin = pinned_ckpt(run_dir, tag)
    if not pin.exists():
        say(f"  pinning {ckpt.name} -> {pin.name}")
        shutil.copy2(ckpt, pin)
    rc = subprocess.call(
        [PY, "-u", str(REPO / "autopilot" / "run_guarded_probe.py"),
         probe_name, str(pin)],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)})
    say(f"  probe {probe_name}: rc={rc}")
    if res.exists():
        auc = json.loads(res.read_text()).get("test_auc")
        say(f"  probe {probe_name}: test_auc={auc}")
        return auc
    return None


def acquire_lock() -> bool:
    CAMP.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip().split()[0])
        except Exception:  # noqa: BLE001
            old = -1
        alive = False
        if old > 0:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {old}", "/NH"],
                capture_output=True, text=True).stdout
            alive = str(old) in out
        if alive:
            say(f"ABORT: another chain is already running (pid {old})")
            return False
        say(f"stale lock from pid {old}; taking over")
    LOCK.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return True


def release_lock() -> None:
    try:
        LOCK.unlink()
    except Exception:  # noqa: BLE001
        pass


def collect_status() -> dict:
    out = {"endpoint_epoch": ENDPOINT_EPOCH, "legs": []}
    for policy, seed in QUEUE:
        name, cfg_path, run_dir, tag = leg_paths(policy, seed)
        ck = endpoint_ckpt(run_dir, tag)
        probe_name = f"meanpool_{name}_ep{ENDPOINT_EPOCH}"
        res = (pathlib.Path(r"D:\jepa_phase0\runs")
               / f"frozen_{probe_name}" / "results.json")
        auc = None
        if res.exists():
            try:
                auc = json.loads(res.read_text()).get("test_auc")
            except Exception:  # noqa: BLE001
                pass
        out["legs"].append({
            "leg": name, "policy": policy, "seed": seed,
            "trained": ck is not None,
            "rolling_epoch": rolling_epoch(run_dir, tag),
            "probe_auc": auc,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--skip_probes", action="store_true")
    a = ap.parse_args()

    if a.plan:
        for i, (p, s) in enumerate(QUEUE, 1):
            print(f"  {i}. {leg_name(p, s)}  ({p.upper()}, seed {s}) "
                  f"-> epoch {ENDPOINT_EPOCH}")
        return 0
    if a.status:
        print(json.dumps(collect_status(), indent=2))
        return 0

    if not acquire_lock():
        return 1
    try:
        say("=" * 70)
        say(f"G1 REPLICATION chain start: {len(QUEUE)} legs, endpoint epoch "
            f"{ENDPOINT_EPOCH}, strictly sequential")
        t0 = time.time()
        results = {}
        for i, (policy, seed) in enumerate(QUEUE, 1):
            name, cfg_path, run_dir, tag = leg_paths(policy, seed)
            say("-" * 70)
            say(f"[leg {i}/{len(QUEUE)}] {name}  (+{(time.time() - t0) / 3600:.1f} h)")

            # Re-verified before EVERY launch, per the replication contract.
            if not verify_ancestor():
                say("ABORT: ancestor verification failed; nothing further launched")
                set_status(stage=f"ancestor_failed_before_{name}", results=results)
                return 2
            say(f"  ancestor sha256 OK ({ANCESTOR_SHA256[:16]}...)")

            set_status(stage=f"train_{name}", results=results)
            ck = ensure_endpoint(name, cfg_path, run_dir, tag)
            if ck is None:
                say(f"ABORT: {name} produced no epoch-{ENDPOINT_EPOCH} checkpoint")
                set_status(stage=f"failed_train_{name}", results=results)
                return 3
            say(f"  endpoint checkpoint: {ck}")

            if a.skip_probes:
                results[name] = "trained"
            else:
                set_status(stage=f"probe_{name}", results=results)
                results[name] = probe_leg(name, ck, run_dir, tag)
            set_status(results=results)
            say(f"  results so far: {results}")

        set_status(stage="done", results=results)
        say(f"CHAIN COMPLETE in {(time.time() - t0) / 3600:.1f} h: {results}")
        say("=" * 70)
        return 0
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
