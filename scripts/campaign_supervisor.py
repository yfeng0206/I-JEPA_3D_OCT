"""Crash-safe supervisor for a multi-day pretraining campaign.

Two problems this solves, both of which bite silently.

1. NO AUTO-RESUME.  ``train_patch.py`` resumes from whatever ``meta.read_checkpoint``
   names, and the campaign config names the ep25 FORK.  A crash at epoch 60
   followed by a naive restart therefore replays from epoch 26 and quietly
   discards ~38 hours.  The trainer does write a rolling ``<tag>-last.pth.tar``
   every epoch (atomically, via a .tmp + os.replace), so the resume point
   exists -- nothing points at it.  This supervisor rewrites
   ``read_checkpoint`` to the rolling checkpoint whenever one is present.

2. NO RESTART.  Over 155 hours a DataLoader worker OOM, a driver hiccup or a
   reboot is likely.  The supervisor restarts, with a cap so a genuinely broken
   run cannot spin forever, and refuses to restart if no epoch progress was made
   since the previous attempt (which would indicate a deterministic failure
   rather than a transient one).

It also records per-epoch health: wall time, train/val loss, and the sampler's
own COVER statistics, and flags a stall, a val-loss blow-up against a baseline
curve, or a floor regression.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import yaml

EPOCH_RE = re.compile(
    r"^Epoch (\d+)/(\d+)\s+\((\d+)s\)\s+train_loss=([\d.]+)(?:\s+val_loss=([\d.]+))?"
)
COVER_RE = re.compile(
    r"\[COVER\] hidden=([\d.]+)\s+visible_cells=([\d.]+)\s+floor_ok=([\d.]+)"
)


def find_last(run_dir: pathlib.Path, tag: str):
    p = run_dir / f"{tag}-last.pth.tar"
    return p if p.exists() else None


def ckpt_epoch(path: pathlib.Path):
    import torch
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        return int(d.get("epoch", -1))
    except Exception as e:  # noqa: BLE001
        print(f"  [supervisor] could not read {path}: {e}", flush=True)
        return -1


def build_resume_config(base_cfg_path: pathlib.Path, ckpt: pathlib.Path,
                        work_dir: pathlib.Path) -> pathlib.Path:
    cfg = yaml.safe_load(base_cfg_path.read_text())
    cfg["meta"]["read_checkpoint"] = str(ckpt)
    cfg["meta"]["load_checkpoint"] = True
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / (base_cfg_path.stem + "_resume.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def scan_log(log_path: pathlib.Path, state: dict):
    """Read new lines and update health state."""
    if not log_path.exists():
        return []
    events = []
    with open(log_path, "r", errors="ignore") as fh:
        fh.seek(state.get("pos", 0))
        for line in fh:
            m = EPOCH_RE.match(line.strip())
            if m:
                ep, tot, secs, tr = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
                vl = float(m.group(5)) if m.group(5) else None
                state["last_epoch"] = ep
                state["epochs"].append(dict(epoch=ep, secs=secs, train=tr, val=vl))
                events.append(f"epoch {ep}/{tot} {secs}s train={tr:.4f}"
                              + (f" val={vl:.4f}" if vl is not None else ""))
            c = COVER_RE.search(line)
            if c:
                state["cover"] = dict(hidden=float(c.group(1)),
                                      visible=float(c.group(2)),
                                      floor_ok=float(c.group(3)))
        state["pos"] = fh.tell()
    return events


def health_flags(state: dict, baseline_epoch_s: float, val_baseline: dict):
    """Return (warnings, abort_reason).

    ``abort_reason`` is non-None when training should be STOPPED and not
    relaunched.  A warning-only monitor is useless for a multi-day run: the
    blob arm's collapse was visible in val loss by epoch 30 (0.1527 vs
    envelope's 0.1200, a 27.3% excess) and nothing acted on it.
    """
    flags, abort = [], None
    eps = state["epochs"]
    if eps:
        last = eps[-1]
        if last["secs"] > 1.5 * baseline_epoch_s:
            flags.append(f"SLOW epoch {last['epoch']}: {last['secs']}s vs "
                         f"{baseline_epoch_s:.0f}s baseline")
        bl = val_baseline.get(str(last["epoch"]))
        if bl and last["val"]:
            ratio = last["val"] / bl
            if ratio > 1.35:
                abort = (f"val loss {last['val']:.4f} at ep{last['epoch']} is "
                         f"{ratio:.2f}x the baseline {bl:.4f} (>1.35x hard stop)")
            elif ratio > 1.20:
                state["consec_high"] = state.get("consec_high", 0) + 1
                flags.append(f"VAL LOSS HIGH ep{last['epoch']}: {last['val']:.4f} "
                             f"= {ratio:.2f}x baseline "
                             f"({state['consec_high']} consecutive)")
                if state["consec_high"] >= 2:
                    abort = (f"val loss above 1.20x baseline for "
                             f"{state['consec_high']} consecutive epochs "
                             f"(last {ratio:.2f}x) -- collapse signature")
            else:
                state["consec_high"] = 0
    c = state.get("cover")
    # Only meaningful once the ramp has actually engaged COVER.  During the
    # warm-up (r_t = 0) every block is a stock uniform rectangle, no COVER mask
    # is produced, and floor_ok is trivially 0 -- flagging that would cry wolf
    # for several epochs and desensitise the log to a real regression.
    if c and c["hidden"] > 0.0 and c["floor_ok"] < 0.999:
        flags.append(f"FLOOR REGRESSION: cover_floor_ok={c['floor_ok']:.3f} "
                     f"(hidden={c['hidden']:.3f})")
    return flags, abort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--python", default=r"D:\jepa_phase0\.venv\Scripts\python.exe")
    ap.add_argument("--max_restarts", type=int, default=8)
    ap.add_argument("--baseline_epoch_s", type=float, default=4500.0)
    ap.add_argument("--val_baseline_json", default="")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--state_dir", default=r"D:\jepa_phase0\campaign")
    args = ap.parse_args()

    base_cfg = pathlib.Path(args.config).resolve()
    cfg = yaml.safe_load(base_cfg.read_text())
    run_dir = pathlib.Path(cfg["logging"]["folder"])
    tag = cfg["logging"]["write_tag"]
    target_epochs = int(cfg["optimization"]["epochs"])
    run_dir.mkdir(parents=True, exist_ok=True)
    state_dir = pathlib.Path(args.state_dir); state_dir.mkdir(parents=True, exist_ok=True)

    val_baseline = {}
    if args.val_baseline_json and pathlib.Path(args.val_baseline_json).exists():
        val_baseline = json.loads(pathlib.Path(args.val_baseline_json).read_text())

    sup_log = state_dir / "supervisor.log"

    def say(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(sup_log, "a") as fh:
            fh.write(line + "\n")

    say(f"supervisor start: config={base_cfg} run_dir={run_dir} target_ep={target_epochs}")

    restarts = 0
    last_progress_epoch = -1
    while restarts <= args.max_restarts:
        ck = find_last(run_dir, tag)
        if ck is not None:
            ep = ckpt_epoch(ck)
            if ep >= target_epochs:
                say(f"DONE: rolling checkpoint already at epoch {ep} >= {target_epochs}")
                return 0
            say(f"resuming from {ck.name} (epoch {ep})")
            cfg_path = build_resume_config(base_cfg, ck, state_dir)
            # Guard: if a restart makes no progress, the failure is deterministic.
            if ep <= last_progress_epoch and restarts > 0:
                say(f"ABORT: restart made no progress (still epoch {ep}); "
                    f"failure looks deterministic, not transient")
                return 2
            last_progress_epoch = ep
        else:
            say("no rolling checkpoint yet -> initial launch from the configured fork")
            cfg_path = base_cfg

        run_log = run_dir / f"train_attempt{restarts}.log"
        say(f"launching attempt {restarts} -> {run_log.name}")
        env = dict(os.environ, PYTHONPATH=str(pathlib.Path(__file__).resolve().parents[1]))
        with open(run_log, "w") as lf:
            proc = subprocess.Popen(
                [args.python, "-u", "src/train_patch.py", "--config", str(cfg_path)],
                cwd=str(pathlib.Path(__file__).resolve().parents[1]),
                stdout=lf, stderr=subprocess.STDOUT, env=env)

        st = {"pos": 0, "epochs": [], "last_epoch": -1}
        aborted = None
        while proc.poll() is None:
            time.sleep(args.poll)
            for ev in scan_log(run_log, st):
                say("  " + ev)
            fl, abort = health_flags(st, args.baseline_epoch_s, val_baseline)
            for f in fl:
                say("  !! " + f)
            if abort:
                aborted = abort
                say(f"  *** ABORTING RUN: {abort}")
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except Exception:  # noqa: BLE001
                    proc.kill()
                break
            (state_dir / "health.json").write_text(json.dumps(st, indent=2))

        rc = proc.returncode
        for ev in scan_log(run_log, st):
            say("  " + ev)
        if aborted:
            say(f"STOPPED by health gate: {aborted}")
            say("Not relaunching. The rolling checkpoint is intact for inspection.")
            return 4
        if rc == 0:
            ck = find_last(run_dir, tag)
            ep = ckpt_epoch(ck) if ck else -1
            if ep >= target_epochs:
                say(f"COMPLETE at epoch {ep}")
                return 0
            say(f"exited cleanly at epoch {ep} but target is {target_epochs}; relaunching")
        else:
            say(f"CRASH rc={rc}; will restart ({restarts + 1}/{args.max_restarts})")
            tail = (run_log.read_text(errors="ignore").splitlines() or [])[-25:]
            for t in tail:
                say("    | " + t)
        restarts += 1
        time.sleep(30)

    say("ABORT: exceeded max restarts")
    return 3


if __name__ == "__main__":
    sys.exit(main())
