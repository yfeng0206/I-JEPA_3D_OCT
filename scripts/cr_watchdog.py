#!/usr/bin/env python
"""Resource watchdog for the cr_seed_v1 campaign (separate, long-running, read-only).

Every ``--interval`` seconds (default 60) appends one row to
``<state_dir>/watchdog.csv``: RAM available, commit charge used/limit, pagefile
usage, GPU memory/util/temperature/power, trainer PID liveness (from the campaign
state file; PID + creation time), latest logged epoch/iteration, seconds since the
trainer log last changed, and free space on C: and D:.

Writes ``<state_dir>/ALERT_<KIND>_<timestamp>.txt`` when
  RAM_LOW        available RAM < 700 MB for 5 consecutive samples
  COMMIT         commit headroom (limit - used) < 1.5 GiB AND C: free < 6 GiB.  The
                 system-managed pagefile on C: grows the commit limit on demand
                 (training adds ~30 GB of commit; limit observed 48-53 GB), so low
                 headroom alone is normal while C: still has room to grow it.
  GPU_MEM        GPU memory used > 23.5 GiB (training normally holds ~21.7 GiB incl. desktop)
  GPU_TEMP       GPU temperature >= 87 C
  LOG_STALL      no new trainer log output for 30 min while the trainer is alive
  DISK_D/DISK_C  free space on D: < 40 GiB / C: < 8 GiB
  TRAINER_SUSPENDED  a process in the trainer's own tree has all threads suspended
                 for 2 consecutive samples (e.g. a pause_ctl suspend)
  GPU_FORBIDDEN  a process matching the campaign's forbid_gpu_process_regex holds a
                 GPU context while the trainer is alive (e.g. a game)
While the campaign is PAUSED (cr_campaign.py --request-pause) LOG_STALL, TRAINER_SUSPENDED
and GPU_FORBIDDEN are suppressed and current_step shows "(PAUSED)"; hardware alerts stay on.
An alert is written at onset and repeated every ``--realert-minutes`` while the
condition persists.  The watchdog never kills or signals any process and opens
no process other than the tracked trainer (PID/creation time only).

    python scripts/cr_watchdog.py                       # run forever
    python scripts/cr_watchdog.py --once                # one sample, print it
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cr_common as cc  # noqa: E402

DEFAULT_STATE_DIR = r"D:\jepa_phase0\campaign\cr_seed_v1"
DEFAULT_CAMPAIGN = SCRIPTS.parent / "configs" / "cr_seed_v1" / "campaign.json"

THRESHOLDS = {
    "ram_low_mb": 700.0, "ram_low_consecutive": 5,
    "commit_headroom_mb": 1.5 * 1024, "commit_c_free_gib": 6.0,
    "gpu_mem_mb": 23.5 * 1024,
    "gpu_temp_c": 87.0,
    "log_stall_s": 30 * 60,
    "disk_d_gib": 40.0, "disk_c_gib": 8.0,
    "suspended_consecutive": 2,
}

COLUMNS = ["timestamp", "ram_avail_mb", "commit_used_mb", "commit_limit_mb", "commit_headroom_mb",
           "pagefile_used_mb", "pagefile_total_mb", "gpu_mem_used_mb", "gpu_mem_total_mb",
           "gpu_util_pct", "gpu_temp_c", "gpu_power_w", "gpu_power_limit_w", "gpu_forbidden_apps",
           "current_step", "trainer_pid", "trainer_alive", "trainer_suspended", "latest_epoch", "latest_iter",
           "seconds_since_log", "disk_free_c_gib", "disk_free_d_gib", "errors"]

ITER_RE = re.compile(r"\[Epoch (\d+)/\d+ \| Iter (\d+)/(\d+)\]")
EPOCH_RE = re.compile(r"^Epoch (\d+)/\d+\s+\(", re.M)


def latest_progress(log_path) -> dict:
    out = {"latest_epoch": None, "latest_iter": None, "log_size": None, "log_mtime": None}
    if not log_path or not os.path.exists(log_path):
        return out
    st = os.stat(log_path)
    out["log_size"], out["log_mtime"] = st.st_size, st.st_mtime
    text = cc.tail_text(log_path, 65536)
    it = ITER_RE.findall(text)
    if it:
        e, i, n = it[-1]
        out["latest_iter"] = "%s:%s/%s" % (e, i, n)
    ep = EPOCH_RE.findall(text)
    if ep:
        out["latest_epoch"] = int(ep[-1])
    return out


def default_readers(forbid_regex=None) -> dict:
    def gpu_forbidden():
        if not forbid_regex:
            return []
        return [os.path.basename(a.get("name", "")) for a in cc.gpu_compute_apps()
                if re.search(forbid_regex, a.get("name", ""), re.I)]
    return {"memory": cc.memory_status, "gpu": cc.gpu_status, "gpu_forbidden": gpu_forbidden,
            "disk": cc.disk_free_gib, "alive": cc.same_process_alive, "progress": latest_progress,
            "suspended": cc.suspended_in_tree,
            "now": time.time}


class Watchdog:
    def __init__(self, state_dir, *, readers=None, thresholds=None, realert_s=3600.0, forbid_regex=None):
        self.state_dir = Path(state_dir)
        self.csv_path = self.state_dir / "watchdog.csv"
        self.readers = readers or default_readers(forbid_regex)
        self.th = dict(THRESHOLDS, **(thresholds or {}))
        self.realert_s = float(realert_s)
        self.ram_low_run = 0
        self.suspended_run = 0
        self.last_alert = {}
        self._log_seen = {}  # log path -> (size, time the size last changed)

    def sample(self) -> dict:
        r = self.readers
        row = {c: None for c in COLUMNS}
        errors = []
        row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["now"]()))
        try:
            m = r["memory"]()
            row.update({k: m.get(k) for k in ("ram_avail_mb", "commit_used_mb", "commit_limit_mb",
                                               "pagefile_used_mb", "pagefile_total_mb")})
            if m.get("commit_limit_mb") is not None and m.get("commit_used_mb") is not None:
                row["commit_headroom_mb"] = round(m["commit_limit_mb"] - m["commit_used_mb"], 1)
        except Exception as e:  # noqa: BLE001
            errors.append("memory:%r" % (e,))
        try:
            row.update(r["gpu"]())
        except Exception as e:  # noqa: BLE001
            errors.append("gpu:%r" % (e,))
        try:
            fb = r["gpu_forbidden"]()
            row["gpu_forbidden_apps"] = ";".join(fb) if fb else ""
        except Exception as e:  # noqa: BLE001
            errors.append("gpu_apps:%r" % (e,))
        st = cc.read_json(self.state_dir / "state.json", {}) or {}
        cur = st.get("current") or {}
        # Graceful campaign pause (cr_campaign.py --request-pause): no trainer by design.
        row["paused"] = st.get("status") == "PAUSED" or bool(cur.get("paused"))
        row["current_step"] = ("%s (PAUSED)" % (cur.get("step") or "campaign")) if row["paused"] else cur.get("step")
        row["trainer_pid"] = cur.get("pid")
        if cur.get("pid"):
            try:
                row["trainer_alive"] = bool(r["alive"](cur.get("pid"), cur.get("create_time")))
            except Exception as e:  # noqa: BLE001
                errors.append("alive:%r" % (e,))
        else:
            row["trainer_alive"] = False
        if row["trainer_alive"] and "suspended" in r:
            try:
                sus = r["suspended"](cur.get("pid"), cur.get("create_time"))
                row["trainer_suspended"] = ";".join(str(x) for x in sus) if sus else ""
            except Exception as e:  # noqa: BLE001
                errors.append("suspended:%r" % (e,))
        try:
            pg = r["progress"](cur.get("log"))
            row["latest_epoch"], row["latest_iter"] = pg.get("latest_epoch"), pg.get("latest_iter")
            if pg.get("log_size") is not None:
                # NTFS may defer mtime updates for a file held open by the writer,
                # so a size change seen by this process also counts as activity.
                now = r["now"]()
                size, changed = self._log_seen.get(cur.get("log"), (None, None))
                if size != pg["log_size"]:
                    # First sight counts as activity too: an open NTFS log can carry a stale
                    # mtime, so a stall needs an observed interval of unchanged size.
                    changed = now
                    self._log_seen[cur.get("log")] = (pg["log_size"], changed)
                last = max(pg["log_mtime"], changed or pg["log_mtime"])
                row["seconds_since_log"] = round(max(0.0, now - last), 1)
        except Exception as e:  # noqa: BLE001
            errors.append("progress:%r" % (e,))
        row["disk_free_c_gib"] = r["disk"]("C:\\")
        row["disk_free_d_gib"] = r["disk"]("D:\\")
        row["errors"] = ";".join(errors)
        return row

    def evaluate(self, row) -> list[tuple[str, str]]:
        th = self.th
        conds = {}
        ram = row.get("ram_avail_mb")
        if ram is not None and ram < th["ram_low_mb"]:
            self.ram_low_run += 1
        else:
            self.ram_low_run = 0
        conds["RAM_LOW"] = (self.ram_low_run >= th["ram_low_consecutive"],
                            "available RAM %s MB < %s MB for %d consecutive samples"
                            % (ram, th["ram_low_mb"], self.ram_low_run))
        paused = bool(row.get("paused"))
        if row.get("trainer_suspended") and not paused:
            self.suspended_run += 1
        else:
            self.suspended_run = 0
        conds["TRAINER_SUSPENDED"] = (self.suspended_run >= th["suspended_consecutive"],
                                      "trainer tree pid(s) %s fully suspended for %d samples (step %s); "
                                      "nothing was resumed or killed; check whether the coordinator pause (pause_ctl.ps1: paused_pids.txt / "
                                      "pause_watch.stop) is active"
                                      % (row.get("trainer_suspended"), self.suspended_run, row.get("current_step")))
        hr = row.get("commit_headroom_mb")
        cfree = row.get("disk_free_c_gib")
        conds["COMMIT"] = (hr is not None and hr < th["commit_headroom_mb"]
                           and cfree is not None and cfree < th["commit_c_free_gib"],
                           "commit headroom %s MB < %s MB (used %s / limit %s MB) and C: free %s GiB < %s GiB "
                           "(the pagefile cannot grow much further)"
                           % (hr, th["commit_headroom_mb"], row.get("commit_used_mb"), row.get("commit_limit_mb"),
                              cfree, th["commit_c_free_gib"]))
        gm = row.get("gpu_mem_used_mb")
        conds["GPU_MEM"] = (gm is not None and gm > th["gpu_mem_mb"],
                            "GPU memory used %s MiB > %s MiB" % (gm, th["gpu_mem_mb"]))
        gt = row.get("gpu_temp_c")
        conds["GPU_TEMP"] = (gt is not None and gt >= th["gpu_temp_c"],
                             "GPU temperature %s C >= %s C" % (gt, th["gpu_temp_c"]))
        ssl = row.get("seconds_since_log")
        conds["LOG_STALL"] = (not paused and bool(row.get("trainer_alive")) and ssl is not None
                              and ssl > th["log_stall_s"],
                              "no trainer log output for %s s while pid %s is alive (step %s, last %s)"
                              % (ssl, row.get("trainer_pid"), row.get("current_step"), row.get("latest_iter")))
        dd, dc = row.get("disk_free_d_gib"), row.get("disk_free_c_gib")
        conds["DISK_D"] = (dd is not None and dd < th["disk_d_gib"], "D: free %s GiB < %s GiB" % (dd, th["disk_d_gib"]))
        conds["DISK_C"] = (dc is not None and dc < th["disk_c_gib"], "C: free %s GiB < %s GiB" % (dc, th["disk_c_gib"]))
        conds["GPU_FORBIDDEN"] = (not paused and bool(row.get("trainer_alive")) and bool(row.get("gpu_forbidden_apps")),
                                  "forbidden GPU process(es) while training: %s" % row.get("gpu_forbidden_apps"))
        now = time.time()
        fired = []
        for kind, (on, text) in conds.items():
            if not on:
                self.last_alert.pop(kind, None)
                continue
            last = self.last_alert.get(kind)
            if last is None or now - last >= self.realert_s:
                self.last_alert[kind] = now
                fired.append((kind, text))
        return fired

    def append(self, row) -> None:
        new = not self.csv_path.exists()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def step(self) -> tuple[dict, list]:
        row = self.sample()
        self.append(row)
        alerts = self.evaluate(row)
        paths = []
        for kind, text in alerts:
            paths.append(cc.write_alert(self.state_dir, kind, text + "\nsample: " + json.dumps(row, default=str)))
        return row, paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--realert-minutes", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    camp = cc.read_json(args.campaign, {}) or {}
    state_dir = args.state_dir or camp.get("state_dir") or DEFAULT_STATE_DIR
    forbid = (camp.get("preflight") or {}).get("forbid_gpu_process_regex")
    wd = Watchdog(state_dir, realert_s=args.realert_minutes * 60, forbid_regex=forbid)
    if args.once:
        row, paths = wd.step()
        print(json.dumps(row, indent=1, default=str))
        for p in paths:
            print("ALERT", p)
        return 0
    lock = cc.ExclusiveLock(Path(state_dir) / "watchdog.lock")
    try:
        lock.acquire()
    except cc.LockHeld as e:
        print("LOCKED: %s" % e)
        return 3
    try:
        while True:
            t0 = time.time()
            try:
                row, paths = wd.step()
                for p in paths:
                    print("[%s] ALERT %s" % (row["timestamp"], p), flush=True)
            except Exception as e:  # noqa: BLE001  (keep watching)
                print("[%s] sample failed: %r" % (time.strftime("%H:%M:%S"), e), flush=True)
            time.sleep(max(1.0, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
