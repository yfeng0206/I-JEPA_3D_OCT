"""Autopilot state manager.

Single source of truth = autopilot/state.json.
Renders CURRENT_STATUS.md, TASK_LEDGER.md, AGENT_STATUS.md, RUN_STATE.json,
PROCESS_REGISTRY.csv and RESUME_COMMAND.txt so the operator can recover the run
from any point.

Usage:
  python state.py init
  python state.py task <id> <status> [note]
  python state.py agent <id> <field=value> ...
  python state.py proc  <job_id> <field=value> ...
  python state.py phase <phase> [note]
  python state.py decide "<decision>" "<rationale>"
  python state.py render
"""
import json
import os
import sys
import csv
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
REPO = os.path.dirname(HERE)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load():
    if not os.path.exists(STATE):
        return {"started": now(), "phase": "P0", "phase_note": "", "tasks": {},
                "agents": {}, "procs": {}, "decisions": [], "updated": now()}
    with open(STATE, "r", encoding="utf-8") as f:
        return json.load(f)


def save(s):
    s["updated"] = now()
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)


def sample_resources():
    """Best-effort live resource snapshot for CURRENT_STATUS.md."""
    out = {"gpu": "unknown", "ram": "unknown", "disk": "unknown"}
    try:
        g = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=25)
        if g.returncode == 0 and g.stdout.strip():
            u, t, mu, mt = [x.strip() for x in g.stdout.strip().splitlines()[0].split(",")]
            out["gpu"] = "util %s%%, %s C, %s / %s MiB (%.1f%%)" % (
                u, t, mu, mt, 100.0 * int(mu) / int(mt))
    except Exception as e:
        out["gpu"] = "sample failed: %s" % e
    try:
        import shutil
        parts = []
        for d in ("C:\\", "D:\\"):
            if os.path.exists(d):
                tot, used, free = shutil.disk_usage(d)
                parts.append("%s %.1f GB free (%.1f%%)" % (d[0], free / 1e9, 100.0 * free / tot))
        out["disk"] = "; ".join(parts)
    except Exception as e:
        out["disk"] = "sample failed: %s" % e
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        out["ram"] = "%.1f / %.1f GB used (%d%%)" % (
            (m.ullTotalPhys - m.ullAvailPhys) / 1e9, m.ullTotalPhys / 1e9, m.dwMemoryLoad)
    except Exception as e:
        out["ram"] = "sample failed: %s" % e
    return out


TASK_ORDER = ["pending", "in_progress", "blocked", "done", "failed", "skipped"]


def render(s):
    res = sample_resources()
    tasks = s.get("tasks", {})
    agents = s.get("agents", {})
    procs = s.get("procs", {})

    def bystat(st):
        return [(k, v) for k, v in sorted(tasks.items()) if v.get("status") == st]

    # ---- TASK_LEDGER.md
    L = ["# TASK LEDGER", "", "Updated: %s" % now(), "",
         "| id | phase | task | status | owner | started | expected | verified | retries | output |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(tasks.items(), key=lambda kv: (kv[1].get("phase", ""), kv[0])):
        L.append("| %s | %s | %s | **%s** | %s | %s | %s | %s | %s | %s |" % (
            k, v.get("phase", ""), v.get("title", ""), v.get("status", ""), v.get("owner", "coordinator"),
            v.get("started", "-"), v.get("expected", "-"), v.get("verified", "no"),
            v.get("retries", 0), v.get("output", "-")))
    L.append("")
    L.append("## Counts")
    for st in TASK_ORDER:
        L.append("- %s: %d" % (st, len(bystat(st))))
    write(os.path.join(HERE, "TASK_LEDGER.md"), "\n".join(L))

    # ---- AGENT_STATUS.md
    A = ["# AGENT STATUS", "", "Updated: %s" % now(), "",
         "| agent_id | model | task | start | expected | last_heartbeat | status | output | deps | retries | verification | next action |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(agents.items(), key=lambda kv: kv[1].get("start", "")):
        A.append("| `%s` | %s | %s | %s | %s | %s | **%s** | %s | %s | %s | %s | %s |" % (
            k, v.get("model", ""), v.get("task", ""), v.get("start", ""), v.get("expected", ""),
            v.get("last_heartbeat", "-"), v.get("status", ""), v.get("output", "-"),
            v.get("deps", "-"), v.get("retries", 0), v.get("verification", "pending"),
            v.get("next_action", "-")))
    if not agents:
        A.append("| (none yet) | | | | | | | | | | | |")
    write(os.path.join(HERE, "AGENT_STATUS.md"), "\n".join(A))

    # ---- PROCESS_REGISTRY.csv
    cols = ["job_id", "pid", "command", "owner", "input_artifacts", "start_time",
            "expected_completion", "last_output_time", "cpu_mem", "gpu_mem",
            "output_dir", "exit_code", "recovery_command"]
    with open(os.path.join(HERE, "PROCESS_REGISTRY.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k, v in sorted(procs.items()):
            row = {c: v.get(c, "") for c in cols}
            row["job_id"] = k
            w.writerow(row)

    # ---- RUN_STATE.json
    rs = {"run_started": s.get("started"), "updated": now(), "phase": s.get("phase"),
          "phase_note": s.get("phase_note", ""), "resources": res,
          "task_counts": {st: len(bystat(st)) for st in TASK_ORDER},
          "active_agents": [k for k, v in agents.items() if v.get("status") in ("running", "idle")],
          "active_procs": [k for k, v in procs.items() if v.get("exit_code", "") == ""],
          "tasks": tasks, "agents": agents, "procs": procs,
          "decisions": len(s.get("decisions", []))}
    write(os.path.join(HERE, "RUN_STATE.json"), json.dumps(rs, indent=1))

    # ---- CURRENT_STATUS.md
    nxt = [v.get("title", k) for k, v in sorted(tasks.items())
           if v.get("status") == "in_progress"][:3]
    if len(nxt) < 3:
        nxt += [v.get("title", k) for k, v in sorted(tasks.items(), key=lambda kv: kv[0])
                if v.get("status") == "pending"][:3 - len(nxt)]
    zipst = s.get("zip_status", "not yet built")
    C = ["# CURRENT STATUS", "",
         "Updated: **%s**" % now(),
         "Run started: %s" % s.get("started"), "",
         "## Current phase", "**%s** - %s" % (s.get("phase"), s.get("phase_note", "")), "",
         "## Completed work",
         ]
    done = bystat("done")
    if done:
        for k, v in done:
            C.append("- [%s] %s%s" % (k, v.get("title", ""),
                     "  (verified: %s)" % v.get("verified") if v.get("verified") else ""))
    else:
        C.append("- (none yet)")
    C += ["", "## Active subagents"]
    act = [(k, v) for k, v in agents.items() if v.get("status") in ("running", "idle")]
    if act:
        for k, v in act:
            C.append("- `%s` %s - %s (last heartbeat %s)" % (k, v.get("model", ""), v.get("task", ""), v.get("last_heartbeat", "-")))
    else:
        C.append("- (none)")
    C += ["", "## Active processes"]
    ap = [(k, v) for k, v in procs.items() if v.get("exit_code", "") == ""]
    if ap:
        for k, v in ap:
            C.append("- `%s` pid=%s %s" % (k, v.get("pid", "?"), v.get("command", "")[:90]))
    else:
        C.append("- (none)")
    C += ["", "## Resources",
          "- GPU: %s" % res["gpu"],
          "- RAM: %s" % res["ram"],
          "- Disk: %s" % res["disk"], ""]
    C += ["## Blockers"]
    bl = bystat("blocked")
    if bl:
        for k, v in bl:
            C.append("- [%s] %s - %s" % (k, v.get("title", ""), v.get("note", "")))
    else:
        C.append("- (none)")
    C += ["", "## Next three actions"]
    for i, t in enumerate(nxt if nxt else ["(all tasks complete)"], 1):
        C.append("%d. %s" % (i, t))
    C += ["", "## Revised completion estimate",
          s.get("eta", "see TIMELINE_AND_CRITICAL_PATH.md"), "",
          "## Final ZIP status", zipst, ""]
    write(os.path.join(HERE, "CURRENT_STATUS.md"), "\n".join(C))

    # ---- RESUME_COMMAND.txt
    R = ["# How to resume this autopilot run",
         "# Generated %s" % now(), "",
         "# 1. Restart the resource monitor (if not running):",
         "powershell -NoProfile -ExecutionPolicy Bypass -File "
         r"C:\Users\Gary\Desktop\jepa\autopilot\resource_monitor.ps1 -IntervalSec 45",
         "",
         "# 2. Inspect state:",
         r"type C:\Users\Gary\Desktop\jepa\autopilot\CURRENT_STATUS.md",
         r"type C:\Users\Gary\Desktop\jepa\autopilot\TASK_LEDGER.md",
         "",
         "# 3. Python for all analysis (torch lives here, NOT in system python):",
         r"D:\jepa_phase0\.venv\Scripts\python.exe",
         "",
         "# 4. LaTeX build:",
         r'& "D:\jepa_phase0\tools\tectonic\tectonic.exe" -X compile main_submission.tex --keep-intermediates',
         r"#   (run from C:\Users\Gary\Desktop\jepa\paper\genai4health2026)",
         "",
         "# 5. HARD CONSTRAINT - do NOT resume pretraining. COVER-0.21 stays at epoch 73.",
         "#    Forbidden: scripts\\chain_cover_f021.py, scripts\\campaign_chain.py, any src/train*.py",
         "",
         "# 6. Outstanding in-progress tasks at last render:"]
    for k, v in bystat("in_progress"):
        R.append("#    - [%s] %s -> %s" % (k, v.get("title", ""), v.get("next", "resume")))
    write(os.path.join(HERE, "RESUME_COMMAND.txt"), "\n".join(R))

    # ---- DECISION_LOG.md
    D = ["# DECISION LOG", "", "Autonomous decisions taken while the operator is away.",
         "Policy: inspect evidence -> choose the most conservative scientifically valid option -> record -> continue.", ""]
    for d in s.get("decisions", []):
        D += ["## %s - %s" % (d["time"], d["decision"]), "", d["rationale"], ""]
    write(os.path.join(HERE, "DECISION_LOG.md"), "\n".join(D))


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    s = load()
    if len(sys.argv) < 2:
        render(s)
        return
    cmd = sys.argv[1]
    if cmd == "task":
        tid, st = sys.argv[2], sys.argv[3]
        t = s["tasks"].setdefault(tid, {})
        t["status"] = st
        if st == "in_progress" and "started" not in t:
            t["started"] = now()
        if len(sys.argv) > 4:
            t["note"] = " ".join(sys.argv[4:])
        if st in ("done", "failed"):
            t["finished"] = now()
    elif cmd == "addtask":
        tid = sys.argv[2]
        t = s["tasks"].setdefault(tid, {"status": "pending", "retries": 0})
        for kv in sys.argv[3:]:
            k, _, v = kv.partition("=")
            t[k] = v
    elif cmd == "agent":
        aid = sys.argv[2]
        a = s["agents"].setdefault(aid, {"retries": 0})
        for kv in sys.argv[3:]:
            k, _, v = kv.partition("=")
            a[k] = v
        a["last_heartbeat"] = now()
        with open(os.path.join(HERE, "SUBAGENT_HEARTBEATS.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": now(), "agent_id": aid, **a}) + "\n")
    elif cmd == "proc":
        jid = sys.argv[2]
        p = s["procs"].setdefault(jid, {})
        for kv in sys.argv[3:]:
            k, _, v = kv.partition("=")
            p[k] = v
    elif cmd == "phase":
        s["phase"] = sys.argv[2]
        s["phase_note"] = " ".join(sys.argv[3:])
    elif cmd == "set":
        for kv in sys.argv[2:]:
            k, _, v = kv.partition("=")
            s[k] = v
    elif cmd == "decide":
        s.setdefault("decisions", []).append(
            {"time": now(), "decision": sys.argv[2], "rationale": sys.argv[3]})
    elif cmd == "init":
        pass
    save(s)
    render(s)
    print("state updated:", cmd)


if __name__ == "__main__":
    main()
