#!/usr/bin/env python
"""Fail-closed, resumable, single-instance runner for the cr_seed_v1 campaign.

Reads configs/cr_seed_v1/campaign.json (ordered ``sequence`` of ``<RUN>:train`` /
``<RUN>:probe`` steps plus the ``ANCHORS:probe`` step) and executes it one GPU job
at a time.  State lives in ``<state_dir>/state.json`` (atomic writes) so the runner
can be stopped and restarted at any time; a trainer or probe that is still
running is re-attached, never relaunched.

Per training run:
  * the pinned git commit (``git_commit`` in campaign.json) is checked out as a
    detached worktree under ``<worktree_root>/<sha>``; trainer, configs and probe all
    come from that worktree;
  * attempt 0 forks from the shared epoch-25 ancestor (SHA-256 verified) with the
    generated config and ``--stop-after-epoch <stop_epoch>``; stdout/stderr go to
    ``<run_dir>/train_a<k>.log``;
  * after each logged epoch the G1 gate (scripts/cr_gate_check.py) runs; WARN/HOLD are
    recorded, HOLD writes an ALERT file and training continues; STOP-class results
    (NaN/Inf, wrong horizon/start epoch) stop the campaign and terminate the
    runner's OWN trainer process tree (the trainer has no non-finite abort);
  * at the first logged epoch of every attempt the trainer's run_manifest.json must
    match the launch (pinned commit, clean tree, config file SHA-256, stop epoch, fork/
    exact policy, seed, fp32 teacher, 64 x 8 x 1 = 512, horizon 100, ENVELOPE legacy
    sampler); a mismatch terminates the runner's own trainer and stops the campaign;
  * exit 0 requires ``stop_epoch_<NNN>.json``; the periodic epoch-N checkpoint it names
    is re-hashed and deserialized (``epoch == N``) and copied atomically to
    ``<run_dir>/pinned/<tag>-ep<NNN>.pth.tar`` (re-hashed after the copy);
  * a crash (nonzero/unknown exit, no stop file) resumes with ``resume_policy: exact``
    from ``<tag>-last.pth.tar`` (config regenerated without fork fields), at most
    ``max_restarts`` times between coordinator acknowledgements; a started run is never
    re-forked.
Per probe: fresh output directory named with the checkpoint hash prefix and the
attempt number, required outputs created after launch, encoder SHA-256 recorded by
the probe == pinned checkpoint, sealed-test manifest present, no numeric test AUC,
pinned checkpoint unchanged afterwards.

Any failed check stops the campaign (exit 2) with ``status: STOPPED`` and an
ALERT_STOP file; ``--clear-stop "<note>"`` acknowledges it.  The runner never kills
a process it did not start.

Usage:
  python scripts/cr_campaign.py --dry-run            # plan + preflight report, changes nothing
  python scripts/cr_campaign.py                      # run / resume the campaign
  python scripts/cr_campaign.py --only R1            # only R1's steps (or ANCHORS)
  python scripts/cr_campaign.py --status
  python scripts/cr_campaign.py --set-commit <sha> [--runs R1,C1,...]
  python scripts/cr_campaign.py --clear-stop "reason checked, safe to continue"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cr_common as cc  # noqa: E402
import cr_gate_check as gate  # noqa: E402
import cr_make_configs as mk  # noqa: E402

REPO = SCRIPTS.parent
DEFAULT_CAMPAIGN = REPO / "configs" / "cr_seed_v1" / "campaign.json"
STATE_SCHEMA = "cr_campaign_state_v1"
STOP_FILE_SCHEMA = "jepa_stop_epoch_v1"
SEAL_SCHEMA = "cr_sealed_v1"
EXIT_STOPPED = 2
EXIT_LOCKED = 3
EXIT_STOP_NOT_REACHED = 3  # trainer: training ended before --stop-after-epoch
EXIT_NONFINITE_LOSS = 4    # trainer: non-finite epoch loss, aborted without saving that epoch
# run_manifest.json / stop file `mask_policy` written by the trainer (i1, 04:00).
MASK_POLICY = {"random": "uniform_multiblock", "centroid": "anatomical_prior", "envelope": "mirage_envelope"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
NO_WINDOW = {"creationflags": CREATE_NO_WINDOW} if os.name == "nt" else {}  # no console flashes under pythonw

# Hooks (monkeypatched by tests).
gpu_compute_apps = cc.gpu_compute_apps
foreign_compute_python = cc.foreign_compute_python
memory_status = cc.memory_status
disk_free_gib = cc.disk_free_gib


class CampaignStop(RuntimeError):
    """A gate or identity check failed: the campaign must stop."""


def fmt_ts(t: float | None) -> str | None:
    if t is None:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


# ---------------------------------------------------------------------------
# campaign spec
# ---------------------------------------------------------------------------

def load_campaign(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        camp = json.load(fh)
    errs = []
    for key in ("campaign", "repo", "python", "state_dir", "worktree_root", "ancestor",
                "stop_epoch", "max_restarts", "train_command", "probe", "runs", "sequence"):
        if key not in camp:
            errs.append("missing key %s" % key)
    if errs:
        raise ValueError("invalid campaign file %s: %s" % (path, "; ".join(errs)))
    ids = [r["id"] for r in camp["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run ids in %s" % path)
    known = set(ids) | ({camp["anchors"]["id"]} if camp.get("anchors") else set())
    for step in camp["sequence"]:
        rid, _, kind = step.partition(":")
        if rid not in known or kind not in ("train", "probe"):
            raise ValueError("bad sequence step %r" % step)
    for r in camp["runs"]:
        if int(r.get("stop_epoch", camp["stop_epoch"])) != int(camp["stop_epoch"]):
            raise ValueError("run %s stop_epoch differs from the campaign stop_epoch" % r["id"])
    return camp


def run_spec(camp: dict, rid: str) -> dict:
    if camp.get("anchors") and camp["anchors"]["id"] == rid:
        return camp["anchors"]
    for r in camp["runs"]:
        if r["id"] == rid:
            return r
    raise KeyError(rid)


def frozen_spec(spec: dict) -> dict:
    """Fields of a run that must not change once the run has started."""
    keys = ("id", "arm", "seed", "config", "git_commit", "stop_epoch")
    return {k: spec.get(k) for k in keys}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, camp_path, *, poll_seconds=None, out=print):
        self.camp_path = str(camp_path)
        self.camp = load_campaign(camp_path)
        self.state_dir = Path(self.camp["state_dir"])
        self.state_path = self.state_dir / "state.json"
        self.lock = cc.ExclusiveLock(self.state_dir / "campaign.lock")
        self.poll = float(poll_seconds if poll_seconds is not None else self.camp.get("poll_seconds", 30))
        self.out = out
        self.children = {}  # pid -> Popen (only processes this runner started)
        self.state = None
        self.repo = Path(self.camp["repo"])
        self.stop_epoch = int(self.camp["stop_epoch"])
        self.fork_start = int(self.camp.get("fork_start_epoch", 25))
        self.run_root = self.camp.get("run_root", mk.RUN_ROOT)
        self.readonly = False  # --dry-run / --status: never write anything

    # -- logging / state --------------------------------------------------
    def say(self, msg: str) -> None:
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        self.out(line)
        if self.readonly:
            return
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_dir / "campaign.log", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def load_state(self) -> dict:
        st = cc.read_json(self.state_path)
        if st is None:
            st = {"schema": STATE_SCHEMA, "campaign": self.camp["campaign"], "status": "NEW",
                  "created_at": cc.now_iso(), "runs": {}, "events": [], "acks": []}
        if st.get("schema") != STATE_SCHEMA or st.get("campaign") != self.camp["campaign"]:
            raise CampaignStop("state file %s belongs to another campaign/schema" % self.state_path)
        self.state = st
        return st

    def save(self) -> None:
        self.state["updated_at"] = cc.now_iso()
        cc.atomic_write_json(self.state_path, self.state)

    def event(self, msg: str, **kw) -> None:
        self.state.setdefault("events", []).append(dict({"ts": cc.now_iso(), "msg": msg}, **kw))
        self.state["events"] = self.state["events"][-500:]
        self.say(msg)

    def rstate(self, rid: str) -> dict:
        rs = self.state["runs"].setdefault(rid, {})
        rs.setdefault("train", {"status": "PENDING", "attempts": [], "restarts_since_ack": 0,
                                "epochs": {}, "gates": {}})
        rs.setdefault("probe", {"status": "PENDING", "attempts": []})
        return rs

    def stop(self, msg: str):
        raise CampaignStop(msg)

    # -- spec freezing ----------------------------------------------------
    def check_frozen(self, rid: str, spec: dict) -> None:
        rs = self.rstate(rid)
        fz = frozen_spec(spec)
        if "spec" not in rs:
            rs["spec"] = fz
            return
        if rs["spec"] != fz:
            self.stop("%s spec changed after the run started: was %s now %s" % (rid, rs["spec"], fz))

    # -- git worktree -----------------------------------------------------
    def git(self, *args, cwd=None) -> str:
        r = subprocess.run(["git", "-C", str(cwd or self.repo)] + list(args), capture_output=True,
                           text=True, timeout=300, **NO_WINDOW)
        if r.returncode != 0:
            raise RuntimeError("git %s failed: %s" % (" ".join(args), (r.stderr or r.stdout).strip()))
        return r.stdout

    def worktree_path(self, commit: str) -> Path:
        return Path(self.camp["worktree_root"]) / commit

    def ensure_worktree(self, commit, *, create=True) -> Path:
        if not commit or not SHA_RE.match(str(commit)):
            self.stop("git_commit %r is not a full 40-hex SHA (set it with --set-commit)" % (commit,))
        try:
            self.git("cat-file", "-e", "%s^{commit}" % commit)
        except RuntimeError as e:
            self.stop("pinned commit %s not in repo %s: %s" % (commit, self.repo, e))
        wt = self.worktree_path(commit)
        if not wt.exists():
            if not create:
                return wt
            self.git("worktree", "prune")
            wt.parent.mkdir(parents=True, exist_ok=True)
            self.git("worktree", "add", "--detach", str(wt), commit)
            self.event("created worktree %s at %s" % (wt, commit))
        head = self.git("rev-parse", "HEAD", cwd=wt).strip()
        if head != commit:
            self.stop("worktree %s is at %s, expected %s (not touching it)" % (wt, head, commit))
        dirty = self.git("status", "--porcelain", "--untracked-files=no", cwd=wt).strip()
        if dirty:
            self.stop("worktree %s has modified tracked files:\n%s" % (wt, dirty[:2000]))
        untracked = [n for n in self.git("ls-files", "--others", "--exclude-standard", cwd=wt).splitlines()
                     if n.strip()]
        if untracked:
            self.stop("worktree %s has untracked files %s" % (wt, untracked[:20]))
        return wt

    # -- preflight --------------------------------------------------------
    def preflight(self, what: str, *, allow_pids=(), fatal=True, wait_gpu=True) -> dict:
        pf = self.camp.get("preflight", {})
        rep = {"what": what, "at": cc.now_iso(), "problems": []}
        try:
            rep["memory"] = memory_status()
        except Exception as e:  # noqa: BLE001
            rep["memory"] = {"error": repr(e)}
        rep["disk_free_gib"] = {}
        for drive, need in (pf.get("min_free_gib") or {}).items():
            free = disk_free_gib(drive)
            rep["disk_free_gib"][drive] = free
            if free is None or free < float(need):
                rep["problems"].append("disk %s free %s GiB < %s GiB" % (drive, free, need))
        allow = {int(p) for p in allow_pids if p}
        deadline = time.time() + (float(pf.get("gpu_clear_wait_seconds", 600)) if wait_gpu else 0)
        while True:
            gpu_problem = None
            try:
                py_apps, other_apps = cc.split_gpu_apps(
                    [a for a in gpu_compute_apps() if a["pid"] not in allow])
                rep["gpu_python_apps"] = py_apps
                rep["gpu_other_apps"] = sorted({os.path.basename(a.get("name", "")) for a in other_apps})
                if py_apps and pf.get("require_no_gpu_compute_apps", True):
                    gpu_problem = "python/torch processes on the GPU: %s" % py_apps
                rx = pf.get("forbid_gpu_process_regex")
                if rx:
                    bad = [a for a in other_apps if re.search(rx, a.get("name", ""), re.I)]
                    if bad:
                        gpu_problem = (gpu_problem + "; " if gpu_problem else "") + \
                            "forbidden GPU processes: %s" % [a.get("name") for a in bad]
            except Exception as e:  # noqa: BLE001
                gpu_problem = "cannot query GPU compute processes: %r" % (e,)
            procs = foreign_compute_python(exclude_pids=allow)
            rep["training_like_python"] = procs
            proc_problem = ("training/probe python processes running: %s" % procs) if procs else None
            if not (gpu_problem or proc_problem) or time.time() >= deadline:
                break
            time.sleep(min(15.0, max(0.05, deadline - time.time())))
        for p in (gpu_problem, proc_problem):
            if p:
                rep["problems"].append(p)
        mem = rep.get("memory", {})
        self.say("preflight %s: RAM avail %s MB, commit %s/%s MB, disk %s, gpu python apps %d, "
                 "other GPU clients %s, problems %d"
                 % (what, mem.get("ram_avail_mb"), mem.get("commit_used_mb"), mem.get("commit_limit_mb"),
                    rep["disk_free_gib"], len(rep.get("gpu_python_apps", [])),
                    rep.get("gpu_other_apps", []), len(rep["problems"])))
        if rep["problems"] and fatal:
            self.stop("preflight %s failed: %s" % (what, "; ".join(rep["problems"])))
        return rep

    def verify_ancestor(self, fatal=True) -> dict:
        anc = self.camp["ancestor"]
        res = {"path": anc["path"]}
        try:
            size = os.path.getsize(anc["path"])
            digest, cached = cc.cached_sha256(anc["path"], self.state_dir / "hash_cache.json",
                                              write=not self.readonly)
            res.update({"size": size, "sha256": digest, "from_cache": cached})
            ok = digest == anc["sha256"] and (anc.get("size") is None or size == int(anc["size"]))
        except OSError as e:
            res["error"] = repr(e)
            ok = False
        res["ok"] = ok
        if not ok and fatal:
            self.stop("ancestor check failed: %s (expected sha256 %s size %s)"
                      % (res, anc["sha256"], anc.get("size")))
        return res

    # -- processes --------------------------------------------------------
    def launch(self, argv, cwd, log_path) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(cwd)
        env["MPLBACKEND"] = "Agg"
        env["PYTHONUNBUFFERED"] = "1"
        for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                  "PYTORCH_CUDA_ALLOC_CONF"):  # allocator config kept at the benchmarked default
            env.pop(k, None)
        env.update({str(k): str(v) for k, v in (self.camp.get("child_env") or {}).items()})
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "xb")  # never overwrite an earlier attempt's log
        flags = (CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW) if cc.IS_WINDOWS else 0
        try:
            fh.write(("# cr_campaign launch %s\n# cwd %s\n# argv %s\n"
                      % (cc.now_iso(), cwd, json.dumps(argv))).encode("utf-8"))
            fh.flush()
            p = subprocess.Popen(argv, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 creationflags=flags, close_fds=True)
        finally:
            fh.close()
        self.children[p.pid] = p
        ct = cc.process_create_time(p.pid)
        return {"pid": p.pid, "create_time": ct, "started_at": time.time(),
                "started_at_iso": cc.now_iso(), "argv": argv, "cwd": str(cwd), "log": str(log_path)}

    def proc_alive(self, a: dict) -> bool:
        p = self.children.get(a.get("pid"))
        if p is not None:
            return p.poll() is None
        return cc.same_process_alive(a.get("pid"), a.get("create_time"))

    def proc_rc(self, a: dict):
        p = self.children.get(a.get("pid"))
        if p is not None:
            return p.poll()
        return None  # re-attached: exit code unknown

    def kill_own_tree(self, a: dict, why: str) -> None:
        """Terminate a process tree this runner launched (never anything else)."""
        pid = a.get("pid")
        own = pid in self.children or cc.same_process_alive(pid, a.get("create_time"))
        if not own or not cc.same_process_alive(pid, a.get("create_time")):
            return
        self.event("terminating OWN process tree pid=%s (%s)" % (pid, why))
        if cc.IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=120,
                           **NO_WINDOW)
        else:
            try:
                os.kill(int(pid), 15)
            except OSError:
                pass
        p = self.children.get(pid)
        if p is not None:
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass

    # -- training ---------------------------------------------------------
    def run_paths(self, cfg: dict) -> tuple[Path, str]:
        return Path(cfg["logging"]["folder"]), cfg["logging"]["write_tag"]

    def stop_file(self, run_dir: Path) -> Path:
        return run_dir / ("stop_epoch_%03d.json" % self.stop_epoch)

    def load_run_config(self, spec: dict, wt: Path) -> tuple[Path, dict, str]:
        import yaml
        cfg_path = wt / spec["config"]
        if not cfg_path.exists():
            self.stop("config %s is not in pinned commit %s" % (spec["config"], spec.get("git_commit")))
        text = cfg_path.read_bytes()
        cfg = yaml.safe_load(text.decode("utf-8"))
        try:
            mk.validate_run_config(cfg, spec["arm"], int(spec["seed"]), run_root=self.run_root,
                                   ancestor=self.camp["ancestor"]["path"])
        except ValueError as e:
            self.stop(str(e))
        return cfg_path, cfg, cc.sha256_bytes(text)

    def do_train(self, rid: str, spec: dict) -> None:
        rs = self.rstate(rid)
        tr = rs["train"]
        if tr["status"] == "DONE":
            return
        self.check_frozen(rid, spec)
        wt = self.ensure_worktree(spec.get("git_commit"))
        cfg_path, cfg, cfg_sha = self.load_run_config(spec, wt)
        run_dir, tag = self.run_paths(cfg)
        tr.update({"run_dir": str(run_dir), "tag": tag, "worktree": str(wt),
                   "fork_config": str(cfg_path), "fork_config_sha256": cfg_sha})
        if tr["status"] == "PENDING":
            tr["status"] = "RUNNING"
        self.state["current"] = {"step": "%s:train" % rid}
        self.save()
        max_restarts = int(self.camp["max_restarts"])
        while True:
            a = tr["attempts"][-1] if tr["attempts"] else None
            if a is None:
                if run_dir.exists() and any(run_dir.iterdir()):
                    self.stop("%s: run folder %s is not empty but the campaign never started this run"
                              % (rid, run_dir))
                self.verify_ancestor()
                self.preflight("%s fork launch" % rid)
                if not rs.get("run_uuid"):
                    import uuid
                    rs["run_uuid"] = str(uuid.uuid4())
                self.start_attempt(rid, spec, wt, cfg_path, cfg_sha, kind="fork")
                continue
            if a["status"] == "LAUNCHING":
                self.stop("%s attempt %d was being launched when the runner died; verify no trainer "
                          "is running, then --clear-stop" % (rid, a["k"]))
            if a["status"] == "RUNNING":
                self.monitor_train(rid, spec, a)
            if a["status"] == "EXITED":
                verdict = self.resolve_exit(rid, a, run_dir)
                if verdict == "OK":
                    if not (a.get("manifest_checked") or {}).get("ok"):
                        # e.g. the trainer's verify-only epoch-N recovery logs no new epoch line
                        self.check_manifest(rid, spec, a, run_dir)
                    self.verify_and_pin(rid, spec, run_dir, tag)
                    tr["status"] = "DONE"
                    self.state["current"] = None
                    self.save()
                    return
                # crash -> exact resume
                if tr["restarts_since_ack"] >= max_restarts:
                    self.stop("%s: trainer failed and the restart budget (%d) is used up; last log %s"
                              % (rid, max_restarts, a["log"]))
                last = run_dir / ("%s-last.pth.tar" % tag)
                if not last.exists():
                    self.stop("%s: trainer failed before the first rolling checkpoint (%s missing); "
                              "refusing to re-fork a started run" % (rid, last))
                ep = read_checkpoint_epoch(last, self.camp["python"])
                if ep > self.stop_epoch:
                    self.stop("%s: %s holds epoch %d > stop epoch %d; coordinator must inspect"
                              % (rid, last.name, ep, self.stop_epoch))
                if ep == self.stop_epoch:
                    # Crash between the epoch-N saves and the stop file: the trainer's exact
                    # resume with --stop-after-epoch N re-verifies <tag>-epN and -last, writes
                    # the stop file and exits 0 without training; the runner re-verifies anyway.
                    self.event("%s: -last is at the stop epoch %d without a stop file; exact resume will "
                               "verify and write the stop file without training" % (rid, ep))
                if ep <= self.fork_start:
                    self.stop("%s: rolling checkpoint epoch %d is not past the fork epoch %d"
                              % (rid, ep, self.fork_start))
                rcfg_path = self.write_resume_config(rid, spec, cfg, last, len(tr["attempts"]))
                self.preflight("%s exact resume from epoch %d" % (rid, ep))
                tr["restarts_since_ack"] += 1
                self.start_attempt(rid, spec, wt, rcfg_path, cc.sha256_file(rcfg_path), kind="exact",
                                   resume_from_epoch=ep)
                continue
            if a["status"] not in ("RUNNING", "EXITED"):
                self.stop("%s attempt %d in unexpected status %s" % (rid, a["k"], a["status"]))

    def write_resume_config(self, rid, spec, fork_cfg, last, k) -> Path:
        import yaml
        rcfg = mk.make_resume_config(fork_cfg, str(last))
        mk.validate_run_config(rcfg, spec["arm"], int(spec["seed"]), expect_fork=False,
                               run_root=self.run_root, ancestor=self.camp["ancestor"]["path"])
        if "fork_start_epoch" in rcfg["meta"] or rcfg["meta"]["resume_policy"] != "exact":
            self.stop("resume config for %s still carries fork fields" % rid)
        path = self.state_dir / "configs" / ("%s_resume_a%d.yaml" % (rid, k))
        if path.exists():
            # A runner that died between writing this file and journaling the attempt left it
            # behind: reuse it only if it is exactly the config we would write now.
            try:
                existing = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                existing = None
            if existing != rcfg:
                self.stop("resume config %s already exists with different content" % path)
            self.event("%s: reusing existing identical resume config %s" % (rid, path.name))
            return path
        text = ("# exact-resume config for %s attempt %d, generated by cr_campaign.py %s\n"
                % (rid, k, cc.now_iso())) + yaml.safe_dump(rcfg, sort_keys=False)
        cc.atomic_write_text(path, text)
        return path

    def start_attempt(self, rid, spec, wt, cfg_path, cfg_sha, *, kind, resume_from_epoch=None):
        tr = self.rstate(rid)["train"]
        k = len(tr["attempts"])
        run_dir = Path(tr["run_dir"])
        log = run_dir / ("train_a%d.log" % k)
        a = {"k": k, "kind": kind, "status": "LAUNCHING", "config": str(cfg_path), "config_sha256": cfg_sha,
             "resume_from_epoch": resume_from_epoch, "log": str(log)}
        tr["attempts"].append(a)
        self.save()
        run_dir.mkdir(parents=True, exist_ok=True)
        argv = [self.fmt(x, python=self.camp["python"], config=str(cfg_path), stop_epoch=self.stop_epoch)
                for x in self.camp["train_command"]]
        info = self.launch(argv, wt, log)
        a.update(info)
        a["status"] = "RUNNING"
        self.state["current"] = {"step": "%s:train" % rid, "pid": a["pid"], "create_time": a["create_time"],
                                 "log": str(log), "run_dir": str(run_dir), "attempt": k}
        self.event("%s: launched %s attempt %d pid=%s log=%s" % (rid, kind, k, a["pid"], log))
        self.save()

    @staticmethod
    def fmt(tmpl: str, **kw) -> str:
        return str(tmpl).format(**kw)

    def monitor_train(self, rid, spec, a) -> None:
        tr = self.rstate(rid)["train"]
        run_dir = Path(tr["run_dir"])
        self.state["current"] = {"step": "%s:train" % rid, "pid": a.get("pid"), "create_time": a.get("create_time"),
                                 "log": a.get("log"), "run_dir": str(run_dir), "attempt": a.get("k")}
        self.save()
        self.say("%s: monitoring attempt %d pid=%s" % (rid, a["k"], a["pid"]))
        seen = set(tr["epochs"].keys())
        sus_run, sus_alerted = 0, False
        while True:
            alive = self.proc_alive(a)
            self.observe_epochs(rid, spec, a, run_dir, seen)
            if not alive:
                break
            sus = cc.suspended_in_tree(a.get("pid"), a.get("create_time"))
            sus_run = sus_run + 1 if sus else 0
            if sus_run >= 3 and not sus_alerted:
                sus_alerted = True
                path = cc.write_alert(self.state_dir, "TRAINER_SUSPENDED_%s" % rid,
                                      "%s attempt %d: process(es) %s fully suspended for %d polls; the runner "
                                      "does not resume or kill them; check whether the coordinator pause (pause_ctl.ps1: "
                                      "paused_pids.txt) is active" % (rid, a["k"], sus, sus_run))
                self.event("%s: trainer tree suspended %s -> %s" % (rid, sus, path))
            elif not sus:
                sus_alerted = False
            time.sleep(self.poll)
        rc = self.proc_rc(a)
        a["rc"] = rc
        a["ended_at"] = cc.now_iso()
        a["status"] = "EXITED"
        self.children.pop(a.get("pid"), None)
        self.event("%s: attempt %d exited rc=%s" % (rid, a["k"], rc))
        self.save()

    def observe_epochs(self, rid, spec, a, run_dir, seen) -> None:
        text = cc.tail_text(a["log"], max_bytes=1 << 30)
        new = []
        for line in text.splitlines():
            m = gate.EPOCH_RE.match(line.strip())
            if m and m.group(1) not in seen:
                new.append(int(m.group(1)))
                seen.add(m.group(1))
        if not new:
            return
        tr = self.rstate(rid)["train"]
        now = time.time()
        late = len(new) > 1
        if not a.get("manifest_checked"):
            self.check_manifest(rid, spec, a, run_dir)
        for e in sorted(set(new)):
            tr["epochs"][str(e)] = {"observed_at": now, "observed_at_iso": fmt_ts(now), "attempt": a["k"],
                                    "observed_late": late}
        res = gate.check_run(run_dir, spec["arm"], self.camp.get("reference_curves", gate.REFERENCE))
        gdir = self.state_dir / "gates"
        cc.atomic_write_json(gdir / ("%s_gate_latest.json" % rid), res)
        with open(gdir / ("%s_gates.jsonl" % rid), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": cc.now_iso(), "epoch": res["latest_epoch"],
                                 "status": res["latest_status"], "summary": res["summary"]}) + "\n")
        new_statuses = []
        for e in sorted(set(new)):
            er = res["epochs"].get(str(e))
            st = er["status"] if er else res["latest_status"]
            if e == res["latest_epoch"]:
                st = res["latest_status"]
            new_statuses.append(st)
            tr["gates"][str(e)] = {"status": st, "summary": res["summary"] if e == res["latest_epoch"] else None}
            tr["epochs"][str(e)].update({"train_loss": er and er["train_loss"], "val_loss": er and er["val_loss"],
                                         "loop_seconds": er and er.get("loop_seconds")})
        tr["latest_gate"] = {"epoch": res["latest_epoch"], "status": res["latest_status"],
                             "worst": res["worst_status"], "summary": res["summary"]}
        self.say("%s %s" % (rid, res["summary"]))
        status = gate.worst(res["latest_status"], *new_statuses)
        if status == "HOLD":
            path = cc.write_alert(self.state_dir, "HOLD_%s_ep%03d" % (rid, res["latest_epoch"] or 0),
                                  res["summary"] + "\nTraining continues; the coordinator decides.\n"
                                  "Gate JSON: %s" % (gdir / ("%s_gate_latest.json" % rid)))
            self.event("%s: G1 HOLD at epoch %s -> %s" % (rid, res["latest_epoch"], path))
        self.update_projection(rid, spec)
        self.save()
        if status == "STOP":
            a["gate_stop"] = res["summary"]
            self.save()
            self.kill_own_tree(a, "G1 STOP: " + res["summary"])
            a["status"] = "EXITED"
            a["rc"] = self.proc_rc(a)
            a["ended_at"] = cc.now_iso()
            a["terminated_by_runner"] = True
            self.children.pop(a.get("pid"), None)
            self.save()
            self.stop("%s: G1 STOP-class failure: %s" % (rid, res["summary"]))

    def check_manifest(self, rid, spec, a, run_dir: Path) -> None:
        """The trainer's run_manifest.json must describe exactly the launched attempt."""
        man = cc.read_json(run_dir / "run_manifest.json")
        if not isinstance(man, dict):
            self.stop("%s: run_manifest.json missing after the first epoch" % rid)
        want = {
            "git_commit": spec.get("git_commit"), "git_dirty": False,
            "invocation.config_file_sha256": a.get("config_sha256"),
            "invocation.stop_after_epoch": self.stop_epoch,
            "resume_policy": "fork" if a["kind"] == "fork" else "exact",
            "seed": int(spec["seed"]), "amp.amp_target": False,
            "batch.world_size": 1, "batch.batch_size": 64, "batch.accum_steps": 8,
            "batch.effective_batch": 512, "batch.epochs": 100,
        }
        if spec["arm"] == "envelope":
            want["config.mask.curriculum.mirage_overlap_fallback"] = mk.ENVELOPE_OVERLAP_FALLBACK
        if spec["arm"] in MASK_POLICY:
            want["mask_policy"] = MASK_POLICY[spec["arm"]]
        tr = self.rstate(rid)["train"]
        # Constant across fork -> exact resumes (i1): run_uuid and run_contract_sha256.
        for key, slot in (("run_uuid", "trainer_run_uuid"), ("run_contract_sha256", "trainer_contract_sha256")):
            first = next((x.get(slot) for x in tr["attempts"] if x is not a and x.get(slot)), None)
            if a["kind"] == "exact" and first:
                want[key] = first
        bad = {k: json_get(man, k) for k, v in want.items() if json_get(man, k) != v}
        for key in ("run_uuid", "run_contract_sha256"):
            if not man.get(key):
                bad[key] = man.get(key)
                want.setdefault(key, "<non-empty>")
        started = a.get("started_at")
        mpath = run_dir / "run_manifest.json"
        if started and mpath.stat().st_mtime + 2 < float(started):
            bad["run_manifest.json mtime"] = mpath.stat().st_mtime
            want["run_manifest.json mtime"] = ">= attempt launch %s" % started
        a["trainer_run_uuid"] = man.get("run_uuid")
        a["trainer_contract_sha256"] = man.get("run_contract_sha256")
        a["manifest_checked"] = {"ok": not bad, "at": cc.now_iso(), "mismatch": bad}
        self.save()
        if bad:
            if a.get("status") == "RUNNING":
                self.kill_own_tree(a, "run_manifest mismatch")
                a["status"] = "EXITED"
                a["rc"] = self.proc_rc(a)
                a["terminated_by_runner"] = True
            a["gate_stop"] = "run_manifest mismatch %s" % bad
            self.children.pop(a.get("pid"), None)
            self.save()
            self.stop("%s: run_manifest.json does not match the launched attempt: %s (expected %s)"
                      % (rid, bad, {k: want[k] for k in bad}))

    def resolve_exit(self, rid, a, run_dir) -> str:
        rc = a.get("rc")
        sf = self.stop_file(run_dir)
        has = sf.exists()
        if a.get("gate_stop"):
            self.stop("%s: attempt %d was terminated by a G1 STOP: %s" % (rid, a["k"], a["gate_stop"]))
        if has and rc in (0, None):
            return "OK"
        if has:
            self.stop("%s: inconsistent exit: rc=%s but %s exists" % (rid, rc, sf.name))
        if rc == 0:
            self.stop("%s: trainer exited 0 without %s" % (rid, sf.name))
        if rc == EXIT_STOP_NOT_REACHED:
            self.stop("%s: trainer reports training ended before epoch %d (exit 3)" % (rid, self.stop_epoch))
        if rc == EXIT_NONFINITE_LOSS:
            self.stop("%s: trainer aborted on a non-finite epoch loss (exit 4); not resuming automatically"
                      % rid)
        return "CRASH"

    def verify_and_pin(self, rid, spec, run_dir: Path, tag: str) -> None:
        tr = self.rstate(rid)["train"]
        sf = self.stop_file(run_dir)
        stop = cc.read_json(sf)
        want_epoch = self.stop_epoch
        probs = []
        if not isinstance(stop, dict):
            self.stop("%s: unreadable stop file %s" % (rid, sf))
        if stop.get("schema") != STOP_FILE_SCHEMA:
            probs.append("schema %r" % stop.get("schema"))
        if int(stop.get("epoch", -1)) != want_epoch or stop.get("status") != "verified":
            probs.append("epoch/status %r/%r" % (stop.get("epoch"), stop.get("status")))
        if stop.get("git_commit") != spec.get("git_commit"):
            probs.append("git_commit %r != pinned %r" % (stop.get("git_commit"), spec.get("git_commit")))
        if stop.get("git_dirty") is not False:
            probs.append("git_dirty %r" % stop.get("git_dirty"))
        if stop.get("write_tag") != tag:
            probs.append("write_tag %r != %r" % (stop.get("write_tag"), tag))
        if stop.get("output_dir") and norm(stop["output_dir"]) != norm(run_dir):
            probs.append("output_dir %r != %r" % (stop.get("output_dir"), str(run_dir)))
        periodic = [c for c in stop.get("checkpoints", []) if c.get("role") == "periodic"]
        expected_ckpt = run_dir / ("%s-ep%d.pth.tar" % (tag, want_epoch))
        if len(periodic) != 1:
            probs.append("expected exactly one periodic checkpoint record, got %d" % len(periodic))
        else:
            c = periodic[0]
            if norm(c.get("path", "")) != norm(expected_ckpt):
                probs.append("periodic checkpoint %r != %r" % (c.get("path"), str(expected_ckpt)))
            if int(c.get("epoch", -1)) != want_epoch:
                probs.append("periodic record epoch %r" % c.get("epoch"))
        if probs:
            self.stop("%s: stop file %s failed checks: %s" % (rid, sf, "; ".join(probs)))
        c = periodic[0]
        src = expected_ckpt
        if not src.exists():
            self.stop("%s: %s missing" % (rid, src))
        size = src.stat().st_size
        sha = cc.sha256_file(src)
        if sha != c.get("sha256") or size != int(c.get("bytes", -1)):
            self.stop("%s: %s sha/size %s/%d do not match stop file %s/%s"
                      % (rid, src.name, sha, size, c.get("sha256"), c.get("bytes")))
        ep = read_checkpoint_epoch(src, self.camp["python"])
        if ep != want_epoch:
            self.stop("%s: %s deserializes to epoch %d, not %d (never relabelled)" % (rid, src.name, ep, want_epoch))
        man = cc.read_json(run_dir / "run_manifest.json")
        if not isinstance(man, dict):
            self.stop("%s: run_manifest.json missing or unreadable in %s" % (rid, run_dir))
        if man.get("git_commit") != spec.get("git_commit"):
            self.stop("%s: run_manifest git_commit %r != pinned %r" % (rid, man.get("git_commit"), spec.get("git_commit")))
        known = {x.get("trainer_run_uuid") for x in tr["attempts"] if x.get("trainer_run_uuid")}
        ruuid = stop.get("run_uuid")
        if not ruuid or ruuid != man.get("run_uuid") or (known and known != {ruuid}):
            self.stop("%s: trainer run_uuid inconsistent: stop file %r, manifest %r, attempts %s"
                      % (rid, ruuid, man.get("run_uuid"), sorted(known)))
        contracts = {x.get("trainer_contract_sha256") for x in tr["attempts"] if x.get("trainer_contract_sha256")}
        rc_sha = stop.get("run_contract_sha256")
        if not rc_sha or rc_sha != man.get("run_contract_sha256") or (contracts and contracts != {rc_sha}):
            self.stop("%s: run_contract_sha256 inconsistent: stop file %r, manifest %r, attempts %s"
                      % (rid, rc_sha, man.get("run_contract_sha256"), sorted(contracts)))
        if spec["arm"] in MASK_POLICY and stop.get("mask_policy") != MASK_POLICY[spec["arm"]]:
            self.stop("%s: stop file mask_policy %r != %r" % (rid, stop.get("mask_policy"), MASK_POLICY[spec["arm"]]))
        self.rstate(rid)["trainer_run_uuid"] = ruuid
        pinned = run_dir / "pinned" / ("%s-ep%03d.pth.tar" % (tag, want_epoch))
        if pinned.exists():
            psha = cc.sha256_file(pinned)
            if psha != sha:
                self.stop("%s: existing pinned file %s has sha %s != %s" % (rid, pinned, psha, sha))
        else:
            pinned.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(pinned) + ".tmp")
            if tmp.exists():
                tmp.unlink()
            with open(src, "rb") as fi, open(tmp, "xb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 22)
                fo.flush()
                os.fsync(fo.fileno())
            os.replace(tmp, pinned)
            psha = cc.sha256_file(pinned)
            if psha != sha:
                self.stop("%s: pinned copy hash %s != source %s" % (rid, psha, sha))
        tr["stop_file"] = {"path": str(sf), "sha256": cc.sha256_file(sf)}
        tr["pinned"] = {"path": str(pinned), "sha256": sha, "bytes": size, "epoch": ep,
                        "source": str(src), "pinned_at": cc.now_iso()}
        self.event("%s: pinned epoch %d checkpoint %s sha256 %s" % (rid, ep, pinned, sha))
        self.save()

    # -- probes -----------------------------------------------------------
    def required_repo_files(self, target: dict) -> list[str]:
        """Repo-relative files the probe needs in the pinned worktree (campaign + per-target)."""
        return list(self.camp["probe"].get("requires_repo_files") or []) + \
            list(target.get("requires_repo_files") or [])

    def probe_spec_ok(self, fatal=True) -> list[str]:
        ps = self.camp["probe"]
        probs = []
        if not ps.get("spec_confirmed"):
            probs.append("campaign.json probe.spec_confirmed is false (probe CLI not confirmed)")
        for key in ("command", "outputs", "encoder_sha_field", "val_auc_field"):
            if not ps.get(key):
                probs.append("probe.%s missing" % key)
        if probs and fatal:
            self.stop("; ".join(probs))
        return probs

    def do_probe(self, rid: str, spec: dict) -> None:
        rs = self.rstate(rid)
        pr = rs["probe"]
        if pr["status"] == "DONE":
            if "g2" not in rs and pr.get("result"):
                self.run_g2(rid, spec, pr["result"], self.ensure_worktree(spec.get("git_commit")))
            return
        if not spec.get("probe", {}).get("enabled", True):
            pr["status"] = "DISABLED"
            self.event("%s: probe disabled in campaign.json (skipped, not run)" % rid)
            self.save()
            return
        tr = rs["train"]
        if tr.get("status") != "DONE" or not tr.get("pinned"):
            self.stop("%s: probe requested but training is not DONE/pinned" % rid)
        self.check_frozen(rid, spec)
        self.probe_spec_ok()
        wt = self.ensure_worktree(spec.get("git_commit"))
        pin = tr["pinned"]
        target = {"name": "%s_%s_s%s" % (rid, spec["arm"], spec["seed"]), "checkpoint": pin["path"],
                  "sha256": pin["sha256"], "arm": spec["arm"], "seed": spec["seed"], "mode": "checkpoint",
                  "role": "new", "run_uuid": rs.get("trainer_run_uuid") or rs.get("run_uuid", ""),
                  "run_manifest": str(Path(tr["run_dir"]) / "run_manifest.json"),
                  # i4 04:23: the stop file as provenance also proves the probed sha is a verified checkpoint
                  "run_provenance": tr.get("stop_file", {}).get("path") or str(self.stop_file(Path(tr["run_dir"])))}
        target["expect"] = {"identity.run_uuid": target["run_uuid"], "identity.train_seed": int(spec["seed"]),
                            "identity.role": "new", "identity.run_provenance_checkpoint_role": "periodic",
                            "identity.epoch": self.stop_epoch}
        res = self.run_probe_target(rid, pr, target, wt)
        pr["status"] = "DONE"
        pr["result"] = res
        self.state["current"] = None
        self.event("%s: probe DONE val_auc=%s dir=%s" % (rid, res.get("val_auc"), res.get("out_dir")))
        self.save()
        self.run_g2(rid, spec, res, wt)

    def run_g2(self, rid, spec, res, wt) -> None:
        """G2 (validation only): new probe vs the same arm's re-probed original anchor.
        A flag is recorded and alerted for the coordinator (decision tree); never auto-stops.
        Integrity errors of the gate tool stop the campaign."""
        g2 = self.camp.get("g2") or {}
        rs = self.rstate(rid)
        if not g2.get("enabled"):
            rs["g2"] = {"status": "DISABLED"}
            self.event("%s: G2 disabled in campaign.json" % rid)
            self.save()
            return
        anc_id = (self.camp.get("anchors") or {}).get("id")
        items = ((self.state["runs"].get(anc_id) or {}).get("probe") or {}).get("items") or {}
        anchor_dir = None
        for item in (self.camp.get("anchors") or {}).get("items", []):
            ist = items.get(item["name"]) or {}
            if item.get("arm") == spec["arm"] and ist.get("status") == "DONE":
                anchor_dir = ist["result"]["out_dir"]
        if not anchor_dir:
            rs["g2"] = {"status": "NO_ANCHOR", "at": cc.now_iso()}
            self.event("%s: G2 not computed: no completed %s anchor" % (rid, spec["arm"]))
            self.save()
            return
        gdir = self.state_dir / "gates"
        gdir.mkdir(parents=True, exist_ok=True)
        out_json = gdir / ("%s_g2.json" % rid)
        argv = [self.fmt(x, python=self.camp["python"], run_probe_dir=res["out_dir"],
                         anchor_probe_dir=anchor_dir, out_json=str(out_json)) for x in g2["command"]]
        env = dict(os.environ, PYTHONPATH=str(wt), MPLBACKEND="Agg")
        env.update({str(k): str(v) for k, v in (self.camp.get("child_env") or {}).items()})
        try:
            r = subprocess.run(argv, cwd=str(wt), capture_output=True, text=True, env=env,
                               timeout=float(g2.get("timeout_seconds", 3600)), **NO_WINDOW)
        except subprocess.TimeoutExpired:
            self.stop("%s: G2 gate timed out" % rid)
        (gdir / ("%s_g2.log" % rid)).write_text((r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or ""),
                                               encoding="utf-8")
        if r.returncode != 0:
            self.stop("%s: G2 gate tool failed rc=%s: %s" % (rid, r.returncode, (r.stderr or r.stdout)[-800:]))
        payload = cc.read_json(out_json)
        try:
            gates = payload["gates"]
            flag = any(bool(g["primary_flag"]) for g in gates)
        except (TypeError, KeyError):
            self.stop("%s: G2 output %s unreadable" % (rid, out_json))
        rs["g2"] = {"status": "FLAG" if flag else "PASS", "anchor_dir": anchor_dir, "json": str(out_json),
                    "table": (r.stdout or "")[-2000:], "at": cc.now_iso()}
        if flag:
            path = cc.write_alert(self.state_dir, "G2_FLAG_%s" % rid,
                                  "G2 validation AUC differs from the %s anchor by more than the threshold; apply "
                                  "the pre-declared decision tree (PLAN 3.4). Campaign continues.\n%s"
                                  % (spec["arm"], r.stdout))
            self.event("%s: G2 FLAG -> %s" % (rid, path))
        else:
            self.event("%s: G2 PASS vs %s" % (rid, anchor_dir))
        self.save()

    def do_anchors(self, rid: str, spec: dict) -> None:
        rs = self.rstate(rid)
        pr = rs["probe"]
        if pr["status"] == "DONE":
            return
        self.check_frozen(rid, spec)
        self.probe_spec_ok()
        wt = self.ensure_worktree(spec.get("git_commit"))
        pr.setdefault("items", {})
        for item in spec["items"]:
            name = item["name"]
            ist = pr["items"].setdefault(name, {"status": "PENDING", "attempts": []})
            if ist["status"] == "DONE":
                continue
            if not item.get("enabled", True):
                ist["status"] = "DISABLED"
                self.event("ANCHORS: %s disabled (skipped, not run)" % name)
                self.save()
                continue
            target = dict(item)
            target.setdefault("mode", "checkpoint")
            target.setdefault("role", "anchor")
            target["expect"] = dict({"identity.role": "anchor", "identity.epoch": self.stop_epoch},
                                    **({"identity.run_uuid": target["run_uuid"]} if target.get("run_uuid") else {}))
            if not target.get("command") and self.camp["probe"].get("anchor_command"):
                target["command"] = self.camp["probe"]["anchor_command"]
            res = self.run_probe_target(rid, ist, target, wt)
            ist["status"] = "DONE"
            ist["result"] = res
            self.event("ANCHORS: %s DONE val_auc=%s" % (name, res.get("val_auc")))
            self.save()
        pr["status"] = "DONE"
        self.state["current"] = None
        self.save()

    def run_probe_target(self, rid, pst: dict, target: dict, wt: Path) -> dict:
        ps = self.camp["probe"]
        mode = target.get("mode", "checkpoint")
        if mode == "checkpoint":
            ck = target.get("checkpoint")
            if not ck or not os.path.exists(ck):
                self.stop("%s: probe checkpoint %r missing" % (rid, ck))
            want = target.get("sha256")
            if not want:
                self.stop("%s: probe target %s has no expected sha256" % (rid, target.get("name")))
            got = cc.sha256_file(ck)
            if got != want:
                self.stop("%s: probe checkpoint %s sha256 %s != expected %s" % (rid, ck, got, want))
            ident = want
        elif mode == "legacy_cache":
            ident = target.get("expected_identity")
            if not ident or not target.get("command"):
                self.stop("%s: legacy-cache anchor %s needs 'command' and 'expected_identity' (spec incomplete)"
                          % (rid, target.get("name")))
        else:
            self.stop("%s: unknown probe mode %r" % (rid, mode))
        attempts = pst.setdefault("attempts", [])
        last = attempts[-1] if attempts else None
        if last and last["status"] == "DONE":
            # The attempt was validated but its parent step was not finalized (runner death):
            # re-validate the same output directory and finalize without relaunching.
            self.say("%s: re-validating completed probe attempt %d (%s)" % (rid, last["k"], last.get("out_dir")))
            try:
                return self.validate_probe(rid, last, target, ident)
            except CampaignStop:
                last["status"] = "FAILED"
                self.save()
                raise
        if last and last["status"] == "RUNNING":
            self.say("%s: re-attaching to probe pid=%s" % (rid, last.get("pid")))
            a = last
            self.state["current"] = {"step": "%s:probe" % rid, "pid": a.get("pid"),
                                     "create_time": a.get("create_time"), "log": a.get("log"),
                                     "target": target["name"]}
            self.save()
        else:
            if last and last["status"] == "LAUNCHING":
                self.stop("%s: probe attempt %d state unknown (runner died while launching)" % (rid, last["k"]))
            if last and last["status"] not in ("FAILED",):
                self.stop("%s: probe attempt %d in status %s" % (rid, last["k"], last["status"]))
            k = len(attempts)
            out_dir = Path(self.camp["probe"].get("out_root", self.run_root)) / (
                "cr_probe_%s_ep%03d_%s_a%d" % (target["name"], self.stop_epoch, str(ident)[:12], k))
            if out_dir.exists():
                self.stop("%s: probe output dir %s already exists; refusing to reuse results" % (rid, out_dir))
            log = self.state_dir / "logs" / (out_dir.name + ".log")
            a = {"k": k, "status": "LAUNCHING", "out_dir": str(out_dir), "log": str(log), "target": target}
            attempts.append(a)
            self.save()
            self.preflight("%s probe %s" % (rid, target["name"]))
            missing = [f for f in self.required_repo_files(target) if not (wt / f).exists()]
            if missing:
                self.stop("%s: pinned commit %s lacks files the probe needs: %s"
                          % (rid, str(wt.name)[:12], missing))
            cmd = target.get("command") or ps["command"]
            kw = dict(python=self.camp["python"], checkpoint=target.get("checkpoint", ""),
                      out_dir=str(out_dir), probe_config=ps.get("config", ""),
                      head_seeds=ps.get("head_seeds", ""), name=target["name"],
                      cache_dir=target.get("cache_dir", ""), worktree=str(wt),
                      arm=target.get("arm", ""), seed=target.get("seed", ""),
                      role=target.get("role", "new"), run_uuid=target.get("run_uuid", ""),
                      run_manifest=target.get("run_manifest", ""),
                      run_provenance=target.get("run_provenance", ""),
                      expected_sha256=target.get("sha256") or target.get("expected_identity") or "")
            argv = [self.fmt(x, **kw) for x in cmd]
            a.update(self.launch(argv, wt, log))
            a["status"] = "RUNNING"
            self.state["current"] = {"step": "%s:probe" % rid, "pid": a["pid"], "create_time": a["create_time"],
                                     "log": str(log), "target": target["name"]}
            self.event("%s: probe %s launched pid=%s out=%s" % (rid, target["name"], a["pid"], out_dir))
            self.save()
        while self.proc_alive(a):
            time.sleep(self.poll)
        a["rc"] = self.proc_rc(a)
        a["ended_at"] = cc.now_iso()
        self.children.pop(a.get("pid"), None)
        try:
            res = self.validate_probe(rid, a, target, ident)
        except CampaignStop:
            a["status"] = "FAILED"
            self.save()
            raise
        a["status"] = "DONE"
        self.save()
        return res

    def validate_probe(self, rid, a, target, ident) -> dict:
        ps = self.camp["probe"]
        rc = a.get("rc")
        out_dir = Path(a["out_dir"])
        if rc not in (0, None):
            self.stop("%s: probe %s exited rc=%s (log %s)" % (rid, target["name"], rc, a["log"]))
        started = float(a.get("started_at", 0))
        files = {}
        for role, rel in ps["outputs"].items():
            p = out_dir / rel
            if not p.exists():
                self.stop("%s: probe output %s (%s) missing" % (rid, p, role))
            if p.stat().st_mtime + 2 < started:
                self.stop("%s: probe output %s predates the launch (stale)" % (rid, p))
            files[role] = p
        agg_role = ps.get("aggregate_role", "aggregate")
        agg = cc.read_json(files[agg_role])
        if not isinstance(agg, dict):
            self.stop("%s: unreadable aggregate %s" % (rid, files[agg_role]))
        for key, want in (ps.get("required_values") or {}).items():
            if json_get(agg, key) != want:
                self.stop("%s: probe %s field %s = %r, required %r" % (rid, target["name"], key,
                                                                      json_get(agg, key), want))
        enc = json_get(agg, ps["encoder_sha_field"])
        id_field = target.get("identity_field")
        if target.get("mode") == "legacy_cache" and id_field:
            enc = json_get(agg, id_field)
        if enc != ident:
            self.stop("%s: probe identity %r != expected %r" % (rid, enc, ident))
        arm_field = ps.get("identity_arm_field")
        if arm_field and target.get("arm") and json_get(agg, arm_field) != target["arm"]:
            self.stop("%s: probe records arm %r, expected %r" % (rid, json_get(agg, arm_field), target["arm"]))
        for key, want in (target.get("expect") or {}).items():
            if json_get(agg, key) != want:
                self.stop("%s: probe records %s = %r, expected %r" % (rid, key, json_get(agg, key), want))
        prov_uuid = json_get(agg, "identity.run_provenance_run_uuid")
        if target.get("role") == "new" and prov_uuid is not None and prov_uuid != target.get("run_uuid"):
            self.stop("%s: probe provenance run_uuid %r != trainer run_uuid %r" % (rid, prov_uuid, target.get("run_uuid")))
        rx = ps.get("forbid_numeric_key_regex")
        if rx:
            leaks = find_numeric_keys(agg, re.compile(rx))
            if leaks:
                self.stop("%s: sealed-test violation, numeric test metrics in aggregate: %s" % (rid, leaks[:5]))
        if ps.get("cache_splits"):
            # fresh probes extract cold/warm; only the RANDOM legacy anchor may use verified legacy caches
            allowed = set(target.get("cache_states") or ps.get("fresh_cache_states") or [])
            for split in ps["cache_splits"]:
                state = json_get(agg, "cache_provenance.%s.primary.state" % split)
                if state not in allowed:
                    self.stop("%s: cache_provenance.%s.primary.state %r not in %s"
                              % (rid, split, state, sorted(allowed)))
        if ps.get("check_legacy_integrity"):
            got, want = agg.get("legacy_cache_integrity"), target.get("legacy_cache_integrity")
            if got != want:
                self.stop("%s: legacy_cache_integrity %r, expected %r" % (rid, got, want))
        seal_role = ps.get("seal_role", "sealed_manifest")
        seal = None
        if seal_role in files:
            seal = self.validate_seal(rid, out_dir, files[seal_role], agg, ident, target)
        elif ps.get("require_seal", True):
            self.stop("%s: probe outputs do not declare a %r role; cannot verify the test seal" % (rid, seal_role))
        val = json_get(agg, ps["val_auc_field"])
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = float("nan")
        if not math.isfinite(val):
            self.stop("%s: probe val AUC %r not finite" % (rid, json_get(agg, ps["val_auc_field"])))
        if target.get("mode", "checkpoint") == "checkpoint":
            after = cc.sha256_file(target["checkpoint"])
            if after != ident:
                self.stop("%s: checkpoint %s changed during the probe" % (rid, target["checkpoint"]))
        return {"out_dir": str(out_dir), "val_auc": val, "identity": ident,
                "files": {k: {"path": str(v), "sha256": cc.sha256_file(v)} for k, v in files.items()},
                "seal": seal, "validated_at": cc.now_iso()}

    def validate_seal(self, rid, out_dir: Path, seal_path: Path, agg: dict, ident, target) -> dict:
        """Verify the sealed test predictions by bytes only (no test metric is computed):
        receipt hash, schema, metrics_computed False, identity == results identity/target,
        and every listed prediction/sidecar file hash, bound 1:1 to the aggregate's
        (variant, head_seed) records."""
        def bad(msg):
            self.stop("%s: invalid test seal %s: %s" % (rid, seal_path, msg))
        receipt = Path(str(seal_path) + ".sha256")
        if not receipt.exists():
            bad("receipt %s missing" % receipt.name)
        try:
            recorded = receipt.read_text(encoding="utf-8").split()[0].lower()
        except (OSError, IndexError):
            bad("receipt unreadable")
        seal_sha = cc.sha256_file(seal_path)
        if seal_sha != recorded:
            bad("sha256 %s != receipt %s" % (seal_sha, recorded))
        try:
            seal = json.loads(Path(seal_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            bad("unparseable (%s)" % e)
        if not isinstance(seal, dict):
            bad("not a JSON object")
        if seal.get("schema") != SEAL_SCHEMA:
            bad("schema %r" % seal.get("schema"))
        if seal.get("metrics_computed") is not False:
            bad("metrics_computed %r" % seal.get("metrics_computed"))
        sid = seal.get("identity")
        if not isinstance(sid, dict) or sid != agg.get("identity"):
            bad("identity differs from results.json identity")
        if sid.get("checkpoint_sha256") != ident:
            bad("identity checkpoint_sha256 %r != %r" % (sid.get("checkpoint_sha256"), ident))
        if target.get("run_uuid") and sid.get("run_uuid") != target["run_uuid"]:
            bad("identity run_uuid %r != %r" % (sid.get("run_uuid"), target["run_uuid"]))
        lab = seal.get("test_label_identity") or {}
        if not isinstance(lab.get("n"), int) or lab["n"] <= 0 or not lab.get("labels_int8_sha256"):
            bad("test_label_identity incomplete")
        entries = seal.get("files")
        if not isinstance(entries, list) or not entries:
            bad("no sealed prediction files listed")
        sealed, sealed_paths = {}, {}
        root = Path(out_dir).resolve()
        for e in entries:
            try:
                key = (str(e["variant"]), int(e["head_seed"]))
                rels = [(e["path"], e["sha256"])]
                if e.get("sidecar_path") or e.get("sidecar_sha256"):
                    rels.append((e["sidecar_path"], e["sidecar_sha256"]))
            except (KeyError, TypeError, ValueError):
                bad("malformed file entry %r" % (e,))
            if key in sealed:
                bad("duplicate entry %s" % (key,))
            for rel, want in rels:
                p = (Path(out_dir) / rel).resolve()
                if root not in p.parents:
                    bad("file %s outside the probe directory" % rel)
                if not p.exists():
                    bad("listed file %s missing" % rel)
                if cc.sha256_file(p) != want:
                    bad("listed file %s hash mismatch" % rel)
            sealed[key] = e["sha256"]
            sealed_paths[key] = os.path.normcase(os.path.normpath(e["path"]))
        recorded_pairs, recorded_paths = {}, {}
        try:
            for variant, v in (agg.get("variants") or {}).items():
                for r in (v or {}).get("per_seed") or []:
                    recorded_pairs[(str(variant), int(r["head_seed"]))] = r.get("test_predictions_sha256")
                    recorded_paths[(str(variant), int(r["head_seed"]))] = r.get("test_predictions")
        except (KeyError, TypeError, ValueError, AttributeError):
            bad("results.json variants/per_seed malformed")
        if set(recorded_pairs) != set(sealed):
            bad("sealed (variant, head_seed) set %s != results.json %s" % (sorted(sealed), sorted(recorded_pairs)))
        for key, sha in recorded_pairs.items():
            if sha is not None and sha != sealed[key]:
                bad("results.json test_predictions_sha256 for %s differs from the seal" % (key,))
            rp = recorded_paths.get(key)
            if rp is not None and os.path.normcase(os.path.normpath(rp)) != sealed_paths[key]:
                bad("results.json test_predictions path for %s differs from the seal" % (key,))
        return {"sha256": seal_sha, "n_files": len(entries), "pairs": len(sealed)}

    # -- projection -------------------------------------------------------
    def epoch_minutes(self, arm: str) -> float:
        meas = []
        for rid, rs in self.state["runs"].items():
            sp = rs.get("spec") or {}
            if sp.get("arm") == arm:
                meas += epoch_durations(rs["train"])
        if len(meas) >= 2:
            return statistics.median(meas) / 60.0
        return float(self.camp.get("epoch_minutes_default", {}).get(arm, 80.0))

    def update_projection(self, current_rid=None, current_spec=None) -> dict:
        now = time.time()
        probe_h = float(self.camp.get("probe_hours_default", 1.25))
        t = now
        rows = []
        seq = self.camp["sequence"]
        started = False
        cur_end = None
        for step in seq:
            rid, _, kind = step.partition(":")
            sp = run_spec(self.camp, rid)
            rs = self.state["runs"].get(rid, {})
            if not sp.get("enabled", True):
                continue
            if kind == "train":
                trs = rs.get("train", {})
                if trs.get("status") == "DONE":
                    continue
                per = self.epoch_minutes(sp["arm"]) * 60.0
                durs = epoch_durations(trs) if trs else []
                if durs:
                    per = statistics.median(durs[-5:])
                last_e = max([int(e) for e in trs.get("epochs", {})] or [self.fork_start])
                last_t = max([v["observed_at"] for v in trs.get("epochs", {}).values()] or [t])
                remaining = self.stop_epoch - last_e
                base = max(t, last_t) if started else (last_t if trs.get("epochs") else t)
                end = base + remaining * per
                if rid == current_rid:
                    cur_end = end
                rows.append({"step": step, "epochs_remaining": remaining, "sec_per_epoch": round(per, 1),
                             "end": fmt_ts(end)})
                t = end
            else:
                prs = rs.get("probe", {})
                if prs.get("status") in ("DONE", "DISABLED"):
                    continue
                if rid == (self.camp.get("anchors") or {}).get("id"):
                    n = len([i for i in sp.get("items", []) if i.get("enabled", True)])
                    dur = n * float(self.camp.get("anchor_hours_each", 1.0)) * 3600
                else:
                    if not sp.get("probe", {}).get("enabled", True):
                        continue
                    dur = probe_h * 3600
                t = t + dur
                rows.append({"step": step, "hours": round(dur / 3600, 2), "end": fmt_ts(t)})
            started = True
        proj = {"computed_at": cc.now_iso(), "current_run": current_rid,
                "current_run_training_end": fmt_ts(cur_end), "campaign_end": fmt_ts(t), "steps": rows,
                "freeze": self.camp.get("freeze")}
        self.state["projection"] = proj
        if self.readonly:
            return proj
        try:
            cc.atomic_write_json(self.state_dir / "projection.json", proj)
        except OSError:
            pass
        if current_rid:
            self.say("projection: %s training ends %s; campaign ends %s" % (current_rid, proj["current_run_training_end"],
                                                                            proj["campaign_end"]))
        return proj

    # -- main loop --------------------------------------------------------
    def steps(self, only=None):
        for step in self.camp["sequence"]:
            rid, _, kind = step.partition(":")
            if only and rid != only:
                continue
            yield step, rid, kind

    def run(self, only=None) -> int:
        st = self.load_state()
        if st.get("status") == "STOPPED":
            self.say("campaign is STOPPED: %s\nacknowledge with --clear-stop \"<note>\" after checking"
                     % st.get("stop_reason"))
            return EXIT_STOPPED
        st["status"] = "RUNNING"
        st["runner"] = cc.own_identity()
        st["campaign_json"] = {"path": self.camp_path, "sha256": cc.sha256_file(self.camp_path)}
        self.save()
        try:
            for step, rid, kind in self.steps(only):
                spec = run_spec(self.camp, rid)
                rs = self.rstate(rid)
                if not spec.get("enabled", True):
                    if rs["train"]["status"] == "PENDING":
                        rs["train"]["status"] = "DISABLED"
                    if rs["probe"]["status"] == "PENDING":
                        rs["probe"]["status"] = "DISABLED"
                    self.event("%s: %s disabled in campaign.json (skipped, not run)" % (step, rid))
                    self.save()
                    continue
                self.say("=== step %s" % step)
                if kind == "train":
                    self.do_train(rid, spec)
                elif (self.camp.get("anchors") or {}).get("id") == rid:
                    self.do_anchors(rid, spec)
                else:
                    self.do_probe(rid, spec)
                self.update_projection()
                self.save()
        except CampaignStop as e:
            self.state["status"] = "STOPPED"
            self.state["stop_reason"] = str(e)
            self.state["stopped_at"] = cc.now_iso()
            self.save()
            path = cc.write_alert(self.state_dir, "STOP", str(e))
            self.say("CAMPAIGN STOPPED: %s (alert %s)" % (e, path))
            return EXIT_STOPPED
        except KeyboardInterrupt:
            self.state["status"] = "INTERRUPTED"
            self.save()
            self.say("runner interrupted; child processes keep running and will be re-attached")
            return 130
        except Exception as e:  # noqa: BLE001  (unexpected bug/IO error: fail closed, keep children)
            import traceback
            tb = traceback.format_exc()
            self.state["status"] = "STOPPED"
            self.state["stop_reason"] = "runner exception: %r" % (e,)
            self.state["stopped_at"] = cc.now_iso()
            try:
                self.save()
            except Exception:  # noqa: BLE001
                pass
            path = cc.write_alert(self.state_dir, "STOP", "runner exception (children keep running and "
                                  "will be re-attached after --clear-stop)\n" + tb)
            self.say("CAMPAIGN STOPPED by runner exception %r (alert %s)" % (e, path))
            return EXIT_STOPPED
        self.state["status"] = "DONE" if not only else "IDLE"
        self.state["current"] = None
        self.save()
        self.say("campaign %s" % ("complete" if not only else "step(s) for %s complete" % only))
        return 0

    def clear_stop(self, note: str) -> int:
        st = self.load_state()
        if st.get("status") != "STOPPED":
            self.say("nothing to clear (status %s)" % st.get("status"))
            return 0
        st.setdefault("acks", []).append({"ts": cc.now_iso(), "note": note, "cleared": st.get("stop_reason")})
        for rid, rs in st["runs"].items():
            tr = rs.get("train", {})
            tr["restarts_since_ack"] = 0
            for a in tr.get("attempts", []):
                if a.get("status") == "LAUNCHING":
                    a["status"] = "EXITED"
                    a["rc"] = None
                    a["note"] = "launch state unknown; cleared by coordinator"
                if a.get("gate_stop"):
                    a["gate_stop_acknowledged"] = a.pop("gate_stop")
            pr = rs.get("probe", {})
            groups = [pr.get("attempts", [])] + [it.get("attempts", []) for it in (pr.get("items") or {}).values()]
            for attempts in groups:
                for a in attempts:
                    if a.get("status") == "LAUNCHING":
                        a["status"] = "FAILED"
                        a["note"] = "launch state unknown; cleared by coordinator"
        st["status"] = "READY"
        st["stop_reason"] = None
        self.save()
        self.say("stop cleared: %s" % note)
        return 0

    # -- dry run / status -------------------------------------------------
    def dry_run(self, only=None) -> int:
        self.readonly = True
        st = cc.read_json(self.state_path) or {"runs": {}, "status": "NEW"}
        self.state = st
        self.say("DRY RUN campaign %s (state %s: %s)" % (self.camp["campaign"], self.state_path, st.get("status")))
        problems = []
        for step, rid, kind in self.steps(only):
            spec = run_spec(self.camp, rid)
            rs = st.get("runs", {}).get(rid, {})
            sub = rs.get(kind if kind == "train" else "probe", {}) if rs else {}
            commit = spec.get("git_commit")
            line = "%-14s enabled=%-5s commit=%s status=%s" % (step, spec.get("enabled", True),
                                                               (commit or "UNSET")[:12], sub.get("status", "PENDING"))
            self.out(line)
            if not spec.get("enabled", True):
                continue
            if kind == "probe":
                problems += ["%s: %s" % (rid, p) for p in self.probe_spec_ok(fatal=False)]
            if not commit or not SHA_RE.match(str(commit)):
                problems.append("%s: git_commit not set" % rid)
                continue
            wt = self.worktree_path(commit)
            self.out("    worktree %s (%s)" % (wt, "exists" if wt.exists() else "will be created"))
            if kind == "train":
                try:
                    cfg_text = self.git("show", "%s:%s" % (commit, spec["config"].replace("\\", "/")))
                    import yaml
                    cfg = yaml.safe_load(cfg_text)
                    mk.validate_run_config(cfg, spec["arm"], int(spec["seed"]), run_root=self.run_root,
                                           ancestor=self.camp["ancestor"]["path"])
                    run_dir = Path(cfg["logging"]["folder"])
                    self.out("    config %s @%s OK; run dir %s (%s)" % (
                        spec["config"], commit[:12], run_dir,
                        "non-empty!" if run_dir.exists() and any(run_dir.iterdir()) else "empty/new"))
                    argv = [self.fmt(x, python=self.camp["python"], config=str(wt / spec["config"]),
                                     stop_epoch=self.stop_epoch) for x in self.camp["train_command"]]
                    self.out("    cmd: %s" % " ".join(argv))
                    if run_dir.exists() and any(run_dir.iterdir()) and not sub.get("attempts"):
                        problems.append("%s: run dir %s not empty" % (rid, run_dir))
                except Exception as e:  # noqa: BLE001
                    problems.append("%s: config problem: %s" % (rid, e))
            else:
                targets = [it for it in spec.get("items", []) if it.get("enabled", True)] \
                    if rid == (self.camp.get("anchors") or {}).get("id") else [{}]
                for t in targets:
                    for f in self.required_repo_files(t):
                        try:
                            self.git("cat-file", "-e", "%s:%s" % (commit, f.replace("\\", "/")))
                        except RuntimeError:
                            problems.append("%s: pinned commit %s lacks %s" % (rid, commit[:12], f))
                if rid == (self.camp.get("anchors") or {}).get("id"):
                    for it in spec.get("items", []):
                        self.out("    anchor %s mode=%s enabled=%s" % (it["name"], it.get("mode", "checkpoint"),
                                                                     it.get("enabled", True)))
                        if it.get("enabled", True) and it.get("mode") == "legacy_cache" and not (
                                it.get("command") and it.get("expected_identity")):
                            problems.append("ANCHORS %s: legacy-cache spec incomplete" % it["name"])
                else:
                    self.out("    probe cmd: %s" % " ".join(self.camp["probe"].get("command", [])))
        anc = self.verify_ancestor(fatal=False)
        self.out("ancestor: %s" % anc)
        if not anc.get("ok"):
            problems.append("ancestor check failed")
        rep = self.preflight("dry-run", fatal=False, wait_gpu=False)
        problems += rep["problems"]
        proj = self.update_projection()
        self.out("projected campaign end (from now): %s" % proj["campaign_end"])
        problems = list(dict.fromkeys(problems))
        for p in problems:
            self.out("PROBLEM: %s" % p)
        self.out("dry run: %d problem(s)" % len(problems))
        return 1 if problems else 0

    def status(self) -> int:
        st = cc.read_json(self.state_path)
        if st is None:
            self.out("no state yet (%s)" % self.state_path)
            return 0
        lock = self.lock.owner() or {}
        live = self.lock.is_locked_by_other()
        self.out("campaign %s status=%s updated=%s runner_lock=%s" % (
            st.get("campaign"), st.get("status"), st.get("updated_at"),
            ("held (recorded pid %s)" % lock.get("pid")) if live else "free"))
        if st.get("stop_reason"):
            self.out("STOP REASON: %s" % st["stop_reason"])
        cur = st.get("current") or {}
        if cur:
            alive = cc.same_process_alive(cur.get("pid"), cur.get("create_time"))
            self.out("current: %s pid=%s alive=%s log=%s" % (cur.get("step"), cur.get("pid"), alive, cur.get("log")))
        for step in self.camp["sequence"]:
            rid, _, kind = step.partition(":")
            rs = st.get("runs", {}).get(rid, {})
            if kind == "train":
                tr = rs.get("train", {})
                eps = sorted(int(e) for e in tr.get("epochs", {}))
                lg = tr.get("latest_gate", {})
                self.out("%-14s %-9s attempts=%d restarts=%s last_epoch=%s gate=%s worst=%s pinned=%s" % (
                    step, tr.get("status", "PENDING"), len(tr.get("attempts", [])), tr.get("restarts_since_ack", 0),
                    eps[-1] if eps else "-", lg.get("status", "-"), lg.get("worst", "-"),
                    (tr.get("pinned") or {}).get("sha256", "-")[:12]))
            else:
                pr = rs.get("probe", {})
                res = pr.get("result") or {}
                self.out("%-14s %-9s val_auc=%s" % (step, pr.get("status", "PENDING"), res.get("val_auc", "-")))
        proj = st.get("projection") or {}
        if proj:
            self.out("projection (%s): current run training end %s, campaign end %s" % (
                proj.get("computed_at"), proj.get("current_run_training_end"), proj.get("campaign_end")))
        alerts = sorted(self.state_dir.glob("ALERT_*.txt"))
        if alerts:
            self.out("alerts (%d): %s" % (len(alerts), ", ".join(a.name for a in alerts[-8:])))
        return 0

    # -- commit pinning helper -------------------------------------------
    def set_commit(self, commit: str, runs=None) -> int:
        commit = self.git("rev-parse", "--verify", "%s^{commit}" % commit).strip()
        camp = json.loads(Path(self.camp_path).read_text(encoding="utf-8"))
        st = cc.read_json(self.state_path) or {"runs": {}}
        targets = []
        all_specs = list(camp["runs"]) + ([camp["anchors"]] if camp.get("anchors") else [])
        for sp in all_specs:
            if runs and sp["id"] not in runs:
                continue
            if not runs and not sp.get("enabled", True):
                continue
            started = bool(st.get("runs", {}).get(sp["id"], {}).get("spec"))
            if started and sp.get("git_commit") != commit:
                raise SystemExit("refusing to change the commit of started run %s" % sp["id"])
            if sp.get("config"):
                try:
                    self.git("cat-file", "-e", "%s:%s" % (commit, sp["config"].replace("\\", "/")))
                except RuntimeError:
                    raise SystemExit("commit %s does not contain %s (%s)" % (commit[:12], sp["config"], sp["id"]))
            sp["git_commit"] = commit
            targets.append(sp["id"])
        cc.atomic_write_text(self.camp_path, json.dumps(camp, indent=2) + "\n")
        self.out("pinned %s to %s" % (", ".join(targets), commit))
        return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_checkpoint_epoch(path, python=None, timeout=900) -> int:
    """``epoch`` stored in a checkpoint, read in a short-lived subprocess so the
    long-running runner never holds torch (~0.4 GB) while a trainer needs the RAM."""
    code = ("import gc, sys, torch\n"
            "p = sys.argv[1]\n"
            "try:\n"
            "    ck = torch.load(p, map_location='cpu', weights_only=False, mmap=True)\n"
            "except Exception:\n"
            "    ck = torch.load(p, map_location='cpu', weights_only=False)\n"
            "print('EPOCH=%d' % int(ck['epoch']) if isinstance(ck, dict) and 'epoch' in ck else 'EPOCH=NONE')\n")
    try:
        r = subprocess.run([python or sys.executable, "-c", code, str(path)], capture_output=True, text=True,
                           timeout=timeout, **NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise CampaignStop("reading the epoch of %s timed out after %d s" % (path, timeout))
    m = re.search(r"EPOCH=(\d+|NONE)", r.stdout or "")
    if r.returncode != 0 or not m or m.group(1) == "NONE":
        raise CampaignStop("cannot read the epoch of %s (rc=%s): %s" % (path, r.returncode,
                                                                       (r.stderr or r.stdout)[-500:]))
    return int(m.group(1))


def epoch_durations(tr: dict) -> list[float]:
    """Wall seconds per epoch from observed epoch-line times within one attempt."""
    eps = tr.get("epochs", {})
    att = {a["k"]: a for a in tr.get("attempts", [])}
    out = []
    for e in sorted(eps, key=int):
        rec = eps[e]
        if rec.get("observed_late"):
            continue
        prev = eps.get(str(int(e) - 1))
        if prev and prev.get("attempt") == rec.get("attempt") and not prev.get("observed_late"):
            out.append(rec["observed_at"] - prev["observed_at"])
        elif rec.get("attempt") in att and att[rec["attempt"]].get("started_at"):
            out.append(rec["observed_at"] - float(att[rec["attempt"]]["started_at"]))
    return [d for d in out if d > 0]


def json_get(obj, dotted: str):
    cur = obj
    for p in str(dotted).split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def find_numeric_keys(obj, rx, prefix="") -> list[str]:
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = "%s.%s" % (prefix, k) if prefix else str(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and rx.search(str(k)):
                hits.append(key)
            hits += find_numeric_keys(v, rx, key)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_numeric_keys(v, rx, "%s[%d]" % (prefix, i))
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default=None, help="run only this run id's steps (e.g. R1 or ANCHORS)")
    ap.add_argument("--clear-stop", default=None, metavar="NOTE")
    ap.add_argument("--set-commit", default=None, metavar="SHA")
    ap.add_argument("--runs", default=None, help="comma-separated run ids for --set-commit")
    ap.add_argument("--poll-seconds", type=float, default=None)
    args = ap.parse_args(argv)
    r = Runner(args.campaign, poll_seconds=args.poll_seconds)
    if args.status:
        return r.status()
    if args.dry_run:
        return r.dry_run(args.only)
    if args.set_commit:
        return r.set_commit(args.set_commit, args.runs.split(",") if args.runs else None)
    try:
        info = r.lock.acquire()
    except cc.LockHeld as e:
        print("LOCKED: %s" % e)
        return EXIT_LOCKED
    try:
        if info.get("previous_owner"):
            r.say("acquired lock; previous recorded owner %s is gone" % info["previous_owner"])
        if args.clear_stop is not None:
            return r.clear_stop(args.clear_stop)
        if args.only and args.only not in [s.partition(":")[0] for s in r.camp["sequence"]]:
            print("unknown run id %s" % args.only)
            return 1
        return r.run(args.only)
    finally:
        r.lock.release()


if __name__ == "__main__":
    sys.exit(main())
