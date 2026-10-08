"""Graceful pause state machine (V3 sections 8-9 reproductions, converted to required behaviour).

No real trainer or GPU process: the runner is driven directly with a synthetic attempt, a
synthetic log and a JSON stand-in for the -last checkpoint.
"""
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

from scripts import cr_campaign as camp
from tests.test_cr_campaign_runner import env  # noqa: F401  (fixture)


@pytest.fixture
def case(env, monkeypatch):  # noqa: F811
    r = camp.Runner(env.camp_path, poll_seconds=0.01, out=lambda *_: None)
    r.load_state()
    rs = r.rstate("R1")
    sp = camp.run_spec(r.camp, "R1")
    cfg = yaml.safe_load((env.repo / sp["config"]).read_text())
    env.run_dir.mkdir(parents=True)
    log = env.run_dir / "train_a0.log"
    a = {"k": 0, "kind": "fork", "status": "RUNNING", "pid": 2147483646, "create_time": 1.0,
         "log": str(log), "started_at": time.time() - 10, "resume_from_epoch": None}
    tr = rs["train"]
    tr.update({"status": "RUNNING", "run_dir": str(env.run_dir), "tag": env.tag, "attempts": [a],
               "epochs": {str(e): {"attempt": 0, "observed_at": time.time() - 3 + e - 28} for e in range(26, 29)}})
    r.state["status"] = "RUNNING"
    r.state["current"] = {"step": "R1:train", "pid": a["pid"], "create_time": a["create_time"], "log": str(log)}
    last = env.run_dir / (env.tag + "-last.pth.tar")
    last.write_text(json.dumps({"epoch": 28}))
    log.write_text("Epoch 28/100  (1s)  train_loss=0.1194  val_loss=0.1232\n  [CKPT] epoch=28 saved: last\n")
    r.state["pause"] = {"status": "PENDING", "step": "R1:train", "attempt": 0, "target_epoch": 28}
    r.pause_path.parent.mkdir(parents=True, exist_ok=True)
    r.pause_path.write_text(json.dumps({"note": "synthetic pause"}))
    killed, alive = [], {"value": True}

    def stop_own(a, why, tree=None):
        killed.append((a["pid"], a["create_time"], why))
        alive["value"] = False
        return True
    monkeypatch.setattr(r, "kill_own_tree", stop_own)
    monkeypatch.setattr(r, "proc_alive", lambda a: alive["value"])
    monkeypatch.setattr(r, "proc_rc", lambda a: None if alive["value"] else 1)
    monkeypatch.setattr(camp, "read_checkpoint_epoch", lambda path, *x, **k: int(json.loads(Path(path).read_text())["epoch"]))
    return {"env": env, "runner": r, "spec": sp, "cfg": cfg, "attempt": a, "train": tr, "log": log,
            "last": last, "killed": killed, "alive": alive}


def attempt_pause(c):
    return c["runner"].try_pause_trainer("R1", c["spec"], c["attempt"], c["env"].run_dir)


def test_complete_boundary_records_verified_epoch_hash_and_no_restart(case):
    c = case
    assert attempt_pause(c) is True
    a, r = c["attempt"], c["runner"]
    assert r.state["status"] == "PAUSED" and a["status"] == "EXITED" and a["paused"] is True
    assert a["pause"]["epoch"] == 28 and a["pause"]["last_sha256"] == camp.cc.sha256_file(c["last"])
    assert a["pause_intent"]["epoch"] == 28 and a["pause_intent"]["finalized_at"]
    assert c["train"]["restarts_since_ack"] == 0 and not c["alive"]["value"]


def test_request_during_current_save_waits_for_ckpt_marker(case):
    c = case
    c["runner"].state["pause"] = None
    c["last"].write_text(json.dumps({"epoch": 27}))
    c["log"].write_text("  [CKPT] epoch=27 saved: last\nEpoch 28/100  (1s)  train_loss=0.1194  val_loss=0.1232\n")
    assert attempt_pause(c) is False
    assert c["runner"].state["pause"]["target_epoch"] == 28 and c["killed"] == []
    c["last"].write_text(json.dumps({"epoch": 28}))
    with c["log"].open("a") as f:
        f.write("  [CKPT] epoch=28 saved: last\n")
    assert attempt_pause(c) is True and c["attempt"]["pause"]["epoch"] == 28


def test_stale_pending_target_never_kills_during_a_later_epoch_save(case):
    c = case
    with c["log"].open("a") as f:
        f.write("Epoch 29/100  (1s)  train_loss=0.1207  val_loss=0.1213\n")
    c["last"].write_text(json.dumps({"epoch": 29}))  # epoch 29's saves still running
    assert attempt_pause(c) is False and c["killed"] == []
    with c["log"].open("a") as f:
        f.write("  [CKPT] epoch=29 saved: last\n")
    assert attempt_pause(c) is True and c["attempt"]["pause"]["epoch"] == 29  # next complete boundary


def test_stop_epoch_never_killed_even_with_an_older_pending_target(case):
    c = case
    c["runner"].state["pause"]["target_epoch"] = 49
    c["log"].write_text("Epoch 49/100  (1s)  train_loss=0.1413  val_loss=0.1419\n  [CKPT] epoch=49 saved: last\n"
                        "Epoch 50/100  (1s)  train_loss=0.1413  val_loss=0.1423\n"
                        "  [CKPT] epoch=50 saved: last, periodic\n")
    c["last"].write_text(json.dumps({"epoch": 50}))
    assert attempt_pause(c) is False and not c["killed"]


def test_new_request_at_epoch_49_saves_targets_stop_epoch_and_never_kills(case):
    c = case
    c["runner"].state["pause"] = None
    c["log"].write_text("Epoch 49/100  (1s)  train_loss=0.1413  val_loss=0.1419\n  [CKPT] epoch=49 saved: last\n")
    assert attempt_pause(c) is False
    assert c["runner"].state["pause"]["target_epoch"] == 50 and c["runner"].state["pause"]["stop_epoch_in_flight"]
    assert attempt_pause(c) is False and not c["killed"]


def test_failed_taskkill_must_not_declare_paused(case, monkeypatch):
    c = case
    r = c["runner"]
    monkeypatch.setattr(r, "kill_own_tree", camp.Runner.kill_own_tree.__get__(r))
    monkeypatch.setattr(camp.cc, "same_process_alive", lambda *_: True)
    monkeypatch.setattr(camp.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "synthetic access denied"))
    with pytest.raises(camp.CampaignStop, match="could not terminate"):
        attempt_pause(c)
    assert c["alive"]["value"] is True
    assert r.state["status"] != "PAUSED" and r.state["pause"]["status"] == "PENDING"
    assert r.state["current"].get("pid") == c["attempt"]["pid"] and c["attempt"]["status"] == "RUNNING"
    assert "kill_failed" in c["attempt"]["pause_intent"] and "paused" not in c["attempt"]


def test_mismatched_pid_creation_time_never_reaches_taskkill(case, monkeypatch):
    c = case
    monkeypatch.setattr(camp.cc, "same_process_alive", lambda *_: False)
    monkeypatch.setattr(camp.subprocess, "run", lambda *a, **k: pytest.fail("taskkill touched a mismatched PID"))
    assert camp.Runner.kill_own_tree(c["runner"], c["attempt"], "synthetic stale PID") is True


class LaunchIntercept(Exception):
    pass


@pytest.mark.parametrize("path", ["fork", "crash_resume", "pause_resume", "probe"])
def test_request_arriving_during_preflight_prevents_any_new_launch(env, monkeypatch, path):  # noqa: F811
    r = camp.Runner(env.camp_path, poll_seconds=0.01, out=lambda *_: None)
    r.load_state()
    sp = camp.run_spec(r.camp, "R1")
    cfg = yaml.safe_load((env.repo / sp["config"]).read_text())
    rs = r.rstate("R1")
    tr = rs["train"]
    tr.update({"run_dir": str(env.run_dir), "tag": env.tag})
    launched, waited = [], []

    def preflight(*a, **k):
        if not waited:  # the request appears while the slow checks run
            r.pause_path.parent.mkdir(parents=True, exist_ok=True)
            r.pause_path.write_text(json.dumps({"note": "arrived during preflight"}))
        return {"problems": []}

    def launch(*a, **k):
        launched.append(r.pause_path.exists())
        raise LaunchIntercept()
    original_wait = r.wait_until_resumed

    def wait(where):
        if r.pause_requested():
            waited.append(where)
            r.pause_path.unlink()  # the user resumes
        return original_wait(where)
    monkeypatch.setattr(r, "preflight", preflight)
    monkeypatch.setattr(r, "wait_until_resumed", wait)
    monkeypatch.setattr(r, "verify_ancestor", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(r, "ensure_worktree", lambda *a, **k: env.repo)
    monkeypatch.setattr(r, "start_attempt", launch)
    monkeypatch.setattr(r, "launch", launch)
    monkeypatch.setattr(camp, "read_checkpoint_epoch", lambda *a, **k: 28)
    a = {"k": 0, "kind": "fork", "status": "EXITED", "rc": 1, "log": "fake"}
    if path in ("crash_resume", "pause_resume"):
        env.run_dir.mkdir(parents=True)
        last = env.run_dir / (env.tag + "-last.pth.tar")
        last.write_bytes(b"fake verified epoch28")
        tr["attempts"].append(a)
        a["pause"] = {"epoch": 28, "last_path": str(last), "last_sha256": camp.cc.sha256_file(last)}
    with pytest.raises(LaunchIntercept):
        if path in ("fork", "crash_resume"):
            r.do_train("R1", sp)
        elif path == "pause_resume":
            r.resume_paused_attempt("R1", sp, a, env.repo, cfg, env.run_dir, env.tag)
        else:
            anc = r.camp["ancestor"]
            target = {"name": "fake_probe", "mode": "checkpoint", "checkpoint": anc["path"], "sha256": anc["sha256"],
                      "arm": "random", "seed": 1234, "run_uuid": "fake"}
            r.run_probe_target("R1", rs["probe"], target, env.repo)
    # the launch happened only after the pause was honoured and lifted, never with a request present
    assert waited and launched == [False]


def test_request_publication_waits_for_an_in_progress_launch(env):  # noqa: F811
    r = camp.Runner(env.camp_path, poll_seconds=0.01, out=lambda *_: None)
    order = []

    def requester():
        order.append("request-start")
        r.request_pause("concurrent")
        order.append("request-published")
    with r.launch_lock():
        t = threading.Thread(target=requester)
        t.start()
        time.sleep(0.3)
        assert not r.pause_requested()  # cannot be published while a launch is being recorded
        order.append("launch-recorded")
    t.join(timeout=10)
    assert order == ["request-start", "launch-recorded", "request-published"] and r.pause_requested()


def test_paused_last_mutation_refuses_exact_resume(case, monkeypatch):
    c = case
    r = c["runner"]
    assert attempt_pause(c)
    r.pause_path.unlink()
    c["last"].write_text(json.dumps({"epoch": 29}))
    monkeypatch.setattr(r, "preflight", lambda *a, **k: pytest.fail("preflight after digest mismatch"))
    monkeypatch.setattr(r, "start_attempt", lambda *a, **k: pytest.fail("launched mutated checkpoint"))
    with pytest.raises(camp.CampaignStop, match="changed or vanished"):
        r.resume_paused_attempt("R1", c["spec"], c["attempt"], c["env"].repo, c["cfg"], c["env"].run_dir, c["env"].tag)


def test_fresh_runner_keeps_paused_state_until_request_removed(case):
    c = case
    assert attempt_pause(c)
    r2 = camp.Runner(c["env"].camp_path, poll_seconds=0.01, out=lambda *_: None)
    r2.load_state()
    a = r2.rstate("R1")["train"]["attempts"][0]
    launched, errors = [], []
    r2.preflight = lambda *x, **k: {"problems": []}
    r2.start_attempt = lambda *x, **k: launched.append(k)

    def resume():
        try:
            r2.resume_paused_attempt("R1", c["spec"], a, c["env"].repo, c["cfg"], c["env"].run_dir, c["env"].tag)
        except BaseException as e:  # noqa: BLE001
            errors.append(repr(e))
    t = threading.Thread(target=resume)
    t.start()
    try:
        time.sleep(0.15)
        assert t.is_alive() and not launched
        assert json.loads(r2.state_path.read_text())["status"] == "PAUSED"
    finally:
        r2.pause_path.unlink(missing_ok=True)
        t.join(timeout=10)
    assert not t.is_alive() and not errors
    assert launched[0]["kind"] == "exact" and launched[0]["after_pause"] is True and launched[0]["resume_from_epoch"] == 28
    assert r2.rstate("R1")["train"]["restarts_since_ack"] == 0
    restored = yaml.safe_load((r2.state_dir / "configs" / "R1_resume_a1.yaml").read_text())
    assert restored["meta"]["resume_policy"] == "exact" and "fork_start_epoch" not in restored["meta"]
    for section in ("data", "mask", "optimization"):
        assert restored[section] == c["cfg"][section]


def test_runner_death_after_termination_recovers_verification(case, monkeypatch):
    """Runner dies after the journaled intent + verified termination, before -last verification."""
    c = case
    real_read = camp.read_checkpoint_epoch
    calls = {"n": 0}

    def die_once(path, *x, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt()
        return real_read(path, *x, **k)
    monkeypatch.setattr(camp, "read_checkpoint_epoch", die_once)
    with pytest.raises(KeyboardInterrupt):
        attempt_pause(c)
    persisted = json.loads(c["runner"].state_path.read_text())
    pa = persisted["runs"]["R1"]["train"]["attempts"][0]
    assert pa["status"] == "RUNNING" and pa["pause_intent"]["epoch"] == 28 and "paused" not in pa
    assert persisted["pause"]["status"] == "PAUSING" and persisted["status"] != "PAUSED"
    fresh = camp.Runner(c["env"].camp_path, poll_seconds=0.01, out=lambda *_: None)
    fresh.load_state()
    a = fresh.rstate("R1")["train"]["attempts"][0]
    fresh.proc_alive = lambda a: False  # the owned trainer is gone
    fresh.observe_epochs = lambda *x, **k: None
    fresh.monitor_train("R1", c["spec"], a)
    assert a["paused"] is True and a["pause"]["epoch"] == 28 and fresh.state["status"] == "PAUSED"
    assert fresh.rstate("R1")["train"]["restarts_since_ack"] == 0


def test_runner_death_before_termination_reapplies_boundary_rule(case, monkeypatch):
    """Intent journaled, runner died before terminating: the trainer is alive; the intent is dropped
    and the boundary rule re-applied by the next runner."""
    c = case
    r = c["runner"]
    a = c["attempt"]
    a["pause_intent"] = {"epoch": 28, "at": "t"}
    r.state["pause"]["status"] = "PAUSING"
    r.save()
    fresh = camp.Runner(c["env"].camp_path, poll_seconds=0.01, out=lambda *_: None)
    fresh.load_state()
    fa = fresh.rstate("R1")["train"]["attempts"][0]
    alive = {"v": True}
    fresh.proc_alive = lambda a: alive["v"]
    fresh.proc_rc = lambda a: None if alive["v"] else 1
    fresh.observe_epochs = lambda *x, **k: None
    killed = []

    def stop_own(a, why, tree=None):
        killed.append(why)
        alive["v"] = False
        return True
    fresh.kill_own_tree = stop_own
    fresh.monitor_train("R1", c["spec"], fa)
    assert len(killed) == 1 and fa["paused"] is True and fa["pause"]["epoch"] == 28
    assert fresh.state["status"] == "PAUSED"


def test_legacy_paused_attempt_without_receipt_recovers_to_idle(case):
    """V3 P1-D shape: EXITED/paused persisted without the verification record."""
    c = case
    a = c["attempt"]
    a.update({"status": "EXITED", "paused": True})
    c["runner"].save()
    fresh = camp.Runner(c["env"].camp_path, poll_seconds=0.01, out=lambda *_: None)
    fresh.load_state()
    fa = fresh.rstate("R1")["train"]["attempts"][0]
    reached = []

    def idle(where):  # stands in for the real wait: the operator resumes
        reached.append(where)
        fresh.pause_path.unlink(missing_ok=True)
    fresh.wait_until_resumed = idle
    fresh.preflight = lambda *x, **k: {"problems": []}
    fresh.start_attempt = lambda *x, **k: None
    fresh.resume_paused_attempt("R1", c["spec"], fa, c["env"].repo, c["cfg"], c["env"].run_dir, c["env"].tag)
    assert reached and fa["pause"]["epoch"] == 28 and fa["pause_intent"]["recovered"] is True


# ---------------------------------------------------------------------------
# V3 section 9: stop epoch already running; descendants must be verified across restart
# ---------------------------------------------------------------------------

def test_stop_epoch_already_training_is_not_interrupted(case):
    c = case
    c["runner"].state["pause"]["target_epoch"] = 49
    c["log"].write_text("Epoch 49/100  (1s)  train_loss=0.1413  val_loss=0.1419\n"
                        "  [CKPT] epoch=49 saved: last\n"
                        "  [Epoch 50/100 | Iter 50/9375] loss=0.1413\n")
    c["last"].write_text(json.dumps({"epoch": 49}))
    assert attempt_pause(c) is False and not c["killed"]


def test_epoch_49_boundary_defers_until_stop_epoch_is_pinned(case):
    c = case
    c["runner"].state["pause"]["target_epoch"] = 49
    c["log"].write_text("Epoch 49/100  (1s)  train_loss=0.1413  val_loss=0.1419\n  [CKPT] epoch=49 saved: last\n")
    c["last"].write_text(json.dumps({"epoch": 49}))
    assert attempt_pause(c) is False and not c["killed"]  # the trainer is entering epoch 50


def _tree_mocks(monkeypatch, live, root_pid, child_pid):
    class Child:
        pid = child_pid

    class Root:
        def children(self, recursive=False):
            return [Child()]
    monkeypatch.setattr(camp.cc.psutil, "Process", lambda pid: Root())
    monkeypatch.setattr(camp.cc, "process_create_time", lambda pid: 1.0 if pid == root_pid else 2.0)
    monkeypatch.setattr(camp.cc, "same_process_alive", lambda pid, ct: live.get(pid, False))


def test_pause_journals_full_tree_identity_before_signalling(case, monkeypatch):
    c = case
    r = c["runner"]
    root_pid, child_pid = c["attempt"]["pid"], 2147483645
    live = {root_pid: True, child_pid: True}
    _tree_mocks(monkeypatch, live, root_pid, child_pid)
    monkeypatch.setattr(r, "proc_alive", lambda _: live[root_pid])
    seen = {}

    def kill(a, why, tree=None):
        seen["persisted"] = json.loads(r.state_path.read_text())["runs"]["R1"]["train"]["attempts"][0]["pause_intent"]
        live[root_pid] = live[child_pid] = False
        return True
    monkeypatch.setattr(r, "kill_own_tree", kill)
    assert attempt_pause(c) is True
    assert seen["persisted"]["tree"] == [[root_pid, 1.0], [child_pid, 2.0]]
    assert r.state["status"] == "PAUSED"


def test_restart_after_root_exit_must_recheck_owned_descendants(case, monkeypatch):
    c = case
    r = c["runner"]
    root_pid, child_pid = c["attempt"]["pid"], 2147483645
    live = {root_pid: True, child_pid: True}
    r.camp["kill_wait_seconds"] = 60
    _tree_mocks(monkeypatch, live, root_pid, child_pid)
    monkeypatch.setattr(r, "proc_alive", lambda _: live[root_pid])
    monkeypatch.setattr(r, "kill_own_tree", camp.Runner.kill_own_tree.__get__(r))

    def partial_taskkill(argv, **kwargs):
        live[root_pid] = False
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(camp.subprocess, "run", partial_taskkill)
    original_sleep = camp.time.sleep

    def power_loss_while_waiting_for_child(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr(camp.time, "sleep", power_loss_while_waiting_for_child)
    with pytest.raises(KeyboardInterrupt):
        attempt_pause(c)
    monkeypatch.setattr(camp.time, "sleep", original_sleep)
    assert not live[root_pid] and live[child_pid]
    persisted = json.loads(r.state_path.read_text())
    assert persisted["pause"]["status"] == "PAUSING"
    assert persisted["runs"]["R1"]["train"]["attempts"][0]["pause_intent"]["tree"] == [[root_pid, 1.0], [child_pid, 2.0]]
    fresh = camp.Runner(c["env"].camp_path, poll_seconds=0.01, out=lambda *_: None)
    fresh.camp["pause_recovery_wait_seconds"] = 0.5
    fresh.load_state()
    a = fresh.rstate("R1")["train"]["attempts"][0]
    fresh.proc_alive = lambda _: live[root_pid]
    fresh.proc_rc = lambda _: 1
    fresh.observe_epochs = lambda *x, **k: None
    with pytest.raises(camp.CampaignStop, match="still"):
        fresh.monitor_train("R1", c["spec"], a)
    assert live[child_pid] and fresh.state["status"] != "PAUSED" and "paused" not in a
    # once the orphaned descendant has exited, the next restart finalizes the pause
    live[child_pid] = False
    fresh2 = camp.Runner(c["env"].camp_path, poll_seconds=0.01, out=lambda *_: None)
    fresh2.load_state()
    a2 = fresh2.rstate("R1")["train"]["attempts"][0]
    fresh2.proc_alive = lambda _: False
    fresh2.proc_rc = lambda _: 1
    fresh2.observe_epochs = lambda *x, **k: None
    fresh2.monitor_train("R1", c["spec"], a2)
    assert a2["paused"] is True and a2["pause"]["epoch"] == 28 and fresh2.state["status"] == "PAUSED"
