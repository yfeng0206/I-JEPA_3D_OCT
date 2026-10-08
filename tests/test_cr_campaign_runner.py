"""End-to-end tests of scripts/cr_campaign.py with a fake git repo, fake trainer and fake probe.

The fake trainer/probe live in a throwaway git repository (they are what the runner
checks out into its pinned worktree), so the worktree, launch, monitor, gate, pin,
resume and probe-validation paths all run for real on the CPU in a few seconds.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from scripts import cr_campaign as camp_mod
from scripts import cr_common as cc
from scripts import cr_make_configs as mk

FAKE_TRAINER = r'''
import argparse, hashlib, json, os, subprocess, sys, time, uuid
import torch, yaml
ap = argparse.ArgumentParser()
ap.add_argument("--config"); ap.add_argument("--stop-after-epoch", type=int)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
out, tag, meta = cfg["logging"]["folder"], cfg["logging"]["write_tag"], cfg["meta"]
ctl = json.load(open(os.environ["CR_FAKE_CONTROL"]))
attempt = len([f for f in os.listdir(out) if f.startswith("train_a") and f.endswith(".log")]) - 1
mode = ctl.get("modes", {}).get(str(attempt), ctl.get("mode", "ok"))
ref = json.load(open(ctl["reference"]))["arms"][ctl.get("curve", "random")]
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
if meta["resume_policy"] == "fork":
    if any(f.endswith(".pth.tar") for f in os.listdir(out)):
        print("refusing fork into folder with checkpoints"); sys.exit(1)
    start = int(meta["fork_start_epoch"])
    run_uuid = uuid.uuid4().hex
else:
    assert "fork_start_epoch" not in meta
    _ck = torch.load(meta["read_checkpoint"], weights_only=False)
    start = int(_ck["epoch"])
    run_uuid = _ck["training_state"]["run_uuid"] if mode != "uuid_change" else uuid.uuid4().hex
cur = cfg["mask"].get("curriculum") or {}
mask_policy = cur["mode"] if cur.get("enabled") else "uniform_multiblock"
contract = hashlib.sha256(json.dumps({k: cfg[k] for k in ("data", "mask", "optimization")},
                                     sort_keys=True).encode()).hexdigest()
if mode == "contract_change":
    contract = "f" * 64
json.dump({"git_commit": commit, "git_dirty": False, "resume_policy": meta["resume_policy"], "seed": meta["seed"],
           "attempt": attempt, "argv": sys.argv, "run_uuid": run_uuid, "mask_policy": mask_policy,
           "run_contract_sha256": contract, "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
           "invocation": {"config_file_sha256": hashlib.sha256(open(a.config, "rb").read()).hexdigest(),
                          "stop_after_epoch": a.stop_after_epoch},
           "amp": {"amp_target": True if mode == "bad_manifest" else meta["amp_target"]},
           "batch": {"world_size": 1, "batch_size": cfg["data"]["batch_size"],
                     "accum_steps": cfg["optimization"]["accum_steps"],
                     "effective_batch": cfg["data"]["batch_size"] * cfg["optimization"]["accum_steps"],
                     "epochs": cfg["optimization"]["epochs"]},
           "config": cfg}, open(os.path.join(out, "run_manifest.json"), "w"))
def save(path, epoch):
    torch.save({"epoch": epoch, "encoder": {"w": torch.zeros(2)}, "predictor": {}, "target_encoder": {},
                "opt": {"param_groups": [{}]}, "scaler": None,
                "training_state": {"run_uuid": run_uuid}}, path + ".tmp")
    os.replace(path + ".tmp", path)
def write_stop(e):
    ck = os.path.join(out, "%s-ep%d.pth.tar" % (tag, e))
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
    payload = {"schema": "jepa_stop_epoch_v1", "epoch": e, "status": "verified", "git_commit": commit,
               "git_dirty": False, "write_tag": tag, "output_dir": os.path.abspath(out),
               "run_uuid": run_uuid, "mask_policy": mask_policy, "run_contract_sha256": contract,
               "checkpoints": [{"role": "periodic", "path": os.path.abspath(ck), "sha256": sha,
                                "bytes": os.path.getsize(ck), "epoch": e}]}
    json.dump(payload, open(os.path.join(out, "stop_epoch_%03d.json" % e), "w"))
    print("STOP: epoch %d" % e, flush=True)
if mode == "die_early":
    print("crash before first epoch", flush=True); sys.exit(1)
if meta["resume_policy"] == "exact" and start >= a.stop_after_epoch:
    # i1 trainer: exact resume already at the stop epoch verifies epN and writes the stop file.
    if start != a.stop_after_epoch or not os.path.exists(os.path.join(out, "%s-ep%d.pth.tar" % (tag, start))):
        sys.exit(1)
    print("recovered without training", flush=True)
    write_stop(start)
    sys.exit(0)
for e in range(start + 1, 101):
    if mode == "crash" and e == ctl.get("crash_epoch", start + 2):
        print("simulated crash in epoch %d" % e, flush=True); sys.exit(1)
    tr, va = ref["train"][str(e)], ref["val"][str(e)]
    if mode in ("nan", "nan_exit4") and e == ctl["nan_epoch"]:
        tr = float("nan")
    for it in (50, 100):
        print("  [Epoch %d/100 | Iter %d/9375] loss=%.4f  lr=2.0e-04  wd=0.1  ema=0.997  gpu=1MB" % (e, it, tr), flush=True)
    print("Epoch %d/100  (1s)  train_loss=%.4f  val_loss=%.4f" % (e, tr, va), flush=True)
    time.sleep(ctl.get("epoch_sleep", 0.0))
    if mode == "nan" and e == ctl["nan_epoch"]:
        time.sleep(120)
    if mode == "nan_exit4" and e == ctl["nan_epoch"]:
        sys.exit(4)
    save(os.path.join(out, tag + "-last.pth.tar"), e)
    if e % 5 == 0 or e == a.stop_after_epoch:
        save(os.path.join(out, "%s-ep%d.pth.tar" % (tag, e)), e)
    if e == a.stop_after_epoch:
        if mode == "exit0_nostop":
            sys.exit(0)
        if mode == "crash_after_save":
            print("crash between the epoch-%d saves and the stop file" % e, flush=True); sys.exit(1)
        if mode == "relabel":
            save(os.path.join(out, "%s-ep%d.pth.tar" % (tag, e)), e + 1)
        write_stop(e)
        sys.exit(0)
sys.exit(3)
'''

FAKE_PROBE = r'''
import argparse, hashlib, json, os, sys, time
ap = argparse.ArgumentParser()
for f in ("--config", "--encoder-checkpoint", "--output-dir", "--head-seeds", "--extra-pooling", "--arm",
          "--train-seed", "--role", "--run-uuid", "--run-provenance", "--legacy-cache-policy",
          "--expected-checkpoint-sha256"):
    ap.add_argument(f)
ap.add_argument("--seal-test", action="store_true")
a = ap.parse_args()
ctl = json.load(open(os.environ["CR_FAKE_CONTROL"]))
mode = ctl.get("probe_mode", "ok")
if mode == "fail":
    print("probe exploded"); sys.exit(1)
os.makedirs(a.output_dir)
legacy = a.legacy_cache_policy == "allow_verified"
sha = a.expected_checkpoint_sha256 if legacy else hashlib.sha256(open(a.encoder_checkpoint, "rb").read()).hexdigest()
ident = {"checkpoint_sha256": sha, "arm": a.arm, "role": a.role, "run_uuid": a.run_uuid, "epoch": 50,
         "train_seed": int(a.train_seed) if a.train_seed else None}
if a.run_provenance:
    prov = json.load(open(a.run_provenance))
    ident["run_provenance_run_uuid"] = prov.get("run_uuid")
    if "checkpoints" in prov:
        roles = [c["role"] for c in prov["checkpoints"] if c["sha256"] == sha]
        if not roles:
            print("probed checkpoint not in provenance stop file"); sys.exit(1)
        ident["run_provenance_checkpoint_role"] = "periodic" if "periodic" in roles else roles[0]
if mode == "wrong_sha":
    ident["checkpoint_sha256"] = "0" * 64
if mode == "wrong_seed":
    ident["train_seed"] = 1
if mode == "wrong_prov":
    ident["run_provenance_run_uuid"] = "deadbeef"
TOFU = "tofu_pinned_2026-10-08; historical integrity supported by timestamps and head replay, not cryptographic"
def fsha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
variant = "patchmean_slicemean"
per_seed, entries = [], []
for hs in [int(x) for x in (a.head_seeds or "42,43").split(",")]:
    d = os.path.join(a.output_dir, variant, "seed%d" % hs)
    os.makedirs(d)
    pred = os.path.join(d, "test_predictions_sealed_seed%d.npz" % hs)
    open(pred, "wb").write(b"sealed probs %d" % hs)
    side = pred[:-4] + ".manifest.json"
    open(side, "w").write(json.dumps({"sealed": True, "head_seed": hs}))
    rel, srel = os.path.relpath(pred, a.output_dir), os.path.relpath(side, a.output_dir)
    per_seed.append({"head_seed": hs, "best_val_auc": 0.851, "test_predictions": rel,
                     "test_predictions_sha256": fsha(pred)})
    entries.append({"variant": variant, "head_seed": hs, "path": rel, "sha256": fsha(pred),
                    "sidecar_path": srel, "sidecar_sha256": fsha(side)})
res = {"schema": "cr_probe_v1", "status": "complete", "sealed": bool(a.seal_test), "best_val_auc": 0.851,
       "test_auc": None, "identity": ident,
       "variants": {variant: {"val_auc_mean": 0.851, "per_seed": per_seed}},
       "legacy_cache_integrity": (TOFU if legacy else None) if mode != "legacy_bad" else "unpinned",
       "cache_provenance": {sp: {"primary": {"state": "legacy_verified" if (legacy or mode == "bad_cache_state")
                                              else "cold"}} for sp in ("Training", "Validation", "Test")}}
if mode == "test_leak":
    res["variants"][variant]["test_auc_mean"] = 0.87
seal = {"schema": "cr_sealed_v1", "identity": dict(ident), "metrics_computed": False,
        "test_label_identity": {"split": "Test", "n": 3000, "labels_int8_sha256": "ab" * 32},
        "files": entries}
if mode == "seal_identity":
    seal["identity"]["run_uuid"] = "foreign-run"
if mode == "seal_metrics":
    seal["metrics_computed"] = True
if mode == "seal_missing_file":
    seal["files"].append({"variant": variant, "head_seed": 99, "path": "missing.npz", "sha256": "0" * 64})
if mode == "seal_unbound":
    res["variants"][variant]["per_seed"][0]["test_predictions_sha256"] = "1" * 64
seal_path = os.path.join(a.output_dir, "sealed_manifest.json")
if mode != "no_sealed":
    text = {"seal_empty": "", "seal_invalid": "{"}.get(mode, json.dumps(seal, indent=2))
    open(seal_path, "w").write(text)
    open(seal_path + ".sha256", "w").write("%s  sealed_manifest.json\n" % (
        "f" * 64 if mode == "seal_receipt" else fsha(seal_path)))
if mode == "seal_altered_file":
    open(os.path.join(a.output_dir, entries[0]["path"]), "ab").write(b"tampered")
for name, obj in (("results.json", res), ("cache_provenance.json", {})):
    json.dump(obj, open(os.path.join(a.output_dir, name), "w"))
if mode == "stale":
    for name in ("results.json", "sealed_manifest.json"):
        p = os.path.join(a.output_dir, name)
        os.utime(p, (time.time() - 86400, time.time() - 86400))
print("probe ok")
'''

FAKE_STATS = r'''
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("cmd"); ap.add_argument("--run"); ap.add_argument("--anchor"); ap.add_argument("--out")
a = ap.parse_args()
mode = json.load(open(os.environ["CR_FAKE_CONTROL"])).get("g2_mode", "ok")
if mode == "integrity":
    print("ERROR: integrity", file=sys.stderr); sys.exit(2)
assert os.path.isdir(a.run) and os.path.isdir(a.anchor)
json.dump({"validation_only": True, "gates": [{"run": a.run, "anchor": a.anchor, "primary_flag": mode == "flag"}]},
          open(a.out, "w"))
print("random patchmean_slicemean 0.8510 0.8450 +0.0060 %s" % ("FLAG" if mode == "flag" else "ok"))
'''


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t"] + list(args),
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fake repo + campaign for one RANDOM run (R1) and its probe."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "configs" / "cr_seed_v1").mkdir(parents=True)
    (repo / "src" / "train_patch.py").write_text(FAKE_TRAINER, encoding="utf-8")
    (repo / "src" / "eval_downstream.py").write_text(FAKE_PROBE, encoding="utf-8")
    (repo / "autopilot").mkdir()
    (repo / "autopilot" / "cr_stats.py").write_text(FAKE_STATS, encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    run_root = tmp_path / "runs"
    ancestor = tmp_path / "ancestor.pth.tar"
    ancestor.write_bytes(b"fake ancestor bytes")
    cfgs = {}
    for arm, seed in (("random", 1234), ("centroid", 1234)):
        cfg = mk.build_config(arm, seed)
        cfg["logging"]["folder"] = mk.run_folder(arm, seed, str(run_root))
        cfg["meta"]["read_checkpoint"] = str(ancestor)
        rel = "configs/cr_seed_v1/cr_seed_v1_%s_s%d.yaml" % (arm, seed)
        (repo / rel).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        cfgs[arm] = cfg
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fake")
    commit = git(repo, "rev-parse", "HEAD")
    control = tmp_path / "control.json"
    ctl = {"reference": str(Path(camp_mod.gate.REFERENCE)), "mode": "ok", "curve": "random"}
    control.write_text(json.dumps(ctl), encoding="utf-8")
    probe_cmd = ["{python}", "-u", "src/eval_downstream.py", "--config", "{probe_config}",
                 "--encoder-checkpoint", "{checkpoint}", "--output-dir", "{out_dir}", "--head-seeds",
                 "{head_seeds}", "--seal-test", "--extra-pooling", "patch_max", "--arm", "{arm}",
                 "--train-seed", "{seed}", "--role", "new", "--run-uuid", "{run_uuid}",
                 "--run-provenance", "{run_provenance}"]
    camp = {
        "schema": "cr_campaign_v1", "campaign": "cr_test", "repo": str(repo), "python": sys.executable,
        "state_dir": str(tmp_path / "state"), "worktree_root": str(tmp_path / "wt"), "run_root": str(run_root),
        "reference_curves": str(camp_mod.gate.REFERENCE),
        "ancestor": {"path": str(ancestor), "sha256": cc.sha256_file(ancestor), "size": ancestor.stat().st_size},
        "fork_start_epoch": 25, "stop_epoch": 50, "max_restarts": 2, "poll_seconds": 0.1,
        "child_env": {"CR_FAKE_CONTROL": str(control)},
        "preflight": {"min_free_gib": {}, "gpu_clear_wait_seconds": 0},
        "train_command": ["{python}", "-u", "src/train_patch.py", "--config", "{config}",
                          "--stop-after-epoch", "{stop_epoch}"],
        "probe": {"spec_confirmed": True, "command": probe_cmd, "cache_splits": ["Training", "Validation", "Test"],
                  "fresh_cache_states": ["cold", "warm"], "check_legacy_integrity": True,
                  "anchor_command": probe_cmd[:14] + ["--arm", "{arm}", "--role", "anchor", "--run-uuid", "{run_uuid}"],
                  "config": "configs/none.yaml", "head_seeds": "42,43", "out_root": str(tmp_path / "probes"),
                  "outputs": {"aggregate": "results.json", "sealed_manifest": "sealed_manifest.json"},
                  "encoder_sha_field": "identity.checkpoint_sha256", "val_auc_field": "best_val_auc",
                  "required_values": {"schema": "cr_probe_v1", "status": "complete", "sealed": True},
                  "identity_arm_field": "identity.arm",
                  "forbid_numeric_key_regex": "(?i)(test.*auc|auc.*test)"},
        "runs": [{"id": "R1", "arm": "random", "seed": 1234, "git_commit": commit, "enabled": True,
                  "config": "configs/cr_seed_v1/cr_seed_v1_random_s1234.yaml", "stop_epoch": 50,
                  "probe": {"enabled": True}},
                 {"id": "C1", "arm": "centroid", "seed": 1234, "git_commit": commit, "enabled": False,
                  "config": "configs/cr_seed_v1/cr_seed_v1_centroid_s1234.yaml", "stop_epoch": 50}],
        "sequence": ["R1:train", "R1:probe", "C1:train", "C1:probe"],
        "g2": {"enabled": True, "command": ["{python}", "-u", "autopilot/cr_stats.py", "gate", "--run",
                                            "{run_probe_dir}", "--anchor", "{anchor_probe_dir}", "--out", "{out_json}"]},
    }
    camp_path = tmp_path / "campaign.json"
    camp_path.write_text(json.dumps(camp, indent=1), encoding="utf-8")
    monkeypatch.setattr(camp_mod, "gpu_compute_apps", lambda: [])
    monkeypatch.setattr(camp_mod, "foreign_compute_python", lambda exclude_pids=(): [])

    class E:
        pass
    e = E()
    e.tmp, e.repo, e.commit, e.camp_path, e.camp, e.control = tmp_path, repo, commit, camp_path, camp, control
    e.run_dir = Path(cfgs["random"]["logging"]["folder"])
    e.tag = cfgs["random"]["logging"]["write_tag"]
    e.state_path = tmp_path / "state" / "state.json"

    def set_ctl(**kw):
        d = json.loads(control.read_text())
        d.update(kw)
        control.write_text(json.dumps(d), encoding="utf-8")
    e.set_ctl = set_ctl

    def rewrite(**kw):
        c = json.loads(camp_path.read_text())
        for k, v in kw.items():
            c[k] = v
        camp_path.write_text(json.dumps(c, indent=1), encoding="utf-8")
    e.rewrite = rewrite

    def run(*args):
        return camp_mod.main(["--campaign", str(camp_path), "--poll-seconds", "0.1"] + list(args))
    e.run = run
    e.state = lambda: json.loads(e.state_path.read_text())
    return e


def alerts(e, kind):
    return sorted((e.tmp / "state").glob("ALERT_%s_*.txt" % kind))


def test_success_pins_only_epoch_50_and_probes(env, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    assert env.run() == 0
    assert json.loads((env.run_dir / "run_manifest.json").read_text())["alloc_conf"] is None
    st = env.state()
    assert st["status"] == "DONE"
    tr = st["runs"]["R1"]["train"]
    assert tr["status"] == "DONE" and len(tr["attempts"]) == 1 and tr["attempts"][0]["kind"] == "fork"
    pinned = list((env.run_dir / "pinned").iterdir())
    assert [p.name for p in pinned] == ["%s-ep050.pth.tar" % env.tag]
    import torch
    assert torch.load(pinned[0], weights_only=False)["epoch"] == 50
    assert tr["pinned"]["sha256"] == cc.sha256_file(pinned[0]) == cc.sha256_file(env.run_dir / ("%s-ep50.pth.tar" % env.tag))
    assert sorted(int(k) for k in tr["epochs"]) == list(range(26, 51))
    assert tr["latest_gate"]["status"] == "PASS"
    pr = st["runs"]["R1"]["probe"]
    assert pr["status"] == "DONE" and pr["result"]["val_auc"] == pytest.approx(0.851)
    out_dir = Path(pr["result"]["out_dir"])
    assert tr["pinned"]["sha256"][:12] in out_dir.name and out_dir.name.endswith("_a0")
    argv = pr["attempts"][0]["argv"]
    assert argv[argv.index("--encoder-checkpoint") + 1] == str(pinned[0])
    man = json.loads((env.run_dir / "run_manifest.json").read_text())
    assert man["mask_policy"] == "uniform_multiblock"
    assert argv[argv.index("--run-uuid") + 1] == man["run_uuid"] == st["runs"]["R1"]["trainer_run_uuid"]
    assert argv[argv.index("--run-provenance") + 1] == str(env.run_dir / "stop_epoch_050.json")
    assert st["runs"]["C1"]["train"]["status"] == "DISABLED"
    assert st["runs"]["R1"]["g2"]["status"] == "NO_ANCHOR"
    wt = Path(env.camp["worktree_root"]) / env.commit
    assert git(wt, "rev-parse", "HEAD") == env.commit
    assert (env.tmp / "state" / "projection.json").exists()
    assert alerts(env, "STOP") == []
    assert env.run() == 0  # idempotent: nothing relaunched
    assert len(env.state()["runs"]["R1"]["train"]["attempts"]) == 1


def test_crash_resumes_exact_without_fork_fields(env):
    env.set_ctl(modes={"0": "crash"}, crash_epoch=29)
    assert env.run() == 0
    tr = env.state()["runs"]["R1"]["train"]
    assert [a["kind"] for a in tr["attempts"]] == ["fork", "exact"]
    assert tr["attempts"][1]["resume_from_epoch"] == 28
    rcfg = yaml.safe_load(open(tr["attempts"][1]["config"]))
    assert rcfg["meta"]["resume_policy"] == "exact" and "fork_start_epoch" not in rcfg["meta"]
    assert rcfg["meta"]["read_checkpoint"] == str(env.run_dir / ("%s-last.pth.tar" % env.tag))
    fork = yaml.safe_load(open(tr["attempts"][0]["config"]))
    d = mk.diff_flat(fork, rcfg)
    assert set(d["changed"]) == {"meta.resume_policy", "meta.read_checkpoint"} and set(d["removed"]) == {"meta.fork_start_epoch"}
    assert (env.run_dir / "train_a0.log").exists() and (env.run_dir / "train_a1.log").exists()
    assert tr["status"] == "DONE" and tr["pinned"]["epoch"] == 50


def test_crash_after_epoch50_save_recovers_via_exact_resume(env):
    # Crash between the epoch-50 saves and the stop file: the exact resume must not train
    # (the trainer verifies ep50 and writes the stop file); the runner re-verifies and pins ep50.
    env.set_ctl(modes={"0": "crash_after_save"})
    assert env.run() == 0
    tr = env.state()["runs"]["R1"]["train"]
    assert [a["kind"] for a in tr["attempts"]] == ["fork", "exact"]
    assert tr["attempts"][1]["resume_from_epoch"] == 50
    assert "recovered without training" in (env.run_dir / "train_a1.log").read_text()
    assert tr["pinned"]["epoch"] == 50 and sorted(p.name for p in (env.run_dir / "pinned").iterdir()) == [
        "%s-ep050.pth.tar" % env.tag]


def test_repeated_crash_hits_restart_cap_and_stops(env):
    env.set_ctl(mode="crash", crash_epoch=None)
    env.set_ctl(modes={"0": "crash", "1": "crash", "2": "crash"})
    d = json.loads(env.control.read_text())
    d.pop("crash_epoch")
    env.control.write_text(json.dumps(d))
    assert env.run() == camp_mod.EXIT_STOPPED
    st = env.state()
    assert st["status"] == "STOPPED" and "restart budget" in st["stop_reason"]
    assert [a["kind"] for a in st["runs"]["R1"]["train"]["attempts"]] == ["fork", "exact", "exact"]
    assert alerts(env, "STOP")
    assert env.run() == camp_mod.EXIT_STOPPED  # stays stopped until acknowledged
    assert len(env.state()["runs"]["R1"]["train"]["attempts"]) == 3
    env.set_ctl(modes={})
    env.set_ctl(mode="ok")
    assert env.run("--clear-stop", "checked") == 0
    assert env.run() == 0
    tr = env.state()["runs"]["R1"]["train"]
    assert [a["kind"] for a in tr["attempts"]] == ["fork", "exact", "exact", "exact"]


def test_crash_before_first_checkpoint_never_reforks(env):
    env.set_ctl(mode="die_early")
    assert env.run() == camp_mod.EXIT_STOPPED
    st = env.state()
    assert "refusing to re-fork" in st["stop_reason"]
    assert len(st["runs"]["R1"]["train"]["attempts"]) == 1


def test_exit0_without_stop_file_stops(env):
    env.set_ctl(mode="exit0_nostop")
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "without stop_epoch_050.json" in env.state()["stop_reason"]
    assert not (env.run_dir / "pinned").exists()


def test_manifest_mismatch_stops_and_terminates(env):
    env.set_ctl(mode="bad_manifest", epoch_sleep=0.5)
    assert env.run() == camp_mod.EXIT_STOPPED
    st = env.state()
    assert "amp.amp_target" in st["stop_reason"]
    a = st["runs"]["R1"]["train"]["attempts"][0]
    assert a["terminated_by_runner"] and not cc.same_process_alive(a["pid"], a["create_time"])
    assert not (env.run_dir / "pinned").exists()


def test_relabelled_epoch_is_rejected(env):
    env.set_ctl(mode="relabel")
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "deserializes to epoch 51" in env.state()["stop_reason"]
    assert not (env.run_dir / "pinned").exists()


def test_nonempty_run_dir_refused(env):
    env.run_dir.mkdir(parents=True)
    (env.run_dir / "jepa_patch_old-last.pth.tar").write_bytes(b"x")
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "not empty" in env.state()["stop_reason"]
    assert env.state()["runs"]["R1"]["train"]["attempts"] == []


def test_ancestor_mismatch_stops_before_launch(env):
    c = json.loads(env.camp_path.read_text())
    c["ancestor"]["sha256"] = "f" * 64
    env.camp_path.write_text(json.dumps(c))
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "ancestor check failed" in env.state()["stop_reason"]
    assert not env.run_dir.exists()


def test_preflight_gpu_python_process_blocks_launch(env, monkeypatch):
    monkeypatch.setattr(camp_mod, "gpu_compute_apps",
                        lambda: [{"pid": 99999, "name": r"C:\Python311\python.exe", "used_mb": "[N/A]"},
                                 {"pid": 5, "name": r"C:\Windows\explorer.exe", "used_mb": "[N/A]"}])
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "python/torch processes on the GPU" in env.state()["stop_reason"]
    assert not env.run_dir.exists()


def test_nan_gate_stops_and_terminates_own_trainer(env):
    env.set_ctl(mode="nan", nan_epoch=28)
    t0 = time.time()
    assert env.run() == camp_mod.EXIT_STOPPED
    assert time.time() - t0 < 100, "trainer should have been terminated, not waited for"
    st = env.state()
    assert "G1 STOP" in st["stop_reason"]
    a = st["runs"]["R1"]["train"]["attempts"][0]
    assert not cc.same_process_alive(a["pid"], a["create_time"])
    assert alerts(env, "STOP")


def test_hold_writes_alert_and_continues(env):
    env.set_ctl(curve="envelope")  # a RANDOM run on the ENVELOPE curve: bands + identity HOLD
    assert env.run() == 0
    tr = env.state()["runs"]["R1"]["train"]
    assert tr["status"] == "DONE"
    assert alerts(env, "HOLD_R1")
    assert any(g["status"] == "HOLD" for g in tr["gates"].values())


@pytest.mark.parametrize("mode,msg", [("fail", "exited rc=1"), ("wrong_sha", "probe identity"),
                                      ("wrong_seed", "identity.train_seed"), ("wrong_prov", "provenance run_uuid"),
                                      ("seal_empty", "invalid test seal"), ("seal_invalid", "invalid test seal"),
                                      ("seal_receipt", "invalid test seal"), ("seal_identity", "invalid test seal"),
                                      ("seal_metrics", "metrics_computed"), ("seal_missing_file", "missing"),
                                      ("seal_altered_file", "hash mismatch"), ("seal_unbound", "differs from the seal"),
                                      ("bad_cache_state", "primary.state"), ("legacy_bad", "legacy_cache_integrity"),
                                      ("stale", "predates the launch"), ("no_sealed", "sealed_manifest"),
                                      ("test_leak", "sealed-test violation")])
def test_probe_failures_stop(env, mode, msg):
    env.set_ctl(probe_mode=mode)
    assert env.run() == camp_mod.EXIT_STOPPED
    st = env.state()
    assert msg in st["stop_reason"]
    assert st["runs"]["R1"]["train"]["status"] == "DONE"
    assert st["runs"]["R1"]["probe"]["attempts"][-1]["status"] == "FAILED"
    # retry after acknowledgement uses a NEW probe directory
    env.set_ctl(probe_mode="ok")
    assert env.run("--clear-stop", "fixed") == 0
    assert env.run() == 0
    pr = env.state()["runs"]["R1"]["probe"]
    assert pr["status"] == "DONE" and pr["result"]["out_dir"].endswith("_a1")


def test_preexisting_probe_dir_is_never_reused(env):
    env.rewrite(sequence=["R1:train"])
    assert env.run() == 0
    sha = env.state()["runs"]["R1"]["train"]["pinned"]["sha256"]
    stale = Path(env.camp["probe"]["out_root"]) / ("cr_probe_R1_random_s1234_ep050_%s_a0" % sha[:12])
    stale.mkdir(parents=True)
    (stale / "results.json").write_text(json.dumps({"best_val_auc": 0.99}))
    env.rewrite(sequence=["R1:train", "R1:probe"])
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "refusing to reuse" in env.state()["stop_reason"]


def test_unconfirmed_probe_spec_stops(env):
    c = json.loads(env.camp_path.read_text())
    c["probe"]["spec_confirmed"] = False
    env.camp_path.write_text(json.dumps(c))
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "spec_confirmed" in env.state()["stop_reason"]


def test_reattach_to_running_trainer_after_runner_death(env):
    env.set_ctl(epoch_sleep=0.05)
    r = camp_mod.Runner(env.camp_path, poll_seconds=0.1, out=lambda *_: None)
    r.load_state()
    spec = camp_mod.run_spec(r.camp, "R1")
    rs = r.rstate("R1")
    rs["spec"] = camp_mod.frozen_spec(spec)
    wt = r.ensure_worktree(env.commit)
    cfg_path, cfg, sha = r.load_run_config(spec, wt)
    rs["train"].update({"run_dir": str(env.run_dir), "tag": env.tag, "status": "RUNNING"})
    r.start_attempt("R1", spec, wt, cfg_path, sha, kind="fork")
    r.save()
    del r  # the runner "dies"; its trainer keeps running
    time.sleep(0.5)
    assert env.run() == 0
    tr = env.state()["runs"]["R1"]["train"]
    assert len(tr["attempts"]) == 1 and tr["attempts"][0]["rc"] is None
    assert tr["status"] == "DONE" and tr["pinned"]["epoch"] == 50


def test_live_lock_blocks_second_runner(env):
    lock = env.tmp / "state" / "campaign.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    code = ("import sys, time; sys.path.insert(0, %r); import cr_common as cc\n"
            "cc.ExclusiveLock(%r).acquire(); print('READY', flush=True); time.sleep(60)\n"
            % (str(Path(cc.__file__).parent), str(lock)))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "READY"
        assert env.run() == camp_mod.EXIT_LOCKED
        assert not env.state_path.exists()
        assert p.poll() is None
    finally:
        p.kill()
        p.wait()
    assert env.run() == 0  # the OS released the dead holder's lock; no stale-file handling needed


def test_started_run_spec_change_is_refused(env):
    env.set_ctl(mode="die_early")
    assert env.run() == camp_mod.EXIT_STOPPED
    c = json.loads(env.camp_path.read_text())
    c["runs"][0]["seed"] = 5678
    env.camp_path.write_text(json.dumps(c))
    assert env.run("--clear-stop", "x") == 0
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "spec changed" in env.state()["stop_reason"]


def test_dry_run_changes_nothing(env, capsys):
    assert env.run("--dry-run") == 0
    out = capsys.readouterr().out
    assert "R1:train" in out and "dry run: 0 problem(s)" in out
    assert not (env.tmp / "state").exists() and not (Path(env.camp["worktree_root"])).exists()


def test_set_commit_refuses_started_runs(env):
    git(env.repo, "commit", "--allow-empty", "-q", "-m", "second")
    new = git(env.repo, "rev-parse", "HEAD")
    assert camp_mod.main(["--campaign", str(env.camp_path), "--set-commit", new, "--runs", "R1"]) == 0
    assert json.loads(env.camp_path.read_text())["runs"][0]["git_commit"] == new
    env.set_ctl(mode="die_early")
    env.run()
    with pytest.raises(SystemExit):
        camp_mod.main(["--campaign", str(env.camp_path), "--set-commit", env.commit, "--runs", "R1"])


def test_anchors_run_between_train_and_probe(env):
    anc = env.camp["ancestor"]
    env.rewrite(anchors={"id": "ANCHORS", "enabled": True, "git_commit": env.commit, "items": [
        {"name": "anchor_centroid_ep50", "arm": "centroid", "mode": "checkpoint",
         "checkpoint": anc["path"], "sha256": anc["sha256"]},
        {"name": "anchor_off", "arm": "random", "mode": "legacy_cache", "enabled": False}]},
        sequence=["R1:train", "ANCHORS:probe", "R1:probe"])
    assert env.run() == 0
    st = env.state()
    items = st["runs"]["ANCHORS"]["probe"]["items"]
    assert items["anchor_centroid_ep50"]["status"] == "DONE" and items["anchor_off"]["status"] == "DISABLED"
    a = items["anchor_centroid_ep50"]["attempts"][0]
    assert "anchor" in a["argv"] and "--train-seed" not in a["argv"]
    seq = [e["msg"] for e in st["events"] if "launched" in e["msg"]]
    assert seq[0].startswith("R1: launched fork") and "anchor_centroid_ep50" in seq[1] and "R1_random" in seq[2]


def test_anchor_checkpoint_hash_mismatch_stops(env):
    anc = env.camp["ancestor"]
    env.rewrite(anchors={"id": "ANCHORS", "enabled": True, "git_commit": env.commit, "items": [
        {"name": "anchor_bad", "arm": "centroid", "checkpoint": anc["path"], "sha256": "1" * 64}]},
        sequence=["ANCHORS:probe"])
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "sha256" in env.state()["stop_reason"]


def test_trainer_nonfinite_exit4_is_not_resumed(env, monkeypatch):
    # The trainer's own abort (exit 4) must stop the campaign even if the NaN line was missed.
    monkeypatch.setattr(camp_mod.gate, "check_run", lambda *a, **k: {
        "latest_epoch": None, "latest_status": "PASS", "worst_status": "PASS", "epochs": {},
        "summary": "gate mocked"})
    env.set_ctl(mode="nan_exit4", nan_epoch=28)
    assert env.run() == camp_mod.EXIT_STOPPED
    st = env.state()
    assert "exit 4" in st["stop_reason"]
    assert len(st["runs"]["R1"]["train"]["attempts"]) == 1


def test_exact_resume_must_keep_trainer_run_uuid(env):
    env.set_ctl(modes={"0": "crash", "1": "uuid_change"}, crash_epoch=29)
    assert env.run() == camp_mod.EXIT_STOPPED
    st = env.state()
    assert "run_manifest.json does not match" in st["stop_reason"] and "run_uuid" in st["stop_reason"]
    assert [a["kind"] for a in st["runs"]["R1"]["train"]["attempts"]] == ["fork", "exact"]


def test_exact_resume_must_keep_run_contract(env):
    env.set_ctl(modes={"0": "crash", "1": "contract_change"}, crash_epoch=29)
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "run_contract_sha256" in env.state()["stop_reason"]


def _random_anchor(env):
    anc = env.camp["ancestor"]
    env.rewrite(anchors={"id": "ANCHORS", "enabled": True, "git_commit": env.commit, "items": [
        {"name": "anchor_random_ep50", "arm": "random", "mode": "checkpoint", "run_uuid": "orig-random-ep50",
         "checkpoint": anc["path"], "sha256": anc["sha256"]}]},
        sequence=["R1:train", "ANCHORS:probe", "R1:probe"])


@pytest.mark.parametrize("g2_mode,status", [("ok", "PASS"), ("flag", "FLAG")])
def test_g2_against_same_arm_anchor(env, g2_mode, status):
    _random_anchor(env)
    env.set_ctl(g2_mode=g2_mode)
    assert env.run() == 0
    st = env.state()
    g2 = st["runs"]["R1"]["g2"]
    assert g2["status"] == status
    anchor_dir = st["runs"]["ANCHORS"]["probe"]["items"]["anchor_random_ep50"]["result"]["out_dir"]
    assert g2["anchor_dir"] == anchor_dir
    assert bool(alerts(env, "G2_FLAG_R1")) == (g2_mode == "flag")
    assert st["status"] == "DONE"


def test_g2_integrity_error_stops_and_reruns_after_ack(env):
    _random_anchor(env)
    env.set_ctl(g2_mode="integrity")
    assert env.run() == camp_mod.EXIT_STOPPED
    st = env.state()
    assert "G2 gate tool failed rc=2" in st["stop_reason"] and st["runs"]["R1"]["probe"]["status"] == "DONE"
    env.set_ctl(g2_mode="ok")
    assert env.run("--clear-stop", "fixed") == 0
    assert env.run() == 0
    st = env.state()
    assert st["runs"]["R1"]["g2"]["status"] == "PASS"
    assert len(st["runs"]["R1"]["probe"]["attempts"]) == 1  # probe not rerun


def test_probe_refuses_when_pinned_commit_lacks_required_files(env):
    c = json.loads(env.camp_path.read_text())
    c["probe"]["requires_repo_files"] = ["src/eval_downstream.py", "configs/cr_seed_v1/legacy_hashes.json"]
    env.camp_path.write_text(json.dumps(c))
    assert env.run("--dry-run") == 1
    assert env.run() == camp_mod.EXIT_STOPPED
    st = env.state()
    assert "lacks files the probe needs" in st["stop_reason"] and "legacy_hashes.json" in st["stop_reason"]
    assert st["runs"]["R1"]["train"]["status"] == "DONE"



# ---------------------------------------------------------------------------
# V3 verification regressions (P0-2, P1-1, P1-2, P1-3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seal_kind", ["empty", "invalid_json", "foreign_identity_and_missing_predictions"])
def test_v3_validate_probe_rejects_invalid_seal(tmp_path, seal_kind):
    camp = json.loads((Path(camp_mod.REPO) / "configs" / "cr_seed_v1" / "campaign.json").read_text())
    camp["state_dir"] = str(tmp_path / "state")
    cpath = tmp_path / "campaign.json"
    cpath.write_text(json.dumps(camp))
    r = camp_mod.Runner(cpath, out=lambda *_: None)
    r.load_state()
    out = tmp_path / "probe"
    out.mkdir()
    ck = tmp_path / "epoch50.pth.tar"
    ck.write_bytes(b"isolated fake checkpoint")
    sha = cc.sha256_file(ck)
    ident = {"checkpoint_sha256": sha, "run_uuid": "the-training-run", "arm": "random", "train_seed": 1234,
             "role": "new", "epoch": 50, "run_provenance_run_uuid": "the-training-run",
             "run_provenance_checkpoint_role": "periodic"}
    (out / "results.json").write_text(json.dumps({"schema": "cr_probe_v1", "status": "complete", "sealed": True,
                                                  "best_val_auc": 0.85, "identity": ident, "test_auc": None,
                                                  "legacy_cache_integrity": None,
                                                  "cache_provenance": {sp: {"primary": {"state": "cold"}} for sp in
                                                                       ("Training", "Validation", "Test")}}))
    seal = {"empty": "", "invalid_json": "{"}.get(seal_kind) or json.dumps({
        "schema": "not_cr_sealed_v1", "metrics_computed": True, "identity": {"run_uuid": "foreign-run"},
        "files": [{"path": "missing_test_predictions.npz", "sha256": "0" * 64}]})
    (out / "sealed_manifest.json").write_text(seal)
    (out / "sealed_manifest.json.sha256").write_text(cc.sha256_file(out / "sealed_manifest.json") + "  x\n")
    target = {"name": "R1", "checkpoint": str(ck), "mode": "checkpoint", "role": "new", "arm": "random",
              "run_uuid": "the-training-run",
              "expect": {"identity.%s" % k: ident[k] for k in
                         ("run_uuid", "train_seed", "role", "epoch", "run_provenance_checkpoint_role")}}
    a = {"rc": 0, "out_dir": str(out), "started_at": time.time() - 2, "log": "fake"}
    with pytest.raises(camp_mod.CampaignStop, match="invalid test seal"):
        r.validate_probe("R1", a, target, sha)


def test_v3_manifest_mismatch_during_epoch50_recovery_stops(env):
    env.set_ctl(modes={"0": "crash_after_save", "1": "bad_manifest"})
    assert env.run() == camp_mod.EXIT_STOPPED
    st = env.state()
    assert "amp.amp_target" in st["stop_reason"]
    a1 = st["runs"]["R1"]["train"]["attempts"][1]
    assert a1["kind"] == "exact" and a1["manifest_checked"]["ok"] is False
    assert "pinned" not in st["runs"]["R1"]["train"]
    assert not (env.run_dir / "pinned").exists()


def test_v3_resume_config_crash_window_recovers(env, monkeypatch):
    env.set_ctl(modes={"0": "crash"}, crash_epoch=29)
    original = camp_mod.Runner.preflight

    def interrupted(self, what, **kw):
        if "exact resume" in what:
            raise KeyboardInterrupt()
        return original(self, what, **kw)
    monkeypatch.setattr(camp_mod.Runner, "preflight", interrupted)
    assert env.run() == 130
    cfg = env.tmp / "state" / "configs" / "R1_resume_a1.yaml"
    assert cfg.exists()
    monkeypatch.setattr(camp_mod.Runner, "preflight", original)
    assert env.run() == 0
    tr = env.state()["runs"]["R1"]["train"]
    assert [a["kind"] for a in tr["attempts"]] == ["fork", "exact"]
    assert tr["attempts"][1]["config"] == str(cfg) and tr["pinned"]["epoch"] == 50


def test_v3_differing_existing_resume_config_still_stops(env, monkeypatch):
    env.set_ctl(modes={"0": "crash"}, crash_epoch=29)
    original = camp_mod.Runner.preflight

    def interrupted(self, what, **kw):
        if "exact resume" in what:
            raise KeyboardInterrupt()
        return original(self, what, **kw)
    monkeypatch.setattr(camp_mod.Runner, "preflight", interrupted)
    assert env.run() == 130
    cfg = env.tmp / "state" / "configs" / "R1_resume_a1.yaml"
    d = yaml.safe_load(cfg.read_text())
    d["meta"]["seed"] = 999
    cfg.write_text(yaml.safe_dump(d))
    monkeypatch.setattr(camp_mod.Runner, "preflight", original)
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "different content" in env.state()["stop_reason"]


def test_v3_probe_done_journal_window_recovers_without_relaunch(env):
    assert env.run() == 0
    st = env.state()
    rs = st["runs"]["R1"]
    assert rs["probe"]["attempts"][-1]["status"] == "DONE"
    rs["probe"]["status"] = "PENDING"
    rs["probe"].pop("result")
    rs.pop("g2", None)
    st["status"] = "RUNNING"
    env.state_path.write_text(json.dumps(st))
    assert env.run() == 0
    pr = env.state()["runs"]["R1"]["probe"]
    assert pr["status"] == "DONE" and len(pr["attempts"]) == 1
    assert pr["result"]["out_dir"].endswith("_a0") and pr["result"]["seal"]["pairs"] == 2


def test_v3_anchor_done_journal_window_recovers_without_relaunch(env):
    _random_anchor(env)
    assert env.run() == 0
    st = env.state()
    item = st["runs"]["ANCHORS"]["probe"]["items"]["anchor_random_ep50"]
    item["status"] = "PENDING"
    item.pop("result")
    st["runs"]["ANCHORS"]["probe"]["status"] = "PENDING"
    st["status"] = "RUNNING"
    env.state_path.write_text(json.dumps(st))
    assert env.run() == 0
    item = env.state()["runs"]["ANCHORS"]["probe"]["items"]["anchor_random_ep50"]
    assert item["status"] == "DONE" and len(item["attempts"]) == 1



TOFU = ("tofu_pinned_2026-10-08; historical integrity supported by timestamps and head replay, "
        "not cryptographic")


def _legacy_anchor(env, **extra):
    probe_cmd = env.camp["probe"]["command"]
    item = {"name": "anchor_random_legacy", "arm": "random", "mode": "legacy_cache", "run_uuid": "orig-random-ep50",
            "checkpoint": str(env.tmp / "not_on_disk.pth.tar"), "expected_identity": "ab" * 32,
            "identity_field": "identity.checkpoint_sha256", "cache_states": ["legacy_verified"],
            "legacy_cache_integrity": TOFU,
            "command": probe_cmd[:14] + ["--legacy-cache-policy", "allow_verified", "--expected-checkpoint-sha256",
                                         "{expected_sha256}", "--arm", "random", "--role", "anchor",
                                         "--run-uuid", "{run_uuid}"]}
    item.update(extra)
    env.rewrite(anchors={"id": "ANCHORS", "enabled": True, "git_commit": env.commit, "items": [item]},
                sequence=["ANCHORS:probe"])


def test_legacy_anchor_requires_tofu_and_legacy_verified_caches(env):
    _legacy_anchor(env)
    assert env.run() == 0
    item = env.state()["runs"]["ANCHORS"]["probe"]["items"]["anchor_random_legacy"]
    assert item["status"] == "DONE" and item["result"]["identity"] == "ab" * 32


def test_legacy_anchor_wrong_integrity_marker_stops(env):
    _legacy_anchor(env, legacy_cache_integrity="something else")
    assert env.run() == camp_mod.EXIT_STOPPED
    assert "legacy_cache_integrity" in env.state()["stop_reason"]
