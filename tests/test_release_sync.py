import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autopilot import release_assets as assets
from autopilot import verify_punctuation_only as punctuation
from scripts import sync_overleaf as sync


def test_remote_manual_edit_and_delete_remain_conflicts(tmp_path):
    local = tmp_path / "main.tex"
    local.write_text("local")
    items = [(str(local), "main.tex")]
    state = {"files": {"main.tex": {"local_sha256": "baseline", "remote_sha256": "baseline"}}}
    assert sync.classify(items, {"main.tex": "remote-edit"}, state)[0]["status"] == sync.CONFLICT_EDITED
    assert sync.classify(items, {}, state)[0]["status"] == sync.CONFLICT_DELETED
    assert sync.classify(items, {"main.tex": "unknown"}, {"files": {}})[0]["status"] == sync.CONFLICT_UNKNOWN


def test_remote_tip_must_be_fresh(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="newtip\trefs/heads/master\n", returncode=0)
    monkeypatch.setattr(sync, "run", run)
    with pytest.raises(ValueError, match="moved"):
        sync.verify_remote_tip("unused", "master", "reviewedtip")
    assert "ls-remote" in calls[0]
    assert "--force" not in calls[0]


def test_staged_git_filters_cannot_change_validated_bytes(tmp_path, monkeypatch):
    source = tmp_path / "file.tex"
    source.write_bytes(b"validated\r\nsource")
    monkeypatch.setattr(sync.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=b"validated\nsource"))
    sync.verify_staged_bytes(tmp_path, [(str(source), "main.tex")])
    monkeypatch.setattr(sync.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=b"changed by filter"))
    with pytest.raises(ValueError, match="staged bytes"):
        sync.verify_staged_bytes(tmp_path, [(str(source), "main.tex")])


def test_false_or_empty_release_manifest_cannot_sync(tmp_path):
    for manifest in ({"ALL_PASS": False}, {"ALL_PASS": True, "checks": {}},
                     {"ALL_PASS": True, "checks": {"some_gate": True}}):
        with pytest.raises(ValueError):
            sync.verify_local_release(tmp_path, manifest)


def test_punctuation_checker_requires_real_expected_change():
    assert punctuation.verify("same", "same")
    assert not punctuation.verify("a; next", "a. Next")
    assert punctuation.verify("a; next 1", "a. Next 2")
    assert punctuation.verify("a; next", "A. Next")
    assert punctuation.verify("a; next", "a. Next\nextra")


def test_authentication_is_scoped_to_environment_not_argv_or_config(tmp_path, monkeypatch, capsys):
    import base64
    token = "SYNTHETIC_TEST_CREDENTIAL_NOT_USABLE"
    encoded = base64.b64encode(("git:" + token).encode()).decode()
    monkeypatch.setenv("OVERLEAF_TOKEN", token)
    monkeypatch.setenv("OVERLEAF_PROJECT_ID", "syntheticproject")
    monkeypatch.setenv("GIT_TRACE_CURL", "1")
    monkeypatch.setenv("GIT_TRACE_REDACT", "0")
    monkeypatch.setattr(sync, "_SECRETS", [])
    paper = tmp_path / "paper"
    paper.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(sync.assets, "unique_work", lambda *args, **kwargs: work)
    calls, configs = [], []

    def fake_git(command, **kwargs):
        assert all(token not in str(argument) and encoded not in str(argument) for argument in command)
        calls.append((command, kwargs["env"]))
        if "clone" in command:
            clone = Path(command[-1])
            (clone / ".git").mkdir(parents=True)
            config = clone / ".git" / "config"
            config.write_text('[remote "origin"]\n\turl = ' + command[-2] + "\n")
            configs.append(config)
            # This is the persisted state even if termination happens now,
            # before any cleanup is possible.
            assert token not in config.read_text() and encoded not in config.read_text()
        output = "tip\n" if "rev-parse" in command else "master\n" if "symbolic-ref" in command else ""
        return SimpleNamespace(returncode=0, stdout=output, stderr="")
    monkeypatch.setattr(sync.subprocess, "run", fake_git)
    monkeypatch.setattr(sync, "rmtree_force", lambda path: (_ for _ in ()).throw(OSError("locked checkout")))
    with pytest.raises(RuntimeError, match="sanitized residual"):
        sync.main(["--check", "--paper-dir", str(paper), "--state", str(tmp_path / "state.json")])
    clone_command, auth = next(item for item in calls if "clone" in item[0])
    assert clone_command[-2] == "https://git.overleaf.com/syntheticproject"
    settings = {auth["GIT_CONFIG_KEY_%d" % i]: auth["GIT_CONFIG_VALUE_%d" % i]
                for i in range(int(auth["GIT_CONFIG_COUNT"]))}
    assert settings["credential.helper"] == ""
    assert settings["http.extraHeader"] == ""
    assert settings["http.followRedirects"] == "false"
    assert settings["http.https://git.overleaf.com/syntheticproject.extraHeader"] == "Authorization: Basic " + encoded
    assert "OVERLEAF_TOKEN" not in auth and "GIT_TRACE_CURL" not in auth
    assert auth["GIT_TRACE_REDACT"] == "1"
    for command, environment in calls:
        if "clone" not in command:
            assert encoded not in environment.values() and "OVERLEAF_TOKEN" not in environment
    assert configs[0].exists()
    assert token not in configs[0].read_text() and encoded not in configs[0].read_text()
    output = capsys.readouterr().out
    assert token not in output and encoded not in output


def test_cleanup_failure_sanitizes_legacy_config_without_echoing(tmp_path, monkeypatch):
    token = "SYNTHETIC_LEGACY_TEST_ONLY"
    monkeypatch.setattr(sync, "_SECRETS", [token])
    config = tmp_path / ".git" / "config"
    config.parent.mkdir()
    config.write_text('[remote "origin"]\nurl=https://git:' + token + '@git.overleaf.com/project\n')
    reflog = config.parent / "logs" / "HEAD"
    reflog.parent.mkdir()
    reflog.write_text("clone: from https://git:" + token + "@git.overleaf.com/project\n")
    fetched = config.parent / "FETCH_HEAD"
    fetched.write_text("from https://git:" + token + "@git.overleaf.com/project\n")
    monkeypatch.setattr(sync, "rmtree_force", lambda path: (_ for _ in ()).throw(OSError("cleanup failed")))
    with pytest.raises(RuntimeError, match="sanitized residual") as failure:
        sync.cleanup_clone(tmp_path)
    assert token not in str(failure.value)
    assert token not in config.read_text() and "@" not in config.read_text()
    assert token not in reflog.read_text() and token not in fetched.read_text()


def test_auth_output_redacts_plain_and_encoded_forms(monkeypatch, capsys):
    import base64
    token = "SYNTHETIC_OUTPUT_TEST_ONLY"
    encoded = base64.b64encode(("git:" + token).encode()).decode()
    monkeypatch.setattr(sync, "_SECRETS", [])
    env = sync.authentication_environment(token, "https://git.overleaf.com/project")
    monkeypatch.setattr(sync.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout=token, stderr=encoded))
    with pytest.raises(SystemExit) as failure:
        sync.run(["git", "ls-remote", "https://git.overleaf.com/project"], env=env)
    output = capsys.readouterr().out + str(failure.value)
    assert token not in output and encoded not in output


def test_token_cli_argument_is_rejected_without_echo(capsys):
    assert sync.main(["--token", "SYNTHETIC_ARGUMENT_TEST_ONLY"]) == 1
    assert "SYNTHETIC_ARGUMENT_TEST_ONLY" not in capsys.readouterr().out


def test_release_work_and_recovery_directories_are_ignored():
    import subprocess
    for path in (r"_release_work\example\clone\.git\config",
                 r"_release_tests\example\artifact.docx",
                 r"paper\genai4health2026\.release-recovery-example\main_submission.docx.previous"):
        result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", path],
                                cwd=assets.REPO, capture_output=True, env=sync.child_environment())
        assert result.returncode == 0, path


def test_real_local_clone_does_not_persist_environment_authentication(tmp_path, monkeypatch):
    import base64
    import subprocess
    token = "SYNTHETIC_LOCAL_CLONE_TEST_ONLY"
    encoded = base64.b64encode(("git:" + token).encode()).decode()
    monkeypatch.setattr(sync, "_SECRETS", [])
    env = sync.authentication_environment(token, "https://git.overleaf.com/project")
    source, clone = tmp_path / "empty-source", tmp_path / "clone"
    subprocess.run(["git", "init", "--quiet", str(source)], check=True,
                   capture_output=True, env=sync.child_environment())
    subprocess.run(["git", "-c", "protocol.allow=never", "-c", "protocol.file.allow=always",
                    "clone", "--quiet", str(source), str(clone)], check=True,
                   capture_output=True, env=env)
    # Assert before cleanup: hard termination cannot expose a stored token.
    config = (clone / ".git" / "config").read_text()
    clean = token not in config and encoded not in config and "Authorization" not in config
    assert clean
    sync.cleanup_clone(clone)
