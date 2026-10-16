"""HTTP (browser-session) Overleaf transport: plan, push, refusal and guards.

No network. FakeOverleaf answers the same endpoints the real site does
(verified on a scratch project), so the real OverleafHTTP client code runs
end to end against it.
"""
import hashlib
import io
import json as jsonlib
import pickle
import re
import zipfile
from types import SimpleNamespace

import pytest
import requests

from scripts import overleaf_http as olh
from scripts import sync_overleaf as sync

REAL_CLIENT = olh.OverleafHTTP

PID = "0123456789abcdef01234567"
SECRET = "SYNTHETIC_SESSION_COOKIE_NOT_REAL"
STORED_CSRF = "SYNTHETIC_STORED_CSRF_NOT_REAL"
FRESH_CSRF = "SYNTHETIC_FRESH_CSRF_NOT_REAL"
SELF_ID = "5f00000000000000000000aa"
OTHER_ID = "5f00000000000000000000bb"
PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                    "0000000d49444154789c6360000002000154a24f5f0000000049454e44ae426082")


class Resp:
    def __init__(self, status, body=b"", ctype="application/json"):
        self.status_code = status
        self.content = body if isinstance(body, bytes) else body.encode("utf-8")
        self.headers = {"Content-Type": ctype}

    @property
    def text(self):
        return self.content.decode("utf-8")

    def json(self):
        return jsonlib.loads(self.text)


class FakeOverleaf:
    """In-memory Overleaf project speaking the endpoints overleaf_http uses.

    Every content change is also appended to a project history (one update per
    change, attributed to a user id), served by /updates like the real site.
    """
    TEXT = (".tex", ".bib", ".sty", ".txt")

    def __init__(self, project_id, files, permissions="owner"):
        self.project_id = project_id
        self.permissions = permissions
        self.headers = {}
        self.cookies = requests.cookies.RequestsCookieJar()
        self.counter = 0
        self.version, self.updates, self.clock = 0, [], 1_700_000_000_000
        self.record_history, self.history_errors = True, []
        self.hide_versions, self.drop_newest, self.hold_clock = set(), False, 0
        self.page_user, self.refresh_cookie = SELF_ID, None
        self.tree_hook = self.upload_hook = None
        self.respond = {}
        self.folders = {"": self._id()}
        self.entities = {}
        for rel, data in files.items():
            self._put(rel, data)
        self.writes, self.log = [], []
        self.zip_hook = self.mangle = None
        self.override, self.queue = {}, {}

    def _id(self):
        self.counter += 1
        return "%024x" % (0xABC000 + self.counter)

    def _put(self, rel, data, user=SELF_ID):
        parts = rel.split("/")[:-1]
        for i in range(len(parts)):
            self.folders.setdefault("/".join(parts[:i + 1]), self._id())
        kind = "doc" if rel.endswith(self.TEXT) else "file"
        if kind == "doc":
            data = data.replace(b"\r\n", b"\n")
        old = self.entities.get(rel)
        eid = old[1] if old and old[0] == kind == "doc" else self._id()
        self.entities[rel] = [kind, eid, data]
        self._record(user, [rel] if kind == "doc" else [], [] if kind == "doc" else [{"add": {"pathname": rel}}])
        return kind, eid

    def _record(self, user, pathnames, project_ops):
        if not self.record_history:
            return
        if self.hold_clock:
            self.hold_clock -= 1
        else:
            self.clock += 1000
        self.updates.append({"fromV": self.version, "toV": self.version + 1,
                             "meta": {"users": [{"id": user}] if user else [],
                                      "start_ts": self.clock, "end_ts": self.clock},
                             "labels": [], "pathnames": pathnames, "project_ops": project_ops})
        self.version += 1

    def edit_as(self, rel, data, user=OTHER_ID):
        """A co-author's change through the editor, attributed in history."""
        return self._put(rel, data, user)

    @property
    def files(self):
        return {rel: entry[2] for rel, entry in self.entities.items()}

    def zip_bytes(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            for rel, data in sorted(self.files.items()):
                archive.writestr(rel, data)
        return buf.getvalue()

    def tree(self, path=""):
        node = {"_id": self.folders[path], "name": path.rpartition("/")[2] or "rootFolder",
                "docs": [], "fileRefs": [], "folders": []}
        for rel, (kind, eid, _data) in sorted(self.entities.items()):
            if rel.rpartition("/")[0] == path:
                node["docs" if kind == "doc" else "fileRefs"].append(
                    {"_id": eid, "name": rel.rpartition("/")[2]})
        for sub in sorted(self.folders):
            if sub and sub.rpartition("/")[0] == path:
                node["folders"].append(self.tree(sub))
        return node

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def request(self, method, url, params=None, data=None, files=None, json=None,
                headers=None, timeout=None, allow_redirects=True):
        assert url.startswith("https://www.overleaf.com/") and allow_redirects is False
        path = url[len("https://www.overleaf.com"):]
        self.log.append((method, path))
        headers = headers or {}
        project = "/project/" + self.project_id
        if (method, path) in self.respond:
            return self.respond[(method, path)](self)
        if (method, path) in self.override:
            return Resp(self.override[(method, path)], b"Forbidden", "text/plain")
        if self.queue.get((method, path)):
            return Resp(self.queue[(method, path)].pop(0), b"Too Many Requests", "text/plain")
        if method == "GET" and path == project:
            user = '<meta name="ol-user_id" content="%s">' % self.page_user if self.page_user else ""
            page = ('<html><head><meta name="ol-project_id" content="%s">'
                    '<meta name="ol-csrfToken" content="%s">%s</head></html>' % (self.project_id, FRESH_CSRF, user))
            if self.refresh_cookie:
                self.cookies.set("overleaf_session2", self.refresh_cookie, domain=".overleaf.com", path="/")
            return Resp(200, page, "text/html; charset=utf-8")
        if method == "GET" and path == project + "/download/zip":
            if self.zip_hook:
                self.zip_hook(self)
            return Resp(200, self.zip_bytes(), "application/zip")
        if method == "GET" and path == project + "/updates":
            status = self.history_errors.pop(0) if self.history_errors else None
            if isinstance(status, bytes):
                return Resp(200, status)
            if status:
                return Resp(status, jsonlib.dumps({"error": "history unavailable"}))
            # The documented cursor model: a page ends at nextBeforeTimestamp
            # and the next one returns rows with end_ts strictly before it.
            newest = [u for u in reversed(self.updates) if u["toV"] not in self.hide_versions]
            if self.drop_newest:
                newest = newest[1:]
            if "before" in params:
                newest = [u for u in newest if u["meta"]["end_ts"] < int(params["before"])]
            count = int(params["min_count"])
            body = {"updates": newest[:count]}
            if len(newest) > count:
                body["nextBeforeTimestamp"] = newest[count - 1]["meta"]["start_ts"]
            return Resp(200, jsonlib.dumps(body))
        if method == "GET" and path == project + "/latest/history":
            start = max(0, self.version - 3)
            return Resp(200, jsonlib.dumps({"chunk": {"startVersion": start, "history": {
                "snapshot": {"files": {}}, "changes": [{"operations": []}] * (self.version - start)}}}))
        if method == "GET" and path == "/socket.io/1/":
            assert params["projectId"] == self.project_id
            if self.tree_hook:
                self.tree_hook(self)
            return Resp(200, "fakesid:60:60:websocket,xhr-polling", "text/plain")
        if path == "/socket.io/1/xhr-polling/fakesid":
            if method == "POST":
                return Resp(200, "1", "text/plain")
            event = jsonlib.dumps({"name": "joinProjectResponse", "args": [{
                "project": {"_id": self.project_id, "rootFolder": [self.tree()]},
                "permissionsLevel": self.permissions, "protocolVersion": 2}]})
            body = "".join("\ufffd%d\ufffd%s" % (len(m), m) for m in ("1::", "5:::" + event))
            return Resp(200, body, "text/plain")
        if (headers.get("X-Csrf-Token") or headers.get("X-CSRF-TOKEN")) != FRESH_CSRF:
            return Resp(403, b"Forbidden", "text/plain")
        by_id = {value: key for key, value in self.folders.items()}
        if method == "POST" and path == project + "/folder":
            parent = by_id[json["parent_folder_id"]]
            new = (parent + "/" if parent else "") + json["name"]
            if new in self.folders or new in self.entities:
                return Resp(400, b"file already exists", "text/plain")
            self.folders[new] = self._id()
            self.writes.append(("folder", new))
            return Resp(200, jsonlib.dumps({"name": json["name"], "_id": self.folders[new],
                                            "docs": [], "fileRefs": [], "folders": []}))
        if method == "POST" and path == project + "/upload":
            if self.upload_hook:
                hook, self.upload_hook = self.upload_hook, None
                hook(self)
            parent = by_id[params["folder_id"]]
            rel = (parent + "/" if parent else "") + data["name"]
            content = files["qqfile"][1]
            if self.mangle:
                content = self.mangle(rel, content)
            kind, eid = self._put(rel, content)
            self.writes.append(("upload", rel, content))
            return Resp(200, jsonlib.dumps({"success": True, "entity_id": eid, "entity_type": kind}))
        match = re.fullmatch(re.escape(project) + r"/(doc|file)/([0-9a-f]{24})", path)
        if method == "DELETE" and match:
            for rel, (kind, eid, _data) in list(self.entities.items()):
                if (kind, eid) == match.groups():
                    del self.entities[rel]
                    self.writes.append(("delete", rel))
                    self._record(SELF_ID, [], [{"remove": {"pathname": rel}}])
                    return Resp(204, b"", "text/plain")
            return Resp(404, b"", "text/html")
        raise AssertionError("unexpected request %s %s" % (method, path))


def lf(data):
    return data.replace(b"\r\n", b"\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


LOCAL = {
    "main_submission.tex": b"\\documentclass{article}\r\n\\input{auto/numbers}\r\nbody\r\n",
    "references.bib": b"@article{x,title={T}}\n",
    "auto/numbers.tex": b"\\newcommand{\\N}{1}\n",
    "figures/fig.png": PNG,
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    paper = tmp_path / "paper"
    for rel, data in LOCAL.items():
        (paper / rel).parent.mkdir(parents=True, exist_ok=True)
        (paper / rel).write_bytes(data)
    remote = {"main.tex": lf(LOCAL["main_submission.tex"]), "references.bib": LOCAL["references.bib"],
              "auto/numbers.tex": LOCAL["auto/numbers.tex"], "figures/fig.png": PNG,
              "figures/withdrawn.png": b"withdrawn figure bytes",
              "README_OVERLEAF.txt": b"remote-only, never ours"}
    agreed = {rel: {"remote_sha256": sha(data), "local_sha256": sha(data)}
              for rel, data in remote.items() if rel != "README_OVERLEAF.txt"}
    state_path = tmp_path / "state.json"
    state_path.write_text(jsonlib.dumps({"version": 1, "project": sync.project_fingerprint(PID),
                                         "files": agreed}), encoding="utf-8")
    fake = FakeOverleaf(PID, remote)
    install(monkeypatch, fake, tmp_path)
    w = SimpleNamespace(paper=paper, state=state_path, fake=fake, tmp=tmp_path)

    def run(*args, project=PID):
        return sync.main(["--transport", "http", "--project-id", project, "--paper-dir", str(paper),
                          "--state", str(state_path), *args])
    w.run = run
    return w


def install(monkeypatch, fake, tmp_path):
    counter = iter(range(10 ** 6))

    def unique_work(*args, **kwargs):
        path = tmp_path / ("work-%d" % next(counter))
        path.mkdir()
        return path
    monkeypatch.setattr(sync, "_SECRETS", [])
    monkeypatch.setattr(sync.assets, "unique_work", unique_work)
    monkeypatch.setattr(olh, "load_session", lambda path=None: ({"overleaf_session2": SECRET}, STORED_CSRF))
    monkeypatch.setattr(olh, "OverleafHTTP", lambda *a, **k: REAL_CLIENT(*a, session=fake, sleep=lambda s: None, **k))


def release(monkeypatch):
    monkeypatch.setattr(sync, "validate", lambda paper, manifest, work: (sync.collect(paper), {"ALL_PASS": True}))
    monkeypatch.setattr(sync, "verify_local_release", lambda paper, manifest: None)


def no_secrets(output):
    return all(value not in output for value in (SECRET, STORED_CSRF, FRESH_CSRF))


def test_dry_run_reports_uploads_and_deletes_without_writing(world, capsys):
    (world.paper / "auto" / "numbers.tex").write_bytes(b"\\newcommand{\\N}{2}\r\n")
    before = world.state.read_bytes()
    assert world.run("--dry-run") == 0
    out = capsys.readouterr().out
    assert "would update in Overleaf (1):\n   auto/numbers.tex" in out
    assert "would delete from Overleaf (removed locally since last sync) (1):\n   figures/withdrawn.png" in out
    assert "1 file(s) would be written to Overleaf." in out
    assert "1 file(s) would be deleted from Overleaf." in out
    assert "CONFLICT" not in out
    assert world.fake.writes == [] and world.state.read_bytes() == before
    assert not any(path.endswith("/socket.io/1/") for _m, path in world.fake.log)
    assert no_secrets(out)


def test_push_uploads_nested_binary_crlf_deletes_and_verifies(world, monkeypatch, capsys):
    release(monkeypatch)
    main = b"\\documentclass{article}\r\n\\input{auto/numbers}\r\nnew body\r\n"
    (world.paper / "main_submission.tex").write_bytes(main)
    (world.paper / "figures" / "sub").mkdir()
    (world.paper / "figures" / "sub" / "new.png").write_bytes(PNG + b"nested")
    (world.paper / "auto" / "extra.tex").write_bytes(b"crlf\r\nfile\r\n")
    assert world.run("--release-manifest", "unused.json") == 0
    out = capsys.readouterr().out
    files = world.fake.files
    assert files["main.tex"] == lf(main)
    assert files["figures/sub/new.png"] == PNG + b"nested"
    assert files["auto/extra.tex"] == b"crlf\nfile\n"
    assert "figures/withdrawn.png" not in files
    assert files["README_OVERLEAF.txt"] == b"remote-only, never ours"
    uploads = {w[1]: w[2] for w in world.fake.writes if w[0] == "upload"}
    assert set(uploads) == {"main.tex", "figures/sub/new.png", "auto/extra.tex"}
    assert b"\r\n" not in uploads["main.tex"]
    assert ("folder", "figures/sub") in world.fake.writes
    assert ("delete", "figures/withdrawn.png") in world.fake.writes
    assert "pushed and verified" in out and no_secrets(out)
    state = jsonlib.loads(world.state.read_text(encoding="utf-8"))
    assert state["last_action"] == "push"
    assert state["files"]["main.tex"]["remote_sha256"] == sha(lf(main))
    assert state["files"]["figures/withdrawn.png"] == {"remote_sha256": None, "local_sha256": None}
    capsys.readouterr()
    # The recorded agreement is coherent: an immediate dry run is a no-op.
    assert world.run("--dry-run") == 0
    again = capsys.readouterr().out
    assert "0 file(s) would be written" in again and "0 file(s) would be deleted" in again


def test_remote_edit_refuses_push_with_diff_and_writes_nothing(world, monkeypatch, capsys):
    release(monkeypatch)
    world.fake._put("main.tex", lf(LOCAL["main_submission.tex"]) + b"collaborator sentence\n")
    (world.paper / "main_submission.tex").write_bytes(LOCAL["main_submission.tex"] + b"our edit\r\n")
    before = world.state.read_bytes()
    assert world.run("--release-manifest", "unused.json") == 2
    out = capsys.readouterr().out
    assert "CONFLICT - edited in Overleaf since last sync (1):\n   main.tex" in out
    assert "--- overleaf/main.tex" in out and "-collaborator sentence" in out and "+our edit" in out
    assert "REFUSING TO PUSH" in out
    assert world.fake.writes == [] and world.state.read_bytes() == before


def test_remote_edit_of_locally_removed_file_blocks_deletion(world, monkeypatch, capsys):
    release(monkeypatch)
    world.fake._put("figures/withdrawn.png", b"replaced by a co-author")
    assert world.run("--release-manifest", "unused.json") == 2
    out = capsys.readouterr().out
    assert "CONFLICT - edited in Overleaf since last sync, but removed locally (1):" in out
    assert world.fake.writes == []


def test_file_left_out_of_release_but_present_locally_is_retained(world, monkeypatch, capsys):
    monkeypatch.setattr(sync, "validate", lambda paper, manifest, work: (
        [item for item in sync.collect(paper) if item[1] != "figures/fig.png"], {"ALL_PASS": True}))
    monkeypatch.setattr(sync, "verify_local_release", lambda paper, manifest: None)
    (world.paper / "figures" / "withdrawn.png").write_bytes(b"withdrawn figure bytes")
    assert world.run("--release-manifest", "unused.json") == 0
    assert "figures/fig.png" in world.fake.files and "figures/withdrawn.png" in world.fake.files
    assert not [w for w in world.fake.writes if w[0] == "delete"]
    state = jsonlib.loads(world.state.read_text(encoding="utf-8"))
    assert state["files"]["figures/fig.png"]["remote_sha256"] == sha(PNG)


def test_verification_mismatch_does_not_record_state(world, monkeypatch, capsys):
    release(monkeypatch)
    (world.paper / "auto" / "numbers.tex").write_bytes(b"\\newcommand{\\N}{3}\n")
    world.fake.mangle = lambda rel, data: data + b"% server-side change\n" if rel == "auto/numbers.tex" else data
    before = world.state.read_bytes()
    assert world.run("--release-manifest", "unused.json") == 1
    out = capsys.readouterr().out
    assert "VERIFY FAILED" in out and "auto/numbers.tex" in out
    assert world.state.read_bytes() == before


def test_remote_moving_between_review_and_push_refuses(world, monkeypatch, capsys):
    release(monkeypatch)
    (world.paper / "auto" / "numbers.tex").write_bytes(b"\\newcommand{\\N}{4}\n")
    downloads = []

    def hook(fake):
        downloads.append(1)
        if len(downloads) == 2:
            fake._put("references.bib", b"@article{x,title={Edited mid-sync}}\n")
    world.fake.zip_hook = hook
    assert world.run("--release-manifest", "unused.json") == 1
    assert "moved since conflict review" in capsys.readouterr().out
    assert world.fake.writes == []


def test_pull_over_http_uses_the_shared_pull(world, capsys):
    edited = lf(LOCAL["main_submission.tex"]) + b"co-author line\n"
    world.fake._put("main.tex", edited)
    assert world.run("--pull") == 3
    assert "main.tex  ->  main_submission.tex" in capsys.readouterr().out
    assert world.run("--pull", "--yes") == 0
    assert (world.paper / "main_submission.tex").read_bytes() == edited
    assert jsonlib.loads(world.state.read_text(encoding="utf-8"))["last_action"] == "pull"
    assert world.fake.writes == []


# ------------------------------------------------- concurrent co-author edits (V22 P1)

def ours_changed(world):
    main = LOCAL["main_submission.tex"] + b"our camera-ready sentence\r\n"
    (world.paper / "main_submission.tex").write_bytes(main)
    return lf(main)


def test_edit_injected_when_file_tree_starts_refuses_before_any_write(world, monkeypatch, capsys):
    """V22's reproduction: the co-author edit lands right after the old freshness check."""
    release(monkeypatch)
    ours_changed(world)
    world.fake.tree_hook = lambda fake: fake.edit_as("main.tex", b"co-author edit during push\n")
    before = world.state.read_bytes()
    assert world.run("--release-manifest", "unused.json") == 1
    assert "moved since conflict review" in capsys.readouterr().out
    assert world.fake.writes == [] and world.state.read_bytes() == before
    assert world.fake.files["main.tex"] == b"co-author edit during push\n"


@pytest.mark.parametrize("victim", ["main.tex", "references.bib"])
def test_edit_injected_after_pre_write_check_is_flagged_and_state_not_written(world, monkeypatch, capsys, victim):
    """No compare-and-swap: the edit may be overwritten, but the push is not certified."""
    release(monkeypatch)
    ours = ours_changed(world)
    world.fake.upload_hook = lambda fake: fake.edit_as(victim, b"co-author edit after the check\n")
    before = world.state.read_bytes()
    assert world.run("--release-manifest", "unused.json") == 4
    out = capsys.readouterr().out
    assert "PUSH NOT CERTIFIED" in out and "another or unattributed user, files: %s" % victim in out
    assert "Overleaf History keeps every earlier version" in out
    assert "Sync state NOT updated" in out and "pushed and verified" not in out
    assert world.state.read_bytes() == before
    if victim == "main.tex":
        assert world.fake.files["main.tex"] == ours  # overwritten, which is why it must be flagged


@pytest.mark.parametrize("breakage, message", [
    ("unreadable", "HTTP 500"),
    ("silent", "shows no update for this push"),
    ("no_user_entry", "another or unattributed user"),
    ("interior_gap", "has a gap between"),
    ("stale_newest", "but the project is at"),
    ("malformed_json", "malformed Overleaf response during updates_since (JSONDecodeError)"),
    ("inverted_row", "unexpected project-history response"),
])
def test_history_that_cannot_certify_the_push_blocks_state(world, monkeypatch, capsys, breakage, message):
    release(monkeypatch)
    ours_changed(world)
    hooks = {
        "unreadable": lambda: setattr(world.fake, "history_errors", [None, 500]),
        "malformed_json": lambda: setattr(world.fake, "history_errors", [None, b'{"updates": [ not json']),
        "silent": lambda: setattr(world.fake, "upload_hook", lambda fake: setattr(fake, "record_history", False)),
        "no_user_entry": lambda: setattr(world.fake, "upload_hook", lambda fake: fake._record(None, ["main.tex"], [])),
        # the row of our own first upload goes missing from every page
        "interior_gap": lambda: setattr(world.fake, "upload_hook",
                                        lambda fake: fake.hide_versions.add(fake.version + 1)),
        "stale_newest": lambda: setattr(world.fake, "drop_newest", True),
        "inverted_row": lambda: setattr(world.fake, "upload_hook", lambda fake: fake.updates.append(
            {"fromV": 99, "toV": 3, "meta": {"users": [{"id": SELF_ID}]}, "pathnames": []})),
    }
    hooks[breakage]()
    saves = []
    monkeypatch.setattr(sync, "save_state", lambda *a, **k: saves.append(a))
    assert world.run("--release-manifest", "unused.json") == 4
    out = capsys.readouterr().out
    assert "PUSH NOT CERTIFIED" in out and message in out and saves == []
    assert "Overleaf History keeps every earlier version" in out


def test_v22_page_boundary_tie_with_default_page_size_is_caught(world, monkeypatch, capsys):
    """V22's reproduction: 50 uploads, default page size, and a foreign edit sharing the
    first upload's timestamp, so a strict before-cursor skips it at the page boundary."""
    release(monkeypatch)
    assert olh.HISTORY_PAGE == 50
    (world.paper / "figures" / "withdrawn.png").write_bytes(b"withdrawn figure bytes")
    ours = ours_changed(world)
    for i in range(49):
        (world.paper / "auto" / ("gen%02d.tex" % i)).write_bytes(b"%d\n" % i)

    def foreign_edit_then_tie(fake):
        fake.edit_as("main.tex", b"co-author edit sharing the first upload's timestamp\n")
        fake.hold_clock = 1
    world.fake.upload_hook = foreign_edit_then_tie
    saves = []
    monkeypatch.setattr(sync, "save_state", lambda *a, **k: saves.append(a))
    assert world.run("--release-manifest", "unused.json") == 4
    out = capsys.readouterr().out
    assert "PUSH NOT CERTIFIED" in out and saves == []
    assert "another or unattributed user, files: main.tex" in out  # found, not merely a gap
    assert sum(1 for w in world.fake.writes if w[0] == "upload") == 50
    assert world.fake.files["main.tex"] == ours  # overwritten, therefore it must not certify
    first = min(u["meta"]["start_ts"] for u in world.fake.updates if u["meta"]["users"] == [{"id": OTHER_ID}])
    assert sum(1 for u in world.fake.updates if u["meta"]["start_ts"] == first) == 2  # the tie is real


def test_default_page_size_multi_page_continuous_history_certifies(world, monkeypatch, capsys):
    release(monkeypatch)
    ours_changed(world)
    for i in range(55):
        (world.paper / "auto" / ("gen%02d.tex" % i)).write_bytes(b"%d\n" % i)
    assert world.run("--release-manifest", "unused.json") == 0
    out = capsys.readouterr().out
    assert "history shows only this session's 57 update(s)" in out  # 56 uploads + 1 deletion
    assert sum(1 for m, p in world.fake.log if p.endswith("/updates")) >= 3
    assert jsonlib.loads(world.state.read_text(encoding="utf-8"))["last_action"] == "push"


def test_clean_push_with_a_page_boundary_tie_still_certifies(world, monkeypatch, capsys):
    """The inclusive cursor re-reads the boundary timestamp, so our own tied rows leave no gap."""
    release(monkeypatch)
    ours_changed(world)
    for i in range(55):
        (world.paper / "auto" / ("gen%02d.tex" % i)).write_bytes(b"%d\n" % i)
    downloads = []

    def tie_at_page_boundary(fake):
        downloads.append(1)
        if len(downloads) == 3:  # the post-push verification download, after every write
            newest = list(reversed(fake.updates))
            boundary, older = newest[49], newest[50]
            older["meta"]["start_ts"] = older["meta"]["end_ts"] = boundary["meta"]["start_ts"]
    world.fake.zip_hook = tie_at_page_boundary
    assert world.run("--release-manifest", "unused.json") == 0
    assert "history shows only this session's 57 update(s)" in capsys.readouterr().out


def test_failure_after_the_first_write_uses_the_not_certified_contract(world, monkeypatch, capsys):
    release(monkeypatch)
    ours_changed(world)
    (world.paper / "auto" / "extra.tex").write_bytes(b"extra\n")
    real_request, uploads = world.fake.request, []

    def second_upload_fails(method, url, **kwargs):
        if method == "POST" and url.endswith("/upload"):
            uploads.append(url)
            if len(uploads) == 2:
                return Resp(500, jsonlib.dumps({"error": "storage_unavailable"}))
        return real_request(method, url, **kwargs)
    world.fake.request = second_upload_fails
    before = world.state.read_bytes()
    assert world.run("--release-manifest", "unused.json") == 4
    out = capsys.readouterr().out
    assert "PUSH NOT CERTIFIED" in out and "stopped part-way" in out and "storage_unavailable" in out
    assert world.state.read_bytes() == before


def test_malformed_frame_is_a_sanitized_error_and_stops_before_writes(world, monkeypatch, capsys):
    poll = ("GET", "/socket.io/1/xhr-polling/fakesid")
    world.fake.respond[poll] = lambda fake: Resp(200, "\ufffd%s\ufffd1::" % SECRET, "text/plain")
    client = REAL_CLIENT(PID, {"overleaf_session2": SECRET}, STORED_CSRF, session=world.fake)
    with pytest.raises(olh.OverleafHTTPError) as failure:
        client.file_tree()
    assert "malformed Overleaf response during file_tree (ValueError)" in str(failure.value)
    assert SECRET not in str(failure.value)
    release(monkeypatch)
    ours_changed(world)
    before = world.state.read_bytes()
    assert world.run("--release-manifest", "unused.json") == 1
    out = capsys.readouterr().out
    assert "malformed Overleaf response during file_tree" in out and SECRET not in out
    assert world.fake.writes == [] and world.state.read_bytes() == before


def test_malformed_json_in_reads_is_a_sanitized_transport_error():
    fake = FakeOverleaf(PID, {"main.tex": b"x\n"})
    client = REAL_CLIENT(PID, {"overleaf_session2": SECRET}, STORED_CSRF, session=fake)
    for route, call in ((("GET", "/project/%s/updates" % PID), client.history_head),
                        (("GET", "/project/%s/latest/history" % PID), client.current_version)):
        fake.respond[route] = lambda fake: Resp(200, '{"echo": "%s", broken' % SECRET)
        with pytest.raises(olh.OverleafHTTPError) as failure:
            call()
        assert "malformed Overleaf response" in str(failure.value) and SECRET not in str(failure.value)


def test_unknown_session_user_refuses_before_writing(world, monkeypatch, capsys):
    release(monkeypatch)
    ours_changed(world)
    world.fake.page_user = None
    assert world.run("--release-manifest", "unused.json") == 1
    assert "did not identify the logged-in user" in capsys.readouterr().out
    assert world.fake.writes == []


def test_paged_history_covering_only_our_updates_certifies(world, monkeypatch, capsys):
    release(monkeypatch)
    monkeypatch.setattr(olh, "HISTORY_PAGE", 2)
    ours_changed(world)
    for name in ("a", "b", "c", "d"):
        (world.paper / "auto" / (name + ".tex")).write_bytes(b"%s\n" % name.encode())
    world.fake.edit_as("README_OVERLEAF.txt", b"co-author edit long before the push\n")
    assert world.run("--release-manifest", "unused.json") == 0
    out = capsys.readouterr().out
    assert "history shows only this session's 6 update(s)" in out
    assert sum(1 for m, p in world.fake.log if p.endswith("/updates")) > 2


# ------------------------------------------------------------ dry run (V22 P2)

@pytest.mark.parametrize("flags", [("--dry-run", "--pull"), ("--dry-run", "--pull", "--yes"),
                                   ("--dry-run", "--yes")])
@pytest.mark.parametrize("transport", ["http", "git"])
def test_dry_run_cannot_combine_with_pull_or_yes(world, monkeypatch, capsys, flags, transport):
    world.fake.edit_as("main.tex", b"remote differs, so a pull --yes would copy it\n")
    copies, saves = [], []
    monkeypatch.setattr(sync.shutil, "copy2", lambda *a, **k: copies.append(a))
    monkeypatch.setattr(sync, "save_state", lambda *a, **k: saves.append(a))
    local, state = (world.paper / "main_submission.tex").read_bytes(), world.state.read_bytes()
    world.fake.log.clear()
    if transport == "http":
        assert world.run(*flags) == 1
    else:
        assert sync.main([*flags, "--project-id", PID, "--paper-dir", str(world.paper),
                          "--state", str(world.state)]) == 1
    assert "--dry-run cannot be combined" in capsys.readouterr().out
    assert copies == [] and saves == [] and world.fake.log == []
    assert (world.paper / "main_submission.tex").read_bytes() == local and world.state.read_bytes() == state


# ------------------------------------------------- credentials in errors (V22 P2)

REFRESHED = "SYNTHETIC_REFRESHED_COOKIE_NOT_REAL"


def test_transport_errors_never_carry_credentials():
    fake = FakeOverleaf(PID, {"main.tex": b"x\n"})
    seen = []
    client = REAL_CLIENT(PID, {"overleaf_session2": SECRET}, STORED_CSRF, session=fake, on_secret=seen.append)
    client.open_project()
    values = (SECRET, STORED_CSRF, FRESH_CSRF, REFRESHED)

    def echo(fake):
        fake.cookies.set("overleaf_session2", REFRESHED, domain=".overleaf.com", path="/")
        text = " ".join(values)
        return Resp(500, jsonlib.dumps({"error": text, "message": text}))
    fake.respond[("GET", "/project/%s/download/zip" % PID)] = echo
    with pytest.raises(olh.OverleafHTTPError) as failure:
        client.download_zip()
    assert "HTTP 500" in str(failure.value)
    assert not any(value in str(failure.value) for value in values)
    assert REFRESHED in seen and FRESH_CSRF in seen and SECRET in seen
    assert REFRESHED in client.secrets()

    tree = client.file_tree()
    fake.respond[("POST", "/project/%s/upload" % PID)] = lambda fake: Resp(
        200, jsonlib.dumps({"success": False, "error": " ".join(values)}))
    with pytest.raises(olh.OverleafHTTPError) as failure:
        client.upload("main.tex", b"y\n", tree)
    assert not any(value in str(failure.value) for value in values)
    fake.respond[("POST", "/project/%s/upload" % PID)] = lambda fake: Resp(
        200, jsonlib.dumps({"success": False, "error": "invalid_filename"}))
    with pytest.raises(olh.OverleafHTTPError, match="invalid_filename"):
        client.upload("main.tex", b"y\n", tree)


def test_code_shaped_secret_is_still_scrubbed():
    token = "lowercase_session_value_shaped_like_a_code"
    fake = FakeOverleaf(PID, {"main.tex": b"x\n"})
    client = REAL_CLIENT(PID, {"overleaf_session2": token}, None, session=fake)
    fake.respond[("GET", "/project/%s/download/zip" % PID)] = lambda fake: Resp(500, jsonlib.dumps({"error": token}))
    with pytest.raises(olh.OverleafHTTPError) as failure:
        client.download_zip()
    assert token not in str(failure.value) and "<REDACTED>" in str(failure.value)


def test_cookie_refreshed_by_any_successful_response_is_registered_at_once():
    later = "SYNTHETIC_COOKIE_REFRESHED_ON_DOWNLOAD"
    fake = FakeOverleaf(PID, {"main.tex": b"x\n"})
    seen = []
    client = REAL_CLIENT(PID, {"overleaf_session2": SECRET}, STORED_CSRF, session=fake, on_secret=seen.append)
    fake.zip_hook = lambda fake: fake.cookies.set("overleaf_session2", later, domain=".overleaf.com", path="/")
    client.download_zip()
    assert later in seen


def test_history_window_that_pagination_cannot_cover_fails_closed(monkeypatch):
    fake = FakeOverleaf(PID, {"a.tex": b"1\n", "b.tex": b"2\n", "c.tex": b"3\n", "d.tex": b"4\n", "e.tex": b"5\n"})
    client = REAL_CLIENT(PID, {"overleaf_session2": SECRET}, None, session=fake)
    monkeypatch.setattr(olh, "HISTORY_PAGE", 1)
    assert [u["toV"] for u in client.updates_since(3)] == [5, 4]
    monkeypatch.setattr(olh, "MAX_HISTORY_PAGES", 1)
    with pytest.raises(olh.OverleafHTTPError, match="has a gap between v1 and v5"):
        client.updates_since(1)
    monkeypatch.setattr(olh, "MAX_HISTORY_PAGES", 10)
    real_request = fake.request

    def without_next(*args, **kwargs):
        response = real_request(*args, **kwargs)
        if args[1].endswith("/updates"):
            body = response.json()
            body.pop("nextBeforeTimestamp", None)
            response = Resp(200, jsonlib.dumps(body))
        return response
    fake.request = without_next
    with pytest.raises(olh.OverleafHTTPError, match="has a gap between v1 and v5"):
        client.updates_since(1)


def test_refreshed_session_cookie_is_registered_with_the_cli_scrubber(world, capsys):
    world.fake.refresh_cookie = REFRESHED
    assert world.run("--check") == 0
    assert REFRESHED in sync._SECRETS and sync.scrub("x" + REFRESHED) == "x<REDACTED>"


def test_expired_session_names_the_login_step(world, capsys):
    world.fake.override[("GET", "/project/" + PID)] = 403
    assert world.run("--dry-run") == 1
    out = capsys.readouterr().out
    assert "OVERLEAF_LOGIN_double_click.cmd" in out and no_secrets(out)


def test_read_only_session_cannot_push(world, monkeypatch, capsys):
    release(monkeypatch)
    world.fake.permissions = "readOnly"
    (world.paper / "auto" / "numbers.tex").write_bytes(b"\\newcommand{\\N}{5}\n")
    assert world.run("--release-manifest", "unused.json") == 1
    assert "cannot edit the project" in capsys.readouterr().out
    assert world.fake.writes == []


def test_rate_limit_waits_out_the_window_then_gives_up():
    fake = FakeOverleaf(PID, {"main.tex": b"x\n"})
    route = ("GET", "/project/%s/download/zip" % PID)
    fake.queue[route] = [429, 429]
    waits, notes = [], []
    client = REAL_CLIENT(PID, {"overleaf_session2": SECRET}, None, session=fake,
                         sleep=waits.append, notify=notes.append)
    assert zipfile.ZipFile(io.BytesIO(client.download_zip())).namelist() == ["main.tex"]
    assert waits == list(olh.RATE_LIMIT_WAITS[:2]) and len(notes) == 2
    fake.queue[route] = [429] * (len(olh.RATE_LIMIT_WAITS) + 1)
    with pytest.raises(olh.OverleafHTTPError, match="rate limit"):
        client.download_zip()


# ------------------------------------------------------------ real project

def test_real_project_push_needs_both_flags_before_anything_runs(world, monkeypatch, capsys):
    monkeypatch.setattr(sync, "validate", lambda *a: pytest.fail("validation must not run"))
    monkeypatch.setattr(olh, "OverleafHTTP", lambda *a, **k: pytest.fail("no transport for a refused push"))
    for args in ((), ("--release-manifest", "m.json"), ("--i-understand-this-writes-the-real-project",)):
        assert world.run(*args, project=olh.REAL_PROJECT_ID) == 1
        assert "REFUSING: that is the real paper project" in capsys.readouterr().out
    # The guard applies to the git transport too.
    assert sync.main(["--project-id", olh.REAL_PROJECT_ID.upper(), "--release-manifest", "m.json",
                      "--paper-dir", str(world.paper), "--state", str(world.state)]) == 1


def test_client_refuses_writes_to_real_project_without_explicit_permission():
    fake = FakeOverleaf(olh.REAL_PROJECT_ID, {"main.tex": b"x\n"})
    client = REAL_CLIENT(olh.REAL_PROJECT_ID, {"overleaf_session2": SECRET}, STORED_CSRF, session=fake)
    client.open_project()
    tree = client.file_tree()
    for attempt in (lambda: client.upload("main.tex", b"y\n", tree),
                    lambda: client.upload("new/dir/file.tex", b"y\n", tree),
                    lambda: client.delete("main.tex", tree)):
        with pytest.raises(olh.RealProjectWriteRefused):
            attempt()
    assert fake.writes == []
    assert olh.is_real_project(" " + olh.REAL_PROJECT_ID.upper())
    assert olh.fingerprint(olh.REAL_PROJECT_ID) == olh.REAL_PROJECT_FINGERPRINT
    assert not olh.is_real_project(PID)


@pytest.mark.parametrize("pushing, manifest, confirm, allowed", [
    (False, "m.json", True, False), (True, None, True, False),
    (True, "m.json", False, False), (True, "m.json", True, True)])
def test_write_permission_needs_push_manifest_and_confirmation(monkeypatch, pushing, manifest, confirm, allowed):
    monkeypatch.setattr(sync, "_SECRETS", [])
    monkeypatch.setattr(olh, "load_session", lambda path=None: ({"overleaf_session2": SECRET}, STORED_CSRF))
    args = SimpleNamespace(olauth=None, project_id=olh.REAL_PROJECT_ID,
                           release_manifest=manifest, confirm_real_project=confirm)
    client = sync.open_http_transport(args, pushing)
    assert client._allow_real is allowed
    assert SECRET in sync._SECRETS and sync.scrub("x" + SECRET) == "x<REDACTED>"


def test_real_project_dry_run_reads_only(world, monkeypatch, capsys):
    fake = FakeOverleaf(olh.REAL_PROJECT_ID, {"main.tex": lf(LOCAL["main_submission.tex"])})
    install(monkeypatch, fake, world.tmp)
    before = world.state.read_bytes()
    assert world.run("--dry-run", project=olh.REAL_PROJECT_ID) == 0
    assert fake.writes == [] and all(m == "GET" for m, _p in fake.log)
    assert world.state.read_bytes() == before


# ------------------------------------------------------- session and ZIP

def test_session_file_loads_plain_data_only(tmp_path):
    good = tmp_path / "good"
    good.write_bytes(pickle.dumps({"cookie": {"overleaf_session2": SECRET, "other": "x"}, "csrf": STORED_CSRF}))
    assert olh.load_session(good) == ({"overleaf_session2": SECRET}, STORED_CSRF)
    hostile = tmp_path / "hostile"
    hostile.write_bytes(pickle.dumps({"cookie": {"overleaf_session2": SECRET}, "csrf": SimpleNamespace()}))
    with pytest.raises(olh.SessionError) as failure:
        olh.load_session(hostile)
    assert SECRET not in str(failure.value)
    with pytest.raises(olh.SessionError, match="OVERLEAF_LOGIN_double_click"):
        olh.load_session(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.write_bytes(pickle.dumps({"cookie": {}, "csrf": None}))
    with pytest.raises(olh.SessionError):
        olh.load_session(empty)


@pytest.mark.parametrize("names", [["../escape.tex"], ["a/../../x.tex"], ["/abs.tex"],
                                   ["c:evil.tex"], ["Main.tex", "main.tex"]])
def test_zip_extraction_refuses_unsafe_or_colliding_names(tmp_path, names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name in names:
            archive.writestr(name, b"x")
    with pytest.raises(olh.OverleafHTTPError):
        olh.extract_zip(buf.getvalue(), tmp_path / "out")
    assert not (tmp_path / "escape.tex").exists()


def test_socket_frames_and_tree_index():
    assert olh.decode_frames("1::") == ["1::"]
    assert olh.decode_frames("\ufffd3\ufffd1::\ufffd4\ufffd2:::") == ["1::", "2:::"]
    fake = FakeOverleaf(PID, {"main.tex": b"x", "figures/a/b.png": PNG})
    tree = olh.index_tree(fake.tree())
    assert tree["entities"]["main.tex"][0] == "doc"
    assert tree["entities"]["figures/a/b.png"][0] == "file"
    assert set(tree["folders"]) == {"", "figures", "figures/a"}


def test_unsafe_state_path_is_refused(tmp_path):
    state = {"files": {"../outside.tex": {"remote_sha256": "a", "local_sha256": "a"}}}
    with pytest.raises(ValueError, match="unsafe path"):
        sync.plan_deletions([], {}, state, str(tmp_path))


def test_git_stays_the_default_transport(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OVERLEAF_TOKEN", raising=False)
    monkeypatch.setattr(sync.assets, "unique_work", lambda *a, **k: tmp_path)
    monkeypatch.setattr(olh, "load_session", lambda *a: pytest.fail("git must not load the browser session"))
    (tmp_path / "paper").mkdir()
    assert sync.main(["--dry-run", "--project-id", PID, "--paper-dir", str(tmp_path / "paper"),
                      "--state", str(tmp_path / "state.json")]) == 1
    assert "OVERLEAF_TOKEN" in capsys.readouterr().out
