#!/usr/bin/env python
"""Two-way, conflict-aware sync between this repository and an Overleaf project.

Why
---
The original loop was: build the ZIP, read CHANGED.txt, then hand-upload two or
three files through the Overleaf web UI. That is the slowest and most
error-prone step left, and it has already misled once - CHANGED.txt reported
"nothing changed" immediately after a 269-line edit, because it diffed against
the last build rather than the last upload.

Overleaf exposes every project as a git remote, so the whole step can be one
command. A successful release manifest selects the exact validated source tree
and separately checked Word attachment. Text is LF-normalized for git; unused
local figures are not implicitly uploaded. Remote-only files are preserved.

THE DATA-LOSS BUG THIS VERSION CLOSES
-------------------------------------
The first version was one-way and unconditional. It cloned the project, copied
our files over the top, committed and pushed. If the operator opened main.tex in
the Overleaf editor, fixed a sentence, and a sync then ran, that edit was
silently destroyed - the push carried no record of what it replaced, so there
was not even a diff to look at afterwards. Nothing was actually lost only
because the two copies happened to be byte-identical at the time.

The fix is a three-way comparison instead of a blind overwrite. After every
successful push or pull the script records the SHA-256 of every managed file as
it then stood, in .overleaf_sync.json. That file is the answer to "what did
Overleaf look like the last time we agreed with it". On the next run each
managed file falls into exactly one case:

    remote == local                     nothing to do
    remote == recorded, local differs    ours is newer; safe to overwrite
    remote != recorded                   SOMEONE EDITED OVERLEAF. Refuse.
    no recorded state at all             unknown; refuse unless told what to do

A refusal prints a unified diff of remote versus local so the operator can see
exactly what would have been destroyed, and exits non-zero. The only ways past
it are explicit reconciliation or --pull (take Overleaf's copy instead).
There is no force-push or overwrite-conflicts option.

.overleaf_sync.json holds hashes only. The project id is stored as a truncated
SHA-256 so the file carries nothing sensitive and can be committed; committing
it is the point, otherwise a fresh clone of this repository would start with no
memory of the last agreement and refuse the first sync.

REMOTE DELETIONS
----------------
A file the operator deleted in Overleaf is also a remote edit. If it is present
in the recorded state it is reported as a conflict, not silently re-added. If it
is absent from Overleaf at the moment of a --pull, the pull records it as
intentionally absent, so later syncs leave it alone instead of resurrecting it.
This is what happened to the uncited figure variants the first sync uploaded and
the operator then deleted by hand.

LINE ENDINGS
------------
Git for Windows ships core.autocrlf=true in the system config, so a plain clone
rewrites every LF in the checkout to CRLF. Hashing that checkout would then
disagree with this repository on every text file at once. The scratch clone
therefore pins core.autocrlf=input: no translation on checkout, so the working
tree bytes are the blob bytes, and CRLF folded to LF on commit, so the project
history keeps the line endings it already has. Hashes of text files are taken
over LF-normalised content for the same reason - see content_hash().

Requirements
------------
1. An Overleaf project id. Open the project; the URL is
       https://www.overleaf.com/project/<PROJECT_ID>
2. An Overleaf git authentication token:
       Account Settings -> Git Integration -> Generate token
   HONEST CAVEAT: Overleaf's git integration is a PREMIUM feature. On a free
   plan the remote will refuse authentication. If that happens, nothing here
   will work and the loose-file mirror remains the fallback.

Configure credentials only in the process environment (not CLI arguments):

Set OVERLEAF_PROJECT_ID or pass --project-id. Inject OVERLEAF_TOKEN from an
existing secret store into this process's environment; do not put its value in
shell arguments, checked-in files, or remote URLs.

The clone URL is credential-free. Authentication is passed only through
subprocess-scoped Git configuration in the environment, with credential
helpers, prompting, redirects and tracing disabled. Nothing writes the token
to argv, repository configuration, credential storage or diagnostic files.

Usage
-----
    python scripts/sync_overleaf.py --check          # test auth only
    python scripts/sync_overleaf.py --dry-run        # classify, push nothing
    python scripts/sync_overleaf.py --release-manifest PATH  # validate, then push
    python scripts/sync_overleaf.py --pull           # show what Overleaf would
                                                     # change locally
    python scripts/sync_overleaf.py --pull --yes     # actually write it locally

Exit codes
----------
    0  success
    1  bad usage, missing credentials, clone or validation failure
    2  refused: Overleaf has changes we have not seen
    3  refused: --pull would change local files and --yes was not given

The push is refused unless the build passes all of p13_build_zip.py's checks,
so a broken or over-length paper can never reach the project.
"""
import argparse
import base64
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from autopilot import release_assets as assets
PAPER_DEFAULT = os.path.join(REPO, "paper", "genai4health2026")
VENV_PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
STATE_DEFAULT = os.path.join(REPO, ".overleaf_sync.json")
STATE_VERSION = 1

# What an Overleaf project needs. Deliberately NOT the whole repository: the
# project should contain the paper and nothing else.
#
# NOTE ON NAMING, learned by inspecting the live project before the first push:
# Overleaf compiles main.tex, while this repository calls the same document
# main_submission.tex. A naive copy would have added main_submission.tex and
# DELETED main.tex, leaving the project with no document to compile. It also
# carries neurips_2026.sty, without which the build fails. Both are handled by
# mapping local name -> Overleaf name below.
FILE_MAP = {
    "main_submission.tex": "main.tex",
    "references.bib": "references.bib",
    "neurips_2026.sty": "neurips_2026.sty",
    # A Word rendering for collaborators who do not write LaTeX. Named so it
    # cannot be mistaken for the source: Overleaf compiles main.tex, and edits
    # made in Word have to be carried back by hand. Rebuild it with
    # autopilot/make_docx.py after any change to the manuscript.
    "main_submission.docx": "main_editable.docx",
}
INCLUDE_DIRS = ["auto", "figures"]
SKIP_EXT = (".aux", ".log", ".out", ".blg", ".synctex.gz", ".fls",
            ".fdb_latexmk", ".toc")
TEXT_EXT = (".tex", ".bib", ".sty", ".cls", ".bst", ".txt", ".md", ".csv")
DIFF_LINES = 40

# Anything appended here is stripped out of every line this script prints.
_SECRETS = []

# Git is invoked with checkout-time line-ending translation disabled
# everywhere; see the LINE ENDINGS note in the module docstring.
GIT = ["git", "-c", "core.autocrlf=input", "-c", "core.eol=lf"]


def scrub(text):
    """Remove the token from anything on its way to the terminal."""
    if not text:
        return text
    for s in _SECRETS:
        if s:
            text = text.replace(s, "<REDACTED>")
    return text


def say(text=""):
    print(scrub(str(text)))


def child_environment():
    env = dict(os.environ)
    for key in list(env):
        if (key == "OVERLEAF_TOKEN" or key.startswith("GIT_TRACE")
                or key.startswith("GIT_CONFIG_KEY_") or key.startswith("GIT_CONFIG_VALUE_")
                or key in ("GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "GIT_CURL_VERBOSE")):
            env.pop(key, None)
    env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never", "GIT_TRACE_REDACT": "1"})
    return env


def authentication_environment(token, remote_url):
    if not remote_url.startswith("https://git.overleaf.com/") or "@" in remote_url:
        raise ValueError("authentication requires a credential-free Overleaf HTTPS URL")
    encoded = base64.b64encode(("git:" + token).encode("utf-8")).decode("ascii")
    _SECRETS.extend((token, encoded))
    env = child_environment()
    configuration = [
        ("credential.helper", ""),
        ("http.extraHeader", ""),
        ("http.%s.extraHeader" % remote_url, "Authorization: Basic " + encoded),
        ("http.followRedirects", "false"),
        ("credential.interactive", "never"),
        ("core.hooksPath", os.devnull),
        ("init.templateDir", ""),
    ]
    env["GIT_CONFIG_COUNT"] = str(len(configuration))
    for index, (key, value) in enumerate(configuration):
        env["GIT_CONFIG_KEY_%d" % index] = key
        env["GIT_CONFIG_VALUE_%d" % index] = value
    return env


def run(cmd, cwd=None, check=True, quiet=False, env=None):
    if any(secret and secret in str(argument) for secret in _SECRETS for argument in cmd):
        raise ValueError("credential material must not enter command arguments")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       env=env if env is not None else child_environment())
    r.stdout, r.stderr = scrub(r.stdout), scrub(r.stderr)
    if not quiet and r.stdout.strip():
        say(r.stdout.rstrip())
    if check and r.returncode != 0:
        say(r.stderr.rstrip())
        raise SystemExit(scrub("command failed: %s" % " ".join(cmd)))
    return r


def rmtree_force(path):
    """Delete a git checkout on Windows.

    shutil.rmtree(ignore_errors=True) does NOT clear a scratch clone: git marks
    everything under .git/objects read-only, the unlink fails, the error is
    swallowed, and the next run then dies with "destination path already exists
    and is not an empty directory". Clear the read-only bit and retry.
    """
    def onerror(func, p, _exc):
        os.chmod(p, 0o700)
        func(p)
    if os.path.exists(path):
        shutil.rmtree(path, onerror=onerror)


def sanitize_clone_config(clone):
    """Sanitize legacy transport metadata; never print its contents."""
    clone = Path(clone).resolve()
    config = assets.safe_path(clone, os.path.join(".git", "config"))
    candidates = [config, assets.safe_path(clone, os.path.join(".git", "FETCH_HEAD"))]
    logs = assets.safe_path(clone, os.path.join(".git", "logs"))
    if logs.is_dir():
        for root, directories, filenames in os.walk(logs, followlinks=False):
            directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
            for name in filenames:
                candidates.append(assets.safe_path(clone, (Path(root) / name).relative_to(clone)))
    for path in candidates:
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        if (any(secret and secret in raw for secret in _SECRETS)
                or re.search(r"https?://[^\s/]*@", raw)
                or re.search(r"(?i)extraheader|credential", raw)):
            os.chmod(path, 0o600)
            replacement = "[core]\n\tbare = false\n" if path == config else "Legacy authentication metadata removed.\n"
            path.write_text(replacement, encoding="utf-8")


def cleanup_clone(clone):
    try:
        sanitize_clone_config(clone)
        rmtree_force(clone)
    except Exception:
        try:
            sanitize_clone_config(clone)
        except Exception:
            raise RuntimeError("clone cleanup failed; sanitization unconfirmed at " + str(clone)) from None
        raise RuntimeError("clone cleanup failed; sanitized residual checkout retained at " + str(clone)) from None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(path, rel):
    """SHA-256 of the bytes git will store for this file.

    Not simply the hash of the file on disk. The generated tables under auto/
    are written by Python on Windows and land with CRLF, while the blobs in the
    Overleaf repository are LF because git normalised them on the way in. A
    byte-exact comparison therefore flagged twelve files as remote edits when
    the text was character-for-character identical, which would have trained the
    operator to ignore the conflict warning - the one outcome that makes this
    whole mechanism worthless. Compare the normalised content instead, which is
    exactly what a commit would compare.
    """
    if not is_text(rel):
        return sha256_file(path)
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def is_text(rel):
    return rel.lower().endswith(TEXT_EXT)


def short(sha):
    return sha[:12] if sha else "-"


# ---------------------------------------------------------------- sync state

def load_state(path):
    """Return the recorded agreement, or an empty one if there is none."""
    if not os.path.isfile(path):
        return {"version": STATE_VERSION, "files": {}}, False
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
    except Exception as e:
        say("  warning: %s is unreadable (%s); treating as first run" %
            (os.path.basename(path), e))
        return {"version": STATE_VERSION, "files": {}}, False
    if not isinstance(st, dict) or "files" not in st:
        return {"version": STATE_VERSION, "files": {}}, False
    return st, True


def project_fingerprint(project_id):
    """Identify the project without storing the id itself."""
    return hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:16]


def save_state(path, project_id, action, files):
    st = {
        "version": STATE_VERSION,
        "_comment": ("SHA-256 of every file this repository manages in the "
                     "Overleaf project, as of the last agreed sync. Used to "
                     "tell an unseen remote edit apart from a stale remote. "
                     "Hashes only - safe to commit."),
        "project": project_fingerprint(project_id),
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_action": action,
        "files": {k: files[k] for k in sorted(files)},
    }
    work = assets.unique_work(prefix="sync-state")
    candidate = work / "state.json"
    assets.write_json(candidate, st)
    assets.promote([(candidate, path)])
    candidate.unlink()
    work.rmdir()
    return st


# ------------------------------------------------------------ file discovery

def collect(paper):
    """Local files we manage, as (local absolute path, Overleaf relative path)."""
    items = []
    for local, remote in FILE_MAP.items():
        p = os.path.join(paper, local)
        if os.path.isfile(p):
            items.append((p, remote))
    for d in INCLUDE_DIRS:
        root = os.path.join(paper, d)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if fn.endswith(SKIP_EXT):
                    continue
                src = os.path.join(dirpath, fn)
                items.append((src, os.path.relpath(src, paper).replace("\\", "/")))
    return items


def managed_remote(rel):
    """Is this Overleaf path one we are responsible for?

    Everything else in the project - README_OVERLEAF.txt, a compiled main.pdf,
    scratch files the operator uploaded - is none of our business and is never
    touched, in either direction.
    """
    if rel.endswith(SKIP_EXT):
        return False
    if rel in FILE_MAP.values():
        return True
    return any(rel.startswith(d + "/") for d in INCLUDE_DIRS)


def remote_to_local(rel, paper):
    """Reverse the name map: Overleaf main.tex -> local main_submission.tex."""
    for local, remote in FILE_MAP.items():
        if rel == remote:
            return os.path.join(paper, local)
    return os.path.join(paper, rel.replace("/", os.sep))


def scan_remote(clone):
    """SHA-256 of every managed file in the checkout, keyed by Overleaf path."""
    out = {}
    for dirpath, dirs, files in os.walk(clone):
        dirs[:] = [d for d in dirs if d != ".git"]
        for fn in files:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, clone).replace("\\", "/")
            if managed_remote(rel):
                out[rel] = content_hash(full, rel)
    return out


# -------------------------------------------------------------------- diffs

def show_diff(remote_path, local_path, remote_rel, local_rel, limit=DIFF_LINES):
    """Unified diff, remote versus local, trimmed so a huge file cannot flood."""
    if not is_text(remote_rel):
        rs = os.path.getsize(remote_path) if remote_path and os.path.exists(remote_path) else 0
        ls = os.path.getsize(local_path) if local_path and os.path.exists(local_path) else 0
        say("      binary file; overleaf %d bytes, local %d bytes" % (rs, ls))
        return

    def read(p):
        if not p or not os.path.exists(p):
            return []
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines(keepends=True)

    d = list(difflib.unified_diff(read(remote_path), read(local_path),
                                  fromfile="overleaf/" + remote_rel,
                                  tofile="local/" + local_rel, n=2))
    if not d:
        say("      (no textual difference)")
        return
    for line in d[:limit]:
        say("      " + line.rstrip("\n"))
    if len(d) > limit:
        say("      ... %d more diff lines suppressed" % (len(d) - limit))


# --------------------------------------------------------------- validation

def verify_local_release(paper, manifest):
    """Recheck identity after validation and immediately before publishing."""
    if manifest.get("ALL_PASS") is not True or not manifest.get("checks"):
        raise ValueError("manifest is not a successful release")
    required = {"immutable_figure_inputs", "no_placeholders", "all_graphics_present", "manuscript", "numeric_evidence",
                "numeric_review_input", "citation_metadata", "compiles_standalone", "page_limit", "anonymous",
                "no_undefined_refs", "docx_generated", "docx_complete"}
    if not required.issubset(manifest["checks"]) or not all(
            manifest["checks"][name] is True for name in required):
        raise ValueError("required release gates are missing or failed")
    sources = manifest.get("source_files", {})
    if not sources or "main.tex" not in sources:
        raise ValueError("empty or incomplete validated source tree")
    expected = {item["source"]: item["sha256"] for item in sources.values()}
    current_inputs = assets.input_hashes(paper)
    if current_inputs != expected:
        raise ValueError("local release inputs changed since validation")
    expected_names = {"main.tex" if rel == "main_submission.tex" else rel: rel
                      for rel in current_inputs}
    if {rel: item["source"] for rel, item in sources.items()} != expected_names:
        raise ValueError("release source name mapping differs from the compiled tree")
    if assets.asset_inventory(paper) != manifest.get("assets"):
        raise ValueError("asset identity/producer declaration changed; rebuild release")
    reported_review = manifest.get("evidence", {}).get("numbers", {}).get("review_sha256")
    assets.verify_numeric_review(manifest.get("numeric_review"), reported_review)
    for artifact in manifest["artifacts"].values():
        if assets.sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError("release artifact changed: " + artifact["path"])
    for rel, item in manifest.get("attachments", {}).items():
        if rel != "main_editable.docx" or item.get("check") != "check_docx.py":
            raise ValueError("unsupported/unvalidated extra attachment: " + rel)
        if assets.sha256(assets.safe_path(paper, item["source"])) != item["sha256"]:
            raise ValueError("Word attachment changed; reconcile comments/edits before regenerating")
    if "main_editable.docx" not in manifest.get("attachments", {}):
        raise ValueError("required checked Word attachment is missing")


def validate(paper, manifest_path, work):
    """Freeze only manifest-declared bytes; independently enforce Word checks."""
    import zipfile
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    verify_local_release(paper, manifest)
    sources = manifest["source_files"]
    stage = Path(work) / "validated"
    stage.mkdir()
    items = []
    with zipfile.ZipFile(manifest["artifacts"]["zip"]["path"]) as archive:
        expected_names = set(sources) | {"main.pdf", "README_OVERLEAF.txt"}
        if set(archive.namelist()) != expected_names or len(archive.namelist()) != len(expected_names):
            raise ValueError("ZIP tree differs from the exact validated manifest")
        for rel, item in sources.items():
            data = archive.read(rel)
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("ZIP source hash mismatch: " + rel)
            target = assets.safe_path(stage, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            items.append((str(target), rel))
    aux = manifest.get("aux")
    if not aux or assets.sha256(aux["path"]) != aux["sha256"]:
        raise ValueError("validated aux unavailable/changed; rebuild release")
    word = assets.safe_path(paper, manifest["attachments"]["main_editable.docx"]["source"])
    py = VENV_PY if os.path.isfile(VENV_PY) else sys.executable
    result = run([py, os.path.join(REPO, "autopilot", "check_docx.py"),
                  "--paper-dir", str(paper), "--docx", str(word), "--aux", aux["path"]],
                 cwd=REPO, check=False, quiet=True)
    if result.returncode:
        raise ValueError("Word completeness gate failed: " + result.stdout[-1000:])
    frozen_word = stage / "main_editable.docx"
    shutil.copyfile(word, frozen_word)
    if assets.sha256(frozen_word) != manifest["attachments"]["main_editable.docx"]["sha256"]:
        raise ValueError("Word changed while freezing attachment")
    items.append((str(frozen_word), "main_editable.docx"))
    verify_local_release(paper, manifest)
    return items, manifest


def verify_remote_tip(clone, branch, expected, *, env=None):
    result = run(GIT + ["ls-remote", "origin", "refs/heads/" + branch],
                 cwd=clone, quiet=True, env=env)
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][0] != expected:
        raise ValueError("Overleaf moved since conflict review; restart sync (no force)")


def verify_staged_bytes(clone, items):
    """Git attributes/filters must not change what passed the release gates."""
    for source, rel in items:
        result = subprocess.run(GIT + ["show", ":" + rel], cwd=clone, capture_output=True,
                                env=child_environment())
        expected = Path(source).read_bytes()
        if is_text(rel):
            expected = expected.replace(b"\r\n", b"\n")
        if result.returncode or result.stdout != expected:
            raise ValueError("git staged bytes differ from validated source: " + rel)


# ----------------------------------------------------------- classification

IN_SYNC = "in_sync"
ADD = "add"
UPDATE = "update"
AGREED_ABSENT = "agreed_absent"
CONFLICT_UNKNOWN = "conflict_no_recorded_state"
CONFLICT_EDITED = "conflict_remote_edited"
CONFLICT_DELETED = "conflict_remote_deleted"
CONFLICTS = (CONFLICT_UNKNOWN, CONFLICT_EDITED, CONFLICT_DELETED)


def classify(items, rhashes, state):
    """Three-way compare of local, remote and the last agreed state.

    Returns one record per managed local file. This is the whole safety
    argument of the script, so each branch is spelled out rather than folded
    into a clever expression.
    """
    recorded = state.get("files", {})
    out = []
    for src, rel in sorted(items, key=lambda x: x[1]):
        lh = content_hash(src, rel)
        rh = rhashes.get(rel)
        entry = recorded.get(rel)
        known = isinstance(entry, dict)
        sr = entry.get("remote_sha256") if known else None
        sl = entry.get("local_sha256") if known else None

        if rh == lh:
            status = IN_SYNC
        elif not known:
            # First run for this file. We have no idea whether the remote copy
            # is an operator edit or just a stale push, so we do not guess.
            status = ADD if rh is None else CONFLICT_UNKNOWN
        elif rh != sr:
            # The remote moved underneath us since the last agreement.
            status = CONFLICT_DELETED if rh is None else CONFLICT_EDITED
        elif rh is None:
            # Agreed absent at the last sync. If the local file has not changed
            # since, the operator meant it to stay out of the project.
            status = AGREED_ABSENT if sl == lh else ADD
        else:
            status = UPDATE

        out.append({"local": src, "remote": rel, "local_sha256": lh,
                    "remote_sha256": rh, "recorded_remote": sr,
                    "status": status})
    return out


def report(records, paper, clone, verbose_diffs):
    buckets = {}
    for r in records:
        buckets.setdefault(r["status"], []).append(r)
    order = [(CONFLICT_EDITED, "CONFLICT - edited in Overleaf since last sync"),
             (CONFLICT_DELETED, "CONFLICT - deleted in Overleaf since last sync"),
             (CONFLICT_UNKNOWN, "CONFLICT - no recorded state; cannot tell who is newer"),
             (UPDATE, "would update in Overleaf"),
             (ADD, "would add to Overleaf"),
             (AGREED_ABSENT, "left out of Overleaf by earlier agreement"),
             (IN_SYNC, "already identical")]
    for key, label in order:
        rows = buckets.get(key)
        if not rows:
            continue
        say("\n%s (%d):" % (label, len(rows)))
        show = rows if (key in CONFLICTS or len(rows) <= 12) else rows[:12]
        for r in show:
            say("   %s" % r["remote"])
            if key in CONFLICTS and verbose_diffs:
                say("      overleaf %s  recorded %s  local %s" %
                    (short(r["remote_sha256"]), short(r["recorded_remote"]),
                     short(r["local_sha256"])))
                rp = os.path.join(clone, r["remote"].replace("/", os.sep))
                show_diff(rp if os.path.exists(rp) else None, r["local"],
                          r["remote"],
                          os.path.relpath(r["local"], paper).replace("\\", "/"))
        if len(show) < len(rows):
            say("   ... and %d more" % (len(rows) - len(show)))
    return buckets


# -------------------------------------------------------------------- pull

def do_pull(clone, rhashes, items, paper, state_path, project_id, yes):
    """Bring Overleaf's managed files back into the local paper directory."""
    changes, same = [], []
    for rel in sorted(rhashes):
        dst = remote_to_local(rel, paper)
        cur = content_hash(dst, rel) if os.path.isfile(dst) else None
        (same if cur == rhashes[rel] else changes).append((rel, dst, cur))

    local_only = [rel for _src, rel in items if rel not in rhashes]

    say("\npull: Overleaf has %d managed files; %d already identical locally" %
        (len(rhashes), len(same)))
    if local_only:
        say("      %d managed local files are not in the project at all; a pull"
            % len(local_only))
        say("      records them as intentionally absent so later syncs do not")
        say("      resurrect them. Reconcile intentionally absent files explicitly.")

    if not changes:
        say("\nlocal paper already matches Overleaf; nothing to write.")
    else:
        say("\nthese local files WOULD CHANGE (%d):" % len(changes))
        for rel, dst, cur in changes:
            say("   %s  ->  %s%s"
                % (rel, os.path.relpath(dst, paper).replace("\\", "/"),
                   "   (new file)" if cur is None else ""))
            show_diff(os.path.join(clone, rel.replace("/", os.sep)), dst, rel,
                      os.path.relpath(dst, paper).replace("\\", "/"))
        if not yes:
            say("\nrefusing to overwrite local work without confirmation.")
            say("re-run with --pull --yes to apply the %d change(s) above."
                % len(changes))
            return 3
        for rel, dst, _cur in changes:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(clone, rel.replace("/", os.sep)), dst)
            say("   wrote %s" % os.path.relpath(dst, paper).replace("\\", "/"))

    files = {rel: {"remote_sha256": rhashes[rel], "local_sha256": rhashes[rel]}
             for rel in rhashes}
    for src, rel in items:
        if rel not in rhashes:
            files[rel] = {"remote_sha256": None, "local_sha256": content_hash(src, rel)}
    save_state(state_path, project_id, "pull", files)
    say("\nrecorded agreement in %s (%d files)"
        % (os.path.relpath(state_path, REPO), len(files)))
    return 0


# -------------------------------------------------------------------- main

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(argument.startswith("--token") for argument in argv):
        say("Token CLI arguments are disabled; use the process OVERLEAF_TOKEN environment variable.")
        return 1
    ap = argparse.ArgumentParser(
        description="Two-way, conflict-aware Overleaf sync.")
    ap.add_argument("--project-id", default=os.environ.get("OVERLEAF_PROJECT_ID"))
    ap.add_argument("--dry-run", action="store_true",
                    help="classify everything and push nothing")
    ap.add_argument("--check", action="store_true", help="test auth only")
    ap.add_argument("--pull", action="store_true",
                    help="copy Overleaf's managed files back into the paper dir")
    ap.add_argument("--yes", action="store_true",
                    help="confirm a --pull that would change local files")
    ap.add_argument("--release-manifest",
                    help="successful p13 release manifest; required for a push")
    ap.add_argument("--paper-dir", default=PAPER_DEFAULT,
                    help="local paper directory (default: paper/genai4health2026)")
    ap.add_argument("--state", default=STATE_DEFAULT,
                    help="sync state file (default: .overleaf_sync.json)")
    ap.add_argument("--message", default="sync validated submission")
    a = ap.parse_args(argv)
    token = os.environ.get("OVERLEAF_TOKEN")
    if token:
        _SECRETS.append(token)
    paper = os.path.abspath(a.paper_dir)
    state_path = os.path.abspath(a.state)

    if not os.path.isdir(paper):
        say("no such paper directory: %s" % paper)
        return 1

    # Discovery-only operations retain the historical broad inventory. A push
    # replaces it below with only the frozen, manifest-validated tree.
    items = collect(paper)
    manifest = None
    work = assets.unique_work(prefix="sync")
    if not (a.pull or a.check) and (a.release_manifest or not a.dry_run):
        if not a.release_manifest:
            say("A successful --release-manifest is required; no files pushed.")
            return 1
        try:
            items, manifest = validate(paper, a.release_manifest, work)
        except (OSError, ValueError, KeyError) as exc:
            say("Release validation failed: " + str(exc))
            return 1
    elif a.dry_run:
        say("Discovery-only dry run: no release manifest supplied; not publication validation.")
    total = sum(os.path.getsize(s) for s, _ in items)
    say("local paper: %s" % paper)
    say("files we manage: %d, %.2f MB" % (len(items), total / 1e6))
    for _s, rel in sorted(items, key=lambda x: x[1])[:12]:
        say("   %s" % rel)
    if len(items) > 12:
        say("   ... and %d more" % (len(items) - 12))

    state, had_state = load_state(state_path)
    if had_state:
        say("sync state: %s, %d files, last %s at %s"
            % (os.path.relpath(state_path, REPO), len(state.get("files", {})),
               state.get("last_action", "?"), state.get("updated_utc", "?")))
        if a.project_id and state.get("project") not in (
                None, project_fingerprint(a.project_id)):
            say("  warning: this state file was written against a DIFFERENT")
            say("  project; treating every file as first-run.")
            state = {"version": STATE_VERSION, "files": {}}
            had_state = False
    else:
        say("sync state: none recorded yet (first run)")

    if not a.project_id or not token:
        say("\nMissing credentials. Set OVERLEAF_PROJECT_ID and OVERLEAF_TOKEN,")
        say("or pass --project-id with OVERLEAF_TOKEN set in the process environment.")
        say("Token: Overleaf Account Settings -> Git Integration -> Generate token.")
        say("Note: Overleaf git access is a premium feature; on a free plan this")
        say("will fail authentication and the loose-file mirror stays the fallback.")
        return 1

    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.project_id):
        say("Invalid project identifier.")
        return 1
    remote_url = "https://git.overleaf.com/%s" % a.project_id
    auth_env = authentication_environment(token, remote_url)

    # The scratch checkout lives inside the ignored _release_work tree
    # rather than in the system temp directory, so a crash leaves the evidence
    # next to the work instead of somewhere nobody will look.
    clone = str(work / "clone")
    try:
        say("\ncloning %s" % remote_url)
        r = run(GIT + ["clone", "--depth", "1", remote_url, clone],
                check=False, quiet=True, env=auth_env)
        if r.returncode != 0:
            say(r.stderr.strip()[:400])
            say("\nClone failed. Most likely causes: wrong project id, wrong or")
            say("expired token, or a free plan without git integration.")
            return 1
        say("  clone OK - authentication works")
        # Persist the setting so add/commit agree with what was checked out.
        run(["git", "config", "core.autocrlf", "input"], cwd=clone, quiet=True)
        run(["git", "config", "core.eol", "lf"], cwd=clone, quiet=True)
        remote_tip = run(GIT + ["rev-parse", "HEAD"], cwd=clone, quiet=True).stdout.strip()
        branch = run(GIT + ["symbolic-ref", "--short", "HEAD"], cwd=clone, quiet=True).stdout.strip()
        if a.check:
            say("\n--check: authentication verified, nothing pushed")
            return 0

        rhashes = scan_remote(clone)
        say("  overleaf has %d managed files" % len(rhashes))

        if a.pull:
            return do_pull(clone, rhashes, items, paper, state_path,
                           a.project_id, a.yes)

        records = classify(items, rhashes, state)
        report(records, paper, clone, verbose_diffs=True)

        conflicts = [r for r in records if r["status"] in CONFLICTS]
        if manifest and any(record["status"] == AGREED_ABSENT for record in records):
            say("REFUSING: a required validated input was intentionally deleted remotely; reconcile first.")
            return 2
        to_push = [r for r in records if r["status"] in (ADD, UPDATE)]

        # --dry-run reports and never fails. It writes nothing in either
        # direction, so giving it the refusal exit code would only make it
        # useless inside a shell that stops on error.
        if a.dry_run:
            say("\ndry run; nothing pushed, nothing written locally.")
            if conflicts:
                say("a real run would REFUSE: %d file(s) changed in Overleaf "
                    "since the last sync (exit 2)." % len(conflicts))
                say("%d other file(s) would have been written." % len(to_push))
            else:
                say("%d file(s) would be written to Overleaf." % len(to_push))
            return 0

        if conflicts:
            say("\nREFUSING TO PUSH. %d file(s) changed in Overleaf since the last"
                % len(conflicts))
            say("sync; pushing would destroy those changes.")
            if not had_state:
                say("There is no recorded sync state, so this may simply be the")
                say("first run against an already-populated project. Either way,")
                say("we do not get to assume ours wins.")
            say("\nChoose one:")
            say("  --pull          show what Overleaf would put back locally")
            say("  --pull --yes    take Overleaf's version into the repository")
            say("  reconcile       review and merge edits before rebuilding a release")
            return 2

        if not to_push:
            verify_local_release(paper, manifest)
            verify_remote_tip(clone, branch, remote_tip, env=auth_env)
            say("\nOverleaf already matches the local paper; nothing to push.")
            files = {r["remote"]: {"remote_sha256": r["remote_sha256"],
                                   "local_sha256": r["local_sha256"]}
                     for r in records}
            save_state(state_path, a.project_id, "verify", files)
            say("recorded agreement in %s (%d files)"
                % (os.path.relpath(state_path, REPO), len(files)))
            return 0

        say("\npublishing only the frozen validated source tree and checked Word attachment")

        # Update only what we manage. An earlier version wiped the checkout and
        # copied ours in, which would have deleted README_OVERLEAF.txt - a file
        # that exists only in the project. Anything Overleaf has that we do not
        # manage is left untouched.
        for r in to_push:
            dst = str(assets.safe_path(clone, r["remote"]))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(r["local"], dst)

        run(GIT + ["add", "--"] + [record["remote"] for record in to_push], cwd=clone, quiet=True)
        verify_staged_bytes(clone, [(record["local"], record["remote"]) for record in to_push])
        st = run(GIT + ["status", "--porcelain"], cwd=clone, quiet=True)
        if not st.stdout.strip():
            verify_local_release(paper, manifest)
            verify_remote_tip(clone, branch, remote_tip, env=auth_env)
            say("\nOverleaf already matches the local paper; nothing to push.")
        else:
            say("\nchanging in Overleaf:")
            for line in st.stdout.strip().splitlines():
                say("   %s" % line)
            run(GIT + ["-c", "user.email=sync@local", "-c", "user.name=paper-sync",
                       "commit", "-m", a.message], cwd=clone, quiet=True)
            verify_local_release(paper, manifest)
            verify_remote_tip(clone, branch, remote_tip, env=auth_env)
            run(GIT + ["push", "origin", "HEAD"], cwd=clone, quiet=True, env=auth_env)
            say("\npushed. Overleaf will recompile on next open.")

        # Record the agreement from the pushed bytes rather than a re-read of
        # the remote: the two are identical and this avoids a second round trip.
        pushed = {r["remote"]: r["local_sha256"] for r in to_push}
        files = {}
        for r in records:
            if r["remote"] in pushed:
                h = pushed[r["remote"]]
                files[r["remote"]] = {"remote_sha256": h, "local_sha256": h}
            elif r["status"] == AGREED_ABSENT:
                files[r["remote"]] = {"remote_sha256": None,
                                      "local_sha256": r["local_sha256"]}
            else:
                files[r["remote"]] = {"remote_sha256": r["remote_sha256"],
                                      "local_sha256": r["local_sha256"]}
        # Retain historical records for excluded attachments without touching
        # their remote bytes or asserting that they were checked this time.
        files = {**state.get("files", {}), **files}
        save_state(state_path, a.project_id, "push", files)
        say("recorded agreement in %s (%d files)"
            % (os.path.relpath(state_path, REPO), len(files)))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        say("REFUSING TO PUSH: " + str(exc))
        return 1
    finally:
        cleanup_clone(clone)


if __name__ == "__main__":
    sys.exit(main())
