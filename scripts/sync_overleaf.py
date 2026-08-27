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
command. This script pushes the SAME staged tree that p13_build_zip.py
validates, so what lands in Overleaf is byte-identical to what passed the gates.

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
it are --force (push anyway, with a loud warning) or --pull (take Overleaf's
copy instead).

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

Configure either by environment variable or by flag:

    setx OVERLEAF_PROJECT_ID  <project id>
    setx OVERLEAF_TOKEN       <token>

The token is never printed. It is scrubbed out of every line of git output this
script emits, including failure messages.

Usage
-----
    python scripts/sync_overleaf.py --check          # test auth only
    python scripts/sync_overleaf.py --dry-run        # classify, push nothing
    python scripts/sync_overleaf.py                  # validate, then push
    python scripts/sync_overleaf.py --pull           # show what Overleaf would
                                                     # change locally
    python scripts/sync_overleaf.py --pull --yes     # actually write it locally
    python scripts/sync_overleaf.py --force          # overwrite remote edits

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
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def run(cmd, cwd=None, check=True, quiet=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
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
        try:
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass
    if os.path.exists(path):
        shutil.rmtree(path, onerror=onerror)


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
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(st, f, indent=1, sort_keys=False)
        f.write("\n")
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

def validate():
    """Refuse to publish anything the gates have not cleared."""
    py = VENV_PY if os.path.isfile(VENV_PY) else sys.executable
    r = subprocess.run([py, os.path.join("autopilot", "p13_build_zip.py")],
                       cwd=REPO, capture_output=True, text=True)
    out = r.stdout
    ok = "ALL_PASS = True" in out
    for line in out.splitlines():
        if "PASS" in line or "FAIL" in line or "main content" in line:
            say("  " + line.strip())
    if not ok:
        raise SystemExit("build did not pass; refusing to push to Overleaf")
    return True


def warn_if_docx_stale(paper):
    """Say so if the Word copy predates the manuscript it was rendered from.

    The .docx is a convenience for collaborators who do not write LaTeX, and a
    stale one circulating among them is worse than none, because its numbers
    look as authoritative as the current ones. This does not block the push:
    the .tex is what gets submitted, and refusing to publish a corrected
    manuscript because a derived Word file lagged would be the wrong trade.
    """
    tex = os.path.join(paper, "main_submission.tex")
    docx = os.path.join(paper, "main_submission.docx")
    if not os.path.exists(docx):
        say("  note: no Word copy yet; build one with autopilot/make_docx.py")
        return
    if os.path.getmtime(docx) < os.path.getmtime(tex):
        say("  WARNING: main_submission.docx is older than the .tex it renders.")
        say("           Rebuild it: python autopilot/make_docx.py")
    else:
        say("  Word copy is newer than the manuscript")


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
        say("      resurrect them. Use --force to push them anyway.")

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

def main():
    ap = argparse.ArgumentParser(
        description="Two-way, conflict-aware Overleaf sync.")
    ap.add_argument("--project-id", default=os.environ.get("OVERLEAF_PROJECT_ID"))
    ap.add_argument("--token", default=os.environ.get("OVERLEAF_TOKEN"))
    ap.add_argument("--dry-run", action="store_true",
                    help="classify everything and push nothing")
    ap.add_argument("--check", action="store_true", help="test auth only")
    ap.add_argument("--pull", action="store_true",
                    help="copy Overleaf's managed files back into the paper dir")
    ap.add_argument("--yes", action="store_true",
                    help="confirm a --pull that would change local files")
    ap.add_argument("--force", action="store_true",
                    help="push over remote edits (never the default)")
    ap.add_argument("--paper-dir", default=PAPER_DEFAULT,
                    help="local paper directory (default: paper/genai4health2026)")
    ap.add_argument("--state", default=STATE_DEFAULT,
                    help="sync state file (default: .overleaf_sync.json)")
    ap.add_argument("--message", default="sync validated submission")
    a = ap.parse_args()

    if a.token:
        _SECRETS.append(a.token)
    paper = os.path.abspath(a.paper_dir)
    state_path = os.path.abspath(a.state)

    if a.pull and a.force:
        say("--pull and --force are opposites; pick one.")
        return 1
    if not os.path.isdir(paper):
        say("no such paper directory: %s" % paper)
        return 1

    items = collect(paper)
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

    if not a.project_id or not a.token:
        say("\nMissing credentials. Set OVERLEAF_PROJECT_ID and OVERLEAF_TOKEN,")
        say("or pass --project-id and --token.")
        say("Token: Overleaf Account Settings -> Git Integration -> Generate token.")
        say("Note: Overleaf git access is a premium feature; on a free plan this")
        say("will fail authentication and the loose-file mirror stays the fallback.")
        return 1

    remote_url = "https://git:%s@git.overleaf.com/%s" % (a.token, a.project_id)
    safe_url = "https://git:<REDACTED>@git.overleaf.com/%s" % a.project_id

    # The scratch checkout lives inside the repository (gitignored by _tmp_*)
    # rather than in the system temp directory, so a crash leaves the evidence
    # next to the work instead of somewhere nobody will look.
    clone = os.path.join(REPO, "_tmp_overleaf_clone")
    rmtree_force(clone)
    try:
        say("\ncloning %s" % safe_url)
        r = run(GIT + ["clone", "--depth", "1", remote_url, clone],
                check=False, quiet=True)
        if r.returncode != 0:
            say(r.stderr.strip()[:400])
            say("\nClone failed. Most likely causes: wrong project id, wrong or")
            say("expired token, or a free plan without git integration.")
            return 1
        say("  clone OK - authentication works")
        # Persist the setting so add/commit agree with what was checked out.
        run(["git", "config", "core.autocrlf", "input"], cwd=clone, quiet=True)
        run(["git", "config", "core.eol", "lf"], cwd=clone, quiet=True)
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
        to_push = [r for r in records if r["status"] in (ADD, UPDATE)]

        if a.force and conflicts:
            say("\n" + "!" * 68)
            say("!! --force: these files will be OVERWRITTEN in Overleaf. Any edit")
            say("!! made in the Overleaf editor since the last sync is LOST:")
            for r in conflicts:
                say("!!   %s" % r["remote"])
            say("!" * 68)
            to_push += conflicts

        # --dry-run reports and never fails. It writes nothing in either
        # direction, so giving it the refusal exit code would only make it
        # useless inside a shell that stops on error.
        if a.dry_run:
            say("\ndry run; nothing pushed, nothing written locally.")
            if conflicts and not a.force:
                say("a real run would REFUSE: %d file(s) changed in Overleaf "
                    "since the last sync (exit 2)." % len(conflicts))
                say("%d other file(s) would have been written." % len(to_push))
            else:
                say("%d file(s) would be written to Overleaf." % len(to_push))
            return 0

        if conflicts and not a.force:
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
            say("  --force         keep ours and overwrite Overleaf")
            return 2

        if not to_push:
            say("\nOverleaf already matches the local paper; nothing to push.")
            files = {r["remote"]: {"remote_sha256": r["remote_sha256"],
                                   "local_sha256": r["local_sha256"]}
                     for r in records}
            save_state(state_path, a.project_id, "verify", files)
            say("recorded agreement in %s (%d files)"
                % (os.path.relpath(state_path, REPO), len(files)))
            return 0

        say("\nvalidating before publish")
        validate()
        warn_if_docx_stale(paper)

        # Update only what we manage. An earlier version wiped the checkout and
        # copied ours in, which would have deleted README_OVERLEAF.txt - a file
        # that exists only in the project. Anything Overleaf has that we do not
        # manage is left untouched.
        for r in to_push:
            dst = os.path.join(clone, r["remote"].replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(r["local"], dst)

        run(GIT + ["add", "-A"], cwd=clone, quiet=True)
        st = run(GIT + ["status", "--porcelain"], cwd=clone, quiet=True)
        if not st.stdout.strip():
            say("\nOverleaf already matches the local paper; nothing to push.")
        else:
            say("\nchanging in Overleaf:")
            for line in st.stdout.strip().splitlines():
                say("   %s" % line)
            run(GIT + ["-c", "user.email=sync@local", "-c", "user.name=paper-sync",
                       "commit", "-m", a.message], cwd=clone, quiet=True)
            run(GIT + ["push", "origin", "HEAD"], cwd=clone, quiet=True)
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
        save_state(state_path, a.project_id, "push", files)
        say("recorded agreement in %s (%d files)"
            % (os.path.relpath(state_path, REPO), len(files)))
        return 0
    finally:
        rmtree_force(clone)


if __name__ == "__main__":
    sys.exit(main())
