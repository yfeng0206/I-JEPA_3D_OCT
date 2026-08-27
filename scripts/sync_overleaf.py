#!/usr/bin/env python
"""Push the validated submission straight into an Overleaf project.

Why
---
The current loop is: build the ZIP, read CHANGED.txt, then hand-upload two or
three files through the Overleaf web UI. That is the slowest and most
error-prone step left, and it has already misled once - CHANGED.txt reported
"nothing changed" immediately after a 269-line edit, because it diffed against
the last build rather than the last upload.

Overleaf exposes every project as a git remote, so the whole step can be one
command. This script pushes the SAME staged tree that p13_build_zip.py
validates, so what lands in Overleaf is byte-identical to what passed the gates.

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

Usage
-----
    python scripts/sync_overleaf.py --dry-run     # show what would be pushed
    python scripts/sync_overleaf.py --check       # test auth only, push nothing
    python scripts/sync_overleaf.py               # build, validate, push

The push is refused unless the build passes all of p13_build_zip.py's checks,
so a broken or over-length paper can never reach the project.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper", "genai4health2026")
VENV_PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"

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
}
INCLUDE_DIRS = ["auto", "figures"]


def run(cmd, cwd=None, check=True, quiet=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if not quiet and r.stdout.strip():
        print(r.stdout.rstrip())
    if check and r.returncode != 0:
        print(r.stderr.rstrip())
        raise SystemExit("command failed: %s" % " ".join(cmd))
    return r


def validate():
    """Refuse to publish anything the gates have not cleared."""
    py = VENV_PY if os.path.isfile(VENV_PY) else sys.executable
    r = subprocess.run([py, os.path.join("autopilot", "p13_build_zip.py")],
                       cwd=REPO, capture_output=True, text=True)
    out = r.stdout
    ok = "ALL_PASS = True" in out
    for line in out.splitlines():
        if "PASS" in line or "FAIL" in line or "main content" in line:
            print("  " + line.strip())
    if not ok:
        raise SystemExit("build did not pass; refusing to push to Overleaf")
    return True


def collect():
    items = []
    for local, remote in FILE_MAP.items():
        p = os.path.join(PAPER, local)
        if os.path.isfile(p):
            items.append((p, remote))
    for d in INCLUDE_DIRS:
        root = os.path.join(PAPER, d)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if fn.endswith((".aux", ".log", ".out", ".blg", ".synctex.gz", ".fls",
                                ".fdb_latexmk", ".toc")):
                    continue
                src = os.path.join(dirpath, fn)
                items.append((src, os.path.relpath(src, PAPER).replace("\\", "/")))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", default=os.environ.get("OVERLEAF_PROJECT_ID"))
    ap.add_argument("--token", default=os.environ.get("OVERLEAF_TOKEN"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="test auth only")
    ap.add_argument("--message", default="sync validated submission")
    a = ap.parse_args()

    items = collect()
    total = sum(os.path.getsize(s) for s, _ in items)
    print("files to publish: %d, %.2f MB" % (len(items), total / 1e6))
    for _s, rel in sorted(items, key=lambda x: x[1])[:12]:
        print("   %s" % rel)
    if len(items) > 12:
        print("   ... and %d more" % (len(items) - 12))

    if a.dry_run:
        print("\ndry run; nothing pushed")
        return 0

    if not a.project_id or not a.token:
        print("\nMissing credentials. Set OVERLEAF_PROJECT_ID and OVERLEAF_TOKEN,")
        print("or pass --project-id and --token.")
        print("Token: Overleaf Account Settings -> Git Integration -> Generate token.")
        print("Note: Overleaf git access is a premium feature; on a free plan this")
        print("will fail authentication and the loose-file mirror stays the fallback.")
        return 1

    remote = "https://git:%s@git.overleaf.com/%s" % (a.token, a.project_id)
    safe = "https://git:***@git.overleaf.com/%s" % a.project_id

    tmp = tempfile.mkdtemp(prefix="overleaf_")
    try:
        print("\ncloning %s" % safe)
        r = run(["git", "clone", "--depth", "1", remote, tmp], check=False, quiet=True)
        if r.returncode != 0:
            print(r.stderr.strip()[:400])
            print("\nClone failed. Most likely causes: wrong project id, wrong or")
            print("expired token, or a free plan without git integration.")
            return 1
        print("  clone OK - authentication works")
        if a.check:
            print("\n--check: authentication verified, nothing pushed")
            return 0

        print("\nvalidating before publish")
        validate()

        # Update only what we manage. An earlier version wiped the checkout and
        # copied ours in, which would have deleted README_OVERLEAF.txt - a file
        # that exists only in the project. Anything Overleaf has that we do not
        # manage is left untouched.
        for src, rel in items:
            dst = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        run(["git", "add", "-A"], cwd=tmp, quiet=True)
        st = run(["git", "status", "--porcelain"], cwd=tmp, quiet=True)
        if not st.stdout.strip():
            print("\nOverleaf already matches the local paper; nothing to push.")
            return 0
        print("\nchanged in Overleaf:")
        for line in st.stdout.strip().splitlines():
            print("   %s" % line)
        run(["git", "-c", "user.email=sync@local", "-c", "user.name=paper-sync",
             "commit", "-m", a.message], cwd=tmp, quiet=True)
        run(["git", "push", "origin", "HEAD"], cwd=tmp, quiet=True)
        print("\npushed. Overleaf will recompile on next open.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
