"""Browser-session (HTTP) transport for scripts/sync_overleaf.py.

Why
---
Overleaf's git bridge is a premium feature and the account that owns the paper
does not have it, so the git transport fails with "no git access". This module
reaches the same project through the HTTP endpoints the Overleaf web editor
itself uses, authenticated by the browser session that `ols login` stores in
%USERPROFILE%\\.overleaf\\.olauth. To create or renew that session, double-click
Downloads\\OVERLEAF_LOGIN_double_click.cmd and log in; Overleaf sessions expire,
and an expired one surfaces here as SessionError with the same instruction.

It is a transport and nothing else. Which files are managed, which differ, what
would be destroyed and what to record all stay in sync_overleaf.py. This module
is handed a list of uploads and deletions only after that three-way check has
passed, and its snapshot is fed to exactly the same comparison as a git clone.

Endpoints (each verified against a throwaway scratch project, 2026-10-15)
---------
GET    /project/<id>                       editor page; 200 means the session can
                                           open the project. Carries a fresh
                                           ol-csrfToken and ol-project_id.
GET    /project/<id>/download/zip          the snapshot. Docs come back as LF text,
                                           binary files byte-exact.
GET    /socket.io/1/?projectId=<id>        socket.io 0.9 handshake, then
GET    /socket.io/1/xhr-polling/<sid>      emits joinProjectResponse: the file tree
                                           with folder and entity ids
POST   /project/<id>/folder                {"name", "parent_folder_id"} -> folder;
                                           400 "file already exists" on duplicates
POST   /project/<id>/upload?folder_id=F    multipart qqfile + name. Replaces a
                                           same-named entity; a doc keeps its id,
                                           so history and comments survive. CRLF
                                           in a text doc is stored as LF.
DELETE /project/<id>/doc|file/<entity_id>  204; 404 if already gone
GET    /project/<id>/updates?min_count=N   project history, newest first, after
                                           flushing pending edits: fromV/toV,
                                           meta.users[].id, start_ts/end_ts,
                                           pathnames, project_ops; paged with
                                           before=<nextBeforeTimestamp>. Uploads and
                                           deletes appear at once, attributed to the
                                           session's own ol-user_id.
GET    /project/<id>/latest/history        latest history chunk; startVersion plus
                                           the number of its changes is the
                                           project's current version (matches the
                                           newest toV from /updates).
Mutating calls need the X-Csrf-Token header (403 without it). A request
without a valid session gets 403, never a redirect.

Rate limits: Overleaf answers 429 with no Retry-After. Its public source
limits zip-download to 10 per 60 s per project and file-upload to 500 per
15 min; a push downloads the ZIP three times (review, pre-write re-check,
post-push verification). On 429 the client waits out the window (15, 30, 60,
60 s), reporting each wait, before giving up.

No compare-and-swap
-------------------
Unlike a git push, an upload or delete is unconditional: Overleaf offers no
"only if unchanged" write. sync_overleaf.py therefore (1) re-downloads the
project immediately before the first write and aborts, with nothing written,
if any managed file moved since the conflict review, and (2) after the push
reads the project history from a version taken before that re-check, and
refuses to record the sync if any update in the window came from another user
(or from no identifiable user), or if the history cannot be read or does not
cover every version from that baseline to the project's current version
without a gap. That turns a
lost update into a detected one. It cannot prevent it, and it cannot tell the
operator's own browser edits (same account) from the script's. Operational
rule: ask co-authors not to edit, and do not edit yourself, while a push runs.
Overleaf History keeps every prior version, so a flagged edit can be restored
from the editor's History panel.

Safety
------
* The session file is a credential. It is read with an unpickler that refuses
  every global, so only plain dicts and strings load. Cookie and CSRF values
  are never printed, logged, written or put in a URL. Every value the client
  holds - loaded cookies, CSRF tokens and cookies the server refreshes - is
  registered (on_secret callback) and scrubbed from every exception the client
  raises. Server error text is only quoted when it is a short error code.
* Every request goes to https://www.overleaf.com with redirects disabled, so
  the cookie cannot be carried to another host.
* The real paper project refuses every mutating call unless the client was
  built with allow_real_project_write=True. sync_overleaf.py only does that for
  a --release-manifest push that also passes
  --i-understand-this-writes-the-real-project.
"""
import functools
import hashlib
import html
import io
import json
import os
from pathlib import Path
import pickle
import re
import time
from urllib.parse import unquote
import zipfile

HOST = "www.overleaf.com"
BASE = "https://" + HOST
REAL_PROJECT_ID = "6a8fb344bf38f34a9e545791"
# Same truncated SHA-256 that .overleaf_sync.json stores in "project".
REAL_PROJECT_FINGERPRINT = "868079f07aa2a3d7"
OLAUTH_DEFAULT = os.path.join(os.path.expanduser("~"), ".overleaf", ".olauth")
LOGIN_HINT = ("Log in again: double-click Downloads\\OVERLEAF_LOGIN_double_click.cmd, "
              "sign in with the account that owns the project, close the window once "
              "the project list loads, then re-run this command.")
COOKIE_NAMES = ("overleaf_session2", "GCLB")
PROJECT_ID = re.compile(r"[0-9a-f]{24}")
TIMEOUT = 120
MAX_ATTEMPTS = 4
# Overleaf answers 429 without Retry-After; its public limits are per minute
# (zip-download: 10 per 60 s per project; file-upload: 500 per 15 min), so
# waiting out a full window is enough.
RATE_LIMIT_WAITS = (15, 30, 60, 60)
MAX_POLLS = 8
MAX_HISTORY_PAGES = 10
HISTORY_PAGE = 50
SAFE_REASON = re.compile(r"[a-z][a-z0-9_]{0,59}")
MAX_ZIP_BYTES = 512 * 1024 * 1024
WRITE_ROLES = ("owner", "readAndWrite")


class OverleafHTTPError(Exception):
    """The HTTP transport could not complete a request safely."""


class SessionError(OverleafHTTPError):
    """No usable browser session: missing, unreadable, expired or forbidden."""


class RealProjectWriteRefused(OverleafHTTPError):
    """A mutating call targeted the real paper project without both gates."""


def fingerprint(project_id):
    return hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:16]


def is_real_project(project_id):
    """Mongo ids are case-insensitive hex, so compare a normalised form."""
    normalised = (project_id or "").strip().lower()
    return normalised == REAL_PROJECT_ID or fingerprint(normalised) == REAL_PROJECT_FINGERPRINT


class _PlainDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("session file may only hold plain data")


def load_session(path=None):
    """Return (cookies, csrf) from an `ols login` session file, in memory only."""
    path = Path(path or OLAUTH_DEFAULT)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise SessionError("no Overleaf session file at %s. %s" % (path, LOGIN_HINT)) from None
    except OSError as exc:
        raise SessionError("cannot read the Overleaf session file (%s)." % type(exc).__name__) from None
    try:
        store = _PlainDataUnpickler(io.BytesIO(raw)).load()
    except Exception:
        raise SessionError("the Overleaf session file is not a plain ols session. " + LOGIN_HINT) from None
    cookie = store.get("cookie") if isinstance(store, dict) else None
    if not isinstance(cookie, dict) or not isinstance(cookie.get("overleaf_session2"), str) \
            or not cookie["overleaf_session2"]:
        raise SessionError("the Overleaf session file holds no session cookie. " + LOGIN_HINT)
    cookies = {name: value for name, value in cookie.items()
               if name in COOKIE_NAMES and isinstance(value, str) and value}
    csrf = store.get("csrf")
    return cookies, (csrf if isinstance(csrf, str) and csrf else None)


def decode_frames(body):
    """Split a socket.io 0.9 polling payload into its messages."""
    if not body.startswith("\ufffd"):
        return [body] if body else []
    frames, index = [], 0
    while index < len(body):
        if body[index] != "\ufffd":
            raise OverleafHTTPError("malformed socket.io payload")
        end = body.index("\ufffd", index + 1)
        length = int(body[index + 1:end])
        frames.append(body[end + 1:end + 1 + length])
        index = end + 1 + length
    return frames


def index_tree(root_folder):
    """Flatten a joinProject rootFolder into path -> (kind, id) and folder -> id."""
    entities, folders = {}, {"": root_folder["_id"]}

    def walk(folder, prefix):
        for doc in folder.get("docs", []):
            entities[prefix + doc["name"]] = ("doc", doc["_id"])
        for ref in folder.get("fileRefs", []):
            entities[prefix + ref["name"]] = ("file", ref["_id"])
        for sub in folder.get("folders", []):
            path = prefix + sub["name"]
            folders[path] = sub["_id"]
            walk(sub, path + "/")

    walk(root_folder, "")
    return {"entities": entities, "folders": folders}


def extract_zip(data, dest):
    """Unpack a project ZIP into dest, refusing anything that could escape it.

    Returns the extracted relative paths. The tree is case-checked because two
    Overleaf names differing only in case would collide on Windows and one
    would silently shadow the other in the three-way comparison.
    """
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=False)
    root = root.resolve()
    names, folded, total = [], set(), 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            parts = name.split("/")
            if ("\\" in name or ":" in name or name.startswith("/")
                    or any(part in ("", ".", "..") for part in parts)):
                raise OverleafHTTPError("unsafe path in project ZIP: %r" % name)
            if name.lower() in folded:
                raise OverleafHTTPError("project ZIP has names differing only in case: %r" % name)
            folded.add(name.lower())
            total += info.file_size
            if total > MAX_ZIP_BYTES:
                raise OverleafHTTPError("project ZIP is implausibly large; refusing to unpack")
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise OverleafHTTPError("unsafe path in project ZIP: %r" % name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
            names.append(name)
    return sorted(names)


def _scrubbed(method):
    """The transport's error boundary.

    Transport errors are re-raised with every registered secret removed.
    Anything a malformed response can make a parser raise (bad JSON, missing
    keys, wrong types, a broken socket.io frame or ZIP) becomes a transport
    error naming only the operation and the exception type: the original
    message may quote response text and is never carried over.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except OverleafHTTPError as exc:
            self._register_jar()
            raise type(exc)(self._scrub(str(exc))) from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError, zipfile.BadZipFile) as exc:
            self._register_jar()
            raise OverleafHTTPError("malformed Overleaf response during %s (%s); Overleaf's API may "
                                    "have changed" % (method.__name__, type(exc).__name__)) from None
    return wrapper


class OverleafHTTP:
    """One Overleaf project reached through a logged-in browser session."""

    def __init__(self, project_id, cookies, csrf=None, *, allow_real_project_write=False,
                 session=None, sleep=time.sleep, notify=None, on_secret=None):
        if not isinstance(project_id, str) or not PROJECT_ID.fullmatch(project_id):
            raise ValueError("an Overleaf project id is 24 lowercase hex digits")
        self.project_id = project_id
        self.user_id = None
        self._allow_real = bool(allow_real_project_write)
        self._csrf = csrf
        self._secrets = []
        self._on_secret = on_secret or (lambda value: None)
        for value in list(cookies.values()) + [csrf]:
            self._register(value)
        self._sleep = sleep
        self._notify = notify or (lambda message: None)
        if session is None:
            import requests
            session = requests.Session()
        self._http = session
        self._http.headers["User-Agent"] = "jepa-sync-overleaf/1.0"
        for name, value in cookies.items():
            # Overleaf re-issues its session cookie for .overleaf.com; using the
            # same domain lets that refresh replace this copy instead of
            # duplicating it. _request() never contacts any other host.
            self._http.cookies.set(name, value, domain=".overleaf.com", path="/", secure=True)

    # ------------------------------------------------------------- plumbing

    def secrets(self):
        """Values the caller must scrub from anything it prints."""
        self._register_jar()
        return list(self._secrets)

    def _register(self, value):
        if not isinstance(value, str) or len(value) < 4:
            return
        for form in {value, unquote(value)}:
            if form not in self._secrets:
                self._secrets.append(form)
                self._on_secret(form)

    def _register_jar(self):
        """Cookies the server set or refreshed are credentials too."""
        for cookie in list(self._http.cookies):
            self._register(cookie.value)

    def _scrub(self, text):
        for value in sorted(self._secrets, key=len, reverse=True):
            text = text.replace(value, "<REDACTED>")
        return text

    @property
    def is_real_project(self):
        return is_real_project(self.project_id)

    def _require_write(self):
        if self.is_real_project and not self._allow_real:
            raise RealProjectWriteRefused(
                "refusing to modify the real paper project: a write needs both "
                "--release-manifest and --i-understand-this-writes-the-real-project")
        if not self._csrf:
            raise SessionError("no CSRF token for a write; open the project first. " + LOGIN_HINT)

    def _request(self, method, path, *, expect=(200,), retry=None, **kwargs):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("request path must be site-relative")
        import requests
        kwargs.setdefault("timeout", TIMEOUT)
        kwargs["allow_redirects"] = False
        retry = method == "GET" if retry is None else retry
        transient = limited = 0
        while True:
            try:
                response = self._http.request(method, BASE + path, **kwargs)
            except requests.RequestException as exc:
                transient += 1
                if not retry or transient >= MAX_ATTEMPTS:
                    raise OverleafHTTPError("%s %s failed: %s" % (method, path, type(exc).__name__)) from None
                self._sleep(2 ** transient * 2)
                continue
            # 429 means the request was not processed, so even a write may retry.
            if response.status_code == 429 and limited < len(RATE_LIMIT_WAITS):
                wait = response.headers.get("Retry-After", "")
                delay = min(int(wait), 300) if wait.isdigit() else RATE_LIMIT_WAITS[limited]
                limited += 1
                self._notify("  Overleaf rate limit on %s %s; waiting %d s before retrying"
                             % (method, path, delay))
                self._sleep(delay)
                continue
            if retry and response.status_code in (502, 503, 504):
                transient += 1
                if transient < MAX_ATTEMPTS:
                    self._sleep(2 ** transient * 2)
                    continue
            break
        self._register_jar()
        status = response.status_code
        if 300 <= status < 400:
            if "/login" in response.headers.get("Location", ""):
                raise SessionError("the Overleaf session has expired. " + LOGIN_HINT)
            raise OverleafHTTPError("%s %s was redirected (HTTP %d); not following" % (method, path, status))
        if status in (401, 403):
            raise SessionError("Overleaf refused the session (HTTP %d): the login expired, the CSRF "
                               "token was rejected, or this account cannot open the project. %s"
                               % (status, LOGIN_HINT))
        if status == 429:
            raise OverleafHTTPError("Overleaf rate limit (HTTP 429) on %s %s; wait and re-run" % (method, path))
        if status not in expect:
            raise OverleafHTTPError("%s %s returned HTTP %d%s" % (method, path, status, _reason(response)))
        return response

    def _json_headers(self):
        return {"X-Csrf-Token": self._csrf, "Accept": "application/json"}

    # --------------------------------------------------------------- reading

    @_scrubbed
    def open_project(self):
        """Prove the session can open this project; take a fresh CSRF token and our user id."""
        response = self._request("GET", "/project/%s" % self.project_id)
        page = response.text
        shown = re.search(r'<meta\s+name="ol-project_id"\s+content="([^"]*)"', page)
        if not shown or html.unescape(shown.group(1)) != self.project_id:
            raise OverleafHTTPError("the editor page did not identify the requested project; "
                                    "Overleaf's page layout may have changed")
        token = re.search(r'<meta\s+name="ol-csrfToken"\s+content="([^"]+)"', page)
        if token:
            self._csrf = html.unescape(token.group(1))
            self._register(self._csrf)
        user = re.search(r'<meta\s+name="ol-user_id"\s+content="([0-9a-f]{24})"', page)
        self.user_id = user.group(1) if user else None
        return True

    @_scrubbed
    def download_zip(self):
        response = self._request("GET", "/project/%s/download/zip" % self.project_id)
        data = response.content
        if "zip" not in response.headers.get("Content-Type", "") or not zipfile.is_zipfile(io.BytesIO(data)):
            raise OverleafHTTPError("the project download was not a ZIP archive")
        return data

    @_scrubbed
    def snapshot(self, dest):
        """Download the project and unpack it like a fresh checkout."""
        return extract_zip(self.download_zip(), dest)

    @_scrubbed
    def file_tree(self):
        """Folder and entity ids, via the editor's real-time joinProject.

        Overleaf only exposes ids through the editor's socket.io channel. The
        server joins the project named in the handshake and emits
        joinProjectResponse; the xhr-polling transport reads it with plain
        HTTP. GCLB stickiness set by earlier responses keeps both requests on
        the same backend.
        """
        params = lambda: {"t": str(int(time.time() * 1000)), "projectId": self.project_id}
        handshake = self._request("GET", "/socket.io/1/", params=params()).text.split(":")
        if len(handshake) < 4 or not re.fullmatch(r"[A-Za-z0-9_-]+", handshake[0]) \
                or "xhr-polling" not in handshake[3].split(","):
            raise OverleafHTTPError("unexpected socket.io handshake; Overleaf's real-time protocol may have changed")
        poll = "/socket.io/1/xhr-polling/%s" % handshake[0]
        try:
            for _ in range(MAX_POLLS):
                body = self._request("GET", poll, params=params(), retry=False).text
                for frame in decode_frames(body):
                    kind = frame.split(":", 1)[0]
                    if kind == "2":
                        self._request("POST", poll, params=params(), data="2::", retry=False)
                    elif kind == "0":
                        raise OverleafHTTPError("real-time service disconnected before sending the file tree")
                    elif kind == "7":
                        raise OverleafHTTPError("real-time service error while joining the project")
                    elif kind == "5":
                        event = json.loads(frame.split(":", 3)[3])
                        name = event.get("name")
                        if name == "joinProjectResponse":
                            return self._tree_from(event.get("args") or [{}])
                        if name in ("connectionRejected", "forceDisconnect"):
                            raise SessionError("real-time service rejected the session (%s). %s"
                                               % (name, LOGIN_HINT))
            raise OverleafHTTPError("no joinProjectResponse after %d polls; Overleaf's real-time "
                                    "protocol may have changed" % MAX_POLLS)
        finally:
            try:
                self._http.post(BASE + poll, params=params(), data="0::", timeout=30, allow_redirects=False)
            except Exception:
                pass

    def _tree_from(self, args):
        payload = args[0] if args and isinstance(args[0], dict) else {}
        project = payload.get("project") or {}
        if project.get("_id") != self.project_id:
            raise OverleafHTTPError("joinProject answered for a different project")
        tree = index_tree(project["rootFolder"][0])
        tree["permissions"] = payload.get("permissionsLevel")
        return tree

    # --------------------------------------------------------------- history

    def _updates_page(self, before, count):
        params = {"min_count": str(count)}
        if before is not None:
            params["before"] = str(before)
        body = self._request("GET", "/project/%s/updates" % self.project_id, params=params,
                             headers={"Accept": "application/json"}).json()
        updates = body.get("updates") if isinstance(body, dict) else None
        if not isinstance(updates, list) or not all(_valid_update(u) for u in updates):
            raise OverleafHTTPError("unexpected project-history response; Overleaf's history API may have changed")
        cursor = body.get("nextBeforeTimestamp")
        if cursor is not None and type(cursor) is not int:
            raise OverleafHTTPError("unexpected project-history cursor")
        return updates, cursor

    @_scrubbed
    def history_head(self):
        """Latest project-history version. The /updates call flushes pending edits first."""
        updates, _before = self._updates_page(None, 1)
        return max((u["toV"] for u in updates), default=0)

    @_scrubbed
    def current_version(self):
        """The project's current history version, from the latest history chunk."""
        body = self._request("GET", "/project/%s/latest/history" % self.project_id,
                             headers={"Accept": "application/json"}).json()
        chunk = body["chunk"]
        start, changes = chunk["startVersion"], chunk["history"]["changes"]
        if type(start) is not int or start < 0 or not isinstance(changes, list):
            raise OverleafHTTPError("unexpected latest-history response; Overleaf's history API may have changed")
        return start + len(changes)

    @_scrubbed
    def updates_since(self, version):
        """Every summarised history update newer than `version`, proven complete.

        The returned version intervals [fromV, toV] must chain without a gap
        from `version` up to the newest version listed, and that newest
        version must equal the project's current version. Reaching an old row
        is not enough: a row skipped by pagination (for example one sharing
        the page-boundary timestamp) or missing from a page would leave a gap,
        and an unseen foreign edit could hide in it. The cursor is applied
        inclusively, so boundary rows are re-read; duplicates are dropped.
        Anything short of complete coverage raises.
        """
        rows, before = {}, None
        for _ in range(MAX_HISTORY_PAGES):
            page, cursor = self._updates_page(before, HISTORY_PAGE)
            new = [u for u in page if (u["fromV"], u["toV"]) not in rows]
            for u in new:
                rows[(u["fromV"], u["toV"])] = u
            head = max((to for _f, to in rows), default=version)
            if _chain_reaches(rows, version, head) or cursor is None:
                break
            if new:
                before = cursor + 1
            elif before != cursor:
                # The boundary rows alone filled the page; step past them.
                # Anything skipped this way shows up as a gap below.
                before = cursor
            else:
                break
        head = max((to for _f, to in rows), default=version)
        if head < version:
            raise OverleafHTTPError("project history ends at v%d, before the baseline v%d" % (head, version))
        if not _chain_reaches(rows, version, head):
            raise OverleafHTTPError("project history has a gap between v%d and v%d; a concurrent edit "
                                    "could not be ruled out" % (version, head))
        current = self.current_version()
        if current != head:
            raise OverleafHTTPError("project history lists up to v%d but the project is at v%d" % (head, current))
        return sorted((u for u in rows.values() if u["toV"] > version), key=lambda u: -u["toV"])

    def foreign(self, update):
        """True unless the update is attributed only to this session's user."""
        ids = [user.get("id") if isinstance(user, dict) else user
               for user in (update.get("meta") or {}).get("users") or []]
        return not self.user_id or not ids or any(i != self.user_id for i in ids)

    # --------------------------------------------------------------- writing

    @_scrubbed
    def ensure_folder(self, folder, tree):
        """Folder id for a project-relative folder path, creating what is missing."""
        if folder in tree["folders"]:
            return tree["folders"][folder]
        if folder in tree["entities"]:
            raise OverleafHTTPError("cannot create folder %s: a file of that name exists" % folder)
        parent, _, name = folder.rpartition("/")
        parent_id = self.ensure_folder(parent, tree)
        self._require_write()
        response = self._request("POST", "/project/%s/folder" % self.project_id,
                                 json={"name": name, "parent_folder_id": parent_id},
                                 headers=self._json_headers())
        created = response.json()
        if created.get("name") != name or not PROJECT_ID.fullmatch(str(created.get("_id", ""))):
            raise OverleafHTTPError("unexpected folder-creation response for %s" % folder)
        tree["folders"][folder] = created["_id"]
        return created["_id"]

    @_scrubbed
    def upload(self, rel, data, tree):
        """Create or replace one file; Overleaf decides doc versus binary file."""
        self._require_write()
        if rel in tree["folders"]:
            raise OverleafHTTPError("cannot upload %s: a folder of that name exists" % rel)
        folder, _, name = rel.rpartition("/")
        folder_id = self.ensure_folder(folder, tree)
        response = self._request(
            "POST", "/project/%s/upload" % self.project_id, params={"folder_id": folder_id},
            data={"name": name}, files={"qqfile": (name, data, "application/octet-stream")},
            headers={"X-CSRF-TOKEN": self._csrf, "Accept": "application/json"})
        body = response.json()
        if body.get("success") is not True or body.get("entity_type") not in ("doc", "file"):
            raise OverleafHTTPError("Overleaf rejected the upload of %s%s" % (rel, _safe_reason(body)))
        tree["entities"][rel] = (body["entity_type"], body.get("entity_id"))
        return body

    @_scrubbed
    def delete(self, rel, tree):
        """Delete one doc or file by path, using the id from file_tree()."""
        self._require_write()
        entry = tree["entities"].get(rel)
        if entry is None:
            raise OverleafHTTPError("cannot delete %s: not in the project's file tree" % rel)
        kind, entity_id = entry
        if kind not in ("doc", "file") or not PROJECT_ID.fullmatch(str(entity_id)):
            raise OverleafHTTPError("cannot delete %s: unexpected entity record" % rel)
        self._request("DELETE", "/project/%s/%s/%s" % (self.project_id, kind, entity_id),
                      expect=(200, 204), headers=self._json_headers())
        del tree["entities"][rel]


def _valid_update(update):
    """One summarised history row with a sane version interval and attribution list."""
    if not isinstance(update, dict) or not isinstance(update.get("meta"), dict):
        return False
    low, high, paths = update.get("fromV"), update.get("toV"), update.get("pathnames", [])
    return (type(low) is int and type(high) is int and 0 <= low < high
            and isinstance(update["meta"].get("users"), list)
            and isinstance(paths, list) and all(isinstance(p, str) for p in paths))


def _chain_reaches(intervals, low, high):
    """Do the (fromV, toV) intervals cover every version from low to high without a gap?"""
    reach = low
    for start, end in sorted(intervals):
        if start > reach:
            break
        reach = max(reach, end)
    return reach >= high


def _reason(response):
    """A server-supplied error code, if it is one; never free text or request data."""
    try:
        body = response.json()
    except Exception:
        return ""
    return _safe_reason(body)


def _safe_reason(body):
    """Quote only short identifier-like codes (e.g. invalid_filename).

    Arbitrary server text could echo request headers or cookies, so it is
    dropped rather than trusted to the scrubber alone.
    """
    if isinstance(body, dict):
        for key in ("error", "message"):
            value = body.get(key)
            if isinstance(value, str) and SAFE_REASON.fullmatch(value):
                return ": " + value
    return ""
