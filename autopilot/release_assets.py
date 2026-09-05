"""Small shared primitives for release staging and source coverage.

Hashes establish identity, not scientific validity. A release manifest is the
commit marker for a set of individually atomic file replacements.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper" / "genai4health2026"
NUMBER = re.compile(r"(?<![\d.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def unique_work(root=None, prefix="release"):
    root = Path(root) if root else REPO / "_release_work"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (prefix + "-" + uuid.uuid4().hex)
    path.mkdir()
    return path


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("path escapes source tree: " + str(relative))
    return path


def uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def group(text, start):
    """Read a balanced TeX/BibTeX brace group; return its contents and end."""
    if start >= len(text) or text[start] != "{":
        raise ValueError("expected brace group")
    depth, i = 1, start + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise ValueError("unclosed brace group")


def macros(text):
    """Return complete bodies (including nested formatting) and source spans."""
    out = []
    text = uncomment(text)
    pattern = r"\\(?:newcommand|renewcommand|providecommand)\*?\s*\{\\([A-Za-z]+)\}\s*(?:\[[^\]]*\]\s*)?"
    for match in re.finditer(pattern, text):
        body, end = group(text, match.end())
        out.append((match.group(1), body, match.start(), end))
    return out


def without_definitions(text):
    text = uncomment(text)
    for _, _, start, end in reversed(macros(text)):
        text = text[:start] + " " * (end - start) + text[end:]
    return text


def source_tree(paper, main=None):
    """Read all literal input/include dependencies, rejecting missing/cyclic inputs."""
    paper = Path(paper).resolve()
    main = main or ("main_submission.tex" if (paper / "main_submission.tex").exists() else "main.tex")
    files = {}

    def read(path, stack):
        path = path.resolve()
        if not path.is_relative_to(paper):
            raise ValueError("TeX input escapes source tree: " + str(path))
        if path in stack:
            raise ValueError("cyclic TeX input: " + str(path))
        text = uncomment(path.read_text(encoding="utf-8"))
        files[path.relative_to(paper).as_posix()] = text

        def include(match):
            rel = match.group(1).strip()
            if "\\" in rel or "#" in rel:
                raise ValueError("dynamic TeX input needs explicit support: " + rel)
            candidate = safe_path(paper, rel)
            if not candidate.suffix:
                candidate = candidate.with_suffix(".tex")
            if not candidate.is_file():
                raise FileNotFoundError("missing TeX input: " + str(candidate))
            return read(candidate, stack + [path])
        return re.sub(r"\\(?:input|include)\s*\{([^}]+)\}", include, text)

    return read(safe_path(paper, main), []), files


def expanded_body(text):
    definitions = {name: value for name, value, _, _ in macros(text)}
    body = without_definitions(text)
    # Parameterised formatting definitions cannot be safely expanded as scalars.
    definitions = {key: val for key, val in definitions.items() if "#" not in val}
    for _ in range(len(definitions) + 1):
        result = re.sub(r"\\([A-Za-z]+)\b",
                        lambda m: definitions.get(m.group(1), m.group(0)), body)
        if result == body:
            return result
        body = result
    raise ValueError("cyclic scalar macro expansion")


def graphics(text):
    folded = re.sub(r"(?<!\\)%[^\n]*\n\s*", "", text)
    return sorted(set(re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", folded)))


def resolve_graphic(paper, name):
    for prefix in ("", "figures", "auto"):
        for ext in ("", ".png", ".pdf", ".jpg", ".jpeg"):
            candidate = safe_path(paper, str(Path(prefix) / (name + ext)))
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("missing graphic: " + name)


def input_hashes(paper):
    text, files = source_tree(paper)
    paths = set(files)
    paths.add("references.bib")
    if (Path(paper) / "neurips_2026.sty").exists():
        paths.add("neurips_2026.sty")
    for name in graphics(text):
        paths.add(resolve_graphic(paper, name).relative_to(Path(paper).resolve()).as_posix())
    return {rel: sha256(safe_path(paper, rel)) for rel in sorted(paths)}


def asset_inventory(paper, registry=None):
    registry = Path(registry) if registry else Path(__file__).with_suffix(".json")
    declared = json.loads(registry.read_text(encoding="utf-8"))
    source, _ = source_tree(paper)
    rows = []
    for name in graphics(source):
        path = resolve_graphic(paper, name)
        rel = path.relative_to(Path(paper).resolve()).as_posix()
        digest = sha256(path)
        if rel in declared["producers"]:
            row = {"path": rel, "sha256": digest, "kind": "available_generator",
                   **declared["producers"][rel], "identity_status": "generated_input_hashed"}
            expected = row.get("expected_output_sha256")
            if expected is not None and digest != expected:
                row["identity_status"] = "pinned_mismatch"
        elif rel in declared["fixed_inputs"]:
            expected = declared["fixed_inputs"][rel]
            row = {"path": rel, "sha256": digest, "kind": "fixed_external_input",
                   "expected_sha256": expected,
                   "identity_status": "pinned_match" if digest == expected else "pinned_mismatch"}
        else:
            row = {"path": rel, "sha256": digest, "kind": "undeclared",
                   "identity_status": "undeclared"}
        rows.append(row)
    return {"registry_sha256": sha256(registry), "items": rows,
            "ALL_PASS": all(row["identity_status"] not in ("undeclared", "pinned_mismatch") for row in rows)}


def assert_unchanged(paper, snapshot):
    if input_hashes(paper) != snapshot:
        raise ValueError("release inputs changed during validation; rebuild required")


def copy_verified(source, target, expected):
    shutil.copyfile(source, target)
    if sha256(target) != expected:
        raise ValueError("copied input differs from captured snapshot: " + str(source))


def copy_snapshot(paper, stage, snapshot, *, rename_main=False):
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    expected = {}
    for rel, digest in snapshot.items():
        remote = "main.tex" if rename_main and rel == "main_submission.tex" else rel
        target = safe_path(stage, remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_verified(safe_path(paper, rel), target, digest)
        expected[remote] = digest
    assert_unchanged(stage, expected)
    return expected


def stage_numeric_review(paper, work, review_file=None):
    source = Path(review_file) if review_file else Path(paper) / "numeric_reviews.json"
    if not source.exists():
        if review_file:
            raise FileNotFoundError("explicit numeric review input is missing: " + str(source))
        return None
    source = source.resolve()
    digest = sha256(source)
    archived = Path(work) / "evidence" / "numeric_reviews.json"
    archived.parent.mkdir(parents=True, exist_ok=True)
    copy_verified(source, archived, digest)
    receipt = {
        "source": str(source), "archived": str(archived), "sha256": digest,
        "encoding": "base64", "content": base64.b64encode(archived.read_bytes()).decode("ascii"),
        "scope": "Exact QA input archived in the release manifest; not an automatic review approval "
                 "and not part of the anonymous source ZIP or Overleaf upload.",
    }
    verify_numeric_review(receipt)
    return receipt


def verify_numeric_review(receipt, reported_hash=...):
    if receipt is None:
        if reported_hash is not ... and reported_hash is not None:
            raise ValueError("numeric gate used an unarchived review input")
        return
    digest = receipt["sha256"]
    if reported_hash is not ... and reported_hash != digest:
        raise ValueError("numeric gate review hash differs from the staged review input")
    if receipt.get("encoding") != "base64":
        raise ValueError("unsupported numeric review archive encoding")
    content = base64.b64decode(receipt["content"], validate=True)
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("embedded numeric review archive is corrupted")
    for name in ("source", "archived"):
        if sha256(receipt[name]) != digest:
            raise ValueError("numeric review input changed: " + name)


def check_word_conflict(target, expected_hash=None, receipt=None):
    target = Path(target)
    if not target.exists():
        return None
    if expected_hash is None and receipt and Path(receipt).exists():
        expected_hash = json.loads(Path(receipt).read_text(encoding="utf-8")).get("docx_sha256")
    observed = sha256(target)
    if not expected_hash or observed != expected_hash:
        raise ValueError("Word edit conflict or untracked Word copy: preserve it and supply "
                         "--expected-docx-sha256 only after reviewing its contents")
    return observed


def aux_labels(path):
    text = Path(path).read_text(encoding="utf-8")
    labels = {}
    for match in re.finditer(r"\\newlabel\{([^}]+)\}\s*", text):
        fields, _ = group(text, match.end())
        number, _ = group(fields, 0)
        labels[match[1]] = number
    return labels


def source_refs(text, labels):
    refs = []
    for index, match in enumerate(re.finditer(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", text)):
        if match[1] not in labels:
            raise ValueError("reference missing from compiled aux: " + match[1])
        refs.append({"bookmark": "jeparef%04d" % index, "key": match[1],
                     "number": labels[match[1]], "command": match[0]})
    return refs


class PromotionError(RuntimeError):
    def __init__(self, cause, recovery_paths, rollback_errors=()):
        self.recovery_paths = recovery_paths
        self.rollback_errors = list(rollback_errors)
        message = "promotion failed: %s" % cause
        if rollback_errors:
            message += "; rollback failures: " + "; ".join(rollback_errors)
        if recovery_paths:
            message += "; recovery retained at: " + ", ".join(recovery_paths)
        super().__init__(message)


class ExclusiveFile:
    """Windows file handle denying noncooperating write/delete access.

    Renaming by this handle keeps the claim on the same file identity. Keeping
    installed handles open until the manifest is installed also protects the
    transaction from intervening editor writes. No advisory-lock fallback.
    """
    def __init__(self, path):
        if os.name != "nt":
            raise RuntimeError("safe release replacement currently requires Windows sharing semantics")
        import ctypes
        from ctypes import wintypes
        import msvcrt
        self.path = Path(path)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        handle = create(str(self.path), 0x80000000 | 0x10000, 1, None, 3, 0x00200000 | 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            class AttributeTagInfo(ctypes.Structure):
                _fields_ = [("attributes", wintypes.DWORD), ("tag", wintypes.DWORD)]
            attributes = AttributeTagInfo()
            inspect = kernel.GetFileInformationByHandleEx
            inspect.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
            inspect.restype = wintypes.BOOL
            if not inspect(handle, 9, ctypes.byref(attributes), ctypes.sizeof(attributes)):
                raise ctypes.WinError(ctypes.get_last_error())
            if attributes.attributes & 0x400:
                raise ValueError("refusing to replace a reparse-point destination")
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            close = kernel.CloseHandle
            close.argtypes = [wintypes.HANDLE]
            close(handle)
            raise
        self.stream = os.fdopen(fd, "rb")
        self.handle = handle

    def digest(self):
        self.stream.seek(0)
        return hashlib.file_digest(self.stream, "sha256").hexdigest()

    def close(self):
        self.stream.close()


def _rename_locked(file, target):
    """Atomic no-overwrite rename of the locked file, not a pathname relookup."""
    import ctypes
    from ctypes import wintypes

    class RenameInfo(ctypes.Structure):
        _fields_ = [("Flags", wintypes.DWORD), ("RootDirectory", wintypes.HANDLE),
                    ("FileNameLength", wintypes.DWORD), ("FileName", wintypes.WCHAR * 1)]

    target = Path(target).absolute()
    name = str(target).encode("utf-16-le")
    buffer = ctypes.create_string_buffer(RenameInfo.FileName.offset + len(name) + 2)
    info = RenameInfo.from_buffer(buffer)
    info.Flags = 0  # No REPLACE_IF_EXISTS: a new editor-created path wins.
    info.RootDirectory = None
    info.FileNameLength = len(name)
    ctypes.memmove(ctypes.addressof(buffer) + RenameInfo.FileName.offset, name, len(name))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    rename = kernel.SetFileInformationByHandle
    rename.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    rename.restype = wintypes.BOOL
    if not rename(file.handle, 3, buffer, len(buffer)):
        raise ctypes.WinError(ctypes.get_last_error())
    file.path = target


def _rollback_entry(entry):
    failures = []
    if entry["installed"]:
        try:
            _rename_locked(entry["incoming"], entry["pending"])
            entry["installed"] = False
        except BaseException as exc:
            failures.append("could not withdraw %s: %s" % (entry["target"], exc))
    if entry["claimed"]:
        try:
            _rename_locked(entry["previous"], entry["target"])
            entry["claimed"] = False
        except BaseException as exc:
            failures.append("could not restore %s from %s: %s" %
                            (entry["target"], entry["backup"], exc))
    if failures:
        raise RuntimeError("; ".join(failures))


def promote(pairs, *, expected_current=None):
    """Claim/CAS promotion, manifest last; retain recoverable prior versions.

    Each rename is atomic and never overwrites a newly created pathname. There
    can be a brief absent-path window between claim and installation; readers
    must honor the manifest and its hashes. Windows handle sharing prevents a
    noncooperating writer from changing the file being checked/replaced.
    """
    expected_current = {Path(key).absolute(): value for key, value in (expected_current or {}).items()}
    entries, recovery_dirs, failure, rollback_errors = [], set(), None, []
    transaction = uuid.uuid4().hex
    try:
        for source, target in pairs:
            source, target = Path(source), Path(target).absolute()
            target.parent.mkdir(parents=True, exist_ok=True)
            recovery = target.parent / (".release-recovery-" + transaction)
            recovery.mkdir(exist_ok=True)
            recovery_dirs.add(recovery)
            pending = recovery / (target.name + ".pending")
            backup = recovery / (target.name + ".previous")
            expected = (expected_current[target] if target in expected_current else
                        sha256(target) if target.exists() else None)
            copy_verified(source, pending, sha256(source))
            entries.append({"target": target, "pending": pending, "backup": backup,
                            "expected": expected, "new_hash": sha256(pending),
                            "incoming": None, "previous": None, "claimed": False, "installed": False})
        for entry in entries:
            entry["incoming"] = ExclusiveFile(entry["pending"])
            if entry["incoming"].digest() != entry["new_hash"]:
                raise ValueError("prepared replacement changed: " + str(entry["target"]))
            if entry["expected"] is not None:
                entry["previous"] = ExclusiveFile(entry["target"])
                if entry["previous"].digest() != entry["expected"]:
                    raise ValueError("destination changed; preserving conflicting version: " + str(entry["target"]))
                _rename_locked(entry["previous"], entry["backup"])
                entry["claimed"] = True
            _rename_locked(entry["incoming"], entry["target"])
            entry["installed"] = True
    except BaseException as exc:
        failure = exc
        for entry in reversed(entries):
            try:
                _rollback_entry(entry)
            except BaseException as restore_error:
                rollback_errors.append(str(restore_error))
    finally:
        for entry in entries:
            for key in ("incoming", "previous"):
                if entry[key] is not None:
                    try:
                        entry[key].close()
                    except BaseException as close_error:
                        rollback_errors.append("handle cleanup failed: " + str(close_error))
        # Only empty recovery directories are disposable. Never remove the sole
        # previous version after a failed restoration (or an interrupted run).
        for directory in recovery_dirs:
            try:
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                rollback_errors.append("recovery cleanup incomplete at " + str(directory))
    retained = sorted(str(path) for path in recovery_dirs if path.exists())
    if failure or rollback_errors:
        raise PromotionError(failure or "handle cleanup", retained, rollback_errors) from failure
    return retained
