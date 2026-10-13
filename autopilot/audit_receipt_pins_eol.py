"""Read-only audit: do the receipt-pinned repository sources reproduce their pins from Git?

The numeric release check (p15_verify_numbers.py) pins evidence by SHA-256 of the
exact file bytes. A pin only survives a fresh checkout on every OS when the Git
blob itself holds those bytes; a CRLF working-tree file stored as an LF blob
hashes differently on LF (non-Windows) checkouts.

Pins enumerated (the ones p15 checks):
  * numeric_reviews.json "sources" (root repo/paper/stats, path, sha256) and the
    inline sha256 copies inside literal/figure receipts;
  * numeric_reviews.json figure receipt identities (paper-relative graphics);
  * release_assets.json expected_output_sha256 / fixed_inputs of active graphics.

A source is "cited" when a review macro, literal receipt or figure receipt
references it; p15 rejects stale literal/figure receipts, so on a p15 run without
gate errors every such receipt is applied. Figure pins are cited when the graphic
is included by the current paper source. Stats-root sources live outside the
repository and are listed only for completeness. Cited sources that are
deliberately gitignored (acquired locally, not redistributed) are reported as
notes, not as Git-blob failures.

Exit status 1 if any cited, tracked in-repository pin differs from the SHA-256 of
its index-blob contents (or a cited in-repository source is neither tracked nor
gitignored), or if an inline sha256 copy in a receipt contradicts its source pin.
No files are written unless --json is given; no network access.

Usage: python autopilot/audit_receipt_pins_eol.py [--verbose] [--json out.json]
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_assets as assets  # noqa: E402

PAPER_REL = assets.PAPER.relative_to(assets.REPO).as_posix()


def digest(raw):
    return None if raw is None else hashlib.sha256(raw).hexdigest()


def referenced_sources(value, found):
    if isinstance(value, dict):
        if isinstance(value.get("source"), str):
            found.append(value)
        for child in value.values():
            referenced_sources(child, found)
    elif isinstance(value, list):
        for child in value:
            referenced_sources(child, found)
    return found


def git(args, data=None):
    return subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=assets.REPO, input=data,
                          stdout=subprocess.PIPE, check=True).stdout


def blobs(names):
    """Raw blob bytes for revision:path names (None when missing), one git process."""
    if not names:
        return {}
    out = git(["cat-file", "--batch"], "".join(n + "\n" for n in names).encode())
    result, pos = {}, 0
    for name in names:
        end = out.index(b"\n", pos)
        header = out[pos:end].split()
        pos = end + 1
        if len(header) == 3 and header[1] == b"blob":
            size = int(header[2])
            result[name] = out[pos:pos + size]
            pos += size + 1
        else:
            result[name] = None
    return result


def text_attributes(paths):
    if not paths:
        return {}
    fields = git(["check-attr", "-z", "--stdin", "text"], "".join(p + "\0" for p in paths).encode())
    fields = fields.decode("utf-8").split("\0")
    return {fields[i]: fields[i + 2] for i in range(0, len(fields) - 2, 3)}


def ignored(paths):
    if not paths:
        return set()
    out = subprocess.run(["git", "-c", "core.quotepath=off", "check-ignore", "-z", "--stdin"], cwd=assets.REPO,
                         input="".join(p + "\0" for p in paths).encode(), stdout=subprocess.PIPE).stdout
    return set(filter(None, out.decode("utf-8").split("\0")))


def collect_pins(review_file, registry_file):
    reviews = json.loads(Path(review_file).read_text(encoding="utf-8-sig"))
    registry = json.loads(Path(registry_file).read_text(encoding="utf-8-sig"))
    text, _ = assets.source_tree(assets.PAPER)
    active = {assets.resolve_graphic(assets.PAPER, name).relative_to(assets.PAPER.resolve()).as_posix()
              for name in assets.graphics(text)}
    citations, conflicts = {}, []
    receipts = ([("macro:" + name, spec) for name, spec in reviews.get("macros", {}).items()]
                + [("literal:%s#%s/%d" % (e["file"], e["context_sha256"][:12], e["token_index"]), e)
                   for e in reviews.get("literals", [])]
                + [("figure:" + f["path"], f.get("inputs", [])) for f in reviews.get("figures", [])])
    sources = reviews.get("sources", {})
    for label, receipt in receipts:
        for ref in referenced_sources(receipt, []):
            citations.setdefault(ref["source"], set()).add(label)
            inline = ref.get("sha256")
            if inline and ref["source"] in sources and inline != sources[ref["source"]]["sha256"]:
                conflicts.append("%s: inline sha256 for %s differs from its source pin" % (label, ref["source"]))
    pins = []
    for name, spec in sources.items():
        rel = spec["path"].replace("\\", "/")
        root = spec["root"]
        path = rel if root == "repo" else PAPER_REL + "/" + rel if root == "paper" else None
        pins.append({"origin": "numeric_reviews.sources", "id": name, "root": root, "path": path or rel,
                     "in_repo": path is not None, "pin": spec["sha256"],
                     "cited": name in citations, "citations": len(citations.get(name, ()))})
    for figure in reviews.get("figures", []):
        pins.append({"origin": "numeric_reviews.figures", "id": figure["path"], "root": "paper",
                     "path": PAPER_REL + "/" + figure["path"], "in_repo": True, "pin": figure["sha256"],
                     "cited": figure["path"] in active, "citations": int(figure["path"] in active)})
    declared = [(rel, spec.get("expected_output_sha256")) for rel, spec in registry.get("producers", {}).items()]
    declared += list(registry.get("fixed_inputs", {}).items())
    for rel, expected in declared:
        if expected:
            pins.append({"origin": "release_assets", "id": rel, "root": "paper", "path": PAPER_REL + "/" + rel,
                         "in_repo": True, "pin": expected, "cited": rel in active, "citations": int(rel in active)})
    unknown = sorted(set(citations) - set(sources))
    return pins, conflicts, unknown


def audit(review_file, registry_file):
    pins, conflicts, unknown = collect_pins(review_file, registry_file)
    problems = []
    repo_paths = sorted({p["path"] for p in pins if p["in_repo"]})
    index = blobs([":" + p for p in repo_paths])
    head = blobs(["HEAD:" + p for p in repo_paths])
    attrs = text_attributes(repo_paths)
    local_only = ignored([p for p in repo_paths if index[":" + p] is None])
    for pin in pins:
        if not pin["in_repo"]:
            pin["status"] = "external"
            continue
        path = assets.REPO / pin["path"]
        raw = path.read_bytes() if path.is_file() else None
        pin.update(worktree=digest(raw), index=digest(index[":" + pin["path"]]),
                   head=digest(head["HEAD:" + pin["path"]]),
                   lf_normalized=digest(raw.replace(b"\r\n", b"\n")) if raw is not None else None,
                   text_attr=attrs.get(pin["path"], "unspecified"))
        # V15 P2-1 criterion: checkout bytes match the pin, the blob is their LF-normalized form.
        pin["eol_only_drift"] = (pin["worktree"] == pin["pin"] != pin["index"]
                                 and pin["index"] == pin["lf_normalized"])
        if pin["index"] == pin["pin"]:
            pin["status"] = "ok"
        elif pin["index"] is None and pin["path"] in local_only:
            # Deliberately not redistributed (.gitignore); acquired locally against its pin.
            pin["status"] = "local_ignored_match" if pin["worktree"] == pin["pin"] else "local_ignored_mismatch"
        else:
            reason = "untracked" if pin["index"] is None else "eol" if pin["worktree"] == pin["pin"] else "stale"
            pin["status"] = ("FAIL_" if pin["cited"] else "uncited_") + reason
    failures = [p for p in pins if p["status"].startswith("FAIL")]
    cited = [p for p in pins if p["in_repo"] and p["cited"]]
    summary = {
        "pins": len(pins),
        "in_repo_pins": sum(p["in_repo"] for p in pins),
        "external_stats_pins": sum(not p["in_repo"] for p in pins),
        "cited_in_repo_pins": len(cited),
        "cited_index_blob_matches_pin": sum(p["status"] == "ok" for p in cited),
        "cited_index_blob_differs_from_pin": len(failures),
        "cited_gitignored_local_sources": sum(p["status"].startswith("local_ignored") for p in cited),
        "cited_eol_only_drift_distinct_paths": len(
            {p["path"] for p in cited if p["eol_only_drift"]}),
        "cited_worktree_differs_from_pin": sum(p["worktree"] != p["pin"] for p in cited),
        "uncited_in_repo_pins": sum(p["in_repo"] and not p["cited"] for p in pins),
        "inline_pin_conflicts": len(conflicts),
        "status_counts": dict(sorted(Counter(p["status"] for p in pins).items())),
    }
    for p in cited:
        if p["status"].startswith("local_ignored"):
            problems.append("cited source %s is gitignored (not redistributed); a fresh checkout must acquire %s "
                            "with sha256 %s" % (p["id"], p["path"], p["pin"]))
    if unknown:
        problems.append("receipts cite sources without a numeric_reviews pin (built-in/unpinned): " + ", ".join(unknown))
    return {"summary": summary, "conflicts": conflicts, "problems": problems, "pins": pins,
            "PASS": not failures and not conflicts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review-file", default=str(assets.PAPER / "numeric_reviews.json"))
    parser.add_argument("--registry", default=str(Path(__file__).with_name("release_assets.json")))
    parser.add_argument("--json", help="also write the full machine-readable audit here")
    parser.add_argument("--verbose", action="store_true", help="print every pin, not only cited non-ok ones")
    args = parser.parse_args(argv)
    report = audit(args.review_file, args.registry)
    short = lambda h: (h or "-")[:12]  # noqa: E731
    print("status\tcited\ttext\tpin\tworktree\tindex\tHEAD\tid\tpath")
    for p in report["pins"]:
        if args.verbose or (p["cited"] and p["status"] not in ("ok", "external")):
            print("\t".join([p["status"], "yes" if p["cited"] else "no", p.get("text_attr", "-"), short(p["pin"]),
                             short(p.get("worktree")), short(p.get("index")), short(p.get("head")),
                             p["id"], p["path"]]))
    for conflict in report["conflicts"]:
        print("FAIL:", conflict)
    for problem in report["problems"]:
        print("NOTE:", problem)
    for key, value in report["summary"].items():
        print("%s: %s" % (key, value))
    if args.json:
        assets.write_json(args.json, report)
    print("RESULT:", "PASS" if report["PASS"] else "FAIL")
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
