"""Check bibliographic existence and strict normalized-title agreement.

An identifier's presence is not successful verification. Resolve against arXiv
or Crossref, or replay a persisted, unchanged per-entry authority record. Empty,
missing and unresolved entries fail. This does NOT assess claim support or
appropriateness; those require a separate claim-to-source review.

Usage:
    python autopilot/verify_citations.py --record PATH  # offline record replay
    python autopilot/verify_citations.py --online       # resolve and persist
"""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request
import unicodedata
try:
    from . import release_assets as assets
except ImportError:
    import release_assets as assets

TEX = "paper/genai4health2026/main_submission.tex"
BIB = "paper/genai4health2026/references.bib"
UA = {"User-Agent": "citation-verifier/1.0 (academic reference check)"}


def cited_keys(tex):
    keys = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", assets.uncomment(tex)):
        keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def bib_entries(bib):
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", bib):
        start = m.end()
        depth, i = 1, bib.index("{", m.start())
        i += 1
        while i < len(bib) and depth:
            if bib[i] == "{":
                depth += 1
            elif bib[i] == "}":
                depth -= 1
            i += 1
        out[m.group(2).strip()] = (m.group(1).lower(), bib[start:i])
    return out


def field(body, name):
    match = re.search(r"\b" + re.escape(name) + r"\s*=\s*", body, re.I)
    if not match:
        return None
    start = match.end()
    if body[start:start + 1] == "{":
        value, _ = assets.group(body, start)
    elif body[start:start + 1] == '"':
        end = start + 1
        while end < len(body) and (body[end] != '"' or body[end - 1] == "\\"):
            end += 1
        value = body[start + 1:end]
    else:
        value = re.split(r"[,}\n]", body[start:], maxsplit=1)[0]
    return re.sub(r"\s+", " ", value).strip()


def identifier(body):
    doi = field(body, "doi")
    if doi:
        doi = doi.strip()
        # arXiv mints DataCite DOIs of the form 10.48550/arXiv.2310.02492.
        # Crossref does not serve those, so resolve them at arXiv instead.
        m = re.match(r"^10\.48550/arxiv\.(.+)$", doi, re.I)
        if m:
            eprint = field(body, "eprint") or ""
            pinned = re.fullmatch(r"(\d{4}\.\d{4,5})v\d+", eprint, re.I)
            if pinned and m.group(1).lower() == pinned.group(1).lower():
                return "arxiv", eprint
            return "arxiv", m.group(1)
        return "doi", doi
    eprint = field(body, "eprint") or field(body, "archiveprefix")
    if eprint and re.match(r"^\d{4}\.\d{4,5}", eprint.strip()):
        return "arxiv", eprint.strip()
    url = field(body, "url") or ""
    m = re.search(r"arxiv\.org/abs/([\d.]+)", url)
    if m:
        return "arxiv", m.group(1)
    m = re.search(r"doi\.org/(10\.[^\s}]+)", url)
    if m:
        return "doi", m.group(1)
    if url:
        return "url", url
    return None, None


def search_arxiv_title(title):
    """Find a paper by title when the entry carries no usable identifier."""
    clean = re.sub(r"[{}\\$]", "", title or "")
    clean = re.sub(r"[^A-Za-z0-9 :\-]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None
    q = ("https://export.arxiv.org/api/query?search_query=ti:"
         + urllib.parse.quote('"%s"' % clean) + "&max_results=3")
    try:
        with urllib.request.urlopen(urllib.request.Request(q, headers=UA),
                                    timeout=25) as r:
            x = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    hits = re.findall(r"<entry>.*?<title>(.*?)</title>", x, re.S)
    for h in hits:
        h = re.sub(r"\s+", " ", h).strip()
        a, b = norm(clean), norm(h)
        if a and a == b:
            return h
    return None


def resolve_arxiv(aid):
    q = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(aid)}"
    with urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=25) as r:
        x = r.read().decode("utf-8", "replace")
    if "<entry>" not in x:
        return None
    m = re.search(r"<entry>.*?<title>(.*?)</title>", x, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def resolve_doi(doi):
    q = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        with urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=25) as r:
            d = json.load(r)
    except Exception:
        return None
    t = d.get("message", {}).get("title") or []
    return t[0] if t else None


def norm(s):
    value = html.unescape(s or "")
    value = re.sub(r"\\(?:textit|textbf|emph|mathrm)\s*", "", value)
    value = re.sub(r"""\\['"`^~=.uvHckbdtr]\s*\{?([A-Za-z])\}?""", r"\1", value)
    value = value.replace("{", "").replace("}", "").replace("$", "")
    value = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def verify(tex, bib, *, online=False, record=None, expected_keys=()):
    keys = sorted(cited_keys(tex))
    entries = bib_entries(bib)
    errors = []
    if not keys:
        errors.append("no citations found")
    missing_expected = sorted(set(expected_keys) - set(keys))
    if missing_expected:
        errors.append("expected keys not cited: " + ", ".join(missing_expected))
    prior = {item["key"]: item for item in (record or {}).get("items", [])}
    rows = []
    for key in keys:
        row = {"key": key, "claim_support": "not_assessed"}
        if key not in entries:
            row["status"] = "missing_bib_entry"
            rows.append(row)
            continue
        body = entries[key][1]
        kind, value = identifier(body)
        title = field(body, "title")
        entry_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        row.update({"entry_sha256": entry_hash, "identifier_kind": kind, "identifier": value,
                    "expected_title": title, "checked_utc": datetime.now(timezone.utc).isoformat()})
        if not norm(title):
            row["status"] = "missing_title"
            rows.append(row)
            continue
        cached = prior.get(key, {})
        if (not online and cached.get("status") == "matched"
                and cached.get("entry_sha256") == entry_hash
                and cached.get("identifier_kind") == kind and cached.get("identifier") == value
                and cached.get("authority") and cached.get("checked_utc")
                and norm(cached.get("resolved_title")) == norm(title)):
            row.update({name: cached[name] for name in
                        ("resolved_title", "authority", "checked_utc")})
            row["status"] = "matched"
            row["verification_mode"] = "persisted_authority_record"
        elif not online:
            row["status"] = "unresolved_offline"
            row["action"] = "Resolve with authority or supply a matching persisted per-entry record."
        else:
            authority = ("https://export.arxiv.org/api/query" if kind != "doi"
                         else "https://api.crossref.org/works")
            try:
                got = (resolve_arxiv(value) if kind == "arxiv" else
                       resolve_doi(value) if kind == "doi" else search_arxiv_title(title))
                row.update({"authority": authority, "resolved_title": got,
                            "verification_mode": "online"})
                row["status"] = ("unresolved" if not got else
                                 "matched" if norm(title) == norm(got) else "title_mismatch")
            except Exception as exc:
                row.update({"status": "authority_error", "error": str(exc), "authority": authority})
            time.sleep(0.35)
        rows.append(row)
    return {"version": 1, "scope": "bibliographic existence and strict normalized-title agreement; "
            "does not assess claim support or appropriateness",
            "source_sha256": hashlib.sha256(tex.encode("utf-8")).hexdigest(),
            "bib_sha256": hashlib.sha256(bib.encode("utf-8")).hexdigest(),
            "cited_count": len(keys), "expected_keys": sorted(expected_keys), "errors": errors,
            "items": rows, "ALL_PASS": bool(rows) and not errors
            and all(row["status"] == "matched" for row in rows)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--paper-dir", default=str(assets.PAPER))
    ap.add_argument("--record", help="prior authority verification JSON (no network in offline mode)")
    ap.add_argument("--report", help="output per-item record, including unresolved entries")
    ap.add_argument("--expected-key", action="append", default=[])
    args = ap.parse_args(argv)
    output = Path(args.report) if args.report else assets.unique_work(prefix="citations") / "verification.json"
    try:
        tex, _ = assets.source_tree(args.paper_dir)
        bib = (Path(args.paper_dir) / "references.bib").read_text(encoding="utf-8")
        record = json.loads(Path(args.record).read_text(encoding="utf-8")) if args.record else None
        result = verify(tex, bib, online=args.online, record=record, expected_keys=args.expected_key)
    except (OSError, ValueError, KeyError) as exc:
        result = {"ALL_PASS": False, "errors": [str(exc)], "items": []}
    assets.write_json(output, result)
    matched = sum(row["status"] == "matched" for row in result["items"])
    print("Existence/title matched: %d/%d" % (matched, len(result["items"])))
    print("Claim support: NOT ASSESSED")
    print("Per-item record:", output)
    print("RESULT:", "PASS" if result["ALL_PASS"] else "FAIL")
    return 0 if result["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
