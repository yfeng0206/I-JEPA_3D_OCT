"""Re-key literal receipts after the post-freeze integration (coordinator note 3).

p15 keys a literal receipt by (file, paragraph or table-row context hash, token index).
``make_cr_results.py integrate`` rewrites passages inside some paragraphs, so the literals of an
edited paragraph get new keys and their receipts go stale ("stale or inapplicable literal
review"). A receipt is re-keyed automatically only through the reviewed integration mapping
(V8-3): the old paragraph, with exactly the replacements ``integrate`` applies for these
results (``make_cr_results.integration_pairs``) and nothing else, must reproduce the new
paragraph exactly and exactly once; the literal must lie outside every replaced passage, keep
its value and its +-W-character window (default 24), and map one-to-one to an unresolved
literal of the same file. The claim text around the number is then unchanged by construction.
A paragraph changed in any other way (another arm, split, qualifier, ...) is never renewed
automatically, however similar the local window. Such literals, and literals inside a replaced
passage that reappear in its replacement, are listed as manual, with a suggested predecessor
(same file and value, one side of the window equal, unique) and both contexts; they are re-keyed
only with --accept <id> --reviewer <name> after a human compared the contexts. Stale receipts
of literals removed by the integration (inside a replaced passage or the tab:replication
skeleton) are dropped unless suggested; p15 still blocks every literal without a receipt.
Without --results-dir nothing is mapped automatically. Nothing is written without --apply.

Usage (repository root)
  python autopilot/p15_verify_numbers.py --paper-dir <paper> --stats-dir <p1_stats> ^
      --review-file <paper>/numeric_reviews.json --report p15.json
  python autopilot/cr_migrate_receipts.py --paper-dir <paper> --old-dir <pre-integration copy> ^
      --results-dir <paper>/auto --report p15.json --out plan.json ^
      [--apply [--allow-real-paper]] [--accept ID ... --reviewer NAME]
Exit codes: 0 ok (plan written, applied if requested), 2 refused input.
"""
import argparse
import datetime
import json
from pathlib import Path
import re
import sys

try:
    from . import make_cr_results as mk
    from . import numeric_bindings as numeric
    from . import release_assets as assets
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_cr_results as mk
    import numeric_bindings as numeric
    import release_assets as assets

FILES = ("main_submission.tex", "compact_protocol.tex", "compact_precision.tex")
STALE = re.compile(r"stale or inapplicable literal review \(context/token changed\): "
                   r"\('([^']+)', '([0-9a-f]+)', (\d+)\)")
# Paragraph rule of numeric_bindings.literals (context blocks).
BLOCK = re.compile(r"(?:[^\n]|\n(?!\s*\n))+")


class MigrationError(RuntimeError):
    pass


def _norm(text):
    return re.sub(r"\s+", " ", text)


def scan(source, rel, width):
    """Non-structural literals of one file: by key and by offset, with their +-width window."""
    by_key, by_offset = {}, {}
    for row in numeric.literals(source, rel):
        if row["status"] == "structural":
            continue
        start = row["offset"]
        end = start + len(row["value"])
        before = _norm(source[max(0, start - width * 3):start])[-width:]
        after = _norm(source[end:end + width * 3])[:width]
        row = dict(row, window=(before, row["value"], after))
        by_key[(rel, row["context_sha256"], row["token_index"])] = row
        by_offset[start] = row
    return by_key, by_offset


def read_sources(directory, files):
    return {rel: assets.uncomment((Path(directory) / rel).read_text(encoding="utf-8"))
            for rel in files if (Path(directory) / rel).is_file()}


def load_pairs(results_dir):
    """Integration replacements per file for the results the paper was integrated with."""
    sidecar, _ = mk.read_json(Path(results_dir) / "cr_results.values.json")
    scenario = mk.sidecar_scenario(sidecar, results_dir)
    pairs = {}
    for rel, old, new, _ in mk.integration_pairs(scenario, bool(sidecar.get("figure"))):
        pairs.setdefault(rel, []).append((old, new))
    return pairs


def map_block(text, pairs):
    """Apply the integration passages (each at most once, non-overlapping) to one paragraph.
    Returns (mapped text, unchanged segments (old start, old end, new start), replaced spans),
    or None when a passage is ambiguous."""
    hits = []
    for old, new in pairs:
        found = list(re.finditer(mk._ws(old), text))
        if len(found) > 1:
            return None
        if found:
            hits.append((found[0].start(), found[0].end(), new))
    hits.sort()
    if any(a[1] > b[0] for a, b in zip(hits, hits[1:])):
        return None
    out, segments, pos, cur = [], [], 0, 0
    for start, end, new in hits:
        out.append(text[pos:start])
        segments.append((pos, start, cur))
        cur += start - pos
        out.append(new)
        cur += len(new)
        pos = end
    out.append(text[pos:])
    segments.append((pos, len(text), cur))
    return "".join(out), segments, [(s, e) for s, e, _ in hits]


def map_literal(old_src, new_src, row, pairs, new_by_offset, table_span):
    """('mapped', new row) | ('removed', why) | ('unmatched', why) for one old literal."""
    off, length = row["offset"], len(row["value"])
    if table_span and table_span[0] <= off < table_span[1]:
        return "removed", "inside the tab:replication skeleton the integration removed"
    block = next(b for b in BLOCK.finditer(old_src) if b.start() <= off < b.end())
    mapped = map_block(block.group(0), pairs)
    if mapped is None:
        return "unmatched", "an integration passage is ambiguous in this paragraph"
    text, segments, replaced = mapped
    o = off - block.start()
    if any(s < o + length and o < e for s, e in replaced):
        return "removed", "inside a passage the integration replaced"
    seg = next(s for s in segments if s[0] <= o and o + length <= s[1])
    m = seg[2] + (o - seg[0])
    sub = next(b for b in BLOCK.finditer(text) if b.start() <= m < b.end())
    starts = [b.start() for b in BLOCK.finditer(new_src) if b.group(0) == sub.group(0)]
    if len(starts) != 1:
        return "unmatched", ("the paragraph changed outside the integration mapping"
                             if not starts else "the mapped paragraph occurs more than once")
    new_row = new_by_offset.get(starts[0] + m - sub.start())
    if new_row is None or new_row["value"] != row["value"]:
        return "unmatched", "no literal of the same value at the mapped position"
    return "mapped", new_row


def plan_migration(paper_dir, old_dir, report, review_file, width=24, files=FILES, accept=(),
                   pairs=None):
    reviews = json.loads(Path(review_file).read_bytes().decode("utf-8"))
    entries = {(e["file"], e["context_sha256"], e["token_index"]): e
               for e in reviews.get("literals", [])}
    stale = set()
    for error in report.get("errors", []):
        m = STALE.match(error)
        if m:
            stale.add((m[1], m[2], int(m[3])))
    old_src, new_src = read_sources(old_dir, files), read_sources(paper_dir, files)
    old_rows, new_rows, new_offsets, tables = {}, {}, {}, {}
    for rel, text in old_src.items():
        old_rows.update(scan(text, rel, width)[0])
        table = mk.REPLICATION_TABLE.search(text)
        tables[rel] = (table.start(), table.end()) if table else None
    for rel, text in new_src.items():
        by_key, by_offset = scan(text, rel, width)
        new_rows.update(by_key)
        new_offsets[rel] = by_offset
    unresolved = [(r["file"], r["context_sha256"], r["token_index"]) for r in report.get("items", [])
                  if r.get("kind") == "literal" and r.get("status") == "unresolved"]
    unresolved_set = set(unresolved)
    migrations, outcome, used_old, used_new = [], {}, set(), set()
    for key in sorted(stale):
        rel = key[0]
        if key not in old_rows or key not in entries or rel not in new_src:
            outcome[key] = ("unmatched", "not a scanned literal of the snapshot")
            continue
        if pairs is None:
            outcome[key] = ("unmatched", "no --results-dir: automatic mapping disabled")
            continue
        status, info = map_literal(old_src[rel], new_src[rel], old_rows[key], pairs.get(rel, []),
                                   new_offsets[rel], tables.get(rel))
        if status != "mapped":
            outcome[key] = (status, info)
            continue
        new_key = (rel, info["context_sha256"], info["token_index"])
        if new_key not in unresolved_set or new_key in used_new:
            outcome[key] = ("unmatched", "mapped literal is not an unresolved literal")
            continue
        if old_rows[key]["window"] != info["window"]:
            outcome[key] = ("unmatched", "a replaced passage reaches into the literal's window")
            continue
        used_old.add(key)
        used_new.add(new_key)
        outcome[key] = ("mapped", None)
        migrations.append({"new": list(new_key), "old": list(key), "value": info["value"],
                           "window": list(info["window"]),
                           "rule": "integration-mapped paragraph, unchanged claim text"})
    accept = set(accept)
    accepted, manual = [], []
    for key in unresolved:
        if key in used_new:
            continue
        row = new_rows.get(key)
        if row is None:
            manual.append({"key": list(key), "reason": "not a scanned prose literal of %s"
                           % ", ".join(files)})
            continue
        before, value, after = row["window"]
        suggested = [k for k in sorted(stale) if k not in used_old and k in old_rows
                     and k in entries and k[0] == key[0] and old_rows[k]["value"] == value and
                     (old_rows[k]["window"][0] == before or old_rows[k]["window"][2] == after)]
        item = {"key": list(key), "id": "%s:%s:%d" % (key[0], key[1][:16], key[2]),
                "value": value, "window": list(row["window"]), "new_context": row["context"][:600],
                "claim_class": row["claim_class"], "suggested": None}
        if len(suggested) == 1:
            old = suggested[0]
            item["suggested"] = {"old": list(old), "old_window": list(old_rows[old]["window"]),
                                 "old_context": old_rows[old]["context"][:600],
                                 "why_not_automatic": outcome.get(old, ("unmatched", ""))[1],
                                 "review": entries[old].get("review")}
        if item["id"] in accept and item["suggested"]:
            used_old.add(suggested[0])
            accepted.append({"new": list(key), "old": list(suggested[0]), "value": value,
                             "window": list(row["window"]),
                             "old_window": item["suggested"]["old_window"]})
        else:
            manual.append(item)
    unknown = accept - set("%s:%s:%d" % (a["new"][0], a["new"][1][:16], a["new"][2])
                           for a in accepted)
    if unknown:
        raise MigrationError("--accept ids without a unique suggestion: %s" % sorted(unknown))
    suggested_keys = set(tuple(m["suggested"]["old"]) for m in manual if m.get("suggested"))
    leftover = sorted(stale - used_old)
    obsolete = [k for k in leftover if outcome.get(k, ("",))[0] == "removed"
                and k not in suggested_keys]
    return {"schema": "cr_receipt_migration_v2", "width": width, "files": list(files),
            "review_file": str(review_file), "automatic_mapping": pairs is not None,
            "n_stale": len(stale), "n_unresolved_literals": len(unresolved),
            "migrations": migrations, "accepted": accepted, "manual": manual,
            "obsolete_dropped": [{"key": list(k), "value": old_rows[k]["value"],
                                  "window": list(old_rows[k]["window"]),
                                  "why": outcome[k][1]} for k in obsolete],
            "stale_not_migrated": [{"key": list(k),
                                    "value": old_rows[k]["value"] if k in old_rows else None,
                                    "why": outcome.get(k, ("", ""))[1]}
                                   for k in leftover if k not in obsolete]}, reviews, entries


def apply_migration(plan, reviews, entries, review_file, stamp, accept_stamp=None):
    raw = Path(review_file).read_bytes()
    moves = [(item, stamp) for item in plan["migrations"]] + \
        [(item, accept_stamp) for item in plan["accepted"]]
    for item, note in moves:
        old_key, new_key = tuple(item["old"]), tuple(item["new"])
        if new_key in entries:
            raise MigrationError("a receipt already exists for %s" % (new_key,))
        entry = dict(entries.pop(old_key))
        entry["file"], entry["context_sha256"], entry["token_index"] = new_key
        if isinstance(entry.get("review"), dict):
            review = dict(entry["review"])
            if note not in str(review.get("reviewer", "")):
                review["reviewer"] = str(review.get("reviewer", "")) + note
            entry["review"] = review
        entries[new_key] = entry
    for item in plan["obsolete_dropped"]:
        entries.pop(tuple(item["key"]), None)
    reviews["literals"] = list(entries.values())
    text = json.dumps(reviews, indent=2, ensure_ascii=False) + "\n"
    if b"\r\n" in raw:
        text = text.replace("\n", "\r\n")
    Path(review_file).write_bytes(text.encode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper-dir", required=True, help="integrated paper directory")
    ap.add_argument("--old-dir", required=True, help="pre-integration copy of the .tex sources")
    ap.add_argument("--results-dir", help="results the paper was integrated with (paper/auto); "
                                          "without it nothing is mapped automatically")
    ap.add_argument("--report", required=True, help="p15 coverage JSON of the integrated paper")
    ap.add_argument("--review-file", help="default: <paper-dir>/numeric_reviews.json")
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--out", required=True, help="migration plan JSON")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--accept", nargs="*", default=[],
                    help="ids (file:context16:index) of manual items whose suggested predecessor "
                         "receipt a human checked against the new context")
    ap.add_argument("--reviewer", default="", help="name of the human accepting --accept items")
    ap.add_argument("--allow-real-paper", action="store_true")
    a = ap.parse_args(argv)
    paper = Path(a.paper_dir).resolve()
    review_file = Path(a.review_file).resolve() if a.review_file else paper / "numeric_reviews.json"
    try:
        real = (assets.PAPER / "numeric_reviews.json").resolve()
        if a.apply and review_file == real and not a.allow_real_paper:
            raise MigrationError("refusing to rewrite the real numeric_reviews.json without "
                                 "--allow-real-paper")
        if a.accept and not a.reviewer.strip():
            raise MigrationError("--accept needs --reviewer (the human who checked the receipts)")
        if Path(a.old_dir).resolve() == paper:
            raise MigrationError("--old-dir must be the pre-integration snapshot, not the paper")
        pairs = load_pairs(a.results_dir) if a.results_dir else None
        report = json.loads(Path(a.report).read_text(encoding="utf-8"))
        plan, reviews, entries = plan_migration(paper, a.old_dir, report, review_file, a.width,
                                                accept=a.accept, pairs=pairs)
        plan["applied"] = False
        today = datetime.date.today().isoformat()
        if a.apply and (plan["migrations"] or plan["accepted"] or plan["obsolete_dropped"]):
            stamp = ("; context re-keyed after the post-freeze integration (paragraph changed "
                     "only by the reviewed integration mapping, claim text and +-%d-character "
                     "window unchanged; cr_migrate_receipts.py, %s)" % (a.width, today))
            accept_stamp = ("; re-keyed after the post-freeze integration changed its context, "
                            "checked against the new context by %s (%s)" % (a.reviewer.strip(),
                                                                           today))
            apply_migration(plan, reviews, entries, review_file, stamp, accept_stamp)
            plan["applied"] = True
    except (MigrationError, mk.InputError, OSError, ValueError, KeyError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    Path(a.out).write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    print("stale receipts %d; migrated %d, accepted %d, obsolete dropped %d%s; manual %d; stale "
          "kept %d%s -> %s" % (plan["n_stale"], len(plan["migrations"]), len(plan["accepted"]),
                               len(plan["obsolete_dropped"]),
                               " (applied)" if plan["applied"] else "", len(plan["manual"]),
                               len(plan["stale_not_migrated"]),
                               "" if pairs is not None else " (no --results-dir: nothing mapped)",
                               a.out))
    for item in plan["manual"]:
        print("  manual %s value %s window %r%s" % (
            item.get("id", item["key"]), item.get("value"), item.get("window"),
            "; suggested predecessor %s (%s; accept with --accept %s after comparing "
            "old_context and new_context in the plan)" % (
                item["suggested"]["old"][1][:16], item["suggested"]["why_not_automatic"],
                item["id"]) if item.get("suggested") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
