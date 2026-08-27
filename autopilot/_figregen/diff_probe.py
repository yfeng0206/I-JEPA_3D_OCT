"""Prove that a presentation-only edit changed no plotted value.

Compares the normalised {x, y, ylo, yhi} records captured by probe.py before and
after the edit. A record is a JSON string, so equality is exact to the last bit
of every float64; series longer than 64 points are compared by SHA-256 of their
raw float64 bytes.

Exit code 1 if any figure's data differs.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def strip_label(rec_json):
    d = json.loads(rec_json)
    d.pop("label", None)
    return json.dumps(d, sort_keys=True)


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def compare(before, after, tag):
    """Data equality excludes the label, which is allowed to change.

    A record's x / y / ylo / yhi are compared exactly; series longer than 64
    points are compared by SHA-256 of their float64 bytes. Label changes are
    reported separately so a renamed legend entry can never be mistaken for a
    changed value, and a changed value can never hide behind a renamed label.
    """
    ok = True
    figs = sorted(set(before) | set(after))
    for fig in figs:
        b = before.get(fig, {}).get("canonical", [])
        a = after.get(fig, {}).get("canonical", [])
        bd = sorted(strip_label(r) for r in b)
        ad = sorted(strip_label(r) for r in a)
        same = (bd == ad)
        npts = sum(json.loads(r)["x"]["n"] for r in b)
        bl = sorted(json.loads(r).get("label") or "" for r in b)
        al = sorted(json.loads(r).get("label") or "" for r in a)
        note = "" if bl == al else "  (series labels renamed)"
        print("%-36s series %2d -> %2d  plotted_scalars %6d  %s%s"
              % (fig, len(b), len(a), npts * 2,
                 "DATA IDENTICAL" if same else "DATA DIFFERS", note))
        if bl != al:
            for L in sorted(set(bl) - set(al)):
                print("      label removed: %r" % L)
            for L in sorted(set(al) - set(bl)):
                print("      label added  : %r" % L)
        if not same:
            ok = False
            sb, sa = set(bd), set(ad)
            for r in sorted(sb - sa):
                print("    only before: %s" % r[:300])
            for r in sorted(sa - sb):
                print("    only after : %s" % r[:300])
    print("[%s] %s" % (tag, "ALL FIGURES DATA IDENTICAL" if ok else "MISMATCH"))
    return ok


def labels(after, tag):
    print("\n--- rendered strings after the edit (%s)" % tag)
    bad = []
    for fig in sorted(after):
        for L in after[fig]["labels"]:
            if "oracle" in L.lower():
                bad.append((fig, L))
    print("strings containing 'oracle' on any canvas: %d" % len(bad))
    for f, L in bad:
        print("   %s : %s" % (f, L))
    return not bad


def main():
    ok = True
    ok &= compare(load("before_p8.json"), load("after_p8.json"), "p8")
    print()
    ok &= compare(load("before_story.json"), load("after_story.json"), "story")
    ok &= labels(load("after_p8.json"), "p8")
    ok &= labels(load("after_story.json"), "story")
    print("\nRESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
