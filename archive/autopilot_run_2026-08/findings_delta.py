"""Append a findings delta after each milestone.

The operator asked that any new finding be surfaced and listed rather than
silently folded into the paper. After every refresh this writes a dated entry to
autopilot/FINDINGS_LOG.md recording what changed since the previous milestone:
new probes, new AUCs, and any headline contrast that moved.
"""
import json
import os
import sys
from datetime import datetime

STATS = r"D:\jepa_phase0\autopilot_out\p1_stats"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "FINDINGS_LOG.md")
SNAP = os.path.join(STATS, "_findings_snapshot.json")

KEY_CONTRASTS = [
    ("oracle", "random", 50), ("oracle", "random", 75), ("oracle", "random", 100),
    ("envelope", "random", 50), ("envelope", "random", 75), ("envelope", "random", 100),
]


def load_now():
    p = os.path.join(STATS, "p1c_stats.json")
    if not os.path.exists(p):
        return {}
    s = json.load(open(p))
    aucs = {t["key"]: round(t["auc"], 6) for t in s["table"]}
    con = {}
    for c in s["contrasts"]:
        if c["kind"] != "A_primary_matched":
            continue
        for a, b, ep in KEY_CONTRASTS:
            if c["epoch"] == ep and {c["arm_a"], c["arm_b"]} == {a, b}:
                d = c["delta_a_minus_b"] if c["arm_a"] == a else -c["delta_a_minus_b"]
                con["%s-%s@ep%d" % (a, b, ep)] = round(d, 6)
    return {"aucs": aucs, "contrasts": con}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "milestone"
    now = load_now()
    prev = json.load(open(SNAP)) if os.path.exists(SNAP) else {"aucs": {}, "contrasts": {}}

    new_probes = sorted(set(now.get("aucs", {})) - set(prev.get("aucs", {})))
    moved = []
    for k, v in now.get("contrasts", {}).items():
        pv = prev.get("contrasts", {}).get(k)
        if pv is not None and abs(pv - v) > 5e-5:
            moved.append((k, pv, v))

    lines = ["", "## %s  -  %s" % (tag, datetime.now().astimezone().isoformat(timespec="seconds")), ""]
    if new_probes:
        lines.append("**New probes since last milestone:**")
        lines.append("")
        lines.append("| probe | test AUC |")
        lines.append("|---|---|")
        for p in new_probes:
            lines.append("| `%s` | %.6f |" % (p, now["aucs"][p]))
        lines.append("")
    else:
        lines.append("No new probes since the last milestone.")
        lines.append("")

    if moved:
        lines.append("**Headline contrasts that moved:**")
        lines.append("")
        lines.append("| contrast | before | after | delta |")
        lines.append("|---|---|---|---|")
        for k, a, b in moved:
            lines.append("| %s | %+.6f | %+.6f | %+.6f |" % (k, a, b, b - a))
        lines.append("")
    else:
        lines.append("No headline contrast moved by more than 5e-5.")
        lines.append("")

    if now.get("contrasts"):
        lines.append("**Current headline contrasts:**")
        lines.append("")
        lines.append("| contrast | delta AUC |")
        lines.append("|---|---|")
        for k in sorted(now["contrasts"]):
            lines.append("| %s | %+.6f |" % (k, now["contrasts"][k]))
        lines.append("")

    if not os.path.exists(LOG):
        header = ("# FINDINGS LOG\n\nOne entry per milestone. Records new probes and any "
                  "headline contrast that moved, so nothing changes in the paper without "
                  "being surfaced here first.\n")
        open(LOG, "w", encoding="utf-8").write(header)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    json.dump(now, open(SNAP, "w"), indent=1)
    print("findings delta written for '%s': %d new probes, %d moved contrasts"
          % (tag, len(new_probes), len(moved)))


if __name__ == "__main__":
    main()
