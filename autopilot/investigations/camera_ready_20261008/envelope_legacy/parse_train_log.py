"""Parse the trained ENVELOPE run's [MIRAGE] lines into per-epoch statistics.

Each ``[MIRAGE]`` line is logged right after an ``[Epoch E/100 | Iter I/N]``
line (every 50 micro-batches, one B=64 batch each), so it is attributed to
1-based epoch E.  Output: per-epoch means (comparable to A1's
envelope_mirage_stats_by_epoch.json) plus pooled per-batch SD for ep31-100.

Usage: python parse_train_log.py TRAIN_LOG OUT.json
"""
import json
import re
import sys

import numpy as np

LOG, OUT = sys.argv[1], sys.argv[2]
epoch_re = re.compile(r"\[Epoch (\d+)/\d+ \| Iter (\d+)/(\d+)\]")
kv_re = re.compile(r"([a-z_/]+)=([-0-9.eE+naninf]+)")
by_epoch = {}
epoch = None
with open(LOG, "r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        m = epoch_re.search(line)
        if m:
            epoch = int(m.group(1))
            continue
        if "[MIRAGE]" in line and epoch is not None:
            row = {k: float(v) for k, v in kv_re.findall(line.split("[MIRAGE]", 1)[1])}
            by_epoch.setdefault(epoch, []).append(row)

out = {"source": LOG, "per_epoch": {}, "pooled_ep31_100": {}}
for e in sorted(by_epoch):
    rows = by_epoch[e]
    keys = rows[0].keys()
    out["per_epoch"][str(e)] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["per_epoch"][str(e)]["n_lines"] = len(rows)
full = [r for e in by_epoch if 31 <= e <= 100 for r in by_epoch[e]]
for k in full[0]:
    vals = np.array([r[k] for r in full])
    out["pooled_ep31_100"][k] = {"mean": float(vals.mean()), "sd_batch": float(vals.std(ddof=1)),
                                 "n_batches": int(vals.size)}
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(out, handle, indent=1)
print("epochs", min(by_epoch), "-", max(by_epoch), "lines/epoch",
      sorted({len(v) for v in by_epoch.values()}), "pooled n", len(full))
