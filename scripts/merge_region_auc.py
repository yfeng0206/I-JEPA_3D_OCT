"""Merge the per-arm region_auc.json shards and report the comparison table.

Each arm runs in its own process writing its own output dir, so the shards have
to be stitched back together.  Also computes the numbers the whole experiment
exists to produce: how much of the downstream signal survives when the frozen
encoder is pooled over background cells only, versus anatomy cells only, versus
the uniform all-cell pooling the published protocol uses.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

OUT = pathlib.Path(r"D:\jepa_phase0\reports\downstream_region_auc")

rows = []
for shard in sorted(OUT.glob("*/region_auc.json")):
    for r in json.loads(shard.read_text()):
        rows.append(dict(
            tag=r["tag"],
            all=r["all"]["test"], anatomy=r["anatomy"]["test"],
            background=r["background"]["test"],
            all_val=r["all"]["val"], anatomy_val=r["anatomy"]["val"],
            background_val=r["background"]["val"],
        ))

if not rows:
    raise SystemExit("no shards found yet")

df = pd.DataFrame(rows).drop_duplicates("tag").sort_values("tag")

# chance-corrected retention: how much of the arm's above-chance signal does a
# region-restricted pooling keep?
for c in ("anatomy", "background"):
    df[f"{c}_keep_%"] = 100 * (df[c] - 0.5) / (df["all"] - 0.5)
df["bg_minus_anat"] = df["background"] - df["anatomy"]

pd.set_option("display.width", 220)
print("\n=== TEST AUC by pooled region (glaucoma classification) ===")
print(df[["tag", "all", "anatomy", "background",
          "anatomy_keep_%", "background_keep_%", "bg_minus_anat"]]
      .to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

df.to_csv(OUT / "region_auc_summary.csv", index=False)
(OUT / "region_auc_summary.json").write_text(
    json.dumps(df.to_dict(orient="records"), indent=2))
print(f"\nwrote {OUT / 'region_auc_summary.csv'}")
