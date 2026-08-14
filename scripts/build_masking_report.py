"""Merge loss curves + mask composition + downstream AUC into one workbook."""
import json
import os

import pandas as pd

REP = r"D:\jepa_phase0\reports"
ARMS = ["random", "oracle", "envelope", "anatomy"]

ROWS = [
    ("mask_tokens_mean", "TOKENS PER MASK (one target block)"),
    ("mask_on_anat_mean", "  ... on anatomy"),
    ("mask_bg_mean", "  ... background (black)"),
    ("mask_uniq_cells_mean", "  ... distinct cells (dedup of padding)"),
    ("mask_uniq_on_anat_mean", "      ... on anatomy"),
    ("mask_uniq_bg_mean", "      ... background"),
    ("n_ctx_mean", "CONTEXT TOKENS LEFT to encoder"),
    ("ctx_on_anat_mean", "  ... on anatomy"),
    ("ctx_bg_mean", "  ... background (black)"),
    ("ctx_pct_on_anat", "  ... % of context that is anatomy"),
    ("ctx_share_of_all_anat", "  ... % of ALL anatomy cells kept visible"),
    ("n_hidden_mean", "HIDDEN per image (union of 4 masks)"),
    ("hid_on_anat_mean", "  ... on anatomy"),
    ("hid_bg_mean", "  ... background"),
    ("hidden_pct_on_anat", "  ... % of hidden that is anatomy"),
    ("hidden_share_of_all_anat", "  ... % of ALL anatomy cells hidden"),
    ("n_slots_mean", "predictor slots per image (4 masks)"),
    ("slot_dupes_mean", "  ... duplicate slots"),
    ("ctx_frac_of_grid", "context as fraction of 256 patches"),
    ("hidden_frac_of_grid", "hidden as fraction of 256 patches"),
    ("anat_cells_mean", "anatomy cells present in image"),
]


def sheet(path):
    if not os.path.exists(path):
        return None, None
    d = json.load(open(path))
    rows = []
    for key, label in ROWS:
        r = {"metric": label, "key": key}
        for a in ARMS:
            r[a] = d.get(a, {}).get(key)
        rows.append(r)
    return pd.DataFrame(rows), d.get("_meta", {})


fair, fair_meta = sheet(os.path.join(REP, "mask_stats_fairvision.json"))
goals, goals_meta = sheet(os.path.join(REP, "mask_stats_goals.json"))

auc = pd.DataFrame([
    ["random",   50, 0.8641, 0.8451, "measured, frozen mean-pool, 100 slices"],
    ["random",   75, 0.8723, 0.8546, "measured"],
    ["random",  100, 0.8746, 0.8559, "measured"],
    ["oracle",   50, 0.8740, 0.8544, "measured"],
    ["oracle",   75, 0.8836, 0.8624, "measured"],
    ["oracle",  100, 0.8855, 0.8636, "measured"],
    ["envelope", 30, 0.8539, 0.8467, "measured"],
    ["envelope", 50, 0.8761, 0.8594, "measured"],
    ["envelope", 75, 0.8803, 0.8584, "measured"],
    ["envelope", 100, 0.8807, 0.8599, "measured"],
    ["anatomy (pre-bridge)", 30, 0.8583, 0.8461, "measured, 5 seeds"],
    ["anatomy (bridged)", 35, 0.8661, 0.8472, "measured, seed 42"],
    ["anatomy (bridged)", 40, 0.8683, 0.8485, "measured, seed 42"],
    ["anatomy (bridged)", 50, 0.8654, 0.8464, "measured, seed 42"],
    ["anatomy (bridged)", 100, None, None, "NOT RUN - stopped at ep56"],
], columns=["arm", "epoch", "test_auc", "val_auc", "status"])

meta = pd.DataFrame(
    [["fairvision", json.dumps(fair_meta)], ["goals", json.dumps(goals_meta)]],
    columns=["source", "meta"],
)

src = os.path.join(REP, "loss_curves_full.xlsx")
out = os.path.join(REP, "masking_report.xlsx")
loss = pd.read_excel(src, sheet_name=None)

with pd.ExcelWriter(out, engine="openpyxl") as xw:
    if fair is not None:
        fair.to_excel(xw, sheet_name="mask_stats_fairvision", index=False)
    if goals is not None:
        goals.to_excel(xw, sheet_name="mask_stats_goals", index=False)
    auc.to_excel(xw, sheet_name="downstream_auc", index=False)
    for name, df in loss.items():
        df.to_excel(xw, sheet_name=("loss_" + name)[:31], index=False)
    meta.to_excel(xw, sheet_name="mask_stats_meta", index=False)

print("wrote", out)
for name, df in (("fairvision", fair), ("goals", goals)):
    if df is not None:
        print(f"  {name}: {len(df)} metrics x {len(ARMS)} arms")
