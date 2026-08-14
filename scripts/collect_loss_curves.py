"""Collect per-epoch train/val loss for every pretraining arm into one workbook.

Sources differ in fidelity and that is recorded per row:
  envelope / anatomy : parsed densely from stdout logs (every epoch)
  random / oracle    : only the sparse tables in docs/experiments/pretraining
                       survive locally; the raw AML logs live in blob storage.
"""
import glob
import os
import re

import pandas as pd

OUT_DIR = r"D:\jepa_phase0\reports"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCH_RE = re.compile(
    r"^Epoch (\d+)/100\s+\((\d+)s\)\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)"
)


def parse_logs(paths):
    """Return {epoch: (wall_s, train, val)}; later files win on overlap."""
    out = {}
    for p in sorted(paths, key=os.path.getmtime):
        if not os.path.exists(p):
            continue
        with open(p, "r", errors="ignore") as fh:
            for line in fh:
                m = EPOCH_RE.match(line.strip())
                if m:
                    ep = int(m.group(1))
                    out[ep] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


# ---- dense arms ---------------------------------------------------------
envelope = parse_logs([
    r"C:\Users\Gary\Desktop\jepa\results\pretraining\pretrain_mirage_envelope\combined_stdout.log",
    r"D:\jepa_phase0\runs\patch_mirage_envelope\train.log",
])
anatomy = parse_logs(
    glob.glob(r"D:\jepa_phase0\runs\anatomy_v2_ep25\train_bridge_*.log")
    # earliest surviving stdout for this same run (ep27-28); the separate
    # anatomy_main_stdout.log belongs to the OLD patch_mirage_anatomy arm and
    # is deliberately excluded.
    + [r"D:\jepa_phase0\runs\anatomy_v2_ep25_stdout.log"]
)

# ---- sparse arms (docs/experiments/pretraining/*.md) --------------------
# random_100ep.md  -> epoch, train, val
random_rows = {
    1: (0.0521, 0.0764), 5: (0.0842, 0.0822), 10: (0.0919, 0.0944),
    25: (0.1174, 0.1197), 50: (0.1413, 0.1423), 75: (0.1445, 0.1469),
    88: (0.1384, 0.1442), 92: (0.1362, 0.1398), 95: (0.1357, 0.1432),
    96: (0.1361, 0.1415), 100: (0.1352, 0.1419),
}
# oracle_100ep.md -> epoch, train, val
oracle_rows = {
    26: (0.1186, 0.1202), 30: (0.1197, 0.1242), 35: (0.1232, 0.1310),
    50: (0.1316, 0.1400), 60: (0.1388, 0.1489), 75: (0.1404, 0.1507),
    88: (0.1335, 0.1454), 95: (0.1306, 0.1449), 100: (0.1300, 0.1430),
}

MAXEP = 100


def frame(name, dense=None, sparse=None, source=""):
    rows = []
    if dense:
        for ep in sorted(dense):
            if ep <= MAXEP:
                w, t, v = dense[ep]
                rows.append({"epoch": ep, "train_loss": t, "val_loss": v,
                             "wall_s": w, "arm": name, "fidelity": "per-epoch",
                             "source": source})
    if sparse:
        for ep in sorted(sparse):
            if ep <= MAXEP:
                t, v = sparse[ep]
                rows.append({"epoch": ep, "train_loss": t, "val_loss": v,
                             "wall_s": None, "arm": name, "fidelity": "sampled",
                             "source": source})
    return pd.DataFrame(rows)


arms = {
    "random": frame("random", sparse=random_rows,
                    source="docs/experiments/pretraining/random_100ep.md"),
    "oracle": frame("oracle", sparse=oracle_rows,
                    source="docs/experiments/pretraining/oracle_100ep.md"),
    "envelope": frame("envelope", dense=envelope,
                      source="results/pretraining/pretrain_mirage_envelope/combined_stdout.log"),
    "anatomy_bridged": frame("anatomy_bridged", dense=anatomy,
                             source="D:/jepa_phase0/runs/anatomy_v2_ep25/train_bridge_*.log"),
}

# ---- wide comparison sheet ---------------------------------------------
wide = pd.DataFrame({"epoch": range(1, MAXEP + 1)})
for name, df in arms.items():
    if df.empty:
        continue
    s = df.set_index("epoch")
    wide[f"{name}_train"] = wide["epoch"].map(s["train_loss"])
    wide[f"{name}_val"] = wide["epoch"].map(s["val_loss"])

notes = pd.DataFrame([
    ["Shared prefix", "All arms warm-start from the SAME random-init run at ep25 "
     "(oracle explicitly; envelope resumes at ep26; anatomy_v2 from a random ep25 "
     "checkpoint). Epochs 1-25 are therefore the random arm for every arm."],
    ["random fidelity", "Only the sampled table in random_100ep.md survives locally "
     "(ep 1,5,10,25,50,...). Raw per-epoch log is in Azure blob "
     "ijepa-results/patch_vit_base_ps16_ep100_bs64_lr0.00025_20260411_063607."],
    ["oracle fidelity", "Same: sampled table only, from oracle_100ep.md. Blob "
     "ijepa-results/patch_vit_base_ps16_ep100_bs32_lr0.00025_20260602_093108."],
    ["val comparability", "train_patch.py:452 pins validation to a plain uniform "
     "MaskCollator for EVERY arm, so val_loss IS directly comparable across arms. "
     "train_loss is NOT - each arm predicts a different number of cells."],
    ["anatomy coverage", "Dense from ep29 (first bridged epoch) to ep56. ep26-28 "
     "were run before the surviving logs were opened."],
], columns=["topic", "note"])

xlsx = os.path.join(OUT_DIR, "loss_curves_full.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
    wide.to_excel(xw, sheet_name="comparison_wide", index=False)
    for name, df in arms.items():
        if not df.empty:
            df.to_excel(xw, sheet_name=name[:31], index=False)
    notes.to_excel(xw, sheet_name="notes", index=False)

csv = os.path.join(OUT_DIR, "loss_curves_full.csv")
pd.concat(arms.values(), ignore_index=True).to_csv(csv, index=False)

print("wrote", xlsx)
print("wrote", csv)
for name, df in arms.items():
    if df.empty:
        print(f"  {name:18s} NO DATA")
    else:
        print(f"  {name:18s} {len(df):3d} rows, epochs "
              f"{df.epoch.min()}-{df.epoch.max()} ({df.fidelity.iloc[0]})")
