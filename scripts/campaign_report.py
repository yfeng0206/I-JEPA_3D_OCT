"""Results table and figures for the COVER-then-RANDOM campaign.

Safe to run at any point: it reports whatever AUCs and loss curves exist so far,
so it can be called after each milestone rather than only at the end.

Reads:
  D:\\jepa_phase0\\campaign\\chain_status.json   AUCs produced by the chain
  D:\\jepa_phase0\\runs\\<run>\\*.log             per-epoch train/val loss
  the archived arms' published AUCs (below)

Writes a markdown table, a CSV, and two figures into
D:\\jepa_phase0\\reports\\campaign\\.
"""
from __future__ import annotations

import glob
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CAMP = pathlib.Path(r"D:\jepa_phase0\campaign")
OUT = pathlib.Path(r"D:\jepa_phase0\reports\campaign")
FORK_AUC = 0.8487

# Published frozen mean_pool TEST AUCs for the archived arms.  These were all
# produced under the stock `prefix` crop, fp32 targets and the explicit
# attention path, so they are an IMPERFECT baseline for the new arm -- see the
# deviation ledger in docs/experiments/masking/cover_random_campaign.md.
ARCHIVED = {
    "random":   {50: 0.8641, 75: 0.8723, 100: 0.8746},
    "oracle":   {50: 0.8740, 75: 0.8836, 100: 0.8855},
    "envelope": {30: 0.8539, 50: 0.8761, 75: 0.8803, 100: 0.8807},
    "blob":     {35: 0.8661, 40: 0.8683, 50: 0.8654},
}
EPOCH_RE = re.compile(
    r"^Epoch (\d+)/\d+\s+\((\d+)s\)\s+train_loss=([\d.]+)(?:\s+val_loss=([\d.]+))?")


def loss_curve(run_dir: str):
    """Per-epoch losses, from the run logs AND the supervisor log.

    The supervisor mirrors every completed epoch into its own log, which makes
    it the authoritative source: earlier milestone legs used to write
    train_attempt0.log and a later leg could overwrite it, losing those epochs
    from the run dir entirely.
    """
    rows = {}
    sources = glob.glob(str(pathlib.Path(run_dir) / "*.log"))
    for f in sources:
        for line in pathlib.Path(f).read_text(errors="ignore").splitlines():
            m = EPOCH_RE.match(line.strip())
            if m:
                rows[int(m.group(1))] = dict(
                    epoch=int(m.group(1)), secs=int(m.group(2)),
                    train=float(m.group(3)),
                    val=float(m.group(4)) if m.group(4) else np.nan)
    sup = CAMP / "supervisor.log"
    if sup.exists():
        sup_re = re.compile(
            r"epoch (\d+)/\d+ (\d+)s train=([\d.]+)(?: val=([\d.]+))?")
        for line in sup.read_text(errors="ignore").splitlines():
            m = sup_re.search(line)
            if m:
                ep = int(m.group(1))
                rows.setdefault(ep, dict(
                    epoch=ep, secs=int(m.group(2)), train=float(m.group(3)),
                    val=float(m.group(4)) if m.group(4) else np.nan))
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["epoch"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = json.loads((CAMP / "chain_status.json").read_text()) if (
        CAMP / "chain_status.json").exists() else {}
    cover = {int(k.split("ep")[-1]): v
             for k, v in (st.get("cover_aucs") or {}).items() if v}
    blob_new = {int(k.split("ep")[-1]): v
                for k, v in (st.get("blob_aucs") or {}).items() if v}

    rows = []
    for ep in sorted(set(list(cover) + [30, 50, 75, 100])):
        r = {"epoch": ep, "cover_window": cover.get(ep)}
        for arm, d in ARCHIVED.items():
            r[arm] = d.get(ep)
        if r["cover_window"] is not None:
            r["gain_vs_fork"] = r["cover_window"] - FORK_AUC
            if r.get("envelope"):
                r["vs_envelope"] = r["cover_window"] - r["envelope"]
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "auc_table.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"=== stage: {st.get('stage')}  (updated {st.get('updated')}) ===\n")
    print("=== frozen mean_pool TEST AUC ===")
    print(df.to_string(index=False, na_rep="-",
                       float_format=lambda v: f"{v:.4f}"))
    if blob_new:
        print(f"\nblob (resumed from the copied ep56 seed): {blob_new}")

    cur = loss_curve(r"D:\jepa_phase0\runs\cover_random_ep25")
    if not cur.empty:
        cur.to_csv(OUT / "cover_loss_curve.csv", index=False)
        print(f"\n=== COVER loss curve: {len(cur)} epochs "
              f"({cur.epoch.min()}..{cur.epoch.max()}) ===")
        print(cur.tail(6).to_string(index=False,
                                    float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for arm, d in ARCHIVED.items():
        if d:
            e = sorted(d)
            ax[0].plot(e, [d[k] for k in e], "o--", lw=1.6, label=arm, alpha=.75)
    if cover:
        e = sorted(cover)
        ax[0].plot(e, [cover[k] for k in e], "s-", lw=2.6, color="k",
                   label="COVER/window (this run)")
    ax[0].axhline(FORK_AUC, color="grey", ls=":", lw=1.5)
    ax[0].text(30, FORK_AUC + 0.0005, f"fork ep25 = {FORK_AUC:.4f}", fontsize=8)
    ax[0].set_xlabel("pretraining epoch"); ax[0].set_ylabel("frozen TEST AUC")
    ax[0].set_title("Downstream AUC by arm", fontweight="bold")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    if not cur.empty:
        ax[1].plot(cur.epoch, cur.train, label="train", lw=2)
        ax[1].plot(cur.epoch, cur.val, label="val", lw=2)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("smooth-L1 loss")
        ax[1].set_title("COVER/window loss", fontweight="bold")
        ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)

    fig.suptitle("COVER-then-RANDOM campaign — archived arms used the stock crop, "
                 "so they are an imperfect baseline (see deviation ledger)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "campaign_results.png", dpi=135)
    print(f"\nwrote {OUT / 'campaign_results.png'}")


if __name__ == "__main__":
    main()
