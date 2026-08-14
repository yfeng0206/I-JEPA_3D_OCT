"""Figures for the background-signal investigation."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REP = pathlib.Path(r"D:\jepa_phase0\reports\background_signal")
COMP = pathlib.Path(r"D:\jepa_phase0\reports\target_composition\summary.csv")

sig = json.loads((REP / "background_signal.json").read_text())
skill = json.loads((REP / "skill_scores.json").read_text())
comp = pd.read_csv(COMP)

ARM_OF = {"fork": "random", "random": "random", "oracle": "oracle",
          "envelope": "envelope", "blob": "anatomy"}
COL = {"random": "#4363d8", "oracle": "#e6194b",
       "envelope": "#f58231", "anatomy": "#3cb44b"}

rows = []
for r in sig:
    a = r.get("ablation")
    if not a:
        continue
    fam = r["tag"].split("_")[0]
    rows.append(dict(tag=r["tag"], fam=ARM_OF[fam], ep=r["epoch"],
                     v_bg=(a["err_drop_bg"] - a["err_full"]) / a["k_dropped"],
                     v_anat=(a["err_drop_anat"] - a["err_full"]) / a["k_dropped"],
                     err_full=a["err_full"], err_bg=r["err_bg"],
                     err_anat=r["err_anat"], auc=r.get("anatomy_probe_auc")))
df = pd.DataFrame(rows)
df["ratio"] = df.v_anat / df.v_bg

fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))

# 1. marginal value of a context token, by type, over training
for fam, g in df.groupby("fam"):
    g = g.sort_values("ep")
    ax[0, 0].plot(g.ep, g.v_bg * 1e3, "o--", color=COL[fam], label=f"{fam} background")
    ax[0, 0].plot(g.ep, g.v_anat * 1e3, "s-", color=COL[fam], label=f"{fam} anatomy")
ax[0, 0].set_xlabel("pretraining epoch")
ax[0, 0].set_ylabel("error rise per token removed  ($\\times 10^{-3}$)")
ax[0, 0].set_title("Marginal value of ONE context token\n"
                   "(solid = anatomy token, dashed = background token)")
ax[0, 0].legend(fontsize=7, ncol=2)
ax[0, 0].grid(alpha=.3)

# 2. the ratio -- how many background tokens is one anatomy token worth
for fam, g in df.groupby("fam"):
    g = g.sort_values("ep")
    ax[0, 1].plot(g.ep, g.ratio, "o-", color=COL[fam], label=fam, lw=2)
ax[0, 1].axhline(1.0, color="k", ls=":", lw=1)
ax[0, 1].text(32, 1.05, "parity: a background token is worth as much\n"
                        "as an anatomy token", fontsize=7)
ax[0, 1].set_xlabel("pretraining epoch")
ax[0, 1].set_ylabel("value(anatomy token) / value(background token)")
ax[0, 1].set_title("An anatomy context token is worth 4-6 background tokens\n"
                   "-- except in the blob arm, which inverts")
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(alpha=.3)

# 3. skill against the position-only predictor
sk = pd.DataFrame([dict(tag=s["tag"], bg=s["bg"]["skill_vs_pos"],
                        anat=s["anat"]["skill_vs_pos"]) for s in skill])
x = np.arange(len(sk)); w = .38
ax[1, 0].bar(x - w / 2, sk.bg, w, label="background targets",
             color="#666666", edgecolor="k")
ax[1, 0].bar(x + w / 2, sk.anat, w, label="anatomy targets",
             color="#e6194b", edgecolor="k")
ax[1, 0].axhline(0, color="k", lw=1)
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(sk.tag, rotation=18, fontsize=8)
ax[1, 0].set_ylabel("skill vs position-only prediction")
ax[1, 0].set_title("Background targets ARE predicted from context, not position\n"
                   "(0 = no better than the per-cell mean; blob has collapsed)")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(alpha=.3, axis="y")

# 4. background share of the gradient budget vs downstream AUC
AUC = {"random": 0.8746, "oracle": 0.8855, "envelope": 0.8807, "anatomy": 0.8654}
c = comp.set_index("arm")
for arm in ("random", "oracle", "envelope", "anatomy"):
    ax[1, 1].scatter(c.loc[arm, "slots_bg_pct"], AUC[arm], s=190,
                     color=COL[arm], edgecolor="k", zorder=3)
    ax[1, 1].annotate(arm, (c.loc[arm, "slots_bg_pct"], AUC[arm]),
                      textcoords="offset points", xytext=(9, 5), fontsize=9)
ax[1, 1].set_xlabel("% of predicted slots that are BACKGROUND\n"
                    "(= share of the content-blind gradient budget)")
ax[1, 1].set_ylabel("downstream frozen AUC")
ax[1, 1].set_title("Neither extreme wins: the best arms sit near 52-55%\n"
                   "(4 arms, correlational -- not a causal claim)")
ax[1, 1].grid(alpha=.3)

fig.tight_layout()
fig.savefig(REP / "background_signal.png", dpi=140)
print("wrote", REP / "background_signal.png")

df.to_csv(REP / "marginal_token_value.csv", index=False)
print(df.to_string(index=False))
