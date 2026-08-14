# Masking Composition Report — why the anatomy arm underperforms

**Date** 2026-08-13 · **Artifacts** `D:\jepa_phase0\reports\masking_report.xlsx`,
`loss_curves_to_ep50.xlsx`, `mask_stats_fairvision.json`, `mask_stats_goals.json`
· **Scripts** `scripts/mask_composition_probe.py`,
`scripts/mask_composition_probe_goals.py`, `scripts/collect_loss_curves.py`

---

## 1. Headline

The anatomy arm is not solving a harder version of the same task. It is solving a
**different and substantially easier** one, and the way it differs is the most
likely cause of the downstream AUC deficit.

Measured on 10,000 real FairVision slices and again on GOALS with ground-truth
segmentation, with all four arms driven by the production mask code:

| | random | oracle | envelope | **anatomy** |
|---|---:|---:|---:|---:|
| context tokens the encoder sees | 70.3 | 79.9 | 78.4 | **162.0** |
| hidden tokens (unique) | 112.1 | 102.5 | 118.0 | **54.3** |
| predictor slots | 158.4 | 159.1 | 160.1 | **64.0** |
| hidden that is background | 76.9 | 61.6 | 67.0 | **1.6** |

*(FairVision Training, 100 volumes × 100 slices, seed 42, r_t = 1.0)*

The anatomy arm shows the encoder **2.1–2.3× more of the image**, hides **less
than half as much**, and gives the predictor **2.5× less work**. In I-JEPA the
masking ratio *is* the difficulty knob, so this is a large, unintended change to
the pretext task.

**This does not say anatomy-guided masking is a bad idea.** The oracle arm is
also anatomy-guided and it holds the best recorded ep100 AUC (0.8855). The
oracle keeps a *normal* hidden budget. Our arm did not.

---

## 2. Loss curves to ep50

Full per-epoch data in `loss_curves_to_ep50.xlsx` (sheet `comparison_wide`).

| epoch | random tr/val | oracle tr/val | envelope tr/val | anatomy tr/val |
|---:|---|---|---|---|
| 25 | 0.1174 / 0.1197 | — | — | — |
| 26 | — | 0.1186 / 0.1202 | 0.1182 / 0.1191 | — |
| 30 | — | 0.1197 / 0.1242 | 0.1198 / 0.1200 | 0.0629 / **0.1527** |
| 35 | — | 0.1232 / 0.1310 | 0.1184 / 0.1305 | 0.0628 / **0.2477** |
| 40 | — | — | 0.1189 / 0.1374 | 0.0682 / **0.2783** |
| 45 | — | — | 0.1188 / 0.1376 | 0.0708 / **0.3026** |
| **50** | 0.1413 / **0.1423** | 0.1316 / **0.1400** | 0.1216 / **0.1401** | 0.0761 / **0.3163** |

Two things stand out.

**All three prior arms land on the same validation loss at ep50** — 0.1400,
0.1401, 0.1423 — despite masking very differently. Ours is **0.3163, 2.25×
higher**. Validation is pinned to a uniform rectangular collator for every arm
(`src/train_patch.py:452`), so these numbers *are* directly comparable.

**The oracle kills the "anatomy is harder" explanation.** The oracle also
targets retina, and its val loss is 0.1400 — indistinguishable from random and
envelope. So predicting anatomy does not by itself move val loss. Something
else in our configuration does.

Caveat on fidelity: envelope and anatomy are parsed per-epoch from stdout logs.
Random and oracle survive locally only as the sampled tables in
`docs/experiments/pretraining/*.md` (raw logs are in Azure blob). Rows are
labelled `per-epoch` vs `sampled` in the workbook.

Train loss is **not** comparable across arms — each predicts a different number
of cells (64 slots for us vs ~160 for everyone else). Our lower train loss is a
scale artifact, not an advantage.

---

## 3. Mask composition — FairVision, 10,000 slices

Anatomy reference: MIRAGE guide occupancy ≥ 0.25 (the production threshold).
Mean anatomy content of an image: **65.9 / 256 patches**.

| metric | random | oracle | envelope | **anatomy** |
|---|---:|---:|---:|---:|
| **context tokens** | 70.3 | 79.9 | 78.4 | **162.0** |
| … on anatomy | 19.1 | 15.7 | 9.3 | 11.4 |
| … background | 51.1 | 64.2 | 69.1 | **150.6** |
| … % of context that is anatomy | 27.2% | 19.6% | 11.9% | **7.0%** |
| … % of all anatomy left visible | 29.0% | 23.8% | 14.1% | 17.2% |
| **hidden tokens (unique)** | 112.1 | 102.5 | 118.0 | **54.3** |
| … on anatomy | 35.2 | 40.9 | 51.0 | 52.7 |
| … background | 76.9 | 61.6 | 67.0 | **1.6** |
| … % of hidden that is anatomy | 31.4% | 39.9% | 43.2% | **97.0%** |
| … % of all anatomy hidden | 53.3% | 62.0% | 77.4% | **79.9%** |
| **predictor slots** | 158.4 | 159.1 | 160.1 | **64.0** |
| … duplicate slots | 46.3 | 56.6 | 42.0 | 9.7 |
| context as fraction of grid | 0.274 | 0.312 | 0.306 | **0.633** |
| hidden as fraction of grid | 0.438 | 0.400 | 0.461 | **0.212** |

## 4. Mask composition — GOALS, ground-truth segmentation

Same measurement with MIRAGE removed entirely: the guide *is* the ground-truth
label map, so this isolates masking **policy** from segmentation error.
100 GOALS B-scans × 10 mask draws = 1,000 samples. Mean anatomy: 43.8 / 256.

| metric | random | oracle | envelope | **anatomy** |
|---|---:|---:|---:|---:|
| context tokens | 76.7 | 82.3 | 83.9 | **172.5** |
| … on anatomy | 11.0 | 9.9 | 7.6 | **4.4** |
| … background | 65.7 | 72.4 | 76.3 | **168.1** |
| … % of context that is anatomy | 14.3% | 12.0% | 9.0% | **2.5%** |
| … % of all anatomy left visible | 25.1% | 22.6% | 17.3% | **10.0%** |
| hidden tokens (unique) | 111.4 | 104.6 | 109.3 | **40.2** |
| … on anatomy | 27.0 | 28.6 | 33.0 | 38.8 |
| … background | 84.4 | 76.0 | 76.4 | **1.4** |
| … % of hidden that is anatomy | 24.2% | 27.3% | 30.2% | **96.4%** |
| … % of all anatomy hidden | 61.7% | 65.3% | 75.4% | **88.6%** |
| predictor slots | 156.7 | 160.3 | 161.9 | **64.0** |
| … duplicate slots | 45.3 | 55.8 | 52.5 | 23.8 |

Ground truth reproduces the FairVision pattern exactly, so **none of this is a
MIRAGE segmentation artifact — it is the policy itself.**

---

## 5. What this means

### 5.1 The hypothesis "harder but narrow" is half right

Narrow: **yes, extremely.** 97% of hidden cells are anatomy, versus 24–43% for
every other arm. The arm predicts essentially nothing but retina.

Harder: **no, the opposite.** More context, fewer targets, fewer predictor
slots. Every axis that controls difficulty moved toward *easier*.

### 5.2 The context is not "more informative" — it is mostly black

The expectation was that our context is larger but richer in anatomy. The
measurement says the reverse:

* On FairVision our context keeps **11.4 anatomy cells vs random's 19.1** — we
  keep *fewer* anatomy cells in absolute terms, and 150.6 of our 162 context
  tokens are background.
* On GOALS it is starker: **4.4 anatomy cells out of 172.5 context tokens
  (2.5%)**, and only **10.0% of all anatomy cells remain visible** — the lowest
  of any arm.

So the encoder is handed a large, overwhelmingly empty context and asked to
reconstruct the retina from it. That is close to unconditional generation of an
average retina, which is exactly the kind of task a model can solve with a
prototype rather than by learning discriminative structure — and it would show
up as **low train loss with a weak encoder**. That is what we observe
(train 0.076 vs envelope 0.122; AUC 0.8654 vs 0.8761).

### 5.3 The predictor is doing a quarter of the work

64 slots vs ~160. Worse, `resample_to_k` (`src/masks/utils.py:20`) pads short
targets **with replacement**, so on GOALS **23.8 of those 64 slots are duplicate
cells (37%)** — only ~40 distinct predictions per image. On FairVision it is
9.7 duplicates (15%). The other arms also have duplicate slots, but theirs come
from target blocks legitimately overlapping, not from padding.

### 5.4 The comparison is confounded

The ep50 result (**anatomy 0.8654 vs envelope 0.8761**) cannot be read as
"anatomy-shaped targets are worse than rectangles". Three variables moved at
once: **target shape**, **hidden fraction** (0.21 vs 0.46), and **anatomy/background
balance of targets** (97% vs 43%). The oracle arm is the proof that anatomy
guidance alone is not the problem.

---

## 6. Downstream AUC — and the ep50 question

> *"we have epoch 50 — do all arms at ep50; if they beat at 100 do they beat here too?"*

**We cannot answer this yet, and not for a subtle reason: the checkpoints do not
exist locally.**

| arm | ep30 | ep50 | ep75 | ep100 |
|---|---:|---:|---:|---:|
| random | — | **not measured** | — | 0.8746 |
| oracle | — | **not measured** | — | 0.8855 |
| envelope | 0.8539 | **0.8761** | 0.8803 | 0.8807 |
| anatomy (pre-bridge) | 0.8583 (5 seeds) | — | — | — |
| anatomy (bridged) | — | **0.8654** | — | not run (stopped ep56) |

Only **envelope** and **ours** have a measured ep50. A filesystem search found
just one non-envelope/anatomy checkpoint on disk —
`fairvision-glaucoma\checkpoint-ep25\jepa_patch-random_posfix-ep25.pth.tar`, the
shared warm start. Random and oracle epoch checkpoints live in Azure blob
(`ijepa-results/patch_vit_base_ps16_ep100_bs64_lr0.00025_20260411_063607` and
`..._bs32_lr0.00025_20260602_093108`). The oracle doc even lists the ep50/75/100
probe as *planned, not done*.

And no, ep100 ranking cannot be assumed to hold at ep50. The envelope gains
+0.0046 from ep50→ep100 while the three ep100 numbers span only 0.0109 — the
arms are closer together than the amount each still moves, so ordering can swap.

**To close this properly:** download random ep50 and oracle ep50 from blob and
run the same frozen mean-pool probe (~1 h each). That yields a true four-way
matched-epoch table.

---

## 7. Recommended next experiment

Match the hidden budget, then the shape claim becomes testable.

| knob | now | proposed | rationale |
|---|---|---|---|
| `pred_target_k` | 16 | **30** | brings hidden cells 54 → ~110, matching the other arms |
| `num_pred_masks` | 4 | 4 | keep, so only one variable moves |
| target composition | 97% anatomy | consider `r_max` 0.5–0.7 | restores background targets and full-field gradient |

`pred_target_k: 30` is the single highest-value change: it fixes hidden fraction
and predictor workload together, leaves target *shape* as the only remaining
difference from envelope, and needs no code change — only config.

A cheap parallel win: run the random-ep50 and oracle-ep50 probes so the
matched-epoch table is complete regardless of how the next pretraining run goes.

---

## 8. Method notes and honest limits

* All arms use the production classes verbatim — `MaskCollator` for random,
  `CurriculumMaskGenerator` for the other three — at `r_t = 1.0`
  (`set_epoch(50, 100)`, past `T_total=30`).
* FairVision uses the real `GuidedOCTSliceDataset` with the production paired
  random-resized-crop, so the numbers include augmentation. GOALS resizes
  directly to 256 so the ground-truth reference stays exact; that is why the two
  tables differ slightly in absolute counts.
* Both tables use the same anatomy definition: **patch coverage ≥ 0.25**.
* FairVision scores against MIRAGE's *predicted* guide, so it inherits any
  segmentation error. GOALS does not — and agrees. That agreement is what lets
  us attribute the effect to policy.
* Envelope in production used an older guide cache
  (`fairvision-glaucoma\mirage_guides`) than the anatomy run
  (`mirage_soft_guides\base512_enc_...`). This probe drives **both** from the
  newer schema-2 cache so the policy comparison is clean; the production
  envelope run is therefore not bit-identical to the envelope row here.
* Mask draws are stochastic. FairVision averages 10,000 draws, GOALS 1,000
  (100 B-scans × 10 repeats).
* All AUC numbers for bridged anatomy are **single-seed (42)**. The one 5-seed
  measurement in the programme (anatomy-vs-envelope at ep30) had a spread of
  ±0.0003–0.0018, so the −0.0107 gap at ep50 is far outside seed noise — but a
  multi-seed confirmation would harden it.
