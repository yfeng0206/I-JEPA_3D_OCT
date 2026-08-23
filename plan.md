# PLAN — GenAI4Health @ NeurIPS 2026

Last updated 2026-08-23 02:30 PDT. Operator asleep; autopilot running.

Deadline: **2026-09-05, 11:59 PM AoE**. Notification 2026-09-29.
Track: Research Paper, 9 pages excluding references and appendix, double-blind,
no rebuttal, non-archival. NeurIPS Paper Checklist **not required** (workshop
waives it, verified against the CFP).

---

## Current deliverable

`C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_FINAL.zip`
Mirrored at `paper/dist/OCT_JEPA_GenAI4Health2026_Overleaf.zip` and pushed to
branch `docs/background-signal-findings` for colleague review.

Validated on every rebuild by extracting to a scratch directory and compiling
there with no access to the working tree:

| check | state |
|---|---|
| compiles standalone | pass |
| main content within 9 pages | pass, exactly 9 |
| no undefined citations or references | pass |
| anonymous (double-blind) | pass |
| all referenced graphics present | pass |

Two placeholders remain by design and render red with a dagger: COVER epoch 75
and epoch 100. Phase B is computing them.

---

## Stages

| stage | work | ETA | deliverable |
|---|---|---|---|
| A | finish frozen probes: COVER ep73, random fp32 ep50 | 2026-08-23 ~03:40 | ZIP refresh, COVER ep73 point |
| B1 | COVER f=0.21 pretrain ep73 -> 75, probe | 2026-08-23 ~06:40 | COVER ep75 AUC |
| B2 | COVER f=0.21 pretrain ep75 -> 100, probe | 2026-08-24 ~08:00 | COVER ep100 AUC, arm complete |
| C1 | blob fp32 ep56 -> 75, probe | 2026-08-25 ~09:00 | clean fp32 ep75, replaces excluded 0.862492 |
| C2 | blob fp32 ep75 -> 100, probe | 2026-08-26 ~19:00 | clean fp32 ep100, replaces excluded 0.860364 |
| D | final mock review round, last ZIP | 2026-08-26 ~22:00 | submission-grade package |

Measured rates: COVER ~59 min/epoch; blob fp32 ~68 min/epoch (fp32 targets are
1.66x slower than fp16 per `train_patch.py`); frozen probe 60-80 min.

Driver: `autopilot/sequencer.py`, detached. Serialises the GPU so a trainer and a
probe never contend. Log `D:\jepa_phase0\autopilot_out\sequencer.log`.
Monitor loop: schedule #17, every 2 h.

---

## After every milestone, automatically

`autopilot/refresh_all.py` runs eight stages in dependency order:

1. `p3b_integrate_fp32.py` pick up any new probe
2. `p1b_full_inventory.py` re-label and de-duplicate the evidence base
3. `p1c_stats.py` AUCs, paired bootstrap CIs, DeLong, BH families
4. `subgroup_analysis.py` subgroup and severity join
5. `p7b_gap_trend.py` gap trends, BH across attributes, branch-level
6. `p8_make_assets.py` regenerate every macro, table and figure
7. `tectonic` recompile
8. `p13_build_zip.py` rebuild and validate the archive

Then `findings_delta.py` appends what changed to `FINDINGS_LOG.md`, so a new
result is surfaced rather than silently folded in.

The inventory uses globs (`frozen_meanpool_cover_f021_ep*`,
`frozen_meanpool_blob_fp32_ep*`), so newly written probe directories are picked
up without editing code. Verified 2026-08-23.

---

## Comparators for the pending numbers

Epoch-matched only. Frozen MeanPool + Linear, seed 42.

| epoch | random | intensity | envelope | anatomy-v2 | cover |
|---|---|---|---|---|---|
| 50 | 0.8641 | 0.8740 | 0.8761 | 0.8654 | 0.8643 |
| 75 | 0.8723 | 0.8836 | 0.8803 | pending (Phase C) | pending (Phase B) |
| 100 | 0.8746 | 0.8855 | 0.8807 | pending (Phase C) | pending (Phase B) |

Never cite: `frozen_cover_random_*` (retracted, fp16 targets and
`enc_truncate: window`). Excluded pending replacement: anatomy-v2 ep75 0.862492
and ep92 0.860364, both after the epoch-56 precision splice.

---

## Open items after Phase D

- Label-efficiency curves (1/5/10/25/100 percent) from cached features, ~3 h.
  Never run. Reviewer 2 asked for it.
- DINOv3 / ImageNet I-JEPA external baselines, ~2 h. Code exists at
  `ablation/dinov3_probe/`, never executed.
- Seed replicate: >= 3 paired pretraining continuations from the shared
  checkpoint. This is the only experiment that removes the n=1 limitation and it
  is the top follow-up named by the mock review.

---

## Standing rules

- Never invent a number. Absent `results.json` means PENDING.
- Every quantity in the manuscript is a macro in `auto/auto_numbers.tex`,
  generated from stored per-case predictions. Never hand-edit a number.
- Epoch-matched and precision-matched comparisons only.
- No emoji, no tick or cross symbols.
- Bulk output to D:. C: is near its floor (~19 GB).
- Sub-agent runs use gpt-5.6-sol at xhigh.
- Push progress to `docs/background-signal-findings`. Do not merge to main.
