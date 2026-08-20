# Crop and precision audit — encoder-context crop, teacher precision, and the COVER confounds

**Status:** Audit complete. The COVER-then-RANDOM campaign (`ep26→100`) and the
blob `ep57→92` resume are **RETRACTED as comparable evidence** — both carried
the `amp_target` defect (Finding 1); COVER additionally carried
`enc_truncate: window` (Finding 2) and a guide-cache mismatch versus envelope
(Finding 8). No training, no GPU job, and no new pretraining run was started
this session. Every number below comes from a forward-only mask-statistics
probe, a config diff, or a repo grep over frozen checkpoints/configs.

**Date:** 2026-08-18
**Companion docs:** [`cover_random_campaign.md`](cover_random_campaign.md) (run
plan; see its new RESULTS AND RETRACTION section),
[`background_signal.md`](background_signal.md),
[`mask_composition_report.md`](mask_composition_report.md),
[`comparison.md`](comparison.md)
**Scripts:** all new this session, listed in [§12](#12-scripts-created-this-session)

## Evidence language used here

- **MEASURED** — observed this session under a stated script, slice count and config.
- **CONFIRMED** — corroborated by a second independent source (e.g. a log file).
- **RETRACTION** — an earlier claim in this repo is walked back; the earlier text is quoted or cited.
- **REJECTED** — a proposed mechanism was evaluated and not adopted; the reason is stated.
- **UNVERIFIED** — measured under a condition that does not exactly match production; treat as a bound, not a result.
- **DECIDED** — the next action, locked pending one named confirmation.

## Executive summary

| # | Finding | Verdict |
|---|---|---|
| 1 | `amp_target` (fp16 teacher) silently enabled for COVER and blob resume | MEASURED defect, now fixed repo-wide |
| 2 | `enc_truncate: window` is a second anatomy-guided intervention, unique to COVER | MEASURED confound |
| 3 | The row-major prefix crop is stock I-JEPA collation, not a local bug | **RETRACTION** of earlier framing |
| 4 | Full six-way arm mask statistics, one pass, matched RNG | MEASURED |
| 5 | Oracle-fallback rescues ~81% of COVER's zero-anatomy slices | MEASURED, **not implemented** |
| 6 | Oracle fallback rejected — it would be a fourth, COVER-only fallback mechanism | REJECTED |
| 7 | Blanking is 3-way draw-dependent; offline tagging needs ~half the dataset | MEASURED |
| 8 | Three MIRAGE guide sets in use; COVER matches blob's guide, not envelope's | MEASURED, pre-existing confound |
| 9 | Config diff of COVER vs envelope: exactly one unexpected difference (`amp_target`), now fixed | MEASURED |
| 10 | Frozen AUC table; every COVER number is fp16-teacher contaminated | MEASURED, flagged non-comparable |
| 11 | Zero-anatomy accounting figure balances exactly; figure itself is not bit-reproducible | MEASURED, open item |

---

## Finding 1 — `amp_target` precision confound (a real defect in real runs)

`src/train_patch.py:482`: `amp_target = bool(meta_cfg.get('amp_target', False))`.
It controls whether the EMA **target/teacher** encoder forward runs under fp16
autocast.

- The **student/context** encoder has ALWAYS run under fp16 autocast in every
  arm. Only the teacher differs. This is specifically a teacher-forward
  precision change, not "fp16 vs fp32 training".
- `layer_norm` is always taken in fp32 (`h.float()`); validation is always fp32.
- Default is `False`. The in-code comment (`src/train_patch.py`) reads: *"Off
  by default because every previously trained arm used fp32 targets."*
- Measured numerical effect of enabling it: cosine **1.00000000**, mean
  `|diff|` **2.5e-04** on layer-normed targets, versus **~0.4%/step** EMA
  drift — the perturbation is ~16× smaller than the teacher's own intended
  per-step movement.
- Speed benefit: **1.66×** (68 → ~45 min/epoch).

**Who was affected:**

| run | `amp_target` |
|---|---|
| COVER ep26→100 | `true` (affected) |
| blob ep57→92 resume | `true` (affected) |
| random, oracle, envelope, blob ep1→56 | `false` (fp32 targets, archived baselines) |

**Consequence:** the COVER frozen AUCs are not comparable to the baselines. The
blob ep57→92 checkpoints sit on the far side of a mid-run precision switch and
do not cleanly continue the ep35/40/50 series.

**Remediation applied this session:** `amp_target` set to `false` in
`configs/patch_cover_random_ep25.yaml`, `D:\jepa_phase0\campaign\patch_blob_resume.yaml`,
and the two generated `*_resume.yaml` files, each with a provenance comment
recording what the archived run actually used. Repo-wide grep now returns
**zero** `amp_target: true`.

**Process note, stated plainly:** the comment block directly above the setting
in the COVER config already said *"a single arm must not differ in how its
targets are computed. Turn on only when every arm in the comparison uses it."*
The warning was written and then violated on the next line.

---

## Finding 2 — `enc_truncate: window` confound

`src/masks/curriculum.py:427`: `self.enc_truncate = str(cfg.get("enc_truncate", "prefix"))`,
with the comment *"Defaults to prefix so previously trained arms remain
reproducible; a run must opt in."*

COVER was the **only** arm in the repo to set `window`. `window` picks the crop
offset by `argmax` over anatomy retained, which requires reading the MIRAGE
guide — making it a **second anatomy-guided intervention** layered on top of
the masking method being tested.

---

## Finding 3 — the crop is stock I-JEPA, not a local bug (corrects earlier claims)

`src/masks/multiblock.py`, `_truncate_and_stack`:

```python
min_len = min(t.numel() for t in group)
...
t[:min_len]
```

This is in the plain `MaskCollator` adapted from the original I-JEPA. **Earlier
documentation in this repo framed this as a defect specific to this project**
(`cover_random_campaign.md` §3: *"The defect. To make a batch rectangular, the
collator cuts every encoder mask to the batch-wide minimum length..."*). That
framing is **too strong and is retracted here**: the row-major-prefix
truncation mechanism itself is stock I-JEPA behaviour, not something
introduced by this codebase.

Because indices are row-major (`index = row*16 + col`), sorted order is
reading order, so `t[:min_len]` always keeps the TOP of the image and deletes
the BOTTOM. On OCT B-scans, where the retina is a thin horizontal band that
often sits low, this can delete all anatomy from the encoder context. That
consequence is real and worth engineering around (`enc_truncate: window`
attempts exactly that) — but the mechanism being patched is the same one
stock I-JEPA ships with, not a defect unique to this repo.

**Measured magnitude.** Comparing `batch_size=1` (no truncation possible,
since `min_len` equals the image's own length) against `batch_size=64`. B=1
figures are from 256 slices; B=64 figures from 1,534 slices. Script:
`scripts/arm_stats_table.py`. Epoch 50/100.

| arm | context B=1 | context B=64 | % lost to crop | blank B=1 | blank B=64 |
|---|---:|---:|---:|---:|---:|
| random (stock JEPA) | 108.5 | 69.0 | −36% | 0.00% | 4.63% |
| oracle | 116.8 | 78.3 | −33% | 0.39% | 4.56% |
| envelope | 108.4 | 75.2 | −31% | 1.56% | 10.10% |
| blob | 174.8 | 160.0 | −8% | 2.34% | 1.24% |
| COVER 0.15 | 103.0 | 65.7 | −36% | 2.34% | 11.02% |

**Key interpretation:** the crop removes a similar *fraction* from every
rectangle-based arm, including plain random. What differs is headroom: random
retains ~19 anatomy cells after the crop, COVER retains ~7, so normal variance
lands COVER on zero far more often. Anatomy-guided masking deliberately spends
the margin that protects stock I-JEPA from its own truncation.

Also note: at B=1, with no crop at all, envelope already blanks on 1.56% and
COVER on 2.34% — the targets alone can cover all anatomy on some slices. The
crop multiplies this roughly 5×; it does not create it.

---

## Finding 4 — full arm mask statistics

Measured on 1,534 slices, all arms generated in ONE pass on identical slices
with identical RNG per batch, batch 64, epoch 50/100, all under
`enc_truncate: prefix` except the last column. Script:
`scripts/arm_stats_table.py`.

| metric | random | oracle | envelope | blob | COVER 0.15 prefix | COVER 0.15 window |
|---|---:|---:|---:|---:|---:|---:|
| context tokens | 69.0 | 78.3 | 75.2 | 160.0 | 65.7 | 65.7 |
| … on anatomy | 17.6 | 14.4 | 8.0 | 10.0 | 6.5 | 8.1 |
| … % of context that is anatomy | 25.5% | 18.5% | 10.7% | 6.3% | 10.0% | 12.4% |
| … % of all anatomy left visible | 28.2% | 22.9% | 12.7% | 15.3% | 10.6% | 12.9% |
| … ZERO-anatomy slices | 4.63% | 4.56% | 10.10% | 1.24% | 11.02% | 2.28% |
| hidden tokens (unique) | 113.0 | 103.4 | 118.1 | 53.8 | 122.2 | 122.2 |
| … on anatomy | 35.0 | 40.1 | 50.4 | 52.5 | 51.6 | 51.6 |
| … % of hidden that is anatomy | 31.1% | 39.2% | 42.7% | 97.5% | 43.0% | 43.0% |
| … % of all anatomy hidden | 53.8% | 61.8% | 78.1% | 82.1% | 80.1% | 80.1% |

**Additional COVER floors, under `prefix`, same run:**

| COVER floor | context | … on anatomy | % anatomy visible | ZERO-anatomy % | % anatomy hidden |
|---:|---:|---:|---:|---:|---:|
| 0.15 (from table above) | 65.7 | 6.5 | 10.6% | 11.02% | 80.1% |
| 0.20 | 64.4 | 8.6 | 13.8% | 9.26% | 75.3% |
| 0.25 | 64.4 | 10.8 | 17.2% | 7.63% | 70.3% |
| 0.30 | 67.5 | 13.3 | 21.2% | 5.80% | 65.0% |
| 0.35 | 70.4 | 15.7 | 24.9% | 4.24% | 60.2% |

**Two conclusions:**

(a) Envelope is the WORST-blanking rectangle-family arm at 10.10% and still
scored the second-best AUC (0.8807), so blanking alone does not determine
downstream AUC.

(b) COVER at floor 0.15 under `prefix` sits at 11.02% — statistically
alongside envelope. It was never anomalous, which means the `window`
intervention was not necessary for comparability.

---

## Finding 5 — oracle-fallback feasibility (measured, NOT implemented)

Rule tested offline: generate COVER normally; if a slice has zero anatomy in
context after the crop, substitute that slice's oracle (`anatomical_prior`)
mask. Script: `scripts/cover_oracle_fallback.py`, 1,534 slices.

| floor | COVER blanks | of those, oracle ALSO blanks | rescued | residual blank |
|---:|---:|---:|---:|---:|
| 0.15 | 10.89% | 18.6% | 8.87% | 2.02% |
| 0.20 | 9.26% | 21.8% | 7.24% | 2.02% |
| 0.25 | 7.63% | 25.6% | 5.67% | 1.96% |
| 0.30 | 5.67% | 31.0% | 3.91% | 1.76% |
| 0.35 | 4.11% | 25.4% | 3.06% | 1.04% |

Failures are correlated (oracle blanks 18.6% of COVER's bad slices vs its
4.56% base rate — ~4× enriched) but 81% are still rescued. Metrics after
fallback at floor 0.15:

| metric | value |
|---|---:|
| context tokens | 67.3 |
| … on anatomy | 7.5 |
| … % anatomy visible | 12.2% |
| blank rate | 2.02% |
| hidden tokens | 120.7 |
| … on anatomy | 50.0 |
| … % of hidden that is anatomy | 42.0% |
| … % of anatomy hidden | 77.3% |

**Two blockers, both must be stated:**

1. **UNVERIFIED**: the measurement cropped the oracle mask to *oracle's own*
   `min_len` (~78 context tokens), not COVER's shorter one (~66). Whether the
   rescue survives the tighter cut is unknown, so **2.02% is an upper bound**.
2. `min_len` is a batch-level quantity, so naive substitution is circular (new
   mask length → new `min_len` → new crop for everyone → new blanks). Solvable
   by freezing `min_len` from the COVER masks before substituting, then
   cropping the oracle mask to that same fixed length — single pass, no
   iteration.

---

## Finding 6 — the fallback was REJECTED, and why

Inventory of fallbacks that already exist (from `src/masks/curriculum.py` and
`src/masks/multiblock.py`):

| arm | existing fallback | trigger |
|---|---|---|
| random (stock `MaskCollator`) | `best_indices = all_patches[:min_keep]` | context block cannot honour `min_keep` |
| all curriculum modes | uniform random placement, counted `fallback_invalid` | guide missing or QC-invalid |
| all curriculum modes | uniform random for that block only, counted `infeasible` | no admissible window for a block |
| all curriculum modes | fully uniform, counted `unbiased_by_ramp` | curriculum ramp not yet engaged |
| COVER | uniform random, counted `fallback_invalid` | `cover_info["ok"]` false, hard floors unsatisfiable |
| envelope | explicitly NONE for visibility (see comment at `curriculum.py:364`) | returns best of `mirage_max_attempts`, stays guided |

Observed rates in real training logs: 0–2 per batch of 64.

**Decisive point:** every existing fallback fires at MASK-GENERATION time and
falls back to UNIFORM RANDOM. None fires after the crop. An oracle fallback
would be a new mechanism of a different kind, present only in COVER —
structurally the same error as `enc_truncate: window` (Finding 2): a
COVER-only, guide-consulting intervention layered on top of the method under
test. It was therefore rejected.

---

## Finding 7 — blanking is draw-dependent, so offline tagging cannot work

Script: `scripts/blank_proneness.py`. 256 slices, 12 independent draws each
(pool reshuffled every repeat so crop draw, target placement, and batch
composition all vary). Floor 0.15, prefix. Overall blank rate **6.51%**.

Per-slice frequency of blanking across the 12 draws:

| draws blanked | share of slices |
|---|---:|
| never (0%) | 52.0% |
| 1–25% of draws | 44.5% |
| 25–50% of draws | 3.5% |
| 50–75% of draws | 0.0% |
| >75% of draws | 0.0% |
| always (100%) | 0.0% |

Pure chance with p=0.0651 predicts `(1-0.0651)^12 = 44.8%` never-blank;
observed 52.0%. So concentration exists but is weak.

Tagging the worst K% of slices and routing them to oracle:

| K (tagged) | % of blanks covered | residual blank rate |
|---:|---:|---:|
| 5% | 26.0% | 4.82% |
| 10% | 42.0% | 3.78% |
| 20% | 64.0% | 2.34% |
| 30% | 77.0% | 1.50% |
| 50% | 100% | 0.00% |

**Conclusion:** you would have to tag half the dataset. Three independent
random sources must align badly, all resampled every epoch: `RandomResizedCrop`
at scale (0.3, 1.0); target block placement; and batch composition, which sets
`min_len`. That third one alone makes offline tagging impossible.

**Caveat:** this run's 6.51% base rate is below production's 11.02% because
batches were drawn from a fixed 256-slice pool, narrowing the `min_len`
distribution. The distributional conclusion (weak concentration, three
independent sources) is unaffected.

---

## Finding 8 — three different MIRAGE guide sets are in use (pre-existing confound)

| guides | format | used by |
|---|---|---|
| `D:\jepa_phase0\fairvision-glaucoma\mirage_guides` | `.npz`, ~20–34 KB each, 6000 files | envelope (**CONFIRMED** in its `train.log`: "MIRAGE guides: D:\jepa_phase0\fairvision-glaucoma\mirage_guides") |
| `D:\jepa_phase0\fairvision-glaucoma\mirage_soft_guides\base512_cfg7_3186b1fa278bc97f` | soft | `patch_mirage_anatomy.yaml` (anatomy v1, the ep30 0.8583 arm) |
| `C:\jepa_data\mirage_soft_guides\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy` | soft `.npy` 8.0 MB + `.json` sidecar, 6000 each | `patch_anatomy_v2.yaml` (blob), `patch_cover_ep25.yaml`, `patch_cover_random_ep25.yaml` |

So COVER is guide-matched to blob but **not** to envelope. Note carefully and
neutrally: the programme's headline ep30 result (anatomy 0.8582 ± 0.0003 vs
envelope 0.8528 ± 0.0018, +0.0054, Welch p=0.00219, Cohen's d 4.20, paired
bootstrap +0.0044 CI [+0.0010, +0.0077], p=0.012) compares an arm on the cfg7
soft guides against an arm on the older `.npz` hard guides. This is stated as
a **fact requiring follow-up** — it is **not** a declaration that the ep30
result is invalid.

---

## Finding 9 — config audit result

Script created: `scripts/config_diff_arms.py`, which deep-diffs a candidate
config against a baseline and classifies each difference as intended (masking
method, paths, run identity) or unexpected.

Running COVER against envelope: **exactly ONE unexpected difference**,
`meta.amp_target`, now set to `false`, which matches the baseline's absent
default (Finding 1 remediation).

Also verified:

- `meta.seed` default is 0 (`train_patch.py:149`), so COVER's explicit
  `seed: 0` matches.
- Both slice caches are 4-file memmaps.
- `data.prefetch_factor` is performance-only.
- `crop_scale: [0.3, 1.0]` is identical in COVER and envelope, so it is shared
  and not a confound.

---

## Finding 10 — frozen AUC status

Frozen MeanPool + Linear probe (zero probe params), harness sanity-gated
(random ep100 reproduced 0.8746 exactly on re-run).

| arm | ep30 | ep50 | ep75 | ep100 |
|---|---:|---:|---:|---:|
| fork ep25 (shared start) | 0.8487 (at ep25) | — | — | — |
| random | not measured | 0.8641 | 0.8723 | 0.8746 |
| oracle | not measured | 0.8740 | 0.8836 | 0.8855 |
| envelope | 0.8539 (0.8528 ± 0.0018, 5 seeds) | 0.8761 | 0.8803 | 0.8807 |
| blob (bridged) | — | 0.8654 | not measured | not measured |
| anatomy v1 (pre-bridge) | 0.8583 (0.8582 ± 0.0003, 5 seeds) | — | — | — |
| COVER | 0.8558 | 0.8590 | 0.8612 | 0.8607 |

blob also has ep35 0.8661 and ep40 0.8683. All bridged-anatomy AUCs are
single-seed (42).

**Every COVER number in that row must be marked as fp16-teacher contaminated
and not comparable.** COVER's ep30 seed-42 vs envelope seed-42 (0.8540) is
+0.0018, which is well inside envelope's 5-seed spread of 0.8497–0.8542 and is
therefore meaningless.

**Standing observation:** COVER had the LOWEST pretraining val loss of any arm
(~0.94× envelope) while having the worst downstream AUC — the pretraining
objective did not track downstream quality, consistent with the general rule
that I-JEPA loss is not a quality signal (`docs/lessons_learned.md` general
rule 1).

---

## Finding 11 — figure

`D:\jepa_phase0\reports\arm_stats\zero_anatomy_floor20.png`, produced by
`scripts/show_zero_anatomy_slices.py`. Five columns per failing slice: image
with anatomy contour; the 4 target blocks each in its own colour with
per-block anatomy counts; a three-way budget map (grey = reaches encoder,
cyan = not masked by any target yet still withheld, black = removed by
targets); the encoder context under `prefix`; and the oracle-fallback context.

Its central quantitative result — anatomy accounting balances exactly on every
failing slice:

| slice | anatomy cells | masked by targets | withheld before reaching encoder | reaches encoder |
|---:|---:|---:|---:|---:|
| 28 | 77 | 60 | 17 | 0 |
| 64 | 90 | 61 | 29 | 0 |
| 67 | 86 | 64 | 22 | 0 |
| 72 | 65 | 50 | 15 | 0 |

Raw patch budget on slice 72 (256 patches total):

| category | patches |
|---|---:|
| total | 256 |
| removed by the 4 target blocks | 90 |
| withheld before reaching encoder | 123 |
| actually delivered to encoder | 43 |

Note honestly that the "withheld" category combines crop truncation **and**
cells outside the sampled context block, so it should not be attributed to the
crop alone.

**Open item:** the figure is not bit-reproducible. The mask generators hold
internal `torch.Generator` state that global re-seeding does not reset, so
repeat renders select different slices — measured **11.46% / 10.42% / 9.38%**
on the same 192 slices across three renders. The RATE is stable and consistent
with the 1,534-slice measurement (Finding 4); individual slice IDs are not.
Flag this as an open item.

---

## 12. Scripts created this session

| script | purpose |
|---|---|
| `scripts/arm_stats_table.py` | Computes context/hidden/anatomy mask statistics across arms at matched batch size and epoch (Findings 3, 4). |
| `scripts/cover_oracle_fallback.py` | Measures the rescue rate of substituting the oracle mask on COVER zero-anatomy slices (Finding 5). |
| `scripts/show_zero_anatomy_slices.py` | Renders the per-slice anatomy-accounting figure for zero-anatomy failures (Finding 11). |
| `scripts/blank_proneness.py` | Repeated-draw test of whether zero-anatomy blanking is slice-specific or draw-dependent (Finding 7). |
| `scripts/config_diff_arms.py` | Deep-diffs a candidate training config against a baseline and classifies each difference as intended or unexpected (Finding 9). |
| `scripts/cover_floor_sweep.py` | Paired floor sweep (0.15–0.30) with McNemar statistics against a `random`/`oracle`/`envelope` reference set (see §14, running). |
| `scripts/probe_blob_epochs.py` | Frozen AUC probe across blob epoch checkpoints (Finding 10). |

---

## 13. The decided next run

**DECIDED**, pending one confirmation:

```yaml
mode:                    mirage_cover
cover_min_visible_frac:  0.23        # the only value changed vs the previous run
cover_fill:              random_legal
enc_truncate:            prefix      # default, identical to all baselines
amp_target:              false       # default, identical to all baselines
```

No fallback beyond the shared ones every arm already has (Finding 6). Frozen
AUC planned at ep30/50/75/100. Expected blank rate ~8%, which is **below**
envelope's 10.10%.

**Blob restart, recorded here:** blob needs a clean fp32 restart from
`blob_resume_seed.pth.tar` (ep56, the last fp32 checkpoint), 44 epochs, with
AUC at ep75 and ep100. The ep92 rolling checkpoint has been pinned to
`jepa_patch_blob_resume-ep92-pinned.pth.tar`.

## 14. Floor sweep 0.15–0.30 (running)

**PLACEHOLDER — results pending.** A 6,144-slice paired sweep with McNemar
statistics versus a `random`/`oracle`/`envelope` reference set is in progress
via `scripts/cover_floor_sweep.py`. Results will be filled in here once
available.

Why the earlier 1,534-slice fine sweep was inconclusive: SE ~0.7% at p~0.08,
and the blank-rate column was non-monotonic across floors 0.20–0.25:

| floor | 0.20 | 0.21 | 0.22 | 0.23 | 0.24 | 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| blank rate | 9.13% | 8.21% | 9.00% | 6.98% | 8.54% | 7.50% |

This is impossible physically — raising the floor protects more anatomy and
cannot increase blanking — so those wiggles bound the noise at roughly
**±1–2 points**.
