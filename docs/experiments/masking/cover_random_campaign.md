# COVER-then-RANDOM campaign — locked run plan

**Status:** LOCKED, awaiting fire-ready approval. No training has started.
**Owner config:** `configs/patch_cover_random_ep25.yaml`
**Run dir:** `D:\jepa_phase0\runs\cover_random_ep25`  **tag:** `jepa_patch_cover_random`
**Fork point:** `D:\jepa_phase0\fairvision-glaucoma\checkpoint-ep25\jepa_patch-random_posfix-ep25.pth.tar`
(the common ancestor of random, oracle, envelope and blob — frozen AUC 0.8487)

---

## 1. What this arm is

Stock patch-level I-JEPA in every respect **except where the four target
rectangles are placed**. Placement is greedy: rectangles are positioned to hide
anatomy (from the frozen MIRAGE guide) until 85% of the anatomy soft-mass is
covered, subject to two hard floors that keep ≥15% visible; any block not needed
for coverage is placed as a **plain uniform I-JEPA rectangle**, restricted to
windows that keep the floor intact.

Unchanged from stock: block count (4), block sizes and how they are drawn,
rectangle shape, aspect ratio, `enc_mask_scale`, `min_keep`, `allow_overlap`,
the encoder-mask construction, the loss, the EMA schedule, and `pred_target_k`
(deliberately unset, so the stock global-min truncation applies exactly as in
random/envelope).

### Why this arm exists

Earlier work this session established, forward-only over frozen checkpoints:

- The predictor's query for a target is `mask_token + pos_embed[j]`; the mask
  token is one shared vector, so the query carries **position only** and all
  content must be retrieved by attention over context.
- The loss is `smooth_l1` with `reduction='mean'` — **content-blind**, so the
  fraction of target slots on background *is* the share of the gradient budget
  spent on background.
- **Background targets are real supervision**: skill against a position-only
  predictor is +0.585…+0.680 across arms, and background representations have
  *higher* effective rank than anatomy (22.5 vs 12.8).
- **The blob arm collapsed**: ep30→ep56 its error tripled (0.105→0.289), the
  marginal value of an anatomy context token fell to statistical zero (0.27σ vs
  5–8σ elsewhere), the anatomy/background ratio inverted to 0.74×, and skill
  over a position-only predictor fell to +0.13. Blob spends only **1.8%** of its
  target slots on background.

COVER-then-RANDOM keeps a fraction of the four blocks as plain random
rectangles, so background prediction stays in the loss (53.7% of slots) while
anatomy coverage stays high (78.7%).

**Measured block composition** (production sampler, 1,273 images, full ramp,
`tau=0.10`) — these are the authoritative figures; an earlier standalone probe
reported 2.95/1.05 but used `tau=0.30` and a non-production block-size draw:

| block role | blocks of 4 | share |
|---|---|---|
| `cover` — placed greedily to hide anatomy | **3.32** | 83.1% |
| `random_legal` — plain uniform rectangle, floor-constrained | **0.68** | 16.9% |
| `random_violation` — floor unsatisfiable, image discarded by caller | 0.00 | 0.04% |

Note these are two *independent* mechanisms and should not be conflated:
`cover_fill` governs **target placement**; `enc_truncate` (§3) governs which
**encoder context** tokens survive the batch crop and never touches targets.

---

## 2. Locked configuration

| knob | value | note |
|---|---|---|
| `mode` | `mirage_cover` | |
| `cover_fill` | `random_legal` | leftover blocks = uniform rectangles, floor-constrained |
| `cover_leave_frac` | 0.15 | soft stop |
| `cover_min_visible_frac` | 0.15 | HARD floor (mass **and** occupancy) |
| `cover_min_visible_cells` | 4 | HARD floor |
| `anatomy_tau` | 0.10 | |
| `enc_truncate` | **`window`** | see §3 |
| `amp_target` | `false` | see §4 |
| `pred_target_k` | unset | stock global-min truncation |
| `epochs` | 100 | `save_every: 5` → checkpoints at 30/50/75/100 |
| `T_warm` / `T_total` | 25 / 30 | same ramp window as envelope/oracle forks |
| guide cache | `C:\jepa_data\mirage_soft_guides\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy` | **SSD**, shared with blob |
| slice cache | `C:\jepa_data\slice_cache` | **SSD**, shared with blob |

Both caches are arm-independent (guides are frozen MIRAGE outputs keyed by a
content hash; slices are raw decoded B-scans). **Nothing needs rebuilding and
nothing in the hot path is on the HDD.**

---

## 3. The encoder-crop fix (`enc_truncate: window`)

**The defect.** To make a batch rectangular, the collator cuts every encoder
mask to the batch-wide minimum length with `t[:min_len]` on *sorted* indices.
That is a **row-major prefix**: it always keeps the top of the image and deletes
the bottom. The retina is a roughly horizontal band, so on slices where it sits
low the crop removes **all** anatomy from the encoder's context.

**Measured** (production sampler, full ramp):

| stage | anatomy in context | zero-anatomy |
|---|---|---|
| eligible, before the crop | 13.8 cells (21.1%) | **0.00%** |
| after the stock `prefix` crop | 6.3 cells (9.6%) | **14.48%** |

COVER's own floor is exact — 0.00% before the crop. The crop then discards
**54% of the eligible anatomy**. With a full-image encoder block the crop still
causes 10.6%, so the block explains ~4 pts and the crop owns the rest.

This is a defect in the **shared collation path**, not something this arm
introduces — envelope is hit at 7.4%.

**The fix.** `window` takes a **contiguous run** of the same sorted context
indices — so the retained context is still one coherent region, exactly as in
stock — but slides the run to the offset retaining the most anatomy instead of
always starting at index 0.

| crop | anatomy hidden | anatomy in context | zero-anatomy | context shape |
|---|---|---|---|---|
| `prefix` (stock) | 78.9% | 9.9% | 14.38% | coherent, always the top |
| **`window`** | **78.8%** | **12.5%** | **1.93%** | **coherent band** |
| `random` / `guard` | 78.8% | 8.7 / 11.6% | 3.18 / 2.94% | scattered speckle — rejected |

`window` is free: identical coverage, 7.4× fewer failures, *more* anatomy
context. `random`/`guard` were rejected because they destroy the spatial
coherence of the context block, which stock I-JEPA guarantees.

Raising the floor is a second, costlier dial (0.20 → 0.26% zero but 74% hidden;
0.25 → 0.10% but 69% hidden). **We keep floor 0.15** and take the crop fix only.

---

## 4. Deviation ledger

Everything that differs from the code the comparison arms were trained under.
This exists so the writeup can be honest rather than reconstructed later.

| deviation | status | rationale |
|---|---|---|
| `enc_truncate: window` | **ENABLED** | Fixes a defect corrupting ~14–16% of samples. Same operation as stock (contiguous run of sorted indices), only the offset moves. **BUT it selects that offset using the MIRAGE guide**, so it is a SECOND anatomy-guided intervention — see the attribution limit below. |
| fused SDPA attention | **REVERTED** | Forward is identical (cos 1.0000) but under fp16 the **input-gradient cosine is 0.9907**. ~1% gradient deviation is unsafe at the ~0.002–0.005 AUC effects being compared. `USE_SDPA = False`. |
| `amp_target` (fp16 teacher) | **ENABLED** | 1.66× (68 → ~45 min/epoch). Changes the regression targets, but only numerically (cos 1.0000, mean diff 2.5e-04 on layer-normed features, versus ~0.4%/step EMA drift). |
| occupancy floor in COVER | ENABLED | The floor was enforced only on the soft-score support, a superset of the occupancy mask the audits use; 24.2% of slices fell below the advertised 15%. Now 0%. |
| `iterations_per_epoch` ceil | FIXED | Floor division left the LR/WD/EMA fast-forward one step behind per resumed epoch. |
| crop RNG per worker/rank/epoch | FIXED | Every DataLoader worker previously rebuilt the generator from the same constant, replaying one correlated stream. |

### Attribution limit — state this in any writeup

The 2×2 controls (`random/window` and `COVER/prefix` at ep50) were **considered
and dropped** to save ~34 h. Consequently this arm differs from the archived
baselines in **two** guided ways at once: where the target rectangles go, and
which contiguous slice of encoder context survives the batch crop. **If it beats
envelope, the gain cannot be cleanly attributed between the two.**

A guide-free crop was tested as a way out and is **worse than stock**
(21.1% zero-anatomy vs 16.0% for `prefix`), because a uniformly random band
misses the retina more often than the top of the image does. So there is no
cheap escape: the crop fix necessarily consults the guide.

**Known consequence:** the archived envelope numbers (ep30 0.8539, ep50 0.8761,
ep75 0.8803, ep100 0.8807) were produced under `prefix`, fp32 targets and the
explicit attention path. They are an **imperfect** baseline for this run.

---

## 5. Validation evidence (all passing on the locked config)

| gate | result |
|---|---|
| preflight | **PASS** (on-anatomy 31.7% vs 25% rectangle threshold) |
| preflight regressions (`patch_cover_ep25`, `patch_anatomy_v2`) | **PASS**, unchanged |
| edge-case gate (50 hardest slices) | **0 failures**, 0 zero-visible, 0 fallbacks, exit 0 |
| 6k-slice scale validation | min visible **15.0%**, **0** below floor, **0** zero-visible |
| target cardinality | 39.82 cells/block — identical to random/envelope |
| block shape | bbox fill 1.000 at batch 160; 0.980 at 6k — **identical to stock** (blob is 0.572) |
| checkpoint load | `strict=True` OK |
| RNG consumption across fill modes | identical shared-RNG state |

**Fallback rates over 6,000 slices:** MIRAGE guide failed QC → stock uniform
masking **0.47%**; slice has no anatomy at all 0.05%; COVER `ok=False` ~0.17%.
**≈99.4% of slices get genuine anatomy-guided targets.**

---

## 6. Schedule and ETAs

**Execution model: sequential milestones.** Training holds ~98% of the 24 GB
card, so a frozen-AUC probe cannot run alongside it. `scripts/campaign_chain.py`
therefore trains to a milestone, stops the trainer cleanly, probes that
checkpoint, and resumes for the next leg.

The stop is implemented via the supervisor's `--stop_after_epoch`, **not** by
shortening `optimization.epochs`. That value sizes the cosine LR/WD/EMA
schedules; setting it to 30 for the first leg would anneal the learning rate to
`final_lr` at epoch 30 instead of at epoch 100 and silently change the
experiment. It stays at 100 for every leg, and resume fast-forwards the
schedules to the correct position (now correct, after the ceil fix in §4).

`amp_target` is **ON**. The 2×2 controls were dropped (see §4), so this run is
compared against archived arms rather than against freshly generated controls;
`amp_target` is one more entry in the ledger and buys 1.66× (68 → ~45 min/epoch).

| leg | stage | ETA |
|---|---|---|
| 1 | COVER ep26→30 | Aug 14 ~10:40 |
| 2 | **frozen AUC @ ep30** | ~11:40 |
| 3 | ep30→50 | Aug 15 ~03:00 |
| 4 | **AUC @ ep50** — key comparator | ~04:00 |
| 5 | ep50→75 | Aug 16 ~01:00 |
| 6 | **AUC @ ep75** | ~02:00 |
| 7 | ep75→100 | Aug 16 ~22:00 |
| 8 | **AUC @ ep100** | ~23:00 |
| 9 | blob resume ep56→100, from the COPIED seed | Aug 18 ~12:00 |
| 10 | **AUC @ blob ep75 / ep100** | Aug 18 ~14:00 |

**≈3.5 days.**

Comparators to report against, per epoch:

| epoch | random | oracle | envelope | blob |
|---|---|---|---|---|
| 25 (fork) | 0.8487 | — | — | — |
| 30 | — | — | 0.8539 | 0.8583 (anatomy v1) |
| 35/40 | — | — | — | 0.8661 / 0.8683 |
| 50 | 0.8641 | 0.8740 | **0.8761** | 0.8654 |
| 75 | 0.8723 | 0.8836 | 0.8803 | — (this campaign) |
| 100 | 0.8746 | 0.8855 | 0.8807 | — (this campaign) |

---

## 7. Risks and mitigations

| risk | likelihood | mitigation |
|---|---|---|
| Crash / power loss mid-run | high over 155 h | `-last` checkpoint written **every epoch**; resume verified by a kill+restart test in P0 before firing |
| DataLoader worker OOM | medium (there is a `train_run1_ep26-32_oom.log` precedent) | monitor RSS; drop `num_workers` 10→8 if it climbs; batch 64 uses only 7.4–9.9 GB VRAM |
| Silent stall (no progress) | low | monitor asserts epoch time within 1.5× of the 68 min baseline |
| Val loss diverges (collapse like blob) | medium — this is the thing being tested | monitor compares val loss against archived envelope/random curves each epoch; **abort rule** below |
| Disk fills | low | 15 checkpoints × 1.4 GB ≈ 21 GB; D: has 783 GB free |
| Comparability challenged in review | certain | deviation ledger (§4) written up front |

**Abort rule.** If val loss at ep30 exceeds the archived envelope ep30 val loss
by >20%, or the frozen AUC at ep30 is below the fork baseline (0.8487), stop and
report rather than burning 80 more hours. Blob's failure was visible in exactly
these signals by ep40.

---

## 8. Monitoring and crash-safety

**A blocking defect was found during P0 and fixed.** `train_patch.py` resumes
from whatever `meta.read_checkpoint` names, and this config names the **ep25
fork**. A crash at epoch 60 followed by a naive restart would therefore replay
from epoch 26 and silently discard ~38 hours. The trainer *does* write a rolling
`<tag>-last.pth.tar` every epoch (atomically, `.tmp` + `os.replace`), so a
resume point exists — nothing pointed at it.

`scripts/campaign_supervisor.py` closes this. It:

- scans the run dir for `<tag>-last.pth.tar`, reads its epoch, and rewrites
  `read_checkpoint` to it before every launch (verified: correctly read epoch 56
  from the frozen blob run, rewrote the path, and preserved `epochs: 100`,
  `enc_truncate: window`, `cover_fill: random_legal` and the output folder);
- falls through to the configured fork when no rolling checkpoint exists yet;
- restarts on crash up to `--max_restarts` (default 8), printing the last 25 log
  lines each time;
- **refuses to restart if a restart made no epoch progress**, so a deterministic
  failure aborts instead of spinning;
- stops as soon as the rolling checkpoint reaches the target epoch.

Per-epoch it records wall time, train/val loss and the `[COVER]` sampler stats,
writing `health.json`, and flags:

| flag | condition |
|---|---|
| `SLOW` | epoch time > 1.5× the 4,500 s baseline |
| `VAL LOSS HIGH` | val loss > 1.20× the archived baseline at the same epoch |
| `FLOOR REGRESSION` | `cover_floor_ok` < 0.999 |

Launch:

```
python scripts/campaign_supervisor.py --config configs/patch_cover_random_ep25.yaml
```

---

## 9. Deferred / rejected, with reasons

- **Mask caching** (requested). **Not implemented — measured as not a
  bottleneck.** Mask generation is 1.37 ms/slice = 14.7% of CPU work, and the
  input pipeline has **6.1× headroom** over the GPU (60 ms CPU vs 348 ms GPU per
  batch). The run is GPU-bound; caching masks would buy ≈0.
- **More CPU workers.** Same reason — 6.1× headroom already.
- **Moving caches to SSD.** Already there; the HDD holds only the raw volumes
  used to *build* the caches, plus checkpoint output.
- **Larger batch.** +4% throughput for 2.5× the memory. Not worth it.
- **Eroded-background probe.** Unrelated to this campaign but still the cheapest
  high-value experiment (~10 min/arm, no training); can slot into an AUC gap.
- **Raising the visibility floor to 0.20/0.25.** Would take zero-anatomy to
  0.26%/0.10%, but costs 5–10 pts of anatomy coverage — the quantity the arm
  exists to maximise. Rejected in favour of the crop fix alone.

---

## 10. Open decision

`amp_target` on/off — 58 hours versus one more entry in the deviation ledger.
Currently **off**. This is the only unresolved item; everything else is locked.
