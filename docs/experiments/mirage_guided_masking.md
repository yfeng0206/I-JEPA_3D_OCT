# MIRAGE-Guided Masking (Rung 1b)

Replaces the hand-crafted anatomical **oracle** prior of
[`curriculum_masking.md`](curriculum_masking.md) §5.1 with a real segmentation
model — **MIRAGE-Large**, fine-tuned on GOALS — as the source of the I-JEPA
target-block location prior.

The question is the same as the oracle rung: *does biasing the JEPA target
blocks onto the diagnostic retinal band produce a better encoder than random
masking?* The difference is that the region now comes from a model rather than
from a hand-tuned band, so a positive result is not an artefact of us having
hand-fitted the band to this dataset.

---

## The experimental contract

The **only** variable versus the random and oracle arms is **where the four
target blocks land**. Everything else is held fixed:

| held fixed | value |
|---|---|
| architecture | ViT-B/16, 256×256, pred_depth 6, pred_emb_dim 384 |
| blocks | nenc 1, npred 4, enc scale 0.85–1.0, pred scale 0.15–0.2, AR 0.75–1.5 |
| loss | Smooth-L1 against the EMA target encoder |
| optimiser | AdamW, lr 2.5e-4 cosine → 1e-6, warmup 5, wd 0.04 → 0.4 |
| EMA | [0.996, 1.0] cosine |
| effective batch | 512 |
| curriculum | T_warm 25, T_total 30, r_max 1.0, linear |
| warm-start | random-arm epoch 25 (SHA-256 `e5ad5b0c…e7b`) |

`configs/patch_mirage_envelope.yaml` was diffed field-by-field against
`configs/patch_oracle_anatomical.yaml`. The only differences are paths, the
masking policy itself, and two documented single-GPU accommodations
(`accum_steps` 2→8 to hold the effective batch at 512 on one GPU instead of
four; `num_workers`). Schedule steps per epoch are identical at **1171**
(original 600000/4/64 = 2343, //2 = 1171; ours 600000/64 = 9375, //8 = 1171).

---

## The guide

MIRAGE segments RNFL, GCIPL and choroid. Those are three disconnected bands
with the unlabelled mid-retina between them, which is not usable as a placement
region, so `src/guides/mirage_envelope.py` repairs each slice into a single
connected retinal envelope: drop small/short components, close column gaps,
fill holes, enforce boundary continuity.

![MIRAGE guide pipeline](../../results/masking/mirage_guide_pipeline.png)

*Left to right: the B-scan; MIRAGE's raw three-class output (red RNFL, amber
GCIPL, blue choroid); the raw union, visibly three separate bands; the repaired
envelope as one connected structure; the fractional 16×16 patch occupancy; and
the boolean placement region at threshold 0.25. `runs/col` is the mean number of
separate vertical runs per occupied column — the repair drives it to ≈1.00,
i.e. every column becomes a single unbroken band.*

Repair parameters are frozen and fingerprinted (`9a25a2cdb36f9cba`). Every
cached guide stores its fingerprint and the dataset refuses to load a guide
built with different parameters — a stale cache fails loudly instead of
silently training on the wrong anatomy.

Effect of the repair over the corpus: mean vertical runs per column
**2.133 → 1.002**, area 13.3% → 20.3%.

Guides are precomputed by `scripts/mirage_precompute_guides.py`; all 6,000
training volumes are covered.

## The three arms

![The three arms](../../results/masking/mirage_masking_arms.png)

*Same slice, same crop, same RNG seed, so the four block sizes are identical
across arms and only placement moves. Dimmed = withheld from the encoder;
the four colours are the four target blocks the predictor must reconstruct.
The per-slice on-retina numbers are a three-slice sample and are noisy — only
the 1,000-volume aggregate below is meaningful.*

Regenerate both figures with `python scripts/mirage_doc_figures.py`.

---

## Selected policy: occupancy threshold 0.25, dilation 0

MIRAGE gives a *fractional* retina occupancy per 16×16 patch. The threshold
turns that fraction into the boolean grid a target block may be drawn from.

Measured over **1,000 volumes / 19,987 slices** (`scripts/mirage_method_sweep.py`),
scored against a MIRAGE-free pixel-level tissue reference
(`src/guides/tissue_truth.py`). Chance purity is 0.3749.

| metric | RANDOM | ORACLE | **thr 0.25** | thr 0.50 | thr 0.75 |
|---|---|---|---|---|---|
| purity | 0.4530 | 0.5602 | **0.6320** | 0.6507 | 0.6597 |
| lift vs chance | 1.21× | 1.49× | **1.69×** | 1.74× | 1.76× |
| unique targets (= masked area) | 112.4 | 101.9 | **101.7 (−0.2%)** | 97.8 (−4.0%) | 95.4 (−6.4%) |
| context tokens | 107.6 | 116.6 | **117.2 (+0.5%)** | 121.2 (+3.9%) | 123.8 (+6.2%) |
| images with a random-placed block | — | — | **6.2%** | 10.0% | 15.0% |

**0.25 is not the highest-purity setting — it ranks 5th.** It is chosen because
it is the only setting whose masking *geometry* stays within 0.5% of the oracle
arm on every axis. Higher thresholds score better on purity but mask
progressively **less** of the image, which makes the pretext task easier and
would make any downstream AUC gain unattributable to the masking policy.

### Dilation is monotonically harmful

Tested at every threshold. `+1` costs 6–8 purity points, `+2` costs 12–14. The
best dilated variant (thr 1.00 +1, 0.5994) still loses to the *worst* undilated
one (thr 0.10, 0.6189). The added ring is ~94% void (mean intensity 36.2 and
6.4% bright, vs 65.5 and 76% inside the envelope), so it lets the
`min_retina_visible` rule satisfy itself off-anatomy. Set to **0**.

Purity appears to *rise* with threshold under dilation only because a higher
threshold undoes the dilation — it is a morphological closing back toward the
original envelope.

---

## Bug: blocks were placed from one grid and scored against another

Found in pre-launch review. **This is the most important thing in this
document** because it was completely silent.

Two independent places turn the fractional occupancy into a boolean:

| | builds | file |
|---|---|---|
| dataset | the **placement region** — cells a block may be drawn from | `oct_slices_guided.py:213` |
| collator | the **scoring truth** — what "on retina" and "retina still visible" are judged against | `curriculum.py:564` |

The config set `mirage_occupancy_threshold: 0.25`. The collator read it. But
`train_patch.py` constructed `GuidedOCTSliceDataset` **without** the argument,
so the dataset kept its `0.5` default. Blocks were drawn from the *smaller* 0.5
region and scored against the *larger* 0.25 truth.

Nothing crashed and no logged metric looked obviously wrong.

Paired A/B, 2,560 images at full guidance (`scripts/threshold_fix_masks.py --aggregate`):

| metric | 0.50 (bug) | 0.25 (fixed) | change |
|---|---|---|---|
| admissible region (cells) | 67.03 | 74.85 | +11.7% |
| unique target patches | 115.47 | 120.56 | +4.4% |
| context tokens | 116.52 | 111.82 | −4.0% |
| accept rate | 0.43 | 0.47 | +9.3% |
| **infeasible blocks / batch** | **2.20** | **0.80** | **−63.6%** |

The clearest signal is infeasibility: the 0.5 region was frequently too small to
host a block at all, forcing a uniform-random fallback.

Pinned by `tests/test_mirage_config_wiring.py`, which asserts the kwarg is
wired, that it reads `mirage_occupancy_threshold`, and that the shipped config
still carries the selected policy.

![The threshold bug](../../results/masking/mirage_threshold_bug.png)

*Columns 2–3 are the two regions; column 4 shows in magenta the cells only the
0.25 threshold admits. Columns 5–6 are the resulting masks from an identical
RNG seed, so the four block sizes match and only placement moved.*

Figures: `scripts/masking_explained.py` also renders `predictor.png`, showing
the encoder's context and each of the four target blocks separately.

---

## Bug: 200× read amplification made training disk-bound

The first launch ran at **7.6 img/s with the GPU idle at 0–1% and 30 W** —
about 68 days for the run.

`OCTSliceDataset.__getitem__` did `np.load(npz)["oct_bscans"]`, which decodes
the entire **8 MB** volume to obtain one **40 KB** slice. The training split is
48 GB against 31.9 GB RAM on a 7200 rpm SATA disk, so the page cache thrashed:
51 MB/s ÷ 8 MB = 6.4 vol/s, exactly the observed rate. Benchmarks on a few
hundred volumes missed it entirely because that fits in cache.

Measured GPU ceiling with synthetic tensors: **177.7 img/s**, 0.94 h/epoch.

`scripts/build_slice_cache.py` writes exactly the sampled slices into one flat
`uint8` memmap per split — Training 24 GB, Validation 4 GB, ~12 min to build.
The memmap is opened lazily per worker and excluded from `__getstate__` so the
OS handle never crosses a process boundary.

This is an **I/O layout change only**; the bytes fed to the transform are
identical, and `tests/test_slice_cache.py` asserts bit-equality of both the raw
slice and the fully transformed tensor.

Result: loader 207–301 img/s, run GPU-bound at **173.3 img/s, 0.96 h/epoch** —
a **22.8×** recovery.

---

## Rejected approaches

All of these were measured and discarded. Recorded so they are not re-tried.
The scripts that produced them were removed after this commit and remain
recoverable from git history.

### Centre-anchored placement — rejected

Anchoring all four blocks on the retina masks it almost entirely.
`keep_TRUE` (retina inside the context crop *and* not covered by a target)
falls to **0.0296** — 90–97% of the retina hidden.

The geometry cannot work: four blocks of 38–51 patches against a 55–70 patch
retinal band. Four full-size blocks do not fit inside the retina.

### Centre-anchored + forced trim — rejected

Retry, then trim block edges to force ≥20% visible retina. Reached the floor on
only **37% of slices** (mean 0.108), so the "20% floor" was not a floor. Worse,
trimming changes block size and aspect ratio, so it is no longer a
location-only intervention — it breaks the experimental contract.

### 2 guided + 2 uniform blocks — rejected

Proposed to guarantee context by construction. Measured keep 0.2833 / purity
0.5168 / unique targets 123.1, versus the oracle's 0.2966 / 0.5655 / 102.0 —
**the oracle beats it on all three**, and every variant inflates unique targets
by ~20%.

### Constant-budget swap — rejected as the training method

Uncover the minimum number of retinal target patches to hit a 20% floor, then
re-cover the same number elsewhere so the token budget is preserved.

Over 1,000 volumes / 9,942 slices (`scripts/swap_budget_eval.py`):

| method | keep_TRUE | floor% | **adjacency** | solid% | budget | uniq |
|---|---|---|---|---|---|---|
| thr 0.25 (solid blocks) | 0.2076 | 53% | 0.2511 | 100% | 101.9 | 101.9 |
| centre-anchored, no floor | 0.0287 | 2% | 0.2105 | 100% | 177.3 | 128.1 |
| centre + constant-budget swap | 0.2079 | 100% | **0.4178** | **7%** | 177.3 | 137.1 |

It does hit the floor and preserve per-block token count, but:

1. **Adjacency leak 0.21 → 0.42.** The uncovered patches sit *inside* the
   retina next to surviving targets, so 42% of target patches touch a visible
   context patch. I-JEPA is hard precisely because targets are predicted from
   *distant* context; this halves that distance and makes the pretext task
   easier, so any AUC gain would be unattributable.
2. Only **7%** of blocks remain rectangles.
3. Unique targets rise 128.1 → 137.1, so the per-block budget is constant but
   the **union** is not — masked area is not actually preserved.

Worth revisiting as a *separate* experiment, not as a fix to this one.

### MIRAGE ∪ salience guide — rejected

Salience alone covers 37.0% of the frame; MIRAGE ∪ salience covers 37.1%.
MIRAGE contributes **0.1%**. The guide would stop being MIRAGE-guided.

---

## Known deviations, deliberately not fixed

### Collator truncates targets to the batch minimum

`curriculum.py:1196` takes `global_min_pred = min(t.numel() …)` and truncates
every target to it. Because `_block_to_indices` returns row-major sorted
indices, truncation removes the **bottom row** of a block. Combined with four
*independently* sampled block sizes (mean spread 7.5 patches; all four equal 0%
of the time), the result is that **~37.5% of delivered blocks are not
rectangles** and ~10% of target patches are discarded.

The official `facebookresearch/ijepa` samples **one** `p_size` shared by all
`npred` blocks, so its equivalent truncation is a no-op (measured 0.00% loss vs
our 10.26%). Our `multiblock.py` is a from-scratch reimplementation.

**Not fixed**, because all three arms share this behaviour and the epoch-25
checkpoint was trained with it. Changing it now would introduce a second
variable. Candidate fix for a v2: sample one shared `p_size` across `npred`.

### `mirage_min_retina_visible` is best-effort, not a guarantee

After `mirage_max_attempts` the sampler returns the attempt that left the most
retina visible, with **all blocks still guided**. It does *not* fall back to
uniform, because doing so would inject random-baseline behaviour into the
guided arm and weaken the contrast being tested. Only a genuinely *infeasible*
block (no window anywhere reaches `min_block_fill`) is placed uniformly.

Measured: the 0.25 floor is met outright on ~47% of images; ~6.2% of images
contain a uniformly-placed block. **The threshold policy above was calibrated
with these exact semantics in force.**

### Accumulation tail

The loop steps on the incomplete accumulation window at epoch end. The oracle's
tail is 1 microbatch (256 samples), ours is 7 (448). This cannot be made
identical on one GPU; leaving it differs from the oracle by 3/8 of an average
update, whereas skipping it differs by 4/8 *and* drops an EMA/scheduler step.
Left unchanged.

### Schedule fast-forward on resume

Schedules fast-forward by `start_epoch × 1171` while the loop executes 1172
steps per epoch. The oracle warm-start had the identical off-by-25 at its own
epoch-25 resume, so arm parity holds. Consequence: a resumed run sits 25
schedule steps behind per completed resumed epoch.

---

## Run status

| item | value |
|---|---|
| config | `configs/patch_mirage_envelope.yaml` |
| warm-start | random-arm ep25, SHA-256 `e5ad5b0c…e7b` |
| data | 6,000 training / 1,000 validation volumes, 100 slices each |
| throughput | 173.3 img/s, ~0.96 h/epoch, ~3.0 days for epochs 25→100 |
| monitoring | `python scripts/run_status.py --watch` |

`scripts/run_status.py` prints every completed epoch beside the matching row of
[`pretraining/oracle_100ep.md`](pretraining/oracle_100ep.md) with deltas.

### First segment (epochs 26–32)

| epoch | r_t | train | val | cos_sim | rep_div | oracle train/val |
|---|---|---|---|---|---|---|
| 26 | 0.0 | 0.1182 | 0.1191 | 0.874 | 0.337 | 0.1186 / 0.1202 |
| 27 | 0.2 | 0.1207 | 0.1189 | 0.901 | 0.344 | — |
| 28 | 0.4 | 0.1216 | 0.1200 | 0.896 | 0.372 | — |
| 29 | 0.6 | 0.1215 | 0.1198 | 0.855 | 0.284 | — |
| 30 | 0.8 | 0.1207 | 0.1199 | 0.868 | 0.258 | 0.1197 / 0.1242 |
| 31 | 1.0 | 0.1169 | 0.1266 | 0.878 | 0.252 | — |
| 32 | 1.0 | 0.1170 | 0.1273 | 0.828 | 0.263 | — |

Epoch 26 runs at `r_t = 0` — pure random masking from the same weights as the
oracle's fork point — so it *should* reproduce the oracle, and it does to within
0.0004 train / 0.0011 val. That is end-to-end evidence that the warm-start, the
schedule fast-forward, the slice cache and the loss path are all correct.

**No collapse**: `rep_diversity` 0.25–0.37 against a 1.0 collapse point, and
trending *down* into the oracle's 0.17–0.33 band as guidance engages.

This segment ended in a crash at epoch 32 — Windows error **1455**, the system
commit limit, caused by the validation loader spawning 6 workers on top of the
training loader's 6 while an unrelated analysis job was running. Fixed by
`val_num_workers: 2`. Periodic saves were every 25 epochs so only epoch 27
survived; `save_every: 5` now.

---

## Reproducing

```bash
# 1. guides (once; ~6000 volumes)
python scripts/mirage_precompute_guides.py

# 2. slice cache (once; ~12 min, 28 GB) - removes the 200x read amplification
python scripts/build_slice_cache.py \
    --data-dir  D:\jepa_phase0\fairvision-glaucoma\data \
    --cache-dir D:\jepa_phase0\fairvision-glaucoma\slice_cache \
    --splits Training Validation --num-slices 100

# 3. train
python src/train_patch.py --config configs/patch_mirage_envelope.yaml

# 4. monitor against the oracle reference
python scripts/run_status.py --watch
```

Analysis entry points:

| script | purpose |
|---|---|
| `mirage_method_sweep.py` | the 1,000-volume policy sweep; caches masks so re-scoring under a new tissue truth takes seconds |
| `mirage_method_panels.py` | per-method visual panels |
| `masking_explained.py` | the threshold bug and the predictor's view |
| `threshold_fix_masks.py` | threshold A/B, `--aggregate` measures at true batch size |
| `context_keep_eval.py` | `keep_TRUE` (retina visible after context *and* target masking) |
| `compare_mirage_vs_oracle.py` | parity check against the oracle arm |

> Per-image mask statistics do **not** extrapolate to training, because the
> collator truncates every target to the shortest in the batch. Always measure
> at the real batch size.
