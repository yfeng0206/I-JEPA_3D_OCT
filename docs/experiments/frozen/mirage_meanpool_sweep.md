# Frozen MeanPool Sweep — MIRAGE-guided (ep50/75/100)

MeanPool + Linear frozen probe across the MIRAGE-guided checkpoints (warm-start
from random ep25, [`../pretraining/mirage_100ep.md`](../pretraining/mirage_100ep.md)).
Byte-identical probe config to the oracle sweep
([`oracle_meanpool_sweep.md`](oracle_meanpool_sweep.md)) and to the random ep100
MeanPool run ([`mean_pool.md`](mean_pool.md)), so the three arms are
apples-to-apples. The probe is the zero-parameter ablation floor (mean over
slices → LinearHead), so it reads encoder quality almost directly.

**Completed** 2026-08-05, locally on one RTX 3090
(`configs/frozen_meanpool_mirage_ep{50,75,100}.yaml`).

**Headline: the masking ladder's working assumption fails here.** MIRAGE masks
the retina *more purely* than the hand-crafted oracle band and still produces a
*worse* encoder at ep100 (0.8807 vs 0.8855, paired bootstrap CI excludes zero).
Better segmentation did not buy better representations.

## Config (matches `oracle_meanpool_sweep.md` / `mean_pool.md` / `d1_sweep.md`)

| Parameter | Value |
|---|---|
| Probe | MeanPool + LinearHead (0 probe params, 2.3K head) |
| Encoder | Frozen ViT-B/16 (MIRAGE-guided) |
| Num slices | 100 |
| Batch size | 256 |
| Epochs / patience | 50 / 15 |
| Warmup | 5 |
| LR (head) | 4e-4 |
| Weight decay | 0.05 |
| Dropout | 0.2 |
| Seed | 42 |

Only `model.encoder_checkpoint` differs from the oracle sweep's configs.

## MIRAGE results

| Checkpoint | Best epoch | Train AUC | Val AUC | Test AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| ep50 | 45 | 0.8758 | 0.8594 | 0.8761 | 0.757 | 0.846 |
| ep75 | 48 | 0.8848 | 0.8584 | 0.8803 | 0.764 | 0.844 |
| **ep100** | 46 | 0.8869 | 0.8599 | **0.8807** | 0.767 | 0.844 |

Monotonic with pretraining length, but the ep75 → ep100 step is only +0.0004 —
the arm is flat after ep75, where the oracle was still gaining (+0.0019).
Train > Val ~0.027 and Test > Val ~0.021, both the same mild pattern every other
run on this FairVision val(1000)/test(3000) split shows.

## Three arms — frozen MeanPool Test AUC

Recomputed from the committed prediction files with `sklearn.roc_auc_score`
(n = 3000, positives = 1466, identical label vector in all nine files):

| Epoch | random | oracle | **MIRAGE** | MIRAGE − oracle | MIRAGE − random |
|---|---|---|---|---|---|
| ep50 | 0.8641 | 0.8740 | **0.8761** | +0.0020 | +0.0120 |
| ep75 | 0.8723 | 0.8836 | **0.8803** | −0.0033 | +0.0080 |
| ep100 | 0.8746 | 0.8855 | **0.8807** | −0.0047 | +0.0062 |

Predictions: `results/downstream/meanpool_sweep_{random,oracle,mirage}/ep{50,75,100}_test_predictions.npz`
(keys `labels`, `probs`).

### Paired bootstrap

B = 2000 stratified resamples, seed 0, the *same* resample indices applied to
both arms in each replicate, so the CI is on the **difference**
(`scripts/bootstrap_paired_arms.py`, method per
[`ablation_analysis.md`](ablation_analysis.md)):

```bash
python scripts/bootstrap_paired_arms.py --seed 0 \
    --a results/downstream/meanpool_sweep_mirage/ep100_test_predictions.npz \
    --b results/downstream/meanpool_sweep_oracle/ep100_test_predictions.npz \
    --b results/downstream/meanpool_sweep_random/ep100_test_predictions.npz \
    --name-a MIRAGE --name-b oracle --name-b random
```

**MIRAGE vs oracle**

| Epoch | Δ | 95% CI | p (2-sided) | verdict |
|---|---|---|---|---|
| ep50 | +0.0020 | [−0.0028, +0.0068] | 0.396 | ns |
| ep75 | −0.0033 | [−0.0077, +0.0012] | 0.151 | ns |
| ep100 | **−0.0047** | **[−0.0091, −0.0002]** | 0.041 | **oracle wins, barely** |

**MIRAGE vs random**

| Epoch | Δ | 95% CI | p (2-sided) |
|---|---|---|---|
| ep50 | +0.0120 | [+0.0067, +0.0173] | <0.0005 *** |
| ep75 | +0.0080 | [+0.0029, +0.0130] | 0.001 ** |
| ep100 | +0.0062 | [+0.0009, +0.0112] | 0.022 * |

**Read carefully.** MIRAGE beats random at every epoch, so anatomy-guided
masking is doing *something* real — the rung is not a null result. But the gap
over random *shrinks* with pretraining length (+0.0120 → +0.0062) while the
oracle's stayed flat at ~+0.010, and by ep100 MIRAGE has fallen significantly
*behind* the oracle. The ep100 CI upper bound is −0.0002: this clears zero by a
hair on 2000 resamples and should not be treated as a precisely-measured
effect size. What it does rule out is the claim the rung was built to test —
that a real segmentation model would match or beat the hand-crafted band.

## The paradox

| quantity | ORACLE | MIRAGE thr 0.25 | who wins |
|---|---|---|---|
| target-block purity (1,000 vols, [`../mirage_guided_masking.md`](../mirage_guided_masking.md)) | 0.5602 | **0.6320** | MIRAGE by 13% |
| targets on retina (2,339 slices, `scripts/oracle_failure_cases.py`) | 0.458 | **0.506** | MIRAGE |
| slices where MIRAGE puts more targets on retina | — | 63.1% | MIRAGE |
| oracle band purity against pixel tissue truth (same 2,339 slices) | 0.653 | not measured | (oracle is <0.50 pure on 13.8% of slices) |
| **frozen MeanPool Test AUC, ep100** | **0.8855** | 0.8807 | **oracle** |

Every masking-quality metric the program built says MIRAGE is the better prior.
The only metric that matters — downstream AUC — says the opposite. This is
`lessons_learned.md` #1 in its sharpest form: *the proxy is not the signal.*

Two candidate explanations were on the table. Both are measured below.

## Hypothesis A — MIRAGE trades inner retina for choroid

**Claim to test:** MIRAGE marks *less* of the inner retina (RNFL + GCIPL, the
layers glaucoma thins) and *more* of the choroid than the oracle band does, so
its extra purity is spent on tissue that carries less glaucoma signal.

Nothing in the pipeline could answer this, because both the policy sweep and
`oracle_failure_cases.py` score against a single binary "is this retina" truth
that merges all three MIRAGE classes.
[`../../../scripts/mirage_vs_oracle_region_split.py`](../../../scripts/mirage_vs_oracle_region_split.py)
re-scores the *same* placements against the **class** map instead
(`src/guides/mirage_envelope.py`: inner = `CLASS_RNFL` 1 + `CLASS_GCIPL` 2,
choroid = `CLASS_CHOROID` 3), reading the cached hard label maps directly so no
MIRAGE inference is needed.

Both arms are built exactly as the training run builds them — the oracle from
`CurriculumMaskGenerator._anatomical_prior_weight_grid_for_image` under
`configs/patch_oracle_anatomical.yaml`, MIRAGE from the
`GuidedOCTSliceDataset` placement channel at the shipped policy (occupancy
≥ 0.25, dilation 0) — and both are sampled with the **same RNG seed per slice**,
so the four block sizes are identical across arms and only placement moves.
Values are mean per-cell pixel fractions, so they are threshold-free.

```bash
python scripts/mirage_vs_oracle_region_split.py --volumes 400
```

**2,374 slices from 400 volumes.** The oracle band area comes out at 0.273 of
the frame, matching `oracle_failure_cases.py` exactly — an independent check
that the grid is constructed the same way.

| admissible region (cells a block may be drawn from) | ORACLE | MIRAGE |
|---|---|---|
| area (fraction of frame) | 0.273 | 0.290 |
| inner retina (RNFL+GCIPL) | 0.178 | **0.253** |
| choroid | 0.209 | **0.321** |
| inner share of labelled tissue | 0.461 | 0.440 |

| placed target blocks (union of the four) | ORACLE | MIRAGE |
|---|---|---|
| area (fraction of frame) | 0.394 | 0.425 |
| inner retina (RNFL+GCIPL) | 0.1164 | 0.1171 |
| choroid | 0.1526 | 0.1757 |
| inner share of labelled tissue | 0.433 | **0.400** |
| slices where MIRAGE places **less** inner retina | — | 47.0% |
| slices where MIRAGE places **more** choroid | — | 67.1% |

### Verdict: half confirmed, half refuted

**Refuted — MIRAGE does not mark less inner retina.** Its target blocks cover
the *same* absolute amount of inner retina as the oracle's (0.1171 vs 0.1164, a
ratio of 1.007), and its admissible region covers **42% more** inner retina
(0.253 vs 0.178). Only 47.0% of slices — a coin flip — have MIRAGE placing less
inner retina than the oracle, so there is no systematic loss of it.

**Confirmed — MIRAGE does shift onto the choroid.** Choroid coverage in the
placed blocks rises 0.1526 → 0.1757 (+15.2%), on 67.1% of slices, and the inner
share of labelled tissue in the mask falls 0.433 → 0.400.

The correct statement is therefore neither the original hypothesis nor its
negation:

> MIRAGE does not take inner retina away — it **adds choroid on top**.
> On-tissue masking rises by 0.0239 of the frame going from oracle to MIRAGE,
> and **96.8% of that increase is choroid** (+0.0231) against **3.2% inner
> retina** (+0.0008).

That is a precise, falsifiable account of where the purity gain went: the
0.560 → 0.632 purity improvement is almost entirely extra choroid masking, and
the inner retina — the tissue glaucoma actually damages — is masked no more
than before. On this evidence MIRAGE's purity advantage is real but
**diagnostically inert**, which is consistent with it buying no AUC. It is not,
however, sufficient to explain MIRAGE ending up *behind* the oracle; extra
choroid masking should be neutral, not harmful. For that, see B.

Reproducible with a disjoint slice sample (`--volumes 150 --slice-stride 7
--seed 11`, 2,229 slices): inner ratio 1.001, choroid ratio 1.168, inner share
of tissue 0.454 → 0.416, all within 0.005 of the numbers above.

## Hypothesis B — masked area and placement entropy, not purity

The competing account (labelled "B2" in prior review) is that the driver is
*geometry*: a smaller, more predictable target region makes the pretext task
easier, and an easier pretext task yields a weaker encoder. The same script
measures this, counting how many top-left positions each arm's sampler could
have chosen for the four block sizes it actually drew (admissible = block fill
≥ `mirage_min_block_fill` 0.40):

| placement freedom | ORACLE | MIRAGE |
|---|---|---|
| admissible windows per block | 61.8 | 48.8 (−21%) |
| bits (log2 admissible windows) | 5.93 | 5.34 |
| oracle sampler's true Shannon entropy | 6.41 | n/a |

The oracle's sampler is multinomial over block-summed band weights across the
*whole* grid with a `weight_eps` floor, so it can place a block anywhere with
small probability; its true placement entropy is 6.41 bits. MIRAGE's sampler is
restricted to admissible windows, at most 5.34 bits. That is **1.07 bits, i.e.
2.1× fewer effective placements per block**. Four blocks per image compounds it.
The MIRAGE pretraining run's consistently **lower train loss at every matched
epoch** (−0.007 to −0.011 from ep50 onward,
[`../pretraining/mirage_100ep.md`](../pretraining/mirage_100ep.md)) at
essentially identical val loss is exactly what an easier pretext task looks
like.

### The masked-area control does not hold in the config that trained

The policy sweep chose threshold 0.25 over the higher-purity 0.50/0.75
specifically to keep masked area within 0.5% of the oracle (unique targets
101.7 vs 101.9). **That control was measured with `mirage_spread` disabled, and
the run trained with it enabled.**

- `scripts/mirage_method_sweep.py` declares `spread: bool = False` on its
  `Method` dataclass and passes `"mirage_spread": method.spread`, so every
  MIRAGE row in the sweep table was measured with spread **off**.
- `configs/patch_mirage_envelope.yaml` sets `mirage_spread: true` (and
  `CurriculumMaskGenerator` defaults it to `True` anyway), so the run trained
  with spread **on**. Spread assigns the four blocks to disjoint lateral
  segments, which pushes them apart and inflates their union.

Measured with the same script, same slices, same seeds:

| unique target patches (union of 4 blocks) | ORACLE | MIRAGE |
|---|---|---|
| `--spread off` (what the sweep measured) | 100.6 | 100.9 (+0.4%) |
| `--spread config` (what trained) | 100.9 | **108.8 (+7.8%)** |

The MIRAGE arm's own training log agrees on direction and is higher still:
13,651 `[MIRAGE]` collator lines in
`D:\jepa_phase0\runs\patch_mirage_envelope\train.log` average
`unique_targets=120.2`, `context=111.4`, `on_region=0.462`. (Per-image offline
statistics do not extrapolate exactly to the batched training protocol — see
the warning at the end of [`../mirage_guided_masking.md`](../mirage_guided_masking.md)
— so the *paired* offline comparison above is the load-bearing number and the
log is corroboration, not a second estimate of the same quantity.)

So the two arms were **not** area-matched during training: MIRAGE masked ~8%
more of the image while having ~2× less freedom about where to put it. Both
directions make the comparison something other than a pure
location-only intervention, which is what the experimental contract promised.

### Is equal area enough anyway?

No — and this is worth stating even if a future run fixes `mirage_spread`.
Matching *area* does not match *shape*. The oracle band is a wide ribbon
spanning 60% of the width at a fixed height, so a block can slide a long way
laterally and stay admissible. The MIRAGE envelope is thinner, tracks the
tissue, and its admissible set is fragmented by the fill threshold — hence 48.8
admissible windows against 61.8 at essentially the same region area (0.290 vs
0.273). Area equality at the patch-grid level is a necessary but not sufficient
control; **admissible-window count / placement entropy is the quantity that
should be matched**, and no run so far has matched it.

## What this does and does not establish

**Establishes.** On this dataset and this probe, target-block purity does not
predict downstream AUC. A prior can be measurably more accurate about anatomy
and still produce a worse encoder. Any future rung on the masking ladder must
be judged on AUC, and any purity-style metric used to select a policy must
first be shown to correlate with AUC.

**Does not establish.**

1. **Not a controlled test of purity.** Because `mirage_spread` differed between
   the sweep and the run, the MIRAGE arm differs from the oracle in masked area
   (+7.8%) and placement entropy (−1.07 bits) as well as in purity. Hypothesis A
   and Hypothesis B are *not* separated by this experiment. The clean test is a
   re-run with `mirage_spread: false`, which the measurement above shows
   restores area parity to +0.4%.
2. **Single seed per arm.** One pretraining run each. The bootstrap CIs measure
   test-sample uncertainty with the encoders held fixed; they say nothing about
   pretraining-seed variance. A −0.0047 gap whose CI stops at −0.0002 is inside
   the range a second seed could plausibly move.
3. **Frozen MeanPool only.** No fine-tune, CrossAttnPool or d=1 numbers exist
   for the MIRAGE arm. The oracle's frozen advantage did survive fine-tuning
   ([`../finetune/oracle_finetune.md`](../finetune/oracle_finetune.md)), but that
   is not evidence about this arm.
4. **Choroid is not shown to be harmful.** Hypothesis A shows where the extra
   masking went, not that putting it there hurts. Inner-retina masking is
   unchanged between arms, so the choroid shift is better read as *wasted* than
   as *damaging*.

## Context: where this sits

| Arm | Frozen MeanPool ep100 | Fine-tune MeanPool ep100 |
|---|---|---|
| random | 0.8746 | 0.8868 |
| oracle | **0.8855** | **0.8947** |
| MIRAGE | 0.8807 | not run |

Fine-tune numbers from `results/downstream/finetune_{oracle,random}/`; the
frozen-probe ceiling on this dataset remains ~0.88 and fine-tuning adds the next
~1%, unchanged by this rung.

## Next

1. Re-run the MIRAGE arm with `mirage_spread: false` to restore area parity, and
   re-measure. Until then purity and geometry are confounded.
2. Add admissible-window count to the parity check in
   `scripts/compare_mirage_vs_oracle.py`, which currently gates on block count,
   unique targets and context tokens but not on placement freedom — the axis
   that actually differed.
3. If a future prior is to be judged before training, judge it on
   inner-retina coverage, not on whole-envelope purity.
