# Class-conditioned structural loss

Replaces full Gram-MSE with an objective that transfers only what I-JEPA is
good at and explicitly protects what MIRAGE is good at.

Scripts: `adapter_structural_loss.py`, `adapter_data_diversity.py`,
`fairvision_before_after.py`.

## Motivation

`class_relations.md` established that I-JEPA cannot discriminate inner retina
from choroid (AUC 0.6945, below even an untrained encoder at 0.8288, versus
MIRAGE encoder at 0.9773), and that driving `L_rel` down drags MIRAGE's
inner-choroid similarity from 0.304 toward I-JEPA's 0.719.

## The objective

Each 16×16 cell receives a coarse pseudo-class from **frozen MIRAGE itself** —
no external labels — as I inner, C choroid, B background. Pairs are binned, and
I-JEPA teaches only the safe set

```
S = { I-I, C-C, I-B, C-B }
```

The **I-C block is excluded** — that is the relation I-JEPA gets wrong. B-B is
excluded as carrying nothing worth learning.

Absolute cosine scales differ between independently trained representations
(MIRAGE 0.667/0.304, I-JEPA 0.800/0.718), so matching raw values imports
I-JEPA's *scale* rather than its *structure*. Both sides are z-scored over the
safe set per image.

A one-sided barrier protects the separation MIRAGE already has:

```
delta  = (mu_II + mu_CC)/2 - mu_IC
L_sep  = relu(delta_frozen - delta_adapted)^2
L      = L_struct + lambda_sep * L_sep
```

No penalty while adaptation separates the tissues at least as well as frozen
MIRAGE. Pseudo-labels come from the frozen forward so the class assignment
cannot drift during training.

## Training data (prerequisite finding)

Every prior adapter experiment used a cache containing **slice 100 only** — one
fixed depth out of 200. A stratified cache spanning 100 depths exists and is
strictly better:

| train cache | images | L_rel own | L_rel other depth | GOALS Dice Δ |
|---|---|---|---|---|
| middle (1 depth) | 4800 | 24.50% | 21.18% | −0.00035 |
| stratified (100 depths) | 4800 | 24.28% | **21.84%** | **+0.00010** |

Cross-depth generalisation gap: 2.4pts stratified vs 3.3pts middle-only.

Saturation is real and is **not** a diversity artifact — 1,200 → 2,400 → 4,800
gives 20.1% → 23.2% → 24.3%, so the final doubling buys +1.0pt. **One pass over
4,800 stratified slices (300 steps) is the right budget.** All results below use
it.

## Result 1: the barrier works exactly as designed

Late teacher (ep100), encoder tap. Frozen MIRAGE: Dice 0.9457, separation
Δ=+0.3637, AUC 0.9773.

| α | loss | separation Δ | separation AUC |
|---|---|---|---|
| 0.05 | Gram-MSE | +0.3376 | 0.9782 |
| 0.05 | structural λ=100 | +0.3584 | 0.9813 |
| 0.10 | Gram-MSE | +0.3041 | 0.9780 |
| 0.10 | structural λ=100 | **+0.3544** | **0.9854** |
| 0.25 | Gram-MSE | +0.2105 | 0.9641 |
| 0.25 | structural λ=100 | **+0.3490** | **0.9845** |
| 0.50 | Gram-MSE | +0.1434 | 0.8989 |
| 0.50 | structural λ=100 | **+0.3785** | **0.9748** |

At α=0.50 Gram-MSE destroys **61%** of the separation; the structural loss
preserves **104%** of it. The λ ablation confirms the barrier is the active
ingredient — λ=0 (safe blocks + z-score, no barrier) gives Δ=+0.1532, barely
better than Gram-MSE.

## Result 2: it does **not** remove the segmentation damage

| α | Gram-MSE Dice Δ | structural λ=100 Dice Δ |
|---|---|---|
| 0.05 | +0.00010 | +0.00028 |
| 0.10 | −0.00072 | −0.00062 |
| 0.25 | −0.00324 | **−0.00970** |
| 0.50 | −0.00609 | **−0.02042** |

At low α the two are equivalent and both neutral. At high α the structural loss
is **worse**.

This refutes the premise that separation collapse *causes* the Dice damage. At
α=0.50 the structural adapter holds separation **above frozen** (0.3785 vs
0.3637) and still loses 0.0204 Dice. Separation is therefore not the binding
constraint.

**Class geometry and segmentation damage are independent axes.** The barrier
controls the first; α — the magnitude of representational drift — controls the
second.

Why structural damages more at high α: z-scoring removes the absolute-scale
constraint, freeing the adapter to move the representation further in order to
match the *pattern*. Guide Jaccard confirms greater movement: 0.9087 vs 0.9470
at α=0.50.

### The honest positive finding

At **matched Dice cost**, the structural loss preserves more class structure:

| | Dice Δ | separation Δ | % of frozen separation |
|---|---|---|---|
| Gram-MSE α=0.10 | −0.00072 | +0.3041 | 84% |
| **structural λ=100 α=0.10** | **−0.00062** | **+0.3544** | **97%** |

Same (neutral) Dice, **17% more separation preserved**. That is the setting to
use.

## Result 3: teacher maturity dominates

Both losses, α=0.50:

| teacher | loss | Dice Δ | separation Δ |
|---|---|---|---|
| early ep27 | Gram-MSE | −0.02294 | +0.1284 |
| early ep27 | structural λ=100 | −0.02970 | +0.3690 |
| late ep100 | Gram-MSE | **−0.00609** | +0.1434 |
| late ep100 | structural λ=100 | −0.02042 | +0.3785 |

An early teacher is **3.8× more damaging** than a late one under Gram-MSE
(−0.0229 vs −0.0061). This bears directly on refresh cadence: adapting against
an immature I-JEPA costs far more segmentation quality than adapting against a
converged one.

Note ep25 does not exist on disk; `resume-ep27` is the earliest checkpoint
(T_warm=25, so guidance had barely begun and masking was effectively random).

## Result 4: what actually changes, visually

`fairvision_before_after.png`, `fairvision_guide_change.png`, four stratified
FairVision slices (depths 0, 66, 132, 199) at α=0.50:

| adapter | pixels changed | guide Jaccard |
|---|---|---|
| late ep100, Gram-MSE | 2.54% | 0.898 |
| late ep100, structural λ=100 | 4.61% | 0.809 |
| early ep27, structural λ=100 | 5.39% | 0.776 |

The change is **not diffuse noise**. It concentrates almost entirely at the
**deep (outer) choroid boundary**, and in the guide figure nearly every changed
cell is *gained* (yellow) rather than lost — the adapter systematically extends
the retina downward.

GOALS ground truth says this extension is wrong: Dice falls. MIRAGE's original
outer-choroid boundary was the more accurate one. The outer choroid boundary is
also the least certain boundary in OCT, which is consistent with it being where
a weak teacher exerts most influence.

## Recommendation

1. Use the **stratified** cache, 4,800 images, one pass.
2. Use **structural λ=100 at α=0.10** — Dice-neutral, preserves 97% of the
   tissue separation versus 84% for Gram-MSE.
3. Do **not** refresh against an immature teacher; early-teacher adaptation is
   3.8× more damaging.
4. Do not raise α above ~0.10 under either loss.

## Caveat

None of this establishes that the adapter improves the downstream task. It
establishes how to adapt without damaging segmentation. Whether adaptation
helps glaucoma AUC at all remains untested and requires the frozen-guide
pretraining ablation.
