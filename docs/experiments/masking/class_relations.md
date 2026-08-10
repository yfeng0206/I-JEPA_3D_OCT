# Do MIRAGE and I-JEPA agree about tissue-class relationships?

`L_rel` rests on one assumption: that MIRAGE and I-JEPA should agree about
which patches resemble which other patches. This tests that assumption
directly, decomposed **by tissue class** using GOALS human labels.

Run: `scripts/class_relation_probe.py`, figures from
`scripts/class_relation_figures.py`.

## Method

30 held-out GOALS images. Each image is cut into a 16×16 grid; every cell
inherits a ground-truth class by majority vote and is kept only if ≥70% pure
(294 cells kept: 120 inner retina, 174 choroid). For each model, cosine
similarity is computed between all cell pairs **within an image** — matching
`L_rel`, which compares per-image Gram matrices — and averaged per class-pair.

An **untrained I-JEPA** is included as a control: any structure it shows comes
from raw pixel statistics, not from learning.

## Result: I-JEPA does not separate the two tissues

Mean cosine similarity by class pair:

| model | bg-bg | inner-inner | chor-chor | **inner-chor** | contrast |
|---|---|---|---|---|---|
| MIRAGE H0 | 0.966 | 0.942 | 0.935 | **−0.523** | **+1.461** |
| MIRAGE enc | 0.461 | 0.664 | 0.669 | **0.304** | **+0.362** |
| JEPA ep100 (envelope) | 0.346 | 0.837 | 0.785 | **0.719** | **+0.092** |
| JEPA ep30 (anatomy) | 0.444 | 0.897 | 0.866 | **0.803** | **+0.079** |
| JEPA untrained (control) | 0.784 | 0.696 | 0.744 | 0.526 | +0.194 |

*contrast = mean(within-class) − (inner-choroid)*

MIRAGE holds the classes apart: at the encoder, cross-class similarity (0.304)
is less than half within-class (≈0.667). At H0 the separation is extreme —
inner and choroid are **anti-correlated** (−0.523).

I-JEPA does not. Its cross-class similarity (0.719–0.803) is nearly as high as
its within-class similarity (0.785–0.897). **In I-JEPA's representation, an
inner-retina patch and a choroid patch look almost the same.**

### I-JEPA is worse than an untrained network at this

| | inner-vs-choroid contrast |
|---|---|
| untrained control | **+0.194** |
| JEPA ep100 | +0.092 |
| JEPA ep30 | +0.079 |

Pretraining on OCT **halved** the model's ability to relationally distinguish
inner retina from choroid. Raw pixel statistics separate the two tissues better
than the trained representation does.

This is not a failure of I-JEPA — it is a consequence of the objective. What
I-JEPA did learn is the tissue/background distinction:

| | tissue-vs-background contrast |
|---|---|
| untrained control | +0.435 |
| JEPA ep100 | **+0.506** |
| JEPA ep30 | **+0.517** |

Predicting masked regions rewards knowing *where the retina is*. Nothing in the
objective rewards knowing *which layer* — so that distinction is free to
collapse, and it does.

### Scale-free confirmation

Block means are **not** comparable across models — each has its own similarity
scale, so a model with globally higher cosine values appears to have stronger
within-class similarity without separating anything. These metrics remove that:

| model | within | between | Cohen's d | discrimination AUC |
|---|---|---|---|---|
| MIRAGE H0 | 0.937 | −0.523 | 20.20 | **1.0000** |
| MIRAGE enc | 0.667 | 0.304 | 2.82 | **0.9773** |
| MIRAGE enc + α=0.05 | 0.694 | 0.356 | 2.90 | 0.9808 |
| MIRAGE enc + α=0.50 | 0.835 | 0.672 | 2.53 | 0.9654 |
| JEPA ep100 (envelope) | 0.800 | 0.718 | **0.65** | **0.6945** |
| JEPA ep30 (anatomy) | 0.875 | 0.803 | 0.84 | 0.7714 |
| JEPA untrained (control) | 0.730 | 0.526 | **1.38** | **0.8288** |

*discrimination AUC = P(a random same-tissue pair is more similar than a random
different-tissue pair); 0.5 is chance.*

MIRAGE at H0 **never** confuses the two tissues (AUC 1.0000). I-JEPA is barely
above chance (0.6945). The effect size gap is 4.3× (2.82 vs 0.65).

Note that I-JEPA's raw within-class similarity (0.800) is *higher* than
MIRAGE's (0.667) — which is exactly why the absolute numbers mislead. What
matters is that 0.800 vs 0.718 is almost no gap, whereas 0.667 vs 0.304 is a
large one.

The untrained control outperforms both trained models on every scale-free
measure, confirming the collapse is caused by pretraining and is not an
artifact of scale.

## The damage mechanism, observed directly

> **Superseded — read this first.** The paragraph below was written before the
> class-conditioned structural loss existed, and its causal claim does not
> survive that experiment. `structural_loss.md` shows a configuration that
> holds separation **above** frozen MIRAGE (Δ=+0.3785 vs +0.3637) and *still*
> loses 0.0204 Dice. Separation collapse is therefore **not** the cause of the
> segmentation damage. What Gram-MSE does is move separation and Dice in the
> same harmful direction; class geometry and representational drift are
> independent axes, and α controls the second. Treat what follows as a
> description of *what Gram-MSE does to class geometry*, not as a mechanism for
> the Dice loss.

Applying the encoder adapter and re-measuring shows MIRAGE being pulled toward
I-JEPA's class geometry:

| | inner-chor similarity | contrast | GOALS Dice |
|---|---|---|---|
| MIRAGE enc (frozen) | 0.304 | +0.362 | 0.9457 |
| + adapter α=0.05 | 0.356 | +0.336 | 0.9454 |
| + adapter α=0.50 | **0.672** | **+0.163** | 0.9328 |
| *(I-JEPA ep100 target)* | *0.719* | *+0.092* | — |

At α=0.50 the inner/choroid similarity has moved **88.8%** of the way from
MIRAGE's value to I-JEPA's `((0.672−0.304)/(0.719−0.304))`, and the class
contrast has collapsed by **55.0%**.

Corroborating: agreement with MIRAGE's block structure is *higher* for the
**untrained** I-JEPA (r=0.738) than for either trained one (r=0.663, r=0.663).
Training moved I-JEPA away from MIRAGE's geometry.

## Implication

`L_rel` transfers I-JEPA's class geometry into MIRAGE. Since I-JEPA's geometry
merges inner retina and choroid, the transfer necessarily degrades exactly the
distinction the segmentation head exists to make.

This explains the *class-geometry* observations:

- monotone collapse of the inner/choroid contrast as α rises
- the adapter's inability to add anything on this axis — the teacher does not
  represent it

It does **not**, on its own, explain the Dice damage: `structural_loss.md`
demonstrates that separation can be fully protected while Dice still falls.

If a relational signal is wanted, the target must come from a representation
that *does* separate the tissues. I-JEPA's does not.

## Terminology caveat

Class 0 is **"Elsewhere"** in the GOALS/MergedV3 label set, not "background".
It contains true background *and* retinal tissue that is neither inner retina
nor choroid — outer retina in particular lies between the two labelled bands.
Statements about "tissue vs background" in this document should be read as
"inner-retina-plus-choroid vs everything else". The inner-vs-choroid results
are unaffected, because both of those classes are labelled explicitly.

## Figures

- `class_blocks.png` — 3×3 similarity matrix per model; MIRAGE morphing into
  I-JEPA as α rises is visible directly
- `class_contrast.png` — the two contrasts as bar charts
- `segmentation_before_after.png` — GOALS predictions, the four most-changed
  images: α=0.05 is visually indistinguishable from frozen, α=0.50 visibly
  thickens and blurs the inner/choroid boundary

## Caveat

The oracle ep100 checkpoint requested for this comparison is no longer on disk
(searched all of `D:\`); `envelope ep100` is the surviving 100-epoch model and
was used instead. The untrained control carries the argument regardless, since
it bounds what pixel statistics alone provide.
