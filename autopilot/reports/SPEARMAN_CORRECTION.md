# Correction: the geometry rank correlations were wrong

Found 2026-08-26 while fixing R4 finding N2. This is a substantive correction to a
load-bearing claim, not a wording change.

## What the paper said

Section 5.2, in the paragraph headed "anatomical targeting does not order the
results":

> Over the four rectangle arms, the Spearman correlation between
> fraction-of-anatomy-hidden and AUC is **exactly 0.00**: knowing how much anatomy
> a policy hides tells you nothing about its downstream AUC. Mask *purity* --- the
> share of masked patches lying on tissue --- **trends negatively (-0.40)**.
> [...] neither shows the positive relationship the anatomy-guidance hypothesis
> predicts.

## What the data says [MEASURED]

Computed with `scipy.stats.spearmanr` on Table 2's own printed values:

| arm set | anatomy hidden vs AUC | purity vs AUC |
|---|---|---|
| **4 rectangle arms** (the paper's own stated set) | **+0.80** (p=0.20) | **+0.40** (p=0.60) |
| all 5 arms | +0.50 (p=0.39) | +0.20 (p=0.75) |

Two independent errors in one paragraph:

1. **anatomy hidden: stated 0.00, actual +0.80.**
2. **purity: stated -0.40, actual +0.40 --- the sign was flipped.**

And therefore the sentence "neither shows the positive relationship the
anatomy-guidance hypothesis predicts" was exactly backwards. Both coefficients are
positive.

## Why it was not caught

The number was **hand-typed in the .tex**, not emitted as a macro. Every gate in
this project checks generated macros against artifacts; a literal `$0.00$` in the
prose is invisible to all three. The same is true of Table 2's geometry column,
which is also hardcoded.

## Cross-check against the stored artifact

`D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json` stores
`spearman.pct_anat_hid = 0.4`. That file's arm list is random, oracle, envelope,
cover_f021, blob, and **cover_f021 has `auc: null`**, so its correlation runs over
random/oracle/envelope/blob. Recomputing on exactly that set reproduces **+0.40**,
which confirms both the artifact and the method used here. The artifact was never
wrong; the paper quoted a number that matches no arm set at all.

## What was changed

The rank correlations cannot support the claim, so the claim now rests on the
direct comparison, which is unaffected and stronger:

- the anatomy arm places **97.3 percent** of masked cells on tissue and does not
  separate from the null;
- \ArmBest places **41.1 percent** on tissue and reaches the highest epoch-100 AUC.

Corrected at all four propagation sites:

| site | line | was | now |
|---|---|---|---|
| Abstract | 69 | "fraction of anatomy hidden is uncorrelated with downstream AUC" | "the most anatomically precise policy is not the best performer" |
| Contribution bullet | 134 | "fraction-of-anatomy-hidden does not order downstream AUC" | same reframing |
| Section 5.2 | 493-502 | 0.00 and -0.40, "neither shows the positive relationship" | +0.80 / +0.40 with p-values, no inference drawn at n=4-5 |
| Conclusion | 698 | "Downstream AUC is uncorrelated with how much anatomy a policy hides" | same reframing |

## Standing lesson

A hardcoded number in prose is outside every gate this project has. The gates
verify macros; they cannot verify what was never a macro. Any numeric claim that
is not a macro should be treated as unverified until recomputed by hand.
