# Contribution/ethics consistency fix

## Verification

The contribution list originally said:

> “The subgroup ordering survives every policy tried, and the headline result
> reproduces to $10^{-5}$ from public weights on different hardware and
> precision.”

The ethics appendix says:

> “The subgroup analysis in Section~\ref{sec:subgroup} is a caution, not a
> contribution. Every masking policy we tested leaves the same groups
> worst-served, and the best-performing policy narrows none of them reliably,
> even though every subgroup point estimate rises at the matched epoch.”

**Verdict: real contradiction.** The empirical statement in the original
bullet is narrow and true: it claims stable subgroup ordering, not a fairness
improvement. However, placing that statement in the explicitly titled
contribution list classifies the subgroup analysis as a contribution, while the
appendix explicitly classifies the same analysis as a caution rather than a
contribution. This is not a difference of empirical scope.

## Contribution change

The bullet now says:

> “The subgroup audit is a caution: every policy leaves the same groups
> worst-served. Public-weight reproduction matches the headline result to
> $10^{-5}$ across hardware and precision.”

This preserves the audit and its negative finding, explicitly presents it as a
caution, makes no fairness-achievement claim, and retains the independently
supported reproduction contribution. The appendix and all of its limitations,
including the missing paired per-cell intervals, remain unchanged.

## Abstract change

The abstract did delay the paper-specific question: three generic setup
sentences preceded the study statement. The opening was tightened without
changing any result, caveat, or number.

### Before

> Masked predictive pretraining asks a model to reconstruct the representation
> of hidden image regions. Which regions are hidden is a free design choice,
> and in medical imaging there is an intuitive answer: hide the tissue that
> carries the diagnosis. We test that intuition from one shared ancestor
> checkpoint under a matched schedule and probe protocol. Six masking policies
> are continued from a single shared I-JEPA checkpoint on 3D retinal OCT,
> differing in how predictor targets are placed, and evaluated with a frozen
> linear-probe protocol for glaucoma classification (FairVision,
> $N{=}\Ntest$). Each policy was continued once, so what follows describes these
> runs rather than an expected ranking over retrainings.

### After

> In 3D retinal OCT, we test whether masked predictive pretraining benefits from
> directing predictor targets toward the tissue that carries the diagnosis.
> From one shared I-JEPA checkpoint, we continue six masking policies under a
> matched schedule, varying target placement, and evaluate each encoder with
> the same frozen linear-probe protocol for glaucoma classification
> (FairVision, $N{=}\Ntest$). Because each policy was continued once, the results
> describe these runs rather than an expected ranking over retrainings.

## Validation

- `p13_build_zip.py`: 6/6 checks passed; main content is exactly 9 pages;
  `ALL_PASS = True`.
- `check_manuscript.py`: `RESULT: PASS`; 56 labels, 56 references, and 0
  dangling references.
- `p15_verify_numbers.py`: `RESULT: PASS`.
