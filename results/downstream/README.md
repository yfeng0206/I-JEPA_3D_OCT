# Downstream evaluation artifacts

Per-volume predictions and metrics for every downstream arm, committed so that
statistical comparisons are reproducible **from a clean clone on any machine**.

Previously these files were untracked and existed only on the machine that ran
the eval, which made cross-machine paired comparisons impossible.

## What is here

| file | contents |
|---|---|
| `*_test_predictions.npz` | `labels` (N,) float32, `probs` (N,) float16 — one row per Test volume |
| `*_val_predictions.npz` | same, for the Validation split |
| `*_results.json` | test AUC, sensitivity, specificity, best epoch, full resolved config |
| `*_config.yaml` | the eval config as run |
| `*_train_log.csv` | per-epoch probe training curve |

Written automatically by `src/eval_downstream.py` (frozen probe: line ~706;
fine-tune: line ~1333). Model weights (`*.pt`) are gitignored and stay local.

### Arms

| directory | pretraining arm | probe |
|---|---|---|
| `meanpool_sweep_random/` | random masking, ep50/75/100 | frozen MeanPool |
| `meanpool_sweep_oracle/` | oracle anatomical masking, ep50/75/100 | frozen MeanPool |
| `finetune_random/` | random masking | fine-tuned |
| `finetune_oracle/` | oracle anatomical masking | fine-tuned |

All arms share the FairVision **Test** split: **N = 3000**, 1466 positive /
1534 negative, `seed: 42`, `shuffle=False`. The `labels` array is byte-identical
across arms — this is precisely what makes a paired test valid.

## Pretrained encoders

The encoders that produced these predictions are on Hugging Face (private repo —
request access):

**<https://huggingface.co/yfeng0206/ijepa-3d-oct-checkpoints>**

| directory here | Hugging Face path |
|---|---|
| `meanpool_sweep_random/`, `finetune_random/` | `random-posfix-100ep/jepa_patch-ep{025,050,075,100}.pth.tar` |
| `meanpool_sweep_oracle/`, `finetune_oracle/` | `oracle-anatomical-100ep/jepa_patch_oracle-ep{050,075,100}.pth.tar` |

`MANIFEST.json` there records sha256, epoch and original run path per file, so a
prediction file here traces to the exact encoder that produced it.

## Running a paired comparison

```bash
python scripts/bootstrap_paired_arms.py \
  --a results/downstream/meanpool_sweep_oracle/ep100_test_predictions.npz \
  --b results/downstream/meanpool_sweep_random/ep100_test_predictions.npz \
  --name-a oracle_ep100 --name-b random_ep100
```

Reference output — use this to verify a new machine reproduces the published
result before trusting any new arm:

```
oracle_ep100 - random_ep100 | 0.8855 | 0.8746 | +0.0109 | [+0.0058, +0.0162] | <0.0005 ***
```

`scripts/bootstrap_frozen_meanpool.py` is the original fixed oracle-vs-random
version; `bootstrap_paired_arms.py` generalises it to any two arms.

## Adding a new arm (e.g. MIRAGE)

Run the **same** probe, changing only `model.encoder_checkpoint`. Everything
below must match the existing arms or the paired test is invalid:

- FairVision Test split, `shuffle=False`
- `training.seed: 42`
- probe type, depth, head, `num_slices`, epochs, lr, weight decay, dropout

Then commit the resulting `*_test_predictions.npz` here. The bootstrap script
asserts label-vector equality and aborts loudly if the splits diverge.

## Why paired, not marginal

A marginal 95% CI on a single model's AUC (e.g. ±0.0122) is dominated by *which
volumes were drawn* — variance shared by every model on the cohort. The paired
bootstrap resamples once per replicate and scores **both** arms on the identical
indices, so that shared term cancels:

```
Var(A - B) = Var(A) + Var(B) - 2*Cov(A, B)
```

Model AUCs on a common cohort are strongly positively correlated, so `Cov` is
large and the paired CI is roughly 2.5x tighter (±0.005). A marginal CI that
happens to contain another model's point estimate is **not** evidence the two
are indistinguishable — it is the wrong test.

## Scope limit

These CIs quantify **test-sample uncertainty with both models held fixed**. They
say nothing about pretraining-seed variance. The oracle arm was additionally
warm-started from random-arm ep25 rather than trained as an independent lineage,
so it carries a resume/RNG confound that no bootstrap can address.
