# Pretraining Experiments

Self-supervised I-JEPA pretraining on 600K OCT slices (FairVision Training split, 6K volumes × 100 slices).

## Terminology

`mirage_envelope` and `mirage_anatomy` are different masking modes.

| Mode | MIRAGE role | Target shape | Interpretation |
|---|---|---|---|
| `mirage_envelope` | Places ordinary rectangular I-JEPA targets on the repaired retinal envelope | Rectangles | Baseline: MIRAGE placement without anatomy-shaped targets |
| `mirage_anatomy` | Supplies a soft tissue guide used to form connected irregular targets | Connected anatomy-shaped blobs | Contribution: target shape changes from rectangles to tissue-shaped blobs |

MIRAGE guidance alone is not the novelty. The `mirage_envelope` baseline already uses MIRAGE for placement. The tested novelty is target **shape**.

## Runs

| Run | Mode | Init | Epochs | Status | Checkpoints now on disk | Downstream AUC |
|---|---|---|---:|---|---|---|
| [Random-init 100ep](random_100ep.md) | Random rectangles | Random | 100 | Completed baseline | Not re-verified in the current disk audit | Historic ep100: 0.8746 |
| [Oracle anatomical 100ep](oracle_100ep.md) | `anatomical_prior` rectangles placed on a hand-crafted retina band | Warm-start random ep25 | 100 | Completed historic comparison | **No** — searched all of `D:` and checkpoints are no longer present | Historic ep100: 0.8855 |
| [MIRAGE envelope 100ep](envelope_100ep.md) | `mirage_envelope`: MIRAGE-placed rectangles | Warm-start random ep25 | 100 | Completed baseline | **Yes** — `D:\jepa_phase0\runs\patch_mirage_envelope\` has ep30,35,40,45,...,100 plus `resume-ep27.pth.tar` and `best` | Historic ep100: 0.8807; matched ep30: 0.8528 ± 0.0018 |
| [MIRAGE anatomy ep30](anatomy_ep30.md) | `mirage_anatomy`: connected anatomy-shaped blobs | Warm-start envelope `resume-ep27.pth.tar` | 30 so far | Halted in ep32; last checkpoint ep30 | **Yes** — `D:\jepa_phase0\runs\patch_mirage_anatomy\` has `ep30.pth.tar` and `best.pth.tar` only | Matched ep30: 0.8582 ± 0.0003 |
| Planned anatomy restart | `mirage_anatomy` with new adapter design | Warm-start random ep25 | Planned 100 | Planned, not run | No | Unknown |

Shared completed-run config unless noted: ViT-B/16, peak LR 0.00025, warmup 5 epochs, EMA 0.996→1.0, effective batch 512, weight_decay 0.04→0.4 cosine, no early stopping.

## Downstream comparisons

Frozen mean-pool probe, 100 slices/volume, true fp32, test n=3000.

| Comparison | AUC | Notes |
|---|---:|---|
| random ep100 | 0.8746 | Historic 100-epoch number |
| `mirage_envelope` ep100 | 0.8807 | Historic 100-epoch number; rectangles placed by MIRAGE |
| oracle ep100 | 0.8855 | Historic 100-epoch number; checkpoints no longer on disk |
| `mirage_envelope` ep30 | 0.8528 ± 0.0018 | 5 probe seeds |
| `mirage_anatomy` ep30 | 0.8582 ± 0.0003 | 5 probe seeds |
| anatomy − envelope at ep30 | +0.0054 | Welch t p=0.00219; Mann-Whitney p=0.0079; Cohen's d 4.20; arms fully separated |
| anatomy − envelope paired bootstrap | +0.0044 | Bootstrap over volumes; 95% CI [+0.0010,+0.0077]; p=0.012 |

The ep30 anatomy-vs-envelope comparison is the only fair head-to-head currently available. The ep100 historic numbers compare different training lengths for the anatomy question because the anatomy arm has no checkpoint beyond ep30.

## Validation-loss caveat

`src/train_patch.py` hard-codes validation to a plain `MaskCollator` with 42-cell rectangles and the comment "ALWAYS uniform — keeps val loss comparable". This makes validation loss comparable across runs, but it destroys train-vs-validation comparability for curriculum runs whose training targets are different. The anatomy run trains on 16-cell anatomy blobs, while validation uses 42-cell rectangles.

Do not interpret the anatomy ep30–32 validation rise as degradation. Anatomy and `random_default` had near-identical train loss, 0.0247 vs 0.0246, while validation differed by 1.5×. The baseline shows the same divergence when its guidance ramp completes. Downstream AUC is the quality signal.

## Downstream precision and cache correction

Earlier downstream evaluations requested `data.use_amp: false`, but a bug in `src/eval_downstream.py` ignored that setting at 5 of 6 `autocast()` call sites. Those evaluations therefore ran mostly in fp16 despite the config. The measured impact was immaterial for the anatomy ep30 probe, 0.8582 fp16 vs 0.8583 fp32, so earlier results are not invalidated. The fix is in and evaluations are now genuinely fp32.

A second latent bug omitted precision from the feature-cache key, so an fp16 cache could be silently reused by an fp32 run. The cache key is now `{split}_s{n}_{amp|fp32}`.

## Planned next run

The next run is planned, not completed: a fresh `mirage_anatomy` run warm-started from the random ep25 checkpoint, using the new adapter design described in [structural_loss.md](../masking/structural_loss.md) and [adapter_placement.md](../masking/adapter_placement.md). The design uses encoder-tap placement with a class-conditioned structural loss.

The adapter and guide cache will refresh at epochs 25/50/75/100 to match checkpoint boundaries. A guide-cache rebuild costs 66 minutes, measured. Caches are sha-tagged by adapter, so a new adapter writes a new cache directory and old results stay reproducible.

## Key takeaways

1. **Peak LR 0.00025 for OCT + effective batch 512.** Correlated gradients from less diverse OCT data make the effective LR higher than nominal; I-JEPA's default 0.0005 for ImageNet is too hot here.
2. **Warmup gate for early-stopping and best-checkpoint save.** Pre-warmup epochs have artificially low validation loss because the EMA target has not diverged from the online encoder.
3. **I-JEPA loss is not a quality signal.** It changes with EMA maturity, target geometry, and the validation collator. Use downstream AUC for model quality.
4. **No early stopping for fixed comparisons.** The 100-epoch arms use fixed epochs and saved checkpoints; the current anatomy arm has no checkpoint past ep30 and should only be compared to matched ep30 baselines.
5. **MIRAGE placement is not the contribution.** `mirage_envelope` already uses MIRAGE to place rectangles. The contribution is replacing rectangles with connected anatomy-shaped `mirage_anatomy` targets.
6. **Masking purity is not a proxy for downstream AUC.** Historic ep100 numbers show `mirage_envelope` at 0.8807 and oracle at 0.8855 despite MIRAGE's more retinal placement. Judge masking priors on AUC, with matched epochs when training length differs.

