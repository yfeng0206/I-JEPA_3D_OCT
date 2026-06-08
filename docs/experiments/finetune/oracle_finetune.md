# Fine-tune: Oracle ep100 vs Random ep100 — three probes

Oracle ep100 encoder fine-tuned (encoder unfrozen, LLRD γ=0.5, lr 2e-4, 64 slices, warmup 10, 50 ep / patience 15) with each probe, vs the matched random ep100 fine-tunes (`ablation_analysis.md`). Identical configs per probe — only the encoder checkpoint differs.

## Results

| Probe | Random FT | Oracle FT | Δ (oracle − random) | 95% CI (paired bootstrap) | p (2-sided) |
|---|---|---|---|---|---|
| **MeanPool** | 0.8868 | **0.8947** | +0.0079 | [+0.0030, +0.0130] | 0.0010 *** |
| **CrossAttn** | 0.8872 | 0.8937 | +0.0065 | [+0.0019, +0.0115] | 0.0090 ** |
| d=1 | 0.8878 | 0.8901 | +0.0023 | [−0.0018, +0.0065] | 0.257 ns |

Paired stratified bootstrap (B=2000, seed 42, shared resample indices on the 3000-volume Test split; `scripts/bootstrap_finetune.py`). Random FT predictions reproduced 0.8878/0.8872/0.8868 exactly.

**Oracle beats random under fine-tuning, significantly, for MeanPool (+0.0079) and CrossAttn (+0.0065).** d=1 is positive but not significant — consistent with the d=1 attentive probe overfitting (`ablation_analysis.md`) and masking the encoder difference. Best overall number: **oracle FT MeanPool 0.8947**.

## Caveat — the FT protocol overfits

Every FT run (oracle and random) peaks val AUC at **ep3-4 of 50** — during the 10-epoch warmup, before the encoder meaningfully adapts — then overfits (val loss climbs sharply). So these are "lightly fine-tuned" numbers; the selected checkpoint is barely past frozen. The comparison is fair (matched configs, both overfit identically), but the protocol is not exercising real fine-tuning. A proper re-tune (lower encoder/head LR, stronger regularization or partial freezing, multi-seed) is backlogged. The frozen result is the cleaner headline.

## Combined story (frozen + fine-tune)

Oracle beats random at every measured point:
- **Frozen MeanPool**: +0.010 at ep50/75/100, p<0.0005 (`../frozen/oracle_meanpool_sweep.md`).
- **Fine-tuned**: +0.0079 MeanPool / +0.0065 CrossAttn (significant), +0.0023 d=1 (ns).

Consistent across regimes and probes. The anatomy-guided oracle masking produces a better encoder for glaucoma classification.
