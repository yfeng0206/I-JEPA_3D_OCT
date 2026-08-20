# Gaps and required follow-up

There are **5 `\TODO` occurrences** in `main.tex`, representing four scientific
gaps. The first two occurrences refer to the same pending measurement.

## TODO ledger

| Location | Missing evidence | Exact run/command that fills it |
|---|---|---|
| COVER dose-response text; AUC table ep50 cell | Clean COVER floor-0.21 epoch-50 frozen-probe AUC | The existing live chain is already filling this; **do not launch a duplicate**. Its command is `D:\jepa_phase0\.venv\Scripts\python.exe scripts\chain_cover_f021.py`. Completion artifact: `D:\jepa_phase0\runs\frozen_meanpool_cover_f021_ep50\results.json`. |
| Limitations: paired pretraining continuations | Pretraining-seed variance for shape arms | Create three seed-matched envelope/anatomy config pairs from one identical checkpoint, setting `meta.seed` to `0`, `1`, and `2` and unique run directories. Run each with `D:\jepa_phase0\.venv\Scripts\python.exe src\train_patch.py --config <paired-config.yaml>`, then evaluate each checkpoint with `D:\jepa_phase0\.venv\Scripts\python.exe -m src.eval_downstream --config <frozen-probe-config.yaml>`. The continuation, not probe seed, is the replicate. |
| Limitations: guide/budget-matched shape comparison | A causal rectangle-versus-shape comparison | Build two configs with the same soft guide cache, target anatomy fraction, unique hidden budget, ramp granularity, target count, checkpoint, and seed; only target geometry may differ. Run both with `D:\jepa_phase0\.venv\Scripts\python.exe src\train_patch.py --config <config.yaml>` and probe with `D:\jepa_phase0\.venv\Scripts\python.exe -m src.eval_downstream --config <probe-config.yaml>`. Repeat for the three paired seeds above. |
| Limitations: floor-to-AUC relation | Matched epoch-50 AUC across COVER floors | Minimal informative set: floors `0.15`, `0.21`, `0.25`, `0.30`, all from the shared epoch-25 checkpoint with `enc_truncate: prefix`, `amp_target: false`, and otherwise identical configs. For each: `D:\jepa_phase0\.venv\Scripts\python.exe src\train_patch.py --config <cover-floor-config.yaml>`, followed by `D:\jepa_phase0\.venv\Scripts\python.exe -m src.eval_downstream --config <ep50-probe-config.yaml>`. |

## Other missing measurements

1. **External confirmation.** The same FairVision test split has informed
   multiple historical decisions. A locked external OCT glaucoma cohort or an
   untouched FairVision holdout would strengthen all downstream conclusions.
2. **Random/oracle checkpoint provenance — AUCs CLOSED, checkpoints still open.**
   Their epoch-50 mean-pool AUCs are **measured and verified** (2026-08-19):
   `results\downstream\meanpool_sweep_random\ep50_results.json` → `test_auc`
   **0.8640970649809413** (best_epoch 46, val 0.8450730), and
   `results\downstream\meanpool_sweep_oracle\oracle_ep50.json` → `test_auc`
   **0.8740299460522829** (best_epoch 47, val 0.8543915). Both are
   `probe_type: mean_pool`, `head_type: linear`, `num_slices: 100`, `seed: 42`,
   i.e. the same protocol as the envelope/blob arms, so they are directly
   comparable in the purity→AUC table. *(Any doc calling these "unmeasured" is
   stale; note the files live under `C:\Users\Gary\Desktop\jepa\results\`, not
   `D:\jepa_phase0\results\`.)*
   Two things remain genuinely missing:
   - **No per-sample predictions.** Neither JSON stores `test_predictions`, so
     random and oracle **cannot enter the subgroup/fairness analysis**. This is
     why the equity table has 12 arms and not 14.
   - **No local checkpoints.** Both runs record `data_dir:
     /tmp/fairvision_data/data` and `encoder_checkpoint:
     /tmp/ijepa_checkpoints/...`, i.e. they were produced on a different machine
     (June 2026). The encoders are not available locally, so these two arms
     cannot be re-probed or seed-replicated.
   - **Do not confuse with the region-restricted probe.** A separate probe run
     (`D:\jepa_phase0\reports\downstream_region_auc\{random,oracle}_ep50\region_auc.json`)
     reports `all` test AUC **0.8608341** (random) and **0.8682588** (oracle).
     These differ from the sweep numbers because they are a different probe with
     different early stopping (epoch 43/42 vs 46/47). Both are correct; they are
     **not interchangeable** and must never appear in the same table.
3. **Qualitative bit reproducibility.** The measured F5 source PNG is valid,
   but rerendering can select different slices because an internal generator
   state is not reset. Fix and rerun `scripts\show_zero_anatomy_slices.py`
   forward-only before replacing the appendix panel.
4. **F5 vector content.** Its PDF is a raster-preserving wrapper because the
   only stable source artifact is PNG. A truly vector qualitative figure
   requires a bit-reproducible rerender, not image tracing.
5. **LaTeX compile check.** No `latexmk`, `pdflatex`, or `xelatex` executable
   was available. Compile on Overleaf or a TeX installation and confirm the
   main text remains within nine pages.
6. **Working-tree branch mismatch.** Package creation occurred on the existing
   checkout `docs/background-signal-findings`, not the requested
   `normalfix-update`; the checkout was not changed because unrelated tracked
   and untracked work was present and existing files were not to be modified.
