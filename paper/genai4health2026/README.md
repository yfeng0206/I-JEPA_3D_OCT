# GenAI4Health 2026 paper package

## Chosen central claim

**Stock-style I-JEPA batch collation is not anatomy-neutral:** truncating sorted
context indices to the batch minimum deterministically removes lower image
rows, and in OCT this can remove all retinal anatomy from the encoder.

This is the headline because it is supported by code inspection and two
measured batch-size audits. The anatomy-shaped-target hypothesis is presented
as a nuanced negative result because it reverses by epoch 50 and is confounded.

## Build

Overleaf: upload this directory as a project and set `main.tex` as the main
document.

Locally, this project is built with Tectonic (self-contained; no TeX Live
install required). Neither `latexmk` nor `pdflatex` is present on the
development machine:

```powershell
& "D:\jepa_phase0\tools\tectonic\tectonic.exe" -X compile main.tex --keep-intermediates
```

`--keep-intermediates` is required to retain `main.aux`; `--keep-logs` does
not retain it. The main-body page count is then read from the `endofmain`
label, which must stay at page 9 for the GenAI4Health 9-page limit:

```powershell
Select-String -Path main.aux -Pattern "endofmain"
```

The `Fontconfig error: Cannot load default config file` message is benign.

If a full TeX distribution is available instead:

```powershell
latexmk -pdf main.tex
```

The official `neurips_2026.sty` was obtained from the official NeurIPS 2026
formatting archive. No LaTeX executable was available on the package-creation
machine, so local compilation could not be performed.

## Regenerate figures

From the repository root:

```powershell
& 'D:\jepa_phase0\.venv\Scripts\python.exe' `
  'paper\genai4health2026\scripts\make_figures.py'
```

The script uses the Matplotlib `Agg` backend and performs no training, probing,
mask generation, or GPU work.

## Figures

| Figure | Purpose | Measured inputs |
|---|---|---|
| F1 `fig1_crop_defect` | B=1 versus B=64 context loss and zero-anatomy rates | `D:\jepa_phase0\reports\arm_stats_b1\arm_stats.json`; `D:\jepa_phase0\reports\arm_stats\arm_stats.json` |
| F1b `fig1b_context_excision` | Stored retained-versus-discarded encoder context over a B-scan | Crop of `D:\jepa_phase0\reports\arm_stats\zero_anatomy_floor20.png`; no new sampling |
| F2 `fig2_composition_vs_auc` | Epoch-50 composition versus AUC | `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json` |
| F3 `fig3_cover_floor_dose_response` | COVER floor composition, blanking, and explicitly incomplete AUC evidence | `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json`; clean COVER probe `results.json` files |
| F4 `fig4_auc_trajectories` | All clean frozen-probe milestones from the shared ancestor | Probe `results.json` files plus stored random/oracle sweep reports |
| F5 `fig5_zero_anatomy_example` | Qualitative per-slice anatomy accounting | Existing measured `zero_anatomy_floor20.png` |

F1--F4 are natively plotted as vector PDFs and PNGs. F5 begins from a measured
PNG; its PDF preserves that raster without tracing or altering the evidence.

## Number provenance

`EVIDENCE.md` maps every quantitative paper claim to a file and key or line
range. The main source families are:

- context/composition: the three `arm_stats` JSON reports;
- downstream AUC: `D:\jepa_phase0\runs\*\results.json` where present;
- historical random/oracle AUC: stored frozen-sweep reports and composition
  JSON;
- ep30 statistics and confounds: the masking experiment audit documents;
- implementation behavior: `src\masks\multiblock.py`,
  `src\masks\cover.py`, and `src\masks\curriculum.py`.

No number was estimated to fill a missing result. Missing measurements remain
visible as `\TODO{...}` and are enumerated in `GAPS.md`.
