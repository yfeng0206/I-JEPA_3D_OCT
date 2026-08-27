# Full run of both installed skills

Date: 2026-08-26
Skills: `.agents/skills/scientific-visualization`, `.agents/skills/statistical-analysis` (v2.64.0)
Python: `D:\jepa_phase0\.venv\Scripts\python.exe`
Scratch (deleted after the run): `autopilot/_scratch_skills/`
Nothing in `paper/`, `results/` or `plots/` was written, overwritten or regenerated.

Two packages were installed into the venv because bundled tools refused to run without them:
`pypdf` (6.16.2) - required by `image_metadata.py` for any PDF; and `statsmodels` (0.14.6) -
required by `assumption_checks.check_regression_diagnostics`. `seaborn` imported fine.

---

## 1. `scientific-visualization/scripts/palette_audit.py`

### 1a. Every paper figure, colours sampled from the rendered pixels

Command (27 PNGs in `paper/genai4health2026/figures` and `.../auto`; dominant chromatic
colours extracted with a scratch Pillow quantiser, near-duplicate anti-alias variants merged
at Euclidean sRGB distance 42, then one `palette_audit.py` invocation per figure):

```
python .agents\skills\scientific-visualization\scripts\palette_audit.py `
       --color <hex> [--color <hex> ...] --role graphical `
       --output autopilot\_scratch_skills\palette\merged_<fig>.json --force
```

Real output (`bg_review` = colours below WCAG 3.0 against white; `gs_review` = colour pairs
below the 10.0 dL\* greyscale screen; only failing pairs listed):

```
fig1_crop_defect.png | n=5 | bg_review=0/5 [] | gs_review=3/10 [#2B6CB0/#2F855A=4.875,#2B6CB0/#992B2A=9.428,#DD6B20/#2F855A=8.391]
fig1_policies_compact.png | n=8 | bg_review=1/8 [#23BC72=2.468] | gs_review=6/28 [#398546/#5D6676=6.631,#398546/#AF2A4B=9.557,#394D8E/#5D6676=8.717,#394D8E/#AF2A4B=5.791,#B18F30/#23BC72=6.717,#5D6676/#AF2A4B=2.926]
fig1b_context_excision.png | n=5 | bg_review=2/5 [#0AAABC=2.804,#BDA3A3=2.351] | gs_review=2/10 [#0AAABC/#009803=9.194,#0AAABC/#BDA3A3=5.516]
fig2_composition_vs_auc.png | n=1 | bg_review=0/1 [] | gs_review=0/0 []
fig3_cover_floor_dose_response.png | n=5 | bg_review=3/5 [#FAECE0=1.158,#EBD1BB=1.461,#C3906E=2.787] | gs_review=1/10 [#FAECE0/#EBD1BB=8.754]
fig4_auc_trajectories.png | n=8 | bg_review=4/8 [#D8B1A4=1.954,#87A4A6=2.661,#A4BDBD=1.983,#B6988F=2.666] | gs_review=8/28 [#D8B1A4/#A4BDBD=0.478,#87A4A6/#B6988F=0.055, ...]
fig5_zero_anatomy_example.png | n=6 | bg_review=2/6 [#0EAEBF=2.684,#B4CEC7=1.667] | gs_review=1/15 [#2B2B2B/#014505=7.023]
fig6_subgroup_disparity.png | n=8 | bg_review=2/8 [#FF7F0C=2.532,#D7BAB4=1.815] | gs_review=11/28 [#826B8E/#1F77B4=0.547,#1F77B4/#D72728=0.94,#826B8E/#D72728=1.487, ...]
fig_auc_trajectories_v2.png | n=5 | bg_review=1/5 [#58AE62=2.747] | gs_review=3/10 [#58AE62/#AD8871=4.743,#B23365/#6476A7=7.627,#6476A7/#AD8871=9.558]
fig_geometry_panel.png | n=4 | bg_review=1/4 [#FF7F0E=2.532] | gs_review=3/6 [#2CA02C/#E6194B=8.375,#2CA02C/#FF7F0E=8.975,#4363D8/#E6194B=3.791]
fig_masking_policies.png | n=3 | bg_review=0/3 [] | gs_review=1/3 [#A8751B/#34763E=8.805]
fig_precision_paradox.png | n=2 | bg_review=0/2 [] | gs_review=0/1 []
fig_specificity_ladder.png | n=6 | bg_review=2/6 [#FF7F0E=2.532,#ADD8AD=1.591] | gs_review=2/15 [#4363D8/#E6194B=3.791,#FF7F0E/#689F69=6.309]
figS1_background_matters.png | n=6 | bg_review=1/6 [#B99B9B=2.553] | gs_review=8/15 [#C53030/#2B6CB0=0.249,#2F855A/#AA6868=1.567, ...]
figS2_inverted_u.png | n=1 | bg_review=1/1 [#EED1C0=1.447] | gs_review=0/0 []
figS3_collapse_mechanism.png | n=5 | bg_review=2/5 [#F1D4D4=1.39,#C3A7A7=2.233] | gs_review=1/10 [#C53030/#4A5568=8.623]
figS4_coverage_floor.png | n=6 | bg_review=3/6 [#77A1CD=2.705,#E5C0B3=1.677,#C19180=2.753] | gs_review=4/15 [#2368AF/#C62827=0.465,#77A1CD/#C19180=0.539,#2368AF/#B6655C=8.575,#C62827/#B6655C=8.11]
figS5_mask_statistics.png | n=6 | bg_review=1/6 [#82A19E=2.785] | gs_review=5/15 [#2B6CB0/#2F855A=4.875,#2B6CB0/#9B2C2C=8.872, ...]
interp_04_window_occlusion_W7.png | n=3 | bg_review=1/3 [#A7D2BA=1.671] | gs_review=1/3 [#CA7373/#73929D=0.547]
interp_14_odos_mirror_test.png | n=3 | bg_review=1/3 [#8DADC7=2.35] | gs_review=0/3 []
interp_heatmap_grid.png | n=3 | bg_review=2/3 [#B89688=2.708,#96A7AF=2.487] | gs_review=1/3 [#B89688/#96A7AF=2.651]
interp_slice_contribution_by_outcome.png | n=2 | bg_review=1/2 [#A1BDD6=1.951] | gs_review=0/1 []
interp_slice_contribution_curves.png | n=4 | bg_review=2/4 [#66B476=2.513,#A2D0AC=1.728] | gs_review=2/6 [#7A9799/#66B476=6.707,#7A9799/#B27978=3.82]
fig_fairness.png | n=6 | bg_review=1/6 [#FF7F0E=2.532] | gs_review=9/15 [#1F77B4/#946D50=1.286,#2CA02C/#6A8A71=3.419, ...]
fig_labeleff.png | n=5 | bg_review=1/5 [#99C7A8=1.894] | gs_review=1/10 [#3F7498/#BA2D32=4.659]
fig_roc.png | n=5 | bg_review=0/5 [] | gs_review=1/10 [#586A78/#C32429=1.057]
fig_trajectories_ci.png | n=4 | bg_review=1/4 [#8CBFA2=2.081] | gs_review=2/6 [#6E8390/#B3565A=5.327,#586773/#B3565A=5.449]
```

Where a bracket ends in `...` the failing-pair list was abbreviated for length; the complete
pair records were written to one JSON per figure under the scratch directory.

A first pass without anti-alias merging is in `figure_colors.tsv` / the un-prefixed JSONs; it
inflates `gs_review` (up to 49/66 for figS1) because it counts edge-blend colours as palette
entries. The merged table above is the honest one.

### 1b. The two arm palettes as declared in code

`scripts/five_arm_audit.py:210-211` (`random`, `cover`, `envelope`, `oracle`, `anatomy`):

```
python ... palette_audit.py --color 888888 --color 3cb44b --color 4363d8 --color f58231 --color e6194b --role graphical
```

```
contrast_screen.review_count = 2   (#3CB44B ratio 2.685, #F58231 ratio 2.586, threshold 3.0)
grayscale_screen.review_count = 5 / 10 pairs
  #888888 / #3CB44B  dL* = 8.347   review
  #888888 / #F58231  dL* = 9.509   review
  #888888 / #E6194B  dL* = 7.175   review
  #3CB44B / #F58231  dL* = 1.162   review     <- cover vs oracle
  #4363D8 / #E6194B  dL* = 3.791   review     <- envelope vs anatomy
```

`scripts/composition_vs_auc.py:198-199` (tab10-based):

```
python ... palette_audit.py --color 888888 --color 2ca02c --color 1f77b4 --color d62728 --color ff7f0e --role graphical
```

```
bg_review 1 / 5 : #FF7F0E contrast 2.532 review
gs_review 6 / 10 :
  #888888 / #2CA02C  dL* = 1.199   <- random vs oracle
  #1F77B4 / #D62728  dL* = 1.138   <- envelope vs blob
  #888888 / #1F77B4  dL* = 8.721
  #888888 / #D62728  dL* = 9.86
  #2CA02C / #1F77B4  dL* = 9.921
  #2CA02C / #FF7F0E  dL* = 8.975
```

**Verdict: REDUNDANT for the finding, USEFUL for coverage.** `autopilot/reports/FIGURE_AUDIT.md`
already states "in two of the three palettes the two headline arms are indistinguishable in
grayscale" and already quotes `#2B6CB0` vs `#C53030` dL\* = 0.25. The re-run adds nothing
qualitatively new; it extends the same conclusion from the 15 submission PNGs to all 27 figures
and to the two palettes as declared in source, and puts a number on the worst pair in the
five-arm palette (`#3CB44B`/`#F58231` dL\* = 1.162). No new class of defect.

---

## 2. `scientific-visualization/scripts/image_metadata.py`

Run on all 27 PNGs and all 17 PDFs in the figure directories, then on the two built manuscripts.

### 2a. PDFs (only possible after installing `pypdf`; this is why the earlier pass skipped them)

```
python ... image_metadata.py <fig>.pdf --target-width-mm 139.7 --min-dpi 300
```

```
fig1_crop_defect.pdf                | 182.5x71.0mm   | fonts=3 all_embedded=True subtypes=/Type0
fig1_policies_compact.pdf           | 262.4x208.4mm  | fonts=3 all_embedded=True subtypes=/Type3
fig1b_context_excision.pdf          | 182.9x70.9mm   | fonts=0 (text drawn as paths)
fig2_composition_vs_auc.pdf         | 183.9x65.7mm   | fonts=2 all_embedded=True subtypes=/Type0
fig3_cover_floor_dose_response.pdf  | 180.6x115.5mm  | fonts=4 all_embedded=True subtypes=/Type0
fig4_auc_trajectories.pdf           | 182.9x93.2mm   | fonts=2 all_embedded=True subtypes=/Type0
fig5_zero_anatomy_example.pdf       | 182.9x138.8mm  | fonts=0 (text drawn as paths)
fig6_subgroup_disparity.pdf         | 288.6x106.2mm  | fonts=2 all_embedded=True subtypes=/Type3
fig_auc_trajectories_v2.pdf         | 184.4x128.9mm  | fonts=1 all_embedded=True subtypes=/Type3
fig_geometry_panel.pdf              | 416.5x100.6mm  | fonts=2 all_embedded=True subtypes=/Type3
fig_precision_paradox.pdf           | 187.9x132.8mm  | fonts=1 all_embedded=True subtypes=/Type3
fig_specificity_ladder.pdf          | 194.5x120.3mm  | fonts=2 all_embedded=True subtypes=/Type3
figS1_background_matters.pdf        | 190.5x62.4mm   | fonts=2 all_embedded=True subtypes=/Type3
figS2_inverted_u.pdf                | 165.6x73.3mm   | fonts=2 all_embedded=True subtypes=/Type3
figS3_collapse_mechanism.pdf        | 186.7x62.4mm   | fonts=1 all_embedded=True subtypes=/Type3
figS4_coverage_floor.pdf            | 186.5x68.3mm   | fonts=3 all_embedded=True subtypes=/Type3
figS5_mask_statistics.pdf           | 187.8x109.7mm  | fonts=2 all_embedded=True subtypes=/Type3
```

Example of the raw record, `fig_auc_trajectories_v2.pdf`:

```
"font_resources": {"all_embedded": true, "embedded_count": 1,
  "fonts": [{"base_font": "/CIEHBG+DejaVuSans", "embedded": true,
             "resource": "/F1", "subtype": "/Type3"}],
  "scope": "first page resource dictionary only", "unembedded_count": 0}
```

11 of 17 PDF figures embed **Type 3** fonts. Six do not (four `/Type0`, two with no font
resources because the text was converted to paths). Scope note printed by the tool itself:
"first page resource dictionary only".

The two built manuscripts, first page only:

```
main_submission.pdf | pages=34 | 215.9x279.4mm | fonts=7 all_embedded=True | all /Type1
main.pdf            | pages=12 | 215.9x279.4mm | fonts=8 all_embedded=True | all /Type1
```

Scoping this honestly: `main_submission.tex` includes **PNG** figures only, so the submission
build is not exposed. `main.tex` includes `figS1`, `figS2`, `figS3`, `figS4`, `figS5`, `fig6`
and `fig1_crop_defect` as PDFs, so six Type 3 figures land in `main.pdf`.

### 2b. PNGs at a fixed 139.7 mm (NeurIPS `\textwidth` = 5.5 in, from `neurips_2026.sty`)

Every PNG is `mode=RGBA`, `has_alpha=True`, `icc=absent`, `meta_dpi=None`. Effective DPI at a
naive full-width placement flagged 8 failures. That is the wrong denominator for figures placed
at a fraction of `\linewidth`, so the run was repeated at the width each figure is actually
printed at, parsed from `main_submission.tex`:

```
fig1_policies_compact.png             | 74.04mm  | 2066x1640 | eff_dpi=708.8 | pass
fig_trajectories_ci.png               | 108.97mm | 2039x840  | eff_dpi=475.3 | pass
figS5_mask_statistics.png             | 139.7mm  | 1769x1036 | eff_dpi=321.6 | pass
fig_masking_policies.png              | 139.7mm  | 2666x1972 | eff_dpi=484.7 | pass
fig_fairness.png                      | 139.7mm  | 1880x780  | eff_dpi=341.8 | pass
fig_labeleff.png                      | 83.82mm  | 1080x580  | eff_dpi=327.3 | pass
interp_slice_contribution_curves.png  | 128.52mm | 1785x803  | eff_dpi=352.8 | pass
interp_04_window_occlusion_W7.png     | 128.52mm | 1785x804  | eff_dpi=352.8 | pass
interp_14_odos_mirror_test.png        | 128.52mm | 1806x1280 | eff_dpi=356.9 | pass
interp_slice_contribution_by_outcome.png | 128.52mm | 2785x760 | eff_dpi=550.4 | pass
interp_heatmap_grid.png               | 55.87mm  | 731x2513  | eff_dpi=332.3 | pass   (height-limited by height=0.84\textheight)
fig_roc.png                           | 86.61mm  | 919x880   | eff_dpi=269.5 | FAIL
fig_precision_paradox.png             | 64.26mm  | 1482x1046 | eff_dpi=585.8 | pass
fig_specificity_ladder.png            | 83.82mm  | 1533x947  | eff_dpi=464.5 | pass
fig_geometry_panel.png                | 139.7mm  | 3280x793  | eff_dpi=596.4 | pass
```

Root cause of the single failure, located while checking: `autopilot/p8_make_assets.py:843`
saves `fig_roc.png` with `dpi=200`.

**Verdict: USEFUL.** The PDF font inspection is new (see Q1). The PNG half is REDUNDANT:
`FIGURE_AUDIT.md` already says "Above 300 dpi: all except `fig_roc.png` (270 dpi at
`0.62\linewidth`)" and "All 15 PNGs are RGBA with an alpha channel", including the correct
height-constrained reasoning for `interp_heatmap_grid.png`. This run reproduced both to the
decimal and extended them to the 12 figures the earlier pass did not cover.

---

## 3. `scientific-visualization/scripts/style_preview.py` (and `style_presets.py`)

```
python ... style_preview.py --output autopilot\_scratch_skills\style\preview_default_okabe `
       --style default --palette okabe_ito_on_white --formats png --dpi 300 --manifest --force
```

```
outputs: preview_default_okabe.png  (199691 bytes)
palette_audit: contrast_screen.review_count = 0
               grayscale_screen.review_count = 4 / 10
settings: dpi 300.0, font_mode "truetype", transparent false, facecolor white
versions: matplotlib 3.11.0, pillow 12.2.0
```

**It cannot render our arm palette.** `style_preview.py:26-35` resolves `--palette` only against
`style_presets.available_palettes()`, and errors with
`unknown palette ...; available: okabe_ito, okabe_ito_on_white, tol_bright, tol_dark,
tol_high_contrast, tol_light, tol_medium_contrast, tol_muted, tol_pale, tol_vibrant, wong`.
There is no hex passthrough. Our five arm hexes cannot be previewed without editing the skill's
`assets/color_palettes.py`, which was out of scope. What it can do is preview a candidate
replacement, and `okabe_ito_on_white` scores 0 contrast reviews (vs 2 for our five-arm palette)
but still 4/10 greyscale reviews, so it is not a free fix either.

The sibling tool `style_presets.py` does work on demand:

```
python ... style_presets.py --write autopilot\_scratch_skills\style\nature_arm.mplstyle `
       --palette tol_high_contrast nature --force
```

produced a style file whose relevant lines are `savefig.dpi: 450`, `savefig.transparent: False`,
`savefig.facecolor: white`, `pdf.fonttype: 42`, `ps.fonttype: 42`, `svg.fonttype: none`.

**Verdict: NOT APPLICABLE for the stated purpose** (it cannot take our palette), **marginally
USEFUL as a side effect**: the generated `.mplstyle` is a one-line-per-item statement of exactly
the three settings that would fix the Type 3, alpha and DPI findings above.

---

## 4. `scientific-visualization/scripts/export_plan.py`

```
python ... export_plan.py --list
```

Profiles available: `acs, bmc, cell, elsevier, ieee, nature, plos, science` (snapshot dated
2026-07-23). **There is no NeurIPS/workshop profile**, and the venue width we actually need,
139.7 mm, is not any profile's named width. The closest is `ieee` `full` = 182.0 mm
(`single` = 88.9 mm).

```
python ... export_plan.py --publisher ieee --figure-type combination --width full `
       --phase submission --input <fig>.pdf --width-tolerance-mm 1.0
```

```
fig_auc_trajectories_v2.pdf         | page_w=184.4mm | format=pass eff_raster_dpi=unknown final_width_mm=fail
fig_geometry_panel.pdf              | page_w=416.5mm | fail
fig_precision_paradox.pdf           | page_w=187.9mm | fail
fig_specificity_ladder.pdf          | page_w=194.5mm | fail
fig1_crop_defect.pdf                | page_w=182.5mm | PASS
fig1_policies_compact.pdf           | page_w=262.4mm | fail
fig1b_context_excision.pdf          | page_w=182.9mm | PASS
fig2_composition_vs_auc.pdf         | page_w=183.9mm | fail
fig3_cover_floor_dose_response.pdf  | page_w=180.6mm | fail
fig4_auc_trajectories.pdf           | page_w=182.9mm | PASS
fig5_zero_anatomy_example.pdf       | page_w=182.9mm | PASS
fig6_subgroup_disparity.pdf         | page_w=288.6mm | fail
figS1_background_matters.pdf        | page_w=190.5mm | fail
figS2_inverted_u.pdf                | page_w=165.6mm | fail
figS3_collapse_mechanism.pdf        | page_w=186.7mm | fail
figS4_coverage_floor.pdf            | page_w=186.5mm | fail
figS5_mask_statistics.pdf           | page_w=187.8mm | fail
```

The `final_width_mm` verdicts are meaningless for us: they compare native page width against an
IEEE target we are not submitting to. LaTeX rescales every figure to `\linewidth` anyway, so a
native page width of 416.5 mm is a font-scaling concern, not a compliance failure. The one
genuinely portable line the tool emits is its own note: "IEEE states greater than 300 dpi for
non-vector color/grayscale and greater than 600 dpi for black-and-white line art."

**Verdict: NOT APPLICABLE.** No profile matches the venue, and the check it can perform
(native page width vs a publisher's fixed column width) is not the constraint a LaTeX
`\includegraphics[width=...]` workflow operates under. It produced zero actionable findings for
this paper.

---

## 5. `scientific-visualization/scripts/figure_export.py`

Write-only, as expected. Sent to scratch; no existing figure touched.

```
python ... figure_export.py --demo autopilot\_scratch_skills\export\demo_neurips `
       --formats pdf,png --dpi 400 --font-mode truetype --style nature `
       --palette okabe_ito_on_white --manifest --force
```

```
outputs: demo_neurips.pdf (28444 bytes, vector-container), demo_neurips.png (106735 bytes)
settings: dpi 400.0, font_mode "truetype", transparent false, facecolor white,
          bbox_inches null, pad_inches 0.1, figure_size_inches [3.5, 2.5]
warnings: []
```

Verified what it would change, by running `image_metadata.py` back over its own output:

```
demo_neurips.pdf | PDF | 88.9x63.5mm | fonts=1 subtypes=/Type0 all_embedded=True
demo_neurips.png | PNG | 1400x1000px | mode=RGBA | alpha=True | meta_dpi=None
```

So if our figure scripts routed through it, `--font-mode truetype` would replace the `/Type3`
resources with `/Type0` (this is the fix for finding Q1), and `transparent: false` /
`facecolor: white` would replace the RGBA transparency. It would **not** fix two things: its own
PNG output is still `mode=RGBA` with `has_alpha=True` and still carries no DPI metadata, so
those two properties of our figures are a matplotlib default, not a defect specific to us.

**Verdict: USEFUL, as a proof rather than a tool.** It cannot be applied to existing figures,
but running it demonstrated concretely that `pdf.fonttype=42` produces `/Type0` and closes the
Type 3 finding.

---

## 6. `statistical-analysis/scripts/assumption_checks.py`

First fact: **the script has no CLI.** `python assumption_checks.py --help` ignores the flag,
executes the `__main__` synthetic demo, and then crashes on Windows:

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 0
  File "...assumption_checks.py", line 616, in comprehensive_assumption_check
    print("\u2713 All assumptions met. Proceed with parametric test (t-test, ANOVA).")
```

It only runs under `PYTHONIOENCODING=utf-8`. Driver used:
`autopilot/_scratch_skills/stats/run_assumption_checks.py`, importing the module and calling
`detect_outliers`, `check_normality`, `check_normality_per_group`,
`check_homogeneity_of_variance`, `comprehensive_assumption_check`, `check_linearity` and
`check_regression_diagnostics` against real artifacts, with `MPLBACKEND=Agg`.

```
paired design confirmed: n=3000 cases, n_pos=1466, identical labels across 3 arms
```

(all 20 `*_test_predictions.npz` under `results/downstream/` share byte-identical `labels`.)

### A. Paired per-case predicted-probability deltas, frozen meanpool ep100

```
--- delta = p(oracle) - p(random) ---
    mean=0.003519  sd=0.091795  median=0.000000  min=-0.354004  max=0.571045
    outliers(IQR 1.5): Found 124 outliers (4.1% of data)  bounds=[-0.1965, 0.2000]
    outliers(IQR 3.0): Found 7 outliers (0.2% of data)
    outliers(|z|>3):   Found 26 outliers (0.9% of data)
    normality: Data do not appear normally distributed (W = 0.981, p = 0.000) (n=3000)

--- delta = p(mirage) - p(random) ---
    mean=0.000800  sd=0.090367  median=0.000854
    outliers(IQR 1.5): Found 183 outliers (6.1% of data)
    normality: Data do not appear normally distributed (W = 0.968, p = 0.000)

--- delta = p(oracle) - p(mirage) ---
    mean=0.002719  sd=0.076009  median=0.000000
    outliers(IQR 1.5): Found 186 outliers (6.2% of data)
    normality: Data do not appear normally distributed (W = 0.972, p = 0.000)
```

Paired per-case squared error (the Brier decomposition a paired bootstrap would resample):

```
SE(oracle) - SE(random) : mean=-0.007548 sd=0.079098 median=-0.003601
                          outliers(IQR 1.5): Found 559 outliers (18.6% of data)
                          normality: do not appear normal (W = 0.863, p = 0.000)
SE(mirage) - SE(random) : mean=-0.005180 sd=0.077892 median=-0.001707
                          outliers(IQR 1.5): Found 582 outliers (19.4% of data)
                          normality: do not appear normal (W = 0.867, p = 0.000)
```

### B. `comprehensive_assumption_check` on raw per-case scores grouped by arm

```
1. OUTLIER DETECTION
   Found 0 outliers (0.0% of data)

2. NORMALITY CHECK (by arm)
 Group    N        W      p-value Normal
random 3000 0.917406 1.069940e-37     No
oracle 3000 0.895606 3.762587e-41     No
mirage 3000 0.900092 1.718208e-40     No
   All groups normal: No

3. HOMOGENEITY OF VARIANCE
   Variances do not appear homogeneous (F = 21.446, p = 0.000, variance ratio = 1.16)
   Consider Welch's correction or transformation

SUMMARY
Normality violated. Use non-parametric alternative.
```

### C. Subgroup operating-point deltas (`results/p16_subgroup_operating.json`, spec target 0.90, ep100)

```
--- subgroup delta sensitivity: envelope - random (k=5 subgroups) ---
    Asian=-0.0169, Black=-0.0149, White=+0.0250, Female=+0.0037, Male=+0.0275
    outliers(IQR 1.5): Found 0 outliers (0.0% of data)
    normality: Data appear normally distributed (W = 0.859, p = 0.225) (n=5)

--- subgroup delta specificity: envelope - random ---
    Asian=+0.0301, Black=+0.0245, White=-0.0048, Female=+0.0000, Male=+0.0032
    0 outliers; normal (W = 0.866, p = 0.252)

--- subgroup delta brier: envelope - random ---
    Asian=-0.0003, Black=-0.0005, White=-0.0066, Female=-0.0044, Male=-0.0062
    0 outliers; normal (W = 0.834, p = 0.149)

--- subgroup delta sensitivity: intensity - random ---
    Asian=+0.0085, Black=+0.0037, White=+0.0343, Female=+0.0259, Male=+0.0275
    0 outliers; normal (W = 0.895, p = 0.383)

--- subgroup delta specificity: intensity - random ---
    Asian=+0.0075, Black=+0.0245, White=-0.0162, Female=-0.0077, Male=-0.0127
    0 outliers; normal (W = 0.897, p = 0.392)

--- subgroup delta brier: intensity - random ---
    Asian=-0.0086, Black=-0.0046, White=-0.0080, Female=-0.0076, Male=-0.0075
    outliers(IQR 1.5): Found 1 outliers (20.0% of data); values=[-0.0046]
    normality: Data appear normally distributed (W = 0.805, p = 0.089) (n=5)

Levene across arms on subgroup sensitivity:
    Variances appear homogeneous (F = 0.058, p = 0.944, variance ratio = 1.54)
    Group  N        W  p-value Normal
   random  5 0.896004 0.388196    Yes
 envelope  5 0.984471 0.957024    Yes
intensity  5 0.940516 0.669591    Yes
```

### D. The dashed least-squares line in `fig_fairness.png` (race gap vs AUC)

```
n probes with a race summary = 19 (paper reports n=19)
    Pearson r=0.4199  r^2=0.1763
    outliers in gap: Found 1 outliers (5.3% of data); values=[0.0935]
    OLS slope=0.4053 (SE 0.2124), p=0.0735, R^2=0.1763
    residual normality (Shapiro-Wilk): W=0.9102 p=0.0746 ok=True
    heteroscedasticity (Breusch-Pagan): LM=0.1919 p=0.6614 ok=True
    autocorrelation (Durbin-Watson): DW=0.6494 ok=False
    max VIF=0.9999999999999998
    Concerns: possible autocorrelation (if data are ordered/temporal, use time-series
    methods or cluster-robust SEs)
```

### Does any of this matter for the estimators we actually used?

`main_submission.tex:314-320`: "percentile bootstrap intervals for primary AUCs and paired deltas
(\Nboot{} class-stratified subject resamples ...) and DeLong tests for correlated ROC curves".
No t-test, no ANOVA, no OLS inference appears in the paper.

| Assumption flagged | Matters for our estimator? | Why |
|---|---|---|
| Per-case deltas non-normal (W = 0.968-0.981, p < 1e-10) | **No** | A percentile bootstrap makes no distributional assumption about the per-case deltas; that is the entire point of choosing it. It would matter for a paired t-test, which we do not report. |
| Per-case Brier deltas non-normal (W = 0.863-0.867) | **No** | Same reason. |
| 4.1-19.4% IQR "outliers" in the deltas | **No** | Most cases have delta = 0 exactly (median 0.000000 for two of three pairs), so the IQR is tiny and the flag counts the ordinary tail, not data errors. The IQR 3.0 rule finds 0.2-0.4%. Bootstrap resampling already propagates this tail into the interval width. |
| Raw scores non-normal within arm (p ~ 1e-37) | **No** | An AUC is rank-based; DeLong's variance is built from placement values, not from Gaussian scores. |
| Levene rejects homoscedasticity across arms (F = 21.446, p < 0.001) | **No** | Variance ratio is 1.16, i.e. a 16% difference detected only because n = 9000; and the comparison is *paired on identical cases*, so no equal-variance assumption is used anywhere. |
| Subgroup deltas normal, homoscedastic, no outliers | **Uninformative** | Shapiro-Wilk at n = 5 has essentially no power. "Normal: Yes" here means "cannot reject", not "is normal". No conclusion should be drawn either way. |
| Durbin-Watson 0.649 on the LS line | **No new action** | The 19 probes are not a time series; DW is picking up that the points cluster by arm and by checkpoint, i.e. non-independence. The paper already declines to interpret that line: "The slope is positive but fails multiplicity correction at checkpoint level and does not survive branch-level aggregation ... so we draw no trend conclusion from it." The tool corroborates an existing, disclosed caveat. |

One positive: BP p = 0.6614 and residual-normality p = 0.0746 mean the OLS line itself is not
additionally pathological; the only issue with it is the clustering the paper already names.

**Verdict: USEFUL, and its usefulness is confirmatory.** It found no defect. What it did do is
convert "we used a non-parametric paired bootstrap because that is the careful thing to do" into
a measured statement: the per-case delta distributions are strongly non-normal with heavy tails,
so a paired t-test or normal approximation on them would have been the wrong tool. Nothing in
the paper needs to change.

---

## Q1. Did any tool find something the earlier audits missed?

Yes, one thing, from `image_metadata.py` on PDFs:

> `"fonts": [{"base_font": "/CIEHBG+DejaVuSans", "embedded": true, "resource": "/F1", "subtype": "/Type3"}]`

11 of the 17 PDF figures embed Type 3 fonts. Searching the repository, no audit report,
checklist or note mentions Type 3, `fonttype`, or font embedding anywhere outside the skill's own
reference files - the only hits are `.agents/skills/.../publication_guidelines.md:159` and
`journal_requirements.md:20`. The earlier pass could not have found it: `image_metadata.py`
exits with `error: pypdf is required for PDF metadata` unless `pypdf` is installed, so every PDF
was silently skipped.

Scope, stated honestly: `main_submission.tex` includes PNGs only, so the submission build is
unaffected. `main.tex` includes six of the Type 3 PDFs, so `main.pdf` carries them. Type 3 is a
common automated-checker rejection (arXiv, IEEE PDF eXpress); the fix is one line,
`matplotlib.rcParams["pdf.fonttype"] = 42`, and `figure_export.py`'s own output confirms it
yields `/Type0`.

Everything else the visualization tools reported was already in `autopilot/reports/FIGURE_AUDIT.md`,
including the exact `fig_roc.png` 270 dpi figure and the height-constrained
`interp_heatmap_grid.png` correction. `assumption_checks.py` had never been executed, so all of
its output is new, but none of it changes a conclusion.

## Q2. Was each tool worth running on THIS paper?

| Tool | Verdict | Reason |
|---|---|---|
| `palette_audit.py` | REDUNDANT | Same class of finding as the earlier pass. Extended coverage from 15 to 27 figures and quantified the two source-declared arm palettes, but found no new kind of problem. Worth keeping as a regression check, not worth re-running for insight. |
| `image_metadata.py` | USEFUL | PDF font inspection produced the one genuinely new finding. The PNG DPI/alpha half was a byte-for-byte reproduction of what was already known. Net: worth it, but only because `pypdf` was installed this time. |
| `style_preview.py` | NOT APPLICABLE | Cannot accept custom hex colours; bundled palettes only. It cannot preview our arm palette, which was the reason to run it. |
| `style_presets.py` | Marginal | Not on the requested list but bundled; its generated `.mplstyle` happens to encode the three settings that would fix the Type 3, alpha and DPI findings. |
| `export_plan.py` | NOT APPLICABLE | No NeurIPS/workshop profile exists, the venue's 139.7 mm width is not a named width in any profile, and native PDF page width is not the constraint a `\includegraphics[width=\linewidth]` workflow has. Zero actionable findings. |
| `figure_export.py` | USEFUL (as proof) | Cannot be applied to existing figures, but its demo output proved `--font-mode truetype` yields `/Type0`, closing the loop on the Type 3 finding. Also showed the RGBA/no-DPI-metadata properties are matplotlib defaults, not our defect. |
| `assumption_checks.py` | USEFUL (confirmatory) | Found no defect. It converted an untested methodological choice into a measured one: the paired per-case deltas are strongly non-normal with heavy tails, which is exactly the case where a non-parametric paired bootstrap is correct and a t-test is not. Also surfaced that the script has no CLI and crashes on Windows cp1252 without `PYTHONIOENCODING=utf-8`. |

### Only actionable item from the whole run

Set `pdf.fonttype = 42` in the figure scripts that emit PDFs and regenerate the 11 affected PDF
figures - but only if `main.tex` (not `main_submission.tex`) is a build target. No paper text,
number or figure was changed by this run.
