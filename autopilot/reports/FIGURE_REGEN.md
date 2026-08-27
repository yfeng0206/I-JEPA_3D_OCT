# Figure regeneration - publication standard, presentation only

Scope: act on `autopilot/reports/FIGURE_AUDIT.md` and
`autopilot/reports/SKILLS_FULL_RUN.md` using the installed skill
`.agents/skills/scientific-visualization` (`SKILL.md`,
`references/publication_guidelines.md`, `references/color_palettes.md`,
`assets/color_palettes.py`, `scripts/palette_audit.py`).

Rule obeyed throughout: **no plotted value changed.** Every claim below is
backed by a machine check, not by inspection.

Headless: `matplotlib.use("Agg")` is set before any `pyplot` import in both
generators and in the probe harness; `MPLBACKEND=Agg` was also exported in every
shell. `plt.show()` is never called anywhere in the touched code. No window
appeared.

---

## 1. What is regenerable, and what is not

Only two generators exist in the repository. A name search for every
`\includegraphics` target across all `*.py` confirms the rest have no source:

| generator | figures |
|---|---|
| `autopilot/p8_make_assets.py` | `fig_trajectories_ci`, `fig_fairness`, `fig_roc`, `fig_labeleff` |
| `paper/genai4health2026/scripts/make_story_figures.py` | `figS1`-`figS4` (not cited) and `figS5_mask_statistics` (cited) |

**No generator exists** for `fig1_policies_compact`, `fig_masking_policies`,
`fig_precision_paradox`, `fig_specificity_ladder`, `fig_geometry_panel`,
`interp_heatmap_grid`, `interp_slice_contribution_curves`,
`interp_04_window_occlusion_W7`, `interp_14_odos_mirror_test`,
`interp_slice_contribution_by_outcome`. Those are listed in section 7.

> **Correction, 2026-08-27.** Two of those now have generators, written after
> this report. `fig_geometry_panel` is built by
> `autopilot/make_fig_geometry_panel.py` from the same artifact that backs
> Table 2, which resolved a numerical disagreement between the figure and the
> table. `fig_specificity_ladder` is built by
> `autopilot/make_fig_specificity_ladder.py` and is no longer a truncated-axis
> bar chart. The remaining eight are still unregenerable.

9 figure stems were regenerated (14 files counting the `.pdf` twins).

---

## 2. Determinism gate, run BEFORE any edit

Without this, an after-diff proves nothing. The unmodified generators were run
against the same inputs and every output byte-compared to the shipped file.

```
auto_numbers.tex                   IDENTICAL     fig_trajectories_ci.png   IDENTICAL
table_allprobes.tex                IDENTICAL     fig_fairness.png          IDENTICAL
table_fairness.tex                 IDENTICAL     fig_roc.png               IDENTICAL
table_fp32.tex                     IDENTICAL     fig_labeleff.png          IDENTICAL
table_labeleff.tex                 IDENTICAL     figS1_background_matters.png IDENTICAL
table_main.tex                     IDENTICAL     figS2_inverted_u.png         IDENTICAL
table_operating.tex                IDENTICAL     figS3_collapse_mechanism.png IDENTICAL
table_paired_subgroup.tex          IDENTICAL     figS4_coverage_floor.png     IDENTICAL
table_subgroup_operating.tex       IDENTICAL     figS5_mask_statistics.png    IDENTICAL
table_subgroup_trends.tex          IDENTICAL
```

`figS5_mask_statistics.pdf` was the single byte difference, and it is a PDF
timestamp, not data:

```
/CreationDate (D:20260826124054-07'00')   ->   /CreationDate (D:20260827020953-07'00')
identical after stripping CreationDate: True
```

The manuscript includes the PNG at that stem, which was byte-identical.
**Determinism holds. The gate is valid.**

---

## 3. Proof that no plotted value changed

### 3a. The source data was never touched

Every generator input, with SHA-256 prefix and last-write time. All predate this
session's edits (which began 2026-08-27 02:0x); the generators only read them.

| file | sha256[0:16] | last written |
|---|---|---|
| `p1b_full_inventory.json` | `C75D7B544090CC1B` | 2026-08-26 16:19 |
| `p1c_stats.json` | `1653ADDB98CCBA8C` | 2026-08-26 16:21 |
| `p7_fairness.json` | `13D296AA7C0304F2` | 2026-08-22 19:06 |
| `p7_gap_correlation.json` | `442D58BDE5569F7B` | 2026-08-22 19:06 |
| `p5_label_efficiency.json` | `E3415B7FAECE7E67` | 2026-08-26 12:29 |
| `p17_subgroup_multiplicity.json` | `E06C4777EAEA7AC3` | 2026-08-26 22:43 |
| `composition_vs_auc_ep50.json` | `6C5753C9D3698B25` | 2026-08-19 14:07 |
| `marginal_token_value.csv` | `1E60B9F1F56B93CA` | 2026-08-14 00:08 |
| `skill_scores.json` | `BB48082C4AF65091` | 2026-08-14 00:07 |
| `region_auc_summary.json` | `0976570C31DEECA8` | 2026-08-14 03:05 |
| `attribution_summary.csv` | `A076607221180D3F` | 2026-08-14 03:14 |

Also unchanged: all ten generated `.tex` files, including `auto_numbers.tex`
(400 macros), are byte-identical to the pre-edit versions. No number that
reaches the prose or a table moved.

### 3b. The arrays fed to matplotlib were captured and compared

`autopilot/_figregen/probe.py` monkeypatches `Axes.plot`, `Axes.bar`,
`Axes.errorbar`, `Axes.scatter`, `Axes.fill_between`, `Axes.axhline` and
`Axes.axvline`, and reads the numbers back **off the returned artists** rather
than parsing the call arguments. That is what makes the bar-to-dot-and-interval
conversion checkable: `bar(x, h, yerr=...)` and `errorbar(x, y, yerr=...)`
normalise to the same record `{x, y, ylo, yhi}` with absolute interval bounds
recovered from the error-bar line segments.

Styling is deliberately not recorded, because styling is what was allowed to
change. Series of 64 points or fewer are stored as full float64 values; longer
series are compared by SHA-256 of their raw float64 bytes. `diff_probe.py`
compares data separately from labels, so a renamed legend entry can never be
mistaken for a changed value and a changed value can never hide behind a
renamed label.

```
fig_fairness.png             series 23 -> 23  plotted_scalars    156  DATA IDENTICAL
fig_labeleff.png             series  8 ->  8  plotted_scalars     80  DATA IDENTICAL
fig_roc.png                  series  4 ->  4  plotted_scalars   7566  DATA IDENTICAL
fig_trajectories_ci.png      series 11 -> 11  plotted_scalars     76  DATA IDENTICAL
[p8] ALL FIGURES DATA IDENTICAL

figS1_background_matters.pdf series  6 ->  6  plotted_scalars     70  DATA IDENTICAL
figS1_background_matters.png series  6 ->  6  plotted_scalars     70  DATA IDENTICAL
figS2_inverted_u.pdf         series 12 -> 12  plotted_scalars     36  DATA IDENTICAL
figS2_inverted_u.png         series 12 -> 12  plotted_scalars     36  DATA IDENTICAL
figS3_collapse_mechanism.pdf series 11 -> 11  plotted_scalars     82  DATA IDENTICAL  (labels renamed)
figS3_collapse_mechanism.png series 11 -> 11  plotted_scalars     82  DATA IDENTICAL  (labels renamed)
figS4_coverage_floor.pdf     series  7 ->  7  plotted_scalars    160  DATA IDENTICAL
figS4_coverage_floor.png     series  7 ->  7  plotted_scalars    160  DATA IDENTICAL
figS5_mask_statistics.pdf    series  6 ->  6  plotted_scalars     60  DATA IDENTICAL
figS5_mask_statistics.png    series  6 ->  6  plotted_scalars     60  DATA IDENTICAL
[story] ALL FIGURES DATA IDENTICAL

strings containing 'oracle' on any canvas: 0   (p8)
strings containing 'oracle' on any canvas: 0   (story)

RESULT: PASS
```

8,714 plotted scalars across 14 files, all bit-identical. The only flagged
change is on `figS3`, and it is confined to two legend strings:

```
label removed: 'oracle'              label added: 'centroid'
label removed: 'blob (near-pure)'    label added: 'anatomy-v2 (near-pure)'
```

with the x and y SHA-256 of those very series unchanged
(`631b8c11d9ef85b4...`, `7b1e660574681964...` on both sides).

Reproduce:

```
D:\jepa_phase0\.venv\Scripts\python.exe autopilot\_figregen\probe.py p8    autopilot\_figregen\after_p8.json
D:\jepa_phase0\.venv\Scripts\python.exe autopilot\_figregen\probe.py story autopilot\_figregen\after_story.json
D:\jepa_phase0\.venv\Scripts\python.exe autopilot\_figregen\diff_probe.py
```

---

## 4. One arm palette, and why redundancy is mandatory rather than preferred

The audit found three conflicting arm palettes, `centroid #C1272D` against
`anatomy-v2 #8C564B` at **dL\* = 0.36**, and `ancestor #999999` failing the 3:1
background screen at 2.85.

One mapping now governs `p8_make_assets.py` and `make_story_figures.py`:

| arm | hex | L\* | contrast vs white | marker | line |
|---|---|---|---|---|---|
| `random` | `#000000` | 0.00 | 21.00 | `o` | solid (fp16) |
| `ancestor` | `#333333` | 21.25 | 12.63 | `*` open | single point |
| `centroid` (key `oracle`) | `#882255` | 31.88 | 8.73 | `s` | solid (fp16) |
| `anatomy-v2` (key `blob`) | `#666666` | 43.19 | 5.74 | `v` | dashed (fp32) |
| `envelope` | `#0072B2` | 45.97 | 5.19 | `^` | solid (fp16) |
| `cover` | `#009E73` | 57.74 | 3.42 | `D` | dashed (fp32) |
| `anatomy-v1` | `#CC79A7` | 61.05 | 3.06 | `P` | dashed (fp32) |

Values are Okabe-Ito and Paul Tol muted, taken from
`.agents/skills/scientific-visualization/assets/color_palettes.py`. Verified,
not eyeballed:

```
palette_audit.py --color 000000 --color 333333 --color 882255 --color 666666 \
                 --color 0072B2 --color 009E73 --color CC79A7 --role graphical

background review_count: 0 / 7   threshold 3.0
grayscale  review_count: 2 / 21  min dL* 10.0
  review pair #666666 #0072B2  dL* = 2.781    (anatomy-v2 vs envelope)
  review pair #009E73 #CC79A7  dL* = 3.304    (cover vs anatomy-v1)
```

Before: **9 of 21** greyscale pairs failing, worst 0.36, and one colour failing
the background screen. After: **2 of 21**, worst 2.78, and **zero** background
failures.

### The exhaustive-search result

The two residual pairs are not an oversight. I enumerated **all 58 colours
bundled in `assets/color_palettes.py`** (Okabe-Ito, Wong, and the nine Paul Tol
schemes). 30 of them clear 3:1 against white. Searching every subset of those 30
for one whose every pair clears dL\* >= 10:

```
subset size 2 : YES  #000000(0.00) #004488(29.19)
subset size 3 : YES  + #0072B2(45.97)
subset size 4 : YES  + #009988(56.64)
subset size 5 : YES  + #222255(16.21)
subset size 6 : IMPOSSIBLE
```

So **five is the hard ceiling, and six is impossible** at contrast >= 3. Worse,
the only surviving five-set is `#000000 / #222255 / #004488 / #0072B2 /
#009988` - black, navy, navy, blue, teal - which is near-monochromatic and
therefore useless as a *qualitative* palette. Greyscale separation and hue
separation trade directly against each other in this colour space.

Six or seven arms therefore **cannot** be separated by colour alone under both
screens. Marker and line-style redundancy is a *necessity* here, not a stylistic
preference, and the skill's own guidance ("Use color plus marker, line style,
hatching, direct label, or panel separation") is the only available resolution.
The palette above deliberately trades the last 2 pairs of greyscale separation
for hue diversity that survives colour-vision deficiency, and pays for it with
redundant encoding.

Redundancy added accordingly. Both residual pairs are additionally separated by
**line style** (solid = fp16 family, dashed = fp32 family) and by a **globally
unique marker**, so neither pair is ever distinguished by hue alone:

- `fig_trajectories_ci` (a): the three fp16 arms were all `-o`; each now carries
  its own marker. The fp32 markers were reassigned so the six arms use six
  distinct markers rather than recycling `s` and `^`.
- `fig_trajectories_ci` (b): `fmt="o"` for all three arms -> per-arm marker.
- `fig_roc`: all three curves are fp16 at epoch 100, so line style stays solid
  and identity is carried by sparse markers (`markevery`), keeping the
  solid/dashed = precision semantics of the trajectory figure intact.
- `fig_labeleff`: all four arms were `marker="o"` -> per-arm marker.
- `fig_fairness` right: per-arm marker as well as colour.
- Race subgroups get their own audited triple, since they are a different
  variable from the arms: White `#000000` (L\* 0.00), Black `#882255` (31.88),
  Asian `#0072B2` (45.97), markers `o` / `s` / `^`. All clear both screens.

---

## 5. Per-figure changes

### `fig_fairness.png` - the truncated bar baseline (audit item: SERIOUS)

- **Left panel converted from bars to dot-and-interval.** `ax.bar(...)` with
  `ax.set_ylim(0.75, 0.93)` encoded 0.006-0.012 AUC effects as bar *length*
  measured from 0.75, so an 0.833 vs 0.900 pair rendered at roughly a 1:2 height
  ratio. Points encode position, which is the honest encoding at this effect
  size and is what the skill's guidelines recommend for exactly this case.
- **The hand-set limit is gone.** The axis is now derived from the data plus a
  reserved band for the key. Widening a point-encoding axis can only understate
  an effect, never exaggerate it.
- **The intervals are kept and named on the figure itself**, not only in the
  caption: "vertical bars: 95% percentile bootstrap CI (3,000 resamples)", with
  the resample count read from `p7_fairness.json["n_bootstrap"]` rather than
  typed. Per-group `n` remains in the legend.
- **Right panel: legend added** (audit item: the scatter was coloured by arm
  with no key at all). Built from proxy `Line2D` handles so the scatter stays
  one call per probe and the plotted values are untouched, with labels drawn
  from `ARM_PLOT` so the artifact name `oracle` cannot reach the canvas. The
  dashed least-squares line is now a labelled legend entry instead of an
  unexplained line; it was retained because removing it would change plotted
  data, and the caption already discloses that the quoted statistic is Spearman.
- **Legend occlusion fixed.** A first pass placed the key over the data and hid
  an `anatomy-v2` point at gap 0.0936. Caught by eye on the rendered PNG, not by
  any tool. Both panels now reserve headroom so no marker sits behind a key.

### `fig_trajectories_ci.png`

Per-arm markers on the fp16 arms; six distinct markers across the six arms;
ancestor recoloured from `#999999` (which failed the background screen at 2.85)
to `#333333` and redrawn as an open star so it is not confused with `random`'s
filled circle in the shared marker map. Axis limits, data and the paired-CI
panel are untouched.

### `fig_roc.png`

Unified palette plus sparse per-arm markers. This figure was already exemplary
on axes (full 0-1 range) and is unchanged in that respect.

### `fig_labeleff.png`

Unified palette; the four hard-coded hexes now resolve through the shared `COL`
map; per-arm markers replace the uniform `marker="o"`. The +/-1 SD bands are
unchanged (proved: `DATA IDENTICAL`).

### `figS5_mask_statistics.png`

Repalletted to the shared mapping. Bars already sat on zero baselines and are
direct-labelled, so the encoding needed no change.

### `figS1`, `figS2`, `figS3`, `figS4` (not cited by the manuscript)

Regenerated so the palette in the code and the artifacts on disk do not drift
apart. While doing so, the label scan caught two arm-name leaks that the earlier
audit had not reported, both in *regenerable* figures:

- `figS1` panel (a) abbreviated the arm as `orac`, and panels (b) and (c) printed
  the raw tick labels `oracle` and `blob`. Now routed through `LABEL`.
- `figS3` panel (a) used the raw family name `oracle` as a legend entry, panel
  (c) built tick labels as `oracle\n100`, and an annotation read `blob`.

All now read `centroid` and `anatomy-v2`.

---

## 6. Caption updates forced by the regeneration

`paper/genai4health2026/main_submission.tex`:

1. **`fig:fair` (was line 991) - stale truncation disclosure deleted.** The
   sentence "the bars sit on a truncated axis, so bar *height* exaggerates the
   differences and only the plotted positions should be read" described a defect
   that no longer exists. Replaced with "Points are group AUCs and vertical bars
   are 3,000-resample class-stratified percentile bootstrap intervals within
   each group". Added "the legend keys colour and marker to policy" for the
   right panel.
2. **`fig:traj` (line ~397) - "open circle" -> "open star"**, matching the new
   ancestor marker.

Deliberately **kept**, as instructed and as still accurate:

- `fig:traj` "note the truncated AUC axis" (line ~394). Panel (a) is a
  line/point encoding, where a non-zero limit is legitimate and disclosed.
- The four disclosures naming `\texttt{oracle}` at lines ~867, ~1829, ~1842,
  ~1858. They document the artifact name printed on
  `fig1_policies_compact`, `fig_masking_policies`, `fig_precision_paradox`,
  `fig_specificity_ladder` and `fig_geometry_panel`, none of which can be
  regenerated. Removing the disclosure while the word is still on the canvas
  would make the caption false.
- `fig:interp-heatmap` "no colour bar is drawn" (line ~1529). Still true; see
  below.

---

## 7. Not fixed, and why

1. **`fig_specificity_ladder.png` - truncated bar baseline, the worst encoding
   in the submission.** No generator exists anywhere in the repository. Bars are
   read from a 0.8600 baseline, giving a 3.9x apparent height ratio for AUCs
   that differ by about 1.4 percent relative. Already disclosed in its caption.
   Out of scope by instruction and unfixable without rewriting the source.
2. **`interp_heatmap_grid.png` - no colour bar.** Requested in item 4, but the
   figure has no in-repo generator (name search across all `*.py` returns
   nothing). It cannot be regenerated, so no colour bar, `TwoSlopeNorm` centre
   or `interpolation="nearest"` could be applied. The caption already states
   that no colour bar is drawn and that magnitude is therefore not readable.
3. **Rendered `oracle` / `cover-f0.21` strings on five figures.** Confined to
   `fig1_policies_compact`, `fig_masking_policies`, `fig_precision_paradox`,
   `fig_specificity_ladder`, `fig_geometry_panel`. No generators. Mitigated by
   caption disclosure only. Note that `ARMW` at `p8_make_assets.py:101` is *not*
   a display path: it builds LaTeX macro *names* such as `\AUCOracleEpHundred`,
   which never render. It was left alone deliberately; renaming it would touch
   about 40 call sites in the manuscript for zero visible effect.
4. **`fig_masking_policies.png` per-column AUCs at unmatched epochs.** No
   generator. Caption-mitigated only.
5. **RGBA / transparent PNG backgrounds.** All PNGs still carry an alpha
   channel; this is a matplotlib default, confirmed in `SKILLS_FULL_RUN.md`
   section 5 as not a defect of ours. Removing it needs a post-export flatten
   step, which is a new tool in the pipeline rather than a presentation fix, and
   it carries a re-encode risk for zero reviewer-visible benefit on a white
   page.
6. **The 2 residual greyscale pairs** (dL\* 2.78 and 3.30). Mathematically
   unavoidable at seven arms; both carry marker and line-style redundancy. See
   section 4.
7. **`fig_precision_paradox.png` overlapping point labels.** No generator.

---

## 8. Verification

Run from `C:\Users\Gary\Desktop\jepa` after the final regeneration.

```
D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p13_build_zip.py
  1_compiles_standalone                    PASS
  2_main_content_within_9_pages            PASS
  3_no_undefined_refs                      PASS
  4_anonymous                              PASS
  5_no_placeholders                        PASS
  6_all_graphics_present                   PASS
  main content pages   : 9 (limit 9)
  ALL_PASS = True

D:\jepa_phase0\.venv\Scripts\python.exe autopilot\check_manuscript.py
  macros defined 400 | used 169 | duplicates 0 | undefined 0
  citations 47 cited, 67 in bib, 0 missing
  labels 56, refs 56, dangling 0
  RESULT: PASS

D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p15_verify_numbers.py
  AUC macros verified against inventory : 20
  RESULT: PASS

D:\jepa_phase0\.venv\Scripts\python.exe autopilot\_figregen\diff_probe.py
  RESULT: PASS
```

Page count re-checked after every regeneration pass and held at 9 of 9
throughout. `check_manuscript.py` reports the same two pre-existing warnings as
before this work (unused generated macros; the `92 probes` literal); neither is
figure-related and neither was introduced here.

`p13_build_zip.py` reports `needs re-upload: auto/fig_fairness.png,
auto/fig_trajectories_ci.png, main.tex` relative to the last uploaded state,
plus `auto/fig_labeleff.png`, `auto/fig_roc.png` and
`figures/figS5_mask_statistics.png` from the earlier pass in the same session.

## 9. Files touched

Modified: `autopilot/p8_make_assets.py`,
`paper/genai4health2026/scripts/make_story_figures.py`,
`paper/genai4health2026/main_submission.tex`, plus the 14 regenerated figure
files and the rebuilt `main_submission.pdf`.

Proof artifacts kept under `autopilot/_figregen/`: `probe.py`, `diff_probe.py`,
`before_p8.json`, `after_p8.json`, `before_story.json`, `after_story.json`.

Nothing was committed.
