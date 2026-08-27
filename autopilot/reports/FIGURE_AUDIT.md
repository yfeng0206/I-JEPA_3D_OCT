# Figure audit - encoding honesty and accessibility

Paper: `paper/genai4health2026/main_submission.tex`
PDF:   `paper/genai4health2026/main_submission.pdf`
Method: skill `.agents/skills/scientific-visualization` (`SKILL.md`,
`references/publication_guidelines.md`), plus its `scripts/palette_audit.py` and
`scripts/image_metadata.py`, run offline with
`D:\jepa_phase0\.venv\Scripts\python.exe`.
Ground truth for what was drawn: `autopilot/p8_make_assets.py` (4 figures),
`paper/genai4health2026/scripts/make_story_figures.py` (1 figure), and pixel
inspection of the delivered PNGs for the rest.

15 `\includegraphics` targets are referenced: 2 in the body
(`fig:policies`, `fig:traj`) and 13 in the appendix.

## Summary

- **10 of 15 figures had at least one defect.** 3 are serious.
- Most serious: **`fig_specificity_ladder.png` (Fig. `fig:ladder`) is a bar chart
  with `ylim = 0.8600-0.8800`.** Bar length is measured from 0.86, so the
  `envelope` bar renders about **4x** the height of the `random` bar for a true
  AUC difference of about 1.4 percent relative. No interval is drawn on any bar.
  `fig_fairness.png` (left) repeats the same error with `ylim = 0.75-0.93`.
- One hard caption/panel factual error found and fixed: the `fig:fair` caption
  said the scatter covered `\NprobesSub` (23) checkpoints; the panel is drawn
  from `p7_fairness.json`, which holds **19** arms, and the printed rho/p are
  computed on n = 19. Same class of error as the "six intervals" bug.
- The Fig. 2 caption error that was previously caught is **now correct**: panel
  (b) plots nine intervals, the caption claims only that every `envelope` and
  `centroid` interval excludes zero (6 of 9, verified true) and that `cover`
  straddles zero at ep50 and is negative after (verified true).
- **Three mutually inconsistent arm-to-colour mappings** are in use across the
  manuscript. Green is `cover` in the p8 figures and `centroid` in the ladder
  figures; red is `centroid` in the p8 figures and `cover`/`anatomy` elsewhere.
- Colour is the only cue separating arms in 5 figures. Grayscale screening finds
  `centroid` `#C1272D` vs `anatomy-v2` `#8C564B` at **dL\* = 0.36** and
  `envelope` `#2B6CB0` vs `anatomy` `#C53030` at **dL\* = 0.25**: invisible
  differences on a monochrome printer.
- Missing-vs-zero handling is **correct everywhere checked**. `anatomy-v1`
  (ep30 only) and `anatomy-v2` (no ep100) simply end their lines; no segment is
  drawn across an unmeasured epoch and no zero is implied.

## Per-figure table

Verdict key: OK / MINOR / SERIOUS.

| # | Figure (label) | Axis limits | Truncated where length/area is encoded? | Uncertainty labelling | Colour safety | Caption vs panel | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | `fig1_policies_compact.png` (`fig:policies`, body) | image panels, no quantitative axis | n/a | n/a - no estimate shown | 4 target-block colours are the only cue; red/green pair; green contour shares hue with block 1 | Six panels, six policies: correct. Contour = anatomy reference: correct. **Panel heading prints the artefact name `oracle` for `\ArmBest` (centroid), a word the paper never defines** | MINOR |
| 2 | `fig_trajectories_ci.png` (`fig:traj`, body) | (a) y approx 0.847-0.887 AUC, x 25-100 epochs; (b) y approx -0.025 to +0.018 dAUC with an explicit zero line, x categorical | (a) truncated but it is a line/point encoding, legitimate; effects of 0.006-0.012 occupy 15-30 percent of panel height. (b) contains zero: correct | (a) **none** - caption explains why. (b) title states "95% bootstrap CI (10000 draws, same test cases)"; caption now names it percentile and cites `\Nboot`; Methods give unit of replication (test volume = subject) and n = 3000 | Three solid fp16 arms all use `-o`, colour is the only cue. `centroid` vs `envelope` dL\* = 5.25 (fails grayscale screen). Dashed fp32 arms do carry distinct markers (s, ^, D) - good | Nine intervals in (b); caption claims 6 of 9 exclude zero and describes `cover` separately: **verified correct**. Ancestor open circle: correct. Line ends where each arm was last probed: correct gap | MINOR (was: axis + interval name undisclosed; fixed in caption) |
| 3 | `figS5_mask_statistics.png` (`fig:maskstats`) | six panels, every bar from 0; (a) 0-85%, (b) 0-100%, (c) 0-30%, (d) 0-20 cells, (e) 0-160 tokens, (f) 0-8% | No - correct zero baselines | **None.** Means over a 6,137-slice sweep with no SD/SE/CI on any bar | Palette C, 7 of 10 pairs fail grayscale screen; bars are direct-labelled with values, which rescues legibility | 73.1 vs 77.6 in (a): correct. Monotone purity in (b): correct. Caption listed 3 items for panels (c)-(f) (4 panels) - **fixed** | MINOR |
| 4 | `fig_masking_policies.png` (`fig:policies5`) | image grid, 7 columns x 5 rows | n/a | n/a | as #1 | Six policies plus an unmasked leftmost column: correct. Caption listed only 3 rectangle arms of 4 - **fixed**. **Column headings print per-arm AUC at each arm's own epoch (ep100/ep100/ep100/ep30/ep40/ep50), inviting an unmatched-epoch comparison the paper forbids** - now disclosed in the caption. Headings use `oracle` - now disclosed | SERIOUS (mitigated by caption; figure should change) |
| 5 | `fig_fairness.png` (`fig:fair`) | **Left: bars, y = 0.750-0.930.** Right: x 0.848-0.886, y 0.047-0.094 | **YES - left panel bars are truncated at 0.75.** A 0.833 vs 0.900 AUC pair renders as bars in roughly 1:2 height ratio | Left: 95% CI whiskers, group n in legend, but the caption said only "bootstrap intervals" - no level, no method, no unit. Right: none | Left: default C0/C1/C2, no hatching, colour-only for the three race groups. **Right: scatter is coloured by arm with no legend at all** - the colour encoding is unreadable | **Caption said `\NprobesSub` = 23 checkpoints; only 19 points are plotted and rho = +0.51, p = 0.026 is computed on n = 19 (`p7_gap_correlation.json`: `"n": 19`). FIXED to `\NprobesRace` = 19.** Also: dashed line is a least-squares fit while the quoted statistic is Spearman - now disclosed | SERIOUS |
| 6 | `fig_labeleff.png` (`fig:labeleff`) | y approx 0.65-0.89, x log 1-100% with a `ScalarFormatter` | Line encoding, nonzero limit legitimate; log base declared in the x label | **Best in the paper.** Caption: "Shading is one standard deviation over `\LERepeats` label subsets, with every arm seeing identical subsets" - names the interval, the n, the unit of replication, and the pairing | Colour only: all four arms use `marker="o"` and solid lines. `cover` green vs `random` grey dL\* = 5.32 | Four arms, five fractions: correct. Convergence/divergence claim matches the shape | MINOR |
| 7 | `interp_slice_contribution_curves.png` (`fig:interp-slice`) | y -0.10 to +0.04 dlogit, x 0-199 with a twin top axis 0-63 | Line encoding, contains zero: correct | Means with no interval; per-class n in legend | Red/blue/green: `#D62728` vs `#1F77B4` dL\* = 1.14. Class is redundantly coded (solid/dashed) but **probe identity is colour-only** | **Caption said "averaged over all `\Ntest` test volumes". It is not: each probe contributes two class-conditional means (glaucoma n = 1466, healthy n = 1534) and six curves are drawn. FIXED.** Twin axis is a monotone relabelling of the same variable and is labelled: acceptable | SERIOUS (caption); fixed |
| 8 | `interp_04_window_occlusion_W7.png` (`fig:interp-window`) | y -0.31 to +0.22, x 0-199 (data only where a full window fits) | Contains zero: correct | Means, no interval; n in legend | as #7 | Six curves, not three; class split was unstated - **fixed**. Edge truncation is a genuine gap, now stated | MINOR; fixed |
| 9 | `interp_14_odos_mirror_test.png` (`fig:interp-odos`) | 6 panels, x 0-200; y +/-0.05 (meanpool), +/-0.023 (crossattn), +/-0.055 (d1). Left and right panels of a row share limits | Contains zero: correct | Cluster n in legend; no interval | Blue vs dark red, plus solid/dashed in the right column: acceptable redundancy | **Caption said clustering "returns near-perfect mirror images", full stop. The `d1` row shows corr(c1, flip(c2)) = +0.237, which is not near-perfect. The body already carries the caveat "for the two well-structured probes"; the caption had dropped it. FIXED.** Glaucoma-only and per-row y ranges now stated | SERIOUS (caption); fixed |
| 10 | `interp_slice_contribution_by_outcome.png` (`fig:interp-outcome`) | 3 panels sharing y -0.11 to +0.055, x 0-199 | Contains zero, shared limits across compared panels: correct | Means, per-outcome n in legend, no interval - now stated | Dark red / blue plus solid (correct) / dashed (error): good redundancy | TP/FN/TN/FP as described; FN is a scaled TP and FP tracks TN: correct | OK; caption tightened |
| 11 | `interp_heatmap_grid.png` (`fig:interp-heatmap`) | image panels; diverging red/blue colormap | n/a | none | **No colour bar at all.** The caption asserts a "shared scale" that the reader cannot verify, the diverging centre is undeclared, and there are no units | Caption claims per-patch maps, but the overlay is rendered with smoothing so apparent detail is finer than the patch grid; panels are hand-picked "representative" volumes with no stated selection rule. All three now disclosed in the caption | SERIOUS (needs a colour bar); mitigated |
| 12 | `fig_roc.png` (`fig:roc`) | x 0-1, y 0-1 | **Full range, no truncation. Exemplary** | No CI band, but the caption explicitly says the curves are nearly superimposed and that the effect must not be overstated | Three solid curves at the same width, colour-only; AUC is direct-labelled in the legend, which rescues identification | Three arms at epoch 100 on N = 3000: correct. Caption is the most honest in the paper | OK |
| 13 | `fig_precision_paradox.png` (`fig:paradox`) | y approx 0.8635-0.8770, x approx 25-105% purity | Point encoding, so a nonzero limit is defensible, but the whole panel spans about 0.013 AUC, which magnifies the effect | **None.** Five points, a least-squares line through them, no interval anywhere | Palette B, 6 of 10 pairs fail grayscale screen; every point is direct-labelled, which rescues it | Five arms plotted, caption says "four to five arms ... descriptive, not inferential" - honest. Point label `oracle` and the fitted line were undisclosed - **fixed**. Overlapping text labels collide with the title and with each other | MINOR; fixed |
| 14 | `fig_specificity_ladder.png` (`fig:ladder`) | **bars, y = 0.8600-0.8800** | **YES. Worst case in the paper.** Visible bar heights above the 0.86 baseline are 0.0041 (`random`) and 0.0161 (`envelope`): a 3.9x apparent ratio for AUCs that differ by about 1.4 percent | **None on any bar,** although the underlying deltas all have published bootstrap intervals | Palette B; bars direct-labelled | Four bars plus an `oracle` reference line; the two-line caption disclosed none of this - **fixed** | SERIOUS |
| 15 | `fig_geometry_panel.png` (`fig:geom`) | 4 panels, every bar from 0: 0-46%, 0-68%, 0-80%, 0-160 slots | No - correct zero baselines | **None.** Means over 600 slices, no interval | Palette B; bars direct-labelled | **Caption said the four rectangle policies "are closely matched on every axis". Panel 3 (% of anatomy hidden) shows 52.2 / 62.2 / 76.9 / 74.1 - a 25-point spread. FIXED** to "matched on masking ratio, context kept and predictor loss slots, and separate only on the manipulated axis". The 1.6x / 0.4x / half claims all check out. Bars labelled `oracle` and `cover-f0.21` - now disclosed | SERIOUS (caption); fixed |

## Palette audit results (`scripts/palette_audit.py`, background FFFFFF, role `graphical`)

**Palette A - `p8_make_assets.py:41-44`** (trajectories, fairness scatter, ROC, label efficiency):
`random #4C4C4C`, `centroid #C1272D`, `envelope #1F77B4`, `anatomy-v1 #9467BD`,
`anatomy-v2 #8C564B`, `cover #2CA02C`, `ancestor #999999`.
9 of 21 pairs fail the CIE L\* separation screen. Worst pairs:

| pair | dL\* |
|---|---|
| `centroid #C1272D` vs `anatomy-v2 #8C564B` | **0.36** |
| `envelope #1F77B4` vs `anatomy-v1 #9467BD` | 3.48 |
| `centroid #C1272D` vs `envelope #1F77B4` | 5.25 |
| `cover #2CA02C` vs `ancestor #999999` | 5.32 |

`ancestor #999999` also fails the 3:1 graphical-object contrast screen against
white (2.85).

**Palette B - ladder / paradox / geometry**: `#7F7F7F`, `#2CA02C`, `#4363D8`,
`#E6194B`, `#FF7F0E`. 6 of 10 pairs fail; `#7F7F7F` vs `#E6194B` dL\* = 3.66,
`#4363D8` vs `#E6194B` dL\* = 3.79. `#FF7F0E` fails background contrast (2.53).

**Palette C - `make_story_figures.py:37-44`** (figS5): `#6B7280`, `#2F855A`,
`#2B6CB0`, `#C53030`, `#DD6B20`. 7 of 10 pairs fail; `#2B6CB0` vs `#C53030`
dL\* = **0.25**, `#6B7280` vs `#2F855A` dL\* = 1.73.

The same arm therefore carries three different hues across the manuscript, and
in two of the three palettes the two headline arms are indistinguishable in
grayscale.

## File metadata (`scripts/image_metadata.py`)

All 15 PNGs are RGBA with an alpha channel, i.e. transparent backgrounds. The
skill recommends an explicit opaque background for submission because
transparency changes apparent contrast against an unknown page. Effective
resolution at the `\includegraphics` widths used (5.5 in text block):

- Above 300 dpi: all except `fig_roc.png` (270 dpi at `0.62\linewidth`).
- `interp_heatmap_grid.png` reports 133 dpi at `\linewidth`, but it is
  height-constrained by `height=0.84\textheight`, giving about 330 dpi as
  actually placed. Not a defect.

## What I fixed (captions only, no figure regenerated, no digit changed)

All in `paper/genai4health2026/main_submission.tex`:

1. `fig:fair` - **`\NprobesSub` (23) -> `\NprobesRace` (19)**, the macro that is
   actually defined as `fair["n_probes_with_race_summary"]`
   (`p8_make_assets.py:271`) and matches both the 19 plotted points and the n
   behind the printed rho. Also named the interval (percentile bootstrap,
   `\Nboot` resamples, over test volumes, per-group n in the legend), disclosed
   the truncated bar axis, and disclosed that the dashed line is least-squares
   while the quoted correlation is Spearman.
2. `fig:traj` (body) - added "note the truncated AUC axis" to (a) and named
   the interval "percentile bootstrap" in (b). Length compensated by tightening
   "Paired differences are plotted rather than" to "Paired differences replace".
   No caveat removed.
3. `fig:policies5` - corrected the rectangle-arm list from three arms to four
   (`random`, `\ArmBest`, `envelope`, `cover`); disclosed that the column
   headings carry each arm's own probe epoch so the printed AUCs are not a
   matched-epoch comparison; disclosed the `oracle` artefact name.
4. `fig:geom` - replaced "closely matched on every axis" with the accurate
   "matched on masking ratio, context kept and predictor loss slots, and
   separate only on the manipulated axis"; added zero-baseline and no-interval
   statements; disclosed `oracle` and `cover-f0.21`.
5. `fig:ladder` - disclosed the truncated bar axis, the absence of intervals,
   pointed the reader to Table `tab:main`, and disclosed `oracle`.
6. `fig:paradox` - disclosed the truncated axis, the absence of intervals, that
   the dashed line is a least-squares fit, and `oracle`.
7. `fig:interp-slice` - **corrected "averaged over all `\Ntest` test volumes"**;
   the curves are per-class means (glaucoma solid, healthy dashed) with the n in
   the legend and no interval.
8. `fig:interp-window` - stated the same class split and that the curve stops
   where a full window no longer fits.
9. `fig:interp-odos` - **restored the body's caveat "for the two well-structured
   probes"** and stated that the `d1` row does not show the structure; added
   glaucoma-only, cluster n, and per-row y range.
10. `fig:interp-outcome` - stated that curves are per-outcome means with n in
    the legend, that panels share a y range, and that no interval is drawn.
11. `fig:interp-heatmap` - stated that the volumes are hand-picked
    illustrations, that the overlay is smoothed so it carries no detail finer
    than a patch, and that no colour bar is drawn so magnitude is not readable.
12. `fig:maskstats` - mapped the three named confounds onto panels (c, d), (e)
    and (f); added zero-baseline, no-interval, and per-panel-scale statements.

## What needs regeneration (NOT done - would change a rendered figure)

Ranked by severity. No number changes are involved in any of these; they are
axis, marker and annotation changes only.

### Blocking-class

1. **`fig_specificity_ladder.png` - truncated bar baseline.**
   No generator for this figure exists anywhere in the repository (searched all
   files; only `main_submission.tex` and audit reports mention it). The source
   script must be recovered or rewritten. Required change: either set the bar
   axis to include zero, or replace the bars with a dot-and-whisker plot
   carrying the same bootstrap intervals already published in
   Table `tab:main`. A bar whose length is read from 0.86 is the single most
   deceptive encoding in the submission.

2. **`fig_fairness.png` left panel - truncated bar baseline.**
   `autopilot/p8_make_assets.py:761` -> `ax.set_ylim(0.75, 0.93)`.
   Change to a dot-and-whisker (`ax.errorbar(..., fmt="o")`) on the same
   truncated range, which is honest because points encode position, or keep bars
   and drop the `set_ylim` call. Do not simply widen the bar axis to 0-1, which
   would hide the subgroup structure; the dot-and-whisker is the right fix.

3. **`fig_fairness.png` right panel - no legend.**
   `p8_make_assets.py:770-776`. The scatter is coloured by arm with no key, so
   the colour encoding conveys nothing. Add a legend or direct labels. While
   there, `np.polyfit` at line 777 draws a least-squares line under a Spearman
   statistic printed at line 781; either plot a rank-consistent trend or drop
   the line.

4. **`interp_heatmap_grid.png` - no colour bar.**
   No in-repo generator. Add a colour bar with units (`Delta` logit), an
   explicit diverging centre (`TwoSlopeNorm` or `CenteredNorm` at zero) and
   stated limits, and set `interpolation="nearest"` so the patch grid is not
   smoothed into apparent sub-patch detail.

### Consistency-class

5. **`fig_masking_policies.png` - per-column AUC at unmatched epochs.**
   No in-repo generator. Either drop the AUC from the column headings, or move
   the epoch into the heading so the mismatch is visible without reading the
   caption. Currently mitigated by caption text only.

6. **One arm-to-colour mapping for the whole manuscript.**
   Three palettes are in use (`p8_make_assets.py:41-44`,
   `make_story_figures.py:37-44`, and an unrecovered generator for palette B).
   Green means `cover` in one figure and `centroid` in another. Adopt one
   colourblind-safe qualitative set (Okabe-Ito is bundled at
   `.agents/skills/scientific-visualization/assets/color_palettes.py`) and use
   it everywhere.

7. **`oracle` / `cover-f0.21` artefact names on rendered panels.**
   Five figures print `oracle` where the paper says `centroid`
   (`\ArmBest`, defined at `main_submission.tex:28`). The p8 figures already do
   this correctly through `ARM_PLOT` (`p8_make_assets.py:37-40`); the non-p8
   generators need the same mapping. Fig. 1 is a body figure, so this is the
   first thing a reviewer sees.

### Accessibility-class

8. **Redundant encoding beyond colour.**
   - `p8_make_assets.py:690-691`: the three fp16 arms are all plotted `"-o"`.
     Give each a distinct marker and line style, as the fp32 arms at lines
     693-698 already do.
   - `p8_make_assets.py:730`: `fmt="o"` for all three arms in panel (b).
   - `p8_make_assets.py:801-802` (ROC): add `ls=` per arm.
   - `p8_make_assets.py:836` (label efficiency): per-arm `marker=` and
     `linestyle=`.
   - `p8_make_assets.py:41`: `centroid #c1272d` and `anatomy-v2 #8c564b` are
     0.36 dL\* apart. Change one.
   - `p8_make_assets.py:42`: `ancestor #999999` fails 3:1 contrast on white.
   - `fig_fairness.png` left: add hatching to the three race bars.

### Cosmetic

9. `fig_precision_paradox.png`: point labels overlap the title and each other
   (`envelope` collides with the subtitle, `anatomy-v2` with `unguided null`).
10. All 15 PNGs are RGBA. Save with an explicit opaque background.
11. `fig_roc.png` renders at 270 dpi at its placed width; regenerate at a higher
    dpi if the venue asks for 300.
12. `fig1_policies_compact.png` and `fig_masking_policies.png` carry an in-image
    title that duplicates the LaTeX caption.

### Reproducibility gap found during the audit

Only 5 of the 15 figures have a generator in the repository:
`p8_make_assets.py` emits `fig_trajectories_ci`, `fig_fairness`, `fig_roc` and
`fig_labeleff`; `paper/genai4health2026/scripts/make_story_figures.py` emits
`figS5_mask_statistics` (along with figS1-figS4, which are not referenced).
`fig1_policies_compact`, `fig_masking_policies`, `fig_precision_paradox`,
`fig_specificity_ladder`, `fig_geometry_panel` and the five `interp_*` figures
have no recoverable source, so none of the fixes above can be applied to them
without first rewriting their generator. `SOURCES.md` lists all 15 as
"generated" without saying by what.

## Verification after the caption edits

```
p13_build_zip.py      6/6 PASS, main content 9 pages (limit 9), ALL_PASS = True
check_manuscript.py   RESULT: PASS, labels 56, refs 56, dangling 0
p15_verify_numbers.py RESULT: PASS, 20 AUC macros verified against inventory
```

No digit was changed, no caveat was removed, no figure was regenerated, and
nothing was committed.
