# GenAI4Health @ NeurIPS 2026 poster (draft)

Poster for the accepted workshop paper "Where to Predict in Retinal OCT?
Anatomy-Guided Target Selection for I-JEPA" (GenAI for Health Workshop,
NeurIPS 2026, Sydney, Dec 12 2026; every accepted paper presents a poster).

Status: draft. Authors and affiliations are final. The QR link is pending a
decision (see "What to fill in later"). The registered-replication block fills
in automatically from the generated results after the Oct 22 results freeze;
until then it shows TBD placeholders and a RESULTS PENDING banner.

## Files

| file | role |
|---|---|
| `poster_size.tex` | the single size parameter `\PosterSize` |
| `poster_setup.tex` | maps `\PosterSize` to the print size and the design canvas |
| `poster.tex` | design master (tikzposter); produces `poster.pdf` |
| `poster_print.tex` | scales `poster.pdf` to the exact print size; produces `poster_print.pdf` |
| `build.ps1` | results gate (`make_preview.py --precheck`), both PDFs with Tectonic, PNG preview and build report |
| `make_preview.py` | results gate (validates the generated bundle, writes `generated/`), headless PNG render (PyMuPDF) and checks: page count, size, Overfull boxes, fit, placeholders, figure allow-list and registered poster-variant hashes, KeyBox/Answer wording against the paper macros, every result cell/label/late-run mark in the PDF against the generator's metadata |
| `tests/test_results_gate.py` | regression tests of the results gate on a synthetic bundle (tampered numbers, labels, macros, figure, evidence incl. panel (b) and encoding, sidecar, registered RANDOM reference) |
| `tests/test_build_guard.ps1` | regression tests of the synthetic-output repository guard (both shells, with and without git) |
| `generated/` (ignored) | `results_state.tex/.json` and `fig_cr_seed_spread_poster.pdf`, written by the gate for the current results only |
| `poster_print.pdf` | file to print (exact physical size) |
| `poster_preview.png` | low-resolution preview (50 dpi by default) |

The poster reads its sources in place (no copies):
`../../autopilot/investigations/camera_ready_20261008/figures_v2/poster/fig_oct_strategies_v2_poster.pdf`
and `.../fig_oct_jepa_volume_v2_poster.pdf` (I13's poster variants of the
paper's Fig. 1 and Fig. 2: same data, labels of at least 14 pt at 12 in width;
registered with their SHA-256 under `poster_variant` in
`../../autopilot/release_assets.json`, which the report checks),
`../../paper/genai4health2026/auto/auto_numbers.tex` (Table 1 macros) and
`../../paper/genai4health2026/auto/cr_numbers.tex` (delivered-mask, guide
agreement and reference macros). After the results freeze the gate validates
the bundle that `autopilot/make_cr_results.py` writes to the same `auto/`
folder (`cr_results.tex`, `cr_results.values.json`, the generated tables,
`fig_cr_seed_spread.pdf/.png` and `fig_cr_seed_spread.evidence.json`); the
poster then inputs `cr_results.tex` (`\Rep*` macros) and the gate's
`generated/results_state.tex`. Every number on the poster is one of these
macros; do not type result numbers into `poster.tex`.

### Layout (R1 polish review, section 5)

- Left column: why OCT (question, then a navy **Answer** box with the
  epoch-50 original-run differences `\DOracleRandomEpFifty`,
  `\DEnvelopeRandomEpFifty` and, once results are validated, the generated
  replication labels, digit-free); **Five target-selection strategies** (v2
  poster figure; the old strategy table is dropped because the figure's column
  heads and target-rule row carry it); **Same I-JEPA, different sampler;
  volume-level probe** (v2 framework figure with the one-line $\Pi_s$ caption).
- Right column: neutral title "Single runs: tissue-directed rectangles scored
  higher" (the replication may disagree); KeyBox at the registered epoch 50
  for both arms with 95% CIs, "positive at epochs 50, 75 and 100" with the
  per-arm range (R1 H1), ANATOMY-V2 minus ENVELOPE at epoch 50 and COVER minus
  RANDOM at epoch 100 (no post hoc peak); no bold AUC; delivered-mask table
  with the context column on a light fill; guide chart with the overall rate
  as a dashed reference and "% of B-scans" axis; takeaways unchanged.
- The report checks the wording that depends on the original-run macros
  (`Claims`): all six guided-minus-RANDOM differences positive, the range end
  points (ENVELOPE epoch 100 to 50, CENTROID epoch 50 to 75), ANATOMY-V2
  minus ENVELOPE and COVER minus RANDOM negative, and both context gaps
  positive ("guided masks also leave more context visible").
- Replication block: the label rule (pre-registration section 3, amendment 1)
  in the RESULTS PENDING banner and, with results, at the end of the outcome
  strip; pooling panel notes the mean+max variant (paper).
- Credits: "adapted (cropped, downsampled, overlays added)"; QR placeholder
  "paper, code & pre-registration" (R1 M15).

## Build

```powershell
cd poster\genai4health2026
pwsh -NoProfile -File .\build.ps1                 # size from poster_size.tex
pwsh -NoProfile -File .\build.ps1 -Size a0        # one-off size override
pwsh -NoProfile -File .\build.ps1 -OnlyCached     # offline (after one online build)
pwsh -NoProfile -File .\build.ps1 -Dpi 100        # larger PNG
pwsh -NoProfile -File .\build.ps1 -AllowSynthetic # generator dry runs, scratch copies only
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build.ps1   # Windows PowerShell 5.1
```

- Tectonic: `D:\jepa_phase0\tools\tectonic\tectonic.exe` (override with
  `$env:TECTONIC`). It runs at Idle priority, one compile at a time (the script
  refuses to compile if another `tectonic` process is running). All packages
  (tikzposter, FiraSans, unicode-math with Fira Math, pgfplots, qrcode,
  pdfpages) are in the Tectonic bundle and are cached after the first build.
- Python: `$env:POSTER_PYTHON`, else `D:\jepa_phase0\.venv\Scripts\python.exe`
  if present (has PyMuPDF and matplotlib), else `python`. It runs CPU-only
  (`CUDA_VISIBLE_DEVICES=-1`, one thread). Python output is relayed line by
  line, so Windows PowerShell 5.1 does not turn stderr into a
  `NativeCommandError`.
- Other TeX installations: run `make_preview.py --precheck` first (results
  are shown only with its validated `generated/results_state.tex`), then
  compile `poster.tex` and `poster_print.tex` with XeLaTeX or LuaLaTeX
  (fontspec/unicode-math; pdfLaTeX will not work), from this folder so the
  `../../paper/...` and `../../autopilot/...` paths resolve.
- `build.ps1 -Size X` writes temporary job files `poster-X.tex` and
  `poster_print-X.tex`, builds `poster-X.pdf`, `poster_print-X.pdf` and
  `poster-X_preview.png`, then deletes the job files.
- Before compiling, `build.ps1` runs the results gate
  (`make_preview.py --precheck`, see below). Invalid results and synthetic
  results (without `-AllowSynthetic`) stop the build before anything is
  compiled, so the tracked outputs are never overwritten with them.
- The report ends with `Checks OK` or `PROBLEMS ...` (exit code 1): more than
  one page, content running past the bottom margin (tikzposter does not warn
  about this itself), a figure outside the allow-list, a paper figure that is
  not the committed poster variant or whose SHA-256 differs from its
  `poster_variant` entry in `autopilot/release_assets.json`, KeyBox/Answer
  wording that no longer matches the paper macros, a `POSTER-PROBLEM`
  line in the log, an invalid or synthetic (without `-AllowSynthetic`)
  results bundle, a stale gate state, or any result cell, label or late-run
  mark in the PDF that differs from the sidecar. The `Results` lines state
  whether results were loaded (`none`, `real` or `SYNTHETIC DRY RUN`), what was
  validated, the seed indices shown and descriptive, the matched-control
  label with the placement (P) and budget (B) contrasts' on-time/late status,
  and the figure used; `Cells` gives the number of result slots compared.

## Registered replication (automatic)

### Results gate (before compiling)

`make_preview.py --precheck` validates the bundle in
`../../paper/genai4health2026/auto/` and fails closed. The sidecar
`cr_results.values.json` is the root of trust; everything else must agree
with it:

- sidecar: schema `cr_results_v1`, boolean `synthetic`, `outputs` hash map
  (must include `cr_results.tex` and `cr_table_replication.tex`), `figure`
  (`null` or PNG/PDF hashes), `scenario` (labels, `complete_seed_indices`,
  `reference_missing`, `matched_control` with `label`, `cb_present`,
  `cb_eligible`, `P`/`B`, `P_descriptive`/`B_descriptive` and
  `reference_missing`), `probe_runs`, `macros` and `text_macros` records;
- sidecar self-consistency: every numeric `display` equals its recorded
  format expression applied to its `full_precision` value(s) (`{,}` grouping
  aside); the matched-control label (and name) equals the pre-registered rule
  of amendment 2 applied to the sidecar's P, B, registered RANDOM reference
  and eligibility flags (as `make_cr_results.matched_outcome`);
- registered RANDOM reference (pre-registration section 3, reused by
  amendment 2; generator V10-2): the on-time re-probed original RANDOM plus
  the new RANDOM runs of at least two complete seed indices. The recorded
  `reference_missing` reasons (identical in `scenario` and
  `matched_control`) must agree with the sidecar: `indices` exactly when
  fewer than two seed indices are complete, `anchor_missing` exactly when
  there is no original RANDOM run (`probe_runs`, `\RepAUCRandomOrig`),
  `anchor_late` only with an original RANDOM result (its lateness is recorded
  only there), `range` only alone. With the reference, the matched-control
  range and `\RepRandomRange`/`Min`/`Max` must equal the range, minimum and
  maximum of the full-precision AUCs of exactly those RANDOM runs. Without
  it: no registered description (`M_incomplete` unless `M_na`, `M_noP` or
  `M_desc` applies), no range in the sidecar, no `\RepRandomRange`/`Min`/`Max`
  (so no range is shown), and neither guided arm may carry a registered
  replication label (only `incomplete`/`not available`) when the original
  RANDOM is missing or late or fewer than two indices are complete;
- every file the sidecar lists (all generated tables, `cr_tldr.txt`, the
  figure PNG and PDF) matches its SHA-256, the generator's own convention;
  a figure file without sidecar hashes is rejected;
- `cr_results.tex` contains nothing but one `\newcommand{\Rep...}` per line
  and defines exactly the sidecar's numeric and text macros with identical
  values; the four sentinels exist; `\RepLabelCentroid`/`\RepLabelEnvelope`
  equal the sidecar labels; `\RepPlacementDelta`/`\RepBudgetShift` exist
  exactly when the sidecar has P/B (with the same full precision); the
  synthetic marker agrees with the flag;
- with a figure, `fig_cr_seed_spread.evidence.json` is required: same schema
  and synthetic flag, the sidecar's figure hashes, and its arms, points and
  means must equal (as sets: nothing extra, nothing missing, no duplicates,
  exact record keys) the series built from the sidecar alone: one point per
  `\RepAUC<Arm><Run>` macro with its full-precision AUC, head-seed s.d.,
  position, seed and status, one mean per `\RepMeanAUC<Arm>` macro with its
  value, position and the number of complete seed indices; if the evidence has
  panel (b) (current generator), it must equal the panel built from the
  sidecar in the same way: one point per paired difference
  `\RepDelta<Arm>Seed<X>` (registered iff the index is complete), one mean per
  `\RepDeltaMean<Arm>`, and the band drawn exactly when `\RepRandomRange`
  exists (registered RANDOM reference) with that full-precision range and the
  recorded `reference_missing`; its `encoding` (contract
  `cr_seed_spread_encoding_v2`: arm colours `#RRGGBB`, arm markers, fill by
  status, seed letters, offsets equal to the poster's, grey band colour, and
  the optional `late_original` entry) is validated; another contract or any
  encoding field the poster does not know is refused (update the poster
  render);
- late originals (generator V14-1: a re-probed original that finished after
  the freeze): the hash-checked per-run table `cr_table_seeds.tex` must list
  every original AUC macro once with analysis status `original` or
  `original, late`; the original RANDOM is late exactly when
  `reference_missing` says `anchor_late`; the evidence `late_originals` (run
  names, mapped to strategies through the sidecar's `probe_runs`) must name
  exactly the late ones (absent only when none is late), and the encoding must
  then have its `late_original` entry;
- if `autopilot/make_cr_results.py` is importable (repository layout, or
  `$env:POSTER_GENERATOR_DIR`), its own `verified_results_metadata()` must
  also pass (read-only use).

On success it writes `generated/results_state.tex/.json` (control label,
`cb_present`/`cb_eligible`, P/B late flags, the missing parts of the
registered RANDOM reference (JSON only), shown and descriptive seed
indices, the list of expected macros, MD5s) and renders
`generated/fig_cr_seed_spread_poster.pdf`, a poster-size version of the
seed-spread plot drawn from that sidecar-built series (nothing is plotted
from the evidence file itself; `results_state.json` lists the plotted points,
means, panel (b) points and band, and the encoding used) with the generator's
encoding read from the validated evidence (I12's contract
`cr_seed_spread_encoding_v2`: the paper's arm colours RANDOM `#000000`,
CENTROID `#882255`, ENVELOPE `#0072B2`, RANDOM-CB `#E69F00`; arm markers
o/s/^/hexagon; filled = registered or on-time control, left half filled =
re-probed original, open = descriptive, including a late original (encoding
`late_original`; legend "descriptive (late)" when no other run is
descriptive); seed letters O/A/B/C under the
points; arm-coloured registered means; head-seed s.d. error bars; panel (b)
paired differences with the grey ±RANDOM-range band only with the registered
reference). Evidence without an encoding (older generator) uses the same v2
values as poster defaults; evidence without panel (b) gives panel (a) only.
The render is 8.1 x 3.4 in with 14 pt text (13 pt arm names), legend below
(without matplotlib it falls back to the validated generator PDF). Any
failure deletes `generated/` state and exits 1.

`poster.tex` shows results only if `cr_results.tex` defines the sentinel
macros `\RepEndpoint`, `\RepNSeedsWord`, `\RepLabelCentroid` and
`\RepLabelEnvelope` and a gate state exists whose MD5 matches the current
`cr_results.tex` (and, for the figure, the figure file); otherwise it keeps
the placeholders and logs `POSTER-PROBLEM`, which fails the report. The
report then re-validates the bundle and the state and extracts every result
cell, the labels, the control text and the late-run marks from the print PDF
and compares them with the sidecar.

### Cells

Every result cell is `\PosterCell{<macro>}` (or `\PosterCellM` for signed
differences, set in math): the generated value when results are loaded;
`---` only where the validated sidecar has no such macro (a missing or
unregistered run, exactly as in the paper's generated tables); `MISSING` (and
a failed report) for a macro the sidecar lists but the file lacks; the TBD
placeholder before the freeze. The cells use the same quantities as the
paper's generated tables:

| poster cell | macro | paper table (generated) |
|---|---|---|
| Repeated continuations, `original` (RANDOM / ENVELOPE / CENTROID) | `\RepAUCRandomOrig`, `\RepAUCEnvelopeOrig`, `\RepAUCCentroidOrig` | `tab:replication`, row "original (re-probed)" |
| `seed A/B/C` (one column per seed index with any run) | `\RepAUC<Arm>Seed<A/B/C>` | `tab:replication`, rows "seed A/B/C" |
| `mean` | `\RepMeanAUCRandom`, `\RepMeanAUCEnvelope`, `\RepMeanAUCCentroid` | `tab:replication`, row "mean (new seeds)" |
| `mean Δ` (ENVELOPE, CENTROID; RANDOM is `---`) | `\RepDeltaMeanEnvelope`, `\RepDeltaMeanCentroid` | `tab:replication`, row "mean (new seeds)", Δ columns |
| Matched-budget control, CENTROID − RANDOM-CB (placement) | `\RepPlacementDelta`, `†` if `P_descriptive` | `tab:control`, last row |
| RANDOM-CB − RANDOM (budget) | `\RepBudgetShift`, `†` if `B_descriptive` | `tab:control`, RANDOM-CB row Δ |
| Pooling, max `within` B-scan (ENVELOPE, CENTROID) | `\RepPoolPatchMaxDeltaEnvelope`, `\RepPoolPatchMaxDeltaCentroid` | `tab:pooling`, row "max, mean" Δ |
| Pooling, max `across` B-scans | `\RepPoolSliceMaxDeltaEnvelope`, `\RepPoolSliceMaxDeltaCentroid` | `tab:pooling`, row "mean, max" Δ |
| Outcome strip, registered labels | `\RepLabelCentroid`, `\RepLabelEnvelope`, `\RepRandomRange` (only with the registered RANDOM reference) | `tab:pooling` labels, results text |
| Outcome strip, matched-budget control | sidecar `scenario.matched_control.label` (via the gate state) | registered description (pre-registration amendment 2) |
| Seed-spread panel | `generated/fig_cr_seed_spread_poster.pdf`, panels (a) AUC per encoder and (b) paired Δ vs RANDOM (from the sidecar; equal to `fig_cr_seed_spread.evidence.json`) | `fig:seed_spread` |

Before the freeze the block has 17 TBD cells (the "new seeds" column is one
placeholder per strategy), a FIGURE PENDING box and a RESULTS PENDING banner.

### Scenarios

- Seed indices: a column appears for every seed index with a generated AUC
  (two or three; seed C may be absent). A seed index with runs that is not in
  the sidecar's `complete_seed_indices` (incomplete at the freeze, for
  example a late seed C or a missing run) gets `*` in its header and the note
  "Incomplete at the freeze: descriptive"; it is not in the mean, Δ or label.
- Missing runs show `---` (for example CENTROID seed A, RANDOM seed A, no
  RANDOM-CB run, no complete index for a mean).
- Late runs in the matched control: when the sidecar marks the placement (P)
  or budget (B) contrast as descriptive (`P_descriptive`, `B_descriptive`,
  for example a RANDOM-CB run after the freeze, also with no CENTROID result
  (`M_noP`), or a late RANDOM seed-A run with an on-time P), the value gets a
  dagger and the control panel says "†From a run after the freeze:
  descriptive."; the figure shows such runs with open markers.
- Late originals: a re-probed original that finished after the freeze (per-run
  table "original, late") gets a dagger on its `original` cell, as the paper's
  generated replication table marks its original row, and the outcome strip
  says "†Original re-probed after the freeze: descriptive." (with a
  descriptive seed: "*Incomplete at the freeze, †original re-probed after the
  freeze: descriptive."); the figure draws it open.
- Matched-budget control description: M+ "placement accounts for most of the
  gap", M0 "placement difference not detected", M− "placement difference
  reversed", both contribute (`M_partial`), each marked "(registered
  description)"; without a description: `M_desc` (a run finished after the
  freeze), `M_unassessed` (no on-time budget shift), `M_noP` (no CENTROID
  result at seed A), `M_na` (no RANDOM-CB result), `M_incomplete` "not
  assessed (no registered RANDOM range)" (the original RANDOM is missing or
  finished after the freeze, or fewer than two seed indices are complete; P
  and B are shown as values without a description and no RANDOM range is
  shown). Any other label is rejected by the gate.
- Takeaways: the original-results bullets are unchanged. With results loaded
  the third bullet ends after "move together." and the replication takeaway
  states only the generated labels ("Registered replication labels: CENTROID
  `\RepLabelCentroid`, ENVELOPE `\RepLabelEnvelope`."), with no digits and no
  claim beyond the labels; the Answer box says the same ("Registered
  replication: CENTROID ..., ENVELOPE ...."), both checked in the PDF.
- The RESULTS PENDING banner, the grey title and the TBD styling disappear
  when results are loaded; the block title becomes "Registered replication:
  repeated continuations and matched-budget control". The label rule stays
  (banner before, outcome strip after).
- Seed-spread panel: the validated figure, else a FIGURE PENDING box of the
  same size, with the caption "Error bars: s.d. over probe-head seeds.
  Letters: seed index (O: original). Open markers: descriptive runs (late or
  incomplete). (b): guided minus RANDOM, same seed." Figure plus caption take
  the height of the table panels plus the outcome strip, so the figure never
  makes the block taller.

### Synthetic guard

The generator marks dry runs with `% SYNTHETIC DRY RUN` in `cr_results.tex`
and `"synthetic": true` in `cr_results.values.json` and
`fig_cr_seed_spread.evidence.json`; the gate requires all three to agree.
With synthetic results:

- the poster shows a yellow "SYNTHETIC DRY RUN --- NOT RESULTS" banner across
  the top, a diagonal watermark, a vermillion block title "SYNTHETIC DRY RUN,
  NOT RESULTS: registered replication", and the outcome strip and takeaway are
  prefixed with "SYNTHETIC DRY RUN" (overlays do not change the layout);
- `build.ps1` stops before compiling and `make_preview.py` exits 1, unless
  `-AllowSynthetic` is given;
- `-AllowSynthetic` is refused (fail closed, git not needed) if this folder,
  the results folder or any existing item in this folder or its
  `build_logs`/`generated` folders (outputs, logs, links) has a `.git`
  directory or `.git` file (linked worktree, submodule) in itself or any
  parent, checked on the lexical path and on the physical path with every
  junction, symbolic link and mount point resolved by Windows
  (`GetFinalPathNameByHandle`, also in Windows PowerShell 5.1); also if
  `GIT_DIR`/`GIT_WORK_TREE` is set or a path cannot be resolved or a parent
  cannot be inspected; so synthetic posters can only be built in a scratch
  copy outside any repository;
- `make_preview.py` also checks that the PDF text contains the banner when
  (and only when) the results are synthetic.

Scratch dry run (as used for the fit test): copy this folder to
`<scratch>\poster\genai4health2026\`, and create
`<scratch>\paper\genai4health2026\auto\` (`auto_numbers.tex`,
`cr_numbers.tex` and the generator's synthetic output folder),
`<scratch>\autopilot\release_assets.json` and
`<scratch>\autopilot\investigations\camera_ready_20261008\figures_v2\poster\`
(the two `*_v2_poster.pdf` files); then run
`build.ps1 -OnlyCached -AllowSynthetic` in the scratch poster folder (set
`$env:POSTER_GENERATOR_DIR` to the repository's `autopilot` folder to include
the generator's own metadata check).

### Tests

```powershell
# results gate: baseline passes (plotted series = sidecar AUC/mean macros, panel (b) =
# sidecar paired differences with the band only with the registered reference, encoding
# read from the evidence; registered RANDOM reference, control label and shown range =
# sidecar), 28 tampered variants fail (V7 and V9 counterexamples included) plus 5 or 6
# registered-reference variants that depend on the bundle (complete reference: marked
# late, range moved, \RepRandomRange dropped; original RANDOM missing/late or < 2 complete
# indices: the registered label a partial range would give, reasons cleared, range macro
# or registered label added; both: reason lists disagree, reason swapped) plus, with a
# current generator, 4 panel (b) variants (extra point, changed difference, forged band,
# extra mean), 3 encoding variants (unknown contract, unknown field, malformed colour) and
# 2 or 3 late-original variants (with a late original: dropped from the evidence, encoding
# entry removed, table mark removed and rehashed; without: invented in the evidence, table
# mark added and rehashed); relative paths are fine
$env:POSTER_GENERATOR_DIR = '<repo>\autopilot'   # optional: also run the generator's check
D:\jepa_phase0\.venv\Scripts\python.exe tests\test_results_gate.py `
  --bundle <synthetic auto folder> --other-bundle <another scenario> --work <scratch dir>
# repository guard, 17 cases x {pwsh, Windows PowerShell 5.1} child builds: git init trees,
# .git file, parent repo, no git on PATH, GIT_DIR, junction/symlink aliases into a repo,
# junctioned results or output folder, symlinked output file (skipped if symlinks are not
# permitted); the launcher itself runs in either shell
pwsh -NoProfile -File tests\test_build_guard.ps1 -Work <scratch dir>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests\test_build_guard.ps1 -Work <scratch dir>
```

Both write only under the given scratch directory and never compile (links are
removed as links before the guard test deletes its directory); the bundle must
be synthetic (the gate test refuses real results).

## Poster size

Set `\providecommand{\PosterSize}{...}` in `poster_size.tex`:

| value | print size | design canvas | body text, A0-equivalent |
|---|---|---|---|
| `neurips-ws` (default) | 24 x 36 in portrait | 841 x 1261.5 mm | 29.9 pt |
| `a0` | 841 x 1189 mm | 892 x 1261.5 mm | 28.1 pt |
| `a1` | 594 x 841 mm | 892 x 1261.5 mm | 28.2 pt |
| `us-36x48` | 36 x 48 in portrait | 946 x 1261.5 mm | 26.5 pt |

The design canvas is 841 mm (A0) wide for the 2:3 workshop format, so font
sizes are A0-equivalent: block body 29.9 pt, tables and notes 24.9 pt, block
titles 43 pt, title 84 pt. Shorter formats keep the 1261.5 mm canvas height
and gain width, so the same content fits every size; figures keep their
height. `poster_print.tex` scales the canvas to the print size. All four sizes
were built on 2026-10-08 without overflow or Overfull boxes; on 2026-10-12
(new layout) the default and the `one_index` synthetic scenario were rebuilt
at `a0`, `a1` and `us-36x48` (no overflow, 0 Overfull, at least 73 pt spare).
Landscape formats need a different column layout and are not supported.

### Sources (checked 2026-10-08; NeurIPS 2026 page rechecked 2026-10-11)

- NeurIPS 2026 poster instructions,
  <https://neurips.cc/Conferences/2026/PosterInstructions>: on 2026-10-08 and
  again on 2026-10-11 the page read only "This page has not yet been updated
  to 2026"; no 2026 workshop poster size has been published there.
- NeurIPS 2025 poster instructions,
  <https://neurips.cc/Conferences/2025/PosterInstructions>: workshop posters
  "24W x 36H inches portrait layout" (San Diego) and "24W x 36H inches"
  (Mexico City), lightweight paper, taped to the wall with command strips;
  main-conference boards are 96 x 48 in.
- NeurIPS 2024 poster information,
  <https://neurips.cc/Conferences/2024/PosterInstructions>: workshop posters
  "24W x 36H inches", lightweight paper, painter's tape; optional ARC printing
  recommends 100 dpi at poster size (pdf, jpg, tiff, png).
- GenAI4Health 2026, <https://genai4health.github.io/2026-NeurIPS/> (and its
  source repository `genai4health/2026-NeurIPS`): "All accepted papers will be
  presented with posters"; the schedule lists an "Interactive Poster Session";
  no size or format is given (rechecked 2026-10-11). The 2024 and 2025
  workshop sites (`genai4health/2024-NeurIPS`, `genai4health/2025-NeurIPS`)
  give no size either.
- Not used: third-party poster-template sites claiming that Sydney venues
  expect A0; no official source was found.

Default: 24 x 36 in portrait, the size NeurIPS asked of workshop posters in
2024 and in both 2025 venues. If the 2026 page or the organizers specify
another size (A0 is common at Australian venues), change `poster_size.tex`
and rebuild.

## What to fill in later

1. **Authors and affiliations**: done (2026-10-10). `\PosterAuthors` and
   `\PosterAffiliations` in `poster.tex` carry the same names and order as
   `paper/genai4health2026/camera_ready_authors.json`.
2. **QR code: pending a decision by the authors.** It should point to the
   public paper together with the code and the pre-registration
   (`PREREGISTRATION_cr_seed_v1.md`, R1 M15: "registered" must be
   verifiable); the placeholder says "paper, code & pre-registration". Set
   `\providecommand*{\PosterURL}{...}` in `poster.tex` to the chosen URL
   (avoid `#` and `%`). An empty value draws the "QR CODE PENDING" box; a URL
   draws a real code (tested).
3. **Registered replication, matched-budget control, pooling, seed spread**:
   automatic, nothing to edit in `poster.tex`. After the Oct 22 freeze, once
   `autopilot/make_cr_results.py build` has written the real results to
   `paper/genai4health2026/auto/` (with its sidecar, figure and evidence),
   rebuild and check the report: `Precheck ... real; validated ...;
   generator check passed`, the seed indices and descriptive seeds you
   expect, the matched-control label with P/B on-time or late,
   `seed-spread figure: poster render of the validated evidence`, `Cells ...
   match the sidecar`, `Placeholders` listing only the QR code (if still
   pending) and `Checks OK`. Look at the preview: the outcome strip shows the
   registered labels exactly as in the paper. Do not copy synthetic dry-run
   outputs into the paper's `auto/` folder (the build refuses them).
4. **Poster size** once the 2026 guidance appears (see above; not published as
   of 2026-10-11).
5. **Acknowledgments** do not apply to the poster (the paper's
   acknowledgments are not reproduced; the credits line covers the cited
   methods, data and the licensed OCT example).
6. Rebuild with `build.ps1` and check `Overfull boxes 0` and `Checks OK`.

## Content rules

- OCT pixels come only from the two I13 v2 figures built on the CC BY 4.0
  OCTDL crop (Kulyabin et al., arXiv:2312.08255v3, Fig. 1), attributed in the
  figures and in the credits line ("adapted (cropped, downsampled, overlays
  added)"). No FairVision images are shown. The only other figure is the
  seed-spread plot (no images): the poster render of the validated generator
  evidence (or the generator's own PDF). `make_preview.py` fails if any other
  figure file is included, or if a v2 figure is not the registered poster
  variant.
- Colours: Okabe-Ito palette; white text only on dark blue, navy or grey
  (contrast at least 5:1); black text on the vermillion synthetic title (about
  5.4:1) and on the yellow synthetic banner; placeholders use text ("TBD",
  "PENDING") as well as colour.
- Minimal text; every caveat on the poster is taken from the paper. The
  replication block prints only generated macros and the registered label and
  description names; it adds no claim of its own.

## Known limitations

- Text inside the two v2 figures (I13's poster variants, labels at least
  14 pt at their 12 in width): the strategies figure is shown at 348 x 201 mm
  (height cap 200 mm, scale 1.14, so at least about 16 pt A0-equivalent, 11.6
  pt printed at 24 x 36 in) and the framework figure at 322 x 217 mm (height
  cap 216 mm, scale 1.06, at least about 14.8 pt A0-equivalent, 10.7 pt
  printed). Full width would need about 20 mm more in the left column, which
  the fit does not allow; body text is 29.9 pt A0-equivalent.
- Seed-spread figure: the poster renders panels (a) and (b) from the
  sidecar-built series at 8.1 x 3.4 in with 14 pt text (13 pt arm names, 12 pt
  seed letters), shown at about 1.0 scale (about 10 pt printed at
  24 x 36 in, against 7 pt in the paper's figure at 5.5 in); panel (b) made
  the text 1 to 2 pt smaller than the earlier single-panel render. The render
  repeats the generator's plotting conventions through the validated encoding
  contract (`cr_seed_spread_encoding_v2`); a new contract is refused until
  `render_poster_figure` in `make_preview.py` is updated.
- Spare space at 24 x 36 in (design canvas, layout of 2026-10-12): 90.3 pt
  (32 mm) with placeholders; at least 11.6 pt (about 4.1 mm) in every fresh
  synthetic replication scenario tested (17 bundles from the generator of
  2026-10-12 01:00: 12.2 pt with a descriptive seed C and three complete arms,
  16.6 to 17.2 pt with a descriptive seed or long labels, 36.1 pt when no
  registered RANDOM reference exists, 41.6 to 42.2 pt otherwise; late
  originals, generator of 03:00: 11.6 pt with the longest labels, a
  descriptive seed A, a late budget contrast and a late CENTROID original,
  17.2 pt with a late CENTROID original, 36.1 pt with a late RANDOM original
  and a late seed C)
  (two or three seed indices, late or missing runs, every control state,
  late placement or budget contrasts, missing or late original RANDOM, the
  longest labels; real results have no synthetic prefixes and at least as much
  space). The outcome strip ends with the label rule (2 lines). The block
  spacing is 8 mm. A longer title, author list or table would need trimming
  (the fit check reports it).
- The sidecar `cr_results.values.json` is the root of trust: the gate checks
  that every other generated file, macro value, label and figure point agrees
  with it and that its displays and control label follow from its own full
  precision values and P/B/range/eligibility, but it does not replay the
  analysis from the probe evidence (that is the generator's `--check` and the
  paper's review).
