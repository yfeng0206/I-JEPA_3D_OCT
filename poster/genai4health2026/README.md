# GenAI4Health @ NeurIPS 2026 poster (draft)

Poster for the accepted workshop paper "Where to Predict in Retinal OCT?
Anatomy-Guided Target Selection for I-JEPA" (GenAI for Health Workshop,
NeurIPS 2026, Sydney, Dec 12 2026; every accepted paper presents a poster).

Status: draft. Authors, affiliations, the QR link and the registered
replication results are placeholders (see "What to fill in later").

## Files

| file | role |
|---|---|
| `poster_size.tex` | the single size parameter `\PosterSize` |
| `poster_setup.tex` | maps `\PosterSize` to the print size and the design canvas |
| `poster.tex` | design master (tikzposter); produces `poster.pdf` |
| `poster_print.tex` | scales `poster.pdf` to the exact print size; produces `poster_print.pdf` |
| `build.ps1` | builds both PDFs with Tectonic, then writes the PNG preview and a build report |
| `make_preview.py` | headless PNG render (PyMuPDF) and checks: page count, size, Overfull boxes, fit, placeholders, figure allow-list |
| `poster_print.pdf` | file to print (exact physical size) |
| `poster_preview.png` | low-resolution preview (50 dpi by default) |

The poster reads the paper's files in place (no copies):
`../../paper/genai4health2026/figures/fig_oct_jepa_comparison_final_cr.pdf`,
`../../paper/genai4health2026/figures/fig_oct_all_strategies.pdf`,
`../../paper/genai4health2026/auto/auto_numbers.tex` (Table 1 macros) and
`../../paper/genai4health2026/auto/cr_numbers.tex` (delivered-mask, guide
agreement and reference macros). Every number on the poster is one of these
macros; do not type result numbers into `poster.tex`.

## Build

```powershell
cd poster\genai4health2026
pwsh -NoProfile -File .\build.ps1                 # size from poster_size.tex
pwsh -NoProfile -File .\build.ps1 -Size a0        # one-off size override
pwsh -NoProfile -File .\build.ps1 -OnlyCached     # offline (after one online build)
pwsh -NoProfile -File .\build.ps1 -Dpi 100        # larger PNG
```

- Tectonic: `D:\jepa_phase0\tools\tectonic\tectonic.exe` (override with
  `$env:TECTONIC`). It runs at Idle priority, one compile at a time (the script
  refuses to start if another `tectonic` process is running). All packages
  (tikzposter, FiraSans, unicode-math with Fira Math, pgfplots, qrcode,
  pdfpages) are in the Tectonic bundle and are cached after the first build.
- Other TeX installations: compile `poster.tex` and then `poster_print.tex`
  with XeLaTeX or LuaLaTeX (fontspec/unicode-math; pdfLaTeX will not work),
  from this folder so the `../../paper/...` paths resolve.
- `build.ps1 -Size X` writes temporary job files `poster-X.tex` and
  `poster_print-X.tex`, builds `poster-X.pdf`, `poster_print-X.pdf` and
  `poster-X_preview.png`, then deletes the job files.
- The report ends with `Checks OK` or `PROBLEMS ...` (exit code 1): more than
  one page, content running past the bottom margin (tikzposter does not warn
  about this itself) or a figure outside the allow-list.

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
were built on 2026-10-08 without overflow or Overfull boxes. Landscape formats
need a different column layout and are not supported.

### Sources (checked 2026-10-08)

- NeurIPS 2025 poster instructions,
  <https://neurips.cc/Conferences/2025/PosterInstructions>: workshop posters
  "24W x 36H inches portrait layout" (San Diego) and "24W x 36H inches"
  (Mexico City), lightweight paper, taped to the wall with command strips;
  main-conference boards are 96 x 48 in.
- NeurIPS 2024 poster information,
  <https://neurips.cc/Conferences/2024/PosterInstructions>: workshop posters
  "24W x 36H inches", lightweight paper, painter's tape; optional ARC printing
  recommends 100 dpi at poster size (pdf, jpg, tiff, png).
- NeurIPS 2026 poster instructions,
  <https://neurips.cc/Conferences/2026/PosterInstructions>: "This page has not
  yet been updated to 2026".
- GenAI4Health 2026, <https://genai4health.github.io/2026-NeurIPS/> (and its
  source repository `genai4health/2026-NeurIPS`): "All accepted papers will be
  presented with posters"; the schedule lists an "Interactive Poster Session";
  no size or format is given. The 2024 and 2025 workshop sites
  (`genai4health/2024-NeurIPS`, `genai4health/2025-NeurIPS`) give no size
  either.
- Not used: third-party poster-template sites claiming that Sydney venues
  expect A0; no official source was found.

Default: 24 x 36 in portrait, the size NeurIPS asked of workshop posters in
2024 and in both 2025 venues. If the 2026 page or the organizers specify
another size (A0 is common at Australian venues), change `poster_size.tex`
and rebuild.

## What to fill in later

1. **Authors and affiliations** (pending from the authors; same placeholder
   approach as the paper). In `poster.tex`, set `\PosterAuthors` and
   `\PosterAffiliations` to the OpenReview author line and affiliations (use
   the same names as `paper/genai4health2026/camera_ready_authors.json`), then
   delete `\CRAuthorsPending` and `\CRAffilPending`.
2. **QR code.** Set `\providecommand*{\PosterURL}{...}` in `poster.tex` to the
   final OpenReview or repository URL (avoid `#` and `%`). An empty value draws
   the "QR CODE PENDING" box; a URL draws a real code (tested).
3. **Registered replication, matched-budget control, pooling** (after the
   Oct 22 results freeze). When `paper/genai4health2026/auto/cr_results.tex`
   exists, `poster.tex` inputs it automatically (log line
   `POSTER: cr_results.tex found and loaded`). Then, in the grey block
   "Registered replication":
   - replace each `\CRCell` with the corresponding generated macro, using the
     same values as the paper's replication table and paragraphs (repeated
     continuations: original, new seeds, mean, difference from RANDOM;
     matched-budget control: CENTROID minus RANDOM-CB and RANDOM-CB minus
     RANDOM; pooling: differences from RANDOM for the two max-pooling probes);
   - adjust the column headers if the paper's final table differs (for
     example one column per seed);
   - remove `\PosterPendingBanner{...}`, the words "results pending" from the
     block title and the `{\colorlet{blocktitlebgcolor}{PendingTitle} ...}`
     wrapper so the block uses the normal colour;
   - revise "Takeaways" if the paper's result sentences (currently
     `\CRSeedResultsPending`) change the conclusions.
4. **Poster size** once the 2026 guidance appears (see above).
5. Rebuild and check that the report shows `Placeholders none`,
   `Overfull boxes 0` and `Checks OK`.

## Content rules

- OCT pixels come only from the paper figures built on the CC BY 4.0 OCTDL
  crop (Kulyabin et al., arXiv:2312.08255v3, Fig. 1), attributed in the
  figures and in the credits line. No FairVision images are shown;
  `make_preview.py` fails if any other figure file is included.
- Colours: Okabe-Ito palette; white text only on dark blue, navy or grey
  (contrast at least 5:1); placeholders use text ("TBD", "PENDING") as well as
  colour.
- Minimal text; every caveat on the poster is taken from the paper.

## Known limitations

- Text inside the two reused paper figures is about 13 to 20 pt
  A0-equivalent (a few labels about 10 pt; measured from the PDF), because
  the figures are used unchanged; poster-specific renders with larger labels
  would need the figure producers to be rerun. All poster text is at least
  24.9 pt A0-equivalent (sub- and superscripts excepted).
- The framework figure inherits a slight overlap of the subscript in
  "p_C, p_T_m" with "+ mask" from the paper figure.
- At 24 x 36 in, 24 mm of the design canvas (about 17 mm printed) is spare at
  the bottom; long author lists or larger result tables may need trimming (the
  fit check reports it).
