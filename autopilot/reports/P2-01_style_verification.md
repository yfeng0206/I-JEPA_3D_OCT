# P2-01 Style-file verification

Agent: `sty-verify` (gpt-5.6-sol, xhigh) | agent_id `f2878f18-8262-46ca-b712-2a9c02276f39`
Completed 2026-08-22, elapsed 3964 s. Report transcribed by the coordinator because
the agent's execution policy blocked file writes.

## VERDICT: IDENTICAL

The official NeurIPS 2026 `neurips_2026.sty` and our local copy are byte-for-byte
identical.

| property | value |
|---|---|
| size | 13,704 bytes |
| lines | 443 |
| differing lines | **0** (`difflib.unified_diff` emitted nothing) |
| SHA-256 (both files) | `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a` |
| official ZIP SHA-256 | `82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9` |

**Recommendation: keep the local file. Replacement would change nothing.**

## Evidence trail

| URL | HTTP | finding |
|---|---|---|
| `https://neurips.cc/Conferences/2026/CallForPapers` | 200 | official CFP, links the author ZIP |
| `https://neurips.cc/Conferences/2026/MainTrackHandbook` | 200 | 2026 formatting + double-blind instructions |
| `https://neurips.cc/Conferences/2026/PaperInformation/StyleFiles` | 404 | no page at this path |
| `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip` | 200 | **official ZIP**, 20,259 bytes, contains `neurips_2026.sty`, `neurips_2026.tex`, `checklist.tex` |
| `https://media.neurips.cc/Conferences/NeurIPS2026/neurips_2026.sty` | 404 | standalone file not served |
| `https://genai4health.github.io/2026-NeurIPS/` | 200 | requires `\usepackage{neurips_2026}` **with no options** |
| Overleaf NeurIPS 2026 template | 200 | template page exists |

## Package record

```latex
\ProvidesPackage{neurips_2026}[2026-01-29 NeurIPS 2026 submission/camera-ready style file]
```
(`neurips_2026.sty:22`)

Declared options (`:26-96`): `final`, `nonatbib`, `preprint`, `main`, `position`,
`eandd`, `creativeai`, `education`, `sglblindworkshop`, `dblblindworkshop`,
`nonanonymous`.

With **no options** the initial states remain: non-final, non-preprint,
natbib-enabled, and **anonymous** (`:25,34,41,78`). This is exactly what the
GenAI4Health CFP requires.

## Geometry (official vs local: identical on every row)

| property | value |
|---|---|
| paper | US Letter, 8.5 x 11 in |
| `\textwidth` | 5.5 in |
| `\textheight` | 9 in |
| margins | 1.5 in left/right, 1 in top/bottom |
| `headheight` / `headsep` / `footskip` | 12 pt / 25 pt / 30 pt |
| normal font | 10 pt on 10.95 pt baseline (`\@setfontsize\normalsize\@xpt\@xipt`, `:149-157`) |
| roman / sans | `ptm` / `phv` |

Geometry set at `:127-141` via `\newgeometry{textheight=9in, textwidth=5.5in,
top=1in, headheight=12pt, headsep=25pt, footskip=30pt}`.

## Anonymity check

`:77-78`
```latex
\newif\if@anonymous\@anonymoustrue
```

`:336-353`
```latex
\if@anonymous
  \begin{tabular}[t]{c}\bf\rule{\z@}{24\p@}
    Anonymous Author(s) \\
    Affiliation \\
    Address \\
    \texttt{email} \\
  \end{tabular}%
\else
  ...\@author...
\fi
```

`\usepackage{neurips_2026}` with no options **does render anonymously**.
Independently corroborated by the coordinator: the compiled
`main_submission.pdf` page 1 renders "Anonymous Author(s) / Affiliation /
Address / email", and the page media box is 612 x 792 pt (US Letter).

## Notice / watermark / line numbers

In no-option submission mode the package (`:358-365, 390-438`):
- prints the first-page notice "Submitted to 40th Conference on Neural
  Information Processing Systems (NeurIPS 2026). Do not distribute."
- enables `lineno` and calls `\linenumbers`
- hides the `ack` environment
- adds **no** watermark

Identical in the official and local files. Our compiled PDF shows the expected
submission line numbers.

## Consequence for the submission

Desk-reject risk R1 (style mismatch) is **closed**. No action required.
