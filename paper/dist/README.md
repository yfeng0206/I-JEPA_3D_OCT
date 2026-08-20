# Built paper bundles

These are committed so they can be downloaded from any machine without
rebuilding. Nothing here is generated at clone time.

## Download without cloning

Browse the folder:

    https://github.com/yfeng0206/I-JEPA_3D_OCT/tree/docs/background-signal-findings/paper/dist

Direct download of the current Overleaf bundle:

    https://github.com/yfeng0206/I-JEPA_3D_OCT/raw/docs/background-signal-findings/paper/dist/OCT_JEPA_GenAI4Health2026_Overleaf.zip

Or with curl:

    curl -L -o OCT_JEPA_GenAI4Health2026_Overleaf.zip \
      https://github.com/yfeng0206/I-JEPA_3D_OCT/raw/docs/background-signal-findings/paper/dist/OCT_JEPA_GenAI4Health2026_Overleaf.zip

## Manifest

### Current

| File | Bytes | Entries | Built (local) | SHA-256 |
|---|---:|---:|---|---|
| `OCT_JEPA_GenAI4Health2026_Overleaf.zip` | 5,530,348 | 57 | 2026-08-19 23:12:34 | `2C25C0DA2F84E1A36C8EEC37E224CCBEA580E2DCD072B623F72587D443852F7F` |
| `OCT_JEPA_3D_CVPR_2027.zip` | 37,907 | 16 | 2026-08-19 18:17:54 | `96B10CA5123555B6BDF39C6C7C5A9AA62AD715DE003E8A4B87BFFA239EB0375F` |

`OCT_JEPA_GenAI4Health2026_Overleaf.zip` is the authoritative GenAI4Health
2026 submission bundle. Upload it straight to Overleaf via
New Project -> Upload Project.

It contains `main.tex`, `neurips_2026.sty`, `references.bib`, `main.bbl`,
24 files under `figures/`, 6 under `scripts/`, the `research/` verification
records, and `README.md` / `TODO.md` / `EVIDENCE.md` / `GAPS.md`. The compiled
PDF is included as `PREVIEW_main.pdf` rather than `main.pdf` so that Overleaf
does not mistake it for build output and refuse to overwrite it. No
`__pycache__`, `.pyc` or `.bak` files are present.

The bundle compiles clean under tectonic 0.17.0 with zero undefined
references and zero undefined citations. Main body ends on page 9, within the
9-page Research-track limit; the appendix begins on page 10.

`OCT_JEPA_3D_CVPR_2027.zip` is an unrelated early skeleton for a separate
CVPR 2027 write-up. It is kept here only so it is not lost.

### Superseded

`archive/` holds earlier builds of the same GenAI4Health paper, retained for
provenance. Do not submit these - they predate the verified-number pass.

| File | Bytes | Entries | Built (local) | SHA-256 |
|---|---:|---:|---|---|
| `archive/2026-08-19_2128_genai4health2026_overleaf.SUPERSEDED.zip` | 4,287,642 | 38 | 2026-08-19 21:28:44 | `B6870B41FCC4BC5640FC4495DFA86610F5D76006479DEB27E28FEFE4A1DB22E6` |
| `archive/2026-08-19_1809_overleaf_genai4health2026.SUPERSEDED.zip` | 3,077,806 | 17 | 2026-08-19 18:09:58 | `3C8941B5453AD69B08CCDB262953A0EAC71C1A8CB0AD03C46279639C19832F32` |

## Verifying a download

PowerShell:

    Get-FileHash .\OCT_JEPA_GenAI4Health2026_Overleaf.zip -Algorithm SHA256

Linux / macOS:

    sha256sum OCT_JEPA_GenAI4Health2026_Overleaf.zip

Compare against the table above.

## Rebuilding

From `paper/genai4health2026/`:

    D:\jepa_phase0\tools\tectonic\tectonic.exe -X compile main.tex --keep-intermediates --keep-logs

Both flags are required: `--keep-intermediates` produces the `.aux` used to
confirm the page count, and `--keep-logs` produces the `.log` checked for
undefined references. Verify the main body still ends on page 9 by reading the
`endofmain` entry in `main.aux` rather than by counting pages in the PDF,
which is unreliable.

After any change to `main.tex`, `figures/` or `scripts/`, rebuild the bundle
and update both the file and its hash in this manifest.
