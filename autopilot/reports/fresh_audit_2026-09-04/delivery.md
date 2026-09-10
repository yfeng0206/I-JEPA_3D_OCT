# Fresh delivery and verification audit

**Audit date:** 2026-09-04  
**Repository:** `C:\Users\Gary\Desktop\jepa`  
**Audited HEAD:** `de145d7005f57e871bc0181bf58b271775d1d25d`  
**Mode:** read-only audit. No production script was run if it could overwrite the paper, PDF, DOCX, Downloads, sync state, or Overleaf. No network or citation API was used.

## Executive conclusion

The current submission ZIP is a real, standalone-compiling artifact, and its current page boundary is nine body pages followed by References at the top of page 10. The current DOCX contains all 16 image payloads, 16 table captions, 15 figure captions, and 47 bibliography entries. Those are useful positive facts.

They do **not** justify the prior broad claims:

- `6/6 PASS` is a narrow packaging result, not a numerical-correctness certificate.
- “Every number is protected” is not established by either numeric checker.
- “47/47 citations real” is a historical assertion from an online run, not a retained per-entry verification record; the verifier has false-success paths and does not test whether a citation supports a claim.
- The DOCX is substantial and editable in Word-compatible software, but its appendix cross-references disagree with the LaTeX `.aux`, it has no References heading, and `check_docx.py` cannot fail.
- `.overleaf_sync.json` has the correct `last_action`/`updated_utc` schema and exact local hash parity, but it records an August 27 agreement. It is not evidence of the remote project state on September 4.

## Current delivery board

| Artifact | Current evidence | Bounded status |
|---|---|---|
| `paper/genai4health2026/main_submission.tex` | 118,070 bytes; SHA-256 `09365d58b2a2843b1f44de5b082992c499a7757c09607a03239469f6bb1b22bf` | Current source input at audited HEAD. |
| `paper/genai4health2026/main_submission.pdf` | 8,681,289 bytes; SHA-256 `5e7f72df571be429c164552f1723980e9471e9c4fd15ed996406390b2a314f8a`; 36 pages | References is the only References heading, page 10 at y=72.8 pt. No Acknowledgments/Acknowledgements heading exists. PDF metadata author/title-identifying fields are blank. |
| `C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_FINAL.zip` | 17,388,596 bytes; SHA-256 `d33f6b2813150718d71b589b1564d2b6daa7ce7e80cf97cd8c1392d2810aa523`; 30 entries | Fresh isolated extraction and Tectonic compilation succeeded: 36 pages, References page 10 at y=72.8, no undefined-reference warning. No required source was missing. |
| `paper/genai4health2026/main_submission.docx` | 8,999,181 bytes; SHA-256 `7610d2c8137a17990381ee79843c9b9d44c73d6156f47ece3ba4fd94eff3a9ee` | Substantial OOXML document, but not cross-reference-equivalent to the LaTeX source; details below. |
| `.overleaf_sync.json` | version 1; `last_action: "push"`; `updated_utc: 2026-08-27T12:48:31+00:00`; 59 files | All 59 current local managed hashes equal the recorded local hashes, and all 59 recorded remote/local pairs are equal. This is historical agreement, not a fresh remote observation. |

The August 27 `D:\jepa_phase0\autopilot_out\zip_validate\validation.json` corresponds to the existing ZIP and records 6/6 and `ALL_PASS: true`. The fresh audit did not treat that record as sufficient: it separately extracted and compiled the ZIP in the authorized isolated session directory.

## What the checks actually establish

| Check | Fresh result | What it proves | What it does not prove |
|---|---:|---|---|
| `check_manuscript.py` | exit 0; 400 macros defined, 0 duplicate/undefined; 47 cited keys, 0 missing; 56 refs, 0 dangling | Selected macro syntax, citation-key presence, label presence, selected banned phrases, and two artifact count checks. | Correct values, complete macro coverage, citation existence/support, or that all rendered numbers are generated. Hand-typed result-like values are warnings only and use a narrow regex (`autopilot/check_manuscript.py:83-97`). |
| `p15_verify_numbers.py` | exit 0; **20** AUC macros checked, 1 skipped | For the 20 recognized `AUC<known-arm><known-epoch>` numeric macros, the rounded value matches at least one primary inventory record for that arm/epoch. | The other generated macros, deltas, CIs, p/q values, counts, tables, figures, literals, or a minimum expected coverage count. A zero-check run passes. |
| `p13_build_zip.py` historical 6/6 | current ZIP independently compiles | Standalone compilation, its page heuristic, its log regex, its limited text anonymity scan, its placeholder scan, and referenced-graphics presence all returned true on the August 27 build. | Numerical correctness, DOCX correctness, citation existence/support, exact pushed Overleaf tree, or robust References/acknowledgments boundary detection. |
| `verify_citations.py` offline | exit 0; 47 cited, 67 bib; 38/47 DOI/arXiv identifiers; 9 “manual eyeball” | Citation-key extraction and identifier availability. | Existence of the nine no-ID entries, title correctness, claim support, or appropriateness. |
| `verify_punctuation_only.py` | exit 0; zero changed lines; “PUNCTUATION ONLY” | The current working manuscript equals `HEAD:paper/.../main_submission.tex`. | The historical semicolon edit was punctuation-only. The comparison is now vacuous and the script never returns nonzero. |
| `check_docx.py` | exit 0; six hard-coded snippets found; 16 media; no leaked selected macro names | A few sentinel strings and rough OOXML counts exist. | Completeness, correct tables/figures/references/cross-references, or even success: it only prints `MISSING`. |

## Findings

### P0 — The build path is not fail-closed and can replace a good deliverable with a failed one

**Evidence**

- `refresh_all.run()` defaults `quiet_ok=True` and raises only when callers explicitly pass false (`autopilot/refresh_all.py:36-48`). Every pipeline call, including `check_manuscript.py`, `p15_verify_numbers.py`, and the first compilation, uses the permissive default (`autopilot/refresh_all.py:59-84`).
- The final invocation also passes `--allow-placeholders` (`autopilot/refresh_all.py:83-84`).
- `p13_build_zip.py` computes `ok` at line 169, but then unconditionally deletes the requested output and writes a ZIP at lines 171-186. It copies the newly compiled PDF into the repository whenever compilation succeeded, even if another gate failed (`autopilot/p13_build_zip.py:214-225`). Only the loose mirror is conditional on `ok` (`autopilot/p13_build_zip.py:232`).
- `refresh_all.py` prints `REFRESH COMPLETE` even when the final return code is nonzero (`autopilot/refresh_all.py:86-89`).

**Minimal isolated reproduction**

1. A synthetic child returning 7 was passed to `refresh_all.run()` with defaults. It printed `rc=7`, returned 7, and did not raise.
2. `p13.build()` was pointed at an isolated synthetic paper, its compiler was forced to fail, and its output path first contained a sentinel representing an existing good artifact. The function returned 1 and printed `ALL_PASS = False`, but the sentinel was deleted and replaced by a new ZIP containing `main.tex`, the style, bibliography, and README.

**Impact**

A failed run can leave a newly timestamped, plausibly named ZIP and/or PDF at the publication locations. A human who sees the file but not the exit code can distribute a failed artifact. Earlier hard checks can fail without preventing later publication.

**Minimal remedy**

Make `run()` fail by default; explicitly mark only genuinely optional steps as nonfatal. Build into a unique staging ZIP/PDF, and atomically replace the requested output and repository PDF only after every gate passes. Remove `--allow-placeholders` from release refreshes.

**Confidence:** high.

### P1 — The sync safety claim validates one tree but pushes a larger, independently collected tree

**Evidence**

`sync_overleaf.py` says it pushes “the SAME staged tree” and that the remote is byte-identical to what passed the gates (`scripts/sync_overleaf.py:13-14`). It does not:

- p13 copies all of `auto/` but only figures referenced by `main_submission.tex` (`autopilot/p13_build_zip.py:68-83`).
- Sync independently includes the DOCX plus **all** files below `auto/` and `figures/` (`scripts/sync_overleaf.py:128-138`, `scripts/sync_overleaf.py:272-289`).
- The current sync state covers 59 files. The current ZIP contains 28 of those managed source files. The remaining 31 are `main_editable.docx` plus 30 unused figure variants.
- Sync runs p13 validation, then warns but does not block on DOCX staleness (`scripts/sync_overleaf.py:357-390`, `scripts/sync_overleaf.py:677-679`), and then pushes the independently collected set.

**Impact**

The compiled source tree is currently sound, but the statement that every pushed byte was gated is false. Unused figures and the Word binary are not covered by p13’s compile, anonymity, placeholder, or completeness checks.

**Minimal remedy**

Have p13 emit a manifest of the exact validated stage, and make sync consume only that manifest. If collaborator files must also be uploaded, validate them separately and label them explicitly as nongated binary attachments.

**Confidence:** high.

### P1 — “6/6” and the number checks do not establish numerical correctness or complete macro protection

**Evidence**

- None of p13’s six checks is numerical (`autopilot/p13_build_zip.py:160-168`).
- p15 selects only names matching `^AUC(<six known arms>)(<listed epochs>)$` (`autopilot/p15_verify_numbers.py:54-61`). Nonmatching macros are ignored, placeholders/wrapped/non-float values are skipped (`autopilot/p15_verify_numbers.py:63-80`), and there is no expected-count assertion before PASS (`autopilot/p15_verify_numbers.py:90-106`).
- Current p15 output was 20 checked out of 400 generated macro definitions; this is useful but narrow.
- An isolated `auto_numbers.tex` with no matching AUC macro produced “AUC macros verified: 0” and `RESULT: PASS`.
- `check_manuscript.py` searches only two numeric shapes before Discussion/appendix and reports them as warnings (`autopilot/check_manuscript.py:83-97`).
- Even a genuine p15/check-manuscript failure would currently be ignored by `refresh_all.py`.

**Impact**

The prior phrase “all numbers macro-protected” is stronger than the executed assertions. The checks neither prove that every displayed quantity is a macro nor trace every macro to its source statistic.

**Minimal remedy**

Generate a typed value manifest from the asset generator: every rendered numeric macro, source artifact/key, expected formatting, and every manuscript/table/figure consumer. Fail on missing, extra, skipped, unused-but-published, or literal result values, and require a nonzero expected coverage count.

**Confidence:** high.

### P1 — Placeholder and anonymity gates have concrete blind spots

**Evidence**

- The placeholder resolver scans placeholder macros in `auto_numbers.tex` but considers them used only if the macro name appears directly in `main_submission.tex` (`autopilot/p13_build_zip.py:100-114`). A normal use through `\input{auto/table_main.tex}` is invisible.
- Isolated reproduction: `main_submission.tex` input an auto table; that table used `\Hidden`; `auto_numbers.tex` defined `\Hidden` as `\ph{TODO}`. p13 reported `unresolved_placeholders: []` and the placeholder check passed.
- Release refresh explicitly bypasses any detected placeholder via `--allow-placeholders` (`autopilot/refresh_all.py:83-84`).
- The anonymity scan joins text only through the first page containing “References” (`autopilot/p13_build_zip.py:123-140`). The appendix begins after the bibliography (`paper/genai4health2026/main_submission.tex:770-775`), so appendix text and independently pushed extra files are outside the scan.

**Current-artifact bound**

The current source search found only the definition of `\ph`, not a use, and the current PDF metadata has blank author fields. No current identifying URL from p13’s list was found. These blind spots are gate defects, not evidence that this particular PDF is identifying.

**Minimal remedy**

Flatten all `\input` files before placeholder reachability analysis, never allow placeholders in a release target, scan every PDF page plus all uploaded text/OOXML metadata, and separate generic surname collision checks from exact project/account identifiers.

**Confidence:** high for the blind spots; high that the current PDF metadata is clean.

### P1 — The DOCX is materially complete but has wrong appendix references, no References heading, and no enforcing checker

**Fresh OOXML inventory**

- 1,159 paragraphs, 17 `<w:tbl>` elements.
- 16 `TableCaption` paragraphs versus 16 LaTeX table environments.
- 15 `ImageCaption` paragraphs versus 15 LaTeX figure environments.
- 16 drawings/media: the extra image is Figure 1’s second subpanel. Every embedded media byte hash matches a current local PNG. In document order the first two are exactly `fig_pipeline_schematic.png` and `fig1_policies_compact.png`, so both Figure 1 subpanels survived.
- 47 `Bibliography` paragraphs, matching the 47 cited keys.
- OOXML core creator is empty.

**Defects**

1. The LaTeX `.aux` assigns appendices letters, e.g. A, B, D, E, F, G, and P (`paper/genai4health2026/main_submission.aux:169-217`, `:278`). The DOCX contains 36 Appendix references rendered as numeric sections 8–23 and 12.1. None of the 16 unique expected lettered references A–P/E.1 appears. Example: source “Appendix A” becomes “Appendix 8”.
2. The 47 bibliography paragraphs have no preceding `References` heading; the paragraph immediately before the first bibliography item is ordinary table content.
3. `check_docx.py` contains six hard-coded string sentinels (`autopilot/check_docx.py:34-44`) and only prints diagnostics. It has no return or `sys.exit`. An isolated DOCX containing only “garbage only” printed six `MISSING` lines and exited 0.
4. `refresh_all.py` calls neither `make_docx.py` nor `check_docx.py`. Sync’s staleness check compares only DOCX mtime against the `.tex` mtime and is nonblocking (`scripts/sync_overleaf.py:372-390`); regenerated auto tables, figures, or bibliography can therefore stale the DOCX without triggering the warning.

**Version nuance**

The DOCX banner says it was generated from commit `e785668` at 2026-08-27 05:46. Diffing that commit to HEAD shows no change to `main_submission.tex`, bibliography, or current generated source payload; only the pipeline schematic/DOCX were added and the PDF republished. The banner commit is older than HEAD, but there is no evidence of current prose/source staleness.

**“Editable in Overleaf”**

The file is valid OOXML and is editable in Word-compatible applications. The sync code maps it to `main_editable.docx` and explicitly says Word edits must be carried back by hand (`scripts/sync_overleaf.py:128-136`). Uploading it to an Overleaf project establishes a downloadable binary attachment, not an editable Overleaf manuscript source.

**Minimal remedy**

Teach the conversion to preserve `\appendix` letter numbering and add an explicit References heading. Make `check_docx.py` compare all source captions/tables/media hashes, bibliography count/heading, and every typed `\ref` against `main_submission.aux`, then return nonzero on any mismatch. Wire DOCX build/check after all source and asset generation.

**Confidence:** high.

### P1 — Citation verification can return success for missing, unchecked, empty, or weakly matched references

**Evidence**

- Missing cited keys are printed but never added to the failure set; offline mode then returns 0 (`autopilot/verify_citations.py:149-173`).
- No-ID/title-search failures increment `skipped`, not `bad`, so online mode can report zero problems and return 0 (`autopilot/verify_citations.py:182-208`).
- Zero cited keys also returns 0.
- DOI/arXiv title matching accepts a title when the set of overlapping words reaches half the BibTeX title length (`autopilot/verify_citations.py:197-200`). Synthetic example: “Deep learning for retinal OCT classification” was accepted as matching “Deep learning for chest X ray classification” because four common words exceeded the threshold of three.
- The algorithm verifies identifier resolution and approximate title agreement only. It never reads the cited paper’s claim context, so it cannot establish evidentiary support or appropriateness.

**Fresh current result**

Offline mode found 47 cited keys, 67 BibTeX entries, 38 DOI/arXiv identifiers, and 9 entries requiring manual/title verification. No live resolution was performed by this audit.

Commit `54dac13` records a historical online assertion of “47 of 47 resolved and title-matched, 0 problems”, but no per-entry response, normalized title pair, resolver timestamp, or immutable result file is retained. Given the false-success paths, that commit message is not independently replayable evidence of 47/47.

**Minimal remedy**

Fail on missing keys, empty citation sets when citations are expected, and every unresolved/skipped entry. Persist one row per citation with key, identifier, returned canonical title, resolver, timestamp, exact-match metric, and status. Use a stricter title metric with stopword removal and token-order/edit similarity. Treat support/appropriateness as a separate manual claim-to-source review.

**Confidence:** high.

### P1 — `refresh_all.py` does not regenerate most required figures or the DOCX

**Evidence**

The refresh documentation says step 6 regenerates “every macro, table and figure” (`autopilot/refresh_all.py:13`). Its only figure producer is `p8_make_assets.py` (`autopilot/refresh_all.py:72`), which writes four current included figures: trajectories, fairness, ROC, and label efficiency (`autopilot/p8_make_assets.py:812-963`).

The manuscript has 15 raster `\includegraphics` inputs: four resolve under `auto/` and eleven resolve under `figures/`. The eleven current required images under `figures/` are not regenerated by refresh:

- `fig1_policies_compact.png`
- `figS5_mask_statistics.png`
- `fig_masking_policies.png`
- five `interp_*.png` images
- `fig_precision_paradox.png`
- `fig_specificity_ladder.png`
- `fig_geometry_panel.png`

Separate, unwired producers exist for `figS5_mask_statistics` (`paper/genai4health2026/scripts/make_story_figures.py:354`), `fig_specificity_ladder` (`autopilot/make_fig_specificity_ladder.py:124`), and `fig_geometry_panel` (`autopilot/make_fig_geometry_panel.py:79`). Repository-wide searches found no producing `savefig`/write code for `fig1_policies_compact`, `fig_masking_policies`, `fig_precision_paradox`, or the five interpretation PNGs; some are only mentioned as externally stored evidence.

**Current-artifact bound**

All required binaries are currently present, packaged, and compile. This is a reproducibility/freshness gap, not a missing-file defect in the existing ZIP.

**Minimal remedy**

Declare every deliverable asset and its producer in one dependency manifest. Wire all available producers before compilation; for externally produced figures, retain the generation script and immutable numeric/image source or mark them as non-regenerable inputs with hashes. Then run `make_docx.py` and an enforcing DOCX check.

**Confidence:** high that refresh omits 11 required images; medium-high that eight lack a local producer of any language, because the search was repository-wide but cannot exclude an undocumented external workflow.

### P2 — The page-limit result is correct for the current PDF, but the heuristic is not robust

**Current geometry**

The body’s conclusion occupies page 9. The only References heading is at page 10, y=72.8 pt, clearly within the script’s “page-topping” branch. There is no acknowledgments heading. Therefore the reported nine body pages is correct for the current artifact.

**Heuristic defects**

- The script selects the first occurrence returned by `search_for("References")` on any page (`autopilot/p13_build_zip.py:132-136`), not a heading with font/size/outline confirmation. A synthetic ten-page PDF with “References are available…” on page 1 and the actual heading on page 10 was classified as one main-content page.
- It uses a hard 95 pt y threshold (`autopilot/p13_build_zip.py:144-151`). Synthetic headings with extracted y=93.97 and 95.97 switched the result by a full page.
- It has no acknowledgments-boundary rule. If acknowledgments are excluded by a target venue and appear before References, they are counted as body; if they are included, no explicit assertion records that policy.

**Minimal remedy**

Locate the bibliography boundary from TeX structure and corroborate it against a uniquely styled PDF heading/outline. Store the venue’s explicit treatment of acknowledgments and appendices as configuration, and fail on ambiguous or multiple candidate headings.

**Confidence:** high.

### P2 — The punctuation verifier and historical sync markers are status aids, not gates

**Punctuation**

`verify_punctuation_only.py` compares `git show HEAD:<path>` to the working file (`autopilot/verify_punctuation_only.py:21-30`). At a clean manuscript it necessarily reports zero changes. It also whitelists any entire changed line containing one of five phrases (`autopilot/verify_punctuation_only.py:23-25`, `:48-54`) and only prints a verdict (`autopilot/verify_punctuation_only.py:86`), so `REVIEW REQUIRED` would still exit 0.

**Sync state**

The schema is correct: `.overleaf_sync.json:2-7` uses `version`, `updated_utc`, `last_action`, and `files`; there is no `last_push` key. `save_state()` writes those fields (`scripts/sync_overleaf.py:252-266`). Current local content hash parity is exact for all 59 entries.

That establishes: “the current local managed files equal the locally recorded August 27 agreement.” It does not establish: “the current Overleaf remote still equals them.” Only a fresh authenticated clone/classification could do that, and this audit was explicitly prohibited from authenticating.

The loose mirror’s `CHANGED.txt` says “since the previous publish”, not since a confirmed upload. No `.uploaded.json` exists, so p13 falls back from the upload baseline to the previous build manifest (`autopilot/p13_build_zip.py:243-257`). “UPLOAD THESE: nothing” therefore proves no change since the previous local build, not current Overleaf parity.

**Minimal remedy**

Give the punctuation verifier an explicit base commit/range and a failing exit status. For remote status, report the state timestamp as historical and require a fresh read-only remote classification before saying “current remote in sync”.

**Confidence:** high.

## Deliverable structure and source coverage

The current ZIP contains:

- `main.tex`, `neurips_2026.sty`, `references.bib`
- all 14 current `auto/` payload files, including the four referenced raster PNGs under `auto/`
- the 11 referenced raster PNGs under `figures/`
- compiled `main.pdf`
- `README_OVERLEAF.txt`

Thus the ZIP has exactly 15 manuscript raster inputs, not 11 total figures: 4 under `auto/` plus 11 under `figures/`. The inline TikZ pipeline panel is compiled from `main.tex`; `make_docx.py` rasterizes that panel, which is why the DOCX has 16 image media objects for the 15 raster inputs plus the TikZ panel.

Fresh isolated Tectonic compilation succeeded without repository access. This is strong evidence that the ZIP has no missing compile-time source. The compile emitted layout warnings and an invalid-UTF-8 warning in `lineno.sty`, but no undefined citation/reference warning and no fatal error.

The ZIP intentionally omits unused figure variants and the DOCX. That is reasonable for a submission ZIP. It is also why the independently collected Overleaf sync tree must not be described as identical to the validated ZIP tree.

## Commands and bounded results

- `git rev-parse HEAD` → `de145d7005f57e871bc0181bf58b271775d1d25d`.
- Read-only check exits: manuscript 0; p15 0; citations-offline 0; punctuation 0; DOCX 0.
- Isolated ZIP compile: Tectonic 0; 36 pages; References page 10 y=72.8; no undefined warning.
- Current DOCX OOXML: 16/16 embedded media hashes matched local PNGs; 16 table captions; 15 image captions; 47 bibliography entries; 36 appendix-reference occurrences use numeric rather than `.aux` letter numbering.
- Sync state: 59/59 current local hashes equal recorded local hashes; 59/59 recorded remote/local pairs equal; state age eight days at audit time.
- Synthetic tests were confined to `C:\Users\Gary\.copilot\session-state\66ccb9a5-db78-4934-afd2-9f61b6d9c57b\files\fresh_audit_20260904_delivery`.

## Scope limitations

- No Overleaf authentication, clone, push, pull, or remote observation was performed.
- No citation network request was made, so the historical 47/47 existence claim was not independently re-resolved.
- PDF inspection was headless text/geometry extraction with PyMuPDF only; no PDF/image was opened on the desktop.
- The manuscript was used only as structural input for ZIP/DOCX/reference comparison, not re-audited for prose or scientific claims.
- External `D:` statistical inventories were read only by the existing read-only checks; their scientific validity belongs to the manuscript/statistics audit.

## Reusable isolated reproductions and timing

No standalone synthetic-driver `.py` file was retained: the reproductions were run as PowerShell here-strings recorded in the session history. The isolated fixtures and outputs were retained under:

`C:\Users\Gary\.copilot\session-state\66ccb9a5-db78-4934-afd2-9f61b6d9c57b\files\fresh_audit_20260904_delivery`

Retained paths:

- Fresh extracted/compiled release ZIP: `...\zip_compile\main.tex`, `main.pdf`, `main.log`, and intermediates.
- Failed-p13 fixture/source: `...\p13_paper`.
- Failed-p13 validation record: `...\p13_scratch\validation.json`.
- ZIP that replaced the pre-existing sentinel despite failure: `...\failed_replaced.zip`.
- Indirect-placeholder output: `...\placeholder_case.zip`; its final validation record is `...\p13_scratch\validation.json`.
- Vacuous-p15 fixture: `...\p15_paper\auto\auto_numbers.tex` and `...\p15_inventory.json`.
- DOCX exit-zero fixture: `...\docx_case\paper\genai4health2026\main_submission.docx`.

Exact reusable refresh default-failure reproduction, which writes no production artifact:

```powershell
@'
import importlib.util, sys
p = r"C:\Users\Gary\Desktop\jepa\autopilot\refresh_all.py"
s = importlib.util.spec_from_file_location("refresh_repro", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
print("returned =", m.run(
    "synthetic failure",
    [sys.executable, "-c", "import sys; sys.exit(7)"],
))
'@ | & 'D:\jepa_phase0\.venv\Scripts\python.exe' -
```

Expected result: the child reports `rc=7`; `run()` returns `7` instead of raising because `quiet_ok` defaults true.

Exact reusable p13 failed-publication reproduction, confined to a fresh subdirectory of the authorized session workspace:

```powershell
@'
import importlib.util, json, shutil, zipfile
from pathlib import Path

base = Path(r"C:\Users\Gary\.copilot\session-state\66ccb9a5-db78-4934-afd2-9f61b6d9c57b\files\fresh_audit_20260904_delivery\p13_repro_again")
shutil.rmtree(base, ignore_errors=True)
paper, scratch = base / "paper", base / "scratch"
(paper / "auto").mkdir(parents=True)
(paper / "figures").mkdir()
(paper / "main_submission.tex").write_text(
    r"\documentclass{article}\begin{document}x\end{document}",
    encoding="utf-8",
)
(paper / "neurips_2026.sty").write_text("", encoding="utf-8")
(paper / "references.bib").write_text("", encoding="utf-8")
out = base / "release.zip"
out.write_bytes(b"PREEXISTING_VALID_SENTINEL")

p = r"C:\Users\Gary\Desktop\jepa\autopilot\p13_build_zip.py"
s = importlib.util.spec_from_file_location("p13_repro", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.PAPER = str(paper)
m.SCRATCH = str(scratch)
m.subprocess.call = lambda *args, **kwargs: 1
rc = m.build(str(out), False)
print("return =", rc)
print("sentinel_preserved =", out.read_bytes() == b"PREEXISTING_VALID_SENTINEL")
print("replacement_is_zip =", zipfile.is_zipfile(out))
print("ALL_PASS =", json.loads((scratch / "validation.json").read_text())["ALL_PASS"])
'@ | & 'D:\jepa_phase0\.venv\Scripts\python.exe' -
```

Expected result: return 1, `sentinel_preserved = False`, `replacement_is_zip = True`, and `ALL_PASS = False`.

The isolated release ZIP can be recompiled from the retained extraction with:

```powershell
Push-Location 'C:\Users\Gary\.copilot\session-state\66ccb9a5-db78-4934-afd2-9f61b6d9c57b\files\fresh_audit_20260904_delivery\zip_compile'
& 'D:\jepa_phase0\tools\tectonic\tectonic.exe' -X compile main.tex --keep-logs --keep-intermediates
Pop-Location
```

**Timing:** audit request timestamp/start was `2026-09-04T22:02:22.763Z`. The exact first-report finish timestamp is unavailable from the retained audit output. The report was complete before the amendment timestamp `2026-09-04T22:17:01.053Z`, giving an upper-bound request-to-observed-completion window of 14 minutes 38.290 seconds; this is not claimed as exact active runtime.
