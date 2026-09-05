# Paper workflow state

## Current reviewed release

Source release: `6f4d62e`, branch `fix/jepa-delivered-task-audit`.
`main` is not merged. The approved engineering investigation, bounded GPU
diagnostics, source review and workshop revision are complete; no corrected
policy was pretrained. See
`autopilot\investigations\delivered_task\RESULTS.md`.

The delivered PDF has nine body pages (34 total). The current Word copy is
`main_editable.docx` in Overleaf. The exact source/Word manifest was synchronized
and independently rechecked; remote-only historical files remain preserved.
Use the validated Downloads ZIP, not a whole-project export, for anonymous
source submission. The old loose `_files` mirror is historical.

Release commands from the repository root:

    $env:MPLBACKEND = 'Agg'
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p13_build_zip.py --citation-record autopilot\investigations\delivered_task\evidence\citation_authorities.json

The publisher uses `paper\genai4health2026\numeric_reviews.json`, retained local
statistics and source-review evidence. Full third-party PDFs are not committed;
their acquisition manifest is under the investigation's `literal_sources\public`.
Missing or changed evidence blocks release rather than being silently skipped.
The Word conflict guard must not be bypassed if collaborators have edited it.

For authorized sync, load credentials from Windows user scope in the same
process without printing them, then use the newly generated release manifest:

    D:\jepa_phase0\.venv\Scripts\python.exe scripts\sync_overleaf.py --release-manifest C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_FINAL.release.json

No force sync, automatic main merge, sustained pretraining or actual OpenReview
submission is implied by these commands.

> **Historical snapshot below (August 27).** A fresh September 4 audit found
> gaps in the earlier readiness, inference, and build-safety assurances.
> See `VERSION_BOARD.md` and `autopilot\reports\fresh_audit_2026-09-04`
> for the current evidence-bounded status. In particular, an internal AI review
> is not a workshop decision, and a recorded sync is not a fresh remote check.

Resume point for this submission. Written 2026-08-27. Update it when the state
changes; it exists so a fresh session, or a Windows restart, can pick up without
re-deriving everything.

## Why this file and not a `paper-builder` skill

A phased paper-builder skill was proposed - inspect, verify, diagnose, analyze,
figures, write, review, experiment. Checked against this repository, seven of the
eight phases already exist, and in a stronger form than a skill can offer: these
are **build gates that fail non-zero and block shipping**, not advisory checks an
agent may skip.

| proposed phase | already implemented by |
|---|---|
| 2 Verify: every number to its source | `auto/auto_numbers.tex` macros generated from artifacts, `research/numbers_master.csv`, `results/downstream/ARTIFACT_MAP.json` |
| 3 Diagnose inconsistencies | `autopilot/check_manuscript.py` (dangling refs, undefined macros, missing citations), `autopilot/p15_verify_numbers.py` (cross-arm attribution is a build error) |
| 4 Analyze, paired bootstrap and CIs | `autopilot/p17_adjust_subgroup_multiplicity.py` |
| 5 Figures from real data | `autopilot/p8_make_assets.py` - reads stored JSON/CSV/NPZ, never invents |
| 7 Review: compile, page limit, refs | `autopilot/p13_build_zip.py` - six checks, exits non-zero |
| 8 Experiments, gated | `scripts/chain_replication.py` with `campaign_supervisor`, idempotent and resumable |
| delivery | `scripts/sync_overleaf.py`, `upload_weights.py`, `download_weights.py` |

The one thing genuinely missing was **resumable state**, which is this file.
Building the rest would have duplicated working infrastructure nine days before a
deadline.

## Current state

- Submission: `paper/genai4health2026/main_submission.tex`
- Venue: GenAI4Health @ NeurIPS 2026, Research Paper track. **Deadline 2026-09-05 AoE.**
- Body limit **9 pages**, appendix uncapped, double-blind, **no rebuttal phase**.
- Status: 9 pages, 6/6 ZIP checks, `check_manuscript` and `p15_verify_numbers` both PASS.
- Latest blind review (R11): **4/6 Weak Accept**, Quality 3/4, Clarity 4/4, confidence 5/5.
- Title: *Anatomy-Guided Masking for JEPA Representations on Retinal OCT:
  Segmentation-Free Guidance Matches a Segmenter.*
- Overleaf: **in sync** as of 2026-08-27, project `6a8fb344bf38f34a9e545791`.

### Style, measured against the field

`autopilot/reports/VENUE_STYLE_RESEARCH.md` measured 26 recent arXiv cs.CV
papers and 15 landmark papers. Keep the manuscript inside these:

| | field norm | this paper |
|---|---|---|
| em dashes | median 0.00 per 1k words, 57.7% have zero | 3, all Table 1 "not applicable" cells |
| semicolons | median 1.74 per 1k words | 1.91 |
| emoji / non-ASCII | - | 0, file is pure ASCII |
| citations verified to exist | - | 47 of 47 |

**No LLM-use disclosure is required.** NeurIPS 2026 states verbatim that writing,
editing and formatting use "does not need to be documented". Do not add one.
The one behaviour it *does* sanction is unverified references, with revocation
of publication status even post-acceptance, which is what `verify_citations.py`
exists to prevent.

## Commands

    # build, validate, publish ZIP and loose mirror to Downloads
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p13_build_zip.py

    # the two gates
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\check_manuscript.py
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p15_verify_numbers.py

    # push to the live Overleaf project (refuses on conflict; --force to override)
    D:\jepa_phase0\.venv\Scripts\python.exe scripts\sync_overleaf.py
    # after you upload by hand, reset the change baseline:
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p13_build_zip.py --mark-uploaded

    # regenerate figures and generated tables from stored artifacts
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p8_make_assets.py

    # resume the paused pretraining replication (about 6 days, 6 legs)
    D:\jepa_phase0\.venv\Scripts\python.exe -u scripts\chain_replication.py

    # style and integrity checks added 2026-08-27
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\verify_citations.py --online
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\count_semicolons.py
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\verify_punctuation_only.py

    # the two figures that now have generators
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\make_fig_geometry_panel.py
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\make_fig_specificity_ladder.py

    # rebuild the Word copy for collaborators, then verify it
    # (sync_overleaf warns, but does not block, if this is older than the .tex)
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\make_docx.py
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\check_docx.py

## Open items, all operator decisions

1. **Public repository and double-blind.** `main_submission.pdf` is world-readable
   at `yfeng0206/I-JEPA_3D_OCT` under the author's real identity, while the call
   names "a GitHub repository revealing authorship" as grounds for desk rejection.
   Not acted on because changing visibility affects a collaborator's clone. See
   `autopilot/reports/GITHUB_AUDIT.md`.
2. **Rotate the Overleaf git token.** It was pasted into a chat session. The
   sync itself works and the project is currently up to date; after rotating,
   update the user-scope variable with
   `setx OVERLEAF_TOKEN <new token>` and open a new shell.
3. **Replication.** Paused at epoch 26, resumable in one command. Venue evidence
   says it is not required: of 17 accepted 2025 research papers only one reported
   multiple seeds, and two orals had no seed protocol at all.

## Hard-won conventions, do not relearn these

- **Never hand-type a number into the .tex.** Every number is a macro generated
  from a stored artifact. Twenty-five wrong numbers were found in one day and
  every one was hand-typed, which is exactly why the gates could not see them.
- **A reported contradiction is not always an error.** Twice, two numbers that
  looked contradictory were both true at different scopes. Verify against the
  artifact before changing a digit; "fixing" either would have introduced an error.
- **The body has zero slack at 9 pages.** Source-line count does not predict
  typeset pages: two added table-caption lines cost a full page while a net
  three-line reduction did not recover it. Always re-run `p13_build_zip.py`.
- **Trust the gate's page count, not a heuristic.** "References appears in the
  first 400 characters of a page" reported 9 pages while the real gate reported
  10, because body text preceded the heading and pushed it to character 428.
  `p13_build_zip.py` measures the heading's y-offset and is the authority.
- **A frozen References y-offset means a rendered-line change.** If the offset
  is *identical* across two different edits that both overflow, the cause is a
  changed number of rendered lines, not text length. A three-line `\title` that
  wraps to four lines does this, and shortening the individual `\\` segments
  fixes it where trimming total characters does not.
- **Never edit the .tex with PowerShell `-replace` or any shell regex.** A
  replacement containing `$0.784$` was read as group references and recursively
  duplicated text into a committed failing build. Use the `edit` tool, or Python
  with plain `str.replace` and an exact-count assertion.
- **Punctuation is a scored style axis, not cosmetics.** Em dashes and
  semicolons both ran far above the field norm and ten prose reviews never
  looked, because they only checked vocabulary. Re-measure after any prose pass.
  Prefer `;` becoming `.`, which is width-neutral and so page-safe.
- **Crossref does not serve arXiv's DataCite DOIs.** A real reference with DOI
  `10.48550/arXiv.2310.02492` looks fabricated if resolved at Crossref. Route
  those to the arXiv API instead.
- **Overleaf credentials live at Windows *user* scope.** `setx` values are not
  inherited by an already-running process tree, so `$env:OVERLEAF_TOKEN` reads
  empty in a fresh shell. Load them in the same call and never echo the token:

      $env:OVERLEAF_TOKEN = [Environment]::GetEnvironmentVariable('OVERLEAF_TOKEN','User')
      $env:OVERLEAF_PROJECT_ID = [Environment]::GetEnvironmentVariable('OVERLEAF_PROJECT_ID','User')

- **PDF text extraction has three traps here.** TeX ligatures ff/fi/fl/ffi/ffl are
  single glyphs (normalise NFKD); words hyphenate across lines; and line numbers
  interleave *inside* sentences, so "Hard patches mining" extracts as
  "Hard 557 patches mining". All three have produced false findings.
- **Set `MPLBACKEND=Agg`.** A skill script with no CLI ran a demo calling
  `plt.show()` and opened windows on the operator's machine.
- **Never `git reset --hard`, never rewrite history.** A collaborator has a clone.
