# Paper workflow state

## Status

**Accepted** to the GenAI4Health NeurIPS 2026 workshop. Release v9 (v8 plus one
Introduction sentence on medical imaging foundation models) is merged to `main`;
its evidence is `autopilot\investigations\submission_refresh_20260910\release_handoff_v9.json`.
Overleaf still holds v8. Camera-ready/poster work continues on branch `poster-ready`:
use the `genai4health_2026` template, restore authors, and address reviewer feedback.
Camera-ready deadline: October 25, 2026, 11:59 PM AoE (= Oct 26 04:59 PDT); internal
upload target Oct 24. **The staged camera-ready plan (seed replication, matched-budget
control, reproduction gates, GPU schedule, paper changes) is
`paper\genai4health2026\CAMERA_READY_PLAN.md`.** Nothing in it has been executed yet.

## Post-freeze runbook (cr_seed_v1; results freeze Thu 2026-10-22 08:00 PDT)

`autopilot\cr_postfreeze.ps1` runs the whole post-freeze sequence fail-closed (every analysis,
build, integration and gate step must exit 0 and the first failure stops it; only the phase-1
receipt preview's p15 exit and the scratch poster build's exit are recorded in `REVIEW.md`
instead; transcript and per-step timing in the output root). It refuses to run before the
freeze, is CPU-only and Idle/BelowNormal priority (Tectonic and p13 with its Tectonic child at
Idle), waits for >= 4 GiB free commit before heavy steps and for any other Tectonic compile,
and never touches a running campaign. Rehearsed end to end on synthetic campaigns on
2026-10-11 and again after the V8 fixes (report: session files
`camera_ready\impl_i12_postfreeze_runbook.md`, section "V8 fixes").

1. **Phase 1, about 9-10 min in the rehearsal (unseal ~1 min, final ~5-5.5 min, MD ~2 min);
   allow 20-30 min with the real MD caches (manuscript text unchanged).** From the repository
   root:

       powershell -NoProfile -ExecutionPolicy Bypass -File autopilot\cr_postfreeze.ps1

   Preflight plan (campaign state, probe dirs, late / in-progress runs) -> `cr_stats`
   inventory -> unseal every ready probe dir that has no receipt (never twice) -> `final`
   (~5-6 min) -> MD on the test split (`cr_md_regression.py --split test`, the registered
   secondary analysis; 2 min on synthetic caches; the real caches are larger, allow ~10-20 min;
   it always runs, build always gets `--md`, and a missing MD metadata file stops the preflight
   before anything is unsealed) -> RANDOM-CB budget audit from its trainer logs ->
   `make_cr_results build --update-reviews` -> `build --check` -> scratch copy, integrate,
   Tectonic -> receipt preview
   -> scratch poster build -> `REVIEW.md`. `final` and MD are cached by a key over their input
   hashes *and* the completion evidence (stop file path / existence / hash, seal and probe
   times), and the cache record also holds the sha256 of the output it produced (an output
   edited afterwards is recomputed, with a warning). A cached or fresh `final.json` must agree
   run by run with the current evidence, and its labels, complete indices, contrasts, means and
   RANDOM range must be what the registered rules give for the receipt-verified per-run AUCs
   (`cr_postfreeze.py check-final`; `make_cr_results` repeats both checks and refuses a stale
   or edited final or MD JSON). Phase 1 leaves the manuscript text unchanged but
   already writes `paper\genai4health2026\auto\cr_*` and the `Rep*` bindings in
   `numeric_reviews.json` (revert both if you abandon before phase 2).
   Output root `D:\jepa_phase0\autopilot_out\cr_postfreeze`. Exit 3 = a registered run is still
   in progress (step 4), exit 2 = refused input (read `logs\`).
2. **Human review.** `REVIEW.md` (scenario, labels, matched control, display warnings, TL;DR =
   `paper\genai4health2026\auto\cr_tldr.txt` for OpenReview) and the scratch PDF
   `scratch\paper\genai4health2026\main_submission.pdf`: abstract, introduction, contributions,
   section 5.2 with Tables 2-3, discussion, limitations, conclusion, appendix tables (incl.
   `tab:md`) and the seed-spread figure ((a) test AUC per encoder in the paper's arm colours,
   seed letters O/A/B/C; (b) paired differences from RANDOM with the grey +/- r band only when
   the registered RANDOM range exists; printed 1:1 at `\linewidth`, text >= 7 pt; its colour /
   marker / label contract is in `fig_cr_seed_spread.evidence.json` under `encoding`); at most
   10 main pages (rehearsal worst case after the 2026-10-12 figure and text polish: main text ends at 9.93),
   no undefined references. Fill the acknowledgments (`\CRAcknowledgmentsPending`) before step 3.
3. **Phase 2, about 1-2 min.**

       powershell -NoProfile -ExecutionPolicy Bypass -File autopilot\cr_postfreeze.ps1 -ApplyToPaper -FigureReviewer "<name>, 2026-10-22"

   Integrates the real paper (including the scenario-dependent coordinator-note-2 edits),
   registers the seed-spread figure (producer in `autopilot\release_assets.json`, receipt in
   `numeric_reviews.json`, signed by `-FigureReviewer`), migrates context-hashed receipts, runs
   p15 and then `p13_build_zip.py --camera-ready` (the camera-ready release command below). A
   receipt is re-keyed automatically only when its paragraph changed solely through the
   integration's own replacements (same file, value and window, found exactly once, one to
   one); every other changed claim context is listed as `manual` in `receipt_migration.json`
   with `old_context` / `new_context` and needs `-AcceptReceipts <id>` after a human compared
   them (the `-FigureReviewer` name signs it). The first pass is expected to stop with exit 4
   on one literal inside a replaced passage: the `epoch-25` in `app:replication`
   (`compact_protocol.tex:b3f23f54ee770b0c:35` as of the 2026-10-12 text polish; the id changes whenever that paragraph changes, so read the current one from `receipt_migration.json`). Compare its
   old and new context, then re-run phase 2 adding
   `-AcceptReceipts compact_protocol.tex:b3f23f54ee770b0c:35`. p13 fails only on
   `authors_present` while the acknowledgments placeholder remains. Then review `git diff`,
   commit, and sync per the release commands below.
4. **If E3 (or any run) is late.** A run counts only if its trainer stop file, seal and probe
   all predate the freeze (`cr_stats.completion_eligibility`, timestamps written by the trainer
   and probe; keep the stop files where they are). A seed index with a late or missing
   RANDOM/CENTROID/ENVELOPE run is descriptive; the labels then come from the complete indices.
   The registered labels, including the matched-control description, also need the re-probed
   original RANDOM anchor on time: without it (or with fewer than two complete indices) no
   label is assigned, the matched control is reported descriptively (`M_incomplete`, no
   RANDOM range shown), and the text names that reason.
   If E3 is still training or probing at 08:00, phase 1 stops with exit 3. Either wait until the
   campaign marks E3's probe DONE (E3 is then reported as finished after the freeze, an open
   marker in the figure) and re-run, or re-run now with `-AllowIncomplete` (E3 is reported as
   not completed, pre-registration section 7). Both leave the primary analysis unchanged; decide
   before phase 2. A re-run of phase 1 opens only new probe dirs and recomputes `final`/MD only
   when their inputs or the completion evidence changed. After phase 2, going back needs `git checkout --
   paper/genai4health2026/main_submission.tex paper/genai4health2026/compact_protocol.tex
   paper/genai4health2026/numeric_reviews.json autopilot/release_assets.json` and deleting
   `pre_integration_paper` in the output root.
5. **Rehearsal (synthetic only).** `python tests\cr_postfreeze_fixtures.py <scratch>\fx [--late
   envelope:9012 | --in-progress envelope:9012 | --matched M_na]`, then the script with
   `-Rehearsal -FixtureRoot <scratch>\fx -OutRoot <scratch> -TargetRepo C:\t\<short> -FreshMirror`
   (a mirror copy of this working tree stands in for the repository). Real run, campaign,
   worktree, paper and repository paths are refused; the fixture root needs the builder's
   `CR_SYNTHETIC_FIXTURE.json` and `"synthetic": true` in campaign.json, state.json and every
   results.json, and every path derived from the synthetic state (probe dirs, stop files,
   trainer logs, feature caches, sealed entries; junctions resolved) must lie inside the
   fixture root before it is read or unsealed with `--allow-before-freeze`. Fixture timestamps
   are fixed (on time: 2026-10-20; late: after the freeze), independent of the wall clock.

## Current reviewed release

Submission revision: **2026-09-10**, developed on branch `fix/jepa-delivered-task-audit` (merged to `main`).
Title: **Where to Predict in Retinal OCT? Anatomy-Guided Target Selection for
I-JEPA**. The narrative now introduces JEPA before the masking strategies,
adds a JEPA formulation and an openly licensed real-OCT illustration, distinguishes
the study from US-JEPA and DSeq-JEPA, and omits separate secondary research threads.
Historical AUCs are unchanged; the subsequent probe-budget diagnostics are not
substituted for the reported results. No corrected policy was pretrained.

The final PDF has **eight main-content pages (15 total)**. The requested PDF,
source ZIP and release manifest are in Downloads with stem
`OCT_JEPA_GenAI4Health2026_20260910_v8`. This compact revision includes the symmetric
uniform-versus-guided OCT framework diagram and a number-free abstract led by
OCT annotation constraints and self-supervised learning.
The extended low-label, subgroup/calibration, fine-tuning, background and
replication-plan sections are omitted from this workshop submission; their
research records and previous long manuscript remain preserved.
The original attribution work now motivates retinal targeting in the introduction.
The feature-occlusion delta-logit definition is kept; the archived heatmap
panels were removed in the camera-ready (2026-10-12) because they are adapted
FairVision B-scans (CC BY-NC-ND 4.0), and the text now says the displays are not
reproduced. Their use was illustrative rather than a validated localization result.
The final notation models targets and context jointly, defines strategy-specific
guidance and explicitly identifies the tissue statistics as a predicted proxy.
The diagram retains the correct EMA teacher, context encoder, predictor,
positional queries and latent loss. All 13 release gates passed. The current
Word copy is `main_editable.docx` in Overleaf, with editable display
equations. Word is generated internally but is no longer a requested deliverable.
The 13 managed source/attachment files were synchronized to Overleaf;
remote-only historical files remain preserved.
Use the validated Downloads ZIP, not a whole-project export, for anonymous
source submission. The old loose `_files` mirror is historical.
No OpenReview submission or merge to `main` was performed by this workflow.

Release evidence:
`autopilot\investigations\submission_refresh_20260910\release_handoff_v8.json`.
The previous `6f4d62e` release and engineering investigation are documented in
`autopilot\investigations\delivered_task\RESULTS.md`.

Release commands from the repository root:

    $env:MPLBACKEND = 'Agg'
    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p13_build_zip.py --out C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_20260910_v8.zip --stats-dir D:\jepa_phase0\autopilot_out\p1_stats --citation-record autopilot\investigations\submission_refresh_20260910\citation_authorities.json --review-file paper\genai4health2026\numeric_reviews.json

Camera-ready release (2026-10-08 tooling; authors-present gate, 10 main pages,
workshop template gate). It fails while `\CRAuthorsPending`,
`\CRAcknowledgmentsPending` or any `\ph` result placeholder remains, and until
`paper\genai4health2026\camera_ready_authors.json` is confirmed (authors were
filled and confirmed 2026-10-10; acknowledgments are still pending):

    D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p13_build_zip.py --camera-ready --stats-dir D:\jepa_phase0\autopilot_out\p1_stats --citation-record autopilot\investigations\camera_ready_20261008\paper_i8\citation_authorities_cr.json --review-file paper\genai4health2026\numeric_reviews.json

The publisher uses `paper\genai4health2026\numeric_reviews.json`, retained local
statistics and source-review evidence. Full third-party PDFs are not committed;
their acquisition manifest is under the investigation's `literal_sources\public`.
Missing or changed evidence blocks release rather than being silently skipped.
The Word conflict guard must not be bypassed if collaborators have edited it.

For authorized sync, load credentials from Windows user scope in the same
process without printing them, then use the newly generated release manifest:

    D:\jepa_phase0\.venv\Scripts\python.exe scripts\sync_overleaf.py --release-manifest C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_20260910_v8.release.json

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

    # DO NOT RUN the old replication chain: it would resume a stale epoch-26
    # checkpoint under changed code and train ENVELOPE with a changed sampler.
    # See CAMERA_READY_PLAN.md (fixes E1-E19 and gates G0-G2 come first).
    # D:\jepa_phase0\.venv\Scripts\python.exe -u scripts\chain_replication.py

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
3. **Replication.** Now requested by the area chair and all three reviewers for the
   camera-ready. The August chain must not be resumed as-is; follow
   `paper\genai4health2026\CAMERA_READY_PLAN.md`.

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
