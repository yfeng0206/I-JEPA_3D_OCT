# Release-safety implementation

Baseline: `de145d7`; branch: `fix/jepa-delivered-task-audit`.
Owner: repair-release, GPT-6 Astra. No commits, authentication, remote writes,
GPU use, or production DOCX regeneration by this agent.

## Latest bounded follow-up: review-input destination collision

Independent review closed the original five findings, then identified an
additional collision: a release output could overwrite the selected numeric
review input. The publisher now adds the **resolved review source** to its
protected input set for every ZIP/PDF/DOCX/DOCX-receipt/release-manifest target.

The new regression reproduced the old missing guard. After repair, all **ten**
explicit/default × destination cases reject before validation and preserve
review, source, artifact and sync-state bytes. The default DOCX-receipt case
emulates resolved file-link identity because its derived suffix cannot equal
the canonical default basename textually.

Fresh current combined release run: **83 passed**, zero failures/errors/skips.
The new `release_review_collision_fix_manifest.json` pins 15 current source
files and the actual new `release_review_collision_tests.xml` run.
`release_critic_fix_manifest.json` remains historical and was not restamped.
No `p15` edits, authentication, production regeneration/promotion, commit or push.

## Accepted critic findings: corrected implementation

The independent critic found five real gaps in the initial implementation.
They are now addressed; the earlier test counts below are historical milestones,
not evidence that those gaps never existed.

1. **Snapshot ABA:** every source copy must equal its captured digest; the final
   staged tree and archived source members must match the captured manifest.
   Standalone Word renders from its own verified private copy, not live inputs.
2. **Word replacement race:** the initial guard returns the identity it actually
   checked. Promotion rechecks that identity through a Windows handle denying
   noncooperating write/delete access, claims the prior file by that handle, then
   installs without overwriting an intervening pathname. Installed handles remain
   held until the manifest is installed. Real independent Python writers verify
   both rejection and preservation of conflicting versions.
3. **Rollback failure:** every restoration is attempted independently. Prior and
   candidate versions are retained in ignored `.release-recovery-*` directories;
   errors reveal recovery locations. No cleanup deletes the sole recovery copy.
4. **Word tables:** exact ordered row/cell numeric sequences, spans and labels
   replace subset counters. Surplus values, row/cell/label permutations, extra
   tables and caption numbers fail. Citation years are derived with citeproc.
   Only documented empty spacer rows and drawing-only figure layout tables are
   excluded; grouping/sign-glyph normalization does not erase precision.
5. **Credentials:** clone URLs are credential-free; token CLI arguments are
   rejected without echo. Authentication is environment-scoped Git configuration,
   with helpers, redirects, hooks, prompts and tracing disabled. Cleanup failures
   are visible and legacy config/FETCH_HEAD/reflog authentication text is sanitized.
   Both `_release_work` and test/recovery directories are ignored.

**Latest result: 64 targeted tests passed** (`release_critic_fix_tests.xml`).
Tests include real Windows file-sharing races and a real empty **local** Git
clone proving environment authentication is not persisted, plus mocked Overleaf
transport/cleanup failures. No actual Overleaf authentication was attempted.
The preserved full Word snapshot still passes all 134 references, 16 data tables,
31 captions, 16 image payloads and 47 bibliography entries
(`release_word_critic_validation.json`).

The claim/install protocol intentionally has a brief absent-path window.
Installation renames are atomic, but this is not a multi-file filesystem
transaction: readers must use the manifest and verify its hashes. Prior versions
are deliberately retained for crash recovery. The exclusive implementation is
Windows-specific and has no unsafe advisory-lock fallback.
`p15`, numeric bindings/schema, the manuscript, bibliography and source figures
were not edited during this critic-fix pass.

## Implemented

- `refresh_all.py` stops on every mandatory failure. Available figure producers
  run before release compilation; eight external figures are explicitly
  baseline-pinned inputs, not allegedly regenerated outputs.
- `p13_build_zip.py` uses unique project-local staging. It validates recursive
  inputs, placeholders, nonempty numeric/citation evidence, compilation, page
  budget, selectable-text anonymity, and Word completeness before promotion.
  ZIP/PDF/DOCX replacements are individually atomic with rollback; the release
  manifest is last. No validation failure replaces prior deliverables.
- `release_assets.py` / `release_assets.json` provide recursive input discovery,
  hashes, fixed/generated asset declarations, Word-conflict guards and promotion.
- `p15_verify_numbers.py` records every source numeric token, generated macro
  status and unresolved raster asset. Known AUCs, CIs, contrasts, counts and
  Table 1 cells bind to named statistics, not nearest matching values.
  Unknown/wrapped/absent required values and zero coverage cannot silently pass.
- Citation verification requires exact normalized-title agreement and persists
  per-entry authority records. Empty/missing/unresolved entries fail.
  Bibliographic existence is explicitly **not** claim support.
- Word conversion uses current compiled `.aux` labels, reference bookmarks,
  a styled References heading, source-derived caption/table checks, exact image
  payloads, bibliography coverage and macro-leak checks. Receipts hash the source,
  aux and DOCX. Untracked/edited Word files are preserved; mtime is irrelevant.
- Sync consumes only the manifest's exact archived source tree plus checked
  Word attachment. It rechecks local bytes, staged Git bytes and the fresh remote
  tip; remote edits/deletions are conflicts. No `--force`. Extra old remote files
  remain untouched. Manifest-free dry runs are explicitly discovery-only.
- Punctuation verification requires an explicit comparison base and rejects
  empty comparisons, unexpected edits and changed numeric/citation/ref tokens.

## Validation

```powershell
$env:MPLBACKEND='Agg'
& 'D:\jepa_phase0\.venv\Scripts\python.exe' -m pytest `
  tests\test_release_safety.py tests\test_release_evidence.py `
  tests\test_release_sync.py -q --basetemp .\_release_tests\verified
```

**40 passed**, including a real Tectonic/Pandoc standalone release followed by
a rejected numeric rebuild. Compiler, page, numeric, zero-coverage and Word
failure injections preserve old ZIP/PDF/DOCX/manifest/sync-state sentinels.
Authority calls are mocked in tests; no citation network requests were made.
Machine-readable results are retained in `release_tests.xml`.

An isolated full manuscript snapshot passed: **134 cross-references, 16 table
captions, 15 figure captions, 17 tables, 16 images and 47 bibliography entries**,
including caption prose/numbers and source-cell checks. This snapshot has source
hash `af47a1c8904d0f82b7a1e76c727a31f2945baa4a995554fcabc89a6dd9c9e07c`;
the coordinator has since edited the live manuscript. It is not the final Word
deliverable. Details/hashes: `release_word_validation.json`.
The isolated Word/source evidence remains under
`_release_work\word-real-a117144e663347fa9125147816d6d383`.
Earlier test and Word-conversion intermediates were cleaned; the final pytest
fixtures remain under `_release_tests\verified`.

Production ZIP, PDF and DOCX still match the protected baseline hashes:
`d33f6b…`, `5e7f72…`, `7610d2…`, respectively.

## Explicit release blockers / handoff

For source snapshot `df1251a97e826a829918f05543e42b921e5e3c2ee7cb429c7aa8ce2b6b54cd6d`,
numeric coverage verifies **62 macros, including 12 rendered AUCs**. It blocks on
**93 macros, 1,259 conservative literal records and 14 raster assets**.
Literal records include layout/protocol constants requiring classification;
raster quantities are not individually enumerated. These are coverage gaps,
not findings that all those values are wrong. See `release_numeric_coverage.json`.
The anatomy-envelope CI initially appeared mismatched because the checker used
bootstrap rather than the generator's DeLong interval; the binding was corrected,
not the manuscript.

All **47 citations remain offline-unresolved** without retained matching
authority records (`release_citation_coverage.json`). This does not assert the
papers are nonexistent. **The real manuscript cannot currently pass strict
release gates.** Do not describe the repaired pipeline as universal numerical or
scientific verification, and do not bypass these failures.

After evidence review and manuscript stabilization, the coordinator can build
with `--paper-dir`, `--staging-root`, `--out`, `--pdf-out`, `--docx-out`,
`--manifest-out`, and `--citation-record` overrides. Adopting an existing
untracked Word file requires its explicitly reviewed `--expected-docx-sha256`.
Sync then requires `--release-manifest <zip-stem>.release.json`.
The legacy loose `_files` mirror and manual upload marking are not refreshed;
use the validated ZIP/manifest workflow. Anonymity checks do not perform OCR.

## Coordinator source-update follow-up

Two additional isolated regressions passed after the title/Figure 1/DSeq update
(`release_source_update_tests.xml`): a single TikZ pipeline produces one image
and one caption with the canonical title preserved; a versioned arXiv eprint
is no longer overridden by its unversioned arXiv DOI. Published non-arXiv DOI
precedence is unchanged. No real-manuscript regeneration, bibliography edit,
handoff overwrite or `p1c_stats` value/family-key change was made.

The subsequently selected `fig_delivered_task_token_maps.png` is registered
with its real `--maps-only` producer, replay/config dependencies, export manifest,
and PNG/PDF/SVG identities. Script, replay-input and installed-output hashes were
checked against that manifest without regeneration. The PNG is a pinned reviewed
export: a changed image fails identity validation until a new version is reviewed.
`rendered_metrics` is explicitly empty; this is one illustrative audited failure,
not an aggregate-frequency or downstream-AUC chart. Quantitative DT values remain
separate. Two targeted asset-registration tests passed
(`release_maps_registration_tests.xml`). This follow-up changes only asset
registration/pinning, not the Windows promotion or authentication fixes.

## Numeric-review publisher integration

`p13_build_zip.py` and `refresh_all.py` now accept `--review-file`; absent that
flag, an existing `paper-dir\numeric_reviews.json` is selected. The publisher
copies the exact bytes into private release evidence, passes that immutable copy
to `p15`, requires the reported `review_sha256` to match, and rechecks the source
and archived copies before publication. The exact bytes are also embedded as
base64 in the release manifest for durable recovery. This preserves review
identity, **not** scientific approval. Sync verifies this receipt again.

Review metadata is deliberately excluded from the anonymous source ZIP and
Overleaf's managed files; it may contain internal reviewer/protocol notes.
Six targeted routing, tamper/ABA and failure-preservation tests passed
(`release_review_input_tests.xml`). No `p15` or schema edits were made.

Integration issue routed to the numeric owner: staged root documents are named
`main.tex`, while several new binding/review lookups use `main_submission.tex`.
Those need a narrow logical-name alias without changing actual input hashes or
the approved review bytes before live staged acceptance can be claimed.

## Stable-source Word candidate and remaining blockers

The numeric owner has now implemented the logical root-name alias. On the
coordinator's explicitly supplied source hash
`61d8d2cceb397f58313b166d481a03a3f94c2c0c914fb2ee8caca25cef7e27e6`,
an isolated Word candidate and receipt were generated at:

`_release_work\final-word-6a96c09d7be443468e3a4023c5829a0c\main_submission.docx`

Word SHA-256:
`ef2f6537e4defea8ddadee2a0b5eae9c13317ea8457372446a57ac920e0aa687`.
Parity passes **135 references, 17 data tables/captions, 16 figure captions,
16 images and 45 bibliography entries**. Its private PDF has 38 pages, main
content within nine pages, and no identifying selectable-text/metadata hits
after fixing a false positive on cited author **Gary S. Collins**. Exact author
identity/project accounts and author-specific local paths remain checked; the
manuscript/bibliography were not changed to satisfy the scanner.

Offline retained authority records match **45/45** currently cited entries in
that snapshot. Numeric coverage still blocks on **385 literal tokens and seven
figures**; seven plotted-value validators now pass with no mismatch or gate
errors. An initial numeric CLI attempt hit an `int64` serialization exception;
subsequent direct-audit and canonical-CLI checks serialized normally. No generic
serialization workaround or `p15` edit was applied.

**72 release regression tests pass** (`release_stable_source_tests.xml`).
Production DOCX remains at the protected hash `7610d2…`.

During final verification, the live source changed to
`6536f11df140206f1ed8be7bc7a7a3a5136b1adeec6f164c345d84625ef1f05a`
and stopped referencing five interpretation figures. Consequently this Word
candidate is valid for the supplied snapshot, **not certified as current for
the newer live source**. No production promotion occurred. Exact snapshot,
gate reports, review identity and drift details are retained in the work folder
and `release_final_word_summary.json`.

## Adopted replacement assets

The coordinator-adopted `fig_purity_auc_ep50_fp32.png` and
`fig_policy_family_token_maps.png` are now registered with their actual producer,
independent validator, source manifest, numeric-validator/illustration receipt
handoffs, dependencies and pinned output identities. Both supplied PNG hashes,
all companion hashes, producer/validator hashes, and the two public numeric
source hashes match the retained records. Registry identity is not a substitute
for the numeric owner's quantitative scatter gate or illustration review.

All eight now-unreferenced legacy assets moved from active `fixed_inputs` into
`retired_inputs` metadata. Their original files and hashes are unchanged.
The current source requires ten declared raster assets and no retired one;
accidental reintroduction of an unreviewed retired asset fails the registry gate.
Two targeted registry tests pass (`release_replacement_registration_tests.xml`);
the active inventory is retained in `release_replacement_registry.json`.
No figure regeneration, caption/manuscript edit, numerical receipt merge, Word
regeneration or publication occurred. Renewed Word remains deferred pending the
coordinator's new stable-source hashes.

## Renewed source `33926d…`: current isolated candidate

All three renewed coordinator hashes matched, and the full live input set
remained unchanged during verification. Candidate:

`_release_work\final-word-replacements-cd5a455ee6424cde8e29995e354cf49d\main_submission.docx`

SHA-256:
`fd5cde70e7c15f70e7d8eb4dbe7c5cf55439ee9eae3ebb013c23ad7f45fbd19e`.
Word parity passes **128 references, 17 data tables/captions, 11 figure
captions/images and 45 bibliography entries**. The private PDF is **34 pages**,
with **nine main-content pages** and no identifying selectable-text/metadata
hits. Offline citations pass **45/45**; manuscript checks pass.

Numeric coverage remains blocked for the captured review input
`c71ff195…`: **377 literal tokens and the two new replacement figures**.
The registry knows and pins both figures, but registry identity does not install
the numeric owner's quantitative validator or merge the separate illustration
receipt. Seven older plotted-value validators and the delivered-task map
illustration pass; no numeric gate errors are reported.

`release_renewed_word_summary.json` records all exact paths, hashes and gate
outcomes. Production DOCX remains `7610d2…`; no promotion or remote operation
occurred. This candidate supersedes the earlier, revoked `61d8…` candidate
for the renewed input set, but is not a scientific release approval.

## Anonymous archive and local-source exclusion

An additional production-code assembly regression inspected actual synthetic ZIP
members with all seven withdrawn stems present beforehand as PNG/PDF/SVG files.
None entered the ZIP; every original remained unchanged. Referenced synthetic
full-paper/README sources also remained local: the QA archive contains the exact
selected review JSON, not dereferenced document contents. The test passed
(`release_archive_privacy_tests.xml`).

The then-current live source hash `5ec7b4b4…` had 22 allowlisted source members
and zero withdrawn stems (`release_archive_scope_check.json`). This is a source
allowlist check, **not** a claim that the final release ZIP has been built or
passed its still-blocked scientific gates. The preserved Downloads ZIP is an
older baseline, not certified for the current anonymous scope. A complete
Overleaf project export can also retain remote-only historical extras; only the
new gated archive should be used unless the coordinator separately audits those
extras. Original legacy figures and local full-paper/full-README evidence remain
preserved. Current ignore-policy identity is recorded in the scope report.

## Final isolated Word for approved `5ec7…` source

Built once against main `5ec7b4b4be077e3e1a46f08e01ff896c3c11eec02ea4304ee81ead9530b5bca1`
and parent review `2d1985e01d7be99510e1181d77171d18b38ed53cec9c5a3420fffa834eac3db6`.
Main, bibliography, delivered macros, all other source inputs and the selected
review remained unchanged throughout.

Folder: `_release_work\final-word-5ec7-c3ce64e6a2824229940bc487db057f20`

- `main_submission.docx`:
  `8d2dd3f81875f00ae340f5421d647547b1683c4444d11f8c6708762bedc3045b`
- `main_submission.docx.provenance.json`:
  `0e30b3c3538e10da75b70691e706e3d5ae2e9c8fef208d81fb95bbe0f0a603c6`

Parity checked against the **live approved source** passes 129 references,
17 data tables/captions, 11 figure captions/images and 45 bibliography entries.
Private PDF: 34 pages, nine main-content pages, no identifying selectable-text
or metadata hits. Production DOCX remains `7610d2…` and was not overwritten.
No new numerical audit or publication was performed; the parent still controls
the actual final `p13` run after numeric `ALL_PASS`.
Exact paths/hashes/results: `release_final_5ec7_word.json`.

## Numeric acceptance confirmed; parent integration ready

The canonical numeric report now has actual `ALL_PASS: true`, no errors,
unresolved entries or mismatches, for the unchanged approved `5ec7…` source.
Selected reviewed bundle:
`evidence\paper_release_numeric_candidate.json`, SHA-256
`e21e113037a0e213842c937d37a867a5a3133c531a5867356b73ec89bfa54a16`.
It records eight programmatically verified plots and two explicitly
non-mathematical source-reviewed illustrations; the numeric owner reports
103 passing tests.

The existing final Word receipt's input hashes still match live source.
No new freeze or duplicate Word generation was performed. Its build-time
review snapshot remains recorded as `2d1985…`; the newly confirmed numeric
bundle is linked separately in `release_final_5ec7_word.json`, without rewriting
the Word or its receipt. Production DOCX remains unchanged. The parent must
perform the final full `p13`/all-tests run using the selected successful review
bundle, then decide promotion/publication. No release ZIP or remote publication
by this agent is claimed.
