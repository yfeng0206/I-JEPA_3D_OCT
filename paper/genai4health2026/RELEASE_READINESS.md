# Camera-ready and poster release readiness

Approved scope: 2026-10-18. This is the current execution checklist for submission 252,
**Where to Predict in Retinal OCT? Anatomy-Guided Target Selection for I-JEPA**.
It does not change the [pre-registration or amendments](PREREGISTRATION_cr_seed_v1.md).
The [post-freeze runbook](../../PAPER_STATE.md) supplies the detailed commands.

## Current update — October 22, 2026

The user approved pausing optional third-seed training to focus on finalization.
R3 paused after its saved epoch-29 checkpoint on October 22 at 21:29:45 PDT; do not start C3/E3
or resume the queue without a new user instruction. This supersedes the earlier
"continue campaign" scheduling instruction below, not the registered reporting rules.

Post-freeze phase 1 has produced review copies of the paper and poster, with real
results, verified numeric outputs and successful builds. Only seed 1234 is complete
on time; E2's late finish makes index 5678 descriptive. Both guided outcomes remain
incomplete, and the matched control is descriptive. The latest main text ends at 9.93 pages
after the author's October 23 future-work and background-context clarifications
(future-work-only draft: 9.84; initial draft: 9.49);
the print poster remains 610 x 850 mm. The source manuscript awaits final integration.
Author review, acknowledgment/funding wording and final receipt migration/release checks
remain open. See the current checkpoint in `PAPER_STATE.md` for artifact paths and findings.
The newest Overleaf review ZIP is in Downloads; its extracted source compiles as
`main.tex` and matches the latest review PDF. It remains a review package pending those
open items, not a certified final release.

## Scope and timing

- Continue `cr_seed_v1` in its current order. Finish CB and E2 as the immediate priorities.
- Keep test predictions sealed until **2026-10-22 08:00 PDT**. Use validation-only gates.
- Prepare the review candidate at the freeze using `-AllowIncomplete` when needed;
  do not make release preparation wait for the last training run.
- Review on October 22–23; target upload on October 24. The official deadline is
  October 25, 23:59 AoE (October 26, 04:59 PDT).
- Retain the current paper structure and 61 x 85 cm portrait poster. Make only necessary
  result integration, clarity, consistency and small readability fixes.
- Defer new datasets, RETFound, new masking methods, longer training, and storage cleanup.
  Preserve the already implemented registered pooling and severity analyses.

The official [workshop FAQ](https://aihealth.ischool.utexas.edu/GenAI4HealthNeurips2026/)
allows 10 main-text pages, excluding acknowledgments, references and appendices.
The [NeurIPS poster instructions](https://neurips.cc/Conferences/2026/PosterInstructions)
specify 61 x 85 cm portrait for workshops and a November 4, 23:00 AEST advance printing
deadline for Sydney. Rules were checked October 18; confirm ordering details before purchase.

## Verified starting point

At October 18, 17:13 PDT, the working branch was `poster-ready`, based on `b5ea672`:

| Item | State |
|---|---|
| R1, C1, E1, R2, C2 | Training and probes complete; new test results still sealed |
| CB | Epoch 42 saved, epoch 43 training; arm-specific gate INFO as designed |
| E2, R3, C3, E3 | Queued |
| Runner and watchdog | Both Windows scheduled tasks Running |
| Authors | Confirmed in `camera_ready_authors.json`; acknowledgments/funding pending |
| Poster | Size and QR implemented; real registered-results block still pending |
| Tracked paper PDF | Historical reviewed v9, not the final camera-ready build |
| Overleaf | Last documented version v8; live contents not verified in this audit |

The runner's 17:10 projection puts CB's probe at October 19 ~04:43, E2 at October 20
~11:27, R3 at October 21 ~18:59, C3 at October 23 ~02:51 and E3 at October 24 ~09:35 PDT.
These are estimates, not deadlines or evidence of completion; re-read live state.

Only complete on-time RANDOM/CENTROID/ENVELOPE seed indices enter the primary analysis.
With the projected C3/E3 delay, R3 is descriptive even if it finishes before the freeze.
The primary RANDOM range uses the original anchor plus RANDOM runs from complete eligible
indices; the separate all-completed range does not replace it. Two complete indices are
the expected primary design if E2 and its probe finish on time. Actual stop, seal and probe
timestamps decide eligibility. Never select partial-epoch checkpoints to improve coverage.

## Reviewer-point coverage and remaining action

This table summarizes scientific concerns without reproducing confidential reviews.
Locations are LaTeX labels in `main_submission.tex` or `compact_protocol.tex`.

| Concern | Existing response/evidence | Remaining action and claim boundary |
|---|---|---|
| Training variance | Registered repeated continuations; `sec:replication`, `app:replication` | Finish E2; generate per-seed differences and registered labels. Shared ancestor, not independent initializations; head-seed SD is not training-seed uncertainty. |
| Placement versus budgets | Corrected delivered-mask replay, `tab:geometry`; CB sampler and logged budgets | Audit CB versus CENTROID shadow-mask unique targets, loss slots and visible context; report C1-CB and CB-R1. One matched comparison; ENVELOPE is not budget-matched. |
| Adaptive test reuse | Frozen endpoint, sealed predictions, registered analysis; `sec:limits` | Unseal only after freeze. Same previously inspected split; no claim of an untouched external test or confirmatory replication. |
| Pooling sensitivity | Extra pooling outputs collected by the existing probes | Report all registered variants; retain mean/mean as primary regardless of which scores highest. |
| CENTROID failures | Stratified guide-agreement audit; `app:guides` | Retain subgroup/low-signal limitations. Disagreement with MIRAGE is not a ground-truth failure rate. |
| Segmentation guidance quality | Envelope plausibility audit and GOALS provenance; `app:guides` | Keep GOALS Dice explicitly in-domain; FairVision has no layer labels for anatomical accuracy. |
| Strategy clarity | All-five-strategy figure, `fig:target_selection`; `tab:strategies`, `eq:centroid` | Final figure/text consistency check; no replacement masking method needed. |
| Narrow scope/external baseline | Limits and future work in `sec:limits` | Defer RETFound/new dataset. Registered MD regression remains a secondary severity endpoint on the same cohort. |
| Earlier experiment history | Original-run table separated from replication; excluded-run/provenance discussion | Preserve ANATOMY-V1 provenance and ANATOMY-V2 early-stop disclosures; never fill its missing endpoint. |

The corrected mask replay and guide audit are already incorporated. Their evidence lives in
`autopilot/investigations/camera_ready_20261008/{mask_audit,guide_quality}`. Reuse that
evidence and the reviewed figures rather than repeating expensive analyses.

## Before the freeze

- [x] Reconcile operational documents and distinguish historical plans/PDFs from current work.
- [x] Map reviewer concerns to existing source and remaining outputs.
- [x] Check manuscript structure and numeric coverage without opening new test results.
- [x] Confirm required MD metadata, CB benchmark evidence and Tectonic executable exist.
- [ ] Receive the authors' exact acknowledgments/funding/compute disclosure, or explicit none.
- [ ] Record completed CB and E2 gates and provenance as their probes finish.

October 18 checks: manuscript checker found zero undefined macros, duplicate definitions,
missing citations or dangling references. It **failed** on reachable placeholders, as
expected before integration. Numeric verification checked 12 required historical AUCs,
with no audit errors or mismatches, but **failed** on five unresolved placeholder macros:
`CRCell`, `CRControlResultsPending`, `CRPoolingResultsPending`, `CRSeedResultsPending`,
and `ph`. These are release blockers, not waivers. The full local audit is retained as
`readiness_20261018_numbers.json` in the operator's handoff outputs.

Prerequisite existence is not a completed real-data run. Do not run the actual post-freeze
pipeline, regenerate result macros, accept numeric receipts or remove placeholders early.

## Freeze, review and release

1. At/after October 22 08:00 PDT, recheck campaign state and completion evidence. Run:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File autopilot\cr_postfreeze.ps1 -AllowIncomplete
   ```

   This is phase 1: it writes generated result artifacts/receipts and a scratch review
   package, while leaving the manuscript text unchanged. Review its recorded errors,
   not just the outer process exit code. Let the existing training queue continue.

2. Independently verify the real results and outcome wording, including eligibility,
   primary versus descriptive seed indices, matched-control budgets, uncertainty labels,
   pooling and the registered MD analysis. Use the previously requested gpt-6-astra numeric
   verification. Compare the scratch paper, poster and OpenReview TL;DR with the same source.
   Obtain the author's review before final integration; this is review of actual results.

3. Fix the candidate's completion snapshot before phase 2. Do not wait for E3 to begin
   review. A late-run refresh is optional before final integration and must repeat the
   affected checks; it never promotes late runs to the primary analysis. After integration,
   use the documented recovery procedure before attempting a fresh integration.

4. Integrate through the existing runbook after review. Resolve the actual context-bound
   receipt findings by reading their old/new text; do not blindly reuse historical receipt
   IDs. Fill acknowledgments only from author-supplied information. Run numeric, asset,
   manuscript and camera-ready release gates; all must pass for the final release.

5. Verify the resulting PDF is within 10 main pages, has the confirmed authors/workshop
   notice, no unresolved results or references, and embedded fonts. The previous worst-case
   synthetic rehearsal ended at 9.93 pages; this is not proof of the real-result page count.

6. Rebuild the poster from the same verified real result bundle. Check a single 610 x 850 mm
   page, no clipping/overflow, correct values and labels, no pending or synthetic banners,
   and the working QR. Inspect actual-size crops of the critical results and figure labels;
   existing smallest labels are about 8.8–10.5 pt, so fit alone does not prove readability.

7. Inspect the release diff, build the final package, check current Overleaf contents and
   collaborator edits, then use the manifest-gated sync. Push the approved release on
   `poster-ready`; update the QR destination's PDF/status links together and verify them
   while signed out. Supply the PDF and matching title/abstract/TL;DR for the October 24
   upload target. Keep the branch available for printed QR codes.

Do not print the current placeholder poster. Final printing and presenter logistics can
follow the verified release; no purchase or registration change is part of this checklist.
