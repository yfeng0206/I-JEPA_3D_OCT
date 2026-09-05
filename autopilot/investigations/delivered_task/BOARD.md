# Delivered-task investigation

Approved author-side engineering investigation and workshop revision.
Branch: `fix/jepa-delivered-task-audit`.
Historical baseline: `de145d7005f57e871bc0181bf58b271775d1d25d`.

## Status

Milestone A complete: baseline preserved; GitHub branches agree at de145d7;
authenticated Overleaf dry-run found all 59 managed files identical.
No collaborator edits need reconciliation at this baseline.

Known-good release snapshot:
`C:\Users\Gary\.copilot\session-state\66ccb9a5-db78-4934-afd2-9f61b6d9c57b\files\delivered_task\baseline_de145d7`.

The pre-existing resource-monitor CSV modification is excluded from our commits.
No automatic main merge, sustained pretraining, or OpenReview submission.

## Ownership

| Role | Model | Scope | State |
|---|---|---|---|
| Coordinator | GPT-6 Astra | Board, source integration, claims, paper, commits and sync | Active |
| Mask/data engineer: repair-masks | GPT-6 Astra high | Masks, guides, dataset/paired transforms, mask regression tests and diagnostics | Complete; critic findings repaired |
| Training/evaluation engineer: repair-training | GPT-6 Astra high | train_patch, helper, model/loss/eval state, training regressions, bounded GPU | Complete; GPU released; fork follow-up reviewed |
| Release engineer: repair-release | GPT-6 Astra high | Release scripts, checkers, dependency/asset manifests, DOCX and sync tests | Implementation handed off; awaiting critic |
| Literature/design: literature-contract | Opus 5 xhigh | Primary methods, ablations, official code, glossary | Complete; narrowed after coordinator critique |
| Independent critic/judge | GPT-5.6 Sol Fast xhigh | Reproduction and evidence adjudication after owner handoff | Pending |
| Numeric coverage: complete-number-coverage | GPT-6 Astra high | Exact statistic bindings, structural-literal classification and evidence-receipt schema | Active |
| Release critic: release-fix-critic | GPT-5.6 Sol Fast xhigh | Read-only review of failure preservation, Word and conflict-aware sync | Active |

One writer per source file. Mask engineer must request changes in train_patch
through training engineer. Only coordinator edits the manuscript, this board,
root state and shared claims. No agent performs commits or remote mutations.

## Evidence rules

Posts live in separate `agents\<agent-id>.jsonl` files. Each identifies model,
role, baseline, commands, files, observations versus interpretation, limitations,
evidence paths and requested peer action. Sensitive case manifests remain outside
public git; use synthetic fixtures and aggregate/public-safe diagnostic outputs.

Findings move from suspected to reproduced, proposed fix, independently checked,
and accepted/rejected/blocked. Existing AUCs describe the historical code.
Corrected-code smoke tests never become a claim of improved downstream AUC.

GPU work requires the one lease in `gpu_lease.json`. No existing training is
resumed. Small fixed-data diagnostics/probes only; sustained pretraining asks
the author separately. Jobs must name an explicit batch/update bound and stop
conditions before launch.

## Initial decisive questions

1. Do cropped guides, image pixels and token positions identify the same tissue?
2. Do configured guided/random branches actually emit targets at full ramp?
3. Does tissue outside target masks survive into encoder context?
4. Do intended targets survive collation with their intended loss weights?
5. Are image, context, target-block and teacher-token ordering aligned?
6. Do online/predictor gradients, EMA and successful optimizer steps correspond?
7. Which historical comparisons remain valid after identifying a defect?
8. What do DSeq-JEPA/MAE ablations actually isolate, versus motivate?

## Accepted progress

- Audit baseline and protected-release manifest published on the working branch
  at `5a47648`; main remains unchanged.
- Literature matrix uses pinned full sources including DSeq-JEPA v4 and official
  code. Its region-selection/order ablation is an analogous factor comparison,
  not our exact OCT configuration. No single-run ablation establishes significance.
- Parent independently confirmed the arXiv v4 metadata and the official
  DSeq region/context function. Highest-ranked saliency context is different
  from indiscriminately hiding all guide-positive tissue.
- Removed misleading exact-analogy, identical-ramp and fixed-CENTROID claims
  from the literature reports before use. Guidelines remain motivation, not
  proof of our mechanism.
- Manuscript draft now has standard title/terms, explicitly scoped comparisons,
  corrected subgroup and paused-replication statements, and a larger standalone
  pipeline schematic. Real masks remain in the readable appendix panel.
- The draft compiles in isolated output: 35 pages total, References on page 9
  below remaining body text, no overfull boxes or undefined references.
  This is not the final release; engineering integration and reader review remain.
- Minor weight-inventory listing defect repaired and three focused regressions
  passed. No weight upload/download performed.
- Citation-authority record now retains a strict title/identifier match for all
  47 cited entries at that source snapshot. Claim support remains a separate
  review obligation; no blanket scientific-correctness certificate is implied.
- Release owner implemented staging and failure preservation with isolated
  regressions. The real release remains blocked while numeric/figure coverage,
  Word regeneration and independent review are completed.
- Training review found an intentional-fork regression in the new exact-resume
  guard. The owner is adding an explicit fork policy rather than silently
  weakening exact continuation checks.
- Explicit exact/fork handling is now implemented; follow-up review found no
  significant issue in that correction. Final integrated regression run remains.
- Coordinator re-aggregation of the saved full-batch rows confirms 576 views,
  573 valid guides, historical 300 target-truncation losses and 55 zero-guide
  contexts. Final occupancy-cell criterion misses are 358 / 320 / 0 for
  historical COVER / exact-prefix v2 / guarded v2, respectively.
  This checks aggregation, not clinical segmentation truth.
- Mask critic identified two remaining diagnostic-safety issues: valid but
  infeasible guide labeling and a baseline-parity helper importing current COVER
  code. Owner is correcting these without reassigning historical AUCs.
- A fresh reader reviewed the integrated manuscript and found five bounded
  wording/method issues, now addressed. New diagnostic macros and the token-map
  illustration distinguish engineering evidence from unmeasured corrected AUC.
- Both mask-critic findings are repaired. The corrected baseline helper isolates
  historical dependencies and detects a deliberately mutated current COVER
  function. All 15 isolated legacy controls pass; scanning 5,400 saved records
  found no changes to their source counts. Sparse/mixed-ramp regressions now
  distinguish invalid guides, infeasibility and genuinely unguided slots.
- Parent integrated regression run: 149 tests passed across training, evaluation
  identity, mask/guide/collation and weight inventory. Release/evidence checks
  are a separate, still-pending acceptance path.
- Reviewed engineering milestone published at `73e0b55` on the working branch.
  Main remains unchanged.
- The strict plot validator exposed a precision mismatch in the trajectory
  figure. Parent fixed the selected null precision, added 11 regressions and
  regenerated only that PNG; every generated table/macro stayed unchanged.
- Seven legacy figure blockers were adjudicated: five unretained attribution
  displays were excluded from the submission with an explicit limitation, and
  two other figures were replaced from exact retained sources under new names.
  All original images remain preserved.
- Six residual literal-reporting issues were corrected without changing primary
  statistics: cached-head reproduction scope, stale geometry agreement count,
  subgroup-specific eligibility, unretained initial-audit count, a rounded
  spread presented as an exact bound, and historical-environment scope.
- Repaired credential-free Overleaf authentication succeeded in a new
  discovery-only dry run. No remote edits/conflicts were observed; nothing was
  pushed. Final publication will use the exact release manifest, not the broader
  discovery list of historical files.

## Two tracks

Workshop: valid historical comparisons, measured delivered-task diagnostics,
explicit limitations, readable terminology and reliable release artifacts.

Subsequent work: corrected multi-arm pretraining where required, not authorized
by this investigation. A major engineering finding does not require rerunning
every arm to finish an honest workshop submission.
