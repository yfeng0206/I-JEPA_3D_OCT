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
| Mask/data engineer: repair-masks | GPT-6 Astra high | Masks, guides, dataset/paired transforms, mask regression tests and diagnostics | Active |
| Training/evaluation engineer: repair-training | GPT-6 Astra high | train_patch, helper, model/loss/eval state, training regressions, bounded GPU | Active |
| Release engineer: repair-release | GPT-6 Astra high | Release scripts, checkers, dependency/asset manifests, DOCX and sync tests | Active |
| Literature/design: literature-contract | Opus 5 xhigh | Primary methods, ablations, official code, glossary | Active |
| Independent critic/judge | GPT-5.6 Sol Fast xhigh | Reproduction and evidence adjudication after owner handoff | Pending |

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

## Two tracks

Workshop: valid historical comparisons, measured delivered-task diagnostics,
explicit limitations, readable terminology and reliable release artifacts.

Subsequent work: corrected multi-arm pretraining where required, not authorized
by this investigation. A major engineering finding does not require rerunning
every arm to finish an honest workshop submission.
