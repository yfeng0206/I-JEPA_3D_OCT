# TIMELINE AND CRITICAL PATH
Autopilot run started: 2026-08-22 18:24 PDT (America/Los_Angeles)
Operator: away. Mode: autonomous execution.
Target: complete internal mock paper package + validated final ZIP for GenAI4Health @ NeurIPS 2026.

## Absolute boundary (restated, enforced)
NO representation pretraining of any kind. COVER-0.21 stays frozen at its existing epoch-73
checkpoint. No encoder / target-encoder / EMA-teacher / predictor / MIRAGE-backbone updates.
Permitted GPU work: inference, frozen-feature extraction, frozen linear probes, downstream-head-only
training, embedding analysis, figures. Encoder-hash assertion required before/after every head run.

---

## Ground truth established in Phase 0 (all [MEASURED] 2026-08-22 18:25-18:40 PDT)

| Fact | Value | Consequence |
|---|---|---|
| GPU | RTX 3090, 0% util, 879 MiB / 24576 MiB, 56 C | free for eval work |
| System RAM | 31.9 GB total, 20.4 GB free | supports 1 GPU job + agents |
| C: free | **19.8 GB (8.5%)** | **BELOW 25 GB stop threshold - all large output routed to D:** |
| D: free | 694 GB (37.3%) | primary output volume |
| COVER-0.21 `-last.pth.tar` | **epoch 73** | the only COVER checkpoint we may evaluate |
| Saved `test_predictions.npz` | **19 runs** | CIs / ROC / DeLong computable with ZERO GPU |
| Test vector identity | **identical across all 19** (n=3000, 1466 pos) | **paired statistics are valid** |
| Probe protocol | **exactly 1 distinct protocol** across all 19 | **protocol-matched comparison available** |

The last two rows are the pivotal discovery of Phase 0. The strongest reviewer objection against the
current draft was that arms were compared under mixed protocols and without confidence intervals.
Both are fixable from data already on disk, with no GPU and no retraining.

### Definitive arm mapping (sourced from `results.json -> config.model.encoder_checkpoint`, not filenames)

| Arm | Pretrain run dir | Probe dirs available |
|---|---|---|
| ancestor (shared fork) | `fairvision-glaucoma/checkpoint-ep25` (random_posfix) | ep25 |
| random (COVER-family control) | `cover_random_ep25` | ep30, ep50, ep75, ep100 |
| envelope | `patch_mirage_envelope` | ep30, ep50, ep75, ep100 |
| anatomy-v1 | `patch_mirage_anatomy` | ep30 |
| anatomy-v2 / bridge / blob | `anatomy_v2_ep25` then `blob_resume_ep56` | ep35, ep40, ep50, ep75, ep92 |
| COVER f=0.21 | `cover_f021_ep25` | ep27, ep30, ep34, ep50 (+ep73 to be probed) |

NOTE: probe dirs named `frozen_meanpool_mirage_*` are the **envelope** arm
(`config.model.encoder_checkpoint -> patch_mirage_envelope/...`). The directory name is misleading.
This must not be mis-transcribed into the manuscript.

---

## Phase plan, durations, and dependencies

Legend: [CPU] no GPU needed | [GPU] requires GPU | [EXT] external research | [CP] on critical path

| # | Phase | Est. wall clock | Type | Depends on | Concurrency |
|---|---|---|---|---|---|
| P0 | Setup, inventory, resource monitor | 0.5 h (DONE) | CPU | - | - |
| P1 | Paired statistics from saved predictions | 3-5 h | CPU [CP] | P0 | runs with P2 |
| P2 | Style-file verification + related-work research | 3-6 h | EXT | P0 | runs with P1, P3 |
| P3 | COVER ep73 frozen probe | 1.5-2.5 h | GPU | P0 | runs with P1, P2 |
| P4 | Frozen feature cache (all arms) | 3-5 h | GPU [CP] | P3 | serialized on GPU |
| P5 | Label-efficiency curves from cached features | 2-4 h | CPU/GPU | P4 | runs with P6 |
| P6 | Embedding / representation-structure analysis | 3-5 h | CPU | P4 | runs with P5 |
| P7 | Subgroup / fairness analysis | 2-4 h | CPU | P1 | runs with P5, P6 |
| P8 | Figure + table regeneration | 3-4 h | CPU [CP] | P1,P5,P6,P7 | - |
| P9 | Manuscript rewrite and integration | 6-10 h | CPU [CP] | P8 | - |
| P10 | Mock review round 1 (3 reviewers + meta) | 3-5 h | EXT [CP] | P9 | - |
| P11 | Revision pass | 4-6 h | CPU [CP] | P10 | - |
| P12 | Mock review round 2 (final) | 2-4 h | EXT [CP] | P11 | - |
| P13 | Final compile, validation, ZIP | 1-2 h | CPU [CP] | P12 | - |

### Critical path
P0 -> P1 -> P8 -> P9 -> P10 -> P11 -> P12 -> P13
GPU phases P3/P4 feed P5/P6 which feed P8, but P8 can proceed on P1 alone if GPU work slips.
This means **no GPU failure can block the paper** - a deliberate design choice.

### Expected completion window
- First regenerated statistics + tables: **2026-08-22 ~23:30 PDT** (+5 h)
- First full compiled PDF of the revised manuscript: **2026-08-23 ~14:00 PDT** (+20 h)
- Mock review round 1 complete: **2026-08-23 ~20:00 PDT** (+26 h)
- Revision complete: **2026-08-24 ~06:00 PDT** (+36 h)
- Final mock-review round complete: **2026-08-24 ~14:00 PDT** (+44 h)
- **Final validated ZIP: 2026-08-24 16:00 PDT to 2026-08-25 12:00 PDT** (+46 h to +66 h)

Stated range accounts for the operator's 2-5 day expectation. The plan targets the early end but
will not terminate early merely because a rough draft exists; the two mock-review rounds and their
revisions are mandatory gates.

### Paper-review schedule
| Round | When | Reviewers | Gate |
|---|---|---|---|
| R1 | after first full compile | 3x GPT-5.6 Sol xhigh (independent) + 1 meta-reviewer | every "fatal" objection must be resolved or explicitly conceded in Limitations |
| R2 | after revision | 2x GPT-5.6 Sol xhigh fresh-eyes + numerical re-verification | no unverified number may remain; no undefined citation; page limit satisfied |

### Verification policy
Every number entering the manuscript is produced by a script that writes a JSON artifact.
All research and all numerical claims receive independent verification by a GPT-5.6 Sol xhigh agent
that re-derives values from the artifacts rather than trusting prose.

---

## Resource budget

| Resource | Capacity | Budgeted | Guard |
|---|---|---|---|
| GPU VRAM | 24576 MiB | <= 12 GB single job | warn 85%, stop new jobs 90% |
| GPU thermal | - | - | warn 84 C, stop job if >=84 C for 120 s |
| System RAM | 31.9 GB | <= 20 GB | warn 80%, serialize 85% |
| C: disk | 19.8 GB free | **control files only (<50 MB)** | already past stop threshold |
| D: disk | 694 GB free | features + figures + checkpoints | warn 15% |
| Concurrent GPU jobs | 1 | 1 | strict, registry-enforced |
| Concurrent subagents | <= 3 | 3 | reduced to 1 under RAM pressure |

## Major risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | `neurips_2026.sty` differs from official 2026 style -> desk reject | med | high | P2 diffs against official download; swap and rebuild if different |
| R2 | C: fills during figure/PDF generation | med | high | monitor active; all bulk output to D:; C: used only for control files |
| R3 | No patient IDs in saved predictions -> cannot cluster bootstrap | med | med | P1 attempts join to test manifest by index; if impossible, report image-level CI and state the clustering limitation explicitly |
| R4 | No demographic columns -> fairness analysis impossible | med | med | P7 checks FairVision metadata first; drops the subgroup claim rather than inventing it |
| R5 | n=1 seed per arm remains unfixable without pretraining | **certain** | high | forbidden to fix; must be stated prominently in Limitations and the claim language weakened to match |
| R6 | COVER-0.21 incomplete at ep73 -> H2 unresolved | certain | high | present COVER as an in-progress trajectory with an explicit projection band, never as a completed result |
| R7 | Subagent stalls silently | med | low | heartbeat ledger, 50%-overrun stall rule, max 2 recoveries then reassign |
| R8 | Feature cache exceeds disk | low | med | write to D:, fp16 storage, per-arm cleanup after use |

## Deviation policy
This file is updated whenever any phase deviates from its estimate by more than 20%.
Update history is appended at the bottom of this file.

---

## Update history
- 2026-08-22 18:40 PDT: initial version written after Phase 0 inventory.
