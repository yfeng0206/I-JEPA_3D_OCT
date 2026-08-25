# TASK LEDGER

Updated: 2026-08-24T18:11:37-07:00

| id | phase | task | status | owner | started | expected | verified | retries | output |
|---|---|---|---|---|---|---|---|---|---|
| P0-01 | P0 | Inventory checkpoints, predictions, protocols | **done** | coordinator | - | 0.5h | no | 0 | - |
| P0-02 | P0 | Launch resource monitor and control plane | **done** | coordinator | - | 0.2h | no | 0 | - |
| SCHED-01 | P0 | Stop obsolete COVER pretraining monitor schedule (conflicts with no-pretraining boundary) | **done** | coordinator | - | 0.1h | no | 0 | - |
| P1-01 | P1 | Build protocol-matched paired master table from 19 saved prediction sets | **done** | coordinator | 2026-08-22T18:31:55-07:00 | 1.0h | no | 0 | - |
| P1-02 | P1 | DeLong + paired bootstrap CIs for every arm contrast | **done** | coordinator | - | 1.5h | no | 0 | - |
| P1-03 | P1 | ROC / PR / calibration / operating-point metrics per arm | **done** | coordinator | - | 1.0h | no | 0 | - |
| P1-04 | P1 | Locate patient IDs; attempt patient-level clustered bootstrap | **done** | coordinator | - | 1.0h | no | 0 | - |
| P1-05 | P1 | Corrected inventory + family-partitioned paired stats (p1b/p1c) | **done** | coordinator | - | 2.0h | no | 0 | - |
| P10-01 | P10 | Mock review round 1: 3 independent reviewers | **done** | agent | 2026-08-22T19:33:26-07:00 | 3.0h | no | 0 | - |
| P10-02 | P10 | Meta-review and objection triage | **pending** | agent | - | 1.0h | no | 0 | - |
| P11-01 | P11 | Revision pass addressing every R1 objection | **in_progress** | coordinator | 2026-08-22T20:14:57-07:00 | 5.0h | no | 0 | - |
| P12-01 | P12 | Mock review round 2 + numerical re-verification | **pending** | agent | - | 3.0h | no | 0 | - |
| P13-01 | P13 | Final compile, anonymity + citation validation | **pending** | coordinator | - | 1.0h | no | 0 | - |
| P13-02 | P13 | Build and validate final Overleaf ZIP | **in_progress** | coordinator | 2026-08-22T19:29:21-07:00 | 1.0h | no | 0 | - |
| P13-03 | P13 | ZIP builder with 6-check standalone validation | **done** | coordinator | - | 1.0h | no | 0 | - |
| P2-01 | P2 | Diff neurips_2026.sty against official NeurIPS 2026 style | **done** | agent | 2026-08-22T18:31:54-07:00 | 0.5h | no | 0 | - |
| P2-02 | P2 | Related-work + citation research for anatomy-guided masked SSL | **done** | agent | 2026-08-22T18:31:54-07:00 | 3.0h | no | 0 | - |
| P2-03 | P2 | Verify P2-01/P2-02 findings independently | **pending** | agent | - | 1.0h | no | 0 | - |
| P2-04 | P2 | Submission-requirements audit incl. NeurIPS checklist requirement | **done** | agent | 2026-08-22T19:39:40-07:00 | 1.0h | no | 0 | - |
| P3-00 | P3 | fp32 re-probe envelope ep50/75/100 (confound fix, blocks headline claim) | **in_progress** | coordinator | 2026-08-22T18:44:18-07:00 | 4.0h | no | 0 | - |
| P3-01 | P3 | Frozen linear probe of COVER-0.21 epoch-73 checkpoint | **pending** | coordinator | - | 2.0h | no | 0 | - |
| P3-02 | P3 | Assert encoder frozen + hash unchanged before/after probe | **pending** | coordinator | - | 0.2h | no | 0 | - |
| P3-03 | P3 | Trim unnecessary ep75 fp32 probes after ep50 completes | **in_progress** | coordinator | 2026-08-22T23:41:02-07:00 | 0.2h | no | 0 | - |
| P4-01 | P4 | Extract and cache frozen features for all arms | **pending** | coordinator | - | 4.0h | no | 0 | - |
| P5-01 | P5 | Label-efficiency curves (1/5/10/25/100%) from cached features | **pending** | coordinator | - | 3.0h | no | 0 | - |
| P5-01b | P5 | Label-efficiency curves from cached features (written, gated on RAM, runs after Phase C) | **blocked** | coordinator | - | 1.0h | no | 0 | - |
| P6-01 | P6 | Embedding structure: PCA/UMAP, class separation, Cohen d | **done** | coordinator | - | 4.0h | no | 0 | - |
| P6-02 | P6 | Integrate existing class-relations evidence into the paper as a mechanism hypothesis | **pending** | coordinator | - | 1.0h | no | 0 | - |
| P7-01 | P7 | Subgroup / fairness analysis on FairVision metadata | **done** | coordinator | 2026-08-22T18:44:18-07:00 | 3.0h | no | 0 | - |
| P7-02 | P7 | Subgroup gap trends across 21 probes (gender/race/ethnicity/language/marital/age/severity) | **done** | coordinator | - | 1.0h | no | 0 | - |
| P8-01 | P8 | Regenerate all figures and tables from verified artifacts | **done** | coordinator | 2026-08-22T19:07:08-07:00 | 3.5h | no | 0 | - |
| P9-01 | P9 | Rewrite manuscript around protocol-matched paired evidence | **in_progress** | coordinator | 2026-08-22T19:24:54-07:00 | 8.0h | no | 0 | - |
| P9-02 | P9 | Compile first full PDF and check page limit | **done** | coordinator | - | 0.5h | no | 0 | - |
| PC-01 | P9 | Citation integrity check: 40/40 keys resolve, 0 malformed | **done** | coordinator | - | 0.2h | no | 0 | - |
| PV-01 | PV | Independent numerical audit of all statistics (GPT-5.6 Sol xhigh) | **done** | agent | 2026-08-22T19:30:17-07:00 | 1.5h | no | 0 | - |

## Counts
- pending: 9
- in_progress: 5
- blocked: 1
- done: 20
- failed: 0
- skipped: 0
