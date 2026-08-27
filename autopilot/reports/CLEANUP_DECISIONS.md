# CLEANUP DECISIONS

Companion to `autopilot/reports/REPO_CLEANUP_AUDIT.md`. This document resolves
bucket E and verifies bucket C. It assigns every file a tier and lists the exact
commands to execute.

Prepared: 2026-08-26. Branch `docs/background-signal-findings`, HEAD `c679ae1`
at the start of the scan; MEASURED `021ac9b` "Remove two unsupportable claims and
report a measured disparity" landed while this document was being written, which
committed `autopilot/reports/G1_REPLICATION.md` and `REBUTTAL.md`. No file
classified below changed in that commit.

**This investigation was READ ONLY.** Nothing was moved, deleted, renamed or
modified. No command below was run. No process was touched. Nothing under
`D:\jepa_phase0\` was read except three directory listings (names and sizes
only); nothing under `C:\jepa_data` was accessed at all.

Evidence labels: **MEASURED** = read directly from a file, a command's output or
a process table. **INFERRED** = a judgement derived from that evidence.
**PENDING** = not established; the missing step is named.

---

## 0. THE TIER POLICY APPLIED HERE

| tier | meaning | test |
|---|---|---|
| KEEP | stays exactly where it is | load-bearing, or investigation could not positively clear it |
| ARCHIVE | `git mv` into `archive/`, reversible, never deleted | encodes a CHOICE, an EXPERIMENT or a CITED RESULT |
| REMOVE | deleted | regenerable byte-for-byte, or a byte-identical duplicate with a proven canonical twin |

The deciding test for REMOVE: *could this be reproduced EXACTLY by running
something we still have?* If not, it is ARCHIVE or KEEP.

---

## 1. STATE CHANGES SINCE THE AUDIT WAS WRITTEN

The audit's snapshot is stale in four ways that change its conclusions. All
MEASURED.

1. **Part of the audit's own plan has already been executed.** Commit `5f5b602`
   "repo hygiene: remove identity leaks, fix the collaborator README, untrack
   build artifacts" is an ancestor of HEAD. It performed Stage 1 (the three
   OneDrive/Microsoft path leaks), Stage 2 (`paper/genai4health2026/README.md`),
   and part of Stage 3. `git ls-files paper/dist` now returns only
   `OCT_JEPA_3D_CVPR_2027.zip` and `README.md`.
2. **Two bucket C files are no longer tracked.** Both SUPERSEDED Overleaf zips
   were untracked by `5f5b602` and are now covered by `.gitignore:47`
   (`paper/dist/archive/`). They remain on disk (3.08 MB + 4.29 MB). They are
   out of scope for a *tracked-file* cleanup.
3. **`git ls-files` now returns 876 paths, not 853.**
4. **A pretraining job is executing.** MEASURED from the process table:

   | PID | command |
   |---:|---|
   | 26152 | `D:\jepa_phase0\.venv\Scripts\python.exe -u scripts\chain_replication.py` |
   | 11296 | `... scripts/campaign_supervisor.py --config ...\configs\replication\rep_random_s1234.yaml --stop_a...` |
   | 19360 | `... src/train_patch.py --config ...\configs\replication\rep_random_s1234.yaml` |
   | 1616 | `powershell.exe ... -File .\resource_monitor.ps1 -IntervalSec 45` |

   MEASURED: `D:\jepa_phase0\runs\rep_random_s1234\` exists. This run reaches
   into files the audit classified as scratch. See section 3.2.

### 1.1 A counting discrepancy in the audit

MEASURED. The audit's bucket C subtotals sum to 188 (42 + 29 + 16 + 101), but
the files actually **named** in sections 6.1 to 6.4 sum to 198. Section 6.4's
heading says 101 while its own bullet list names 111 scripts; section 14 Stage 7
says "the 101 files listed in section 6.4" and section 16 limitation 1 says
"the 103 scripts". Of the 198 named files, 196 are still tracked (the two
untracked zips from item 2 above). **All 196 were enumerated and checked here**,
so this document covers a superset of the audit's bucket C, not a subset.

---

## 2. BUCKET E RESOLVED (18 files, 3.21 MB)

Method per file: `git log --follow`, `git grep` for the basename and full path
across all 876 tracked files, opening the file, and for the configs a
name-listing of `D:\jepa_phase0\runs\` to establish whether a run exists.

| # | file | what it is | evidence gathered | tier | justification |
|---:|---|---|---|---|---|
| 1 | `configs/frozen_meanpool_bridge_ep41.yaml` (645 B) | frozen mean-pool linear-probe config; `model.encoder_checkpoint` = `D:\jepa_phase0\runs\anatomy_v2_ep25\jepa_patch_mirage-ep41.pth.tar`, `logging.output_dir` = `D:\jepa_phase0\runs\frozen_meanpool_bridge_ep41` | MEASURED: added in `922f512` (2026-08-14) with the COVER-then-RANDOM arm. MEASURED: `git grep frozen_meanpool_bridge_ep41` matches only the config itself and the audit. MEASURED: `D:\jepa_phase0\runs\frozen_meanpool_bridge_ep41\` **exists but contains zero files**, while its siblings ep35/ep40/ep50/ep75/ep92 all exist. INFERRED: the ep41 probe was configured and launched but produced no output. | ARCHIVE | An abandoned probe in a cited series. The config is the only surviving record that ep41 was attempted and yielded nothing, which is itself a fact about how the bridge series was sampled. Not regenerable, so never REMOVE. |
| 2 | `configs/patch_vitb16_ep100_R2_loss.yaml` (2325 B) | dynamic-curriculum ablation: loss-driven mask schedule | MEASURED: sole commit `29d1b67` (2026-05-31) "feat(curriculum): dynamic masking curriculum (R2 loss, R3a intensity, R3b cluster)". MEASURED: named by no tracked file. MEASURED: no `D:\jepa_phase0\runs\` directory matches R2/R3a/R3b. | ARCHIVE | Encodes an experimental choice (three named curriculum variants) with no generator. Cannot be reproduced from anything we have. |
| 3 | `configs/patch_vitb16_ep100_R3a_intensity.yaml` (2102 B) | dynamic-curriculum ablation: intensity-driven | same commit and same negative reference/run evidence as #2 | ARCHIVE | as #2 |
| 4 | `configs/patch_vitb16_ep100_R3b_cluster.yaml` (2824 B) | dynamic-curriculum ablation: cluster-driven | same commit and same negative reference/run evidence as #2 | ARCHIVE | as #2 |
| 5 | `configs/arm_anatomy.yaml` (1583 B) | pretraining config, arm C of the three matched arms; `curriculum.mode: mirage_anatomy`, `logging.folder: D:\jepa_phase0\runs\arm_anatomy` | MEASURED: `autopilot/COVER_AUDIT.md:66` lists `arm_anatomy` in the `mirage_anatomy` row. MEASURED: `scripts/compare_arms.py:39-41` maps arm C to `runs/arm_anatomy`. MEASURED: `D:\jepa_phase0\runs\arm_anatomy\jepa_patch_mirage-log.csv` exists, 1,255,746 B. The arm ran. | ARCHIVE | A real experiment with a surviving training log. `COVER_AUDIT.md` is itself a KEEP file that `HANDOFF.md` points collaborators at by name. |
| 6 | `configs/arm_random_default.yaml` (1591 B) | arm A, stock I-JEPA rectangles, `pred_mask_scale: 0.15-0.2` | MEASURED: `COVER_AUDIT.md:66`. MEASURED: no `runs/arm_random_default` on D:; MEASURED `compare_arms.py:37-38` maps arm A to `runs/patch_mirage_anatomy` instead. INFERRED: the arm-A run was satisfied by reusing an existing run rather than executing this config. | ARCHIVE | The config records the *intended* arm-A specification, which is what makes the budget-matching argument checkable. It records a choice even though it was not executed under this name. |
| 7 | `configs/arm_random_matched.yaml` (1594 B) | arm B, rectangles with `pred_mask_scale` lowered to `0.055-0.075` | MEASURED: the only difference from #6 is `pred_mask_scale`. MEASURED: `compare_arms.py:1-12` states "armB is the arm that isolates SHAPE. Without it, anatomy differs from random in both WHERE it masks and HOW MUCH". MEASURED: `D:\jepa_phase0\runs\arm_random_matched\` exists. | ARCHIVE | This file *is* the budget-matched control. Its `pred_mask_scale` values are the numerical content of the design that lets the shape claim be attributed to targeting. Losing it loses the control. |
| 8 | `configs/finetune_mirage_ep100.yaml` (1310 B) | fine-tune config for the MIRAGE/envelope ep100 encoder | MEASURED: sole commit `376665d` (2026-08-09) "eval: make downstream fp32 honest, and measure the real VRAM ceiling". MEASURED: named by no tracked file. MEASURED: no matching run directory on D:. INFERRED: a fine-tune arm that was configured during the fp32/VRAM work and not carried into the paper, whose supplementary fine-tune table covers oracle and random only. | ARCHIVE | Records an evaluation that was set up and not reported. Not regenerable. |
| 9 | `configs/archive/patch_anatomy_v2.FROZEN.yaml` (1735 B) | a freeze marker beside `configs/patch_anatomy_v2.yaml` | MEASURED: content is identical to `configs/patch_anatomy_v2.yaml` after CRLF normalisation (both sha256 `7a8167f8...`). MEASURED: replaying all 4 commits that touched either path shows the FROZEN copy was created in `922f512` (2026-08-14) *already identical*, and the two have never differed. MEASURED: it is the only file in `configs/archive/`; there is no README there. | **KEEP** | INFERRED: it is a deliberate immutable snapshot taken when the anatomy-v2 policy was locked, and its continuing identity to the live config is the evidence that the live config has not drifted since the freeze. That is the same class of evidence as `results/masking/structural_loss/regen01.json`. It is 1.7 KB and already lives in an archive directory; there is nothing to gain by touching it. |
| 10 | `logs/patch_run3_log.csv` (2,090,945 B) | per-step pretraining log for patch run 3 | MEASURED: byte-identical to `logs/pretraining/run3_patch_log.csv` (sha256 `7ea35bec...`). MEASURED: introduced by `b0e25af` (2026-03-25); `afc8a32` (2026-03-30) "Add training logs, plots, and plotting script" **added** `logs/pretraining/run3_patch_log.csv` as a new 14,059-line file alongside `logs/pretraining/run{1,2,3}_epoch_summary.csv` without deleting the original. MEASURED: `git grep` finds no reference to either path from any tracked file. | **REMOVE** | The directory reorganisation left the original behind. The `logs/pretraining/` copy follows the convention shared by its four siblings and is therefore canonical. This is a byte-identical duplicate whose twin survives in the same repository, so it satisfies the reproduce-exactly test. **Delete `logs/patch_run3_log.csv` only; `logs/pretraining/run3_patch_log.csv` must survive.** |
| 11 | `logs/downstream_unfrozen_d3_s64/train_log_full.csv` (1252 B) | fine-tune training log with a `_full` suffix | MEASURED: byte-identical to `train_log.csv` in the same directory. MEASURED: `train_log.csv` was added by `afc8a32`; `train_log_full.csv` was added later by `cbf08d2` (2026-03-30) "Update report: full fine-tune results, comparison table, next steps", in a commit that also rewrote `README.md`. MEASURED: no tracked file references either path. | ARCHIVE | Technically a byte-identical duplicate, but the `_full` name asserts something the bytes contradict, and the commit that created it simultaneously rewrote the results narrative in `README.md`. The identity may itself indicate that the "full fine-tune" numbers were reported from the wrong log. Under "when in doubt, archive", 1.2 KB is not worth the risk of destroying that trace. |
| 12 | `paper/genai4health2026/research/mechanism_data.json` (14,116 B) | machine-readable evidence table behind `research/mechanism.md` | MEASURED: the file declares `schema_version`, `claim_labels` (MEASURED/INFERRED/ASSUMED) and `composition_ep50.rows` keyed by `arm`. MEASURED: `research/mechanism.md` cites `[S1: rows[arm=random]]`, `[S1: rows[arm=oracle]]`, `[S1: rows[arm=cover_f021]]`, `[S1: rows[arm=envelope]]`, `[S1: rows[arm=blob]]` at lines 17, 101, 102 and throughout its composition table. The `rows[arm=...]` selector syntax matches this file's structure exactly. | **KEEP** | It is source [S1] of `mechanism.md`, which the paper's mechanism section rests on. The audit's "named by no tracked file" is a false negative: it is cited by *selector*, not by filename. This is load-bearing evidence, not an uncertain file. |
| 13 | `results/all_train_logs.json` (56,733 B) | aggregated per-epoch downstream training logs, keyed by arm label e.g. `"ImageNet->SSL ep32 (MLP, d=3)"` | MEASURED: sole commit `8bd6458` (2026-04-06) "Restructure docs: experiment tracking with sub-links, plots, and results data". MEASURED: `git grep all_train_logs` matches only the audit. MEASURED: no tracked script writes it. | ARCHIVE | An aggregate of runs from April 2026 with no surviving generator in the repository. Cannot be reproduced. |
| 14 | `results/summary/oracle_summary.png` (86,264 B) | headline oracle-vs-random comparison figure | MEASURED: `scripts/plot_oracle_comparison.py:8,93,94` names and writes it. MEASURED: that script hardcodes `ORACLE_F = [0.8740, 0.8836, 0.8855]`, `RANDOM_F = [0.8641, 0.8723, 0.8746]`, `ORACLE_FT`, `RANDOM_FT` and `FT_STARS = ['***','**','ns']`; there are no data inputs. MEASURED: no doc cites the PNG. | ARCHIVE | The numbers survive in the generator, so the *content* is safe, but a Matplotlib PNG is not guaranteed byte-identical across Matplotlib/FreeType versions, so it fails the reproduce-exactly test. It also depicts the cited 0.8855 oracle result. Archive, do not remove. |
| 15 | `results/unfrozen_train_loss.png` (130,250 B) | training-loss curve, abandoned full-finetune experiment | MEASURED: `docs/README.md:55` lists `results/unfrozen_*.png` under a heading **"Unreferenced Legacy Artifacts"** with the text "3 PNGs - unfrozen training curves from the abandoned full-finetune experiment" and the closing line "These are retained for provenance but are not expected to appear in the paper." MEASURED: added by `f899d4f` (2026-04-08). MEASURED: no generator exists in the repository. | ARCHIVE | The audit's "referenced by no doc" is a false negative: `docs/README.md` references them by glob, and does so specifically to record a deliberate retention decision. They are the only surviving record of an abandoned arm. **If archived, `docs/README.md:55` must be repointed in the same commit.** |
| 16 | `results/unfrozen_training_curves.png` (294,817 B) | combined training curves, same experiment | same as #15 (`f899d4f`; `docs/README.md:55` glob) | ARCHIVE | as #15 |
| 17 | `results/unfrozen_val_auc.png` (118,953 B) | validation AUC curve, same experiment | same as #15, plus MEASURED touched by `503e591` (2026-04-09) "Remove all SLIViT references: not a fair comparison (different data)" | ARCHIVE | as #15. The `503e591` touch means the current image is a *post-retraction* redraw, so it also records a correction. |
| 18 | `paper/genai4health2026/figures/fig1_policies_compact.pdf` (401,373 B) | vector twin of the cited `fig1_policies_compact.png` | MEASURED: `main_submission.tex:255` reads `\includegraphics[width=0.66\linewidth]{fig1_policies_compact.png}` and `SOURCES.md:36` lists `fig1_policies_compact.png`. The `.pdf` is cited by neither. MEASURED: `p13_build_zip.py:77-82` resolves each `\includegraphics` target by trying suffixes `("", ".png", ".pdf", ".jpg")` and stops at the first hit, so with the explicit `.png` in the tex the `.pdf` never ships. MEASURED: the PDF carries `/CreationDate (D:20260822165309-07'00')`, so it is not byte-reproducible. | ARCHIVE | Currently unused, but it is the vector form of the paper's Figure 1 and would become load-bearing the moment the camera-ready switches to vector figures. Not reproducible byte-for-byte. Archive so it can be recovered with one `git mv`. |

### 2.1 Bucket E outcome

| tier | files | bytes | MB |
|---|---:|---:|---:|
| ARCHIVE | 15 | 1,103,616 | 1.05 |
| REMOVE | 1 | 2,090,945 | 1.99 |
| KEEP | 2 | 15,851 | 0.02 |
| **total** | **18** | **3,210,412** | **3.06** |

No bucket E file remains uncertain. Nothing is PENDING.

---

## 3. BUCKET C VERIFIED (196 tracked files)

### 3.1 Verification method

Every one of the 196 files was checked; none was skipped, and this is a census
rather than a spot check. Six independent MEASURED passes:

1. **Enumeration.** The names in audit sections 6.1 to 6.4 were expanded to 198
   paths and intersected with `git ls-files` (876 paths). 196 are tracked; the
   two SUPERSEDED zips are not (section 1).
2. **Inbound reference scan.** Every tracked text file under 8 MB was read (the
   audit report itself excluded) and searched for each candidate's full path,
   basename and, for Python, `import <stem>` / `from <stem> import`.
3. **Outbound artifact scan.** For each `.py`, `.sh` and `.ps1`, every string
   literal ending in `.png .json .csv .npz .pdf .md .txt .svg .yaml` was
   extracted and resolved against the tracked-file index; each resolved artifact
   was then searched for across every tracked `.md` and `.tex`.
4. **Live-run cross-check.** All 196 basenames were searched inside the four
   protected replication drivers (`scripts/chain_replication.py`,
   `scripts/make_replication_configs.py`, `scripts/smoke_replication.py`,
   `configs/replication/*`), `autopilot/reports/G1_REPLICATION.md`,
   `autopilot/reports/REBUTTAL.md`, `HANDOFF.md`, `README.md`,
   `autopilot/refresh_all.py`, `SOURCES.md` and `main_submission.tex`.
5. **Code-usage scan.** All 264 non-bucket-C code files, tracked **and
   untracked**, were scanned for imports of or path references to each of the
   196. Including untracked files is what surfaced the two live-run
   dependencies; a tracked-only scan misses them.
6. **Duplicate and reproducibility checks.** SHA-256 over all 876 tracked files;
   newline-normalised comparison of each backup file against every historical
   revision of its counterpart; PDF trailer inspection for `/ID` and
   `/CreationDate`.

MEASURED: **50 of the 196** name at least one tracked artifact that a current
`.md` or `.tex` cites, making them the sole surviving record of how a cited
number or figure was produced.

MEASURED: **0 of the 196 is a byte-identical duplicate of another tracked
file.** All 11 duplicate sets in the repository lie in bucket E, in the
deliberate `figures/` <-> `results/summary/` mirror, in the intentional
`configs/` <-> `results/pretraining/.../config.yaml` mirror, in the protected
`alpha01.json` / `regen01.json` pair, or in the three empty `__init__.py`
package markers.

MEASURED: PDFs built by tectonic carry a `/ID` array (`main.pdf`,
`main_workshop.pdf`, `main_scaffold.pdf` all do), which is generated per build,
so no tracked PDF is byte-reproducible.

**Consequence: bucket C yields zero REMOVE candidates.** Nothing in it satisfies
the reproduce-exactly test. The audit proposes `git rm` for six bucket C paths
plus `git rm -r results/archive`; every one of those is corrected below.

### 3.2 Mislabelled files (9 findings, covering 24 files)

These are the files that are **not** genuinely scratch.

| # | file(s) | audit said | evidence | correct tier | severity |
|---:|---|---|---|---|---|
| M1 | `paper/genai4health2026/figures/figS5_mask_statistics.png` (201,182 B) | 6.1, one of "26 orphan figures"; Stage 5 `git mv` to archive | MEASURED: `main_submission.tex:796` reads `\includegraphics[width=\linewidth]{figS5_mask_statistics.png}`. MEASURED: `SOURCES.md:37` lists it. MEASURED: `p8_make_assets.py` writes only to `auto/`, so it is **not** regenerated by `refresh_all.py`. MEASURED: `p13_build_zip.py` ships only cited figures, so this one ships. | **KEEP** | **Critical.** Moving it breaks the live submission build and the Overleaf bundle. The `.pdf` twin is genuinely unused; the `.png` is not. The audit's own Stage 5 guard loop would have printed a hit and told the operator to STOP, but the classification is wrong. |
| M2 | `autopilot/run_guarded_probe.py` (bucket C 6.2) | "none is invoked by `refresh_all.py`" | MEASURED: `scripts/chain_replication.py:217` executes `[PY, "-u", str(REPO / "autopilot" / "run_guarded_probe.py"), ...]` and line 34 documents "Probing is delegated to autopilot/run_guarded_probe.py, which pins the fp32...". MEASURED: `chain_replication.py` is running now as PID 26152. MEASURED: `G1_REPLICATION.md:193` and `REBUTTAL.md:516` both describe it as the encoder-hash-verifying probe runner. | **KEEP** | **Critical.** Moving it aborts the probe stage of the run currently executing. |
| M3 | `scripts/download_weights.py` (bucket C 6.4) | "development aid" | MEASURED: `scripts/make_replication_configs.py:51` cites `scripts/download_weights.py --ancestor-ep25` as the provenance of the locked ancestor. MEASURED: `G1_REPLICATION.md:36` and `:207` record the same. MEASURED: the file is **modified in the working tree right now** (`git status`: ` M scripts/download_weights.py`), and the diff adds `--ancestor-ep25`, `ANCESTOR_SHA256 = e5ad5b0c...` and a fatal hash check. | **KEEP** | **Critical.** It is being extended for the run in progress. |
| M4 | `autopilot/resource_monitor.ps1`, `autopilot/PROCESS_REGISTRY.csv` | 6.2, "meaningless off this machine" | MEASURED: process table shows PID 1616 = `powershell.exe -File .\resource_monitor.ps1 -IntervalSec 45`. MEASURED: `git status` shows both `autopilot/PROCESS_REGISTRY.csv` and `autopilot/RESOURCE_MONITOR.csv` modified, i.e. actively appended. | **KEEP** | High. Moving a script or an output file out from under a running process. Revisit after the chain finishes. |
| M5 | `paper/genai4health2026/main.tex` (35,543 B) | 6.1, "an older full draft"; Stage 4 `git mv` | MEASURED: `autopilot/reports/G1_REPLICATION.md:108-110` reads "MEASURED. `paper/genai4health2026/main.tex` line 508 reads `& oracle ep100 & 0.8854852 \\`, i.e. the paper's own source maps the value to the run named `oracle`." That is step 2 of the three-step proof identifying CENTROID with the headline 0.8854852 result. | **KEEP** | High. `autopilot/reports/**` is protected, so the citation cannot be repointed. Moving `main.tex` breaks a live evidence chain by exact line number. `main.pdf` and `main.bbl` are not cited and stay ARCHIVE. |
| M6 | `paper/genai4health2026/main_submission.tex.bak_autopilot` (37,416 B), `references.bib.bak_premerge` (26,014 B) | 6.1; Stage 4 `git rm`, on the reasoning "git history holds the prior revisions, which is what a backup file is for" | MEASURED: after newline normalisation, the `.bak_autopilot` content matches **none of the 41 historical revisions** of `main_submission.tex`, and `.bak_premerge` matches **none of the 2 revisions** of `references.bib`. Both capture uncommitted intermediate states. | **ARCHIVE** | High. The stated justification for deletion is false. `git rm` here destroys 63 KB of content that exists nowhere else, including the pre-`merge_bib.py` bibliography that is the only way to audit what that merge did. |
| M7 | `results/archive/**` (14 files, 1,840,202 B) | 6.3; Stage 8 `git rm -r results/archive` | MEASURED: `docs/README.md:43-58` carries a section titled "Unreferenced Legacy Artifacts" that names `results/archive/` -- "13 PNGs - early pretraining, frozen-probe, and normfix plots from before the current three-arm design" -- and ends "These are retained for provenance but are not expected to appear in the paper." MEASURED: `results/archive/README.md` says "Kept for historical traceability." | **ARCHIVE, in place** | High. A current doc records a deliberate decision to retain exactly these files. `git rm -r` reverses a documented decision and orphans a `docs/README.md` row. They already sit in an archive directory; no move is needed. |
| M8 | `scripts/coverage_probe.py` (bucket C 6.4) | "answered one-off question" | MEASURED: `autopilot/reports/REBUTTAL.md:590` states "`scripts\coverage_probe.py` does not include a CENTROID arm" and `:607` proposes "**Optional appendix, only if measured before freeze:** extend `scripts\coverage_probe.py` to all..." | **KEEP** | Medium. It is the named subject of an open, pre-freeze rebuttal action item. Archive it after the item is closed, not before. |
| M9 | 50 scripts that are the sole named generator of a doc-cited artifact -- including `scripts/plot_pretraining.py` (12 cited artifacts), `scripts/baselines_eval.py` (6), `scripts/plot_results.py` (4), `scripts/elementwise_gate_probe.py` (4), `scripts/compare_arms.py`, `scripts/variable_k_policies.py`, `scripts/variable_k_probe.py`, `scripts/demo_collation_fix.py`, `scripts/archive/plot_normfix_results.py` (6) | 6.4 "superseded plotters", "development aids", "answered one-off questions" | MEASURED example: `scripts/compare_arms.py` is the only tracked file that writes `results/masking/arms/{arms.json,arms.png,train_val_separation.png}`, and `docs/experiments/masking/comparison.md` and `engineering_notes.md` both cite `results/masking/arms`. MEASURED example: `scripts/variable_k_policies.py` writes `results/masking/variable_k/policies.json`, which is cited by `HANDOFF.md`, `SOURCES.md`, `main_submission.tex` and 30 other files. MEASURED example: `scripts/demo_collation_fix.py` writes `results/masking/collation/collation.json`, cited by 30 files including `main_submission.tex`. | **ARCHIVE** (correct tier, wrong label) | Medium. The audit's Stage 7 already archives rather than deletes these, so no data is at risk. The label "scratch" is wrong: each is the only record of how a cited number was produced, which is exactly the ARCHIVE criterion. Recorded so nobody later "simplifies" Stage 7 into a `git rm`. |

Also corrected: `autopilot/RESUME_COMMAND.txt` is cited by
`G1_REPLICATION.md:317` ("carries a prior-session constraint reading...").
It stays ARCHIVE, not REMOVE, and the citation should be repointed if it moves.

### 3.3 Files positively cleared as genuinely scratch

MEASURED: the remaining 172 files have no inbound reference from any tracked
file other than the audit itself, name no doc-cited artifact, are imported by no
module, and appear in no live driver. They are still **ARCHIVE, not REMOVE**,
because none of them is reproducible byte-for-byte: they are hand-written
scripts, hand-written LaTeX fragments, point-in-time status snapshots, or
non-deterministic PDFs. "Unreferenced" is not the same as "regenerable", and
only the second permits deletion.

### 3.4 Bucket C outcome

| tier | files | bytes | MB |
|---|---:|---:|---:|
| ARCHIVE | 189 | 11,415,072 | 10.89 |
| KEEP (mislabelled, see 3.2) | 7 | 261,339 | 0.25 |
| REMOVE | 0 | 0 | 0.00 |
| **total** | **196** | **11,676,411** | **11.14** |

ARCHIVE split by audit group: 6.1 = 38 files / 8,501,954 B; 6.2 = 26 files /
154,088 B; 6.3 = 16 files / 1,854,101 B (no move required, see M7);
6.4 = 109 files / 904,929 B.

---

## 4. EXECUTION COMMANDS

**PROPOSAL ONLY. NONE OF THIS WAS RUN.**

Prerequisite: the chain in section 1 item 4 must have finished, or at minimum
Stage A must be skipped for the four files in M2/M3/M4. Verify with
`Get-CimInstance Win32_Process -Filter "Name like '%python%'"` and confirm no
`chain_replication.py`, `campaign_supervisor.py` or `train_patch.py` is running.

```powershell
# =========================================================================
# STAGE 0 - safety net
# =========================================================================
cd C:\Users\Gary\Desktop\jepa
git status --porcelain      # commit or stash first; must be empty
git tag pre-cleanup-decisions-2026-08-26
git checkout -b chore/repo-cleanup

New-Item -ItemType Directory -Force -Path archive\paper_superseded_2026-08
New-Item -ItemType Directory -Force -Path archive\autopilot_run_2026-08
New-Item -ItemType Directory -Force -Path archive\scripts_oneoff_2026-08
New-Item -ItemType Directory -Force -Path archive\configs_2026-08
New-Item -ItemType Directory -Force -Path archive\results_legacy_2026-08
```

### 4.1 TIER REMOVE

Two items only. Nothing else in the repository qualifies.

```powershell
# R1. Byte-identical duplicate left behind by the 2026-03-30 logs/ reorg.
#     GUARD - this must print two identical hashes before you delete anything:
Get-FileHash logs\patch_run3_log.csv, logs\pretraining\run3_patch_log.csv |
    Format-Table Hash, Path
#     Expected: 7EA35BEC... twice.  If they differ, STOP.
git rm logs/patch_run3_log.csv
git commit -m "chore: drop duplicate patch run3 log superseded by logs/pretraining/"

# R2. Untracked build cruft.  Already covered by .gitignore:1-2, so this
#     touches the working tree only and cannot change any commit.
Get-ChildItem -Path . -Directory -Recurse -Filter __pycache__ |
    Where-Object { $_.FullName -notlike '*\.git\*' } |
    Remove-Item -Recurse -Force
```

### 4.2 TIER ARCHIVE

#### 4.2.1 Superseded manuscript material -> `archive\paper_superseded_2026-08\` (38 files)

Guard first. This must print nothing:

```powershell
foreach ($f in @('fig1_crop_defect','fig1b_context_excision','fig2_composition_vs_auc',
                 'fig3_cover_floor_dose_response','fig4_auc_trajectories',
                 'fig5_zero_anatomy_example','fig6_subgroup_disparity',
                 'figS1_background_matters','figS2_inverted_u','figS3_collapse_mechanism',
                 'figS4_coverage_floor','fig_auc_trajectories_v2','main_scaffold',
                 'main_workshop')) {
    git grep -n -- $f paper/genai4health2026/main_submission.tex
}
# Expect: NO output.  Note that figS5_mask_statistics is deliberately absent
# from this list because main_submission.tex:796 includes it - see M1.
```

```powershell
git mv paper/genai4health2026/main_scaffold.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/main_scaffold.tex archive/paper_superseded_2026-08/
git mv paper/genai4health2026/main.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/main.bbl archive/paper_superseded_2026-08/
git mv paper/genai4health2026/main_workshop.tex archive/paper_superseded_2026-08/
git mv paper/genai4health2026/main_workshop.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/main_workshop.bbl archive/paper_superseded_2026-08/
git mv paper/genai4health2026/main_submission.tex.bak_autopilot archive/paper_superseded_2026-08/
git mv paper/genai4health2026/references.bib.bak_premerge archive/paper_superseded_2026-08/
git mv paper/dist/OCT_JEPA_3D_CVPR_2027.zip archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig1_crop_defect.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig1_crop_defect.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig1b_context_excision.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig1b_context_excision.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig2_composition_vs_auc.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig2_composition_vs_auc.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig3_cover_floor_dose_response.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig3_cover_floor_dose_response.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig4_auc_trajectories.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig4_auc_trajectories.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig5_zero_anatomy_example.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig5_zero_anatomy_example.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig6_subgroup_disparity.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig6_subgroup_disparity.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/figS1_background_matters.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/figS1_background_matters.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/figS2_inverted_u.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/figS2_inverted_u.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/figS3_collapse_mechanism.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/figS3_collapse_mechanism.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/figS4_coverage_floor.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/figS4_coverage_floor.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/figS5_mask_statistics.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig_auc_trajectories_v2.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig_auc_trajectories_v2.png archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig_geometry_panel.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig_precision_paradox.pdf archive/paper_superseded_2026-08/
git mv paper/genai4health2026/figures/fig_specificity_ladder.pdf archive/paper_superseded_2026-08/
git commit -m "chore: archive superseded manuscript variants, orphan figures and unused vector twins"
```

Not in this list, deliberately: `paper/genai4health2026/main.tex` (M5) and
`paper/genai4health2026/figures/figS5_mask_statistics.png` (M1).

Follow-up edit required in the same branch: `paper/genai4health2026/EVIDENCE.md`,
`research/OUTLINE.md`, `research/critique.md`, `research/draft_notes.md`,
`research/subgroup_findings.md` and `research/verify_sections_4_6.md` cite the
archived figure basenames; `paper/genai4health2026/scripts/make_figures.py`,
`make_story_figures.py` and `make_fairness_figure.py` still write to
`figures/`. Repoint them or accept that a rerun recreates the orphans.

#### 4.2.2 Autopilot run residue -> `archive\autopilot_run_2026-08\` (26 files)

```powershell
git mv autopilot/gpu_queue.py archive/autopilot_run_2026-08/
git mv autopilot/gpu_queue2.py archive/autopilot_run_2026-08/
git mv autopilot/chain_queues.py archive/autopilot_run_2026-08/
git mv autopilot/sequencer.py archive/autopilot_run_2026-08/
git mv autopilot/seed_tasks.py archive/autopilot_run_2026-08/
git mv autopilot/state.py archive/autopilot_run_2026-08/
git mv autopilot/findings_delta.py archive/autopilot_run_2026-08/
git mv autopilot/watch_and_refresh.py archive/autopilot_run_2026-08/
git mv autopilot/queue_ep75_null.py archive/autopilot_run_2026-08/
git mv autopilot/trim_ep75.py archive/autopilot_run_2026-08/
git mv autopilot/state.json archive/autopilot_run_2026-08/
git mv autopilot/RUN_STATE.json archive/autopilot_run_2026-08/
git mv autopilot/SUBAGENT_HEARTBEATS.jsonl archive/autopilot_run_2026-08/
git mv autopilot/AGENT_STATUS.md archive/autopilot_run_2026-08/
git mv autopilot/CURRENT_STATUS.md archive/autopilot_run_2026-08/
git mv autopilot/RESUME_COMMAND.txt archive/autopilot_run_2026-08/
git mv autopilot/TASK_LEDGER.md archive/autopilot_run_2026-08/
git mv autopilot/TIMELINE_AND_CRITICAL_PATH.md archive/autopilot_run_2026-08/
git mv autopilot/appendix_subgroup_new.tex archive/autopilot_run_2026-08/
git mv autopilot/ft_block.tex archive/autopilot_run_2026-08/
git mv autopilot/tab_contrasts.tex archive/autopilot_run_2026-08/
git mv autopilot/fig_paradox_block.tex archive/autopilot_run_2026-08/
git mv autopilot/check_citations.py archive/autopilot_run_2026-08/
git mv autopilot/merge_bib.py archive/autopilot_run_2026-08/
git mv plan.md archive/autopilot_run_2026-08/
git mv submit_slice.sh archive/autopilot_run_2026-08/
git commit -m "chore: archive completed autopilot run drivers and status snapshots"
```

Not in this list, deliberately: `autopilot/run_guarded_probe.py` (M2),
`autopilot/resource_monitor.ps1` and `autopilot/PROCESS_REGISTRY.csv` (M4).

#### 4.2.3 One-time probe and demo scripts -> `archive\scripts_oneoff_2026-08\` (109 files)

```powershell
git mv scripts/budget_mask_k_sweep.py archive/scripts_oneoff_2026-08/
git mv scripts/budget_mask_visualize.py archive/scripts_oneoff_2026-08/
git mv scripts/cover_mask_prototype.py archive/scripts_oneoff_2026-08/
git mv scripts/cover_mask_visualize.py archive/scripts_oneoff_2026-08/
git mv scripts/cover_vs_envelope.py archive/scripts_oneoff_2026-08/
git mv scripts/dilation_prototype.py archive/scripts_oneoff_2026-08/
git mv scripts/dilation_sweep_visual.py archive/scripts_oneoff_2026-08/
git mv scripts/connected_rim_visual.py archive/scripts_oneoff_2026-08/
git mv scripts/connectivity_visual.py archive/scripts_oneoff_2026-08/
git mv scripts/bridge_visual.py archive/scripts_oneoff_2026-08/
git mv scripts/jepa_loop_visual.py archive/scripts_oneoff_2026-08/
git mv scripts/full_pipeline_visual.py archive/scripts_oneoff_2026-08/
git mv scripts/five_way_masking.py archive/scripts_oneoff_2026-08/
git mv scripts/visualize_context_loss.py archive/scripts_oneoff_2026-08/
git mv scripts/shape_knob_sweep.py archive/scripts_oneoff_2026-08/
git mv scripts/variable_k_policies.py archive/scripts_oneoff_2026-08/
git mv scripts/variable_k_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/topk_budget_degeneracy.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_backprop_effect.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_before_after_grid.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_class_balance.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_collation_fix.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_coverage_fixes.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_full_pipeline.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_guided_masking.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_mirage_to_targets.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_pipeline_trace.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_seg_before_after.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_split_fix.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_three_methods.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_v1_adapter_pipeline.py archive/scripts_oneoff_2026-08/
git mv scripts/demo_v2_masking.py archive/scripts_oneoff_2026-08/
git mv scripts/adapter_guardrails.py archive/scripts_oneoff_2026-08/
git mv scripts/adapter_refresh_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/mirage_adapter_feasibility.py archive/scripts_oneoff_2026-08/
git mv scripts/mirage_gap_anatomy.py archive/scripts_oneoff_2026-08/
git mv scripts/mirage_input_scale_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/mirage_method_panels.py archive/scripts_oneoff_2026-08/
git mv scripts/mirage_soft_guide_dump.py archive/scripts_oneoff_2026-08/
git mv scripts/compare_mirage_base_large_512.py archive/scripts_oneoff_2026-08/
git mv scripts/visualize_mirage_base_large_512.py archive/scripts_oneoff_2026-08/
git mv scripts/guide_equivalence_check.py archive/scripts_oneoff_2026-08/
git mv scripts/goals_adapter_sweep.py archive/scripts_oneoff_2026-08/
git mv scripts/goals_visual_compare.py archive/scripts_oneoff_2026-08/
git mv scripts/score_goals_merged.py archive/scripts_oneoff_2026-08/
git mv scripts/build_seg_merged_v2.py archive/scripts_oneoff_2026-08/
git mv scripts/seg_v2_preflight.py archive/scripts_oneoff_2026-08/
git mv scripts/seg_run_status.py archive/scripts_oneoff_2026-08/
git mv scripts/ap_intersection_cost.py archive/scripts_oneoff_2026-08/
git mv scripts/arm_coverage_distributions.py archive/scripts_oneoff_2026-08/
git mv scripts/arms_connectivity_compare.py archive/scripts_oneoff_2026-08/
git mv scripts/b2_predictor_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/bidirectional_semantic_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/bridge_diagonals_sweep.py archive/scripts_oneoff_2026-08/
git mv scripts/choroid_extent_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/choroid_supervision_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/collation_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/collation_union_recount.py archive/scripts_oneoff_2026-08/
git mv scripts/compare_arms.py archive/scripts_oneoff_2026-08/
git mv scripts/compare_truncation_modes.py archive/scripts_oneoff_2026-08/
git mv scripts/context_keep_eval.py archive/scripts_oneoff_2026-08/
git mv scripts/cover_coverage_audit.py archive/scripts_oneoff_2026-08/
git mv scripts/ctx_anatomy_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/ctx_informative_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/diag_context_and_extent.py archive/scripts_oneoff_2026-08/
git mv scripts/dump_decoder_matrices.py archive/scripts_oneoff_2026-08/
git mv scripts/elementwise_gate_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/latent_anatomy_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/lsem_lseg_conflict.py archive/scripts_oneoff_2026-08/
git mv scripts/relational_distill_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/sampler_equivalence_dump.py archive/scripts_oneoff_2026-08/
git mv scripts/sampler_equivalence_replay.py archive/scripts_oneoff_2026-08/
git mv scripts/why_starved_probe.py archive/scripts_oneoff_2026-08/
git mv scripts/integration_smoke.py archive/scripts_oneoff_2026-08/
git mv scripts/patch_mirage_balanced_sampler.py archive/scripts_oneoff_2026-08/
git mv scripts/patch_mirage_void_argmax.py archive/scripts_oneoff_2026-08/
git mv scripts/plot_background_signal.py archive/scripts_oneoff_2026-08/
git mv scripts/plot_cover_random.py archive/scripts_oneoff_2026-08/
git mv scripts/plot_pretraining.py archive/scripts_oneoff_2026-08/
git mv scripts/plot_region_auc.py archive/scripts_oneoff_2026-08/
git mv scripts/plot_region_auc_explained.py archive/scripts_oneoff_2026-08/
git mv scripts/plot_results.py archive/scripts_oneoff_2026-08/
git mv scripts/build_masking_report.py archive/scripts_oneoff_2026-08/
git mv scripts/fairvision_three_way.py archive/scripts_oneoff_2026-08/
git mv scripts/queue_reruns.sh archive/scripts_oneoff_2026-08/
git mv scripts/run_baselines_eval.sh archive/scripts_oneoff_2026-08/
git mv scripts/run_downstream.sh archive/scripts_oneoff_2026-08/
git mv scripts/run_interpretability.sh archive/scripts_oneoff_2026-08/
git mv scripts/run_patch_aggregate.sh archive/scripts_oneoff_2026-08/
git mv scripts/run_attribution_ep50.ps1 archive/scripts_oneoff_2026-08/
git mv scripts/run_background_signal_sweep.ps1 archive/scripts_oneoff_2026-08/
git mv scripts/run_bridge_probe_queue.ps1 archive/scripts_oneoff_2026-08/
git mv scripts/run_region_auc_ep50.ps1 archive/scripts_oneoff_2026-08/
git mv scripts/run_region_auc_ep50_parallel.ps1 archive/scripts_oneoff_2026-08/
git mv scripts/bench_dataloader.py archive/scripts_oneoff_2026-08/
git mv scripts/profile_iteration.py archive/scripts_oneoff_2026-08/
git mv scripts/vram_profile.py archive/scripts_oneoff_2026-08/
git mv scripts/vram_reconcile.py archive/scripts_oneoff_2026-08/
git mv scripts/fetch_hf_checkpoints.py archive/scripts_oneoff_2026-08/
git mv scripts/sample_dataset.py archive/scripts_oneoff_2026-08/
git mv scripts/preflight_run.py archive/scripts_oneoff_2026-08/
git mv scripts/monitor_run.py archive/scripts_oneoff_2026-08/
git mv scripts/frozen_status.py archive/scripts_oneoff_2026-08/
git mv scripts/campaign_report.py archive/scripts_oneoff_2026-08/
git mv scripts/baselines_eval.py archive/scripts_oneoff_2026-08/
git mv scripts/threshold_fix_masks.py archive/scripts_oneoff_2026-08/
git mv scripts/slice_position_cache.py archive/scripts_oneoff_2026-08/
git mv scripts/smoke_bridge.py archive/scripts_oneoff_2026-08/
git mv scripts/verify_bridge_wiring.py archive/scripts_oneoff_2026-08/
git commit -m "chore: archive one-time probe and demo scripts"
```

Not in this list, deliberately: `scripts/download_weights.py` (M3) and
`scripts/coverage_probe.py` (M8).

Note: `scripts/fetch_hf_checkpoints.py` still carries the Finding D2 HuggingFace
handle. Archiving does not remove the string from the working tree. Decide D2
separately.

#### 4.2.4 Bucket E configs -> `archive\configs_2026-08\` (8 files)

```powershell
git mv configs/frozen_meanpool_bridge_ep41.yaml archive/configs_2026-08/
git mv configs/patch_vitb16_ep100_R2_loss.yaml archive/configs_2026-08/
git mv configs/patch_vitb16_ep100_R3a_intensity.yaml archive/configs_2026-08/
git mv configs/patch_vitb16_ep100_R3b_cluster.yaml archive/configs_2026-08/
git mv configs/arm_anatomy.yaml archive/configs_2026-08/
git mv configs/arm_random_default.yaml archive/configs_2026-08/
git mv configs/arm_random_matched.yaml archive/configs_2026-08/
git mv configs/finetune_mirage_ep100.yaml archive/configs_2026-08/
git commit -m "chore: archive configs for abandoned and unreported arms"
```

Follow-up edit required: `scripts/compare_arms.py:36-42` names `arm_anatomy` and
`arm_random_matched` as `D:\jepa_phase0\runs\` directories, not as config paths,
so it does not break. `autopilot/COVER_AUDIT.md:66` names them as arm labels,
which likewise does not break. No repointing is needed for these eight.

#### 4.2.5 Bucket E results -> `archive\results_legacy_2026-08\` (7 files)

```powershell
git mv results/all_train_logs.json archive/results_legacy_2026-08/
git mv results/summary/oracle_summary.png archive/results_legacy_2026-08/
git mv results/unfrozen_train_loss.png archive/results_legacy_2026-08/
git mv results/unfrozen_training_curves.png archive/results_legacy_2026-08/
git mv results/unfrozen_val_auc.png archive/results_legacy_2026-08/
git mv logs/downstream_unfrozen_d3_s64/train_log_full.csv archive/results_legacy_2026-08/
git mv paper/genai4health2026/figures/fig1_policies_compact.pdf archive/results_legacy_2026-08/
git commit -m "chore: archive legacy unfrozen-arm records and unused vector figure"
```

**Mandatory same-commit edit:** `docs/README.md:55` must be repointed from
`results/unfrozen_*.png` to `archive/results_legacy_2026-08/unfrozen_*.png`,
otherwise the "Unreferenced Legacy Artifacts" table names three files that no
longer exist. Consider repointing `results/archive/` in the same table too if
4.2.6 is ever executed.

#### 4.2.6 Already archived in place - NO COMMAND (16 files)

MEASURED: these already live in a directory named `archive`, are described as
superseded by their own READMEs, and are pointed at by `docs/README.md:53` and
`scripts/archive/README.md`. Moving them buys nothing and breaks two doc rows.

```
results/archive/README.md
results/archive/downstream_auc_curves.png
results/archive/frozen_probe_loss.png
results/archive/frozen_probe_normfix_curves.png
results/archive/frozen_probe_val_auc.png
results/archive/imagenet_degradation.png
results/archive/normfix_frozen_comparison.png
results/archive/normfix_impact.png
results/archive/pretraining_all_runs.png
results/archive/pretraining_imagenet_init.png
results/archive/pretraining_random_init.png
results/archive/test_auc_comparison.png
results/archive/vol0312_slice43.png
results/archive/vol0448_slice52.png
scripts/archive/plot_normfix_results.py
scripts/archive/README.md
```

The audit's Stage 8 (`git rm -r results/archive`) must not be run. See M7.

### 4.3 TIER KEEP - explicitly do not move (9 files)

```
autopilot/run_guarded_probe.py                              M2
autopilot/resource_monitor.ps1                              M4
autopilot/PROCESS_REGISTRY.csv                              M4
scripts/download_weights.py                                 M3
scripts/coverage_probe.py                                   M8
paper/genai4health2026/main.tex                             M5
paper/genai4health2026/figures/figS5_mask_statistics.png    M1
configs/archive/patch_anatomy_v2.FROZEN.yaml                E#9
paper/genai4health2026/research/mechanism_data.json         E#12
```

### 4.4 VERIFY

```powershell
git grep -n "figS5_mask_statistics.png" paper/genai4health2026/main_submission.tex
#   expect one hit at line 796

D:\jepa_phase0\.venv\Scripts\python.exe -m pytest -q
#   expect 134 tests collected, all passing

D:\jepa_phase0\.venv\Scripts\python.exe autopilot\check_manuscript.py
D:\jepa_phase0\.venv\Scripts\python.exe autopilot\p15_verify_numbers.py
#   expect rc=0 from both

# Full rebuild.  CPU-heavy (10,000-resample bootstrap).  Run ONLY when no GPU
# training job is active.
D:\jepa_phase0\.venv\Scripts\python.exe autopilot\refresh_all.py
git diff --stat pre-cleanup-decisions-2026-08-26 -- paper/genai4health2026/auto
#   expect NO change.  Any change means a pipeline input was moved.  Revert.
```

---

## 5. BYTE ACCOUNTING

MEASURED, from `Get-Item().Length` on the working tree at the time of this scan.

| tier | files | bytes | MB |
|---|---:|---:|---:|
| **ARCHIVE** | **204** | **12,518,688** | **11.94** |
| - bucket C 6.1 manuscript material | 38 | 8,501,954 | 8.11 |
| - bucket C 6.2 autopilot residue | 26 | 154,088 | 0.15 |
| - bucket C 6.3 already-archived, no move | 16 | 1,854,101 | 1.77 |
| - bucket C 6.4 one-time scripts | 109 | 904,929 | 0.86 |
| - bucket E configs | 8 | 13,974 | 0.01 |
| - bucket E results, logs, figure | 7 | 1,089,642 | 1.04 |
| **REMOVE** | **183** | **5,455,120** | **5.20** |
| - `logs/patch_run3_log.csv` (tracked) | 1 | 2,090,945 | 1.99 |
| - untracked `__pycache__` (already gitignored) | 182 | 3,364,175 | 3.21 |
| **KEEP, reclassified out of C and E** | 9 | 277,190 | 0.26 |

Bytes affected by tier, counting only files under version control:

- ARCHIVE: **12,518,688 B (11.94 MB)** across 204 tracked files. Zero bytes are
  lost; every one is recoverable with `git mv` in the reverse direction. Only
  188 of the 204 actually move (the 16 in 4.2.6 stay put).
- REMOVE: **2,090,945 B (1.99 MB)** of tracked content, in one file that has a
  byte-identical twin still in the repository, plus **3,364,175 B (3.21 MB)** of
  untracked `__pycache__` already covered by `.gitignore:1-2`. Combined
  **5,455,120 B (5.20 MB)**.

Neither figure shrinks `.git`. MEASURED at the audit: `.git` is 1991.8 MB, of
which 1437.7 MB is dead LFS checkpoint weight and roughly 421.7 MB is loose
objects. Only `git gc` and, at much higher risk, LFS history rewriting touch
those. That remains out of scope.

---

## 6. DO NOT TOUCH - REAFFIRMED

Verified as still applicable at HEAD `c679ae1`.

| path | why |
|---|---|
| `results/masking/structural_loss/regen01.json` | MEASURED byte-identical to `alpha01.json` (both 1.6 KB). The identity IS the reproducibility finding the paper reports. It appears in the duplicate table and must never be treated as a duplicate. **KEEP.** |
| `D:\jepa_phase0\**` | Not part of the repository. Worth GPU-days. Only three directory name listings were taken during this investigation; no file was read. |
| `C:\jepa_data\**` | Live training data. Read continuously by PID 19360. Not accessed at all. |
| `paper/genai4health2026/**` | The live submission. Only the specific superseded files in 4.2.1 are proposed for archiving; `main_submission.tex`, `main.tex`, `figures/figS5_mask_statistics.png`, `auto/**`, `SOURCES.md`, `references.bib`, `neurips_2026.sty` and every cited figure stay. |
| `autopilot/reports/**` | The review and audit record. Nothing here is classified. It is also the reason `main.tex` is KEEP: `G1_REPLICATION.md:109` cites it by line number and cannot be repointed. |
| `configs/replication/**`, `scripts/chain_replication.py`, `scripts/make_replication_configs.py`, `scripts/smoke_replication.py` | Drive the run executing now (PIDs 26152, 11296, 19360). Not classified. Their dependencies `autopilot/run_guarded_probe.py` and `scripts/download_weights.py` are pulled out of bucket C into KEEP for the same reason. |

---

## 7. PENDING AND LIMITATIONS

1. **PENDING: the two untracked SUPERSEDED zips.** 3,077,806 B and 4,287,642 B
   sit in `paper\dist\archive\` on disk, untracked and gitignored since
   `5f5b602`. They are outside a tracked-file cleanup. To clear them I would
   need a decision on whether the pre-2026-08-19 Overleaf bundles are still
   wanted on this machine; the tracked history already contains their contents
   at the commits they were built from. No tier assigned.
2. **PENDING: `scripts/probe_hf_checkpoints.py` and `scripts/score_head_on_cache.py`.**
   MEASURED: both were committed by `5f5b602`, so the audit's limitation 4 is
   resolved; they are now tracked and were not part of bucket C or E, so they
   are not classified here.
3. **PENDING: the four live replication drivers are untracked.** MEASURED:
   `scripts/chain_replication.py`, `scripts/make_replication_configs.py`,
   `scripts/smoke_replication.py` and `configs/replication/` are all `??` in
   `git status`, while the run that uses them is executing. A `git checkout` of
   a cleanup branch would leave them in place (untracked files survive branch
   switches) but a `git clean` would destroy them. Commit them before any
   cleanup branch is created. MEASURED: `autopilot/reports/G1_REPLICATION.md`
   and `REBUTTAL.md` were still untracked mid-scan but are tracked as of
   `021ac9b`, so that part of this item is resolved.
4. **Limitation: `D:\jepa_phase0\` contents were not read.** Only three
   directory listings were taken (`runs\`, `runs\frozen_meanpool_bridge_ep41\`,
   `runs\arm_anatomy\`). A script invoked only from a `D:` wrapper or an
   interactive shell would still appear unreferenced. This is why every
   uncleared file is ARCHIVE and not REMOVE.
5. **Limitation: the working tree moved during this investigation.** MEASURED:
   `autopilot/PROCESS_REGISTRY.csv` and `autopilot/RESOURCE_MONITOR.csv` are
   being appended to by PID 1616 right now, and `scripts/download_weights.py`,
   `paper/genai4health2026/main_submission.tex` and
   `paper/genai4health2026/main_submission.pdf` carry uncommitted changes.
   Byte counts are as of the scan.
6. **Limitation: no test suite was run.** Nothing was changed, so there is
   nothing to regress; section 4.4 is where the gates belong.
