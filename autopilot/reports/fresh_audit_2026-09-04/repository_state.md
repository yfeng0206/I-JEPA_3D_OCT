# Repository and artifact state

Author-side audit working draft. Measured 2026-09-04.

## Version snapshot

| Surface | Observed state |
|---|---|
| Local branch | `docs/background-signal-findings` |
| Local HEAD | `de145d7005f57e871bc0181bf58b271775d1d25d` |
| Live origin/main | Same SHA, read with `git ls-remote --heads origin` |
| Live origin/docs/background-signal-findings | Same SHA |
| Live GitHub visibility | PUBLIC, via `gh repo view` |
| Last GitHub push | 2026-08-27T12:49:42Z |
| Pre-existing working change | `autopilot/RESOURCE_MONITOR.csv`, not modified by this audit |
| Active historical branch normalfix-update | Not a current local or origin branch |
| Archived alternate work | Tag `archive/volume-moe`, not current paper's implementation |

The current commit descends from position fix `721cd26` and normalization fix
`e625738`. The duplicate-lineage hashes `93b8fdc` and `8760371` are not ancestors
of this HEAD. This is why selecting a branch by its old name is insufficient.

Tracked inventory from `git ls-files`: 1,021 files, including 35 source Python
files, 10 test Python files, 27 configs, 103 script files, 111 autopilot files,
104 paper files, 45 documentation files, 280 result files and 195 archive files.
This is an inventory, not a claim that all 1,021 files were substantively read.
See the component reports for actual review coverage.

## Numerical artifact integrity

`measure_snapshot.py` reads the 43 five-column prediction records in `HANDOFF.md`
and recomputes ROC AUC from the saved labels and probabilities, without fitting.
Its persistent output is `prediction_snapshot.json`.

| Historical status | Files present | AUC agrees to recorded six decimals |
|---|---:|---:|
| Primary | 31 | 31 |
| Excluded | 2 | 2 |
| Retracted | 4 | 4 |
| Supplementary fine-tuning | 6 | 6 |
| Total | 43 | 43 |

All have 3,000 cases, 1,466 positive and 1,534 negative labels. Their label
sequences match. The files inspected store labels and scores, not a demonstrated
subject-ID join: equal label arrays alone cannot establish identity pairing.
Recomputed values do not rehabilitate excluded or retracted experiments.

## Head and checkpoint availability

All 17 `head_local` paths in the August 21 `ARTIFACT_MAP.json` are absent at their
recorded repository-relative locations. This is a stale-location problem, not
proof that all heads were lost: six frozen MeanPool heads exist in
`D:\jepa_phase0\checkpoints_hf\downstream-heads\frozen-meanpool`.
Their SHA-256 values match the manifest for RANDOM and CENTROID at epochs
50, 75 and 100. Each is 11,185 bytes.

The three CENTROID pretraining checkpoints are present under
`D:\jepa_phase0\checkpoints_hf\oracle-anatomical-100ep`.
The epoch-25 ancestor is present at its documented path. Existence was checked;
the multi-gigabyte encoder files were not re-hashed or executed in this pass.
Do not confuse absent old paths, available alternate caches, and newly fetched
artifacts. Nothing was downloaded.

An unauthenticated request to the published Hugging Face model API returned
HTTP 401. This does not prove the repository or weights are absent. Current
external access to every claimed checkpoint was not authenticated or verified.

## Experiment activity

No Python process matching the bounded JEPA training/launcher names was observed.
The GPU snapshot showed 5% utilization and 882 MiB used, which is not evidence
of a JEPA training run. No process was stopped.

The only `rep_*` run directory is `rep_random_s1234`. The latest August 26 log
reports loading the rolling epoch-26 checkpoint and beginning epoch 27; the
last CSV entries are epoch 27, iterations 7 and 8. No completed epoch-50
replication result was found in these locations. The older RUNNING report is
historical. Remote/cloud jobs were not inspected.

## Additional reproducible interface finding

**P2: documented weight listing crashes.**

- Location: `scripts/upload_weights.py:62-72,115-125`.
- Observation: `python scripts\upload_weights.py --list` prints ANATOMY-V1, then
  raises `KeyError: 'run'`, exit 1.
- Evidence: ANATOMY-V2 uses an `explicit` source list instead of a `run` key,
  but the listing loop unconditionally indexes `v["run"]`.
- Impact: the documented inventory command is broken; this is not evidence that
  past uploads failed.
- Requested action: list explicit-source arms through their supported schema.
  No upload or source change was made to reproduce this.

`scripts/download_weights.py:5-8,57-122,169-236` also mixes two eras: `--arm`
selects current paper checkpoints, but `--encoder` selects the legacy ImageNet
epoch-32 model and `--all` downloads the legacy set. Its help phrase "everything"
is not a reliable route to the six current paper arms.
