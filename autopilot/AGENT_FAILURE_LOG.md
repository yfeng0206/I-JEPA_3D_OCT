# AGENT AND PROCESS FAILURE LOG

## 2026-08-22 19:45 PDT - concurrent GPU jobs (process supervision fault)

**Severity:** medium. No incorrect results; wasted throughput and a violated
invariant.

**What happened.** `autopilot/chain_queues.py` was told to wait for the queue-1
driver to exit before starting queue 2. It was given PID **25120**, which I had
obtained with:

```powershell
$pid1=(Get-Process python* | Sort-Object WorkingSet64 -Descending | Select-Object -First 1).Id
```

That selects the process with the **largest working set**, which was the
`eval_downstream.py` *child* doing feature extraction, not the `gpu_queue.py`
*driver*. When probe 1 finished at about 19:43 the child exited, `chain_queues`
concluded queue 1 was done, saw the GPU briefly idle between probes, and started
queue 2.

Result: from 19:45 two probes ran concurrently ---
`meanpool_envelope_fp32_ep50` (queue 1, pid 21724) and
`meanpool_random_ep100_fp32` (queue 2, pid 14428).

**Detection.** The resource monitor showed VRAM rising from 4,921 MiB to
9,053 MiB and `nvidia-smi --query-compute-apps` listed two new PIDs. The
give-away was throughput: feature extraction slowed from 361 s to 832 s per
1,000 volumes, a 2.3x wall-clock regression from CPU contention on the four
dataloader workers each job spawns.

**Impact assessment.**
- Correctness: **none**. The two jobs write to different output directories,
  load different encoder checkpoints, and each is independently hash-guarded.
  They are different probes, not duplicates of the same probe.
- Safety: within limits. VRAM peaked at 36.8% of 24,576 MiB, system RAM at
  59.9% of 31.9 GB, GPU at 78 C. All below the warn thresholds
  (85% VRAM, 80% RAM, 84 C).
- Throughput: roughly neutral. Two jobs at ~2.3x individual slowdown is close to
  break-even, and the GPU stayed at 100% utilisation throughout.

**Decision: let both run to completion.** Killing one would discard up to an
hour of feature extraction to restore an invariant that exists for safety, and
the safety margins are comfortable. The queues cover disjoint probe names
(queue 1: envelope fp32 + cover ep73; queue 2: intensity/random fp32), so no
duplicate work is possible.

**Corrective actions.**
1. `PROCESS_REGISTRY.csv` now records the driver PID and the child PID
   separately.
2. Future chaining must wait on the **driver** PID, captured from the launching
   call, never inferred by sorting on memory. A driver is identifiable by its
   command line containing `gpu_queue`, not by its footprint.
3. Added a guard rule: before starting a new queue, assert that no process whose
   command line contains `eval_downstream.py` is alive, in addition to checking
   GPU memory.

**Follow-up:** monitor VRAM every sample while both run; if VRAM exceeds 85% or
GPU exceeds 84 C for 120 s, pause the queue-2 job (it is the later-started one
and its checkpoint is cached, so it is the cheaper one to restart).

## 2026-08-23 06:20 PDT - RAM at 97 percent during COVER Phase B (assessed, no action)

**Alert:** RESOURCE_ALERTS.log fired [STOP] "System RAM at 96.5-97.4% (>= 85)"
continuously from about 06:13. VRAM also reported 96.7%.

**Cause, measured.** Six dataloader worker processes hold 2.85-4.16 GB each
(~23 GB total) plus the trainer. That is the configured 
um_workers fan-out for
this run, not a leak. The VRAM figure is the torch caching allocator's
reservation: torch itself reports gpu=11789MB allocated, i.e. 48 percent of the
card. The operator's runbook documents this ratchet as expected for this arm.

**Is it harming the run? No.** The decisive test is epoch wall time, which would
balloon under swapping:

| epoch | seconds | vs 3517 s historical mean |
|---|---|---|
| 74 (today, under the alert) | 3487 | -0.9 percent |

Epoch 74 completed slightly FASTER than the historical mean, so the run is not
paging. Windows memory compression is absorbing the pressure (2.18 GB compressed).

**Action taken: none, deliberately.**
- Every python process was audited and all are legitimate: trainer, supervisor,
  chain, sequencer, and the six workers. No stale queue or watcher processes
  remain to reclaim.
- Reducing 
um_workers would require editing a config knob in flight, which the
  operating instructions forbid, and would cost a restart of a healthy run.
- No new GPU or memory-heavy process will be launched while training holds the
  card, which is what the [STOP] guidance actually requires.

**Note for the next milestone refresh.** efresh_all.py --fast is cheap in RAM:
it loads 39 prediction files of about 24 KB each and the bootstrap allocates
roughly 3 MB, so running it during a training window is safe. The expensive
subgroup re-run is skipped under --fast and should be deferred to a window when
no trainer holds the card.

**Standing watch:** if an epoch time exceeds about 4200 s (a 20 percent
regression) while this alert is active, that is genuine paging and the run should
be paused and diagnosed rather than left to grind.

## 2026-08-24 22:30 PDT - Phase C health verified against the arm's own reference run

The blob fp32 continuation resumes from the same epoch-56 seed the original fp16
run used, so the two are directly comparable at matched epochs. That gives a
strong validity check the COVER arm never had.

| epoch | fp32 rerun train / val | fp16 original train / val | val delta |
|---|---|---|---|
| 57 | 0.0797 / 0.3338 | 0.0797 / 0.3338 | 0.0000 |
| 58 | 0.0804 / 0.3312 | 0.0804 / 0.3297 | +0.0015 |
| 59 | 0.0805 / 0.3355 | 0.0803 / 0.3330 | +0.0025 |
| 60 | 0.0813 / 0.3373 | 0.0810 / 0.3343 | +0.0030 |

Three conclusions.

1. The rerun is NOT diverging. It reproduces the reference trajectory to within
   0.003 of validation loss, and the gap grows slowly, which is what two runs
   that differ only in target precision should do once their numerics separate.

2. The mild upward drift in validation loss is characteristic of this arm rather
   than a fault: the fp16 original went 0.3338 at epoch 57 to 0.3428 at epoch 75
   on the same schedule. Rising validation loss is expected for an EMA-target
   JEPA and is documented for this programme.

3. Speed is explained. 5450 s against 3650 s is 1.49x, matching the cost of fp32
   targets and the earlier 1.45x estimate. The run is not slow, the original
   projection simply used the wrong per-epoch figure.

Predicted validation at epoch 75 is therefore about 0.343 if it keeps tracking.
That is a check to apply, not a claim: the measured value will be reported.

## 2026-08-25 08:28 - Near-miss: GPU 0% read at epoch boundary is NOT a stall

Tick sampled `nvidia-smi` once and saw `0 %, 56C` mid-Phase-C. Looked like a dead trainer.

It was the epoch rollover. Epoch 66 ended 06:58; +91 min puts the boundary at 08:29, and the
sample was taken 08:27-08:28. Between epochs the GPU idles while the checkpoint is written and
the 6 dataloader workers respawn. Temperature falls quickly on this card, so a cool reading does
not rule the artifact out.

Re-sampled 5x over 20s: 99/100/100/100/100 %, 24280 MiB, temp rising 69->73C. Trainer pid 12388
accumulating CPU (31988s user). Entirely healthy.

RULE for future ticks: never conclude "GPU idle" from a single nvidia-smi sample. Sample at least
3 times a few seconds apart AND check that the trainer process CPU time is advancing, before
treating it as a stall. Restarting a healthy chain would destroy in-flight epoch work.

## 2026-08-26 - Consistency gate was blind to any line containing a percent sign

`check_manuscript.py` stripped TeX comments with `re.sub(r"%[^\n]*", "", tex)`.
That treats the escaped literal percent in `43.7\%` as the start of a comment and
deletes the remainder of the line before any check runs.

Impact: every check in the gate (undefined macros, banned phrases, macro usage) was
blind to the tail of any line containing a percentage. The measured mask-geometry
table is the most percent-dense table in the paper, so precisely the table the new
narrative depends on was the least protected. Detected because the gate reported
`AUCCoverEpFifty` as unused while it is plainly present at main_submission.tex:474.

Fix: negative lookbehind, `(?<!\\)%[^\n]*`. Macro usage detection went 147 -> 149
and the false "unused" entries cleared. Same defect fixed in `p13_build_zip.py`
(ZIP graphics validation) and `gen_sources.py`.

RULE: when a checker reports something surprising about its own subject matter,
suspect the checker before dismissing it as harmless. A warning that is wrong in the
harmless direction usually means the same logic is also wrong in the dangerous one.

## 2026-08-26 02:40 - Near-miss: plain substring search on PDF text gives false negatives

Verified the new label-efficiency appendix had shipped by searching the compiled PDF for
"Label efficiency". Result: False. The appendix was in fact present on pages 17-18.

Cause: TeX typesets "ffi" as the single ligature glyph U+FB03, so the extracted text contains
"Label ef<ffi>ciency" and a plain `in` test fails. Same applies to fi (U+FB01), fl, ff, ffl -
which is why earlier page dumps showed "beneﬁt" and "difﬁculty".

The tell was that the PDF had grown 25 -> 26 pages, which contradicted the "absent" reading.
Believing the search would have meant re-adding an appendix that was already there, or worse,
concluding the build pipeline was broken when it was fine.

RULE: before searching extracted PDF text, normalise ligatures:
    unicodedata.normalize("NFKD", txt).replace("\ufb00","ff").replace("\ufb01","fi")
        .replace("\ufb02","fl").replace("\ufb03","ffi").replace("\ufb04","ffl")
And always cross-check a negative against an independent signal such as page count.
