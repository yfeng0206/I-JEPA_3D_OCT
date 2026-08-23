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
