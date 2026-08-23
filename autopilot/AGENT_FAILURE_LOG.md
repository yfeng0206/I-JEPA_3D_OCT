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
