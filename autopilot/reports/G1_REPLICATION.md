# G1 REPLICATION -- six paired post-fork continuations

Status at time of writing: chain RUNNING, leg 1 of 6.
Generated 2026-08-26, launched 2026-08-26 16:50:50 -07:00.

Every quantity below is labelled MEASURED (observed on this machine in this
session), INFERRED (derived arithmetically from a measured quantity), or
PENDING (not yet observable).

---

## 1. Why this run exists

The Area Chair meta-review (`autopilot/reports/AC_meta_review.md`) returned Weak
Reject with one decisive blocker: every masking-policy contrast in the paper
rests on ONE pretraining continuation per policy, so policy is confounded with
post-fork optimisation noise. The AC asks for "at least three independently
randomized, paired post-fork continuations per policy for RANDOM, ENVELOPE and
CENTROID, all run from the same locked epoch-25 ancestor to the same locked
endpoint, with the continuation -- not the test subject or probe seed -- as the
unit of policy inference."

One continuation per policy already exists. This chain adds TWO more per policy:
three policies times two new seeds equals six continuations.

---

## 2. Ancestor (MEASURED)

    Hugging Face repo : yfeng0206/ijepa-3d-oct-checkpoints
    Path in repo      : random-posfix-100ep/jepa_patch-ep025.pth.tar
    Local path        : D:\jepa_phase0\checkpoints_hf\random-posfix-100ep\jepa_patch-ep025.pth.tar
    Size              : 1,507,519,602 bytes
    SHA-256           : e5ad5b0c2aadfa15449409786afbfa39d8b5405b699be8f02f2e540195e97e7b

Fetched with `scripts/download_weights.py --ancestor-ep25 --output-dir
D:\jepa_phase0\checkpoints_hf`. That script previously only knew three weights
in a different repository, so an `--ancestor-ep25` path was added to it; the
SHA-256 above is hard-coded in the script and a mismatch is a fatal error, not a
warning.

Independent corroboration (MEASURED): the pre-existing local mirror
`D:\jepa_phase0\fairvision-glaucoma\checkpoint-ep25\jepa_patch-random_posfix-ep25.pth.tar`
hashes to the same value, and that value is what its published `SHA256SUMS` and
`README.md` (Hugging Face repo `yfeng0206/I-JEPA-OCT-random-posfix-ep25`,
revision `3624b4100ab39b1c989fc61ead0d5248c177735c`) record. Three independent
sources agree, so the fork point is unambiguous.

The chain re-hashes the ancestor immediately before EVERY one of the six
launches and refuses to start that leg on a mismatch
(`scripts/chain_replication.py::verify_ancestor`). Six continuations that did
not all start from the same bytes would not be a replication.

---

## 3. The three policies and how each was confirmed

Configs are GENERATED, not hand-written, by
`scripts/make_replication_configs.py`, into `configs/replication/`. The
generator asserts that outside `mask.curriculum` all six configs are identical
except `meta.seed` and `logging.*`, and prints `INVARIANTS OK` (MEASURED). A
hand-maintained trio of YAML files drifts silently; a generator cannot.

### RANDOM -- the unguided null

Source of truth: `configs/patch_vitb16_ep100.yaml`, the config of the
`random-posfix-100ep` run that produced the ancestor itself and continued to
epoch 100 under stock uniform multiblock masking.

Realisation here: `mask.curriculum.enabled: false`, which routes training
through the stock `MaskCollator` -- one context block at scale 0.85-1.0 and four
target blocks at scale 0.15-0.2, aspect ratio 0.75-1.5. This is exactly the
masking the ancestor itself was trained under, so a RANDOM continuation is a
pure "keep going" arm.

### ENVELOPE -- rectangles restricted to retinal tissue

Source of truth: `configs/patch_mirage_envelope.yaml`, mode `mirage_envelope`,
the config behind `D:\jepa_phase0\runs\patch_mirage_envelope`.

Every placement knob is copied verbatim: `mirage_dilate_patches: 0`,
`mirage_min_block_fill: 0.40`, `mirage_min_retina_visible: 0.25`,
`mirage_max_attempts: 30`, `mirage_occupancy_threshold: 0.25`,
`mirage_spread: true`, `mirage_overlap_tolerance: 0.25`, ramp `T_warm 25 ->
T_total 30`, `r_max 1.0`, linear.

Guide directory `D:\jepa_phase0\fairvision-glaucoma\mirage_guides` (MEASURED:
6,000 volumes, schema 1, packed 1-bit envelopes). This is the directory the
original ENVELOPE run actually read, confirmed by its own training logs, which
print `MIRAGE guides: D:\jepa_phase0\fairvision-glaucoma\mirage_guides
(dilate=0 patches, occupancy_threshold=0.25)` (MEASURED, from
`D:\jepa_phase0\runs\patch_mirage_envelope\train.log`). The newer
`C:\jepa_data\mirage_soft_guides\...` cache is a DIFFERENT product (schema 2,
two-channel soft scores) used by the COVER and ANATOMY arms, and was
deliberately not substituted.

### CENTROID -- confirmed as the arm named `oracle` in older files

Source of truth: `configs/patch_oracle_anatomical.yaml`, mode
`anatomical_prior`, knobs `oracle_region_frac: 0.28`, `oracle_lateral_frac:
0.6`, `oracle_row_offset: 0.0`, `oracle_min_band_rows: 3`.

Four independent confirmations that this, and not `intensity_foreground`, is the
arm whose epoch-100 test AUC is 0.8854851648:

1. MEASURED. `autopilot/reports/P-VERIFY_numerical_audit.md` line 41 records
   `meanpool_sweep_oracle/ep100 | oracle | 100 | float16 | 0.885485165 | 0`.
   The only arm in that audit carrying 0.885485 is `oracle`.
2. MEASURED. `paper/genai4health2026/main.tex` line 508 reads
   `& oracle ep100 & 0.8854852 \\`, i.e. the paper's own source maps the value
   to the run named `oracle`.
3. MEASURED. `paper/genai4health2026/main_submission.tex` defines
   `\newcommand{\ArmBest}{\textsc{centroid}}` with the comment "it locates a
   retinal band by a per-column intensity centroid and uses NO segmentation
   model", and the arm list describes CENTROID as "a band whose vertical
   position is located per slice by a per-column intensity-weighted row
   centroid, smoothed across columns". That is precisely what
   `_anatomical_prior_weight_grid_for_image` in `src/masks/curriculum.py`
   computes; the docstring in `configs/patch_oracle_anatomical.yaml` calls it
   the "v2 retina-following band" and
   `docs/experiments/curriculum_masking.md` describes it as a "PER-COLUMN
   centroid (ribbon)".
4. MEASURED. `intensity_foreground` is a SEPARATE mode in
   `src/masks/curriculum.py` (line 1861 groups `intensity_foreground` and
   `anatomical_prior` as two distinct stateless modes) and belongs to
   `configs/patch_vitb16_ep100_R3a_intensity.yaml`, an arm that does not appear
   in the 0.885485 row.

Conclusion (INFERRED from the four MEASURED facts above): CENTROID is
`anatomical_prior`, config `configs/patch_oracle_anatomical.yaml`. The
replication uses that mode and those knobs.

---

## 4. Seeds

`seed` is a real knob. `src/train_patch.py` line 149 reads
`seed = int(meta_cfg.get('seed', 0))`, forms `run_seed = seed + rank`, and calls
`random.seed`, `np.random.seed`, `torch.manual_seed` and
`torch.cuda.manual_seed_all` with it (MEASURED, read directly). DataLoader
worker RNGs are derived from the torch global generator, so the seed reaches the
workers where crops and guided mask placement are drawn.

Seeds used: **1234 and 5678**, the same two values in all three policies, so the
design is paired and a continuation-level paired analysis is available.

What the seed does and does not randomise (MEASURED, from the code):

* Randomised by the seed: random-resized-crop draws, target and context block
  draws (including the ENVELOPE rejection sampler and the CENTROID band
  sampler), dropout, and every DataLoader worker stream.
* NOT randomised by the seed: the order in which slices are visited.
  `DistributedSampler` is constructed with its default `seed=0` and is advanced
  with `set_epoch(epoch)`, so the permutation depends on the epoch but not on
  `meta.seed`. This is also true of the three original continuations, so it is
  a shared constant of the design rather than a new asymmetry. The reported
  unit of inference is therefore "post-fork stochastic optimisation noise given
  a fixed data order", which is exactly the quantity the AC says is currently
  unestimated.

Cross-check observed at launch (MEASURED): with `r_t = 0` at epoch 25 the
RANDOM and ENVELOPE smoke runs produced identical first losses
(0.13522, 0.10942, 0.11318, 0.11198, 0.13362), which is the expected signature
of a correctly seeded, deterministic warm-start; the arms only diverge once the
ramp engages at epoch 26.

---

## 5. Endpoint

Epoch 50, non-negotiable: every existing arm already carries a frozen MeanPool
probe at epoch 50, so it is the only matched comparison point.

`optimization.epochs` stays at **100** in all six configs. It drives the cosine
LR, weight-decay and EMA schedules; setting it to 50 would anneal the learning
rate to `final_lr` at the endpoint and make these continuations incomparable
with the originals. The endpoint is enforced instead by
`campaign_supervisor.py --stop_after_epoch 50`, which stops the trainer cleanly
once epoch 50 is logged (MEASURED at launch: the supervisor prints "milestone
mode: will stop cleanly after epoch 50 (schedules still sized for 100
epochs)").

---

## 6. Reused infrastructure

Nothing new was written where something existed.

* `scripts/campaign_supervisor.py` -- crash-safe trainer wrapper. Rewrites
  `read_checkpoint` to the rolling `<tag>-last.pth.tar` on resume, restarts
  after a transient crash up to 8 times, refuses to restart when a restart made
  no epoch progress, and supports `--stop_after_epoch`.
* `autopilot/run_guarded_probe.py` -- frozen-probe runner pinned to the fp32
  protocol shared by the existing comparator probes (mean_pool, linear head,
  lr 4e-4, weight decay 0.05, dropout 0.2, 50 epochs, patience 15, warmup 5,
  probe seed 42, batch 256, `use_amp: false`). It hashes the encoder before and
  after and marks the run invalid if the hash moved. Idempotent: it skips a
  probe whose `results.json` exists.
* `scripts/chain_cover_f021.py` -- the idiom the new chain follows.

New files, all thin:

* `scripts/make_replication_configs.py` -- generates the six configs and
  asserts the pairing invariant.
* `scripts/smoke_replication.py` -- the pre-launch smoke test.
* `scripts/chain_replication.py` -- the six-leg sequencer.
* `scripts/download_weights.py` -- gained an `--ancestor-ep25` path with a
  fatal SHA-256 check.

---

## 7. Smoke test (MEASURED, all PASS)

`python scripts/smoke_replication.py --config <cfg>` launches real training from
the downloaded ancestor into a throwaway folder, waits until the trainer has
written at least twelve real iterations to its CSV log, checks the losses are
finite and positive, then stops only the process it started.

| Config | Result | Evidence |
|---|---|---|
| `rep_centroid_s1234.yaml` | PASS | ancestor loaded at epoch 25, "Starting training from epoch 26 to 100", curriculum `mode=anatomical_prior`, 25 iterations, first losses 0.12499 0.12622 0.11508 0.12120 0.12093 |
| `rep_envelope_s1234.yaml` | PASS | guides resolved to `D:\jepa_phase0\fairvision-glaucoma\mirage_guides`, curriculum `mode=mirage_envelope`, 39 iterations, first losses 0.13522 0.10942 0.11318 0.11198 0.13362 |
| `rep_random_s1234.yaml` | PASS | no curriculum line (stock collator), 17 iterations, identical first losses to ENVELOPE at `r_t = 0` as expected |

All three additionally reported: `Training: 600000 slices (6000 volumes)`,
`Slice cache: C:\jepa_data\slice_cache`, `Validation: 100000 slices (1000
volumes)`, `Batches per epoch: 9375 (1172 iters x 8 accum)`, `Effective batch
size: 512`, `Target-encoder autocast (amp_target): False`. Smoke artefacts were
deleted after the test.

---

## 8. Launch

    Chain PID     : 26152  (detached; survives session shutdown)
    Command       : D:\jepa_phase0\.venv\Scripts\python.exe -u scripts\chain_replication.py
    Working dir   : C:\Users\Gary\Desktop\jepa
    Chain stdout  : D:\jepa_phase0\campaign\replication\chain_stdout.log
    Chain stderr  : D:\jepa_phase0\campaign\replication\chain_stderr.log
    Chain log     : D:\jepa_phase0\campaign\replication\chain_replication.log
    Status JSON   : D:\jepa_phase0\campaign\replication\chain_replication_status.json
    Lock file     : D:\jepa_phase0\campaign\replication\chain_replication.lock

Launched as:

    Start-Process -FilePath "D:\jepa_phase0\.venv\Scripts\python.exe" `
      -ArgumentList "-u","scripts\chain_replication.py" `
      -WorkingDirectory "C:\Users\Gary\Desktop\jepa" `
      -RedirectStandardOutput "D:\jepa_phase0\campaign\replication\chain_stdout.log" `
      -RedirectStandardError  "D:\jepa_phase0\campaign\replication\chain_stderr.log" `
      -WindowStyle Hidden -PassThru

Inspect without disturbing it:

    D:\jepa_phase0\.venv\Scripts\python.exe scripts\chain_replication.py --status
    D:\jepa_phase0\.venv\Scripts\python.exe scripts\chain_replication.py --plan

Re-running `scripts\chain_replication.py` after a reboot is safe and is the
recovery command: a PID lock refuses a second concurrent chain, a leg whose
epoch-50 checkpoint exists is skipped, a probe with `results.json` is skipped,
and an interrupted leg resumes from its rolling per-epoch checkpoint.

### Queue order (seed-major, deliberately)

    1. rep_random_s1234    (RANDOM,   seed 1234) -> epoch 50
    2. rep_envelope_s1234  (ENVELOPE, seed 1234) -> epoch 50
    3. rep_centroid_s1234  (CENTROID, seed 1234) -> epoch 50
    4. rep_random_s5678    (RANDOM,   seed 5678) -> epoch 50
    5. rep_envelope_s5678  (ENVELOPE, seed 5678) -> epoch 50
    6. rep_centroid_s5678  (CENTROID, seed 5678) -> epoch 50

Seed-major rather than policy-major so that a COMPLETE paired triple exists
after roughly a third of the wall time. If the deadline forces an early stop,
legs 1-3 alone still yield n=2 continuations per policy for all three policies.

Strict serialisation: one RTX 3090, one training process at a time. The chain
runs each leg to completion before starting the next, and the PID lock prevents
a second chain instance.

---

## 9. Where the outputs land

| Artefact | Path |
|---|---|
| Training run dirs | `D:\jepa_phase0\runs\rep_<policy>_s<seed>\` |
| Rolling resume checkpoint (written EVERY epoch, atomically) | `.../jepa_patch_rep_<policy>_s<seed>-last.pth.tar` |
| Milestone checkpoints (`save_every: 5`: ep 30, 35, 40, 45, 50) | `.../jepa_patch_rep_<policy>_s<seed>-ep<N>.pth.tar` |
| Pinned endpoint copy handed to the probe | `.../jepa_patch_rep_<policy>_s<seed>-ep50-pinned.pth.tar` |
| Per-iteration CSV | `.../jepa_patch_rep_<policy>_s<seed>-log.csv` |
| Per-attempt trainer log | `.../train_<timestamp>_a<N>.log` |
| Per-leg supervisor log and health JSON | `D:\jepa_phase0\campaign\replication\rep_<policy>_s<seed>\` |
| Frozen probe results | `D:\jepa_phase0\runs\frozen_meanpool_rep_<policy>_s<seed>_ep50\results.json` |
| Probe hash guards | `D:\jepa_phase0\autopilot_out\probe_guards\guard_meanpool_rep_<policy>_s<seed>_ep50.json` |

Checkpoint disk cost (INFERRED): 7 files times about 1.5 GB per leg, about
10.5 GB per leg, about 63 GB for all six. Free space on D: at launch was
686 GB (MEASURED), so this is not a constraint.

---

## 10. Resource limits observed

* `num_workers: 6` in every config, `val_num_workers: 2`, `prefetch_factor: 2`.
  Not raised. MEASURED five minutes into leg 1: 12.0 GB of 31.9 GB RAM free, so
  the box is not paging.
* MEASURED at launch: GPU 98 percent utilisation, 17,251 MiB of 24,576 MiB
  VRAM in use by the trainer.
* No process not started by this task was touched. No `Stop-Process` by name
  was used anywhere.
* `C:\jepa_data` was read only. Its `slice_cache` is consumed via
  `data.slice_cache_dir`; nothing under it was created, moved or deleted.
* `D:\jepa_phase0` receives only new run outputs under `runs\rep_*`,
  `runs\frozen_meanpool_rep_*`, `campaign\replication\` and
  `checkpoints_hf\random-posfix-100ep\`, which is the established convention for
  this repository.
* `autopilot/RESUME_COMMAND.txt` carries a prior-session constraint reading "do
  NOT resume pretraining. COVER-0.21 stays at epoch 73. Forbidden:
  scripts\chain_cover_f021.py, scripts\campaign_chain.py". That constraint is
  about the COVER arm; this chain does not read, write or resume
  `D:\jepa_phase0\runs\cover_f021_ep25` or any other pre-existing run
  directory, and does not invoke either forbidden script.

---

## 11. Rate and projected completion

| Quantity | Value | Label |
|---|---|---|
| Iterations per epoch | 9,375 batches (1,172 optimizer steps x 8 accumulation) | MEASURED |
| Throughput, leg 1 epoch 26, 300 s window | 2.673 iterations/s (802 iterations) | MEASURED |
| Implied training-only epoch time | 58.5 min | INFERRED |
| Historic reference rate quoted for this box | 57.7 min/epoch | prior MEASURED |
| First full epoch wall time, including validation | see below | PENDING at first write |
| Six legs x 25 epochs | see below | PENDING at first write |
| Frozen probe wall time, per leg | about 2.7 h (reference: `frozen_meanpool_oracle_ep50_fp32` ran 2026-08-22 22:26:33 to 2026-08-23 01:08:17) | MEASURED, prior run |
| Projected chain completion | see below | PENDING at first write |

### Update after the first completed epoch

PENDING. This section is rewritten with MEASURED values as soon as epoch 26 of
leg 1 prints its `Epoch 26/100 (Ns) train_loss=...` line. No projection based on
a full epoch is recorded until then.

---

## 12. What this does and does not establish

Establishes (once complete): three independently randomised, paired post-fork
continuations per policy for RANDOM, ENVELOPE and CENTROID -- the one already
published plus the two run here -- all from the same locked epoch-25 ancestor
to the same locked epoch-50 endpoint, each probed under the identical frozen
fp32 MeanPool protocol. That makes the continuation the unit of policy
inference, which is what the AC asked for.

Does not establish: anything about epoch 75 or epoch 100 for the new
continuations, anything about ANATOMY-V2 or COVER, and nothing about
generalisation beyond the single FairVision glaucoma test split. The
anatomical-precision claim and the title still need narrowing on the AC's
separate grounds; this chain addresses only the n=1 blocker.
