# Training/evaluation engineering evidence

Owner: `repair-training`, GPT-6 Astra. Historical baseline: `de145d7`.
This is an implementation investigation, not a new pretraining campaign or a
downstream comparison. Existing checkpoints and predictions were not overwritten.

## Validation record

Initial selected baseline: **51 passed**. First combined training/evaluation
contracts plus those baseline integration tests: **70 passed**, 12 warnings,
17.09 seconds. Warnings are existing/deprecated CUDA AMP API and expected
CPU-only pinned-memory notices, not failed assertions.
`evidence\training_cpu_final\` preserves pytest output, JUnit results, command,
interpreter, branch and tested source hashes. `git diff --check` passed for the
owned production-source edits. No new dependency or test runner was installed.
The subsequent mask-owner logging handoff adds one regression: **71 passed**;
its results and frozen-fixture loss audit are under
`evidence\training_cpu_mask_handoff\`.
The critic-requested explicit-fork repair and joint loss-accounting sentinel
bring the scoped suite to **73 passed** (14 warnings, 16.40 seconds), preserved
with tested source hashes under `evidence\training_cpu_fork_policy\`.
Final validation after the real guided-fixture integration is **74 passed**
(14 warnings, 15.83 seconds), with current code/config hashes under
`evidence\training_cpu_guided_final\`. Both assigned training/GPU todos are
complete; the GPU lease is available and no background job remains.

## Reproduced faults and repairs

### Partial accumulation

`test_historical_closure_reproduces_partial_window_underweight` extracts and
executes the actual `_forward_backward` closure from `git show de145d7:src/train_patch.py`.
For three microbatches in a four-microbatch window, the historical update is
exactly **3/4** of the correctly normalized SGD update. The repaired production
path divides by the actual window size. A separate seven-microbatch test matches
an explicit full-batch reference across both a full and a partial window.

Historical applicability: any final window with `r < accum_steps` contributes
`r / accum_steps` of the intended mean gradient. For example, a seven-of-eight
window contributes 7/8. This does not quantify an AUC effect, and the SGD witness
is not a claim that AdamW parameter deltas scale linearly with this ratio.
The same coupled fault was corrected in downstream fine-tuning; its step count
now uses a ceiling. Historical frozen-head runs do not use accumulation.

### AMP overflow and successful-step state

The actual historical post-update block advances LR, WD and EMA solely because
an accumulation boundary was reached, even if GradScaler skipped the optimizer.
An executable baseline-block regression demonstrates this teacher movement
without an optimizer update.

`optimizer_step` now reports actual success using GradScaler's scale transition.
LR/WD/EMA and the successful-update count advance only on success. Tests cover
both fake injected skips and the installed PyTorch **real CPU GradScaler** with
an injected infinite gradient. Downstream scheduler calls use the same guard.
Curriculum loss observations remain per-microbatch/attempted-window statistics;
they are not claimed to be successful-optimizer counters.

**The existing post-optimizer scheduler phase is preserved**, not “corrected.”
The expected phase was already correct. Restoring logged LR/WD from the optimizer
and the last saved/consumed EMA removes stale scalars in the initial accumulation
microbatches after resume; that logging defect did not change those updates.

### Checkpoint continuation

New epoch-boundary checkpoints carry successful-update counts, full LR/WD
scheduler state, best validation loss, patience counter, last logged scalars,
run/worker-contract digest, and per-rank Python/NumPy/Torch/CUDA RNG and curriculum
state. They retain the old encoder/predictor/teacher/optimizer/scaler/epoch keys.
NumPy RNG arrays are encoded as ordinary lists so the new state also loads with
PyTorch's weights-only reader.

**Explicit fork mode (critic-requested correction).** Default
`meta.resume_policy: exact` keeps strict configuration/topology checks. An
intentional warm-start fork must explicitly set:

```yaml
meta:
  load_checkpoint: true
  read_checkpoint: path_to_trusted_source_checkpoint.pth.tar
  resume_policy: fork
  fork_start_epoch: 0
  seed: 42
```

`fork_start_epoch` is optional: omitting it retains the source's completed epoch
number; setting zero starts the new configured schedule at epoch zero. The chosen
epoch must fit the new horizon. This is **not exact continuation**.

Fork retains encoder, predictor, EMA teacher, optimizer moments/step counters and
available compatible scaler state—not merely the online encoder. It discards
source RNG, curriculum and best/patience state; reseeds from configured seed plus
rank; rebuilds LR/WD/EMA using the new configuration at the declared epoch;
and initializes fresh curriculum/selection state. Optimizer LR/WD values are
reconstructed, while moments/counters remain inherited. Console output declares
all of these choices. Model and optimizer shape compatibility remain mandatory
in fork mode.

New checkpoints persist lineage with the source full-file digest/path/epoch,
source topology, declared fork epoch, seed and schedule offset. To resume this
new lineage exactly later, use `resume_policy: exact`, remove `fork_start_epoch`,
and retain its scientific run configuration. Operational load/fork flags are
excluded from the scientific contract digest.

Regression evidence rejects mismatched new-format checkpoints in default/exact
mode, while an explicit fork restores all three models, AdamW moments/counters
and scaler, leaves fresh curriculum/RNG untouched by source restoration, and
completes the tiny production CPU loop under a changed mask curriculum and
shortened horizon. No GPU or sustained training is needed for this correction.

The production `train_patch.main` loop is exercised with a small CPU ViT,
seven microbatches/epoch, real loss/backprop/AdamW/EMA, validation, diagnostics,
checkpoint writes, and two epochs. Interrupting after epoch one and resuming
produces **bitwise-identical model tensors**, scheduler/selection state and
loss/LR/WD/EMA CSV values to uninterrupted execution. This is tested both with
and without an injected first skipped update.
Resuming a checkpoint whose patience was already exhausted performs no new update.

Separate real DataLoader tests reconstruct the next epoch identically with
**zero and two nonpersistent workers**. Curriculum epoch/ramp state also round
trips. Missing legacy state emits an explicit “not exact” warning; legacy
schedules are reconstructed from epoch count because actual old overflow counts
cannot be recovered. Missing curriculum state is explicitly a cold curriculum
branch, not exact continuation.

Limits: no multi-rank GPU replay was run. Rank-state collection and strict topology
checks are implemented, but saving global RNG does **not** restore persistent
worker streams, in-flight prefetched batches, mid-epoch iterators, arbitrary
external asynchronous RNG consumers, or changed configs/topology. The exact
main-loop witness is single-rank CPU with uploads disabled.

## CPU model/loss invariants

- B=3, two context masks, three prediction groups: sentinel sample, context and
  target IDs agree through `apply_masks`, `repeat_interleave_batch`, and the
  actual predictor's first transformer input. An intentionally wrong repeat
  ordering is detected.
- With fixed preprocessing, changing pixels exclusively in hidden patches
  leaves masked-online output exactly unchanged and hidden-pixel gradients zero.
  The full-image teacher is intentionally allowed to change: this is JEPA target
  contextualization, not hidden-pixel access by the online encoder.
- Rectangular-grid position tests establish row/column axis semantics.
- Actual selected-token teacher normalization gives approximately zero mean/unit
  variance per token. The selected-token Smooth-L1 matches the scalar objective.
- Actual tiny encoder/predictor backward gives finite gradients on every
  trainable parameter and no teacher gradient. No ViT or tensor-ordering source
  fix was warranted by these passing valid-input tests.

## Bounded RTX 3090 evidence

Reusable driver: `scripts\training_layer_diagnostic.py`.
Configuration: `configs\diagnostic_training_layers.yaml`.
Full local ancestor SHA-256:
`e5ad5b0c2aadfa15449409786afbfa39d8b5405b699be8f02f2e540195e97e7b`.
The full file was streamed and verified; no download occurred.

### Initial failure: multi-context collation

`evidence\training_gpu_v1\` preserves the first manifest and failure.
At B=1 with `nenc=2`, the actual uniform collator returned context group lengths
112 and 128. Production `apply_masks` rejected their concatenation. The job
stopped before its first optimizer update and released hooks/VRAM.

This is a generator rectangularization contract failure handed to the mask
engineer, not a reason to alter the model's correctly ordered gather. Historical
production configs use `nenc=1`, so this witness alone does not establish a
historical-run failure. No silent “fix” or truncation adapter was inserted into
the diagnostic.
Subsequent mask-owner handoff reports the collator repair: global context-length
rectangularization across all groups/images, with B1/B3 `apply_masks` regressions
and 113 focused CPU tests passing. That repair is owned/evidenced in
`mask_report.md`; no ViT workaround or additional nenc2 GPU replay was performed.
The original v1 failure artifact is retained as a historical diagnostic result.

### Historical-shape replay

`evidence\training_gpu_v2\` contains the manifest, all block/gradient summaries,
per-target losses and verdict. Eight **independent one-update cases** reset to
the same ancestor: B1/B2 × uniform/intensity-foreground curriculum × fp32/AMP.
They are not eight sequential pretraining steps. Both policies consume fixed
real Training-only slices with fixed full-resize/ImageNet preprocessing.
Seed reuse is not asserted to pair all realized mask draws across policies.

| Fixed case | Loss fp32 | Loss AMP | Weighted target slots | Repeated target slots per image |
|---|---:|---:|---:|---|
| Uniform B1 | 0.04075471 | 0.04076665 | 168 | 60 |
| Intensity B1 | 0.08897037 | 0.08897410 | 168 | 58 |
| Uniform B2 | 0.07400207 | 0.07399333 | 336 | 60, 47 |
| Intensity B2 | 0.08336469 | 0.08333489 | 336 | 58, 60 |

Every case visits all **30 transformer blocks**: 12 online, 12 teacher, six
predictor. Every activation and online/predictor gradient is finite; feature
variation is nonzero; online/predictor parameter updates are nonzero; teacher
gradients are absent. Teacher parameter updates match the implemented EMA
formula with **maximum absolute error zero**. Per-target Smooth-L1 contributions
reconcile with the scalar objective. Hidden-pixel perturbation leaves online
output exactly unchanged while the full teacher responds.

The fp32/AMP loss absolute differences range from 3.725e-6 to 2.980e-5. Peak
allocated GPU memory was 2.33 GB; total driver wall time was 13.97 seconds.
AMP used initial loss scale 128, fp32 teacher targets, explicit attention (the
production default), and the same model parameters in each paired case.

Measurement caveats: the v2 representative-gradient cosine used an fp32 reduction
and rounded slightly above one; **do not interpret that cosine field**. The
driver now computes it in float64. V2 block-gradient RMS values are pre-unscale
for AMP (scale 128); aggregate parameter-gradient norms are correctly unscaled.
The driver now records unscaled block RMS too. Absolute gradient/loss differences
and the finite/update/EMA/isolation tests are unaffected by these reporting fixes.

All leases were recorded before launch, PID/output bounds were captured, and no
unrelated desktop process was killed. At v2 completion, the Python GPU process
had exited and `nvidia-smi` showed no remaining Python training job.

### Final synthetic-mask handoff

The mask owner supplied `mask_replay64_v2\synthetic_final_masks.pt`, SHA-256
`a482d0a76c80ead340319d7e8d77c659e34ba263425a0ad19734f25d2e203277`.
The driver now exposes `--cpu-fixture ... --output ...` for exact frozen-mask
reuse. No masks are redrawn and no counterfactual policies are relabeled.

With the supplied B2/context1/target4 fixture, 40 slots/target, a two-block
16-dimensional CPU ViT/predictor and guide channel zero thresholded at 0.25:

- 100 tissue slots + 220 background slots = 320 selected loss slots.
- 233 first-occurrence slots + 87 repeated slots = the same 320 slots.
- Each partition reconstructs scalar Smooth-L1 **0.43737465**, as do the
  four per-target means; online/predictor gradients are finite and teacher
  gradients absent.

This is a synthetic arithmetic/gradient check, not real-data tissue performance
or another production-size GPU test. No optimizer or additional GPU update ran.
The final real segmentation replay statistics remain owned/adjudicated by the
mask engineer/coordinator.

Trainer console logging now declares `cover_algorithm` and `cover_context_guard`,
and emits all three `delivered_context_*` status/intervention counters. Missing
measurements print `delivered_context_floor=not_reported`, not a passing zero.
Legacy `hidden/visible_cells/floor_ok` spellings are retained for the existing
supervisor parser, but explicitly carry `stats_scope=policy_target_complement`;
they are never substituted for the new delivered-context fields. A regression
checks both semantics and parser compatibility. No campaign was launched.

### Coordinator-authorized narrow REAL guided-mask follow-up

`evidence\training_gpu_guided_v1\` records **three independent fp32 updates**,
one each for legacy COVER, exact-prefix v2 and v2 plus context guard, resetting
the same ancestor before each case. Inputs are the **first TWO predeclared
Training observations** from the mask engineer's existing audited scope, using
their exact saved **B2** masks—not redraws or outcome-selected examples.
The private fixture SHA-256 is
`269f1c143af8e91daa179796fb0bfbd40583d91fafebbc0dbd45cba9c6c4692e`.
Raw images, guide arrays, source identities and representative gradients remain
under ignored `.audit`; public evidence contains only ordinal rows and aggregates.

All policies receive identical already-normalized images and valid guide arrays.
Tissue is the explicit **guide-channel-zero occupancy >= 0.25 proxy**, not
clinical importance or diagnostic truth. The two full grids contain 85 and 77
tissue cells. All four target groups have K=40, giving 320 duplicate-weighted
loss slots across B2. Legacy context length is 126/image; v2 and guarded v2
have 132/image. Therefore legacy-versus-v2 is not claimed to hold context-token
budget constant. V2 versus guarded v2 does hold both target tensors and context
budgets constant.

| Actual delivered measurement | Legacy COVER | Exact-prefix v2 | v2 + context guard |
|---|---:|---:|---:|
| Encoder-visible tissue, row 0 / row 1 | 18 / 17 | 18 / 17 | 18 / 17 |
| Target-complement tissue, row 0 / row 1 | 22 / 18 | 18 / 17 | 18 / 17 |
| Tissue / background loss slots | 177 / 143 | 178 / 142 | 178 / 142 |
| First-occurrence / repeated loss slots | 244 / 76 | 242 / 78 | 242 / 78 |
| Tissue contribution to scalar loss | 0.02612709 | 0.04941365 | 0.04941365 |
| Background contribution to scalar loss | 0.02047275 | 0.04089354 | 0.04089354 |
| Actual scalar Smooth-L1 | 0.04659984 | 0.09030718 | 0.09030718 |

Per-image and tissue/background × first/repeated-slot sums are in
`results.json` and `joint_summary.json`; each partition reconciles with the
actual scalar. “First occurrence” is an accounting partition in target-group
order, not a replacement unique-token objective: predictions for a repeated
spatial index can differ across target groups.

These examples directly show why target-complement tissue is not final encoder
context: legacy loses four and one of the complement's tissue cells, respectively.
Nevertheless all three policies already meet the 21% tissue floor on these two
predeclared examples (18/85 and 17/77). **The context guard changed zero of these
two contexts**, verified by exact tensor equality; its equality of loss and
gradients with prefix-only v2 is therefore expected. We did not select additional
cases to obtain a favorable or larger guard effect. Population/batch64 geometry
and intervention frequencies remain separate mask-owner evidence.

Delivered random-fill counts are **1/8 target blocks** for legacy and **0/8**
for v2/guard; images without random fill are respectively **1/2** and **2/2**.
These are target-block counts, **not curriculum image-draw probabilities or
estimates of `r_max`**. The one legacy `random_legal` block has 14 tissue and
26 background slots; its actual mean Smooth-L1 is 0.06196470. Thus random fill
is not synonymous with background-only supervision, and background slots still
occur in guided target blocks. No frequency inference is made from B2.

All three production-size cases passed all 30 block activation/gradient checks,
nonzero online/predictor updates, absent teacher gradients, exact EMA error zero
and masked-online hidden-pixel sensitivity zero. Total driver time was 6.94 s,
peak allocated memory 2.33 GB. The lease preceded launch, PID 21284 exited,
hooks/VRAM were released, and no Python GPU process remained.

The higher instantaneous v2 loss does **not** rank policies or predict downstream
AUC. It measures a changed delivered conditional-prediction task under a fixed
historical ancestor. No preferred policy was changed, no sustained training was
started, and no additional observations or updates were requested.

## Evaluation identity and compatibility

- Future cached features are keyed by a versioned complete source manifest:
  checkpoint **full-file** digest/path/component and model config; dataset root
  and split; exact ordered source-case list with sizes/mtimes; sampled slice
  indices/resolution; preprocessing, model/dataset code, precision, encode chunk
  size and PyTorch version.
- This is an ordered-file/stat identity, **not** an assertion that every raw
  volume byte was hashed. Full dataset hashing/encoding was deliberately avoided.
- Loaded v2 caches must match provenance, dimensions, finite values and binary
  labels. Legacy caches are rejected by default. Explicit `allow_unverified`
  allows shape-checked old data only with a warning and unverified-order flag;
  it does not fabricate verified provenance.
- A regression with identical first 1 MiB and different file tails establishes
  that the new complete checkpoint digest distinguishes what the historical
  prefix hash could not.
- Future prediction archives retain labels/probabilities and add stable
  pseudonymous source-case IDs, row indices, a manifest digest and a verified-
  order flag. Sidecars bind the source order and selected head/config. These
  IDs derive from source file stems; independent patient linkage is **unverified**,
  and hashing is not an anonymization guarantee.
- Raw case names and source-order sidecars stay local. The real diagnostic
  input map/images are under ignored `.audit\delivered_task_training`; no raw
  case identifiers are reproduced in this report.
- `configs\downstream_frozen_meanpool_canonical.yaml` explicitly chooses frozen
  MeanPool + LinearHead, fp32 and verified cache handling. Historical attentive
  configs/default compatibility remain available and were not overwritten.
- A tiny synthetic two-case feature extraction/cache round trip exercises the
  actual preprocessing/export/cache code. Its source and order validation passes.
  This is not a glaucoma performance estimate.

## Not run / inference boundaries

- Fixed-teacher overfit up to 100 updates: **omitted**; the healthy real
  forward/backward/update/EMA/isolation witnesses resolved the plumbing question.
- Sustained or resumed pretraining: **not authorized and not run**.
- Full 3000-case extraction, new test-set tuning, exhaustive 43-AUC rescoring:
  **not run**; not needed for these implementation contracts.
- Recovered-head rescoring: **omitted**; the six-head recovery/manifest was
  already verified and no new head discrepancy was demonstrated here.
- Real guided tissue/background GPU loss partition: **completed only for the
  three authorized B2 policy cases above**. The earlier synthetic fixture and
  uniform/intensity diagnostic remain separately labeled; no full-dataset
  encoder-loss sweep or guided AMP comparison was run.
- No repaired-code AUC improvement is inferred. Historical results remain
  outcomes of their historical implementations; determining downstream impact
  requires a separately approved, matched training/evaluation comparison.
