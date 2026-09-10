# Fresh code audit: active OCT I-JEPA patch pretraining and downstream evaluation

**Date / baseline:** 2026-09-04, `de145d7005f57e871bc0181bf58b271775d1d25d`  
**Scope:** active patch pretraining, mask policies, launch/resume code, and patch downstream evaluation. The archived slice-only pipeline, manuscript prose/statistics, autopilot delivery, and DOCX tooling were excluded.  
**Change policy:** read-only audit; no production code, data, checkpoints, or run state were modified.

## Bottom line

The active model is a **2-D slice encoder with volume-level downstream aggregation**, not a native 3-D encoder. The current position-vector and input-normalization fixes are present and passed targeted checks. The frozen downstream path correctly loads and freezes the EMA `target_encoder`, computes one logit per OCT volume, selects the linear head by validation AUC, and evaluates the selected head once on Test.

No P0 code defect was found. The report records **5 P1 and 5 P2 findings**.
The most consequential implementation/configuration issues are:

1. the reproducible anatomy-v1 launch path is spliced from the envelope arm at epoch 27 rather than directly from the epoch-25 ancestor;
2. the disclosed COVER post-placement collation defect is confirmed, and its training diagnostics report pre-collation rather than delivered budgets;
3. training checkpoints omit RNG and best-validation state, so restarts are not exact stochastic continuations and `*-best` is only best since the latest process launch;
4. the checked-in generic downstream config and legacy sweep script do not encode the paper's fp32 frozen-MeanPool protocol.

## Real active pipeline

### Pretraining

1. `OCTSliceDataset` lexically sorts each split's `.npz` files and uniformly selects `num_slices=100` indices from the first volume axis (`src/datasets/oct_slices.py:55-67`, `src/datasets/oct_slices.py:134-146`).
2. Each selected B-scan is independently resized to 256×256, duplicated to RGB, randomly resized-cropped, and ImageNet-normalized (`src/datasets/oct_slices.py:151-173`, `src/transforms.py:43-91`).
3. ViT-B/16 maps one slice to a 16×16 grid of 256 patch tokens. Fixed 2-D sinusoidal position vectors are added before masking (`src/models/vision_transformer.py:353-461`).
4. The online encoder sees one large context rectangle with the target union removed. The frozen EMA teacher sees the full slice, is layer-normalized, and supplies target tokens. The predictor receives context tokens plus positional mask tokens and predicts four target groups (`src/train_patch.py:699-734`, `src/models/vision_transformer.py:469-610`).
5. Optimization is Smooth-L1 prediction loss, AdamW, cosine LR/WD, and cosine EMA momentum (`src/train_patch.py:491-530`, `src/train_patch.py:720-781`).

Pretraining has **no inter-slice attention or 3-D convolution**: every slice is an independent self-supervised example.

### Downstream and the meaning of “3D”

1. `OCTVolumeDataset` returns 100 uniformly spaced slices per volume (`src/datasets/oct_volumes.py:58-69`, `src/datasets/oct_volumes.py:109-123`).
2. The frozen path loads checkpoint key `target_encoder`, disables gradients, and uses evaluation mode (`src/eval_downstream.py:519-532`).
3. Each slice is ImageNet-normalized, independently encoded, and mean-pooled over its 256 patch tokens, yielding `(volume, 100 slices, 768 features)` (`src/eval_downstream.py:370-397`).
4. The paper's active probe is `MeanPool`: it averages the 100 slice vectors, discarding slice order, then applies `LinearHead = LayerNorm(768) + Linear(768,1)` (`src/models/attentive_pool_minimal.py:102-119`, `src/eval_downstream.py:183-192`).
5. AUC therefore has **one inference unit per volume**, not per slice. “3D” here means that a full multi-slice volume contributes to one prediction; the active MeanPool head is permutation-invariant along the slice axis.

### Checkpoint and head selection

- Encoder checkpoint: the EMA `target_encoder`, not the online context encoder (`src/eval_downstream.py:527`).
- Probe checkpoint: `best_model.pt`, selected by strict improvement in validation AUC (`src/eval_downstream.py:630-698`).
- Test: the selected `probe` and `head` are reloaded, then Test is evaluated once (`src/eval_downstream.py:709-744`).
- Active MeanPool probes have 0 probe parameters and a 2,305-parameter LayerNorm+linear head. The file is always named `best_model.pt`; arm identity resides in the output directory.

Examples observed in the bounded run inventory:

| Policy result | Encoder checkpoint used | Head directory/file |
|---|---|---|
| random ep50 fp32 | `checkpoints_hf\random-posfix-100ep\jepa_patch-ep050.pth.tar` | `runs\frozen_meanpool_random_ep50_fp32\best_model.pt` |
| oracle ep50 fp32 | `checkpoints_hf\oracle-anatomical-100ep\jepa_patch_oracle-ep050.pth.tar` | `runs\frozen_meanpool_oracle_ep50_fp32\best_model.pt` |
| envelope ep50 fp32 | `runs\patch_mirage_envelope\jepa_patch_mirage-ep50.pth.tar` | `runs\frozen_meanpool_envelope_fp32_ep50\best_model.pt` |
| anatomy-v1 ep30 | `runs\patch_mirage_anatomy\jepa_patch_mirage-ep30.pth.tar` | `runs\frozen_meanpool_anatomy_ep30\best_model.pt` |
| anatomy-v2 ep50 | `runs\anatomy_v2_ep25\jepa_patch_mirage-ep50.pth.tar` | `runs\frozen_meanpool_bridge_ep50\best_model.pt` |
| cover f=.21 ep50 | `runs\cover_f021_ep25\jepa_patch_cover_f021-ep50.pth.tar` | `runs\frozen_meanpool_cover_f021_ep50\best_model.pt` |
| clean anatomy-v2 ep75 | `runs\blob_fp32_ep56\jepa_patch_blob_fp32-ep75.pth.tar` | `runs\frozen_meanpool_blob_fp32_ep75\best_model.pt` |

## Policy keys and delivered mask contract

| Paper policy | Code key / principal config | Checkpoint tag |
|---|---|---|
| random | curriculum disabled; `MaskCollator` | `jepa_patch` |
| oracle / intensity-located band | `curriculum.mode: anatomical_prior` (`configs/patch_oracle_anatomical.yaml:35`) | `jepa_patch_oracle` |
| envelope | `curriculum.mode: mirage_envelope` (`configs/patch_mirage_envelope.yaml:75`) | `jepa_patch_mirage` |
| anatomy-v1 | `mirage_anatomy`, `pred_target_k: 16`, diagonal bridging absent/default false (`configs/patch_mirage_anatomy.yaml:31-38`) | `jepa_patch_mirage` |
| anatomy-v2 / blob | `mirage_anatomy`, `pred_target_k: 16`, `anatomy_bridge_diagonals: true` (`configs/patch_anatomy_v2.yaml:32-51`) | initial `jepa_patch_mirage`; clean continuation `jepa_patch_blob_fp32` |
| cover f=.21 | `mirage_cover`, leave/visible fraction .21, `random_legal`, prefix context truncation (`configs/patch_cover_f021_ep25.yaml:40-66`, `:93`) | `jepa_patch_cover_f021` |

Common settings are four predictor targets, one context mask, 15–20% sampled target scale, 85–100% sampled context scale, and `allow_overlap: false`.

- `allow_overlap: false` prevents **context–target** overlap; it does not prohibit overlap among the four target rectangles (`src/masks/multiblock.py:145-183`).
- Stock rectangle and envelope targets are globally front-truncated to the shortest of the four sampled target sizes (`src/masks/multiblock.py:198-210`; curriculum equivalent at `src/masks/curriculum.py:1778-1783`).
- Anatomy targets are connectedly reduced/resampled to exactly K=16 per target, giving 64 predictor loss slots per image (`src/masks/curriculum.py:1283-1332`, `:1768-1776`).
- Context is sampled against the full target union and then each context group is truncated to the batch minimum. The stock/default operation is a row-major prefix (`src/masks/multiblock.py:217-230`; `src/masks/curriculum.py:1657-1690`).
- Envelope target overlap is only best-effort bounded; when no admissible window clears the tolerance, the least-overlapping admissible window is used (`src/masks/curriculum.py:781-810`).
- COVER explicitly permits overlapping rectangles while optimizing anatomy coverage (`src/masks/cover.py:1-22`).
- Anatomy-v2 bridging forbids annexing another target's cells (`src/masks/curriculum.py:1308-1329`).

## Position, normalization, precision, and scheduler status

### Verified fixed / no regression found

- **2-D position vectors:** height and width coordinates are now separately constructed and tiled in row-major token order (`src/models/vision_transformer.py:52-119`). A non-square 2×3 CPU check produced shape `(2,3,8)` and non-zero, equal-magnitude row and column changes. Encoder position vectors are added before mask selection (`:403-451`).
- **Teacher target normalization:** pretraining applies fp32 `layer_norm` to teacher output before target extraction (`src/train_patch.py:667-674`, `:711-720`); validation does the same (`:548-554`).
- **Downstream input normalization:** both cached frozen features and end-to-end patch evaluation call `imagenet_normalize` (`src/eval_downstream.py:388`, `:1016`). A CPU comparison against the explicit ImageNet formula had maximum error 0.
- **Downstream fp32 switch:** `data.use_amp: false` is propagated through `set_amp`/`amp_ctx`, including feature extraction, head training, and evaluation (`src/eval_downstream.py:39-57`, `:541-545`, `:1422-1425`). Current local arm builders set it false (`scripts/probe_blob_epochs.py:43`, `scripts/chain_blob_fp32.py:89`, `scripts/probe_hf_checkpoints.py:95`).
- **Target encoder is frozen downstream:** confirmed at `src/eval_downstream.py:527-532`.
- **Same-IPE scheduler resume:** LR, WD, and EMA are reconstructed by fast-forwarding `start_epoch * iterations_per_epoch` steps (`src/train_patch.py:524-530`). A small CPU trace matched the uninterrupted schedule exactly when steps/epoch were unchanged.
- **Curriculum phase:** an explicit `T_total` is preserved; with `T_warm=25`, `T_total=30`, `r_max=1`, epoch index 25 has zero guidance and epoch 30 has full guidance (`src/masks/curriculum.py:266-276`, `:586-604`).

The paused `rep_random_s1234` evidence is consistent with correct scheduler
fast-forward, not an LR/WD reset. Its epoch-27 iteration-8 CSV values
(`lr=0.0002211659383`, `wd=0.0967815209`, `ema=0.9966309159`) exactly match a
CPU reconstruction after fast-forwarding 26×1,172 steps and then advancing once.
The preceding iteration-1–7 values of `.0001/.04/.996` are a logging defect
described in F10: no optimizer step occurs until iteration 8.

### Precision provenance

The current code defaults pretraining teacher targets to fp32 (`src/train_patch.py:477-484`). The clean anatomy-v2 continuation explicitly forces `amp_target=False` (`scripts/chain_blob_fp32.py:73`). The earlier `campaign_chain.py` override that changed resumed anatomy targets to fp16 is corrected and documented in code (`scripts/campaign_chain.py:179-186`). The historical precision-spliced checkpoints remain experimental limitations; this audit found no current reintroduction of that override.

The legacy remote downstream sweep omitted `data.use_amp`, so current code interprets it as fp16. The local re-probe builders explicitly request fp32. These are two different probe protocols and should remain labeled as such.

## Findings

### F1 — P1 — Anatomy-v1's reproducible launch path is a policy-spliced continuation

`configs/patch_mirage_anatomy.yaml:56` reads:

`D:\jepa_phase0\runs\patch_mirage_envelope\resume-ep27.pth.tar`

while switching the policy to `mirage_anatomy` at `:35`. The downstream anatomy-v1 config then evaluates `patch_mirage_anatomy\jepa_patch_mirage-ep30.pth.tar`. Thus the checked-in path is envelope through epoch 27 followed by anatomy through epoch 30, not a direct epoch-25-to-30 anatomy continuation.

**Impact:** anatomy-v1 cannot support a literal “all six arms directly continue the same epoch-25 checkpoint with only their named policy” interpretation. It does still descend from the epoch-25 ancestor, but it contains an intervening envelope segment.

**Evidence bound:** the config mismatch is confirmed. The anatomy-v1 run directory has no retained trainer log or embedded parent-checkpoint hash, so this audit cannot independently prove which historical command created the existing ep30 tensor. A separate archived launch record could resolve that.

### F2 — P1, already disclosed — COVER optimizes full rectangles but trains on prefix-truncated rectangles

COVER places and scores full rectangles first (`src/masks/curriculum.py:1390-1490`), then the shared collator globally finds the shortest target and applies `t[:global_min_pred]` to every target (`:1778-1783`). Because indices are sorted row-major, this is not shape-preserving truncation.

The same batch diagnostics are accumulated before collation (`:1373`, `:1480`, `:1528`, `:1654`) and exposed as `cover_hidden_frac`, `patches_per_block`, and `context_patches` (`:1804-1818`). `truncated_target_patches` is computed but the trainer does not print it.

**Cheap reproduction:** on a synthetic 16×16 retinal band at the production f=.21 settings:

- logged pre-collation anatomy hidden: **0.7812**
- delivered post-collation anatomy hidden: **0.7646**
- logged patches/block: **45.25**; delivered K: **42**
- logged context: **118.38**; delivered context K: **94**

**Impact:** the arm's observed trajectory is real for the implemented policy, but it does not identify the intended “hide at least as much anatomy as envelope” intervention. Training logs alone cannot recover delivered coverage/budgets. This is the known COVER limitation, confirmed rather than newly discovered.

### F3 — P1 — Checkpoints omit RNG state, so campaign restarts are not exact continuations

Every process launch reseeds Python, NumPy, torch, and CUDA from `seed + rank` (`src/train_patch.py:149-154`). Checkpoints save models, optimizer, scaler, epoch, and curriculum state, but no RNG states (`src/helper.py:305-333`).

The campaign supervisor repeatedly relaunches from rolling checkpoints. On every relaunch, the crop/mask RNG sequence restarts from its initial seed. `DistributedSampler.set_epoch(epoch)` changes sample order, but crop and mask draws are not the draws an uninterrupted run would have made.

**Impact:** resume is numerically valid but not stochastic-state equivalent. Arms with different restart boundaries have an additional training-randomness difference beyond mask policy. This is an experimental reproducibility limitation, not evidence that a reported AUC is arithmetically wrong.

### F4 — P1 — `*-best` and patience state reset on every pretraining resume

After checkpoint load, `best_val_loss` is reset to infinity and `epochs_no_improve` to zero (`src/train_patch.py:511-530`, `:571-573`). Neither value is stored by `save_checkpoint` (`src/helper.py:305-333`).

**Impact:** the first eligible post-resume validation epoch can overwrite `<tag>-best.pth.tar` even when it is worse than the true pre-resume best; early-stop patience also restarts. Active policy configs use `patience: 9999`, so early stopping did not truncate those runs, and downstream comparisons use named epoch checkpoints. Nevertheless, a pretraining `-best` file is only “best since the current process launch,” not necessarily best over the whole trajectory.

### F5 — P2 — The final partial accumulation step is underweighted

The loop always divides every microbatch loss by `accum_steps` (`src/train_patch.py:730`, `:734`) but also steps on a short final window (`:742`). For the documented active case of 9,375 microbatches and accumulation 8, the last optimizer step contains seven microbatches but is divided by eight.

**Impact:** the last update of each epoch has 0.875 of the intended averaged gradient magnitude. The scheduler correctly uses `ceil(9375/8)=1172` steps, so this is not a schedule-count error; it is a small gradient-scaling mismatch affecting one of 1,172 steps per epoch.

### F6 — P2 — Non-prefix encoder truncation ignores the epoch when reseeding

`set_epoch` stores `self._epoch` (`src/masks/curriculum.py:592`), but `generate` derives the truncation epoch from nonexistent `self.epoch` (`:1666`). It therefore falls back to zero and reseeds the truncation generator to the same epoch-zero stream on every call.

**Cheap reproduction:** with identical global seeds and `enc_truncate=window_free`, epoch 30 and epoch 31 both recorded `_enc_trunc_epoch=0` and produced identical context masks.

**Impact:** non-prefix context-window tie breaking is more correlated than documented. The current main COVER f=.21 config uses `enc_truncate: prefix` and is unaffected. The historical `patch_cover_random_ep25.yaml` uses `enc_truncate: window` (`:78`) and is affected, in addition to its separately disclosed intervention/precision limitations.

### F7 — P1 — Checked-in generic downstream entry points do not reproduce the active paper protocol

`configs/downstream_patch.yaml`:

- points to `jepa_patch-run3-ep11.pth.tar` (`:10`);
- omits `probe_type`, so code defaults to `attentive` (`src/eval_downstream.py:589`);
- omits `data.use_amp`, so code defaults to AMP/fp16 (`:541`);
- uses different learning rates, decay, patience, and warmup (`configs/downstream_patch.yaml:18-23`).

Likewise, `scripts/run_linear_sweep.sh` defaults to `PROBE_TYPE=attentive` (`:27`) and its generated data block has no `use_amp` field (`:169-196`), so it uses fp16.

The active paper protocol instead comes from dynamically generated configs in `probe_hf_checkpoints.py`, `probe_blob_epochs.py`, `campaign_chain.py`, and `chain_blob_fp32.py`: `mean_pool`, `use_amp: false`, LR 4e-4, WD .05, 50 epochs, patience 15, warmup 5.

**Impact:** current results can be traced, but the obvious checked-in config/script is not a canonical reproduction entry point. A user invoking it without knowing the generated-config history will run a materially different probe.

### F8 — P2 — Prediction artifacts preserve labels/probabilities but not case identity

Inference order is deterministic in code: volume files are lexically sorted (`src/datasets/oct_volumes.py:58`), and validation/test loaders do not shuffle (`src/eval_downstream.py:581-584`). Six inspected policy artifacts each had 3,000 labels, 1,466 positives, and the same label-byte hash.

However, `test_predictions.npz` and `val_predictions.npz` contain only `labels` and `probs` (`src/eval_downstream.py:747-752`). They do not contain filenames, subject IDs, or a split-manifest hash.

**Impact:** no ordering defect was observed, and AUC computation itself is correct for the saved arrays. But equal label sequences are not a proof that row *i* is the same subject across artifacts; exact paired-case identity cannot be audited from the prediction files alone.

### F9 — P2 — Feature-cache identity is incomplete

The cache filename includes split, slice count, resolution, precision, and a 12-hex hash of only the first 1 MiB of the checkpoint (`src/eval_downstream.py:356-365`, `:549-560`). It does not include the data-root path, file manifest/content hash, or a full checkpoint hash.

**Impact:** reusing an output directory after replacing a dataset split can silently load stale features/labels, and checkpoint-prefix collisions are possible in principle. No collision or stale-cache reuse was observed in the inspected arm directories, which use separate outputs; this is a latent correctness risk rather than an established corruption of current results.

### F10 — P2 — Resumed CSVs log stale LR/WD/EMA during the first accumulation window

The schedules are correctly fast-forwarded before training (`src/train_patch.py:524-531`), which writes the resumed LR and WD into the optimizer. Immediately afterward, however, the scalar variables used only for logging are reset to `start_lr`, initial WD, and initial EMA (`:581-583`). They are not refreshed until the first optimizer-step boundary (`:768-773`), while CSV rows are written for every microbatch (`:839-842`).

For `rep_random_s1234`, resumed from completed epoch 26 with accumulation 8:

- expected optimizer state before the first epoch-27 update: LR `0.0002211681864`, WD `0.0967780037`, EMA `0.9966308769`;
- CSV iterations 1–7 instead report `.0001/.04/.996`;
- iteration 8 reports `0.0002211659383/0.0967815209/0.9966309159`, exactly the expected next schedule values.

**Impact:** model training and scheduler phase are correct. Only the first `accum_steps - 1` CSV rows after each process launch are mislabeled. This is material for iteration-level schedule forensics, but it does not invalidate the paused checkpoint or imply an LR reset. The run has no completed epoch-50 checkpoint/probe and must not be counted as a completed replication.

## AUC, ordering, and precision status

- `roc_auc_score(labels, sigmoid(logits))` is used without thresholding (`src/eval_downstream.py:299-336`).
- The positive target is the dataset's integer `glaucoma` value (`src/datasets/oct_volumes.py:120-123`).
- Test loaders use fixed lexical volume order and no shuffle.
- The active frozen fp32 result builders set `use_amp: false`; the legacy remote sweep omitted the key and therefore ran fp16 under current defaults.
- The feature cache distinguishes `amp` from `fp32`, so a same-directory fp16 cache is not selected by an fp32 run. Encoder identity is partially included as described in F9.
- Validation AUC selects the head epoch; Test AUC does not select a checkpoint.
- MeanPool gives exactly one probability per volume. It cannot model axial order.

## Tests run and coverage

Command:

```powershell
$env:MPLBACKEND='Agg'; $env:PYTHONDONTWRITEBYTECODE='1'; & 'D:\jepa_phase0\.venv\Scripts\python.exe' -m pytest -q tests\test_pred_target_k.py tests\test_mirage_config_wiring.py tests\test_mirage_envelope.py::test_paired_transform_matches_unpaired_image_path tests\test_mirage_envelope.py::test_context_excludes_targets tests\test_mirage_envelope.py::test_bootstrap_epoch_is_unbiased tests\test_mirage_envelope.py::test_block_sizes_are_unchanged_by_guidance tests\test_mirage_anatomy_mode.py::test_mirage_anatomy_refuses_without_pred_target_k tests\test_mirage_anatomy_mode.py::test_emits_four_connected_targets_of_exactly_k tests\test_mirage_anatomy_mode.py::test_context_excludes_the_full_union tests\test_mirage_anatomy_mode.py::test_generator_emits_edge_connected_targets_when_enabled tests\test_slice_cache.py::test_cached_slices_are_bit_identical
```

Environment: `MPLBACKEND=Agg`, `PYTHONDONTWRITEBYTECODE=1`.

**Result: 31 passed in 36.68 s.**

Covered by existing tests:

- fixed-K target resampling, valid indices, and context exclusion;
- anatomy K=16 requirement and connected-target behavior;
- paired image/guide crop alignment;
- envelope ramp-off behavior and target/context contract;
- occupancy-threshold config wiring;
- slice-cache byte identity.

Not covered by existing tests:

- COVER placement followed by final collation and post-collation logging;
- `enc_truncate` epoch seeding;
- checkpoint RNG/best-state restoration;
- final partial gradient accumulation;
- exact LR/WD/EMA resume equivalence;
- truthful LR/WD/EMA logging immediately after resume;
- 2-D position-vector axis semantics;
- downstream normalization/precision propagation;
- target-vs-online checkpoint selection;
- cache invalidation and case-ID ordering.

## Surfaces read

Principal code: `src/train_patch.py`, `src/helper.py`, `src/utils/schedulers.py`, `src/utils/tensors.py`, `src/models/vision_transformer.py`, `src/models/attentive_pool_minimal.py`, `src/masks/{multiblock,curriculum,anatomy,cover}.py`, `src/datasets/{oct_slices,oct_slices_guided,oct_volumes}.py`, `src/guides/{mirage_envelope,tissue_truth}.py`, `src/transforms.py`, and `src/eval_downstream.py`.

Configs/scripts: the six policy configs, frozen-MeanPool configs, `campaign_chain.py`, `campaign_supervisor.py`, `chain_blob_fp32.py`, `probe_blob_epochs.py`, `probe_hf_checkpoints.py`, `run_ep50_probe_pipeline.ps1`, `run_linear_sweep.sh`, and `run_patch.sh`.

Bounded external evidence: selected run-directory filenames, generated campaign configs, trainer provenance lines, small `results.json` files, and prediction-label hashes. No process was inspected or stopped.

## Commands and limitations

- Confirmed HEAD equals the requested baseline and there are no source/config/script diffs from it.
- Ran only the targeted CPU tests above and small synthetic invariant checks.
- Reconstructed the paused replication's LR/WD/EMA schedule on CPU; no checkpoint tensor load or training was performed.
- Did not train, use the GPU, download/install anything, create caches, load 1.5 GB checkpoint tensors, or alter run/data state.
- The COVER reproduction is synthetic and establishes the code path, not its exact real-data population effect.
- Learning/AUC consequences of F3–F6 require controlled reruns; none are inferred here.
- Existing run artifacts establish current naming and bounded metadata, but checkpoints do not embed a parent hash or complete launch config, limiting retrospective ancestry proof.
