# Lessons Learned

Mistakes, debug-traps, and invariants we've paid to learn. Keep these visible so they don't sneak back in.

---

## Pretraining

### 1. LR=0.0005 too high for OCT
- **What happens:** Model learns fine during warmup, diverges once LR hits peak.
- **Why:** OCT is less diverse than ImageNet, gradients are more correlated, effective LR is higher than nominal. sqrt-scaling from the I-JEPA paper's LR overestimates.
- **Rule:** For OCT + ViT-B/16 + effective batch 512, **peak LR = 0.00025**.

### 2. Pre-warmup val loss is artificially low
- **What happens:** Before warmup ends, the EMA target encoder hasn't diverged from the online encoder. Prediction is trivial → val_loss looks great → nothing ever beats it again → patience counter / best-checkpoint save latches onto epoch 1.
- **Rule:** Every best-AUC / best-loss / patience decision must be **gated on `past_warmup`**. See `train_patch.py` + `eval_downstream.py` fine-tune path (commit `135ba2a`, off-by-one fix `0dcd9d0`).

### 3. Blocking blob uploads stall DDP
- **What happens:** Rank 0 uploads a 1.5 GB checkpoint synchronously, takes >5 min, other ranks block on the next collective → NCCL 30-min timeout → hang.
- **Rule:** Background-thread uploads during training, only `blocking=True` after the training loop ends.

### 4. DDP early-stop `break` skips `dist.broadcast`
- **What happens:** Rank 0 hits `break` inside `if is_main:` before reaching `dist.broadcast()`. Other ranks sit on the broadcast forever.
- **Rule:** Use a `should_stop` flag, broadcast, THEN break on all ranks.

### 5. Grad accumulation must gate scheduler + EMA
- **What happens:** Scheduler / EMA step every micro-batch. With `accum_steps=2`, scheduler runs 2× too fast.
- **Rule:** Only step scheduler / EMA inside `if (itr + 1) % accum_steps == 0:`.

### 6. DDP val-loss must be all-reduced across ranks
- **What happens:** Each rank sees a different `val_loss` (its own shard). Early-stop decisions diverge across ranks → NCCL hang at next collective.
- **Rule:** `dist.all_reduce(sum, count)` → global mean before any rank-comparison decision. Fixed in commit `135ba2a`.

---

## Downstream

### 7. Encoder representations are the bottleneck, not probe capacity
- **What happens:** Probe depth d=3 (21M) vs d=1 (7M) moves Val AUC by ~0.002. 3× capacity, 0× gain.
- **Rule:** When a frozen probe plateaus, don't scale the probe. Unfreeze the encoder or change the pretraining source.

### 8. 100 slices OOM with encoder gradients
- **What happens:** Unfrozen ViT-B/16 at bs=1 × 100 slices blows past 16 GB on T4. bs=1 × 64 slices fits (~11 GB).
- **Rule:** Frozen probe can use all 100 slices. Unfrozen fine-tune: max 64 on T4 16GB.

### 9. Eval preprocessing must match pretraining
- **What happens:** Pretraining normalizes with ImageNet mean/std; downstream forgets to. Frozen probe AUC drops ~10 pts.
- **Rule:** `imagenet_normalize()` before the encoder in both frozen and unfrozen paths. Applied in `eval_downstream.py`.

### 10. Attentive probes overfit small medical datasets
- **What happens:** Even d=1 (7M params) + weight_decay=0.05 + dropout=0.2 hits train AUC → 1.0 by epoch 10-15 on 6K samples. Val AUC peaks at epoch 4-8 then drifts.
- **Why:** 7M params / 6K samples ≈ 1200 params/sample. Small-data attentive probes are known to over-parameterize — see [Attention, Please! ICLR 2026](https://arxiv.org/abs/2506.10178).
- **Rule:** Frozen-probe overfit pattern is normal and invisible in the paper (only best-val is reported). Ceiling is the encoder, not the probe.

### 11. Print buffering hides training progress under `tee`
- **What happens:** Python `print()` is block-buffered (~4KB) when piped. Per-epoch progress sits in the buffer for ~40 epochs.
- **Rule:** `sys.stdout.reconfigure(line_buffering=True)` at `__main__` (or `flush=True` on every print). `train_patch.py` does it via a `log()` helper; `eval_downstream.py` sets line buffering globally (commit `61f08c3`).

### 12. Linear scaling rule for LR with batch size
- **What happens:** Copy a literature LR verbatim without accounting for batch size. Literature's bs=1024 at LR=1e-3 ≠ our bs=256 at LR=1e-3.
- **Rule:** `LR_ours = LR_ref × (bs_ours / bs_ref)`. For bs=256 from a bs=1024 reference, LR=4e-4.

### 13. Torchrun port conflict with multiple DDP jobs on same node
- **What happens:** Two DDP jobs on the same compute both bind port 29500. Second crashes with `Address already in use`.
- **Rule:** Unique `MASTER_PORT` per job if co-scheduled, or run sequentially.

### 14. Shell scripts need LF line endings for bash on Linux
- **What happens:** Windows-edited scripts get CRLF. AML Linux bash chokes with `$'\r': command not found`.
- **Rule:** `.gitattributes` has `*.sh text eol=lf`, but the working copy can still drift. Run `sed -i 's/\r$//' scripts/*.sh` before submitting.

### 17. A config value that two components read must be passed to BOTH
- **What happens:** `configs/patch_mirage_envelope.yaml` set `mirage_occupancy_threshold: 0.25`. The collator read it from the curriculum config, but `train_patch.py` built `GuidedOCTSliceDataset` without the kwarg, so the dataset kept its `0.5` default. The dataset builds the *placement region* (where a target block may go) and the collator builds the *scoring truth* (what "on retina" is measured against) — so blocks were **placed from one grid and scored against another**. Nothing crashed; no logged metric looked wrong. Measured cost: infeasible blocks +175%, admissible region −10%.
- **Rule:** When a single config key drives behaviour in more than one component, assert the wiring in a test rather than trusting the call site. `tests/test_mirage_config_wiring.py` parses the constructor call with `ast` and fails if the kwarg is missing. Prefer a default that is *obviously invalid* over a plausible one — a silently-wrong 0.5 is far worse than a crash.

### 18. `.npz` member access decodes the WHOLE array — 200× read amplification
- **What happens:** `np.load(path)["oct_bscans"]` decompresses the entire (200, 200, 200) 8 MB volume even when you index one slice out of it. Reading a 40 KB slice costs 8 MB of I/O. Training ran at **7.6 img/s with the GPU idle at 0–1% and 30 W** (~68 days) while the GPU ceiling was 177.7 img/s (~2.9 days).
- **Why it hides:** benchmarks on a few hundred volumes fit in the OS page cache and look fine. It only appears once the working set (48 GB) exceeds RAM (32 GB).
- **Rule:** Always confirm whether a run is GPU-bound before trusting a throughput number — `nvidia-smi` power draw is the fastest tell (a busy 3090 pulls 250 W+; 30 W means starved). For slice-sampled volume datasets use `scripts/build_slice_cache.py`, which stores exactly the sampled slices in a flat memmap. Assert bit-equality against the original path so the cache can never change training data.

### 19. Validation DataLoader workers stack on top of the training loader's
- **What happens:** the val loader was created with `num_workers=data_cfg['num_workers']` (6) and re-spawned every epoch while the training loader's 6 were still alive. On Windows this exhausted the system commit limit and killed the run mid-validation with `RuntimeError: Couldn't open shared file mapping ... error code: 1455`.
- **Rule:** Give validation its own small worker count (`val_num_workers: 2`) — it is a small fraction of wall time. Never run heavy analysis jobs on the training box. And keep `save_every` small (5, not 25): a crash at epoch 32 with 25-epoch saves could only fall back to epoch 27.

### 20. Masking purity does not predict downstream AUC
- **What happens:** the MIRAGE-guided arm masks the retina *more purely* than the hand-crafted oracle band (target purity 0.632 vs 0.560; targets-on-retina 0.506 vs 0.458; better on 63% of slices) and still lands **below** it on the frozen MeanPool probe at ep100 (0.8807 vs 0.8855, paired-bootstrap 95% CI [−0.0091, −0.0002]). Every offline masking-quality metric the program built pointed the wrong way.
- **Why it hides:** purity merges all segmentation classes. Re-scored per class (`scripts/mirage_vs_oracle_region_split.py`, 2,374 slices), **96.8% of MIRAGE's extra on-tissue masking is choroid** while inner-retina coverage is unchanged (ratio 1.007) — the purity gain was real and diagnostically inert.
- **Rule:** A masking prior is only ever validated by downstream AUC (general rule 2). Never select a policy on a proxy metric that has not itself been shown to correlate with AUC. If a proxy must be used, make it *region-specific* (inner-retina coverage), not whole-envelope purity.

### 21. An offline policy sweep must run the config that will train
- **What happens:** the threshold-0.25 policy was chosen because it matched the oracle arm's masked area to within 0.5% (unique targets 101.7 vs 101.9). `scripts/mirage_method_sweep.py` measured every MIRAGE row with `mirage_spread=False` (its `Method` dataclass defaults `spread: bool = False`), but `configs/patch_mirage_envelope.yaml` sets `mirage_spread: true` — and so does `CurriculumMaskGenerator`'s own default. Re-measured on the same slices with the same seeds: 100.9 vs 108.8 unique targets, **+7.8%**, not +0.2%. The trained comparison therefore varies masked area and placement entropy as well as target location, which is exactly what the sweep existed to prevent.
- **Rule:** Same class as #17 — a policy sweep must construct its samplers **from the shipped config file**, not from a hand-written dict that silently re-defaults keys. Assert in a test that the sweep's effective curriculum config equals the training config for every key it does not deliberately vary.

---

## Fine-tuning

### 15. Layer-wise LR decay (LLRD) is standard for ViT fine-tuning
- **Rule:** γ=0.65 for ViT-B, γ=0.75 for ViT-L, MAE convention. Top encoder layers get the full base LR; bottom layers get ~base × γ^num_layers. Our single flat `lr_encoder` was leaving 500× LR on the table for top encoder layers. Implemented in `build_finetune_param_groups` (commit `f78876f`).

### 16. Warmup gate also applies to fine-tune best-save and patience
- **Rule:** Same bug class as pretraining #2. `epoch > warmup_epochs` (fine-tune loop is 1-indexed, not 0-indexed like pretraining — commit `0dcd9d0` fixes the off-by-one).

---

## Code Differences from Official I-JEPA

Intentional, not bugs:

| Aspect | Official | Ours | Impact |
|---|---|---|---|
| Momentum schedule | Linear | Cosine | Negligible for our EMA ranges |
| Target path AMP | Under autocast | fp32 (no autocast) | Slightly more precise targets |
| LayerNorm epsilon | 1e-6 | 1e-5 (PyTorch default) | Minor |
| CLS token | Interpolation code present | No CLS, direct pos_embed add | Cleaner, avoids the no-CLS interpolation bug |
| Target block sizes | **One `p_size` shared by all `npred` blocks** | 4 independently sampled sizes | **Not benign.** The collator truncates every target in a batch to the shortest, and indices are row-major sorted, so truncation shears the bottom row off a block. Official's truncation is a no-op (0.00% loss); ours discards ~10% of target patches and leaves ~37.5% of delivered blocks non-rectangular. Shared by all arms and by the ep25 checkpoint, so it is left alone rather than introducing a second variable — but fix it in a v2. See [`docs/experiments/masking/ablations.md`](experiments/masking/ablations.md#open-blockers-before-training). |

---

## General rules

1. **I-JEPA loss is NOT a reliable quality metric.** Low loss can mean collapse; high loss can mean healthy learning. Monitor `rep_diversity` and `cos_sim` instead.
2. **Downstream AUC is the quality signal.** Use the linear probe sweep across ep25/50/75/100 to pick the pretraining checkpoint.
3. **No early stopping for final pretraining runs.** Literature standard (RETFound, V-JEPA) is fixed-epoch.
4. **Upload checkpoints to blob during training.** Don't wait for the end — jobs crash.
5. **DDP cleanup (barrier + destroy) must run on all ranks before any rank-specific code.**
6. **Never revert local-only configs** (blob-storage accounts, compute names) to placeholders when committing.
