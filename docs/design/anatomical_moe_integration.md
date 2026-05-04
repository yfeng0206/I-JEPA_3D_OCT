# Design: Integrating AnatomicalMoEPool into the Downstream Pipeline

**Status**: 2026-05-04 (revised), branch `volume-moe`.
**Module**: `src/models/anatomical_moe_pool.py`.

> **Important note for readers (current state):**
> The recommended path is now **`moe_scope: volume`** with `skip_wq: true` and
> `axial_pos_embed: learned`. Sections 1–4 below describe the **per-slice**
> design that was the initial proposal; it remains supported as an
> ablation but is **not** the primary configuration. See section 9 for the
> volume-scope design and audit fixes (routing scaling, optimizer plumbing
> for axial pos embed, runnable config).

This document captures every architectural decision in the integration. It exists so reviewers
can challenge each choice independently before any commit lands.

---

## 1. The current pipeline (what we're modifying)

### Frozen-probe path (`eval_downstream.py` `precompute_features`)

```
OCT volume                       (1, S=64, 3, H=256, W=256)
    │
    ▼  squeeze + chunked
per-slice batch                  (chunk, 3, 256, 256)
    │
    ▼  encoder.forward()
per-slice patch features         (chunk, 256, 768)        ← 256 patches per 16x16 grid
    │
    ▼  out.mean(dim=1)            ← *** THE STEP WE'RE REPLACING ***
per-slice mean-pooled token      (chunk, 768)
    │
    ▼  cat across slices, store as cache
volume tensor                    (S=64, 768)
    │
    ▼  probe(volume)
volume representation            (1, 768)
    │
    ▼  head
glaucoma logit                   (1,)
```

### Fine-tune path (`eval_downstream.py` `DownstreamModel.forward`)

```
OCT volume                       (B, S=64, 3, H, W)
    │
    ▼  flatten + chunk + encoder + mean   (same as above per slice)
slice tokens                     (B, S=64, 768)
    │
    ▼  probe + head
logit                            (B,)
```

The *only* difference between frozen and fine-tune is whether encoder weights are updated.
**Both paths funnel through the same `out.mean(dim=1)` operation.**

---

## 2. The proposed pipeline (what we're adding)

### Frozen-probe path with AnatomicalMoEPool

```
OCT volume                       (1, S=64, 3, 256, 256)
    │
    ▼  squeeze + chunk
per-slice batch                  (chunk, 3, 256, 256)
    │
    ▼  encoder.forward()
per-slice patch features         (chunk, 256, 768)
    │
    ▼  anatomical_moe_pool(out)   ← *** NEW STEP ***
per-slice prototype tokens       (chunk, E*S=32, 768)     ← 32 anatomical prototypes per slice
    │
    ▼  cat across slices, store
volume tensor                    (S=64, E*S=32, 768)
    │
    ▼  flatten the (S, E*S) axes for probe input
volume token sequence            (S*E*S=2048, 768)
    │
    ▼  probe(volume)              ← NB: probe sees 2048 tokens not 64
volume representation            (1, 768)
    │
    ▼  head
glaucoma logit                   (1,)
```

The fine-tune path mirrors this — same swap, same downstream consumption.

---

## 3. Design decisions, with rationale

Each decision is intentionally listed separately so they can be challenged independently.

### D1. Where in the pipeline does the new pool sit?

**Decision**: replace the `out.mean(dim=1)` operation, leaving everything upstream
(encoder forward) and downstream (probe + head) intact.

**Why**: the mean is the only lossy collapse in the pipeline. Encoder is doing real work and
shouldn't change for this paper. Probe is doing real work and we want to keep the existing
sweep results comparable. The mean is the architectural weakness; surgically replacing it
isolates the contribution.

**Alternatives considered**:
- *In-block MoE-FFN (PaMoE-style)*: requires re-pretraining the encoder. Too risky.
- *Predictor-side MoE (M3-JEPA-style)*: would require re-pretraining JEPA. Save for follow-up.
- *MoE inside the probe*: doesn't address the mean-pool information loss; downstream of it.

### D2. Output shape: (B, E*S, embed_dim) per slice — why this choice?

**Decision**: each slice produces `E*S` prototype tokens of dim `embed_dim`. With E=8, S=4 →
**32 prototypes per slice**. For a 64-slice volume, **2048 prototype tokens total**.

**Why**:
- `E*S` is the natural output count for soft-MoE with multiple experts and slots per expert.
- Keeping `embed_dim` as the per-prototype dim means downstream probes can be reused without
  shape changes — they still see `(B, num_tokens, embed_dim)`, just with `num_tokens=2048`
  instead of `num_tokens=64`.
- 32 prototypes per slice is in the right ballpark for retinal-layer biology (vitreous, RNFL,
  GCL, IPL, INL, OPL, ONL, ELM, OS, RPE, choroid, sclera = ~12 named layers; 32 gives slack
  for sub-regional specialization).

**Param budget**: with `E=8, S=4, H=8, slot_dim=256, lora=16, embed=768`, the module is
**~316K params** — between CrossAttnPool (277K) and AttentiveProbe d=1 (7.17M). Reasonable.

### D3. How does the existing probe consume `(B, S, E*S, D)`?

**Decision**: flatten the `(S, E*S)` axes to a single sequence dim → `(B, S*E*S, D)`. The
probes operate on this longer sequence as if it were a sequence of tokens.

**Why**:
- *Cleanest integration*: probe forward functions take `(B, N, D)`. **MeanPool** is
  truly shape-agnostic over `N`. **CrossAttnPool** and **AttentiveProbe** are NOT —
  their learnable `pos_embed` is sized to `num_slices`, so we must (a) construct them
  with `num_slices = S*E*S` from the start AND (b) drop pos_embed entirely (D4),
  because there is no meaningful axial ordering along the prototype axis. So in the
  MoE path the probes are *re-instantiated* with `use_pos_embed=False` and
  `num_slices = S*E*S`, not "the same probe with a longer sequence".
- *Clean ablation*: MeanPool over all 2048 tokens vs MoE prototypes is "mean over patches
  per slice, then mean over slices" vs "anatomical-MoE per slice, then mean over all
  prototypes" — informative comparison.
- *Memory (prototype features)*: 2048 tokens × 768 dim per volume = ~6 MB. Frozen-probe
  cache of POOLED prototypes for 3000 volumes ≈ 18 GB. **But the trainable-MoE design
  requires caching RAW patch features instead** (256 × 768 per slice instead of 32 × 768)
  → 8× larger, ~150 GB at fp32 for 3000 vols, ~800 GB across all splits.
  See D9 for the caching strategy. This is the dominant resource constraint.

**Alternatives considered**:
- *Hierarchical probe (within-slice pool, then across-slice pool)*: more principled but
  more code, and risks throwing away cross-slice anatomical relationships.
- *Aggregate prototypes within slice first*: defeats the point — we lose the prototype-level
  granularity that makes the interpretability story possible.

### D4. Position-embedding strategy for probes consuming the longer sequence

**Decision**: drop position embedding entirely for the AnatomicalMoEPool path. Specifically,
add a `use_pos_embed: bool` flag to `CrossAttnPool` and `AttentiveProbe` that defaults to
True (current behaviour) but can be set False for the new path.

**Why**:
- The encoder *already* injects positional information via its 2D sincos pos_embed at the
  patch level. Each prototype token is a soft-attention-weighted sum of patch tokens that
  retain that positional signal in their feature values.
- Slice axis position is *NOT* preserved in the prototype output (the soft-MoE pool is
  permutation-equivariant over its 256 input patches, but the output `E*S` tokens have no
  intrinsic axial meaning).
- This is exactly the V-JEPA argument for why their attentive probe has no pos_embed:
  "patch tokens whose spatial positions are encoded by the ViT's own pos_embed" (cited in
  our existing `attentive_pool_minimal.py` docstring).
- *Slice-axis info loss*: small. Glaucoma signal localizes per-slice (RNFL at slice ~63),
  but the relevant info gets baked into the prototypes via the encoder's spatial reasoning
  before prototype aggregation. If we lose 0.5pp from this, that's the cost.

**Alternative to consider for v2** (if v1 disappoints): add a 2D additive pos_embed
`pos[s, k] = slice_pos[s] + proto_pos[k]` learnable, of size `(S * E*S, embed_dim)`. Adds
~1.5M params for `S=64, E*S=32, D=768`.

### D5. Config-flag-driven A/B selection

**Decision**: add a single config field `pool_type: mean | anatomical_moe` (default `mean`).
This routes through the existing path or the new path based on config.

**Why**:
- A/B comparisons need to be one-flag toggles for clean experiments.
- Default `mean` preserves all existing reproducibility (current commit reproduces current
  numbers).
- Don't break the existing pipeline by silently changing behavior.

### D6. AnatomicalMoEPool hyperparameters — defaults

**Decision**: ship with these defaults, exposed via config:
```yaml
pool_type: anatomical_moe
anatomical_moe:
  num_experts: 8
  num_slots: 4
  num_heads: 8
  slot_dim: 256
  lora_rank: 16
  share_phi: true
  dropout: 0.0
```

**Why each value**:
- `num_experts=8`: MAMMOTH uses 30 for ~10K WSI patches. We have 256 OCT patches, so scaled
  proportionally: 30 × (256/10000) ≈ 0.8 — minimum useful is ~4–8. Ship at 8, ablate later.
- `num_slots=4`: MAMMOTH uses 9. We scale down to match the lower expert count.
- `num_heads=8`: MAMMOTH uses 16. We use 8 because slot_dim=256 with H=8 gives head_dim=32
  — comparable to MAMMOTH's head_dim_input.
- `slot_dim=256`: MAMMOTH default. Reduces 768→256 dim before routing; cheaper attention.
- `lora_rank=16`: MAMMOTH default. Shared-Phi keeps params manageable.
- `share_phi=true`: MAMMOTH ablation showed −1.4% from removing weight sharing. Keep it.
- `dropout=0.0`: start clean; add regularization in Phase B if overfitting.

### D7. Slot prototype initialization

**Decision (v1)**: random `trunc_normal_(std=0.02)`, exactly per MAMMOTH default.
Add `init_slots_from_kmeans(cluster_centers)` API for Phase B warm-start ablation.

**Why**:
- Random init is the baseline. We need to know what the model can do without external priors.
- K-means warm-start (PaMoE trick) is a defensible Phase B ablation; if random init slot
  specialization is incoherent, this is the recovery move.
- We don't have a CONCH-equivalent for OCT. K-means over our own ViT features is the OCT
  analog.

### D8. Loss / regularization additions

**Decision**: no additional loss terms in v1.

**Why**:
- Soft-MoE is empirically stable without load-balancing losses (Puigcerver 2024). We don't
  need the Switch-Transformer-style aux losses that hard MoE requires.
- Adding too many knobs at once muddies the ablation signal.
- If slot collapse is observed empirically, we add an entropy regularizer in v2.

### D9. Caching strategy

**Decision**: separate cache directory for AnatomicalMoEPool features, keyed by config.

**Why**:
- Cache shape changes from `(N, S, D)` to `(N, S*E*S, D)` — incompatible with mean cache.
- Cache contents depend on AnatomicalMoEPool weights (it's a learned module, not a
  deterministic mean). For the **frozen-probe** path the AnatomicalMoEPool weights are
  trained jointly with the probe, so the cache must be invalidated when those weights
  change — practically, cache only the **encoder output** (256, 768) per slice, then run
  AnatomicalMoEPool live.

**Implication**: the caching layer needs adjustment. For v1, just disable the cache when
`pool_type=anatomical_moe` (recompute encoder forwards each epoch — slower but correct).
Phase B cleanup: cache the (256, 768) encoder outputs and run AnatomicalMoEPool live on top.

---

## 4. Concrete code changes — list of edits

These are the surgical changes needed to wire AnatomicalMoEPool through. Each
references the design decision above.

### Change 1: `src/models/attentive_pool_minimal.py`
- Add `use_pos_embed: bool = True` flag to `CrossAttnPool.__init__` (D4).
- Skip `x = x + self.pos_embed` when `use_pos_embed=False`.
- Same flag for `AttentiveProbe` in `eval_downstream.py`.
- MeanPool needs no change (no pos_embed).

### Change 2: `src/eval_downstream.py` `_build_probe`
- Read new config flags (D5, D6).
- When `pool_type=anatomical_moe`, instantiate AnatomicalMoEPool and pass it
  to the encode function.
- Pass `use_pos_embed=False` to probes when `pool_type=anatomical_moe` (D4).

### Change 3: `src/eval_downstream.py` `precompute_features`
- Accept an optional `aggregator` argument (the AnatomicalMoEPool module or None).
- When `aggregator is None`: existing `out.mean(dim=1)` path.
- When `aggregator is not None`: `aggregator(out).reshape(...)` path (D1, D2, D3).
- For frozen probe: cache the (256, 768) raw encoder features instead of pooled, run
  aggregator at probe time (D9).

### Change 4: `src/eval_downstream.py` `DownstreamModel.forward`
- Accept an optional `aggregator` module attribute.
- When set, replace `out.mean(dim=1)` with `aggregator(out).reshape(...)`.

### Change 5: Add config templates
- `configs/downstream_patch_anatomical_moe.yaml` — example config for the new path.

---

## 5. What we're NOT changing

- Encoder architecture (ViT-B/16): unchanged.
- Encoder pretraining (JEPA SSL): unchanged. Use existing ep100 random-init checkpoint.
- Existing config files: unchanged. New configs are additive.
- Existing probe forward shapes: probes still take `(B, N, D)`. Just `N` differs.
- Existing test suite: unchanged. New module has its own test in `scripts/`.
- Existing reproducibility: setting `pool_type=mean` (default) reproduces all current numbers.

---

## 6. Risks specific to this integration

| Risk | Likelihood | Mitigation |
|---|---|---|
| `pool_type` flag plumbing introduces config-key typos | Medium | Validate in `_build_probe` with explicit error |
| 2048-token sequences blow up probe memory at fine-tune time | Low | A100 80GB; ~6MB per volume, batch=1-4 fits easily |
| Disabling pos_embed costs measurable AUC | Medium | Phase B will test; can reintroduce as 2D pos_embed |
| Slot init from random doesn't specialize | Medium | Phase B k-means warm-start fallback; PaMoE confirmed this trick works |
| Caching invalidation breaks reproducibility | Low | Disable cache for new path in v1; reintroduce after correctness verified |

---

## 7. Validation plan

1. **Smoke test (done)**: `scripts/test_anatomical_moe_pool.py` — 8 cases, all pass.
2. **End-to-end test (next)**: run the new path on 50 volumes, verify shapes flow correctly
   through encoder → AnatomicalMoEPool → probe → head → loss.
3. **Frozen probe sweep with AnatomicalMoEPool + each of 3 probes** (Phase A — the make-or-break).
4. **Fine-tune sweep with AnatomicalMoEPool + best frozen probe** (Phase A continuation).
5. **Hyperparameter sweep** (Phase B): E, S, H, lora_rank, share_phi, slot init.

---

## 8. Decision log (date-keyed)

- **2026-04-27**: Initial design captured. AnatomicalMoEPool module written and smoke-tested.
  Integration not yet started. This doc is the gating artifact.

---

## 9. Current canonical design — volume scope (2026-05-04)

The recommended config:

```yaml
pool_type: anatomical_moe
anatomical_moe:
  moe_scope: volume
  skip_wq: true
  axial_pos_embed: learned
  num_experts: 8
  num_slots: 4
  num_heads: 8
  lora_rank: 16
  share_phi: true
```

**Data flow** (replacing the per-slice flow in section 2):

```
encoder(64 slices)  →  (B, 64, 256, 768)
                            ↓  + axial_pos_embed[s]   (1, 64, 1, 768) learnable
                       (B, 64, 256, 768)
                            ↓  reshape
                       (B, 16384, 768)
                            ↓  AnatomicalMoEPool (one call, skip_wq=True)
                       (B, E*S=32, 768)
                            ↓
                          Probe sees 32 tokens
```

**Why volume + skip_wq + learned axial:**
1. One MoE call per "slide-equivalent" (= one volume) — true MAMMOTH analog.
2. `skip_wq=True` keeps slot prototypes in encoder feature space; ~183K params saved.
3. `axial_pos_embed=learned` lets prototypes specialize by axial slice position
   (peripapillary vs macular vs peripheral), which is where OCT clinical signal
   actually lives.
4. Probe sees only 32 tokens instead of 2048 — AttentiveProbe is now trivial.

### 9.1 Audit fixes applied (GPT 5.5 review, 2026-05-04)

The first volume-scope implementation had two P0 bugs and one P1 design issue that
have been fixed:

**P0-1 — `axial_pos_embed: learned` was created as `nn.Parameter` on `DownstreamModel`
but was NOT included in `build_finetune_param_groups` and was NOT saved/loaded in
checkpoints.** Result: the axial pos embed received gradients during backward but
was never stepped by the optimizer — effectively a random fixed offset. Fixed:
- `build_finetune_param_groups` now accepts `axial_pos_embed=` and emits a separate
  param group at base/probe LR.
- `run_patch_finetune` threads `raw.axial_pos_embed` into that call.
- Checkpoint save/load includes `axial_pos_embed` key.
- Regression test `test_finetune_param_groups_includes_axial_pos_embed` guards.

**P0-2 — `configs/downstream_patch_anatomical_moe.yaml` defaulted to frozen-probe
path because `freeze_encoder` was unset (`main()` defaults to `True`).** The frozen
path then raised `NotImplementedError` for `pool_type=anatomical_moe`. Fixed: the
config now sets `freeze_encoder: false`, `batch_size: 1`, `accum_steps: 16`,
LLRD with `layer_decay: 0.5`. Documents itself as a fine-tune config in the header.

**P1-1 — Routing logits were unscaled.** Standard attention divides by
`sqrt(head_dim)`. Without scaling, with `skip_wq=True` (so head_dim=96 not 16),
softmax over 16K tokens collapsed to ~hard top-2 (2.8 effective tokens out of
16K, top-1 weight 0.74). Fixed: `AnatomicalMoEPool.forward` applies
`routing_scale = head_dim**-0.5`. Verified post-fix: top-1 weight 0.002,
~10K effective tokens routed — true soft pooling. Regression test
`test_routing_scaling_keeps_dispatch_soft` guards.

### 9.2 Outstanding limitations

- **Frozen probe + MoE: still unsupported.** `run_patch_downstream` raises
  `NotImplementedError` for `pool_type != 'mean'`. The cache strategy
  (~150-800 GB raw patch features) is the open question.
- **`skip_wq` interpretability claim is partial:** routing happens in encoder
  feature space (so slot prototypes are interpretable via routing weights),
  but output prototype tokens go through `phi → ReLU → expert_w → norm_out`
  before reaching the probe. The "prototypes ARE in encoder space" framing
  applies to routing-side analysis, not to the output tokens.
- **AML launcher exposes the new config fields** via `POOL_TYPE`, `MOE_SCOPE`,
  `MOE_SKIP_WQ`, `MOE_AXIAL_POS`, etc. environment variables (see
  `scripts/run_downstream.sh:187-211`). Default `mean` reproduces the
  existing flow.
