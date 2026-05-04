# Anatomical Mixture-of-Experts (AnatomicalMoEPool)

Within-volume soft mixture-of-experts that replaces the per-slice mean-pool in the OCT
fine-tune pipeline. Adapted from MAMMOTH (Shao et al., ICLR 2026). Default behavior is
unchanged when `pool_type: mean` (or omitted) in the config — the new path is fully opt-in.

## 1. Motivation

The standard pipeline collapses 256 patch tokens per slice into a single 768-d slice token
via `out.mean(dim=1)`. This is the only lossy aggregation step in the OCT downstream pipeline
and discards within-slice anatomical structure (vitreous, RNFL, IPL, INL, OPL, ONL, RPE,
choroid, sclera).

AnatomicalMoEPool replaces that mean with a learned soft mixture of `E*S` slot prototypes.
Each prototype softly pools the patches that most resemble it; together the `E*S` prototypes
form an anatomical decomposition of each volume. The probe then operates on these prototype
tokens.

## 2. Architecture overview

| Stage | Shape (volume) | Notes |
|---|---|---|
| Encoder per slice | `(64, 256, 768)` | ViT-B/16, 16x16 patch grid per slice |
| Stack across slices | `(B, 64, 256, 768)` | one slice = 256 patches |
| `+ axial_pos_embed[s]` (volume scope only) | `(B, 64, 256, 768)` | broadcast over patches; encodes axial slice position |
| Reshape (volume scope) | `(B, 16384, 768)` | one MoE call per volume |
| AnatomicalMoEPool | `(B, E*S, 768)` | E=8, S=4 by default → 32 prototypes per volume |
| Probe | `(B, 768)` | MeanPool / CrossAttnPool / AttentiveProbe (no pos_embed in MoE path) |
| Linear head | `(B, 1)` | binary glaucoma logit |

## 3. Math, step by step

For one volume input to the aggregator (volume scope, `skip_wq=True`, `E=8, S=4, H=8`,
`head_dim=96`, `lora_rank=16`):

```
x:                       (B, N=16384, 768)            after axial pos add + flatten
q  = LayerNorm(x)        (B, N, 768)                  no wq projection (skip_wq=True)
q_heads = q.view(...)    (B, N, H=8, head_dim=96)     split into 8 heads

# Routing logits (per-head similarity vs slot prototypes), scaled by 1/sqrt(head_dim)
slot_embeds:             (E=8, H=8, S=4, head_dim=96)  trainable prototypes
logits = einsum(q_heads, slot_embeds_norm) * head_dim**-0.5
                         (B, N, E, H, S)

# Soft slot pooling: softmax over TOKENS (not over experts)
dispatch = softmax(logits, dim=N=tokens)
                         (B, N, E, H, S)              each (e,h,s) sums to 1 over N
slots = einsum(q_heads, dispatch)
                         (B, E, H, S, head_dim=96)   weighted patch summary per (e,h,s)

# Factorized expert: shared Phi (per head) + per-expert W_low
r = einsum(slots, Phi) + Phi_bias
                         (B, E, H, S, lora=16)
r = ReLU(r)
z = einsum(r, W_low) + W_low_bias
                         (B, E, H, S, out_per_head=96)

# Concat heads back to embed_dim per (E,S) prototype
z = z.permute(...).reshape(B, E*S, embed_dim)
                         (B, 32, 768)
z = LayerNorm(z)
```

Routing scaling is critical: without `1/sqrt(head_dim)`, with `head_dim=96` the dot products
have std ~10, which collapses softmax-over-16k-tokens to near-hard top-2 selection
(top-1 weight ~0.74, ~2.8 effective tokens routed). With scaling: top-1 weight ~0.002,
~10K effective tokens routed — true soft pooling.

## 4. Configuration

Recommended config for OCT (also in `configs/downstream_patch_anatomical_moe.yaml`):

```yaml
model:
  freeze_encoder: false       # MUST be false; frozen path raises NotImplementedError for MoE
  probe_type: cross_attn_pool
  probe_depth: 1
  head_type: linear
  pool_type: anatomical_moe
  anatomical_moe:
    moe_scope: volume         # one MoE call per volume (recommended)
    skip_wq: true             # route directly in encoder feature space
    axial_pos_embed: learned  # adds (1, num_slices, 1, embed_dim) trainable
    num_experts: 8            # E
    num_slots: 4              # S
    num_heads: 8              # H — independent routing subspaces
    slot_dim: 256             # ignored when skip_wq=true
    lora_rank: 16             # bottleneck rank for factorized experts
    share_phi: true           # share Phi matrix across experts within a head
    dropout: 0.0
training:
  layer_decay: 0.5            # LLRD on encoder
  lr_probe: 2.0e-4
  lr_encoder: 2.0e-4
  lr_head: 2.0e-4
  weight_decay: 0.05
  epochs: 50
  patience: 15
  warmup_epochs: 5
  accum_steps: 16
  dropout: 0.1
data:
  batch_size: 1               # fine-tune scale; effective batch via accum_steps
  num_slices: 64              # axial_pos_embed shape locks to this
```

### Config knobs (model.anatomical_moe)

| Key | Type | Default | Notes |
|---|---|---|---|
| `moe_scope` | `'per_slice'` \| `'volume'` | `'per_slice'` | volume = one MoE call per volume; per_slice = one per slice (legacy/ablation) |
| `skip_wq` | bool | `false` | when true, `slot_dim` forced to `embed_dim` and `wq` is `nn.Identity` |
| `axial_pos_embed` | `'none'` \| `'learned'` \| `'sincos'` | `'none'` | only honored when `moe_scope=volume` |
| `num_experts` (E) | int | 8 | number of expert prototype groups |
| `num_slots` (S) | int | 4 | slots per expert |
| `num_heads` (H) | int | 8 | independent routing subspaces |
| `slot_dim` | int | 256 | routing space dim (ignored when `skip_wq=true`) |
| `lora_rank` | int | 16 | bottleneck for factorized experts |
| `share_phi` | bool | `true` | share Phi across experts within a head |
| `dropout` | float | 0.0 | dropout on slot prototypes |

### Mode selection cheat-sheet

| `pool_type` | `freeze_encoder` | Path entered | MoE active? |
|---|---|---|---|
| `mean` (default / omitted) | `true` (default) | `run_patch_downstream` (frozen probe) | no — original mean-pool |
| `mean` | `false` | `run_patch_finetune` | no — original mean-pool |
| `anatomical_moe` | `true` | `run_patch_downstream` | **raises `NotImplementedError`** |
| `anatomical_moe` | `false` | `run_patch_finetune` | **yes** — recommended path |

## 5. Where the slice count is baked in

The aggregator itself is shape-agnostic over the token axis (works on any `N`). The
constraint comes from `axial_pos_embed`:

| `axial_pos_embed` | num_slices flexibility |
|---|---|
| `none` | any num_slices works (no axial info encoded) |
| `learned` | tensor shape `(1, num_slices, 1, embed_dim)` — **locks to training num_slices** |
| `sincos` | precomputed buffer of same shape — also locks at construction |

This is the same constraint as ViT pos_embed locking to a specific image size. Standard
deployment workarounds (resize input to match training num_slices, or interpolate
`axial_pos_embed` at load time) apply. See [`docs/design/anatomical_moe_integration.md`](../design/anatomical_moe_integration.md)
section 9.2 for details.

## 6. Parameter budget

Default volume-scope config (`E=8, S=4, H=8, slot_dim=embed_dim=768, lora=16`,
`skip_wq=true`):

```
slot_embeds  (E=8, H=8, S=4, head_dim=96)         24,576
phi           (H=8, head_dim=96, lora=16)         12,288
phi_bias      (H=8, lora=16)                          128
expert_w     (E=8, H=8, lora=16, out_per_head=96)  98,304
expert_b     (E=8, H=8, out_per_head=96)            6,144
LayerNorms (norm_q, norm_slots, norm_out)           3,264
─────────────────────────────────────────────────────────
AnatomicalMoEPool                                ~144,704

axial_pos_embed (1, 64, 1, 768) learned            49,152
─────────────────────────────────────────────────────────
Total added by MoE path                          ~193,856
```

For comparison: CrossAttnPool probe is ~277K params; AttentiveProbe d=1 is ~7.17M.
Volume MoE + learned axial sits below CrossAttnPool in size.

## 7. Memory profile (16 GB GPU, FT batch=1, accum=16, fp16)

```
Encoder forward (per chunk):              ~600 MB
Volume MoE on (B=1, 16384, 768):          ~70 MB activations
Probe (any) on 32 prototype tokens:        <2 MB
Forward total:                            ~700 MB
+ Backward + AdamW states:                ~3 GB total
─────────────────────────────────────────────────────
Comfortable on 16 GB
```

## 8. Implementation files

| Path | Role |
|---|---|
| `src/models/anatomical_moe_pool.py` | `AnatomicalMoEPool` module |
| `src/eval_downstream.py` | `_build_aggregator`, `_build_probe`, `DownstreamModel`, `build_finetune_param_groups` |
| `configs/downstream_patch_anatomical_moe.yaml` | Example fine-tune config |
| `scripts/run_downstream.sh` | AML launcher (exposes `POOL_TYPE`, `MOE_*` env vars) |
| `scripts/test_anatomical_moe_pool.py` | Module-level unit tests |
| `scripts/test_anatomical_moe_integration.py` | Integration tests including regression guards |
| `docs/design/anatomical_moe_integration.md` | Design rationale and audit history |

## 9. Backward compatibility

- Setting `pool_type: mean` (or omitting it) reproduces the existing mean-pool flow exactly.
- `CrossAttnPool` and `AttentiveProbe` constructors gained `use_pos_embed=True` default,
  preserving the original RNG init order so same-seed runs match historical numbers
  bit-for-bit.
- Frozen-probe path (`run_patch_downstream`) raises `NotImplementedError` for
  `pool_type != mean` — fail-fast rather than silently running the baseline. Frozen MoE
  wiring is tracked as a follow-up (see design doc).

## 10. Regression test coverage

| Test | What it guards |
|---|---|
| `test_routing_scaling_keeps_dispatch_soft` | Routing logits are scaled by `1/sqrt(head_dim)`; without scaling, soft pool collapses to near-hard top-k at `head_dim=96` |
| `test_finetune_param_groups_includes_axial_pos_embed` | Learnable axial pos embed is included in optimizer groups (was a P0: param had grad but was never stepped) |
| `test_checkpoint_save_load_round_trip` | `aggregator` + `axial_pos_embed` survive a save/load round-trip with byte-equal values |
| `test_checkpoint_missing_axial_raises` | Loading an old checkpoint into an MoE-configured model raises a clear error rather than silently leaving axial pos embed at random init |
| `test_rng_init_order_preserved_for_default` | `CrossAttnPool` and `AttentiveProbe` default constructors produce the same Linear weights for the same seed as before the `use_pos_embed` flag was added |
| `test_downstream_model_rejects_pos_embed_with_aggregator` | Hard `ValueError` if a probe with `use_pos_embed=True` is paired with an aggregator |
| `test_downstream_model_rejects_unknown_scope` | Hard `ValueError` for typos in `moe_scope` |

Run with:
```
python scripts/test_anatomical_moe_pool.py
python scripts/test_anatomical_moe_integration.py
```
