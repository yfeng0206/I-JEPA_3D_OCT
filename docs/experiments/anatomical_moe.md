# Anatomical Mixture-of-Experts

A volume-scope soft mixture-of-experts that aggregates ViT patch features from
an OCT volume into `E·S` learned anatomical prototype tokens. Each prototype
softly pools the patches resembling it (in `H` independent feature subspaces),
with axial position embeddings allowing prototypes to specialize by retinal
layer × axial slice position. Adapted from MAMMOTH (Shao et al., ICLR 2026)
applied at OCT-volume scope.

## Architecture

| Stage | Shape |
|---|---|
| ViT-B/16 per slice | `(S, P, D) = (64, 256, 768)` |
| `+ axial_pos_embed[s]` | `(B, S, P, D)` |
| Flatten | `(B, S·P, D) = (B, 16384, 768)` |
| AnatomicalMoEPool | `(B, E·S, D) = (B, 32, 768)` |
| Probe | `(B, D)` |
| Linear head | `(B, 1)` |

Defaults: `E=8` experts × `S=4` slots = 32 prototypes; `H=8` heads;
`head_dim = D/H = 96` (with `skip_wq=True`); `lora_rank = 16`.

## Routing

Patch tokens `x ∈ ℝ^{B×N×D}` (with `N = S·P`) are routed to slot prototypes
`Φ ∈ ℝ^{E×H×S×d_h}`, where `d_h = D/H`. Splitting `x` into `H` head subspaces
`x_h ∈ ℝ^{B×N×H×d_h}` and normalizing both `x_h` and `Φ`:

$$
\ell_{n,e,h,s} = \frac{1}{\sqrt{d_h}} \langle x_{n,h},\, \Phi_{e,h,s} \rangle
$$

Soft assignment by softmax **over tokens** (not experts):

$$
a_{n,e,h,s} = \frac{\exp(\ell_{n,e,h,s})}{\sum_{n'}\exp(\ell_{n',e,h,s})}
$$

Each slot is a soft summary of the patches that match its prototype:

$$
u_{e,h,s} = \sum_n a_{n,e,h,s}\, x_{n,h}
$$

The `1/√d_h` scaling is necessary for soft pooling: without it, with `d_h = 96`
softmax over `N = 16384` collapses to near-hard top-2 selection (top-1
weight ≈ 0.74; ≈ 2.8 effective tokens routed). With scaling: top-1 ≈ 0.002,
≈ 10 K effective tokens.

## Factorized expert

Per-slot summaries `u ∈ ℝ^{B×E×H×S×d_h}` are projected to output dim through a
factorized two-stage MLP. The first stage `Φ̃ ∈ ℝ^{H×d_h×r}` is shared across
experts within a head; the second stage `W_e ∈ ℝ^{E×H×r×d_o}` is per-expert
(`r = 16`, `d_o = D/H = 96`):

$$
z_{e,h,s} = W_{e,h}\, \mathrm{ReLU}(\tilde\Phi_h\, u_{e,h,s} + b_{1,h}) + b_{2,e,h}
$$

Heads are concatenated to recover `D` per `(E, S)` prototype, followed by
`LayerNorm`. Output: `(B, E·S, D)`.

## Axial position embedding

Axial slice index is added as a `(1, S, 1, D)` tensor before flattening:

$$
\tilde x_{s,p} = x_{s,p} + p_s
$$

`learned`: trainable `nn.Parameter`, registered in the optimizer and
checkpoint. `sincos`: 1D sinusoidal buffer. `none`: no axial encoding.

Axial pos is essential for OCT volume scope. Without it, the MoE cannot
distinguish slice 30 from slice 60 because per-slice features are similar in
content but anatomically distinct (e.g., peripapillary RNFL vs. macular RNFL).

## Configuration

```yaml
model:
  freeze_encoder: false
  probe_type: cross_attn_pool
  head_type: linear
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
training:
  layer_decay: 0.5
  lr_probe: 2.0e-4
  lr_encoder: 2.0e-4
  weight_decay: 0.05
  epochs: 50
  warmup_epochs: 5
  accum_steps: 16
data:
  batch_size: 1
  num_slices: 64
```

| Knob | Default | Effect |
|---|---|---|
| `moe_scope` | `volume` | one MoE call per volume; `per_slice` available as ablation |
| `skip_wq` | `false` | when `true`, route in encoder feature space (`slot_dim = D`) |
| `axial_pos_embed` | `none` \| `learned` \| `sincos` | per-slice position injection |
| `num_experts (E)` | 8 | number of prototype groups |
| `num_slots (S)` | 4 | slots per expert |
| `num_heads (H)` | 8 | independent routing subspaces |
| `lora_rank` | 16 | bottleneck for factorized expert |
| `share_phi` | `true` | share `Φ̃` across experts within a head |

## Parameters

Volume scope, `E=8, S=4, H=8, lora=16, skip_wq=true, D=768`:

| Component | Shape | Count |
|---|---|---|
| Slot prototypes | `(E, H, S, d_h)` | 24,576 |
| `Φ̃` (shared) | `(H, d_h, r)` | 12,288 |
| `W_e` (per-expert) | `(E, H, r, d_o)` | 98,304 |
| Biases + LayerNorms |  | 9,536 |
| **AnatomicalMoEPool** |  | **144,704** |
| Axial pos embed (learned) | `(1, S, 1, D)` | 49,152 |
| **Total** |  | **193,856** |

CrossAttnPool: 277 K. AttentiveProbe d=1: 7.17 M. AnatomicalMoEPool sits
below both at comparable expressivity.

## Memory (FT batch=1, accum=16, fp16)

| Component | Memory |
|---|---|
| ViT-B/16 forward | ~600 MB |
| AnatomicalMoEPool activations | ~70 MB |
| Probe on 32 tokens | <2 MB |
| Backward + AdamW state | ~2 GB |
| **Total** | **~3 GB** |

## Slice-count constraint

The aggregator is shape-agnostic over `N`. The constraint comes from
`axial_pos_embed`: `learned` and `sincos` are fixed-shape tensors of size
`(1, S_train, 1, D)`. Inference at a different `S` requires (a) resampling
the input to `S_train` slices, or (b) interpolating the axial embedding at
load time. Same constraint as ViT pos_embed at variable image size.

## Files

| Path | Role |
|---|---|
| `src/models/anatomical_moe_pool.py` | Module |
| `src/eval_downstream.py` | Factory, `DownstreamModel`, optimizer, checkpoint |
| `configs/downstream_patch_anatomical_moe.yaml` | Example FT config |
| `scripts/run_downstream.sh` | Launcher (`POOL_TYPE`, `MOE_*` env vars) |
| `scripts/test_anatomical_moe_pool.py` | Unit tests |
| `scripts/test_anatomical_moe_integration.py` | Integration + regression tests |
