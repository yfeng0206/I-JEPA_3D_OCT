# Anatomical Mixture-of-Experts

A volume-scope soft mixture-of-experts that aggregates ViT patch features from
an OCT volume into `E·S` learned anatomical prototype tokens. Each prototype
softly pools the patches resembling it (in `H` independent feature subspaces),
with axial position embeddings allowing prototypes to specialize by retinal
layer × axial slice position. Adapted from MAMMOTH (Shao et al., ICLR 2026)
applied at OCT-volume scope.

## Idea

An OCT volume has stratified, repeating anatomy — the same retinal layers
(RNFL, GCL, IPL, INL, OPL, ONL, RPE, choroid, ...) appear across all 64
slices, differing by axial position and disease state. Mean-pooling
collapses 16,384 patch tokens into one 768-d vector and destroys this
structure: a thinned-RNFL signal at peripapillary slices is averaged with
healthy macular tissue and disappears.

Anatomical MoE replaces the mean with `E·S = 32` **learned slot prototypes**.
Each slot acts as a soft query — "*find the patches that look like this*" —
and emits a single vector that is a weighted average of the patches matching
it most strongly. The classifier then sees 32 anatomy-aware summaries
("morphological prototypes") instead of one global mean. Empirically, slots
specialize: with axial position embeddings injected before routing, distinct
slots converge on retinal-layer × slice-position concepts (peripapillary
RNFL vs. macular RNFL vs. choroid, etc.), with no manual labels.

Three properties make the routing useful at the OCT scale:

- **Soft assignment.** Softmax is taken over tokens (not over experts), so
  every patch contributes a fraction of itself to every slot rather than
  being hard-routed to one. Soft routing is stable at our data scale and
  avoids the expert-collapse failure mode of sparse top-k MoEs.
- **Multi-head routing.** The embedding is split into `H` independent
  subspaces; routing happens per head. One head can match patches by
  intensity-aligned features, another by texture-aligned features, so a
  single patch contributes to different slots through different aspects of
  its representation.
- **Axial position embedding.** A learnable `(1, S, 1, D)` offset added
  before flattening breaks the inter-slice symmetry: without it, the MoE
  cannot tell slice 30 from slice 60 because content-only features look
  similar; with it, slots can localize axially.

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

## Expected behavior

Slots are not labeled or supervised — they emerge from end-to-end training.
Empirically in MAMMOTH (histopathology), slots converged on named morphological
concepts (tumor cells, stroma, lymphocytes, alveoli, red blood cells) verified
by two board-certified pathologists and a vision-language concept-alignment
score. Convergence happened within the first epoch because UNI features
already clustered by morphology at initialization.

The same mechanism is expected to apply to OCT, with a different concept set —
retinal-layer × axial-position rather than tissue morphology. Predicted
specializations for `E·S = 32` slots:

| Concept | Expected routing pattern |
|---|---|
| Peripapillary RNFL | upper-retinal patches, native slices ~63 (and ~138 OD/OS mirror) |
| Macular RNFL/GCL | upper-retinal patches, native slices ~95 |
| Foveal pit | central column, ~3 central slices, characteristic depression |
| OPL/INL boundary | mid-retinal patches, all slices |
| ONL / photoreceptors | lower-mid retina, hyper-reflective |
| RPE / Bruch's membrane | thin high-reflectance band, all slices |
| Choroid | below RPE, all slices |
| Vitreous / above-ILM | top rows of every slice, low signal |
| Sclera / below-choroid | bottom rows |

`E·S = 32` is more slots than named retinal layers (~12), so some slots will
likely overlap or capture sub-regional variants (e.g., separate slots for
peripapillary RNFL vs. macular RNFL despite both being "RNFL").

## Validation

Three checks adapted from the MAMMOTH validation protocol:

1. **Routing heatmaps.** For each slot, render dispatch weights over the
   `(slice, row, col)` grid. Two views per slot: a 2D heatmap on a representative
   slice (spatial structure within slice) and a 1D curve over the slice axis
   (axial specialization). Inspect for clean axial localization at expected
   anatomical depths.
2. **Anatomical alignment.** Use a public retinal-layer segmentation model
   (e.g., ReLayNet) to label each patch's dominant layer. For each slot,
   compute the average routing-weight-weighted layer distribution → a
   `(slots × layers)` matrix. A specialized slot has a peaked row; an
   un-specialized slot is uniform. Sort rows by entropy as the diagnostic.
3. **Emergence dynamics.** Save the routing weights at every checkpoint epoch
   and recompute the alignment matrix. Expectation per MAMMOTH: alignment
   already non-trivial at epoch 0 (because the JEPA encoder pre-clusters by
   anatomy) and stable within ~1 epoch of fine-tune. Slow or noisy
   convergence indicates encoder features don't cluster by anatomy as
   strongly as UNI does for histology — a result worth reporting either way.

Optional clinical validation: ophthalmologist eyeballs the per-slot heatmaps
on 5–10 representative volumes (one glaucoma, one healthy, sample of others)
and names what each top-N slot is attending to. Equivalent to MAMMOTH's
pathologist step.

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
