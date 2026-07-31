# Semantic-Teacher-Guided Target Allocation for I-JEPA

Branch: `vlm-guided-masking`

> This is the active research plan. The immediate goal is a small preliminary
> comparison, not a complete conference experiment matrix. Phase 0 map tooling
> exists; semantic-guided I-JEPA training does not.

## 0. Current decision

Phase 0 is a local, frozen-model ImageNet-50 study. It has three deliverables:

1. audit existing published and Hugging Face/GitHub I-JEPA ImageNet results and
   checkpoints, including ImageNet-50/100 results if they exist;
2. run frozen kNN and linear-probe evaluation on a reproducible ImageNet-50
   subset when no directly comparable published result exists;
3. generate a human-reviewable semantic-map atlas for I-JEPA, DINOv3,
   Qwen3-VL, and Molmo.

The encoders remain frozen. kNN trains nothing, and a linear probe trains only
a small classifier head. There is no encoder fine-tuning or masked pretraining
in Phase 0.

Phase 1, only after positive Phase-0 evidence, will compare:

```text
random I-JEPA targets
        vs
DINOv3-guided targets
        vs
the best grounded VLM-guided target policy
```

### Scope lock

- ImageNet-50 is the primary quick test domain.
- ImageNet-100 is a fallback/extension when a directly comparable published
  result or already-prepared subset makes it inexpensive.
- DINOv3 is the primary dense vision guide.
- Qwen3-VL and Molmo are the two VLMs evaluated in Phase 0.
- Use official paper/model-card checkpoints and published parameter counts.
- The primary system-size comparison is DINOv3 ViT-7B/16 (6.716B) against
  Qwen3-VL-8B (8.767B) and MolmoPoint-8B (8.678B). Do not add smaller DINO
  variants to the primary Phase-0 table.
- Report both total and visual-stack parameter counts. Total-size matching does
  not isolate language supervision because the VLM visual stacks contain only
  576M (Qwen) and 469M (Molmo vision plus connector) parameters.
- The dataset supplies images only. Any caption or fixed generic instruction is
  internal training-time supervision.
- Every VLM atlas panel includes the generated text beside the shaded regions.
- The guide changes only where ordinary I-JEPA target blocks are sampled.
- Guide features and text are not given to the I-JEPA predictor.
- Target block count, shapes, unique coverage, overlap, retained context, query
  count, and loss normalization must be matched.
- External guides remain frozen for the first experiment.

### Explicitly deferred

The preliminary experiment does not include:

- SigLIP or SigLIP 2;
- CLIP as a primary guide;
- SAM or an external segmentation refinement stage;
- MedGemma;
- both VLMs in guided pretraining (only the Phase-0 winner may advance);
- guide fine-tuning or LoRA;
- feature alignment or distillation;
- predictor-capacity sweeps;
- semantic-region dilation sweeps;
- from-scratch confirmation;
- OCT transfer;
- 3-D or axial modeling.

MILAN remains required prior art and a design warning. It is not a primary
modern guide arm.

## 0.1 Status tracker

Legend: DONE / IN PROGRESS / TODO / DEFERRED

| ID | Work item | State | Gate |
|---|---|---|---|
| P0-A | Common frozen-guide tensor API | DONE | Dense tokens map to a validated spatial grid |
| P0-B | Audit online I-JEPA checkpoints/results | IN PROGRESS | Official assets and published ImageNet-1K/50/100 results recorded |
| P0-C | Choose reproducible ImageNet-50 definition | IN PROGRESS | Published class manifest and legitimate data source pinned |
| P0-D | DINOv3-7B adapter/checkpoint | IN PROGRESS | Approved checkpoint loads in BF16 and produces stable maps |
| P0-E | Qwen3-VL and Molmo grounding smoke | TODO | Captions and native grounding work on 20 images |
| P0-F | Frozen ImageNet-50 kNN/linear evaluation | TODO | Comparable fixed protocol for all available encoders |
| P0-G | Generate a 100-200 image atlas | TODO | DINO and VLM maps are spatially meaningful and stable |
| P1-A | Implement exact-budget score-to-target sampler | TODO | Guided masks match random realized statistics |
| P1-B | Short random/DINO/VLM continuation | TODO | One guide beats random on development evaluation |
| P2 | Replication and larger evaluation | DEFERRED | Preliminary result is positive |
| P3 | OCT guide comparison and transfer | DEFERRED | ImageNet guide policy survives |

## 0.2 What the current code does

Implemented:

- `src/guides/base.py` defines a common dense-token output contract.
- `src/guides/hf_guides.py` exposes DINOv3, SigLIP 2, and CLIP vision maps.
- `src/guides/ijepa.py` exposes a local I-JEPA teacher-token baseline.
- `src/guides/maps.py` computes native maps, global-to-patch cosine, token PCA,
  and basic map diagnostics.
- `scripts/semantic_map_atlas.py` creates deterministic center-crop maps,
  metrics, NPZ files, and visual atlases.
- `configs/semantic_maps/phase0_guides.yaml` records current model identifiers.
- `requirements-phase0.txt` defines the isolated modern sidecar environment.

Not implemented:

- caption generation;
- entity/relation parsing;
- phrase or object grounding;
- Qwen3-VL/Molmo adapters;
- ImageNet-50 data preparation and frozen evaluation;
- map-to-I-JEPA target sampling;
- `train_patch.py` guide integration;
- guided-pretraining configs;
- fixed-budget mask matching;
- any DINO/VLM training result.

## 1. How encoder features map back to the image

For a ViT with input size 224 and patch size 16:

```text
224 x 224 image
      |
      v
14 x 14 = 196 spatial patches
      |
      v
196 patch tokens, each with d features
```

Patch tokens preserve raster order:

```text
token 0   -> row 0, col 0
token 1   -> row 0, col 1
...
token 195 -> row 13, col 13
```

Therefore dense tokens already map to image regions:

\[
F \in \mathbb{R}^{196 \times d}
\longrightarrow
F_{\mathrm{grid}} \in \mathbb{R}^{14 \times 14 \times d}.
\]

PCA is not required for this spatial mapping. PCA is one way to reduce each
patch's \(d\)-dimensional feature into a scalar visualization.

## 1.1 DINOv3 score maps

The primary DINOv3 readout is selected during the small atlas stage.

Candidate readouts:

1. model-native CLS/global-to-patch relevance;
2. global-to-patch cosine similarity;
3. first token-PCA component as a diagnostic;
4. optional feature clustering only if the first two fail.

Global-to-patch cosine:

\[
s_i =
\cos\left(
\operatorname{LN}(f_i),
\operatorname{LN}(f_{\mathrm{global}})
\right).
\]

Token PCA:

```text
patch tokens
    -> subtract per-image patch mean
    -> compute first principal direction
    -> project each patch token
    -> reshape scalar values to 14 x 14
```

PCA measures dominant variation. It does not prove semantic importance. It may
highlight foreground/background, texture, color, illumination, or boundaries.
For this reason PCA remains a visual diagnostic rather than the primary guide.

The publication-style PCA panel is separate from the 224-pixel diagnostic. It
uses ViT-7B final normalized patch features at 768-pixel, aspect-preserving
resolution. The released DINOv3 notebook protocol is reproduced as an explicit
adaptation: a foreground classifier trained on the nine released masks,
3x3 median filtering, foreground-only whitened PCA, `sigmoid(2*z)` colors, and
black background. All 48 PCA sign/RGB permutations are retained rather than
silently selecting the most attractive one. A DINOv2-style unsupervised
two-stage PCA is reported separately with both unresolved PC1 polarities.

## 1.2 Grounded-VLM score maps

The desired VLM path is:

```text
image only
    |
    v
automatic image description
    |
    v
entities and relations
    |
    v
native boxes, points, or phrase-to-patch scores
    |
    v
one fused semantic score map
```

Example:

```text
caption: "A dog is drinking water from a bowl."

entities:
  dog
  water
  bowl

relation:
  dog drinking water
```

Native grounded points are the matched Qwen/Molmo endpoint. A point-only
Gaussian map may be derived on a declared common grid for candidate-target
illustration, but it is named `derived_grounding_raster`, not a native
confidence or feature map. Qwen boxes are displayed as a separate capability
and are not merged into the matched point comparison.

Raw decoder attention is not accepted as grounding without validation. The
selected VLM must expose native boxes, pointing, or another demonstrably
faithful spatial readout.

## 1.3 Qwen3-VL and Molmo

Phase 0 evaluates the standard/recommended official checkpoints from the
Qwen3-VL and Molmo papers/model cards. Record, rather than recalculate:

- official checkpoint name and revision;
- published vision-tower and total parameter counts;
- input resolution and visual-token count;
- precision/quantization recommended for inference;
- license and access requirements;
- grounding/pointing/box interface;
- peak local inference memory and elapsed time.

The full VLM is needed to produce generated text and native grounding. It
remains frozen and is used only for batched inference. No VLM weights are
trained in Phase 0.

Both VLMs receive the complete source image. Qwen uses its official
aspect-preserving dynamic-resolution processor. Molmo uses its official global
view plus up to 24 local 378x378 crops. The 224 center crop is reserved for the
I-JEPA/DINO frozen-classification protocol and is not applied before either
VLM processor.

Primary grounding is a class-label-free, matched single-point task. A separate
plural-point run measures multi-object coverage. Qwen multi-box grounding is
run separately because MolmoPoint has no native box decoder. Raw points/boxes
remain the primary qualitative output; all coordinate rasters are declared
analyst post-processing.

Run a 20-image smoke test on images containing:

- one large object;
- multiple interacting objects;
- small objects;
- clutter;
- occlusion;
- non-central objects.

Each model must:

- generate a relevant per-image description;
- identify more than one entity when appropriate;
- return usable spatial grounding;
- avoid gross hallucinations;
- retain reasonable grounding under resize/flip;
- fit available inference memory and runtime.

Both models may appear in the atlas and frozen encoder table. Only the better
grounded guide may advance to Phase 1. If neither passes, stop the VLM arm and
continue only with DINOv3.

## 2. Phase 0: data, published baselines, and frozen evaluation

### 2.1 Local storage

All datasets, downloaded checkpoints, feature caches, and generated atlases
live on `D:`. The repository remains on `C:`.

```text
D:\jepa_phase0\
  datasets\
    imagenet50\
  models\
    ijepa\
    dinov3\
    qwen3_vl\
    molmo\
  cache\
    frozen_features\
    maps\
  results\
    atlases\
    probes\
    manifests\
```

`D:` currently has sufficient capacity for the subset and model caches. These
paths remain gitignored and are never uploaded to Azure.

### 2.2 ImageNet-50 subset

Use one published ImageNet-50 class manifest if a recognized definition is
found. Record:

- paper/repository source;
- WordNet IDs and class-index mapping;
- train/validation counts;
- source archive/dataset revision;
- license/access requirements;
- per-file checksums or a deterministic manifest hash.

If no recognized ImageNet-50 definition exists, create a deterministic
ImageNet-50 development subset by taking the first 50 classes from the selected
published ImageNet-100 manifest. Name it as this project's derived subset and
do not present it as a canonical benchmark.

Images remain on disk. DataLoader workers decode and transfer mini-batches;
the dataset is never loaded into GPU memory as a whole.

### 2.3 Online I-JEPA checkpoint and result audit

Before downloading or evaluating models, search:

- the official Meta I-JEPA repository and checkpoint links;
- official Hugging Face I-JEPA model cards;
- the I-JEPA paper and supplement;
- later papers that evaluate frozen I-JEPA on ImageNet-50/100.

Record:

- model architecture, patch size, input size, and official parameter count;
- pretraining dataset and epoch count;
- checkpoint URL, revision, hash, and license;
- published full ImageNet-1K kNN/linear/fine-tune results;
- published ImageNet-50/100 results, if any;
- downstream classification/dense tasks and evaluation protocol.

If no published ImageNet-50/100 result exists, evaluate the frozen official
checkpoint locally. Do not retrain I-JEPA.

Published ImageNet-1K values and local ImageNet-50 values remain separate
tables because they are not directly comparable.

### 2.4 Frozen ImageNet-50 evaluation

Evaluate:

1. official I-JEPA checkpoint;
2. selected DINOv3 checkpoint;
3. Qwen3-VL vision tower;
4. Molmo vision tower.

For every encoder:

- load images in mini-batches;
- use the model's official preprocessing;
- extract a frozen global/pooled image representation;
- cache features to `D:\jepa_phase0\cache\frozen_features`;
- run kNN classification;
- train a linear classifier only;
- report fixed train/validation protocol, top-1/top-5 accuracy, model source,
  parameter count, throughput, and GPU-hours.

Terminology:

- kNN: no learned model parameters;
- linear probe: encoder frozen, only a linear head is trained;
- fine-tuning: encoder weights updated, which Phase 0 does not do.

Use the official paper/model-card parameter count and recommended inference
precision. Determine mini-batch size from the official architecture and memory
requirements, with a safety reserve; confirm once with a smoke batch. Do not
perform a batch-size benchmark sweep.

### 2.5 Small map sanity test

#### Data

- 100-200 ImageNet-50 training images.
- 30-50 manually reviewed images.
- Use existing boxes or ImageNet-S masks when readily available.
- Do not use locked final evaluation data for guide selection.

#### Atlas output

For each image:

1. original image;
2. DINOv3 primary map;
3. DINOv3 token-PCA diagnostic;
4. Qwen3-VL generated text and grounded regions;
5. Molmo generated text and grounded regions;
6. fused VLM score maps;
7. DINO-VLM difference maps;
8. sample shaded regions and guided target rectangles.

Generated text must be rendered beside its shaded region overlay in the saved
atlas, not only written to JSON.

#### Minimal diagnostics

- foreground pointing accuracy or overlap;
- multi-object coverage;
- map center and border bias;
- map entropy/effective support;
- geometric stability after alignment;
- caption validity;
- grounding validity;
- hallucination and abstention rate;
- guide inference time and peak memory.

This stage rejects obviously broken maps. It is not a full benchmark.

#### Phase-0 decisions

| Observation | Decision |
|---|---|
| Both VLMs ground poorly despite good captions | Drop VLM |
| VLM and DINO maps are effectively identical | Keep DINO only |
| DINO map is better and more stable | Keep DINO only |
| One VLM adds reliable multi-entity/relation localization | Advance that VLM |
| Both maps are center/border shortcuts | Stop semantic-guide experiment |

## 3. Turning a score map into I-JEPA targets

The guide produces a non-negative score map \(S\).

For candidate rectangle \(R\):

\[
q(R)=\frac{1}{|R|}\sum_{i\in R}S_i.
\]

The guided policy raises the probability of target configurations covering
high-scoring patches.

### Fixed-budget contract

First sample a canonical random I-JEPA target configuration:

\[
\mathbf{R}^{\mathrm{ref}}
=
(R_1^{\mathrm{ref}},R_2^{\mathrm{ref}},
R_3^{\mathrm{ref}},R_4^{\mathrm{ref}}).
\]

The guided configuration must match the reference per image:

- ordered block-shape vector;
- unique target-token union;
- pairwise target-overlap signature;
- retained context-token count;
- prediction query count;
- loss normalization.

Only target locations and guide scores may differ.

If no matched guided configuration exists, fall back to the paired random
configuration and log the fallback. Never silently relax the invariants.

### Preliminary target policy

The first experiment uses semantic regions as prediction targets because this
directly tests the existing retinal-ribbon hypothesis.

Important-as-visible, MILAN-style sampling is deferred. It becomes the first
fallback if semantic-as-target guidance fails.

## 4. Phase 1: short preliminary training

### Arms

1. random matched-budget targets;
2. DINOv3-guided targets;
3. grounded-VLM-guided targets.

### Protocol

- Start every arm from the same complete random I-JEPA ep25 checkpoint.
- Continue ep25 to ep50.
- Use one exploratory seed.
- Use identical data order, augmentation, optimizer, scheduler, target shapes,
  and realized target/context budget.
- Keep guides frozen and in inference mode.
- The guide receives the exact augmented crop with guide-specific normalization.
- Use development evaluation only.

### Evaluation

- frozen kNN or linear probe;
- fixed-checkpoint downstream accuracy;
- representation variance/effective rank;
- predictor-target cosine;
- train/validation loss;
- target coverage and overlap telemetry;
- map entropy and fallback rate;
- update-matched and GPU-hour-matched cost.

### Preliminary decision table

| Result | Decision |
|---|---|
| VLM > DINO > random | Continue language-grounded research |
| DINO approximately equals VLM and both beat random | Drop language; use DINO |
| DINO > VLM | Drop language |
| VLM > random but not DINO | Drop language |
| Neither guide beats random | Stop frozen semantic guidance |
| Training or representation collapse | Stop affected arm |

The preliminary result is a screen, not a publication claim. Independent
prefixes, multiple seeds, longer training, and locked evaluation occur only
after a positive screen.

## 5. Prior art boundary

The broad idea "use a language-trained model to guide masking" is occupied.

### MILAN

MILAN:

- uses frozen CLIP image features as reconstruction targets;
- uses final CLIP CLS attention to sample important patches;
- keeps high-attention patches visible;
- reconstructs CLIP visual-token features with an MAE-style prompting decoder;
- does not generate per-image captions;
- does not ground caption entities or relations;
- is not I-JEPA.

MILAN's semantic sampling contributed only a modest fine-tuning gain after its
CLIP target and prompting decoder. Its linear-probe results warn that focusing
too strongly on important visible patches can specialize the encoder.

### Other required references

- DSeq-JEPA: endogenous attention-based regions and sequential JEPA prediction.
- Mask What Matters (arXiv:2509.23054): the closest broad pipeline. It uses an
  LLM to generate task/category prompts, BiomedCLIP plus M2IB saliency, K-means
  binarization, connected components, SAM refinement, expanded ROI boxes, and
  differentiated ROI/background masking in SparK/MAE-style medical pretraining.
  It already occupies VLM+SAM-guided semantic masking for medical SSL.
- OA-VCD / Mask What Matters (arXiv:2602.11737): partially overlapping only at
  the external saliency-selector level. It thresholds DINO CLS attention to
  remove salient pixels from an auxiliary image for inference-time MLLM logit
  contrast; it has no SAM, MIM/JEPA training, entity grounding, or target-budget
  matching.
- AttMask: teacher-attention masking.
- SemMAE: semantic parts and part-level masking.
- TC-JEPA: caption-conditioned JEPA and patch-word relations.

### Narrow future claim

Only after evidence:

> A frozen semantic teacher can improve geometry- and budget-controlled I-JEPA
> target allocation; a grounded modern VLM adds value beyond DINOv3.

Do not claim:

- first VLM-guided masking;
- first VLM+SAM semantic masking for medical self-supervised learning;
- first text-guided high-ROI/low-background masking;
- first prompt-free CLIP masking;
- first caption-conditioned JEPA;
- first patch-text grounding in JEPA;
- language supervision as the isolated causal factor;
- efficiency without guide inference cost.

## 6. Deferred work after a positive preliminary result

Only after one guide beats random:

1. repeat with independent prefixes/seeds;
2. extend passing arms to ep75/100;
3. run a faithful DSeq-JEPA baseline;
4. compare semantic-as-target versus semantic-as-visible;
5. test from-scratch training;
6. evaluate ImageNet-S/ADE20K or another dense task;
7. transfer the winning policy to OCT;
8. compare MIRAGE, RETFound, and the best justified language-aligned OCT guide;
9. consider selector distillation;
10. consider anchored low-LR guide adaptation.

## 7. OCT transfer, deferred

If the ImageNet experiment survives:

Primary OCT guides:

1. MIRAGE-Base;
2. RETFound-OCT;
3. DINOv3 generic control.

Language candidates such as LO-VLM or RetinaVLM enter only if their phrase
maps pass an OCT grounding gate and add value beyond MIRAGE.

No OCT guide advances without:

- weight/license verification;
- training-data overlap audit;
- external anatomical map evaluation;
- cross-device robustness analysis.

## 8. Hard stop rules

1. Drop the VLM arm if no modern VLM grounds entities reliably.
2. Drop language if DINOv3 ties or wins.
3. Stop semantic guidance if neither guide beats random.
4. Stop an arm if coverage/context matching fails.
5. Stop an arm if representation health degrades materially.
6. Do not extend losing arms beyond ep50.
7. Do not run multiple seeds before a positive preliminary result.
8. Do not fine-tune guides before a frozen guide wins.
9. Do not begin OCT transfer before ImageNet evidence.
10. Do not claim efficiency if guide overhead removes the gain.

## 9. Immediate next actions

No training code changes are included yet.

Next:

1. finish the online I-JEPA checkpoint/result audit;
2. pin a published or derived ImageNet-50 class manifest and data source;
3. record official Qwen3-VL, Molmo, DINOv3, and I-JEPA checkpoints, parameter
   counts, revisions, licenses, and recommended inference settings;
4. download ImageNet-50 and checkpoints to `D:\jepa_phase0`;
5. add Qwen3-VL and Molmo adapters using official inference/grounding code;
6. run the 20-image grounding smoke;
7. run frozen ImageNet-50 kNN/linear evaluation;
8. generate the 100-200 image atlas with generated text and shaded regions;
9. select and lock one readout per guide;
10. implement the fixed-budget score-to-target sampler only after Phase-0
    review;
11. run the three-arm ep25-to-50 preliminary comparison in Phase 1.
