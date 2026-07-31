# Phase-0 ImageNet-50 Evidence Lock

**Evidence cutoff:** 2026-07-16

This document records the primary-source decisions used to implement the
locked Phase-0 plan. Published values and future local measurements are kept
separate.

## 1. Dataset decision

There is no official or field-wide canonical ImageNet-50 benchmark. The Phase-0
subset is:

```text
phase0-cmc-in100-prefix50-v1
```

It is explicitly project-derived and noncanonical: the first 50 ordered WNIDs
from CMC's published ImageNet-100 manifest.

Primary source:

- CMC repository:
  [`HobbitLong/CMC`](https://github.com/HobbitLong/CMC/tree/7b227be0b10ef4e526c72af07664f5079ed9ee09)
- Source file:
  [`imagenet100.txt`](https://github.com/HobbitLong/CMC/blob/7b227be0b10ef4e526c72af07664f5079ed9ee09/imagenet100.txt)
- Source blob SHA-1:
  `6dccf9be4f2f5efe33d0f5ae03a20bc438d3e658`
- Derived WNID manifest SHA-256:
  `206e19b5df0c7118ce22a67972ae43f354bf8dc147a7b8850ddbd499a37b11af`
- Human-readable class labels are resolved by WNID from TensorFlow's
  [`imagenet_class_index.json`](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json)
  (SHA-256
  `a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc`).
  Labels are review metadata and are never supplied to the VLMs.

Expected counts:

| Split | Images |
|---|---:|
| Train | 63,747 |
| Validation | 2,500 |

Every validation class contains 50 images. Five train classes contain fewer
than 1,300 images:

| WNID | Train images |
|---|---:|
| `n03062245` | 1,154 |
| `n04485082` | 1,160 |
| `n04429376` | 976 |
| `n03764736` | 1,097 |
| `n02087046` | 860 |

The subset must always be described as project-derived. Results must not be
reported as a canonical ImageNet-50 benchmark.

## 2. Data access

Permitted acquisition routes:

1. existing terms-compliant official ILSVRC2012 archives;
2. [official ImageNet access](https://image-net.org/download.php);
3. the official Kaggle ImageNet localization challenge after accepting its
   rules;
4. gated
   [`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
   after accepting ImageNet terms.

Do not use public repackaged ImageNet-50/100 mirrors.

Published archive MD5 values used by torchvision:

```text
train   1d675b47d978889d74fa0da5fadfb00e
val     29b22e2961454d5413ddabcf34fc5622
devkit  fa75699e90414af021442c21a62c3abf
```

Source:
[`torchvision/datasets/imagenet.py`](https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py).

The official ImageNet server exposed the ILSVRC2012 archives directly on
2026-07-16. The locked 50 classes were materialized under:

```text
D:\Users\Gary\Desktop\ImageNet_Data\phase0-cmc-in100-prefix50-v1
```

To avoid retaining the 147.9 GB full training archive, the acquisition process
indexed its 1,000 class-tar headers with HTTP byte ranges and downloaded only
the 50 selected class payloads. The complete validation archive and devkit were
downloaded, MD5-verified, used to recover validation WNIDs, and the temporary
validation archive was removed after extraction.

Actual local data:

| Split | Classes | Images | Content-manifest SHA-256 |
|---|---:|---:|---|
| Train | 50 | 63,747 | `5f80288517408e01577279b5a56f7ba35dcff6977db625c48ea0610134f7a6d8` |
| Validation | 50 | 2,500 | `e1192dca08f9fa4a3f996cf6f5125853eb8938121de3bde379af0f2a126e85f7` |

All 66,247 retained files passed PIL decode verification. The exact selected
official class-tar payloads total 7,679,006,720 bytes; the extracted train and
validation JPEGs total 7,998,298,132 bytes (7.45 GiB). Thus a quoted 6.27 GB
size does not describe these exact official files for the locked WNID set,
although the final retained dataset is still only ImageNet-50 rather than the
147.9 GB ImageNet-1K training archive.

Per-file SHA-256 records:

```text
D:\Users\Gary\Desktop\ImageNet_Data\manifests\
  phase0-cmc-in100-prefix50-v1\train.json
  phase0-cmc-in100-prefix50-v1\val.json
```

### Official ImageNet split format

- **Train:** the official archive contains 1,000 nested `n########.tar` class
  archives. The WNID is the label. The Phase-0 output retains only the locked 50
  WNID directories.
- **Validation:** the official archive contains 50,000 flat, ordered JPEGs.
  Labels come from `ILSVRC2012_validation_ground_truth.txt`; `meta.mat` maps
  integer synset IDs to WNIDs. The selected subset contains exactly 50 images
  for each of the 50 classes.
- **Test:** the official classification test archive contains flat images but
  public ground-truth labels are not released. It is not downloaded or used.
  Phase 0 fits probes on `train` and reports once on `val`.

Final loader layout:

```text
phase0-cmc-in100-prefix50-v1\
  train\<WNID>\*.JPEG
  val\<WNID>\*.JPEG
```

## 3. Existing I-JEPA results

Meta did not publish ImageNet-50 or ImageNet-100 kNN/linear results for its
released checkpoints. Meta also did not publish an ImageNet-1K kNN result.

Official full ImageNet-1K frozen-linear results:

| Architecture | Pretraining | Frozen linear top-1 |
|---|---|---:|
| ViT-B/16 | IN1K, 600 epochs | 72.9 |
| ViT-L/16 | IN1K, 600 epochs | 77.5 |
| ViT-H/14 | IN1K, 300 epochs | 79.3 |
| ViT-H/16, 448 px | IN1K, 300 epochs | 81.1 |

Source:
[I-JEPA paper Table 1](https://ar5iv.labs.arxiv.org/html/2301.08243#S4.T1).

The often-quoted 73.3% for H/14 is ImageNet-1% low-shot fine-tuning, not a
full-data frozen result.

Official evaluation uses the EMA `target_encoder`, average-pools its patch
tokens, and trains a linear classifier. The official I-JEPA model has no CLS
token.

Sources:

- [I-JEPA paper and supplement](https://arxiv.org/abs/2301.08243)
- [official repository](https://github.com/facebookresearch/ijepa/tree/52c1ae95d05f743e000e8f10a1f3a79b10cff048)

### Third-party ImageNet-100 context

These results are not directly comparable to the local Phase-0 protocol:

- CNN-JEPA retrained I-JEPA on ImageNet-100 for 200 epochs:
  - ViT-S: 42.30 linear / 34.84 kNN
  - ViT-B: 46.36 linear / 31.36 kNN
  - source:
    [CNN-JEPA Table 1](https://arxiv.org/html/2408.07514v2#S4.T1)
- A 2026 paper reports an IN1K-pretrained H/14-scale I-JEPA frozen attentive
  probe on ImageNet-100:
  - 88.7 top-1 / 98.6 top-5
  - this is not linear or kNN
  - source:
    [SCOTT Table 3](https://arxiv.org/html/2502.18056v2#S4.T3)

No published I-JEPA ImageNet-50 result was found.

### Official downstream tasks and protocols

The official paper evaluates the ImageNet-1K-pretrained ViT-H/14 on frozen
transfer classification and low-level prediction tasks:

| Protocol | Architecture | CIFAR-100 | Places205 | iNat18 |
|---|---|---:|---:|---:|
| Frozen linear transfer | ViT-H/14 | 87.5 | 58.4 | 47.6 |

Source:
[I-JEPA paper Table 3](https://ar5iv.labs.arxiv.org/html/2301.08243#S5.T3).

| Protocol | Architecture | CLEVR/Count | CLEVR/Dist |
|---|---|---:|---:|
| Frozen linear low-level transfer | ViT-H/14 | 86.7 | 72.4 |

Source:
[I-JEPA paper Table 4](https://ar5iv.labs.arxiv.org/html/2301.08243#S6.T4).

The transfer protocol freezes the encoder and uses average-pooled patch
representations because the I-JEPA encoder has no CLS token. It reports the best
linear result from either the final layer or concatenated final four layers,
with either a linear head or batch-normalization plus a linear head, otherwise
following the VISSL recipe. CIFAR-100 images are resized to 224 x 224. CLEVR
counting and distance use center crop plus horizontal flip rather than random
crop, because cropping can remove objects or depth cues.

The official ImageNet-1K frozen-linear protocol is not the local Phase-0
logistic-regression protocol: it average-pools patch tokens, uses LARS with
batch size 16,384 for 50 epochs, decays the learning rate by 10 every 15 epochs,
and sweeps reference learning rates `[0.01, 0.05, 0.001]` and weight decays
`[0.0005, 0.0]`.

For ImageNet-1% low-shot evaluation, the paper adapts the full encoder for 50
epochs with AdamW and cosine decay; this is fine-tuning, not a frozen linear
result. Its ViT-H/14 top-1 is 73.3%. The appendix also reports full ImageNet-1K
fine-tuning for a 448-pixel ViT-H/16 at 87.1% top-1
([Table 15](https://ar5iv.labs.arxiv.org/html/2301.08243#A4.T15)).

The original release does not report semantic segmentation or object detection
for these image checkpoints. It calls CLEVR counting and distance “local” or
“low-level and dense prediction” tasks, but evaluates them with frozen global
linear probes rather than a dense segmentation/detection head.

Primary protocol source:
[I-JEPA Appendix A.2](https://ar5iv.labs.arxiv.org/html/2301.08243#A2).

## 4. I-JEPA local checkpoint

Locked primary checkpoint:

```text
https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
state key: target_encoder
```

Architecture:

| Property | Value |
|---|---:|
| Input | 224 x 224 |
| Patch | 14 |
| Patch tokens | 256 |
| Width | 1,280 |
| Blocks | 32 |
| Heads | 16 |
| Parameters | 630,762,240 |

Preprocessing:

```text
resize shorter side to 256
center crop 224
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Readout:

```text
final target-encoder patch tokens
-> mean over 256 patches
-> FP32 L2 normalization
```

Locally verified artifacts:

```text
raw archive SHA-256
0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22

extracted target_encoder safetensors SHA-256
03fdafbd89f4e20184a83e5ce71a605fe0ff2ce69aee9eb4b6dabc8c8f5cd899
```

The Hugging Face checkpoint is a fallback only because the conversion uses
the context `encoder`, not the raw `target_encoder`, and its processor uses
different normalization.

## 5. Frozen guide checkpoints

The exact lock is stored in
`configs/semantic_maps/manifests/phase0_models.yaml`.

### DINOv3

```text
facebook/dinov3-vit7b16-pretrain-lvd1689m
revision b80367753773648a6793235ab9c65cdbb029506f
official size: 6,716M
artifact parameters: 6,716,035,072
artifact bytes: 26,864,283,632
tensor count: 687
```

Hugging Face access was approved and authenticated on 2026-07-16. The custom
DINOv3 license still applies. ViT-7B is the from-scratch teacher in the official
release; ViT-S/S+/B/L/H+ are distilled from it.

This primary control matches the deployed system scale of the two 8B VLMs:

| Guide | Total parameters | Visual parameters used for frozen features |
|---|---:|---:|
| DINOv3 ViT-7B/16 | 6.716B | 6.716B |
| Qwen3-VL-8B | 8.767B | 576M |
| MolmoPoint-8B | 8.678B | 469M vision plus connector |

The total systems are within 1.31x. The visual encoders are not matched, so
results must not be described as isolating language supervision. ViT-B (86M)
and ViT-L (300M) are excluded from the primary result table to avoid an
obviously smaller DINO control.

### Qwen3-VL

```text
Qwen/Qwen3-VL-8B-Instruct
revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
official size: 8B
artifact parameters: 8,767,123,696
```

Use the full frozen VLM for generated captions and native JSON boxes/points.
Use post-merger visual tokens for frozen feature evaluation.
The artifact contains 576,388,336 visual parameters.

### Molmo

```text
allenai/MolmoPoint-8B
revision 188130f961c8e0888a34e11121a1423c461a01ba
official size: 8B
artifact parameters: 8,677,855,065
```

Use the full frozen VLM for generated captions and native points. MolmoPoint
has no official box API; any disk/Gaussian drawn around a point is analyst
post-processing and must be labeled as such.
The vision backbone plus connector contains 469,115,216 parameters.

## 6. Local frozen protocol

All values produced by this protocol are local Phase-0 measurements.

### Features

| Model | Readout |
|---|---|
| I-JEPA | mean final raw `target_encoder` patch tokens |
| DINOv3 | final normalized CLS token |
| Qwen3-VL | mean final post-merger image tokens |
| MolmoPoint | mean valid post-pooling/post-projector vision tokens |

Use a single deterministic view and each model's official preprocessing.
Aggregate in FP32, L2-normalize, and cache features with model, processor,
dataset-manifest, and image hashes. Cache writes use resumable FP32 memory maps,
contiguous progress commits, and final SHA-256 verification; interrupted VLM
inference resumes at the first uncommitted image.

### kNN

```text
weighted cosine kNN
k = 20
temperature = 0.07
full 63,747-image train feature bank
```

Report top-1, top-5, macro top-1, and per-class accuracy.

### Linear probe

Use cached frozen features and multinomial L2 logistic regression:

- intercept enabled;
- LBFGS;
- FP64 solver;
- `max_iter=1000`;
- `tol=1e-12`;
- select regularization `C` on a deterministic per-class 90/10 train split;
- retrain on all train features;
- evaluate the official validation split once.

This is a shared local protocol, not the published I-JEPA linear recipe.

## 7. Literature-backed map paths

### DINO attention

Use final-layer CLS-to-patch attention as a diagnostic, based on the original
DINO official visualization:

- preserve all heads;
- remove CLS and four DINOv3 register tokens;
- arithmetic head mean is an explicitly adapted summary.

Source:
[`facebookresearch/dino/visualize_attention.py`](https://github.com/facebookresearch/dino/blob/affca658f19b5509920ec065797b6abb88149982/visualize_attention.py).

### Three-component PCA

The former 224-pixel all-patch panel is retained only as a diagnostic. It is
not the paper visualization: it has 196 patches, no foreground mask, and
per-channel min-max display scaling.

The paper-style ViT-7B adaptation follows the released DINOv3 notebook:

- final normalized patch features;
- aspect-preserving 768-pixel input aligned to patch size 16;
- supervised foreground classifier retrained on the nine released image/mask
  pairs for the ViT-7B 4,096-dimensional feature space;
- 3x3 median filtering and foreground threshold 0.5;
- `PCA(n_components=3, whiten=True)`;
- PCA fit on foreground patches only;
- `sigmoid(2*z)` color transfer and black background;
- all 48 sign/component-to-RGB orientations retained, with no automatic
  aesthetic selection.

This is labeled **DINOv3 notebook-style ViT-7B adaptation with supervised
foreground mask**. The foreground helper is supervised even though the rainbow
part colors are PCA-derived.

An unsupervised companion uses DINOv2-style two-stage PCA and retains both PC1
foreground polarities because sign selection is otherwise subjective.

Sources:

- [DINOv3 PCA notebook](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/notebooks/pca.ipynb)
- [DINOv3 foreground notebook](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/notebooks/foreground_segmentation.ipynb)

The DINOv3 paper's page-2 maps use unspecified high-resolution inputs. Figure 13
uses 1280x960 ViT-7B features (80x60 tokens), while Section 6.1.1 inspects all
eight sign choices and six RGB permutations and publishes the visually most
compelling one. Those choices explain why a paper panel should not be compared
directly with an automatic 224-pixel first-three-PC plot.

### TokenCut-style NCut

Use the official MIT-licensed TokenCut implementation at commit:

```text
fed52cd5b60891baefd8ec7110dafa73be816ee1
```

Lock:

- final normalized patch/key features;
- `tau=0.2`;
- absent-edge `eps=1e-5`;
- generalized second eigenvector;
- mean threshold;
- seed-based sign and connected-component selection.

Because DINOv3 replaces the original DINO feature source, call this
`TokenCut-style NCut on DINOv3`, not an official TokenCut reproduction.

## 8. Twenty-image grounding gate

Use 20 preselected ImageNet training/development images, one from each of the
first 20 WNIDs, fixed by manifest before inference.

Prompts:

```text
Caption:
Describe this image.

Matched Qwen point:
Identify and locate the single most visually prominent visible object.
Return exactly one point JSON object with a specific noun label.

Matched Molmo point:
Point to the single most visually prominent object.

Plural coverage:
Each model locates all visually prominent objects with native points.

Qwen-only capability:
Locate visually prominent objects using bounding boxes.
```

Use deterministic/greedy decoding for the locked comparison. Preserve raw
text, parsed output, coordinates, latency, memory, and failures.
Neither VLM receives the ImageNet class label. Molmo's extracted point is
therefore generically labeled, not retrospectively assigned the ground-truth
class. Qwen must return concrete object labels but is free to select a person
instead of the ImageNet class object; that is a measured salience failure, not
repaired with class conditioning.

Qwen receives the full source through dynamic native resolution. Molmo receives
the full source through its global-plus-local-crop processor. Results produced
from the earlier hidden 224 crop are superseded and cannot be used for the
grounding decision.

Pass requirements:

- 20/20 completed inference and nonempty caption;
- at least 19/20 valid in-bounds grounding outputs;
- at least 16/20 captions compatible with the class or clear hypernym;
- at least 16/20 primary-object point hits;
- Qwen box IoU at least 0.5 on at least 12/20;
- aligned transform consistency;
- peak reserved GPU memory at most 22.5 GiB.

At least one VLM must pass for the language arm to continue.

### One-image integration smokes

These are API/memory checks only, not the 20-image grounding gate and not an
ImageNet result.

| Model | Result | Visual tokens | End-to-end latency | Peak reserved |
|---|---|---:|---:|---:|
| Qwen3-VL | Caption plus valid dog box and point | 16 x 16 x 4096 | 8.26 s | 16.66 GiB |
| MolmoPoint | Caption plus one native point, no fabricated box | 392 x 4096 pooled nonspatial tokens | 13.30 s | 17.64 GiB |

Qwen used the official Qwen-VL
[`demo.jpeg`](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg)
(SHA-256
`9eeaa87013b4e800930e8a411b58ff9e2fd5383906b1a022f4a712720af34cc2`).
The corrected fixed 512 x 512 input produces the locked 16 x 16 post-merger
grid. Molmo used a local marina API-smoke image (SHA-256
`e6fae09d2b5f76706f3193f929819e7f60a5d231b95528c8a4ee826b6be1c7b0`);
its point was produced from the fixed image-only prompt. Frozen Molmo feature
extraction calls the pinned model's official ViT and connector modules directly
and does not run the language decoder for every cached image.

Local records:

```text
D:\jepa_phase0\results\manifests\qwen_smoke.json
D:\jepa_phase0\results\manifests\molmo_smoke.json
D:\jepa_phase0\results\atlases\ijepa_qwen_smoke
D:\jepa_phase0\results\atlases\molmo_smoke_visiononly
```

Both models fit the 22.5 GiB safety gate on these examples. No reliability,
stability, or localization-rate conclusion is permitted until the manifest-
locked 20-image test runs.

## 9. Current access blockers

At the time of this audit:

- DINOv3 manual access had not been approved.

The official public I-JEPA, Qwen3-VL, and MolmoPoint assets are present under
`D:\jepa_phase0`, and the official ImageNet-50 materialization is present under
`D:\Users\Gary\Desktop\ImageNet_Data`. Only DINOv3 access remains blocked.

Do not bypass these controls using public repackaged ImageNet mirrors or
unofficial DINO weights.

No ImageNet accuracy may be reported until frozen feature extraction and the
locked probes have actually run on this materialization.

## 10. Allowed claims

- local Phase-0 frozen results use a project-derived, noncanonical 50-class
  prefix of a published ImageNet-100 manifest;
- published full ImageNet-1K values are context, not directly comparable;
- VLM captions and grounding use frozen external supervision;
- TokenCut on DINOv3 is an adaptation;
- PCA and attention maps are diagnostics, not calibrated explanations.

Do not claim:

- canonical ImageNet-50 performance;
- official I-JEPA ImageNet-50/100 kNN or linear results;
- language supervision isolated from architecture/data differences;
- model-native DINOv3 saliency from the current CLS-cosine heuristic;
- segmentation quality from PCA/attention alone.
