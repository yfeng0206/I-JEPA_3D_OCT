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

At the time of the local audit, no ImageNet/Kaggle/Hugging Face credentials
were present. Data download therefore remains blocked until the account terms
are accepted and credentials are configured.

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
facebook/dinov3-vitl16-pretrain-lvd1689m
revision ea8dc2863c51be0a264bab82070e3e8836b02d51
official size: 300M
artifact parameters: 303,129,600
```

Manual Hugging Face approval and the custom DINOv3 license are required.

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

Use the official DINOv3 notebook procedure as the source for visualization:

- final normalized patch features;
- `PCA(n_components=3, whiten=True)`;
- transform patches to three display channels;
- fit on all patches because the official foreground helper is supervised;
- label this as an all-patch adaptation, not the official foreground-PCA
  reproduction.

Sources:

- [DINOv3 PCA notebook](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/notebooks/pca.ipynb)
- [DINOv3 foreground notebook](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/notebooks/foreground_segmentation.ipynb)

The existing local one-PC signed map is only a diagnostic and must not be
presented as the official DINOv3 visualization.

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
Describe this image in one factual sentence of at most 20 words.

Qwen box:
Return one JSON item with bbox_2d and label for the most prominent named
class instance.

Qwen point:
Return one JSON item with point_2d and label for the most prominent named
class instance.

Molmo:
Point to the most prominent object in the image.
```

Use deterministic/greedy decoding for the locked comparison. Preserve raw
text, parsed output, coordinates, latency, memory, and failures.
Neither VLM receives the ImageNet class label. Molmo's extracted point is
therefore labeled `prominent object`, not retrospectively assigned the
ground-truth class.

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

- no Hugging Face token was configured;
- no Kaggle credentials were configured;
- no local official ImageNet archive was identified;
- DINOv3 manual access had not been approved.

The official public I-JEPA, Qwen3-VL, and MolmoPoint assets are present under
`D:\jepa_phase0`; only ImageNet and DINOv3 access remain blocked.

Do not bypass these controls using public repackaged ImageNet mirrors or
unofficial DINO weights.

If ImageNet access remains unavailable, only user-owned/CC0 smoke images may
be used and no ImageNet accuracy may be reported.

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
