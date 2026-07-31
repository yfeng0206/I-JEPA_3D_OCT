# Qwen3-VL to SAM3 Ten-Image Screen

**Experiment:** `qwen3-sam3-imagenet10-v1`  
**Date:** 2026-07-16  
**Scope:** first 10 deterministic entries of the locked ImageNet-50 grounding
manifest.

## Method

1. Qwen3-VL-8B selects one to three foreground concepts, assigns an
   uncalibrated semantic-importance value, and produces normalized boxes.
2. Qwen exits and releases GPU memory.
3. SAM3 base receives each noun phrase separately.
4. Among SAM3 queries scoring above 0.3, the instance with maximum overlap with
   the corresponding Qwen box is selected.
5. The selected continuous SAM3 probability map is average-pooled into the
   I-JEPA 16x16 patch grid and multiplied by Qwen importance.
6. Entity maps are fused with a pixelwise maximum. The I-JEPA map itself is
   never thresholded; the 0.5 binary SAM3 mask is retained only as a separate
   diagnostic.

## Models

| Role | Model | Revision |
|---|---|---|
| Concept selection | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Segmentation | `facebook/sam3` | `3c879f39826c281e95690f02c7821c4de09afae7` |

SAM3 ran through isolated Transformers 5.14.1 because the official package
imports Triton, which is unavailable on native Windows.

## Results

| Measure | Result |
|---|---:|
| Images | 10 |
| Strict Qwen JSON with unique labels | 8/10 |
| Valid after recorded duplicate-label union | 10/10 |
| Qwen concepts | 21 |
| Successful SAM3 concepts | 18/21 (85.7%) |
| SAM3 failures | 3/21 |
| Mean selected-mask/Qwen-box IoU | 0.807 |
| Selected masks with IoU >= 0.5 | 15/18 |
| Qwen peak reserved GPU memory | 16.81 GiB |
| SAM3 peak reserved GPU memory | 3.58 GiB |

Qwen duplicate labels occurred on the three-person hat collage and the
three-heart image. They were repaired without rerunning the model by unioning
same-label boxes and retaining maximum importance; all repairs are recorded.

SAM3 failures:

- `mushroom cluster`: degenerate 99.85% full-image mask;
- `person using vacuum`: no query above native score 0.3;
- `vacuum cleaner`: no query above native score 0.3.

## Qualitative findings

Strong examples:

- separate baby/adult monkey maps;
- clean Doberman, snake, puppies, and heart maps;
- distinct bartender, martini-glass, and stacked-glass maps;
- fine spider and flower localization.

Failure modes:

- Qwen sometimes violates the intended foreground-only contract by selecting
  `wooden branch`, `exercise equipment`, or a nearly full-image `spider web`;
- Qwen selected object parts (`mushroom cap`, `mushroom stem`) despite the
  prompt;
- relation-bearing phrases such as `person using vacuum` can be less compatible
  with SAM3's text-concept vocabulary than simple nouns;
- a high-importance broad concept can dominate max fusion.

## Verdict

**Promising, but the predefined screen does not fully pass.**

The core data product is meaningful: Qwen supplies semantic concepts and
importance while SAM3 supplies object-shaped continuous maps. Before scaling,
the concept-normalization stage must:

1. merge repeated instances explicitly;
2. reduce relation phrases to segmentable head nouns while retaining relations
   in metadata;
3. reject or downweight background/support concepts;
4. detect degenerate SAM3 masks;
5. preserve separate entity maps so broad concepts cannot silently dominate.

## Closest prior art and claim boundary

[Mask What Matters (arXiv:2509.23054)](https://arxiv.org/abs/2509.23054)
already performs text-guided semantic masking for medical self-supervised
pretraining. Its pipeline is LLM-generated task/category prompt -> BiomedCLIP
and M2IB saliency -> K-means binary mask -> connected components -> SAM ->
expanded ROI box -> differentiated ROI/background masking in SparK/MAE-style
pixel reconstruction.

The Qwen3-SAM3 screen is therefore **strongly overlapping in broad concept**,
not a first VLM+SAM masking method. The remaining narrower differences are:

- per-image self-generated concepts rather than a user/task prompt;
- multiple interaction entities and importance ordering;
- continuous SAM3 probability maps rather than binary K-means/SAM ROI boxes;
- future latent I-JEPA target allocation rather than pixel-reconstruction MIM;
- exact realized target/context budget matching.

Any future claim must be limited to those differences and must demonstrate that
they add value beyond a faithful Mask What Matters baseline.

## Local artifacts

```text
D:\jepa_phase0\experiments\qwen3-sam3-imagenet10-v1
C:\Users\Gary\Desktop\jepa\results\phase0_local_examples\
  qwen3-sam3-imagenet10-v1
```

Image-containing review artifacts remain local and gitignored.
