# Citations & Related Work

> Every entry states what we use it for in THIS repository and whether the
> dependency is direct (code/weights) or conceptual (motivation/related work).

---

## Core method

### I-JEPA

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M.,
LeCun, Y., & Ballas, N. (2023). *Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture.* CVPR 2023.
[arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

**Use:** Our pretraining method. The entire `src/` training loop, masking
collator, ViT encoder, predictor, and EMA target encoder implement I-JEPA.
Direct dependency.

```bibtex
@inproceedings{assran2023ijepa,
  title     = {Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author    = {Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  booktitle = {CVPR},
  year      = {2023}
}
```

### MIRAGE

> NEEDS VERIFICATION: exact publication venue and author list for the MIRAGE
> retinal segmentation model. The weights are loaded via
> `scripts/jepa_to_mirage_probe.py::build_mirage()` from a local checkpoint.

**Use:** Frozen anatomy teacher. Provides the 4-class retinal segmentation
guide that drives anatomy-guided target sampling. All 95.6M parameters are
frozen (`p.requires_grad_(False)` at `scripts/jepa_to_mirage_probe.py:87`).
Direct dependency (weights).

### FairVision

Luo, Y., Shi, D., Tao, R., Elze, T., & Wang, M. (2024). *FairVision:
Equitable Deep Learning for Eye Disease Screening via Fair Identity Scaling.*
[arXiv:2310.02492](https://arxiv.org/abs/2310.02492)

**Use:** Our pretraining and evaluation dataset. Glaucoma subset: 10,000
subjects (6K/1K/3K Train/Val/Test), 200×200×200 OCT volumes, binary label.
Direct dependency.

```bibtex
@article{luo2024fairvision,
  title   = {FairVision: Equitable Deep Learning for Eye Disease Screening via Fair Identity Scaling},
  author  = {Luo, Yan and Shi, Dayi and Tao, Ruobing and Elze, Tobias and Wang, Mengyu},
  journal = {arXiv preprint arXiv:2310.02492},
  year    = {2024}
}
```

---

## Model components

### Vision Transformer (ViT)

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S.,
Uszkoreit, J., & Houlsby, N. (2021). *An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale.* ICLR 2021.
[arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

**Use:** ViT-B/16 is our encoder and target encoder architecture
(`src/models/vision_transformer.py`). Direct dependency (architecture).

```bibtex
@inproceedings{dosovitskiy2021image,
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author    = {Dosovitskiy, Alexei and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle = {ICLR},
  year      = {2021}
}
```

### ConvNeXt (MIRAGE output adapter)

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S.
(2022). *A ConvNet for the 2020s.* CVPR 2022.
[arXiv:2201.03545](https://arxiv.org/abs/2201.03545)

**Use:** The MIRAGE output adapter is a `ConvNeXtAdapter` (9.5M params) used
for semantic segmentation decoding. We do not train it — it is frozen inside
the MIRAGE teacher. Indirect dependency (architecture of a frozen component).

```bibtex
@inproceedings{liu2022convnet,
  title     = {A ConvNet for the 2020s},
  author    = {Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle = {CVPR},
  year      = {2022}
}
```

---

## Optimisation

### AdamW

Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.*
ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

**Use:** Optimizer for both pretraining (`src/helper.py:194`) and downstream
evaluation (`src/eval_downstream.py:575`). Direct dependency.

```bibtex
@inproceedings{loshchilov2019decoupled,
  title     = {Decoupled Weight Decay Regularization},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {ICLR},
  year      = {2019}
}
```

### Cosine annealing schedule

Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with
Warm Restarts.* ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)

**Use:** Warmup-cosine LR schedule for pretraining (`src/utils/schedulers.py`,
`WarmupCosineSchedule`) and cosine-with-warmup for downstream
(`src/eval_downstream.py:195`). Direct dependency.

```bibtex
@inproceedings{loshchilov2017sgdr,
  title     = {{SGDR}: Stochastic Gradient Descent with Warm Restarts},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {ICLR},
  year      = {2017}
}
```

### EMA / momentum encoder

Grill, J.-B., Strub, F., Altché, F., Tallec, C., Richemond, P. H.,
Buchatskaya, E., Doersch, C., Pinto, B. Á., Zheng, Z., Azabou, M., et al.
(2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.*
NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)

**Use:** The EMA target encoder idea (context encoder → exponential moving
average → target encoder, cosine schedule 0.996→1.0) follows BYOL / I-JEPA
convention. Implemented in `src/train_patch.py:102–106`. Direct dependency
(technique).

```bibtex
@inproceedings{grill2020bootstrap,
  title     = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
  author    = {Grill, Jean-Bastien and Strub, Florian and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre H and Buchatskaya, Elena and Doersch, Carl and Pinto, Bernardo {\'A}vila and Zheng, Zhan and Azabou, Mohammad and others},
  booktitle = {NeurIPS},
  year      = {2020}
}
```

---

## Framework

### PyTorch

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G.,
Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. (2019). *PyTorch: An
Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.

**Use:** Entire codebase. Direct dependency (`torch==2.7.1`).

```bibtex
@inproceedings{paszke2019pytorch,
  title     = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle = {NeurIPS},
  year      = {2019}
}
```

### timm (PyTorch Image Models)

Wightman, R. (2019). *PyTorch Image Models.*
[GitHub](https://github.com/huggingface/pytorch-image-models)

**Use:** Used only for the ImageNet ViT download script
(`scripts/download_imagenet_vit.py`). Not a core dependency.

```bibtex
@misc{wightman2019timm,
  author = {Wightman, Ross},
  title  = {PyTorch Image Models},
  year   = {2019},
  url    = {https://github.com/huggingface/pytorch-image-models}
}
```

---

## Evaluation references

### Attentive probing for MIM

Kakogeorgiou, I., Gidaris, S., Bursuc, A., Komodakis, N., & Laptev, I.
(2026). *Attention, Please! Revisiting Attentive Probing for Masked Image
Modeling.* ICLR 2026.
[arXiv:2506.10178](https://arxiv.org/abs/2506.10178)

**Use:** Context for our probe-architecture ablation. Documents that attentive
probes are over-parameterised. Our finding (CrossAttnPool beats d=1 at 26×
fewer params) aligns. Background/related work.

```bibtex
@inproceedings{kakogeorgiou2026attentive,
  title     = {Attention, Please! Revisiting Attentive Probing for Masked Image Modeling},
  author    = {Kakogeorgiou, Ioannis and Gidaris, Spyros and Bursuc, Andrei and Komodakis, Nikos and Laptev, Ivan},
  booktitle = {ICLR},
  year      = {2026}
}
```

### Zhou et al. 2025 — Generalist vs Specialist VFMs

Zhou, Y., et al. (2025). *Generalist vs Specialist Vision Foundation Models
for Ocular Disease and Oculomics.*
[arXiv:2509.03421](https://arxiv.org/abs/2509.03421v1)

**Use:** Benchmarks that motivate our approach (DINOv3 > DINOv2, fine-tune >
linear probe by 2–5 %). Phase 4 baseline comparisons. Background.

```bibtex
@article{zhou2025generalist,
  title   = {Generalist vs Specialist Vision Foundation Models for Ocular Disease and Oculomics},
  author  = {Zhou, Yukun and others},
  journal = {arXiv preprint arXiv:2509.03421},
  year    = {2025}
}
```

### RETFound

Zhou, Y., Chia, M. A., Wagner, S. K., et al. (2023). *A Foundation Model for
Generalizable Disease Detection from Retinal Images.* Nature 2023.
[DOI:10.1038/s41586-023-06555-x](https://www.nature.com/articles/s41586-023-06555-x)

**Use:** Medical-VFM baseline. Key glaucoma numbers: fine-tuned OCT AUC ~0.91.
Phase 4 comparison target. Background.

```bibtex
@article{zhou2023retfound,
  title   = {A Foundation Model for Generalizable Disease Detection from Retinal Images},
  author  = {Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and others},
  journal = {Nature},
  year    = {2023},
  doi     = {10.1038/s41586-023-06555-x}
}
```

---

## Masking strategy literature

### SemMAE

Li, G., Zheng, H., Liu, D., Wang, C., Su, B., & Zheng, C. (2022). *SemMAE:
Semantic-Guided Masking for Learning Masked Autoencoders.* NeurIPS 2022.
[arXiv:2206.10207](https://arxiv.org/abs/2206.10207)

**Use:** Related work for guided masking. Their semantic partitioning of patches
is analogous to our anatomy-guided target placement. Background.

```bibtex
@inproceedings{li2022semmae,
  title     = {{SemMAE}: Semantic-Guided Masking for Learning Masked Autoencoders},
  author    = {Li, Gang and Zheng, Heliang and Liu, Daqing and Wang, Chaoyue and Su, Bing and Zheng, Changwen},
  booktitle = {NeurIPS},
  year      = {2022}
}
```

### AttMask

Kakogeorgiou, I., Gidaris, S., Psomas, B., Avrithis, Y., Bursuc, A.,
Karantzalos, K., & Komodakis, N. (2022). *What to Hide from Your Students:
Attention-Guided Masked Image Modeling.* ECCV 2022.
[arXiv:2203.12719](https://arxiv.org/abs/2203.12719)

**Use:** Related work. Teacher-attention-driven masking for MIM. Background.

```bibtex
@inproceedings{kakogeorgiou2022attmask,
  title     = {What to Hide from Your Students: Attention-Guided Masked Image Modeling},
  author    = {Kakogeorgiou, Ioannis and Gidaris, Spyros and Psomas, Bill and Avrithis, Yannis and Bursuc, Andrei and Karantzalos, Konstantinos and Komodakis, Nikos},
  booktitle = {ECCV},
  year      = {2022}
}
```

### AnatoMask

> NEEDS VERIFICATION: exact author list.

(2024). *AnatoMask: Enhancing Medical Image Segmentation with
Reconstruction-guided Self-masking.* ECCV 2024.
[arXiv:2407.06468](https://arxiv.org/abs/2407.06468)

**Use:** Related work for anatomy-aware masking in medical imaging. Background.

```bibtex
@inproceedings{anatomask2024,
  title     = {{AnatoMask}: Enhancing Medical Image Segmentation with Reconstruction-guided Self-masking},
  booktitle = {ECCV},
  year      = {2024}
}
```

### DMT-JEPA

(2024). *DMT-JEPA: Discriminative Masked Targets for Joint-Embedding
Predictive Architecture.*
[arXiv:2405.17995](https://arxiv.org/abs/2405.17995)

**Use:** Related work — discriminative target selection for JEPA. Background.

```bibtex
@article{dmtjepa2024,
  title   = {{DMT-JEPA}: Discriminative Masked Targets for Joint-Embedding Predictive Architecture},
  journal = {arXiv preprint arXiv:2405.17995},
  year    = {2024}
}
```

### V-JEPA

Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran,
M., & Ballas, N. (2024). *Revisiting Feature Prediction for Learning Visual
Representations from Video.*
[arXiv:2404.08471](https://arxiv.org/html/2404.08471v1)

**Use:** Video JEPA. Precedent for multi-frame/slice aggregation and 4-block
attentive probe design. Background.

```bibtex
@article{bardes2024vjepa,
  title   = {Revisiting Feature Prediction for Learning Visual Representations from Video},
  author  = {Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Chen, Xinlei and Rabbat, Michael and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal = {arXiv preprint arXiv:2404.08471},
  year    = {2024}
}
```

### LeJEPA

Balestriero, R. & LeCun, Y. (2025). *LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics.*
[arXiv:2511.08544](https://arxiv.org/abs/2511.08544)

**Use:** Theoretical grounding for JEPA objectives. Background.

```bibtex
@article{balestriero2025lejepa,
  title   = {{LeJEPA}: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author  = {Balestriero, Randall and LeCun, Yann},
  journal = {arXiv preprint arXiv:2511.08544},
  year    = {2025}
}
```

---

## Method positioning / related work

These citations position our contributions relative to existing literature.
Each includes the user's original justification preserved verbatim, followed by
a verified implementation note.

### 1. Relational Knowledge Distillation

Park, W., Kim, D., Lu, Y., & Cho, M. (2019). *Relational Knowledge
Distillation.* CVPR 2019.
[arXiv:1904.05068](https://arxiv.org/abs/1904.05068)

**Justification:** The closest established literature family for our `L_rel`:
rather than forcing one feature vector to exactly match another, transfer
relationships among representations. Our exact loss is different — we compare
spatial patch cosine Gram matrices — but RKD is the right citation for the
general idea of relational rather than pointwise transfer.

**Implementation link:** `L_rel` is defined in `scripts/adapter_stage.py:13`
as `MSE(Gram(pool(H)), sg(Gram(Z_ema)))` and equivalently in
`scripts/jepa_to_mirage_probe.py:10`. The `gram()` function
(L2-normalise → inner product) at `scripts/adapter_stage.py:85` produces the
cosine Gram matrix that defines our relational structure.

```bibtex
@inproceedings{park2019relational,
  title     = {Relational Knowledge Distillation},
  author    = {Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
  booktitle = {CVPR},
  year      = {2019}
}
```

### 2. DSeq-JEPA

He, et al. (2026). *Discriminative Sequential Joint-Embedding Predictive
Architecture.* ECCV 2026.

> NEEDS VERIFICATION: exact author list, arXiv ID, and page numbers for
> DSeq-JEPA (ECCV 2026). The user asserts this paper exists; bibliographic
> details not independently confirmed.

**Justification:** Important related work because it explicitly replaces purely
random JEPA targeting with semantically/discriminatively selected image regions
using an attention-derived saliency map. Our method differs substantially:
anatomy comes from MIRAGE, targets remain our four anatomy-guided regions, and
we are not using DSeq's sequential prediction procedure. But this is the
closest JEPA paper for the claim that WHERE you choose targets matters
semantically.

**Implementation link:** Our anatomy-guided target sampler lives in
`src/masks/curriculum.py` (the `CurriculumMaskGenerator` class). The guide
source is the precomputed MIRAGE segmentation cache
(`configs/patch_mirage_anatomy.yaml`), not an attention-derived saliency map.

```bibtex
@inproceedings{he2026dseqjepa,
  title     = {Discriminative Sequential Joint-Embedding Predictive Architecture},
  author    = {He, et al.},
  booktitle = {ECCV},
  year      = {2026},
  note      = {NEEDS VERIFICATION: exact bibliographic details}
}
```

### 3. ReZero

Bachlechner, T., Majumder, B. P., Mao, H., Cottrell, G. W., & McAuley, J.
(2020). *ReZero is All You Need: Fast Convergence at Large Depth.*
[arXiv:2003.04887](https://arxiv.org/abs/2003.04887)

**Justification:** Optional but useful for the zero-initialised residual idea.
Our adapter starts with zero output so H = H0 exactly at initialisation. ReZero
uses zero-initialised residual gating rather than our exact architecture, so
cite as precedent for the design principle, not as the source of our adapter.

**Implementation link:** The zero-init is at
`scripts/adapter_stage.py:79–80` (`nn.init.zeros_` on the output conv weight
and bias) and identically at `scripts/jepa_to_mirage_probe.py:196–197`. The
`Adapter.forward` returns `h0 + alpha * tanh(out(trunk(h0)))`, which is exactly
`h0` when `out` is zero-initialised.

```bibtex
@article{bachlechner2020rezero,
  title   = {{ReZero} is All You Need: Fast Convergence at Large Depth},
  author  = {Bachlechner, Thomas and Majumder, Bodhisattwa Prasad and Mao, Huanru and Cottrell, Garrison W and McAuley, Julian},
  journal = {arXiv preprint arXiv:2003.04887},
  year    = {2020}
}
```

### 4. PEFT without Catastrophic Forgetting

Bafghi, R. E., Harl, M., & Bowyer, K. W. (2024). *Parameter Efficient
Fine-tuning of Self-supervised ViTs without Catastrophic Forgetting.*

> NEEDS VERIFICATION: exact venue (likely CVPR-W or arXiv) and arXiv ID for
> Bafghi et al. 2024.

**Justification:** Used in the discussion about adapting a pretrained vision
model without destroying its previous representation. It studies PEFT
approaches such as block expansion and LoRA while trying to preserve pretrained
capabilities. Our final solution is stricter — all original MIRAGE parameters
are frozen and only our adapter changes — so this is motivation/related work,
not something we directly implement.

**Implementation link:** MIRAGE freezing is at
`scripts/jepa_to_mirage_probe.py:84–87` (loop over all parameters setting
`requires_grad_(False)`). The adapter's 689K parameters are the ONLY trainable
component when adapting MIRAGE's output.

```bibtex
@article{bafghi2024peft,
  title   = {Parameter Efficient Fine-tuning of Self-supervised {ViTs} without Catastrophic Forgetting},
  author  = {Bafghi, Reza Esfandiarpour and Harl, Myra and Bowyer, Kevin W.},
  year    = {2024},
  note    = {NEEDS VERIFICATION: exact venue and arXiv ID}
}
```

### 5. ADIOS

Shi, Y., Siddharth, N., Paige, B., & Torr, P. H. S. (2022). *Adversarial
Masking for Self-Supervised Learning.* NeurIPS 2022.

> NEEDS VERIFICATION: confirm NeurIPS 2022 venue and arXiv ID for ADIOS.

**Justification:** We considered an adversarial masker because ADIOS learns a
masking function jointly with an encoder. We REJECTED this direction because it
adds a minimax objective and can choose pathological/impossible targets. Still a
useful related-work citation when discussing learned masking alternatives.

**Implementation link:** Rejection is documented in
`docs/experiments/masking/ablations.md` under "Rejected designs". We do not
implement any adversarial masking component.

```bibtex
@inproceedings{shi2022adios,
  title     = {Adversarial Masking for Self-Supervised Learning},
  author    = {Shi, Yuge and Siddharth, N and Paige, Brooks and Torr, Philip H S},
  booktitle = {NeurIPS},
  year      = {2022},
  note      = {NEEDS VERIFICATION: exact venue confirmation}
}
```
