# Citations & Related Work

> Single source for the paper's bibliography. Every entry states what we use it
> for in THIS repository, whether the dependency is direct (code/weights) or
> conceptual (motivation/related work), and how it relates to our method.
>
> **Our method in one line:** Anatomy-shaped connected masking targets for
> I-JEPA derived from a frozen MIRAGE segmentation model, with a small
> trainable adapter updated by a relational loss against the JEPA EMA target
> encoder's patch Gram matrix.

---

## 1. Directly used (code / weight dependencies)

> These are things the codebase actually imports or loads.

### I-JEPA

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M.,
LeCun, Y., & Ballas, N. (2023). *Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture.* CVPR 2023.
[arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

**Repo file:** `src/` (entire training loop, masking collator, ViT encoder, predictor, EMA target encoder).

**Relation to this work:** Our base pretraining method. We keep the I-JEPA
encoder/predictor/EMA architecture intact and modify only the *target selection
strategy* — replacing random block targets with anatomy-guided connected regions
from MIRAGE.

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

**Repo file:** `scripts/jepa_to_mirage_probe.py:87` (frozen teacher, all 95.6M params).

**Relation to this work:** Frozen anatomy teacher. Provides the 4-class retinal
segmentation guide that drives anatomy-guided target sampling. Unlike methods
that learn a masking function end-to-end, we use a *fixed* external segmentation
model — this avoids co-adaptation and keeps the masking semantics interpretable.

### FairVision

Luo, Y., Khan, M. O., Tian, Y., Shi, M., Dou, Z., Elze, T., Fang, Y., &
Wang, M. (2024). *FairVision: Equitable Deep Learning for Eye Disease
Screening via Fair Identity Scaling.*
[arXiv:2310.02492](https://arxiv.org/abs/2310.02492)

**Repo file:** Data loading configs under `configs/`.

**Relation to this work:** Our pretraining and evaluation dataset (glaucoma
subset: 10,000 subjects, 200\u00d7200\u00d7200 OCT volumes, binary label). Provides
demographic metadata enabling fairness evaluation.

```bibtex
@article{luo2024fairvision,
  title   = {FairVision: Equitable Deep Learning for Eye Disease Screening via Fair Identity Scaling},
  author  = {Luo, Yan and Khan, Muhammad Osama and Tian, Yu and Shi, Min and Dou, Zehao and Elze, Tobias and Fang, Yi and Wang, Mengyu},
  journal = {arXiv preprint arXiv:2310.02492},
  year    = {2024}
}
```

### Vision Transformer (ViT)

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S.,
Uszkoreit, J., & Houlsby, N. (2021). *An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale.* ICLR 2021.
[arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

**Repo file:** `src/models/vision_transformer.py`.

**Relation to this work:** ViT-B/16 is our encoder and target encoder
architecture. We do not modify the ViT architecture itself.

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

**Repo file:** Frozen inside the MIRAGE teacher (9.5M params `ConvNeXtAdapter`).

**Relation to this work:** Architecture of a frozen component. We do not train
it; it decodes MIRAGE's segmentation outputs that drive our masking.

```bibtex
@inproceedings{liu2022convnet,
  title     = {A ConvNet for the 2020s},
  author    = {Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle = {CVPR},
  year      = {2022}
}
```

### AdamW

Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.*
ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

**Repo file:** `src/helper.py:194`, `src/eval_downstream.py:575`.

**Relation to this work:** Optimizer for pretraining and downstream evaluation.

```bibtex
@inproceedings{loshchilov2019decoupled,
  title     = {Decoupled Weight Decay Regularization},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {ICLR},
  year      = {2019}
}
```

### Cosine annealing schedule (SGDR)

Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with
Warm Restarts.* ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)

**Repo file:** `src/utils/schedulers.py` (`WarmupCosineSchedule`), `src/eval_downstream.py:195`.

**Relation to this work:** Warmup-cosine LR schedule for pretraining and downstream.

```bibtex
@inproceedings{loshchilov2017sgdr,
  title     = {{SGDR}: Stochastic Gradient Descent with Warm Restarts},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {ICLR},
  year      = {2017}
}
```

### EMA / momentum encoder (BYOL)

Grill, J.-B., Strub, F., Altche, F., Tallec, C., Richemond, P. H.,
Buchatskaya, E., Doersch, C., Pinto, B. A., Zheng, Z., Azabou, M., et al.
(2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.*
NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)

**Repo file:** `src/train_patch.py:102-106` (EMA cosine schedule 0.996 -> 1.0).

**Relation to this work:** The EMA target encoder convention follows BYOL/I-JEPA.
Our relational loss compares adapter output against EMA-encoded patch features,
creating a self-distillation loop that does not exist in standard I-JEPA.

```bibtex
@inproceedings{grill2020bootstrap,
  title     = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
  author    = {Grill, Jean-Bastien and Strub, Florian and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre H and Buchatskaya, Elena and Doersch, Carl and Pinto, Bernardo {\'A}vila and Zheng, Zhan and Azabou, Mohammad and others},
  booktitle = {NeurIPS},
  year      = {2020}
}
```

### PyTorch

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G.,
Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. (2019). *PyTorch: An
Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.

**Repo file:** Entire codebase (`torch==2.7.1`).

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

**Repo file:** `scripts/download_imagenet_vit.py`. Not a core dependency.

```bibtex
@misc{wightman2019timm,
  author = {Wightman, Ross},
  title  = {PyTorch Image Models},
  year   = {2019},
  url    = {https://github.com/huggingface/pytorch-image-models}
}
```

---

## 2. Masked image modelling foundations

> Core MIM/JEPA methods that define the architectural family we build on.

### MAE — Masked Autoencoders

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022).
*Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022, pages
16000-16009. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

**Relation to this work:** MAE established the mask-then-predict paradigm for
vision. I-JEPA replaces pixel reconstruction with latent prediction; our method
further replaces random masking with anatomy-guided masking. We differ from MAE
in three ways: (1) we predict in embedding space not pixel space, (2) we use
connected semantic targets not random patches, (3) we add a relational adapter
loss.

```bibtex
@inproceedings{he2022mae,
  title     = {Masked Autoencoders Are Scalable Vision Learners},
  author    = {He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle = {CVPR},
  pages     = {16000--16009},
  year      = {2022}
}
```

### V-JEPA

Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran,
M., & Ballas, N. (2024). *Revisiting Feature Prediction for Learning Visual
Representations from Video.* [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)

**Relation to this work:** V-JEPA extends I-JEPA to video with spatiotemporal
masking. Precedent for multi-frame/slice aggregation and 4-block attentive probe
design. We differ by operating on single 2D slices (not video), using anatomy
rather than motion for target selection, and adding the relational adapter.

```bibtex
@article{bardes2024vjepa,
  title   = {Revisiting Feature Prediction for Learning Visual Representations from Video},
  author  = {Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Chen, Xinlei and Rabbat, Michael and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal = {arXiv preprint arXiv:2404.08471},
  year    = {2024}
}
```

### Hiera

Ryali, C., Hu, Y.-T., Bolya, D., Wei, C., Fan, H., Huang, P.-Y., Aggarwal,
V., Chowdhury, A., Poursaeed, O., Hoffman, J., Malik, J., Li, Y., &
Feichtenhofer, C. (2023). *Hiera: A Hierarchical Vision Transformer without
the Bells-and-Whistles.* ICML 2023, pages 29441-29454, PMLR.
[arXiv:2306.00989](https://arxiv.org/abs/2306.00989)

**Relation to this work:** Hiera demonstrates that strong MAE pretraining
removes the need for complex hierarchical modules. Our philosophy is similar:
we keep the ViT backbone simple and invest complexity only in the masking
strategy. We do not use Hiera's architecture but its finding supports our
design choice of a vanilla ViT with smarter targets.

```bibtex
@inproceedings{ryali2023hiera,
  title     = {Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles},
  author    = {Ryali, Chaitanya and Hu, Yuan-Ting and Bolya, Daniel and Wei, Chen and Fan, Haoqi and Huang, Po-Yao and Aggarwal, Vaibhav and Chowdhury, Arkabandhu and Poursaeed, Omid and Hoffman, Judy and Malik, Jitendra and Li, Yanghao and Feichtenhofer, Christoph},
  booktitle = {ICML},
  pages     = {29441--29454},
  year      = {2023},
  publisher = {PMLR}
}
```

### LeJEPA

Balestriero, R. & LeCun, Y. (2025). *LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics.*
[arXiv:2511.08544](https://arxiv.org/abs/2511.08544)

**Relation to this work:** Theoretical grounding for JEPA objectives. Shows
that joint-embedding prediction can be derived from principled information-
theoretic objectives rather than heuristics. Supports our claim that modifying
target selection is a valid way to inject inductive bias without changing the
learning objective.

```bibtex
@article{balestriero2025lejepa,
  title   = {{LeJEPA}: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author  = {Balestriero, Randall and LeCun, Yann},
  journal = {arXiv preprint arXiv:2511.08544},
  year    = {2025}
}
```

---

## 3. Adaptive / guided masking

> Methods that go beyond random masking to select meaningful regions.

### AttMask

Kakogeorgiou, I., Gidaris, S., Psomas, B., Avrithis, Y., Bursuc, A.,
Karantzalos, K., & Komodakis, N. (2022). *What to Hide from Your Students:
Attention-Guided Masked Image Modeling.* ECCV 2022, pages 300-318, Springer.
[arXiv:2203.12719](https://arxiv.org/abs/2203.12719)

**Relation to this work:** Uses teacher attention to decide where to mask. We
differ in two ways: (1) our guide is an external segmentation model (MIRAGE)
not internal attention, providing anatomical rather than saliency-based
semantics; (2) we target I-JEPA's prediction regions, not MAE's reconstruction
targets.

```bibtex
@inproceedings{kakogeorgiou2022attmask,
  title     = {What to Hide from Your Students: Attention-Guided Masked Image Modeling},
  author    = {Kakogeorgiou, Ioannis and Gidaris, Spyros and Psomas, Bill and Avrithis, Yannis and Bursuc, Andrei and Karantzalos, Konstantinos and Komodakis, Nikos},
  booktitle = {ECCV},
  pages     = {300--318},
  year      = {2022},
  publisher = {Springer}
}
```

### SemMAE

Li, G., Zheng, H., Liu, D., Wang, C., Su, B., & Zheng, C. (2022). *SemMAE:
Semantic-Guided Masking for Learning Masked Autoencoders.* NeurIPS 2022, pages
14290-14302. [arXiv:2206.10207](https://arxiv.org/abs/2206.10207)

**Relation to this work:** Partitions patches by semantic part (learned
segmentation) and masks entire parts. Most similar to our anatomy-guided
approach in spirit. Key differences: (1) SemMAE learns the partition jointly;
we use a frozen external model — avoiding co-adaptation. (2) SemMAE operates
in pixel reconstruction (MAE); we operate in latent prediction (JEPA).
(3) SemMAE targets natural images; we target retinal OCT anatomy.

```bibtex
@inproceedings{li2022semmae,
  title     = {{SemMAE}: Semantic-Guided Masking for Learning Masked Autoencoders},
  author    = {Li, Gang and Zheng, Heliang and Liu, Daqing and Wang, Chaoyue and Su, Bing and Zheng, Changwen},
  booktitle = {NeurIPS},
  pages     = {14290--14302},
  year      = {2022}
}
```

### AutoMAE

Chen, H., Zhang, W., Wang, Y., & Yang, X. (2023). *Improving Masked
Autoencoders by Learning Where to Mask.* PRCV 2023, pages 377-390, Springer.

**Relation to this work:** Learns a masking policy to maximize downstream task
performance. Unlike our method, AutoMAE uses reinforcement learning to discover
mask patterns rather than leveraging pre-existing anatomical knowledge. We
argue that in medical imaging, domain-specific segmentation provides a stronger
inductive bias than a learned task-agnostic masking policy.

```bibtex
@inproceedings{chen2023automae,
  title     = {Improving Masked Autoencoders by Learning Where to Mask},
  author    = {Chen, Haijian and Zhang, Wendong and Wang, Yunbo and Yang, Xiaokang},
  booktitle = {PRCV},
  pages     = {377--390},
  year      = {2023},
  publisher = {Springer}
}
```

### Self-Guided MAE

Shin, J., Lee, I., Lee, J., & Lee, J. (2024). *Self-Guided Masked
Autoencoder.* NeurIPS 2024.

**Relation to this work:** Uses the model's own representations to guide
masking. We differ by using an external anatomy oracle (MIRAGE) rather than
self-derived signals. Self-guidance risks circular reasoning (the model
reinforces its own biases); external guidance from a frozen segmentation model
provides an independent inductive bias.

```bibtex
@inproceedings{shin2024selfguided,
  title     = {Self-Guided Masked Autoencoder},
  author    = {Shin, Jeongwoo and Lee, Inseo and Lee, Junho and Lee, Joonseok},
  booktitle = {NeurIPS},
  year      = {2024}
}
```

### Mask What Matters (text-guided medical masking)

Wang, R., Xu, S., Liu, B., Huang, R., Chen, D., & Su, W. (2025). *Mask What
Matters: Controllable Text-Guided Masking for Self-Supervised Medical Image
Analysis.* [arXiv:2509.23054](https://arxiv.org/abs/2509.23054)

> NEEDS VERIFICATION: arXiv ID 2509.23054 has an unusually high sequence
> number for a 2509.xxxxx paper. Verify this ID is correct.

**Relation to this work:** Uses text prompts to guide masking for medical
images. We use segmentation maps rather than text. Both share the insight that
domain knowledge should guide masking in medical SSL. Our approach requires no
language model and is deterministic given the segmentation; theirs is more
flexible but adds a text encoder dependency.

```bibtex
@article{wang2025maskwhatmatters,
  title   = {Mask What Matters: Controllable Text-Guided Masking for Self-Supervised Medical Image Analysis},
  author  = {Wang, Ruilang and Xu, Shuotong and Liu, Bowen and Huang, Runlin and Chen, Donglong and Su, Weifeng},
  journal = {arXiv preprint arXiv:2509.23054},
  year    = {2025}
}
```

### AnatoMask

> NEEDS VERIFICATION: exact author list for AnatoMask.

(2024). *AnatoMask: Enhancing Medical Image Segmentation with
Reconstruction-guided Self-masking.* ECCV 2024.
[arXiv:2407.06468](https://arxiv.org/abs/2407.06468)

**Relation to this work:** Anatomy-aware masking for medical image segmentation.
Closest prior work in the medical domain. Key difference: AnatoMask uses
reconstruction-guided self-masking (the model's own reconstructions guide future
masks), while we use a frozen external segmentation model, completely decoupling
the masking signal from the encoder being trained.

```bibtex
@inproceedings{anatomask2024,
  title     = {{AnatoMask}: Enhancing Medical Image Segmentation with Reconstruction-guided Self-masking},
  booktitle = {ECCV},
  year      = {2024},
  note      = {NEEDS VERIFICATION: exact author list}
}
```

---

## 4. JEPA target design

> Methods that modify what/how targets are selected in JEPA-like architectures.

### DMT-JEPA

Mo, S. & Yun, S. (2024). *DMT-JEPA: Discriminative Masked Targets for
Joint-Embedding Predictive Architecture.*
[arXiv:2405.17995](https://arxiv.org/abs/2405.17995)

**Relation to this work:** Proposes discriminative target selection for JEPA
using feature-space diversity. We share the insight that target selection
matters, but differ fundamentally: DMT-JEPA uses feature statistics to choose
targets at training time; we use precomputed anatomical regions. Our targets are
stable across training (fixed anatomy), whereas DMT-JEPA's evolve with the
encoder.

```bibtex
@article{mo2024dmtjepa,
  title   = {{DMT-JEPA}: Discriminative Masked Targets for Joint-Embedding Predictive Architecture},
  author  = {Mo, Shentong and Yun, Sukmin},
  journal = {arXiv preprint arXiv:2405.17995},
  year    = {2024}
}
```

### DSeq-JEPA

He, X., Sakai, S., Chandhok, S., Beery, S., Yuan, K., Padoy, N., Hasegawa,
T., & Sigal, L. (2025). *DSeq-JEPA: Discriminative Sequential Joint-Embedding
Predictive Architecture.* [arXiv:2511.17354](https://arxiv.org/abs/2511.17354)

> **INCONSISTENCY FLAG:** The user's draft cites DSeq-JEPA as "He et al., ECCV
> 2026" in the Method Positioning section but as "arXiv:2511.17354, 2025" in
> the reference list. Web verification confirms arXiv:2511.17354 exists (Nov
> 2025 preprint). The ECCV 2026 venue claim is **unverified** — ECCV 2026 has
> not yet occurred as of this writing. Using the confirmed arXiv details.
>
> Note: The user-supplied author list [6] differs slightly from the web-verified
> list (user omits Chandhok and Beery). Using verified list.

**Relation to this work:** Most closely related JEPA paper. Replaces random
targeting with semantically/discriminatively selected regions using an
attention-derived saliency map and sequential prediction. We differ: (1)
anatomy comes from MIRAGE, not attention; (2) targets are our four anatomy
classes, not arbitrary salient regions; (3) we do not use sequential prediction;
(4) we add a relational loss absent in DSeq-JEPA.

**Repo link:** Our anatomy-guided target sampler in `src/masks/curriculum.py`
(`CurriculumMaskGenerator`); guide source is precomputed MIRAGE segmentation
cache (`configs/patch_mirage_anatomy.yaml`).

```bibtex
@article{he2025dseqjepa,
  title   = {{DSeq-JEPA}: Discriminative Sequential Joint-Embedding Predictive Architecture},
  author  = {He, Xiangteng and Sakai, Shunsuke and Chandhok, Shivam and Beery, Sara and Yuan, Kun and Padoy, Nicolas and Hasegawa, Tatsuhito and Sigal, Leonid},
  journal = {arXiv preprint arXiv:2511.17354},
  year    = {2025}
}
```

---

## 5. Representation transfer and adapters

> Knowledge distillation, adapter design, and parameter-efficient methods.

### Relational Knowledge Distillation (RKD)

Park, W., Kim, D., Lu, Y., & Cho, M. (2019). *Relational Knowledge
Distillation.* CVPR 2019. [arXiv:1904.05068](https://arxiv.org/abs/1904.05068)

**Relation to this work:** Closest established literature for our `L_rel`. RKD
transfers *relationships among representations* rather than forcing pointwise
alignment. Our exact loss differs — we compare spatial patch cosine Gram
matrices via MSE — but RKD is the correct citation for the general idea of
relational rather than pointwise transfer.

**Repo link:** `L_rel` at `scripts/adapter_stage.py:13`; `gram()` function at
`scripts/adapter_stage.py:85` (L2-normalise -> inner product -> cosine Gram
matrix).

```bibtex
@inproceedings{park2019relational,
  title     = {Relational Knowledge Distillation},
  author    = {Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
  booktitle = {CVPR},
  year      = {2019}
}
```

### ReZero

Bachlechner, T., Majumder, B. P., Mao, H., Cottrell, G. W., & McAuley, J.
(2020). *ReZero is All You Need: Fast Convergence at Large Depth.*
[arXiv:2003.04887](https://arxiv.org/abs/2003.04887)

**Relation to this work:** Precedent for zero-initialised residual connections.
Our adapter starts with zero output so `H = H0` exactly at initialisation,
ensuring no disruption to the frozen encoder at the start of adapter training.
ReZero uses zero-initialised gating rather than our exact architecture, but the
design principle (identity at init) is the same.

**Repo link:** Zero-init at `scripts/adapter_stage.py:79-80`
(`nn.init.zeros_` on output conv); forward: `h0 + alpha * tanh(out(trunk(h0)))`.

```bibtex
@article{bachlechner2020rezero,
  title   = {{ReZero} is All You Need: Fast Convergence at Large Depth},
  author  = {Bachlechner, Thomas and Majumder, Bodhisattwa Prasad and Mao, Huanru and Cottrell, Garrison W and McAuley, Julian},
  journal = {arXiv preprint arXiv:2003.04887},
  year    = {2020}
}
```

### PEFT without Catastrophic Forgetting

Bafghi, R. A., Harilal, N., Monteleoni, C., & Raissi, M. (2024). *Parameter
Efficient Fine-tuning of Self-supervised ViTs without Catastrophic Forgetting.*
CVPR Workshops (eLVM) 2024. [arXiv:2404.17245](https://arxiv.org/abs/2404.17245)

**Relation to this work:** Studies PEFT approaches (block expansion, LoRA) while
preserving pretrained capabilities. Our solution is stricter — all original
MIRAGE parameters are frozen and only our 689K adapter parameters change. This
paper motivates our design choice of complete freezing over partial fine-tuning.

**Repo link:** MIRAGE freezing at `scripts/jepa_to_mirage_probe.py:84-87`.

```bibtex
@inproceedings{bafghi2024peft,
  title     = {Parameter Efficient Fine-tuning of Self-supervised {ViTs} without Catastrophic Forgetting},
  author    = {Bafghi, Reza Akbarian and Harilal, Nidhin and Monteleoni, Claire and Raissi, Maziar},
  booktitle = {CVPR Workshops (eLVM)},
  year      = {2024},
  eprint    = {2404.17245},
  archivePrefix = {arXiv}
}
```

---

## 6. Medical imaging and OCT

> Medical vision models and datasets relevant to our application domain.

### Avram et al. — Deep vision model pretrained with 2D scans

Avram, O., Durmus, B., Rakocz, N., Corradetti, G., An, U., Nittala, M. G.,
et al. (2025). *Accurate Prediction of Disease-Risk Factors from Volumetric
Medical Scans by a Deep Vision Model Pre-trained with 2D Scans.* Nature
Biomedical Engineering, 2025.

**Relation to this work:** Demonstrates that 2D-pretrained models transfer
effectively to volumetric medical scans. Supports our slice-based approach:
we pretrain I-JEPA on 2D OCT slices rather than requiring 3D architectures,
consistent with their finding that 2D pretraining is competitive.

```bibtex
@article{avram2025accurate,
  title   = {Accurate Prediction of Disease-Risk Factors from Volumetric Medical Scans by a Deep Vision Model Pre-trained with 2D Scans},
  author  = {Avram, Oren and Durmus, Berkin and Rakocz, Nadav and Corradetti, Giulia and An, Ulzee and Nittala, Muneeswar G and others},
  journal = {Nature Biomedical Engineering},
  year    = {2025}
}
```

### OCTCube

Liu, Z., Xu, H., Woicik, A., Shapiro, L. G., Blazes, M., Wu, Y., Steffen,
V., Cukras, C., Lee, C. S., Zhang, M., Lee, A. Y., & Wang, S. (2024).
*OCTCube: A 3D Foundation Model for Optical Coherence Tomography.*
[arXiv:2408.11227](https://arxiv.org/abs/2408.11227)

> NEEDS VERIFICATION: The original paper title on arXiv is "OCTCube-M" (the
> multimodal version). An earlier version may be titled just "OCTCube". Using
> the arXiv ID confirmed via web search. Also published as "A three-dimensional
> multi-modal foundation model for optical coherence tomography" in Nature
> Biomedical Engineering 2026 (DOI:10.1038/s41551-026-01662-2).

**Relation to this work:** 3D OCT foundation model using volumetric masked
autoencoding. Represents the alternative design choice of full 3D processing.
We deliberately chose 2D slice-based I-JEPA for computational efficiency and
because anatomy-guided masking is more naturally defined on 2D cross-sections
where retinal layers are visible. OCTCube requires substantially more compute
for pretraining.

```bibtex
@article{liu2024octcube,
  title   = {{OCTCube}: A 3D Foundation Model for Optical Coherence Tomography},
  author  = {Liu, Zixuan and Xu, Hanwen and Woicik, Addie and Shapiro, Linda G and Blazes, Marian and Wu, Yue and Steffen, Verena and Cukras, Catherine and Lee, Cecilia S and Zhang, Miao and Lee, Aaron Y and Wang, Sheng},
  journal = {arXiv preprint arXiv:2408.11227},
  year    = {2024}
}
```

### RETFound

Zhou, Y., Chia, M. A., Wagner, S. K., et al. (2023). *A Foundation Model for
Generalizable Disease Detection from Retinal Images.* Nature 2023.
[DOI:10.1038/s41586-023-06555-x](https://www.nature.com/articles/s41586-023-06555-x)

**Relation to this work:** Medical-VFM baseline (MAE pretrained on retinal
fundus/OCT). Key glaucoma numbers: fine-tuned OCT AUC ~0.91. Phase 4
comparison target. RETFound uses random MAE masking; we hypothesise that
anatomy-guided masking learns better retinal representations.

```bibtex
@article{zhou2023retfound,
  title   = {A Foundation Model for Generalizable Disease Detection from Retinal Images},
  author  = {Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and others},
  journal = {Nature},
  year    = {2023},
  doi     = {10.1038/s41586-023-06555-x}
}
```

### Zhou et al. 2025 — Generalist vs Specialist VFMs

Zhou, Y., et al. (2025). *Generalist vs Specialist Vision Foundation Models
for Ocular Disease and Oculomics.*
[arXiv:2509.03421](https://arxiv.org/abs/2509.03421)

**Relation to this work:** Benchmarks motivating our approach (DINOv3 > DINOv2,
fine-tune > linear probe by 2-5%). Informs our Phase 4 baseline comparisons.

```bibtex
@article{zhou2025generalist,
  title   = {Generalist vs Specialist Vision Foundation Models for Ocular Disease and Oculomics},
  author  = {Zhou, Yukun and others},
  journal = {arXiv preprint arXiv:2509.03421},
  year    = {2025}
}
```

### Robust multimodal learning for ophthalmic disease grading

Wang, X., Wang, Y., Liang, S., Tang, F., Liu, C., Hu, M., Hu, C., He, J.,
Ge, Z., & Razzak, I. (2025). *Robust Multimodal Learning for Ophthalmic
Disease Grading via Disentangled Representation.* 2025.

> NEEDS VERIFICATION: exact venue (likely arXiv or a 2025 conference). No
> arXiv ID provided by user.

**Relation to this work:** Multimodal approach to ophthalmic grading using
disentangled representations. We are unimodal (OCT only) with anatomy-guided
SSL. Their multimodal fusion is complementary to our approach and could be a
future extension if additional modalities (e.g., fundus photos) are added.

```bibtex
@article{wang2025robust,
  title   = {Robust Multimodal Learning for Ophthalmic Disease Grading via Disentangled Representation},
  author  = {Wang, Xinkun and Wang, Yifang and Liang, Senwei and Tang, Feilong and Liu, Chengzhi and Hu, Ming and Hu, Chao and He, Junjun and Ge, Zongyuan and Razzak, Imran},
  year    = {2025},
  note    = {NEEDS VERIFICATION: exact venue and arXiv ID}
}
```

---

## 7. Efficiency / probing

> Efficient attention and probe design for evaluation.

### FlashAttention

Dao, T., Fu, D., Ermon, S., Rudra, A., & Re, C. (2022). *FlashAttention: Fast
and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

**Relation to this work:** Enables efficient training of our ViT encoder at
longer sequence lengths. Used as an implementation detail for attention
computation; does not change the mathematical formulation of our method.

```bibtex
@inproceedings{dao2022flashattention,
  title     = {{FlashAttention}: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author    = {Dao, Tri and Fu, Dan and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle = {NeurIPS},
  year      = {2022}
}
```

### Attentive probing (Psomas et al.)

Psomas, B., Christopoulos, D., Baltzi, E., Kakogeorgiou, I., Aravanis, T.,
Komodakis, N., Karantzalos, K., Avrithis, Y., & Tolias, G. (2026).
*Attention, Please! Revisiting Attentive Probing Through the Lens of
Efficiency.* ICLR 2026.

**Relation to this work:** Documents that attentive probes are
over-parameterised for MIM evaluation. Our finding (CrossAttnPool beats d=1 at
26x fewer params) aligns with their conclusions. Motivates our lightweight
probing protocol.

```bibtex
@inproceedings{psomas2026attentive,
  title     = {Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency},
  author    = {Psomas, Bill and Christopoulos, Dionysis and Baltzi, Eirini and Kakogeorgiou, Ioannis and Aravanis, Tilemachos and Komodakis, Nikos and Karantzalos, Konstantinos and Avrithis, Yannis and Tolias, Giorgos},
  booktitle = {ICLR},
  year      = {2026}
}
```

### PatchSAE

Lim, H., Choi, J., Choo, J., & Schneider, S. (2025). *Sparse Autoencoders
Reveal Selective Remapping of Visual Concepts During Adaptation.* ICLR 2025.
[arXiv:2412.05276](https://arxiv.org/abs/2412.05276)

**Relation to this work:** Sparse autoencoder-based patch clustering for
interpreting ViT representations. Referenced in our Method section as a score
function component — PatchSAE-style clustering can quantify which anatomy
regions are best predicted, informing the curriculum schedule.

```bibtex
@inproceedings{lim2025patchsae,
  title     = {Sparse Autoencoders Reveal Selective Remapping of Visual Concepts During Adaptation},
  author    = {Lim, Hyesu and Choi, Jinho and Choo, Jaegul and Schneider, Steffen},
  booktitle = {ICLR},
  year      = {2025}
}
```

### Galileo — MAE across data modalities

Tseng, G., Fuller, A., Reil, M., Herzog, H., Beukema, P., Bastani, F., Green,
J. R., Shelhamer, E., Kerner, H., & Rolnick, D. (2025). *Galileo: Learning
Global and Local Features of Many Remote Sensing Modalities.* ICML 2025.
[arXiv:2502.09356](https://arxiv.org/abs/2502.09356)

**Relation to this work:** Demonstrates masked modelling across heterogeneous
data modalities. Supports the generality of our approach: anatomy-guided masking
could extend to other modalities (e.g., multi-spectral OCT, OCTA) following
Galileo's multi-modal framework. We currently operate on a single modality.

```bibtex
@inproceedings{tseng2025galileo,
  title     = {Galileo: Learning Global and Local Features of Many Remote Sensing Modalities},
  author    = {Tseng, Gabriel and Fuller, Anthony and Reil, Marlena and Herzog, Henry and Beukema, Patrick and Bastani, Favyen and Green, James R and Shelhamer, Evan and Kerner, Hannah and Rolnick, David},
  booktitle = {ICML},
  year      = {2025}
}
```

---

## 8. Considered and rejected

> Methods we evaluated and deliberately chose not to adopt.

### ADIOS — Adversarial Masking for Self-Supervised Learning

Shi, Y., Siddharth, N., Torr, P. H. S., & Kosiorek, A. R. (2022).
*Adversarial Masking for Self-Supervised Learning.* ICML 2022, pages
20026-20040, PMLR. [arXiv:2201.13100](https://arxiv.org/abs/2201.13100)

**Reason for rejection:** ADIOS jointly trains a masking function and encoder
via an adversarial (minimax) objective. We rejected this for three reasons:
(1) the minimax objective adds training instability; (2) the adversarial masker
can select pathological/impossible targets that harm learning; (3) our frozen
MIRAGE approach provides stable, interpretable anatomy targets without these
failure modes.

**Repo link:** Rejection documented in `docs/experiments/masking/ablations.md`
under "Rejected designs". No adversarial masking component is implemented.

```bibtex
@inproceedings{shi2022adios,
  title     = {Adversarial Masking for Self-Supervised Learning},
  author    = {Shi, Yuge and Siddharth, N and Torr, Philip H S and Kosiorek, Adam R},
  booktitle = {ICML},
  pages     = {20026--20040},
  year      = {2022},
  publisher = {PMLR}
}
```

---

## Verification summary

| Entry | Status | Notes |
|-------|--------|-------|
| I-JEPA [1] | Confirmed | CVPR 2023, arXiv:2301.08243 |
| Avram et al. [2] | Partial | Nature BME 2025 per user; no arXiv ID to verify |
| V-JEPA [3] | Confirmed | arXiv:2404.08471 |
| AutoMAE [4] | Partial | PRCV 2023 per user; no arXiv ID to verify |
| MAE [5] | Confirmed | CVPR 2022, arXiv:2111.06377 |
| DSeq-JEPA [6] | Confirmed | arXiv:2511.17354, 2025. Author list corrected (8 authors). **ECCV 2026 venue UNVERIFIED** |
| AttMask [7] | Confirmed | ECCV 2022, arXiv:2203.12719 |
| SemMAE [8] | Confirmed | NeurIPS 2022, arXiv:2206.10207 |
| FairVision [9] | Confirmed | arXiv:2310.02492. Author list updated from user's version |
| DMT-JEPA [10] | Confirmed | arXiv:2405.17995 |
| Attentive probing [11] | Partial | ICLR 2026 per user; author list updated from [11] |
| Hiera [12] | Confirmed | ICML 2023, arXiv:2306.00989, PMLR 202:29441-29454 |
| Self-Guided MAE [13] | Partial | NeurIPS 2024 per user; no arXiv ID to verify |
| Mask What Matters [14] | NEEDS VERIFICATION | arXiv:2509.23054 — unusually high sequence number |
| Wang et al. multimodal [15] | NEEDS VERIFICATION | No venue or arXiv ID |
| OCTCube | Confirmed | arXiv:2408.11227; Nature BME 2026 |
| FlashAttention | Confirmed | NeurIPS 2022, arXiv:2205.14135 |
| Galileo | Confirmed | ICML 2025, arXiv:2502.09356 |
| PatchSAE | Confirmed | ICLR 2025, arXiv:2412.05276 |
| ADIOS | Confirmed | ICML 2022 (corrected from NeurIPS), arXiv:2201.13100. Author corrected (Kosiorek not Paige) |
| RKD | Confirmed | CVPR 2019, arXiv:1904.05068 |
| ReZero | Confirmed | arXiv:2003.04887 |
| Bafghi PEFT | Confirmed | CVPR-W (eLVM) 2024, arXiv:2404.17245. Author list corrected |
| MIRAGE | NEEDS VERIFICATION | No public citation found |
| AnatoMask | NEEDS VERIFICATION | Author list unknown |

---

## Inconsistencies found in user-supplied reference list

1. **DSeq-JEPA venue conflict:** Draft body says "He et al., ECCV 2026"; reference list [6] says "arXiv:2511.17354, 2025". The arXiv paper is confirmed to exist (Nov 2025). ECCV 2026 has not occurred yet — this venue claim cannot be verified.

2. **DSeq-JEPA author list:** User's [6] lists 6 authors (He, Sakai, Yuan, Padoy, Hasegawa, Sigal). Verified list has 8 authors including Shivam Chandhok and Sara Beery.

3. **ADIOS venue:** Existing file said "NeurIPS 2022"; verified as **ICML 2022**. Author "Brooks Paige" was listed but verified author is "Adam R. Kosiorek".

4. **Bafghi et al. authors:** User listed "Harl, M." and "Bowyer, K. W."; verified as "Harilal, Nidhin", "Monteleoni, Claire", and "Raissi, Maziar" (4 authors, not 3).

5. **FairVision authors:** User's [9] has different author set than existing file entry. User's list is more complete (includes Khan, Tian, Shi, Dou, Fang); used user's version.

6. **Attentive probing:** Existing file cited "Kakogeorgiou et al. 2026" (5 authors); user's [11] is "Psomas et al. 2026" (9 authors). These appear to be the same paper with Psomas as first author. Used [11]'s author list.
