# P2-02_related_work.md

# Related Work: Masking Policies for 3D I-JEPA Pretraining on Retinal OCT

**Research date:** 22 August 2026  
**Scope:** Foundations of masked/joint-embedding pretraining; informed masking; counter-evidence; medical and retinal SSL; evaluation methodology; Harvard-FairVision; and NeurIPS 2025 GenAI4Health venue calibration.

## Executive verdict

**Direct verdict:** “Informed masking does not reliably beat random masking” is a **known but contested result**, not a new general observation. Several influential papers report benefits from semantic, saliency, attention, or learned masking, but the benefits depend strongly on the objective, downstream protocol, task, and guidance quality. Conversely, MAE and SimMIM show simple random masking beating structured alternatives; SemMAE shows naïve semantic-part masking matching or severely underperforming random masking; AutoMAE is nearly tied with random MAE under full-data fine-tuning; Hard Patches Mining shows that excessive hard-mask selection is worse than random; and attention-guided methods report settings with ties or reversals.

What appears substantially more specific—and therefore plausibly novel—is the controlled result that, for **3D retinal OCT I-JEPA**, with architecture, data, schedule, and frozen-probe evaluation fixed, a **cheap image-derived localization heuristic beats masking driven by a trained retinal segmentation/foundation model**. The existing literature contains evidence that sophisticated guidance may be unnecessary or brittle, but I found no verified paper reporting this exact comparison in 3D OCT joint-embedding prediction.

---

# A. Foundations

## A1. I-JEPA

**Title:** Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture  
**Authors:** Mahmoud Assran; Quentin Duval; Ishan Misra; Piotr Bojanowski; Pascal Vincent; Michael Rabbat; Yann LeCun; Nicolas Ballas  
**Venue/year:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 15619–15629  
**Identifiers:** DOI: `10.1109/CVPR52729.2023.01499`; arXiv: `2301.08243`

I-JEPA predicts target-block representations from a context block rather than reconstructing pixels. Its masking ablation is unusually strong: with a ViT-B/16 and 1%-ImageNet linear evaluation, multi-block masking obtained 54.2% top-1, versus 20.2% for a single block, 17.6% for independently random target patches, and 15.5% for rasterized quadrants.

**Relation:** This is the direct methodological baseline. It shows that mask geometry can be essential in JEPA, but it does not evaluate anatomy-, segmentation-, or intensity-guided targets.

**Verification:** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html)

## A2. V-JEPA

**Title:** Revisiting Feature Prediction for Learning Visual Representations from Video  
**Authors:** Adrien Bardes; Quentin Garrido; Jean Ponce; Xinlei Chen; Michael Rabbat; Yann LeCun; Mahmoud Assran; Nicolas Ballas  
**Venue/year:** Transactions on Machine Learning Research, 2024  
**Identifiers:** arXiv: `2404.08471`; OpenReview: `QaCCuDfBk2`

V-JEPA extends joint-embedding feature prediction to videos and learns from masked spatiotemporal views without pixel reconstruction or a pretrained image encoder. Its largest frozen model is evaluated across motion- and appearance-oriented tasks.

**Relation:** It establishes JEPA-style feature prediction beyond 2D images and motivates volumetric/spatiotemporal extensions, but it does not test anatomical guidance.

**Verification:** [arXiv metadata](https://export.arxiv.org/api/query?id_list=2404.08471), [DBLP](https://dblp.org/rec/journals/tmlr/BardesGPCRLAB24)

## A3. MAE

**Title:** Masked Autoencoders Are Scalable Vision Learners  
**Authors:** Kaiming He; Xinlei Chen; Saining Xie; Yanghao Li; Piotr Dollár; Ross Girshick  
**Venue/year:** CVPR, 2022, pp. 16000–16009  
**Identifiers:** DOI: `10.1109/CVPR52688.2022.01553`; arXiv: `2111.06377`

MAE reconstructs randomly removed image patches with an asymmetric encoder-decoder and finds that a 75% random masking ratio is effective. In Table 1f, random 75% masking achieved 84.9% fine-tuning and 73.5% linear accuracy, versus 82.8%/63.9% for block masking at 75%, 83.9%/72.3% for block masking at 50%, and 84.0%/66.0% for grid masking.

**Relation:** Strong direct counter-evidence: simple random masking outperformed more structured block and grid masks, especially in linear evaluation.

**Verification:** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)

## A4. BEiT

**Title:** BEiT: BERT Pre-Training of Image Transformers  
**Authors:** Hangbo Bao; Li Dong; Songhao Piao; Furu Wei  
**Venue/year:** International Conference on Learning Representations (ICLR), 2022  
**Identifiers:** arXiv: `2106.08254`; OpenReview: `p-BhZSz59o4`

BEiT masks image patches and predicts discrete visual tokens produced by a tokenizer. It helped establish masked image modeling but introduced extra tokenization machinery and commonly used block-style masking.

**Relation:** Foundational masked-modeling baseline and a source of the block masks later found inferior to random masking in MAE and SimMIM.

**Verification:** [arXiv metadata](https://export.arxiv.org/api/query?id_list=2106.08254)

## A5. SimMIM

**Title:** SimMIM: A Simple Framework for Masked Image Modeling  
**Authors:** Zhenda Xie; Zheng Zhang; Yue Cao; Yutong Lin; Jianmin Bao; Zhuliang Yao; Qi Dai; Han Hu  
**Venue/year:** CVPR, 2022, pp. 9653–9663  
**Identifiers:** DOI: `10.1109/CVPR52688.2022.00943`; arXiv: `2111.09886`

SimMIM deliberately simplifies masked modeling through random masking, direct RGB regression, and a light prediction head. Its Table 1 reports a best random-mask result of 83.0% ImageNet top-1, versus at most 82.7% for block-wise masking and 82.6% for square masking.

**Relation:** Strong counter-evidence supporting random masking as a surprisingly difficult-to-beat baseline.

**Verification:** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.html)

## A6. DINO

**Title:** Emerging Properties in Self-Supervised Vision Transformers  
**Authors:** Mathilde Caron; Hugo Touvron; Ishan Misra; Hervé Jégou; Julien Mairal; Piotr Bojanowski; Armand Joulin  
**Venue/year:** IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 9650–9660  
**Identifiers:** DOI: `10.1109/ICCV48922.2021.00951`; arXiv: `2104.14294`

DINO trains a student and momentum teacher through self-distillation and demonstrates that ViT attention maps can reveal semantic object structure without labels. Those maps subsequently became guidance signals for AttMask, SemMAE, AutoMAE, and related methods.

**Relation:** Provides the attention/semantic features used by many informed-mask methods, but is not itself a masked autoencoder.

**Verification:** [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)

## A7. DINOv2

**Title:** DINOv2: Learning Robust Visual Features without Supervision  
**Authors:** Maxime Oquab; Timothée Darcet; Théo Moutakanni; Huy V. Vo; Marc Szafraniec; Vasil Khalidov; Pierre Fernandez; Daniel Haziza; Francisco Massa; Alaaeldin El-Nouby; Mahmoud Assran; Nicolas Ballas; Wojciech Galuba; Russell Howes; Po-Yao Huang; Shang-Wen Li; Ishan Misra; Michael Rabbat; Vasu Sharma; Gabriel Synnaeve; Hu Xu; Hervé Jégou; Julien Mairal; Patrick Labatut; Armand Joulin; Piotr Bojanowski  
**Venue/year:** Transactions on Machine Learning Research, 2024  
**Identifiers:** arXiv: `2304.07193`; OpenReview: `a68SUt6zFt`

DINOv2 scales self-supervised representation learning using a curated 142-million-image corpus, model scaling, stabilization, and distillation. It produces strong image- and pixel-level features and is a common general-purpose foundation-model comparator.

**Relation:** Provides a strong non-medical SSL baseline and can generate semantic/attention guidance without segmentation supervision.

**Verification:** [DBLP](https://dblp.org/rec/journals/tmlr/OquabDMVSKFHMEA24), [arXiv metadata](https://export.arxiv.org/api/query?id_list=2304.07193)

## A8. DINOv3

**Title:** DINOv3  
**Authors:** Oriane Siméoni; Huy V. Vo; Maximilian Seitzer; Federico Baldassarre; Maxime Oquab; Cijo Jose; Vasil Khalidov; Marc Szafraniec; Seung Eun Yi; Michaël Ramamonjisoa; Francisco Massa; Daniel Haziza; Luca Wehrstedt; Jianyuan Wang; Timothée Darcet; Théo Moutakanni; Leonel Sentana; Claire Roberts; Andrea Vedaldi; Jamie Tolan; John Brandt; Camille Couprie; Julien Mairal; Hervé Jégou; Patrick Labatut; Piotr Bojanowski  
**Venue/year:** Transactions on Machine Learning Research, 2026  
**Identifiers:** arXiv: `2508.10104`; OpenReview: `2NlGyqNjns`

DINOv3 scales self-supervised visual learning further and introduces Gram anchoring to prevent degradation of dense feature maps during long training. It supplies high-quality dense features across natural, aerial, and other image domains.

**Relation:** A current foundation-model comparator, but not direct evidence about target-mask policies.

**Verification:** [DBLP](https://dblp.org/rec/journals/tmlr/SimeoniVSBOJKSYRMHWWDMS26), [arXiv metadata](https://export.arxiv.org/api/query?id_list=2508.10104)

## A9. data2vec

**Title:** data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language  
**Authors:** Alexei Baevski; Wei-Ning Hsu; Qiantong Xu; Arun Babu; Jiatao Gu; Michael Auli  
**Venue/year:** Proceedings of the 39th International Conference on Machine Learning, PMLR 162, 2022, pp. 1298–1312  
**Identifiers:** arXiv: `2202.03555`

data2vec predicts contextualized latent representations of the full input from masked inputs using a teacher-student framework shared across speech, language, and vision. It is an important precursor/neighbor to joint-embedding prediction.

**Relation:** Provides a latent-target masked-prediction baseline and conceptual bridge between masked modeling and JEPA.

**Verification:** [PMLR](https://proceedings.mlr.press/v162/baevski22a.html)

---

# B. Informed, guided, semantic, and learned masking

## B1. AttMask / “What to Hide from Your Students”

**Title:** What to Hide from Your Students: Attention-Guided Masked Image Modeling  
**Authors:** Ioannis Kakogeorgiou; Spyros Gidaris; Bill Psomas; Yannis Avrithis; Andrei Bursuc; Konstantinos Karantzalos; Nikos Komodakis  
**Venue/year:** European Conference on Computer Vision (ECCV), 2022, pp. 300–318  
**Identifiers:** DOI: `10.1007/978-3-031-20056-4_18`; arXiv: `2203.12719`

AttMask uses teacher self-attention to hide highly attended patches from a student. On 20% ImageNet, AttMask-High reached 49.7 k-NN and 57.9 linear accuracy, versus 47.8/56.7 for equal-ratio random patch masking and 46.7/56.4 for default block-wise masking. However, in its DINO-only ablation, random masking scored 43.4 k-NN and AttMask-High 43.5, a negligible 0.1-point difference; CIFAR-10 fine-tuning also tied the iBOT baseline at 98.8%.

**Relation:** Supports attention guidance in some settings but also supplies direct evidence that informed masking can be practically indistinguishable from random.

**Verification:** [ECCV/arXiv](https://arxiv.org/abs/2203.12719), [official repository](https://github.com/gkakogeorgiou/attmask)

## B2. SemMAE

**Title:** SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders  
**Authors:** Gang Li; Heliang Zheng; Daqing Liu; Chaoyue Wang; Bing Su; Changwen Zheng  
**Venue/year:** Advances in Neural Information Processing Systems 35, 2022, pp. 14290–14302  
**Identifiers:** DOI: `10.52202/068431-1039`; arXiv: `2206.10207`

SemMAE learns coarse semantic parts from self-supervised ViT attention and gradually changes from within-part patch masking to whole-part masking. Its best adaptive strategy improved linear probing from 66.8% for random masking to 68.7%, but masking 75% of patches within each semantic part scored 66.5%—slightly below random—and masking 75% of entire semantic parts collapsed to 52.9%. Moreover, raw iBOT-derived semantic parts scored 63.6% versus 63.7% without semantic parts; only the separately refined parts improved performance to 65.0%.

**Relation:** The most direct published precedent for the claim that semantic guidance is not automatically useful: naïve or imperfect semantic masks can match or severely underperform random masking.

**Verification:** [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2022/hash/5c186016d0844767209dc36e9e61441b-Abstract-Conference.html)

## B3. AutoMAE

**Title:** Improving Masked Autoencoders by Learning Where to Mask  
**Authors:** Haijian Chen; Wendong Zhang; Yunbo Wang; Xiaokang Yang  
**Venue/year:** Chinese Conference on Pattern Recognition and Computer Vision (PRCV), 2023, pp. 377–390  
**Identifiers:** DOI: `10.1007/978-981-99-8543-2_31`; arXiv: `2303.06583`

AutoMAE jointly trains an adversarial differentiable mask generator with masked reconstruction. It clearly improves linear probing—66.7% with an MAE-initialized generator versus 63.7% for MAE—but full-data ImageNet fine-tuning is effectively tied: 83.32% AutoMAE, 83.34% SemMAE, and 83.26% MAE.

**Relation:** Supports learned masking for representation separability and low-data transfer, while showing almost no practical advantage over random MAE under full-data fine-tuning.

**Verification:** [Springer/Crossref metadata](https://api.crossref.org/works?query.title=Improving%20masked%20autoencoders%20by%20learning%20where%20to%20mask&rows=3), [arXiv](https://arxiv.org/abs/2303.06583)

## B4. Hard Patches Mining

**Title:** Hard Patches Mining for Masked Image Modeling  
**Authors:** Haochen Wang; Kaiyou Song; Junsong Fan; Yuxi Wang; Jin Xie; Zhaoxiang Zhang  
**Venue/year:** CVPR, 2023, pp. 10375–10385  
**Identifiers:** DOI: `10.1109/CVPR52729.2023.01000`; arXiv: `2304.05919`

HPM predicts patch-wise reconstruction loss and progressively biases masking toward difficult patches. Moderate easy-to-hard guidance improved fine-tuning from 82.49% for random masking to 82.95%, but always choosing the hardest patches reduced accuracy to 81.40%; hard-to-easy scheduling scored 81.71%, and selecting easy patches scored 82.36%. The authors explicitly conclude that harder tasks do not consistently improve performance and that retaining randomness is beneficial.

**Relation:** Strong evidence that “more informed” is not monotonically better; guidance must remain mixed with random exploration.

**Verification:** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Hard_Patches_Mining_for_Masked_Image_Modeling_CVPR_2023_paper.html)

## B5. MST

**Title:** MST: Masked Self-Supervised Transformer for Visual Representation  
**Authors:** Zhaowen Li; Zhiyang Chen; Fan Yang; Wei Li; Yousong Zhu; Chaoyang Zhao; Rui Deng; Liwei Wu; Rui Zhao; Ming Tang; Jinqiao Wang  
**Venue/year:** Advances in Neural Information Processing Systems 34, 2021, pp. 13165–13176  
**Identifiers:** arXiv: `2106.05656`

MST masks low-attention patches so that important semantic structure remains visible while a decoder reconstructs spatial information. In its mask-strategy ablation, no masking scored 73.1%, random masking degraded to 63.2%, and attention-guided masking reached 73.7%.

**Relation:** Shows a large failure of unconstrained random masking for this particular contrastive/restoration objective, but the informed policy improved only 0.6 points over no masking.

**Verification:** [NeurIPS/DBLP](https://dblp.org/rec/conf/nips/LiCYLZZDWZTW21), [arXiv](https://arxiv.org/abs/2106.05656)

## B6. ADIOS

**Title:** Adversarial Masking for Self-Supervised Learning  
**Authors:** Yuge Shi; N. Siddharth; Philip H. S. Torr; Adam R. Kosiorek  
**Venue/year:** Proceedings of the 39th International Conference on Machine Learning, PMLR 162, 2022, pp. 20026–20040  
**Identifiers:** arXiv: `2201.13100`

ADIOS jointly learns an occlusion model that maximizes and an encoder that minimizes representation disagreement. Improvements vary substantially: with a ResNet-18 on ImageNet100-S, linear accuracy rose from 55.1 to 55.9 for SimCLR, 59.5 to 60.4 for SimSiam, and 56.3 to 61.4 for BYOL. The first two gains are small, while the BYOL gain is large.

**Relation:** Supports learned adversarial masks but illustrates strong interaction between the mask generator and the underlying SSL objective.

**Verification:** [PMLR](https://proceedings.mlr.press/v162/shi22d.html)

## B7. Attention-Guided Masked Autoencoders

**Title:** Attention-Guided Masked Autoencoders for Learning Image Representations  
**Authors:** Leon Sick; Dominik Engel; Pedro Hermosilla; Timo Ropinski  
**Venue/year:** Winter Conference on Applications of Computer Vision (WACV), 2025, pp. 836–846  
**Identifiers:** DOI: `10.1109/WACV61041.2025.00091`; arXiv: `2402.15172`

This paper retains MAE’s random input masks but reweights the reconstruction loss using an unsupervised object-discovery map. It improved ImageNet linear accuracy from 73.5 to 74.4 after 800 epochs and from 75.1 to 75.9 after 1,600 epochs. However, 10%-label fine-tuning decreased from 78.5 to 78.1, and the authors explicitly state that their method does not outperform vanilla MAE under standard fine-tuning; directly applying the attention map to input masking was also substantially worse than loss guidance.

**Relation:** Particularly relevant counter-evidence: semantic information may help as soft loss weighting while harming or failing to help when used directly to choose masks.

**Verification:** [CVF Open Access](https://openaccess.thecvf.com/content/WACV2025/html/Sick_Attention-Guided_Masked_Autoencoders_for_Learning_Image_Representations_WACV_2025_paper.html)

---

# C. Counter-evidence summary

## Direct answer

The literature supports the following characterization:

> **Known but contested:** random masking is a very strong baseline, and informed masking does not reliably win. Sophisticated guidance helps in some objectives and protocols, but published ablations repeatedly show ties, marginal gains, or substantial regressions when semantic guidance is coarse, overly hard, poorly matched to the objective, or applied directly to mask selection.

| Study | Random/simple baseline | Informed/structured result | Interpretation |
|---|---:|---:|---|
| MAE | 84.9 fine-tune / 73.5 linear | Block-75: 82.8 / 63.9; grid-75: 84.0 / 66.0 | Random decisively better |
| SimMIM | 83.0 top-1 | Best block: 82.7; square: 82.6 | Random modestly better |
| SemMAE | Random: 66.8 linear | Within-part: 66.5; whole-part: 52.9; adaptive: 68.7 | Naïve semantics hurt; carefully scheduled semantics help |
| SemMAE part-quality ablation | No parts: 63.7 | raw iBOT parts: 63.6; refined parts: 65.0 | Imperfect semantic model adds no value |
| AutoMAE | MAE: 83.26 full fine-tune | AutoMAE: 83.32 | Effectively tied under full fine-tuning |
| HPM | Random: 82.49 | moderate guidance: 82.95; hardest-only: 81.40 | Small benefit only with retained randomness |
| AttMask/DINO | Random: 43.4 k-NN | AttMask-High: 43.5; Hint: 43.6 | Nearly indistinguishable |
| AttG-MAE | MAE: 78.5 at 10% labels | Guided: 78.1 | Guidance loses under this protocol |
| SSiT/Messidor-2 | No saliency masking: 79.48 κ | 25% saliency mask: 77.53; 50%: 79.97 | Dataset-dependent and non-monotonic |

### Novelty assessment

1. **Not novel in general:** It is already known that random masking can equal or outperform block, semantic, attention-guided, or overly hard masking.
2. **Still contested:** SemMAE, AttMask, AutoMAE, ADIOS, HPM, medical MSMAE, and anatomically guided aneurysm pretraining report gains under some settings.
3. **Likely novel in the paper’s exact form:** I found no verified controlled study that holds everything fixed in 3D retinal OCT I-JEPA and shows that a cheap intensity-derived retina prior outperforms targets derived from a trained segmentation/foundation model.
4. **Recommended claim wording:** “Our result does not establish that anatomical masking is generally ineffective. Instead, it shows that, under controlled 3D OCT joint-embedding pretraining, most benefit came from coarse retina localization; additional segmentation-model complexity did not translate into a statistically detectable downstream gain.”

---

# D. Medical imaging SSL and retinal OCT

## D1. RETFound

**Title:** A Foundation Model for Generalizable Disease Detection from Retinal Images  
**Authors:** Yukun Zhou; Mark A. Chia; Siegfried K. Wagner; Murat S. Ayhan; Dominic J. Williamson; Robbert R. Struyven; Timing Liu; Moucheng Xu; Mateo G. Lozano; Peter Woodward-Court; Yuka Kihara; UK Biobank Eye & Vision Consortium; Andre Altmann; Aaron Y. Lee; Eric J. Topol; Alastair K. Denniston; Daniel C. Alexander; Pearse A. Keane  
**Venue/year:** Nature 622(7981), 156–163, 2023  
**Identifier:** DOI: `10.1038/s41586-023-06555-x`

RETFound uses masked-autoencoder pretraining on 1.6 million unlabeled retinal images and adapts the learned encoder to ocular and systemic disease prediction. The paper reports better label efficiency and generalization than comparison models across several tasks.

**Relation:** Principal retinal foundation-model baseline. Its use of random MAE masking contrasts with the paper’s controlled anatomy-guided target policies.

**Verification:** [Nature/Crossref](https://api.crossref.org/works/10.1038/s41586-023-06555-x), [Europe PMC author record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:37704728&resulttype=core&format=json)

## D2. MIRAGE

**Title:** Multimodal Foundation Model and Benchmark for Comprehensive Retinal OCT Image Analysis  
**Authors:** José Morano; Botond Fazekas; Emese Sükei; Ronald Fecso; Taha Emre; Markus Gumpinger; Georg Faustmann; Marzieh Oghbaie; Ursula Schmidt-Erfurth; Hrvoje Bogunović  
**Venue/year:** npj Digital Medicine 8(1), article 576, 2025  
**Identifiers:** DOI: `10.1038/s41746-025-01852-3`; arXiv: `2506.08900`

MIRAGE uses MultiMAE-style multimodal pretraining on paired OCT, SLO, and automatically generated retinal-layer pseudo-labels. Adding layer pseudo-labels to OCT pretraining improved average AUROC from 93.75 to 95.44 and average segmentation Dice from 66.37 to 69.07, but did not improve every dataset: OCT-only was better on OLIVES AUROC and OPTIMA9C AUROC. The paper also states that OCT pretraining/classification used central 2D B-scans rather than complete 3D volumes.

**Relation:** Supplies the segmentation/foundation model used for anatomy guidance and positive evidence that retinal-layer pseudo-labels can help on average. The per-dataset reversals and 2D limitation support testing whether such sophistication transfers to 3D JEPA masking.

**Verification:** [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41746-025-01852-3&resulttype=core&format=json), [official repository](https://github.com/j-morano/MIRAGE)

## D3. Uni4Eye++

**Title:** Uni4Eye++: A General Masked Image Modeling Multi-Modal Pre-Training Framework for Ophthalmic Image Classification and Segmentation  
**Authors:** Zhiyuan Cai; Li Lin; Huaqing He; Pujin Cheng; Xiaoying Tang  
**Venue/year:** IEEE Transactions on Medical Imaging 43(12), 4419–4429, 2024  
**Identifier:** DOI: `10.1109/TMI.2024.3422102`

Uni4Eye++ jointly handles 2D and 3D ophthalmic inputs through unified masked image modeling and uses multiple reconstruction targets. It evaluates transfer to ophthalmic classification and segmentation tasks.

**Relation:** Establishes masked multimodal ophthalmic pretraining as a close domain baseline, though it does not isolate anatomical mask selection.

**Verification:** [Crossref](https://api.crossref.org/works/10.1109/TMI.2024.3422102)

## D4. SSiT

**Title:** SSiT: Saliency-Guided Self-Supervised Image Transformer for Diabetic Retinopathy Grading  
**Authors:** Yijin Huang; Junyan Lyu; Pujin Cheng; Roger Tam; Xiaoying Tang  
**Venue/year:** IEEE Journal of Biomedical and Health Informatics 28(5), 2806–2817, 2024  
**Identifiers:** DOI: `10.1109/JBHI.2024.3362878`; arXiv: `2210.10969`

SSiT removes low-saliency fundus patches from a momentum encoder and adds saliency-map prediction. On DDR, 25% saliency masking improved κ from 79.98 to 81.88; on APTOS it improved 92.28 to 92.97. On Messidor-2, however, 25% saliency masking reduced κ from 79.48 to 77.53, while 50% masking reached 79.97; 75% masking hurt all three datasets.

**Relation:** Medical-domain evidence that cheap image saliency can help, but effects are dataset-dependent and informed masking may fail.

**Verification:** [Crossref](https://api.crossref.org/works/10.1109/JBHI.2024.3362878), [arXiv](https://arxiv.org/abs/2210.10969)

## D5. Medical Supervised Masked Autoencoder

**Title:** Medical Supervised Masked Autoencoder: Crafting a Better Masking Strategy and Efficient Fine-Tuning Schedule for Medical Image Classification  
**Authors:** Jiawei Mao; Shujian Guo; Xuesong Yin; Yuanqi Chang; Binling Nie; Yigang Wang  
**Venue/year:** Applied Soft Computing 169, article 112536, 2025  
**Identifiers:** DOI: `10.1016/j.asoc.2024.112536`; arXiv: `2305.05871`

MSMAE uses supervised attention maps to mask lesion-associated regions during pretraining and fine-tuning. On Messidor-2, random-mask MAE scored 60.54%, generic attention masking 60.92%, and supervised attention pretraining 62.07%; adding supervised masked fine-tuning raised the final result to 63.41%.

**Relation:** Supports medical attention guidance, but its supervision, masking, and fine-tuning schedule change together, so it is not a clean mask-policy-only comparison.

**Verification:** [Crossref](https://api.crossref.org/works/10.1016/j.asoc.2024.112536), [official repository](https://github.com/Talented-Q/MSMAE)

## D6. Anatomically guided aneurysm MAE

**Title:** Anatomically-Guided Masked Autoencoder Pre-Training for Aneurysm Detection  
**Authors:** Alberto M. Ceballos Arroyo; Jisoo Kim; Chu-Hsuan Lin; Lei Qin; Geoffrey S. Young; Huaizu Jiang  
**Venue/year:** IEEE/CVF Winter Conference on Applications of Computer Vision, 2026, pp. 5693–5694  
**Identifiers:** DOI: `10.1109/WACV61042.2026.00552`; arXiv: `2502.21244`

This method prioritizes patches near cerebral arteries and reconstructs both CT intensity and artery-distance maps. Its full artery-informed system obtained sensitivities of 92.9, 96.0, 92.0, and 78.3 across four test sets, versus 78.6, 79.2, 72.6, and 53.4 for vanilla MAE. However, the method jointly changes crop sampling, mask sampling, reconstruction targets, and input channels.

**Relation:** Positive evidence for anatomical guidance in 3D CT, but not a mask-only ablation; it therefore does not contradict a controlled finding that segmentation-guided masking alone offers no benefit.

**Verification:** [Crossref](https://api.crossref.org/works/10.1109/WACV61042.2026.00552), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13335401/)

## D7. AnatPaste

**Title:** Anatomy-Aware Self-Supervised Learning for Anomaly Detection in Chest Radiographs  
**Authors:** Junya Sato; Yuki Suzuki; Tomohiro Wataya; Daiki Nishigaki; Kosuke Kita; Kazuki Yamagata; Noriyuki Tomiyama; Shoji Kido  
**Venue/year:** iScience 26(7), article 107086, 2023  
**Identifier:** DOI: `10.1016/j.isci.2023.107086`

AnatPaste uses threshold-based lung segmentation to constrain synthetic anomaly placement to anatomically plausible lung regions. It improved AUC over ordinary CutPaste by 14.6, 18.0, and 11.3 points across ZhangLab, CheXpert, and RSNA datasets.

**Relation:** Supports coarse anatomical localization, but it uses segmentation to constrain augmentation rather than to choose JEPA targets. Its success is compatible with the hypothesis that coarse localization supplies most of the useful prior.

**Verification:** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10331430/), [Crossref](https://api.crossref.org/works/10.1016/j.isci.2023.107086)

## D8. Multimodal retinal SSL

**Title:** Self-Supervised Feature Learning via Exploiting Multi-Modal Data for Retinal Disease Diagnosis  
**Authors:** Xiaomeng Li; Mengyu Jia; Md Tauhidul Islam; Lequan Yu; Lei Xing  
**Venue/year:** IEEE Transactions on Medical Imaging 39(12), 4023–4033, 2020  
**Identifier:** DOI: `10.1109/TMI.2020.3008871`

This work learns retinal representations by exploiting complementary multimodal imaging information and transfers them to disease diagnosis. It predates current retinal foundation models and demonstrates the value of ophthalmic cross-modal supervision.

**Relation:** Provides medical SSL context but does not evaluate target-mask policies.

**Verification:** [Crossref](https://api.crossref.org/works/10.1109/TMI.2020.3008871)

---

# E. Evaluation methodology

## E1. DeLong test

**Title:** Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach  
**Authors:** Elizabeth R. DeLong; David M. DeLong; Daniel L. Clarke-Pearson  
**Venue/year:** Biometrics 44(3), 837–845, 1988  
**Identifier:** DOI: `10.2307/2531595`

DeLong et al. give a nonparametric covariance estimator and test for comparing correlated ROC-AUCs. It is appropriate when models are evaluated on the same independent subjects.

**Relation:** Provides the standard paired AUC test for comparing masking policies on a shared test cohort.

**Verification:** [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:3203132&format=json)

## E2. Clustered ROC analysis

**Title:** Nonparametric Analysis of Clustered ROC Curve Data  
**Author:** Nancy A. Obuchowski  
**Venue/year:** Biometrics 53(2), 567–578, 1997  
**Identifier:** DOI: `10.2307/2533958`

Obuchowski extends structural-component ROC methods to clustered diagnostic data. This matters when a patient contributes multiple eyes, visits, volumes, or slices.

**Relation:** Warns that ordinary DeLong inference is not sufficient if OCT observations remain clustered within patients.

**Verification:** [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:9192452&format=json)

## E3. Patient-cluster bootstrap

**Title:** Bootstrap Estimation of Diagnostic Accuracy with Patient-Clustered Data  
**Author:** Carolyn M. Rutter  
**Venue/year:** Academic Radiology 7(6), 413–419, 2000  
**Identifier:** DOI: `10.1016/S1076-6332(00)80381-5`

Rutter describes bootstrap inference for sensitivity, specificity, and ROC area when observations are clustered within patients. Resampling patients preserves within-patient correlation.

**Relation:** Direct basis for patient-level bootstrap confidence intervals for OCT AUC.

**Verification:** [Crossref](https://api.crossref.org/works/10.1016/S1076-6332(00)80381-5)

## E4. Metrics Reloaded

**Title:** Metrics Reloaded: Recommendations for Image Analysis Validation  
**Authors:** Lena Maier-Hein; Annika Reinke; Patrick Godau; Minu D. Tizabi; Florian Buettner; Evangelia Christodoulou; Ben Glocker; Fabian Isensee; Jens Kleesiek; Michal Kozubek; Mauricio Reyes; Michael A. Riegler; Manuel Wiesenfarth; A. Emre Kavur; Carole H. Sudre; Michael Baumgartner; Matthias Eisenmann; Doreen Heckmann-Nötzel; Tim Rädsch; Laura Acion; Michela Antonelli; Tal Arbel; Spyridon Bakas; Arriel Benis; Matthew B. Blaschko; M. Jorge Cardoso; Veronika Cheplygina; Beth A. Cimini; Gary S. Collins; Keyvan Farahani; Luciana Ferrer; Adrian Galdran; Bram van Ginneken; Robert Haase; Daniel A. Hashimoto; Michael M. Hoffman; Merel Huisman; Pierre Jannin; Charles E. Kahn; Dagmar Kainmueller; Bernhard Kainz; Alexandros Karargyris; Alan Karthikesalingam; Florian Kofler; Annette Kopp-Schneider; Anna Kreshuk; Tahsin Kurc; Bennett A. Landman; Geert Litjens; Amin Madani; Klaus Maier-Hein; Anne L. Martel; Peter Mattson; Erik Meijering; Bjoern Menze; Karel G. M. Moons; Henning Müller; Brennan Nichyporuk; Felix Nickel; Jens Petersen; Nasir Rajpoot; Nicola Rieke; Julio Saez-Rodriguez; Clara I. Sánchez; Shravya Shetty; Maarten van Smeden; Ronald M. Summers; Abdel A. Taha; Aleksei Tiulpin; Sotirios A. Tsaftaris; Ben Van Calster; Gaël Varoquaux; Paul F. Jäger  
**Venue/year:** Nature Methods 21(2), 195–212, 2024  
**Identifiers:** DOI: `10.1038/s41592-023-02151-z`; arXiv: `2206.01653`

Metrics Reloaded gives problem-driven recommendations for selecting and reporting image-analysis metrics and emphasizes matching metrics to the target task, data properties, and failure costs. It discourages relying on a single aggregate number without uncertainty or task-specific justification.

**Relation:** Supports reporting AUC with confidence intervals, paired testing, subgroup results, and clinically interpretable effect sizes.

**Verification:** [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41592-023-02151-z&resulttype=core&format=json)

## E5. Benchmark variance

**Title:** Accounting for Variance in Machine Learning Benchmarks  
**Authors:** Xavier Bouthillier; Pierre Delaunay; Mirko Bronzi; Assya Trofimov; Brennan Nichyporuk; Justin Szeto; Nazanin Mohammadi Sepahvand; Edward Raff; Kanika Madan; Vikram Voleti; Samira Ebrahimi Kahou; Vincent Michalski; Tal Arbel; Chris Pal; Gael Varoquaux; Pascal Vincent  
**Venue/year:** Proceedings of Machine Learning and Systems 3, 747–769, 2021  
**Identifier:** arXiv: `2103.03098`

This paper models variation arising from data sampling, augmentation, initialization, and hyperparameter selection and shows that these sources can materially alter benchmark conclusions. It recommends comparisons that account for more than a single training run or fortunate seed.

**Relation:** Supports treating a single-seed SSL difference as provisional and reporting multiple pretraining/probe seeds when feasible.

**Verification:** [MLSys Proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)

### Recommended evaluation design

- Split and resample at the **patient level**, not at the B-scan, eye, or volume-row level.
- If each patient contributes one independent test unit, use paired DeLong plus patient bootstrap confidence intervals.
- If multiple eyes/visits/volumes per patient are retained, use cluster-aware ROC inference or a patient-cluster bootstrap.
- Report paired AUC differences and confidence intervals, not only separate AUC confidence intervals.
- Separate pretraining-seed, linear-head-seed, and finite-test-sample uncertainty where computationally possible.
- State clearly whether “statistically indistinguishable” means failure to reject, an equivalence test, or a confidence interval contained within a prespecified practical-equivalence margin.

---

# F. Fairness in ophthalmic AI and Harvard-FairVision

## F1. FairVision

**Title:** FairVision: Equitable Deep Learning for Eye Disease Screening via Fair Identity Scaling  
**Authors:** Yan Luo; Muhammad Osama Khan; Yu Tian; Min Shi; Zehao Dou; Tobias Elze; Yi Fang; Mengyu Wang  
**Venue/year:** arXiv preprint, revised 2024  
**Identifier:** arXiv: `2310.02492`

The paper introduces Harvard-FairVision and studies fairness across 2D and 3D ophthalmic models and race, gender, and ethnicity. The dataset contains **30,000 subjects**, divided into 10,000 each for AMD, diabetic retinopathy, and glaucoma. Each subject has one SLO fundus image and one OCT B-scan volume; glaucoma volumes are `200×200×200`, while AMD and DR volumes are `128×200×200`.

Six recorded demographic attributes are age, gender, race, ethnicity, preferred language, and marital status. The dataset materials specify **CC BY-NC-ND 4.0**, non-commercial research use only, and prohibit use for clinical decisions or patient care.

**Relation:** Supplies the downstream glaucoma cohort and requires patient-level evaluation and demographic subgroup auditing.

**Verification:** [arXiv metadata](https://export.arxiv.org/api/query?id_list=2310.02492), [official repository](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision), [dataset card](https://huggingface.co/datasets/harvardairobotics/FairVision)

### Access caveat

The FairVision repository currently links to a Hugging Face resource named **FairGenMed**, whose current card describes a 10,000-subject glaucoma subset, while the separate Harvard-FairVision card describes the full 30,000-subject dataset. The paper should record the exact repository, revision, disease subset, files, and download date used. No peer-reviewed venue for the FairVision paper was verified; cite it as an arXiv preprint.

---

# Venue calibration: NeurIPS 2025 GenAI4Health

The official NeurIPS 2025 program lists **nine oral papers and 89 additional accepted posters**, for 98 accepted works. The call allowed nine pages for research papers and five pages for demonstration or position papers, excluding references and appendices. Accepted papers were explicitly non-archival and received at least two anonymous reviews.

The accepted scope is broad:

- algorithmic contributions and controlled benchmarks;
- synthetic ECG, EEG, MRI, radiograph, and clinical-data generation;
- LLM and multimodal medical reasoning;
- fairness, robustness, safety, and policy;
- demos and position papers;
- careful negative or baseline-centered studies.

Representative rigor levels include:

- **MedVAL:** 840 physician-annotated outputs, six medical tasks, ten language models, a physician-defined error taxonomy, and reported `p < 0.001`; average F1 rose from 66% to 83%.
- **High-Fidelity Synthetic ECG Generation:** PTB-XL evaluation spanning morphology, clinical coherence, privacy, personalization, and downstream utility; a 70% reduction in inter-lead correlation error was reported.
- **Towards Memory-Efficient Foundation Models in Medical Imaging:** experiments on three medical-imaging datasets under non-IID federated conditions.
- **Count-Based Approaches Remain Strong:** comparison of count-based, sequential-transformer, and mixture-of-agents pipelines over four EHR outcomes; simple count models led most AUROC/AUPR comparisons.
- **Statistically Significant Results … Do Not Guarantee Generalizable Results:** an explicitly negative methodological contribution emphasizing evaluator dependence and non-generalizable significance.

**Calibration conclusion:** A controlled 3D OCT masking study with a fixed encoder/probe pipeline, paired patient-level uncertainty, honest negative findings, and a clinically interpretable simplicity result is well aligned with this workshop. The work need not introduce a massive foundation model; a rigorous empirical finding about when domain knowledge does not help is within the demonstrated scope. The strongest version should include repeated seeds or a clear seed limitation, patient-level CIs, paired AUC differences, and exact computational overhead for MIRAGE-guided versus intensity-guided masking.

**Verification:** [Official NeurIPS program](https://neurips.cc/virtual/2025/loc/san-diego/workshop/109566), [workshop site](https://aihealth.ischool.utexas.edu/GenAI4HealthNeurips2025/), [OpenReview group](https://openreview.net/group?id=NeurIPS.cc/2025/Workshop/GenAI4Health)

---

# Recommended related-work positioning

A defensible paper-level statement is:

> Prior work has often treated semantic, attention-based, or learned mask selection as an improvement over random sampling. However, the empirical record is mixed. Random masking outperformed block and grid masks in MAE and block/square masks in SimMIM; SemMAE found that naïve whole-part masking caused a 13.9-point linear-probe drop and that unrefined semantic parts did not improve over the no-parts baseline; AutoMAE’s full-data fine-tuning result was within 0.06 points of MAE; and Hard Patches Mining found that always masking the hardest patches underperformed random masking. Our controlled 3D OCT results extend this evidence to anatomy-guided joint-embedding prediction: coarse retina localization was sufficient, while a trained segmentation model did not produce a detectable advantage and was outperformed by a cheap intensity-derived prior.

Avoid claiming that anatomy-aware masking never works. Published medical results from MSMAE, AnatPaste, MIRAGE, SSiT, and aneurysm MAE show that anatomy can help, especially when it changes supervision, reconstruction targets, augmentation realism, or downstream tuning in addition to mask selection.

---

# UNVERIFIED — do not cite

1. **“Measuring and Mitigating the Random Seed Effect in Self-Supervised Learning,” allegedly CVPR 2024.** I could not verify an authoritative proceedings entry matching this title. Do not cite.
2. **“Organ-aware masked autoencoders for self-supervised medical image analysis,” allegedly arXiv:2405.12909.** The title/identifier combination did not verify. Do not cite.
3. **“Anatomy-Aware Masked Image Modeling for Self-Supervised Learning on 3D Brain MRI.”** An OpenReview record appears to exist, but I did not verify an archival publication venue or final bibliographic status. Do not cite as published work.
4. **FairVision peer-reviewed venue.** The paper and dataset are verified, but no peer-reviewed publication was found; cite only the arXiv preprint.

---

