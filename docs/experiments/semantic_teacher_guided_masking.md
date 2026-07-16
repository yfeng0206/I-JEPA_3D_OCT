# Semantic-Teacher-Guided Target Allocation for I-JEPA

Branch: `vlm-guided-masking`

> This is a living research plan. It defines the experiments, controls, decision gates, and allowed claims for a new direction. Offline Phase 0 map tooling is implemented; training-time semantic target allocation is not.

## 0. Executive decision

The project will test whether a frozen semantic vision model can improve I-JEPA by changing only where its ordinary target blocks are sampled.

The primary comparison is:

```text
random I-JEPA targets
        vs
frozen DINOv3-guided targets
        vs
frozen language-aligned vision-guided targets
```

The default guide is **DINOv3**, not a VLM. A language-aligned guide remains in scope only if it produces better spatial maps and better downstream representations under matched architecture, mask budget, compute accounting, and evaluation.

Primary ImageNet guides:

1. DINOv3 ViT-B/16.
2. SigLIP 2 Base/16-224 vision tower.
3. I-JEPA ep25 endogenous saliency and clustering baselines.
4. A MILAN-style frozen CLIP image-attention baseline.

Primary OCT guides, if the ImageNet direction survives:

1. MIRAGE-Base.
2. RETFound-OCT.
3. DINOv3 as the generic control.
4. MedSigLIP or EyeCLIP as language-aligned controls.
5. RetinaVLM and LO-VLM only as optional pseudo-text/localization studies.

MedGemma is not a primary guide because the proposed experiment does not require language generation. Its image encoder is the relevant component.

## 0.1 Status tracker

Legend: DONE / IN PROGRESS / TODO / STOPPED

| ID | Work item | State | Gate |
|---|---|---|---|
| R0 | Primary-source literature and implementation audit | IN PROGRESS | Papers complete; interfaces, weights, and licenses pending |
| R1 | Create dedicated branch | DONE | `vlm-guided-masking` |
| P0-A | Pin model weights, licenses, transforms, and checkpoints | IN PROGRESS | CLIP/SigLIP tested; DINOv3 access and full legal manifest pending |
| P0-B | Build common semantic-map extraction API | DONE | Lazy DINOv3, SigLIP 2, CLIP, and local I-JEPA adapters |
| P0-C | Produce ImageNet visual atlas and scorecard | IN PROGRESS | Atlas CLI works; labeled ImageNet scorecard pending |
| P0-D | Analyze why OCT ribbon beat random | TODO | Anatomy vs position vs geometry separated |
| P1 | Frozen-guide ImageNet continuation | TODO | Guide beats matched-budget random |
| P2-A | Mask scale, dilation, and predictor interaction | TODO | Robust optimum, not one tuned setting |
| P2-B | Anchored low-LR guide adaptation | TODO | Beats frozen guide without semantic drift |
| P2-C | From-scratch confirmation | TODO | Gain not limited to ep25 curriculum |
| P3-A | OCT guide map benchmark | TODO | External labels show useful localization |
| P3-B | OCT pretraining and transfer | TODO | FairVision plus external OCT gain |

### 0.2 Implemented Phase 0 scaffold

Files:

- `src/guides/base.py` — validated dense-guide tensor contract.
- `src/guides/maps.py` — native/cosine/PCA maps and label-free diagnostics.
- `src/guides/hf_guides.py` — DINOv3, SigLIP 2, and CLIP vision adapters.
- `src/guides/ijepa.py` — local target-encoder map adapter.
- `scripts/semantic_map_atlas.py` — deterministic shared-crop extraction, NPZ/JSON output, and atlas rendering.
- `configs/semantic_maps/phase0_guides.yaml` — model identifiers and defaults.
- `requirements-phase0.txt` — isolated modern sidecar environment.
- `tests/test_semantic_guides.py` — download-free tensor/map tests.

Example:

```powershell
python scripts/semantic_map_atlas.py `
  --inputs "path\to\images\*.png" `
  --guides clip siglip2 `
  --output-dir results\semantic_maps `
  --device cpu
```

Outputs:

- `semantic_map_metrics.json`
- `maps/<image>__<guide>.npz`
- `atlases/<image>.png`

DINOv3 is intentionally not silently substituted when its gated checkpoint is unavailable. The JSON report records the access failure and the experiment remains blocked until approved weights or a verified local path are supplied.

## 1. Research question

The direction asks:

> Can a frozen semantic teacher identify spatial regions that are better I-JEPA prediction targets than uniformly random blocks?

The experiment separates three questions:

1. **Dense semantics:** Are DINOv3 patch features sufficient?
2. **Language-aligned teacher comparison:** Does the selected language-aligned teacher outperform the selected DINOv3 teacher?
3. **Target allocation:** Does converting these maps into matched-budget I-JEPA blocks improve learned representations?

The project does not initially use text generation, language decoding, or language tokens inside I-JEPA.

## 1.1 Working hypotheses

| Hypothesis | Prediction |
|---|---|
| H1: semantic target density | Frozen semantic guides beat random at fixed target/context budget |
| H2: language-aligned teacher comparison | SigLIP 2 empirically beats the selected DINOv3 comparator on map quality and transfer |
| H3: DINO is sufficient | DINOv3 ties or beats the language-aligned guide |
| H4: OCT oracle is anatomical | True retina-following maps beat center, shifted, rolled, and background controls |
| H5: curriculum matters | ep25 switch wins while guided-from-epoch-0 does not |
| H6: guide adaptation helps | Anchored low-LR adaptation beats the frozen guide |
| H7: guide adaptation harms semantics | Frozen guide remains better and maps drift/collapse when unfrozen |

## 2. What this method is and is not

### 2.1 Semantic-guided spatial target allocation

A frozen guide generates a per-image score map:

\[
G(x) \rightarrow F_G \in \mathbb{R}^{B \times H_G \times W_G \times d_G},
\qquad
g_G \in \mathbb{R}^{B \times d_G}.
\]

One common diagnostic score is global-to-patch cosine similarity:

\[
r_i =
\cos\left(
\operatorname{LN}(f_i),
\operatorname{LN}(g_G)
\right).
\]

Rank normalization makes scores comparable across models:

\[
S_i =
\frac{\operatorname{rank}(r_i)-0.5}{H_GW_G}.
\]

This readout is not a valid model-neutral primary map by itself. DINOv3 exposes a CLS/global token in the same transformer stream, while SigLIP 2 obtains its global representation through learned attention pooling and projection. Each guide therefore uses a model-native primary readout, selected on a calibration split and locked before evaluation. Shared readouts such as token PCA, feature norm, and clustering remain cross-guide diagnostics.

The selected score map changes only the probability of sampling target-block locations:

\[
M \sim \pi(T_{\text{sem}}(x)),
\]

\[
\mathcal{L}_{\text{I-JEPA}} =
D\left(
P(E_\theta(x_{\bar M}),M),
\operatorname{sg}(E_\xi(x)_M)
\right).
\]

The semantic teacher features are not prediction targets and are not inputs to the I-JEPA predictor.

### 2.2 Representation alignment or distillation

Alignment is a different method:

\[
\mathcal{L}_{\text{align}} =
D\left(
A(z_{\text{student}}),
\operatorname{sg}(z_{\text{teacher}})
\right).
\]

The guide determines the feature values the model must match.

This is related to, but distinct from, the proposed frozen-guide allocation.

### 2.3 Representation fusion

Fusion supplies frozen features as model inputs:

\[
\hat y = P(x,z_{\text{teacher}}).
\]

This also differs from target allocation.

### 2.4 VLA/vision-language JEPA precedent

Three similarly named works must not be conflated:

- [VL-JEPA](https://arxiv.org/abs/2512.10942) predicts text-semantic embeddings from frozen V-JEPA-2 visual features and a query.
- [VLA-JEPA](https://arxiv.org/abs/2602.10098) uses VLM latent-action states to help predict future frozen V-JEPA-2 world features.
- [JEPA-VLA](https://arxiv.org/abs/2602.11832) injects frozen V-JEPA-2 features into a VLA through projection or gated cross-attention.

These papers support frozen semantic features, representation fusion, and slow semantic adaptation. They do not establish semantic spatial target allocation for image I-JEPA.

Therefore this project must not be described as "mapping VLM features into JEPA space" unless an explicit alignment loss is added in Phase 2.

## 3. Prior art and novelty boundary

The broad idea "use a VLM or CLIP to guide masking" is occupied.

| Work | What it already establishes | Difference from this plan |
|---|---|---|
| [MILAN](https://arxiv.org/abs/2208.06049) | Frozen CLIP image attention guides visible-patch sampling and CLIP patch features become MAE targets | MAE, visible-patch selection, not contiguous I-JEPA target blocks |
| [CMT-MAE](https://arxiv.org/abs/2412.17566) | Frozen CLIP and momentum-student attention guide MAE masks; CLIP/student features are targets | MAE and feature reconstruction, not standard I-JEPA targets |
| [TC-JEPA](https://arxiv.org/abs/2605.03245) | Automatically synthesized captions condition JEPA and learn patch-word relations | Captions condition prediction; they do not choose target-block locations |
| [DSeq-JEPA](https://arxiv.org/abs/2511.17354) | EMA-attention semantic regions and a random-to-informed JEPA curriculum | Endogenous saliency plus sequential prediction, not frozen external guide comparison |
| [Mask What Matters](https://arxiv.org/abs/2509.23054) | Text/VLM/SAM medical ROI localization and differentiated masking | Uses prompts and external localization stack; pixel reconstruction |
| [RetinaVLM](https://arxiv.org/abs/2407.08410) | Generated OCT reports and phrase-specific GradCAM maps | Report/prompt dependent and post-hoc; not I-JEPA target allocation |
| [AttMask](https://arxiv.org/abs/2203.12719) | EMA-teacher attention guides masking | Internal EMA teacher rather than frozen foundation guide |
| [SemMAE](https://arxiv.org/abs/2206.10207) | Semantic parts and part-level masking curriculum | MAE with separately learned semantic parts |

### 3.1 Defensible narrow claim

A future claim may be:

> Prompt-free semantic target-block allocation for I-JEPA uses a frozen vision or vision-language teacher to derive per-image spatial scores, converts those scores into geometry- and budget-controlled contiguous target blocks, and leaves the standard EMA I-JEPA target representation unchanged.

This claim is only defensible if:

- the guide receives no human prompt or downstream label;
- the I-JEPA predictor receives no guide features or text;
- target count, shape, union, overlap, and retained context are matched;
- DINOv3, MILAN-style CLIP, DSeq, clustering, and random baselines are included;
- ImageNet-1K and OCT results are both reported.

Even a matched patch size, hidden width, and parameter count do not isolate the causal effect of language supervision: DINOv3 and SigLIP 2 differ in data, objectives, teacher lineage, optimization, and auxiliary losses. The project may claim that one teacher empirically outperforms the other, not that language alignment alone caused the difference.

### 3.2 Claims that are not allowed

Do not claim:

- first VLM-guided masking;
- first prompt-free CLIP-guided masking;
- first caption-conditioned JEPA;
- first patch-text grounding in JEPA;
- first medical text-guided masking;
- first OCT VLM saliency;
- representation alignment when only mask locations change;
- language-free self-supervision without disclosing the frozen guide's image-text pretraining;
- efficiency without including guide inference cost.

## 4. Common tensor contract

### 4.1 I-JEPA

For \(x \in \mathbb{R}^{B\times3\times H\times W}\) and patch size \(p=16\):

\[
N = (H/p)(W/p).
\]

\[
E_\theta(x) \rightarrow Z \in \mathbb{R}^{B\times N\times768}.
\]

The EMA target encoder is:

\[
\xi_t=m_t\xi_{t-1}+(1-m_t)\theta_t.
\]

The predictor estimates target representations:

\[
\hat H_{\mathcal M}
=P_\psi(E_\theta(x_\mathcal C),\mathcal C,\mathcal M).
\]

\[
\mathcal L_{\text{JEPA}}
=\frac1{|\mathcal M|}
\sum_{i\in\mathcal M}
\operatorname{SmoothL1}(\hat h_i,\operatorname{sg}(h_i)).
\]

ImageNet target grid: 224/16 = 14, so \(N=196\).

Current OCT grid: 256/16 = 16, so \(N=256\).

### 4.2 Guide adapters

| Guide | Primary dense tensor | Notes |
|---|---|---|
| DINOv3 ViT-B/16 | patch tokens, 14×14×768 | Native readout: CLS/global-to-patch affinity or validated dense-feature readout |
| SigLIP 2 Base/16-224 | patch tokens, 14×14×768 | Native readout: pooling-query attention or another validated model-native map; do not compare pooled and raw tokens naively |
| CLIP ViT-B/16 | final patch tokens/attention | Reproduce MILAN-style frozen guide baseline |
| I-JEPA ep25 | 14×14×768 | Endogenous saliency, DSeq-map, and clustering controls |
| MIRAGE-Base | 16×16×768 for 512/32 OCT | Exact spatial grid match to current OCT |
| RETFound-OCT | 14×14×1024 at 224/16 | Requires exposing patch tokens |
| EyeCLIP | coarse CLIP spatial tokens | OCT-supported but no released mask API |
| RetinaVLM | approximately 6×6 ResNet feature map | Phrase/report-conditioned GradCAM |

References:

- [DINOv3](https://arxiv.org/abs/2508.10104)
- [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [MIRAGE](https://arxiv.org/abs/2506.08900)
- [RETFound](https://doi.org/10.1038/s41586-023-06555-x)
- [EyeCLIP](https://arxiv.org/abs/2409.06644)
- [RetinaVLM](https://arxiv.org/abs/2407.08410)

## 5. Matched-budget masking contract

The scientific comparison must not inherit the current variable target-union confound.

For every image and every arm, match:

- number of target blocks;
- target block shape vector;
- unique target-token union;
- pairwise target-overlap signature;
- retained context-token count;
- prediction query count;
- loss normalization.

### 5.1 Paired reference masks and joint sampling

The causal comparison begins by sampling a canonical random I-JEPA reference mask set:

\[
\mathbf R^{\text{ref}} =
(R_1^{\text{ref}},R_2^{\text{ref}},R_3^{\text{ref}},R_4^{\text{ref}}).
\]

The reference determines:

- ordered block-shape vector;
- unique target-union cardinality;
- pairwise overlap signature;
- retained context-token count after context sampling and truncation;
- number of prediction queries.

The guided sampler constructs a feasible set \(\mathcal F(\mathbf R^{\text{ref}})\) of four-window configurations with the same realized statistics. The policy samples a complete four-window configuration, not four independent windows.

For candidate rectangle \(R\):

\[
q(R)=\frac1{|R|}\sum_{i\in R}S_i.
\]

For a candidate four-window configuration \(\mathbf R\):

\[
Q(\mathbf R)=\sum_{j=1}^4 q(R_j).
\]

Guided joint sampling:

\[
\pi(\mathbf R)=
(1-\epsilon)
\frac{\exp(Q(\mathbf R)/\tau)}
{\sum_{\mathbf R'\in\mathcal F(\mathbf R^{\text{ref}})}
\exp(Q(\mathbf R')/\tau)}
+\frac{\epsilon}{|\mathcal F(\mathbf R^{\text{ref}})|},
\]

where every feasible configuration matches the reference geometry, union, overlap, query, and context statistics.

Initial values \(\epsilon=0.2\) and \(\tau=0.15\) are provisional and must be calibrated before locked evaluation.

If no matched guided configuration exists, use logged rejection sampling or fall back to the paired random reference. Never silently relax the invariants.

### 5.2 Budget strata

Phase 0 must first measure the empirical canonical I-JEPA distributions of:

- target union;
- target overlap;
- retained context;
- block shape;
- target/context distance.

Define preregistered budget strata from those distributions. Every causal comparison occurs within a stratum or against a paired reference mask. The plan does not replace canonical I-JEPA with a new 61–66% non-overlapping target regime.

A canonical unconditioned random I-JEPA arm remains in the final table to show whether matched-budget conditioning itself changes performance.

### 5.3 Required telemetry

Log every step:

- block shape and coordinates;
- unique target positions;
- context count;
- pairwise target IoU;
- target-in-guide-region fraction;
- map entropy;
- selected guide score;
- sampling probability;
- fallback frequency;
- guide inference time;
- peak memory.

## 6. Phase 0: map validity and mechanism analysis

Phase 0 produces maps and measurements before any pretraining comparison.

### 6.1 Artifact gate

For every guide:

1. Pin paper, code commit, model identifier, weight SHA256, license, and preprocessing.
2. Record pretraining data and possible ImageNet/OCT overlap.
3. Verify exact input resolution and normalization.
4. Run a deterministic forward test.
5. Save guide outputs and map checksums for a small public fixture set.

For I-JEPA:

1. Verify the ImageNet ep25 checkpoint contains encoder, target encoder, predictor, optimizer, scaler, and scheduler state.
2. Replay a known random-mask batch and compare loss to the original log.
3. If no complete checkpoint exists, train one canonical random prefix and fan out from it.

### 6.2 ImageNet semantic-map benchmark

Data:

1. 10,000 stratified unlabeled ImageNet training images for development.
2. ImageNet-S `train-semi` labels for cross-validated guide/readout calibration and Phase 1 dense-proxy gates.
3. A fixed ImageNet training-derived probe development split for Phase 1 linear-probe gates.
4. Locked ImageNet validation for one-shot final classification evaluation after method/arm lock.
5. Locked ImageNet-S validation for one-shot final map and dense evaluation after method/arm lock.
6. ILSVRC localization boxes for weak localization.
7. Optional COCO/ADE20K confirmation without retuning extraction.

[ImageNet-S](https://arxiv.org/abs/2106.03149) is the primary labeled map benchmark.

### 6.3 Map extraction methods

Primary readouts are model native:

1. DINOv3: validated CLS/global affinity or another documented dense readout.
2. SigLIP 2: pooling-query attention or another validated vision-tower readout.
3. CLIP: final-layer attention/importance matching the MILAN policy.
4. I-JEPA: endogenous mean-token affinity, attention proxy, or clustering baseline.

Cross-guide diagnostics:

5. Token PCA with sign oriented on calibration data.
6. Feature norm and novelty.
7. Spherical k-means.
8. TokenCut/NCut-style affinity partition.
9. DSeq-style saliency and connected components.

Evaluate guide × readout interactions on ImageNet-S `train-semi` cross-validation. Lock one primary readout per guide before any ImageNet-S validation or locked downstream evaluation.

### 6.4 Label-free map diagnostics

These measure stability and degeneracy, not semantic correctness.

Equivariance for geometric transform \(a\) and inverse warp \(W_a\):

\[
Q_{\text{equiv}}
=
\mathbb E_a
\left[
\rho(S(a(x)),W_aS(x))
\right].
\]

Top-\(q\) set stability:

\[
Q_{\text{top}}(q)
=
\mathbb E_a
\frac{|T_q(S(a(x)))\cap W_aT_q(S(x))|}
{|T_q(S(a(x)))\cup W_aT_q(S(x))|}.
\]

Evaluate \(q\in\{0.1,0.25,0.5\}\).

Effective support:

\[
p_i=\operatorname{softmax}(S_i/\tau),
\qquad
A_{\text{eff}}
=\frac{\exp(-\sum_i p_i\log p_i)}{N}.
\]

Additional diagnostics:

- patch correspondence cycle consistency;
- cluster ARI/NMI across augmentations;
- connected-component count and size distribution;
- normalized total variation;
- Moran's \(I\);
- image-edge and map-boundary alignment;
- sensitivity to JPEG, blur, grayscale, and background replacement;
- center-of-mass bias;
- border occupancy.

### 6.5 Labeled map metrics

For foreground/semantic mask \(Y\):

- patch/pixel foreground AP;
- AUROC;
- calibrated IoU and Dice;
- boundary F1;
- pointing-game accuracy;
- CorLoc or MaxBoxAcc;
- multi-object coverage;
- foreground/background feature-separation margin;
- cluster purity, NMI, and Hungarian-matched mIoU.

Report all component metrics; a composite score may be used for screening but cannot replace them.

### 6.6 Visual atlas

For each guide and image, render:

1. original image;
2. token-PCA RGB;
3. global-to-patch cosine;
4. native-attention map;
5. top 10/25/50% masks;
6. connected components;
7. ground-truth overlay;
8. DINO/VLM difference map;
9. sampled matched-budget target rectangles;
10. metric panel.

Stratify examples by:

- object size;
- multiple objects;
- clutter;
- texture;
- occlusion;
- center bias;
- guide disagreement;
- high stability but low semantic accuracy.

The manual review requested by the project owner will use this atlas. Development metrics determine model selection; locked metrics are read once after the method and final arms are fixed.

### 6.7 DINO versus language-aligned guide gate

SigLIP 2 remains only if:

- it improves cross-validated ImageNet-S `train-semi` foreground AP/mIoU over DINOv3 by a preregistered meaningful margin;
- equivariance and top-set stability are not materially worse;
- the advantage survives map extraction and spatial-resolution controls.

Provisional screening values:

- +0.02 foreground AP or +0.015 mIoU;
- stability degradation no worse than 0.02.

These values are planning thresholds and must be recalibrated using development variance before final use.

If DINOv3 ties or wins, drop the VLM hypothesis and continue as dense-semantic-teacher target allocation.

### 6.8 Why did the OCT retinal ribbon win?

Existing evidence is hypothesis-generating:

- ribbon masking improved frozen MeanPool by +0.0099, +0.0113, and +0.0109 at ep50/75/100;
- fine-tuned MeanPool improved +0.0079;
- raw JEPA loss was not consistently higher;
- representation diversity remained healthy.

Therefore "the ribbon won because it was consistently harder" is not supported. Oracle validation loss was higher at some checkpoints, so the plan must report the full trajectory rather than summarize it as uniformly easier or harder.

Competing explanations:

| Hypothesis | Prediction |
|---|---|
| task-relevant supervision density | true retinal band beats equal-area shifted/background maps |
| center/position shortcut | fixed center stripe matches retina-following ribbon |
| cleaner/easier targets | lower gradient variance and better transfer |
| anatomy-rich representation | stronger layer/lesion probes and regional effective rank |
| warm-start specialization | ep25 switch wins; guided-from-epoch-0 does not |
| geometry-only effect | rolled or shifted maps retain the gain |

The historical Test split informed the ribbon design and has been evaluated repeatedly. It is Legacy-Development, not confirmatory evidence.

Retrospective checkpoint analysis:

1. Reconstruct target-coverage distributions.
2. Obtain independent retinal/layer masks for at least 250 blinded, stratified slices that were not used to design the ribbon.
3. Measure target overlap with retina, RNFL, vitreous, and choroid.
4. Compute by region:
   - per-token JEPA error;
   - feature variance and effective rank;
   - feature norm and covariance spectrum;
   - patch-label probe performance;
   - cross-checkpoint CKA.
5. Estimate encoder gradient signal-to-noise:

\[
G_{\text{SNR}}
=
\frac{\|\mathbb E[g]\|_2^2}
{\mathbb E[\|g-\mathbb E[g]\|_2^2]+\epsilon}.
\]

Causal controls from the same ep25 checkpoint:

- random matched-budget;
- intensity-localized retina-following ribbon;
- fixed center stripe;
- ribbon shifted up/down;
- spatially rolled ribbon;
- anti-retina/background;
- image-independent geometry-matched positions.

Screen for 10 epochs; extend only random, true ribbon, and the strongest shortcut control.

If center/shifted controls match the true ribbon, reframe the OCT result as positional or geometric curriculum rather than anatomy-guided masking.

## 7. Phase 1: frozen-guide ImageNet continuation

### 7.1 Primary arms

| Arm | Score map | Purpose |
|---|---|---|
| Random matched-budget | uniform feasible-window scores | primary causal control |
| DSeq-map | endogenous I-JEPA saliency/components | nearest JEPA where-to-predict baseline |
| Endogenous attention | EMA-target/global-token or validated attention proxy | AttMask-style internal saliency control |
| Clustering | ep25 I-JEPA clusters ranked by observed error | endogenous semantic/difficulty baseline |
| DINOv3 | frozen dense guide | primary generic semantic teacher |
| SigLIP 2 | frozen language-aligned vision tower | empirical teacher comparison; does not isolate language supervision |
| MILAN attention-policy ablation | frozen CLIP attention | direct prompt-free CLIP masking-policy precedent; not a full MILAN reproduction |
| Low-level controls | center/radial, edge, gradient, entropy | tests whether semantics beat positional and low-level heuristics |

Faithful DSeq-JEPA is not equivalent to a DSeq score map. It changes prediction order and model behavior; a faithful implementation belongs in the final comparison if the project advances.

### 7.2 DSeq-map approximation

The current I-JEPA has no output CLS token. Use a parameter-free query:

\[
q=\frac1N\sum_iF_i^{(10)}.
\]

\[
A_i=\frac{q^\top F_i^{(10)}}{\sqrt d}.
\]

Normalize \(A\), apply Otsu thresholding and connected components, rank regions by mean score, then convert the ranks to \(S\).

This is named `DSeq-map`; it must not be presented as faithful DSeq-JEPA.

### 7.3 Cluster baseline

\[
c_i=\arg\max_k \hat f_i^\top\hat\mu_k.
\]

This arm is a hardness control, not a pure semantic baseline.

Collect error labels through periodic uniform audit masks, independent of the cluster-guided policy. For audit target tokens:

\[
e_i=\frac1d\|\hat h_i-h_i\|_2^2.
\]

Update cluster score only from the audit stream:

\[
\ell_k
\leftarrow
\alpha\ell_k
+(1-\alpha)
\operatorname{mean}_{i:c_i=k} e_i.
\]

\[
S_i=\operatorname{rank}(\ell_{c_i}).
\]

Select \(K\) from `{4,6,8}` using map stability and effective occupancy, not downstream labels. Log audit propensities and cluster observation counts; never let selected clusters exclusively determine their own score.

### 7.4 Training schedule

1. Start every arm from the same complete random I-JEPA ep25 checkpoint.
2. Continue ep25-to-50 with identical optimizer, scheduler, data order, augmentations, target shapes, and budget.
3. Keep external guides frozen and in inference mode.
4. The guide receives the exact augmented crop with its own normalization.
5. Extend passing arms to ep75 and ep100.
6. Save ep50/75/100 checkpoints and complete guide/compute telemetry.

Cached full-image maps are prohibited unless the exact random crop and geometric transformation can be applied without interpolation error.

### 7.5 Compute controls

- load vision towers only;
- record guide FLOPs, memory, and step time;
- run a guide-forward-but-ignore-map control on a diagnostic subset;
- report update-matched and GPU-hour-matched results;
- do not claim efficiency if teacher overhead removes the convergence gain.

### 7.6 Evaluation

Representation health:

- JEPA train/validation loss;
- predictor-target cosine;
- token standard deviation;
- covariance off-diagonal energy;
- effective rank;
- alignment and uniformity;
- patch/image feature norms;
- map entropy and target coverage.

ImageNet:

- frozen kNN;
- frozen linear probe on the development probe split for screening;
- 1% and 10% label probes;
- full fine-tune for finalists;
- ImageNet-A/R/Sketch/V2 robustness;
- optional iNaturalist, CUB, and Cars transfer.

Dense:

- ImageNet-S `train-semi` cross-validated linear segmentation mIoU for screening;
- ADE20K frozen/linear decoder;
- COCO detection/segmentation for finalists.

Locked final endpoints after method/arm lock:

- ImageNet validation frozen linear-probe top-1;
- ImageNet-S validation linear segmentation mIoU;
- robustness and transfer results for finalists.

Statistics:

- paired training/downstream seeds;
- patient/image bootstrap where appropriate;
- hierarchical seed/image uncertainty;
- multiplicity adjustment across confirmatory arms;
- fixed checkpoints, not best-checkpoint selection.

### 7.7 Phase 1 gates

Extend an arm past ep50 only if:

- no collapse or invariant violation;
- meaningful positive movement on the development linear-probe or ImageNet-S `train-semi` cross-validated mIoU;
- acceptable teacher overhead.

The final frozen-guide winner must:

- have a one-shot locked confidence interval above random on at least one final endpoint;
- avoid meaningful regression on the other locked endpoint;
- remain positive across independent seeds;
- pass shuffled, shifted, inverse, and random-weight controls.

Stop frozen semantic guidance if the best arm's upper confidence bound excludes a practically meaningful gain.

## 8. Phase 2: scale, co-adaptation, and from-scratch training

### 8.1 Semantic-region scale

For thresholded semantic region \(R\):

\[
A_R=|R|,
\qquad
d_R=2\sqrt{A_R/\pi}.
\]

Define target-to-region area ratio:

\[
\gamma=\frac{A_M}{A_R}.
\]

Define relative dilation:

\[
\delta=\frac{r_{\text{dilation}}}{d_R}.
\]

Test:

- erosion/dilation `{-2,-1,0,+1,+2}` patches;
- \(\gamma \in \{0.5,1.0,2.0\}\);
- four small regions versus one large region;
- interior versus boundary ring versus interior+context;
- semantic core plus random exploration.

This tests the hypothesis that the target window should be larger than the localized semantic core.

### 8.2 Predictor capacity interaction

The transformer predictor already has global attention; do not call this convolutional receptive-field scaling.

| Predictor | Depth | Width |
|---|---:|---:|
| Small | 3 | 192 |
| Baseline | 6 | 384 |
| Large | 9 | 384 |

Track:

\[
\kappa=\frac{K_{\text{context}}}{K_{\text{target}}}.
\]

This predictor-capacity study runs from scratch. It does not load the Phase 1 ep25 predictor because changing depth/width would introduce predictor-restart and weight-transfer confounds.

Use successive halving:

1. all combinations to ep40;
2. top four to ep60;
3. top two to ep100.

### 8.3 Why ordinary unfreezing does not work

Discrete top-k/window sampling provides no gradient from I-JEPA loss to the guide.

Low-LR unfreezing therefore requires a separate differentiable objective. Without one, the guide will not adapt.

### 8.4 Anchored guide adaptation

Train a saliency head:

\[
s_i=w^\top G_\phi(x)_i.
\]

Use observed JEPA errors for pairwise ranking:

\[
\mathcal L_{\text{rank}}
=
\sum_{i,j}
\omega_i\omega_j
\log\left(
1+\exp[-y_{ij}(s_i-s_j)]
\right),
\]

\[
y_{ij}=\operatorname{sign}(e_i-e_j),
\qquad
\omega_i=\frac1{\max(\pi_i,\pi_{\min})}.
\]

Anchor the adapted guide to the frozen source:

\[
\mathcal L_{\text{anchor}}
=
\frac1{ND}
\left\|
\hat G_\phi(x)
-\operatorname{sg}(\hat G_{\phi_0}(x))
\right\|_2^2.
\]

Map equivariance:

\[
\mathcal L_{\text{eq}}
=
\|S_\phi(a(x))-W_aS_\phi(x)\|_1.
\]

Total:

\[
\mathcal L_G
=
\mathcal L_{\text{rank}}
+\lambda_a\mathcal L_{\text{anchor}}
+\lambda_e\mathcal L_{\text{eq}}
+\lambda_h\mathcal L_{\text{entropy}}.
\]

Schedule:

1. train saliency head only at approximately `1e-4`;
2. optionally unfreeze the top two guide blocks at approximately `1e-6`;
3. keep the remaining guide frozen;
4. retain random exploration;
5. compare semantic-only, hardness-only, and semantic/hardness mixtures.

Stop adaptation if:

- equivariance drops materially;
- anchor drift grows monotonically;
- entropy collapses;
- frozen guide remains better after the preregistered adaptation window.

### 8.5 Optional explicit alignment ablation

Inspired by VL-JEPA/VLA-JEPA, an explicit feature-alignment branch may be tested:

\[
\mathcal L
=
\mathcal L_{\text{I-JEPA}}
+\lambda_{\text{align}}
D\left(
A(E_\theta(x)),
\operatorname{sg}(G_{\phi_0}(x))
\right).
\]

This is a separate method family. It must be named representation distillation/alignment and compared against allocation-only.

### 8.6 From-scratch confirmation

Run:

1. random from epoch 0;
2. frozen-guide winner from epoch 0;
3. random ep0-to-25 then guide winner;
4. optional faithful DSeq-JEPA.

Use independent seeds for the first three.

Interpretation:

- only switch arm wins: curriculum/specialization claim;
- from-scratch winner also wins: general semantic-allocation claim;
- alignment wins but allocation does not: pivot to semantic distillation;
- frozen wins but adapted loses: keep guide frozen.

A 100-epoch result is a controlled confirmation, not full ImageNet convergence. Longer 300/600-epoch runs occur only after positive multi-seed evidence.

## 9. Phase 3: OCT transfer

### 9.1 Legal, access, and data-overlap gate

No OCT guide advances until a model manifest records:

- code license;
- weight/model-card terms;
- access requirements;
- allowed frozen inference;
- allowed modification/fine-tuning;
- allowed redistribution;
- publication requirements;
- pretraining datasets and possible overlap with evaluation cohorts.

The public MIRAGE repository currently states CC BY 4.0 for code and models, while RETFound and several other candidates have more restrictive terms. The exact weight terms must still be verified from the downloaded artifact/model card before use. Missing or ambiguous terms block adaptation and redistribution.

### 9.2 Model priority

| Tier | Model | Role and limitation |
|---|---|---|
| A | MIRAGE-Base | primary OCT spatial guide; dense 16×16 map; anatomy privileged by layer pseudo-label pretraining |
| A | RETFound-OCT | strong OCT encoder; patch tokens require adapter exposure |
| A control | DINOv3-B/16 | generic dense baseline |
| B | MedSigLIP | medical language-aligned image encoder; OCT specificity unproven |
| B | EyeCLIP | ophthalmology CLIP with OCT support; coarse/non-native spatial map |
| B | RetinaVLM | generated reports and phrase GradCAM; coarse and prompt/phrase dependent |
| B | LO-VLM/OCT-BLIP | compact OCT caption generator; no released spatial-mask API |
| C | OCTCube-M | 3-D OCT representation; later volume-level extension |

Primary references:

- [MIRAGE](https://arxiv.org/abs/2506.08900)
- [RETFound](https://doi.org/10.1038/s41586-023-06555-x)
- [EyeCLIP](https://arxiv.org/abs/2409.06644)
- [RetinaVLM](https://arxiv.org/abs/2407.08410)
- [LO-VLM](https://doi.org/10.1101/2025.08.07.669187)
- [MedGemma/MedSigLIP](https://arxiv.org/abs/2507.05201)
- [OCTCube-M](https://arxiv.org/abs/2408.11227)

LO-VLM is a pseudo-text generator, not a direct mask source unless its BLIP vision tokens or cross-attention are instrumented.

RetinaVLM saliency is phrase/report conditioned and validated on fovea-centered Topcon OCT. It is not a universal OCT mask generator.

MIRAGE has the strongest verified direct spatial interface but imports privileged retinal-layer pseudo-label supervision.

### 9.3 OCT map benchmark

Dataset roles must remain disjoint:

- GOALS training annotations: guide/readout calibration through internal cross-validation.
- Locked GOALS evaluation: one-shot map and glaucoma/layer evaluation.
- RETOUCH: external fluid segmentation and multi-vendor robustness; no guide selection.
- OLIVES: external biomarker and longitudinal transfer; no guide selection.
- OCTID: external disease transfer and limited delineation.
- FairVision: development only until a locked final protocol exists.

Metrics:

- retinal/layer/lesion AP;
- Dice and boundary F1;
- layer-wise target coverage;
- small-lesion recall;
- cross-device map equivariance;
- adjacent B-scan consistency;
- center/border bias;
- guide disagreement and failure taxonomy.

Axial consistency after alignment:

\[
Q_{\text{axial}}
=
\frac1{S-1}
\sum_s
\rho(S(x_s),S(x_{s+1})).
\]

### 9.4 OCT pretraining

1. Reproduce matched-budget random OCT baseline.
2. Resume a common random ep25 checkpoint.
3. Compare:
   - winning ImageNet policy with DINOv3;
   - MIRAGE;
   - RETFound;
   - best language-aligned guide only if Phase 0/1 justified it.
4. Screen ep25-to-50.
5. Extend top two to ep75/100.
6. Run winner from scratch.
7. Consider OCTCube-M or axial consistency only after the 2-D claim is established.

### 9.5 OCT downstream tasks

| Dataset | Task | Primary metric |
|---|---|---|
| FairVision | volume-level glaucoma classification | patient-level AUROC |
| GOALS | glaucoma classification | AUROC/accuracy |
| GOALS | retinal-layer segmentation | Dice/boundary F1 |
| RETOUCH | fluid segmentation | macro Dice |
| OLIVES | biomarker prediction | macro AUROC/F1 |
| OCTID | disease classification | macro AUROC/F1 |
| Kermany OCT2017 | secondary leakage-audited disease benchmark only | macro AUROC |

All primary splits must group both eyes and all visits by patient. Kermany OCT2017 lacks dependable public patient identifiers and cannot serve as a confirmatory benchmark unless a trustworthy grouping key is recovered.

### 9.6 OCT gate

The winning policy must:

- improve FairVision under paired multi-seed evaluation;
- improve or tie at least one independent OCT dataset;
- avoid major subgroup degradation;
- preserve map quality under device shift;
- retain a positive effect across independent seeds;
- pass a training-data overlap audit for every external guide.

If gains appear only on FairVision, the method is dataset specific.

## 10. Compute and scaling gates

Define \(C_{100}\) as measured GPU-hours for one canonical 100-epoch run on the selected hardware.

Before Phase 1:

1. Run a 1,000-step throughput and memory benchmark for each guide.
2. Include guide inference in total GPU-hour accounting.
3. Do not extend every arm beyond ep50.
4. Do not run multiple seeds for losing arms.
5. Do not start co-adaptation without a frozen-guide winner.
6. Do not start 300/600-epoch training without positive multi-seed ep100 evidence.
7. Do not load language decoders or text towers in the primary ImageNet experiment.

Screening cost is computed after Phase 0 eliminates unsupported guides:

\[
C_{\text{screen}}
=
A_{\text{primary}}\times0.25C_{100}
+C_{\text{controls}}
+C_{\text{probes}},
\]

where \(A_{\text{primary}}\) is the preregistered number of ep25-to-50 arms. Six primary arms cost 1.5 \(C_{100}\) before mechanism controls, downstream evaluation, and guide overhead.

Confirmatory seeds require independently trained ep0-to-25 prefixes. Multiple continuations from one prefix do not constitute independent pretraining seeds.

## 11. Branch implementation plan

### Phase 0 files

- `src/guides/base.py`: guide tensor contract.
- `src/guides/dinov3.py`: DINOv3 adapter.
- `src/guides/siglip2.py`: SigLIP 2 vision adapter.
- `src/guides/clip.py`: MILAN-style CLIP adapter.
- `src/guides/ijepa.py`: endogenous I-JEPA map adapter.
- `scripts/semantic_map_benchmark.py`: extraction and metrics.
- `scripts/plot_semantic_map_atlas.py`: visual atlas.
- `configs/semantic_maps/*.yaml`: pinned guide configs.

### Phase 1 files

- `src/masks/fixed_budget.py`: exact-budget target sampler.
- `src/masks/semantic_guided.py`: common score-to-window policy.
- `src/masks/dseq_map.py`: DSeq-map baseline.
- `src/masks/cluster_guide.py`: cluster baseline.
- `configs/imagenet_guided/*.yaml`: continuation arms.
- `scripts/compare_guided_pretraining.py`: fixed-checkpoint statistics.
- checkpoint and compute manifests.

### Phase 2 files

- guide saliency head and ranking objective;
- anchor/equivariance losses;
- dilation and scale sweep configs;
- predictor-capacity configs;
- from-scratch configs.

### Phase 3 files

- MIRAGE, RETFound, EyeCLIP, MedSigLIP, RetinaVLM adapters;
- optional LO-VLM pseudo-text adapter;
- OCT semantic-map benchmark;
- patient-level evaluation configs;
- data-overlap and license manifest.

No heavy checkpoints or raw datasets are committed.

## 12. Hard stop and pivot rules

1. Drop the VLM hypothesis if DINOv3 ties or wins.
2. Stop frozen semantic guidance if no guide beats matched-budget random.
3. Pivot to positional masking if center/shifted maps reproduce the OCT result.
4. Pivot to explicit anatomy if only MIRAGE/labeled anatomy maps work.
5. Keep the guide frozen if low-LR adaptation fails.
6. Restrict the claim to curriculum if only ep25 switch wins.
7. Pivot to 3-D/axial modeling if no 2-D guide transfers beyond FairVision.
8. Abandon efficiency claims if quality gain per GPU-hour is nonpositive.
9. Stop publication claims if improvements disappear across seeds or external datasets.

## 13. Allowed and forbidden claims

Allowed only after evidence:

- frozen semantic teachers can improve matched-budget I-JEPA target allocation;
- language-aligned pretraining adds value beyond DINOv3;
- semantic-region dilation interacts with predictor capacity;
- anchored low-LR adaptation improves a frozen guide;
- an ImageNet policy transfers to OCT.

Forbidden:

- first VLM-guided masking;
- first prompt-free CLIP masking;
- first caption-conditioned JEPA;
- first OCT VLM saliency;
- "mapped into JEPA space" without alignment loss;
- language-free training without disclosing external image-text pretraining;
- efficiency without guide compute;
- clinical mask validity without external anatomical evaluation.

## 14. References

1. Assran et al., [I-JEPA](https://arxiv.org/abs/2301.08243), CVPR 2023.
2. Siméoni et al., [DINOv3](https://arxiv.org/abs/2508.10104), 2025.
3. Tschannen et al., [SigLIP 2](https://arxiv.org/abs/2502.14786), 2025.
4. He et al., [DSeq-JEPA](https://arxiv.org/abs/2511.17354), 2025.
5. Hou et al., [MILAN](https://arxiv.org/abs/2208.06049), 2022.
6. [CMT-MAE](https://arxiv.org/abs/2412.17566), 2024.
7. Apple, [TC-JEPA](https://arxiv.org/abs/2605.03245), 2026.
8. Wang et al., [Mask What Matters](https://arxiv.org/abs/2509.23054), 2025.
9. Chen et al., [VL-JEPA](https://arxiv.org/abs/2512.10942), ICLR 2026.
10. Sun et al., [VLA-JEPA](https://arxiv.org/abs/2602.10098), 2026.
11. Miao et al., [JEPA-VLA](https://arxiv.org/abs/2602.11832), 2026.
12. Gao et al., [ImageNet-S](https://arxiv.org/abs/2106.03149), 2021.
13. Kakogeorgiou et al., [AttMask](https://arxiv.org/abs/2203.12719), ECCV 2022.
14. Li et al., [SemMAE](https://arxiv.org/abs/2206.10207), NeurIPS 2022.
15. Li et al., [AnatoMask](https://arxiv.org/abs/2407.06468), ECCV 2024.
16. Morano et al., [MIRAGE](https://arxiv.org/abs/2506.08900), npj Digital Medicine 2025.
17. Zhou et al., [RETFound](https://doi.org/10.1038/s41586-023-06555-x), Nature 2023.
18. Shi et al., [EyeCLIP](https://arxiv.org/abs/2409.06644), npj Digital Medicine 2025.
19. Holland et al., [RetinaVLM](https://arxiv.org/abs/2407.08410), 2024.
20. Haghighi et al., [LO-VLM](https://doi.org/10.1101/2025.08.07.669187), bioRxiv 2025.
21. Sellergren et al., [MedGemma and MedSigLIP](https://arxiv.org/abs/2507.05201), 2025.
22. Liu et al., [OCTCube-M](https://arxiv.org/abs/2408.11227), 2024.
23. Fang et al., [GOALS](https://arxiv.org/abs/2207.14447), 2022.
24. Prabhushankar et al., [OLIVES](https://arxiv.org/abs/2209.11195), 2022.
25. Gholami et al., [OCTID](https://arxiv.org/abs/1812.07056), 2018.
