# Novelty Assessment

## C1 — Stock I-JEPA context-truncation defect

**Verdict: NOVEL, with a qualified priority claim.**

### Closest prior work

1. **I-JEPA** [@assran2023ijepa] introduced multi-block latent prediction. Its official collator finds the smallest sampled mask length in a batch and retains each tensor's prefix so masks can be stacked. The released implementation therefore contains the mechanism we audit, but neither the paper nor the repository documents its row-major spatial bias or its effect on anatomy-sparse medical images.

2. **C-JEPA** [@mo2024cjepa] analyzes different I-JEPA failure modes: possible representation collapse and inadequate learning of mean patch representations. It adds VICReg-style variance, covariance, and invariance regularization. It does not analyze batch-minimum context truncation, prefix selection, or anatomical erasure.

3. **DSeq-JEPA** [@he2025dseqjepa] replaces parallel random target prediction with saliency-ordered sequential prediction. It shows that target ordering and semantic importance affect JEPA learning, but it does not identify or correct stock I-JEPA's batch collation behavior.

### What this paper adds

We trace the behavior to stock-style I-JEPA collation rather than presenting it as a local implementation error. In our OCT setting, context indices are ordered spatially before `t[:min_len]` is applied. The retained prefix is therefore spatially biased rather than a random subset. Comparing batch size one with batch size 64 shows losses of approximately 31--36% of sampled context in rectangle-based arms, with some anatomy-guided policies leaving up to 11.02% of slices with no retinal anatomy in the delivered encoder context.

The priority claim is necessarily search-qualified: failure to find a prior report is not proof that none exists.

### Recommended claim wording

> We identify a previously undocumented medical-imaging consequence of stock I-JEPA's mask batching. Batch-minimum prefix truncation removes 31--36% of sampled context in our rectangle-based arms and, because retained indices follow spatial order, leaves up to 11.0% of OCT slices with no visible retinal anatomy. To our knowledge, prior I-JEPA papers and analyses have not reported this spatially biased context-erasure failure mode.

### Wording to avoid

- “We discovered a bug unique to our implementation.”
- “This defect always removes the lower retina.”
- “We are the first paper ever to notice I-JEPA truncation.”
- “All stock I-JEPA batches lose 31--36% of context.”  
  The percentage is specific to our grid, mask settings, data, and batch size.

---

## C2 — Target composition versus downstream AUC

**Verdict: INCREMENTAL.**

### Closest prior work

1. **AttMask** [@kakogeorgiou2022attmask] uses teacher attention to mask high-attention image regions, establishing that target content matters more than uniformly random patch selection. It studies natural-image masked modeling, not retinal anatomy composition or frozen-probe AUC as a function of target purity.

2. **AnatoMask** [@li2024anatomask] uses reconstruction loss to identify difficult anatomical regions and progressively guide 3D medical-image masking. It establishes that adaptive medical masking can improve representation learning, but it does not hold out a frozen segmentation teacher or measure a target-anatomy-purity/AUC response curve.

3. **Mask What Matters** [@wang2025maskwhatmatters] uses vision-language localization to apply different masking rates to prompt-relevant and background regions, and reports benefits from lower overall masking ratios. It is strong prior art for semantic composition and masking-ratio design, but it does not study retinal OCT, JEPA latent prediction, or a matched comparison spanning moderate to near-pure anatomy targets.

### What this paper adds

Our measured epoch-50 arms show a non-monotonic association: random targets at about 31.6% anatomy purity obtain AUC 0.8641; oracle and envelope rectangles at about 39.7% and 43.2% obtain 0.8740 and 0.8761; and the 97.5%-pure anatomy-shaped policy falls to 0.8654. This is useful OCT-specific evidence against the simple rule “mask more anatomy.”

It is not yet a causal estimate of an optimum. The arms also differ in target shape, unique hidden fraction, predictor workload, and anatomy remaining in context. The phrase “optimum around 40--43%” should therefore be descriptive and restricted to these measured arms.

### Recommended claim wording

> Across our measured masking arms, downstream frozen-probe AUC is non-monotonic in target anatomy purity: the strongest epoch-50 arms place approximately 40--43% of target patches on anatomy, whereas a 97.5%-pure anatomy policy performs substantially worse. Because these arms also differ in target shape, hidden fraction, and context composition, we interpret this pattern as an observational dose-response hypothesis rather than a universal optimum.

### Wording to avoid

- “We prove that 40--43% anatomy is optimal.”
- “Target purity causes downstream AUC.”
- “Near-pure anatomy masking always collapses.”
- “This is the first study showing that masking strategy matters.”

---

## C3 — Anatomy-shaped versus rectangular targets

**Verdict: INCREMENTAL. Broad first-anatomy-guided-masking claims are SCOOPED.**

### Closest prior work

1. **Ceballos Arroyo et al.** [@ceballosarroyo2026anatomymae] use a pretrained artery segmenter to preferentially mask regions near cerebral arteries during 3D CT MAE pretraining and reconstruct both image intensity and artery-distance maps. This is the closest direct precedent for an external segmentation model guiding medical masked pretraining. We instead use retinal-layer segmentation, 2D OCT slices, and JEPA latent prediction, and we audit composition and truncation effects.

2. **AMAP** [@huang2025amap] uses anatomically guided masked-autoencoder pretraining and domain-adaptive prompting for multimodal cerebral aneurysm detection and segmentation. It confirms that anatomy-guided medical masking predates our work. It does not study I-JEPA, retinal OCT layers, context-prefix truncation, or an anatomy-shaped-versus-rectangular trajectory reversal.

3. **VAMAE** [@abolade2026vamae] uses vesselness and skeleton cues to emphasize vessel-rich OCT angiography regions and reconstruct appearance, structural, and topological targets. It is particularly close in modality and anatomical motivation, but concerns 2D OCTA vasculature and MAE reconstruction rather than structural OCT retinal layers and JEPA prediction.

### What this paper adds

The defensible contribution is not the first use of anatomy-guided medical masking. It is a narrower OCT/I-JEPA investigation using a frozen MIRAGE retinal-layer model, together with an explicit comparison between anatomy-shaped targets and anatomy-guided rectangles. The anatomy-shaped arm wins at epoch 30 but loses by epoch 50. That result is informative because it is reported with a confound ledger: target shape, unique hidden fraction, target purity, predictor workload, guide-cache version, and pretraining-seed replication are not all controlled simultaneously.

### Recommended claim wording

> Prior work has used anatomical priors and pretrained segmenters to guide masked medical-image pretraining, so we do not claim the first anatomy-guided masking method. Our narrower contribution is an OCT/I-JEPA study using a frozen retinal-layer model, including an anatomy-shaped-versus-rectangular comparison whose direction reverses between epochs 30 and 50 and whose remaining composition and workload confounds are reported explicitly.

### Wording to avoid

- “We introduce the first anatomy-guided masking method for medical imaging.”
- “Segmentation-guided masking is novel.”
- “Anatomy-shaped masks outperform rectangles.”
- “The epoch-50 reversal proves that rectangular masks are better.”
