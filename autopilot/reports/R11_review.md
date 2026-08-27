# GenAI4Health @ NeurIPS 2026 — Research Paper Review

## 1. Summary

This paper studies whether I-JEPA pretraining for 3D retinal OCT benefits from placing prediction targets on retinal tissue. Six masking policies continue from one shared epoch-25 checkpoint, ranging from unguided rectangles to segmenter-located rectangles, a segmentation-free intensity-centroid band, anatomy-shaped targets, and a coverage policy. Frozen mean-pooled linear probes are evaluated for glaucoma classification on the same 3,000-subject FairVision test split.

The strongest supported empirical result is narrow but useful: in the reported continuations, moving ordinary rectangular targets toward tissue improves AUC over random placement, while the simple segmentation-free CENTROID policy is not separated from segmenter-guided ENVELOPE at epochs 50 and 75 and is higher at epoch 100. The paper also shows that the more anatomically precise implementations do not improve this task, but it correctly documents that mask ratio, delivered context, loss slots, collation, and guide provenance prevent a causal conclusion about anatomical precision itself.

## 2. Strengths

1. **A useful and unusually well-audited comparison.** The shared ancestor, schedule, effective batch size, evaluation cases, and probe protocol make the rectangle-family comparison much stronger than unrelated end-to-end runs (Sections 3–4). Table 1 and Table 14 consistently report ENVELOPE-minus-RANDOM gains of +0.0120, +0.0080, and +0.0062 and CENTROID-minus-RANDOM gains of +0.0099, +0.0113, and +0.0109 at epochs 50, 75, and 100.

2. **The paper measures the masks actually delivered rather than relying on intended policy names.** Table 2, Figure 3, and Appendices D and G expose material differences in anatomy hidden, purity, mask ratio, retained context, and loss slots. Discovering and sharply limiting the interpretation of the COVER collation defect is scientifically valuable.

3. **The statistical reporting is mostly careful.** Paired subject-level intervals are appropriate for comparisons on identical cases; Table 14 reports the full declared contrast family; Table 8 uses simultaneous intervals for subgroup changes; and the paper repeatedly distinguishes test-subject uncertainty from pretraining-run uncertainty.

4. **The method is now legible.** Figure 1(a) makes the full path easy to follow: visible tokens enter the context encoder, positional target queries enter the predictor, EMA features provide the regression target, and the frozen EMA encoder feeds the volume-level probe. This schematic earns its space. Figure 1(b) is cramped, but it gives immediate visual meaning to the six policies and Figure 4 supplies the inspectable version.

5. **Scope and deployment limitations are substantively analyzed.** Tables 7–12 go beyond aggregate AUC to disease severity, subgroup uncertainty, operating points, and calibration. In particular, Appendix K does not disguise the fact that a shared threshold changes realized specificity.

## 3. Weaknesses

### Design problems fatal to stronger causal or confirmatory readings

1. **The test set is not an untouched test set.** Section 4 states that policy, checkpoint, analysis, and stopping choices followed repeated inspection of the same test split; Section 6 says the number of inspections is unknown. Consequently, the bootstrap intervals, DeLong tests, and multiplicity corrections cannot support confirmatory generalization. They describe this adaptively selected artifact. The paper acknowledges this, so I do not penalize it twice, but the word “confirmatory” in Appendix N is not defensible.

2. **H1 is not isolated as “location alone” in the delivered training input.** Section 5.1 calls ENVELOPE versus RANDOM the cleanest control and says they differ in location alone. Yet Table 2 reports mask ratio 44.5% versus 46.5%, and Table 5 reports delivered context 24.7% versus 30.7%. Thus the implementation change is the placement sampler, but the encoder also receives a materially different context budget. The comparison establishes that this implemented policy performed better in this continuation; it does not establish that avoiding background, rather than induced context/overlap geometry, caused the gain.

3. **The anatomical-precision and coverage questions remain unidentified.** At epoch 50, ANATOMY-V2 has a 21.3% mask ratio, 67.7% per-image context, and 64 loss slots, versus roughly 40–47%, 41–46%, and 159 for rectangle policies (Table 2); Appendix D shows an even larger delivered-context difference. Appendix P adds guide-provenance differences, while Appendix G invalidates COVER as a test of aggressive coverage. These results rule out a monotone ranking for the implementations tested, but not the broader claim that anatomical precision “does not add.” The positive ANATOMY-V1 comparison at epoch 30 further counsels restraint.

4. **One continuation per reported policy leaves optimization noise unresolved.** This is not independently disqualifying at this workshop, and the paper appropriately phrases the main result as applying to “these runs.” However, subject-bootstrap significance cannot establish an expected policy ranking. Appendix B’s replication has no submitted result, and its fixed data order would still be narrower than independent retraining.

### Presentation and internal-consistency problems

1. **Figure 1(b) is too small for detailed inspection.** It earns conceptual space beside the excellent schematic, but labels and mask boundaries are difficult to read at page scale. A simpler three-policy main-text panel, leaving all six to Figure 4, would improve the balance.

2. **Figure 15 does not numerically reproduce Table 2 despite both being captioned as 600-slice mask-geometry measurements.** For example, RANDOM anatomy hidden is 52.2% in Figure 15 versus 54.0% in Table 2, and RANDOM loss slots are 157.7 versus 159.9. Table 4 supports the Table 2 values. If Figure 15 is a different draw or legacy sweep, its caption must identify that provenance; otherwise this is an internal numerical inconsistency.

3. **Appendix E calls the four retracted rows “coverage probes,” while Table 3 and Appendix F explicitly identify them as RANDOM/unguided probes from a coverage campaign.** This is a labeling contradiction, albeit not one that changes a result.

4. **Terminology is needlessly unstable.** Appendix figures retain “oracle” for CENTROID and “blob” is explained as an artifact label for ANATOMY-V2. Captions decode these names, but canonical labels throughout would reduce cognitive load.

## 4. Questions for the authors

1. Can ENVELOPE be rerun with the delivered context budget and unique mask ratio matched to RANDOM, so that “avoiding background” is separated from overlap and context effects?
2. What exact sampling draw produced Figure 15, and why does it differ from the same-600-slice values in Tables 2 and 4?
3. Given the adaptive use of the test split, why is the nine-contrast family called “confirmatory” in Appendix N rather than exploratory?
4. Do the continuation-level replications preserve the ENVELOPE-minus-RANDOM and CENTROID-minus-RANDOM signs, and can any future evaluation use an untouched external cohort?

## 5. Claims not supported by the submitted evidence and numerical cross-check

- **“H1 holds” and “location alone” are too strong.** Tables 2 and 5 show consequential geometry and delivered-context changes, so the evidence supports a policy comparison, not the stated mechanism.
- **“Anatomical precision does not add” is supportable only as a description of these implementations.** H2 is mixed and H3 is explicitly unidentified; the paper should consistently use the narrower wording.
- **The operating-point gain is not a same-specificity gain.** Appendix K correctly notes that the shared threshold changes test specificity, so the sensitivity increase cannot by itself establish clinical utility.
- **Figure 15 and the “coverage probes” label require correction as described above.**

I otherwise found the headline arithmetic internally consistent: Table 1 deltas agree with the precision-matched AUCs in Table 3 and with Table 14; Table 7’s three deltas agree with its arm values; the subgroup counts in Appendix E sum to 3,000 and severity counts sum to 1,466 positives; Table 10 supports the stated +0.0496 five-percent-label margin; and Table 11 supports the stated +0.0266 sensitivity difference, subject to its specificity caveat.

## 6. Scores

- **Quality: 3/4.** The empirical auditing and paired evaluation are strong, but adaptive test reuse and uncontrolled delivered-context differences preclude the strongest causal interpretation.
- **Clarity: 4/4.** The paper is unusually explicit about what was run and what failed, and Figure 1(a) makes the architecture readily understandable despite a cramped policy panel.
- **Significance: 3/4.** A carefully bounded negative/control result about anatomy-guided medical SSL is useful, though the measured gain is modest and limited to one task and cohort.
- **Originality: 3/4.** Informed-versus-random masking is not new, but the segmentation-free versus segmenter-guided ladder, delivered-mask audit, and retinal OCT setting provide a distinct contribution.
- **Overall: 4/6 (weak accept).** At this workshop’s research-paper bar, the submitted evidence substantiates the narrow run-level comparison and offers useful negative evidence, while the design flaws prevent a stronger acceptance.
- **Confidence: 5/5.** The main text, tables, figures, and appendices provide enough detail to assess both the intended comparisons and their confounds.

## 7. Clinically actionable today

**No.** The result is retrospective on one public cohort, uses one adaptively inspected test split, has no submitted continuation-level replication or external validation, and shows material subgroup threshold-transfer disparities (Section 6; Appendices K and L). It is evidence about pretraining design, not a validated screening system.
