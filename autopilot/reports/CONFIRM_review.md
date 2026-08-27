# GenAI4Health @ NeurIPS 2026 Review

## 1. Summary

This paper studies whether the target-mask distribution in 3D retinal OCT I-JEPA pretraining should be guided toward retinal tissue. Six policies continue from one epoch-25 ancestor: unguided rectangles (RANDOM), tissue-located rectangles (ENVELOPE), an intensity-centroid band (CENTROID), two segmentation-shaped variants (ANATOMY-V1/V2), and a coverage policy (COVER). Frozen mean-pool probes are evaluated for glaucoma classification on the same FairVision test set of 3,000 volumes. In the reported runs, ENVELOPE and CENTROID outperform RANDOM, whereas increasingly anatomy-shaped targets do not. The paper supplements this comparison with mask-geometry measurements, low-label probes, subgroup analyses, precision controls, operating-point results, and an audit showing that COVER did not implement its intended coverage constraint.

The most defensible result is narrow: among the rectangle policies in this single continuation, moving targets toward tissue improved test AUC. The broader conclusions about anatomical precision, mechanism, and expected performance across retraining runs are not established by the design.

## 2. Strengths

1. **A useful within-family control.** ENVELOPE changes rectangle location while retaining the RANDOM rectangle sampler's shape, size, and count (Section 3.2). At epoch 50 it yields 0.8761 versus 0.8641 for RANDOM, a paired difference of +0.0120 with 95% CI [+0.0068, +0.0173] (Table 1; Table 11). This is much more informative than comparing one proposed anatomy method only against an unrelated baseline.

2. **The paper measures delivered masks rather than trusting configuration.** Table 2 reports anatomy hidden, purity, mask ratio, retained context, and loss slots from production samplers. Appendix E then identifies directional post-placement truncation in COVER and correctly states that this arm does not test aggressive coverage. This audit prevents a tempting but invalid conclusion from the declining COVER trajectory.

3. **Important statistical distinctions are stated explicitly.** Sections 5 and 6 distinguish paired test-subject uncertainty from pretraining-seed uncertainty and acknowledge that one continuation per policy cannot establish the ranking. Table 11 uses paired bootstrap intervals and correlated-ROC tests on identical cases rather than comparing marginal intervals.

4. **The clinical and subgroup reporting is more informative than AUC alone.** Tables 5, 6, 8, and 9 provide severity-stratified AUC, subgroup uncertainty, calibration, and transferred-threshold sensitivity. The authors appropriately note that CENTROID's sensitivity gain at the nominal 0.90-specificity threshold occurs with test specificity changing from 0.8794 to 0.8696 (Appendix I).

5. **Several technical controls are unusually careful.** Section 5.5 reports a different-hardware re-encoding check, and Table 12 directly measures fp16/fp32 probe effects. The paper also separates excluded precision-spliced runs in Appendix D instead of silently pooling them.

6. **Presentation of the central negative result is generally candid.** Section 6 states that target shape is not identified, the corrected COVER arm was not run, generality is untested, and the test split was repeatedly inspected. Figure 12 also makes the small absolute ROC separation visually clear.

## 3. Weaknesses

### Fatal design problems

1. **The experimental unit for the masking-policy claim has sample size one.** Each policy has one pretraining continuation (Section 5; Section 6). The 3,000 test cases support conditional comparison of two fixed trained models, but they do not estimate continuation-to-continuation variance. Shared initialization removes ancestor variance, not stochastic continuation variance. Consequently, the paired CIs and very small p-values cannot establish that a policy is better in expectation. This is fatal to the paper's principal method-ranking language.

2. **The study does not identify the effect of anatomical precision.** ANATOMY-V2 simultaneously changes mask ratio (21.4% versus 40.0–46.4%), context retained (67.9% versus 40.5–45.6%), loss slots (64 versus about 158–160), shape, and collation (Table 2; Section 5.2). ANATOMY-V1 and V2 are also different implementations observed at different epochs, while COVER has a known defect (Appendix E). Thus H2 and H3 are not answered by a controlled intervention. The data show that one confounded anatomy-shaped configuration did not win; they do not show that anatomical precision itself fails to help.

3. **The nominal test set is no longer an untouched test set.** Section 6 says that policies, checkpoints, and analyses were chosen after repeated inspection of this same split. Section 4 further says ANATOMY-V2 was stopped after its epoch-75 deficit. This adaptive reuse and selective horizon invalidate confirmatory interpretations of the reported intervals, DeLong tests, and multiplicity correction. With no independent external cohort or final untouched split, this problem cannot be repaired by additional analysis of the submitted predictions.

4. **The intended coverage experiment failed and was not rerun.** Appendix E shows that COVER's delivered targets hide 73.1% of anatomy versus ENVELOPE's 77.6%, despite being designed to hide more. Its epoch-100 decline therefore cannot answer the coverage question. Retaining this arm is acceptable descriptively, but it materially reduces the advertised six-policy ladder to a set in which one key intervention is defective.

### Presentation and secondary problems

1. **Pretraining reproducibility is incomplete.** Section 4 does not report the pretraining-set size, validation-set size, sampling of volumes versus slices, augmentation details, optimizer values, learning-rate schedule values, or MIRAGE thresholding and quality on this cohort. Appendix G gives detailed probe hyperparameters, but the central pretraining continuations are less fully specified.

2. **Scope is very narrow.** All evidence comes from one glaucoma label, one OCT dataset, one repeatedly reused split, and one ancestor. The result may depend on anatomy quality, disease signal, acquisition, and downstream head. The appendix's fine-tuning experiment (Table 13) is useful but does not provide an independent dataset or pretraining replication.

3. **The clinical effect remains exploratory.** The AUC improvement is small, the cohort prevalence is 0.4887, the threshold does not preserve matched specificity across arms, smaller racial strata lack resolved sensitivity gains, and there is no external, prospective, subgroup-calibration, or decision-analytic validation (Appendix I and J).

4. **Some causal wording exceeds the analysis.** For example, Appendix H's heading “Errors are weaker signal, not different anatomy” and its attribution-based explanation of error saturation are stronger than the observational occlusion evidence warrants.

## 4. Questions for the authors

1. What are the exact pretraining and validation sample sizes, sampling unit, augmentations, optimizer hyperparameters, and learning-rate schedule for the continuations?
2. Why were no stochastic continuation replicates run, given that Section 6 itself identifies these as the required unit of replication?
3. Can the main conclusion survive a factorial control that matches target area, context budget, loss slots, and collation while changing only anatomical alignment or target shape?
4. Was MIRAGE trained on, tuned on, or otherwise exposed to any FairVision subjects, and how accurate is its retinal envelope on this cohort?
5. How many times and for which decisions was the test split inspected before the reported policies, stopping horizons, and analyses were fixed?
6. Why does the full-supervision label-efficiency experiment not reproduce every Table 1 arm within the stated 0.0003?
7. What false-discovery-rate threshold is intended in Table 4, and why does its caption count three surviving attributes when four rows share or beat q=0.0668?
8. Why does Figure 2(b)'s caption state that all six intervals exclude zero while the panel also plots COVER at three epochs, including an epoch-50 interval that contains zero?

## 5. Unsupported claims and internal numerical audit

### Internal inconsistencies

1. **The “within 0.0003” reproduction claim is false for two displayed arms.** Section 5.3 and Appendix G state that full-supervision Table 7 reproduces Table 1 within 0.0003. RANDOM differs by 0.0002 (0.8748 versus 0.8746) and CENTROID by 0.0001 (0.8856 versus 0.8855), but ENVELOPE differs by 0.0006 (0.8813 versus 0.8807) and COVER by 0.0009 (0.8586 versus 0.8577).

2. **Table 4's multiplicity caption is numerically inconsistent with its rows.** It says three attributes survive correction: sex, disease severity, and race. The table gives q=0.0038 for sex and q=0.0668 for race, ethnicity, and disease severity. At a 0.05 threshold only sex survives; at a threshold admitting 0.0668, four attributes survive, not three. No alternative threshold is stated.

3. **Figure 2(b)'s caption is misleading about interval count.** It says “All six intervals exclude zero,” but the plotted legend has three contrasts at three epochs. Section 5.1 gives COVER minus RANDOM at epoch 50 as +0.0002 with CI [-0.0050, +0.0053], which includes zero. The intended six ENVELOPE/CENTROID contrasts should be identified explicitly.

### Claims not supported by the submitted evidence

1. **“Anatomical precision did not help” is not causally identified.** Table 2 documents simultaneous changes in at least four geometry and optimization variables, and Appendix E invalidates the intended coverage intervention.

2. **The proposed consistency mechanism is speculative.** Section 6 says the required random-position band control was not run, yet Section 7 suggests consistency rather than anatomical correctness is the operative variable. The current results do not distinguish consistency from masking ratio or task difficulty.

3. **Stable worst-group identity does not show that disparity was “not introduced” by a policy.** Section 5.4 establishes that the same group remains worst; a policy could still widen or otherwise alter a pre-existing disparity without changing which group ranks last.

4. **The claimed label-scarcity concentration is descriptive, not established.** Section 5.3 explicitly says the widening across fractions was not tested. Repeated probe fits on fixed encoders also do not address pretraining-run variance.

5. **A “usable sensitivity gain” is premature.** Appendix I compares a shared numerical threshold at different achieved specificities and uses the repeatedly inspected test cohort. The paper itself correctly states that deployment-grade comparison needs locked, arm-specific validation thresholds and an untouched cohort.

### Checks that do agree

The central Table 1 arithmetic agrees with Table 11 and the surrounding text: 0.8761 minus 0.8641 gives +0.0120 for ENVELOPE at epoch 50, and 0.8855 minus 0.8746 gives approximately +0.0109 for CENTROID at epoch 100. The apparent COVER epoch-100 rounding difference is explained by its comparison to the fp32 RANDOM value in Table 12 rather than the displayed fp16 RANDOM value. The differing mask percentages in Table 2 and Figure 3 are also labeled as different sweeps (600 versus 6,137 slices) and are not themselves a contradiction.

## 6. Clinical actionability

**Clinically actionable today: No.** This is a retrospective representation-learning comparison on one public dataset and a repeatedly inspected split. There is no independent external or prospective validation, no deployment-grade matched-specificity comparison, incomplete subgroup calibration, and no evidence that the small conditional AUC gain is stable across pretraining continuations.

## 7. Scores

- **Quality: 2/4.** The audits and conditional paired analyses are careful, but one continuation per policy, adaptive test reuse, a defective arm, and uncontrolled anatomy-family confounds prevent reliable method-level inference.
- **Clarity: 3/4.** The main narrative, tables, and limitations are unusually explicit, but several captions and numerical cross-references are internally inconsistent and key pretraining details are missing.
- **Significance: 2/4.** A controlled negative result in medical SSL is relevant, but the evidence is confined to one task and cannot establish the central anatomy-precision conclusion or its stability.
- **Originality: 3/4.** The shared-ancestor retinal I-JEPA comparison and delivered-mask audit are novel and useful, although the paper's own Table 10 shows that the qualitative direction is already known.
- **Overall: 3/6 (Weak Reject).** The submission contains a valuable descriptive case study, but its fatal replication, confounding, and test-reuse limitations are too severe for acceptance as evidence about masking-policy effectiveness.
- **Confidence: 4/5.** The paper provides enough numerical and methodological detail to assess the main design and cross-check its tables, though exact continuation variance cannot be inferred from the artifact.
