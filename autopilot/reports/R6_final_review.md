# GenAI4Health @ NeurIPS 2026 Research Paper Review

## 1. Summary

This paper studies where I-JEPA predictor targets should be placed during self-supervised pretraining of 3D retinal OCT. Six masking policies continue from one shared epoch-25 checkpoint: unguided rectangles (RANDOM), rectangles rejection-sampled toward retinal tissue (ENVELOPE), a segmentation-free intensity-centroid band (CENTROID), two segmentation-shaped anatomy variants, and a coverage-constrained policy (COVER). Evaluation uses frozen mean-pooled linear probes for glaucoma on the patient-disjoint FairVision test set (N=3000), with paired subject bootstrap and correlated-ROC tests.

For the particular continuations run, ENVELOPE improves over RANDOM by +0.0120 AUC at epoch 50, and CENTROID improves by +0.0109 at epoch 100 (Tables 1 and 14). ANATOMY-V2 is indistinguishable from RANDOM at epoch 50 and lower at epoch 75, while COVER degrades after its epoch-73 peak. A 600-slice geometry audit shows that the shaped anatomy arm has 97.1% mask purity but differs greatly in mask ratio, context, and loss slots (Table 2). The paper also reports label-efficiency, subgroup, operating-point, precision, fine-tuning, and attribution analyses. The focused empirical question is appropriate for this workshop, but the headline prescription is stronger than the design can establish.

## 2. Strengths

1. **A useful near-single-variable control within the rectangle family.** ENVELOPE and RANDOM retain the same target family, shapes, sizes, and counts while changing placement. At epoch 50, their AUCs are 0.8761 and 0.8641, respectively, with paired Delta AUC +0.0120 and 95% CI [+0.0068, +0.0173] (Section 5.1; Tables 1 and 14). This is the paper's strongest evidence that target location can matter in this run.

2. **The paper measures delivered masks rather than relying on intended configurations.** Table 2 and Appendix D quantify anatomy hidden, purity, mask ratio, context, and loss slots on production samplers. This reveals that ANATOMY-V2 has 64.0 loss slots versus about 159 for rectangle policies and a 21.3% mask ratio versus 40.3-46.5%, making the non-monotonic result much more interpretable.

3. **Several evaluation checks are technically careful.** The paired bootstrap applies identical resamples to predictions on the same 3000 subjects (Section 4), the full fp32 re-probe changes AUC by at most 0.000192 (Table 15), and re-encoding the released CENTROID model on different hardware reproduces AUC within 9.8 x 10^-6 (Section 5.6). These checks address subject-sampling, precision, and implementation reproducibility for fixed trained models.

4. **The analysis goes beyond one aggregate frozen-probe number.** Table 10 reports a larger point-estimate margin at 5% labels (+0.0496); Tables 7-9 examine severity and demographic strata; Table 11 reports sensitivity, specificity, predictive values, Brier score, and ECE; and Table 16 shows that the mean-pool fine-tuning gap remains +0.0079. These are relevant health-AI views, even where the inferential limits are substantial.

5. **Scope and related work are well positioned.** Section 2 and Appendix M explicitly place the result among mixed prior findings on informed masking rather than presenting the direction as unprecedented. The body fits the nine-page research-paper limit, and the main narrative and figures are generally polished and readable.

## 3. Weaknesses

### A. Fatal or claim-limiting design problems

1. **The nominal test split was repeatedly used for research decisions.** Section 4 states that policy, checkpoint, analysis, and stopping-horizon choices followed repeated inspection of the same test set; Section 6 adds that the number of inspections is unknown. Consequently, the bootstrap intervals, DeLong p-values, and Benjamini-Hochberg q-values do not account for adaptive selection. This is fatal to confirmatory inference from those values and leaves even the fixed-run differences vulnerable to selection optimism. A separate validation set for fitting the head does not repair adaptive use of the test outcomes.

2. **There is one pretraining continuation per policy and no reproducible probe-seed estimate.** Section 6 correctly notes that subject bootstrap does not measure optimization or probe variance, while Appendix B's multi-continuation replication is explicitly PENDING. Therefore, Tables 1 and 14 establish differences among these fitted models on this reused cohort, not an expected effect of the masking policies. This is fatal to a policy-level recommendation, especially for AUC gaps of 0.0062-0.0120.

3. **The central “not how much to cover” conclusion is not identified.** Section 5.2 says H3 is not identified: ANATOMY-V2 differs from ENVELOPE in mask ratio (21.3% versus 46.5%), context (67.7% versus 40.6% per image), loss slots (64.0 versus 159.7), collation, and guide provenance (Table 2; Appendices D, G, and P). COVER is additionally defective: its targets are shortened after placement, and the corrected arm was not run (Appendix G). Thus the title, Abstract, Section 5.2 heading, and Conclusion imply a causal distinction between aim and amount that the submitted experiment does not isolate.

4. **Stopping and horizon are selective across policies.** ANATOMY-V2 was stopped at epoch 75 after its deficit was seen, ANATOMY-V1 has only epoch 30, and epoch 50 is the only common comparison for five policies (Section 4). The paper acknowledges that ANATOMY-V2 could recover, but the asymmetric horizon still prevents a balanced long-horizon ranking.

5. **External validity is narrow.** All evidence comes from one FairVision glaucoma task, one test split, one OCT setting, and one segmenter ecosystem (Section 6). This is acceptable scope for a workshop study, but it cannot support a general prescription for medical masked pretraining without another disease, modality, cohort, or scanner.

### B. Presentation and reporting problems

1. **The inferential reporting is internally inconsistent.** Section 5.5 combines the branch-level sex correlation, rho = -0.821, with q = 0.0038, but Table 6 shows branch-level p = 0.0234; q = 0.0038 belongs to the checkpoint-level result (rho = -0.664, p = 0.0005). This makes the claimed branch-level multiplicity evidence incorrect as written.

2. **The paper uses an unclear significance threshold for the race trend.** Appendix E says p = 0.0225 “passes Benjamini-Hochberg” with q = 0.0668, while the Table 6 caption says only sex survives correction at the conventional 0.05 level and explicitly places race above that threshold. The threshold must be stated consistently; at 0.05, q = 0.0668 does not pass.

3. **The full-supervision parity claim disagrees across sections.** Section 5.4 says the label-efficiency sweep reproduces Table 1 within 0.0009. Appendix I instead says within 0.0003. Tables 1 and 10 support the former: COVER is 0.8586 versus 0.8577 (difference 0.0009), and ENVELOPE is 0.8813 versus 0.8807 (difference 0.0006).

4. **The fairness limitations contradict an analysis that is actually reported.** Appendix E and Appendix L say there is no subgroup calibration, but Appendix K reports race-stratified ECE changes: white 0.0382 to 0.0321, Asian 0.0832 to 0.0729, and Black 0.0525 to 0.0569. The intended distinction, perhaps “no complete subgroup calibration analysis,” should replace the categorical statement.

5. **Some important exploratory contrasts are cited but not tabulated.** For example, Section 5.1 gives ANATOMY-V2 minus ENVELOPE at epoch 50 with an interval, while Table 14 contains only the nine rectangle-family contrasts and the other 25 contrasts are described as stored statistics. A compact appendix table of every contrast used in prose would make the submitted artifact independently auditable.

## 4. Questions for the authors

1. How many times was the test split inspected, and which policy, checkpoint, stopping, and analysis decisions were made after each inspection? If this cannot be reconstructed, what evidence can distinguish the reported gain from adaptive selection?
2. Do the pending continuation seeds preserve the signs and approximate magnitudes of ENVELOPE-RANDOM and CENTROID-RANDOM at epoch 50? Without those results, why should the headline be stated at policy rather than run level?
3. What does a corrected coverage arm show when coverage is actually delivered, with mask ratio, context, target count, loss slots, collation, and guide provenance matched to the location control?
4. Which exact MIRAGE guide and residual-adapter path was used by ANATOMY-V2? Appendix P explicitly describes ENVELOPE, ANATOMY-V1, and COVER, but the ANATOMY-V2 provenance is not equally explicit.
5. How many volumes and slices were used for self-supervised pretraining, and were they exclusively from the FairVision training split? The downstream test size is clear, but the pretraining cohort size is not.
6. How large is primary-probe seed variance under the exact reported protocol? The paper says an earlier multi-seed check cannot be reproduced, leaving a key source of uncertainty unquantified.
7. Can the band-position control proposed in Section 6 be run while holding mask area and delivered context fixed, so that consistent retinal location can be separated from masking ratio and prediction difficulty?

## 5. Unsupported claims and numerical consistency audit

### Claims not supported by the shown evidence

- **“Anatomy says where to aim, not how much to cover.”** The observed ordering is real for these runs, but amount is never isolated; H3 is explicitly unidentified and the only intended high-coverage arm is defective (Section 5.2; Appendix G). The evidence supports “simple guidance performed best among these implementations,” not the causal headline.
- **“Background targets are not wasted.”** Appendix H shows that background targets are predictable beyond a position-only reference and that background representations change, but it does not show that including those targets improves downstream learning. The eroded-background and content-shuffle controls are both PENDING, so “genuine prediction signal” is supported while “not wasted” is not causally established.
- **“The advantage concentrates where labels are scarce.”** Table 10 has larger point-estimate gaps at low label fractions, but Section 5.4 explicitly says differences between fractions were not tested. This should remain an observation, not a comparative conclusion.
- **A “usable sensitivity gain” at a shared deployed threshold.** In Appendix K, CENTROID sensitivity is 0.7428 versus 0.7162, but specificity changes from 0.8794 to 0.8696. The paper notes this caveat; without model-specific validation thresholds and an untouched test cohort, the result does not establish a deployment benefit at matched false-positive burden.

### Internal numerical/reporting contradictions found

1. **Sex trend:** Section 5.5 reports branch-level rho = -0.821 with q = 0.0038; Table 6 reports branch-level rho = -0.821 with p = 0.0234, while q = 0.0038 belongs to checkpoint-level rho = -0.664.
2. **Race trend:** Appendix E calls q = 0.0668 a Benjamini-Hochberg pass; Table 6 says only sex survives at 0.05, so the race result does not pass that stated threshold.
3. **Full-supervision reproduction:** Section 5.4 says agreement within 0.0009; Appendix I says within 0.0003. Tables 1 and 10 show a maximum difference of 0.0009 (COVER: 0.8586 versus 0.8577).
4. **Subgroup calibration:** Appendices E and L say none is reported, but Appendix K gives subgroup ECE values for all three race strata.

The principal AUC arithmetic in Table 1 is otherwise consistent at the shown precision: 0.8761 - 0.8641 = 0.0120 and 0.8855 - 0.8746 = 0.0109. The severity deltas in Tables 7 and 8 and the sensitivity deltas in Table 11 also agree at the displayed precision.

## 6. Official scores

- **Quality: 2/4.** The fixed-model paired evaluation and engineering checks are careful, but adaptive test reuse, one continuation per policy, unmeasured probe variance, and confounded or defective key arms prevent rigorous policy-level inference.
- **Clarity: 4/4.** The hypotheses, policies, limitations, tables, and figures are unusually well organized and readable, notwithstanding the specific reporting contradictions above.
- **Significance: 3/4.** Choosing medical-SSL masking targets is relevant and the low-label point estimates are potentially useful, but the absolute main gains are modest and demonstrated only on one retrospective glaucoma cohort.
- **Originality: 3/4.** The controlled OCT/I-JEPA comparison and delivered-mask audit are distinctive, although Appendix M correctly shows that the broader finding that informed masking need not beat random masking is not new.
- **Overall: 3/6 (weak reject).** This is a strong workshop question and a valuable descriptive study, but the submitted evidence does not yet support its causal headline or a policy-level recommendation; continuation replication, an untouched evaluation, and a corrected matched coverage control are needed.
- **Confidence: 5/5.** I read the full submission, including all appendices, and cross-checked the headline, subgroup, calibration, precision, and fine-tuning numbers against the corresponding tables.

## 7. Clinically actionable today

**No.** The models are evaluated retrospectively on one repeatedly inspected dataset with no external or prospective validation, one continuation per policy, no clinically locked model-specific threshold comparison at matched realized specificity, and unresolved subgroup threshold-transfer disparities (Appendix K). The paper itself states in Appendix L that the models are not clinically validated or intended for deployment.