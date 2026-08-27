# GenAI4Health @ NeurIPS 2026 — Research Paper Review

## 1. Summary

This paper studies whether the spatial policy used to choose I-JEPA prediction targets affects representation learning for glaucoma classification from 3D retinal OCT. Six masking policies continue from one shared epoch-25 checkpoint: unguided random rectangles, segmenter-guided rectangles placed within the retinal envelope, a segmentation-free intensity-centroid band, two anatomy-shaped variants, and a coverage-oriented variant. Encoders are evaluated with frozen mean-pooled linear probes on the FairVision test set (3,000 volumes).

The cleanest result is the comparison between RANDOM and ENVELOPE: moving otherwise identically sampled rectangles onto the retinal envelope improves test AUC by 0.0120 at epoch 50, with smaller positive differences at epochs 75 and 100 (Table 1; Table 14). CENTROID is best at epoch 100, improving over RANDOM by 0.0109. The paper also reports mask-geometry measurements, low-label probes, subgroup analyses, operating-point metrics, and extensive post-hoc audits.

The positive “where to aim” result is credible as a descriptive result for these particular continuations. The stronger negative claim in the title—“not how much to cover”—is not established. The intended coverage arm has a collation defect, no corrected arm was run, and anatomy-shaped arms differ in delivered context, mask ratio, loss slots, collation, and guide provenance. Moreover, policies and analyses were selected after repeated inspection of the same test split. I therefore view this as promising workshop work with one useful finding, but not yet sufficiently identified to support its full central claim.

## 2. Strengths

1. **A useful, health-relevant question with a strong baseline.** The paper tests an intuitively appealing medical-SSL assumption rather than merely introducing another guided masker. The RANDOM–ENVELOPE comparison is especially informative: rectangle shape, nominal size, and count are held fixed while placement changes (Sections 3.2 and 5.1; Tables 1 and 14).

2. **The principal positive effect is consistently measured across the reported checkpoints.** ENVELOPE minus RANDOM is +0.0120, +0.0080, and +0.0062 at epochs 50, 75, and 100; CENTROID minus RANDOM is +0.0099, +0.0113, and +0.0109 (Table 14). All six paired intervals exclude zero after the declared nine-contrast correction. These intervals quantify test-subject sampling for fixed fitted models, and the paper correctly avoids presenting them as seed-to-seed uncertainty (Sections 5.1 and 6).

3. **Unusually strong numerical and protocol auditing.** The paper inventories every probe (Table 3), partitions comparisons by probe precision (Section 4), re-runs the full headline family at fp32 (Table 15), measures production mask geometry over 600 slices and three draws (Tables 2 and 4), and separately measures batch-delivered context (Table 5). Re-encoding from released weights reportedly reproduces the headline AUC within 9.8 × 10^-6 (Section 5.6).

4. **The paper exposes rather than hides failed assumptions.** Appendix G documents that post-placement target truncation defeats the COVER objective, and Section 5.2 explicitly lists the context, target-area, and loss-slot confounds. This does not repair the design, but it makes the submitted artifact much more interpretable.

5. **The evaluation goes beyond one aggregate AUC.** Label efficiency (Table 10), severity and demographic strata (Tables 7–9), operating points and calibration (Table 11), and transferred-threshold subgroup sensitivity (Table 12) are relevant to the intended health setting. The authors generally distinguish point-estimate observations from adjusted findings.

6. **Clarity is generally high despite the density.** The nine-page body is logically structured, figures flag truncated axes, and limitations are tied to specific claims. Figure 2 usefully separates matched precision families and plots paired differences rather than misleading marginal error bars.

## 3. Weaknesses

### A. Design problems fatal to particular central claims

1. **The negative half of the central claim is not tested.** COVER was intended to test aggressive anatomical coverage, but the collator shortens targets after placement; delivered COVER masks hide 73.1% of anatomy versus ENVELOPE’s 77.6% (Appendix G). The corrected arm is unrun, only one floor (`f=0.21`) was pretrained, and there is no coverage dose-response. Thus the title, Section 5.2 statement “Anatomy tells you where to aim, not how much to cover,” Section 6 statement “Anatomical coverage does not,” and the corresponding conclusion go beyond the evidence.

2. **H2/H3 are not identified by the six-arm design.** ANATOMY-V2 has 64 loss slots versus about 159 for rectangle arms, a 21.3% mask ratio versus 40.3–46.5%, and 63.5% batch-delivered context versus 24.7–32.9% (Tables 2 and 5). The families also use different collation, and segmenter-driven arms do not all use the same guide: ENVELOPE uses unmodified MIRAGE occupancy, whereas ANATOMY-V1 and COVER use adapter-derived soft guides (Appendix P). A deficit cannot therefore be attributed to anatomical precision or target shape. The paper admits this, but still uses the result to motivate its headline.

3. **The test set is adaptively reused as a development set.** Section 4 and Section 6 state that policy, checkpoint, analysis, and stopping choices followed repeated inspection of the same 3,000-case test split, with the number of inspections unknown. Consequently, the paired intervals and adjusted p-values do not control the selection process, and even the point-estimate ranking may be optimistically selected. Appendix B’s pending continuation replication would address optimization variability but explicitly would not supply an untouched evaluation cohort.

4. **Even the clean rectangle comparison does not completely isolate anatomy from delivered geometry.** Although RANDOM and ENVELOPE sample the same nominal rectangles, placement changes overlap and therefore realized mask ratio and context: 44.5% versus 46.5% mask ratio and 41.9% versus 40.6% per-image context (Table 2). This is a much smaller problem than the anatomy-family confounds, and the comparison remains useful as a policy-level result, but it does not by itself prove that avoiding background rather than the changed delivered task geometry caused the gain.

### B. Important but nonfatal empirical limitations

1. **One continuation and one probe seed per policy.** At this workshop, single-run pretraining is not independently disqualifying, and the paper carefully uses run-level language. Nevertheless, the reported intervals cannot establish an expected policy ordering under retraining, and the planned continuation-level replication is still pending (Section 6; Appendix B). This matters because the effects are approximately 0.006–0.012 AUC.

2. **External validity is narrow.** All central results use one retrospective dataset, one disease, one OCT preprocessing pipeline, one shared ancestor, and one frozen-probe architecture (Sections 4 and 6). There is no independent scanner, institution, population, disease, modality, or untouched temporal cohort.

3. **The segmentation guidance is insufficiently characterized.** The paper does not report MIRAGE segmentation quality on FairVision, and Appendix P introduces a residual-adapter guide provenance difference whose downstream effect was never isolated. This limits interpretation of “anatomical precision does not add.”

4. **Several secondary conclusions are stronger than their tests.** The Section 5.4 heading says the advantage “concentrates where labels are scarce,” but the paper explicitly says it did not test whether the gap widens across label fractions. Appendix J uses three fine-tuned probes and hand-picked patch maps to support a claim about the main frozen-probe experiment. Appendix K’s sensitivity benefit is measured at a shared threshold where achieved specificity changes from 0.8794 to 0.8696, so it is not a pure gain at matched false-positive rate.

5. **The subgroup audit is underpowered for the smaller groups and incomplete for deployment.** Adjusted intervals include zero for Black and Asian race strata (Table 8) and for their transferred-threshold sensitivity changes (Table 12). Intersectional changes have no paired intervals (Table 9). The threshold-specificity disparity—0.895 for White versus 0.736 for Black under RANDOM—is much larger than the masking-policy effects (Appendix K).

### C. Presentation and reporting problems

1. **The artifact is too diffuse.** The body and 25-page appendix mix the primary masking experiment with background-token mechanisms, attribution, fairness trends, calibration, fine-tuning, a pending replication protocol, and audit history. The transparency is valuable, but a shorter paper centered on H1–H3 would make the evidentiary boundary clearer.

2. **Some terminology leaks from internal artifacts.** Figure 4 and Figures 13–15 explain that “oracle” means CENTROID, while Appendix E.1 mentions “BLOB probes” without defining BLOB as one of the six submitted policies. These labels are distracting and make provenance harder to follow.

3. **Some declarative headings overstate carefully qualified prose.** Examples are “Aim, not coverage” (Section 5.2), “The advantage concentrates where labels are scarce” (Section 5.4), and “Diffuse attribution supports the main claim” (Appendix J). The underlying paragraphs are more cautious than their headings.

## 4. Questions for the authors

1. Can the headline RANDOM–ENVELOPE and RANDOM–CENTROID comparisons be evaluated once on an untouched external or temporal cohort, with policies, checkpoints, probes, and analysis frozen before evaluation?

2. What are the completed results of the precommitted continuation replication in Appendix B? In particular, do all three paired-by-seed differences retain their sign?

3. Can a corrected COVER arm be retrained from the shared ancestor at multiple coverage floors, with coverage measured after collation, before making the “not how much to cover” claim?

4. Can target shape be isolated using the design proposed in Appendix D: matched delivered mask ratio, batch context, approximately 159 loss slots, guide provenance, precision, collation, and endpoint?

5. What is MIRAGE’s segmentation accuracy on FairVision, and is there any training-data overlap or domain relationship that could affect guide quality? Why were different guide variants used across the segmenter-driven arms?

6. Please reconcile the race-probe counts: are race summaries available for 19 probes or 23? Also define “BLOB” and replace all internal artifact names with submitted policy names.

7. Can the low-label claim be tested directly with paired uncertainty on the difference-in-gains across fractions, rather than inferred from point-estimate spacing?

## 5. Unsupported claims and internal numerical cross-check

### Claims not supported by the evidence shown

- **“Not how much to cover” / “anatomical coverage does not.”** This is unsupported because COVER did not deliver its intended coverage and no corrected or dose-response experiment exists (title; Abstract; Sections 5.2, 6, and 7; Appendix G).

- **“Six masking policies ... identical in every other respect.”** The opening of Section 3 is contradicted by different collation, mask ratio, delivered context, loss slots, guide provenance, and unequal stopping horizons documented in Sections 4–6, Tables 2 and 5, and Appendix P.

- **A causal effect of anatomical precision or target shape.** The paper can say the submitted ANATOMY-V2 continuation did not beat ENVELOPE or RANDOM at the specified checkpoints, but the design cannot attribute that result to precision or shape (Section 5.2).

- **“The advantage concentrates where labels are scarce.”** Table 10 shows a larger point-estimate gap at low label fractions, but no test of the change in gap across fractions is reported; Section 5.4 itself acknowledges this.

- **A deployment-ready sensitivity improvement.** Table 11 is informative, but the shared threshold yields different realized specificities, and there is no untouched cohort. The result is exploratory rather than evidence of clinical utility.

### Internal numerical contradiction

There is one clear count contradiction:

- Appendix A says **23** valid probes have joined subgroup metadata but only **19** have a usable race summary.
- Figure 5’s caption likewise says its race-gap plot uses **19 probes** carrying a race summary.
- In contrast, Table 6 labels the race analysis as per-checkpoint **n=23**, reports Black as worst in **23/23**, and Appendix E says Black is lowest in all **23 probes**. The reported Spearman result (`ρ=+0.473`, `p=0.0225`) is also presented under the n=23 heading.

The paper must state which four probes lack usable race summaries and recompute or relabel the affected count, trend, and unanimity claim.

### Checks that are internally consistent

- Table 1’s rounded headline deltas agree with its AUCs: 0.8761−0.8641=0.0120, 0.8807−0.8746=0.0061 from rounded entries (compatible with the reported 0.0062 from unrounded values), and 0.8855−0.8746=0.0109. Table 14 supplies the corresponding unrounded-analysis contrasts and intervals.
- Table 10 agrees with Section 5.4: 0.8335−0.7839=0.0496 at 5% labels, and 0.8856−0.8748=0.0108 at full labels.
- Table 11 agrees with Appendix K: 0.7428−0.7162=0.0266 and 0.7851−0.7701=0.0150.
- Table 15’s largest displayed fp16/fp32 change is 0.000192, consistent with the stated bound of at most approximately 2×10^-4.
- The COVER decline from the printed rounded values is 0.8647−0.8577=0.0070, while Section 5.1 reports 0.0071; this is compatible with unrounded values and is not, by itself, a contradiction.

## 6. Official scores

- **Quality: 2/4.** The paired evaluation and auditing are strong, but adaptive test reuse and failure to identify coverage/shape substantially weaken technical soundness for the headline claim.
- **Clarity: 3/4.** The writing, tables, and caveats are generally clear, though the paper is overly diffuse and contains inconsistent probe counts and internal artifact terminology.
- **Significance: 3/4.** A cheap segmentation-free masking policy and a controlled warning against assuming anatomical specificity are potentially useful for medical SSL, but evidence is confined to one retrospective glaucoma dataset and the effect is small.
- **Originality: 2/4.** The controlled OCT/I-JEPA measurement and geometry audit are useful, but the paper itself shows that random masking is a strong baseline and that informed masking is not uniformly better are already established findings (Section 2; Table 13).
- **Overall: 3/6 (Weak Reject).** Calibrated as a non-archival workshop paper, the single-run design alone would not prevent acceptance, but the defective coverage arm, unresolved cross-family confounds, and adaptively reused test split leave the full central claim unsupported at submission.
- **Confidence: 4/5.** I read the complete 34-page submitted artifact, including all appendices, and cross-checked the central tables and prose; remaining uncertainty concerns domain-specific guide behavior and unreported artifacts rather than the stated design.

## 7. Clinically actionable today

**No.** The study is retrospective and restricted to one repeatedly inspected test split, has no untouched external or prospective validation, does not establish stable gains across retraining, and shows substantial subgroup threshold-transfer disparities. The authors also explicitly state that no reported model is clinically validated or intended for deployment (Section 6; Appendix L).
