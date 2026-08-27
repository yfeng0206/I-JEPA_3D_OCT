# Summary of submission

This paper compares six masking policies for 3D I-JEPA pretraining on retinal OCT, with all continuations forked from one epoch-25 ancestor and evaluated using frozen MeanPool probes for glaucoma classification. The intended ladder runs from unguided rectangles, through location-biased rectangles, to targets shaped by a retinal segmenter (Section 3.2; Section 4, “Pretraining”). In the submitted runs, ENVELOPE exceeds RANDOM by \(+0.0120\) AUC at epoch 50 and \(+0.0062\) at epoch 100, while the segmentation-free CENTROID arm exceeds RANDOM by \(+0.0109\) at epoch 100 (Table 1; Appendix L, Table “Primary contrasts”). ANATOMY-V2 is indistinguishable from RANDOM at epoch 50 and below it by \(-0.0111\) at epoch 75, whereas the earlier ANATOMY-V1 implementation is \(+0.0044\) above ENVELOPE at epoch 30 (Section 5.1; Table 1). The paper therefore argues that coarse location guidance helps, that increasing anatomical precision does not reliably help further, and that the best observed endpoint uses no segmentation model.

The submission is format-compliant: the main paper occupies PDF pages 1–9, references begin on page 10, and the appendix begins on page 14 (`main_submission.pdf`, pp. 1–14). It is double-blind in the submitted artifact.

# Independent assessment (formed before reading reviews)

I read the paper and formed the following assessment before consulting the reviews.

The strongest contribution is not the full six-arm ladder but the cleaner comparison within the rectangle family. RANDOM and ENVELOPE use the same target shapes, scales, counts, optimizer, schedule, effective batch size, ancestor, probe protocol, and matched checkpoints; their intended difference is target location (Section 3.2, RANDOM and ENVELOPE; Section 4, “Pretraining”; Table 1). CENTROID adds a simple segmentation-free location heuristic and is the highest observed epoch-100 endpoint at 0.8855 AUC (Table 1). These are useful workshop-level observations, especially because negative results and inexpensive alternatives to segmentation-based guidance are relevant to a health-focused audience.

However, the paper does not establish masking-policy effects. Each policy has exactly one stochastic continuation, and the bootstrap and DeLong analyses resample test subjects while holding each trained encoder fixed (Section 5 opening; Section 6, “One pretraining run per policy”). Thus the narrow statement “these stored score vectors differ on this test set” is supported, but “guidance helps under retraining” is not.

The anatomical-precision claim has a second identification problem. At epoch 50, ANATOMY-V2 has 21.4% mask ratio, 67.9% context kept, and 64 loss slots, versus 40–46% mask ratio, 40–46% context kept, and approximately 158–160 loss slots for the rectangle arms; it also uses a different collation path (Table 2; Section 5.2, paragraphs beginning “Second” and “These are not consequences”). Consequently, the study does not isolate shape or anatomical precision. The direct ANATOMY-V2–ENVELOPE difference of \(-0.0107\) at epoch 50 is real for these two trained models, but its cause is unidentified (Section 5.1, paragraph beginning “H2 is not supported”; Table 2).

Three additional design choices reduce confidence. First, ANATOMY-V2 was stopped after its epoch-75 deficit was observed, so its epoch-100 outcome is unknown (Section 4, “Pretraining”). Second, the COVER arm has a post-placement truncation defect and does not realize the aggressive coverage intervention it was designed to test (Appendix E, “A collation defect in the coverage arm”). Third, policies, checkpoints, and analyses were chosen after repeated inspection of the same test split (Section 6, “Scope, coverage and an incomplete arm”; Appendix C, “Limitations”). These are not cured by candid disclosure. Self-criticism is valuable for interpretation, but it is not additional evidence.

My pre-review conclusion was therefore that this is a potentially useful exploratory case study with unusually good audit disclosure, but that the title and principal policy-level conclusions exceed the experimental unit and the intervention actually studied.

# Points of consensus

1. **The central inferential limitation is one continuation per policy.** I agree with R1, R2, and R3. Test-subject resampling is appropriate for comparing fixed score vectors on the paired cohort, but it cannot estimate post-fork optimization variance (Section 5 opening; Section 6, “One pretraining run per policy”). The five probe seeds reported for two fixed encoders are technical replicates and are explicitly not pretraining replicates (Section 6, same paragraph).

2. **The anatomy comparison is not a single-variable intervention.** I agree with R1 and R3 that target shape, masking ratio, context budget, number of loss slots, and collation change together (Table 2; Figure 13; Section 5.2). This is independent of the \(n=1\) problem.

3. **The health evidence is narrow.** I agree with R2 that the study is confined to one FairVision glaucoma task, one test split, and one segmentation model, with no independent scanner, population, disease, or dense retinal task (Section 4, “Data”; Section 6, “Scope, coverage and an incomplete arm”). The operating-point appendix is useful, but at the shared threshold RANDOM and CENTROID achieve different test specificities, so the sensitivity difference is not measured at a common realized false-positive rate (Appendix I, Table “Operating points”; text immediately below that table).

4. **The paper is admirably transparent but not thereby validated.** All reviewers recognize the value of the defect disclosure and calibrated limitations. I agree, particularly regarding the COVER postmortem (Appendix E). I also agree that the current artifact still contains avoidable record-keeping inconsistencies: the contribution list says “21 frozen probes,” the subgroup audit uses 23 valid probes, and the all-probe inventory contains 31 rows, without a concise reconciliation (Introduction, “Contributions”; Appendix A, Table “Every frozen probe”; Appendix C opening).

5. **The main body is readable, while the appendix is insufficiently integrated.** R4 is right that the nine-page body has a coherent question-result-reversal structure, but several appendices develop secondary projects rather than supporting the decision-critical claims. The five-page occlusion/laterality sequence is the clearest example (Appendix H, especially “A bimodal attribution curve that is an artefact, not anatomy”).

# Points of disagreement and how I resolved them

## Overall severity: R1/R2 versus R3

R1 and R2 assign 3/6 overall, whereas R3 assigns 2/6. I do not average these scores. R1/R2 are closer to my view because the paper has a real workshop-level contribution as an exploratory negative case study: the clean fixed-run ENVELOPE–RANDOM and CENTROID–RANDOM observations are consistent across epochs, and the implementation audit is informative (Table 1; Appendix E). R3 is nevertheless right that these observations cannot support the causal title “Location Beats Shape.” I resolve this as **Weak Reject**, rather than Reject: the work is close in relevance and candor, but the missing evidence requires new pretraining runs rather than a textual clarification.

## Originality: R1 versus R2/R3

R1 scores originality 2, while R2 and R3 score 3. R2/R3 are right for this workshop. The broad direction—that informed masking is not uniformly superior to random masking—is not new, as the paper itself documents (Section 2, “Evidence that informed masking is not uniformly better”; Appendix K, Table “Published ablations”). The controlled OCT/I-JEPA comparison against a trained medical segmenter, the segmentation-free CENTROID alternative, and the COVER failure analysis nevertheless constitute moderate workshop originality (Section 2 final paragraph; Section 3.2; Appendix E). This supports 3, not a claim of major methodological novelty.

## Clarity: R3 versus R1/R2 and R4

R3 scores clarity 4, whereas R1/R2 score 3 and R4 gives writing 3/4 but narrative coherence 2/4. R1/R2 and R4 are right. The central argument is understandable, but clarity is reduced by unresolved statements that matter to interpretation. In particular, “no test volume is seen during model selection” conflicts with outcome-dependent stopping and acknowledged adaptive test inspection (Section 4, “Data”; Section 4, “Pretraining”; Section 6, final limitations paragraph). The probe-count inconsistency also remains (Introduction, “Contributions”; Appendix A; Appendix C). These are not merely stylistic defects.

## Whether the submitted paper contradicts itself about ANATOMY-V2

R1 and R3 criticize the paper for saying ANATOMY-V2 is indistinguishable from RANDOM despite a significant epoch-75 deficit. That criticism is not correct for the current submission. The Discussion’s “indistinguishable” statement is explicitly about the matched epoch-50 result, while the Abstract, Introduction, setup, and Conclusion all state that ANATOMY-V2 falls below RANDOM at epoch 75 and that epoch 100 is unmeasured (Abstract; Section 1 final paragraph; Section 4, “Pretraining”; Section 6, “What it does not support”; Conclusion). The paper’s current interpretation—no harm at epoch 50, a deficit at epoch 75, and no claim about epoch 100—is internally coherent. The selective stopping concern remains valid.

## Whether H2 is directly tested

R1 says H2 is handled by invalid transitive non-significance logic and that no direct ANATOMY-V2–ENVELOPE contrast is reported. R1 is wrong for the submitted paper. Section 5.1 directly reports ANATOMY-V2 minus ENVELOPE at epoch 50 as \(-0.0107\), CI \([-0.0167,-0.0046]\), \(p=0.0006\), and separately reports the opposite-direction ANATOMY-V1 comparison at epoch 30 (Section 5.1, paragraph beginning “H2 is not supported”). The paper appropriately concludes only that a uniform advantage for anatomical precision is unsupported. I nevertheless agree with R1 that the direct contrast is causally confounded by Table 2’s budget and collation differences.

## Clinical operating-point interpretation

R2 is right on the substance and wrong about what the current paper claims. The current operating-point analysis applies one RANDOM-validation-selected numeric threshold to separately fitted heads; test specificity shifts from 0.8794 for RANDOM to 0.8696 for CENTROID while sensitivity rises from 0.7162 to 0.7428 (Appendix I, Table “Operating points”). This is not an equal-false-positive-rate comparison. However, the paper now says exactly that and calls for separately validation-selected thresholds evaluated on an untouched cohort (Appendix I, two paragraphs following the table). Therefore R2’s methodological criticism stands, but its allegation that the submission presents “two to three more cases at the same false-positive rate” is factually outdated.

## Narrative review

R4’s high-level verdict is fair: the main body is strong at sentence level but the appendix is overgrown. I give less weight to several of R4’s listed “serious” contradictions because they are absent from the current artifact, as detailed below. The remaining narrative issues are secondary to the experimental-design problem and do not drive my recommendation.

# Reviewer errors identified

The following reviewer claims are factually wrong for the submitted paper:

1. **R1’s label-efficiency protocol claim is outdated.** R1 reports an alternate full-batch protocol and a \(+0.0038\) full-label CENTROID–RANDOM margin. The submitted Appendix G explicitly specifies the same minibatch AdamW, learning rate, warmup/cosine schedule, 50 epochs, and validation selection as Table 1, and reports agreement within 0.0003 at full supervision (Appendix G, “Protocol parity”; Table “Label efficiency”). The current full-label values are 0.8748 and 0.8856, not R1’s quoted 0.8811 and 0.8850 (Appendix G, Table “Label efficiency”).

2. **R1 incorrectly says the paper predicts that continuing ANATOMY-V2 would deepen its deficit.** The current text says the opposite: “a trajectory can recover,” and its epoch-100 value is “unmeasured, not known to be worse” (Section 4, “Pretraining”).

3. **R1 incorrectly says no direct H2 contrast is reported.** The direct ANATOMY-V2–ENVELOPE epoch-50 contrast, interval, and \(p\)-value appear in Section 5.1, as noted above.

4. **R2 quotes stale label-efficiency numbers.** Its 5% values of 0.8108 and 0.7925 do not appear in the submitted table; the current values are 0.8335 for CENTROID and 0.7839 for RANDOM, a descriptive \(+0.0496\) difference (Appendix G, Table “Label efficiency”). The paper correctly says the widening across fractions was not tested (Section 5.3).

5. **R3’s ANATOMY-V2 contradiction and epoch-75 precision contradiction are absent.** The current paper distinguishes epoch 50 from epoch 75, and Table 1’s lower block states that each delta uses an fp32 null at the same epoch (Table 1 caption; Section 5.1; Section 6).

6. **R3 incorrectly says the Conclusion claims matched compute across unequal horizons.** The current Conclusion claims a matched schedule, effective batch size, and shared ancestor, not equal total compute across all reported endpoints (Conclusion). Section 4 explicitly lists the unequal horizons.

7. **R3 incorrectly says the Results call COVER the only segmenter-driven policy reaching epoch 100.** The current sentence says COVER is the only policy using the segmenter to choose *how much* to hide, which is true of the defined policies; ENVELOPE uses the segmenter only for location (Section 3.2; Section 5.1, “Carried to epoch 100 the coverage arm degrades”).

8. **R3’s subgroup-scope contradiction is overstated.** The current limitations acknowledge transferred-threshold subgroup sensitivity and state only that subgroup calibration and intersectional analysis are absent (Appendix C, “Limitations”; Appendix I, Table “Change in sensitivity”). Appendix I reports aggregate Brier/ECE and subgroup sensitivity, not subgroup calibration.

9. **R4’s serious findings N1–N5 describe an older artifact, not the submission reviewed here.** The current Conclusion no longer says mask geometry is uncorrelated with anatomy hidden (Conclusion); the epoch-50 Spearman values \(+0.80\) and \(+0.40\) agree with Table 2 (Section 5.2; Table 2); the public-weights reproduction is reported in Section 5.5; the operating-point cross-reference points to Appendix I (Section 4, “Evaluation”); and Table 1 no longer says that an fp32 epoch-75 null is missing (Table 1 caption).

Some nearby reviewer criticisms remain correct. R3 and R4 are right about the unreconciled probe counts, R3 is right about the overly broad “no test volume” statement and ENVELOPE containment language, and R4 is right that “Rows appear as the re-probes complete” is inappropriate live-document text in a frozen submission (Appendix M, Table “Full fp32 re-probe” caption).

# Is the n=1 criticism fatal?

It is fatal to the policy-level interpretation, but not to every contribution.

The shared ancestor is useful: it removes variance accumulated before epoch 25 and makes each observed post-fork difference easier to audit (Section 4, “Pretraining”). It does not provide replication after the policy intervention. Once the branches fork, policy and stochastic continuation are perfectly confounded. Repeated checkpoints are correlated measurements of the same branch, and paired test-subject intervals condition on the resulting fixed encoders (Section 5 opening; Section 6, “One pretraining run per policy”).

Accordingly, \(n=1\) is survivable only if the work is framed as an exploratory report about these particular continuations. It is fatal to “guidance helps” as an expected effect, to “the best-performing policy” as a reproducible ranking, and especially to “Location Beats Shape” as a causal conclusion. The separate cross-family confounds mean that pretraining replication alone would still not identify shape.

For this decision, the criticism is acceptance-determinative. There is no rebuttal phase, and the missing unit-level replication cannot be supplied by explanation. The paper’s candor lowers the risk of reader misinterpretation but does not increase the evidentiary sample size.

# Minimum evidence that would change my view

The minimum evidence needed to move me to acceptance is:

1. **Independent continuation replication for the clean claims.** At least three independently randomized, paired post-fork continuations per policy for RANDOM, ENVELOPE, and CENTROID, all run from the same locked epoch-25 ancestor to the same locked endpoint, with the continuation—not the test subject or probe seed—as the unit of policy inference. The paper itself identifies at least three paired continuations as the needed design (Section 6, “One pretraining run per policy”).

2. **A matched anatomy intervention if the title and H2 claim are retained.** At least one segmentation-shaped arm must match ENVELOPE on delivered mask ratio, visible context, loss slots, target count, and collation, and must receive the same replicated continuation design through a common endpoint. Otherwise the acceptable claim must be narrowed to the implemented arms rather than “location versus shape” (Table 2; Section 5.2).

3. **A genuinely untouched evaluation set for the final locked comparison.** This is necessary because the submitted test split influenced policies, checkpoints, analyses, and stopping (Section 6, final limitations paragraph). A new cohort would also begin to address R2’s external-validity concern.

Stable continuation-level effects for the first item would be enough for a workshop acceptance if the anatomical-precision claim and title were narrowed. Retaining the current title requires all three items.

# Recommendation and confidence

**Recommendation: Weak Reject.**

**Confidence: 4/5.**

The question is well chosen for GenAI4Health, the clean rectangle-arm observations are useful, and the candid negative result and COVER audit have genuine workshop value. Those considerations prevent a stronger Reject. However, the central claims concern policies, while the evidence consists of one continuation per policy; the anatomical comparison also changes several training variables simultaneously; and the test set was used adaptively (Section 5; Table 2; Section 6). The required remedy is new pretraining and evaluation evidence, not clarification. With no rebuttal phase, the submitted evidence falls below the acceptance bar.
