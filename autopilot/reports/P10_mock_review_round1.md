# P10 Mock Review Round 1 — GenAI4Health @ NeurIPS 2026

**Material reviewed:** rendered `main_submission.pdf` (16 pages: main text through p.9, then references/appendix), `main_submission.tex`, `SOURCES.md`, `auto/auto_numbers.tex`, `p1c_stats.json`, `p7b_gap_trend.json`, `p1b_full_inventory.json`, and the generating statistical scripts.

**Important review basis:** reviewers judge the PDF, not the newer TeX source. The PDF predates the latest TeX and auto-generated figures. Several problems below are therefore submission-artifact problems even where the current TeX has already been corrected.

---

# Reviewer 1 — Methodology and statistics

## Summary (3 sentences)

This paper branches six retinal-OCT I-JEPA masking policies from a common epoch-25 checkpoint and evaluates frozen volume-level glaucoma probes on the same 3,000 FairVision subjects. In the single observed continuation per policy, rectangle placement guided toward retinal tissue improves test AUC over random masking, whereas the segmentation-shaped and coverage-constrained variants are near the random arm at epoch 50. The paper supplements paired test-case inference with mask-geometry measurements and argues that coarse guidance, rather than increasingly exact anatomical targeting, explains the observed ordering.

## Score

- **Overall score:** **3/10**
- **Recommendation:** **Reject**
- **Confidence:** **5/5**

## Strengths

1. **The basic fork design is much stronger than unrelated pretraining runs.** All arms share the epoch-25 ancestor, optimiser schedule, effective batch size, and probe test cases (Experimental setup, PDF p.4 lines 134–146). This removes initialisation and dataset-split differences at the fork point and makes the study substantially more interpretable than a collection of independently sourced checkpoints.

2. **The paired case-level analysis is technically appropriate for fixed trained models.** `p1c_stats.py` stratifies each bootstrap draw by class and applies the identical resampled subject indices to every arm; all 3,000 volumes have distinct subject identifiers according to `SOURCES.md`. The DeLong implementation is the standard correlated-ROC construction, and using DeLong for the two-sided test while reporting a paired percentile-bootstrap interval is defensible.

3. **The paper explicitly states the scope of the bootstrap.** The limitation on PDF p.9 lines 272–276 correctly says that the interval covers test-set sampling, not seed-to-seed pretraining variation. That distinction is essential and too often omitted.

4. **The authors disclose rather than hide major design confounds.** Section 5.2 reports the anatomy-v2 differences in masking ratio (21.4% versus 40–46%), context retained (67.9% versus 40–46%), and predictor-loss slots (64 versus 158–160), and correctly says target shape is not identified (PDF p.7 lines 201–210).

5. **Tables 1 and 2 use matched epochs for the fp16 random/oracle/envelope family.** The nine displayed pairwise rows compare epoch 50 with 50, 75 with 75, and 100 with 100; their AUCs and paired intervals agree with `p1c_stats.json`.

## Weaknesses

1. **[FATAL] One continuation per policy cannot establish the paper's training-level causal claims.** The common ancestor controls the pre-fork state, but after the fork policy is perfectly confounded with one stochastic optimisation path. DeLong and the paired bootstrap can establish that two *fixed score vectors* differ on resampled FairVision subjects; they cannot establish “guidance helps,” “precision does not,” or a policy ranking over retraining (title; Abstract, p.1 lines 7–23; Results, p.5–6). The paper admits this only after eight pages of categorical claims.
   - **Actionable fix:** run at least three paired continuation seeds for random, envelope, oracle/intensity-band, and the decisive budget-matched anatomy control, keeping data order and all non-policy randomness paired where possible. Report seed-level arm differences and a hierarchical or paired seed analysis. If this is impossible, retitle and rewrite throughout as a single-run exploratory case study; do not call test-case significance evidence that a pretraining policy is better.

2. **[FATAL] The submitted PDF contains the exact epoch-mismatch failure the paper says it avoids.** PDF Table 3 on p.7 prints random 0.8746, oracle 0.8855, and envelope 0.8807 (their epoch-100 AUCs), anatomy-v2 0.8654 (epoch 50), and cover as `0.0000†`. The adjacent prose and Figure 3 claim a matched-epoch-50 analysis (p.6 lines 190–198; Figure 3 caption), for which the correct first three values are 0.8641/0.8740/0.8761 and cover is 0.8643. Thus the rendered evidence table mixes epochs and also contains an undisclosed broken placeholder.
   - **Actionable fix:** rebuild the PDF from the corrected TeX, then perform an automated PDF-text assertion for every table value and epoch before submission. The current TeX Table 3 is corrected, but reviewers will never see that unless the PDF is rebuilt.

3. **[MAJOR] “H2 fails” is not the hypothesis test that was performed.** H2 is defined as anatomy arms \(>\) envelope (p.4 lines 130–132), yet the main inference describes anatomy-v2 and cover as non-significantly different from random (p.6 lines 169–182). Failure to reject a difference from random is neither evidence of equivalence to random nor a direct test of H2. The artifact's epoch-50 anatomy-v2-minus-envelope estimate is -0.01068 (paired CI [-0.01677, -0.00455]) and cover-minus-envelope is -0.01178 ([-0.01701, -0.00667]), but both are cross-precision and therefore cannot identify the policy effect.
   - **Actionable fix:** make the direct anatomy-versus-envelope contrast primary, define a clinically/scientifically justified equivalence or non-inferiority margin in advance, and test it under matched probe precision and matched target area/context/loss-slot budgets. Until then, say “the single observed anatomy runs showed no gain over random,” not “anatomical precision does not help.”

4. **[MAJOR] The precision disclosure is good, but the current evidence does not resolve the confound for H2.** Random/oracle/envelope probes are fp16, while anatomy-v2/cover are fp32 (Experimental setup, p.5 lines 153–160). The \(9.8\times10^{-6}\) check on p.8 lines 245–253 re-encodes features and re-scores a *fixed trained head*; it does not measure the effect of refitting the head under fp32, which is the protocol difference actually at issue. Nevertheless, the abstract and title retain H2 as a headline while the paper simultaneously says cross-precision rows are excluded from headline claims.
   - **Actionable fix:** land the disclosed full fp32 refits before submission and make every H2 comparison fully precision matched. If they do not finish, remove H2 from the title, abstract, conclusion, and practitioner recommendation.

5. **[MAJOR] The multiplicity accounting in the PDF is factually inconsistent with the artifact.** Table 2 says \(q\) is BH “over these nine” contrasts, but `p1c_stats.py` and `p1c_stats.json` apply BH over **13** same-epoch/same-precision contrasts, including four additional fp32 comparisons. For example, envelope-vs-oracle at epoch 50 has artifact \(q_{13}=0.4457\), exactly the value printed, whereas BH over the displayed nine gives \(q_9=0.4114\). The abstract also says “All nine ... contrasts against the null exclude zero” (p.1 lines 10–11), although only six rows are against random; the other three are oracle-versus-envelope, and two of those intervals include zero.
   - **Actionable fix:** define the family before looking at outcomes, state all 13 tests or recompute over the displayed nine, and correct “nine against the null” to “six against the null.” Explain where pre-specification was recorded; declaring a family in a post-hoc script is not pre-registration.

6. **[MAJOR] Multiplicity and dependence are not handled for the subgroup trend tests.** Seven gap-versus-AUC Spearman tests are run, but the main text highlights race \(p=0.0270\) without correction; BH across the seven gives \(q=0.0945\). More seriously, the nominal \(n=21\) treats repeated epochs from the same single training branch as independent. Averaging the 19 non-spliced probes within the seven unique branches gives race \(\rho=0.429,\ p=0.337\), not evidence of a trend.
   - **Actionable fix:** treat branch/seed as the unit, use one pre-specified checkpoint per branch or a clustered permutation/bootstrap, and correct over the seven attributes. With one branch per policy, report this only as descriptive.

7. **[MAJOR] The test split has been reused adaptively.** Appendix C states “one dataset, one test split, reused throughout the programme's history” (PDF p.14 lines 500–503). Even if no probe head was selected directly on test AUC, policies, checkpoints, hypotheses, figures, and the narrative can all be selected after repeated test inspection, so “pre-specified” confirmatory \(p\)-values are not credible.
   - **Actionable fix:** evaluate the frozen final analysis on a genuinely untouched external or locked holdout, or explicitly label all current inference exploratory and reserve confirmation for a new cohort.

8. **[MAJOR] The paper overstates what geometry correlation can establish.** A sample Spearman correlation of exactly 0.00 across four rectangle policies (p.6 lines 192–198) is extremely underpowered and does not show that anatomy hidden is generally “uncorrelated” with AUC. Moreover, anatomy-v2—the most anatomically pure arm—is excluded from that \(n=4\) correlation precisely because it differs on multiple budgets.
   - **Actionable fix:** replace the causal Section 5.2 title “it is not anatomical targeting” with “geometry is not monotonic in these five observed arms,” and run controlled area/context/slot-matched interventions.

9. **[MINOR] The promised independent DeLong validation artifact is absent.** `SOURCES.md` lists `delong_validation.json`, but that file is not present under `D:\jepa_phase0\autopilot_out\p1_stats`; only the validation script exists. The implementation looks standard, but the provenance claim is currently unfulfilled.
   - **Actionable fix:** rerun `p1_validate_delong.py` against the corrected inventory, update it so it does not depend on the superseded/mislabelled master inventory, and include the resulting JSON.

10. **[MINOR] The fine-tuning statement is descriptive but written inferentially.** Section 5.3 says fine-tuning “narrows but does not erase the gap” from one mean-pool head per arm (p.7 lines 212–215), with no paired interval, multiplicity treatment across heads, or seed variation.
    - **Actionable fix:** provide the paired test-case CI and the pretraining-seed caveat, or call these two point estimates only.

## Questions to the authors

1. Where, before inspecting these test outcomes, were the nine (or 13) “pre-specified” comparisons, checkpoints, subgroup axes, and direction of tests recorded?
2. Why is H2 evaluated primarily via anatomy-versus-random when H2 is explicitly anatomy-versus-envelope?
3. Were post-fork data order, augmentation draws, optimiser state, and probe-head seeds paired across arms, or does “seed 42” refer only to the downstream probe?
4. How many times were FairVision test AUCs inspected while choosing the policy ladder, \(f=0.21\), epoch 50, the “oracle” story, and the subgroup analyses?
5. What equivalence margin would count as evidence that an anatomy policy “does not help”?

## Specific unsupported claim

> “All nine pre-specified paired contrasts against the null exclude zero.” (Abstract, PDF p.1 lines 10–11)

This is false as written. Table 2 contains only six contrasts against random; the remaining three compare oracle with envelope, and the epoch-50 and epoch-75 intervals include zero. In addition, the printed \(q\)-values were corrected over 13 artifact contrasts, not the nine shown.

---

# Reviewer 2 — Clinical and health-AI relevance

## Summary (3 sentences)

The study asks whether retinal anatomy should guide masked predictive pretraining for OCT glaucoma classification and finds that a cheap image-intensity band is at least as promising as guidance from a retinal segmentation model in the observed runs. It then audits demographic AUC disparities and glaucoma-severity strata across repeatedly probed checkpoints on FairVision. The practical message is that aggregate representation gains neither guarantee equitable gains nor solve the much harder mild-disease detection problem, although the present analyses do not yet support that message at the strength claimed.

## Score

- **Overall score:** **3/10**
- **Recommendation:** **Reject**
- **Confidence:** **5/5**

## Strengths

1. **The paper fits both workshop themes.** It studies a foundation-model-style generative/predictive imaging objective and asks a trustworthy-AI question about demographic and severity performance. This is more on-topic for GenAI4Health than a generic OCT classifier paper.

2. **The central practical comparison is relevant.** If a first-order intensity centroid performs comparably to or better than a trained segmenter-guided pipeline, that could reduce engineering complexity, segmentation-domain-shift risk, and compute. The paper clearly describes the intensity signal (Method, p.3 lines 118–120), so the reader can understand what is and is not clinically informed.

3. **FairVision is an appropriate cohort for a first subgroup audit.** The held-out test split has 3,000 subjects, including 431 Black and 251 Asian participants; metadata are not used as model inputs (p.4 lines 134–141). The deterministic metadata join is validated against all stored labels.

4. **The severity construction is conceptually correct.** Because mean deviation defines case status, within-severity-bin AUC would be undefined; comparing each positive severity stratum with the shared 1,534 negatives is a sensible way to quantify stage-specific discrimination (Appendix C, p.14 lines 493–499).

5. **The paper does not claim clinical deployment readiness.** The broader-impact paragraph correctly says that frozen-probe AUC on one public dataset is not evidence of clinical utility (p.9 lines 287–294).

## Weaknesses

1. **[FATAL] The headline fairness trend is pseudo-replicated, multiplicity-uncorrected, internally contradictory, and includes excluded probes.** The main text treats 21 checkpoints as the Spearman sample and claims a widening race gap (\(\rho=0.482,\ p=0.027\); p.8 lines 229–236), even though checkpoints share subjects, a common ancestor, and repeated epochs within one branch. `p7b_gap_trend.json` includes anatomy-v2 epochs 75 and 92, which `p1b_full_inventory.json` excludes because of the EMA-target precision splice. Appendix C instead uses “12” probes, reports \(\rho=0.427,\ p=0.167\), and explicitly says not to claim that better models are less fair (p.13–14 lines 477–492); the appendix figure uses yet another 19-probe value (\(\rho\approx0.51,\ p\approx0.026\)). Across the seven tested attributes, race also fails BH (\(q=0.0945\)).
   - **Actionable fix:** choose one provenance-clean inventory; exclude the two precision-spliced checkpoints; select one checkpoint per independent branch or cluster by training run; correct across attributes; and make the main text, appendix, and figure identical. Without seed replicates, remove the trend \(p\)-value and call the scatter descriptive.

2. **[MAJOR] “Selecting against the Black subgroup” is not what the reported AUCs show.** At epoch 100, random has Black AUC 0.8325 and oracle has 0.8472, an absolute improvement of about +0.0147; the max-min race gap changes only from 0.07084 to 0.07174 because the Asian AUC also rises. A max-min gap can widen while every group improves, so the sentence on p.8 lines 232–235 incorrectly turns unequal gains into harm to the Black subgroup.
   - **Actionable fix:** report within-group paired arm deltas and CIs, both absolute subgroup performance and disparity changes, and identify which group drives each max-min gap. Use “benefits were unequal” if supported, not “select against.”

3. **[MAJOR] The abstract's claim that better masking “did not help early disease” is contradicted by the underlying severity AUCs.** At epoch 100, mild-disease AUC is 0.8164 for random and 0.8301 for oracle (+0.0137); severe AUC rises by only +0.0063. A persistent severe-to-mild *gap* does not mean mild performance did not improve, and in this comparison the gap actually narrows from 0.1361 to 0.1286. No paired CI for these severity-specific arm deltas is presented.
   - **Actionable fix:** replace the claim with “all models remain substantially worse for mild than severe disease.” Report paired CIs for mild, moderate, and severe arm deltas and for the change in severe-to-mild gap before claiming policy invariance.

4. **[MAJOR] The severity result is clinically relevant but not evidence that masking “is not the lever.”** Mild functional glaucoma is harder to separate from normal OCT by construction, while severe disease has larger structural changes. The 0.1286–0.1400 range across highly dependent checkpoints quantifies this difficulty but does not isolate a policy mechanism, and comparing its 0.0114 spread with the aggregate-AUC spread is not a statistical test (p.8 lines 237–243).
   - **Actionable fix:** frame this as a clinically important failure mode shared by all evaluated models, not the study's “strongest evidence” against masking. Validate on a progression/early-detection endpoint or an external cohort and test stage-specific deltas directly.

5. **[MAJOR] The practitioner recommendation outruns the clinical evidence.** “A segmentation stage can be removed from this pipeline without measured loss” (p.8 lines 259–264) rests on one dataset, one glaucoma classification task, one frozen mean-pool probe, one run per policy, cross-precision anatomy comparisons, and unmatched masking budgets. Segmentation-aware pretraining might matter for layer segmentation, lesion localisation, progression, or another population even if it does not help this pooled label.
   - **Actionable fix:** narrow the statement to this single FairVision frozen-probe experiment. Ideally add an external OCT cohort and a dense downstream task before making a pipeline recommendation.

6. **[MAJOR] The fairness audit is too AUC-centric to support a trustworthy-health contribution.** There is no subgroup calibration, sensitivity at a clinically chosen specificity, predictive value, threshold transfer, or intersectional analysis. Subgroup marginal CIs overlap, but the relevant quantity—paired difference in gaps between policies—has no interval.
   - **Actionable fix:** add paired subgroup-delta and gap-delta intervals; report calibration and sensitivity at a validation-selected high-specificity threshold; include intersectional groups only where cell sizes support them. Explicitly distinguish performance auditing from bias mitigation.

7. **[MINOR] Persistence of the worst group does not show that policies did not cause or contribute to disparity.** The claim that a disparity surviving every policy is “plainly not caused by any of them” (p.8 lines 222–227) is not logically valid: every tested policy could preserve or amplify a disparity created by the data, label, site, or shared ancestor.
   - **Actionable fix:** say the ordering is robust to the tested policy changes; do not make a causal attribution.

8. **[MINOR] “Early disease” and “mild disease” should not be used interchangeably.** The strata are cross-sectional visual-field mean-deviation bins, not time since onset, conversion risk, or longitudinal progression. Mild functional loss is clinically important, but it is not necessarily early in every patient.
   - **Actionable fix:** use “mild functional disease by mean deviation” throughout and reserve “early detection” for a longitudinal or clinically defined early cohort.

## Questions to the authors

1. Are race groups confounded with acquisition site, device, scan quality, age, or disease severity in FairVision, and were any of these factors examined?
2. Does the MIRAGE segmentation model have subgroup-specific quality differences on FairVision that could mediate the masking results?
3. For random versus oracle at matched epoch 100, what are the paired CIs for Black, White, Asian, mild, moderate, and severe AUC differences?
4. Why is max-min AUC gap the fairness target rather than worst-group AUC, equal opportunity at a fixed operating point, or calibration?
5. Were the demographic and severity analyses planned before viewing the test predictions, or selected after the aggregate results were known?

## Specific unsupported claim

> “Better masking made the model better; it did not make it fairer, and it did not help early disease.” (Abstract, PDF p.1 lines 22–23)

The last clause is contradicted by the stored predictions: oracle epoch 100 raises mild-stratum AUC from 0.8164 (random) to 0.8301 (+0.0137). Whether that increase is statistically reliable has not been tested, but the presented evidence cannot support “did not help.”

---

# Reviewer 3 — Novelty, positioning, and presentation

## Summary (3 sentences)

The paper presents an OCT-specific masking-policy ladder for I-JEPA and observes that coarse retina placement and an intensity-centroid band outperform random masking in the available frozen probes, while more segmentation-shaped variants do not. It positions this as a caution against assuming that increasingly anatomical targets yield increasingly useful representations. The manuscript is candid about several confounds and has a visually intuitive story, but the conceptual novelty is incremental and the rendered submission contains severe numerical and presentation inconsistencies.

## Score

- **Overall score:** **4/10**
- **Recommendation:** **Weak Reject**
- **Confidence:** **4/5**

## Strengths

1. **The question and qualitative ladder are easy to understand.** Figure 1 gives a concrete same-slice view of all six samplers, and the random/envelope/intensity-band/anatomy distinction is legible even to readers outside I-JEPA.

2. **The related-work section is unusually candid.** It acknowledges Guo et al. (2025), where SemMAE falls below random under a common protocol; AutoMAE's “patch selection dilemma,” where modest foreground bias helps but excessive bias hurts; and Lee et al. (2025), where a simple intensity-derived foreground prior helps on 3D medical tasks (PDF p.2–3 lines 76–90). That is the right literature to discuss.

3. **The paper contains a potentially useful OCT-specific negative result.** A controlled comparison showing that a trained segmentation model is not automatically superior to a cheap image statistic would be valuable as an empirical workshop contribution if replicated and budget matched.

4. **The limitations are more transparent than average.** The manuscript names the single-run problem, the target-budget confounds, the precision mismatch, the incomplete cover trajectory, and the single-dataset scope rather than burying them.

5. **The main Tables 1–2 communicate the fp16 trajectory well.** Epochs, deltas, intervals, \(p\), and \(q\) are compactly visible, and Figure 2(b)'s use of paired-difference intervals is preferable to relying on overlapping marginal AUC intervals.

## Weaknesses

1. **[FATAL] The PDF is not a coherent frozen submission artifact.** It was built before the supplied TeX and auto figures were last modified. The most visible consequence is PDF Table 3's mixed epoch-100/epoch-50 AUCs and `0.0000†` cover entry, but the problem recurs: the main text reports 21-probe race \(\rho=0.482\), the appendix reports 12-probe \(\rho=0.427\) and the opposite inferential conclusion, and the fairness figure reports approximately 19-probe \(\rho=0.51\). Appendix Table 4 also labels random/oracle/envelope as fp32 while the main text says those long-horizon probes are fp16; “target precision” is undefined and conflicts with “protocol identical throughout.”
   - **Actionable fix:** create one versioned evidence manifest, regenerate every macro/table/figure from it, rebuild once, and run automated PDF-text checks for arm, epoch, precision, probe count, and every headline statistic. Do not submit until the PDF—not just the source—passes.

2. **[MAJOR] The high-level phenomenon is not novel.** The paper's own related work already documents both halves of the story: modest foreground bias can help while overly foreground-focused masking hurts (AutoMAE), a semantic masking method can underperform random under frozen evaluation (Guo et al.), and simple intensity foreground selection can work across medical volumes (Lee et al.). The observed ordering is therefore a useful OCT replication/controlled case study, not a discovery that overturns informed masking.
   - **Actionable fix:** position the novelty narrowly as a shared-fork OCT comparison of segmentation-derived placement versus segmentation-free intensity placement. To earn a stronger claim, add the proposed vertical-band randomisation or a budget-matched mechanism experiment that distinguishes consistency from anatomy.

3. **[MAJOR] “Oracle” is a misleading arm name.** In SSL and medical-AI papers, “oracle” normally implies ground-truth labels, privileged annotations, or an upper bound. Here it is a per-column intensity-weighted row centroid computed from the input image (Method, p.3 lines 118–120); it is neither ground truth nor an oracle.
   - **Actionable fix:** rename it `intensity-band`, `centroid-band`, or `image-derived band` everywhere, including filenames shown to reviewers. This change would immediately prevent a predictable no-rebuttal misunderstanding.

4. **[MAJOR] The title is ambiguous and too categorical.** “Precision Does Not” can mean numerical fp16/fp32 precision—the manuscript has an entire section on that issue—or anatomical specificity. It also asserts a population-level null from one run and a target-budget-confounded comparison.
   - **Actionable fix:** use a descriptive title such as “Coarse Retinal Guidance Outperforms Segmentation-Shaped Targets in a Controlled Single-Run OCT I-JEPA Study,” then strengthen it only after replication.

5. **[MAJOR] Several figures are misleading or not publication-ready.** Figure 3's left-panel labels overlap the title (“envelope / matched epoch 50”), and “unguided null” collides with anatomy-v2. Figure 2(a) plots excluded anatomy-v2 epoch-75/92 checkpoints on a line labelled `anatomy-v2 (fp32)`, although those checkpoints follow the target-precision splice; its caption says dashed lines are fp32. The “specificity ladder” and purity scatter largely duplicate one another, while the appendix fairness figure disagrees numerically with the main text.
   - **Actionable fix:** remove excluded checkpoints or mark them explicitly as confounded, fix label collisions, show uncertainty on all inferential plots, delete one redundant specificity plot, and regenerate the fairness figure from the final inventory.

6. **[MAJOR] The manuscript repeatedly calls this a six-arm epoch-100 study although the evidence is incomplete and uneven.** Experimental setup says “Every arm ... continues to epoch 100” (p.4 lines 142–146), but cover was deliberately stopped at epoch 73, anatomy-v2 has only a valid pre-splice result through epoch 50, and anatomy-v1 has only epoch 30. At the claimed common epoch 50 there are five policies, not six, and the geometry table omits anatomy-v1 despite the contribution claiming geometry “for every policy.”
   - **Actionable fix:** state exact valid checkpoints per arm in the main setup, call it six policy definitions but a five-policy epoch-50 comparison, and either add anatomy-v1 geometry/results or qualify “every.”

7. **[MINOR] “Anatomical specificity” is treated as a total ordering without a defined metric.** Cover, envelope, and the intensity band vary in location, purity, anatomy hidden, and shape; deciding that one is “more specific” than another is partly rhetorical. The non-monotonic bar chart therefore looks more mechanistic than the design warrants.
   - **Actionable fix:** define specificity quantitatively or call the x-axis policy categories rather than a ladder.

8. **[MINOR] The manuscript uses strong mechanism language after correctly admitting mechanism is unidentified.** Section 5.2 is titled “it is not anatomical targeting,” yet p.9 lines 268–271 says consistency, ratio, and task difficulty remain unresolved. This weakens trust because the caveat and headline point in opposite directions.
   - **Actionable fix:** rename the section and conclusion around “non-monotonicity” rather than mechanism.

9. **[MINOR] The provenance package is not double-blind if distributed as supplied.** `SOURCES.md` contains a local username/path and public GitHub/Hugging Face account identifiers. The PDF metadata itself is clean, but including this file in a blinded supplement or Overleaf zip would reveal identity.
   - **Actionable fix:** create a separate anonymised provenance document with neutral artifact IDs and anonymous links; keep the identifying internal ledger out of the submission package.

## Questions to the authors

1. What is the precise novelty over AutoMAE's foreground-bias “dilemma” and Lee et al.'s intensity-based medical foreground masking?
2. Why retain the term “oracle” when the policy uses only an ordinary image statistic?
3. Was the intended main contribution the negative anatomy result, the simple intensity heuristic, or the fairness/severity audit? The current paper tries to headline all three.
4. Why are the invalid anatomy-v2 epoch-75/92 points drawn in Figure 2 as fp32?
5. Which exact PDF hash will be submitted, and what automated check guarantees its tables match the final JSON and TeX?

## Specific unsupported claim

> “Every number in this paper is emitted by a single script from stored per-case predictions, so the tables, figures and prose cannot disagree.” (Reproducibility, PDF p.8 lines 255–258)

They do disagree in the rendered paper: Table 3 mixes epochs and prints cover as 0.0000; the main and appendix use 21 versus 12 subgroup probes with different race correlations and opposite conclusions; and the fairness figure uses a third correlation/probe count.

---

# Meta-review

## Reconciliation

All three reviewers agree that the paper asks a workshop-relevant question, uses a valuable shared-checkpoint design, reports fixed-model case-level uncertainty correctly, and is more transparent about confounds than most submissions. They also agree that the paper's strongest valid result is narrower than its title: in one observed fp16 continuation family, envelope and intensity-band policies produce higher FairVision test AUC than random at matched epochs. Reviewer 1 finds that this cannot support training-policy causality or equivalence because there is one run per arm, H2 is tested against the wrong comparator, and multiplicity/provenance are inconsistent; Reviewer 2 finds that the fairness and severity claims are statistically invalid or contradicted by absolute subgroup results; Reviewer 3 finds the conceptual contribution incremental and the actual PDF internally broken.

## Decision

**Reject (current submission).**

The decision is not based on the disclosed pending cover endpoint or pending full-fp32 table. It is based on three independently sufficient current-paper problems: (1) the headline policy claims have no training-seed replication; (2) the PDF contains a real mixed-epoch result table and incompatible versions of the subgroup analysis; and (3) the health-equity/early-disease conclusions use dependent checkpoints as replicates and, for mild disease, contradict the underlying AUCs. With no rebuttal, reviewers cannot assume the newer TeX or pending jobs will repair the PDF.

**Unique meta-level count:** **3 fatal objections; 10 major objections.**

## Priority fix-list

### MUST-FIX-BEFORE-SUBMISSION

1. **Freeze and rebuild one internally consistent PDF.** Use the corrected epoch-50 Table 3 values; remove `0.0000†`; reconcile 12/19/21-probe subgroup analyses; define and correct every precision label; remove or visibly quarantine the spliced anatomy-v2 points. Verify extracted PDF text against JSON.

2. **Address training-seed uncertainty or radically narrow the claims.** The preferred fix is at least three paired continuation seeds for the decisive arms. Without them, change the title, abstract, Results headings, and conclusion from “helps/does not help/best” to “in the observed single continuations,” and do not use test-case \(p\)-values as policy-level evidence.

3. **Repair H2's estimand.** Compare anatomy directly with envelope, under the same probe precision and matched mask ratio/context/loss-slot budget; predefine an equivalence margin. If this cannot be done, drop “anatomical precision does not help” as a headline and report only non-monotonic observed point estimates.

4. **Complete the precision-matched refits or remove cross-precision headline conclusions.** The fixed-head \(9.8\times10^{-6}\) check is useful but does not validate refitted-head equivalence.

5. **Correct the inferential family.** Resolve nine versus 13 BH tests, correct “nine against the null” to six, document genuine pre-specification, and treat the repeatedly reused test set as exploratory unless a fresh holdout is obtained.

6. **Rebuild the subgroup analysis around independent units.** Exclude precision-spliced probes, use one checkpoint per branch or a clustered analysis, correct across seven attributes, and remove the nominal 21-checkpoint race-trend \(p\)-value if branch-level replication is unavailable.

7. **Rewrite the race interpretation.** Report absolute group-specific changes and paired CIs. Oracle improves Black AUC at epoch 100; a slightly larger max-min gap is unequal benefit, not evidence of selecting against Black participants.

8. **Rewrite the severity conclusion.** State that all models remain much worse on mild functional disease, but acknowledge the observed oracle mild-AUC increase (+0.0137 at epoch 100). Add paired stage-specific intervals before claiming invariance or lack of benefit.

9. **Remove the pipeline-level clinical recommendation.** Restrict conclusions to one FairVision frozen mean-pool glaucoma probe unless an external cohort/dense task is added.

10. **Align the paper's scope and inventory.** State that six policies are defined, only five have epoch-50 probes, and not every arm validly reaches epoch 100. Do not claim geometry for “every policy” if anatomy-v1 is omitted.

### NICE-TO-HAVE

1. Rename `oracle` to `intensity-band` or `centroid-band`.
2. Retitle to make “anatomical specificity” unambiguous and the single-run scope honest.
3. Replace one redundant specificity figure with a budget-matched design schematic; fix all label collisions.
4. Add calibration and sensitivity at a validation-selected clinical specificity, plus paired subgroup-gap intervals.
5. Validate on an external OCT cohort and, ideally, a dense retinal task.
6. Regenerate and include the missing DeLong-validation JSON.
7. Provide an anonymised provenance supplement; do not ship identifying local paths/account names.

## Estimated workshop acceptance probability

**18%.**

The topic is an excellent GenAI4Health fit, the negative result is potentially useful, the shared-fork design is appealing, and the authors' transparency could attract a sympathetic reviewer. However, the present PDF has a reviewer-verifiable epoch mismatch and incompatible analyses, the central training-policy inference is \(n=1\) per arm, and the fairness/severity story—the main trustworthy-AI hook—does not survive appropriate unit-of-analysis and absolute-performance checks. A fully rebuilt, carefully narrowed paper with matched fp32 probes could become borderline; seed replication plus a clean health analysis could move it into weak-accept territory.
