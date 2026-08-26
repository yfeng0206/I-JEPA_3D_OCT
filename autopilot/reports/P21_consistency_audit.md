Agent is idle (waiting for messages). agent_id: 6c4899f6-6fec-49e1-b552-55a8b23cee61, agent_type: code-review, status: idle, description: Exhaustive consistency audit, elapsed: 969s, total_turns: 1, model: gpt-5.6-sol

[Turn 0]
I found 24 issues: 21 factual/statistical contradictions, 1 stale cross-reference, and 2 draft/layout issues. The three most serious are the omitted anatomy-v1 result that contradicts the central H2 conclusion, the misdescribed operating-point protocol, and the false cross-family “only difference” claims.

I could not write `C:\Users\Gary\Desktop\jepa\autopilot\reports\P21_consistency_audit.md` because this review environment is read-only. Complete report follows.

## Factual contradictions and stale claims

1. **The central “anatomical precision does not help” conclusion omits a matched result that supports H2**
   - **Location [MEASURED]:** `C:\Users\Gary\Desktop\jepa\paper\genai4health2026\main_submission.tex:53-54, 113-116, 414-421, 699-702`; PDF pp. 1, 2, 6, 9.
   - **Statements [MEASURED]:** “anatomical \emph{precision} does not help”; “\textbf{H2 fails}”; and “the benefit does not increase with anatomical precision.”
   - **Contradicting evidence [MEASURED]:** `D:\jepa_phase0\autopilot_out\p1_stats\p1c_stats.json`, contrast `anatomy-v1@ep30@fp32 - envelope@ep30@fp32`, reports anatomy-v1 AUC 0.858274 versus envelope 0.853917, difference `+0.004357`, bootstrap CI `[+0.000930,+0.007827]`, DeLong `p=0.0127`, BH `q=0.0186`. This is matched on epoch and precision and favors the anatomy arm.
   - **Correct reading [INFERRED]:** Results are mixed across anatomy versions and epochs: anatomy-v1 supports H2 at epoch 30, while anatomy-v2 refutes it at epoch 50. The broad conclusion is false.
   - **Minimal fix [INFERRED]:** Replace “H2 fails” and equivalent broad claims with:  
     **“Results for anatomy-shaped masking were mixed: anatomy-v1 exceeded envelope at matched epoch 30, whereas anatomy-v2 fell below envelope at matched epoch 50. Because these are different implementations and epochs, the study does not support a uniform claim that anatomical precision helps or hurts.”**

2. **Multiple unscoped “only difference” and “matched compute” claims contradict the acknowledged cross-family confounds**
   - **Location [MEASURED]:** `main_submission.tex:44-47, 115-121, 221-223, 408-410, 697-701, 1300-1302`; PDF pp. 1, 2, 3, 6, 9, 21.
   - **Statements [MEASURED]:** “differing \emph{only} in how predictor targets are placed”; “hold everything else fixed”; “They differ only in placement and shape”; “differ only in masking policy”; “Under matched compute”; and “the masking policy as the only difference.”
   - **Contradicting statements [MEASURED]:** `main_submission.tex:498-522` says the anatomy arm has a 21.4% mask ratio versus 40–46%, 67.9% context versus 40–46%, 64 loss slots versus 158–160, and different collation. Lines 518-522 explicitly say “The two families are collated differently.” Lines 947-952 repeat this. PDF pp. 7, 17–18.
   - **Additional contradiction [MEASURED]:** Lines 221-222 say every policy produces four targets, while lines 929-933 report that only 32.5% of cover images retain all four rectangles and the delivered mean is 2.51 of 4.
   - **Correct statement [MEASURED]:** Lines 299-302 already contain the accurate scope: “Within the rectangle family the only difference between arms is the mask sampler; the anatomy family additionally differs in collation.”
   - **Minimal fix [INFERRED]:** Use throughout:  
     **“All arms share an ancestor, optimiser schedule, and batch size. Within the rectangle family, the sampler is the sole configured difference; anatomy-family comparisons additionally differ in collation, realized mask ratio, context budget, and loss slots.”**  
     Replace “matched compute” with “matched schedule and effective batch size.”

3. **The operating-point experiment is described as per-arm threshold selection, but the guided arms use thresholds borrowed from the random arm**
   - **Location [MEASURED]:** `main_submission.tex:1162-1166, 1170-1172, 1177-1193`; PDF pp. 19–20.
   - **Statements [MEASURED]:** “For each arm the threshold is selected on the validation split at a fixed target specificity”; “The threshold is chosen on validation to hit the target specificity”; and “two to three more cases detected per hundred at the same false-positive rate.”
   - **Evidence [MEASURED]:** In `D:\jepa_phase0\autopilot_out\p1_stats\p8b_operating_points.json`, the envelope and intensity rows say `threshold_source: "shared validation split (arm's own not stored)"`; all three arms use thresholds 0.485595703125 and 0.55517578125. Only random uses its own validation threshold.
   - **Further numerical contradiction [MEASURED]:** The prose says a 0.90 target yields test specificity 0.8794–0.8807, omitting centroid’s 0.8696 in `auto\table_operating.tex:6,8,10`. Thus the true range is 0.8696–0.8807. At 0.90, random has 185 false positives and centroid 200, so the sensitivity difference is not at the same false-positive rate.
   - **Correct reading [INFERRED]:** These are shared-random-threshold comparisons, not per-arm fixed-specificity comparisons.
   - **Minimal fix [INFERRED]:** Replace the protocol text with:  
     **“Thresholds are selected on the random arm’s validation split at target specificities 0.85 and 0.90, then applied unchanged to every arm and the test split. These are shared-threshold comparisons, not comparisons at identical achieved false-positive rates.”**  
     Replace the range with **“0.8696–0.8807.”**

4. **The contribution claims a single evaluation protocol, while the setup explicitly denies one**
   - **Location [MEASURED]:** `main_submission.tex:124-127` and `331-340`; PDF pp. 2 and 5.
   - **Statements [MEASURED]:** “a single evaluation protocol” versus “We therefore do \emph{not} claim a single protocol across every probe in the study.”
   - **Evidence [MEASURED]:** The latter is correct: random, centroid, and envelope long-horizon probes use fp16, while anatomy, cover, and ancestor probes use fp32.
   - **Minimal fix [INFERRED]:** Replace “a single evaluation protocol” with:  
     **“a common frozen MeanPool evaluation architecture, with comparisons partitioned by numerical precision.”**

5. **“21 frozen probes” is stale; 22 is correct only for the subgroup subset, not the whole inventory**
   - **Location [MEASURED]:** `main_submission.tex:124-127`; PDF p. 2. The abstract and subgroup sections use 22 at lines 68-69 and 556-560; PDF pp. 1 and 8.
   - **Statements [MEASURED]:** “21 frozen probes” versus “a subgroup audit over 22 probes.”
   - **Evidence [MEASURED]:** `p1b_full_inventory.json` contains 43 records: 6 fine-tuning records and 37 frozen-probe records. Of the 37 frozen records, 31 have `status: "primary"` and 6 are excluded/retracted. This agrees with `auto\auto_numbers.tex:215-217`: `\Nprobes=31`, `\NprobesSub=22`.
   - **Correct count [MEASURED]:** The study currently has **31 valid frozen probes, or 37 frozen inventory rows including six excluded/retracted runs**. The number **22 is correct only for the current subgroup-trend artifact**. The number 21 is not correct.
   - **Minimal fix [INFERRED]:** Replace “(21 frozen probes)” with:  
     **“(31 valid frozen MeanPool probes; 37 inventory rows including six excluded or retracted runs).”**

6. **The subgroup appendix falsely says its 22 probes are exactly all inventory-valid probes**
   - **Location [MEASURED]:** `main_submission.tex:783-789`; PDF p. 15.
   - **Statement [MEASURED]:** “The 22 probes used here are exactly those the inventory marks valid.”
   - **Evidence [MEASURED]:** `p1b_full_inventory.json` has 31 primary frozen probes. `p7b_gap_trend.json` has 22 rows and lists eight collapsed fp32 re-probes. After collapsing those eight, 23 distinct valid encoders remain, not 22. The missing one is the valid clean `frozen_meanpool_blob_fp32_ep75` anatomy-v2 probe.
   - **Correct reading [INFERRED]:** The subgroup analysis uses a 22-probe subset and is not exhaustive over valid distinct encoders.
   - **Minimal fix [INFERRED]:** Preferably add the clean anatomy-v2 epoch-75 probe and recompute the audit with 23 probes. Otherwise replace the sentence with:  
     **“The subgroup artifact contains 22 distinct-encoder probes after excluding six invalid runs and collapsing eight precision re-probes; the clean anatomy-v2 epoch-75 probe is not present in that artifact.”**

7. **Epoch 50 does not carry every comparison involving anatomy or cover**
   - **Location [MEASURED]:** `main_submission.tex:312-315`; PDF p. 5.
   - **Statement [MEASURED]:** “Epoch 50 … carries every cross-policy comparison that involves the anatomy or coverage arms.”
   - **Contradicting evidence [MEASURED]:** Table 1 and lines 443-446 report cover-versus-random comparisons at epochs 75 and 100. Table 1 also reports anatomy-v2 versus random at epoch 75.
   - **Correct statement [MEASURED]:** Epoch 50 is only the epoch with all five analyzed policies simultaneously available; it is not the only epoch used for pairwise anatomy/cover comparisons.
   - **Minimal fix [INFERRED]:** Replace with:  
     **“Epoch 50 is the only epoch at which all five analyzed policies are simultaneously available and therefore the only epoch used for the five-policy cross-sectional comparison; later pairwise comparisons use the arms available at those epochs.”**

8. **The ROC caption says only three arms reached epoch 100, but four did**
   - **Location [MEASURED]:** `main_submission.tex:1228-1235`; float appears on PDF p. 28.
   - **Statement [MEASURED]:** “The three arms that reach the full horizon are nearly superimposed.”
   - **Contradicting evidence [MEASURED]:** Lines 303-312 and `auto\table_allprobes.tex:17` show random, centroid, envelope, and cover all reach epoch 100.
   - **Correct reading [INFERRED]:** The figure plots three selected full-horizon arms and omits cover, whose AUC is materially lower.
   - **Minimal fix [INFERRED]:** Replace with:  
     **“The three plotted fp16 long-horizon arms are nearly superimposed. Cover also reaches epoch 100 but is omitted from this panel.”**

9. **The anatomy-v2 continuation sentence implies both excluded probes were replaced**
   - **Location [MEASURED]:** `main_submission.tex:304-309`; PDF p. 5.
   - **Statement [MEASURED]:** “its epoch-75 and epoch-92 probes … are excluded and replaced by a clean fp32 continuation.”
   - **Evidence [MEASURED]:** `p1b_full_inventory.json` contains excluded earlier-splice probes at epochs 75 and 92, one clean primary replacement at epoch 75, and no clean epoch-92 replacement.
   - **Correct reading [MEASURED]:** Only epoch 75 was replaced.
   - **Minimal fix [INFERRED]:** Replace with:  
     **“The earlier-splice epoch-75 and epoch-92 probes are excluded; epoch 75 was replaced by a clean fp32 continuation, and no clean epoch-92 probe was run.”**

10. **Cover is not the study’s only negative result**
    - **Location [MEASURED]:** `main_submission.tex:436-442`; PDF p. 6.
    - **Statement [MEASURED]:** Cover “produces the study’s one \emph{negative} result.”
    - **Contradicting evidence [MEASURED]:** `auto\auto_numbers.tex:68-71, 86-93` reports anatomy-v2 versus random at epoch 75 as `-0.0111`, CI `[-0.0181,-0.0042]`; cover is also negative at epochs 75 and 100.
    - **Correct reading [MEASURED]:** There are at least three matched negative contrasts: anatomy-v2 at 75 and cover at 75 and 100.
    - **Minimal fix [INFERRED]:** Replace “the study’s one negative result” with **“a negative trajectory”** or **“one of the study’s negative matched contrasts.”**

11. **Table 1’s fp32-null note and “every headline claim” statement are stale**
    - **Location [MEASURED]:** `main_submission.tex:356-366`; PDF p. 6.
    - **Statements [MEASURED]:** “The upper block … carries every headline claim”; “no fp32 null exists at [epoch 75] yet”; and “the cover epoch-75 delta is marked \(\ddagger\).”
    - **Contradicting evidence [MEASURED]:** `auto\table_allprobes.tex:33` and `auto\table_fp32.tex:11` contain a random epoch-75 fp32 result. No `\ddagger` appears in the cover table row. The abstract and conclusion also make headline claims from the fp32 lower block about anatomy-v2 and cover.
    - **Correct reading [MEASURED]:** Current lower-block deltas can all be matched on epoch and precision.
    - **Minimal fix [INFERRED]:** Replace the affected caption text with:  
      **“The upper block contains the nine primary fp16 contrasts. The lower block is fp32; all deltas use an fp32 random probe matched on epoch and precision and support the anatomy-v2 and cover trajectory claims.”**

12. **The “Full fp32 re-probe” is incomplete, and its accuracy claims are numerically overstated**
    - **Location [MEASURED]:** `main_submission.tex:134-136, 617-622, 1371-1389`; PDF pp. 2, 8, 21–22.
    - **Statements [MEASURED]:** “a \(10^{-5}\) reproduction”; “A full re-fit”; “Full fp32 re-probe”; “two to three orders of magnitude below”; and “Rows appear as the re-probes complete.”
    - **Evidence [MEASURED]:** `p3b_fp32.json` says `n_available: 8`, `n_expected: 9`, with `frozen_meanpool_oracle_ep75_fp32` pending. `auto\table_fp32.tex` consequently has eight rows and no centroid epoch-75 row.
    - **Numerical evidence [MEASURED]:** The largest fp16/fp32 shift is 0.000192. Against the six primary null-relative effects, 0.0062–0.0120, that is roughly 32–62 times smaller, not two to three orders of magnitude. The epoch-100 headline contrast changes by about \(9.6\times10^{-5}\), which is \(10^{-4}\)-level rather than within \(10^{-5}\).
    - **Minimal fix [INFERRED]:** Complete the ninth re-probe and regenerate the table. Until then, use:  
      **“Eight of nine planned fp32 re-probes are complete; centroid epoch 75 is pending. The largest observed shift is \(1.92\times10^{-4}\), more than 30 times smaller than the six primary null-relative effects.”**  
      Replace “\(10^{-5}\) reproduction” with **“\(10^{-4}\)-level reproduction.”**

13. **Centroid does not give the largest null-relative gain in the study**
    - **Location [MEASURED]:** `main_submission.tex:503-508`; PDF p. 7.
    - **Statement [MEASURED]:** “\ArmBest{} … gives the largest gain in the study.”
    - **Evidence [MEASURED]:** `auto\auto_numbers.tex:96` gives envelope at epoch 50 as `+0.0120`. Centroid’s gains are `+0.0099`, `+0.0113`, and `+0.0109` at epochs 50, 75, and 100 (`auto_numbers.tex:120-130`).
    - **Correct reading [MEASURED]:** Envelope at epoch 50 has the largest null-relative gain; centroid has the highest epoch-100 AUC.
    - **Minimal fix [INFERRED]:** Replace with:  
      **“\ArmBest{} places 41.1% on tissue, gives a +0.0099 gain at epoch 50, and achieves the highest epoch-100 AUC. Envelope’s +0.0120 at epoch 50 is the largest null-relative gain.”**

14. **“Same worst-served group every time” overgeneralizes a result that fails for language and age**
    - **Location [MEASURED]:** `main_submission.tex:68-69, 552, 1249-1252`; PDF pp. 1, 8, 21.
    - **Statements [MEASURED]:** “finds the same worst-served group every time”; subsection title “the ordering never changes”; and “Every masking policy we tested leaves the same groups worst-served.”
    - **Contradicting evidence [MEASURED]:** `auto\table_subgroup_trends.tex:10,12` says the same worst group occurs for language in only 18/22 probes and for age in 21/22. It is 22/22 for sex, race, ethnicity, marital status, and severity.
    - **Minimal fix [INFERRED]:** Replace the broad claim with:  
      **“The worst-served group is invariant across all 22 probes for sex, race, ethnicity, marital status, and severity; it changes in four language probes and one age probe.”**  
      Rename the subsection **“Subgroups: most orderings are stable.”**

15. **The severity-table caption commits the exact between-strata inference the text rejects**
    - **Location [MEASURED]:** `main_submission.tex:841-844` versus `859-864`; PDF p. 16.
    - **Statements [MEASURED]:** Caption: “Every stratum improves, and mild disease improves most.” Later text: “we did not test the paired difference between strata, so it does not establish that mild disease benefits most.”
    - **Evidence [MEASURED]:** `auto\table_paired_subgroup.tex:5-7` gives three intervals against zero but no paired contrast between the three gains.
    - **Correct statement [MEASURED]:** Each stratum improves; mild has the largest point estimate; differential benefit is unresolved.
    - **Minimal fix [INFERRED]:** Replace the caption sentence with:  
      **“Every stratum improves; mild disease has the largest point estimate, but between-stratum differences in gain were not tested.”**

16. **The limitations and ethics sections still describe an AUC-only audit despite the operating-point analysis**
    - **Location [MEASURED]:** `main_submission.tex:883-890, 1249-1257`; PDF pp. 17 and 21.
    - **Statements [MEASURED]:** “AUC as the only metric”; “no sensitivity at a fixed specificity”; “no threshold-transfer analysis”; and “the audit is AUC-only.”
    - **Contradicting evidence [MEASURED]:** Lines 1159-1226 and Tables 8–9 report validation-selected thresholds transferred to test, overall sensitivity, specificity, PPV, NPV, Brier, ECE, and subgroup sensitivity changes. `p16_subgroup_operating.json` additionally contains subgroup PPV, NPV, Brier, and ECE values.
    - **Correct reading [MEASURED]:** The audit includes subgroup AUC and fixed-threshold sensitivity for race and sex. Some subgroup calibration and predictive-value quantities were computed, although they are not presented with comparative uncertainty in the manuscript.
    - **Minimal fix [INFERRED]:** Replace both stale passages with:  
      **“The subgroup audit reports AUC and fixed-threshold sensitivity for race and sex. It does not provide uncertainty for subgroup calibration or predictive values, thresholded results for the other attributes, or an intersectional analysis, so it remains incomplete.”**

17. **“The segmentation stage bought nothing” contradicts the positive envelope result**
    - **Location [MEASURED]:** `main_submission.tex:630-638`; PDF p. 9.
    - **Statements [MEASURED]:** The paragraph first says segmenter-guided envelope is worth `+0.0120`, then says “the segmentation stage bought nothing we could measure.”
    - **Evidence [MEASURED]:** Envelope uses MIRAGE segmentation for location and significantly beats random at all three epochs. The evidence supports “not necessary” or “no benefit beyond simpler location guidance,” not “bought nothing.”
    - **Additional contradiction [MEASURED]:** The same sentence says using a segmenter “to maximise anatomy coverage” does not add and can subtract, while lines 669-679 and 940-945 explicitly say cover never realized that coverage and cannot answer that question.
    - **Minimal fix [INFERRED]:** Replace with:  
      **“A segmentation model was not necessary: envelope improved over random, but it did not outperform the simpler segmentation-free centroid policy. The defective cover run cannot establish the effect of maximizing anatomy coverage.”**

18. **Race groups whose intervals include zero are described as improving**
    - **Location [MEASURED]:** `main_submission.tex:772-776, 816-825`; PDF p. 15.
    - **Statements [MEASURED]:** Figure caption: “\emph{every} group improves”; prose: “the black subgroup improves … a larger absolute gain than the white subgroup’s.”
    - **Contradicting evidence [MEASURED]:** `auto\table_paired_subgroup.tex:9-10` gives Black `+0.01467`, CI `[-0.00055,+0.03060]`, and Asian `+0.01558`, CI `[-0.00079,+0.03283]`. Both include zero.
    - **Correct wording [MEASURED]:** Main-text lines 572-579 already use the correct formulation: every race point estimate rises, but only White is separated from zero.
    - **Minimal fix [INFERRED]:** Replace “every group improves” with **“every group’s point estimate rises.”** Replace the Black sentence with:  
      **“The Black subgroup point estimate rises from 0.8325 to 0.8472 (+0.0147; CI [-0.00055,+0.03060]), but the interval includes zero.”**

19. **The label-efficiency conclusion is untested, and the claimed 0.0002 protocol agreement is slightly false**
    - **Location [MEASURED]:** `main_submission.tex:531-549, 1004-1021, 1041-1044`; PDF pp. 7–8 and 18.
    - **Statements [MEASURED]:** “The advantage concentrates where labels are scarce”; “The margin grows sharply”; “they now agree to within 0.0002”; and “At full supervision the four arms lie within 0.027.”
    - **Evidence [MEASURED]:** `p5_label_efficiency.json` reports means and standard deviations but no paired test comparing the arm gap across label fractions. Thus concentration is a point-estimate ordering, not an established difference.
    - **Numerical evidence [MEASURED]:** Random full-supervision AUC is 0.8748054556 versus primary 0.8745808958, a difference of 0.00022456, greater than 0.0002. The full-data spread is 0.8856443577 − 0.8585615543 = 0.02708280, greater than 0.027.
    - **Minimal fix [INFERRED]:** Use **“The observed point-estimate margin is larger with fewer labels; a paired test of the gap across fractions was not performed.”** Replace “within 0.0002” with **“within 0.0003”** and “within 0.027” with **“about 0.0271.”**

20. **“Fine-tuning narrows but does not erase the gap” is asserted without a paired test**
    - **Location [MEASURED]:** `main_submission.tex:1392-1401`; PDF p. 22.
    - **Statements [MEASURED]:** Section title “Fine-tuning narrows but does not erase the gap”; prose: “partially, but not entirely, absorbed.”
    - **Evidence [MEASURED]:** `p1b_full_inventory.json` supplies only point AUCs for the six fine-tuned heads. Neither `p1c_stats.json` nor another permitted artifact reports a paired interval or test for the fine-tuned centroid–random difference.
    - **Correct reading [INFERRED]:** Only the observed point-estimate gap is known to remain positive.
    - **Minimal fix [INFERRED]:** Rename the section **“Fine-tuning reduces the observed point-estimate gap”** and replace the conclusion with:  
      **“The observed gap is smaller under fine-tuning; without paired inference for these score vectors, we do not claim that a nonzero gap remains established.”**

21. **Probe-seed noise is not an order of magnitude below the reported effects**
    - **Location [MEASURED]:** `main_submission.tex:654-665`; PDF p. 9.
    - **Statement [MEASURED]:** “probe noise is an order of magnitude below the effects in Table 1.”
    - **Evidence [MEASURED]:** The larger reported probe-seed SD is 0.0018. The six primary null-relative effects range from 0.0062 to 0.0120, only 3.4–6.7 times larger. If all Table 1 deltas are included, the claim is even less defensible.
    - **Minimal fix [INFERRED]:** Replace with:  
      **“The larger probe-seed SD, 0.0018, is below the six primary null-relative effects by factors of approximately 3.4–6.7, but these technical replicates do not estimate pretraining variance.”**

## Stale cross-reference

22. **Operating-point metrics are sent to Appendix A instead of Appendix I**
    - **Location [MEASURED]:** `main_submission.tex:326-330`; PDF p. 5.
    - **Statement [MEASURED]:** “reporting the operating-point metrics alongside AUC rather than AUC alone (Appendix A).”
    - **Evidence [MEASURED]:** `\ref{app:allprobes}` resolves to Appendix A, which is only the frozen-probe inventory. Operating points are in `\label{app:operating}` at line 1160, Appendix I, PDF pp. 19–20.
    - **Minimal fix [INFERRED]:** Replace `Appendix~\ref{app:allprobes}` with `Appendix~\ref{app:operating}`.
    - **Reference sweep [MEASURED]:** All 57 `\ref`/`\pageref` uses resolve to defined labels; this was the only semantic mis-target found.

## Draft and layout issues

23. **The corrected-cover status still reads like an in-progress submission note**
    - **Location [MEASURED]:** `main_submission.tex:956-960`; PDF p. 18.
    - **Statement [MEASURED]:** “At submission the corrected arm is not run. Until it exists…”
    - **Evidence [MEASURED]:** `auto\auto_numbers.tex` resolves `\CoverFixedStatus` to “not run.” The scientific limitation is valid, but “At submission” and “Until it exists” are draft-state phrasing.
    - **Minimal fix [INFERRED]:** Replace the paragraph opening with:  
      **“No corrected coverage arm was run. Consequently, whether hiding most of the anatomy is harmful remains open, and nothing in this paper answers that question.”**

24. **The published-ablation table has overly aggressive wrapping that looks clipped, although its text does not actually cross the page margin**
    - **Location [MEASURED]:** `main_submission.tex:1269-1297`, especially the column specification at line 1281; Table 10 on PDF p. 21.
    - **Source [MEASURED]:** `\begin{tabular}{p{2.1cm}p{3.0cm}p{4.0cm}p{3.6cm}}`.
    - **PDF evidence [MEASURED]:** FitZ places the rightmost table text at approximately `x=395.9–499.3`, inside the body’s right edge near `x=504`. The apparent fragments are wrapped continuations: “scheduled” continues as “semantics help”; “ran-” continues as “domness”; and “non-” continues as “monotonic.” I could not reproduce literal clipping beyond the page margin.
    - **Minimal fix [INFERRED]:** Give the interpretation column more width and reduce padding/font size:
      ```tex
      \scriptsize
      \setlength{\tabcolsep}{3pt}
      \begin{tabular}{p{1.8cm}p{2.7cm}p{3.8cm}p{4.2cm}}
      ```
      This preserves portrait orientation while preventing the misleading mid-word wraps.