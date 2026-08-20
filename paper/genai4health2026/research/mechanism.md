# Mechanism: why near-pure anatomy targeting can fail

## Evidence labels

- **[MEASURED]** is an observed value from a stored artifact or documented run.
- **[INFERRED]** is an interpretation or arithmetic consequence of measured values.
- **[ASSUMED]** is a causal premise not identified by the current experiments.

## Verdict

**[MEASURED] Background-position tokens carry readable glaucoma signal:** the strongest background-only regional probe reached test AUC **0.870075** for the envelope ep50 encoder. **[LIMITATION]** This does not establish signal in optically black pixels: ViT tokens have global receptive fields, the region mask recall is **0.89340**, and a separately fitted head was used for each pool [S2, S3].

**[MEASURED] Background targets also provide nontrivial self-supervised supervision:** random ep100 removed **67.98%** of the background prediction error left by a per-position, no-context reference [S4]. **[MEASURED]** However, count-matched context removal did not isolate a positive background-content effect beyond generic token removal; healthy models valued one anatomy context token roughly **4–6.6 times** as much as one background-position token [S5, S3].

**[INFERRED — moderate confidence]** The most defensible mechanism is therefore not “black pixels directly encode glaucoma.” It is that the blob policy creates an underconstrained, compositionally narrow prediction task—near-pure anatomy targets, only 64 target slots, connected-blob geometry, replacement padding, and about 160 context tokens—and the predictor subsequently drifts toward a weak/positional solution. Excluding background targets is one plausible contributor because healthy predictors learn substantial context-dependent variation on those targets, but it is not isolated from the other changes.

**[INFERRED — low confidence]** “Anatomy starvation in context” is not sufficient as the explanation. Blob has the lowest anatomy *percentage* in context (**6.26%**) but still has **9.97 anatomy cells** in context on average, more than envelope (**8.63**) and COVER f0.21 (**9.28**), because its total context is much larger. Blob also has the lowest zero-anatomy rate (**1.24%**) [S1]. The percentage-only argument is therefore undermined; a chronic supervision/shortcut account is better supported.

## 1. The causal question is confounded at the intervention

**[MEASURED]** The intended intervention is target placement, but the effective blob intervention changes at least target composition, geometry, target count, duplicate mechanism, hidden fraction, and context budget [S1, S6, S7].

| arm | anatomy hidden % | anatomy target purity % | context anatomy % | anatomy cells in context | total context | zero-anatomy context % | AUC ep50 | AUC ep100 | status and provenance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random | 53.0445 | 31.5832 | 26.3785 | 18.2526 | 69.0909 | 3.6826 | 0.8641 | 0.8746 | **[MEASURED]** [S1: `rows[arm=random]`; S8] |
| oracle | 61.5847 | 39.6878 | 19.3439 | 14.9281 | 77.2257 | 4.1877 | 0.8740 | 0.8855 | **[MEASURED]** [S1: `rows[arm=oracle]`; S8] |
| COVER f0.21 | 73.0945 | 40.8776 | 14.5550 | 9.2765 | 63.6166 | 7.8377 | pending | — | **[MEASURED]** [S1: `rows[arm=cover_f021]`] |
| envelope | 77.5822 | 43.1927 | 11.3839 | 8.6345 | 76.4109 | 8.0658 | 0.8761 | 0.8807 | **[MEASURED]** [S1: `rows[arm=envelope]`; S8] |
| blob | 82.0662 | 97.5011 | 6.2635 | 9.9668 | 159.9974 | 1.2386 | 0.8654 | — | **[MEASURED]** [S1: `rows[arm=blob]`] |

**[LIMITATION]** The “one identical 6,137-slice pass” description does not apply to blob. The script states that random/oracle/envelope/COVER come from the 6,137-slice sweep, while blob comes from a separate 1,534-slice arm-statistics pass; the output itself labels blob `arm_stats(n=1534)` [S1, S9].

**[MEASURED]** In a separate 500-slice target-composition audit, blob supplied **64.000** target slots and **54.744** unique target cells, versus envelope’s **154.624** slots and **118.314** unique cells. Blob alone used `pred_target_k: 16`, whose shortfall is padded with replacement [S6, S7].

**[INFERRED]** The ep50 curve is non-monotonic in purity: AUC rises from 31.6% to 43.2% purity and falls at 97.5%. **[LIMITATION]** Four completed points, one pretraining seed per arm, and simultaneous movement on several axes cannot identify an optimum or a causal dose response.

## 2. Does non-anatomy/background carry glaucoma signal?

### 2.1 Downstream disease signal

**[MEASURED]** The region experiment used the same frozen encoder, 2,000 training, 600 validation, and 1,000 test volumes, with 25 stratified slices per volume; only the pooled token positions changed [S3].

| rank among all arm-region cells | frozen encoder | pooled positions | test AUC | all-cell AUC | interpretation | provenance |
|---:|---|---|---:|---:|---|---|
| 1 | envelope ep50 | anatomy | **0.8784438** | 0.8729913 | **[MEASURED]** highest region-pooled AUC | [S2: `tag=envelope_ep50.anatomy.test`] |
| 2 | random ep50 | anatomy | **0.8746555** | 0.8608341 | **[MEASURED]** | [S2: `tag=random_ep50.anatomy.test`] |
| 3 | oracle ep50 | anatomy | **0.8746475** | 0.8682588 | **[MEASURED]** | [S2: `tag=oracle_ep50.anatomy.test`] |
| 4 | envelope ep50 | background | 0.8700750 | 0.8729913 | **[MEASURED]** strongest background-only result | [S2: `tag=envelope_ep50.background.test`] |
| — | oracle ep50 | background | 0.8652265 | 0.8682588 | **[MEASURED]** | [S2: `tag=oracle_ep50.background.test`] |
| — | blob ep50 | anatomy | 0.8606021 | 0.8593499 | **[MEASURED AUCs; INFERRED arithmetic]** +0.001252 over all | [S2: `tag=blob_ep50.anatomy.test`] |
| — | blob ep50 | background | 0.8590499 | 0.8593499 | **[MEASURED]** nearly equal to all | [S2: `tag=blob_ep50.background.test`] |
| — | random ep50 | background | 0.8543614 | 0.8608341 | **[MEASURED]** | [S2: `tag=random_ep50.background.test`] |

**[MEASURED]** Anatomy-only pooling improved over all-cell pooling for every encoder. **[INFERRED]** The three highest arm-region entries all use anatomy positions, so the available region experiment supports anatomy as the most efficient readout region.

**[MEASURED]** Background-only AUC remains high for every encoder. **[INFERRED]** This establishes disease readability from background *positions*, not disease information in black pixel content. **[LIMITATION]** The experiment has no inner-retina, RNFL, optic-disc, or choroid-specific AUCs; it cannot rank finer clinical subregions.

### 2.2 Self-supervised background target signal

| question | result | status | provenance |
|---|---|---|---|
| Can context beat a position-only predictor on background targets? | random ep100 background skill **0.679809**; healthy checkpoint range **0.584979–0.679809** | **[MEASURED]** yes | [S4: `tag=random_ep100.bg.skill_vs_pos`; all healthy tags] |
| Is background target variation collapsed to a constant? | random ep100 background effective rank **22.51**, anatomy **12.76** | **[MEASURED]** no | [S5: `tag=random_ep100.effr_bg`, `effr_anat`] |
| Is one background context token as valuable as anatomy? | healthy anatomy/background marginal-value ratios **3.94–6.60** | **[MEASURED]** no | [S5: healthy rows, `ratio`] |
| Does the removal test isolate semantic black-pixel content? | background-specific excess versus count-matched random removal is negative and within about two SEM in healthy checkpoints | **[MEASURED]** no | [S3: lines 188–199] |

**[INFERRED]** Background prediction is a real contextual task and may regularize the encoder through full-field constraints. **[ASSUMED]** That regularization transfers to glaucoma discrimination; no single-variable pretraining ablation has tested it.

## 3. What changed inside the blob model?

| blob checkpoint | full-context error | anatomy/background marginal token value | anatomy excess over count-matched random removal | status and provenance |
|---|---:|---:|---:|---|
| ep30 | 0.104868 | 3.529× | 0.007768 ± 0.001210 | **[MEASURED]** [S5: `tag=blob_ep30`] |
| ep40 | 0.224373 | 0.998× | 0.001906 ± 0.001817 | **[MEASURED]** [S5: `tag=blob_ep40`] |
| ep50 | 0.271158 | 0.830× | 0.000573 ± 0.002099 | **[MEASURED]** [S5: `tag=blob_ep50`] |
| ep56 | 0.289461 | 0.737× | 0.000624 ± 0.002264 | **[MEASURED]** [S5: `tag=blob_ep56`] |

**[INFERRED arithmetic]** Blob full-context error rose **2.76×** from ep30 to ep56. **[MEASURED]** By ep50, skill versus the position-only reference was only **0.217523** on anatomy and **0.132442** on background [S3, S4].

**[INFERRED — moderate confidence]** This within-arm trajectory is the strongest mechanistic evidence: the predictor loses its preferential dependence on anatomy context and becomes weak relative to a positional baseline after prolonged training on small, nearly pure anatomy target sets.

**[LIMITATION]** Predictor deterioration is diagnostic, not causal. It could be caused by target purity, 2.42× fewer target observations, connected geometry, replacement padding, the larger context, or their interaction.

## 4. Exact patch attribution: blob emphasizes anatomy but extracts less class separation

| arm | anatomy purity % | abs contribution per anatomy patch | abs contribution per background patch | background/anatomy per patch | AUC from anatomy contribution | AUC from background contribution | provenance |
|---|---:|---:|---:|---:|---:|---:|---|
| random ep50 | 31.5832 | 0.000267927 | 0.000340289 | 1.2701 | 0.863850 | 0.846961 | **[MEASURED]** [S10: random] |
| oracle ep50 | 39.6878 | 0.000327909 | 0.000382038 | 1.1651 | 0.864250 | 0.859438 | **[MEASURED]** [S10: oracle] |
| envelope ep50 | 43.1927 | 0.000336753 | 0.000358456 | 1.0644 | 0.865403 | 0.858594 | **[MEASURED]** [S10: envelope] |
| blob ep50 | 97.5011 | 0.000415497 | 0.000359912 | 0.8662 | **0.846425** | **0.855678** | **[MEASURED]** [S10: blob] |

**[MEASURED]** The per-patch background/anatomy influence ratio falls monotonically as anatomy targeting increases. Blob is the only arm that weights anatomy positions more strongly per patch, yet its background contribution separates glaucoma better than its anatomy contribution.

**[INFERRED]** Blob’s failure is not that the frozen head ignores anatomy positions. It is that the features at those emphasized positions carry comparatively weak class separation. This is consistent with learning a generic retinal prototype/location signal rather than discriminative retinal structure.

**[MEASURED — indirect]** In a separate GOALS tissue-relation probe, envelope ep100 I-JEPA achieved inner-retina-versus-choroid relation AUC **0.6945**, below an untrained encoder’s **0.8288** and MIRAGE encoder’s **0.9773** [S11]. **[INFERRED]** This independently shows that the I-JEPA objective can preserve “retina versus elsewhere” while compressing clinically finer within-retina distinctions. **[LIMITATION]** It is not a blob-specific downstream experiment.

## 5. Evidence against an overstrong anatomy-starvation story

1. **[MEASURED]** Blob’s anatomy context percentage is lowest, but its absolute anatomy context (**9.97 cells**) exceeds envelope (**8.63**) and COVER f0.21 (**9.28**) [S1].
2. **[MEASURED]** Blob’s zero-anatomy context rate (**1.24%**) is lower than every rectangle/COVER arm in the composition table [S1].
3. **[MEASURED]** Anatomy-only downstream pooling is best for every encoder, so anatomy is not an intrinsically harmful information source [S2].
4. **[MEASURED — suggestive]** A separate ep30 anatomy-shaped continuation scored **0.8582 ± 0.0003** versus envelope’s **0.8528 ± 0.0018**, although it had one pretraining trajectory and four documented confounds [S12].

**[INFERRED]** These facts undermine the claim that blob fails simply because too little anatomy is visible. A narrower claim remains plausible: the *quality and diversity* of visible anatomical evidence may be insufficient for reconstructing near-complete anatomy, even when the absolute cell count is not uniquely low.

## 6. Alternatives not ruled out

| alternative | supporting evidence | evidence against / missing test | verdict |
|---|---|---|---|
| Background-target removal eliminates useful regularization | **[MEASURED]** healthy background target skill reaches 0.680; blob target purity is 97.5% | **[MISSING]** fixed-budget target-composition ablation | **[INFERRED] plausible, unproven** |
| Too few supervised constraints | **[MEASURED]** 64 slots/54.7 unique cells versus 154.6/118.3 for envelope | **[MISSING]** budget-matched blob at `pred_target_k: 30` | **[INFERRED] highly plausible** |
| Connected geometry encourages prototype reconstruction | **[MEASURED]** blob uses connected anatomy targets; rectangles do not | **[MISSING]** same count/composition with alternate geometry | **[INFERRED] plausible** |
| Replacement padding reduces information | **[MEASURED]** only blob pads ragged targets with replacement | **[MEASURED]** aggregate duplicate percentage is not highest, so padding alone is insufficient | **[INFERRED] possible contributor** |
| Larger context makes the task too easy/shortcut-prone | **[MEASURED]** blob context is about 160 versus 63–77 | **[MISSING]** context-budget-matched blob | **[INFERRED] plausible** |
| Pure anatomy targeting is intrinsically harmful | **[MEASURED]** ep30 anatomy continuation was initially competitive/better | **[LIMITATION]** one trajectory and confounds | **[INFERRED] not supported** |
| Black pixels themselves contain glaucoma signal | **[MEASURED]** background-position AUC up to 0.8701 | **[LIMITATION]** global token mixing, mask leakage, no eroded/shuffled-pixel control | **[INFERRED] not established** |

## 7. Final causal statement for the paper

**[INFERRED — moderate confidence]** Near-pure anatomy targeting fails here because it changes the learning problem from broad, context-dependent field prediction into a small, compositionally narrow anatomy-reconstruction task. The blob predictor initially uses anatomy context, then loses that dependence and approaches a weak positional solution. The downstream encoder consequently emphasizes anatomy positions without extracting strong glaucoma separation from them.

**[INFERRED — low-to-moderate confidence]** Removing background targets is likely part of this failure because background prediction is demonstrably context-dependent and background-position features retain disease information. **[LIMITATION]** Current evidence cannot separate that mechanism from target-count, geometry, padding, and context-budget confounds.

**[ASSUMED — should not appear as a conclusion]** Optically black/non-anatomy pixels directly contain causal glaucoma biomarkers.

## Sources

- **S1:** `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json`, keys `rows[arm=*].{pct_anat_hid,ctx_anat,pct_ctx_anat,pct_tgt_anat,zero_pct,ctx,auc,src}`.
- **S2:** `D:\jepa_phase0\reports\downstream_region_auc\{random_ep50,oracle_ep50,envelope_ep50,blob_ep50}\region_auc.json`, keys `{all,anatomy,background}.{test,val,epoch}`.
- **S3:** `docs\experiments\masking\background_signal.md:274-357` for protocol, mask quality, global-mixing/leakage caveats, and regional interpretation; `:188-199` for the matched-removal caveat.
- **S4:** `D:\jepa_phase0\reports\background_signal\skill_scores.json`, entries selected by `tag`, keys `{bg,anat}.{skill_vs_pos,err_pred,err_pos}`.
- **S5:** `D:\jepa_phase0\reports\background_signal\background_signal.json`, entries selected by `tag`, keys `ablation.{err_full,err_drop_bg,err_drop_anat,err_drop_rand,k_dropped,excess_bg,excess_anat,excess_bg_sem,excess_anat_sem}`, plus `effrank_bg`, `effrank_anat`; derived token values match `marginal_token_value.csv`.
- **S6:** `D:\jepa_phase0\reports\target_composition\summary.json`, entries `arm=anatomy|envelope`, keys `{slots,unique,slots_bg_pct,ctx_tokens,ctx_bg_pct,dup_pct}`.
- **S7:** `configs\patch_anatomy_v2.yaml:21,32,39,50` (`num_pred_masks`, `pred_target_k`, `r_max`, `anatomy_tau`).
- **S8:** `scripts\composition_vs_auc.py:38-44`, AUC dictionary; the same values are serialized in the ep50/ep100 JSON files.
- **S9:** `scripts\composition_vs_auc.py:8-14`, source/sample-size statement for sweep versus blob.
- **S10:** `D:\jepa_phase0\reports\patch_attribution\{random_ep50,oracle_ep50,envelope_ep50,blob_ep50}_attrib.json`, keys `{per_patch_anatomy,per_patch_background,auc_anatomy_only,auc_background_only,total_abs_anatomy,total_abs_background,max_abs_residual}`.
- **S11:** `docs\experiments\masking\class_relations.md:68-91`.
- **S12:** `docs\experiments\masking\anatomy_vs_rectangle_ep30.md:13-44,82-120,184-186`.
