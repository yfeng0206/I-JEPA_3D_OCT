# Conflicts and documentation-only results

This ledger lists every inventory entry whose status is `CONFLICT` or `DOC-ONLY`. Rounded documentation values that agree with artifacts were not treated as conflicts.

## frozen-d1-epoch-sweep-doc — DOC-ONLY

**Issue.** How did the original d=1 attentive frozen probe vary with pretraining epoch?

**Assessment.** No local per-run result JSON was found for this four-checkpoint sweep.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| ep25 | test_auc | 0.8558 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\d1_sweep.md#L29` |
| ep50 | test_auc | 0.8611 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\d1_sweep.md#L30` |
| ep75 | test_auc | 0.8691 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\d1_sweep.md#L31` |
| ep100 | test_auc | 0.8706 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\d1_sweep.md#L32` |
| ep25 | sensitivity | 0.609 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\d1_sweep.md#L29` |
| ep25 | specificity | 0.910 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\d1_sweep.md#L29` |
| ep100 | sensitivity | 0.821 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\d1_sweep.md#L32` |
| ep100 | specificity | 0.716 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\d1_sweep.md#L32` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\d1_sweep.md#L25-L55`

## frozen-probe-architecture-doc — DOC-ONLY

**Issue.** Which frozen pooling head performs best on the random ep100 encoder?

**Assessment.** Mean-pool point estimate is artifact-backed elsewhere; d=1/cross-attention local result JSONs were not found.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| d1 | test_auc | 0.8706 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mean_pool.md#L41` |
| meanpool | test_auc | 0.8746 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mean_pool.md#L40` |
| crossattn | test_auc | 0.8791 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mean_pool.md#L42` |
| d1 | parameters | 7170000 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mean_pool.md#L41` |
| meanpool | parameters | 2300 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mean_pool.md#L40` |
| crossattn | parameters | 277000 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mean_pool.md#L42` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mean_pool.md#L25-L46`

## probe-bootstrap-matrix-doc — DOC-ONLY

**Issue.** Are frozen/fine-tuned probe-architecture differences statistically significant?

**Assessment.** Analysis says predictions live in AML/blob; local bootstrap artifact was not found.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| frozen_cross-minus-d1 | delta_auc | 0.0085 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L22` |
| frozen_cross-minus-d1 | p_two_sided | 0.004 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L22` |
| frozen_cross-minus-mean | delta_auc | 0.0046 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L23` |
| frozen_cross-minus-mean | p_two_sided | 0.088 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L23` |
| frozen_mean-minus-d1 | delta_auc | 0.0041 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L24` |
| frozen_mean-minus-d1 | p_two_sided | 0.163 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L24` |
| ft_d1-minus-cross | delta_auc | 0.0005 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L32` |
| ft_d1-minus-cross | p_two_sided | 0.81 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L32` |
| ft_d1-minus-mean | delta_auc | 0.0009 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L33` |
| ft_d1-minus-mean | p_two_sided | 0.69 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L33` |
| ft_cross-minus-mean | delta_auc | 0.0004 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L34` |
| ft_cross-minus-mean | p_two_sided | 0.63 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L34` |
| d1_finetune_uplift | delta_auc | 0.0172 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L42` |
| mean_finetune_uplift | delta_auc | 0.0122 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L43` |
| cross_finetune_uplift | delta_auc | 0.0080 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L44` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\ablation_analysis.md#L1-L96`

## oracle-random-bootstrap-doc — DOC-ONLY

**Issue.** Are oracle-vs-random mean-pool AUC gains significant at matched epochs?

**Assessment.** Point AUCs are artifact-backed; paired-bootstrap prediction artifact is not local.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| ep50 | delta_auc | 0.0099 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\oracle_meanpool_sweep.md#L38` |
| ep75 | delta_auc | 0.0113 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\oracle_meanpool_sweep.md#L39` |
| ep100 | delta_auc | 0.0109 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\oracle_meanpool_sweep.md#L40` |
| ep50 | p_two_sided_upper_bound | 0.0005 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\oracle_meanpool_sweep.md#L38` |
| ep75 | p_two_sided_upper_bound | 0.0005 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\oracle_meanpool_sweep.md#L38` |
| ep100 | p_two_sided_upper_bound | 0.0005 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\oracle_meanpool_sweep.md#L38` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\oracle_meanpool_sweep.md#L22-L51`

## mirage-three-arm-bootstrap-doc — DOC-ONLY

**Issue.** How do MIRAGE envelope, oracle, and random compare with paired uncertainty?

**Assessment.** Point AUCs are artifact-backed; paired bootstrap is doc-only locally.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| ep50_envelope-minus-random | delta_auc | 0.0120 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mirage_meanpool_sweep.md#L56` |
| ep50_envelope-minus-oracle | delta_auc | 0.0020 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mirage_meanpool_sweep.md#L82` |
| ep75_envelope-minus-random | delta_auc | 0.0080 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mirage_meanpool_sweep.md#L91` |
| ep75_envelope-minus-oracle | delta_auc | -0.0033 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mirage_meanpool_sweep.md#L83` |
| ep100_envelope-minus-random | delta_auc | 0.0062 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mirage_meanpool_sweep.md#L92` |
| ep100_envelope-minus-oracle | delta_auc | -0.0047 | `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mirage_meanpool_sweep.md#L84` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\frozen\mirage_meanpool_sweep.md#L49-L101`

## anatomy-envelope-probe-seeds — DOC-ONLY

**Issue.** Does anatomy-shaped masking beat envelope rectangles at ep30 across probe seeds?

**Assessment.** Critical: five probe seeds reuse one encoder per arm. The pretraining comparison is confounded by continuation and hard-vs-soft guide caches.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| envelope | mean_test_auc | 0.8528 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\anatomy_vs_rectangle_ep30.md#L119` |
| envelope | sd_test_auc | 0.0018 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\anatomy_vs_rectangle_ep30.md#L120` |
| anatomy | mean_test_auc | 0.8582 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\anatomy_vs_rectangle_ep30.md#L119` |
| anatomy | sd_test_auc | 0.0003 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\anatomy_vs_rectangle_ep30.md#L120` |
| anatomy-minus-envelope | delta_auc | 0.0054 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\anatomy_vs_rectangle_ep30.md#L123` |
| welch | p | 0.00219 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\anatomy_vs_rectangle_ep30.md#L124` |
| mann-whitney | p | 0.0079 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\anatomy_vs_rectangle_ep30.md#L125` |
| cohen | d | 4.20 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\anatomy_vs_rectangle_ep30.md#L126` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\anatomy_vs_rectangle_ep30.md#L79-L137`

## adapter-saturation-doc — DOC-ONLY

**Issue.** How many cached slices are needed before the adapter saturates?

**Assessment.** The doc explicitly says no figure/artifact exists locally and attributes numbers to a user-provided specification.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| 480 | lrel_reduction_pct | 3.1 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L125` |
| 1200 | lrel_reduction_pct | 18.1 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L126` |
| 2400 | lrel_reduction_pct | 27.7 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L127` |
| 4800 | lrel_reduction_pct | 30.7 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L128` |
| 9600 | lrel_reduction_pct | 31.9 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L129` |
| 19200 | lrel_reduction_pct | 32.5 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L130` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L118-L147`

## amp-vs-fp32-guide-doc — DOC-ONLY

**Issue.** Is AMP safe for one-time guide generation?

**Assessment.** Only the one-uint8-level cache check has a local artifact; the full timing/agreement table is doc-only.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| amp-vs-fp32 | pixel_argmax_agreement | 0.999960 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L155` |
| amp-vs-fp32 | score_abs_diff_mean | 0.000019 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L156` |
| amp-vs-fp32 | score_abs_diff_max | 0.0028 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L157` |
| amp-vs-fp32 | mask_jaccard_mean | 0.9984 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L158` |
| amp-vs-fp32 | mask_jaccard_min | 0.609 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L158` |
| amp-vs-fp32 | identical_masks | 254 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L159` |
| amp | milliseconds_per_image | 3.61 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L161` |
| fp32 | milliseconds_per_image | 7.72 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L161` |
| amp | speedup | 2.14 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L162` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L149-L179`

## context-loss-figures-only — DOC-ONLY

**Issue.** How does context removal affect JEPA prediction loss?

**Assessment.** D:\jepa_phase0\reports\context_loss contains PNGs only, so no exact numeric table could be verified.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| — | no exact local numeric artifact | — | documentation/figure only |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\background_signal.md#L149-L239`

## interpretability-slice-agreement — DOC-ONLY

**Issue.** Do three probe families attend to the same volume slices?

**Assessment.** Underlying NPZ outputs are external/blob, not local.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| mean-vs-cross | slice_pearson_r | 0.94 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L31` |
| window7 | peak_abs_logit | 0.22 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L37` |
| single_slice | peak_abs_logit | 0.03 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L37` |
| mean-vs-cross | patch_r_slice20 | 0.45 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L60` |
| mean-vs-cross | patch_r_slice43 | 0.48 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L60` |
| mean-vs-d1 | slice_pearson_r | 0.53 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L61` |
| cross-vs-d1 | slice_pearson_r | 0.59 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L62` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L25-L73`

## interpretability-patch-significance — DOC-ONLY

**Issue.** How spatially concentrated and statistically non-zero is patch-level attribution?

**Assessment.** Underlying outputs are not local; do not confuse this nonlinear occlusion study with exact linear patch decomposition.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| patches | nonzero_ci_pct_low | 84 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L53` |
| patches | nonzero_ci_pct_high | 91 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L53` |
| curated | max_abs_patch_delta | 0.003 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L78` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L41-L99`

## interpretability-completeness — DOC-ONLY

**Issue.** Which occlusion primitive recovers the model logit most completely?

**Assessment.** Window occlusion is non-additive for d=1 and overshoots the baseline logit.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| meanpool_w1 | completeness_pct | 1.3 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L93` |
| meanpool_w7 | completeness_pct | 32.4 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L93` |
| cross_w1 | completeness_pct | 6.0 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L94` |
| cross_w7 | completeness_pct | 52.0 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L94` |
| d1_w1 | completeness_pct | 48.6 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L95` |
| d1_w7 | completeness_pct | 304.9 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L95` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L87-L100`

## interpretability-od-os-mirror — DOC-ONLY

**Issue.** Are the two population attribution peaks bilateral anatomy or an OD/OS storage mirror artifact?

**Assessment.** Rejects the earlier bilateral-rim interpretation.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| meanpool | raw_cluster_corr | -0.124 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L123` |
| meanpool | flipped_cluster_corr | 0.971 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L123` |
| cross | raw_cluster_corr | -0.478 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L124` |
| cross | flipped_cluster_corr | 0.988 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L124` |
| d1 | raw_cluster_corr | 0.228 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L125` |
| d1 | flipped_cluster_corr | 0.237 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L125` |
| cross | realigned_secondary_peak | 0.001 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L133` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L101-L142`

## interpretability-errors-confidence — DOC-ONLY

**Issue.** Do errors attend to different anatomy or simply show weaker signal?

**Assessment.** Document reports the shape conclusion; local numerical arrays were not found.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| confidence | max_abs_pearson_r | 0.25 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L145` |
| threshold | error_rate_pct | 20 | `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L145` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\interpretability.md#L143-L159`

## random-pretraining-diagnostics-doc — DOC-ONLY

**Issue.** What was the random-init pretraining diagnostic trajectory?

**Assessment.** Exact per-epoch log artifact was not included in the requested local artifact set.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| ep25 | train_loss | 0.1174 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\random_100ep.md#L40` |
| ep25 | val_loss | 0.1197 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\random_100ep.md#L40` |
| ep50 | train_loss | 0.1413 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\random_100ep.md#L41` |
| ep50 | val_loss | 0.1423 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\random_100ep.md#L41` |
| ep75 | train_loss | 0.1445 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\random_100ep.md#L42` |
| ep75 | val_loss | 0.1469 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\random_100ep.md#L42` |
| ep100 | train_loss | 0.1352 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\random_100ep.md#L47` |
| ep100 | val_loss | 0.1419 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\random_100ep.md#L47` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\random_100ep.md#L33-L70`

## oracle-pretraining-diagnostics-doc — DOC-ONLY

**Issue.** How did the oracle pretext diagnostics evolve?

**Assessment.** Downstream AUC artifacts are local; pretraining log summary is doc-only here.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| ep26 | train_loss | 0.1186 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\oracle_100ep.md#L54` |
| ep26 | val_loss | 0.1202 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\oracle_100ep.md#L54` |
| ep50 | train_loss | 0.1316 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\oracle_100ep.md#L57` |
| ep50 | val_loss | 0.1400 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\oracle_100ep.md#L57` |
| ep75 | train_loss | 0.1404 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\oracle_100ep.md#L59` |
| ep75 | val_loss | 0.1507 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\oracle_100ep.md#L59` |
| ep100 | train_loss | 0.1303 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\oracle_100ep.md#L62` |
| ep100 | val_loss | 0.1432 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\oracle_100ep.md#L62` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\oracle_100ep.md#L50-L80`

## envelope-pretraining-diagnostics-doc — DOC-ONLY

**Issue.** How did MIRAGE-envelope pretext diagnostics evolve?

**Assessment.** Validation uses uniform rectangles and is not a train/generalization gap for curriculum arms.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| ep26 | train_loss | 0.1182 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\envelope_100ep.md#L63` |
| ep26 | val_loss | 0.1191 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\envelope_100ep.md#L63` |
| ep50 | train_loss | 0.1216 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\envelope_100ep.md#L66` |
| ep50 | val_loss | 0.1401 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\envelope_100ep.md#L66` |
| ep75 | train_loss | 0.1332 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\envelope_100ep.md#L68` |
| ep75 | val_loss | 0.1514 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\envelope_100ep.md#L68` |
| ep100 | train_loss | 0.1234 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\envelope_100ep.md#L71` |
| ep100 | val_loss | 0.1448 | `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\envelope_100ep.md#L71` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\pretraining\envelope_100ep.md#L59-L94`

## engineering-timing-cache-doc — DOC-ONLY

**Issue.** What are the sampler and guide-cache costs?

**Assessment.** Cache equivalence has a local artifact; timing decomposition is doc-only.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| random_matched-minus-default | seconds_per_epoch | 1027 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L11` |
| anatomy-minus-random_matched | seconds_per_epoch | 876 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L12` |
| is_viable | milliseconds_per_image | 18.89 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L17` |
| build_targets | milliseconds_per_image | 14.24 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L18` |
| guide_cache | volumes | 6000 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L24` |
| guide_cache | slices | 600000 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L24` |
| guide_cache | build_seconds | 3941 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L25` |
| guide_cache | size_gib | 3.85 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L26` |
| pool-before-softmax | jaccard | 0.587 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L35` |
| pool-before-softmax | identical_masks | 0 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L34` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\engineering_notes.md#L5-L41`

## sampler-mass-cap-doc — DOC-ONLY

**Issue.** Which anatomy mass cap balances hidden tissue and retained context?

**Assessment.** No single local artifact containing the full displayed table was identified.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| random | target_cells | 122.0 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\sampler_ablations.md#L11` |
| cap0.80 | target_cells | 46.1 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\sampler_ablations.md#L12` |
| cap0.85 | target_cells | 50.5 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\sampler_ablations.md#L13` |
| cap0.90 | target_cells | 55.9 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\sampler_ablations.md#L14` |
| cap0.95 | target_cells | 63.3 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\sampler_ablations.md#L15` |
| cap0.99 | target_cells | 69.2 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\sampler_ablations.md#L16` |
| cap0.90 | zero_retina_context_pct | 0.80 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\sampler_ablations.md#L14` |
| cap0.99 | zero_retina_context_pct | 21.00 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\sampler_ablations.md#L16` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\sampler_ablations.md#L5-L33`

## curriculum-ablation-status — DOC-ONLY

**Issue.** Was a clean curriculum schedule ablation run?

**Assessment.** The programme ran curriculum-based arms, but did not vary curriculum schedule as a clean single-variable downstream experiment.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| curriculum | warm_epoch | 25 | `C:\Users\Gary\Desktop\jepa\docs\experiments\curriculum_masking.md#L212` |
| curriculum | full_guidance_epoch | 30 | `C:\Users\Gary\Desktop\jepa\docs\experiments\curriculum_masking.md#L212` |
| planned_training_arms | count | 3 | `C:\Users\Gary\Desktop\jepa\docs\experiments\curriculum_masking.md#L85` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\curriculum_masking.md#L32-L96`

## conflict-ep50-checkpoint-availability — CONFLICT

**Issue.** Were random and oracle ep50 downstream results measured?

**Assessment.** Trust committed result JSONs: they are later machine-readable outputs; the document records an earlier filesystem state.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| doc_random_ep50 | measured_flag | 0 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\mask_composition_report.md#L179` |
| doc_oracle_ep50 | measured_flag | 0 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\mask_composition_report.md#L180` |
| artifact_random_ep50 | test_auc | 0.8640970649809413 | `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_random\ep50_results.json#$.test_auc` |
| artifact_oracle_ep50 | test_auc | 0.8740299460522829 | `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_oracle\oracle_ep50.json#$.test_auc` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\mask_composition_report.md#L171-L202`
- `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_random\ep50_results.json`
- `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_oracle\oracle_ep50.json`

## conflict-cover-amp-target-ledger — CONFLICT

**Issue.** Was amp_target false or enabled in the completed COVER campaign?

**Assessment.** Trust the preserved run ledger and retraction for the archived run; the config file was remediated to false after completion.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| locked_config | amp_target_true | 0 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\cover_random_campaign.md#L112` |
| deviation_ledger | amp_target_true | 1 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\cover_random_campaign.md#L174` |
| completed_run_retraction | amp_target_true | 1 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\cover_random_campaign.md#L22` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\cover_random_campaign.md#L101-L180`

## conflict-gate-excluded-patch-count — CONFLICT

**Issue.** How many target-excluded patches were measured in the element-wise gate probe?

**Assessment.** Trust each count only for its own batch/context realization. The current artifact is exact for batch=8 and 71 context tokens (8×(256−71)=1480); the doc likely cites a different run.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| documentation | excluded_patches | 1360 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L114` |
| gate_real_artifact | excluded_patches | 1480 | `C:\Users\Gary\Desktop\jepa\results\masking\gate_real\elementwise_gate_probe.json#$.q1.n_patches_outside_context` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\adapter_ablations.md#L110-L116`
- `C:\Users\Gary\Desktop\jepa\results\masking\gate_real\elementwise_gate_probe.json`

## conflict-cover-training-status — CONFLICT

**Issue.** Did a COVER-then-RANDOM checkpoint/AUC exist?

**Assessment.** The document is time-stale. Later artifacts prove a contaminated completed run and a remediated live run.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| background_doc | checkpoint_exists | 0 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\background_signal.md#L472` |
| completed_campaign | auc_count | 4 | `D:\jepa_phase0\campaign\chain_status.json#$.cover_aucs#length` |
| clean_campaign | ep30_test_auc | 0.8522489776969857 | `D:\jepa_phase0\campaign\chain_f021_status.json#$.aucs.ep30` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\background_signal.md#L415-L476`
- `D:\jepa_phase0\campaign\chain_status.json`
- `D:\jepa_phase0\campaign\chain_f021_status.json`

## conflict-floor-sweep-state — CONFLICT

**Issue.** Was the COVER floor sweep still running or complete?

**Assessment.** Trust the newer completed n=6137 artifact; the doc's n=6144 was a planned target and its running state is stale.

**Recorded values.**

| Source condition | Quantity | Value | Locator |
|---|---|---:|---|
| doc_planned_n | n | 6144 | `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\crop_and_precision_audit.md#L451` |
| artifact_completed_n | n | 6137 | `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json#$.0.15.n` |
| floor0.15 | blank_rate_pct | 10.738145673781977 | `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json#$.0.15.zero_pct` |
| floor0.30 | blank_rate_pct | 5.507576992015642 | `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json#$.0.30.zero_pct` |

**Provenance.**
- `C:\Users\Gary\Desktop\jepa\docs\experiments\masking\crop_and_precision_audit.md#L449-L469`
- `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json`
