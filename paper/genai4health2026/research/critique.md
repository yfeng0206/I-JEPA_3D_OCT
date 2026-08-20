# Adversarial pre-submission audit

## Verdict

**WEAK REJECT — the code-level context-truncation claim is substantiated, but the paper packages it with single-pretraining-seed, confounded, adaptively test-set-evaluated composition/mechanism/equity stories that exceed the evidence and still contain visible TODOs.**

The single most likely rejection reason is that the only clean contribution is C1, while the title, abstract, figures, and contribution list sell a much broader scientific story than the current experiments identify.

## P0 — must fix before submission

### P0.1 — The submitted PDF visibly says “TODO”

**Exact quotes**

- Line 361: `\TODO{Add the clean COVER floor-$0.21$ epoch-$50$ probe when available.}`
- Lines 443–449: `\TODO{Recover or recreate...}`, `\TODO{Repeat anatomy-shaped...}`, and `\TODO{Measure matched epoch-$50$ AUC...}`

These render in red on PDF pages 6–7. This is an immediate desk-level signal that the work is unfinished. The venue explicitly permits work in progress only when current results substantiate the central claim; advertising three missing central controls invites rejection.

**Concrete replacement**

> The clean COVER \(f=.21\) run is reported only through epoch 34; epoch-50 and cross-floor AUCs are unavailable and are not used to support a floor–quality claim. Pretraining-seed replication and a guide-, budget-, and target-count-matched shape comparison remain future work; all current policy comparisons are exploratory associations.

Delete every rendered TODO and the unused `\TODO` macro.

### P0.2 — The mechanism section contains a directly false “highest count” claim

**Exact quote, lines 272–275**

> “Blob has the lowest anatomy fraction in context (6.26\%) yet the highest absolute anatomy count (9.97 cells) and the lowest zero-anatomy rate (1.24\%).”

Table 1 contradicts this sentence: random has **18.25** anatomy cells and oracle **14.93**, both well above blob’s **9.97**. Blob is highest only among blob/envelope/COVER, not among all reported arms. A reviewer will catch this immediately and distrust the rest of the accounting.

**Concrete replacement**

> Blob has the lowest anatomy fraction in context (6.26%) but retains more anatomy cells than envelope (9.97 versus 8.63) and COVER (9.28), and it has the lowest zero-anatomy rate (1.24%). Thus low anatomy percentage or blank-context frequency alone does not explain blob’s deficit relative to those guided arms.

### P0.3 — The ep30 “reversal” conflates different anatomy runs and presents pseudoreplicated inference

**Exact quotes**

- Lines 60–64: “anatomy is ahead by 0.0054 at epoch 30, but the direction reverses by \(-0.0107\) at epoch 50”
- Lines 261–266: “Welch \(p=0.00219\), \(d=4.20\) ... At epoch 50 the direction reverses”
- Lines 99–100: “trajectory reversal”

The ep30 value is from `patch_mirage_anatomy/...ep30`; the ep50 value is from the distinct `anatomy_v2_ep25/...ep50` continuation. This is not a within-run trajectory reversal. Moreover, the five “replicates” are probe seeds on one frozen encoder per arm. The Welch \(p\)-value and \(d\) do not test a masking-policy effect. The ep30 comparison also uses different guide caches and continuation histories; the source experiment document records further differences, including an ep100-derived look-ahead guide, ramp granularity, and target-count policy.

**Concrete replacement**

> For one fixed pair of ep30 encoders, anatomy v1 has a \(+0.0054\) mean probe-seed difference; the probe-seed runs are technical replicates and provide no pretraining-level inference. A separately configured anatomy v2 continuation scores 0.0107 below envelope at epoch 50. Because lineage, guide cache, geometry, target count, padding, and context budget differ, these observations neither form a single trajectory nor isolate target shape.

Remove the Welch \(p\)-value and \(d\) from the main scientific argument unless raw per-seed artifacts are restored and the statistic is explicitly labeled as probe-fitting stability only.

### P0.4 — Figure 6 makes the fairness overclaim that the prose denies

**Exact source inclusion, line 325**

> `\includegraphics[width=\linewidth]{fig6_subgroup_disparity.pdf}`

The baked-in panel titles say:

> “Mask policy modulates racial disparity”

and

> “The mild-disease penalty is invariant to mask policy”

The first is unsupported: policy is confounded with pretraining seed and epoch, the AUC–gap association is non-significant (\(p=0.167\)), and 12 probes represent only five non-independent runs. The second is an unjustified universal statement from one dataset and five trajectories. “Black is the worst-served group” is also too strong when no individual racial gap is significant; only the lowest point-estimate ordering is supported.

**Concrete replacement**

- Panel (a): **“Observed racial AUC gaps across saved probes”**
- Subtitle: **“Black subgroup has the lowest point estimate in 12/12 non-independent probes; no individual gap is significant”**
- Panel (b): **“Severity-gap estimates are similar across the tested probes”**

Also replace lines 334–341 with:

> The Black subgroup has the lowest point-estimate AUC in all 12 saved probes, which arise from five pretraining runs. No individual racial gap is significant, and policy is confounded with seed and epoch; we therefore report a repeated ordering, not evidence that masking policy changes disparity.

### P0.5 — Figure 1b is both factually contradictory and camera-ready unacceptable

**Exact source inclusion and caption, lines 166–172**

> `\includegraphics[width=\linewidth]{fig1b_context_excision.png}`
>
> “The total withheld count includes pre-collation context selection as well as batch truncation.”

The raster itself says **“WHERE THE REST WENT,” “ENCODER GETS,” “(FAILURE),”** and, worse, **“withheld by CROP: 123 cells.”** The caption correctly admits that 123 includes both cells outside the sampled context block and truncation. The image therefore makes a stronger causal attribution than the caption and is factually misleading. Panel labels “2)” and “3)” advertise a missing panel 1. This is an internal diagnostic screenshot, not a publication figure.

**Concrete replacement**

Re-render from the underlying masks with clean `(a)`/`(b)` labels, no informal language, and four explicitly separated categories:

1. target patches;
2. non-target patches outside the sampled context block;
3. sampled context patches discarded by batch-minimum truncation;
4. delivered encoder context.

Only category 3 may be labeled as truncation loss.

### P0.6 — The novelty language exceeds the allowed search-qualified claim

**Exact quotes**

- Lines 31–32: “We identify a previously undocumented medical-imaging consequence”
- Lines 89–91: “we identify a previously undocumented medical-imaging consequence”

“Previously undocumented” asserts the literature state as fact. The evidence supports only a search-qualified priority statement. The later sentence “we found no prior report” is acceptable; the abstract should not be stronger than the related-work section.

**Concrete replacement**

> We found no prior report of the following medical-imaging consequence of released I-JEPA-style batching: ...

Use that wording consistently. Do not use “previously undocumented,” “novel defect,” or any first-discovery equivalent.

### P0.7 — The purity figure visually manufactures an optimum band from four confounded points

**Exact quotes**

- Lines 243–246: “observational dose-response hypothesis, not an identified optimum”
- Lines 253–257: “the strongest epoch-50 policies therefore place approximately 40–43\% of target patches on anatomy”

Despite the caveat, Figure 2 shades 39.7–43.2% and labels it **“highest AUCs at 39.7–43.2%.”** This is exactly how one visually sells an optimum band. Four single-seed points on one dataset, with geometry, target count, guide source, hidden fraction, and context budget moving together, do not support a dose-response curve or a useful range.

**Concrete replacement**

Remove the shaded band and annotation. Replace the prose with:

> Among four confounded, single-seed arms, the two highest observed AUCs occur at purities of 39.7% and 43.2%. This sparse comparison supports only rejection of a monotonic “more anatomy is better” rule; it does not estimate a preferred range or dose response.

### P0.8 — The ablation table includes numbers whose raw artifacts are missing

**Exact quote, lines 368–372**

> “The \(d=1\) and cross-attention values are retained in experiment documentation; their local result JSONs were not recovered.”

The table then reports 0.8706 and 0.8791 as if they were equivalent to artifact-backed results. `ablation_inventory.json` marks these entries **DOC-ONLY**. This is not acceptable provenance for an adversarially reviewed quantitative table.

**Concrete replacement**

Either recover the result JSONs/predictions or remove the \(d=1\) and cross-attention rows. A defensible replacement is:

> The artifact-backed mean-pool result is 0.8746. Historical documentation lists attentive and cross-attention point estimates, but raw local artifacts were not recovered; we exclude them from quantitative comparison.

### P0.9 — Adaptive reuse of the test split invalidates confirmatory language and is buried

**Exact quote, lines 437–439**

> “The study uses one dataset and frozen probes, with historical reuse of its test split and no external cohort.”

This is not a routine limitation. The test split has informed multiple method, checkpoint, probe, and narrative decisions; it functions partly as a development set. Consequently, downstream AUC, purity selection, mechanism interpretation, and subgroup analyses are exploratory and vulnerable to adaptive overfitting. Hiding this on page 7 while the abstract reports four-decimal test AUCs is misleading.

**Concrete replacement**

Add to the setup and abstract:

> The FairVision test split was consulted during prior development, so all downstream and subgroup results are exploratory rather than confirmatory; no untouched external cohort is available.

Do not use “held-out” or imply external validity. Ideally, add an untouched confirmation cohort before submission.

## P1 — should fix

### P1.1 — “Predictor collapse” is not operationally defined

Lines 74, 268, 297–304, and 429 use “predictor collapse,” but the reported evidence is rising prediction error and declining marginal token-value ratio. Those establish deterioration/shortcutting, not necessarily collapse in the conventional representation-variance sense.

**Replacement wording**

> We use “predictor deterioration toward a positional/prototype shortcut” to denote the jointly observed error increase and loss of preferential anatomy-context dependence; we do not claim encoder representation collapse.

If “collapse” is retained, define a preregistered metric such as output variance/effective rank and report it across healthy controls.

### P1.2 — The code proof shown is not explicitly connected to the guided-arm production path

Lines 142–180 show `src/masks/multiblock.py`, while envelope/COVER/blob are generated through `src/masks/curriculum.py`. The latter independently sorts encoder indices and applies prefix truncation, but the paper never says so. A reviewer can reasonably ask whether the displayed code is the code that produced the guided-arm results.

**Replacement wording**

> The same sorted-encoder-index followed by batch-minimum prefix operation is used in the curriculum generator that produced the guided arms; the stock and guided paths therefore share the spatial consequence.

### P1.3 — Two different composition audits produce visibly different zero-anatomy rates without explanation

Figure 1 reports random/envelope zero-anatomy rates of 4.63%/10.10% from an \(n=1{,}534\) pass; Table 1 reports 3.68%/8.07% from the separate \(n=6{,}137\) composition sweep. Both are valid, but the manuscript does not explicitly explain the discrepancy.

**Replacement wording**

> Figure 1 and Table 1 use independent mask draws and different audit sizes; their zero-anatomy rates are therefore separate estimates and should not be numerically interchanged.

### P1.4 — “Isolates the collation mechanism” is too strong for unpaired audits

**Exact quote, lines 352–353**

> “The \(B=1\rightarrow64\) intervention ... isolates the collation mechanism”

The B=1 and B=64 artifacts use different sample counts (256 versus 1,534) and fresh stochastic draws. The deterministic code proof identifies the mechanism, but the numerical difference is not a paired intervention on identical masks.

**Replacement wording**

> The code proof identifies prefix truncation as the mechanism; separate B=1 and B=64 audits quantify its practical magnitude under the two batch settings.

### P1.5 — The long-horizon oracle sentence remains causal-suggestive

**Exact quote, lines 403–405**

> “The epoch-100 oracle/random difference shows that anatomical placement can remain useful”

With one trajectory per policy, “shows ... useful” is stronger than the next clause’s disclaimer.

**Replacement wording**

> The epoch-100 oracle point estimate is compatible with a persistent benefit from anatomical placement, but one trajectory per policy does not identify a policy effect.

### P1.6 — The severity result is partly tautological and should not be sold as an equity discovery

Lines 343–348 acknowledge that mean deviation defines the label. Positives near the \(-2\) threshold being harder than severe positives is therefore expected from label construction. Calling this a “policy-invariant mild-disease penalty” elevates a threshold-proximity diagnostic into a broader fairness/clinical claim.

**Replacement wording**

> When each positive severity stratum is compared with the common negative pool, threshold-near mild positives are consistently harder than severe positives across the tested probes. Because mean deviation defines the label, this is a label-proximity diagnostic, not an independent subgroup-fairness effect.

### P1.7 — The offline-filtering conclusion is categorical beyond the test performed

**Exact quote, lines 209–212**

> “The failure is draw-dependent, so offline filtering cannot identify a fixed set of slices to remove.”

The audit shows no slice is always blank and that blanking varies by draw. It does not prove that a probabilistic risk score or conservative screening rule could not reduce failures. The same analysis reports that the worst 20% of slices cover 64% of observed blanks, albeit in-sample.

**Replacement wording**

> No deterministic fixed set captures all failures because blanking varies across draws; whether a separately validated risk screen could reduce the rate was not tested.

### P1.8 — The setup is too thin to reproduce the downstream comparisons

Lines 120–136 omit pretraining optimizer/schedule, effective epochs per arm, target counts, hidden/context budgets, guide-cache identity, and early-stopping details. These omitted details are exactly the confounds on which the conclusions depend.

Add a compact configuration table with, per arm: lineage/checkpoint, guide cache, target geometry/count, truncation mode, context budget, pretraining epochs, pretraining seed count, probe protocol, and test-set status.

### P1.9 — The paper remains structurally diffuse

The title and abstract promise target composition, context loss, predictor mechanism, equity, and severity. C1 is a crisp deductive/software-audit contribution; C2 is a four-point association; the collapse diagnosis is confounded; equity is a post-hoc repeated-checkpoint analysis. The fairness section reads appended rather than necessary to the central mechanism.

Make C1 the explicit spine. Present C2/mechanism as exploratory consequences of auditing effective masks, and equity/severity as secondary post-hoc analyses. Otherwise reviewers will judge the weakest claim as representative of the whole paper.

## P2 — polish

### P2.1 — Figure text will be too small at print size

Figure 2 uses 6.2–6.5 pt annotations across three panels; Figure 6 places 12 horizontal bars, long labels, a legend, and a second panel in one line-width figure. Simplify and use at least the template’s normal caption-scale text. Figure 1b is a 980×380 raster with no DPI metadata.

### P2.2 — Arm names are informal or misleading

“blob” sounds like debugging shorthand, and “oracle” sounds like an upper bound despite being a hand-crafted retinal band. Prefer “connected-anatomy” and “band-guided” in reader-facing prose, retaining internal names only in a mapping table.

### P2.3 — The 44.8% chance calculation is arithmetically inconsistent

Line 210 reports 44.8%. The stored aggregate blank rate is 6.5104%; \((1-0.065104)^12=44.58\%\), which rounds to **44.6%**, not 44.8%. This is small but unnecessary ammunition for a skeptical reviewer.

### P2.4 — The full commit hash damages typesetting

Lines 176–181 produce underfull boxes around the 40-character hash. Use a short hash in prose and put the full hash in a footnote or reproducibility statement.

### P2.5 — Demographic capitalization is inconsistent

The paper uses “Black,” “Asian,” and “white.” Use one explicit style consistently (for example, Black, Asian, White) or quote the dataset’s exact lowercase category labels.

### P2.6 — Double-blind and page-limit checks pass, but remove dead scaffolding

The rendered PDF is eight pages, contains only “Anonymous Author(s),” and contains no `C:\`, `D:\`, `/tmp/`, GitHub/repository URL, personal name, or institution leak. Ceballos Arroyo et al. is cited and C3 novelty is explicitly disclaimed. After removing TODOs, also remove the unused `\TODO` and `\pp` commands.

## Numbers verified

I opened the underlying JSON/CSV/NPZ artifacts directly; the large inventories were queried programmatically rather than read wholesale.

| Claim | Value in paper | Source file actually checked | Result |
|---|---:|---|---|
| FairVision split | 6,000 / 1,000 / 3,000 | File counts under `D:\jepa_phase0\fairvision-glaucoma\data\{Training,Validation,Test}` | **MATCHES** |
| OCT array | \(200\times200\times200\), uint8 | `Test\data_07001.npz::oct_bscans` | **MATCHES** |
| Test labels | 1,466 positive / 1,534 negative | `metadata\data_summary_glaucoma.csv`, filtered to `use=test` | **MATCHES** |
| Shared epoch-25 AUC | 0.8487 | `D:\jepa_phase0\runs\frozen_meanpool_fork_ep25\results.json` = 0.8486800 | **MATCHES** |
| Random standard ep50 AUC/protocol | 0.8640971; mean-pool, linear, 100 slices, seed 42; stop 46 | `results\downstream\meanpool_sweep_random\ep50_results.json` | **MATCHES** |
| Oracle standard ep50 AUC/protocol | 0.8740299; mean-pool, linear, 100 slices, seed 42; stop 47 | `results\downstream\meanpool_sweep_oracle\oracle_ep50.json` | **MATCHES** |
| Envelope ep50 AUC | 0.8761 | `D:\jepa_phase0\runs\frozen_meanpool_mirage_ep50\results.json` = 0.8760641 | **MATCHES** |
| Blob ep50 AUC | 0.8654 | `D:\jepa_phase0\runs\frozen_meanpool_bridge_ep50\results.json` = 0.8653855 | **MATCHES** |
| Random composition | 53.04 recall, 31.58 purity, 26.38% context anatomy, 18.25 anatomy cells, 69.09 context, 3.68% zero; \(n=6,137\) | `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json::random` | **MATCHES** |
| Oracle composition | 61.58, 39.69, 19.34, 14.93, 77.23, 4.19%; \(n=6,137\) | Same file, `::oracle` | **MATCHES** |
| Envelope composition | 77.58, 43.19, 11.38, 8.63, 76.41, 8.07%; \(n=6,137\) | Same file, `::envelope` | **MATCHES** |
| COVER \(f=.21\) composition | 73.09, 40.88, 14.56, 9.28, 63.62, 7.84%; \(n=6,137\) | Same file, `::0.21` | **MATCHES** |
| Blob composition and separate pass | 82.07, 97.50, 6.26, 9.97, 160.00, 1.24%; \(n=1,534\) | `composition_vs_auc_ep50.json::rows[arm=blob]`, sourced from `arm_stats.json` | **MATCHES** |
| Random crop audit | 0.00%→4.63%; 36.4% context loss | `arm_stats_b1\arm_stats.json` and `arm_stats\arm_stats.json` | **MATCHES** |
| Envelope crop audit | 1.56%→10.10%; 30.7% context loss | Same two files | **MATCHES** |
| COVER .15 crop audit | 2.34%→11.02%; 36.2% context loss | Same two files | **MATCHES** |
| Failing-slice accounting | 256 = 90 + 123 + 43; anatomy 65 = 50 + 15 + 0 | `arm_stats\zero_anatomy_floor20.png` and `crop_and_precision_audit.md` | **MATCHES**, but image-only |
| Draw dependence | 0% always; 52.0% never | `arm_stats\blank_proneness.json` = 0.0 / 51.9531 | **MATCHES** |
| Chance never-blank rate | 44.8% | Derived from stored 6.5104% blank rate and 12 draws = 44.58% | **MISMATCH**; should round to 44.6% |
| Blob predictor errors | 0.104868 / 0.224373 / 0.271158 / 0.289461 | `background_signal\background_signal.json` | **MATCHES** |
| Blob marginal-value ratios | 3.529 / 0.998 / 0.830 / 0.737 | `background_signal\marginal_token_value.csv` | **MATCHES** |
| Blob error rise | 2.76× | 0.2894608 / 0.1048676 = 2.76025 | **MATCHES** |
| Target slots/unique cells | blob 64/54.744; envelope 154.624/118.314 | `target_composition\summary.json` | **MATCHES** |
| Envelope regional background/all AUC | 0.870075 / 0.8730 | `downstream_region_auc\envelope_ep50\region_auc.json` | **MATCHES** |
| Blob exact attribution AUC | background 0.855678; anatomy 0.846425 | `patch_attribution\blob_ep50_attrib.json` | **MATCHES** |
| Random regional protocol | all 0.8608341, anatomy 0.8746555, background 0.8543614; all stop 43 | `downstream_region_auc\random_ep50\region_auc.json` | **MATCHES** |
| Oracle regional protocol | all 0.8682588, anatomy 0.8746475, background 0.8652265; all stop 42 | `downstream_region_auc\oracle_ep50\region_auc.json` | **MATCHES** |
| Fairness join/exclusions | 3,000 labels agree in all 16; 12 valid / 4 retracted | `subgroup\subgroup_auc.json` | **MATCHES** |
| Race counts/order | White 2,318 / Black 431 / Asian 251; Black lowest in 12/12 point estimates | Same file | **MATCHES** |
| Race gap and overall spread | 0.0475→0.0935 = 1.968×; AUC spread 0.0324 | Same file, recomputed across valid entries | **MATCHES** |
| AUC–race-gap association | Spearman 0.427, \(p=0.167\) | Same file, recomputed with SciPy | **MATCHES** |
| Severity gap | 0.1306–0.1394; spread 0.0088 | Same file | **MATCHES** |
| Clean COVER trajectory | 0.8483 / 0.8522 / 0.8571 at ep27/30/34 | `frozen_meanpool_cover_f021_ep{27,30,34}\results.json` | **MATCHES** |
| Long-horizon random/oracle | 0.8745809 / 0.8854852 | `meanpool_sweep_random\ep100_results.json`; `meanpool_sweep_oracle\oracle_ep100.json` | **MATCHES** |

## Untraceable numbers

1. **Probe-head AUCs 0.8706 (\(d=1\)) and 0.8791 (cross-attention).** These are traceable only to `docs/experiments/frozen/mean_pool.md`; `ablation_inventory.json` marks them **DOC-ONLY**, and the raw result JSONs/predictions were not recovered. Remove them or recover the artifacts.
2. **The five-seed ep30 aggregates and inferential statistics** — anatomy \(0.8582\pm0.0003\), envelope \(0.8528\pm0.0018\), Welch \(p=0.00219\), and \(d=4.20\) — are also marked **DOC-ONLY** in `ablation_inventory.json`. Seed-42 result JSONs and predictions exist, but the per-seed result artifacts needed to regenerate the reported mean/SD/test were not found.
3. **“No individual racial gap is statistically significant.”** The subgroup JSON stores marginal subgroup bootstrap intervals, not a stored paired gap distribution or gap-test \(p\)-value. The statement is present in the narrative evidence, but its exact test artifact is not recoverable from the checked JSON. Store the pairwise gap bootstrap output or weaken this to “the reported subgroup intervals overlap.”
4. **The failing-slice 90/123/43 and 50/15/0 accounting** is preserved only in a large diagnostic PNG plus narrative documentation, not a structured per-cell artifact. It is visually traceable but fragile and not independently queryable. Save the mask indices/counts as JSON before submission.
