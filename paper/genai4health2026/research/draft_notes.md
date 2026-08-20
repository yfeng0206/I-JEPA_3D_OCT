# Draft notes

## Claim policy

- `[MEASURED]` means stored-artifact or retained experiment-documentation
  evidence.
- `[INFERRED]` means arithmetic, deterministic code consequence, or
  interpretation of measured values.
- `[ASSUMED]` means an untested premise. No assumed numerical conclusion is
  presented as a result in `main.tex`.
- The ledger below covers every scientific number in the manuscript. LaTeX
  dimensions, section/figure numbers, style years, and bibliography metadata
  are not experimental claims.

## What changed

1. Reframed the paper around the stock-style I-JEPA prefix-truncation defect.
2. Explicitly rejected anatomy-shape superiority and reported both the
   epoch-30 advantage and epoch-50 sign reversal.
3. Added the five-arm post-collation composition table, four-point purity
   analysis, predictor-collapse mechanism, equity analysis, and a real
   ablation section.
4. Corrected the composition provenance: random/oracle/COVER/envelope use
   `n=6,137`; blob uses a separate `n=1,534` pass.
5. Added the full confound ledger, including non-independent probes,
   retracted COVER runs, guide/continuation mismatch, missing pretraining-seed
   replication, subgroup uncertainty, and the document-only head results.
6. Removed the qualitative appendix and less central trajectory figures to
   keep the complete draft within the page limit.
7. Separated the standard mean-pool composition AUCs from the independently
   trained regional probes, and documented the missing random/oracle
   predictions and locally unavailable checkpoints.
8. Sharpened C1 into the verified row-major $\rightarrow$ sorted encoder mask
   $\rightarrow$ prefix-truncation chain, separated the unsorted predictor
   path, and added a crop of the stored retained-versus-discarded diagnostic.

## Quantitative provenance

| Claim family and numbers used in `main.tex` | Status | Source |
|---|---|---|
| FairVision split `6,000/1,000/3,000`; `oct_bscans` `200x200x200` uint8; probe samples `100/200` B-scans; test `1,466` positive / `1,534` negative; CC BY-NC-ND `4.0` | `[MEASURED]` | `research/dataset_facts.md`; `research/subgroup_findings.md` Sec. 1; `EVIDENCE.md` C02-C03 |
| ViT-B/16; `16x16` patch grid; probe seed `42`; shared epoch-25 ancestor AUC `0.8487` | `[MEASURED]` | `research/OUTLINE.md` Sec. 3; `EVIDENCE.md` C01-C04 |
| Encoder crop causal chain: grid rows/columns and `r * width + c`; ascending `sorted(best_indices)` at encoder-mask storage; per-group `min_len` followed by `t[:min_len]`. The retained prefix is therefore top-row/small-index context and deletion is bottom-row/large-index context. | code `[MEASURED]`, deterministic spatial consequence `[INFERRED]` | `src/masks/multiblock.py:51-53,103,187-189,216-229`; upstream commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`, `src/masks/multiblock.py:145-175`, as verified in `research/novelty_assessment.md` C1 |
| Predictor masks use an unsorted storage path and a separate global minimum across all predictor groups and samples before `t[:global_min_pred]`; no bottom-deletion claim is made for this path | `[MEASURED]` | `src/masks/multiblock.py:193-210` |
| Rectangle-arm context loss `31-36%`; batch-size audit `B=1` versus `B=64`; audit sizes `n=256` and `n=1,534` | `[MEASURED]` | `EVIDENCE.md` C09-C11; `research/OUTLINE.md` Sec. 4 |
| Zero-anatomy rates: random `0.00->4.63%`, envelope `1.56->10.10%`, COVER `2.34->11.02%` | `[MEASURED]` | `EVIDENCE.md` C12; `research/OUTLINE.md` Sec. 4 |
| Failing-slice accounting: `256 = 90 + 123 + 43`; anatomy `65 = 50 + 15 + 0` | `[MEASURED]` | `EVIDENCE.md` C13; `docs/experiments/masking/crop_and_precision_audit.md:382-403` |
| Retained-versus-discarded panel: `123` sampled non-target cells withheld, including `15` anatomy cells; encoder receives `43` cells and `0` anatomy | `[MEASURED]` | Stored diagnostic `D:\jepa_phase0\reports\arm_stats\zero_anatomy_floor20.png`; reproducible crop in `scripts/make_figures.py::figure_context_excision`; paper asset `figures/fig1b_context_excision.png` |
| Draw dependence: `0%` always blank; `52.0%` never blank versus `44.8%` chance | `[MEASURED]`; draw-dependent conclusion `[INFERRED]` | `EVIDENCE.md` C14; `research/OUTLINE.md` Sec. 4 |
| Composition pass sizes: four rectangle/COVER arms `n=6,137`; blob `n=1,534` | `[MEASURED]` | `research/mechanism.md` Sec. 1 and S1/S9; `scripts/composition_vs_auc.py:8-14` |
| Random composition `53.04/31.58/26.38/18.25/69.09/3.68`, ep50 AUC `.8640971`; mean-pool, linear head, `100` slices, seed `42`, best epoch `46` | `[MEASURED]` | Composition: `research/mechanism.md` Sec. 1, random row. AUC/protocol: `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_random\ep50_results.json` |
| Oracle composition `61.58/39.69/19.34/14.93/77.23/4.19`, ep50 AUC `.8740299`; mean-pool, linear head, `100` slices, seed `42`, best epoch `47` | `[MEASURED]` | Composition: `research/mechanism.md` Sec. 1, oracle row. AUC/protocol: `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_oracle\oracle_ep50.json` |
| COVER `f=.21` composition `73.09/40.88/14.56/9.28/63.62/7.84` | `[MEASURED]` | `research/mechanism.md` Sec. 1, COVER row; S1 |
| Envelope composition `77.58/43.19/11.38/8.63/76.41/8.07`, ep50 AUC `.8761` | `[MEASURED]` | `research/mechanism.md` Sec. 1, envelope row; S1/S8 |
| Blob composition `82.07/97.50/6.26/9.97/160.00/1.24`, ep50 AUC `.8654` | `[MEASURED]` | `research/mechanism.md` Sec. 1, blob row; S1 |
| Purity/AUC points `31.6->.8640971`, `39.7->.8740299`, `43.2->.8761`, `97.5->.8654`; descriptive `40-43%` range; four completed points | values `[MEASURED]`, non-monotonicity `[INFERRED]` | Random/oracle AUCs: the two `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_*` JSONs above. Composition and envelope/blob AUCs: `research/mechanism.md` Sec. 1; `EVIDENCE.md` C15-C16 |
| Ep30 anatomy `.8582+-.0003` vs envelope `.8528+-.0018`; delta `+.0054`; Welch `p=.00219`, `d=4.20`; bootstrap `+.0044`, 95% CI `[+.0010,+.0077]`, `p=.012`; ep50 reversal `-.0107` | `[MEASURED]`; no-shape-superiority conclusion `[INFERRED]` | `research/conflicts.md:116-136`; `research/ablation_inventory.md:47740-47779`; `research/OUTLINE.md:12-21`; `EVIDENCE.md` C23-C27 |
| Blob predictor errors ep30/40/50/56: `.104868/.224373/.271158/.289461`; anatomy/background token values `3.529/.998/.830/.737`; error rise `2.76x` | values `[MEASURED]`, ratio arithmetic and collapse interpretation `[INFERRED]` | `research/mechanism.md` Sec. 3 and S5 |
| Blob target slots/unique cells `64/54.744`; envelope `154.624/118.314` | `[MEASURED]` | `research/mechanism.md` Sec. 1 and S6/S7 |
| Envelope background-position AUC `.870075` versus all-position `.8730`; blob contribution AUC background `.855678` versus anatomy `.846425` | `[MEASURED]`; “positions, not black pixels” `[INFERRED]` | `research/mechanism.md` Secs. 2 and 4, S2/S10 |
| Demographic join `3,000/3,000` on all `16` probe directories; `4` contaminated probes excluded; random/oracle epoch-50 aggregate JSONs have no per-sample predictions and therefore cannot enter the equity analysis | `[MEASURED]` | `research/subgroup_findings.md` Secs. 1-2; inspected top-level contents of the two `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_*` JSONs |
| `12` valid probes from `5` pretraining runs; race counts white `2,318`, Black `431`, Asian `251`; Black lowest in `12/12` | counts/order `[MEASURED]`, five-run independence count `[INFERRED]` | `research/subgroup_findings.md` Secs. 2-3 |
| Race gap `1.97x`, `.0475` COVER ep34 to `.0935` blob ep50; overall AUC spread `.032`; Spearman `rho=.427`, `p=.167` | `[MEASURED]`; no accuracy-fairness tradeoff `[INFERRED]` | `research/subgroup_findings.md` Sec. 4 |
| Shared negative pool `1,534`; severe-to-mild gap `.1306-.1394`; spread `.009` | `[MEASURED]`; policy-invariant interpretation `[INFERRED]` | `research/subgroup_findings.md` Sec. 5 |
| Clean COVER AUCs ep27/30/34 `.8483/.8522/.8571` | `[MEASURED]` | `research/ablation_inventory.md` “cover-f021-clean-interim-probes”; `EVIDENCE.md` C19 |
| Probe heads: attentive d1 `.8706`, mean-pool `.8746`, cross-attention `.8791` on random ep100 | `[MEASURED, DOC-ONLY for d1/cross-attention]` | `research/ablation_inventory.md:47635-47657`; `research/conflicts.md` “frozen-probe-architecture-doc”; `docs/experiments/frozen/mean_pool.md:25-46` |
| Separately trained regional probes, `n=1,000`: random ep50 all/anatomy/background `.8608341/.8746555/.8543614`, all-position best epoch `43`; oracle ep50 `.8682588/.8746475/.8652265`, all-position best epoch `42`. These are not the standard mean-pool composition AUCs. | `[MEASURED]` | `D:\jepa_phase0\reports\downstream_region_auc\random_ep50\region_auc.json`; `D:\jepa_phase0\reports\downstream_region_auc\oracle_ep50\region_auc.json`; `research/ablation_inventory.md:1059-1109` |
| Epoch-100 random `.8745809` versus oracle `.8854852` | `[MEASURED]` | `research/ablation_inventory.md` “random-meanpool-epoch-sweep” and “oracle-vs-random-meanpool” |
| Limit counts: one pretraining seed per policy; `12` probes from `5` runs; `4` retracted COVER-random probes; `4` purity/AUC points; composition `1,534` vs `6,137`; subgroup `431/251` | measured/documented counts; independence interpretation `[INFERRED]` | `research/OUTLINE.md` Sec. 9; `research/subgroup_findings.md` Secs. 2 and 7; `research/mechanism.md` Sec. 1 |

The six slash-separated composition values are, in order: anatomy recall,
target purity, anatomy percentage in context, absolute anatomy cells in
context, total context cells, and zero-anatomy percentage.

## Qualitative and causal wording

- `[INFERRED]` Prefix truncation is spatially biased because sorted row-major
  encoder indices are sliced by prefix. This inference is deductive for the
  encoder path; the predictor path is deliberately left spatially
  uncharacterized because its stored indices are not explicitly sorted.
- `[INFERRED]` The four purity/AUC points are non-monotonic; they do not identify
  a universal optimum.
- `[INFERRED, moderate confidence]` Blob develops predictor collapse. The
  within-arm trajectory supports this diagnosis, but purity, geometry, target
  count, replacement padding, and context size are not isolated.
- `[INFERRED]` Background-position AUC does not establish signal in optically
  black pixels because token features mix globally.
- `[INFERRED]` No accuracy-fairness tradeoff is supported because the measured
  AUC-gap correlation is non-significant.
- `[ASSUMED, excluded from conclusions]` Near-pure anatomy alone causes
  collapse; black pixels contain causal glaucoma biomarkers; or the observed
  subgroup ordering generalizes to another cohort.

## Remaining TODOs

1. Clean COVER floor-0.21 epoch-50 probe. Close with the completed frozen-probe
   `results.json` from the live clean run.
2. Pretraining-seed replication. The archived random/oracle JSONs reference
   `/tmp/ijepa_checkpoints/...` on another machine, and those checkpoints are
   unavailable locally. Close by recovering or recreating them, then running
   replicated continuations for each policy and using continuation, not probe
   seed, as the unit.
3. Matched anatomy-shape comparison. Close with anatomy and envelope runs using
   the same guide cache, hidden budget, target-count policy, and continuation
   protocol.
4. Matched COVER-floor AUC sweep. Close with epoch-50 frozen probes for multiple
   floors; the completed composition-only sweep is insufficient.

## Conflicts and resolutions

- `research/conflicts.md` records a stale document claiming random/oracle
  epoch-50 AUCs were unmeasured. The controlling artifacts are
  `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_random\ep50_results.json`
  and
  `C:\Users\Gary\Desktop\jepa\results\downstream\meanpool_sweep_oracle\oracle_ep50.json`;
  they establish `.8640971` and `.8740299` under the same mean-pool, linear,
  100-slice, seed-42 protocol.
- The evidence base also contains random/oracle regional all-position AUCs
  `.8608341/.8682588`. These are separately trained `n=1,000` regional probes,
  stopped at epochs `43/42`, not the `n=3,000` standard mean-pool runs stopped
  at `46/47`. The manuscript keeps them in the regional ablation and explicitly
  prohibits cross-protocol comparison.
- The two standard mean-pool JSONs contain aggregate metrics but no per-sample
  predictions. Their data and encoder fields point to `/tmp/...` paths on
  another machine, and the checkpoints are unavailable locally. Consequently
  these measured AUCs cannot be locally re-probed, seed-replicated, or included
  in the equity table.
- The completed contaminated COVER campaign’s locked-config text conflicts
  with its deviation/retraction ledger on `amp_target`. The retraction ledger
  and completed run state are controlling; all four probes are excluded.
- Documentation described the COVER floor sweep as running after the stored
  `n=6,137` artifact was complete. The artifact is controlling for composition;
  downstream AUC remains unavailable across floors.
- Earlier prose implied a single `6,137`-slice composition pass for every arm.
  Blob is explicitly labeled as a separate `1,534`-slice pass and is now
  reported that way everywhere.
- Anatomy experiment documentation records an intermediate envelope
  continuation and a different guide cache. The paper therefore says all arms
  share an epoch-25 ancestor, not that the anatomy/envelope trajectories are
  otherwise identical.

## Build verification

- Command: `tectonic -X compile main.tex --keep-logs`
- Result after the full rewrite: `8` pages in `main.log`.
- Undefined citations/references: `0`.
- Overfull boxes: `0`.
