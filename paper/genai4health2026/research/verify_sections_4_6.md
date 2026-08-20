# Verification of Sections 4--6

## Summary verdict

Audited 44 numeric or factual claims in `main.tex` lines 140--317.

| Verdict | Count |
|---|---:|
| CONFIRMED | 32 |
| WRONG | 2 |
| UNVERIFIABLE | 3 |
| MISLEADING | 7 |

The composition table correctly distinguishes `ctx_anat` (absolute anatomy-cell count) from `pct_ctx_anat` (percentage of context on anatomy). The paper also keeps the mean-pool composition AUCs separate from the region-restricted probe AUCs.

## Table of every claim checked

| ID | Line(s) | Claim from `main.tex` | Raw artifact and field/path checked | Independently read or computed value | Verdict |
|---|---:|---|---|---|---|
| C01 | 144--153 | “The defect is a three-link causal chain in the mask collator” followed by row-major indexing, sorted encoder storage, and prefix truncation. | `C:\Users\Gary\Desktop\jepa\src\masks\multiblock.py`: `_block_to_indices`, encoder append, `_truncate_and_stack`. | The source contains all three operations and applies them in that order to encoder masks. | **CONFIRMED** |
| C02 | 155--160 | “every index in a lower image row exceeds every index in a higher row… `_truncate_and_stack` retains `t[:min_len]`. It therefore keeps the smallest indices—the top rows—and discards… the bottom rows.” | `src\masks\multiblock.py`: `r * self.width + c`, encoder mask construction, and prefix stack. | For columns `0..width-1`, every index in row `r+1` exceeds every index in row `r`; the prefix therefore retains the uppermost selected rows/cells. | **CONFIRMED** |
| C03 | 160--161 | “The `sorted()` call is what makes this excision deterministic and spatially systematic rather than a random subset.” | `src\masks\multiblock.py`: `_block_to_indices` returns `sorted(indices)` before `sorted(best_indices)` is called at storage. Upstream commit also uses `torch.nonzero(mask.flatten())`, which is row-major ordered. | The spatial prefix effect is real, but the encoder-storage `sorted()` call is redundant in this port: `best_indices` is already row-major/sorted. The ordered representation, not that one call alone, causes the effect. | **MISLEADING** |
| C04 | 162--164 | “In OCT B-scans the retinal band occupies a limited vertical extent, so this is a systematic anatomical excision, not benign padding.” | `D:\jepa_phase0\reports\arm_stats\zero_anatomy_floor20.png`; B=1/B=64 audits in `arm_stats_b1\arm_stats.json` and `arm_stats\arm_stats.json`. | The diagnostic shows a vertically localized retinal guide and a delivered prefix containing zero of 65 anatomy cells (50 were targeted and 15 were withheld); blank-context rates rise strongly with batching for random, envelope, and COVER. | **CONFIRMED** |
| C05 | 169--174 | “Cyan marks 123 non-target cells… including 15 anatomy cells; the encoder receives 43 smallest-index cells… and zero anatomy… [and] includes pre-collation context selection as well as batch truncation.” | Stored source image `D:\jepa_phase0\reports\arm_stats\zero_anatomy_floor20.png`, slice 72; cropped paper image `figures\fig1b_context_excision.png`; generator `scripts\show_zero_anatomy_slices.py`. | Raw diagnostic annotation: anatomy 65; targeted anatomy 50; withheld 123 including 15 anatomy; delivered context 43 with zero anatomy. Generator defines withheld as `(~tgt) & (~ctx)`, so it includes more than batch truncation alone. | **CONFIRMED** |
| C06 | 178--180 | “faithful port of released upstream behavior (I-JEPA commit `52c1…`, mask-collator lines 145--175), not a local defect introduced by our anatomy code.” | Upstream raw file at commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`, `src/masks/multiblock.py`: `torch.nonzero(mask.flatten())`, global `min_keep_enc`, and `cm[:min_keep_enc]`; local `src\masks\multiblock.py`. | Upstream constructs flattened row-major index tensors and truncates encoder masks to a batch minimum by prefix. The local encoder behavior preserves that defect. | **CONFIRMED** |
| C07 | 180--181 | “We found no prior report of this failure mode.” | No finite raw artifact can establish absence from all prior literature, issues, forks, or private reports. | No authoritative artifact proving novelty was found. | **UNVERIFIABLE** |
| C08 | 181--182 | “In our rectangle arms at batch size 64 it removes 31--36% of sampled context, including for plain random masks.” | `arm_stats_b1\arm_stats.json` and `arm_stats\arm_stats.json`, `ctx` fields. | Computed loss: random 36.4129%, oracle 32.9751%, envelope 30.6904%, COVER f=.15 36.1779%. Rounded to whole percentages: 31--36%; random is included. | **CONFIRMED** |
| C09 | 184--188 | “The proof above applies only to encoder/context masks… one global minimum is taken across every predictor group and sample before retaining `t[:global_min_pred]`… no claim about the spatial character of predictor-mask truncation.” | `src\masks\multiblock.py`: predictor append, `global_min_pred`, and prefix stack. | Predictor masks are stored without an additional `sorted()` at append and use one minimum across all predictor groups and samples. The prose does not overextend the encoder proof to predictor masks. | **CONFIRMED** |
| C10 | 193--197 | “random 0.00→4.63%, envelope 1.56→10.10%, and COVER 2.34→11.02%. The audits use n=256 and n=1,534 slices.” | `arm_stats_b1\arm_stats.json`; `arm_stats\arm_stats.json`; `paper\genai4health2026\scripts\make_figures.py`, arm label mapping. | Values and sample sizes are correct. However, 11.0169% is specifically **COVER f=.15**, whereas Section 5 uses COVER f=.21 (7.8377%). The figure itself says “COVER .15,” but the caption does not. | **MISLEADING** |
| C11 | 203--209 | “all 256 patches balance exactly as 90 targets, 123 withheld patches, and 43 patches delivered… 65 anatomy patches… 50 targeted and 15 withheld, leaving zero anatomy visible.” | Stored diagnostic `zero_anatomy_floor20.png`, slice 72; generator category definitions. | `90 + 123 + 43 = 256`; `50 + 15 + 0 = 65`. The withheld category includes context-block selection and batch truncation. | **CONFIRMED** |
| C12 | 211--213 | “0% of slices are always blank, and 52.0% are never blank versus 44.8% expected from the aggregate chance rate.” | `arm_stats\blank_proneness.json`: `overall_blank_pct=6.5104166667`, `never_pct=51.953125`, `always_pct=0`; `scripts\blank_proneness.py`: 12 repeats. | Always blank = 0.0%; never blank = 51.9531%. Independence benchmark: `(1 - 0.0651041667)^12 × 100 = 44.5819%`, which rounds to **44.6%**, not 44.8%. | **WRONG** |
| C13 | 213--214 | “The failure is draw-dependent, so offline filtering cannot identify a fixed set of slices to remove.” | `blank_proneness.json`, including concentration fields. | The audit shows no always-blank slices in 12 draws, but it does not prove that no fixed risk-based filter can be useful. In the same artifact, the top 20% of slices cover 64% of observed blanks. | **MISLEADING** |
| C14 | 222--225 | “Values are percentages except context cell counts and AUC… Random, oracle, COVER, and envelope share an n=6,137 audit; blob is a separate n=1,534 pass.” | `arm_stats_sweep\cover_floor_sweep.json`: each relevant `n=6137`; `arm_stats\arm_stats.json`: blob `n=1534`; composition JSON field semantics. | Sample sizes and units match. `ctx_anat` is the absolute count; `pct_ctx_anat` is the percentage. Available probe `results.json` files use 100 slices, seed 42, mean pooling, and a linear head. | **CONFIRMED** |
| C15 | 233 | Random row: “53.04, 31.58, 26.38, 18.25, 69.09, 3.68, .8641.” | `composition_vs_auc_ep50.json`, `rows[arm=random]`. | `pct_anat_hid=53.04447`; `pct_tgt_anat=31.58322`; `pct_ctx_anat=26.37846`; `ctx_anat=18.25257`; `ctx=69.09092`; `zero_pct=3.68258`; `auc=0.8641`. | **CONFIRMED** |
| C16 | 234 | Oracle row: “61.58, 39.69, 19.34, 14.93, 77.23, 4.19, .8740.” | `composition_vs_auc_ep50.json`, `rows[arm=oracle]`. | `61.58471`; `39.68776`; `19.34386`; absolute `ctx_anat=14.92814`; `ctx=77.22568`; `zero=4.18771`; `auc=0.8740`. | **CONFIRMED** |
| C17 | 235 | COVER f=.21 row: “73.09, 40.88, 14.56, 9.28, 63.62, 7.84, --.” | `composition_vs_auc_ep50.json`, `rows[arm=cover_f021]`. | `73.09452`; `40.87762`; `14.55499`; absolute `ctx_anat=9.27652`; `ctx=63.61659`; `zero=7.83771`; `auc=null`. | **CONFIRMED** |
| C18 | 236 | Envelope row: “77.58, 43.19, 11.38, 8.63, 76.41, 8.07, .8761.” | `composition_vs_auc_ep50.json`, `rows[arm=envelope]`. | `77.58225`; `43.19269`; `11.38393`; absolute `ctx_anat=8.63451`; `ctx=76.41095`; `zero=8.06583`; `auc=0.8761`. | **CONFIRMED** |
| C19 | 237 | Blob row: “82.07, 97.50, 6.26, 9.97, 160.00, 1.24, .8654.” | `composition_vs_auc_ep50.json`, `rows[arm=blob]`. | `82.06625`; `97.50111`; `6.26354`; absolute `ctx_anat=9.96675`; `ctx=159.99739`; `zero=1.23859`; `auc=0.8654`. | **CONFIRMED** |
| C20 | 245--248 | “four completed arms… response is non-monotonic… policy changes also alter shape, target count, context budget, and guide source… observational… not an identified optimum.” | Composition JSON; target-composition JSON; arm configs. | The four AUC-bearing rows are random, oracle, envelope, and blob. Purity/AUC is not monotonic, and the configs and budgets differ on the named axes. | **CONFIRMED** |
| C21 | 252 | “Table… separates what the sampler targets from what the encoder receives.” | Composition JSON field paths: `pct_tgt_anat`, `pct_anat_hid` versus `ctx_anat`, `pct_ctx_anat`, `ctx`, `zero_pct`. | Target and delivered-context quantities are distinct fields and were not swapped in the table. | **CONFIRMED** |
| C22 | 253--255 | “AUC increases from 0.8641 at 31.6% purity to 0.8740 and 0.8761 at 39.7% and 43.2%, then falls to 0.8654 at 97.5%.” | `composition_vs_auc_ep50.json`, four AUC-bearing rows. | Exact stored pairs: (31.5832, .8641), (39.6878, .8740), (43.1927, .8761), (97.5011, .8654). | **CONFIRMED** |
| C23 | 255--256 | “the two highest observed AUCs occur at purities of 39.7% and 43.2%.” | Same four rows. | Highest: envelope .8761 at 43.1927%; second: oracle .8740 at 39.6878%. | **CONFIRMED** |
| C24 | 257--260 | “supports only rejection of a monotonic ‘more anatomy is always better’ rule; it does not estimate a preferred range, a dose response, or a causal purity effect.” | Composition JSON and differing arm configs. | The observed points reject monotonicity across these policies; the confounding prevents an optimum or causal purity estimate. | **CONFIRMED** |
| C25 | 263--265 | “anatomy-shaped encoder reaches 0.8582±0.0003 versus 0.8528±0.0018… +0.0054 (Welch p=0.00219, d=4.20).” | Exhaustive `D:\jepa_phase0\runs\**\results.json` scan for the two ep30 checkpoints. Only `frozen_meanpool_anatomy_ep30\results.json` and `frozen_meanpool_envelope_ep30\results.json` were found. | Available raw seed-42 AUCs are 0.85827430 and 0.85391695 (difference +0.00435735). The other four probe-seed outputs needed to regenerate means, SDs, Welch p, and d were not found. | **UNVERIFIABLE** |
| C26 | 265--266 | “A paired test-volume bootstrap gives +0.0044 with 95% CI [+0.0010,+0.0077] and p=0.012.” | Raw `test_predictions.npz` for both ep30 probes; `scripts\bootstrap_paired_arms.py` method (stratified paired bootstrap, B=2,000, seed 42). | Recomputed: delta `+0.00435735`; percentile CI `[+0.00096820,+0.00776357]`; two-sided bootstrap `p=0.014`. Rounded upper CI is `+0.0078`, not `+0.0077`. A 10,000-resample sensitivity run gave p=0.0126 and CI `[+0.0009307,+0.0078148]`, so it also does not reproduce the printed CI. | **WRONG** |
| C27 | 266--267 | “These error bars vary probe seeds on one frozen encoder per arm.” | Same exhaustive run scan and the two available result configs. | The two stored seed-42 results do point to one checkpoint per arm, but the missing four-seed artifacts prevent verification that the reported error bars were actually computed as stated. | **UNVERIFIABLE** |
| C28 | 267--268 | “envelope uses hard guides while anatomy uses a soft-guide cache.” | `configs\patch_mirage_envelope.yaml`: `mirage_guide_dir=...\mirage_guides`; `configs\patch_mirage_anatomy.yaml`: `mirage_guide_dir=...\mirage_soft_guides\...`. | Guide sources differ exactly as stated. | **CONFIRMED** |
| C29 | 268--270 | “At epoch 50 the direction reverses by -0.0107.” | `frozen_meanpool_mirage_ep50\results.json` = 0.87606410; `frozen_meanpool_bridge_ep50\results.json` = 0.86538550; checkpoint paths in both files. | Difference is `0.86538550 - 0.87606410 = -0.01067860`. However, ep30 uses anatomy-v1 checkpoint `patch_mirage_anatomy`, while ep50 uses the separately configured anatomy-v2/blob continuation `anatomy_v2_ep25`; it is not a within-run reversal. | **MISLEADING** |
| C30 | 274--275 | “The obvious explanation for blob's decline is too little visible anatomy, but the accounting refutes it.” | Composition JSON. | The accounting rejects two simple scalar explanations (mean absolute anatomy count and blank-context rate), but blob still has the lowest anatomy fraction and the design is confounded. It does not refute anatomy availability as a mechanism generally. | **MISLEADING** |
| C31 | 275--278 | “Blob has the lowest anatomy fraction… 6.26%; yet among the guided arms… most anatomy cells… 9.97 versus 9.28… and 8.63… and… lowest zero-anatomy rate 1.24%.” | Composition JSON: `pct_ctx_anat`, `ctx_anat`, `zero_pct`. | Blob: 6.2635%, 9.9668 cells, 1.2386% zero; COVER: 9.2765 cells; envelope: 8.6345 cells. Blob has the lowest zero rate across all five rows. | **CONFIRMED** |
| C32 | 278--280 | “rectangle arms deliver more anatomy still (18.25 random, 14.93 oracle), so absolute count does not order the arms by AUC.” | Composition JSON, `ctx_anat` and `auc`. | Random 18.2526/.8641; oracle 14.9281/.8740; blob 9.9668/.8654; envelope 8.6345/.8761. AUC is not monotonic in absolute context anatomy count. | **CONFIRMED** |
| C33 | 280--281 | “Anatomy starvation, by either absolute count or blank-context rate, is therefore insufficient to explain blob's deficit relative to the guided arms.” | Same composition rows. | Mean count and blank rate alone do not track AUC, but these aggregates do not rule out anatomy amount, spatial placement, fraction, or interaction with target design. | **MISLEADING** |
| C34 | 286--288 | “Marginal value is the prediction error increase per removed anatomy token divided by that per removed background-position token.” | `background_signal\background_signal.json`, each checkpoint’s `ablation`; `background_signal\marginal_token_value.csv`, `v_anat`, `v_bg`, `ratio`. | Example ep30: `(err_drop_anat-err_full)/k = 0.00112577`; background = 0.000318964; ratio = 3.52945. Definition matches the stored calculation. | **CONFIRMED** |
| C35 | 294 | “ep30 0.104868; 3.529×.” | `marginal_token_value.csv`, row `blob_ep30`. | `err_full=0.1048676069`; `ratio=3.5294545903`. | **CONFIRMED** |
| C36 | 295 | “ep40 0.224373; 0.998×.” | Same CSV, `blob_ep40`. | `0.2243728328`; `0.9977315399`. | **CONFIRMED** |
| C37 | 296 | “ep50 0.271158; 0.830×.” | Same CSV, `blob_ep50`. | `0.2711579536`; `0.8301551373`. | **CONFIRMED** |
| C38 | 297 | “ep56 0.289461; 0.737×.” | Same CSV, `blob_ep56`. | `0.2894607930`; `0.7371171316`. | **CONFIRMED** |
| C39 | 302--304 | “Prediction error rises 2.76× from epoch 30 to 56, while preferential reliance on anatomy context disappears.” | Same CSV. | `0.2894607930 / 0.1048676069 = 2.76024982`; anatomy/background value falls from 3.529 to 0.998, 0.830, then 0.737. | **CONFIRMED** |
| C40 | 304--306 | “blob supplies 64 target slots and 54.744 unique cells, versus 154.624 slots and 118.314 unique cells for envelope, and pads shortfalls with replacement.” | `target_composition\summary.json`, rows `anatomy` and `envelope`; `src\masks\utils.py`, `resample_to_k`; blob config `pred_target_k: 16`. | Stored means match exactly. Four targets × 16 gives 64 slots; `resample_to_k` samples a shortfall with replacement. | **CONFIRMED** |
| C41 | 307--308 | “Near-pure targets, connected geometry, target count, padding, and larger context are not separately identified.” | Composition JSON; target-composition JSON; blob/envelope configs and mask source. | All named dimensions differ simultaneously between blob and envelope, so none is isolated. | **CONFIRMED** |
| C42 | 308--309 | “together they support drift toward a weak positional/prototype solution.” | Background-signal and target-composition artifacts. | The artifacts show rising prediction error, reduced marginal anatomy value, and readable background-position signal, but no intervention isolates or directly measures a positional/prototype solution. | **MISLEADING** |
| C43 | 311--314 | “Envelope epoch-50 background-position pooling reaches AUC 0.870075 versus 0.8730 using all positions; this establishes readable signal at background positions, not in black pixels, because ViT tokens mix globally.” | `downstream_region_auc\envelope_ep50\region_auc.json`, `background.test` and `all.test`. | Background = 0.8700750075; all = 0.8729912991. These are correctly used only for the region-restricted probe, not mixed with the .8761 mean-pool composition AUC. | **CONFIRMED** |
| C44 | 314--317 | “blob is the only arm whose background contribution separates glaucoma better than its anatomy contribution (0.855678 versus 0.846425), despite assigning greater per-patch influence to anatomy.” | Four raw files in `patch_attribution\*_ep50_attrib.json`. | Blob: background AUC 0.85567757 > anatomy AUC 0.84642464; random, oracle, and envelope all have anatomy AUC > background AUC. Blob per-patch anatomy influence 0.000415497 > background 0.000359912. | **CONFIRMED** |

## WRONG and MISLEADING claims in detail, with exact corrected text to use

1. **Line 160--161 — MISLEADING causal attribution to one `sorted()` call.**  
   Corrected text: “The row-major ascending ordering—created by `_block_to_indices` and redundantly enforced by `sorted()` at encoder storage—makes prefix truncation deterministic and spatially systematic rather than a random subset.”

2. **Lines 193--197 — MISLEADING COVER label.**  
   Corrected text: “Zero-anatomy context increases from B=1 to B=64: random 0.00→4.63%, envelope 1.56→10.10%, and **COVER f=.15** 2.34→11.02%.”

3. **Lines 211--213 — WRONG expected percentage.**  
   Corrected text: “0% of slices are always blank, and 52.0% are never blank versus **44.6%** expected under 12 independent draws at the aggregate 6.5104% blank rate.”

4. **Lines 213--214 — MISLEADING absolute filtering conclusion.**  
   Corrected text: “The failure varies across draws; the 12-draw audit found no fixed always-blank subset, although blanking risk remains concentrated in some slices.”

5. **Lines 265--266 — WRONG bootstrap p-value and upper CI under the repository’s standard bootstrap.**  
   Corrected text: “A paired stratified bootstrap of the saved seed-42 predictions (B=2,000, seed 42) gives **+0.00436**, 95% CI **[+0.00097,+0.00776]**, and **p=0.014**.”  
   If a different resample count is intended, state it and retain its raw output; B=10,000, seed 42 gives p=0.0126 and CI `[+0.00093,+0.00781]`.

6. **Lines 268--270 — MISLEADING “direction reverses.”**  
   Corrected text: “A separately configured anatomy-v2/blob continuation scores **0.010679 below** envelope at epoch 50 (0.865386 versus 0.876064); because this is not the ep30 anatomy-v1 trajectory, it is not a within-run reversal.”

7. **Lines 274--275 — MISLEADING “accounting refutes it.”**  
   Corrected text: “The accounting does not support mean absolute anatomy count or blank-context rate as sole explanations for blob’s lower AUC.”

8. **Lines 280--281 — MISLEADING general starvation conclusion.**  
   Corrected text: “Neither mean anatomy-cell count nor zero-anatomy rate alone orders the arms by AUC; this does not rule out effects of anatomy fraction, spatial placement, or interactions with target design.”

9. **Lines 307--309 — MISLEADING mechanism inference.**  
   Corrected text: “These jointly changed design factors are consistent with, but do not establish, drift toward a positional or prototype-based solution.”

## Unverifiable claims that should be cut or softened

- **Lines 180--181, novelty:** Replace “We found no prior report” with “We are not aware of a prior report,” or remove it.
- **Lines 263--265, five-probe-seed means/SDs/Welch/d:** The raw run tree contains only the seed-42 result for each ep30 encoder. Restore all five per-seed `results.json`/prediction artifacts or report only the stored seed-42 values: anatomy 0.858274 and envelope 0.853917.
- **Lines 266--267, error-bar provenance:** Keep only if the missing per-seed artifacts are restored. Otherwise say that the available comparison holds both encoders fixed and reports test-volume bootstrap uncertainty.

## Figure-vs-prose consistency

- **`fig1b_context_excision.png`: numeric content agrees**, including 123 withheld cells, 15 withheld anatomy cells, 43 delivered cells, and zero delivered anatomy.
- **`fig1b_context_excision.png`: baked annotation conflicts with the prose caveat.** The source generator `scripts\show_zero_anatomy_slices.py:199` writes “cyan = withheld by CROP,” but `main.tex:172--173,208--209` correctly says that the count includes pre-collation context selection as well as batch truncation. Regenerate with: “cyan = non-target cells not delivered: 123 cells (15 anatomy).”
- **`fig1_crop_defect.pdf`: values and title agree with the audit.** The generated visual explicitly labels “COVER .15,” while the LaTeX caption omits the floor; the caption should also say f=.15.
- **`fig2_composition_vs_auc.pdf`: consistent.** The baked title says “observational; four arms,” plots `ctx_anat` as the absolute count, and makes no optimum or causal claim.
- No figure in these sections mixes the mean-pool random ep50 AUC (0.8641) with the region-restricted random “all” AUC (0.860834). The envelope region comparison at lines 311--314 correctly uses 0.872991/0.870075 from the region probe.

## Artifacts consulted

- `C:\Users\Gary\Desktop\jepa\paper\genai4health2026\main.tex`
- `C:\Users\Gary\Desktop\jepa\src\masks\multiblock.py`
- `C:\Users\Gary\Desktop\jepa\src\masks\curriculum.py`
- `C:\Users\Gary\Desktop\jepa\src\masks\utils.py`
- Upstream `facebookresearch/ijepa` raw `src/masks/multiblock.py` at commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`
- `D:\jepa_phase0\reports\arm_stats\arm_stats.json`
- `D:\jepa_phase0\reports\arm_stats_b1\arm_stats.json`
- `D:\jepa_phase0\reports\arm_stats\blank_proneness.json`
- `D:\jepa_phase0\reports\arm_stats\zero_anatomy_floor20.png`
- `D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json`
- `D:\jepa_phase0\reports\composition_vs_auc\composition_vs_auc_ep50.json`
- `D:\jepa_phase0\reports\target_composition\summary.json`
- `D:\jepa_phase0\reports\background_signal\background_signal.json`
- `D:\jepa_phase0\reports\background_signal\marginal_token_value.csv`
- `D:\jepa_phase0\reports\downstream_region_auc\envelope_ep50\region_auc.json`
- `D:\jepa_phase0\reports\downstream_region_auc\region_auc_summary.json`
- `D:\jepa_phase0\reports\patch_attribution\random_ep50_attrib.json`
- `D:\jepa_phase0\reports\patch_attribution\oracle_ep50_attrib.json`
- `D:\jepa_phase0\reports\patch_attribution\envelope_ep50_attrib.json`
- `D:\jepa_phase0\reports\patch_attribution\blob_ep50_attrib.json`
- `D:\jepa_phase0\runs\frozen_meanpool_anatomy_ep30\results.json`
- `D:\jepa_phase0\runs\frozen_meanpool_anatomy_ep30\test_predictions.npz`
- `D:\jepa_phase0\runs\frozen_meanpool_envelope_ep30\results.json`
- `D:\jepa_phase0\runs\frozen_meanpool_envelope_ep30\test_predictions.npz`
- `D:\jepa_phase0\runs\frozen_meanpool_bridge_ep50\results.json`
- `D:\jepa_phase0\runs\frozen_meanpool_mirage_ep50\results.json`
- `C:\Users\Gary\Desktop\jepa\scripts\bootstrap_paired_arms.py`
- `C:\Users\Gary\Desktop\jepa\scripts\blank_proneness.py`
- `C:\Users\Gary\Desktop\jepa\scripts\show_zero_anatomy_slices.py`
- `C:\Users\Gary\Desktop\jepa\paper\genai4health2026\scripts\make_figures.py`
- `C:\Users\Gary\Desktop\jepa\paper\genai4health2026\figures\fig1b_context_excision.png`
- `C:\Users\Gary\Desktop\jepa\paper\genai4health2026\figures\fig1_crop_defect.png`
- `C:\Users\Gary\Desktop\jepa\paper\genai4health2026\figures\fig2_composition_vs_auc.png`
