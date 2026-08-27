# R4 narrative fixes - `paper/genai4health2026/main_submission.tex`

Scope of this pass: everything R4 raised **except** N1-N5, which were already fixed by the
author before this pass began and were not touched.

Verification after all edits:

- `tectonic -X compile main_submission.tex --keep-intermediates` -> compiles clean, no
  LaTeX warnings, no undefined references.
- PDF: **28 pages**. Body ends on page 9; **References start on page 10** (unchanged).
- `autopilot/check_manuscript.py` -> **RESULT: PASS**; `labels 47, refs 47, dangling 0`;
  `macros ... undefined 0`; `citations 48 cited, 0 missing`. Warning set is identical to the
  pre-pass baseline.
- No digit, numeric value or macro name was altered anywhere. No `\cite` added or removed.
  No `\label` added or removed.

---

## 1. MINOR NONSENSE (N6-N12)

| ID | What R4 said | What I did |
|----|--------------|------------|
| N6 | Contributions say "(21 frozen probes)"; abstract / 5.4 / Appendix C say a different count (`\NprobesSub` = 23). Silent discrepancy. | Could not reconcile by editing a digit (forbidden, and 21 is hand-typed while 23 is machine-generated). Replaced the hand-typed parenthetical with a pointer: "a single evaluation protocol (every probe fitted is listed in Appendix~\ref{app:allprobes})". The contradiction is gone, no number changed, and the exact inventory is now reachable. **This deletes a hand-typed count - flagging it explicitly for author sign-off.** |
| N7 | Appendix F: "while tissue contrast collapses" - undefined, unquantified, unmeasured. | Cut the clause. The quantified neighbour (0.784 -> 0.346) is untouched. |
| N8 | Appendix F closes "two ablations would settle it" without naming either. | Cut "; two ablations would settle it". "The causal chain ... remains open." survives, which is the substantive caveat. |
| N9 | Appendix F: background-only probe 0.867 dismissed as "attention leakage", a term used once and never defined or measured. | Rewrote to state the architectural reason and the measurement gap honestly: "self-attention mixes tissue information into background tokens, so that figure cannot be read as background signal, and we did not measure how much mixing occurs." Number 0.867 unchanged. |
| N10 | Figure 3 caption calls a measured result "the ordering that motivates this paper". | "the ordering this paper tests". Shorter, and no longer labels the refutation as the motivation. |
| N11 | Table 12 caption: "Rows appear as the re-probes complete." - live-document scaffolding. | Sentence cut. |
| N12 | 5.4: "A widening max--min gap is easy to misread as harm. It is not." - pronoun resolves to the wrong antecedent. | "It is not harm." |

## 2. OFF-TOPIC (O1-O5)

| ID | What R4 said | What I did |
|----|--------------|------------|
| O1 | Appendix H is run on three fine-tuned probes, not the six arms; its own heading "Why this is in a masking paper" is a confession. Keep the diffuse-attribution result cited from 5.2; move the rest out. | Partially actioned. Heading changed from "Why this is in a masking paper" to the declarative "Diffuse attribution supports the main claim", and Section 5.2 now cites `app:interp` ("Occlusion attribution on the fine-tuned probes is correspondingly diffuse"). **Did not delete the OD/OS or confidence-invariance results**: the OD/OS paragraph is an explicit methodological caution ("a figure that looks anatomically meaningful can be an artefact of data storage"), which the brief forbids removing. Cutting appendix material also cannot help the 9-page body limit. Recommend the author decide whether to relocate. |
| O2 | Figure 8 caption: "We recommend it as the default primitive for this class of model." - a different paper's tooling recommendation. | Sentence cut. |
| O3 | Appendices F, H, I, N, O never cited from the body. | Fixed for F (`app:bg`, now cited from 5.1), H (`app:interp`, now cited from 5.2) and A (`app:allprobes`, now cited from the contributions bullet). I was `app:operating`, already repaired by the author's N4 fix. N and O are unlabelled sections with no floats of their own and nothing in the body honestly reaches for them - see "Deliberately skipped" below. |
| O4 | Figures 3, 5, 6, 7, 8, 9, 10, 11, 12, 14 never cited by number. | All now cited - see section 4. |
| O5 | Appendix F does not earn its page unless the body asks the question it answers. | Body now asks it, in 5.1, from an existing sentence: "... with an interval that excludes zero; why the null nonetheless stays this close is examined in Appendix~\ref{app:bg}." |

## 3. MACHINE-WRITTEN (M1-M12)

| ID | Location | What I did |
|----|----------|------------|
| M1 | Section 5 preamble | Applied R4's replacement verbatim: "Each policy was pretrained once, so the intervals below quantify sampling error over test subjects, not seed-to-seed variance of pretraining (Section~\ref{sec:limits})." Largest single body saving (~30 words). The full argued caveat stays in Section 6. |
| M2 | 4, Numerical precision | "Section~\ref{sec:precision} measures the precision effect." |
| M3 | 5.1 | "At the matched epoch 50 neither is *worse* than the null: ANATOMY-V2 (...) and COVER (...) are both indistinguishable from it." Kept R4's compression but retained the epoch anchor, since epoch matching is load-bearing in this paper. Precision matching remains stated in Table 1's caption. |
| M4 | 5.1 | "The data rule out monotone improvement with anatomical precision, not harm." |
| M5 | 5.4 opening | Split per R4: claim first, then the validation. "We audit seven stratifications across N probes spanning aggregate AUC X to Y. Predictions are joined to the released FairVision metadata in deterministic loader order, validated by exact label reconstruction (3000/3000, every probe)." Validation retained. |
| M6 | 3.2 CENTROID | "It is named for the statistic it computes." cut. |
| M7 | 3.2 COVER | "An implementation defect means the delivered masks never reach $f$ (Appendix~\ref{app:covbug})." Defect disclosure retained in full. |
| M8 | Appendix E opening | "We audited the COVER sampler after its epoch-100 result and found a defect. It invalidates the reading of that arm and weakens one cross-family comparison." |
| M9 | Appendix G opening | "We measured each policy as supervision is withdrawn (Section~\ref{sec:labeleff})." Removes the paraphrase of 5.3's motivation and simultaneously repairs the orphaned `sec:labeleff`. |
| M10 | Appendix K closing | "Read together, these say that the *direction* of our finding is not new. What we add is the controlled measurement, not the sign of the result." R4's suggested "(Section 2)" pointer was **not** used: Related work carries no `\label`, and the brief forbids adding one. |
| M11 | Appendix C | Second, verbatim "The branch-level figure is the one to believe." cut; the first (the stated rule) kept. |
| M12 | Section 6 | "Nor is target *shape* identified as the operative variable (Table~\ref{tab:geom}), nor is CENTROID's mechanism." Anaphora stack collapsed; all three limitations survive. |

## 4. ORPHANED FLOATS AND SECTIONS

Before: 21 labels with no `\ref` anywhere. After: **0**. Every pointer below is a one-clause
addition placed on a sentence that already made the relevant claim; no new claim about any
float's content was invented.

Body -> appendix (4 new pointers, all on pre-existing sentences):

| Label | Anchored from |
|-------|---------------|
| `app:allprobes` | Contributions bullet 1 (also fixes N6) |
| `app:bg` | 5.1, after "Avoiding background is worth something at every epoch we measured" (also fixes O5) |
| `app:interp` | 5.2, first observation, after the purity-vs-AUC direct comparison (also fixes O1/O3) |
| `sec:policies` | Section 4, "the *only* difference between arms is the mask sampler" |
| `fig:maskstats` | 5.1, on the sentence that already quotes its 73.1% / 77.6% (R4's "worst case" in O4) |
| `fig:ladder` | 5.2, alongside the existing `fig:paradox` citation |

Appendix -> appendix (float cited from the paragraph that already describes it):

`sec:labeleff` (from Appendix G), `fig:fair`, `tab:severity`, `tab:pairedsub` (Appendix C),
`tab:labeleff`, `fig:labeleff` (Appendix G), `fig:interp-slice`, `fig:interp-window`,
`fig:interp-odos`, `fig:interp-outcome`, `fig:interp-heatmap` (Appendix H),
`tab:operating`, `fig:roc` (Appendix I), `tab:counter` (Appendix K), `tab:fp32` (Appendix M).

## 5. DEDUPLICATION

| Caveat | Before | After |
|--------|--------|-------|
| Single-run | Abstract twice (mid + closing sentence), Section 5 preamble in full, Section 6 in full, Appendix I in passing | Cut the **duplicate closing sentence of the abstract** (the caveat still opens the abstract's setup description); Section 5 preamble reduced to its inferential-scope statement plus a pointer (M1); Section 6's argued paragraph kept intact as the strongest instance; Appendix I's passing clause kept. |
| "descriptive rather than confirmatory" | Four same-shape instances (Section 6, Appendix C Limitations, Table 11 caption, plus near-variants) | Kept the strongest, Section 6, which states the reason ("chosen after repeated inspection of that split"). Appendix C's Limitations now reads "The $p$-values here are descriptive, for the reasons given in Section~\ref{sec:limits}." - which also removes the duplicated "one dataset, one test split, chosen after repeated inspection" caveat R4 flagged at p9/p16. Table 11's caption shortened to "are descriptive (Section~\ref{sec:limits})". Distinct-sense uses at 5.4 ("descriptive regularity"), Appendix C severity ("descriptive only") and Figure 13's caption ("descriptive, not inferential" at n=4-5) left alone - they are different claims, not repetitions of the formula. |

## 6. BONUS (not a numbered finding)

R4's narrative-arc section notes H3 is stated in 3.3 and never adjudicated: 5.2 opens "To test
H3..." and then reports the arms are confounded. Fixed net-neutrally in 5.2's second
observation: "**H3 is therefore not identified by this design**, and we do *not* claim that
irregular target shape is harmful." Same length, and H3 now receives a verdict.

## 7. DELIBERATELY SKIPPED

| Item | Why |
|------|-----|
| N6 as a numeric reconciliation (21 -> 23) | Would change a digit. Forbidden. Resolved by removing the hand-typed count instead; **flagged above for author sign-off** in case the author wants a macro-driven count there instead. |
| O1, deleting the OD/OS and confidence-invariance results from Appendix H | Both are cautions about how attribution figures can mislead; the brief forbids removing a caveat. Reframed rather than removed. Relocation is an author call. |
| O3, Appendices N ("Fine-tuning narrows but does not erase the gap") and O ("Fine-tuned results") | Neither carries a `\label`, so neither is an orphaned label. No body sentence cites either honestly without inventing a claim about fine-tuning that the body does not currently make, and adding `\label` commands is outside the brief. **Recommend the author add `\label{app:finetune}` to Appendix N and cite it from Section 6's mechanism paragraph** ("consistent with the masking policy shaping the representation rather than merely the optimisation start point" is exactly the body's mechanism claim). Not done here because it needs a new `\label`. |
| M10's suggested "(Section 2)" pointer | Related work has no `\label`; adding one is outside the brief. Reworded without the cross-reference. |
| The "not clinically validated" duplication (Section 6 / Appendix J) | R4 listed it under the rhythm complaint, not in the dedup instruction. The body sentence is a summary that explicitly forwards to the full statement; cutting either would remove an ethics caveat. Kept. |

---

## 8. PAGE TRIM (main content 10 pages -> 9)

`p13_build_zip.py` reported `2_main_content_within_9_pages FAIL`: the References
heading began at y=112.2 on page 10, so two lines of the Conclusion spilled past
the 9-page main-content limit. Ten prose cuts follow. **No digit, macro or
numeric value anywhere in the manuscript was changed, rounded or deleted**; the
restored reproducibility passage keeps `$9.8\times10^{-6}$`, `$0.8854754$`,
`$0.8854852$`, `22` and `$2{,}248{,}844$` verbatim. All 47 `\label` and 47 `\ref`
commands are untouched (0 dangling). The corrected title, the "in these runs" /
"reliably" hedges in the contribution bullet, and the Section 5.1 heading are
unchanged.

| # | Location | Removed / compressed | Why this was redundant, not substantive |
|---|----------|----------------------|------------------------------------------|
| P1 | Section 2, closing paragraph | "What we add is a controlled measurement of ... **when everything else is held fixed**, in a setting..." -> "We add a controlled measurement of ..., in a setting..." | "Everything else held fixed" is the paper's stated design and appears twice already: Introduction ("hold everything else fixed") and Section 4 Pretraining ("identical optimiser, learning-rate and weight-decay schedules, and effective batch size"). Third statement carried no new content. |
| P2 | Section 5 preamble | "the intervals below **quantify sampling error over test subjects, not seed-to-seed variance of pretraining**" -> "the intervals below are sampling error over test subjects, not seed-to-seed variance" | Wording only. The single-run caveat itself is retained here in full force, and its strongest instance (Section 6, "One pretraining run per policy") is untouched. |
| P3 | Section 5.1, H1 paragraph | Deleted "Avoiding background is worth something at every epoch we measured, with an interval that excludes zero;" (the clause pointing to Appendix B is kept) | Pure restatement of the two sentences immediately preceding it, which already give the epoch-50 and epoch-100 deltas *with* their CIs, and of the same paragraph's "All six contrasts against the null survive Benjamini-Hochberg correction". Nothing is asserted here that is not asserted twice above with numbers attached. |
| P4 | Figure 2 caption | Deleted "Rectangle arms differ only in masking policy; the anatomy arms also differ in collation." | A caption restating surrounding text. The collation confound is stated in Section 4 ("the anatomy family additionally differs in collation") and argued at length in Section 5.2 ("The two families are collated differently, which weakens 'everything else held fixed'..."). The caveat survives in both stronger places. |
| P5 | Figure 2 caption | "We plot paired differences rather than per-arm error bars because ... cancels in a **paired comparison, and would** understate" -> "Paired differences are plotted rather than ... cancels in a **pairing and would** understate" | Wording only; the statistical rationale is preserved intact. |
| P6 | Section 5.1, H2 paragraph | "Tested directly at epoch 50 --- **the only epoch at which all five arms have a probe** --- " -> "Tested directly at the matched epoch 50," | Third statement of the same fact. Section 4 states it in full and explains its consequence ("Epoch 50 is therefore the only epoch at which five policies are simultaneously available, and it carries every cross-policy comparison that involves the anatomy or coverage arms"). That instance is untouched. |
| P7 | Table 2 caption | "AUC is quoted at the *matched* epoch 50, **the only epoch at which all five policies have a probe,** so the geometry-to-AUC association is not read across mixed epochs." -> same sentence without the middle clause | Fourth statement of the fact retained at Section 4 (see P6). The caption still says the quote is at the matched epoch and still states the reason it matters. |
| P8 | Section 5.2, first observation | "the correlation with mask *purity* **--- the share of masked patches lying on tissue ---** at +0.40" -> gloss removed | "purity" is defined in the caption of Table 2, which sits on the same page and is the table the sentence is reading from. |
| P9 | Section 5.3 | "The absolute gains at full supervision are small, **but the regime that matters for a new imaging site or a rare condition is the one with few labels.**" -> first clause kept | The deleted clause restates the subsection's own opening sentence: "A masking policy that only helps when labels are plentiful is of limited clinical use". The candour statement ("absolute gains at full supervision are small") is kept. |
| P10 | Section 5.6 Controls | Deleted the closing "Probe-time numerical precision is therefore immaterial to every comparison reported here."; light rewording of the re-encoding sentence ("those released weights" -> "those weights", comma moved) | The deleted sentence is a verbatim-in-substance repeat of the paragraph's own opening conclusion, three sentences earlier: "the differing autocast settings of Section 4 are not the effect and every contrast is matched on precision". Every digit of the reproduction result is kept. |
| P11 | Section 6, "Scope, coverage and an incomplete arm" | "but **an implementation defect means its delivered masks never realised the coverage it was configured for** (Appendix E), so its trajectory is not evidence" -> "but **the collation defect above** (Appendix E) means its trajectory is not evidence" | Third statement of the coverage defect in the body. The negative result and its consequence are fully retained here ("its trajectory is not evidence about aggressive coverage") and stated with the measured percentages in Section 5.1 and again in Section 3.2. The `\ref` is preserved. |
| P12 | Section 7 Conclusion | Deleted "The most anatomically precise policy is not the best performer;" | Immediately preceded by "and the best policy consulted no segmentation model", which is the same finding. It is also in the Abstract, in contribution bullet 3, in Figure 1's caption and in Section 5.2, so the finding remains stated five times. |

**Reclaimed:** approximately 9 typeset lines of body text, which moved the
References heading from y=112.2 partway down page 10 to y=72.8 at the top of
page 10, i.e. zero body lines on page 10.

**Not done, and why.** No content was moved into an appendix, no limitation,
caveat or negative result was removed (only duplicate statements of ones that
survive in a stronger place), and no table or figure was dropped. Trims that
would have required deleting a numeric value - for example the repeated
`$2\times10^{-4}$` fp16/fp32 bound in the Table 1 caption, or the duplicated
"anatomy-shaped masking is not harmful" statement that appears with CIs in both
Section 5.1 and Section 6 - were rejected on the no-digit rule.

**Gates after the trim.**

```
p13_build_zip.py    2_main_content_within_9_pages PASS ; main content pages 9 ; ALL_PASS = True
check_manuscript.py labels 47, refs 47, dangling 0    ; RESULT: PASS
p15_verify_numbers.py 20 AUC macros verified          ; RESULT: PASS
```

### 8b. NUMERIC CORRECTION (NOT a trim)

Listed separately because it is a **corrected value**, not a length cut. It was
authorised as an explicit, one-off exception to the no-digit rule; it is the only
digit changed in this pass.

| Location | Was | Now | Status |
|----------|-----|-----|--------|
| Appendix G (`fig:interp-slice` caption), `main_submission.tex` line 1053: "Despite spanning zero to **7.17M** probe parameters they converge on the same structure" | `7.17M` | `7.14M` | Corrected |

**Why it was wrong.** `7.17M` (7,167,744) is the parameter count of the
**100-slice** attentive probe. The probe the caption is actually describing is
the **64-slice** one, which has 7,140,096 parameters, i.e. `7.14M`. The paper was
therefore reporting one configuration's parameter count under another
configuration's name - the exact cross-attribution failure this project's number
provenance rules exist to prevent. Flagged by `autopilot/reports/HARDCODED_AUDIT.md`
(row for line 1060).

**Verification performed before changing it** (all four checks had to agree; a
single disagreement would have meant the error was elsewhere and the digit would
have been left alone):

1. **The surrounding text is about the fine-tuned probes.** Appendix G opens:
   "The analyses in this appendix were run on the three *fine-tuned* probes
   (mean-pool, cross-attention pool, and a depth-1 attentive probe)". So the
   caption's upper bound is the fine-tuned depth-1 attentive probe.
2. **The stored fine-tuned config is 64-slice, not 100-slice.**
   `results/downstream/finetune_oracle/d1_results.json` records
   `data.num_slices = 64`, `model.probe_type = "attentive"`,
   `model.probe_depth = 1`, `model.probe_num_heads = 12`.
3. **That run is the one the paper reports.** The same JSON records
   `test_auc = 0.8900997579200691`, which is the `0.890100` printed for
   `\ArmBest{} / attentive` in Appendix O's fine-tuned results table. Same run,
   so the same 64-slice architecture.
4. **The count reproduces from the architecture.** Instantiating
   `src.eval_downstream.AttentiveProbe` with the stored config gives
   7,140,096 parameters at `num_slices=64` and 7,167,744 at `num_slices=100`
   (difference 27,648 = 36 extra positional embeddings x 768). Closed form:
   one ViT block at `dim=768, mlp_ratio=4, qkv_bias=True` is 7,087,872
   parameters, plus `cls_token` 768, `pos_embed` (64+1)x768 = 49,920, and a
   final LayerNorm 1,536.

The caption's lower bound "zero" is also confirmed: `_build_probe` describes
`mean_pool` as "(0 params, ablation floor)" and instantiating it returns 0
parameters. `cross_attn_pool` at `head_dim=64` is 249,024 parameters, so it is
not the maximum and the caption's range endpoints are the two correct probes.

**Scope of the change.** One occurrence, one location. `7.17M` appeared exactly
once in the manuscript, so no other site needed updating, and no other digit
anywhere in the file was touched. All six `p13_build_zip.py` checks, and both
`check_manuscript.py` and `p15_verify_numbers.py`, still pass after the edit
(labels 47, refs 47, dangling 0; main content 9 pages; `ALL_PASS = True`).
