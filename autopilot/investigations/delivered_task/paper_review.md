# Author-side final-reader report — working draft

> Author-authorized, local-only scientific assessment. This is not an assigned
> peer review, acceptance estimate, or editorial outcome. The accountable authors
> should verify each location and decide the final wording.

## Scope and materials reviewed

- Read `paper/genai4health2026/main_submission.tex` completely at the current
  branch state, including the delivered-task appendix, all captions, exclusions,
  secondary analyses, limitations, and reproducibility disclosures.
- Read the current generated result macros/tables needed to interpret the claims,
  plus `mask_report.md`, `training_report.md`,
  `evidence/mask_replay600_v2/full_batch64_confirmation.json`,
  `evidence/training_gpu_guided_v1/joint_summary.json`,
  `literature_matrix.md`, and `terminology.md`.
- Inspected the isolated PDF locally and headlessly. It has 38 pages, all
  612 x 792 pt. `References` begins on PDF page 9 at y=515.8; the appendix begins
  after the bibliography on page 13. Thus the paper has nine body pages as
  represented in the supplied build.
- No manuscript, table, figure, code, configuration, checkpoint, or prediction
  was edited. No training, network call, upload, or external text processing was
  performed.

## Overall reader judgment

A fresh JEPA reader can understand the central problem and most of the delivered
system: this is a 2-D ViT-B/16 B-scan encoder, the downstream volume prediction
comes from mean pooling across per-slice features, and the primary comparison is
a frozen MeanPool + linear probe. The paper also now makes the most important
audit distinction—candidate context, target complement, and final delivered
context—and correctly presents the corrected COVER samplers as implementation
diagnostics without corrected-policy AUC.

The positive CENTROID and ENVELOPE observations remain legitimate descriptive
results. The manuscript is unusually candid about one continuation per policy,
adaptive reuse of the test split, mixed historical precision, exclusions,
selective stopping, and the absence of corrected multi-arm retraining. The
remaining problems are primarily claim calibration and a few exact method
descriptions; none requires new pretraining if the claims remain bounded.

# Comments to authors

## Required revisions

### P0

None identified.

### Major comment P1-1

**Remove the remaining policy-level/“controlled” overclaim.**

- Location: `main_submission.tex:159`, `:423`, `:655`, and `:1902`
  (PDF pages 3, 5, 8, and 27).
- Observation: The phrases “controlled test,” “support H1 at the policy
  level,” “the policy improves AUC,” and “controlled measurement” conflict with
  the paper's own design: one continuation per policy, repeatedly inspected test
  data, and policy-dependent delivered context/target budgets.
- Evidence or criterion: Sections 4–6 explicitly state that the intervals
  quantify test-subject sampling for these continuations, not training-seed
  variation, and that H3 is not identified.
- Why it matters: These four phrases can restore the stronger policy-causal
  reading that the rest of the revision carefully removes.
- Requested action: Make only local substitutions, for example:
  “run-level examination,” “are consistent with H1 in these continuations,”
  “the observed continuation has higher AUC,” and “audited run-level and
  delivered-task measurement.” Preserve the reported positive deltas.

### Major comment P1-2

**State the actual flat predictor and make the guide definition exact.**

- Location: Method background and policy definitions,
  `main_submission.tex:198-230`; guide provenance, `:2052-2055`.
- Observation: The paper never directly says that this implementation
  predicts independently sampled target blocks in parallel, although the
  DSeq-JEPA comparison depends on that distinction. “Identical rectangles” can
  also be read as paired realized draws rather than common nominal
  size/aspect/count rules. Finally, “unmodified MIRAGE occupancy map” is
  ambiguous against the implemented MIRAGE-derived **repaired** envelope.
- Evidence or criterion: `terminology.md` and
  `src/guides/mirage_envelope.py:1-25` define a flat/parallel predictor and a
  repaired RNFL+GCIPL+choroid envelope. The fixed-crop replay explicitly does
  not claim exactly paired placement RNG. CENTROID is recomputed from each
  transformed view; the cached envelope is transformed jointly with its image.
- Why it matters: These are central method facts and the basis for the
  literature distinction from DSeq-JEPA.
- Requested action: Add one sentence stating “parallel prediction, with no
  sequential/causal conditioning.” Replace “identical rectangles” with “the same
  nominal rectangle count, scale, and aspect-ratio distributions.” Call the
  ENVELOPE guide the “MIRAGE-derived repaired envelope,” and clarify that
  “unmodified” means without the residual adapter, not an unrepaired raw union.
  A short per-view transform clause would complete the definition.

### Major comment P1-3

**Resolve the internal macro-provenance contradiction.**

- Location: `main_submission.tex:2067-2077` versus `:2088-2101`
  (PDF pages 28–29).
- Observation: “Every quantity in this paper resolves through a generated
  macro” and “prose, tables and figures cannot disagree” are contradicted two
  paragraphs later by the audit of 310 hand-typed numeric occurrences, including
  75 without a located producing artifact.
- Evidence or criterion: The latter paragraph is the manuscript's own
  disclosure and is consistent with the source comments, which already say that
  macro resolution does not verify every claim.
- Why it matters: This is a literal factual inconsistency in the
  reproducibility section and can reduce trust in otherwise strong provenance
  reporting.
- Requested action: Narrow the first sentence to generated result quantities
  or AUC macros, and say that the gates prevent drift for those fields—not for
  every numeric statement in the paper.

### Major comment P1-4

**Do not use the COVER defect to explain label-efficiency performance.**

- Location: `main_submission.tex:1519-1522` (PDF page 23).
- Observation: The sentence that COVER's low-label result is “consistent
  with its targets being degraded rather than merely differently placed”
  attributes a representation outcome to the implementation defect.
- Evidence or criterion: No corrected COVER policy was pretrained. The
  delivered-task replay and B=2 one-update diagnostics validate mask/loss
  plumbing, not population benefit or downstream AUC.
- Why it matters: This is the one clear place where the engineering finding
  is allowed to shade into a representation explanation, contrary to the main
  text's stronger discipline.
- Requested action: Retain the descriptive fact that historical COVER is
  lowest at each fraction, but replace the explanatory clause with an explicit
  non-attribution: corrected-policy training is absent, so the deficit cannot be
  assigned to truncation rather than the other policy differences.

### Major comment P1-5

**Locally qualify the early ANATOMY-v1 positive result.**

- Location: Abstract `main_submission.tex:55-57`; Results `:460-469`;
  provenance disclosure `:329-337`.
- Observation: The abstract says an early anatomy-shaped “policy improves”
  on its comparator, while the exact historical ANATOMY-v1 launch is not
  retained and the checked-in configuration resumes ENVELOPE epoch 27 rather
  than establishing a direct epoch-25 fork.
- Evidence or criterion: The setup discloses this correctly, but the positive
  result is highlighted remotely from that caveat.
- Why it matters: A reader may treat the early result as a matched policy
  contrast comparable to the primary rectangle-family contrasts.
- Requested action: Preserve the positive observation, but call it an
  “early, incompletely documented ANATOMY-v1 continuation” in the abstract or
  add a compact cross-reference where it is highlighted.

## Material clarity suggestions (P2)

### Minor comment P2-1

**Use the family/implementation count consistently.**

- Location: `main_submission.tex:194`, `:213`, `:246`, and Figure 4 caption
  at `:869`.
- Observation: “Six policies” merges ANATOMY-v1 and ANATOMY-v2 into a
  policy count, whereas the audited vocabulary is five policy families and six
  historical implementations.
- Evidence or criterion: The canonical terminology record treats ANATOMY-v1 and
  ANATOMY-v2 as distinct implementations within one policy family.
- Why it matters: Consistent counting prevents readers from treating the two
  historical implementations as independent policy concepts.
- Requested action: Change only these count labels to “five policy families,
  six implementations” or “all six implementations.” No arm result changes.

### Minor comment P2-2

**Add the statistical boundary to the DSeq-JEPA analogy.**

- Location: `main_submission.tex:136-146`.
- Observation: The factor distinction is fair and no exact equivalence is
  claimed, but the cited discriminative+flat and uniform+flat cells are
  single-run point estimates; the reported 0.4 gap is within the paper's only
  quoted seed spread.
- Evidence or criterion: DSeq-JEPA Table 4 reports the relevant cells without
  cell-level seed intervals; its reported headline seed spread is ±0.4.
- Why it matters: The current wording is directionally correct, but the boundary
  prevents a reader from converting the analogy into a significance claim.
- Requested action: Add “single-run cells” or “without a significance
  inference” to the existing “point estimates” wording. Do not expand the
  related-work paragraph otherwise.

### Minor comment P2-3

**Make the bounded GPU unit unmistakable.**

- Location: `main_submission.tex:1420-1430`.
- Observation: The appendix accurately reports three independent one-update
  policy checks on the first two predeclared views and zero guard changes, but
  does not literally say that each case is the same B=2 batch reset from the
  ancestor.
- Evidence or criterion: `training_report.md` and
  `training_gpu_guided_v1/joint_summary.json` identify one fixed two-view batch
  and three ancestor-reset policy cases.
- Why it matters: The sampling unit determines what the diagnostic can and
  cannot establish about guard frequency or population behavior.
- Requested action: Add “one fixed B=2 batch” if space permits. This prevents
  reading the three cases as three independent sampled batches.

## Strengths that should be preserved

1. **Architecture and evaluation are substantially clear.** The abstract,
   setup, and Figure 1 correctly describe a 2-D B-scan encoder and downstream
   volume-level mean pooling; the full-image EMA teacher is not mislabeled as
   online-encoder leakage.
2. **The intended-versus-delivered task is now intelligible.** The three context
   sets are separated, target truncation is measured after collation, and the
   576-view/573-valid-guide denominators and within-volume dependence are stated.
3. **The repair is not oversold.** Exact-prefix v2 still has 320 valid-guide
   misses; the guard is correctly identified as a second guide-aware
   intervention that may select outside the original context rectangle. No
   corrected-policy AUC is implied.
4. **The GPU evidence is bounded correctly.** The text reports loss
   reconciliation, gradient/update/EMA contracts, and zero guard interventions
   on the predeclared two-view batch; it does not convert instantaneous loss
   differences into policy ranking.
5. **Historical uncertainty is unusually well disclosed.** The ANATOMY-v1
   epoch-27 continuation uncertainty, excluded precision-spliced probes,
   retracted null probes, selective stopping, one continuation per policy, fixed
   data order, and adaptive test reuse are all visible.
6. **Literature distinctions are mostly fair.** DSeq-JEPA is presented as an
   analogous factor combination rather than the same intervention; its salient
   region remains context, and the manuscript does not equate its schedule with
   this project's per-target placement ramp. MAE/SemMAE/AnatMAE evidence is
   explicitly domain- and protocol-bounded.
7. **Figures are captioned conservatively.** Captions disclose truncated axes,
   mixed epochs/precision where relevant, descriptive bars, hand-picked
   attribution examples, and the non-random delivered-task illustration. The
   delivered token-map PDF itself contains the expected four stage labels and no
   raw OCT pixels or AUC claim.
8. **The paper does not need corrected multi-arm retraining to remain honest.**
   The current workshop contribution can be the run-level results plus the
   implementation audit, provided the P1 wording above is corrected.

## Review limitations and bounded unknowns

- I did not run training, re-score predictions, reproduce the numeric proof
  chain, or audit release/build scripts; those were explicitly outside scope.
- I inspected the compiled PDF through local text and geometry only. Most chart
  interiors are raster images, so their literal in-panel labels cannot be
  independently verified from PDF text extraction. No caption/source-text
  quantitative mislabel was found. The delivered token-map source PDF is vector
  text and its stage labels were verified headlessly.
- I did not independently validate MIRAGE segmentation quality or whether
  guide-positive tissue is clinically important. The paper correctly treats
  occupancy as a proxy.
- This report assesses claim/evidence alignment and reader comprehension, not
  venue acceptance.

## Status

**No P0 issue. Five bounded P1 revisions are required, all addressable by
manuscript wording or one-sentence method clarification; no new experiment is
required.** After those revisions, the paper reads as an honest
engineering-first workshop paper: positive CENTROID/ENVELOPE run-level evidence,
mixed anatomy-shaped observations, and a novel delivered-task implementation
finding whose downstream effect remains open.

# Confidential comments to editor

Not applicable. This is an author-requested final-reader working draft, not an
assigned peer review, and it contains no editor-only scientific criticism,
integrity allegation, or editorial recommendation.

## Reviewed-source integrity record

Baseline SHA-256 values recorded before review:

| Material | SHA-256 |
|---|---|
| `paper/genai4health2026/main_submission.tex` | `FB1CF5758DE69AF14DCAB10F466EF5DDC91DCDDBFE1CA407C388753FE3831E22` |
| `mask_report.md` | `A907D3EEDF54F4F532A80CF6FDE8ED2A25463F84575E51B32962476BB976BA6A` |
| `training_report.md` | `1148C1343853E20FE1F324245AAA1330B5AA4A09BED2C8E3156D3F6626B83AD4` |
| `full_batch64_confirmation.json` | `E1F1A3F510FA88CB676689F50362B5A36350A023089C4D314CB91E65A230BCB5` |
| `training_gpu_guided_v1/joint_summary.json` | `49128F3D2592083A00F470084583DF39E25A204DF57D37C7F173960FA51797DE` |
| `literature_matrix.md` | `A6382E67EE58042582C0DA1A576E3C1E5015F2E6E1AC0A9D0346441CCCB7AB97` |
| `terminology.md` | `3FD3ABC509C20D7BC9B1EE8C1A69D260BFF55B5C3998B29AA5DF853C5932CA3D` |
| isolated `main_submission.pdf` | `AC8542765BBB7919C988FBB371238D79BA89DACCB493CDF6F71427399C35EF6D` |

Post-review SHA-256 recomputation matched all eight baseline values exactly.
The only file written for this task was this report.
