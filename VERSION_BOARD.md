# OCT JEPA version and readiness board

> **Completion update:** the engineering and workshop track below has now been
> implemented on `fix/jepa-delivered-task-audit`, with reviewed source release
> `6f4d62e`. The final PDF/ZIP/Word passed their release gates and the 23 managed
> Overleaf source/Word items were rechecked after synchronization. See
> `autopilot\investigations\delivered_task\RESULTS.md` for the completed work,
> limits and actual deliverables. The September 4 audit below is retained as
> the historical baseline, not a list of still-unfixed issues.

**Fresh audit: 2026-09-04. Author-side AI-assisted working draft.**
Baseline: `de145d7005f57e871bc0181bf58b271775d1d25d`.
This board supersedes the readiness assurances in the August 27 conversation.
It does not replace raw artifacts or announce an editorial outcome.

## Bottom line

The saved AUC results are traceable, and the existing PDF/ZIP is buildable.
The paper is a **single-continuation, retrospective policy comparison**, not a
demonstration of statistical equivalence, isolated anatomy causality, native
3D JEPA learning, or clinical utility.

There are real corrections to make before another release: the build pipeline
can overwrite a good deliverable after failure; a fairness summary is false;
some central language exceeds its own limitations; and Word cross-references
are incorrect. The old "everything is done and all gates protect everything"
description was too strong.

## Current version board

| Surface | Version or evidence | Status |
|---|---|---|
| Local code and live GitHub branches | `de145d7`; `main` and `docs/background-signal-findings` agree; last push August 27 | No newer committed research found |
| Working tree before audit | Modified `autopilot\RESOURCE_MONITOR.csv` | Preserved; not our change |
| Canonical manuscript | `paper\genai4health2026\main_submission.tex` | Current paper, not the older local `main.tex` |
| PDF | 36 total pages; References begins at top of page 10 | Nine body pages; existing artifact intact |
| Submission ZIP | `Downloads\OCT_JEPA_GenAI4Health2026_FINAL.zip` | Independently compiled in an isolated directory |
| Word | `paper\genai4health2026\main_submission.docx` | Content present, appendix cross-references defective |
| Overleaf agreement record | August 27, 12:48:31 UTC; 59/59 local hashes agree with stored agreement | Historical local agreement; current remote not authenticated |
| Saved predictions | 43/43 files found and AUCs recomputed | 31 primary, 6 supplementary, 2 excluded, 4 retracted |
| Frozen head recovery | Six cached RANDOM/CENTROID MeanPool heads match manifest SHA-256 | Weights matched; model inference not rerun |
| Replication | One completed epoch after fork; partial epoch 27; no local epoch-50 result | Paused, not running in the observed local process snapshot |
| Actual submission/decision | No OpenReview receipt or decision inspected | Unknown; an internal "Weak Accept" is not acceptance |

The old `normalfix-update` branch name is not a current local/origin branch.
The active history includes position fix `721cd26` and normalization fix
`e625738`. The archived `volume-moe` tag is not the current implementation.

## What the model actually does

One 256 x 256 B-scan becomes 256 patch tokens in a 2D ViT-B/16. The context
encoder and predictor learn against a frozen EMA target encoder. Downstream,
the EMA encoder is frozen; patch tokens are mean-pooled per slice, then over
100 slices, followed by LayerNorm and a linear glaucoma classifier.

The input examination is volumetric OCT, but there is no learned cross-slice
interaction in this MeanPool pipeline. One AUC observation is one volume, not
one slice. The position-vector and normalization corrections remain present.

## Result and terminology board

These are frozen-probe point estimates from the existing continuations.
The rectangle-family table uses its fp16 measurements; other arms' comparisons
must use their matching fp32 controls, not cross-precision subtraction.

| Canonical policy | Old artifact/code name | ep50 AUC | ep75 AUC | ep100 AUC | Interpretation |
|---|---|---:|---:|---:|---|
| RANDOM | `random`, curriculum disabled | 0.8641 | 0.8723 | 0.8746 | Unguided comparator |
| CENTROID | `oracle`, `anatomical_prior` | 0.8740 | 0.8836 | 0.8855 | No segmentation model; intensity-based placement |
| ENVELOPE | `mirage`, `mirage_envelope` | 0.8761 | 0.8803 | 0.8807 | Segmenter-guided rectangle placement |
| ANATOMY-V2 | `anatomy`, `bridge`, `blob` | 0.8654 | 0.8612 | Not measured | Shape/budget/provenance confounds; ep75 is clean fp32 continuation |
| COVER | `cover-f021`, `mirage_cover` | 0.8643 | 0.8639 | 0.8577 | Observed implementation has delivered-target truncation defect |
| ANATOMY-V1 | early `mirage_anatomy` | Not measured | Not measured | Not measured | ep30 AUC 0.8583; checked-in launch path splices from ENVELOPE ep27 |

The README's **0.8947** is a fine-tuned result. The paper's **0.8855** is a
frozen-probe result. Neither should silently substitute for the other.

**Supported statement:** in these continuations on this repeatedly inspected
test split, the implemented ENVELOPE and CENTROID placement policies have
higher matched frozen-probe AUC than RANDOM. Higher tissue purity/coverage is
not monotonically associated with higher AUC across the tested implementations.

**Not established:** CENTROID-ENVELOPE equivalence, location as the isolated
causal mechanism, an expected ranking over retrainings, or a coverage effect.
The CENTROID-ENVELOPE difference is unresolved at ep50/75 and positive at ep100.
No equivalence margin was defined. "Matches a Segmenter" also risks implying a
segmentation-accuracy comparison that was not performed.

## Prioritized action board

These priorities are the coordinator's consolidation, not a sum of overlapping
specialist findings. Detailed reports retain their original assessments.

| ID | Priority | Finding | Minimum action |
|---|---|---|---|
| B01 | P0 release safety | `refresh_all.run()` tolerates failed checks; `p13_build_zip.py` replaces the FINAL ZIP even when `ok` is false and can publish a failed PDF | Fail closed; stage outputs and replace release files only after all checks pass |
| B02 | P1 factual correction | Ethics summary says every subgroup point estimate rises. Divorced-group ep100 AUC instead falls from 0.88280 to 0.87346, n=191 | Scope the summary to tested attributes; retain the exception and avoid all-group reassurance |
| B03 | P1 claim scope | Title/equivalence, "H1 holds", and "precision does not add" are stronger than the comparison identifies | Describe run-specific policy results; remove equivalence and isolated-mechanism implications |
| B04 | P1 protocol disclosure | Background narrative joins random ep100, envelope ep100 and a smaller ep50 regional probe into one explanation | Identify each arm, epoch, slice count and subset; state mechanism remains unresolved |
| B05 | P1 reproducibility | Numerical/citation checks have empty/skipped false-success paths; 6/6 contains no numeric check | Require explicit expected coverage, nonzero failure exits and retained per-item evidence |
| B06 | P1 regeneration | `refresh_all` omits 11 included raster figures and DOCX; sync pushes 31 extra managed files outside the compiled stage | Declare asset dependencies; separately validate collaborator attachments |
| B07 | P1 collaborator copy | Word contains all figures/tables/references, but 36 appendix references use numeric sections instead of letters; References heading missing; checker always exits zero | Repair conversion and make structural completeness checks fail on mismatches |
| B08 | P1 provenance/status | Paper says replication is running; no completed local replication. ANATOMY-V1 config uses ENVELOPE ep27 as parent | Correct status; disclose splice or recover historical launch evidence rather than assuming direct ep25 fork |
| B09 | P2 entry-point/docs | README, experiment index, changelog and generic configs describe different generations/protocols | Name one canonical current reproduction route; label historical alternatives |
| B10 | P2 future-run reliability | Resume omits RNG/best-validation state; partial accumulation and non-prefix reseeding have defects | Address before future campaigns; do not claim existing named-epoch AUCs are thereby invalid |
| B11 | P2 precision/clarity | "3D" means volumetric data with 2D encoding; label-efficiency, calibration, attribution and gender terminology need bounded wording | Clarify constructs and descriptive scope without adding results |
| B12 | P2 tooling | `upload_weights.py --list` crashes on an explicit-source arm; download defaults select legacy weights | Repair inventory handling and distinguish legacy flags from current `--arm` |

B01 threatens **future publication operations**, not the integrity of the
already-inspected current ZIP. B02 is an actual counterexample to a sentence,
not evidence of statistically significant subgroup harm. B08's ANATOMY-V1
splice is established for the checked-in configuration; the historical
checkpoint's exact producing launch remains unverified.

Known limitations are not new discoveries: single continuation, adaptive test
reuse, the COVER defect, no external clinical validation, and confounded
cross-family budgets. No long GPU experiment is automatically required just
to make the current account honest.

## Version history worth retaining

| Version | Meaning |
|---|---|
| April 10: `721cd26`, `e625738` | Position and input-normalization corrections in the active ancestry |
| `42aab52` | Method schematic and EMA encoder clarification |
| `d5ee750`, `226d51b` | Dash and semicolon edits; stylistic preferences, not scientific validation |
| `5cf230e` | Some R11 wording and probe-count checker corrections |
| `b3e3e77`, `60033e8` | Geometry regeneration and paired-difference ladder figure |
| `54dac13` | Citation helper and historical style research |
| `b6c3f35` | Current keyword-led title |
| `e785668` | August 27 synchronization/state commit |
| `74b00dc`, `de145d7` | Word attachment plus latest published PDF/log commit |
| This September 4 pass | Local audit board and evidence reports only; no manuscript, code, training or remote changes |

## Venue and outstanding authority

The live workshop page confirms **September 5, 2026, 23:59 AoE**, equivalent to
**September 6, 04:59 PDT**. Research papers have nine content pages and no
rebuttal stage. Author-list completion matters because additions after the
deadline are disallowed.

Public repository visibility alone was previously described too categorically
as a desk-rejection trigger. The workshop bans identifying submission content
and non-anonymized links and permits non-archival preprints. Its own rules
control; the main-track handbook separately permits public preprints. Audit
the actual submitted package rather than infer a violation from visibility.
No visibility or credential change was made.

Sources checked September 4:

- https://genai4health.github.io/2026-NeurIPS/
- https://neurips.cc/Conferences/2026/MainTrackHandbook

## Mixed-model and skill routing

This pass used three separate GPT critics and one Opus research thread.
No global model settings or installed skills were changed.

| Role | Model used | Observed contribution |
|---|---|---|
| Coordinator | GPT-6 Astra | Live version/state snapshot, independent saved-AUC and head-hash checks, source adjudication, this board |
| Scientific critic | GPT-5.6 Sol Fast, xhigh | Full manuscript and selected analysis review; concrete fairness counterexample and claim/protocol gaps |
| Implementation critic | GPT-5.6 Sol Fast, xhigh | Active pipeline trace, CPU checks, confirmed fixes and bounded resume/collation issues |
| Delivery critic | GPT-5.6 Sol Fast, xhigh | Isolated package build, destructive-publication reproduction and DOCX/checker gaps |
| Research synthesis | Claude Opus 5, xhigh | Public methodological precedent and alternative research designs, subject to coordinator correction |

The GPT threads returned after roughly 14 minutes of observed launch-to-result
elapsed time, with tool work included. Opus reported roughly nine minutes for
its recorded active research pass, while launch-to-result elapsed time was
roughly 16 minutes; a corrective follow-up was then required. Those are different
clocks and different tasks, not evidence of a general speed or capability ranking.
Parallel work reduced waiting, but consumed several agents.
The Opus thread reached its revised handoff after roughly 31 minutes of
launch-to-read elapsed time including feedback and waiting; this is not active
model-compute time.

The Opus first draft made two methodological errors that the coordinator rejected:
identical labels do not prove subject-ID pairing, and a matched-epoch AUC
difference need not exceed across-epoch improvement to be detectable. It also
initially proposed work partly duplicating existing mask-geometry audits.
This is why broad synthesis should propose alternatives, not define the final
statistical acceptance rule unreviewed. The project interpreter has SciPy 1.17.1;
an agent's different environment is not evidence that a project dependency is absent.

| Task shape | Useful skill | Role pairing |
|---|---|---|
| Claim/evidence appraisal | `peer-review` | Focused critic, then coordinator adjudication |
| Literature and reference metadata | `citation-management` | Research model; distinguish source existence from support |
| Uncertainty, sampling units, planned analyses | `statistical-analysis` | Statistical reasoning plus deterministic computations |
| Figure truthfulness and readable encodings | `scientific-visualization` | Design review plus source-linked plotting code |
| Paper structure and section revisions | `ml-paper-writing` | Evidence-bounded writing, then an independent critic |

Skills specify a procedure; they do not supply extra models or guarantee correct
results. Prefer direct deterministic tools for arithmetic, file parity and
artifact validation. Use model diversity for genuinely different reasoning
roles, not repeated parallel reads of the same small scope.

### Research ideas retained after adjudication

1. **Existing-prediction provenance and epoch sensitivity:** recover case/order
   manifests and report the already-matched epoch comparisons descriptively.
   Across-epoch improvement is not a noise floor. If ordering differs, realign
   by case ID or re-export; do not assume that makes the samples independent.
2. **Incremental delivered-budget audit:** build on the existing 600-slice
   measurements, add distribution-level post-collation evidence at production
   batch size, and specify a budget-matched future control. This is diagnostic,
   not proof of a causal mediator.
3. **Authorized continuation replication later:** the six-leg plan remains the
   reference. A reduced four-leg pilot is an option to evaluate, not an approved
   plan change or an automatic path to a stronger claim. Resolve resume/provenance
   issues first. The logged compute floors are roughly 144 hours for six legs and
   96 hours for four, excluding overhead; neither is a precise completion ETA.

No new experiment or manuscript result was authorized by these proposals.

## Evidence and review coverage

Reports in `autopilot\reports\fresh_audit_2026-09-04`:

- `repository_state.md`: version, artifact availability and bounded process snapshot.
- `prediction_snapshot.json` and `measure_snapshot.py`: reproducible saved-score AUC audit.
- `science.md` and `science_findings.json`: full manuscript, captions, generated tables and selected statistics.
- `code.md`: active patch pipeline, configs, targeted CPU invariants and limitations.
- `delivery.md`: isolated ZIP compile, OOXML comparison and synthetic gate-failure reproductions.
- `context_corrections.md`: corrections to prior assistant statements.
- `research_strategy.md`: bounded research proposals and public precedent; the
  coordinator's accepted scope and adjudication take precedence over hypotheses.
- `claim_evidence.csv`: eleven explicitly scoped claims and actions; validation
  checks its structure, not scientific truth.

No full retraining, new bootstrap analysis, new clinical experiment, exhaustive
archival-file audit, case-ID reconstruction, authenticated Overleaf check or
actual OpenReview submission check was performed. The broad repository
inventory is not a claim to have read every historical file.
