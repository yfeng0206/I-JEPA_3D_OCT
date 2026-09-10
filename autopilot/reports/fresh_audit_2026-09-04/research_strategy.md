# Research strategy: narrow contribution, post-audit ideas, and work design

> Coordinator closure: the proposals below are unrun options. Final adjudication
> also removed an automatic unpaired-analysis fallback, an assumption that existing
> configs need no resume fixes, and an unsupported attribution of timing differences
> solely to orchestration overhead. `VERSION_BOARD.md` states the accepted scope.

Author-side fresh audit working draft, 2026-09-04. Research/synthesis role.
Baseline commit: `de145d7005f57e871bc0181bf58b271775d1d25d`.

**What this document is.** A strategy memo produced by one bounded research pass:
a strategic thesis about the strongest defensible claim for this evidence class,
three research ideas for *after* the current audit, verified public methodological
precedent, and a provisional routing plan for models and skills. **Revised 2026-09-04
after coordinator audit;** §6 records which inferences were rejected and why. The parent
agent independently validates every claim here against its own findings.

**What this document is not.** Not an acceptance-odds estimate, not a review, not a
manuscript edit, not an authorization to train. No production file was changed, no
training was started or stopped, no commit or push was made, no credential was read.
No manuscript text, review text, or case-level data was sent to any external service;
the only external calls were generic methodological literature queries (OpenAlex,
CrossRef, arXiv, MLSys proceedings index). All numeric values below were recomputed
in-session from repository artifacts by script, not asserted from memory.

---

## 0. Verified factual grounding for this memo

These were checked directly in this session and are the basis for everything after.

| Fact | Evidence |
|---|---|
| Nine frozen mean-pool probe runs exist with per-case test predictions: 3 policies × {ep50, ep75, ep100} | `results/downstream/meanpool_sweep_{random,oracle,mirage}/ep{50,75,100}_test_predictions.npz` |
| Each holds `labels` (3000,) and `probs` (3000,), float16 probs | `numpy` load of all nine files |
| Label vectors are byte-identical across all nine files; n = 3000, positives = 1466 | recomputed, all nine `labels_identical_to_first = True` |
| The nine files contain **only** `labels` and `probs` — no case IDs, indices, or manifest | `npz.files` == `['labels','probs']` for all nine |
| Test loaders are constructed with `shuffle=False` | `src/eval_downstream.py:582-584`, `:971-972`, `:1352-1353` |
| Policy score vectors are highly correlated at ep100 (Spearman 0.955–0.969) | recomputed pairwise from the three ep100 files |
| **Future** replication is six legs = 3 policies × 2 seeds, endpoint epoch 50 | `configs/replication/rep_{random,centroid,envelope}_s{1234,5678}.yaml` |
| Those six **replication** configs are generator-produced and differ only in `mask.curriculum`, `meta.seed`, and `logging` | `configs/replication/rep_random_s1234.yaml:3-6` (generator header), `:16-18`, `:60-62` |
| Shared local fork point is a single epoch-25 ancestor with a recorded sha256 | `configs/replication/rep_random_s1234.yaml:8-12` |
| The paused replication reached epoch 26 complete + 8 iterations of epoch 27 | `D:\jepa_phase0\runs\rep_random_s1234\jepa_patch_rep_random_s1234-log.csv`, epochs present `[26, 27]` |
| One epoch = 9375 logged iterations; median per-iteration forward time ≈ 369 ms | recomputed from that CSV (`backward_time_ms` logs as 0.0, so this is a **floor**) |
| No encoder checkpoint lives inside the repo tree (0 `*.pth.tar`); they live on `D:\jepa_phase0` | recursive filter over both roots |
| Dense multi-epoch encoder checkpoints exist for several runs: `cover_random_ep25` (17), `cover_f021_ep25` (20), `patch_mirage_envelope` (17), `anatomy_v2_ep25` (9), `oracle-anatomical-100ep` (3) | checkpoint inventory grouped by run directory on `D:\jepa_phase0` |

**On "paired" — an important qualification.** The nine files have byte-identical label
vectors and highly correlated scores, and the test loaders use `shuffle=False`. Together
these are consistent with **label/order consistency under a documented deterministic
loader**. They do **not** prove that row *i* refers to the same subject across files:
identical label vectors and high score correlation are exactly what one also expects from
any two orderings that agree on class composition, and the saved artifacts contain no
case IDs, indices, or manifest (`npz.files == ['labels','probs']`). **Case-ID provenance
is not established.** Every paired analysis in this memo — including the intervals below
— is therefore *conditional on the ordering assumption*, and confirming it (by saving
case IDs, or by recovering the split manifest and re-deriving the order) is a
prerequisite for reporting any paired interval in the manuscript.

Two derived quantities, stated with their assumptions:

- **Logged-compute floor for replication.** 9375 iterations × 0.3694 s ≈ 3463 s ≈
  **0.962 h per epoch per leg**. Six legs × 25 epochs (ep25→ep50) = 150 epoch-runs ≈
  **144.3 GPU-hours ≈ 6.0 days** if run serially on one GPU. This is a *logged-compute
  floor, not a wall-clock ETA*: it counts logged forward time only, and ignores data
  stalls, validation passes, checkpoint writes, restarts, and queueing. Real wall-clock
  will be higher, by an unmeasured margin. It is consistent with the inherited ~6-day
  figure, which it corroborates in order of magnitude rather than confirming precisely.
- **Time to deadline.** Sept 5 2026 23:59 AoE = **2026-09-06 11:59 UTC**. From this
  memo's start (22:07 UTC Sept 4) that is **≈ 37.9 hours**.

**Consequence that drives the whole strategy:** the *full planned six-leg campaign*
(≈ 144.3 GPU-h floor) and the *proposed four-leg ep25→50 pilot* (2 policies × 2 seeds ×
25 epochs ≈ **96.2 GPU-hours ≈ 4.0 days** floor) both exceed the 37.9-hour horizon on
one GPU, and neither can inform this submission. This is a statement about **those two
specific campaigns at that budget**, not the claim that no training of any kind fits:
shorter continuations, fewer epochs, or sufficient parallel GPUs were not costed here
and are not ruled out. What follows from it is narrow — any pre-deadline improvement
should come from existing artifacts, and the training-based idea below is post-deadline
and authorization-gated.

### Matched-epoch comparisons recomputed from existing artifacts

Frozen mean-pool test AUC, recomputed in-session from the nine prediction files
(cross-checked against `sklearn.metrics.roc_auc_score`; agreement to ≤ 1.1e-16):

| policy | ep50 | ep75 | ep100 | within-policy spread across probe epoch |
|---|---:|---:|---:|---:|
| random | 0.8641 | 0.8723 | 0.8746 | 0.0105 |
| oracle (centroid) | 0.8740 | 0.8836 | 0.8855 | 0.0115 |
| mirage (envelope) | 0.8761 | 0.8803 | 0.8807 | 0.0047 |

Epoch is an **explicitly matched design factor**: every policy is compared to every
other at the same probe epoch. The across-epoch column above is therefore a
*training-progress* effect, not a random draw and not a replication of anything. The
substantive comparisons are the matched-epoch contrasts, which are these:

| matched epoch | contrast | ΔAUC | 95% bootstrap CI (cases resampled jointly) |
|---|---|---:|---|
| ep50 | oracle − random | +0.0099 | [+0.0051, +0.0147] |
| ep50 | mirage − random | +0.0120 | [+0.0068, +0.0171] |
| ep75 | oracle − random | +0.0113 | [+0.0060, +0.0165] |
| ep75 | mirage − random | +0.0080 | [+0.0029, +0.0131] |
| ep100 | oracle − random | +0.0109 | [+0.0057, +0.0159] |
| ep100 | mirage − random | +0.0062 | [+0.0010, +0.0113] |

(2000 bootstrap replicates, seed 0, computed with the project venv.)

What this actually shows, stated without inflation or deflation:

1. **All six matched-epoch contrasts against random are positive, and all six intervals
   exclude zero.** The random-masking continuation scores lowest at every matched epoch.
   These differences between the saved fitted models on this evaluation set are
   **resolved**, not ambiguous.
2. The magnitude of the advantage is **epoch-sensitive**: oracle − random is stable
   (+0.0099 to +0.0113), while mirage − random declines monotonically with epoch
   (+0.0120 → +0.0080 → +0.0062). This is a descriptive property of these runs worth
   reporting, not a defect.
3. The ordering **between the two guided policies** changes (mirage highest at ep50,
   oracle at ep75 and ep100). The ordering of either guided policy **against random**
   does not change. Any ranking claim should therefore be made against random, not
   between the two guided arms.

Three things these numbers do **not** establish, which no re-reading should smuggle back:
they involve one continuation per policy, so they carry no seed-level inference and no
policy-level generalization (§1); the intervals are conditional on the ordering
assumption discussed above, since case identity was never saved; and an interval
excluding zero is not evidence about the *mechanism* (guidance vs. incidental geometry),
which is Idea 3's subject.

---

## 1. Strategic thesis: the strongest narrow contribution

### The claim to make

> A matched-harness comparison of rectangle-family I-JEPA masking policies on retinal
> OCT, in which the guidance source is varied while the mask geometry family and the
> pre-training ancestor are held fixed, showing that **both guided continuations
> outperform the random-masking continuation at every matched probe epoch, with
> uncertainty intervals excluding zero** — reported at the level the design supports
> (these fitted models, this evaluation set), together with an explicit account of why
> the remaining anatomy-shaped and coverage arms cannot support the same comparison.

The contribution is **a positive, carefully-scoped run-level result plus a control
design and a confound taxonomy**. Three properties make this the strongest available
framing:

1. **The central comparison is resolved, not ambiguous.** Six of six matched-epoch
   contrasts against random are positive with intervals excluding zero (§0). Reporting
   this as a null, a non-result, or "not resolvable at this budget" would understate the
   evidence. What the single continuation per policy limits is *generalization to the
   policy level* — an inference about masking policies in general — not the validity of
   the comparison between the saved models that were actually fitted.
2. **The confound taxonomy is a genuine second deliverable.** The project already knows
   *why* the two anatomy-shaped variants and the cover arm cannot be read causally
   (geometry, guide, and collation differences; delivered-target truncation in cover).
   Naming the conditions a masking-policy comparison must satisfy to be causal — same
   geometry family, same guide pipeline, same collation, delivered-target verification —
   is a reusable methodological result independent of which policy scores highest.
3. **Scope discipline is what makes the positive claim durable.** A result stated as
   "these continuations, this evaluation set, one seed each" cannot be overturned by a
   later seed replication; it can only be extended or bounded by one. A policy-level
   superiority claim made now *could* be overturned. The limitations are what protect
   the finding, which is a different thing from the limitations *being* the finding.

The delivered-target truncation in the cover arm should still be **promoted from erratum
to finding**: it is concrete evidence that an intended masking intervention was not the
delivered one, which generalizes past this paper and motivates Idea 3.

### Boundaries of the claims

Defensible with existing evidence:

- **Matched-epoch difference statements about these fitted models on this evaluation
  set**, with uncertainty — e.g. "at every probe epoch examined, both guided
  continuations scored above the random continuation, with 95% intervals excluding
  zero," conditional on the ordering assumption of §0.
- Epoch-sensitivity statements: the oracle advantage was stable across epochs, the
  mirage advantage declined.
- The confound taxonomy for the anatomy-shaped and cover arms.
- Parity statements about the **six generated replication configs**, which are
  generator-produced and differ only in `mask.curriculum`, `meta.seed`, and `logging`.

Not defensible, and must be stated as such:

- **Policy-level generalization.** One continuation per policy. A single run is a draw
  from a distribution over seeds; nothing here estimates that distribution
  (precedent: Bouthillier et al. 2021, §3 ref [2]). This limits the *scope of inference*
  from the observed differences; it does not make the observed differences invalid.
- **Historical launch parity across the paper's arms.** The generator guarantee covers
  the *future* replication configs only. It is **not** evidence that the arms actually
  reported in the paper were launched under matched configurations; that would require
  auditing each arm's own recorded config and launch record, which this pass did not do
  and which the parent should not treat as established.
- **Causal attribution to anatomy guidance.** The remaining arms are confounded, and for
  cover the delivered intervention is documented to differ from the intended one
  (`results/masking/table2_geometry/`), so that arm does not test intended coverage.
  A resolved difference between two
  continuations does not identify *which* property of the policy produced it.
- **Causal coverage effects.** Delivered coverage ≠ intended coverage.
- **Equivalence or "no difference."** No pre-specified equivalence bound, no replication.
  Non-significance is not equivalence, and post-hoc power is never evidence of
  equivalence (precedent: Lakens 2017, §3 ref [4]). Not applicable to the contrasts
  against random, which are positive, but relevant to any future oracle-vs-mirage claim.
- **Domain generality.** One dataset, one label, one architecture, one pooling scheme,
  one retrospectively reused test split.

Three practices to refuse explicitly, since they are the tempting shortcuts here:

- Selecting or re-selecting a winning policy, epoch, or head on the reused test split.
- Converting a non-significant paired comparison into an equivalence or "matches" claim.
- Removing limitations to make the narrative tidier. The limitations *are* the contribution.

---

## 2. Three research ideas for after this audit

Ordered by value per unit of risk. Ideas 1 and 3 need no new training; Idea 2 does and
is explicitly gated.

### Idea 1 — Epoch-sensitivity and provenance audit of the existing matched comparisons

**Framing.** This is a **descriptive audit**, not a hypothesis test. It does not
introduce an acceptance criterion, and nothing in it should later be described as
pre-specified, as a power analysis, or as a noise-floor estimate. Its purpose is to make
the §0 comparisons reportable with correct uncertainty and correct provenance.

**Question 1 (provenance — the blocking one).** Does row *i* of each prediction file
refer to the same subject across all nine files? This is currently *assumed*, not shown:
the artifacts hold only `labels` and `probs`, with no case IDs
(`npz.files == ['labels','probs']`). Resolve it by recovering the split manifest and
re-deriving the deterministic test order (`shuffle=False` at
`src/eval_downstream.py:582-584`), or by re-emitting predictions with case IDs attached.
**This is genuinely falsifiable and consequential:** if row ordering differs, recover
case identifiers and realign predictions, or re-export them with an auditable manifest,
before recomputing paired intervals. If identity cannot be recovered, the current paired
intervals cannot be verified from these files alone. Switching automatically to an
independent-sample analysis is not a remedy: shuffled rows do not make observations on
the same subjects independent. Paired treatment is appropriate once pairing is
established (DeLong et al. 1988, §3 ref [1]).

**Question 2 (epoch sensitivity — descriptive).** Epoch is an explicitly matched design
factor, so the object of interest is the *function* Δ(epoch) for each contrast, not a
comparison between the between-policy contrast and the across-epoch training gain.
Those two quantities measure different things — one is a contrast between arms at fixed
training progress, the other is training progress itself — and comparing their
magnitudes answers no well-posed question. Report Δ(epoch) with its interval at each
available matched epoch and describe its shape. The existing three points already show
two distinguishable shapes (oracle flat, mirage declining); the dense checkpoint
inventory on `D:` would turn three points into a curve.

**Existing artifacts vs new training.** Question 1 needs no training. Question 2 at the
three existing epochs needs no training — it is already done in §0. Extending Question 2
to a denser grid needs *cheap frozen-probe fitting only* (linear head, `head_params =
2305`, `probe_params = 0`, per
`results/downstream/meanpool_sweep_random/ep100_results.json`), no pre-training.
Caveat to confirm first: run-directory names on `D:` do not map transparently onto the
six described arms, and `oracle-anatomical-100ep` holds only 3 checkpoints, so per-policy
grid density is **not established**. A second caveat: probabilities are stored float16
(~1840–1920 unique values per file), fine for AUC and rank statistics but to be checked
before any calibration claim.

**Interpretation.** If provenance confirms the ordering, §0's intervals stand as
reported and the manuscript can state the matched-epoch result with uncertainty. If it
fails, the intervals are replaced, the point estimates survive unchanged, and the
finding is reported more conservatively. If the denser grid shows the mirage decline
continuing, that is a reportable descriptive property of that run; if it shows the
decline was an artifact of three sparse points, that is equally reportable. None of
these outcomes converts into an equivalence claim, and none licenses re-selecting a
preferred epoch or policy on the reused evaluation set — the epochs to report should be
fixed before the denser grid is examined.

**Cost.** Grounded: seconds of CPU for the existing nine vectors (already spent).
Provenance recovery and any denser grid are **unknown** until the checkpoint-to-arm
mapping and split manifest are located.

**Why it differs from the existing arms.** The arms produce point estimates; neither
their pairing entitlement nor the epoch-dependence of the contrast has been
characterized. Both are prerequisites for reporting the arms correctly, and neither
requires a new arm.

---

### Idea 2 — A limited four-leg paired continuation pilot

**What this is.** A *pilot* that produces the project's first observation of seed-to-seed
variation in a continuation from the shared epoch-25 ancestor. It is **not** a noise-floor
estimate, **not** a substitute for the planned six-leg campaign, and **not** a mechanism
for converting the §0 run-level result into a policy-level claim. Those framings were
considered and rejected (see §6).

**Design.** Two policies × two seeds = 4 legs, ep25→ep50, using the existing generated
configs. Two seeds per policy yield a single degree of freedom per policy: enough to see
whether seed effects are visibly small or visibly large relative to the ~0.006–0.012
contrasts in §0, and nothing more. A simple range comparison between a two-seed spread
and a between-policy difference **cannot** establish that seed variation is "reliably
smaller," and no such wording should be attached to the outcome. What the pilot
legitimately produces is a first magnitude estimate to *inform the design and priority*
of the full six-leg campaign — including whether six legs is the right number.

**Two honesty conditions.** First, choosing which two policies to run by looking at the
observed ep100 ordering is **retrospective selection**, and must be recorded as such;
the pilot's contrasts are not independent of the data that motivated them. Second, the
existing six-leg plan should remain the reference design; this pilot does not amend it,
and a decision to change it needs its own justification.

**Existing artifacts vs new training.** Requires new training; configs already exist and
are generator-produced, but launch provenance and the resume-state issues in `code.md`
should be resolved before another campaign. **Post-deadline and
authorization-gated.** Per §0, 4 legs ≈ 96.2 GPU-hours ≈ 4.0 days as a *logged-compute
floor* on one GPU against a 37.9-hour horizon — it cannot inform this submission and
must not be started on that hope.

**Interpretation, including the null.** Report each paired seed-level contrast and its
scope. Large or small observed spreads can inform planning, but two new seeds do not
bound the distribution of training variability. Neither a four-leg pilot nor completion
of the original six-leg plan automatically guarantees a policy-level conclusion.
Neither outcome retroactively invalidates the arithmetic of the existing fitted-model
comparisons; both add evidence about how well those comparisons reproduce.

**Cost.** Logged-compute floor: 0.962 GPU-h per epoch per leg × 25 epochs × 4 legs ≈
96.2 GPU-hours ≈ 4.0 days serially, derived from the paused run's own iteration log.
Real wall-clock is higher by an unmeasured overhead margin; on other hardware or with
parallel GPUs it is unknown.

**Why it differs from the existing arms.** The existing arms vary policy at fixed seed.
This varies seed at fixed policy — the one variance component never sampled, and the
component that governs whether a policy-level claim is reachable at all.

---

### Idea 3 — Incremental extension of the existing delivered-geometry audit

**Correction first: this work substantially exists.** `results/masking/table2_geometry/`
already contains production-sampler geometry over 600 slices (24 volumes, 25 slices per
volume, 16×16 grid, 256 patches) for **five arms** — random, oracle/centroid, envelope,
anatomy, cover — with 37 statistics each, including context/mask/hidden token counts
split by on-anatomy vs background, unique-cell counts, loss-slot counts, slot
duplication, and grid fractions. Coverage spans `bs1` and `bs64`, cover floors 0.21 and
0.15, and three redraw seeds (42, 1234, 2026). `table2_comparison.md` already reports
printed-vs-regenerated verdicts, a delivered-to-encoder bs64 variant, rank-order
stability across all three replicate seeds, and Spearman ρ against AUC@ep50. The main
Table 2 and the delivered-context appendix are built from these. **My earlier claim that
"none of the arms characterizes the intervention itself" was false and is withdrawn**
(§6, R8). Anything proposed here must be incremental to that, and this memo proposes no
re-run of it.

**What the existing audit already establishes.** Post-collation effects at production
batch size are measured, not hypothetical: context-kept drops sharply from bs1 to bs64
in every arm (random 42.1 → 24.72, centroid 45.6 → 32.94, envelope 40.5 → 30.66, cover
43.5 → 30.06, anatomy-v2 67.9 → 63.46). Rank order on anatomy-hidden and purity is
stable across the three redraw seeds. The arm-level association with downstream AUC is
already computed and is not significant (anatomy-hidden vs AUC@ep50, four rectangles,
ρ = +0.80, p = 0.20 — n = 4 arms).

**The incremental gaps.** Three, each checkable against the filenames and `_meta` blocks
before any work is authorized:

1. **Distribution-level, not moment-level.** Everything currently reported is a per-arm
   mean/SD or an arm-level scalar. There is no per-image distributional comparison of
   delivered budgets, so the question "do these arms' per-image budgets overlap?" is
   currently unanswerable from the saved outputs even though the sampler could answer it.
2. **Thin seed coverage at the production setting.** The three-seed redraw structure
   exists at `bs1`/`f=0.21`; at `bs64` the files are `coverf015_seed42`,
   `coverf015_seed1234`, and `coverf021_seed42` — i.e. the production-batch-size,
   f=0.21 condition rests on **a single redraw seed**. Matching the bs1 replicate
   structure there is a small, well-defined extension.
3. **Crop/seed matching across arms is assumed, not documented.** Within a file,
   `anat_cells_mean` is identical across all five arms (67.36), which is consistent with
   shared underlying slices. That is evidence, not proof, that crop draws are identical
   across arms; it should be stated explicitly in the appendix or verified.

**The proposal.** (a) Verify — from the existing JSONs and sampler — that delivered
budgets are compared under *identical crops and seeds at production batch size*, and
report them at **distribution level** rather than as means alone; (b) use the resulting
budget distributions to **specify a genuinely matched future control**: an arm whose
delivered budget distribution is matched to a comparison arm, so that guidance source
varies while delivered budget does not. (b) is a design specification to be written now
and run later under separate authorization, not a run.

**Interpretation — and a hard epistemic bound.** Both outcomes are **diagnostic only**:

- **Substantial overlap of marginal coverage distributions cannot prove that coverage
  fails to explain the downstream effects.** Marginals can overlap while per-image or
  joint structure differs systematically; the downstream effect may depend on *which*
  patches are hidden rather than how many; and arm-level association tests over four or
  five arms are severely underpowered, as the existing non-significant ρ = +0.80 already
  illustrates.
- **Separation cannot establish a causal mediator.** Observing that arms differ on a
  geometry statistic *and* differ on AUC is confounded by everything else that differs
  between arms. Mediation requires manipulating the candidate mediator while holding the
  policy fixed — which is exactly what the matched control in (b) is for.

So neither branch settles the mechanism question. A causal test is still needed, and
this idea's real output is the specification of that test, not a verdict from it.

**Existing artifacts vs new training.** No encoder training and no GPU either way;
sampling is CPU-only. Gaps 1 and 3 are re-analysis and documentation of existing
outputs. Gap 2 is a small additional sampler run at bs64/f=0.21 for two further seeds.
Feasibility caveat: the sampler needs the OCT slices and the MIRAGE guide directory
recorded in `_meta` (`C:\jepa_data\mirage_soft_guides\base512_enc_...`); neither that
path nor the slice cache was verified in this pass.

**Cost.** CPU-only, no model forward pass. Wall-clock **unknown**; the existing runs'
own durations were not recorded in the JSONs and were not measured here.

**Why it differs from the existing audit.** It does not re-measure delivered geometry —
that exists. It adds a distributional view where only moments are saved, closes a
single-seed gap at the production batch size, documents the crop-matching assumption,
and converts the measurements into the design of a budget-matched control arm. The
existing audit describes what was delivered; this specifies the experiment that could
make a delivered-budget claim causal.

---

## 3. Verified public methodological precedent

Four references, each verified this session against a primary bibliographic source.
Metadata below is what the named source actually returned; nothing is reconstructed
from memory.

**[1] DeLong, DeLong & Clarke-Pearson (1988).** "Comparing the Areas under Two or More
Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach."
*Biometrics* 44(3), p. 837. DOI `10.2307/2531595`.
Verified via CrossRef (`doi_to_bibtex.py`) and OpenAlex `W2328176404`; both record
volume 44, issue 3, first page 837. *Note:* both sources record last page = 837; the
commonly cited range 837–845 was **not** confirmed by either source in this pass, so
cite the start page or verify against the journal before printing a range.
**Supports:** Idea 1's requirement that AUC comparisons on the same subjects be treated
as correlated. **General precedent** (diagnostic-test statistics), not OCT-specific.

**[2] Bouthillier et al. (2021).** "Accounting for Variance in Machine Learning
Benchmarks." *Proceedings of Machine Learning and Systems* 3 (MLSys 2021).
Preprint arXiv:2103.03098 (submitted 2021-03-01; arXiv comment field reads
"Submitted to MLSys2021"). Publication verified directly in the MLSys 2021 proceedings
index (HTTP 200), entry
`https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html`.
**Supports:** Idea 2 and the §1 boundary that a single run per condition does not
license a method-level claim; the paper models data-sampling, initialization, and
hyperparameter variance and gives recommendations for performance comparisons.
**General precedent** (ML benchmarking methodology), not OCT-specific.

**[3] Li, Zheng, Liu, Wang, Su & Zheng (2022).** "SemMAE: Semantic-Guided Masking for
Learning Masked Autoencoders." arXiv:2206.10207 (submitted 2022-06-21); arXiv comment
field reads "Accepted by NeurIPS 2022" — i.e. *Advances in Neural Information Processing
Systems* 35. OpenAlex additionally lists a conference-paper record (`W7133213208`).
**Supports:** the framing that guidance-informed masking is an established, non-trivial
design axis in masked image modeling, so a null or inconclusive result on OCT is a
contribution to an open question rather than a failed reproduction.
**General precedent** (natural images, masked autoencoders). It says **nothing** about
OCT, about I-JEPA specifically, or about anatomy-guided masking in medical imaging, and
must not be cited as if it did.

**[4] Lakens (2017).** "Equivalence Tests: A Practical Primer for t-Tests, Correlations,
and Meta-Analyses." *Social Psychological and Personality Science* 8(4), 355–362.
DOI `10.1177/1948550617697177`. Verified via CrossRef; OpenAlex also records the 2016
preprint (`10.31234/osf.io/97gpc`), so cite the journal version.
**Supports:** the §1 refusal to convert non-significance into equivalence, and the
requirement that any future equivalence claim pre-specify bounds and be powered for
them. **General precedent** (statistical methodology), not OCT-specific.

**Explicit gap.** None of these four is OCT-specific, and this pass deliberately did not
search for OCT or retinal-imaging precedent — domain related-work belongs to the science
agent, and the brief restricted queries to generic methodological terms. Do not let any
of [1]–[4] be cited as domain evidence about retinal OCT, anatomy-guided masking in
medical imaging, or I-JEPA on medical data. Those citations, if needed, must come from a
separate domain search.

---

## 4. Provisional work design: model and skill routing

**Framing, stated plainly.** What follows assigns *roles* based on task shape and on
what this project's failure modes have been. It is **not** a measured ranking of model
capability. No head-to-head benchmark was run, none of these assignments is evidence
that one model is generally better than another, and the assignments should be revised
when observed performance contradicts them. The user's stated intent is to *observe*
speed and usefulness rather than assume superiority; this section is written to be
falsifiable by that observation.

| Role | Assigned to | Why this task shape | Failure mode it guards against |
|---|---|---|---|
| Coordinator and numeric verifier | GPT-family, longer-running | Owns the version/status board, reconciles claim→artifact mapping, adjudicates conflicts between agents | Divergent parallel reports; unreconciled numbers between board and manuscript |
| Focused code critic | Fast GPT-family, narrow scope | File-scoped audits with explicit pass/fail criteria; parallelizable and cheap | Broad-scope agents producing prose instead of verdicts |
| Focused scientific critic | Fast GPT-family, narrow scope | Claim-by-claim checks against artifacts | Rhetorical drift; unsupported strengthening of claims |
| Broad synthesis, experimental alternatives | Opus-family (this role) | Confound taxonomy, alternative designs, strategic framing, literature framing — open-ended, judgment-heavy, low-verifiability-per-token | Locally correct audits that never question whether the comparison can answer the question |
| Arithmetic, metrics, figures | **Deterministic scripts, no model** | AUC, CIs, bootstrap, spreads, table regeneration | Model-computed numbers entering the manuscript |

Three rules that matter more than the assignment table:

1. **No model-generated number is final.** Every figure in the manuscript should trace
   to a script that regenerates it. This memo illustrates the rule in **both**
   directions: the AUC and interval tables in §0 were computed by script from the `.npz`
   files and cross-checked against a second implementation, and they survived audit
   unchanged — whereas the compute-floor multiplication (0.962 × 150) was done in prose,
   not by a script, and was wrong by 6 GPU-hours until the verifier caught it (§6, R6).
   The numbers that went through a script were right; the one that did not was not.
2. **Synthesis output must be independently validated** — exactly the parent-validates-
   subagent structure already in use. A broad-synthesis role has the widest error
   surface and the least self-checkable output; it should never be the last reader.
3. **Route by verifiability, not by prestige.** Tasks with crisp pass/fail criteria go to
   fast, cheap, narrow agents. Tasks with no crisp criterion go to a synthesis role *and
   then* to a verifier. Cost follows uncertainty, not importance.

**Skills are workflows, not model choices.** They compose with any of the roles above and
are selected by task type:

- `citation-management` — loaded and used in this pass for §3 (OpenAlex search, CrossRef
  DOI extraction, arXiv metadata). Appropriate whenever references are added or checked.
- `statistical-analysis` — the right home for Idea 1's paired bootstrap and for keeping
  the equivalence-testing boundary honest.
- `scientific-visualization` — for figure honesty (axis ranges, uncertainty display) if
  the §0 stability diagnostic becomes a figure.
- `peer-review` — for structured self-assessment, with the standing caveat from the
  audit's context corrections that a generated review is not a venue outcome.

No skill was installed, created, or modified in this pass, and none is proposed.

---

## 5. Session log

| Item | Value |
|---|---|
| Start (UTC) | 2026-09-04T22:02:28Z |
| First version finished (UTC) | 2026-09-04T22:11:26Z |
| Revision pass 1 (UTC) | 2026-09-04T22:17Z – 2026-09-04T22:23Z |
| Revision pass 2 (UTC) | 2026-09-04T22:25Z – 2026-09-04T22:31Z |
| **Reported active pass** | ≈ 9 minutes for the first version — time this agent spent working |
| **Launch-to-notification** | ≈ 16 minutes for the same pass — wall-clock the parent observed, including scheduling, startup, and return overhead |
| Approximate tool calls | ≈ 38 first pass, ≈ 12 revision 1, ≈ 10 revision 2, ≈ 60 total |
| Files written | 1 — this file (created, then revised in place twice) |
| Production changes | none |
| Training started/stopped | none |
| Commits / pushes | none |
| Network use in revision passes | none |

**On the two timing figures.** They measure different things and neither is a benchmark.
The first figure covers the interval recorded by this agent; it is not a measurement
of model inference time. The launch-to-notification figure is what the parent observed.
The difference has not been decomposed into reasoning before the first timestamp,
tool time, scheduling, or other overhead. Neither figure is a head-to-head comparison:
no other model was given
this task, the passes differ in scope, and the revision passes were spent on corrections
whose cost properly belongs to the first pass's errors. Any use of these numbers to rank
models would be unfounded.

**Repository artifacts inspected.**
`autopilot/reports/fresh_audit_2026-09-04/{scope.md,context_corrections.md}`;
`results/all_results.json`; `results/downstream/` directory inventory and all nine
`meanpool_sweep_*/ep{50,75,100}_test_predictions.npz`;
`results/downstream/meanpool_sweep_random/ep100_results.json` and
`meanpool_sweep_oracle/oracle_ep100.json`; `results/{masking,summary,pretraining}`
inventories; `logs/pretraining/*_epoch_summary.csv`; `configs/replication/` (six configs;
`rep_random_s1234.yaml` read in full); on `D:\jepa_phase0` — run-directory listing,
checkpoint inventory by directory, `checkpoint-ep25/` contents, and
`runs/rep_random_s1234/` including its 1.5 MB iteration log. Revision pass 2 added
`results/masking/table2_geometry/` — all six `mask_geometry_600slices_*.json` filenames
and `_meta` blocks, one JSON read in full, and `table2_comparison.md` read in full.

**External sources queried** (generic methodological terms only; no manuscript, review,
or case data transmitted). OpenAlex API — 4 searches (benchmark variance; correlated ROC
comparison; semantic-guided masking; equivalence tests). CrossRef via
`doi_to_bibtex.py` — 3 DOIs. arXiv API — 2 IDs (2103.03098, 2206.10207), used for the
comment/journal-ref fields. `proceedings.mlsys.org/paper_files/paper/2021` — 1 fetch, to
confirm [2]'s venue.

**Bounded coverage and limits.**
- Two `arxiv.org/abs` HTML fetches returned title-only content; the arXiv API was used
  instead. Recorded rather than retried.
- **Interpreter correction.** The AUC and rank statistics in the first version of this
  memo were computed with the *user-level* interpreter
  `C:\Users\Gary\AppData\Local\Programs\Python\Python311\python.exe`, which has no
  `scipy`; that is an interpreter-specific fact, **not** a global one, and the earlier
  phrasing "scipy is unavailable in the environment" was wrong. The project interpreter
  `D:\jepa_phase0\.venv\Scripts\python.exe` provides scipy 1.17.1 and numpy 2.4.4. All
  §0 values have since been recomputed with the project interpreter and cross-checked
  against `sklearn.metrics.roc_auc_score`, agreeing to ≤ 1.1e-16; the earlier
  "not cross-checked" limitation is closed.
- **Case-ID provenance is not established** (§0). The paired intervals in §0 are
  conditional on the ordering assumption. This is the most important open limitation in
  this memo and is Idea 1's first deliverable.
- **Historical launch parity across the paper's arms was not audited.** The generator
  guarantee covers the six future replication configs only.
- DeLong's page range could not be confirmed beyond the start page from CrossRef or
  OpenAlex (§3 [1]).
- Run-directory names on `D:` were **not** mapped to the six described arms; this bounds
  how dense a probe-epoch grid Idea 1 can use.
- Data and slice-cache directories referenced by the configs were not verified, nor was
  the MIRAGE guide directory recorded in the geometry `_meta` blocks, so Idea 3's
  sampler-side feasibility is likely but unconfirmed.
- The existing delivered-geometry audit was **read, not re-run**: findings quoted from
  `results/masking/table2_geometry/` are as recorded in those files, and no sampler was
  executed in any pass.
- No manuscript and no prior review was read — those belong to other agents in this
  audit, per the brief. The only `src/` inspection was a targeted grep for test-loader
  ordering (`src/eval_downstream.py`, `src/train_patch.py`), used solely to check the
  determinism claim in §0.
- Compute figures are **logged-compute floors ignoring overhead**, not wall-clock ETAs.

**Where this pass was useful.**
1. Established that all six matched-epoch contrasts against the random-masking
   continuation are positive with bootstrap intervals excluding zero — the concrete
   basis for a *positive*, scope-limited thesis, and a correction to this memo's own
   first-version framing.
2. Characterized the epoch-sensitivity of those contrasts (oracle stable, mirage
   declining), and showed the ordering change occurs only *between* the two guided
   policies, never against random.
3. Showed that case-ID provenance is not established, making every paired interval
   conditional and giving Idea 1 a genuinely blocking first deliverable.
4. Derived a logged-compute floor from the paused run's own iteration log
   (0.962 h/epoch/leg → 144.3 GPU-h for six legs, 96.2 for four), converting an
   inherited assertion into a bounded, arithmetic-checked figure.
5. Found dense multi-epoch checkpoints on `D:` (17–20 per run for several runs), which
   would extend Idea 1's three epoch points into a curve — subject to the name-mapping
   caveat above.

---

## 6. Cross-model adjudication record

This memo was produced by the broad-synthesis role (§4) and then audited by the
coordinator/numeric-verifier role before entering the version board. The following
inferences were **rejected** and have been corrected in place. This record exists so
that a reader of the board can see what was discarded and why, and so that the rejected
claims cannot re-enter by being quoted from an earlier draft.

### Rejected inferences

| # | Rejected claim (first version) | Why rejected | Disposition |
|---|---|---|---|
| R1 | The test split is "genuinely **paired**"; identical label vectors plus high score correlation establish the pairing | Identical labels and correlated scores do not prove subject identity at row *i*; they are consistent with any orderings agreeing on class composition. The artifacts save no case IDs | Restated as label/order consistency under a documented deterministic loader; **case-ID provenance not established**; all paired intervals marked conditional (§0) |
| R2 | Idea 1's criterion: a policy difference counts only if it exceeds the within-policy spread across probe epochs | Epoch is an **explicitly matched design factor**, not a random nuisance draw or a replication. Across-epoch change is training progress; comparing it to a between-arm contrast answers no well-posed question, is not a resolution or noise-floor estimate, and its failure cannot falsify a detectable matched-epoch difference. It was also a post-hoc criterion presented as pre-specified | Idea 1 rewritten as a **descriptive epoch-sensitivity and provenance audit**; the criterion is withdrawn entirely, not weakened |
| R3 | Strategic thesis as "a candid non-result"; the difference is "not resolvable at this budget" | The matched frozen-model differences **are** resolved: 6/6 matched-epoch contrasts against random are positive with intervals excluding zero. Absent seed inference limits *generalization to the policy level*, not the validity of the saved-run comparisons | Thesis rewritten as a **positive, scope-limited run-level result** plus control design and confound taxonomy (§1) |
| R4 | Generator parity (byte-identical outside `mask.curriculum`) certifies configuration parity for the paper's arms | The generator guarantee covers the **six future replication configs** only, and even there `meta.seed` and `logging` differ. It is not evidence about how the historical paper arms were launched | Scoped to the replication configs; "historical launch parity not established" added to the non-defensible list (§1) |
| R5 | Idea 2 establishes a "seed-variance floor"; if seed spread is "reliably smaller" than the policy gap, a policy-level claim becomes defensible; it replaces the six-leg plan | Two seeds per policy give one degree of freedom per policy; a range comparison cannot support "reliably smaller." The two policies were chosen from the observed ep100 ordering, which is retrospective selection. A pilot does not confer policy-level claim authority, and does not amend the reference design | Reframed as a **limited four-leg paired continuation pilot** with both honesty conditions recorded; six-leg plan retained as reference (§2) |
| R6 | 0.96 h × 150 ≈ 150 GPU-h ≈ 6.2 days; 0.96 h × 100 ≈ 100 GPU-h ≈ 4.2 days; "no replication variant fits" | Arithmetic error: 144.3 h and 96.2 h. The blanket claim also over-generalized from two specific campaigns | Corrected to 144.3 h (≈ 6.0 d) and 96.2 h (≈ 4.0 d); bounded as **logged-compute floors ignoring overhead**, not wall-clock ETAs; the infeasibility claim scoped to the six-leg and four-leg ep25→50 campaigns on one GPU (§0) |
| R7 | "`scipy` is unavailable in the environment" | Interpreter-specific, not global. The project interpreter `D:\jepa_phase0\.venv\Scripts\python.exe` has scipy 1.17.1 | Corrected to name the interpreter actually used; §0 values recomputed with the project interpreter and cross-checked against sklearn (≤ 1.1e-16), closing the earlier "not cross-checked" caveat (§5) |
| R8 | Idea 3's premise: "All six arms characterize the *outcome* of an intervention. **None characterizes the intervention itself.**" | Factually false. `results/masking/table2_geometry/` holds 600-slice production-sampler geometry for five arms with 37 statistics each, at `bs1` and `bs64`, cover floors 0.21 and 0.15, and three redraw seeds; `table2_comparison.md` already reports the bs64 delivered variant, rank stability across seeds, and Spearman ρ against AUC@ep50. Main Table 2 and the delivered-context appendix are built from it. The idea as drafted would have re-run existing work | Idea 3 rewritten as an **incremental** extension: distribution-level rather than moment-level reporting, the single-seed gap at `bs64`/f=0.21, documenting the crop-matching assumption, and specifying a budget-matched future control. Explicit bound added: marginal overlap cannot prove coverage does not explain downstream effects, and separation cannot establish a causal mediator — both outcomes diagnostic, causal test still required |

### Accepted and retained

- The **strategic contribution structure** of §1 — matched-harness result, confound
  taxonomy, scope discipline — retained with the thesis re-polarized from negative to
  positive.
- **Promoting the cover delivered-target truncation from erratum to finding.**
- **Idea 1** in rewritten descriptive form; its provenance question is now the highest-value
  next action, and is genuinely blocking for any reported interval.
- **Idea 2** as a bounded, post-deadline, authorization-gated pilot.
- **Idea 3** substantially rewritten (§6, R8): the delivered-geometry audit already
  exists in `results/masking/table2_geometry/`, so the idea is now scoped to three
  incremental gaps plus the specification of a budget-matched control, with explicit
  limits on what overlap or separation could show.
- The **four verified references** (§3) and the general-vs-OCT-specific separation.
- The **compute-floor derivation method** (per-iteration logged time × iterations/epoch),
  with corrected arithmetic and floor framing.
- The **model/skill routing** in §4.

### Note on what this record demonstrates

The rejected items fall into two distinct failure modes, both worth naming. R2 and R3
are the mode §4 predicts for a broad-synthesis role: confident inferential framing built
on top of correctly-computed numbers, where the error is in *what the numbers license*
rather than in the numbers. Both produced coherent, well-written arguments for
conclusions the data did not support, and neither was self-detectable from within the
synthesis pass. R8 is a different and more basic failure: a confident factual claim
about what the repository does **not** contain, asserted without searching for it, when
a directory of exactly that work existed. R1, R4 and R7 are variants of the same
over-reach — treating consistent-with as established-by. R6 was ordinary arithmetic done
in prose instead of by a script.

The practical lesson is narrower than "verify the synthesis role." It is that
*negative* claims about a codebase ("nothing does X," "no artifact exists for Y") are
the cheapest to state, the most load-bearing when they motivate new work, and the least
likely to be checked — and that they should be treated as search obligations, not
assertions. This is evidence about **this workflow's control structure**; it is **not**
a measurement of general model capability and should not be cited as one. A single
adjudication of one memo, with no controlled comparison, supports nothing broader.
