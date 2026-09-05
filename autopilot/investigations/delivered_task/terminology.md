# Terminology and writing guide (Stage C)

Owner: `literature-contract`. Baseline `de145d7`. Compiled 2026-09-04.
Purpose: give the manuscript one vocabulary that a JEPA-literate reviewer will
recognise, that does not overclaim, and that never renames our system into
something we did not build. Definitions follow community usage in
arXiv:2301.08243v3, arXiv:2111.06377v3, arXiv:2511.17354v4 and arXiv:2206.10207v3;
no prose is copied from them.

---

## 1. Standard JEPA vocabulary (use these words, with these meanings)

| Term | Meaning as used in this literature | Note for our paper |
|---|---|---|
| **Joint-embedding predictive architecture (JEPA)** | predicts the *representation* of a target signal from a compatible context signal, with the loss applied in embedding space | say "latent-space prediction", never "reconstruction" |
| **Context encoder** (online/student) | the gradient-updated encoder that sees only the visible context tokens | ours: ViT-B/16 on 2-D B-scans |
| **Target encoder** (teacher) | an EMA copy of the context encoder that encodes the **whole image**; targets are taken by masking its **output** | full-image teacher access is intended I-JEPA design (Table 11), not leakage — say so once, explicitly |
| **Predictor** | a narrow ViT conditioned on positional/mask tokens that maps context representations to predicted target representations | ours is *parallel*: all targets predicted independently in one pass |
| **Context block / context mask** | the set of visible patches given to the context encoder, after removing overlap with targets | keep three sets distinct and named (see §3): **candidate context block**, **target complement**, **delivered context** |
| **Target block** | a contiguous region whose teacher representation is the prediction target | |
| **Multi-block masking** | I-JEPA's default: 4 target blocks at scale (0.15, 0.2), aspect (0.75, 1.5), one context block at scale (0.85, 1.0) | this is our RANDOM baseline's policy |
| **Masking ratio** | fraction of patches removed from the encoder input (MAE vocabulary) | not the same as our "fraction of tissue hidden"; keep the two apart |
| **Latent prediction loss** | Smooth-L1/Huber or L2 between predicted and teacher token embeddings after feature-dim LayerNorm | ours: `F.smooth_l1_loss` — the same function the official I-JEPA code runs |
| **EMA / momentum encoder** | teacher parameter update `θ̄ ← m θ̄ + (1−m) θ` with m scheduled 0.996→1.0 | |
| **Flat (parallel) prediction** | all targets predicted independently; permutation-symmetric | **this is us** |
| **Sequential / autoregressive region prediction** | target k+1 is predicted conditioned on the representations of targets 1..k, with masked (causal) attention | **this is not us** — never describe our predictor this way |
| **Saliency / attention-derived importance** | an importance map computed *from a network's own attention or gradients*, recomputed during training | **not** what our guide is |
| **Anatomical prior / segmentation-derived guide** | a spatial prior computed outside the pretraining objective, which the pretraining run never updates | **this is us** — but "not updated by training" is not the same as "fixed": CENTROID is recomputed from each transformed image, and the cached MIRAGE envelope is transformed jointly with its image, so the delivered guide varies per view |
| **Frozen-feature linear/probe evaluation** | downstream head trained on frozen encoder features | our **primary** protocol: frozen encoder + mean-pool probe, patient-level AUC |
| **Fine-tuning evaluation** | whole backbone updated downstream | we **do** report fine-tuning diagnostics — appendix "Fine-tuning narrows the observed gap" (`tab:finetuned`) and the fine-tuned-probe attribution appendix. Never write "we do not report fine-tuning"; write "our primary protocol is the frozen probe; fine-tuning appears as an appendix diagnostic" |

---

## 2. Canonical arm names, aliases and one-line definitions

Use the **canonical** name everywhere in the manuscript, tables, figures and
logs. Aliases are listed only so that readers of the repository, logs and
earlier drafts can map them; do not introduce new synonyms.

| Canonical | One-line definition | Aliases seen in repo/logs/drafts | Never call it |
|---|---|---|---|
| **RANDOM** | stock I-JEPA multi-block sampler: target rectangles at uniformly random locations, context block minus overlap | `multiblock`, R1, baseline, "stock", "unguided" | "uniform masking" (ambiguous with MAE's i.i.d. patch masking) |
| **CENTROID** | segmentation-free policy: the retinal band is located **from each transformed training image** by a per-column intensity-weighted row centroid (smoothed); rectangles are biased onto that band | `oracle` (macro/`\ArmBest` in the .tex, run dirs), "intensity centroid" | "oracle" in prose — it uses no labels and no ground truth; "precomputed" — it is computed per view during training |
| **ENVELOPE** | rectangles biased onto the MIRAGE-derived **repaired retinal envelope** (union of RNFL+GCIPL+choroid with the unlabelled mid-retina gap closed), cached in native label space and carried through the same geometric transform as its image | `mirage_envelope`, `patch_mirage_envelope`, "MIRAGE union" | "the MIRAGE union" (the repair makes it strictly larger — the module says so); "segmentation ground truth"; "fixed guide" without the transform caveat |
| **ANATOMY** | class-aware irregular target blobs grown inside the envelope per class, partitioned by geometry (farthest-point seeding + multi-source BFS). **Two implementations: v1 (`patch_mirage_anatomy`) and v2 (`patch_anatomy_v2`, run `anatomy_v2_ep25`)** — never merge them into one arm | `mirage_anatomy`, `anatomy_v2`, "blobs" | "semantic parts" (that is SemMAE's learned object-part construct); "the ANATOMY arm" without a version |
| **COVER** | envelope-shaped rectangles greedily **placed** to hide as much guide-positive tissue as possible; shape/count/size identical to ENVELOPE | `mirage_cover`, "greedy coverage" | an "over-coverage condition" — `src/masks/cover.py` and `autopilot/COVER_AUDIT.md` record that delivered COVER hides *less* tissue (73.1 %) than ENVELOPE (77.6 %) after truncation |

**Counting rule:** say **five policy families** (RANDOM, CENTROID, ENVELOPE,
ANATOMY, COVER) and **six implementations**, because ANATOMY v1 and v2 are
distinct implementations with distinct runs. A COVER `delivered_v2`
configuration additionally exists under the mask owner's active repair; if it
ever produces results, it must carry its own version label and must never be
merged with the historical COVER arm. Legacy and corrected outputs stay
separately labelled everywhere — tables, figures, filenames and prose.

Related internal vocabulary that should either be defined once or kept out of
the manuscript: `loss_guided` (R2, per-position predictor-loss EMA bias),
`intensity_foreground` (R3a), `cluster_foreground` (R3b), `r_t` / `r_max`
(guided-draw ramp), `pred_target_k`, "global-min truncation".

**Importance is a proxy.** Standing phrasing, to be reused verbatim rather than
re-invented per section:

> Guide-positive tissue is a **proxy for importance**, not a measurement of
> diagnostic information. It marks where retinal tissue is, not where the
> evidence for glaucoma lies.

Corollaries that must never be asserted: that guide-positive patches carry more
diagnostic signal; that hiding them makes the task "harder" in an informative
sense; that a higher hidden-tissue fraction is a better intervention.

---

## 3. What our model is — and the four things it is not

**Is:**
* a **2-D B-scan encoder**, ViT-B/16, trained with I-JEPA-style latent
  prediction on individual OCT B-scans;
* volume-level prediction obtained **downstream** by pooling per-slice features
  (mean pooling over the sampled B-scan stack — 100 slices in the delivered
  protocol) under a frozen encoder;
* trained with `F.smooth_l1_loss` between predictor outputs and
  LayerNorm-normalised EMA-teacher token embeddings;
* a **flat/parallel** predictor over independently sampled target blocks;
* guided by a spatial prior that **the pretraining objective never updates**,
  but which is **not** a frozen bitmap: CENTROID is recomputed from each
  transformed image, and the cached MIRAGE envelope is carried through the same
  geometric transform as its image, so the delivered guide differs per view.

**Is not:**

| Do not write | Why | Write instead |
|---|---|---|
| "3-D encoder", "volumetric model" | the encoder never attends across slices; pooling happens after freezing | "2-D B-scan encoder with volume-level mean pooling in the downstream probe" |
| "autoregressive", "sequential region prediction", "curriculum over prediction order" | our predictor has no ordering and no causal attention; that is DSeq-JEPA's contribution | "parallel prediction of independently sampled target blocks" |
| "attention-guided", "saliency-guided", "learned importance" | our guide is a segmentation head's output or an intensity heuristic; the pretraining objective never updates it, and it is not the model's attention | "anatomy-guided" / "segmentation-derived prior" / "intensity-derived band" (add "computed per transformed view" where the distinction matters) |
| "semantic masking", "part-based masking" | SemMAE's parts are learned and object-centric | "anatomy-guided target placement" |

One more distinction the plan asks us to keep visible: there are **three
distinct sets**, and a number for one must never appear under the label of
another —

1. **candidate context block**: the context rectangle the sampler draws
   (I-JEPA's scale-(0.85, 1.0) block), before any removal;
2. **target complement**: all patches not covered by target masks — a
   *different* set, and larger than the candidate block;
3. **delivered context**: what the encoder actually receives after the candidate
   block has had overlapping targets removed and after batch-min truncation.

Use "candidate context", "target complement" and "delivered context" by name.
"Nominal context" is ambiguous between (1) and (2); if it is used at all, define
it as (1) at first use.

---

## 4. The author's hypothesis, stated as testable

Keep it, but keep it as a proposal:

> **Working hypothesis.** Pretraining is more useful for downstream OCT
> classification when the predictive task (i) places targets on informative
> retinal tissue, (ii) leaves the encoder a sufficiently informative visible
> context, and (iii) retains some background/unguided target diversity.

Rules for using it:
* introduce it as *the hypothesis this work was designed to probe*, never as an
  established principle or a "requirement";
* the three clauses are **separable** and were **not separately tested here** —
  say so;
* the only clause with external support is (ii), and that support is bounded:
  I-JEPA's context-scale ablation (Table 9) varies its own sampler's context
  block on ImageNet-1 % low-shot probing. Cite it as a reason to *measure* our
  delivered context, never as a predicted effect size for OCT;
* clause (i) has mixed external evidence under constructions that are not ours
  (DSeq-JEPA Table 4: guided+flat 72.0 vs uniform+flat 72.4, single runs, seed
  spread ±0.4, different saliency/geometry/domain; SemMAE Table 3: whole-part
  52.9 vs random 66.8; AnatMAE Table 3: guided masking alone positive). Present
  the tension; draw no significance conclusion and claim no equivalence between
  their interventions and ours.

---

## 5. Proposed intro / method outline (structure only, not draft prose)

Four movements, kept strictly separate. Target ≈ intro 4 paragraphs, method 4
subsections; this is an outline for the writer, not text to paste.

**Introduction**
1. *Setting.* Frozen-feature self-supervised pretraining for OCT-based glaucoma
   classification; why latent prediction (I-JEPA family) rather than pixel
   reconstruction (MAE family) — one sentence each, with the objective
   difference stated plainly.
2. *Motivation (hypothesis, clearly labelled as such).* Retinal B-scans are
   mostly background; masking policies that ignore anatomy may spend supervision
   on empty space. State the three-clause working hypothesis from §4 as the
   question, and name the proxy nature of the guide in the same breath.
3. *What we did.* Five masking policy families (RANDOM, CENTROID, ENVELOPE,
   ANATOMY, COVER) across six implementations (ANATOMY v1/v2) under a shared
   I-JEPA-style pipeline, evaluated primarily with a frozen encoder and
   patient-level AUC (with fine-tuning diagnostics in the appendices); plus an
   audit of what task was actually delivered to the model.
4. *Contributions, in the order the evidence supports them:* (a) run-level
   comparisons in which CENTROID and ENVELOPE exceed RANDOM under matched
   frozen-probe evaluation, with their stated limitations; (b) a measured
   delivered-task audit showing where intended and realised masking policies
   diverge (e.g. the documented COVER coverage inversion); (c) a scoped
   statement of what remains untested — above all, the mechanism.

**Method**
1. *Pretraining objective.* I-JEPA-style latent prediction: full-image EMA
   teacher, LayerNorm over the feature dimension, Smooth-L1 on predictor
   outputs, parallel prediction of independently sampled target blocks. Note
   that Smooth-L1 matches the official I-JEPA implementation.
2. *Guides.* MIRAGE-derived repaired retinal envelope (and why "repaired"),
   cached in label space but transformed with its image; and the
   segmentation-free intensity centroid computed per transformed view. Explicit
   proxy caveat. Fallback behaviour when the guide is invalid.
3. *Masking policies.* One paragraph per canonical family (with ANATOMY v1/v2
   distinguished), each stating: what is held fixed relative to RANDOM (shape,
   count, size, ratio) and what changes. State the guided/unguided mixture rule
   and the ramp in the units actually used (a per-predictor-block Bernoulli over
   *placement*), and note where the delivered mixture is measured rather than
   assumed.
4. *Delivered-task measurement and downstream protocol.* Candidate context vs
   target complement vs delivered context, kept as three named sets; how the
   audited quantities are counted, with denominators; frozen encoder, mean-pool
   probe over the slice stack, patient-level AUC as primary, appendix
   fine-tuning diagnostics as secondary, and the uncertainty treatment already
   in use.

**Two things this outline deliberately keeps apart**
* *Empirical, run-level:* the AUC comparisons, their CIs and their limitations.
* *Untested mechanism:* why CENTROID/ENVELOPE might help. Any mechanism
  sentence must be marked as a hypothesis and must not be supported by citing
  DSeq-JEPA or MAE, for the reasons in `literature_matrix.md` §4.

**Related-work framing that survives review** (one paragraph): guided masking
has a mixed published record — SemMAE's whole-part masking alone is far worse
than random, DSeq-JEPA reports a gain only in the cell where guided selection
and sequential prediction are changed together, and anatomy-guided masking helps
in a fine-tuned 3-D detection pipeline where crop selection mattered more than
mask selection. Each of those constructions differs from ours in guide source,
mask geometry, objective and evaluation, so none of them predicts our outcome.
Our study sits in that unresolved space and reports a bounded result, not a
resolution.

---

## 6. Phrases to avoid, with replacements

| Avoid | Replacement |
|---|---|
| "important-region masking works / is known to work" | "guided masking has a mixed published record (see related work)" |
| "our attention-based mask" | "our segmentation-derived / intensity-derived guide" |
| "the model learns to focus on the retina" | (drop, unless a measurement supports it) |
| "3-D JEPA" | "2-D B-scan JEPA with volume-level pooling downstream" |
| "oracle policy" | "CENTROID (intensity-centroid) policy" |
| "COVER hides more anatomy" | "COVER was intended to increase hidden tissue; delivered COVER hides less than ENVELOPE (measured)" |
| "lower prediction loss shows better learning" | (drop — moving-teacher loss is not comparable across arms) |
| "equivalent", "identical task" | "matched on X, differing in Y" with both named |
| "our exact configuration is DSeq-JEPA's losing cell" | "an analogous factor combination (guided placement with flat prediction) reads 72.0 vs 72.4 in their Table 4; their construction differs from ours in guide source, mask geometry, conditioning, dataset and evaluation, and the cells are single runs" |
| "their gain requires sequential prediction" | "they attribute their gain to the combination of selection and sequencing, in their setting" |
| "our ramp endpoint is SemMAE's α=1" | (drop — different intervention on a different object) |
| "the five arms" / "the ANATOMY arm" | "five policy families, six implementations"; name the ANATOMY version |
| "we do not report fine-tuning" | "our primary protocol is the frozen probe; fine-tuning appears as an appendix diagnostic" |
| "nominal context" (undefined) | "candidate context block" / "target complement" / "delivered context" |
| "DSeq-JEPA shows the model the most discriminative tissue" | "DSeq-JEPA keeps its highest-saliency-ranked region in the context" |
