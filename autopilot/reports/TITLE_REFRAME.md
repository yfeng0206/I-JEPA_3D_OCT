# Title change and n=1 reframing

Target file: `paper/genai4health2026/main_submission.tex`
Date: 2026-08-26. Gates re-run after every edit; final state at the bottom.

---

## 1. THE TITLE

### Chosen

> **Where to Aim, Not How Much to Cover: Segmentation-Free Anatomy Guidance for
> Masked Predictive Pretraining on Retinal OCT**

LaTeX (three source lines, renders as exactly three typeset lines, so the title
block is no taller than the one it replaces):

```latex
\title{Where to Aim, Not How Much to Cover:\\
Segmentation-Free Anatomy Guidance for\\
Masked Predictive Pretraining on Retinal OCT}
```

Replaced: *"Does Anatomy-Guided Masking Help? A Controlled Comparison of Six
Masking Policies for Joint-Embedding Predictive Pretraining on Retinal OCT"*.

### The exact sentences of the paper that support it

Clause 1 -- **"Where to Aim, Not How Much to Cover"** is a near-verbatim lift of
the paper's own finding sentence, `main_submission.tex:482` (Section 5.2,
*Aim, not coverage*):

> "First, **anatomy tells you where to aim, not how much to cover**. Of the four
> guided arms the best performer hides the *least* anatomy, while the arm that
> hides the most --- masking almost nothing but tissue, at 97.1% purity --- does
> not separate from the null (Table 2)."

Corroborated in the abstract (`:63`) "anatomy says where to aim, not how much to
cover", and in the Conclusion "anatomy mattered for *where* targets are aimed and
not for *how much* anatomy they cover".

Clause 2 -- **"Segmentation-Free Anatomy Guidance"** is supported by
`main_submission.tex:443`:

> "**The strongest policy consults no segmentation model.** CENTROID finds its
> band from a per-column intensity centroid: no model and no annotation."

and by the Discussion: "For *this* task the segmentation stage bought no more
than location, and a one-line intensity statistic recovered the whole benefit."

### Sentence-by-sentence safety check against the blocking failure mode

The previous "Location Beats Shape" title asserted a contrast that Section 5.2
declares unidentified. The new title was checked against the same three facts:

| Fact that must not be contradicted | New title |
| --- | --- |
| Table 2: CENTROID hides 62.1% of anatomy, the **least** of any guided arm | Title says coverage is not the operative axis ("Not How Much to Cover"). Consistent. |
| Table 2: ANATOMY-V2 hides the most (79.9% at 97.1% purity) and loses to the null | Title makes no "more anatomy is better" claim; it denies exactly that reading. Consistent. |
| Section 5.2 `:500`: "**H3 is therefore not identified by this design**, and we do *not* claim that irregular target shape is harmful" | Title contains **no** word about shape, raggedness, blobs or geometry. Consistent. |
| CENTROID uses no segmentation model (`:443`) | Stated as "Segmentation-Free ... Guidance", a noun phrase, not a comparative. Consistent. |
| CENTROID vs ENVELOPE is significant only at epoch 100 ("indistinguishable at epochs 50 and 75", `:449`) | Title asserts no beat over the segmenter, only that guidance can be had without one. Consistent. |

Nothing elsewhere contradicts it: the abstract's first sentence is a generic
definition of masked predictive pretraining, the abstract already carries both
clauses verbatim, the Introduction says "Anatomy turns out to matter for *where*
targets are aimed rather than for how much anatomy they cover ... The best
performer locates a band by a per-column intensity centroid and consults no
segmentation model", and the Conclusion repeats both. No sentence in the paper
frames the work as answering a yes/no question, so dropping the interrogative
form leaves no dangling reference.

### Rejected candidates and why

1. **"A Segmentation-Free Band Beats a Segmenter: Rethinking Anatomy-Guided
   Masking in OCT"** -- REJECTED as false at two of three epochs. `:449` states
   CENTROID exceeds ENVELOPE at epoch 100 "though the two are indistinguishable
   at epochs 50 and 75", and in Table 2 the matched-epoch-50 AUC that is
   **bolded** is ENVELOPE's, not CENTROID's. "Beats a Segmenter" is exactly the
   kind of unsupported comparative that sank "Location Beats Shape".
2. **"Where to Mask, Not How Much"** -- REJECTED on ambiguity. "How much" without
   "to cover" reads most naturally as *mask ratio*, and the paper explicitly
   cannot speak to that: "Whether what the mask *leaves visible* matters is *not*
   separable here: the arms that differ in retained context also differ in mask
   ratio and loss slots." Restoring "to Cover" pins the quantity to anatomy
   coverage, which is what Table 2 measures.
3. **"Aim, Not Coverage: What Anatomy Guidance Actually Buys in Medical JEPA
   Pretraining"** -- REJECTED as a near-miss. Accurate, but it duplicates the
   Section 5.2 heading verbatim, "Actually Buys" is a rhetorical hedge of the
   same family as the question form being removed, and it loses both "OCT" and
   the segmentation-free hook, which is the genuinely quotable result.
4. Any variant naming shape, blobs, ragged targets or "irregular masks" --
   REJECTED outright by `:500` (H3 not identified).
5. Keeping "A Controlled Comparison of Six Masking Policies" in the subtitle --
   REJECTED for length only; the control story survives in the abstract
   ("under tightly controlled conditions"), the Introduction ladder paragraph,
   and the newly reframed limitation paragraph.

Typesetting note: two intermediate line-break arrangements of the same words
wrapped to four typeset lines and one of them pushed the body to 10 pages. The
committed arrangement was verified in the rendered PDF to occupy three lines,
matching the height of the title it replaced.

---

## 2. THE n=1 LIMITATION, REFRAMED

Nothing was deleted. The `n=1` concession, the probe-seed concession, the
Bouthillier citation and the entire running-replication passage with its
**PENDING** label are all still there; only the framing of the first four
sentences changed, from an apology to a scoping statement. The paper still makes
no reference of any kind to what other submissions do.

### BEFORE (`\paragraph{One pretraining run per policy, and a replication in progress.}`)

> The paired bootstrap quantifies sampling error on a fixed test set; it does not
> quantify seed-to-seed variance of pretraining. With $n{=}1$ continuation per
> arm the *ranking* is not statistically established even where individual
> pairwise differences are significant on this test set.
> \citet{bouthillier2021variance} show that single-seed comparisons routinely
> mistake optimisation noise for method effects. Probe-seed variance is not
> quantified here either: an earlier multi-seed probe check is not reproducible
> from retained artifacts, so this paper states no bound on probe noise. A
> replication addressing the first gap is running and every result of it is
> PENDING: [...unchanged...]

### AFTER

> This design holds the ancestor checkpoint, optimiser, schedule, effective batch
> size and probe protocol fixed across arms, leaving masking policy the only
> moving variable; what it does not resample is pretraining stochasticity, so
> each arm is one continuation. The paired bootstrap bounds sampling error on a
> fixed test set, so the intervals support statements about *these* continuations
> rather than an expected *ranking* over retrainings, which pairwise significance
> on one test set does not establish~\citep{bouthillier2021variance}. Probe-seed
> variance is outside the design too: an earlier multi-seed probe check is not
> reproducible from retained artifacts, so we state no bound on probe noise. A
> replication addressing the first gap is running and every result of it is
> PENDING: [...unchanged, verbatim...]

### What changed, precisely

* Leads with what the design **does** control -- ancestor checkpoint, optimiser,
  schedule, effective batch size, probe protocol -- so the reader meets the
  controls before the caveat, and the caveat lands as the boundary of a
  deliberate design rather than as a missing experiment.
* States the unit of inference positively and exactly: the intervals **support**
  statements about these continuations; they **do not support** an expected
  ranking over retrainings.
* "With $n{=}1$ continuation per arm the ranking is not statistically
  established" (an admission) becomes "which pairwise significance on one test
  set does not establish" (the same content as a statement of scope). The
  Bouthillier citation is retained, moved from a sentence *about our weakness*
  to a parenthetical support for the scoping claim.
* "Probe-seed variance is **not quantified here either**" -> "is **outside the
  design too**". Same missing quantity, same explicit statement that no bound is
  offered, no apologetic "either".
* "this paper states no bound" -> "we state no bound": active, not distancing.
* Wording chosen to be near length-neutral, because page 9 was already full to
  the last line. An intermediate draft that said "the pretraining seed is not
  fixed" was replaced with "what it does not resample is pretraining
  stochasticity", since the former could be misread as claiming the six arms used
  *different* seeds, which the replication appendix contradicts ("each at two
  further seeds", i.e. the reported arms share one).
* Not done, deliberately: no claim about venue norms, no comparison to other
  submissions, no softening of the replication's PENDING status.

---

## 3. VERIFICATION (final state, all three gates green)

| Gate | Result |
| --- | --- |
| `autopilot\p13_build_zip.py` | 6/6 PASS, main content **9** pages (limit 9), references start p.10 at y=72.79, `ALL_PASS = True` |
| `autopilot\check_manuscript.py` | `RESULT: PASS`, macros undefined 0, citations 0 missing, labels 55 / refs 55, **dangling 0** |
| `autopilot\p15_verify_numbers.py` | `RESULT: PASS`, 20 AUC macros verified, no cross-arm attribution |

No digit anywhere in the manuscript was altered; no limitation was removed. Not
committed, per instruction.

One transient `1_compiles_standalone FAIL` was observed mid-session and did not
reproduce: a concurrent agent was building the same manuscript through the shared
scratch stage `D:\jepa_phase0\autopilot_out\zip_validate\stage`. The LaTeX log
for that run contained no error, and the immediately following build returned
6/6 PASS. Concurrent builds of this paper race on that directory.
