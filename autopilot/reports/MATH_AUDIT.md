# MATH_AUDIT — does this paper need formal notation, and where?

Target: `paper/genai4health2026/main_submission.tex` (read end to end; 1709 source lines when read,
1728 after the concurrent edit noted below; 9-page body + unconstrained appendix). Nothing was
edited by this audit. Every claim below about what the
paper's methods actually do was checked against the production code, not inferred from the prose.

**Note on a concurrent edit.** The manuscript was edited by another agent while this audit was in
progress (guide-provenance paragraph for the adapter: +4 body lines in Sec. 5.3, +15 appendix lines
in Appendix P, `app:repro`). Line numbers below are for the file as it stands after that edit and were
re-verified against it. The edit changes one verdict, candidate (d), which is now assessed against
the adapter text that actually landed rather than treated as hypothetical. It also consumes body
budget: see Sec. 5.

Line costs are **estimates**, not measurements: no LaTeX toolchain is installed on this machine
(`latexmk`, `xelatex`, `pdflatex` all absent), so the paper could not be recompiled to measure the
insertions. The conversion used is the NeurIPS 2026 geometry in `neurips_2026.sty`
(`textwidth=5.5in`, `textheight=9in`, 10pt): approximately **95 characters per typeset line** and
**approximately 54 body lines per page**. Costs are given in typeset body lines.

---

## 1. Overall recommendation

**The prose-first style is correct for this venue and should be kept. Do not add a Method
section of equations.** This paper's contribution is a controlled comparison, not a new
objective; its risk at review is credibility of the comparison, not apparent formality. A block of
standard I-JEPA notation would consume the body budget it has none of, and would signal that the
authors mistook decoration for rigour.

**But three specific things in the paper are genuinely ambiguous, and two of them are load-bearing.**
In each case the ambiguity was confirmed by reading the implementation and finding that the prose
admits a reading the code contradicts. These are not stylistic preferences; a reader who
reimplemented from the current text would build a different system.

Recommended total: **one inline formula and one clause in the body (net cost approximately
+1 typeset line, and a named cut that pays for it), plus two appendix paragraphs that cost nothing.**

Ranked by how much ambiguity is removed per line spent:

1. **CENTROID definition (Sec. 3.2)** — NEEDED. The prose lets a reader believe the band *is* the
   target set. It is not: it is a placement prior for the same four rectangles. This misreading
   changes what the paper's best-performing arm is.
2. **Table 2's five quantities** — NEEDED, and the current caption is actively wrong on one of them
   ("loss slots is the number of predictor targets contributing to the loss" — there are four
   targets; the number printed is approximately 159). Definitions belong in the appendix with one
   disambiguating clause in the caption.
3. **Paired bootstrap estimator** — NEEDED, appendix only, zero body cost. The statistics reviewer
   who scored this 2/6 cannot currently tell what was resampled, what interval type was used, or
   that the head was held fixed.
4. **JEPA objective** — NOT NEEDED in the body. Argued both ways in Sec. 4.
5. **Adapter objective** — NOT NEEDED. The adapter entered the paper during this audit, but only as
   a provenance caveat with no claim attached to it. See Sec. 4.

---

## 2. Candidate table

| # | Candidate | Location | Verdict | Where it goes | Body line cost |
|---|-----------|----------|---------|---------------|----------------|
| a1 | CENTROID band definition | Sec. 3.2 | **NEEDED** | Body, inline math | +1 (net 0 after the paid cut) |
| a2 | ENVELOPE acceptance predicate | Sec. 3.2 | NEEDED but appendix | Appendix (new sampler spec) | 0 |
| a3 | COVER greedy objective and floor | Sec. 3.2 | NEEDED but appendix | Appendix (new sampler spec) | 0 |
| a4 | ANATOMY-v1/v2 target construction | Sec. 3.2 | NOT NEEDED | — | 0 |
| b | Paired bootstrap estimator | Sec. 4 "Evaluation", App. N | **NEEDED** | Appendix N (`app:contrasts`) | 0 |
| c | I-JEPA objective | Sec. 3.1 | **NOT NEEDED** in body | Optional 1 line in appendix | 0 |
| d | Adapter objective | Sec. 5.3, App. P | **NOT NEEDED** | — | 0 |
| e | The five Table 2 quantities | Tab. 2 caption | **NEEDED** | Appendix defs + 1 caption clause | +1, offset to approximately +0.3 |
| f | "partitions the token grid" | Sec. 3.1 | Wording fix, no math | Body, one word | 0 |
| g | Spearman never named in body | Sec. 5.4 | Wording fix, no math | Body, one word | 0 |

---

## 3. The candidates in detail

### (a1) CENTROID — NEEDED

**Current text (Sec. 3.2):**

> A band whose vertical position is located per slice by a per-column intensity-weighted row
> centroid, smoothed across columns. It adapts to the retina's position and curvature from a
> first-order intensity statistic, with no segmentation model and no ground-truth annotation of any
> kind.

**Why this is not merely informal — it is misleading.** Verified in
`src/masks/curriculum.py:1033-1116` (`_anatomical_prior_weight_grid_for_image`) and
`src/masks/curriculum.py:649-692` (`_sample_biased_location`):

- The band is **not** the target set. It is a 0/1 weight grid, and each of the four rectangles has
  its top-left corner drawn by `torch.multinomial` with probability proportional to the band mass
  inside that window. The arm therefore still emits four ordinary rectangles. Table 2 confirms this
  and contradicts the natural reading of the prose: CENTROID's mask ratio is 40.3 percent of the
  grid (approximately 103 cells) and it has approximately 159 loss slots, whereas the band itself is
  7 rows by 10 columns = 70 cells at the production configuration
  (`configs/patch_oracle_anatomical.yaml`: `oracle_region_frac: 0.28`, `oracle_lateral_frac: 0.6`).
  A reader who takes "a band ... " literally will describe the paper's winning arm incorrectly, and
  will not be able to reconcile it with Table 2.
- The intensity is a **patch-grid** quantity (mean over each 16x16 pixel cell), not a pixel quantity.
- The centroid is computed on `I - min(I)` per slice. This is not cosmetic: the training transform
  applies ImageNet normalisation, so raw intensities are signed, and a centroid of signed weights is
  not the same statistic. The code comments record that omitting this caused a real failure mode.
  A reimplementation from the current prose would hit exactly that bug.
- "Smoothed across columns" does not say how. It is a three-tap box filter
  (`avg_pool1d`, `kernel_size=3`, `count_include_pad=False`).

So the answer to the question posed is: **no, CENTROID cannot be reimplemented from the prose**, and
the failure is not a missing hyperparameter, it is a missing mechanism. That earns its space.

**Proposed LaTeX (inline math, not a display, to keep the cost at approximately one line).**
Replaces the existing `\item[\ArmBest{} (segmentation-free).]` block:

```latex
\item[\ArmBest{} (segmentation-free).] With $I(r,c)$ the mean intensity of grid cell $(r,c)$ and
  $\tilde{I}=I-\min I$, the band centre in column $c$ is the intensity-weighted row centroid
  $\bar{r}(c)=\sum_{r} r\,\tilde{I}(r,c)\big/\sum_{r}\tilde{I}(r,c)$, box-smoothed over three
  columns. The four rectangles are then drawn with probability proportional to the band mass they
  cover: the band is a placement prior, not the target set. No segmentation model and no
  annotation of any kind (constants in Appendix~\ref{app:samplers}).
```

**Cost.** Current block is approximately 330 characters, approximately 3.5 typeset lines.
Replacement is approximately 480 characters, approximately 5 lines. **Net +1.5, call it +1 to +2.**
`\min` and `\big/` need no packages beyond `amsmath`, already loaded (line 10).

**What pays for it.** Sec. 5.1 currently opens a bolded paragraph with a restatement that the
formula makes redundant:

> \ArmBest{} finds its band from a per-column intensity centroid: no model and no annotation.

Trim to "\ArmBest{} consults no segmentation model and no annotation." — **saves approximately
1.5 lines**, and the paragraph's argument (the margin is largest at epoch 100 and does not decay) is
untouched. Net cost of (a1) after this cut: approximately **0**.

**If a display equation is preferred instead**, the same content as a numbered display costs
approximately +3.5 lines (equation body plus `\abovedisplayskip` and `\belowdisplayskip`) and makes
it the paper's only numbered equation, which draws attention to the absence of others. The inline
form is strictly better here.

### (a2) ENVELOPE acceptance predicate — NEEDED, APPENDIX

"Rejection-sampled onto the retinal envelope" names the mechanism unambiguously but not the
acceptance test, which is the policy's only free parameter. From
`configs/patch_mirage_envelope.yaml`: a candidate rectangle is accepted only if at least 40 percent
of its cells lie on the envelope (`mirage_min_block_fill: 0.40`), subject to a minimum retained
visible-retina fraction (`mirage_min_retina_visible: 0.25`), with up to 30 attempts
(`mirage_max_attempts: 30`) before falling back to uniform placement, at occupancy threshold 0.25.
This matters to a reader of Sec. 5.1, which says "fewer than half of ENVELOPE's masked cells land on
tissue" — that number is a direct consequence of the 0.40 acceptance bar, and stating the bar turns
a surprising observation into an expected one.

Appendix, costs nothing. See the combined proposal in Sec. 6 below.

### (a3) COVER greedy objective and floor — NEEDED, APPENDIX

"Targets chosen greedily to cover the anatomy subject to a hard floor leaving a fraction $f$ of
tissue visible in the context" leaves three things open, and the code (`src/masks/cover.py:132-240`)
resolves each in a way the prose does not imply:

- The floor is on anatomy **mass** (the sum of soft segmenter scores over the not-hidden cells), and
  there is a **second** floor on the anatomy **cell count** (`min_visible_cells`, default 4). "A
  fraction $f$ of tissue" reads as a cell-count fraction.
- "Visible in the context" is not the context set. The floor is evaluated on anatomy **not covered
  by targets**, which is a strictly larger set than anatomy inside the context block, because a cell
  can be in neither (see (f) below). The paper uses "visible" here and "context kept" in Table 2 for
  two different sets.
- Blocks not needed for coverage are not coverage blocks at all: they are placed to straddle the
  tissue/background boundary (`fill = "transition"`, the shipped default). This is a second,
  unmentioned policy inside the arm.

This does not change any claim in the paper (the arm's result is retracted as evidence about
coverage anyway, Appendix G, `app:covbug`), which is exactly why it belongs in the appendix and not
in the body.

### (a4) ANATOMY-v1 / v2 — NOT NEEDED

"Ragged connected components grown on the MIRAGE score map ... v2 additionally bridges diagonal
adjacencies" plus the already-stated $K{=}16$ resampling is sufficient for the argument the paper
makes about these arms, which is that they are confounded and not identified. A precise definition
would let a reader reproduce an arm the paper explicitly declines to draw a conclusion from. Adding
notation here would be padding.

### (b) THE PAIRED BOOTSTRAP — NEEDED, APPENDIX, ZERO BODY COST

**Where it is described today.** Three places, in three different vocabularies, none complete:

- Sec. 4 "Evaluation": "paired bootstrap confidence intervals over test volumes (10,000 resamples,
  stratified by class, the same resampled index set applied to every arm)".
- Table 8 caption (`tab:pairedsub`): "Each bootstrap draw resamples subjects once and applies the
  same resampled set
  to both arms and every stratum".
- Table 11 caption (`tab:subop`): "Paired bootstrap over positives within each stratum".

**What a statistics reviewer cannot currently determine, and the code's answer:**

| Question | Answerable from the paper? | Actual (`autopilot/p1c_stats.py:70-125`) |
|---|---|---|
| Resampling unit | Yes — test volumes, one per subject | positives resampled to $n_+$, negatives to $n_-$ |
| Are probes or heads resampled or refit? | **No** | No. `S[k]` is a stored fixed score vector; only the index set moves |
| Interval type: percentile, basic, BCa? | **No** | Percentile, `np.percentile(db, [2.5, 97.5])`, no bias correction, no acceleration |
| Is the point estimate the full-sample value or the bootstrap mean? | **No** | Full-sample `fast_auc(y, S[k])` |
| Is prevalence fixed across draws? | Implied by "stratified", not stated | Yes, exactly $n_+$ and $n_-$ every draw |
| Is `tab:subop` the same estimator as `tab:pairedsub`? | **No — the captions differ and neither says why** | No. `autopilot/p16_subgroup_operating.py:160` resamples positives only, within stratum |

The third and sixth rows are the ones that will cost marks. "Percentile interval, no BCa" is a
choice a reviewer is entitled to see stated, and the paper currently reports two different
estimators under one name.

**Proposed LaTeX. Insert into Appendix N (`app:contrasts`), before `\begin{table}`. Body unchanged.**

```latex
\paragraph{The estimator.} Let $i=1,\dots,N$ index the $N{=}\Ntest$ test volumes, one per subject,
with labels $y_i$, and let $s^{A}_{i}$ be the score stored for volume $i$ by arm $A$'s frozen probe.
The point estimate is the full-sample difference
$\widehat{\Delta}_{AB}=\mathrm{AUC}(y,s^{A})-\mathrm{AUC}(y,s^{B})$, not a bootstrap mean. For each
of $B{=}\Nboot$ draws we resample the positive indices with replacement to size $n_{+}$ and the
negative indices with replacement to size $n_{-}$, giving one index multiset $\mathcal{I}_b$ that is
applied to \emph{every} arm, so that
$\Delta^{*}_{b}=\mathrm{AUC}(y_{\mathcal{I}_b},s^{A}_{\mathcal{I}_b})-\mathrm{AUC}(y_{\mathcal{I}_b},s^{B}_{\mathcal{I}_b})$
and the between-case variance common to the two arms cancels within a draw. The reported interval is
the \emph{percentile} interval, the $2.5$th and $97.5$th percentiles of
$\{\Delta^{*}_{b}\}_{b=1}^{B}$; no bias correction or acceleration is applied. Because the strata are
resampled at their observed sizes, prevalence is held at $n_{+}/N$ in every draw.

\paragraph{What is and is not resampled.} Cases only. The probe is not resampled and the head is not
refitted: $s^{A}$ is a fixed stored vector, so the interval carries sampling error over the test
cohort and \emph{no} component of probe-fitting, probe-seed or pretraining variance
(Section~\ref{sec:limits}). One variant is used, and only in Table~\ref{tab:subop}: sensitivity at a
fixed threshold is defined on positives alone, so there each draw resamples the positives within the
stratum and leaves the negatives fixed. Every other interval in this paper, including
Table~\ref{tab:pairedsub}, uses the scheme above.
```

**Cost: 0 body lines**, approximately 16 appendix lines. Appendix N (`app:contrasts`) already exists
and already carries the multiplicity-family discussion, so this is the natural home.

### (c) THE JEPA OBJECTIVE — NOT NEEDED (argued both ways)

**The case for restating it.** (i) Table 2's "loss slots" column is defined by reference to the loss
sum, and without the loss written down the phrase has no referent — indeed the caption's current
gloss of it is wrong (Sec. 3(e)). (ii) A one-line loss in which the mask set is the only
policy-dependent argument would make the controlled-comparison design visible at a glance: the arms
share every symbol except one. (iii) It is one line and it is uncontroversial.

**The case against.** (i) The objective is stock I-JEPA, cited at
`\citep{assran2023ijepa}`, and unmodified; restating a citable loss invites the reviewer to ask what
about it is the contribution, which is the opposite of the impression the paper wants. (ii) The
design point that only the sampler differs is already stated twice in Sec. 4 and stated
unambiguously ("Within the rectangle family the *only* difference between arms is the mask
sampler"). Prose already does this job. (iii) Three body lines in a paper with zero slack must come
out of a result. (iv) The ambiguities that actually exist in this paper are all on the mask side,
not the loss side; an equation for the loss would answer a question nobody asked.

**Verdict: NOT NEEDED in the body.** Optionally, one line in the appendix sampler section purely as
the anchor for "loss slots":

```latex
The predictor is regressed on every target token of every target block, so the number of terms in
the loss for one image is $\sum_{m=1}^{M}|B_m|$, counting a cell once per block that claims it. This
is the ``loss slots'' column of Table~\ref{tab:geom}.
```

Cost: 0 body lines, 3 appendix lines.

### (d) THE ADAPTER OBJECTIVE — NOT NEEDED

The adapter landed in the manuscript during this audit (Sec. 5.3, four lines; Appendix P
(`app:repro`), fifteen
lines). What landed is a **provenance caveat**, not a method: ANATOMY-v1 and COVER read soft guides
from a MIRAGE carrying "a frozen residual adapter, trained label-free to align MIRAGE's
patch-similarity structure with the pretraining teacher's", and the paper states explicitly that the
adapter "was an abandoned line of work", that it "never improved segmentation", and that "we make no
claim about it here".

**Verdict: NOT NEEDED, and adding the objective would be actively harmful.** The claim the paragraph
supports is "the guides differ, so this is one more axis on which the anatomy arms are confounded".
That claim is fully carried by the fact of a difference; the *form* of the adapter's training loss
is irrelevant to it. Writing the loss down would suggest the adapter is a component of the method
being evaluated, which is the opposite of what the paragraph is for, and would invite a reviewer to
ask for the very downstream experiment the paragraph says was not run.

**Conditional rule, if this ever changes.** The moment any claim in the paper depends on what the
adapter optimised — for example if a corrected arm is run with and without adapted guides — its
objective becomes NEEDED and belongs in the body, because the reader must then be able to see what
was optimised and whether that optimisation was shared across arms. Budget approximately 4 body
lines and expect to cut a paragraph, not a sentence.

**One notation check on the new text.** The new material introduces no symbols, and "soft guides"
versus "occupancy map" is a distinction the existing appendices already draw (Appendix D
(`app:geomprov`) measures anatomy "from the MIRAGE guide occupancy channel at threshold $0.25$").
No collision, no action.

### (e) THE FIVE TABLE 2 QUANTITIES — NEEDED

The prompt is right that Table 2's whole argument rests on five quantities defined only in a
caption. It is worse than that: **two of the five are not defined at all, and one of the two that is
defined is defined incorrectly.**

- Defined in the caption: "purity" (fraction of masked cells lying on tissue) — correct; "loss
  slots" (the number of predictor targets contributing to the loss) — **wrong**. There are $M{=}4$
  predictor targets. The printed value is approximately 159. The quantity is the number of target
  *cells*, counted once per block that claims them.
- Not defined anywhere: **"anatomy hidden"** and **"mask ratio"**. "Context kept" is defined only in
  an appendix table caption (Appendix D, `tab:delivered`), a long way from Table 2.
- "Anatomy hidden" and "purity" share a numerator (masked tissue cells) and differ only in
  denominator (all tissue cells; all masked cells). Nothing in the paper says this, and the section
  turns on the contrast between them: "the best performer hides the least anatomy, while the arm
  that hides the most --- at 97.1 percent purity --- does not separate from the null". A reader who
  guesses the wrong denominator reads that sentence as self-contradictory.
- The hardest re-read in the paper: RANDOM has mask ratio 44.5 percent, i.e. approximately 114 of
  256 cells, but 159.9 loss slots. Nothing in the text explains how the mask can be 114 cells and
  the loss 160 terms. The answer is that blocks overlap and slots count multiplicity, which no
  reader can obtain from the prose.

**Verification that the proposed definitions are the right ones.** Using only the printed Table 2
values and the definitions below, the implied anatomy area $|A|$ can be recovered independently from
each of the five rows. It must be identical for all five, because $A$ is a property of the images
and not of the policy:

| policy | implied $\lvert T\rvert$ | implied $\lvert T\cap A\rvert$ | implied $\lvert A\rvert$ | slots $-\ \lvert T\rvert$ |
|---|---|---|---|---|
| random | 113.92 | 35.88 | **66.45** | 45.98 |
| centroid | 103.17 | 41.27 | **66.45** | 55.83 |
| envelope | 119.04 | 51.54 | **66.42** | 40.66 |
| cover | 110.59 | 48.88 | **66.51** | 48.51 |
| anatomy-v2 | 54.53 | 52.95 | **66.27** | 9.47 |

Five independent rows agree on $|A| \approx 66.4$ cells (25.9 percent of the grid) to within 0.4
percent, and the last column is positive everywhere, confirming both the denominators and the
multiplicity counting. The definitions below are therefore the ones the numbers were produced by,
not a guess. (Confirmed independently in
`scripts/mask_composition_probe.py:182-203` and `autopilot/compare_table2_geometry.py:37-43`:
`hidden_share_of_all_anat`, `hidden_pct_on_anat`, `hidden_frac_of_grid`, `ctx_frac_of_grid`,
`n_slots_mean`.)

**Proposed LaTeX, option E2 — RECOMMENDED. Appendix definitions (free) plus one caption clause.**

Appendix D (`app:geomprov`), in the "How the geometry was measured" paragraph:

```latex
Write $G$ for the $16\times16$ cell grid, $B_1,\dots,B_M$ for the delivered target blocks,
$T=\bigcup_m B_m$ for the masked cells, $C$ for the context cells the encoder retains, and $A$ for
the anatomy cells (MIRAGE occupancy above $0.25$). $T$ and $C$ are disjoint but need not exhaust
$G$. The five columns of Table~\ref{tab:geom} are then
anatomy hidden $=|T\cap A|/|A|$, purity $=|T\cap A|/|T|$, mask ratio $=|T|/|G|$,
context kept $=|C|/|G|$, and loss slots $=\sum_{m}|B_m|$, which counts a cell once per block that
claims it and therefore exceeds $|T|$ whenever blocks overlap. Each column is a ratio of
per-image means over the 600 slices, not a mean of per-image ratios.
```

Table 2 caption: replace

> ``purity'' is the fraction of masked cells lying on tissue; ``loss slots'' is the number of
> predictor targets contributing to the loss.

with

```latex
``anatomy hidden'' and ``purity'' share a numerator, the masked tissue cells, and differ in
denominator: all tissue cells, and all masked cells. ``loss slots'' counts a target cell once per
block that claims it, so it exceeds the masked-cell count (definitions in
Appendix~\ref{app:geomprov}).
```

**Cost.** Removed approximately 165 characters, added approximately 290: **net +1.3 lines**. Offset
by shortening the caption's last sentence from "AUC is quoted at the *matched* epoch 50, so the
geometry-to-AUC association is not read across mixed epochs" to "AUC is at the *matched* epoch 50,
so no association is read across epochs" (**saves approximately 0.8 lines**, keeps the defensive
point). **Net approximately +0.5 lines.** Appendix cost 10 lines, free.

**Option E1, not recommended at the current budget:** put the full symbolic legend in the caption
itself (net +1.5 to +2 body lines). Better paper, but it costs a result-bearing sentence, and E2
retains the two disambiguations that actually matter to a body-only reader.

---

## 4. Notation consistency audit

Checked every math-mode token in the source. Findings, in order of severity:

**S1. "Partitions" is contradicted by the paper's own table (Sec. 3.1, line 205).** "I-JEPA
partitions the token grid of an image into a context block and $M$ target blocks." A partition
implies $C\cup T = G$ and $C\cap T=\varnothing$. Table 2 gives RANDOM mask ratio 44.5 percent and
context kept 41.9 percent, summing to 86.4 percent. Cells that are in neither set exist because the
context rectangle need not cover the grid. **Fix: replace "partitions" with "samples from" — one
word, zero lines.** A careful reviewer will notice this, and it is the kind of small internal
inconsistency that makes a statistics reviewer distrust the rest.

**S2. $\rho$ is used four times in the body without ever naming the statistic (Sec. 5.4, lines
578-588).** "Spearman" appears only once in the whole paper, in Appendix D (`app:geomprov`) line 936,
and there it
refers to a *different* correlation (anatomy-hidden versus AUC). A reader of the body cannot tell
whether the subgroup-gap trends are Spearman or Pearson. **Fix: "at Spearman $\rho=\SubRaceRho$" at
first use — one word, zero lines.**

**S3. Dangling numeric cross-reference (Appendix D, line 936).** "The $+0.80$ anatomy-hidden/AUC
Spearman coefficient of Section~\ref{sec:results-h3}" — Section 5.3 no longer prints $+0.80$; it
now says only that "rank correlations over these arms are positive ... at $n{=}4$--$5$ arms they
resolve nothing". The appendix attributes to the body a number the body does not contain. Not a
notation issue, but found during this audit and worth fixing: either restore the value in the body
or change the appendix to "the $+0.80$ anatomy-hidden/AUC Spearman coefficient behind the rank
correlations of Section~\ref{sec:results-h3}".

**S4. "Visible" denotes two different sets.** Sec. 3.2 defines COVER's floor as "leaving a fraction
$f$ of tissue **visible in the context**"; Table 2's "context kept" is $|C|/|G|$. The code's floor is
on anatomy **not in $T$**, which is a strictly larger set than anatomy in $C$. Low impact on claims
(the arm's coverage reading is retracted), but a reader comparing $f{=}0.21$ against Table 2's
"context kept" column will compare incommensurable things. Handled by the appendix sampler spec.

**S5. $n$ is overloaded across five referents** — arms ($n{=}4$--$5$), continuations ($n{=}1$),
labelled cases ($n{=}\LENFive$), slices ($n{=}1{,}534$), subgroup sizes ($n{=}\NBlack$), plus $n_{+}$
for stratum positives. This is conventional and I would not change most of it, **except one
collision**: the caption of `fig:maskstats` (line 821) says the ANATOMY-v2 sweep ran on
"$n{=}1{,}534$" slices,
and Sec. 4 says the test split is "1466 positive / 1534 negative" volumes. The same numeral, the same
symbol, two unrelated quantities, and a reader can plausibly conclude the mask sweep was run on the
test negatives — which would contradict the paper's own statement that the sweep uses Training
slices only. **Fix: "$n{=}1{,}534$ slices" — one word, zero lines.**

**Clean, no action:** $M$ (target blocks, introduced at first use in Sec. 3.1 and reused correctly in
Sec. 3.2); $f$ (COVER floor, introduced at first use, consistent across all seven occurrences);
$K$ (anatomy resampling, introduced at first use in Sec. 5.3, consistent in Appendices G and I);
$q$ (BH-adjusted, introduced in context at first use); $N$ (test cohort size, consistent everywhere);
$\Delta$ (defined in every caption that uses it); $p$ (DeLong, stated); $W{=}7$ (occlusion window,
local to Appendix J, no clash). The $k$ in "$k$-NN" (`tab:counter`) is a quoted metric name from cited
work and does not collide with $K$ in practice, though a reader skimming would not confuse them.

**Symbols proposed above are all currently unused in the paper:** $G$, $T$, $C$, $A$, $B_m$, $I$,
$\bar{r}$, $\mathcal{I}_b$, $s^{A}$, $\Delta^{*}$, $n_{-}$. No new collisions are introduced.

---

## 5. Space accounting

The body has zero slack, so every proposal above is either free or paid for by a named cut.

| Item | Body cost | Paid by |
|---|---|---|
| (a1) CENTROID inline formula and placement-prior clause | +1.5 | Trim the Sec. 5.1 restatement "\ArmBest{} finds its band from a per-column intensity centroid: no model and no annotation." to "\ArmBest{} consults no segmentation model and no annotation." (−1.5) |
| (e) Table 2 caption denominator clause and slots correction | +1.3 | Shorten the caption's closing "so the geometry-to-AUC association is not read across mixed epochs" to "so no association is read across epochs" (−0.8) |
| (f) "partitions" to "samples from" | 0 | — |
| (g) "Spearman $\rho$" | 0 | — |
| (S5) "$n{=}1{,}534$ slices" | 0 | — |
| **Body total** | **approximately +0.5 lines** | — |
| (b) bootstrap estimator | 0 | Appendix N, unconstrained |
| (a2)(a3)(c) sampler specification | 0 | New appendix, unconstrained |
| (e) geometry definitions | 0 | Appendix D, unconstrained |

**The budget moved during this audit.** The concurrent adapter edit added approximately 4 typeset
lines to Sec. 5.3 (the guide-provenance sentence), which is eight times the net cost of everything
proposed here. If the body was at exactly 9 pages before that edit it is now over, and the cuts
named in this section are needed regardless of whether any of this audit's proposals is accepted.
That is an argument
for accepting the proposals, not against: (a1) and (e) each come with their own offsetting cut, so
they are close to self-financing, whereas the adapter sentence came with none.

Approximately +0.5 typeset lines is under one percent of a page and is within the noise of a single
paragraph reflow; it is very likely absorbed without any page-break change, but since the document
could not be recompiled here this should be confirmed by a build before committing. If it is not
absorbed, a further approximately 2 lines are available at no argumentative cost by deleting the
duplicated sentence in the "Broader impact and ethics" paragraph of Sec. 6, which is a compressed restatement
of Appendix L (`app:ethics`) and is the one paragraph in the body whose content is fully recoverable
from the appendix.

**Second reserve, if more is needed:** Sec. 5.1's parenthetical "(Table~\ref{tab:contrasts},
Figure~\ref{fig:traj}b)" duplicates a pointer given one sentence earlier.

---

## 6. Consolidated appendix proposal (free)

If the above is accepted, one new appendix section carries (a2), (a3) and optionally (c). Suggested
placement: immediately after Appendix D (`app:geomprov`), which already discusses the samplers.

```latex
\section{Sampler specification}
\label{app:samplers}

All arms draw $M{=}4$ target blocks and one context block per image under identical scale and
aspect-ratio ranges; block sizes are sampled once per batch and shared across the images in it, so
the arms differ only in where those blocks land. Write $G$, $B_m$, $T$, $C$ and $A$ as in
Appendix~\ref{app:geomprov}.

\paragraph{\textsc{envelope}.} A candidate rectangle $B$ is accepted only if
$|B\cap A|/|B|\geq 0.40$ and its acceptance would leave at least a fixed fraction of the retina
outside $T$; up to $30$ candidates are drawn per block, after which the arm falls back to uniform
placement. Anatomy is the MIRAGE occupancy channel at threshold $0.25$. The $0.40$ bar is why fewer
than half of \textsc{envelope}'s masked cells lie on tissue (Table~\ref{tab:geom}): a block need
only be $40\%$ on-tissue to be legal.

\paragraph{\ArmBest{}.} On the production configuration the band is $7$ grid rows tall and spans the
central $10$ of the $16$ columns. Its per-column centre is the smoothed centroid $\bar{r}(c)$ of
Section~\ref{sec:policies}, and the four rectangles are drawn with probability proportional to the
band mass inside each candidate window, so the band biases placement rather than defining $T$.
Subtracting the per-slice minimum before taking the centroid makes the statistic invariant to the
affine input normalisation.

\paragraph{\textsc{cover}.} Blocks are placed greedily to maximise the anatomy mass they hide,
subject to two hard floors: a fraction $f$ of anatomy \emph{mass} and a minimum number of anatomy
\emph{cells} must remain outside $T$. Note that this floor is on anatomy not covered by a target,
which is a larger set than the anatomy inside $C$. Blocks not needed to reach the floor are placed
to straddle the tissue/background boundary rather than dropped on vitreous. The defect of
Appendix~\ref{app:covbug} acts after all of this.

\paragraph{Loss slots.} The predictor is regressed on every target token of every target block, so
the number of terms in the loss for one image is $\sum_{m}|B_m|$, counting a cell once per block
that claims it. This is the ``loss slots'' column of Table~\ref{tab:geom}, and it is why the
anatomy arms' fixed $K{=}16$ gives exactly $4\times16=64$.
```

Approximately 30 appendix lines, zero body lines. This also retires the body's need to explain the
$4\times16=64$ arithmetic twice (Sec. 5.3 and Appendix D).

---

## 7. What was deliberately rejected

Listed so the decision is on record rather than an omission:

- A Method section with the I-JEPA loss, EMA update and predictor definition. Standard, citable,
  and would cost 8 to 12 body lines the paper does not have.
- Notation for the hypotheses H1/H2/H3 (for example $\mathrm{AUC}(\pi_{\mathrm{env}}) >
  \mathrm{AUC}(\pi_{\mathrm{rand}})$). The prose statements are already unambiguous and the symbolic
  version is strictly longer.
- A formal definition of the frozen-probe protocol (mean-pool, LayerNorm, linear). Prose is exact
  and shorter.
- DeLong's statistic. Cited, standard, and used only as a secondary $p$-value.
- Benjamini-Hochberg. The family is declared explicitly in Appendix N (`app:contrasts`), which is the
  part that
  actually matters and is already done well.
- A definition of AUC.

---

## 8. Evidence trail

| Claim in this report | Checked against |
|---|---|
| CENTROID band is a placement prior, not the target set | `src/masks/curriculum.py:1033-1116`, `:649-692` |
| Band is 7 rows by 10 columns at production settings | `configs/patch_oracle_anatomical.yaml:48,50`; matches the config's own comment |
| Centroid uses patch means and per-slice min subtraction | `src/masks/curriculum.py:1056-1072` |
| Three-tap box smoothing across columns | `src/masks/curriculum.py:1091-1094` |
| ENVELOPE acceptance bar 0.40, 30 attempts, threshold 0.25 | `configs/patch_mirage_envelope.yaml:95,99,100,107` |
| COVER floors are on mass and on cell count; leftover blocks straddle | `src/masks/cover.py:147-240` |
| Bootstrap: class-stratified case resampling, shared index set | `autopilot/p1c_stats.py:70-79` |
| Bootstrap: percentile interval, no BCa, full-sample point estimate | `autopilot/p1c_stats.py:86,122-123` |
| $B = 10{,}000$ | `autopilot/p1c_stats.py:34`; `auto/auto_numbers.tex` `\Nboot` |
| `tab:subop` uses a different, positives-only scheme | `autopilot/p16_subgroup_operating.py:160` |
| Subgroup AUC bootstrap matches the main scheme | `autopilot/p7c_paired_subgroup.py:112-113,129,147` |
| The five geometry definitions | `scripts/mask_composition_probe.py:182-203`; `autopilot/compare_table2_geometry.py:37-43`; independently reproduced from Table 2's printed values (Sec. 3(e)) |
| No adapter content in the paper | superseded: adapter provenance added mid-audit at Sec. 5.3 and Appendix P (`app:repro`), assessed in Sec. 3(d) |
| Page geometry used for line costs | `neurips_2026.sty:131-132` |
| No LaTeX toolchain available to measure costs | `Get-Command latexmk,xelatex,pdflatex` returns nothing |
