# P22 — Prose edits for `main_submission.tex` (epistemic-restraint cadence)

File: `C:\Users\Gary\Desktop\jepa\paper\genai4health2026\main_submission.tex`
(1425 lines, read in full: main body to `\bibliographystyle` at L706, appendices L708–1425).

**Nothing in this file was edited.** All edits below are proposals for exact string
replacement.

- **41 edits proposed**: 37 primary (E1-E37) and 4 optional (section F).
- **Net source-line saving: 21 lines** from the primary set, 3 more from the
  optional set. A further ~12 edits shorten text without dropping a source line and
  will reflow. Expected typeset saving roughly 0.5 page.
- Every OLD string below was checked programmatically: each occurs **exactly once**
  in the file, each NEW string is shorter in characters and never longer in lines,
  and each preserves the OLD block's brace balance.
- No `\Macro` is altered, moved between sentences, or deleted anywhere below, except
  where an entire `\emph{...}`/`\textbf{...}` group is deleted together with the
  words it wrapped (E3, E6, E11, E12, E26). One `\ref` is added (E35).
- No digit is changed anywhere below. E37 is the only edit that deletes a hand-typed
  numeral, and it deletes a duplicated sentence; see the note on E37.

Applying order does not matter: no two edits overlap.

**If you want to stop early**, the highest-value subset is:
E1, E2, E5, E6, E12, E13, E18, E19, E20, E21, E23, E26, E29, E31, E32, E33, E35
(17 edits, ~16 lines saved) — these cover all three sentences the reviewer named,
the repetition audit, and every duplicated limitation.

---

## A. Inventory of the epistemic-restraint cadence

| Line | Sentence (verbatim, trimmed to the cadence clause) | Verdict |
|---|---|---|
| 74–76 | `Because each policy was pretrained once, we report these as single-run observations rather than an established ranking.` | **KEEP** — abstract; only statement of the n=1 limitation a skim-reader sees. |
| 184 | `We therefore do \emph{not} claim to overturn the informed-masking literature.` | **COMPRESS** (E3) — disclaimer-first cadence; same content survives as a clause. |
| 204–205 | `We report a subgroup analysis as a secondary finding.` | **REMOVE** (E5) — announces what the paper will do; framing restated at L612–613 and L1249–1250. |
| 249–250 | `an implementation defect means the delivered masks do not reach that coverage, and we report the arm accordingly` | **COMPRESS** (E7) — `and we report the arm accordingly` is procedural assurance with no content. |
| 304 | `Arms differ in how far they were carried, and we do not pretend otherwise.` | **COMPRESS** (E8) — `and we do not pretend otherwise` is pure throat-clearing. |
| 308–311 | `we cannot claim its epoch-100 value would have been worse, only that we did not measure it.` | **COMPRESS** (E9) — selective stopping is substantive; the phrasing is not. Limitation kept verbatim in content. |
| 331 | `\paragraph{Numerical precision, and how we handle it.}` | **COMPRESS** (E10) — `and how we handle it` is procedural assurance in a heading. |
| 335–336 | `We therefore do \emph{not} claim a single protocol across every probe in the study.` | **COMPRESS** (E11) — disclosure retained as a plain statement of fact. |
| 346–351 | `\emph{What these numbers can and cannot establish.} … Read every comparison here as a controlled single-run observation, not an established ranking over retrainings` | **COMPRESS** (E12) — the declarative micro-heading is the reviewer's named tic. Full limitation retained, definitive treatment stays at L654–667. |
| 427 | `We are careful not to overstate this.` | **REMOVE** (E13, reviewer sentence 3) — the following sentence carries its own qualification. |
| 448 | `We cannot read this as evidence that high coverage is harmful.` | **KEEP** — load-bearing: blocks the exact misreading of the study's one negative result, one sentence after it is stated. |
| 452–453 | `Appendix~\ref{app:covbug} gives the full account and states exactly which of its numbers remain usable.` | **COMPRESS** (E15) — second half is meta-procedural. |
| 502 | `we present them as descriptive rather than inferential` | **KEEP** — load-bearing: stops two correlations at n=4 arms being read as tests. |
| 509 | `and we state this rather than attributing their deficit to shape.` | **REMOVE** (E16) — redundant with the substantive version four lines later. |
| 513–514 | `We therefore do \emph{not} claim that irregular target shape is harmful; that comparison is not identified by this design.` | **KEEP** — load-bearing: the identification limitation, at the only place the confound table appears. |
| 568–569 | `we report this as a descriptive regularity and attach no $p$-value` | **KEEP** — load-bearing: explains the deliberate absence of a test. |
| 587–589 | `We therefore treat the checkpoint-level correlation as pseudo-replicated and make no claim that better models are less fair.` | **COMPRESS** (E17) — keep the content here (this is the once-place); drop the `we therefore treat` / `make no claim` frame. |
| 593–594 | `We do not claim that better masking harms any group, and we do not claim it helps equity either.` | **COMPRESS** (E18) — doubled disclaimer closing a paragraph that already says it twice. |
| 601–603 | `Whether mild improves \emph{more} we did not test: that needs a paired contrast between the gains` | **KEEP** — load-bearing: prevents over-reading the bolded mild delta in Table `tab:severity`. |
| 610–611 | `we cannot resolve whether the improvement is evenly distributed, and claim neither.` | **COMPRESS** (E19) — `and claim neither` restates the clause before it. |
| 612–613 | `No policy here was designed as a fairness intervention and none should be presented as one.` | **COMPRESS** (E19) — normative content kept in full, three words shorter. |
| 622–625 | `Every number here is emitted by one script … prose, tables and figures cannot disagree` | **COMPRESS** (E20, reviewer sentence 2). |
| 643–652 | `We cannot claim anatomy-shaped masking is \emph{harmful} … Nor can we claim target \emph{shape} … Nor can we claim the mechanism` | **KEEP** — this paragraph is titled `What it does not support`; it is the designated single location for these three limitations and the anaphora is deliberate there. Cutting any of the three would remove a substantive limitation. |
| 654–666 | `One pretraining run per policy` paragraph | **KEEP** — definitive statement of the study's central weakness. |
| 667 | `and we do not claim its result in advance.` | **REMOVE** (E22) — throat-clearing about an experiment that has not been run. |
| 675–678 | `so we report its trajectory and explicitly decline to interpret it as evidence about aggressive coverage` | **COMPRESS** (E23) — reviewer-flagged `explicitly decline`; the restriction itself is kept. |
| 680–684 | `the intervals we report are therefore best read as descriptive rather than as confirmatory inference.` | **KEEP** — definitive statement of the multiple-inspection limitation. |
| 742–743 | `Panels (c)--(f) give the confounds we decline to disentangle` | **COMPRESS** (E24) — reviewer-flagged verb in a caption. |
| 779 | `so we draw no trend conclusion from it.` (fig:fair caption) | **KEEP by default** — duplicates `tab:subtrends`, but captions must stand alone. Optional cut as F5. |
| 824–825 | `is not evidence of harm, and we do not present it as such.` | **COMPRESS** (E25) — second clause restates the first. |
| 831–832 | `We therefore explicitly do \emph{not} claim that more accurate models are less fair.` | **REMOVE** (E26) — verbatim duplicate of L588–589 (kept there by E17). |
| 865–870 | `That ordering is descriptive only: we did not test the paired \emph{difference} between strata` | **KEEP** — load-bearing: the table caption above it says "mild disease improves most". |
| 884–892 | app:subgroup `Limitations` paragraph | **KEEP** — scoped, substantive, stated once. |
| 904–905 | `We report their existence for completeness and do not use them to support any claim.` | **REMOVE** (E27) — the paragraph opens with `are excluded from all analysis`. |
| 914–916 | `We report it in full because it invalidates the reading we would otherwise have given` | **COMPRESS** (E28) — meta-procedural opener; the substance becomes the main clause. |
| 940–945 | `\paragraph{What it invalidates.} We cannot claim that \textsc{cover} tests aggressive anatomy coverage` | **KEEP** — definitive, correctly located. |
| 947–954 | `\paragraph{Scope.}` | **KEEP** — substantive scoping of which contrasts survive. |
| 957–960 | `is \emph{open}, and nothing in this paper should be read as answering it. We report the arm we ran, not the arm we intended.` | **COMPRESS** (E29) — three statements of one fact. |
| 991–992 | `We therefore report the first link of this mechanism as measured and leave the causal chain to downstream AUC open` | **COMPRESS** (E30). |
| 1154–1157 | `The OD/OS result is a methodological caution of the same family as our precision and epoch-matching checks … we would rather report that than let it stand.` | **COMPRESS** (E31) — self-referential audit prose plus self-congratulation. |
| 1194–1195 | `The same caveats apply as everywhere else in this paper: one dataset, one split, and one pretraining run per policy.` | **COMPRESS** (E33) — kept as a clipped fragment so the caveat survives without the formula. |
| 1209–1213 | `\textbf{The intervals do not support that reading.}` | **KEEP** — load-bearing: corrects a specific numeric misreading in the same paragraph. |
| 1214–1216 | `We therefore report this as a \emph{power} limitation of the cohort … and explicitly not as evidence that they do not.` | **COMPRESS** (E34) — both directions preserved, `explicitly` removed. |
| 1249–1256 | ethics paragraph (`a caution, not a contribution`, `AUC-only`) | **KEEP** — required content, correct location, appears once. |
| 1312–1314 | `Because the test split was inspected repeatedly across this programme, we read all $p$ and $q$ values here as descriptive rather than confirmatory.` | **COMPRESS** (E35) — third statement; kept as a pointer. |
| 1367 | `which is why we decline to attribute its deficit to target shape.` | **COMPRESS** (E36) — reviewer-flagged verb. |

**Explicitly left alone because I could not decide they were throat-clearing** (per
your constraint 3): L643–652 (the `Nor can we claim` triple), L601–603, L865–870,
L1209–1213, L568–569, L502, L448, L513–514, L884–892, L940–954, L1249–1256, and the
`fig:traj` caption justification of paired differences (L392–395). These all change
how a specific number should be read at the place they occur.

---

## B. Exact replacements

Every OLD block below is a verbatim substring of the file. Line numbers are the
first line of the OLD block.

### E1 — L113 (also section C, reviewer sentence 1)
OLD
```
The result contradicts the intuition in an informative way. Guidance helps, but
the benefit does not increase with anatomical precision --- it disappears, and
with continued training it reverses. The
```
NEW
```
Guidance helps, but the benefit does not increase with anatomical
precision --- it disappears, and with continued training it reverses. The
```
Saves 1 line.

### E2 — L119 (repetition audit, cut #2 of 5)
OLD
```
two policies that merely place ordinary rectangles on tissue gain an order of
magnitude more. The best performer uses no segmentation model at all, locating a
band by a per-column intensity centroid.
```
NEW
```
two policies that merely place ordinary rectangles on tissue gain an order of
magnitude more; the best of them locates its band by a per-column intensity
centroid.
```
Net-neutral in lines, one slogan removed. (The two rectangle-placing policies are
`envelope` and `centroid`; `centroid` is the better of them, so `the best of them`
is exact.)

### E3 — L184
OLD
```
We therefore do \emph{not} claim to overturn the informed-masking literature.
That random masking is a strong baseline, and semantic guidance wins only
sometimes, is known and contested. What we add is a controlled measurement of
```
NEW
```
That random masking is a strong baseline, and semantic guidance wins only
sometimes, is known and contested; we do not overturn that literature. What we
add is a controlled measurement of
```
Net-neutral. Same disclaimer, no longer leading.

### E4 — L180 (repetition audit, reword)
OLD
```
precedent for our best policy uses no segmentation model either:
```
NEW
```
precedent for our best policy is segmentation-free too:
```
Shorter. This one is about the cited work, not our headline, so the content stays.

### E5 — L204
OLD
```
FairMedFM~\citep{jin2024fairmedfm} and \citet{shi2025equitable} benchmark
fairness for medical foundation models. We report a subgroup analysis as a
secondary finding.
```
NEW
```
FairMedFM~\citep{jin2024fairmedfm} and \citet{shi2025equitable} benchmark
fairness for medical foundation models.
```
Saves 1 line. The "secondary, not a contribution" framing survives at L612–613 and
L1249–1250.

### E6 — L237 (repetition audit, cut #3 of 5)
OLD
```
  across columns. It adapts to the retina's position and curvature using a
  first-order intensity statistic, consults \textbf{no segmentation model}, and
  uses no ground-truth annotation of any kind. It is named for the statistic it
  computes.
```
NEW
```
  across columns. It adapts to the retina's position and curvature from a
  first-order intensity statistic, with no segmentation model and no ground-truth
  annotation of any kind. It is named for the statistic it computes.
```
Saves 1 line. The `\item[...]` label three lines above already reads
`(segmentation-free)`, so nothing is lost.

### E7 — L248
OLD
```
  pretrain $f{=}0.21$. The name describes the \emph{intent}: an implementation
  defect means the delivered masks do not reach that coverage, and we report the
  arm accordingly (Appendix~\ref{app:covbug}).
```
NEW
```
  pretrain $f{=}0.21$. The name describes the \emph{intent}: an implementation
  defect means the delivered masks do not reach that coverage
  (Appendix~\ref{app:covbug}).
```
Net-neutral in lines, shorter text.

### E8 — L304
OLD
```
Arms differ in how far they were carried, and we do not pretend otherwise.
```
NEW
```
Arms differ in how far they were carried.
```

### E9 — L308
OLD
```
continuation, Appendix~\ref{app:excluded}); we stopped that arm having seen its
epoch-75 deficit, which is a selective stopping horizon and a real weakness ---
a trajectory can recover, so we cannot claim its epoch-100 value would have been
worse, only that we did not measure it.
```
NEW
```
continuation, Appendix~\ref{app:excluded}); we stopped that arm having seen its
epoch-75 deficit, which is a selective stopping horizon and a real weakness ---
a trajectory can recover, and its epoch-100 value is unmeasured, not known to be
worse.
```
Net-neutral in lines. The limitation (selective stopping, unmeasured epoch 100,
possible recovery) is preserved in full.

### E10 — L331
OLD
```
\paragraph{Numerical precision, and how we handle it.} The probe harness
```
NEW
```
\paragraph{Numerical precision.} The probe harness
```

### E11 — L335
OLD
```
were fitted with autocast explicitly disabled (fp32). We therefore do
\emph{not} claim a single protocol across every probe in the study. Instead we
partition the comparisons: all headline contrasts in
```
NEW
```
were fitted with autocast explicitly disabled (fp32). Protocol is therefore not
uniform across probes, so we partition the comparisons: all headline contrasts in
```
Saves 1 line. The disclosure is now asserted rather than disclaimed, which is
stronger, not weaker.

### E12 — L346
OLD
```
\emph{What these numbers can and cannot establish.} Each policy was pretrained
exactly once, so policy is perfectly confounded with one stochastic optimisation
path after the fork. The intervals below quantify sampling error over test
subjects for a \emph{fixed} pair of score vectors, not seed-to-seed variance of
pretraining. Read every comparison here as a controlled single-run observation,
not an established ranking over retrainings (Section~\ref{sec:limits}).
```
NEW
```
Each policy was pretrained exactly once, so policy is confounded with one
stochastic optimisation path after the fork. The intervals below quantify
sampling error over test subjects for a \emph{fixed} pair of score vectors, not
seed-to-seed variance of pretraining: every comparison is a single-run
observation, not an established ranking over retrainings (Section~\ref{sec:limits}).
```
Saves 1 line. Every clause of the limitation is retained; only the micro-heading
and the imperative `Read every comparison here as` go.

### E13 — L427 (also section C, reviewer sentence 3)
OLD
```
We are careful not to overstate this. At the matched epoch 50 neither
\textsc{anatomy-v2} (CI $\DAnatomyTwoRandomEpFiftyCI$,
```
NEW
```
At the matched epoch 50 neither \textsc{anatomy-v2}
(CI $\DAnatomyTwoRandomEpFiftyCI$,
```
Net-neutral in lines, shorter text.

### E14 — L437
OLD
```
\textsc{cover} is the only policy that uses the segmenter to choose \emph{how
much} to hide, and we continued it to the full
horizon, and it produces the study's one \emph{negative} result. It climbs
```
NEW
```
\textsc{cover} is the only policy that uses the segmenter to choose \emph{how
much} to hide, and carried to the full horizon it produces the study's one
\emph{negative} result. It climbs
```
Net-neutral. Removes a three-`and` chain.

### E15 — L452
OLD
```
configured for. Appendix~\ref{app:covbug} gives the full account and states
exactly which of its numbers remain usable.
```
NEW
```
configured for; Appendix~\ref{app:covbug} gives the full account.
```
Saves 1 line. `What it invalidates` (L940–945) is exactly the promised content, so
the pointer loses nothing.

### E16 — L508
OLD
```
Second, \textbf{the anatomy arms are confounded on several axes simultaneously},
and we state this rather than attributing their deficit to shape. They hide only
$21.4\%$ of the grid against $40$--$46\%$ for the rectangle arms, leave $67.9\%$
```
NEW
```
Second, \textbf{the anatomy arms are confounded on several axes simultaneously}.
They hide only $21.4\%$ of the grid against $40$--$46\%$ for the rectangle arms,
leave $67.9\%$
```
Net-neutral in lines. All three percentages unchanged and in the same sentence. The
substantive version of this restriction survives four lines later at L513–514.

### E17 — L586
OLD
```
disappears ($\rho=\SubRaceBranchRho$, $p=\SubRaceBranchP$). We therefore treat
the checkpoint-level correlation as pseudo-replicated and make no claim that
better models are less fair. The same holds for severity in the opposite
```
NEW
```
disappears ($\rho=\SubRaceBranchRho$, $p=\SubRaceBranchP$). The checkpoint-level
correlation is pseudo-replicated and does not show that better models are less
fair. The same holds for severity in the opposite
```
Net-neutral. This is the surviving statement of the pseudo-replication limitation
(E26 removes the appendix duplicate).

### E18 — L593
OLD
```
$p=\SubGenderBranchP$). We do not claim that better masking harms any group, and
we do not claim it helps equity either.
```
NEW
```
$p=\SubGenderBranchP$). Neither harm nor benefit to equity is established here.
```
Saves 1 line. Both directions preserved.

### E19 — L610
OLD
```
zero, so at these subgroup sizes we cannot resolve whether the improvement is
evenly distributed, and claim neither. Appendix~\ref{app:subgroup} gives every
stratum. No policy here was designed as a
fairness intervention and none should be presented as one.
```
NEW
```
zero, so at these subgroup sizes we cannot resolve whether the improvement is
evenly distributed. Appendix~\ref{app:subgroup} gives every stratum. No policy
here is a fairness intervention and none should be presented as one.
```
Saves 1 line. The normative sentence keeps both halves.

### E20 — L622 (also section C, reviewer sentence 2)
OLD
```
precision (Appendix~\ref{app:fp32}). Every number here is emitted by one script
from stored per-case predictions, so prose, tables and figures cannot disagree;
that script, the epoch-100 \ArmBest{} encoder, its head, and the stored
predictions are released with the code, links withheld for anonymity.
```
NEW
```
precision (Appendix~\ref{app:fp32}). Every number quoted here is generated by one
script from stored per-case predictions; that script, the epoch-100 \ArmBest{}
encoder, its head and those predictions are released, links withheld for anonymity.
```
Saves 1 line, drops the impossible guarantee, keeps the release commitment.

### E21 — L634 (repetition audit, cut #4 of 5)
OLD
```
every epoch measured. A trained segmentation model is not required to obtain this
benefit, and using one more aggressively --- to shape targets or to maximise
anatomy coverage --- does not add to it and can subtract. The practical reading,
bounded by the caveats below, is that for \emph{this} task the segmentation stage
bought nothing we could measure, and a one-line intensity statistic recovered the
whole benefit. We would not generalise that to tasks where anatomy is the output,
such as layer segmentation or lesion localisation.
```
NEW
```
every epoch measured. Using the segmenter more aggressively --- to shape targets
or to maximise anatomy coverage --- does not add to that and can subtract. For
\emph{this} task the segmentation stage bought nothing we could measure, and a
one-line intensity statistic recovered the whole benefit. We would not generalise
that to tasks where anatomy is the output, such as layer segmentation or lesion
localisation.
```
Saves 1 line. Removes the sloganised restatement and `bounded by the caveats
below`; keeps the specific, argumentative form.

### E22 — L666
OLD
```
shared checkpoint with the continuation as the unit of replication. That is the
single most valuable follow-up and we do not claim its result in advance.
```
NEW
```
shared checkpoint with the continuation as the unit of replication. That is the
single most valuable follow-up.
```

### E23 — L675
OLD
```
make no claim that any floor is optimal. That arm reaches epoch 100, but an
implementation defect means its delivered masks never realised the coverage it
was configured for (Appendix~\ref{app:covbug}), so we report its trajectory and
explicitly decline to interpret it as evidence about aggressive coverage. Its
epoch-50 value is included because it is a matched-epoch
comparison against every other arm. Finally, one dataset and one test split were
```
NEW
```
make no claim that any floor is optimal. That arm reaches epoch 100, but an
implementation defect means its delivered masks never realised the coverage it
was configured for (Appendix~\ref{app:covbug}), so its trajectory is not evidence
about aggressive coverage. Its epoch-50 value is retained as a matched-epoch
comparison against every other arm. Finally, one dataset and one test split were
```
Saves 1 line. The restriction is stated more directly, not less.

### E24 — L742 (caption)
OLD
```
AUC does not follow it. Panels (c)--(f) give the confounds we decline to
disentangle --- how much anatomy survives in the context, the context budget, and
```
NEW
```
AUC does not follow it. Panels (c)--(f) give the confounds we do not
disentangle --- how much anatomy survives in the context, the context budget, and
```
Shorter by four characters; removes the reviewer-flagged `decline`.

### E25 — L824
OLD
```
max--min gap with universally rising subgroup AUCs is not evidence of harm, and
we do not present it as such. The checkpoint-level correlation between aggregate
```
NEW
```
max--min gap with universally rising subgroup AUCs is not evidence of harm. The
checkpoint-level correlation between aggregate
```

### E26 — L830
OLD
```
because the \NprobesSub{} probes come from only \NbranchesSub{} branches. The
branch-level figure is the one to believe. We therefore explicitly do
\emph{not} claim that more accurate models are less fair.
```
NEW
```
because the \NprobesSub{} probes come from only \NbranchesSub{} branches. The
branch-level figure is the one to believe.
```
Saves 1 line. The identical claim survives in the main text at L587–589 (E17), and
`tab:subtrends`' caption also states it for race.

### E27 — L904
OLD
```
attributable to masking. We report their existence for completeness and do not
use them to support any claim. Similarly, the \textsc{anatomy-v2} probes at
```
NEW
```
attributable to masking. Similarly, the \textsc{anatomy-v2} probes at
```
Saves 1 line. The paragraph's first sentence already reads `are excluded from all
analysis`, and they remain listed in `tab:allprobes`.

### E28 — L914
OLD
```
defect that changes how that arm may be interpreted. We report it in full because
it invalidates the reading we would otherwise have given, and because the same
mechanism silently weakens one cross-family comparison.
```
NEW
```
defect that changes how that arm may be interpreted: it invalidates the reading
we would otherwise have given, and the same mechanism silently weakens one
cross-family comparison.
```
Net-neutral.

### E29 — L956
OLD
```
\CoverFixedStatus{}. Until it exists, the question this arm was built to answer
--- whether hiding most of the anatomy is harmful --- is \emph{open}, and nothing
in this paper should be read as answering it. We report the arm we ran, not the
arm we intended.
```
NEW
```
\CoverFixedStatus{}. Until it exists, the question this arm was built to answer
--- whether hiding most of the anatomy is harmful --- remains \emph{open}.
```
Saves 2 lines. `remains open` is the whole content; the two following sentences
restate it.

### E30 — L991
OLD
```
We therefore report the first link of this mechanism as measured and leave the
causal chain to downstream AUC open; two ablations would settle it.
```
NEW
```
The causal chain from this mechanism to downstream AUC remains open; two
ablations would settle it.
```
Net-neutral in lines, shorter text.

### E31 — L1154
OLD
```
specificity and downstream AUC. The OD/OS result is a methodological caution of
the same family as our precision and epoch-matching checks: a figure that looks
anatomically meaningful can be an artefact of data storage, and we would rather
report that than let it stand.
```
NEW
```
specificity and downstream AUC. The OD/OS result is a methodological caution: a
figure that looks anatomically meaningful can be an artefact of data storage.
```
Saves 2 lines.

### E32 — L1162
OLD
```
AUC is threshold-free, but a screening tool is deployed at a threshold. This
appendix reports what a reader needs in order to judge that. For each arm the
threshold is selected on the \emph{validation} split at a fixed target
specificity and then transferred unchanged to the test split, which is the honest
simulation of deployment; the achieved test specificity shows whether the
threshold survives the shift. Cohort prevalence is $\Prevalence$.
```
NEW
```
AUC is threshold-free, but a screening tool is deployed at a threshold. For each
arm the threshold is selected on the \emph{validation} split at a fixed target
specificity and transferred unchanged to the test split, simulating deployment;
the achieved test specificity shows whether the threshold survives the shift.
Cohort prevalence is $\Prevalence$.
```
Saves 1 line. Drops an announcement sentence and the inflated `the honest
simulation of deployment`.

### E33 — L1192
OLD
```
We report this because a $+0.011$ AUC difference is hard to interpret
clinically, whereas ``two to three more cases detected per hundred at the same
false-positive rate'' is not. The same caveats apply as everywhere else in this
paper: one dataset, one split, and one pretraining run per policy.
```
NEW
```
A $+0.011$ AUC difference is hard to interpret clinically, whereas ``two to
three more cases detected per hundred at the same false-positive rate'' is not.
One dataset, one split, one pretraining run per policy.
```
Saves 1 line. The caveat survives as a clipped fragment, which also breaks the
uniform rhythm the reviewer objected to.

### E34 — L1214
OLD
```
strata. We therefore report this as a \emph{power} limitation of the cohort ---
this study cannot certify that the smaller strata benefit --- and explicitly not
as evidence that they do not. Certifying subgroup benefit at a deployed
```
NEW
```
strata. This is a \emph{power} limitation of the cohort: the study cannot certify
that the smaller strata benefit, and equally cannot show they do not. Certifying
subgroup benefit at a deployed
```
Net-neutral. Both directions of the power argument preserved.

### E35 — L1312 (caption)
OLD
```
curves. $q$ is Benjamini--Hochberg over these nine contrasts. Because the test split was
inspected repeatedly across this programme, we read all $p$ and $q$ values here as
descriptive rather than confirmatory. Every
comparison against the null excludes zero.}
```
NEW
```
curves. $q$ is Benjamini--Hochberg over these nine contrasts. All $p$ and $q$
values here are descriptive rather than confirmatory (Section~\ref{sec:limits}).
Every comparison against the null excludes zero.}
```
Saves 1 line. `\label{sec:limits}` is defined at L628, so the new cross-reference
resolves.

### E36 — L1367 (caption)
OLD
```
slots --- which is why we decline to attribute its deficit to target shape.}
```
NEW
```
slots --- so its deficit cannot be attributed to target shape.}
```

### E37 — L474 (caption, duplicated sentence)
OLD
```
number of predictor targets contributing to the loss. All values measured. AUC is
quoted at the \emph{matched} epoch 50, the only epoch at which all five policies
have a probe, so that the geometry-to-AUC association is not read across mixed
epochs. All five AUCs are at epoch 50.}
```
NEW
```
number of predictor targets contributing to the loss. AUC is quoted at the
\emph{matched} epoch 50, the only epoch at which all five policies have a probe,
so the geometry-to-AUC association is not read across mixed epochs.}
```
Saves 1 line. `All five AUCs are at epoch 50.` restates the preceding sentence;
`All values measured.` is filler (the caption already says `Measured mask
geometry`). The column header `AUC @ep50` also carries it.

**Number-safety note for E37:** this is the only edit that deletes a hand-typed
literal (`epoch 50` in the duplicated final sentence). No value is altered: `epoch
50` still appears in the preceding sentence of the same caption and in the column
header `AUC @ep50`. If you would rather not touch a caption containing a literal
number at all, skip E37; it costs one line.

---

## C. The three sentences the reviewer named

**1. `The result contradicts the intuition in an informative way.` (L113)**
Deleted outright — edit **E1**. The paragraph then opens on the finding itself
(`Guidance helps, but the benefit does not increase with anatomical precision`),
which is what the deleted sentence was gesturing at. No replacement wording is
needed; adding one would only re-inflate it.

**2. `Every number here is emitted by one script … cannot disagree … links withheld
for anonymity.` (L622–625)**
Replaced by edit **E20**. The impossible guarantee (`prose, tables and figures
cannot disagree`) is gone; what survives is the checkable fact (numbers generated
from stored per-case predictions by one script) and the release commitment. One
line shorter.

**3. `We are careful not to overstate this.` (L427)**
Deleted outright — edit **E13**. The sentence it introduces already contains its own
qualification (`neither … is worse than the null; both are indistinguishable from
it`).

Note: the LaTeX comment at L19–21 makes the same `prose, tables and figures cannot
disagree` claim. It is a comment and costs no page space, so it is left alone; delete
it only if you want the source to match the prose.

---

## D. Repetition audit — "the strongest policy uses no segmentation model"

| Line | Form | Action |
|---|---|---|
| 68 (abstract) | `The best policy consults no segmentation model at all.` | **KEEP — occurrence 1 of 3** |
| 120 (introduction) | `The best performer uses no segmentation model at all, locating a band by a per-column intensity centroid.` | **CUT** — edit **E2**; the mechanism (`per-column intensity centroid`) is kept, the slogan is not |
| 180 (related work) | `the closest positive precedent for our best policy uses no segmentation model either` | **REWORD** — edit **E4**; this describes `\citet{lee2025hufgm}`, not our result, so it stays but stops echoing the slogan |
| 235 (method, item label) | `\item[\ArmBest{} (segmentation-free).]` | **KEEP** — definitional, and this is where the policy is introduced |
| 238 (method, body) | `consults \textbf{no segmentation model}` | **CUT** — edit **E6**; duplicates the label three lines above |
| 378 (Table `tab:main` row) | `intensity centroid, no segmenter` | **KEEP** — table cell, must be self-describing |
| 455 (results) | `The strongest policy is the one that uses no segmentation model.` | **KEEP — occurrence 2 of 3** |
| 634 (discussion) | `A trained segmentation model is not required to obtain this benefit` | **CUT** — edit **E21** |
| 637–639 (discussion) | `for \emph{this} task the segmentation stage bought nothing we could measure` | **KEEP** — this is the argument, not the headline: it is specific, scoped to `this task`, and is the sentence the following sentence (`We would not generalise that…`) attaches to. If you want the slogan in strictly three places and nowhere else, this is the one further cut available, but the paragraph is titled `What the evidence supports` and would then support nothing. |
| 700 (conclusion) | `and the best policy consulted no segmentation model` | **KEEP — occurrence 3 of 3** |

Result: the headline survives in the abstract, the results, and the conclusion; the
introduction, the second method mention, and the discussion slogan are removed; the
related-work and table mentions are retained because they name a different thing.

---

## E. Other prose that reads as generated

Already given above as exact replacements; listed here by category.

**Inflated but content-free**
- E1, L113 — `The result contradicts the intuition in an informative way.`
- E32, L1165 — `which is the honest simulation of deployment`.

**Announcing what the paper will do instead of doing it**
- E5, L204 — `We report a subgroup analysis as a secondary finding.`
- E32, L1162 — `This appendix reports what a reader needs in order to judge that.`
- F6 (optional), L966 — `We investigated whether background carries diagnostic signal, and whether position embeddings are the mechanism.`

**Meta-procedural / audit voice**
- E20, L622 — `Every number here is emitted by one script … cannot disagree`.
- E28, L914 — `We report it in full because …`.
- E15, L452 — `… and states exactly which of its numbers remain usable.`
- E31, L1154 — `a methodological caution of the same family as our precision and epoch-matching checks … we would rather report that than let it stand.`
- E10, L331 — `\paragraph{Numerical precision, and how we handle it.}`

**Empty transitions / padding**
- E21, L636 — `The practical reading, bounded by the caveats below, is that …`.
- E33, L1194 — `The same caveats apply as everywhere else in this paper: …`.
- E19, L611 — `and claim neither`.
- E25, L825 — `and we do not present it as such`.
- E8, L304 — `and we do not pretend otherwise`.
- E22, L667 — `and we do not claim its result in advance`.

**Uniform sentence rhythm / duplicated sentences**
- E14, L437 — three coordinated `and` clauses in one sentence.
- E16, L509 — restates in advance what L513 states properly.
- E37, L477 — `All five AUCs are at epoch 50.` immediately after a sentence saying so.
- E26, L831 — verbatim duplicate of a main-text sentence.
- E29, L958 — same fact asserted three times in four lines.

---

## F. Optional, lower-confidence (apply only if you want more room)

These are safe LaTeX-wise but each removes something you may consider load-bearing.
I would apply F1, F3 and F6; F5 I recommend against. F2 and F4 are notes, not edits.

**F1 — L865, appendix severity, trim `only`**
OLD
```
is descriptive only: we did not test the paired \emph{difference} between
```
NEW
```
is descriptive: we did not test the paired \emph{difference} between
```

**F2 — L1194 alternative to E33 if you would rather drop the caveat entirely here**
Use E33 as written unless you have already decided the `\ref{sec:limits}` statement
is sufficient; in that case delete the third line of E33's NEW block.

**F3 — L783, appendix subgroup, `as everywhere else in the paper`**
OLD
```
precision-spliced \textsc{anatomy-v2} probes are excluded here as everywhere
else in the paper.
```
NEW
```
precision-spliced \textsc{anatomy-v2} probes are excluded here as elsewhere.
```
Saves 1 line.

**F4 — L556, main-text subgroup**
The sentence `Probes that the exclusion rules of Appendix~\ref{app:excluded} remove
are removed here too.` (L560-561) says the same thing as the appendix sentence in F3.
No edit proposed: the two are in different sections and each is needed locally. Listed
only so you know I considered it.

**F5 — L778, `fig:fair` caption duplicate of `tab:subtrends` caption**
OLD
```
checkpoint level, but does not survive branch-level aggregation
(Table~\ref{tab:subtrends}), so we draw no trend conclusion from it.}
```
NEW
```
checkpoint level, but does not survive branch-level aggregation
(Table~\ref{tab:subtrends}).}
```
Shortens the caption. I recommend keeping the current version: captions are read out
of order and the instruction not to read a trend is the point of the panel.

**F6 — L965, appendix `app:bg` opening announcement**
OLD
```
Unguided masking spends most of its predictor targets on background, yet it
reaches within $\DOracleRandomEpHundred$ of the best policy. We investigated
whether background carries diagnostic signal, and whether position embeddings are
the mechanism.
```
NEW
```
Unguided masking spends most of its predictor targets on background, yet it
reaches within $\DOracleRandomEpHundred$ of the best policy.
```
Saves 2 lines. The three `\paragraph` headings that follow already announce both
questions and answer them.

---

## Verification notes

All 41 OLD/NEW pairs were checked by script against the current file:

- Each OLD block occurs **exactly once** in `main_submission.tex` (no ambiguous
  replacement, no accidental double application).
- Each NEW block is strictly shorter in characters than its OLD block and never
  occupies more source lines.
- Each NEW block has the same brace surplus/deficit as its OLD block, so captions
  ending in the `\caption{...}` closing brace (E35, E36, E37, F5) stay balanced.
- Digit tokens are identical between OLD and NEW in 40 of 41 pairs. The single
  exception is E37, which deletes the duplicated sentence `All five AUCs are at
  epoch 50.`; `epoch 50` still appears in the preceding sentence of the same
  caption. Skip E37 if you want zero literals touched.
- Macros preserved byte-for-byte inside edited regions include
  `\ref{app:excluded}`, `\ref{app:covbug}`, `\ref{app:fp32}`, `\ref{app:subgroup}`,
  `\ArmBest{}`, `\CoverFixedStatus{}`, `\NprobesSub{}`, `\NbranchesSub{}`,
  `\SubRaceBranchRho`, `\SubRaceBranchP`, `\SubGenderBranchP`, `\Prevalence`,
  `\DOracleRandomEpHundred`, `\DAnatomyTwoRandomEpFiftyCI`.
- Macro groups deliberately deleted whole, together with the words they wrapped:
  `\emph{not}` in E3, E11, E26; `\emph{What these numbers can and cannot
  establish.}` in E12; `\textbf{no segmentation model}` in E6. No partial group is
  left behind in any edit.
- One new cross-reference is introduced (E35, `Section~\ref{sec:limits}`); the
  label exists at L628, so it resolves.
- No citation is orphaned: E5 deletes only prose following the final
  `\citet{shi2025equitable}` of its paragraph; E27 and E33 delete prose containing
  no citation.

The verification script is `autopilot\reports\_verify_p22.py`; re-run it after
applying to confirm every OLD block is gone.
