# REFRAME: Results narrative rewrite

Target file: `paper/genai4health2026/main_submission.tex`
Build: `p13_build_zip.py` 6/6 PASS, 9 main-content pages, `ALL_PASS = True`.
`check_manuscript.py` RESULT: PASS (labels 53, refs 52, dangling 0).
`p15_verify_numbers.py` RESULT: PASS.

## 1. Number density, before and after

Counting rule is the acceptance probe: `(?<![A-Za-z])[0-9]+\.[0-9]+` per 100 words
of extracted page text.

| page | before | after | after, prose only | table cells on page |
|------|--------|-------|-------------------|---------------------|
| p5   | 4.4    | 5.1   | **0.8**           | 27 (Table 1)        |
| p6   | 9.1    | 2.2   | **2.0**           | 1                   |
| p7   | 8.5    | 6.9   | **1.9**           | 33 (Table 2)        |
| p8   | 5.6    | 2.3   | **2.2**           | 1                   |

(Front matter also fell: p1 3.7 -> 0.9.)

Before-column values are the ones supplied with the task, measured on the shipped
PDF. Rebuilding the as-found `.tex` first (the shipped PDF was one edit stale)
gave p5 4.4, p6 9.1, p7 7.1, p8 5.3; every "after" number below is measured
against that rebuild.

The stated target -- under about 3 per 100 -- is met in **running prose on every
body page**. It is not met by the raw per-page count on p5 and p7, and cannot be:
those two pages carry Table 1 and Table 2, whose cells contribute 27 and 33 of
the numbers counted there. A page holds at most about 660 words once a float that
size is placed, so 27/660 = 4.1 is the arithmetic floor for the Table 1 page even
with zero numbers in its prose. The defect the measurement was pointing at --
prose reading the tables aloud -- is gone: prose numbers across pages 5-8 fell
from 114 to 47, and page 6 went from 53 numbers in 583 words to 12 in 586.

Prose-vs-table split was measured by separating full-width extracted lines
(body text and captions) from one- and two-token lines (table cells).

## 2. Positive reframing: what changed in the story

Section 5 was a list of things that failed. It is now four things that matter and
one that does not, in the order the evidence supports.

**5.1 was "Guidance helps; anatomical precision does not reliably add"; it is now
"Region matters: where you aim beats not aiming at all".**
Leads with the cleanest control in the paper, stated as a control: `ENVELOPE`
differs from `RANDOM` in location alone -- same rectangle shapes, sizes and
counts, rejection-sampled onto the retinal envelope. The H2 paragraph is now
"Precision beyond location does not add, and the evidence is mixed rather than
uniformly negative", which is the same finding stated as a boundary rather than a
failure. The `CENTROID` paragraph is promoted to a bolded lead-in: the strongest
policy consults no segmentation model.

**5.2 was "Mask geometry does not explain the ordering"; it is now "Aim, not
coverage: what the mask leaves visible".**
First observation is now "anatomy tells you where to aim, not how much to cover",
carried by the honest reading of Table 2: of the four guided arms the best
performer hides the **least** anatomy, and the arm that hides the most, at 97.1%
purity, does not separate from the null. Second observation reframes context
retention as the live variable and keeps the confound accounting intact.

**5.3 is new: "Background matters -- for pretraining, not for the classifier".**
The headline of Appendix H ("Why unguided masking is such a strong baseline") is
now in the body: the null's predictor beats a per-position no-context reference
by 0.680 on background targets, above its 0.633 on anatomy; background
self-similarity falls from 0.784 untrained to 0.346; 90.8% of across-position
input variance at background cells comes from `pos_embed` against 40.8% at
tissue. Then the split: background is 95.2% linearly reconstructible from
anatomy, residualised test AUC 0.5515, and appending it lowers test AUC for the
null arm. Appendix H is still referenced from this section (and still holds the
full account, the background-only probe caveat, and the two unrun controls).

**Discussion and Conclusion** now open on the same structure -- region matters,
what is left visible matters, background matters for pretraining, coverage does
not -- instead of "guidance helped but precision did not".

**Abstract and Introduction** were realigned to the same story and de-densified
(p1 3.7 -> 0.9 per 100). Every abstract caveat was carried over verbatim in
substance: one continuation per policy, the epoch-50 match and epoch-75 fall for
the shaped arm, the coverage arm's decline, the collation audit, "the gains are
not separated from one another" for severity, and the race-subgroup hedge.

### Overclaiming guard
No sentence anywhere says or implies that hiding more anatomy is better. The
paper now states the opposite explicitly, twice, from Table 2: the winner hides
the least anatomy of any guided arm and the arm at 97.1% purity loses to the
null.

## 3. Numbers deleted from prose (all still in a table or appendix)

Deleted from Section 5 prose because the sentence was not arguing about that
specific value:

- 5.1: envelope@100 delta and CI; centroid@100 CI; envelope/centroid CIs at 50;
  `43.3%` purity; the four "gains X" restatements at epoch 50; anatomy-v2 and
  cover p-values; anatomy-v2/cover-vs-null CIs and p-values; cover peak AUC,
  epoch-100 AUC, the three gap-vs-null values and CI; `73.1%` / `77.6%` delivered
  anatomy; centroid's three-epoch margin sequence; envelope's three-epoch decay
  sequence; centroid@100 delta, CI, p and q in the "strongest policy" paragraph.
  All are in Table 1, Table 13 (Appendix N) or Figure 2b.
- 5.2: `21.3%`, `67.7%`, `41-45%` ranges. All in Table 2.
- 5.4: the two per-arm label-efficiency AUCs behind the 5% gap. In Appendix I.
- 5.5: black/asian point deltas; three correlation p-values; the five
  specificity-transfer values; moderate and severe severity deltas; both severity
  CIs; the aggregate AUC span restatement; the sex paired-difference pair. All in
  the severity table, the paired-subgroup table or the subgroup operating-point
  table (Appendices E and K), each of which the prose now points at directly.
- 5.6: the `0.8854754` vs `0.8854852` parenthetical. The `9.8e-6` agreement and
  the discordant-pair count remain.

## 4. Space accounting

Body was 9 pages and had to stay 9 while gaining a new subsection. Paid for by:
number deletions above; abstract trimmed; the display `quote` in the
Introduction inlined; Related work "informed masking is not uniformly better" and
"retinal foundation models" paragraphs merged and tightened (no citation
removed); `itemsep`/`parskip` zeroed on the contributions, policy and hypothesis
lists; Table 2 set `\footnotesize`; Figure 1 `0.66 -> 0.53\linewidth` and
Figure 2 `0.88 -> 0.78\linewidth`; Table 1, Table 2, Figure 1 and Figure 2
captions tightened.

## 5. Invariants held

- **No digit changed.** Verified mechanically: the only numeric literals in the
  file that do not appear in `git show HEAD:...main_submission.tex` or in
  `auto/auto_numbers.tex` are `0.53` and `0.78`, the two `\includegraphics`
  widths. Every number pulled into the body from Appendix H was string-matched
  against the appendix (0.680, 0.633, 0.784, 0.346, 90.8, 40.8, 95.2, 0.5515 --
  each occurs once in the body and once in the appendix).
- **No caveat, limitation or negative result removed.** Present and checked:
  the n=1-continuation-per-arm caveat and the replication PENDING block; probe
  seed variance unbounded and the lost multi-seed check; the confound accounting
  and "H3 is not identified by this design"; the COVER collation defect,
  "never realised the coverage it was configured for", and "its trajectory is not
  evidence about aggressive coverage"; the selective stopping horizon for
  anatomy-v2; repeated inspection of the test split and "descriptive rather than
  confirmatory"; "indistinguishable rather than worse" for anatomy-shaped
  masking; "we did not test" for the label-efficiency trend and for differential
  severity benefit; "not provably positive in every group" and "no policy here is
  a fairness intervention"; "the two controls that would settle it were not run";
  "what makes the best policy work is not identified here".
- **Labels and refs.** 53 labels / 52 refs / 0 dangling, up from 52 / 51 / 0 --
  one label and one reference added for the new Section 5.3. No label or
  reference was deleted; Appendix H is still referenced from the body.
- No emoji, tick or cross symbols.
- Nothing committed.
