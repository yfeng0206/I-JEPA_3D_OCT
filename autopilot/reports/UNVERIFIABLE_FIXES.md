# Unverifiable-number fixes

Source spec: `session-state/.../files/LITERALS_PASS2.md`, "UNVERIFIABLE" section (15 entries).
Target: `paper\genai4health2026\main_submission.tex`.

Scope: reading and editing only. No training, probe, or GPU job was run.

Rule applied: remove the unbackable quantity, keep the qualitative finding. No number was
invented or estimated. Every replacement is the same length or shorter. No `\label`, `\ref`,
or macro definition was touched.

## Summary

- Reworded: 11 entries (U3--U15, where U10/U11 and U14/U15 share one edit each).
- Already absent, confirmed and left alone: 2 (U1, U2).
- Distinct edits made: 9.

## Entries

### U1 -- per-cell hit-map Gini (`0.400`, `0.316`--`0.338`)
**Status:** already absent. Confirmed by search: no occurrence of `Gini`, `0.400`, `0.316`,
or `0.338` anywhere in the file. Not restored.

### U2 -- five-probe-seed SDs (`0.0003`, `0.0018`)
**Status:** already absent as a probe-noise bound. Confirmed: `0.0018` does not occur in the
file. The surviving `0.0003` at line ~1191 is the label-efficiency protocol-parity agreement
between two stored protocols, a different claim, and was left in place. Not restored.

### U3 -- COVER pre/post-truncation coverage (`78.6%`, `73.9%`)
**Was:** "On training slices, \textsc{cover}'s placement hides $78.6\%$ of anatomy mass, close
to its configured target, but the masks actually delivered hide $73.9\%$; an independent sweep
over $6{,}137$ slices gives $73.1\%$ against $77.6\%$ for \textsc{envelope}."

**Now:** "A one-off audit found that collation reduced \textsc{cover}'s anatomy coverage; an
independent sweep over $6{,}137$ slices gives $73.1\%$ against $77.6\%$ for \textsc{envelope}."

The collation defect and the direction of the effect are kept; the stored floor-sweep pair
`73.1%`/`77.6%` is retained because it is artifact-backed.

### U4 -- retained-target statistics (`32.5%`, `2.51`)
**Was:** "Only $32.5\%$ of images retain all four rectangular targets, the mean being $2.51$ of
$4$, and across $24{,}000$ emitted targets only $73.4\%$ remain perfect rectangles."

**Now:** "Across $24{,}000$ emitted targets only $73.4\%$ remain perfect rectangles."

`24,000` and `73.4%` are backed by `D:\jepa_phase0\reports\cover_random_scale\scale_validation.json`.

### U4b -- provenance sentence (dependent wording repair, not a new entry)
The following provenance sentence still claimed the paragraph contained the now-removed
figures, so it was made accurate and shorter.

**Was:** "the pre- and post-truncation coverage figures and the retained-target statistics in
this paragraph come from a one-off CPU audit of 194 accepted slices whose raw output was not
persisted, so they cannot be recomputed from the released artifacts."

**Now:** "the one-off CPU audit of 194 accepted slices that first exposed the defect was not
persisted, so its pre- and post-truncation figures are no longer asserted."

### U5 -- superseded full-batch null AUC (`0.8811`)
**Was:** "An earlier version of this experiment used a cheaper full-batch fit, which put the
null at $0.8811$ at full supervision against $\AUCRandomEpHundred$ in the primary protocol and
so made the two tables disagree."

**Now:** "An earlier cheaper full-batch fit disagreed with the primary protocol."

### U6 -- slice-level cross-probe correlation in `fig:interp-slice` caption (`r=0.94`)
**Was:** "...they converge on the same structure (mean-pool vs cross-attention pool $r{=}0.94$)."

**Now:** "...they converge on the same structure."

### U7 -- window/single-slice amplification (`25x`)
**Was:** "the windowed variant recovers about $25\times$ more."

**Now:** "the windowed variant recovers more signal."

The separate "roughly sevenfold" in the same caption follows from $W{=}7$ and was not in scope.

### U8 -- population peak/dip positions (`63`, `137`, `95`)
**Was:** "The population-averaged curve peaks at native slice $\approx 63$ and $\approx 137$
with a dip near $95$."

**Now:** "The population-averaged curve is bimodal with a central dip."

The later reference to "the two peaks" still reads correctly against "bimodal".

### U9 -- per-volume peak correlations (`-0.22`, `-0.07`, `-0.14`)
**Was:** "...; the observed correlation is slightly \emph{negative} ($-0.22$, $-0.07$, $-0.14$
across the three probes)."

**Now:** "...; the peak contributions are not positively correlated across volumes."

The test and its conclusion survive; only the magnitudes are gone.

### U10 / U11 -- flipped and raw cluster correlations (`0.971`, `0.988`, `-0.124`, `-0.478`)
Single edit, as the spec directs.

**Was:** "near-perfect \emph{mirror images} along the slice axis
($\mathrm{corr}(c_1,\mathrm{flip}(c_2)) = 0.971$ and $0.988$ for the two well-structured
probes, against raw correlations of $-0.124$ and $-0.478$)."

**Now:** "near-perfect \emph{mirror images} along the slice axis for the two well-structured
probes."

### U12 -- confidence/attribution correlation bound (`|r| <= 0.25`)
**Was:** "Attribution structure is also near-invariant to prediction confidence ($|r| \leq
0.25$ between peak contribution magnitude and $|\text{logit}|$)."

**Now:** "Attribution structure is also near-invariant to prediction confidence."

### U13 -- patch-significance range and CI level (`84%`, `91%`, `95%`, `B=500`)
**Was:** "Between $84\%$ and $91\%$ of patches have a $95\%$ bootstrap interval excluding zero
on the glaucoma class ($B{=}500$), so attribution is broadly distributed rather than
concentrated in a few patches."

**Now:** "Attribution is broadly distributed rather than concentrated in a few patches."

### U14 / U15 -- caption cross-probe correlations (`r=0.94`; `0.35`--`0.48`)
Single edit, as the spec directs.

**Was:** "Cross-probe agreement is strong at slice level ($r{=}0.94$) and moderate at patch
level (per-volume $r \approx 0.35$--$0.48$)."

**Now:** "Agreement is stronger at slice than patch level."

## Not reverted

The nine WRONG corrections already applied by the user before this pass were checked and left
untouched: Table 2's geometry rows (lines 476--480), the Section 5.2 purity contrast
(`97.1%` / `40.0%`, lines 497--498), the envelope purity `43.3%` (line 388), the geometry
ranges (lines 503--506), and the `fig:paradox` caption `40%--43%` (line 1584).

## Verification

Run from `C:\Users\Gary\Desktop\jepa`, after the edits above.

| Gate | Result |
|---|---|
| `autopilot\p13_build_zip.py` | 6/6 PASS; main content 9 pages (limit 9); total 32; `ALL_PASS = True` |
| `autopilot\check_manuscript.py` | `RESULT: PASS` (0 hard failures; 3 pre-existing warnings) |
| `autopilot\p15_verify_numbers.py` | `RESULT: PASS` (20 AUC macros verified) |

`p13_build_zip.py` was run without `--mark-uploaded` and did not throw.

No commit was made.
