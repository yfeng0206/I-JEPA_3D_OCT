# Contradiction fixes

## 1. Checkpoint-level q attached to a branch-level claim

**Reviewer claim.** The subgroup discussion attached \(q=0.0038\), which was
computed per checkpoint, to a branch-level sex result.

**Artifact verification.** `auto/table_subgroup_trends.tex` separates per
checkpoint (\(n=23\)) from per branch (\(n=7\)). Its sex row places
\(q=0.0038\) only in the per-checkpoint columns; the branch columns contain
\(\rho=-0.821\) and \(p=0.0234\), with no branch-level q. The same values appear
as `SubGenderQ`, `SubGenderBranchRho`, and `SubGenderBranchP` in
`auto/auto_numbers.tex`.

**Finding and change.** Verified real. The body parenthetical read
“branch-level \(\rho=\ldots\), \(q=\ldots\),” which incorrectly made the
checkpoint q look branch-level. It now explicitly reports branch-level rho and
p, followed by checkpoint-level q. No digit changed.

## 2. Race described as passing correction

**Reviewer claim.** Appendix E said the race trend passed
Benjamini--Hochberg correction even though its \(q=0.0668\) exceeds the table
caption's conventional 0.05 criterion.

**Artifact verification.** `auto/table_subgroup_trends.tex` gives race
\(p=0.0225\), \(q=0.0668\) per checkpoint and branch-level \(p=0.4821\).
`auto/auto_numbers.tex` agrees. The Table 6 caption states that only sex
survives correction at 0.05 and that race, ethnicity, and disease severity
share the q just above that level.

**Finding and change.** Verified real. The appendix sentence saying race “It
passes Benjamini--Hochberg” now says it does not pass correction at the stated
conventional threshold. The q value and all caveats about pseudo-replication
and branch aggregation remain unchanged.

## 3. Full-supervision agreement bounds

**Reviewer claim.** The body gives agreement within 0.0009 while the
label-efficiency appendix gives agreement within 0.0003, making two valid but
differently scoped statements look contradictory.

**Artifact verification.** Comparing the 100% row of
`auto/table_labeleff.tex` with the epoch-100 values represented by
`auto/auto_numbers.tex` gives absolute differences of 0.0002 for the null,
0.0001 for CENTROID, 0.0006 for ENVELOPE, and 0.0009 for COVER. Thus the
four-arm maximum is 0.0009, while the explicitly named null-and-CENTROID pair
is within 0.0003.

**Finding and change.** The numbers are both correct; this was not a numeric
error. The body now explicitly calls 0.0009 the bound across all four arms in
the sweep. The appendix explicitly restricts 0.0003 to the null and CENTROID
and says that this two-arm bound is distinct from the body’s four-arm bound.
Neither digit changed.
