# Corrections to the inherited context

Author-side audit working draft, 2026-09-04. These are corrections to earlier
assistant assurances, not allegations about the research.

## 1. An AI review is not a workshop outcome

`autopilot/reports/R11_review.md:65-69` records an internal generated review,
including a weak-accept recommendation. No actual OpenReview submission receipt,
review, acceptance, or rejection was inspected. The version board must report
the paper's technical state separately from submission and editorial status.
Earlier statements such as "paper stands at 4/6 Weak Accept" blur this boundary.

## 2. Public repository visibility is not automatically a desk rejection

The live repository is PUBLIC. The workshop prohibits identifying information
and non-anonymized links in the submission and says violations may lead to desk
rejection. It also explicitly allows non-archival preprints. The main-track
handbook additionally states that a non-anonymous preprint's existence alone
does not result in rejection. Thus, "a public repo is a desk-reject trigger" is
too categorical without establishing a prohibited link, identifying submission
content, or other applicable violation. Public distribution of an anonymous
submission and the appropriateness of its preprint formatting still deserve
care, but no repository-visibility change is authorized by this audit.

Sources read 2026-09-04:

- https://genai4health.github.io/2026-NeurIPS/ (Review Process; Publication Policy)
- https://neurips.cc/Conferences/2026/MainTrackHandbook (Preprints; Double-blind)

## 3. Citation checking is not the only author responsibility

The handbook's author-LLM section discusses hallucinated references as an example,
not the sole sanctionable behavior. It also requires correct, original content
and prohibits review manipulation; the handbook separately addresses plagiarism,
dual submission and other requirements. An existence/title check does not establish
whether a source supports the sentence citing it, and cannot "close" scientific
integrity as a whole.

Ordinary editing assistance does not require documentation under the main-track
handbook. That is not a blanket exemption for methodological LLM use, and the
workshop's own rules must be distinguished from main-track requirements.

## 4. The 66-em-dash comparison used the wrong document scope

Recomputed from the parent of `d5ee750` using `git show`:

| Scope in the old main_submission.tex | Literal LaTeX em dashes |
|---|---:|
| Complete source including appendices | 66 |
| Before bibliography | 25 |
| Remainder | 41 |

The before-bibliography count also includes table-cell punctuation. Comparing
all 66 to an assumed nine-page-body word count inflated the claimed rate.
`VENUE_STYLE_RESEARCH.md:95-141` itself describes a different denominator
(all source including appendices), gives a landmark maximum above the recent
sample maximum, and says its contemporaneous file did not reproduce 66.

The user's preference for fewer dashes and no emoji remains valid. Punctuation
counts are not an authorship detector, a venue requirement, or evidence that
a paper will receive a particular score. This audit does not undo the edits.

## 5. Documentation has multiple, inconsistent generations

`README.md:5-22` foregrounds fine-tuned AUC 0.8947. The current submission's main
comparison is frozen-probe AUC, including CENTROID 0.8855. These are different
regimes, not conflicting measurements.

More seriously, `docs/experiments/README.md:5-9` still calls target shape the
contribution and says an early comparison isolates shape. The current paper
describes confounds preventing that causal conclusion. `CHANGELOG.md:3-17`
still describes the archived semantic-teacher direction as active.

`HANDOFF.md:9-25` claims all numbers are generated and all current macros are
measurements, whereas the same file later documents an unrun corrected-COVER
slot. It is a generated historical inventory, not authority for those universal
claims.

## 6. A paused experiment is not running because its old report says so

`autopilot/reports/G1_REPLICATION.md:3-5` says RUNNING at its August 26 writing
time. The local process snapshot on September 4 found no matching JEPA training
process. Only `rep_random_s1234` exists among the bounded `rep_*` run directories.
Its latest log records restoring the epoch-26 checkpoint and starting epoch 27;
the last CSV rows are iterations 7 and 8 of that incomplete epoch.

This establishes no completed replication result in the inspected local
locations. It does not establish that no job exists on a remote machine.
No experiment was restarted or cancelled by this audit.

## 7. The date has advanced

The live workshop page still says September 5, 2026, 23:59 AoE. That is
September 6, 2026, 11:59 UTC / 04:59 PDT. The old "nine days remaining" language
is stale. The research-track limit is nine content pages; the page count is
visible in the raw HTML track badge. No rebuttal stage is provided.
