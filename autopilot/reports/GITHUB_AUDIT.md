# GitHub audit, 2026-08-26

Two findings. The first is more important than anything else in this cleanup.

---

## FINDING 1 - ANONYMITY RISK. Needs an operator decision.

**[MEASURED]** The submission is double-blind. Its own Controls section states:

> "that script, the epoch-100 CENTROID encoder, its head and those predictions
> are released, **links withheld for anonymity**."

But:

| | |
|---|---|
| repository | `yfeng0206/I-JEPA_3D_OCT` |
| visibility | **PUBLIC** |
| owner | `yfeng0206` - the author's real identity |
| `paper/genai4health2026/main_submission.pdf` | **publicly readable**, 8,472,410 bytes |

So the complete submission PDF, plus its full LaTeX source, commit history and
author identity, is world-readable. A reviewer who searches any distinctive
phrase from the paper can find the repository and therefore the author. The
in-document anonymisation passes its own gate (`4_anonymous PASS`), so the
manuscript itself is clean; the exposure is entirely the public repository.

**I did not act on this.** Changing repository visibility affects a collaborator
who has a clone and may break external links, and it is not a decision to take
while the operator is away. The options, in increasing order of disruption:

1. Leave it. Many authors keep public repos and reviewers rarely search. This is
   a real but bounded risk.
2. Remove only `main_submission.pdf` from the default branch until the decision
   notification. The source would still be findable, so this is partial mitigation.
3. Make the repository private until the reviewing period ends, then restore it.
   This is the only option that actually closes the vector. It requires telling
   the teammate first.

Recommended: option 3 before the submission deadline, if the teammate agrees.

---

## FINDING 2 - Permanent history bloat, not fixable without a rewrite.

**[MEASURED]** 20 distinct blobs of `paper/dist/*.zip` are reachable from history
and occupy **272.8 MB** uncompressed. The remote repository is **245.8 MB**, which
is almost entirely this.

**Cause**, found today: `paper/.gitignore` line 15 carried `!dist/*.zip`, a
negation that re-enabled exactly what the root `.gitignore` excludes. The root
file records the reason for that exclusion - superseded dist copies leaked a
personal path in an embedded `EVIDENCE.md`. The negation silently won for months,
so every rebuilt Overleaf bundle was committed.

The negation is now removed and verified, so **no further bundles can be added**.
The 272.8 MB already in history can only be removed by rewriting history, which:

- breaks the teammate's clone,
- is forbidden by the standing rule for this repository,
- and is an operator decision, not an automated one.

**Left in place deliberately.** Recorded here so the size is understood rather
than mysterious.

---

## Current remote state - clean

| | |
|---|---|
| branches | `main`, `docs/background-signal-findings` |
| deleted this session | `ijepa-mask`, `vlm-guided-masking` (both verified 0 commits ahead) |
| tagged before deletion | `archive/volume-moe` - **125 commits** that were never merged, verified still reachable |
| other tags | `submission-prereframe-2026-08-26`, `review-panel-2026-08-26`, `phase1-masking-20260806` |
| releases | `phase1-masking-20260806` |
| untracked junk in working tree | 0 |

Branch merge status was verified directly with `git merge-base --is-ancestor`
rather than trusted from notes. `volume-moe` would have lost 125 commits; its tag
was pushed and confirmed present on origin **before** the branch was deleted, and
all 125 commits were re-verified reachable afterwards.
