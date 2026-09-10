# Substantive literal review: six original issues resolved

**Updated after targeted rereview of parent source
`5ec7b4b4be077e3e1a46f08e01ff896c3c11eec02ea4304ee81ead9530b5bca1`:**
all current manuscript literal/macro quantities now have passing source
bindings or explicit reviews. The parent removed the misleading pair-count
assertion and the unretained194 count, replaced the stale geometry count with
the seed42 provenance, correctly scoped the two subgroup workflows, made the
0.027 spread explicitly approximate, and removed the outdated extra-package
count. Those changes were individually rereviewed, not blindly rebased.

The two replacements are now integrated: the scatter passes independent
source/artist/PNG replay, and the token-map receipt retains explicit
non-mathematical illustration status. **The complete current numeric gate
passes, including an immutable staged-root test;103 regressions pass.**
Use `evidence\paper_release_numeric_candidate.json` for the complete receipt
bundle. Final Word/publication remains deferred pending the coordinator's new
stable-source signal. The six findings below describe the original61d source
and are retained as the reason for correction, **not current blockers**.

## Result and scope

**378 of the original385 unresolved tokens passed before source correction.**
The other seven concerned the six issues below, which are now resolved in the
revised source—not hundreds of unknown biological measurements. All **35 published-ablation
tokens** were checked against original public papers; the dataset licence
version was also checked against the commit-pinned official dataset card.

The reviewed candidate contains the existing DT bindings and selected
maps-only illustration receipt. It does **not** replace the parent-owned
`paper\genai4health2026\numeric_reviews.json`.

- Frozen-source candidate: `evidence\literal_review_candidate_61d.json`.
- Current-source candidate: `evidence\literal_review_candidate_current.json`.
- Exact source inventory/field operations: `evidence\literal_review_findings.json`.
- Independent reductions: `evidence\literal_sources\`.
- Primary-publication PDFs and acquisition hashes: `evidence\literal_sources\public\`.
- Initial substantive regression result: **94 passed**; the current expanded
  result is retained in `evidence\literal_review_tests.xml`.

The announced frozen source was `61d8d2cceb397f58313b166d481a03a3f94c2c0c914fb2ee8caca25cef7e27e6`.
It was validated using the exact retained final-Word source copy. Meanwhile,
the live manuscript changed during the independent figure cleanup; the
first revised candidate was checked against
`33926d717c530bbb42ba84977d5654f18df322f29d9c107fd0fb15488e17eb00`.
Only exact unchanged contexts and three individually rereviewed replacement
contexts were carried forward. Withdrawn attribution assertions were not
silently approved. `evidence\literal_context_migration.json` records this.

## Resolved adjudications from the original61d source

### 1. “22 discordant pairs” conflates a net difference with disagreements

The fixed original head was reapplied on CPU to the already retained fp32
feature cache. No encoder execution, training or GPU computation occurred.
The AUC reproduces as **0.8854753820184948**, versus original
**0.88548516482246**: absolute difference **0.000009782803965130427**.
The denominator **2,248,844** is correct.

However, the pair-order comparison contains **175 strict order flips** and
**324 tie-status changes**, hence **499 pair-order disagreements**.
**22 is the absolute net concordance-pair equivalent**, not the number of
discordant pairs.

**Minimal honest fix:** retain 22 but call it a *net pair-equivalent
difference*. Do not replace it with 499 as though those were the same metric.
The reproduction and its stated precision remain valid.

Evidence: `literal_sources\fixed_head_reproduction.json`, including hashes of
the saved head, cached features, original predictions and ancestor checkpoint.

### 2. “18 of 25 geometry cells” refers to an obsolete printed table

The old comparison script contains an earlier `PRINTED` table. Independently
formatting the current source-backed Table 2 cells from the seed-42 artifact
gives **15/25** inside the *unrounded three-seed min–max range*, or **25/25**
under the old script's *redraw-spread-or-rounding* criterion. Neither gives 18.

**Minimal honest fix:** remove the stale 18-count sentence, or state the
precisely chosen comparison rule. Do not silently interchange range containment
and a rounding-aware deviation criterion. The stored geometry values and
three-seed SDs themselves remain supported.

Evidence: `literal_sources\geometry_reduction.json` (`cells`,
`inside_unrounded_range`, `within_redraw_range_or_rounding`) and
`autopilot\compare_table2_geometry.py`'s obsolete `PRINTED` values.

### 3. The universal 50/10 subgroup rule describes the wrong workflow

`autopilot\p7_fairness.py` uses the **50 cases / 10 per class** rule. But the
seven-attribute trend artifact is generated by
`paper\genai4health2026\scripts\subgroup_analysis.py` and consumed by
`p7b_gap_trend.py`. That producer uses **minimum n=40**, skips
`unknown`/blank/`na` labels, and omits undersized groups.

The retained artifact independently confirms that **Spanish n=44 is included**,
while **marital unknown n=71** and **legally separated n=22** are absent from
the trend levels. The latter 71 is genuinely present in full-cohort metadata;
the numerical count and 2.4% rounded proportion are correct.

**Required fix:** scope the 50/10 sentence to its actual artifact, and correct
the adjacent universal claims that no category is omitted and unknown marital
status is retained in all analyses. Distinguish full-cohort composition counts
from each analysis's eligible levels. Do not regenerate the statistics under a
new exclusion policy merely to match the prose.

Evidence: the two producer implementations; retained
`subgroup_auc.json` language/marital levels; `literal_sources\metadata_reduction.json`.

### 4. The initial audit's “194 accepted slices” lacks retained raw evidence

This count survives in `autopilot\COVER_AUDIT.md` as a narrative report, but
the producing output was not persisted, as the manuscript itself states.
It cannot be independently verified as an actual sample count.

**Minimal fix:** omit 194, or explicitly attribute it as the earlier report's
unverified recollection. The retained 6,137-slice coverage comparison and
24,000-target rectangle audit were independently source-bound and need not be
removed.

This is the sole remaining token here whose producing numerical evidence was
genuinely unavailable, rather than a current contradiction or scope problem.

### 5. “Within 0.027” is not an exact bound on the label-efficiency spread

The exact full-supervision four-arm spread is **0.027082803431451863**,
which rounds to 0.027 but exceeds it.

**Minimal honest fix:** say the spread is *approximately 0.027*, retaining the
reported rounded number. The separate four-arm agreement bound **0.0009** is
valid: its maximum is **0.0008977946002479698**. The two-arm 0.0003 bound is
also valid.

Evidence: exact `p5_label_efficiency.json` arm fields and primary `p1c_stats.json`
rows. The candidate uses an executable inequality, so rounding cannot turn a
failed bound into a pass.

### 6. The environment has 19 additional distributions, not 16

All **87 lock pins match** the installed versions. The current interpreter
contains **19** additional distributions; the three additions relative to the
earlier count are `decompyle3`, `spark-parser` and `xdis`.

**Minimal fix:** remove or update/date the incidental extra-package count.
The Python, PyTorch, CUDA/cuDNN, NumPy, SciPy, scikit-learn, plotting/test package,
Tectonic, GPU-model, VRAM and driver declarations were checked. These checks
establish the local analysis environment, not every historical training host.

Evidence: `literal_sources\environment_snapshot.json` and
`literal_sources\hardware_snapshot.json`. Hardware data came from cached Windows
driver registry fields; no GPU computation was used.

## What was actually completed

The candidate now discharges epoch and architecture declarations, 16×16/256
geometry, 100-slice evaluation, configured schedules/seeds, the 95% interval
convention and null zero, label-efficiency fractions and valid bounds,
subgroup/severity counts, intersectional counts/gaps/deltas, positional-variance
shares, background similarity and residual-probe values/CIs, geometry
ratios/rank correlations, probability normality/variance diagnostics, checkpoint
bytes and fixed-head reproduction, software/hardware declarations, and the
original-publication ablation values. Statements remain conditional on their
actual source and scope; no `auto_numbers.tex` self-oracle or blanket approval
was used.

The original validation left seven literal tokens. Current validation has
**zero** literal, macro or figure blockers. The source-reviewed token-map
receipt was adopted without relabelling it mathematical verification; the new
scatter is registered for independent quantitative replay. The archived61d validation
also sees stale figure-identity requirements in its old isolated copy; that is
not an additional literal-data defect.

The parent has resolved the six issues. Further source changes require a new
stable-source signal and targeted rereview of changed contexts. Neither candidate changes
manuscript numbers, primary statistics, production Word/PDF/ZIP files,
publication state, or release-gate pass flags.
