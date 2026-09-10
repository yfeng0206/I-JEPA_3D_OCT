# Numeric coverage report

Status: **PASS**. No manuscript quantities were changed.
Verification here means independently reformatting exact stored source fields, not rerunning scientific analyses.
Historical illustration review is kept distinct from programmatic plotted-value verification.

Binding/review input: `C:\Users\Gary\Desktop\jepa\paper\genai4health2026\numeric_reviews.json` (SHA-256 `2d1985e01d7be99510e1181d77171d18b38ed53cec9c5a3420fffa834eac3db6`).

## Coverage

- explicit_no_result: 3
- nonnumeric_definition: 1
- programmatically_verified_plotted_values: 8
- reviewed_citation: 36
- reviewed_formula: 35
- reviewed_protocol: 166
- reviewed_source_illustration: 2
- structural: 809
- unused: 243
- verified: 892

## Gate errors


## Exact unresolved claims

Structural TeX numbers have been removed from this list. Protocol/formula/citation literals below still require an explicit review; they are not automatically labelled empirical measurements.

Unresolved classification (none are silently approved):

## Literal contexts

Each listed token is independently unresolved. Token indices address the normalized context hash, not manuscript line numbers.

## Review mechanism

References are {source, pointer}; pointers are RFC6901, never value searches.
Expressions are {op, args, ...}: format (Python %-format), subtract, add,
multiply, divide, min, max, length, mean, stdev (sample), abs, and literal.
Literal expressions are allowed only inside a documented protocol declaration,
not as empirical evidence. Custom macros use {"expression": expression}.

An optional paper/numeric_reviews.json contains version=1, sources, macros,
literals, figures. Sources specify root (repo/stats/paper), path, sha256.
Each literal specifies file, context_sha256, token_index, value and either an
expression, an assertion, or a review. An expression may select one numeric
component of its independently formatted display. Assertions compare an exact
source expression (lt/le/gt/ge) with a decimal or explicit power-of-ten bound;
their display component must match the printed token. Rounding never repairs
a failed inequality. Context is a whitespace-normalised paragraph or table
row, not a line number. Reviews require kind (protocol/formula/citation),
reviewer, rationale and evidence [{source, pointer} or {source, excerpt}].
P15 treats a staged active root main.tex as logical main_submission.tex for
binding/review lookups only; input_hashes and each literal's source_file retain
the physical paths, and approved receipt bytes are never rewritten.
Citation reviews additionally require immutable_locator (versioned DOI/arXiv
or a retained publication hash) and locator (page/table/section).
Retained PDF/image inputs expose /sha256 and /byte_length identity metadata
for HUMAN reviews only; this is not extraction or mathematical verification.
No reviews are inferred or written by the auditor.

Figure receipts require path, sha256, caption_sha256, inputs and validation.
Illustrations require method=reviewed_source_illustration (or the existing
reviewed_historical_illustration category),
reviewer, limitations and quantitative_scope=illustrative_only. Such receipts
are explicitly NOT mathematical verification. Programmatic receipts use
method=svg_coordinates, series=[{element_id, x, y, x_scale, x_offset, y_scale,
y_offset}]. The x/y expressions must resolve to source-derived arrays; the
actual SVG polyline/polygon/M-L-z coordinates are read and compared. Stable
Matplotlib group gids and source-valued bars are supported by method=svg_bars;
see numeric_svg_review.py for complete axis/decoration/label receipt fields.
Numeric ticks independently check bar affine mappings. Mixed token-map/bar
figures keep reviewed illustrations separate from mathematically checked bars.
Raster plots
need a real independent validator, or honest historical review; matching
their identity or attaching asserted numeric payloads does not verify pixels.

Known local raster producers are also executed CPU-only into memory/project-local scratch. Independent artist validators compare source-derived values, intervals and numeric annotations before requiring exact equality to the delivered PNG. The temporary producer outputs are removed; no manuscript asset is replaced.
No blanket literal approval, closest-value matching, self-validation against auto_numbers, or raster hash-as-numeric-proof is accepted.
Source-writing can move lines without invalidating a review; changing its paragraph/table-row content invalidates its context hash.
Run `python autopilot\p15_verify_numbers.py --report <coverage.json> --markdown-report <report.md>` after source stabilization.
The machine-readable report includes every token, semantic table key, source pointer/executable expression, and evidence hash.
