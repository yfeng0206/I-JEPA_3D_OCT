# Numeric coverage report

Status: **BLOCKED**. No manuscript quantities were changed.
Verification here means independently reformatting exact stored source fields, not rerunning scientific analyses.
Historical illustration review is kept distinct from programmatic plotted-value verification.

Binding/review input: `autopilot\investigations\delivered_task\evidence\literal_review_candidate_61d.json` (SHA-256 `549b66226798cbb3e02c73b1b6c5630efcc2b950c0a6cec5297427eb223b609c`).

## Coverage

- explicit_no_result: 3
- mismatch: 4
- nonnumeric_definition: 1
- programmatically_verified_plotted_values: 7
- reviewed_citation: 36
- reviewed_formula: 34
- reviewed_protocol: 169
- reviewed_source_illustration: 1
- structural: 814
- unresolved: 10
- unused: 243
- verified: 898

## Gate errors

- asset identity declaration failed

## Exact unresolved claims

Structural TeX numbers have been removed from this list. Protocol/formula/citation literals below still require an explicit review; they are not automatically labelled empirical measurements.

Unresolved classification (none are silently approved):
- empirical_or_unclassified: 7
- figure: 7

### `figures/fig_masking_policies.png` (figure; unresolved)
- Action: Independent plotted-value validator or explicit historical-illustration review required; identity is not numeric proof.

### `figures/fig_precision_paradox.png` (figure; unresolved)
- Action: Independent plotted-value validator or explicit historical-illustration review required; identity is not numeric proof.

### `figures/interp_04_window_occlusion_W7.png` (figure; unresolved)
- Action: Independent plotted-value validator or explicit historical-illustration review required; identity is not numeric proof.

### `figures/interp_14_odos_mirror_test.png` (figure; unresolved)
- Action: Independent plotted-value validator or explicit historical-illustration review required; identity is not numeric proof.

### `figures/interp_heatmap_grid.png` (figure; unresolved)
- Action: Independent plotted-value validator or explicit historical-illustration review required; identity is not numeric proof.

### `figures/interp_slice_contribution_by_outcome.png` (figure; unresolved)
- Action: Independent plotted-value validator or explicit historical-illustration review required; identity is not numeric proof.

### `figures/interp_slice_contribution_curves.png` (figure; unresolved)
- Action: Independent plotted-value validator or explicit historical-illustration review required; identity is not numeric proof.

## Literal contexts

Each listed token is independently unresolved. Token indices address the normalized context hash, not manuscript line numbers.

### `main_submission.tex` — context `49b2714ba17267f8e190b22b8d13447615d688ab8ab94138aab59ffa2f1afbfd`
- Tokens: `22` (index 11; empirical_or_unclassified)
- Context: The eight recorded fp32 re-probes shift AUC by less than $2\times10^{-4}$, orders of magnitude below the effects in Table~\ref{tab:main}, so the differing autocast settings of Section~\ref{sec:setup} are not the effect and every contrast is matched on precision (Appendix~\ref{app:fp32}). The generating script, the epoch-100 \ArmBest{} encoder, its head and its stored per-case predictions are released (links withheld for anonymity). Re-encoding the test split from those weights on different hardware, at fp32 rather than fp16 and with a different encoder chunk size, reproduces the reported AUC to $9.8\times10^{-6}$: 22 discordant pairs out of $2{,}248{,}844$.
- Mismatch at index 11: expected `499`.

### `main_submission.tex` — context `204a825e56fca6b4b7d68f677ee2dd40d466a244816f9247ccb1f1dd7f8e4e2c`
- Tokens: `18` (index 0; empirical_or_unclassified)
- Context: \paragraph{Cell-level agreement and seed noise.} Table~\ref{tab:geom} now carries the regenerated values, so the question is how stable they are across draws. 18 of the 25 cells fall inside the three-seed range of the measurement and do not exceed observed re-draw variation. Across the four rectangle arms the \emph{loss slots} spread is smaller than the per-arm seed spread (Table~\ref{tab:geomprov}), so that column's ordering carries no information and no claim in this paper depends on it. The rank order of the arms on anatomy hidden and on purity is identical across all three draws, which is what the rank correlations in the main text rely on.
- Mismatch at index 0: expected `15`.

### `main_submission.tex` — context `ae4e0f91cf3199313e276a4596c8cdcd80ff0a0f1e72caabf34b22ed461db136`
- Tokens: `50` (index 0; empirical_or_unclassified); `10` (index 1; empirical_or_unclassified)
- Context: Separately from missingness, a group with fewer than 50 cases, or fewer than 10 of either class, is still computed and reported but is marked underpowered and held out of the max--min gap summaries. That is an exclusion from the fairness claim, not from the data, and it is the only place a group is set aside.

### `main_submission.tex` — context `cb967b5622113a897990d85cd8c49bbe5dc773c7c6bc467e983e5101e99f488e`
- Tokens: `194` (index 5; empirical_or_unclassified)
- Context: \paragraph{Measured consequences.} A one-off audit found that collation reduced \textsc{cover}'s anatomy coverage. An independent sweep over $6{,}137$ slices gives $73.1\%$ against $77.6\%$ for \textsc{envelope}. The arm intended to hide \emph{more} anatomy than \textsc{envelope} in fact hides \emph{less}. Across $24{,}000$ emitted targets only $73.4\%$ remain perfect rectangles. Because target cells are enumerated row-major, truncation keeps the top of each rectangle and discards the bottom, so the loss is directional rather than a neutral shrink. \emph{Provenance:} the one-off CPU audit of 194 accepted slices that first exposed the defect was not persisted, so its pre- and post-truncation figures are no longer asserted. The $73.1\%$ against $77.6\%$ comparison, which is the only figure the body relies on, is backed by the stored floor sweep and is unaffected.

### `main_submission.tex` — context `1792b23d3f534f5869ef651223c9fccd8193bb3842d9298fa35f6f8067ba9af4`
- Tokens: `0.027` (index 1; empirical_or_unclassified)
- Context: \begin{figure}[h] \centering \includegraphics[width=0.60\linewidth]{fig_labeleff.png} \caption{The policies converge as labels are added and separate as they are removed. Shading is one standard deviation over \LERepeats{} label subsets, with every arm seeing identical subsets. At full supervision the four arms lie within $0.027$ of one another; at $5\%$ the spread is $0.085$.} \label{fig:labeleff} \end{figure}
- Mismatch at index 1: expected `0.027`.

### `main_submission.tex` — context `44168e4e42060c235832486d385f43da7eec48abdca9b70728145869da5b7c9a`
- Tokens: `16` (index 14; empirical_or_unclassified)
- Context: \paragraph{Software and hardware environment.} Every number in this paper was produced under one environment, and the versions below were read from the live interpreter rather than recalled. Python 3.11.9 on 64-bit Windows (build 10.0.26200); PyTorch 2.7.1 compiled against CUDA 12.8 with cuDNN 9.7.1; one NVIDIA GeForce RTX 3090 with 24\,GiB of memory, driver 610.62. The statistics and figures use NumPy 2.4.4, SciPy 1.17.1 and scikit-learn 1.9.0, and the manuscript is compiled by Tectonic 0.17.0. A lock file pinning 87 distributions accompanies the artifact. We re-resolved it against the live interpreter and every pin matches the installed version exactly, with no conflict. That interpreter also carries 16 further packages (test, plotting and PDF-inspection tools such as pytest 9.1.1, statsmodels 0.14.6 and seaborn 0.13.2), which are deliberately absent from the lock file because no reported number depends on them. The chain from stored per-case predictions through macros, tables and figures to the compiled archive runs as one ordered script, so the released predictions, the released head and the numbers printed here are joined by code rather than by transcription. We did not re-verify these results under a second CUDA or PyTorch version, so the fp32 re-encode of Appendix~\ref{app:fp32} on different hardware is the only cross-environment evidence we have.
- Mismatch at index 14: expected `19`.

## Review mechanism

References are {source, pointer}; pointers are RFC6901, never value searches.
Expressions are {op, args, ...}: format (Python %-format), subtract, add,
multiply, divide, min, max, length, mean, stdev (sample), abs, and literal.
Literal expressions are allowed only inside a documented protocol declaration,
not as empirical evidence. Custom macros use {"expression": expression}.

An optional paper/numeric_reviews.json contains version=1, sources, macros,
literals, figures. Sources specify root (repo/stats/paper), path, sha256.
Each literal specifies file, context_sha256, token_index, value and either an
expression or a review. Context is a whitespace-normalised paragraph or table
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
