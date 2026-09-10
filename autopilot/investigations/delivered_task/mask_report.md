# Delivered-mask engineering report

Baseline: `de145d7`. Branch: `fix/jepa-delivered-task-audit`.
Owner: repair-masks / GPT-6 Astra. CPU only; no training, GPU work, commits,
uploads, manuscript edits, historical-config edits, or dependency installation.

## Verdict

**The historical delivered task does not fully implement the intended COVER
coverage/context contract.** This is a reproduced token-level finding, not an
explanation of downstream AUC.

The existing `results\masking\table2_geometry` measurements and
`autopilot\COVER_AUDIT.md` already measured truncation/coverage problems. This
work does **not** claim their first discovery. It adds a fixed-crop,
production-worker replay, final-context accounting, explicit source bookkeeping,
a versioned repair, and adversarial regression tests.

The corrected policy is opt-in:
`configs\patch_cover_delivered_v2.yaml`. It is a **future experiment with no
training results**, not a reinterpretation of an old arm.

## Production path and bounded replay

`GuidedOCTSliceDataset.read_slice` -> slice-cache bytes -> native-to-256 PIL
resize -> one `PairedRandomResizedCrop` draw for image, hard guide and both soft
score channels -> patch occupancy/placement/scores -> `MirageMaskCollator` in
a real DataLoader worker -> `CurriculumMaskGenerator.generate` -> target/context
collation -> measurements from the actual returned tensors.

- Reused the historical seed-42, 24-volume, 25-slices/volume, **600-slice Training
  scope**, reconstructed using the existing probe's selection rule. No expanded
  case search. Began with the first 64 scoped slices.
- New fixed crops: seed `91009 + ordinal * 997`. These are **not reconstructed
  historical crops**. The older script interleaved crop RNG with policy RNG;
  exact historical crops are not recoverable from its aggregate JSON alone.
- Every policy/batch-size replay verifies identical per-image crop and guide
  hashes. Drawn target/context sizes are explicitly injected from an independent
  size RNG (`3107 + floor(ordinal/64)`). All rectangle arms receive the same sizes.
- Placement RNG seeds are controlled, but branches consume RNG differently.
  **Placement draws are not claimed to be exactly paired across policies.**
  Batch-size comparisons also change some placements, so they are not isolated
  causal estimates of batch size. Within-image pre/post-collation differences
  are direct measurements of collation loss.
- Native cache bytes matched the original Training volume at two fixed scoped
  observations: maximum absolute byte error **0** for both.
- All 600 items carried four guide channels. Four guides failed post-crop QC.
  Channel 0 is hard-envelope occupancy; channel 1 is the placement region;
  channels 2/3 are separately cropped/pooled class scores. Soft support
  (`sum(scores) > .1`) and occupancy (`>= .25`) differed by **13.602 cells/image**
  on average. They are not interchangeable definitions.
- Synthetic coordinate-coded tests independently reconstruct expected image,
  nearest-neighbour hard-guide and soft-channel crops/pooling. One crop draw is
  used; deliberate guide shifts are detected. The implementation uses bicubic
  resized-crop interpolation for soft scores, despite an old bilinear comment.
  No reproduced spatial-alignment defect justified changing dataset/transforms.
  These checks do not establish clinical segmentation accuracy.

## Measured task change at nominal batch size 64

600 observations; same fixed inputs and injected sizes. As in the old 600-slice
probe, this is **nine full 64-image batches plus one 24-image tail batch**.
Every row records its actual batch size. Quantities below are per-image means
unless counts are explicitly given; the strict full-batch confirmation follows.

| Policy | Loss slots | Unique target cells | Duplicate loss slots | Target tissue / background slots | Final context cells / tissue cells |
|---|---:|---:|---:|---:|---:|
| RANDOM | 156.48 | 111.902 | 44.578 | 53.240 / 103.240 | 74.627 / 20.327 |
| intensity-localized ORACLE/CENTROID | 156.48 | 101.920 | 54.560 | 67.743 / 88.737 | 85.773 / 17.678 |
| ENVELOPE | 156.48 | 117.858 | 38.622 | 74.342 / 82.138 | 83.107 / 11.002 |
| ANATOMY, historical K=16 | 64.00 | 54.675 | 9.325 | 61.905 / 2.095 | 165.600 / 12.767 |
| COVER legacy | 156.48 | 118.665 | 37.815 | 69.110 / 87.370 | 68.560 / 9.978 |
| COVER exact-prefix v2 | 156.48 | 117.752 | 38.728 | 73.317 / 83.163 | 82.973 / 11.045 |
| v2 + explicit context guard | 156.48 | 117.752 | 38.728 | 73.317 / 83.163 | 82.973 / 14.960 |

The rectangle candidate areas totalled **173.52 slots**, but global-min
collation delivered **156.48**: **17.04 removed slots/image**. The v2 target
budget is exactly the same delivered budget, not 4 x K16. No rectangle-arm
budget was silently rewritten.

For historical COVER:

- **311/600** images lost unique tissue targets during prefix truncation.
  Mean loss: **2.710 tissue cells/image**, or **5.228** among affected images.
- On the **596 valid guided images**, scored hidden soft mass averaged
  **78.4947%**, but delivered hidden mass averaged **74.6343%**.
- In v2, scored and delivered mass both averaged **78.5510%**, agreeing to
  numerical precision. There were **0/600** target-truncation tissue losses.
- Historical final context retained **9.978 tissue cells**, versus
  **13.142** immediately before context collation and **18.010** in the
  complement of final targets. **The complement is not encoder context.**
- V2 excludes only delivered target indices when constructing context, instead
  of also excluding discarded candidate-rectangle cells. This changes the
  resulting context budget as well as target placement. It is not presented as
  an isolated location-only comparison with historical arms.

**Strict production-size check, excluding the short tail:** on the 576 images
in full 64-image batches (573 valid guides), historical COVER loses tissue
targets on **300/576**, has **55 zero-tissue contexts**, and misses the final
floor on **358/573 valid guides**. Scored/delivered soft coverage is
**78.5078% / 74.5705%**. Exact-prefix v2 has zero target-truncation losses but
**320/573** context-floor misses. V2+guard has **0/573** valid-guide misses.
Its one remaining zero-tissue context has an explicitly invalid guide.
Thus the defect and repair are not artifacts of the 24-image tail.

### A deidentified real failing example

Scoped observation **94**, historical COVER, batch size 64:

- Tissue: **74 cells**.
- Drawn targets: `(6,7), (8,6), (6,7), (6,6)`; intended slots **168**, final **144**.
- Intended unique tissue targets **57** -> delivered **44**.
- Scored soft coverage **78.9419%** -> delivered **60.7999%**.
- Context **94 -> 69 tokens** during collation; tissue **14 -> 0**.
- Final-target complement still contained **30 tissue cells**.
- Sources: two coverage blocks and two `random_legal` blocks.

Same cropped observation and drawn sizes under v2+guard: **144 slots**,
**57 unique tissue targets**, and **16 actual context tissue cells**, satisfying
`ceil(.21*74)=16`. Placement itself is not exactly paired. The raw rows preserve
both sets of final indices, so the example is reproducible rather than a visual
assertion. Another example, ordinal 66, goes from 11 pre-collation context tissue
cells to 0 historically, despite 28 tissue cells outside final targets.

## Per-image distributions and batch dependence

There are 600 rows per policy/batch-size, not just means. Every summary includes
minimum, p05, median, p95 and maximum; every row retains final and pre-collation
indices, source labels, guide validity, crop/guide hashes, loss-slot counts,
duplicate weights and tissue/background counts.

| Historical COVER batch size | Mean final context tissue | p05 / median / p95 | Zero-tissue contexts, all 600 | Floor misses among 596 valid guides |
|---|---:|---:|---:|---:|
| 1 | 13.182 | 6 / 14 / 20 | 1 | 238 |
| 2 | 12.655 | 4.95 / 13 / 19.05 | 8 | 263 |
| 64 | 9.978 | 0 / 11 / 18 | 55 | 378 |

The comparison floor is `max(ceil(.21*N_occupancy), min(4,N_occupancy))`.
It is a COVER design threshold, **not a promised constraint of RANDOM,
ORACLE, ENVELOPE or ANATOMY**.

Exact-prefix v2 without a context guard missed that floor on **247 / 258 / 339**
valid observations at batch sizes **1 / 2 / 64**. Correcting target scoring alone
is therefore insufficient. With the separately enabled guard, all **596 valid
observations** satisfied the final occupancy-cell floor at each batch size.
The **four invalid guides are explicitly uncertified**. One has zero tissue;
one invalid observation at batch size 1 still misses the numerical threshold.
Neither is silently counted as a guaranteed success.

Guard-only ablation was checked using exact arrays, not assumed from seeds:
all **1,800** v2/v2+guard pairs have identical targets, pre-collation context and
final context token budgets. The guard changes **247 / 258 / 339** contexts,
respectively. It replaces background tokens with available non-target tissue,
possibly outside the originally sampled context rectangle. This is an explicit
**second guide-aware intervention**, not merely target placement.

These are descriptive engineering measurements on a fixed scope containing
multiple slices per volume, not 600 independent experimental replications.
No hypothesis tests, population confidence intervals or downstream-effect
claims are inferred from them.

## Guided branches, random fills and background supervision

At `r_max=1` after ramp-up, every usable COVER/ENVELOPE target takes the guided
branch. This is separate from COVER's leftover-block fill mechanism.

At batch size 64:

- Legacy COVER: **1,889** coverage blocks, **495** `random_legal` blocks,
  **16** invalid-guide fallback blocks. **177/600 images had no random slots**.
- V2: **1,974** coverage blocks, **410** `random_legal` blocks, the same
  **16** fallback blocks. **245/600 images had no random slots**.
- Legacy random-source slots averaged **33.773**, including **8.392 tissue**
  and **25.382 background** slots. V2 averaged **28.552**, including **6.800
  tissue** and **21.752 background** slots.
- Background prediction is not absent when no random blocks remain: coverage
  blocks also contain background. Total v2 background slots averaged **83.163**.
- ENVELOPE had **25 per-block infeasible-uniform fallbacks**, plus **16**
  invalid-guide fallback blocks. Its other **2,359** blocks were guide placed.
  These uniform fallbacks were previously obscured by per-image source labels.

The repair **does not force a preferred random/background policy to win**.
Indeed, exact-prefix v2 produces fewer random-fill slots in this replay.
Whether a guaranteed random quota or a different background dose would help
requires a separate policy definition and new evidence.

A final unit control explicitly holds all four ramp flags true and obtains
**one random-legal fill on a thin-band fixture versus zero on a thicker-band
fixture**. Thus `r_max=1` and random-fill count are not interchangeable.
The literature matrix's I-JEPA context-scale sensitivity and DSeq's
top-saliency-context design motivate joint target/context measurement, not a
claim that those systems prove this mechanism. Our `r_max=1` is **not**
SemMAE whole-part `alpha=1`. The per-image rows jointly retain final context
tissue and tissue/background/duplicate target slots; actual SmoothL1 values and
gradient partitions are delegated to the training engineer's exact-mask
diagnostic, not inferred here from token counts.

## Repairs and compatibility

1. `cover_algorithm: legacy_v1` remains the default. The explicit
   `delivered_v2` path scores and returns the exact row-major prefix of each
   candidate rectangle, including its partial final row. Candidate-placement
   bounds still use the full drawn rectangle.
2. `cover_context_guard: true` is separately selected in the new config.
   It preserves the final encoder token count and never overlaps targets.
   It certifies occupancy-cell floors, not a post-context soft-score-mass floor.
   Invalid guides, insufficient non-target tissue and insufficient context
   budgets receive explicit statuses. The guard is independent of the target
   ramp, including at `r_t=0`; that is part of the new policy, not historical
   warm-up behavior.
3. Fixed the nonprefix `_epoch` versus nonexistent `epoch` reseeding defect.
   `epoch_worker_v2` also avoids restarting the same RNG stream every batch and
   uses configured rank when worker-side distributed state is unavailable.
   `enc_truncate_rng: legacy_v1` replays the historical epoch-zero/restarting
   behavior. **Current prefix arms are not attributed this defect.**
4. Impossible-context fallback formerly inserted initial grid tokens, potentially
   exposing targets. It now keeps a smaller legal nonempty context, or raises
   explicitly when no non-target context exists. Demonstrated on a full-grid
   target fixture; no frequency claim is made for real training.
5. Corrected diagnostic bookkeeping: ANATOMY ramp-off is not an invalid guide;
   guided-image counts reflect delivered anatomy targets; ENVELOPE labels its
   per-slot uniform fallbacks; transition-mode least-tissue fallback is labeled
   `boundary_fallback`, not falsely counted as uniform random fill.
6. Probe scripts now distinguish injected size pairing from seed equality,
   preserve full guide channels, and retain per-image statistics. The edge probe
   now runs through production final collation and displays/counts actual
   encoder context rather than the target complement. Its older, separate real
   edge dataset was **not** replayed; its new path was tested synthetically.
7. Follow-up from the training engineer: `nenc > 1` context groups were
   independently rectangularized, although `apply_masks` concatenates groups
   along the batch dimension and requires a shared token count. Both stock and
   curriculum samplers now use the global minimum across context groups/images.
   This repairs a previously unsupported multi-context configuration;
   historical `nenc=1` masks are unchanged.

The latest fifteen synthetic historical controls compare returned mask tensors
with samplers **and mask dependencies** independently loaded from
`git show de145d7`: five arms x three seeds, **all bitwise equal** under historical
defaults. This supersedes the earlier COVER controls' circular dependency
binding, detailed in the critic follow-up below. Historical YAMLs are untouched.

## Validation and exact commands

Run from `C:\Users\Gary\Desktop\jepa`. All commands used the existing interpreter.

```powershell
$env:MPLBACKEND='Agg'
& 'D:\jepa_phase0\.venv\Scripts\python.exe' -m pytest tests\test_delivered_masks.py tests\test_mirage_envelope.py tests\test_mirage_anatomy_mode.py tests\test_pred_target_k.py tests\test_slice_cache.py tests\test_mirage_config_wiring.py -q --disable-warnings --basetemp autopilot\investigations\delivered_task\evidence\mask_pytest_work
```

**123 passed**, 12.64 seconds pytest-reported after the critic fixes
(figure phase: 116; full-ramp: 114; multi-context: 113; initial: 106).
Controls cover exact prefix
scoring, historical parity, guide shift, invalid/ramp-off branches, worker
pickle/epoch/rank streams, source labels and duplicate weighting, nonempty
in-bounds masks, no encoder/target overlap, context infeasibility, deliberate
wrong target count, and independent coordinate-coded paired-crop reconstruction.

```powershell
& 'D:\jepa_phase0\.venv\Scripts\python.exe' scripts\delivered_mask_audit.py --count 64 --batch-sizes 64 --workers 1 --out autopilot\investigations\delivered_task\evidence\mask_replay64_v2
& 'D:\jepa_phase0\.venv\Scripts\python.exe' scripts\delivered_mask_audit.py --count 600 --batch-sizes 1 2 64 --workers 1 --out autopilot\investigations\delivered_task\evidence\mask_replay600_v2
& 'D:\jepa_phase0\.venv\Scripts\python.exe' scripts\delivered_mask_audit.py --count 64 --batch-sizes 1 2 64 --workers 1 --out autopilot\investigations\delivered_task\evidence\mask_final64_v2
& 'D:\jepa_phase0\.venv\Scripts\python.exe' scripts\delivered_mask_audit.py --count 64 --batch-sizes 64 --workers 1 --arms cover_legacy cover_v2 cover_v2_guard --out autopilot\investigations\delivered_task\evidence\mask_releasecheck64_v2
```

The final current-source 64-slice run took **131.15 seconds observed wall time**
(including 21 Windows worker startups), checked **192 exact guard-only pairs**,
and again found 46/64 historical target-tissue truncation losses and 21/64
historical context-floor deficits at batch size 64, versus zero of each with
v2+guard. GPU time: **0**. The 600-slice run was repeated once to verify corrected
ENVELOPE source labels; mask tensors and substantive mask counts were unchanged.

Evidence:

- `evidence\mask_replay600_v2\summary.json`: complete 21-arm/batch cells, quantiles,
  baseline/code hashes, Python/Torch versions and native-cache controls.
- Same directory's `*_bs*.jsonl`: 12,600 deidentified per-image rows.
- `exact_guard_pair_verification.json`: independent exact-array checks of all
  1,800 guard-only pairs. Current harness also checks these automatically.
- `full_batch64_confirmation.json`: strict 576-image production-batch check,
  excluding the 24-image tail rather than silently treating it as batch 64.
- `evidence\mask_final64_v2\summary.json`: all-arm final confirmation.
- `evidence\mask_releasecheck64_v2\summary.json`: final source hashes after
  docstring clarification, before the later multi-context fix.
- Its `validation_manifest.json` records hashes of both configs, the
  data/guide/transform path, mask utilities, harness and regression tests at that
  phase. The subsequent multi-context validation is separately recorded below.
- `historical_replay_controls.json`: 15 bitwise historical-default controls.

The 600-slice evidence precedes the final addition of the transition-fallback
diagnostic label (its COVER runs use `random_legal`, so no measured mask changed).
Final-source hashes and checks are therefore separately preserved rather than
pretending the earlier replay used later source bytes.

### Multi-context follow-up: failing before, passing after

The training owner's initial GPU diagnostic stopped before its first completed
batch: context groups had K112 and K128. Its preserved evidence is
`evidence\training_gpu_v1\verdict.json`. This is **not evidence of corruption in
historical training**, which used `nenc=1`.

CPU reproduction against `de145d7`, with the same explicit size draws before and
after, exercises the actual `src.masks.utils.apply_masks`:

| Sampler | B | Before context-group K | After K | Actual apply_masks result |
|---|---:|---:|---:|---|
| stock | 1 | 69 / 157 | 69 / 69 | `(2,69,3)`, passes |
| curriculum | 1 | 69 / 157 | 69 / 69 | `(2,69,3)`, passes |
| stock | 3 | 49 / 133 | 49 / 49 | `(6,49,3)`, passes |
| curriculum | 3 | 49 / 133 | 49 / 49 | `(6,49,3)`, passes |

All four old-code calls fail with unequal-size concatenation. Targets are
bitwise unchanged; new contexts are exactly the common prefix of old contexts.
Additional regressions cover guarded COVER and deliberately corrupted
cross-group context lengths. No model-side workaround was made.

`evidence\mask_nenc2_contract_v1\verification.json` records the failures,
successes, source hashes, and a fresh successful replay of all 15 historical
single-context controls. A final worker check used:

```powershell
& 'D:\jepa_phase0\.venv\Scripts\python.exe' scripts\delivered_mask_audit.py --count 64 --batch-sizes 64 --workers 1 --arms cover_legacy cover_v2 cover_v2_guard --out autopilot\investigations\delivered_task\evidence\mask_nenc2_single_context64
```

The new directory's summary hashes match current source. Its
`single_context_exact_parity.json` verifies **192 exact real-input target/context
pairs** against the pre-fix `nenc=1` worker replay, not merely equal means.
Historical 46/64 target-loss and 21/64 floor-miss counts, and corrected 0/64
counts, are unchanged.

## Independent-critic follow-up: provenance and baseline isolation

The two bounded defects supplied by the coordinator were reproduced before
changing source. Five targeted regression cases failed as expected:
`evidence\mask_critic_fix_v1\before_tests.txt`.

1. **Valid but unusable COVER guides were mislabeled invalid.** A declared-valid
   four-cell guide now increments `infeasible`, not invalid-guide `fallbacks`.
   When a mixed-ramp attempt fails, false ramp flags keep the `unguided` source:
   `FFTT` now yields `unguided, unguided, infeasible, infeasible`, not four
   infeasible labels. Truly invalid guides and entirely ramp-off viable guides
   retain their distinct reasons. Source diagnostics now explicitly carry
   **`target_source_schema_version: 2`**. Both legacy and opt-in paths were
   tested; all four before/after synthetic failure fixtures have **bitwise
   unchanged context and target tensors**.
2. **Earlier COVER historical controls were circular.** Executing the baseline
   curriculum had imported the current COVER module. The diagnostic now loads
   baseline `utils`, `anatomy`, `cover`, `multiblock` and `curriculum` in isolated
   module objects, binds baseline dependencies during import, and restores all
   affected `sys.modules` entries and package attributes in `finally`.
   Assertions verify baseline COVER function objects are not current objects.
   Both normal and injected-failure restoration are regression-tested.

The discriminating adversarial test shifts only the current COVER implementation
(including its current curriculum alias). Before the fix, the circular control
incorrectly passed. After the fix, it correctly raises
`Historical default masks changed: cover_legacy, seed 0`.
Without corruption, the **true independently bound 15 controls all pass**.
The earlier control defect did not invalidate the measured mask arrays, but
its three COVER entries were not an independent compatibility test.

**Saved-evidence impact:** all **5,400 records** in the nine saved COVER files
(three variants x batch sizes 1/2/64 x the existing 600-slice scope) were checked
using their recorded validity, viability and ramp flags. There are **zero**
valid-nonviable cases, zero valid failed attempts and zero mixed-ramp failures
in those full-ramp records. Consequently, **zero source labels or reported
source counts change**. The schema-2 projected counts are recorded explicitly
alongside the old counts; no saved file or old label was rewritten.

The final real-worker 64-slice smoke also verifies **192 exact image, guide,
context, target and source-label pairs unchanged**, with the new schema marker.
Historical 46/64 target-loss and 21/64 context-floor-miss counts, and corrected
0/64 counts, remain unchanged. No new GPU work was performed.

Evidence:

- `evidence\mask_critic_fix_v1\after_tests.txt`: **123 passed**.
- `before_masks.json`: immutable pre-fix failure-fixture tensors/provenance.
- `provenance_impact.json`: before/after synthetic bookkeeping and all nine
  saved-file hashes, old counts and schema-2 counts.
- `verification.json`: isolated baseline identity assertions, 15 passing
  controls, adversarial rejection and exact real-worker parity.
- `real64\historical_replay_controls.json`: independently bound replay output.

Commands:

```powershell
& 'D:\jepa_phase0\.venv\Scripts\python.exe' -m pytest tests\test_delivered_masks.py -q -k 'valid_nonviable or mixed_ramp_failure or detects_current_cover_mutation' --disable-warnings
& 'D:\jepa_phase0\.venv\Scripts\python.exe' -m pytest tests\test_delivered_masks.py tests\test_mirage_envelope.py tests\test_mirage_anatomy_mode.py tests\test_pred_target_k.py tests\test_slice_cache.py tests\test_mirage_config_wiring.py -q --disable-warnings --basetemp autopilot\investigations\delivered_task\evidence\mask_critic_fix_v1\pytest_work
& 'D:\jepa_phase0\.venv\Scripts\python.exe' scripts\delivered_mask_audit.py --count 64 --batch-sizes 64 --workers 1 --arms cover_legacy cover_v2 cover_v2_guard --out autopilot\investigations\delivered_task\evidence\mask_critic_fix_v1\real64
```

The first command was run before the source fixes to record the five failures.
The figure script and canvases are now parent-owned and were not changed in
this follow-up.

## Parent-requested deidentified figure

Headless source: `scripts\delivered_mask_figure.py`.

```powershell
$env:MPLBACKEND='Agg'
& 'D:\jepa_phase0\.venv\Scripts\python.exe' scripts\delivered_mask_figure.py
```

`evidence\mask_figure_v1\delivered_masks.png` and `.pdf` show:

- One previously audited fixed Training view as **token-grid masks only**,
  without raw OCT pixels, case filenames or subject identifiers.
- Historical candidate targets/context versus the exact final indices, then
  separately labeled opt-in prefix scoring plus guide-aware context correction.
- Only aggregate numerical evidence: hidden guide mass over 573 valid views
  and encoder-visible tissue over 576 views from nine complete batches of 64.
  All bar axes start at zero; no AUC or downstream-benefit claim is made.

The directory also contains `aggregate_metrics.csv`, `caption.txt`,
`alt_text.txt`, `figure_manifest.json` and `palette_screen.json`. Source evidence
and script hashes are recorded; the chosen view's image/guide hashes were
verified against existing audit rows, but neither pixels nor identifiers are
rendered or exported. No measured sample was added.

Both delivered exports were inspected headlessly. PNG: **2115 x 1725**, opaque
RGB, 300 DPI. PDF: one **507.6 x 414 pt** page, embedded Type0 fonts, **zero
embedded raster images**. The initial label collisions were corrected before
handoff. Color-only screening flags the blue/orange pair's similar lightness;
context hatching, direct stage labels and black tissue circles supply redundant
encodings. This is **not an accessibility or publisher-compliance certification**;
the parent must check final-size LaTeX placement and may use the appendix.
Figure-specific tests verify short-tail exclusion, invalid-guide denominators,
and rejection of context-target overlap. Training/mask implementation is unchanged.

## Trainer fixture handoff and remaining uncertainty

`evidence\mask_final64_v2\synthetic_final_masks.pt` is safe to share: **synthetic
pixels only**. Schema version 1:

```text
images: [B,3,256,256] CPU float tensor
masks_enc: list of [B,Kenc] CPU long tensors
masks_pred: four [B,Kpred] CPU long tensors
guides: [B,4,16,16], guide_valid: [B]
metadata: synthetic source, policy, seed
```

The trainer engineer was notified promptly of the same interface and the first
fixture under `mask_replay64_v2`. Diagnostic policy labels must reuse these exact
tensors. New production statistics expose delivered context floor
satisfied/unsatisfied/intervention counts; trainer logging changes were requested
from its owner, not made here.

The later real-data handoff is **private and must never accompany the paper**:
`.audit\delivered_task_training\mask_handoff\mask_engineer_real_b1_b2_final_v3.pt`.
It contains 14 exact frozen-mask cases (seven policies x B1/B2) from two
already-audited Training observations, tissue booleans derived from channel 0
at `>= .25`, structured preprocessing/source hashes and a private source map.
Git-ignore status, safe CPU reload and source/crop/guide identities were checked.
SHA-256: `51d516c3564d0b106255b28fed28f5b3290b6487af7ee743ad3908efe918dcb6`.
The training engineer owns any separately approved bounded GPU use.

For the coordinator's real-mask joint-loss follow-up (at most **three independent
fp32 updates**, same ancestor reset, under the training owner's lease), the
exact requested three-policy B2
subset is privately saved as `real_cases.pt` in the same ignored directory;
its original identity map is separate in `real_cases_identity_map.json`.
Fixture SHA-256:
`9effcc2d02daa7460375f7b8589fad4042192c25f20f49b0dafd65c6a5c0357b`.
It uses the **first two predeclared observations without replacement or
outcome-based selection**. The guard is nonbinding on that pair: prefix-only v2
and guarded v2 have identical final masks. This remains an intentional negative
control; it must not be replaced by a more favorable example.

Final export readiness is verified in
`evidence\mask_real_loss_handoff_v1\handoff_manifest.json`. Aggregated over the
two predeclared views: legacy has **320 target slots = 177 tissue + 143
background**, with **76 repeated slots**, and **35 tissue cells in 252 context
tokens**. Both v2 cases have **320 = 178 + 142**, with **78 repeated slots**,
and **35 tissue cells in 264 context tokens**. These are verified index counts,
**not measured SmoothL1 values**; the training owner records actual loss and
gradient partitions. There is no mask-export blocker.

### Real guided-loss follow-up completed by the training owner

`evidence\training_gpu_guided_v1\verdict.json` records **three completed
independent fp32 updates**, resetting the same ancestor for each frozen policy.
These are real Training inputs, not the earlier synthetic/tiny-ViT check.
The mask owner performed a separate **CPU-only, read-only reconciliation**
against the exact exported mask/guide tensors:
`evidence\mask_real_loss_handoff_v1\guided_loss_reconciliation.json`.

Every reported context-tissue, tissue/background slot and repeated-slot count
matches the fixture. Actual scalar SmoothL1 and its normalized contributions:

| Policy | Scalar loss | Tissue contribution | Background contribution | First-occurrence contribution | Repeated-slot contribution |
|---|---:|---:|---:|---:|---:|
| Legacy COVER | 0.04659984 | 0.02612709 | 0.02047275 | 0.03441324 | 0.01218660 |
| Exact-prefix v2 | 0.09030718 | 0.04941365 | 0.04089354 | 0.07256228 | 0.01774491 |
| V2 + guard | 0.09030718 | 0.04941365 | 0.04089354 | 0.07256228 | 0.01774491 |

Each contribution is a category's per-slot loss sum divided by all **320**
delivered slots. Tissue + background and first + repeated are **two alternative
partitions of the same scalar**; the four columns must not be added together.
Both decompositions reconcile the scalar within floating-point tolerance.
The 35 encoder-visible tissue cells summed across the two views are unchanged
across policies. V2/guard masks and losses are exactly identical, confirming the
predeclared nonbinding-guard negative control.

The trainer reports finite activity through all 30 transformer blocks, nonzero
online/predictor gradients and updates, no teacher gradients, zero EMA error and
zero hidden-pixel effect on the masked online branch. The three-update bound
completed cleanly. **Larger or smaller loss here is not a policy ranking or AUC
result**: target/context tasks differ, and two fixed views cannot evaluate a
representation-learning strategy. No policy was changed based on these losses.

Provenance clarification: the GPU manifest records the earlier private container
SHA `269f1c143af8e91daa179796fb0bfbd40583d91fafebbc0dbd45cba9c6c4692e`.
Its explicitly selected `bs2` tensors for the three policies were checked to be
exactly equal to the narrowed `real_cases.pt` handoff. Different serialized
container hashes therefore do **not** indicate different images or masks.

No real image, case filename, label or patient manifest is in these public-safe
reports. Source paths remain local in the existing configurations.

Established: target-scoring/truncation mismatch, final-context losses, actual
random/background slots, versioned repair contracts, and preserved historical
default masks on the tested controls. Unresolved: segmentation accuracy as an
importance proxy, optimal target/context/background balance, downstream AUC
effect, and full historical run provenance beyond the existing audits.
Old checkpoints remain results of their historical delivered tasks. They cannot
be relabeled as corrected-policy results or evidence that intended over-coverage
caused harm. A downstream comparison requires separately authorized, matched
new training, not this bounded replay.
