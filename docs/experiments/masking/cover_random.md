# COVER-then-RANDOM masking

**Status:** implemented and forward-gated; **not trained**.
**Config:** `configs/patch_cover_random_ep25.yaml`
**Main interpretation:** [background signal investigation](background_signal.md)

## Why this variant exists

The anatomy-blob arm hides the desired cells but changes target shape and
supplies far fewer prediction slots. The rectangular envelope arm preserves
stock I-JEPA geometry but leaves much more anatomy visible than intended.
COVER was introduced to keep ordinary batch-shared rectangles while changing
only where they are placed.

The requested variant adds a second phase:

1. greedily place rectangles to hide anatomy;
2. stop before either anatomy-visibility floor is crossed;
3. place every leftover block uniformly, as in stock I-JEPA, but only among
   windows that keep the floors legal.

This is **COVER-then-RANDOM**, implemented as `fill="random_legal"`.

## Contract

For each image, `src/masks/cover.py::build_targets` receives block sizes already
sampled once per batch by `CurriculumMaskGenerator`. It does not resample shape
or area, so comparisons with stock and envelope rectangles preserve the target
geometry and number of slots.

The two hard floors are:

```text
min_visible_frac   minimum fraction of anatomy soft mass left in context
min_visible_cells  minimum number of anatomy-support cells left in context
```

The soft stop is:

```text
leave_frac         stop spending blocks on coverage after this fraction remains
```

The current run config sets all fractional values to 0.15 and the cell floor
to four.

## Placement algorithm

### Phase 1: cover anatomy

For every remaining block size, the sampler evaluates all legal windows on the
16×16 patch grid using summed-area tables. It chooses uniformly among windows
with maximal newly covered anatomy mass. Legality is evaluated against the
cumulative union of blocks, not each rectangle independently.

Non-guided blocks selected by the curriculum ramp are placed first and folded
into the same bookkeeping. This keeps the final-union floor honest and
preserves the ramp's per-block Bernoulli behavior.

### Phase 2: spend leftover slots

`fill` selects one of three policies:

| value | leftover placement | floor enforced |
|---|---|---|
| `transition` | maximize anatomy/background boundary balance | yes |
| `random` | plain uniform window | no |
| `random_legal` | uniform among legal windows | yes |

`random_legal` does no boundary scoring and no greedy coverage scoring in phase
2. Its only departure from a stock uniform draw is rejection of windows that
would violate the stated floor.

If no legal window exists, the least-damaging window is marked
`random_violation`, `info["floor_violation"]` becomes true, and the caller
discards the guided result in favor of a counted uniform fallback. A floor
breach is never silently emitted as a successful guided sample.

Every slot records provenance in `info["slot_kind"]`, including `cover`,
`random_legal`, `transition`, `unguided`, and fallback/violation states.

## Integration

`src/masks/curriculum.py` exposes:

```yaml
mask:
  curriculum:
    mode: mirage_cover
    cover_leave_frac: 0.15
    cover_min_visible_frac: 0.15
    cover_min_visible_cells: 4
    cover_fill: random_legal
```

The default remains `transition`, preserving existing COVER behavior.
`cover_transition` remains as a backwards-compatible boolean; `cover_fill`
takes precedence.

The run config forks from the common random ep25 checkpoint and otherwise
retains the envelope/oracle block count, scale, aspect ratio, curriculum
window, optimizer, EMA, batch, and accumulation settings. No
`pred_target_k` is set because rectangular targets already have batch-shared
lengths.

## 600-slice audit

`scripts/cover_random_probe.py` used 600 Training slices from 30 volumes, with
20 evenly spaced slices per volume. The three policies received identical
slices and block sizes.

### Stored artifact (`anatomy_tau=0.30`)

| policy | anatomy hidden | anatomy visible | cover blocks | leftover blocks | floor OK | usable |
|---|---:|---:|---:|---:|---:|---:|
| unconstrained random | 86.372% | 13.628% | 2.948 | 1.052 random | 60.0% | 59.83% |
| random legal | 84.425% | 15.575% | 2.948 | 1.052 random | 100% | 99.83% |
| transition | 84.425% | 15.575% | 2.948 | 1.045 transition + 0.007 random | 100% | 99.83% |

The legal and transition policies have identical anatomy coverage. The
unconstrained policy is allowed to hide more anatomy because 40% of samples
fall below the 15% visibility floor. Therefore “zero cost to coverage” applies to
**legal random versus transition**, not to legal versus unconstrained random.

For legal random, anatomy hidden had p10 84.10%, mean 84.42%, and p90 84.96%.
The stored block distribution was:

| legal-random blocks of 4 | slices |
|---:|---:|
| 0 | 92 |
| 1 | 388 |
| 2 | 118 |
| 3 | 1 |
| 4 | 1 |

The four-random case is the single no-anatomy fallback. Total unusable rate was
1/600 = 0.167%.

### Production-threshold parity check (`anatomy_tau=0.10`)

The run config uses `anatomy_tau=0.10`, but the audit script currently
hard-codes `TAU=0.30`. The same 600-case audit was therefore repeated at 0.10
during documentation:

| policy | anatomy hidden | anatomy visible | cover blocks | leftover blocks | floor OK | usable |
|---|---:|---:|---:|---:|---:|---:|
| unconstrained random | 86.247% | 13.753% | 3.215 | 0.785 random | 64.0% | 63.83% |
| random legal | 84.582% | 15.418% | 3.215 | 0.785 random | 100% | 99.83% |
| transition | 84.582% | 15.418% | 3.215 | 0.778 transition + 0.007 random | 100% | 99.83% |

The broader support at 0.10 requires more cover blocks, but the scientific
conclusion is unchanged: legal random preserves the floor and matches
transition coverage; unconstrained random violates the floor on 36% of
samples.

At 0.10, legal-random block counts were 0 on 202 slices, 1 on 327, 2 on 70,
and 4 on the one fallback slice.

## Why the measured hidden rate is below 85%

The shipped values set:

```text
leave_frac = min_visible_frac = 0.15
```

The soft target and hard floor consequently meet at exactly the same boundary.
The greedy phase can only take a discrete rectangle if the resulting soft mass
remains legal, so it normally stops just short of 85%. This is expected rather
than a placement failure.

If exactly 85% hidden is scientifically important, the stop and floor must be
separated—for example, retain an 85% soft target but lower the hard floor—and
the new setting must be re-audited. The current policy prioritizes its
visibility guarantee.

## Edge-case gate

`scripts/mask_edge_case_test.py --fill random_legal` exercises 50 deliberately
hard slices: five depth positions from each of ten volumes, including the
nearly empty volume edges. It calls the production sampler and production block
size generator.

| metric | result |
|---|---:|
| assertion failures | 0 |
| fallbacks | 0 |
| zero-visible anatomy slices | 0 |
| mean strict-occupancy anatomy hidden | 83.39% |
| range | 57.14–88.10% |
| minimum strict-occupancy cells visible | 3 |
| strict-occupancy slices below 15% visible | 13/50 |
| strict-occupancy slices below 4 visible cells | 1/50 |
| block split, 3 cover + 1 random | 31/50 |
| block split, 4 cover + 0 random | 12/50 |
| block split, 2 cover + 2 random | 6/50 |
| block split, 1 cover + 3 random | 1/50 |

The gate's assertions cover zero-visible anatomy, fallback behavior, and a sane
union size. They do not assert that a stricter external anatomy definition has
exactly the same floor as COVER's internal soft support.

### Sparse-slice definition mismatch

For `data_08569/slice_199`:

| definition | anatomy/support cells | visible cells |
|---|---:|---:|
| edge audit: occupancy ≥ 0.25 | 7 | 3 |
| COVER: summed soft score > 0.10 | 9 | 4 |

COVER also leaves 37.2% of its soft anatomy mass visible on that slice.
`floor_violation=False` is therefore correct for the sampler's contract, even
though the stricter occupancy audit sees only three visible cells. This is a
definition mismatch, not an unflagged zero-context failure.

The mismatch is broader for the fractional floor: 13 of 50 slices leave less
than 15% visible under the strict occupancy definition. Only the case above
falls below four strict-occupancy cells, and no case reaches zero. Any paper or
monitoring dashboard must name which anatomy definition its “15% visible”
claim uses.

## What the gates establish

**Established without training:**

- target rectangles retain stock batch-shared shape and count;
- leftover blocks can be uniform while respecting the anatomy floor;
- unconstrained uniform leftovers violate that floor frequently;
- legal random and transition reach the same anatomy coverage;
- the production-threshold rerun and edge-case suite complete without a legal
  floor failure or zero-visible sample.

**Not established:**

- downstream AUC;
- predictor stability;
- whether random leftovers outperform transition leftovers;
- whether the 15% floor is optimal;
- behavior across multiple pretraining seeds.

## Run decision

The arm is ready for a controlled pretraining run **in the implementation
sense**. It should be treated as a new experimental arm, not a validated
improvement. The clean comparison is COVER-transition versus
COVER-then-RANDOM from the same ep25 ancestor, with identical seeds, schedules,
block draws, validation masks, and downstream protocol.

No such training was run during this investigation.

## Artifacts

| artifact | contents |
|---|---|
| `D:\jepa_phase0\reports\cover_random\summary.csv` | aggregate 600-slice results at `tau=0.30` |
| `D:\jepa_phase0\reports\cover_random\per_slice.csv` | per-slice coverage and block provenance |
| `D:\jepa_phase0\reports\cover_random\cover_random_visual.png` | representative legal-random masks |
| `D:\jepa_phase0\reports\edge_cases_random_legal\cover_edge_cases.json` | 50-slice gate summary |
| `D:\jepa_phase0\reports\edge_cases_random_legal\cover_edge_cases.png` | hardest sparse slices |
