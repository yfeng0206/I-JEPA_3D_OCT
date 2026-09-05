# Source-backed replacement handoff

**Ready for parent adoption.** No manuscript or existing figure was changed.
The rejected historical attribution assets/reports remain intact; excluding the
ancillary appendix is the parent's separate action.

## New assets

All paths below are in
`autopilot\investigations\delivered_task\evidence\legacy_figure_reviews\replacements`.

| New stem | Formats | Intended paper path |
|---|---|---|
| `fig_purity_auc_ep50_fp32` | PNG, PDF, SVG | `figures/fig_purity_auc_ep50_fp32.png` |
| `fig_policy_family_token_maps` | PNG, PDF, SVG | `figures/fig_policy_family_token_maps.png` |

These are new filenames. Copy them into `paper\genai4health2026\figures` only as
part of the parent's approved replacement, without overwriting the legacy files.
Exact ready-to-use captions are in the matching `.caption.txt` files. The token
receipt is bound to the proposed caption's whitespace-normalized hash; changing
caption content requires a fresh review, not a blind hash update.

## Scatter: what is verified

- Exactly five semantic primary frozen-probe records, all **epoch 50, fp32**.
- Current Table 2 geometry source field `hidden_pct_on_anat`.
- Independent source selection compares every Matplotlib marker offset,
  annotation data anchor, marker shape, and semantic label.
- No fitted line, null line, intervals, or causal/ranking assertion.
- A second render of independently checked artists equals the delivered PNG
  **byte for byte**.

Source table:

| Policy | Purity (%) | fp32 AUC | Geometry pointer | Inventory AUC pointer |
|---|---:|---:|---|---|
| Random | 31.466034477712412 | 0.8641212996544002 | `/random/hidden_pct_on_anat` | `/records/34/auc` |
| Centroid | 40.008725298518314 | 0.8740152718463352 | `/oracle/hidden_pct_on_anat` | `/records/29/auc` |
| Envelope | 43.30093279923807 | 0.8760634352582928 | `/envelope/hidden_pct_on_anat` | `/records/23/auc` |
| ANATOMY-v2 | 97.0935374668334 | 0.8653855047304304 | `/anatomy/hidden_pct_on_anat` | `/records/9/auc` |
| COVER | 44.18566087795465 | 0.864281381901101 | `/cover/hidden_pct_on_anat` | `/records/17/auc` |

The source files and their exact hashes are in `source_manifest.json`.
Inventory pointers are valid for that hash; production selection also checks
arm, epoch, precision, family and status and rejects duplicate matches.

### Numeric-owner integration

`numeric_validator_registration.json` supplies expected semantic series and
executable source expressions for registration in the separately owned
`numeric_plot_review.py`.

New producer: `..\generate_replacements.py`.

```python
fig = producer.build_scatter(geometry_path, inventory_path)
records = verifier.verify_scatter_artists(fig, geometry_path, inventory_path)
rendered_png = producer.png_bytes(fig)
```

The verifier is `..\verify_replacements.py` and independently reads source fields;
it does not call the producer's `scatter_data` to establish expected values.
The numeric owner should still perform its strict source/artist and delivered
PNG equality checks. The registration JSON is a handoff, **not** an invented
numeric-review validation method or a claim that integration has already occurred.
Scatter replay does not load the private fixture and can run without raw OCT.

## Token maps: what is verified

- The original frozen private fixture hash is
  `269f1c143af8e91daa179796fb0bfbd40583d91fafebbc0dbd45cba9c6c4692e`.
- Selection is **BS2, anonymous ordinal zero**, the first of the already
  predeclared audited Training views. There is no case search or mask redraw.
- The same stored view supplies all five families: random, oracle, envelope,
  anatomy, and historical `cover_legacy`. Anatomy is the **v2 representative**,
  not fabricated v1; neither corrected COVER variant is substituted.
- The producer checks the selected mask arrays against the first existing
  `mask_final64_v2\*_bs2.jsonl` rows, including identical crop/guide hashes.
- The independent verifier separately loads the frozen fixture and checks all
  **1,280 displayed token cells**, exact target-union/context membership, and
  all tissue circles against occupancy channel zero at threshold 0.25.
- No image tensor is passed to the plotting function. There are no raster image
  artists, extra clinical/prediction annotations, or three-dimensional axes.
  All PDF/SVG exports contain **zero embedded raster images**.
- A second render of independently checked artists exactly equals the PNG.

The public manifest exports only anonymous selection metadata, source selectors,
hashes, and policy labels—not raw image/guide arrays or subject identifiers.
Raw inputs remain under `.audit`. This is a selected engineering illustration,
not population geometry, clinical interpretation, loss weighting, or performance
evidence. The caption explicitly distinguishes target unions from repeated
loss slots and warns that policy placements are not exactly paired.

`candidate_replacement_reviews.json` contains one genuine
`reviewed_source_illustration` receipt for this new token map. It passes the
actual `numeric_bindings.read_reviews` and `figure_receipt` APIs and remains
`mathematically_verified: false` under that illustration category. It does not
approve the legacy assets or the quantitative scatter.

The existing `..\audit_legacy_figures.py --merge-base ... --merge-candidate ...`
helper may merge this candidate into a new owned output file. It rejects
duplicate source/macro keys and figure/literal identities; never overwrite an
existing receipt to make it pass.

## Validation

**19 targeted tests passed.** The canonical baseline label is **Random** in both
figures and the strict registration handoff; supplied captions are unchanged.
Negative controls reject:

- fp16 values substituted into the fp32 scatter;
- duplicate source records, fitted lines, changed transforms, and wrong PNGs;
- changed token membership, missing tissue circles, misleading legend colors;
- false ANATOMY-v1 labels, hidden clinical text, and even hidden raster artists;
- changed receipt captions.

Headless renderer geometry also found no out-of-figure text or text-rectangle
overlaps. This is not a claim of comprehensive accessibility or venue compliance.

```powershell
$env:MPLBACKEND='Agg'
& D:\jepa_phase0\.venv\Scripts\python.exe autopilot\investigations\delivered_task\evidence\legacy_figure_reviews\verify_replacements.py
& D:\jepa_phase0\.venv\Scripts\python.exe -m unittest discover -s autopilot\investigations\delivered_task\evidence\legacy_figure_reviews -p test_replacements.py -v
```

To deliberately rebuild only this owned output:

```powershell
& D:\jepa_phase0\.venv\Scripts\python.exe autopilot\investigations\delivered_task\evidence\legacy_figure_reviews\generate_replacements.py --force
```

Rebuilding pins current producer/validator hashes in the manifest and regenerates
the proposed receipt. No training, GPU, network, raw-case export, or desktop
viewer is involved.
