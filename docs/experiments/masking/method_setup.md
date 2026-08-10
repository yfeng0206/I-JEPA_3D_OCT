# Masking method and setup

This page defines the masking modes used in the submission notes. The important distinction is not whether MIRAGE is used, but what MIRAGE is used for.

| mode | config | MIRAGE role | target shape | status |
|---|---|---|---|---|
| `random_default` | I-JEPA default | none | rectangles | baseline |
| `random_matched` | area-matched random control | none | rectangles | control only |
| `mirage_envelope` | `configs/patch_mirage_envelope.yaml` | places rectangles on the retinal envelope | rectangles | baseline |
| `mirage_anatomy` | `configs/patch_mirage_anatomy.yaml` | shapes connected targets to tissue support | connected irregular anatomy blobs | contribution |

The baseline can already use MIRAGE for placement. The contribution is therefore **shape**: replacing rectangular targets with connected anatomy-shaped targets. MIRAGE is the frozen MultiMAE-based OCT foundation model from arXiv:2506.08900 / npj Digital Medicine 2025; it is not CLIP-trained.

## A. Method and Design Decisions

### Why anatomy-shaped connected targets

![Three masking methods: random blocks, guided rectangles, anatomy shapes](../../../results/masking/explain/three_methods.png)

*Left: random blocks. Centre: guided rectangles (legacy). Right: anatomy-shaped
connected targets. The anatomy method places targets directly on tissue
structure rather than enclosing it in rectangles.*

![The v2 sampler with mass_cap=0.90 and the real I-JEPA context policy](../../../results/masking/demo/v2_masking_cap090.png)

*Production anatomy sampler (v2) at `mass_cap=0.90`. Four connected targets
cover inner retina and choroid; the I-JEPA context block (orange) retains the
majority of the image.*

Anatomy is a minority of an OCT B-scan (~17.6% of grid cells). A rectangular
target wastes most of its area on vitreous/sclera. Anatomy-shaped targets
concentrate the prediction budget on tissue, leaving more context visible.
Measured over **1,000 slices** (commit 8ef247d):

| metric | RANDOM rect | ANATOMY (`mass_cap=0.90`) |
|---|---:|---:|
| target union cells | 122.6 | **53.4** |
| context after removal | 107.4 | **175.0** |
| inner retina masked | 51.6% | **82.5%** |
| dead targets (no anatomy at `tau>0.10`) | 14.12% | **2.12%** |
| inner/choroid balance ratio | 0.899 | **0.966** |

Connected shapes are required because the JEPA predictor uses positional
embeddings: a disconnected index set leaks shape information through position
rather than through learned representation (rejected design `A'`, §Rejected).

### Mass-cap sweep: why 0.90

Swept over **500 slices** with multi-component growth and the production
collator (commit 8ef247d):

| cap | target cells | inner retina masked | context tokens | zero-retina context |
|---|---:|---:|---:|---:|
| RANDOM | 122.0 | 0.516 | 107.7 | 1.00% |
| 0.80 | 46.1 | 0.725 | 181.3 | 0.80% |
| 0.85 | 50.5 | 0.774 | 177.4 | 0.80% |
| **0.90** | **55.9** | **0.825** | **172.7** | **0.80%** |
| 0.95 | 63.3 | 0.875 | 166.1 | 1.00% |
| 0.99 | 69.2 | 0.885 | 160.7 | **21.00%** |

At 0.99, zero-retina-context rate jumps to 21% — the context encoder sees no
tissue in one-fifth of batches. 0.90 is the operating point where inner-retina
coverage (82.5%) is high and context collapse is absent.

**Known cost:** at the stricter `score>0.50` threshold, 76.2% of slices have no
confident anatomy left in context at `mass_cap=0.90`, versus 2.0% for random.

### Support threshold: τ = 0.10

Standard definition of meaningful anatomy support. The void-class probability
across **25,600 cells** was `1.1e-5` (correlation with corrected score:
0.99999997), so the threshold is not contaminated by the softmax void channel.

### Target count: n = 4

Preserves the I-JEPA four-target task structure. No sweep was run over target
count; this is a compatibility decision.

### Budget lock

`build_targets_fixed_cells` holds cell count to the frozen reference guide.
This separates "the guide moved targets" from "the guide grew the task".
Measured on **192 held-out images** (commit 778791b): free budget gives 59.8
cells vs locked 47.2 from a 50.6-cell reference; mask Jaccard 0.805 (free) vs
0.737 (locked). ~26% of cells relocate in either mode.

### Adapter architecture: why cfg-7

![Adapter architecture sweep: 12 configurations](../../../results/masking/adapter_sweep/adapter_sweep.png)

*Twelve configurations swept over one pass of 6,000 FairVision images
(375 optimizer steps). cfg-7 sits at the depth-2 knee.*

| cfg | depth | width | alpha | peak LR | params | L_rel ↓ | drift | s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 64 | 0.50 | 1e-3 | 86,528 | 20.6% | 0.192 | 52 |
| **7** | **2** | **128** | **0.50** | **1e-3** | **689,664** | **29.9%** | **0.185** | **60** |
| 11 | 4 | 128 | 0.50 | 1e-3 | 1,280,512 | 30.3% | 0.186 | 62 |

Depth 0→2: +9.3 pp. Depth 2→4: +0.4 pp. The knee is at depth 2.
Alpha is the largest lever; LR second; depth third. OneCycle peak 1e-3 beats
1e-4 in every matched pair. AdamW with gradient clipping at 1.0, dropout 0.

**Selected:** cfg-7 — depth 2, width 128, alpha 0.5, OneCycle peak 1e-3,
dropout 0. 689,664 params (46% fewer than depth 4). Held-out generalisation:
29.79% vs 29.59% train (no gap, §Guardrails T1).

**Memorisation warning:** an earlier probe on **24 slices × 400 steps** reported
76.8% L_rel reduction. That was memorisation. The honest single-pass number is
29.9% (cfg-7) / 30.3% (sweep max) over 6,000 distinct images.

### Pipeline

![Full pipeline trace on one real slice](../../../results/masking/pipeline/full_pipeline.png)

![Pipeline tensor shapes at every stage](../../../results/masking/pipeline/pipeline_trace.png)

*End-to-end data flow from paired crop through MIRAGE, adapter, sampler, and
JEPA collator. Every tensor shape is annotated.*

```text
same sampled crop
├── JEPA view:   (B,3,256,256), ImageNet-normalized
└── MIRAGE view: (B,1,512,512), raw per-slice min-max

MIRAGE-Base@512 (95.6M params, ALL frozen, drop_path disabled)
  H0:                        (B,384,64,64)
  H = H0 + 0.5·tanh(A(H0)): (B,384,64,64)   [cfg-7 adapter, 689K params]
  frozen seg head:           (B,4,64,64)
  softmax → P_inner, P_choroid: (B,2,64,64)
  avg pool 4×4 → (B,2,16,16) → detach → grow + partition → 4 target indices
  MaskCollator: original context block − target union
```

The frozen seg head reads the **adapted** `H`, not raw `H0`. Routing into a
separate head was a dead end: 0.000e+00 gradient without labelled L_seg.

The mask is hard and detached: `grad_MIRAGE(L_JEPA) = 0` by construction.
Only L_rel = MSE(Gram(pool(H)), sg(Gram(h_full))) trains the adapter, where
h_full comes from the JEPA EMA target encoder.

![MIRAGE-to-targets: score maps, grown regions, and final targets](../../../results/masking/demo/mirage_to_targets.png)

*MIRAGE score maps → grown connected regions → partitioned targets.*

---

## Historical Precursors

### Legacy MIRAGE-envelope arm (rectangle placement)

![Legacy MIRAGE guide construction](../../../results/masking/mirage_guide_pipeline.png)

![Legacy arms comparison](../../../results/masking/mirage_masking_arms.png)

Policy sweep over **1,000 volumes / 19,987 slices**:

| Method | Target purity | Unique cells | Context |
|---|---:|---:|---:|
| RANDOM | 0.4530 | 112.4 | 107.6 |
| ORACLE | 0.5602 | 101.9 | 116.6 |
| MIRAGE envelope | **0.6320** | 101.7 | 117.2 |

Downstream AUC (FairVision Test, **3,000 volumes**, frozen MeanPool ep100):

| Arm | AUC |
|---|---:|
| random | 0.8746 |
| MIRAGE envelope | 0.8807 |
| **oracle** | **0.8855** |

Mask purity was not a validated proxy for downstream AUC.

![Threshold wiring bug](../../../results/masking/mirage_threshold_bug.png)

![Oracle failure cases](../../../results/masking/oracle_failure_cases.png)

### v1 adapter pipeline

![v1 adapter pipeline](../../../results/masking/v1_demo/v1_adapter_pipeline.png)

*The original v1 adapter routing (before the dead-head finding). Retained for
historical reference.*

![Guide equivalence verification](../../../results/masking/v1_demo/guide_equivalence.png)

*Verified that the v1 guide matches the production envelope within rounding.*

### Phase-1 masking investigation

![Phase-1 masking demo](../../../results/masking/phase1/masking_demo.png)

*Early-stage masking demonstrations before the anatomy sampler existed.*

![Raw native argmax comparison](../../../results/masking/phase1/raw_native_argmax.png)

*Argmax of MIRAGE logits at native 200×200 resolution.*

![v1/v2/v3 raw masking comparison](../../../results/masking/phase1/v1_v2_v3_raw.png)

*Evolution of mask computation: v1 (binary envelope), v2 (two-class soft), v3
(connected anatomy shapes).*

### JEPA-error scorer — geometric confound

![Error vs anatomy analysis](../../../results/masking/error_vs_anatomy/error_vs_anatomy.png)

*Over 20 slices: error correlates with distance to context centroid (+0.57),
not anatomy (−0.27). After controlling for distance and intensity, partial
correlation with anatomy is +0.04.*

### JEPA-teacher sensitivity (circular)

![Epoch-30 vs epoch-100 JEPA-teacher sensitivity](../../../results/masking/jepa_to_mirage/jepa_to_mirage.png)

*Both teachers came from MIRAGE-derived-mask runs. The probe is circular:
sensitivity evidence only.*

---

## Additional Figures

### Sampler demonstrations

![Anatomy masking: targets on real B-scans](../../../results/masking/demo/anatomy_masking.png)

*Multiple real B-scans with anatomy-shaped targets overlaid. Targets follow
tissue structure rather than rectangular bounding boxes.*

![Mask pressure analysis](../../../results/masking/demo/mask_pressure.png)

*Pressure (cells demanded vs cells available) across slices. High pressure
triggers the random fallback path.*

![Sampler comparison: all methods side by side](../../../results/masking/demo/sampler_comparison.png)

*Side-by-side comparison of random blocks, guided rectangles, and anatomy
shapes on identical slices.*

![v2 masking overview](../../../results/masking/demo/v2_masking.png)

*The v2 sampler operating on a batch of real slices.*

### Dataset samples

![FairVision sample](../../../results/masking/sample_fairvision.png)
![GOALS sample](../../../results/masking/sample_goals.png)
![Duke DME sample](../../../results/masking/sample_duke_dme.png)
![AROI sample](../../../results/masking/sample_aroi.png)

*Representative B-scans from each dataset used in the programme.*

### Other supporting figures

![Dataset comparison](../../../results/masking/dataset_compare.png)

*Cross-dataset comparison of anatomy distribution and image characteristics.*

![Merged dataset verification](../../../results/masking/merged_verify.png)

*Verification that the merged multi-dataset loader produces balanced sampling.*

---
