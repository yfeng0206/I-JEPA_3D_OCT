# Literature-to-implementation matrix (Stage C)

Owner: `literature-contract` (Opus 5 xhigh). Baseline `de145d7`; branch
`fix/jepa-delivered-task-audit` (working tip `5a47648` at time of writing).
Compiled 2026-09-04, 16:08-16:30 PDT. Only generic public queries were issued;
no manuscript text, case identifiers or raw OCT data left this machine.

Everything below was read from the **full methods, ablation tables and official
source code** of the pinned versions listed in §1. Nothing is inferred from an
abstract or from a paper's introduction alone. Quoted fragments are short and
carry a location; no prose was copied.

---

## 1. Source registry (exact versions verified through primary APIs)

| Key | Work | Version read | Primary metadata check | Official code read |
|---|---|---|---|---|
| I-JEPA | Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* | **arXiv:2301.08243v3** (submitted 2023-01-19, v3 2023-04-13) | **Venue of record: CVPR 2023** — OpenAlex conference record `doi:10.1109/cvpr52729.2023.01499`, pp. 15619-15629; preprint record `doi:10.48550/arxiv.2301.08243`. ⚠ The arXiv API `comment` field for v3 reads verbatim *"2023 IEEE/CVF International Conference on Computer Vision"* (no `journal_ref`, no arXiv DOI). **That field is unreliable here and must not set the venue**; our bib entry `assran2023ijepa` already says CVPR and is correct | `facebookresearch/ijepa` @ `52c1ae95d05f743e000e8f10a1f3a79b10cff048` (2023-06-13, the repo's only commit) |
| MAE | He et al., *Masked Autoencoders Are Scalable Vision Learners* | **arXiv:2111.06377v3** (v1 2021-11-11, v3 2021-12-19; comment: "v3: add robustness evaluation") | arXiv API entry `2111.06377v3`; OpenAlex `doi:10.48550/arxiv.2111.06377`, CVPR record `doi:10.1109/cvpr52688.2022.01553` | `facebookresearch/mae` @ `efb2a8062c206524e35e47d04501ed4f544c0ae8` (2022-04-20) |
| DSeq-JEPA | He, Sakai et al., *DSeq-JEPA: Discriminative Sequential Joint-Embedding Predictive Architecture* | **arXiv:2511.17354v4, updated 2026-08-04**, arXiv comment "Accepted to ECCV 2026", project page `https://dseqjepa-project.com` | arXiv API entry `2511.17354v4`; OpenAlex `doi:10.48550/arxiv.2511.17354` (preprint record, still v1-era metadata) | **`SkyShunsuke/DSeq-JEPA` @ `123082374389f78e4516157187681f9d5edb4310` (2026-08-31, MIT)** — official implementation, found by code search, not linked from the PDF. Files actually read (there is **no** top-level `pretrain.py`; a root-level path 404s): `src/frameworks/dseqjepa/pretrain.py`, `src/frameworks/dseqjepa/attention.py`, `configs/dseqjepa/vitb16_in1k.yaml`. Raw URL form used: `https://raw.githubusercontent.com/SkyShunsuke/DSeq-JEPA/123082374389f78e4516157187681f9d5edb4310/src/frameworks/dseqjepa/pretrain.py` |
| SemMAE | Li et al., *SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders* | **arXiv:2206.10207v3** (NeurIPS 2022) | arXiv API entry `2206.10207v3`; OpenAlex `doi:10.48550/arxiv.2206.10207` | not read (repo `ucasligang/SemMAE` exists; not needed for the claims below, which come from the paper's own algorithm box and ablation table) |
| AnatMAE-Aneurysm | Ceballos-Arroyo et al., *Anatomically-guided masked autoencoder pre-training for aneurysm detection* | **arXiv:2502.21244v1** (2025-02-28; WACV 2026 in our bib as `ceballosarroyo2026aneurysm`) | arXiv API entry `2502.21244v1`; OpenAlex `doi:10.48550/arxiv.2502.21244` | none located |

Version caveats that matter for the manuscript:

* **DSeq-JEPA changed since the version this project previously inspected.** The
  earlier note in the plan cites `arxiv.org/html/2511.17354v1`. The current
  record is **v4** (ECCV 2026 camera-ready-style, 21 pages incl. supplement).
  All DSeq-JEPA numbers below are v4 numbers. `arxiv.org/html/2511.17354v4`
  served only Section 1 to the fetcher, so the full text was obtained by
  downloading the v4 PDF (8,235,132 bytes, 21 pages, 53,325 characters of
  extracted text) and reading it headlessly with PyMuPDF. Nothing here comes
  from the introduction.
* OpenAlex still shows the DSeq-JEPA preprint with 2025/v1-era metadata and
  `cited_by 0`; the arXiv API is the authoritative record for v4/ECCV 2026.
  If we cite it, the entry should say ECCV 2026 with the v4 date, or stay a
  `@misc` preprint entry with an explicit version. (Bib edits are the
  coordinator's; this file only records the evidence.)

---

## 2. The matrix

Column "OURS" describes the delivered system only as needed for contrast; it is
not a literature claim and defers to the engineering owners' measurements.

### 2.1 Importance definition and guide source

| Dimension | I-JEPA | MAE | DSeq-JEPA (v4) | SemMAE | AnatMAE-Aneurysm | OURS |
|---|---|---|---|---|---|---|
| Is there an importance notion? | **No.** Location is uniform-random; "semantic" refers to *block size*, not content | **No.** "we sample random patches without replacement, following a uniform distribution" (p.3, Masking) | **Yes** — attention-derived saliency, explicitly called "a proxy for visual importance" (abstract, p.1) | **Yes** — learned semantic *parts* | **Yes** — anatomical proximity to arteries | **Yes** — segmentation-derived / intensity-derived retinal tissue |
| Guide source | n/a | n/a | The **EMA target encoder itself**: similarity between an auxiliary `[CLS]` token and patch embeddings at block *l* (default layer 10; config `attention.depth: 10`, `strategy: similarity`) | A **separate pretrained network**: iBOT-pretrained ViT-S, then a learned part-attention module trained with a StyleGAN decoder reconstruction loss + diversity loss (§3.1, Eqs. 1-7) | A **separately trained nnU-Net artery segmenter** (mDice 0.89 on held-out set), converted to signed distance maps (§2.1) | MIRAGE-Large GOALS segmentation head → repaired retinal envelope (`src/guides/mirage_envelope.py`), or a segmentation-free per-column intensity centroid (CENTROID) |
| Guide adapts during pretraining? | n/a | n/a | **Yes** — saliency is recomputed every step from the current EMA teacher | No — parts are learned in a frozen first stage | No — fixed segmentation | **Neither "fixed" nor adaptive: computed per transformed view, from a source that does not learn.** CENTROID is computed from *each transformed training image* (per-column intensity-weighted row centroid), so it follows every crop/augmentation. The MIRAGE envelope is cached once in native 200×200 label space but is **carried through the same geometric transform as its image**, so the delivered guide differs per view. Neither guide is updated by training |
| Importance ≙ | nothing | nothing | *saliency of a patch to the teacher's `[CLS]` token at block 10, on natural images* — the authors' own word is **proxy**; "discriminative" is their label for the ranking, not a measured property of the content | *object-part membership* | *anatomical proximity to the lesion-bearing structure* | *tissue presence / anatomical class* |

Region extraction, verified in code (`SkyShunsuke/DSeq-JEPA:src/frameworks/dseqjepa/attention.py:50-90`)
and matching the paper (p.6, Eqs. 1-3): min-max normalise saliency → **Otsu**
threshold → `cv2.connectedComponents(..., connectivity=8)` → drop components with
`area < alpha * N` (`alpha: 0.15`, i.e. the paper's `|R_k| < 0.15hw`) → score each
region by mean normalised saliency → sort descending → keep **top N−1** → region
`R_N` is the **complement** ("Last Region R_N (Background)", line 87-90).

### 2.2 Objective, teacher and normalisation

| Dimension | I-JEPA | MAE | DSeq-JEPA | SemMAE | AnatMAE-Aneurysm | OURS |
|---|---|---|---|---|---|---|
| Prediction space | latent (target-encoder output) | **pixels** (optionally per-patch normalised) | latent | **pixels** (follows MAE) | **pixels** — CT intensities **plus** an artery-distance-map channel | latent |
| Loss | paper: "average L2 distance" (p.4, Loss). **Code: `F.smooth_l1_loss(z, h)`** (`facebookresearch/ijepa:src/train.py`, `loss_fn`) — a paper/code divergence worth citing when we justify Smooth-L1 | MSE on masked patches only (`models_mae.py:forward_loss`) | **Smooth-L1 / Huber, δ=1**, stated in Eq. (7) p.8; code `masked_smooth_l1_loss` with `reduction='mean'` over selected positions (`pretrain.py:57-73`) | MSE (MAE recipe) | MSE for both channels | `F.smooth_l1_loss(predictions, targets)` (`src/train_patch.py:139`) |
| Teacher | EMA of context encoder; **sees the full image**; targets are taken by masking the *output* (`train.py:forward_target`) | none (single encoder + decoder) | EMA teacher, **full image**, `forward_with_intermediate(imgs, blocks=[attn.depth])`, then `F.layer_norm` (`pretrain.py:381-384`) | none | none | EMA teacher, full image, `F.layer_norm(h_full…)` (`src/train_patch.py:134`) |
| Target normalisation | `F.layer_norm(h, (h.size(-1),))` over the feature dim | optional per-patch pixel normalisation (Table 1d: 85.4/73.9 vs 84.9/73.5) | same `F.layer_norm` over feature dim | n/a | n/a | same |
| EMA schedule | 0.996 → 1.0 (`configs/in1k_vith14_ep300.yaml`) | n/a | 0.996 → 1.0 (paper A.1(2); config `ema_start/ema_end`) | n/a | n/a | project config |

### 2.3 Context/target structure, order and attention

| Dimension | I-JEPA | MAE | DSeq-JEPA | SemMAE | AnatMAE-Aneurysm | OURS |
|---|---|---|---|---|---|---|
| Targets | M=4 blocks, scale (0.15, 0.2), aspect (0.75, 1.5), sampled **independently and possibly overlapping** | 75 % of patches, i.i.d. | N=5 **irregular, non-overlapping** regions ("we employ irregular, non-overlapping masks", p.7) | patch subsets defined *within/over* 6 parts, fixed 75 % ratio | 75 % of patches, **biased toward artery-adjacent patches** | rectangles (RANDOM/CENTROID/ENVELOPE/COVER) or irregular class-aware blobs (ANATOMY) |
| Context | one block, scale (0.85, 1.0), unit aspect, **minus** any overlapping target region (p.4, Context) | the 25 % unmasked patches | **the union of the already-predicted, more-salient regions only** — everything else is masked out of the encoder (`generate_dseq_masks`, `pretrain.py:36-55`; `context_masks = 1 - masks` with "0 for keep") | the unmasked complement | the unmasked complement | I-JEPA context block, then batch-min truncation (see `src/masks/anatomy.py` header, D1) |
| Prediction order | parallel/flat, permutation-symmetric | parallel (one decoder pass) | **sequential, most→least discriminative**, `ŝ_{R_{k+1}} = g_θ(p_{R_{k+1}}, h_{R_1},…,h_{R_k})` (Eq. 5, p.8) | parallel | parallel | **parallel/flat** |
| Attention restriction | none beyond token dropping | none | **masked / causal attention**: "we perform prediction with masked attention" (`pretrain.py:398`, `:426`), and the cost table attributes the extra memory to "sequential region tokens and causal masking" (p.14) | none | factorised 3D self-attention (an encoder-efficiency device, unrelated to masking) | none |
| Is the top-ranked region ever a target? | n/a | n/a | **No.** `R_1` (highest saliency score) is only ever context; targets are `regions[:,1:]`, i.e. `R_2…R_N`, ending with the **background complement** `R_N` | yes (parts are masked) | yes (artery-adjacent patches are masked) | **yes — the guided arms place targets on guide-positive tissue** |

Structural contrast, stated at factor level only: **DSeq-JEPA keeps its
highest-saliency region in the encoder context and predicts progressively
lower-saliency regions from it, ending on the background complement; our guided
arms place targets on guide-positive tissue.** Read this as a difference in one
design factor between two systems that also differ in guide source, mask
geometry, conditioning, data domain and evaluation — **not** as an inverted
analogy, and not as a claim that one direction is correct. On natural images
`R_1` is a saliency-ranked region, not a region shown to carry the diagnostic
information; the same caution applies to our guide.

### 2.4 Mixture / curriculum rules (the "random vs guided" schedule)

| Method | Rule | Where it lives |
|---|---|---|
| I-JEPA | none (always random) | — |
| MAE | none (always random) | — |
| DSeq-JEPA (paper) | probability λ rises **linearly 0→1 over epochs**; Algorithm 1 (p.7) draws `b_k ~ Bernoulli(λ)` **per region**, replacing a rejected region by a random region `R(u,v,w,h)`; the prose says "each sample uses our discriminative region selection with probability λ" (p.7) | Alg. 1, §3.2 |
| DSeq-JEPA (official code) | **per training step, whole batch**: `use_dseqjepa = (torch.rand(1).item() < lambda_val)`; if false the step runs the ordinary I-JEPA path with the dataloader's random multiblock masks and flat prediction (`pretrain.py:389-391, 448-451`) | `pretrain.py` |
| SemMAE | interpolate the *per-part masking counts* between "mask 75 % of patches inside each part" and "mask whole parts", with `α = (epoch/total)^γ`, γ=2 (Alg. 1, p.6) | Alg. 1 |
| AnatMAE | no schedule; masking is always artery-biased | §2.2 |
| OURS | `r_t` ramped to `r_max`, consumed as a per-pred-block Bernoulli (`src/masks/curriculum.py` header); `r_max=1` removes unguided draws at full ramp | `src/masks/curriculum.py` |

Three consequences for us, all source-verified:

1. **The four schedules are four different interventions on four different
   objects, and none of them is ours.** DSeq-JEPA switches an entire training
   step between two complete pipelines; SemMAE interpolates *how many patches
   per part* are masked; AnatMAE applies an artery-proximity bias on **every**
   step with no schedule at all (a parallel, pixel-reconstruction model running
   permanently guided); ours draws a coin **per predictor block** to decide
   whether that block's *placement* is guided. `r_max=1` removes unguided
   placement draws in our sampler — it is **not** SemMAE's α=1 (mask whole
   semantic parts) and not DSeq's λ=1 (every step takes the sequential path).
   Do not equate them; measure ours.
2. The paper-vs-code granularity of DSeq's Bernoulli differs (per region in
   Algorithm 1 vs per batch in the code). Our per-pred-block Bernoulli is a
   third granularity. This is a real design axis, not a detail.
3. DSeq-JEPA's fallback branch is a *complete* I-JEPA step (random masks **and**
   flat prediction), not "guided masks with a random location". Our mixture
   swaps only the placement.

### 2.5 Data, schedule, evaluation

| Method | Pretraining data | Schedule | Downstream protocol |
|---|---|---|---|
| I-JEPA | IN1k / IN22k | ViT-B/16 600 ep (ablations 300 ep) | linear probe on frozen features; ImageNet-1 % low-shot for all ablations |
| MAE | IN1k | ViT-L/16 800 ep default, 1600 ep headline | fine-tune and linear probe |
| DSeq-JEPA | IN1k, batch 2048, 600 ep (ViT-B/L), 300 ep ViT-H/16@448 | AdamW, lr 1e-4→1e-3 (15-epoch warmup)→1e-6 cosine; wd 0.04→0.4 | linear probe on **concatenated last-4-block tokens** (28 epochs, SGD), MAE-style fine-tune, COCO/ADE20K, CLEVR |
| SemMAE | IN1k | ViT-B 800 ep, batch 4096, 8×8 patches for the headline | linear probe + fine-tune |
| AnatMAE | 6,796 head CT scans (4 public sources) | 100 ep pretrain, 50 ep fine-tune | **lesion-level sensitivity at 0.5 FP/scan**, detection fine-tuning (not a frozen probe) |
| OURS | OCT B-scans | project schedule | frozen encoder + mean-pool probe, patient-level AUC as the primary protocol; the manuscript **also** reports fine-tuned-probe and full fine-tuning diagnostics in its appendices (`main_submission.tex` §"Fine-tuning narrows the observed gap", `tab:finetuned`, and the fine-tuned-probe attribution appendix). Do not write that we report no fine-tuning |

---

## 3. Which ablation isolates what (and which does not)

This is the section the author asked for. "Isolated" means the table changes one
named factor with everything else fixed.

| Ablation | Location | Factor actually isolated | Result | Confounds / limits |
|---|---|---|---|---|
| **Region generation × prediction strategy** | DSeq-JEPA **Table 4**, p.11 (ViT-B/16, IN1k linear probe / iNat21) | region generation (uniform vs discriminative) crossed with prediction strategy (flat vs sequential), **inside DSeq-JEPA's own construction** | uniform+flat (=I-JEPA) **72.4** / 35.9; uniform+sequential **72.3** / 34.9; **discriminative+flat 72.0 / 35.7**; discriminative+sequential **73.5** / 36.4 | Report the numbers; draw **no significance conclusion**. Single-run cells; the paper's only reported seed spread (±0.4 on the headline config) exceeds the 0.4 gap between 72.0 and 72.4. The "discriminative+flat" cell is **not** an I-JEPA-with-a-guide control: it retains DSeq's `[CLS]`-similarity saliency, Otsu/connected-component irregular non-overlapping regions, the complement region `R_N`, masked-attention conditioning, ImageNet natural images and a last-4-block-concat linear probe. It is an **analogous factor combination**, not a reproduction of any other system's configuration |
| Prediction order, regions fixed | DSeq-JEPA **Table 5**, p.12 | ordering only, "keeping the same set of discriminative regions" | Flat **72.0**, Random 71.7, Spatial 72.7, Inverse 71.3, Truncating(Top-3) 73.0, DSeq **73.5** | same |
| `[CLS]` token addition | DSeq-JEPA **Table 6**, p.13 | whether the added token alone helps | I-JEPA 72.4 vs I-JEPA+`[CLS]` 72.4 | good control; rules out the architectural add-on |
| Saliency proxy | DSeq-JEPA **Table 7**, p.13 | `[CLS]`-similarity vs label-free Grad-CAM | 73.5 vs 73.4 (Top-20 patch IoU 0.41 between them) | shows *insensitivity* to the estimator, given the ordering |
| Number of regions N | DSeq-JEPA §4.3 p.13-14 | N ∈ {3,5,7} | 72.9 / 73.5 / 73.4 | |
| **Masking strategy** | I-JEPA **Table 6**, p.8 (IN1k-1 %, ViT-B/16, 300 ep) | *multi-block vs rasterized vs block vs random* | 54.2 / 15.5 / 20.2 / **17.6** | **Does not isolate location semantics.** Every arm is random-location; the arms differ simultaneously in target count (4/3/1/1), target size, contiguity **and** average context ratio (0.25 vs 0.4). It is evidence about *block geometry*, not about *where* to predict |
| Target block scale | I-JEPA **Table 8**, p.14 | target size at fixed count/context scale | (0.075,0.2)→19.2; (0.1,0.2)→39.2; (0.125,0.2)→42.4; **(0.15,0.2)→54.2**; (0.2,0.25)→38.9; (0.2,0.3)→33.6 | larger targets also shrink the *realised* context (targets are removed from it), which the authors acknowledge: gains hold "as long as the context is sufficiently informative" |
| **Context scale** | I-JEPA **Table 9**, p.14 | context block scale only, everything else fixed | (0.40,1.0)→31.2; (0.65,1.0)→47.1; (0.75,1.0)→49.3; **(0.85,1.0)→54.2** | The **cleanest single-factor published result bearing on the context clause**: monotone and large within its own setting. Boundaries: it varies the *sampler's context-block scale* on ImageNet-1 % low-shot linear probing with ViT-B/16 at 300 epochs. It is **not** a measurement of delivered post-collation context, and the magnitude does not transfer to OCT or to patient-level AUC. Cite it as motivation for measuring our delivered context, not as a predicted effect size |
| Number of targets | I-JEPA **Table 10**, p.14 | target count only | 1→9.0; 2→22.0; 3→48.5; **4→54.2** | supports "several targets", relevant to duplicate-vs-unique slot accounting |
| Masking teacher output vs input | I-JEPA **Table 11**, p.15 | where the target mask is applied | output **67.3** vs input 56.1 (ViT-H/16, 300 ep) | confirms full-image teacher access is *intended*, not leakage |
| Predictor depth / width | I-JEPA Tables 12, 14 | predictor capacity | depth 6→64.0 vs 12→66.9; width 384→70.7 vs 1024→68.4 | |
| **Mask sampling** | MAE **Table 1f**, p.5 | sampling family at fixed ratio | random 75 % **84.9 ft / 73.5 lin**; block 50 % 83.9/72.3; **block 75 % 82.8/63.9**; grid 75 % 84.0/66.0 | MAE's own conclusion: "Simple random sampling works the best for our MAE" (p.6). Random masking is the *original* MAE; every guided variant is someone else's modification |
| Masking ratio | MAE Fig. 5, p.4 | ratio only | linear probe peaks at 75 % (73.5); 10 %→54.6, 90 %→66.1 | |
| **Masking strategy (semantic)** | SemMAE **Table 3**, p.7 (ViT-B, 8×8 patches, linear probe) | random vs part-internal vs whole-part vs scheduled | random **66.8**; mask 75 % patches per part (α=0) **66.5**; **mask 75 % parts (α=1) 52.9**; adaptive 1→0 (wrong direction) 63.3; adaptive 0→1: γ=1/3 66.2, 1/2 67.3, 1 67.9, **γ=2 68.7**, 3 68.6 | The cleanest available warning: *masking whole semantic regions is catastrophic on its own* (−13.9); the benefit comes from a schedule that spends most of training near patch-level masking |
| Guide quality | SemMAE **Table 4**, p.8 | which part maps drive masking | baseline (no parts) 63.7; **iBOT parts 63.6**; learned parts 65.0 | a plausible-looking attention guide gave **zero** benefit; the guide, not the idea, carried the effect |
| Anatomy pipeline components | AnatMAE **Table 3**, p.8 | masking / crop-sampling / distance-reconstruction, cumulative | A (all, 6796 scans) 92.9/93.1/90.5/72.6; D (no distance recon.) 91.3/93.1/82.1*/70.8; **E (biased masking only) 82.5*/91.1/78.9*/58.0***; **F (plain MAE) 78.6*/79.2*/72.6*/53.4***; G (no pretraining) 80.2*/82.2*/76.8*/58.9* | E vs F is the closest thing in medical imaging to "anatomy-guided masking alone": it helps. But the authors' own summary is that "the biggest impact results from sampling sub-scans from areas intersecting with vessels" (p.8), i.e. **data selection beat mask selection**; and plain MAE pretraining was *worse than no pretraining* (F 78.6 vs G 80.2). The column flags are cumulative and the header lists four labels for three checkmarks, so read E/F as "masking only" and "nothing" per the accompanying text rather than from the glyphs alone |

---

## 4. What these papers support, and what they do **not** prove for us

### Supported (source-verified)

1. **Where and in what order predictions are made are live design factors in a
   JEPA.** In DSeq-JEPA's own setting the reported gain appears in the cell where
   both are changed together, and the authors attribute it to the combination
   (Table 4). Report that as *their finding in their construction*; do not
   generalise it into "guided selection requires sequencing" — nobody has tested
   that claim across guide sources, mask geometries or domains.
2. **Context block scale matters in I-JEPA's sampler** (Table 9). Within that
   experiment the effect is monotone and large. It supports *measuring* our
   delivered context; it does not predict a magnitude for us.
3. **Predicting several relatively large blocks beats one or few** (Tables 8, 10).
4. **A full-image EMA teacher with output-side masking is the intended I-JEPA
   design** (Table 11) — teacher contextualisation is not leakage.
5. **Guided masking can help in a medical pipeline** (AnatMAE E vs F), and
   **the provenance of the guide changed the outcome** in SemMAE's setting
   (Table 4: iBOT-derived parts 63.6 ≈ no-parts baseline 63.7, vs learned parts
   65.0). Both are single-setting results in their own domains.
6. **Smooth-L1 in latent space is standard practice**, not an idiosyncrasy: it is
   what `facebookresearch/ijepa` actually runs, and what DSeq-JEPA specifies
   (δ=1) and implements.

### Not proven — do not write these

1. ❌ *"MAE and DSeq-JEPA show that masking important regions improves
   representations."* Not supported. Original MAE is uniform random and its own
   ablation prefers random over block/grid (Table 1f). In DSeq-JEPA's Table 4
   the **analogous factor combination** to ours — guided region generation with
   flat prediction — reads 72.0 against 72.4 for uniform+flat. State the numbers
   and stop there: it is *not* our configuration (different saliency source,
   mask geometry, background complement, attention conditioning, dataset and
   evaluation), the cells are single runs, and the gap is inside the only seed
   spread the paper reports. It removes a supposed prior; it establishes nothing
   about our system's sign or size.
2. ❌ *"Guided/semantic masking is a general improvement."* SemMAE Table 3:
   whole-part masking alone is 52.9 vs 66.8 random. Only the scheduled mixture
   wins, and by +1.9.
3. ❌ *"These results transfer to OCT classification with a frozen probe."*
   None of the five works evaluates a frozen probe on 3D-volume-level disease
   classification. AnatMAE is the only clinical one, and it measures lesion
   sensitivity after **end-to-end detector fine-tuning**.
4. ❌ *"Segmentation confidence is diagnostic importance."* DSeq-JEPA's own
   framing is a **proxy** ("serves as a proxy for visual importance", abstract),
   derived from the model's representation, and it is *recomputed* each step. A
   fixed anatomical prior is a different object.
5. ❌ *"Our result must be a bug because the literature says this should work."*
   Read in full, this literature reports **mixed** outcomes for guided masking
   under several different constructions — positive (AnatMAE E vs F; SemMAE's
   scheduled variant), neutral (SemMAE per-part masking; SemMAE's iBOT-derived
   guide) and negative (SemMAE whole-part; MAE block-75). There is therefore no
   published prior that a null or negative guided-masking result *must* indicate
   a defect. That is all this establishes: it neither exonerates nor implicates
   our code, which only the delivered-task audit can address.
6. ❌ Any statement that our CENTROID/ENVELOPE run-level advantage is *explained*
   by these mechanisms. Nothing here licenses a mechanism claim; the positive
   run-level comparisons stand on their own evidence and keep their existing
   limitations.
7. ❌ *"Low-shot ImageNet numbers show that predicting important anatomy helps."*
   Every quantitative example in §3 is ImageNet/iNat/COCO/ADE/CLEVR (or head CT
   detection). None of them measures anatomy, diagnostic content, or a frozen
   volume-level medical classifier. They constrain *design reasoning* about
   masking factors; they are not evidence about learned anatomical importance,
   and they must never appear as numeric support for a claim about our guide.

### Paper↔code divergences worth a sentence in the manuscript or an appendix

| Divergence | Evidence |
|---|---|
| I-JEPA paper states an average-L2 loss; the official code runs `F.smooth_l1_loss` | paper p.4 vs `ijepa:src/train.py` `loss_fn` |
| DSeq-JEPA paper: per-region Bernoulli curriculum (Alg. 1) and per-sample prose; code: one Bernoulli draw per **batch/step** selecting the whole DSeq path vs the whole I-JEPA path | paper p.7 vs `DSeq-JEPA:src/frameworks/dseqjepa/pretrain.py:389-391, 448-451` |
| DSeq-JEPA context sequence **may be** off by one relative to Eq. (5): `generate_dseq_masks` appears to emit keep-sets `[R1, R1, R1∪R2, R1∪R2∪R3]` for targets `[R2, R3, R4, R5]`, because the union is taken over `prev_masks` *before* the current region is appended. **Uncertain**: a static read of ~20 lines, no execution, no author confirmation, and the intended semantics of `logical_or_for_mask` were not checked. Treat as an open question about their code, never as a defect claim, and **do not propagate it into our implementation** — no upstream-derived change to our sampler follows from it | `DSeq-JEPA:src/frameworks/dseqjepa/pretrain.py:43-55` |
| Even DSeq-JEPA's I-JEPA baseline path uses masked attention over the full token grid rather than token dropping ("Different from the I-JEPA implementation, we perform prediction with masked attention") | `pretrain.py:398` |
| I-JEPA's own collator truncates every collated mask row to the batch minimum (`cm[:min_keep_pred]`, `cm[:min_keep_enc]`) — the upstream analogue of our batch-min truncation concern | `ijepa:src/masks/multiblock.py`, `__call__` |

---

## 5. Three discriminating questions for the mask and training owners

Each is answerable on CPU with fixed inputs, and each has a literature-grounded
reason to exist. These are engineering questions, not hypotheses about AUC.

**Q1 (mask/data owner) — How much guide-positive tissue survives into the
delivered encoder context, and how much reaches the loss?**
For a fixed Training-only slice manifest and each arm (RANDOM, CENTROID,
ENVELOPE, ANATOMY v1/v2, COVER), report the joint distribution of
(a) guide-positive tissue inside the **delivered, post-collation encoder
context**, and (b) guide-positive tissue inside the **realised loss slots**,
with the three context sets kept distinct (candidate context block drawn by the
sampler; complement of the target masks; delivered post-collation context).
Motivation, not prediction: I-JEPA's Table 9 shows its own benchmark is
sensitive to *sampler* context scale, and DSeq-JEPA's construction differs from
ours on exactly this axis (its top-ranked region stays in context). We currently
cannot state, per arm and per image, what the encoder actually retains after
block intersection and batch-min truncation. *Deliverable: a per-arm 2-D
histogram with explicit denominators, low-tissue infeasible cases counted
separately, and no cross-domain effect-size borrowing.*

**Q2 (mask/data owner) — What is the realised mixture at full ramp, and what
exactly is the Bernoulli's unit?**
Our `r_t` is a coin **per predictor block** governing whether that block's
*placement* is guided; `r_max=1` removes unguided placement draws after ramp.
That is its own intervention — it is not SemMAE's part-level mask-count
schedule and not DSeq's per-step pipeline switch, and no result from either
transfers to it. Measure, on delivered masks at production microbatch: the
per-step fraction of pred blocks actually guided, the number of post-ramp steps
with zero unguided blocks, the COVER random-fill frequency, and whether the
delivered fraction matches the configured `r_t`. The literature's only usable
contribution here is that *schedule granularity and endpoint are design axes
worth reporting*, so the manuscript should state ours explicitly either way.

**Q3 (training/eval owner) — Are duplicate-weighted loss slots changing the
effective target distribution across arms?**
I-JEPA's target count ablation (Table 10: 1→9.0, 4→54.2) shows the loss is very
sensitive to how many distinct target blocks contribute, and both official
implementations expand targets by `repeat_interleave` before the loss
(`ijepa:src/train.py` `forward_target`; `DSeq-JEPA:pretrain.py:402-407`), while
DSeq-JEPA's masked loss averages over *selected positions only*. For each arm,
report unique masked cells vs duplicate-weighted predictor slots vs the actual
per-slot contribution to the scalar Smooth-L1, on identical crops. If two arms
differ in effective slot count at equal nominal `npred`, the arms differ in more
than placement, and the COVER note in `src/masks/cover.py` (158 vs 64 slots)
says this is already live.

---

## 6. Gaps, uncertainties and what was not done

* **Not verified:** SemMAE's official code (`ucasligang/SemMAE`) was not read;
  all SemMAE statements come from the paper's Algorithm 1 and Tables 1-5.
  AnatMAE has no located public implementation.
* **Ambiguity flagged:** AnatMAE Table 3's header prints four labels
  (`Mod. Mask. Sampl. Reco.`) over three checkmark columns. The row readings
  above follow the surrounding text ("biggest impact… sampling sub-scans";
  "MAE pre-training without any of our changes results in worse performance
  when compared to fully supervised training"). If we cite the E-vs-F contrast
  in the manuscript, cite the sentence, not the glyphs.
* **DSeq-JEPA statistical strength:** three-seed stability is reported only for
  the headline ViT-B configuration (73.8 ± 0.4); Table 4/5 cells are single
  runs. A 0.4-point gap between "discriminative+flat" (72.0) and I-JEPA (72.4)
  is *within* that seed spread. State the direction, not a significance claim.
* **Not attempted:** no OCT- or JEPA-for-OCT-specific literature sweep, no
  attempt to find a paper doing importance-guided masking with a *parallel*
  JEPA predictor on medical volumes. If such a work exists it would be the
  closest prior art to our design; the bounded 4-6 source budget excluded the
  search. Suggested follow-up query for whoever has budget:
  `("joint-embedding predictive" OR JEPA) AND (mask OR masking) AND (anatomy OR
  saliency OR segmentation) AND (OCT OR retina OR medical)` on OpenAlex, 2024-2026.
* **Version risk:** DSeq-JEPA v4 is 2026-08-04 and the repo moved 2026-08-31.
  Any number quoted here should be re-checked against the version we cite in the
  final bibliography before submission.

---

## 7. Source / inference boundary (read this before citing anything above)

| Status | What it covers |
|---|---|
| **Verified from primary source** | Every numeric table value in §3, every quoted phrase with a page or line locator, all version identifiers in §1, and every code behaviour attributed to a named file+commit. These were read from the pinned PDFs and from the pinned repository blobs. |
| **Verified but domain-bounded** | All results are ImageNet-1k/iNat21/CUB/Cars/COCO/ADE20K/CLEVR, or head-CT aneurysm detection. No value in this file was measured on OCT, on retinal anatomy, on a frozen volume-level probe, or on patient-level AUC. Directions and factor structure may inform design reasoning; magnitudes do not transfer. |
| **Static-read observation, uncertain** | The possible one-step lag in `generate_dseq_masks` (§4 divergence table). Not executed, not author-confirmed, and explicitly not a basis for changing our code. |
| **Our-system statements** | Everything in the "OURS" columns is a description for contrast, sourced from repository files (`src/masks/*`, `src/guides/mirage_envelope.py`, `src/train_patch.py`, configs) and owned by the mask/training engineers. Where they measure something different, their measurement wins and this file should be corrected. |
| **Not established anywhere in this file** | Any mechanism for our CENTROID/ENVELOPE run-level advantage; any claim that guide-positive tissue carries diagnostic information; any inference from a diagnostic to downstream AUC; any significance conclusion from DSeq-JEPA's single-run ablation cells. |

## 8. Corrections applied after coordinator review (2026-09-04, 16:35 PDT)

Recorded so the earlier version of this file is not cited by mistake. Each item
was an overclaim by this agent, not a source error.

1. **"Our exact configuration / the losing cell" (DSeq Table 4)** → replaced with
   *analogous factor combination*, plus an explicit list of what differs
   (saliency source, mask geometry, background complement, attention
   conditioning, dataset, evaluation). Numbers retained; **no significance
   conclusion**; "their gain *needs* sequencing" replaced by "the authors
   attribute the gain to the combination, in their construction".
2. **`r_max=1` ≙ SemMAE α=1** → removed entirely. Our per-pred-block placement
   coin and SemMAE's part-level mask-count interpolation are different
   interventions on different objects; no SemMAE number bears on our ramp.
3. **"Nobody in this set runs 100 % guided with a parallel predictor"** → removed
   as self-contradicting: AnatMAE is a permanently guided parallel model, and
   SemMAE's schedule ends at its whole-part endpoint.
4. **I-JEPA venue** → the arXiv `comment` field text is quoted verbatim and
   flagged as unreliable; venue of record recorded as **CVPR 2023** with the
   conference DOI. Re-verified against the live arXiv API response.
5. **"Fixed precomputed guide, identical every epoch" (ours)** → corrected:
   CENTROID is computed from each transformed image during training; the MIRAGE
   envelope is cached in native label space but transformed jointly with its
   image. Also corrected: arm counting is **five policy families / six
   implementations** (ANATOMY v1 and v2 are distinct; a COVER `delivered_v2`
   config additionally exists under the mask owner's active repair and must keep
   a distinct version label); the manuscript **does** report fine-tuning
   diagnostics in its appendices; "nominal vs delivered context" expanded to the
   three distinct sets.
6. **"DSeq shows the most discriminative tissue"** → `R_1` is a *saliency-ranked*
   region on natural images, not demonstrated important content; the structural
   contrast is now stated at factor level with no inverted analogy. Official
   code paths corrected to `src/frameworks/dseqjepa/pretrain.py` /
   `attention.py` (no root-level `pretrain.py` exists). The Eq. 5-vs-code lag is
   marked uncertain and non-actionable for us.
7. **Low-shot numbers as anatomy evidence** → new §4 item 7 forbids it; the
   context-scale finding is kept but bounded to its own sampler and benchmark.
