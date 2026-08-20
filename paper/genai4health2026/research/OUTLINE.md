# Paper outline — GenAI4Health @ NeurIPS 2026 (Track 1, Research Paper, ≤9 pages)

**Status of this document.** This is the integration layer over the four
research agents' outputs. Every slot names its evidence source. Claims are
labelled **[MEASURED]** / **[INFERRED]** / **[ASSUMED]**, and every
**[ASSUMPTION]** the draft is allowed to make is called out explicitly so it
can be checked or removed before submission.

---

## The framing decision (read this first)

The programme was built to show **anatomy-shaped prediction targets beat
rectangles**. **That claim is not supported and must not be the paper.**

- ep30 favours anatomy (+0.0054, Welch p=0.0022) but the error bars are
  **probe-seed technical replicates on one frozen encoder**, and the two arms
  used **different guide caches** (envelope: `.npz` hard guides; anatomy: soft
  guides).
- **ep50 reverses the sign** (anatomy 0.8654 vs envelope 0.8761 = −0.0107).
- **No pretraining-seed replication exists anywhere in the programme.**

Writing the paper as "anatomy masking wins" would be caught by any competent
reviewer, and the venue explicitly warns: *"Work in progress is acceptable only
if the current results already substantiate the central claim."*

**Independently confirmed by the literature review:** even had the result held,
the *framing* was already taken — Ceballos Arroyo et al. (WACV 2026) guide 3D
MAE masking with a pretrained segmenter. So "anatomy-guided medical masking" is
**SCOOPED** on novelty as well as unsupported on evidence. Two independent
reasons to abandon it.

**Therefore the paper's thesis is reframed to what the data actually
substantiate:**

> **How you choose JEPA prediction targets matters — through mechanisms the
> field does not currently measure. We show (i) a latent, spatially-biased
> defect in stock I-JEPA masking that silently destroys context, (ii) that
> target *composition* rather than target *shape* tracks representation
> quality and collapses at near-purity, and (iii) that target policy modulates
> demographic disparity ~2x at matched accuracy, while leaving the
> mild-disease penalty untouched.**

This is honest, novel, and health-relevant — and (iii) is squarely on the
venue's stated topics.

---

## Title (candidates)

1. *What Should a Joint-Embedding Predictor Predict? Target Composition, Context
   Loss, and Equity in Medical Image SSL* ← preferred
2. *Masking Is Not Neutral: Hidden Context Loss and Demographic Disparity in
   Anatomy-Guided I-JEPA*

## Abstract (≈180 words) — slot plan
Frame SSL masking as an unexamined design axis in medical imaging → the crop
defect (quantified) → composition/AUC non-monotonicity with the purity collapse
→ the mechanism (predictor collapse, not anatomy starvation) → the equity result
→ explicit statement that we do **not** claim shape superiority.

---

## 1. Introduction (~1 p.)
- I-JEPA/JEPA in medical imaging; masking policy is chosen by convention and
  almost never audited.
- Three contributions, stated with their limits attached.
- **Explicit non-claim** paragraph: we do not show anatomy-shaped targets beat
  rectangles; we show why that comparison is harder than it looks.
- Evidence: `research/mechanism.md`, `research/subgroup_findings.md`, `EVIDENCE.md`.

## 2. Background & related work (~0.75 p.)
- I-JEPA, MAE, masking-ratio studies; anatomy/segmentation-guided SSL; fairness
  in ophthalmic AI; FairVision.
- **NOVELTY VERDICTS RETURNED** (lit-review agent, 43 verified refs):
  - **C1 crop defect — NOVEL**, phrased as *"we found no prior report."* Stock
    batch-minimum prefix truncation confirmed in upstream
    `facebookresearch/ijepa` commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`,
    `src/masks/multiblock.py:145-175`. Closest threat `mo2024cjepa` (C-JEPA)
    analyses *other* I-JEPA failure modes, not truncation.
  - **C2 composition→AUC — INCREMENTAL.** Closest threat **AnatoMask**
    (`li2024anatomask`, DOI `10.1007/978-3-031-73027-6_9`). Our OCT purity–AUC
    observations appear new, but **we may not claim a universal 40–43% optimum.**
  - **C3 anatomy-guided medical masking — SCOOPED.** Ceballos Arroyo et al.,
    **WACV 2026** (`ceballosarroyo2026anatomymae`, DOI
    `10.1109/WACV61042.2026.00552`) already guides 3D MAE masking with a
    pretrained segmenter. **A "first anatomy-guided medical masking" claim is
    dead** — cite it as prior art and position against it.
- **This independently confirms the reframe above:** the contribution weight
  belongs on C1 (novel) + composition/mechanism + equity, none of which the
  scooping affects.
- Evidence: `research/references.bib`, `research/novelty_assessment.md`.

## 3. Setup (~0.75 p.)
- FairVision glaucoma OCT. **[VERIFIED against the data]** splits
  **6,000 / 1,000 / 3,000** volumes; each sample is a **200x200x200** uint8
  B-scan stack, of which the probe samples **100** slices; test set is
  1,466 positive / 1,534 negative.
  - **[CORRECTION from lit-review]** FairVision and **Harvard-GF3300 are
    distinct datasets**, not aliases (Harvard-GF: 3,300 patients, 2,100/300/900).
    Do not conflate them. *Checked: no file in `paper/` or `docs/` currently
    makes this error.*
  - Data licence **CC BY-NC-ND 4.0** (repo MIT licences cover code only) —
    relevant to the venue's data-use statement.
- ViT-B/16 patch I-JEPA; frozen mean-pool linear probe, probe seed 42.
- All comparison arms fork from **one shared epoch-25 checkpoint** (test AUC
  0.8487) — state this prominently; it is what makes any arm comparison
  meaningful at all.
- Mask policies: random, oracle, envelope, anatomy, blob, COVER(floor).
- Evidence: `configs/`, `src/eval_downstream.py`, `research/dataset_facts.md`.

## 4. Contribution 1 — a spatially-biased context defect in stock I-JEPA (~1.5 p.) **[STRONGEST NOVEL]**
- Mechanism: `src/masks/multiblock.py::_truncate_and_stack` (L216–229) truncates
  every sample to the batch minimum, `t = t[:min_len]`. Indices are row-major,
  so truncation **always deletes from the image bottom**.
- **[MEASURED]** Removes 31–36% of context in *every* rectangle arm, including
  plain random — i.e. this is stock behaviour, not something we introduced.
- **[MEASURED]** Zero-anatomy-in-context rate, B=1 → B=64: random 0.00→4.63%,
  envelope 1.56→10.10%, COVER 2.34→11.02%.
- **[MEASURED]** Accounting balances exactly on failing slices: 256 patches =
  90 targets + 123 withheld + 43 delivered; 65 anatomy = 50 masked + 15
  withheld, **0 reaching the encoder**.
- **[MEASURED]** `blank_proneness.json`: 0% of slices are *always* blank, none
  blank >50% of draws; 52.0% never blank vs 44.8% expected by chance → failure
  is **draw-dependent, not slice-intrinsic**, so bad slices cannot be pre-tagged.
- **Figure 1** (exists) + **Figure 5** qualitative (exists).
- Why it matters: any reported mask composition that is measured pre-collation
  is wrong. This is a measurement-methodology contribution.
- Evidence: `arm_stats/arm_stats.json` (B=64, n=1534) vs `arm_stats_b1/` (B=1).

## 5. Contribution 2 — target composition, not shape, tracks quality (~2 p.)
- Composition table (anatomy hidden %, target purity, context anatomy, blanking,
  context budget) per arm.
  - **[CORRECTION — must be honoured]** random/oracle/envelope/COVER come from
    the **n=6137** sweep; **blob comes from a separate n=1534 pass**. Do not
    describe these as "one identical pass". Source: `research/mechanism.md` §1.
- Purity→AUC at ep50: 31.6→0.8641, 39.7→0.8740, 43.2→0.8761, **97.5→0.8654**.
  - **[RESOLVED GAP]** random ep50 (0.8640971) and oracle ep50 (0.8740299) *are*
    measured — `results/downstream/meanpool_sweep_random/ep50_results.json` and
    `meanpool_sweep_oracle/oracle_ep50.json`. The doc that called them
    unmeasured was stale. **No GPU work needed to complete this table.**
    Re-verified 2026-08-19: both are `probe_type: mean_pool`, `head_type:
    linear`, `num_slices: 100`, `seed: 42` — same protocol as envelope/blob, so
    the four purity points are directly comparable.
  - **[HAZARD — two different "random ep50 AUC" values exist.]** The region
    ablation uses a *separately trained* region-restricted probe
    (`D:\jepa_phase0\reports\downstream_region_auc\{random,oracle}_ep50\region_auc.json`)
    whose `all` test AUCs are **0.8608341** (random) and **0.8682588** (oracle),
    with different early-stopping epochs (43/42 vs 46/47). Both sets are correct
    but **not comparable**. Keep them in separate tables with a footnote, or a
    reviewer will read it as an inconsistency. Note also that the
    anatomy 0.8746555 / background 0.8543614 pair belongs to the **random ep50**
    arm specifically — it is not a global anatomy-vs-background result.
  - **[LIMITATION]** Neither sweep JSON stores per-sample predictions, so random
    and oracle **cannot enter the fairness analysis** (hence 12 arms, not 14).
    Both also ran off-machine (`/tmp/ijepa_checkpoints/...`), so their encoders
    are not available locally and cannot be re-probed or seed-replicated.
- **Figure 2** (exists). **Figure 3** COVER floor dose-response (exists) — the
  unmeasured-AUC axis must stay visually explicit.
- **[LIMITATION, prominent]** 4 completed points, one pretraining seed each,
  several axes moving together → **an association, not an identified optimum**.

## 6. Contribution 3 — why near-pure targeting collapses (~1.25 p.)
The mechanism section, and it overturns the intuitive story:
- **[MEASURED]** Background *positions* carry glaucoma signal: envelope ep50
  background-only pooling reaches **0.8701** vs 0.8730 all-cell.
- **[MEASURED — refutes the obvious explanation]** "Anatomy starvation in
  context" is **not sufficient**: blob has the lowest anatomy *percentage* in
  context (6.26%) but *more* anatomy *cells* (9.97) than envelope (8.63),
  because its context is far larger; blob also has the **lowest** zero-anatomy
  rate (1.24%).
- **[MEASURED] The supported mechanism is predictor collapse.** Within blob:
  full-context error rises **2.76×** (ep30 0.1049 → ep56 0.2895) and the
  anatomy/background marginal token-value ratio falls **3.53× → 0.74×** — the
  predictor stops preferentially using anatomy context.
- **[MEASURED]** Patch attribution: blob is the *only* arm whose background
  contribution separates glaucoma better than its anatomy contribution
  (0.8557 vs 0.8464), despite weighting anatomy patches most per-patch.
- **[INFERRED — moderate]** Near-pure, small (64 slots, `pred_target_k: 16`,
  replacement-padded), connected targets make the task underconstrained; the
  predictor drifts to a positional/prototype solution.
- **[LIMITATION]** Diagnostic, not causal — purity, 2.42× fewer target
  observations, geometry, padding and context size all moved together.

## 7. Contribution 4 — equity (~1.25 p.) **[NEW, ZERO-GPU, ON-TOPIC]**
- Method: order-based join of saved `test_predictions.npz` to FairVision
  demographics, **verified 3000/3000 by label reconstruction** on all 16 probes.
- **[MEASURED]** Black patients worst-served in **12/12** non-retracted probes.
- **[MEASURED]** Racial gap varies **1.97×** across policies (0.0475 COVER ep34
  → 0.0935 blob ep50) at overall AUCs differing by only 0.032.
- **[MEASURED]** Spearman(overall AUC, race gap) = 0.427, **p=0.167** → we
  explicitly **do not** claim accuracy trades off against fairness.
- **[MEASURED]** The `md` trap: mean deviation *defines* the label, so naive
  severity stratification is undefined; corrected analysis scores each positive
  stratum against all negatives.
- **[MEASURED]** Mild-disease penalty **0.1306–0.1394 in all 12 arms** — a
  spread of 0.009. Masking policy does not touch it, and mild+moderate is
  1,132 of 1,466 positives.
- **Figure 6** (built): `figures/fig6_subgroup_disparity.{pdf,png}`.
- Evidence: `research/subgroup_findings.md`.

## 8. Ablations (~0.75 p.) — required by the user
Draw from `research/ablation_inventory.md` (91 catalogued: 57 SOLID / 20
DOC-ONLY / 5 CONFLICT / 4 CONFOUNDED / 5 RETRACTED). Include only SOLID:
probe head (mean-pool 0.8746 / d1 0.8706 / cross-attn 0.8791), COVER floor
sweep (0.15→0.35 hidden/blank curve), structural-loss variant, context-budget
and target-count ablations, region-pooled readout.
**Must carry a visible confound ledger**, including the 5 RETRACTED items.

## 9. Limitations & confound ledger (~0.5 p.) — non-negotiable, there is no rebuttal stage
1. **One pretraining seed per policy**; error bars elsewhere are probe-seed
   technical replicates. State this in the abstract too.
2. ep30 anatomy-vs-envelope is **guide-cache mismatched**; ep50 reverses it.
3. **Retracted COVER campaign** (`enc_truncate: window`, `amp_target: true`) —
   listed and struck, never used as evidence.
4. blob composition is n=1534, others n=6137.
5. Single dataset/split, reused historically; no external cohort.
6. Subgroup n small (Black 431, Asian 251); no individual gap significant.

## 10. Conclusion (~0.25 p.)
Masking policy is a first-class, under-audited design axis with measurable
consequences for context integrity, representation quality, and equity.

---

## Figure inventory
| # | file | status |
|---|---|---|
| F1 crop defect | `figures/fig1_crop_defect.*` | **built** |
| F2 composition vs AUC | `figures/fig2_composition_vs_auc.*` | **built** |
| F3 COVER dose-response | `figures/fig3_cover_floor_dose_response.*` | **built** |
| F4 AUC trajectories | `figures/fig4_auc_trajectories.*` | **built** |
| F5 qualitative failure | `figures/fig5_zero_anatomy_example.*` | **built** (raster) |
| F6 subgroup disparity | `figures/fig6_subgroup_disparity.*` | **built** |

## Build status
`main.tex` compiles with Tectonic 0.17.0 → **8 pages**, 0 undefined refs,
0 overfull boxes. Under the 9-page limit with ~1 page of headroom for §7 and §8.

**Bibliography is DONE and verified (this session).** `references.bib` now holds
**50 entries**: the 43 agent-verified ones (venues + DOIs checked) merged with
the 7 previously-cited entries the verified set did not cover. BibTeX runs clean;
**all 11 current citations resolve, 0 undefined**; 39 verified entries are in
reserve for the expanded draft. Backup of the old 12-entry file: `references.bib.bak`.

Two corrections applied during the merge:
- `luo2023fairvision` was a **duplicate** of `luo2024fairvision` (same paper,
  arXiv 2310.02492). Dropped the old key; `main.tex` updated to cite
  `luo2024fairvision`.
- The old `morano2025mirage` entry gave the venue as an arXiv preprint. MIRAGE is
  **npj Digital Medicine 8:576 (2025)**, DOI `10.1038/s41746-025-01852-3`. Fixed.

## Open assumptions the draft may make (flag, then verify)
- **[ASSUMPTION]** COVER f0.21 ep50 will land between envelope and blob. **Do
  not write a number**; the run is live (epoch 35/100). Leave the cell pending.
- **[RESOLVED]** Citation accuracy — `research/unverified.md` now on disk. Five
  citations were excluded or downgraded; two (`wang2025maskwhatmatters`,
  `balestriero2025lejepa`) were re-verified and are included **as arXiv
  preprints only**, not as peer-reviewed papers. Three remain excluded and
  **must not be cited**: the arXiv-only generalist/specialist paper,
  `wang2025robust`, and `psomas2026attentive`.
- **[RESOLVED]** Contribution framing survived the novelty check, but only in
  its reframed form — see §2. C3 is **scooped**; C1 carries the novelty.
