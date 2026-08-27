# Adapter artifacts: what they were, what they found, and what to do with them

Investigation only. No paper file was edited, no artifact was moved or deleted,
no training was run. Every number below is labelled MEASURED (read directly from
an artifact or recomputed here), INFERRED (a deduction, with its evidence), or
PENDING (never run).

Scope note: the brief named four directories. The work is larger than that. The
same programme also produced `results/masking/placement/`,
`results/masking/structural_loss/`, `results/masking/class_relations/` and
`results/masking/data_diversity/`, and it is documented in four files under
`docs/experiments/masking/`. Those are included because they are what determines
the verdict.

---

## 1. Verdict first

The adapter experiment is **real, finished, thoroughly documented work that
answered its own question in the negative**. It is not scratch and it is not a
forgotten positive result. Its own headline metrics are **off-topic** for a paper
about masking policy and should not be added to it.

However, the investigation turned up one item that **is** on-topic and is
**not currently disclosed in the paper**: three of the paper's six arms consume a
MIRAGE guide that was modified by one of these adapters, and one arm consumes an
unmodified MIRAGE guide. That is a provenance fact about the controlled
comparison, not a new result.

Per-artifact verdict:

| Artifact | Verdict | Reason |
|---|---|---|
| `results/masking/adapter_stage/adapter_cfg7.pt` | **KEEP** (do not archive) | Legacy-hash identity **3186b1fa278bc97f** matches the guide cache `base512_cfg7_3186b1fa278bc97f` consumed by `configs/patch_mirage_anatomy.yaml`. This is the only committed copy of a **production input to a paper arm**. |
| `results/masking/adapter_sweep/adapter_cfg0..11.pt` (12 files, ~30 MB) | **REMOVE** | Hyperparameter-search weights. Referenced by nothing. Every metric is already transcribed in `sweep.json`, `docs/.../adapter_ablations.md` and `ablation_inventory.md`. Regenerable in 50-62 s each. |
| `results/masking/adapter_sweep/best_eval.npz` (7.6 MB) | **ARCHIVE** | Eval-time dump for cfg-11 only; its scalars are all in `sweep.json`. Keep only if the `adapter_sweep.png` figure is to remain regenerable, which requires it. |
| `results/masking/adapter_sweep/sweep.json`, `adapter_sweep.png` | **KEEP** | Cited by `docs/experiments/masking/adapter_ablations.md`. |
| `results/masking/adapter_stage/adapter_cfg7_ep30teacher.pt` | **ARCHIVE** | Refresh-probe input, superseded. |
| `results/masking/adapter_stage/adapter_stage.json` | **KEEP** | Training record; already transcribed into the inventory. Tiny. |
| `results/masking/adapter_refresh/adapter_refresh.json` | **KEEP** | 571 bytes, already transcribed. |
| `results/masking/adapter_guardrails/adapter_cfg7.pt` | **REMOVE** | Bare `state_dict`, no metadata, referenced by nothing. |
| `results/masking/adapter_guardrails/*.json`, `*.png` | **KEEP** | Cited by `docs/experiments/masking/adapter_ablations.md` (including the 3.88 MB `before_after_20.png`, already confirmed as cited in `REPO_CLEANUP_AUDIT.md:524`). |

Net disk recoverable without losing a cited artifact: about 30 MB of `.pt`
weights plus optionally 7.6 MB for `best_eval.npz`.

**One RECOVER item, and it is a disclosure, not a result** - see Section 7.

---

## 2. What the experiment was

### 2.1 The wiring

MIRAGE (Morano et al., a multimodal retinal segmentation model) is used as the
*anatomy guide*: its softmax over inner-retina and choroid becomes the score map
that the guided mask samplers consult to decide where to place targets.

The question the adapter asks: **can the guide be taught, without any labels, to
describe anatomy in terms the JEPA student already understands?** MIRAGE stays
frozen. A small zero-initialised residual adapter is inserted into MIRAGE's
segmentation path, and the frozen segmentation head reads the *adapted* feature,
so the head's weights never move but its output does.

Quoted from `scripts/adapter_stage.py`, lines 66-84:

```python
class Adapter(nn.Module):
    """cfg 7: depth 2, width 128, alpha 0.5.  Zero-init => identity at step 0."""

    def __init__(self, depth=2, width=128, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        layers = [nn.Conv2d(384, width, 1), nn.GELU()]
        layers += [ResBlock(width) for _ in range(depth)]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, 384, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h0):
        return h0 + self.alpha * torch.tanh(self.out(self.trunk(h0)))
```

The `tanh` bounds the residual and the zero-init makes the adapter an exact
identity at step 0, which the script asserts at runtime
(`assert pre['feature_drift'] < 1e-6, 'zero-init identity broken'`).

### 2.2 The loss - the operator's "loss we did for the adaptor"

There were **two** objectives, in that order.

**Objective 1: relational (Gram) distillation, `L_rel`.** From the
`scripts/adapter_stage.py` docstring, lines 21-23:

```
    H     = H0 + alpha * tanh(A(H0))
    L     = FrozenSegHead(H)
    L_rel = MSE( Gram(pool(H)), sg(Gram(Z_ema)) )
```

and the implementation, `scripts/adapter_stage.py` lines 217-220:

```python
        H = mod(h0(i))
        U = F.adaptive_avg_pool2d(H, (GRID, GRID)).flatten(2).transpose(1, 2)
        loss = F.mse_loss(gram(U), rj(i))
```

with, at lines 88-90:

```python
def gram(x):
    x = F.normalize(x.float(), dim=-1)
    return x @ x.transpose(1, 2)
```

In plain terms: pool the adapted MIRAGE feature to a 16x16 grid of 256 cells,
L2-normalise each cell, and form the 256x256 matrix of pairwise cosine
similarities. Do the same for the JEPA EMA **target** encoder's tokens on the
same image. Minimise the mean squared error between the two matrices. The JEPA
side is a stop-gradient (it is produced under `torch.no_grad`). No labels are
used anywhere. The objective is therefore: *make MIRAGE agree with JEPA about
which patches resemble which other patches*, while leaving MIRAGE's own weights
and its segmentation head untouched.

**Objective 2: class-conditioned structural loss + separation barrier.** This
replaced `L_rel` after the class-relations measurement. Quoted from
`scripts/adapter_structural_loss.py` lines 14-36:

```
   S = { I-I, C-C, I-B, C-B }

   The I-C block -- the one I-JEPA gets wrong -- is excluded, as is B-B ...
   Both sides are therefore z-scored over the safe set per image before
   comparison.

2. SEPARATION BARRIER.  A one-sided hinge protects what MIRAGE already knows:

       delta = (mu_II + mu_CC)/2 - mu_IC
       L_sep = relu(delta_frozen - delta_adapted)^2

    L = L_struct + lambda_sep * L_sep
```

Implementation of the transfer term, `scripts/adapter_structural_loss.py`
lines 122-123:

```python
        ls.append(F.mse_loss(zscore(RM[k][m['safe']]),
                             zscore(RJ[k][m['safe']]).detach()))
```

Each 16x16 cell is given a coarse pseudo-class by frozen MIRAGE itself (inner
retina / choroid / background), so no external labels are needed and the
assignment cannot drift during training. Only the "safe" pair blocks are
transferred; the inner-vs-choroid block is deliberately excluded because that is
the relation JEPA gets wrong. Both sides are z-scored so JEPA's *structure* is
imported rather than its *scale*. The hinge is inactive while the adapted model
separates the tissues at least as well as frozen MIRAGE.

### 2.3 What cfg0..cfg11 varied

A full factorial, `scripts/adapter_sweep.py` lines 264-268:

```python
        dict(depth=d, width=w, lr=lr, alpha=al)
        for d, w in ((0, 64), (2, 128), (4, 128))
        for lr in (1e-4, 1e-3)
        for al in (0.25, 0.5)
```

3 trunk sizes x 2 OneCycle peak learning rates x 2 residual gains = 12 configs.
`alpha` is the residual gain in `H0 + alpha*tanh(...)`; `depth` counts 3x3
residual blocks (depth 0 is the shallow 1x1/3x3/1x1 original). One pass over
6,000 fresh FairVision images (375 steps at batch 16), AdamW, weight decay 1e-4,
gradient clip 1.0. MEASURED from `sweep.json`: `n_images: 6000`, `batch: 16`,
`n_eval: 64`.

The sweep exists because an earlier probe had trained on 24 slices for 400 steps
and reported 76.8% loss reduction. That was memorisation. The sweep is the honest
single-pass replacement; `adapter_sweep.py` says so in its own docstring, line 3.

---

## 3. What it found

### 3.1 Architecture sweep - MEASURED, `results/masking/adapter_sweep/sweep.json`

| cfg | depth | width | peak LR | alpha | params | L_rel reduction | feature drift | seg agreement | mask Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 64 | 1e-4 | 0.25 | 86,528 | 8.61% | 0.0934 | 0.9884 | 0.8672 |
| 3 | 0 | 64 | 1e-3 | 0.50 | 86,528 | 20.56% | 0.1918 | 0.9734 | 0.7296 |
| 7 | 2 | 128 | 1e-3 | 0.50 | 689,664 | **29.92%** | 0.1848 | 0.9713 | 0.7454 |
| 11 | 4 | 128 | 1e-3 | 0.50 | 1,280,512 | **30.32%** | 0.1861 | 0.9716 | 0.7364 |

Ordering of levers, MEASURED across the full 12-row table: alpha is the largest,
learning rate second, depth third. Depth 0 to 2 buys +9.3 pp; depth 2 to 4 buys
+0.4 pp. cfg-7 was selected as the knee (46% fewer parameters than cfg-11 for
0.4 pp less transfer).

`best_eval.npz` (MEASURED, opened here with numpy) stores `ci = 11`, not 7 - the
sweep's "best" selector is pure `L_rel_reduction_pct`, so the saved eval dump is
cfg-11's while the *selected* config is cfg-7. Contents: `L0`/`Lf` logits
(64, 4, 64, 64), masks `m0`/`m1` (64, 16, 16), `eval_idx` (64,), `loss` (375,).
Recomputed from the arrays: loss first-100 mean 0.278929, last-100 mean 0.233078;
mean mask cells 37.078 before, 43.797 after (+18.1%); mask Jaccard mean 0.7364,
min 0.1837.

### 3.2 Guardrails - MEASURED, `results/masking/adapter_guardrails/guardrails.json`

| Test | Result |
|---|---|
| T1 generalisation | train 29.589% reduction vs held-out 29.794%. No gap. |
| T2 budget lock | mean cells frozen 50.56, adapted-free 59.77, adapted-locked 47.17; Jaccard vs frozen 0.8054 free, 0.7365 locked |
| T3 localisation | corr(abs score change, margin) = **-0.7159** (held-out); mean change 0.14307 at uncertain cells vs 0.01399 at confident cells, a 10.2x ratio |

`seg_before_after.json` (MEASURED): 2.946% of pixels change class; the mean
frozen-head margin is 0.5573 where a pixel changed and 0.9837 where it did not.
The adapter acts almost exclusively at boundaries MIRAGE was already unsure about.

### 3.3 Staging - MEASURED, `results/masking/adapter_stage/adapter_stage.json`

Step-0 identity check passed exactly: `feature_drift 0.0`, `seg_agreement 1.0`.
After 300 steps in 55.04 s: `L_rel` 0.391888 to 0.294718, a **24.795%** reduction,
feature drift 0.17726, seg agreement 0.98272, mean score change 0.00814. Teacher
recorded as `patch_mirage_anatomy/jepa_patch_mirage-ep30`, `jepa_sha
f708ff0678d48705`.

INFERRED, with evidence: this JSON describes `adapter_cfg7_ep30teacher.pt`, not
the sibling `adapter_cfg7.pt`. `torch.load` of `adapter_cfg7_ep30teacher.pt`
returns `jepa_sha 'f708ff0678d48705'` (matching the JSON) while
`adapter_cfg7.pt` returns teacher `patch_mirage_envelope/jepa_patch_mirage-ep100`,
`jepa_sha 'dce47ab86fd4f627'` (not matching). File mtimes agree: the JSON and the
ep30-teacher checkpoint are both 2026-08-09 14:23:07, the other is 2026-08-08
21:30:20. The script writes a fixed JSON path and so overwrote the earlier record.

### 3.4 Refresh - MEASURED, `results/masking/adapter_refresh/adapter_refresh.json`

Adapter A (taught by envelope ep100) versus adapter B (taught by anatomy ep30),
on n=192 slices: feature relative L2 0.07895, score absolute difference 0.000944,
segmentation argmax agreement 0.99779, **mask Jaccard 0.98257** (min 0.86885),
53.65% of masks bit-identical, mean cells 53.354 vs 52.979.

Reading: **which** JEPA checkpoint teaches the adapter barely matters. Combined
with the saturation curve in `adapter_ablations.md` (27.7% of achievable
reduction at 2,400 images, 32.5% at 19,200), this is the result that justified
training the adapter **once** and precomputing the guide cache **once**, instead
of re-running MIRAGE every JEPA epoch. That saved a 66-minute cache rebuild per
refresh and is the reason the production pipeline has a static guide.

### 3.5 The decisive later results

`docs/experiments/masking/adapter_placement.md`, raw data
`results/masking/placement/enc_saved.json` (MEASURED): the `h0` tap used by
cfg-7 is dominated everywhere. Moving the tap to the encoder reaches the same
transfer for about 15x less segmentation damage; efficiency (transfer per unit
Dice damage) 70,000 at `enc` alpha=0.05 versus 6,268 at `h0` alpha=0.50, 11x
better. GOALS Dice delta at `enc` alpha=0.05 is -0.00035 (paired t p=0.140,
neutral); at `enc` alpha=0.50 it is -0.01297 (p=3.06e-10, real harm).

`docs/experiments/masking/class_relations.md` (MEASURED, 30 GOALS images, 294
pure cells): inner-retina-vs-choroid discrimination AUC is 0.9773 for the MIRAGE
encoder, **0.6945** for JEPA ep100, and **0.8288** for an *untrained* JEPA.
Pretraining on OCT more than halved the model's ability to relationally
distinguish inner retina from choroid, while *improving* its tissue-vs-background
contrast (+0.506 vs +0.435 untrained). The `L_rel` objective was therefore asking
JEPA to teach MIRAGE the one relationship JEPA is worst at.

`docs/experiments/masking/structural_loss.md` (MEASURED): the class-conditioned
loss fixes the class geometry exactly as designed - at alpha=0.50 Gram-MSE
destroys 61% of the tissue separation while the structural loss preserves 104% -
but it does **not** remove the segmentation damage, and at high alpha it is
worse (Dice delta -0.02042 vs -0.00609). The honest positive: at matched,
neutral Dice cost (alpha=0.10) the structural loss preserves 97% of frozen
MIRAGE's separation versus 84% for Gram-MSE.

### 3.6 Comparison against the paper's frozen-probe AUCs

**PENDING - this comparison does not exist and cannot be made.**

No adapter artifact in the repository contains a downstream glaucoma AUC. Every
adapter metric is an intrinsic one: `L_rel` reduction, feature drift,
segmentation agreement, mask Jaccard, GOALS Dice. There is no adapted-guide
pretraining run and no frozen-guide control to compare it against, so nothing
here can be placed beside random 0.8746, centroid 0.8855 or envelope 0.8807 at
epoch 100. The source says so itself, in `structural_loss.md` under "Caveat":

> None of this establishes that the adapter improves the downstream task. It
> establishes how to adapt without damaging segmentation. Whether adaptation
> helps glaucoma AUC at all remains untested and requires the frozen-guide
> pretraining ablation.

and again in `adapter_placement.md` under "Caveat": "it does not establish that
the adapter earns its place at all."

**Did it work?** Yes as an engineering result and no as a scientific one. The
adapter reliably does what it was built to do (transfers relational structure,
generalises to held-out data, acts only where MIRAGE is uncertain, never touches
the frozen head's weights). It never demonstrated a benefit. Its measurable
downstream effect on segmentation quality is zero at safe settings and negative
at aggressive ones.

---

## 4. Result or dead end - the git and documentation record

Every commit touching these paths, MEASURED via `git log`:

```
2e8b7a2 2026-08-09 fix: encode full identity in adapter filenames
0489384 2026-08-09 fix: address adversarial review - 2 P0, 8 P1, and contradicted doc claims
11577ee 2026-08-09 docs: separate comparison from ablation, disambiguate MIRAGE-guided
fdadcd0 2026-08-09 feat(adapter): class-conditioned structural loss + separation barrier
2bbe199 2026-08-09 feat(analysis): I-JEPA does not separate inner retina from choroid
d1ddcba 2026-08-09 feat(adapter): placement ablation - encoder tap dominates H0
a3e04f0 2026-08-09 feat(eval): matched ep30 anatomy-vs-rectangle comparison
a98f908 2026-08-09 goals: score the adapter against real ground truth for the first time
5ed437d 2026-08-08 pipeline: three matched epochs, adapter stage, and verified guide cache
64c6730 2026-08-08 pipeline: adapter stage, one-time guide precompute, and matched-arm comparison
8ef247d 2026-08-08 masking: fix two region-growth bugs in the anatomy target sampler
```

Work started 2026-08-08 and stopped on 2026-08-09. No commit message says the
direction was abandoned. The reason it stopped is stated plainly in the
documentation instead: the two "Caveat" sections quoted above both say the same
thing - the one experiment that would justify the adapter, a downstream AUC
comparison against a frozen-MIRAGE guide, was never run. The programme then moved
to the six-arm masking-policy comparison that became this paper.

So: **not forgotten, and not a hidden positive.** It is a completed methods
investigation that ended with a well-characterised null and an explicit statement
of the experiment that was never done. Write-ups already exist at:

- `docs/experiments/masking/adapter_ablations.md`
- `docs/experiments/masking/adapter_placement.md`
- `docs/experiments/masking/class_relations.md`
- `docs/experiments/masking/structural_loss.md`
- `docs/experiments/masking/engineering_notes.md`
- `paper/genai4health2026/research/ablation_inventory.md` (8 adapter entries,
  all marked SOLID with per-entry "Paper decision" lines written for an earlier,
  CVPR-scoped plan in which the adapter was part of the contribution)

---

## 5. The one thing that IS relevant to this paper

The paper is a controlled comparison of six masking policies. The adapter's own
metrics do not bear on masking policy. But the adapter **is inside three of the
six arms**, and the paper does not say so.

MEASURED, by recomputing the exact digest function
`scripts/adapter_stage.py::sha` over every committed `.pt`:

| Guide cache in config | Recomputed match | Adapter identity |
|---|---|---|
| `base512_enc_a`**`d4f09adfa9f05f0b`**`_m31a932eef403c3e8_npy` | full-file SHA-256[:16] of `results/masking/structural_loss/adapter_ep100_structl100_a010_e3_lr0.001_n4800.pt` | `torch.load` metadata: `{'ch': 768, 'depth': 2, 'width': 128, 'alpha': 0.1}`, `tap: 'enc'`, `loss: 'structural'`, `lam_sep: 100.0`, `epochs: 3`, `lr: 0.001`, `n_train: 4800`, teacher `patch_mirage_envelope/jepa_patch_mirage-ep100` |
| `base512_cfg7_`**`3186b1fa278bc97f`** | legacy first-1-MiB SHA-256[:16] of `results/masking/adapter_stage/adapter_cfg7.pt` | `torch.load` metadata: `{'depth': 2, 'width': 128, 'alpha': 0.5}`, H0 tap, Gram-MSE, teacher `patch_mirage_envelope/jepa_patch_mirage-ep100` |

(The legacy partial digest is expected: the `sha()` docstring records that it
"used to hash only the first 1 MiB" before commit 2e8b7a2 widened it.)

Which arms consume which guide, MEASURED from the configs:

| Paper arm | Config | Guide | Adapter in the loop |
|---|---|---|---|
| `random` | - | none | no |
| `centroid` | - | none (per-column intensity statistic) | no |
| `envelope` | `patch_mirage_envelope.yaml` | `D:\...\mirage_guides` (`.npz` hard guides) | **no** |
| `anatomy-v1` (ep30) | `patch_mirage_anatomy.yaml` | `base512_cfg7_3186b1fa278bc97f` | **yes** - H0 tap, alpha 0.5, Gram-MSE |
| `anatomy-v2` | `patch_anatomy_v2.yaml` | `base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy` | **yes** - encoder tap, alpha 0.1, structural loss lambda=100 |
| `cover` (all three variants) | `patch_cover_ep25.yaml`, `patch_cover_f021_ep25.yaml`, `patch_cover_random_ep25.yaml` | same as anatomy-v2 | **yes** |

How much this moves the delivered guide, MEASURED:

- Encoder-tap alpha=0.10 adapter (the anatomy-v2 / cover guide) versus no
  adapter: guide Jaccard **0.9587** over 600 stratified FairVision slices
  (`results/masking/structural_loss/cadence_guide_jaccard.json`, `frozen`/`ep100`
  cell). About 4% of guide cells differ. Small.
- H0-tap cfg-7 alpha=0.50 adapter (the anatomy-v1 guide) versus no adapter:
  mask Jaccard **0.7454** and mean mask cells 37.1 to 43.9, a **+18%** budget
  expansion (`sweep.json` cfg-7 row, corroborated by the `best_eval.npz` arrays
  recomputed above, and by the comment at `src/masks/anatomy.py:202-206`). Not
  small.

Two consequences, stated neutrally:

1. The paper describes the guided arms as using "MIRAGE" without qualification
   (`main_submission.tex` lines 224, 235, 861). Three arms in fact used a MIRAGE
   whose decoder features had been perturbed by a JEPA-distilled adapter. This is
   a reproducibility gap, not an error in any reported number.
2. The `anatomy-v1` guide additionally carried an **18% larger mask budget** than
   the frozen guide. The paper already argues at length (Section 5.2,
   Figure `fig:geom`) that the anatomy arms are confounded on masking ratio,
   context budget and predictor loss slots simultaneously. Adapter-induced budget
   expansion is a *named, measured mechanism* for part of that confound, and one
   of the arms it affects is the ep30 `anatomy-v1` arm that supplies the paper's
   only positive anatomy contrast.

This is already known internally: `docs/experiments/masking/crop_and_precision_audit.md`
"Finding 8 - three different MIRAGE guide sets are in use (pre-existing
confound)" records exactly this table and calls it "a fact requiring follow-up".
It has simply never reached the manuscript. Grep of `main_submission.tex` for
`adapt(ed|er|ation)`, `soft guide`, `guide cache` and `distill` returns nothing
relevant (the single `adaptation` hit at line 1683 is about representation
adaptation to the masking policy).

---

## 6. Is the adapter result itself relevant to the paper?

**No.** Judged against the three tests in the brief:

- *Does it bear on masking policy?* Only indirectly, via provenance (Section 5).
  The adapter is not a masking policy and was never compared as one.
- *Does it bear on the frozen-probe protocol?* No. It never produced a probe AUC.
- *Does it bear on the background/anatomy question?* The `class_relations`
  finding is genuinely adjacent - JEPA's representation merges inner retina and
  choroid while sharpening tissue-versus-background, which is a plausible
  mechanism for why anatomy-*precise* masking adds nothing over
  aiming-at-tissue. But it is measured on 30 GOALS images with a segmentation
  model as reference, not on the paper's cohort or its probe, and it is an
  attribute of the *encoder*, not of any masking policy. Presenting it would
  require its own methods paragraph, its own caveat about the missing oracle
  ep100 checkpoint, and its own defence of the untrained-control argument.

The paper is at a hard page limit. Adding the adapter would cost roughly a
column for a result whose own documentation says its central question is
untested. That is a bad trade.

---

## 7. What I would add, and exactly where

One RECOVER item, and it is disclosure rather than a finding. It costs about
three lines of the appendix and closes a real reproducibility gap.

**Where:** Appendix `\section{Reproducibility and numeric provenance}`
(`main_submission.tex`, `\label{app:repro}`, line 1637), as a new
`\paragraph{Guide provenance.}` after the existing checkpoint-hashing paragraph.
Second choice: Appendix `\section{Mask-geometry provenance}` (`\label` at line
848), adjacent to the existing sentence about the guides existing for the
Training split only. Not the body.

**What it should say**, in substance (exact wording is the author's; every fact
below is MEASURED and cited above):

> The three segmenter-guided arms do not share one guide. `envelope` consumes the
> original MIRAGE hard-envelope guides. `anatomy-v2` and `cover` consume a cached
> soft guide produced by MIRAGE with a frozen, zero-initialised residual adapter
> at the encoder tap (residual gain 0.1), and `anatomy-v1` consumes an earlier
> cache produced with a larger-gain adapter at the final feature map. The adapter
> is trained without labels to align MIRAGE's patch-similarity structure with the
> pretrained encoder's; MIRAGE's weights and its segmentation head are frozen
> throughout. Relative to an unadapted guide the low-gain variant changes about
> 4% of guide cells (Jaccard 0.959 over 600 slices); the higher-gain variant used
> by `anatomy-v1` enlarges the mean guide occupancy by 18%, which adds a further
> axis to the masking-ratio confound already discussed in Section 5.2.

**What I would not add:** the 12-config sweep table, the guardrail tables, the
saturation curve, the placement ablation, the structural-loss tables and the
class-relations tables. All are solid; none is about masking policy; all are
already written up in `docs/experiments/masking/`.

**Follow-up that is out of scope for this paper** but worth recording: the
experiment that would settle the adapter is a pretraining run using the
*unadapted* MIRAGE guide, matched to `anatomy-v2` on every other axis, probed
under the same frozen-probe protocol. That is one training run, and it would also
retire the provenance caveat above. It is PENDING and should not gate this
submission.

---

## 8. Reproduction notes for this investigation

Commands used, all read-only:

- `git --no-pager log --oneline -- results/masking/adapter_* scripts/adapter_*.py`
- `D:\jepa_phase0\.venv\Scripts\python.exe` to open `best_eval.npz` with numpy and
  to `torch.load` each `.pt` for its metadata block
- the same `hashlib.sha256` digest used by `scripts/adapter_stage.py::sha`,
  computed both full-file and over the first 1 MiB, over every `.pt` under
  `results/masking/`, to match the guide-cache tags against on-disk checkpoints

Nothing under `D:\jepa_phase0` or `C:\jepa_data` was written. No paper file was
opened for editing. No artifact was moved or deleted.
