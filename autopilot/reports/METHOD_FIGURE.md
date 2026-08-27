# METHOD_FIGURE — the method overview figure (Figure 1a)

Date: 2026-08-27. Paper: `paper/genai4health2026/main_submission.tex`.

## What changed

`fig:policies` was a single-image float (`fig1_policies_compact.png`, 0.47\linewidth). It is now a
**two-panel float in the same slot**: panel (a) is a new inline **TikZ** schematic of the pipeline,
panel (b) is the unchanged policy image at 0.35\linewidth. No float was added; the policy panel was
not dropped. The caption now describes both panels. Three `\ref{fig:policies}` call sites were
narrowed to `\ref{fig:policies}(b)` (they refer to the B-scan renderings) and one sentence in
Background points at panel (a). Nothing else in the manuscript was touched; no digit was altered.

Measured float cost (compiled in the NeurIPS style, top-of-figure to bottom-of-caption):

| version | figure + caption |
|---|---|
| before (image only, old caption) | 200.8 pt |
| after (two panels, new caption)  | 193.5 pt |

The new float is 7.3 pt **shorter** than the one it replaces, so the body stays at exactly 9 pages.

## Every component drawn, and the code that confirms it

| drawn element | label in the figure | confirmed at |
|---|---|---|
| B-scan input, 2D slice, 256x256 | "B-scan" / grid icon | `configs/patch_vitb16_ep100.yaml` `data.crop_size: 256`; `src/train_patch.py:379` (`slice_size=crop_size`); `src/datasets/oct_slices.py:159-186` returns one `(3, 256, 256)` slice |
| 16x16 token grid = 256 tokens | "$=16{\times}16$ tokens", "all 256 tokens" | `src/masks/multiblock.py:51-53` (`height = input//patch`, `num_patches` "256 for 16x16"); `configs/patch_vitb16_ep100.yaml` `mask.patch_size: 16` |
| policy places M=4 target blocks | "$M{=}4$ targets (dark)" | `configs/patch_vitb16_ep100.yaml` `mask.num_pred_masks: 4`; `src/train_patch.py:243`; `src/masks/multiblock.py:45,148-153` |
| one context block, excluding the targets | "one context block (light)" | `mask.num_enc_masks: 1`, `allow_overlap: false` in the same config; `src/train_patch.py:242`; `src/masks/multiblock.py:155-180` (context indices with target indices removed) |
| context encoder, ViT-B/16, visible tokens only | "context encoder $f_\theta$ / ViT-B/16, visible tokens" | `meta.model_name: vit_base`, `mask.patch_size: 16`; `src/helper.py` `init_patch_model`; `src/models/vision_transformer.py:449-455` (pos-embed added, then `apply_masks` selects the visible subset) |
| predictor, 6 blocks, width 384 | "predictor $g_\phi$, 6 blocks, $d{=}384$" | `meta.pred_depth: 6`, `meta.pred_emb_dim: 384`; `src/helper.py` builds `VisionTransformerPredictor(predictor_embed_dim=pred_emb_dim, depth=pred_depth)` |
| predictor conditioned on a shared mask token plus the positional embedding of each target site | "mask token $+$ target pos." | `src/models/vision_transformer.py:511` (single learnable `mask_token`), `:520` (`predictor_pos_embed`), `:586-592` (`pos_embs = apply_masks(pos_embs, masks)`, `pred_tokens = mask_token.repeat(...) + pos_embs`), `:604-605` (only mask-token outputs are projected back) |
| context embedding also carries its own positional embedding | (implicit, arrow ctx -> predictor) | `src/models/vision_transformer.py:580-582` |
| EMA target encoder, gradient-free copy of the encoder | "EMA target encoder $f_{\bar\theta}$" | `src/train_patch.py:218-220` (`copy.deepcopy(encoder)`, `requires_grad = False`) |
| EMA update (dashed arrow, "EMA") | dashed ctx -> tgt | `src/train_patch.py:767-780`: on optimiser steps, `p_target.mul_(m).add_((1-m) * p_online)`, with `m` from `momentum_schedule(ema_start, ema_end, total_steps)` (`:523-525`) and `optimization.ema: [0.996, 1.0]` in the config |
| target encoder runs on the full, unmasked grid; features LayerNormed | "all 256 tokens, LayerNorm" | `src/train_patch.py:713-716` (`h = target_encoder(imgs)` with no mask, then `F.layer_norm(h.float(), ...)`) |
| targets are taken at the M masked sites | "at the $M$ target sites" | `src/train_patch.py:720-722` (`apply_masks(h, masks_pred)`, then `repeat_interleave_batch`) |
| smooth-L1 regression loss | "smooth-$L_1$ loss" | `src/train_patch.py:730` and `:734` (`F.smooth_l1_loss(z, h_rep)`); val path `:554` |
| downstream: the frozen encoder is the EMA target encoder | "frozen $f_{\bar\theta}$" | `src/eval_downstream.py:527` (`encoder.load_state_dict(ckpt['target_encoder'])`), `:530-532` (`requires_grad=False`, `.eval()`); `configs/frozen_meanpool_*.yaml` `freeze_encoder: true` |
| mean-pool over patch tokens within a B-scan | "mean-pool patch tokens" | `src/eval_downstream.py:390-391` (`out.mean(dim=1)` over the patch axis), also `:1021` in the end-to-end path |
| mean-pool over the 100 B-scans of the volume | "then 100 B-scans" | `probe_type: mean_pool` in `configs/frozen_meanpool_*.yaml` -> `src/models/attentive_pool_minimal.py:96-119` (`x.mean(dim=1)` over the slice axis, zero parameters); `data.num_slices: 100` in the same config |
| LayerNorm + linear head | "LayerNorm $+$ linear head" | `head_type: linear` -> `src/eval_downstream.py:183-192` (`LinearHead` = `nn.LayerNorm(in_dim)` then `nn.Linear(in_dim, 1)`) |
| volume-level glaucoma label | "volume-level glaucoma label" | `src/datasets/oct_volumes.py` `OCTVolumeDataset(..., return_label=True)` used at `src/eval_downstream.py:369-372`; `nn.BCEWithLogitsLoss` at `:619` |

## Deliberate simplifications (drawn schematically, not literally)

These are true of the drawing, not defects in the code; recording them so nobody reads the icon as a
sampled mask:

1. **The grid icon is illustrative, not a sampled mask.** The light region is drawn as one large
   rectangle with four dark blocks on it. In the sampler the context block is a rectangle covering
   `enc_mask_scale` = 0.85-1.0 of the grid and target indices are then removed from it
   (`src/masks/multiblock.py:155-180`); target blocks are drawn from `pred_mask_scale` = 0.15-0.2
   with aspect ratio 0.75-1.5, so their real sizes vary batch to batch.
2. **Block *sizes* are shared across a batch, locations are per image**
   (`src/masks/multiblock.py:120-134`). Not drawn.
3. **`min_keep=10` and the fallback placement path** (`src/masks/multiblock.py:172-180`), and the
   per-group truncation to a common index count (`:186-199`), are not drawn.
4. **Guided arms use a different collator.** The schematic shows one generic "policy" step. The
   guided arms build masks in the dataloader workers via `MirageMaskCollator`
   (`src/train_patch.py:400-412`) from a precomputed guide, while `random` uses `MaskCollator`. The
   ladder itself is panel (b), so the schematic stays policy-agnostic.
5. **Probe input normalisation** (`imagenet_normalize`, `src/eval_downstream.py:388`) and feature
   caching are not drawn.
6. **EMA momentum values** (0.996 -> 1.0) are verified but were left off the drawing for space; the
   arrow is labelled only "EMA".

## Things worth flagging to the author

* **Which encoder is frozen downstream.** The Evaluation paragraph says only "The encoder is
  frozen". The code loads `ckpt['target_encoder']` (`src/eval_downstream.py:527`), i.e. the EMA
  branch, not the online context encoder. The schematic therefore labels the downstream box
  `frozen $f_{\bar\theta}$`. This is new information relative to the prose; it is not a
  contradiction, but you may want a half-sentence in Section 3 to match.
* **Nothing in the requested component list was unverifiable.** Grid size, M, the EMA update rule,
  the predictor conditioning and the probe pooling were all confirmed in code and configs, as cited
  above. No component was drawn from memory and none was quietly omitted.
* The only value in the figure that is not literally repeated elsewhere in the paper is the
  predictor width `d=384` (`meta.pred_emb_dim: 384`); the paper already states patch size 16,
  predictor depth 6, the 16x16 grid, 256 cells, M=4 and 100 B-scans.

## Style and build

* Vector TikZ, inline in `main_submission.tex`; only `\usepackage{tikz}` and
  `\usetikzlibrary{arrows.meta}` were added, both already resolvable by the pinned Tectonic bundle.
* Greyscale only in panel (a): `black!10` context, `black!45` targets, black rules. No colour-only
  encoding; the arm palette in panel (b) is untouched.
* Text stays selectable and Type 1: the published PDF reports **255 Type 1 font instances and 0
  Type 3** (baseline before the change: 248 Type 1, 0 Type 3). TikZ added no new font programs.

## Gate results (from `C:\Users\Gary\Desktop\jepa`)

| gate | result |
|---|---|
| `autopilot\p13_build_zip.py` | 6/6 PASS, main content **9 pages** (limit 9), References heading at y=72.79 on page 10, `ALL_PASS = True` |
| `autopilot\check_manuscript.py` | `RESULT: PASS`, labels 56, refs 56, **dangling 0**, 0 undefined macros, 0 missing citations (2 pre-existing warnings, unchanged) |
| `autopilot\p15_verify_numbers.py` | `RESULT: PASS` |
| Type 3 fonts in `main_submission.pdf` | **0** |

**Final page count: 9 pages of main content (36 pages total including references and appendices).**
Not committed.
