"""Does the predictor actually LEARN background, or just fall back to position?

The companion probe (``background_signal_probe.py``) found that background target
cells carry HIGHER smooth-L1 error than anatomy cells.  That alone proves
nothing: high error can mean "rich structure the model is still learning" or it
can mean "irreducible speckle noise".  Those two have opposite implications for
whether spending target slots on background is worthwhile.

This script separates them with two reference predictors that use NO context:

  ``mean``  -- always predict the global mean target vector.  Beating this only
               requires knowing that cells differ at all.
  ``pos``   -- predict the per-CELL mean target vector, i.e. the average h at
               that exact grid position over the dataset.  This is the pure
               positional prior: the best you can do from ``pos_embed[j]`` alone
               with no context whatsoever.

The skill score is the fraction of the reference error the real predictor
removes::

    skill = 1 - err_pred / err_reference

``skill_vs_pos`` is the number that matters.  The predictor's query is
``mask_token + pos_embed[j]`` and nothing else, so ``pos`` is exactly the
degenerate strategy of ignoring the context entirely.  If ``skill_vs_pos`` is
~0 on background cells, then background predictions are pure positional prior
and those slots teach the model nothing about the image in front of it -- the
"black here means nothing is nearby" story would be doing no work.  If it is
clearly positive, background genuinely is being predicted FROM context.

Reported separately for background and anatomy cells so the two are comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.multiblock import MaskCollator                     # noqa: E402
from src.masks.utils import apply_masks                           # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402
from scripts.background_signal_probe import load_jepa, CROP, PATCH, GRID, NPATCH, OCC_T  # noqa: E402


@torch.no_grad()
def run(ckpt, ds, idxs, device, args):
    encoder, predictor, tenc, epoch = load_jepa(ckpt, device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    collator = MaskCollator(
        input_size=(CROP, CROP), patch_size=PATCH, enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2), aspect_ratio=(0.75, 1.5),
        nenc=1, npred=4, min_keep=10, allow_overlap=False)

    # ---- pass 1: per-cell mean target vector (the positional prior) --------
    cell_sum = None
    cell_n = 0
    cache = []
    for start in range(0, len(idxs), args.batch_size):
        items = [ds[i] for i in idxs[start:start + args.batch_size]]
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        B = imgs.size(0)
        h = tenc(imgs.to(device))
        h = F.layer_norm(h, (h.size(-1),))
        cell_sum = h.sum(0) if cell_sum is None else cell_sum + h.sum(0)
        cell_n += B
        cache.append((imgs, guides, h.cpu()))
    cell_mean = (cell_sum / cell_n)                       # (256, D) positional prior
    glob_mean = cell_mean.mean(0, keepdim=True)           # (1, D)   global mean

    acc = {k: {"pred": [], "mean": [], "pos": []} for k in ("bg", "anat")}

    # ---- pass 2: real predictor vs the two reference predictors -----------
    for imgs, guides, h_cpu in cache:
        B = imgs.size(0)
        x = imgs.to(device)
        h_all = h_cpu.to(device)
        anat = (guides[:, 0].reshape(B, -1).numpy() >= OCC_T)

        for _ in range(args.draws):
            _, m_enc, m_pred = collator([im for im in imgs])
            v_enc = [m.to(device) for m in m_enc]
            v_pred = [m.to(device) for m in m_pred]

            z = predictor(encoder(x, v_enc), v_enc, v_pred)
            h_t = apply_masks(h_all, v_pred)

            # reference predictions at the same target cells
            cm = cell_mean.unsqueeze(0).expand(B, -1, -1)
            z_pos = apply_masks(cm, v_pred)
            z_mean = glob_mean.unsqueeze(0).expand(B, NPATCH, -1)
            z_mean = apply_masks(z_mean, v_pred)

            e_p = F.smooth_l1_loss(z, h_t, reduction="none").mean(-1).cpu().numpy()
            e_o = F.smooth_l1_loss(z_pos, h_t, reduction="none").mean(-1).cpu().numpy()
            e_m = F.smooth_l1_loss(z_mean, h_t, reduction="none").mean(-1).cpu().numpy()

            for b in range(B):
                for j, m in enumerate(m_pred):
                    idx = m[b].numpy()
                    a = anat[b][idx]
                    r = j * B + b
                    for key, sel in (("anat", a), ("bg", ~a)):
                        if sel.any():
                            acc[key]["pred"].append(e_p[r][sel])
                            acc[key]["mean"].append(e_m[r][sel])
                            acc[key]["pos"].append(e_o[r][sel])

    out = dict(checkpoint=str(ckpt), epoch=epoch, n_slices=len(idxs))
    for key in ("bg", "anat"):
        p = np.concatenate(acc[key]["pred"])
        m = np.concatenate(acc[key]["mean"])
        o = np.concatenate(acc[key]["pos"])
        out[key] = dict(
            n=int(p.size),
            err_pred=float(p.mean()), err_mean=float(m.mean()), err_pos=float(o.mean()),
            skill_vs_mean=float(1 - p.mean() / m.mean()),
            skill_vs_pos=float(1 - p.mean() / o.mean()),
        )
    del encoder, predictor, tenc
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--data_dir", default=r"D:\jepa_phase0\fairvision-glaucoma\data")
    ap.add_argument("--guide_dir", default=(
        r"C:\jepa_data\mirage_soft_guides"
        r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy"))
    ap.add_argument("--slice_cache", default=r"C:\jepa_data\slice_cache")
    ap.add_argument("--split", default="Training")
    ap.add_argument("--volumes", type=int, default=12)
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--slices_per_volume", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\background_signal")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paired = make_paired_transforms(
        crop_size=CROP, crop_scale=(0.3, 1.0), gaussian_blur=False,
        horizontal_flip=False, color_distortion=False, color_jitter=0.0)
    sc = os.path.join(args.slice_cache, args.split)
    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(args.data_dir, args.split),
        guide_dir=os.path.join(args.guide_dir, args.split),
        num_slices=args.num_slices, slice_size=CROP, transform=paired,
        patch_size=PATCH, dilate_patches=0, occupancy_threshold=OCC_T,
        slice_cache=sc if os.path.isdir(sc) else None)

    rng = random.Random(args.seed)
    vols = sorted(rng.sample(range(len(ds.file_paths)),
                             min(args.volumes, len(ds.file_paths))))
    step = max(1, args.num_slices // args.slices_per_volume)
    idxs = [v * args.num_slices + s
            for v in vols for s in range(0, args.num_slices, step)]
    print(f"{len(idxs)} slices from {len(vols)} volumes", flush=True)

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tags = args.tags or [pathlib.Path(c).stem for c in args.ckpts]
    res = []
    for tag, ck in zip(tags, args.ckpts):
        print(f"\n===== {tag} =====", flush=True)
        r = run(ck, ds, idxs, device, args)
        r["tag"] = tag
        res.append(r)
        for key in ("bg", "anat"):
            d = r[key]
            print(f"  {key:>4}  err_pred {d['err_pred']:.4f}  "
                  f"err_pos {d['err_pos']:.4f}  "
                  f"skill_vs_pos {d['skill_vs_pos']:+.3f}  "
                  f"skill_vs_mean {d['skill_vs_mean']:+.3f}", flush=True)
        (out / "skill_scores.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out / 'skill_scores.json'}", flush=True)


if __name__ == "__main__":
    main()
