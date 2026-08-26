"""CPU-only, no GPU allocation.  Tests the MECHANISM half of H-b at layer 0.

ViT forward is:   x = patch_embed(img) + pos_embed          (src/models/vision_transformer.py:451)
`pos_embed` is a frozen sincos table (requires_grad=False, line 406).

Question: for BACKGROUND patches, how much of the layer-0 token is content and how
much is position?  If background patches are near-identical in content, then the
only thing that distinguishes one background token from another at the encoder
input is `pos_embed`.  That is exactly the PI's proposed mechanism.

Only the patch_embed Conv2d is evaluated (16x16x3 -> 768, 256 patches/slice).
No transformer blocks are run.  Runs on CPU in seconds.
"""
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

torch.set_num_threads(4)
torch.set_grad_enabled(False)

CKPTS = {
    "random_ep50": r"D:\jepa_phase0\checkpoints_hf\random-posfix-100ep\jepa_patch-ep050.pth.tar",
    "random_ep100": r"D:\jepa_phase0\checkpoints_hf\random-posfix-100ep\jepa_patch-ep100.pth.tar",
    "oracle_ep100": r"D:\jepa_phase0\checkpoints_hf\oracle-anatomical-100ep\jepa_patch_oracle-ep100.pth.tar",
}
DATA = r"D:\jepa_phase0\fairvision-glaucoma\data\Test"
MASKCACHE = r"D:\jepa_phase0\reports\anatomy_mask_cache\Test_s100_r256.npz"
N_VOL, N_SLICE, SIZE, P, GRID = 12, 8, 256, 16, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sincos_pos_embed(dim=768, grid=16):
    import sys
    sys.path.insert(0, r"C:\Users\Gary\Desktop\jepa")
    from src.models.vision_transformer import get_2d_sincos_pos_embed
    return get_2d_sincos_pos_embed(dim, grid)


def load_slices():
    files = sorted(glob.glob(os.path.join(DATA, "*.npz")))[:N_VOL]
    idx = np.linspace(0, 199, N_SLICE).round().astype(int)
    imgs, raws = [], []
    for f in files:
        d = np.load(f)
        v = d["oct_bscans"]
        for s in idx:
            a = np.asarray(v[int(s)])
            pil = Image.fromarray(a, mode="L").resize((SIZE, SIZE), Image.BILINEAR).convert("RGB")
            t = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
            imgs.append(t)
            raws.append(np.asarray(pil.convert("L"), dtype=np.float32) / 255.0)
        d.close()
    x = torch.stack(imgs)
    return (x - MEAN) / STD, np.stack(raws), files


def patch_intensity(raws):
    """Mean raw [0,1] luminance per 16x16 cell -> (N, 256)."""
    N = raws.shape[0]
    r = raws.reshape(N, GRID, P, GRID, P).mean(axis=(2, 4))
    return r.reshape(N, GRID * GRID)


pos = torch.from_numpy(sincos_pos_embed()).float()          # (256, 768)
uniq = len(np.unique(pos.numpy().round(6), axis=0))
x, raws, files = load_slices()
inten = patch_intensity(raws)

# background definition from the SAME intensity convention the project uses for
# anatomy masks; also cross-checked against the shipped anatomy_mask_cache below.
bg_lo, an_hi = 0.06, 0.30
bg_mask = inten <= bg_lo
an_mask = inten >= an_hi

out = {
    "n_slices": int(x.shape[0]),
    "n_volumes": int(len(files)),
    "pos_embed_unique_rows_of_256": int(uniq),
    "pos_embed_row_norm_mean": float(pos.norm(dim=1).mean()),
    "pos_embed_row_norm_std": float(pos.norm(dim=1).std()),
    "patch_intensity_bg_threshold": bg_lo,
    "patch_intensity_anat_threshold": an_hi,
    "frac_cells_below_bg_threshold": float(bg_mask.mean()),
    "frac_cells_above_anat_threshold": float(an_mask.mean()),
    "ckpts": {},
}

for tag, path in CKPTS.items():
    sd = torch.load(path, map_location="cpu", weights_only=False)
    enc = sd.get("target_encoder", sd.get("encoder"))
    enc = {k.replace("module.", ""): v for k, v in enc.items()}
    W = enc["patch_embed.proj.weight"].float()
    B = enc["patch_embed.proj.bias"].float()
    e = torch.nn.functional.conv2d(x, W, B, stride=P)                 # (N, 768, 16, 16)
    e = e.flatten(2).transpose(1, 2)                                   # (N, 256, 768)
    del sd, enc

    bg = torch.from_numpy(bg_mask)
    an = torch.from_numpy(an_mask)

    # ---- Decomposition of ACROSS-POSITION variance among background cells ----
    # For each slice, take its background cells only; the layer-0 token is e+pos.
    # Var_across_positions(e + pos) = Var(e) + Var(pos) + 2*Cov(e, pos).
    ve, vp, vt, cov = [], [], [], []
    for i in range(e.shape[0]):
        m = bg[i]
        if m.sum() < 8:
            continue
        ei, pi = e[i][m], pos[m]
        ti = ei + pi
        # variance summed over the 768 dims, taken across the selected positions
        ve.append(float(ei.var(0, unbiased=False).sum()))
        vp.append(float(pi.var(0, unbiased=False).sum()))
        vt.append(float(ti.var(0, unbiased=False).sum()))
        cov.append(float(((ei - ei.mean(0)) * (pi - pi.mean(0))).mean(0).sum()))
    ve, vp, vt, cov = map(np.array, (ve, vp, vt, cov))

    # same decomposition restricted to ANATOMY cells, as the control
    ve_a, vp_a, vt_a = [], [], []
    for i in range(e.shape[0]):
        m = an[i]
        if m.sum() < 8:
            continue
        ei, pi = e[i][m], pos[m]
        ve_a.append(float(ei.var(0, unbiased=False).sum()))
        vp_a.append(float(pi.var(0, unbiased=False).sum()))
        vt_a.append(float((ei + pi).var(0, unbiased=False).sum()))
    ve_a, vp_a, vt_a = map(np.array, (ve_a, vp_a, vt_a))

    nb = float(e[bg].norm(dim=1).mean()) if bg.any() else float("nan")
    na = float(e[an].norm(dim=1).mean()) if an.any() else float("nan")

    d = {
        "checkpoint": path,
        "content_norm_mean_background": nb,
        "content_norm_mean_anatomy": na,
        "pos_norm_over_content_norm_background": float(pos.norm(dim=1).mean()) / nb,
        "pos_norm_over_content_norm_anatomy": float(pos.norm(dim=1).mean()) / na,
        "background_cells": {
            "n_slices_used": int(len(ve)),
            "var_content": float(ve.mean()),
            "var_position": float(vp.mean()),
            "var_total": float(vt.mean()),
            "cov_content_position_x2": float(2 * cov.mean()),
            "position_share_of_layer0_variance": float(vp.mean() / vt.mean()),
            "content_share_of_layer0_variance": float(ve.mean() / vt.mean()),
        },
        "anatomy_cells": {
            "n_slices_used": int(len(ve_a)),
            "var_content": float(ve_a.mean()),
            "var_position": float(vp_a.mean()),
            "var_total": float(vt_a.mean()),
            "position_share_of_layer0_variance": float(vp_a.mean() / vt_a.mean()),
            "content_share_of_layer0_variance": float(ve_a.mean() / vt_a.mean()),
        },
    }
    out["ckpts"][tag] = d
    print("== %s ==" % tag)
    print("  ||pos|| = %.3f ; ||content|| bg = %.3f, anat = %.3f"
          % (out["pos_embed_row_norm_mean"], nb, na))
    print("  layer-0 across-position variance among BACKGROUND cells:")
    print("     position share %.4f | content share %.4f | 2cov %.3f"
          % (d["background_cells"]["position_share_of_layer0_variance"],
             d["background_cells"]["content_share_of_layer0_variance"],
             d["background_cells"]["cov_content_position_x2"]))
    print("  layer-0 across-position variance among ANATOMY cells:")
    print("     position share %.4f | content share %.4f"
          % (d["anatomy_cells"]["position_share_of_layer0_variance"],
             d["anatomy_cells"]["content_share_of_layer0_variance"]))

p = r"C:\Users\Gary\Desktop\jepa\autopilot\bgsig\a3_layer0_position_content.json"
with open(p, "w") as f:
    json.dump(out, f, indent=2)
print("\npos_embed unique rows: %d / 256" % uniq)
print("wrote", p)
