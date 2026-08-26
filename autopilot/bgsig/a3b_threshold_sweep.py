"""CPU-only robustness sweep for a3: does the layer-0 position/content split hold
across background and anatomy intensity thresholds?"""
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

torch.set_num_threads(4)
torch.set_grad_enabled(False)
import sys
sys.path.insert(0, r"C:\Users\Gary\Desktop\jepa")
from src.models.vision_transformer import get_2d_sincos_pos_embed

CKPTS = {
    "random_ep100": r"D:\jepa_phase0\checkpoints_hf\random-posfix-100ep\jepa_patch-ep100.pth.tar",
    "oracle_ep100": r"D:\jepa_phase0\checkpoints_hf\oracle-anatomical-100ep\jepa_patch_oracle-ep100.pth.tar",
}
DATA = r"D:\jepa_phase0\fairvision-glaucoma\data\Test"
N_VOL, N_SLICE, SIZE, P, G = 12, 8, 256, 16, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

files = sorted(glob.glob(os.path.join(DATA, "*.npz")))[:N_VOL]
sidx = np.linspace(0, 199, N_SLICE).round().astype(int)
imgs, raws = [], []
for f in files:
    d = np.load(f)
    v = d["oct_bscans"]
    for s in sidx:
        pil = Image.fromarray(np.asarray(v[int(s)]), mode="L").resize((SIZE, SIZE), Image.BILINEAR)
        raws.append(np.asarray(pil, dtype=np.float32) / 255.0)
        imgs.append(torch.from_numpy(
            np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1))
    d.close()
x = (torch.stack(imgs) - MEAN) / STD
raws = np.stack(raws)
inten = raws.reshape(-1, G, P, G, P).mean(axis=(2, 4)).reshape(-1, G * G)
pos = torch.from_numpy(get_2d_sincos_pos_embed(768, G)).float()

BG_T = [0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
AN_T = [0.20, 0.30, 0.40]
out = {"n_slices": int(x.shape[0]), "bg_thresholds": BG_T, "anat_thresholds": AN_T, "ckpts": {}}


def share(e, sel):
    vp, vt = [], []
    for i in range(e.shape[0]):
        m = sel[i]
        if m.sum() < 8:
            continue
        ei, pi = e[i][m], pos[m]
        vp.append(float(pi.var(0, unbiased=False).sum()))
        vt.append(float((ei + pi).var(0, unbiased=False).sum()))
    if not vp:
        return None, 0
    return float(np.mean(vp) / np.mean(vt)), len(vp)


for tag, path in CKPTS.items():
    sd = torch.load(path, map_location="cpu", weights_only=False)
    enc = {k.replace("module.", ""): v for k, v in
           sd.get("target_encoder", sd.get("encoder")).items()}
    e = torch.nn.functional.conv2d(x, enc["patch_embed.proj.weight"].float(),
                                   enc["patch_embed.proj.bias"].float(), stride=P)
    e = e.flatten(2).transpose(1, 2)
    del sd, enc
    r = {"background": {}, "anatomy": {}}
    print("== %s ==" % tag)
    for t in BG_T:
        sel = torch.from_numpy(inten <= t)
        s, n = share(e, sel)
        r["background"]["<=%.2f" % t] = {"position_share": s, "n_slices": n,
                                         "cell_frac": float((inten <= t).mean())}
        print("  bg  thr<=%.2f  cells %.3f  position share of layer-0 var = %.4f (n=%d)"
              % (t, (inten <= t).mean(), s, n))
    for t in AN_T:
        sel = torch.from_numpy(inten >= t)
        s, n = share(e, sel)
        r["anatomy"][">=%.2f" % t] = {"position_share": s, "n_slices": n,
                                      "cell_frac": float((inten >= t).mean())}
        print("  ana thr>=%.2f  cells %.3f  position share of layer-0 var = %.4f (n=%d)"
              % (t, (inten >= t).mean(), s, n))
    out["ckpts"][tag] = r

p = r"C:\Users\Gary\Desktop\jepa\autopilot\bgsig\a3b_threshold_sweep.json"
with open(p, "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote", p)
