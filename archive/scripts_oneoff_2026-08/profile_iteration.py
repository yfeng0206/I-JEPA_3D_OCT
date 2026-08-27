"""Where does a training iteration actually go?

Before optimising the sampler it is worth knowing whether the sampler is on the
critical path at all: the mask generator runs inside the DataLoader workers, so
its cost is hidden unless it exceeds the GPU step divided by the worker count.

Times three things separately on the real config:
  1. dataset __getitem__  (slice + guide I/O, from the SSD cache)
  2. the collator alone   (MIRAGE guide handling + COVER greedy placement)
  3. the GPU step         (encoder + predictor forward/backward under autocast)

and reports the implied per-iteration budget with N workers.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.helper import init_patch_model                           # noqa: E402
from src.masks.curriculum import CurriculumMaskGenerator          # noqa: E402
from src.masks.utils import apply_masks                           # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402

CROP, PATCH = 256, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=r"configs\patch_cover_random_ep25.yaml")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--epoch", type=int, default=50)
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    d, mc = cfg["data"], cfg["mask"]
    cc = dict(mc["curriculum"]); cc.pop("enabled", None)
    B = int(d["batch_size"])
    nw = int(d["num_workers"])

    paired = make_paired_transforms(
        crop_size=CROP, crop_scale=tuple(d["crop_scale"]), gaussian_blur=False,
        horizontal_flip=False, color_distortion=False, color_jitter=0.0)
    sc = os.path.join(d["slice_cache_dir"], "Training")
    ds = GuidedOCTSliceDataset(
        data_dir=os.path.join(d["data_dir"], "Training"),
        guide_dir=os.path.join(cc["mirage_guide_dir"], "Training"),
        num_slices=int(d["num_slices"]), slice_size=CROP, transform=paired,
        patch_size=PATCH, dilate_patches=int(cc.get("mirage_dilate_patches", 0)),
        occupancy_threshold=float(cc.get("mirage_occupancy_threshold", 0.25)),
        slice_cache=sc if os.path.isdir(sc) else None)

    gen = CurriculumMaskGenerator(
        input_size=(CROP, CROP), patch_size=PATCH,
        enc_mask_scale=tuple(mc["enc_mask_scale"]),
        pred_mask_scale=tuple(mc["pred_mask_scale"]),
        aspect_ratio=tuple(mc["aspect_ratio"]),
        nenc=int(mc["num_enc_masks"]), npred=int(mc["num_pred_masks"]),
        min_keep=int(mc["min_keep"]), allow_overlap=bool(mc["allow_overlap"]),
        curriculum_cfg=cc)
    gen.set_epoch(args.epoch, 100)

    rng = random.Random(0)
    n = len(ds)

    # ---- 1. dataset I/O ---------------------------------------------------
    t_io = []
    batches = []
    for _ in range(args.batches):
        idx = [rng.randrange(n) for _ in range(B)]
        t0 = time.perf_counter()
        items = [ds[i] for i in idx]
        t_io.append(time.perf_counter() - t0)
        batches.append(items)

    # ---- 2. collator (mask generation) ------------------------------------
    t_mask = []
    packs = []
    for items in batches:
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        valid = torch.stack([it[2] for it in items], 0)
        t0 = time.perf_counter()
        m_enc, m_pred = gen.generate(batch_size=imgs.size(0),
                                     guide_grids=guides, guide_valid=valid)
        t_mask.append(time.perf_counter() - t0)
        packs.append((imgs, m_enc, m_pred))

    # ---- 3. GPU step ------------------------------------------------------
    dev = torch.device("cuda")
    enc, pred = init_patch_model(dev, patch_size=PATCH, crop_size=CROP,
                                 model_name=cfg["meta"]["model_name"])
    import copy
    tenc = copy.deepcopy(enc)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    t_gpu = []
    for imgs, m_enc, m_pred in packs:
        x = imgs.to(dev, non_blocking=True)
        ve = [m.to(dev) for m in m_enc]
        vp = [m.to(dev) for m in m_pred]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            h = tenc(x)
            h = F.layer_norm(h, (h.size(-1),))
            h = apply_masks(h, vp)
        with torch.autocast("cuda", dtype=torch.float16):
            z = pred(enc(x, ve), ve, vp)
            loss = F.smooth_l1_loss(z, h)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); t_gpu.append(time.perf_counter() - t0)

    io = np.median(t_io); mk = np.median(t_mask); gp = np.median(t_gpu)
    print(f"\nbatch_size={B}  num_workers={nw}  (medians over {args.batches} batches)")
    print(f"  1. dataset I/O per batch      {io*1000:8.1f} ms")
    print(f"  2. mask generation per batch  {mk*1000:8.1f} ms   "
          f"({mk/B*1000:.2f} ms/slice)")
    print(f"  3. GPU step per batch         {gp*1000:8.1f} ms")
    cpu = io + mk
    print(f"\n  CPU work per batch            {cpu*1000:8.1f} ms")
    print(f"  CPU work / {nw} workers         {cpu/nw*1000:8.1f} ms  "
          f"<- what the GPU actually waits for")
    if cpu / nw < gp:
        head = gp / max(cpu / nw, 1e-9)
        print(f"\n  VERDICT: GPU-BOUND. CPU pipeline has {head:.1f}x headroom;"
              f" more workers will NOT help.")
        print(f"  Per-epoch floor set by GPU: 9375 x {gp*1000:.0f} ms = "
              f"{9375*gp/60:.0f} min")
    else:
        need = int(np.ceil(cpu / gp))
        print(f"\n  VERDICT: CPU/DATA-BOUND. Need >= {need} workers to hide it "
              f"(have {nw}).")
    print(f"\n  mask gen as a share of CPU work: {100*mk/cpu:.1f}%")


if __name__ == "__main__":
    main()
