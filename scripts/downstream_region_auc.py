"""How much of the downstream frozen AUC actually comes from background patches?

``eval_downstream.py:391`` pools the frozen encoder with a UNIFORM mean over all
256 patch cells::

    out = encoder(chunk)              # (chunk, 256, D)
    parts.append(out.mean(dim=1))     # every cell weighted 1/256

Anatomy is only ~26% of cells, so background supplies ~74% of the pooled vector
by construction.  That is an arithmetic fact and says nothing about whether the
background part is USEFUL.  This script answers the useful part by rebuilding the
same probe on three different poolings of the very same frozen features:

    all      mean over all 256 cells         (reproduces the published protocol)
    anatomy  mean over anatomy cells only
    background mean over background cells only

Because ``probe_type: mean_pool`` has zero parameters and ``head_type: linear``,
the whole path from patch tokens to logit is linear, so pooling over slices and
patches up front is exactly equivalent to the original pipeline -- one 768-vector
per volume per pooling.  Only the encoder forward pass is expensive.

The anatomy mask comes from ``scripts/fit_anatomy_mask.py`` (held-out volume AUC
0.979, Dice 0.872 against MIRAGE).  It is computed from raw pixels only, so it is
byte-identical across arms and checkpoints -- that is what makes the comparison
fair -- and it is cached once and reused, since it does not depend on the
encoder.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from torch.utils.data import DataLoader                            # noqa: E402
from src.datasets.oct_volumes import OCTVolumeDataset              # noqa: E402
from src.helper import init_patch_model                            # noqa: E402

CROP, PATCH = 256, 16
GRID = CROP // PATCH
NPATCH = GRID * GRID
SPLITS = ("Training", "Validation", "Test")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_normalize(x):
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


# --------------------------------------------------------------- mask ------
def batch_patch_features(imgs: torch.Tensor) -> np.ndarray:
    """(B, 256, F) -- vectorised twin of fit_anatomy_mask.patch_features."""
    g = imgs.mean(1)                                        # (B, 256, 256)
    p = g.unfold(1, PATCH, PATCH).unfold(2, PATCH, PATCH)   # (B,16,16,16,16)
    mean = p.mean(dim=(-1, -2)).numpy()
    std = p.std(dim=(-1, -2)).numpy()
    mx = p.amax(dim=(-1, -2)).numpy()
    mn = p.amin(dim=(-1, -2)).numpy()
    B = mean.shape[0]

    rows = np.broadcast_to(np.arange(GRID, dtype=np.float32)[None, :, None],
                           (B, GRID, GRID)).copy()
    cols = np.broadcast_to(np.arange(GRID, dtype=np.float32)[None, None, :],
                           (B, GRID, GRID)).copy()
    colmax = mean.max(axis=1, keepdims=True)
    colmin = mean.min(axis=1, keepdims=True)
    rel = (mean - colmin) / np.maximum(colmax - colmin, 1e-6)
    argmx = mean.argmax(axis=1)[:, None, :].astype(np.float32)
    d_to_peak = np.abs(rows - argmx)
    z = (mean - mean.mean(axis=(1, 2), keepdims=True)) / (
        mean.std(axis=(1, 2), keepdims=True) + 1e-6)
    up = np.concatenate([mean[:, :1], mean[:, :-1]], axis=1)
    dn = np.concatenate([mean[:, 1:], mean[:, -1:]], axis=1)

    feats = np.stack([mean, std, mx, mn, rows, cols, rel, d_to_peak, z, up, dn,
                      np.broadcast_to(colmax, mean.shape).copy(),
                      np.broadcast_to(colmin, mean.shape).copy()], axis=-1)
    return feats.reshape(B, NPATCH, feats.shape[-1])


def build_mask_cache(args):
    """Anatomy mask for every slice of every volume.  Encoder-independent."""
    import joblib
    blob = joblib.load(pathlib.Path(args.mask_model))
    clf, thr = blob["model"], blob["threshold"]
    cache = pathlib.Path(args.mask_cache); cache.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        dst = cache / f"{split}_s{args.num_slices}_r{CROP}.npz"
        if dst.exists():
            print(f"  {split}: mask cache exists, skipping", flush=True)
            continue
        ds = OCTVolumeDataset(os.path.join(args.data_dir, split),
                              num_slices=args.num_slices, slice_size=CROP,
                              return_label=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)
        out = np.zeros((len(ds), args.num_slices, NPATCH), dtype=bool)
        t0 = time.time()
        for i, (vol, _) in enumerate(loader):
            # OCTVolumeDataset yields UN-normalised slices (eval_downstream.py
            # applies imagenet_normalize itself right before the encoder), but
            # the mask model was fitted on ImageNet-normalised images by
            # make_paired_transforms.  Match it, or the intensity features land
            # on a completely different scale and anatomy is under-predicted.
            flat = imagenet_normalize(vol.squeeze(0))
            X = batch_patch_features(flat).reshape(-1, 13)
            p = clf.predict_proba(X)[:, 1].reshape(args.num_slices, NPATCH)
            out[i] = p >= thr
            if (i + 1) % 500 == 0:
                r = out[:i + 1].mean()
                print(f"    {split} {i + 1}/{len(ds)}  ({time.time() - t0:.0f}s)"
                      f"  anatomy rate {r:.3f}", flush=True)
        np.savez_compressed(dst, mask=np.packbits(out, axis=-1),
                            shape=np.array(out.shape))
        print(f"  {split}: wrote {dst}  anatomy rate {out.mean():.4f}", flush=True)


def load_mask(cache, split, num_slices, n_expect):
    d = np.load(pathlib.Path(cache) / f"{split}_s{num_slices}_r{CROP}.npz")
    shape = tuple(d["shape"])
    m = np.unpackbits(d["mask"], axis=-1)[..., :NPATCH].astype(bool)
    m = m.reshape(shape)
    assert m.shape[0] == n_expect, f"{split}: mask {m.shape[0]} vs data {n_expect}"
    return m


# ------------------------------------------------------------- features ----
@torch.no_grad()
def extract(encoder, args, masks_by_split, device):
    """Per volume: one pooled 768-vector for each of the three regions."""
    # The published config runs fp32, but this pass is GPU-bound at ~99% and
    # fp32 costs ~2.3 h per checkpoint.  autocast is applied IDENTICALLY to
    # every arm and every region, so the region contrast and the cross-arm
    # comparison -- the only things this script is used for -- are unaffected.
    # Absolute AUC may differ in the last decimal from the published fp32 runs.
    amp = torch.autocast("cuda", dtype=torch.float16, enabled=args.amp)
    feats, labels = {}, {}
    limits = {"Training": args.limit_train, "Validation": args.limit_val,
              "Test": args.limit_test}
    for split in SPLITS:
        ds = OCTVolumeDataset(os.path.join(args.data_dir, split),
                              num_slices=args.num_slices, slice_size=CROP,
                              return_label=True)
        M = masks_by_split[split]
        # Subsample by a SEEDED permutation, not a prefix: volumes may be
        # ordered by label or acquisition site, and every arm must see exactly
        # the same volumes for the comparison to stay paired.
        keep = None
        lim = limits[split]
        if lim and lim < len(ds):
            keep = np.sort(np.random.default_rng(args.seed).permutation(len(ds))[:lim])
            ds = torch.utils.data.Subset(ds, keep.tolist())
            M = M[keep]
            print(f"  {split}: subsampled to {len(ds)} volumes", flush=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        F = {k: [] for k in ("all", "anatomy", "background")}
        L = []
        t0 = time.time()
        for i, (vol, lab) in enumerate(loader):
            flat = vol.squeeze(0).to(device)                  # (S,3,H,W)
            m_np = M[i]
            if args.slices_used and args.slices_used < flat.size(0):
                # STRATIFIED, not a contiguous block and not an unconstrained
                # random draw: split the volume into `slices_used` equal bins
                # and take one random slice from each.  That guarantees the
                # picks are spread across the whole depth (an unconstrained
                # draw can clump and leave gaps), while still being random.
                # Seeded per volume index so every arm and every checkpoint
                # sees the SAME slices of the SAME volumes.
                rg = np.random.default_rng(args.seed * 1000003 + i)
                edges = np.linspace(0, flat.size(0), args.slices_used + 1).astype(int)
                sel = np.array([rg.integers(edges[k], max(edges[k + 1], edges[k] + 1))
                                for k in range(args.slices_used)])
                sel = np.unique(sel)
                flat = flat[torch.from_numpy(sel).to(device)]
                m_np = m_np[sel]
            parts = []
            for j in range(0, flat.size(0), args.chunk):
                c = imagenet_normalize(flat[j:j + args.chunk])
                with amp:
                    parts.append(encoder(c).float())          # (c,256,D)
            h = torch.cat(parts, 0)                           # (S,256,D)

            m = torch.from_numpy(m_np).to(device)             # (S,256)
            mf = m.float().unsqueeze(-1)
            na = mf.sum(1).clamp(min=1e-6)
            nb = (1 - mf).sum(1).clamp(min=1e-6)

            F["all"].append(h.mean(1).mean(0).cpu())
            F["anatomy"].append(((h * mf).sum(1) / na).mean(0).cpu())
            F["background"].append(((h * (1 - mf)).sum(1) / nb).mean(0).cpu())
            L.append(int(lab.squeeze()))
            if (i + 1) % 1000 == 0:
                print(f"    {split} {i + 1}/{len(ds)} ({time.time() - t0:.0f}s)",
                      flush=True)
        feats[split] = {k: torch.stack(v).float() for k, v in F.items()}
        labels[split] = torch.tensor(L).long()
        print(f"  {split}: {len(ds)} volumes in {time.time() - t0:.0f}s", flush=True)
    return feats, labels


# ---------------------------------------------------------------- probe ----
def train_probe(Xtr, ytr, Xva, yva, Xte, yte, args, device):
    """Same optimiser/schedule/selection as the frozen mean_pool eval."""
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(args.seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xva, Xte = [(t - mu) / sd for t in (Xtr, Xva, Xte)]
    Xtr, Xva, Xte = Xtr.to(device), Xva.to(device), Xte.to(device)
    ytr_d = ytr.float().to(device)

    head = nn.Sequential(nn.Dropout(args.dropout),
                         nn.Linear(Xtr.size(1), 1)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss()
    n = Xtr.size(0)
    best = dict(val=-1.0, test=None, epoch=-1)
    bad = 0
    for ep in range(1, args.epochs + 1):
        lr = args.lr * (ep / max(args.warmup, 1) if ep <= args.warmup else
                        0.5 * (1 + np.cos(np.pi * (ep - args.warmup) /
                                          max(args.epochs - args.warmup, 1))))
        for g in opt.param_groups:
            g["lr"] = lr
        head.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            opt.zero_grad()
            loss = crit(head(Xtr[idx]).squeeze(-1), ytr_d[idx])
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xva).squeeze(-1).cpu().numpy()
            pt = head(Xte).squeeze(-1).cpu().numpy()
        va = roc_auc_score(yva.numpy(), pv)
        if va > best["val"]:
            best = dict(val=float(va), test=float(roc_auc_score(yte.numpy(), pt)),
                        epoch=ep)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--build_masks", action="store_true")
    ap.add_argument("--data_dir", default=r"D:\jepa_phase0\fairvision-glaucoma\data")
    ap.add_argument("--mask_model", default=(
        r"D:\jepa_phase0\reports\anatomy_mask_calib\anatomy_mask_model.joblib"))
    ap.add_argument("--mask_cache", default=r"D:\jepa_phase0\reports\anatomy_mask_cache")
    ap.add_argument("--feat_cache", default=r"D:\jepa_phase0\reports\region_features")
    ap.add_argument("--num_slices", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast for the encoder pass (~2-3x faster)")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--slices_used", type=int, default=0,
                    help="random slices per volume (0 = all)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\downstream_region_auc")
    args = ap.parse_args()

    if args.build_masks:
        print("=== building anatomy mask cache (encoder-independent) ===")
        build_mask_cache(args)
        if not args.ckpts:
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fc = pathlib.Path(args.feat_cache); fc.mkdir(parents=True, exist_ok=True)

    counts = {s: len(OCTVolumeDataset(os.path.join(args.data_dir, s),
                                      num_slices=args.num_slices,
                                      slice_size=CROP, return_label=True))
              for s in SPLITS}
    masks = {s: load_mask(args.mask_cache, s, args.num_slices, counts[s])
             for s in SPLITS}
    print({s: (masks[s].shape, round(float(masks[s].mean()), 4)) for s in SPLITS})

    res_path = out / "region_auc.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["tag"] for r in results}

    for tag, ck in zip(args.tags, args.ckpts):
        if tag in done:
            print(f"skip {tag} (already done)"); continue
        print(f"\n===== {tag} =====", flush=True)
        cache = fc / f"{tag}_s{args.num_slices}.pt"
        if cache.exists():
            print("  loading cached region features", flush=True)
            d = torch.load(cache, map_location="cpu")
            feats, labels = d["feats"], d["labels"]
        else:
            encoder, _ = init_patch_model(device, patch_size=PATCH,
                                          crop_size=CROP, model_name="vit_base")
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            enc_sd = {k.replace("module.", ""): v for k, v in sd["target_encoder"].items()}
            encoder.load_state_dict(enc_sd, strict=True)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            feats, labels = extract(encoder, args, masks, device)
            torch.save({"feats": feats, "labels": labels}, cache)
            del encoder
            torch.cuda.empty_cache()

        row = dict(tag=tag, checkpoint=str(ck))
        for region in ("all", "anatomy", "background"):
            b = train_probe(feats["Training"][region], labels["Training"],
                            feats["Validation"][region], labels["Validation"],
                            feats["Test"][region], labels["Test"], args, device)
            row[region] = b
            print(f"  {region:>10}: val {b['val']:.4f}  TEST {b['test']:.4f}"
                  f"  (ep {b['epoch']})", flush=True)
        results.append(row)
        res_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {res_path}")


if __name__ == "__main__":
    main()
