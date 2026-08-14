"""Is background (non-anatomy) actually carrying signal in patch I-JEPA?

Motivating claim
----------------
The predictor's query for target cell j is ``mask_token + pos_embed[j]``.  The
mask token is ONE shared learned vector, and ``predictor_pos_embed`` is a fixed
sinusoidal code defined over all 256 grid cells.  So the query carries position
and nothing else -- every bit of content must be retrieved by attention over the
context tokens, and that attention is unmasked (``vision_transformer.py:560-564``
concatenates context and mask tokens and runs plain blocks).  A background
context token is therefore a real, attended input, and "there is nothing at
(r,c)" is an addressable fact.

That makes three DIFFERENT claims, which this script separates:

  (A) background as CONTEXT is informative -- removing it should hurt
      predictions of anatomy targets, beyond the damage of merely having fewer
      context tokens;
  (B) background as TARGET is a learning signal -- if h(background) is nearly
      constant, those slots are trivially predictable and contribute almost no
      gradient;
  (C) an arm that never predicts background fails to REPRESENT it -- probe how
      separable anatomy is from background in the frozen features.

Confound that must be controlled
--------------------------------
``scripts/jepa_error_confound_check.py`` already established on this codebase
that per-cell prediction error is dominated by distance-to-context
(corr = +0.5687) and that the raw anatomy correlation (-0.2675) collapses to
+0.0425 once distance is partialled out.  Any measurement here that compares
background to anatomy without holding distance fixed will simply re-derive that
artifact, so measurement (2) reports distance-binned and partial statistics, and
measurement (1) is paired against a count-matched random-drop control.

Fairness
--------
Every arm is evaluated under the SAME stock uniform mask distribution, mirroring
``train_patch.py:452`` which pins validation to a uniform collator so the number
compares encoders rather than mask policies.
"""
from __future__ import annotations

import argparse
import copy
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
from src.helper import init_patch_model                           # noqa: E402
from src.masks.multiblock import MaskCollator                     # noqa: E402
from src.masks.utils import apply_masks                           # noqa: E402
from src.transforms import make_paired_transforms                 # noqa: E402

CROP, PATCH = 256, 16
GRID = CROP // PATCH
NPATCH = GRID * GRID
OCC_T = 0.25

ROWS, COLS = np.divmod(np.arange(NPATCH), GRID)


# ----------------------------------------------------------------- model ----
def load_jepa(ckpt, device):
    encoder, predictor = init_patch_model(
        device, patch_size=PATCH, crop_size=CROP, model_name="vit_base")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)

    def strip(d):
        return {k.replace("module.", ""): v for k, v in d.items()}

    encoder.load_state_dict(strip(sd["encoder"]), strict=True)
    predictor.load_state_dict(strip(sd["predictor"]), strict=True)
    target_encoder = copy.deepcopy(encoder)
    target_encoder.load_state_dict(strip(sd["target_encoder"]), strict=True)
    for m in (encoder, predictor, target_encoder):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    return encoder, predictor, target_encoder, int(sd.get("epoch", -1))


def partial_corr(x, y, Z):
    """corr(x, y) after linearly removing every column of Z from both."""
    keep = np.isfinite(x) & np.isfinite(y)
    for z in Z:
        keep &= np.isfinite(z)
    x, y = x[keep], y[keep]
    Z = [z[keep] for z in Z]
    A = np.column_stack([np.ones(len(x))] + [z.ravel() for z in Z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def eff_rank(X):
    """Participation ratio of the eigenvalue spectrum (a soft rank)."""
    if X.shape[0] < 4:
        return float("nan")
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    if s.sum() <= 0:
        return float("nan")
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


# ---------------------------------------------------------- measurements ----
@torch.no_grad()
def measure(ckpt, ds, idxs, device, args):
    encoder, predictor, tenc, epoch = load_jepa(ckpt, device)
    # Re-seed every RNG the data path and the collator touch, so that EVERY
    # checkpoint sees byte-identical crops and byte-identical context/target
    # draws.  Without this the paired ablation would compare arms across
    # different images and the deltas would be dominated by crop noise.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    collator = MaskCollator(
        input_size=(CROP, CROP), patch_size=PATCH, enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2), aspect_ratio=(0.75, 1.5),
        nenc=1, npred=4, min_keep=10, allow_overlap=False)

    # (B) h-geometry accumulators
    cos_bg, cos_an, er_bg, er_an, bg_frac = [], [], [], [], []
    # (C) probe data
    Hs, Ls = [], []
    # (2) per-slot error records
    rec_err, rec_anat, rec_dist, rec_row = [], [], [], []
    # (1) ablation accumulators, paired per batch
    abl = {k: [] for k in ("full", "drop_bg", "drop_anat", "drop_rand")}
    abl_k = []

    rng = np.random.default_rng(args.seed)

    for start in range(0, len(idxs), args.batch_size):
        chunk = idxs[start:start + args.batch_size]
        items = [ds[i] for i in chunk]
        imgs = torch.stack([it[0] for it in items], 0)
        guides = torch.stack([it[1] for it in items], 0)
        B = imgs.size(0)
        x = imgs.to(device)

        occ = guides[:, 0].reshape(B, -1).numpy()
        anat = occ >= OCC_T                                    # (B, 256) bool

        # ---- h geometry / probe features (no masking at all) -------------
        h_all = tenc(x)
        h_all = F.layer_norm(h_all, (h_all.size(-1),))
        hn = F.normalize(h_all, dim=-1).cpu().numpy()
        hcpu = h_all.cpu().numpy()
        for b in range(B):
            a = anat[b]
            bg_frac.append(1.0 - a.mean())
            for sel, cl, el in ((a, cos_an, er_an), (~a, cos_bg, er_bg)):
                if sel.sum() >= 4:
                    V = hn[b][sel]
                    S = V @ V.T
                    iu = np.triu_indices(len(V), 1)
                    cl.append(float(S[iu].mean()))
                    el.append(eff_rank(hcpu[b][sel]))
            if len(Hs) < args.probe_cells:
                pick = rng.choice(NPATCH, size=min(32, NPATCH), replace=False)
                Hs.append(hcpu[b][pick])
                Ls.append(a[pick])

        # ---- masked draws -------------------------------------------------
        for _ in range(args.draws):
            _, m_enc, m_pred = collator([im for im in imgs])
            v_enc = [m.to(device) for m in m_enc]
            v_pred = [m.to(device) for m in m_pred]

            z = predictor(encoder(x, v_enc), v_enc, v_pred)
            h_t = apply_masks(h_all, v_pred)
            e = F.smooth_l1_loss(z, h_t, reduction="none").mean(-1).cpu().numpy()

            ctx = m_enc[0].numpy()                              # (B, N_ctx)
            for b in range(B):
                d_all = np.hypot(ROWS[:, None] - ROWS[ctx[b]][None, :],
                                 COLS[:, None] - COLS[ctx[b]][None, :]).min(1)
                for j, m in enumerate(m_pred):
                    idx = m[b].numpy()
                    rec_err.append(e[j * B + b])
                    rec_anat.append(anat[b][idx].astype(np.float64))
                    rec_dist.append(d_all[idx])
                    rec_row.append(ROWS[idx].astype(np.float64))

            # ---- (1) count-matched context ablation ----------------------
            n_bg = np.array([(~anat[b][ctx[b]]).sum() for b in range(B)])
            n_an = np.array([anat[b][ctx[b]].sum() for b in range(B)])
            k = int(min(n_bg.min(), n_an.min(), args.max_drop))
            if k < args.min_drop:
                continue
            keep = {"drop_bg": [], "drop_anat": [], "drop_rand": []}
            for b in range(B):
                c = ctx[b]
                is_a = anat[b][c]
                for tag, pool in (("drop_bg", np.flatnonzero(~is_a)),
                                  ("drop_anat", np.flatnonzero(is_a)),
                                  ("drop_rand", np.arange(len(c)))):
                    drop = rng.choice(pool, size=k, replace=False)
                    mask = np.ones(len(c), bool)
                    mask[drop] = False
                    keep[tag].append(c[mask])
            # error on ANATOMY target cells only, paired across variants
            base = _tgt_err(encoder, predictor, x, v_enc, v_pred, h_all,
                            anat, m_pred, B, device)
            abl["full"].append(base)
            for tag, rowsl in keep.items():
                ve = [torch.as_tensor(np.stack(rowsl), dtype=torch.long,
                                      device=device)]
                abl[tag].append(_tgt_err(encoder, predictor, x, ve, v_pred,
                                         h_all, anat, m_pred, B, device))
            abl_k.append(k)

    err = np.concatenate(rec_err)
    an = np.concatenate(rec_anat)
    dist = np.concatenate(rec_dist)
    row = np.concatenate(rec_row)

    # distance-binned error, background vs anatomy
    bins = [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 99)]
    binned = []
    for lo, hi in bins:
        s = (dist >= lo) & (dist < hi)
        if s.sum() < 50:
            continue
        binned.append(dict(
            lo=lo, hi=hi, n=int(s.sum()),
            err_anat=float(err[s & (an > 0.5)].mean()) if (s & (an > 0.5)).sum() else None,
            err_bg=float(err[s & (an < 0.5)].mean()) if (s & (an < 0.5)).sum() else None))

    out = dict(
        checkpoint=str(ckpt), epoch=epoch,
        n_slices=len(idxs), n_slots=int(err.size),
        bg_frac_mean=float(np.mean(bg_frac)),
        # (B) is background trivially predictable?
        err_anat=float(err[an > 0.5].mean()), err_bg=float(err[an < 0.5].mean()),
        err_anat_sd=float(err[an > 0.5].std()), err_bg_sd=float(err[an < 0.5].std()),
        frac_slots_bg=float((an < 0.5).mean()),
        corr_err_anat=float(np.corrcoef(err, an)[0, 1]),
        corr_err_dist=float(np.corrcoef(err, dist)[0, 1]),
        partial_err_anat_given_dist=partial_corr(err, an, [dist]),
        partial_err_anat_given_dist_row=partial_corr(err, an, [dist, row]),
        err_by_distance=binned,
        cos_within_bg=float(np.mean(cos_bg)) if cos_bg else None,
        cos_within_anat=float(np.mean(cos_an)) if cos_an else None,
        effrank_bg=float(np.nanmean(er_bg)) if er_bg else None,
        effrank_anat=float(np.nanmean(er_an)) if er_an else None,
    )

    # (1) ablation deltas -- the decisive numbers
    if abl["full"]:
        A = {k: np.array(v) for k, v in abl.items()}
        out["ablation"] = dict(
            k_dropped=float(np.mean(abl_k)), n_batches=int(len(A["full"])),
            err_full=float(A["full"].mean()),
            err_drop_bg=float(A["drop_bg"].mean()),
            err_drop_anat=float(A["drop_anat"].mean()),
            err_drop_rand=float(A["drop_rand"].mean()),
            # excess damage over the count-matched random control
            excess_bg=float((A["drop_bg"] - A["drop_rand"]).mean()),
            excess_anat=float((A["drop_anat"] - A["drop_rand"]).mean()),
            excess_bg_sem=float((A["drop_bg"] - A["drop_rand"]).std(ddof=1)
                                / np.sqrt(len(A["full"]))),
            excess_anat_sem=float((A["drop_anat"] - A["drop_rand"]).std(ddof=1)
                                  / np.sqrt(len(A["full"]))),
        )

    # (C) linear separability of anatomy in frozen features
    if Hs:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        X = np.concatenate(Hs).astype(np.float64)
        y = np.concatenate(Ls).astype(int)
        n = len(y)
        tr = rng.permutation(n)
        cut = int(0.7 * n)
        a, b = tr[:cut], tr[cut:]
        if y[a].min() != y[a].max() and y[b].min() != y[b].max():
            mu, sd = X[a].mean(0), X[a].std(0) + 1e-6
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit((X[a] - mu) / sd, y[a])
            p = clf.predict_proba((X[b] - mu) / sd)[:, 1]
            out["anatomy_probe_auc"] = float(roc_auc_score(y[b], p))
            out["anatomy_probe_n"] = int(n)

    del encoder, predictor, tenc
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def _tgt_err(encoder, predictor, x, v_enc, v_pred, h_all, anat, m_pred, B, device):
    """Mean error over ANATOMY target cells only, for one context variant."""
    z = predictor(encoder(x, v_enc), v_enc, v_pred)
    h_t = apply_masks(h_all, v_pred)
    e = F.smooth_l1_loss(z, h_t, reduction="none").mean(-1).cpu().numpy()
    tot = cnt = 0.0
    for b in range(B):
        for j, m in enumerate(m_pred):
            idx = m[b].numpy()
            sel = anat[b][idx]
            if sel.any():
                tot += e[j * B + b][sel].sum()
                cnt += sel.sum()
    return tot / max(cnt, 1)


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
    ap.add_argument("--max_drop", type=int, default=24)
    ap.add_argument("--min_drop", type=int, default=4)
    ap.add_argument("--probe_cells", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"D:\jepa_phase0\reports\background_signal")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

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

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tags = args.tags or [pathlib.Path(c).stem for c in args.ckpts]

    results = []
    for tag, ck in zip(tags, args.ckpts):
        print(f"\n===== {tag} =====", flush=True)
        r = measure(ck, ds, idxs, device, args)
        r["tag"] = tag
        results.append(r)
        print(json.dumps({k: v for k, v in r.items()
                          if k not in ("err_by_distance",)}, indent=2)[:1400],
              flush=True)
        (out / "background_signal.json").write_text(json.dumps(results, indent=2))

    print(f"\nwrote {out / 'background_signal.json'}", flush=True)


if __name__ == "__main__":
    main()
