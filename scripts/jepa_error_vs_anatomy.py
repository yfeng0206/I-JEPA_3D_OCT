"""Does JEPA find anatomy HARDER or EASIER to predict than background?

This decides the sign of the scorer loss, which is the one unresolved piece of
the residual-adapter design.  The plan trains the adapter from detached
per-target errors e_i, but never says which direction:

    push score toward HIGH e   -> sampler seeks hard targets
    push score toward LOW  e   -> sampler seeks easy targets (measured collapse)

Neither is safe in the abstract.  What settles it is an empirical fact nobody
has measured: how per-patch prediction difficulty is spatially distributed
relative to retinal anatomy.

    corr(e, anatomy) < 0  =>  "seek hard" drags the sampler OFF anatomy and
                              undoes MIRAGE guidance entirely
    corr(e, anatomy) > 0  =>  "seek hard" and "seek anatomy" agree, and the
                              adapter has a coherent objective

Method: for each slice, sample K real context masks from the production
collator, predict ALL 256 patches, and accumulate each patch's smooth-L1 error
only on draws where that patch was OUTSIDE the context (otherwise the encoder
has already seen it and the error is trivially small).  Anatomy comes from the
same MIRAGE L0 dump, pooled 8x8 to the same 16x16 grid.

Repo venv (needs torch + matplotlib).  Reads the L0.npz written by
scripts/mirage_logit_scale.py --dump so MIRAGE is not re-run.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.helper import init_patch_model                      # noqa: E402
from src.masks.multiblock import MaskCollator                # noqa: E402
from src.masks.utils import apply_masks                      # noqa: E402

TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ANATOMY = (1, 2)
CROP, PATCH, GRID, POOL = 256, 16, 16, 8


def softmax_np(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def anatomy_grid(L0):
    """(N,4,128,128) logits -> (N,16,16) soft anatomy occupancy."""
    M = softmax_np(L0, axis=1)[:, ANATOMY].sum(axis=1)
    n, h, w = M.shape
    return M.reshape(n, h // POOL, POOL, w // POOL, POOL).mean(axis=(2, 4))


def load_slice(name):
    """'data_07050:4' -> normalised (3,256,256) tensor, exactly as training sees it."""
    import cv2
    vol_id, _, sl = name.partition(':')
    with np.load(TEST / ('%s.npz' % vol_id), allow_pickle=True) as z:
        vol = z['oct_bscans']
    raw = np.asarray(vol[int(sl)], dtype=np.float32)
    lo, hi = raw.min(), raw.max()
    unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    img = cv2.resize(unit, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img)[None].repeat(3, 1, 1)          # ToTensor equivalent
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def load_jepa(ckpt, device):
    encoder, predictor = init_patch_model(device, patch_size=PATCH, crop_size=CROP,
                                          model_name='vit_base')
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    strip = lambda d: {k.replace('module.', ''): v for k, v in d.items()}
    encoder.load_state_dict(strip(sd['encoder']), strict=True)
    predictor.load_state_dict(strip(sd['predictor']), strict=True)
    target_encoder = copy.deepcopy(encoder)
    target_encoder.load_state_dict(strip(sd['target_encoder']), strict=True)
    for m in (encoder, predictor, target_encoder):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    return encoder, predictor, target_encoder, int(sd.get('epoch', -1))


@torch.no_grad()
def patch_errors(encoder, predictor, target_encoder, img, collator, k, device):
    """Mean smooth-L1 error per patch, counted only when the patch is hidden."""
    x = img[None].to(device)
    h = target_encoder(x)
    h = F.layer_norm(h, (h.size(-1),))
    all_idx = torch.arange(GRID * GRID, device=device)[None]

    err_sum = torch.zeros(GRID * GRID, device=device)
    err_cnt = torch.zeros(GRID * GRID, device=device)
    for _ in range(k):
        _, masks_enc, _ = collator([img])
        venc = [m.to(device) for m in masks_enc]
        z = encoder(x, venc)
        z = predictor(z, venc, [all_idx])                    # predict every patch
        h_t = apply_masks(h, [all_idx])
        e = F.smooth_l1_loss(z, h_t, reduction='none').mean(-1)[0]   # (256,)
        hidden = torch.ones(GRID * GRID, device=device, dtype=torch.bool)
        hidden[venc[0][0]] = False                           # exclude visible patches
        err_sum[hidden] += e[hidden]
        err_cnt[hidden] += 1
    return (err_sum / err_cnt.clamp(min=1)).reshape(GRID, GRID).cpu().numpy(), \
           err_cnt.reshape(GRID, GRID).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=pathlib.Path, required=True, help='L0.npz from mirage_logit_scale')
    ap.add_argument('--ckpt', type=pathlib.Path, nargs='+', required=True,
                    help='one or more checkpoints; >1 enables the learning-progress test')
    ap.add_argument('--k', type=int, default=32, help='context-mask draws per slice')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=pathlib.Path,
                    default=pathlib.Path('results/masking/error_vs_anatomy'))
    a = ap.parse_args()

    import random
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = np.load(a.npz, allow_pickle=True)
    names = [str(s) for s in z['names']]
    anat = anatomy_grid(z['logits'])

    collator = MaskCollator(input_size=(CROP, CROP), patch_size=PATCH)
    imgs = [load_slice(nm) for nm in names]

    per_ckpt = {}
    for ck in a.ckpt:
        encoder, predictor, target_encoder, epoch = load_jepa(ck, device)
        # identical context masks for every checkpoint -> paired comparison
        random.seed(a.seed)
        torch.manual_seed(a.seed)
        errs = []
        for img in imgs:
            e, cnt = patch_errors(encoder, predictor, target_encoder, img, collator,
                                  a.k, device)
            errs.append(e)
        per_ckpt[epoch] = np.stack(errs)
        print('  ep%-4d mean err %.5f' % (epoch, per_ckpt[epoch].mean()))
        del encoder, predictor, target_encoder
        torch.cuda.empty_cache()

    epochs = sorted(per_ckpt)
    err = per_ckpt[epochs[-1]]
    rep = {'n_slices': len(names), 'k': a.k, 'epochs': epochs,
           'checkpoints': [str(c) for c in a.ckpt]}

    # ---- correlation, globally and per slice ----------------------------
    rep['corr_global'] = float(np.corrcoef(err.ravel(), anat.ravel())[0, 1])
    per = [float(np.corrcoef(err[i].ravel(), anat[i].ravel())[0, 1])
           for i in range(len(names))]
    rep['corr_per_slice'] = {'mean': float(np.mean(per)), 'std': float(np.std(per)),
                             'min': float(np.min(per)), 'max': float(np.max(per)),
                             'n_negative': int(sum(p < 0 for p in per)),
                             'values': per}

    # ---- error stratified by anatomy occupancy --------------------------
    bins = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.95), (0.95, 1.01)]
    strat = []
    for lo, hi in bins:
        m = (anat >= lo) & (anat < hi)
        strat.append({'bin': f'[{lo:.2f},{hi:.2f})', 'frac': float(m.mean()),
                      'err_mean': float(err[m].mean()) if m.any() else float('nan'),
                      'err_std': float(err[m].std()) if m.any() else float('nan')})
    rep['error_by_anatomy_bin'] = strat

    on = anat > 0.5
    rep['err_on_anatomy'] = float(err[on].mean())
    rep['err_off_anatomy'] = float(err[~on].mean())
    rep['err_ratio_on_over_off'] = float(err[on].mean() / err[~on].mean())

    # ---- learning progress: the noisy-TV fix ----------------------------
    # Raw error is anti-correlated with anatomy because unstructured speckle is
    # genuinely unpredictable ("noisy TV").  The intrinsic-motivation literature
    # replaces error with LEARNING PROGRESS: how much the error at a location
    # actually fell as training advanced.
    #
    # CONFOUND: in I-JEPA the target is the EMA target encoder, which moves with
    # training, so e_t at different t are measured against DIFFERENT targets and
    # are not directly comparable -- observed here as mean error RISING with
    # epoch.  We therefore normalise each checkpoint's error map by its own mean
    # before differencing, which cancels any global target-scale drift and
    # leaves only the change in RELATIVE difficulty, which is the quantity a
    # masking policy actually needs.
    if len(epochs) > 1:
        norm = {e: per_ckpt[e] / per_ckpt[e].mean() for e in epochs}
        prog, prog_n = {}, {}
        for i in range(len(epochs) - 1):
            e0, e1 = epochs[i], epochs[-1]
            for tag, src, dest in (('raw', per_ckpt, prog), ('norm', norm, prog_n)):
                p = src[e0] - src[e1]                # positive = became easier
                pc = [float(np.corrcoef(p[j].ravel(), anat[j].ravel())[0, 1])
                      for j in range(len(names))]
                dest[f'ep{e0}_to_ep{e1}'] = {
                    'corr_global': float(np.corrcoef(p.ravel(), anat.ravel())[0, 1]),
                    'corr_per_slice_mean': float(np.mean(pc)),
                    'n_positive_slices': int(sum(v > 0 for v in pc)),
                    'progress_on_anatomy': float(p[on].mean()),
                    'progress_off_anatomy': float(p[~on].mean()),
                }
        rep['learning_progress'] = prog
        rep['learning_progress_normalised'] = prog_n
        rep['corr_error_by_epoch'] = {
            str(e): float(np.corrcoef(per_ckpt[e].ravel(), anat.ravel())[0, 1])
            for e in epochs}
        rep['corr_norm_error_by_epoch'] = {
            str(e): float(np.corrcoef(norm[e].ravel(), anat.ravel())[0, 1])
            for e in epochs}
        rep['mean_error_by_epoch'] = {str(e): float(per_ckpt[e].mean()) for e in epochs}
        np.savez_compressed(a.out / 'per_ckpt_errors.npz', anat=anat,
                            names=np.array(names),
                            **{f'ep{e}': per_ckpt[e] for e in epochs})

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'error_vs_anatomy.json').write_text(json.dumps(rep, indent=2))

    print('\n--- JEPA per-patch error vs MIRAGE anatomy (ep%d, K=%d, n=%d) ---'
          % (epochs[-1], a.k, len(names)))
    print('  corr global            : %+.4f' % rep['corr_global'])
    c = rep['corr_per_slice']
    print('  corr per slice         : mean %+.4f  std %.4f  range [%+.4f, %+.4f]'
          % (c['mean'], c['std'], c['min'], c['max']))
    print('  slices with negative r : %d / %d' % (c['n_negative'], len(names)))
    print('\n  error by anatomy occupancy bin:')
    for s in strat:
        print('    %-14s cells %5.1f%%   err %.5f +- %.5f'
              % (s['bin'], 100 * s['frac'], s['err_mean'], s['err_std']))
    print('\n  on anatomy  %.5f   off anatomy  %.5f   ratio %.4f'
          % (rep['err_on_anatomy'], rep['err_off_anatomy'],
             rep['err_ratio_on_over_off']))

    if 'learning_progress' in rep:
        print('\n--- LEARNING PROGRESS (does it point at anatomy where error does not?) ---')
        print('  corr(error, anatomy) by epoch      raw / mean-normalised:')
        for e in epochs:
            print('    ep%-4s %+.4f / %+.4f   (mean err %.5f)'
                  % (e, rep['corr_error_by_epoch'][str(e)],
                     rep['corr_norm_error_by_epoch'][str(e)],
                     rep['mean_error_by_epoch'][str(e)]))
        for label, key in (('RAW  (confounded by moving EMA target)', 'learning_progress'),
                           ('NORMALISED (target drift removed)', 'learning_progress_normalised')):
            print('  corr(progress, anatomy)  %s:' % label)
            for k, v in rep[key].items():
                print('    %-16s global %+.4f  per-slice %+.4f  positive %2d/%d'
                      % (k, v['corr_global'], v['corr_per_slice_mean'],
                         v['n_positive_slices'], len(names)))
                print('      progress on anatomy %+.5f   off anatomy %+.5f'
                      % (v['progress_on_anatomy'], v['progress_off_anatomy']))

    _figure(a.out, err, anat, rep, names)
    print('\nwrote %s' % (a.out / 'error_vs_anatomy.json'))


def _figure(out, err, anat, rep, names):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))

    ax[0].scatter(anat.ravel(), err.ravel(), s=3, alpha=0.2)
    ax[0].set_xlabel('MIRAGE anatomy occupancy'); ax[0].set_ylabel('JEPA patch error')
    ax[0].set_title('r = %+.4f (global)' % rep['corr_global'])

    s = rep['error_by_anatomy_bin']
    ax[1].bar(range(len(s)), [b['err_mean'] for b in s],
              yerr=[b['err_std'] for b in s], color='tab:orange', capsize=3)
    ax[1].set_xticks(range(len(s)))
    ax[1].set_xticklabels([b['bin'] for b in s], rotation=45, ha='right', fontsize=7)
    ax[1].set_ylabel('mean patch error'); ax[1].set_title('error by anatomy bin')

    im = ax[2].imshow(anat[0], cmap='viridis'); ax[2].set_title('anatomy  %s' % names[0])
    ax[2].set_xticks([]); ax[2].set_yticks([]); plt.colorbar(im, ax=ax[2], fraction=0.046)
    im = ax[3].imshow(err[0], cmap='magma'); ax[3].set_title('JEPA error  %s' % names[0])
    ax[3].set_xticks([]); ax[3].set_yticks([]); plt.colorbar(im, ax=ax[3], fraction=0.046)

    fig.suptitle('Is anatomy harder or easier for JEPA to predict? (decides scorer sign)',
                 fontsize=13)
    fig.tight_layout()
    p = out / 'error_vs_anatomy.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % p)


if __name__ == '__main__':
    main()
