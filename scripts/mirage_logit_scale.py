"""Measure MIRAGE's raw logit scale L0, to choose alpha in L = L0 + alpha*tanh(dL).

The residual-adapter design gives the trainable head a total authority of
+/- alpha in LOGIT units, because tanh bounds dL to (-1, 1).  Whether that is
negligible or overwhelming depends entirely on the scale of MIRAGE's own
logits, which is what this script measures.  alpha must be read off the data,
not guessed.

Captured tensor: the output of ``output_adapters.semseg.final_layer``, i.e.
L0 in R^{4 x 128 x 128}, BEFORE the bilinear x8 upsample
(MIRAGE/mirage/output_adapters.py:512-515).  That is the true segmentation
resolution and the point where dL would be added.

V3-merged taxonomy:  0 Elsewhere | 1 InnerRetina | 2 Choroid | 3 Background/ignore
Anatomy score:       M = P1 + P2   (softmax over classes FIRST, then pool)
JEPA grid:           128 -> 16 by 8x8 average pooling

Two stages, because MIRAGE inference needs the MIRAGE venv (no matplotlib):
  stage 1  --dump logits.npz        (MIRAGE venv)
  stage 2  --analyze-from logits.npz  (repo venv)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

CKPT = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3\MergedV3'
        r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth')
TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')

CLASS_NAMES = ('Elsewhere', 'InnerRetina', 'Choroid', 'Background')
ANATOMY = (1, 2)
NON_ANATOMY = (0, 3)
RES = 1024
GRID = 16          # JEPA patch grid
POOL = 8           # 128 // 16
ALPHAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def pick_slices(n: int, seed: int = 0) -> list:
    """Deterministic spread of volume:slice pairs across the Test split."""
    vols = sorted(p.stem for p in TEST.glob('data_*.npz'))
    if not vols:
        raise SystemExit('no Test volumes under %s' % TEST)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(vols), size=min(n, len(vols)), replace=False)
    picks = []
    for i, vi in enumerate(sorted(chosen)):
        # spread slice depth across the volume rather than always the centre
        frac = (i + 0.5) / len(chosen)
        picks.append((vols[vi], frac))
    return picks


def dump(out_path: pathlib.Path, n_slices: int):
    """Stage 1: run MIRAGE and save raw L0 for every slice."""
    import os
    import torch
    from compare_512_vs_1024 import build
    from fairvision_model_compare import MIRAGE_WS
    import cv2

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build(RES, CKPT, device)

    captured = {}
    layer = model.output_adapters['semseg'].final_layer
    handle = layer.register_forward_hook(
        lambda m, i, o: captured.__setitem__('L0', o.detach().float().cpu()))

    picks = pick_slices(n_slices)
    logits, names = [], []
    for vol_id, frac in picks:
        with np.load(TEST / ('%s.npz' % vol_id), allow_pickle=True) as z:
            vol = z['oct_bscans']
        d = int(frac * (len(vol) - 1))
        raw = np.asarray(vol[d], dtype=np.float32)
        lo, hi = raw.min(), raw.max()
        unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        img = cv2.resize(unit, (RES, RES), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img)[None, None].to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            model({'bscan': t})            # fp32 on purpose: measuring logit scale
        logits.append(captured['L0'][0].numpy())
        names.append('%s:%d' % (vol_id, d))
        print('  %s  L0 %s' % (names[-1], tuple(logits[-1].shape)))

    handle.remove()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, logits=np.stack(logits), names=np.array(names))
    print('wrote %s  logits %s' % (out_path, np.stack(logits).shape))


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def pool(m: np.ndarray) -> np.ndarray:
    """(N,128,128) -> (N,16,16) mean over 8x8 cells."""
    n, h, w = m.shape
    return m.reshape(n, h // POOL, POOL, w // POOL, POOL).mean(axis=(2, 4))


def analyze(npz_path: pathlib.Path, out_dir: pathlib.Path):
    """Stage 2: all statistics + figure, in the repo venv."""
    z = np.load(npz_path, allow_pickle=True)
    L0, names = z['logits'], [str(s) for s in z['names']]
    n = L0.shape[0]
    rep = {'n_slices': n, 'logit_shape': list(L0.shape[1:]), 'checkpoint': CKPT}

    # --- 1. logit dynamic range -------------------------------------------
    qs = [0, 1, 50, 99, 100]
    rep['logit_percentiles'] = {
        CLASS_NAMES[c]: {f'p{q}': float(np.percentile(L0[:, c], q)) for q in qs}
        for c in range(4)
    }
    rep['logit_abs_max'] = float(np.abs(L0).max())
    rep['logit_std'] = float(L0.std())

    srt = np.sort(L0, axis=1)
    margin = srt[:, 3] - srt[:, 2]          # winner minus runner-up
    rep['winner_margin'] = {f'p{q}': float(np.percentile(margin, q))
                            for q in (1, 10, 50, 90, 99)}

    # --- 2. anatomy probability -------------------------------------------
    P = softmax(L0, axis=1)
    M = P[:, ANATOMY].sum(axis=1)           # (N,128,128) in [0,1]
    rep['M_pixel'] = {'mean': float(M.mean()),
                      **{f'p{q}': float(np.percentile(M, q)) for q in (1, 25, 50, 75, 99)}}
    rep['M_saturated_high_frac'] = float((M > 0.99).mean())
    rep['M_saturated_low_frac'] = float((M < 0.01).mean())
    rep['M_soft_frac'] = float(((M >= 0.01) & (M <= 0.99)).mean())

    # --- 3. pooled JEPA-grid score vs the hard occupancy it replaces -------
    soft = pool(M)
    hard = pool((np.isin(L0.argmax(axis=1), ANATOMY)).astype(np.float32))
    rep['pooled_soft'] = {'mean': float(soft.mean()), 'std': float(soft.std())}
    rep['pooled_hard'] = {'mean': float(hard.mean()), 'std': float(hard.std())}
    rep['pooled_corr'] = float(np.corrcoef(soft.ravel(), hard.ravel())[0, 1])
    rep['pooled_mean_abs_diff'] = float(np.abs(soft - hard).mean())
    won = L0.argmax(axis=1)
    rep['class_win_frac'] = {CLASS_NAMES[c]: float((won == c).mean()) for c in range(4)}

    # --- 4. how much authority does alpha actually buy? -------------------
    # Best case for the adapter: push all anatomy logits up and all
    # non-anatomy logits down by the full tanh range, and vice versa.
    d = np.zeros((1, 4, 1, 1), dtype=np.float32)
    d[0, list(ANATOMY)] = 1.0
    d[0, list(NON_ANATOMY)] = -1.0
    sweep = {}
    for a in ALPHAS:
        up = softmax(L0 + a * d, axis=1)[:, ANATOMY].sum(axis=1)
        dn = softmax(L0 - a * d, axis=1)[:, ANATOMY].sum(axis=1)
        auth = pool(up) - pool(dn)          # (N,16,16) full control range
        sweep[str(a)] = {
            'mean_authority': float(auth.mean()),
            'p90_authority': float(np.percentile(auth, 90)),
            'dead_cell_frac': float((auth < 0.01).mean()),
            'pooled_mean_up': float(pool(up).mean()),
            'pooled_mean_dn': float(pool(dn).mean()),
        }
    rep['alpha_sweep'] = sweep

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'mirage_logit_scale.json').write_text(json.dumps(rep, indent=2))

    _figure(out_dir, L0, M, soft, hard, margin, sweep, names)

    # --- console summary --------------------------------------------------
    print('slices: %d   L0 shape: %s' % (n, tuple(L0.shape[1:])))
    print('\n--- 1. logit dynamic range ---')
    for c in range(4):
        p = rep['logit_percentiles'][CLASS_NAMES[c]]
        print('  %-12s p0 %8.2f  p50 %8.2f  p100 %8.2f' %
              (CLASS_NAMES[c], p['p0'], p['p50'], p['p100']))
    print('  |L0| max %.2f   std %.2f' % (rep['logit_abs_max'], rep['logit_std']))
    m = rep['winner_margin']
    print('  winner-runnerup margin  p1 %.2f  p50 %.2f  p99 %.2f'
          % (m['p1'], m['p50'], m['p99']))
    print('\n--- 2. anatomy probability M = P1+P2 ---')
    print('  mean %.4f   saturated>0.99 %.1f%%   saturated<0.01 %.1f%%   soft %.1f%%'
          % (rep['M_pixel']['mean'], 100 * rep['M_saturated_high_frac'],
             100 * rep['M_saturated_low_frac'], 100 * rep['M_soft_frac']))
    print('\n--- 3. pooled 16x16 score ---')
    print('  soft mean %.4f   hard mean %.4f   corr %.4f   mean|diff| %.4f'
          % (rep['pooled_soft']['mean'], rep['pooled_hard']['mean'],
             rep['pooled_corr'], rep['pooled_mean_abs_diff']))
    print('  argmax wins: ' + '  '.join('%s %.3f' % (k, v)
                                        for k, v in rep['class_win_frac'].items()))
    print('\n--- 4. adapter authority vs alpha (full tanh range) ---')
    print('  %6s  %14s  %14s  %12s' % ('alpha', 'mean authority', 'p90 authority',
                                       'dead cells'))
    for a in ALPHAS:
        s = sweep[str(a)]
        print('  %6.1f  %14.4f  %14.4f  %11.1f%%'
              % (a, s['mean_authority'], s['p90_authority'],
                 100 * s['dead_cell_frac']))
    print('\nwrote %s' % (out_dir / 'mirage_logit_scale.json'))


def _figure(out_dir, L0, M, soft, hard, margin, sweep, names):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; skipping figure')
        return

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    for c in range(4):
        ax[0, 0].hist(L0[:, c].ravel(), bins=120, histtype='step',
                      label=CLASS_NAMES[c], density=True)
    ax[0, 0].set_title('1. raw logits $L_0$ per class')
    ax[0, 0].set_xlabel('logit'); ax[0, 0].legend(fontsize=8); ax[0, 0].set_yscale('log')

    ax[0, 1].hist(margin.ravel(), bins=120, color='tab:purple', density=True)
    ax[0, 1].set_title('winner $-$ runner-up margin\n'
                       r'(how far $\alpha\cdot\tanh$ must reach to flip a pixel)')
    ax[0, 1].set_xlabel('logit margin'); ax[0, 1].set_yscale('log')

    ax[0, 2].hist(M.ravel(), bins=120, color='tab:green', density=True)
    ax[0, 2].set_title('2. anatomy prob $M=P_1+P_2$\n'
                       f'soft (0.01–0.99): {100*((M>=.01)&(M<=.99)).mean():.1f}%')
    ax[0, 2].set_xlabel('M'); ax[0, 2].set_yscale('log')

    ax[1, 0].scatter(hard.ravel(), soft.ravel(), s=2, alpha=0.15, color='tab:blue')
    ax[1, 0].plot([0, 1], [0, 1], 'k--', lw=1)
    ax[1, 0].set_xlabel('hard occupancy (argmax)'); ax[1, 0].set_ylabel('soft score')
    ax[1, 0].set_title('3. pooled 16x16: soft vs hard\n'
                       f'corr {np.corrcoef(soft.ravel(), hard.ravel())[0,1]:.4f}')

    al = [float(a) for a in sweep]
    ax[1, 1].plot(al, [sweep[str(a)]['mean_authority'] for a in al], 'o-',
                  label='mean')
    ax[1, 1].plot(al, [sweep[str(a)]['p90_authority'] for a in al], 's-',
                  label='p90')
    ax[1, 1].axhline(0.05, color='r', ls='--', lw=1, label='0.05 (usable?)')
    ax[1, 1].set_xscale('log'); ax[1, 1].set_xlabel(r'$\alpha$')
    ax[1, 1].set_ylabel('pooled-score control range')
    ax[1, 1].set_title(r'4. adapter authority vs $\alpha$')
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    im = ax[1, 2].imshow(soft[0], cmap='viridis', vmin=0, vmax=1)
    ax[1, 2].set_title('example pooled 16x16 score\n%s' % names[0])
    ax[1, 2].set_xticks([]); ax[1, 2].set_yticks([])
    plt.colorbar(im, ax=ax[1, 2], fraction=0.046)

    fig.suptitle(r'MIRAGE V3-merged @1024: logit scale and residual-adapter authority',
                 fontsize=13)
    fig.tight_layout()
    p = out_dir / 'mirage_logit_scale.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path, help='stage 1 (MIRAGE venv)')
    ap.add_argument('--analyze-from', type=pathlib.Path, help='stage 2 (repo venv)')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out', type=pathlib.Path,
                    default=pathlib.Path('results/masking/logit_scale'))
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump, a.n)
    if a.analyze_from:
        return analyze(a.analyze_from, a.out)
    raise SystemExit('need --dump or --analyze-from')


if __name__ == '__main__':
    main()
