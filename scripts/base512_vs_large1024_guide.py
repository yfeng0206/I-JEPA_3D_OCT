"""MIRAGE-Base @512 as the in-loop guide: is it equivalent, and can it be afforded?

The ViT-B/512 model is a far better fit for in-loop guidance than ViT-L/1024:

    params        95.6M   vs 315.5M          (3.3x smaller)
    pos_emb       (1,768,16,16) NATIVE       vs (1,1024,32,32) interpolated
    decoder out   64x64 -> pool 4x4 -> 16x16 vs 128x128 -> pool 8x8 -> 16x16
    decoder dim   384                        vs 384   (identical)

The native pos_emb matters: the 1024 arm interpolates its position table UP from
the released 16x16, and regenerating rather than interpolating it was previously
measured to drop Inner Dice 0.969 -> 0.606.  ViT-B/512 needs no interpolation at
all, so that whole failure mode disappears.

Two questions this script answers:

  Q1  EQUIVALENCE -- do the two models produce the same 16x16 anatomy grid?
      That grid is all the sampler ever sees, so if they agree there, the
      cheaper model is strictly better for guidance.

  Q2  AFFORDABILITY -- what does running MIRAGE in-loop actually cost?
      Measures VRAM and per-image latency for both, which is open blocker B3.

Stage 1 (MIRAGE venv, GPU): --dump    Stage 2 (repo venv): --analyze-from
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

TEST = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Test')
CK_LARGE = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3\MergedV3'
            r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth')
CK_BASE = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3-base-512\MergedV3'
           r'\MIRAGE-Base_frozen_convnext_CEGDice-ignore\checkpoint-best.pth')
ANATOMY, GRID = (1, 2), 16


def build(size: str, res: int, ckpt: str, device: str):
    """MIRAGE at an arbitrary square resolution.

    pos_emb is INTERPOLATED from the checkpoint when the grid differs, never
    regenerated: despite learnable_pos_emb=False those tables differ from fresh
    sin-cos by up to 2.003, and regenerating drops Inner Dice 0.969 -> 0.606.
    ViT-B/512 hits its native 16x16 grid, so no interpolation happens at all.
    """
    from argparse import Namespace
    from fairvision_model_compare import MIRAGE_WS
    if str(MIRAGE_WS / 'MIRAGE') not in sys.path:
        sys.path.insert(0, str(MIRAGE_WS / 'MIRAGE'))
    from fm_seg_config import fm_factory
    from mirage.model import model_factory
    from mirage.output_adapters import ConvNeXtAdapter

    g = res // 32
    cfg = fm_factory['mirage-%s' % size]()
    cfg.build_domain_conf()
    ra = Namespace(grid_sizes={'bscan': [g, g]}, input_size={'bscan': [res, res]})
    ia = {'bscan': cfg.domain_conf['bscan']['input_adapter'](
        stride_level=1, patch_size_full=[32, 32], image_size=[res, res],
        learnable_pos_emb=False)}
    oa = {'semseg': ConvNeXtAdapter(
        num_classes=4, preds_per_patch=16, depth=4, interpolate_mode='bilinear',
        main_tasks=['bscan'], embed_dim=6144, patch_size=[32, 32],
        task='semseg', image_size=[res, res])}
    model = model_factory[cfg.model](
        args=ra, input_adapters=ia, output_adapters=oa,
        num_global_tokens=1, drop_path_rate=0.1)
    sd = dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'])
    pe = sd['input_adapters.bscan.pos_emb']
    interpolated = pe.shape[-1] != g
    if interpolated:
        sd['input_adapters.bscan.pos_emb'] = F.interpolate(
            pe.float(), size=(g, g), mode='bicubic', align_corners=False)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), interpolated


def dump(out_path, n_slices, bench_iters):
    import os
    import cv2
    from fairvision_model_compare import MIRAGE_WS

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arms = {'large_1024': ('large', 1024, CK_LARGE), 'base_512': ('base', 512, CK_BASE)}

    vols = sorted(p.stem for p in TEST.glob('data_*.npz'))[:n_slices]
    imgs, names = [], []
    for vi, vol_id in enumerate(vols):
        with np.load(TEST / ('%s.npz' % vol_id), allow_pickle=True) as z:
            vol = z['oct_bscans']
        d = int((vi + 0.5) / len(vols) * (len(vol) - 1))
        raw = np.asarray(vol[d], dtype=np.float32)
        lo, hi = raw.min(), raw.max()
        imgs.append((raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw))
        names.append('%s:%d' % (vol_id, d))

    out, meta = {}, {}
    for tag, (size, res, ck) in arms.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated() / 2**20
        model, interp = build(size, res, ck, device)
        grab = {}
        model.output_adapters['semseg'].final_layer.register_forward_hook(
            lambda m, i, o: grab.update(H=i[0].detach(), L0=o.detach()))
        model_mem = torch.cuda.memory_allocated() / 2**20 - base_mem

        grids, hpool = [], []
        for im in imgs:
            s = cv2.resize(im, (res, res), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(s)[None, None].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                model({'bscan': x})
            M = grab['L0'].softmax(dim=1)[:, ANATOMY].sum(dim=1)
            grids.append(F.adaptive_avg_pool2d(M[:, None], (GRID, GRID))[0, 0].cpu().numpy())
            hpool.append(F.adaptive_avg_pool2d(grab['H'], (GRID, GRID))[0]
                         .permute(1, 2, 0).reshape(-1, grab['H'].shape[1]).cpu().numpy())

        # ---- Q2: latency + peak VRAM under a realistic in-loop batch --------
        bench = {}
        for bs in (1, 8, 16):
            x = torch.randn(bs, 1, res, res, device=device)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():                       # warmup
                for _ in range(3):
                    model({'bscan': x})
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(bench_iters):
                    model({'bscan': x})
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / bench_iters
            bench['bs%d' % bs] = {
                'sec_per_batch': dt, 'img_per_sec': bs / dt,
                'peak_vram_mb': torch.cuda.max_memory_allocated() / 2**20,
            }
            del x
        out[tag + '_grid'] = np.stack(grids)
        out[tag + '_hpool'] = np.stack(hpool)
        meta[tag] = {
            'size': size, 'res': res, 'checkpoint': ck,
            'pos_emb_interpolated': bool(interp),
            'params_M': sum(p.numel() for p in model.parameters()) / 1e6,
            'model_vram_mb': model_mem,
            'decoder_out': list(grab['L0'].shape[-2:]),
            'pool_factor': grab['L0'].shape[-1] // GRID,
            'bench': bench,
        }
        print('%s  params %.1fM  interp=%s  decoder %s  pool %dx%d'
              % (tag, meta[tag]['params_M'], interp,
                 meta[tag]['decoder_out'], meta[tag]['pool_factor'],
                 meta[tag]['pool_factor']))
        for bs, b in bench.items():
            print('   %s  %.1f img/s  peak %.0f MB' % (bs, b['img_per_sec'], b['peak_vram_mb']))
        del model, grab
        torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, names=np.array(names), meta=json.dumps(meta), **out)
    print('wrote %s' % out_path)


def analyze(npz_path, out_dir):
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z['meta']))
    gL, gB = z['large_1024_grid'], z['base_512_grid']
    n = len(gL)
    rep = {'n_slices': n, 'meta': meta}

    rep['grid_equivalence'] = {
        'mean_large': float(gL.mean()), 'mean_base': float(gB.mean()),
        'corr_global': float(np.corrcoef(gL.ravel(), gB.ravel())[0, 1]),
        'corr_per_slice_mean': float(np.mean(
            [np.corrcoef(gL[i].ravel(), gB[i].ravel())[0, 1] for i in range(n)])),
        'mean_abs_diff': float(np.abs(gL - gB).mean()),
    }
    # what the sampler actually consumes: the admissible-cell SET
    for thL, thB in ((0.10, 0.10), (0.10, 0.15), (0.10, 0.20)):
        A, B = gL >= thL, gB >= thB
        iou = ((A & B).sum(axis=(1, 2)) / (A | B).sum(axis=(1, 2)).clip(min=1)).mean()
        rep['grid_equivalence']['iou_L%.2f_B%.2f' % (thL, thB)] = float(iou)
        rep['grid_equivalence']['cells_L%.2f' % thL] = float(A.sum(axis=(1, 2)).mean())
        rep['grid_equivalence']['cells_B%.2f' % thB] = float(B.sum(axis=(1, 2)).mean())

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'base512_vs_large1024.json').write_text(json.dumps(rep, indent=2))

    e = rep['grid_equivalence']
    print('=== MIRAGE-Base@512 vs MIRAGE-Large@1024 as the guide ===')
    for tag in ('large_1024', 'base_512'):
        m = meta[tag]
        print('  %-11s %6.1fM params   pos_emb interpolated: %-5s  decoder %s -> pool %dx%d'
              % (tag, m['params_M'], m['pos_emb_interpolated'],
                 m['decoder_out'], m['pool_factor'], m['pool_factor']))
    print('\n  Q1 equivalence of the 16x16 grid the sampler sees:')
    print('     mean occupancy  large %.4f   base %.4f' % (e['mean_large'], e['mean_base']))
    print('     correlation     global %.4f   per-slice %.4f'
          % (e['corr_global'], e['corr_per_slice_mean']))
    print('     mean |diff|     %.4f' % e['mean_abs_diff'])
    for k in sorted(k for k in e if k.startswith('iou_')):
        print('     %-22s %.4f' % (k, e[k]))
    for k in sorted(k for k in e if k.startswith('cells_')):
        print('     %-22s %.1f / 256' % (k, e[k]))
    print('\n  Q2 in-loop cost (open blocker B3):')
    print('     %-11s %8s %12s %14s' % ('arm', 'batch', 'img/s', 'peak VRAM MB'))
    for tag in ('large_1024', 'base_512'):
        for bs, b in meta[tag]['bench'].items():
            print('     %-11s %8s %12.1f %14.0f' % (tag, bs, b['img_per_sec'], b['peak_vram_mb']))
    print('\nwrote %s' % (out_dir / 'base512_vs_large1024.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path)
    ap.add_argument('--analyze-from', type=pathlib.Path)
    ap.add_argument('--n-slices', type=int, default=30)
    ap.add_argument('--bench-iters', type=int, default=20)
    ap.add_argument('--out', type=pathlib.Path,
                    default=REPO / 'results/masking/base512')
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump, a.n_slices, a.bench_iters)
    if a.analyze_from:
        return analyze(a.analyze_from, a.out)
    raise SystemExit('need --dump or --analyze-from')


if __name__ == '__main__':
    main()
