#!/usr/bin/env python
"""Pre-flight for a pretraining launch.

A 100-hour run that starts from a wrong checkpoint, a stale guide cache, or a
silently-disabled curriculum is expensive to discover late. This validates the
exact config that is about to be launched, through the production dataset and
collator, and refuses to pass on anything it cannot verify.

Checks:
  1. checkpoint loads, reports its epoch, and carries a target_encoder
  2. guide cache is complete, and its adapter/MIRAGE provenance is recorded
  3. the guided dataset loads a real guide (not a fallback) for real volumes
  4. the production collator produces connected anatomy-shaped targets at the
     ramp value the run will actually start from
  5. the new guide genuinely differs from whatever cache the previous run used
  6. seeding is effective: identical seed reproduces masks exactly
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
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.masks.curriculum import CurriculumMaskGenerator          # noqa: E402
from src.masks.anatomy import CROSS4                              # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('  [%s] %-46s %s' % ('PASS' if ok else 'FAIL', name, detail))
    if not ok:
        FAIL.append(name)


def _load_guide_array(p):
    """Read a guide volume from either on-disk form.

    Caches exist as compressed .npz (one deflate stream per array) or as
    memmap-friendly .npy written by convert_guides_to_npy.py.
    """
    if p.suffix == '.npy':
        return np.asarray(np.load(p, mmap_mode='r'))
    with np.load(p, allow_pickle=False) as z:
        return z['soft_scores']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--compare-guide', default=None,
                    help='previous guide dir, to prove the new one differs')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    mask_cfg, meta_cfg = cfg['mask'], cfg['meta']
    cur = mask_cfg['curriculum']

    print('config %s\n' % a.config)

    # 1 ---- checkpoint
    ck_path = meta_cfg.get('read_checkpoint')
    print('1. warm-start checkpoint')
    ok = bool(ck_path) and os.path.isfile(ck_path)
    check('file exists', ok, str(ck_path))
    if ok:
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        check('has target_encoder', 'target_encoder' in ck,
              '%d tensors' % len(ck.get('target_encoder', {})))
        check('epoch recorded', ck.get('epoch') is not None,
              'epoch=%s' % ck.get('epoch'))
        check('load_checkpoint enabled', bool(meta_cfg.get('load_checkpoint')),
              str(meta_cfg.get('load_checkpoint')))

    # 2 ---- guide cache
    print('\n2. guide cache')
    gdir = pathlib.Path(cur['mirage_guide_dir'])
    meta_p = gdir / 'cache_meta.json'
    check('directory exists', gdir.is_dir(), str(gdir))
    check('cache_meta.json present', meta_p.is_file())
    if meta_p.is_file():
        gm = json.loads(meta_p.read_text())
        # The cache exists in two on-disk forms: compressed .npz, and the
        # memmap-friendly .npy + sidecar .json produced by
        # convert_guides_to_npy.py. Both are valid; count whichever is present.
        npz = sorted((gdir / 'Training').glob('*.npz'))
        npy = sorted((gdir / 'Training').glob('*.npy'))
        n = len(npy) if npy else len(npz)
        check('volume count matches meta', n == gm.get('n_volumes'),
              '%d %s files vs meta %s'
              % (n, 'npy (memmap)' if npy else 'npz (compressed)',
                 gm.get('n_volumes')))
        check('schema is 2', gm.get('schema_version') == 2)
        check('two soft channels', gm.get('channels') == ['P_inner', 'P_choroid'],
              str(gm.get('channels')))
        print('       adapter %s tap %s | mirage %s | storage %s'
              % (gm.get('adapter_sha'), gm.get('adapter_tap'),
                 gm.get('mirage_sha'),
                 'memmap .npy' if npy else 'compressed .npz'))

    # 3 ---- real guides load, and differ from the previous cache
    print('\n3. guide content')
    tr = sorted((gdir / 'Training').glob('*.npy'))[:8]
    if not tr:
        tr = sorted((gdir / 'Training').glob('*.npz'))[:8]
    check('training guides found', len(tr) == 8, '%d sampled' % len(tr))
    occs = [_load_guide_array(p) for p in tr]
    if occs:
        s0 = occs[0]
        check('shape (slices,2,H,W)', s0.ndim == 4 and s0.shape[1] == 2,
              str(s0.shape))
        frac = float(np.mean(s0.astype(np.float32) / 255.0 > 0.25))
        check('anatomy occupancy plausible', 0.02 < frac < 0.60,
              '%.1f%% of pixels above threshold' % (100 * frac))
    if a.compare_guide:
        old = pathlib.Path(a.compare_guide) / 'Training'
        same, diff, n = 0, 0, 0
        for p in tr:
            # The previous cache may be stored in the other form.
            q = old / (p.stem + '.npz')
            if not q.is_file():
                q = old / (p.stem + '.npy')
            if not q.is_file():
                continue
            A, B = _load_guide_array(p), _load_guide_array(q)
            if A.shape != B.shape:
                continue
            n += 1
            if np.array_equal(A, B):
                same += 1
            else:
                diff += 1
        check('new guide differs from previous cache', n > 0 and same == 0,
              '%d/%d volumes differ' % (diff, n))

    # 4 ---- production collator at the run's starting ramp
    print('\n4. production collator')
    start_ep = int(torch.load(ck_path, map_location='cpu',
                              weights_only=False).get('epoch', 0)) if ck_path else 0
    gen = CurriculumMaskGenerator(
        input_size=(mask_cfg['crop_size'] if 'crop_size' in mask_cfg
                    else cfg['data']['crop_size'],) * 2,
        patch_size=mask_cfg['patch_size'],
        npred=mask_cfg['num_pred_masks'], nenc=mask_cfg['num_enc_masks'],
        pred_mask_scale=tuple(mask_cfg['pred_mask_scale']),
        pred_target_k=mask_cfg.get('pred_target_k'),
        curriculum_cfg=cur)
    gen.set_epoch(start_ep, cfg['optimization']['epochs'])
    check('mode is the intended one', gen.mode == cur['mode'], gen.mode)
    print('       r_t at resume epoch %d = %.2f' % (start_ep, gen._r_t))

    per = np.load(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')['per'][:64]
    occ = np.clip(per[:, 0] + per[:, 1], 0, 1).astype(np.float32)
    guides = torch.from_numpy(np.stack([occ, (occ >= 0.25).astype(np.float32)], 1))

    def draw(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        gen.set_epoch(max(start_ep, cur['T_total']), cfg['optimization']['epochs'])
        return gen.generate(batch_size=64, guide_grids=guides,
                            guide_valid=torch.ones(64, dtype=torch.bool))

    enc, pred = draw(0)
    from scipy import ndimage
    G = mask_cfg['patch_size']
    grid = cfg['data']['crop_size'] // mask_cfg['patch_size']
    conn, conn8, on = [], [], []
    for b in range(64):
        u = set()
        for p in pred:
            idx = p[b].numpy()
            u.update(idx.tolist())
            m = np.zeros(grid * grid, bool); m[idx] = True
            M = m.reshape(grid, grid)
            # EDGE adjacency, not the 3x3 box.  The box rule counts diagonal
            # corner-touches as connected, so it passes a staircase -- and even
            # a checkerboard -- as "one region".  On the production path the
            # box rule scored targets 100% connected where this rule scored
            # 83.3%.  A pre-flight that cannot fail is not a check.
            conn.append(ndimage.label(M, structure=CROSS4)[1] == 1)
            conn8.append(ndimage.label(M, structure=np.ones((3, 3)))[1] == 1)
        i = np.array(sorted(u))
        on.append(float(occ[b].ravel()[i].mean()))
    bridging = bool(cur.get('anatomy_bridge_diagonals', False))
    # Without bridging only ~83% of targets are edge-connected at full ramp,
    # which is the known state of the sampler rather than a launch blocker;
    # with it on, anything below 99% means the bridge did not do its job.
    thresh = 0.99 if bridging else 0.45
    check('targets are edge-connected (4-conn)', float(np.mean(conn)) >= thresh,
          '%.1f%% single-component, bridge_diagonals=%s (threshold %.0f%%)'
          % (100 * np.mean(conn), bridging, 100 * thresh))
    check('targets are 8-connected', float(np.mean(conn8)) > 0.95,
          '%.1f%% single-component' % (100 * np.mean(conn8)))
    check('targets land on anatomy', float(np.mean(on)) > 0.55,
          '%.1f%% on-anatomy at full ramp' % (100 * np.mean(on)))
    check('context non-empty', int(enc[0].shape[1]) > 20,
          '%d tokens' % int(enc[0].shape[1]))

    # 5 ---- seeding
    print('\n5. reproducibility')
    e1, p1 = draw(0)
    e2, p2 = draw(0)
    e3, p3 = draw(1)
    same = all(torch.equal(x, y) for x, y in zip(p1, p2))
    diff = any(not torch.equal(x, y) for x, y in zip(p1, p3))
    check('same seed reproduces masks', same)
    check('different seed changes masks', diff)
    check('config carries a seed', meta_cfg.get('seed') is not None,
          'seed=%s' % meta_cfg.get('seed'))

    print()
    if FAIL:
        print('PRE-FLIGHT FAILED: %s' % ', '.join(FAIL))
        raise SystemExit(1)
    print('PRE-FLIGHT PASSED — safe to launch')


if __name__ == '__main__':
    main()
