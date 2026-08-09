#!/usr/bin/env python3
"""Does refreshing the adapter against a newer JEPA teacher change the guide?

This decides whether the guide should be rebuilt periodically during training
or trained once and frozen.

    adapter A   taught by jepa_patch_mirage-ep100   (RECTANGLE-masked teacher)
    adapter B   taught by patch_mirage_anatomy-ep30 (ANATOMY-masked teacher)

Both start from the same frozen MIRAGE and the same zero-init.  If B's guide
is materially different from A's, the teacher matters and a periodic refresh
buys something.  If the masks are near-identical, one-time training is
sufficient and a refresh loop is 17 h of compute for nothing.

Reports agreement at three layers, because a difference can be real at the
feature level and vanish by the time it reaches the mask:

    features   relative L2 distance between adapted H
    score      mean |delta| on the 16x16 anatomy score
    MASK       Jaccard of the final 4-target union   <- the one that matters
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

import src.masks.anatomy as A                                # noqa: E402
from adapter_stage import Adapter                            # noqa: E402
from jepa_to_mirage_probe import build_mirage                # noqa: E402

CACHE = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\slice_pos')
OUT = REPO / 'results/masking/adapter_refresh'
GRID, ANATOMY = 16, (1, 2)
A_CK = REPO / 'results/masking/adapter_stage/adapter_cfg7.pt'
B_CK = REPO / 'results/masking/adapter_stage/adapter_cfg7_ep30teacher.pt'


def load(p, dev):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    m = Adapter(**ck['cfg']).to(dev)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, ck


def union_of(score2):
    cs = [score2[0], score2[1]]
    if not A.is_viable(cs):
        return None
    parts, _ = A.build_targets(cs)
    return np.logical_or.reduce(parts)


def main(n=192, batch=16):
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    idx_all = np.sort(rng.choice(len(im512), n, replace=False))

    mir = build_mirage(dev)
    grab = {}
    head = mir.output_adapters['semseg'].final_layer
    head.register_forward_hook(lambda m, i, o: grab.update(H=i[0].detach()))

    mA, ckA = load(A_CK, dev)
    mB, ckB = load(B_CK, dev)
    print('adapter A taught by %s' % pathlib.Path(ckA['jepa_ckpt']).name)
    print('adapter B taught by %s' % pathlib.Path(ckB['jepa_ckpt']).name)
    print()

    dfeat, dscore, jac, ident, viab = [], [], [], 0, 0
    cellsA, cellsB, agree = [], [], []
    for s in range(0, n, batch):
        i = idx_all[s:s + batch]
        x = torch.from_numpy(np.asarray(im512[i], np.float32) / 255.)[:, None].to(dev)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16):
                mir({'bscan': x})
            H0 = grab['H'].float()
            HA, HB = mA(H0), mB(H0)
            dfeat.append(float((HA - HB).norm() / H0.norm()))
            LA, LB = head(HA).float(), head(HB).float()
            gA = F.adaptive_avg_pool2d(LA.softmax(1)[:, ANATOMY], (GRID, GRID)).cpu().numpy()
            gB = F.adaptive_avg_pool2d(LB.softmax(1)[:, ANATOMY], (GRID, GRID)).cpu().numpy()
            agree.append(float((LA.argmax(1) == LB.argmax(1)).float().mean()))
        dscore.append(float(np.abs(gA - gB).mean()))
        for j in range(len(i)):
            uA, uB = union_of(gA[j]), union_of(gB[j])
            if uA is None or uB is None:
                continue
            viab += 1
            inter = np.logical_and(uA, uB).sum()
            un = np.logical_or(uA, uB).sum()
            jac.append(inter / max(un, 1))
            ident += int((uA == uB).all())
            cellsA.append(int(uA.sum())); cellsB.append(int(uB.sum()))

    res = {
        'n': int(n), 'viable': int(viab),
        'adapter_A_teacher': str(ckA['jepa_ckpt']),
        'adapter_B_teacher': str(ckB['jepa_ckpt']),
        'feature_rel_l2': float(np.mean(dfeat)),
        'score_abs_diff': float(np.mean(dscore)),
        'seg_argmax_agreement': float(np.mean(agree)),
        'mask_jaccard': float(np.mean(jac)),
        'mask_jaccard_min': float(np.min(jac)),
        'identical_masks_pct': 100.0 * ident / max(viab, 1),
        'cells_A': float(np.mean(cellsA)), 'cells_B': float(np.mean(cellsB)),
    }
    print('%-34s %s' % ('slices compared', viab))
    print('%-34s %.4f' % ('feature relative L2  A vs B', res['feature_rel_l2']))
    print('%-34s %.5f' % ('16x16 score mean |diff|', res['score_abs_diff']))
    print('%-34s %.4f' % ('seg argmax agreement', res['seg_argmax_agreement']))
    print('%-34s %.4f  (min %.3f)' % ('FINAL MASK Jaccard',
                                      res['mask_jaccard'], res['mask_jaccard_min']))
    print('%-34s %.1f%%' % ('identical masks', res['identical_masks_pct']))
    print('%-34s %.1f vs %.1f' % ('union cells  A vs B',
                                  res['cells_A'], res['cells_B']))
    print()
    j = res['mask_jaccard']
    if j > 0.95:
        print('VERDICT: the teacher barely matters. Jaccard %.4f means a refreshed' % j)
        print('  adapter reproduces almost the same mask, so periodic rebuilds would')
        print('  cost compute for no change in what JEPA sees.')
    elif j > 0.85:
        print('VERDICT: modest change (Jaccard %.4f). A refresh moves the guide but' % j)
        print('  most of the mask is unchanged; worth doing only if cheap.')
    else:
        print('VERDICT: the teacher MATTERS (Jaccard %.4f). The guide a refreshed' % j)
        print('  adapter produces is substantially different, so freezing after one')
        print('  training pass discards real signal.')
    (OUT / 'adapter_refresh.json').write_text(json.dumps(res, indent=2))
    print()
    print('wrote', OUT / 'adapter_refresh.json')


if __name__ == '__main__':
    main()
