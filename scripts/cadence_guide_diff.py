#!/usr/bin/env python
"""How different are the masking guides produced by adapters from different
I-JEPA teacher epochs?

The refresh-cadence question is not "does the adapter change with the teacher"
but "does the GUIDE THE SAMPLER CONSUMES change".  If adapter(ep30) and
adapter(ep100) produce the same 16x16 occupancy grids, refreshing between them
buys nothing and costs a 66-minute cache rebuild.

Reference points for reading the Jaccard values:
  adapter vs NO adapter          0.737   (measured previously; a real change)
  cache rebuild cost             66 min
"""
from __future__ import annotations

import argparse
import itertools
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

from adapter_placement_ablation import Adapter                       # noqa: E402
from goals_eval import VOID, RES                                     # noqa: E402
from jepa_to_mirage_probe import build_mirage                        # noqa: E402

CACHE = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\slice_pos')
SRC = REPO / 'results/masking/structural_loss'
GRID, OCC = 16, 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--adapters', default=('ep30_structl100_a010,ep50_structl100_a010,'
                                           'ep75_structl100_a010,ep100_structl100_a010'))
    a = ap.parse_args()
    dev = 'cuda'
    mir = build_mirage(dev)
    sem = mir.output_adapters['semseg']

    im512 = np.load(CACHE / 'im512.npy', mmap_mode='r')
    idx = np.linspace(0, len(im512) - 1, a.n).astype(int)

    def guides(adapter=None):
        h = None
        if adapter is not None:
            def pre(m, args):
                x = args[0]
                return (adapter(x.float()).to(x.dtype),) + args[1:]
            h = sem.proj_dec.register_forward_pre_hook(pre)
        out = []
        try:
            for s in range(0, len(idx), 16):
                i = idx[s:s + 16]
                x = torch.from_numpy(
                    np.asarray(im512[i], np.float32) / 255.)[:, None].to(dev)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    o = mir({'bscan': x})
                lg = (o['semseg'] if isinstance(o, dict) else o).float()
                if lg.shape[-1] != RES:
                    lg = F.interpolate(lg, size=(RES, RES), mode='bilinear',
                                       align_corners=False)
                lg[:, VOID] = float('-inf')
                pred = lg.argmax(1)
                m = torch.isin(pred, torch.tensor([1, 2], device=dev)).float()
                g = F.adaptive_avg_pool2d(m[:, None], (GRID, GRID))[:, 0] >= OCC
                out.append(g.cpu().numpy())
        finally:
            if h is not None:
                h.remove()
        return np.concatenate(out)

    def jac(x, y):
        i = (x & y).reshape(len(x), -1).sum(1)
        u = np.maximum((x | y).reshape(len(x), -1).sum(1), 1)
        return float(np.mean(i / u))

    names = [s.strip() for s in a.adapters.split(',')]
    G = {'frozen': guides(None)}
    for nm in names:
        f = SRC / ('adapter_%s.pt' % nm)
        ck = torch.load(f, map_location=dev)
        m = Adapter(ck['cfg']['ch'], ck['cfg']['depth'], ck['cfg']['width'],
                    ck['cfg']['alpha']).to(dev)
        m.load_state_dict(ck['state_dict']); m.eval()
        G[nm.split('_')[0]] = guides(m)
    keys = list(G)

    print('guide Jaccard over %d stratified FairVision slices\n' % a.n)
    print('%-10s' % '' + ''.join('%10s' % k for k in keys))
    for x in keys:
        print('%-10s' % x + ''.join('%10.4f' % jac(G[x], G[y]) for y in keys))

    teach = [k for k in keys if k != 'frozen']
    print('\nconsecutive-refresh gain (what a 66-min rebuild buys):')
    for x, y in zip(teach, teach[1:]):
        print('   %-6s -> %-6s   guide Jaccard %.4f   cells changed %.2f%%'
              % (x, y, jac(G[x], G[y]),
                 100 * float(np.mean(G[x] != G[y]))))
    print('\n   %-6s -> %-6s   guide Jaccard %.4f   (widest teacher gap)'
          % (teach[0], teach[-1], jac(G[teach[0]], G[teach[-1]])))
    print('   %-6s vs frozen  guide Jaccard %.4f   (what the adapter buys at all)'
          % (teach[-1], jac(G[teach[-1]], G['frozen'])))

    out = {'n_slices': int(a.n),
           'jaccard': {x: {y: jac(G[x], G[y]) for y in keys} for x in keys}}
    (SRC / 'cadence_guide_jaccard.json').write_text(json.dumps(out, indent=2))
    print('\nwrote', SRC / 'cadence_guide_jaccard.json')


if __name__ == '__main__':
    main()
