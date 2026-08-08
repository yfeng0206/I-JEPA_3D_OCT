"""Anatomy-budgeted, topology-preserving mask construction: choose K from data.

Replaces the fixed K=105 budget (inherited from rectangular ImageNet I-JEPA) with
a budget defined relative to how much anatomy each slice actually contains:

    a_i = P_inner(i) + P_choroid(i)      soft anatomy occupancy on the 16x16 grid
    A   = sum_i a_i                      total anatomy mass in this slice
    select target cells until   (anatomy hidden by union) / A  <=  rho

so a slice with a large retina gets a larger target and a slice with a small
retina gets a smaller one, instead of every slice paying the same image-area
budget regardless of how much of it is background.

Three construction methods are compared:

  topk      global top-K by score.  Connectivity is accidental -- it holds today
            only because MIRAGE anatomy happens to be smooth.
  grow      score-guided region growing from a seed, adding the best 8-connected
            boundary neighbour each step.  Connectivity is STRUCTURAL.
  grow4     four overlapping connected regions, one per horizontal band, so
            masks_pred keeps FOUR tensors.  I-JEPA's own ablation reports
            1/2/3/4 targets -> 9.0/22.0/48.5/54.2 on 1% ImageNet, so collapsing
            to a single target set is a design change, not plumbing.

Topology metrics include HOLE COUNT, which components/largest-fraction cannot
detect: a mask can be 100% one component and still be a ring around a cavity.

Stage 1 (MIRAGE venv, GPU): --dump      Stage 2 (repo venv): --analyze-from
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

TRAIN = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
CK_BASE = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3-base-512\MergedV3'
           r'\MIRAGE-Base_frozen_convnext_CEGDice-ignore\checkpoint-best.pth')
CK_LARGE = (r'D:\jepa_phase0\mirage-goals\outputs\mergedv3\MergedV3'
            r'\MIRAGE-Large_frozen_convnext_CEGDice-ignore\checkpoint-34.pth')
GRID, ANATOMY = 16, (1, 2)
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ----------------------------------------------------------------- dump ----
def dump(out_path, n_slices, arm, crop):
    import os
    import cv2
    import torch
    import torch.nn.functional as F
    from base512_vs_large1024_guide import build
    from fairvision_model_compare import MIRAGE_WS

    os.chdir(MIRAGE_WS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size, res, ck = ('base', 512, CK_BASE) if arm == 'base' else ('large', 1024, CK_LARGE)
    model, interp = build(size, res, ck, device)
    assert not interp, 'unexpected pos_emb interpolation'
    grab = {}
    model.output_adapters['semseg'].final_layer.register_forward_hook(
        lambda m, i, o: grab.update(L0=o.detach()))

    vols = sorted(p.stem for p in TRAIN.glob('data_*.npz'))
    rng = np.random.default_rng(0)
    grids, names = [], []
    per_vol = max(1, n_slices // max(len(vols), 1))
    for vi, vol_id in enumerate(vols):
        if len(grids) >= n_slices:
            break
        with np.load(TRAIN / ('%s.npz' % vol_id), allow_pickle=True) as z:
            vol = z['oct_bscans']
        for d in rng.choice(len(vol), size=min(per_vol, len(vol)), replace=False):
            raw = np.asarray(vol[int(d)], dtype=np.float32)
            lo, hi = raw.min(), raw.max()
            unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
            if crop:
                # same RandomResizedCrop distribution the JEPA loader uses, so the
                # measured budget reflects what training will actually see
                from PIL import Image
                import torchvision.transforms as T
                import torchvision.transforms.functional as TF
                pil = Image.fromarray((unit * 255).astype(np.uint8))
                i, j, h, w = T.RandomResizedCrop.get_params(
                    pil, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.))
                unit = np.asarray(TF.resized_crop(
                    pil, i, j, h, w, [res, res],
                    T.InterpolationMode.BICUBIC), dtype=np.float32) / 255.0
            else:
                unit = cv2.resize(unit, (res, res), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(unit)[None, None].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                model({'bscan': x})
            M = grab['L0'].softmax(dim=1)[:, ANATOMY].sum(dim=1)
            grids.append(F.adaptive_avg_pool2d(M[:, None], (GRID, GRID))[0, 0].cpu().numpy())
            names.append('%s:%d' % (vol_id, int(d)))
            if len(grids) >= n_slices:
                break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, grids=np.stack(grids), names=np.array(names),
                        arm=arm, crop=bool(crop))
    print('wrote %s  grids %s  arm=%s crop=%s'
          % (out_path, np.stack(grids).shape, arm, crop))


# ------------------------------------------------------- constructors ----
def sel_topk(a, budget):
    """Global top-K by score until the hidden-anatomy budget is met."""
    order = np.argsort(a.ravel())[::-1]
    cum = np.cumsum(a.ravel()[order])
    k = int(np.searchsorted(cum, budget) + 1)
    m = np.zeros(a.size, bool)
    m[order[:min(k, a.size)]] = True
    return m.reshape(a.shape)


def sel_grow(a, budget, seed_rc=None, forbid=None):
    """Score-guided region growing: connectivity holds by construction."""
    h, w = a.shape
    m = np.zeros((h, w), bool)
    cand = a.copy()
    if forbid is not None:
        cand = np.where(forbid, -np.inf, cand)
    if seed_rc is None:
        r, c = np.unravel_index(np.argmax(cand), cand.shape)
    else:
        r, c = seed_rc
    if not np.isfinite(cand[r, c]):
        return m
    m[r, c] = True
    got = a[r, c]
    frontier = {}
    while got < budget:
        for (rr, cc) in list(np.argwhere(m)):
            for dr, dc in NB8:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not m[nr, nc]:
                    if forbid is None or not forbid[nr, nc]:
                        frontier[(nr, nc)] = a[nr, nc]
        if not frontier:
            break
        best = max(frontier, key=frontier.get)
        del frontier[best]
        m[best] = True
        got += a[best]
    return m


def sel_grow4(a, budget, n=4, claim_penalty=0.50):
    """Four overlapping connected regions, one seeded per horizontal band.

    Keeps masks_pred = [T1, T2, T3, T4] so the predictor is used exactly as
    I-JEPA intends.  The UNION is what must respect the anatomy budget, so the
    regions are grown round-robin until the UNION reaches it -- dividing the
    budget by n instead would under-cover, because I-JEPA permits targets to
    overlap and overlapping cells would then be counted more than once.

    ``claim_penalty`` down-weights a candidate cell once another region already
    holds it.  Without it every region grows into the same highest-scoring
    anatomy and the four targets become near-duplicates (measured: 72.5%
    pairwise overlap, union/sum 0.377), which collapses the effective target
    count towards one -- and I-JEPA reports 1 target -> 9.0 versus 4 -> 54.2 on
    1% ImageNet.  Calibrated against the repo's own rectangular collator
    (union/sum 0.706, pairwise overlap 23.9% of a block); penalty 0.50 gives
    0.762 and 19.6%, the closest match on a 120-slice sweep.
    """
    h, w = a.shape
    edges = np.linspace(0, w, n + 1).astype(int)
    parts, frontiers = [], []
    for i in range(n):
        band = np.zeros_like(a, bool)
        band[:, edges[i]:edges[i + 1]] = True
        cand = np.where(band, a, -np.inf)
        m = np.zeros_like(a, bool)
        if np.isfinite(cand).any() and cand.max() > 0:
            r, c = np.unravel_index(np.argmax(cand), cand.shape)
            m[r, c] = True
        parts.append(m)
        frontiers.append({})

    claims = np.zeros_like(a, dtype=np.int16)
    for p in parts:
        claims += p.astype(np.int16)

    def union():
        u = np.zeros_like(a, bool)
        for p in parts:
            u |= p
        return u

    stalled = [False] * n
    while (a * union()).sum() < budget and not all(stalled):
        for i in range(n):
            if (a * union()).sum() >= budget:
                break
            if stalled[i] or not parts[i].any():
                stalled[i] = True
                continue
            for (rr, cc) in np.argwhere(parts[i]):
                for dr, dc in NB8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not parts[i][nr, nc]:
                        frontiers[i][(nr, nc)] = None      # rescored below
            if not frontiers[i]:
                stalled[i] = True
                continue
            # rescore every candidate: anatomy value, discounted by how many
            # OTHER regions already claim that cell
            best, best_v = None, -np.inf
            for (nr, nc) in frontiers[i]:
                held = claims[nr, nc] - int(parts[i][nr, nc])
                v = a[nr, nc] * (claim_penalty ** held)
                if v > best_v:
                    best, best_v = (nr, nc), v
            del frontiers[i][best]
            parts[i][best] = True
            claims[best] += 1
    return parts


# ------------------------------------------------------------ metrics ----
def topology(m):
    from scipy import ndimage
    if m.sum() == 0:
        return dict(components=0, largest_frac=0.0, holes=0, hole_area=0.0,
                    perim_over_area=0.0, n_cells=0)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    # holes = connected background components fully enclosed by the mask
    filled = ndimage.binary_fill_holes(m)
    hole_mask = filled & ~m
    _, n_holes = ndimage.label(hole_mask, structure=np.ones((3, 3)))
    perim = int((m != np.roll(m, 1, 0)).sum() + (m != np.roll(m, 1, 1)).sum())
    return dict(components=int(n), largest_frac=float(sizes.max() / m.sum()),
                holes=int(n_holes), hole_area=float(hole_mask.sum()),
                perim_over_area=float(perim / m.sum()), n_cells=int(m.sum()))


def analyze(npz_path, out_dir, rhos):
    z = np.load(npz_path, allow_pickle=True)
    G = z['grids']
    n = len(G)
    rep = {'n_slices': n, 'arm': str(z['arm']), 'crop': bool(z['crop']),
           'anatomy_mass': {'mean': float(G.sum(axis=(1, 2)).mean()),
                            'p10': float(np.percentile(G.sum(axis=(1, 2)), 10)),
                            'p90': float(np.percentile(G.sum(axis=(1, 2)), 90))},
           'rho_sweep': {}}

    for rho in rhos:
        res = {k: [] for k in ('topk', 'grow', 'grow4')}
        for i in range(n):
            a = G[i]
            A = a.sum()
            if A <= 1e-6:
                continue
            budget = rho * A
            masks = {'topk': sel_topk(a, budget), 'grow': sel_grow(a, budget)}
            parts = sel_grow4(a, budget)
            union = np.zeros_like(a, bool)
            for p in parts:
                union |= p
            masks['grow4'] = union
            for k, m in masks.items():
                t = topology(m)
                t['visible'] = float(1 - (a * m).sum() / A)
                t['purity'] = float((a * m).sum() / max(m.sum(), 1))
                if k == 'grow4':
                    t['part_cells'] = [int(p.sum()) for p in parts]
                    t['min_part_cells'] = int(min(p.sum() for p in parts))
                    t['mean_part_cells'] = float(np.mean([p.sum() for p in parts]))
                    # per-part connectivity: each target must itself be coherent
                    t['parts_connected'] = float(np.mean(
                        [topology(p)['components'] == 1 for p in parts if p.sum()]))
                    ov = sum((parts[x] & parts[y]).sum()
                             for x in range(4) for y in range(x + 1, 4))
                    t['pairwise_overlap_cells'] = int(ov)
                res[k].append(t)
        agg = {}
        for k, lst in res.items():
            if not lst:
                continue
            d = {kk: float(np.mean([x[kk] for x in lst]))
                 for kk in ('n_cells', 'visible', 'purity', 'components',
                            'largest_frac', 'holes', 'hole_area', 'perim_over_area')}
            d['visible_p05'] = float(np.percentile([x['visible'] for x in lst], 5))
            d['frac_visible_ge_25'] = float(np.mean([x['visible'] >= 0.25 for x in lst]))
            d['frac_connected'] = float(np.mean([x['components'] == 1 for x in lst]))
            d['frac_holefree'] = float(np.mean([x['holes'] == 0 for x in lst]))
            d['n_cells_p10'] = float(np.percentile([x['n_cells'] for x in lst], 10))
            d['n_cells_p90'] = float(np.percentile([x['n_cells'] for x in lst], 90))
            if k == 'grow4':
                d['min_part_cells_mean'] = float(np.mean([x['min_part_cells'] for x in lst]))
                d['min_part_cells_p05'] = float(np.percentile(
                    [x['min_part_cells'] for x in lst], 5))
                d['mean_part_cells'] = float(np.mean([x['mean_part_cells'] for x in lst]))
                d['parts_connected'] = float(np.mean([x['parts_connected'] for x in lst]))
                d['pairwise_overlap_cells'] = float(np.mean(
                    [x['pairwise_overlap_cells'] for x in lst]))
            agg[k] = d
        rep['rho_sweep'][str(rho)] = agg

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'anatomy_budget_sweep.json').write_text(json.dumps(rep, indent=2))

    m = rep['anatomy_mass']
    print('slices %d   arm %s   crop %s' % (n, rep['arm'], rep['crop']))
    print('anatomy mass: mean %.1f  p10 %.1f  p90 %.1f  (of 256 cells)'
          % (m['mean'], m['p10'], m['p90']))
    print('\n%5s %7s %7s %7s %7s %6s %7s %6s %7s %7s %7s'
          % ('rho', 'method', 'cells', 'visible', 'vis_p05', '>=25%', 'purity',
             'comps', 'conn%', 'holes', 'holefr%'))
    for rho in rhos:
        for k in ('topk', 'grow', 'grow4'):
            d = rep['rho_sweep'][str(rho)].get(k)
            if not d:
                continue
            print('%5.2f %7s %7.1f %7.4f %7.4f %6.2f %7.4f %6.2f %7.3f %7.2f %7.3f'
                  % (rho, k, d['n_cells'], d['visible'], d['visible_p05'],
                     d['frac_visible_ge_25'], d['purity'], d['components'],
                     d['frac_connected'], d['holes'], d['frac_holefree']))
    print('\ngrow4 target-size detail (predictor gets 4 tensors):')
    for rho in rhos:
        d = rep['rho_sweep'][str(rho)].get('grow4')
        if d:
            print('  rho %.2f  union %.1f  mean part %.1f  min part %.1f (p05 %.1f)'
                  '  overlap %.1f  parts connected %.3f'
                  % (rho, d['n_cells'], d['mean_part_cells'], d['min_part_cells_mean'],
                     d['min_part_cells_p05'], d['pairwise_overlap_cells'],
                     d['parts_connected']))
    print('\n  reference: I-JEPA target blocks are scale (0.15,0.2) of 256 = 38-51 cells each,')
    print('             4 of them, union ~102-112 cells.')
    print('\nwrote %s' % (out_dir / 'anatomy_budget_sweep.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=pathlib.Path)
    ap.add_argument('--analyze-from', type=pathlib.Path)
    ap.add_argument('--n-slices', type=int, default=400)
    ap.add_argument('--arm', default='base', choices=['base', 'large'])
    ap.add_argument('--crop', action='store_true', default=True)
    ap.add_argument('--no-crop', dest='crop', action='store_false')
    ap.add_argument('--rhos', type=float, nargs='+',
                    default=[0.50, 0.60, 0.70, 0.75])
    ap.add_argument('--out', type=pathlib.Path,
                    default=REPO / 'results/masking/anatomy_budget')
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump, a.n_slices, a.arm, a.crop)
    if a.analyze_from:
        return analyze(a.analyze_from, a.out, a.rhos)
    raise SystemExit('need --dump or --analyze-from')


if __name__ == '__main__':
    main()
