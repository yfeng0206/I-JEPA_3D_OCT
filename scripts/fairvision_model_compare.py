"""Head-to-head FairVision comparison of the GOALS-only baseline segmenter and
the merged-data (GOALS + Duke DME + AROI) segmenter.

FairVision has no layer ground truth, so this measures UNSUPERVISED anatomical
plausibility.  Both models are scored on the SAME slices with the SAME rule,
which is the only way the numbers mean anything.

Fair-comparison rules applied here
----------------------------------
* The baseline emits 4 classes (Elsewhere/RNFL/GCIPL/Choroid); the merged model
  emits 3 usable classes (Elsewhere/InnerRetina/Choroid) plus an ignore index.
  Both are collapsed to the SAME two regions -- InnerRetina and Choroid -- by
  fusing the baseline's RNFL+GCIPL.  Topology is then the single ordering
  constraint "InnerRetina sits above Choroid", identical for both arms.
* Because that rule is more lenient than the 3-class ordering used previously,
  the baseline is RE-MEASURED here rather than compared against the older
  published 0.0154 figure.
* Identical volumes, identical depths, identical preprocessing.

Metrics (lower is better unless noted)
* topology violation  - fraction of evaluable columns with impossible order
* runs per column     - vertical fragments per occupied column, 1.0 = unbroken
* union area          - fraction of the frame inside the envelope (descriptive,
                        NOT better-when-lower; it is reported so a change in
                        plausibility can be separated from a change in extent)
* largest-component   - fraction of union area in its biggest connected blob
                        (HIGHER is better; scattered output scores low)
* n_components        - connected components in the union (lower is better)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

MIRAGE_WS = pathlib.Path(r'D:\jepa_phase0\mirage-goals')
DATA = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Training')

BASELINE_CKPT = (MIRAGE_WS / 'outputs' / 'official-vitlarge' / 'GOALS' /
                 'MIRAGE-Large_frozen_convnext_CEGDice' / 'checkpoint-best.pth')

# (inner_classes, choroid_classes, n_logits, ignore_class_or_None)
ARMS = {
    'baseline': ((1, 2), (3,), 4, None),
    'merged': ((1,), (2,), 4, 3),
}


def build_model(num_classes: int, ckpt: pathlib.Path, device: str):
    from argparse import Namespace
    sys.path.insert(0, str(MIRAGE_WS / 'MIRAGE'))
    from fm_seg_config import fm_factory
    from mirage.model import model_factory
    from mirage.output_adapters import ConvNeXtAdapter

    cfg = fm_factory['mirage-large']()
    cfg.build_domain_conf()
    runtime_args = Namespace(grid_sizes={'bscan': [32, 32]},
                             input_size={'bscan': [1024, 1024]})
    input_adapters = {
        'bscan': cfg.domain_conf['bscan']['input_adapter'](
            stride_level=1, patch_size_full=[32, 32],
            image_size=[1024, 1024], learnable_pos_emb=False)
    }
    output_adapters = {
        'semseg': ConvNeXtAdapter(
            num_classes=num_classes, preds_per_patch=16, depth=4,
            interpolate_mode='bilinear', main_tasks=['bscan'], embed_dim=6144,
            patch_size=[32, 32], task='semseg', image_size=[1024, 1024])
    }
    model = model_factory[cfg.model](
        args=runtime_args, input_adapters=input_adapters,
        output_adapters=output_adapters, num_global_tokens=1,
        drop_path_rate=0.1)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = state['model'] if 'model' in state else state
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def topology_violation(inner: np.ndarray, choroid: np.ndarray):
    """Columns where InnerRetina does not sit above Choroid."""
    bad = total = 0
    for c in range(inner.shape[1]):
        ri = np.where(inner[:, c])[0]
        rc = np.where(choroid[:, c])[0]
        if ri.size == 0 or rc.size == 0:
            continue
        total += 1
        if ri.mean() >= rc.mean():
            bad += 1
    return bad, total


def mean_runs(mask: np.ndarray):
    vals = []
    for c in range(mask.shape[1]):
        col = mask[:, c]
        if not col.any():
            continue
        vals.append(int(np.sum(col[1:] & ~col[:-1])) + int(col[0]))
    return vals


def components(mask: np.ndarray):
    """Connected-component count and largest-component share (8-connectivity)."""
    import cv2
    if not mask.any():
        return 0, 0.0
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return int(len(sizes)), float(sizes.max() / sizes.sum())


def col_coverage(mask: np.ndarray) -> float:
    """Fraction of image columns in which this region is predicted at all.

    A retinal layer spans the full B-scan width, so the ideal value is 1.0.
    This is the direct measure of the user's "middle part missing" complaint.
    """
    return float(np.mean(mask.any(axis=0)))


def largest_gap(mask: np.ndarray) -> float:
    """Longest run of consecutive empty columns, as a fraction of width.

    Distinguishes a layer that fades at both edges (benign, cropping) from one
    with a hole punched through the middle (the failure the user reported).
    """
    present = mask.any(axis=0)
    if present.all():
        return 0.0
    best = run = 0
    for v in present:
        run = 0 if v else run + 1
        best = max(best, run)
    return float(best / mask.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged-ckpt', required=True)
    ap.add_argument('--baseline-ckpt', default=str(BASELINE_CKPT))
    ap.add_argument('--volumes', type=int, default=20)
    ap.add_argument('--slices', type=int, default=5)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--save-panels', default=None,
                    help='directory for side-by-side overlay PNGs')
    ap.add_argument('--save-panels-n', type=int, default=12)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    os.chdir(MIRAGE_WS)
    import cv2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    files = sorted(DATA.glob('*.npz'))
    if not files:
        raise SystemExit('no FairVision volumes at %s' % DATA)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(files), size=min(args.volumes, len(files)),
                      replace=False)
    depths = np.linspace(20, 180, num=args.slices).astype(int)

    # Preprocess once so both arms see byte-identical input.
    inputs = []
    for vi in pick:
        with np.load(files[int(vi)], allow_pickle=True) as z:
            vol = z['oct_bscans']
        for d in depths:
            raw = np.asarray(vol[int(d)], dtype=np.float32)
            lo, hi = raw.min(), raw.max()
            unit = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
            inputs.append(cv2.resize(unit, (1024, 1024),
                                     interpolation=cv2.INTER_LINEAR))
    print('slices: %d  (%d volumes x %d depths)'
          % (len(inputs), len(pick), len(depths)))

    # Anatomical anchor. In a macular OCT B-scan the brightest band in each
    # column is the RPE/Bruch's complex. Anatomy is then non-negotiable:
    # RNFL+GCIPL lie ABOVE it and choroid lies BELOW it. That gives an error
    # rate that needs no ground truth, unlike the plausibility proxies.
    rpe_rows = []
    for a in inputs:
        s = cv2.resize(a, (200, 200), interpolation=cv2.INTER_LINEAR)
        prof = cv2.GaussianBlur(s, (1, 9), 0)
        rpe_rows.append(np.argmax(prof, axis=0))

    ckpts = {'baseline': pathlib.Path(args.baseline_ckpt),
             'merged': pathlib.Path(args.merged_ckpt)}
    results, hards = {}, {}

    for arm, (inner_cls, chor_cls, n_logits, ignore_cls) in ARMS.items():
        ckpt = ckpts[arm]
        if not ckpt.exists():
            raise SystemExit('missing checkpoint for %s: %s' % (arm, ckpt))
        print('\n[%s] %s' % (arm, ckpt))
        model = build_model(n_logits, ckpt, device)

        bad = tot = 0
        runs, area, ncomp, big, ig_rate = [], [], [], [], []
        inner_area, chor_area = [], []
        inner_cov, chor_cov, inner_gap = [], [], []
        inner_ncomp, inner_big = [], []
        inner_wrong, chor_wrong = [], []
        per_arm_hard = []
        for si, arr in enumerate(inputs):
            t = torch.from_numpy(arr)[None, None].to(device=device,
                                                     dtype=torch.float32)
            with torch.inference_mode(), torch.autocast(
                    device_type='cuda', dtype=torch.float16,
                    enabled=device == 'cuda'):
                out = model({'bscan': t})
            logits = out['semseg'] if isinstance(out, dict) else out
            logits = logits.float().clone()  # inference_mode tensors are read-only
            # Diagnostic: how often WOULD the void channel have won? Measured
            # before suppression, because after suppression it is 0 by
            # construction and tells us nothing.
            if ignore_cls is not None:
                raw = logits.argmax(1)[0]
                ig_rate.append(float((raw == ignore_cls).float().mean()))
                logits[:, ignore_cls] = float('-inf')
            hard = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            hard = cv2.resize(hard, (200, 200), interpolation=cv2.INTER_NEAREST)
            per_arm_hard.append(hard)

            inner = np.isin(hard, inner_cls)
            chor = np.isin(hard, chor_cls)
            union = inner | chor

            b, tt = topology_violation(inner, chor)
            bad += b
            tot += tt
            runs += mean_runs(union)
            area.append(float(union.mean()))
            nc, lc = components(union)
            ncomp.append(nc)
            big.append(lc)

            # Per-region decomposition. The union can shrink either because the
            # model got more precise or because one region vanished; only the
            # split tells them apart.
            inner_area.append(float(inner.mean()))
            chor_area.append(float(chor.mean()))
            inner_cov.append(col_coverage(inner))
            chor_cov.append(col_coverage(chor))
            inner_gap.append(largest_gap(inner))
            i_nc, i_lc = components(inner)
            inner_ncomp.append(i_nc)
            inner_big.append(i_lc)

            rows_idx = np.arange(hard.shape[0])[:, None]
            below = rows_idx > rpe_rows[si][None, :]
            if inner.any():
                inner_wrong.append(float((inner & below).sum() / inner.sum()))
            if chor.any():
                chor_wrong.append(float((chor & ~below).sum() / chor.sum()))

        del model
        torch.cuda.empty_cache()
        hards[arm] = per_arm_hard
        results[arm] = {
            'topology_violation': (bad / tot) if tot else float('nan'),
            'evaluable_columns': tot,
            'runs_per_column': float(np.mean(runs)) if runs else float('nan'),
            'union_area': float(np.mean(area)),
            'n_components': float(np.mean(ncomp)),
            'largest_component_share': float(np.mean(big)),
            'inner_area': float(np.mean(inner_area)),
            'choroid_area': float(np.mean(chor_area)),
            'inner_col_coverage': float(np.mean(inner_cov)),
            'choroid_col_coverage': float(np.mean(chor_cov)),
            'inner_largest_gap': float(np.mean(inner_gap)),
            'inner_n_components': float(np.mean(inner_ncomp)),
            'inner_largest_share': float(np.mean(inner_big)),
            'inner_below_rpe': float(np.mean(inner_wrong)) if inner_wrong else float('nan'),
            'choroid_above_rpe': float(np.mean(chor_wrong)) if chor_wrong else float('nan'),
        }
        if ig_rate:
            results[arm]['predicted_ignore_rate'] = float(np.mean(ig_rate))

    agree = float(np.mean([
        (np.isin(a, ARMS['baseline'][0]) | np.isin(a, ARMS['baseline'][1])) ==
        (np.isin(b, ARMS['merged'][0]) | np.isin(b, ARMS['merged'][1]))
        for a, b in zip(hards['baseline'], hards['merged'])]))

    print('\n' + '=' * 72)
    print('FairVision transfer, %d slices, unsupervised plausibility'
          % len(inputs))
    print('=' * 72)
    rows = [
        ('topology violation', 'lower', 'topology_violation'),
        ('runs per column', 'lower', 'runs_per_column'),
        ('n components', 'lower', 'n_components'),
        ('largest-comp share', 'HIGHER', 'largest_component_share'),
        ('union area', 'descr.', 'union_area'),
        ('-- inner retina --', '', None),
        ('inner col coverage', 'HIGHER', 'inner_col_coverage'),
        ('inner largest gap', 'lower', 'inner_largest_gap'),
        ('inner n components', 'lower', 'inner_n_components'),
        ('inner largest share', 'HIGHER', 'inner_largest_share'),
        ('inner area', 'descr.', 'inner_area'),
        ('-- anatomy (RPE-anchored error rates) --', '', None),
        ('inner below RPE', 'lower', 'inner_below_rpe'),
        ('choroid above RPE', 'lower', 'choroid_above_rpe'),
        ('-- choroid --', '', None),
        ('choroid col coverage', 'HIGHER', 'choroid_col_coverage'),
        ('choroid area', 'descr.', 'choroid_area'),
    ]
    print('  %-20s %8s %10s %10s %10s'
          % ('metric', 'better', 'baseline', 'merged', 'delta'))
    for label, better, key in rows:
        if key is None:
            print('  %s' % label)
            continue
        b, m = results['baseline'][key], results['merged'][key]
        print('  %-20s %8s %10.4f %10.4f %+10.4f' % (label, better, b, m, m - b))
    print('\n  envelope agreement between the two models: %.4f' % agree)
    if 'predicted_ignore_rate' in results['merged']:
        print('  merged model predicted the ignore class on %.5f of pixels'
              % results['merged']['predicted_ignore_rate'])

    verdict = []
    if results['merged']['topology_violation'] < results['baseline']['topology_violation']:
        verdict.append('topology improved')
    else:
        verdict.append('topology NOT improved')
    if results['merged']['runs_per_column'] < results['baseline']['runs_per_column']:
        verdict.append('fragmentation improved')
    else:
        verdict.append('fragmentation NOT improved')
    if results['merged']['largest_component_share'] > results['baseline']['largest_component_share']:
        verdict.append('connectivity improved')
    else:
        verdict.append('connectivity NOT improved')
    print('\n  SUMMARY: ' + '; '.join(verdict))
    print('\n  Caveat: FairVision has no layer ground truth. These are')
    print('  plausibility proxies, not accuracy. Read them together with the')
    print('  GOALS-test regression guardrail.')

    if args.save_panels:
        import cv2
        pan = pathlib.Path(args.save_panels)
        pan.mkdir(parents=True, exist_ok=True)
        # inner retina = red, choroid = blue, drawn over the greyscale B-scan
        # so the user can judge continuity and placement by eye.
        for i in range(min(args.save_panels_n, len(inputs))):
            small = cv2.resize(inputs[i], (200, 200),
                               interpolation=cv2.INTER_LINEAR)
            grey = (np.clip(small, 0, 1) * 255).astype(np.uint8)
            tiles = [cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)]
            for arm in ('baseline', 'merged'):
                inner_cls, chor_cls = ARMS[arm][0], ARMS[arm][1]
                hard = hards[arm][i]
                ov = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
                ov[np.isin(hard, inner_cls)] = (60, 60, 235)
                ov[np.isin(hard, chor_cls)] = (235, 140, 60)
                tiles.append(ov)
            panel = np.hstack([np.pad(t, ((16, 0), (0, 4), (0, 0)))
                               for t in tiles])
            for j, lab in enumerate(('input', 'baseline(GOALS)', 'merged')):
                cv2.putText(panel, lab, (j * 204 + 4, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
            cv2.imwrite(str(pan / ('panel_%02d.png' % i)), panel)
        print('\n  wrote %d panels to %s'
              % (min(args.save_panels_n, len(inputs)), pan))

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {'n_slices': len(inputs), 'volumes': len(pick),
             'depths': depths.tolist(), 'seed': args.seed,
             'envelope_agreement': agree, 'arms': results}, indent=2))
        print('\nwrote', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
