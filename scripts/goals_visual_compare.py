"""Side-by-side GOALS-test figure: B-scan | GT | baseline | merged | error map.

This is the only visual in the workstream backed by real ground truth.
The FairVision panels can only show plausibility; here we can show who is
actually right, per pixel.

Slice selection is deliberately NOT cherry-picked.  Per-image foreground
Dice is computed for both arms, and rows are drawn from the extremes and
the median of the (merged - baseline) margin: the biggest merged win, the
biggest merged loss, and the median case.  A figure that only showed wins
would be worthless for deciding whether to ship the model.

Palettes are imported from score_goals_merged so the pixels drawn here are
the exact pixels that produced the reported Dice numbers.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score_goals_merged import (  # noqa: E402
    CHOROID, ELSEWHERE, GOALS_4CLASS, INNER, MERGED_3CLASS, load_indexed,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# InnerRetina cyan, Choroid amber; chosen to stay distinguishable on a
# greyscale B-scan and for red/green colour-blind readers.
CLASS_RGB = {
    ELSEWHERE: (0, 0, 0),
    INNER: (0, 190, 210),
    CHOROID: (250, 176, 40),
}
SIZE = 1024


def colorize(idx: np.ndarray) -> np.ndarray:
    rgb = np.zeros(idx.shape + (3,), dtype=np.uint8)
    for k, c in CLASS_RGB.items():
        rgb[idx == k] = c
    return rgb


def overlay(gray: np.ndarray, idx: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    lab = colorize(idx).astype(np.float32)
    m = (idx != ELSEWHERE)[..., None]
    return np.clip(base * (1 - alpha * m) + lab * (alpha * m), 0, 255).astype(np.uint8)


def fg_dice(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """Mean Dice over the two foreground classes, ignoring void pixels."""
    scores = []
    for c in (INNER, CHOROID):
        p, g = (pred == c) & valid, (gt == c) & valid
        denom = p.sum() + g.sum()
        if denom == 0:
            continue
        scores.append(2.0 * (p & g).sum() / denom)
    return float(np.mean(scores)) if scores else float('nan')


def error_map(base_p: np.ndarray, mrg_p: np.ndarray, gt: np.ndarray,
              valid: np.ndarray) -> np.ndarray:
    """Red = baseline wrong only, blue = merged wrong only, grey = both wrong."""
    be = (base_p != gt) & valid
    me = (mrg_p != gt) & valid
    rgb = np.zeros(gt.shape + (3,), dtype=np.uint8)
    rgb[be & ~me] = (220, 40, 40)
    rgb[me & ~be] = (40, 90, 230)
    rgb[be & me] = (130, 130, 130)
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--bscan', required=True)
    ap.add_argument('--baseline-preds', required=True)
    ap.add_argument('--merged-preds', required=True)
    ap.add_argument('--merged-label', default='Merged V3 ep35')
    ap.add_argument('--rows', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    gt_dir = pathlib.Path(a.gt)
    bs_dir = pathlib.Path(a.bscan)
    bp_dir = pathlib.Path(a.baseline_preds)
    mp_dir = pathlib.Path(a.merged_preds)

    recs = []
    for gp in sorted(gt_dir.glob('*.png')):
        stem = gp.name.split('__', 1)[-1]
        bp, mp = bp_dir / stem, mp_dir / stem
        if not (bp.exists() and mp.exists()):
            continue
        gt = load_indexed(gp, MERGED_3CLASS, SIZE)
        valid = gt != 255
        if not valid.any():
            continue
        base = load_indexed(bp, GOALS_4CLASS, SIZE)
        mrg = load_indexed(mp, MERGED_3CLASS, SIZE)
        db, dm = fg_dice(base, gt, valid), fg_dice(mrg, gt, valid)
        recs.append(dict(stem=stem, gt_path=gp, base_path=bp, mrg_path=mp,
                         db=db, dm=dm, margin=dm - db))
    if not recs:
        print('no paired images found')
        return 1

    recs.sort(key=lambda r: r['margin'])
    n = len(recs)
    # biggest merged loss, two around the median, biggest merged win
    picks = [recs[0], recs[n // 3], recs[2 * n // 3], recs[-1]][:a.rows]
    tags = ['worst case for merged', 'typical', 'typical', 'best case for merged'][:a.rows]

    print('paired images: %d' % n)
    print('mean fg Dice   baseline %.4f   merged %.4f' %
          (np.nanmean([r['db'] for r in recs]), np.nanmean([r['dm'] for r in recs])))
    wins = sum(1 for r in recs if r['margin'] > 0)
    print('merged beats baseline on %d/%d images' % (wins, n))

    rows = len(picks)
    fig, axes = plt.subplots(rows, 5, figsize=(21, 4.3 * rows))
    if rows == 1:
        axes = axes[None, :]
    titles = ['B-scan', 'Ground truth', 'Baseline (GOALS-only)',
              a.merged_label, 'Errors vs GT']

    for r, (rec, tag) in enumerate(zip(picks, tags)):
        g = np.array(Image.open(bs_dir / rec['gt_path'].name).convert('L'))
        if g.shape != (SIZE, SIZE):
            g = np.array(Image.fromarray(g).resize((SIZE, SIZE), Image.BILINEAR))
        gt = load_indexed(rec['gt_path'], MERGED_3CLASS, SIZE)
        valid = gt != 255
        base = load_indexed(rec['base_path'], GOALS_4CLASS, SIZE)
        mrg = load_indexed(rec['mrg_path'], MERGED_3CLASS, SIZE)

        panels = [g, overlay(g, gt), overlay(g, base), overlay(g, mrg),
                  error_map(base, mrg, gt, valid)]
        for c, (ax, im, t) in enumerate(zip(axes[r], panels, titles)):
            ax.imshow(im, cmap='gray' if c == 0 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(t, fontsize=13, fontweight='bold')
        axes[r, 0].set_ylabel('%s\n%s' % (rec['stem'], tag), fontsize=10)
        axes[r, 2].set_xlabel('fg Dice %.4f' % rec['db'], fontsize=11)
        axes[r, 3].set_xlabel('fg Dice %.4f  (%+.4f)' % (rec['dm'], rec['margin']),
                              fontsize=11,
                              fontweight='bold' if rec['margin'] > 0 else 'normal')

    handles = [
        Patch(color=np.array(CLASS_RGB[INNER]) / 255, label='InnerRetina (RNFL+GCIPL)'),
        Patch(color=np.array(CLASS_RGB[CHOROID]) / 255, label='Choroid'),
        Patch(color=(220 / 255, 40 / 255, 40 / 255), label='baseline wrong only'),
        Patch(color=(40 / 255, 90 / 255, 230 / 255), label='merged wrong only'),
        Patch(color=(130 / 255, 130 / 255, 130 / 255), label='both wrong'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, 0.002))
    fig.suptitle('GOALS test set - ground truth available, so errors are objective',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print('wrote %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
