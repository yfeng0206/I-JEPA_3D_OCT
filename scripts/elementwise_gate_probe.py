"""Empirically test the input-level element-wise gate design: x_gated = s * x.

The proposal is to multiply the OCT image by a MIRAGE-derived pixel score s
before it enters I-JEPA, so that the JEPA latent loss backpropagates into
MIRAGE.  This script measures whether that gradient path can actually carry a
masking signal, using the real encoder/predictor/collator from this repo.

Two questions are answered numerically:

  Q1  Where is dL/ds non-zero?
      PatchEmbed is a stride==kernel Conv2d, so pixel (i,j) feeds exactly one
      token, and `apply_masks` gathers the context tokens *before* any
      transformer block.  Prediction: dL/ds is identically zero on every patch
      that is not in the context mask -- i.e. zero exactly where a masking
      policy would need signal.

  Q2  Where does minimizing L_JEPA w.r.t. s converge?
      Two wirings:
        shared  : both context and EMA-target branches see s * x  (the diagram)
        ctx_only: context sees s * x, target sees the raw x
      Prediction: `shared` collapses (loss -> 0, s uninformative) because a
      blank input is trivially predictable; `ctx_only` saturates s -> 1 because
      revealing more always lowers the loss.

Usage:
    python scripts/elementwise_gate_probe.py --steps 300 --out results/masking/gate
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.helper import init_patch_model
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch

GRID = 16
PATCH = 16
CROP = 256


def load_images(batch: int, device: torch.device, npz: Path | None):
    """Real OCT slices when available, otherwise a structured stand-in.

    The Q1 result is exact and content independent; real data only makes the
    Q2 optimisation trace realistic.
    """
    if npz is not None and npz.exists():
        import numpy as np
        arr = np.load(npz)
        key = next((k for k in ('oct', 'slices', 'octs', 'image', 'data') if k in arr),
                   list(arr.keys())[0])
        vol = torch.from_numpy(arr[key]).float()
        if vol.ndim == 4:              # (D,H,W,C) or (D,C,H,W)
            vol = vol[..., 0] if vol.shape[-1] <= 4 else vol[:, 0]
        if vol.ndim == 3:              # (D,H,W) -> take evenly spaced slices
            idx = torch.linspace(0, vol.shape[0] - 1, batch).long()
            vol = vol[idx].unsqueeze(1)
        elif vol.ndim == 2:
            vol = vol.unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1, 1)
        vol = F.interpolate(vol, size=(CROP, CROP), mode='bilinear', align_corners=False)
        vol = (vol - vol.mean()) / (vol.std() + 1e-6)
        return vol.repeat(1, 3, 1, 1).to(device), f'real:{npz.name}:{key}'

    g = torch.Generator().manual_seed(0)
    y = torch.linspace(0, 1, CROP).view(1, 1, CROP, 1)
    band = torch.exp(-((y - 0.45) ** 2) / 0.004) + 0.6 * torch.exp(-((y - 0.72) ** 2) / 0.002)
    x = band.repeat(batch, 3, 1, CROP) + 0.15 * torch.randn(batch, 3, CROP, CROP, generator=g)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x.to(device), 'synthetic-band'


def jepa_loss(encoder, target_encoder, predictor, x_ctx, x_tgt, masks_enc, masks_pred):
    """Exactly the loss in src/train_patch.py: target under no_grad, smooth L1."""
    with torch.no_grad():
        h = target_encoder(x_tgt)
        h = F.layer_norm(h, (h.size(-1),))
        h_pred_full = apply_masks(h, masks_pred)
        h_rep = repeat_interleave_batch(h_pred_full, x_tgt.size(0), repeat=len(masks_enc))
    z = encoder(x_ctx, masks_enc)
    z = predictor(z, masks_enc, masks_pred)
    return F.smooth_l1_loss(z, h_rep)


def load_encoder_weights(encoder, target_encoder, ckpt: Path):
    """Load a trained I-JEPA encoder so the loss landscape is meaningful."""
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    enc_sd = sd.get('encoder', sd.get('model', sd))
    enc_sd = {k.replace('module.', ''): v for k, v in enc_sd.items()}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    tgt_sd = sd.get('target_encoder', enc_sd)
    tgt_sd = {k.replace('module.', ''): v for k, v in tgt_sd.items()}
    target_encoder.load_state_dict(tgt_sd, strict=False)
    return {'missing': len(missing), 'unexpected': len(unexpected),
            'loaded': len(enc_sd) - len(unexpected)}


def make_figure(out_dir, x, gpatch, in_ctx, in_tgt, report, s_maps):
    """Visual proof: where the gradient lands, and what the loss actually wants."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; skipping figure')
        return None

    img = x[0, 0].detach().cpu().numpy()
    g = gpatch[0].detach().cpu().reshape(GRID, GRID).numpy()
    ctx = in_ctx[0].detach().cpu().reshape(GRID, GRID).numpy()
    tgt = in_tgt[0].detach().cpu().reshape(GRID, GRID).numpy()

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))

    ax[0, 0].imshow(img, cmap='gray'); ax[0, 0].set_title('OCT B-scan  x')
    ax[0, 1].imshow(ctx, cmap='Blues', vmin=0, vmax=1)
    ax[0, 1].set_title(f'context mask (visible)\n{int(ctx.sum())} of 256 tokens')
    ax[0, 2].imshow(tgt, cmap='Reds', vmin=0, vmax=1)
    ax[0, 2].set_title(f'target blocks (to predict)\n{int(tgt.sum())} tokens')

    im = ax[1, 0].imshow(g, cmap='inferno')
    ax[1, 0].set_title(r'$|\partial L/\partial s|$ per token'
                       f"\noutside context: {report['q1']['grad_sum_outside_context']:.1e}")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.046)

    for wiring, c in (('shared', 'tab:red'), ('ctx_only', 'tab:blue')):
        cur = report[f'q3_{wiring}']['curve']
        ax[1, 1].plot([p['alpha'] for p in cur], [p['loss'] for p in cur],
                      'o-', color=c, label=f'{wiring}')
    ax[1, 1].set_xlabel(r'uniform gate $\alpha$   ($\alpha$=0 blank, 1 = real image)')
    ax[1, 1].set_ylabel(r'$L_{JEPA}$')
    ax[1, 1].set_title('Q3: what the loss wants\n(minimum = where MIRAGE is pushed)')
    ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

    ax[1, 2].imshow(s_maps['shared'], cmap='viridis', vmin=0, vmax=1)
    r = report['q2_shared']
    ax[1, 2].set_title('learned gate s after 200 steps\n'
                       f"outside ctx: max|s-0.5| = {r['s_max_dev_outside_ctx']:.0e}")

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    ax[1, 1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax[1, 1].set_yticks(ax[1, 1].get_yticks())

    fig.suptitle('Element-wise gate  s $\\odot$ x  into I-JEPA: gradient reach and loss landscape',
                 fontsize=13)
    fig.tight_layout()
    p = out_dir / 'elementwise_gate_probe.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--npz', type=Path, default=None)
    ap.add_argument('--ckpt', type=Path, default=None,
                    help='trained I-JEPA checkpoint (.pth.tar); random init if omitted')
    ap.add_argument('--pred-ckpt-key', default='predictor')
    ap.add_argument('--out', type=Path, default=REPO / 'results' / 'masking' / 'gate')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)

    encoder, predictor = init_patch_model(device, patch_size=PATCH, crop_size=CROP,
                                          model_name='vit_base')
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    ckpt_info = None
    if args.ckpt is not None and args.ckpt.exists():
        ckpt_info = load_encoder_weights(encoder, target_encoder, args.ckpt)
        sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        if args.pred_ckpt_key in sd:
            psd = {k.replace('module.', ''): v for k, v in sd[args.pred_ckpt_key].items()}
            predictor.load_state_dict(psd, strict=False)
        del sd
    encoder.eval(); target_encoder.eval(); predictor.eval()

    x, src = load_images(args.batch, device, args.npz)

    collator = MaskCollator(input_size=(CROP, CROP), patch_size=PATCH)
    dummy = [x[i].cpu() for i in range(args.batch)]
    _, masks_enc, masks_pred = collator(dummy)
    masks_enc = [m.to(device) for m in masks_enc]
    masks_pred = [m.to(device) for m in masks_pred]

    report = {'device': str(device), 'image_source': src, 'batch': args.batch,
              'n_context_tokens': int(masks_enc[0].shape[1]),
              'n_target_tokens': int(masks_pred[0].shape[1])}

    # ---------------- Q1: where is dL/ds non-zero? ----------------
    logits = torch.zeros(args.batch, 1, CROP, CROP, device=device, requires_grad=True)
    s = torch.sigmoid(logits)
    loss = jepa_loss(encoder, target_encoder, predictor, s * x, s * x, masks_enc, masks_pred)
    loss.backward()

    # Per-patch gradient energy on the 16x16 token grid.
    gpatch = logits.grad.abs().squeeze(1)
    gpatch = gpatch.unfold(1, PATCH, PATCH).unfold(2, PATCH, PATCH).sum(dim=(-1, -2))
    gpatch = gpatch.reshape(args.batch, GRID * GRID)

    in_ctx = torch.zeros(args.batch, GRID * GRID, dtype=torch.bool, device=device)
    in_ctx.scatter_(1, masks_enc[0], True)
    in_tgt = torch.zeros(args.batch, GRID * GRID, dtype=torch.bool, device=device)
    for m in masks_pred:
        in_tgt.scatter_(1, m, True)

    report['q1'] = {
        'grad_sum_inside_context': float(gpatch[in_ctx].sum()),
        'grad_sum_outside_context': float(gpatch[~in_ctx].sum()),
        'grad_sum_on_target_patches': float(gpatch[in_tgt].sum()),
        'max_abs_grad_outside_context': float(gpatch[~in_ctx].abs().max()),
        'nonzero_patches_outside_context': int((gpatch[~in_ctx].abs() > 0).sum()),
        'n_patches_outside_context': int((~in_ctx).sum()),
    }

    # ---------------- Q2: where does optimizing s converge? ----------------
    s_maps = {}
    for wiring in ('shared', 'ctx_only'):
        logits = torch.zeros(args.batch, 1, CROP, CROP, device=device, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=args.lr)
        trace = []
        for step in range(args.steps):
            s = torch.sigmoid(logits)
            x_ctx = s * x
            x_tgt = s * x if wiring == 'shared' else x
            loss = jepa_loss(encoder, target_encoder, predictor,
                             x_ctx, x_tgt, masks_enc, masks_pred)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
                with torch.no_grad():
                    sv = torch.sigmoid(logits)
                    trace.append({'step': step, 'loss': float(loss),
                                  's_mean': float(sv.mean()), 's_std': float(sv.std()),
                                  'gated_input_std': float((sv * x).std())})
        # Confirming split: with fixed masks, pixels outside the context mask
        # receive exactly zero gradient, so Adam never moves them off sigmoid(0)=0.5.
        with torch.no_grad():
            sv = torch.sigmoid(logits).squeeze(1)
            spatial_ctx = in_ctx.reshape(-1, GRID, GRID).float()
            spatial_ctx = F.interpolate(spatial_ctx.unsqueeze(1), size=(CROP, CROP),
                                        mode='nearest').squeeze(1).bool()
            s_in, s_out = sv[spatial_ctx], sv[~spatial_ctx]
            s_maps[wiring] = sv[0].cpu().numpy()
        report[f'q2_{wiring}'] = {
            'loss_start': trace[0]['loss'], 'loss_end': trace[-1]['loss'],
            's_mean_end': trace[-1]['s_mean'], 's_std_end': trace[-1]['s_std'],
            's_mean_inside_ctx': float(s_in.mean()), 's_std_inside_ctx': float(s_in.std()),
            's_mean_outside_ctx': float(s_out.mean()), 's_std_outside_ctx': float(s_out.std()),
            's_max_dev_outside_ctx': float((s_out - 0.5).abs().max()),
            'input_std_start': trace[0]['gated_input_std'],
            'input_std_end': trace[-1]['gated_input_std'],
            'trace': trace,
        }

    # ---------------- Q3: the loss landscape in the gate ----------------
    # Even with a perfect gradient path, the question is what L_JEPA *wants*.
    # Sweep a uniform gate alpha over [0,1] and read off the minimiser.
    #   shared   -> both branches see alpha*x   (the proposed diagram)
    #   ctx_only -> only the context branch is gated
    # A minimum at alpha=0 means the objective pays MIRAGE to erase the image.
    for wiring in ('shared', 'ctx_only'):
        curve = []
        with torch.no_grad():
            for a in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]:
                xa = a * x
                l = jepa_loss(encoder, target_encoder, predictor,
                              xa, xa if wiring == 'shared' else x,
                              masks_enc, masks_pred)
                curve.append({'alpha': a, 'loss': float(l)})
        best = min(curve, key=lambda c: c['loss'])
        report[f'q3_{wiring}'] = {
            'curve': curve,
            'argmin_alpha': best['alpha'], 'min_loss': best['loss'],
            'loss_at_alpha_0': curve[0]['loss'], 'loss_at_alpha_1': curve[-1]['loss'],
        }

    out = args.out / 'elementwise_gate_probe.json'
    out.write_text(json.dumps(report, indent=2))
    fig_path = make_figure(args.out, x, gpatch, in_ctx, in_tgt, report, s_maps)

    q1 = report['q1']
    print(f"source={src} device={device} ctx_tokens={report['n_context_tokens']}")
    print(f"checkpoint={args.ckpt}  load={ckpt_info}")
    print('--- Q1: gradient reaching s ---')
    print(f"  sum |dL/ds| inside context mask : {q1['grad_sum_inside_context']:.6e}")
    print(f"  sum |dL/ds| outside context mask: {q1['grad_sum_outside_context']:.6e}")
    print(f"  sum |dL/ds| on target patches   : {q1['grad_sum_on_target_patches']:.6e}")
    print(f"  patches outside ctx with any grad: "
          f"{q1['nonzero_patches_outside_context']} / {q1['n_patches_outside_context']}")
    for wiring in ('shared', 'ctx_only'):
        r = report[f'q2_{wiring}']
        print(f"--- Q2 [{wiring}] learned gate ---")
        print(f"  loss {r['loss_start']:.5f} -> {r['loss_end']:.5f}")
        print(f"  s mean {r['s_mean_end']:.4f}  s std {r['s_std_end']:.4f}")
        print(f"  inside ctx : mean {r['s_mean_inside_ctx']:.4f} std {r['s_std_inside_ctx']:.4f}")
        print(f"  outside ctx: mean {r['s_mean_outside_ctx']:.4f} std "
              f"{r['s_std_outside_ctx']:.2e}  max|s-0.5| {r['s_max_dev_outside_ctx']:.2e}")
        print(f"  gated-input std {r['input_std_start']:.4f} -> {r['input_std_end']:.4f}")
    for wiring in ('shared', 'ctx_only'):
        r = report[f'q3_{wiring}']
        print(f"--- Q3 [{wiring}] loss vs uniform gate alpha ---")
        print('  ' + '  '.join(f"a={c['alpha']:.2f}:{c['loss']:.4f}" for c in r['curve']))
        print(f"  argmin alpha = {r['argmin_alpha']}  "
              f"(L[0]={r['loss_at_alpha_0']:.4f}, L[1]={r['loss_at_alpha_1']:.4f})")
    print(f'wrote {out}')
    if fig_path:
        print(f'wrote {fig_path}')


if __name__ == '__main__':
    main()
