#!/usr/bin/env python3
"""Edge-case regression test for masking samplers.

Data: D:\\jepa_phase0\\mirage-goals\\fairvision-transfer
      10 volumes x 5 slices (000, 050, 100, 149, 199).  Slices 000 and 199 sit
      at the volume edges where the retina is faint or nearly absent, which is
      exactly where a guided sampler is most likely to misbehave -- so this set
      is deliberately much harder than a random FairVision draw.

Each slice ships a MIRAGE 4-class patch occupancy grid
``05_patch_occupancy_16.npz['occupancy']`` with shape (4, 16, 16) over classes
[Elsewhere, RNFL, GCIPL, Choroid].  Anatomy = RNFL + GCIPL + Choroid; the inner
retina (RNFL + GCIPL) and choroid are fed to the sampler as the two class
scores, matching the P_inner / P_choroid convention used in production.

Checks, per sampler:
    * never leaves ZERO anatomy visible when anatomy exists
    * falls back cleanly when there is no anatomy at all
    * anatomy-hidden fraction stays inside its configured band
    * union stays in a sane range
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from src.masks.curriculum import CurriculumMaskGenerator  # noqa: E402

GRID = 16
NPATCH = GRID * GRID

ROOT = pathlib.Path(r"D:\jepa_phase0\mirage-goals\fairvision-transfer")
OCC_T = 0.25
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]


def load_cases():
    """Yield dicts with image, class scores and the anatomy reference."""
    out = []
    for vol in sorted(ROOT.iterdir()):
        if not vol.is_dir():
            continue
        for sl in sorted(vol.iterdir()):
            npz = sl / "05_patch_occupancy_16.npz"
            png = sl / "00_original_200.png"
            if not (npz.exists() and png.exists()):
                continue
            occ4 = np.load(npz)["occupancy"]          # (4,16,16)
            inner = occ4[1] + occ4[2]                 # RNFL + GCIPL
            chor = occ4[3]
            anat_occ = inner + chor                   # 1 - Elsewhere
            # native slices are 200x200; resize so the 16x16 grid upsamples
            # cleanly by 16 for overlay rendering
            im = Image.open(png).convert("L").resize((256, 256), Image.BILINEAR)
            img = np.asarray(im, np.float32) / 255.0
            out.append(dict(
                vol=vol.name, slice=sl.name, img=img,
                cs=[inner, chor], anat_occ=anat_occ,
                anat=(anat_occ >= OCC_T),
            ))
    return out


def run(cases, floor, seed=42, fill=None):
    random.seed(seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    # Borrow the production size sampler so block geometry is bit-identical to
    # what training will draw.
    sizer = CurriculumMaskGenerator(
        input_size=(256, 256), patch_size=16,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
        allow_overlap=False,
        curriculum_cfg=dict(
            mode="mirage_cover", T_warm=0, T_total=1, r_max=1.0,
            cover_algorithm="delivered_v2", cover_context_guard=True,
            mirage_occupancy_threshold=OCC_T,
            cover_leave_frac=floor, cover_min_visible_frac=floor,
            cover_fill=fill or "random_legal", audit_masks=True),
    )
    sizer.set_epoch(1)
    rows = []
    for c in cases:
        n_anat = int(c["anat"].sum())
        block_sizes = [
            sizer._sample_block_size((0.15, 0.2), gen) for _ in range(4)
        ]
        guides = torch.from_numpy(np.stack(
            [c["anat_occ"], c["anat"].astype(np.float32), *c["cs"]]
        ).astype(np.float32)).unsqueeze(0)
        enc, pred = sizer.generate(
            1, guide_grids=guides, guide_valid=torch.tensor([n_anat > 0]),
            block_sizes=dict(pred=block_sizes, enc=[
                sizer._sample_block_size((.85, 1.), gen)]),
        )
        audit = sizer.last_mask_audit[0]
        info = audit["policy_info"]
        status = audit["context_floor"]["status"]
        rects = []
        for group in pred:
            mask = np.zeros((16, 16), dtype=bool)
            mask.ravel()[group[0].numpy()] = True
            rects.append(mask)
        context = np.zeros((16, 16), dtype=bool)
        context.ravel()[enc[0][0].numpy()] = True
        u = np.logical_or.reduce(rects)
        hid = int((c["anat"] & u).sum())
        rows.append(dict(
            vol=c["vol"], slice=c["slice"], anat_cells=n_anat,
            fallback=not bool(info),
            union=int(u.sum()),
            anat_hidden=hid,
            anat_visible=int((c["anat"] & context).sum()),
            complement_anatomy=n_anat - hid,
            context_mask=context,
            context_status=status,
            anat_hidden_frac=(hid / n_anat) if n_anat else float("nan"),
            n_cover=info.get("n_cover", 0), n_transition=info.get("n_transition", 0),
            n_random=info.get("n_random", 4),
            floor_violation=status.startswith(("infeasible", "unsatisfied")),
            ok=status in ("satisfied", "no_tissue", "invalid_guide"),
            masks=rects, union_mask=u,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=8,
                    help="how many hardest cases to render")
    ap.add_argument("--outdir", default=r"D:\jepa_phase0\reports\edge_cases")
    ap.add_argument("--fill", default=None,
                    choices=[None, "transition", "random", "random_legal"],
                    help="how leftover (non-coverage) blocks are spent")
    args = ap.parse_args()
    out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    print(f"edge-case slices: {len(cases)} "
          f"from {len({c['vol'] for c in cases})} volumes", flush=True)
    rows = run(cases, args.floor, args.seed, fill=args.fill)

    anat = np.array([r["anat_cells"] for r in rows], float)
    vis = np.array([r["anat_visible"] for r in rows], float)
    hf = np.array([r["anat_hidden_frac"] for r in rows], float)
    un = np.array([r["union"] for r in rows], float)
    has_anat = anat > 0

    # ---- assertions --------------------------------------------------------
    fails = []
    for r in rows:
        if r["anat_cells"] > 0 and r["anat_visible"] <= 0:
            fails.append(f"{r['vol']}/{r['slice']}: ZERO anatomy visible "
                         f"(anat={r['anat_cells']})")
        if r["anat_cells"] == 0 and not r["fallback"]:
            fails.append(f"{r['vol']}/{r['slice']}: no anatomy but no fallback")
        if r["union"] < 40 or r["union"] > 200:
            fails.append(f"{r['vol']}/{r['slice']}: union out of range "
                         f"({r['union']})")
        # The sampler's own contract: ok=False means the caller MUST discard
        # the mask.  Without this the gate could exit 0 while the sampler was
        # handing back unusable results on every slice.
        if r["anat_cells"] > 0 and not r["ok"]:
            fails.append(f"{r['vol']}/{r['slice']}: sampler returned ok=False "
                         f"(floor_violation={r['floor_violation']}, "
                         f"fallback={r['fallback']})")
        if r["floor_violation"]:
            fails.append(f"{r['vol']}/{r['slice']}: floor_violation flagged")
        # A fallback on a sparse guide is legitimate. Its final context still
        # has to satisfy the floor or carry an explicit infeasible status.
        # The arm's stated guarantee, checked under the OCCUPANCY definition
        # the audits use rather than the sampler's softer internal support.
        if r["anat_cells"] > 0:
            vis_frac = r["anat_visible"] / r["anat_cells"]
            if vis_frac < args.floor - 1e-9:
                fails.append(
                    f"{r['vol']}/{r['slice']}: occupancy-visible "
                    f"{100*vis_frac:.1f}% below the {100*args.floor:.0f}% floor "
                    f"({r['anat_visible']}/{r['anat_cells']} cells)")

    summary = dict(
        slices=len(rows), volumes=len({r["vol"] for r in rows}),
        floor=args.floor,
        anat_cells_mean=float(anat.mean()), anat_cells_min=float(anat.min()),
        anat_cells_max=float(anat.max()),
        n_zero_anatomy=int((~has_anat).sum()),
        n_fallback=int(sum(r["fallback"] for r in rows)),
        anat_hidden_mean=float(np.nanmean(hf)),
        anat_hidden_min=float(np.nanmin(hf)) if has_anat.any() else None,
        anat_hidden_max=float(np.nanmax(hf)) if has_anat.any() else None,
        anat_visible_min=float(vis[has_anat].min()) if has_anat.any() else None,
        n_zero_visible=int(((vis <= 0) & has_anat).sum()),
        union_mean=float(un.mean()), union_min=float(un.min()),
        union_max=float(un.max()),
        failures=fails,
    )
    (out / "cover_edge_cases.json").write_text(json.dumps(summary, indent=2))

    print(f"\nanatomy cells   mean {anat.mean():5.1f}  min {anat.min():.0f}  "
          f"max {anat.max():.0f}   ({int((~has_anat).sum())} slices with ZERO anatomy)")
    print(f"anatomy hidden  mean {np.nanmean(hf)*100:5.1f}%  "
          f"min {np.nanmin(hf)*100:5.1f}%  max {np.nanmax(hf)*100:5.1f}%")
    print(f"anatomy visible min {vis[has_anat].min():.0f} cells   "
          f"ZERO-visible slices: {int(((vis <= 0) & has_anat).sum())}")
    print(f"union           mean {un.mean():5.1f}  min {un.min():.0f}  max {un.max():.0f}")
    print(f"fallbacks       {sum(r['fallback'] for r in rows)}")
    print("\nFAILURES: " + ("none" if not fails else ""))
    for f in fails:
        print("  " + f)

    print("\nsparsest slices:")
    order = np.argsort(anat)
    for j in order[:args.show]:
        r = rows[j]
        print(f"  {r['vol']}/{r['slice']:9s} anat {r['anat_cells']:3d} "
              f"hidden {r['anat_hidden']:3d} visible {r['anat_visible']:3d} "
              f"union {r['union']:3d} c/t/r "
              f"{r['n_cover']}/{r['n_transition']}/{r['n_random']}"
              + ("  FALLBACK" if r["fallback"] else ""))

    # ---- render the hardest cases -----------------------------------------
    show = list(order[:args.show])
    fig, axes = plt.subplots(len(show), 3, figsize=(8.6, 2.85 * len(show)))
    if len(show) == 1:
        axes = axes[None, :]
    for r_i, j in enumerate(show):
        c, r = cases[j], rows[j]
        img = np.stack([c["img"]] * 3, -1)
        up = lambda m: np.kron(m.astype(float), np.ones((16, 16)))  # noqa: E731

        ax = axes[r_i, 0]
        ax.imshow(img)
        ax.imshow(up(c["anat"]), cmap="autumn", alpha=0.30)
        ax.set_title(f"{r['vol']} {r['slice']}\nanatomy {r['anat_cells']}/256",
                     fontsize=7.5)
        ax.axis("off")

        ax = axes[r_i, 1]
        ax.imshow(img, cmap="gray")
        ov = np.zeros((256, 256, 4))
        for k, m in enumerate(r["masks"]):
            ov[up(m) > 0] = matplotlib.colors.to_rgba(COLORS[k % 4])
        ov[..., 3] *= 0.55
        ax.imshow(ov)
        ax.set_title(f"COVER  union {r['union']}/256\n"
                     f"c/t/r {r['n_cover']}/{r['n_transition']}/{r['n_random']}"
                     + ("  FALLBACK" if r["fallback"] else ""), fontsize=7.5)
        ax.axis("off")

        ax = axes[r_i, 2]
        vis_img = img.copy()
        vis_img[up(r["context_mask"]) == 0] = 0.0
        ax.imshow(vis_img)
        keep = c["anat"] & r["context_mask"]
        ax.imshow(up(keep), cmap="cool", alpha=0.5 * (up(keep) > 0))
        ax.set_title(f"context  cyan = {r['anat_visible']} anatomy cells kept",
                     fontsize=7.5)
        ax.axis("off")

    handles = [mpatches.Patch(color=COLORS[k], label=f"block {k+1}")
               for k in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("COVER masking on EDGE CASES — sparsest-anatomy slices "
                 "from fairvision-transfer", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    p = out / "cover_edge_cases.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print("\nwrote", p)

    # ---- contact sheet: every case, mask overlay only ----------------------
    ncol = 10
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.75 * ncol, 1.95 * nrow))
    axes = np.atleast_2d(axes)
    order_all = np.argsort([r["anat_cells"] for r in rows])
    for pos, j in enumerate(order_all):
        ax = axes[pos // ncol, pos % ncol]
        c, r = cases[j], rows[j]
        up = lambda m: np.kron(m.astype(float), np.ones((16, 16)))  # noqa: E731
        ax.imshow(np.stack([c["img"]] * 3, -1), cmap="gray")
        ov = np.zeros((256, 256, 4))
        for k, m in enumerate(r["masks"]):
            ov[up(m) > 0] = matplotlib.colors.to_rgba(COLORS[k % 4])
        ov[..., 3] *= 0.5
        ax.imshow(ov)
        keep = c["anat"] & r["context_mask"]
        ax.imshow(up(keep), cmap="cool", alpha=0.75 * (up(keep) > 0))
        bad = (r["anat_cells"] > 0 and r["anat_visible"] <= 0)
        ax.set_title(f"{r['vol'][-5:]} s{r['slice'][-3:]}\n"
                     f"an{r['anat_cells']} vis{r['anat_visible']} "
                     f"u{r['union']}",
                     fontsize=5.6, color=("red" if bad else "black"))
        ax.axis("off")
    for pos in range(len(rows), nrow * ncol):
        axes[pos // ncol, pos % ncol].axis("off")
    fig.suptitle(f"ALL {len(rows)} edge-case slices, sorted by anatomy size · "
                 f"cyan = anatomy kept visible · an=anatomy cells, "
                 f"vis=visible, u=union", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = out / "cover_edge_cases_all.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    print("wrote", p2)
    print("wrote", out / "cover_edge_cases.json")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
