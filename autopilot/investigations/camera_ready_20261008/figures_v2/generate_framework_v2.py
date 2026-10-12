"""I-JEPA framework and 3-D volume handling figure (camera-ready v2, I13).

Three panels in one float:

(a) Volume handling: a FairVision volume holds 200 B-scans of 200 x 200 px;
    the loaders use 100 evenly spaced ones (np.linspace(0, 199, 100) cast to
    int: 0, 2, ..., 196, 199), each resized to 256 x 256 and cut into a 16 x 16
    grid of 16-px patches (N = 256 tokens).
(b) Pretraining: one B-scan view per sample (random resized crop to 256 x 256);
    the sampler draws (C, T) uniformly (RANDOM) or from a guide A_s(x); the
    context encoder f_theta sees only C; the predictor g_phi receives h_C and
    mask tokens with target positions and predicts every T_m in parallel; the
    EMA target encoder sees the full view, its per-token LayerNorm output is
    selected at T_m under stop-gradient; Smooth-L1 per target group.
(c) Evaluation: the frozen EMA target encoder encodes each of the volume's
    100 B-scans (full view, no crop); patch tokens (256 x 768) are averaged per
    B-scan, the 100 B-scan vectors are averaged per volume, and a LayerNorm +
    linear head gives the glaucoma probability. Patch-max / B-scan-max variants
    are secondary.

Every protocol fact drawn is located in the source (file, line, sha256) and
recorded in the evidence manifest. The masks in (b) are the CENTROID masks of
the strategy figure (exact delivered indices from the staged replay). OCT
pixels: Kulyabin et al., OCTDL Fig. 1 (CC BY 4.0); the volume in (a)/(c) is a
schematic drawing; no FairVision pixels are read or drawn.
"""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle_v2 as st  # noqa: E402

ROOT = st.ROOT
STEM = "fig_oct_jepa_volume_v2"
POSTER_STEM = "fig_oct_jepa_volume_v2_poster"
STAGED_EVIDENCE = st.STAGING / "fig_oct_all_strategies.evidence.json"
STAGED_EVIDENCE_SHA256 = "57b725073338f4b3fa54e6f82b6b772d24d1c3b00d85a75c08273dd2be8e03a7"
EXAMPLE_ARM = "centroid"
USED_FILL, USED_EDGE = "#E3EEF8", "#2B5D8A"

# Protocol facts drawn in the figure, each tied to exact source lines.
FACTS = [
    ("src/datasets/oct_slices.py", ["(200, 200, 200) (uint8)", "0, 199, num=num_slices, dtype=np.int64",
                                    "(self.slice_size, self.slice_size), Image.BILINEAR",
                                    "return len(self.file_paths) * self.num_slices"],
     "FairVision volume (200, 200, 200); 100 indices np.linspace(0, 199, num_slices) as int64; bilinear "
     "resize of each B-scan to slice_size; every B-scan is its own pretraining sample"),
    ("src/datasets/oct_volumes.py", ["0, 199, num=num_slices, dtype=np.int64",
                                     "(self.slice_size, self.slice_size), Image.BILINEAR",
                                     "slices_tensor = torch.stack(slices, dim=0)"],
     "evaluation volumes use the same 100 indices, bilinear resize to 256, no crop transform"),
    ("src/transforms.py", ["        T.RandomResizedCrop(", "crop_scale: (min, max) area fraction for ``RandomResizedCrop``.",
                           "top, left, height, width = T.RandomResizedCrop.get_params("],
     "pretraining view: RandomResizedCrop(crop_size, scale=crop_scale)"),
    ("configs/patch_vitb16_ep100.yaml", ["num_slices: 100", "crop_size: 256", "crop_scale: [0.3, 1.0]",
                                         "patch_size: 16"],
     "RANDOM run config: 100 B-scans per volume, 256 crop, crop scale 0.3-1.0, patch 16"),
    ("configs/patch_oracle_anatomical.yaml", ["num_slices: 100", "crop_size: 256", "patch_size: 16"],
     "CENTROID run config: same data settings"),
    ("src/train_patch.py", [("def jepa_forward_loss(", "h_full = target_encoder(images)"),
                            ("def jepa_forward_loss(", "h_full = F.layer_norm(h_full.float(), (h_full.size(-1),))"),
                            "targets = apply_masks(h_full, masks_pred)",
                            "predictions = predictor(encoder(images, masks_enc), masks_enc, masks_pred)",
                            "loss = F.smooth_l1_loss(predictions, targets)"],
     "no-grad EMA teacher on the full view, per-token LayerNorm, select targets, predictor on context "
     "encoder output with context/target masks, Smooth-L1"),
    ("src/models/vision_transformer.py", ["x = self.patch_embed(x)  # (B, N, D)",
                                          ("x = self.patch_embed(x)  # (B, N, D)", "x = apply_masks(x, masks)")],
     "patch tokens; context tokens selected before the transformer blocks"),
    ("src/helper.py", ["'vit_base':  dict(embed_dim=768,  depth=12, num_heads=12),"],
     "ViT-Base feature dimension 768"),
    ("src/eval_downstream.py", [("def run_patch_downstream(", "encoder.load_state_dict(ckpt['target_encoder'])"),
                                "parts.append(out.mean(dim=1).cpu())  # (chunk, D)",
                                "max_parts.append(out.amax(dim=1).cpu())",
                                "PRIMARY_VARIANT = 'patchmean_slicemean'",
                                ("class LinearHead(nn.Module):", "self.norm = nn.LayerNorm(in_dim)"),
                                ("class LinearHead(nn.Module):", "self.linear = nn.Linear(in_dim, out_dim)"),
                                "return self.linear(self.norm(x))", ("def evaluate(", "probs = torch.sigmoid(logits)")],
     "frozen EMA target encoder; patch mean per B-scan (patch max cached for secondary variants); primary "
     "variant patch-mean then slice-mean; LayerNorm + linear head; sigmoid probability"),
    ("src/models/attentive_pool_minimal.py", [("class MeanPool(nn.Module):", "return x.mean(dim=1)")],
     "MeanPool: mean across the B-scans of a volume"),
    ("configs/downstream_cr_seed_v1.yaml", ["num_slices: 100", "slice_size: 256", "probe_type: mean_pool",
                                            "head_type: linear", "extra_pooling: patch_max"],
     "registered probe: 100 B-scans at 256, mean pool, linear head, patch-max variants"),
]


def slice_indices():
    idx = np.linspace(0, 199, num=100, dtype=np.int64)
    assert idx[0] == 0 and idx[-1] == 199 and len(set(idx.tolist())) == 100
    assert idx.tolist() == [199 * k // 99 for k in range(100)]
    return idx


class Canvas:
    """Drawing helpers in design inches (paper layout); the poster scales the same layout."""

    def __init__(self, ax, k, fs, min_size=0.0):
        self.ax, self.k, self.fs, self.min_size = ax, k, fs, min_size
        self.texts = []

    def pt(self, size):
        return max(size, self.min_size) * self.fs

    def lw(self, width):
        return width * self.k

    def text(self, x, y, s, size=6.5, **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw.setdefault("color", st.TEXT)
        artist = self.ax.text(x, y, s, fontsize=self.pt(size), zorder=kw.pop("zorder", 20), **kw)
        self.texts.append(artist)
        return artist

    def box(self, xc, yc, w, h, fill, edge, lw=0.7, ls="-", r=0.035, z=4):
        self.ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=%g" % r,
                                         facecolor=fill, edgecolor=edge, linewidth=self.lw(lw), linestyle=ls,
                                         zorder=z))

    def trapezoid(self, xc, yc, w, h, fill, edge, narrow="right", lw=0.8, z=4):
        s = h * 0.22
        if narrow == "right":
            pts = [(xc - w / 2, yc - h / 2), (xc + w / 2, yc - h / 2 + s), (xc + w / 2, yc + h / 2 - s),
                   (xc - w / 2, yc + h / 2)]
        else:
            pts = [(xc - w / 2, yc - h / 2 + s), (xc + w / 2, yc - h / 2), (xc + w / 2, yc + h / 2),
                   (xc - w / 2, yc + h / 2 - s)]
        self.ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, linewidth=self.lw(lw),
                                  zorder=z, joinstyle="round"))

    def arrow(self, p0, p1, color=st.TEXT, lw=0.7, ls="-", rad=0.0, head=5.0, z=8, shrink=0.0):
        self.ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head * self.k,
                                          color=color, linewidth=self.lw(lw), linestyle=ls,
                                          connectionstyle="arc3,rad=%g" % rad, zorder=z,
                                          shrinkA=shrink, shrinkB=shrink))

    def line(self, pts, color=st.TEXT, lw=0.7, ls="-", z=8):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, linewidth=self.lw(lw), linestyle=ls, zorder=z,
                     solid_capstyle="butt")

    def image(self, img, xc, yc, s, cmap=None, edge="#333333", z=5):
        self.ax.imshow(img, cmap=cmap, vmin=0, vmax=255 if cmap else None, interpolation="none",
                       extent=(xc - s / 2, xc + s / 2, yc - s / 2, yc + s / 2), zorder=z)
        self.ax.add_patch(Rectangle((xc - s / 2, yc - s / 2), s, s, facecolor="none", edgecolor=edge,
                                    linewidth=self.lw(0.5), zorder=z + 0.5))

    def grid_cells(self, xc, yc, s, cells_list, colors, lw=0.9, halo="white"):
        """Outline cell sets of a 16x16 grid drawn on an image of side s centred at (xc, yc)."""
        x0, y0 = xc - s / 2, yc + s / 2
        for cells, colour in zip(cells_list, colors):
            segs = [[(x0 + px * s / 16, y0 - py * s / 16) for px, py in seg] for seg in st.boundary_segments(cells)]
            from matplotlib.collections import LineCollection
            if halo:
                self.ax.add_collection(LineCollection(segs, colors=halo, linewidths=self.lw(lw * 2.2), zorder=6.4))
            self.ax.add_collection(LineCollection(segs, colors=colour, linewidths=self.lw(lw), zorder=6.5))

    def tokens(self, x, y, colors, size=0.075, gap=0.012, vertical=True, edge="#333333"):
        for i, colour in enumerate(colors):
            if vertical:
                xx, yy = x - size / 2, y + (len(colors) / 2 - i - 1) * (size + gap) + gap / 2
            else:
                xx, yy = x + (i - len(colors) / 2) * (size + gap) + gap / 2, y - size / 2
            self.ax.add_patch(Rectangle((xx, yy), size, size, facecolor=colour, edgecolor=edge,
                                        linewidth=self.lw(0.4), zorder=7))

    def bar(self, xc, yc, w, h, fill, edge="#333333", z=6, cells=0):
        self.ax.add_patch(Rectangle((xc - w / 2, yc - h / 2), w, h, facecolor=fill, edgecolor=edge,
                                    linewidth=self.lw(0.5), zorder=z))
        for i in range(1, cells):
            yy = yc - h / 2 + i * h / cells
            self.line([(xc - w / 2, yy), (xc + w / 2, yy)], color=edge, lw=0.25, z=z + 0.1)

    def stack(self, xc, yc, w, h, n, dx, dy, used=None, retina=True):
        """Schematic stack of B-scans (drawn, not data). used[i] marks highlighted slices."""
        for i in reversed(range(n)):
            ox, oy = i * dx, i * dy
            is_used = True if used is None else used[i]
            fill = USED_FILL if is_used else "#FFFFFF"
            edge = USED_EDGE if is_used else "#C4C4C4"
            self.ax.add_patch(Rectangle((xc - w / 2 + ox, yc - h / 2 + oy), w, h, facecolor=fill,
                                        edgecolor=edge, linewidth=self.lw(0.6 if is_used else 0.45),
                                        zorder=3 + (n - i) * 0.01))
        if retina:
            xs = np.linspace(xc - w / 2 + 0.02, xc + w / 2 - 0.02, 50)
            for off, col, width in ((0.10, "#8A8A8A", 0.9), (0.02, "#6E6E6E", 1.3), (-0.06, "#9A9A9A", 0.7)):
                ys = yc + off * h / 0.4 + 0.035 * h / 0.4 * np.sin((xs - xc) / w * np.pi * 1.2)
                self.ax.plot(xs, ys, color=col, linewidth=self.lw(width), zorder=3.5 + n * 0.01)


VOLUME_STACK_SLICES = 12      # drawn slices in the schematic volume of band (a)
EVAL_STACK_SLICES = 7         # drawn slices in the schematic volume of band (c)
BODY_PT = 7.0                 # every label at print size (5.5 in): >= 6.6 pt; math >= 7.5 pt
MATH_PT = 7.6
SYMBOL_PT = 9.5
TITLE_PT = 7.8
TOKEN = 0.105                 # numbered target tokens (in)


def render(image, masks, guide_cells, idx, out_dir, stem, poster=False):
    """Three full-width bands: (a) volume -> tokens, (b) pretraining, (c) evaluation."""
    W, H = 5.5, 3.70
    k = 12.0 / 5.5 if poster else 1.0
    fs = k                    # poster: same layout scaled; labels print at >= 15 pt at 12 in
    with plt.rc_context(st.rc(fs)):
        fig = plt.figure(figsize=(W * k, H * k))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        c = Canvas(ax, k, fs)
        ctx = masks["context"][0]
        T, M, S, B = TITLE_PT, MATH_PT, SYMBOL_PT, BODY_PT
        bands = {"a": (0.02, 2.92, 5.46, 0.76), "b": (0.02, 1.06, 5.46, 1.80), "c": (0.02, 0.02, 5.46, 0.98)}
        titles = {"a": "(a) One OCT volume: 100 of its 200 B-scans, each 256 patch tokens",
                  "b": "(b) Pretraining: I-JEPA on one B-scan view per sample (random crop)",
                  "c": "(c) Frozen evaluation: one prediction per volume"}
        for key, (x, y, w, h) in bands.items():
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor="#FAFAFA", edgecolor="#C8C8C8", linewidth=c.lw(0.6), zorder=1))
            c.text(x + 0.07, y + h - 0.095, titles[key], size=T, ha="left", weight="bold")

        # ---------------- (a) volume -> B-scans -> tokens ----------------
        ya = 2.92
        used = [i % 2 == 0 for i in range(VOLUME_STACK_SLICES)]
        c.stack(0.40, ya + 0.27, 0.40, 0.25, VOLUME_STACK_SLICES, 0.025, 0.0165, used=used)
        c.text(1.02, ya + 0.47, "200 B-scans per volume, 200 × 200 px", size=B, ha="left")
        ax.add_patch(Rectangle((1.02, ya + 0.285), 0.085, 0.085, facecolor=USED_FILL, edgecolor=USED_EDGE,
                               linewidth=c.lw(0.6), zorder=5))
        c.text(1.15, ya + 0.3275, "used: 100, evenly spaced", size=B, ha="left")
        ax.add_patch(Rectangle((2.55, ya + 0.285), 0.085, 0.085, facecolor="#FFFFFF", edgecolor="#C4C4C4",
                               linewidth=c.lw(0.45), zorder=5))
        c.text(2.68, ya + 0.3275, "skipped", size=B, ha="left", color=st.MUTED)
        c.text(1.02, ya + 0.17, "indices 0, 2, …, 196, 199 (pretraining and evaluation)", size=B, ha="left")
        c.arrow((3.70, ya + 0.33), (3.92, ya + 0.33), lw=0.7)
        c.text(3.78, ya + 0.45, "resize", size=B, color=st.MUTED)
        s_a = 0.52
        xa = 4.20
        c.image(image, xa, ya + 0.33, s_a, cmap="gray")
        for g in range(1, 16):
            v = xa - s_a / 2 + g * s_a / 16
            hh = ya + 0.33 - s_a / 2 + g * s_a / 16
            c.line([(v, ya + 0.33 - s_a / 2), (v, ya + 0.33 + s_a / 2)], color="white", lw=0.25, z=6)
            c.line([(xa - s_a / 2, hh), (xa + s_a / 2, hh)], color="white", lw=0.25, z=6)
        c.text(4.52, ya + 0.53, "256 × 256 px", size=B, ha="left")
        c.text(4.52, ya + 0.39, "16 × 16 patches", size=B, ha="left")
        c.text(4.52, ya + 0.25, "$N = 256$ tokens", size=M, ha="left")
        c.text(4.52, ya + 0.11, "(OCTDL image)", size=B, ha="left", color=st.MUTED)

        # ---------------- (b) pretraining ----------------
        yb0 = 1.06
        yt, ym, yb = yb0 + 1.26, yb0 + 0.82, yb0 + 0.38     # context row, sampler row, target row
        x0, x1, x2, x3, x4, x5, x6 = 0.43, 1.27, 2.12, 3.10, 3.92, 4.62, 5.14
        s_t = 0.46
        # inputs: guide (top) and view x (bottom)
        from matplotlib.colors import to_rgb
        guide = np.repeat((image.astype(np.float32) / 255.0)[..., None], 3, axis=2)
        gcol = np.array(to_rgb(st.GUIDE_COLOR), np.float32)
        for yy, xx in zip(*np.nonzero(guide_cells)):
            blk = guide[yy * 16:(yy + 1) * 16, xx * 16:(xx + 1) * 16]
            blk[:] = 0.58 * blk + 0.42 * gcol
        c.image((guide * 255).astype(np.uint8), x0, yt, s_t)
        c.grid_cells(x0, yt, s_t, [guide_cells], [st.GUIDE_COLOR], lw=0.7, halo="black")
        c.text(x0, yt + 0.31, "guide $A_s(x)$", size=M)
        c.text(x0, yt - 0.30, "RANDOM: none", size=B, color=st.MUTED)
        c.image(image, x0, yb, s_t, cmap="gray")
        c.text(x0, yb + 0.31, "view $x$", size=M)
        # sampler
        c.box(x1, ym, 0.68, 0.38, "#FFFFFF", "#333333", lw=0.7)
        c.text(x1, ym + 0.075, "sampler", size=B)
        c.text(x1, ym - 0.075, "$(C, T) \\sim \\Pi_s$", size=M)
        c.arrow((x0 + s_t / 2, yt - 0.10), (x1 - 0.22, ym + 0.19), lw=0.7, rad=-0.2)
        c.arrow((x0 + s_t / 2, yb + 0.10), (x1 - 0.22, ym - 0.19), lw=0.7, rad=0.2)
        # context branch leaves the sampler's top; target positions leave its right side
        ctx_view = np.full(image.shape, 222, np.uint8)
        cm = st.cells_mask(ctx)
        for yy, xx in zip(*np.nonzero(cm)):
            ctx_view[yy * 16:(yy + 1) * 16, xx * 16:(xx + 1) * 16] = image[yy * 16:(yy + 1) * 16, xx * 16:(xx + 1) * 16]
        c.image(ctx_view, x2, yt, s_t, cmap="gray")
        c.text(x2, yt + 0.31, "context $C$", size=M)
        c.line([(x1, ym + 0.19), (x1, yt)], lw=0.7)
        c.arrow((x1, yt), (x2 - s_t / 2, yt), lw=0.7)
        c.line([(x1 + 0.34, ym), (x4, ym)], lw=0.7)
        c.text(x1 + 0.39, ym + 0.10, "target positions", size=B, ha="left")
        for j in range(4):
            st.draw_token(ax, 2.70 + (j - 1.5) * (TOKEN + 0.015), ym, j, TOKEN, c.pt(6.8), c.lw(0.4), z=9)
        # full view for the teacher
        c.image(image, x2, yb, s_t, cmap="gray")
        c.text(x2, yb - 0.29, "full view $x$", size=M)
        c.arrow((x0 + s_t / 2, yb - 0.12), (x2 - s_t / 2, yb - 0.12), lw=0.7)
        # encoders
        c.trapezoid(x3, yt, 0.52, 0.44, st.TRAIN_FILL, st.TRAIN_EDGE, narrow="right")
        c.text(x3 - 0.03, yt, "$f_\\theta$", size=S)
        c.text(x3, yt + 0.30, "context encoder", size=B)
        c.arrow((x2 + s_t / 2, yt), (x3 - 0.26, yt), lw=0.7)
        c.trapezoid(x3, yb, 0.52, 0.44, st.EMA_FILL, st.EMA_EDGE, narrow="right")
        c.text(x3 - 0.03, yb, "$f_{\\bar\\theta}$", size=S)
        c.text(x3, yb - 0.29, "EMA target encoder", size=B)
        c.arrow((x2 + s_t / 2, yb), (x3 - 0.26, yb), lw=0.7)
        c.arrow((x3 - 0.10, yt - 0.215), (x3 - 0.10, yb + 0.215), lw=0.7, ls=(0, (2.5, 1.6)), color=st.MUTED)
        c.text(x3 - 0.04, ym - 0.115, "EMA", size=B, ha="left", color=st.MUTED)
        # predictor and teacher-side selection
        c.trapezoid(x4, yt, 0.52, 0.44, st.TRAIN_FILL, st.TRAIN_EDGE, narrow="left")
        c.text(x4 + 0.03, yt, "$g_\\phi$", size=S)
        c.text(x4, yt + 0.30, "predictor", size=B)
        c.arrow((x3 + 0.26, yt), (x4 - 0.26, yt), lw=0.7)
        c.text((x3 + x4) / 2, yt + 0.10, "$h_C$", size=M)
        c.box(x4, yb, 0.70, 0.40, st.EMA_FILL, st.EMA_EDGE, lw=0.6)
        c.text(x4, yb + 0.08, "LayerNorm,", size=B)
        c.text(x4, yb - 0.08, "select $T_m$", size=M)
        c.arrow((x3 + 0.26, yb), (x4 - 0.35, yb), lw=0.7)
        c.arrow((x4, ym), (x4, yt - 0.22), lw=0.7)
        c.arrow((x4, ym), (x4, yb + 0.20), lw=0.7)
        c.text(x4 + 0.05, ym + 0.115, "mask tokens", size=B, ha="left", color=st.MUTED)
        # predictions and targets (numbered tokens)
        for j in range(4):
            st.draw_token(ax, x5, yt + (1.5 - j) * (TOKEN + 0.012), j, TOKEN, c.pt(6.8), c.lw(0.4), z=9)
            st.draw_token(ax, x5, yb + (1.5 - j) * (TOKEN + 0.012), j, TOKEN, c.pt(6.8), c.lw(0.4), z=9)
        c.text(x5 + 0.10, yt + 0.20, "$\\hat z_m$", size=M + 0.6, ha="left")
        c.text(x5 + 0.10, yb - 0.20, "$z_m$", size=M + 0.6, ha="left")
        c.arrow((x4 + 0.26, yt), (x5 - 0.08, yt), lw=0.7)
        c.arrow((x4 + 0.35, yb), (x5 - 0.08, yb), lw=0.7)
        c.text((x4 + 0.35 + x5 - 0.08) / 2, yb + 0.10, "sg", size=B, color=st.MUTED)
        c.box(x6, ym, 0.56, 0.44, st.LOSS_FILL, st.LOSS_EDGE, lw=0.8)
        c.text(x6, ym + 0.085, "Smooth-L1", size=B)
        c.text(x6, ym - 0.085, "per $T_m$", size=M)
        c.arrow((x5 + 0.07, yt - 0.05), (x6, ym + 0.23), lw=0.7, rad=-0.3)
        c.arrow((x5 + 0.07, yb + 0.05), (x6, ym - 0.23), lw=0.7, rad=0.3)

        # ---------------- (c) evaluation: data -> operators as boxes on the arrow ----------------
        ye = 0.49
        top, bot = ye + 0.285, ye - 0.275
        xs_, xe_, xm_, xo1, xb_, xo2, xv_, xh_, xp_ = 0.40, 1.03, 1.68, 2.25, 2.78, 3.33, 3.78, 4.33, 5.11
        c.stack(xs_ - 0.07, ye - 0.03, 0.30, 0.23, EVAL_STACK_SLICES, 0.022, 0.017, used=None, retina=True)
        c.text(xs_, top, "100 B-scans", size=B)
        c.text(xs_, bot, "full view", size=B, color=st.MUTED)
        c.arrow((xs_ + 0.22, ye), (xe_ - 0.22, ye), lw=0.7)
        c.trapezoid(xe_, ye, 0.42, 0.38, st.EMA_FILL, st.EMA_EDGE, narrow="right")
        c.text(xe_ - 0.03, ye, "$f_{\\bar\\theta}$", size=S)
        c.text(xe_, top, "frozen", size=B)
        c.text(xe_, bot, "EMA encoder", size=B, color=st.MUTED)
        c.arrow((xe_ + 0.21, ye), (xm_ - 0.13, ye), lw=0.7)
        mat_w, mat_h = 0.26, 0.34
        ax.add_patch(Rectangle((xm_ - mat_w / 2, ye - mat_h / 2), mat_w, mat_h, facecolor=st.TRAIN_FILL,
                               edgecolor="#333333", linewidth=c.lw(0.5), zorder=6))
        for g in range(1, 8):
            yy = ye - mat_h / 2 + g * mat_h / 8
            c.line([(xm_ - mat_w / 2, yy), (xm_ + mat_w / 2, yy)], color="#7DA7CF", lw=0.3, z=6.2)
        c.text(xm_, top, "per B-scan", size=B)
        c.text(xm_, bot, "256 × 768", size=B)
        for xo, (l1, l2), x_from, x_to in ((xo1, ("mean over", "patches"), xm_ + mat_w / 2, xb_ - 0.08),
                                           (xo2, ("mean over", "B-scans"), xb_ + 0.08, xv_ - 0.03)):
            c.arrow((x_from, ye), (xo - 0.31, ye), lw=0.7)
            c.box(xo, ye, 0.62, 0.32, "#FFFFFF", "#7A7A7A", lw=0.5, r=0.05)
            c.text(xo, ye + 0.065, l1, size=B)
            c.text(xo, ye - 0.07, l2, size=B)
            c.arrow((xo + 0.31, ye), (x_to, ye), lw=0.7)
        for j in range(4):
            c.bar(xb_ - 0.054 + j * 0.036, ye, 0.027, 0.34, st.TRAIN_FILL)
        c.text(xb_, top, "B-scan vectors", size=B)
        c.text(xb_, bot, "100 × 768", size=B)
        c.bar(xv_, ye, 0.032, 0.34, st.TRAIN_FILL)
        c.text(xv_, top, "volume", size=B)
        c.text(xv_, bot, "768", size=B)
        c.arrow((xv_ + 0.02, ye), (xh_ - 0.31, ye), lw=0.7)
        c.box(xh_, ye, 0.62, 0.36, st.TRAIN_FILL, st.TRAIN_EDGE, lw=0.7)
        c.text(xh_, ye + 0.075, "LayerNorm", size=B)
        c.text(xh_, ye - 0.08, "+ linear", size=B)
        c.text(xh_, top, "trained head", size=B)
        c.arrow((xh_ + 0.31, ye), (xp_ - 0.33, ye), lw=0.7)
        c.text(xp_, ye + 0.075, "sigmoid:", size=B)
        c.text(xp_, ye - 0.08, "p(glaucoma)", size=B)
        c.text(2.76, 0.068, "Secondary variants: patch max per B-scan; max, or mean and max, over B-scans.",
               size=B * 0.96, color=st.MUTED)
        # Every drawn text, including the token digits: no collision, fully inside the canvas.
        guard = st.assert_layout(fig)
        outputs = st.save_all(fig, out_dir, stem, "I-JEPA pretraining and volume-level evaluation",
                              dpi=300 if poster else 400)
        plt.close(fig)
    return outputs, [round(W * k, 4), round(H * k, 4)], guard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    assert st.sha(STAGED_EVIDENCE) == STAGED_EVIDENCE_SHA256, "staged replay evidence changed"
    staged = json.loads(STAGED_EVIDENCE.read_text(encoding="utf-8"))
    native, image = st.load_crop()
    masks = staged["masks"][EXAMPLE_ARM]
    guide_cells = np.asarray(staged["guide_overlays"][EXAMPLE_ARM], bool)
    idx = slice_indices()
    facts = []
    for rel, needles, claim in FACTS:
        rec = st.source_lines(rel, needles)
        rec["claim"] = claim
        facts.append(rec)
    selftest = st.guard_selftest()
    with st.math_scripts():
        paper_outputs, paper_inches, paper_guard = render(image, masks, guide_cells, idx, out, STEM)
        poster_outputs, poster_inches, poster_guard = render(image, masks, guide_cells, idx, out / "poster",
                                                             POSTER_STEM, poster=True)
    sizes = st.pdf_text_sizes(out / (STEM + ".pdf"))
    assert min(s for s, _ in sizes) >= 6.0, sorted(set(s for s, _ in sizes))
    evidence = {
        "status": "schematic_ijepa_framework_and_volume_handling_with_external_OCT_example",
        "generator": Path(__file__).name, "generator_sha256": st.sha(Path(__file__)),
        "style_module": "figstyle_v2.py", "style_module_sha256": st.sha(HERE / "figstyle_v2.py"),
        "source_image_file": str(st.SOURCE.relative_to(ROOT)), "source_image_sha256": st.sha(st.SOURCE),
        "primary_license_file": str(st.LICENSE.relative_to(ROOT)), "primary_license_sha256": st.sha(st.LICENSE),
        "license": "CC BY 4.0", "license_source": "https://arxiv.org/abs/2312.08255v3",
        "attribution": st.ATTRIBUTION, "crop_xyxy": list(st.CROP), "input_tensor_sha256": st.INPUT_TENSOR_SHA256,
        "mask_source": {"file": str(STAGED_EVIDENCE.relative_to(ROOT)), "sha256": STAGED_EVIDENCE_SHA256,
                        "arm": EXAMPLE_ARM, "sampler_commit": staged["samplers"]["commit"],
                        "seed": staged["seed"], "block_size_seed": staged["block_size_seed"]},
        "plotted": {
            "a_volume": {"b_scans_per_volume": 200, "b_scan_native_pixels": [200, 200], "b_scans_used": 100,
                         "indices": idx.tolist(), "index_rule": "np.linspace(0, 199, 100, dtype=int64) == "
                         "floor(199 k / 99), k = 0..99", "resized_pixels": [256, 256], "patch_pixels": 16,
                         "grid": [16, 16], "tokens": 256,
                         "schematic_stack": ("%d drawn slices, alternate ones marked used (illustrates every "
                                             "second B-scan; not data)" % VOLUME_STACK_SLICES)},
            "b_pretraining": {"example_arm": EXAMPLE_ARM, "context": masks["context"][0],
                              "targets": masks["targets"], "guide_cells": guide_cells.astype(int).tolist(),
                              "view": "one B-scan per sample; RandomResizedCrop to 256 (scale 0.3-1.0)",
                              "edges": [["x", "sampler"], ["A_s(x)", "sampler"], ["sampler", "C"],
                                        ["C", "f_theta"], ["f_theta", "h_C"], ["h_C", "g_phi"],
                                        ["sampler", "T positions"], ["T positions", "mask tokens + p_Tm"],
                                        ["mask tokens + p_Tm", "g_phi"], ["g_phi", "z_hat_m"],
                                        ["x (full view)", "f_theta_bar"], ["f_theta_bar", "LayerNorm, select T_m"],
                                        ["T positions", "LayerNorm, select T_m"],
                                        ["LayerNorm, select T_m", "sg", "z_m"],
                                        ["z_hat_m", "Smooth-L1 per T_m"], ["z_m", "Smooth-L1 per T_m"]],
                              "parameter_update": [["f_theta", "f_theta_bar", "EMA; no gradient"]]},
            "c_evaluation": {"encoder": "frozen EMA target encoder (checkpoint key target_encoder)",
                             "schematic_stack": "%d drawn slices (not data)" % EVAL_STACK_SLICES,
                             "b_scans": 100, "view": "full B-scan resized to 256, no crop",
                             "tokens_per_b_scan": 256, "feature_dim": 768,
                             "primary": "patchmean_slicemean: mean over 256 patches, mean over 100 B-scans",
                             "head": "LayerNorm(768) + Linear(768 -> 1), sigmoid",
                             "secondary": ["patchmax_slicemean", "patchmean_slicemax",
                                           "patchmean_slicemeanmax"]}},
        "source_facts": facts,
        "palette": {"targets": st.TARGET_COLORS, "guide": st.GUIDE_COLOR, "trainable": st.TRAIN_FILL,
                    "ema_or_frozen": st.EMA_FILL, "loss": st.LOSS_FILL, "source": "Okabe-Ito"},
        "clinical_pixels": "Published external OCTDL crop only (panels a, b); volume stacks are drawn "
                           "schematics; no FairVision pixels",
        "identifiers_or_patient_metadata_exported": False, "numerical_readouts": False,
        "limits": "Schematic of the implemented protocol; not a measured feature, cohort statistic or result.",
        "typography": {"pdf_text_min_pt": min(s for s, _ in sizes),
                       "pdf_text_sizes_pt": sorted(set(s for s, _ in sizes)),
                       "smallest_spans": sorted(sizes)[:6],
                       "design": "labels %.1f pt, math %.1f pt, encoder symbols %.1f pt, titles %.1f pt at the "
                                 "5.5 in print width; single-level sub/superscripts only, drawn at %.1f x "
                                 "their base size (matplotlib default 0.7) so that they print at >= 6 pt" % (
                                     BODY_PT, MATH_PT, SYMBOL_PT, TITLE_PT, st.MATH_SCRIPT_SCALE),
                       "math_script_scale": st.MATH_SCRIPT_SCALE,
                       "target_identity": "numbered tokens 1-4 in the Okabe-Ito group colours"},
        "layout_guard": {"paper": paper_guard, "poster": poster_guard,
                         "scope": ("figstyle_v2.layout_guard: every visible text artist (band titles, labels, "
                                   "math labels, encoder symbols, all token digits) pairwise without collision "
                                   "(2 pt horizontal / 0.5 pt vertical clearance) and fully inside the canvas "
                                   "(0.5 pt margin); asserted before export"),
                         "selftest": selftest},
        "caption_summary": (
            "From an OCT volume to I-JEPA pretraining and volume-level evaluation. (a) Each FairVision volume "
            "has 200 B-scans; pretraining and evaluation use the same 100 evenly spaced ones, resized to 256 by "
            "256 pixels (N=256 patches of 16 by 16 pixels). (b) Each B-scan is a separate sample (random resized "
            "crop). Strategies differ in the sampler of context C (grey: omitted tokens) and targets T_m (uniform "
            "or from A_s(x); CENTROID's masks of Figure 1); encoders, predictor and loss are shared. The predictor "
            "sees h_C and mask tokens at the target positions; the EMA target encoder's normalized outputs at T_m "
            "are the stop-gradient Smooth-L1 targets. (c) The frozen EMA encoder sees uncropped B-scans; "
            "patch features are averaged per B-scan, then across the volume, before a LayerNorm-linear head. "
            "Stacks are schematic; the B-scan is the OCTDL crop."),
        "alt_text": ("Three panels. (a) A schematic stack of OCT B-scans with alternate slices marked used, "
                     "an example B-scan with a 16 by 16 patch grid. (b) I-JEPA: view x and guide feed a "
                     "sampler; the context view goes through the context encoder and predictor with mask "
                     "tokens at the target positions (numbered tokens 1 to 4 in the group colours); the full view goes "
                     "through the EMA target encoder, "
                     "LayerNorm and target selection with stop-gradient; Smooth-L1 per target group. "
                     "(c) The frozen EMA encoder encodes each of 100 B-scans; patch tokens are averaged per "
                     "B-scan, B-scan vectors averaged per volume, then LayerNorm and linear head give the "
                     "glaucoma probability."),
        "outputs": {"paper": {"inches": paper_inches, **paper_outputs},
                    "poster": {"inches": poster_inches, "dir": "poster", **poster_outputs}},
        "out_dir": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
    }
    (out / f"{STEM}.evidence.json").write_text(json.dumps(evidence, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"facts": [{"path": f["path"], "lines": f["lines"]} for f in facts],
                      "outputs": evidence["outputs"]}, indent=1))


if __name__ == "__main__":
    main()
