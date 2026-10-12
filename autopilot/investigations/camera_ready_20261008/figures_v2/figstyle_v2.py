"""Shared visual language for the camera-ready v2 figures (I13).

One palette, one font family and one set of drawing helpers for the strategy
figure and the framework/volume figure, so that colours mean the same thing
everywhere:

* target groups T1..T4: Okabe-Ito orange, sky blue, bluish green, reddish purple
  (colour-blind-safe categorical set; identity only, never relevance);
* placement guide A_s(x): Okabe-Ito yellow (hard cells) / yellow alpha ramp (soft);
* hidden-but-not-target cells: neutral grey; visible context: the image itself;
* trainable modules blue, EMA / frozen modules grey, loss vermillion.

The OCT pixels are the attributed CC BY 4.0 OCTDL crop (Kulyabin et al.,
arXiv:2312.08255v3, Fig. 1); no FairVision pixels are read or drawn.
"""
from pathlib import Path
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
LEGACY_FIG = ROOT / "autopilot" / "investigations" / "submission_refresh_20260910" / "figure"
STAGING = ROOT / "autopilot" / "investigations" / "camera_ready_20261008" / "figures_staging"
SOURCE = LEGACY_FIG / "octdl_figure1_original.png"
LICENSE = LEGACY_FIG / "octdl_primary_license.html"
SOURCE_SHA256 = "75bfdf6c9b2eb54e8cb9391f7d0bb92e30bbd1e4ba2a906c166a183b666ee7e5"
LICENSE_SHA256 = "98f0608fea4d233a4b41040413a224aaeb1e5ff8cf93a827be5751d8dc18ce18"
CROP = (3200, 245, 4480, 1525)
INPUT_TENSOR_SHA256 = "dcf1044bd39ab98061bce12cbae92c090cdd9c5753076792d0908ab25f29c64f"
ATTRIBUTION = "Mikhail Kulyabin et al., OCTDL, Fig. 1, arXiv:2312.08255v3 (2024)"

# Okabe-Ito (Wong 2011, Nat. Methods 8:441) colour-blind-safe set.
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9",
             "bluish_green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
             "vermillion": "#D55E00", "reddish_purple": "#CC79A7"}
TARGET_COLORS = [OKABE_ITO["orange"], OKABE_ITO["sky_blue"], OKABE_ITO["bluish_green"],
                 OKABE_ITO["reddish_purple"]]
GUIDE_COLOR = OKABE_ITO["yellow"]
GUIDE_EDGE = "#8C7F00"
HIDDEN_GREY = "#B4B4B4"
TRAIN_FILL, TRAIN_EDGE = "#DCEBF7", OKABE_ITO["blue"]
EMA_FILL, EMA_EDGE = "#EDEDED", "#5A5A5A"
LOSS_FILL, LOSS_EDGE = "#FBE3D2", OKABE_ITO["vermillion"]
DATA_FILL, DATA_EDGE = "#FFFFFF", "#5A5A5A"
TEXT = "#1A1A1A"
MUTED = "#555555"

GRID = 16


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rc(scale=1.0, salt="i13_figures_v2"):
    """Matplotlib rc for paper (scale 1: 7 pt base at 1:1 print size) or poster."""
    base = 7.0 * scale
    return {"font.family": "DejaVu Sans", "font.size": base, "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.5 * scale, "pdf.fonttype": 42, "svg.fonttype": "none",
            "svg.hashsalt": salt, "lines.solid_capstyle": "butt"}


def load_crop():
    """The 256x256 OCTDL crop exactly as the staged mask replay used it."""
    from PIL import Image
    assert sha(SOURCE) == SOURCE_SHA256, "OCTDL source image changed"
    assert sha(LICENSE) == LICENSE_SHA256, "OCTDL licence record changed"
    assert "creativecommons.org/licenses/by/4.0" in LICENSE.read_text(encoding="utf-8")
    original = Image.open(SOURCE)
    assert original.size == (4768, 1764)
    native = np.asarray(original.crop(CROP).convert("L")).copy()
    image = np.asarray(Image.fromarray(native).resize((256, 256), Image.Resampling.BILINEAR)).copy()
    tensor = (image.astype(np.float32) / 255.0)[None].repeat(3, axis=0)
    # Same bytes as torch.from_numpy(image).float().div(255).unsqueeze(0).repeat(3, 1, 1).
    assert hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest() == INPUT_TENSOR_SHA256
    return native, image


def cells_mask(indices):
    grid = np.zeros((GRID, GRID), bool)
    for i in indices:
        grid[divmod(int(i), GRID)] = True
    return grid


def boundary_segments(cells):
    """Outline of a set of grid cells as unit segments in (x, y) cell coordinates."""
    segments = []
    for y, x in zip(*np.nonzero(cells)):
        for dy, dx, seg in ((-1, 0, [(x, y), (x + 1, y)]), (1, 0, [(x, y + 1), (x + 1, y + 1)]),
                            (0, -1, [(x, y), (x, y + 1)]), (0, 1, [(x + 1, y), (x + 1, y + 1)])):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < cells.shape[0] and 0 <= nx < cells.shape[1] and cells[ny, nx]):
                segments.append(seg)
    return segments


def draw_outline(ax, cells, color, lw, x0=0.0, y0=0.0, scale=1.0, zorder=5, halo=None, halo_lw=None,
                 **kw):
    """Cell-set outline; an optional halo is drawn as a separate collection underneath
    (per-segment stroke effects would render as dashes)."""
    from matplotlib.collections import LineCollection
    segs = [[(x0 + px * scale, y0 + py * scale) for px, py in seg] for seg in boundary_segments(cells)]
    if not segs:
        return
    if halo is not None:
        ax.add_collection(LineCollection(segs, colors=halo, linewidths=halo_lw or 2.2 * lw,
                                         zorder=zorder - 0.5, capstyle="projecting"))
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=zorder,
                                     capstyle="projecting", **kw))


def rgba(color, alpha):
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def numeral_color(k):
    """Digit colour on target-group colour k (dark text except on bluish green)."""
    return "white" if TARGET_COLORS[k] == OKABE_ITO["bluish_green"] else "black"


def draw_token(ax, x, y, k, size, fontsize, lw, z=7, edge="#333333"):
    """Square token of target group k (colour + numeral k+1), centred at (x, y) in data units."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, facecolor=TARGET_COLORS[k],
                           edgecolor=edge, linewidth=lw, zorder=z))
    ax.text(x, y, str(k + 1), ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color=numeral_color(k), zorder=z + 0.5)


def place_numerals(groups, grid=GRID, min_sep=2.3, dot_clear=1.05):
    """Numeral anchor (cell units, x = column) for every target group, by one fixed rule.

    Top-left corner inside the block: for group 1, 2, 3, 4 in turn, the numeral sits at the
    centre of the first 2x2 square of the group's own cells in raster order (top row first,
    then leftmost), skipping a square only while its centre is closer than min_sep to an
    already placed numeral or dot_clear to a repeated-index dot. The numeral is drawn above all
    outlines with a white halo, so no edge crosses it. Groups without any 2x2 square (one-cell
    or two-cell groups) get the nearest target-free 2x2 square outside, with a leader line to
    their nearest cell."""
    import math
    from collections import Counter
    sets = [set(int(i) for i in g) for g in groups]
    counts = [Counter(int(i) for i in g) for g in groups]
    union_all = set().union(*sets)
    dots = [(c + 0.5, r + 0.5) for j in range(len(groups)) for i, n in counts[j].items() if n > 1
            for r, c in [divmod(i, grid)]]
    placed, leaders, rule = {}, {}, {}

    def clear(px, py):
        return (all(math.hypot(px - qx, py - qy) >= min_sep for qx, qy in placed.values())
                and all(math.hypot(px - dx, py - dy) >= dot_clear for dx, dy in dots))

    for k in range(len(groups)):
        choice, skipped = None, 0
        for r in range(grid - 1):
            for c in range(grid - 1):
                square = [(r + dr) * grid + c + dc for dr in range(2) for dc in range(2)]
                if not all(i in sets[k] for i in square):
                    continue
                if clear(c + 1.0, r + 1.0):
                    choice = (c + 1.0, r + 1.0, "top-left 2x2 inside" + (
                        "" if not skipped else " (%d earlier square(s) skipped: spacing)" % skipped))
                    break
                skipped += 1
            if choice:
                break
        if choice is None:
            cells = [divmod(i, grid) for i in sorted(sets[k])]
            best = None
            for r in range(grid - 1):
                for c in range(grid - 1):
                    square = [(r + dr) * grid + c + dc for dr in range(2) for dc in range(2)]
                    if any(i in union_all for i in square) or not clear(c + 1.0, r + 1.0):
                        continue
                    d = min(math.hypot(c + 1.0 - cc - 0.5, r + 1.0 - rr - 0.5) for rr, cc in cells)
                    if best is None or d < best[0] - 1e-9:
                        best = (d, c + 1.0, r + 1.0)
            choice = (best[1], best[2], "outside 2x2 with leader (no 2x2 square of own cells)")
            leaders[k] = min(((cc + 0.5, rr + 0.5) for rr, cc in cells),
                             key=lambda p: math.hypot(p[0] - best[1], p[1] - best[2]))
        placed[k] = (choice[0], choice[1])
        rule[k] = choice[2]
    return placed, leaders, rule

def draw_numerals(ax, placed, leaders, fontsize, halo, lw, z=13):
    """Bold black group numerals with a white halo; leader lines in the group colour."""
    import math
    from matplotlib import patheffects
    for k, (px, py) in placed.items():
        if k in leaders:
            lx, ly = leaders[k]
            d = math.hypot(lx - px, ly - py)
            sx, sy = px + (lx - px) * 0.55 / d, py + (ly - py) * 0.55 / d   # start outside the digit
            ex, ey = lx - (lx - px) * 0.42 / d, ly - (ly - py) * 0.42 / d   # stop at the cell edge
            ax.plot([sx, ex], [sy, ey], color="white", linewidth=lw * 2.4, zorder=11.4,
                    solid_capstyle="butt")
            ax.plot([sx, ex], [sy, ey], color=TARGET_COLORS[k], linewidth=lw * 1.2, zorder=11.5,
                    solid_capstyle="butt")
        ax.text(px, py, str(k + 1), ha="center", va="center_baseline", fontsize=fontsize,
                fontweight="bold", color="black", zorder=z,
                path_effects=[patheffects.withStroke(linewidth=halo, foreground="white")])


def text_overlaps(fig, artists, pad_pt=2.0, pad_y_pt=0.5):
    """Pairs of text artists whose rendered boxes, grown by pad_pt horizontally and pad_y_pt
    vertically, intersect (display coordinates)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad, pad_y = pad_pt * fig.dpi / 72.0, pad_y_pt * fig.dpi / 72.0
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in artists if t.get_text().strip()]
    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if (a.x0 - pad < b.x1 and b.x0 - pad < a.x1 and a.y0 - pad_y < b.y1 and b.y0 - pad_y < a.y1):
                hits.append([boxes[i][0], boxes[j][0]])
    return hits

def pdf_text_sizes(pdf_path):
    """Font sizes (pt) of every text span in a one-page figure PDF, at its own print size."""
    import fitz
    spans = []
    with fitz.open(str(pdf_path)) as doc:
        for block in doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        spans.append((round(span["size"], 2), span["text"]))
    return spans


def ijepa_view(image, targets, context, alpha=0.55, hidden=HIDDEN_GREY):
    """RGB rendering of the I-JEPA view of one B-scan on the 16x16 grid.

    Context cells keep the image; target cells are the image tinted with their
    group colour (alpha-composited per group in order, so overlaps mix);
    cells that are neither context nor target are flat grey.
    """
    from matplotlib.colors import to_rgb
    rgb = np.repeat((image.astype(np.float32) / 255.0)[..., None], 3, axis=2)
    size = image.shape[0] // GRID
    ctx = cells_mask(context)
    union = np.zeros((GRID, GRID), bool)
    for k, group in enumerate(targets):
        cells = cells_mask(group)
        union |= cells
        colour = np.array(to_rgb(TARGET_COLORS[k]), np.float32)
        for y, x in zip(*np.nonzero(cells)):
            block = rgb[y * size:(y + 1) * size, x * size:(x + 1) * size]
            block[:] = (1 - alpha) * block + alpha * colour
    grey = np.array(to_rgb(hidden), np.float32)
    for y, x in zip(*np.nonzero(~ctx & ~union)):
        rgb[y * size:(y + 1) * size, x * size:(x + 1) * size] = grey
    return rgb, ctx, union


def save_all(fig, out_dir, stem, title, dpi=300):
    """PNG/PDF/SVG with deterministic metadata; returns {suffix: {file, sha256}}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        if suffix == "pdf":
            meta = {"Creator": "Matplotlib", "Title": title, "Author": "", "CreationDate": None,
                    "ModDate": None}
        elif suffix == "svg":
            meta = {"Creator": "Matplotlib", "Title": title, "Date": None}
        else:
            meta = {"Software": "Matplotlib", "Title": title}
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, facecolor="white", transparent=False, metadata=meta)
        outputs[suffix] = {"file": path.name, "sha256": sha(path)}
    return outputs


def source_lines(rel, needles):
    """1-based line numbers of each needle in a repository source.

    A plain string must occur on exactly one line; an (anchor, needle) pair means the first
    line containing needle after the single line containing anchor."""
    path = ROOT / rel
    lines = path.read_text(encoding="utf-8").splitlines()

    def unique(text, start=0):
        hits = [i + 1 for i, line in enumerate(lines) if i >= start and text in line]
        return hits

    found = {}
    for item in needles:
        if isinstance(item, (tuple, list)):
            anchor, needle = item
            anchors = unique(anchor)
            if len(anchors) != 1:
                raise AssertionError("%s: anchor %r found %s" % (rel, anchor, anchors))
            hits = unique(needle, anchors[0])
            if not hits:
                raise AssertionError("%s: %r not found after %r" % (rel, needle, anchor))
            found[needle] = hits[0]
        else:
            hits = unique(item)
            if len(hits) != 1:
                raise AssertionError("%s: expected one line with %r, found %s" % (rel, item, hits))
            found[item] = hits[0]
    return {"path": rel, "sha256": sha(path), "lines": found}
