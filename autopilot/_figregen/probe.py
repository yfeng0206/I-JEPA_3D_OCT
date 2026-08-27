"""Capture every numeric series that a figure generator hands to matplotlib.

The point is to prove that a presentation-only edit changed no plotted VALUE.
Series are read back off the returned artists rather than parsed out of the
call arguments, so a bar chart and a dot-and-interval plot of the same numbers
normalise to the same record: {x, y, ylo, yhi}. That is what makes the
before/after diff meaningful across an encoding change.

Styling (colour, marker, linestyle, width, axis limits) is deliberately NOT
recorded, because that is exactly what is allowed to change.

Usage:
  python probe.py p8   <out.json>
  python probe.py figs5 <out.json>
"""
import hashlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

REPO = r"C:\Users\Gary\Desktop\jepa"

_records = []          # (fig_id, record)
_labels = []           # (fig_id, kind, text)
FIGURES = {}           # basename -> [record, ...]
LABELS = {}            # basename -> [text, ...]


def _arr(v):
    a = np.asarray(v, dtype=np.float64).ravel()
    return a


def _ser(v):
    """Serialise one numeric series exactly and compactly."""
    a = _arr(v)
    d = {"n": int(a.size),
         "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    if a.size <= 64:
        # repr at full float64 precision so the JSON itself is a proof artifact
        d["values"] = [float(x) for x in a]
    else:
        d["head"] = [float(x) for x in a[:4]]
        d["tail"] = [float(x) for x in a[-4:]]
        d["sum"] = float(a.sum())
    return d


def _rec(ax, kind, label, x, y, ylo=None, yhi=None):
    r = {"kind": kind, "axes": id(ax), "label": None if label is None else str(label),
         "x": _ser(x), "y": _ser(y)}
    if ylo is not None:
        r["ylo"] = _ser(ylo)
        r["yhi"] = _ser(yhi)
    _records.append((id(ax.figure), r))


def _errbar_bounds(container):
    """Absolute (lo, hi) per point from an ErrorbarContainer's bar segments."""
    try:
        barlinecols = container.lines[2]
    except Exception:
        return None, None
    if not barlinecols:
        return None, None
    segs = barlinecols[0].get_segments()
    if not segs:
        return None, None
    lo = [float(np.min(np.asarray(s)[:, 1])) for s in segs]
    hi = [float(np.max(np.asarray(s)[:, 1])) for s in segs]
    return lo, hi


_orig = {}
_inside = {"bar": False}


def patch():
    _orig["plot"] = Axes.plot
    _orig["bar"] = Axes.bar
    _orig["errorbar"] = Axes.errorbar
    _orig["scatter"] = Axes.scatter
    _orig["fill_between"] = Axes.fill_between
    _orig["axhline"] = Axes.axhline
    _orig["axvline"] = Axes.axvline
    _orig["text"] = Axes.text
    _orig["annotate"] = Axes.annotate
    _orig["set_xticklabels"] = Axes.set_xticklabels
    _orig["savefig"] = Figure.savefig

    def plot(self, *a, **k):
        lines = _orig["plot"](self, *a, **k)
        for ln in lines:
            _rec(self, "line", ln.get_label(),
                 ln.get_xdata(orig=True), ln.get_ydata(orig=True))
        return lines

    def bar(self, *a, **k):
        # bar() calls errorbar() internally with fmt='none'; suppress the nested
        # record so one bar group yields exactly one {x, y, ylo, yhi} entry.
        _inside["bar"] = True
        try:
            c = _orig["bar"](self, *a, **k)
        finally:
            _inside["bar"] = False
        xs = [p.get_x() + p.get_width() / 2.0 for p in c.patches]
        ys = [p.get_height() for p in c.patches]
        lo = hi = None
        if getattr(c, "errorbar", None) is not None:
            lo, hi = _errbar_bounds(c.errorbar)
        _rec(self, "interval", k.get("label"), xs, ys, lo, hi)
        return c

    def errorbar(self, *a, **k):
        c = _orig["errorbar"](self, *a, **k)
        if _inside["bar"]:
            return c
        dl = c.lines[0]
        lo, hi = _errbar_bounds(c)
        _rec(self, "interval", k.get("label"),
             dl.get_xdata(orig=True), dl.get_ydata(orig=True), lo, hi)
        return c

    def scatter(self, *a, **k):
        c = _orig["scatter"](self, *a, **k)
        off = np.asarray(c.get_offsets(), dtype=np.float64)
        _rec(self, "points", k.get("label"), off[:, 0], off[:, 1])
        return c

    def fill_between(self, x, y1, y2=0, *a, **k):
        c = _orig["fill_between"](self, x, y1, y2, *a, **k)
        _rec(self, "band", k.get("label"), x, y1, _arr(y2).tolist(), None)
        # y2 lives in ylo; store it explicitly so the band is fully captured
        _records[-1][1]["yhi"] = _ser(y2)
        _records[-1][1]["ylo"] = _ser(y1)
        _records[-1][1]["y"] = _ser(y1)
        return c

    def axhline(self, y=0, *a, **k):
        _rec(self, "hline", k.get("label"), [0.0], [y])
        return _orig["axhline"](self, y, *a, **k)

    def axvline(self, x=0, *a, **k):
        _rec(self, "vline", k.get("label"), [x], [0.0])
        return _orig["axvline"](self, x, *a, **k)

    def text(self, x, y, s, *a, **k):
        _labels.append((id(self.figure), "text", str(s)))
        return _orig["text"](self, x, y, s, *a, **k)

    def annotate(self, t, *a, **k):
        _labels.append((id(self.figure), "annotate", str(t)))
        return _orig["annotate"](self, t, *a, **k)

    def set_xticklabels(self, labels, *a, **k):
        for L in labels:
            _labels.append((id(self.figure), "xtick", str(L)))
        return _orig["set_xticklabels"](self, labels, *a, **k)

    def savefig(self, fname, *a, **k):
        r = _orig["savefig"](self, fname, *a, **k)
        try:
            base = os.path.basename(str(fname))
        except Exception:
            base = "<stream>"
        fid = id(self)
        recs = [rr for (f, rr) in _records if f == fid]
        labs = [t for (f, kk, t) in _labels if f == fid]
        for ax in self.axes:
            for h, L in zip(*ax.get_legend_handles_labels()):
                labs.append("legend:" + str(L))
            if ax.get_title():
                labs.append("title:" + ax.get_title())
            if ax.get_xlabel():
                labs.append("xlabel:" + ax.get_xlabel())
            if ax.get_ylabel():
                labs.append("ylabel:" + ax.get_ylabel())
        FIGURES.setdefault(base, []).extend(recs)
        LABELS.setdefault(base, []).extend(labs)
        return r

    Axes.plot = plot
    Axes.bar = bar
    Axes.errorbar = errorbar
    Axes.scatter = scatter
    Axes.fill_between = fill_between
    Axes.axhline = axhline
    Axes.axvline = axvline
    Axes.text = text
    Axes.annotate = annotate
    Axes.set_xticklabels = set_xticklabels
    Figure.savefig = savefig


def canonical(recs):
    """Order-independent, axes-id-independent view of one figure's data."""
    out = []
    for r in recs:
        c = {k: v for k, v in r.items() if k != "axes"}
        out.append(json.dumps(c, sort_keys=True))
    return sorted(out)


def main():
    which, out = sys.argv[1], sys.argv[2]
    patch()
    if which == "p8":
        sys.path.insert(0, os.path.join(REPO, "autopilot"))
        import p8_make_assets
        p8_make_assets.main()
    elif which in ("figs5", "story", "story_orig"):
        sys.path.insert(0, os.path.join(REPO, "paper", "genai4health2026", "scripts"))
        if which == "story_orig":
            import _story_orig as msf
        else:
            import make_story_figures as msf
        if which == "figs5":
            msf.fig_mask_stats()
        else:
            msf.fig_signal()
            msf.fig_inverted_u()
            msf.fig_collapse()
            msf.fig_floor()
            msf.fig_mask_stats()
    else:
        raise SystemExit("unknown target %r" % which)

    payload = {b: {"canonical": canonical(v), "labels": sorted(set(LABELS.get(b, [])))}
               for b, v in FIGURES.items()}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print("probe wrote %s : %d figures" % (out, len(payload)))
    for b in sorted(payload):
        print("   %-34s %d series" % (b, len(payload[b]["canonical"])))


if __name__ == "__main__":
    main()
