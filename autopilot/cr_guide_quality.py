"""I6 camera-ready guide-quality analysis: CENTROID ribbon vs hard MIRAGE envelope.

Pre-declared plan: autopilot/investigations/camera_ready_20261008/guide_quality/ANALYSIS_PLAN.md

Read-only with respect to repository source.  The production CENTROID guide
(``CurriculumMaskGenerator._anatomical_prior_weight_grid_for_image``) is extracted by
AST from git revision fc4527a (the revision bound to the trained CENTROID run) and
executed as-is; ``ribbon_with_intermediates`` below is an attributed copy of that
method (src/masks/curriculum.py:1061-1150 at HEAD, :569 at fc4527a) that additionally
returns its intermediates, and its grid is asserted equal to the original on every
B-scan.  ``patch_occupancy``/``occupancy_is_valid`` are copied from
src/guides/mirage_envelope.py:387-421 with the shipped RepairParams defaults.

CPU only, headless.  No image pixels are written or plotted (FairVision CC BY-NC-ND):
outputs contain per-B-scan statistics with hashed case identifiers only.  Test split is
never read.
"""

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

torch.set_num_threads(1)

REPO = Path(r"C:\Users\Gary\Desktop\jepa")
FV = Path(r"D:\jepa_phase0\fairvision-glaucoma")
MIRAGE_OUT = Path(r"D:\jepa_phase0\mirage-goals\outputs\official-vitlarge\GOALS"
                  r"\MIRAGE-Large_frozen_convnext_CEGDice")
OUT = REPO / "autopilot" / "investigations" / "camera_ready_20261008" / "guide_quality"
PLAN = OUT / "ANALYSIS_PLAN.md"

PRODUCING_REV = "fc4527a"
FUNC = "_anatomical_prior_weight_grid_for_image"
NATIVE = 200
SIZE = 256
PATCH = 16
GRID = 16
OCC_T = 0.25
MIN_VALID_AREA = 0.05
MIN_VALID_SPAN = 0.40
EXPECTED_FINGERPRINT = "9a25a2cdb36f9cba"
K_POS = np.linspace(0, 99, 10).round().astype(int)        # positions in 100-slice set
SLICE_INDICES = np.linspace(0, 199, 100).astype(np.int64)  # dataset slice indices
PER_STRATUM = 50
VAL_PER_STRATUM = 25
SAMPLE_SEED = 620261008
VAL_SEED = 620261009
BOOT_SEED = 20261008
BOOT_B = 2000
NATIVE_PX_PER_PATCH = NATIVE / GRID                      # 12.5


class RibbonParams:
    """Production CENTROID parameters (results/pretraining/pretrain_oracle_anatomical/config.yaml:37-40)."""
    patch_size = PATCH
    height = GRID
    width = GRID
    oracle_region_frac = 0.28
    oracle_lateral_frac = 0.6
    oracle_row_offset = 0.0
    oracle_min_band_rows = 3


BAND_H = max(RibbonParams.oracle_min_band_rows,
             min(int(round((RibbonParams.oracle_region_frac
                            / RibbonParams.oracle_lateral_frac) * GRID)), GRID))
X_KEEP = max(1, min(int(round(RibbonParams.oracle_lateral_frac * GRID)), GRID))
X0 = (GRID - X_KEEP) // 2
X1 = X0 + X_KEEP
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

_PROD_FN = None


# ----------------------------------------------------------------------------- provenance

def _method_node(source):
    tree = ast.parse(source)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "CurriculumMaskGenerator")
    return next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == FUNC)


def _normalized_dump(node):
    node = ast.parse(ast.unparse(node)).body[0]
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None),
                                                            ast.Constant):
        node.body = body[1:]
    return ast.dump(node, include_attributes=False)


def extract_function_sources():
    sources = {}
    for label, rev in (("fc4527a", PRODUCING_REV), ("HEAD", "HEAD")):
        sources[label] = subprocess.run(
            ["git", "show", "%s:src/masks/curriculum.py" % rev], cwd=REPO,
            capture_output=True, text=True, check=True, encoding="utf-8").stdout
    sources["working_tree"] = (REPO / "src" / "masks" / "curriculum.py").read_text(encoding="utf-8")
    info = {}
    dumps = {}
    for label, src in sources.items():
        node = _method_node(src)
        dumps[label] = _normalized_dump(node)
        info[label] = {"lineno": node.lineno,
                       "normalized_ast_sha256": hashlib.sha256(dumps[label].encode()).hexdigest(),
                       "segment_sha256": hashlib.sha256(
                           ast.get_source_segment(src, node).encode()).hexdigest()}
    info["fc4527a_equals_HEAD"] = dumps["fc4527a"] == dumps["HEAD"]
    info["fc4527a_equals_working_tree"] = dumps["fc4527a"] == dumps["working_tree"]
    prod_segment = ast.unparse(_method_node(sources["fc4527a"]))
    return prod_segment, info


def init_production(segment):
    global _PROD_FN
    ns = {"torch": torch}
    exec(compile(segment, "%s:src/masks/curriculum.py:%s" % (PRODUCING_REV, FUNC), "exec"), ns)
    fn = ns[FUNC]
    params = RibbonParams()
    _PROD_FN = lambda image: fn(params, image)


# ----------------------------------------------------------------------------- guides

def ribbon_with_intermediates(img_cpu):
    """Copy of curriculum.py CENTROID method (fc4527a) returning intermediates."""
    C, Hp, Wp = img_cpu.shape
    ps, H, W = PATCH, GRID, GRID
    patches = img_cpu.permute(1, 2, 0).reshape(H, ps, W, ps, C).permute(0, 2, 1, 3, 4)
    patch_mean = patches.float().mean(dim=(2, 3, 4))
    prof = patch_mean - patch_mean.min()
    ys = torch.arange(H, dtype=torch.float32)
    eps = 1e-6
    row_mass = prof.sum(dim=1)
    gtot = float(row_mass.sum().item())
    flat = not gtot > eps
    global_c = float((ys * row_mass).sum().item() / gtot) if gtot > eps else H / 2.0
    col_mass = prof.sum(dim=0)
    col_c = torch.where(
        col_mass > eps,
        (ys.view(H, 1) * prof).sum(dim=0) / col_mass.clamp(min=eps),
        torch.full((W,), global_c),
    )
    col_c = F.avg_pool1d(col_c.view(1, 1, W), kernel_size=3, stride=1, padding=1,
                         count_include_pad=False).view(W)
    col_c = col_c + RibbonParams.oracle_row_offset * H
    grid = torch.zeros(H, W, dtype=torch.float32)
    tops, clamped = [], []
    for x in range(X0, X1):
        c = int(round(float(col_c[x])))
        raw_top = c - BAND_H // 2
        top = max(0, min(raw_top, H - BAND_H))
        grid[top:top + BAND_H, x] = 1.0
        tops.append(top)
        clamped.append(raw_top != top)
    return {"grid": grid, "col_c": col_c.numpy().astype(np.float64), "tops": np.array(tops),
            "clamped": np.array(clamped), "zero_mass_cols": (col_mass <= eps).numpy(),
            "flat": flat}


def patch_occupancy(mask, patch_size=PATCH):
    """Copy of src/guides/mirage_envelope.py:387-405."""
    h, w = mask.shape
    return (mask.astype(np.float32)
            .reshape(h // patch_size, patch_size, w // patch_size, patch_size).mean(axis=(1, 3)))


def occupancy_is_valid(grid):
    """Copy of src/guides/mirage_envelope.py:408-421 with RepairParams defaults."""
    grid = np.asarray(grid, dtype=np.float32)
    return bool(float(grid.mean()) >= MIN_VALID_AREA
                and float((grid.max(axis=0) > 0.0).mean()) >= MIN_VALID_SPAN)


def image_tensor(native_u8):
    img = Image.fromarray(native_u8, mode="L").resize((SIZE, SIZE), Image.BILINEAR).convert("RGB")
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def snr_proxy(native_u8):
    x = native_u8.ravel().astype(np.float64)
    med = np.percentile(x, 50)
    bg = x[x <= med]
    sd_raw = bg.std()
    sd = max(sd_raw, 1.0)
    signal = np.percentile(x, 99)
    return 20.0 * np.log10(max(signal - bg.mean(), 1.0) / sd), bool(sd_raw < 1.0)


def centroid_descriptors(rib):
    cc = rib["col_c"][X0:X1]
    xs = np.arange(X0, X1, dtype=np.float64)
    return {
        "ribbon_cells": int(rib["grid"].sum().item()),
        "clamped_cols": int(rib["clamped"].sum()),
        "zero_mass_cols_all": int(rib["zero_mass_cols"].sum()),
        "zero_mass_cols_ribbon": int(rib["zero_mass_cols"][X0:X1].sum()),
        "flat_image": bool(rib["flat"]),
        "centroid_range": float(cc.max() - cc.min()),
        "abs_slope": float(abs(np.polyfit(xs, cc, 1)[0])),
        "abs_quad": float(abs(np.polyfit(xs, cc, 2)[0])),
        "centroid_mean_row": float(cc.mean() + 0.5),
    }


def analyze_bscan(native_u8, env_native):
    t = image_tensor(native_u8)
    rib = ribbon_with_intermediates(t)
    prod = _PROD_FN(t)
    if not torch.equal(prod, rib["grid"]):
        raise AssertionError("copy of CENTROID function diverged from production extract")
    normed = (t - IMAGENET_MEAN) / IMAGENET_STD
    row = centroid_descriptors(rib)
    row["imagenet_equal"] = bool(torch.equal(prod, _PROD_FN(normed)))
    row["affine_equal"] = bool(torch.equal(prod, _PROD_FN(t * 0.25 + 0.1)))
    snr, floored = snr_proxy(native_u8)
    row["bscan_snr_db"] = snr
    row["bscan_sd_floored"] = floored
    row["bscan_mean_intensity"] = float(native_u8.mean())
    if env_native is None:
        return row

    R = rib["grid"].numpy() > 0
    env_img = Image.fromarray(env_native.astype(np.uint8) * 255, mode="L").resize(
        (SIZE, SIZE), Image.NEAREST)
    env_px = np.asarray(env_img) > 127
    occ = patch_occupancy(env_px)
    E = occ >= OCC_T
    inter, union = int((R & E).sum()), int((R | E).sum())
    row.update({
        "env_area_native": float(env_native.mean()),
        "env_empty": bool(not env_native.any()),
        "env_valid_center": occupancy_is_valid(occ),
        "env_cells": int(E.sum()),
        "iou": inter / union if union else 0.0,
        "ribbon_zero_env_frac": float((occ[R] == 0).mean()),
        "ribbon_outside_frac": float((occ[R] < OCC_T).mean()),
        "ribbon_purity": float(occ[R].mean()),
        "env_covered": inter / int(E.sum()) if E.any() else np.nan,
    })
    Ec = E[:, X0:X1]
    row["env_covered_central"] = (int((R[:, X0:X1] & Ec).sum()) / int(Ec.sum())
                                  if Ec.any() else np.nan)
    rows_c = np.arange(GRID) + 0.5
    offsets, n_miss, n_noenv, n_out = [], 0, 0, 0
    for j, x in enumerate(range(X0, X1)):
        col = occ[:, x]
        mass = float(col.sum())
        if mass <= 0:
            n_noenv += 1
            continue
        top = int(rib["tops"][j])
        miss = float(col[top:top + BAND_H].sum()) == 0.0
        off = float((col * rows_c).sum() / mass - (top + BAND_H / 2.0))
        offsets.append(off)
        n_miss += int(miss)
        n_out += int(abs(off) > BAND_H / 2.0 or miss)
    offsets = np.array(offsets)
    row.update({
        "col_miss": n_miss, "col_noenv": n_noenv, "col_offset_out": n_out,
        "median_abs_offset": float(np.median(np.abs(offsets))) if offsets.size else np.nan,
        "max_abs_offset": float(np.abs(offsets).max()) if offsets.size else np.nan,
        "median_signed_offset": float(np.median(offsets)) if offsets.size else np.nan,
    })
    # envelope characterization (native 200x200)
    s8 = np.ones((3, 3), dtype=bool)
    row["n_comp_native"] = int(ndimage.label(env_native, structure=s8)[1])
    row["n_comp_grid"] = int(ndimage.label(E, structure=s8)[1])
    col_has = env_native.any(axis=0)
    c0 = int(round(NATIVE * 0.2))
    row["cols_without_env_frac"] = float(1.0 - col_has.mean())
    row["cols_without_env_central_frac"] = float(1.0 - col_has[c0:NATIVE - c0].mean())
    row["grid_cols_without_env"] = int((~E.any(axis=0)).sum())
    if col_has.any():
        thick = env_native.sum(axis=0)[col_has] / NATIVE_PX_PER_PATCH
        med = float(np.median(thick))
        outl = (thick > 2 * med) | (thick < 0.5 * med)
        starts = env_native[0].astype(int) + (np.diff(env_native.astype(np.int8), axis=0) == 1).sum(0)
        gthick = E.sum(axis=0)
        gthick = gthick[gthick > 0]
        row.update({
            "thick_median": med, "thick_p10": float(np.percentile(thick, 10)),
            "thick_p90": float(np.percentile(thick, 90)), "thick_max": float(thick.max()),
            "thick_outlier_frac": float(outl.mean()),
            "grid_thick_median": float(np.median(gthick)) if gthick.size else 0.0,
            "grid_thick_max": int(gthick.max()) if gthick.size else 0,
            "multi_run_frac": float((starts[col_has] > 1).mean()),
            "edge_touch_frac": float((env_native[0] | env_native[-1])[col_has].mean()),
            "env_centroid_row": float((np.nonzero(env_native)[0].mean() + 0.5)
                                      / NATIVE_PX_PER_PATCH),
        })
    return row


# ----------------------------------------------------------------------------- volumes

class SliceReader:
    """Seek/read access to the flat (N,100,200,200) uint8 slice cache.

    Plain reads instead of np.memmap: mapping the 24 GB file in many worker
    processes hit the Windows commit limit (WinError 1455).
    """

    def __init__(self, split):
        self.path = FV / "slice_cache" / split / "slice_cache.u8"
        self.n = (self.path.stat().st_size) // (100 * NATIVE * NATIVE)

    def __getitem__(self, key):
        v, ks = key
        scalar = np.isscalar(ks)
        ks = np.atleast_1d(ks)
        out = np.empty((len(ks), NATIVE, NATIVE), dtype=np.uint8)
        with open(self.path, "rb", buffering=0) as f:
            for j, k in enumerate(ks):
                f.seek((int(v) * 100 + int(k)) * NATIVE * NATIVE)
                buf = f.read(NATIVE * NATIVE)
                assert len(buf) == NATIVE * NATIVE
                out[j] = np.frombuffer(buf, dtype=np.uint8).reshape(NATIVE, NATIVE)
        return out[0] if scalar else out


def open_cache(split):
    meta = json.loads((FV / "slice_cache" / split / "slice_cache.json").read_text())
    assert meta["num_slices"] == 100 and meta["height"] == NATIVE and meta["width"] == NATIVE
    assert np.array_equal(np.array(meta["slice_indices"]), SLICE_INDICES)
    reader = SliceReader(split)
    assert reader.n == len(meta["volumes"])
    return meta["volumes"], reader


def case_hash(name):
    return hashlib.sha256(Path(name).stem.encode()).hexdigest()[:16]


def volume_job(args):
    """Analyze one volume's 10 B-scans; returns list of per-B-scan dicts."""
    split, vol_idx, name, with_env = args
    _, mm = open_cache(split)
    block = np.asarray(mm[vol_idx, K_POS])
    env = guide_meta = qc = None
    if with_env:
        with np.load(FV / "mirage_guides" / split / name, allow_pickle=False) as z:
            assert str(z["source_filename"].item()) == name
            assert int(z["schema_version"]) == 1
            assert str(z["params_fingerprint"].item()) == EXPECTED_FINGERPRINT
            assert np.array_equal(z["slice_indices"], SLICE_INDICES)
            packed = z["packed_envelopes"][K_POS]
            guide_meta = {"valid": z["valid"][K_POS], "area_frac": z["area_frac"][K_POS],
                          "span_frac": z["span_frac"][K_POS], "glaucoma": int(z["glaucoma"])}
        env = np.unpackbits(packed, axis=1, count=NATIVE * NATIVE).reshape(-1, NATIVE, NATIVE) > 0
        with np.load(FV / "mirage_masks" / split / name, allow_pickle=False) as z:
            assert str(z["source_filename"].item()) == name
            assert np.array_equal(z["slice_indices"], SLICE_INDICES)
            qc = {k: z[k][K_POS] for k in ("mean_confidence", "mean_entropy",
                                            "topology_violation_fraction",
                                            "all_classes_present", "constant_input")}
    rows = []
    for j, k in enumerate(K_POS):
        r = analyze_bscan(block[j], None if env is None else env[j])
        r.update({"split": split, "case": case_hash(name), "vol_idx": int(vol_idx),
                  "pos": int(k), "bscan_index": int(SLICE_INDICES[k])})
        if with_env:
            r["env_valid_stored"] = bool(guide_meta["valid"][j])
            r["env_area_stored"] = float(guide_meta["area_frac"][j])
            r["env_span_stored"] = float(guide_meta["span_frac"][j])
            r["guide_glaucoma"] = guide_meta["glaucoma"]
            for key, val in qc.items():
                r["mirage_" + key] = val[j].item()
        rows.append(r)
    return rows


def proxy_job(args):
    split, start, stop = args
    _, mm = open_cache(split)
    out = []
    for v in range(start, stop):
        block = np.asarray(mm[v, K_POS])
        snrs, floored = zip(*(snr_proxy(b) for b in block))
        out.append((v, float(np.median(snrs)), float(block.mean()), int(sum(floored))))
    return out


def worker_init(segment):
    torch.set_num_threads(1)
    init_production(segment)


def training_running():
    try:
        import psutil
    except ImportError:
        return False
    for p in psutil.process_iter(["name"]):
        try:
            # Read cmdline only for python processes; never touch other processes.
            if "python" in (p.info["name"] or "").lower() and any(
                    "train_patch" in str(c) for c in (p.cmdline() or [])):
                return True
        except Exception:
            continue
    return False


# ----------------------------------------------------------------------------- statistics

def add_flags(df):
    df = df.copy()
    df["ref_available"] = (~df["env_empty"]) & df["env_valid_stored"] & df["env_valid_center"]
    df["F1"] = (df["iou"] < 0.20) | (df["ribbon_zero_env_frac"] > 0.50)
    df["F2"] = df["col_offset_out"] >= 3
    df["F3"] = (df["iou"] < 0.20) | (df["ribbon_outside_frac"] > 0.50)
    nonempty = ~df["env_empty"]
    df["env_invalid"] = ~(df["env_valid_stored"] & df["env_valid_center"])
    df["fragmented"] = nonempty & (df["n_comp_native"] >= 3)
    df["thickness_irregular"] = nonempty & (df["thick_outlier_frac"] > 0.20)
    df["position_implausible"] = nonempty & (df["median_abs_offset"] > BAND_H / 2.0)
    df["env_any_flag"] = (df["env_empty"] | df["env_invalid"] | df["fragmented"]
                          | df["thickness_irregular"] | df["position_implausible"])
    return df


# flag -> eligibility column (None = all B-scans)
FLAG_ELIG = {
    "F1": "ref_available", "F2": "ref_available", "F3": "ref_available",
    "env_empty": None, "env_invalid": None, "fragmented": "nonempty",
    "thickness_irregular": "nonempty", "position_implausible": "nonempty",
    "env_any_flag": None,
}


def boot_weights(strata, rng, B):
    """Volume-cluster bootstrap counts, resampling volumes within strata."""
    n = len(strata)
    W = np.zeros((B, n), dtype=np.float32)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        draws = rng.integers(0, idx.size, size=(B, idx.size))
        for b in range(B):
            W[b, idx] += np.bincount(draws[b], minlength=idx.size)
    return W


def rate_table(df, vol, W, pop_share=None):
    """Cluster-bootstrap rates for every flag x group cell."""
    df = df.assign(nonempty=~df["env_empty"], all=True)
    vidx = df["vol_pos"].to_numpy()
    nv = len(vol)
    groups = {"overall": np.ones(len(df), bool)}
    for lab in (0, 1):
        groups["label=%d" % lab] = (df["label"] == lab).to_numpy()
    for t in range(3):
        groups["snr_t%d" % t] = (df["snr_tertile"] == t).to_numpy()
        groups["meanint_t%d" % t] = (df["meanint_tertile"] == t).to_numpy()
        groups["range_t%d" % t] = (df["range_tertile"] == t).to_numpy()
    for lab in (0, 1):
        for t in range(3):
            groups["label=%d,snr_t%d" % (lab, t)] = ((df["label"] == lab)
                                                     & (df["snr_tertile"] == t)).to_numpy()
    out = {}
    for flag, elig in FLAG_ELIG.items():
        e = df[elig].to_numpy() if elig else np.ones(len(df), bool)
        f = df[flag].to_numpy() & e
        res = {}
        for g, m in groups.items():
            num = np.bincount(vidx, weights=(f & m).astype(float), minlength=nv)
            den = np.bincount(vidx, weights=(e & m).astype(float), minlength=nv)
            if den.sum() == 0:
                continue
            bn, bd = W @ num, W @ den
            with np.errstate(invalid="ignore", divide="ignore"):
                br = bn / bd
            res[g] = {"rate": float(num.sum() / den.sum()), "k": int(num.sum()),
                      "n": int(den.sum()),
                      "ci95": np.nanpercentile(br, [2.5, 97.5]).tolist(), "_boot": br}
        if pop_share is not None:
            wr, wb = 0.0, 0.0
            for (lab, t), share in pop_share.items():
                cell = res["label=%d,snr_t%d" % (lab, t)]
                wr += share * cell["rate"]
                wb = wb + share * cell["_boot"]
            res["overall_popweighted"] = {"rate": wr,
                                          "ci95": np.nanpercentile(wb, [2.5, 97.5]).tolist()}
        for name, a, b in (("diff_glaucoma_minus_normal", "label=1", "label=0"),
                           ("diff_lowsnr_minus_highsnr", "snr_t0", "snr_t2"),
                           ("diff_highrange_minus_lowrange", "range_t2", "range_t0")):
            if a in res and b in res:
                d = res[a]["_boot"] - res[b]["_boot"]
                res[name] = {"diff": res[a]["rate"] - res[b]["rate"],
                             "ci95": np.nanpercentile(d, [2.5, 97.5]).tolist()}
        for v in res.values():
            v.pop("_boot", None)
        out[flag] = res
    # volume-level F1
    ok = df["ref_available"].to_numpy()
    fails = np.bincount(vidx, weights=(df["F1"].to_numpy() & ok).astype(float), minlength=nv)
    has = np.bincount(vidx, weights=ok.astype(float), minlength=nv) > 0
    vol_res = {}
    for thr in (1, 3):
        ind = ((fails >= thr) & has).astype(float)
        den = has.astype(float)
        br = (W @ ind) / (W @ den)
        vol_res["volumes_with_ge%d_F1" % thr] = {
            "rate": float(ind.sum() / den.sum()), "k": int(ind.sum()), "n": int(den.sum()),
            "ci95": np.percentile(br, [2.5, 97.5]).tolist()}
    out["volume_level"] = vol_res
    return out


CONT = ["iou", "ribbon_zero_env_frac", "ribbon_outside_frac", "ribbon_purity", "env_covered",
        "env_covered_central", "median_abs_offset", "median_signed_offset", "max_abs_offset",
        "col_miss", "col_noenv", "env_area_native", "n_comp_native", "n_comp_grid",
        "cols_without_env_frac", "cols_without_env_central_frac", "grid_cols_without_env",
        "thick_median", "thick_p10", "thick_p90", "thick_max", "grid_thick_median",
        "grid_thick_max", "thick_outlier_frac", "multi_run_frac", "edge_touch_frac",
        "centroid_range", "abs_slope", "abs_quad", "clamped_cols", "zero_mass_cols_all",
        "bscan_snr_db", "bscan_mean_intensity", "mirage_mean_confidence",
        "mirage_topology_violation_fraction"]


def describe(series):
    x = series.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return None
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}


def continuous_summary(df, cols=CONT):
    res = {}
    for c in cols:
        if c not in df:
            continue
        ref_only = c in ("iou", "ribbon_zero_env_frac", "ribbon_outside_frac", "ribbon_purity",
                         "env_covered", "env_covered_central", "median_abs_offset",
                         "median_signed_offset", "max_abs_offset", "col_miss", "col_noenv")
        d = df[df["ref_available"]] if ref_only and "ref_available" in df else df
        entry = {"all": describe(d[c])}
        for lab in (0, 1):
            entry["label=%d" % lab] = describe(d.loc[d["label"] == lab, c])
        for t in range(3):
            entry["snr_t%d" % t] = describe(d.loc[d["snr_tertile"] == t, c])
        res[c] = entry
    return res


def n_comp_distribution(df):
    d = df[~df["env_empty"]]
    bins = {"1": int((d["n_comp_native"] == 1).sum()), "2": int((d["n_comp_native"] == 2).sum()),
            ">=3": int((d["n_comp_native"] >= 3).sum())}
    return {"n_nonempty": int(len(d)), "counts": bins, "env_empty": int(df["env_empty"].sum())}


# ----------------------------------------------------------------------------- MIRAGE context

def mirage_context():
    ctx = {}
    log = MIRAGE_OUT / "log.txt"
    lines = log.read_text(encoding="utf-8").splitlines()
    best, keys = None, set()
    for i, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.update(rec)
        v = rec.get("val/mean_iou")
        if v is not None and (best is None or v > best[1]):
            best = (i, v, rec.get("epoch"), rec.get("val/pixel_accuracy"))
    ctx["training_log"] = {
        "path": str(log), "n_lines": len(lines),
        "best_val_mean_iou_percent": best[1], "best_epoch": best[2], "line": best[0],
        "val_pixel_accuracy_at_best": best[3],
        "dice_keys_in_log": sorted(k for k in keys if "dice" in k.lower()),
        "note": "No validation Dice is logged; val/mean_iou is the logged selection metric."}
    ev = MIRAGE_OUT / "evaluation-summary.json"
    ev_lines = ev.read_text(encoding="utf-8").splitlines()
    evj = json.loads("\n".join(ev_lines))
    def line_of(key):
        return next(i for i, l in enumerate(ev_lines, 1) if '"%s"' % key in l)
    ctx["goals_test_evaluation"] = {
        "path": str(ev),
        **{k: {"value": evj[k], "line": line_of(k)} for k in (
            "foreground_mean_dice", "all_class_mean_dice", "mean_iou", "mean_hd95",
            "paper_goals_dice")},
        "caveat": ("GOALS held-out test (30 images) aggregated by the official run_seg_eval "
                   "filename grouping (results.csv IDs 007-010); A4 04_eval_analysis.md s6.2. "
                   "In-domain GOALS only; not a FairVision Dice.")}
    man = FV / "manifests" / "mirage-preprocess-run.json"
    man_lines = man.read_text(encoding="utf-8").splitlines()
    ctx["guide_producer"] = {
        "manifest": str(man),
        "checkpoint_sha256_line": next(i for i, l in enumerate(man_lines, 1)
                                       if "mirage_checkpoint_sha256" in l),
        "checkpoint_sha256": json.loads("\n".join(man_lines))["mirage_checkpoint_sha256"],
        "preprocessing": json.loads("\n".join(man_lines))["preprocessing"]}
    rep = REPO / "archive" / "scripts_oneoff_2026-08"
    ctx["repo_mentions"] = [
        {"path": str(rep / "score_goals_merged.py"), "line": 7,
         "text": "published baseline number (all-class Dice 0.9426 / foreground 0.9251) is a FOUR-class score"},
        {"path": str(rep / "seg_run_status.py"), "line": 24, "text": "BASELINE_BEST_VAL_MIOU = 90.64"}]
    return ctx


# ----------------------------------------------------------------------------- main

def chunk_job(jobs):
    rows = []
    for j in jobs:
        rows.extend(volume_job(j))
    return rows


def run_volumes(jobs, workers, segment, log, chunk=25):
    rows = []
    t0 = time.perf_counter()
    if workers <= 1:
        for i, j in enumerate(jobs):
            rows.extend(volume_job(j))
            if (i + 1) % 100 == 0:
                log("  %d/%d volumes, %.1fs" % (i + 1, len(jobs), time.perf_counter() - t0))
    else:
        chunks = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
        done = 0
        with ProcessPoolExecutor(workers, initializer=worker_init, initargs=(segment,)) as ex:
            for r in ex.map(chunk_job, chunks, chunksize=1):
                rows.extend(r)
                done += 1
                if done % 20 == 0:
                    log("  %d/%d chunks, %.1fs" % (done, len(chunks), time.perf_counter() - t0))
    return pd.DataFrame(rows), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-census", action="store_true")
    ap.add_argument("--census-only", action="store_true",
                    help="run only the census stage using the saved primary summary")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log_lines = []
    log_path = OUT / ("run_log_census.txt" if args.census_only else "run_log.txt")
    log_path.write_text("", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    t_start = time.perf_counter()
    train_busy = training_running()
    workers = 2 if train_busy else args.workers
    log("I6 guide quality start %s; python %s; numpy %s; torch %s; scipy-ndimage; pandas %s"
        % (time.strftime("%Y-%m-%dT%H:%M:%S"), platform.python_version(), np.__version__,
           torch.__version__, pd.__version__))
    plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    log("plan sha256 %s; training_running=%s; workers=%d" % (plan_sha, train_busy, workers))
    segment, func_info = extract_function_sources()
    init_production(segment)
    log("CENTROID function provenance: %s" % json.dumps(func_info))
    log("ribbon geometry: band_h=%d x_keep=%d cols %d..%d" % (BAND_H, X_KEEP, X0, X1 - 1))

    # metadata labels
    meta = pd.read_csv(FV / "metadata" / "data_summary_glaucoma.csv")
    label_of = dict(zip(meta["filename"], (meta["glaucoma"] == "yes").astype(int)))
    use_of = dict(zip(meta["filename"], meta["use"]))

    # proxy pass over all Training + Validation volumes
    splits = {}
    proxy_rows = []
    t0 = time.perf_counter()
    proxy_cache = OUT / "_proxy_cache.parquet"
    for split in ("Training", "Validation"):
        names, _ = open_cache(split)
        splits[split] = names
        exp_use = {"Training": "training", "Validation": "validation"}[split]
        assert all(use_of[n] == exp_use for n in names), split
    if proxy_cache.exists():
        proxy = pd.read_parquet(proxy_cache)
        log("proxy pass: reused %s (%d volumes)" % (proxy_cache.name, len(proxy)))
    else:
        for split in ("Training", "Validation"):
            names = splits[split]
            chunks = [(split, s, min(s + 50, len(names))) for s in range(0, len(names), 50)]
            with ProcessPoolExecutor(workers) as ex:
                for res in ex.map(proxy_job, chunks):
                    for v, snr, mint, nfl in res:
                        proxy_rows.append({"split": split, "vol_idx": v, "name": names[v],
                                           "case": case_hash(names[v]),
                                           "label": label_of[names[v]],
                                           "snr_db": snr, "mean_intensity": mint,
                                           "n_sd_floored": nfl})
        proxy = pd.DataFrame(proxy_rows)
        proxy.to_parquet(proxy_cache)
    t_proxy = time.perf_counter() - t0
    log("proxy pass: %d volumes in %.1fs" % (len(proxy), t_proxy))
    tr = proxy[proxy["split"] == "Training"]
    snr_cut = np.percentile(tr["snr_db"], [100 / 3, 200 / 3])
    mi_cut = np.percentile(tr["mean_intensity"], [100 / 3, 200 / 3])
    proxy["snr_tertile"] = np.digitize(proxy["snr_db"], snr_cut)
    proxy["meanint_tertile"] = np.digitize(proxy["mean_intensity"], mi_cut)
    log("SNR cut-points %s; mean-intensity cut-points %s" % (snr_cut.tolist(), mi_cut.tolist()))

    # stratified samples
    rng = np.random.default_rng(SAMPLE_SEED)
    tr = proxy[proxy["split"] == "Training"]
    pick = []
    for lab in (0, 1):
        for t in range(3):
            idx = np.sort(tr.index[(tr["label"] == lab) & (tr["snr_tertile"] == t)].to_numpy())
            pick.extend(rng.choice(idx, PER_STRATUM, replace=False).tolist())
    sample = proxy.loc[pick].reset_index(drop=True)
    vrng = np.random.default_rng(VAL_SEED)
    va = proxy[proxy["split"] == "Validation"]
    vpick = []
    for lab in (0, 1):
        for t in range(3):
            idx = np.sort(va.index[(va["label"] == lab) & (va["snr_tertile"] == t)].to_numpy())
            vpick.extend(vrng.choice(idx, min(VAL_PER_STRATUM, idx.size), replace=False).tolist())
    vsample = proxy.loc[vpick].reset_index(drop=True)
    pop = tr.groupby(["label", "snr_tertile"]).size()
    pop_share = {k: float(v / len(tr)) for k, v in pop.items()}
    log("Training stratum sizes %s" % {str(k): int(v) for k, v in pop.items()})

    # slice-cache bit-equality spot check (first 3 sampled Training + 1 Validation volumes)
    spot = []
    for split, df_ in (("Training", sample.head(3)), ("Validation", vsample.head(1))):
        _, mm = open_cache(split)
        for _, r in df_.iterrows():
            with np.load(FV / "data" / split / r["name"], allow_pickle=False) as z:
                vol = z["oct_bscans"]
                eq = all(np.array_equal(vol[SLICE_INDICES[k]], mm[r["vol_idx"], k]) for k in K_POS)
                assert int(z["glaucoma"]) == r["label"]
            spot.append(eq)
    assert all(spot), "slice cache differs from npz"
    log("slice-cache spot check equal for %d volumes" % len(spot))

    # primary analysis
    if args.census_only:
        summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        assert summary["census_decision"]["run"], "census rule was not met in the primary run"
        assert summary["plan_sha256"] == plan_sha
        range_cut = np.array(summary["design"]["centroid_range_cutpoints_patches"])
        census, t_cen = run_census(proxy, range_cut, workers, segment, log)
        summary["census"] = census
        summary["timing_seconds"]["census_stage"] = t_cen
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float),
                                          encoding="utf-8")
        write_appendix(summary)
        proxy_cache.unlink(missing_ok=True)
        log("census stage done in %.1fs" % (time.perf_counter() - t_start))
        return

    jobs = [("Training", int(r.vol_idx), r.name, True) for r in sample.itertuples()]
    prim, t_prim = run_volumes(jobs, workers, segment, log)
    log("primary: %d B-scans in %.1fs" % (len(prim), t_prim))
    census_ok = (not args.no_census) and (t_prim * 20 < 45 * 60) and not training_running()
    log("census decision (runtime rule): primary %.1fs x20 = %.1f min -> %s"
        % (t_prim, t_prim * 20 / 60, census_ok))

    prim = attach(prim, sample)
    assert (prim["guide_glaucoma"] == prim["label"]).all()
    prim = add_flags(prim)
    range_cut = np.percentile(prim["centroid_range"], [100 / 3, 200 / 3])
    prim["range_tertile"] = np.digitize(prim["centroid_range"], range_cut)
    vol = prim.groupby("vol_pos").first()
    strata = (vol["label"] * 3 + vol["snr_tertile"]).to_numpy()
    W = boot_weights(strata, np.random.default_rng(BOOT_SEED), BOOT_B)
    rates = rate_table(prim, vol, W, pop_share)

    # validation CENTROID-only supplement
    vjobs = [("Validation", int(r.vol_idx), r.name, False) for r in vsample.itertuples()]
    vrows, t_val = run_volumes(vjobs, workers, segment, log)
    vrows = attach(vrows, vsample)
    log("validation supplement: %d B-scans in %.1fs" % (len(vrows), t_val))
    vcols = ["centroid_range", "abs_slope", "abs_quad", "clamped_cols", "zero_mass_cols_all",
             "centroid_mean_row", "bscan_snr_db"]
    val_summary = {
        "n_volumes": int(vrows["vol_idx"].nunique()), "n_bscans": int(len(vrows)),
        "training_sample": {c: describe(prim[c]) for c in vcols},
        "validation_sample": {c: describe(vrows[c]) for c in vcols},
        "flat_images": {"training": int(prim["flat_image"].sum()),
                        "validation": int(vrows["flat_image"].sum())},
        "bscans_with_clamped_col": {"training": float((prim["clamped_cols"] > 0).mean()),
                                    "validation": float((vrows["clamped_cols"] > 0).mean())},
        "bscans_with_zero_mass_col": {"training": float((prim["zero_mass_cols_all"] > 0).mean()),
                                      "validation": float((vrows["zero_mass_cols_all"] > 0).mean())},
    }

    checks = {
        "ribbon_cells_always_70": bool((prim["ribbon_cells"] == BAND_H * X_KEEP).all()
                                       and (vrows["ribbon_cells"] == BAND_H * X_KEEP).all()),
        "imagenet_mismatch": int((~prim["imagenet_equal"]).sum() + (~vrows["imagenet_equal"]).sum()),
        "affine_mismatch": int((~prim["affine_equal"]).sum() + (~vrows["affine_equal"]).sum()),
        "copy_equals_production": True,
        "bscan_sd_floored": int(prim["bscan_sd_floored"].sum()),
        "volumes_with_sd_floored_any_split": int((proxy["n_sd_floored"] > 0).sum()),
    }

    summary = {
        "plan_sha256": plan_sha, "function_provenance": func_info,
        "design": {"frame": "Training (envelopes exist only for Training)", "per_stratum": PER_STRATUM,
                   "n_volumes": int(prim["vol_idx"].nunique()), "n_bscans": int(len(prim)),
                   "bscan_indices": SLICE_INDICES[K_POS].tolist(), "sample_seed": SAMPLE_SEED,
                   "boot_seed": BOOT_SEED, "boot_B": BOOT_B, "snr_cutpoints_db": snr_cut.tolist(),
                   "meanint_cutpoints": mi_cut.tolist(),
                   "centroid_range_cutpoints_patches": range_cut.tolist(),
                   "population_stratum_share": {"label=%d,snr_t%d" % k: v for k, v in pop_share.items()},
                   "ribbon": {"band_h": BAND_H, "x_keep": X_KEEP, "cols": [X0, X1 - 1],
                              "region_frac": .28, "lateral_frac": .6, "row_offset": 0.0,
                              "min_band_rows": 3},
                   "envelope": {"occupancy_threshold": OCC_T, "resize": "nearest 200->256",
                                "fingerprint": EXPECTED_FINGERPRINT},
                   "view": "full B-scan PIL bilinear 200->256, no crop (probe/center view)"},
        "reference_available": {"k": int(prim["ref_available"].sum()), "n": int(len(prim))},
        "rates": rates, "continuous": continuous_summary(prim),
        "n_components": n_comp_distribution(prim),
        "validation_centroid_supplement": val_summary, "checks": checks,
        "mirage_context": mirage_context(),
        "timing_seconds": {"proxy": t_proxy, "primary": t_prim, "validation": t_val},
        "census_decision": {"run": bool(census_ok), "primary_seconds": t_prim,
                            "rule": "primary seconds x20 < 45 min and no train_patch.py running"},
    }

    # write primary outputs first
    drop = ["vol_idx", "vol_pos"]
    prim.drop(columns=drop).to_csv(OUT / "per_bscan_metrics.csv.gz", index=False,
                                   compression="gzip", float_format="%.5g")
    vrows.drop(columns=drop).to_csv(OUT / "validation_centroid_descriptors.csv.gz", index=False,
                                    compression="gzip", float_format="%.5g")
    proxy["sampled_primary"] = proxy.index.isin(pick)
    proxy["sampled_validation"] = proxy.index.isin(vpick)
    proxy.drop(columns=["name", "vol_idx"]).to_csv(OUT / "volume_proxy.csv.gz", index=False,
                                                   compression="gzip", float_format="%.6g")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    write_appendix(summary)
    make_plots(prim)
    log("primary outputs written (%.1fs)" % (time.perf_counter() - t_start))

    # census sensitivity
    if census_ok:
        census, t_cen = run_census(proxy, range_cut, workers, segment, log)
        summary["census"] = census
        summary["timing_seconds"]["census_stage"] = t_cen
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float),
                                          encoding="utf-8")
        write_appendix(summary)
    proxy_cache.unlink(missing_ok=True)
    log("done in %.1fs" % (time.perf_counter() - t_start))


def attach(df, ref):
    keys = ref[["vol_idx", "label", "snr_db", "mean_intensity", "snr_tertile",
                "meanint_tertile"]]
    df = df.merge(keys, on="vol_idx", how="left", validate="many_to_one")
    df["vol_pos"] = pd.factorize(df["vol_idx"])[0]
    return df


def run_census(proxy, range_cut, workers, segment, log):
    t0 = time.perf_counter()
    trp = proxy[proxy["split"] == "Training"]
    cjobs = [("Training", int(r.vol_idx), r.name, True) for r in trp.itertuples()]
    cen, t_cen = run_volumes(cjobs, max(workers, 1), segment, log)
    log("census: %d B-scans in %.1fs" % (len(cen), t_cen))
    cen = attach(cen, trp)
    assert (cen["guide_glaucoma"] == cen["label"]).all()
    cen = add_flags(cen)
    cen["range_tertile"] = np.digitize(cen["centroid_range"], range_cut)
    cvol = cen.groupby("vol_pos").first()
    cstr = (cvol["label"] * 3 + cvol["snr_tertile"]).to_numpy()
    CW = boot_weights(cstr, np.random.default_rng(BOOT_SEED + 1), 500)
    census = {"n_volumes": int(cen["vol_idx"].nunique()), "n_bscans": int(len(cen)),
              "boot_B": 500,
              "reference_available": {"k": int(cen["ref_available"].sum()), "n": int(len(cen))},
              "rates": rate_table(cen, cvol, CW, None),
              "continuous": continuous_summary(cen, ["iou", "ribbon_zero_env_frac",
                                                      "ribbon_purity", "env_covered",
                                                      "median_abs_offset", "thick_median",
                                                      "n_comp_native", "cols_without_env_frac"]),
              "n_components": n_comp_distribution(cen),
              "imagenet_mismatch": int((~cen["imagenet_equal"]).sum()),
              "affine_mismatch": int((~cen["affine_equal"]).sum()),
              "ribbon_cells_always_70": bool((cen["ribbon_cells"] == BAND_H * X_KEEP).all()),
              "seconds": t_cen}
    cen.drop(columns=["vol_idx", "vol_pos"]).to_csv(
        OUT / "census_per_bscan_metrics.csv.gz", index=False, compression="gzip",
        float_format="%.5g")
    return census, time.perf_counter() - t0


def _fmt(cell, pct=True):
    if cell is None:
        return "n/a"
    lo, hi = cell["ci95"]
    s = 100 if pct else 1
    return "%.1f%% [%.1f, %.1f]" % (cell["rate"] * s, lo * s, hi * s)


def write_appendix(summary):
    rates = summary["rates"]
    cols = [("overall_popweighted", "Overall (pop.-wt.)"), ("label=0", "Normal"),
            ("label=1", "Glaucoma"), ("snr_t0", "SNR low"), ("snr_t1", "SNR mid"),
            ("snr_t2", "SNR high")]
    names = {"F1": "CENTROID F1: IoU<.20 or >50% ribbon cells with zero envelope",
             "F2": "CENTROID F2: >=3/10 ribbon columns off-centre/missed",
             "F3": "CENTROID F3 (strict): IoU<.20 or >50% ribbon cells occ<.25",
             "env_empty": "Envelope empty", "env_invalid": "Envelope invalid (stored or centre view)",
             "fragmented": "Envelope >=3 components", "thickness_irregular":
             "Envelope thickness irregular (>20% outlier cols)",
             "position_implausible": "Envelope centroid outside ribbon (median |offset|>3.5)",
             "env_any_flag": "Any envelope flag"}
    md = ["| Rule | " + " | ".join(c[1] for c in cols) + " |",
          "|---|" + "---:|" * len(cols)]
    csv_rows = []
    for flag, label in names.items():
        r = rates[flag]
        cells = [_fmt(r.get(c[0])) if c[0] in r else "n/a" for c in cols]
        md.append("| %s | %s |" % (label, " | ".join(cells)))
        for c in cols:
            if c[0] in r:
                csv_rows.append({"rule": flag, "group": c[0], "rate": r[c[0]]["rate"],
                                 "ci_lo": r[c[0]]["ci95"][0], "ci_hi": r[c[0]]["ci95"][1],
                                 "k": r[c[0]].get("k"), "n": r[c[0]].get("n")})
    md.append("")
    md.append("| CENTROID rule by tilt/curvature (centroid-range tertile) | low | mid | high |")
    md.append("|---|---:|---:|---:|")
    for flag in ("F1", "F2", "F3"):
        r = rates[flag]
        md.append("| %s | %s |" % (flag, " | ".join(_fmt(r.get("range_t%d" % t)) for t in range(3))))
        for t in range(3):
            c = r["range_t%d" % t]
            csv_rows.append({"rule": flag, "group": "range_t%d" % t, "rate": c["rate"],
                             "ci_lo": c["ci95"][0], "ci_hi": c["ci95"][1], "k": c["k"], "n": c["n"]})
    cont = summary["continuous"]
    md.append("")
    md.append("| Continuous (median [P5, P95]) | All | Normal | Glaucoma |")
    md.append("|---|---:|---:|---:|")
    for c, lab in (("iou", "Ribbon-envelope IoU"), ("ribbon_zero_env_frac", "Ribbon cells with zero envelope"),
                   ("env_covered", "Envelope covered by ribbon"),
                   ("median_abs_offset", "Median |envelope centroid - ribbon centre| (patches)"),
                   ("thick_median", "Envelope median thickness (patches)"),
                   ("cols_without_env_frac", "Native columns without envelope"),
                   ("n_comp_native", "Envelope components (native)")):
        e = cont[c]
        md.append("| %s | %s |" % (lab, " | ".join(
            "%.3g [%.3g, %.3g]" % (e[g]["median"], e[g]["p05"], e[g]["p95"]) if e[g] else "n/a"
            for g in ("all", "label=0", "label=1"))))
    if "census" in summary:
        cr = summary["census"]["rates"]
        md.append("")
        md.append("| Census check (all %d Training volumes x 10 B-scans) | Overall | Normal | Glaucoma |"
                  % summary["census"]["n_volumes"])
        md.append("|---|---:|---:|---:|")
        for flag, label in names.items():
            md.append("| %s | %s |" % (label, " | ".join(
                _fmt(cr[flag].get(g)) for g in ("overall", "label=0", "label=1"))))
            for g in ("overall", "label=0", "label=1"):
                c = cr[flag][g]
                csv_rows.append({"rule": flag, "group": "census_" + g, "rate": c["rate"],
                                 "ci_lo": c["ci95"][0], "ci_hi": c["ci95"][1], "k": c["k"],
                                 "n": c["n"]})
    (OUT / "appendix_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(OUT / "appendix_table.csv", index=False)


def make_plots(prim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = prim[prim["ref_available"]]
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=150)
    bins = np.linspace(0, 1, 41)
    for lab, name, ls in ((0, "Normal", "-"), (1, "Glaucoma", "--")):
        ax.hist(d.loc[d["label"] == lab, "iou"], bins=bins, histtype="step", linestyle=ls,
                linewidth=1.4, label="%s (n=%d)" % (name, (d["label"] == lab).sum()))
    ax.axvline(0.20, color="k", linewidth=0.8)
    ax.set_xlabel("CENTROID ribbon vs MIRAGE envelope IoU (16x16 grid)")
    ax.set_ylabel("B-scans")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "hist_iou.png")
    plt.close(fig)
    d = prim[~prim["env_empty"]]
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=150)
    bins = np.linspace(0, 8, 41)
    for lab, name, ls in ((0, "Normal", "-"), (1, "Glaucoma", "--")):
        ax.hist(d.loc[d["label"] == lab, "thick_median"], bins=bins, histtype="step", linestyle=ls,
                linewidth=1.4, label=name)
    ax.axvline(BAND_H, color="k", linewidth=0.8)
    ax.set_xlabel("Envelope median column thickness (patches)")
    ax.set_ylabel("B-scans")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "hist_env_thickness.png")
    plt.close(fig)


def posthoc():
    """Exploratory, NOT pre-declared: decomposes F1 using the saved census CSV."""
    df = pd.read_csv(OUT / "census_per_bscan_metrics.csv.gz")
    d = df[df["ref_available"]].copy()
    res = {"label": "post hoc exploratory; not in ANALYSIS_PLAN.md; census Training B-scans",
           "n_ref_bscans": int(len(d))}
    iou_part = d["iou"] < 0.20
    zero_part = d["ribbon_zero_env_frac"] > 0.50
    res["F1_components"] = {"iou_lt_020_only": int((iou_part & ~zero_part).sum()),
                            "zero_gt_050_only": int((zero_part & ~iou_part).sum()),
                            "both": int((iou_part & zero_part).sum()),
                            "F1_total": int(d["F1"].sum())}
    tcut = np.percentile(d["thick_median"], [100 / 3, 200 / 3])
    d["thick_t"] = np.digitize(d["thick_median"], tcut)
    res["thickness_cutpoints_patches"] = tcut.tolist()
    res["F1_by_label_x_thickness_tertile"] = {
        "label=%d,thick_t%d" % (lab, t): {
            "rate": float(d.loc[(d["label"] == lab) & (d["thick_t"] == t), "F1"].mean()),
            "n": int(((d["label"] == lab) & (d["thick_t"] == t)).sum())}
        for lab in (0, 1) for t in range(3)}
    res["F2_by_label_x_thickness_tertile"] = {
        "label=%d,thick_t%d" % (lab, t): float(
            d.loc[(d["label"] == lab) & (d["thick_t"] == t), "F2"].mean())
        for lab in (0, 1) for t in range(3)}
    res["F1_by_bscan_index"] = {int(k): float(v) for k, v in d.groupby("bscan_index")["F1"].mean().items()}
    res["F2_by_bscan_index"] = {int(k): float(v) for k, v in d.groupby("bscan_index")["F2"].mean().items()}
    res["spearman"] = {
        "snr_vs_centroid_range": float(d["bscan_snr_db"].corr(d["centroid_range"], method="spearman")),
        "snr_vs_iou": float(d["bscan_snr_db"].corr(d["iou"], method="spearman")),
        "thickness_vs_ribbon_zero_frac": float(d["thick_median"].corr(d["ribbon_zero_env_frac"],
                                                                      method="spearman")),
        "mirage_confidence_vs_iou": float(d["mirage_mean_confidence"].corr(d["iou"], method="spearman")),
    }
    for name, m in (("F1", d["F1"]), ("not_F1", ~d["F1"])):
        sub = d[m]
        res["profile_" + name] = {
            "n": int(len(sub)),
            "median_signed_offset": float(sub["median_signed_offset"].median()),
            "frac_signed_offset_positive": float((sub["median_signed_offset"] > 0).mean()),
            "median_thick": float(sub["thick_median"].median()),
            "median_snr_db": float(sub["bscan_snr_db"].median()),
            "median_centroid_range": float(sub["centroid_range"].median()),
            "median_mirage_confidence": float(sub["mirage_mean_confidence"].median()),
            "median_iou": float(sub["iou"].median()),
        }
    (OUT / "exploratory_posthoc.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if "--posthoc" in sys.argv:
        posthoc()
    else:
        main()
