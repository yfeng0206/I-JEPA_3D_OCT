"""CPU-only. Re-analyse ALREADY-SAVED artifacts:
  D:\jepa_phase0\reports\anatomy_mask_cache\Test_s100_r256.npz   (packed 256-cell anatomy masks)
  D:\jepa_phase0\reports\patch_attribution\*_attrib.npz          (256-cell attribution maps)
No model forward, no GPU.
"""
import json
import numpy as np
from scipy import stats

CACHE = r"D:\jepa_phase0\reports\anatomy_mask_cache\Test_s100_r256.npz"
ATTR = r"D:\jepa_phase0\reports\patch_attribution\{}_ep50_attrib.npz"
ARMS = ["random", "oracle", "envelope", "blob"]

z = np.load(CACHE)
packed = z["mask"]                      # (V, S, 32) uint8, bit-packed 256 cells
V, S, _ = packed.shape
bits = np.unpackbits(packed, axis=2)    # (V, S, 256)
freq = bits.reshape(-1, 256).mean(0)    # per-cell P(anatomy) over V*S slices
grid = freq.reshape(16, 16)

out = {
    "source_mask_cache": CACHE,
    "n_volumes": int(V),
    "n_slices_per_volume": int(S),
    "n_slice_instances": int(V * S),
    "global_anatomy_rate": float(freq.mean()),
    "per_cell_anatomy_freq_grid_16x16": [[round(float(x), 5) for x in r] for r in grid],
    "row_anatomy_freq": [round(float(x), 5) for x in grid.mean(1)],
    "col_anatomy_freq": [round(float(x), 5) for x in grid.mean(0)],
}

# "always background" cell sets at several strictness levels
levels = [0.01, 0.05, 0.10]
BG = 0.01  # "near-always background" reference set
cellsets = {}
for lv in levels:
    m = freq <= lv
    cellsets[lv] = m
    out[f"n_cells_anatfreq_le_{lv}"] = int(m.sum())

arms = {}
for arm in ARMS:
    a = np.load(ATTR.format(arm))
    pm, pa = a["patch_mean"], a["patch_absmean"]
    tot = float(pa.sum())
    r_pear = stats.pearsonr(pa, freq)
    r_spear = stats.spearmanr(pa, freq)
    d = {
        "total_absmean_mass": tot,
        "pearson_absattr_vs_anatfreq": [float(r_pear[0]), float(r_pear[1])],
        "spearman_absattr_vs_anatfreq": [float(r_spear[0]), float(r_spear[1])],
        "mean_absattr_all_cells": float(pa.mean()),
    }
    for lv in levels:
        m = cellsets[lv]
        d[f"share_of_absmass_on_cells_anatfreq_le_{lv}"] = float(pa[m].sum() / tot)
        d[f"cellfrac_anatfreq_le_{lv}"] = float(m.mean())
        d[f"mean_absattr_on_cells_anatfreq_le_{lv}"] = float(pa[m].mean())
        # enrichment = mass share / cell share ; 1.0 means "exactly proportional"
        d[f"enrichment_anatfreq_le_{lv}"] = float((pa[m].sum() / tot) / m.mean())
    # tissue-ish reference: cells that are anatomy in >50% of slices
    hi = freq >= 0.5
    d["n_cells_anatfreq_ge_0.5"] = int(hi.sum())
    d["mean_absattr_on_cells_anatfreq_ge_0.5"] = float(pa[hi].mean())
    d["ratio_meanabs_alwaysbg_over_mostlyanat"] = float(pa[cellsets[BG]].mean() / pa[hi].mean())
    # spatial structure WITHIN the always-background cells: is |attr| positionally organised?
    m0 = cellsets[BG]
    rows = np.repeat(np.arange(16), 16)[m0]
    cols = np.tile(np.arange(16), 16)[m0]
    vals = pa[m0]
    d["within_alwaysbg_cv_of_absattr"] = float(vals.std() / vals.mean())
    d["within_alwaysbg_spearman_vs_row"] = [float(x) for x in stats.spearmanr(vals, rows)]
    d["within_alwaysbg_spearman_vs_col"] = [float(x) for x in stats.spearmanr(vals, cols)]
    # distance from the anatomy centre of mass (positional, content-free)
    rr, cc = np.mgrid[0:16, 0:16]
    com_r = float((rr.ravel() * freq).sum() / freq.sum())
    com_c = float((cc.ravel() * freq).sum() / freq.sum())
    dist = np.sqrt((rr.ravel() - com_r) ** 2 + (cc.ravel() - com_c) ** 2)
    d["anatomy_centre_of_mass_rc"] = [com_r, com_c]
    d["within_alwaysbg_spearman_absattr_vs_dist_to_anat_com"] = [
        float(x) for x in stats.spearmanr(vals, dist[m0])
    ]
    d["allcells_spearman_absattr_vs_dist_to_anat_com"] = [
        float(x) for x in stats.spearmanr(pa, dist)
    ]
    d["signed_patch_mean_sum"] = float(pm.sum())
    d["signed_patch_mean_on_alwaysbg_sum"] = float(pm[m0].sum())
    arms[arm] = d

out["arms"] = arms
p = r"C:\Users\Gary\Desktop\jepa\autopilot\bgsig\a1_position_attribution.json"
with open(p, "w") as f:
    json.dump(out, f, indent=2)

print("global anatomy rate: %.5f" % out["global_anatomy_rate"])
for lv in levels:
    print("cells with anat freq <= %.2f : %d / 256" % (lv, out[f"n_cells_anatfreq_le_{lv}"]))
print()
for arm in ARMS:
    d = arms[arm]
    print("== %s ==" % arm)
    print("  pearson |attr| vs anat freq : r=%.4f p=%.3g" % tuple(d["pearson_absattr_vs_anatfreq"]))
    print("  spearman |attr| vs anat freq: r=%.4f p=%.3g" % tuple(d["spearman_absattr_vs_anatfreq"]))
    for lv in levels:
        print("  anatfreq<=%.2f : cells %.3f  mass %.4f  enrichment %.3f"
              % (lv, d[f"cellfrac_anatfreq_le_{lv}"],
                 d[f"share_of_absmass_on_cells_anatfreq_le_{lv}"],
                 d[f"enrichment_anatfreq_le_{lv}"]))
    print("  mean|attr| always-bg / mostly-anat = %.4f" % d["ratio_meanabs_alwaysbg_over_mostlyanat"])
    print("  within always-bg CV of |attr| = %.4f" % d["within_alwaysbg_cv_of_absattr"])
    print("  within always-bg spearman vs dist-to-anat-COM: r=%.4f p=%.3g"
          % tuple(d["within_alwaysbg_spearman_absattr_vs_dist_to_anat_com"]))
print("\nwrote", p)
