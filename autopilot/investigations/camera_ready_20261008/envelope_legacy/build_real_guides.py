"""Build real post-crop ENVELOPE guide grids with the production loader (CPU).

Uses HEAD's GuidedOCTSliceDataset + paired crop configured exactly as
src/train_patch.py configures it from configs/patch_mirage_envelope.yaml
(hard schema-1 guides, dilate 0, occupancy 0.25, slice cache).  Crops are
fixed by seeding; the saved grids are the shared input for both the golden
mask comparison and the fingerprint replay.

Usage: python build_real_guides.py N OUT.npz [selection_seed] [crop_seed]
"""
import os
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.transforms import make_paired_transforms  # noqa: E402

torch.set_num_threads(2)
N = int(sys.argv[1])
OUT = sys.argv[2]
SEL_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 20261008
CROP_SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 11

cfg = C.load_prod_yaml()
data_cfg, curr = cfg["data"], cfg["mask"]["curriculum"]
pt = make_paired_transforms(
    crop_size=data_cfg["crop_size"],
    crop_scale=tuple(data_cfg.get("crop_scale", (0.3, 1.0))),
    gaussian_blur=data_cfg.get("use_gaussian_blur", False),
    horizontal_flip=data_cfg.get("use_horizontal_flip", False),
    color_distortion=data_cfg.get("use_color_distortion", False),
    color_jitter=data_cfg.get("color_jitter_strength", 0.0),
)
ds = GuidedOCTSliceDataset(
    data_dir=os.path.join(data_cfg["data_dir"], "Training"),
    guide_dir=os.path.join(curr["mirage_guide_dir"], "Training"),
    num_slices=data_cfg.get("num_slices", 32),
    slice_size=data_cfg["crop_size"],
    transform=pt,
    patch_size=cfg["mask"]["patch_size"],
    dilate_patches=int(curr.get("mirage_dilate_patches", 1)),
    occupancy_threshold=float(curr.get("mirage_occupancy_threshold", 0.5)),
    slice_cache=os.path.join(data_cfg["slice_cache_dir"], "Training"),
)
rs = np.random.RandomState(SEL_SEED)
idx = rs.choice(len(ds), size=N, replace=False)
C.seed_all(CROP_SEED)
t0 = time.time()
guides, valid = [], []
for k, i in enumerate(idx):
    item = ds[int(i)]
    guides.append(item[1].numpy().astype(np.float32))
    valid.append(bool(item[2]))
    if (k + 1) % 640 == 0:
        print("  %d/%d  %.1fs" % (k + 1, N, time.time() - t0), flush=True)
G = np.stack(guides)
np.savez_compressed(
    OUT, guides=G, valid=np.array(valid), idx=idx,
    selection_seed=SEL_SEED, crop_seed=CROP_SEED, dataset_len=len(ds),
)
print("saved", OUT, G.shape, "valid", int(np.sum(valid)), "of", N,
      "volumes", len(ds.file_paths), "%.1fs" % (time.time() - t0))
