"""Verify worker-parallel MIRAGE mask generation end to end.

Windows uses spawn for DataLoader workers, so this must live in a real module
(guarded by __main__) rather than being piped through stdin.
"""

import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset  # noqa: E402
from src.masks.curriculum import MirageMaskCollator  # noqa: E402
from src.transforms import make_paired_transforms  # noqa: E402

DATA = Path(r"D:\jepa_phase0\fairvision-glaucoma\data\Training")
GUIDES = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_guides\Training")


def main() -> None:
    names = [p.name for p in sorted(GUIDES.glob("data_*.npz"))[:60]]
    tmp = Path(tempfile.mkdtemp(dir=r"D:\jepa_phase0"))
    (tmp / "Training").mkdir()
    (tmp / "guides").mkdir()
    for name in names:
        os.link(DATA / name, tmp / "Training" / name)
        os.link(GUIDES / name, tmp / "guides" / name)

    try:
        dataset = GuidedOCTSliceDataset(
            data_dir=str(tmp / "Training"),
            guide_dir=str(tmp / "guides"),
            num_slices=100,
            slice_size=256,
            transform=make_paired_transforms(256, (0.3, 1.0)),
            patch_size=16,
            dilate_patches=1,
        )
        collator = MirageMaskCollator(
            input_size=(256, 256),
            patch_size=16,
            enc_mask_scale=(0.85, 1.0),
            pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5),
            nenc=1,
            npred=4,
            min_keep=10,
            allow_overlap=False,
            curriculum_cfg={
                "mode": "mirage_envelope",
                "T_warm": 25,
                "T_total": 30,
                "r_max": 1.0,
                "mirage_min_block_fill": 0.40,
                "mirage_min_retina_visible": 0.25,
            },
        )
        assert isinstance(pickle.loads(pickle.dumps(collator)), MirageMaskCollator)

        for workers in (0, 4):
            collator.set_epoch(30, 100)
            loader = DataLoader(
                dataset,
                batch_size=64,
                shuffle=True,
                num_workers=workers,
                collate_fn=collator,
                drop_last=True,
                pin_memory=True,
            )
            iterator = iter(loader)
            next(iterator)  # warm up workers / caches
            started = time.perf_counter()
            count = 8
            for _ in range(count):
                imgs, masks_enc, masks_pred, stats = next(iterator)
            elapsed = (time.perf_counter() - started) / count
            print(
                f"num_workers={workers}: {elapsed:.3f}s per 64-image microbatch "
                f"-> {elapsed * 8:.2f}s per 512 optimizer step, "
                f"{64 / elapsed:.0f} img/s"
            )
            print(
                f"    img={tuple(imgs.shape)} enc={tuple(masks_enc[0].shape)} "
                f"pred={tuple(masks_pred[0].shape)} accept={stats['accept_rate']:.2f} "
                f"fill={stats['mean_block_fill']:.3f} vis={stats['retina_visible']:.3f} "
                f"on_region={stats['target_on_region']:.3f}"
            )
            del iterator, loader

        # The ramp must reach the workers through the pickled collator.
        for epoch in (25, 27, 30):
            collator.set_epoch(epoch, 100)
            loader = DataLoader(
                dataset,
                batch_size=64,
                shuffle=True,
                num_workers=4,
                collate_fn=collator,
                drop_last=True,
            )
            _imgs, _enc, _pred, stats = next(iter(loader))
            print(
                f"    epoch {epoch}: guided={stats['guided_images']} "
                f"unbiased={stats['unbiased_by_ramp']} "
                f"on_region={stats['target_on_region']:.3f}"
            )
            del loader
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
