"""CPU production of MIRAGE placement guides for the external OCTDL crop.

The all-strategy figure needs the guides that ENVELOPE, ANATOMY-V2 and COVER
place targets on.  This script runs the two production guide pipelines on the
same licensed external crop used by Figure 1 (no FairVision pixels):

* ENVELOPE (hard guide): MIRAGE-Large, GOALS fine-tuned frozen ConvNeXt head,
  as in fairvision-glaucoma/scripts/preprocess_mirage_masks.py -- per-slice
  min-max, OpenCV INTER_LINEAR to 1024, softmax argmax, nearest-exact to the
  200x200 native grid -- then src.guides.mirage_envelope build_union +
  repair_union(DEFAULT_REPAIR), exactly as scripts/mirage_precompute_guides.py.
* ANATOMY-V2 / COVER (soft guide): MIRAGE-Base MergedV3 plus the structural
  adapter at the encoder tap, as in scripts/precompute_soft_guides.py --
  min-max, 512, softmax, P_inner/P_choroid, bilinear to 200x200, uint8.

Differences from the cached training guides, recorded in the output: CPU fp32
instead of CUDA fp16 autocast, and a 1280x1280 crop of a published figure
instead of a 200x200 FairVision B-scan.  The checkpoints are verified against
the hashes recorded by the training guide caches.  The MIRAGE models were not
trained on OCTDL, so the result is an illustration of the production pipeline,
not a validated segmentation.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from src.guides.mirage_envelope import (  # noqa: E402
    DEFAULT_REPAIR, build_union, params_fingerprint, repair_union)

STAGING = ROOT / "autopilot" / "investigations" / "camera_ready_20261008" / "figures_staging"
SOURCE = HERE / "octdl_figure1_original.png"
CROP = (3200, 245, 4480, 1525)
NATIVE = 200
MIRAGE_WORKSPACE = Path(r"D:\jepa_phase0\mirage-goals")
LARGE_CKPT = (MIRAGE_WORKSPACE / "outputs" / "official-vitlarge" / "GOALS"
              / "MIRAGE-Large_frozen_convnext_CEGDice" / "checkpoint-best.pth")
LARGE_SHA256 = "589b0ef236cf1c8f62c978ce09c0b029915d33ba2c310bba21e13121908a0a83"
ADAPTER = ROOT / "results" / "masking" / "structural_loss" / "adapter_ep100_structl100_a010_e3_lr0.001_n4800.pt"
SOFT_CACHE_TAG = dict(mirage_sha="31a932eef403c3e8", adapter_sha="d4f09adfa9f05f0b")
PREPROCESS_RUN = Path(r"D:\jepa_phase0\fairvision-glaucoma\manifests\mirage-preprocess-run.json")
SOFT_CACHE_META = Path(r"D:\jepa_phase0\fairvision-glaucoma\mirage_soft_guides"
                       r"\base512_enc_ad4f09adfa9f05f0b_m31a932eef403c3e8_npy\cache_meta.json")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def minmax_resize(native, size):
    import cv2
    values = native.astype(np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        raise RuntimeError("Constant crop")
    return cv2.resize((values - lo) / (hi - lo), (size, size), interpolation=cv2.INTER_LINEAR)


def build_large_mmap():
    """orchestrate.build_trained_model('cpu'), but memory-mapping the checkpoint.

    Same construction arguments as the training guide producer; mmap avoids a
    second in-memory copy of the 1.3 GB checkpoint on this memory-tight host.
    """
    from argparse import Namespace
    sys.path.insert(0, str(MIRAGE_WORKSPACE))
    import orchestrate
    sys.path.insert(0, str(orchestrate.MIRAGE_ROOT))
    cwd = os.getcwd()
    os.chdir(MIRAGE_WORKSPACE)
    try:
        from fm_seg_config import fm_factory
        from mirage.model import model_factory
        from mirage.output_adapters import ConvNeXtAdapter
        config = fm_factory["mirage-large"]()
        config.build_domain_conf()
        runtime = Namespace(grid_sizes={"bscan": [32, 32]}, input_size={"bscan": [1024, 1024]})
        inputs = {"bscan": config.domain_conf["bscan"]["input_adapter"](
            stride_level=1, patch_size_full=[32, 32], image_size=[1024, 1024],
            learnable_pos_emb=False)}
        outputs = {"semseg": ConvNeXtAdapter(
            num_classes=4, preds_per_patch=16, depth=4, interpolate_mode="bilinear",
            main_tasks=["bscan"], embed_dim=6144, patch_size=[32, 32], task="semseg",
            image_size=[1024, 1024])}
        model = model_factory[config.model](args=runtime, input_adapters=inputs,
                                            output_adapters=outputs, num_global_tokens=1,
                                            drop_path_rate=0.1)
        checkpoint = torch.load(orchestrate.EXPERIMENT_DIR / "checkpoint-best.pth",
                                map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
    finally:
        os.chdir(cwd)
    assert Path(orchestrate.EXPERIMENT_DIR / "checkpoint-best.pth").resolve() == LARGE_CKPT.resolve()
    return model.eval()


@torch.inference_mode()
def hard_guide(native):
    model = build_large_mmap()
    x = torch.from_numpy(minmax_resize(native, 1024))[None, None].float()
    logits = model({"bscan": x})["semseg"]
    probabilities = torch.softmax(logits.float(), dim=1)
    hard = probabilities.argmax(dim=1)
    labels = F.interpolate(hard[:, None].float(), size=(NATIVE, NATIVE),
                           mode="nearest-exact")[0, 0].to(torch.uint8).numpy()
    class_fraction = probabilities.mean(dim=(2, 3))[0].tolist()
    del model, logits, probabilities, hard
    gc.collect()
    envelope, valid, stats = repair_union(build_union(labels), params=DEFAULT_REPAIR)
    return labels, envelope, bool(valid), stats, class_fraction


@torch.inference_mode()
def soft_guide(native):
    from jepa_to_mirage_probe import CK_MIRAGE, build_mirage
    mirage = build_mirage("cpu")
    grab = {}
    semseg = mirage.output_adapters["semseg"]
    head = semseg.final_layer
    head.register_forward_hook(lambda m, i, o: grab.update(H=i[0].detach()))
    checkpoint = torch.load(ADAPTER, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["cfg"])
    if "ch" in cfg:
        from adapter_placement_ablation import Adapter as TapAdapter
        adapter = TapAdapter(cfg["ch"], cfg["depth"], cfg["width"], cfg["alpha"])
    else:
        from adapter_stage import Adapter
        adapter = Adapter(**cfg)
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()
    tap = checkpoint.get("tap") or ("h0" if "ch" not in cfg else "enc")
    if tap == "enc":
        semseg.proj_dec.register_forward_pre_hook(
            lambda m, args: (adapter(args[0].float()).to(args[0].dtype),) + args[1:])
    x = torch.from_numpy(minmax_resize(native, 512))[None, None].float()
    mirage({"bscan": x})
    H = grab["H"].float()
    if tap == "h0":
        H = adapter(H)
    P = head(H).float().softmax(1)[:, (1, 2)]
    P = F.interpolate(P, size=(NATIVE, NATIVE), mode="bilinear", align_corners=False)
    scores = (P.clamp(0, 1) * 255).round().byte()[0].numpy()
    record = dict(mirage_checkpoint=str(CK_MIRAGE), mirage_sha256=sha(CK_MIRAGE),
                  adapter=str(ADAPTER.relative_to(ROOT)), adapter_sha256=sha(ADAPTER),
                  adapter_tap=tap, adapter_cfg=cfg)
    del mirage, adapter, checkpoint
    gc.collect()
    return scores, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=STAGING / "external_guides")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    original = Image.open(SOURCE)
    native = np.asarray(original.crop(CROP).convert("L")).copy()
    started = time.perf_counter()
    labels, envelope, valid, stats, class_fraction = hard_guide(native)
    hard_seconds = time.perf_counter() - started
    large_sha = sha(LARGE_CKPT)
    started = time.perf_counter()
    scores, soft_record = soft_guide(native)
    soft_seconds = time.perf_counter() - started
    checks = dict(
        large_checkpoint_matches_training_preprocess_manifest=large_sha == LARGE_SHA256,
        soft_mirage_matches_cache_tag=soft_record["mirage_sha256"].startswith(SOFT_CACHE_TAG["mirage_sha"]),
        adapter_matches_cache_tag=soft_record["adapter_sha256"].startswith(SOFT_CACHE_TAG["adapter_sha"]))
    if not all(checks.values()):
        raise AssertionError(f"Guide checkpoints differ from the training caches: {checks}")
    npz = args.out_dir / "octdl_crop_guides.npz"
    np.savez_compressed(npz, hard_labels=labels, envelope=envelope, envelope_valid=valid,
                        soft_scores=scores)
    soft_envelope = (scores.astype(np.float32) / 255.0).sum(0) >= 0.5
    record = dict(
        source_image=SOURCE.name, source_image_sha256=sha(SOURCE), crop_xyxy=list(CROP),
        native_crop_pixels=list(native.shape), native_crop_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
        hard=dict(pipeline=("min-max -> cv2 INTER_LINEAR 1024 -> MIRAGE-Large semseg softmax "
                            "argmax -> nearest-exact 200 -> build_union -> repair_union(DEFAULT_REPAIR)"),
                  checkpoint=str(LARGE_CKPT), checkpoint_sha256=large_sha,
                  repair_params_fingerprint=params_fingerprint(DEFAULT_REPAIR),
                  envelope_valid=valid, repair_stats={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                                                      for k, v in stats.items()},
                  class_fraction_elsewhere_rnfl_gcipl_choroid=class_fraction,
                  label_fraction={str(c): float((labels == c).mean()) for c in range(4)},
                  envelope_area_frac=float(envelope.mean()), seconds=hard_seconds),
        soft=dict(pipeline=("min-max -> cv2 INTER_LINEAR 512 -> MIRAGE-Base MergedV3 + adapter "
                            "(encoder tap) -> softmax -> P_inner, P_choroid -> bilinear 200 -> uint8"),
                  **soft_record, mean_scores=[float(s.mean() / 255) for s in scores],
                  soft_envelope_area_frac=float(soft_envelope.mean()), seconds=soft_seconds),
        checkpoint_checks=checks,
        differences_from_training_caches=[
            "CPU fp32 inference; the training caches used CUDA fp16 autocast",
            "input is a 1280x1280 crop of a published OCTDL figure, not a 200x200 FairVision B-scan",
            "MIRAGE was not trained on OCTDL: an illustration of the pipeline, not a validated segmentation"],
        output=dict(file=npz.name, sha256=sha(npz)),
        generator=Path(__file__).name, generator_sha256=sha(Path(__file__)),
        torch=torch.__version__, threads=args.threads)
    (args.out_dir / "octdl_crop_guides.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("checkpoint_checks", "output")}, indent=2))
    print("hard envelope valid", valid, "area", record["hard"]["envelope_area_frac"],
          "soft envelope area", record["soft"]["soft_envelope_area_frac"])


if __name__ == "__main__":
    main()
