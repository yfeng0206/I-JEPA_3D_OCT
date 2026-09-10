"""Delivered-mask contracts, including historical replay and adversarial controls."""
import random
import pickle
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode

from src.datasets.oct_slices_guided import GuidedOCTSliceDataset
from src.guides.mirage_envelope import DEFAULT_REPAIR, patch_occupancy
from src.masks.cover import build_targets
from src.masks.curriculum import CurriculumMaskGenerator, MirageMaskCollator
from src.masks.multiblock import MaskCollator
from src.transforms import make_paired_transforms


SIZES = {"pred": [(8, 6), (6, 7), (7, 7), (6, 6)], "enc": [(15, 16)]}


def guide_batch(n=2):
    guide = torch.zeros(n, 4, 16, 16)
    for b in range(n):
        top = 8 + b % 3
        guide[b, :2, top:top + 3] = 1
        guide[b, 2, top:top + 2] = 1
        guide[b, 3, top + 2:top + 3] = 1
    return guide


def generator(**overrides):
    cfg = dict(mode="mirage_cover", T_warm=0, T_total=1, r_max=1.,
               cover_leave_frac=.21, cover_min_visible_frac=.21,
               cover_fill="random_legal", audit_masks=True)
    cfg.update(overrides)
    obj = CurriculumMaskGenerator(curriculum_cfg=cfg)
    obj.set_epoch(50)
    return obj


def run(obj, guides=None, sizes=SIZES):
    guides = guide_batch() if guides is None else guides
    return obj.generate(len(guides), guide_grids=guides,
                        guide_valid=torch.ones(len(guides), dtype=torch.bool),
                        block_sizes=sizes)


def test_delivered_prefix_is_scored_at_exact_candidate_geometry():
    score = np.random.default_rng(41).random((8, 9))
    masks, info = build_targets(
        [score], [(4, 5)], delivered_k=12, min_visible_frac=0,
        min_visible_cells=0, leave_frac=0, tau=0, rng=random.Random(1),
    )
    gains = []
    for r in range(5):
        for c in range(5):
            ids = [y * 9 + x for y in range(r, r + 4)
                   for x in range(c, c + 5)][:12]
            gains.append(score.ravel()[ids].sum())
    assert masks[0].sum() == 12
    assert score[masks[0]].sum() == pytest.approx(max(gains))
    assert info["covered_frac"] == pytest.approx(max(gains) / score.sum())


def test_boundary_fallback_is_not_counted_as_uniform_random_fill():
    score = np.zeros((4, 4))
    score[:2] = 1
    _, info = build_targets([score], [(1, 1)], leave_frac=1,
                            min_visible_frac=1, min_visible_cells=1,
                            fill="transition", rng=random.Random(1))
    assert info["slot_kind"] == ["boundary_fallback"]
    assert info["n_boundary_fallback"] == 1
    assert info["n_random"] == 0
    _, uniform = build_targets([score], [(1, 1)], leave_frac=1,
                               min_visible_frac=0, min_visible_cells=0,
                               fill="random", rng=random.Random(1))
    assert uniform["slot_kind"] == ["random"]
    assert uniform["n_random"] == 1


def test_legacy_default_is_explicitly_replayable():
    random.seed(51)
    before = run(generator())
    random.seed(51)
    replay = run(generator(cover_algorithm="legacy_v1"))
    for a, b in zip(before[0] + before[1], replay[0] + replay[1]):
        assert torch.equal(a, b)


@pytest.mark.parametrize("algorithm", ["legacy_v1", "delivered_v2"])
def test_cover_valid_nonviable_guide_is_infeasible_not_invalid(algorithm):
    small = torch.zeros(1, 4, 16, 16)
    small[:, :2, 8:10, 8:10] = 1
    small[:, 2, 8:10, 8:10] = 1
    obj = generator(cover_algorithm=algorithm)
    random.seed(0)
    run(obj, small)
    row = obj.last_mask_audit[0]
    assert row["guide_valid"] and not row["guide_viable"]
    assert obj.mirage_stats["fallbacks"] == 0
    assert obj.mirage_stats["infeasible"] == 1
    assert row["target_sources"] == ["infeasible"] * 4
    assert row["target_source_schema_version"] == 2


@pytest.mark.parametrize("algorithm", ["legacy_v1", "delivered_v2"])
def test_cover_mixed_ramp_failure_preserves_unguided_sources(algorithm):
    obj = generator(cover_algorithm=algorithm, r_max=.5,
                    cover_min_visible_frac=1., cover_leave_frac=0.)
    random.seed(0)
    with patch("src.masks.curriculum.random.random", side_effect=[.9, .9, .1, .1]):
        run(obj, guide_batch(1))
    row = obj.last_mask_audit[0]
    assert row["ramp_guided_flags"] == [False, False, True, True]
    assert row["guide_valid"] and row["guide_viable"]
    assert obj.mirage_stats["fallbacks"] == 0
    assert obj.mirage_stats["infeasible"] == 1
    assert row["target_sources"] == ["unguided", "unguided", "infeasible", "infeasible"]


def test_historical_control_detects_current_cover_mutation():
    import src.masks.cover as current_cover
    import src.masks.curriculum as current_curriculum
    from scripts.delivered_mask_audit import historical_replay_controls
    original = current_cover.build_targets

    def shifted_current_targets(*args, **kwargs):
        masks, info = original(*args, **kwargs)
        return [np.roll(mask, 1, axis=0) for mask in masks], info

    with patch.object(current_cover, "build_targets", shifted_current_targets), \
            patch.object(current_curriculum, "cover_build_targets", shifted_current_targets):
        with pytest.raises(AssertionError, match="cover_legacy"):
            historical_replay_controls()


def test_historical_dependencies_are_isolated_and_restored():
    import sys
    import src.masks as package
    import src.masks.cover as current_cover
    import src.masks.curriculum as current_curriculum
    from scripts.delivered_mask_audit import _load_historical_mask_modules
    names = ("utils", "anatomy", "cover", "multiblock", "curriculum")
    previous = {name: sys.modules[f"src.masks.{name}"] for name in names}
    old = _load_historical_mask_modules()
    assert old["curriculum"].cover_build_targets is old["cover"].build_targets
    assert old["curriculum"].cover_is_viable is old["cover"].is_viable
    assert old["curriculum"].cover_build_targets is not current_cover.build_targets
    assert old["curriculum"].cover_build_targets is not current_curriculum.cover_build_targets
    assert sys.modules["src.masks.cover"] is current_cover
    assert package.cover is current_cover
    assert sys.modules["src.masks.curriculum"] is current_curriculum
    assert package.curriculum is current_curriculum
    assert all(sys.modules[f"src.masks.{name}"] is previous[name] for name in names)
    assert all(getattr(package, name) is previous[name] for name in names)


def test_historical_dependency_restore_survives_failed_baseline_import():
    import sys
    import src.masks as package
    import src.masks.cover as current_cover
    import scripts.delivered_mask_audit as audit
    check_output = audit.subprocess.check_output
    names = ("utils", "anatomy", "cover", "multiblock", "curriculum")
    previous = {name: sys.modules[f"src.masks.{name}"] for name in names}

    def broken_baseline(command, **kwargs):
        if command[-1].endswith("src/masks/cover.py"):
            return "raise RuntimeError('injected baseline import failure')"
        return check_output(command, **kwargs)

    with patch.object(audit.subprocess, "check_output", side_effect=broken_baseline):
        with pytest.raises(RuntimeError, match="injected baseline import failure"):
            audit._load_historical_mask_modules()
    assert sys.modules["src.masks.cover"] is current_cover
    assert package.cover is current_cover
    assert all(sys.modules[f"src.masks.{name}"] is previous[name] for name in names)
    assert all(getattr(package, name) is previous[name] for name in names)


def test_corrected_config_explicitly_versions_policy_without_k16():
    from pathlib import Path
    import yaml
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs"
                            / "patch_cover_delivered_v2.yaml").read_text())
    assert "pred_target_k" not in config["mask"]
    cfg = config["mask"]["curriculum"]
    assert cfg["cover_algorithm"] == "delivered_v2"
    assert cfg["cover_context_guard"] is True
    assert cfg["enc_truncate"] == "prefix"
    assert config["meta"]["amp_target"] is False


def test_v2_matches_scored_indices_and_delivered_context_floor():
    obj = generator(cover_algorithm="delivered_v2", cover_context_guard=True)
    random.seed(12)
    enc, pred = run(obj, guide_batch(64))
    assert all(g.shape == (64, 36) for g in pred)
    for b, row in enumerate(obj.last_mask_audit):
        assert row["targets"] == row["intended_targets"]
        assert row["context_floor"]["status"] == "satisfied"
        assert row["context_tissue"][0] >= row["context_floor"]["required"]
        assert not set(enc[0][b].tolist()).intersection(
            i for block in row["targets"] for i in block)
        assert row["loss_slots"] == 144
        assert row["duplicate_loss_slots"] == 144 - row["unique_target_union"]
        assert len(row["target_sources"]) == 4


def test_no_guard_never_reports_unmet_floor_as_satisfied():
    obj = generator(cover_algorithm="delivered_v2", cover_context_guard=False)
    random.seed(12)
    run(obj, guide_batch(64))
    rows = obj.last_mask_audit
    failures = [r for r in rows if r["context_tissue"][0] < r["context_floor"]["required"]]
    assert failures
    assert all(r["context_floor"]["status"] != "satisfied" for r in failures)


def test_guard_ablation_reuses_exact_delivered_targets_and_context_budget():
    plain = generator(cover_algorithm="delivered_v2")
    guarded = generator(cover_algorithm="delivered_v2", cover_context_guard=True)
    random.seed(12)
    enc_a, pred_a = run(plain, guide_batch(64))
    random.seed(12)
    enc_b, pred_b = run(guarded, guide_batch(64))
    assert all(torch.equal(a, b) for a, b in zip(pred_a, pred_b))
    assert enc_a[0].shape == enc_b[0].shape
    assert all(a["context_before_collation"] == b["context_before_collation"]
               for a, b in zip(plain.last_mask_audit, guarded.last_mask_audit))


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("kind", ["stock", "curriculum", "cover_guard"])
def test_multiple_context_groups_match_actual_apply_masks_contract(kind, batch_size):
    from scripts.delivered_mask_audit import validate_delivered
    from src.masks.utils import apply_masks
    sizes = {"pred": [(6, 6)] * 4, "enc": [(12, 12), (16, 16)]}
    images = torch.zeros(batch_size, 3, 256, 256)
    random.seed(0)
    if kind == "stock":
        obj = MaskCollator(nenc=2, audit_masks=True)
        _, enc, pred = obj(list(images), block_sizes=sizes)
    else:
        cfg = dict(mode="loss_guided", r_max=0, audit_masks=True)
        if kind == "cover_guard":
            cfg.update(mode="mirage_cover", r_max=1, T_warm=0, T_total=1,
                       cover_algorithm="delivered_v2", cover_context_guard=True)
        obj = CurriculumMaskGenerator(nenc=2, curriculum_cfg=cfg)
        obj.set_epoch(50)
        enc, pred = obj.generate(
            batch_size, guide_grids=guide_batch(batch_size),
            guide_valid=torch.ones(batch_size, dtype=torch.bool), block_sizes=sizes)
    raw_lengths = [len(group) for row in obj.last_mask_audit
                   for group in row["context_before_collation"]]
    assert len(set(raw_lengths)) > 1, "fixture must reproduce unequal raw lengths"
    assert enc[0].shape == enc[1].shape == (batch_size, min(raw_lengths))
    validate_delivered(enc, pred, batch_size=batch_size, nenc=2)
    codes = torch.arange(batch_size * 256 * 3).reshape(batch_size, 256, 3).float()
    actual = apply_masks(codes, enc)
    expected = torch.cat([
        torch.stack([codes[b, group[b]] for b in range(batch_size)])
        for group in enc
    ])
    assert torch.equal(actual, expected)
    assert actual.shape == (batch_size * 2, min(raw_lengths), 3)


def test_validator_detects_corrupted_cross_group_context_length():
    from scripts.delivered_mask_audit import validate_delivered
    enc = [torch.tensor([[0, 1]]), torch.tensor([[0, 1, 2]])]
    pred = [torch.tensor([[10, 11]]) for _ in range(4)]
    with pytest.raises(ValueError, match="Unequal context group budgets"):
        validate_delivered(enc, pred, batch_size=1, nenc=2)


@pytest.mark.parametrize("reason", ["budget", "union", "invalid"])
def test_context_guard_reports_infeasible_or_invalid(reason):
    obj = generator(cover_algorithm="delivered_v2", cover_context_guard=True)
    guide = guide_batch(1)
    tissue = (guide[0, 0].flatten() > 0).nonzero().flatten()
    enc = [torch.tensor([[0, 1]]) if reason == "budget"
           else torch.arange(20).unsqueeze(0)]
    pred = [tissue.unsqueeze(0) if reason == "union" else torch.tensor([[30]])]
    result = obj._guard_delivered_context(enc, pred, guide, torch.tensor([reason != "invalid"]))
    expected = dict(budget="infeasible_context_budget", union="infeasible_target_union",
                    invalid="invalid_guide")
    assert result[0]["status"] == expected[reason]


def test_nonprefix_rng_epoch_worker_rank_and_progression():
    obj = generator(enc_truncate="random")
    obj._enc_trunc_epoch = obj._epoch
    obj._reseed_enc_trunc()
    a = torch.rand(5, generator=obj._enc_trunc_gen)
    obj._reseed_enc_trunc()
    b = torch.rand(5, generator=obj._enc_trunc_gen)
    assert not torch.equal(a, b), "a second batch must not restart the stream"
    replay = generator(enc_truncate="random")
    replay._enc_trunc_epoch = replay._epoch
    replay._reseed_enc_trunc()
    assert torch.equal(a, torch.rand(5, generator=replay._enc_trunc_gen))
    obj.set_epoch(51)
    run(obj)
    assert obj._enc_trunc_epoch == 51
    assert obj._enc_trunc_key[1] == 51
    other = generator(enc_truncate="random")
    other.rank = 1
    other._enc_trunc_epoch = 50
    other._reseed_enc_trunc()
    assert not torch.equal(a, torch.rand(5, generator=other._enc_trunc_gen))
    worker = type("Worker", (), {"id": 2})()
    with patch("torch.utils.data.get_worker_info", return_value=worker):
        replay._reseed_enc_trunc()
    assert replay._enc_trunc_key[-1] == 2
    assert not torch.equal(a, torch.rand(5, generator=replay._enc_trunc_gen))


def test_legacy_rng_option_preserves_epoch_zero_restarting_behavior():
    obj = generator(enc_truncate="random", enc_truncate_rng="legacy_v1")
    obj._reseed_enc_trunc()
    a = torch.rand(5, generator=obj._enc_trunc_gen)
    obj._reseed_enc_trunc()
    assert torch.equal(a, torch.rand(5, generator=obj._enc_trunc_gen))
    run(obj)
    assert obj._enc_trunc_epoch == 0


@pytest.mark.parametrize("kind", ["stock", "curriculum"])
def test_impossible_context_does_not_leak_targets(kind):
    sizes = {"pred": [(16, 16)] * 4, "enc": [(16, 16)]}
    with pytest.raises(ValueError, match="No non-target context"):
        if kind == "stock":
            MaskCollator()([torch.zeros(3, 256, 256)], block_sizes=sizes)
        else:
            run(generator(r_max=0), guide_batch(1), sizes)


def test_anatomy_ramp_off_is_not_invalid_guide():
    obj = CurriculumMaskGenerator(pred_target_k=16, curriculum_cfg=dict(
        mode="mirage_anatomy", T_warm=25, T_total=30, audit_masks=True))
    obj.set_epoch(0)
    run(obj)
    assert obj.mirage_stats["fallbacks"] == 0
    assert obj.mirage_stats["unbiased_by_ramp"] == 2
    assert obj.mirage_stats["guided_images"] == 0


def test_worker_pickle_preserves_epoch_but_rebuilds_generator():
    collator = MirageMaskCollator(curriculum_cfg=dict(mode="mirage_cover",
                                  T_warm=25, T_total=30, r_max=1))
    collator.set_epoch(27, 100)
    collator._get_generator()
    restored = pickle.loads(pickle.dumps(collator))
    assert restored._generator is None
    assert restored._get_generator().r_t == pytest.approx(.4)
    restored.set_epoch(50, 100)
    assert restored._get_generator().r_t == 1


def test_invalid_guide_and_ramp_off_have_distinct_sources():
    guides = guide_batch()
    obj = generator(cover_algorithm="delivered_v2", cover_context_guard=True)
    obj.generate(2, guide_grids=guides, guide_valid=torch.zeros(2, dtype=torch.bool),
                 block_sizes=SIZES)
    assert all(r["target_sources"] == ["fallback_invalid"] * 4
               for r in obj.last_mask_audit)
    assert all(r["context_floor"]["status"] == "invalid_guide"
               for r in obj.last_mask_audit)
    obj.set_epoch(0)
    run(obj, guides)
    assert all(r["target_sources"] == ["unbiased_by_ramp"] * 4
               for r in obj.last_mask_audit)
    assert obj.mirage_stats["fallbacks"] == 0


def test_full_ramp_guidance_and_random_legal_fill_are_separate():
    thin = guide_batch(1)
    thick = torch.zeros(1, 4, 16, 16)
    thick[:, :2, 4:12] = 1
    thick[:, 2, 4:8] = 1
    thick[:, 3, 8:12] = 1
    random_counts = []
    for guide in (thin, thick):
        obj = generator(cover_algorithm="delivered_v2")
        random.seed(12)
        run(obj, guide)
        row = obj.last_mask_audit[0]
        assert row["ramp_guided_flags"] == [True] * 4
        random_counts.append(row["target_sources"].count("random_legal"))
    assert random_counts == [1, 0]


def test_envelope_per_slot_infeasible_uniform_source_is_retained():
    obj = generator(mode="mirage_envelope", mirage_min_block_fill=.5)
    occupancy = torch.zeros(16, 16)
    occupancy[8, 8] = 1
    masks, stats = obj._sample_mirage_blocks(
        [(1, 1), (8, 8)], occupancy, occupancy,
        [True, True], [None, None])
    assert stats["slot_kind"] == ["envelope", "infeasible_uniform"]
    assert len(masks) == 2


def test_shifted_guide_changes_targets_but_does_not_prove_semantic_accuracy():
    obj = generator(cover_algorithm="delivered_v2")
    original = guide_batch(1)
    shifted = original.roll(-5, dims=2)
    random.seed(42)
    _, a = run(obj, original)
    random.seed(42)
    _, b = run(obj, shifted)
    ia = torch.cat([x[0] for x in a]).unique()
    ib = torch.cat([x[0] for x in b]).unique()
    assert not torch.equal(ia, ib)
    assert int(original[0, 0].flatten()[ia].sum()) > int(original[0, 0].flatten()[ib].sum())


def test_wrong_target_count_and_duplicate_weighting_controls():
    from scripts.delivered_mask_audit import validate_delivered, measure
    enc = [torch.tensor([[0, 1, 2]])]
    pred = [torch.tensor([[128, 128]]) for _ in range(4)]
    validate_delivered(enc, pred, batch_size=1)
    with pytest.raises(ValueError, match="Wrong target"):
        validate_delivered(enc, pred[:-1], batch_size=1)
    row = dict(intended_targets=[[128]] * 4, targets=[[128, 128]] * 4,
               context=[[0, 1, 2]], context_before_collation=[[0, 1, 2]],
               target_sources=["random_legal"] * 4)
    result = measure(row, guide_batch(1)[0])
    assert result["delivered_loss_slots"] == 8
    assert result["unique_target_union"] == 1
    assert result["duplicate_loss_slots"] == 7
    assert result["random_tissue_slots"] == 8
    assert result["random_background_slots"] == 0


def test_edge_probe_reports_actual_context_not_target_complement():
    from scripts.mask_edge_case_test import run as edge_run
    guide = guide_batch(1)[0].numpy()
    case = dict(vol="synthetic", slice="band", cs=[guide[2], guide[3]],
                anat_occ=guide[0], anat=guide[0] >= .25)
    row = edge_run([case], .21, seed=42)[0]
    assert row["anat_visible"] == int((case["anat"] & row["context_mask"]).sum())
    assert row["anat_visible"] <= row["complement_anatomy"]
    assert row["context_status"] == "satisfied"


def test_figure_aggregates_exclude_short_tail_and_keep_invalid_contexts():
    from scripts.delivered_mask_figure import cohort_metrics
    row = dict(batch_size=64, guide_valid=True, policy_info={"ok": True},
               scored_hidden_mass_fraction=.8, delivered_hidden_mass_fraction=.7,
               context_tissue_before_collation=12, context_tissue=8)
    invalid = dict(row, guide_valid=False, policy_info={},
                   context_tissue_before_collation=0, context_tissue=0)
    short = dict(row, batch_size=24, context_tissue=200)
    metrics = cohort_metrics([row, invalid, short], [row, invalid, short])
    mass = next(r for r in metrics if r["metric"] == "hidden_guide_mass_percent"
                and r["stage"] == "Delivered")
    context = next(r for r in metrics if r["metric"] == "context_tissue_cells"
                   and r["stage"] == "Final")
    assert mass["n"] == 1 and mass["mean"] == pytest.approx(70)
    assert context["n"] == 2 and context["mean"] == 4


def test_figure_rejects_context_target_overlap():
    import matplotlib.pyplot as plt
    from scripts.delivered_mask_figure import mask_panel
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="context-target overlap"):
            mask_panel(ax, np.zeros((16, 16), dtype=bool), [[1, 2]], [2, 3], "invalid")
    finally:
        plt.close(fig)


def test_dataset_paired_crop_preserves_coordinates_and_score_channels():
    ds = object.__new__(GuidedOCTSliceDataset)
    ds.num_slices, ds.slice_size, ds.patch_size, ds.grid_size = 1, 256, 16, 16
    ds.dilate_patches, ds.occupancy_threshold = 0, .25
    ds.require_guides, ds.repair_params = True, DEFAULT_REPAIR
    rows, cols = np.indices((200, 200))
    image = ((rows + cols) % 256).astype(np.uint8)
    soft = np.stack([(rows >= 80) & (rows < 105),
                     (rows >= 125) & (rows < 140)]).astype(np.float32)
    envelope = soft.sum(0) > .5
    ds.read_slice = lambda *_: image
    ds._load_guide = lambda *_: (envelope, True)
    ds._load_soft_guide = lambda *_: soft
    ds.transform = make_paired_transforms()
    crop = (32, 40, 176, 160)
    with patch("src.transforms.T.RandomResizedCrop.get_params", return_value=crop) as draw:
        result, guide, valid = ds[0]
    assert draw.call_count == 1
    native = Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR).convert("RGB")
    expected = TF.resized_crop(native, *crop, [256, 256], InterpolationMode.BICUBIC)
    expected = TF.normalize(TF.to_tensor(expected), ds.transform.normalize_mean,
                            ds.transform.normalize_std)
    assert torch.equal(result, expected)
    hard = Image.fromarray(envelope.astype(np.uint8) * 255).resize(
        (256, 256), Image.Resampling.NEAREST)
    hard = TF.resized_crop(hard, *crop, [256, 256], InterpolationMode.NEAREST)
    occupancy = patch_occupancy(np.asarray(hard) > 127, 16)
    np.testing.assert_array_equal(guide[0].numpy(), occupancy)
    assert bool(valid)
    assert not np.array_equal(np.roll(guide[0].numpy(), 4, axis=0), occupancy)
    for ch in range(2):
        rgb = np.zeros((200, 200, 3), dtype=np.uint8)
        rgb[..., ch] = (soft[ch] * 255).astype(np.uint8)
        resized = Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR)
        resized = TF.resized_crop(resized, *crop, [256, 256], InterpolationMode.BICUBIC)
        pooled = (np.asarray(resized)[..., ch].astype(np.float32) / 255).reshape(
            16, 16, 16, 16).mean((1, 3))
        np.testing.assert_allclose(guide[ch + 2].numpy(), pooled, atol=1e-7)
