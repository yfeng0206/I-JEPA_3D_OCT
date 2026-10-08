"""RANDOM-CB (E9): uniform placement at CENTROID's exact delivered budgets.

Per-image U/L/D/C (and H for ``matching: strict``) equal the production
CENTROID shadow; no context/target leakage; ramp-0 is the stock sampler bit for
bit; deterministic replay; the matcher never advances the global RNG; the exact
fallback is uniform and logged; worker pickling under Windows spawn; trainer
wiring end to end on the tiny CPU loop.  CPU only.
"""

import ast
import itertools
import json
import os
import pickle
import random

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.masks.centroid_budget import (
    CentroidBudgetRandomCollator,
    _RectTable,
    _or_rows,
    _popcount,
    exact_conditional_sample,
    match_budgets,
)
from src.masks.curriculum import CENTROID_BUDGET_RANDOM_MODE, CurriculumMaskGenerator
from src.masks.multiblock import MaskCollator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATCH = os.path.join(REPO_ROOT, "src", "train_patch.py")
CB_CONFIG = os.path.join(REPO_ROOT, "configs", "cr_seed_v1", "cr_seed_v1_random_cb_s1234.yaml")
SLICE_CACHE = r"C:\jepa_data\slice_cache\Training"
DATA_DIR = r"D:\jepa_phase0\fairvision-glaucoma\data\Training"

KW = dict(input_size=(256, 256), patch_size=16, enc_mask_scale=(0.85, 1.0),
          pred_mask_scale=(0.15, 0.2), aspect_ratio=(0.75, 1.5), nenc=1, npred=4,
          min_keep=10, allow_overlap=False, pred_target_k=None)
CENTROID = dict(enabled=True, mode="anatomical_prior", T_warm=25, T_total=30, r_max=1.0,
                ramp_shape="linear", oracle_region_frac=0.28, oracle_lateral_frac=0.6,
                oracle_row_offset=0.0, oracle_min_band_rows=3, enc_truncate="prefix")


def cb_cfg(matching="strict", **extra):
    return dict(CENTROID, mode=CENTROID_BUDGET_RANDOM_MODE, matching=matching, **extra)


def band_images(batch, seed, size=256, patch=16):
    """Bright horizontal bands at varying rows, one flat image (centroid fallback)."""
    grid = size // patch
    gen = torch.Generator().manual_seed(seed)
    img = torch.rand(batch, 1, grid, grid, generator=gen) * 0.2
    for i in range(batch):
        row = int(torch.randint(1, grid - 3, (1,), generator=gen))
        img[i, :, row:row + 3] += 1.0
    img[0] = 0.0
    return img.repeat(1, 3, 1, 1).repeat_interleave(patch, 2).repeat_interleave(patch, 3)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rect_union(pred, sizes, i, grid_w=16):
    """Independent recount of the full (pre-truncation) union of image i's targets."""
    cells = set()
    for group, (h, w) in zip(pred, sizes):
        first = int(group[i][0])
        top, left = divmod(first, grid_w)
        cells |= {(top + r) * grid_w + left + c for r in range(h) for c in range(w)}
    return cells


def budget(enc, pred, sizes, i, grid_w=16, cells=256):
    slots = torch.cat([p[i] for p in pred]).tolist()
    context = enc[0][i].tolist()
    full = rect_union(pred, sizes, i, grid_w)
    assert set(slots) <= full
    assert not set(context) & full, "context leaks a target cell"
    assert context == sorted(set(context)), "context must be sorted and unique"
    assert all(0 <= v < cells for v in slots + context)
    unique = len(set(slots))
    return {"U": unique, "L": len(slots), "D": len(slots) - unique, "C": len(context),
            "H": len(full)}


def make(matching="strict", epoch=30, **extra):
    col = CentroidBudgetRandomCollator(**KW, curriculum_cfg=cb_cfg(matching, **extra))
    col.set_epoch(epoch, 100)
    return col


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------


@pytest.mark.parametrize("change", [
    {"matching": None}, {"matching": "exact"}, {"matching": "Strict"},
    {"enc_truncate": "random"}, {"mode": "anatomical_prior"}, {"cb_max_proposals": 0},
])
def test_invalid_curriculum_keys_raise_at_construction(change):
    cfg = cb_cfg()
    cfg.update(change)
    if cfg.get("matching") is None:
        cfg.pop("matching")
    with pytest.raises(ValueError):
        CentroidBudgetRandomCollator(**KW, curriculum_cfg=cfg)


@pytest.mark.parametrize("change", [{"pred_target_k": 16}, {"allow_overlap": True}, {"nenc": 2}])
def test_unsupported_mask_settings_raise(change):
    with pytest.raises(ValueError):
        CentroidBudgetRandomCollator(**dict(KW, **change), curriculum_cfg=cb_cfg())


def test_generator_in_cb_mode_refuses_to_generate():
    gen = CurriculumMaskGenerator(**KW, curriculum_cfg=cb_cfg())
    gen.set_epoch(30, 100)
    with pytest.raises(RuntimeError, match="CentroidBudgetRandomCollator"):
        gen.generate(2, imgs_cpu=band_images(2, 0))


def test_shadow_config_is_the_centroid_config():
    col = make()
    assert col._kwargs["curriculum_cfg"] == dict(CENTROID, audit_masks=False)
    shadow = col._get_generator()
    assert (shadow.mode, shadow.oracle_lateral_frac, shadow.oracle_region_frac,
            shadow.oracle_row_offset, shadow.oracle_min_band_rows) == (
        "anatomical_prior", 0.6, 0.28, 0.0, 3)
    assert (shadow.T_warm, shadow.T_total, shadow.r_max, shadow.enc_truncate) == (
        25, 30, 1.0, "prefix")


# ----------------------------------------------------------------------
# budgets equal the shadow, image by image
# ----------------------------------------------------------------------


@pytest.mark.parametrize("matching", ["basic", "strict"])
@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize("epoch", [27, 30])
def test_per_image_budgets_equal_the_centroid_shadow(matching, batch, epoch):
    col = make(matching, epoch)
    for draw in range(2):
        seed_all(100 * batch + 10 * epoch + draw)
        out = col.collate_masks(band_images(batch, draw), audit=True)
        sizes = out["sizes"]["pred"]
        assert out["stats"]["bypass_ramp0"] == 0 and out["stats"]["exact_fallbacks"] == 0
        k = out["masks_pred"][0].shape[1]
        assert out["shadow_pred"][0].shape[1] == k
        assert out["masks_enc"][0].shape == out["shadow_enc"][0].shape
        for i in range(batch):
            shadow = budget(out["shadow_enc"], out["shadow_pred"], sizes, i)
            matched = budget(out["masks_enc"], out["masks_pred"], sizes, i)
            for key in ("U", "L", "D", "C"):
                assert matched[key] == shadow[key], (key, i)
            if matching == "strict":
                assert matched["H"] == shadow["H"], i
            assert matched["L"] == KW["npred"] * k
        assert out["matched_unique"].tolist() == out["shadow_unique"].tolist()


def test_matched_placements_differ_from_the_shadow():
    col = make()
    seed_all(5)
    out = col.collate_masks(band_images(64, 5), audit=True)
    same = sum(torch.equal(a, b) for a, b in zip(out["masks_pred"], out["shadow_pred"]))
    assert same == 0


def test_shadow_is_production_centroid_and_global_rng_is_untouched():
    """The CB shadow == CurriculumMaskGenerator(anatomical_prior) on the same RNG,
    and the matcher's private stream leaves Python/Torch/NumPy global RNG as the
    plain CENTROID call leaves them."""
    imgs = band_images(16, 9)
    ref = CurriculumMaskGenerator(**KW, curriculum_cfg=CENTROID)
    ref.set_epoch(28, 100)
    seed_all(77)
    ref_enc, ref_pred = ref.generate(16, imgs_cpu=imgs)
    after_ref = (random.getstate(), torch.get_rng_state(), np.random.get_state()[1].copy())
    col = make("strict", 28)
    seed_all(77)
    out = col.collate_masks(imgs, audit=True)
    after_cb = (random.getstate(), torch.get_rng_state(), np.random.get_state()[1].copy())
    for a, b in zip(ref_enc + ref_pred, out["shadow_enc"] + out["shadow_pred"]):
        assert torch.equal(a, b)
    assert after_cb[0] == after_ref[0]
    assert torch.equal(after_cb[1], after_ref[1])
    assert np.array_equal(after_cb[2], after_ref[2])


@pytest.mark.parametrize("matching", ["basic", "strict"])
@pytest.mark.parametrize("batch", [1, 8, 64])
def test_ramp_zero_is_bit_identical_to_stock_random_and_centroid(matching, batch):
    imgs = band_images(batch, 3)
    seed_all(42)
    _, r_enc, r_pred = MaskCollator(**{k: v for k, v in KW.items() if k != "pred_target_k"})(
        list(imgs))
    centroid = CurriculumMaskGenerator(**KW, curriculum_cfg=CENTROID)
    centroid.set_epoch(25, 100)
    seed_all(42)
    c_enc, c_pred = centroid.generate(batch, imgs_cpu=imgs)
    col = make(matching, 25)
    seed_all(42)
    got_imgs, enc, pred, stats = col(list(imgs))
    assert stats["bypass_ramp0"] == 1 and stats["r_t"] == 0.0
    assert torch.equal(got_imgs, imgs)
    for a, b, c in zip(enc + pred, r_enc + r_pred, c_enc + c_pred):
        assert torch.equal(a, b) and torch.equal(a, c)


def test_deterministic_replay_and_epoch_keyed_private_stream():
    imgs = band_images(32, 4)

    def draw(epoch):
        col = make("strict", epoch)
        seed_all(2024)
        return col.collate_masks(imgs, audit=True)

    a, b, c = draw(30), draw(30), draw(31)
    for x, y in zip(a["masks_enc"] + a["masks_pred"], b["masks_enc"] + b["masks_pred"]):
        assert torch.equal(x, y)
    # r_t is 1 at both epochs, so the shadow is identical but the matcher stream is not.
    for x, y in zip(a["shadow_enc"] + a["shadow_pred"], c["shadow_enc"] + c["shadow_pred"]):
        assert torch.equal(x, y)
    assert any(not torch.equal(x, y) for x, y in zip(a["masks_pred"], c["masks_pred"]))


def test_consecutive_batches_use_fresh_private_streams():
    col = make()
    imgs = band_images(8, 6)
    seed_all(1)
    first = col.collate_masks(imgs, block_sizes={"pred": [(6, 7)] * 4, "enc": [(16, 14)]})
    seed_all(1)
    second = col.collate_masks(imgs, block_sizes={"pred": [(6, 7)] * 4, "enc": [(16, 14)]})
    assert any(not torch.equal(x, y) for x, y in zip(first["masks_pred"], second["masks_pred"]))


def test_call_index_restarts_with_each_new_epoch_only():
    col = make("strict", 30)
    seed_all(1)
    col.collate_masks(band_images(4, 1))
    col.collate_masks(band_images(4, 2))
    col.set_epoch(30, 100)  # same epoch: keep counting
    assert col._calls == 2
    col.set_epoch(31, 100)
    assert col._calls == 0


class _ScriptedRNG:
    """Uniform-proposal stand-in whose first qualifying proposal is ``first_hit``."""

    def __init__(self, first_hit):
        self.calls = self.proposals = 0
        self.first_hit = first_hit

    def integers(self, high, size=None):
        if size is None:  # the exact sampler's single draw
            return 0
        group = self.calls % 5  # four target groups, then the context window
        self.calls += 1
        out = np.zeros(size, dtype=np.int64)
        if group == 3:
            out[:] = 1
            index = self.first_hit - self.proposals - 1
            if 0 <= index < size[1]:
                out[:, index] = 0
        if group == 4:
            self.proposals += size[1]
        return out


@pytest.mark.parametrize("cap", [65, 2 ** 21])
def test_rejection_never_accepts_past_the_cap(cap):
    from src.masks import centroid_budget as cbmod

    assert cbmod.DEFAULT_CB_MAX_PROPOSALS == 2 ** 21
    tabs = [_RectTable(1, 1, 2, 2, 1) for _ in range(4)]
    enc = _RectTable(2, 2, 2, 2, None)
    # Qualifying tuple: all four 1x1 targets on cell 0 (U = H = 1).
    _, _, props, exact = match_budgets(tabs, enc, 1, np.array([1]), np.array([1]),
                                       _ScriptedRNG(first_hit=cap), max_proposals=cap)
    assert props.tolist() == [cap] and exact == []  # the cap-th proposal is still allowed
    _, _, props, exact = match_budgets(tabs, enc, 1, np.array([1]), np.array([1]),
                                       _ScriptedRNG(first_hit=cap + 1), max_proposals=cap)
    assert props.tolist() == [cap] and exact == [0]  # beyond it: the logged exact path


# ----------------------------------------------------------------------
# exact fallback and its distribution
# ----------------------------------------------------------------------


@pytest.mark.parametrize("matching", ["basic", "strict"])
def test_exact_fallback_preserves_counts_and_is_reported(matching):
    kw = dict(KW, input_size=(128, 128))
    col = CentroidBudgetRandomCollator(**kw, curriculum_cfg=cb_cfg(matching, cb_max_proposals=1))
    col.set_epoch(30, 100)
    seed_all(8)
    out = col.collate_masks(band_images(8, 8, size=128), audit=True)
    stats = out["stats"]
    assert stats["exact_fallbacks"] == len(out["exact_fallback_images"]) >= 1
    sizes = out["sizes"]["pred"]
    for i in range(8):
        shadow = budget(out["shadow_enc"], out["shadow_pred"], sizes, i, grid_w=8, cells=64)
        matched = budget(out["masks_enc"], out["masks_pred"], sizes, i, grid_w=8, cells=64)
        keys = ("U", "L", "D", "C", "H") if matching == "strict" else ("U", "L", "D", "C")
        assert all(matched[k] == shadow[k] for k in keys)


def _tiny_problem():
    sizes = [(2, 2), (2, 2), (1, 3)]
    k = min(h * w for h, w in sizes)
    tabs = [_RectTable(h, w, 5, 5, k) for h, w in sizes]
    enc = _RectTable(5, 4, 5, 5, None)
    ref = [np.array([0]), np.array([1]), np.array([5])]  # a concrete overlapping placement
    unique = int(_popcount(_or_rows(tabs, ref, "deliv"))[0])
    hidden = int(_popcount(_or_rows(tabs, ref, "full"))[0])
    return tabs, enc, unique, hidden, 13  # 13 of the 20-cell window: 1 or 2 windows fit


def _brute_force(tabs, enc, unique, hidden, context):
    feasible = []
    for combo in itertools.product(*[range(t.n) for t in tabs]):
        d = set().union(*[set(t.idx[p, :t.k].tolist()) for t, p in zip(tabs, combo)])
        f = set().union(*[set(t.idx[p].tolist()) for t, p in zip(tabs, combo)])
        if len(d) != unique or (hidden is not None and len(f) != hidden):
            continue
        for e in range(enc.n):
            if len(set(enc.idx[e].tolist()) - f) >= context:
                feasible.append(combo + (e,))
    return feasible


@pytest.mark.parametrize("strict", [False, True])
def test_exact_sampler_counts_and_uniformity_match_brute_force(strict):
    tabs, enc, unique, hidden, context = _tiny_problem()
    feasible = _brute_force(tabs, enc, unique, hidden if strict else None, context)
    assert len(feasible) > 10
    # Some target tuples admit one context window, others two: the joint draw
    # must weight target tuples by their number of feasible windows.
    windows_per_target = {}
    for row in feasible:
        windows_per_target[row[:-1]] = windows_per_target.get(row[:-1], 0) + 1
    assert set(windows_per_target.values()) == {1, 2}
    rng = np.random.default_rng(123)
    draws = 20 * len(feasible)
    hist = dict.fromkeys(feasible, 0)
    for _ in range(draws):
        pos, e, total = exact_conditional_sample(tabs, enc, unique, hidden if strict else None,
                                                 context, rng)
        assert total == len(feasible)
        hist[tuple(pos) + (e,)] += 1
    counts = np.array(list(hist.values()))
    chi2 = float(((counts - 20.0) ** 2 / 20.0).sum())
    dof = len(feasible) - 1
    assert counts.min() > 0
    assert chi2 < dof + 6 * np.sqrt(2 * dof), chi2


@pytest.mark.parametrize("strict", [False, True])
def test_rejection_sampler_is_uniform_over_the_same_set(strict):
    tabs, enc, unique, hidden, context = _tiny_problem()
    feasible = _brute_force(tabs, enc, unique, hidden if strict else None, context)
    n = 20 * len(feasible)
    rng = np.random.default_rng(321)
    pos, epos, _, exact = match_budgets(
        tabs, enc, context, np.full(n, unique), np.full(n, hidden) if strict else None, rng)
    assert exact == []
    hist = dict.fromkeys(feasible, 0)
    for row, e in zip(pos.tolist(), epos.tolist()):
        hist[tuple(row) + (e,)] += 1  # KeyError = an infeasible tuple was accepted
    counts = np.array(list(hist.values()))
    chi2 = float(((counts - 20.0) ** 2 / 20.0).sum())
    dof = len(feasible) - 1
    assert chi2 < dof + 6 * np.sqrt(2 * dof), chi2


# ----------------------------------------------------------------------
# workers (Windows spawn) and pickling
# ----------------------------------------------------------------------


class BandDataset(Dataset):
    def __init__(self, n=16):
        self.images = band_images(n, 99)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def test_pickle_drops_generator_and_caches_but_keeps_epoch():
    col = make("strict", 29)
    seed_all(0)
    col.collate_masks(band_images(4, 0))
    assert col._generator is not None and col._tables
    copy = pickle.loads(pickle.dumps(col))
    assert copy._generator is None and copy._tables == {}
    assert (copy.epoch, copy.matching, copy.max_proposals) == (29, "strict", col.max_proposals)


def _loader_run(col, epoch):
    col.set_epoch(epoch, 100)
    loader = DataLoader(BandDataset(16), batch_size=8, num_workers=2, collate_fn=col,
                        shuffle=False, persistent_workers=False, prefetch_factor=1,
                        multiprocessing_context="spawn")
    torch.manual_seed(0)
    return list(loader)


def test_spawn_workers_deliver_valid_deterministic_batches():
    col = make("strict", 30)
    first = _loader_run(col, 30)
    again = _loader_run(col, 30)
    ramp0 = _loader_run(col, 25)
    assert len(first) == 2
    for (imgs, enc, pred, stats), (_, enc2, pred2, stats2) in zip(first, again):
        assert stats["bypass_ramp0"] == 0 and stats["r_t"] == 1.0 and stats["epoch"] == 30
        assert stats["exact_fallbacks"] == 0
        for x, y in zip(enc + pred, enc2 + pred2):
            assert torch.equal(x, y)
        k = pred[0].shape[1]
        assert stats["loss_slots"] == 4 * k and stats["context"] == enc[0].shape[1]
        for i in range(imgs.shape[0]):
            slots = set(torch.cat([p[i] for p in pred]).tolist())
            assert not slots & set(enc[0][i].tolist())
    assert all(batch[3]["bypass_ramp0"] == 1 for batch in ramp0)


# ----------------------------------------------------------------------
# trainer wiring
# ----------------------------------------------------------------------


def test_train_patch_builds_the_cb_collator_from_the_whole_curriculum_dict():
    with open(TRAIN_PATCH, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=TRAIN_PATCH)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "CentroidBudgetRandomCollator"]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw["curriculum_cfg"], ast.Name) and kw["curriculum_cfg"].id == "curr_cfg"
    source = ast.unparse(tree)
    assert "use_cb = use_curriculum and curr_cfg.get('mode') == CENTROID_BUDGET_RANDOM_MODE" in source
    assert "if use_mirage or use_cb:" in source


def test_tiny_trainer_runs_cb_end_to_end(tmp_path, monkeypatch, capsys):
    from src import train_patch as training
    from tests.test_cr_trainer_run_control import install_tiny, make_config, run

    monkeypatch.setenv("JEPA_SHA256_CACHE", str(tmp_path / "sha256_cache.json"))
    install_tiny(monkeypatch.setattr)
    folder = tmp_path / "run"
    config = make_config(tmp_path, folder, epochs=3)
    config["mask"]["curriculum"] = dict(
        enabled=True, mode=CENTROID_BUDGET_RANDOM_MODE, matching="strict", T_warm=1, T_total=2,
        r_max=1.0, ramp_shape="linear", oracle_region_frac=0.28, oracle_lateral_frac=0.6,
        oracle_row_offset=0.0, oracle_min_band_rows=3, enc_truncate="prefix")
    assert run(tmp_path, config) == 0
    out = capsys.readouterr().out
    assert "RANDOM-CB: matching=strict shadow=anatomical_prior" in out
    assert "[CB-EPOCH] ep=0 batches=7 bypass_ramp0_batches=7 exact_fallback_images=0" in out
    assert "[CB-EPOCH] ep=2 batches=7 bypass_ramp0_batches=0 exact_fallback_images=0" in out
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert manifest["mask_policy"] == CENTROID_BUDGET_RANDOM_MODE
    assert training.format_cb_stats({
        "matching": "strict", "r_t": 1.0, "bypass_ramp0": 0, "target_len": 40, "context": 80,
        "loss_slots": 160, "unique_targets": 100.0, "duplicates": 60.0, "hidden_shadow": 110.0,
        "hidden_matched": 110.0, "proposals_mean": 3.0, "proposals_max": 9,
        "exact_fallbacks": 0, "shadow_ms": 1.0, "match_ms": 1.0}).startswith("    [CB] ")


def test_zero_worker_exact_resume_matches_uninterrupted_cb_training(tmp_path, monkeypatch):
    """num_workers=0: the trainer's own collator carries across epochs, so its
    private stream must not depend on how many epochs this process has run."""
    from tests.test_cr_trainer_run_control import exact_from, install_tiny, make_config, run

    monkeypatch.setenv("JEPA_SHA256_CACHE", str(tmp_path / "sha256_cache.json"))
    install_tiny(monkeypatch.setattr)
    curriculum = dict(
        enabled=True, mode=CENTROID_BUDGET_RANDOM_MODE, matching="strict", T_warm=0, T_total=1,
        r_max=1.0, ramp_shape="linear", oracle_region_frac=0.28, oracle_lateral_frac=0.6,
        oracle_row_offset=0.0, oracle_min_band_rows=3, enc_truncate="prefix")
    full_cfg = make_config(tmp_path, tmp_path / "full", epochs=3)
    full_cfg["mask"]["curriculum"] = dict(curriculum)
    assert run(tmp_path, full_cfg) == 0
    part_cfg = make_config(tmp_path, tmp_path / "part", epochs=3)
    part_cfg["mask"]["curriculum"] = dict(curriculum)
    assert run(tmp_path, part_cfg, stop=2) == 0  # internal epochs 0 (bypass) and 1 (matched)
    assert run(tmp_path, exact_from(part_cfg, tmp_path / "part" / "test-last.pth.tar")) == 0
    full = torch.load(tmp_path / "full" / "test-last.pth.tar", weights_only=False)
    resumed = torch.load(tmp_path / "part" / "test-last.pth.tar", weights_only=False)
    assert full["epoch"] == resumed["epoch"] == 3
    for model in ("encoder", "predictor", "target_encoder"):
        for key in full[model]:
            torch.testing.assert_close(full[model][key], resumed[model][key], rtol=0, atol=0)


@pytest.mark.skipif(not os.path.isfile(CB_CONFIG), reason="CB config not generated")
def test_generated_config_builds_the_collator_as_the_trainer_does():
    with open(CB_CONFIG, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    mask = cfg["mask"]
    curr = mask.get("curriculum", {}) or {}
    col = CentroidBudgetRandomCollator(
        input_size=(cfg["data"]["crop_size"],) * 2, patch_size=mask["patch_size"],
        enc_mask_scale=tuple(mask["enc_mask_scale"]), pred_mask_scale=tuple(mask["pred_mask_scale"]),
        aspect_ratio=tuple(mask["aspect_ratio"]), nenc=mask["num_enc_masks"],
        npred=mask["num_pred_masks"], min_keep=mask["min_keep"],
        allow_overlap=mask["allow_overlap"], pred_target_k=mask.get("pred_target_k"),
        curriculum_cfg=curr, rank=0)
    assert col.matching == "strict"
    assert col._kwargs["curriculum_cfg"]["mode"] == "anatomical_prior"
    assert col._kwargs["curriculum_cfg"]["oracle_lateral_frac"] == 0.6


# ----------------------------------------------------------------------
# a few real B-scans (read-only slice cache)
# ----------------------------------------------------------------------


@pytest.mark.skipif(not (os.path.isdir(SLICE_CACHE) and os.path.isdir(DATA_DIR)),
                    reason="real slice cache not available")
def test_real_bscans_match_the_shadow_budgets():
    from src.datasets.oct_slices import OCTSliceDataset
    from src.transforms import make_transforms

    ds = OCTSliceDataset(DATA_DIR, num_slices=100, slice_size=256, slice_cache=SLICE_CACHE,
                         transform=make_transforms(crop_size=256, crop_scale=(0.3, 1.0)))
    items = []
    for idx in (5, 140, 2210, 31050, 47777, 51234, 59999, 300, 12345, 40404, 222, 33333,
                44444, 55555, 1000, 25050):
        seed_all(idx)
        items.append(ds[idx])
    imgs = torch.stack(items)
    col = make("strict", 30)
    for draw in range(3):
        seed_all(draw)
        out = col.collate_masks(imgs, audit=True)
        sizes = out["sizes"]["pred"]
        for i in range(len(items)):
            shadow = budget(out["shadow_enc"], out["shadow_pred"], sizes, i)
            matched = budget(out["masks_enc"], out["masks_pred"], sizes, i)
            assert shadow == matched
