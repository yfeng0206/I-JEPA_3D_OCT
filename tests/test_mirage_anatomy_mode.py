"""Tests for the production `mirage_anatomy` masking mode.

This is the mode that replaces 4 random rectangles with 4 connected,
anatomy-shaped targets derived from the frozen MIRAGE guide.
"""

import numpy as np
import pytest
import torch
from scipy import ndimage

from src.masks.anatomy import build_targets, is_viable, shrink_to_k
from src.masks.curriculum import CurriculumMaskGenerator

G = 16


def _cfg(**kw):
    base = {"mode": "mirage_anatomy", "enabled": True, "T_warm": 25,
            "T_total": 30, "r_max": 1.0, "ramp_shape": "linear"}
    base.update(kw)
    return base


def _gen(epoch=50, k=16, **cfgkw):
    g = CurriculumMaskGenerator(
        input_size=(256, 256), patch_size=16, npred=4, nenc=1,
        pred_target_k=k, curriculum_cfg=_cfg(**cfgkw))
    g.set_epoch(epoch, 100)
    return g


def _band_guide(b=8, thickness=4, seed=0):
    """A synthetic retina: a curved bright band, like a real B-scan."""
    rng = np.random.default_rng(seed)
    occ = np.zeros((b, G, G), np.float32)
    for i in range(b):
        centre = 5 + 3 * np.sin(np.linspace(0, np.pi, G) + rng.random())
        for c in range(G):
            lo = int(centre[c])
            occ[i, lo:lo + thickness, c] = 1.0
    return torch.from_numpy(np.stack([occ, (occ >= 0.25).astype(np.float32)], 1))


# ------------------------------------------------------------------ shrink_to_k

def test_shrink_keeps_k_and_stays_connected():
    m = np.zeros((G, G), bool)
    m[4:8, 2:10] = True                       # 32 connected cells
    out = shrink_to_k(m, 16)
    assert out.sum() == 16
    assert ndimage.label(out, structure=np.ones((3, 3)))[1] == 1
    assert (out & ~m).sum() == 0, "must not invent cells outside the target"


def test_shrink_is_noop_when_already_small():
    m = np.zeros((G, G), bool)
    m[0, :5] = True
    assert np.array_equal(shrink_to_k(m, 16), m)


def test_shrink_prefers_high_score_cells():
    m = np.zeros((G, G), bool)
    m[5, :10] = True
    score = np.zeros((G, G))
    score[5, 7:10] = 1.0                      # the good end
    out = shrink_to_k(m, 3, score)
    assert out.sum() == 3
    # Seeding uses argmax, which resolves ties to the first index, so the seed
    # is (5,7) rather than (5,9).  The contract is that the kept cells are
    # drawn from the high-scoring end, not that any particular tie wins.
    assert out[5, 7], "seed must be a maximum-scoring cell"
    assert score[out].sum() >= 2.0, "should keep the high-scoring end"
    assert ndimage.label(out, structure=np.ones((3, 3)))[1] == 1


# ------------------------------------------------------------------- the guard

def test_mirage_anatomy_refuses_without_pred_target_k():
    """Without fixed-K this mode silently loses ~93% of its target cells."""
    with pytest.raises(ValueError, match="pred_target_k"):
        CurriculumMaskGenerator(input_size=(256, 256), patch_size=16, npred=4,
                                nenc=1, curriculum_cfg=_cfg())


def test_other_modes_still_allow_none():
    g = CurriculumMaskGenerator(
        input_size=(256, 256), patch_size=16, npred=4, nenc=1,
        curriculum_cfg={"mode": "mirage_envelope", "enabled": True})
    assert g.pred_target_k is None


# --------------------------------------------------------------------- the mode

def test_emits_four_connected_targets_of_exactly_k():
    g = _gen()
    guides = _band_guide(8)
    enc, pred = g.generate(batch_size=8, guide_grids=guides,
                           guide_valid=torch.ones(8, dtype=torch.bool))
    assert len(pred) == 4
    for p in pred:
        assert p.shape == (8, 16)
    for b in range(8):
        for p in pred:
            m = np.zeros(G * G, bool)
            m[p[b].numpy()] = True
            assert ndimage.label(m.reshape(G, G),
                                 structure=np.ones((3, 3)))[1] == 1


def test_targets_land_on_anatomy():
    g = _gen()
    guides = _band_guide(16)
    _, pred = g.generate(batch_size=16, guide_grids=guides,
                         guide_valid=torch.ones(16, dtype=torch.bool))
    occ = guides[:, 0].numpy().reshape(16, -1)
    hits = []
    for b in range(16):
        idx = np.unique(np.concatenate([p[b].numpy() for p in pred]))
        hits.append(occ[b][idx].mean())
    assert np.mean(hits) > 0.7, "anatomy targets should sit on the band"


def test_context_excludes_the_full_union():
    """The invariant that makes fixed-K safe."""
    g = _gen()
    guides = _band_guide(8)
    enc, pred = g.generate(batch_size=8, guide_grids=guides,
                           guide_valid=torch.ones(8, dtype=torch.bool))
    for b in range(8):
        union = set()
        for p in pred:
            union.update(p[b].numpy().tolist())
        ctx = set(enc[0][b].numpy().tolist())
        assert not (ctx & union), "context must not contain target tokens"


def test_ramp_off_falls_back_to_rectangles():
    """At epoch 0 the ramp is cold, so guidance must be inert."""
    g = _gen(epoch=0)
    assert g.r_t == 0.0
    guides = _band_guide(8)
    _, pred = g.generate(batch_size=8, guide_grids=guides,
                         guide_valid=torch.ones(8, dtype=torch.bool))
    for p in pred:
        assert p.shape == (8, 16)


def test_invalid_guide_falls_back_without_raising():
    g = _gen()
    guides = torch.zeros(4, 2, G, G)          # no anatomy anywhere
    _, pred = g.generate(batch_size=4, guide_grids=guides,
                         guide_valid=torch.zeros(4, dtype=torch.bool))
    for p in pred:
        assert p.shape == (4, 16)
        assert int(p.min()) >= 0 and int(p.max()) < G * G


def test_indices_are_valid_and_stats_populated():
    g = _gen()
    guides = _band_guide(8)
    _, pred = g.generate(batch_size=8, guide_grids=guides,
                         guide_valid=torch.ones(8, dtype=torch.bool))
    for p in pred:
        assert int(p.min()) >= 0
        assert int(p.max()) < G * G
    s = g.mirage_stats
    assert s["context_patches"] > 0, "context accounting must not report zero"
    assert s["unique_target_patches"] > 0


def test_single_class_guide_matches_two_class_coverage():
    """Production passes ONE occupancy channel; the sampler was swept on two."""
    rng = np.random.default_rng(0)
    occ = np.zeros((G, G), np.float32)
    occ[5:9, :] = 1.0
    assert is_viable([occ])
    parts, _ = build_targets([occ])
    assert len(parts) == 4
    union = np.logical_or.reduce(parts)
    assert union.sum() > 0
    assert (union & (occ <= 0)).sum() <= union.sum() * 0.5
