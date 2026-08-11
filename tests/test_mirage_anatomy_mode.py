"""Tests for the production `mirage_anatomy` masking mode.

This is the mode that replaces 4 random rectangles with 4 connected,
anatomy-shaped targets derived from the frozen MIRAGE guide.
"""

import numpy as np
import pickle
import pytest
import random
import torch
from scipy import ndimage

from src.masks.anatomy import (
    build_targets,
    bridge_diagonals,
    is_viable,
    make_connected,
    n_components,
    n_components4,
    shrink_to_k,
    shrink_to_k_connected,
    trim_to_k_4conn,
)
from src.masks.curriculum import CurriculumMaskGenerator, MirageMaskCollator

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


# ------------------------------------------------- diagonal bridging (4-conn)

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _conn4(m):
    return ndimage.label(m, structure=CROSS)[1]


def test_n_components4_disagrees_with_the_box_rule_on_a_checkerboard():
    """The defect this work fixes: the box rule calls a checkerboard one shape."""
    m = np.zeros((G, G), bool)
    r, c = np.indices((G, G))
    m[(r + c) % 2 == 0] = True               # true checkerboard: diagonals touch
    assert n_components(m) == 1              # 8-connectivity: one component
    assert n_components4(m) == int(m.sum())  # 4-connectivity: all isolated


def test_bridge_closes_a_diagonal_chain():
    m = np.zeros((G, G), bool)
    for i in range(6):
        m[2 + i, 2 + i] = True               # pure diagonal staircase
    assert _conn4(m) == 6
    out, added = bridge_diagonals(m, np.ones((G, G)))
    assert _conn4(out) == 1
    assert added == 5, "one bridge per diagonal step"
    assert (out & m).sum() == m.sum(), "original cells are never removed"


def test_bridge_is_a_noop_when_already_edge_connected():
    m = np.zeros((G, G), bool)
    m[4:8, 2:10] = True
    out, added = bridge_diagonals(m)
    assert added == 0
    assert np.array_equal(out, m)


def test_bridge_stays_inside_the_grid_and_the_bounding_box():
    m = np.zeros((G, G), bool)
    m[0, 0] = m[1, 1] = m[G - 2, G - 2] = m[G - 1, G - 1] = True
    out, _ = bridge_diagonals(m, np.ones((G, G)))
    assert out.shape == m.shape
    rs, cs = np.nonzero(m)
    grown = out & ~m
    gr, gc = np.nonzero(grown)
    assert gr.min() >= rs.min() and gr.max() <= rs.max()
    assert gc.min() >= cs.min() and gc.max() <= cs.max()


def test_bridge_prefers_the_higher_scoring_cell():
    m = np.zeros((G, G), bool)
    m[4, 4] = m[5, 5] = True
    score = np.zeros((G, G))
    score[4, 5] = 0.9                        # this one should be chosen
    score[5, 4] = 0.1
    out, added = bridge_diagonals(m, score)
    assert added == 1
    assert out[4, 5] and not out[5, 4]


def test_trim_reduces_to_k_and_keeps_edge_connectivity():
    m = np.zeros((G, G), bool)
    m[4:8, 2:10] = True                      # 32 cells
    out = trim_to_k_4conn(m, 16, np.random.default_rng(0).random((G, G)))
    assert out.sum() == 16
    assert _conn4(out) == 1
    assert (out & ~m).sum() == 0


def test_trim_drops_the_lowest_scoring_cell_first():
    m = np.zeros((G, G), bool)
    m[4, 2:8] = True                         # a 6-cell bar; ends are removable
    score = np.ones((G, G))
    score[4, 7] = 0.0                        # cheapest end
    out = trim_to_k_4conn(m, 5, score)
    assert out.sum() == 5
    assert not out[4, 7]


def test_shrink_connected_keeps_the_exact_cell_count():
    """Budget must not move: resample_to_k would undo any surplus.

    Uses deterministic 8-connected staircases.  An earlier version of this
    test drew random masks, every one of which was already disconnected under
    8-adjacency, so the loop hit `continue` on all 20 iterations and asserted
    nothing at all.
    """
    checked = 0
    for length in range(3, 12):
        for width in (1, 2, 3):
            m = np.zeros((G, G), bool)
            for i in range(length):                # diagonal staircase
                m[2 + i, 2 + i:2 + i + width] = True
            if n_components(m) != 1:
                continue
            occ = np.zeros((G, G))
            occ[m] = np.linspace(0.2, 1.0, int(m.sum()))
            for k in (4, 8, 16):
                plain = shrink_to_k(m, k, occ)
                conn = shrink_to_k_connected(m, k, occ)
                assert conn.sum() == plain.sum(), (length, width, k)
                assert n_components4(conn) == 1, (length, width, k)
                checked += 1
    assert checked >= 20, "test asserted almost nothing: only %d cases" % checked


def test_bridge_merges_the_most_components_not_the_highest_score():
    """Ranking on score alone is not minimal.

    (0,0),(0,2),(1,1) are three separate edge-components.  Adding (0,1) joins
    all three at once; adding the higher-scoring (1,0) joins only two and
    forces a second cell -- and every extra bridge cell is paid for by a real
    anatomy cell at the trim step.
    """
    m = np.zeros((G, G), bool)
    m[0, 0] = m[0, 2] = m[1, 1] = True
    assert n_components4(m) == 3
    score = np.zeros((G, G))
    score[1, 0] = 1.0                        # highest score, merges only 2
    score[0, 1] = 0.5                        # merges all 3
    out, added = bridge_diagonals(m, score)
    assert n_components4(out) == 1
    assert added == 1, "should need exactly one cell, used %d" % added
    assert out[0, 1]


def test_bridge_respects_forbidden_cells():
    """A bridge must not annex another target's cell."""
    m = np.zeros((G, G), bool)
    m[4, 4] = m[5, 5] = True
    forbid = np.zeros((G, G), bool)
    forbid[4, 5] = True                      # belongs to a neighbouring target
    out, added = bridge_diagonals(m, np.ones((G, G)), forbid=forbid)
    assert not out[4, 5]
    assert out[5, 4] and added == 1
    assert n_components4(out) == 1


def test_bridge_handles_the_anti_diagonal():
    m = np.zeros((G, G), bool)
    m[4, 5] = m[5, 4] = True                 # anti-diagonal pair
    out, added = bridge_diagonals(m, np.ones((G, G)))
    assert added == 1
    assert n_components4(out) == 1


def test_make_connected_never_returns_a_fragmented_mask():
    """Postcondition holds even on input that violates the precondition."""
    m = np.zeros((G, G), bool)
    m[0, 0] = m[1, 1] = True                 # one 8-connected piece
    m[10, 10:13] = True                      # a far-away bar: unreachable
    out = make_connected(m, np.ones((G, G)))
    assert n_components4(out) == 1
    assert out.sum() <= m.sum()


def test_make_connected_is_deterministic():
    rng = np.random.default_rng(0)
    m = np.zeros((G, G), bool)
    for i in range(7):
        m[3 + i, 3 + i] = True
    score = rng.random((G, G))
    a = make_connected(m, score)
    b = make_connected(m, score)
    assert np.array_equal(a, b)


def test_generator_flag_defaults_off_for_checkpoint_compatibility():
    g = _gen()
    assert g.anatomy_bridge_diagonals is False


def test_generator_emits_edge_connected_targets_when_enabled():
    g = _gen(anatomy_bridge_diagonals=True)
    guides = _band_guide(b=8, seed=1)
    enc, pred = g.generate(batch_size=8, guide_grids=guides,
                           guide_valid=torch.ones(8, dtype=torch.bool))
    for p in pred:
        for j in range(8):
            idx = p[j].numpy()
            m = np.zeros(G * G, bool)
            m[idx] = True
            assert _conn4(m.reshape(G, G)) == 1


def test_bridging_does_not_change_the_hidden_budget():
    """Per-target UNIQUE cell counts must match.

    Asserting `p.shape[1]` would be vacuous: `resample_to_k` pads every target
    to exactly K regardless, so that assertion passes even if bridging is
    ignored entirely.  The real quantity is how many DISTINCT cells each
    target hides.
    """
    uniq = {}
    for flag in (False, True):
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        g = _gen(anatomy_bridge_diagonals=flag)
        guides = _band_guide(b=8, seed=2)
        enc, pred = g.generate(batch_size=8, guide_grids=guides,
                               guide_valid=torch.ones(8, dtype=torch.bool))
        uniq[flag] = [len(set(p[j].numpy().tolist()))
                      for p in pred for j in range(8)]
    assert uniq[False] == uniq[True], "hidden budget moved"
    assert sum(uniq[False]) > 0


def test_collator_is_picklable_with_bridging_on():
    """Production pickles the COLLATOR, not the generator.

    `MirageMaskCollator.__getstate__` drops its cached generator, so this is
    the object whose picklability actually matters -- workers are respawned
    every epoch and receive a pickled copy, which is how the ramp reaches them.
    """
    col = MirageMaskCollator(
        input_size=(256, 256), patch_size=16, npred=4, nenc=1,
        pred_target_k=16, curriculum_cfg=_cfg(anatomy_bridge_diagonals=True))
    col.set_epoch(50, 100)
    back = pickle.loads(pickle.dumps(col))
    gen = back._get_generator()
    assert gen.anatomy_bridge_diagonals is True
    guides = _band_guide(b=4, seed=5)
    batch = [(torch.zeros(3, 256, 256), guides[i], torch.tensor(True))
             for i in range(4)]
    imgs, m_enc, m_pred, _ = back(batch)
    for p in m_pred:
        for j in range(4):
            m = np.zeros(G * G, bool)
            m[p[j].numpy()] = True
            assert _conn4(m.reshape(G, G)) == 1
