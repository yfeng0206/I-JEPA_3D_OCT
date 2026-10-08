"""RANDOM-CB: uniform rectangle placement at CENTROID's exact delivered budgets.

Camera-ready control E9 (``mask.curriculum.mode: centroid_budget_random``).
For every training microbatch, inside the DataLoader worker:

1. Draw the batch's rectangle sizes exactly as the stock generator does and
   produce the production CENTROID masks for the real crops (the "shadow":
   ``anatomical_prior`` with the run's own ramp and oracle settings, target
   prefix equalisation to ``K`` and prefix context truncation to ``C``).
2. Read the shadow's budgets: per-target length ``K`` and context length ``C``
   (shared by the batch), and per image the delivered unique-target count
   ``U_i`` and, for ``matching: strict``, the full pre-truncation hidden union
   ``H_i``.  Loss slots ``L = npred * K`` and duplicates ``D_i = L - U_i``
   follow from these.
3. Draw fresh target-rectangle and context-window tuples with the SAME sizes,
   each top-left uniform over its legal positions and without consulting the
   image, and keep the first tuple with delivered union ``U_i`` (and full
   union ``H_i`` when strict) whose context window holds at least ``C``
   non-target cells.  Deliver the targets' row-major ``K`` prefixes and the
   row-major ``C`` prefix of the context window minus the full target union,
   i.e. exactly the stock delivery rules.

The accepted tuple is uniform over all tuples that satisfy the constraints
(conditional-uniform placement).  The shadow's own tuple satisfies them, so a
match always exists.  An image still unmatched after ``cb_max_proposals``
proposals is sampled EXACTLY from the same conditional distribution by
enumerating every target tuple (``exact_fallbacks`` in the batch stats, logged
by the trainer).  Counts, not placements, are what the fallback preserves --
there is no unmatched or CENTROID-placed fallback.

During ramp-0 epochs (``r_t == 0``) the shadow is the stock uniform sampler and
its masks are returned unchanged (bit-identical to RANDOM / CENTROID ramp-0).

RNG: the shadow consumes the worker's global Python/Torch streams exactly as
the stock generator would; the matcher uses a private NumPy generator keyed by
(domain, epoch, rank, worker id/seed, call index, size seed), so it never
advances the global streams.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.masks.curriculum import (
    CENTROID_BUDGET_RANDOM_MODE,
    CurriculumMaskGenerator,
)

CB_MATCHINGS = ("basic", "strict")
CB_SHADOW_MODE = "anatomical_prior"
# Proposals per image before switching to exact enumeration: ~0.4 s of
# rejection at the measured ~5 M proposals/s.  Budgets that reach it are rare
# (low U), and those enumerate in well under a second; see impl_i7_random_cb.md.
DEFAULT_CB_MAX_PROPOSALS = 1 << 21
CB_STATS_SCHEMA = 1
# Domain separator for the private matcher stream.
_CB_RNG_DOMAIN = 0xCB0E9
# Bound on the proposals evaluated per vectorised step (memory, not results).
_MAX_STEP_PROPOSALS = 1 << 17
_ENUM_ROWS = 16
_ENUM_STORE_LIMIT = 1 << 22


if hasattr(np, "bitwise_count"):
    def _popcount(bits: np.ndarray) -> np.ndarray:
        return np.bitwise_count(bits).sum(axis=-1, dtype=np.int32)
else:  # pragma: no cover - NumPy < 2.0
    _LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount(bits: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(bits)
        as_bytes = bits.view(np.uint8).reshape(bits.shape[:-1] + (-1,))
        return _LUT[as_bytes].sum(axis=-1, dtype=np.int32)


def _pack(mask: np.ndarray, nwords: int) -> np.ndarray:
    """(n, P) bool -> (n, nwords) uint64 with bit j = cell j."""
    n, cells = mask.shape
    pad = nwords * 64 - cells
    if pad:
        mask = np.concatenate([mask, np.zeros((n, pad), dtype=bool)], axis=1)
    packed = np.packbits(mask, axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u8").astype(np.uint64).reshape(n, nwords)


def _unpack(bits: np.ndarray, cells: int) -> np.ndarray:
    """(n, nwords) uint64 -> (n, cells) bool."""
    raw = np.ascontiguousarray(bits.astype("<u8")).view(np.uint8)
    return np.unpackbits(raw, axis=-1, bitorder="little")[..., :cells].astype(bool)


class _RectTable:
    """Every legal placement of an (h, w) rectangle on the patch grid.

    Placements are ordered by (top, left); ``idx`` rows are the rectangle's
    row-major (= sorted) patch indices, ``full`` its bitset and ``deliv`` the
    bitset of its first ``k`` indices (the delivered target prefix).
    """

    def __init__(self, h: int, w: int, grid_h: int, grid_w: int, k: Optional[int]):
        self.h, self.w = int(h), int(w)
        self.n_top, self.n_left = grid_h - self.h + 1, grid_w - self.w + 1
        if self.n_top <= 0 or self.n_left <= 0:
            raise ValueError("rectangle %dx%d does not fit the %dx%d grid"
                             % (h, w, grid_h, grid_w))
        self.n = self.n_top * self.n_left
        cells = grid_h * grid_w
        nwords = (cells + 63) // 64
        tops = np.repeat(np.arange(self.n_top), self.n_left)
        lefts = np.tile(np.arange(self.n_left), self.n_top)
        rows = tops[:, None, None] + np.arange(self.h)[None, :, None]
        cols = lefts[:, None, None] + np.arange(self.w)[None, None, :]
        self.idx = (rows * grid_w + cols).reshape(self.n, self.h * self.w).astype(np.int64)
        full = np.zeros((self.n, cells), dtype=bool)
        full[np.arange(self.n)[:, None], self.idx] = True
        self.full = _pack(full, nwords)
        self.cells, self.nwords = cells, nwords
        self.k = None
        if k is not None:
            self._set_prefix(k)

    def _set_prefix(self, k: int) -> None:
        self.k = int(k)
        if not 1 <= self.k <= self.h * self.w:
            raise ValueError("prefix %d outside 1..%d" % (self.k, self.h * self.w))
        deliv = np.zeros((self.n, self.cells), dtype=bool)
        deliv[np.arange(self.n)[:, None], self.idx[:, :self.k]] = True
        self.deliv = _pack(deliv, self.nwords)

    def with_prefix(self, k: int) -> "_RectTable":
        """Same placements (shared arrays) with the delivered ``k``-prefix bitset."""
        tab = copy.copy(self)
        tab._set_prefix(k)
        return tab

    def position_of(self, first_index: np.ndarray, grid_w: int) -> np.ndarray:
        top, left = first_index // grid_w, first_index % grid_w
        if (top >= self.n_top).any() or (left >= self.n_left).any() or (first_index < 0).any():
            raise RuntimeError("shadow target does not start at a legal %dx%d top-left"
                               % (self.h, self.w))
        return top * self.n_left + left


def _or_rows(tables: Sequence[_RectTable], picks: Sequence[np.ndarray], attr: str) -> np.ndarray:
    out = getattr(tables[0], attr)[picks[0]]
    for tab, pick in zip(tables[1:], picks[1:]):
        out = out | getattr(tab, attr)[pick]
    return out


def _combos(tables: Sequence[_RectTable], nwords: int) -> Tuple[np.ndarray, np.ndarray]:
    """Delivered and full unions of every placement combination (last group fastest)."""
    deliv = np.zeros((1, nwords), dtype=np.uint64)
    full = np.zeros((1, nwords), dtype=np.uint64)
    for tab in tables:
        deliv = (deliv[:, None, :] | tab.deliv[None, :, :]).reshape(-1, nwords)
        full = (full[:, None, :] | tab.full[None, :, :]).reshape(-1, nwords)
    return deliv, full


def exact_conditional_sample(pred_tables: Sequence[_RectTable], enc_table: _RectTable,
                             unique: int, hidden: Optional[int], context: int,
                             rng: np.random.Generator) -> Tuple[List[int], int, int]:
    """Draw one (target placements, context placement) tuple uniformly among ALL
    tuples with delivered union ``unique`` (and full union ``hidden`` if given)
    whose context window keeps at least ``context`` non-target cells.

    The target groups are split in two halves whose placement combinations are
    enumerated and joined chunk by chunk.  Pruning is exact: a union is at
    least as large as either part and at most their sum.  Feasible target
    tuples are stored with their number of feasible context windows (the
    weight that makes the joint draw uniform); if they exceed
    ``_ENUM_STORE_LIMIT`` a second pass locates the drawn tuple instead.
    Returns (positions per target group, context position, n feasible tuples).
    """
    nwords = enc_table.full.shape[1]
    half = len(pred_tables) // 2
    group_a, group_b = pred_tables[:half], pred_tables[half:]
    deliv_a, full_a = _combos(group_a, nwords)
    deliv_b, full_b = _combos(group_b, nwords)
    size_a, size_b = _popcount(deliv_a), _popcount(deliv_b)
    keep_a, keep_b = size_a <= unique, size_b <= unique
    if hidden is not None:
        keep_a &= _popcount(full_a) <= hidden
        keep_b &= _popcount(full_b) <= hidden
    ids_a = np.flatnonzero(keep_a)
    ids_a = ids_a[np.argsort(size_a[ids_a], kind="stable")]
    ids_b = np.flatnonzero(keep_b)
    ids_b = ids_b[np.argsort(size_b[ids_b], kind="stable")]
    sorted_b = size_b[ids_b]
    da, fa, db, fb = deliv_a[ids_a], full_a[ids_a], deliv_b[ids_b], full_b[ids_b]
    max_a = size_a[ids_a]
    enc_full = enc_table.full

    def chunk(start):
        stop = min(start + _ENUM_ROWS, len(ids_a))
        lo = int(np.searchsorted(sorted_b, unique - int(max_a[stop - 1]), side="left"))
        ok = _popcount(da[start:stop, None, :] | db[None, lo:, :]) == unique
        ia, ib = np.nonzero(ok)
        if not ia.size:
            return None
        ia, ib = ia + start, ib + lo
        raw = fa[ia] | fb[ib]
        if hidden is not None:
            good = _popcount(raw) == hidden
            ia, ib, raw = ia[good], ib[good], raw[good]
        weight = (_popcount(enc_full[None, :, :] & ~raw[:, None, :]) >= context).sum(axis=1)
        nz = weight > 0
        return ia[nz], ib[nz], weight[nz], raw[nz]

    starts = range(0, len(ids_a), _ENUM_ROWS)
    stored, n_stored, total = [], 0, 0
    for start in starts:
        found = chunk(start)
        if found is None:
            continue
        total += int(found[2].sum())
        if stored is not None:
            stored.append(found)
            n_stored += found[0].size
            if n_stored > _ENUM_STORE_LIMIT:
                stored = None
    if total == 0:
        raise RuntimeError(
            "centroid_budget_random: no uniform tuple reaches U=%d H=%s C=%d; the shadow "
            "budget is infeasible for its own sizes" % (unique, hidden, context))
    pick = int(rng.integers(total))

    def locate(parts, pick):
        seen = 0
        for ia, ib, weight, raw in parts:
            here = int(weight.sum())
            if seen + here <= pick:
                seen += here
                continue
            cum = np.cumsum(weight)
            j = int(np.searchsorted(cum, pick - seen, side="right"))
            offset = pick - seen - (int(cum[j - 1]) if j else 0)
            windows = np.flatnonzero(_popcount(enc_full & ~raw[j]) >= context)
            return int(ia[j]), int(ib[j]), int(windows[offset])
        raise AssertionError("exact sampler lost its pick")  # pragma: no cover

    if stored is not None:
        a, b, e = locate(stored, pick)
    else:
        a, b, e = locate((f for f in map(chunk, starts) if f is not None), pick)
    pos_a = np.unravel_index(int(ids_a[a]), [t.n for t in group_a]) if group_a else ()
    pos_b = np.unravel_index(int(ids_b[b]), [t.n for t in group_b]) if group_b else ()
    return [int(p) for p in pos_a] + [int(p) for p in pos_b], e, total


def match_budgets(pred_tables: Sequence[_RectTable], enc_table: _RectTable, context: int,
                  unique: np.ndarray, hidden: Optional[np.ndarray], rng: np.random.Generator,
                  max_proposals: int = DEFAULT_CB_MAX_PROPOSALS):
    """Conditional-uniform placements for every image.

    Returns (positions (B, npred), context positions (B,), proposals (B,),
    list of images resolved by exact enumeration).
    """
    batch = len(unique)
    npred = len(pred_tables)
    positions = np.full((batch, npred), -1, dtype=np.int64)
    enc_positions = np.full(batch, -1, dtype=np.int64)
    proposals = np.zeros(batch, dtype=np.int64)
    exact: List[int] = []
    todo = np.arange(batch)
    width = min(64, max_proposals)
    while todo.size:
        # Unresolved rows have all used the same number of proposals; never draw
        # past the cap, so an image that needs more always takes the logged exact path.
        width = max(1, min(width, _MAX_STEP_PROPOSALS // todo.size,
                           max_proposals - int(proposals[todo].max())))
        picks = [rng.integers(tab.n, size=(todo.size, width)) for tab in pred_tables]
        enc_pick = rng.integers(enc_table.n, size=(todo.size, width))
        ok = _popcount(_or_rows(pred_tables, picks, "deliv")) == unique[todo][:, None]
        rows, cols = np.nonzero(ok)
        if rows.size:
            raw = _or_rows(pred_tables, [p[rows, cols] for p in picks], "full")
            good = np.ones(rows.size, dtype=bool)
            if hidden is not None:
                good &= _popcount(raw) == hidden[todo[rows]]
            good &= _popcount(enc_table.full[enc_pick[rows, cols]] & ~raw) >= context
            rows, cols = rows[good], cols[good]
        done = np.zeros(todo.size, dtype=bool)
        if rows.size:
            # np.nonzero is row-major, so the first hit per row is its earliest proposal.
            first_rows, first_at = np.unique(rows, return_index=True)
            first_cols = cols[first_at]
            images = todo[first_rows]
            for p in range(npred):
                positions[images, p] = picks[p][first_rows, first_cols]
            enc_positions[images] = enc_pick[first_rows, first_cols]
            proposals[images] += first_cols + 1
            done[first_rows] = True
        proposals[todo[~done]] += width
        todo = todo[~done]
        capped = proposals[todo] >= max_proposals
        for image in todo[capped]:
            pos, enc_pos, _ = exact_conditional_sample(
                pred_tables, enc_table, int(unique[image]),
                None if hidden is None else int(hidden[image]), context, rng)
            positions[image] = pos
            enc_positions[image] = enc_pos
            exact.append(int(image))
        todo = todo[~capped]
        width *= 2
    return positions, enc_positions, proposals, exact


def validate_cb_config(curriculum_cfg: dict, *, pred_target_k=None, allow_overlap=False,
                       nenc=1) -> Tuple[str, int]:
    """Check the RANDOM-CB keys eagerly; returns (matching, max_proposals)."""
    cfg = curriculum_cfg or {}
    if cfg.get("mode") != CENTROID_BUDGET_RANDOM_MODE:
        raise ValueError("curriculum.mode must be %r for the RANDOM-CB collator; got %r"
                         % (CENTROID_BUDGET_RANDOM_MODE, cfg.get("mode")))
    matching = cfg.get("matching")
    if matching not in CB_MATCHINGS:
        raise ValueError("curriculum.matching must be one of %s for %s; got %r"
                         % (CB_MATCHINGS, CENTROID_BUDGET_RANDOM_MODE, matching))
    if str(cfg.get("enc_truncate", "prefix")) != "prefix":
        raise ValueError("%s delivers the stock row-major context prefix; "
                         "curriculum.enc_truncate must be 'prefix'" % CENTROID_BUDGET_RANDOM_MODE)
    if pred_target_k is not None:
        raise ValueError("%s matches rectangle prefixes; mask.pred_target_k must be unset"
                         % CENTROID_BUDGET_RANDOM_MODE)
    if allow_overlap:
        raise ValueError("%s requires allow_overlap: false" % CENTROID_BUDGET_RANDOM_MODE)
    if int(nenc) != 1:
        raise ValueError("%s supports exactly one context group (num_enc_masks: 1)"
                         % CENTROID_BUDGET_RANDOM_MODE)
    max_proposals = int(cfg.get("cb_max_proposals", DEFAULT_CB_MAX_PROPOSALS))
    if max_proposals < 1:
        raise ValueError("curriculum.cb_max_proposals must be >= 1")
    return str(matching), max_proposals


def shadow_curriculum_cfg(curriculum_cfg: dict) -> dict:
    """The CENTROID configuration the shadow runs: same keys, production mode."""
    cfg = dict(curriculum_cfg or {})
    cfg["mode"] = CB_SHADOW_MODE
    for key in ("matching", "cb_max_proposals"):
        cfg.pop(key, None)
    cfg["audit_masks"] = False
    return cfg


class CentroidBudgetRandomCollator:
    """DataLoader ``collate_fn`` for ``centroid_budget_random`` (RANDOM-CB).

    Constructed with the same keyword arguments as ``MirageMaskCollator`` /
    ``CurriculumMaskGenerator`` (``curriculum_cfg`` = the whole
    ``mask.curriculum`` dict).  Returns ``(images, masks_enc, masks_pred,
    stats)``.  Like ``MirageMaskCollator`` it must be re-pickled into the
    workers every epoch (``set_epoch`` before the iterator is created; no
    persistent workers).
    """

    def __init__(self, **generator_kwargs):
        curriculum_cfg = generator_kwargs.get("curriculum_cfg") or {}
        self.matching, self.max_proposals = validate_cb_config(
            curriculum_cfg,
            pred_target_k=generator_kwargs.get("pred_target_k"),
            allow_overlap=bool(generator_kwargs.get("allow_overlap", False)),
            nenc=int(generator_kwargs.get("nenc", 1)),
        )
        kwargs = dict(generator_kwargs)
        kwargs["curriculum_cfg"] = shadow_curriculum_cfg(curriculum_cfg)
        kwargs.pop("world_size", None)
        kwargs.pop("device", None)
        self.rank = int(kwargs.pop("rank", 0))
        self._kwargs = kwargs
        self._generator: Optional[CurriculumMaskGenerator] = None
        self._tables: Dict[tuple, _RectTable] = {}
        self.epoch = 0
        self.total_epochs: Optional[int] = None
        self._calls = 0
        # Fail at startup, not in a worker: the shadow must build.
        CurriculumMaskGenerator(**kwargs)

    # -- epoch / pickling ---------------------------------------------------
    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        epoch = int(epoch)
        if epoch != self.epoch:
            # The call index keys the private stream within an epoch only, so a
            # zero-worker exact resume (fresh collator) replays the same masks.
            self._calls = 0
        self.epoch = epoch
        self.total_epochs = total_epochs

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_generator"] = None  # holds a torch.Generator; rebuilt per worker
        state["_tables"] = {}
        return state

    def _get_generator(self) -> CurriculumMaskGenerator:
        if self._generator is None:
            self._generator = CurriculumMaskGenerator(**self._kwargs)
        self._generator.set_epoch(self.epoch, self.total_epochs)
        return self._generator

    def _table(self, h: int, w: int, k: Optional[int]) -> _RectTable:
        key = (int(h), int(w), None if k is None else int(k))
        tab = self._tables.get(key)
        if tab is None:
            if len(self._tables) >= 512:
                self._tables.clear()
            base = self._tables.get(key[:2] + (None,))
            if base is None:
                gen = self._generator
                base = _RectTable(h, w, gen.height, gen.width, None)
                self._tables[key[:2] + (None,)] = base
            tab = base if k is None else base.with_prefix(k)
            self._tables[key] = tab
        return tab

    def _matching_rng(self, size_seed: Optional[int]) -> np.random.Generator:
        info = torch.utils.data.get_worker_info()
        worker = int(info.id) + 1 if info is not None else 0
        base = int(info.seed) if info is not None else int(torch.initial_seed())
        entropy = [_CB_RNG_DOMAIN, self.epoch, self.rank, worker,
                   base & 0xFFFFFFFF, (base >> 32) & 0xFFFFFFFF, self._calls,
                   0 if size_seed is None else int(size_seed) + 1]
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))

    # -- collation ------------------------------------------------------------
    def __call__(self, batch, *, block_sizes=None):
        items = [item[0] if isinstance(item, (tuple, list)) else item for item in batch]
        images = torch.stack(items, dim=0)
        out = self.collate_masks(images, block_sizes=block_sizes)
        return images, out["masks_enc"], out["masks_pred"], out["stats"]

    def collate_masks(self, images: torch.Tensor, block_sizes=None, audit: bool = False) -> dict:
        """Shadow CENTROID masks, then the budget-matched uniform masks.

        ``audit=True`` also returns the shadow masks and per-image counts.
        """
        t0 = time.perf_counter()
        gen = self._get_generator()
        batch = int(images.size(0))
        size_seed = None
        if block_sizes is None:
            # Exactly generate()'s own draw, so the shadow consumes the global
            # stream like the stock sampler and the sizes are known here.
            size_seed = random.randint(0, 2 ** 31)
            gen._size_gen.manual_seed(size_seed)
            pred_sizes = [gen._sample_block_size(gen.pred_mask_scale, gen._size_gen)
                          for _ in range(gen.npred)]
            enc_sizes = [gen._sample_block_size(gen.enc_mask_scale, gen._size_gen)
                         for _ in range(gen.nenc)]
        else:
            pred_sizes = [tuple(int(v) for v in s) for s in block_sizes["pred"]]
            enc_sizes = [tuple(int(v) for v in s) for s in block_sizes["enc"]]
        sizes = {"pred": pred_sizes, "enc": enc_sizes}
        shadow_enc, shadow_pred = gen.generate(batch, imgs_cpu=images, block_sizes=sizes)
        self._calls += 1
        t1 = time.perf_counter()
        k = int(shadow_pred[0].shape[1])
        c = int(shadow_enc[0].shape[1])
        stats = {
            "cb_schema": CB_STATS_SCHEMA, "mode": CENTROID_BUDGET_RANDOM_MODE,
            "matching": self.matching, "epoch": self.epoch, "r_t": float(gen.r_t),
            "images": batch, "target_len": k, "context": c,
            "loss_slots": gen.npred * k, "bypass_ramp0": 0,
            "proposals_mean": 0.0, "proposals_max": 0, "exact_fallbacks": 0,
            "shadow_ms": (t1 - t0) * 1000.0, "match_ms": 0.0,
        }
        pred_tables = [self._table(h, w, k) for h, w in pred_sizes]
        enc_table = self._table(enc_sizes[0][0], enc_sizes[0][1], None)
        unique, hidden = self._shadow_budget(shadow_enc, shadow_pred, pred_tables)
        stats.update(unique_targets=float(unique.mean()),
                     duplicates=float(gen.npred * k - unique.mean()),
                     hidden_shadow=float(hidden.mean()))
        result = {"sizes": sizes, "stats": stats}
        if audit:
            result.update(shadow_enc=shadow_enc, shadow_pred=shadow_pred,
                          shadow_unique=unique.copy(), shadow_hidden=hidden.copy())
        if gen.r_t <= 0.0:
            # Ramp-0: the shadow IS the stock uniform sampler.  No matching.
            stats.update(bypass_ramp0=1, hidden_matched=float(hidden.mean()))
            result.update(masks_enc=shadow_enc, masks_pred=shadow_pred,
                          matched_unique=unique, matched_hidden=hidden)
            return result
        rng = self._matching_rng(size_seed)
        strict = self.matching == "strict"
        positions, enc_positions, proposals, exact = match_budgets(
            pred_tables, enc_table, c, unique, hidden if strict else None, rng,
            self.max_proposals)
        masks_pred = [torch.from_numpy(tab.idx[positions[:, p], :k].copy())
                      for p, tab in enumerate(pred_tables)]
        raw = _or_rows(pred_tables, [positions[:, p] for p in range(len(pred_tables))], "full")
        avail = _unpack(enc_table.full[enc_positions] & ~raw, gen.height * gen.width)
        context = np.stack([np.flatnonzero(row)[:c] for row in avail]).astype(np.int64)
        masks_enc = [torch.from_numpy(context)]
        matched_unique, matched_hidden = self._verify(
            masks_enc, masks_pred, raw, unique, hidden if strict else None, k, c)
        stats.update(hidden_matched=float(matched_hidden.mean()),
                     proposals_mean=float(proposals.mean()),
                     proposals_max=int(proposals.max()),
                     exact_fallbacks=len(exact),
                     match_ms=(time.perf_counter() - t1) * 1000.0)
        result.update(masks_enc=masks_enc, masks_pred=masks_pred, proposals=proposals,
                      exact_fallback_images=exact, matched_unique=matched_unique,
                      matched_hidden=matched_hidden)
        return result

    def _shadow_budget(self, shadow_enc, shadow_pred, pred_tables):
        """Per-image U (delivered) and H (full union) of the shadow's rectangles.

        Each delivered shadow target is the row-major prefix of a rectangle of
        known size, so its first index is the top-left and the full rectangle
        is recovered exactly; this is checked, not assumed.
        """
        width = self._generator.width
        positions = []
        for p, tab in enumerate(pred_tables):
            delivered = shadow_pred[p].numpy()
            pos = tab.position_of(delivered[:, 0], width)
            if not np.array_equal(tab.idx[pos, :tab.k], delivered):
                raise RuntimeError("shadow target group %d is not a %dx%d rectangle prefix"
                                   % (p, tab.h, tab.w))
            positions.append(pos)
        unique = _popcount(_or_rows(pred_tables, positions, "deliv"))
        raw = _or_rows(pred_tables, positions, "full")
        hidden = _popcount(raw)
        hidden_cells = _unpack(raw, self._generator.num_patches)
        enc = shadow_enc[0].numpy()
        if hidden_cells[np.arange(enc.shape[0])[:, None], enc].any():
            raise RuntimeError("shadow context overlaps its own target union")
        return unique, hidden

    def _verify(self, masks_enc, masks_pred, raw, unique, hidden, k, c):
        """Recount the delivered tensors; any mismatch is a bug, so raise."""
        cells = self._generator.num_patches
        batch = masks_enc[0].shape[0]
        rows = np.arange(batch)[:, None]
        delivered = np.zeros((batch, cells), dtype=bool)
        for group in masks_pred:
            g = group.numpy()
            if g.shape != (batch, k) or g.min() < 0 or g.max() >= cells:
                raise RuntimeError("matched target tensor has the wrong shape or range")
            delivered[rows, g] = True
        got_unique = delivered.sum(axis=1)
        got_hidden = _popcount(raw)
        enc = masks_enc[0].numpy()
        if enc.shape != (batch, c) or enc.min() < 0 or enc.max() >= cells:
            raise RuntimeError("matched context tensor has the wrong shape or range")
        if c > 1 and (np.diff(enc, axis=1) <= 0).any():
            raise RuntimeError("matched context is not strictly increasing")
        if _unpack(raw, cells)[rows, enc].any():
            raise RuntimeError("matched context overlaps the target union")
        if not np.array_equal(got_unique, unique):
            raise RuntimeError("matched unique-target counts differ from the shadow")
        if hidden is not None and not np.array_equal(got_hidden, hidden):
            raise RuntimeError("matched hidden-union counts differ from the shadow")
        return got_unique, got_hidden
