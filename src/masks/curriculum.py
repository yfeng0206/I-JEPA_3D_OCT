"""Curriculum mask generator for I-JEPA on OCT data.

Drop-in alternative to ``src.masks.multiblock.MaskCollator`` with the same
output contract:

    generate(batch_size, imgs_cpu=None, h_for_cluster=None)
        -> (masks_enc, masks_pred)
    masks_enc:  list of ``nenc`` LongTensors, each (B, K_enc_global_min)
    masks_pred: list of ``npred`` LongTensors, each (B, K_pred_global_min)

The key differences from the random multiblock collator:

* Pred-block *top-left location* can be biased toward "important" patches.
* The encoder block sampling is bit-identical to ``MaskCollator`` so the
  ablation only changes the predictor-target distribution (R1 vs R2/R3a/R3b
  remain directly comparable on the encoder side).
* State (loss-EMA grid, cluster centroids, per-cluster loss EMA) is held on
  the generator and updated via :meth:`update_after_iter` from the training
  loop.  Updates use DDP ``all_reduce`` unconditionally on every rank.

Modes
-----
* ``loss_guided``  (R2) — bias by per-position predictor-L2 EMA.
* ``intensity_foreground``  (R3a) — bias by per-image patch-mean intensity
  (>= ``intensity_quantile``).  No learned state.
* ``cluster_foreground``  (R3b) — bias by per-image cluster assignment
  (foreground clusters chosen by top-half per-cluster loss EMA).

The fraction of pred blocks that are biased on a given iter is governed by
``r_t``, ramped from 0 at ``T_warm`` to ``r_max`` at ``T_total``.
``r_t`` is consumed as a Bernoulli probability per pred-block (NOT
``round(r_t * npred)``) so small ``r_t`` values still have an effect.

Design fixes baked in (rubber-duck pass, P0/P1):
* P0-1: For ``cluster_foreground``, the caller is responsible for passing
  ``h_for_cluster=h_full`` (already computed by the target encoder forward
  that is moved OUTSIDE ``_forward_backward``).
* P0-2: Encoder sampling matches multiblock exactly — only pred top-left is
  biased.
* P0-5: ``update_after_iter`` expects ``per_token_loss`` as
  ``(B*npred*nenc, K_pred)`` and reshapes to ``(npred, nenc, B, K_pred)``
  internally, averaging over ``nenc``.
* P1: EMA is configured by half-life in OPTIMIZER STEPS (not raw alpha),
  cadence is keyed off ``is_step`` (not ``_iter``), and biased sampling is
  gated until per-position / per-cluster maturity thresholds are reached.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from src.masks.utils import resample_to_k
from src.masks.anatomy import (
    build_targets as anatomy_build_targets,
    is_viable as anatomy_is_viable,
    shrink_to_k as anatomy_shrink_to_k,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _half_life_to_alpha(half_life_steps: float) -> float:
    """Convert a desired half-life (in optimizer steps) to an EMA decay.

    new = alpha * old + (1 - alpha) * incoming, so the value's "memory" decays
    to 50 % after ``half_life_steps`` updates.
    """
    if half_life_steps <= 0:
        return 0.0
    return math.exp(math.log(0.5) / float(half_life_steps))


def _maybe_all_reduce(tensor: torch.Tensor, op=None) -> None:
    """``dist.all_reduce`` that's a no-op when DDP isn't initialized.

    Called on EVERY rank unconditionally (never inside ``if is_main:``).
    """
    if dist.is_available() and dist.is_initialized():
        if op is None:
            op = dist.ReduceOp.SUM
        dist.all_reduce(tensor, op=op)


def _broadcast(tensor: torch.Tensor, src: int = 0) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src=src)


# ----------------------------------------------------------------------
# CurriculumMaskGenerator
# ----------------------------------------------------------------------


def _window_sums_numpy(grid: np.ndarray, block_h: int, block_w: int):
    """Block sums for every (block_h, block_w) window via a summed-area table.

    NumPy rather than torch: the mask grid is 16x16, where per-op dispatch
    overhead dominates and torch is roughly an order of magnitude slower.
    """
    height, width = grid.shape
    n_top, n_left = height - block_h + 1, width - block_w + 1
    if n_top <= 0 or n_left <= 0:
        return None
    padded = np.zeros((height + 1, width + 1), dtype=np.float64)
    padded[1:, 1:] = grid
    sat = padded.cumsum(axis=0).cumsum(axis=1)
    return (
        sat[block_h:, block_w:]
        - sat[:n_top, block_w:]
        - sat[block_h:, :n_left]
        + sat[:n_top, :n_left]
    )


class MirageMaskCollator:
    """Generate I-JEPA masks inside DataLoader workers for ``mirage_envelope``.

    Finding target rectangles that satisfy the fill and retina-visibility rules
    is rejection sampling and costs more CPU time than the GPU step itself.  Run
    inline it leaves the GPU idle; run as a ``collate_fn`` it executes in the
    worker processes and overlaps with compute, so the search becomes free.

    This is safe only because ``mirage_envelope`` masks depend on nothing the
    model produces -- unlike ``cluster_foreground``, which needs the teacher's
    features and must therefore stay in the training loop.

    ``set_epoch`` must be called BEFORE the epoch's iterator is created: workers
    receive a pickled copy of this object at that moment, which is how the ramp
    value reaches them.
    """

    def __init__(self, **generator_kwargs):
        self._kwargs = generator_kwargs
        self._generator: Optional["CurriculumMaskGenerator"] = None
        self.epoch = 0
        self.total_epochs: Optional[int] = None

    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        self.epoch = int(epoch)
        self.total_epochs = total_epochs

    def __getstate__(self):
        # torch.Generator does not pickle; rebuild the generator per worker.
        state = self.__dict__.copy()
        state["_generator"] = None
        return state

    def _get_generator(self) -> "CurriculumMaskGenerator":
        if self._generator is None:
            self._generator = CurriculumMaskGenerator(**self._kwargs)
        self._generator.set_epoch(self.epoch, self.total_epochs)
        return self._generator

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        guides = torch.stack([item[1] for item in batch], dim=0)
        valid = torch.stack([item[2] for item in batch], dim=0)
        generator = self._get_generator()
        masks_enc, masks_pred = generator.generate(
            batch_size=images.size(0),
            guide_grids=guides,
            guide_valid=valid,
        )
        return images, masks_enc, masks_pred, generator.mirage_stats


class CurriculumMaskGenerator:
    """Stateful mask generator with optional curriculum biasing."""

    VALID_MODES = (
        "loss_guided",
        "intensity_foreground",
        "cluster_foreground",
        "anatomical_prior",
        "mirage_envelope",
        "mirage_anatomy",
    )

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        patch_size: int = 16,
        enc_mask_scale: Tuple[float, float] = (0.85, 1.0),
        pred_mask_scale: Tuple[float, float] = (0.15, 0.2),
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        nenc: int = 1,
        npred: int = 4,
        min_keep: int = 10,
        allow_overlap: bool = False,
        pred_target_k: Optional[int] = None,
        *,
        curriculum_cfg: dict,
        world_size: int = 1,
        rank: int = 0,
        device: Optional[torch.device] = None,
    ):
        # --- Static knobs (same contract as multiblock.MaskCollator) ---
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        # When set, every predictor target is resampled to exactly this many
        # indices instead of the whole batch being front-sliced to its
        # minimum.  Off by default: rectangular targets are all the same size,
        # so the min-truncate is harmless there and the old behaviour stays
        # bit-identical.  Irregular anatomy targets are NOT the same size and
        # lose 92.8% of their cells without this.
        self.pred_target_k = pred_target_k
        self.num_patches = self.height * self.width

        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap

        self.world_size = world_size
        self.rank = rank

        # ``device`` is where all curriculum state tensors and collective
        # buffers live.  Defaults to CPU for single-process / unit-test use;
        # the trainer passes the per-rank CUDA device so that NCCL all-reduce
        # / broadcast see CUDA tensors (NCCL backend cannot operate on CPU
        # tensors — would crash at first collective).
        if device is None:
            device = torch.device("cpu")
        self.device = torch.device(device)

        # --- Curriculum config ---
        cfg = dict(curriculum_cfg or {})
        self.mode = cfg.get("mode", "loss_guided")
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"curriculum.mode must be one of {self.VALID_MODES}, "
                f"got {self.mode!r}"
            )
        self.T_warm = int(cfg.get("T_warm", 25))
        self.T_total = int(cfg.get("T_total", 100))
        # When T_total is set explicitly in the config, it WINS over the
        # ``total_epochs`` the training loop passes to set_epoch().  Without
        # this, set_epoch(epoch, opt_cfg['epochs']) would silently overwrite an
        # explicit short ramp (e.g. the oracle's hard-switch T_total=30) with
        # the full run length, turning a quick hard-switch into a slow full-run
        # ramp.  Configs that omit T_total keep the legacy full-run behaviour.
        self._t_total_explicit = "T_total" in cfg
        self.r_max = float(cfg.get("r_max", 0.5))
        self.ramp_shape = cfg.get("ramp_shape", "linear")
        if self.ramp_shape not in ("linear", "cosine"):
            raise ValueError("ramp_shape must be 'linear' or 'cosine'")

        # EMA half-lives expressed in optimizer steps
        self.loss_map_halflife = float(cfg.get("loss_map_halflife_steps", 2000.0))
        self.cluster_centroid_halflife = float(
            cfg.get("cluster_centroid_halflife_steps", 2000.0)
        )
        self.cluster_loss_halflife = float(
            cfg.get("cluster_loss_halflife_steps", 2000.0)
        )
        self._loss_alpha = _half_life_to_alpha(self.loss_map_halflife)
        self._cluster_centroid_alpha = _half_life_to_alpha(
            self.cluster_centroid_halflife
        )
        self._cluster_loss_alpha = _half_life_to_alpha(self.cluster_loss_halflife)

        # Maturity gates — biased sampling stays uniform until satisfied.
        # ``loss_maturity_min_count`` — every (16,16) cell must have been
        # observed at least this many times (cumulative all-reduced).
        self.loss_maturity_min_count = float(
            cfg.get("loss_maturity_min_count", 32.0)
        )
        # ``cluster_maturity_min_count`` — every cluster's loss EMA must have
        # accumulated at least this many observations.
        self.cluster_maturity_min_count = float(
            cfg.get("cluster_maturity_min_count", 64.0)
        )

        # R3a — intensity threshold
        self.intensity_quantile = float(cfg.get("intensity_quantile", 0.6))

        # ORACLE (anatomical_prior, "v2" retina-following band).  Stateless,
        # per-slice.  A fixed-size window CENTERED on the retina (intensity-
        # weighted row centroid) so it follows the band's vertical position
        # per slice while guaranteeing the region size stays in the 20-40%
        # band (so the 4 fixed target blocks fit without piling up).
        #   oracle_region_frac:   target fraction of patches in the masked band.
        #   oracle_lateral_frac:  fraction of the lateral (x) extent kept
        #                         (ignore left/right edges).
        #   oracle_row_offset:    fraction of H to shift the band centre; <0
        #                         shifts UP toward RNFL for a v3 glaucoma focus,
        #                         0 = centred on the whole retinal band.
        #   oracle_min_band_rows: floor on band height (rows) so it is never a
        #                         razor-thin strip.
        self.oracle_region_frac = float(cfg.get("oracle_region_frac", 0.28))
        self.oracle_lateral_frac = float(cfg.get("oracle_lateral_frac", 0.8))
        self.oracle_row_offset = float(cfg.get("oracle_row_offset", 0.0))
        self.oracle_min_band_rows = int(cfg.get("oracle_min_band_rows", 3))

        # R3b — clustering.  Accept ``K`` as an alias for ``n_clusters`` so
        # the config can use either name.
        self.n_clusters = int(cfg.get("n_clusters", cfg.get("K", 4)))
        self.foreground_cluster_ids: Optional[Sequence[int]] = cfg.get(
            "foreground_cluster_ids", None
        )
        # How to pick foreground clusters when ``foreground_cluster_ids`` is
        # not explicitly set.  Two strategies:
        #   "top_half":   pick K/2 hardest clusters by absolute loss EMA.
        #                 Default; matches Self-Guided MAE (Wang 2024).
        #                 Risk: if all clusters have similar loss, the choice
        #                 is essentially random — the clustering wasn't
        #                 finding signal-bearing structure.
        #   "above_mean": pick clusters whose loss EMA exceeds the across-
        #                 cluster mean by ``foreground_loss_zscore`` * std.
        #                 Falls back to top-1 (the hardest cluster) when no
        #                 cluster crosses the threshold — never returns empty.
        #                 More principled when the user is unsure whether
        #                 K=4 is finding meaningful semantic groups.
        self.cluster_foreground_selection = str(
            cfg.get("cluster_foreground_selection", "top_half")
        )
        if self.cluster_foreground_selection not in ("top_half", "above_mean"):
            raise ValueError(
                "cluster_foreground_selection must be 'top_half' or "
                f"'above_mean'; got {self.cluster_foreground_selection!r}"
            )
        # Z-score threshold for "above_mean" mode.  0.5 = +0.5 std above the
        # cross-cluster mean qualifies as foreground.
        self.foreground_loss_zscore = float(
            cfg.get("foreground_loss_zscore", 0.5)
        )

        # Numerical floor for sampling weights so we never have all-zero rows.
        self.weight_eps = float(cfg.get("weight_eps", 1e-6))

        # MIRAGE envelope mode.  Target blocks keep the standard geometry; a
        # block is only admissible when at least ``mirage_min_block_fill`` of
        # its patches lie on the MIRAGE retinal region, and a placement is
        # preferred when at least ``mirage_min_retina_visible`` of the region
        # survives outside the targets so the encoder retains anatomy to reason
        # from.
        #
        # Fallback semantics, which differ per rule -- do not conflate them:
        #   * invalid guide            -> uniform random placement (all blocks)
        #   * no admissible window for
        #     a block (infeasible)     -> uniform random placement, THAT BLOCK
        #                                 only, and the retry loop stops early
        #                                 because admissibility cannot change
        #   * visibility never reached -> NO fallback.  After
        #                                 ``mirage_max_attempts`` the attempt
        #                                 that left the most retina visible is
        #                                 returned and every block stays guided.
        #
        # So ``mirage_min_retina_visible`` is a best-effort retry preference,
        # not a guarantee: measured over 1,000 volumes it is met outright on
        # ~47% of images.  Falling back to uniform there would inject
        # random-baseline behaviour into the guided arm and weaken the very
        # contrast being tested, so it is deliberately not done.  The 0.25
        # threshold policy was calibrated with these exact semantics in force.
        self.mirage_min_block_fill = float(cfg.get("mirage_min_block_fill", 0.40))
        self.mirage_min_retina_visible = float(
            cfg.get("mirage_min_retina_visible", 0.25)
        )
        self.mirage_max_attempts = int(cfg.get("mirage_max_attempts", 30))
        self.mirage_occupancy_threshold = float(
            cfg.get("mirage_occupancy_threshold", 0.5)
        )
        self.mirage_spread = bool(cfg.get("mirage_spread", True))
        self.mirage_overlap_tolerance = float(
            cfg.get("mirage_overlap_tolerance", 0.25)
        )
        # --- mirage_anatomy mode ---
        # mass_cap is how much of the guide's total anatomy score the targets
        # must cover; tau is the floor below which a patch is not considered
        # to support anatomy at all.  0.90/0.10 are the swept defaults.
        self.anatomy_mass_cap = float(cfg.get("anatomy_mass_cap", 0.90))
        self.anatomy_tau = float(cfg.get("anatomy_tau", 0.10))

        if self.mode == "mirage_anatomy" and self.pred_target_k is None:
            # Anatomy targets are ragged by construction (p05 = 6 cells) and
            # the fallback rectangles are ~21, so the global-min truncation
            # would collapse every target in the microbatch to the smallest
            # one.  Measured at batch 64 that retains 7.2% of target cells and
            # hits K=1 in 99.8% of batches -- a silent, catastrophic
            # degradation with no error raised.  Refuse rather than train
            # something that only looks like anatomy masking.
            raise ValueError(
                "mode 'mirage_anatomy' requires pred_target_k to be set "
                "(e.g. 16). Without it the collator falls back to global-min "
                "truncation, which discards ~93% of the anatomy target cells."
            )

        # --- State ---
        self._r_t = 0.0
        self._epoch = 0
        self._iter = 0

        # Loss map (R2): per-grid-cell running EMA of predictor L2.
        # Lives on self.device — used in DDP collectives, so must match
        # the backend (NCCL requires CUDA).
        #
        # Init to NaN (not zero) so the first observation of any cell
        # initializes the EMA directly rather than starting from a sticky
        # zero — otherwise cells observed more often (center, by random
        # block geometry) would converge faster than edge cells and the
        # z-score over loss_map would encode observation FREQUENCY, not
        # actual predictor loss.  Mirror of cluster_loss_ema NaN trick.
        self._loss_map = torch.full(
            (self.height, self.width), float("nan"), device=self.device
        )
        self._loss_count = torch.zeros(self.height, self.width, device=self.device)

        # Pending accumulator: stats from each micro-batch get added here and
        # folded into the EMA only on ``is_step`` so accum_steps>1 doesn't
        # discard half the data (P1 fix).
        self._loss_pending_sum = torch.zeros(
            self.height, self.width, device=self.device
        )
        self._loss_pending_count = torch.zeros(
            self.height, self.width, device=self.device
        )

        # Cluster centroids (R3b): (K, D) — D set lazily at first call.
        self._cluster_centroids: Optional[torch.Tensor] = None
        self._cluster_init_done = False
        # Per-cluster predictor-loss EMA — NaN until first observed (so an
        # unseen cluster doesn't spuriously look like "background").
        self._cluster_loss_ema = torch.full(
            (self.n_clusters,), float("nan"), device=self.device
        )
        self._cluster_loss_count = torch.zeros(self.n_clusters, device=self.device)
        # Per-iter pending for cluster-loss EMA (P1 fix mirror of loss map).
        self._cluster_loss_pending_sum = torch.zeros(
            self.n_clusters, device=self.device
        )
        self._cluster_loss_pending_count = torch.zeros(
            self.n_clusters, device=self.device
        )
        # Per-iter pending for centroid updates.
        self._cluster_centroid_pending_sum: Optional[torch.Tensor] = None
        self._cluster_centroid_pending_count = torch.zeros(
            self.n_clusters, device=self.device
        )

        # Per-image cluster assignment cache — populated by update_after_iter
        # (or by generate when h_for_cluster is provided).  Shape (B, N).
        # Only used between a single (generate, update) pair, never persisted.
        self._last_cluster_assignment: Optional[torch.Tensor] = None

        # Seeded RNG for block *sizes* (shared across ranks within a batch).
        # Locations use python ``random`` per-image (deterministic if
        # torch/random seeds were set upstream).
        self._size_gen = torch.Generator()

        # Diagnostic counters — read by trainer logger; never reset.
        self._bias_attempt_count = 0
        self._bias_success_count = 0

        # MIRAGE batch statistics, overwritten by every ``generate`` call.
        self._mirage_stats = {}

    # ------------------------------------------------------------------
    # set_epoch
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        """Update ``self._r_t`` for the current epoch.

        Called from the training loop at the top of every epoch.  Pure
        function of ``epoch`` — safe on resume.
        """
        self._epoch = int(epoch)
        # Only adopt the loop's run length when T_total was NOT set explicitly
        # in the config (see _t_total_explicit).  An explicit T_total (e.g. the
        # oracle hard-switch) must not be overwritten by the full run length.
        if total_epochs is not None and not self._t_total_explicit:
            self.T_total = int(total_epochs)
        denom = max(self.T_total - self.T_warm, 1)
        frac = max(0.0, min(1.0, (self._epoch - self.T_warm) / float(denom)))
        if self.ramp_shape == "linear":
            self._r_t = frac * self.r_max
        else:  # cosine
            self._r_t = 0.5 * self.r_max * (1.0 - math.cos(math.pi * frac))

    @property
    def r_t(self) -> float:
        return self._r_t

    # ------------------------------------------------------------------
    # Block sampling helpers (location may be biased)
    # ------------------------------------------------------------------

    def _sample_block_size(
        self, scale: Tuple[float, float], generator: torch.Generator
    ) -> Tuple[int, int]:
        min_s, max_s = scale
        num_target = int(self.num_patches * (
            min_s + (max_s - min_s) * torch.rand(1, generator=generator).item()
        ))
        num_target = max(num_target, 1)
        min_ar, max_ar = self.aspect_ratio
        ar = min_ar + (max_ar - min_ar) * torch.rand(1, generator=generator).item()
        block_h = int(round(math.sqrt(num_target * ar)))
        block_w = int(round(math.sqrt(num_target / ar)))
        block_h = max(1, min(block_h, self.height))
        block_w = max(1, min(block_w, self.width))
        return block_h, block_w

    @staticmethod
    def _sample_uniform_location(
        block_h: int, block_w: int, grid_h: int, grid_w: int
    ) -> Tuple[int, int]:
        top = random.randint(0, grid_h - block_h)
        left = random.randint(0, grid_w - block_w)
        return top, left

    def _sample_biased_location(
        self,
        block_h: int,
        block_w: int,
        weight_grid: torch.Tensor,
    ) -> Tuple[int, int]:
        """Sample a top-left corner proportional to block-summed weights.

        ``weight_grid`` is a (H, W) non-negative tensor.  Computes the
        sum of weights inside every possible (block_h, block_w) window via
        a 2D summed-area table, then samples one window proportional to
        those sums.  Falls back to uniform when the total weight is 0.
        """
        H, W = self.height, self.width
        n_top = H - block_h + 1
        n_left = W - block_w + 1
        if n_top <= 0 or n_left <= 0:
            # Block doesn't fit — fall back to clipped uniform.
            return 0, 0

        # Bring the weight grid to CPU so the multinomial step doesn't force a
        # CUDA sync per image (this is a per-image fallback path inside
        # ``generate`` and runs at most ~B*npred times per iter).
        weight_grid = weight_grid.detach().to(device="cpu", dtype=torch.float32)

        # Summed-area table (integral image), padded by one zero row/col so
        # we can compute block sums without branching at the edges.
        padded = torch.zeros(H + 1, W + 1, dtype=torch.float32)
        padded[1:, 1:] = weight_grid
        sat = padded.cumsum(dim=0).cumsum(dim=1)
        # block_sum[r, c] = sum of weight_grid[r:r+block_h, c:c+block_w]
        block_sum = (
            sat[block_h:, block_w:]
            - sat[:n_top, block_w:]
            - sat[block_h:, :n_left]
            + sat[:n_top, :n_left]
        )
        # Add eps so an all-zero weight grid still samples uniformly.
        flat = block_sum.flatten().clamp(min=self.weight_eps)
        idx = int(torch.multinomial(flat, num_samples=1).item())
        top = idx // n_left
        left = idx % n_left
        return top, left

    def _window_sums(self, grid: torch.Tensor, block_h: int, block_w: int):
        """Block sums for every (block_h, block_w) window via a summed-area table."""
        H, W = self.height, self.width
        n_top, n_left = H - block_h + 1, W - block_w + 1
        if n_top <= 0 or n_left <= 0:
            return None
        padded = torch.zeros(H + 1, W + 1, dtype=torch.float32)
        padded[1:, 1:] = grid
        sat = padded.cumsum(dim=0).cumsum(dim=1)
        return (
            sat[block_h:, block_w:]
            - sat[:n_top, block_w:]
            - sat[block_h:, :n_left]
            + sat[:n_top, :n_left]
        )

    def _sample_mirage_blocks(
        self,
        pred_sizes: List[Tuple[int, int]],
        occupancy: torch.Tensor,
        placement: torch.Tensor,
        biased_flags: List[bool],
        fixed_uniform: List[Optional[List[int]]],
    ) -> Tuple[List[List[int]], dict]:
        """Place target blocks on the MIRAGE retinal region.

        ``placement`` is the dilated admissibility mask -- it grants blocks
        tolerance for MIRAGE's boundary error.  ``occupancy`` is the *true*
        segmentation and is what retina-visibility is measured against, so
        dilation can never inflate the metric that protects encoder context.

        Blocks whose Bernoulli flag is False keep the pre-drawn uniform
        locations in ``fixed_uniform`` across every retry.  Re-drawing them per
        attempt would let the accept test select uniform placements that happen
        to spare retina, quietly biasing blocks the ramp says must be random.

        Implemented in NumPy on the 16x16 grid: torch op overhead dominates at
        this size, and the admissibility of every window depends only on the
        region and the block size, so it is computed once and reused across all
        retries rather than rebuilt per attempt.

        Returns the per-block index lists plus statistics for logging.
        """
        region = np.asarray(
            (placement > 0).to(dtype=torch.float32).cpu().numpy(), dtype=np.float64
        )
        truth = np.asarray(
            (occupancy >= self.mirage_occupancy_threshold)
            .to(dtype=torch.float32)
            .cpu()
            .numpy(),
            dtype=bool,
        )
        region_cells = int(region.sum())
        truth_flat = truth.reshape(-1)
        truth_cells = int(truth.sum())
        rng = np.random.default_rng(random.randrange(2 ** 31))

        # Invariant across retries: which windows are admissible for each block.
        candidate_cache: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        for index, (bh, bw) in enumerate(pred_sizes):
            if not biased_flags[index] or region_cells == 0:
                candidate_cache.append(None)
                continue
            counts = _window_sums_numpy(region, bh, bw)
            if counts is None:
                candidate_cache.append(None)
                continue
            fill = counts / float(bh * bw)
            candidate_cache.append((fill, fill >= self.mirage_min_block_fill))

        occupied_cols = np.flatnonzero(region.any(axis=0)) if region_cells else np.empty(0)
        best: Optional[Tuple[List[List[int]], dict]] = None

        for attempt in range(max(1, self.mirage_max_attempts)):
            claimed = np.zeros_like(region)
            segments: List[Tuple[int, int]] = []
            if self.mirage_spread and occupied_cols.size:
                edges = np.linspace(
                    float(occupied_cols.min()),
                    float(occupied_cols.max()) + 1.0,
                    len(pred_sizes) + 1,
                )
                segments = [
                    (int(math.floor(edges[i])), int(math.ceil(edges[i + 1])))
                    for i in range(len(pred_sizes))
                ]
                segments = [segments[i] for i in rng.permutation(len(segments))]

            per_image_pred: List[List[int]] = []
            fills: List[float] = []
            guided_used = 0
            feasible = True
            for index, (bh, bw) in enumerate(pred_sizes):
                if not biased_flags[index]:
                    indices = fixed_uniform[index]
                    per_image_pred.append(indices)
                    top = indices[0] // self.width
                    left = indices[0] % self.width
                    claimed[top : top + bh, left : left + bw] = 1.0
                    continue
                cached = candidate_cache[index]
                if cached is None:
                    feasible = False
                    top, left = self._sample_uniform_location(
                        bh, bw, self.height, self.width
                    )
                else:
                    fill, candidates = cached
                    if segments and candidates.any():
                        low, high = segments[index]
                        centres = np.arange(candidates.shape[1]) + bw / 2.0
                        inside = (centres >= low) & (centres < high)
                        banded = np.zeros_like(candidates)
                        banded[:, inside] = candidates[:, inside]
                        if banded.any():
                            candidates = banded
                    if candidates.any():
                        overlap = _window_sums_numpy(claimed, bh, bw)
                        if overlap is not None:
                            free = candidates & (
                                overlap <= self.mirage_overlap_tolerance * bh * bw
                            )
                            if free.any():
                                candidates = free
                            else:
                                # No admissible window clears the tolerance --
                                # common, because the retinal band is thinner
                                # than the block and mirage_min_block_fill
                                # already restricts placement to a narrow strip.
                                # This used to fall through with NO overlap
                                # constraint and pick uniformly, which is how
                                # measured overlap reached 40.5% against a 0.25
                                # tolerance -- worse than unguided random's
                                # 28.9%. Fall back to the LEAST-overlapping
                                # admissible window instead of an arbitrary one.
                                worst = np.where(candidates, overlap,
                                                 np.inf).min()
                                least = candidates & (overlap <= worst)
                                if least.any():
                                    candidates = least
                        rows, cols = np.nonzero(candidates)
                        pick = int(rng.integers(rows.size))
                        top, left = int(rows[pick]), int(cols[pick])
                        fills.append(float(fill[top, left]))
                        guided_used += 1
                    else:
                        # No window anywhere reaches the fill threshold: this is
                        # an infeasible guide, not a successful guided mask.
                        feasible = False
                        top, left = self._sample_uniform_location(
                            bh, bw, self.height, self.width
                        )
                indices = self._block_to_indices(top, left, bh, bw)
                per_image_pred.append(indices)
                claimed[top : top + bh, left : left + bw] = 1.0

            union = set()
            for indices in per_image_pred:
                union.update(indices)
            masked_truth = int(truth_flat[list(union)].sum()) if union else 0
            visible = (
                (truth_cells - masked_truth) / truth_cells if truth_cells > 0 else 1.0
            )
            stats = {
                "region_cells": region_cells,
                "truth_cells": truth_cells,
                "guided_blocks": guided_used,
                "mean_block_fill": float(np.mean(fills)) if fills else 0.0,
                "retina_visible": visible,
                "attempts": attempt + 1,
                "feasible": feasible,
                "accepted": feasible and visible >= self.mirage_min_retina_visible,
            }
            if (
                best is None
                # Prefer feasible attempts; among equals prefer more visible retina.
                or (stats["feasible"], stats["retina_visible"])
                > (best[1]["feasible"], best[1]["retina_visible"])
            ):
                best = (per_image_pred, stats)
            if stats["accepted"]:
                return per_image_pred, stats
            if not feasible:
                # Admissibility does not change between retries, so a guide that
                # cannot host the blocks will never succeed -- stop early.
                break
        return best  # type: ignore[return-value]

    def _block_to_indices(
        self, top: int, left: int, block_h: int, block_w: int
    ) -> List[int]:
        indices = []
        for r in range(top, top + block_h):
            for c in range(left, left + block_w):
                indices.append(r * self.width + c)
        return sorted(indices)

    # ------------------------------------------------------------------
    # Weight grids per mode
    # ------------------------------------------------------------------

    def _is_loss_map_mature(self) -> bool:
        if float(self._loss_count.min().item()) < self.loss_maturity_min_count:
            return False
        return True

    def _are_clusters_mature(self) -> bool:
        if not self._cluster_init_done:
            return False
        if torch.isnan(self._cluster_loss_ema).any().item():
            return False
        if float(self._cluster_loss_count.min().item()) < self.cluster_maturity_min_count:
            return False
        return True

    def _foreground_cluster_mask(self) -> torch.Tensor:
        """Boolean (K,) on CPU — True for clusters in the foreground set.

        Selection logic:
          1. If ``foreground_cluster_ids`` is explicitly configured (e.g. the
             user knows from prior analysis that clusters 1 and 3 are the
             retina/optic-disc clusters), use exactly those IDs.
          2. Otherwise apply ``cluster_foreground_selection``:
               - "top_half" (default): the K/2 hardest clusters by loss EMA.
                 Caveat: with low spread across clusters this picks noise.
               - "above_mean": clusters whose loss exceeds (mean + z*std)
                 where ``z = foreground_loss_zscore``.  Robust when
                 clustering isn't finding signal (returns at least 1 cluster
                 even if no cluster crosses the bar — picks the argmax).

        NaN clusters (never observed) are excluded from both branches so an
        unseen cluster never spuriously looks "hard".
        """
        if self.foreground_cluster_ids is not None:
            fg = torch.zeros(self.n_clusters, dtype=torch.bool)
            for k in self.foreground_cluster_ids:
                if 0 <= int(k) < self.n_clusters:
                    fg[int(k)] = True
            return fg
        # Operate on CPU so the boolean result is host-side regardless of
        # whether the EMA lives on CUDA.
        ema = self._cluster_loss_ema.detach().to(device="cpu").clone()
        nan_mask = torch.isnan(ema)
        if bool(nan_mask.all().item()):
            # No cluster ever observed — bias is meaningless, fall back to
            # marking all clusters as "foreground" so cluster_weight_grid is
            # the uniform 1s and sampling reverts to the uniform top-left
            # distribution.  bias_active gate should already have blocked us.
            return torch.ones(self.n_clusters, dtype=torch.bool)
        ema[nan_mask] = -float("inf")

        if self.cluster_foreground_selection == "top_half":
            k_half = max(1, self.n_clusters // 2)
            topk = torch.topk(ema, k=k_half).indices
            fg = torch.zeros(self.n_clusters, dtype=torch.bool)
            fg[topk] = True
            return fg

        # "above_mean" — robust to "useless cluster" failure mode.
        finite = ema[~nan_mask]
        mean = float(finite.mean().item())
        std = float(finite.std(unbiased=False).item())
        threshold = mean + self.foreground_loss_zscore * std
        fg = ema > threshold
        if not bool(fg.any().item()):
            # No cluster crossed the threshold (spread too small) — fall back
            # to the single hardest cluster so we never return an empty set
            # (empty fg -> all-zero weight grid -> sampling crash).
            fg = torch.zeros(self.n_clusters, dtype=torch.bool)
            fg[int(ema.argmax().item())] = True
        return fg

    def cluster_loss_spread(self) -> Tuple[float, float, float, float]:
        """Diagnostic — (min, max, mean, std) of finite cluster loss EMAs.

        Used by the trainer to log per-epoch cluster signal quality.  Small
        spread (e.g. max-min < 0.1*mean) is a red flag that the clustering
        isn't finding semantically distinct groups and R3b bias will be
        essentially noise — switch to R3a or rerun with a different K.
        """
        ema = self._cluster_loss_ema.detach().to(device="cpu").clone()
        finite = ema[~torch.isnan(ema)]
        if finite.numel() == 0:
            return (float("nan"), float("nan"), float("nan"), float("nan"))
        return (
            float(finite.min().item()),
            float(finite.max().item()),
            float(finite.mean().item()),
            float(finite.std(unbiased=False).item()),
        )

    def _loss_weight_grid(self) -> torch.Tensor:
        """(H, W) weight grid for R2 loss-guided sampling."""
        # Use the loss EMA directly; clamp to non-negative just in case.
        # Defensive: replace any still-NaN cell (shouldn't happen because
        # _is_loss_map_mature gates this on count >= 32 per cell, but in
        # case of partial restore weirdness, NaN -> 0 stays neutral).
        wg = torch.nan_to_num(self._loss_map, nan=0.0)
        return wg.clamp(min=0.0)

    def _intensity_weight_grid_for_image(
        self, img_cpu: torch.Tensor
    ) -> torch.Tensor:
        """(H, W) 0/1 mask of patches whose mean intensity is above quantile.

        ``img_cpu`` is (C, H_pix, W_pix) on CPU.  Reduces to per-patch mean
        across channels and pixels, then thresholds at the per-image quantile.

        Tie-safe (FINDING-G2): uses strict ``>`` and falls back to top-K
        selection when the threshold lands on a tied value (common for OCT
        slices with large dark vitreous / sclera regions — when background
        fraction >= quantile, the threshold is the tied background value
        itself and ``>=`` would silently pick 100% of patches, degrading
        R3a to no-biasing).
        """
        C, Hp, Wp = img_cpu.shape
        ps = self.patch_size
        # (H, W, ps, ps, C) after reshape
        patches = (
            img_cpu.permute(1, 2, 0)
            .reshape(self.height, ps, self.width, ps, C)
            .permute(0, 2, 1, 3, 4)
        )
        patch_mean = patches.float().mean(dim=(2, 3, 4))  # (H, W)
        thresh = torch.quantile(patch_mean, self.intensity_quantile)
        mask = (patch_mean > thresh)
        # Tie fallback: if strict > selects nothing (or too few because the
        # quantile landed on a tied value), pick exactly the top-K patches
        # by intensity so foreground fraction stays close to intended.
        intended_fg = max(1, int(round((1.0 - self.intensity_quantile)
                                       * patch_mean.numel())))
        if int(mask.sum().item()) < intended_fg:
            flat = patch_mean.flatten()
            topk_idx = torch.topk(flat, intended_fg).indices
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[topk_idx] = True
            mask = mask.view(self.height, self.width)
        return mask.float()

    def _anatomical_prior_weight_grid_for_image(
        self, img_cpu: torch.Tensor
    ) -> torch.Tensor:
        """(H, W) 0/1 grid over the retinal band — ORACLE v2 (retina-following).

        Localizes the bright retinal band per slice from the row-intensity
        profile (the retina is the bright band between dark vitreous above and
        darker choroid/sclera below), then masks a fixed-size window CENTRED on
        the band across most of the lateral extent.  Centring follows the
        retina's actual vertical position per slice (handles curvature/tilt);
        the fixed window size keeps the region in the 20-40% band so the 4
        target blocks fit without piling up.

        Intensity is used only to LOCALIZE the retina here; this is not the
        (dropped) ``intensity_foreground`` mode, which used intensity as the
        mask signal itself.

        ``img_cpu`` is (C, H_pix, W_pix) on CPU.
        """
        C, Hp, Wp = img_cpu.shape
        ps = self.patch_size
        H, W = self.height, self.width
        patches = (
            img_cpu.permute(1, 2, 0)
            .reshape(H, ps, W, ps, C)
            .permute(0, 2, 1, 3, 4)
        )
        patch_mean = patches.float().mean(dim=(2, 3, 4))  # (H, W)

        # Affine-invariant brightness map.  Subtract the per-slice min so the
        # weighting is RELATIVE to this slice's own contrast.  The training
        # transform applies ImageNet Normalize; that is affine per channel and,
        # averaged over the grayscale-replicated channels, stays affine in the
        # raw value (a*v + b, a>0).  A centroid of (brightness - min) is
        # affine-invariant, so the band localized is identical whether the
        # input is raw [0,1] or normalized.  Also removes the old clamp(min=0)
        # failure where a dim NORMALIZED slice collapsed to center-row masking.
        prof = patch_mean - patch_mean.min()  # (H, W) >= 0
        ys = torch.arange(H, dtype=torch.float32)
        eps = 1e-6

        # Global row centroid -> fallback for dark/edge columns.
        row_mass = prof.sum(dim=1)  # (H,)
        gtot = float(row_mass.sum().item())
        global_c = float((ys * row_mass).sum().item() / gtot) if gtot > eps \
            else H / 2.0

        # PER-COLUMN centroid -> a curve-following RIBBON, not a rectangle.
        # Real OCT retina is a curved/tilted band; a single rectangular window
        # catches only the centre and dilutes with vitreous in the corners.
        # Tracking the retina's vertical centre per column follows the curve.
        col_mass = prof.sum(dim=0)  # (W,)
        col_c = torch.where(
            col_mass > eps,
            (ys.view(H, 1) * prof).sum(dim=0) / col_mass.clamp(min=eps),
            torch.full((W,), global_c),
        )  # (W,)
        # Smooth across columns so the ribbon is not jagged (reflect-safe avg).
        col_c = torch.nn.functional.avg_pool1d(
            col_c.view(1, 1, W), kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        ).view(W)
        # Optional vertical offset (negative shifts up toward RNFL for a
        # v3-style glaucoma-focused oracle; 0 = centred on the band).
        col_c = col_c + self.oracle_row_offset * H

        # Band height chosen so region ~= oracle_region_frac given the lateral
        # fraction — guarantees "not too little".  Same total area as the box;
        # the rows just shift per column to follow the retina.
        lat_frac = float(self.oracle_lateral_frac)
        band_h = int(round((self.oracle_region_frac / max(lat_frac, eps)) * H))
        band_h = max(self.oracle_min_band_rows, min(band_h, H))

        # Central lateral extent (ignore left/right edges, keep most of x).
        x_keep = max(1, min(int(round(lat_frac * W)), W))
        x0 = (W - x_keep) // 2
        x1 = x0 + x_keep

        grid = torch.zeros(H, W, dtype=torch.float32)
        for x in range(x0, x1):
            c = int(round(float(col_c[x])))
            top = max(0, min(c - band_h // 2, H - band_h))
            grid[top:top + band_h, x] = 1.0
        return grid

    def _cluster_weight_grid_for_image(
        self, assignment: torch.Tensor
    ) -> torch.Tensor:
        """(H, W) 0/1 mask of patches whose cluster ID is in the foreground."""
        fg = self._foreground_cluster_mask()  # (K,)
        fg_per_token = fg.to(assignment.device)[assignment]  # (N,)
        return fg_per_token.float().view(self.height, self.width).cpu()

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(
        self,
        batch_size: int,
        imgs_cpu: Optional[torch.Tensor] = None,
        h_for_cluster: Optional[torch.Tensor] = None,
        guide_grids: Optional[torch.Tensor] = None,
        guide_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Produce (masks_enc, masks_pred) with the MaskCollator contract.

        Args:
            batch_size: B.
            imgs_cpu: (B, C, H_pix, W_pix) on CPU.  REQUIRED for
                ``intensity_foreground`` mode, ignored otherwise.
            h_for_cluster: (B, N, D) target-encoder full output.  REQUIRED
                for ``cluster_foreground`` mode (caller must compute the
                target forward BEFORE calling ``generate``).
            guide_grids: (B, H, W) per-patch retinal occupancy.  REQUIRED for
                ``mirage_envelope`` mode.
            guide_valid: (B,) bool flags; images whose guide failed quality
                control fall back to uniform random placement.
        """
        B = int(batch_size)

        # Shared block sizes for this batch (matches multiblock).
        seed = random.randint(0, 2 ** 31)
        self._size_gen.manual_seed(seed)
        pred_sizes = [
            self._sample_block_size(self.pred_mask_scale, self._size_gen)
            for _ in range(self.npred)
        ]
        enc_sizes = [
            self._sample_block_size(self.enc_mask_scale, self._size_gen)
            for _ in range(self.nenc)
        ]

        # ------------------------------------------------------------------
        # Decide whether biased sampling is ACTIVE this batch.
        # Biased sampling is gated by:
        #   * mode-specific maturity gate
        #   * r_t > 0
        # When gated off, we behave exactly like multiblock (R1).
        # ------------------------------------------------------------------
        bias_active = self._r_t > 0.0
        if bias_active:
            if self.mode == "loss_guided":
                bias_active = self._is_loss_map_mature()
            elif self.mode == "cluster_foreground":
                bias_active = (
                    h_for_cluster is not None and self._are_clusters_mature()
                )
            elif self.mode == "intensity_foreground":
                bias_active = imgs_cpu is not None
            elif self.mode == "anatomical_prior":
                bias_active = imgs_cpu is not None
            elif self.mode == "mirage_envelope":
                bias_active = guide_grids is not None
            elif self.mode == "mirage_anatomy":
                bias_active = guide_grids is not None

        # Pre-compute cluster assignments per image (R3b) so update_after_iter
        # can reuse them without re-clustering.
        per_image_cluster_assignment: Optional[List[torch.Tensor]] = None
        if (
            self.mode == "cluster_foreground"
            and h_for_cluster is not None
            and self._cluster_init_done
        ):
            per_image_cluster_assignment = self._assign_per_image(h_for_cluster)
            self._last_cluster_assignment = torch.stack(
                per_image_cluster_assignment, dim=0
            )

        # ------------------------------------------------------------------
        # Per-image mask sampling.
        # ------------------------------------------------------------------
        masks_enc = [[] for _ in range(self.nenc)]
        masks_pred = [[] for _ in range(self.npred)]
        mirage_batch = {
            "images": 0,
            "guided_images": 0,
            "accepted": 0,
            "infeasible": 0,
            "fallback_invalid": 0,
            "unbiased_by_ramp": 0,
            "block_fill_sum": 0.0,
            "retina_visible_sum": 0.0,
            "target_on_region_sum": 0.0,
            "attempts_sum": 0,
            "patches_per_block_sum": 0,
            "unique_target_sum": 0,
            "context_patch_sum": 0,
        }

        for b in range(B):
            # Build the weight grid once per image (intensity / cluster need
            # per-image weights; loss-guided is shared across the batch but
            # we still build per-image for code uniformity).
            weight_grid: Optional[torch.Tensor] = None
            if bias_active:
                if self.mode == "loss_guided":
                    weight_grid = self._loss_weight_grid()
                elif self.mode == "intensity_foreground":
                    weight_grid = self._intensity_weight_grid_for_image(
                        imgs_cpu[b]
                    )
                elif self.mode == "anatomical_prior":
                    weight_grid = self._anatomical_prior_weight_grid_for_image(
                        imgs_cpu[b]
                    )
                elif self.mode == "cluster_foreground":
                    weight_grid = self._cluster_weight_grid_for_image(
                        per_image_cluster_assignment[b]
                    )
                elif self.mode == "mirage_envelope":
                    weight_grid = guide_grids[b].detach().to(
                        device="cpu", dtype=torch.float32
                    )

            # --- Pred blocks ---
            pred_indices_union = set()
            per_image_pred = []
            if self.mode == "mirage_anatomy":
                # Connected, anatomy-shaped targets that follow the retinal
                # band instead of rectangles aimed at it.  Channel 0 of the
                # guide is per-patch occupancy -- the FRACTION of the patch
                # covered by the envelope -- so it is already the soft score
                # the sampler wants, no extra cache needed.
                guide = None
                if guide_grids is not None:
                    guide = guide_grids[b].detach().to(
                        device="cpu", dtype=torch.float32
                    )
                    if guide.dim() == 2:
                        guide = torch.stack([guide, guide], dim=0)
                occupancy = guide[0] if guide is not None else None
                placement = guide[1] if guide is not None else None
                usable = bias_active and guide is not None
                if usable and guide_valid is not None:
                    usable = bool(guide_valid[b])
                # The ramp is per IMAGE here, not per block: a target set that
                # mixed anatomy shapes with random rectangles would not be a
                # coherent partition of the retina.
                use_anatomy = bool(usable and random.random() < self._r_t)
                # A schema-2 guide carries P_inner and P_choroid in channels
                # 2 and 3.  Class-aware growth is what the adapter sweep
                # validated; channel 0 alone merges inner retina and choroid
                # into a single band, which still works but is the degraded
                # form.
                if guide is not None and guide.shape[0] >= 4:
                    class_scores = [guide[2].numpy(), guide[3].numpy()]
                elif occupancy is not None:
                    class_scores = [occupancy.numpy()]
                else:
                    class_scores = None
                # `is_viable` is not optional: roughly 1.7% of slices cannot
                # fill four targets because the guide finds too little
                # anatomy, and those must fall back rather than emit an empty
                # target.
                if use_anatomy and class_scores is not None and anatomy_is_viable(
                    class_scores, n=self.npred, mass_cap=self.anatomy_mass_cap,
                    tau=self.anatomy_tau
                ):
                    parts, _ = anatomy_build_targets(
                        class_scores, n=self.npred,
                        mass_cap=self.anatomy_mass_cap, tau=self.anatomy_tau,
                    )
                    if self.pred_target_k is not None:
                        # Shrink CONNECTEDLY here rather than letting
                        # resample_to_k subsample uniformly at collation,
                        # which was measured to leave only 231/256 targets
                        # as a single component.
                        ref = class_scores[0] if len(class_scores) == 1 else (
                            class_scores[0] + class_scores[1])
                        parts = [
                            anatomy_shrink_to_k(p, int(self.pred_target_k), ref)
                            for p in parts
                        ]
                    per_image_pred = [
                        np.flatnonzero(np.asarray(p).ravel()).tolist()
                        for p in parts
                    ]
                else:
                    mirage_batch["fallback_invalid"] += 1
                    for p in range(self.npred):
                        bh, bw = pred_sizes[p]
                        top, left = self._sample_uniform_location(
                            bh, bw, self.height, self.width
                        )
                        idx = self._block_to_indices(top, left, bh, bw)
                        if self.pred_target_k is not None and len(idx) > self.pred_target_k:
                            # The fallback rectangle is larger than K, and
                            # resample_to_k would subsample it UNIFORMLY at
                            # collation, shattering a connected block into
                            # scattered cells. Measured: this alone accounted
                            # for all the disconnected targets on real guides
                            # (233/256 connected). Shrink it connectedly here,
                            # exactly as the anatomy branch does.
                            m = np.zeros((self.height, self.width), bool)
                            m.ravel()[np.asarray(idx)] = True
                            m = anatomy_shrink_to_k(m, int(self.pred_target_k))
                            idx = np.flatnonzero(m.ravel()).tolist()
                        per_image_pred.append(idx)
                for indices in per_image_pred:
                    pred_indices_union.update(indices)
                if occupancy is not None and pred_indices_union:
                    flat = occupancy.flatten()
                    on_region = sum(
                        1 for i in pred_indices_union
                        if float(flat[i]) >= self.mirage_occupancy_threshold
                    )
                    mirage_batch["target_on_region_sum"] += on_region / len(
                        pred_indices_union
                    )
                mirage_batch["images"] += 1
                mirage_batch["guided_images"] += int(use_anatomy)
                mirage_batch["patches_per_block_sum"] += sum(
                    len(indices) for indices in per_image_pred
                )
                mirage_batch["unique_target_sum"] += len(pred_indices_union)
            elif self.mode == "mirage_envelope":
                # Statistics must be measurable even when the ramp is off, so
                # the guide is read independently of ``bias_active``.
                guide = None
                if guide_grids is not None:
                    guide = guide_grids[b].detach().to(
                        device="cpu", dtype=torch.float32
                    )
                    if guide.dim() == 2:  # occupancy only; no dilated channel
                        guide = torch.stack([guide, guide], dim=0)
                occupancy = guide[0] if guide is not None else None
                placement = guide[1] if guide is not None else None
                usable = bias_active and guide is not None
                if usable and guide_valid is not None:
                    usable = bool(guide_valid[b])
                # Bernoulli(r_t) per block, matching every other mode: the ramp
                # decides how many of the four blocks follow the guide.
                biased_flags = [
                    bool(usable and random.random() < self._r_t)
                    for _ in range(self.npred)
                ]
                # Draw the non-guided block locations ONCE so retries cannot
                # quietly bias blocks the ramp designated as random.
                fixed_uniform: List[Optional[List[int]]] = []
                for p in range(self.npred):
                    if biased_flags[p]:
                        fixed_uniform.append(None)
                    else:
                        bh, bw = pred_sizes[p]
                        top, left = self._sample_uniform_location(
                            bh, bw, self.height, self.width
                        )
                        fixed_uniform.append(
                            self._block_to_indices(top, left, bh, bw)
                        )
                if usable and any(biased_flags):
                    per_image_pred, stats = self._sample_mirage_blocks(
                        pred_sizes, occupancy, placement, biased_flags, fixed_uniform
                    )
                    mirage_batch["guided_images"] += 1
                    mirage_batch["accepted"] += int(stats["accepted"])
                    mirage_batch["infeasible"] += int(not stats["feasible"])
                    mirage_batch["block_fill_sum"] += stats["mean_block_fill"]
                    mirage_batch["retina_visible_sum"] += stats["retina_visible"]
                    mirage_batch["attempts_sum"] += stats["attempts"]
                else:
                    if guide is None or (
                        guide_valid is not None and not bool(guide_valid[b])
                    ):
                        mirage_batch["fallback_invalid"] += 1
                    else:
                        mirage_batch["unbiased_by_ramp"] += 1
                    per_image_pred = [
                        indices for indices in fixed_uniform if indices is not None
                    ]
                for indices in per_image_pred:
                    pred_indices_union.update(indices)
                if occupancy is not None and pred_indices_union:
                    flat = occupancy.flatten()
                    on_region = sum(
                        1
                        for i in pred_indices_union
                        if float(flat[i]) >= self.mirage_occupancy_threshold
                    )
                    mirage_batch["target_on_region_sum"] += on_region / len(
                        pred_indices_union
                    )
                mirage_batch["images"] += 1
                mirage_batch["patches_per_block_sum"] += sum(
                    len(indices) for indices in per_image_pred
                )
                mirage_batch["unique_target_sum"] += len(pred_indices_union)
            else:
                for p in range(self.npred):
                    bh, bw = pred_sizes[p]
                    # Bernoulli(r_t) per block (P1 fix — never round to 0).
                    if (
                        bias_active
                        and weight_grid is not None
                        and random.random() < self._r_t
                    ):
                        top, left = self._sample_biased_location(bh, bw, weight_grid)
                    else:
                        top, left = self._sample_uniform_location(
                            bh, bw, self.height, self.width
                        )
                    indices = self._block_to_indices(top, left, bh, bw)
                    per_image_pred.append(indices)
                    pred_indices_union.update(indices)

            # --- Enc blocks (BIT-IDENTICAL to multiblock) ---
            for e in range(self.nenc):
                bh, bw = enc_sizes[e]
                best_indices: Optional[List[int]] = None
                for _attempt in range(50):
                    top, left = self._sample_uniform_location(
                        bh, bw, self.height, self.width
                    )
                    indices = self._block_to_indices(top, left, bh, bw)
                    if self.allow_overlap:
                        kept = indices
                    else:
                        kept = [
                            i for i in indices if i not in pred_indices_union
                        ]
                    if len(kept) >= self.min_keep:
                        best_indices = kept
                        break
                if best_indices is None:
                    all_patches = list(range(self.num_patches))
                    if self.allow_overlap:
                        best_indices = all_patches
                    else:
                        best_indices = [
                            i for i in all_patches if i not in pred_indices_union
                        ]
                    if len(best_indices) < self.min_keep:
                        best_indices = all_patches[: self.min_keep]
                masks_enc[e].append(
                    torch.tensor(sorted(best_indices), dtype=torch.long)
                )
                if self.mode in ("mirage_envelope", "mirage_anatomy"):
                    mirage_batch["context_patch_sum"] += len(best_indices)

            for p in range(self.npred):
                masks_pred[p].append(
                    torch.tensor(per_image_pred[p], dtype=torch.long)
                )

        # --- Per-group enc min-truncate ---
        collated_masks_enc = []
        for group in masks_enc:
            min_len = max(1, min(t.numel() for t in group))
            collated_masks_enc.append(
                torch.stack([t[:min_len] for t in group], dim=0)
            )

        # --- GLOBAL pred min-truncate (across all groups AND batch) ---
        # With irregular anatomy targets this is destructive: one small target
        # anywhere in the microbatch front-slices every other target down to
        # its length.  `pred_target_k` replaces it with a per-target resample
        # that keeps the length uniform without discarding anyone's cells.
        collated_masks_pred = []
        if self.pred_target_k is not None:
            k = int(self.pred_target_k)
            for group in masks_pred:
                collated_masks_pred.append(torch.stack(
                    [resample_to_k(t, k) for t in group], dim=0
                ))
            # Reported as the per-target length that actually reached the
            # predictor, which under this policy is K by construction.
            global_min_pred = k
        else:
            global_min_pred = max(
                1, min(t.numel() for group in masks_pred for t in group)
            )
            for group in masks_pred:
                collated_masks_pred.append(
                    torch.stack([t[:global_min_pred] for t in group], dim=0)
                )

        if self.mode in ("mirage_envelope", "mirage_anatomy"):
            images = max(mirage_batch["images"], 1)
            guided = max(mirage_batch["guided_images"], 1)
            self._mirage_stats = {
                "images": mirage_batch["images"],
                "guided_images": mirage_batch["guided_images"],
                # Guides that could not be used at all: QC-invalid or missing.
                "fallbacks": mirage_batch["fallback_invalid"],
                # Guided attempts where some block found no admissible window.
                "infeasible": mirage_batch["infeasible"],
                # Images the ramp deliberately left fully random (r_t < 1).
                "unbiased_by_ramp": mirage_batch["unbiased_by_ramp"],
                "accept_rate": mirage_batch["accepted"] / guided,
                "mean_block_fill": mirage_batch["block_fill_sum"] / guided,
                "retina_visible": mirage_batch["retina_visible_sum"] / guided,
                "target_on_region": mirage_batch["target_on_region_sum"] / images,
                "target_background": 1.0
                - mirage_batch["target_on_region_sum"] / images,
                "patches_per_block": mirage_batch["patches_per_block_sum"]
                / (images * self.npred),
                "unique_target_patches": mirage_batch["unique_target_sum"] / images,
                "context_patches": mirage_batch["context_patch_sum"] / images,
                "mean_attempts": mirage_batch["attempts_sum"] / guided,
                "truncated_target_patches": global_min_pred,
            }

        return collated_masks_enc, collated_masks_pred

    # ------------------------------------------------------------------
    # update_after_iter
    # ------------------------------------------------------------------

    def update_after_iter(
        self,
        *,
        per_token_loss: torch.Tensor,
        masks_pred_idx: List[torch.Tensor],
        h_for_cluster: Optional[torch.Tensor] = None,
        is_step: bool = False,
    ) -> None:
        """Fold new observations into the curriculum state.

        Args:
            per_token_loss: (B*npred*nenc, K_pred) squared error per pred
                token, averaged over the embed dim.  Computed by the
                training loop right after the forward.  Moved to
                ``self.device`` here so DDP collectives match the backend
                (NCCL requires CUDA tensors).
            masks_pred_idx: list of ``npred`` LongTensors (B, K_pred) — the
                actual indices used in the forward (same objects that
                ``generate`` returned).
            h_for_cluster: (B, N, D) layer-normed target encoder full output.
                Required only for ``cluster_foreground`` mode.
            is_step: True when the training loop just took an optimizer step
                (i.e. ``(itr + 1) % accum_steps == 0`` or last micro-batch).
                Folds the pending accumulators into the EMA when True.

        Always called on EVERY rank — collectives are unconditional.

        FAST PATH for ``intensity_foreground`` mode: that mode's foreground
        prior is computed per-image from the input intensity and needs no
        learned state.  We early-return after bumping the iter counter so
        R3a doesn't pay loss-map accumulation or all-reduce costs.
        """
        # Stateless modes (R3a intensity, ORACLE anatomical_prior) compute the
        # foreground prior per-image from the input — no learned state to fold.
        if self.mode in ("intensity_foreground", "anatomical_prior"):
            self._iter += 1
            return

        per_token_loss = per_token_loss.detach().float().to(self.device)

        # NaN/Inf guard — any non-finite per-token loss (AMP overflow, model
        # NaN) would silently corrupt loss_map / cluster_loss EMAs and bias
        # later sampling.  Replace bad cells with 0 and exclude them from
        # the count so maturity gates don't advance on garbage data.
        finite_mask = torch.isfinite(per_token_loss)
        if not bool(finite_mask.all().item()):
            n_bad = int((~finite_mask).sum().item())
            n_total = int(finite_mask.numel())
            if self.rank == 0:
                print(
                    "[Curriculum] WARN: %d/%d per-token losses non-finite "
                    "at iter %d — skipping those positions in this update."
                    % (n_bad, n_total, self._iter)
                )
            per_token_loss = torch.where(
                finite_mask, per_token_loss, torch.zeros_like(per_token_loss)
            )
        else:
            finite_mask = None  # all good, no need to mask counts later

        # Recover (npred, nenc, B, K) from (B*npred*nenc, K) ordering.
        B = int(masks_pred_idx[0].shape[0])
        K = int(per_token_loss.shape[-1])
        total = int(per_token_loss.shape[0])
        if total != self.npred * self.nenc * B:
            raise ValueError(
                f"per_token_loss has {total} rows, expected "
                f"{self.npred * self.nenc * B} = npred({self.npred}) * "
                f"nenc({self.nenc}) * B({B})"
            )
        per_token = per_token_loss.view(self.npred, self.nenc, B, K).mean(dim=1)
        # per_token: (npred, B, K) — averaged over nenc copies.

        # Same shape (npred, nenc, B, K) collapse for the finite mask, then
        # average to per-(npred,B,K) presence (any non-finite token in any
        # enc-copy disqualifies the cell from count advancement).
        if finite_mask is not None:
            fm = finite_mask.view(self.npred, self.nenc, B, K).all(dim=1).float()
        else:
            fm = None

        # ------------------------------------------------------------------
        # Loss-map / per-cluster-loss accumulation (used by R2 and R3b).
        # Everyone scatters; mode gates only what we *use*.
        # ------------------------------------------------------------------
        positions_flat = torch.stack(
            [m.detach().to(self.device) for m in masks_pred_idx], dim=0
        )  # (npred, B, K)

        # Per-position (H*W) sum + count for the loss map.
        flat_pos = positions_flat.view(-1)
        flat_loss = per_token.reshape(-1)
        if fm is not None:
            flat_finite = fm.reshape(-1)
        else:
            flat_finite = torch.ones_like(flat_loss)
        loss_sum_flat = torch.zeros(self.num_patches, device=self.device)
        loss_count_flat = torch.zeros(self.num_patches, device=self.device)
        # Mask out non-finite contributions for both sum and count.
        loss_sum_flat.scatter_add_(0, flat_pos, flat_loss * flat_finite)
        loss_count_flat.scatter_add_(0, flat_pos, flat_finite)

        # DDP all-reduce SUM — unconditional on every rank.
        _maybe_all_reduce(loss_sum_flat)
        _maybe_all_reduce(loss_count_flat)

        # Pending accumulators — folded into EMA only on optimizer-step
        # boundary so accum_steps>1 doesn't discard half the data.
        self._loss_pending_sum += loss_sum_flat.view(self.height, self.width)
        self._loss_pending_count += loss_count_flat.view(self.height, self.width)
        # Permanent observation count (drives maturity gate).
        self._loss_count += loss_count_flat.view(self.height, self.width)

        # ------------------------------------------------------------------
        # Cluster update (R3b only).
        # ------------------------------------------------------------------
        if self.mode == "cluster_foreground" and h_for_cluster is not None:
            cached = self._last_cluster_assignment
            self._update_clusters(
                h_for_cluster=h_for_cluster,
                per_token=per_token,
                positions_flat=positions_flat,
                fm=fm,
                assignment=cached,
            )
            # Always clear after consumption — never let a stale cache from
            # this iter leak into the next (different batch, different B).
            self._last_cluster_assignment = None

        # ------------------------------------------------------------------
        # EMA fold on optimizer-step boundary.
        # ------------------------------------------------------------------
        if is_step:
            self._fold_pending_into_ema()

        self._iter += 1

    def _fold_pending_into_ema(self) -> None:
        # Loss-map EMA (R2): per-cell mean of new observations, blended in.
        # NaN-safe first-observation init (FINDING-G1): a cell observed for
        # the first time takes ``new_mean`` directly; subsequent observations
        # do the standard EMA blend.  Without this, every cell starts at 0
        # and converges at a rate proportional to its observation FREQUENCY,
        # so center cells (observed more by random block geometry) would
        # look artificially "higher loss" than edge cells purely because
        # their EMA had more time to escape zero.
        cnt = self._loss_pending_count.clamp(min=1.0)
        new_mean = self._loss_pending_sum / cnt
        observed = self._loss_pending_count > 0
        alpha = self._loss_alpha
        nan_mask = torch.isnan(self._loss_map)
        first_obs = observed & nan_mask
        ema_obs = observed & ~nan_mask
        # First observation: initialize directly.
        self._loss_map = torch.where(first_obs, new_mean, self._loss_map)
        # Subsequent observations: standard EMA blend.
        self._loss_map = torch.where(
            ema_obs,
            alpha * self._loss_map + (1.0 - alpha) * new_mean,
            self._loss_map,
        )
        self._loss_pending_sum.zero_()
        self._loss_pending_count.zero_()

        # Per-cluster loss EMA (R3b).
        cnt_k = self._cluster_loss_pending_count.clamp(min=1.0)
        new_k = self._cluster_loss_pending_sum / cnt_k
        observed_k = self._cluster_loss_pending_count > 0
        alpha_k = self._cluster_loss_alpha
        # NaN-safe: first observation initializes; subsequent observations EMA.
        nan_mask = torch.isnan(self._cluster_loss_ema)
        first_obs = observed_k & nan_mask
        ema_obs = observed_k & ~nan_mask
        self._cluster_loss_ema = torch.where(
            first_obs, new_k, self._cluster_loss_ema
        )
        self._cluster_loss_ema = torch.where(
            ema_obs,
            alpha_k * self._cluster_loss_ema + (1.0 - alpha_k) * new_k,
            self._cluster_loss_ema,
        )
        self._cluster_loss_count += self._cluster_loss_pending_count
        self._cluster_loss_pending_sum.zero_()
        self._cluster_loss_pending_count.zero_()

        # Centroid EMA (R3b).
        if (
            self._cluster_centroid_pending_sum is not None
            and self._cluster_init_done
        ):
            cnt_c = self._cluster_centroid_pending_count.clamp(min=1.0)
            new_c = self._cluster_centroid_pending_sum / cnt_c.unsqueeze(-1)
            observed_c = (self._cluster_centroid_pending_count > 0).unsqueeze(-1)
            alpha_c = self._cluster_centroid_alpha
            self._cluster_centroids = torch.where(
                observed_c,
                alpha_c * self._cluster_centroids + (1.0 - alpha_c) * new_c,
                self._cluster_centroids,
            )
            self._cluster_centroid_pending_sum.zero_()
            self._cluster_centroid_pending_count.zero_()

    # ------------------------------------------------------------------
    # Clustering (R3b) helpers
    # ------------------------------------------------------------------

    def _assign_per_image(self, h: torch.Tensor) -> List[torch.Tensor]:
        """Per-image hard assignment to nearest centroid.

        Returns a list of B LongTensors, each (N,), on ``self.device``.
        """
        if not self._cluster_init_done or self._cluster_centroids is None:
            raise RuntimeError(
                "_assign_per_image called before cluster init — guard "
                "this with _cluster_init_done."
            )
        with torch.no_grad():
            h_dev = h.detach().float().to(self.device)        # (B, N, D)
            centroids = self._cluster_centroids               # (K, D), self.device
            # (B, N, K) squared distance — compute on device to avoid CPU sync.
            diff = h_dev.unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0)
            d2 = diff.pow(2).sum(dim=-1)
            assignment = d2.argmin(dim=-1)  # (B, N)
        return [assignment[b] for b in range(assignment.shape[0])]

    def _init_clusters_from(self, h: torch.Tensor) -> None:
        """Rank-0 k-means++ init on a flattened batch; broadcast to all ranks.

        Called once, lazily, when the first ``update_after_iter`` arrives
        with ``h_for_cluster`` set.  Init compute happens on rank-0's local
        CPU (k-means++ inner loop is small and easier to debug there); the
        result is then moved to ``self.device`` before broadcast so the
        collective matches the backend (NCCL = CUDA only).
        """
        D = int(h.shape[-1])
        # Allocate the broadcast buffer on self.device so NCCL is happy.
        centroids = torch.zeros(
            self.n_clusters, D, dtype=torch.float32, device=self.device
        )
        if self.rank == 0:
            tokens = h.detach().float().cpu().reshape(-1, D)
            # k-means++ init on CPU.
            cent_cpu = torch.zeros(self.n_clusters, D, dtype=torch.float32)
            n = tokens.shape[0]
            idx0 = random.randint(0, n - 1)
            cent_cpu[0] = tokens[idx0]
            for k in range(1, self.n_clusters):
                d2 = (tokens.unsqueeze(1) - cent_cpu[:k].unsqueeze(0)).pow(
                    2
                ).sum(dim=-1).min(dim=1).values
                d2 = d2.clamp(min=0.0)
                if float(d2.sum().item()) <= 0.0:
                    next_idx = random.randint(0, n - 1)
                else:
                    probs = d2 / d2.sum()
                    next_idx = int(torch.multinomial(probs, num_samples=1).item())
                cent_cpu[k] = tokens[next_idx]
            centroids.copy_(cent_cpu.to(self.device))
        # Broadcast happens on EVERY rank with a self.device tensor.
        _broadcast(centroids, src=0)
        self._cluster_centroids = centroids
        self._cluster_centroid_pending_sum = torch.zeros(
            self.n_clusters, D, device=self.device
        )
        self._cluster_init_done = True

    def _update_clusters(
        self,
        h_for_cluster: torch.Tensor,
        per_token: torch.Tensor,
        positions_flat: torch.Tensor,
        fm: Optional[torch.Tensor] = None,
        assignment: Optional[torch.Tensor] = None,
    ) -> None:
        """Online mini-batch k-means update + per-cluster loss accumulation.

        ``h_for_cluster``: (B, N, D) on any device — moved to ``self.device``.
        ``per_token``:    (npred, B, K_pred) on ``self.device``.
        ``positions_flat``: (npred, B, K_pred) on ``self.device``.
        ``fm``: optional (npred, B, K_pred) float finite-mask in {0,1};
            None means all finite.
        ``assignment``: optional (B, N) LongTensor of per-token cluster ids
            on ``self.device``.  When provided (the common path — populated
            by ``generate`` from the same ``h_for_cluster`` snapshot), we
            skip the argmin recompute.  Set to None to recompute fresh.
        """
        if not self._cluster_init_done:
            self._init_clusters_from(h_for_cluster)
            # First call: skip the update — assignment isn't trustworthy yet.
            return

        with torch.no_grad():
            h_dev = h_for_cluster.detach().float().to(self.device)  # (B, N, D)
            B, N, D = h_dev.shape
            centroids = self._cluster_centroids  # (K, D) on self.device

            if assignment is not None:
                # Validate shape — guard against stale cache from a prior
                # batch with a different B.  Cheap; fails loud if mismatched.
                if assignment.shape != (B, N):
                    raise RuntimeError(
                        f"_update_clusters: cached assignment shape "
                        f"{tuple(assignment.shape)} does not match expected "
                        f"({B}, {N}) — stale cache?"
                    )
                assign = assignment.to(self.device, non_blocking=True)
            else:
                # Fallback: recompute the per-token assignment for the whole
                # batch — on device.  Only hit when the caller didn't run
                # generate() (e.g. tests, or a future code path).
                diff = h_dev.unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0)
                d2 = diff.pow(2).sum(dim=-1)
                assign = d2.argmin(dim=-1)  # (B, N)

            # Centroid sums + counts.
            flat_h = h_dev.reshape(-1, D)              # (B*N, D)
            flat_a = assign.reshape(-1)                # (B*N,)
            cent_sum = torch.zeros(
                self.n_clusters, D, device=self.device
            )
            cent_cnt = torch.zeros(self.n_clusters, device=self.device)
            cent_sum.scatter_add_(
                0, flat_a.unsqueeze(-1).expand(-1, D), flat_h
            )
            cent_cnt.scatter_add_(
                0, flat_a, torch.ones_like(flat_a, dtype=torch.float32)
            )

            _maybe_all_reduce(cent_sum)
            _maybe_all_reduce(cent_cnt)

            # Pending accumulator — fold on is_step.
            self._cluster_centroid_pending_sum = (
                self._cluster_centroid_pending_sum + cent_sum
            )
            self._cluster_centroid_pending_count = (
                self._cluster_centroid_pending_count + cent_cnt
            )

            # Per-cluster loss: look up the cluster ID at each (image, pred
            # position) and accumulate the per-token loss into per-cluster
            # sums.  positions_flat is (npred, B, K) — gather along N.
            npred = positions_flat.shape[0]
            K = positions_flat.shape[-1]
            assign_for_pred = torch.gather(
                assign.unsqueeze(0).expand(npred, -1, -1),
                dim=2,
                index=positions_flat,
            )  # (npred, B, K)
            flat_cluster = assign_for_pred.reshape(-1)
            flat_loss = per_token.reshape(-1)
            if fm is not None:
                flat_finite = fm.reshape(-1)
            else:
                flat_finite = torch.ones_like(flat_loss)
            cl_sum = torch.zeros(self.n_clusters, device=self.device)
            cl_cnt = torch.zeros(self.n_clusters, device=self.device)
            cl_sum.scatter_add_(0, flat_cluster, flat_loss * flat_finite)
            cl_cnt.scatter_add_(0, flat_cluster, flat_finite)

            _maybe_all_reduce(cl_sum)
            _maybe_all_reduce(cl_cnt)

            self._cluster_loss_pending_sum += cl_sum
            self._cluster_loss_pending_count += cl_cnt

    # ------------------------------------------------------------------
    # Stateless DataLoader collate (used in place of the collator object)
    # ------------------------------------------------------------------

    @property
    def mirage_stats(self) -> dict:
        """Per-batch MIRAGE masking statistics from the last ``generate`` call."""
        return dict(self._mirage_stats)

    @staticmethod
    def stack_collate(batch):
        """Stack a list of (C, H, W) tensors into a (B, C, H, W) batch.

        Replaces ``MaskCollator`` as the train ``DataLoader.collate_fn`` when
        curriculum is enabled.  Mask generation moves into the training loop
        because it depends on stateful generator + ``h_for_cluster``.
        """
        return torch.stack(batch, dim=0)

    # ------------------------------------------------------------------
    # State serialization (for checkpoint save/load)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a dict suitable for inclusion in a torch checkpoint."""
        return {
            "version": 1,
            "r_t": float(self._r_t),
            "epoch": int(self._epoch),
            "iter": int(self._iter),
            "loss_map": self._loss_map.clone(),
            "loss_count": self._loss_count.clone(),
            "loss_pending_sum": self._loss_pending_sum.clone(),
            "loss_pending_count": self._loss_pending_count.clone(),
            "cluster_centroids": (
                self._cluster_centroids.clone()
                if self._cluster_centroids is not None
                else None
            ),
            "cluster_init_done": bool(self._cluster_init_done),
            "cluster_loss_ema": self._cluster_loss_ema.clone(),
            "cluster_loss_count": self._cluster_loss_count.clone(),
            "cluster_loss_pending_sum": self._cluster_loss_pending_sum.clone(),
            "cluster_loss_pending_count": self._cluster_loss_pending_count.clone(),
            "cluster_centroid_pending_sum": (
                self._cluster_centroid_pending_sum.clone()
                if self._cluster_centroid_pending_sum is not None
                else None
            ),
            "cluster_centroid_pending_count": self._cluster_centroid_pending_count.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state previously saved by :meth:`state_dict`.

        All tensors are moved onto ``self.device`` so subsequent collective
        calls (which use the same tensors as buffers) match the DDP backend
        (NCCL requires CUDA).  If the saved checkpoint pre-dates the device
        param (state was CPU), ``.to(self.device)`` migrates it cleanly.

        Validates that every restored tensor shape matches the current
        config so a silently-changed knob (e.g. ``n_clusters: 4`` -> ``8``,
        or ``patch_size: 16`` -> ``8``) fails LOUD here rather than later
        with an opaque scatter/view error mid-train.
        """
        if state is None:
            return
        if not state:
            # Empty dict but present — corrupt save.  Refuse to silently
            # cold-start; the operator should explicitly pass mask_gen=None
            # to opt out of restore.
            raise RuntimeError(
                "Curriculum state present in checkpoint but empty — refusing "
                "to silently cold-start.  Either fix the checkpoint or pass "
                "mask_gen=None to deliberately discard the state."
            )

        def _to(t):
            return t.detach().clone().to(self.device) if t is not None else None

        def _check_shape(name: str, loaded: torch.Tensor, expected: tuple) -> None:
            got = tuple(loaded.shape)
            if got != expected:
                raise RuntimeError(
                    "Curriculum restore: shape mismatch on '%s' — checkpoint "
                    "has %s, current config expects %s.  A config knob "
                    "(patch_size, n_clusters, num_pred_masks, ...) likely "
                    "changed since the checkpoint was saved." %
                    (name, got, expected)
                )

        self._r_t = float(state.get("r_t", 0.0))
        self._epoch = int(state.get("epoch", 0))
        self._iter = int(state.get("iter", 0))

        # ---- Loss-map tensors (H, W) — driven by patch_size + image size.
        hw = (self.height, self.width)
        loss_map = _to(state["loss_map"])
        _check_shape("loss_map", loss_map, hw)
        self._loss_map = loss_map
        loss_count = _to(state["loss_count"])
        _check_shape("loss_count", loss_count, hw)
        self._loss_count = loss_count
        lps = _to(state["loss_pending_sum"])
        _check_shape("loss_pending_sum", lps, hw)
        self._loss_pending_sum = lps
        lpc = _to(state["loss_pending_count"])
        _check_shape("loss_pending_count", lpc, hw)
        self._loss_pending_count = lpc

        # ---- Cluster tensors — driven by n_clusters and embed_dim.
        centroids = state.get("cluster_centroids", None)
        if centroids is not None:
            # Embed dim is unknown until first forward; validate K only.
            if centroids.shape[0] != self.n_clusters:
                raise RuntimeError(
                    "Curriculum restore: cluster_centroids has K=%d, "
                    "current config expects n_clusters=%d." %
                    (centroids.shape[0], self.n_clusters)
                )
        self._cluster_centroids = _to(centroids)
        self._cluster_init_done = bool(state.get("cluster_init_done", False))

        k = (self.n_clusters,)
        cle = _to(state["cluster_loss_ema"])
        _check_shape("cluster_loss_ema", cle, k)
        self._cluster_loss_ema = cle
        clc = _to(state["cluster_loss_count"])
        _check_shape("cluster_loss_count", clc, k)
        self._cluster_loss_count = clc
        clps = _to(state["cluster_loss_pending_sum"])
        _check_shape("cluster_loss_pending_sum", clps, k)
        self._cluster_loss_pending_sum = clps
        clpc = _to(state["cluster_loss_pending_count"])
        _check_shape("cluster_loss_pending_count", clpc, k)
        self._cluster_loss_pending_count = clpc

        cps = state.get("cluster_centroid_pending_sum", None)
        if cps is not None and cps.shape[0] != self.n_clusters:
            raise RuntimeError(
                "Curriculum restore: cluster_centroid_pending_sum has K=%d, "
                "current config expects n_clusters=%d." %
                (cps.shape[0], self.n_clusters)
            )
        self._cluster_centroid_pending_sum = _to(cps)
        ccpc = _to(state["cluster_centroid_pending_count"])
        _check_shape("cluster_centroid_pending_count", ccpc, k)
        self._cluster_centroid_pending_count = ccpc
