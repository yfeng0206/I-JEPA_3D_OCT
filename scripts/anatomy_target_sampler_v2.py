"""Backwards-compatible shim.

The anatomy target sampler now lives in ``src/masks/anatomy.py`` because it
is production code that the mask collator imports, not a research script.
The ~15 probe and figure scripts that already say

    import anatomy_target_sampler_v2 as A

keep working through this re-export, so moving it did not invalidate any of
the recorded measurements.
"""
from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.masks.anatomy import *          # noqa: F401,F403
from src.masks.anatomy import (          # noqa: F401
    allocate,
    build_targets,
    build_targets_fixed_cells,
    geodesic_partition,
    grow_components,
    grow_components_fixed_cells,
    is_viable,
    n_components,
    rebalance,
    region_capacity,
)
