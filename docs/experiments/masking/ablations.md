# Masking ablations — moved

This file used to mix method setup, comparisons, sampler ablations, adapter ablations, engineering notes, corrections, and paper-planning inventory. The content has been split into focused pages:

- [`method_setup.md`](method_setup.md) — method definitions, sampler pipeline, config pointers, and historical method context.
- [`comparison.md`](comparison.md) — downstream and mask-budget comparisons.
- [`sampler_ablations.md`](sampler_ablations.md) — mask-budget, mass-cap, collation, coverage, region-growth, integration, and rejected sampler designs.
- [`adapter_ablations.md`](adapter_ablations.md) — adapter sweep, guardrails, saturation, and guide-generation precision.
- [`engineering_notes.md`](engineering_notes.md) — efficiency, cache, VRAM, limitations, corrections, blockers, and paper inventory.

Terminology correction: `mirage_envelope` means MIRAGE-placed rectangles and is a baseline; `mirage_anatomy` means MIRAGE-shaped connected anatomy targets and is the contribution.
