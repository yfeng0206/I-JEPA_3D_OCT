# Masking experiments

The authoritative record for MIRAGE-guided anatomy masking is:

- **[`ablations.md`](ablations.md)** — method, mask-budget and sampler
  ablations, adapter sweep, guardrails, bug fixes, rejected designs, blockers,
  corrections, and reproduction paths.
- **[`findings.md`](findings.md)** — consolidated quantitative findings from
  all masking experiments: three-arm comparison, collation fix, timing, guide
  cache, VRAM, slice-depth validation, and open questions.

Related documents retained because they cover distinct experiment lines:

- [`../curriculum_masking.md`](../curriculum_masking.md) — historical
  oracle/self-guided/curriculum research plan.
- [`../pretraining/mirage_100ep.md`](../pretraining/mirage_100ep.md) — completed
  legacy MIRAGE-envelope pretraining arm.
- [`../frozen/mirage_meanpool_sweep.md`](../frozen/mirage_meanpool_sweep.md) —
  downstream frozen-probe comparison of random, oracle, and the legacy
  MIRAGE-envelope arm.

`../mirage_guided_masking.md` is retained only as a compatibility pointer for
older links. New masking findings belong in `ablations.md`.
