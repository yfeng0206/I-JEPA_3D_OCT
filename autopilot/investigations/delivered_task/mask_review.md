# Independent mask-fix review and adjudication

Reviewer: `mask-fix-critic`, GPT-5.6 Sol Fast xhigh.
The coordinator captured the specialist's returned findings here because the
specialist did not leave the requested report file.

## Findings returned

1. COVER fallback conflated valid-but-infeasible guides with invalid guides,
   and applied one fallback reason to all target slots, overwriting unguided
   provenance for false ramp flags. This affected diagnostic labels/counters,
   not the final masks.
2. The historical parity helper executed old curriculum source but imported
   current COVER code, making the COVER controls circular. The reviewer
   independently bound historical COVER code and observed current tensor parity,
   but required the shipped validator itself to be independent.

## Owner repair and parent acceptance

The owner separated invalid-guide, valid-infeasible and unguided provenance,
versioned the diagnostic schema, and added sparse/mixed-ramp regressions.
The parity helper now isolates baseline dependencies, restores imports and
detects an adversarial mutation of current COVER behavior.

The owner recorded five pre-fix failures, 123 post-fix passing tests, 15 isolated
legacy parity controls and a scan of 5,400 saved rows with zero source-count
changes. Evidence remains in `evidence\mask_critic_fix_v1`.

The coordinator's integrated training/mask/identity/weight suite subsequently
passed 149 tests (`evidence\coordinator_engineering_tests.xml`).
The review does not establish an AUC benefit of corrected masking.
