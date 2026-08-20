# Ranked falsification experiments

**Ranking rule:** expected causal value divided by incremental GPU-hours. **[PROPOSED]** CPU-only analyses of saved artifacts are first and cost **0 GPU-hours**.

## 1. Paired uncertainty on exact anatomy/background contributions

- **[PROPOSED arm/data]** All four ep50 encoders using the already saved arrays `C_an`, `C_bg`, `logit`, and `label` in `D:\jepa_phase0\reports\patch_attribution\*_attrib.npz`.
- **[PROPOSED analysis]** Paired stratified bootstrap over the 1,000 test volumes, 10,000 resamples. Report CIs for anatomy-contribution AUC, background-contribution AUC, their paired difference, and the AUC loss when either contribution is replaced by its cohort mean.
- **[COST]** **0 GPU-hours**; CPU/post-hoc only; runtime not benchmarked.
- **[CONFIRMS]** Background-position contribution retains a clearly above-chance AUC and is not reliably weaker than anatomy contribution in blob.
- **[KILLS]** Wide intervals spanning chance or a robust anatomy-over-background advantage in blob.
- **[VALUE]** Highest-value free test because the current attribution table has no uncertainty.

## 2. Conditional information using saved region features

- **[PROPOSED arm/data]** `random_ep50_s100.pt`, `oracle_ep50_s100.pt`, `envelope_ep50_s100.pt`, and `blob_ep50_s100.pt` in `D:\jepa_phase0\reports\region_features\`.
- **[PROPOSED analysis]** Fit pre-registered L2 logistic heads on: anatomy features, background features, concatenated anatomy+background features, and background residualized on anatomy without labels. Use Training n=2,000, tune regularization on Validation n=600, evaluate once on Test n=1,000; repeat probe seeds 42–46 and paired-bootstrap test differences.
- **[COST]** **0 GPU-hours**; CPU/post-hoc only; no encoder forward; runtime not benchmarked.
- **[CONFIRMS]** Background features improve joint test AUC beyond anatomy-only, especially for non-blob arms, or residual background remains predictive.
- **[KILLS]** Joint features do not improve anatomy-only and residual background is at chance.
- **[VALUE]** Distinguishes redundant global mixing from complementary regional information without new representation learning.

## 3. Eroded-background regional probe

- **[PROPOSED arm]** Rerun the ep50 regional pooling/attribution evaluation for random, oracle, envelope, and blob.
- **[PROPOSED knob]** Erode the nominal background pool away from anatomy by patch-grid radius `r ∈ {0,1,2,3,4}`; equivalently dilate the anatomy exclusion mask before background pooling.
- **[COST]** **~10 min/arm, no training**, documented in `docs\experiments\masking\cover_random_campaign.md:332-333`; **[INFERRED arithmetic]** four arms imply **~0.67 GPU-hours** if run sequentially.
- **[CONFIRMS black-pixel/background-region signal]** Background-only AUC remains high several patch rows from the retina.
- **[KILLS black-pixel/background-region signal]** AUC collapses with erosion while anatomy AUC remains stable.
- **[LIMITATION]** Even eroded token positions retain global self-attention; combine with experiment 4.

## 4. Background-content shuffle with positions and token count fixed

- **[PROPOSED arm]** Reuse the frozen background-signal protocol on healthy random/oracle/envelope checkpoints and blob ep30/40/50/56.
- **[PROPOSED intervention]** For background context positions, replace encoded content with the same-position token from another image while preserving positions, number of tokens, target indices, and attention shape. Compare with within-image random-token replacement and anatomy-token shuffle.
- **[PROPOSED endpoint]** Paired change in anatomy-target and background-target prediction error relative to full context.
- **[COST]** No training; forward-only GPU inference. Existing protocol uses 108 slices, but wall time was not benchmarked, so GPU-hours are **unknown rather than invented**.
- **[CONFIRMS]** Cross-image background-content shuffle hurts more than count-matched replacement.
- **[KILLS]** Shuffle is indistinguishable from generic token replacement; background’s semantic content is then unsupported.

## 5. Fixed-budget background-target rescue

- **[PROPOSED arms]** Two new paired continuations from the same ep25 fork and fixed pretraining seed: blob baseline and blob-plus-random-target rescue.
- **[PROPOSED knobs]** Hold `num_pred_masks: 4` and `pred_target_k: 16`; compare `r_max: 1.0` against `r_max: 0.75`. In `mirage_anatomy`, the Bernoulli ramp is per image rather than per block (`docs\experiments\masking\anatomy_vs_rectangle_ep30.md:29-38`), so the rescue makes about 25% of images uniform at full ramp rather than making one target group per image uniform. Audit realized composition before launch.
- **[PROPOSED epochs]** ep26→50, then the identical frozen mean-pool probe.
- **[COST]** Current measured basis is 58.8 min/epoch plus 60 min/probe (`D:\jepa_phase0\runs\cover_f021_ep25\RESUME_NOTE.txt:38`). **[INFERRED]** One 25-epoch arm costs about **25.5 GPU-hours**; a paired two-arm comparison costs **51 GPU-hours per pretraining seed**, or **153 GPU-hours for three paired seeds**.
- **[CONFIRMS background-target hypothesis]** Restoring uniform/background targets improves predictor health and downstream AUC without changing target count.
- **[KILLS]** No improvement despite a verified target-purity shift.
- **[LIMITATION]** The rescue also mixes target geometry; it isolates composition better than the current comparison but not perfectly.

## 6. Fixed-purity anatomy-context rescue

- **[PROPOSED arms]** Paired blob continuations from the same ep25 fork.
- **[PROPOSED knob]** Keep `r_max: 1.0`, `pred_target_k: 16`, and connected-blob geometry; compare `anatomy_mass_cap: 0.90` with `0.75`. First run the existing CPU composition audit to verify that target purity stays near-pure while more anatomy remains visible.
- **[PROPOSED epochs]** ep26→50 plus frozen probe and predictor-health measurements at ep30/40/50.
- **[COST]** **[INFERRED]** 51 GPU-hours per paired seed; 153 GPU-hours for three paired seeds, using the same measured timing basis as experiment 5.
- **[CONFIRMS anatomy-starvation hypothesis]** More visible anatomy prevents the predictor-health inversion and improves AUC while target purity remains near-pure.
- **[KILLS]** Predictor and downstream results remain unchanged after a verified increase in absolute anatomy context.

## 7. Target-count/context-budget 2×2

- **[PROPOSED arms]** `pred_target_k ∈ {16,30}` × `r_max ∈ {1.0,0.75}`, all from the same ep25 fork with fixed seeds.
- **[MEASURED design basis]** The stored budget audit reports k=30 gives mean target union 117.077 and context 138.923, versus k=16 union 62.610 and context 193.390 (`D:\jepa_phase0\reports\budget_masks\budget_mask_audit_fairvision.json`, keys `"16"` and `"30"`).
- **[PROPOSED epochs]** ep26→50, checkpoints at ep30/40/50, then frozen probes.
- **[COST]** **[INFERRED]** Four arms cost about **102 GPU-hours per pretraining seed** and **306 GPU-hours for three seeds**.
- **[CONFIRMS underconstraint mechanism]** k=30 improves predictor health regardless of `r_max`, while `r_max:0.75` adds an independent gain.
- **[KILLS]** Neither target count nor restored random targets changes the result.
- **[VALUE]** Most definitive factorial design, but lowest value/GPU-hour.

## Ongoing evidence that is not a falsification

- **[MEASURED]** COVER f0.21 is a live, read-only run with target purity 40.88%, anatomy hidden 73.09%, and context anatomy 14.55%; its AUC is pending.
- **[LIMITATION]** A positive COVER result would be consistent with avoiding the blob extreme but would not isolate background targets, because geometry and context budget still differ.
- **[RULE]** Do not change, restart, or inspect checkpoints beyond read-only logs as part of these analyses.
