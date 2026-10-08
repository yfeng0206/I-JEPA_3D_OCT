# Camera-ready plan: GenAI4Health @ NeurIPS 2026 (Accept, Poster), Submission 252

Status: PLAN ONLY (no code changed, no GPU job launched). Version 2, 2026-10-08 ~01:45 PDT,
revised after an independent plan critique.
Branch: `poster-ready` (= main `b941814`). Evidence reports (local Copilot session workspace, `files\camera_ready\`; not committed because they quote confidential review text):
`01_provenance.md` (A1), `02_seed_code_verification.md` (A2), `03_matched_control.md` (A3),
`04_eval_analysis.md` (A4), `05_external_research.md` (A5), `06_review_triage_paper_plan.md` (A6).
Labels: [M] measured, [I] inferred, [A] assumption/estimate.

---

## 0. Deadline arithmetic

| Item | Value |
|---|---|
| Hard deadline | Sun Oct 25 2026 23:59 AoE = **Mon Oct 26 04:59 PDT** |
| Internal upload target | **Sat Oct 24** (Oct 25 is buffer) |
| Results freeze (last GPU result accepted) | **Thu Oct 22 08:00 PDT** |
| Today | Thu Oct 8 |
| Page limit | 10 main pages (excl. acks, refs, appendix); v9 main text = 7.58 pages [M] |
| Template | `genai4health_2026.sty` (loads neurips_2026 [dblblindworkshop,final]); authors visible |

---

## 1. What the reviews require (A6 triage: 34 points)

21 real (9 must-fix, 12 should-fix), 6 partly addressed already, 2 nitpicks, 5 out of scope.

| Cluster | Who | Verdict | Response |
|---|---|---|---|
| Seeds: one run per strategy; ~0.01 AUC unchecked against seed variance | AC (explicit), ZTfW, PGP8, GZo3 | REAL, P0 | 2 new continuation seeds x RANDOM/CENTROID/ENVELOPE, ep25 -> ep50 |
| Confound: visible context / target size / loss slots | AC, ZTfW, PGP8 Q1, GZo3 | REAL; P0 writing, P1 experiment | Delivered context in a main table (re-measured, old and new shown); RANDOM-CB matched-budget run paired with CENTROID s1234 |
| Adaptive test reuse | AC, ZTfW, PGP8 | REAL (procedure) | Committed analysis plan; gates use validation only; new-run test AUCs unsealed once, after the freeze |
| Pooling | PGP8 Q3, AC | REAL, cheap | Patch-max within B-scan (same encoder pass) + across-slice max/mean+max |
| CENTROID failure rate | PGP8 Q4 | REAL, CPU | Band-vs-envelope agreement / miss rate, stratified |
| MIRAGE quality | AC, PGP8 Q5, GZo3 | REAL, characterize only | Guide producer (MIRAGE-Large fine-tuned on GOALS; in-domain Dice 0.925 as context only), failure/plausibility rates |
| External model | AC, GZo3 | SHOULD | FairVision supervised 3D ResNet 0.8649 (cite, free); optional RETFound_mae_natureOCT probe |
| Strategy clarity; Fig. 1 shows a subset | GZo3 | partly misread; figure point valid | Strategy table; all-strategy figure |
| ANATOMY-V2 early stop; V1 provenance | AC, PGP8 | disclosed | Keep; V1 detail to appendix |
| One dataset/task | all | out of scope | Future work; MD severity regression is only a weak partial answer |
| Theory-founded masking; beyond I-JEPA | GZo3 | out of scope / nitpick | Short discussion paragraph (A5 references) |

---

## 2. Verified facts that shape the plan

1. **ENVELOPE code drift [M].** `fc49f61` (Aug 8, after ENVELOPE finished Aug 4) changed the overlap fallback; HEAD masks differ from run-era candidate `804c639`. RANDOM and CENTROID samplers are bitwise identical to run-era code [M]. `804c639` is a candidate, not a certified producing revision: certify the legacy flag against the run's logged mask statistics.
2. **Original seeds/RNG/git SHA not recorded [M].** Exact replay is impossible; gates compare curves, mask statistics and AUC bands.
3. **Config seed does not change data order [M].** Fix so seeds vary order. Same seed index across arms then shares data order only (not crops or masks).
4. **Trainer semantics changed since originals [M]** (IPE ceil, tail normalization, overflow-aware LR/WD/EMA). New runs = corrected-code continuations; originals reported separately.
5. **Keep `epochs: 100` and stop after the saved ep50 [M].** A 50-epoch config changes LR/WD/EMA from the first step.
6. **Effective batch 512 everywhere [M]** (RANDOM accum 2 from ancestor Adam step 29,292).
7. **Existing chain is unsafe [M]**, and `runs\rep_random_s1234` still holds a stale Aug-26 epoch-26 checkpoint the supervisor would resume. Use a new namespace.
8. **Teacher targets were fp32 in all three originals [M]** (source for every candidate revision; runtime log for ENVELOPE), matching new runs with `amp_target: false`.
9. **Probe RNG depends on cache state [M];** existing fp32 caches predate `73e0b55` and are rejected by the current evaluator by default. Original anchors must be re-extracted on the GPU under the frozen new evaluator.
10. **All nine headline probes were fp16 [M];** fp32 re-probes of 8/9 match within 2e-4. New tables are fp32 only.
11. **Table 2 delivered-context used non-production settings and an unpaired protocol [M].** CENTROID lateral 0.8 (trained 0.6), ENVELOPE soft guides + new sampler. Current-code replay (3 audit seeds, small number of budget draws): RANDOM 27.8 / CENTROID 31.4 / ENVELOPE 28.9 (trained, pre-fix sampler; 27.0 with HEAD) % of grid, audit-seed SD 1.9 / 2.7 / 1.5; published 24.7 / 32.9 / 30.7. Part of the change (RANDOM 24.7 -> 27.8) is protocol and Monte-Carlo noise, not settings. Re-measure with >= 100 budget draws per seed and report the paired gap; show old and new with a one-line reason. Figures 1-2 generators also use 0.8.
12. **RANDOM ep50/75/100 weights are not on disk [M];** HF returns 401 without a token. Needed for the RANDOM anchor re-extraction: the token is a blocking input by Fri Oct 9.
13. **Machine [M]:** RTX 3090 24 GB, 315 W cap (logon task), game client on the GPU, RAM 31.9 GB with 0.6 GB free during training at 6 workers, C: 24 GB free (slice cache + pagefile; past error 1455), D: 583 GB free.

---

## 3. Time and GPU budget

### 3.1 Per-run cost on the local 3090 (ep25 -> ep50 = 25 epochs)

| Run | min/epoch | Training | + probe | Total | Uncertainty |
|---|---|---|---|---|---|
| RANDOM | ~69 [M, n=1 epoch, cap unknown] | 28.8 h | 1.25 h | ~30 h | +-10% throughput = +-3 h |
| CENTROID | 69-85 [I] | ~32 h | 1.25 h | ~33 h | up to ~37 h |
| ENVELOPE | 80.5 [M] | 33.5 h | 1.25 h | ~35 h | |
| RANDOM-CB | basic matcher in main process ~94 min [I]; strict (H-matched) ~123 min [I]; in workers 72-86 min only if verified | 39 h basic / 51 h strict / 30-36 h workers | 1.25 h | plan ~40 h (basic) | time end to end in the smoke test; use strict only if worker delivery is verified |

Probe: ~60 min solo [M]; never run two probes concurrently (each ~2.6 h). Disk ~14 GB per run on D:.

### 3.2 Schedule (local 3090 only)

Pre-launch GPU work on Fri Oct 9 (after Stage 1 fixes pass): smoke tests ~3 h (all four arm configs ~200 iterations each, stop/resume, throughput) and original-anchor re-extraction ~4 h (R/C/E ep50 under the frozen new evaluator; also the evaluator-drift check).

| # | Run | Seed | Hours | Planned end (PDT), R1 starts Sat Oct 10 00:00 | Milestone |
|---|---|---|---|---|---|
| 1 | RANDOM | 1234 | 30 | Sun Oct 11 06:00 | G1 at ep26-27; first slip projection; G2 at ep50 |
| 2 | CENTROID | 1234 | 33 | Mon Oct 12 15:00 | G1 identity check ep31-33 |
| 3 | ENVELOPE (certified legacy sampler) | 1234 | 35 | Wed Oct 14 02:00 | seed set 1 done |
| 4 | RANDOM | 5678 | 30 | Thu Oct 15 08:00 | within-protocol RANDOM spread available |
| 5 | CENTROID | 5678 | 33 | Fri Oct 16 17:00 | primary contrast at 2 seeds |
| 6 | RANDOM-CB (CENTROID s1234 budgets, basic U/L/C matcher unless strict is verified fast) | 1234 | 40 (31-52) | Sun Oct 18 09:00 | matched control done |
| 7 | ENVELOPE (legacy) | 5678 | 35 | Mon Oct 19 20:00 | seed set 2 done |
| 8 | Optional RETFound frozen probe | - | ~4 | Tue Oct 20 00:00 | GPU campaign done |

- **GPU total ~247 h** (smoke 3 + anchors 4 + runs ~236 + RETFound 4) of ~310 h between Fri Oct 9 noon and the Oct 22 08:00 freeze (~80% duty).
- **Nominal slack ~56 h; realistic 20-40 h** after throughput uncertainty (+-24 h over the campaign) and RANDOM-CB risk (+12 h if strict). Re-project the end date after every epoch.
- **Fallback order:** drop E2 first, then RANDOM-CB. Decision points: R1 ep27 and C1 ep28 (throughput known), then each run end. Runs 1-5 are the protected core (R and C at two new seeds each, E at one).
- **Not feasible locally:** a third seed per arm (R3+C3 ~63 h), or ep100 horizons.

### 3.3 Optional acceleration (user decision)
- Azure 4x T4 (~35 min/epoch, ~15 h per run): move **whole seed indices only** (all arms of one seed on the same hardware), otherwise paired contrasts mix hardware. With it, a third seed index becomes feasible by ~Oct 16. Setup ~0.5-1 day.
- No gaming on the PC while a run is active. Free C: to >= 40 GB or move the pagefile to D:. Keep or raise the 315 W cap (decide before launch; keep it fixed for the whole campaign).

### 3.4 Gates (pre-declared)

| Gate | When | Rule |
|---|---|---|
| G0 (CPU) | before smoke | Ancestor SHA; optimizer state loaded (Adam step 29,292, scaler 2^20, first-update LR/WD/EMA as expected); golden masks: RANDOM/CENTROID HEAD == run-era; ENVELOPE legacy flag == `804c639` on fixtures **and** CPU replay on real slices + hard guides matches the run's logged mask statistics (unique_targets ~119-121, accept rate ~0.48, fallbacks ~0.87); config diffs vs archived configs only in declared keys; new empty output folders; run launched from a pinned commit/worktree, resume refused on source-hash change |
| Smoke | Fri Oct 9 | All four configs, ~200 iterations, stop/resume, measured throughput; RANDOM-CB >= training rate |
| G1 loss | ep26-27 | Absolute bands vs reference curves (A1 section 3; run-to-run noise ~0.002): warn > 0.003, hold > 0.005 on two consecutive epochs; NaN/Inf, wrong precision/horizon, K collapse, leakage = stop |
| G1 identity | ep31-33 (first full-guidance epochs) | Train loss closer to its own arm's reference curve than to the other arms'; ENVELOPE logged mask statistics within the fingerprint ranges |
| G2 reproduction | each ep50 probe | **Validation AUC only** vs the same arm's original re-extracted under the same evaluator. RANDOM is the engineering anchor (guided originals were chosen after test inspection and may regress). Decision tree below |
| Freeze | Thu Oct 22 08:00 | Unfinished runs reported as not completed; no partial epochs |

**G2 decision tree** (single crossings of 0.01 are expected by chance: with seed SD ~0.005, P(at least one of 7 runs crosses) ~60%):
- Common shift across arms, losses and mask stats clean: continue; contrasts within the new runs remain valid; do not pool with originals.
- One arm shifts **and** its loss or mask statistics mismatch: stop that arm, fix, rerun (pay with E2/CB).
- AUC shifts but loss and mask statistics are clean: report as variance; no GPU time on diagnosis.

---

## 4. Stages

### Stage 0 - Decisions (Thu Oct 8)
See section 7. Blocking for launch: approval, HF token, C:/pagefile, ENVELOPE route, no-gaming window.

### Stage 1 - Engineering fixes + verification (Thu Oct 8 - Fri Oct 9 noon)
Each change: minimal code + unit test + independent gpt-6-astra (xhigh) verification. **Must land before R1:** E1, E3, E4, E5, E6, E7, E8, E10, E11 (probe parts), E18, E19, plus the committed analysis plan. **Before run 3:** E2 (certified). **Before run 6:** E9. **Before first probe (Oct 11):** E10, E16, E18 frozen.

| ID | Change | Pri |
|---|---|---|
| E1 | Trainer-native `--stop-after-epoch N`, exits only after atomic save + hash | P0 |
| E2 | ENVELOPE sampler version flag (`legacy_uniform_v1` / `least_overlap_v2`), certified per G0 | P0 |
| E3 | Chain fail-closed: exact `epoch==50` + hash before pinning; propagate failures; validate probe identity; no relabelling | P0 |
| E4 | Resume forces `resume_policy: exact`, removes fork fields | P1 |
| E5 | Health gate counts distinct epochs; NaN/Inf parse | P1 |
| E6 | `meta.seed` -> `DistributedSampler(seed=...)` | P1 |
| E7 | Atomic campaign lock + GPU ownership preflight | P1 |
| E8 | Per-run provenance: source hashes, rendered config, guide/cache manifests, runtime flags | P1 |
| E9 | RANDOM-CB sampler in DataLoader workers (CENTROID shadow masks at lateral 0.6; match U, L, C, H; ramp-0 bypass; no silent cap) | P1 |
| E10 | Probe reseeds after cache creation; 5 head seeds; records cache path | P0 |
| E11 | Seed-aware stats inventory (run UUID, checkpoint digest, seed) | P0 |
| E12 | Mask-audit scripts use trained settings, >= 100 budget draws/seed, record sampler hashes | P0 |
| E13 | Figure generators lateral 0.6 | P0 |
| E14 | `table_fp32` prints `<0.001` | P1 |
| E15 | Camera-ready release gates (authors present, PAGE_LIMIT 10, stage `genai4health_2026.sty`, Overleaf FILE_MAP, Word author field, receipts) | P0 release |
| E16 | Patch-max collected with patch-mean in one encoder pass | P1 |
| E17 | RETFound adapter (224, ImageNet norm, patch-token mean) | P2 |
| E18 | Sealed test: probe writes hashed test predictions without computing/printing test AUC until unsealed after freeze | P0 |
| E19 | New run namespace `cr_seed_v1_<arm>_s<seed>`, explicit `resume_policy: fork`, `fork_start_epoch: 25`, sampler-version key; refuse non-empty output folder | P0 |

### Stage 2 - Reproduction gate (Fri Oct 9 - Sun Oct 11)
G0 -> smoke -> anchors (original R/C/E ep50 re-extracted, validation AUC + sealed test) -> R1 with G1 -> R1 ep50 G2 on validation.

### Stage 3 - Campaign (Oct 10 - Oct 19)
Runs 2-8 through the fixed chain; monitoring every 2 h (loss bands, mask fingerprint, disk, RAM, projected end).

### Stage 4 - CPU analyses (front-loaded Oct 8-9 while the GPU is idle; afterwards only light jobs)
1. Table 2 / delivered-context regeneration (E12) for all arms; old vs new comparison.
2. CENTROID robustness: band-vs-envelope agreement and miss rate on a pre-declared stratified sample.
3. MIRAGE characterization (guide producer, empty/discontinuous envelope rate, plausibility).
4. All-strategy figure; Figures 1-2 regenerated at lateral 0.6.
5. After runs: across-slice pooling and MD severity regression from new caches (small, sequential).

### Stage 5 - Paper (Oct 9 - Oct 22)
- Oct 9-12: template migration (already compiles in scratch), author block, acks; writing-only edits: confound in main text, strategy table, centroid formula, guide provenance, FairVision 0.8649 reference row, ANATOMY-V1 to appendix, p-value formatting.
- Oct 12-19: pre-write the outcome versions; fill tables as validation results land.
- Oct 20-21: unseal test predictions once; stats; assign outcome by pre-declared rules; insert text; regenerate receipts.
- Oct 22: text freeze candidate; independent review (opus 5.5 xhigh) + numeric verification (gpt-6-astra xhigh).

### Stage 6 - Release (Oct 22 - Oct 24)
Full p13 build with camera-ready gates; Overleaf sync; push `poster-ready`, PR to `main` after approval; user uploads PDF and updates title/abstract/TL;DR on OpenReview by Sat Oct 24.

### Stage 7 - Poster (after camera-ready)
On `poster-ready`, once the workshop's poster format is known.

---

## 5. Pre-registered analysis (commit before any new probe)

- **Runs:** seeds 1234, 5678 for RANDOM, CENTROID, ENVELOPE (certified legacy sampler); RANDOM-CB s1234. Shared ep25 ancestor, `epochs: 100`, stop after ep50, amp_target false, microbatch 64 x accum 8, prefix truncation.
- **Endpoint:** ep50 test AUC, fp32, cache-independent probe, mean over 5 head seeds (head-seed SD reported). Test predictions sealed until the freeze; gates use validation only.
- **Primary (new seeds only, n=2 per arm):** per seed, CENTROID - RANDOM and ENVELOPE - RANDOM with a paired test-case bootstrap CI per seed. No seed-level bootstrap at n=2.
- **Outcome labels per guided arm:**
  - *Consistent direction*: both new-seed deltas > 0 **and** mean delta > the range of RANDOM's three realizations (original re-extracted + 2 new). (Note: 2 of 2 positive happens 25% of the time under a symmetric null; not called "replicated".)
  - *Within run-to-run variation*: mean delta > 0 but either condition fails.
  - *Reversed*: mean delta <= 0.
- **Sensitivity:** pooled n=3 including the original run (different code/hardware), descriptive only.
- **Matched control:** CENTROID s1234 - RANDOM-CB s1234 (placement at equal budgets) and RANDOM-CB - RANDOM s1234, judged against RANDOM's spread.
- **Secondary:** validation deltas; pooling variants; MD severity regression within glaucoma cases. Every started run reported.

---

## 6. Paper change list and page budget (A6)

v9 7.58 pages -> target ~9.3 pages. Additions: seed table (+0.45), matched control (+0.35), pooling (+0.10), guide quality / CENTROID robustness (+0.10), reference rows (+0.05), strategy table with delivered context (+0.20), centroid formula/rationale (+0.10), replication paragraph (+0.15), all-strategy figure (0-0.15). Trims: ANATOMY-V1 detail, COVER mechanics, precision paragraph, long captions.
Wording rules: "repeated continuations from a shared epoch-25 checkpoint" (not "repeated pretraining runs"); "paired" = shared data order only; "same test split; analysis committed (hash) before evaluating the new runs"; CENTROID is intensity-derived, not segmentation; tissue is a MIRAGE-derived proxy.

---

## 7. Decisions needed from the user

1. Approve the GPU campaign (~242 GPU-h, Oct 9 -> ~Oct 19) and no gaming on this PC while runs are active.
2. **HF token** for `yfeng0206/ijepa-3d-oct-checkpoints` (blocking by Fri Oct 9: RANDOM anchor).
3. ENVELOPE route: legacy sampler flag (recommended) vs worktree of run-era code.
4. Free C: (>= 40 GB) or move the pagefile to D:; keep or raise the 315 W cap (fixed for the whole campaign).
5. Azure 4x T4 available? (whole seed indices only).
6. RETFound frozen probe (optional; gated; CC BY-NC 4.0): yes/no.
7. Camera-ready admin: author order, affiliations, emails, acknowledgments/funding, current OpenReview TL;DR, code link yes/no, public visibility, poster requirements.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| New seeds land far from originals | G1/G2 gates on validation; decision tree; code-drift list (A2 section 5.3); never rerun-until-match |
| Silent policy change (ENVELOPE sampler, guide dir, lateral fraction, amp_target, schedule) | G0 golden masks + fingerprint; G1 identity check; pinned source; provenance capture |
| Crash/OOM (0.6 GB RAM free; error 1455 history) | exact resume from rolling-last; free C:/pagefile; CPU work front-loaded; no gaming |
| RANDOM-CB slow | sampler in workers; smoke throughput; basic variant fallback; CB is second to drop |
| Effect does not replicate | pre-written outcome text; exploratory framing stays valid |
| Release tooling fails on camera-ready format | E15 early (Oct 9-12) |
| Table 2 change looks self-serving | show old and new numbers with the reason; paired gap with >= 100 draws per seed |

---

## 9. Progress log

| When | Event |
|---|---|
| 2026-10-08 00:20 | v9 merged to main (PR #4); `poster-ready` created |
| 2026-10-08 00:30-01:10 | Six investigation agents (A1-A6) reported |
| 2026-10-08 01:20 | Plan v1 drafted |
| 2026-10-08 02:10 | A1 final: originals used fp32 teacher targets; trained ENVELOPE pinned to pre-fix sampler behaviour; RANDOM-CB cost revised to ~39 h (basic) / ~51 h (strict); schedule shifted ~5 h |
| 2026-10-08 01:45 | Plan v2 after critique: smoke/anchors before R1, absolute G1 bands + identity check, validation-only G2 with decision tree, sealed test, new-seeds-primary stats, run order R1 C1 E1 R2 C2 CB E2 |
