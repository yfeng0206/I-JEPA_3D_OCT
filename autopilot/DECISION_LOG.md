# DECISION LOG

Autonomous decisions taken while the operator is away.
Policy: inspect evidence -> choose the most conservative scientifically valid option -> record -> continue.

## 2026-08-22T18:30:43-07:00 - Route all bulk output to D:, keep only control files on C:

C: has 19.8 GB free (8.5 percent), already below the 25 GB / 10 percent stop threshold in the operator directive. D: has 694 GB free. Writing features, figures and PDFs to C: risks filling the system drive.

## 2026-08-22T18:30:43-07:00 - Rebuild the paper's core evidence from the 19 saved test_predictions.npz rather than the previous mixed-protocol master table

All 19 runs share one identical probe protocol (mean_pool/linear/100 slices/seed 42/frozen/lr 4e-4/50 epochs) and an identical 3000-sample test label vector. This makes paired DeLong and bootstrap tests valid and removes the mixed-protocol confound that the prior adversarial review rated fatal. It requires no GPU and no retraining.

## 2026-08-22T18:42:11-07:00 - GPU priority reordered: fp32 re-probe of the envelope arm (ep50/75/100) comes BEFORE the COVER epoch-73 probe

eval_downstream.py line 541 reads use_amp = data_cfg.get('use_amp', True), so the default is fp16 AMP. The three envelope probe configs omit use_amp entirely and therefore ran in fp16, while all 16 comparator runs set use_amp: False and ran fp32. The envelope arm is the winning arm and the source of the headline +0.0200 AUC vs random. A precision mismatch between the winner and its comparators is a confound that would invalidate the paper's central claim, and an adversarial reviewer would find it. Re-probing a frozen encoder is explicitly permitted under the no-pretraining boundary. Until the fp32 re-probe lands, the envelope-vs-random contrast is marked PENDING-CONFOUNDED and must not be written as a result.

## 2026-08-22T19:01:49-07:00 - Extend GPU work to a complete fp32 family for random, oracle and envelope at ep50/75/100 (6 extra probes, queue 2)

The inventory shows the paper's primary family (random, oracle, envelope) is entirely fp16 while the H2 arms (anatomy-v1, anatomy-v2, cover-f021, ancestor) are entirely fp32. Table 1 therefore compares anatomy-v2 and cover against a null probed at a different numerical precision while claiming a single shared protocol. All required encoders are available locally (checkpoints_hf/random-posfix-100ep and oracle-anatomical-100ep), so the confound can be eliminated rather than merely disclosed. The interrupted run frozen_meanpool_oracle_ep100_fp32 already holds a complete fp32 feature cache, saving about 60 minutes.

## 2026-08-22T19:01:49-07:00 - Keep the fp16 primary-family results as the paper's primary numbers until the fp32 family completes

Within the fp16 family every contrast is precision-matched and internally valid, and my independent paired bootstrap reproduces the published headline (oracle minus random at ep100 = +0.0109, CI [+0.0057,+0.0160] against the paper's [+0.0058,+0.0162]). Discarding these would lose the only complete three-epoch trajectory. The fp32 family will be reported as a precision-robustness replication.

## 2026-08-22T19:15:55-07:00 - Reuse the repo's existing subgroup_analysis.py rather than my own p7_fairness.py as the authoritative fairness source

The repo script joins to D:/jepa_phase0/fairvision-glaucoma/metadata/data_summary_glaucoma.csv, which carries age and visual-field mean deviation (md) in addition to the six attributes stored inside each volume .npz. That enables age bins and severity strata, which my version could not produce. The two implementations agree exactly where they overlap (oracle ep100 race gap 0.0717 in both), so the richer one is preferred and mine is retained as an independent cross-check.

## 2026-08-22T19:24:54-07:00 - Every number in the manuscript is emitted as a LaTeX macro by p8_make_assets.py rather than typed by hand

The prior draft contained transcription errors between prose, tables and figures (an epoch-mismatched comparison and a correlation quoted from the wrong sample were both caught in earlier review). Generating all 136 quantities from the verified JSON artifacts makes that class of error structurally impossible and lets a correction propagate everywhere with one re-run.

## 2026-08-22T19:32:39-07:00 - Anonymity compliance verified locally by the coordinator while the style agent completes its byte-diff

The compiled PDF renders 'Anonymous Author(s) / Affiliation / Address / email' on page 1, page geometry is US Letter 612x792, and neurips_2026.sty sets \\newif\\if@anonymous\\@anonymoustrue as the default with only 'final'/'preprint'/'nonanonymous' style options clearing it. The manuscript passes no options, so double-blind rendering is confirmed. This closes the highest-severity part of desk-reject risk R1 without waiting for the agent; the outstanding item is only the byte-level diff against the official upstream file.

## 2026-08-22T19:39:05-07:00 - Keep the local neurips_2026.sty unchanged; desk-reject risk R1 closed

The sty-verify agent downloaded the official archive from media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip and diffed it against our copy: 443 lines each, zero differing lines, identical SHA-256 c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a. Geometry, font sizes and the anonymity branch all match, and the GenAI4Health CFP's required no-option usage renders anonymously. I independently corroborated the anonymous rendering and US Letter geometry from the compiled PDF.

## 2026-08-22T19:39:05-07:00 - Transcribe research-agent reports to disk manually

Research-type subagents in this session are blocked by execution policy from writing files; sty-verify produced a complete report that existed only in its reply. Their deliverables must be captured from the inline reply or they are lost at session end. The relwork agent has been warned to print its report and BibTeX inline as a fallback.

## 2026-08-22T20:01:45-07:00 - Accept the numerical audit's finding and exclude anatomy-v2 ep75/ep92 from the subgroup analysis, reducing it from 21 to 19 probes

p7b_gap_trend.py filtered only on subgroup_analysis.py's RETRACTED tag, which does not know about the anatomy-v2 EMA-target precision splice at epoch 56. Two probes the manuscript declares excluded were therefore silently re-admitted. This was not cosmetic: it changed the severity trend from non-significant (p=0.065) to significant (p=0.022) and shifted the race correlation. p7b now joins p1b_full_inventory.json for exclusion status.

## 2026-08-22T20:01:45-07:00 - Withdraw the claim that the racial AUC gap widens as models improve

The mock review and the numerical audit independently showed the claim fails on three counts: it does not survive Benjamini-Hochberg over the seven attributes tested (q=0.0595), it vanishes when pseudo-replication is removed by averaging within training branches (rho=+0.429, p=0.337), and it misdescribes the data, since every race subgroup improves and the black subgroup gains more than the white subgroup (+0.0147 vs +0.0112). The gap widens only because the already-highest asian subgroup gains most. The paper now states this explicitly and makes no fairness-trend claim.

## 2026-08-22T20:14:57-07:00 - Move the broader-impact statement to the appendix

The submission-requirements audit established that GenAI4Health requires no standalone ethics section, and that any such discussion in the main body counts against the 9-page limit while appendix material does not. Moving the full statement to an appendix preserves it in its entirety while recovering roughly twelve lines of main-body space, which was the margin needed to bring main content back inside 9 pages.

## 2026-08-22T20:14:57-07:00 - Do not add the NeurIPS Paper Checklist

The official NeurIPS 2026 archive bundles checklist.tex and the main-track template calls it mandatory, but the GenAI4Health workshop CFP states explicitly that the NeurIPS Paper Checklist is not required. The workshop-specific instruction governs a workshop submission, so no checklist is added.

## 2026-08-22T20:18:39-07:00 - Stopped the 2-hourly COVER-0.21 pretraining monitor schedule (#16)

That schedule dates from 2026-08-19 and instructs the agent to run the COVER f=0.21 chain to epoch 100, relaunching scripts/chain_cover_f021.py whenever it finds the trainer dead. The current operator directive states the opposite and takes precedence: 'Do not resume COVER-0.21 beyond its existing epoch-73 checkpoint. Do not complete COVER-0.21 epoch 100. This directive overrides any conflicting GPU or training instruction elsewhere in this prompt.' Leaving the schedule armed would repeatedly present an instruction to perform forbidden pretraining, and the chain script is idempotent so a single mistaken relaunch would immediately resume training from the rolling -last checkpoint. Stopping it is the conservative action. The COVER arm's evidence is preserved: its epoch-73 checkpoint is pinned and is being probed as a frozen encoder, which the directive explicitly permits.

## 2026-08-22T23:03:28-07:00 - Do not re-run the embedding-structure analysis; it already exists

results/masking/class_relations/class_relations.json (commit 2bbe199, 2026-08-09) already reports patch-token cosine structure by tissue class over 30 held-out GOALS images with 1028 within-class and 865 between-class cell pairs, including Cohen's d and a representation-discrimination AUC for MIRAGE H0, the MIRAGE encoder, two adapter settings, JEPA ep30, JEPA ep100 and an untrained control. That is the same family of metrics P6-01 would have produced, so re-running it would spend about four GPU hours to recompute existing numbers.

## 2026-08-22T23:03:28-07:00 - Do not re-run probe-ep25; it is already complete

The pending todo probe-ep25 asks for a frozen mean-pool AUC on jepa_patch-random_posfix-ep25.pth.tar. That probe exists as D:/jepa_phase0/runs/frozen_meanpool_fork_ep25 with test_auc 0.848680 under the identical protocol, and it is already the shared-ancestor row in the paper. The todo is stale.

## 2026-08-22T23:41:02-07:00 - Cancel the ep75 fp32 re-probes; keep ep50 and the completed ep100

Operator challenged whether the ep50/ep75 fp32 probes were needed. Checking p1c_stats.json, every cross-precision contrast in the manuscript sits at epoch 50: anatomy-v2 and cover-f021 exist only at fp32 and are compared there against an fp16 null, producing six confounded contrasts. At epochs 75 and 100 only random, intensity and envelope exist and all three are already fp16, so no confound exists to remove and an fp32 re-probe would only re-confirm a precision effect the completed ep100 trio already measured at 1e-6 to 2e-4. The ep50 trio is therefore kept because it converts the H2 comparison from confounded to precision-matched; the ep75 pair is cancelled, saving roughly 2.5 GPU hours for zero claim. The envelope ep75 probe already running is allowed to finish because most of its cost is sunk and it yields a third fp16/fp32 pair. Queue 1 is untouched so the COVER epoch-73 probe still runs, since that is new evidence rather than replication.

## 2026-08-23T12:48:40-07:00 - Phase C verified ready by config diff before it runs, rather than discovering a fault when Phase B ends

chain_blob_fp32.py has never executed, and a failure would surface at about 08:00 on 24 August when Phase B releases the GPU, wasting the window. Dry run confirms: the epoch-56 seed exists, campaign_supervisor accepts the argument list (val_baseline_json and baseline_epoch_s are optional), and the generated config differs from configs/patch_anatomy_v2.yaml in exactly 4 of 62 flattened keys, all intended: logging.folder, logging.write_tag, meta.read_checkpoint, meta.amp_target=False. The entire mask section is identical, so the rerun differs from the contaminated original only in EMA-target precision, which is precisely the controlled comparison intended.

