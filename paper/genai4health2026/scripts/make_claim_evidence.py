"""Emit research/claim_evidence.csv, the claim-level evidence map for the paper.

This is the CLAIM-side companion to the NUMBER-side machinery. Every printed
number already resolves through a generated macro in ``auto/auto_numbers.tex``
with ``research/numbers_master.csv`` and ``results/downstream/ARTIFACT_MAP.json``
as provenance, and a build gate makes reporting one arm's number under another
arm's name an error. What that machinery cannot catch is a *claim* that outruns
a correct number: a scope quietly dropped, an arm silently swapped, a stale
count carried forward. This file catalogues those.

The rows below are hand-curated by reading ``main_submission.tex`` against the
stored artifacts. They live in code rather than in the CSV so that quoting,
column order and path checking cannot rot, and so that a diff of the map is
reviewable. Adding a claim means adding a tuple here and re-running:

    D:\\jepa_phase0\\.venv\\Scripts\\python.exe paper\\genai4health2026\\scripts\\make_claim_evidence.py

The script fails loudly if any ``evidence_path`` does not exist on disk, so a
row can never cite an invented artifact. Paths are repo-relative with forward
slashes, or absolute when the artifact lives outside the repository. Line
numbers are deliberately NOT used as locators: the manuscript is edited
concurrently and line numbers rot within minutes. Locators are macro names,
JSON key paths, table/figure labels or section labels.

See research/CLAIM_MAP.md for the status vocabulary and the update procedure.
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PAPER))
OUT = os.path.join(PAPER, "research", "claim_evidence.csv")

COLUMNS = [
    "claim_id",
    "section",
    "claim_text",
    "claim_type",
    "evidence_path",
    "evidence_locator",
    "status",
    "note",
]

CLAIM_TYPES = {"MEASURED", "DERIVED", "SCOPED", "INTERPRETIVE", "LIMITATION"}
STATUSES = {"SUPPORTED", "SCOPED-OK", "PENDING", "UNSUPPORTED"}

# Short aliases for the artifacts cited most often.
AUTONUM = "paper/genai4health2026/auto/auto_numbers.tex"
INVENTORY = r"D:\jepa_phase0\autopilot_out\p1_stats\p1b_full_inventory.json"
STATS = r"D:\jepa_phase0\autopilot_out\p1_stats\p1c_stats.json"
TREND = r"D:\jepa_phase0\autopilot_out\p1_stats\p7b_gap_trend.json"
FAIR = r"D:\jepa_phase0\autopilot_out\p1_stats\p7_fairness.json"
FP32 = r"D:\jepa_phase0\autopilot_out\p1_stats\p3b_fp32.json"
META = r"D:\jepa_phase0\autopilot_out\p1_stats\test_metadata_summary.json"
SUBALL = r"D:\jepa_phase0\autopilot_out\subgroup\subgroup_auc.json"
INTER = r"D:\jepa_phase0\reports\subgroup\intersectional_auc.json"
GEOM = ("results/masking/table2_geometry/"
        "mask_geometry_600slices_bs1_coverf021_seed42.json")
GEOM64 = ("results/masking/table2_geometry/"
          "mask_geometry_600slices_bs64_coverf021_seed42.json")
GEOM64F15 = ("results/masking/table2_geometry/"
             "mask_geometry_600slices_bs64_coverf015_seed42.json")
MULT = "results/p17_subgroup_multiplicity.json"
SOP = "results/p16_subgroup_operating.json"
LABEFF = "results/p5_label_efficiency.json"
BGA2 = "autopilot/bgsig/a2_region_incremental.json"
SKILL = r"D:\jepa_phase0\reports\background_signal\skill_scores.json"
HARDAUDIT = "autopilot/reports/HARDCODED_AUDIT.md"
T2PROV = "autopilot/reports/TABLE2_PROVENANCE.md"
PROVADD = "autopilot/reports/PROVENANCE_ADD.md"

EPOCH50_NOTE = (
    "The artifact's own 'source' field is "
    "region_features\\{}_ep50_s100.pt and its 'note' says 'ep50 checkpoints'. "
    "The passage that carries this number opens 'At epoch 100' and never "
    "restates an epoch, so a reader will read it as epoch 100. Closes by "
    "labelling these four numbers epoch 50 or by re-running the region "
    "analysis on the epoch-100 features."
)

WORSTGROUP_NOTE = (
    "True for four of the seven attributes only. The paper's own "
    "tab:subtrends prints language 'other languages (19/23)' and age "
    "'<60 (22/23)', so the worst-served group is NOT the same in every probe "
    "for those two. The correctly scoped version is the sentence in "
    "sec:subgroup that names sex, race, ethnicity and disease severity. "
    "Closes by adding that scope wherever the unqualified form appears "
    "(abstract, contributions bullet, sec:subgroup opener and heading, "
    "app:ethics)."
)

MATCHES_NOTE = (
    "'Matches' rests on non-rejection at epochs 50 and 75, not on an "
    "equivalence test; no equivalence margin or TOST is stated anywhere. The "
    "widest bound against ArmBest is -0.0069 at epoch 50, so the honest "
    "reading is 'no difference larger than about 0.007 detected'. At epoch "
    "100 ArmBest exceeds envelope, so the claim is conservative there."
)

ROWS = [
    # ------------------------------------------------------------------ title
    ("T-01", "Title",
     "Segmentation-Free Anatomy Guidance Matches Segmenter-Driven Target Placement",
     "DERIVED", AUTONUM,
     "\\DOracleEnvelopeEpFiftyCI, \\DOracleEnvelopeEpSeventyFiveCI, \\DOracleEnvelopeEpHundredCI",
     "SCOPED-OK", MATCHES_NOTE),

    # --------------------------------------------------------------- abstract
    ("A-01", "Abstract",
     "Six masking policies are continued from a single shared I-JEPA checkpoint, differing in how predictor targets are placed",
     "MEASURED", INVENTORY,
     "records[*].arm = random, envelope, oracle, anatomy-v1, anatomy-v2, cover-f021; ancestor record epoch=25",
     "SUPPORTED",
     "Six distinct policies present in the inventory; all descend from the epoch-25 ancestor record."),

    ("A-02", "Abstract",
     "evaluated with a frozen linear-probe protocol for glaucoma classification (FairVision, N=3000)",
     "MEASURED", STATS,
     "n_test=3000, n_pos=1466; seed 42 in results/downstream/meanpool_sweep_random/ep100_results.json config.training.seed",
     "SUPPORTED", ""),

    ("A-03", "Abstract",
     "Each policy was continued once, so what follows describes these runs rather than an expected ranking over retrainings",
     "SCOPED", INVENTORY,
     "one pretraining trajectory per arm; no seed replicates among records[*]",
     "SCOPED-OK",
     "The seed replication described in app:replication is running and is explicitly unreported."),

    ("A-04", "Abstract",
     "aiming the same rectangles at retinal tissue rather than at the whole frame improves AUC by +0.0120 over unguided masking at the matched epoch",
     "MEASURED", AUTONUM,
     "\\DEnvelopeRandomEpFifty = +0.0120; \\DEnvelopeRandomEpFiftyCI = [+0.0068,+0.0173]",
     "SUPPORTED", ""),

    ("A-05", "Abstract",
     "a band located by a first-order intensity statistic, using no segmentation model at all, improves it by +0.0109 at epoch 100",
     "MEASURED", AUTONUM,
     "\\DOracleRandomEpHundred; arm definition in configs/patch_oracle_anatomical.yaml curriculum mode anatomical_prior",
     "SUPPORTED",
     "app:repro independently states ArmBest and random consult no MIRAGE guide of either kind."),

    ("A-06", "Abstract",
     "Paired intervals for both arms exclude zero at all three epochs",
     "MEASURED", AUTONUM,
     "\\DEnvelopeRandomEp{Fifty,SeventyFive,Hundred}CI and \\DOracleRandomEp{Fifty,SeventyFive,Hundred}CI, all six strictly positive",
     "SUPPORTED", ""),

    ("A-07", "Abstract",
     "the band matches the segmenter-guided rectangles at epochs 50 and 75 and exceeds them at 100 (+0.0047)",
     "DERIVED", AUTONUM,
     "\\DOracleEnvelopeEpFiftyCI [-0.0069,+0.0029]; ...SeventyFiveCI [-0.0013,+0.0079]; ...HundredCI [+0.0004,+0.0091]",
     "SCOPED-OK", MATCHES_NOTE),

    ("A-08", "Abstract",
     "A policy that places 97% of masked patches on tissue does not separate from unguided masking at epoch 50 and falls below it by epoch 75",
     "MEASURED", GEOM,
     "anatomy.hidden_pct_on_anat = 97.094; deltas in \\DAnatomyTwoRandomEpFiftyCI and \\DAnatomyTwoRandomEpSeventyFiveCI",
     "SUPPORTED", ""),

    ("A-09", "Abstract",
     "carried to epoch 100 the coverage arm peaks at epoch 73 and then declines",
     "MEASURED", AUTONUM,
     "\\CoverPeakEpoch = 73; \\DCoverPeakToHundred = -0.0071; \\DCoverPeakToHundredP < 0.0001",
     "SUPPORTED", ""),

    ("A-10", "Abstract",
     "an audit found its targets are shortened after placement, so it does not test aggressive coverage",
     "MEASURED", r"D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json",
     '["0.21"].pct_anat_hid = 73.09 against ["envelope"].pct_anat_hid = 77.58',
     "SUPPORTED",
     "Mechanism narrated in autopilot/COVER_AUDIT.md; the paper retains only the sweep-backed figure."),

    ("A-11", "Abstract",
     "Direct measurement of mask geometry across 600 slices shows the winner hides the least anatomy of any guided policy",
     "MEASURED", GEOM,
     "hidden_share_of_all_anat: oracle 62.13 < cover 73.55 < envelope 77.58 < anatomy 79.89",
     "SUPPORTED",
     "Correctly restricted to guided policies: unguided random is lower still at 53.96."),

    ("A-12", "Abstract",
     "how much anatomy a policy hides is confounded here with mask ratio and retained context, so this design does not identify it",
     "INTERPRETIVE", GEOM,
     "hidden_frac_of_grid and ctx_frac_of_grid differ across arms alongside hidden_share_of_all_anat",
     "SUPPORTED", ""),

    ("A-13", "Abstract",
     "Nor are the null's background targets wasted -- they are a genuine pretraining signal",
     "MEASURED", SKILL,
     "random ep100 background skill 0.6798 above anatomy skill 0.6334",
     "SUPPORTED", ""),

    ("A-14", "Abstract",
     "even though background contributes almost nothing to the classifier",
     "MEASURED", BGA2,
     "random.bg_residual_on_anatomy.test_auc = 0.55151",
     "PENDING", EPOCH50_NOTE),

    ("A-15", "Abstract",
     "A subgroup audit over 23 probes finds the same worst-served group every time",
     "MEASURED", TREND,
     "worst_group_consistency: gender/race/ethnicity/severity unanimous 23/23; language 19/23; age 22/23",
     "UNSUPPORTED", WORSTGROUP_NOTE),

    ("A-16", "Abstract",
     "all three disease-stage point estimates rise (largest for mild, +0.0137)",
     "MEASURED", MULT,
     "auc_family.contrasts['severity:mild'|'severity:moderate'|'severity:severe'].estimate = +0.01372/+0.01016/+0.00626",
     "SUPPORTED", ""),

    ("A-17", "Abstract",
     "only the multiplicity-adjusted white race-group interval excludes zero",
     "MEASURED", MULT,
     "auc_family.contrasts['race:White'].simultaneous_excludes_zero = true; Black and Asian false",
     "SUPPORTED",
     "This is the corrected form of the earlier claim that every disease stage improved with intervals excluding zero."),

    ("A-18", "Abstract",
     "The gains are not separated from one another",
     "MEASURED", MULT,
     "contrasts['race:Black-minus-Asian'] and contrasts['sex:Female-minus-Male'] simultaneous CIs contain zero",
     "SUPPORTED", ""),

    ("A-19", "Abstract",
     "the severe-to-mild gap stays within 0.0114 of constant across every policy",
     "DERIVED", TREND,
     "trends.severity.gap_min = 0.12863, gap_max = 0.13998; 0.1400 - 0.1286 = 0.0114 = \\SeverityGapSpread",
     "SUPPORTED", ""),

    # ----------------------------------------------- introduction and contribs
    ("I-01", "Introduction",
     "All arms continue from one shared pretrained checkpoint under one optimiser, schedule, effective batch size and frozen-probe protocol (512 for all arms)",
     "MEASURED", "src/train_patch.py",
     "effective batch printed as data_cfg['batch_size'] * world_size * accum_steps; oracle 64x4x2, envelope/anatomy-v1/anatomy-v2/cover 64x1x8",
     "PENDING",
     "512 is verified from configs for five of six arms. The random arm's producing config is not in this checkout "
     "(configs/patch_vitb16_ep100.yaml is an older remote config at batch 32 and peak LR 0.0005, not the shared 0.00025), "
     "and its checkpoint stores batch_size and world_size but not accum_steps. "
     "docs/experiments/pretraining/README.md asserts effective batch 512 for all completed runs. "
     "Note also that an earlier session flagged a '4x effective-batch confound' computed as batch x world_size only; "
     "including accum_steps removes it. Closes by recovering the random run's config or a training log line "
     "'Effective batch size: 512'."),

    ("I-02", "Introduction",
     "the mask geometry each policy delivers is measured, not assumed equal",
     "MEASURED", GEOM,
     "all five printed columns emitted per arm by scripts/mask_composition_probe.py",
     "SUPPORTED",
     "Regenerated this cycle; an earlier audit had 24 of 25 geometry cells unbacked (see " + T2PROV + ")."),

    ("I-03", "Introduction",
     "The two policies whose targets track the segmented retina most faithfully have intervals spanning zero against unguided masking at epoch 50, and the one we carried further falls significantly below it by epoch 75",
     "MEASURED", AUTONUM,
     "\\DAnatomyTwoRandomEpFiftyCI [-0.0055,+0.0082]; \\DCoverRandomEpFiftyCI [-0.0050,+0.0053]; \\DAnatomyTwoRandomEpSeventyFiveCI [-0.0181,-0.0042]",
     "SUPPORTED", ""),

    ("I-04", "Introduction",
     "One earlier shaped policy did exceed its rectangle counterpart at epoch 30, so the anatomy result is mixed rather than uniformly negative",
     "MEASURED", AUTONUM,
     "\\DAnatomyOneEnvelopeEpThirty = +0.0044; ...CI [+0.0009,+0.0078]; ...P = 0.0127",
     "SCOPED-OK",
     "Different implementation at a non-matched epoch; the paper says explicitly it does not read this as settling H2."),

    ("I-05", "Introduction (contributions)",
     "A controlled six-arm study, with a shared ancestor checkpoint and one evaluation protocol; every probe fitted is listed",
     "MEASURED", INVENTORY,
     "status counts: primary 31, excluded 2, retracted 4 = the 37 rows of tab:allprobes",
     "SUPPORTED",
     "Six further records marked 'supplementary' are not part of the 37 and are not tabulated."),

    ("I-06", "Introduction (contributions)",
     "The most anatomically precise policy is not the best performer, from direct measurement of mask geometry on 600 slices",
     "MEASURED", GEOM,
     "anatomy.hidden_pct_on_anat = 97.09 with epoch-50 AUC \\AUCAnatomyTwoEpFifty against \\AUCOracleEpHundred",
     "SUPPORTED", ""),

    ("I-07", "Introduction (contributions)",
     "The subgroup audit is a caution: every policy leaves the same groups worst-served",
     "INTERPRETIVE", TREND,
     "worst_group_consistency across the seven attributes",
     "UNSUPPORTED",
     "The 'caution' framing now agrees with app:ethics, which is the fix for the earlier contribution/caution mismatch. "
     "The residual problem is the same over-generalisation as A-15: language 19/23 and age 22/23 are not unanimous."),

    ("I-08", "Introduction (contributions)",
     "Public-weight reproduction matches the headline result to 1e-5 across hardware and precision",
     "MEASURED", HARDAUDIT,
     "row '9.8x10^-6; 0.8854754; 0.8854852; 22; 2,248,844'; HF repo in results/downstream/ARTIFACT_MAP.json :: hf_repo",
     "SUPPORTED",
     "Head recovered by anonymous download from the public HF repo; the reference AUC was produced remotely at fp16 and "
     "the reproduction locally at fp32, which is what 'across hardware and precision' means here."),

    # ------------------------------------------------------------------ setup
    ("S-01", "Sec. Experimental setup",
     "held-out test split of N=3000 volumes (1466 positive / 1534 negative), seed 42, fixed loader order",
     "MEASURED", STATS,
     "n_test = 3000, n_pos = 1466; seed at results/downstream/meanpool_sweep_random/ep100_results.json :: config.training.seed",
     "SUPPORTED", ""),

    ("S-02", "Sec. Experimental setup",
     "No test volume is used to fit or select the probe head, which is selected on a separate validation split",
     "MEASURED", "src/eval_downstream.py",
     "head selected on validation AUC; protocol recorded in configs/frozen_meanpool_mirage_ep50.yaml",
     "SUPPORTED", ""),

    ("S-03", "Sec. Experimental setup",
     "the study's choices of policy, checkpoint, analysis and stopping horizon were made after repeated inspection of that same test split",
     "LIMITATION", "autopilot/DECISION_LOG.md",
     "app:replication states the number of inspections 'was not logged and cannot now be reconstructed'",
     "SCOPED-OK",
     "Self-declared and not independently verifiable; the paper says so, which is the right handling."),

    ("S-04", "Sec. Experimental setup",
     "Labels are byte-identical across arms, so any two arms are paired-comparable on the same cases",
     "MEASURED", SUBALL,
     "label_agreement = 1.0 for all 37 probe entries",
     "SUPPORTED", ""),

    ("S-05", "Sec. Experimental setup",
     "Every arm resumes from the same epoch-25 checkpoint and continues with identical optimiser, learning-rate and weight-decay schedules",
     "MEASURED", "src/train_patch.py",
     "load_checkpoint restores encoder, predictor, target encoder, optimizer and scaler; ancestor SHA-256 e5ad5b0c... in app:repro",
     "SUPPORTED",
     "docs/experiments/pretraining/README.md notes random is the only cold-start arm. Because the epoch-25 ancestor IS "
     "random's own checkpoint and resume restores optimiser and scaler state, the fork point is identical; the note is "
     "about lineage, not about a weight or optimiser-state mismatch."),

    ("S-06", "Sec. Experimental setup",
     "we stopped anatomy-v2 having seen its epoch-75 deficit, which is a selective stopping horizon and a real weakness",
     "LIMITATION", INVENTORY,
     "anatomy-v2 primary records at epochs 35, 40, 50, 75; no epoch-100 record",
     "SUPPORTED", ""),

    ("S-07", "Sec. Experimental setup",
     "Epoch 50 is the only epoch at which five policies are simultaneously available",
     "DERIVED", INVENTORY,
     "primary records at epoch 50: random, envelope, oracle, anatomy-v2, cover-f021",
     "SUPPORTED", ""),

    ("S-08", "Sec. Experimental setup",
     "all headline contrasts in Table 1 are drawn from arms sharing both epoch and precision",
     "MEASURED", STATS,
     "family_key.A_primary_matched; multiplicity.confirmatory_family_size = 9; auto/table_main.tex is fp16 only",
     "SUPPORTED", ""),

    ("S-09", "Sec. Experimental setup",
     "Each test volume is a distinct subject, so case-level resampling is already subject-level",
     "MEASURED", META,
     "n_unique_subject_ids = 3000",
     "SUPPORTED", ""),

    ("S-10", "Sec. Experimental setup",
     "the anatomy-v1, anatomy-v2, cover and ancestor probes were fitted with autocast explicitly disabled (fp32)",
     "MEASURED", INVENTORY,
     "records[*].precision; precision_note",
     "SCOPED-OK",
     "precision_note records that precision is 'inferred from stored probs dtype', not read from each config. The paper "
     "says the same in the tab:allprobes caption, so the inference is disclosed."),

    # ---------------------------------------------------------------- results
    ("R-01", "Sec. Results (region)",
     "all six contrasts survive Benjamini-Hochberg correction over that family of nine (q <= 0.0299)",
     "DERIVED", AUTONUM,
     "max of the six Q macros is \\DEnvelopeRandomEpHundredQ = 0.0299; family size at " + STATS + " :: multiplicity.confirmatory_family_size = 9",
     "SUPPORTED", ""),

    ("R-02", "Sec. Results (region)",
     "the rectangles are large, so fewer than half of envelope's masked cells land on tissue",
     "MEASURED", GEOM,
     "envelope.hidden_pct_on_anat = 43.30",
     "SUPPORTED", ""),

    ("R-03", "Sec. Results (region)",
     "anatomy-v2 sits -0.0107 below envelope at epoch 50 (CI [-0.0167,-0.0046])",
     "MEASURED", AUTONUM,
     "\\DAnatomyTwoEnvelopeEpFifty, ...CI, ...P = 0.0006, ...Q = 0.0013",
     "SUPPORTED", ""),

    ("R-04", "Sec. Results (region)",
     "Nor is anatomy-v2 or cover worse than the null at the matched epoch: both sit within a paired interval that comfortably contains zero",
     "MEASURED", AUTONUM,
     "\\DAnatomyTwoRandomEpFiftyCI; \\DCoverRandomEpFiftyCI; null precision \\DAnatomyTwoRandomEpFiftyNullPrec = fp32",
     "SUPPORTED",
     "Both deltas are taken against an fp32 null re-probed at the same epoch, as the tab:main caption states."),

    ("R-05", "Sec. Results (region)",
     "cover's gap against the null widens monotonically across the three matched epochs",
     "DERIVED", AUTONUM,
     "\\DCoverRandomEpFifty +0.0002, \\DCoverRandomEpSeventyFive -0.0084, \\DCoverRandomEpHundred -0.0168",
     "SUPPORTED", ""),

    ("R-06", "Sec. Results (region)",
     "Its margin over the null is the largest at epoch 100 and does not decay with training, whereas envelope's does",
     "DERIVED", AUTONUM,
     "\\DOracleRandomEp{Fifty,SeventyFive,Hundred} = +0.0099/+0.0113/+0.0109; \\DEnvelopeRandomEp{...} = +0.0120/+0.0080/+0.0062",
     "SCOPED-OK",
     "True read as 'largest of any arm's margin at epoch 100'. It is not a within-arm maximum: ArmBest's own margin "
     "peaks at epoch 75 (+0.0113) and is +0.0109 at 100. The non-decay half of the sentence is right either way."),

    ("R-07", "Sec. Results (aim)",
     "Of the four guided arms the one that hides the most, at 97.1% purity, does not separate from the null",
     "MEASURED", GEOM,
     "anatomy.hidden_share_of_all_anat = 79.89 (max of the guided arms); anatomy.hidden_pct_on_anat = 97.09; delta at \\DAnatomyTwoRandomEpFiftyCI",
     "SUPPORTED", ""),

    ("R-08", "Sec. Results (aim)",
     "it retains more context than any other rectangle arm",
     "MEASURED", GEOM,
     "ctx_frac_of_grid: oracle 0.455 > cover 0.432 > random 0.419 > envelope 0.406",
     "SUPPORTED", ""),

    ("R-09", "Sec. Results (aim)",
     "Rank correlations over these arms are positive for both anatomy hidden and purity, but at n=4-5 arms they resolve nothing in either direction",
     "SCOPED", T2PROV,
     "Spearman recomputes to +0.80 at the production cover floor 0.21 and +0.40 at floor 0.15",
     "SUPPORTED",
     "Internal inconsistency worth a look: app:geomprov still speaks of 'the +0.80 anatomy-hidden/AUC Spearman "
     "coefficient of Section 5.2', but Section 5.2 no longer prints any coefficient. The value itself is backed."),

    ("R-10", "Sec. Results (aim)",
     "anatomy-v2 collapses to 64 predictor loss slots against about 159 for every rectangle arm",
     "MEASURED", GEOM,
     "n_slots_mean: anatomy 64.00, rectangles 158.99-159.91; npred=4 and pred_target_k=16 in configs/patch_anatomy_v2.yaml",
     "SUPPORTED", ""),

    ("R-11", "Sec. Results (aim)",
     "H3 is therefore not identified by this design, and we do not claim that irregular target shape is harmful",
     "INTERPRETIVE", GEOM,
     "mask ratio, context kept and loss slots all differ for anatomy-v2; guide provenance differs per app:repro",
     "SUPPORTED", ""),

    ("R-12", "Sec. Results (background)",
     "At epoch 100 the null's predictor beats a per-position, no-context reference by 0.680 on background targets, above its 0.633 on anatomy",
     "MEASURED", SKILL,
     "random ep100 background 0.6798, anatomy 0.6334 (skill_vs_pos = 1 - err_predictor/err_position_only_reference)",
     "SUPPORTED", ""),

    ("R-13", "Sec. Results (background)",
     "90.8% of across-position input variance at background comes from pos_embed, against 40.8% at tissue",
     "MEASURED", "autopilot/bgsig/a3b_threshold_sweep.json",
     'ckpts.random_ep100.background["<=0.10"].position_share = 0.90821; ckpts.random_ep100.anatomy[">=0.20"].position_share = 0.40799',
     "SUPPORTED",
     "The two figures come from different thresholds of the sweep (background <=0.10, tissue >=0.20). The primary run "
     "autopilot/bgsig/a3_layer0_position_content.json uses <=0.06 / >=0.30 and gives 94.8% / 36.8%. The paper's "
     "'stable across six thresholds' is the right hedge; the specific pair is not the primary one."),

    ("R-14", "Sec. Results (background) and App. background",
     "Pretraining spends real capacity on them, and background self-similarity falls from 0.784 untrained to 0.346",
     "MEASURED", "results/masking/class_relations/class_relations.json",
     "'JEPA untrained (control)'.bg_bg = 0.7842 and 'JEPA ep100 (envelope)'.bg_bg = 0.3460",
     "UNSUPPORTED",
     "Both numbers are real, but 0.346 is measured on the ENVELOPE encoder at epoch 100. Both passages place it inside "
     "a paragraph explicitly about the random/null arm at epoch 100 ('the null's predictor...', 'the random arm's "
     "predictor...'), so the arm is silently swapped. The artifact contains no random-arm entry at all: its only JEPA "
     "keys are untrained control, ep30 anatomy and ep100 envelope. Closes by attributing the number to envelope, by "
     "measuring bg_bg on the random ep100 encoder, or by restating the sentence as a claim about pretraining in general."),

    ("R-15", "Sec. Results (background) and App. background",
     "Pooled background features are 95.2% linearly reconstructible from pooled anatomy features",
     "MEASURED", BGA2,
     "random.bg_residual_on_anatomy.ridge_test_R2_bg_from_anatomy = 0.95222",
     "PENDING", EPOCH50_NOTE),

    ("R-16", "Sec. Results (background) and App. background",
     "residualised, background predicts glaucoma at test AUC 0.5515, barely above chance, though the interval excludes it",
     "MEASURED", BGA2,
     "random.bg_residual_on_anatomy.test_auc = 0.55151, test_auc_ci95 = [0.51645,0.58930]",
     "PENDING", EPOCH50_NOTE),

    ("R-17", "App. background",
     "appending background to anatomy puts the change at -0.0076 (CI [-0.0139,-0.0012]) for the random arm and under +/-0.002 for every other",
     "MEASURED", BGA2,
     "delta_cat_minus_anatomy_ci95_and_mean: random [-0.01385,-0.00119,-0.00761]; oracle +0.00029; envelope +0.00161; blob -0.00034",
     "PENDING", EPOCH50_NOTE),

    ("R-18", "App. background",
     "A background-only probe scores 0.867; self-attention mixes tissue information into background tokens",
     "MEASURED", BGA2,
     "random.background.test_auc = 0.86665",
     "PENDING", EPOCH50_NOTE),

    ("R-19", "Sec. Results (background)",
     "The mechanism is described rather than established: the two controls that would settle it were not run",
     "LIMITATION", BGA2,
     "no eroded-background or background-content-shuffle entry exists in autopilot/bgsig/",
     "SUPPORTED",
     "Absence of the artifacts is exactly what the claim asserts."),

    ("R-20", "Sec. Results (label efficiency)",
     "at 5% of the labelled set (n=300) ArmBest leads the null by +0.0496, against +0.0108 at full supervision, more than four times the advantage",
     "DERIVED", LABEFF,
     "arms.intensity['0.05'].auc_mean 0.833529 - arms.random['0.05'].auc_mean 0.783911 = 0.049618; full 0.885644 - 0.874805 = 0.010840; ratio 4.58",
     "SUPPORTED", ""),

    ("R-21", "Sec. Results (label efficiency)",
     "across all four arms in this sweep, the full-supervision fits reproduce their corresponding epoch-100 entries in Table 1 to within 0.0009",
     "DERIVED", LABEFF,
     "arms.*['1.00'].auc_mean against the stored epoch-100 probe AUCs; the binding arm is cover, 0.858562 against \\AUCCoverEpHundred 0.8577",
     "SCOPED-OK",
     "A four-arm bound. It is a different quantity from the two-arm 0.0003 bound in app:labeleff, and the paper says so. "
     "Both are correct as written."),

    ("R-22", "App. label efficiency",
     "restricting this check to the null and ArmBest, their full-supervision results agree to within 0.0003",
     "DERIVED", LABEFF,
     "random 0.8748055 against stored 0.8745809 (2.25e-4); ArmBest 0.8856444 against stored 0.8854852 (1.59e-4)",
     "SCOPED-OK",
     "Recomputed exactly this cycle. The paper explicitly distinguishes this two-arm bound from the four-arm 0.0009 bound."),

    ("R-23", "Fig. label efficiency caption",
     "At full supervision the four arms lie within 0.027 of one another; at 5% the spread is 0.085",
     "DERIVED", LABEFF,
     "full: 0.885644 - 0.858562 = 0.02708; 5%: 0.833529 - 0.748361 = 0.08517",
     "SUPPORTED", ""),

    ("R-24", "Sec. Subgroups",
     "Across seven stratifications and 23 probes, no masking policy reorders the subgroups",
     "MEASURED", TREND,
     "n_probes = 23; worst_group_consistency.language.unanimous = false; worst_group_consistency.age.unanimous = false",
     "UNSUPPORTED", WORSTGROUP_NOTE),

    ("R-25", "Sec. Subgroups",
     "The worst-performing group is the same in every one of the 23 probes for sex (female), race (black), ethnicity (hispanic) and disease severity (mild)",
     "MEASURED", TREND,
     "worst_group_consistency.{gender,race,ethnicity,severity}.unanimous = true, counts 23 each",
     "SUPPORTED",
     "This is the correctly scoped sentence; the unqualified statements at A-15, I-07, R-24 should be brought in line with it."),

    ("R-26", "Sec. Subgroups",
     "Predictions are joined to the released FairVision metadata in deterministic loader order, validated by exact label reconstruction (3000/3000 cases, every probe)",
     "MEASURED", SUBALL,
     "label_agreement = 1.0 for all 37 entries; independent index_alignment_proof.aligned = true in " + META,
     "SUPPORTED", ""),

    ("R-27", "Sec. Subgroups",
     "at epoch 100 the strongest policy raises the point estimate for every race stratum over the null, and the max-min gap widens only because the already-best asian subgroup gains most",
     "DERIVED", AUTONUM,
     "\\RaceRandom{White,Black,Asian} 0.8741/0.8325/0.9033 -> \\RaceOracle{...} 0.8853/0.8472/0.9189; gap 0.0708 -> 0.0717; asian gain 0.0156 largest",
     "SUPPORTED", ""),

    ("R-28", "Sec. Subgroups",
     "Family-wise simultaneous paired intervals exclude zero only for the white subgroup (CI [+0.00286,+0.01972])",
     "MEASURED", MULT,
     "auc_family.contrasts['race:White'].simultaneous_ci95_lo/hi",
     "SUPPORTED", ""),

    ("R-29", "Sec. Subgroups",
     "the race gap correlates with aggregate AUC at rho=+0.473 (q=0.0668), which does not pass Benjamini-Hochberg correction at the stated conventional threshold",
     "MEASURED", TREND,
     "trends.race.spearman_rho = 0.47332, q_bh_across_attributes = 0.066756",
     "SUPPORTED",
     "This is the corrected form of an earlier draft that described q=0.0668 as passing a stated 0.05 threshold. The "
     "current wording is right in both the body and the tab:subtrends caption."),

    ("R-30", "Sec. Subgroups",
     "Collapsing to one point per branch, the association disappears (rho=+0.321)",
     "MEASURED", TREND,
     "trends.race.branch_spearman_rho = 0.32143, branch_spearman_p = 0.48207, n_branches = 7",
     "SUPPORTED", ""),

    ("R-31", "Sec. Subgroups",
     "Only the sex gap survives checkpoint-level correction, and it narrows (branch rho=-0.821, raw p=0.0234; checkpoint q=0.0038)",
     "MEASURED", TREND,
     "trends.gender.branch_spearman_rho/-_p and q_bh_across_attributes = 0.0038496; direction = 'narrows'",
     "SUPPORTED", ""),

    ("R-32", "Sec. Subgroups",
     "after adjustment over the declared 10-contrast family only the mild interval excludes zero; the moderate and severe intervals no longer do",
     "MEASURED", MULT,
     "auc_family.family_size = 10; severity:moderate and severity:severe conclusion_changed = true, simultaneous_excludes_zero = false",
     "SUPPORTED",
     "Directly fixes the earlier 'every disease stage improves with intervals excluding zero' error: the unadjusted "
     "intervals did exclude zero for all three."),

    ("R-33", "Sec. Subgroups",
     "between the worst- and best-served race strata the paired gain difference is -0.00091 (CI [-0.03285,+0.03104])",
     "MEASURED", MULT,
     "auc_family.contrasts['race:Black-minus-Asian'].estimate and simultaneous_ci95",
     "SUPPORTED", ""),

    ("R-34", "Sec. Subgroups",
     "the race-by-sex cells, whose gap exceeds the race margin in every arm",
     "MEASURED", INTER,
     "reproduced by paper/genai4health2026/scripts/intersectional_claims.py: 'intersectional exceeds marginal race 18/18'",
     "SUPPORTED",
     "Closes the earlier 'we cannot speak to intersectional groups' gap, which was stated three times while this "
     "analysis already existed on disk."),

    ("R-35", "Sec. Controls (precision)",
     "A full re-fit of the probe pipeline at fp32 shifts every arm by less than 2e-4",
     "MEASURED", FP32,
     "rows[*].delta_fp32_minus_fp16; largest |delta| = 1.921e-4 (oracle epoch 100)",
     "SCOPED-OK",
     "8 of 9 planned re-probes exist; the artifact lists pending = ['frozen_meanpool_oracle_ep75_fp32']. 'Every arm' is "
     "true per arm; it is not every arm-epoch. The tab:fp32 caption's 'eight DeLong p-values' discloses the count."),

    ("R-36", "Sec. Controls (precision)",
     "Re-encoding the test split from those weights on different hardware, at fp32 rather than fp16, reproduces the reported AUC to 9.8e-6: 22 discordant pairs out of 2,248,844",
     "MEASURED", HARDAUDIT,
     "recomputed 0.8854753820 against reference 0.8854851648; 1466*1534 = 2,248,844",
     "SUPPORTED", ""),

    ("R-37", "Sec. Controls (precision)",
     "The generating script, the epoch-100 ArmBest encoder, its head and its stored per-case predictions are released",
     "MEASURED", "results/downstream/ARTIFACT_MAP.json",
     "hf_repo = yfeng0206/ijepa-3d-oct-checkpoints; runs[*].artifacts.{encoder_hf, head_hf, predictions_local}",
     "SUPPORTED",
     "The HF repo was verified anonymously readable when the heads were downloaded; links are withheld for anonymity, "
     "so a reviewer cannot check this at submission time."),

    # ---------------------------------------------------------- discussion
    ("L-01", "Sec. Discussion and limitations",
     "We cannot claim anatomy-shaped masking is harmful: at the matched epoch it sits +0.0013 from the null with an interval that comfortably contains zero",
     "MEASURED", AUTONUM,
     "\\DAnatomyTwoRandomEpFifty and \\DAnatomyTwoRandomEpFiftyCI",
     "SUPPORTED", ""),

    ("L-02", "Sec. Discussion and limitations",
     "Distinguishing them requires an arm that randomises the band's vertical position while holding area and context fixed, which we did not run",
     "LIMITATION", INVENTORY,
     "no such arm among records[*]",
     "SUPPORTED", ""),

    ("L-03", "Sec. Discussion and limitations",
     "an earlier multi-seed probe check is not reproducible from retained artifacts, so we state no bound on probe noise",
     "LIMITATION", "paper/genai4health2026/research/verify_sections_4_6.md",
     "prior audit found only seed-42 result files; the two probe-seed SDs are listed UNBACKED in " + HARDAUDIT,
     "SUPPORTED",
     "Verified this cycle that the withdrawn SDs 0.0003 and 0.0018 no longer appear anywhere in main_submission.tex."),

    ("L-04", "Sec. Discussion / App. replication",
     "A replication is running, with no result of it reported here: six continuations resuming from the same epoch-25 ancestor, SHA-256 verified before each leg",
     "SCOPED", "autopilot/reports/G1_REPLICATION.md",
     "ancestor SHA-256 e5ad5b0c...; 1,507,519,602 bytes; three smoke configs PASS; 'Effective batch size: 512'",
     "SUPPORTED",
     "No epoch-50 replication AUC appears anywhere in the repository, consistent with 'no result reported'."),

    ("L-05", "Sec. Discussion and limitations",
     "Only f=0.21 was pretrained for cover, so we report no dose-response over the visible-tissue floor",
     "MEASURED", "configs/patch_cover_f021_ep25.yaml",
     "cover_leave_frac = cover_min_visible_frac = 0.21; only cover-f021 records exist in the inventory",
     "SUPPORTED", ""),

    # ---------------------------------------------------------- conclusion
    ("C-01", "Conclusion",
     "aiming ordinary rectangles at retinal tissue beat unguided masking, and the best policy consulted no segmentation model, matching the segmenter-guided arm at epochs 50 and 75 and exceeding it at 100",
     "DERIVED", AUTONUM,
     "\\DOracleEnvelopeEpFiftyCI, ...SeventyFiveCI, ...HundredCI",
     "SCOPED-OK", MATCHES_NOTE),

    ("C-02", "Conclusion",
     "shaping targets to trace the segmented retina did not separate from unguided masking at epoch 50 and fell below it by epoch 75",
     "MEASURED", AUTONUM,
     "\\DAnatomyTwoRandomEpFiftyCI; \\DAnatomyTwoRandomEpSeventyFiveCI",
     "SUPPORTED", ""),

    ("C-03", "Conclusion",
     "every positive disease-stage point estimate rises, but only the adjusted mild interval excludes zero; race gains consistent in sign, and the difficulty ordering untouched",
     "MEASURED", MULT,
     "auc_family severity and race contrasts; severity gap range at " + TREND + " :: trends.severity",
     "SUPPORTED",
     "Correctly scoped, unlike the abstract's worst-group sentence."),

    ("C-04", "Conclusion",
     "What makes the best policy work is not identified here: this design cannot separate consistent target placement from mask ratio and task difficulty",
     "INTERPRETIVE", GEOM,
     "oracle hidden_frac_of_grid 0.403 and ctx_frac_of_grid 0.455 against random 0.445 and 0.419",
     "SUPPORTED", ""),

    # ---------------------------------------------- appendices the body uses
    ("P-01", "App. all frozen probes",
     "37 rows: 31 valid probes, plus two probes excluded and four retracted",
     "MEASURED", INVENTORY,
     "status counter over records[*]: primary 31, excluded 2, retracted 4",
     "SUPPORTED",
     "The inventory holds 43 records; six are marked 'supplementary' and are not part of the 37."),

    ("P-02", "App. all frozen probes",
     "Eight are fp32 re-probes of an encoder already probed at fp16, leaving 23 distinct encoder-epoch units",
     "DERIVED", TREND,
     "collapsed_duplicates lists exactly eight fp32/fp16 pairs; 31 - 8 = 23 = n_probes",
     "SUPPORTED", ""),

    ("P-03", "App. all frozen probes",
     "19 of those carry a stored per-group race summary; the four that do not are cover at 73, 75 and 100 and anatomy-v2 at 75",
     "MEASURED", FAIR,
     "n_probes_with_race_summary = 19; arms has 19 entries",
     "SCOPED-OK",
     "The 19-versus-23 split is an artifact-chronology scope. Both counts are correct and the paper states which test "
     "uses which set, so the two race analyses run on different subsets by construction."),

    ("P-04", "App. replication",
     "The ancestor is locked by content: SHA-256 e5ad5b0c..., 1,507,519,602 bytes, re-hashed before each leg",
     "MEASURED", "autopilot/reports/G1_REPLICATION.md",
     "Size 1,507,519,602 bytes and the matching SHA-256 line",
     "SUPPORTED", ""),

    ("P-05", "App. mask-geometry provenance",
     "18 of the 25 cells fall inside the three-seed range of the measurement",
     "MEASURED", T2PROV,
     "'18 of 25 cells fall inside the three-seed range of the measurement itself'",
     "SUPPORTED", ""),

    ("P-06", "App. mask-geometry provenance",
     "Table 4 anatomy-hidden and loss-slot values with standard deviations over seeds 42, 1234 and 2026",
     "MEASURED", GEOM,
     "with the seed1234 and seed2026 siblings: sd(anatomy hidden) = 0.71/0.64/0.38/0.30/0.39; sd(loss slots) = 1.15/0.43/1.00/0.45/0.00",
     "SUPPORTED",
     "Recomputed from the three artifacts this cycle; every printed cell matches."),

    ("P-07", "App. mask-geometry provenance",
     "Table 5 'delivered at batch 64' context fractions 24.7 / 32.9 / 30.7 / 30.1 / 63.5 per cent",
     "MEASURED", GEOM64,
     "ctx_frac_of_grid per arm",
     "SUPPORTED", ""),

    ("P-08", "App. mask-geometry provenance",
     "Re-measured with the floor at 0.15, cover hides 79.5% of anatomy instead of 73.4% at the same batch size",
     "MEASURED", GEOM64F15,
     "cover.hidden_share_of_all_anat = 79.54 against 73.43 in " + GEOM64,
     "SUPPORTED", ""),

    ("P-09", "App. excluded runs",
     "Four probes of an earlier null run are excluded; they were trained with half-precision EMA targets and a different encoder-truncation policy",
     "MEASURED", INVENTORY,
     "records with status = retracted and arm = random-RETRACTED at epochs 30, 50, 75, 100",
     "SUPPORTED",
     "Cause documented in docs/experiments/masking/crop_and_precision_audit.md."),

    ("P-10", "App. collation defect",
     "an independent sweep over 6,137 slices gives 73.1% against 77.6% for envelope",
     "MEASURED", r"D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json",
     '["0.21"].pct_anat_hid = 73.0945; ["envelope"].pct_anat_hid = 77.5822',
     "SUPPORTED", ""),

    ("P-11", "App. collation defect",
     "Across 24,000 emitted targets only 73.4% remain perfect rectangles",
     "MEASURED", r"D:\jepa_phase0\reports\cover_random_scale\scale_validation.json",
     "blocks_checked = 24000; rectangle fraction confirmed in " + HARDAUDIT,
     "SUPPORTED", ""),

    ("P-12", "App. collation defect",
     "the one-off CPU audit of 194 accepted slices was not persisted, so its pre- and post-truncation figures are no longer asserted",
     "LIMITATION", "autopilot/COVER_AUDIT.md",
     "narrative-only 78.62% / 73.88% with no persisted producing artifact",
     "SUPPORTED",
     "Verified this cycle that 78.6 and 73.9 no longer appear anywhere in main_submission.tex."),

    ("P-13", "App. collation defect",
     "At submission the corrected coverage arm is not run",
     "MEASURED", AUTONUM,
     "\\CoverFixedStatus = 'not run'; \\AUCCoverFixedEp{Fifty,SeventyFive,Hundred} = ---",
     "SUPPORTED", ""),

    ("P-14", "App. subgroup (missing data)",
     "per-group denominators sum to the full test split: race 251/431/2318, sex 1716/1284, ethnicity 2894/106, language 2782/174/44, marital 1741/776/199/191/71/22",
     "MEASURED", META,
     "subgroup_counts.race / sex / ethnicity / language / marital",
     "SUPPORTED",
     "Each of the five categorical breakdowns sums to 3000, which is the check the paper invites the reader to apply."),

    ("P-15", "App. subgroup (missing data)",
     "age is present and numeric for all 3000, spanning 9.5-98.0 years",
     "MEASURED", PROVADD,
     "'all 3000 numeric, range 9.47--97.96'",
     "SUPPORTED", ""),

    ("P-16", "App. subgroup (severity)",
     "334 severe, 460 moderate and 672 mild strata sum to all 1466 positives",
     "MEASURED", r"D:\jepa_phase0\reports\subgroup\subgroup_auc.csv",
     "n_pos for severity strata under any arm; 334 + 460 + 672 = 1466",
     "SUPPORTED", ""),

    ("P-17", "App. subgroup (missing data)",
     "The release uses a mean-deviation sentinel of -1 in 3 of the 3000 test records; all 3 carry a negative label",
     "MEASURED", PROVADD,
     "'Mean-deviation sentinel md = -1, n = 3 / 3000 test records ... All 34 carry a negative label'",
     "SUPPORTED", ""),

    ("P-18", "App. intersectional",
     "Table 8: race-by-sex per-cell AUC, n, n_+ and per-arm bootstrap intervals at epoch 100",
     "MEASURED", INTER,
     "cells[*] for sweep_random_ep100, frozen_meanpool_mirage_ep100 and sweep_oracle_ep100",
     "SUPPORTED",
     "All 18 AUCs, all 6 counts and all 6 printed deltas reproduce to the printed precision; recomputed this cycle."),

    ("P-19", "App. intersectional",
     "The black x female cell is lowest in all 18 arms and asian x male highest in all 18; female below male in all 54; black below white in all 36",
     "MEASURED", "paper/genai4health2026/scripts/intersectional_claims.py",
     "script output: 18/18, 18/18, 54/54, 36/36 against " + INTER,
     "SUPPORTED",
     "Re-run this cycle with SUBGROUP_DIR=D:\\jepa_phase0\\reports\\subgroup; output matched the manuscript exactly."),

    ("P-20", "App. intersectional",
     "max-min gap 0.0340 by sex, 0.0653 by race, 0.1046 across cells; understatement 60.1%; additive 0.0993 against 0.1046, ratio 1.053",
     "DERIVED", "paper/genai4health2026/scripts/intersectional_claims.py",
     "script output lines 'mean gap', 'understatement of worst-cell disadvantage', 'additive prediction'",
     "SUPPORTED", ""),

    ("P-21", "App. intersectional",
     "at epoch 50 the asian x female cell falls by 0.0020; envelope minus random is negative in three of six cells at epoch 75 and two of six at epoch 100",
     "MEASURED", INTER,
     "recomputed cell deltas: ep50 oracle-random asian x female -0.002031; ep75 envelope-random -0.014762/-0.003055/-0.002459; ep100 -0.010020/-0.004020",
     "SUPPORTED",
     "This is the honest counterweight to 'every subgroup point estimate rises' and it reproduces exactly."),

    ("P-22", "App. occlusion attribution",
     "the three fine-tuned probes that tie at test AUC approximately 0.887",
     "MEASURED", "results/downstream/finetune_random/mean_pool_results.json",
     "test_auc 0.886756 with the attentive 0.887763 and cross-attention 0.887178 siblings in the same directory",
     "SUPPORTED",
     "The appendix does not name the arm; the three heads at that AUC are the random arm's, which "
     "docs/experiments/interpretability.md also states. Naming the arm would remove the ambiguity."),

    ("P-23", "Sec. Results (aim) and App. occlusion attribution",
     "Occlusion attribution on the fine-tuned probes is correspondingly diffuse",
     "INTERPRETIVE", "docs/experiments/interpretability.md",
     "'All .npz outputs and per-slice contribution tables are on blob at ijepa-interpretability/'; figure paper/genai4health2026/figures/interp_heatmap_grid.png",
     "PENDING",
     "This is the one appendix result the body cites, and it has no local machine-readable artifact: the attribution "
     "arrays live on an external blob absent from this checkout. Its figure is described in its own caption as "
     "hand-picked illustrative volumes with no colour bar. The appendix does disclose that its numbers cannot be "
     "recomputed from the release. Closes by restoring the arrays or re-running occlusion on the released heads."),

    ("P-24", "App. occlusion attribution",
     "Three tests reject the bilateral-anatomy reading; clustering returns near-perfect mirror images",
     "INTERPRETIVE", "docs/experiments/interpretability.md",
     "figures interp_14_odos_mirror_test.png and interp_slice_contribution_curves.png; no local numeric artifact",
     "PENDING",
     "Same missing-array problem as P-23. The specific correlations that supported it (0.971/0.988, -0.124/-0.478) "
     "were removed from this version rather than restated, which is the right call, but the qualitative claim still "
     "rests on figures alone."),

    ("P-25", "App. operating points",
     "a validation target of 0.90 yields test specificity 0.8696 to 0.8807 across the three arms",
     "MEASURED", SOP,
     "arms.{intensity,envelope,random}.overall.specificity = 0.8696 / 0.8807 / 0.8794",
     "SUPPORTED", ""),

    ("P-26", "App. operating points",
     "At a target specificity of 0.90, ArmBest detects 0.7428 against 0.7162, a paired difference of +0.0266 (CI [+0.0136,+0.0396])",
     "MEASURED", AUTONUM,
     "\\SensIntensitySpecNinety, \\SensRandomSpecNinety, \\DSensIntRandSpecNinety, \\DSensIntRandSpecNinetyCI",
     "SUPPORTED",
     "The paper flags these four arm-by-target CIs as one unadjusted exploratory family."),

    ("P-27", "App. operating points",
     "ArmBest is also the best-calibrated arm (Brier 0.1341 against 0.1416; ECE 0.0320 against 0.0355)",
     "MEASURED", SOP,
     "arms.intensity.overall.brier/ece against arms.random.overall.brier/ece",
     "SUPPORTED", ""),

    ("P-28", "App. operating points",
     "Only the white, female and male gains exclude zero; all three remain after simultaneous adjustment over the five-contrast family",
     "MEASURED", MULT,
     "sensitivity_family.family_size = 5; race:White, sex:Female, sex:Male simultaneous_excludes_zero = true",
     "SUPPORTED", ""),

    ("P-29", "App. operating points",
     "The black interval reaches almost exactly the white point estimate, so the two are not separated",
     "DERIVED", MULT,
     "sensitivity_family: race:Black simultaneous_ci95_hi = 0.04494 against race:White estimate = 0.03426",
     "SUPPORTED",
     "The black upper bound actually overshoots the white point estimate by 0.011, so 'almost exactly' understates the "
     "overlap. The conclusion drawn from it is correct."),

    ("P-30", "App. operating points",
     "under random the shared threshold realises 0.895 specificity in the white stratum but 0.736 in the black stratum, and under ArmBest 0.879 and 0.761",
     "MEASURED", SOP,
     "arms.random.groups.race.{White,Black}.specificity = 0.89499/0.73620; arms.intensity = 0.87884/0.76074",
     "SUPPORTED", ""),

    ("P-31", "App. operating points",
     "ECE improves overall and in the white and asian strata but worsens in the black stratum (0.0525 to 0.0569)",
     "MEASURED", SOP,
     "arms.{random,intensity}.groups.race.*.ece = White 0.03825/0.03209, Asian 0.08318/0.07293, Black 0.05246/0.05687",
     "SUPPORTED", ""),

    ("P-32", "App. ethics",
     "the glaucoma arm of Harvard-FairVision: 10,000 subjects with a fixed 6,000/1,000/3,000 partition; data licence CC BY-NC-ND 4.0; separate MIT licence covering code only",
     "MEASURED", "paper/genai4health2026/research/dataset_facts.md",
     "VERIFIED entries for cohort size, split, data licence and the MIT/code distinction",
     "SUPPORTED", ""),

    ("P-33", "App. ethics",
     "This project holds no IRB or ethics-committee approval number, no consent documentation, and no written exemption determination",
     "LIMITATION", PROVADD,
     "'Explicitly NOT established' items 1-3",
     "SUPPORTED",
     "Stated as outstanding rather than satisfied, which is the correct handling."),

    ("P-34", "App. reproducibility",
     "Python 3.11.9, PyTorch 2.7.1 / CUDA 12.8 / cuDNN 9.7.1, one RTX 3090 driver 610.62, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.9.0, Tectonic 0.17.0",
     "MEASURED", PROVADD,
     "measured version table read from the live interpreter",
     "SUPPORTED", ""),

    ("P-35", "App. reproducibility",
     "A lock file pinning 87 distributions, every pin matching, plus 16 further packages deliberately absent from it",
     "MEASURED", PROVADD,
     "'87 lock entries, 103 installed, 0 version conflicts, 0 lock entries missing'; 16 additions enumerated",
     "SUPPORTED", ""),

    ("P-36", "App. reproducibility",
     "per-case paired differences are non-normal (Shapiro-Wilk W=0.968-0.981) and Levene rejects equal variance (F=21.4) with variance ratio 1.16 at n=9,000",
     "MEASURED", "autopilot/reports/SKILLS_FULL_RUN.md",
     "'W = 0.981' and 'W = 0.968' rows; 'F = 21.446, p = 0.000, variance ratio = 1.16'; n = 9000",
     "SUPPORTED", ""),

    ("P-37", "App. reproducibility",
     "a provenance check resolves each AUC macro's arm and epoch from its own name and fails if the value does not match that arm's stored probe",
     "MEASURED", "autopilot/p15_verify_numbers.py",
     "gate scripts autopilot/check_manuscript.py and autopilot/p15_verify_numbers.py",
     "SUPPORTED",
     "Scripts confirmed present. Not re-executed in this session, so this row records existence rather than a passing run."),

    ("P-38", "App. reproducibility",
     "we audited every numeric quantity typed into the source: 310 occurrences, 234 confirmed, 75 with no producing artifact, and one wrong",
     "MEASURED", HARDAUDIT,
     "'Hardcoded numeric occurrences in scope: 310'; 'WRONG: 1'; 'UNBACKED: 75'; 'CONFIRMED: 234'",
     "PENDING",
     "The counts are exact for the audited snapshot, which the artifact identifies as 1,424 lines with SHA-256 "
     "99891543...; main_submission.tex is now about 2,000 lines. At least 25 of the 75 unbacked items have since been "
     "closed (24 Table 2 geometry cells by the regeneration, plus the interpretability numbers that were removed), so "
     "the figures no longer describe the current source and the paper does not say which snapshot they describe. "
     "Closes by re-running the audit on the current source, or by naming the snapshot in the sentence."),

    ("P-39", "App. reproducibility",
     "Guide provenance differs across the segmenter arms: anatomy-v1 and cover read soft guides from a MIRAGE carrying a frozen residual adapter",
     "MEASURED", "autopilot/reports/ADAPTER_INVESTIGATION.md",
     "guide directory names embedding the adapter configuration; configs/patch_cover_f021_ep25.yaml against configs/patch_mirage_envelope.yaml",
     "SUPPORTED",
     "Recorded as a further confound axis rather than as a result about the adapter, which is the right scope."),

    ("P-40", "App. fine-tuning",
     "ArmBest reaches 0.8947 (mean-pool head) against random's 0.8868, a gap of +0.0079",
     "MEASURED", "results/downstream/finetune_oracle/meanpool_results.json",
     "test_auc 0.8946677 against results/downstream/finetune_random/mean_pool_results.json test_auc 0.8867558",
     "SUPPORTED",
     "Table 13's six ordered values all match their per-run JSONs to six decimals."),

    ("P-41", "App. published ablations",
     "published ablation values for MAE, SimMIM, SemMAE, AutoMAE, HPM, AttMask, AttG-MAE and SSiT",
     "MEASURED", "autopilot/reports/P2-02_related_work.md",
     "literature extraction table",
     "SUPPORTED",
     "Second-hand from the cited papers, which the caption states; metrics differ across rows and are not comparable "
     "between rows."),

    ("P-42", "App. masking policies across the volume",
     "measured on the 6,137-slice sweep (anatomy-v2 on n=1,534); panel (a) reaches 73.1% against envelope's 77.6%",
     "MEASURED", r"D:\jepa_phase0\reports\arm_stats_sweep\cover_floor_sweep.json",
     "*.n = 6137; anatomy arm sample size in D:\\jepa_phase0\\reports\\arm_stats\\arm_stats.json",
     "SUPPORTED", ""),

    # ----------------------------------------------------------- figure claims
    ("F-01", "Fig. trajectories caption",
     "Every envelope and ArmBest interval excludes zero; cover straddles zero at epoch 50 and is negative thereafter",
     "MEASURED", AUTONUM,
     "\\DCoverRandomEpFiftyCI [-0.0050,+0.0053]; ...SeventyFiveCI [-0.0146,-0.0025]; ...HundredCI [-0.0233,-0.0105]",
     "SUPPORTED", ""),

    ("F-02", "Fig. fairness caption",
     "one point per probe over the 19 probes that carry a race summary; the slope fails multiplicity correction and does not survive branch-level aggregation",
     "MEASURED", FAIR,
     "n_probes_with_race_summary = 19; trend values at " + TREND + " :: trends.race",
     "SCOPED-OK",
     "The scatter's 19 probes and the trend test's 23 probes are different sets by construction, and the paper says so "
     "in app:allprobes. Both counts are correct as written."),

    ("F-03", "Fig. precision paradox and specificity ladder captions",
     "The policy whose masks are 97% on-tissue does not separate from the null, while policies at 40-43% purity gain the most; with four to five arms this is descriptive, not inferential",
     "MEASURED", GEOM,
     "hidden_pct_on_anat: anatomy 97.09, envelope 43.30, oracle 40.01, cover 44.19, random 31.47",
     "SUPPORTED",
     "The '40-43%' band describes envelope and ArmBest; cover at 44.19 sits just outside it and is not the gainer, so "
     "the band is a description of the two winning arms rather than of all mid-purity arms."),
]


def main():
    bad_type = [r for r in ROWS if r[3] not in CLAIM_TYPES]
    bad_status = [r for r in ROWS if r[6] not in STATUSES]
    if bad_type or bad_status:
        for r in bad_type:
            print("BAD claim_type: %s %s" % (r[0], r[3]), file=sys.stderr)
        for r in bad_status:
            print("BAD status: %s %s" % (r[0], r[6]), file=sys.stderr)
        return 2

    ids = [r[0] for r in ROWS]
    if len(set(ids)) != len(ids):
        print("duplicate claim_id", file=sys.stderr)
        return 2

    missing = []
    for r in ROWS:
        p = r[4]
        full = p if os.path.isabs(p) else os.path.join(REPO, p.replace("/", os.sep))
        if not os.path.exists(full):
            missing.append((r[0], p))
    if missing:
        for cid, p in missing:
            print("MISSING evidence_path: %s -> %s" % (cid, p), file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in ROWS:
            w.writerow(r)

    counts = {}
    for r in ROWS:
        counts[r[6]] = counts.get(r[6], 0) + 1
    print("wrote %s" % OUT)
    print("claims: %d" % len(ROWS))
    for s in ("SUPPORTED", "SCOPED-OK", "PENDING", "UNSUPPORTED"):
        print("  %-12s %d" % (s, counts.get(s, 0)))
    print("all %d evidence paths exist" % len(ROWS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
