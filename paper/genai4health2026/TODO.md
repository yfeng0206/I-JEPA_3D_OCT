# TODO / ROADMAP -- GenAI4Health @ NeurIPS 2026 submission

Working doc. Not compiled into the PDF. Delete or keep out of the Overleaf
compile -- it is a `.md`, so LaTeX ignores it.

Submission facts: Research track, **<= 9 pages main content** (references and
appendix excluded), deadline **Sep 5 2026 AoE**, double-blind, **no rebuttal**,
non-archival.

Current state: **main body = 9 pages exactly (zero slack). Appendix starts p10
and appendix pages are unlimited.** Verify after every edit with:

```
cd paper/genai4health2026
tectonic -X compile main.tex --keep-intermediates
grep endofmain main.aux      # the second number must stay 9
```

---

## 1. THE HEADLINE UPCOMING EXPERIMENT -- dense transfer / segmentation

**This is now written into the paper as Appendix Section "Planned extension:
dense transfer".** It is stated as a design plus a falsifiable prediction, with
an explicit banner that nothing in it has been run. That framing is deliberate:
it lets us claim the direction without claiming a result we do not have.

**The claim we want to be able to make later:**

> Take the final JEPA encoder pretrained with anatomy-shaped (MIRAGE-guided)
> targets and our loss, attach a segmentation decoder, fine-tune on a NEW
> labelled dataset, and get a better segmentation score than the
> rectangular-target baseline at matched compute.

**Why this is the right next experiment (and why it belongs in the paper as
future work rather than as a result):** the current audit uses frozen linear
probes on a volume-level classification label. That is the setting *least*
likely to reward target geometry, because a volume label can be read off pooled
features that do not preserve where anatomy is. Segmentation is where target
geometry has an actual mechanism to matter, because the pretext target and the
downstream output share a coordinate system. So a null result on classification
does not predict a null result on segmentation -- and saying so in print, in
advance, is much stronger than saying it afterwards.

### What has to happen, in order

- [ ] **Acquire an external labelled retinal-layer segmentation set.** Must be
      disjoint from the audited cohort. This is the single blocking dependency.
      Candidate sources need a license check before use.
- [ ] Decide the decoder. Keep it lightweight and **identical across all arms** --
      the whole point is that the encoder is the only thing that differs.
- [ ] Run BOTH regimes. Frozen encoder measures what pretraining already
      encoded; full fine-tuning measures what it made easy to reach. They answer
      different questions and reviewers will ask for both.
- [ ] Transfer all five arms: random, oracle, COVER, envelope, anatomy-shaped.
      No extra pretraining -- inherit the compute-matched schedule so the
      warm-start confound is the same one already documented.
- [ ] Report Dice + boundary error per layer, stratified by race and sex, with
      **cell sizes printed** and bootstrap intervals over volumes.
- [ ] Pre-commit to the falsification condition: overlapping intervals between
      anatomy-shaped and rectangular means target geometry is NOT transferable.
      Report it either way.

### Honesty guardrail

Until those runs exist there is **no segmentation number** to quote. Do not let
a draft slide from "we predict" to "we show". The appendix section is worded to
make that slide visible if anyone attempts it.

---

## 2. Open items before submission

### Blocking
- [ ] **COVER floor-0.21 epoch-50 AUC.** Pretraining is live; ETA ~07:00 Aug 20.
      Fills the `--` in the composition table and converts the dashed pending
      line in the inverted-U figure.
      **Branch risk:** COVER currently holds the three lowest AUCs of all 18
      arms. If ep50 lands below random (.8641), Section 7 must be rewritten as a
      negative result. Draft both versions.
- [ ] Land the intersectional race x gender result in the equity section.
      Result is already computed and unanimous across 18 arms. Two distinct
      "18/18" claims now exist (marginal vs intersectional) -- **word them
      differently or a reviewer will read it as one recycled number.**

### High value, cheap, no GPU
- [ ] Reframe from pure audit to **audit + minimal fix**. We already own two
      remedies (the COVER visibility floor, the collation repair) and currently
      disclaim both. The comparable 2025 oral that reads as a pure audit is
      actually ~1/3 proposed remedy.
- [ ] Name the failure mode in clinically legible language ("silent anatomical
      erasure") and cite it into the shortcut-learning literature.
- [ ] Open with a one-patient vignette instead of a method sentence.
- [ ] Write the single "not because X, but because Y" thesis sentence and place
      it in the abstract right after the method sentence.
- [ ] Add standard errors to the composition table. Values are computed and
      independently verified against the binomial formula. Use the compact
      `3.68 (.24)` form and define it in the caption -- much narrower than `\pm`.
- [ ] State the bootstrap iteration count and the corrected alpha explicitly.
      We already compute these and under-report the protocol.
- [ ] Convert at least two planned figures into tables. Reviewers scan tables
      for rigor and figures for the story; six figures and no tables reads thin.
- [ ] Expand the appendix. Precedent appendix is ~9 pages; ours is currently two
      figures plus the planned-extension section. Appendix pages are free.
- [ ] Terminology: "predictor collapse" -> "predictor deterioration consistent
      with a shortcut" (4 sites).
- [ ] Name the estimator behind the +0.0054 / +0.0044 pair; both are correct,
      the defect is that neither is named.

### Deliberately NOT doing (precedent says unnecessary)
- Pretraining-seed replication -- the 2025 fairness oral used a single training
  run and said so.
- Human / reader study -- the Best Paper had none.
- Foundation-scale trained baseline -- use frozen off-the-shelf encoders instead.
- Model-weights release -- code + repro script + eval harness is sufficient.

---

## 3. SECTIONS WE NEED HELP WITH

Flagged honestly. These are the parts most likely to be wrong or weak, and the
places where a second pair of eyes buys the most.

1. **Clinical framing of the harm.** The mechanism is solid; the *clinical*
   consequence argument is the weakest writing in the paper and is exactly what
   carried the 2025 Best Paper. Need a clinician read on: is "the encoder is
   never required to explain the region where glaucoma is diagnosed" a claim an
   ophthalmologist finds compelling or overstated?

2. **External baselines.** We currently have **zero**. Cheapest fix is frozen
   public encoders (ImageNet I-JEPA, DINOv3) probed identically and reframed as
   "auditing existing foundation models" -- no retraining. Need help choosing
   which encoders are defensible and getting the preprocessing honest.

3. **The segmentation dataset.** Blocking item 1 above. Need a source that is
   licensed for this use and genuinely disjoint from the audited cohort.

4. **Equity statistics.** Small cells (smallest is n=123). Need a review of
   whether the intersectional claim is stated at the right strength, and whether
   the additivity read should stay labelled INFERRED (it is not a formal
   interaction test).

5. **The two-audit-pass discrepancy.** Two independent passes measure the same
   quantity and disagree; the largest gap is ~2 points at z=2.41, which is NOT
   statistically compatible. Currently disclosed in the figure caption. Needs a
   decision: disclose only, re-measure, or drop the cross-reference. Also
   unresolved whether the smaller pass is a subset of the larger, which would
   invalidate the two-sample test entirely.

6. **Related work.** Thin. Two known missing citations already identified.

---

## 4. Housekeeping

- [ ] Anonymity sweep before submission: no author names, no institution, no
      repository links, no identifying paths in figures.
- [ ] **`paper/genai4health2026/` is currently untracked in git.** Decide
      whether to commit it. Nothing here is backed up by version control right
      now.
- [ ] Re-verify every number against raw artifacts, and grep the figure
      generators as well as the prose, before the final build.

## 5. Style rules in force

- No emoji, no tick or check symbols anywhere.
- Never invent a number. If evidence does not exist, say so.
- Label every claim MEASURED / INFERRED / ASSUMED / HYPOTHESIS.
- Three AUC families are mutually incomparable (mean-pool sweep,
  region-restricted, attribution `auc_full`). **Never cross-quote them.**
