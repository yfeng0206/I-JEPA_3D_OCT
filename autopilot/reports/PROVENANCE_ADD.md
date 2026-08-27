# Provenance additions: dataset licence/ethics, environment, missing data

Closes the three gaps raised in the `peer-review` skill pass recorded in
`autopilot/reports/SKILLS_ROUND2.md`, "Genuinely new findings only", items 1--3.
All three were written into the **appendix only**. The body is unchanged and
still measures 9 pages.

---

## Gap 1 -- dataset licence and ethics provenance

### What was established (all from repository records, none guessed)

| fact | value | evidence in repo |
|---|---|---|
| cohort | glaucoma arm of Harvard-FairVision, 10,000 subjects | `paper/genai4health2026/research/dataset_facts.md` (VERIFIED) |
| split | 6,000 train / 1,000 val / 3,000 test | `dataset_facts.md`; confirmed live: `data_summary_glaucoma.csv` `use` column = training 6000, validation 1000, test 3000 |
| **data licence** | **CC BY-NC-ND 4.0** -- non-commercial research only, not for clinical decisions or patient care | `research/dataset_facts.md` (VERIFIED, sourced to the official FairVision dataset card and the Harvard-Ophthalmology-AI-Lab repository); independently re-confirmed as check **S3-06 CONFIRMED** in `research/verify_sections_3_7_8.md`; already stated in `paper/genai4health2026/main.tex:119` |
| code licence | MIT, covers code only, does **not** relax the data licence | `research/dataset_facts.md` (VERIFIED) |
| access route | public release; no DUA, application or credentialed-access step recorded | searched `configs/`, `docs/`, `README.md`, `scripts/`, `paper/genai4health2026/` -- zero hits for DUA / data use agreement / access request |
| de-identification | release ships per-subject archives holding an OCT volume, a binary label, a visual-field `md`, and coded demographics -- no name, date or free text | verified directly against the released metadata CSV columns (`filename, age, gender, race, ethnicity, language, maritalstatus, md, glaucoma, use`) and the `.npz` attribute keys read by `autopilot/p1_test_metadata.py` |

### What could NOT be established -- written as "not recorded"

- **No IRB / ethics-committee approval number.** A case-insensitive search of
  `docs/`, `configs/`, `README.md` and the whole `paper/genai4health2026/` tree
  for `IRB`, `ethics`, `consent` returned **no approval record of any kind** --
  the only `ethics` hits are the paper's own appendix heading.
- **No consent documentation.**
- **No written non-human-subjects or exemption determination.**

The appendix now says this explicitly and says what closing it would need
(citing the FairVision primary publication's ethics statement, and obtaining a
written determination from the authors' own institution). Nothing was invented.

### What was written

`\section{Broader impact and ethics}` (`app:ethics`), two new paragraphs after
the existing opening paragraph, which was **not** deleted or altered:

- `\paragraph{Dataset provenance and licence.}` -- cohort, split, CC BY-NC-ND 4.0
  terms, MIT-covers-code-only, public access route with no DUA recorded,
  de-identification content of the release, and a statement that the released
  artifacts are weights / head / prediction scores and redistribute no images.
- `\paragraph{What our ethics record does not contain.}` -- the honest gap above.

---

## Gap 2 -- reproducible environment specification

### What was established (measured, not assumed)

Queried directly from `D:\jepa_phase0\.venv\Scripts\python.exe`:

| item | measured value |
|---|---|
| Python | 3.11.9 |
| platform | Windows-10-10.0.26200-SP0, 64-bit |
| PyTorch | 2.7.1+cu128 |
| CUDA (torch build) | 12.8 |
| cuDNN | 9.7.1 (`90701`) |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| GPU (`nvidia-smi`) | NVIDIA GeForce RTX 3090, 24576 MiB, driver 610.62 |
| LaTeX engine | Tectonic 0.17.0 (`D:\jepa_phase0\tools\tectonic\tectonic.exe --version`) |

**Is `requirements-phase0.lock.txt` current?** Yes, and it is cited. `pip freeze`
against the live venv was diffed against the lock file: **87 lock entries, 103
installed, 0 version conflicts, 0 lock entries missing from the venv.** Every
difference is an *addition* -- 16 test/plot/PDF-inspection packages absent from
the lock (`pytest 9.1.1`, `statsmodels 0.14.6`, `seaborn 0.13.2`, `PyMuPDF`,
`pypdf`, `openpyxl`, `patsy`, `pluggy`, `iniconfig`, `click`, `Pygments`,
`remotezip`, `et_xmlfile`, `decompyle3`, `xdis`, `spark-parser`). So the lock is
current for everything a reported number depends on; the appendix says exactly
that rather than claiming the lock is a complete environment image.

### What was written

`\section{Reproducibility and numeric provenance}` (`app:repro`), new leading
`\paragraph{Software and hardware environment.}` with the measured versions
above, the lock-file reconciliation, the 16-package caveat, the single-ordered-
script chain from stored predictions to the compiled archive, and an honest
limitation: no second CUDA/PyTorch version was tested, so the fp32 re-encode of
`app:fp32` is the only cross-environment evidence.

---

## Gap 3 -- missing-data statement

### What the code ACTUALLY does

- `autopilot/p1_test_metadata.py` reads each attribute as `int(z[a])` for all
  3,000 test `.npz` files. There is **no drop path, no NaN path and no
  imputation** -- a genuinely absent key would raise, not be silently skipped.
- `autopilot/p7_fairness.py:76-78` builds group masks from
  `sorted(set(r[col] for r in meta))`, i.e. every case falls into exactly one
  level of every attribute. Nothing is discarded.
- The only set-aside is **from claims, not from data**:
  `p7_fairness.py:32,109,117-119` -- `MIN_N=50`, `MIN_CLASS=10`; a group below
  either is still computed and reported but flagged `underpowered` and excluded
  from the max--min gap summaries.

### Do the per-group n sum to the full test split? YES -- verified live

From `D:\jepa_phase0\autopilot_out\p1_stats\test_metadata.csv` (3,000 rows) and
`test_metadata_summary.json`:

| attribute | levels | counts | sum |
|---|---|---|---|
| race | 3 | Asian 251 / Black 431 / White 2318 | **3000** |
| sex | 2 | Female 1716 / Male 1284 | **3000** |
| ethnicity | 2 | Non-Hispanic 2894 / Hispanic 106 | **3000** |
| language | 3 | English 2782 / Other 174 / Spanish 44 | **3000** |
| marital status | 6 | Married-or-partnered 1741 / Single 776 / Widowed 199 / Divorced 191 / **Unknown 71** / Legally separated 22 | **3000** |
| age | continuous | all 3000 numeric, range 9.47--97.96 | **3000** |
| label | 2 | 1466 positive / 1534 negative | **3000** |

Also: `n_unique_subject_ids` = 3000, and the index-alignment proof in
`test_metadata_summary.json` reports `aligned: true` over 19 prediction files.
Cross-checked against `results/p16_subgroup_operating.json`, whose race
(251+431+2318) and sex (1716+1284) denominators likewise sum to 3000.

**So the denominators DO sum -- there is no silent complete-case reduction.**
The genuine missing-data content is narrower and is now stated:

1. **Marital status "unknown", n = 71 / 3000 (2.4%).** The FairVision metadata
   CSV encodes this as the literal string `unknown` (214 across the full 10,000
   cohort); the distributed `.npz` encodes it as `-1`, which is why
   `p1_test_metadata.py` renders it `code_-1` (its `MARITAL` dict maps Unknown to
   `5`, a code that never occurs). It is carried as its own category -- not
   dropped, not pooled -- so it is subject only to the underpowering rule.
2. **Mean-deviation sentinel `md = -1`, n = 3 / 3000 test records** (34 across
   the full 10,000: 24 train, 7 val, 3 test). **All 34 carry a negative label.**
   Because every severity stratum is scored against the undifferentiated pool of
   all 1,534 negatives, the sentinel never enters a stratum definition.
3. **Severity strata sum to all positives:** 334 severe + 460 moderate + 672
   mild = **1466**, exactly `\Npos`. No positive is lost at a stratum boundary.
   (Confirmed live from `data_summary_glaucoma.csv` `md` with the paper's
   thresholds; matches `tab:severity`.)
4. Nothing is imputed anywhere.

### What was written

`\section{Subgroup and severity analysis}` (`app:subgroup`), new
`\paragraph{Missing data.}` plus two follow-on paragraphs inserted before
`\paragraph{Why we report two sample sizes.}`, giving all of the above counts and
distinguishing the underpowering set-aside (an exclusion from the fairness claim)
from an exclusion of data.

---

## Explicitly NOT established, left as "not recorded" in the paper

1. IRB / ethics-committee approval number for this secondary analysis -- absent
   from the repository.
2. Consent documentation -- absent.
3. A written non-human-subjects / exemption determination -- absent.
4. Which distribution endpoint the local copy was actually downloaded from.
   `research/dataset_facts.md` cites the official Harvard-Ophthalmology-AI-Lab
   repository and the `harvardairobotics/FairVision` dataset card as the licence
   source, while `README.md:48` links a third-party mirror
   (`ming0100/Harvard_FairVision`). The repository does not record which was
   used. The paper therefore says only "taken from the public release" and makes
   no endpoint claim.
5. A FairVision release version or DOI -- not recorded anywhere in the repo, so
   none is quoted.
6. Cross-environment reproduction under a second CUDA/PyTorch version -- not run;
   stated as a limitation rather than implied.

## Hard rules honoured

- No licence, IRB number, approval, version or count invented. Every number in
  the new text traces to a live query or a stored artifact.
- No existing limitation deleted. The pre-existing `app:ethics` opening
  paragraph, the incompleteness-of-audit paragraph and the dual-use paragraph are
  all intact; the new text is additive only.
- No emoji, tick or cross symbols. A PDF codepoint scan above U+2190 returns only
  the pre-existing `->`, `Delta`, minus, `approx`, `<=` and the `fi`/`fl`
  ligatures.

## Verification

Run from `C:\Users\Gary\Desktop\jepa`:

| gate | result |
|---|---|
| `autopilot\p13_build_zip.py` | **6/6 PASS**, main content **9 pages** (limit 9), `ALL_PASS = True`; total 35 pages (was 34 -- the appendix grew by one page, the body did not move; references still start on page 10) |
| `autopilot\check_manuscript.py` | **RESULT: PASS**, undefined macros 0, missing citations 0, labels 56 / refs 56, **dangling 0** |
| `autopilot\p15_verify_numbers.py` | **RESULT: PASS**, 20 AUC macros verified, no cross-arm attribution |

Rendered-PDF spot check: `Dataset provenance and licence` and `What our ethics
record does not contain` on page 25, `Software and hardware environment` on page
26, `Missing data` on page 17. The PDF was rebuilt in place; it was not deleted
or moved.

## Note on commits

**This task did not run `git commit`.** However, a concurrently running agent
committed `paper/genai4health2026/main_submission.tex` at 23:57:36 as
`06d7552 "Explain why the two race analyses run on different probe counts"`,
and that commit swept up the three appendix paragraphs written here alongside its
own one-line prose change. Nothing was reverted, since undoing another agent's
commit would be more destructive than leaving it. All three gates were re-run
against the post-commit working tree and still return
6/6 + 9 pages + `ALL_PASS = True`, `RESULT: PASS` with dangling 0, and
`RESULT: PASS`.
