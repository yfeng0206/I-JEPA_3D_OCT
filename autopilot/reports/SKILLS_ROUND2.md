# Second-round skill evaluation

## Citation errors found

The citation audit found defects that the existing “0 missing” gate could not detect.
The first three affect the rendered paper.

1. **The cited MSMAE reference is an obsolete preprint.** The manuscript cites
   `mao2023msmae`, so the PDF gives the 2023 arXiv version, five authors, and no
   venue. A final journal record is already present in the same `.bib` as the
   unused key `mao2025msmae`: *Applied Soft Computing* 169 (2025), article
   112536, DOI `10.1016/j.asoc.2024.112536`. Crossref gives the final author
   order as Jiawei Mao, Shujian Guo, Xuesong Yin, Yuanqi Chang, Binling Nie,
   and Yigang Wang. This is simultaneously a preprint-versus-published
   mismatch, wrong final author list/order, wrong year, and missing final
   venue in the citation that readers see.

2. **The cited HU foreground-masking entry has the wrong final year and
   incomplete venue metadata.** `lee2025hufgm` contains DOI
   `10.1007/978-3-032-09569-5_13`, but the live Crossref record says first
   online 2 January 2026, pages 122--131, in the Springer book *Applications
   of Medical Artificial Intelligence* (Lecture Notes in Computer Science).
   The `.bib` says 2025, gives only generic “MICCAI” as `booktitle`, and has no
   pages. Its ten-author list matches Crossref.

3. **Hard Patches Mining is rendered twice.** Both cited keys
   `wang2023hardpatches` and `wang2023hpm` have the same title, six authors,
   pages, and DOI `10.1109/CVPR52729.2023.01000`. `main_submission.bbl`
   renders them as separate 2023a and 2023b bibliography items. This is one
   paper, not two.

4. **The cited Maetschke glaucoma entry is incomplete.**
   `maetschke2019glaucoma` omits the stable identifier and article number.
   OpenAlex, PubMed PMID 31260494, PLOS, and Crossref agree on DOI
   `10.1371/journal.pone.0219126` and article number `e0219126`. Its title,
   six authors, journal, year, volume, and issue are otherwise correct.

Defects in currently unused `.bib` entries were also found because the request
was to validate every entry:

- `abolade2026vamae` still describes an accepted arXiv preprint whose final
  proceedings metadata was unavailable. Crossref now has the Springer/ICPR
  chapter, first online 4 August 2026, pages 394--408, DOI
  `10.1007/978-3-032-31654-7_27`; the author list is unchanged. Crossref also
  records a 2027 print year, so 2026 is valid only when citing first-online
  publication.
- `mo2024cjepa` is an arXiv `@misc`, but OpenAlex now identifies a published
  NeurIPS 2024 record in *Advances in Neural Information Processing Systems
  37*, pages 2348--2377, DOI `10.52202/079017-0077`; its two-author list and
  year are unchanged.
- `zhou2021modelsgenesis` has the wrong given name for its third author:
  `Pang, Jiang` should be `Pang, Jiaxuan`. The installed PubMed extractor
  verified PMID 33188996, year 2021, *Medical Image Analysis* 67, article
  101840, DOI `10.1016/j.media.2020.101840`. The current entry also omits the
  article number and DOI.
- `ceballosarroyo2026anatomymae` and
  `ceballosarroyo2026aneurysm` duplicate DOI
  `10.1109/WACV61042.2026.00552`. Their page ranges disagree
  (`5693--5702` versus `5693--5694`). Crossref currently gives
  `5693--5694`, while the longer range could not be confirmed from an
  accessible primary IEEE/CVF record. The authoritative pagination is
  **PENDING**; the duplicate itself is certain.

No other high-confidence wrong title, year, venue, or author list was found.
All 51 DOI-bearing entries resolved: 43 through Crossref and eight arXiv
records through DataCite. There were no lookup transport failures and no
unresolvable DOI. Seventeen entries have no DOI field; the specific actionable
omissions or publication upgrades are listed above.

## Skill 1: `citation-management`

**Decision: INSTALL.**

It earned its place because structural citation resolution was insufficient:
the run found a cited preprint whose final publication was already in the
bibliography, a wrong final year, a duplicated rendered paper, an incorrect
author name, and newly available publication metadata. It was installed at
project scope in `.agents/skills/citation-management`, pinned to upstream
commit `36d8f13a1e754618794bf42f417884940077b4ae`. The required `requests`
runtime dependency was absent and was installed as version 2.34.2.

### What was inspected and run

- Read the raw `SKILL.md` before installation and read all ten bundled
  references.
- Ran `gh skill preview K-Dense-AI/scientific-agent-skills
  citation-management`; the preview contained `SKILL.md`, two assets, ten
  references, and eight Python scripts.
- Ran the installed `validate_citations.py` with `--check-dois` and the real
  LaTeX manuscript.
- Parsed and checked all 68 BibTeX entries, searched exact titles in OpenAlex,
  queried DOI registrars, and checked 24 biomedical records in PubMed.
- Ran the installed PubMed metadata extractor directly on PMID 33188996.

The bundled validator reported 28 missing-recommended-field warnings, 20
unused entries, and two duplicate DOI pairs. Its LaTeX citation parser also
mistook the literal table header `AUC @ep50` for a citation and reported a
nonexistent unresolved key. A separate LaTeX-command parse found the actual
48 unique `\cite...{}` keys and zero unresolved keys. That `@ep50` warning is
a validator false positive, not a paper defect.

### Security audit

Result: **acceptable with a broader network surface than “OpenAlex and
PubMed only.”**

- No `subprocess`, `os.system`, `eval`, `exec`, shell execution, or file
  deletion call was found.
- Network code is limited to citation retrieval, but it is not limited to two
  hosts. The scripts name `api.openalex.org`, `api.crossref.org`,
  `api.datacite.org`, `doi.org`, `eutils.ncbi.nlm.nih.gov`,
  `www.ncbi.nlm.nih.gov` for PMCID conversion, and `export.arxiv.org`.
  The optional Scholar script uses the third-party `scholarly` package and
  optional proxies. `extract_metadata.py --url` can fetch a URL explicitly
  supplied by the operator to discover citation metadata. There is no hidden
  telemetry or unrelated fixed endpoint.
- The only environment values read are optional `OPENALEX_EMAIL`,
  `NCBI_EMAIL`, and `NCBI_API_KEY`; code sends each only to its corresponding
  service. It does not enumerate the environment.
- Scripts write only to an explicit output path; `format_bibtex.py` can
  overwrite its input only when `--in-place` is explicitly supplied.
- The markdown prompt-injection grep found no override, jailbreak, hidden
  prompt, secret-disclosure, or “ignore prior instructions” pattern.
- All eight installed Python files parsed successfully. This audit run itself
  used only Crossref, DataCite, OpenAlex, and PubMed.

## Skill 2: `peer-review`

**Decision: INSTALL.**

It earned its place because its structured reporting, reproducibility, ethics,
and figure/table pass differs from seven general NeurIPS-form reviews. It was
installed at project scope in `.agents/skills/peer-review`, pinned to the same
commit. The raw `SKILL.md` and all six references were read first, and
`gh skill preview K-Dense-AI/scientific-agent-skills peer-review` showed nine
assets, six references, and eight Python scripts.

### What was run

- Read `autopilot/reports/R6_final_review.md` before assessing novelty.
- Processed the real `main_submission.pdf` locally. SHA-256:
  `F195EA4104405D2D19A0F05C9F072A78820F5A8F3C4DBBB841D61F9CF1415205`.
  The PDF has 34 pages total, including the nine-page body and appendices.
- Passed the installed intake gate with status `READY_FOR_LOCAL_REVIEW`.
- The installed reporting selector chose TRIPOD+AI 2024 for the health
  prediction-model evaluation profile.
- Completed and ran the installed 22-item statistics/reproducibility audit.
  It returned `VALID_WITH_REVIEW_GAPS`: four verified-present, 16
  partly-documented, one missing, and one not-applicable item.

### Genuinely new findings only

1. **Ethics/governance support is too implicit.** PDF pages 9 and 24 say the
   data are public and de-identified, used “under its release terms,” and that
   the work involves no human subjects. The PDF does not identify the exact
   FairVision release/version or license, cite the source dataset's
   ethics/consent statement, or state the basis for an exemption/non-human-
   subjects determination. Name the governing release terms and source ethics
   record so readers can verify the assertion. Funding and competing-interest
   reporting should also be supplied wherever the anonymized venue permits it.

2. **The reproducibility claim lacks an environment specification in the
   paper.** PDF page 8 says a script, weights, head, and predictions are
   released with links withheld for anonymity, and the appendices give many
   seeds and hashes. The PDF does not identify the training framework and
   package versions, a lock/environment file, or end-to-end run instructions.
   Ensure the anonymous artifact contains those items and connect released
   checkpoints and predictions to versioned code/configuration.

3. **Missing-data handling is not stated.** The PDF gives the 3,000-case test
   denominator and several subgroup denominators, but does not say whether any
   scans, labels, or demographic fields were missing or how missing fields
   were handled. A one-sentence accounting would close this TRIPOD+AI-style
   reporting gap.

No genuinely new figure/table-integrity defect was found. The truncated axes
in Figures 13--14 and the absent color bar in Figure 11 are explicitly
disclosed and their interpretations are bounded. Findings already present in
R6, including adaptive test reuse, one continuation per policy, confounded
arms, selective horizons, pretraining cohort counts, and earlier numerical
contradictions, were intentionally not repeated.

### Security audit

Result: **acceptable and local-only.**

- No network library or URL, `subprocess`, `os.system`, `eval`, `exec`, shell
  execution, environment read, or pickle/dynamic-code path was found in the
  eight scripts.
- Output uses bounded local JSON/CSV/Markdown processing. The only delete or
  replace operations are cleanup and atomic replacement of a temporary output
  file; the shared path guards reject symlinks and implicit overwrite.
- The markdown prompt-injection grep found no active injection pattern. The
  word “exfiltration” appears only in
  `references/security_validation.md`, which documents removal of older
  unsafe schematic scripts and a subsequent zero-finding scan.
- All eight installed Python files parsed successfully.

## Repository impact

Only the two pinned project skills and this report are retained. Neither
`paper/genai4health2026/main_submission.tex` nor
`paper/genai4health2026/references.bib` was edited.
