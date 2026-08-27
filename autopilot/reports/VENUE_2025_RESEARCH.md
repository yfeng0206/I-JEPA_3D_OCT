# GenAI4Health @ NeurIPS — Venue Calibration Research (2025 edition, with 2026 call)

Research date: 2026-08-27. Prepared for the 2026 submission (deadline 2026-09-05).

Every finding below is labelled **MEASURED** (read directly off a page or a PDF I downloaded and
parsed) or **INFERRED** (reasoning on top of measured facts). Where I could not verify something,
it is stated plainly under "Gaps and uncertainties" rather than guessed.

Important access limitation, stated up front: **openreview.net is behind a Cloudflare Turnstile
browser challenge from this environment.** Every request to `openreview.net`, `api.openreview.net`
and `api2.openreview.net` returned HTTP 403 with `ChallengeRequiredError`, and the Wayback Machine
has no snapshots of any GenAI4Health forum page. So I could not read OpenReview forum pages, review
threads, or the OpenReview-hosted PDFs. I worked around this using (a) the workshop's own site
JavaScript bundle, (b) the official NeurIPS virtual site listing of accepted papers, and (c) arXiv
copies of the accepted papers, which for this venue are overwhelmingly the camera-ready versions.

---

## 1. THE PAGE LIMIT

### **MEASURED: 9 pages of main content for the Research Paper track — stated by the workshop itself, in both the 2025 and the 2026 call. Your 9 body pages are exactly at the limit, not over it. There is no desk-reject problem on length.**

**2026 call (the one that governs your submission).** The number is on the page, but it is rendered
as a badge in the track header, which is why plain-text scrapes and the earlier web search missed
it. Exact HTML from https://genai4health.github.io/2026-NeurIPS/ :

```html
<article class="genai-track-panel">
  <header><h4>Research Papers</h4><span class="genai-track-page-limit">Up to 9 pages</span></header>
```

Rendered, the three track headers read:

- "Research Papers — **Up to 9 pages**"
- "Demonstration Papers — **Up to 5 pages**"
- "Position Papers — **Up to 5 pages**"

and separately, in Submission Format Requirements:

> "All page limits exclude acknowledgments, references, and appendix."

URL: https://genai4health.github.io/2026-NeurIPS/ (sections "Submission Tracks" and "Submission
Format Requirements"). MEASURED.

**2025 call.** The 2025 site is a React single-page app whose text lives in
`https://genai4health.github.io/2025-NeurIPS/static/js/main.897dc88a.js`. The call-for-papers track
definitions are stored verbatim in that bundle:

> `title: "Track 1: Research Papers", subtitle: "Form the core of the program and present
> methodological advances and empirical evaluations", description: "`**`The main content of the
> paper should be no longer than 9 pages.`**`"`

> `title: "Track 2: Demonstration Papers" ... "The main content of the paper should be no longer
> than 5 pages. <strong>The paper title should start with 'Demo:'</strong>."`

> `title: "Track 3: Position Papers" ... "The main content of the paper should be no longer than 5
> pages. <strong>The paper title should start with 'Position:'</strong>."`

Same bundle, Submission Format Requirements:

> "**All page limits exclude acknowledgments, references, and appendix**"
> "Papers may be rejected without consideration of their merits if they fail to meet the submission
> requirements"

URL: https://genai4health.github.io/2025-NeurIPS/ (Call for Papers section; text extracted from
`/2025-NeurIPS/static/js/main.897dc88a.js`). MEASURED.

**Independent corroboration from the accepted papers themselves.** I downloaded and parsed 17
research-track accepted papers from arXiv and located the page on which the References section
begins. Body length (everything before References) was:

- **9 body pages: 14 of 17 papers**
- 8 body pages: 2 papers
- 7 body pages: 1 paper
- **more than 9 body pages: 0 papers**

One Demo-track paper I parsed ("Demo: Statistically Significant Results ...", arXiv 2511.02246) has
exactly **5** body pages, matching the 5-page demo limit. This is a very tight distribution piled up
against a hard ceiling, which is what a real 9-page limit looks like. MEASURED (page counts);
INFERRED (that the pile-up is caused by the limit).

Two further direct corroborations:
- arXiv 2511.19940 ("Editing with AI", accepted poster) carries the author comment "**9 pages**, 2
  figures, 1 table". https://arxiv.org/abs/2511.19940 MEASURED.
- arXiv 2510.03700 ("H-DDx", accepted oral) carries the comment "**GenAI4Health @NeurIPS 2025**" and
  is 9 body pages + 7 appendix pages. https://arxiv.org/abs/2510.03700 MEASURED.

**Confidence: very high.** The number is stated by the workshop, in both years, in the workshop's own
words, and is confirmed by the empirical length distribution of accepted papers. Do not shorten the
paper. Note also that the earlier "9 pages from the NeurIPS main-track handbook" claim happens to
coincide, but the workshop's own statement is the authoritative one and is what is cited here.

**Appendix is effectively uncapped and is heavily used.** Measured appendix lengths among accepted
papers: 3, 3, 4, 6, 6, 7, 11, 11, 12, 12, 14, 15, 16, 17, 18, **25** pages. The median accepted
research paper is roughly 9 body pages plus a 12-page appendix. Anything you cannot fit in 9 pages —
per-seed tables, the implementation post-mortem detail, extra ablations — belongs in the appendix,
and the venue's own accepted papers show reviewers are used to that. MEASURED.

---

## 2. ACCEPTED PAPERS FROM 2025 (research-paper track survey)

**Venue scale.** 99 accepted papers total. Source: the NeurIPS virtual page for the workshop,
https://neurips.cc/virtual/2025/workshop/109566 , which lists 9 oral presentations plus 89 accepted
posters and links each to an OpenReview forum id; and https://aiworkshoptracker.com/workshop/neurips-2025-genai4health/
which states "Accepted papers (99) — Fetched from OpenReview (v2) on 2026-06-10". MEASURED.

**Track split.** The 2025 call required title prefixes ("Demo:", "Position:"), so track membership is
readable off the title list. Of the 89 accepted posters: **15 Demo, 9 Position, 65 Research**. Of the
9 orals: 7 Research, 1 Demo, 1 Position. So the Research track is roughly two-thirds of the program,
about 72 papers. MEASURED (counts), INFERRED (that prefix perfectly determines track).

**Acceptance rate: not published.** See Gaps.

### Survey table

Body pages = pages before the References section, measured by parsing the PDF. "Variance" = whether
the paper reports results over multiple runs/seeds.

| # | Title | Source | Body / total pp. | Contribution type | Multiple seeds? |
|---|---|---|---|---|---|
| 1 | Uncovering Intervention Opportunities for Suicide Prevention with Language Model Assistants (**oral**) | arXiv 2508.18541 | 9 / 21 | Clinical evaluation + expert study | **No multi-seed training.** Bootstrap CIs and p-values on the evaluation set only |
| 2 | H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis (**oral**) | arXiv 2510.03700 (comment: "GenAI4Health @NeurIPS 2025") | 9 / 17 | Evaluation framework / benchmark over 22 models | **No.** Zero occurrences of "seed", no std, no error bars |
| 3 | High-Fidelity Synthetic ECG Generation via Mel-Spectrogram Informed Diffusion Training (**oral**) | arXiv 2510.05492 | 9 / 16 | New method (diffusion training objective) | **No seed statement.** Uses ± widely but no repeated-training-run vocabulary |
| 4 | Count-Based Approaches Remain Strong: A Benchmark Against Transformer and LLM Pipelines on Structured EHR | arXiv 2511.00782 | 9 / 22 | **Benchmark / negative result** | **Partially.** mean ± sd for two arms; the TabPFN and mixture-of-agents arms are literally reported as "**±N/A**", i.e. a single run |
| 5 | Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs | arXiv 2510.12255 (comment: "Accepted as a poster at NeurIPS 2025 Workshop on GenAI for Health") | 9 / 22 | Benchmark + **negative evaluation finding** | **No — explicitly single-run.** "All experiments employ deterministic decoding with temperature=0 and fix the random seed to 42 for reproducibility." |
| 6 | Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions | arXiv 2510.13931 (states "this version corresponds to the camera-ready paper accepted ... GenAI4Health") | 8 / 12 | Bias/fairness audit, **negative result** | **No — and it says so.** Concedes "ordering-based selection may introduce temporal bias; future work could adopt random sampling with fixed seeds". Accepted anyway |
| 7 | When the Domain Expert Has No Time and the LLM Developer Has No Clinical Expertise: Real-World Lessons from LLM Co-Design in a Safety-Net Hospital | arXiv 2508.08504 | 7 / 22 | **Case study / implementation post-mortem / lessons learned** | **No — explicitly single-run.** "fixed seed value of 0 and temperature setting of 0, ensuring deterministic outputs with a single LLM extraction performed per patient" |
| 8 | Brittleness and Promise: Knowledge Graph-Based Reward Modeling for Diagnostic Reasoning | arXiv 2509.18316 | 9 / 13 | **Mixed/negative result** ("promise and brittleness ... transferability to downstream tasks remain weak") | **No.** No seed or variance vocabulary found |
| 9 | Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction | arXiv 2510.10454 | 9 / 25 | New method | **Yes — the exception.** "The average performance (mean±std) for Traj-CoA across **5 runs with different random seeds** is reported." Even here the sensitivity analyses use one fixed seed |
| 10 | CancerGUIDE: Cancer Guideline Understanding via Internal Disagreement Estimation | arXiv 2509.07325 | 9 / 21 | Benchmark + method | Partial — some ± reporting, one "averaged over runs" phrase; no systematic seed protocol |
| 11 | MedGUIDE: Benchmarking Clinical Decision-Making in Large Language Models | arXiv 2505.11613 | 9 / 28 | Benchmark (7,747 items, 25 LLMs) | **No.** Contains an explicit single-run/compute concession |
| 12 | Editing with AI: How Doctors Refine LLM-Generated Answers to Patient Queries | arXiv 2511.19940 (comment: "9 pages") | 9 / 14 | **Human-subjects clinical evaluation** (9 ophthalmologists, 144 questions) | N/A — human study; reports ± across participants |
| 13 | Large Language Models as Medical Codes Selectors (ICPC-2) | arXiv 2507.14681 (comment names the OpenReview forum, id `Kl7KZwJFEG`) | 9 / 35 (25-page appendix) | Benchmark of 33 LLMs | **No.** No seed or variance vocabulary |
| 14 | HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring | arXiv 2509.07260 | 8 / 26 | Benchmark | **No** |
| 15 | MedBrowseComp: Benchmarking Medical Deep Research and Computer Use | arXiv 2505.14963 | 9 / 26 | Benchmark / dataset | Partial — error bars, one "averaged over runs" phrase |
| 16 | FedMentor: Domain-Aware Differential Privacy for Heterogeneous Federated LLMs in Mental Health | arXiv 2509.14275 | 9 / 16 | New method | **No** — a single ± occurrence |
| 17 | MedVLThinker: Simple Baselines for Multimodal Medical Reasoning | arXiv 2508.02669 | 8 / 15 | **Simple-baseline study** | **No** — a single std occurrence |

Two extra data points measured but excluded from the length statistics because their arXiv versions
are clearly extended, non-workshop versions:
- MedVAL (oral), arXiv 2507.03152 — 15 body / 34 total. The workshop version must have been cut to 9.
- Clinically Grounded Agent-based Report Evaluation (ICARE), arXiv 2508.02808 — 19 body / 29 total,
  journal-style. Notably it *does* report stability "across evaluation seeds, with standard
  deviations below 1%".

### Did any accepted paper report single-run results? Yes — most of them.

**MEASURED. Of the 17 research-track accepted papers I parsed, exactly one (Traj-CoA, #9) reports a
headline result averaged over multiple seeds. Three papers state in plain text that they ran a
single deterministic pass (#5 seed fixed to 42; #7 "a single LLM extraction performed per patient";
#4 reporting "±N/A" for two of its four method arms). The remaining ~12 simply never mention seeds,
repeated runs, or run-to-run variance at all.**

This includes the accepted **orals**: H-DDx (#2) and High-Fidelity Synthetic ECG (#3) both won oral
slots with no multi-seed protocol at all, and #3 is a *new training method* paper — the case where
seed sensitivity matters most.

**INFERRED strategic reading.** n=1 is the norm at this venue, not a disqualifier. Your central
weakness — one pretraining run per policy — is quantitatively *stronger* practice than roughly
three-quarters of the 2025 accepted research track, because you at least know and name the issue.
Two concrete moves that the 2025 record supports:
1. Do not shorten or hedge the claim to dodge the n=1 criticism. Paper #6 (Robust or Suggestible?)
   was accepted while explicitly conceding in its limitations that it should have used random
   sampling with fixed seeds.
2. Put whatever variance evidence you do have — seed sensitivity of the downstream probe, variance
   across evaluation folds, bootstrap CIs on the test metric — in the body, and the rest in the
   appendix. Papers #1 and #15 got credit for bootstrap CIs and error bars on *evaluation*, without
   ever repeating *training*. That is a cheap and venue-consistent way to blunt the criticism.

Also note the 2026 call's explicit instruction, which is effectively an invitation to pre-empt this:

> "There is no author rebuttal or response stage. ... Please ensure your submission is
> self-contained and **anticipates likely questions (e.g., in a limitations section)**."
> — https://genai4health.github.io/2026-NeurIPS/ , Review Process. MEASURED.

Every one of the 17 papers I parsed contains a Limitations section or paragraph. MEASURED.

---

## 3. REVIEWS

**MEASURED (what the workshop states):**

2025 call, Review Process section (from the 2025 site JS bundle):
> "Each paper will receive **at least two anonymous reviews**"
> "**All submissions must be anonymized and may not contain any identifying information**"
> "This policy applies to any supplementary or linked material as well, including code"
> "**Please do not include acknowledgments at submission time. Any papers found to be violating this
> policy will be rejected**"

2026 call, Review Process section:
> "**Number of reviews:** Each submission receives at least two reviews from the program committee,
> matched by topic area and track."
> "**Author response:** There is no author rebuttal or response stage. Given the short turnaround
> between the submission deadline and notification, decisions are made based on reviews and program
> committee discussion only."
> "**Confidentiality:** Submissions are visible only to assigned reviewers and organizers during
> review. **Rejected submissions will not be made public.**"

URL: https://genai4health.github.io/2026-NeurIPS/ . MEASURED.

**Score distribution and review text: NOT OBTAINED.** I could not read a single review. OpenReview
blocked every request (Cloudflare Turnstile, HTTP 403 `ChallengeRequiredError` on
`openreview.net/forum?id=...`, `api.openreview.net`, `api2.openreview.net`, and
`openreview.net/pdf?id=...`), and there are no Wayback snapshots of any GenAI4Health forum page
(checked via the Wayback CDX API for forum ids `fKJtKew2YQ`, `Kl7KZwJFEG`, `EqFncAMILF`,
`2ce9IovlPF` — all "no snapshots"). A web search returned a claim that reviews are public, but the
citations it gave were the **NeurIPS main-track** call for papers and a generic wiki, not the
workshop — the same failure mode as the earlier "9 pages" claim, so I am not repeating it as fact.

**INFERRED, and I would act on this:** the workshop's own text is about *papers* being made public,
never about *reviews* being made public — 2025: "Accepted workshop papers (after the camera-ready
stage) will be made publicly available by default ... their paper (PDF file) will not appear on
OpenReview" if they opt out; 2026: the same, plus "Rejected submissions will not be made public".
The consistent framing is PDF visibility, not review visibility. Combined with the two-reviewer,
no-rebuttal, 24-day-turnaround process, my working assumption is that reviews for this venue are
**not** publicly posted. **Treat this as unverified.** A logged-in OpenReview account would settle it
in one page load: open https://openreview.net/forum?id=fKJtKew2YQ and look for an "Official Review"
note. That is the single highest-value follow-up and it takes a minute.

**What reviewers punished / forgave — INFERRED from what got in, not from review text.** With no
review text available, the only honest signal is the revealed preference of the accepted set:
- **Forgiven:** single-run results; no seed protocol; no error bars on training; benchmarks with no
  new method; short 7-8 page papers; negative and mixed results; small-N human studies (n=9
  clinicians); one-hospital / one-dataset external-validity limits (paper #4's entire Limitations
  paragraph is "single-institution dataset and a limited set of outcomes" and it was still accepted).
- **Punished, per the stated rules rather than observed reviews:** anonymity violations
  (de-anonymising links, acknowledgments at submission) are called out in both years as grounds for
  rejection / desk rejection; failure to meet format requirements ("Papers may be rejected without
  consideration of their merits"); previously published work; and, per the 2026 track description,
  "promising directions without supporting evidence" in the Research track.

---

## 4. NEGATIVE-RESULT AND REPLICATION PAPERS

**MEASURED: yes, clearly and repeatedly. The 2025 program accepted negative results, "simple
baseline wins" benchmarks, brittleness studies, and at least one explicit implementation
post-mortem — including in oral slots.** Your framing is viable at this venue.

The closest structural match to your paper is #4:

> **"Count-Based Approaches Remain Strong: A Benchmark Against Transformer and LLM Pipelines on
> Structured EHR"** — arXiv 2511.00782, accepted poster, 9 body pages + 12 appendix pages.
> Abstract: "Across the eight evaluation tasks, head-to-head wins were largely split between the
> count-based and the mixture-of-agents methods. **Given their simplicity and interpretability,
> count-based models remain a strong candidate for structured EHR benchmarking.**"
> Conclusion: "**our results reaffirm the strength of count-based modeling for structured EHR
> prediction, even in the era of LLMs** ... traditional methods still provide strong baselines".

That is your paper's shape exactly: the sophisticated approach does not beat the simple baseline,
reported as a benchmark rather than as a method paper. It was accepted with a single-institution
dataset, a two-sentence external-validity limitation, and "±N/A" (single run) on two of its four
arms.

The second-closest match is the post-mortem:

> **"When the Domain Expert Has No Time and the LLM Developer Has No Clinical Expertise: Real-World
> Lessons from LLM Co-Design in a Safety-Net Hospital"** — arXiv 2508.08504, accepted poster, **7
> body pages** + 14 appendix pages. This is a "what went wrong and what we learned" paper:
> "we found that **the most critical challenge in this setting is the careful and precise
> specification of what information to surface**", built around a real-world case study at
> Zuckerberg San Francisco General Hospital. Single deterministic LLM pass throughout.

Other accepted papers in the negative / critical / brittleness family (all from
https://neurips.cc/virtual/2025/workshop/109566 ):

- **"Pandemic-Potential Viruses are a Blind Spot for Frontier Open-Source LLMs"** — accepted as an
  **oral**. A pure negative capability finding.
- "Brittleness and Promise: Knowledge Graph-Based Reward Modeling for Diagnostic Reasoning" (#8) —
  "Experiments ... reveal both promise and brittleness: while specific reward optimization and
  distillation lead to strong path-judging performance, **the transferability to downstream tasks
  remain weak**."
- "Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs" (#5) —
  "accuracy dropping from 91.2% to as low as 13.5%"; "**Counterintuitively**, indirect, context-based
  interventions are often more harmful than direct suggestions".
- "Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions" (#6).
- "Faithful or Just Plausible? Evaluating Faithfulness for Medical Reasoning in Closed-Source LLMs".
- "Reliable or Risky? Assessing Diffusion Models for Biomedical Data Generation".
- "Demo: Statistically Significant Results on Biases and Errors of LLMs Do Not Guarantee
  Generalizable Results" — arXiv 2511.02246, 5 body pages. A methodological cautionary paper whose
  entire thesis is that a positive-looking result does not generalise.
- "Examining the Vulnerability of Multi-Agent Medical Systems to Human Interventions for Clinical
  Reasoning".
- "Multi-Turn LLM Systems for Diagnostic Decision-Making: Considerations, Biases, and Challenges".
- "Beyond Overall Accuracy: A Psychometric Deep Dive into the Topic-Specific Medical Capabilities of
  80 Large Language Models".
- "MedVLThinker: **Simple Baselines** for Multimodal Medical Reasoning" (#17).

By my count at least 11 of the ~72 accepted research-track papers — roughly one in six — are
critical evaluations, brittleness studies, negative findings, or "the simple thing still wins"
benchmarks, and one of them took an oral slot. MEASURED (titles and abstracts); INFERRED (the
one-in-six proportion, since I classified from titles and abstracts, not full texts, for the ones I
did not download).

**INFERRED positioning advice.** The accepted negative-result papers all frame the finding as a
*claim about the field*, not as a report of a failed project. "Count-Based Approaches Remain Strong"
is a title that asserts something; it is not "We Tried X and It Didn't Work". Retitle and reframe
along the line of "segmentation-free masking remains a strong baseline for 3D OCT JEPA
pretraining", make the negative result the headline claim, and move the implementation post-mortem
into a clearly-labelled lessons section or appendix. Note also that the 2026 Research-track criteria
explicitly list "novelty of the method, benchmark, dataset, **or findings**" — a novel *finding*
counts. MEASURED (https://genai4health.github.io/2026-NeurIPS/ , "What reviewers will assess").

---

## 5. SUPPLEMENTARY ARTIFACTS AND THE DOUBLE-BLIND PROBLEM

### What the workshop states

**2026 (governs you) — MEASURED**, https://genai4health.github.io/2026-NeurIPS/ :
> "Supplementary materials (code, data, videos) may be submitted as appendices."
> "**Double-blind review:** Submissions must be fully anonymized: no author names, affiliations,
> identifying acknowledgments, or **non-anonymized links (e.g., a GitHub repository revealing
> authorship)**. Use `\usepackage{neurips_2026}` without options to keep the submission anonymous.
> **Violations may lead to desk rejection.**"

**2025 — MEASURED**, from the 2025 site bundle:
> "Supplementary materials (code, data, videos) may be submitted as appendices"
> "**All submissions must be anonymized and may not contain any identifying information**"
> "**This policy applies to any supplementary or linked material as well, including code**"
> "Please do not include acknowledgments at submission time. Any papers found to be violating this
> policy will be rejected"

**2025 camera-ready instructions — MEASURED**, same bundle:
> "**LaTeX Format:** Format your paper using the NeurIPS 2025 LaTeX package with the following
> workshop options: `\usepackage[dblblindworkshop,final]{neurips_2025}`"
> "**Workshop Title:** Include the workshop title in your paper by adding:
> `\workshoptitle{The Second Workshop on GenAI for Health: Potential, Trust, and Policy Compliance}`"
> "**Public Availability:** Your camera-ready paper will be made publicly available on OpenReview
> after October 31, 2025, unless you contact the organizers before the deadline to opt out"

So the venue runs a two-stage identity model: anonymous at submission (`neurips_20XX` with no
options), de-anonymised at camera-ready (the `final` option). INFERRED from the two MEASURED strings.

### What accepted papers actually did

**MEASURED.** Every accepted-paper PDF I parsed is the *camera-ready or post-acceptance* version, and
those carry fully de-anonymised, author-owned artifact links directly in the body:

- Count-Based Approaches Remain Strong: "The source code is available at:
  `https://github.com/cristea-lab/Structured_EHR_Benchmark`" (in the abstract).
- MedVAL: `github.com/StanfordMIMI/MedVAL`, `huggingface.co/datasets/stanfordmimi/MedVAL-Bench`,
  `huggingface.co/stanfordmimi/MedVAL-4B` — code, dataset **and model weights**.
- Shallow Robustness: "Dataset and code available on HuggingFace and GitHub", with
  `huggingface.co/datasets/dynamoai-ml/MedQA-USMLE-4-MultiTurnRobust` and
  `github.com/bmanczak/MedQA-MultiTurnRobustness` in the arXiv comment.
- MedBrowseComp: `github.com/shan23chen/MedBrowseComp`,
  `huggingface.co/datasets/AIM-Harvard/MedBrowseComp`.
- Traj-CoA: `github.com/zengsihang/Traj-CoA`.
- LLMs as Medical Codes Selectors: `github.com/almeidava93/llm-as-code-selectors-paper` — and its
  arXiv comment even names the OpenReview forum: "Accepted at NeurIPS 2025 as a poster presentation
  in The Second Workshop on GenAI for Health ... (https://openreview.net/forum?id=Kl7KZwJFEG)".
- MedGUIDE: `huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K`.
- ICARE: `huggingface.co/IAMJB/...`, `github.com/rajpurkarlab/CXR-Report-Metric`.
- When the Domain Expert Has No Time: `github.com/jjfenglab/social-wayfinder`.

So **artifacts clearly matter at this venue and are common in accepted work** — roughly half the
papers I parsed link code, data or weights. MEASURED.

**What I could NOT verify:** whether the *submitted* (anonymous) versions used
`anonymous.4open.science` mirrors, anonymised Hugging Face accounts, or simply omitted links until
camera-ready. I found zero `anonymous.4open.science` links across all 20 PDFs I parsed — but every
one of those PDFs is a camera-ready, so that observation says nothing about the submission versions,
and the submission versions are only visible inside OpenReview, which I cannot reach. This is a
genuine gap, marked as such.

### Recommended practice for your submission — INFERRED, but tightly constrained by the MEASURED rules

Your specific risk: public GitHub repo `yfeng0206/I-JEPA_3D_OCT` and Hugging Face checkpoints. The
2026 call names "a GitHub repository revealing authorship" as a desk-rejection example, and the 2025
call extends the anonymity rule explicitly to "any supplementary or linked material as well,
including code". Concretely:

1. **Do not put `github.com/yfeng0206/...` or an author-identifying Hugging Face org/user path
   anywhere in the submission PDF, including the appendix.** The username alone reveals authorship.
   This is the direct desk-reject trigger named in the 2026 call.
2. **Safest option, and the one the workshop's own text supports: submit the artifact as an
   appendix, not as a link.** "Supplementary materials (code, data, videos) may be submitted as
   appendices" — an anonymised code bundle or a config/hyperparameter appendix carries the
   reproducibility signal with zero anonymity risk. The appendix is uncapped and accepted papers use
   12+ appendix pages routinely.
3. **If you want a live link, use an anonymised mirror** (`anonymous.4open.science` is the standard
   for NeurIPS-family double-blind venues) and write "code available at an anonymised repository;
   de-anonymised at camera-ready". I could not verify that any 2025 accepted paper did this, so
   treat it as standard-practice reasoning rather than a venue-observed norm.
4. **Do not cite your own arXiv preprint or repo in a way that identifies you**, and per the 2025
   rule, **omit acknowledgments entirely at submission** — that rule is stated with "will be
   rejected", not "may".
5. **De-anonymise at camera-ready.** The 2025 instructions show the exact mechanism
   (`\usepackage[dblblindworkshop,final]{neurips_2025}` plus `\workshoptitle{...}`); expect the 2026
   equivalent. That is the point at which you add the real GitHub and Hugging Face links, as the
   accepted 2025 papers all did.
6. A preprint on arXiv is *not* a problem: "Non-archival preprints (e.g., arXiv) are fine and do not
   count as prior publication" (2026 call, Publication Policy). MEASURED. But a preprint under your
   real names that is trivially findable from the paper's title is in tension with double-blind;
   NeurIPS-family venues generally tolerate this, and the workshop says nothing prohibiting it. The
   thing the workshop *does* prohibit is an identifying link inside the PDF.

---

## 6. OTHER FACTS WORTH HAVING

MEASURED, https://genai4health.github.io/2026-NeurIPS/ unless noted:

- **Dates 2026:** submission Sep 5, 2026; notification **Sep 29, 2026**; camera-ready TBA; workshop
  in **Sydney, Australia, December 11, 2026**. All deadlines 11:59 PM AoE.
- **24 days from deadline to notification, at least two reviews, no rebuttal.** Reviewers are under
  time pressure; the paper must be self-contained and must pre-empt objections in its limitations.
- **Three tracks in 2026** (Frontier Models for Health; Trustworthy AI/Policy/Human-AI Collaboration;
  Toward 360° AI Care). These are *topic areas*; Research/Demonstration/Position are the *submission
  tracks*, and any track may address any topic.
- **Format:** NeurIPS 2026 LaTeX style, `\usepackage{neurips_2026}` with no options. **The NeurIPS
  Paper Checklist is not required.**
- **Non-archival.** Accepted papers are not proceedings publications; you may submit the same or
  extended work to an archival venue afterwards. Previously *published* work is ineligible; work
  under review elsewhere is allowed if the other venue permits it.
- **Authorship is frozen at the deadline.** "No authors may be added after the submission deadline."
  Reordering is allowed for accepted papers; removal needs written consent from all authors.
- **All accepted papers get posters**; orals/spotlights are selected from them; **three Outstanding
  Paper Awards, one per track** — so the Research track has its own award and you are not competing
  against demos and position papers for it.
- 2025 organizers included Ying Ding (UT Austin), Pranav Rajpurkar (Harvard), Fei-Fei Li, Ehsan
  Adeli, Xiaoxiao Li, Junyuan Hong, Tiange Xiang, Jiawei Xu, Changan Chen, Georgios Pavlakos, Zakia
  Hammal, Scott Delp. Source: 2025 site bundle. A largely overlapping committee is likely in 2026.
- 2025 program was heavily LLM/agent-centric. Self-supervised representation learning on 3D medical
  imaging was thinly represented in the accepted set — I saw no JEPA/masked-image-modelling
  pretraining paper among the 99 titles. INFERRED implication: you are unusual for this venue, which
  cuts both ways. Lead with the health framing (OCT, clinical downstream task) rather than the
  self-supervised-learning framing, and make sure the "why this matters for health" paragraph is in
  the first page — the 2026 criteria list "relevance and potential impact for health" as one of four
  assessment axes.

---

## 7. GAPS AND UNCERTAINTIES

1. **Reviews and score distributions: not obtained.** openreview.net is behind Cloudflare Turnstile
   from this environment; all of `openreview.net`, `api.openreview.net`, `api2.openreview.net`,
   `openreview.net/pdf?id=...` and `openreview.net/attachment?id=...` returned HTTP 403
   `ChallengeRequiredError`. No Wayback snapshots exist for any GenAI4Health forum page. **Follow-up:
   log into OpenReview in a browser and open https://openreview.net/forum?id=fKJtKew2YQ (Count-Based
   Approaches Remain Strong) — if an "Official Review" note is visible, reviews are public and the
   whole score/criticism question becomes answerable in an hour.**
2. **Acceptance rate: not published.** The number of submissions is not stated on either workshop
   site, on the NeurIPS virtual page, or on aiworkshoptracker. Only the accepted count (99) is known.
3. **Whether accepted 2025 submissions used anonymised artifact links.** Not verifiable without
   OpenReview access, because only camera-ready versions are on arXiv. My recommendation in section 5
   is derived from the workshop's stated rules, not from observed 2025 submission behaviour.
4. **Page counts are measured from arXiv PDFs, not the OpenReview PDFs.** For the papers whose arXiv
   comment names the workshop (H-DDx, Robust or Suggestible?, Shallow Robustness, LLMs as Medical
   Codes Selectors, Editing with AI) the arXiv version is stated or evidently the workshop version.
   For the rest I inferred equivalence from the tight 9-page clustering. Two papers (MedVAL, ICARE)
   are demonstrably *extended* arXiv versions and were excluded from the length statistics.
5. **Seed reporting was detected by regex over extracted PDF text** ("seed", "±", "standard
   deviation", "averaged over N runs", etc.). A paper could in principle report variance in a way my
   patterns missed, e.g. only inside a figure image. The three explicit single-run statements
   (papers #4, #5, #7) were read in full context and quoted verbatim, so those are solid.
6. **Track assignment for 2025 posters** was inferred from the mandatory "Demo:"/"Position:" title
   prefixes stated in the 2025 call. A research paper that ignored the prefix rule would be
   misclassified by me; a demo or position paper that followed it would not.
7. **11 of the ~72 research-track papers were classified as negative/critical from title and
   abstract only**; 8 of those I read in full. The "roughly one in six" figure is therefore an
   estimate.
8. Several accepted papers have no arXiv version and could not be examined at all, including
   "FairGRPO", "Unanchoring the Mind", "Pandemic-Potential Viruses are a Blind Spot for Frontier
   Open-Source LLMs", "Reliable or Risky?", "Faithful or Just Plausible?", "Beyond Overall
   Accuracy", and "Examining the Vulnerability of Multi-Agent Medical Systems". Their PDFs are on
   OpenReview only.

---

## Sources

- https://genai4health.github.io/2026-NeurIPS/ — 2026 call for papers (page limits, tracks, review
  process, anonymity, publication policy, dates).
- https://genai4health.github.io/2025-NeurIPS/ and its bundle
  https://genai4health.github.io/2025-NeurIPS/static/js/main.897dc88a.js — 2025 call for papers
  (page limits, title prefixes, review process, camera-ready LaTeX options, organizers).
- https://neurips.cc/virtual/2025/workshop/109566 — official NeurIPS listing of the 2025 workshop
  program: 9 orals, 89 accepted posters, OpenReview forum ids, full schedule.
- https://aiworkshoptracker.com/workshop/neurips-2025-genai4health/ — third-party mirror, "Accepted
  papers (99) — Fetched from OpenReview (v2) on 2026-06-10".
- arXiv PDFs downloaded and parsed with pypdf: 2508.18541, 2507.03152, 2510.03700, 2510.05492,
  2511.00782, 2510.12255, 2508.02669, 2505.14963, 2509.14275, 2509.07325, 2509.07260, 2510.10454,
  2510.13931, 2508.08504, 2509.18316, 2511.02246, 2511.19940, 2507.14681, 2508.02808, 2505.11613.
- https://openreview.net/group?id=NeurIPS.cc/2025/Workshop/GenAI4Health — the venue group exists
  (confirmed via the OpenReview groups API before the challenge wall went up) but its contents were
  not readable from this environment.
