# GenAI4Health @ NeurIPS 2026 — Submission-Requirements Audit

**Audit date:** 2026-08-22  
**Submission type audited:** Research Paper  
**Primary workshop source:** https://genai4health.github.io/2026-NeurIPS/  
**Primary submission source:** https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/GenAI4Health

## Summary table

| # | Requirement | Answer |
|---:|---|---|
| 1 | NeurIPS Paper Checklist | **NOT REQUIRED.** The workshop explicitly waives it, notwithstanding the generic main-track checklist bundled in the official ZIP. |
| 2 | Page limit | **Maximum 9 content pages** for a Research Paper. Acknowledgments, references, and appendix are excluded. |
| 3 | Deadline | **September 5, 2026, 11:59 PM AoE (UTC−12)**; notification **September 29, 2026, 11:59 PM AoE**. |
| 4 | Venue/mechanics | **OpenReview.** Research venue group: `NeurIPS.cc/2026/Workshop/GenAI4Health`; submission invitation is live. No separate abstract deadline is published or configured. |
| 5 | Anonymity | **Double-blind.** Author-identifying GitHub, HuggingFace, model, or data links are forbidden in the review PDF. |
| 6 | Archival/dual submission | **Non-archival.** arXiv preprints and concurrent submissions are allowed, subject to the other venue’s policy; previously published archival work is ineligible. |
| 7 | Broader impact/ethics statement | **No standalone section is required.** Any such discussion in the main paper counts toward the 9 pages; appendix material is excluded. Ethics compliance may still be required. |
| 8 | Style option | Use exactly **`\usepackage{neurips_2026}` with no options**. Do not use `dblblindworkshop`. |
| 9 | Supplementary material | Appendices are allowed and excluded from the page limit. The current Research Paper OpenReview form accepts **one PDF, maximum 50 MB**, and has no separate supplement/ZIP field. |
| 10 | Authors | No authors may be added after the deadline. Reordering is allowed after acceptance; removal requires written consent. **UNRESOLVED:** no numerical author-count limit is published. |

---

## Source precedence

There is an apparent checklist conflict:

- The **generic NeurIPS main-track template** bundled in the official ZIP says the checklist is mandatory.
- The **workshop-specific CFP** explicitly says it is not required.

The workshop-specific instruction is direct and unambiguous, so it controls this workshop submission. The generic main-track ZIP explains why `checklist.tex` is present but does not make it mandatory for GenAI4Health.

---

## 1. Is the NeurIPS Paper Checklist required?

## Finding: **No — explicitly not required**

**Workshop source:**  
https://genai4health.github.io/2026-NeurIPS/

> “You must format your submission using the NeurIPS 2026 LaTeX style file. **The NeurIPS Paper Checklist is not required.**”

This is the decisive workshop-specific instruction.

### OpenReview cross-check

**Research venue page:**  
https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/GenAI4Health

**Research submission invitation:**  
https://openreview.net/invitation?id=NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission

**Submission-schema API:**  
https://api2.openreview.net/invitations?id=NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission

The current submission form requires title, authors, keywords, abstract, PDF, and two confirmations. It contains no checklist field and no instruction requiring checklist content in the PDF.

### Why the official ZIP contains `checklist.tex`

**Generic NeurIPS author ZIP:**  
https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip

The bundled `checklist.tex` says:

> “Do not remove the checklist: **The papers not including the checklist will be desk rejected.** The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.”

That is the generic **main-track** instruction. GenAI4Health explicitly overrides it for this workshop.

### Placement and page counting

Because the checklist is **not required**, there is no applicable workshop requirement for where it must appear or whether it is excluded.

The workshop’s own exclusion sentence names only:

> “All page limits exclude acknowledgments, references, and appendix.”

It does **not** name a voluntarily included checklist. Therefore, do not assume that a voluntarily retained checklist receives special page-limit treatment under workshop rules.

### Recommendation

Remove or comment out:

```latex
\input{checklist.tex}
```

Do not include the NeurIPS Paper Checklist in the workshop review PDF.

---

## 2. Exact page limit and exclusions

## Finding: **Maximum 9 content pages**

**Workshop source:**  
https://genai4health.github.io/2026-NeurIPS/

The Research Paper track is labeled:

> “Research Papers”

and:

> “Up to 9 pages”

The format section states:

> “All page limits exclude acknowledgments, references, and appendix.”

Therefore:

| Material | Counts toward 9 pages? |
|---|---:|
| Title, abstract, main text | Yes |
| Figures and tables in main text | Yes |
| Limitations/ethics discussion in main text | Yes |
| Acknowledgments | No, but identifying acknowledgments are forbidden during review |
| References | No |
| Appendix | No |
| NeurIPS Paper Checklist | Not required; no workshop exclusion rule needed |

### Important anonymity interaction

The same workshop page says:

> “Submissions must be fully anonymized: no author names, affiliations, identifying acknowledgments, or non-anonymized links…”

Thus, although acknowledgments are excluded from the page limit, identifying acknowledgments must not appear in the review PDF. The safest review submission is to omit the acknowledgments section entirely.

### Scope warning

“Up to 9 pages” is a maximum, not a target. Critical evidence supporting the central claim should remain in the main nine pages rather than being moved into the appendix.

The generic NeurIPS template describes appendices as optional reading:

> “Think of the appendix as ‘optional reading’ for reviewers. The paper must be able to stand alone without the appendix; for example, adding critical experiments that support the main claims to an appendix is inappropriate.”

**Source:**  
https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip  
Archive member: `neurips_2026.tex`

---

## 3. Deadline and notification date

## Finding

- **Submission:** September 5, 2026, at 11:59 PM AoE
- **Notification:** September 29, 2026, at 11:59 PM AoE
- **Camera-ready:** TBA

**Workshop source:**  
https://genai4health.github.io/2026-NeurIPS/

> “Paper Submission Deadline — **Sep 5, 2026**”

> “Acceptance Notification — **Sep 29, 2026**”

> “Camera-ready Submission — **TBA**”

> “**All deadlines are at 11:59 PM AoE (Anywhere on Earth).**”

AoE means UTC−12. Therefore:

- Submission deadline: **2026-09-05 23:59 UTC−12**
- Equivalent UTC time: **2026-09-06 11:59 UTC**
- Notification deadline/date-time: **2026-09-29 23:59 UTC−12**
- Equivalent UTC time: **2026-09-30 11:59 UTC**

### OpenReview confirmation

The Research submission invitation has:

```json
"duedate": 1788695940000
```

That converts to:

```text
2026-09-06T11:59:00Z
```

which is exactly September 5 at 11:59 PM AoE.

**API source:**  
https://api2.openreview.net/invitations?id=NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission

As of 2026-08-22, the deadline is approximately 14 calendar days away.

---

## 4. Submission venue and mechanics

## Finding: **OpenReview; Research Paper portal is live**

**Workshop source:**  
https://genai4health.github.io/2026-NeurIPS/

> “Choose the appropriate OpenReview submission track for your paper.”

> “**Platform:** All submissions are made through OpenReview…”

The visible prose still says:

> “portal link to be posted on this page”

but the page’s Research Paper card now contains a live OpenReview link. That sentence is stale.

### Exact Research Paper identifiers

**Venue group/domain ID:**

```text
NeurIPS.cc/2026/Workshop/GenAI4Health
```

**Venue page:**

https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/GenAI4Health

**Submission invitation ID:**

```text
NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission
```

**Direct submission invitation URL:**

https://openreview.net/invitation?id=NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission

**Submission venue ID assigned to papers:**

```text
NeurIPS.cc/2026/Workshop/GenAI4Health/Submission
```

The OpenReview group API states:

```json
"submission_id": {
  "value": "NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission"
},
"submission_venue_id": {
  "value": "NeurIPS.cc/2026/Workshop/GenAI4Health/Submission"
}
```

**API source:**  
https://api2.openreview.net/groups?id=NeurIPS.cc/2026/Workshop/GenAI4Health

### Separate abstract deadline

There is **no separately published abstract deadline**.

The current OpenReview invitation places the title, authors, keywords, abstract, and PDF in one submission form governed by the same due date. The schema describes:

> “Abstract of paper. Add TeX formulas using the following formats…”

and:

> “Upload a PDF file that ends with .pdf.”

No separate abstract-registration invitation or abstract due date is present in the current Research Paper form.

Therefore, submit the abstract and full PDF together by September 5, 11:59 PM AoE.

### Current form requirements

The Research Paper form currently requires:

- Title: 1–250 characters
- All authors selected through OpenReview profiles
- Keywords
- Abstract: maximum 5,000 characters
- PDF: maximum 50 MB
- Confirmation that author emails may be shared with program chairs
- Confirmation concerning public release if accepted
- CC BY 4.0 as the configured license
- Optional TL;DR: maximum 250 characters

The author field says:

> “All authors must have an OpenReview profile prior to submitting a paper.”

**Source:**  
https://api2.openreview.net/invitations?id=NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission

### Additional unresolved publication-form inconsistency

The workshop CFP says:

> “Accepted workshop papers … will be made publicly available by default. However, authors may choose to opt out…”

But the current OpenReview form requires confirmation:

> “We authorize the release of our submission and author names to the public in the event of acceptance.”

The relationship between that mandatory form confirmation and the later opt-out is **UNRESOLVED**. If the authors intend to opt out of public posting, confirm the procedure with the organizers before submission.

The OpenReview group lists the contact as:

```text
genai4health@googlegroups.com
```

---

## 5. Anonymity and public model/code links

## Finding: **Strict double-blind; identifying links are forbidden**

**Workshop source:**  
https://genai4health.github.io/2026-NeurIPS/

> “**Double-blind review:** Submissions must be fully anonymized: no author names, affiliations, identifying acknowledgments, or non-anonymized links (e.g., a GitHub repository revealing authorship).”

> “Use `\usepackage{neurips_2026}` without options to keep the submission anonymous.”

> “Violations may lead to desk rejection.”

This prohibition applies equally to an identifying:

- GitHub repository
- HuggingFace repository/model card
- project website
- dataset page
- public weights URL
- DOI or artifact page that exposes the authors
- redirect whose destination exposes the authors
- code archive containing names, email addresses, institutional paths, or revealing metadata

### Direct answer about your weights

A public GitHub or HuggingFace link that reveals the authors is **not compliant** in the review submission.

### Compliant alternatives

Use one of these approaches:

1. **Truly anonymized artifact link**
   - The account, organization, repository name, README, model card, commit history, filenames, package metadata, checkpoint metadata, and linked services must not expose the authors.
   - Test the link in a logged-out/private browser.
   - Ensure redirects and downloadable-file metadata do not reveal identities.
   - Do not merely rename the visible link while redirecting to an identifying account.

2. **Release-on-acceptance statement**
   - If a genuinely anonymous artifact cannot be provided, write a neutral statement such as:
     > “Code and pretrained weights will be released upon acceptance.”
   - Include sufficient methodological and implementation information for review without relying on the unavailable artifact.

3. **Anonymous appendix description**
   - Describe the artifact, expected file structure, license plan, and reproducibility instructions in the appendix without linking to an identifying host.

### Existing public artifacts

If the weights are already publicly associated with the authors, do not link to them from the anonymous workshop PDF. If they constitute prior work that must be cited, cite the associated scholarly work in the third person, consistently with double-blind rules, rather than writing “our previous model” or linking to an author-identifying repository.

### Additional anonymity checks

Before uploading:

- Remove author names and affiliations from the PDF.
- Remove identifying acknowledgments.
- Inspect PDF metadata for author/organization fields.
- Check repository links, QR codes, shortened links, DOI destinations, and supplemental URLs.
- Write self-citations in the third person.
- Ensure appendix links are anonymized too.
- Ensure filenames and screenshots do not contain usernames, organizations, workstation paths, or account avatars.

---

## 6. Dual submission, archival status, and arXiv

## Finding: **Non-archival; concurrent review and arXiv are allowed conditionally**

### Non-archival status

**Workshop source:**  
https://genai4health.github.io/2026-NeurIPS/

> “GenAI4Health is a non-archival workshop. Accepted papers will not be published in any proceedings and will not be considered a formal (archival) publication.”

The format section also states:

> “The accepted papers will be non-archival (NOT included in proceedings or any form of publication).”

### Later submission to another venue

> “Because acceptance here is non-archival, presenting at GenAI4Health does not preclude subsequent submission of the same or extended work to archival conferences … or journals.”

### Concurrent submission elsewhere

> “**Work under review elsewhere may be submitted here,** provided this does not violate the other venue’s dual-submission or anonymity policy.”

Thus, GenAI4Health permits concurrent review, but the other venue may prohibit it. Both policies must permit the overlap.

### arXiv and other preprints

> “Non-archival preprints (e.g., arXiv) are fine and do not count as prior publication.”

Therefore, an arXiv preprint is permitted by this workshop.

However, the submitted workshop PDF still must be anonymous and must not link to author-identifying material. The existence of a public preprint does not authorize authors to identify themselves inside the anonymous submission.

### Previously published work

> “Papers that have already appeared in an archival venue (conference proceedings or journal) at the time of submission are not eligible.”

Also:

> “Papers submitted to the workshop **must not** have been previously published at another venue at the time of submission.”

Before submission, verify that the work has not already appeared in archival proceedings or a journal.

---

## 7. Broader-impact and ethics statement

## Finding: **No standalone section is required**

The workshop CFP does not require a section titled “Broader Impact,” “Societal Impact,” or “Ethics Statement.” It also explicitly waives the checklist containing the main-track broader-impact and ethics questions.

The official NeurIPS handbook states:

> “Please note that you are not required to include a section titled ‘broader impacts’ in your paper.”

It continues:

> “However, you should still consider any potential negative societal impacts of your work.”

and:

> “You may include a discussion of these potential negative societal impacts anywhere in the paper … but this discussion may not exceed the page limit.”

**Source:**  
https://neurips.cc/Conferences/2026/MainTrackHandbook

The workshop itself advises authors to anticipate concerns:

> “Please ensure your submission is self-contained and anticipates likely questions (e.g., in a limitations section).”

**Source:**  
https://genai4health.github.io/2026-NeurIPS/

### Page-limit effect

The only workshop exclusions are:

> “acknowledgments, references, and appendix.”

Therefore:

- A broader-impact, ethics, or limitations section in the main paper counts toward the nine pages.
- Material placed in the appendix is excluded under the workshop’s appendix rule.
- Critical ethical limitations relevant to interpreting the claims should not be hidden solely in optional appendix material.

### Ethics compliance versus an ethics statement

Not requiring a titled statement does not waive ethical obligations. The NeurIPS handbook states, for human-participant research:

> “If the research presented involves direct interactions between the researchers and human participants or between a technical system and human participants, authors are required to follow existing protocols in their institutions (e.g. human subject research accreditation, IRB) and go through the relevant process.”

**Source:**  
https://neurips.cc/Conferences/2026/MainTrackHandbook

For a health/medical-imaging paper, verify as applicable:

- IRB/ethics approval or exemption
- informed-consent basis or waiver
- lawful dataset access and permitted reuse
- privacy/de-identification safeguards
- demographic and sampling limitations
- intended-use and misuse risks
- clinical-validation limitations
- whether the work is research-only and not approved for clinical use

### Recommendation

A concise `Limitations and Ethical Considerations` section is prudent for health research, even though a standalone section is not formally required. Keep it within the nine-page main-paper limit if it affects interpretation of the claims.

---

## 8. Required LaTeX style option

## Finding: **No options**

**Workshop source:**  
https://genai4health.github.io/2026-NeurIPS/

> “Use `\usepackage{neurips_2026}` without options to ensure the submission is anonymous.”

The review-process section repeats:

> “Use `\usepackage{neurips_2026}` without options to keep the submission anonymous.”

Therefore, use:

```latex
\usepackage{neurips_2026}
```

Do **not** use:

```latex
\usepackage[dblblindworkshop]{neurips_2026}
```

Do not use `sglblindworkshop`, `final`, `preprint`, `nonanonymous`, or any other track option for this review submission.

Although the style file declares `dblblindworkshop`, GenAI4Health does not instruct authors to use it. The workshop-specific instruction explicitly says **without options**. Using `dblblindworkshop` would change the track/notice behavior and would not follow the published instruction.

The no-option mode is also the mode independently verified as anonymous in the previous style-file audit.

---

## 9. Supplementary material and upload limits

## Finding: **Appendices are allowed; current form has one PDF field and no separate supplement upload**

**Workshop source:**  
https://genai4health.github.io/2026-NeurIPS/

> “Supplementary materials (code, data, videos) may be submitted as appendices.”

The same page says:

> “All page limits exclude acknowledgments, references, and appendix.”

Thus, an appendix may be appended to the paper PDF and does not count toward the nine content pages.

### Current OpenReview form

**Submission API:**  
https://api2.openreview.net/invitations?id=NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission

The PDF field says:

> “Upload a PDF file that ends with .pdf.”

Its schema is:

```json
"pdf": {
  "order": 7,
  "description": "Upload a PDF file that ends with .pdf.",
  "value": {
    "param": {
      "type": "file",
      "maxSize": 50,
      "extensions": ["pdf"]
    }
  }
}
```

Therefore, the current Research Paper form accepts:

- one PDF
- `.pdf` extension only
- maximum size 50 MB

The full current submission schema contains no:

- supplementary PDF field
- ZIP field
- code-upload field
- video-upload field
- data-upload field

### Direct answers

| Question | Current answer |
|---|---|
| Appendix in the main PDF? | Yes |
| Does appendix count toward 9 pages? | No |
| Separate appendix PDF? | No field exists |
| Separate code/data/video ZIP? | No field exists in the current workshop form |
| Main PDF size limit? | 50 MB |
| Separate supplement size limit? | Not applicable because no separate field exists |
| Could a separate field be added later? | **UNRESOLVED**; no such field exists as of 2026-08-22 |

Do not rely on the generic NeurIPS main-track handbook’s separate 100 MB ZIP allowance. That is a main-track mechanism and is not present in this workshop’s current OpenReview form.

For code or model weights, use an anonymized link described in the appendix or state that they will be released upon acceptance.

---

## 10. Number of authors and author changes

## Finding: **No additions after the deadline**

**Workshop source:**  
https://genai4health.github.io/2026-NeurIPS/

> “**No authors may be added after the submission deadline.** This applies to all stages after submission, including review, camera-ready preparation, and any subsequent versions of the paper.”

The page also says:

> “**Author reordering is permitted for accepted papers,** provided all listed authors consent to the change.”

and:

> “**Author removal after submission requires written consent from all authors,** including the author being removed.”

### OpenReview profile requirement

The Research Paper submission form states:

> “All authors must have an OpenReview profile prior to submitting a paper.”

**Source:**  
https://api2.openreview.net/invitations?id=NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission

Therefore:

- Finalize the complete author set before September 5, 11:59 PM AoE.
- Ensure every author has an active OpenReview profile before submission.
- Do not assume an omitted contributor can be added at camera-ready.
- Reordering after acceptance is allowed with consent.
- Removal requires written consent from all authors.

### Numerical author limit

**UNRESOLVED.**

I checked:

1. The GenAI4Health 2026 CFP and authorship-policy text.
2. The Research Paper OpenReview group.
3. The full Research submission invitation/schema.
4. The OpenReview author-field definition.

None publishes a maximum number of authors. The OpenReview schema has an author-list field but no explicit numerical maximum. Absence of a published cap is not proof that no administrative limit could ever be applied.

---

# Additional submission-form facts

The current OpenReview form includes two mandatory confirmations:

> “We authorize the sharing of all author emails with Program Chairs.”

and:

> “We authorize the release of our submission and author names to the public in the event of acceptance.”

It also configures the submission license as:

```text
CC BY 4.0
```

**Source:**  
https://api2.openreview.net/invitations?id=NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission

The form also indicates that submissions are not public during review through the venue configuration:

```json
"public_submissions": {
  "value": false
}
```

**Source:**  
https://api2.openreview.net/groups?id=NeurIPS.cc/2026/Workshop/GenAI4Health

The workshop separately states:

> “Submissions are visible only to assigned reviewers and organizers during review. Rejected submissions will not be made public.”

There is no rebuttal:

> “There is no author rebuttal or response stage.”

**Source:**  
https://genai4health.github.io/2026-NeurIPS/

---

# ACTION ITEMS FOR THE AUTHORS

## Mandatory before submission

1. **Remove the NeurIPS Paper Checklist**
   - Remove/comment out `\input{checklist.tex}`.
   - Do not include checklist questions or answers in the review PDF.

2. **Use the exact required package invocation**
   ```latex
   \usepackage{neurips_2026}
   ```
   - No `dblblindworkshop`.
   - No `final`, `preprint`, or `nonanonymous`.

3. **Keep main content at or below nine pages**
   - Count title, abstract, text, figures, tables, and main-paper limitations/ethics discussion.
   - References and appendix may follow without counting.
   - Do not move claim-critical evidence solely into the appendix.

4. **Remove all identifying material**
   - No names or affiliations.
   - No identifying acknowledgments.
   - No identifying GitHub/HuggingFace/model/data links.
   - Scrub PDF metadata, filenames, screenshots, URLs, checkpoint metadata, and appendix links.
   - Check the final uploaded PDF, not only the LaTeX source.

5. **Handle weights/code anonymously**
   - Use a genuinely anonymized repository or mirror, or
   - state that code and weights will be released upon acceptance.
   - Do not link to an author-identifying public artifact.

6. **Use the Research Paper portal**
   - Venue: `NeurIPS.cc/2026/Workshop/GenAI4Health`
   - Submit at: https://openreview.net/invitation?id=NeurIPS.cc/2026/Workshop/GenAI4Health/-/Submission

7. **Complete author administration now**
   - Confirm the final author set before the deadline.
   - Ensure every author has an active OpenReview profile.
   - Do not plan to add authors after submission.

8. **Submit by the effective deadline**
   - September 5, 2026, 11:59 PM AoE
   - Equivalent: September 6, 2026, 11:59 UTC
   - Do not rely on last-minute OpenReview profile approval or upload time.

9. **Keep the uploaded PDF under 50 MB**
   - Append supplementary textual material to the same PDF.
   - Do not expect a separate PDF or ZIP upload field.

10. **Check prior-publication and concurrent-review eligibility**
    - Confirm the work has not already appeared in an archival venue.
    - If concurrently submitted elsewhere, confirm that the other venue permits it.

## Strongly recommended for a health paper

11. **Include a concise limitations/ethics discussion**
    - Address data provenance, consent/IRB status, privacy, bias, intended use, clinical limitations, and risks as applicable.
    - A titled section is not mandatory, but material necessary to interpret the claims should appear in the main paper.

12. **Verify all data and model permissions**
    - Dataset license and permitted use.
    - Model-weight license and redistribution terms.
    - Patient-data de-identification and institutional approvals.
    - Permission to release derived weights or code.

13. **Check all OpenReview text fields for anonymity**
    - Title, abstract, keywords, and TL;DR are separate from the PDF.
    - Do not include identifying project names or links in those fields.

14. **Monitor the CFP and OpenReview form for changes**
    - Camera-ready deadline is still TBA.
    - A separate supplement field could theoretically be added, but none exists now.

15. **Contact the organizers if public-posting opt-out matters**
    - The CFP permits opt-out after camera-ready, while the current form requires a public-release confirmation.
    - Contact: `genai4health@googlegroups.com`
    - Resolve this before agreeing to the form if the authors intend to opt out.
