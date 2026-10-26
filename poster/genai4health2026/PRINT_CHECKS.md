# Poster print checks - October 25, 2026

Print `poster_print.pdf` at actual size (100%): one portrait page, **610 x 850 mm**.
The PNG is a preview, not the print master. The existing author block is unchanged;
the author-approved acknowledgment to Prof. Kaveri A. Thakoor is included.

| Check | Result |
|---|---|
| Build | `build.ps1 -OnlyCached -Dpi 100`, exit 0; `Checks OK` |
| Results | Real post-freeze bundle; 148 numeric and 20 text macros validated |
| Displayed result cells | All 21 replication slots, labels and late/incomplete marks match the bundle; 30 original table cells and two key intervals also checked |
| Original-run claims | All six KeyBox/Answer checks agree with the paper macros |
| Reporting | Only index 1234 complete on time; 5678 descriptive; both guided labels incomplete; control descriptive, no registered RANDOM range |
| Figures | New native-resolution OCT structure image and clean pre-batching proposals registered separately; original framework and generated plot hashes checked |
| Units | Differences/intervals in pp; absolute AUC unchanged; seed-spread plot explicitly retains the original AUC scale |
| Text fit | No overfull/underfull boxes; 26.30 design points spare above the bottom margin after QR removal |
| Placeholders | None; no synthetic results |
| Fonts | All 15 embedded |
| QR | Removed with its caption at the author's October 25 request |
| Visual review | Full poster and enlarged results/credits inspected; no clipping or overlap |

The October 25 author PDF was reviewed with tracked edits visible; its clean export
and source sync remain separate manuscript work. The public paper folder includes the
October 24 paper **review** PDF with real results,
Future work and the acknowledgment. The historical v9 PDF is separately labeled.
Final manuscript integration, receipt/release checks, funding disclosure and author
draft review remain open; these poster print checks do not certify that release.

Rebuilding the real-result poster requires the verified generated `auto/cr_*`
bundle described in the README. It is retained locally pending final paper integration.
Before the bundle exists, the source intentionally displays result placeholders.

Downloads keeps `POSTER_print_ready_61x85cm.pdf` beside the latest Overleaf review ZIP.
On the author's cleanup request, the matching PNG, `.checks.json`, print instructions
and separate poster ZIP were moved to `OCT_JEPA_archive_20261024_002752` in Downloads,
along with older project deliverables. Nothing was deleted. Regenerate the poster
and its checks if the wording changes.
