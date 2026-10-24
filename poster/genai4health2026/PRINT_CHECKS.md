# Poster print checks - October 24, 2026

Print `poster_print.pdf` at actual size (100%): one portrait page, **610 x 850 mm**.
The PNG is a preview, not the print master. The existing author block is unchanged;
the author-approved acknowledgment to Prof. Kaveri A. Thakoor is included.

| Check | Result |
|---|---|
| Build | `build.ps1 -OnlyCached -Dpi 100`, exit 0; `Checks OK` |
| Results | Real post-freeze bundle; 148 numeric and 20 text macros validated |
| Displayed result cells | All 21 slots, labels and late/incomplete marks match the bundle |
| Original-run claims | All six KeyBox/Answer checks agree with the paper macros |
| Reporting | Only index 1234 complete on time; 5678 descriptive; both guided labels incomplete; control descriptive, no registered RANDOM range |
| Figures | Registered poster variants and generated plot hashes checked |
| Text fit | No overfull/underfull boxes; 35.09 design points spare above the bottom margin |
| Placeholders | None; no synthetic results |
| Fonts | All 16 embedded |
| QR | Decoded from the rendered PDF to the public paper folder on `poster-ready` |
| Visual review | Full poster and enlarged results/credits inspected; no clipping or overlap |

The QR landing page includes the current paper **review** PDF with real results,
Future work and the acknowledgment. The historical v9 PDF is separately labeled.
Final manuscript integration, receipt/release checks, funding disclosure and author
draft review remain open; these poster print checks do not certify that release.

Rebuilding the real-result poster requires the verified generated `auto/cr_*`
bundle described in the README. It is retained locally pending final paper integration.
Before the bundle exists, the source intentionally displays result placeholders.

Downloads copies: `POSTER_print_ready_61x85cm.pdf`, matching `.png`,
`.checks.json`, `POSTER_PRINT_INSTRUCTIONS.txt`, and
`POSTER_print_package_61x85cm.zip`. Regenerate these if the wording changes.
