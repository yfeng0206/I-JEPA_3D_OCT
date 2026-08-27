import io, re, time, urllib.request
from pypdf import PdfReader

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0 Safari/537.36"}

PAPERS = [
    ("2510.13931", "Robust or Suggestible? Non-Clinical Induction in LLM Drug-Safety"),
    ("2508.08504", "When the Domain Expert Has No Time... Safety-Net Hospital"),
    ("2509.18316", "Brittleness and Promise: KG-Based Reward Modeling"),
    ("2511.02246", "Demo: Statistically Significant Results Do Not Guarantee Generalizable"),
    ("2511.19940", "Editing with AI: How Doctors Refine LLM Answers"),
    ("2507.14681", "LLMs as Medical Codes Selectors (ICPC-2)"),
    ("2508.02808", "Clinically Grounded Agent-based Report Evaluation"),
    ("2505.11613", "MedGUIDE"),
]

for aid, label in PAPERS:
    url = f"https://arxiv.org/pdf/{aid}"
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        r = PdfReader(io.BytesIO(data))
    except Exception as e:
        print(f"=== {label} [{aid}] FAIL: {e}\n")
        continue
    pages = []
    for p in r.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    full = "\n".join(pages)
    refpage = None
    for i, t in enumerate(pages):
        if re.search(r"\bReferences\b", t) and i > 1:
            refpage = i + 1
            break
    print(f"=== {label} [{aid}]")
    print(f"    total_pages={len(r.pages)}  refs_start={refpage}  body_pages={refpage-1 if refpage else 'n/a'}")
    wk = "YES" if re.search(r"Workshop on GenAI for Health|GenAI4Health", full, re.I) else "no"
    print(f"    workshop banner in PDF: {wk}")
    for pat, name in [(r"\bseeds?\b", "seed"), (r"standard deviation|\bstd\b|±", "std"),
                      (r"single run|one run|a single seed", "single-run"),
                      (r"(github\.com|huggingface\.co|anonymous\.4open)/\S+", "link")]:
        m = re.findall(pat, full, re.I)
        if m:
            print(f"    {name}: {len(m)} hits")
    for mm in list(re.finditer(r"seed", full, re.I))[:3]:
        s = max(0, mm.start() - 150)
        print("    SEEDCTX: " + re.sub(r"\s+", " ", full[s:mm.start() + 150]))
    for mm in list(re.finditer(r"(github\.com|huggingface\.co|anonymous\.4open)/\S+", full, re.I))[:4]:
        print("    LINK: " + mm.group(0)[:110])
    # abstract
    a = re.search(r"Abstract(.{200,1400}?)(1\s+Introduction|Introduction)", pages[0] if pages else "", re.S)
    if a:
        print("    ABS: " + re.sub(r"\s+", " ", a.group(1))[:900])
    print()
    time.sleep(2)
