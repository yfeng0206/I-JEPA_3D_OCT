import io, re, sys, time, urllib.request
from pypdf import PdfReader

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0 Safari/537.36"}

PAPERS = [
    ("2508.18541", "Uncovering Intervention Opportunities for Suicide Prevention with LM Assistants (ORAL)"),
    ("2507.03152", "MedVAL: Expert-Level Medical Text Validation (ORAL)"),
    ("2510.03700", "H-DDx: Hierarchical Evaluation Framework for Differential Diagnosis (ORAL)"),
    ("2510.05492", "High-Fidelity Synthetic ECG Generation (ORAL)"),
    ("2511.00782", "Count-Based Approaches Remain Strong (POSTER)"),
    ("2510.12255", "Shallow Robustness, Deep Vulnerabilities (POSTER)"),
    ("2508.02669", "MedVLThinker: Simple Baselines (POSTER)"),
    ("2505.14963", "MedBrowseComp (POSTER)"),
    ("2509.14275", "FedMentor (POSTER)"),
    ("2509.07325", "CancerGUIDE (POSTER)"),
    ("2509.07260", "HealthSLM-Bench (POSTER)"),
    ("2510.10454", "Traj-CoA (POSTER)"),
]

KEYS = {
    "seed": r"\bseed(s)?\b",
    "random seed": r"random seed",
    "3 seeds/5 seeds": r"\b(three|five|3|5|10|ten)\s+(random\s+)?seeds\b",
    "std/stddev": r"standard deviation|std\.?\s?dev|\bstd\b",
    "conf interval": r"confidence interval|95%\s?CI|\bCI\b",
    "error bars": r"error bar",
    "p-value": r"p\s?[<=]\s?0\.0|p-value",
    "bootstrap": r"bootstrap",
    "github": r"github\.com/\S+",
    "huggingface": r"huggingface\.co/\S+",
    "anonymous link": r"anonymous\.4open|anonymous github|anonymized (link|repository)|anon\.4open",
}

for aid, label in PAPERS:
    url = f"https://arxiv.org/pdf/{aid}"
    try:
        req = urllib.request.Request(url, headers=UA)
        data = urllib.request.urlopen(req, timeout=90).read()
    except Exception as e:
        print(f"=== {label} [{aid}] DOWNLOAD FAIL: {e}")
        continue
    try:
        r = PdfReader(io.BytesIO(data))
    except Exception as e:
        print(f"=== {label} [{aid}] PARSE FAIL: {e}")
        continue
    n = len(r.pages)
    pages = []
    for p in r.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    full = "\n".join(pages)
    # locate references start page
    refpage = None
    for i, t in enumerate(pages):
        if re.search(r"^\s*(\d+\s+)?References\s*$", t, re.M | re.I) or re.search(r"\nReferences\n", t):
            refpage = i + 1
            break
    print(f"=== {label} [{aid}]")
    print(f"    total_pages={n}  references_start_page={refpage}  body_pages(approx)={refpage-1 if refpage else 'n/a'}")
    hits = []
    for k, pat in KEYS.items():
        m = re.findall(pat, full, re.I)
        if m:
            hits.append(f"{k}={len(m)}")
    print("    signals: " + (", ".join(hits) if hits else "none"))
    # show seed context
    for mm in list(re.finditer(r"seed", full, re.I))[:4]:
        s = max(0, mm.start() - 130)
        print("    SEEDCTX: " + re.sub(r"\s+", " ", full[s:mm.start() + 130]))
    for mm in list(re.finditer(r"(github\.com|huggingface\.co)/\S+", full, re.I))[:3]:
        print("    LINK: " + mm.group(0)[:100])
    print()
    time.sleep(2)
