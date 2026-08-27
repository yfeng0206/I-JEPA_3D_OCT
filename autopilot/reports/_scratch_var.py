import io, re, time, urllib.request
from pypdf import PdfReader

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0 Safari/537.36"}

P = [
 ("2508.18541","Uncovering Intervention Opportunities (ORAL)"),
 ("2510.03700","H-DDx (ORAL)"),
 ("2510.05492","High-Fidelity Synthetic ECG (ORAL)"),
 ("2511.00782","Count-Based Approaches Remain Strong"),
 ("2510.12255","Shallow Robustness Deep Vulnerabilities"),
 ("2510.13931","Robust or Suggestible?"),
 ("2508.08504","When the Domain Expert Has No Time"),
 ("2509.18316","Brittleness and Promise"),
 ("2509.14275","FedMentor"),
 ("2509.07325","CancerGUIDE"),
 ("2510.10454","Traj-CoA"),
 ("2505.11613","MedGUIDE"),
 ("2511.19940","Editing with AI"),
 ("2507.14681","LLMs as Medical Codes Selectors"),
 ("2509.07260","HealthSLM-Bench"),
 ("2505.14963","MedBrowseComp"),
]
VOCAB = {
 "pm_symbol": r"±",
 "averaged_over_runs": r"averaged over (\w+ )?(runs|seeds|repetitions)|mean (of|over) \w+ runs|across \w+ runs",
 "repeat_runs": r"repeated (the )?(experiments?|runs?)|three runs|five runs|3 runs|5 runs|multiple runs",
 "single_run_admit": r"single run|one run|a single (training )?run|single seed|one seed|due to (compute|resource)",
 "limitation_sec": r"\bLimitations?\b",
}
for aid,label in P:
    try:
        d=urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/pdf/{aid}",headers=UA),timeout=120).read()
        r=PdfReader(io.BytesIO(d))
    except Exception as e:
        print(f"{label}: FAIL {e}"); continue
    pages=[]
    for p in r.pages:
        try: pages.append(p.extract_text() or "")
        except Exception: pages.append("")
    full="\n".join(pages)
    refp=None
    for i,t in enumerate(pages):
        if i>1 and re.search(r"\bReferences\b",t): refp=i+1; break
    out=[]
    for k,pat in VOCAB.items():
        m=re.findall(pat,full,re.I)
        if m: out.append(f"{k}={len(m)}")
    print(f"{label} [{aid}]: total={len(r.pages)} body={refp-1 if refp else '?'} appendix~{len(r.pages)-(refp or 0)} | " + (", ".join(out) if out else "no-variance-vocab"))
    time.sleep(2)
