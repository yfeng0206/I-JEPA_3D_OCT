import io, re, time, urllib.request
from pypdf import PdfReader

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0 Safari/537.36"}

PAPERS = [
    ("2511.00782", "Count-Based Approaches Remain Strong"),
    ("2510.12255", "Shallow Robustness, Deep Vulnerabilities"),
    ("2510.13931", "Robust or Suggestible?"),
    ("2508.08504", "When the Domain Expert Has No Time"),
    ("2509.18316", "Brittleness and Promise"),
    ("2510.03700", "H-DDx"),
    ("2510.05492", "High-Fidelity Synthetic ECG"),
]

for aid, label in PAPERS:
    data = urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/pdf/{aid}", headers=UA), timeout=120).read()
    r = PdfReader(io.BytesIO(data))
    p1 = r.pages[0].extract_text() or ""
    p2 = r.pages[1].extract_text() if len(r.pages) > 1 else ""
    print("=" * 100)
    print(f"{label} [{aid}]")
    print(re.sub(r"\s+", " ", p1)[:2200])
    print()
    time.sleep(2)
