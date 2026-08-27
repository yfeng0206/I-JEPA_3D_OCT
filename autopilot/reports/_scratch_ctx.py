import io, re, urllib.request
from pypdf import PdfReader
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0 Safari/537.36"}

def get(aid):
    d=urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/pdf/{aid}",headers=UA),timeout=120).read()
    r=PdfReader(io.BytesIO(d))
    return "\n".join([(p.extract_text() or "") for p in r.pages])

for aid,label,pats in [
  ("2511.00782","Count-Based Approaches Remain Strong",[r".{300}±.{120}", r".{350}(bootstrap|confidence interval|repeat|variance|reproduc).{250}"]),
  ("2511.00782","Count-Based limitations",[r"Limitation.{900}"]),
  ("2510.12255","Shallow Robustness limitations",[r"Limitation.{900}"]),
  ("2508.08504","Domain Expert lessons",[r"(Lessons|lessons learned).{700}"]),
]:
    t=get(aid)
    print("="*90); print(label)
    for pat in pats:
        m=list(re.finditer(pat,t,re.I|re.S))
        for x in m[:2]:
            print("  >>> " + re.sub(r"\s+"," ",x.group(0))[:1200]); print()
