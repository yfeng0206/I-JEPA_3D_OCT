import io, re, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0 Safari/537.36"}
NS = {"a": "http://www.w3.org/2005/Atom"}

TITLES = [
 "Reliable or Risky? Assessing Diffusion Models for Biomedical Data Generation",
 "Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions",
 "When the Domain Expert Has No Time and the LLM Developer Has No Clinical Expertise",
 "Brittleness and Promise: Knowledge Graph-Based Reward Modeling for Diagnostic Reasoning",
 "Beyond Overall Accuracy: A Psychometric Deep Dive",
 "Statistically Significant Results on Biases and Errors of LLMs Do Not Guarantee Generalizable Results",
 "Examining the Vulnerability of Multi-Agent Medical Systems to Human Interventions",
 "FairGRPO",
 "Pandemic-Potential Viruses are a Blind Spot for Frontier Open-Source LLMs",
 "MedAgentGym",
 "MedGUIDE: Benchmarking Clinical Decision-Making in Large Language Models",
 "Editing with AI: How Doctors Refine LLM-Generated Answers to Patient Queries",
 "Scalable Whole-Slide Vision-Language Modeling with Learned Token Pruning",
 "The Energy to Say No",
 "Multi-Turn LLM Systems for Diagnostic Decision-Making",
 "Physician Perceptions of Large Language Models in Clinical Practice",
 "Large Language Models as Medical Codes Selectors",
 "Clinically Grounded Agent-based Report Evaluation",
 "Stabilizing Reasoning in Medical LLMs with Continued Pretraining",
 "Balancing Safety and Helpfulness in Healthcare AI Assistants",
]

for t in TITLES:
    q = 'ti:"%s"' % t.replace('?', '').replace(':', '')
    url = "http://export.arxiv.org/api/query?search_query=%s&max_results=1" % urllib.parse.quote(q)
    got = False
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
            root = ET.fromstring(raw)
            e = root.find("a:entry", NS)
            if e is None:
                print("MISS| %s" % t)
            else:
                title = re.sub(r"\s+", " ", e.find("a:title", NS).text)
                aid = e.find("a:id", NS).text
                cm = e.find("{http://arxiv.org/schemas/atom}comment")
                cmt = re.sub(r"\s+", " ", cm.text) if cm is not None and cm.text else ""
                print("OK  | %s | %s | C: %s" % (title, aid, cmt))
            got = True
            break
        except Exception as ex:
            time.sleep(8)
    if not got:
        print("ERR | %s" % t)
    time.sleep(4)
