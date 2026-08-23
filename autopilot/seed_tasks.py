"""Seed the autopilot task ledger."""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

TASKS = [
    # id, phase, title, expected duration, owner
    ("P0-01", "P0", "Inventory checkpoints, predictions, protocols", "0.5h", "coordinator"),
    ("P0-02", "P0", "Launch resource monitor and control plane", "0.2h", "coordinator"),
    ("P1-01", "P1", "Build protocol-matched paired master table from 19 saved prediction sets", "1.0h", "coordinator"),
    ("P1-02", "P1", "DeLong + paired bootstrap CIs for every arm contrast", "1.5h", "coordinator"),
    ("P1-03", "P1", "ROC / PR / calibration / operating-point metrics per arm", "1.0h", "coordinator"),
    ("P1-04", "P1", "Locate patient IDs; attempt patient-level clustered bootstrap", "1.0h", "coordinator"),
    ("P2-01", "P2", "Diff neurips_2026.sty against official NeurIPS 2026 style", "0.5h", "agent"),
    ("P2-02", "P2", "Related-work + citation research for anatomy-guided masked SSL", "3.0h", "agent"),
    ("P2-03", "P2", "Verify P2-01/P2-02 findings independently", "1.0h", "agent"),
    ("P3-01", "P3", "Frozen linear probe of COVER-0.21 epoch-73 checkpoint", "2.0h", "coordinator"),
    ("P3-02", "P3", "Assert encoder frozen + hash unchanged before/after probe", "0.2h", "coordinator"),
    ("P4-01", "P4", "Extract and cache frozen features for all arms", "4.0h", "coordinator"),
    ("P5-01", "P5", "Label-efficiency curves (1/5/10/25/100%) from cached features", "3.0h", "coordinator"),
    ("P6-01", "P6", "Embedding structure: PCA/UMAP, class separation, Cohen d", "4.0h", "coordinator"),
    ("P7-01", "P7", "Subgroup / fairness analysis on FairVision metadata", "3.0h", "coordinator"),
    ("P8-01", "P8", "Regenerate all figures and tables from verified artifacts", "3.5h", "coordinator"),
    ("P9-01", "P9", "Rewrite manuscript around protocol-matched paired evidence", "8.0h", "coordinator"),
    ("P9-02", "P9", "Compile first full PDF and check page limit", "0.5h", "coordinator"),
    ("P10-01", "P10", "Mock review round 1: 3 independent reviewers", "3.0h", "agent"),
    ("P10-02", "P10", "Meta-review and objection triage", "1.0h", "agent"),
    ("P11-01", "P11", "Revision pass addressing every R1 objection", "5.0h", "coordinator"),
    ("P12-01", "P12", "Mock review round 2 + numerical re-verification", "3.0h", "agent"),
    ("P13-01", "P13", "Final compile, anonymity + citation validation", "1.0h", "coordinator"),
    ("P13-02", "P13", "Build and validate final Overleaf ZIP", "1.0h", "coordinator"),
]

for tid, phase, title, exp, owner in TASKS:
    subprocess.run([PY, os.path.join(HERE, "state.py"), "addtask", tid,
                    "phase=" + phase, "title=" + title, "expected=" + exp, "owner=" + owner],
                   check=True, capture_output=True)
print("seeded %d tasks" % len(TASKS))
