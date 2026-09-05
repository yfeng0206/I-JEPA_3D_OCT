"""One-command refresh: data -> macros/figures -> PDF -> validated ZIP.

Runs the whole downstream chain in dependency order so a newly landed probe
propagates all the way to the Overleaf archive without any manual step:

  1. p3b_integrate_fp32.py   pick up any new fp32 / COVER probe
  2. p1b_full_inventory.py   re-label and de-duplicate the evidence base
  3. p1c_stats.py            re-derive AUCs, CIs, DeLong, BH families
  4. subgroup_analysis.py    re-run the subgroup/severity join   (--full only)
  5. p7b_gap_trend.py        gap trends, BH across attributes, branch-level
  5b. p17_adjust_subgroup_multiplicity.py
                             simultaneous subgroup-family intervals
  6. p8_make_assets.py and available figure producers
                             regenerate supported assets; fixed inputs are hashed
  7. p13_build_zip.py        compile/validate in isolation, generate/check Word
  8. p13_build_zip.py        publish ZIP/PDF/DOCX with release manifest last

Steps 3 and 4 are the slow ones (bootstrap resampling), so `--fast` skips the
subgroup re-run when no new probe affects it.

Usage:
  python refresh_all.py [--fast] [--out <zip path>]
"""
import argparse
import os
import subprocess
import sys
import time

PY = r"D:\jepa_phase0\.venv\Scripts\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = r"C:\Users\Gary\Desktop\jepa\paper\genai4health2026"
TECTONIC = r"D:\jepa_phase0\tools\tectonic\tectonic.exe"
REPO = r"C:\Users\Gary\Desktop\jepa"


def run(label, cmd, cwd=None):
    t0 = time.time()
    print("\n" + "=" * 72, flush=True)
    print("[%s] %s" % (time.strftime("%H:%M:%S"), label), flush=True)
    print("=" * 72, flush=True)
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       env={**os.environ, "MPLBACKEND": "Agg"})
    out = (p.stdout or "") + (p.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-14:])
    print(tail, flush=True)
    print("[%s] rc=%d  (%.1f s)" % (label, p.returncode, time.time() - t0), flush=True)
    if p.returncode != 0:
        raise SystemExit("%s failed (exit %d); refresh stopped" % (label, p.returncode))
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip the subgroup re-run (slow) when no subgroup input changed")
    ap.add_argument("--out", default=r"C:\Users\Gary\Downloads\OCT_JEPA_GenAI4Health2026_FINAL.zip")
    ap.add_argument("--staging-root")
    ap.add_argument("--expected-docx-sha256",
                    help="reviewed current Word hash when adopting an untracked Word copy")
    ap.add_argument("--citation-record", help="persisted, title-matched authority records")
    ap.add_argument("--review-file", help="explicit numeric binding/review input")
    a = ap.parse_args()

    t0 = time.time()
    run("1/8 integrate fp32 + COVER probes", [PY, os.path.join(HERE, "p3b_integrate_fp32.py")])
    run("2/8 rebuild evidence inventory", [PY, os.path.join(HERE, "p1b_full_inventory.py")])
    run("3/8 paired statistics (bootstrap + DeLong + BH)", [PY, os.path.join(HERE, "p1c_stats.py")])
    if not a.fast:
        run("4/8 subgroup + severity join",
            [PY, os.path.join(PAPER, "scripts", "subgroup_analysis.py"),
             "--out", r"D:\jepa_phase0\autopilot_out\subgroup", "--n-boot", "3000"],
            cwd=REPO)
    else:
        print("\n[4/8] skipped (--fast)", flush=True)
    run("5/8 subgroup gap trends", [PY, os.path.join(HERE, "p7b_gap_trend.py")])
    run("5b/8 adjust subgroup contrast families",
        [PY, os.path.join(HERE, "p17_adjust_subgroup_multiplicity.py")])
    run("6/8 regenerate macros, tables, figures", [PY, os.path.join(HERE, "p8_make_assets.py")])
    run("6a/8 regenerate geometry figure", [PY, os.path.join(HERE, "make_fig_geometry_panel.py")],
        cwd=REPO)
    run("6a2/8 regenerate specificity figure",
        [PY, os.path.join(HERE, "make_fig_specificity_ladder.py")], cwd=REPO)
    run("6a3/8 regenerate mask-statistics figure",
        [PY, "-c", "import runpy; m = runpy.run_path("
         "r'paper\\genai4health2026\\scripts\\make_story_figures.py'); m['fig_mask_stats']()"],
        cwd=REPO)
    run("6b/8 regenerate provenance", [PY, os.path.join(HERE, "gen_sources.py")])
    run("6b2/8 regenerate collaborator handoff", [PY, os.path.join(HERE, "gen_handoff.py")])
    # Gate: the checks below all exist because the corresponding mistake was made
    # and caught during this run. Non-zero here means a regression, not a nit.
    run("6c/8 manuscript consistency check", [PY, os.path.join(HERE, "check_manuscript.py")])
    numeric_command = [PY, os.path.join(HERE, "p15_verify_numbers.py")]
    if a.review_file:
        numeric_command += ["--review-file", a.review_file]
    run("6d/8 number provenance check (no cross-arm attribution)", numeric_command)
    command = [PY, os.path.join(HERE, "p13_build_zip.py"), "--out", a.out]
    for name in ("staging_root", "expected_docx_sha256", "citation_record", "review_file"):
        if getattr(a, name):
            command += ["--" + name.replace("_", "-"), getattr(a, name)]
    rc = run("7-8/8 stage, gate, compile, build/check Word, publish release", command)

    print("\n" + "=" * 72, flush=True)
    print("REFRESH COMPLETE in %.1f min -> %s" % ((time.time() - t0) / 60.0, a.out), flush=True)
    print("=" * 72, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
