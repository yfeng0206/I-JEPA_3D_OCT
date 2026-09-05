"""Check an explicit before/after semicolon edit; empty comparisons fail."""
import argparse
from pathlib import Path
import re
import subprocess
import sys

PATH = "paper/genai4health2026/main_submission.tex"


def tokens(text):
    return (re.findall(r"\\[A-Za-z]+", text),
            re.findall(r"\d+(?:\.\d+)?", text),
            re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", text),
            re.findall(r"\\(?:ref|label)\{([^}]*)\}", text))


def verify(old, new):
    if old == new:
        return ["empty comparison; this does not verify a historical edit"]
    before, after = old.splitlines(), new.splitlines()
    errors = []
    if len(before) != len(after):
        errors.append("line count changed")
    for index, (a, b) in enumerate(zip(before, after), 1):
        if a == b:
            continue
        if len(a) != len(b):
            errors.append("unexpected line change at %d" % index)
            continue
        allowed_caps = {match.end() - 1 for match in re.finditer(r";\s+([a-z])", a)
                        if b[match.start()] == "."}
        if any(x != y and not (x == ";" and y == ".")
               and not (i in allowed_caps and x.islower() and y == x.upper())
               for i, (x, y) in enumerate(zip(a, b))):
            errors.append("unexpected line change at %d" % index)
    if tokens(old) != tokens(new):
        errors.append("numeric, macro, citation or reference tokens changed")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="explicit commit before the edit")
    parser.add_argument("--path", default=PATH)
    args = parser.parse_args(argv)
    result = subprocess.run(["git", "--no-pager", "show",
                             args.base + ":" + args.path.replace("\\", "/")],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        print("Could not read explicit comparison base.")
        return 1
    errors = verify(result.stdout, Path(args.path).read_text(encoding="utf-8"))
    for error in errors:
        print("FAIL:", error)
    print("VERDICT:", "REVIEW REQUIRED" if errors else "PUNCTUATION ONLY")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
