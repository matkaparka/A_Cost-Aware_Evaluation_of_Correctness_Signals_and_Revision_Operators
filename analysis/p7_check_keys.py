"""Checks the paper's number references (P7; CLAUDE.md rule 1): every \\res{key} in paper/ must exist in
paper/generated/results.tex, and every macro of paper/numbers_placeholders.tex used in the text should be defined in
paper/generated/numbers.tex. Prints the missing ones; exit code 1 if any \\res key is missing.
Usage: python analysis/p7_check_keys.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    keys = set(re.findall(r"\\csname res@(.*?)\\endcsname", (P / "generated" / "results.tex").read_text(encoding="utf-8")))
    srcs = ([P / "first_draft.tex", P / "manuscript.tex"] + sorted((P / "sections").glob("*.tex"))
            + sorted((P / "sections" / "fu").glob("*.tex")) + sorted((P / "ms").glob("*.tex")))
    used, missing = set(), []
    for f in srcs:
        for k in re.findall(r"\\res\{([^{}]*)\}", f.read_text(encoding="utf-8")):
            k = " ".join(k.split())
            used.add(k)
            if k not in keys:
                missing.append((f.name, k))
    placeholders = re.findall(r"\\providecommand\{\\(\w+)\}", (P / "numbers_placeholders.tex").read_text(encoding="utf-8"))
    nums = P / "generated" / "numbers.tex"
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", nums.read_text(encoding="utf-8"))) if nums.exists() else set()
    text = "\n".join(f.read_text(encoding="utf-8") for f in srcs)
    undefined_used = [m for m in placeholders if re.search(r"\\" + m + r"(?![A-Za-z])", text) and m not in defined]
    print(f"\\res keys used: {len(used)}; missing: {len(missing)}")
    for f, k in missing:
        print(f"  MISSING res key in {f}: {k}")
    print(f"number macros used but not defined in generated/numbers.tex: {len(undefined_used)}")
    for m in undefined_used:
        print(f"  {m}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
