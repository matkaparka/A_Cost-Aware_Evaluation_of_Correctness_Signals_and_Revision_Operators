"""Second scorer v2 for the scorer-agreement invariant (design §5.6, §8; DECISIONS 2026-11-05).

The first second scorer (`scorer.score_simple`) compares normalized strings and numbers without the standard MATH answer
normalization, so it marks equivalent forms such as \\frac43 vs \\frac{4}{3} or "18" vs "18\\text{ ways.}" wrong. v2 uses
the answer normalization of the MATH benchmark's reference grader (Hendrycks et al. 2021, github.com/hendrycks/math,
modeling/math_equivalence.py, MIT licence), re-implemented here, followed by numeric equality when both sides parse as
numbers. It shares the answer extraction with the main scorer (scorer.extract) and is independent of math-verify.
Fixed before its agreement was computed; not adjusted to the dev disagreements.
"""
import re

from .scorer import extract


def _fix_fracs(s):
    subs = s.split("\\frac")
    out = subs[0]
    for sub in subs[1:]:
        out += "\\frac"
        if sub[:1] == "{":
            out += sub
            continue
        if len(sub) < 2:
            return s
        a, b = sub[0], sub[1]
        if b != "{":
            out += "{" + a + "}{" + b + "}" + sub[2:]
        else:
            out += "{" + a + "}" + b + sub[2:]
    return out


def _fix_a_slash_b(s):
    parts = s.split("/")
    if len(parts) != 2:
        return s
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return s
    return "\\frac{" + str(a) + "}{" + str(b) + "}" if s == f"{a}/{b}" else s


def _remove_right_units(s):
    if "\\text{ " in s:
        return s.split("\\text{ ")[0]
    return s


def _fix_sqrt(s):
    if "\\sqrt" not in s:
        return s
    parts = s.split("\\sqrt")
    out = parts[0]
    for p in parts[1:]:
        out += ("\\sqrt{" + p[0] + "}" + p[1:]) if p[:1] not in ("{", "") else ("\\sqrt" + p)
    return out


def strip_string(s):
    """The MATH reference grader's normalization."""
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    s = _remove_right_units(s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if not s:
        return s
    if s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    return _fix_a_slash_b(s)


_FRAC = re.compile(r"^(-?)\\frac\{(-?\d+)\}\{(\d+)\}$")


def _num(s):
    m = _FRAC.match(s)
    if m:
        return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
    return float(s.replace(",", ""))


def equivalent_v2(pred, gold):
    if pred is None:
        return False
    a, b = strip_string(pred), strip_string(gold)
    if a == b:
        return True
    try:
        return abs(_num(a) - _num(b)) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


def score_v2(text, gold, finish_reason=None):
    """Same extraction as the main scorer; two final boxes must agree with each other under the same comparison."""
    if finish_reason == "length":
        return {"status": "truncated", "correct": False}
    pred, status, final = extract(text)
    if status not in ("ok", "alternatives"):
        return {"status": status, "correct": False}
    if status == "alternatives" and not equivalent_v2(final[-2], final[-1]):
        return {"status": "conflict", "correct": False}
    return {"status": "ok", "correct": equivalent_v2(pred, gold)}
