"""Scorer for the multiple-choice validation task (MMLU-Pro; src/data/mmlupro.py). The answer is the option letter.

extract(text, item) -> (letter or None, status):
  - the content of the last \\boxed{...} (balanced braces). It counts when it is a single option letter, optionally in
    parentheses or followed by ".", ")" or ":" and option text ("B", "(B)", "B. Paris", "\\text{B}"), or when, after
    normalization, it equals the text of exactly one option;
  - with no usable box, the last "answer is (X)" / "answer: X" statement, as status "fallback";
  - otherwise None, status "unparsed"; generations that hit the token cap without a usable answer are "truncated".
score(text, item, finish_reason) -> {"status", "pred", "correct"}; correct iff the letter equals the gold letter.
"""
import re
import unicodedata

MC_SCORER_VERSION = "1.0.0"
LETTERS = "ABCDEFGHIJ"


def _norm(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\", "").replace("$", "")
    return re.sub(r"\s+", " ", s).strip().casefold().rstrip(".")


def last_boxed(text):
    i = text.rfind("\\boxed{")
    while i != -1:
        j, depth = i + len("\\boxed{"), 1
        k = j
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        if depth == 0:
            return text[j:k - 1]
        i = text.rfind("\\boxed{", 0, i)
    return None


def letter_of(content, item):
    n = item["n_options"]
    c = _norm(content)
    m = re.fullmatch(r"\(?([a-j])\)?(?:[.):]\s*.*)?", c)
    if m and LETTERS.index(m.group(1).upper()) < n:
        return m.group(1).upper()
    hits = [LETTERS[k] for k, o in enumerate(item["options"]) if _norm(str(o)) == c]
    return hits[0] if len(hits) == 1 else None


def extract(text, item):
    b = last_boxed(text)
    if b is not None:
        L = letter_of(b, item)
        if L:
            return L, "ok"
    ms = list(re.finditer(r"answer is:?\s*\(?\*{0,2}([A-J])\b\)?|answer:\s*\(?\*{0,2}([A-J])\b\)?", text, re.I))
    for m in reversed(ms):
        L = (m.group(1) or m.group(2)).upper()
        if LETTERS.index(L) < item["n_options"]:
            return L, "fallback"
    return None, "unparsed"


def score(text, item, finish_reason=None):
    pred, status = extract(text, item)
    if pred is None and finish_reason == "length":
        status = "truncated"
    return {"status": status, "pred": pred, "correct": pred is not None and pred == item["gold"]}
