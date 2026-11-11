"""Lore QA scorer (docs/planning/lore_qa/proposal.md §3.5). Deterministic; its source hash goes into every run manifest.

Extraction is the frozen math scorer's (src/score/scorer.py `extract`): thinking removed, the last \\boxed{...},
otherwise "the answer is ...", otherwise unparsed; truncated outputs (finish_reason == "length") and alternatives
offered side by side ("\\boxed{A} or \\boxed{B}") are scored incorrect and counted, as in the math scorer. Only the
equivalence is new, because lore answers are names, short phrases, numbers and years:

- names (answer class "name": relations, and strings whose words of 4+ letters are capitalized): after normalization
  (NFKC, LaTeX text wrappers and markup removed, case folded, hyphens as spaces, surrounding punctuation and a
  leading article dropped) the prediction equals one of the item's answer forms, or contains one as a contiguous run
  of words with every extra word in QUALIFIERS (honorifics and generic words such as "House", "City", "Lady").
  1.1.0 adds: the first clause of the prediction (split at , ; : ( and dashes), or the second when the first holds
  only qualifiers and titles, holds an answer form preceded only by QUALIFIERS/TITLES and followed by nothing, by
  an epithet ("the ...", "of ..."), or by one GENERIC_HEADS word ("Morthaxus the Flesh-Sculptor", "Professor
  Maelis Thornbridge", "Garnetport delegation", "Illegal; penalties vary by severity");
- phrases (answer class "phrase": descriptive strings), 1.1.0: words canonicalized (number words as digits, "%" as
  "percent", "N centuries / millennia / decades" as years). Correct if equal to an answer form; or if the form's
  numbers equal the prediction's and every other predicted word is in the form or APPROX ("20" for "twenty
  meters"); or if the prediction contains the form as a contiguous run not joined to other words by "and"; or, for
  forms of 3+ words, if the prediction is the form without its first word ("reduced tariffs"). A HEDGE word in the
  prediction but not in the form ("or", "not", ...) makes it wrong;
- numbers: exactly one number in the prediction (number words read as digits from 1.1.0), equal to the gold, and
  every other word in NUMBER_WORDS;
- years: exactly one year, equal to the gold year; an era prefix ("3E") may be omitted, but if given it must match;
  other words must be in YEAR_WORDS ("around", "the year", ...).

1.1.0 (2026-11-07) comes from the G3b hand check of the 4B dev phrase answers at budget 512
(docs/planning/lore_qa/pilot/): 1.0.1 marked 20 of 62 right answers wrong and no wrong answer right, and the
primary/second-scorer disagreements on names were all of the same kind. It is validated on dev rows it was not
fitted to (analysis/lore_pilot.py, G3b).
"""
import re
import unicodedata

from src.score.scorer import extract, strip_thinking

LORE_SCORER_VERSION = "1.1.0"   # 1.0.1 (dev code check): possessive 's dropped; 1.1.0 (G3b, dev): see the docstring
QUALIFIERS = {
    "house", "family", "clan", "city", "town", "village", "castle", "keep", "hold", "fortress", "lake", "isle",
    "island", "the", "of", "a", "an", "lord", "lady", "ser", "sir", "dame", "magister", "castellan",
    "king", "queen", "prince", "princess", "duke", "duchess", "marquis", "marchioness", "count", "countess",
    "viscount", "viscountess", "baron", "baroness", "shaped", "layout", "plan", "currency", "parliament",
}
TITLES = {"professor", "doctor", "dr", "master", "territory", "region", "province"}                      # 1.1.0
GENERIC_HEADS = {"delegation", "network", "territory", "region", "people", "dynasty", "faction", "guild", "order",
                 "family", "house", "clan", "line", "bloodline"}                                        # 1.1.0
HEDGE = {"or", "either", "neither", "nor", "not", "no", "never", "unknown", "unclear", "uncertain"}      # 1.1.0
APPROX = {"about", "around", "approximately", "roughly", "some"}                                        # 1.1.0
NUMBER_WORDS = {"years", "year", "old", "aged", "age", "about", "around", "approximately", "roughly", "is", "of",
                "m", "metres", "meters", "metre", "meter", "ft", "feet", "bridges", "bridge", "streams", "stream"}
YEAR_WORDS = {"around", "about", "approximately", "circa", "c", "roughly", "in", "the", "year", "sometime", "early",
              "late", "mid", "of"}
_NUM = re.compile(r"\d+(?:\.\d+)?")
_YEAR = re.compile(r"(?:\b([123])\s*e\b\s*)?~?\s*\b(\d{1,4})s?\b")
_SMALL = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_SCALE = {"thousand": 1000, "million": 1000000}
_YEARS_PER = {"century": 100, "centuries": 100, "millennium": 1000, "millennia": 1000, "decade": 10, "decades": 10}
_SEP = re.compile(r"[,;:(\u2013\u2014]")


def _unwrap(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    for _ in range(3):
        s = re.sub(r"\\(?:text|textbf|textit|mathrm|mathbf|emph|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    return s.replace("\\ ", " ").replace("$", "").replace("**", "").replace("__", "")


def normalize(s):
    s = re.sub(r"[-–—~]", " ", _unwrap(s).casefold())
    s = re.sub(r"'s\b", "", s)                     # 1.0.1: possessives ("Wardell's" for "Wardell"), both sides alike
    s = re.sub(r"\s+", " ", s).strip(" \t\"'.,;:!?()[]{}*`")
    return re.sub(r"^(?:the|a|an) ", "", s).strip()


def _words(s):
    return re.findall(r"[a-z0-9']+", s)


def answer_class(kind, value):
    """As src/data/lore.py answer_class (tests/test_lore_scorer.py checks they agree on every built item)."""
    if kind in ("relation", "number", "year"):
        return "name"
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", str(value))
    return "name" if words and all(w[0].isupper() for w in words if len(w) >= 4) else "phrase"


def _is_num(w):
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", w))


def canon_words(s):
    """1.1.0: normalized words with number words as digits, '%' as 'percent', and centuries etc. as years."""
    s = re.sub(r"\\?%", " percent ", _unwrap(s))
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = normalize(s.replace("per cent", "percent"))
    toks = re.findall(r"\d+(?:\.\d+)?|[a-z][a-z0-9']*", s)
    out, i = [], 0
    while i < len(toks):                                  # runs of number words -> one number
        j, total, cur = i, 0, None
        while j < len(toks) and (toks[j] in _SMALL or toks[j] in _TENS or toks[j] == "hundred" or toks[j] in _SCALE):
            w = toks[j]
            if w in _SMALL or w in _TENS:
                cur = (cur or 0) + _SMALL.get(w, _TENS.get(w, 0))
            elif w == "hundred":
                cur = (cur or 1) * 100
            else:
                total, cur = total + (cur or 1) * _SCALE[w], None
            j += 1
        if j > i:
            out.append(str(total + (cur or 0)))
            i = j
        else:
            out.append(toks[i])
            i += 1
    res = []
    for k, w in enumerate(out):                           # "2 centuries" -> "200 years"
        if w in _YEARS_PER and res and _is_num(res[-1]):
            res[-1] = str(int(float(res[-1]) * _YEARS_PER[w]))
            res.append("years")
        else:
            res.append(w)
    return res


def _strict_name(pw, fw):
    if pw == fw:
        return True
    return any(pw[i:i + len(fw)] == fw and all(w in QUALIFIERS for w in pw[:i] + pw[i + len(fw):])
               for i in range(len(pw) - len(fw) + 1))


def _clause_name(pw, fw):
    """1.1.0: the form, preceded only by qualifiers/titles, followed by nothing, an epithet, or one generic head."""
    for i in range(len(pw) - len(fw) + 1):
        if pw[i:i + len(fw)] != fw or not all(w in QUALIFIERS or w in TITLES for w in pw[:i]):
            continue
        rest = pw[i + len(fw):]
        if (not rest or all(w in QUALIFIERS for w in rest) or (rest[0] in ("the", "of") and len(rest) >= 2)
                or (len(rest) == 1 and rest[0] in GENERIC_HEADS)):
            return True
    return False


def _name_match(pred, forms):
    p = normalize(pred)
    if not p:
        return False
    pw = _words(p)
    fws = [fw for fw in (_words(normalize(f)) for f in forms) if fw]
    if any(_strict_name(pw, fw) for fw in fws):
        return True
    clauses = [_words(normalize(c)) for c in _SEP.split(_unwrap(pred).casefold())]
    clauses = [c for c in clauses if c]
    if not clauses:
        return False
    first = clauses[0]
    if all(w in QUALIFIERS or w in TITLES for w in first) and len(clauses) > 1:
        first = clauses[1]
    return any(_clause_name(first, fw) for fw in fws)


def _phrase_match(pred, forms):
    pw = canon_words(pred)
    if not pw:
        return False
    for f in forms:
        fw = canon_words(f)
        if not fw:
            continue
        if pw == fw:
            return True
        if any(w in HEDGE and w not in fw for w in pw):
            continue
        nf, np_ = sorted(float(w) for w in fw if _is_num(w)), sorted(float(w) for w in pw if _is_num(w))
        if nf:
            if nf != np_:
                continue
            if all(w in fw or w in APPROX for w in pw if not _is_num(w)):
                return True
        for i in range(len(pw) - len(fw) + 1):
            if pw[i:i + len(fw)] == fw and not ((i > 0 and pw[i - 1] == "and") or
                                                (i + len(fw) < len(pw) and pw[i + len(fw)] == "and")):
                return True
        if len(fw) >= 3 and pw == fw[1:]:
            return True
    return False


def _number_match(pred, gold):
    p = " ".join(canon_words(pred))
    nums = _NUM.findall(p)
    rest = [w for w in _words(_NUM.sub(" ", p))]
    return len(nums) == 1 and float(nums[0]) == float(str(gold).replace(",", "")) and all(w in NUMBER_WORDS for w in rest)


def _year_match(pred, gold_era, gold_year):
    p = re.sub(r"\b([123])e(?=\d)", r"\1e ", normalize(pred))        # "3E1570"
    hits = _YEAR.findall(p)
    if len(hits) != 1:
        return False
    era, year = hits[0]
    rest = _words(_YEAR.sub(" ", p))
    return int(year) == gold_year and era in ("", gold_era) and all(w in YEAR_WORDS for w in rest)


def equivalent(pred, answer_forms, answer_kind, answer_cls=None):
    """True when the extracted prediction matches the item's gold (see the module docstring)."""
    if pred is None:
        return False
    if answer_kind == "number":
        return _number_match(pred, answer_forms[0])
    if answer_kind == "year":
        m = re.match(r"^([123])E (\d{1,4})$", answer_forms[0])
        return bool(m) and _year_match(pred, m.group(1), int(m.group(2)))
    if (answer_cls or answer_class(answer_kind, answer_forms[0])) == "phrase":
        return _phrase_match(pred, answer_forms)
    return _name_match(pred, answer_forms)


def score(text, item, finish_reason=None):
    """Score one output against a lore item. Returns {"status", "pred", "correct"} (statuses as in src/score/scorer.py)."""
    forms, kind, cls = item["answer_forms"], item["answer_kind"], item.get("answer_class")
    pred, status, final = extract(text)
    if finish_reason == "length":
        return {"status": "truncated", "pred": pred, "correct": False}
    if status not in ("ok", "alternatives"):
        return {"status": status, "pred": pred, "correct": False}
    if status == "alternatives" and normalize(final[-2]) != normalize(final[-1]) and not (
            equivalent(final[-2], forms, kind, cls) and equivalent(final[-1], forms, kind, cls)):
        return {"status": "conflict", "pred": pred, "correct": False}
    return {"status": "ok", "pred": pred, "correct": equivalent(pred, forms, kind, cls)}


def score_contains(text, item, finish_reason=None):
    """Second scorer (agreement check on dev, design §5.6): an answer form as a whole word in the last box's content,
    or else in the last 200 characters; years and numbers by their digits. Independent of the equivalence rules above."""
    if finish_reason == "length":
        return {"status": "truncated", "correct": False}
    body = strip_thinking(text or "")
    if not body:
        return {"status": "unparsed", "correct": False}
    _, status, final = extract(body)
    tail = final[-1] if status in ("ok", "alternatives") and final else body[-200:]
    tail = normalize(tail)
    forms = item["answer_forms"]
    if item["answer_kind"] == "year":
        forms = [forms[0].split()[-1]]
    hit = any(re.search(r"(?<![a-z0-9])" + re.escape(normalize(f)) + r"(?![a-z0-9])", tail) for f in forms if normalize(f))
    return {"status": "ok", "correct": hit}
