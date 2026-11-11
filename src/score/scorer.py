"""Deterministic answer scorer (design §5.6). Versioned; its source hash goes into every run manifest.

Extraction: the last \\boxed{...} (or \\fbox{...}); otherwise the last "The answer is ..." / "final answer is ...";
otherwise the last "#### ..."; otherwise unparsed. Thinking text (<think>...</think>) is removed first; an
unclosed <think> means the output never reached an answer (unparsed).

Equivalence: normalized string equality, or else math-verify (parse both, verify(gold, pred)), with its timeouts
disabled so the result never depends on wall-clock time; answers longer than MAX_SYMBOLIC_CHARS fall back to normalized
string equality.
1.1.1 (DEVIATIONS 2026-11-05, P4): math-verify does not terminate on some answers (10^{10^{10}}, 9^{9^{9}},
(x+1)^{100000}): the first 4B test runner was killed by the out-of-memory killer while scoring. On Linux every
math-verify call now runs in a child forked by a small helper process (`_server`, a plain `python -c` interpreter that
has loaded math-verify once), with a CPU-time limit (GUARD_CPU_S) and an address-space limit (GUARD_MEM_GB); a call
that exceeds either counts as a math-verify exception did in 1.1.0 (not equivalent unless the normalized strings are
equal) and is recorded in GUARDED. String equality is checked first; 1.1.0 returned "math-verify or string equality",
so the order changes no result. Elsewhere (the laptop: tests and analysis) calls run inline.

Scored incorrect and counted by status: unparsed, truncated (finish_reason == "length"), conflict (the last two
boxed answers are offered as alternatives, separated only by "or", ",", "/", ";" or "and", and are not
equivalent). Otherwise the last box is the answer, so a revision such as "I got \\boxed{5}, but the answer is
\\boxed{7}" scores 7. If the gold has several elements, the final paragraph's boxes are joined into one set
answer. Nothing is dropped.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

SCORER_VERSION = "1.1.1"
MAX_SYMBOLIC_CHARS = 300
GUARD_CPU_S, GUARD_MEM_GB, GUARD_WALL_S = 20, 8, 600   # wall: a backstop only; the CPU limit ends a child first
GUARDED = []   # calls the guard stopped: {"fn", "args", "reason"}, once per distinct call
_CACHE = {}
_SERVER = None

_THINK = re.compile(r"<think>.*?</think>", re.S)
# 1.1.0 (P2 hand check of the Tsui cell): "The correct answer is 8", "the answer to 5 - 1 is 4" also count, and a
# capture ends at the end of its sentence or at ", not ..." ("is **4**, not 3").
_ANSWER_IS = re.compile(r"(?:\b(?:final|correct|right)\s+answer|\bthe answer|^[\W_]*answer(?=\s*:))"
                        r"(?:\s+(?:to|for)\b[^.\n]{0,60}?)?\s*(?:\bis\b|:|=)\s*:?\s*(.*)", re.I)
_TAIL_END = re.compile(r"(?<=\S)[.!?](?=\s|$)|,\s*(?:not|rather than|instead of)\b")
# "8 pieces", "$9.00 is what Chad still owes" -> the leading number ("2 and 3", "1 or 2" are left alone).
_LEADING_NUMBER = re.compile(r"^(-?\$?\d[\d,]*(?:\.\d+)?%?)(?:\s*°|\s+(?!(?:and|or)\b)[A-Za-z])", re.I)
_HASHES = re.compile(r"####\s*(.+)")
_NUMBER = re.compile(r"-?\$?\d(?:[\d,]*\d)?(?:\.\d+)?(?:/\d+)?")


def strip_thinking(text):
    if "<think>" in text and "</think>" not in text:
        return None
    return _THINK.sub("", text).strip()


def _boxed_spans(text):
    """All \\boxed{...} / \\fbox{...} as (start, end, content), with balanced braces."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        j, depth = m.end(), 1
        for k in range(j, len(text)):
            c = text[k]
            depth += (c == "{") - (c == "}")
            if depth == 0:
                out.append((m.start(), k + 1, text[j:k].strip()))
                break
    return out


# Text between two boxes that makes them simultaneous alternatives ("\boxed{3} or \boxed{4}"), as opposed to a
# revision ("I got \boxed{5}, but the correct answer is \boxed{7}"), where the last box is the final answer.
_ALTERNATIVE_GAP = re.compile(r"^[\s$]*(?:or|and|,|/|;|\\text\{\s*or\s*\})?[\s$]*$", re.I)


def _clean_tail(ans):
    """Trim math delimiters, bold markers and trailing punctuation from an "answer is" capture.

    1.1.0: a short bold segment with a digit is the answer ("give back **$500** in change"); a plain-language
    sentence without LaTeX and with exactly one number reduces to that number ("Becca has 153 cards")."""
    ans = ans.replace("\\$", "$")
    bold = [b for b in re.findall(r"\*\*(.+?)\*\*", ans) if re.search(r"\d", b)]
    if bold and len(bold[0]) <= 24:
        ans = bold[0]
    ans = ans.replace("**", "").strip()
    ans = _TAIL_END.split(ans, maxsplit=1)[0].strip()
    m = _LEADING_NUMBER.match(ans)
    if m:
        ans = m.group(1)
    elif not re.search(r"[\\^{}_]", ans) and re.search(r"[A-Za-z]{3,}", ans) and len(_NUMBER.findall(ans)) == 1:
        ans = _NUMBER.findall(ans)[0]
    ans = re.sub(r"[\s.]+$", "", ans)
    ans = re.sub(r"^\$+|\$+$", "", ans).strip()
    ans = re.sub(r"^\\\(|\\\)$", "", ans).strip()
    return ans


def extract(text, last_number=False):
    """Return (answer or None, status, boxed answers in the final paragraph).

    last_number: only for the Tsui continuation protocol, where the model continues a given solution and has no
    answer-format instruction; if nothing else parses, the last number in the text is the answer (status
    "last_number"). Never used for prompts that ask for a boxed answer."""
    if text is None:
        return None, "unparsed", []
    body = strip_thinking(text)
    if body is None:
        return None, "unparsed", []
    boxes = _boxed_spans(body)
    if boxes:
        para_start = body.rfind("\n\n", 0, boxes[-1][0])
        final = [b for b in boxes if b[0] > para_start and b[2]]
        if not final:
            return None, "unparsed", []
        if len(final) >= 2 and _ALTERNATIVE_GAP.match(body[final[-2][1]:final[-1][0]]):
            return final[-1][2], "alternatives", [b[2] for b in final]
        return final[-1][2], "ok", [b[2] for b in final]
    lines = [ln for ln in body.splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        m = _ANSWER_IS.search(lines[i])
        if not m:
            continue
        # "### Final Answer:" alone on its line: the answer is on the next line.
        cap = m.group(1) if m.group(1).strip(" *#:") else (lines[i + 1] if i + 1 < len(lines) else "")
        if _clean_tail(cap):
            return _clean_tail(cap), "ok", []
    for ln in reversed(lines):
        m = _HASHES.search(ln)
        if m and _clean_tail(m.group(1)):
            return _clean_tail(m.group(1)), "ok", []
    if last_number:
        nums = _NUMBER.findall(body)
        if nums:
            return nums[-1].replace("$", "").replace(",", ""), "last_number", []
    return None, "unparsed", []


def _norm_str(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|,|;|!|\s)", "", s)
    s = s.replace("$", "").replace("\\%", "").replace("%", "").replace(",", "").replace(" ", "")
    return s.rstrip(".").casefold()


def _parse(ans):
    from math_verify import parse
    return parse("$\\boxed{" + ans + "}$", parsing_timeout=None)


def _symbolic(pred, gold):
    """math-verify's verdict; False when either side does not parse or verify raises."""
    from math_verify import verify
    g, p = _parse(gold), _parse(pred)
    if not (g and p):
        return False
    try:
        return bool(verify(g, p, timeout_seconds=None))
    except Exception:
        return False


def _parses_to_set(gold):
    g = _parse(gold)
    return bool(g) and isinstance(g[0], (set, frozenset))


_FNS = {"_symbolic": _symbolic, "_parses_to_set": _parses_to_set}


def _server():
    """The guard's helper (Linux): one JSON job [fn, args] per stdin line, one JSON reply [status, value] per stdout
    line. Each job runs in a fresh fork of this single-threaded process under the CPU-time and address-space limits, so
    every call starts from the same state."""
    import logging
    import os
    import resource
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    _symbolic("1", "2")   # load math-verify and its parsers once, before any fork
    for line in sys.stdin:
        fn, args = json.loads(line)
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                resource.setrlimit(resource.RLIMIT_CPU, (GUARD_CPU_S, GUARD_CPU_S + 5))
                vm = int(open("/proc/self/statm").read().split()[0]) * resource.getpagesize()   # what the fork maps
                resource.setrlimit(resource.RLIMIT_AS, (vm + (GUARD_MEM_GB << 30),) * 2)
                out = ["ok", _FNS[fn](*args)]
            except MemoryError:
                out = ["memory", None]
            except Exception as e:
                out = ["error", repr(e)[:200]]
            os.write(w, json.dumps(out).encode())
            os._exit(0)
        os.close(w)
        data = b""
        while chunk := os.read(r, 1 << 16):
            data += chunk
        os.close(r)
        code = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
        out = json.loads(data) if data else ["killed", f"exit code {code}"]
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def _call_server(fn, args):
    """[status, value] from the helper; it is (re)started when absent or dead, and killed at the wall-clock backstop."""
    global _SERVER
    import select
    import subprocess
    job = (json.dumps([fn.__name__, list(args)]) + "\n").encode()
    for _ in range(2):
        if _SERVER is None or _SERVER.poll() is not None:
            _SERVER = subprocess.Popen([sys.executable, "-c", "from src.score.scorer import _server; _server()"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       cwd=Path(__file__).resolve().parents[2])
        try:
            _SERVER.stdin.write(job)
            _SERVER.stdin.flush()
            break
        except BrokenPipeError:
            _SERVER = None
    if not select.select([_SERVER.stdout], [], [], GUARD_WALL_S)[0]:
        _SERVER.kill()
        _SERVER.wait()
        _SERVER = None
        return ["wall", None]
    line = _SERVER.stdout.readline()
    if not line:
        _SERVER = None
        return ["helper exited", None]
    return json.loads(line)


def _guarded(fn, *args):
    """fn(*args) under the guard (Linux), cached; False if the guard stops it."""
    key = (fn.__name__, *args)
    if key in _CACHE:
        return _CACHE[key]
    if not sys.platform.startswith("linux"):
        _CACHE[key] = fn(*args)
        return _CACHE[key]
    status, val = _call_server(fn, args)
    if status != "ok":
        GUARDED.append({"fn": fn.__name__, "args": list(args), "reason": status if val is None else f"{status}: {val}"})
        val = False
    _CACHE[key] = val
    return val


def equivalent(pred, gold):
    if pred is None:
        return False
    if len(pred) > MAX_SYMBOLIC_CHARS or len(gold) > MAX_SYMBOLIC_CHARS:
        return _norm_str(pred) == _norm_str(gold)
    return _norm_str(pred) == _norm_str(gold) or _guarded(_symbolic, pred, gold)


def _gold_is_multi(gold):
    g = _guarded(_parses_to_set, gold) if len(gold) <= MAX_SYMBOLIC_CHARS else False
    return g or ("," in gold and not re.search(r"\d,\d{3}", gold))


def score(text, gold, finish_reason=None, last_number=False):
    """Score one output against the gold. Returns {"status", "pred", "correct"}."""
    if finish_reason == "length":
        pred, _, _ = extract(text, last_number)
        return {"status": "truncated", "pred": pred, "correct": False}
    pred, status, final = extract(text, last_number)
    if status == "last_number":
        return {"status": status, "pred": pred, "correct": equivalent(pred, gold)}
    if status not in ("ok", "alternatives"):
        return {"status": status, "pred": pred, "correct": False}
    if len({_norm_str(b) for b in final}) > 1 and _gold_is_multi(gold):
        joined = ", ".join(final)
        return {"status": "ok", "pred": joined, "correct": equivalent(joined, gold)}
    if status == "alternatives" and not equivalent(final[-2], final[-1]):
        return {"status": "conflict", "pred": pred, "correct": False}
    return {"status": "ok", "pred": pred, "correct": equivalent(pred, gold)}


# ---- Exp 0: option-letter extraction -------------------------------------------------------------

def extract_option(text, letters=("A", "B")):
    """The chosen option letter, or None (invalid). Accepts \\boxed{A}, \\boxed{(A)}, \\boxed{\\text{A}}."""
    pred, status, final = extract(text)
    if status not in ("ok", "alternatives"):
        return None
    picked = {re.sub(r"[()\s]|\\text\{|\}", "", b).upper() for b in final}
    if len(picked) != 1:
        return None
    (x,) = picked
    return x if x in letters else None


# ---- Second scorer (agreement check on dev, design §5.6) ----------------------------------------

def score_simple(text, gold, finish_reason=None):
    """Independent fallback: same extraction, numeric/string comparison without math-verify."""
    if finish_reason == "length":
        return {"status": "truncated", "correct": False}
    pred, status, _ = extract(text)
    if status != "ok":
        return {"status": status, "correct": False}
    p, g = _norm_str(pred), _norm_str(gold)
    frac = re.compile(r"^(-?)\\d?frac\{(-?\d+)\}\{(\d+)\}$")

    def num(s):
        m = frac.match(s)
        if m:
            return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
        return float(s)
    try:
        return {"status": "ok", "correct": abs(num(p) - num(g)) < 1e-6}
    except (ValueError, ZeroDivisionError):
        return {"status": "ok", "correct": p == g}
