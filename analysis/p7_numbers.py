r"""P7 number macros (CLAUDE.md rule 1; paper/NUMBERS_NEEDED.md): every number the paper draft uses, read at run time
from the file and field that NUMBERS_NEEDED.md names, or from a newer authoritative source (listed in the output).

-> paper/generated/numbers.tex        \newcommand{\Name}{value} for every macro whose source resolves
-> paper/generated/tab_revisions.tex  reproducibility table: model / tokenizer revisions, chat-template hash, dataset
                                      revisions, data-manifest, lockfile and scorer hashes, software versions, GPU

This file holds paths, field names, cross-checks and formatting rules only; no value is typed here. A macro whose
source is missing, ambiguous, or disagrees with its cross-check is left undefined (paper/numbers_placeholders.tex then
prints a red [num: Name] marker) and is listed with the reason at the top of numbers.tex and in the printed summary.

Formatting (NUMBERS_NEEDED.md header): the value only, units in the text; percentages as numbers (fractions x 100);
integers >= 1000 with a {,} thousands separator (one style for the whole paper); a negative number starts with
\ensuremath{-}; sets as \ensuremath{\{a, b\}} (the draft uses some sets in running text); dates ISO, timestamps
"YYYY-MM-DD HH:MM UTC"; hashes as full lowercase hex, raw (the draft wraps them in \texttt); decimals as the source
report shows (each macro's comment names the report its rounding follows); constants as the source code writes them.
Every file is written with LF line ends.

Usage: PYTHONUTF8=1 .venv/Scripts/python.exe analysis/p7_numbers.py
"""
import ast
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "generated"
NEEDED = "paper/NUMBERS_NEEDED.md"
PLACEHOLDERS = "paper/numbers_placeholders.tex"

# Abbreviations of NUMBERS_NEEDED.md
D01 = "reports/01_data_tables.md"
EXT = "data/processed/ext/build_ext_manifest.json"
BUILD = "data/processed/build_manifest.json"
P2CFG = "results/p2/4B/p2-exp01-dev-4B-f7e5011/config.json"
TIER2 = "analysis/out/tier2.json"
SV = "analysis/out/selfverdict.json"
STEER_PILOT = "results/p2/4B/p2-steer-pilot-4B-7ea8f7d"
TTT_PILOT = "results/p2/4B/p2-ttt-pilot-4B-ea34050/config.json"
TTT_SERVE = "results/p2/4B/p2-ttt-serve-4B-ea34050/config.json"
PROBE_SEL = "results/p3/4B/p3-probe-dev-ext-4B-b7cf0d6/probe_selection.json"
CHECKS_MD = "reports/03_checks_tables.md"
CHECKS_JSON = "analysis/out/p3_checks.json"
AGREE_MD = "reports/03_scorer_agreement.md"
AGREE_JSON = "analysis/out/scorer_agreement.json"
P6C = "analysis/p6_confirmatory.py"


class Missing(Exception):
    """Source missing, ambiguous or failing its cross-check: the macro stays undefined."""


def need(cond, msg):
    if not cond:
        raise Missing(msg)


# ----------------------------------------------------------------------------------------------------------------
# Readers

def read(rel):
    p = ROOT / rel
    need(p.exists(), f"{rel} not found")
    return p.read_text(encoding="utf-8")


@lru_cache(None)
def jload(rel):
    return json.loads(read(rel))


@lru_cache(None)
def jlines(rel):
    return [json.loads(line) for line in read(rel).splitlines() if line.strip()]


def field(d, *path, src="?"):
    for k in path:
        need(isinstance(d, dict) and k in d, f"{src}: field {'.'.join(map(str, path))} not found")
        d = d[k]
    return d


def one(paths, what):
    need(len(paths) == 1, f"{what}: expected one match, found {len(paths)} ({', '.join(paths) or 'none'})")
    return paths[0]


def rglob(pattern):
    return sorted(Path(p).relative_to(ROOT).as_posix() for p in glob.glob(str(ROOT / pattern)))


@lru_cache(None)
def pyconst(rel):
    """Module-level assignments of a Python file: literal values, or the AST node of a non-literal value."""
    out = {}

    def put(target, value):
        if isinstance(target, ast.Name):
            try:
                out[target.id] = ast.literal_eval(value)
            except (ValueError, SyntaxError, TypeError):
                out[target.id] = value

    for node in ast.parse(read(rel)).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple) and len(t.elts) == len(node.value.elts):
                    for n, v in zip(t.elts, node.value.elts):
                        put(n, v)
                else:
                    put(t, node.value)
    return out


def const(rel, name):
    c = pyconst(rel)
    need(name in c, f"{rel}: constant {name} not found")
    need(not isinstance(c[name], ast.AST), f"{rel}: {name} is not a literal")
    return c[name]


def func_default(rel, func, arg):
    for node in ast.walk(ast.parse(read(rel))):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            a = node.args
            pos = a.posonlyargs + a.args
            for i, x in enumerate(pos):
                j = i - (len(pos) - len(a.defaults))
                if x.arg == arg and j >= 0:
                    return ast.literal_eval(a.defaults[j])
            for x, d in zip(a.kwonlyargs, a.kw_defaults):
                if x.arg == arg and d is not None:
                    return ast.literal_eval(d)
    raise Missing(f"{rel}: default of {func}({arg}=...) not found")


def argparse_default(rel, flag):
    m = re.findall(r'add_argument\(\s*"' + re.escape(flag) + r'"[^)]*?default=([^,)\s]+)', read(rel))
    need(len(m) == 1, f"{rel}: argparse default of {flag} matched {len(m)} times")
    return ast.literal_eval(m[0])


def regex1(rel, pattern, flags=0, text=None):
    m = re.findall(pattern, read(rel) if text is None else text, flags)
    need(len(m) == 1, f"{rel}: pattern {pattern!r} matched {len(m)} times")
    return m[0]


@lru_cache(None)
def md_tables(rel):
    """Markdown tables of a report: header, rows (dicts), the heading path and the text since the previous table."""
    lines = read(rel).splitlines()
    sep = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+\s*$")
    out, heading, ctx, i = [], [], [], 0

    def cells(line):
        return [c.strip().replace("**", "").strip("`").strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]

    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and i + 1 < len(lines) and sep.match(lines[i + 1]):
            header, rows, i = cells(line), [], i + 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(dict(zip(header, cells(lines[i]))))
                i += 1
            out.append({"heading": tuple(heading), "context": "\n".join(ctx), "header": header, "rows": rows})
            ctx = []
            continue
        if re.match(r"^#+\s", line):
            level = len(line) - len(line.lstrip("#"))
            heading = heading[:level - 1] + [line.lstrip("#").strip()]
            ctx = []
        elif line.strip():
            ctx.append(line)
        i += 1
    return out


def md_table(rel, cols, heading=None, context=None):
    ts = [t for t in md_tables(rel) if all(c in t["header"] for c in cols)
          and (heading is None or any(heading in h for h in t["heading"]))
          and (context is None or re.search(context, t["context"]))]
    need(len(ts) == 1, f"{rel}: {len(ts)} tables with columns {cols} (heading {heading!r}, context {context!r})")
    return ts[0]


def md_cell(t, match, col):
    rs = [r for r in t["rows"] if all(r.get(k) == v for k, v in match.items())]
    need(len(rs) == 1, f"{len(rs)} rows match {match}")
    need(col in rs[0], f"column {col!r} not found")
    return rs[0][col]


def numstr(s):
    s = s.replace("%", "").strip()
    need(re.fullmatch(r"-?\d+(\.\d+)?", s) is not None, f"not a number: {s!r}")
    return s


# ----------------------------------------------------------------------------------------------------------------
# Formatting

def dec(x):
    """Exact decimal of an int, a float (as Python writes it), a Decimal or a numeric string."""
    if isinstance(x, bool):
        raise Missing(f"boolean where a number is expected: {x!r}")
    if isinstance(x, Decimal):
        return x
    if isinstance(x, int):
        return Decimal(x)
    if isinstance(x, float):
        return Decimal(repr(x))
    return Decimal(numstr(str(x)))


def minimal(d):
    """The shortest exact decimal: 20.00 -> 20, 1.500 -> 1.5."""
    d = d.normalize()
    return d.quantize(Decimal(1)) if d.as_tuple().exponent > 0 else d


def group(intpart):
    return re.sub(r"(\d)(?=(\d{3})+$)", r"\1{,}", intpart) if len(intpart) > 3 else intpart


def fnum(x, nd=None):
    """Number for the paper. nd = decimals; None keeps the decimals the source shows (a string keeps its digits, a
    float is written as Python writes it, an int has none)."""
    d = dec(x)
    if nd is None:
        nd = max(0, -d.as_tuple().exponent)
    q = d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = format(abs(q), "f")
    ip, _, fp = s.partition(".")
    return ("\\ensuremath{-}" if q < 0 else "") + group(ip) + ("." + fp if fp else "")


def pct(x):
    """A stored fraction as an exact percentage (0.05 -> 5)."""
    return minimal(dec(x) * 100)


def fset(values):
    return "\\ensuremath{\\{" + ", ".join(fnum(v) for v in values) + "\\}}"


def near_one(x):
    """A probability just below 1: decimals through the first digit after the leading 9s, plus one."""
    frac = format(dec(x), "f").partition(".")[2]
    need(dec(x) < 1, f"{x} is not below 1")
    return fnum(x, len(frac) - len(frac.lstrip("9")) + 2)


def esc(s):
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "#": r"\#", "&": r"\&", "$": r"\$", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in str(s))


def fhash(h):
    need(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) is not None, f"not a full sha256: {h!r}")
    return h


def same(a, b, what):
    need(a == b, f"{what}: {a!r} != {b!r}")
    return a


def utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


# ----------------------------------------------------------------------------------------------------------------
# Shared sources

@lru_cache(None)
def runs():
    """runs.jsonl, the last entry of each run ID."""
    out = {}
    for r in jlines("runs.jsonl"):
        out[r["run_id"]] = r
    return out


@lru_cache(None)
def blocks():
    return {b["id"]: b for b in yaml.safe_load(read("configs/blocks.yaml"))["blocks"]}


def test_cfg(pattern, what):
    return jload(one(rglob(f"results/main/{pattern}/config.json"), what))


def exp01_test_4b():
    cs = [p for p in rglob("results/main/4B/main-exp01-test-4B-*/config.json") if jload(p).get("label_set") == "main"]
    return jload(one(cs, "4B Exp 1A/1B test run (main label set)"))


def alt_test_4b():
    cs = [p for p in rglob("results/main/4B/main-exp01-test-*4B-*/config.json") if jload(p).get("label_set") == "alt"]
    return jload(one(cs, "4B Exp 1B test run with the alternative label set"))


def robust_cfgs():
    ps = rglob("results/main/4B/main-exp1c-robust-test-c*-4B-*/config.json")
    need(ps, "no 4B exp1c-robust test run config")
    return [jload(p) for p in ps]


def robust_field(key):
    vals = {json.dumps(field(c, key, src="exp1c-robust config")) for c in robust_cfgs()}
    need(len(vals) == 1, f"exp1c-robust chunk configs disagree on {key}: {sorted(vals)}")
    return json.loads(vals.pop())


def thinking_cfgs():
    ps = rglob("results/main/4B/main-thinking-test-c*-4B-*/config.json")
    need(ps, "no 4B thinking test run config")
    return [jload(p) for p in ps]


def thinking_same(key, value, what):
    for c in thinking_cfgs():
        same(field(c, key, src="thinking test config"), value, f"{what} vs 4B thinking test config {key}")
    return value


def steer_test():
    d = one(rglob("results/main/4B/main-steer-test-4B-*"), "4B Exp 3 test run")
    return jload(d + "/config.json"), jload(d + "/metrics.json")


def ttt_test():
    d = one(rglob("results/main/4B/main-ttt-test-4B-*"), "4B Exp 5 test run")
    return jload(d + "/config.json"), jload(d + "/metrics.json")


def items(table, dataset, col):
    t = md_table(D01, ["Table", "Dataset", "Dev", "Test"], heading="Items per dataset and split")
    return int(numstr(md_cell(t, {"Table": table, "Dataset": dataset}, col)))


def source_rows(ds):
    t = md_table(D01, ["Dataset", "Repo", "Revision", "Rows"], heading="Sources")
    v = int(numstr(md_cell(t, {"Dataset": ds}, "Rows")))
    same(field(jload(BUILD), "sources", ds, "rows", src=BUILD), v, f"{BUILD} rows of {ds} vs report")
    return v


def a0_counts(model, col):
    t = md_table(D01, ["Model", "Dataset", "Test right", "Test wrong"], heading="Initial answers a0")
    rs = [r for r in t["rows"] if r["Model"] == model and r["Dataset"] in ("gsm8k", "math500")]
    need(len(rs) == 2, f"{D01}: {len(rs)} initial-answer rows for {model} gsm8k / math500")
    return sum(int(numstr(r[col])) for r in rs)


def thinking_subset_N(stratum):
    vals = {field(jload(p), "weights", stratum, "N", src=p) for p in rglob("results/main/4B/main-thinking-test-c*-4B-*/subset.json")}
    need(len(vals) == 1, f"thinking subset.json weights.{stratum}.N: {vals}")
    return vals.pop()


def orig_test():
    v = items("dnat", "gsm8k", "Test") + items("dnat", "math500", "Test")
    same(thinking_subset_N("False") + thinking_subset_N("True"), v, "thinking subset pool size vs D-nat original test")
    return v


def dinj_test():
    v = sum(items("dinj", d, "Test") for d in ("gsm8k_sc", "prm800k_sc", "scli5"))
    _, m = steer_test()
    same(m["items"]["dinj"] + m["items"]["dinj_excluded_no_content"], v, "Exp 3 test D-inj items + excluded vs D-inj test")
    return v


def ext_stat(*path):
    return field(jload(EXT), "stats", *path, src=EXT)


def pooled_test():
    v = orig_test() + ext_stat("items_by_split", "test")
    same(blocks()["4B-exp2"]["chunk"], v, "configs/blocks.yaml 4B-exp2 chunk vs pooled D-nat test")
    return v


def selfverdict(key):
    return field(jload(SV), "4B", key, src=SV)


def steer_pilot_metrics():
    return jload(STEER_PILOT + "/metrics.json")


def secondary(direction, key):
    cfg, _ = steer_test()
    sec = [s for s in field(cfg, "secondary", src="Exp 3 test config") if s["direction"] == direction]
    need(len(sec) == 1, f"Exp 3 test config: {len(sec)} secondary settings with direction {direction}")
    main = field(jload(TIER2), "exp3", "4B", "main_setting", src=TIER2)
    pilot = [t for t in steer_pilot_metrics()["top3"] if (t["layer"], t["direction"], t["alpha"]) !=
             (main["layer"], main["direction"], main["alpha"]) and t["direction"] == direction]
    need(len(pilot) == 1 and pilot[0]["layer"] == sec[0]["layer"] and pilot[0]["alpha"] == sec[0]["alpha"],
         f"secondary {direction} setting differs between the pilot top3 and the Exp 3 test config")
    return sec[0][key]


def checks_md_rule():
    return regex1(CHECKS_MD, r"overall truncation ≤ ([\d.]+) %")


def agree_rate(v):
    r = field(jload(AGREE_JSON), "rate", v, src=AGREE_JSON)
    t = md_table(AGREE_MD, ["Source", "Texts", f"{v} %"])
    s = fnum(pct(r), 2)
    same(s, md_cell(t, {"Source": "all"}, f"{v} %"), f"{AGREE_JSON} rate.{v} vs {AGREE_MD}")
    return s


def ledger():
    rows = jlines("cloud/ledger.jsonl")
    need(rows, "cloud/ledger.jsonl is empty")
    open_ = [r.get("session") for r in rows if r.get("pause_time") is None or r.get("hours") is None or r.get("compute_usd") is None]
    need(not open_, f"cloud/ledger.jsonl: sessions not closed: {open_}")
    return rows


def created():
    node = pyconst("cloud/session.py").get("CREATED")
    need(isinstance(node, ast.Call) and getattr(node.func, "id", None) == "datetime", "cloud/session.py: CREATED is not datetime(...)")
    args = [ast.literal_eval(a) for a in node.args]
    return datetime(*args, tzinfo=timezone.utc)


# ----------------------------------------------------------------------------------------------------------------
# Macros: name -> (generator, source description). REPLACED: macros whose source differs from NUMBERS_NEEDED.md.

GEN = {}
REPLACED = {}


def macro(name, source, replaced=None):
    def deco(f):
        GEN[name] = (f, source)
        if replaced:
            REPLACED[name] = replaced
        return f
    return deco


# Data and splits
for _name, _ds in (("NGsmRows", "gsm8k"), ("NMathFiveRows", "math500"), ("NGsmScRows", "gsm8k_sc"),
                   ("NPrmScRows", "prm800k_sc"), ("NScliRows", "scli5")):
    macro(_name, f"{D01} Sources, row {_ds}, Rows (= {BUILD} sources.{_ds}.rows)")(lambda ds=_ds: fnum(source_rows(ds)))


@macro("NSourceRows", f"{D01} 'Source rows' (= sum of {BUILD} source rows)")
def _():
    v = int(regex1(D01, r"Source rows: (\d+); canonical problems: \d+\."))
    same(sum(s["rows"] for s in jload(BUILD)["sources"].values()), v, f"{BUILD} total source rows vs report")
    return fnum(v)


@macro("NCanonical", f"{D01} 'canonical problems' (= rows of data/processed/problems.jsonl)")
def _():
    v = int(regex1(D01, r"Source rows: \d+; canonical problems: (\d+)\."))
    same(len(jlines("data/processed/problems.jsonl")), v, "problems.jsonl rows vs report")
    return fnum(v)


@macro("MinHashThreshold", f"src/data/build.py MINHASH_THRESHOLD (= {BUILD} minhash.threshold)")
def _():
    v = const("src/data/build.py", "MINHASH_THRESHOLD")
    same(field(jload(BUILD), "minhash", "threshold", src=BUILD), v, "build manifest minhash threshold")
    return fnum(v)


def dev_pct():
    v = const("src/data/build.py", "DEV_FRACTION")
    same(field(jload(BUILD), "dev_fraction", src=BUILD), v, "build manifest dev_fraction")
    return pct(v)


macro("DevPct", "src/data/build.py DEV_FRACTION x 100")(lambda: fnum(dev_pct()))
macro("TestPct", "100 - src/data/build.py DEV_FRACTION x 100")(lambda: fnum(100 - dev_pct()))

for _name, _t, _ds, _col in (("NDnatGsmDev", "dnat", "gsm8k", "Dev"), ("NDnatGsmTest", "dnat", "gsm8k", "Test"),
                             ("NDnatMathDev", "dnat", "math500", "Dev"), ("NDnatMathTest", "dnat", "math500", "Test"),
                             ("NDinjGsmDev", "dinj", "gsm8k_sc", "Dev"), ("NDinjGsmTest", "dinj", "gsm8k_sc", "Test"),
                             ("NDinjPrmDev", "dinj", "prm800k_sc", "Dev"), ("NDinjPrmTest", "dinj", "prm800k_sc", "Test"),
                             ("NDinjScliDev", "dinj", "scli5", "Dev"), ("NDinjScliTest", "dinj", "scli5", "Test"),
                             ("NExpZeroDev", "exp0", "exp0", "Dev"), ("NExpZeroTest", "exp0", "exp0", "Test")):
    macro(_name, f"{D01} Items per dataset and split, {_t}/{_ds}, {_col}")(
        lambda t=_t, ds=_ds, col=_col: fnum(items(t, ds, col)))


@macro("NExpZeroTemplates", "data/processed/exp0.jsonl: distinct template_id (= rows / 2)")
def _():
    rows = jlines("data/processed/exp0.jsonl")
    n = len({r["template_id"] for r in rows})
    same(2 * n, len(rows), "exp0.jsonl rows vs 2 x templates")
    return fnum(n)


@macro("NExpZeroItems", "data/processed/exp0.jsonl: rows (= Exp 0 Dev + Test in the data tables)")
def _():
    n = len(jlines("data/processed/exp0.jsonl"))
    same(items("exp0", "exp0", "Dev") + items("exp0", "exp0", "Test"), n, "Exp 0 dev + test vs exp0.jsonl rows")
    return fnum(n)


macro("NDnatOrigTest", f"{D01} dnat gsm8k + math500 Test (= thinking subset.json pool N)")(lambda: fnum(orig_test()))
macro("NDinjTest", f"{D01} dinj Test, sum of the three (= Exp 3 test items + excluded)")(lambda: fnum(dinj_test()))
macro("NExtRows", f"{EXT} stats.math_test_rows")(lambda: fnum(ext_stat("math_test_rows")))
macro("NExtDropped", f"{EXT} stats.dropped_by_reason['in existing pool']")(
    lambda: fnum(ext_stat("dropped_by_reason", "in existing pool")))


@macro("NExtKept", f"{EXT} stats.kept_items (= rows - dropped = dev + test)")
def _():
    v = ext_stat("kept_items")
    same(ext_stat("math_test_rows") - sum(ext_stat("dropped_by_reason").values()), v, "ext rows - dropped vs kept")
    same(ext_stat("items_by_split", "dev") + ext_stat("items_by_split", "test"), v, "ext dev + test vs kept")
    return fnum(v)


macro("NExtDev", f"{EXT} stats.items_by_split.dev")(lambda: fnum(ext_stat("items_by_split", "dev")))
macro("NExtTest", f"{EXT} stats.items_by_split.test")(lambda: fnum(ext_stat("items_by_split", "test")))
macro("NDnatPooledTest", "NDnatOrigTest + NExtTest (= configs/blocks.yaml 4B-exp2 chunk)")(lambda: fnum(pooled_test()))


@macro("ExtMathShare", "(NDnatMathTest + NExtTest) / NDnatPooledTest x 100, integer")
def _():
    share = Decimal(items("dnat", "math500", "Test") + ext_stat("items_by_split", "test")) / Decimal(pooled_test()) * 100
    return fnum(share, 0)


macro("NSpillover", f"{D01} 'Fixed sets', spillover")(
    lambda: fnum(int(regex1(D01, r"Fixed sets: spillover \(test\) (\d+), KL probe \(dev\) \d+"))))
macro("NKLProbe", f"{D01} 'Fixed sets', KL probe")(
    lambda: fnum(int(regex1(D01, r"Fixed sets: spillover \(test\) \d+, KL probe \(dev\) (\d+)"))))


@macro("NOrigWrongFourB", f"{D01} Initial answers a0, 4B, Test wrong, gsm8k + math500 (= thinking subset N, wrong)")
def _():
    return fnum(same(a0_counts("4B", "Test wrong"), thinking_subset_N("False"), "4B test wrong vs thinking subset N"))


@macro("NOrigRightFourB", f"{D01} Initial answers a0, 4B, Test right, gsm8k + math500 (= thinking subset N, right)")
def _():
    return fnum(same(a0_counts("4B", "Test right"), thinking_subset_N("True"), "4B test right vs thinking subset N"))


# Decoding and budgets
@macro("MaxTokInitial", "results/p1/4B/<the p1-initial-4B run marked done in runs.jsonl>/config.json max_tokens",
       replaced="run chosen from runs.jsonl (status done) and checked to be the a0 file of the 4B test runs")
def _():
    done = [k for k, r in runs().items() if re.fullmatch(r"p1-initial-4B-[0-9a-f]+", k) and r.get("status") == "done"]
    rid = one(done, "p1-initial-4B run with status done in runs.jsonl")
    need(rid in exp01_test_4b()["initial"], f"4B test runs read a0 from {exp01_test_4b()['initial']}, not {rid}")
    return fnum(field(jload(f"results/p1/4B/{rid}/config.json"), "max_tokens", src=rid))


for _name, _key in (("MaxTokRevision", "revision_tokens"), ("MaxTokCritique", "critique_tokens"),
                    ("MaxTokExpZero", "exp0_tokens"), ("MaxTokTsui", "tsui_tokens")):
    macro(_name, f"{P2CFG} {_key} (= the 4B Exp 1A/1B test run config)")(
        lambda k=_key: fnum(same(field(jload(P2CFG), k, src=P2CFG), field(exp01_test_4b(), k, src="4B test config"),
                                 f"{k}: dev pilot vs 4B test run")))


@macro("MaxTokSteer", "src/steer/pilot.py CONFIRM_TOKENS (= Exp 3 test config max_new)")
def _():
    return fnum(same(const("src/steer/pilot.py", "CONFIRM_TOKENS"), steer_test()[0]["max_new"], "steering budget"))


@macro("MaxTokExpFive", "src/ttt/test.py MAX_TOKENS (= Exp 5 test config max_tokens = p2 serve config max_tokens)")
def _():
    v = const("src/ttt/test.py", "MAX_TOKENS")
    same(field(jload(TTT_SERVE), "max_tokens", src=TTT_SERVE), v, "p2 serve max_tokens")
    return fnum(same(field(ttt_test()[0], "max_tokens", src="Exp 5 test config"), v, "Exp 5 test max_tokens"))


@macro("MaxTokSelfPred", "src/ttt/test.py SELFPRED_TOKENS (= Exp 5 test config selfpred_tokens)")
def _():
    v = const("src/ttt/test.py", "SELFPRED_TOKENS")
    return fnum(same(field(ttt_test()[0], "selfpred_tokens", src="Exp 5 test config"), v, "selfpred budget"))


for _name, _key in (("ThinkTemp", "temperature"), ("ThinkTopP", "top_p"), ("ThinkTopK", "top_k"), ("ThinkMinP", "min_p")):
    macro(_name, f"src/exp/thinking.py SAMPLING['{_key}'] (= 4B thinking test configs)")(
        lambda k=_key: fnum(thinking_same("sampling", const("src/exp/thinking.py", "SAMPLING"), "SAMPLING")[k]))


@macro("ThinkBudget", f"src/exp/thinking.py --max-tokens default (= {CHECKS_JSON} thinking.chosen_budget = test configs)")
def _():
    v = argparse_default("src/exp/thinking.py", "--max-tokens")
    same(field(jload(CHECKS_JSON), "thinking", "chosen_budget", src=CHECKS_JSON), v, "chosen thinking budget")
    return fnum(thinking_same("max_tokens", v, "--max-tokens default"))


@macro("ThinkBudgetGrid", f"{CHECKS_JSON} thinking.truncation_share.all (budgets tried; = {CHECKS_MD} section 2 columns)",
       replaced=f"{CHECKS_JSON} (the data behind {CHECKS_MD} section 2), checked against the report's columns")
def _():
    grid = sorted(int(k) for k in field(jload(CHECKS_JSON), "thinking", "truncation_share", "all", src=CHECKS_JSON))
    t = md_table(CHECKS_MD, ["Dataset", "Items", "Mean output tokens"], heading="Thinking-mode budget")
    cols = sorted(int(m) for c in t["header"] for m in re.findall(r"^Truncated at (\d+) %$", c))
    same(cols, grid, "budget grid: report columns vs JSON")
    return fset(grid)


@macro("ThinkTruncMax", "analysis/p3_checks.py TRUNC_MAX x 100 (rule stated in reports/03_checks_tables.md section 2)",
       replaced="analysis/p3_checks.py TRUNC_MAX, the constant the report's rule line is generated from")
def _():
    v = pct(const("analysis/p3_checks.py", "TRUNC_MAX"))
    same(minimal(Decimal(checks_md_rule())), v, "truncation rule: report vs TRUNC_MAX")
    return fnum(v)


macro("NThinkSeeds", "src/exp/thinking.py N_SEEDS (= thinking test configs n_seeds)")(
    lambda: fnum(thinking_same("n_seeds", const("src/exp/thinking.py", "N_SEEDS"), "N_SEEDS")))
macro("ThinkPerStratum", "src/exp/thinking.py --per-stratum default (= thinking test configs per_stratum)")(
    lambda: fnum(thinking_same("per_stratum", argparse_default("src/exp/thinking.py", "--per-stratum"), "--per-stratum")))


def think_n(stratum):
    d = jload("analysis/out/p4_4B_thinking.json")
    v = field(d, "n_by_a0", stratum, src="analysis/out/p4_4B_thinking.json")
    subs = [jload(p) for p in rglob("results/main/4B/main-thinking-test-c*-4B-*/subset.json")]
    need(len(subs) == len(d["runs"]), "thinking: chunk runs in the analysis vs subset files")
    for s in subs:
        same(field(s, "weights", stratum, "n", src="subset.json"), v, f"thinking subset.json n ({stratum})")
    same(sum(len(s["chunk_items"]) for s in subs), d["n_items"], "thinking chunk items vs n_items")
    return fnum(v)


_think_repl = ("analysis/out/p4_4B_thinking.json n_by_a0 (both chunks c0, c1 ran), checked against each chunk's "
               "subset.json weights.<stratum>.n; NUMBERS_NEEDED's results/p4/.../metrics.json has no per-stratum count")
macro("NThinkWrong", "analysis/out/p4_4B_thinking.json n_by_a0['False']", replaced=_think_repl)(lambda: think_n("False"))
macro("NThinkRight", "analysis/out/p4_4B_thinking.json n_by_a0['True']", replaced=_think_repl)(lambda: think_n("True"))


# Conditions, verifiers, readout
@macro("FThreeLevels", f"src/exp/runner.py P_LEVELS (= {P6C} P_LEVELS)")
def _():
    v = const("src/exp/runner.py", "P_LEVELS")
    return fset(same(const(P6C, "P_LEVELS"), v, "P_LEVELS runner vs P6"))


@macro("KPerm", f"src/data/verifiers.py K_PERM (= {P6C} K_PERM)")
def _():
    return fnum(same(const(P6C, "K_PERM"), const("src/data/verifiers.py", "K_PERM"), "K_PERM verifiers vs P6"))


@macro("LengthMatchTol", "tests/test_render.py assert abs(la - lb) <= N (= design 'length-matched within +-N tokens')")
def _():
    v = int(regex1("tests/test_render.py", r"assert abs\(la - lb\) <= (\d+)"))
    same(int(regex1("docs/research_design.md", r"length-matched within ±(\d+) tokens")), v, "design tolerance")
    return fnum(v)


@macro("NAltWrapperSubset", "4B Exp 1B alt test run config subset (= configs/blocks.yaml 4B-exp1b-alt chunk)",
       replaced="results/main/4B/main-exp01-test-alt-4B-*/config.json `subset` (NUMBERS_NEEDED's results/p4/... does not exist)")
def _():
    return fnum(same(field(alt_test_4b(), "subset", src="alt config"), blocks()["4B-exp1b-alt"]["chunk"], "alt subset vs blocks.yaml"))


_robust_repl = ("the block ran: results/main/4B/main-exp1c-robust-test-c*-4B-*/config.json (all chunk configs agree), "
                "checked against src/exp/channel.py; blocks.yaml has no such comment")


@macro("NSampledSeeds", "exp1c-robust test configs n_sample_seeds (= src/exp/channel.py N_SAMPLE_SEEDS)", replaced=_robust_repl)
def _():
    return fnum(same(robust_field("n_sample_seeds"), const("src/exp/channel.py", "N_SAMPLE_SEEDS"), "sampled seeds"))


def robust_sampling(key):
    s = same(robust_field("sampling"), const("src/exp/channel.py", "SAMPLING"), "robust sampling vs channel.py")
    return fnum(field(s, key, src="sampling"))


macro("SampledTemp", "exp1c-robust test configs sampling.temperature", replaced=_robust_repl)(lambda: robust_sampling("temperature"))
macro("SampledTopP", "exp1c-robust test configs sampling.top_p", replaced=_robust_repl)(lambda: robust_sampling("top_p"))
macro("NRobustSubset", "exp1c-robust test configs subset", replaced=_robust_repl)(lambda: fnum(robust_field("subset")))


@macro("NMaskRobustSeeds", "exp1c-robust test configs mask_seeds, minus the primary seed (f3_verdict default seed)",
       replaced="run config mask_seeds (= src/exp/channel.py MASK_SEEDS) without the primary seed, the default of "
                "src/data/verifiers.py f3_verdict(seed=...)")
def _():
    seeds = same(robust_field("mask_seeds"), list(const("src/exp/channel.py", "MASK_SEEDS")), "mask seeds")
    primary = func_default("src/data/verifiers.py", "f3_verdict", "seed")
    need(primary in seeds, "primary verifier seed not among the run's mask seeds")
    return fnum(len([s for s in seeds if s != primary]))


@macro("TStarFourB", f"{SV} 4B.t_star_s, 2 decimals as reports/02_selfverdict_tables.md (= 4B test config self_threshold_s)")
def _():
    v = selfverdict("t_star_s")
    same(exp01_test_4b()["self_threshold_s"], v, "t* vs 4B test config")
    same(fnum(v, 2), regex1("reports/02_selfverdict_tables.md", r"## Qwen3-4B\s+.*?t\* = ([\d.]+)", re.S), "t* vs report")
    return fnum(v, 2)


@macro("TStarFourBPyes", f"{SV} 4B.t_star_p_yes_norm, through the first digit after the leading 9s plus one")
def _():
    return near_one(selfverdict("t_star_p_yes_norm"))


@macro("SVBalAccTStar", f"{SV} 4B.balanced_acc_t_star, 3 decimals as reports/02_selfverdict_tables.md")
def _():
    return fnum(selfverdict("balanced_acc_t_star"), 3)


@macro("SVBalAccDefault", f"{SV} 4B.balanced_acc_half, 3 decimals as reports/02_selfverdict_tables.md")
def _():
    return fnum(selfverdict("balanced_acc_half"), 3)


@macro("SVDefaultThreshold", "analysis/p2_selfverdict.py t_half = log(a) - log(b), i.e. P(yes)_norm >= a / (a + b)")
def _():
    a, b = regex1("analysis/p2_selfverdict.py", r"t_half = math\.log\(([\d.]+)\) - math\.log\(([\d.]+)\)")
    return fnum(minimal(Decimal(a) / (Decimal(a) + Decimal(b))))


# Probe (Exp 2)
@macro("ProbeCGrid", f"src/probe/train.py C_GRID (= the C values in {PROBE_SEL} grid)")
def _():
    v = const("src/probe/train.py", "C_GRID")
    same(sorted({g["C"] for g in field(jload(PROBE_SEL), "grid", src=PROBE_SEL)}), sorted(v), "probe C grid")
    return fset(v)


macro("ProbeFolds", "src/probe/train.py FOLDS")(lambda: fnum(const("src/probe/train.py", "FOLDS")))


def probe_sel(*path):
    cfg = test_cfg("4B/main-probe-scores-test-4B-*", "4B probe-scores test run")
    need(Path(PROBE_SEL).parent.as_posix() in cfg["probe"], f"4B test probe {cfg['probe']} is not from {PROBE_SEL}")
    return field(jload(PROBE_SEL), *path, src=PROBE_SEL)


macro("NProbeDev", f"{PROBE_SEL} n_dev")(lambda: fnum(probe_sel("n_dev")))
macro("ProbeLayerFourB", f"{PROBE_SEL} selected.layer")(lambda: fnum(probe_sel("selected", "layer")))
macro("ProbeCFourB", f"{PROBE_SEL} selected.C")(lambda: fnum(probe_sel("selected", "C")))


# Steering (Exp 3)
@macro("SteerSkipTokens", "src/steer/steer.py SKIP_FIRST (= Exp 3 test config skip_first)")
def _():
    return fnum(same(const("src/steer/steer.py", "SKIP_FIRST"), steer_test()[0]["skip_first"], "SKIP_FIRST"))


def layer_range():
    lo, top = regex1("src/steer/pilot.py", r"layers = list\(range\((\d+), n_layers - (\d+), LAYER_STEP\)\)")
    layers = steer_pilot_metrics()["layers"]
    same(min(layers), int(lo), "lowest screened layer vs range bound")
    return int(lo), int(top)


macro("SteerMinBlock", "src/steer/pilot.py range(lo, n_layers - k, LAYER_STEP): lo (= min of pilot metrics layers)")(
    lambda: fnum(layer_range()[0]))
macro("SteerTopMargin", "src/steer/pilot.py range(lo, n_layers - k, LAYER_STEP): exclusive end, so k + 1")(
    lambda: fnum(layer_range()[1] + 1))


@macro("ReachKLThreshold", "src/steer/pilot.py KL_NOISE (= Exp 3 test config kl_noise)")
def _():
    return fnum(same(const("src/steer/pilot.py", "KL_NOISE"), steer_test()[0]["kl_noise"], "KL_NOISE"))


@macro("NRandomDirections", "len(src/steer/pilot.py RANDOM_SEEDS) (= Exp 3 test config random_seeds)")
def _():
    return fnum(len(same(list(const("src/steer/pilot.py", "RANDOM_SEEDS")), steer_test()[0]["random_seeds"], "RANDOM_SEEDS")))


macro("NSteerConfirm", f"len({STEER_PILOT}/metrics.json top3)")(lambda: fnum(len(steer_pilot_metrics()["top3"])))


def main_setting(key):
    v = field(jload(TIER2), "exp3", "4B", "main_setting", key, src=TIER2)
    return same(steer_test()[0]["main"][key], v, f"main setting {key}: tier2 vs Exp 3 test config")


macro("SteerLayer", f"{TIER2} exp3.4B.main_setting.layer")(lambda: fnum(main_setting("layer")))
macro("SteerAlpha", f"{TIER2} exp3.4B.main_setting.alpha")(lambda: fnum(main_setting("alpha")))
_sec_repl = ("results/main/4B/main-steer-test-4B-*/config.json `secondary` (the settings run on test), checked against "
             "the pilot top3 minus the main setting")
macro("SteerSecLayerA", "Exp 3 test config secondary, role, layer", replaced=_sec_repl)(lambda: fnum(secondary("role", "layer")))
macro("SteerSecAlphaA", "Exp 3 test config secondary, role, alpha", replaced=_sec_repl)(lambda: fnum(secondary("role", "alpha")))
macro("SteerSecLayerB", "Exp 3 test config secondary, label, layer", replaced=_sec_repl)(lambda: fnum(secondary("label", "layer")))
macro("SteerSecAlphaB", "Exp 3 test config secondary, label, alpha", replaced=_sec_repl)(lambda: fnum(secondary("label", "alpha")))


@macro("SteerDevGain", f"{TIER2} exp3.4B.main_setting.w2r_gain_over_ref x 100, 1 decimal as reports/02_pilot_tier2_tables.md")
def _():
    return fnum(pct(main_setting("w2r_gain_over_ref")), 1)


@macro("NSteerDevItems", f"{STEER_PILOT}/confirm_rows.jsonl: D-inj items of the main-setting arm",
       replaced=f"{STEER_PILOT}/confirm_rows.jsonl (the report and metrics.json give rates, not the item count)")
def _():
    arm = field(jload(TIER2), "exp3", "4B", "main_run", src=TIER2)
    rows = jlines(STEER_PILOT + "/confirm_rows.jsonl")
    ids = [r["item_id"] for r in rows if r["run"] == arm and r["data"] == "dinj"]
    need(ids and len(ids) == len(set(ids)), f"confirm_rows: no or duplicate D-inj rows for arm {arm}")
    same(len([r for r in rows if r["run"] == "unsteered" and r["data"] == "dinj"]), len(ids), "items per arm")
    return fnum(len(ids))


@macro("SteerFirstCandGainGeneric", f"{STEER_PILOT}/metrics.json selection[0].w2r_gain_over_ref x 100 (generic valid), 1 decimal")
def _():
    s = steer_pilot_metrics()["selection"]
    need(s and s[0]["generic_valid"] is True, "first candidate's generic control is not valid")
    need(sum(x["generic_valid"] for x in s) == 1, "more than one candidate has a valid generic control")
    return fnum(pct(s[0]["w2r_gain_over_ref"]), 1)


def steer_items(key):
    return fnum(field(steer_test()[1], "items", key, src="Exp 3 test metrics"))


_steer_repl = "results/main/4B/main-steer-test-4B-bd96f2f/metrics.json items (the run exists; results/p4/... does not)"
macro("NSteerTest", "Exp 3 test metrics items.dinj", replaced=_steer_repl)(lambda: steer_items("dinj"))
macro("NSteerExcluded", "Exp 3 test metrics items.dinj_excluded_no_content", replaced=_steer_repl)(
    lambda: steer_items("dinj_excluded_no_content"))


@macro("NSteerRtoW", "Exp 3 test metrics items.dnat_rtow (= Exp 3 test config n_rtow)", replaced=_steer_repl)
def _():
    return fnum(same(field(steer_test()[1], "items", "dnat_rtow", src="Exp 3 metrics"), steer_test()[0]["n_rtow"], "R->W items"))


# Operator-level arm (Exp 5)
def ttt_const(rel, name, cfg_rel, cfg_key):
    v = const(rel, name)
    same(field(jload(cfg_rel), cfg_key, src=cfg_rel), list(v) if isinstance(v, tuple) else v, f"{name} vs {cfg_rel}")
    return v


macro("LoraRank", f"src/ttt/update.py RANK (= {TTT_PILOT} rank)")(lambda: fnum(ttt_const("src/ttt/update.py", "RANK", TTT_PILOT, "rank")))
macro("LoraAlpha", f"src/ttt/update.py ALPHA (= {TTT_PILOT} alpha)")(lambda: fnum(ttt_const("src/ttt/update.py", "ALPHA", TTT_PILOT, "alpha")))


@macro("TTTSteps", f"src/ttt/pilot.py STEPS (= {TTT_PILOT} steps = Exp 5 test config steps)")
def _():
    v = ttt_const("src/ttt/pilot.py", "STEPS", TTT_PILOT, "steps")
    return fnum(same(ttt_test()[0]["steps"], v, "Exp 5 test steps"))


def tier2_exp5(key):
    v = field(jload(TIER2), "exp5", "4B", key, src=TIER2)
    same(field(ttt_test()[0], "tier2", key, src="Exp 5 test config"), v, f"Exp 5 {key}: tier2 vs test config")
    return fnum(v)


macro("TTTLrFourB", f"{TIER2} exp5.4B.lr (= Exp 5 test config tier2.lr)")(lambda: tier2_exp5("lr"))
macro("TTTTau", f"{TIER2} exp5.4B.tau (= Exp 5 test config tier2.tau)")(lambda: tier2_exp5("tau"))
macro("TTTTauGrid", f"src/ttt/pilot.py TAU_GRID (= {TTT_PILOT} tau_grid)")(
    lambda: fset(ttt_const("src/ttt/pilot.py", "TAU_GRID", TTT_PILOT, "tau_grid")))
macro("TTTKLMax", f"src/ttt/pilot.py KAPPA (= {TTT_PILOT} kappa)")(
    lambda: fnum(ttt_const("src/ttt/pilot.py", "KAPPA", TTT_PILOT, "kappa")))
macro("TTTLrGrid", f"src/ttt/pilot.py LR_GRID (= {TTT_PILOT} lr_grid)")(
    lambda: fset(ttt_const("src/ttt/pilot.py", "LR_GRID", TTT_PILOT, "lr_grid")))
macro("NTTTTuneItems", f"src/ttt/pilot.py N_TUNE (= {TTT_PILOT} n_tune)")(
    lambda: fnum(ttt_const("src/ttt/pilot.py", "N_TUNE", TTT_PILOT, "n_tune")))
@macro("TTTcFlag", "src/ttt/update.py anomaly rule '\"c_gt_N\" if c > N'")
def _():
    label, bound = regex1("src/ttt/update.py", r'"c_gt_(\d+)" if c > (\d+)')
    return fnum(int(same(bound, label, "c-flag bound vs anomaly label")))


@macro("NormTolPct", "analysis/p2_tier2.py feasibility check realized_norm_max_rel_err <= x, x 100 (= design 'within N% of tau')",
       replaced="analysis/p2_tier2.py, where the feasibility check is coded (src/ttt/pilot.py only records the error)")
def _():
    v = pct(regex1("analysis/p2_tier2.py", r'pm\["realized_norm_max_rel_err"\] <= ([\d.]+)'))
    same(minimal(Decimal(regex1("docs/research_design.md", r"realized norm within (\d+)% of τ"))), v, "design tolerance")
    return fnum(v)


@macro("NExpFivePool", "Exp 5 test config n_items (= Exp 5 test metrics n_items)",
       replaced="results/main/4B/main-ttt-test-4B-eda727a/config.json n_items (the run exists; results/p4/... does not)")
def _():
    c, m = ttt_test()
    return fnum(same(field(c, "n_items", src="Exp 5 config"), field(m, "n_items", src="Exp 5 metrics"), "Exp 5 pool"))


macro("NExpFiveSamples", "src/ttt/test.py N_SAMPLES (= Exp 5 test config n_samples)")(
    lambda: fnum(same(const("src/ttt/test.py", "N_SAMPLES"), ttt_test()[0]["n_samples"], "N_SAMPLES")))
macro("ExpFiveSampleTemp", "src/ttt/test.py T_SAMPLE (= Exp 5 test config t_sample)")(
    lambda: fnum(same(const("src/ttt/test.py", "T_SAMPLE"), ttt_test()[0]["t_sample"], "T_SAMPLE")))
_stream_repl = ("the block ran: results/main/4B/main-streams-test-4B-f15b933/config.json (steps, seeds); "
                "NUMBERS_NEEDED cites the plan (design, blocks.yaml comment), which it matches")


def stream_cfg():
    return test_cfg("4B/main-streams-test-4B-*", "4B streams test run")


macro("NStreamItems", "streams test config steps", replaced=_stream_repl)(lambda: fnum(field(stream_cfg(), "steps", src="streams config")))
macro("NStreamSeeds", "len(streams test config seeds)", replaced=_stream_repl)(lambda: fnum(len(field(stream_cfg(), "seeds", src="streams config"))))


# Scorer and invariants
@macro("ScorerVersion", "src/score/scorer.py SCORER_VERSION (= scorer_version of every test-run config that records it)",
       replaced="test-run configs (manifests record the scorer's sha256, not its version)")
def _():
    v = const("src/score/scorer.py", "SCORER_VERSION")
    vs = {jload(p)["scorer_version"] for p in rglob("results/main/*/*/config.json") if "scorer_version" in jload(p)}
    need(vs == {v}, f"test-run scorer versions {sorted(vs)} vs SCORER_VERSION {v}")
    return esc(v)


@macro("NScorerTests", "pytest --collect-only -q tests/test_scorer.py: collected tests")
def _():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                        "tests/test_scorer.py"], cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)
    m = re.findall(r"^(\d+) tests? collected", r.stdout, re.M)
    need(r.returncode == 0 and len(m) == 1, f"pytest collection failed (exit {r.returncode}): {r.stdout[-300:]}")
    return fnum(int(m[0]))


macro("MVFallbackChars", "src/score/scorer.py MAX_SYMBOLIC_CHARS")(lambda: fnum(const("src/score/scorer.py", "MAX_SYMBOLIC_CHARS")))
macro("GuardCPUSeconds", "src/score/scorer.py GUARD_CPU_S")(lambda: fnum(const("src/score/scorer.py", "GUARD_CPU_S")))
macro("GuardMemGiB", "src/score/scorer.py GUARD_MEM_GB")(lambda: fnum(const("src/score/scorer.py", "GUARD_MEM_GB")))


@macro("ScorerAgreeTarget", f"docs/research_design.md section 8 Scorer row (= {AGREE_MD} header)")
def _():
    v = regex1("docs/research_design.md", r"^\| Scorer \|[^\n]*?≥ (\d+)% on dev", re.M)
    same(regex1(AGREE_MD, r"Invariant \(design §8\): ≥ (\d+) %"), v, "scorer target: design vs report")
    return fnum(v)


macro("ScorerAgreeFirst", f"{AGREE_JSON} rate.v1 x 100, 2 decimals (= {AGREE_MD} row all)",
      replaced=f"{AGREE_JSON} (the data of {AGREE_MD}), checked against the report")(lambda: agree_rate("v1"))
macro("ScorerAgreeVTwo", f"{AGREE_JSON} rate.v2 x 100, 2 decimals (= {AGREE_MD} row all)",
      replaced=f"{AGREE_JSON} (the data of {AGREE_MD}), checked against the report")(lambda: agree_rate("v2"))
macro("ScorerErrCriterion", f"{AGREE_MD} amended invariant '(limit N %)'")(
    lambda: fnum(regex1(AGREE_MD, r"amended invariant (?:passes|fails) \(limit ([\d.]+) %\)")))


@macro("ScorerErrBound", f"{AGREE_JSON} amended_bound x 100, 2 decimals (= {AGREE_MD} 'N % of all texts')",
       replaced=f"{AGREE_JSON} amended_bound, checked against the report")
def _():
    s = fnum(pct(field(jload(AGREE_JSON), "amended_bound", src=AGREE_JSON)), 2)
    return same(s, regex1(AGREE_MD, r"in the disagreements: ([\d.]+) % of all texts"), "bound: JSON vs report")


macro("ScorerErrBoundLevel", f"{AGREE_MD} 'one-sided N % upper bound'")(
    lambda: fnum(regex1(AGREE_MD, r"one-sided (\d+) % upper bound")))


@macro("NScorerHandcheck", "reports/03_handcheck.md")
def _():
    raise Missing("ambiguous: reports/03_handcheck.md states no single count. Hand-checked disagreements were the v2 "
                  "sample of 40 'accepts' (the bound's sample), the 1 'rejects' case (both in analysis/"
                  "p3_scorer_agreement.py HANDCHECK: 40 + 1) and the v1 sample of 30 (03_scorer_agreement.md); the "
                  "sentence could mean 40, 41 or 71. Name the field (e.g. HANDCHECK accepts + rejects 'checked')")


@macro("HFvLLMTarget", "docs/research_design.md Exp 5 backends '>= N% identical greedy outputs' (= reports/03_checks_tables.md section 1)")
def _():
    v = regex1("docs/research_design.md", r"≥ (\d+)% identical greedy outputs")
    same(regex1(CHECKS_MD, r"≥ (\d+) % identical"), v, "HF/vLLM target: design vs report")
    return fnum(v)


def hf_vllm(key, col):
    s = fnum(pct(field(jload(CHECKS_JSON), "hf_vllm", key, src=CHECKS_JSON)), 1)
    t = md_table(CHECKS_MD, ["Items", "Identical", "Identical %"], heading="HF vs vLLM")
    same(s, md_cell(t, {}, col), f"HF/vLLM {key}: JSON vs report")
    return s


_hf_repl = f"{CHECKS_JSON} hf_vllm (the data of {CHECKS_MD} section 1), checked against the report"
macro("HFvLLMIdentical", f"{CHECKS_JSON} hf_vllm.rate x 100, 1 decimal", replaced=_hf_repl)(
    lambda: hf_vllm("rate", "Identical %"))
macro("HFvLLMMedianPos", f"{CHECKS_JSON} hf_vllm.position_agreement_median x 100, 1 decimal", replaced=_hf_repl)(
    lambda: hf_vllm("position_agreement_median", "Per-position agreement, median %"))


# Statistics, margins, power
macro("NBoot", f"{P6C} N_BOOT")(lambda: fnum(const(P6C, "N_BOOT")))
macro("CILevel", "analysis/stats.py boot_ci(level=...) x 100")(lambda: fnum(pct(func_default("analysis/stats.py", "boot_ci", "level"))))
macro("AlphaFamily", f"{P6C} ALPHA")(lambda: fnum(const(P6C, "ALPHA")))


@macro("NFamily", f"len({P6C} FAMILY) (= analysis/p3_min_n.py FAMILY)")
def _():
    return fnum(same(len(const(P6C, "FAMILY")), const("analysis/p3_min_n.py", "FAMILY"), "family size P6 vs P3"))


@macro("DeltaMargin", f"{P6C} DELTA x 100 (= analysis/p6_results.py DELTA in pp)")
def _():
    v = pct(const(P6C, "DELTA"))
    same(minimal(dec(const("analysis/p6_results.py", "DELTA"))), v, "delta: P6 confirmatory vs P6 results")
    return fnum(v)


macro("DeltaAUC", f"{P6C} DELTA_AUC")(lambda: fnum(const(P6C, "DELTA_AUC")))


@macro("DeltaW", "analysis/p6_results.py DELTA_W, pp (= prereg.md H2b row 'delta_w = N pp')")
def _():
    v = minimal(dec(const("analysis/p6_results.py", "DELTA_W")))
    found = re.findall(r"δ_w = (\d+(?:\.\d+)?) pp", read("prereg.md"))
    need(found, "delta_w not stated in prereg.md")
    for s in found:
        same(minimal(Decimal(s)), v, "delta_w in prereg.md vs analysis/p6_results.py")
    return fnum(v)


@macro("PowerTarget", "analysis/p3_min_n.py POWER (= prereg.md 'power ... >= N')")
def _():
    v = const("analysis/p3_min_n.py", "POWER")
    need(re.search(r"power (?:is )?≥ " + re.escape(str(v)), read("prereg.md")), "power target not found in prereg.md")
    return fnum(v)


macro("NNullSims", "tests/test_stats.py SIMS")(lambda: fnum(const("tests/test_stats.py", "SIMS")))
macro("NullTol", "tests/test_stats.py TOL x 100")(lambda: fnum(pct(const("tests/test_stats.py", "TOL"))))


@macro("HOneaDevDiscordance", "reports/03_power_tier1_tables.md, Qwen3-4B, H1a table, pooled, 'Pilot discordant mass (mean abs d, %)'")
def _():
    t = md_table("reports/03_power_tier1_tables.md", ["Scope", "Pilot discordant mass (mean abs d, %)"],
                 heading="Qwen3-4B", context=r"H1a")
    vals = {r["Pilot discordant mass (mean abs d, %)"] for r in t["rows"] if r["Scope"] == "pooled"}
    need(len(vals) == 1, f"H1a pooled discordance: {vals}")
    return fnum(numstr(vals.pop()))


# Dates and hashes
macro("MatrixSearchDate", "paper/related_work_matrix.md 'Search date'")(
    lambda: regex1("paper/related_work_matrix.md", r"^Search date: (\d{4}-\d{2}-\d{2})", re.M))


@macro("FreezeDate", "prereg.md line 1 'FROZEN date' (= DEVIATIONS.md 'frozen at commit ... on date')")
def _():
    v = regex1("prereg.md", r"\A[^\n]*FROZEN (\d{4}-\d{2}-\d{2})")
    same(regex1("DEVIATIONS.md", r"frozen at commit [0-9a-f]+ on (\d{4}-\d{2}-\d{2})"), v, "freeze date")
    return v


@macro("FirstTestAccess", "runs.jsonl p4-first-test-access-4B-* start (= STATUS.md)")
def _():
    rid = one([k for k in runs() if k.startswith("p4-first-test-access-4B-")], "first-test-access run in runs.jsonl")
    start = runs()[rid]["start"]
    same(regex1("STATUS.md", r"first post-feedback access: (\S+)"), start, "first test access: runs.jsonl vs STATUS.md")
    return utc(start).strftime("%Y-%m-%d %H:%M UTC")


@macro("WordingHashMain", f"{P2CFG} wording_sha256 (= every main-label test config and the exp1c-robust configs)")
def _():
    v = fhash(field(jload(P2CFG), "wording_sha256", src=P2CFG))
    cs = [jload(p) for p in rglob("results/main/*/main-exp01-test-*/config.json") if jload(p).get("label_set") == "main"]
    need(cs, "no main-label test config")
    for c in cs + robust_cfgs():
        same(c["wording_sha256"], v, "main wording hash: dev pilot vs test run")
    return v


@macro("WordingHashAlt", "4B Exp 1B alt test config wording_sha256 (= prereg.md)")
def _():
    v = fhash(field(alt_test_4b(), "wording_sha256", src="alt config"))
    need(v in read("prereg.md"), "alt wording hash not in prereg.md")
    return v


@macro("ContextTemplateHash", f"{TTT_SERVE} context_template_sha256 (= Exp 5 test config)")
def _():
    v = fhash(field(jload(TTT_SERVE), "context_template_sha256", src=TTT_SERVE))
    return same(ttt_test()[0]["context_template_sha256"], v, "context template hash: dev serve vs test run")


# Cost and compute
@macro("CloudCapUSD", "CLAUDE.md section 6 CLOUD_CAP_USD (= cloud/session.py CAP_USD)")
def _():
    v = Decimal(regex1("CLAUDE.md", r"^CLOUD_CAP_USD\s*=\s*(\d+(?:\.\d+)?)", re.M))
    same(minimal(dec(const("cloud/session.py", "CAP_USD"))), minimal(v), "cap: CLAUDE.md vs cloud/session.py")
    return fnum(minimal(v))


macro("ReservePct", "CLAUDE.md section 6 RESERVE_USD = f x CLOUD_CAP_USD, f x 100")(
    lambda: fnum(pct(regex1("CLAUDE.md", r"^RESERVE_USD\s*=\s*([\d.]+)\s*×\s*CLOUD_CAP_USD", re.M))))


@macro("CloudGPUHours", "cloud/ledger.jsonl: sum of session hours (every session closed), 1 decimal",
       replaced="cloud/ledger.jsonl, every session closed; checked against billing by Liam (2026-11-08)")
def _():
    return fnum(sum(dec(r["hours"]) for r in ledger()), 1)


@macro("CloudSpentUSD", "cloud/ledger.jsonl sum of compute_usd + storage accrued (cloud/session.py spent() formula) from "
       "CREATED to the last session's pause_time, 2 decimals",
       replaced="cloud/ledger.jsonl compute (every session closed) + storage accrued by cloud/session.py's formula until the "
                "instance was destroyed (cloud/destroyed.json): the final cost, checked against billing by Liam")
def _():
    rows = ledger()
    gbs = {r["storage_gb"] for r in rows}
    need(len(gbs) == 1, f"ledger storage_gb varies: {gbs}")
    gb = same(gbs.pop(), const("cloud/session.py", "STORAGE_GB"), "storage GB: ledger vs cloud/session.py")
    end = max(utc(r["pause_time"]) for r in rows)
    dj = ROOT / "cloud" / "destroyed.json"   # after destruction, storage stops at the confirmed time (final cost)
    if dj.exists():
        end = utc(json.loads(dj.read_text(encoding="utf-8"))["confirmed_absent_at"])
    months = dec((end - created()).total_seconds()) / Decimal(30 * 86400)
    storage = dec(gb) * dec(const("cloud/session.py", "STORAGE_USD_PER_GB_MONTH")) * months
    return fnum(sum(dec(r["compute_usd"]) for r in rows) + storage, 2)


@macro("LocalGPUHours", "no machine-readable source")
def _():
    raise Missing("no machine-readable source: STATUS.md's 'local GPU-h' is hand-maintained (NUMBERS_NEEDED: P8 must "
                  "name a source or the sentence goes)")


# ----------------------------------------------------------------------------------------------------------------
# Reproducibility table

def lock_versions():
    out = {}
    for line in read("configs/lock.txt").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)$", line.strip())
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def sha256_lf(rel):
    return hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def tab_revisions():
    # Test runs of the main study and of the second-family follow-up (results/fu2, DEVIATIONS D15).
    mans = [jload(p) for p in rglob("results/main/*/*/manifest.json") + rglob("results/fu2/main/*/*/manifest.json")]
    need(mans, "no test-run manifest")
    models = yaml.safe_load(read("configs/models.yaml"))

    def common(key, ms=mans):
        vals = {json.dumps(m.get(key)) for m in ms}
        need(len(vals) == 1, f"test-run manifests disagree on {key}: {sorted(vals)}")
        return json.loads(vals.pop())

    scorer, lock, data_man, py, gpu = (common(k) for k in ("scorer_sha256", "lockfile_sha256",
                                                           "data_build_manifest_sha256", "python", "gpu"))
    # The chat template is per model: one hash for the three Qwen3 models, another for Tulu 3.
    tmpl_by = {}
    for key, m in models["models"].items():
        t = common("chat_template_sha256", [x for x in mans if x["model"] == m["repo"]] or [{}])
        need(t and t.startswith(m["chat_template_sha256_16"]), f"{m['repo']}: template hash prefix differs")
        tmpl_by.setdefault(t, []).append(m["repo"].split("/")[-1])
    same(sha256_lf("configs/lock.txt"), lock, "configs/lock.txt sha256 vs manifests' lockfile_sha256")
    same(sha256_lf("src/score/scorer.py"), scorer, "src/score/scorer.py sha256 vs manifests' scorer_sha256")
    same(sha256_lf(BUILD), data_man, f"{BUILD} sha256 vs manifests' data_build_manifest_sha256")
    rows_models = []
    for key, m in models["models"].items():
        revs = {x["model_revision"] for x in mans if x["model"] == m["repo"]}
        need(revs <= {m["revision"]}, f"{m['repo']}: manifest revisions {revs} vs configs/models.yaml {m['revision']}")
        rows_models.append((m["repo"].split("/")[-1], m["repo"], m["revision"], len([x for x in mans if x["model"] == m["repo"]])))
    need(len(rows_models) and all(n > 0 for *_, n in rows_models), "a pinned model has no test-run manifest")

    srcs = [v for v in jload(BUILD)["sources"].values()] + [field(jload(EXT), "source", src=EXT)]
    env, lockv = models["environment"], lock_versions()
    same(py, env["python"], "python: manifests vs configs/models.yaml")
    soft = []
    for label, pkg in (("PyTorch", "torch"), ("vLLM", "vllm"), ("Transformers", "transformers"), ("PEFT", "peft"),
                       ("nnsight", "nnsight"), ("datasets", "datasets"), ("math-verify", "math-verify"),
                       ("scikit-learn", "scikit-learn"), ("statsmodels", "statsmodels")):
        v = lockv.get(pkg)
        need(v, f"{pkg} not in configs/lock.txt")
        if pkg in env:
            need(str(env[pkg]).split("+")[0] == v, f"{pkg}: configs/models.yaml {env[pkg]} vs lock {v}")
            v = str(env[pkg])
        soft.append((label, v))
    backends = {}
    for m in mans:
        b = m.get("backend") or {}
        if b.get("name") == "vllm":
            backends.setdefault("vllm", set()).add(b["version"])
        if b.get("name") == "hf-transformers":
            backends.setdefault("transformers", set()).add(b["version"])
    for pkg, vs in backends.items():
        need(vs == {lockv[pkg]}, f"{pkg} versions in manifests {vs} vs lock {lockv[pkg]}")

    splits = {s["split"] for s in srcs}
    need(len(splits) == 1, f"dataset sources use several splits: {splits}")
    tt = lambda s: "\\texttt{" + esc(s) + "}"
    L = ["% tab_revisions.tex -- generated by analysis/p7_numbers.py from configs/models.yaml, configs/lock.txt,",
         f"% {BUILD}, {EXT} and the test-run manifests results/main/*/*/ and results/fu2/main/*/*/manifest.json. Do not edit.",
         "\\begin{table*}[t]", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}",
         "  \\begin{tabular}{@{}ll@{}}", "    \\toprule",
         "    Component & Pinned value \\\\", "    \\midrule",
         "    \\multicolumn{2}{@{}l}{\\emph{Model revisions (bf16; weights and tokenizer from one snapshot)}} \\\\"]
    for name, repo, rev, _ in rows_models:
        L.append(f"    {esc(name)} ({tt(repo)}) & {tt(rev)} \\\\")
    for t, names in tmpl_by.items():
        L.append(f"    Chat template ({esc(', '.join(names))}), sha256 & {tt(t)} \\\\")
    L += ["    \\midrule",
          f"    \\multicolumn{{2}}{{@{{}}l}}{{\\emph{{Dataset revisions ({esc(splits.pop())} splits)}}}} \\\\"]
    for s in srcs:
        cfg = f", {esc(s['config'])}" if isinstance(s.get("config"), str) else ""
        L.append(f"    {esc(s['name'])} ({tt(s['repo'])}{cfg}) & {tt(s['revision'])} \\\\")
    L += [f"    Data build manifest, sha256 & {tt(data_man)} \\\\", "    \\midrule",
          "    \\multicolumn{2}{@{}l}{\\emph{Code and environment}} \\\\",
          f"    Lockfile, sha256 & {tt(lock)} \\\\",
          f"    Scorer {esc(const('src/score/scorer.py', 'SCORER_VERSION'))}, sha256 & {tt(scorer)} \\\\",
          f"    Python & {esc(py)} \\\\"]
    for label, v in soft:
        L.append(f"    {esc(label)} & {esc(v)} \\\\")
    L += [f"    GPU (every test run) & {esc(gpu)} \\\\", "    \\bottomrule", "  \\end{tabular}",
          f"  \\caption{{Pinned versions. Model revisions and the sha256 hashes of the chat template, the data build "
          f"manifest ({tt(BUILD)}), the lockfile ({tt('configs/lock.txt')}) and the scorer "
          f"({tt('src/score/scorer.py')}) are those recorded in the version manifest of each of the {fnum(len(mans))} "
          f"test runs (identical across runs, except the chat template, which is identical across the runs of each "
          f"model family; Tulu 3 is the exploratory second family, Appendix~\\ref{{app:followups}}); dataset revisions "
          f"from the data build manifests; software versions from "
          f"the lockfile. Generated by {tt('analysis/p7_numbers.py')}.}}",
          "  \\label{tab:revisions}", "\\end{table*}", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------------------

def needed_macros():
    names = re.findall(r"^\| `\\(\w+)` \|", read(NEEDED), re.M)
    ph = re.findall(r"^\\providecommand\{\\(\w+)\}", read(PLACEHOLDERS), re.M)
    return names, ph


def main():
    names, ph = needed_macros()
    problems = []
    if len(names) != len(set(names)):
        problems.append("NUMBERS_NEEDED.md lists a macro twice")
    if set(names) != set(ph):
        problems.append(f"NUMBERS_NEEDED.md vs numbers_placeholders.tex differ: {sorted(set(names) ^ set(ph))}")
    extra = sorted(set(GEN) - set(names))
    if extra:
        problems.append(f"generators for macros not in NUMBERS_NEEDED.md: {extra}")

    defined, undefined, errors = [], [], []
    for n in names:
        if n not in GEN:
            undefined.append((n, "no generator in analysis/p7_numbers.py"))
            continue
        f, src = GEN[n]
        try:
            v = f()
            need(isinstance(v, str) and v.strip() and "\n" not in v, f"empty or multi-line value {v!r}")
            defined.append((n, v, src))
        except Missing as e:
            undefined.append((n, str(e)))
        except Exception as e:   # a bug, not a missing source: still leave the macro undefined, but fail the run
            undefined.append((n, f"ERROR {type(e).__name__}: {e}"))
            errors.append(n)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    head = ["% numbers.tex -- generated by analysis/p7_numbers.py (CLAUDE.md rule 1). Do not edit by hand; rerun:",
            "%   PYTHONUTF8=1 .venv/Scripts/python.exe analysis/p7_numbers.py",
            "% Every value is read at run time from the source named in its comment (paper/NUMBERS_NEEDED.md, or the",
            "% newer source listed below). Formatting: value only (units in the text); percentages as numbers; integers",
            "% >= 1000 with a {,} thousands separator; negative numbers start with \\ensuremath{-}; sets as",
            "% \\ensuremath{\\{...\\}}; dates ISO, timestamps YYYY-MM-DD HH:MM UTC; hashes raw full lowercase hex.",
            f"% Defined: {len(defined)} of {len(names)} macros in NUMBERS_NEEDED.md.",
            "%"]
    head.append(f"% UNDEFINED ({len(undefined)}; numbers_placeholders.tex shows a red marker for each):")
    for n, why in undefined:
        head.append(f"%   \\{n}: {why}")
    if not undefined:
        head.append("%   none")
    done = {d[0] for d in defined}
    repl = [n for n in names if n in REPLACED and n in done]
    head += ["%", f"% SOURCES REPLACED (newer or more direct than NUMBERS_NEEDED.md; {len(repl)}):"]
    head += [f"%   \\{n}: {REPLACED[n]}" for n in repl]
    head += [f"% PROBLEM: {p}" for p in problems]
    body = [f"\\newcommand{{\\{n}}}{{{v}}}% {src}" for n, v, src in defined]
    with open(OUT_DIR / "numbers.tex", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(head + ["", *body, ""]))

    tab_status = "written"
    try:
        tab = tab_revisions()
        with open(OUT_DIR / "tab_revisions.tex", "w", encoding="utf-8", newline="\n") as f:
            f.write(tab)
    except Missing as e:
        tab_status = f"NOT written: {e}"

    print(f"paper/generated/numbers.tex: {len(defined)} of {len(names)} macros defined")
    print(f"paper/generated/tab_revisions.tex: {tab_status}")
    print(f"\nUndefined ({len(undefined)}):")
    for n, why in undefined:
        print(f"  \\{n}: {why}")
    print(f"\nSources replaced ({len(repl)}):")
    for n in repl:
        print(f"  \\{n}: {REPLACED[n]}")
    for p in problems:
        print(f"PROBLEM: {p}")
    if errors or problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
