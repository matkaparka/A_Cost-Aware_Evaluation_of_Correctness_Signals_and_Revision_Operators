"""MMLU-Pro scoring audit (post hoc, exploratory; DEVIATIONS D19; third review, review_and_suggestion.md §6.1).

The frozen scorer (src/score/mc.py 1.0.0, hashed in docs/planning/validation_v2/addendum.md) stays primary. This script
does not change it or analysis/mc_results.py; it imports them. Before writing it, the author had looked at five unparsed
Tulu-3-8B F2 revisions (2026-11-08); the rules below were fixed after that look and before this script was first run.

Rules.
- Every scored output is classed by its frozen status: ok, fallback, truncated, unparsed. An output with no letter
  (truncated or unparsed) is further classed:
  (a) letter-recoverable: the content of the last complete \\boxed{...}, after the frozen normalization, starts with
      one valid option letter, optionally in parentheses, followed by the end of the string or, after optional spaces,
      by one of , ; : . ) - – —. (The frozen scorer accepts the letter only alone or followed by . ) or :, so it misses
      "H, with reservations ...".) A letter followed by a space and a word ("A patient ...") is not recoverable;
  (b) boxed other: a last complete box exists and is not (a) (free text, a number);
  (c) no box: no complete box (answers stated only in prose, truncation before a box, off-topic text).
- The sensitivity scorer is the frozen scorer plus class (a) read as its letter. It is applied to every output of both
  splits (a0 and every condition), so the dev-fitted re-check rules are refitted on dev rescored the same way.
- Correction after the first run (rule v2; v1 is commit bebeb7d). Reading the samples showed that v1 also put in class
  (a) boxes that list several options ("A, I, G, H", "F, E, F, I") and read them as their first letter. v2 moves a box
  to class (m) "several letters" when, besides its first letter, it contains another standalone capital option letter
  (outside \\text{} wrappers); (m) stays without an answer, as in the frozen scorer. v2 also does not read a box that
  starts with "e.g." or "i.e." as a letter (a unit check, not seen in the data). Both versions are reported.
- Descriptive only: fallback letters are split by finish reason (a fallback letter in an output that hit the token cap
  comes from text the model was still producing, often a loop).
- The primary comparisons of the addendum (P1-P3) are recomputed under the sensitivity scorer by the frozen
  mc_results.analyse(), with the same seeds and resamples, and shown next to the frozen values
  (analysis/out/v2_02_mc_results.json). Nothing here replaces a frozen value.

Usage: python analysis/fu_mc_parse_audit.py [--n-boot 10000]
-> reports/fu_10_mc_parse_audit.md, analysis/out/fu_10_mc_parse_audit.json,
   analysis/out/fu_10_mc_parse_audit_samples.md (output endings per class, for inspection)
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import mc_results as MR  # noqa: E402
from p6_confirmatory import N_BOOT  # noqa: E402
from src.score import mc as S  # noqa: E402

CONDS = MR.CONDS
RECOVER = re.compile(r"\(?([a-j])\)?\s*(?:[,;:.)\-\u2013\u2014]|$)")
STANDALONE = re.compile(r"(?<![A-Za-z0-9])([A-J])(?![A-Za-z0-9])")
SAMPLES_PER_CLASS = 8
TAIL = 400


def _raw(s):
    """The box content with \\text{} wrappers, backslashes and $ removed; case kept (unlike the frozen _norm)."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\\text(?:bf)?\{([^{}]*)\}", r"\1", s)
    return s.replace("\\", "").replace("$", "")


def recover(text, item, version=2):
    """Class of an output without a frozen letter, and the recovered letter for class (a)."""
    b = S.last_boxed(text)
    if b is None:
        return "c", None
    c = S._norm(b)
    m = RECOVER.match(c)
    if version >= 2 and re.match(r"(?:e\.g|i\.e)\.", c):
        m = None
    if m and S.LETTERS.index(m.group(1).upper()) < item["n_options"]:
        L = m.group(1).upper()
        if version >= 2 and {x for x in STANDALONE.findall(_raw(b)) if S.LETTERS.index(x) < item["n_options"]} - {L}:
            return "m", None
        return "a", L
    return "b", None


def outputs(model, split, items):
    """All scored outputs of one model and split: (item_id, cond, text, finish_reason, frozen status, frozen correct)."""
    e = MR.exp_of(split)
    di = MR.latest(f"{e}/{model}/{e}-mc-initial-{split}-{model}-*")
    dc = MR.latest(f"{e}/{model}/{e}-mc-conds-{split}-{model}-*")
    out = []
    for r in MR.jl(di / "initial.jsonl"):
        out.append((r["item_id"], "a0", r["a0"], r["finish_reason"], r["status"], bool(r["correct"])))
    gen = {g["prompt_key"]: g for g in MR.jl(dc / "generations.jsonl")}
    for r in MR.jl(dc / "rows.jsonl"):
        if r["cond"] == "B0":
            continue
        g = gen[r["prompt_key"]]
        out.append((r["item_id"], r["cond"], g["text"], g["finish_reason"], r["status"], bool(r["correct"])))
    for iid, cond, text, fin, status, corr in out:   # the frozen verdicts are reproduced from the stored text
        sc = S.score(text, items[iid], fin)
        assert sc["status"] == status and sc["correct"] == corr, (model, split, iid, cond)
    return out


def lenient(out, items, version=2):
    """{(item_id, cond): (class or None, correct under the sensitivity scorer)}."""
    res = {}
    for iid, cond, text, fin, status, corr in out:
        if status in ("ok", "fallback"):
            res[(iid, cond)] = (None, corr)
        else:
            k, L = recover(text, items[iid], version)
            res[(iid, cond)] = (k, L is not None and L == items[iid]["gold"])
    return res


def rescored(model, split, lev):
    D = MR.load(model, split)
    D["y0"] = np.array([lev[(i, "a0")][1] for i in D["ids"]], float)
    for c in CONDS:
        D["after"][c] = np.array([lev[(i, c)][1] for i in D["ids"]], float)
    return D


def primary(o):
    """The addendum's P1-P3 quantities from an analyse() result."""
    p = o["policies"]["F2"]["budgets"]
    return {"a0 accuracy": o["a0 accuracy"],
            "P1": {c: {"net change": o["conditions"][c]["net change"], "W->R": o["conditions"][c]["W->R"],
                       "R->W": o["conditions"][c]["R->W"]} for c in ("F0", "F1", "F2", "F2-scratch")},
            "P2 AUROC task probe - verbal W0": o["task probe - verbal W0"],
            "P2 order task probe - verbal W0": {k: p[k]["task probe"]["minus verbal W0"] for k in ("10", "20")},
            "P3 order agreement - verbal W0 (10 %)": p["10"]["agreement with a0"]["minus verbal W0"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    items = {r["item_id"]: r for r in MR.jl(ROOT / "data" / "processed" / "mmlupro" / "items.jsonl")}
    canon = {i: r["canonical_id"] for i, r in items.items()}
    frozen = json.loads((ROOT / "analysis" / "out" / "v2_02_mc_results.json").read_text(encoding="utf-8"))
    J = {"rules": __doc__.split("Rules.")[1].split("Usage:")[0].strip(), "models": {}}
    samples = ["# MMLU-Pro scoring audit: output endings per class (generated by analysis/fu_mc_parse_audit.py)", "",
               f"Test split. Up to {SAMPLES_PER_CLASS} outputs per model and class (rule v2), chosen by a hash of the item and "
               f"condition; the last {TAIL} characters of each. For inspection of the class rules; not a rate.", ""]
    for model in ("4B", "T8B"):
        out = {s: outputs(model, s, items) for s in ("test", "dev")}
        lev = {v: {s: lenient(out[s], items, v) for s in out} for v in (1, 2)}
        lt = lev[2]["test"]
        ids = set(MR.load(model, "test")["ids"])
        t = [x for x in out["test"] if x[0] in ids]
        a0c = {x[0]: x[5] for x in t if x[1] == "a0"}
        st = defaultdict(Counter)
        for iid, cond, text, fin, status, corr in t:
            st[cond][status] += 1
            if status == "fallback":
                st[cond][f"fallback, finish {fin}"] += 1
            if status not in ("ok", "fallback"):
                st[cond][f"class {lt[(iid, cond)][0]}"] += 1
        # status of the final output among W->R and R->W under the frozen scorer (classes by rule v2)
        trans = defaultdict(Counter)
        for iid, cond, text, fin, status, corr in t:
            if cond == "a0":
                continue
            if a0c[iid] and not corr:
                trans[cond][f"R->W {status}" + (f" ({lt[(iid, cond)][0]})" if status not in ("ok", "fallback") else "")] += 1
            if not a0c[iid] and corr:
                trans[cond][f"W->R {status}"] += 1
        M = {"n": len(ids), "status": {c: dict(v) for c, v in st.items()},
             "final status among transitions": {c: dict(v) for c, v in trans.items()},
             "frozen": primary(frozen["models"][model])}
        for v in (1, 2):
            flips = Counter((cond, "to correct" if lev[v]["test"][(iid, cond)][1] else "to wrong")
                            for iid, cond, text, fin, status, corr in t if lev[v]["test"][(iid, cond)][1] != corr)
            o = MR.analyse(rescored(model, "test", lev[v]["test"]), rescored(model, "dev", lev[v]["dev"]), canon, a.n_boot)
            M[f"rule v{v}"] = {"correctness changed": {f"{c} {d}": k for (c, d), k in flips.items()}, "primary": primary(o)}
        J["models"][model] = M
        byk = defaultdict(list)
        for iid, cond, text, fin, status, corr in t:
            hk = hashlib.sha256(f"{iid}|{cond}".encode()).hexdigest()
            if status not in ("ok", "fallback"):
                byk[lt[(iid, cond)][0]].append((hk, iid, cond, text, fin))
            elif status == "fallback":
                byk["fallback (parsed)"].append((hk, iid, cond, text, fin))
        for k in ("a", "m", "b", "c", "fallback (parsed)"):
            rows = sorted(byk.get(k, []))[:SAMPLES_PER_CLASS]
            samples += [f"## {model}, class {k} ({len(byk.get(k, []))} outputs)", ""]
            for _, iid, cond, text, fin in rows:
                it = items[iid]
                samples += [f"### {iid}, {cond}, finish {fin}; gold {it['gold']}; category {it['category']}", "",
                            "```", text[-TAIL:], "```", ""]
    (ROOT / "analysis" / "out" / "fu_10_mc_parse_audit.json").write_text(json.dumps(J, indent=1, ensure_ascii=False),
                                                                         encoding="utf-8", newline="\n")
    (ROOT / "analysis" / "out" / "fu_10_mc_parse_audit_samples.md").write_text("\n".join(samples) + "\n", encoding="utf-8", newline="\n")

    pc = lambda v: f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"  # noqa: E731
    L = ["# MMLU-Pro scoring audit (generated by analysis/fu_mc_parse_audit.py)", "",
         "**Post hoc, exploratory** (DEVIATIONS D19). The frozen scorer `src/score/mc.py` 1.0.0 stays primary; its results "
         "are in `reports/v2_02_mc_results.md`. Test split. An output without a frozen letter (truncated or unparsed) is "
         "class (a) if the last box starts with a valid option letter followed by end or punctuation (the frozen pattern "
         "misses \"H, with reservations ...\"), (m) if that box also names other option letters (\"A, I, G, H\"), (b) if the "
         "last box holds other text, (c) if there is no complete box. Rule v1 (fixed before the first run, commit bebeb7d) "
         "had no class (m) and read those boxes as their first letter; reading the samples showed the error, and rule v2 "
         "adds (m). The sensitivity scorers read class (a) as its letter (v1: also (m)), on both splits. 95 % "
         "cluster-bootstrap CIs (clusters = canonical stem), pp; AUROC differences × 100.", ""]
    order = ["a0"] + CONDS
    for model, m in J["models"].items():
        L += [f"## {model} (n {m['n']})", "", "### Frozen status of every scored output", "",
              "| Output | ok | fallback (finish stop / length) | truncated | unparsed | truncated + unparsed: (a) one letter | (m) several letters | (b) boxed other | (c) no box |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in order:
            s = m["status"].get(c, {})
            L.append(f"| {c} | {s.get('ok', 0)} | {s.get('fallback', 0)} ({s.get('fallback, finish stop', 0)} / "
                     f"{s.get('fallback, finish length', 0)}) | {s.get('truncated', 0)} | {s.get('unparsed', 0)} | "
                     f"{s.get('class a', 0)} | {s.get('class m', 0)} | {s.get('class b', 0)} | {s.get('class c', 0)} |")
        L += ["", "### Final-output status among the transitions (frozen scorer)", "",
              "| Condition | R->W: wrong letter | R->W: no letter (a / m / b / c) | W->R |", "|---|---|---|---|"]
        for c in CONDS:
            tr = m["final status among transitions"].get(c, {})
            wrong_letter = sum(v for k, v in tr.items() if k.startswith("R->W ok") or k.startswith("R->W fallback"))
            nol = {k2: sum(v for k, v in tr.items() if k.startswith("R->W") and k.endswith(f"({k2})")) for k2 in "ambc"}
            wr = sum(v for k, v in tr.items() if k.startswith("W->R"))
            L.append(f"| {c} | {wrong_letter} | {sum(nol.values())} ({nol['a']} / {nol['m']} / {nol['b']} / {nol['c']}) | {wr} |")
        L.append("")
        for v in (1, 2):
            ch = m[f"rule v{v}"]["correctness changed"]
            L.append(f"Outputs whose correctness the rule-v{v} scorer changes: "
                     + (", ".join(f"{k} {n}" for k, n in sorted(ch.items())) if ch else "none") + ".")
        f, s1, s2 = m["frozen"], m["rule v1"]["primary"], m["rule v2"]["primary"]
        L += ["", "### Primary comparisons: frozen scorer and the two sensitivity scorers", "",
              "| Quantity | Frozen | Rule v1 | Rule v2 |", "|---|---|---|---|",
              f"| a0 accuracy | {100 * f['a0 accuracy']:.1f} % | {100 * s1['a0 accuracy']:.1f} % | {100 * s2['a0 accuracy']:.1f} % |"]
        for c in ("F0", "F1", "F2", "F2-scratch"):
            L.append(f"| P1 {c}: net change (W->R / R->W) | " + " | ".join(
                f"{pc(x['P1'][c]['net change'])} ({100 * x['P1'][c]['W->R']:.1f} % / {100 * x['P1'][c]['R->W']:.1f} %)" for x in (f, s1, s2)) + " |")
        L.append("| P2 AUROC task probe − verbal W0 | " + " | ".join(pc(x["P2 AUROC task probe - verbal W0"]) for x in (f, s1, s2)) + " |")
        for k in ("10", "20"):
            L.append(f"| P2 F2 re-check {k} %: probe order − verbal order | "
                     + " | ".join(pc(x["P2 order task probe - verbal W0"][k]) for x in (f, s1, s2)) + " |")
        L += ["| P3 F2 re-check 10 %: agreement order − verbal order | "
              + " | ".join(pc(x["P3 order agreement - verbal W0 (10 %)"]) for x in (f, s1, s2)) + " |", ""]
    L += ["Output endings per class: `analysis/out/fu_10_mc_parse_audit_samples.md`.", ""]
    (ROOT / "reports" / "fu_10_mc_parse_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
