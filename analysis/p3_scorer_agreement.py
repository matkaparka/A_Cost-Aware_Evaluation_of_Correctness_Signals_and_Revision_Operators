"""Scorer invariant (design §5.6, §8): agreement with a second scorer on dev >= 99%.
-> reports/03_scorer_agreement.md and analysis/out/scorer_agreement.json.

Main scorer: src/score/scorer.py `score` (math-verify equivalence). Two second scorers, both with the main scorer's answer
extraction and without math-verify:
  - v1 `scorer.score_simple`: numeric or normalized-string comparison; it does not accept two boxed answers;
  - v2 `second.score_v2`: the MATH reference grader's normalization, then numeric equality (added on 2026-11-05 after v1
    failed the invariant on equivalent answer forms; DECISIONS 2026-11-05).
Dev texts:
  - P1 initial answers a0 of 1.7B, 4B and 8B, on the original pool and on the expansion;
  - the Exp 1A / 1B revisions of the dev pilot runs (Tsui cell and Exp 0 excluded: they use other rules), one count per
    distinct generation (conditions that share a generation are counted once).
Agreement is on the correct / incorrect label. Disagreements are grouped by type; samples for hand checking (pred and gold
only): 30 random v1 disagreements, and every v2 disagreement up to 40. Dev split only.
Usage: python analysis/p3_scorer_agreement.py
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.score.scorer import SCORER_VERSION, score, score_simple  # noqa: E402
from src.score.second import score_v2  # noqa: E402

THRESHOLD, SEED = 0.99, 1105
SAMPLES = {"v1": 30, "v2": 40}
# Hand-check verdicts on the printed v2 sample (reports/03_handcheck.md, 2026-11-05; the v1 sample is checked too but
# is drawn from another pool): main-scorer errors found among the
# sampled disagreements in which the main scorer accepts, and the disagreements in which it rejects (all checked).
HANDCHECK = {"accepts": {"checked": 40, "main_errors": 0}, "rejects": {"checked": 1, "main_errors": 1}}
SECOND = {"v1": score_simple, "v2": score_v2}
P1_RUNS = {m: [f"p1-initial-{m}-d2ebdea", f"p1-initial-ext-{m}-45575c2"] for m in ("1.7B", "4B", "8B")}
EXP01_RUNS = {m: [f"p2-exp01-dev-{m}-f7e5011", f"p2-exp01-dev-ext-{m}-6cd867e"] for m in ("1.7B", "4B")}
OUT_MD = ROOT / "reports" / "03_scorer_agreement.md"
OUT_JSON = ROOT / "analysis" / "out" / "scorer_agreement.json"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def kind(main, simple):
    if main["status"] != "ok" and main["status"] != simple["status"]:
        return f"main status {main['status']}, second {simple['status']}"
    return f"main {'correct' if main['correct'] else 'incorrect'}, second {'correct' if simple['correct'] else 'incorrect'}"


def main():
    gold = {r["item_id"]: r["gold"] for f in ("dnat.jsonl", "ext/dnat_ext.jsonl", "dinj.jsonl")
            for r in jl(ROOT / "data" / "processed" / f)}
    split = {r["item_id"]: r["split"] for f in ("dnat.jsonl", "ext/dnat_ext.jsonl", "dinj.jsonl")
             for r in jl(ROOT / "data" / "processed" / f)}
    texts = []   # (source, model, item_id, text, finish_reason)
    missing = []
    for m, runs in P1_RUNS.items():
        for run in runs:
            f = ROOT / "results" / "p1" / m / run / "initial.jsonl"
            if not f.exists():
                missing.append(run)
                continue
            src = "a0 expansion" if "-ext-" in run else "a0 original"
            texts += [(src, m, r["item_id"], r["a0"], r["finish_reason"]) for r in jl(f) if r["split"] == "dev"]
    for m, runs in EXP01_RUNS.items():
        for run in runs:
            d = ROOT / "results" / "p2" / m / run
            if not (d / "rows.jsonl").exists():
                missing.append(run)
                continue
            gen = {g["prompt_key"]: g for g in jl(d / "generations.jsonl")}
            seen = set()
            for r in jl(d / "rows.jsonl"):
                if r["exp"] not in ("1A", "1B") or r.get("prompt_key") is None or r["prompt_key"] in seen:
                    continue
                seen.add(r["prompt_key"])
                assert split[r["item_id"]] == "dev"
                g = gen[r["prompt_key"]]
                src = f"revisions {'expansion' if '-ext-' in run else r['data']}"
                texts.append((src, m, r["item_id"], g["text"], g["finish_reason"]))

    by_src = defaultdict(Counter)
    dis = {v: defaultdict(list) for v in SECOND}
    for src, m, iid, text, fr in texts:
        assert split[iid] == "dev"
        a = score(text, gold[iid], finish_reason=fr)
        by_src[src]["n"] += 1
        for v, fn in SECOND.items():
            b = fn(text, gold[iid], finish_reason=fr)
            same = a["correct"] == b["correct"]
            by_src[src][v] += same
            if not same:
                dis[v][kind(a, b)].append({"source": src, "model": m, "item_id": iid, "pred": a["pred"], "gold": gold[iid]})
    n = sum(c["n"] for c in by_src.values())
    agree = {v: sum(c[v] for c in by_src.values()) for v in SECOND}
    rate = {v: agree[v] / n if n else None for v in SECOND}
    rng = random.Random(SEED)
    lines = [f"# Scorer agreement on dev (generated by analysis/p3_scorer_agreement.py; scorer {SCORER_VERSION})", "",
             "Main scorer `score` (math-verify) vs two second scorers with the same extraction: v1 `score_simple` (numeric or "
             "normalized-string comparison) and v2 `score_v2` (MATH reference normalization, then numeric equality; added "
             "after v1 failed). Agreement on the correct / incorrect label; one count per distinct generation. Invariant "
             f"(design §8): ≥ {100 * THRESHOLD:.0f} %.", ""]
    if missing:
        lines += [f"Runs not present (left out): {', '.join(missing)}.", ""]
    lines += ["| Source | Texts | Agree v1 | v1 % | Agree v2 | v2 % |", "|---|---|---|---|---|---|"]
    for src, c in sorted(by_src.items()):
        lines.append(f"| {src} | {c['n']} | {c['v1']} | {100 * c['v1'] / c['n']:.2f} | {c['v2']} | {100 * c['v2'] / c['n']:.2f} |")
    lines += [f"| **all** | {n} | {agree['v1']} | **{100 * rate['v1']:.2f}** | {agree['v2']} | **{100 * rate['v2']:.2f}** |", "",
              f"Against v1 the invariant {'passes' if rate['v1'] >= THRESHOLD else 'fails'}; against v2 it "
              f"{'passes' if rate['v2'] >= THRESHOLD else 'fails'}.", ""]
    # Amended invariant (DECISIONS 2026-11-05): the main scorer's label error located in the v2 disagreements, at the
    # upper 95 % bound, stays below 1 - THRESHOLD of all texts. Disagreements where the main scorer accepts are bounded
    # from the hand-checked sample (exact binomial upper bound); those where it rejects are all counted as errors when
    # the hand check found one.
    from scipy.stats import beta
    acc = sum(len(x) for k, x in dis["v2"].items() if k.startswith("main correct"))
    rej = sum(len(x) for k, x in dis["v2"].items() if not k.startswith("main correct"))
    hc = HANDCHECK["accepts"]
    ub = float(beta.ppf(0.95, hc["main_errors"] + 1, hc["checked"] - hc["main_errors"]))
    bound = (acc * ub + rej * HANDCHECK["rejects"]["main_errors"] / max(1, HANDCHECK["rejects"]["checked"])) / n
    lines += [f"Amended invariant: v2 disagreements where the main scorer accepts {acc}, rejects {rej}. Hand check: "
              f"{hc['main_errors']} main-scorer errors in {hc['checked']} sampled 'accepts' disagreements (one-sided 95 % "
              f"upper bound {100 * ub:.1f} %); {HANDCHECK['rejects']['main_errors']} of {HANDCHECK['rejects']['checked']} "
              f"'rejects' disagreements counted as main-scorer errors. Upper bound on the main scorer's label error located "
              f"in the disagreements: {100 * bound:.2f} % of all texts; the amended invariant "
              f"{'passes' if bound < 1 - THRESHOLD else 'fails'} (limit {100 * (1 - THRESHOLD):.0f} %).", ""]
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")[:120]
    for v in SECOND:
        lines += [f"## Disagreements with {v}", "", "| Type | Count |", "|---|---|"]
        lines += [f"| {k} | {len(x)} |" for k, x in sorted(dis[v].items(), key=lambda kv: -len(kv[1]))]
        pool = [d for x in dis[v].values() for d in x]
        pick = pool if len(pool) <= SAMPLES[v] else rng.sample(pool, SAMPLES[v])
        lines += ["", f"Sample ({len(pick)} of {len(pool)}; index = row number for the hand-check record):", "",
                  "| # | Source | Model | Item | Main | Pred | Gold |", "|---|---|---|---|---|---|---|"]
        for i, d in enumerate(sorted(pick, key=lambda d: (d["source"], d["item_id"])), 1):
            lines.append(f"| {i} | {d['source']} | {d['model']} | {d['item_id']} | "
                         f"{'correct' if any(d in x for k, x in dis[v].items() if k.startswith('main correct')) else 'incorrect'} | "
                         f"`{esc(d['pred'])}` | `{esc(d['gold'])}` |")
        lines.append("")
    OUT_JSON.write_text(json.dumps({"n": n, "agree": agree, "rate": rate, "amended_bound": bound, "by_source": {k: dict(v) for k, v in by_src.items()},
                                    "disagreement_counts": {v: {k: len(x) for k, x in dis[v].items()} for v in SECOND},
                                    "missing": missing}, indent=1))
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
