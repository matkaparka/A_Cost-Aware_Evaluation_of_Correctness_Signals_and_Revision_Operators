"""Lore QA dev pilot: the gates of docs/planning/lore_qa/proposal.md §7, applied mechanically to the control-host runs
of src/exp/lore_pilot.py (HF backend; the gates are re-measured with vLLM on the run host before any test run).

Every saved output is rescored here with the current lore scorer (src/score/lore.py, version in the report); the
verdicts written at run time (scorer 1.0.1) are shown beside them.
- G2 (difficulty, 4B): the budget is the smallest of BUDGETS whose 4B dev initial accuracy is <= G2_HI; it passes if
  that accuracy lies in [G2_LO, G2_HI]. If no budget has accuracy <= G2_HI, D-nat-lore is descriptive only.
- G3 (parsing, 4B at the chosen budget): unparsed + truncated initial answers <= G3_MAX.
- G4 (size): projected 4B initially wrong test items, (1 - dev accuracy) x the number of lore test items, >= G4_MIN.
- G3b (phrase answers): the scorer's disagreement with human-style labels on dev phrase answers <= G3B_MAX, measured
  on held-out pairs: (item, answer) pairs not among the tuning labels (docs/planning/lore_qa/pilot/labels.jsonl, the
  rows scorer 1.1.0 was fitted to). Held-out labels come from a blind judge (blind/verdicts.json, keyed by the
  sheet's row IDs in blind/key.json); the judge sees question, gold, accepted forms, the evidence passage and the
  answer, never the scorer's verdict, the model or the condition.
Also: accuracy by route, answer class and budget; second-scorer agreement; verbal AUROC; the smoke run's outcome
counts per condition (descriptive, small n); agreement between the tuning labels and the blind judge.
-> reports/lore_01_pilot.md, analysis/out/lore_01_pilot.json, docs/planning/lore_qa/pilot/blind/{sheet.md,key.json}
Usage: python analysis/lore_pilot.py --initial results/lore/4B/<run> [--initial results/lore/1.7B/<run>]
                                     [--smoke results/lore/4B/<run>] [--sheet]
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from stats import auroc  # noqa: E402
from src.data import lore, lore_b  # noqa: E402
from src.score import lore as S  # noqa: E402

BUDGETS = lore.BUDGETS
G2_LO, G2_HI, G3_MAX, G4_MIN, G3B_MAX = 0.35, 0.85, 0.05, 150, 0.10
PILOT = ROOT / "docs" / "planning" / "lore_qa" / "pilot"
LABELS, BLIND = PILOT / "labels.jsonl", PILOT / "blind"


def jl(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines()]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [p, c - w, c + w]


def summarize(rows):
    n = len(rows)
    bad = sum(r["status"] in ("unparsed", "truncated", "conflict") for r in rows)
    s = [r["s"] for r in rows if r["s"] is not None]
    y = [r["correct"] for r in rows if r["s"] is not None]
    return {"n": n, "acc": wilson(sum(r["correct"] for r in rows), n),
            "acc as run": sum(r["correct_run"] for r in rows) / n if n else None,
            "unparsed+truncated+conflict": bad / n if n else None,
            "status": dict(Counter(r["status"] for r in rows)),
            "second scorer agreement": sum(r["correct"] == r["correct_second"] for r in rows) / n if n else None,
            "verbal AUROC": auroc(y, s) if 0 < sum(y) < len(y) else None,
            "mean out tokens": float(np.mean([r["out_tokens"] for r in rows])) if n else None}


def all_items():
    return {it["id"]: it for it in [dict(x, route="A") for x in lore.load_built()[1]] + lore_b.load_items_b()}


def rescore(rows, items, text_key):
    for r in rows:
        if r.get("status") == "turn1":
            continue
        s = S.score(r[text_key], items[r["item_id"]], r["finish_reason"])
        r["correct_run"] = r["correct"]
        r.update(status=s["status"], pred=s["pred"], correct=s["correct"])
    return rows


def evidence(it, passages):
    """The passage sentences that hold the gold (last hop), for the blind judge."""
    text = passages[it["hops"][-1]["evidence_pid"]]["text"]
    sents = re.split(r"(?<=[.!?])\s+", text)
    forms = [f.casefold() for f in it["answer_forms"]]
    hit = [s for s in sents if any(f in s.casefold() for f in forms)]
    out = " ".join(hit) if hit else text
    return out if len(out) <= 900 else out[:900] + " …"


def exact(pred, it):
    return S.normalize(pred or "") in {S.normalize(f) for f in it["answer_forms"]}


def h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def write_sheet(pairs, items, passages):
    """Blind sheet: every (item, answer) pair that needs a label, in seeded order; no verdicts, models or conditions."""
    order = sorted(pairs, key=lambda p: h("lore-blind-v1", *p))
    L = ["# Blind labels: lore QA answers", "",
         "For each row, decide whether the ANSWER is right, judged against the GOLD answer (and its accepted forms), the "
         "QUESTION and the EVIDENCE passage. Standard: a careful grader of a reading-comprehension test.",
         "- right: it gives the same thing as the gold. Extra correct detail, a title, an epithet, a restated context, "
         "or an equivalent number format is fine.",
         "- wrong: a different entity or value; a core part of the gold missing; a hedge between alternatives; or a "
         "statement that contradicts the gold.",
         "- unsure: you cannot decide from what is shown.", "",
         "Write blind/verdicts.json as {\"R001\": {\"label\": \"right|wrong|unsure\", \"why\": \"...\"}, ...}; "
         "\"why\" is required for wrong and unsure, optional for right.", ""]
    key = {}
    for n, (iid, pred) in enumerate(order, start=1):
        it, rid = items[iid], f"R{n:03d}"
        key[rid] = {"item_id": iid, "pred": pred}
        L += [f"## {rid}", f"- QUESTION: {it['question']}", f"- GOLD: {it['gold_display']}",
              f"- ACCEPTED FORMS: {' | '.join(it['answer_forms'])}", f"- EVIDENCE: {evidence(it, passages)}",
              f"- ANSWER: {pred}", ""]
    BLIND.mkdir(parents=True, exist_ok=True)
    (BLIND / "sheet.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    (BLIND / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return len(key)


def load_labels():
    tune = {(r["item_id"], r["pred"]): r["human"] for r in map(json.loads, LABELS.read_text(encoding="utf-8").splitlines())}
    blind = {}
    if (BLIND / "verdicts.json").exists() and (BLIND / "key.json").exists():
        key = json.loads((BLIND / "key.json").read_text(encoding="utf-8"))
        for rid, v in json.loads((BLIND / "verdicts.json").read_text(encoding="utf-8")).items():
            k = key[rid]
            blind[(k["item_id"], k["pred"])] = {"right": True, "wrong": False}.get(v["label"])
    return tune, blind


def agreement(pairs, a, b):
    both = [p for p in pairs if a.get(p) is not None and b.get(p) is not None]
    k = sum(a[p] == b[p] for p in both)
    return {"n": len(both), "agree": k, "disagree": len(both) - k,
            "disagreement": wilson(len(both) - k, len(both)) if both else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial", action="append", required=True)
    ap.add_argument("--smoke")
    ap.add_argument("--sheet", action="store_true", help="(re)write the blind sheet for unlabeled pairs")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    items, passages = all_items(), {p["pid"]: p for p in lore.load_built()[0]}
    n_test = sum(1 for it in items.values() if it["split"] == "test")
    out = {"scorer_version": S.LORE_SCORER_VERSION, "n_lore_test_items": n_test, "runs": {}, "gates": {}}
    rows_by, all_rows = {}, []
    for run in a.initial:
        man = json.loads((ROOT / run / "manifest.json").read_text(encoding="utf-8"))
        model = man["model"].split("-")[-1]
        rows = rescore(jl(Path(run) / "rows.jsonl"), items, "a0")
        rows_by[model] = rows
        all_rows += rows
        res = {"run_id": man["run_id"], "code_commit": man["code_commit"], "by_budget": {}}
        for b in sorted({r["budget"] for r in rows}):
            rb = [r for r in rows if r["budget"] == b]
            res["by_budget"][str(b)] = {"all": summarize(rb),
                                        **{f"route {k}": summarize([r for r in rb if r["route"] == k]) for k in ("A", "B")},
                                        **{f"class {k}": summarize([r for r in rb if r["answer_class"] == k]) for k in ("name", "phrase")}}
        out["runs"][model] = res
    smoke = rescore(jl(Path(a.smoke) / "rows.jsonl"), items, "text") if a.smoke else []
    if smoke:   # initial correctness of D-nat rows: the rescored 4B initial answer at the smoke budget, not the run-time verdict
        sb = json.loads((ROOT / a.smoke / "config.json").read_text(encoding="utf-8"))["budget"]
        c0 = {r["item_id"]: r["correct"] for r in rows_by["4B"] if r["budget"] == sb}
        for r in smoke:
            if r["data"] == "dnat":
                r["correct0_run"], r["correct0"] = r["correct0"], c0[r["item_id"]]
    all_rows += [r for r in smoke if r["status"] != "turn1"]

    # ---- labels: tuning set (fitted), blind judge, held-out pairs
    tune, blind = load_labels()
    cls = lambda p: items[p[0]]["answer_class"]  # noqa: E731
    need = {(r["item_id"], r["pred"]) for r in all_rows
            if r["pred"] is not None and (r["answer_class"] if "answer_class" in r else items[r["item_id"]]["answer_class"]) == "phrase"}
    need |= {(r["item_id"], r["pred"]) for r in all_rows if r["pred"] is not None and not exact(r["pred"], items[r["item_id"]])}
    need |= set(tune)
    if a.sheet:
        out["blind sheet rows"] = write_sheet(need, items, passages)
    v11 = {p: S.equivalent(p[1], items[p[0]]["answer_forms"], items[p[0]]["answer_kind"], cls(p)) for p in need | set(blind)}
    held = [p for p in blind if p not in tune]
    lab = {"tuning pairs": len(tune), "blind-labeled pairs": len(blind), "blind unsure": sum(v is None for v in blind.values()),
           "scorer vs tuning labels (in sample)": agreement(list(tune), v11, tune),
           "tuning labels vs blind judge": agreement(list(tune), tune, blind),
           "scorer vs blind, held-out phrase": agreement([p for p in held if cls(p) == "phrase"], v11, blind),
           "scorer vs blind, held-out name": agreement([p for p in held if cls(p) == "name"], v11, blind),
           "unlabeled pairs needing a label": len([p for p in need if p not in tune and p not in blind])}
    errs = [{"item_id": p[0], "gold": items[p[0]]["gold_display"], "pred": p[1], "scorer": v11[p], "blind": blind[p]}
            for p in held if blind[p] is not None and v11[p] != blind[p]]
    lab["held-out disagreements"] = errs
    out["labels"] = lab

    # ---- gates on 4B
    r4 = out["runs"].get("4B")
    if r4:
        acc = {int(b): v["all"]["acc"][0] for b, v in r4["by_budget"].items()}
        ok = [b for b in sorted(acc) if acc[b] <= G2_HI]
        chosen = ok[0] if ok else min(acc, key=lambda b: (acc[b], b))
        g = {"accuracy by budget": acc, "accuracy as run (scorer 1.0.1)": {int(b): v["all"]["acc as run"] for b, v in r4["by_budget"].items()},
             "chosen budget": chosen,
             "G2": ("pass" if G2_LO <= acc[chosen] <= G2_HI else "fail (too hard)") if ok else
                   "fail (above the upper bound at every budget): D-nat-lore descriptive, D-inj-lore primary"}
        bad = r4["by_budget"][str(chosen)]["all"]["unparsed+truncated+conflict"]
        g["G3"] = {"rate": bad, "verdict": "pass" if bad <= G3_MAX else "fail"}
        proj = (1 - acc[chosen]) * n_test
        g["G4"] = {"projected wrong test items": proj, "verdict": "pass" if proj >= G4_MIN else "fail: D-nat-lore descriptive"}
        hp = lab["scorer vs blind, held-out phrase"]
        g["G3b"] = {"held-out phrase pairs": hp["n"], "scorer wrong": hp["disagree"],
                    "verdict": "pending (no blind labels)" if not hp["n"] else
                    ("pass" if hp["disagree"] / hp["n"] <= G3B_MAX else "fail: phrase items descriptive")}
        out["gates"] = g

    # ---- smoke
    if smoke:
        sm = defaultdict(lambda: {"n": 0, "status": Counter(), "correct": 0, "W->R": [0, 0], "R->W": [0, 0], "tokens": []})
        for r in smoke:
            if r["status"] == "turn1":
                continue
            c = sm[f"{r['data']}/{r['cond']}"]
            c["n"] += 1
            c["status"][r["status"]] += 1
            c["correct"] += bool(r["correct"])
            c["tokens"].append(r["out_tokens"])
            key = "R->W" if r["correct0"] else "W->R"
            c[key][0] += (not r["correct"]) if r["correct0"] else bool(r["correct"])
            c[key][1] += 1
        out["smoke"] = {"run": a.smoke, "conditions": {k: {"n": v["n"], "status": dict(v["status"]), "acc": v["correct"] / v["n"],
                                                           "W->R": v["W->R"], "R->W": v["R->W"],
                                                           "mean out tokens": float(np.mean(v["tokens"]))}
                                                       for k, v in sorted(sm.items())}}

    (ROOT / "analysis" / "out" / "lore_01_pilot.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                                    encoding="utf-8", newline="\n")
    pc = lambda v: "—" if v is None else f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"  # noqa: E731
    L = ["# Lore QA dev pilot (generated by analysis/lore_pilot.py)", "",
         f"Control host, HF backend, greedy, dev split only (src/exp/lore_pilot.py). Every output is rescored with lore "
         f"scorer {out['scorer_version']}; \"as run\" is the verdict written at run time (scorer 1.0.1). HF and vLLM "
         "greedy outputs differ, so the gates are re-measured with vLLM on the run host before any test run (proposal "
         "§7). Accuracy in % with a Wilson 95 % interval.", ""]
    for model, res in out["runs"].items():
        L += [f"## {model} initial answers (`{res['run_id']}`)", "",
              "| Budget | Subset | n | Acc | Acc as run | Unparsed + truncated | Second-scorer agreement | Verbal AUROC | Mean out tokens |",
              "|---|---|---|---|---|---|---|---|---|"]
        for b, d in res["by_budget"].items():
            for k, v in d.items():
                if v["n"]:
                    va = "—" if v["verbal AUROC"] is None else f"{v['verbal AUROC']:.3f}"
                    L.append(f"| {b} | {k} | {v['n']} | {pc(v['acc'])} | {100 * v['acc as run']:.1f} | "
                             f"{100 * v['unparsed+truncated+conflict']:.1f} | {100 * v['second scorer agreement']:.1f} | {va} | "
                             f"{v['mean out tokens']:.0f} |")
        L.append("")
    lab = out["labels"]
    fmt = lambda d: "—" if not d["n"] else f"{d['disagree']} of {d['n']} ({pc(d['disagreement'])} %)"  # noqa: E731
    L += ["## Scorer check against labels", "",
          f"- Tuning pairs (scorer {out['scorer_version']} was fitted to these; labels by the main session): {lab['tuning pairs']}; "
          f"scorer disagrees on {fmt(lab['scorer vs tuning labels (in sample)'])}.",
          f"- Blind judge: {lab['blind-labeled pairs']} pairs labeled, {lab['blind unsure']} unsure. Tuning labels vs blind "
          f"judge: {fmt(lab['tuning labels vs blind judge'])} disagree.",
          f"- **Held-out phrase pairs** (the G3b statistic): scorer disagrees with the blind judge on "
          f"{fmt(lab['scorer vs blind, held-out phrase'])}.",
          f"- Held-out name pairs: scorer disagrees on {fmt(lab['scorer vs blind, held-out name'])}.",
          f"- Pairs still unlabeled: {lab['unlabeled pairs needing a label']}.", ""]
    if lab["held-out disagreements"]:
        L += ["| Item | Gold | Answer | Scorer | Blind judge |", "|---|---|---|---|---|"]
        L += [f"| `{e['item_id']}` | {e['gold']} | {e['pred'].replace('|', '/')} | {'right' if e['scorer'] else 'wrong'} | "
              f"{'right' if e['blind'] else 'wrong'} |" for e in lab["held-out disagreements"]]
        L.append("")
    if out["gates"]:
        g = out["gates"]
        L += ["## Gates (4B)", "", f"- Lore test items: {out['n_lore_test_items']}.",
              f"- **G2**: chosen budget {g['chosen budget']}; " + ", ".join(f"{b}: {100 * x:.1f} %" for b, x in g["accuracy by budget"].items())
              + " (as run: " + ", ".join(f"{100 * x:.1f} %" for x in g["accuracy as run (scorer 1.0.1)"].values()) + f") → {g['G2']}.",
              f"- **G3**: {100 * g['G3']['rate']:.1f} % unparsed + truncated + conflict → {g['G3']['verdict']}.",
              f"- **G4**: {g['G4']['projected wrong test items']:.0f} projected initially wrong test items → {g['G4']['verdict']}.",
              f"- **G3b**: {g['G3b']['scorer wrong']} of {g['G3b']['held-out phrase pairs']} held-out phrase pairs scored "
              f"against the blind judge → {g['G3b']['verdict']}.", ""]
    if "smoke" in out:
        L += [f"## Smoke run (`{out['smoke']['run']}`; descriptive, small n)", "",
              "| Data/condition | n | Status | Acc | W->R (k/n) | R->W (k/n) | Mean out tokens |", "|---|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {', '.join(f'{s} {c}' for s, c in v['status'].items())} | {100 * v['acc']:.0f} | "
              f"{v['W->R'][0]}/{v['W->R'][1]} | {v['R->W'][0]}/{v['R->W'][1]} | {v['mean out tokens']:.0f} |"
              for k, v in out["smoke"]["conditions"].items()]
    (ROOT / "reports" / "lore_01_pilot.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
