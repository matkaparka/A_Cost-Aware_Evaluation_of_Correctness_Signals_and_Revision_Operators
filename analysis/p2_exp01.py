"""P2 dev pilots, Exp 0 / 1A / 1B / Tsui cell: results/p2/<model>/p2-exp01-dev-*/rows.jsonl -> reports/02_pilot_exp01.md
and analysis/out/p2_pairs_<model>.json (paired outcome patterns for the power simulation).

Dev split only (the runner refuses test before the freeze). Risk differences in percentage points.
Tsui-cell rows from a run scored by an older scorer are rescored here from its generations with the current scorer
(1.1.0 added the last-number rule for the continuation protocol); the other rows are unchanged by 1.1.0 (checked).
Usage: python analysis/p2_exp01.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "02_pilot_exp01_tables.md"
PAIRS = ROOT / "analysis" / "out"
P_LEVELS = (0.5, 0.7, 0.9, 1.0)
sys.path.insert(0, str(ROOT))
from src.runlib import sha256_file  # noqa: E402
from src.score.scorer import SCORER_VERSION, score  # noqa: E402


def jl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def latest_run(model):
    reg = {r["run_id"]: r for r in jl(ROOT / "runs.jsonl")}
    # Untagged runs only: P3's tagged dev pilots (-ext, -tstar) are not the P2 pilot.
    runs = [p for p in (ROOT / "results" / "p2" / model).glob(f"p2-exp01-dev-{model}-*")
            if re.fullmatch(rf"p2-exp01-dev-{re.escape(model)}-[0-9a-f]{{7}}", p.name)
            and (p / "manifest.json").exists() and reg.get(p.name, {}).get("status") == "done"]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def summarize(rows):
    n = len(rows)
    if not n:
        return None
    wrong0 = [r for r in rows if not r["correct0"]]
    right0 = [r for r in rows if r["correct0"]]
    return {"n": n, "acc": sum(r["correct"] for r in rows) / n,
            "w2r": sum(r["correct"] for r in wrong0) / len(wrong0) if wrong0 else None,
            "r2w": sum(not r["correct"] for r in right0) / len(right0) if right0 else None,
            "n_wrong0": len(wrong0), "acc0": len(right0) / n,
            "truncated": sum(r.get("finish_reason") == "length" for r in rows) / n,
            "unparsed": sum(r.get("status") in ("unparsed", "conflict", "invalid") for r in rows) / n}


def table_1a(lines, by, data):
    lines += [f"### Exp 1A on {data} (role assistant, label self)", "",
              "| Condition | n | Acc % | Δacc vs B0 (pp) | W→R % | R→W % | Truncated % | Unparsed % |",
              "|---|---|---|---|---|---|---|---|"]
    b0 = summarize(by.get(("1A", data, "B0"), []))
    conds = ["B0", "B1", "F0", "F1", "F2", "F2-scratch"] + [f"F3({p})" for p in P_LEVELS] + \
            ["F5", "F3-perm[0]", "F3-perm[1]", "F3-perm[2]", "F4"]
    for c in conds:
        s = summarize(by.get(("1A", data, c), []))
        if s:
            d = None if b0 is None else s["acc"] - b0["acc"]
            lines.append(f"| {c} | {s['n']} | {pct(s['acc'])} | {pct(d)} | {pct(s['w2r'])} | {pct(s['r2w'])} | "
                         f"{pct(s['truncated'])} | {pct(s['unparsed'])} |")
    lines.append("")


def table_1b(lines, by, data):
    lines += [f"### Exp 1B on {data} (role × label)", "",
              "| Feedback | Role | Label | n | W→R % | R→W % | Acc % |", "|---|---|---|---|---|---|---|"]
    for fb in ("F0", "F1", "F3(1.0)"):
        cells = {}
        for role in ("assistant", "user"):
            for label in ("self", "other"):
                s = summarize(by.get(("1B", data, f"{role}/{label}/{fb}"), []))
                cells[(role, label)] = s
                if s:
                    lines.append(f"| {fb} | {role} | {label} | {s['n']} | {pct(s['w2r'])} | {pct(s['r2w'])} | {pct(s['acc'])} |")
        if all(cells.values()) and all(c["w2r"] is not None for c in cells.values()):
            role_eff = (cells[("user", "self")]["w2r"] + cells[("user", "other")]["w2r"]
                        - cells[("assistant", "self")]["w2r"] - cells[("assistant", "other")]["w2r"]) / 2
            label_eff = (cells[("assistant", "other")]["w2r"] + cells[("user", "other")]["w2r"]
                         - cells[("assistant", "self")]["w2r"] - cells[("user", "self")]["w2r"]) / 2
            lines.append(f"| {fb} | **role effect** (user − assistant) | | | {pct(role_eff)} pp | | |")
            lines.append(f"| {fb} | **label effect** (other − self) | | | {pct(label_eff)} pp | | |")
    lines.append("")


def table_exp0(lines, by):
    lines += ["### Exp 0 (paired worlds; LM calibration, not an invariant)", "",
              "| Condition | n | Valid % | Acc % | Gain ratio (acc − 0.5)/(p − 0.5) | Picks A % (valid) | Same pick in both worlds % |",
              "|---|---|---|---|---|---|---|"]
    for c in ["no-feedback", "F0"] + [f"F3({p})" for p in P_LEVELS]:
        rows = by.get(("0", "exp0", c), [])
        if not rows:
            continue
        valid = sum(r.get("valid", r.get("valid0", False)) for r in rows) / len(rows)
        acc = sum(r["correct"] for r in rows) / len(rows)
        gr = "—"
        if c.startswith("F3(") and float(c[3:-1]) > 0.5:
            p = float(c[3:-1])
            gr = f"{(acc - 0.5) / (p - 0.5):.2f}"
        # The two worlds of a template have different gold letters, so a pick that ignores the world (and the verdict)
        # is the same letter in both: accuracy 50% by construction, whatever the feedback.
        picks = {r["item_id"]: r.get("choice") for r in rows if r.get("choice") in ("A", "B")}
        pa = sum(v == "A" for v in picks.values()) / len(picks) if picks else None
        pairs = [(picks[t + "-w0"], picks[t + "-w1"]) for t in {i.rsplit("-", 1)[0] for i in picks}
                 if t + "-w0" in picks and t + "-w1" in picks]
        same = sum(a == b for a, b in pairs) / len(pairs) if pairs else None
        lines.append(f"| {c} | {len(rows)} | {pct(valid)} | {pct(acc)} | {gr} | {pct(pa)} | {pct(same)} (of {len(pairs)} pairs) |")
    lines.append("")


def rescore_tsui(run, by):
    """Rescore the Tsui-cell rows from the run's generations with the current scorer (last-number rule on)."""
    gold = {r["item_id"]: r["gold"] for r in jl(ROOT / "data" / "processed" / "dinj.jsonl")}
    texts = {g["prompt_key"]: g["text"] for g in jl(run / "generations.jsonl")}
    for key, rows in by.items():
        if key[0] != "tsui":
            continue
        for r in rows:
            s = score(texts[r["prompt_key"]], gold[r["item_id"]], finish_reason=r["finish_reason"], last_number=True)
            r.update({"correct": s["correct"], "status": s["status"], "pred": s["pred"]})


def table_tsui(lines, by, note):
    lines += [f"### Tsui replication cell (D-inj; continuation protocol, 1024 tokens; {note})", "",
              "| Dataset | Internal % | Internal + Wait % | External % | External + Wait % | Unparsed % (internal) |",
              "|---|---|---|---|---|---|"]
    for ds in ("gsm8k_sc", "prm800k_sc", "scli5"):
        vals = []
        for c in ("internal", "internal+wait", "external", "external+wait"):
            rows = [r for r in by.get(("tsui", "dinj", c), []) if r["dataset"] == ds]
            vals.append(sum(r["correct"] for r in rows) / len(rows) if rows else None)
        rows = [r for r in by.get(("tsui", "dinj", "internal"), []) if r["dataset"] == ds]
        unp = sum(r.get("status") in ("unparsed", "conflict") for r in rows) / len(rows) if rows else None
        lines.append(f"| {ds} | " + " | ".join(pct(v) for v in vals) + f" | {pct(unp)} |")
    st = Counter(r["status"] for k, rows in by.items() if k[0] == "tsui" for r in rows)
    lines += ["", "Extraction status over all Tsui rows: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())) +
              ". `last_number` = no boxed / \"answer is\" / #### answer; the continuation's last number was used.", ""]


def pairs(by, model):
    """Item-level paired outcomes for the key contrasts (power simulation input)."""
    def vec(exp, data, cond):
        return {r["item_id"]: (r["correct0"], r["correct"]) for r in by.get((exp, data, cond), [])}
    out = {}
    specs = {"H1a_F2_vs_F1": [("1A", "dnat", "F2"), ("1A", "dnat", "F1")],
             "H1b_F3": [("1A", "dnat", f"F3({p})") for p in P_LEVELS],
             "H1d_F5_vs_perm": [("1A", "dnat", "F5")] + [("1A", "dnat", f"F3-perm[{k}]") for k in range(3)],
             "H3a_dinj_F0": [("1B", "dinj", f"{r}/{l}/F0") for r in ("assistant", "user") for l in ("self", "other")],
             "H3a_dnat_F0": [("1B", "dnat", f"{r}/{l}/F0") for r in ("assistant", "user") for l in ("self", "other")],
             "H2b_F0_F2_F3": [("1A", "dnat", "F0"), ("1A", "dnat", "F2"), ("1A", "dnat", "F3(1.0)")]}
    for name, conds in specs.items():
        vs = [vec(*c) for c in conds]
        ids = sorted(set.intersection(*(set(v) for v in vs))) if all(vs) else []
        out[name] = {"conditions": [c[2] for c in conds],
                     "items": [{"item_id": i, "correct0": vs[0][i][0], "outcomes": [v[i][1] for v in vs]} for i in ids]}
    PAIRS.mkdir(parents=True, exist_ok=True)
    (PAIRS / f"p2_pairs_{model}.json").write_text(json.dumps(out), encoding="utf-8")


def main():
    lines = ["# P2 dev pilot tables: Exp 0 / 1A / 1B / Tsui cell (generated by analysis/p2_exp01.py; dev split only)", ""]
    for model in ("1.7B", "4B"):
        run = latest_run(model)
        lines += [f"## Qwen3-{model}", ""]
        if run is None:
            lines += [f"TODO(missing: P2 Exp 0/1 dev run for {model})", ""]
            continue
        man = json.loads((run / "manifest.json").read_text())
        lines += [f"Run `{run.name}` (commit {man['code_commit'][:7]}, {man['gpu']}, VLLM_BATCH_INVARIANT={man['env'].get('VLLM_BATCH_INVARIANT')}).", ""]
        by = defaultdict(list)
        for r in jl(run / "rows.jsonl"):
            by[(r["exp"], r["data"], r["cond"])].append(r)
        table_1a(lines, by, "dnat")
        table_1b(lines, by, "dinj")
        table_1b(lines, by, "dnat")
        f4 = summarize(by.get(("1A", "dinj", "F4"), []))
        if f4:
            lines += [f"F4 location pointer on D-inj (gsm8k_sc): n {f4['n']}, W→R {pct(f4['w2r'])} %.", ""]
        table_exp0(lines, by)
        if man.get("scorer_sha256") == sha256_file(ROOT / "src" / "score" / "scorer.py"):
            note = f"scored at run time by scorer {SCORER_VERSION}"
        else:
            rescore_tsui(run, by)
            note = f"rescored from the run's generations with scorer {SCORER_VERSION}"
        table_tsui(lines, by, note)
        pairs(by, model)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
