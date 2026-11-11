"""Plan Amendment 7 (docs/planning/arr_round/plan.md): thinking on every item under a reduced budget B, the uniform
baseline that review M4 of the ACL v2 draft asked for. Descriptive and post hoc; outside the §5 family; intervals as in
§0 (paired cluster bootstrap over canonical problems, 4,000 replicates, seed 20261110). Stored outputs only.

Inputs
- The c1b runs (src/exp/arr_runs.py think-budget): results/arr/c1b/<model>/c1b-think-budget-<task>-test-b<B>-<model>-<C>,
  each hash-checked against its manifest; all from one code commit (CLAUDE.md rule 8). The smoke run (tag smoke) gives
  the prefix check and a determinism check against the full B = 256 run.
- C1 (analysis/arr/c1_think.py): its labels (c1_labels.jsonl.gz, checked against c1_think_manifest.json), which give
  THINK ALL and the score of every item whose budget-B output is its stored C1 output.
- The populations (KEEP, signals, clusters) exactly as c1_think.py loads them.

Budget-B output of an item (Amendment 7): a reused item's output is its C1 output, so its extraction and score are the
C1 label's. A forced item's think block is closed by the stop-thinking instruction, so the text after the block is
".\\n\\n" + answer_text; the stored trace prefix holds no closing tag (the cut is before the first one). Its extraction
runs on "<think>" + STOP_THINKING + answer_text, whose post-think text is that of the full output, with
c1_think.extract: gold-blind, every extraction frozen and hashed before any gold is read; then scored. A forced answer
that stops at its limit, or has no extractable answer, has no usable answer.
Cost: generated tokens per item = the C1 out_tokens of a reused item; the trace tokens used plus the answer tokens of a
forced one (the inserted instruction is prompt).

Analysis (each setting and B): accuracy; Δ vs KEEP with CI; generated tokens per item; the share over budget and the
forced answers' truncation rate; repair and damage rates (descriptive); and, side by side, THINK ALL and the C1 curve
points of selective THINK by verbal and by probe nearest in cost, with their CIs (no paired test, no interpolation).

Usage: python -B analysis/arr/c1_budget.py
-> analysis/out/arr/c1b_extractions.jsonl.gz (frozen before any gold), c1b_labels.jsonl.gz, c1b_budget.json,
   c1b_budget_manifest.json; reports/arr_c1_budget.md
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1a2_replays import load_math, load_mc  # noqa: E402
from c1_think import SETTINGS, THINK_SAMPLING, clean, curve, extract, fmt, gold_free, nearest, point_ci, think_body  # noqa: E402
from common import (J1, OUT, REPO, Boot, Equivalence, dump, guard_check, manifest, pin, plan_core_hash, plan_hash,  # noqa: E402
                    rows, rows_gz, sha, unique, write_rows_gz)

sys.path.insert(0, str(REPO))
from src.exp.arr_runs import BUDGET_ANSWER_TOKENS, STOP_THINKING  # noqa: E402

BUDGETS = (256, 512, 1024, 2048)
SIGNALS = ("verbal", "probe")


# ----------------------------------------------------------------------------- inputs
def verified(d, name):
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert sha(d / name) == man["files"][name], ("manifest checksum", d.name, name)
    return pin(d / name), man


def c1b_runs(task, model):
    """{budget: run directory} of the synced, complete runs of (task, model), and the smoke run if any."""
    root = REPO / "results" / "arr" / "c1b" / model
    pat = re.compile(rf"^c1b-think-budget-{task}-test-b(\d+)(-smoke)?-{re.escape(model)}-([0-9a-f]{{7}})$")
    runs, smoke = {}, None
    for d in sorted(root.iterdir()) if root.exists() else []:
        m = pat.match(d.name)
        if not m or not (d / "manifest.json").exists():
            continue
        if m.group(2):
            smoke = d
        else:
            assert int(m.group(1)) not in runs, ("two runs of one budget", d.name)
            runs[int(m.group(1))] = d
    return runs, smoke


def load_run(d, task, model, budget, ids):
    path, man = verified(d, "generations.jsonl")
    cfg = json.loads(verified(d, "config.json")[0].read_text(encoding="utf-8"))
    assert (cfg["stage"], cfg["task"], cfg["model"], cfg["split"], cfg["budget"]) == ("think-budget", task, model, "test", budget), d.name
    assert cfg["stop_thinking"] == STOP_THINKING and cfg["budget_answer_tokens"] == BUDGET_ANSWER_TOKENS, d.name
    assert cfg["think_sampling"] == THINK_SAMPLING and man["env"].get("VLLM_BATCH_INVARIANT") == "1", d.name
    recs = unique(rows(path))
    if ids is not None:
        assert set(recs) == set(ids), ("the run does not cover the pool exactly", d.name)
    monitor = {r["item_id"]: bool(r["correct"]) for r in rows(verified(d, "rows.jsonl")[0])}
    metrics = json.loads(verified(d, "metrics.json")[0].read_text(encoding="utf-8"))
    return recs, {"run": d.name, "code_commit": man["code_commit"][:7], "gpu": man["gpu"], "monitor": monitor,
                  "counts": metrics["counts"], "gen_s": metrics["gen_s"], "load_s": metrics["load_s"]}


def forced_text(rec):
    """Text with the same post-think part as the forced output (see the module docstring)."""
    return "<think>" + STOP_THINKING + rec["answer_text"]


# ----------------------------------------------------------------------------- analysis
def budget_policy(P, T, lab_c1, recs, lab_b):
    """Per item: correct, answered, generated tokens, forced, forced-truncated (arrays over P['ids'])."""
    y, ans, gen, forced, trunc = [], [], [], [], []
    for i in P["ids"]:
        r = recs[i]
        if r["reused"]:
            c = lab_c1[i]
            assert r["generated_tokens"] == c["out_tokens"] and r["finish_reason"] == c["finish_reason"], i
            y.append(c["correct"]), ans.append(c["status"] == "ok")
        else:
            b = lab_b[i]
            y.append(b["correct"]), ans.append(b["status"] == "ok")
            assert r["generated_tokens"] == r["trace_tokens_used"] + r["answer_tokens"], i
        gen.append(r["generated_tokens"])
        forced.append(not r["reused"])
        trunc.append((not r["reused"]) and r["finish_reason"] == "length")
    return {k: np.array(v, dtype=float) for k, v in
            {"y": y, "answered": ans, "gen": gen, "forced": forced, "trunc": trunc}.items()}


def describe_budget(P, T, B, boot, think_curves):
    y0, n = P["y0"], P["n"]
    nf = int(B["forced"].sum())
    out = {"accuracy_pct": 100 * float(B["y"].mean()), "delta_vs_keep": boot.ci(B["y"] - y0),
           "generated_tokens_per_item": float(B["gen"].mean()),
           "over_budget_pct": 100 * nf / n, "forced_items": nf,
           "forced_truncated_pct_of_forced": 100 * float(B["trunc"].sum()) / nf if nf else None,
           "no_usable_answer_pct": 100 * float((1 - B["answered"]).mean()),
           "repair_rate": boot.ratio_ci((1 - y0) * B["y"], 1 - y0), "damage_rate": boot.ratio_ci(y0 * (1 - B["y"]), y0),
           "nearest_selective": {}}
    for sig in SIGNALS:
        pt = nearest(think_curves[sig], out["generated_tokens_per_item"])
        out["nearest_selective"][sig] = {**pt, "delta_ci": point_ci(P, T["y"], sig, pt["share_pct"], boot)}
    return out


# ----------------------------------------------------------------------------- report
def report(res):
    L = ["# ARR round, Amendment 7: thinking on every item under a reduced budget", "",
         f"Generated by `analysis/arr/c1_budget.py` (plan `docs/planning/arr_round/plan.md`, core sha256 "
         f"`{res['plan_core_sha256'][:16]}…`). Descriptive and post hoc; outside the §5 family; review M4 of the ACL v2 draft.",
         "Δ in percentage points vs KEEP, with paired 95 % cluster-bootstrap intervals (4,000 replicates, seed 20261110). "
         "Costs are generated tokens per item after the shared a0, thinking included; the inserted stop-thinking "
         "instruction is prompt. The selective THINK points are the C1 curve points nearest in cost (no interpolation, "
         "no paired test).", ""]
    for s, v in res["settings"].items():
        if v.get("status") != "complete" and not v.get("budgets"):
            L += [f"## {s}", "", f"TODO(missing: c1b runs of {s}: {v.get('status')})", ""]
            continue
        t = v["think_all"]
        L += [f"## {s} (n = {v['n']})", "",
              f"KEEP {v['keep_accuracy_pct']:.2f} %. THINK ALL (C1): {t['accuracy_pct']:.2f} %, Δ {fmt(t['delta_vs_keep'])}, "
              f"{t['generated_tokens_per_item']:.0f} tokens per item.", "",
              "| B | Accuracy % | Δ vs KEEP | Tok/item | Over budget % | Forced truncated % | No usable answer % | "
              "Repair % | Damage % | Verbal: share, Δ (tok) | Probe: share, Δ (tok) |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for b in BUDGETS:
            if str(b) not in v["budgets"]:
                L.append(f"| {b} | TODO(missing: run) | | | | | | | | | |")
                continue
            r = v["budgets"][str(b)]
            ns = r["nearest_selective"]
            cells = [f"{ns[sig]['share_pct']} %, {fmt(ns[sig]['delta_ci'])} ({ns[sig]['generated_tokens_per_item']:.0f})"
                     for sig in SIGNALS]
            ft = "—" if r["forced_truncated_pct_of_forced"] is None else f"{r['forced_truncated_pct_of_forced']:.2f}"
            L.append(f"| {b} | {r['accuracy_pct']:.2f} | {fmt(r['delta_vs_keep'])} | {r['generated_tokens_per_item']:.0f} | "
                     f"{r['over_budget_pct']:.1f} | {ft} | {r['no_usable_answer_pct']:.2f} | "
                     f"{r['repair_rate']['estimate']:.1f} | {r['damage_rate']['estimate']:.1f} | {cells[0]} | {cells[1]} |")
        L.append("")
    L += ["## Checks", ""]
    for c in res["checks"]:
        L.append(f"- {c}")
    L += ["", "Per-run details (run IDs, GPUs, round-trip counts, monitoring disagreements): `analysis/out/arr/c1b_budget.json`.", ""]
    (REPO / "reports" / "arr_c1_budget.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- main
def main():
    plan = plan_hash()
    assert b"### Amendment 7" in (REPO / "docs/planning/arr_round/plan.md").read_bytes(), "Amendment 7 is not in the plan"
    a0 = json.loads(pin(OUT / "a0_common_labels.json").read_text(encoding="utf-8"))
    assert a0["plan_core_sha256"] == plan_core_hash() and not a0["guards"]
    c1man = json.loads((OUT / "c1_think_manifest.json").read_text(encoding="utf-8"))
    for name in ("c1_labels.jsonl.gz", "c1_think.json"):
        assert sha(OUT / name) == c1man["outputs"][name], f"{name} differs from c1_think_manifest.json"
    c1 = json.loads(pin(OUT / "c1_think.json").read_text(encoding="utf-8"))
    c1_lab = {}
    for e in rows_gz(pin(OUT / "c1_labels.jsonl.gz")):
        if e["source"] == "C1":
            c1_lab.setdefault(e["setting"], {})[e["item_id"]] = e
    labels = list(rows_gz(pin(OUT / "math_common_labels.jsonl.gz")))
    table = {}
    for r in rows(pin(J1 / "artifacts/math/item_evaluations.jsonl")):
        table.setdefault(r["model"], {}).setdefault(r["condition"], {})[r["item_id"]] = r
    math_pops = load_math(table, labels)
    pops = {"math 4B": math_pops["4B"], "math 8B": math_pops["8B"], "MMLU-Pro 4B": load_mc("4B")}
    mc_items = unique(rows(pin(REPO / "data/processed/mmlupro/items.jsonl")))

    found, status = {}, {}
    for s, (task, model) in SETTINGS.items():
        if c1["settings"][s]["status"] != "complete":
            status[s] = "C1 not complete"
            continue
        runs, smoke = c1b_runs(task, model)
        if not runs:
            status[s] = "no c1b run synced"
            continue
        loaded = {b: load_run(d, task, model, b, pops[s]["ids"]) for b, d in sorted(runs.items())}
        sm = load_run(smoke, task, model, 256, None) if smoke is not None else None
        found[s] = (loaded, sm, smoke)
        status[s] = "complete" if set(runs) == set(BUDGETS) else f"{len(runs)} of {len(BUDGETS)} budgets synced"
    commits = {info["code_commit"] for loaded, sm, _ in found.values() for _, info in
               list(loaded.values()) + ([sm] if sm else [])}
    assert len(commits) <= 1, ("c1b runs from more than one code commit (CLAUDE.md rule 8)", commits)

    eq = Equivalence()
    try:
        # phase 1: forced answers, gold-blind; frozen and hashed before any gold is read
        extractions = []
        for s, (loaded, _, _) in found.items():
            task, _ = SETTINGS[s]
            for b, (recs, _) in loaded.items():
                for i in pops[s]["ids"]:
                    r = recs[i]
                    if r["reused"]:
                        continue
                    g = {"text": forced_text(r), "finish_reason": r["finish_reason"], "out_tokens": r["generated_tokens"],
                         "prompt_tokens": r["prompt_tokens"]}
                    assert think_body(g["text"]) is not None
                    e = extract(task, g, gold_free(mc_items[i]) if task == "mc" else None, eq)
                    extractions.append({"setting": s, "budget": b, "item_id": i, **e})
        frozen = OUT / "c1b_extractions.jsonl.gz"
        write_rows_gz(frozen, extractions)
        frozen_hash = sha(frozen)
        print(f"extractions frozen: {len(extractions)} rows, sha256 {frozen_hash[:16]}", flush=True)
        # phase 2: scoring against gold
        gold = {r["item_id"]: r["gold"] for p in (pin(REPO / "data/processed/dnat.jsonl"),
                                                  pin(REPO / "data/processed/ext/dnat_ext.jsonl")) for r in rows(p)}
        for e in extractions:
            if e["status"] != "ok":
                e["correct"] = 0
            elif SETTINGS[e["setting"]][0] == "math":
                e["correct"] = int(eq(e["pred"], gold[e["item_id"]]))
            else:
                e["correct"] = int(e["pred"] == mc_items[e["item_id"]]["gold"])
        guards = list(eq.guards)
    finally:
        eq.close()
    assert sha(frozen) == frozen_hash
    guards = guard_check(guards)
    write_rows_gz(OUT / "c1b_labels.jsonl.gz", extractions)

    res = {"plan_sha256_lf": plan, "plan_core_sha256": plan_core_hash(), "extractions_sha256_before_gold": frozen_hash,
           "scorer_guards": guards, "budgets": list(BUDGETS), "signals": list(SIGNALS), "settings": {}, "checks": []}
    for s in SETTINGS:
        if s not in found:
            res["settings"][s] = {"status": status[s]}
            continue
        P, (loaded, sm, smoke_dir) = pops[s], found[s]
        lab = c1_lab[s]
        T = {"y": np.array([lab[i]["correct"] for i in P["ids"]], dtype=float),
             "gen": np.array([lab[i]["out_tokens"] for i in P["ids"]], dtype=float)}
        ref = c1["settings"][s]["policies"]["THINK ALL"]
        assert abs(ref["accuracy_pct"] - 100 * T["y"].mean()) < 1e-9 and abs(ref["generated_tokens_per_item"] - T["gen"].mean()) < 1e-9, s
        boot = Boot(P["canonical"])
        think_curves = {sig: curve(P, T["y"], T["gen"], sig) for sig in SIGNALS}
        entry = {"status": status[s], "n": P["n"], "keep_accuracy_pct": 100 * float(P["y0"].mean()),
                 "think_all": {"accuracy_pct": 100 * float(T["y"].mean()), "delta_vs_keep": boot.ci(T["y"] - P["y0"]),
                               "generated_tokens_per_item": float(T["gen"].mean())},
                 "budgets": {}, "runs": {}}
        for b, (recs, info) in loaded.items():
            lab_b = {e["item_id"]: e for e in extractions if e["setting"] == s and e["budget"] == b}
            B = budget_policy(P, T, lab, recs, lab_b)
            entry["budgets"][str(b)] = describe_budget(P, T, B, boot, think_curves)
            disagree = sum(info["monitor"][i] != bool(B["y"][j]) for j, i in enumerate(P["ids"]))
            entry["runs"][str(b)] = {k: v for k, v in info.items() if k != "monitor"} | {"monitor_disagreements": int(disagree)}
        if sm is not None:
            recs_s, info_s = sm
            c = info_s["counts"]
            same = None
            if 256 in loaded:   # determinism: the smoke items again inside the full B = 256 run
                full = loaded[256][0]
                same = sum(full[i] == r for i, r in recs_s.items())
            entry["smoke"] = {"run": info_s["run"], "items": len(recs_s), "prefix_identical": c.get("prefix_identical"),
                              "prefix_items": c.get("prefix_items"), "records_identical_in_full_run": same}
            pc = json.loads(verified(smoke_dir, "prefix_check.json")[0].read_text(encoding="utf-8"))
            assert pc["identical_prefix"] == c.get("prefix_identical"), "prefix_check.json and metrics disagree"
            res["checks"].append(f"{s}: prefix check {pc['identical_prefix']} of {pc['items']} stored traces reproduced "
                                 f"(first 64 tokens); smoke records identical in the full B = 256 run: "
                                 f"{'not run' if same is None else f'{same} of {len(recs_s)}'}.")
        rt = sorted({r["counts"].get("roundtrip_mismatch", 0) for r in entry["runs"].values()})
        pl = sorted({r["counts"].get("prompt_len_mismatch", 0) for r in entry["runs"].values()})
        res["checks"].append(f"{s}: round-trip token-count mismatches {rt} (counted, not corrected); prompt-length "
                             f"mismatches {pl}; monitoring disagreements "
                             f"{[entry['runs'][k]['monitor_disagreements'] for k in sorted(entry['runs'], key=int)]}.")
        res["settings"][s] = entry
    res = clean(res)
    dump(OUT / "c1b_budget.json", res)
    report(res)
    manifest("c1b_budget", {"outputs": {p: sha(OUT / p) for p in ("c1b_extractions.jsonl.gz", "c1b_labels.jsonl.gz",
                                                                  "c1b_budget.json")},
                            "settings_status": status})
    print(json.dumps({"status": status, **{s: {b: round(v["budgets"][b]["delta_vs_keep"]["estimate"], 2)
                                                for b in v.get("budgets", {})} for s, v in res["settings"].items()}}, indent=1))


if __name__ == "__main__":
    main()
