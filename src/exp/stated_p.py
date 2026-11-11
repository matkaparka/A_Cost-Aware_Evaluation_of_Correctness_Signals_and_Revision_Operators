"""Stated-reliability condition F3s(p) (reports/07_paper_review.md #2; docs/planning/next_experiments/plan.md Group 1).
Exploratory, post-freeze; no confirmatory test.

The frozen F3(p) wrapper never tells the model how reliable the checker is. F3s(p) states it:
  "Feedback: a checker whose judgments are right {pct}% of the time reviewed the answer above and judged it {v}."
for p in {0.5, 0.7, 0.9, 1.0}. The verdict of each item at each p is the frozen F3(p) draw (src/data/verifiers.py
f3_verdict, same stream and seed), so F3s(p) and the main run's F3(p) rows are paired item by item and verdict by
verdict; only the stated reliability differs. Everything else is the main runner's (src/exp/runner.py): review
prompt, a0 from the P1 tables, greedy vLLM with VLLM_BATCH_INVARIANT=1, frozen math scorer, revision budget 2048.

Exp 0 (calibration control): the no-feedback first turn is regenerated greedily (batch-invariant, so it should equal
the main run's; metrics.json records how many first turns differ from --main-generations when given), then F3s(p)
per world with EXP0_INSTRUCTION.

The wrapper was fixed before any output was seen; dev is used for validity only (parse and truncation rates).
Usage: VLLM_BATCH_INVARIANT=1 python -m src.exp.stated_p --model 4B --split dev --limit 100 --tag smoke
       VLLM_BATCH_INVARIANT=1 python -m src.exp.stated_p --model 4B --split test --allow-test
           [--main-generations results/main/4B/main-exp01-test-4B-a7a0579/generations.jsonl]
"""
import argparse
import json
import time
from collections import Counter, defaultdict

from ..data.verifiers import f3_verdict
from ..gen import render as R
from ..runlib import MODELS, REPO, register, require_batch_invariant, run_dir, sha256_text, write_manifest
from ..score.scorer import SCORER_VERSION, extract_option
from .runner import P_LEVELS, Plan, generate, jl, score_rows

STATED = "Feedback: a checker whose judgments are right {pct}% of the time reviewed the answer above and judged it {v}."
POOLS = (("data/processed/dnat.jsonl", "results/p1/{m}/p1-initial-{m}-{c}/initial.jsonl"),
         ("data/processed/ext/dnat_ext.jsonl", "results/p1/{m}/p1-initial-ext-{m}-{c}/initial.jsonl"))
P1_COMMITS = {"4B": ("d2ebdea", "45575c2")}     # the locked P1 runs the main 4B test analysis uses (configs/p6_spec_4B_test.json)


def stated(p, verdict):
    return STATED.format(pct=round(100 * p), v="correct" if verdict else "incorrect")


def load_dnat(model, split, limit):
    rows = []
    for (items, init), c in zip(POOLS, P1_COMMITS[model]):
        meta = {r["item_id"]: r for r in jl(REPO / items)}
        for r in jl(REPO / init.format(m=model, c=c)):
            if r["split"] == split:
                m = meta[r["item_id"]]
                rows.append({"item_id": r["item_id"], "dataset": m["dataset"], "question": m["question"], "gold": m["gold"],
                             "a0": r["a0"], "correct0": r["correct"]})
    return rows[:limit] if limit else rows


def build_stage1(tok, args, dnat, exp0):
    plan = Plan()
    for r in dnat:
        base = {"item_id": r["item_id"], "dataset": r["dataset"], "data": "dnat", "correct0": r["correct0"]}
        for p in P_LEVELS:
            v = f3_verdict(r["item_id"], r["correct0"], p)
            plan.add({**base, "exp": "1A", "cond": f"F3s({p})", "verdict": v},
                     R.review_prompt(tok, r["question"], r["a0"], feedback=stated(p, v)), args.revision_tokens)
    for tid in sorted({r["template_id"] for r in exp0}):
        prompt = next(r["prompt"] for r in exp0 if r["template_id"] == tid)
        plan.add({"item_id": tid, "data": "exp0", "exp": "0", "cond": "initial", "stage": 1},
                 R.Rendered(text=R.render_messages(tok, [{"role": "user", "content": prompt}]), spans={}), args.exp0_tokens)
    return plan


def build_stage2(tok, args, plan1, out1, exp0):
    plan = Plan()
    initial = {row["item_id"]: out1[row["prompt_key"]]["text"] for row in plan1.rows if row.get("exp") == "0"}
    for r in exp0:
        first = initial[r["template_id"]]
        choice = extract_option(first)
        correct0 = choice == r["gold_letter"]
        base = {"item_id": r["item_id"], "data": "exp0", "exp": "0", "world": r["world"], "correct0": correct0,
                "valid0": choice is not None}
        for p in P_LEVELS:
            v = f3_verdict(r["item_id"], correct0, p)
            msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": first},
                    {"role": "user", "content": stated(p, v) + "\n\n" + R.EXP0_INSTRUCTION}]
            plan.add({**base, "cond": f"F3s({p})", "verdict": v},
                     R.Rendered(text=R.render_messages(tok, msgs), spans={}), args.exp0_tokens)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(P1_COMMITS))
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--revision-tokens", type=int, default=2048)
    ap.add_argument("--exp0-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="first N D-nat items and N // 2 Exp 0 templates (smoke)")
    ap.add_argument("--main-generations", help="the main run's generations.jsonl, to check the Exp 0 first turns")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("--split test needs --allow-test (post-freeze exploratory run)")
    cfg = {**vars(args), "scorer_version": SCORER_VERSION, "stated_wrapper": STATED, "p_levels": P_LEVELS,
           "wording_sha256": sha256_text(json.dumps([R.PROMPT, R.LABELS, R.INSTRUCTION, STATED, R.EXP0_INSTRUCTION]))}
    dnat = load_dnat(args.model, args.split, args.limit)
    exp0 = [r for r in jl(REPO / "data" / "processed" / "exp0.jsonl") if r["split"] == args.split]
    if args.limit:
        keep = sorted({r["template_id"] for r in exp0})[:max(1, args.limit // 2)]
        exp0 = [r for r in exp0 if r["template_id"] in keep]
    golds = {r["item_id"]: r["gold"] for r in dnat}
    exp0_gold = {r["item_id"]: r["gold_letter"] for r in exp0}
    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODELS[args.model])
        plan1 = build_stage1(tok, args, dnat, exp0)
        print(json.dumps({"dnat": len(dnat), "exp0 templates": len({r['template_id'] for r in exp0}),
                          "stage1 prompts": len(plan1.prompts), "conditions": Counter(r["cond"] for r in plan1.rows)}, indent=1))
        print(plan1.prompts[plan1.rows[0]["prompt_key"]][0].text[-400:])
        return
    require_batch_invariant()
    import vllm
    from vllm import LLM, SamplingParams
    tag = f"-{args.tag}" if args.tag else ""
    run_id, d = run_dir("fu" if args.split == "test" else "fu-dev", args.model, f"stated-p-{args.split}{tag}")
    register(run_id, "running", block="stated-p", exp="stated-p", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=12288)
    tok = llm.get_tokenizer()
    plan1 = build_stage1(tok, args, dnat, exp0)
    t1 = time.time()
    out1 = generate(llm, plan1, SamplingParams)
    plan2 = build_stage2(tok, args, plan1, out1, exp0)
    t2 = time.time()
    out2 = generate(llm, plan2, SamplingParams)
    t3 = time.time()
    with open(d / "generations.jsonl", "w", encoding="utf-8") as f:
        for k, o in {**out1, **out2}.items():
            f.write(json.dumps({"prompt_key": k, **o}, ensure_ascii=False) + "\n")
    score_rows(plan1, out1, golds, exp0_gold)
    score_rows(plan2, out2, golds, exp0_gold)
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for row in plan1.rows + plan2.rows:
            if row.get("stage") != 1:
                f.write(json.dumps(row) + "\n")
    exp0_check = None
    if args.main_generations:   # same prompt -> same key; batch-invariant greedy should give the same first turn
        main_gen = {g["prompt_key"]: g["text"] for g in jl(args.main_generations)}
        keys = [row["prompt_key"] for row in plan1.rows if row.get("exp") == "0"]
        exp0_check = {"first turns": len(keys), "found in main run": sum(k in main_gen for k in keys),
                      "identical": sum(main_gen.get(k) == out1[k]["text"] for k in keys)}
    stats = defaultdict(Counter)
    for row in plan1.rows + plan2.rows:
        if row.get("stage") != 1:
            stats[f"{row['exp']}:{row['data']}:{row['cond']}"]["n"] += 1
            stats[f"{row['exp']}:{row['data']}:{row['cond']}"]["truncated"] += row.get("finish_reason") == "length"
            stats[f"{row['exp']}:{row['data']}:{row['cond']}"]["not ok"] += row.get("status") not in ("ok", "fixed")
    metrics = {"model": args.model, "split": args.split, "vllm": vllm.__version__, "load_s": t1 - t0,
               "stage1_s": t2 - t1, "stage2_s": t3 - t2, "stage1_prompts": len(plan1.prompts),
               "stage2_prompts": len(plan2.prompts), "rows": {k: dict(v) for k, v in sorted(stats.items())},
               "exp0_first_turn_check": exp0_check}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__},
                   {"wording": cfg["wording_sha256"]})
    register(run_id, "done", block="stated-p", exp="stated-p", model=args.model)
    print(json.dumps({"run_id": run_id, **{k: metrics[k] for k in ("stage1_prompts", "stage2_prompts", "stage1_s", "stage2_s")},
                      "exp0_check": exp0_check}))


if __name__ == "__main__":
    main()
