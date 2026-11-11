"""ARR-round generation runs (DEVIATIONS D20; docs/planning/arr_round/plan.md, tiers B and C1, Amendment 2).
Exploratory, after the freeze. Every stage writes its generations first and scores after; the scores stored here use
the frozen scorers for monitoring only, and the analysis re-scores the stored texts gold-blind (plan §0, A0).

Stages (math items: data/processed/dnat.jsonl + ext/dnat_ext.jsonl; MMLU-Pro items: src/data/mmlupro.py)
  samples           K sampled answers to the initial prompt (B1; B2a with --k 1 --max-tokens 8192). The prompt,
                    sampling settings (T 0.7, top-p 0.8, top-k 20) and seed function are those of
                    src/gen/h2a_extra.py `consistency`; the per-request seed comes from the named stream --stream.
                    No gold is read.
  continue          greedy continuation of every a0 that hit the 4,096-token cap, from the initial prompt plus a0's own
                    token IDs, for up to --extra-tokens more tokens (B2b).
  ops               F0 / F2 / F2-scratch on D-nat (B3 with --wording p1|p2; B5 on dev with --wording main), built with
                    the calls, budgets, greedy decoding and prompt keys of src/exp/runner.py build_stage1/2.
  samples-critique  one review turn after a0 that lists the final answers of five sampled attempts (B4), for the items
                    of a selection file written gold-blind by analysis/arr/b4_select.py.
  think             thinking-mode fresh re-answer (C1): the B1 prompt of src/exp/thinking.py (the initial prompt, a0
                    not in context, the model's own think block left open), its sampling settings, seed index 0.
  think-budget      thinking on every item with at most --budget thinking tokens (plan Amendment 7), built from the
                    stored C1 traces: an item whose stored thinking fits the budget keeps its C1 output; any other gets
                    its C1 prompt, the trace through the budget-th thinking token, the stop-thinking instruction of the
                    Qwen3 report, and a forced answer. --prefix-check N first regenerates 64 tokens of N items with the
                    C1 seed and compares them with the stored traces.
Large stages take --chunk-size / --chunk-index over the items in the initial file's order (the seeded order).
All generation needs VLLM_BATCH_INVARIANT=1. --dry-run builds every request with a tokenizer only (no model, no GPU).
Usage: VLLM_BATCH_INVARIANT=1 python -m src.exp.arr_runs <stage> --task math --model 4B --split test --allow-test
       --initial <initial.jsonl> [--initial ...] [stage options] [--dry-run [--tokenizer PATH]]
"""
import argparse
import json
import time
from collections import Counter

from ..data import mmlupro as M
from ..data.build import h
from ..gen import render as R
from ..gen.h2a_extra import TEMPERATURE, TOP_K, TOP_P, seed_for
from ..runlib import MODELS, REPO, needs_bos, register, require_batch_invariant, run_dir, sha256_text, write_manifest
from ..score import mc as S
from ..score.scorer import SCORER_VERSION, score, strip_thinking
from .runner import Plan, generate
from .thinking import SAMPLING as THINK_SAMPLING
from .thinking import thinking

ITEM_TABLES = ("data/processed/dnat.jsonl", "data/processed/ext/dnat_ext.jsonl")
A0_TOKENS, REVISION_TOKENS, CRITIQUE_TOKENS = 4096, 2048, 1024
# B3: two paraphrases of each first-turn operator instruction, fixed before any run (plan Amendment 2). The
# second-turn instruction (R.INSTRUCTION) is the same for every wording.
WORDINGS = {
    "main": {"F2": R.CRITIQUE_TURN, "F2-scratch": R.SCRATCH_TURN},
    "p1": {"F2": "Check the answer above carefully and describe every mistake you can identify. Do not state a final answer yet.",
           "F2-scratch": "Work through the problem once more from the beginning, showing each step. Do not state a final answer yet."},
    "p2": {"F2": "Go through the answer above step by step and point out anything that is wrong. Hold off on giving a final answer.",
           "F2-scratch": "Start over and solve the problem on your own, step by step. Hold off on giving a final answer."},
}
# B4: the sample-aware review instruction (one turn after a0). {what} is "problem" (math) or "question" (MMLU-Pro).
SAMPLE_AWARE = ("Five other attempts at this {what} ended with these final answers: {answers}. Taking them into account, "
                "review your answer above and give your final answer. Put your final answer within \\boxed{{}}.")
NO_ANSWER = "no final answer"
EXP = {"samples": "b1", "continue": "b2", "ops": "b3", "samples-critique": "b4", "think": "c1", "think-budget": "c1b"}
# Amendment 7: the Qwen3 technical report's stop-thinking instruction ("Thinking Budget", p. 11), preceded by "\n\n".
STOP_THINKING = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
                 "directly now.\n</think>.\n\n")
BUDGET_ANSWER_TOKENS = 2048
C1_RUNS = "results/arr/c1/{model}/c1-think-{task}-test-c*-{model}-279e6b0"


def jl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def math_items():
    q = {}
    for t in ITEM_TABLES:
        q.update({r["item_id"]: r for r in jl(REPO / t)})
    return q


def initial_rows(paths, split):
    rows = [r for f in paths for r in jl(f) if r["split"] == split]
    assert len({r["item_id"] for r in rows}) == len(rows), "duplicate items across the initial files"
    return rows


def chunk(rows, args):
    if args.chunk_size:
        rows = rows[args.chunk_index * args.chunk_size:(args.chunk_index + 1) * args.chunk_size]
    return rows[:args.limit] if args.limit else rows


def initial_prompt(tok, question):
    """The P1 initial-answer prompt text (src/gen/initial.py, src/gen/h2a_extra.py, R.fresh_prompt)."""
    return tok.apply_chat_template([{"role": "user", "content": R.PROMPT.format(q=question)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def sample_aware_instruction(answers, what):
    assert len(answers) == 5, answers
    return SAMPLE_AWARE.format(what=what, answers="; ".join(NO_ANSWER if a is None else str(a) for a in answers))


def think_seed(item_id):
    return int(h("think-seed-v1", item_id, 0)[:8], 16)


def question_of(task, item_id, q, its):
    return q[item_id]["question"] if task == "math" else M.render_question(its[item_id])


def score_text(task, text, finish_reason, item_id, q, its):
    if task == "math":
        return score(text, q[item_id]["gold"], finish_reason=finish_reason)
    body = strip_thinking(text)   # None when a think block is left open (scored wrong, as the math scorer does)
    if body is None:
        return {"status": "unclosed", "pred": None, "correct": False}
    if "</think>" in body:         # a closing tag whose opening tag was not in the text: keep the answer part only
        body = body.rsplit("</think>", 1)[1].strip()
    return S.score(body, its[item_id], finish_reason)


# ----------------------------------------------------------------------------- request builders (no model needed)
def build_samples(tok, rows, q):
    return [initial_prompt(tok, q[r["item_id"]]["question"]) for r in rows]


def build_continue(tok, rows, q):
    out = []
    for r in rows:
        prefix = initial_prompt(tok, q[r["item_id"]]["question"])
        ids = tok(prefix, add_special_tokens=needs_bos(tok, prefix)).input_ids
        out.append(ids + list(r["a0_token_ids"]))
    return out


def build_ops_stage1(tok, rows, q, conds, wording):
    plan = Plan()
    words = WORDINGS[wording]
    for r in rows:
        base = {"item_id": r["item_id"], "dataset": r.get("dataset"), "data": "dnat", "correct0": r["correct"]}
        qq, a0 = q[r["item_id"]]["question"], r["a0"]
        if "F0" in conds:
            plan.add({**base, "exp": "1A", "cond": "F0"}, R.review_prompt(tok, qq, a0), REVISION_TOKENS)
        for cond in ("F2", "F2-scratch"):
            if cond in conds:
                plan.add({**base, "exp": "1A", "cond": f"{cond}:turn1", "stage": 1},
                         R.review_prompt(tok, qq, a0, instruction=words[cond]), CRITIQUE_TOKENS)
    return plan


def build_ops_stage2(tok, plan1, out1, rows, q, wording):
    plan = Plan()
    words = WORDINGS[wording]
    a0 = {r["item_id"]: r["a0"] for r in rows}
    for row in plan1.rows:
        if row["cond"].endswith(":turn1"):
            cond = row["cond"].split(":")[0]
            hist = [(words[cond], out1[row["prompt_key"]]["text"])]
            plan.add({k: row[k] for k in ("item_id", "dataset", "data", "correct0")} |
                     {"exp": "1A", "cond": cond, "turn1_key": row["prompt_key"]},
                     R.review_prompt(tok, q[row["item_id"]]["question"], a0[row["item_id"]], history=hist), REVISION_TOKENS)
    return plan


def build_samples_critique(tok, rows, task, q, its, select):
    plan = Plan()
    what = "problem" if task == "math" else "question"
    for r in rows:
        i = r["item_id"]
        plan.add({"item_id": i, "data": task, "exp": "B4", "cond": "F-samples", "correct0": r["correct"]},
                 R.review_prompt(tok, question_of(task, i, q, its), r["a0"],
                                 instruction=sample_aware_instruction(select[i]["answers"], what)), REVISION_TOKENS)
    return plan


def build_think(tok, rows, task, q, its):
    return [thinking(R.fresh_prompt(tok, question_of(task, r["item_id"], q, its))) for r in rows]


def c1_traces(task, model):
    """The stored C1 generations of a setting by item: every chunk of the C1 run (the smoke run excluded)."""
    import glob
    from pathlib import Path
    dirs = sorted(glob.glob(str(REPO / C1_RUNS.format(model=model, task=task))))
    if not dirs:
        raise SystemExit(f"no stored C1 runs for {task} {model}")
    out = {}
    for d in dirs:
        for r in jl(Path(d) / "generations.jsonl"):
            assert r["item_id"] not in out, ("item in two C1 chunks", r["item_id"])
            out[r["item_id"]] = r
    return out


def budget_seed(item_id, budget):
    return int(h("think-budget-v1", item_id, budget)[:8], 16)


def budget_cut(ids, open_id, close_id, budget):
    """(thinking tokens of the stored trace, index after the budget-th thinking token or None when the trace fits).

    The thinking part is the tokens after the first opening tag up to the closing tag; without a closing tag, every
    token after the opening tag. A trace without an opening tag has no thinking (it fits any budget)."""
    if open_id not in ids:
        return 0, None
    o = ids.index(open_id)
    c = ids.index(close_id, o + 1) if close_id in ids[o + 1:] else len(ids)
    think = c - (o + 1)
    return think, (None if think <= budget else o + 1 + budget)


def build_think_budget(tok, rows, task, q, its, traces, budget):
    """Per item: (prompt token IDs, or None when the stored trace fits the budget; record), and check counts.

    A forced item's prompt is its C1 prompt, the stored tokens through the budget-th thinking token, and the
    stop-thinking instruction. The C1 prompt must re-tokenize to the stored prompt_tokens (checked by the caller)."""
    open_id, close_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    stop_ids = tok(STOP_THINKING, add_special_tokens=False).input_ids
    assert stop_ids.count(close_id) == 1, "the stop-thinking instruction must close the think block once"
    out, stats = [], Counter()
    stats["stop_tokens"] = len(stop_ids)
    for r in rows:
        tr = traces[r["item_id"]]
        ids = tok(tr["text"], add_special_tokens=False).input_ids
        # vLLM's token_ids keep the end-of-turn token of an output that stopped, and its text does not
        stats["roundtrip_mismatch"] += len(ids) + (tr["finish_reason"] == "stop") != tr["out_tokens"]
        think, cut = budget_cut(ids, open_id, close_id, budget)
        rec = {"item_id": r["item_id"], "budget": budget, "think_tokens_stored": think,
               "think_tokens_used": min(think, budget)}
        if cut is None:
            stats["fits"] += 1
            out.append((None, {**rec, "reused": True}))
            continue
        prompt = build_think(tok, [r], task, q, its)[0]
        pids = tok(prompt, add_special_tokens=needs_bos(tok, prompt)).input_ids
        stats["prompt_len_mismatch"] += len(pids) != tr["prompt_tokens"]
        stats["over_budget"] += 1
        out.append((pids + ids[:cut] + stop_ids, {**rec, "reused": False, "trace_tokens_used": cut,
                                                  "prefix_text": tok.decode(ids[:cut])}))
    return out, stats


def think_budget_main(args, q, its, rows, cfg):
    """Amendment 7. Writes generations.jsonl, rows.jsonl (monitoring scores only), prefix_check.json (with
    --prefix-check), metrics.json and the manifest.

    generations.jsonl has one record per item and no copy of a stored C1 text (those are in the C1 runs): a reused
    item keeps its C1 output, whose finish reason and token counts are copied; a forced item's full text is the first
    trace_tokens_used tokens of its C1 trace, decoded, then STOP_THINKING, then answer_text."""
    traces = c1_traces(args.task, args.model)
    assert {r["item_id"] for r in rows} <= set(traces), "items without a stored C1 trace"
    if args.dry_run:
        tok = tokenizer(args)
        built, stats = build_think_budget(tok, rows, args.task, q, its, traces, args.budget)
        lens = [len(p) for p, _ in built if p is not None]
        worst = (max(lens) if lens else 0) + BUDGET_ANSWER_TOKENS
        out = {"stage": args.stage, "model": args.model, "task": args.task, "budget": args.budget, "items": len(rows),
               "c1_items": len(traces), **dict(stats), "over_pct": round(100 * stats["over_budget"] / len(rows), 2),
               "think_tokens_used_mean": round(sum(r["think_tokens_used"] for _, r in built) / len(rows), 1),
               "max_prompt_plus_answer": worst, "max_model_len": max_len_for(args), "fits": worst <= max_len_for(args)}
        print(json.dumps(out, indent=1))
        assert out["fits"], "a request exceeds max_model_len"
        assert not stats["prompt_len_mismatch"], "rebuilt C1 prompts differ from the stored prompt_tokens"
        return out
    import vllm
    from vllm import SamplingParams
    kind = f"think-budget-{args.task}-{args.split}-b{args.budget}" + (f"-{args.tag}" if args.tag else "")
    run_id, d = run_dir(EXP[args.stage], args.model, kind)
    register(run_id, "running", block=f"arr-{args.stage}", exp="arr", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    llm = load_llm(args, max_len_for(args))
    tok = llm.get_tokenizer()
    t1 = time.time()
    counts = Counter()
    if args.prefix_check:   # the reuse premise: seeded, batch-invariant decoding reproduces the stored trace's start
        sub = rows[:args.prefix_check]
        params = [SamplingParams(**THINK_SAMPLING, max_tokens=64, seed=think_seed(r["item_id"])) for r in sub]
        outs = llm.generate(build_think(tok, sub, args.task, q, its), params, use_tqdm=False)
        # a 64-token cut can end inside a multi-byte character, which detokenizes as U+FFFD
        same = {r["item_id"]: traces[r["item_id"]]["text"].startswith(o.outputs[0].text.rstrip("�"))
                for r, o in zip(sub, outs)}
        (d / "prefix_check.json").write_text(json.dumps({"items": len(sub), "identical_prefix": sum(same.values()),
                                                         "first_tokens": 64, "per_item": same}, indent=1))
        counts["prefix_items"], counts["prefix_identical"] = len(sub), sum(same.values())
    built, stats = build_think_budget(tok, rows, args.task, q, its, traces, args.budget)
    if stats["prompt_len_mismatch"]:
        raise SystemExit(f"{stats['prompt_len_mismatch']} rebuilt C1 prompts differ from the stored prompt_tokens")
    counts.update(stats)
    req = [i for i, (p, _) in enumerate(built) if p is not None]
    params = [SamplingParams(**THINK_SAMPLING, max_tokens=BUDGET_ANSWER_TOKENS,
                             seed=budget_seed(rows[i]["item_id"], args.budget)) for i in req]
    outs = llm.generate([{"prompt_token_ids": built[i][0]} for i in req], params, use_tqdm=False) if req else []
    t2 = time.time()
    forced = dict(zip(req, outs))
    with open(d / "generations.jsonl", "w", encoding="utf-8") as f, open(d / "rows.jsonl", "w", encoding="utf-8") as g:
        for i, (r, (_, rec)) in enumerate(zip(rows, built)):
            tr = traces[r["item_id"]]
            if rec["reused"]:
                text = tr["text"]
                out = {**rec, "finish_reason": tr["finish_reason"], "generated_tokens": tr["out_tokens"],
                       "prompt_tokens": tr["prompt_tokens"]}
            else:
                o = forced[i]
                c = o.outputs[0]
                assert list(o.prompt_token_ids) == built[i][0], ("prompt changed in vLLM", r["item_id"])
                text = rec["prefix_text"] + STOP_THINKING + c.text
                out = {**{k: v for k, v in rec.items() if k != "prefix_text"}, "answer_text": c.text,
                       "finish_reason": c.finish_reason, "answer_tokens": len(c.token_ids),
                       "generated_tokens": rec["trace_tokens_used"] + len(c.token_ids),
                       "prompt_tokens": len(o.prompt_token_ids)}
                counts["forced_truncated"] += c.finish_reason == "length"
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            counts["generated_tokens"] += out["generated_tokens"]
            s = score_text(args.task, text, out["finish_reason"], r["item_id"], q, its)
            g.write(json.dumps({"item_id": r["item_id"], "cond": f"think-b{args.budget}", "correct0": r["correct"],
                                "reused": rec["reused"], **s, "generated_tokens": out["generated_tokens"],
                                "finish_reason": out["finish_reason"]}) + "\n")
            counts["right"] += bool(s["correct"])
    metrics = {"stage": args.stage, "budget": args.budget, "items": len(rows), "vllm": vllm.__version__,
               "load_s": t1 - t0, "gen_s": t2 - t1, "score_s": time.time() - t2, "counts": dict(counts)}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__},
                   {"initial": R.PROMPT, "stop_thinking": STOP_THINKING})
    register(run_id, "done", block=f"arr-{args.stage}", exp="arr", model=args.model)
    print(json.dumps({"run_id": run_id, "dir": str(d), **metrics}, default=str))


# ----------------------------------------------------------------------------- stages
def load_llm(args, max_len):
    require_batch_invariant()
    from vllm import LLM
    return LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=max_len)


def tokenizer(args):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(args.tokenizer or MODELS[args.model])


def max_len_for(args):
    # samples: the stored Qwen3-4B run used 8192 (src/gen/h2a_extra.py); ops and think: as runner.py and thinking.py
    return {"samples": max(8192, args.max_tokens + 2048), "continue": A0_TOKENS + args.extra_tokens + 2048, "ops": 12288,
            "samples-critique": 12288, "think": args.max_tokens + 8192,
            "think-budget": (args.budget or 0) + 4096 + BUDGET_ANSWER_TOKENS}[args.stage]


def dry_report(args, tok, texts=None, token_lists=None, plans=(), extra=None):
    lens = [len(t) for t in token_lists] if token_lists is not None else \
        [len(tok(t, add_special_tokens=needs_bos(tok, t)).input_ids) for t in (texts or [])]
    for p in plans:
        lens += [len(tok(r.text + r.prefill, add_special_tokens=needs_bos(tok, r.text)).input_ids) + mt for r, mt in p.prompts.values()]
    budget = {"samples": args.max_tokens, "continue": args.extra_tokens, "think": args.max_tokens}.get(args.stage, 0)
    worst = max(lens) + budget if lens else 0
    out = {"stage": args.stage, "model": args.model, "task": args.task, "split": args.split, "requests": len(lens),
           "max_prompt_plus_budget": worst, "max_model_len": max_len_for(args), "fits": worst <= max_len_for(args), **(extra or {})}
    print(json.dumps(out, indent=1, default=str))
    assert out["fits"], "a request exceeds max_model_len"
    return out


def stage_main(args):
    q = math_items() if args.task == "math" else {}
    its = {it["item_id"]: it for it in M.load_items()} if args.task == "mc" else {}
    rows = initial_rows(args.initial, args.split)
    if args.stage == "continue":
        rows = [r for r in rows if r["finish_reason"] == "length"]
    select = None
    if args.stage == "samples-critique":
        select = {r["item_id"]: r for r in jl(args.select)}
        rows = [r for r in rows if r["item_id"] in select]
        assert len(rows) == len(select), "selection items missing from the initial files"
    rows = chunk(rows, args)
    if not rows:
        raise SystemExit("no items")
    wording_hash = sha256_text(json.dumps(WORDINGS, sort_keys=True))
    cfg = {**vars(args), "scorer_version": SCORER_VERSION, "a0_tokens": A0_TOKENS, "revision_tokens": REVISION_TOKENS,
           "critique_tokens": CRITIQUE_TOKENS, "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K},
           "think_sampling": THINK_SAMPLING, "wordings_sha256": wording_hash, "sample_aware": SAMPLE_AWARE,
           "items": len(rows)}
    conds = args.conds.split(",") if args.conds else []
    if args.stage == "think-budget":
        return think_budget_main(args, q, its, rows, {**cfg, "budget": args.budget, "stop_thinking": STOP_THINKING,
                                                      "budget_answer_tokens": BUDGET_ANSWER_TOKENS})
    if args.dry_run:
        tok = tokenizer(args)
        if args.stage == "samples":
            return dry_report(args, tok, texts=build_samples(tok, rows, q))
        if args.stage == "continue":
            ids = build_continue(tok, rows, q)
            mism = sum(tok.decode(r["a0_token_ids"]) != r["a0"] for r in rows)
            return dry_report(args, tok, token_lists=ids, extra={"a0_decode_mismatches": mism})
        if args.stage == "ops":
            return dry_report(args, tok, plans=[build_ops_stage1(tok, rows, q, conds, args.wording)],
                              extra={"note": "stage-2 lengths depend on stage-1 outputs; budget 1024 + 2048 within 12288"})
        if args.stage == "samples-critique":
            return dry_report(args, tok, plans=[build_samples_critique(tok, rows, args.task, q, its, select)])
        return dry_report(args, tok, texts=build_think(tok, rows, args.task, q, its))
    import vllm
    from vllm import SamplingParams
    kind = {"samples": f"samples-{args.split}-k{args.k}-{args.stream}-t{args.max_tokens}",
            "continue": f"continue-{args.split}-x{args.extra_tokens}",
            "ops": f"ops-{args.split}-{args.wording}-{'+'.join(conds)}",
            "samples-critique": f"samples-critique-{args.task}-{args.split}",
            "think": f"think-{args.task}-{args.split}"}[args.stage]
    if args.chunk_size:
        kind += f"-c{args.chunk_index}"
    kind += f"-{args.tag}" if args.tag else ""
    exp = EXP[args.stage] if not (args.stage == "ops" and args.split == "dev") else "b5"
    run_id, d = run_dir(exp if args.split == "test" else f"{exp}-dev", args.model, kind)
    register(run_id, "running", block=f"arr-{args.stage}", exp="arr", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    if select is not None:   # the run keeps its own selection: it enters the manifest and is synced with the outputs
        import shutil
        shutil.copyfile(args.select, d / "selection.jsonl")
    t0 = time.time()
    llm = load_llm(args, max_len_for(args))
    tok = llm.get_tokenizer()
    t1 = time.time()
    counts = Counter()
    if args.stage in ("samples", "continue", "think"):
        if args.stage == "samples":
            prompts = build_samples(tok, rows, q)
            params = [SamplingParams(n=args.k, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=args.max_tokens,
                                     seed=seed_for(r["item_id"], args.stream)) for r in rows]
        elif args.stage == "continue":
            prompts = [{"prompt_token_ids": ids} for ids in build_continue(tok, rows, q)]
            params = [SamplingParams(temperature=0.0, max_tokens=args.extra_tokens, seed=0)] * len(rows)
        else:
            prompts = build_think(tok, rows, args.task, q, its)
            params = [SamplingParams(**THINK_SAMPLING, max_tokens=args.max_tokens, seed=think_seed(r["item_id"])) for r in rows]
        outs = llm.generate(prompts, params, use_tqdm=False)
        t2 = time.time()
        name = {"samples": "samples.jsonl", "continue": "continuation.jsonl", "think": "generations.jsonl"}[args.stage]
        with open(d / name, "w", encoding="utf-8") as f:
            for r, o in zip(rows, outs):
                gens = [{"text": c.text, "finish_reason": c.finish_reason, "out_tokens": len(c.token_ids)} for c in o.outputs]
                rec = {"item_id": r["item_id"], "prompt_tokens": len(o.prompt_token_ids)}
                rec.update({"stream": args.stream, "k": args.k, "samples": gens} if args.stage == "samples" else gens[0])
                if args.stage == "continue":
                    rec["a0_tokens"] = len(r["a0_token_ids"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["generated_tokens"] += sum(g["out_tokens"] for g in gens)
                counts["truncated"] += sum(g["finish_reason"] == "length" for g in gens)
        if args.stage == "think":   # monitoring scores only, written after all generations
            with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
                for r, o in zip(rows, outs):
                    c = o.outputs[0]
                    s = score_text(args.task, c.text, c.finish_reason, r["item_id"], q, its)
                    f.write(json.dumps({"item_id": r["item_id"], "cond": "B1-think", "correct0": r["correct"], **s,
                                        "out_tokens": len(c.token_ids), "finish_reason": c.finish_reason}) + "\n")
                    counts["right"] += bool(s["correct"])
    else:
        if args.stage == "ops":
            plan1 = build_ops_stage1(tok, rows, q, conds, args.wording)
            out1 = generate(llm, plan1, SamplingParams)
            plan2 = build_ops_stage2(tok, plan1, out1, rows, q, args.wording)
            out2 = generate(llm, plan2, SamplingParams)
            plans = ((plan1, out1), (plan2, out2))
        else:
            plan1 = build_samples_critique(tok, rows, args.task, q, its, select)
            out1 = generate(llm, plan1, SamplingParams)
            plans = ((plan1, out1),)
        t2 = time.time()
        with open(d / "generations.jsonl", "w", encoding="utf-8") as f:
            for _, outs in plans:
                for k, o in outs.items():
                    f.write(json.dumps({"prompt_key": k, **o}, ensure_ascii=False) + "\n")
        by_key = {k: o for _, outs in plans for k, o in outs.items()}
        with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
            for p, _ in plans:
                for row in p.rows:
                    if row.get("stage") == 1:   # first turns enter through their second turn's turn1_key
                        continue
                    o = by_key[row["prompt_key"]]
                    s = score_text(args.task, o["text"], o["finish_reason"], row["item_id"], q, its)
                    out = {**row, **s, "out_tokens": o["out_tokens"], "prompt_tokens": o["prompt_tokens"],
                           "finish_reason": o["finish_reason"]}
                    if row.get("turn1_key"):
                        t1o = by_key[row["turn1_key"]]
                        out.update(turn1_out_tokens=t1o["out_tokens"], turn1_prompt_tokens=t1o["prompt_tokens"])
                    f.write(json.dumps(out) + "\n")
                    counts["rows"] += 1
                    counts["right"] += bool(s["correct"])
                    counts["generated_tokens"] += o["out_tokens"] + out.get("turn1_out_tokens", 0)
    metrics = {"stage": args.stage, "items": len(rows), "vllm": vllm.__version__, "load_s": t1 - t0, "gen_s": t2 - t1,
               "score_s": time.time() - t2, "counts": dict(counts)}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__},
                   {"initial": R.PROMPT, "wordings": wording_hash, "sample_aware": SAMPLE_AWARE, "instruction": R.INSTRUCTION})
    register(run_id, "done", block=f"arr-{args.stage}", exp="arr", model=args.model)
    print(json.dumps({"run_id": run_id, "dir": str(d), **metrics}, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(EXP))
    ap.add_argument("--task", default="math", choices=["math", "mc"])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--initial", action="append", required=True, help="initial.jsonl (repeat for the expansion)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stream", default="h2a-consistency")
    ap.add_argument("--max-tokens", type=int, default=A0_TOKENS, help="samples: per sample; think: 16384")
    ap.add_argument("--extra-tokens", type=int, default=4096)
    ap.add_argument("--conds", default="F0,F2,F2-scratch")
    ap.add_argument("--wording", default="main", choices=list(WORDINGS))
    ap.add_argument("--select", default=None, help="samples-critique: selection jsonl from analysis/arr/b4_select.py")
    ap.add_argument("--chunk-size", type=int, default=0)
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--budget", type=int, default=0, help="think-budget: thinking tokens (Amendment 7)")
    ap.add_argument("--prefix-check", type=int, default=0, help="think-budget: items for the 64-token prefix check")
    args = ap.parse_args()
    if args.stage == "think-budget" and args.budget <= 0:
        raise SystemExit("think-budget needs --budget")
    if args.split == "test" and not args.allow_test:
        raise SystemExit("--split test needs --allow-test (post-freeze exploratory run)")
    if args.stage == "samples-critique" and not args.select:
        raise SystemExit("samples-critique needs --select")
    if args.stage == "think" and args.max_tokens == A0_TOKENS:
        args.max_tokens = 16384
    if args.task == "mc" and args.stage not in ("samples-critique", "think", "think-budget"):
        raise SystemExit("MMLU-Pro is used only by samples-critique, think and think-budget in this round")
    if args.stage == "ops" and set(args.conds.split(",")) - {"F0", "F2", "F2-scratch"}:
        raise SystemExit("ops: conditions are F0, F2, F2-scratch")
    stage_main(args)


if __name__ == "__main__":
    main()
