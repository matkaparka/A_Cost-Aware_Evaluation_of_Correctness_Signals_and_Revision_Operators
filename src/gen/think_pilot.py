"""Thinking-mode budget pilot on dev (design §4 decoding; rule 3: budgets are chosen on dev).

Thinking-mode initial answers on the dev D-nat items with the model card's sampling settings (T 0.6, top-p 0.95, top-k 20,
min-p 0; the pinned generation_config.json and model card, checked 2026-11-05), one seed per item, vLLM with
VLLM_BATCH_INVARIANT=1, budget --max-tokens. A per-request seed fixes the sampled sequence, so its first B tokens do not
depend on the budget: the truncation rate at any budget B <= --max-tokens is the share of outputs longer than B (a
length equal to the cap counts as longer). The budget rule is fixed in DECISIONS 2026-11-05 before this runs.
Output: rows (item_id, dataset, out_tokens, finish_reason, think_closed, status, correct) and truncation rates.
Usage: VLLM_BATCH_INVARIANT=1 python -m src.gen.think_pilot --model 4B [--max-tokens 16384] [--dry-run]
"""
import argparse
import json
import time
from collections import Counter, defaultdict

from ..runlib import MODELS, REPO, register, require_batch_invariant, run_dir, sha256_text, write_manifest
from ..score.scorer import SCORER_VERSION, score
from .initial import PROMPT

BUDGETS = (4096, 8192, 16384)
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--max-tokens", type=int, default=max(BUDGETS))
    ap.add_argument("--items", default="data/processed/dnat.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = {**vars(args), "prompt": PROMPT, "thinking": True, "sampling": SAMPLING, "budgets": BUDGETS,
           "scorer_version": SCORER_VERSION}
    order = json.loads((REPO / "data" / "processed" / "split.json").read_text())["order"]["dev"]
    rank = {c: k for k, c in enumerate(order)}
    items = [json.loads(l) for l in open(REPO / args.items, encoding="utf-8")]
    items = sorted((r for r in items if r["split"] == "dev"), key=lambda r: (rank[r["canonical_id"]], r["item_id"]))
    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODELS[args.model])
        p = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=items[0]["question"])}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        print(json.dumps({"items": len(items), "datasets": Counter(r["dataset"] for r in items), "first_prompt": p}, indent=1))
        return

    require_batch_invariant()
    import vllm
    from vllm import LLM, SamplingParams
    run_id, d = run_dir("p2", args.model, "think-pilot-dev")
    register(run_id, "running", block="think-pilot-dev", exp="p3-checks", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_tokens + 2048)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=r["question"])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for r in items]
    params = [SamplingParams(**SAMPLING, max_tokens=args.max_tokens, seed=args.seed) for _ in items]
    t = time.time()
    outs = llm.generate(prompts, params, use_tqdm=False)
    gen_s = time.time() - t
    by = defaultdict(list)
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for r, o in zip(items, outs):
            out = o.outputs[0]
            s = score(out.text, r["gold"], finish_reason=out.finish_reason)
            row = {"item_id": r["item_id"], "dataset": r["dataset"], "out_tokens": len(out.token_ids),
                   "finish_reason": out.finish_reason, "think_closed": "</think>" in out.text,
                   "status": s["status"], "correct": s["correct"]}
            f.write(json.dumps(row) + "\n")
            by[r["dataset"]].append(row)
            by["all"].append(row)
    trunc = {k: {str(b): sum(x["out_tokens"] >= b for x in v) / len(v) for b in BUDGETS if b <= args.max_tokens}
             for k, v in by.items()}
    metrics = {"n": len(items), "gen_s": gen_s, "vllm": vllm.__version__, "truncation_share": trunc,
               "mean_out_tokens": {k: sum(x["out_tokens"] for x in v) / len(v) for k, v in by.items()},
               "accuracy_at_cap": {k: sum(x["correct"] for x in v) / len(v) for k, v in by.items()}}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__}, {"initial": PROMPT})
    register(run_id, "done", block="think-pilot-dev", exp="p3-checks", model=args.model)
    print(json.dumps({"run_id": run_id, **metrics}, indent=1))


if __name__ == "__main__":
    main()
