"""H2a comparators that P1 did not store (DEVIATIONS D13; docs/planning/next_experiments/plan.md, Group 1).
Exploratory, post-freeze: they enter no confirmatory test.

  logprob      teacher-forced per-token log-probs of a0 under the model, in the context a0 was generated in (the
               initial prompt with its non-thinking generation prefix, as src/probe/cache.py), from a0's own token
               IDs. Writes the sum, the count and the mean over a0's content tokens (the end-of-turn token excluded),
               and the minimum token log-prob. HF backend, bf16, logits upcast to fp32.
  consistency  K sampled initial answers per item (vLLM, VLLM_BATCH_INVARIANT=1; T 0.7, top-p 0.8, top-k 20: the
               design's sampled-robustness setting), each scored with the frozen math scorer. Consistency = the share
               of the K samples whose extracted answer is equivalent to a0's extracted answer (src/score/scorer.py
               `equivalent`); also the share that is correct and the majority answer's agreement with a0. Per-item
               sampling seeds come from the named stream "h2a-consistency" (sha256 of stream name and item ID).

Both read the P1 initial-answer tables (a0, a0_token_ids, correct, split) and the D-nat item tables.
Usage: python -m src.gen.h2a_extra logprob --model 4B --initial <initial.jsonl> [--initial ...] --split test --allow-test
       VLLM_BATCH_INVARIANT=1 python -m src.gen.h2a_extra consistency --model 4B --initial ... --split test --allow-test
"""
import argparse
import hashlib
import json
import time
from collections import Counter

from ..gen.initial import PROMPT
from ..runlib import MODELS, REPO, register, require_batch_invariant, run_dir, sha256_text, write_manifest
from ..score.scorer import SCORER_VERSION, equivalent, score

K, TEMPERATURE, TOP_P, TOP_K = 5, 0.7, 0.8, 20
ITEM_TABLES = ("data/processed/dnat.jsonl", "data/processed/ext/dnat_ext.jsonl")


def jl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def seed_for(item_id, stream="h2a-consistency"):
    return int(hashlib.sha256(f"{stream}|{item_id}".encode()).hexdigest(), 16) % (2 ** 31)


def load(args):
    q = {}
    for t in ITEM_TABLES:
        if (REPO / t).exists():
            q.update({r["item_id"]: r for r in jl(REPO / t)})
    rows = [r for f in args.initial for r in jl(f) if r["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    return q, rows


def logprob(args, q, rows):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    end = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    cfg = {**vars(args), "context": "initial prompt + generation prefix + a0 token IDs", "excluded": "end-of-turn token"}
    run_id, d = run_dir("fu" if args.split == "test" else "fu-dev", args.model, f"h2a-logprob-{args.split}{args.tag}")
    register(run_id, "running", block="h2a-logprob", exp="h2a-extra", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t = time.time()
    with open(d / "logprob.jsonl", "w", encoding="utf-8") as f, torch.no_grad():
        for r in rows:
            prefix = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q[r["item_id"]]["question"])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
            p_ids = tok(prefix, add_special_tokens=False).input_ids
            a_ids = list(r["a0_token_ids"])
            if a_ids and a_ids[-1] in end:
                a_ids = a_ids[:-1]
            if not a_ids:
                f.write(json.dumps({"item_id": r["item_id"], "n_tokens": 0, "sum": None, "mean": None, "min": None}) + "\n")
                continue
            ids = torch.tensor([p_ids + a_ids], device="cuda")
            logits = model(input_ids=ids, logits_to_keep=len(a_ids) + 1).logits[0, :-1].float()   # predicts a_ids
            lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(a_ids, device="cuda")[:, None])[:, 0]
            f.write(json.dumps({"item_id": r["item_id"], "n_tokens": len(a_ids), "sum": float(lp.sum()),
                                "mean": float(lp.mean()), "min": float(lp.min())}) + "\n")
    wall = time.time() - t
    (d / "metrics.json").write_text(json.dumps({"n": len(rows), "wall_s": wall,
                                                "peak_vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30}, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__},
                   {"initial": PROMPT})
    register(run_id, "done", block="h2a-logprob", exp="h2a-extra", model=args.model)
    print(json.dumps({"run_id": run_id, "n": len(rows), "wall_s": round(wall)}))


def consistency(args, q, rows):
    require_batch_invariant()
    import vllm
    from vllm import LLM, SamplingParams
    cfg = {**vars(args), "k": K, "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "seed_stream": "h2a-consistency",
           "scorer_version": SCORER_VERSION}
    run_id, d = run_dir("fu" if args.split == "test" else "fu-dev", args.model, f"h2a-consistency-{args.split}{args.tag}")
    register(run_id, "running", block="h2a-consistency", exp="h2a-extra", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=8192)
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q[r["item_id"]]["question"])}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    params = [SamplingParams(n=K, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=args.max_tokens,
                             seed=seed_for(r["item_id"])) for r in rows]
    t1 = time.time()
    outs = llm.generate(texts, params, use_tqdm=False)
    t2 = time.time()
    with open(d / "samples.jsonl", "w", encoding="utf-8") as f:     # generations first, scoring after
        for r, o in zip(rows, outs):
            f.write(json.dumps({"item_id": r["item_id"], "samples": [{"text": c.text, "finish_reason": c.finish_reason,
                                                                      "out_tokens": len(c.token_ids)} for c in o.outputs]},
                               ensure_ascii=False) + "\n")
    with open(d / "consistency.jsonl", "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            gold = q[r["item_id"]]["gold"]
            a0 = score(r["a0"], gold, finish_reason=r.get("finish_reason"))
            sc = [score(c.text, gold, finish_reason=c.finish_reason) for c in o.outputs]
            preds = [s["pred"] for s in sc]
            agree = [a0["pred"] is not None and p is not None and equivalent(p, a0["pred"]) for p in preds]
            votes = Counter(p for p in preds if p is not None)
            top = votes.most_common(1)[0][0] if votes else None
            f.write(json.dumps({"item_id": r["item_id"], "k": K, "a0_pred": a0["pred"], "a0_correct": r["correct"],
                                "consistency": sum(agree) / K, "sample_acc": sum(s["correct"] for s in sc) / K,
                                "unparsed_or_truncated": sum(s["status"] != "ok" for s in sc),
                                "majority_agrees_with_a0": bool(top is not None and a0["pred"] is not None and equivalent(top, a0["pred"]))})
                    + "\n")
    metrics = {"n": len(rows), "k": K, "load_s": t1 - t0, "gen_s": t2 - t1, "score_s": time.time() - t2,
               "vllm": vllm.__version__,
               "sample_tokens": sum(len(c.token_ids) for o in outs for c in o.outputs)}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__}, {"initial": PROMPT})
    register(run_id, "done", block="h2a-consistency", exp="h2a-extra", model=args.model)
    print(json.dumps({"run_id": run_id, **{k: metrics[k] for k in ("n", "gen_s", "sample_tokens")}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["logprob", "consistency"])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--initial", action="append", required=True, help="P1 initial.jsonl (repeat for the expansion)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096, help="consistency: as P1 a0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("--split test needs --allow-test (post-freeze exploratory run)")
    args.tag = f"-{args.tag}" if args.tag else ""
    q, rows = load(args)
    (logprob if args.stage == "logprob" else consistency)(args, q, rows)


if __name__ == "__main__":
    main()
