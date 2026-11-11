"""P1: D-nat initial answers a0 (design §5.4): each model's own greedy, non-thinking answer on every D-nat item,
generated in vLLM with VLLM_BATCH_INVARIANT=1, and scored with the design §5.6 scorer.

Pre-freeze rule (design §5.3): test rows are generated and scored by the machine; this script prints only
counts per split x dataset (n, right, wrong, status counts).
Usage: VLLM_BATCH_INVARIANT=1 python -m src.gen.initial --model 4B
"""
import argparse
import json
import time
from collections import Counter, defaultdict

from ..runlib import MODELS, REPO, register, require_batch_invariant, run_dir, sha256_text, write_manifest
from ..score.scorer import SCORER_VERSION, score

PROMPT = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--items", default="data/processed/dnat.jsonl", help="item table (P3 expansion: data/processed/ext/dnat_ext.jsonl)")
    ap.add_argument("--split-file", default="data/processed/split.json", help="file with the seeded processing orders")
    ap.add_argument("--kind", default="initial", help="run-ID kind (P3 expansion: initial-ext)")
    args = ap.parse_args()
    require_batch_invariant()
    cfg = {**vars(args), "prompt": PROMPT, "decoding": "greedy", "thinking": False, "scorer_version": SCORER_VERSION}

    import vllm
    from vllm import LLM, SamplingParams

    items = [json.loads(line) for line in open(REPO / args.items, encoding="utf-8")]
    order = json.loads((REPO / args.split_file).read_text())["order"]
    rank = {cid: k for s in ("dev", "test") for k, cid in enumerate(order[s])}
    items.sort(key=lambda r: (r["split"] != "dev", rank[r["canonical_id"]], r["item_id"]))

    run_id, d = run_dir("p1", args.model, args.kind)
    register(run_id, "running", block=f"p1-{args.kind}", exp="p1", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t = time.time()
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_tokens + 2048)
    tok = llm.get_tokenizer()
    load_s = time.time() - t
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=r["question"])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for r in items]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    t = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    gen_s = time.time() - t

    counts = defaultdict(Counter)
    with open(d / "initial.jsonl", "w", encoding="utf-8") as f:
        for r, o in zip(items, outs):
            out = o.outputs[0]
            s = score(out.text, r["gold"], finish_reason=out.finish_reason)
            row = {"item_id": r["item_id"], "canonical_id": r["canonical_id"], "dataset": r["dataset"],
                   "split": r["split"], "a0": out.text, "a0_token_ids": list(out.token_ids),
                   "prompt_tokens": len(o.prompt_token_ids), "finish_reason": out.finish_reason,
                   "status": s["status"], "pred": s["pred"], "correct": s["correct"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            key = f"{r['split']}/{r['dataset']}"
            counts[key]["n"] += 1
            counts[key]["right" if s["correct"] else "wrong"] += 1
            counts[key][f"status_{s['status']}"] += 1
    total_out = sum(len(o.outputs[0].token_ids) for o in outs)
    metrics = {"model": args.model, "vllm": vllm.__version__, "load_s": load_s, "gen_s": gen_s, "n": len(items),
               "output_tokens": total_out, "counts": {k: dict(v) for k, v in sorted(counts.items())}}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__}, {"initial": PROMPT})
    register(run_id, "done", block=f"p1-{args.kind}", exp="p1", model=args.model)
    print(json.dumps({"run_id": run_id, **{k: metrics[k] for k in ("n", "gen_s", "output_tokens", "counts")}}, indent=1))


if __name__ == "__main__":
    main()
