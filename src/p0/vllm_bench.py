"""P0 vLLM benchmark and vertical slice on train fixtures.

Round 1 (initial answer) -> score -> round 2 (revision after feedback) -> score, plus an optional
thinking-mode subset and optional multi-LoRA serving (with the zero-adapter = base check).
Usage: python -m src.p0.vllm_bench --model 4B [--n 256] [--think-n 64] [--lora-dir DIR]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from .common import FEEDBACK, MODELS, PROMPT, fixtures, last_boxed, render, run_dir, score, strip_think, write_manifest


def gen_block(llm, prompts, sp, lora=None):
    t = time.time()
    outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
    dt = time.time() - t
    ptok = sum(len(o.prompt_token_ids) for o in outs)
    otok = sum(len(o.outputs[0].token_ids) for o in outs)
    return outs, {"n": len(prompts), "wall_s": dt, "prompt_tokens": ptok, "output_tokens": otok,
                  "output_tok_per_s": otok / dt, "total_tok_per_s": (ptok + otok) / dt,
                  "mean_output_tokens": otok / len(prompts),
                  "max_output_tokens": max(len(o.outputs[0].token_ids) for o in outs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--think-n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--think-max-tokens", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--lora-dir", default=None)
    args = ap.parse_args()
    cfg = vars(args)

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    kind = "vllm-lora" if args.lora_dir else "vllm"
    run_id, d = run_dir("p0", args.model, kind)
    m = {"model": args.model, "vllm": vllm.__version__}

    t = time.time()
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=args.gpu_util,
              max_model_len=args.think_max_tokens + 2048, enable_lora=bool(args.lora_dir),
              max_lora_rank=8, max_loras=8)
    m["load_s"] = time.time() - t
    tok = llm.get_tokenizer()
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)

    fx = fixtures(args.n)
    p1 = [render(tok, [{"role": "user", "content": PROMPT.format(q=x["q"])}]) for x in fx]
    o1, m["round1"] = gen_block(llm, p1, greedy)
    items = []
    for x, o in zip(fx, o1):
        a0 = o.outputs[0].text
        items.append({**x, "a0": a0, "a0_tokens": len(o.outputs[0].token_ids),
                      "pred0": last_boxed(a0), "correct0": score(last_boxed(a0), x["gold"])})

    if not args.lora_dir:
        p2 = [render(tok, [{"role": "user", "content": PROMPT.format(q=it["q"])},
                           {"role": "assistant", "content": it["a0"]},
                           {"role": "user", "content": FEEDBACK}]) for it in items]
        o2, m["round2"] = gen_block(llm, p2, greedy)
        for it, o in zip(items, o2):
            it["a1"] = o.outputs[0].text
            it["pred1"] = last_boxed(it["a1"])
            it["correct1"] = score(it["pred1"], it["gold"])
        m["acc0"] = sum(it["correct0"] for it in items) / len(items)
        m["acc1"] = sum(it["correct1"] for it in items) / len(items)
        m["unparsed0"] = sum(it["pred0"] is None for it in items) / len(items)
        m["w2r"] = sum((not it["correct0"]) and it["correct1"] for it in items)
        m["r2w"] = sum(it["correct0"] and (not it["correct1"]) for it in items)
        m["truncated0"] = sum(it["a0_tokens"] >= args.max_tokens for it in items)

        if args.think_n:
            think = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.think_max_tokens, seed=0)
            pt = [render(tok, [{"role": "user", "content": PROMPT.format(q=x["q"])}], thinking=True)
                  for x in fx[:args.think_n]]
            ot, m["thinking"] = gen_block(llm, pt, think)
            m["thinking"]["acc"] = sum(score(last_boxed(strip_think(o.outputs[0].text)), x["gold"])
                                       for x, o in zip(fx, ot)) / args.think_n
            m["thinking"]["truncated"] = sum(len(o.outputs[0].token_ids) >= args.think_max_tokens for o in ot)
    else:
        # Multi-LoRA serving: round-robin over the trained adapters; then the zero adapter must equal base.
        dirs = sorted(p for p in Path(args.lora_dir).iterdir() if p.is_dir() and p.name.startswith("ttt"))
        reqs = [LoRARequest(p.name, i + 1, str(p)) for i, p in enumerate(dirs)]
        t = time.time()
        _, m["lora_first_load"] = gen_block(llm, p1[:1], greedy, lora=reqs[0])
        lora_list = [reqs[i % len(reqs)] for i in range(len(p1))]
        _, m["lora_roundrobin"] = gen_block(llm, p1, greedy, lora=lora_list)
        m["n_adapters"] = len(reqs)
        zero = LoRARequest("zero", 99, str(Path(args.lora_dir) / "zero"))
        oz, m["lora_zero"] = gen_block(llm, p1, greedy, lora=zero)
        same = sum(list(a.outputs[0].token_ids) == list(b.outputs[0].token_ids) for a, b in zip(o1, oz))
        m["zero_adapter_identical_greedy"] = same / len(p1)

    # vLLM's engine runs in a child process, so read device memory from nvidia-smi.
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    m["gpu_mem_used_gb_at_end"] = int(smi[0]) / 1024 if smi else None
    with open(d / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    (d / "metrics.json").write_text(json.dumps(m, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__})
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
