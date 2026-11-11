"""P0 greedy-determinism check in vLLM (design §8 / Exp 5 feasibility "zero-step = B1 exactly").
Compares greedy token sequences for the same prompts across: a repeat call, a different batch
composition, batch size 1, and a zero (B = 0) LoRA adapter. Run once with VLLM_BATCH_INVARIANT=0
and once with =1. Usage: python -m src.p0.determinism --model 4B --zero-adapter DIR
"""
import argparse
import json
import os
import time

from .common import MODELS, PROMPT, fixtures, render, run_dir, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--zero-adapter", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    bi = os.environ.get("VLLM_BATCH_INVARIANT", "0")
    cfg = {**vars(args), "VLLM_BATCH_INVARIANT": bi}

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    run_id, d = run_dir("p0", args.model, f"determinism-bi{bi}")
    t = time.time()
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85,
              max_model_len=4096, enable_lora=True, max_lora_rank=8, max_loras=4)
    m = {"model": args.model, "VLLM_BATCH_INVARIANT": bi, "load_s": time.time() - t}
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    fx = fixtures(args.n + 64, seed=0)
    prompts = [render(tok, [{"role": "user", "content": PROMPT.format(q=x["q"])}]) for x in fx]
    main_p, extra = prompts[: args.n], prompts[args.n:]

    def toks(outs):
        return [tuple(o.outputs[0].token_ids) for o in outs]

    def same(a, b):
        return sum(x == y for x, y in zip(a, b)) / len(a)

    t = time.time()
    a = toks(llm.generate(main_p, sp, use_tqdm=False))
    m["base_wall_s"] = time.time() - t
    t = time.time()
    b = toks(llm.generate(main_p, sp, use_tqdm=False))
    m["repeat_identical"] = same(a, b)
    mixed = list(reversed(main_p)) + extra
    c = toks(llm.generate(mixed, sp, use_tqdm=False))[: args.n][::-1]
    m["other_batch_identical"] = same(a, c)
    single = [toks(llm.generate([p], sp, use_tqdm=False))[0] for p in main_p[:16]]
    m["batch1_identical_first16"] = same(a[:16], single)
    t = time.time()
    z = toks(llm.generate(main_p, sp, use_tqdm=False, lora_request=LoRARequest("zero", 1, args.zero_adapter)))
    m["zero_adapter_wall_s"] = time.time() - t
    m["zero_adapter_identical"] = same(a, z)
    m["first_divergence_token_zero_adapter"] = sorted(
        next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        for x, y in zip(a, z) if x != y)[:10]

    (d / "metrics.json").write_text(json.dumps(m, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__})
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
