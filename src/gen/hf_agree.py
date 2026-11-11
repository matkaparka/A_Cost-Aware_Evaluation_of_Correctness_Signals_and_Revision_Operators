"""HF vs vLLM greedy agreement on dev (design §6 Exp 5 and §8: >= 98 % identical greedy outputs, otherwise the mismatch
rate is reported and generation-based comparisons stay within vLLM).

The vLLM side is the P1 initial answer a0 (greedy, VLLM_BATCH_INVARIANT=1, base weights). The HF side is the model as HF
uses it in this study (bf16, full-sequence forward, as in TTT training and teacher-forced scoring): a0 is teacher-forced
after its own prompt, and at every position the HF argmax is compared with a0's next token. An item whose argmaxes all
match is one whose HF greedy output equals a0 under full-sequence evaluation; the first mismatch is where greedy HF would
leave a0. Recorded per item: first mismatch position, a0 length, the logit gap between HF's top token and a0's token
there, and the share of matching positions. Dev split, every item of the given table. Batch size 1.
Usage: python -m src.gen.hf_agree --model 4B --initial <p1 initial.jsonl> [--items data/processed/dnat.jsonl] [--n 0]
"""
import argparse
import json
import time

import torch

from ..runlib import MODELS, REPO, register, run_dir, sha256_text, write_manifest
from .initial import PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--items", default="data/processed/dnat.jsonl")
    ap.add_argument("--n", type=int, default=0, help="first N dev rows (0 = all; smoke tests)")
    args = ap.parse_args()
    cfg = {**vars(args), "comparison": "teacher-forced HF argmax vs vLLM greedy a0", "dtype": "bfloat16"}

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    q = {json.loads(l)["item_id"]: json.loads(l)["question"] for l in open(REPO / args.items, encoding="utf-8")}
    rows = [json.loads(l) for l in open(args.initial, encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "dev"]
    if args.n:
        rows = rows[:args.n]

    run_id, d = run_dir("p2", args.model, "hfagree-dev")
    register(run_id, "running", block="hfagree-dev", exp="p3-checks", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    out, t = [], time.time()
    with torch.no_grad():
        for r in rows:
            prefix = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q[r["item_id"]])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
            p_ids = tok(prefix, add_special_tokens=False).input_ids
            a0 = r["a0_token_ids"]
            ids = torch.tensor([p_ids + a0], device="cuda")
            logits = model(input_ids=ids).logits[0, len(p_ids) - 1: len(p_ids) - 1 + len(a0)].float()
            target = torch.tensor(a0, device="cuda")
            top = logits.argmax(-1)
            match = top == target
            miss = (~match).nonzero()
            first = int(miss[0]) if len(miss) else None
            gap = None
            if first is not None:
                gap = float(logits[first, top[first]] - logits[first, target[first]])
            out.append({"item_id": r["item_id"], "dataset": r["dataset"], "a0_len": len(a0), "identical": first is None,
                        "first_mismatch": first, "logit_gap_at_first": gap, "match_share": float(match.float().mean())})
    wall = time.time() - t
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    n_id = sum(o["identical"] for o in out)
    metrics = {"n": len(out), "identical": n_id, "rate": n_id / len(out) if out else None, "wall_s": wall,
               "transformers": transformers.__version__}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__},
                   {"initial": PROMPT})
    register(run_id, "done", block="hfagree-dev", exp="p3-checks", model=args.model)
    print(json.dumps({"run_id": run_id, **metrics}))


if __name__ == "__main__":
    main()
