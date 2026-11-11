"""Exp 5 dev pilot, vLLM stage: every accuracy cell is generated in vLLM with VLLM_BATCH_INVARIANT=1 (design Exp 5,
Backends). For each tuning item: B1 (no update, q only), the context arm per verdict source (no update), and for each
adapter the parameter arm (q only) and parameter + context. The zero adapter must reproduce B1 token for token.

Usage: VLLM_BATCH_INVARIANT=1 python -m src.ttt.serve --model 4B --pilot <ttt-pilot run dir> --initial <p1 initial.jsonl>
"""
import argparse
import json
import time
from pathlib import Path

from ..gen import render as R
from ..runlib import MODELS, REPO, register, require_batch_invariant, run_dir, sha256_text, write_manifest
from ..score.scorer import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    require_batch_invariant()
    cfg = {**vars(args), "context_template_sha256": sha256_text(R.CONTEXT_MEMORY)}

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    pilot = Path(args.pilot)
    adapters = [json.loads(l) for l in open(pilot / "adapters.jsonl", encoding="utf-8")]
    dnat = {json.loads(l)["item_id"]: json.loads(l) for l in open(REPO / "data" / "processed" / "dnat.jsonl", encoding="utf-8")}
    init = {json.loads(l)["item_id"]: json.loads(l) for l in open(args.initial, encoding="utf-8")}
    items = sorted({a["item_id"] for a in adapters if a["item_id"]})

    run_id, d = run_dir("p2", args.model, "ttt-serve")
    register(run_id, "running", block="ttt-serve", exp="exp5", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=12288,
              enable_lora=True, max_lora_rank=8, max_loras=16)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)

    reqs = []   # (row, prompt_text, lora or None)
    lora_ids = {}

    def lora(a):
        if a["name"] not in lora_ids:
            lora_ids[a["name"]] = LoRARequest(a["name"], len(lora_ids) + 1, a["path"])
        return lora_ids[a["name"]]

    for i in items:
        q, a0, c0 = dnat[i]["question"], init[i]["a0"], init[i]["correct"]
        fresh = R.fresh_prompt(tok, q).text
        base = {"item_id": i, "correct0": c0}
        reqs.append(({**base, "arm": "B1", "grounding": None}, fresh, None))
        for v in (True, False):
            reqs.append(({**base, "arm": "context", "verdict": v}, R.context_prompt(tok, q, a0, v).text, None))
        reqs.append(({**base, "arm": "zero-step", "grounding": "zero"}, fresh, lora(next(a for a in adapters if a["name"] == "zero"))))
    for a in adapters:
        if not a["item_id"]:
            continue
        i = a["item_id"]
        q, a0, c0 = dnat[i]["question"], init[i]["a0"], init[i]["correct"]
        base = {"item_id": i, "correct0": c0, "grounding": a["grounding"], "lr": a["lr"], "tau": a["tau"],
                "verdict": a["verdict"], "applied": a["applied"]}
        reqs.append(({**base, "arm": "parameter"}, R.fresh_prompt(tok, q).text, lora(a)))
        if a["verdict"] is not None:
            reqs.append(({**base, "arm": "parameter+context"}, R.context_prompt(tok, q, a0, a["verdict"]).text, lora(a)))

    t = time.time()
    outs = llm.generate([p for _, p, _ in reqs], sp, lora_request=[lr for _, _, lr in reqs], use_tqdm=False)
    wall = time.time() - t
    b1_tokens = {}
    rows, texts = [], []
    for (row, _, _), o in zip(reqs, outs):
        out = o.outputs[0]
        s = score(out.text, dnat[row["item_id"]]["gold"], finish_reason=out.finish_reason)
        row.update({"correct": s["correct"], "status": s["status"], "pred": s["pred"], "out_tokens": len(out.token_ids),
                    "token_ids_sha256": sha256_text(json.dumps(list(out.token_ids)))})
        if row["arm"] == "B1":
            b1_tokens[row["item_id"]] = row["token_ids_sha256"]
        rows.append(row)
        texts.append({"row": len(rows) - 1, "item_id": row["item_id"], "arm": row["arm"], "text": out.text})
    zero = [r for r in rows if r["arm"] == "zero-step"]
    metrics = {"model": args.model, "n_items": len(items), "n_requests": len(reqs), "n_adapters": len(lora_ids),
               "wall_s": wall, "zero_step_equals_B1": sum(r["token_ids_sha256"] == b1_tokens[r["item_id"]] for r in zero) / len(zero),
               "vllm": vllm.__version__}
    # Texts are kept so the scored labels can be hand-checked (P2 found the earlier rows held only token hashes).
    for name, rr in (("rows.jsonl", rows), ("generations.jsonl", texts)):
        with open(d / name, "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r) + "\n")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__},
                   {"context": R.CONTEXT_MEMORY})
    register(run_id, "done", block="ttt-serve", exp="exp5", model=args.model)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
