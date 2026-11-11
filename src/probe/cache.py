"""Exp 2 activation cache (design §6, Exp 2): residual stream at all layers (embeddings + every block output), at
the last token of a0 (pre-feedback; primary). The context is the one a0 was generated in: the initial prompt with
its non-thinking generation prefix, followed by a0.

Pre-feedback quantity: may be computed on test before the freeze (design §5.3), but the probe is trained on dev
and scored on test only after the freeze. This script caches one split per run.
Output: <run>/acts.npz with ids (item_id), correct0, X[n, layers+1, hidden] (float16).
Usage: python -m src.probe.cache --model 4B --split dev --initial <p1 initial.jsonl>
"""
import argparse
import json
import time

import numpy as np
import torch

from ..gen.render import PROMPT
from ..runlib import MODELS, REPO, needs_bos, register, run_dir, sha256_text, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--initial", required=True)
    ap.add_argument("--items", default="data/processed/dnat.jsonl", help="item table (P3 expansion: data/processed/ext/dnat_ext.jsonl)")
    ap.add_argument("--kind-suffix", default="", help="appended to the run-ID kind (P3 expansion: -ext)")
    args = ap.parse_args()
    cfg = {**vars(args), "position": "last token of a0", "context": "initial prompt + generation prefix + a0"}

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    q = {json.loads(l)["item_id"]: json.loads(l)["question"] for l in open(REPO / args.items, encoding="utf-8")}
    rows = [json.loads(l) for l in open(args.initial, encoding="utf-8")]
    rows = [r for r in rows if r["split"] == args.split]

    run_id, d = run_dir("p2" if args.split == "dev" else "main", args.model, f"acts-{args.split}{args.kind_suffix}")
    register(run_id, "running", block=f"acts-{args.split}{args.kind_suffix}", exp="exp2", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    X, ids, y = [], [], []
    t = time.time()
    with torch.no_grad():
        for r in rows:
            prefix = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q[r["item_id"]])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
            # Re-use a0's own generated token IDs so the cached state is exactly the generation state.
            p_ids = tok(prefix, add_special_tokens=needs_bos(tok, prefix)).input_ids
            ids_all = torch.tensor([p_ids + r["a0_token_ids"]], device="cuda")
            hs = model(input_ids=ids_all, output_hidden_states=True).hidden_states
            last = len(p_ids) + len(r["a0_token_ids"]) - 1
            if r["a0_token_ids"] and r["a0_token_ids"][-1] in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
                last -= 1   # the state at the last content token, before the end-of-turn token
            X.append(torch.stack([h[0, last] for h in hs]).to(torch.float16).cpu().numpy())
            ids.append(r["item_id"])
            y.append(bool(r["correct"]))
    wall = time.time() - t
    np.savez_compressed(d / "acts.npz", X=np.stack(X), ids=np.array(ids), correct0=np.array(y))
    (d / "metrics.json").write_text(json.dumps({"n": len(ids), "wall_s": wall, "shape": list(np.stack(X).shape)}, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__},
                   {"initial": PROMPT})
    register(run_id, "done", block=f"acts-{args.split}{args.kind_suffix}", exp="exp2", model=args.model)
    print(json.dumps({"run_id": run_id, "n": len(ids), "wall_s": round(wall)}))


if __name__ == "__main__":
    main()
