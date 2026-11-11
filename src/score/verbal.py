"""Verbal verdict P(yes) on initial answers (design §5.7), teacher-forced in HF under the locked tokenizer.

Query: "Is your final answer correct? Answer yes or no." in a new user turn after (q, a0), thinking disabled.
Candidates Y = {yes, Yes, YES} and N = {no, No, NO}, each with and without a leading space; the probability of a
candidate is its full-sequence probability. P(yes)_norm = P(Y) / (P(Y) + P(N)); the total mass is recorded.
The first-token shortcut is used only if every candidate is one token, the IDs are distinct, and Y and N do not
overlap (checked here and written to metrics.json); otherwise multi-token candidates are scored in full.

A pre-feedback quantity, so it may be computed on test before the freeze (design §5.3); prints counts only.
Usage: python -m src.score.verbal --model 4B --initial /home/results/p1/4B/<run>/initial.jsonl
"""
import argparse
import json
import math
import time

import torch

from ..gen.initial import PROMPT
from ..runlib import MODELS, REPO, needs_bos, register, run_dir, sha256_text, write_manifest

QUERY = "Is your final answer correct? Answer yes or no."
YES = ["yes", "Yes", "YES", " yes", " Yes", " YES"]
NO = ["no", "No", "NO", " no", " No", " NO"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--items", default="data/processed/dnat.jsonl", help="item table (P3 expansion: data/processed/ext/dnat_ext.jsonl)")
    ap.add_argument("--kind", default="verbal", help="run-ID kind (P3 expansion: verbal-ext)")
    args = ap.parse_args()
    cfg = {**vars(args), "query": QUERY, "yes": YES, "no": NO}

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    items = {json.loads(line)["item_id"]: json.loads(line) for line in open(REPO / args.items, encoding="utf-8")}
    rows = [json.loads(line) for line in open(args.initial, encoding="utf-8")]

    cand = {c: tok(c, add_special_tokens=False).input_ids for c in YES + NO}
    single = all(len(v) == 1 for v in cand.values())
    first_ids = [v[0] for v in cand.values()]
    shortcut = single and len(set(first_ids)) == len(first_ids)

    prefixes = []
    for r in rows:
        msgs = [{"role": "user", "content": PROMPT.format(q=items[r["item_id"]]["question"])},
                {"role": "assistant", "content": r["a0"]}, {"role": "user", "content": QUERY}]
        prefixes.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    run_id, d = run_dir("p1", args.model, args.kind)
    register(run_id, "running", block=f"p1-{args.kind}", exp="p1", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)), input=args.initial)
    tok.padding_side = "left"
    t = time.time()
    out_rows = []
    with torch.no_grad():
        for b in range(0, len(prefixes), args.batch):
            batch = prefixes[b:b + args.batch]
            if shortcut:
                enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=needs_bos(tok, batch[0])).to("cuda")
                logp = torch.log_softmax(model(**enc).logits[:, -1].float(), -1)
                lp = {c: logp[:, ids[0]].tolist() for c, ids in cand.items()}
            else:
                lp = {}
                for c, ids in cand.items():
                    enc = tok([p + c for p in batch], return_tensors="pt", padding=True,
                              add_special_tokens=needs_bos(tok, batch[0])).to("cuda")
                    lsm = torch.log_softmax(model(**enc).logits[:, -len(ids) - 1:-1].float(), -1)
                    tgt = enc.input_ids[:, -len(ids):]
                    lp[c] = lsm.gather(-1, tgt[..., None])[..., 0].sum(-1).tolist()
            for k, r in enumerate(rows[b:b + args.batch]):
                py = sum(math.exp(lp[c][k]) for c in YES)
                pn = sum(math.exp(lp[c][k]) for c in NO)
                out_rows.append({"item_id": r["item_id"], "dataset": r["dataset"], "split": r["split"],
                                 "p_yes": py, "p_no": pn, "mass": py + pn,
                                 "p_yes_norm": py / (py + pn) if py + pn > 0 else None})
    wall = time.time() - t
    with open(d / "verbal.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    metrics = {"model": args.model, "n": len(out_rows), "wall_s": wall, "first_token_shortcut": shortcut,
               "candidate_token_ids": cand, "transformers": transformers.__version__,
               "n_mass_below_0.5": sum(r["mass"] < 0.5 for r in out_rows)}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__},
                   {"initial": PROMPT, "verdict_query": QUERY})
    register(run_id, "done", block=f"p1-{args.kind}", exp="p1", model=args.model)
    print(json.dumps({"run_id": run_id, **{k: metrics[k] for k in ("n", "wall_s", "first_token_shortcut", "n_mass_below_0.5")}}))


if __name__ == "__main__":
    main()
