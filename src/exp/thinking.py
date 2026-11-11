"""Thinking subset (exploratory; design "Thinking subset", prereg §1, §10; blocks.yaml 4B-thinking): the Exp 1A
conditions {B1, F0, F1, F2, F3(0.7), F3(1.0), F5} and the Exp 1B 2 x 2 at F0 on D-nat test items, with every
revision turn generated in thinking mode. Written after the freeze (DEVIATIONS D7).

- a0 is the frozen P1 non-thinking initial answer, so the items can be stratified by initial correctness before any
  generation and every verdict (F3 masks, the F5 self-verdict at t*) is the one the non-thinking runs used.
- Items: original-pool D-nat items of the split, stratified by a0 correctness with equal allocation (--per-stratum
  per stratum, the first ones in the seeded order); weights N_stratum / n_stratum in subset.json. Chunk c takes the
  next --chunk-size / 2 items of each stratum, so every chunk is balanced and complete.
- Prompts: the runner's (src/exp/runner.py build_stage1/2 via src/gen/render.py), with the non-thinking generation
  suffix "<think>\\n\\n</think>\\n\\n" removed, so the model opens its own think block. F1's "Wait," prefill goes
  inside the think block ("<think>\\nWait,"). The F2 critique enters the second turn with its think block removed
  (the chat template drops reasoning from earlier turns); an unclosed think block is passed as it is.
- Decoding (prereg §1): T 0.6, top-p 0.95, top-k 20, min-p 0, --max-tokens 16384 per turn, 2 paired seeds: the seed
  of (item, s) is shared by every condition of that item. vLLM with VLLM_BATCH_INVARIANT=1.
- Scoring: src/score/scorer.py (the think block is stripped; truncated or unclosed = wrong, counted per condition).
Usage: VLLM_BATCH_INVARIANT=1 python -m src.exp.thinking --model 4B --split test --allow-test --initial <p1 initial>
       --verbal <p1 verbal> --self-threshold-s T --chunk-index C [--chunk-size 150] [--per-stratum 150] [--limit N]
       [--dry-run --tokenizer PATH]
"""
import argparse
import json
import time
from collections import Counter, defaultdict

from ..data.build import h
from ..data.verifiers import f3_verdict, self_verdict
from ..gen import render as R
from ..runlib import MODELS, REPO, register, require_batch_invariant, run_dir, sha256_text, write_manifest
from ..score import scorer
from ..score.scorer import SCORER_VERSION, score, strip_thinking
from .runner import CELLS_1B, jl

NO_THINK = "<think>\n\n</think>\n\n"
THINK_PREFILL = "<think>\n" + R.TRIGGER_PREFILL
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
N_SEEDS = 2
CONDS_1A = ["B1", "F0", "F1", "F2", "F3(0.7)", "F3(1.0)", "F5"]


def thinking(rendered):
    """The same prompt with the model's own think block left open."""
    if not rendered.text.endswith(NO_THINK):
        raise ValueError("rendered prompt does not end with the non-thinking generation suffix")
    return rendered.text[: -len(NO_THINK)]


def subset(dnat, split, per_stratum):
    """Equal allocation by a0 correctness, seeded order within each stratum."""
    rank = {c: k for k, c in enumerate(json.loads((REPO / "data" / "processed" / "split.json").read_text())["order"][split])}
    out = {}
    for c0 in (False, True):
        pool = sorted((r for r in dnat if r["correct0"] == c0), key=lambda r: (rank[r["canonical_id"]], r["item_id"]))
        out[c0] = {"N": len(pool), "items": [r["item_id"] for r in pool[:per_stratum]]}
    return out


def chunk_items(sub, chunk_index, chunk_size):
    half = chunk_size // 2
    return [i for c0 in (False, True) for i in sub[c0]["items"][chunk_index * half:(chunk_index + 1) * half]]


def turn1_requests(tok, r, self_v):
    """(cond, prompt, verdict) for every first-turn generation of one item."""
    q, a0 = r["question"], r["a0"]
    reqs = [("B1", thinking(R.fresh_prompt(tok, q)), None),
            ("F0", thinking(R.review_prompt(tok, q, a0)), None),
            ("F1", thinking(R.review_prompt(tok, q, a0)) + THINK_PREFILL, None),
            ("F2:turn1", thinking(R.review_prompt(tok, q, a0, instruction=R.CRITIQUE_TURN)), None)]
    for p in (0.7, 1.0):
        v = f3_verdict(r["item_id"], r["correct0"], p)
        reqs.append((f"F3({p})", thinking(R.review_prompt(tok, q, a0, feedback=R.VERDICT[v])), v))
    v = self_v
    reqs.append(("F5", thinking(R.review_prompt(tok, q, a0, feedback=R.VERDICT[v])), v))
    for role, label in CELLS_1B:
        if (role, label) == ("assistant", "self"):
            continue   # identical prompt to 1A F0: one generation, two rows
        reqs.append((f"1B:{role}/{label}/F0", thinking(R.review_prompt(tok, q, a0, role=role, label=label)), None))
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--initial", required=True)
    ap.add_argument("--verbal", required=True)
    ap.add_argument("--self-threshold-s", type=float, required=True)
    ap.add_argument("--chunk-index", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, default=150)
    ap.add_argument("--per-stratum", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0, help="items per stratum in this chunk (smoke tests)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("refusing test split before the prereg freeze (design §5.3)")
    cfg = {**vars(args), "scorer_version": SCORER_VERSION, "sampling": SAMPLING, "n_seeds": N_SEEDS,
           "think_prefill": THINK_PREFILL,
           "wording_sha256": sha256_text(json.dumps([R.PROMPT, R.LABELS, R.INSTRUCTION, R.VERDICT, R.CRITIQUE_TURN,
                                                     R.TRIGGER_PREFILL], sort_keys=True))}

    dnat_meta = {r["item_id"]: r for r in jl(REPO / "data" / "processed" / "dnat.jsonl")}
    init = {r["item_id"]: r for r in jl(args.initial)}
    verbal = {r["item_id"]: r for r in jl(args.verbal)}
    dnat = [{**dnat_meta[i], "a0": init[i]["a0"], "correct0": init[i]["correct"], "verbal": verbal[i]}
            for i in init if init[i]["split"] == args.split and i in dnat_meta]
    by_id = {r["item_id"]: r for r in dnat}
    sub = subset(dnat, args.split, args.per_stratum)
    ids = chunk_items(sub, args.chunk_index, args.chunk_size)
    if args.limit:
        ids = [i for c0 in (False, True) for i in [j for j in ids if by_id[j]["correct0"] == c0][:args.limit]]
    if not ids:
        raise SystemExit(f"chunk {args.chunk_index}: no items")
    self_v = {i: self_verdict(by_id[i]["verbal"], args.self_threshold_s) for i in ids}
    seed = {(i, s): int(h("think-seed-v1", i, s)[:8], 16) for i in ids for s in range(N_SEEDS)}
    weights = {str(c0): {"N": sub[c0]["N"], "n": len(sub[c0]["items"]), "weight": sub[c0]["N"] / len(sub[c0]["items"])}
               for c0 in (False, True)}

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer or MODELS[args.model])
        reqs = [x for i in ids for x in turn1_requests(tok, by_id[i], self_v[i])]
        print(json.dumps({"items": len(ids), "strata": Counter(by_id[i]["correct0"] for i in ids), "weights": weights,
                          "turn1_requests_per_seed": len(reqs), "conds": Counter(c for c, _, _ in reqs)}, indent=1, default=str))
        print("--- F1 prompt tail:", repr(reqs[2][1][-160:]))
        return

    require_batch_invariant()
    import vllm
    from vllm import LLM, SamplingParams
    kind = f"thinking-{args.split}-c{args.chunk_index}" + (f"-{args.tag}" if args.tag else "")
    run_id, d = run_dir("p2" if args.split == "dev" else "main", args.model, kind)
    register(run_id, "running", block=f"{args.model}-thinking", exp="thinking", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85,
              max_model_len=args.max_tokens + 8192)
    tok = llm.get_tokenizer()

    def gen(batch):
        """batch: list of (row, prompt text, seed) -> list of output dicts (text includes any prefill)."""
        params = [SamplingParams(**SAMPLING, max_tokens=args.max_tokens, seed=s) for _, _, s in batch]
        outs = llm.generate([p for _, p, _ in batch], params, use_tqdm=False)
        res = []
        for (row, p, _), o in zip(batch, outs):
            pre = THINK_PREFILL if p.endswith(THINK_PREFILL) else ""
            res.append({"text": pre + o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                        "out_tokens": len(o.outputs[0].token_ids), "prompt_tokens": len(o.prompt_token_ids)})
        return res

    batch1 = []
    for i in ids:
        r = by_id[i]
        base = {"item_id": i, "dataset": r["dataset"], "correct0": r["correct0"]}
        for s in range(N_SEEDS):
            for cond, prompt, v in turn1_requests(tok, r, self_v[i]):
                batch1.append(({**base, "seed": s, "cond": cond, "verdict": v}, prompt, seed[(i, s)]))
    t1 = time.time()
    out1 = gen(batch1)
    batch2 = []
    for (row, _, sd), o in zip(batch1, out1):
        if row["cond"] == "F2:turn1":
            r = by_id[row["item_id"]]
            crit = strip_thinking(o["text"])
            hist = [(R.CRITIQUE_TURN, o["text"] if crit is None else crit)]
            batch2.append(({**row, "cond": "F2"}, thinking(R.review_prompt(tok, r["question"], r["a0"], history=hist)), sd))
    t2 = time.time()
    out2 = gen(batch2)
    t3 = time.time()
    rows = []
    with open(d / "generations.jsonl", "w", encoding="utf-8") as f:
        for (row, _, _), o in zip(batch1 + batch2, out1 + out2):
            f.write(json.dumps({**{k: row[k] for k in ("item_id", "seed", "cond")}, **o}, ensure_ascii=False) + "\n")
            names = [row["cond"]] + (["1B:assistant/self/F0"] if row["cond"] == "F0" else [])
            s = score(o["text"], dnat_meta[row["item_id"]]["gold"], finish_reason=o["finish_reason"])
            for name in names:
                rows.append({**row, "cond": name, "correct": s["correct"], "status": s["status"], "pred": s["pred"],
                             "out_tokens": o["out_tokens"], "finish_reason": o["finish_reason"],
                             "stage": 1 if name == "F2:turn1" else None})
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    stats = defaultdict(Counter)
    for row in rows:
        stats[row["cond"]]["n"] += 1
        stats[row["cond"]][row["status"]] += 1
    (d / "subset.json").write_text(json.dumps({"weights": weights, "chunk_items": ids,
                                               "self_verdict": self_v}, indent=1))
    metrics = {"model": args.model, "split": args.split, "chunk": args.chunk_index, "items": len(ids),
               "vllm": vllm.__version__, "load_s": t1 - t0, "turn1_s": t2 - t1, "turn2_s": t3 - t2,
               "turn1_requests": len(batch1), "turn2_requests": len(batch2),
               "out_tokens_total": sum(o["out_tokens"] for o in out1 + out2),
               "rows": {k: dict(v) for k, v in sorted(stats.items())}, "scorer_guarded": scorer.GUARDED,
               "score_s": time.time() - t3}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__},
                   {"wording": cfg["wording_sha256"]})
    register(run_id, "done", block=f"{args.model}-thinking", exp="thinking", model=args.model)
    print(json.dumps({"run_id": run_id, **{k: metrics[k] for k in ("items", "turn1_requests", "turn2_requests",
                                                                  "turn1_s", "turn2_s", "out_tokens_total")}}))


if __name__ == "__main__":
    main()
