"""Block 4B-exp1c-robust (configs/blocks.yaml priority 9; exploratory, prereg §3.3): Exp 1C channel, sampled
robustness and F3 verifier-mask robustness on the original-pool D-nat items of a split. Written after the freeze;
src/exp/runner.py, src/gen/render.py and src/data/verifiers.py are reused unchanged.

a0 is the frozen P1 non-thinking initial answer (--initial), as in every Exp 1A run. Items run in the split's seeded
order (data/processed/split.json "order", ties by item_id), in complete paired chunks of --chunk-size (100): every
condition below finishes for a chunk's items before the next chunk starts.

1. Exp 1C (all items of the pool), greedy: the F3(0.9) verdict (verifier stream seed 0, as in Exp 1A) delivered as a
   tool response, a plain user message (the existing F3 rendering) and inside the assistant turn that holds a0
   (src/gen/render_channel.py). The plain-user channel is regenerated here so that all three channels come from one
   run at one locked commit (CLAUDE.md rule 8); under VLLM_BATCH_INVARIANT=1 it should equal the Exp 1A F3(0.9)
   generation byte for byte, and its prompt_key uses the runner's key formula so the two can be joined.
   Plus a plain-user F3(0.7) cell on every item, so that H1c's reference contrast, acc F3(0.9) - F3(0.7) on the
   plain-user channel, comes from the same run (rule 8). The masks are nested, so it shares the F3(0.9) "user"
   generation wherever the two verdicts agree (and the mask-robustness seed-0 F3(0.7) generation in the subset).
   Analysis: analysis/p4_channel.py.
2. Sampled robustness (the first --subset = 300 items in the seeded order, i.e. chunks 0-2 at chunk size 100): the
   Exp 1A D-nat conditions B0, B1, F0, F1, F2, F2-scratch, F3(0.5/0.7/0.9/1.0), F5, F3-perm[0..2] (F4 is D-inj only),
   3 seeds, temperature 0.7, top-p 0.8 (design "Sampled robustness"; prereg §3.3); other sampling parameters at vLLM
   defaults (top-k off, min-p 0), the effective values recorded in metrics.json. Budgets as Exp 1A: revision 2048,
   F2 / F2-scratch first turn 1024. Seeds are paired: the seed of (item, s) is h("robust-seed-v1", item, s), shared by
   every condition and both turns of that item. Verdict conditions with the same verdict value then have the same
   prompt and seed, so they share one generation (identical to generating each, under batch invariance).
3. Verifier-mask robustness (the same 300 items), greedy: F3(p), p in {0.5, 0.7, 0.9, 1.0}, under verifier-stream
   seeds 1 and 2 (prereg §6: "seed 0 primary, seeds 1-2 robustness"), plus seed 0 as the in-run reference. Greedy
   output depends only on the verdict value, so this is at most 2 generations per item (one shared with Exp 1C "user").
F2-scratch on the full pool already ran in the Exp 1A runner (src/exp/runner.py), so the block's "F2-scratch" part
needs nothing here beyond its place among the sampled conditions.

F5 uses the model's t* (--self-threshold-s); F3-perm permutes the self-verdicts over the whole pool of the split,
exactly as the runner does, so each item's verdicts equal the Exp 1A ones whatever the chunk.

Each chunk is written as its own run (run kind exp1c-robust-<split>-c<C>) as soon as it completes. One invocation can
run chunks C..L (--through L) with a single engine load (about 150 s per load); --deadline-epoch stops before a chunk
that would not finish in time (estimate: the previous chunk's seconds per generation x 1.2 + 60 s). A chunk killed
mid-way leaves a "running" entry and no rows; it is re-run from scratch at the same locked commit.

Usage (run host): VLLM_BATCH_INVARIANT=1 python -m src.exp.channel --model 4B --split test --allow-test
         --initial <p1 initial.jsonl> --verbal <p1 verbal.jsonl> --self-threshold-s T --chunk-index C [--through L]
         [--deadline-epoch E]
       Smoke test: --split dev --chunk-index 0 --limit 2 --tag smoke
       Dry run (local, counts only): ... --dry-run --tokenizer data/tokenizers/qwen3-4b [--all-chunks]
"""
import argparse
import json
import time
from collections import Counter, defaultdict

from ..data.build import h
from ..data.verifiers import f3_verdict, permute_self_verdicts, self_verdict
from ..gen import render as R
from ..gen import render_channel as RC
from ..runlib import MODELS, REPO, register, require_batch_invariant, run_dir, sha256_text, write_manifest
from ..score import scorer
from ..score.scorer import SCORER_VERSION, score
from .runner import P_LEVELS, jl

P_CHANNEL = 0.9
P_CHANNEL_REF = 0.7            # H1c compares the channel range with acc F3(0.9) - F3(0.7), plain-user channel
MASK_SEEDS = (0, 1, 2)         # verifier stream seeds; 0 = primary (in-run reference), 1-2 = robustness
N_SAMPLE_SEEDS = 3
SAMPLING = {"temperature": 0.7, "top_p": 0.8}
SEED_STREAM = "robust-seed-v1"
MAX_MODEL_LEN = 12288          # as src/exp/runner.py
# The runner's main wording list: its hash must equal the prereg's main wording_sha256 (prereg §1, §13).
MAIN_WORDING = [R.PROMPT, R.LABELS, R.INSTRUCTION, R.VERDICT, R.LOCATION, R.CRITIQUE_TURN, R.SCRATCH_TURN,
                R.TRIGGER_PREFILL, R.EXP0_INSTRUCTION]


def main_wording_sha256():
    return sha256_text(json.dumps(MAIN_WORDING, sort_keys=True))


def sample_seed(item_id, s):
    return int(h(SEED_STREAM, item_id, s)[:8], 16)


class Plan:
    """Requests and the rows they feed. seed None = greedy (temperature 0, seed 0, as the runner); otherwise sampled
    with SAMPLING and that seed. Greedy keys use the runner's formula, so an identical greedy prompt has the same
    prompt_key as in the Exp 1A run."""

    def __init__(self):
        self.rows, self.prompts = [], {}

    def add(self, row, rendered, max_tokens, seed=None):
        key = rendered.text + "\x00" + rendered.prefill + f"\x00{max_tokens}"
        if seed is not None:
            key += f"\x00sample:{seed}"
        key = sha256_text(key)
        self.prompts[key] = (rendered, max_tokens, seed)
        row["prompt_key"] = key
        self.rows.append(row)
        return key


def load_pool(initial, verbal, split, items=REPO / "data" / "processed" / "dnat.jsonl"):
    """Original-pool D-nat items of the split with their frozen a0, sorted in the split's seeded order."""
    meta = {r["item_id"]: r for r in jl(items)}
    init = {r["item_id"]: r for r in jl(initial)}
    verb = {r["item_id"]: r for r in jl(verbal)}
    pool = [{**meta[i], "a0": init[i]["a0"], "correct0": init[i]["correct"], "verbal": verb[i]}
            for i in init if init[i]["split"] == split and i in meta]
    rank = {c: k for k, c in enumerate(json.loads((REPO / "data" / "processed" / "split.json").read_text())["order"][split])}
    return sorted(pool, key=lambda r: (rank[r["canonical_id"]], r["item_id"]))


def verdicts(pool, threshold_s):
    """F5 self-verdicts and their K = 3 stratum permutations over the whole pool (as src/exp/runner.py build_stage1)."""
    self_v = {r["item_id"]: self_verdict(r["verbal"], threshold_s) for r in pool}
    perm = permute_self_verdicts([{"item_id": r["item_id"], "dataset": r["dataset"], "initially_correct": r["correct0"],
                                   "self_says_correct": self_v[r["item_id"]]} for r in pool])
    return self_v, perm


def chunk_items(ordered, chunk_index, chunk_size):
    return ordered[chunk_index * chunk_size:(chunk_index + 1) * chunk_size]


def build_stage1(tok, items, subset, self_v, perm, rev=2048, crit=1024):
    """items: pool rows of this chunk; subset: item_ids that also get the sampled and mask-robustness conditions."""
    plan = Plan()
    for r in items:
        i, c0, q, a0 = r["item_id"], r["correct0"], r["question"], r["a0"]
        base = {"item_id": i, "dataset": r["dataset"], "data": "dnat", "correct0": c0}
        v = f3_verdict(i, c0, P_CHANNEL)
        for ch in RC.CHANNELS:
            plan.add({**base, "exp": "1C", "cond": f"F3({P_CHANNEL}):{ch}", "channel": ch, "verdict": v},
                     RC.channel_prompt(tok, q, a0, R.VERDICT[v], ch), rev)
        v = f3_verdict(i, c0, P_CHANNEL_REF)   # H1c's reference contrast, plain-user channel, in the same run
        plan.add({**base, "exp": "1C", "cond": f"F3({P_CHANNEL_REF}):user", "channel": "user", "verdict": v},
                 RC.channel_prompt(tok, q, a0, R.VERDICT[v], "user"), rev)
        if i not in subset:
            continue
        for k in MASK_SEEDS:
            for p in P_LEVELS:
                v = f3_verdict(i, c0, p, seed=k)
                plan.add({**base, "exp": "mask", "cond": f"F3({p})", "mask_seed": k, "verdict": v},
                         R.review_prompt(tok, q, a0, feedback=R.VERDICT[v]), rev)
        for s in range(N_SAMPLE_SEEDS):
            sd = sample_seed(i, s)
            b = {**base, "exp": "sampled", "sample": s, "seed": sd}
            plan.rows.append({**b, "cond": "B0", "prompt_key": None, "fixed_correct": c0})
            plan.add({**b, "cond": "B1"}, R.fresh_prompt(tok, q), rev, sd)
            plan.add({**b, "cond": "F0"}, R.review_prompt(tok, q, a0), rev, sd)
            plan.add({**b, "cond": "F1"}, R.review_prompt(tok, q, a0, prefill=R.TRIGGER_PREFILL), rev, sd)
            for cond, turn in (("F2", R.CRITIQUE_TURN), ("F2-scratch", R.SCRATCH_TURN)):
                plan.add({**b, "cond": f"{cond}:turn1", "stage": 1}, R.review_prompt(tok, q, a0, instruction=turn), crit, sd)
            for p in P_LEVELS:
                v = f3_verdict(i, c0, p)
                plan.add({**b, "cond": f"F3({p})", "verdict": v}, R.review_prompt(tok, q, a0, feedback=R.VERDICT[v]), rev, sd)
            v = self_v[i]
            plan.add({**b, "cond": "F5", "verdict": v}, R.review_prompt(tok, q, a0, feedback=R.VERDICT[v]), rev, sd)
            for k, v in enumerate(perm[i]):
                plan.add({**b, "cond": f"F3-perm[{k}]", "verdict": v}, R.review_prompt(tok, q, a0, feedback=R.VERDICT[v]),
                         rev, sd)
    return plan


def build_stage2(tok, plan1, out1, by_id, rev=2048):
    """F2 / F2-scratch revisions after the sampled first turn, with the first turn's seed (as src/exp/thinking.py)."""
    plan = Plan()
    for row in plan1.rows:
        if row.get("stage") != 1:
            continue
        r = by_id[row["item_id"]]
        turn = R.CRITIQUE_TURN if row["cond"] == "F2:turn1" else R.SCRATCH_TURN
        hist = [(turn, out1[row["prompt_key"]]["text"])]
        keep = {k: row[k] for k in ("item_id", "dataset", "data", "correct0", "exp", "sample", "seed")}
        plan.add({**keep, "cond": row["cond"].split(":")[0]}, R.review_prompt(tok, r["question"], r["a0"], history=hist),
                 rev, row["seed"])
    return plan


def solution_tokens_identical(tok, r):
    """Design §6.0: a0's token sequence is the same in the three channel prompts (HF fast tokenizer offsets)."""
    v = f3_verdict(r["item_id"], r["correct0"], P_CHANNEL)
    seqs = set()
    for ch in RC.CHANNELS:
        ids, spans = R.token_spans(tok, RC.channel_prompt(tok, r["question"], r["a0"], R.VERDICT[v], ch))
        a, b = spans["solution"]
        seqs.add(tuple(ids[a:b]))
    return len(seqs) == 1


def counts(plan1, n_stage2):
    greedy = sum(seed is None for _, _, seed in plan1.prompts.values())
    return {"stage1_greedy": greedy, "stage1_sampled": len(plan1.prompts) - greedy, "stage2_sampled": n_stage2,
            "generations": len(plan1.prompts) + n_stage2}


def generate(llm, plan, sampling_cls):
    keys = list(plan.prompts)
    if not keys:
        return {}
    texts = [plan.prompts[k][0].text + plan.prompts[k][0].prefill for k in keys]
    params = [sampling_cls(temperature=0.0, max_tokens=mt, seed=0) if sd is None
              else sampling_cls(**SAMPLING, max_tokens=mt, seed=sd) for _, mt, sd in (plan.prompts[k] for k in keys)]
    outs = llm.generate(texts, params, use_tqdm=False)
    return {k: {"text": plan.prompts[k][0].prefill + o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                "out_tokens": len(o.outputs[0].token_ids), "prompt_tokens": len(o.prompt_token_ids)}
            for k, o in zip(keys, outs)}


def score_rows(plan, outs, golds):
    for row in plan.rows:
        if row.get("stage") == 1:
            continue
        if row["prompt_key"] is None:
            row["correct"], row["status"] = row["fixed_correct"], "fixed"
            continue
        o = outs[row["prompt_key"]]
        s = score(o["text"], golds[row["item_id"]], finish_reason=o["finish_reason"])
        row.update({"correct": s["correct"], "status": s["status"], "pred": s["pred"], "out_tokens": o["out_tokens"],
                    "prompt_tokens": o["prompt_tokens"], "finish_reason": o["finish_reason"]})


def run_chunk(llm, tok, hf_tok, sampling_cls, vllm_version, args, cfg, c, items, ordered, by_id, subset, self_v, perm):
    """Generate, score and write one complete chunk as its own run (run ID ...-c<C>-<model>-<commit>)."""
    rev, crit = args.revision_tokens, args.critique_tokens
    cfg = {**cfg, "chunk_index": c}
    flags = {r["item_id"]: solution_tokens_identical(hf_tok, r) for r in items}
    kind = f"exp1c-robust-{args.split}-c{c}" + (f"-{args.tag}" if args.tag else "")
    run_id, d = run_dir("p2" if args.split == "dev" else "main", args.model, kind)
    block = f"{args.model}-exp1c-robust"
    register(run_id, "running", block=block, exp="exp1c-robust", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t1 = time.time()
    plan1 = build_stage1(tok, items, subset, self_v, perm, rev, crit)
    out1 = generate(llm, plan1, sampling_cls)
    plan2 = build_stage2(tok, plan1, out1, by_id, rev)
    t2 = time.time()
    out2 = generate(llm, plan2, sampling_cls)
    t3 = time.time()
    # Generations are written before scoring (DEVIATIONS D1).
    with open(d / "generations.jsonl", "w", encoding="utf-8") as f:
        for plan, outs in ((plan1, out1), (plan2, out2)):
            for k, o in outs.items():
                sd = plan.prompts[k][2]
                f.write(json.dumps({"prompt_key": k, "sampling": "greedy" if sd is None else sd, **o},
                                   ensure_ascii=False) + "\n")
    golds = {i: by_id[i]["gold"] for i in by_id}
    score_rows(plan1, out1, golds)
    score_rows(plan2, out2, golds)
    rows = [row for row in plan1.rows + plan2.rows if row.get("stage") != 1]
    for row in rows:
        if row["exp"] == "1C":
            row["solution_tokens_identical"] = flags[row["item_id"]]
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    stats = defaultdict(Counter)
    for row in rows:
        key = f"{row['exp']}:{row['cond']}" + (f":mask{row['mask_seed']}" if row["exp"] == "mask" else "")
        stats[key]["n"] += 1
        stats[key][row["status"]] += 1
    (d / "chunk.json").write_text(json.dumps({
        "chunk_items": [r["item_id"] for r in items], "subset_items": [r["item_id"] for r in items if r["item_id"] in subset],
        "subset_rule": f"first {args.subset} of the split's original-pool D-nat items in the seeded order "
                       "(data/processed/split.json order, ties by item_id)",
        "pool": len(ordered), "subset": args.subset, "self_verdict": {r["item_id"]: self_v[r["item_id"]] for r in items},
        "solution_tokens_identical": flags}, indent=1))
    metrics = {"model": args.model, "split": args.split, "chunk": c, "items": len(items),
               "subset_items": sum(r["item_id"] in subset for r in items), "vllm": vllm_version,
               "stage1_s": t2 - t1, "stage2_s": t3 - t2, **counts(plan1, len(plan2.prompts)),
               "sampling_params": {"greedy": repr(sampling_cls(temperature=0.0, max_tokens=rev, seed=0)),
                                   "sampled": repr(sampling_cls(**SAMPLING, max_tokens=rev, seed=0))},
               "out_tokens_total": sum(o["out_tokens"] for o in list(out1.values()) + list(out2.values())),
               "solution_tokens_identical": f"{sum(flags.values())}/{len(flags)}",
               "rows": {k: dict(v) for k, v in sorted(stats.items())}, "scorer_guarded": list(scorer.GUARDED),
               "score_s": time.time() - t3}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm_version},
                   {"wording": cfg["wording_sha256"], "channel_wording": cfg["channel_wording_sha256"]})
    register(run_id, "done", block=block, exp="exp1c-robust", model=args.model)
    print(json.dumps({"run_id": run_id, **{k: metrics[k] for k in ("items", "subset_items", "generations",
                                                                  "stage1_s", "stage2_s", "solution_tokens_identical")}}),
          flush=True)
    return time.time() - t1, metrics["generations"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--initial", required=True)
    ap.add_argument("--verbal", required=True)
    ap.add_argument("--self-threshold-s", type=float, required=True)
    ap.add_argument("--chunk-index", type=int, required=True, help="first chunk to run")
    ap.add_argument("--through", type=int, default=None,
                    help="last chunk to run (inclusive) with the same engine; default: --chunk-index only")
    ap.add_argument("--deadline-epoch", type=float, default=0,
                    help="do not start a chunk whose estimated time would end after this epoch")
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--subset", type=int, default=300, help="sampled and mask robustness: first N in the seeded order")
    ap.add_argument("--revision-tokens", type=int, default=2048)
    ap.add_argument("--critique-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="first N items of each chunk (smoke tests)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all-chunks", action="store_true", help="with --dry-run: counts for every chunk of the split")
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("refusing test split before the prereg freeze (design §5.3)")
    operational = ("chunk_index", "through", "deadline_epoch", "dry_run", "all_chunks", "tokenizer")   # set per chunk / not config
    cfg = {**{k: v for k, v in vars(args).items() if k not in operational}, "scorer_version": SCORER_VERSION,
           "sampling": SAMPLING, "n_sample_seeds": N_SAMPLE_SEEDS, "seed_stream": SEED_STREAM,
           "mask_seeds": list(MASK_SEEDS), "p_channel": P_CHANNEL, "p_channel_ref": P_CHANNEL_REF,
           "channels": list(RC.CHANNELS),
           "max_model_len": MAX_MODEL_LEN, "wording_sha256": main_wording_sha256(),
           "channel_wording_sha256": sha256_text(RC.channel_wording_json())}

    ordered = load_pool(args.initial, args.verbal, args.split)
    by_id = {r["item_id"]: r for r in ordered}
    subset = {r["item_id"] for r in ordered[:args.subset]}
    self_v, perm = verdicts(ordered, args.self_threshold_s)
    n_chunks = -(-len(ordered) // args.chunk_size)
    last = args.chunk_index if args.through is None else args.through
    chunks = list(range(n_chunks)) if args.all_chunks else list(range(args.chunk_index, last + 1))
    if not chunks or chunks[0] < 0 or chunks[-1] >= n_chunks:
        raise SystemExit(f"chunks {args.chunk_index}..{last}: the split has chunks 0..{n_chunks - 1}")

    def items_of(c):
        its = chunk_items(ordered, c, args.chunk_size)
        return its[:args.limit] if args.limit else its

    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(args.tokenizer or MODELS[args.model])
    if args.dry_run:
        # Counts only: no prompt text and no a0 is printed (test-split discipline, design §5.3).
        report = {"pool": len(ordered), "chunks": n_chunks, "subset": len(subset), "per_chunk": {}}
        for c in chunks:
            its = items_of(c)
            p1 = build_stage1(hf_tok, its, subset, self_v, perm, args.revision_tokens, args.critique_tokens)
            n2 = sum(row.get("stage") == 1 for row in p1.rows)
            report["per_chunk"][c] = {"items": len(its), "subset_items": sum(r["item_id"] in subset for r in its),
                                      **counts(p1, n2),
                                      "solution_tokens_identical": sum(solution_tokens_identical(hf_tok, r) for r in its)}
            if c == chunks[0]:
                report[f"rows_chunk_{c}"] = Counter(f"{row['exp']}:{row['cond']}" for row in p1.rows)
        report["total_generations"] = sum(v["generations"] for v in report["per_chunk"].values())
        print(json.dumps(report, indent=1, default=str))
        return

    require_batch_invariant()
    import vllm
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85,
              max_model_len=MAX_MODEL_LEN)
    tok = llm.get_tokenizer()
    print(json.dumps({"load_s": time.time() - t0, "chunks": chunks}), flush=True)
    prev = None
    for c in chunks:
        items = items_of(c)
        if args.deadline_epoch and prev:
            p1 = build_stage1(tok, items, subset, self_v, perm, args.revision_tokens, args.critique_tokens)
            n_gen = counts(p1, sum(row.get("stage") == 1 for row in p1.rows))["generations"]
            est = prev[0] / prev[1] * n_gen * 1.2 + 60
            if time.time() + est > args.deadline_epoch:
                print(json.dumps({"budget_stop_before_chunk": c, "estimated_s": est,
                                  "seconds_to_deadline": args.deadline_epoch - time.time()}), flush=True)
                break
        prev = run_chunk(llm, tok, hf_tok, SamplingParams, vllm.__version__, args, cfg, c, items, ordered, by_id,
                         subset, self_v, perm)


if __name__ == "__main__":
    main()
