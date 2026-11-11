"""Exp 5 on the test split (H4a estimation-only; H4b, H4d exploratory; prereg §3.1, §6, §7), frozen from the dev pilot.

Pool ("natural", prereg §3.4): the first --n-items original-pool D-nat test items in the seeded processing order
(data/processed/split.json "order"), the spillover set excluded.
Verdicts, built exactly as the Exp 1A runner builds them (same population, the original-pool D-nat test items of the
--initial run, so perm-self here equals F3-perm there): self = the thresholded verbal verdict at --self-threshold-s;
perm-self k = 0..2 (src/data/verifiers.py); ext-1.0 = the true correctness of a0; shuffled = self-verdicts permuted
within dataset (seed h("shuffled-v1", dataset), as in the dev pilot).
Seven adapters per item at the frozen tau and lr (analysis/out/tier2.json; src/ttt/update.py, 4 steps): self, perm0-2,
ext, shuffled (trained; A seeded by h("ttt-seed-v1", item), shared across groundings) and random (matched norm, seed
h("ttt-random-v1", item)); one zero adapter per run.

Stages, per chunk of --chunk items, each resumable (a stage whose marker file is missing is redone in full):
  train     HF, deterministic. The updates, each logged as in src/ttt/update.py, plus KL(base || adapted) and mean
            |delta logit| over the a0 tokens of the first N_KL KL-probe items, and spillover (H4b): per spillover item,
            the mean log-prob of the tokens of the final \\boxed{gold}, teacher-forced in the base model's own a0 with
            its last boxed answer replaced by the gold (appended if a0 has none); the update's spillover is the mean
            over the spillover set of (adapted - base). Adapters saved in fp32.
  serve     vLLM, VLLM_BATCH_INVARIANT=1, max 2048 new tokens. Per item: B1 (q only), zero-step (must equal B1 token
            for token), context arm (v = correct, v = incorrect), parameter arm (7 adapters, q only), parameter +
            context (the 6 trained adapters, each with its own verdict); each cell greedy plus N_SAMPLES samples at
            T_SAMPLE with one seed per item shared by every cell. H4d self-prediction (greedy, SELFPRED_TOKENS): base,
            self adapter, random adapter, and context with the self verdict. Scored with src/score/scorer.py.
  clean     deletes the chunk's adapters except a seeded KEEP_FRACTION sample (h("ttt-keep-v1", name)).
  finalize  joins the chunk files into rows.jsonl, texts.jsonl, updates.jsonl; metrics.json; manifest.
H4a uses the greedy parameter-arm accuracy, as the dev pilot and the P2 power simulation did; sample accuracy is
secondary. Choices not fixed by the prereg are in DEVIATIONS D2 (made before any Exp 5 test output existed).
Usage: python -m src.ttt.test --model 4B --initial <p1 initial> --verbal <p1 verbal> --self-threshold-s T
       --stage {train,serve,clean,finalize} [--chunk-index C] [--chunk 50] [--n-items 400] [--limit N] [--dry-run]
"""
import argparse
import json
import random
import shutil
import time
from collections import defaultdict

from ..data.build import h
from ..data.verifiers import permute_self_verdicts, self_verdict
from ..gen import render as R
from ..runlib import MODELS, REPO, register, run_dir, sha256_text, write_manifest
from ..score import scorer
from ..score.scorer import _boxed_spans, score
from . import update as U
from .pilot import STEPS

GROUNDINGS = ["self", "perm0", "perm1", "perm2", "ext", "shuffled", "random"]
N_KL, N_SAMPLES, T_SAMPLE = 10, 4, 0.7
MAX_TOKENS, SELFPRED_TOKENS = 2048, 64
KEEP_FRACTION = 0.05
SELFPRED = "State only the final answer you would give to this question. Put it within \\boxed{}."
TIER2 = REPO / "analysis" / "out" / "tier2.json"


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def unit(*parts):
    return int(h(*parts)[:8], 16) / 16 ** 8


def build(args):
    """Pool, chunks, verdicts and the fixed evaluation items; no model needed."""
    P = REPO / "data" / "processed"
    meta = {r["item_id"]: r for r in jl(P / "dnat.jsonl")}
    sp = json.loads((P / "split.json").read_text())
    rank = {c: i for i, c in enumerate(sp["order"]["test"])}
    spill, kl_dev = set(sp["spillover_test"]), set(sp["kl_dev"])
    init_all = {r["item_id"]: r for r in jl(args.initial)}
    init = {i: r for i, r in init_all.items() if r["split"] == "test"}
    verbal = {r["item_id"]: r for r in jl(args.verbal)}
    self_v = {i: self_verdict(verbal[i], args.self_threshold_s) for i in init}
    perm = permute_self_verdicts([{"item_id": i, "dataset": meta[i]["dataset"], "initially_correct": init[i]["correct"],
                                   "self_says_correct": self_v[i]} for i in init])
    shuffled, by_ds = {}, defaultdict(list)
    for i in init:
        by_ds[meta[i]["dataset"]].append(i)
    for ds, ids in by_ds.items():
        ids = sorted(ids)
        vs = [self_v[i] for i in ids]
        random.Random(h("shuffled-v1", ds)).shuffle(vs)
        shuffled.update(zip(ids, vs))
    pool = sorted((i for i in init if meta[i]["canonical_id"] not in spill),
                  key=lambda i: rank[meta[i]["canonical_id"]])[:args.n_items]
    verdicts = {i: {"self": self_v[i], **{f"perm{k}": perm[i][k] for k in range(3)}, "ext": bool(init[i]["correct"]),
                    "shuffled": shuffled[i], "random": None} for i in pool}
    spill_items = sorted((i for i in init if meta[i]["canonical_id"] in spill), key=lambda i: meta[i]["canonical_id"])
    kl_items = sorted((i for i, r in init_all.items() if r["split"] == "dev" and meta[i]["canonical_id"] in kl_dev),
                      key=lambda i: meta[i]["canonical_id"])[:N_KL]
    if args.limit:
        pool, spill_items, kl_items = pool[:args.limit], spill_items[:args.limit], kl_items[:args.limit]
    chunks = [pool[k:k + args.chunk] for k in range(0, len(pool), args.chunk)]
    return {"meta": meta, "init": init_all, "verdicts": verdicts, "chunks": chunks, "spill": spill_items, "kl": kl_items}


def gold_a0_ids(tok, a0_ids, gold):
    """The base model's a0 with its last boxed answer replaced by the gold (appended if there is none)."""
    text = tok.decode(list(a0_ids))
    boxes = _boxed_spans(text)
    if boxes:
        s, e, _ = boxes[-1]
        text = text[:s] + "\\boxed{" + gold + "}" + text[e:]
    else:
        end = "<|im_end|>" if text.endswith("<|im_end|>") else ""
        text = text[:len(text) - len(end)] + "\n\n\\boxed{" + gold + "}" + end
    return tok(text, add_special_tokens=False).input_ids


def masked_batches(tok, seqs, max_cells=6000, max_tokens=16384):
    """seqs: [(ids, mask)]. Left-padded micro-batches with explicit positions; per batch the number of trailing
    positions whose logits are needed (logits at t-1 predict the masked token t).
    max_cells caps kept logit positions; max_tokens caps padded input tokens. Without the second cap the 50 spillover
    sequences (short answer spans) formed one batch padded to the longest (4,218 tokens), about 8x the real tokens,
    and the per-adapter evaluation dominated Exp 5 training (2026-11-05, DEVIATIONS D4)."""
    import torch
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][0]))
    out, cur = [], []
    def need(k):
        ids, mask = seqs[k]
        return len(ids) - (mask.index(1) - 1)
    for k in order:
        trial = cur + [k]
        if cur and (len(trial) * max(need(j) for j in trial) > max_cells
                    or len(trial) * max(len(seqs[j][0]) for j in trial) > max_tokens):
            out.append(cur)
            trial = [k]
        cur = trial
    if cur:
        out.append(cur)
    batches = []
    for group in out:
        L = max(len(seqs[k][0]) for k in group)
        keep = max(need(k) for k in group)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = [[pad] * (L - len(seqs[k][0])) + seqs[k][0] for k in group]
        att = [[0] * (L - len(seqs[k][0])) + [1] * len(seqs[k][0]) for k in group]
        pos = [[0] * (L - len(seqs[k][0])) + list(range(len(seqs[k][0]))) for k in group]
        # Per item: rows of the kept logits that predict a masked token, and those tokens.
        sel, tgt = [], []
        for k in group:
            s_ids, mask = seqs[k]
            off = L - len(s_ids)
            ts = [t for t in range(1, len(s_ids)) if mask[t]]
            sel.append([off + t - 1 - (L - keep) for t in ts])
            tgt.append([s_ids[t] for t in ts])
        batches.append({"items": group, "keep": keep, "ids": torch.tensor(ids, device="cuda"),
                        "att": torch.tensor(att, device="cuda"), "pos": torch.tensor(pos, device="cuda"),
                        "sel": sel, "tgt": tgt})
    return batches


def span_logprobs(pm, batches):
    """{item index: [n_masked, V] fp32 log-softmax at the masked targets' predicting positions}."""
    import torch
    out = {}
    with torch.no_grad():
        for b in batches:
            lg = pm(input_ids=b["ids"], attention_mask=b["att"], position_ids=b["pos"], logits_to_keep=b["keep"]).logits
            for j, k in enumerate(b["items"]):
                out[k] = torch.log_softmax(lg[j, b["sel"][j]].float(), -1)
    return out


def gold_logprob(lp, batches):
    """{item index: mean log-prob of its masked target tokens}."""
    out = {}
    for b in batches:
        for j, k in enumerate(b["items"]):
            t = lp[k].new_tensor(b["tgt"][j], dtype=lp[k].dtype).long()
            out[k] = float(lp[k].gather(-1, t[:, None]).mean())
    return out


def stage_train(args, B, c, d):
    U.deterministic()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t2 = json.loads(TIER2.read_text())["exp5"][args.model]
    tau, lr = float(t2["tau"]), float(t2["lr"])
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    base = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation=U.ATTN)
    pm, layers, params = U.attach(base)
    pm.eval()
    meta, init = B["meta"], B["init"]
    seq = lambda i, ids: U.training_sequence(tok, meta[i]["question"], ids)
    kl_seqs = []
    for i in B["kl"]:
        ids, _ = seq(i, init[i]["a0_token_ids"])
        n0 = len(ids) - len(init[i]["a0_token_ids"])
        kl_seqs.append((ids, [0] * n0 + [1] * (len(ids) - n0)))
    sp_seqs = [seq(i, gold_a0_ids(tok, init[i]["a0_token_ids"], meta[i]["gold"])) for i in B["spill"]]
    assert all(m is not None for _, m in sp_seqs), "a spillover sequence has no answer span"
    kl_b, sp_b = masked_batches(tok, kl_seqs), masked_batches(tok, sp_seqs)
    U.reset(layers, 0)
    lp_kl0 = span_logprobs(pm, kl_b)
    sp0 = gold_logprob(span_logprobs(pm, sp_b), sp_b)
    adir = d / "adapters"
    adir.mkdir(exist_ok=True)
    if not (adir / "zero").exists():
        pm.save_pretrained(str(adir / "zero"))
    rows = []
    for i in B["chunks"][c]:
        ids, mask = seq(i, init[i]["a0_token_ids"])
        for g in GROUNDINGS:
            if g == "random":
                log = U.random_update(layers, tau, seed=int(h("ttt-random-v1", i)[:8], 16))
            else:
                log = U.update(pm, layers, params, ids, mask, B["verdicts"][i][g], lr=lr, steps=STEPS, tau=tau,
                               seed=int(h("ttt-seed-v1", i)[:8], 16))
            lp = span_logprobs(pm, kl_b)
            kl = sum(float((lp_kl0[k].exp() * (lp_kl0[k] - lp[k])).sum()) for k in lp) / sum(v.shape[0] for v in lp.values())
            dl = sum(float((lp[k] - lp_kl0[k]).abs().sum()) for k in lp) / sum(v.numel() for v in lp.values())
            sp = gold_logprob(span_logprobs(pm, sp_b), sp_b)
            name = f"{g}_{i}"
            pm.save_pretrained(str(adir / name))
            rows.append({"item_id": i, "chunk": c, "grounding": g, "verdict": B["verdicts"][i][g], "adapter": name,
                         "tau": tau, "lr": None if g == "random" else lr, **log, "kl_probe": kl, "mean_abs_dlogprob": dl,
                         "spillover": sum(sp[k] - sp0[k] for k in sp) / len(sp),
                         "spillover_items": {B["spill"][k]: sp[k] - sp0[k] for k in sp}})
        print(f"{time.strftime('%H:%M:%S')} chunk {c}: {i} trained", flush=True)
    # Reset invariant at the end of the chunk: B = 0 gives exactly the base logits (design Exp 5).
    U.reset(layers, 0)
    lp = span_logprobs(pm, kl_b)
    with pm.disable_adapter():
        lpb = span_logprobs(pm, kl_b)
    reset_exact = all(torch.equal(lp[k], lpb[k]) for k in lp)
    with open(d / "chunks" / f"updates_{c}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (d / "chunks" / f"train_{c}.json").write_text(json.dumps({"reset_logits_exact": reset_exact, "n_updates": len(rows),
                                                              "base_spillover": {B["spill"][k]: v for k, v in sp0.items()}}))
    if not reset_exact:
        raise SystemExit(f"chunk {c}: reset invariant failed")


def stage_serve(args, B, c, d):
    from ..runlib import require_batch_invariant
    require_batch_invariant()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    meta, init, V = B["meta"], B["init"], B["verdicts"]
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=12288,
              enable_lora=True, max_lora_rank=8, max_loras=args.max_loras, max_cpu_loras=512)
    tok = llm.get_tokenizer()
    greedy = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=0)
    short = SamplingParams(temperature=0.0, max_tokens=SELFPRED_TOKENS, seed=0)
    lora_ids = {}

    def lora(name):
        if name not in lora_ids:
            lora_ids[name] = LoRARequest(name, len(lora_ids) + 1, str(d / "adapters" / name))
        return lora_ids[name]

    def user(content):
        return R.render_messages(tok, [{"role": "user", "content": content}])

    reqs = []   # (row, prompt, sampling params, lora)
    for i in B["chunks"][c]:
        q, a0, c0 = meta[i]["question"], init[i]["a0"], init[i]["correct"]
        fresh = R.fresh_prompt(tok, q).text
        ctx = {v: R.context_prompt(tok, q, a0, v).text for v in (True, False)}
        samp = SamplingParams(n=N_SAMPLES, temperature=T_SAMPLE, max_tokens=MAX_TOKENS, seed=int(h("ttt-sample-v1", i)[:8], 16))
        base = {"item_id": i, "chunk": c, "correct0": c0}

        def cell(row, prompt, lr_=None):
            reqs.append(({**base, **row, "decode": "greedy"}, prompt, greedy, lr_))
            reqs.append(({**base, **row, "decode": "sample"}, prompt, samp, lr_))
        cell({"arm": "B1", "grounding": None}, fresh)
        reqs.append(({**base, "arm": "zero-step", "grounding": "zero", "decode": "greedy"}, fresh, greedy, lora("zero")))
        for v in (True, False):
            cell({"arm": "context", "grounding": None, "verdict": v}, ctx[v])
        for g in GROUNDINGS:
            cell({"arm": "parameter", "grounding": g, "verdict": V[i][g]}, fresh, lora(f"{g}_{i}"))
            if V[i][g] is not None:
                cell({"arm": "parameter+context", "grounding": g, "verdict": V[i][g]}, ctx[V[i][g]], lora(f"{g}_{i}"))
        sq = user(q + "\n\n" + SELFPRED)
        for of, prompt, lr_ in (("base", sq, None), ("self", sq, lora(f"self_{i}")), ("random", sq, lora(f"random_{i}")),
                                ("context", user(R.CONTEXT_MEMORY.format(prompt=q, a0=a0, verdict="correct" if V[i]["self"]
                                                                         else "incorrect") + "\n\n" + SELFPRED), None)):
            reqs.append(({**base, "arm": "selfpred", "grounding": of, "decode": "greedy"}, prompt, short, lr_))
    t = time.time()
    outs = llm.generate([p for _, p, _, _ in reqs], [s for _, _, s, _ in reqs], lora_request=[l for *_, l in reqs],
                        use_tqdm=False)
    wall = time.time() - t
    rows, texts, b1 = [], [], {}
    for (row, _, _, _), o in zip(reqs, outs):
        for k, out in enumerate(o.outputs):
            r = {**row, "sample": -1 if row["decode"] == "greedy" else k}
            s = score(out.text, meta[r["item_id"]]["gold"], finish_reason=out.finish_reason)
            r.update({"correct": s["correct"], "status": s["status"], "pred": s["pred"], "out_tokens": len(out.token_ids),
                      "finish_reason": out.finish_reason, "token_ids_sha256": sha256_text(json.dumps(list(out.token_ids)))})
            if r["arm"] == "B1" and r["sample"] == -1:
                b1[r["item_id"]] = r["token_ids_sha256"]
            rows.append(r)
            texts.append({**{k2: r[k2] for k2 in ("item_id", "arm", "grounding", "decode", "sample")}, "text": out.text,
                          "finish_reason": out.finish_reason})
    zero = [r for r in rows if r["arm"] == "zero-step"]
    zero_ok = sum(r["token_ids_sha256"] == b1[r["item_id"]] for r in zero) / len(zero)
    for name, rr in ((f"rows_{c}.jsonl", rows), (f"texts_{c}.jsonl", texts)):
        with open(d / "chunks" / name, "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (d / "chunks" / f"serve_{c}.json").write_text(json.dumps({"zero_step_equals_B1": zero_ok, "n_requests": len(reqs),
                                                              "n_rows": len(rows), "n_adapters": len(lora_ids), "wall_s": wall,
                                                              "scorer_guarded": scorer.GUARDED}))
    if zero_ok < 1.0:
        raise SystemExit(f"chunk {c}: zero-step != B1 on {1 - zero_ok:.1%} of items")


def stage_clean(args, B, c, d):
    kept = []
    for i in B["chunks"][c]:
        for g in GROUNDINGS:
            p = d / "adapters" / f"{g}_{i}"
            if unit("ttt-keep-v1", p.name) < KEEP_FRACTION:
                kept.append(p.name)
            elif p.exists():
                shutil.rmtree(p)
    (d / "chunks" / f"clean_{c}.json").write_text(json.dumps({"kept": kept}))


def stage_finalize(args, B, d, cfg, run_id):
    from transformers import AutoTokenizer
    import transformers
    n = len(B["chunks"])
    for name in ("updates", "rows", "texts"):
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for c in range(n):
                f.write((d / "chunks" / f"{name}_{c}.jsonl").read_text(encoding="utf-8"))
    tr = [json.loads((d / "chunks" / f"train_{c}.json").read_text()) for c in range(n)]
    sv = [json.loads((d / "chunks" / f"serve_{c}.json").read_text()) for c in range(n)]
    cl = [json.loads((d / "chunks" / f"clean_{c}.json").read_text()) for c in range(n)]
    ups = jl(d / "updates.jsonl")
    applied = [u for u in ups if u.get("applied") and u.get("N_after") is not None]
    metrics = {"model": args.model, "n_items": sum(len(ch) for ch in B["chunks"]), "chunks": n,
               "reset_logits_exact": all(t["reset_logits_exact"] for t in tr),
               "zero_step_equals_B1": min(s["zero_step_equals_B1"] for s in sv),
               "realized_norm_max_rel_err": max(abs(u["N_after"] - u["tau"]) / u["tau"] for u in applied),
               "anomalies": {a: sum(u.get("anomaly") == a for u in ups) for a in sorted({u.get("anomaly") for u in ups} - {None})},
               "adapters_kept": sum(len(x["kept"]) for x in cl), "serve_wall_s": sum(s["wall_s"] for s in sv),
               "scorer_guarded": [g for s in sv for g in s["scorer_guarded"]], "transformers": transformers.__version__}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers+peft / vllm", "version": transformers.__version__},
                   {"context": R.CONTEXT_MEMORY, "selfpred": SELFPRED})
    register(run_id, "done", block="ttt-test", exp="exp5", model=args.model)
    print(json.dumps(metrics, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--verbal", required=True)
    ap.add_argument("--self-threshold-s", type=float, required=True)
    ap.add_argument("--stage", required=True, choices=["train", "serve", "clean", "finalize"])
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--n-items", type=int, default=400)
    ap.add_argument("--max-loras", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: first N pool, spillover and KL items")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    B = build(args)
    cfg = {k: v for k, v in vars(args).items() if k not in ("stage", "chunk_index", "dry_run")}
    cfg.update({"groundings": GROUNDINGS, "n_kl": N_KL, "n_samples": N_SAMPLES, "t_sample": T_SAMPLE,
                "max_tokens": MAX_TOKENS, "selfpred_tokens": SELFPRED_TOKENS, "keep_fraction": KEEP_FRACTION,
                "steps": STEPS, "tier2": json.loads(TIER2.read_text())["exp5"][args.model],
                "context_template_sha256": sha256_text(R.CONTEXT_MEMORY), "selfpred_sha256": sha256_text(SELFPRED)})
    if args.dry_run:
        V = B["verdicts"]
        print(json.dumps({"items": sum(len(ch) for ch in B["chunks"]), "chunks": len(B["chunks"]),
                          "spillover": len(B["spill"]), "kl": len(B["kl"]),
                          "initially_wrong": sum(not B["init"][i]["correct"] for i in V),
                          "self_says_correct": sum(v["self"] for v in V.values()),
                          "perm0_equals_self": sum(v["perm0"] == v["self"] for v in V.values())}, indent=1))
        return
    run_id, d = run_dir("main" if not args.limit else "p2", args.model, "ttt-test" + ("-smoke" if args.limit else ""))
    (d / "chunks").mkdir(exist_ok=True)
    marker = d / "chunks" / (f"{args.stage}_{args.chunk_index}.json" if args.stage != "finalize" else "finalize.json")
    if args.stage == "finalize" and (d / "manifest.json").exists():
        print("finalize: already done")
        return
    if marker.exists() and args.stage != "finalize":
        print(f"{args.stage} chunk {args.chunk_index}: already done")
        return
    if args.stage == "train" and args.chunk_index == 0:
        register(run_id, "running", block="ttt-test", exp="exp5", model=args.model,
                 config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    t0 = time.time()
    if args.stage == "train":
        stage_train(args, B, args.chunk_index, d)
    elif args.stage == "serve":
        stage_serve(args, B, args.chunk_index, d)
    elif args.stage == "clean":
        stage_clean(args, B, args.chunk_index, d)
    else:
        stage_finalize(args, B, d, cfg, run_id)
    print(json.dumps({"run_id": run_id, "stage": args.stage, "chunk": args.chunk_index, "wall_s": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
