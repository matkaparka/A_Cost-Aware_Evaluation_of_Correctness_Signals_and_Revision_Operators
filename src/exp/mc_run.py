"""Independent validation block on MMLU-Pro (docs/planning/validation_v2/plan.md §2.2; DEVIATIONS D18; addendum
docs/planning/validation_v2/addendum.md, frozen before any test output of this block).

Stages, each its own process at the locked commit (models 4B and T8B):
  initial  a0 for every item of --split: vLLM greedy, VLLM_BATCH_INVARIANT=1, A0_TOKENS, the main study's initial
           prompt around the MMLU-Pro question (src/data/mmlupro.py render_question), scored by src/score/mc.py.
  consist  K sampled initial answers per item (T, top-p, top-k as src/gen/h2a_extra.py; seeds from the named stream
           "mc-consistency"); agreement = share of samples whose letter equals a0's letter. Sample texts are not kept.
  hf       for the dev and test rows of initial runs: the fixed verbal readout W0 (and, with --variant, one more
           variant of src/score/verbal_variants.py), activations at the last a0 content token (all layers), a probe
           selected on dev by the src/probe/train.py grid (StratifiedGroupKFold grouped by canonical stem) and applied
           to test, and the model's frozen math probe applied unchanged. acts.npz stays on the run host.
  conds    on one --split: B0 (keep a0), F0, F1, F2 and F2-scratch (two turns), F3(1.0) and F3(0.5) (verdicts from the
           frozen f3_verdict stream), with the main runner's wrappers, request plan and greedy generation.
Usage: VLLM_BATCH_INVARIANT=1 python -m src.exp.mc_run initial --model 4B --split dev
       VLLM_BATCH_INVARIANT=1 python -m src.exp.mc_run consist --model 4B --initial <dir>
       python -m src.exp.mc_run hf --model 4B --initial <dev dir> <test dir> [--variant W2]
       VLLM_BATCH_INVARIANT=1 python -m src.exp.mc_run conds --model 4B --split dev --initial <dir>
"""
import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..data import mmlupro as M
from ..data.verifiers import f3_verdict
from ..gen import render as R
from ..runlib import MODELS, REPO, needs_bos, register, require_batch_invariant, run_dir, sha256_file, sha256_text, write_manifest
from ..score import mc as S
from .runner import Plan, generate

A0_TOKENS, REVISION_TOKENS, CRITIQUE_TOKENS = 4096, 2048, 1024
CONDS_P = (1.0, 0.5)
K, TEMPERATURE, TOP_P, TOP_K = 5, 0.7, 0.8, 20
MATH_PROBES = {"4B": "results/p3/4B/p3-probe-dev-ext-4B-b7cf0d6/probe.npz",
               "T8B": "results/fu2/p2/T8B/p2-probe-dev-T8B-1809070/probe.npz"}
EXP = {"dev": "fu-dev", "test": "fu"}


def jl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def items():
    return {it["item_id"]: it for it in M.load_items()}


def test_guard(rows, args):
    if any(r["split"] == "test" for r in rows) and not args.allow_test:
        raise SystemExit("test rows need --allow-test (the validation addendum must be frozen)")


def finish(d, run_id, args, tok, cfg, backend, block):
    cfg = {**cfg, "items_sha256": sha256_file(M.OUT / "items.jsonl"), "mc_scorer": S.MC_SCORER_VERSION,
           "mc_scorer_sha256": sha256_file(REPO / "src" / "score" / "mc.py")}
    write_manifest(d, run_id, args.model, tok, cfg, backend, {"initial": R.PROMPT, "ask": M.ASK, "query": R.VERIFY_QUERY})
    register(run_id, "done", block=block, exp="mc", model=args.model)


def llm_for(args, max_len):
    require_batch_invariant()
    from vllm import LLM
    return LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=max_len)


# ---------------------------------------------------------------- initial
def stage_initial(args):
    import vllm
    from vllm import SamplingParams
    its = items()
    sel = sorted((it for it in its.values() if it["split"] == args.split), key=lambda it: it["order"])
    sel = sel[:args.limit] if args.limit else sel
    cfg = {**vars(args), "a0_tokens": A0_TOKENS}
    run_id, d = run_dir(EXP[args.split], args.model, f"mc-initial-{args.split}{args.tag}")
    register(run_id, "running", block="mc-initial", exp="mc", model=args.model, config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    llm = llm_for(args, 8192)
    tok = llm.get_tokenizer()
    prompts = [R.fresh_prompt(tok, M.render_question(it)).text for it in sel]
    outs = llm.generate(prompts, [SamplingParams(temperature=0.0, max_tokens=A0_TOKENS, seed=0)] * len(prompts), use_tqdm=False)
    counts = Counter()
    with open(d / "initial.jsonl", "w", encoding="utf-8") as f:
        for it, o in zip(sel, outs):
            c = o.outputs[0]
            sc = S.score(c.text, it, c.finish_reason)
            row = {"item_id": it["item_id"], "canonical_id": it["canonical_id"], "category": it["category"], "split": it["split"],
                   "a0": c.text, "a0_token_ids": list(c.token_ids), "finish_reason": c.finish_reason, "out_tokens": len(c.token_ids),
                   "prompt_tokens": len(o.prompt_token_ids), **sc}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["n"] += 1
            counts["right"] += sc["correct"]
            counts[sc["status"]] += 1
    metrics = {"gen_s": time.time() - t0, "vllm": vllm.__version__, "counts": dict(counts)}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    finish(d, run_id, args, tok, cfg, {"name": "vllm", "version": vllm.__version__}, "mc-initial")
    print(json.dumps({"run_id": run_id, "dir": str(d), **metrics["counts"]}))


# ---------------------------------------------------------------- consist
def seed_for(item_id):
    return int(hashlib.sha256(f"mc-consistency|{item_id}".encode()).hexdigest(), 16) % (2 ** 31)


def stage_consist(args):
    import vllm
    from vllm import SamplingParams
    its = items()
    rows = [r for d_ in args.initial for r in jl(Path(d_) / "initial.jsonl")]
    rows = rows[:args.limit] if args.limit else rows
    test_guard(rows, args)
    split = rows[0]["split"]
    cfg = {**vars(args), "k": K, "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "seed_stream": "mc-consistency"}
    run_id, d = run_dir(EXP[split], args.model, f"mc-consist-{split}{args.tag}")
    register(run_id, "running", block="mc-consist", exp="mc", model=args.model, config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    llm = llm_for(args, 8192)
    tok = llm.get_tokenizer()
    prompts = [R.fresh_prompt(tok, M.render_question(its[r["item_id"]])).text for r in rows]
    params = [SamplingParams(n=K, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=A0_TOKENS, seed=seed_for(r["item_id"]))
              for r in rows]
    outs = llm.generate(prompts, params, use_tqdm=False)
    with open(d / "consistency.jsonl", "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            it = its[r["item_id"]]
            samples = [{**S.score(c.text, it, c.finish_reason), "out_tokens": len(c.token_ids), "finish_reason": c.finish_reason}
                       for c in o.outputs]
            agree = [r["pred"] is not None and s["pred"] == r["pred"] for s in samples]
            f.write(json.dumps({"item_id": r["item_id"], "split": r["split"], "k": K, "a0_pred": r["pred"], "a0_correct": r["correct"],
                                "agreement": sum(agree) / K, "sample_acc": sum(s["correct"] for s in samples) / K,
                                "sample_out_tokens": sum(s["out_tokens"] for s in samples), "samples": samples}) + "\n")
    (d / "metrics.json").write_text(json.dumps({"n": len(rows), "gen_s": time.time() - t0, "vllm": vllm.__version__}, indent=1))
    finish(d, run_id, args, tok, cfg, {"name": "vllm", "version": vllm.__version__}, "mc-consist")
    print(json.dumps({"run_id": run_id, "dir": str(d), "n": len(rows)}))


# ---------------------------------------------------------------- hf
def end_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    return ids


def stage_hf(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..probe.train import C_GRID, cv_auroc, score_frozen
    from ..score.verbal_variants import VARIANTS, readout
    its = items()
    rows = [r for d_ in args.initial for r in jl(Path(d_) / "initial.jsonl")]
    rows = rows[:args.limit] if args.limit else rows
    test_guard(rows, args)
    math_probe = REPO / MATH_PROBES[args.model]
    cfg = {**vars(args), "C_grid": C_GRID, "groups": "canonical stem", "position": "last a0 content token",
           "math_probe": MATH_PROBES[args.model], "math_probe_sha256": sha256_file(math_probe)}
    run_id, d = run_dir("fu" if any(r["split"] == "test" for r in rows) else "fu-dev", args.model, f"mc-hf{args.tag}")
    register(run_id, "running", block="mc-hf", exp="mc", model=args.model, config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    t0 = time.time()
    qs = [M.render_question(its[r["item_id"]]) for r in rows]
    a0s = [r["a0"] for r in rows]
    variants = ["W0"] + ([args.variant] if args.variant and args.variant != "W0" else [])
    vb = {v: readout(tok, model, qs, a0s, *VARIANTS[v], batch=args.verbal_batch) for v in variants}
    end = end_ids(tok)
    X = []
    with torch.no_grad():
        for r, q in zip(rows, qs):
            prefix = tok.apply_chat_template([{"role": "user", "content": R.PROMPT.format(q=q)}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            p_ids = tok(prefix, add_special_tokens=needs_bos(tok, prefix)).input_ids
            hs = model(input_ids=torch.tensor([p_ids + r["a0_token_ids"]], device="cuda"), output_hidden_states=True).hidden_states
            last = len(p_ids) + len(r["a0_token_ids"]) - 1
            if r["a0_token_ids"] and r["a0_token_ids"][-1] in end:
                last -= 1
            X.append(torch.stack([h[0, last] for h in hs]).to(torch.float16).cpu().numpy())
    gpu_s = time.time() - t0
    X = np.stack(X)
    ids = np.array([r["item_id"] for r in rows])
    split = np.array([r["split"] for r in rows])
    y = np.array([bool(r["correct"]) for r in rows])
    np.savez_compressed(d / "acts.npz", X=X, ids=ids, split=split, correct0=y)
    with open(d / "verbal.jsonl", "w", encoding="utf-8") as f:
        for k, r in enumerate(rows):
            f.write(json.dumps({"item_id": r["item_id"], "split": r["split"], **{v: vb[v][k] for v in variants}}) + "\n")
    p_math = score_frozen(math_probe, X)
    dev = split == "dev"
    groups = np.array([its[i]["canonical_id"] for i in ids])
    Xd = X[dev].astype(np.float32)
    grid = [{"layer": k, "C": C, "cv_auroc": float(cv_auroc(Xd[:, k], y[dev], groups[dev], C))}
            for k in range(X.shape[1]) for C in C_GRID] if dev.any() and 0 < y[dev].sum() < dev.sum() else []
    p_task = np.full(len(rows), np.nan)
    sel = None
    if grid:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sel = max(grid, key=lambda g: (g["cv_auroc"], -g["layer"]))
        sc = StandardScaler().fit(Xd[:, sel["layer"]])
        clf = LogisticRegression(C=sel["C"], max_iter=2000).fit(sc.transform(Xd[:, sel["layer"]]), y[dev])
        p_task = clf.predict_proba(sc.transform(X[:, sel["layer"]].astype(np.float32)))[:, 1]
    with open(d / "probe_scores.jsonl", "w", encoding="utf-8") as f:
        for i, s_, a, b in zip(ids.tolist(), split.tolist(), p_math.tolist(), p_task.tolist()):
            f.write(json.dumps({"item_id": i, "split": s_, "p_math_probe": a, "p_task_probe": None if math.isnan(b) else b}) + "\n")
    (d / "probe_selection.json").write_text(json.dumps({"n_dev": int(dev.sum()), "n_dev_wrong": int((~y[dev]).sum()),
                                                        "selected": sel, "grid": grid}, indent=1))
    (d / "metrics.json").write_text(json.dumps({"n": len(rows), "gpu_s": gpu_s, "analysis_s": time.time() - t0 - gpu_s,
                                                "acts_sha256": sha256_file(d / "acts.npz")}, indent=1))
    finish(d, run_id, args, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__}, "mc-hf")
    print(json.dumps({"run_id": run_id, "dir": str(d), "n": len(rows), "selected": sel}))


# ---------------------------------------------------------------- conds
def stage_conds(args):
    its = items()
    rows = sorted((r for d_ in args.initial for r in jl(Path(d_) / "initial.jsonl") if r["split"] == args.split),
                  key=lambda r: its[r["item_id"]]["order"])
    rows = rows[:args.limit] if args.limit else rows
    cfg = {**vars(args), "revision_tokens": REVISION_TOKENS, "critique_tokens": CRITIQUE_TOKENS, "p_levels": CONDS_P}
    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    else:
        import vllm
        from vllm import SamplingParams
        llm = llm_for(args, 12288)
        tok = llm.get_tokenizer()
    plan = Plan()
    q = {r["item_id"]: M.render_question(its[r["item_id"]]) for r in rows}
    for r in rows:
        base = {"item_id": r["item_id"], "category": r["category"], "data": "mc", "correct0": r["correct"]}
        qq, a0 = q[r["item_id"]], r["a0"]
        plan.rows.append({**base, "exp": "1A", "cond": "B0", "prompt_key": None, "fixed_correct": r["correct"]})
        plan.add({**base, "exp": "1A", "cond": "F0"}, R.review_prompt(tok, qq, a0), REVISION_TOKENS)
        plan.add({**base, "exp": "1A", "cond": "F1"}, R.review_prompt(tok, qq, a0, prefill=R.TRIGGER_PREFILL), REVISION_TOKENS)
        for cond, turn in (("F2", R.CRITIQUE_TURN), ("F2-scratch", R.SCRATCH_TURN)):
            plan.add({**base, "exp": "1A", "cond": f"{cond}:turn1", "stage": 1}, R.review_prompt(tok, qq, a0, instruction=turn),
                     CRITIQUE_TOKENS)
        for p in CONDS_P:
            v = f3_verdict(r["item_id"], r["correct"], p)
            plan.add({**base, "exp": "1A", "cond": f"F3({p})", "verdict": v}, R.review_prompt(tok, qq, a0, feedback=R.VERDICT[v]),
                     REVISION_TOKENS)
    if args.dry_run:
        print(json.dumps({"rows": len(plan.rows), "unique prompts": len(plan.prompts),
                          "conditions": Counter(r["cond"] for r in plan.rows)}, indent=1))
        return
    run_id, d = run_dir(EXP[args.split], args.model, f"mc-conds-{args.split}{args.tag}")
    register(run_id, "running", block="mc-conds", exp="mc", model=args.model, config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    out1 = generate(llm, plan, SamplingParams)
    plan2 = Plan()
    a0 = {r["item_id"]: r["a0"] for r in rows}
    for row in plan.rows:
        if row.get("cond") in ("F2:turn1", "F2-scratch:turn1"):
            first = R.CRITIQUE_TURN if row["cond"] == "F2:turn1" else R.SCRATCH_TURN
            plan2.add({k: row[k] for k in ("item_id", "category", "data", "correct0")} | {"exp": "1A", "cond": row["cond"].split(":")[0],
                                                                                          "turn1_key": row["prompt_key"]},
                      R.review_prompt(tok, q[row["item_id"]], a0[row["item_id"]], history=[(first, out1[row["prompt_key"]]["text"])]),
                      REVISION_TOKENS)
    t1 = time.time()
    out2 = generate(llm, plan2, SamplingParams)
    t2 = time.time()
    with open(d / "generations.jsonl", "w", encoding="utf-8") as f:
        for k, o in {**out1, **out2}.items():
            f.write(json.dumps({"prompt_key": k, **o}, ensure_ascii=False) + "\n")
    stats = defaultdict(Counter)
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for pl, outs in ((plan, out1), (plan2, out2)):
            for row in pl.rows:
                if row.get("stage") == 1:
                    continue
                if row["prompt_key"] is None:
                    row.update(correct=row["fixed_correct"], status="fixed")
                else:
                    o = outs[row["prompt_key"]]
                    row.update({"out_tokens": o["out_tokens"], "prompt_tokens": o["prompt_tokens"], "finish_reason": o["finish_reason"],
                                **S.score(o["text"], its[row["item_id"]], o["finish_reason"])})
                    if row.get("turn1_key"):   # count the first turn too (second review: full cost)
                        t1o = out1[row["turn1_key"]]
                        row.update({"turn1_out_tokens": t1o["out_tokens"], "turn1_prompt_tokens": t1o["prompt_tokens"]})
                stats[row["cond"]]["n"] += 1
                stats[row["cond"]]["truncated"] += row.get("finish_reason") == "length"
                stats[row["cond"]]["not ok"] += row["status"] not in ("ok", "fixed", "fallback")
                f.write(json.dumps(row) + "\n")
    metrics = {"vllm": vllm.__version__, "stage1_s": t1 - t0, "stage2_s": t2 - t1, "stage1_prompts": len(plan.prompts),
               "stage2_prompts": len(plan2.prompts), "rows": {k: dict(v) for k, v in sorted(stats.items())}}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    finish(d, run_id, args, tok, cfg, {"name": "vllm", "version": vllm.__version__}, "mc-conds")
    print(json.dumps({"run_id": run_id, "dir": str(d), **{k: metrics[k] for k in ("stage1_prompts", "stage2_prompts", "stage1_s", "stage2_s")}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["initial", "consist", "hf", "conds"])
    ap.add_argument("--model", required=True, choices=["4B", "T8B", "1.7B"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--initial", nargs="+", help="initial run directories")
    ap.add_argument("--variant", help="hf: one more readout variant (src/score/verbal_variants.py)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbal-batch", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("--split test needs --allow-test (the validation addendum must be frozen)")
    args.tag = f"-{args.tag}" if args.tag else ""
    {"initial": stage_initial, "consist": stage_consist, "hf": stage_hf, "conds": stage_conds}[args.stage](args)


if __name__ == "__main__":
    main()
