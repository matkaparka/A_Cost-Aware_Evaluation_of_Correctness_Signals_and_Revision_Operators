"""Lore QA on the run host (docs/planning/lore_qa/proposal.md §6; prereg_addendum.md). Exploratory, estimation only.

Three stages, each its own process, all at the locked commit:
  initial  a0 for every lore item (tracks A and B) of --split at each of --budgets: vLLM greedy,
           VLLM_BATCH_INVARIANT=1, 4096 tokens, the main study's initial prompt around the lore question
           (src/data/lore.py render_question), scored with src/score/lore.py (primary and second scorer). The dev
           rows feed the vLLM re-measurement of gates G2-G4 (analysis/lore_gates.py); test rows are pre-feedback
           quantities and only their counts are printed.
  hf       at one --budget, for dev and test rows of an initial run: verbal P(yes) (design §5.7, as
           src/exp/lore_pilot.py), the Exp 2 activations (all layers, last a0 content token, as src/probe/cache.py),
           Exp 2(a) the frozen 4B math probe applied unchanged, and Exp 2(b) a probe selected on lore dev by the
           src/probe/train.py rule (grid layers x C_GRID, 5-fold StratifiedGroupKFold grouped by cluster), refit on
           dev and applied to test. -> verbal.jsonl, probe_scores.jsonl, probe_selection.json; acts.npz stays on the
           run host.
  conds    at one --budget for one --split: Exp 1A core on D-nat-lore (B0, B1, F0, F1, F2, F2-scratch, F3(p) for
           p in {0.5, 0.7, 0.9, 1.0}, F5, F3-perm K = 3) and Exp 1B (role x label, F0) on D-nat-lore and D-inj-lore,
           with the main runner's request plan, wrappers and generation (src/exp/runner.py). F5's threshold t* is
           chosen on lore dev verbal scores by the main study's rule (analysis/p2_selfverdict.py: the midpoint that
           maximizes balanced accuracy); the F3-perm strata are route x initial correctness.
Usage: VLLM_BATCH_INVARIANT=1 python -m src.exp.lore_run initial --model 4B --split dev --budgets 512 1024 2048
       python -m src.exp.lore_run hf --model 4B --initial <dev initial dir> <test initial dir> --budget 512
       VLLM_BATCH_INVARIANT=1 python -m src.exp.lore_run conds --model 4B --split test --allow-test --budget 512 \
           --initial <test initial dir> --verbal <hf dir>
"""
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..data import lore, lore_b
from ..data.lore import OUT, render_question
from ..data.verifiers import f3_verdict, permute_self_verdicts, self_verdict
from ..gen import render as R
from ..runlib import MODELS, REPO, register, require_batch_invariant, run_dir, sha256_file, sha256_text, write_manifest
from ..score import lore as S
from .runner import P_LEVELS, Plan, generate

A0_TOKENS, REVISION_TOKENS, CRITIQUE_TOKENS = 4096, 2048, 1024
CELLS_1B = [(role, label) for role in ("assistant", "user") for label in ("self", "other")]
MATH_PROBE = REPO / "results" / "p3" / "4B" / "p3-probe-dev-ext-4B-b7cf0d6" / "probe.npz"
MATH_PROBE_SHA = "a62621d9ef55fed97f3de6a16b269715fe402b2e882c4bcae7b436172c8bc833"


def jl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def items_and_passages():
    passages, items_a = lore.load_built()
    items = {it["id"]: dict(it, route="A") for it in items_a}
    items.update({it["id"]: it for it in lore_b.load_items_b()})
    return items, passages


def manifest_extra():
    return {k: sha256_file(OUT / f) for k, f in (("items_a", "items_a.jsonl"), ("items_b", "items_b.jsonl"),
                                                   ("items_inj", "items_inj.jsonl"), ("passages", "passages.jsonl"))}


def finish(d, run_id, args, tok, cfg, backend, block):
    cfg = {**cfg, "lore_data_sha256": manifest_extra(), "lore_scorer": S.LORE_SCORER_VERSION,
           "lore_scorer_sha256": sha256_file(REPO / "src" / "score" / "lore.py")}
    write_manifest(d, run_id, args.model, tok, cfg, backend, {"initial": R.PROMPT, "query": R.VERIFY_QUERY})
    register(run_id, "done", block=block, exp="lore", model=args.model)


def score_row(text, it, finish_reason):
    s1, s2 = S.score(text, it, finish_reason), S.score_contains(text, it, finish_reason)
    return {"status": s1["status"], "pred": s1["pred"], "correct": s1["correct"], "correct_second": s2["correct"]}


# ---------------------------------------------------------------- initial
def stage_initial(args):
    require_batch_invariant()
    import vllm
    from vllm import LLM, SamplingParams
    items, passages = items_and_passages()
    sel = sorted((it for it in items.values() if args.split == "all" or it["split"] == args.split), key=lambda it: it["order"])
    sel = sel[:args.limit] if args.limit else sel
    cfg = {**vars(args), "a0_tokens": A0_TOKENS}
    run_id, d = run_dir("fu" if args.split != "dev" else "fu-dev", args.model, f"lore-initial-{args.split}{args.tag}")
    register(run_id, "running", block="lore-initial", exp="lore", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=8192)
    tok = llm.get_tokenizer()
    reqs = [(it, b, R.fresh_prompt(tok, render_question(it, b, passages))) for b in args.budgets for it in sel]
    outs = llm.generate([r[2].text for r in reqs], [SamplingParams(temperature=0.0, max_tokens=A0_TOKENS, seed=0)] * len(reqs),
                        use_tqdm=False)
    gen_s = time.time() - t0
    counts = defaultdict(Counter)
    with open(d / "initial.jsonl", "w", encoding="utf-8") as f:
        for (it, b, _), o in zip(reqs, outs):
            c = o.outputs[0]
            row = {"item_id": it["id"], "route": it["route"], "template": it["template"], "answer_kind": it["answer_kind"],
                   "answer_class": it["answer_class"], "cluster": it["cluster"], "split": it["split"], "budget": b,
                   "a0": c.text, "a0_token_ids": list(c.token_ids), "finish_reason": c.finish_reason,
                   "out_tokens": len(c.token_ids), "prompt_tokens": len(o.prompt_token_ids), **score_row(c.text, it, c.finish_reason)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            k = f"{it['split']}/{b}"
            counts[k]["n"] += 1
            counts[k]["right"] += row["correct"]
            counts[k][row["status"]] += 1
    metrics = {"gen_s": gen_s, "vllm": vllm.__version__, "counts": {k: dict(v) for k, v in sorted(counts.items())}}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    finish(d, run_id, args, tok, cfg, {"name": "vllm", "version": vllm.__version__}, "lore-initial")
    print(json.dumps({"run_id": run_id, "dir": str(d), **metrics["counts"]}))


# ---------------------------------------------------------------- hf: verbal, activations, Exp 2
def stage_hf(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..probe.train import C_GRID, cv_auroc, score_frozen
    from .lore_pilot import verbal
    if sha256_file(MATH_PROBE) != MATH_PROBE_SHA:
        raise SystemExit("the frozen 4B math probe differs from the one the main analysis used")
    items, passages = items_and_passages()
    rows = [r for d_ in args.initial for r in jl(Path(d_) / "initial.jsonl") if r["budget"] == args.budget]
    rows = rows[:args.limit] if args.limit else rows
    cfg = {**vars(args), "math_probe_sha256": MATH_PROBE_SHA, "C_grid": C_GRID, "groups": "cluster",
           "position": "last a0 content token", "verbal_query": R.VERIFY_QUERY}
    run_id, d = run_dir("fu", args.model, f"lore-hf-{args.budget}{args.tag}")
    register(run_id, "running", block="lore-hf", exp="lore", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    t0 = time.time()
    qs = [render_question(items[r["item_id"]], args.budget, passages) for r in rows]
    vb = verbal(tok, model, qs, [r["a0"] for r in rows], batch=args.verbal_batch)
    end = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    X = []
    with torch.no_grad():
        for r, q in zip(rows, qs):
            prefix = tok.apply_chat_template([{"role": "user", "content": R.PROMPT.format(q=q)}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            p_ids = tok(prefix, add_special_tokens=False).input_ids
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
        for r, v in zip(rows, vb):
            f.write(json.dumps({"item_id": r["item_id"], "split": r["split"], "budget": args.budget, **v}) + "\n")
    # Exp 2(a): the frozen math probe, unchanged
    p_math = score_frozen(MATH_PROBE, X)
    # Exp 2(b): a lore probe selected on dev only
    dev = split == "dev"
    groups = np.array([items[i]["cluster"] for i in ids])
    Xd = X[dev].astype(np.float32)
    grid = [{"layer": k, "C": C, "cv_auroc": float(cv_auroc(Xd[:, k], y[dev], groups[dev], C))}
            for k in range(X.shape[1]) for C in C_GRID] if dev.any() and 0 < y[dev].sum() < dev.sum() else []
    p_lore = np.full(len(rows), np.nan)
    sel = None
    if grid:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sel = max(grid, key=lambda g: (g["cv_auroc"], -g["layer"]))
        sc = StandardScaler().fit(Xd[:, sel["layer"]])
        clf = LogisticRegression(C=sel["C"], max_iter=2000).fit(sc.transform(Xd[:, sel["layer"]]), y[dev])
        p_lore = clf.predict_proba(sc.transform(X[:, sel["layer"]].astype(np.float32)))[:, 1]
    with open(d / "probe_scores.jsonl", "w", encoding="utf-8") as f:
        for i, s_, a, b in zip(ids.tolist(), split.tolist(), p_math.tolist(), p_lore.tolist()):
            f.write(json.dumps({"item_id": i, "split": s_, "p_math_probe": a, "p_lore_probe": None if math.isnan(b) else b}) + "\n")
    (d / "probe_selection.json").write_text(json.dumps({"n_dev": int(dev.sum()), "n_dev_wrong": int((~y[dev]).sum()),
                                                        "selected": sel, "grid": grid}, indent=1))
    (d / "metrics.json").write_text(json.dumps({"n": len(rows), "gpu_s": gpu_s, "analysis_s": time.time() - t0 - gpu_s,
                                                "acts_sha256": sha256_file(d / "acts.npz")}, indent=1))
    finish(d, run_id, args, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__}, "lore-hf")
    print(json.dumps({"run_id": run_id, "dir": str(d), "n": len(rows), "selected": sel}))


# ---------------------------------------------------------------- conds
def t_star(verbal_rows, initial_rows):
    """The main study's rule on lore dev: the midpoint between adjacent distinct scores that maximizes balanced
    accuracy (analysis/p2_selfverdict.py)."""
    corr = {r["item_id"]: bool(r["correct"]) for r in initial_rows}
    sc = [(r["s"], corr[r["item_id"]]) for r in verbal_rows if r["split"] == "dev" and r["s"] is not None]
    s, y = np.array([a for a, _ in sc]), np.array([b for _, b in sc])
    u = np.unique(s)
    cands = (u[:-1] + u[1:]) / 2
    ba = [((s >= t) & y).sum() / max(1, y.sum()) / 2 + ((s < t) & ~y).sum() / max(1, (~y).sum()) / 2 for t in cands]
    return float(cands[int(np.argmax(ba))]), float(max(ba))


def stage_conds(args):
    items, passages = items_and_passages()
    init_rows = [r for d_ in args.initial for r in jl(Path(d_) / "initial.jsonl") if r["budget"] == args.budget]
    verbal = jl(Path(args.verbal) / "verbal.jsonl")
    dev_init = [r for r in jl(Path(args.dev_initial) / "initial.jsonl") if r["budget"] == args.budget]
    ts, ba = t_star(verbal, dev_init)
    vby = {r["item_id"]: r for r in verbal}
    dnat = sorted((r for r in init_rows if r["split"] == args.split), key=lambda r: items[r["item_id"]]["order"])
    inj = sorted((r for r in jl(OUT / "items_inj.jsonl") if r["budget"] == args.budget and r["split"] == args.split),
                 key=lambda r: r["order"])
    if args.limit:
        dnat, inj = dnat[:args.limit], inj[:args.limit]
    cfg = {**vars(args), "t_star_s": ts, "t_star_balanced_acc_dev": ba, "revision_tokens": REVISION_TOKENS,
           "critique_tokens": CRITIQUE_TOKENS, "p_levels": P_LEVELS}
    self_v = {r["item_id"]: self_verdict(vby.get(r["item_id"], {}), ts) for r in dnat}
    perm = permute_self_verdicts([{"item_id": r["item_id"], "dataset": r["route"], "initially_correct": r["correct"],
                                   "self_says_correct": self_v[r["item_id"]]} for r in dnat])
    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    else:
        require_batch_invariant()
        import vllm
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=12288)
        tok = llm.get_tokenizer()
    plan = Plan()
    q = {r["item_id"]: render_question(items[r["item_id"]], args.budget, passages) for r in dnat}
    for r in dnat:
        base = {"item_id": r["item_id"], "route": r["route"], "answer_class": r["answer_class"], "data": "dnat",
                "correct0": r["correct"]}
        qq, a0 = q[r["item_id"]], r["a0"]
        plan.rows.append({**base, "exp": "1A", "cond": "B0", "prompt_key": None, "fixed_correct": r["correct"]})
        plan.add({**base, "exp": "1A", "cond": "B1"}, R.fresh_prompt(tok, qq), REVISION_TOKENS)
        plan.add({**base, "exp": "1A", "cond": "F0"}, R.review_prompt(tok, qq, a0), REVISION_TOKENS)
        plan.add({**base, "exp": "1A", "cond": "F1"}, R.review_prompt(tok, qq, a0, prefill=R.TRIGGER_PREFILL), REVISION_TOKENS)
        for cond, turn in (("F2", R.CRITIQUE_TURN), ("F2-scratch", R.SCRATCH_TURN)):
            plan.add({**base, "exp": "1A", "cond": f"{cond}:turn1", "stage": 1}, R.review_prompt(tok, qq, a0, instruction=turn),
                     CRITIQUE_TOKENS)
        for p in P_LEVELS:
            v = f3_verdict(r["item_id"], r["correct"], p)
            plan.add({**base, "exp": "1A", "cond": f"F3({p})", "verdict": v},
                     R.review_prompt(tok, qq, a0, feedback=R.VERDICT[v]), REVISION_TOKENS)
        v = self_v[r["item_id"]]
        plan.add({**base, "exp": "1A", "cond": "F5", "verdict": v}, R.review_prompt(tok, qq, a0, feedback=R.VERDICT[v]), REVISION_TOKENS)
        for k, v in enumerate(perm[r["item_id"]]):
            plan.add({**base, "exp": "1A", "cond": f"F3-perm[{k}]", "verdict": v},
                     R.review_prompt(tok, qq, a0, feedback=R.VERDICT[v]), REVISION_TOKENS)
        for role, label in CELLS_1B:
            plan.add({**base, "exp": "1B", "cond": f"{role}/{label}/F0"},
                     R.review_prompt(tok, qq, a0, role=role, label=label), REVISION_TOKENS)
    for r in inj:
        it = items[r["base_id"]]
        base = {"item_id": it["id"], "inj_id": r["id"], "route": r["route"], "answer_class": it["answer_class"],
                "data": "dinj", "variant": r["variant"], "correct0": r["variant"] == "right"}
        qq = render_question(it, args.budget, passages)
        for role, label in CELLS_1B:
            plan.add({**base, "exp": "1B", "cond": f"{role}/{label}/F0"},
                     R.review_prompt(tok, qq, r["a0"], role=role, label=label), REVISION_TOKENS)
    if args.dry_run:
        print(json.dumps({"dnat": len(dnat), "dinj": len(inj), "rows": len(plan.rows), "unique prompts": len(plan.prompts),
                          "t_star_s": ts, "conditions": Counter(f"{r['exp']}:{r['data']}:{r['cond']}" for r in plan.rows)}, indent=1))
        return
    tag = args.tag
    run_id, d = run_dir("fu" if args.split == "test" else "fu-dev", args.model, f"lore-conds-{args.split}-{args.budget}{tag}")
    register(run_id, "running", block="lore-conds", exp="lore", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    out1 = generate(llm, plan, SamplingParams)
    plan2 = Plan()
    a0 = {r["item_id"]: r["a0"] for r in dnat}
    for row in plan.rows:
        if row.get("cond") in ("F2:turn1", "F2-scratch:turn1"):
            first = R.CRITIQUE_TURN if row["cond"] == "F2:turn1" else R.SCRATCH_TURN
            plan2.add({k: row[k] for k in ("item_id", "route", "answer_class", "data", "correct0")} | {"exp": "1A", "cond": row["cond"].split(":")[0]},
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
                    row.update({"out_tokens": o["out_tokens"], "finish_reason": o["finish_reason"],
                                **score_row(o["text"], items[row["item_id"]], o["finish_reason"])})
                k = f"{row['exp']}:{row['data']}:{row['cond']}"
                stats[k]["n"] += 1
                stats[k]["truncated"] += row.get("finish_reason") == "length"
                stats[k]["not ok"] += row["status"] not in ("ok", "fixed")
                f.write(json.dumps(row) + "\n")
    metrics = {"vllm": vllm.__version__, "stage1_s": t1 - t0, "stage2_s": t2 - t1, "stage1_prompts": len(plan.prompts),
               "stage2_prompts": len(plan2.prompts), "t_star_s": ts, "rows": {k: dict(v) for k, v in sorted(stats.items())}}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    finish(d, run_id, args, tok, cfg, {"name": "vllm", "version": vllm.__version__}, "lore-conds")
    print(json.dumps({"run_id": run_id, "dir": str(d), **{k: metrics[k] for k in ("stage1_prompts", "stage2_prompts", "stage1_s", "stage2_s", "t_star_s")}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["initial", "hf", "conds"])
    ap.add_argument("--model", required=True, choices=["4B", "1.7B"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--budgets", type=int, nargs="+", default=list(lore.BUDGETS))
    ap.add_argument("--budget", type=int)
    ap.add_argument("--initial", nargs="+", help="initial run directories (hf: dev and test; conds: the split's)")
    ap.add_argument("--dev-initial", help="conds: the dev initial run (for t*)")
    ap.add_argument("--verbal", help="conds: the hf run directory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbal-batch", type=int, default=16, help="hf: 16 fits 80 GB; 2 on a 12 GB laptop GPU")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split in ("test", "all") and not args.allow_test:
        raise SystemExit("--split test needs --allow-test (lore prereg addendum frozen)")
    args.tag = f"-{args.tag}" if args.tag else ""
    {"initial": stage_initial, "hf": stage_hf, "conds": stage_conds}[args.stage](args)


if __name__ == "__main__":
    main()
