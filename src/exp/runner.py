"""Exp 0 / Exp 1A / Exp 1B / Tsui replication cell: build every generation request for one model and split,
generate in vLLM (VLLM_BATCH_INVARIANT=1, greedy), score, and write one row per (item, condition).

Two stages: stage 1 = all single-turn prompts plus the first turns of two-turn conditions (F2 critique,
F2-scratch, Exp 0 no-feedback answer); stage 2 = the turns that depend on stage-1 outputs. Identical prompts are
generated once (greedy decoding is deterministic in the batch-invariant mode), so verdict conditions that share a
verdict value share one generation.

Pre-freeze discipline (design §5.3): --split test is refused unless --allow-test is given, which is only for
post-freeze main runs.
Usage: VLLM_BATCH_INVARIANT=1 python -m src.exp.runner --model 4B --split dev --initial <p1 initial.jsonl> --verbal <p1 verbal.jsonl>
       python -m src.exp.runner ... --dry-run   (build requests only; local tokenizer via --tokenizer)
"""
import argparse
import os
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..data.verifiers import f3_verdict, permute_self_verdicts, self_verdict
from ..gen import render as R
from ..runlib import MODELS, REPO, register, require_batch_invariant, run_dir, sha256_text, write_manifest
from ..score import scorer
from ..score.scorer import SCORER_VERSION, extract_option, score

P_LEVELS = (0.5, 0.7, 0.9, 1.0)
CELLS_1B = [(role, label) for role in ("assistant", "user") for label in ("self", "other")]


def jl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def tsui_messages(row, dataset, internal):
    if dataset == "scli5":
        return row["messages_error_injection_in_model" if internal else "messages_error_in_user"]
    return row["messages_error_injection_in_model_bca" if internal else "messages_error_in_user_bca"]


class Plan:
    """Collects requests; each request is (key, rendered, max_tokens) and maps to one or more result rows."""

    def __init__(self):
        self.rows, self.prompts = [], {}

    def add(self, row, rendered, max_tokens):
        key = sha256_text(rendered.text + "\x00" + rendered.prefill + f"\x00{max_tokens}")
        self.prompts[key] = (rendered, max_tokens)
        row["prompt_key"] = key
        self.rows.append(row)
        return key


def build_stage1(tok, args, dnat, dinj, exp0, raw):
    plan = Plan()
    rev, crit = args.revision_tokens, args.critique_tokens
    # Self-verdicts and their stratum permutation (D-nat, this split).
    self_v = {r["item_id"]: self_verdict(r["verbal"], args.self_threshold_s) for r in dnat}
    perm = permute_self_verdicts([{"item_id": r["item_id"], "dataset": r["dataset"], "initially_correct": r["correct0"],
                                   "self_says_correct": self_v[r["item_id"]]} for r in dnat])
    for r in dnat:
        base = {"item_id": r["item_id"], "dataset": r["dataset"], "data": "dnat", "correct0": r["correct0"]}
        q, a0 = r["question"], r["a0"]
        plan.rows.append({**base, "exp": "1A", "cond": "B0", "prompt_key": None, "fixed_correct": r["correct0"]})
        plan.add({**base, "exp": "1A", "cond": "B1"}, R.fresh_prompt(tok, q), rev)
        plan.add({**base, "exp": "1A", "cond": "F0"}, R.review_prompt(tok, q, a0), rev)
        plan.add({**base, "exp": "1A", "cond": "F1"}, R.review_prompt(tok, q, a0, prefill=R.TRIGGER_PREFILL), rev)
        for cond, turn in (("F2", R.CRITIQUE_TURN), ("F2-scratch", R.SCRATCH_TURN)):
            first = R.review_prompt(tok, q, a0, instruction=turn)
            plan.add({**base, "exp": "1A", "cond": f"{cond}:turn1", "stage": 1}, first, crit)
        for p in P_LEVELS:
            v = f3_verdict(r["item_id"], r["correct0"], p)
            plan.add({**base, "exp": "1A", "cond": f"F3({p})", "verdict": v},
                     R.review_prompt(tok, q, a0, feedback=R.VERDICT[v]), rev)
        v = self_v[r["item_id"]]
        plan.add({**base, "exp": "1A", "cond": "F5", "verdict": v}, R.review_prompt(tok, q, a0, feedback=R.VERDICT[v]), rev)
        for k, v in enumerate(perm[r["item_id"]]):
            plan.add({**base, "exp": "1A", "cond": f"F3-perm[{k}]", "verdict": v},
                     R.review_prompt(tok, q, a0, feedback=R.VERDICT[v]), rev)
        for role, label in ([] if args.only_1a_dnat else CELLS_1B):
            for fb_name, fb, prefill in (("F0", "", ""), ("F1", "", R.TRIGGER_PREFILL),
                                         ("F3(1.0)", R.VERDICT[r["correct0"]], "")):
                plan.add({**base, "exp": "1B", "cond": f"{role}/{label}/{fb_name}"},
                         R.review_prompt(tok, q, a0, role=role, label=label, feedback=fb, prefill=prefill), rev)
    alt = args.label_set == "alt"
    for r in dinj:
        base = {"item_id": r["item_id"], "dataset": r["dataset"], "data": "dinj", "correct0": False}
        q, sol = r["question"], r["solution_with_answer"]
        for role, label in CELLS_1B:
            for fb_name, fb, prefill in (("F0", "", ""), ("F1", "", R.TRIGGER_PREFILL), ("F3(1.0)", R.VERDICT[False], "")):
                plan.add({**base, "exp": "1B", "cond": f"{role}/{label}/{fb_name}"},
                         R.review_prompt(tok, q, sol, role=role, label=label, feedback=fb, prefill=prefill,
                                         labels=R.LABELS_ALT if alt else None), rev)
        if alt:
            continue   # the alternative wrapper set covers Exp 1B only
        if r.get("mistake_step"):
            step = r["steps"][r["mistake_step"] - 1]
            plan.add({**base, "exp": "1A", "cond": "F4"},
                     R.review_prompt(tok, q, sol, feedback=R.LOCATION["step"].format(step=step)), rev)
        row = raw[r["dataset"]][r["source_index"]]
        for internal in (True, False):
            for wait in (False, True):
                plan.add({**base, "exp": "tsui", "cond": f"{'internal' if internal else 'external'}{'+wait' if wait else ''}"},
                         R.tsui_prompt(tok, tsui_messages(row, r["dataset"], internal), wait=wait), args.tsui_tokens)
    for tid in sorted({r["template_id"] for r in exp0}):
        prompt = next(r["prompt"] for r in exp0 if r["template_id"] == tid)
        text = R.render_messages(tok, [{"role": "user", "content": prompt}])
        plan.add({"item_id": tid, "data": "exp0", "exp": "0", "cond": "initial", "stage": 1},
                 R.Rendered(text=text, spans={}), args.exp0_tokens)
    return plan


def build_stage2(tok, args, plan1, out1, dnat_by_id, exp0):
    """Second turns: F2 / F2-scratch revisions and Exp 0 feedback conditions per world."""
    plan = Plan()
    rev = args.revision_tokens
    for row in plan1.rows:
        if row.get("cond") in ("F2:turn1", "F2-scratch:turn1"):
            r = dnat_by_id[row["item_id"]]
            first_turn = R.CRITIQUE_TURN if row["cond"] == "F2:turn1" else R.SCRATCH_TURN
            hist = [(first_turn, out1[row["prompt_key"]]["text"])]
            cond = row["cond"].split(":")[0]
            plan.add({k: row[k] for k in ("item_id", "dataset", "data", "correct0")} | {"exp": "1A", "cond": cond},
                     R.review_prompt(tok, r["question"], r["a0"], history=hist), rev)
    initial = {row["item_id"]: out1[row["prompt_key"]]["text"] for row in plan1.rows if row.get("exp") == "0"}
    for r in exp0:
        first = initial[r["template_id"]]
        choice = extract_option(first)
        correct0 = choice == r["gold_letter"]
        base = {"item_id": r["item_id"], "data": "exp0", "exp": "0", "world": r["world"], "correct0": correct0,
                "valid0": choice is not None}
        plan.rows.append({**base, "cond": "no-feedback", "prompt_key": None, "fixed_correct": correct0})

        def ren(fb, history=()):
            msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": first}]
            for u, a in history:
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": (fb + "\n\n" if fb else "") + R.EXP0_INSTRUCTION})
            return R.Rendered(text=R.render_messages(tok, msgs), spans={})
        plan.add({**base, "cond": "F0"}, ren(""), args.exp0_tokens)
        for p in P_LEVELS:
            v = f3_verdict(r["item_id"], correct0, p)
            plan.add({**base, "cond": f"F3({p})", "verdict": v}, ren(R.VERDICT[v]), args.exp0_tokens)
    return plan


def generate(llm, plan, sampling_cls):
    keys = list(plan.prompts)
    if not keys:   # --label-set alt has no second-turn conditions
        return {}
    texts = [plan.prompts[k][0].text + plan.prompts[k][0].prefill for k in keys]
    params = [sampling_cls(temperature=0.0, max_tokens=plan.prompts[k][1], seed=0) for k in keys]
    outs = llm.generate(texts, params, use_tqdm=False)
    return {k: {"text": plan.prompts[k][0].prefill + o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                "out_tokens": len(o.outputs[0].token_ids), "prompt_tokens": len(o.prompt_token_ids)}
            for k, o in zip(keys, outs)}


def score_rows(plan, outs, golds, exp0_gold):
    for row in plan.rows:
        if row.get("stage") == 1:
            continue
        if row["prompt_key"] is None:
            row["correct"], row["status"] = row["fixed_correct"], "fixed"
            continue
        o = outs[row["prompt_key"]]
        row.update({"out_tokens": o["out_tokens"], "finish_reason": o["finish_reason"]})
        if row["data"] == "exp0":
            choice = extract_option(o["text"]) if o["finish_reason"] != "length" else None
            row.update({"choice": choice, "valid": choice is not None, "correct": choice == exp0_gold[row["item_id"]],
                        "status": "ok" if choice else "invalid"})
        else:
            # The Tsui continuation has no answer-format instruction, so its last number counts (scorer 1.1.0).
            s = score(o["text"], golds[row["item_id"]], finish_reason=o["finish_reason"],
                      last_number=row["exp"] == "tsui")
            row.update({"correct": s["correct"], "status": s["status"], "pred": s["pred"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--initial", required=True)
    ap.add_argument("--verbal", required=True)
    ap.add_argument("--self-threshold-s", type=float, default=0.0,
                    help="self-verdict threshold on log P(yes) - log P(no); 0 = P(yes)_norm >= 0.5 (design 5.7)")
    ap.add_argument("--revision-tokens", type=int, default=2048)
    ap.add_argument("--critique-tokens", type=int, default=1024)
    ap.add_argument("--tsui-tokens", type=int, default=1024)
    ap.add_argument("--exp0-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="first N items per table (smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="", help="run-id suffix, e.g. smoke")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--items", default="data/processed/dnat.jsonl", help="D-nat item table (P3: ext/dnat_ext.jsonl)")
    ap.add_argument("--only-1a-dnat", action="store_true", help="Exp 1A on D-nat only (P3 expansion dev pilot)")
    ap.add_argument("--label-set", default="main", choices=["main", "alt"],
                    help="alt: the Exp 1B robustness run, LABELS_ALT on D-inj only, first --subset items in the seeded order")
    ap.add_argument("--subset", type=int, default=300, help="items for --label-set alt")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("refusing test split before the prereg freeze (design §5.3)")
    cfg = {**vars(args), "scorer_version": SCORER_VERSION,
           "wording_sha256": sha256_text(json.dumps([R.PROMPT, R.LABELS_ALT if args.label_set == "alt" else R.LABELS,
                                                     R.INSTRUCTION, R.VERDICT, R.LOCATION,
                                                     R.CRITIQUE_TURN, R.SCRATCH_TURN, R.TRIGGER_PREFILL,
                                                     R.EXP0_INSTRUCTION], sort_keys=True))}

    P = REPO / "data" / "processed"
    dnat_meta = {r["item_id"]: r for r in jl(REPO / args.items)}
    init = {r["item_id"]: r for r in jl(args.initial)}
    verbal = {r["item_id"]: r for r in jl(args.verbal)}
    dnat = [{**dnat_meta[i], "a0": init[i]["a0"], "correct0": init[i]["correct"], "p_yes_norm": verbal[i]["p_yes_norm"], "verbal": verbal[i]}
            for i in init if init[i]["split"] == args.split]
    dinj = [] if args.only_1a_dnat else [r for r in jl(P / "dinj.jsonl") if r["split"] == args.split]
    exp0 = [] if args.only_1a_dnat else [r for r in jl(P / "exp0.jsonl") if r["split"] == args.split]
    if args.label_set == "alt":
        rank = {c: i for i, c in enumerate(json.loads((P / "split.json").read_text())["order"][args.split])}
        dinj = sorted(dinj, key=lambda r: (rank[r["canonical_id"]], r["item_id"]))[:args.subset]
        dnat, exp0 = [], []
    if args.limit:
        dnat, dinj = dnat[:args.limit], dinj[:args.limit]
        keep = sorted({r["template_id"] for r in exp0})[:max(1, args.limit // 2)]
        exp0 = [r for r in exp0 if r["template_id"] in keep]
    # Raw D-inj rows (Tsui message lists) are not in git; on the run host they live in /home/data_raw (src/data/fetch.py).
    raw_dir = Path(os.environ.get("DATA_RAW", REPO / "data" / "raw"))
    raw_sha = {}
    for ds in ("gsm8k_sc", "prm800k_sc", "scli5"):
        man = json.loads((raw_dir / ds / "manifest.json").read_text())
        built = json.loads((P / "build_manifest.json").read_text())["sources"][ds]["rows_sha256"]
        if man["rows_sha256"] != built:
            raise SystemExit(f"{ds}: raw rows differ from the ones data/processed was built from")
        raw_sha[ds] = man["rows_sha256"]
    cfg["raw_rows_sha256"] = raw_sha
    raw = {ds: {r["source_index"]: r for r in jl(raw_dir / ds / "rows.jsonl")}
           for ds in ("gsm8k_sc", "prm800k_sc", "scli5")}
    golds = {**{r["item_id"]: r["gold"] for r in dnat}, **{r["item_id"]: r["gold"] for r in dinj}}
    exp0_gold = {r["item_id"]: r["gold_letter"] for r in exp0}

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer or MODELS[args.model])
        plan1 = build_stage1(tok, args, dnat, dinj, exp0, raw)
        print(json.dumps({"items": {"dnat": len(dnat), "dinj": len(dinj), "exp0": len(exp0)},
                          "stage1_rows": len(plan1.rows), "stage1_unique_prompts": len(plan1.prompts),
                          "conditions": Counter(f"{r['exp']}:{r['cond']}" for r in plan1.rows)}, indent=1))
        return

    require_batch_invariant()
    import vllm
    from vllm import LLM, SamplingParams
    run_id, d = run_dir("p2" if args.split == "dev" else "main", args.model, f"exp01-{args.split}" + (f"-{args.tag}" if args.tag else ""))
    register(run_id, "running", block=f"exp01-{args.split}", exp="exp01", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=12288)
    tok = llm.get_tokenizer()
    plan1 = build_stage1(tok, args, dnat, dinj, exp0, raw)
    t1 = time.time()
    out1 = generate(llm, plan1, SamplingParams)
    plan2 = build_stage2(tok, args, plan1, out1, {r["item_id"]: r for r in dnat}, exp0)
    t2 = time.time()
    out2 = generate(llm, plan2, SamplingParams)
    t3 = time.time()
    # Generations are written before scoring (DEVIATIONS 2026-11-05: a killed scoring step lost them all).
    with open(d / "generations.jsonl", "w", encoding="utf-8") as f:
        for k, o in {**out1, **out2}.items():
            f.write(json.dumps({"prompt_key": k, **o}, ensure_ascii=False) + "\n")
    score_rows(plan1, out1, golds, exp0_gold)
    score_rows(plan2, out2, golds, exp0_gold)
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for row in plan1.rows + plan2.rows:
            if row.get("stage") != 1:
                f.write(json.dumps(row) + "\n")
    stats = defaultdict(Counter)
    for row in plan1.rows + plan2.rows:
        if row.get("stage") != 1:
            stats[f"{row['exp']}:{row['data']}:{row['cond']}"]["n"] += 1
            stats[f"{row['exp']}:{row['data']}:{row['cond']}"]["truncated"] += row.get("finish_reason") == "length"
    metrics = {"model": args.model, "split": args.split, "vllm": vllm.__version__, "load_s": t1 - t0,
               "stage1_s": t2 - t1, "stage2_s": t3 - t2, "stage1_prompts": len(plan1.prompts),
               "stage2_prompts": len(plan2.prompts), "rows": {k: dict(v) for k, v in sorted(stats.items())},
               "scorer_guarded": scorer.GUARDED, "score_s": time.time() - t3}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "vllm", "version": vllm.__version__},
                   {"wording": cfg["wording_sha256"]})
    register(run_id, "done", block=f"exp01-{args.split}", exp="exp01", model=args.model)
    print(json.dumps({"run_id": run_id, **{k: metrics[k] for k in ("stage1_prompts", "stage2_prompts", "stage1_s", "stage2_s")}}))


if __name__ == "__main__":
    main()
