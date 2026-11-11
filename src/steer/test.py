"""Exp 3 on the test split (H3b; prereg §3.1, §7, §10), with everything frozen from the dev pilot.

Frozen inputs: the pilot run (directions.npz, sha256 checked against prereg §6; its top-3 settings and main setting in
metrics.json), the random seeds, skip-first and generation budget of src/steer/pilot.py.
Arms (one fixed batch layout per chunk, greedy, steering in the prefill only, as in the pilot):
  main            the dev-selected setting
  random0..4      matched random directions at the main setting's layer and alpha (H3b's control)
  secondary0/1    the other two top-3 settings; compared with unsteered only (exploratory; prereg §14)
  unsteered
Items: D-inj test items with solution content (others excluded and counted), then 300 initially right original-pool D-nat
test items for R->W, each list ordered by h("steer-test-v1", item_id) (src/data/build.py). Processed in chunks of
--chunk items that complete every arm before the next chunk; a re-run resumes after the last complete chunk.
Pre-check on the first 50 D-inj items before any chunk: alpha = 0 reproduces the unsteered tokens, and the main setting
is reachable (mean KL > 1e-4). If either fails the run stops (prereg §7: then the block moves to the A30, else H3b is
not run).
Outputs: rows.jsonl (as the pilot's confirm_rows, plus finish_reason), texts.jsonl, metrics.json.
Usage: python -m src.steer.test --model 4B --pilot <pilot run dir> --initial <p1 initial.jsonl> --split test
       [--chunk 100] [--limit N] [--max-new N] [--dry-run]   (limit / max-new: smoke tests only)
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..data.build import h
from ..gen import render as R
from ..runlib import MODELS, REPO, register, run_dir, sha256_text, write_manifest
from ..score import scorer
from ..score.scorer import equivalent, score
from .pilot import CELLS, CONFIRM_TOKENS, KL_NOISE, RANDOM_SEEDS, generate_batch, jl
from .steer import SKIP_FIRST, next_token_logprobs, random_directions, solution_positions

DIRECTIONS_SHA = "8aaa4488419da72d1a577667bdfa434b45c4977c1ab9af7f93269d81a46621a8"   # prereg §6
N_RTOW, N_PRECHECK = 300, 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=CONFIRM_TOKENS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    pilot = Path(args.pilot)
    dsha = hashlib.sha256((pilot / "directions.npz").read_bytes()).hexdigest()
    if dsha != DIRECTIONS_SHA:
        raise SystemExit(f"directions.npz sha256 {dsha} is not the frozen one")
    pm = json.loads((pilot / "metrics.json").read_text())
    main_s = pm["main_setting"]
    secondary = [s for s in pm["top3"] if (s["layer"], s["direction"], s["alpha"]) !=
                 (main_s["layer"], main_s["direction"], main_s["alpha"])]
    cfg = {**vars(args), "directions_sha256": dsha, "main": main_s, "secondary": secondary, "random_seeds": RANDOM_SEEDS,
           "skip_first": SKIP_FIRST, "kl_noise": KL_NOISE, "order": "steer-test-v1", "n_rtow": N_RTOW}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    P = REPO / "data" / "processed"
    init = {r["item_id"]: r for r in jl(args.initial)}

    def rendered(item):
        rr = R.review_prompt(tok, item["question"], item["solution_with_answer"] if "solution_with_answer" in item else item["a0"])
        ids, spans = R.token_spans(tok, rr)
        return rr.text, ids, spans

    def n_content(item):
        spans = []
        for role, label in CELLS:
            rr = R.review_prompt(tok, item["question"], item["solution_with_answer"], role=role, label=label)
            spans.append(R.token_spans(tok, rr)[1]["solution"])
        return min(b - a for a, b in spans) - SKIP_FIRST

    dinj_all = [r for r in jl(P / "dinj.jsonl") if r["split"] == args.split]
    dinj = sorted((r for r in dinj_all if n_content(r) > 0), key=lambda r: h("steer-test-v1", r["item_id"]))
    n_excluded = len(dinj_all) - len(dinj)
    nat = [{**r, "a0": init[r["item_id"]]["a0"]} for r in jl(P / "dnat.jsonl")
           if r["split"] == args.split and init[r["item_id"]]["correct"]]
    nat = sorted(nat, key=lambda r: h("steer-test-v1", r["item_id"]))[:N_RTOW]
    if args.limit:
        dinj, nat = dinj[:args.limit], nat[:args.limit]
    items = [("dinj", r, r["gold"], False, r.get("injected_answer")) for r in dinj] + \
            [("dnat", r, r["gold"], True, init[r["item_id"]]["pred"]) for r in nat]
    chunks = [items[i:i + args.chunk] for i in range(0, len(items), args.chunk)]
    arm_names = ["unsteered", "main"] + [f"random{j}" for j in range(len(RANDOM_SEEDS))] + [f"secondary{k}" for k in range(len(secondary))]
    plan = {"items": {"dinj": len(dinj), "dnat_rtow": len(nat), "dinj_excluded_no_content": n_excluded},
            "chunks": len(chunks), "arms": arm_names, "generations": len(items) * len(arm_names), "main": main_s,
            "secondary": secondary}
    if args.dry_run:
        print(json.dumps(plan, indent=1))
        return

    import transformers
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    z = np.load(pilot / "directions.npz")
    vec = lambda s: torch.tensor(z[f"{s['direction']}_{s['layer']}"])
    L, a = main_s["layer"], main_s["alpha"]
    arms = {"unsteered": (None, None, 0.0), "main": (L, vec(main_s), a)}
    for j, rd in enumerate(random_directions(model.config.hidden_size, RANDOM_SEEDS)):
        arms[f"random{j}"] = (L, rd, a)
    for k, s in enumerate(secondary):
        arms[f"secondary{k}"] = (s["layer"], vec(s), s["alpha"])

    run_id, d = run_dir("main" if args.split == "test" else "p2", args.model,
                        f"steer-{args.split}" + ("-smoke" if args.limit else ""))
    register(run_id, "running", block=f"steer-{args.split}", exp="exp3", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()

    # Pre-check on the main GPU (prereg §7): alpha = 0 identity and reachability of the main setting.
    pre = dinj[:N_PRECHECK]
    pp = [rendered(r) for r in pre]
    sha0 = [o["token_ids_sha"] for o in generate_batch(model, tok, [p[0] for p in pp], [p[2]["solution"] for p in pp],
                                                        None, None, 0.0, args.max_new)]
    sha_a0 = [o["token_ids_sha"] for o in generate_batch(model, tok, [p[0] for p in pp], [p[2]["solution"] for p in pp],
                                                          L, vec(main_s), 0.0, args.max_new)]
    kls = []
    for _, ids, spans in pp:
        lp0 = next_token_logprobs(model, ids)
        lp = next_token_logprobs(model, ids, L, vec(main_s), a, solution_positions(spans["solution"]))
        kls.append(float((lp.exp() * (lp - lp0)).sum()))
    precheck = {"alpha0_identical": sha0 == sha_a0, "main_mean_kl": float(np.mean(kls)),
                "main_reachable": float(np.mean(kls)) > KL_NOISE, "n": len(pre), "gpu": torch.cuda.get_device_name(0)}
    (d / "precheck.json").write_text(json.dumps(precheck, indent=1))
    if not (precheck["alpha0_identical"] and precheck["main_reachable"]):
        register(run_id, "failed", block=f"steer-{args.split}", exp="exp3", model=args.model)
        raise SystemExit(f"pre-check failed: {precheck}")

    # Resume after the last complete chunk.
    rows_f, texts_f = d / "rows.jsonl", d / "texts.jsonl"
    done = defaultdict(set)
    if rows_f.exists():
        for r in jl(rows_f):
            done[r["item_id"]].add(r["run"])
    complete = [all(done[it["item_id"]] >= set(arm_names) for _, it, *_ in ch) for ch in chunks]
    keep = {it["item_id"] for ch, ok in zip(chunks, complete) if ok for _, it, *_ in ch}
    if rows_f.exists():   # drop rows of an incomplete chunk; it is regenerated in full
        for f in (rows_f, texts_f):
            kept = [l for l in open(f, encoding="utf-8") if json.loads(l)["item_id"] in keep]
            f.write_text("".join(kept), encoding="utf-8")
    fr, ft = open(rows_f, "a", encoding="utf-8"), open(texts_f, "a", encoding="utf-8")
    for c, (ch, ok) in enumerate(zip(chunks, complete)):
        if ok:
            continue
        prompts, positions = [], []
        for _, it, *_ in ch:
            text, _, spans = rendered(it)
            prompts.append(text)
            positions.append(spans["solution"])
        for name in arm_names:
            Lr, dv, ar = arms[name]
            outs = generate_batch(model, tok, prompts, positions, Lr, dv, ar, args.max_new)
            for (data, it, gold, c0, pred0), o in zip(ch, outs):
                s = score(o["text"], gold, finish_reason=o["finish_reason"])
                fr.write(json.dumps({"run": name, "data": data, "item_id": it["item_id"], "correct0": c0, "correct": s["correct"],
                                     "changed": not (pred0 and s["pred"] and equivalent(s["pred"], pred0)), "status": s["status"],
                                     "finish_reason": o["finish_reason"], "pred0": pred0, "pred": s["pred"],
                                     "token_ids_sha": o["token_ids_sha"]}) + "\n")
                ft.write(json.dumps({"run": name, "data": data, "item_id": it["item_id"], "text": o["text"],
                                     "finish_reason": o["finish_reason"]}, ensure_ascii=False) + "\n")
        fr.flush()
        ft.flush()
        print(f"{time.strftime('%H:%M:%S')} chunk {c + 1}/{len(chunks)} complete", flush=True)
    fr.close()
    ft.close()
    metrics = {**plan, "precheck": precheck, "wall_s": time.time() - t0, "transformers": transformers.__version__,
               "scorer_guarded": scorer.GUARDED}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__}, {})
    register(run_id, "done", block=f"steer-{args.split}", exp="exp3", model=args.model)
    print(json.dumps({"run_id": run_id, "chunks": len(chunks), "wall_s": round(metrics["wall_s"])}))


if __name__ == "__main__":
    main()
