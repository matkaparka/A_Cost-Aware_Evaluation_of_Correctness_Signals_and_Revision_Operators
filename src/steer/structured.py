"""Exp 3 with structured control directions (review #7; docs/planning/next_experiments/plan.md "steer-structured";
DEVIATIONS D16). Exploratory: it asks whether the H3b drop needs the role direction or follows from any structured
edit of the same size at the same place.

Fixed from the frozen Exp 3 (src/steer/test.py): the pilot's main setting (layer 18, role, alpha 0.2; directions.npz
sha256 checked), positions (reviewed-solution content tokens after the first 5, prefill only), the edit
alpha * d_hat * ||h|| (so every arm has the same perturbation norm per position), items, order, chunks of 100,
greedy decoding with 1024 new tokens, and the scorer.
Control directions, computed on DEV only (D-inj dev items with content, the four Exp 1B renderings, resid_post of the
main layer, mean over content tokens, src/steer/steer.py mean_states):
  pc1      the first principal component of the (assistant, self) states (centered): a high-variance direction;
  dataset  mean(h | gsm8k_sc) - mean(h | prm800k_sc) in the (assistant, self) cell: a real but role-free contrast;
  flip     the role direction's construction with each item's sign drawn from h("steer-flip-v1", item_id):
           mean_i s_i * [mean over labels (h_user - h_assistant)]: the same pipeline with the systematic role signal
           cancelled.
Arms: unsteered, main, pc1, dataset, flip (all at the main layer and alpha). Pre-check on the first 50 D-inj test
items: alpha = 0 reproduces unsteered, and each direction's reachability (mean KL at the last prompt position).
Reproduction: the main and unsteered arms are compared token by token (token_ids_sha) with the frozen Exp 3 run
(--ref-rows); the share identical is in metrics.json.
Usage: python -m src.steer.structured --model 4B --pilot <pilot dir> --initial <p1 initial.jsonl> --split test
       --allow-test --ref-rows <main-steer-test rows.jsonl>   (smoke: --split dev --limit 8 --max-new 64)
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
from ..runlib import MODELS, REPO, register, run_dir, sha256_file, sha256_text, write_manifest
from ..score import scorer
from ..score.scorer import equivalent, score
from .pilot import CELLS, CONFIRM_TOKENS, KL_NOISE, generate_batch, jl
from .steer import SKIP_FIRST, mean_states, next_token_logprobs, solution_positions
from .test import DIRECTIONS_SHA, N_PRECHECK, N_RTOW

ARMS = ["unsteered", "main", "pc1", "dataset", "flip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--ref-rows", default=None)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=CONFIRM_TOKENS)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("--split test needs --allow-test (post-freeze exploratory run)")
    pilot = Path(args.pilot)
    dsha = hashlib.sha256((pilot / "directions.npz").read_bytes()).hexdigest()
    if dsha != DIRECTIONS_SHA:
        raise SystemExit(f"directions.npz sha256 {dsha} is not the frozen one")
    main_s = json.loads((pilot / "metrics.json").read_text())["main_setting"]
    L, a = main_s["layer"], main_s["alpha"]
    cfg = {**vars(args), "directions_sha256": dsha, "main": main_s, "arms": ARMS, "skip_first": SKIP_FIRST,
           "order": "steer-test-v1", "flip_stream": "steer-flip-v1", "n_rtow": N_RTOW}

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    P = REPO / "data" / "processed"
    init = {r["item_id"]: r for r in jl(args.initial)}

    def cell_spans(item):
        out = []
        for role, label in CELLS:
            rr = R.review_prompt(tok, item["question"], item["solution_with_answer"], role=role, label=label)
            ids, spans = R.token_spans(tok, rr)
            out.append((ids, spans["solution"]))
        return out

    def n_content(item):
        return min(b - a_ for _, (a_, b) in cell_spans(item)) - SKIP_FIRST

    run_id, d = run_dir("fu" if args.split == "test" else "fu-dev", args.model,
                        f"steer-structured-{args.split}" + ("-smoke" if args.limit else ""))
    register(run_id, "running", block="steer-structured", exp="exp3", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()

    # ---- control directions from dev
    dev = [r for r in jl(P / "dinj.jsonl") if r["split"] == "dev" and n_content(r) > 0]
    if args.limit:   # smoke: a few items of each dataset, so every direction is defined
        dev = [r for ds_ in ("gsm8k_sc", "prm800k_sc", "scli5") for r in [x for x in dev if x["dataset"] == ds_][:max(args.limit, 6)]]
    H = {c: [] for c in CELLS}
    for it in dev:
        for c, (ids, span) in zip(CELLS, cell_spans(it)):
            H[c].append(mean_states(model, ids, span, [L])[L].numpy())
    H = {c: np.stack(v) for c, v in H.items()}
    a_s = H[("assistant", "self")]
    pc1 = np.linalg.svd(a_s - a_s.mean(0), full_matrices=False)[2][0]
    ds = np.array([it["dataset"] for it in dev])
    if not ((ds == "gsm8k_sc").any() and (ds == "prm800k_sc").any()):
        raise SystemExit("dataset direction undefined: a dataset has no dev item")
    dataset = a_s[ds == "gsm8k_sc"].mean(0) - a_s[ds == "prm800k_sc"].mean(0)
    sgn = np.array([1.0 if int(h("steer-flip-v1", it["item_id"]), 16) % 2 == 0 else -1.0 for it in dev])
    diff = (H[("user", "self")] + H[("user", "other")]) / 2 - (H[("assistant", "self")] + H[("assistant", "other")]) / 2
    flip = (sgn[:, None] * diff).mean(0)
    z = np.load(pilot / "directions.npz")
    role = z[f"{main_s['direction']}_{L}"]
    dirs = {"pc1": pc1, "dataset": dataset, "flip": flip}
    np.savez(d / "directions_structured.npz", **dirs, layer=L, n_dev=len(dev))
    cos = {k: float(v @ role / (np.linalg.norm(v) * np.linalg.norm(role))) for k, v in dirs.items()}
    arms = {"unsteered": (None, None, 0.0), "main": (L, torch.tensor(role), a),
            **{k: (L, torch.tensor(v, dtype=torch.float32), a) for k, v in dirs.items()}}

    # ---- items (as src/steer/test.py)
    dinj_all = [r for r in jl(P / "dinj.jsonl") if r["split"] == args.split]
    dinj = sorted((r for r in dinj_all if n_content(r) > 0), key=lambda r: h("steer-test-v1", r["item_id"]))
    nat = [{**r, "a0": init[r["item_id"]]["a0"]} for r in jl(P / "dnat.jsonl")
           if r["split"] == args.split and init[r["item_id"]]["correct"]]
    nat = sorted(nat, key=lambda r: h("steer-test-v1", r["item_id"]))[:N_RTOW]
    if args.limit:
        dinj, nat = dinj[:args.limit], nat[:args.limit]
    items = [("dinj", r, r["gold"], False, r.get("injected_answer")) for r in dinj] + \
            [("dnat", r, r["gold"], True, init[r["item_id"]]["pred"]) for r in nat]
    chunks = [items[i:i + args.chunk] for i in range(0, len(items), args.chunk)]

    def rendered(item):
        rr = R.review_prompt(tok, item["question"], item["solution_with_answer"] if "solution_with_answer" in item else item["a0"])
        ids, spans = R.token_spans(tok, rr)
        return rr.text, ids, spans

    # ---- pre-check: alpha = 0 identity; reachability of every direction
    pp = [rendered(r) for r in dinj[:N_PRECHECK]]
    sha0 = [o["token_ids_sha"] for o in generate_batch(model, tok, [p[0] for p in pp], [p[2]["solution"] for p in pp],
                                                        None, None, 0.0, args.max_new)]
    sha_a0 = [o["token_ids_sha"] for o in generate_batch(model, tok, [p[0] for p in pp], [p[2]["solution"] for p in pp],
                                                          L, arms["pc1"][1], 0.0, args.max_new)]
    kl = {}
    for name in ("main", "pc1", "dataset", "flip"):
        _, dv, al = arms[name]
        ks = []
        for _, ids, spans in pp:
            lp0 = next_token_logprobs(model, ids)
            lp = next_token_logprobs(model, ids, L, dv, al, solution_positions(spans["solution"]))
            ks.append(float((lp.exp() * (lp - lp0)).sum()))
        kl[name] = float(np.mean(ks))
    precheck = {"alpha0_identical": sha0 == sha_a0, "mean_kl": kl, "reachable": {k: v > KL_NOISE for k, v in kl.items()},
                "cos_with_role": cos, "n": len(pp), "n_dev_directions": len(dev), "gpu": torch.cuda.get_device_name(0)}
    (d / "precheck.json").write_text(json.dumps(precheck, indent=1))
    if not precheck["alpha0_identical"]:
        register(run_id, "failed", block="steer-structured", exp="exp3", model=args.model, reason="alpha 0 not identical")
        raise SystemExit(f"pre-check failed: {precheck}")

    # ---- arms, chunk by chunk (resume after the last complete chunk)
    rows_f, texts_f = d / "rows.jsonl", d / "texts.jsonl"
    done = defaultdict(set)
    if rows_f.exists():
        for r in jl(rows_f):
            done[r["item_id"]].add(r["run"])
    complete = [all(done[it["item_id"]] >= set(ARMS) for _, it, *_ in ch) for ch in chunks]
    keep = {it["item_id"] for ch, ok in zip(chunks, complete) if ok for _, it, *_ in ch}
    if rows_f.exists():
        for f in (rows_f, texts_f):
            f.write_text("".join(l for l in open(f, encoding="utf-8") if json.loads(l)["item_id"] in keep), encoding="utf-8")
    fr, ft = open(rows_f, "a", encoding="utf-8"), open(texts_f, "a", encoding="utf-8")
    for c, (ch, ok) in enumerate(zip(chunks, complete)):
        if ok:
            continue
        prompts, positions = [], []
        for _, it, *_ in ch:
            text, _, spans = rendered(it)
            prompts.append(text)
            positions.append(spans["solution"])
        for name in ARMS:
            Lr, dv, ar = arms[name]
            for (data, it, gold, c0, pred0), o in zip(ch, generate_batch(model, tok, prompts, positions, Lr, dv, ar, args.max_new)):
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

    repro = None
    if args.ref_rows:
        ref = {(r["run"], r["item_id"]): r["token_ids_sha"] for r in jl(args.ref_rows) if r["run"] in ("main", "unsteered")}
        mine = [(r["run"], r["item_id"], r["token_ids_sha"]) for r in jl(rows_f) if r["run"] in ("main", "unsteered")]
        both = [(k1, k2, s) for k1, k2, s in mine if (k1, k2) in ref]
        repro = {"compared": len(both), "identical": sum(ref[(k1, k2)] == s for k1, k2, s in both),
                 "ref_rows_sha256": sha256_file(args.ref_rows)}
    metrics = {"items": {"dinj": len(dinj), "dnat_rtow": len(nat)}, "chunks": len(chunks), "arms": ARMS, "precheck": precheck,
               "reproduction_vs_frozen_exp3": repro, "wall_s": time.time() - t0, "transformers": transformers.__version__,
               "scorer_guarded": scorer.GUARDED}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__}, {})
    register(run_id, "done", block="steer-structured", exp="exp3", model=args.model)
    print(json.dumps({"run_id": run_id, "chunks": len(chunks), "precheck": precheck, "reproduction": repro}))


if __name__ == "__main__":
    main()
