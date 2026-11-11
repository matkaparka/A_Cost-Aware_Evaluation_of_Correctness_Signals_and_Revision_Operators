"""Exp 3 dev pilot (design §6, Exp 3; §3.5 H3b): directions, reachability, screening, confirmation, controls.

1. Directions on dev D-inj, F0 renderings of the four role x label cells: d_role = mean(user) - mean(assistant)
   averaged over labels; d_label = mean(other) - mean(self) averaged over roles (positive alpha = toward external).
2. Generic revision-propensity direction from dev F0 revisions (design: "under F0 on dev"): mean(changed) -
   mean(kept) at the last token of the revision instruction. Source: D-inj, assistant/self cell of the Exp 1B pilot,
   items with a known injected answer (D-nat F0 has too few changed revisions on 4B; DECISIONS 2026-11-04).
3. Markers for the screening proxy, from the same revisions: the first revision tokens most over-represented among
   changed (correction markers) and kept (commit markers) ones.
4. Screen (L, alpha, direction) on own-error items (D-inj, assistant/self/F0) with the proxy
   P(first token in correction markers) - P(in commit markers), steered minus unsteered; a setting is reachable if
   the mean KL at the final prompt position exceeds bf16 noise (1e-4 nats).
5. Confirm the top 3 reachable settings by full generation on dev (D-inj for W->R, D-nat initially right for R->W),
   against the generic direction and 5 random directions at the same layer, span and alpha; the generic control is
   valid if it raises the answer-change rate over unsteered (rate over D-inj items with a known injected answer).
6. Main setting (rule fixed in advance): the largest W->R gain over the generic control, subject to its R->W increase
   being <= that of the generic control (if the generic control is invalid: over the mean random control).

Usage: python -m src.steer.pilot --model 4B --exp01 <p2 exp01 run dir> --initial <p1 initial.jsonl> [--limit N]
"""
import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..data.build import h
from ..gen import render as R
from ..runlib import MODELS, REPO, register, run_dir, sha256_text, write_manifest
from ..score.scorer import equivalent, score
from .steer import (SKIP_FIRST, mean_states, next_token_logprobs, random_directions, solution_positions,
                    state_at, steer_hook)

ALPHAS = (0.05, 0.1, 0.2, 0.4)
LAYER_STEP = 4
N_SCREEN, N_CONF_INJ, N_CONF_NAT = 120, 120, 80
RANDOM_SEEDS = (11, 12, 13, 14, 15)
KL_NOISE = 1e-4
CONFIRM_TOKENS = 1024
CELLS = [("assistant", "self"), ("assistant", "other"), ("user", "self"), ("user", "other")]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def generate_batch(model, tok, prompts, positions, layer, direction, alpha, max_new, batch=32):
    """Greedy HF generation, left-padded, one fixed batch layout; steering in the prefill only."""
    outs = []
    tok.padding_side = "left"
    for b in range(0, len(prompts), batch):
        ids = [tok(p, add_special_tokens=False).input_ids for p in prompts[b:b + batch]]
        L = max(map(len, ids))
        pad = [L - len(x) for x in ids]
        x = torch.tensor([[tok.pad_token_id] * p + i for p, i in zip(pad, ids)], device=model.device)
        att = (torch.arange(L, device=model.device)[None] >= torch.tensor(pad, device=model.device)[:, None]).long()
        pos = [solution_positions(sp, p) for sp, p in zip(positions[b:b + batch], pad)]
        ctx = steer_hook(model, layer, direction, alpha, pos) if layer is not None else torch.no_grad()
        with torch.no_grad(), ctx:
            g = model.generate(input_ids=x, attention_mask=att, do_sample=False, max_new_tokens=max_new,
                               pad_token_id=tok.pad_token_id)
        for k in range(len(ids)):
            new = g[k, L:].tolist()
            finished = tok.convert_tokens_to_ids("<|im_end|>") in new or tok.eos_token_id in new
            outs.append({"text": tok.decode(new, skip_special_tokens=True), "finish_reason": "stop" if finished else "length",
                         "token_ids_sha": sha256_text(json.dumps(new))})
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--exp01", required=True)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = {**vars(args), "alphas": ALPHAS, "layer_step": LAYER_STEP, "n_screen": N_SCREEN, "skip_first": SKIP_FIRST,
           "random_seeds": RANDOM_SEEDS, "kl_noise": KL_NOISE, "confirm_tokens": CONFIRM_TOKENS}

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    n_layers = model.config.num_hidden_layers
    layers = list(range(2, n_layers - 2, LAYER_STEP))

    P = REPO / "data" / "processed"
    dinj = [r for r in jl(P / "dinj.jsonl") if r["split"] == "dev"]
    dnat = {r["item_id"]: r for r in jl(P / "dnat.jsonl")}
    init = {r["item_id"]: r for r in jl(args.initial)}
    rows = jl(Path(args.exp01) / "rows.jsonl")
    gens = {g["prompt_key"]: g for g in jl(Path(args.exp01) / "generations.jsonl")}
    n_screen, n_inj, n_nat = N_SCREEN, N_CONF_INJ, N_CONF_NAT

    def rendered(item, role="assistant", label="self"):
        rr = R.review_prompt(tok, item["question"], item["solution_with_answer"] if "solution_with_answer" in item else item["a0"],
                             role=role, label=label)
        ids, spans = R.token_spans(tok, rr)
        return rr.text, ids, spans

    # Steering acts on the solution content tokens (the span minus its first SKIP_FIRST); an item with none (e.g. scli5
    # "The answer is L.") can neither be steered nor contribute to a direction, so it is excluded from every stage.
    def n_content(item):
        spans = [rendered(item, *c)[2]["solution"] for c in CELLS]
        return min(b - a for a, b in spans) - SKIP_FIRST
    n_dinj_all = len(dinj)
    dinj = [it for it in dinj if n_content(it) > 0]
    n_excluded = n_dinj_all - len(dinj)   # before --limit (P2: the smoke reported n_dinj_all - limit)
    if args.limit:   # smoke test: every stage on a few items
        dinj = dinj[:args.limit]
        n_screen = n_inj = n_nat = args.limit

    run_id, d = run_dir("p2", args.model, "steer-pilot" + ("-smoke" if args.limit else ""))
    register(run_id, "running", block="steer-pilot", exp="exp3", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()

    # 1. Role and label directions.
    acc = {c: {L: torch.zeros(model.config.hidden_size) for L in layers} for c in CELLS}
    for it in dinj:
        for c in CELLS:
            _, ids, spans = rendered(it, *c)
            for L, v in mean_states(model, ids, spans["solution"], layers).items():
                acc[c][L] += v / len(dinj)
    d_role = {L: ((acc[("user", "self")][L] + acc[("user", "other")][L]) - (acc[("assistant", "self")][L] + acc[("assistant", "other")][L])) / 2 for L in layers}
    d_label = {L: ((acc[("assistant", "other")][L] + acc[("user", "other")][L]) - (acc[("assistant", "self")][L] + acc[("user", "self")][L])) / 2 for L in layers}

    # 2. Generic revision-propensity direction and 3. markers, from dev F0 revisions of D-inj (Exp 1B pilot,
    # assistant/self). P2: on D-nat F0, 4B changed its answer in 4 of 364 revisions, too few to estimate either.
    f0 = [r for r in rows if r["exp"] == "1A" and r["data"] == "dnat" and r["cond"] == "F0"]
    dinj_by_id = {it["item_id"]: it for it in dinj}
    rev = [r for r in rows if r["exp"] == "1B" and r["data"] == "dinj" and r["cond"] == "assistant/self/F0"
           and r["item_id"] in dinj_by_id and dinj_by_id[r["item_id"]].get("injected_answer")]
    changed = {r["item_id"]: not (r.get("pred") and equivalent(r["pred"], dinj_by_id[r["item_id"]]["injected_answer"]))
               for r in rev}
    g_acc = {k: {L: torch.zeros(model.config.hidden_size) for L in layers} for k in (True, False)}
    n_g = Counter(changed.values())
    first_tok = {True: Counter(), False: Counter()}
    for r in rev:
        _, ids, spans = rendered(dinj_by_id[r["item_id"]])
        for L, v in state_at(model, ids, spans["instruction"][1] - 1, layers).items():
            g_acc[changed[r["item_id"]]][L] += v / max(1, n_g[changed[r["item_id"]]])
        text = gens[r["prompt_key"]]["text"]
        if text:
            first_tok[changed[r["item_id"]]][tok(text, add_special_tokens=False).input_ids[0]] += 1
    d_generic = {L: g_acc[True][L] - g_acc[False][L] for L in layers}
    # Invariant: every direction is finite (P2: NaN directions made every screening value NaN, read as "0 reachable").
    bad = [f"{n}_{L}" for n, dd in (("role", d_role), ("label", d_label), ("generic", d_generic)) for L, v in dd.items()
           if not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"non-finite directions: {bad}")
    tot = {k: sum(v.values()) for k, v in first_tok.items()}

    def log_odds(t):
        a, b = first_tok[True][t] + 0.5, first_tok[False][t] + 0.5
        return math.log(a / (tot[True] + 1)) - math.log(b / (tot[False] + 1))
    cands = [t for t in set(first_tok[True]) | set(first_tok[False]) if first_tok[True][t] + first_tok[False][t] >= 5]
    # Disjoint by construction: over-represented among changed (log-odds > 0) vs among kept (< 0); with fewer than
    # 10 candidates a plain top-5 / bottom-5 split overlaps and the proxy is identically 0 (P2 fixture run).
    corr_markers = sorted((t for t in cands if log_odds(t) > 0), key=log_odds, reverse=True)[:5]
    commit_markers = sorted((t for t in cands if log_odds(t) < 0), key=log_odds)[:5]

    # 4. Screening on own-error items (D-inj, assistant/self/F0).
    screen_items = sorted(dinj, key=lambda r: h("steer-screen-v1", r["item_id"]))[:n_screen]
    base = {}
    for it in screen_items:
        _, ids, spans = rendered(it)
        base[it["item_id"]] = (ids, spans, next_token_logprobs(model, ids))

    def proxy(lp):
        p = lp.exp()
        return float(p[corr_markers].sum() - p[commit_markers].sum())
    screen = []
    for L in layers:
        for dname, dirs in (("role", d_role), ("label", d_label)):
            for a in ALPHAS:
                gains, kls = [], []
                for it in screen_items:
                    ids, spans, lp0 = base[it["item_id"]]
                    lp = next_token_logprobs(model, ids, L, dirs[L], a, solution_positions(spans["solution"]))
                    gains.append(proxy(lp) - proxy(lp0))
                    kls.append(float((lp.exp() * (lp - lp0)).sum()))
                screen.append({"layer": L, "direction": dname, "alpha": a, "proxy_gain": float(np.mean(gains)),
                               "mean_kl": float(np.mean(kls)), "reachable": float(np.mean(kls)) > KL_NOISE})
    if not all(math.isfinite(s["proxy_gain"]) and math.isfinite(s["mean_kl"]) for s in screen):
        raise RuntimeError("non-finite screening values")
    top = sorted([s for s in screen if s["reachable"]], key=lambda s: -s["proxy_gain"])[:3]
    with open(d / "screen.jsonl", "w", encoding="utf-8") as f:
        for row in screen:
            f.write(json.dumps(row) + "\n")
    print(f"{time.strftime('%H:%M:%S')} screening done: {sum(s['reachable'] for s in screen)} reachable", flush=True)

    # 5. Confirmation by full generation on dev.
    conf_inj = sorted(dinj, key=lambda r: h("steer-confirm-v1", r["item_id"]))[:n_inj]
    nat_right = [r for r in f0 if init[r["item_id"]]["correct"]]
    conf_nat = sorted(nat_right, key=lambda r: h("steer-confirm-v1", r["item_id"]))[:n_nat]
    conf = [("dinj", it, it["gold"], False, it.get("injected_answer")) for it in conf_inj] + \
           [("dnat", {**dnat[r["item_id"]], "a0": init[r["item_id"]]["a0"]}, dnat[r["item_id"]]["gold"], True,
             init[r["item_id"]]["pred"]) for r in conf_nat]
    prompts, positions = [], []
    for _, it, *_ in conf:
        text, _, spans = rendered(it)
        prompts.append(text)
        positions.append(spans["solution"])
    runs = {"unsteered": (None, None, 0.0)}
    for k, s in enumerate(top):
        L, a = s["layer"], s["alpha"]
        runs[f"top{k}:{s['direction']}"] = (L, (d_role if s["direction"] == "role" else d_label)[L], a)
        runs[f"top{k}:generic"] = (L, d_generic[L], a)
        for j, rd in enumerate(random_directions(model.config.hidden_size, RANDOM_SEEDS)):
            runs[f"top{k}:random{j}"] = (L, rd, a)
    if top:
        L0 = top[0]["layer"]
        runs["alpha0-identity"] = (L0, d_role[L0], 0.0)
    # Each confirmation arm is written as it finishes (P2: the 4B pilot ran 6 h on an A30 with no visible progress).
    out_rows, out_texts = [], []
    f_rows = open(d / "confirm_rows.jsonl", "w", encoding="utf-8")
    f_texts = open(d / "confirm_texts.jsonl", "w", encoding="utf-8")
    for k_run, (name, (L, dvec, a)) in enumerate(runs.items()):
        outs = generate_batch(model, tok, prompts, positions, L, dvec, a, CONFIRM_TOKENS)
        for (data, it, gold, c0, pred0), o in zip(conf, outs):
            s = score(o["text"], gold, finish_reason=o["finish_reason"])
            out_rows.append({"run": name, "data": data, "item_id": it["item_id"], "correct0": c0, "correct": s["correct"],
                             "changed": not (pred0 and s["pred"] and equivalent(s["pred"], pred0)),
                             "status": s["status"], "pred0": pred0, "pred": s["pred"], "token_ids_sha": o["token_ids_sha"]})
            out_texts.append({"row": len(out_rows) - 1, "run": name, "item_id": it["item_id"], "text": o["text"]})
            f_rows.write(json.dumps(out_rows[-1]) + "\n")
            f_texts.write(json.dumps(out_texts[-1]) + "\n")
        f_rows.flush()
        f_texts.flush()
        print(f"{time.strftime('%H:%M:%S')} confirmation arm {k_run + 1}/{len(runs)} done: {name}", flush=True)
    f_rows.close()
    f_texts.close()

    def rate(run, data, key, cond=lambda r: True):
        rs = [r for r in out_rows if r["run"] == run and r["data"] == data and cond(r)]
        return sum(r[key] for r in rs) / len(rs) if rs else None
    summary = {}
    for name in runs:
        summary[name] = {"w2r_dinj": rate(name, "dinj", "correct"),
                         "r2w_dnat": None if rate(name, "dnat", "correct") is None else 1 - rate(name, "dnat", "correct"),
                         # scli5 items have no injected answer, so "changed" is undefined for them
                         "change_rate": rate(name, "dinj", "changed", lambda r: r["pred0"] is not None)}
    u = summary["unsteered"]
    ident = [r["token_ids_sha"] for r in out_rows if r["run"] == "alpha0-identity"]
    base_sha = [r["token_ids_sha"] for r in out_rows if r["run"] == "unsteered"]
    selection = []
    for k, s in enumerate(top):
        st, ge = summary[f"top{k}:{s['direction']}"], summary[f"top{k}:generic"]
        rnd = [summary[f"top{k}:random{j}"] for j in range(len(RANDOM_SEEDS))]
        generic_valid = ge["change_rate"] is not None and u["change_rate"] is not None and ge["change_rate"] > u["change_rate"]
        ref_w2r = ge["w2r_dinj"] if generic_valid else float(np.mean([x["w2r_dinj"] for x in rnd]))
        ref_r2w = ge["r2w_dnat"] if generic_valid else float(np.mean([x["r2w_dnat"] for x in rnd]))
        selection.append({**s, "w2r": st["w2r_dinj"], "r2w": st["r2w_dnat"], "generic_valid": generic_valid,
                          "w2r_gain_over_ref": st["w2r_dinj"] - ref_w2r, "r2w_increase_ok": (st["r2w_dnat"] - u["r2w_dnat"]) <= (ref_r2w - u["r2w_dnat"])})
    eligible = [s for s in selection if s["r2w_increase_ok"]]
    main_setting = max(eligible, key=lambda s: s["w2r_gain_over_ref"]) if eligible else None

    np.savez(d / "directions.npz", layers=np.array(layers),
             **{f"{n}_{L}": v.numpy() for n, dd in (("role", d_role), ("label", d_label), ("generic", d_generic)) for L, v in dd.items()})
    metrics = {"model": args.model, "layers": layers, "n_dir_items": len(dinj), "n_dinj_excluded_no_content": n_excluded, "generic_groups": {"changed": n_g[True], "kept": n_g[False]},
               "correction_markers": [tok.decode([t]) for t in corr_markers], "commit_markers": [tok.decode([t]) for t in commit_markers],
               "n_reachable": sum(s["reachable"] for s in screen), "top3": top, "summary": summary, "selection": selection,
               "main_setting": main_setting, "alpha0_identical": ident == base_sha if ident else None,
               "wall_s": time.time() - t0, "transformers": transformers.__version__}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__}, {})
    register(run_id, "done", block="steer-pilot", exp="exp3", model=args.model)
    print(json.dumps({k: metrics[k] for k in ("n_reachable", "top3", "main_setting", "alpha0_identical", "wall_s")}, indent=1, default=str))


if __name__ == "__main__":
    main()
