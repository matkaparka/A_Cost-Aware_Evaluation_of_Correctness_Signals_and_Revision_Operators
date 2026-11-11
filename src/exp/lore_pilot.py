"""Lore QA dev pilot on the control host (docs/planning/lore_qa/proposal.md §7: inputs to G2, G3, G3b, G4). HF
backend, bf16, greedy. Dev items only; the script refuses anything else.

  initial  initial answers a0 for every lore dev item (tracks A and B) at each context budget, scored with
           src/score/lore.py (primary and second scorer), then the verbal P(yes) on those answers (design §5.7)
  smoke    the revision conditions on the first --n dev items in the seeded order, at one budget: F0, F1, F2,
           F2-scratch, F3(1.0), the three other Exp 1B F0 cells; and the D-inj-lore rows of the same items under the
           four Exp 1B F0 cells. Checks rendering, parsing and truncation end to end.

Prompts, wrappers and token budgets are the main study's (src/gen/render.py; prereg §1: a0 4096, revision 2048,
critique 1024). HF and vLLM greedy outputs differ (reports/00_compute_rates.md), so the gates are re-measured on the
run host before any test run (proposal §7).
Usage: python -m src.exp.lore_pilot --model 4B --stage initial
       python -m src.exp.lore_pilot --model 4B --stage smoke --budget 1024 --initial <initial run dir> [--n 20]
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from ..data import lore, lore_b
from ..data.lore import BUDGETS, OUT, render_question
from ..gen import render as R
from ..runlib import MODELS, REPO, code_commit, sha256_file, sha256_text
from ..score import lore as S
from ..score.verbal import NO, QUERY, YES

A0_TOKENS, REVISION_TOKENS, CRITIQUE_TOKENS = 4096, 2048, 1024        # prereg §1
BATCH = {512: 8, 1024: 5, 2048: 3}       # by context budget: a batch must fit 12 GB even if every sequence runs to 4096
VERBAL_BATCH = 2                         # the verbal pass after generation OOMed at 8 with 1024-token contexts (runs.jsonl)
SMOKE_TOKEN_CAP = 16000                  # smoke: review prompts carry a0 (and F2 its critique); 8 at budget 512 OOMed (runs.jsonl)
CELLS_OTHER = [("user", "self"), ("assistant", "other"), ("user", "other")]   # (assistant, self) is F0
COMMIT = code_commit()                     # the locked commit: read once at start, used for the run ID and manifest


def dev_items():
    passages, items_a = lore.load_built()
    items = [dict(it, route=it.get("route", "A")) for it in items_a] + lore_b.load_items_b()
    items = [it for it in items if it["split"] == "dev"]
    return passages, sorted(items, key=lambda it: it["order"])


def load_model(key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[key])
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODELS[key], dtype=torch.bfloat16, device_map="cuda").eval()
    return tok, model


def batches(tok, prompts, max_new, batch, token_cap=None):
    """Batches of similar length. Without token_cap: fixed size, ordered by character length (the initial stage, as
    run at 30b08af). With token_cap: ordered by token length, and a batch closes before batch size x (its longest
    prompt + max_new) would exceed token_cap, so the KV cache fits even if every sequence runs to max_new."""
    if not token_cap:
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i].text))
        return [order[b:b + batch] for b in range(0, len(order), batch)]
    lens = [len(tok(p.text + p.prefill, add_special_tokens=False).input_ids) for p in prompts]
    out, cur = [], []
    for i in sorted(range(len(prompts)), key=lambda i: lens[i]):
        if cur and (len(cur) >= batch or (len(cur) + 1) * (lens[i] + max_new) > token_cap):
            out.append(cur)
            cur = []
        cur.append(i)
    return out + ([cur] if cur else [])


def generate(tok, model, prompts, max_new, batch, token_cap=None):
    """Greedy, left-padded batches (see batches()). prompts: list of Rendered. Returns (text, finish, out_tokens)."""
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    out = [None] * len(prompts)
    for idx in batches(tok, prompts, max_new, batch, token_cap):
        enc = tok([prompts[i].text + prompts[i].prefill for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                               top_k=None, pad_token_id=tok.pad_token_id)
        for k, i in enumerate(idx):
            new = g[k, enc.input_ids.shape[1]:].tolist()
            stop = next((j for j, t in enumerate(new) if t in eos), None)
            toks = new if stop is None else new[:stop]
            out[i] = (prompts[i].prefill + tok.decode(toks, skip_special_tokens=True),
                      "stop" if stop is not None else "length", len(toks))
        del g, enc
        torch.cuda.empty_cache()
    return out


def verbal(tok, model, qs, a0s, batch=VERBAL_BATCH):
    """P(yes)_norm and log-odds s for (q, a0) pairs, as src/score/verbal.py (first-token shortcut checked)."""
    cand = {c: tok(c, add_special_tokens=False).input_ids for c in YES + NO}
    if not (all(len(v) == 1 for v in cand.values()) and len({v[0] for v in cand.values()}) == len(cand)):
        raise SystemExit("verbal candidates are not distinct single tokens under this tokenizer")
    rows = []
    for b in range(0, len(qs), batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": R.PROMPT.format(q=q)},
                                          {"role": "assistant", "content": a}, {"role": "user", "content": QUERY}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for q, a in zip(qs[b:b + batch], a0s[b:b + batch])]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():   # last position only: full-sequence logits of long lore prompts do not fit in 12 GB
            logp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[:, -1].float(), -1)
        for k in range(len(texts)):
            py = sum(math.exp(float(logp[k, cand[c][0]])) for c in YES)
            pn = sum(math.exp(float(logp[k, cand[c][0]])) for c in NO)
            rows.append({"p_yes": py, "p_no": pn, "mass": py + pn,
                         "s": math.log(py) - math.log(pn) if py > 0 and pn > 0 else None})
    return rows


def run_dir(model_key, kind):
    root = REPO / "results" / "lore" / model_key
    run_id = f"lore-{kind}-{model_key}-{COMMIT[:7]}"
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    return run_id, d


def register(run_id, status, **fields):
    """The repository's run registry (CLAUDE.md §4); a control-host run, so no run-host file to merge."""
    import socket
    from datetime import datetime, timezone
    row = {"run_id": run_id, "status": status, "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "host": socket.gethostname(), "code_commit": COMMIT, **fields}
    with open(REPO / "runs.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")


def manifest(d, run_id, key, tok, cfg):
    import transformers
    from huggingface_hub import scan_cache_dir
    revs = [r.commit_hash for repo in scan_cache_dir().repos if repo.repo_id == MODELS[key] for r in repo.revisions]
    man = {"run_id": run_id, "model": MODELS[key], "model_revision": revs, "chat_template_sha256": sha256_text(tok.chat_template or ""),
           "wording_sha256": sha256_file(REPO / "src" / "gen" / "render.py"),
           "lore_scorer_sha256": sha256_file(REPO / "src" / "score" / "lore.py"),
           "items_a_sha256": sha256_file(OUT / "items_a.jsonl"), "items_b_sha256": sha256_file(OUT / "items_b.jsonl"),
           "items_inj_sha256": sha256_file(OUT / "items_inj.jsonl"), "passages_sha256": sha256_file(OUT / "passages.jsonl"),
           "code_commit": COMMIT, "config_sha256": sha256_text(json.dumps(cfg, sort_keys=True)),
           "backend": {"name": "hf-transformers", "version": transformers.__version__, "dtype": "bfloat16", "greedy": True},
           "gpu": torch.cuda.get_device_name(0), "host": "control (laptop)"}
    (d / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8", newline="\n")
    man["files"] = {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != "manifest.json"}
    (d / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stage_initial(args, tok, model, passages, items, save):
    rows = []
    for budget in args.budgets:
        prompts = [R.fresh_prompt(tok, render_question(it, budget, passages)) for it in items]
        outs = generate(tok, model, prompts, A0_TOKENS, args.batch or BATCH[budget])
        vb = verbal(tok, model, [render_question(it, budget, passages) for it in items], [o[0] for o in outs])
        for it, (text, finish, n), v in zip(items, outs, vb):
            s1, s2 = S.score(text, it, finish), S.score_contains(text, it, finish)
            rows.append({"item_id": it["id"], "route": it["route"], "template": it["template"],
                         "answer_kind": it["answer_kind"], "answer_class": it["answer_class"], "budget": budget,
                         "a0": text, "finish_reason": finish, "out_tokens": n, "status": s1["status"],
                         "pred": s1["pred"], "correct": s1["correct"], "correct_second": s2["correct"], **v})
        save(rows)                                       # a crash in a later budget keeps the finished ones
        print(json.dumps({"budget": budget, "n": len(items), "acc": sum(r["correct"] for r in rows if r["budget"] == budget) / len(items)}),
              flush=True)
    return rows


def stage_smoke(args, tok, model, passages, items, save):
    a0 = {r["item_id"]: r for r in map(json.loads, (Path(args.initial) / "rows.jsonl").read_text(encoding="utf-8").splitlines())
          if r["budget"] == args.budget}
    items = [it for it in items if it["id"] in a0][:args.n]
    inj = [r for r in map(json.loads, (OUT / "items_inj.jsonl").read_text(encoding="utf-8").splitlines())
           if r["budget"] == args.budget and r["base_id"] in {it["id"] for it in items}]
    by_id = {it["id"]: it for it in items}
    reqs = []                                   # (row, rendered, max_new)
    for it in items:
        q, sol, c0 = render_question(it, args.budget, passages), a0[it["id"]]["a0"], a0[it["id"]]["correct"]
        base = {"item_id": it["id"], "data": "dnat", "correct0": c0}
        reqs += [({**base, "cond": "F0"}, R.review_prompt(tok, q, sol), REVISION_TOKENS),
                 ({**base, "cond": "F1"}, R.review_prompt(tok, q, sol, prefill=R.TRIGGER_PREFILL), REVISION_TOKENS),
                 ({**base, "cond": "F3(1.0)"}, R.review_prompt(tok, q, sol, feedback=R.VERDICT[c0]), REVISION_TOKENS)]
        reqs += [({**base, "cond": f"{role}/{label}/F0"}, R.review_prompt(tok, q, sol, role=role, label=label), REVISION_TOKENS)
                 for role, label in CELLS_OTHER]
        reqs += [({**base, "cond": f"{c}:turn1"}, R.review_prompt(tok, q, sol, instruction=t), CRITIQUE_TOKENS)
                 for c, t in (("F2", R.CRITIQUE_TURN), ("F2-scratch", R.SCRATCH_TURN))]
    for r in inj:
        it = by_id[r["base_id"]]
        q = render_question(it, args.budget, passages)
        reqs += [({"item_id": it["id"], "data": "dinj", "variant": r["variant"], "correct0": r["variant"] == "right",
                   "cond": f"{role}/{label}/F0"}, R.review_prompt(tok, q, r["a0"], role=role, label=label), REVISION_TOKENS)
                 for role, label in [("assistant", "self")] + CELLS_OTHER]
    outs, t0 = [], time.time()
    gen = lambda ps, max_new: generate(tok, model, ps, max_new, args.batch or BATCH[args.budget], SMOKE_TOKEN_CAP)  # noqa: E731
    for max_new in (REVISION_TOKENS, CRITIQUE_TOKENS):
        part = [x for x in reqs if x[2] == max_new]
        outs += list(zip([x[0] for x in part], gen([x[1] for x in part], max_new)))
        print(json.dumps({"pass": max_new, "n": len(part), "elapsed_s": round(time.time() - t0)}), flush=True)
    turn1 = {(row["item_id"], row["cond"].split(":")[0]): o[0] for row, o in outs if row["cond"].endswith(":turn1")}
    second = []
    for (iid, cond), text in turn1.items():
        it = by_id[iid]
        first = R.CRITIQUE_TURN if cond == "F2" else R.SCRATCH_TURN
        second.append(({"item_id": iid, "data": "dnat", "correct0": a0[iid]["correct"], "cond": cond},
                       R.review_prompt(tok, render_question(it, args.budget, passages), a0[iid]["a0"], history=[(first, text)])))
    outs += list(zip([x[0] for x in second], gen([x[1] for x in second], REVISION_TOKENS)))
    print(json.dumps({"pass": "F2 second turn", "n": len(second), "elapsed_s": round(time.time() - t0)}), flush=True)
    rows = []
    for row, (text, finish, n) in outs:
        if row["cond"].endswith(":turn1"):
            rows.append({**row, "text": text, "finish_reason": finish, "out_tokens": n, "status": "turn1"})
            continue
        s = S.score(text, by_id[row["item_id"]], finish)
        rows.append({**row, "text": text, "finish_reason": finish, "out_tokens": n, "status": s["status"],
                     "pred": s["pred"], "correct": s["correct"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["1.7B", "4B"])
    ap.add_argument("--stage", required=True, choices=["initial", "smoke"])
    ap.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    ap.add_argument("--budget", type=int, default=1024, help="smoke stage")
    ap.add_argument("--initial", help="smoke stage: the initial run directory")
    ap.add_argument("--n", type=int, default=20, help="smoke stage: dev items")
    ap.add_argument("--batch", type=int, default=0, help="override the per-budget batch size")
    ap.add_argument("--check", help="code check only: first 4 dev items, written to this directory, not registered")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    passages, items = dev_items()
    assert all(it["split"] == "dev" for it in items)
    if args.check:
        items = items[:4]
    cfg = {**vars(args), "a0_tokens": A0_TOKENS, "revision_tokens": REVISION_TOKENS, "critique_tokens": CRITIQUE_TOKENS,
           "batch_by_budget": BATCH, "n_dev_items": len(items)}
    if args.stage == "smoke":
        cfg["smoke_token_cap"] = SMOKE_TOKEN_CAP
    tok, model = load_model(args.model)
    kind = "pilot-initial" if args.stage == "initial" else f"pilot-smoke{args.budget}"
    reg = register
    if args.check:
        run_id, d = f"check-{kind}-{args.model}", Path(args.check)
        d.mkdir(parents=True, exist_ok=True)
        reg = lambda *a, **k: None   # noqa: E731
    else:
        run_id, d = run_dir(args.model, kind)
    reg(run_id, "running", block="lore-pilot", exp="lore", model=args.model, split="dev",
        config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t = time.time()
    rows = (stage_initial if args.stage == "initial" else stage_smoke)(args, tok, model, passages, items,
                                                                       lambda rs: write_jsonl(d / "rows.jsonl", rs))
    write_jsonl(d / "rows.jsonl", rows)
    (d / "metrics.json").write_text(json.dumps({"n_rows": len(rows), "wall_s": round(time.time() - t, 1),
                                                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}, indent=1),
                                    encoding="utf-8", newline="\n")
    manifest(d, run_id, args.model, tok, cfg)
    reg(run_id, "done", block="lore-pilot", exp="lore", model=args.model, split="dev")
    print(json.dumps({"run_id": run_id, "dir": str(d), "rows": len(rows), "wall_s": round(time.time() - t)}))


if __name__ == "__main__":
    main()
