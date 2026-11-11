"""Verbal-readout variants for the readout diagnostic (docs/planning/validation_v2/plan.md §2.1; DEVIATIONS D18).

The main study's verbal score asks one fixed question after a0 (W0) and reads P(yes) and P(no) at the next position
(src/score/verbal.py). Tulu-3-8B's W0 score ranks a0 correctness at chance on math (reports/fu_04_verbal_check.md). To
tell a weak readout from a wording-specific one, this module scores a0 under fixed variants, written before any
variant output existed:
  W0  the fixed question;
  W1  third person ("the final answer above");
  W2  one-word reply;
  W3  reversed polarity ("wrong"; the score uses P(no) as the "correct" side);
  W4  true / false instead of yes / no.
For each item and variant: p_pos (the "correct" side), p_neg, mass, s = log p_pos - log p_neg. Dev only.

Selection rule for later use (fixed here): per model, a variant is eligible only if its median dev mass is at least
MIN_MASS (otherwise the model does not answer in that format; the 4B smoke run showed this for W4). Among eligible
variants W1-W4, the one with the highest pooled dev AUROC is the "dev-selected variant" if its AUROC exceeds W0's by at
least SELECT_MARGIN; otherwise there is none. W0 stays the
primary readout everywhere; a dev-selected variant is reported only as a secondary readout on a new task.
Usage: python -m src.score.verbal_variants --model T8B --initial <p1 initial dir> [<ext initial dir>] [--limit N]
       (writes a fu-dev run directory: variants.jsonl, metrics.json)
"""
import argparse
import json
import math
import time
from pathlib import Path

from ..gen import render as R
from ..runlib import MODELS, needs_bos, register, run_dir, sha256_text, write_manifest

YES = ["yes", "Yes", "YES", " yes", " Yes", " YES"]
NO = ["no", "No", "NO", " no", " No", " NO"]
TRUE = ["true", "True", "TRUE", " true", " True", " TRUE"]
FALSE = ["false", "False", "FALSE", " false", " False", " FALSE"]
VARIANTS = {
    "W0": (R.VERIFY_QUERY, YES, NO),
    "W1": ("Is the final answer above correct? Answer yes or no.", YES, NO),
    "W2": ("Is your final answer correct? Reply with one word: yes or no.", YES, NO),
    "W3": ("Is your final answer wrong? Answer yes or no.", NO, YES),       # "no" = not wrong = the correct side
    "W4": ("Is your final answer correct? Answer true or false.", TRUE, FALSE),
}
SELECT_MARGIN = 0.05
MIN_MASS = 0.5


def candidates(tok, words):
    ids = {w: tok(w, add_special_tokens=False).input_ids for w in words}
    if not all(len(v) == 1 for v in ids.values()):
        raise SystemExit(f"candidates are not single tokens: { {w: v for w, v in ids.items() if len(v) != 1} }")
    return sorted({v[0] for v in ids.values()})


def readout(tok, model, qs, a0s, query, pos, neg, batch=16):
    """p_pos, p_neg at the position after the query, for (q, a0) pairs; left padding, last position only."""
    import torch
    cp, cn = candidates(tok, pos), candidates(tok, neg)
    if set(cp) & set(cn):
        raise SystemExit("positive and negative candidates share a token")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for b in range(0, len(qs), batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": R.PROMPT.format(q=q)}, {"role": "assistant", "content": a},
                                          {"role": "user", "content": query}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
                 for q, a in zip(qs[b:b + batch], a0s[b:b + batch])]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=needs_bos(tok, texts[0])).to("cuda")
        with torch.no_grad():
            p = torch.softmax(model(**enc, logits_to_keep=1).logits[:, -1].float(), -1)
        for k in range(len(texts)):
            pp, pn = float(p[k, cp].sum()), float(p[k, cn].sum())
            out.append({"p_pos": pp, "p_neg": pn, "mass": pp + pn,
                        "s": math.log(pp) - math.log(pn) if pp > 0 and pn > 0 else None})
    return out


def select(run):
    """The fixed selection rule (module docstring) applied to a variants run; returns (variant or None, summary)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rows = [json.loads(line) for line in open(Path(run) / "variants.jsonl", encoding="utf-8")]
    y = np.array([r["correct"] for r in rows], int)
    summ = {}
    for v in VARIANTS:
        s = np.array([r[v]["s"] if r[v]["s"] is not None else (np.inf if r[v]["p_neg"] == 0 else -np.inf) for r in rows], float)
        s = np.clip(s, -1e6, 1e6)
        summ[v] = {"auroc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
                   "median mass": float(np.median([r[v]["mass"] for r in rows]))}
    elig = [v for v in VARIANTS if v != "W0" and summ[v]["median mass"] >= MIN_MASS and summ[v]["auroc"] is not None]
    best = max(elig, key=lambda v: summ[v]["auroc"], default=None)
    chosen = best if best and summ["W0"]["auroc"] is not None and summ[best]["auroc"] >= summ["W0"]["auroc"] + SELECT_MARGIN else None
    return chosen, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", help="print the dev-selected variant of this variants run (or 'none') and exit")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--initial", nargs="+", help="p1 initial-answer run directories (dev rows are used)")
    ap.add_argument("--items", nargs="+", default=["data/processed/dnat.jsonl", "data/processed/ext/dnat_ext.jsonl"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.select:
        chosen, summ = select(args.select)
        print(json.dumps(summ), file=__import__("sys").stderr)
        print(chosen or "none")
        return
    if not (args.model and args.initial):
        raise SystemExit("--model and --initial are required")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = Path(__file__).resolve().parents[2]
    q = {}
    for f in args.items:
        for line in open(repo / f, encoding="utf-8"):
            r = json.loads(line)
            q[r["item_id"]] = (r["question"], r["dataset"])
    rows = [json.loads(line) for d in args.initial for line in open(Path(d) / "initial.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "dev" and r["item_id"] in q]
    rows = rows[:args.limit] if args.limit else rows
    cfg = {**vars(args), "variants": {k: v[0] for k, v in VARIANTS.items()}, "select_margin": SELECT_MARGIN}
    run_id, d = run_dir("fu-dev", args.model, "verbal-variants-dev" + (f"-{args.tag}" if args.tag else ""))
    register(run_id, "running", block="verbal-variants", exp="verbal", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    t0 = time.time()
    res = {}
    for name, (query, pos, neg) in VARIANTS.items():
        res[name] = readout(tok, model, [q[r["item_id"]][0] for r in rows], [r["a0"] for r in rows], query, pos, neg, args.batch)
    with open(d / "variants.jsonl", "w", encoding="utf-8") as f:
        for k, r in enumerate(rows):
            f.write(json.dumps({"item_id": r["item_id"], "dataset": q[r["item_id"]][1], "correct": bool(r["correct"]),
                                **{name: res[name][k] for name in VARIANTS}}) + "\n")
    (d / "metrics.json").write_text(json.dumps({"n": len(rows), "gpu_s": time.time() - t0}, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__},
                   {k: v[0] for k, v in VARIANTS.items()})
    register(run_id, "done", block="verbal-variants", exp="verbal", model=args.model)
    print(json.dumps({"run_id": run_id, "dir": str(d), "n": len(rows)}))


if __name__ == "__main__":
    main()
