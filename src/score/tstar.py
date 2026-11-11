"""Self-verdict threshold t* for a new model by the prereg §11 adaptation rule, step 1 (no judgment): the rule of
analysis/p2_selfverdict.py on the model's original-pool dev items. s = log P(yes) - log P(no); t* = the midpoint
between adjacent distinct scores that maximizes dev balanced accuracy (ties: the lowest). Same code path as
analysis/p4_tstar_8B.py, packaged for the run host so a sequence can select t* before its test run; the selection is
registered (runs.jsonl) before any test run of the model. Dev rows only.
Output: <run>/selfverdict.json {model, n_dev, n_unscored, t_star_s, balanced_acc_t_star, balanced_acc_half}.
Usage: python -m src.score.tstar --model T8B --initial <p1 initial.jsonl> --verbal <p1 verbal.jsonl>
"""
import argparse
import json
import math

import numpy as np

from ..runlib import MODELS, register, run_dir, sha256_file, sha256_text, write_manifest


def score(r):
    if r.get("p_yes") is None or r.get("p_no") is None:
        return None
    return math.log(max(r["p_yes"], 1e-300)) - math.log(max(r["p_no"], 1e-300))


def balanced_acc(s, y, t):
    pred = s >= t
    return ((pred & y).sum() / max(1, y.sum()) + (~pred & ~y).sum() / max(1, (~y).sum())) / 2


def select(initial_rows, verbal_rows):
    init = {r["item_id"]: r for r in initial_rows}
    dev = [r for r in verbal_rows if r["split"] == "dev"]
    scored = [(score(r), bool(init[r["item_id"]]["correct"])) for r in dev if score(r) is not None]
    s = np.array([x for x, _ in scored])
    y = np.array([c for _, c in scored], bool)
    u = np.unique(s)
    cands = (u[:-1] + u[1:]) / 2
    ba = np.array([balanced_acc(s, y, t) for t in cands])
    t = float(cands[int(np.argmax(ba))])
    return {"n_dev": len(dev), "n_unscored": len(dev) - len(scored), "t_star_s": t,
            "t_star_p_yes_norm": 1 / (1 + math.exp(-t)) if t < 700 else 1.0,
            "balanced_acc_t_star": float(ba.max()), "balanced_acc_half": float(balanced_acc(s, y, 0.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--verbal", required=True)
    args = ap.parse_args()
    jl = lambda p: [json.loads(line) for line in open(p, encoding="utf-8")]  # noqa: E731
    cfg = {**vars(args), "initial_sha256": sha256_file(args.initial), "verbal_sha256": sha256_file(args.verbal),
           "rule": "analysis/p2_selfverdict.py: max dev balanced accuracy, midpoints, ties lowest"}
    run_id, d = run_dir("p2", args.model, "tstar")
    register(run_id, "running", block="tstar", exp="selfverdict", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    out = {"model": args.model, **select(jl(args.initial), jl(args.verbal))}
    (d / "selfverdict.json").write_text(json.dumps(out, indent=1))
    from transformers import AutoTokenizer
    write_manifest(d, run_id, args.model, AutoTokenizer.from_pretrained(MODELS[args.model]), cfg,
                   {"name": "numpy", "version": np.__version__}, {})
    register(run_id, "done", block="tstar", exp="selfverdict", model=args.model, t_star_s=out["t_star_s"])
    print(json.dumps({"run_id": run_id, **out}))


if __name__ == "__main__":
    main()
