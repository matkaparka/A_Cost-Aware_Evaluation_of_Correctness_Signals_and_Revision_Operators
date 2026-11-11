"""H3c (exploratory, prereg §3.3; design §3.5): are role and label linearly decodable at the reviewed solution's
content tokens under the four Exp 1B prompts? Never run in P4 (DEVIATIONS D13: no activations under the 1B
prompts); docs/planning/next_experiments/plan.md Group 1.

Cache (HF, bf16): for each D-inj item of the split and each (role, label) cell, the F0 review prompt of
src/gen/render.py (as src/exp/runner.py renders it). At every layer (embeddings + every block output, `resid_post`)
the mean hidden state over the reviewed-solution content tokens excluding the first 5 (the span of the Exp 3
directions, design Exp 3); also the last solution token. -> acts.npz (large; it stays on the run host and its
sha256 is in the manifest).

Analysis (same run, CPU): per layer and position, a standardized logistic regression with C fixed a priori at 1.0
(no tuning), 5-fold StratifiedGroupKFold grouped by item (every item's 4 rows fall in one fold), out-of-fold AUROC
for role (user vs assistant) and for label (other vs self). Invariant: at layer 0 (embeddings) the content tokens
are identical across the four cells, so both AUROCs must be 0.5 (checked; metrics.json "layer0_invariant").
The prereg names the quantity, not a decision rule: estimates only.
Usage: python -m src.probe.h3c --model 4B --split test --allow-test   (dev: --split dev --limit 40 --tag smoke)
"""
import argparse
import json
import time

import numpy as np

from ..gen import render as R
from ..runlib import MODELS, REPO, register, run_dir, sha256_file, sha256_text, write_manifest

CELLS = [(role, label) for role in ("assistant", "user") for label in ("self", "other")]
SKIP, C, FOLDS, SEED = 5, 1.0, 5, 0


def cv_auroc(X, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if np.allclose(X, X[:1]):                       # constant features (layer 0): nothing to learn
        return 0.5
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("--split test needs --allow-test (post-freeze exploratory run)")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = {**vars(args), "cells": CELLS, "skip_first": SKIP, "C": C, "folds": FOLDS, "seed": SEED, "feedback": "F0",
           "positions": ["mean over solution tokens after the first 5", "last solution token"]}
    items = [r for r in map(json.loads, open(REPO / "data" / "processed" / "dinj.jsonl", encoding="utf-8"))
             if r["split"] == args.split]
    items = items[:args.limit] if args.limit else items
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda").eval()
    tag = f"-{args.tag}" if args.tag else ""
    run_id, d = run_dir("fu" if args.split == "test" else "fu-dev", args.model, f"h3c-{args.split}{tag}")
    register(run_id, "running", block="h3c-acts", exp="h3c", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t = time.time()
    M, L, ids, role, label, skipped = [], [], [], [], [], []
    with torch.no_grad():
        for it in items:
            rendered = [R.review_prompt(tok, it["question"], it["solution_with_answer"], role=ro, label=la) for ro, la in CELLS]
            spans = [R.token_spans(tok, x) for x in rendered]
            lens = {sp["solution"][1] - sp["solution"][0] for _, sp in spans}
            if len(lens) != 1 or min(lens) <= SKIP:      # identical solution tokens across cells (tests/test_render.py)
                skipped.append(it["item_id"])
                continue
            for (ro, la), (ids_, sp) in zip(CELLS, spans):
                s, e = sp["solution"]
                hs = model(input_ids=torch.tensor([ids_], device="cuda"), output_hidden_states=True).hidden_states
                H = torch.stack([h[0] for h in hs])          # [layers + 1, seq, hidden]
                M.append(H[:, s + SKIP:e].float().mean(1).to(torch.float16).cpu().numpy())
                L.append(H[:, e - 1].to(torch.float16).cpu().numpy())
                ids.append(it["item_id"])
                role.append(ro == "user")
                label.append(la == "other")
    gpu_s = time.time() - t
    M, L = np.stack(M), np.stack(L)
    ids, role, label = np.array(ids), np.array(role), np.array(label)
    np.savez_compressed(d / "acts.npz", mean=M, last=L, ids=ids, role_user=role, label_other=label)
    _, groups = np.unique(ids, return_inverse=True)
    res = {}
    for pos, X in (("mean", M), ("last", L)):
        res[pos] = {"role": [cv_auroc(X[:, k].astype(np.float32), role, groups) for k in range(X.shape[1])],
                    "label": [cv_auroc(X[:, k].astype(np.float32), label, groups) for k in range(X.shape[1])]}
    inv = all(abs(res[p][t][0] - 0.5) < 1e-9 for p in res for t in ("role", "label"))
    metrics = {"n_items": len(set(ids.tolist())), "rows": len(ids), "skipped_items": skipped, "layers": int(M.shape[1]),
               "gpu_s": gpu_s, "analysis_s": time.time() - t - gpu_s, "auroc": res, "layer0_invariant": inv,
               "acts_sha256": sha256_file(d / "acts.npz")}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__},
                   {"labels": json.dumps(R.LABELS), "instruction": R.INSTRUCTION})
    register(run_id, "done" if inv else "failed", block="h3c-acts", exp="h3c", model=args.model,
             **({} if inv else {"reason": "layer-0 invariant failed"}))
    print(json.dumps({"run_id": run_id, "n_items": metrics["n_items"], "layer0_invariant": inv,
                      "max role auroc (mean pos)": max(res["mean"]["role"]), "max label auroc (mean pos)": max(res["mean"]["label"])}))


if __name__ == "__main__":
    main()
