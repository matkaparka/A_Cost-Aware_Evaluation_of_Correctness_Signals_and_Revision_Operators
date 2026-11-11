"""H3c with folds grouped by canonical problem (second review, item G; DEVIATIONS D18). Exploratory.

src/probe/h3c.py grouped its cross-validation folds by item ID, so the four role x label prompts of one item share a
fold. Some D-inj items share a canonical problem, so variants of one problem could fall in different folds. This
script reads the cached activations of the H3c run (acts.npz stays on the run host) and recomputes the same
out-of-fold AUROCs (same probe, C, folds and seed: src/probe/h3c.cv_auroc) with groups = canonical_id from
data/processed/dinj.jsonl. CPU only. The original run and its results are kept; this writes a new run directory.
Usage: python -m src.probe.h3c_regroup --run <h3c run dir>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..runlib import REPO, register, run_dir, sha256_file, sha256_text
from .h3c import cv_auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the H3c run directory with acts.npz")
    args = ap.parse_args()
    src = Path(args.run)
    canon = {json.loads(line)["item_id"]: json.loads(line)["canonical_id"]
             for line in open(REPO / "data" / "processed" / "dinj.jsonl", encoding="utf-8")}
    cfg = {"source_run": src.name, "source_acts_sha256": sha256_file(src / "acts.npz"), "groups": "canonical_id"}
    run_id, d = run_dir("fu", "4B", "h3c-regroup-test")
    register(run_id, "running", block="h3c-regroup", exp="h3c", model="4B", config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    z = np.load(src / "acts.npz")
    ids = z["ids"].astype(str)
    _, groups = np.unique([canon[i] for i in ids], return_inverse=True)
    role, label = z["role_user"], z["label_other"]
    res = {}
    for pos in ("mean", "last"):
        X = z[pos]
        res[pos] = {"role": [cv_auroc(X[:, k].astype(np.float32), role, groups) for k in range(X.shape[1])],
                    "label": [cv_auroc(X[:, k].astype(np.float32), label, groups) for k in range(X.shape[1])]}
    inv = all(abs(res[p][t][0] - 0.5) < 1e-9 for p in res for t in ("role", "label"))
    metrics = {"n_items": len(set(ids.tolist())), "n_canonical": int(groups.max() + 1), "rows": len(ids),
               "analysis_s": time.time() - t0, "auroc": res, "layer0_invariant": inv, **cfg}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    man = {"run_id": run_id, "code_commit": __import__("subprocess").run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                                          text=True, cwd=REPO).stdout.strip(),
           "files": {f: sha256_file(d / f) for f in ("metrics.json", "config.json")}}
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    register(run_id, "done" if inv else "failed", block="h3c-regroup", exp="h3c", model="4B")
    print(json.dumps({"run_id": run_id, "n_canonical": metrics["n_canonical"], "layer0_invariant": inv}))


if __name__ == "__main__":
    main()
