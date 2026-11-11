"""Score cached activations with the frozen Exp 2 probe (H2a; prereg §6): P(correct) per item, written as a small jsonl
so the activations themselves can stay on the run host. Runs only after the prereg freeze on test.
Output: <run>/probe_scores.jsonl with {item_id, p_correct}; the manifest records the probe's sha256.
Usage: python -m src.probe.score --model 4B --probe <probe.npz> --acts <acts.npz> [<acts.npz> ...] --split test
"""
import argparse
import json

import numpy as np

from ..runlib import MODELS, register, run_dir, sha256_file, sha256_text, write_manifest
from .train import score_frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--acts", required=True, nargs="+")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    args = ap.parse_args()
    cfg = {**vars(args), "probe_sha256": sha256_file(args.probe), "acts_sha256": [sha256_file(a) for a in args.acts]}
    run_id, d = run_dir("main" if args.split == "test" else "p2", args.model, f"probe-scores-{args.split}")
    register(run_id, "running", block=f"probe-scores-{args.split}", exp="exp2", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    n = 0
    with open(d / "probe_scores.jsonl", "w", encoding="utf-8") as f:
        for a in args.acts:
            z = np.load(a)
            for i, p in zip(z["ids"].tolist(), score_frozen(args.probe, z["X"]).tolist()):
                f.write(json.dumps({"item_id": i, "p_correct": p}) + "\n")
                n += 1
    (d / "metrics.json").write_text(json.dumps({"n": n}))
    from transformers import AutoTokenizer
    write_manifest(d, run_id, args.model, AutoTokenizer.from_pretrained(MODELS[args.model]), cfg,
                   {"name": "numpy", "version": np.__version__}, {})
    register(run_id, "done", block=f"probe-scores-{args.split}", exp="exp2", model=args.model)
    print(json.dumps({"run_id": run_id, "n": n}))


if __name__ == "__main__":
    main()
