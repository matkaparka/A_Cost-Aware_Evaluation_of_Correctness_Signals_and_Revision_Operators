"""P1: download the study datasets at pinned revisions and export them as JSONL with a manifest (design §5.1).
Runs on the run host (the laptop cannot reach Hugging Face). Output: <out>/<name>/{rows.jsonl, manifest.json}.
Usage: python -m src.data.fetch --out /home/data_raw
"""
import argparse
import hashlib
import json
from pathlib import Path

# (name, repo, revision prefix from the design or None = resolve current main, preferred split)
SOURCES = [
    ("gsm8k", "openai/gsm8k", None, "main", "test"),
    ("math500", "HuggingFaceH4/MATH-500", None, None, "test"),
    ("gsm8k_sc", "kenhktsui/gsm8k_sc", "f9c0af7f", None, "test"),
    ("prm800k_sc", "kenhktsui/prm800k_sc", "62e7a103", None, "test"),
    ("scli5", "kenhktsui/scli5", "9b159157", None, "test"),
]


def resolve(api, repo, prefix):
    if prefix is None:
        return api.dataset_info(repo).sha
    for c in api.list_repo_commits(repo, repo_type="dataset"):
        if c.commit_id.startswith(prefix):
            return c.commit_id
    raise SystemExit(f"{repo}: no commit starts with {prefix}")


# Optional D-nat expansion (design §5.1, decided in P3): the full MATH test split, one config per subject. MATH-500 is a
# subset; src/data/expand.py removes it and every problem linked to the existing pool.
MATH_FULL = ("math_full", "EleutherAI/hendrycks_math", None,
             ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
              "prealgebra", "precalculus"), "test")


def fetch_math_full(api, out):
    from datasets import load_dataset
    name, repo, prefix, configs, split = MATH_FULL
    rev = resolve(api, repo, prefix)
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    n, cols = 0, None
    with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
        for config in configs:
            ds = load_dataset(repo, config, split=split, revision=rev)
            cols = ds.column_names
            for i, row in enumerate(ds):
                f.write(json.dumps({"source_index": n, "config": config, "config_index": i, **row},
                                   ensure_ascii=False, default=str) + "\n")
                n += 1
    sha = hashlib.sha256((d / "rows.jsonl").read_bytes()).hexdigest()
    man = {"name": name, "repo": repo, "revision": rev, "config": list(configs), "split": split,
           "rows": n, "columns": cols, "rows_sha256": sha}
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--math-full", action="store_true", help="fetch only the full MATH test split (P3 expansion)")
    args = ap.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    if args.math_full:
        fetch_math_full(api, args.out)
        return
    for name, repo, prefix, config, split in SOURCES:
        rev = resolve(api, repo, prefix)
        ds = load_dataset(repo, config, split=split, revision=rev)
        d = Path(args.out) / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                f.write(json.dumps({"source_index": i, **row}, ensure_ascii=False, default=str) + "\n")
        sha = hashlib.sha256((d / "rows.jsonl").read_bytes()).hexdigest()
        man = {"name": name, "repo": repo, "revision": rev, "config": config, "split": split,
               "rows": len(ds), "columns": ds.column_names, "rows_sha256": sha}
        (d / "manifest.json").write_text(json.dumps(man, indent=1))
        print(json.dumps(man))


if __name__ == "__main__":
    main()
