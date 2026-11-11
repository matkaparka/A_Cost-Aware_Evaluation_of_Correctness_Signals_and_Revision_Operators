"""Large run outputs that git ignores (over GitHub's file limits or excluded by .gitignore) are attached to a GitHub
Release of the private repository instead. This script gives each a unique asset name, copies it to a staging folder,
and records name, source path, size and sha256 in analysis/out/09_release_assets.json (committed), so the release can be
checked against the run manifests.
Usage: python analysis/p8_release_assets.py <staging dir>
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["results/p2/4B/p2-acts-dev-ext-4B-8360990/acts.npz", "results/main/4B/main-streams-test-4B-f15b933/eval_texts.jsonl",
         "results/p2/8B/p2-acts-dev-8B-3fd21bc/acts.npz", "results/p2/1.7B/p2-acts-dev-ext-1.7B-8360990/acts.npz",
         "results/p2/4B/p2-acts-dev-4B-f7e5011/acts.npz", "results/p2/1.7B/p2-acts-dev-1.7B-f7e5011/acts.npz"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    stage = Path(sys.argv[1])
    stage.mkdir(parents=True, exist_ok=True)
    out = []
    for rel in FILES:
        src = ROOT / rel
        name = "__".join(Path(rel).parts[1:])   # e.g. p2__4B__p2-acts-dev-ext-4B-8360990__acts.npz
        shutil.copy2(src, stage / name)
        out.append({"asset": name, "source": rel, "bytes": src.stat().st_size, "sha256": sha(src)})
    (ROOT / "analysis" / "out" / "09_release_assets.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
