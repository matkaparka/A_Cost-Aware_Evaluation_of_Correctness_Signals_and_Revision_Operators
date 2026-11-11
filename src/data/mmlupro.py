"""MMLU-Pro items for the independent validation block (docs/planning/validation_v2/plan.md §2.2; DEVIATIONS D18).

A public multiple-choice task written independently of this project: TIGER-Lab/MMLU-Pro (MIT), test split, pinned
revision. Only the non-computational categories are used (CATEGORIES); the others need arithmetic or are too mixed to
call non-mathematical. Some questions share a stem but differ in their options (none is an exact duplicate in these
categories); all are kept as items, and the canonical ID is a hash of the normalized stem, so variants of one stem form
one bootstrap cluster and fall in the same split. The split is seeded at the canonical-ID level per category, as in
src/data/build.py (dev fraction DEV_FRACTION); the processing order is a seeded hash per split.

The question q is the stem, the lettered options and one line asking for the option's letter; the main study's wrappers
(src/gen/render.py) are used around it unchanged, so the model is asked to put its final answer within \\boxed{}.
Outputs: data/processed/mmlupro/items.jsonl, build_manifest.json.
Usage: python -m src.data.mmlupro
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "processed" / "mmlupro"
DATASET, REVISION = "TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be"
CATEGORIES = ("law", "history", "philosophy", "psychology", "biology", "health")
DEV_FRACTION = 0.20
LETTERS = "ABCDEFGHIJ"
ASK = "Answer with the letter of the correct option."


def h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def render_question(item):
    opts = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(item["options"]))
    return f"{item['question'].strip()}\n\nOptions:\n{opts}\n\n{ASK}"


def load_items(split=None):
    rows = [json.loads(line) for line in open(OUT / "items.jsonl", encoding="utf-8")]
    return [r for r in rows if split in (None, "all") or r["split"] == split]


def build():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
    ds = pq.read_table(path).to_pylist()
    items = {}
    n_cat, stems = {}, {}
    for r in ds:
        if r["category"] not in CATEGORIES:
            continue
        n_cat[r["category"]] = n_cat.get(r["category"], 0) + 1
        cid = "mp" + h("mmlupro-canon-v1", normalize(r["question"]))[:12]
        stems[cid] = stems.get(cid, 0) + 1
        opts = [o for o in r["options"] if o is not None and str(o).strip() and str(o).strip() != "N/A"]
        if r["answer_index"] >= len(opts) or opts[r["answer_index"]] != r["options"][r["answer_index"]]:
            raise SystemExit(f"answer index does not survive option cleaning: {r['question_id']}")
        items[f"mmlupro-{r['question_id']}"] = {
            "item_id": f"mmlupro-{r['question_id']}", "question_id": r["question_id"], "canonical_id": cid,
            "category": r["category"], "src": r["src"], "question": r["question"], "options": opts,
            "gold": LETTERS[r["answer_index"]], "n_options": len(opts)}
    cat_of = {it["canonical_id"]: it["category"] for it in items.values()}
    split = {}
    for cat in CATEGORIES:
        ids = sorted((c for c in stems if cat_of[c] == cat), key=lambda c: h("mmlupro-split-v1", c))
        n_dev = round(DEV_FRACTION * len(ids))
        for k, c in enumerate(ids):
            split[c] = "dev" if k < n_dev else "test"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in ("dev", "test"):
        sel = [i for i, it in items.items() if split[it["canonical_id"]] == s]
        for k, i in enumerate(sorted(sel, key=lambda i: h("mmlupro-order-v1", s, i))):
            rows.append({**items[i], "split": s, "order": k})
    n_dup = sum(v - 1 for v in stems.values())
    with open(OUT / "items.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = hashlib.sha256((OUT / "items.jsonl").read_bytes()).hexdigest()
    man = {"dataset": DATASET, "revision": REVISION, "split_used": "test", "categories": list(CATEGORIES),
           "rows_in_categories": n_cat, "canonical_stems": len(stems), "items_sharing_a_stem_extra": n_dup, "items": len(rows),
           "by_split": {s: sum(r["split"] == s for r in rows) for s in ("dev", "test")},
           "by_category_split": {f"{c}/{s}": sum(r["category"] == c and r["split"] == s for r in rows)
                                 for c in CATEGORIES for s in ("dev", "test")},
           "dev_fraction": DEV_FRACTION, "seeds": ["mmlupro-canon-v1", "mmlupro-split-v1", "mmlupro-order-v1"],
           "items_sha256": sha}
    (OUT / "build_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    build()
