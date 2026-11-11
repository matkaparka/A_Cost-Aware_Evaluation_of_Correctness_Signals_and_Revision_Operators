"""P1 step 1 (design §5.2): canonical problem IDs, cross-source links, near-duplicate merges, the seeded
dev/test split, seeded processing orders, and the fixed Exp 5 spillover and KL sets. Then the D-nat and
D-inj item tables (design §5.1, §5.4).

Inputs: data/raw/<name>/{rows.jsonl, manifest.json} (src/data/fetch.py).
Outputs (data/processed/): problems.jsonl, merges.jsonl, split.json, dnat.jsonl, dinj.jsonl, build_manifest.json.
Usage: python -m src.data.build
"""
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
DEV_FRACTION = 0.20
MINHASH_PERMS, SHINGLE, MINHASH_THRESHOLD = 128, 5, 0.9
SPILLOVER_N, KL_N = 50, 50


def h(*parts):
    """Seeded hash used for every derived random choice: sha256 of the joined parts (design §5.5)."""
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


_rng = np.random.default_rng(int(hashlib.sha256(b"minhash-v1").hexdigest()[:8], 16))
_A = _rng.integers(1, 2**63, MINHASH_PERMS, dtype=np.uint64) | np.uint64(1)   # odd multipliers
_B = _rng.integers(0, 2**63, MINHASH_PERMS, dtype=np.uint64)


def minhash(text):
    """MinHash over character 5-gram shingles; permutations by multiply-add mod 2^64."""
    s = normalize(text)
    shingles = {s[i:i + SHINGLE] for i in range(max(1, len(s) - SHINGLE + 1))}
    base = np.array([int.from_bytes(hashlib.blake2b(sh.encode(), digest_size=8).digest(), "little")
                     for sh in shingles], dtype=np.uint64)
    with np.errstate(over="ignore"):
        return (base[:, None] * _A[None, :] + _B[None, :]).min(axis=0)


def load(name):
    rows = [json.loads(line) for line in open(RAW / name / "rows.jsonl", encoding="utf-8")]
    man = json.loads((RAW / name / "manifest.json").read_text())
    return rows, man


def gsm8k_gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {n: load(n) for n in ["gsm8k", "math500", "gsm8k_sc", "prm800k_sc", "scli5"]}

    # 1. Members: one entry per source row, with the question text used for identity.
    members = []
    for name, (rows, _) in data.items():
        for r in rows:
            q = r.get("question") or r.get("problem")
            if name == "scli5":
                q = r["messages_error_injection_in_model"][0]["content"]
            members.append({"dataset": name, "source_index": r["source_index"], "q": q, "norm": normalize(q)})

    # 2. Exact normalized matches form groups; then MinHash >= threshold merges remaining near-duplicates.
    parent = list(range(len(members)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    expected_links = {frozenset({"gsm8k", "gsm8k_sc"}), frozenset({"math500", "prm800k_sc"})}
    merges = []

    def union(i, j, how, sim=None):
        if find(i) == find(j):
            return
        a, b = members[i], members[j]
        kind = "link" if frozenset({a["dataset"], b["dataset"]}) in expected_links else "merge"
        merges.append({"kind": kind, "how": how, "a": [a["dataset"], a["source_index"]],
                       "b": [b["dataset"], b["source_index"]], "minhash_jaccard": sim})
        parent[find(j)] = find(i)

    by_norm = defaultdict(list)
    for i, m in enumerate(members):
        by_norm[m["norm"]].append(i)
    for idx in by_norm.values():
        for j in idx[1:]:
            union(idx[0], j, "exact_normalized")

    sigs = np.stack([minhash(m["q"]) for m in members])
    for i in range(len(members)):
        sim = (sigs[i + 1:] == sigs[i]).mean(axis=1)
        for k in np.nonzero(sim >= MINHASH_THRESHOLD)[0]:
            union(i, i + 1 + int(k), "minhash", round(float(sim[k]), 4))

    groups = defaultdict(list)
    for i in range(len(members)):
        groups[find(i)].append(i)

    # 3. Canonical problems. The representative text is the D-nat member if any (GSM8K / MATH-500).
    rank = {"gsm8k": 0, "math500": 1, "gsm8k_sc": 2, "prm800k_sc": 3, "scli5": 4}
    problems, member_cid = [], {}
    for idx in groups.values():
        idx.sort(key=lambda i: (rank[members[i]["dataset"]], members[i]["source_index"]))
        rep = members[idx[0]]
        cid = "p" + h(rep["norm"])[:12]
        datasets = sorted({members[i]["dataset"] for i in idx}, key=rank.get)
        stratum = ("gsm8k" if {"gsm8k", "gsm8k_sc"} & set(datasets) else
                   "math" if {"math500", "prm800k_sc"} & set(datasets) else "scli5")
        problems.append({"canonical_id": cid, "stratum": stratum, "datasets": datasets,
                         "members": [[members[i]["dataset"], members[i]["source_index"]] for i in idx]})
        for i in idx:
            member_cid[(members[i]["dataset"], members[i]["source_index"])] = cid
    assert len({p["canonical_id"] for p in problems}) == len(problems), "canonical ID collision"

    # 4. Seeded split per stratum at the canonical-ID level; seeded processing order per split.
    split = {}
    for stratum in sorted({p["stratum"] for p in problems}):
        ids = sorted((p["canonical_id"] for p in problems if p["stratum"] == stratum), key=lambda c: h("split-v1", c))
        n_dev = round(DEV_FRACTION * len(ids))
        for k, c in enumerate(ids):
            split[c] = "dev" if k < n_dev else "test"
    order = {s: sorted((c for c in split if split[c] == s), key=lambda c: h("order-v1", s, c)) for s in ("dev", "test")}

    # 5. Fixed Exp 5 sets (design §5.2 step 4): spillover from D-nat test, KL probe set from D-nat dev.
    dnat_ids = {member_cid[(d, r["source_index"])] for d in ("gsm8k", "math500") for r in data[d][0]}
    spill = sorted((c for c in dnat_ids if split[c] == "test"), key=lambda c: h("spillover-v1", c))[:SPILLOVER_N]
    kl = sorted((c for c in dnat_ids if split[c] == "dev"), key=lambda c: h("kl-v1", c))[:KL_N]

    # 6. Item tables.
    dnat = []
    for r in data["gsm8k"][0]:
        cid = member_cid[("gsm8k", r["source_index"])]
        dnat.append({"item_id": f"dnat-gsm8k-{r['source_index']}", "canonical_id": cid, "dataset": "gsm8k",
                     "source_index": r["source_index"], "question": r["question"], "gold": gsm8k_gold(r["answer"]),
                     "split": split[cid]})
    for r in data["math500"][0]:
        cid = member_cid[("math500", r["source_index"])]
        dnat.append({"item_id": f"dnat-math500-{r['source_index']}", "canonical_id": cid, "dataset": "math500",
                     "source_index": r["source_index"], "question": r["problem"], "gold": r["answer"],
                     "subject": r["subject"], "level": r["level"], "unique_id": r["unique_id"], "split": split[cid]})

    def final_turn(msgs):
        return [m for m in msgs if m["role"] == "assistant"][-1]["content"]

    dinj = []
    for r in data["gsm8k_sc"][0]:
        cid = member_cid[("gsm8k_sc", r["source_index"])]
        dinj.append({"item_id": f"dinj-gsm8k_sc-{r['source_index']}", "canonical_id": cid, "dataset": "gsm8k_sc",
                     "source_index": r["source_index"], "question": r["question"], "gold": gsm8k_gold(r["answer"]),
                     "solution_with_answer": final_turn(r["messages_error_injection_in_model_aca"]),
                     "solution_no_answer": final_turn(r["messages_error_injection_in_model_bca"]),
                     "injected_answer": r["incorrect_answer"], "steps": r["reasoning_steps_with_one_mistake"],
                     "mistake_step": r["mistake_step"], "mistake_type": r["type_of_mistake"], "split": split[cid]})
    for r in data["prm800k_sc"][0]:
        cid = member_cid[("prm800k_sc", r["source_index"])]
        dinj.append({"item_id": f"dinj-prm800k_sc-{r['source_index']}", "canonical_id": cid, "dataset": "prm800k_sc",
                     "source_index": r["source_index"], "question": r["question"], "gold": r["ground_truth_answer"],
                     "solution_with_answer": final_turn(r["messages_error_injection_in_model_aca"]),
                     "solution_no_answer": final_turn(r["messages_error_injection_in_model_bca"]),
                     "injected_answer": r["pre_generated_answer"], "n_steps": r["n_reasoning_step"],
                     "mistake_step": None, "split": split[cid]})
    for r in data["scli5"][0]:
        cid = member_cid[("scli5", r["source_index"])]
        msgs = r["messages_error_injection_in_model"]
        dinj.append({"item_id": f"dinj-scli5-{r['source_index']}", "canonical_id": cid, "dataset": "scli5",
                     "source_index": r["source_index"], "question": msgs[0]["content"], "gold": r["correct_answer"],
                     "solution_with_answer": final_turn(msgs), "solution_no_answer": None, "task_type": r["type"],
                     "injected_answer": None, "mistake_step": None, "split": split[cid]})

    def dump(name, rows):
        with open(OUT / name, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("problems.jsonl", sorted(problems, key=lambda p: p["canonical_id"]))
    dump("merges.jsonl", merges)
    dump("dnat.jsonl", dnat)
    dump("dinj.jsonl", dinj)
    (OUT / "split.json").write_text(json.dumps({"split": split, "order": order, "spillover_test": spill,
                                                "kl_dev": kl}, indent=0), encoding="utf-8", newline="\n")
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.json*")) if p.name != "build_manifest.json"}
    man = {"sources": {n: data[n][1] for n in data}, "dev_fraction": DEV_FRACTION, "minhash": {
        "perms": MINHASH_PERMS, "shingle": SHINGLE, "threshold": MINHASH_THRESHOLD},
        "seeds": ["split-v1", "order-v1", "spillover-v1", "kl-v1"], "files_sha256": sha}
    (OUT / "build_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")
    counts = defaultdict(int)
    for p in problems:
        counts[(p["stratum"], split[p["canonical_id"]])] += 1
    kinds = defaultdict(int)
    for m in merges:
        kinds[f"{m['kind']}/{m['how']}"] += 1
    print(json.dumps({"members": len(members), "problems": len(problems), "unions_logged": dict(kinds),
                      "problems_by_stratum_split": {f"{a}/{b}": n for (a, b), n in sorted(counts.items())},
                      "dnat": len(dnat), "dinj": len(dinj), "spillover": len(spill), "kl": len(kl)}, indent=1))


if __name__ == "__main__":
    main()
