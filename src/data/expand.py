"""P3 option (design §5.1): expand the D-nat pool with MATH test problems outside MATH-500, without touching the
existing pool, split or orders (data/processed/*). Same rules as src/data/build.py, applied to the new problems only:

1. Identity: normalized text (NFKC, whitespace, case) and MinHash >= 0.9 on character 5-gram shingles.
2. Exclusion: a MATH test problem that matches any member of the existing pool (GSM8K, MATH-500, gsm8k_sc,
   prm800k_sc, scli5; exact or MinHash) is dropped. MATH-500 problems are dropped this way, and so is any problem
   already present through Self-Correction Bench, whatever its split, so no canonical problem gains a second D-nat
   item or changes split. Every drop is logged with the matched member.
3. Near-duplicates among the new problems are merged as in the original build: one canonical ID (so one split),
   every member stays an item (P3: the 12 merges are template variants with different numbers).
4. Gold: the content of the last \\boxed{...} of the MATH solution (the scorer's balanced-brace parser). Problems
   without one are dropped and counted. Check: on the MATH-500 problems found in the MATH test split, this gold is
   equivalent to MATH-500's `answer` field (agreement reported; the build stops below 98 %).
5. Split: a new stratum, math_ext, split by the same seeded rule (sorted by h("split-v1", id), the first 20 % dev).
   Processing order per split: h("order-v1", split, id), kept separate from the existing orders.

Inputs: data/raw/math_full (src/data/fetch.py --math-full), data/raw/<existing sources>, data/processed/split.json.
Outputs (data/processed/ext/): dnat_ext.jsonl, split_ext.json, dropped_ext.jsonl, merges_ext.jsonl,
build_ext_manifest.json.
Usage: python -m src.data.expand
"""
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from ..score.scorer import _boxed_spans, equivalent
from .build import DEV_FRACTION, MINHASH_THRESHOLD, OUT, h, load, minhash, normalize

EXT = OUT / "ext"
MIN_GOLD_AGREEMENT = 0.98


def last_boxed(solution):
    spans = _boxed_spans(solution)
    return spans[-1][2].strip() if spans else None


def main():
    EXT.mkdir(parents=True, exist_ok=True)
    rows, man = load("math_full")
    old = {n: load(n)[0] for n in ["gsm8k", "math500", "gsm8k_sc", "prm800k_sc", "scli5"]}
    old_members = []
    for name, rs in old.items():
        for r in rs:
            q = r.get("question") or r.get("problem")
            if name == "scli5":
                q = r["messages_error_injection_in_model"][0]["content"]
            old_members.append((name, r["source_index"], normalize(q), q))
    old_norm = {m[2]: m for m in old_members}
    old_sigs = np.stack([minhash(m[3]) for m in old_members])
    split_old = json.loads((OUT / "split.json").read_text())["split"]
    problems = {json.loads(l)["canonical_id"]: json.loads(l) for l in open(OUT / "problems.jsonl", encoding="utf-8")}
    cid_of = {tuple(mb): p["canonical_id"] for p in problems.values() for mb in p["members"]}

    # Gold check against MATH-500 on the overlap.
    m500 = {normalize(r["problem"]): r for r in old["math500"]}
    agree = [equivalent(last_boxed(r["solution"]) or "", m500[normalize(r["problem"])]["answer"])
             for r in rows if normalize(r["problem"]) in m500]
    agreement = sum(agree) / len(agree) if agree else None
    assert agreement is not None and agreement >= MIN_GOLD_AGREEMENT, f"gold extraction agrees on {agreement}"

    dropped, kept = [], []
    for r in rows:
        norm = normalize(r["problem"])
        hit, how, sim = old_norm.get(norm), "exact_normalized", None
        if hit is None:
            s = (old_sigs == minhash(r["problem"])).mean(axis=1)
            k = int(s.argmax())
            if s[k] >= MINHASH_THRESHOLD:
                hit, how, sim = old_members[k], "minhash", round(float(s[k]), 4)
        if hit is not None:
            cid = cid_of[(hit[0], hit[1])]
            dropped.append({"source_index": r["source_index"], "reason": "in existing pool", "how": how,
                            "minhash_jaccard": sim, "matched": [hit[0], hit[1]], "canonical_id": cid,
                            "split": split_old[cid]})
            continue
        gold = last_boxed(r["solution"])
        if not gold:
            dropped.append({"source_index": r["source_index"], "reason": "no boxed answer in the solution"})
            continue
        kept.append({**r, "norm": norm, "gold": gold})

    # Near-duplicates among the new problems: exact, then MinHash; the first member represents the group.
    sigs = np.stack([minhash(r["problem"]) for r in kept])
    rep = list(range(len(kept)))
    seen = {}
    for i, r in enumerate(kept):
        if r["norm"] in seen:
            rep[i] = seen[r["norm"]]
            continue
        seen[r["norm"]] = i
        if i:
            s = (sigs[:i] == sigs[i]).mean(axis=1)
            j = int(s.argmax())
            if s[j] >= MINHASH_THRESHOLD:
                rep[i] = rep[j]
    # As in the original build, merged problems share one canonical ID (hence one split) and each stays an item.
    merges = []
    for i, r in enumerate(kept):
        r["canonical_id"] = "p" + h(kept[rep[i]]["norm"])[:12]
        if rep[i] != i:
            merges.append({"kind": "merge", "how": "exact_normalized" if r["norm"] == kept[rep[i]]["norm"] else "minhash",
                           "a": ["math_full", kept[rep[i]]["source_index"]], "b": ["math_full", r["source_index"]]})
    items = kept
    ids = sorted({r["canonical_id"] for r in items})
    assert len(ids) == sum(rep[i] == i for i in range(len(kept))) and not set(ids) & set(problems), "canonical ID collision"

    order_ids = sorted(ids, key=lambda c: h("split-v1", c))
    n_dev = round(DEV_FRACTION * len(order_ids))
    split = {c: ("dev" if k < n_dev else "test") for k, c in enumerate(order_ids)}
    order = {s: sorted((c for c in split if split[c] == s), key=lambda c: h("order-v1", s, c)) for s in ("dev", "test")}

    out_rows = [{"item_id": f"dnat-mathext-{r['source_index']}", "canonical_id": r["canonical_id"], "dataset": "math_full",
                 "source_index": r["source_index"], "question": r["problem"], "gold": r["gold"], "subject": r["type"],
                 "level": r["level"], "stratum": "math_ext", "split": split[r["canonical_id"]]} for r in items]

    def dump(name, rs):
        with open(EXT / name, "w", encoding="utf-8", newline="\n") as f:
            for x in rs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    dump("dnat_ext.jsonl", out_rows)
    dump("dropped_ext.jsonl", dropped)
    dump("merges_ext.jsonl", merges)
    (EXT / "split_ext.json").write_text(json.dumps({"split": split, "order": order}, indent=0), encoding="utf-8", newline="\n")
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(EXT.glob("*.json*")) if p.name != "build_ext_manifest.json"}
    stats = {"math_test_rows": len(rows), "kept_items": len(out_rows), "canonical_ids": len(ids), "merges": len(merges),
             "items_by_split": dict(Counter(r["split"] for r in out_rows)),
             "dropped_by_reason": dict(Counter(d["reason"] for d in dropped)),
             "dropped_in_pool_by_split": dict(Counter(d["split"] for d in dropped if d["reason"] == "in existing pool")),
             "gold_agreement_with_math500": agreement, "gold_check_n": len(agree),
             "by_level_split": {f"{a}/{b}": n for (a, b), n in sorted(Counter((r["level"], r["split"]) for r in out_rows).items())}}
    (EXT / "build_ext_manifest.json").write_text(json.dumps(
        {"source": man, "seeds": ["split-v1", "order-v1"], "rules": "src/data/expand.py docstring", "stats": stats,
         "files_sha256": sha}, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
