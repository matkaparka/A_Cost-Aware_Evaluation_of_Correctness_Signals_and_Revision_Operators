"""A0 (plan §A0): common gold-blind labels for every math model and operator.

The ACL revision re-scored Qwen3-4B's a0, F2 and F2-scratch with one gold-blind extraction rule
(paper/1区/acl2027/protocol/math_vote_scoring_sensitivity.md). This extends the same rule to F0 and to Qwen3-1.7B,
Qwen3-8B and Tulu-3-8B, so that votes, operators and KEEP are scored alike in every math setting.

Two phases, as the ACL rule requires: (1) extract every candidate without the gold and freeze the extraction file
(its hash is recorded); (2) score the frozen candidates with equivalent(pred, gold). Historical labels are kept
beside the new ones and never overwritten. For Qwen3-4B the recomputed a0/F2/F2-scratch candidates and labels must
equal the ACL revision's stored ones (asserted).

Usage: python -B analysis/arr/a0_common_labels.py
-> analysis/out/arr/math_common_extractions.jsonl.gz (gold-blind, frozen first)
   analysis/out/arr/math_common_labels.jsonl.gz, analysis/out/arr/a0_common_labels.json, a0_common_labels_manifest.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ACL, J1, MODELS_MATH, OPS, OUT, REPO, Equivalence, candidate, dump, manifest, pin,  # noqa: E402
                    plan_core_hash, plan_hash, rows, rows_gz, run_dir, sha, unique, write_rows_gz)

KEEP_FIELDS = ("pred", "status", "raw_extraction_status", "out_tokens", "finish_reason")


def verified(run, name):
    d = run_dir(run)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert sha(d / name) == man["files"][name], ("manifest checksum", run, name)
    return pin(d / name)


def main():
    plan_hash()
    table = pin(J1 / "artifacts" / "math" / "item_evaluations.jsonl")
    data = unique(r for p in (pin(REPO / "data/processed/dnat.jsonl"), pin(REPO / "data/processed/ext/dnat_ext.jsonl"))
                  for r in rows(p))
    per_model = {}
    for r in rows(table):
        cond = per_model.setdefault(r["model"], {}).setdefault(r["condition"], {})
        assert r["item_id"] not in cond
        cond[r["item_id"]] = r
    eq = Equivalence()
    records = []
    try:
        for model in MODELS_MATH:
            per = per_model[model]
            ids = sorted(per["F2"])
            assert all(set(per[c]) == set(ids) for c in OPS), model
            initial = {}
            for run in sorted({per["F2"][i]["initial_run"] for i in ids}):
                for r in rows(verified(run, "initial.jsonl")):
                    if r["split"] == "test":
                        assert r["item_id"] not in initial
                        initial[r["item_id"]] = r
            assert set(ids) <= set(initial), model
            gens = {}
            for run in sorted({per[c][i]["source_run"] for c in OPS for i in ids}):
                g = {}
                for r in rows(verified(run, "generations.jsonl")):
                    assert r["prompt_key"] not in g
                    g[r["prompt_key"]] = r
                gens[run] = g
            for n_done, i in enumerate(ids):
                a, d = initial[i], data[i]
                assert d["split"] == "test" and d["canonical_id"] == a["canonical_id"]
                out_tokens = per["F2"][i]["initial_output_tokens"]
                if "a0_token_ids" in a:
                    assert len(a["a0_token_ids"]) == out_tokens
                rec = {"model": model, "item_id": i, "canonical_id": a["canonical_id"], "dataset": a["dataset"],
                       "initial": {**{k: v for k, v in candidate(a["a0"], a["finish_reason"], out_tokens, eq).items()
                                      if k in KEEP_FIELDS}, "hist_correct": int(a["correct"])}}
                for c in OPS:
                    op = per[c][i]
                    assert int(a["correct"]) == op["y0"] and op["canonical_id"] == a["canonical_id"]
                    gen = gens[op["source_run"]][op["final_prompt_key"]]
                    assert gen["out_tokens"] == op["turn2_output_tokens"], (model, c, i)
                    assert gen["finish_reason"] == op["final_finish_reason"], (model, c, i)
                    cand = candidate(gen["text"], gen["finish_reason"], gen["out_tokens"], eq)
                    rec[c] = {**{k: v for k, v in cand.items() if k in KEEP_FIELDS}, "hist_correct": int(op["y1"])}
                records.append(rec)
                if n_done % 1000 == 0:
                    print(f"{model}: extracted {n_done}/{len(ids)}; symbolic calls {eq.calls}", flush=True)
        frozen = OUT / "math_common_extractions.jsonl.gz"
        write_rows_gz(frozen, records)
        frozen_hash = sha(frozen)
        print("Extractions frozen; scoring against gold", flush=True)
        for n_done, rec in enumerate(records):
            gold = data[rec["item_id"]]["gold"]
            for c in ("initial",) + OPS:
                e = rec[c]
                e["correct"] = int(e["status"] == "ok" and eq(e["pred"], gold))
            if n_done % 2000 == 0:
                print(f"scored {n_done}/{len(records)}; symbolic calls {eq.calls}", flush=True)
        guards, seeded, calls = list(eq.guards), eq.seeded, eq.calls
    finally:
        eq.close()
    assert sha(frozen) == frozen_hash
    # Qwen3-4B: must reproduce the ACL revision's same-rule rows exactly.
    acl = unique(rows(pin(ACL / "artifacts/math_vote/same_rule_per_item.jsonl")))
    checked = 0
    for rec in records:
        if rec["model"] != "4B":
            continue
        old = acl[rec["item_id"]]
        for c in ("initial", "F2", "F2-scratch"):
            for k in ("pred", "status"):
                assert rec[c][k] == old[c][k], ("4B parity", rec["item_id"], c, k)
            assert rec[c]["correct"] == old[c]["new_correct"], ("4B parity", rec["item_id"], c)
            assert rec[c]["hist_correct"] == old[c]["original_correct"], ("4B parity", rec["item_id"], c)
            checked += 1
    assert checked == 3 * len(acl)
    write_rows_gz(OUT / "math_common_labels.jsonl.gz", records)
    summary = {"plan_sha256_lf": plan_hash(), "plan_core_sha256": plan_core_hash(), "extractions_sha256_before_gold": frozen_hash,
               "acl_4B_parity_checks": checked, "symbolic_calls": calls, "cache_seeded_from_acl": seeded,
               "guards": guards, "review_required": bool(guards), "models": {}}
    for model in MODELS_MATH:
        recs = [r for r in records if r["model"] == model]
        entry = {"n": len(recs)}
        for c in ("initial",) + OPS:
            hist = sum(r[c]["hist_correct"] for r in recs)
            new = sum(r[c]["correct"] for r in recs)
            changed = Counter((r[c]["hist_correct"], r[c]["correct"]) for r in recs if r[c]["hist_correct"] != r[c]["correct"])
            entry[c] = {"historical_correct": hist, "common_correct": new,
                        "historical_acc_pct": 100 * hist / len(recs), "common_acc_pct": 100 * new / len(recs),
                        "changed_hist_to_common": {f"{a}->{b}": v for (a, b), v in sorted(changed.items())},
                        "status": dict(Counter(r[c]["status"] for r in recs))}
        summary["models"][model] = entry
    dump(OUT / "a0_common_labels.json", summary)
    write_rows_gz(OUT / "math_common_label_changes.jsonl.gz",
                  [r for r in records if any(r[c]["hist_correct"] != r[c]["correct"] for c in ("initial",) + OPS)])
    manifest("a0_common_labels", {"outputs": {p: sha(OUT / p) for p in (
        "math_common_extractions.jsonl.gz", "math_common_labels.jsonl.gz", "a0_common_labels.json")}})
    print(json.dumps({m: {c: (v[c]["common_correct"], v[c]["historical_correct"]) for c in ("initial",) + OPS}
                      for m, v in summary["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
