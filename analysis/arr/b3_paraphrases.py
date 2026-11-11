"""B3 of the ARR plan (docs/planning/arr_round/plan.md §B, B3; Amendment 2): prompt paraphrases of F2 and F2-scratch.

Runs (Amendment 2): `ops --wording p1|p2 --conds F2,F2-scratch` on math test for Qwen3-4B, 8B and Tulu-3-8B. Only the
first turn changes; the second turn is the stored one. The stored main-wording outputs are the third wording: the
parity check on the instance found the main-wording operator outputs byte-identical to the stored ones (100/100), so
no main-wording reruns were added.

Labels: the common gold-blind rule (A0), two phases: every final-turn extraction is written and hashed before any
gold is read. Costs: generated tokens of both turns.

Analysis (plan B3):
- the operator-preference contrast F2 − F2-scratch for each wording, for ALL and at 10 % and 20 % by each signal;
- the spread across the three wordings;
- the sign rule: "the preference flips with the model" survives only if the sign of the contrast is the same under
  all three wordings for each model (and the signs differ between models). Qwen3-1.7B has no paraphrase runs.

Usage: python -B analysis/arr/b3_paraphrases.py  (after b1_samples.py, whose samples give agreement and margin)
-> analysis/out/arr/b3_extractions.jsonl.gz (frozen before gold), b3_labels.jsonl.gz, b3_paraphrases.json,
   b3_paraphrases_manifest.json; reports/arr_b3.md
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1a2_replays as A  # noqa: E402
import b1_samples as B1  # noqa: E402
from common import (J1, OUT, REPO, Boot, Equivalence, candidate, dump, guard_check, manifest, pin, plan_core_hash, plan_hash,  # noqa: E402
                    rows, rows_gz, sha, write_rows_gz)

MODELS = ("4B", "8B", "T8B")
WORDINGS = ("main", "p1", "p2")
CONDS = ("F2", "F2-scratch")


def verified(d, name):
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert sha(d / name) == man["files"][name], ("manifest checksum", d.name, name)
    return pin(d / name), man


def ops_run(model, wording, pool):
    """Final-turn outputs of one paraphrase run (all chunks), hash-checked: {(item, cond): record} or (None, reason)."""
    root = REPO / "results" / "arr" / "b3" / model
    pat = re.compile(rf"^b3-ops-test-{wording}-F2\+F2-scratch(?:-c(\d+))?-{re.escape(model)}-([0-9a-f]{{7}})$")
    by_commit = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        m = pat.match(d.name)
        if m and (d / "manifest.json").exists():
            by_commit.setdefault(m.group(2), {})[None if m.group(1) is None else int(m.group(1))] = d
    if not by_commit:
        return None, "no run synced"
    assert len(by_commit) == 1, ("B3 runs from more than one code commit (CLAUDE.md rule 8)", model, wording, sorted(by_commit))
    (commit, chunks), = by_commit.items()
    out, sizes = {}, set()
    for c, d in sorted(chunks.items(), key=lambda kv: -1 if kv[0] is None else kv[0]):
        cfg = json.loads(verified(d, "config.json")[0].read_text(encoding="utf-8"))
        assert (cfg["stage"], cfg["model"], cfg["split"], cfg["wording"], cfg["conds"]) == ("ops", model, "test", wording, "F2,F2-scratch"), d.name
        assert cfg["chunk_index"] == (c or 0) and bool(cfg["chunk_size"]) == (c is not None), d.name
        path_g, man = verified(d, "generations.jsonl")
        assert man["env"].get("VLLM_BATCH_INVARIANT") == "1" and man["code_commit"].startswith(commit), d.name
        sizes.add(cfg["chunk_size"])
        gens = {g["prompt_key"]: g for g in rows(path_g)}
        for r in rows(verified(d, "rows.jsonl")[0]):
            if r["cond"] not in CONDS:
                continue
            key = (r["item_id"], r["cond"])
            assert key not in out, ("item in two chunks", model, wording, key)
            g = gens[r["prompt_key"]]
            assert g["out_tokens"] == r["out_tokens"] and g["finish_reason"] == r["finish_reason"]
            out[key] = {"text": g["text"], "finish_reason": g["finish_reason"], "out_tokens": int(r["out_tokens"]),
                        "prompt_tokens": int(r["prompt_tokens"]), "turn1_out_tokens": int(r["turn1_out_tokens"]),
                        "turn1_prompt_tokens": int(r["turn1_prompt_tokens"])}
    if None in chunks:
        assert len(chunks) == 1
    else:
        need = math.ceil(len(pool) / sizes.pop())
        if sorted(chunks) != list(range(need)):
            return None, f"{len(chunks)} of {need} chunks synced"
    if set(out) != {(i, c) for i in pool for c in CONDS}:
        return None, "the run does not cover every item and condition"
    return out, {"run_commit": commit, "runs": [chunks[c].name for c in sorted(chunks, key=lambda c: -1 if c is None else c)]}


def fmt(ci):
    return f"{ci['estimate']:+.2f} [{ci['lo']:+.2f}, {ci['hi']:+.2f}]"


def sign(x, tol=1e-12):
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def verdicts(per_model):
    """Sign rule per contrast quantity: the same nonzero sign under all three wordings within each model, and opposite
    signs between at least two models."""
    out = {}
    quantities = sorted({q for v in per_model.values() for q in v})
    for q in quantities:
        signs = {m: [sign(v[q][w]) for w in WORDINGS] for m, v in per_model.items() if q in v}
        consistent = {m: len(set(s)) == 1 and s[0] != 0 for m, s in signs.items()}
        model_sign = {m: s[0] for m, s in signs.items() if consistent[m]}
        flip = len(set(model_sign.values())) > 1
        out[q] = {"signs": signs, "consistent_within_model": consistent,
                  "flip_survives": all(consistent.values()) and flip,
                  "reason": ("consistent within every model and opposite between models" if all(consistent.values()) and flip else
                             "a model's sign changes with the wording" if not all(consistent.values()) else
                             "the same sign for every model: no flip")}
    return out


def report(res):
    L = ["# ARR round, B3: prompt paraphrases of F2 and F2-scratch", "",
         f"Generated by `analysis/arr/b3_paraphrases.py` (plan `docs/planning/arr_round/plan.md`, core sha256 "
         f"`{res['plan_core_sha256'][:16]}…`). Post hoc and exploratory (DEVIATIONS D20); added in revision.",
         "Δ in percentage points with paired 95 % cluster-bootstrap intervals (4,000 replicates, seed 20261110). Common "
         "gold-blind labels. main: the stored wording; p1, p2: the paraphrases of Amendment 2 (first turn only). Qwen3-1.7B "
         "has no paraphrase runs.", ""]
    missing = {m: v for m, v in res["status"].items() if v != "complete"}
    if missing:
        L += ["Not yet complete: " + "; ".join(f"math {m}: TODO(missing: B3 runs, {why})" for m, why in missing.items()) + ".", ""]
    L += ["## F2 − F2-scratch (ALL) under each wording", "",
          "| Setting | Wording | F2 ALL Δ vs KEEP | F2-scratch ALL Δ vs KEEP | F2 − F2-scratch | F2 tok/item | F2-scratch tok/item |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for s, v in res["settings"].items():
        for w in WORDINGS:
            e = v["wordings"][w]
            L.append(f"| {s} | {w} | {fmt(e['F2 ALL'])} | {fmt(e['F2-scratch ALL'])} | {fmt(e['contrast ALL'])} | "
                     f"{e['tokens']['F2']:.0f} | {e['tokens']['F2-scratch']:.0f} |")
    L += ["", "## F2 − F2-scratch at 10 % and 20 % by each signal (same items for both operators)", "",
          "| Setting | Signal | Share | main | p1 | p2 | Spread (max − min) |", "|---|---|---:|---:|---:|---:|---:|"]
    for s, v in res["settings"].items():
        for q, sp in v["spread"].items():
            if q == "contrast ALL" or not q.startswith("contrast "):
                continue
            sig, share = q[len("contrast "):].rsplit(" ", 1)
            cells = [fmt(v["wordings"][w]["selective"][q]) for w in WORDINGS]
            L.append(f"| {s} | {sig} | {share} | " + " | ".join(cells) + f" | {sp:.2f} |")
    L += ["", "Spread of the ALL quantities across the three wordings (max − min, pp): " + "; ".join(
        f"{s}: F2 ALL {v['spread']['F2 ALL']:.2f}, F2-scratch ALL {v['spread']['F2-scratch ALL']:.2f}, contrast {v['spread']['contrast ALL']:.2f}"
        for s, v in res["settings"].items()) + ".", ""]
    L += ["## Sign rule: does \"the preference flips with the model\" survive?", "",
          "| Contrast | Signs (main, p1, p2) per model | Verdict |", "|---|---|---|"]
    for q, v in res["sign_rule"].items():
        signs = "; ".join(f"{m}: " + ", ".join({1: "+", -1: "−", 0: "0"}[x] for x in s) for m, s in v["signs"].items())
        L.append(f"| {q} | {signs} | {'survives' if v['flip_survives'] else 'does not survive'} ({v['reason']}) |")
    L += ["", "Signs are those of the point estimates; the intervals above show which of them are uncertain.", ""]
    (REPO / "reports" / "arr_b3.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def main():
    plan = plan_hash()
    a0 = json.loads(pin(OUT / "a0_common_labels.json").read_text(encoding="utf-8"))
    assert a0["plan_core_sha256"] == plan_core_hash() and not a0["guards"]
    labels = list(rows_gz(pin(OUT / "math_common_labels.jsonl.gz")))
    table = {}
    for r in rows(pin(J1 / "artifacts/math/item_evaluations.jsonl")):
        table.setdefault(r["model"], {}).setdefault(r["condition"], {})[r["item_id"]] = r
    pops = A.load_math(table, labels)
    status, runs = {}, {}
    for model in MODELS:
        got, why = {}, None
        for w in ("p1", "p2"):
            out, info = ops_run(model, w, pops[model]["ids"])
            if out is None:
                why = f"{w}: {info}"
                break
            got[w] = (out, info)
        status[model] = "complete" if why is None else why
        if why is None:
            runs[model] = got
    eq = Equivalence()
    try:
        extractions = []
        for model, got in runs.items():
            for w, (out, _) in got.items():
                for (i, cond), g in sorted(out.items()):
                    c = candidate(g["text"], g["finish_reason"], g["out_tokens"], eq)
                    extractions.append({"model": model, "wording": w, "cond": cond, "item_id": i,
                                        **{f: c[f] for f in ("pred", "status", "raw_extraction_status", "finish_reason")},
                                        "out_tokens": g["out_tokens"] + g["turn1_out_tokens"],
                                        "prompt_tokens": g["prompt_tokens"] + g["turn1_prompt_tokens"]})
        frozen = OUT / "b3_extractions.jsonl.gz"
        write_rows_gz(frozen, extractions)
        frozen_hash = sha(frozen)
        print(f"extractions frozen: {len(extractions)} rows, sha256 {frozen_hash[:16]}", flush=True)
        gold = {r["item_id"]: r["gold"] for p in (pin(REPO / "data/processed/dnat.jsonl"),
                                                  pin(REPO / "data/processed/ext/dnat_ext.jsonl")) for r in rows(p)}
        for e in extractions:
            e["correct"] = int(e["status"] == "ok" and eq(e["pred"], gold[e["item_id"]]))
        write_rows_gz(OUT / "b3_labels.jsonl.gz", extractions)
        res = {"plan_sha256_lf": plan, "plan_core_sha256": plan_core_hash(), "extractions_sha256_before_gold": frozen_hash,
               "status": status, "runs": {m: {w: v[1] for w, v in got.items()} for m, got in runs.items()}, "settings": {}}
        per_model = {}
        for model in runs:
            P = pops[model]
            if model == "4B":
                if B1.attach(P, model, eq) is None:
                    A.add_math_samples(P, eq)
            else:
                B1.attach(P, model, eq)
            lab = {(e["wording"], e["cond"], e["item_id"]): e for e in extractions if e["model"] == model}
            ops = {"main": {c: (P["ops"][c]["y1"], P["ops"][c]["gen"]) for c in CONDS}}
            for w in ("p1", "p2"):
                ops[w] = {c: (np.array([lab[(w, c, i)]["correct"] for i in P["ids"]], dtype=float),
                              np.array([lab[(w, c, i)]["out_tokens"] for i in P["ids"]], dtype=float)) for c in CONDS}
            boot = Boot(P["canonical"])
            block = {"n": P["n"], "signals": list(P["sig"]), "wordings": {}, "spread": {}}
            for w in WORDINGS:
                (yf, gf), (ys, gs) = ops[w]["F2"], ops[w]["F2-scratch"]
                e = {"F2 ALL": boot.ci(yf - P["y0"]), "F2-scratch ALL": boot.ci(ys - P["y0"]), "contrast ALL": boot.ci(yf - ys),
                     "tokens": {"F2": float(gf.mean()), "F2-scratch": float(gs.mean())}, "selective": {}}
                for sig in P["sig"]:
                    for share in A.SHARES:
                        sel = A.select(P["sig"][sig], P["tie"], share)
                        tag = f"{sig} {round(share * 100)}%"
                        e["selective"][f"contrast {tag}"] = boot.ci(sel * (yf - ys))
                        e["selective"][f"F2 {tag}"] = boot.ci(sel * (yf - P["y0"]))
                        e["selective"][f"F2-scratch {tag}"] = boot.ci(sel * (ys - P["y0"]))
                block["wordings"][w] = e
            quantities = ["F2 ALL", "F2-scratch ALL", "contrast ALL"] + list(block["wordings"]["main"]["selective"])
            for q in quantities:
                ests = [(block["wordings"][w][q] if q in block["wordings"][w] else block["wordings"][w]["selective"][q])["estimate"]
                        for w in WORDINGS]
                block["spread"][q] = max(ests) - min(ests)
            per_model[model] = {q: {w: (block["wordings"][w][q] if q in block["wordings"][w] else block["wordings"][w]["selective"][q])["estimate"]
                                    for w in WORDINGS} for q in quantities if q.startswith("contrast")}
            res["settings"][P["setting"]] = block
        guards = list(eq.guards)
    finally:
        eq.close()
    assert sha(frozen) == frozen_hash
    res["scorer_guards"] = guard_check(guards)
    res["sign_rule"] =verdicts(per_model) if len(per_model) == len(MODELS) else {}
    if len(per_model) != len(MODELS):
        res["sign_rule_note"] = "TODO(missing: the sign rule needs all three models)"
    dump(OUT / "b3_paraphrases.json", res)
    report(res)
    manifest("b3_paraphrases", {"outputs": {p: sha(OUT / p) for p in ("b3_extractions.jsonl.gz", "b3_labels.jsonl.gz",
                                                                      "b3_paraphrases.json")}, "settings_status": status})
    print(json.dumps({"status": status, "sign_rule": {q: v["flip_survives"] for q, v in res["sign_rule"].items()}}, indent=1))


if __name__ == "__main__":
    main()
