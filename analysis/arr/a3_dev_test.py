"""A3 of the ARR plan (docs/planning/arr_round/plan.md): choose a revision policy on dev, evaluate it on test.

Settings with dev operator outputs: math Qwen3-1.7B and 4B (1,264 dev items), MMLU-Pro Qwen3-4B and T8B (857).
Stored outputs only; the tokenizer is loaded (CPU) only to rebuild the first-turn prompt keys of the two-turn
operators on dev, as analysis/fu_recheck_cost.py does. No model is loaded.

Steps
1. Dev populations: common gold-blind labels (extraction frozen before gold, as A0), costs of both turns, signals:
   verbal, length, an out-of-fold probe for math (5 folds grouped by canonical problem, frozen layer and C,
   trained on the stored labels the frozen probe used), Qwen3-4B math agreement/margin and s1 from the dev samples,
   MMLU-Pro stored agreement, margin, s1 and the transferred math probe. The MMLU-Pro task probe is in-sample on dev
   and excluded.
2. Parity on test: verbal scores, first-turn token counts (math) and gold-blind agreement (4B math) are rebuilt with
   the dev code and must equal the test values used in A1/A2.
3. Grid: operator x signal x share (0-100 % in 5-point steps). R1 = best dev accuracy (ties: lower dev cost, then
   smaller share); R2 = the same with share <= 20 %. Transfer the share (primary) and the dev score threshold
   (sensitivity) to test. Report test gains with paired CIs, the rank and regret in the test grid, and a joint
   dev+test bootstrap that re-selects in every replicate (dev and test share no canonical problem).
Usage: python -B analysis/arr/a3_dev_test.py
-> analysis/out/arr/a3_dev_test.json, a3_grid.csv, math_dev_extractions.jsonl.gz, math_dev_labels.jsonl.gz,
   a3_dev_test_manifest.json, reports/arr_a3.md
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (J1, OPS, OUT, REPO, SEED, Boot, Equivalence, candidate, dump, groups_of, manifest, pin,  # noqa: E402
                    plan_core_hash, plan_hash, rows, rows_gz, run_dir, sha, tie_key, unique, winner, write_rows_gz)
import a1a2_replays as A  # noqa: E402

sys.path.insert(0, str(REPO / "analysis"))
import fu_recheck_cost as RC  # noqa: E402
from p6_confirmatory import verbal_s  # noqa: E402

STEP_SHARES = [s / 100 for s in range(0, 101, 5)]
K = 5


def verified(d, name):
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if name in man.get("files", {}):
        assert sha(d / name) == man["files"][name], ("manifest checksum", d, name)
    return pin(d / name)


def tokenizer(model):
    from transformers import AutoTokenizer
    from src.runlib import MODELS
    return AutoTokenizer.from_pretrained(MODELS[model])


# ----------------------------------------------------------------------------- math dev
def math_sources(model, split):
    init = {}
    for run in (f"p1-initial-{model}-d2ebdea", f"p1-initial-ext-{model}-45575c2"):
        for r in rows(verified(run_dir(run), "initial.jsonl")):
            if r["split"] == split:
                assert r["item_id"] not in init
                init[r["item_id"]] = r
    verbal = {}
    for run in (f"p1-verbal-{model}-d2ebdea", f"p1-verbal-ext-{model}-45575c2"):
        for r in rows(verified(run_dir(run), "verbal.jsonl")):
            if r["split"] == split:
                verbal[r["item_id"]] = r
    return init, verbal


def load_gens(dirs):
    gens = {}
    for d in dirs:
        for g in rows(verified(d, "generations.jsonl")):
            if g["prompt_key"] in gens:
                assert gens[g["prompt_key"]]["text"] == g["text"]
            gens[g["prompt_key"]] = g
    return gens


def math_dev(model, eq, tok, data):
    init, verbal = math_sources(model, "dev")
    dirs = [run_dir(f"p2-exp01-dev-{model}-f7e5011"), run_dir(f"p2-exp01-dev-ext-{model}-6cd867e")]
    ops = {}
    for d in dirs:
        for r in rows(verified(d, "rows.jsonl")):
            if r["exp"] == "1A" and r["data"] == "dnat" and r["cond"] in OPS:
                slot = ops.setdefault(r["item_id"], {})
                assert r["cond"] not in slot
                slot[r["cond"]] = r
    gens = load_gens(dirs)
    ids = sorted(i for i in init if i in ops and all(c in ops[i] for c in OPS))
    assert len(ids) == len(init) == 1264, (model, len(ids), len(init))
    k1 = RC.first_turn_keys(tok, {i: data[i] for i in ids}, init)
    recs = []
    for i in ids:
        a = init[i]
        rec = {"model": model, "item_id": i, "canonical_id": a["canonical_id"], "dataset": a["dataset"],
               "initial": {**candidate(a["a0"], a["finish_reason"], len(a["a0_token_ids"]), eq), "hist_correct": int(a["correct"])}}
        for c in OPS:
            r = ops[i][c]
            g = gens[r["prompt_key"]]
            assert g["out_tokens"] == r["out_tokens"] and g["finish_reason"] == r["finish_reason"]
            gen, prompt = g["out_tokens"], g["prompt_tokens"]
            if c != "F0":
                g1 = gens[k1[(i, c)]]
                gen, prompt = gen + g1["out_tokens"], prompt + g1["prompt_tokens"]
            rec[c] = {**candidate(g["text"], g["finish_reason"], g["out_tokens"], eq), "hist_correct": int(r["correct"]),
                      "gen": gen, "prompt": prompt}
        rec["verbal"] = verbal_s(verbal[i])
        rec["length"] = -len(a["a0_token_ids"])
        recs.append(rec)
    return recs, init


def dev_samples_4B(eq):
    d = run_dir("fu-dev-h2a-consistency-dev-4B-aac4cc6")
    return unique(rows(verified(d, "samples.jsonl")))


def oof_probe(model, recs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sel = json.loads(pin(run_dir(f"p3-probe-dev-ext-{model}-b7cf0d6") / "probe_selection.json").read_text(encoding="utf-8"))["selected"]
    X, ids = [], []
    for run in (f"p2-acts-dev-{model}-f7e5011", f"p2-acts-dev-ext-{model}-8360990"):
        z = np.load(pin(run_dir(run) / "acts.npz"))
        X.append(z["X"][:, sel["layer"]].astype(np.float32))
        ids += [str(i) for i in z["ids"]]
    X = np.concatenate(X)
    pos = {i: k for k, i in enumerate(ids)}
    Xd = X[np.array([pos[r["item_id"]] for r in recs])]
    y = np.array([r["initial"]["hist_correct"] for r in recs])
    groups = [r["canonical_id"] for r in recs]
    out = np.zeros(len(recs))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xd, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=sel["C"], max_iter=2000)).fit(Xd[tr], y[tr])
        out[te] = clf.predict_proba(Xd[te])[:, 1]
    return out, sel


def math_dev_population(model, recs, probe, samples=None):
    ids = [r["item_id"] for r in recs]
    P = {"setting": f"math {model}", "task": "math", "model": model, "ids": ids, "n": len(ids),
         "canonical": [r["canonical_id"] for r in recs], "source": [r["dataset"] for r in recs],
         "tie": np.array([tie_key(i) for i in ids], dtype=float),
         "y0": np.array([r["initial"]["correct"] for r in recs], dtype=float),
         "a0_answered": np.array([r["initial"]["status"] == "ok" for r in recs]),
         "sig": {"verbal": np.array([r["verbal"] for r in recs], dtype=float), "probe": probe,
                 "length": np.array([r["length"] for r in recs], dtype=float)}, "ops": {}}
    for c in OPS:
        P["ops"][c] = {"y1": np.array([r[c]["correct"] for r in recs], dtype=float),
                       "gen": np.array([r[c]["gen"] for r in recs], dtype=float),
                       "prompt": np.array([r[c]["prompt"] for r in recs], dtype=float),
                       "answered": np.array([r[c]["status"] == "ok" for r in recs])}
    if samples:
        n = P["n"]
        S = {"valid": np.zeros((n, K), bool), "correct": np.zeros((n, K)), "tok": np.zeros((n, K)), "groups": []}
        for j, r in enumerate(recs):
            S["groups"].append(r["groups"])
            for k, c in enumerate(r["samples"]):
                S["valid"][j, k], S["correct"][j, k], S["tok"][j, k] = c["status"] == "ok", c["correct"], c["out_tokens"]
        P["samples"] = S
        P["sig"]["agreement"] = np.array([r["agreement"] for r in recs], dtype=float)
        P["sig"]["margin"] = np.array([winner(g)[2] / K for g in S["groups"]], dtype=float)
    return P


# ----------------------------------------------------------------------------- MMLU-Pro dev
def mc_dev(model):
    base = REPO / "results/v2/fu-dev" / model
    init = unique(r for r in rows(verified(base / f"fu-dev-mc-initial-dev-{model}-ef3e485", "initial.jsonl")) if r["split"] == "dev")
    conds = {}
    for r in rows(verified(base / f"fu-dev-mc-conds-dev-{model}-ef3e485", "rows.jsonl")):
        if r["cond"] in OPS:
            slot = conds.setdefault(r["item_id"], {})
            assert r["cond"] not in slot
            slot[r["cond"]] = r
    hf = base / f"fu-dev-mc-hf-{model}-ef3e485"
    verbal = unique(r for r in rows(verified(hf, "verbal.jsonl")) if r["split"] == "dev")
    probes = unique(r for r in rows(verified(hf, "probe_scores.jsonl")) if r["split"] == "dev")
    cons = unique(r for r in rows(verified(base / f"fu-dev-mc-consist-dev-{model}-ef3e485", "consistency.jsonl")) if r["split"] == "dev")
    meta = {r["item_id"]: r for r in rows(pin(REPO / "data/processed/mmlupro/items.jsonl")) if r["item_id"] in init}
    ids = sorted(init)
    assert len(ids) == 857 and all(set(conds[i]) == set(OPS) for i in ids) and set(ids) <= set(cons) <= set(init) | set(cons)

    def ok(p, i):
        return isinstance(p, str) and len(p) == 1 and "A" <= p < chr(65 + meta[i]["n_options"])
    P = {"setting": f"MMLU-Pro {model}", "task": "mc", "model": model, "ids": ids, "n": len(ids),
         "canonical": [meta[i]["canonical_id"] for i in ids], "source": [meta[i]["category"] for i in ids],
         "tie": np.array([tie_key(i) for i in ids], dtype=float),
         "y0": np.array([bool(init[i]["correct"]) for i in ids], dtype=float),
         "a0_answered": np.array([init[i]["pred"] is not None for i in ids]),
         "sig": {"verbal": np.array([verbal[i]["W0"]["s"] for i in ids], dtype=float),
                 "length": -np.array([init[i]["out_tokens"] for i in ids], dtype=float),
                 "agreement": np.array([cons[i]["agreement"] for i in ids], dtype=float),
                 "math_probe": np.array([probes[i]["p_math_probe"] for i in ids], dtype=float)}, "ops": {}}
    for c in OPS:
        r = [conds[i][c] for i in ids]
        P["ops"][c] = {"y1": np.array([bool(x["correct"]) for x in r], dtype=float),
                       "gen": np.array([x["out_tokens"] + x.get("turn1_out_tokens", 0) for x in r], dtype=float),
                       "prompt": np.array([x["prompt_tokens"] + x.get("turn1_prompt_tokens", 0) for x in r], dtype=float),
                       "answered": np.array([x["pred"] is not None for x in r])}
        assert all((c == "F0") == ("turn1_out_tokens" not in x) for x in r)
    n = P["n"]
    S = {"valid": np.zeros((n, K), bool), "correct": np.zeros((n, K)), "tok": np.zeros((n, K)), "groups": []}
    for j, i in enumerate(ids):
        assert cons[i]["a0_pred"] == init[i]["pred"]
        cands = [{"pred": s["pred"], "status": "ok" if ok(s["pred"], i) else "invalid"} for s in cons[i]["samples"]]
        S["groups"].append(groups_of(cands, lambda a, b, c=cands: c[a]["pred"] == c[b]["pred"]))
        for k, s in enumerate(cons[i]["samples"]):
            S["valid"][j, k] = cands[k]["status"] == "ok"
            S["correct"][j, k] = float(S["valid"][j, k] and s["pred"] == meta[i]["gold"])
            S["tok"][j, k] = s["out_tokens"]
    P["samples"] = S
    P["sig"]["margin"] = np.array([winner(g)[2] / K for g in S["groups"]], dtype=float)
    return P


# ----------------------------------------------------------------------------- grid, selection, transfer
def operators(P):
    ops = {op: (P["ops"][op]["y1"], P["ops"][op]["gen"]) for op in OPS}
    if "samples" in P:
        ops["s1"] = (P["samples"]["correct"][:, 0], P["samples"]["tok"][:, 0])
    return ops


def acquisition(P, sig):
    return P["samples"]["tok"].sum(axis=1) if sig in ("agreement", "margin") else np.zeros(P["n"])


def grid(P, signals):
    """Every operator x signal x share; KEEP once. Returns (keys, delta matrix n x G, cost matrix n x G, thresholds)."""
    y0, n = P["y0"], P["n"]
    keys, D, C, thr = [("KEEP", "-", 0.0)], [np.zeros(n)], [np.zeros(n)], [None]
    for op, (y1, gen) in operators(P).items():
        for sig in signals + ["random"]:
            acq = acquisition(P, sig)
            order = None if sig == "random" else np.lexsort((P["tie"], P["sig"][sig]))
            for share in STEP_SHARES[1:]:
                m = round(n * share)
                if sig == "random":
                    D.append((m / n) * (y1 - y0))
                    C.append(acq + (m / n) * gen)
                    thr.append(None)
                else:
                    sel = np.zeros(n, dtype=bool)
                    sel[order[:m]] = True
                    D.append(sel * (y1 - y0))
                    C.append(acq + (0 if A.reuses_sample(op, sig) else sel * gen))
                    thr.append(float(P["sig"][sig][order[m - 1]]))
                keys.append((op, sig, share))
    return keys, np.column_stack(D), np.column_stack(C), thr


def choose(keys, dmean, cmean, max_share=1.0):
    best = None
    for g, (op, sig, share) in enumerate(keys):
        if share > max_share + 1e-12:
            continue
        cand = (-round(dmean[g], 12), round(cmean[g], 9), share, g)
        if best is None or cand < best:
            best = cand
    return best[3]


def apply_threshold(P, key, thr):
    op, sig, share = key
    if op == "KEEP" or sig == "random" or thr is None:
        return None
    y1, gen = operators(P)[op]
    sel = P["sig"][sig] <= thr
    return sel * (y1 - P["y0"]), acquisition(P, sig) + sel * gen, int(sel.sum())


def a3_setting(dev, test, signals, boot_dev, boot_test):
    kd, Dd, Cd, thr = grid(dev, signals)
    kt, Dt, Ct, _ = grid(test, signals)
    assert kd == kt
    dmean, cmean, tmean = Dd.mean(axis=0), Cd.mean(axis=0), Dt.mean(axis=0)
    out = {"dev_n": dev["n"], "test_n": test["n"], "dev_base_acc_pct": 100 * float(dev["y0"].mean()),
           "test_base_acc_pct": 100 * float(test["y0"].mean()), "signals": signals, "grid_size": len(kd), "rules": {}}
    order_test = np.argsort(-tmean, kind="stable")
    for rule, cap in (("R1", 1.0), ("R2", 0.2)):
        g = choose(kd, dmean, cmean, cap)
        op, sig, share = kd[g]
        entry = {"policy": {"operator": op, "signal": sig, "share_pct": round(share * 100)},
                 "dev_delta_pp": 100 * float(dmean[g]), "dev_generated_tokens": float(cmean[g]),
                 "test_delta_vs_keep": boot_test.ci(Dt[:, g]), "test_generated_tokens": float(Ct[:, g].mean()),
                 "test_rank": int(np.flatnonzero(order_test == g)[0]) + 1,
                 "test_regret_pp": 100 * float(tmean.max() - tmean[g]),
                 "test_best": {"policy": kd[int(order_test[0])], "delta_pp": 100 * float(tmean.max())}}
        if op != "KEEP":
            y1 = operators(test)[op][0]
            entry["test_delta_vs_operator_all"] = boot_test.ci(Dt[:, g] - (y1 - test["y0"]))
        tt = apply_threshold(test, kd[g], thr[g])
        if tt is not None:
            entry["threshold_transfer"] = {"dev_threshold": thr[g], "test_selected_n": tt[2],
                                           "test_delta_vs_keep": boot_test.ci(tt[0]), "test_generated_tokens": float(tt[1].mean())}
        # joint bootstrap: re-select on each dev replicate, evaluate on the paired test replicate
        dev_b = (boot_dev.W.astype(np.float64) @ Dd) / boot_dev.den[:, None]
        cost_pen = np.round(Cd.mean(axis=0), 9)
        allowed = np.array([k[2] <= cap + 1e-12 for k in kd])
        dev_b = np.where(allowed[None, :], dev_b, -np.inf)
        best = dev_b.max(axis=1, keepdims=True)
        tied = dev_b >= best - 1e-12
        pick = np.where(tied, -cost_pen[None, :] - 1e-6 * np.array([k[2] for k in kd])[None, :], -np.inf).argmax(axis=1)
        test_b = (boot_test.W.astype(np.float64) @ Dt) / boot_test.den[:, None]
        chosen = test_b[np.arange(len(pick)), pick]
        entry["joint_reselection"] = {"test_delta_ci_pp": (100 * np.quantile(chosen, [0.025, 0.975])).tolist(),
                                      "share_of_replicates_choosing_point_policy": float((pick == g).mean()),
                                      "distinct_policies_chosen": int(len(set(pick.tolist())))}
        out["rules"][rule] = entry
    out["dev_top5"] = [{"policy": kd[int(g)], "dev_delta_pp": 100 * float(dmean[g]), "test_delta_pp": 100 * float(tmean[g])}
                       for g in np.argsort(-dmean, kind="stable")[:5]]
    rows_out = [{"setting": test["setting"], "operator": k[0], "signal": k[1], "share_pct": round(k[2] * 100),
                 "dev_delta_pp": 100 * float(dmean[g]), "dev_generated_tokens": float(cmean[g]),
                 "test_delta_pp": 100 * float(tmean[g]), "test_generated_tokens": float(Ct[:, g].mean())}
                for g, k in enumerate(kd)]
    return out, rows_out


# ----------------------------------------------------------------------------- main
def main():
    plan = plan_hash()
    data = unique(r for p in (pin(REPO / "data/processed/dnat.jsonl"), pin(REPO / "data/processed/ext/dnat_ext.jsonl")) for r in rows(p))
    labels = list(rows_gz(pin(OUT / "math_common_labels.jsonl.gz")))
    table = {}
    for r in rows(pin(J1 / "artifacts/math/item_evaluations.jsonl")):
        table.setdefault(r["model"], {}).setdefault(r["condition"], {})[r["item_id"]] = r
    test_math = A.load_math(table, labels)
    eq = Equivalence()
    parity, dev_recs = [], []
    try:
        A.add_math_samples(test_math["4B"], eq)
        dev_pop = {}
        for model in ("1.7B", "4B"):
            tok = tokenizer(model)
            # parity on test: verbal and first-turn costs rebuilt with the dev code
            init_t, verbal_t = math_sources(model, "test")
            t2 = table[model]
            ids_t = test_math[model]["ids"]
            vt = np.array([verbal_s(verbal_t[i]) for i in ids_t], dtype=float)
            assert np.allclose(vt, test_math[model]["sig"]["verbal"], rtol=0, atol=1e-9), "verbal parity"
            runs = sorted({t2["F2"][i]["source_run"] for i in ids_t})
            gens_t = load_gens([run_dir(r) for r in runs])
            k1t = RC.first_turn_keys(tok, {i: data[i] for i in ids_t}, init_t)
            for c in ("F2", "F2-scratch"):
                assert all(gens_t[k1t[(i, c)]]["out_tokens"] == t2[c][i]["turn1_output_tokens"] for i in ids_t), "turn-1 parity"
            del gens_t
            parity.append(f"math {model} test: verbal scores and first-turn token counts rebuilt by the dev code match")
            recs, init_d = math_dev(model, eq, tok, data)
            if model == "4B":
                samples = dev_samples_4B(eq)
                for r in recs:
                    cands = [candidate(s["text"], s["finish_reason"], s["out_tokens"], eq) for s in samples[r["item_id"]]["samples"]]
                    assert len(cands) == K
                    r["samples"] = [{k: c[k] for k in ("pred", "status", "out_tokens")} for c in cands]
                    r["groups"] = groups_of(cands, lambda a, b, c=cands: eq.symmetric(c[a]["pred"], c[b]["pred"]))
                    r["agreement"] = sum(eq(c["pred"], r["initial"]["pred"]) for c in cands) / K
                # parity on test: gold-blind agreement rebuilt from the ACL candidates
                per = unique(rows(pin(REPO / "paper/1区/acl2027/artifacts/math_vote/per_item.jsonl")))
                P4 = test_math["4B"]
                rebuilt = np.array([sum(eq(c["pred"], P4["a0_pred"][j]) for c in per[i]["candidates"]) / K
                                    for j, i in enumerate(P4["ids"])])
                assert np.array_equal(rebuilt, P4["sig"]["agreement"]), "agreement parity"
                parity.append("math 4B test: gold-blind agreement rebuilt by the dev code matches the ACL values")
            for r in recs:
                for c in ("initial",) + OPS:
                    r[c] = {k: v for k, v in r[c].items() if k not in ("final_boxes",)}
            dev_recs += recs
            dev_pop[model] = recs
        frozen = OUT / "math_dev_extractions.jsonl.gz"
        write_rows_gz(frozen, dev_recs)
        frozen_hash = sha(frozen)
        for r in dev_recs:
            gold = data[r["item_id"]]["gold"]
            assert data[r["item_id"]]["split"] == "dev"
            for c in ("initial",) + OPS:
                r[c]["correct"] = int(r[c]["status"] == "ok" and eq(r[c]["pred"], gold))
            for s in r.get("samples", []):
                s["correct"] = int(s["status"] == "ok" and eq(s["pred"], gold))
        guards = list(eq.guards)
    finally:
        eq.close()
    assert sha(frozen) == frozen_hash and not guards, guards
    write_rows_gz(OUT / "math_dev_labels.jsonl.gz", dev_recs)
    results, grid_rows, probe_info = {"plan_sha256_lf": plan, "plan_core_sha256": plan_core_hash(), "extractions_sha256_before_gold": frozen_hash,
                                      "parity": parity, "settings": {}}, [], {}
    boot_math_test = Boot(test_math["4B"]["canonical"])
    for model in ("1.7B", "4B"):
        recs = dev_pop[model]
        probe, sel = oof_probe(model, recs)
        P = math_dev_population(model, recs, probe, samples=(model == "4B"))
        y = P["y0"]
        num, den = A.auroc_parts(probe, y, None)
        probe_info[model] = {"layer": sel["layer"], "C": sel["C"], "dev_oof_auroc_common_labels": float(num[0] / den[0])}
        signals = ["verbal", "probe", "length"] + (["agreement", "margin"] if model == "4B" else [])
        out, g = a3_setting(P, test_math[model], signals, Boot(P["canonical"], seed=SEED + 1), boot_math_test)
        results["settings"][f"math {model}"] = out
        grid_rows += g
    test_mc = {}
    for model in ("4B", "T8B"):
        T = A.load_mc(model)
        items = {r["item_id"]: r for r in rows(pin(J1 / f"artifacts/mc/{model}_items.jsonl")) if r["condition"] == "F2"}
        T["sig"]["math_probe"] = np.array([items[i]["math_probe"] for i in T["ids"]], dtype=float)
        hf = REPO / json.loads((J1 / "artifacts/mc/summary.json").read_text(encoding="utf-8"))["models"][model]["runs"]["hf"]
        vt = unique(r for r in rows(pin(hf / "verbal.jsonl")) if r["split"] == "test")
        pt = unique(r for r in rows(pin(hf / "probe_scores.jsonl")) if r["split"] == "test")
        assert np.allclose([vt[i]["W0"]["s"] for i in T["ids"]], T["sig"]["verbal"], rtol=0, atol=1e-9)
        assert np.allclose([pt[i]["p_math_probe"] for i in T["ids"]], T["sig"]["math_probe"], rtol=0, atol=1e-12)
        parity.append(f"MMLU-Pro {model} test: verbal W0 and transferred math-probe scores equal the run files the dev code reads")
        test_mc[model] = T
    boot_mc_test = Boot(test_mc["4B"]["canonical"])
    for model in ("4B", "T8B"):
        P = mc_dev(model)
        out, g = a3_setting(P, test_mc[model], ["verbal", "length", "agreement", "margin", "math_probe"], Boot(P["canonical"], seed=SEED + 1), boot_mc_test)
        results["settings"][f"MMLU-Pro {model}"] = out
        grid_rows += g
    results["dev_probe_out_of_fold"] = probe_info
    dump(OUT / "a3_dev_test.json", results)
    with (OUT / "a3_grid.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(grid_rows)
    report(results)
    manifest("a3_dev_test", {"outputs": {p: sha(OUT / p) for p in ("a3_dev_test.json", "a3_grid.csv", "math_dev_labels.jsonl.gz")}})
    print(json.dumps({s: {r: (v["policy"], round(v["test_delta_vs_keep"]["estimate"], 2)) for r, v in o["rules"].items()}
                      for s, o in results["settings"].items()}, indent=1))


def fmt(ci):
    return f"{ci['estimate']:+.2f} [{ci['lo']:+.2f}, {ci['hi']:+.2f}]"


def report(res):
    L = ["# ARR round, tier A3: policy chosen on dev, evaluated on test", "",
         "Generated by `analysis/arr/a3_dev_test.py` (plan `docs/planning/arr_round/plan.md`, §A3). Post hoc and exploratory "
         "(DEVIATIONS D20). Stored outputs only. Gains in pp vs KEEP with paired 95 % cluster-bootstrap intervals.", "",
         "R1: best dev accuracy over operator × signal × share (0–100 % in 5-point steps; ties: lower dev cost, then smaller share). "
         "R2: the same with at most 20 % of items revised. The share is transferred to test (primary); the dev score threshold is a sensitivity. "
         "The joint interval re-selects on every dev replicate and evaluates on the paired test replicate (dev draws use seed 20261111, test draws 20261110).", "",
         "| Setting | Rule | Policy (operator, signal, share) | Dev Δ | Test Δ vs KEEP | Test Δ vs operator ALL | Threshold transfer | Joint re-selection CI | Test rank / grid | Regret | Gen. tok/item (test) |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, o in res["settings"].items():
        for rule, e in o["rules"].items():
            p = e["policy"]
            thr = fmt(e["threshold_transfer"]["test_delta_vs_keep"]) if "threshold_transfer" in e else "—"
            vsall = fmt(e["test_delta_vs_operator_all"]) if "test_delta_vs_operator_all" in e else "—"
            j = e["joint_reselection"]["test_delta_ci_pp"]
            L.append(f"| {s} | {rule} | {p['operator']}, {p['signal']}, {p['share_pct']} % | {e['dev_delta_pp']:+.2f} | "
                     f"{fmt(e['test_delta_vs_keep'])} | {vsall} | {thr} | [{j[0]:+.2f}, {j[1]:+.2f}] | "
                     f"{e['test_rank']} / {o['grid_size']} | {e['test_regret_pp']:.2f} | {e['test_generated_tokens']:.0f} |")
    L += ["", "Dev base accuracy and test-best grid policy:", ""]
    for s, o in res["settings"].items():
        b = o["rules"]["R1"]["test_best"]
        L.append(f"- {s}: dev n = {o['dev_n']} (KEEP {o['dev_base_acc_pct']:.2f} %), test n = {o['test_n']} (KEEP {o['test_base_acc_pct']:.2f} %); "
                 f"test-best grid policy {b['policy'][0]}, {b['policy'][1]}, {round(b['policy'][2] * 100)} % at {b['delta_pp']:+.2f} pp.")
    L += ["", "Out-of-fold dev probe (math): " + "; ".join(f"{m}: layer {v['layer']}, C {v['C']}, dev AUROC {v['dev_oof_auroc_common_labels']:.3f}"
                                                         for m, v in res["dev_probe_out_of_fold"].items()) + ".",
          "", "Parity: " + "; ".join(res["parity"]) + ".", ""]
    (REPO / "reports" / "arr_a3.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
