"""Cost accounting and selective re-check policies, 4B, pooled D-nat test (second review, review_and_suggestion.md
2026-11-08, items B and "step 3"; DEVIATIONS D18). Exploratory: the test split has been seen; every rule is fitted on
dev only, and the policies, features and budgets are fixed in this file before its first test output.

1. Cost per 1A condition, on top of a0 (all turns counted: the critique / scratch turn of F2 and F2-scratch and the
   revision): generated tokens and prompt tokens processed (prompt tokens summed over turns, i.e. without KV-cache
   reuse across turns; an upper bound on prefill work). Per-condition wall-clock time was not recorded and is not
   recovered from token counts. Net accuracy change, W->R and R->W next to the cost.
2. Signal costs per item: verbal score (a prefill of the verbal prompt: question, a0 and the query; or only the query
   tokens when the a0 cache is reused); probe and token log-probs (zero extra work when read during a0 generation,
   otherwise one prefill of prompt + a0); 5-sample agreement with a0 (5 x (prompt prefill + sampled tokens)); answer
   length (free). Probe training (dev activations, one logistic fit) is a one-off cost, reported separately.
3. Selective re-check: re-run the action (F2, or F0) on a fraction k of items, keep a0 elsewhere. Rankings:
   random (expected value), verbal s, probe, answer length (longer first), minimum token log-prob, agreement with a0,
   two dev-fitted correctness models (cheap: s, length, strata; full: + min log-prob, agreement), two dev-fitted
   net-gain models (multinomial logistic for the change in correctness under the action, ranked by
   P(fix) - P(damage); cheap: s, length, strata, probe; full: + min log-prob, agreement), and a gold reference that
   re-checks initially wrong a0 first, in tie-break order among them (uses gold; not a policy and not an upper bound:
   wrong answers differ in how often the action fixes them, so a ranking can fix more than it). Dev probe features are cross-fitted (5-fold, grouped by
   canonical problem, the frozen probe's layer and C), so no dev item is scored by a probe trained on it.
   Per policy and k: accuracy change, fixes and damages, incremental generated and processed tokens per item
   (signal cost on every item + action cost on the selected items), and paired differences against verbal s and
   against random (95 % cluster-bootstrap CIs, clusters = canonical problem).
-> reports/fu_05_recheck_cost.md, analysis/out/fu_05_recheck_cost.json
Usage: python analysis/fu_recheck_cost.py [--n-boot 10000]
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from p6_confirmatory import N_BOOT, SEED, STRATUM, Boot, jl, verbal_s, wmean  # noqa: E402

RUNS = {"test": ["results/main/4B/main-exp01-test-4B-a7a0579", "results/main/4B/main-exp01-test-ext-4B-a7a0579"],
        "dev": ["results/p2/4B/p2-exp01-dev-4B-f7e5011", "results/p2/4B/p2-exp01-dev-ext-4B-6cd867e"]}
INITIAL = ["results/p1/4B/p1-initial-4B-d2ebdea/initial.jsonl", "results/p1/4B/p1-initial-ext-4B-45575c2/initial.jsonl"]
VERBAL = ["results/p1/4B/p1-verbal-4B-d2ebdea/verbal.jsonl", "results/p1/4B/p1-verbal-ext-4B-45575c2/verbal.jsonl"]
PROBE_TEST = "results/main/4B/main-probe-scores-test-4B-a7a0579/probe_scores.jsonl"
PROBE_SEL = "results/p3/4B/p3-probe-dev-ext-4B-b7cf0d6/probe_selection.json"
DEV_ACTS = ["results/p2/4B/p2-acts-dev-4B-f7e5011/acts.npz", "results/p2/4B/p2-acts-dev-ext-4B-8360990/acts.npz"]
LOGPROB = {"test": "results/fu/4B/fu-h2a-logprob-test-4B-aac4cc6/logprob.jsonl",
           "dev": "results/fu-dev/4B/fu-dev-h2a-logprob-dev-4B-aac4cc6/logprob.jsonl"}
CONS = {"test": "results/fu/4B/fu-h2a-consistency-test-4B-aac4cc6", "dev": "results/fu-dev/4B/fu-dev-h2a-consistency-dev-4B-aac4cc6"}
CONDS = ["B1", "F0", "F1", "F2", "F2-scratch", "F3(0.5)", "F3(1.0)", "F5"]
ACTIONS = ("F2", "F0")
BUDGETS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
CRIT_TOKENS = 1024


def tie_order(ids):
    """A fixed pseudo-random order used to break ties (sha256 of the item ID)."""
    return np.array([int(hashlib.sha256(i.encode()).hexdigest()[:12], 16) for i in ids], float)


def first_turn_keys(tok, items, init):
    """prompt_key of the F2 / F2-scratch first turn of each item, rendered exactly as src/exp/runner.py does."""
    from src.gen import render as R
    from src.runlib import sha256_text
    keys = {}
    for i, it in items.items():
        for cond, turn in (("F2", R.CRITIQUE_TURN), ("F2-scratch", R.SCRATCH_TURN)):
            r = R.review_prompt(tok, it["question"], init[i]["a0"], instruction=turn)
            keys[(i, cond)] = sha256_text(r.text + "\x00" + r.prefill + f"\x00{CRIT_TOKENS}")
    return keys


def verbal_prompt_tokens(tok, items, init):
    from src.score.verbal import PROMPT, QUERY
    out = {}
    for i, it in items.items():
        msgs = [{"role": "user", "content": PROMPT.format(q=it["question"])}, {"role": "assistant", "content": init[i]["a0"]},
                {"role": "user", "content": QUERY}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out[i] = len(tok(text, add_special_tokens=False).input_ids)
    q_only = len(tok(tok.apply_chat_template([{"role": "user", "content": QUERY}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False),
                     add_special_tokens=False).input_ids)
    return out, q_only


def load_split(split, tok, items_all, strat):
    init = {r["item_id"]: r for f in INITIAL for r in jl(f) if r["split"] == split}
    verbal = {r["item_id"]: r for f in VERBAL for r in jl(f) if r["split"] == split}
    rows, gens = defaultdict(dict), {}
    for d in RUNS[split]:
        for r in jl(f"{d}/rows.jsonl"):
            if r["exp"] == "1A" and r["data"] == "dnat":
                rows[r["item_id"]][r["cond"]] = r
        gens.update({g["prompt_key"]: g for g in jl(f"{d}/generations.jsonl")})
    lp = {r["item_id"]: r for r in jl(LOGPROB[split])}
    cons = {r["item_id"]: r for r in jl(f"{CONS[split]}/consistency.jsonl")}
    samp = {r["item_id"]: sum(s["out_tokens"] for s in r["samples"]) for r in jl(f"{CONS[split]}/samples.jsonl")}
    ids = sorted(i for i in init if i in rows and all(c in rows[i] for c in CONDS) and i in verbal
                 and verbal_s(verbal[i]) is not None and i in lp and i in cons and i in strat)
    items = {i: items_all[i] for i in ids}
    k1 = first_turn_keys(tok, items, init)
    vp, q_only = verbal_prompt_tokens(tok, items, init)
    D = {"ids": ids, "y0": np.array([bool(init[i]["correct"]) for i in ids], float),
         "s": np.array([verbal_s(verbal[i]) for i in ids]), "len": np.array([len(init[i]["a0_token_ids"]) for i in ids], float),
         "lpmin": np.array([lp[i]["min"] for i in ids]), "agree": np.array([cons[i]["consistency"] for i in ids]),
         "strata": [strat[i] for i in ids], "a0_prompt": np.array([init[i]["prompt_tokens"] for i in ids], float),
         "verbal_prompt": np.array([vp[i] for i in ids], float), "verbal_query": q_only,
         "samp_gen": np.array([samp[i] for i in ids], float), "after": {}, "gen": {}, "proc": {}}
    missing = 0
    for c in CONDS:
        D["after"][c] = np.array([bool(rows[i][c]["correct"]) for i in ids], float)
        g2 = [gens[rows[i][c]["prompt_key"]] for i in ids]
        gen = np.array([g["out_tokens"] for g in g2], float)
        proc = np.array([g["prompt_tokens"] for g in g2], float)
        if c in ("F2", "F2-scratch"):
            g1 = [gens.get(k1[(i, c)]) for i in ids]
            missing += sum(g is None for g in g1)
            gen += np.array([g["out_tokens"] if g else np.nan for g in g1], float)
            proc += np.array([g["prompt_tokens"] if g else np.nan for g in g1], float)
        D["gen"][c], D["proc"][c] = gen, proc
    D["missing first turns"] = missing
    return D


def dev_probe_cv(dev_ids, canon, init_dev):
    """Cross-fitted dev probe scores with the frozen probe's layer and C."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sel = json.loads((ROOT / PROBE_SEL).read_text(encoding="utf-8"))["selected"]
    X, ids = [], []
    for f in DEV_ACTS:
        z = np.load(ROOT / f)
        X.append(z["X"][:, sel["layer"]].astype(np.float32))
        ids += [str(i) for i in z["ids"]]
    X = np.concatenate(X)
    pos = {i: k for k, i in enumerate(ids)}
    idx = np.array([pos[i] for i in dev_ids])
    Xd, y = X[idx], np.array([bool(init_dev[i]["correct"]) for i in dev_ids])
    groups = [canon[i] for i in dev_ids]
    out = np.zeros(len(dev_ids))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xd, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=sel["C"], max_iter=2000)).fit(Xd[tr], y[tr])
        out[te] = clf.predict_proba(Xd[te])[:, 1]
    return out, sel


def feats(D, strata_levels, probe=None, full=False):
    cols = [D["s"], D["len"]] + [np.array([s == lv for s in D["strata"]], float) for lv in strata_levels]
    if probe is not None:
        cols.append(probe)
    if full:
        cols += [D["lpmin"], D["agree"]]
    return np.column_stack(cols)


def fit(X, y, multinomial=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)


def gain_score(model, X):
    """P(fix) - P(damage) from a 3-class model of the change in correctness (-1, 0, +1)."""
    p = model.predict_proba(X)
    cls = list(model.classes_)
    return (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from transformers import AutoTokenizer

    from src.runlib import MODELS
    tok = AutoTokenizer.from_pretrained(MODELS["4B"])
    items_all, canon, strat = {}, {}, {}
    for f in ("data/processed/dnat.jsonl", "data/processed/ext/dnat_ext.jsonl"):
        for r in jl(f):
            items_all[r["item_id"]], canon[r["item_id"]] = r, r["canonical_id"]
            strat[r["item_id"]] = STRATUM.get(r["dataset"], r["dataset"])
    T, Dv = load_split("test", tok, items_all, strat), load_split("dev", tok, items_all, strat)
    init_dev = {r["item_id"]: r for f in INITIAL for r in jl(f) if r["split"] == "dev"}
    probe_dev, sel = dev_probe_cv(Dv["ids"], canon, init_dev)
    ps = {r["item_id"]: r["p_correct"] for r in jl(PROBE_TEST)}
    keep = np.array([i in ps for i in T["ids"]])
    if not keep.all():   # restrict the test pool to items with a probe score
        for k in ("y0", "s", "len", "lpmin", "agree", "a0_prompt", "verbal_prompt", "samp_gen"):
            T[k] = T[k][keep]
        T["strata"] = [s for s, m in zip(T["strata"], keep) if m]
        T["ids"] = [i for i, m in zip(T["ids"], keep) if m]
        for d in ("after", "gen", "proc"):
            T[d] = {c: v[keep] for c, v in T[d].items()}
    probe_test = np.array([ps[i] for i in T["ids"]])
    n = len(T["ids"])
    boot = Boot([canon[i] for i in T["ids"]], a.n_boot, SEED + 31)
    ci = lambda v: list(boot.ci(lambda w: wmean(v, w)))  # noqa: E731
    y0 = T["y0"]
    wrong, right = y0 == 0, y0 == 1
    out = {"n test": n, "n dev": len(Dv["ids"]), "n wrong test": int(wrong.sum()),
           "missing first turns": {"test": T["missing first turns"], "dev": Dv["missing first turns"]},
           "probe (frozen selection)": sel}

    # ---- 1. cost per condition
    a0_gen = T["len"]
    cost = {}
    for c in CONDS:
        d = T["after"][c] - y0
        cost[c] = {"acc change": ci(d), "W->R": float(T["after"][c][wrong].mean()), "R->W": float(1 - T["after"][c][right].mean()),
                   "generated tokens / item": float(np.nanmean(T["gen"][c])),
                   "generated, relative to a0": float(np.nanmean(T["gen"][c]) / a0_gen.mean()),
                   "processed prompt tokens / item": float(np.nanmean(T["proc"][c]))}
    out["a0 cost"] = {"generated tokens / item": float(a0_gen.mean()), "prompt tokens / item": float(T["a0_prompt"].mean())}
    out["cost per condition"] = cost

    # ---- 2. signal costs (per item, on every item)
    out["signal cost"] = {
        "verbal s": {"generated": 0.0, "processed (no cache)": float(T["verbal_prompt"].mean()),
                     "processed (a0 cache reused)": float(T["verbal_query"])},
        "probe": {"generated": 0.0, "processed (read during a0 generation)": 0.0,
                  "processed (separate prefill)": float((T["a0_prompt"] + T["len"]).mean())},
        "min token log-prob": {"generated": 0.0, "processed (read during a0 generation)": 0.0,
                               "processed (separate prefill)": float((T["a0_prompt"] + T["len"]).mean())},
        "agreement with a0 (5 samples)": {"generated": float(T["samp_gen"].mean()),
                                         "processed (no cache)": float(5 * T["a0_prompt"].mean())},
        "answer length": {"generated": 0.0, "processed": 0.0},
        "probe training (one-off)": {"dev items": len(Dv["ids"]), "prefill tokens": float((Dv["a0_prompt"] + Dv["len"]).sum())}}
    sig_gen = {"random": 0.0, "verbal s": 0.0, "probe": 0.0, "answer length": 0.0, "min token log-prob": 0.0,
               "agreement with a0": float(T["samp_gen"].mean())}

    # ---- 3. policies
    levels = sorted(set(T["strata"]) | set(Dv["strata"]))
    yd = Dv["y0"].astype(int)
    m_cheap = fit(feats(Dv, levels), yd)
    m_full = fit(feats(Dv, levels, full=True), yd)
    scores = {"verbal s": T["s"], "probe": probe_test, "answer length": -T["len"], "min token log-prob": T["lpmin"],
              "agreement with a0": T["agree"],
              "dev-fitted correctness, cheap": m_cheap.predict_proba(feats(T, levels))[:, 1],
              "dev-fitted correctness, full": m_full.predict_proba(feats(T, levels, full=True))[:, 1]}
    sig_gen["dev-fitted correctness, cheap"] = 0.0
    sig_gen["dev-fitted correctness, full"] = sig_gen["agreement with a0"]
    tie = tie_order(T["ids"])
    pol = {}
    for action in ACTIONS:
        after, act_gen = T["after"][action], T["gen"][action]
        gd = (Dv["after"][action] - Dv["y0"]).astype(int)
        g_cheap = fit(feats(Dv, levels, probe=probe_dev), gd)
        g_full = fit(feats(Dv, levels, probe=probe_dev, full=True), gd)
        sc = dict(scores)   # lower = re-check first
        sc["dev-fitted net gain, cheap"] = -gain_score(g_cheap, feats(T, levels, probe=probe_test))
        sc["dev-fitted net gain, full"] = -gain_score(g_full, feats(T, levels, probe=probe_test, full=True))
        sg = dict(sig_gen, **{"dev-fitted net gain, cheap": 0.0, "dev-fitted net gain, full": sig_gen["agreement with a0"],
                              "gold wrong-first (uses gold)": 0.0})
        sc["gold wrong-first (uses gold)"] = y0
        res = {"every item": {"acc change": ci(after - y0), "generated / item": float(np.nanmean(act_gen))}, "budgets": {}}
        for k in BUDGETS:
            n_sel = int(round(k * n))
            fin, sel_masks = {}, {}
            for name, s in sc.items():
                order = np.lexsort((tie, s))
                m = np.zeros(n, bool)
                m[order[:n_sel]] = True
                sel_masks[name] = m
                fin[name] = np.where(m, after, y0)
            exp_rand = y0 + (n_sel / n) * (after - y0)
            row = {}
            for name in sc:
                m, f = sel_masks[name], fin[name]
                row[name] = {"acc change": ci(f - y0), "fixes": int(((f == 1) & wrong).sum()), "damages": int(((f == 0) & right).sum()),
                             "share of selected initially wrong": float(wrong[m].mean()) if m.any() else None,
                             "incremental generated / item": float(sg[name] + np.nansum(act_gen[m]) / n),
                             "minus verbal s": ci(f - fin["verbal s"]) if name != "verbal s" else None,
                             "minus random": ci(f - exp_rand)}
            row["random (expected)"] = {"acc change": ci(exp_rand - y0),
                                        "incremental generated / item": float((n_sel / n) * np.nanmean(act_gen))}
            res["budgets"][f"{int(round(100 * k))}"] = row
        pol[action] = res
    out["policies"] = pol
    (ROOT / "analysis" / "out" / "fu_05_recheck_cost.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    write_md(out)


def write_md(o):
    pc = lambda v: "—" if v is None else f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"  # noqa: E731
    L = ["# Cost accounting and selective re-check, 4B (generated by analysis/fu_recheck_cost.py)", "",
         f"Pooled D-nat test, n {o['n test']} ({o['n wrong test']} initially wrong); dev n {o['n dev']} for every fitted rule. "
         "**Exploratory** (DEVIATIONS D18): the test split has been seen. 95 % cluster-bootstrap CIs (clusters = canonical "
         "problem). Token counts are per item and averaged over all items. Processed prompt tokens are summed over turns "
         "without KV-cache reuse. No wall-clock time is reported: per-condition times were not recorded.",
         f"First-turn generations not found: test {o['missing first turns']['test']}, dev {o['missing first turns']['dev']}.", "",
         f"a0: {o['a0 cost']['generated tokens / item']:.0f} generated, {o['a0 cost']['prompt tokens / item']:.0f} prompt tokens per item.", "",
         "## 1. Cost per condition (on top of a0)", "",
         "| Condition | Accuracy change (pp) | W->R | R->W | Generated tokens / item | × a0 | Processed prompt tokens / item |",
         "|---|---|---|---|---|---|---|"]
    for c, v in o["cost per condition"].items():
        L.append(f"| {c} | {pc(v['acc change'])} | {100 * v['W->R']:.1f} % | {100 * v['R->W']:.1f} % | "
                 f"{v['generated tokens / item']:.0f} | {v['generated, relative to a0']:.2f} | {v['processed prompt tokens / item']:.0f} |")
    L += ["", "F2 and F2-scratch count both turns. F3 and F5 omit the verifier's own cost (simulated, or the verbal readout).", "",
          "## 2. Signal cost per item", "", "| Signal | Cost |", "|---|---|"]
    for s, v in o["signal cost"].items():
        L.append(f"| {s} | " + "; ".join(f"{k} {x:,.0f}" if isinstance(x, float) else f"{k} {x}" for k, x in v.items()) + " |")
    for action, r in o["policies"].items():
        L += ["", f"## 3. Re-check with {action} on a fraction of items (keep a0 elsewhere)", "",
              f"Every item: accuracy change {pc(r['every item']['acc change'])} pp, {r['every item']['generated / item']:.0f} generated tokens per item.", ""]
        for k, row in r["budgets"].items():
            L += [f"**Budget {k} % of items**", "",
                  "| Ranking | Acc change (pp) | Fixes | Damages | Wrong among selected | Incr. generated / item | − verbal s (pp) | − random (pp) |",
                  "|---|---|---|---|---|---|---|---|"]
            for name, v in row.items():
                if name == "random (expected)":
                    L.append(f"| random (expected) | {pc(v['acc change'])} | — | — | — | {v['incremental generated / item']:.0f} | — | — |")
                    continue
                sw = "—" if v["share of selected initially wrong"] is None else f"{100 * v['share of selected initially wrong']:.0f} %"
                L.append(f"| {name} | {pc(v['acc change'])} | {v['fixes']} | {v['damages']} | {sw} | "
                         f"{v['incremental generated / item']:.0f} | {pc(v['minus verbal s'])} | {pc(v['minus random'])} |")
            L.append("")
    (ROOT / "reports" / "fu_05_recheck_cost.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
