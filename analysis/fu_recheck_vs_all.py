"""Selective re-check against re-checking every item (post hoc, exploratory; DEVIATIONS D19; third review,
review_and_suggestion.md §4.2 and §6.2).

The existing re-check tables (reports/fu_05_recheck_cost.md, math; reports/v2_02_mc_results.md, MMLU-Pro) give each
policy's accuracy change against keeping a0 (KEEP) and paired differences against the verbal order and random. They do
not give the paired difference against re-checking every item (ALL). "Fewer re-checks beat re-checking everything" needs
that paired interval, and when ALL itself loses accuracy, beating ALL is not beating KEEP.

This script rebuilds every ranking exactly as the two source scripts do (it imports their loaders and fitting
functions and repeats their ranking code), checks that it reproduces their stored fixes, damages and point estimates
for every ranking and budget, and adds, for every ranking and every budget already in those tables:
- selective minus ALL, paired, 95 % cluster-bootstrap CI (same clusters, seeds and resamples as the source script);
- the share of the selected items whose a0 was wrong.
No budget, ranking or model is chosen here; every cell of the existing grids is reported.
Reading rule (fixed here): "above ALL" only where the paired CI lies above zero; "above KEEP" only where the CI of the
change against a0 lies above zero (from the source tables); otherwise "not resolved".
Math: 4B pooled D-nat test (seen split). MMLU-Pro: 4B and T8B test (addendum split; P2's shares are 10 % and 20 %).

Usage: python analysis/fu_recheck_vs_all.py [--n-boot 10000]
-> reports/fu_09_recheck_vs_all.md, analysis/out/fu_09_recheck_vs_all.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import fu_recheck_cost as RC  # noqa: E402
import mc_results as MR  # noqa: E402
from p6_confirmatory import N_BOOT, SEED, STRATUM, Boot, jl, wmean  # noqa: E402


def table(y0, after, sc, tie, budgets, boot, stored):
    """Per budget and ranking: selective - ALL (CI), share wrong among selected; checked against the stored table."""
    n = len(y0)
    wrong, right = y0 == 0, y0 == 1
    ci = lambda v: list(boot.ci(lambda w: wmean(v, w)))  # noqa: E731
    res = {"ALL": ci(after - y0), "budgets": {}}
    for k in budgets:
        key = f"{int(round(100 * k))}"
        if key == "100":
            continue
        n_sel = int(round(k * n))
        row = {}
        for name, s in sc.items():
            m = np.zeros(n, bool)
            m[np.lexsort((tie, s))[:n_sel]] = True
            f = np.where(m, after, y0)
            st = stored[key][name]
            fixes, damages = int(((f == 1) & wrong).sum()), int(((f == 0) & right).sum())
            assert (fixes, damages) == (st["fixes"], st["damages"]), (key, name, fixes, damages, st["fixes"], st["damages"])
            assert abs(float((f - y0).mean()) - st["acc change"][0]) < 1e-12, (key, name)
            row[name] = {"minus ALL": ci(f - after), "share of selected initially wrong": float(wrong[m].mean()),
                         "acc change (vs KEEP)": st["acc change"]}
        exp_rand = y0 + (n_sel / n) * (after - y0)
        row["random (expected)"] = {"minus ALL": ci(exp_rand - after), "share of selected initially wrong": float(wrong.mean()),
                                    "acc change (vs KEEP)": stored[key]["random (expected)"]["acc change"]}
        res["budgets"][key] = row
    return res


def build_math(n_boot):
    """The math test population, rankings per action and bootstrap, rebuilt as in fu_recheck_cost.main()."""
    from transformers import AutoTokenizer

    from src.runlib import MODELS
    tok = AutoTokenizer.from_pretrained(MODELS["4B"])
    items_all, canon, strat = {}, {}, {}
    for f in ("data/processed/dnat.jsonl", "data/processed/ext/dnat_ext.jsonl"):
        for r in jl(f):
            items_all[r["item_id"]], canon[r["item_id"]] = r, r["canonical_id"]
            strat[r["item_id"]] = STRATUM.get(r["dataset"], r["dataset"])
    # --- as in fu_recheck_cost.main()
    T, Dv = RC.load_split("test", tok, items_all, strat), RC.load_split("dev", tok, items_all, strat)
    init_dev = {r["item_id"]: r for f in RC.INITIAL for r in jl(f) if r["split"] == "dev"}
    probe_dev, _ = RC.dev_probe_cv(Dv["ids"], canon, init_dev)
    ps = {r["item_id"]: r["p_correct"] for r in jl(RC.PROBE_TEST)}
    keep = np.array([i in ps for i in T["ids"]])
    if not keep.all():
        for k in ("y0", "s", "len", "lpmin", "agree", "a0_prompt", "verbal_prompt", "samp_gen"):
            T[k] = T[k][keep]
        T["strata"] = [s for s, m in zip(T["strata"], keep) if m]
        T["ids"] = [i for i, m in zip(T["ids"], keep) if m]
        for d in ("after", "gen", "proc"):
            T[d] = {c: v[keep] for c, v in T[d].items()}
    probe_test = np.array([ps[i] for i in T["ids"]])
    boot = Boot([canon[i] for i in T["ids"]], n_boot, SEED + 31)
    y0 = T["y0"]
    levels = sorted(set(T["strata"]) | set(Dv["strata"]))
    yd = Dv["y0"].astype(int)
    m_cheap = RC.fit(RC.feats(Dv, levels), yd)
    m_full = RC.fit(RC.feats(Dv, levels, full=True), yd)
    scores = {"verbal s": T["s"], "probe": probe_test, "answer length": -T["len"], "min token log-prob": T["lpmin"],
              "agreement with a0": T["agree"],
              "dev-fitted correctness, cheap": m_cheap.predict_proba(RC.feats(T, levels))[:, 1],
              "dev-fitted correctness, full": m_full.predict_proba(RC.feats(T, levels, full=True))[:, 1]}
    tie = RC.tie_order(T["ids"])
    sc_by = {}
    for action in RC.ACTIONS:
        gd = (Dv["after"][action] - Dv["y0"]).astype(int)
        g_cheap = RC.fit(RC.feats(Dv, levels, probe=probe_dev), gd)
        g_full = RC.fit(RC.feats(Dv, levels, probe=probe_dev, full=True), gd)
        sc = dict(scores)
        sc["dev-fitted net gain, cheap"] = -RC.gain_score(g_cheap, RC.feats(T, levels, probe=probe_test))
        sc["dev-fitted net gain, full"] = -RC.gain_score(g_full, RC.feats(T, levels, probe=probe_test, full=True))
        sc["gold wrong-first (uses gold)"] = y0
        sc_by[action] = sc
    # --- end of the repeated code
    return {"T": T, "ids": T["ids"], "y0": y0, "after": {a: T["after"][a] for a in RC.ACTIONS}, "scores": sc_by, "tie": tie,
            "boot": boot, "canon": canon, "budgets": RC.BUDGETS, "actions": RC.ACTIONS}


def build_mc(model, n_boot):
    """One MMLU-Pro model's test population, rankings per action and bootstrap, rebuilt as in mc_results.analyse()."""
    canon = {r["item_id"]: r["canonical_id"] for r in jl(ROOT / "data" / "processed" / "mmlupro" / "items.jsonl")}
    T, Dv = MR.load(model, "test"), MR.load(model, "dev")
    boot = Boot([canon[i] for i in T["ids"]], n_boot, SEED + 51)
    # --- as in mc_results.analyse()
    levels = sorted(set(T["cat"]) | set(Dv["cat"]))
    yd = Dv["y0"].astype(int)
    m_corr = MR.fit(MR.feats(Dv, levels), yd)
    tie = MR.tie_order(T["ids"])
    sc_by = {}
    for action in ("F2", "F0"):
        gd = (Dv["after"][action] - Dv["y0"]).astype(int)
        sc = {f"verbal {v}": s for v, s in T["verbal"].items()}
        sc.update({"task probe": np.nan_to_num(T["task probe"], nan=0.5), "math probe": T["math probe"], "answer length": -T["len"],
                   "agreement with a0": T["agree"], "dev-fitted correctness": m_corr.predict_proba(MR.feats(T, levels))[:, 1],
                   "dev-fitted net gain, cheap": -MR.gain_score(MR.fit(MR.feats(Dv, levels), gd), MR.feats(T, levels)),
                   "dev-fitted net gain, full": -MR.gain_score(MR.fit(MR.feats(Dv, levels, full=True), gd), MR.feats(T, levels, full=True))})
        sc_by[action] = sc
    # --- end of the repeated code
    return {"T": T, "ids": T["ids"], "y0": T["y0"], "after": {a: T["after"][a] for a in ("F2", "F0")}, "scores": sc_by, "tie": tie,
            "boot": boot, "canon": canon, "budgets": MR.BUDGETS, "actions": ("F2", "F0")}


def math_part(n_boot):
    stored = json.loads((ROOT / "analysis" / "out" / "fu_05_recheck_cost.json").read_text(encoding="utf-8"))["policies"]
    B = build_math(n_boot)
    out = {"n": len(B["ids"])}
    for action in B["actions"]:
        out[action] = table(B["y0"], B["after"][action], B["scores"][action], B["tie"], B["budgets"], B["boot"], stored[action]["budgets"])
    return out


def mc_part(model, n_boot):
    stored = json.loads((ROOT / "analysis" / "out" / "v2_02_mc_results.json").read_text(encoding="utf-8"))["models"][model]["policies"]
    B = build_mc(model, n_boot)
    out = {"n": len(B["ids"])}
    for action in B["actions"]:
        out[action] = table(B["y0"], B["after"][action], B["scores"][action], B["tie"], B["budgets"], B["boot"], stored[action]["budgets"])
    return out


def write_md(J):
    pc = lambda v: f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"  # noqa: E731
    L = ["# Selective re-check against re-checking every item (generated by analysis/fu_recheck_vs_all.py)", "",
         "**Post hoc, exploratory** (DEVIATIONS D19). Offline replay of existing per-item outputs: re-check the share k of "
         "items first in a ranking, keep a0 elsewhere. Every ranking is rebuilt as in `analysis/fu_recheck_cost.py` (math) "
         "and the frozen `analysis/mc_results.py` (MMLU-Pro), and their stored fixes, damages and estimates are reproduced "
         "exactly. Accuracy change against KEEP is in `reports/fu_05_recheck_cost.md` and `reports/v2_02_mc_results.md`, "
         "repeated here in the second table. 95 % cluster-bootstrap CIs, pp. Reading rule: \"above ALL\" only where the "
         "paired CI lies above zero; \"above KEEP\" only where the CI against a0 lies above zero.", ""]
    for part, P in J.items():
        for action in ("F2", "F0"):
            r = P[action]
            ks = list(r["budgets"])
            names = list(r["budgets"][ks[0]])
            L += [f"## {part}, re-check with {action} (n {P['n']}; ALL vs KEEP: {pc(r['ALL'])} pp)", "",
                  "Selective − ALL (pp):", "", "| Ranking | " + " | ".join(f"{k} %" for k in ks) + " |", "|---" * (len(ks) + 1) + "|"]
            L += [f"| {nm} | " + " | ".join(pc(r["budgets"][k][nm]["minus ALL"]) for k in ks) + " |" for nm in names]
            L += ["", "Selective − KEEP (pp; from the source table) and share of the selected items with a wrong a0:", "",
                  "| Ranking | " + " | ".join(f"{k} %" for k in ks) + " |", "|---" * (len(ks) + 1) + "|"]
            L += [f"| {nm} | " + " | ".join(f"{pc(r['budgets'][k][nm]['acc change (vs KEEP)'])}; "
                                            f"{100 * r['budgets'][k][nm]['share of selected initially wrong']:.0f} %" for k in ks) + " |"
                  for nm in names]
            above_all = [f"{nm} {k} %" for k in ks for nm in names if r["budgets"][k][nm]["minus ALL"][1] > 0]
            below_all = [f"{nm} {k} %" for k in ks for nm in names if r["budgets"][k][nm]["minus ALL"][2] < 0]
            above_keep = [f"{nm} {k} %" for k in ks for nm in names if r["budgets"][k][nm]["acc change (vs KEEP)"][1] > 0]
            L += ["", f"- Above ALL: {', '.join(above_all) or 'none'}.", f"- Below ALL: {', '.join(below_all) or 'none'}.",
                  f"- Above KEEP: {', '.join(above_keep) or 'none'}.", ""]
    (ROOT / "reports" / "fu_09_recheck_vs_all.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    J = {"math, 4B (D-nat test, seen)": math_part(a.n_boot)}
    for model in ("4B", "T8B"):
        J[f"MMLU-Pro, {model} (addendum test)"] = mc_part(model, a.n_boot)
    (ROOT / "analysis" / "out" / "fu_09_recheck_vs_all.json").write_text(json.dumps(J, indent=1), encoding="utf-8", newline="\n")
    write_md(J)


if __name__ == "__main__":
    main()
