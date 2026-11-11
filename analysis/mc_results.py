"""Independent validation on MMLU-Pro: results (docs/planning/validation_v2/addendum.md; DEVIATIONS D18). Written
and frozen with the addendum before any test output of the new task existed. Estimates with 95 % cluster-bootstrap
CIs (clusters = canonical stem, 10,000 resamples); no tests.

Per model (4B, T8B), on the --split items with every condition:
1. Conditions against a0 (B0): net accuracy change, W->R, R->W, generated tokens per item (both turns of F2 and
   F2-scratch), and F2 - F0, F2 - F1, F2 - F2-scratch.
2. Signals: AUROC for a0 correctness of the verbal readout W0 (and a dev-selected variant if present), the task probe
   (selected and fitted on dev), the frozen math probe, agreement with a0 (5 samples), answer length; task probe minus
   verbal W0 (paired).
3. Selective re-check with F2 (primary) and F0 (secondary): re-check the share k of items first in each ranking, keep
   a0 elsewhere; rankings: random (expected), verbal W0, verbal variant, task probe, math probe, answer length,
   agreement with a0, a dev-fitted correctness model (verbal W0, length, category) and dev-fitted net-gain models
   (P(fix) - P(damage) under the action; cheap: verbal W0, length, category; full: + agreement), fitted on the dev
   split of the same model. Accuracy change, fixes, damages, incremental generated tokens per item, and paired
   differences against the verbal-W0 order and against random.
4. T8B minus 4B on the common items: F2 - a0, F0 - a0, and task probe minus verbal (AUROC difference).
The addendum names the primary comparisons and their reading rules; this script computes them and everything else.
Usage: python analysis/mc_results.py [--split test] [--n-boot 10000]
-> reports/v2_02_mc_results.md, analysis/out/v2_02_mc_results.json (with --split dev: v2_02_mc_results_dev.*)
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
from fu_crossmodel import wauc  # noqa: E402
from p6_confirmatory import N_BOOT, SEED, Boot, wmean  # noqa: E402

RES = ROOT / "results" / "v2"
CONDS = ["F0", "F1", "F2", "F2-scratch", "F3(1.0)", "F3(0.5)"]
BUDGETS = (0.1, 0.2, 0.3, 0.5, 1.0)
PRIMARY_BUDGETS = ("10", "20")


def jl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def latest(pattern):
    hits = sorted(RES.glob(pattern))
    return hits[-1] if hits else None


def exp_of(split):
    return "fu-dev" if split == "dev" else "fu"


def load(model, split):
    e = exp_of(split)
    di = latest(f"{e}/{model}/{e}-mc-initial-{split}-{model}-*")
    dc = latest(f"{e}/{model}/{e}-mc-conds-{split}-{model}-*")
    dk = latest(f"{e}/{model}/{e}-mc-consist-{split}-{model}-*")
    dh = latest(f"fu/{model}/fu-mc-hf-{model}-*") if split == "test" else latest(f"fu-dev/{model}/fu-dev-mc-hf-{model}-*")
    if not all((di, dc, dk, dh)):
        return None
    init = {r["item_id"]: r for r in jl(di / "initial.jsonl")}
    rows = defaultdict(dict)
    for r in jl(dc / "rows.jsonl"):
        rows[r["item_id"]][r["cond"]] = r
    ag = {r["item_id"]: r for r in jl(dk / "consistency.jsonl")}
    vb = {r["item_id"]: r for r in jl(dh / "verbal.jsonl")}
    ps = {r["item_id"]: r for r in jl(dh / "probe_scores.jsonl")}
    ids = sorted(i for i in init if i in ag and i in vb and i in ps and all(c in rows[i] for c in CONDS + ["B0"]))
    variants = sorted(k for k in vb[ids[0]] if k.startswith("W"))
    s_of = lambda r: r["s"] if r["s"] is not None else (1e6 if r["p_neg"] == 0 else -1e6)  # noqa: E731
    D = {"ids": ids, "y0": np.array([bool(init[i]["correct"]) for i in ids], float),
         "cat": [init[i]["category"] for i in ids], "len": np.array([init[i]["out_tokens"] for i in ids], float),
         "agree": np.array([ag[i]["agreement"] for i in ids]), "samp_gen": np.array([ag[i]["sample_out_tokens"] for i in ids], float),
         "verbal": {v: np.array([s_of(vb[i][v]) for i in ids]) for v in variants},
         "task probe": np.array([ps[i]["p_task_probe"] if ps[i]["p_task_probe"] is not None else np.nan for i in ids]),
         "math probe": np.array([ps[i]["p_math_probe"] for i in ids]),
         "after": {c: np.array([bool(rows[i][c]["correct"]) for i in ids], float) for c in CONDS},
         "gen": {c: np.array([rows[i][c]["out_tokens"] + rows[i][c].get("turn1_out_tokens", 0) for i in ids], float) for c in CONDS},
         "dirs": {k: str(v.relative_to(ROOT)) for k, v in (("initial", di), ("conds", dc), ("consist", dk), ("hf", dh))}}
    return D


def feats(D, levels, full=False):
    cols = [D["verbal"]["W0"], D["len"]] + [np.array([c == lv for c in D["cat"]], float) for lv in levels]
    if full:
        cols.append(D["agree"])
    return np.column_stack(cols)


def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)


def gain_score(model, X):
    p = model.predict_proba(X)
    cls = list(model.classes_)
    return (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)


def tie_order(ids):
    return np.array([int(hashlib.sha256(i.encode()).hexdigest()[:12], 16) for i in ids], float)


def analyse(T, Dv, canon, n_boot):
    n = len(T["ids"])
    boot = Boot([canon[i] for i in T["ids"]], n_boot, SEED + 51)
    ci = lambda v: list(boot.ci(lambda w: wmean(v, w)))  # noqa: E731
    y0 = T["y0"]
    wrong, right = y0 == 0, y0 == 1
    o = {"n": n, "a0 accuracy": float(y0.mean()), "initially wrong": int(wrong.sum()), "runs": T["dirs"]}
    o["conditions"] = {c: {"net change": ci(T["after"][c] - y0), "W->R": float(T["after"][c][wrong].mean()),
                           "R->W": float(1 - T["after"][c][right].mean()), "generated tokens / item": float(T["gen"][c].mean())}
                       for c in CONDS}
    o["a0 generated tokens / item"] = float(T["len"].mean())
    o["contrasts"] = {f"{a} - {b}": ci(T["after"][a] - T["after"][b]) for a, b in (("F2", "F0"), ("F2", "F1"), ("F2", "F2-scratch"),
                                                                                    ("F3(1.0)", "F0"), ("F3(1.0)", "F3(0.5)"))}
    sig = {f"verbal {v}": s for v, s in T["verbal"].items()}
    sig.update({"task probe": T["task probe"], "math probe": T["math probe"], "agreement with a0": T["agree"],
                "answer length (shorter = correct)": -T["len"]})
    one = np.ones(n, int)
    o["auroc"] = {k: float(wauc(y0, np.nan_to_num(s, nan=0.5), one)) for k, s in sig.items()}
    o["task probe - verbal W0"] = list(boot.ci(lambda w: wauc(y0, np.nan_to_num(T["task probe"], nan=0.5), w) - wauc(y0, T["verbal"]["W0"], w)))
    o["agreement - verbal W0"] = list(boot.ci(lambda w: wauc(y0, T["agree"], w) - wauc(y0, T["verbal"]["W0"], w)))
    levels = sorted(set(T["cat"]) | set(Dv["cat"]))
    yd = Dv["y0"].astype(int)
    m_corr = fit(feats(Dv, levels), yd)
    tie = tie_order(T["ids"])
    pol = {}
    for action in ("F2", "F0"):
        after, act_gen = T["after"][action], T["gen"][action]
        gd = (Dv["after"][action] - Dv["y0"]).astype(int)
        sc = {f"verbal {v}": s for v, s in T["verbal"].items()}
        sc.update({"task probe": np.nan_to_num(T["task probe"], nan=0.5), "math probe": T["math probe"], "answer length": -T["len"],
                   "agreement with a0": T["agree"], "dev-fitted correctness": m_corr.predict_proba(feats(T, levels))[:, 1],
                   "dev-fitted net gain, cheap": -gain_score(fit(feats(Dv, levels), gd), feats(T, levels)),
                   "dev-fitted net gain, full": -gain_score(fit(feats(Dv, levels, full=True), gd), feats(T, levels, full=True))})
        sig_gen = {k: (float(T["samp_gen"].mean()) if k in ("agreement with a0", "dev-fitted net gain, full") else 0.0) for k in sc}
        res = {"every item": {"acc change": ci(after - y0), "generated / item": float(act_gen.mean())}, "budgets": {}}
        for k in BUDGETS:
            n_sel = int(round(k * n))
            fin = {}
            row = {}
            for name, s in sc.items():
                m = np.zeros(n, bool)
                m[np.lexsort((tie, s))[:n_sel]] = True
                fin[name] = np.where(m, after, y0)
                row[name] = {"mask": m}
            exp_rand = y0 + (n_sel / n) * (after - y0)
            for name in sc:
                f, m = fin[name], row[name].pop("mask")
                row[name].update({"acc change": ci(f - y0), "fixes": int(((f == 1) & wrong).sum()), "damages": int(((f == 0) & right).sum()),
                                  "incremental generated / item": float(sig_gen[name] + act_gen[m].sum() / n),
                                  "minus verbal W0": ci(f - fin["verbal W0"]) if name != "verbal W0" else None,
                                  "minus random": ci(f - exp_rand)})
            row["random (expected)"] = {"acc change": ci(exp_rand - y0), "incremental generated / item": float((n_sel / n) * act_gen.mean())}
            res["budgets"][f"{int(round(100 * k))}"] = row
        pol[action] = res
    o["policies"] = pol
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    canon = {r["item_id"]: r["canonical_id"] for r in jl(ROOT / "data" / "processed" / "mmlupro" / "items.jsonl")}
    out = {"split": a.split, "models": {}}
    data = {}
    for model in ("4B", "T8B"):
        T, Dv = load(model, a.split), load(model, "dev")
        if T is None or Dv is None:
            out["models"][model] = None
            continue
        data[model] = T
        out["models"][model] = analyse(T, Dv, canon, a.n_boot)
    if all(m in data for m in ("4B", "T8B")):
        common = sorted(set(data["4B"]["ids"]) & set(data["T8B"]["ids"]))
        idx = {m: np.array([data[m]["ids"].index(i) for i in common]) for m in data}
        boot = Boot([canon[i] for i in common], a.n_boot, SEED + 52)
        dd = lambda f: list(boot.ci(lambda w: wmean(f, w)))  # noqa: E731
        x = {m: {c: data[m]["after"][c][idx[m]] - data[m]["y0"][idx[m]] for c in ("F2", "F0")} for m in data}
        out["T8B - 4B"] = {"n common": len(common), "F2 - a0": dd(x["T8B"]["F2"] - x["4B"]["F2"]),
                           "F0 - a0": dd(x["T8B"]["F0"] - x["4B"]["F0"])}

        def gap(m, w):
            y = data[m]["y0"][idx[m]]
            return wauc(y, np.nan_to_num(data[m]["task probe"][idx[m]], nan=0.5), w) - wauc(y, data[m]["verbal"]["W0"][idx[m]], w)
        out["T8B - 4B"]["task probe - verbal W0 (AUROC)"] = list(boot.ci(lambda w: gap("T8B", w) - gap("4B", w)))
    suffix = "" if a.split == "test" else "_dev"
    (ROOT / "analysis" / "out" / f"v2_02_mc_results{suffix}.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    pc = lambda v: "—" if v is None else f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"  # noqa: E731
    L = [f"# MMLU-Pro validation, {a.split} (generated by analysis/mc_results.py)", "",
         "Addendum: docs/planning/validation_v2/addendum.md (DEVIATIONS D18). 95 % cluster-bootstrap CIs (clusters = canonical "
         "stem). pp; AUROC differences × 100. Token counts per item; no wall-clock times.", ""]
    for model, o in out["models"].items():
        if o is None:
            L += [f"## {model}: runs missing", ""]
            continue
        L += [f"## {model} (n {o['n']}, a0 accuracy {100 * o['a0 accuracy']:.1f} %, {o['initially wrong']} initially wrong; "
              f"a0 {o['a0 generated tokens / item']:.0f} tokens / item)", "",
              "| Condition | Net change (pp) | W->R | R->W | Generated tokens / item |", "|---|---|---|---|---|"]
        L += [f"| {c} | {pc(v['net change'])} | {100 * v['W->R']:.1f} % | {100 * v['R->W']:.1f} % | {v['generated tokens / item']:.0f} |"
              for c, v in o["conditions"].items()]
        L += ["", "| Contrast | pp |", "|---|---|"] + [f"| {k} | {pc(v)} |" for k, v in o["contrasts"].items()]
        L += ["", "| Signal | AUROC |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in o["auroc"].items()]
        L += ["", f"Task probe − verbal W0: {pc(o['task probe - verbal W0'])}; agreement − verbal W0: {pc(o['agreement - verbal W0'])}.", ""]
        for action, r in o["policies"].items():
            L += [f"### Re-check with {action} (every item: {pc(r['every item']['acc change'])} pp, "
                  f"{r['every item']['generated / item']:.0f} tokens / item)", ""]
            for k, row in r["budgets"].items():
                L += [f"**{k} % of items**", "", "| Ranking | Acc change (pp) | Fixes | Damages | Incr. tokens / item | − verbal W0 (pp) | − random (pp) |",
                      "|---|---|---|---|---|---|---|"]
                for name, v in row.items():
                    if name == "random (expected)":
                        L.append(f"| random (expected) | {pc(v['acc change'])} | — | — | {v['incremental generated / item']:.0f} | — | — |")
                    else:
                        L.append(f"| {name} | {pc(v['acc change'])} | {v['fixes']} | {v['damages']} | {v['incremental generated / item']:.0f} | "
                                 f"{pc(v['minus verbal W0'])} | {pc(v['minus random'])} |")
                L.append("")
    if "T8B - 4B" in out:
        t = out["T8B - 4B"]
        L += [f"## T8B − 4B on {t['n common']} common items", "", f"F2 − a0: {pc(t['F2 - a0'])} pp; F0 − a0: {pc(t['F0 - a0'])} pp; "
              f"task probe − verbal W0 (AUROC): {pc(t['task probe - verbal W0 (AUROC)'])}.", ""]
    (ROOT / "reports" / f"v2_02_mc_results{suffix}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
