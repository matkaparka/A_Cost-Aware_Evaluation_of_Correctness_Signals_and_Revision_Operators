"""Follow-up re-analyses of existing 4B test results (docs/planning/next_experiments/plan.md §1). Written on 2026-11-07,
after the P6 results and the P7 review had been seen: every quantity is exploratory (estimates and 95 % cluster-bootstrap
CIs over canonical problems; no tests, no outcome categories). Loaders and the bootstrap are the frozen analysis's.

1. H1b by verdict truth (review #2). D-nat pooled, the F3(p) rows, all p pooled and per p. Initially wrong items: W->R
   under a true verdict ("incorrect") vs a false one ("correct"). Initially right items: R->W under a true verdict
   ("correct") vs a false one ("incorrect"). Also the share of changed answers in each cell.
2. H3b output status per steering arm (review #7). Per arm: W->R on D-inj, R->W on the D-nat R->W set, changed answers,
   truncated and unparsed outputs; main minus the mean of the 5 random arms, and main minus unsteered.
3. One factorial decomposition on D-inj (review #11). The Exp 1B D-inj rows are a role (2) x label (2) x feedback
   (F0, F1, F3(1.0)) design on the same items, all initially wrong, so accuracy = W->R. Main effects are within-item
   contrasts averaged over the other factors: information F3(1.0) - F0, trigger F1 - F0, role user - assistant, label
   other - self; plus the two-way interactions.
4. Per-condition scorer disagreement (review #17): share of rows whose label changes under the second scorer v2
   (analysis/p6_confirmatory.disagreement).
5. Fix / damage table (review_and_suggestion §2-3): every 1A condition on D-nat pooled: accuracy, W->R, R->W,
   change in accuracy from B0 (= P(wrong) W->R - P(right) R->W), truncated, unparsed, changed answers, mean output
   tokens; then accuracy, W->R and R->W again for items whose a0 is ok vs truncated or unparsed.
6. H2a dev-fitted comparator (review #6b): a logistic model of a0 correctness fitted on dev (verbal s, a0 length in
   tokens, stratum) and applied to test; its AUROC, and probe minus it (paired cluster bootstrap).
7. Re-check policy at a fixed budget (review_and_suggestion §5): re-run F2 (or F0) on the k % of items with the lowest
   probe P(correct), the lowest verbal s, or at random (expected value); accuracy against keeping every a0. Test split,
   already seen: exploratory, not a claim about a deployable policy. CIs condition on the selection made on the full set.
8. Exploratory contrast count (review #15): rows carrying a 95 % CI in the exploratory reports in EXPLORATORY_REPORTS.
"Changed" compares the extracted answer with a0's (scorer normalization; an unparsed answer counts as changed when a0
parsed). "Unparsed": status unparsed, or no extracted answer.
-> reports/fu_01_reanalysis.md, analysis/out/fu_01_reanalysis.json
Usage: python analysis/fu_reanalysis.py [--spec configs/p6_spec_4B_test.json] [--n-boot 10000]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from p6_confirmatory import N_BOOT, P_LEVELS, SEED, Boot, disagreement, exp01, jl, load, probe_scores, verbal_s, wmean  # noqa: E402
from stats import auroc, auroc_diff_boot  # noqa: E402
from src.score.scorer import _norm_str  # noqa: E402

COND_1A = ["B1", "F0", "F1", "F2", "F2-scratch"] + [f"F3({p})" for p in P_LEVELS] + ["F5"] + [f"F3-perm[{k}]" for k in range(3)]
ROLE_LABEL = (("assistant", "self"), ("assistant", "other"), ("user", "self"), ("user", "other"))
FEEDBACK = ("F0", "F1", "F3(1.0)")
BUDGETS = (0.1, 0.2, 0.3, 0.5)
EXPLORATORY_REPORTS = ["reports/06_exploratory_4B.md", "reports/06_exploratory_1.7B.md", "reports/06_exploratory_8B.md",
                       "reports/04_p4_4B_channel.md", "reports/04_p4_4B_thinking.md", "reports/04_p4_8B_thinking.md",
                       "reports/04_p4_4B_streams.md", "reports/04_p4_4B_exp5.md", "reports/04_4B_exp4.md"]


def nz(x):
    return None if x is None else _norm_str(str(x))


def ci3(boot, f, mask=None):
    est, lo, hi = boot.ci(f, mask)
    return [float(est), float(lo), float(hi)]


def mean_ci(boot, v, mask=None):
    """None for an empty cell (e.g. R->W among items whose a0 was truncated: none is initially right)."""
    if mask is not None and not np.asarray(mask).any():
        return None
    return ci3(boot, lambda m: wmean(v, m), mask)


def diff_ci(boot, v1, m1, v2, m2):
    """Difference of two masked means over the same resampled clusters (rows may differ)."""
    return ci3(boot, lambda m: wmean(v1, m * m1) - wmean(v2, m * m2))


def dnat_rows(spec):
    rows = []
    for run in (spec["exp01_orig"], spec.get("exp01_ext")):
        if run:
            rows += [r for r in jl(Path(run) / "rows.jsonl") if r["exp"] == "1A" and r["data"] == "dnat" and r.get("stage") != 1]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/p6_spec_4B_test.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="fu_01_reanalysis")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    spec = json.loads((ROOT / a.spec).read_text(encoding="utf-8"))
    gold, canon, strat, init, verbal = load(spec)
    out = {"spec": a.spec, "n_boot": a.n_boot}

    rows = dnat_rows(spec)
    by_item = defaultdict(dict)
    for r in rows:
        by_item[r["item_id"]][r["cond"]] = r
    ids = sorted(i for i, v in by_item.items() if all(c in v for c in COND_1A + ["B0"]))
    c0 = np.array([bool(init[i]["correct"]) for i in ids])
    a0_ok = np.array([init[i]["status"] == "ok" for i in ids])
    boot = Boot([canon[i] for i in ids], a.n_boot, SEED)

    # ---- 5. fix / damage table
    def col(c, key):
        return np.array([key(by_item[i][c]) for i in ids], float)
    correct = lambda r: bool(r["correct"])
    trunc = lambda r: r.get("status") == "truncated" or r.get("finish_reason") == "length"
    unparsed = lambda r: r.get("status") == "unparsed" or (r.get("status") == "ok" and r.get("pred") is None)
    changed = lambda r: nz(r.get("pred")) != nz(init[r["item_id"]].get("pred"))
    acc0 = c0.astype(float)
    t5 = {"n": len(ids), "n_wrong": int((~c0).sum()), "a0 not ok": int((~a0_ok).sum()), "cells": {}, "by a0 status": {}}
    for c in COND_1A:
        v = col(c, correct)
        t5["cells"][c] = {"acc": mean_ci(boot, v), "W->R": mean_ci(boot, v, ~c0), "R->W": mean_ci(boot, 1 - v, c0),
                          "acc - B0": mean_ci(boot, v - acc0), "truncated": float(col(c, trunc).mean()),
                          "unparsed": float(col(c, unparsed).mean()), "changed": float(col(c, changed).mean()),
                          "mean out tokens": float(col(c, lambda r: r.get("out_tokens") or 0).mean())}
        t5["by a0 status"][c] = {k: {"n": int(m.sum()), "acc": mean_ci(boot, v, m), "W->R": mean_ci(boot, v, m & ~c0),
                                     "R->W": mean_ci(boot, 1 - v, m & c0)}
                                 for k, m in (("a0 ok", a0_ok), ("a0 truncated or unparsed", ~a0_ok)) if m.any()}
    out["fix_damage"] = t5

    # ---- 1. H1b by verdict truth
    f3 = [(i, p, by_item[i][f"F3({p})"]) for i in ids for p in P_LEVELS]
    fb = Boot([canon[i] for i, _, _ in f3], a.n_boot, SEED + 11)
    w = np.array([not init[i]["correct"] for i, _, _ in f3])
    ver = np.array([bool(r["verdict"]) for _, _, r in f3])          # True = the checker says "correct"
    v1 = np.array([bool(r["correct"]) for _, _, r in f3], float)
    ch = np.array([changed(r) for _, _, r in f3], float)
    pa = np.array([p for _, p, _ in f3])
    h1b = {"rows": len(f3), "pooled": {}, "per p": {}}

    def cells(mask):
        d = {"wrong, true verdict ('incorrect')": w & ~ver & mask, "wrong, false verdict ('correct')": w & ver & mask,
             "right, true verdict ('correct')": ~w & ver & mask, "right, false verdict ('incorrect')": ~w & ~ver & mask}
        res = {}
        for k, m in d.items():
            if m.any():
                rate = v1 if k.startswith("wrong") else 1 - v1
                res[k] = {"n": int(m.sum()), "W->R" if k.startswith("wrong") else "R->W": mean_ci(fb, rate, m),
                          "changed": mean_ci(fb, ch, m)}
        if (d["wrong, true verdict ('incorrect')"].any() and d["wrong, false verdict ('correct')"].any()):
            res["W->R: true - false verdict"] = diff_ci(fb, v1, d["wrong, true verdict ('incorrect')"], v1,
                                                        d["wrong, false verdict ('correct')"])
        if (d["right, false verdict ('incorrect')"].any() and d["right, true verdict ('correct')"].any()):
            res["R->W: false - true verdict"] = diff_ci(fb, 1 - v1, d["right, false verdict ('incorrect')"], 1 - v1,
                                                        d["right, true verdict ('correct')"])
        return res
    h1b["pooled"] = cells(np.ones(len(f3), bool))
    for p in P_LEVELS:
        h1b["per p"][str(p)] = cells(pa == p)
    out["H1b_by_verdict"] = h1b

    # ---- 2. H3b per-arm output status
    st = jl(spec["exp3_rows"])
    arms = sorted({r["run"] for r in st})
    sitems = sorted({(r["data"], r["item_id"]) for r in st})
    S = {(r["run"], r["data"], r["item_id"]): r for r in st}
    sb = Boot([canon[i] for _, i in sitems], a.n_boot, SEED + 12)
    dinj = np.array([d == "dinj" for d, _ in sitems])
    get = lambda arm, f: np.array([f(S[(arm, d, i)]) for d, i in sitems], float)
    metrics = {"truncated": trunc, "unparsed": unparsed, "changed": lambda r: bool(r.get("changed")),
               "correct": correct}
    h3b = {"n D-inj": int(dinj.sum()), "n R->W set": int((~dinj).sum()), "arms": {}, "main - random mean": {},
           "main - unsteered": {}}
    for arm in arms:
        h3b["arms"][arm] = {"W->R (D-inj)": mean_ci(sb, get(arm, correct), dinj),
                            "R->W (D-nat)": mean_ci(sb, 1 - get(arm, correct), ~dinj),
                            **{k: mean_ci(sb, get(arm, f)) for k, f in metrics.items() if k != "correct"}}
    rnd = spec["exp3_random"]
    for k, f in metrics.items():
        main_v = get(spec["exp3_main"], f)
        rand_v = np.mean([get(x, f) for x in rnd], axis=0)
        uns_v = get("unsteered", f)
        for name, base in (("main - random mean", rand_v), ("main - unsteered", uns_v)):
            h3b[name][f"{k} (D-inj)"] = mean_ci(sb, main_v - base, dinj)
            h3b[name][f"{k} (D-nat R->W set)"] = mean_ci(sb, main_v - base, ~dinj)
    out["H3b_status"] = h3b

    # ---- 3. factorial decomposition on D-inj
    b = exp01(spec["exp01_1b"], "dinj", "1B", gold, False)
    names = [f"{r}/{l}/{f}" for f in FEEDBACK for r, l in ROLE_LABEL]
    bid = sorted(i for i, v in b.items() if all(c in v for c in names))
    X = {c: np.array([b[i][c] for i in bid], float) for c in names}
    db = Boot([canon[i] for i in bid], a.n_boot, SEED + 13)
    avg = lambda conds: np.mean([X[c] for c in conds], axis=0)
    at = lambda f, r=None, l=None: avg([f"{rr}/{ll}/{f}" for rr, ll in ROLE_LABEL if (r in (None, rr) and l in (None, ll))])
    lab = lambda f: avg([f"{r}/other/{f}" for r in ("assistant", "user")]) - avg([f"{r}/self/{f}" for r in ("assistant", "user")])
    rol = lambda f: avg([f"user/{l}/{f}" for l in ("self", "other")]) - avg([f"assistant/{l}/{f}" for l in ("self", "other")])
    eff = {"information: F3(1.0) - F0": at("F3(1.0)") - at("F0"), "trigger: F1 - F0": at("F1") - at("F0"),
           "role: user - assistant": np.mean([rol(f) for f in FEEDBACK], axis=0),
           "label: other - self": np.mean([lab(f) for f in FEEDBACK], axis=0),
           "label x trigger: label at F1 - label at F0": lab("F1") - lab("F0"),
           "label x information: label at F3(1.0) - label at F0": lab("F3(1.0)") - lab("F0"),
           "role x trigger: role at F1 - role at F0": rol("F1") - rol("F0"),
           "role x information: role at F3(1.0) - role at F0": rol("F3(1.0)") - rol("F0"),
           "role x label: (u/o - u/s) - (a/o - a/s), mean over feedback": np.mean(
               [(X[f"user/other/{f}"] - X[f"user/self/{f}"]) - (X[f"assistant/other/{f}"] - X[f"assistant/self/{f}"])
                for f in FEEDBACK], axis=0)}
    out["D-inj factorial"] = {"n": len(bid), "cells": {c: mean_ci(db, X[c]) for c in names},
                              "effects": {k: mean_ci(db, v) for k, v in eff.items()}}

    # ---- 4. scorer disagreement per condition
    out["scorer disagreement"] = disagreement(spec)

    # ---- 6 and 7 need the H2a item set
    ps = probe_scores(spec)
    h2 = [i for i in ids if i in ps and verbal_s(verbal.get(i, {})) is not None]
    y = np.array([bool(init[i]["correct"]) for i in h2])
    cl = [canon[i] for i in h2]
    strata = sorted({strat[i] for i in h2})

    def feats(rs, vb):
        return np.array([[verbal_s(vb[r["item_id"]]), len(r["a0_token_ids"])] +
                         [float(strat[r["item_id"]] == s) for s in strata] for r in rs], float)
    dev_init = [r for f in spec["initial"] for r in jl(f) if r["split"] == "dev"]
    dev_verbal = {r["item_id"]: r for f in spec["verbal"] for r in jl(f) if r["split"] == "dev"}
    dev_init = [r for r in dev_init if verbal_s(dev_verbal.get(r["item_id"], {})) is not None and r["item_id"] in strat]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(feats(dev_init, dev_verbal), np.array([bool(r["correct"]) for r in dev_init]))
    comb = model.predict_proba(feats([init[i] for i in h2], verbal))[:, 1]
    probe = np.array([ps[i] for i in h2])
    vs = np.array([verbal_s(verbal[i]) for i in h2])
    r6 = auroc_diff_boot(y, probe, comb, clusters=cl, n_boot=a.n_boot, seed=SEED)
    r6v = auroc_diff_boot(y, comb, vs, clusters=cl, n_boot=a.n_boot, seed=SEED)
    out["H2a dev-fitted comparator"] = {"n test": len(h2), "n dev": len(dev_init), "features": ["verbal s", "a0 tokens"] + strata,
                                        "auroc": {"probe": auroc(y, probe), "verbal s": auroc(y, vs), "dev-fitted combined": auroc(y, comb)},
                                        "probe - combined": [r6["diff"], r6["lo"], r6["hi"]],
                                        "combined - verbal s": [r6v["diff"], r6v["lo"], r6v["hi"]]}

    # ---- 7. re-check policy
    hb = Boot(cl, a.n_boot, SEED + 14)
    y0 = y.astype(float)
    pol = {"n": len(h2), "acc keep a0": mean_ci(hb, y0), "actions": {}}
    for action in ("F2", "F0"):
        after = np.array([by_item[i][action]["correct"] for i in h2], float)
        res = {"acc if every item": mean_ci(hb, after)}
        for k in BUDGETS:
            n_sel = int(round(k * len(h2)))
            sel = {}
            for name, score in (("probe", probe), ("verbal s", vs)):
                m = np.zeros(len(h2), bool)
                m[np.argsort(score, kind="stable")[:n_sel]] = True
                sel[name] = np.where(m, after, y0)
            rand = y0 + k * (after - y0)                        # expected value under random selection
            res[f"{int(k * 100)}%"] = {"probe": mean_ci(hb, sel["probe"]), "verbal s": mean_ci(hb, sel["verbal s"]),
                                       "random (expected)": mean_ci(hb, rand),
                                       "probe - verbal s": mean_ci(hb, sel["probe"] - sel["verbal s"]),
                                       "probe - random": mean_ci(hb, sel["probe"] - rand)}
        pol["actions"][action] = res
    out["re-check policy"] = pol

    # ---- 8. exploratory contrast count
    cnt = {}
    for f in EXPLORATORY_REPORTS:
        p = ROOT / f
        if p.exists():
            cnt[f] = sum(1 for ln in p.read_text(encoding="utf-8").splitlines()
                         if ln.startswith("|") and re.search(r"\[\s*[+-]?\d[\d.]*\s*,\s*[+-]?\d", ln))
    out["exploratory CI rows"] = {"per report": cnt, "total": sum(cnt.values())}

    (ROOT / "analysis" / "out" / f"{a.out}.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    write_md(out, a)


def write_md(o, a):
    pc = lambda v: "—" if v is None else f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"
    dp = lambda v: "—" if v is None else f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"
    p1 = lambda x: f"{100 * x:.1f}"
    L = ["# Follow-up re-analyses of existing 4B test results (generated by analysis/fu_reanalysis.py)", "",
         f"Spec `{o['spec']}`; {o['n_boot']} cluster-bootstrap resamples. **Exploratory**, written 2026-11-07 after the "
         "P6 results and the P7 review were seen (docs/planning/next_experiments/plan.md §1): estimates and 95 % CIs, "
         "no tests. Rates in %, differences in pp.", ""]
    t = o["fix_damage"]
    L += [f"## 1. Fix / damage per 1A condition (D-nat pooled, n {t['n']}, {t['n_wrong']} initially wrong; "
          f"{t['a0 not ok']} with a0 truncated or unparsed)", "",
          "| Condition | Acc | W->R | R->W | Acc − B0 | Truncated | Unparsed | Changed | Mean out tokens |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {c} | {pc(v['acc'])} | {pc(v['W->R'])} | {pc(v['R->W'])} | {dp(v['acc - B0'])} | {p1(v['truncated'])} | "
          f"{p1(v['unparsed'])} | {p1(v['changed'])} | {v['mean out tokens']:.0f} |" for c, v in t["cells"].items()]
    L += ["", "By a0 status:", "", "| Condition | a0 status | n | Acc | W->R | R->W |", "|---|---|---|---|---|---|"]
    for c, d in t["by a0 status"].items():
        L += [f"| {c} | {k} | {v['n']} | {pc(v['acc'])} | {pc(v['W->R'])} | {pc(v['R->W'])} |" for k, v in d.items()]
    h = o["H1b_by_verdict"]
    L += ["", f"## 2. H1b by verdict truth (F3(p) rows, {h['rows']} item × p rows)", "",
          "| Scope | Cell | n | W->R or R->W | Changed |", "|---|---|---|---|---|"]
    for scope, d in [("all p", h["pooled"])] + [(f"p = {p}", d) for p, d in h["per p"].items()]:
        for k, v in d.items():
            if isinstance(v, dict):
                rate = v.get("W->R", v.get("R->W"))
                L.append(f"| {scope} | {k} | {v['n']} | {pc(rate)} | {pc(v['changed'])} |")
        for k in ("W->R: true - false verdict", "R->W: false - true verdict"):
            if k in d:
                L.append(f"| {scope} | {k} | | {dp(d[k])} | |")
    s = o["H3b_status"]
    L += ["", f"## 3. H3b output status per steering arm (D-inj n {s['n D-inj']}; D-nat R->W set n {s['n R->W set']})", "",
          "| Arm | W->R (D-inj) | R->W (D-nat) | Truncated | Unparsed | Changed |", "|---|---|---|---|---|---|"]
    L += [f"| {arm} | {pc(v['W->R (D-inj)'])} | {pc(v['R->W (D-nat)'])} | {pc(v['truncated'])} | {pc(v['unparsed'])} | "
          f"{pc(v['changed'])} |" for arm, v in s["arms"].items()]
    L += ["", "| Quantity | Main − mean of random arms | Main − unsteered |", "|---|---|---|"]
    L += [f"| {k} | {dp(v)} | {dp(s['main - unsteered'][k])} |" for k, v in s["main - random mean"].items()]
    f = o["D-inj factorial"]
    L += ["", f"## 4. One factorial decomposition on D-inj (Exp 1B rows, n {f['n']}; accuracy = W->R)", "",
          "| Cell (role/label/feedback) | W->R |", "|---|---|"]
    L += [f"| {c} | {pc(v)} |" for c, v in f["cells"].items()]
    L += ["", "| Effect (within-item contrast) | Estimate [95 % CI] |", "|---|---|"]
    L += [f"| {k} | {dp(v)} |" for k, v in f["effects"].items()]
    L += ["", "## 5. Scorer disagreement per condition (share of rows whose label changes under the second scorer v2)", "",
          "| Condition | Disagreement % |", "|---|---|"]
    L += [f"| {k} | {100 * v:.2f} |" for k, v in o["scorer disagreement"].items()]
    c = o["H2a dev-fitted comparator"]
    L += ["", f"## 6. H2a dev-fitted comparator (test n {c['n test']}; fitted on dev n {c['n dev']}; "
          f"features: {', '.join(c['features'])})", "", "| Predictor | AUROC |", "|---|---|"]
    L += [f"| {k} | {v:.3f} |" for k, v in c["auroc"].items()]
    L += ["", f"Probe − dev-fitted combined: {dp(c['probe - combined'])} (AUROC × 100). "
              f"Combined − verbal s: {dp(c['combined - verbal s'])}.", ""]
    p = o["re-check policy"]
    L += [f"## 7. Re-check policy at a fixed budget (H2a item set, n {p['n']}; keeping every a0: {pc(p['acc keep a0'])})", "",
          "| Action | Budget | Probe-selected | Verbal-selected | Random (expected) | Probe − verbal | Probe − random |",
          "|---|---|---|---|---|---|---|"]
    for act, d in p["actions"].items():
        L.append(f"| {act} | every item | {pc(d['acc if every item'])} | | | | |")
        L += [f"| {act} | {k} | {pc(v['probe'])} | {pc(v['verbal s'])} | {pc(v['random (expected)'])} | "
              f"{dp(v['probe - verbal s'])} | {dp(v['probe - random'])} |" for k, v in d.items() if k != "acc if every item"]
    e = o["exploratory CI rows"]
    L += ["", f"## 8. Exploratory contrasts reported with a CI: {e['total']} table rows", "", "| Report | Rows with a 95 % CI |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in e["per report"].items()]
    L.append("\nThis report adds its own rows on top of these.")
    (ROOT / "reports" / f"{a.out}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
