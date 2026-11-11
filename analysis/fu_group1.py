"""Group 1 follow-ups on the 4B math pools (docs/planning/next_experiments/plan.md; DEVIATIONS D14). Exploratory:
estimates with 95 % cluster-bootstrap CIs (10,000 resamples, clusters = canonical problem), no tests.

1. Stated reliability, F3s(p) (src/exp/stated_p.py), D-nat test pooled (original + expansion), paired with the main
   run's F3(p) on the same items and the same verdicts: accuracy, W->R, R->W and answer-change rate per p, split by
   verdict; F3s(p) - F3(p) per p; and the p-slope difference [F3s(1.0) - F3s(0.5)] - [F3(1.0) - F3(0.5)].
   Exp 0: valid rate, accuracy and gain ratio (acc - 0.5) / (p - 0.5) per p, stated and unstated (descriptive).
2. H2a comparators the P1 tables lacked (src/gen/h2a_extra.py), on the H2a item set (D-nat pooled with a probe score
   and a verbal score): AUROC for a0 correctness of the mean and the minimum token log-prob of a0, and of the
   5-sample consistency with a0; probe minus each (paired cluster bootstrap). Sample accuracy uses the gold answer and
   is not a predictor; it is reported as a mean only. Plus a dev-fitted combined model (verbal s, a0 length, stratum,
   mean log-prob, consistency; logistic, fitted on dev) applied to test, as analysis/fu_reanalysis.py section 6.
3. H3c (src/probe/h3c.py): per-layer out-of-fold AUROC of role and label at the content tokens, both positions, as
   computed on the run host; the layer-0 invariant.
Input: configs/fu1_spec_4B_test.json (the P6 spec plus the follow-up run directories).
-> reports/fu_02_group1.md, analysis/out/fu_02_group1.json
Usage: python analysis/fu_group1.py [--spec configs/fu1_spec_4B_test.json]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from fu_reanalysis import dnat_rows, mean_ci, nz  # noqa: E402
from p6_confirmatory import N_BOOT, P_LEVELS, SEED, Boot, jl, load, probe_scores, verbal_s  # noqa: E402
from stats import auroc, auroc_diff_boot  # noqa: E402


def opt(spec, key, name):
    d = spec.get(key)
    return jl(Path(d) / name) if d and (ROOT / d / name).exists() else []


def stated(spec, init, canon):
    main = defaultdict(dict)
    for r in dnat_rows(spec):
        if r["cond"].startswith("F3(") and r["cond"] != "F3-perm":
            main[r["item_id"]][r["cond"]] = r
    st = defaultdict(dict)
    rows = opt(spec, "stated_p", "rows.jsonl")
    for r in rows:
        if r["data"] == "dnat":
            st[r["item_id"]][r["cond"]] = r
    conds = [f"F3({p})" for p in P_LEVELS]
    ids = sorted(i for i in st if all(f"F3s({p})" in st[i] for p in P_LEVELS) and all(c in main[i] for c in conds))
    if not ids:
        return None
    mismatch = sum(bool(st[i][f"F3s({p})"]["verdict"]) != bool(main[i][f"F3({p})"]["verdict"]) for i in ids for p in P_LEVELS)
    boot = Boot([canon[i] for i in ids], N_BOOT, SEED)
    c0 = np.array([bool(init[i]["correct"]) for i in ids])
    out = {"n": len(ids), "verdict mismatches (must be 0)": mismatch, "per p": {}}
    acc = {}
    for p in P_LEVELS:
        cell = {}
        for kind, rr in (("stated", {i: st[i][f"F3s({p})"] for i in ids}), ("unstated", {i: main[i][f"F3({p})"] for i in ids})):
            v = np.array([bool(rr[i]["correct"]) for i in ids], float)
            ch = np.array([nz(rr[i].get("pred")) != nz(init[i].get("pred")) for i in ids], float)
            says_ok = np.array([bool(rr[i]["verdict"]) for i in ids])
            acc[(kind, p)] = v
            cell[kind] = {"accuracy": mean_ci(boot, v), "W->R": mean_ci(boot, v, ~c0), "R->W": mean_ci(boot, 1 - v, c0),
                          "change | verdict incorrect": mean_ci(boot, ch, ~says_ok),
                          "change | verdict correct": mean_ci(boot, ch, says_ok)}
        cell["stated - unstated (accuracy)"] = mean_ci(boot, acc[("stated", p)] - acc[("unstated", p)])
        out["per p"][str(p)] = cell
    s = acc[("stated", 1.0)] - acc[("stated", 0.5)]
    u = acc[("unstated", 1.0)] - acc[("unstated", 0.5)]
    out["p-slope: stated (1.0 - 0.5)"] = mean_ci(boot, s)
    out["p-slope: unstated (1.0 - 0.5)"] = mean_ci(boot, u)
    out["p-slope difference (stated - unstated)"] = mean_ci(boot, s - u)
    e0 = defaultdict(lambda: {"n": 0, "valid": 0, "correct": 0})
    for kind, rs, pre in (("stated", rows, "F3s"), ("unstated", [r for f in (spec["exp01_orig"],) for r in jl(Path(f) / "rows.jsonl")], "F3")):
        for r in rs:
            if r.get("data") == "exp0" and r["cond"].startswith(pre + "("):
                c = e0[(kind, r["cond"].split("(")[1].rstrip(")"))]
                c["n"] += 1
                c["valid"] += bool(r.get("valid"))
                c["correct"] += bool(r["correct"])
    out["exp0"] = {f"{k} p={p}": {**v, "accuracy": v["correct"] / v["n"],
                                  "gain ratio": (v["correct"] / v["n"] - 0.5) / (float(p) - 0.5) if float(p) > 0.5 else None}
                   for (k, p), v in sorted(e0.items()) if v["n"]}
    return out


def h2a(spec, init, canon, strat, verbal):
    ps = probe_scores(spec)
    lp = {r["item_id"]: r for r in opt(spec, "logprob_test", "logprob.jsonl")}
    cs = {r["item_id"]: r for r in opt(spec, "consistency_test", "consistency.jsonl")}
    if not ps or not lp or not cs:
        return None
    pool = [i for i in init if init[i]["split"] == "test" and i in ps and verbal_s(verbal.get(i, {})) is not None
            and i in lp and lp[i]["mean"] is not None and i in cs]
    y = np.array([bool(init[i]["correct"]) for i in pool])
    cl = [canon[i] for i in pool]
    S = {"probe": np.array([ps[i] for i in pool]), "verbal s": np.array([verbal_s(verbal[i]) for i in pool]),
         "mean token log-prob": np.array([lp[i]["mean"] for i in pool]),
         "min token log-prob": np.array([lp[i]["min"] for i in pool]),
         "5-sample consistency": np.array([cs[i]["consistency"] for i in pool])}
    out = {"n": len(pool), "n wrong": int((~y).sum()), "auroc": {k: auroc(y, v) for k, v in S.items()}, "probe minus": {},
           "mean sample accuracy (uses gold; not a predictor)": float(np.mean([cs[i]["sample_acc"] for i in pool]))}
    for k in ("verbal s", "mean token log-prob", "min token log-prob", "5-sample consistency"):
        r = auroc_diff_boot(y, S["probe"], S[k], clusters=cl, n_boot=N_BOOT, seed=SEED)
        out["probe minus"][k] = [r["diff"], r["lo"], r["hi"]]
    # dev-fitted combined comparator
    dlp = {r["item_id"]: r for r in opt(spec, "logprob_dev", "logprob.jsonl")}
    dcs = {r["item_id"]: r for r in opt(spec, "consistency_dev", "consistency.jsonl")}
    dev_verbal = {r["item_id"]: r for f in spec["verbal"] for r in jl(f) if r["split"] == "dev"}
    dev = [r for f in spec["initial"] for r in jl(f) if r["split"] == "dev" and r["item_id"] in dlp and r["item_id"] in dcs
           and dlp[r["item_id"]]["mean"] is not None and verbal_s(dev_verbal.get(r["item_id"], {})) is not None]
    if dev:
        strata = sorted({strat[i] for i in pool})

        def feats(ids_, vb, lpm, csm):
            return np.array([[verbal_s(vb[i]), len(init_all[i]["a0_token_ids"]), lpm[i]["mean"], csm[i]["consistency"]]
                             + [float(strat[i] == s) for s in strata] for i in ids_], float)
        init_all = {**{r["item_id"]: r for r in dev}, **init}
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        m.fit(feats([r["item_id"] for r in dev], dev_verbal, dlp, dcs), np.array([bool(r["correct"]) for r in dev]))
        comb = m.predict_proba(feats(pool, verbal, lp, cs))[:, 1]
        r = auroc_diff_boot(y, S["probe"], comb, clusters=cl, n_boot=N_BOOT, seed=SEED)
        out["dev-fitted combined"] = {"n dev": len(dev), "features": ["verbal s", "a0 tokens", "mean token log-prob",
                                                                      "5-sample consistency"] + strata,
                                      "auroc": auroc(y, comb), "probe minus": [r["diff"], r["lo"], r["hi"]]}
    return out


def h3c(spec):
    d = spec.get("h3c")
    if not d or not (ROOT / d / "metrics.json").exists():
        return None
    m = json.loads((ROOT / d / "metrics.json").read_text(encoding="utf-8"))
    out = {"n items": m["n_items"], "rows": m["rows"], "skipped": len(m["skipped_items"]), "layer0 invariant": m["layer0_invariant"],
           "curves": m["auroc"], "summary": {}}
    for pos, v in m["auroc"].items():
        for t, curve in v.items():
            k = int(np.argmax(curve))
            out["summary"][f"{t} @ {pos}"] = {"max": curve[k], "layer of max": k, "mean over layers 1+": float(np.mean(curve[1:]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/fu1_spec_4B_test.json")
    ap.add_argument("--out", default="fu_02_group1")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    spec = json.loads((ROOT / a.spec).read_text(encoding="utf-8"))
    gold, canon, strat, init, verbal = load(spec)
    out = {"spec": a.spec, "n_boot": N_BOOT, "stated p": stated(spec, init, canon), "H2a comparators": h2a(spec, init, canon, strat, verbal),
           "H3c": h3c(spec)}
    (ROOT / "analysis" / "out" / f"{a.out}.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    pc = lambda v: "—" if v is None else f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"  # noqa: E731
    dp = lambda v: "—" if v is None else f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"  # noqa: E731
    L = [f"# Group 1 follow-ups, 4B (generated by analysis/fu_group1.py)", "",
         f"Spec `{a.spec}`. **Exploratory** (DEVIATIONS D14): estimates with 95 % cluster-bootstrap CIs ({N_BOOT} "
         "resamples; clusters = canonical problem). Rates in %, differences in pp.", ""]
    s = out["stated p"]
    if s:
        L += [f"## 1. Stated reliability F3s(p) vs the frozen F3(p) (D-nat test, n {s['n']}; verdicts paired, "
              f"mismatches {s['verdict mismatches (must be 0)']})", "",
              "| p | Wrapper | Accuracy | W->R | R->W | Change \\| verdict incorrect | Change \\| verdict correct |", "|---|---|---|---|---|---|---|"]
        for p, c in s["per p"].items():
            for k in ("unstated", "stated"):
                v = c[k]
                L.append(f"| {p} | {k} | {pc(v['accuracy'])} | {pc(v['W->R'])} | {pc(v['R->W'])} | "
                         f"{pc(v['change | verdict incorrect'])} | {pc(v['change | verdict correct'])} |")
        L += ["", "| Contrast | pp |", "|---|---|"]
        L += [f"| stated − unstated accuracy, p = {p} | {dp(c['stated - unstated (accuracy)'])} |" for p, c in s["per p"].items()]
        L += [f"| {k} | {dp(s[k])} |" for k in ("p-slope: unstated (1.0 - 0.5)", "p-slope: stated (1.0 - 0.5)",
                                                 "p-slope difference (stated - unstated)")]
        L += ["", "Exp 0 (descriptive): " + "; ".join(
            f"{k}: acc {100 * v['accuracy']:.1f} %, valid {v['valid']}/{v['n']}"
            + (f", gain ratio {v['gain ratio']:.2f}" if v["gain ratio"] is not None else "") for k, v in s["exp0"].items()), ""]
    h = out["H2a comparators"]
    if h:
        L += [f"## 2. H2a comparators (H2a item set, test n {h['n']}, initially wrong {h['n wrong']})", "",
              "| Predictor | AUROC | Probe minus it (AUROC × 100) |", "|---|---|---|"]
        L += [f"| {k} | {v:.3f} | {dp(h['probe minus'][k]) if k in h['probe minus'] else '—'} |" for k, v in h["auroc"].items()]
        if "dev-fitted combined" in h:
            c = h["dev-fitted combined"]
            L.append(f"| dev-fitted combined ({', '.join(c['features'])}; dev n {c['n dev']}) | {c['auroc']:.3f} | {dp(c['probe minus'])} |")
        L += ["", f"Mean accuracy of the 5 samples (uses the gold answer; not a predictor): "
              f"{100 * h['mean sample accuracy (uses gold; not a predictor)']:.1f} %.", ""]
    h3 = out["H3c"]
    if h3:
        L += [f"## 3. H3c: role and label decodability at content tokens (D-inj test, {h3['n items']} items × 4 prompts; "
              f"layer-0 invariant {'holds' if h3['layer0 invariant'] else 'FAILED'})", "",
              "| Target @ position | Max AUROC | Layer of max | Mean over layers 1+ |", "|---|---|---|---|"]
        L += [f"| {k} | {v['max']:.3f} | {v['layer of max']} | {v['mean over layers 1+']:.3f} |" for k, v in h3["summary"].items()]
        L += ["", "The design's reading: content tokens attend to the role tokens, so decodability does not rule out "
              "role information; it does not show that the model uses it.", ""]
    (ROOT / "reports" / f"{a.out}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
