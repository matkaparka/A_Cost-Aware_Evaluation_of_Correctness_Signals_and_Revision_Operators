"""Exploratory analyses at P6 that analysis/p6_confirmatory.py does not compute (prereg §3.3): estimates and 95 %
cluster-bootstrap CIs (canonical problems; Exp 0: templates), no tests, no outcome categories. Written on 2026-11-07,
after the confirmatory results had been seen (DEVIATIONS D13); every quantity is one prereg §3.1 / §3.3 lists, with the
definition stated here. Loaders and the bootstrap are those of the frozen analysis.
- 1A on D-nat (pooled): accuracy, W->R and R->W of B1, F0, F1, F2, F2-scratch; F2 - F2-scratch and F2-scratch - F1;
  the H1b secondary estimate F3(0.5) - F0 (predicted <= 0) and the adjacent-level risk differences.
- 1B on D-inj (every item initially wrong, so accuracy = W->R), at F0, F1 and F3(1.0): the four role x label cells, the
  role and label contrasts (at F0 they repeat the confirmatory estimands), the interaction (u/o - u/s) - (a/o - a/s), and
  the diagonal (user, other) - (assistant, self), the Self-Correction Bench comparison. D-nat replication at F0 on
  initially wrong items.
- Alternative label wording (the 300-item D-inj subset, --alt): label contrast at F0 with each wording, and their
  difference, on the same items.
- Tsui cell (D-inj): W->R of internal, external, internal+wait, external+wait; external - internal; +wait - plain.
- Exp 0 (calibration and leakage control, prereg §3.3): per condition, valid rate, accuracy (invalid = wrong), and the
  gain ratio (acc - 0.5) / (p - 0.5) for F3(p), p > 0.5.
- H2a comparators (D-nat pooled, the H2a item set): AUROC for a0 correctness of the probe, the verbal score s, minus
  the answer length in tokens, and the dev accuracy of the item's stratum (a fixed predictor from dev); probe minus each
  comparator (paired cluster bootstrap); probe and verbal per stratum and per MATH level (1-5; MATH-500 and the
  expansion together; the level field of data/processed/dnat.jsonl and ext/dnat_ext.jsonl).
-> reports/<out>.md and analysis/out/<out>.json
Usage: python analysis/p6_exploratory.py --spec configs/p6_spec_4B_test.json
                                          [--alt results/main/4B/main-exp01-test-alt-4B-a7a0579] --out 06_exploratory_4B
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
from p6_confirmatory import N_BOOT, P_LEVELS, SEED, Boot, exp01, jl, load, probe_scores, verbal_s, wmean  # noqa: E402
from stats import auroc, auroc_diff_boot  # noqa: E402

ROLE_LABEL = (("assistant", "self"), ("assistant", "other"), ("user", "self"), ("user", "other"))


def ci(boot, v, mask=None):
    est, lo, hi = boot.ci(lambda m: wmean(v, m), mask)
    return [float(est), float(lo), float(hi)]


def contrasts_2x2(C, boot):
    """C columns: a/s, a/o, u/s, u/o."""
    return {"role (user - assistant)": ci(boot, (C[:, 2] + C[:, 3] - C[:, 0] - C[:, 1]) / 2),
            "label (other - self)": ci(boot, (C[:, 1] + C[:, 3] - C[:, 0] - C[:, 2]) / 2),
            "interaction (u/o - u/s) - (a/o - a/s)": ci(boot, (C[:, 3] - C[:, 2]) - (C[:, 1] - C[:, 0])),
            "diagonal (user, other) - (assistant, self)": ci(boot, C[:, 3] - C[:, 0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--alt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    spec = json.loads((ROOT / a.spec).read_text(encoding="utf-8"))
    gold, canon, strat, init, verbal = load(spec)
    out = {"spec": a.spec, "alt": a.alt, "n_boot": a.n_boot}

    # ---- 1A on D-nat
    items = {}
    for run in [spec["exp01_orig"]] + ([spec["exp01_ext"]] if spec.get("exp01_ext") else []):
        items.update(exp01(run, "dnat", "1A", gold, False))
    conds = ["B1", "F0", "F1", "F2", "F2-scratch"] + [f"F3({p})" for p in P_LEVELS]
    ids = sorted(i for i, v in items.items() if all(c in v for c in conds))
    col = lambda c: np.array([items[i][c] for i in ids], float)
    c0 = np.array([bool(init[i]["correct"]) for i in ids])
    boot = Boot([canon[i] for i in ids], a.n_boot, SEED)
    out["1A"] = {"n": len(ids), "n_wrong": int((~c0).sum()), "cells": {}, "contrasts": {}}
    for c in ["B1", "F0", "F1", "F2", "F2-scratch"]:
        v = col(c)
        out["1A"]["cells"][c] = {"acc": ci(boot, v), "W->R": ci(boot, v, ~c0), "R->W": ci(boot, 1 - v, c0)}
    for name, (x, y) in {"F2 - F2-scratch": ("F2", "F2-scratch"), "F2-scratch - F1": ("F2-scratch", "F1"),
                         "F3(0.5) - F0 (H1b secondary, predicted <= 0)": ("F3(0.5)", "F0"),
                         "F3(0.7) - F3(0.5)": ("F3(0.7)", "F3(0.5)"), "F3(0.9) - F3(0.7)": ("F3(0.9)", "F3(0.7)"),
                         "F3(1.0) - F3(0.9)": ("F3(1.0)", "F3(0.9)")}.items():
        out["1A"]["contrasts"][name] = ci(boot, col(x) - col(y))

    # ---- 1B: D-inj at F0, F1, F3(1.0); D-nat at F0 on initially wrong items
    out["1B"] = {}
    b = exp01(spec["exp01_1b"], "dinj", "1B", gold, False)
    for fb in ("F0", "F1", "F3(1.0)"):
        cells = [f"{r}/{l}/{fb}" for r, l in ROLE_LABEL]
        bid = sorted(i for i, v in b.items() if all(c in v for c in cells))
        C = np.array([[b[i][c] for c in cells] for i in bid], float)
        bb = Boot([canon[i] for i in bid], a.n_boot, SEED + 1)
        out["1B"][f"D-inj {fb}"] = {"n": len(bid), "cells": {c: ci(bb, C[:, k]) for k, c in enumerate(cells)},
                                    **contrasts_2x2(C, bb)}
    bn = exp01(spec["exp01_1b"], "dnat", "1B", gold, False)
    cells = [f"{r}/{l}/F0" for r, l in ROLE_LABEL]
    nid = sorted(i for i, v in bn.items() if all(c in v for c in cells) and not init[i]["correct"])
    C = np.array([[bn[i][c] for c in cells] for i in nid], float)
    bb = Boot([canon[i] for i in nid], a.n_boot, SEED + 1)
    out["1B"]["D-nat F0, initially wrong"] = {"n": len(nid), "cells": {c: ci(bb, C[:, k]) for k, c in enumerate(cells)},
                                              **contrasts_2x2(C, bb)}

    # ---- alternative label wording (same items, F0)
    if a.alt:
        alt = exp01(a.alt, "dinj", "1B", gold, False)
        cells = [f"{r}/{l}/F0" for r, l in ROLE_LABEL]
        aid = sorted(i for i, v in alt.items() if all(c in v for c in cells) and i in b and all(c in b[i] for c in cells))
        lab = lambda src: np.array([(src[i][cells[1]] + src[i][cells[3]] - src[i][cells[0]] - src[i][cells[2]]) / 2
                                    for i in aid])
        ab = Boot([canon[i] for i in aid], a.n_boot, SEED + 3)
        out["alt_labels"] = {"n": len(aid), "label, main wording": ci(ab, lab(b)), "label, alternative wording": ci(ab, lab(alt)),
                             "alternative - main": ci(ab, lab(alt) - lab(b))}

    # ---- Tsui cell
    t = exp01(spec["exp01_orig"], "dinj", "tsui", gold, False)
    tc = ["internal", "external", "internal+wait", "external+wait"]
    tid = sorted(i for i, v in t.items() if all(c in v for c in tc))
    T = {c: np.array([t[i][c] for i in tid], float) for c in tc}
    tb = Boot([canon[i] for i in tid], a.n_boot, SEED + 4)
    out["tsui"] = {"n": len(tid), "cells": {c: ci(tb, T[c]) for c in tc},
                   "external - internal": ci(tb, T["external"] - T["internal"]),
                   "internal+wait - internal": ci(tb, T["internal+wait"] - T["internal"]),
                   "external+wait - external": ci(tb, T["external+wait"] - T["external"])}

    # ---- Exp 0
    e0 = defaultdict(dict)
    for r in jl(Path(spec["exp01_orig"]) / "rows.jsonl"):
        if r["exp"] == "0":
            e0[r["item_id"]][r["cond"]] = (r["status"] != "invalid", bool(r["correct"]))
    e0c = sorted({c for v in e0.values() for c in v})
    eid = sorted(i for i, v in e0.items() if all(c in v for c in e0c))
    eb = Boot([i.rsplit("-w", 1)[0] for i in eid], a.n_boot, SEED + 5)
    out["exp0"] = {"n": len(eid), "templates": len({i.rsplit("-w", 1)[0] for i in eid}), "cells": {}}
    for c in e0c:
        valid = np.array([e0[i][c][0] for i in eid], float)
        acc = np.array([e0[i][c][1] for i in eid], float)
        cell = {"valid": ci(eb, valid), "acc": ci(eb, acc)}
        if c.startswith("F3("):
            p = float(c[3:-1])
            if p > 0.5:
                cell["gain ratio"] = [(x - 0.5) / (p - 0.5) for x in ci(eb, acc)]
        out["exp0"]["cells"][c] = cell

    # ---- H2a comparators
    ps = probe_scores(spec)
    level = {}
    for fn in ("data/processed/dnat.jsonl", "data/processed/ext/dnat_ext.jsonl"):
        for r in jl(fn):
            lv = r.get("level")
            if isinstance(lv, str) and lv.startswith("Level ") and lv[6:].isdigit():
                lv = int(lv[6:])
            if isinstance(lv, int):
                level[r["item_id"]] = lv
    if ps is not None:
        pool = sorted(i for i in init if strat[i] in ("gsm8k", "math500", "math_ext")
                      and (spec.get("exp01_ext") or strat[i] != "math_ext"))
        h2 = [i for i in pool if i in ps and verbal_s(verbal.get(i, {})) is not None]
        dev_acc = defaultdict(list)
        for f in spec["initial"]:
            for r in jl(f):
                if r["split"] == "dev":
                    dev_acc[strat[r["item_id"]]].append(bool(r["correct"]))
        dev_acc = {s: float(np.mean(v)) for s, v in dev_acc.items()}
        y = np.array([bool(init[i]["correct"]) for i in h2])
        cl = [canon[i] for i in h2]
        S = {"probe": np.array([ps[i] for i in h2]), "verbal s": np.array([verbal_s(verbal[i]) for i in h2]),
             "minus answer length": -np.array([len(init[i]["a0_token_ids"]) for i in h2], float),
             "stratum dev accuracy": np.array([dev_acc[strat[i]] for i in h2])}
        out["H2a comparators"] = {"n": len(h2), "dev_accuracy_by_stratum": dev_acc,
                                  "auroc": {k: auroc(y, v) for k, v in S.items()}, "probe minus": {}, "per stratum": {}}
        for k in ("minus answer length", "stratum dev accuracy"):
            r = auroc_diff_boot(y, S["probe"], S[k], clusters=cl, n_boot=a.n_boot, seed=SEED)
            out["H2a comparators"]["probe minus"][k] = [r["diff"], r["lo"], r["hi"]]
        hs = np.array([strat[i] for i in h2])
        for s in ("gsm8k", "math500", "math_ext"):
            m = hs == s
            if m.any() and y[m].any() and not y[m].all():
                r = auroc_diff_boot(y[m], S["probe"][m], S["verbal s"][m], clusters=[c for c, k in zip(cl, m) if k],
                                    n_boot=a.n_boot, seed=SEED)
                out["H2a comparators"]["per stratum"][s] = {"n": int(m.sum()), "auroc probe": auroc(y[m], S["probe"][m]),
                                                           "auroc verbal": auroc(y[m], S["verbal s"][m]),
                                                           "probe - verbal": [r["diff"], r["lo"], r["hi"]]}
        lvs = np.array([level.get(i, 0) for i in h2])
        out["H2a comparators"]["per MATH level"] = {}
        for lv in sorted(set(lvs) - {0}):
            m = lvs == lv
            if y[m].any() and not y[m].all():
                r = auroc_diff_boot(y[m], S["probe"][m], S["verbal s"][m], clusters=[c for c, k in zip(cl, m) if k],
                                    n_boot=a.n_boot, seed=SEED)
                out["H2a comparators"]["per MATH level"][str(lv)] = {
                    "n": int(m.sum()), "a0 acc": float(y[m].mean()), "auroc probe": auroc(y[m], S["probe"][m]),
                    "auroc verbal": auroc(y[m], S["verbal s"][m]), "probe - verbal": [r["diff"], r["lo"], r["hi"]]}

    (ROOT / "analysis" / "out" / f"{a.out}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    pc = lambda v: f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"
    dp = lambda v: f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"
    L = [f"# Exploratory analyses at P6, {spec['model']} {spec['split']} (generated by analysis/p6_exploratory.py)", "",
         f"Spec `{a.spec}`. **Exploratory** (prereg §3.3; DEVIATIONS D13): estimates and 95 % cluster-bootstrap CIs "
         f"({a.n_boot} resamples), no tests. Rates in %, differences in pp.", "",
         f"## 1A on D-nat, pooled (n {out['1A']['n']}, {out['1A']['n_wrong']} initially wrong)", "",
         "| Condition | Acc | W->R | R->W |", "|---|---|---|---|"]
    L += [f"| {c} | {pc(v['acc'])} | {pc(v['W->R'])} | {pc(v['R->W'])} |" for c, v in out["1A"]["cells"].items()]
    L += ["", "| Contrast | Estimate [95 % CI] |", "|---|---|"]
    L += [f"| {k} | {dp(v)} |" for k, v in out["1A"]["contrasts"].items()]
    L += ["", "## 1B role × label (W->R)", ""]
    for k, v in out["1B"].items():
        L += [f"**{k}** (n {v['n']}): " + "; ".join(f"{c} {pc(x)}" for c, x in v["cells"].items()), "",
              "| Contrast | Estimate [95 % CI] |", "|---|---|"]
        L += [f"| {c} | {dp(x)} |" for c, x in v.items() if c not in ("n", "cells")]
        L.append("")
    if "alt_labels" in out:
        v = out["alt_labels"]
        L += [f"## Alternative label wording (D-inj subset, F0, n {v['n']})", "", "| Quantity | Estimate [95 % CI] |",
              "|---|---|"] + [f"| {k} | {dp(x)} |" for k, x in v.items() if k != "n"] + [""]
    v = out["tsui"]
    L += [f"## Tsui cell (D-inj, n {v['n']}; W->R)", "", "| Quantity | Estimate [95 % CI] |", "|---|---|"]
    L += [f"| {c} | {pc(x)} |" for c, x in v["cells"].items()]
    L += [f"| {k} | {dp(x)} |" for k, x in v.items() if k not in ("n", "cells")]
    v = out["exp0"]
    L += ["", f"## Exp 0, hidden-state calibration ({v['n']} items, {v['templates']} templates)", "",
          "| Condition | Valid | Acc | Gain ratio (acc − 0.5)/(p − 0.5) |", "|---|---|---|---|"]
    L += [f"| {c} | {pc(x['valid'])} | {pc(x['acc'])} | "
          + (f"{x['gain ratio'][0]:.2f} [{x['gain ratio'][1]:.2f}, {x['gain ratio'][2]:.2f}]" if "gain ratio" in x else "–")
          + " |" for c, x in v["cells"].items()]
    if "H2a comparators" in out:
        v = out["H2a comparators"]
        L += ["", f"## H2a comparators (D-nat pooled, n {v['n']})", "", "| Predictor | AUROC |", "|---|---|"]
        L += [f"| {k} | {x:.3f} |" for k, x in v["auroc"].items()]
        L += ["", "| Probe minus | AUROC difference × 100 [95 % CI] |", "|---|---|"]
        L += [f"| {k} | {dp(x)} |" for k, x in v["probe minus"].items()]
        L += ["", "| Stratum | n | AUROC probe | AUROC verbal | Probe − verbal × 100 [95 % CI] |", "|---|---|---|---|---|"]
        L += [f"| {s} | {x['n']} | {x['auroc probe']:.3f} | {x['auroc verbal']:.3f} | {dp(x['probe - verbal'])} |"
              for s, x in v["per stratum"].items()]
        L += ["", "| MATH level | n | a0 acc | AUROC probe | AUROC verbal | Probe − verbal × 100 [95 % CI] |",
              "|---|---|---|---|---|---|"]
        L += [f"| {lv} | {x['n']} | {100 * x['a0 acc']:.1f} % | {x['auroc probe']:.3f} | {x['auroc verbal']:.3f} | "
              f"{dp(x['probe - verbal'])} |" for lv, x in v["per MATH level"].items()]
    (ROOT / "reports" / f"{a.out}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
