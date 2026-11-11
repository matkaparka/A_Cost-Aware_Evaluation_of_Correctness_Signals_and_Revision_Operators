"""Lore QA results (docs/planning/lore_qa/prereg_addendum.md). Exploratory, estimation only: estimates with 95 %
cluster-bootstrap CIs (10,000 resamples, clusters = the item's `cluster`, one shared resample set per table), no tests.

Pools and their status, fixed by the dev gates (DECISIONS 2026-11-07):
- D-inj-lore, name-class rows: the primary pool. Per role x label cell at F0: W->R on injected-wrong rows, R->W on
  injected-right rows, net accuracy; role effect (user - assistant, averaged over labels), label effect (other - self,
  averaged over roles), interaction (u/o - u/s) - (a/o - a/s), diagonal u/o - a/s; paired within row.
- D-inj-lore phrase-class rows, and D-nat-lore (G2 and G4 failed; phrase items failed G3b): descriptive. D-nat-lore:
  per condition n, accuracy, W->R (k/n), R->W (k/n), truncated and not-ok rates; paired accuracy contrasts against F0
  (and B0), F5 against the mean of its three permutations.
- Exp 2 on D-nat-lore test at the chosen budget: AUROC for initial correctness of the verbal score s, the frozen
  math probe (zero-shot transfer), the lore-dev probe and minus answer length; probe - verbal differences (paired
  cluster bootstrap, analysis/stats.py auroc_diff_boot).
- The vLLM dev gates (G2 per budget, G3, G4) are recomputed from the dev initial run with the addendum's rules.
Input: a spec (configs/lore_spec_<model>_test.json) naming the synced run directories.
-> reports/lore_02_results.md, analysis/out/lore_02_results.json
Usage: python analysis/lore_results.py --spec configs/lore_spec_4B_test.json
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from p6_confirmatory import Boot, wmean  # noqa: E402
from stats import auroc, auroc_diff_boot  # noqa: E402
from src.data import lore, lore_b  # noqa: E402

N_BOOT, SEED = 10000, 0
G2_LO, G2_HI, G3_MAX, G4_MIN = 0.35, 0.85, 0.05, 150
CELLS = ["assistant/self/F0", "assistant/other/F0", "user/self/F0", "user/other/F0"]
COND_1A = ["B0", "B1", "F0", "F1", "F2", "F2-scratch", "F3(0.5)", "F3(0.7)", "F3(0.9)", "F3(1.0)", "F5",
           "F3-perm[0]", "F3-perm[1]", "F3-perm[2]"]


def ci(boot, f, mask):
    """boot.ci, or None when the subset is empty or every resample is undefined."""
    if mask is None or not np.asarray(mask).any():
        return None
    try:
        return boot.ci(f, mask)
    except IndexError:
        return None


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def items():
    out = {it["id"]: dict(it, route="A") for it in lore.load_built()[1]}
    out.update({it["id"]: it for it in lore_b.load_items_b()})
    return out


def verbal_s(r):
    if r.get("p_yes") is None or r.get("p_no") is None:
        return None
    return math.log(max(r["p_yes"], 1e-300)) - math.log(max(r["p_no"], 1e-300))


def gates(dev_rows, n_test):
    acc = {}
    for b in sorted({r["budget"] for r in dev_rows}):
        rb = [r for r in dev_rows if r["budget"] == b]
        acc[b] = sum(r["correct"] for r in rb) / len(rb)
    if not acc:
        return None
    ok = [b for b in sorted(acc) if acc[b] <= G2_HI]
    chosen = ok[0] if ok else min(acc, key=lambda b: (acc[b], b))
    rb = [r for r in dev_rows if r["budget"] == chosen]
    bad = sum(r["status"] in ("unparsed", "truncated", "conflict") for r in rb) / len(rb)
    proj = (1 - acc[chosen]) * n_test
    return {"accuracy by budget": acc, "chosen budget": chosen,
            "G2": ("pass" if G2_LO <= acc[chosen] <= G2_HI else "fail (too hard)") if ok else "fail (above 0.85 at every budget)",
            "G3": {"rate": bad, "verdict": "pass" if bad <= G3_MAX else "fail"},
            "G4": {"projected wrong test items": proj, "verdict": "pass" if proj >= G4_MIN else "fail"}}


def dinj_table(rows, it):
    """Wide D-inj-lore table: one row per injected row, correctness per cell."""
    by = defaultdict(dict)
    meta = {}
    for r in rows:
        if r["data"] == "dinj" and r["cond"] in CELLS:
            by[r["inj_id"]][r["cond"]] = bool(r["correct"])
            meta[r["inj_id"]] = (r["item_id"], r["variant"], r["route"], it[r["item_id"]]["answer_class"])
    ids = sorted(k for k in by if len(by[k]) == 4)
    C = {c: np.array([by[k][c] for k in ids], float) for c in CELLS}
    return ids, C, {k: meta[k] for k in ids}


def dinj_estimates(ids, C, meta, it, mask_extra):
    wrong = np.array([meta[k][1] == "wrong" for k in ids])
    clusters = [it[meta[k][0]]["cluster"] for k in ids]
    boot = Boot(clusters, N_BOOT, SEED)
    a_s, a_o, u_s, u_o = (C[c] for c in CELLS)
    out = {"n wrong rows": int((wrong & mask_extra).sum()), "n right rows": int((~wrong & mask_extra).sum()), "cells": {}}
    for c in CELLS:
        out["cells"][c] = {"W->R": ci(boot, lambda m, v=C[c]: wmean(v, m), wrong & mask_extra),
                           "R->W": ci(boot, lambda m, v=1 - C[c]: wmean(v, m), ~wrong & mask_extra),
                           "accuracy": ci(boot, lambda m, v=C[c]: wmean(v, m), mask_extra)}
    contrasts = {"role (user - assistant)": (u_s + u_o) / 2 - (a_s + a_o) / 2,
                 "label (other - self)": (a_o + u_o) / 2 - (a_s + u_s) / 2,
                 "interaction": (u_o - u_s) - (a_o - a_s), "diagonal (u/o - a/s)": u_o - a_s}
    out["W->R contrasts"] = {k: ci(boot, lambda m, v=v: wmean(v, m), wrong & mask_extra) for k, v in contrasts.items()}
    out["R->W contrasts"] = {k: ci(boot, lambda m, v=-v: wmean(v, m), ~wrong & mask_extra) for k, v in contrasts.items()}
    return out


def dnat_estimates(rows, it, cls):
    by = defaultdict(dict)
    info = {}
    for r in rows:
        if r["data"] == "dnat" and (cls == "all" or r["answer_class"] == cls):
            by[r["item_id"]][r["cond"] if r["exp"] == "1A" else f"1B {r['cond']}"] = r
            info[r["item_id"]] = bool(r["correct0"])
    ids = sorted(by)
    if not ids:
        return None
    conds = [c for c in COND_1A if any(c in by[i] for i in ids)] + [f"1B {c}" for c in CELLS if any(f"1B {c}" in by[i] for i in ids)]
    c0 = np.array([info[i] for i in ids])
    boot = Boot([it[i]["cluster"] for i in ids], N_BOOT, SEED)
    out = {"n items": len(ids), "n initially wrong": int((~c0).sum()), "conditions": {}, "contrasts": {}}
    acc = {}
    for c in conds:
        have = np.array([c in by[i] for i in ids])
        v = np.array([bool(by[i][c]["correct"]) if c in by[i] else False for i in ids], float)
        acc[c] = (v, have)
        tr = np.array([by[i][c].get("finish_reason") == "length" if c in by[i] else False for i in ids])
        bad = np.array([by[i][c]["status"] not in ("ok", "fixed") if c in by[i] else False for i in ids])
        out["conditions"][c] = {"n": int(have.sum()), "accuracy": ci(boot, lambda m, v=v: wmean(v, m), have),
                                "W->R": [int((v * have * ~c0).sum()), int((have & ~c0).sum())],
                                "R->W": [int(((1 - v) * have * c0).sum()), int((have & c0).sum())],
                                "truncated": float(tr[have].mean()) if have.any() else None,
                                "not ok": float(bad[have].mean()) if have.any() else None}
    pairs = [("F0", "B0"), ("B1", "B0"), ("F1", "F0"), ("F2", "F0"), ("F2-scratch", "F0"), ("F3(1.0)", "F0"),
             ("F3(0.5)", "F0"), ("F5", "F0")]
    for a, b in pairs:
        if a in acc and b in acc:
            m = acc[a][1] & acc[b][1]
            out["contrasts"][f"{a} - {b}"] = ci(boot, lambda w, d=acc[a][0] - acc[b][0]: wmean(d, w), m)
    perms = [f"F3-perm[{k}]" for k in range(3)]
    if "F5" in acc and all(p in acc for p in perms):
        m = acc["F5"][1] & np.logical_and.reduce([acc[p][1] for p in perms])
        d = acc["F5"][0] - np.mean([acc[p][0] for p in perms], 0)
        out["contrasts"]["F5 - mean(F3-perm)"] = ci(boot, lambda w, d=d: wmean(d, w), m)
    return out


def exp2(init_rows, verbal, probes, it, cls):
    v = {r["item_id"]: verbal_s(r) for r in verbal}
    p = {r["item_id"]: r for r in probes}
    rows = [r for r in init_rows if (cls == "all" or r["answer_class"] == cls) and v.get(r["item_id"]) is not None
            and r["item_id"] in p and p[r["item_id"]]["p_lore_probe"] is not None]
    if not rows:
        return None
    y = np.array([bool(r["correct"]) for r in rows])
    if y.all() or not y.any():
        return {"n": len(rows), "n wrong": int((~y).sum()), "note": "one class only"}
    S = {"verbal s": np.array([v[r["item_id"]] for r in rows]),
         "math probe (zero-shot)": np.array([p[r["item_id"]]["p_math_probe"] for r in rows]),
         "lore-dev probe": np.array([p[r["item_id"]]["p_lore_probe"] for r in rows]),
         "minus answer length": -np.array([r["out_tokens"] for r in rows], float)}
    cl = [it[r["item_id"]]["cluster"] for r in rows]
    out = {"n": len(rows), "n wrong": int((~y).sum()), "auroc": {k: auroc(y, s) for k, s in S.items()}, "minus verbal": {}}
    for k in ("math probe (zero-shot)", "lore-dev probe", "minus answer length"):
        r = auroc_diff_boot(y, S[k], S["verbal s"], clusters=cl, n_boot=N_BOOT, seed=SEED)
        out["minus verbal"][k] = [r["diff"], r["lo"], r["hi"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default="lore_02_results")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    spec = json.loads((ROOT / a.spec).read_text(encoding="utf-8"))
    it = items()
    n_test = sum(1 for x in it.values() if x["split"] == "test")
    B = spec["budget"]
    dev_init = jl(ROOT / spec["initial_dev"] / "initial.jsonl")
    test_init = [r for r in jl(ROOT / spec["initial_test"] / "initial.jsonl") if r["budget"] == B and r["split"] == spec["split"]]
    verbal = [r for r in jl(ROOT / spec["hf"] / "verbal.jsonl") if r["split"] == spec["split"]]
    probes = [r for r in jl(ROOT / spec["hf"] / "probe_scores.jsonl") if r["split"] == spec["split"]]
    conds = jl(ROOT / spec["conds"] / "rows.jsonl")
    out = {"spec": a.spec, "budget": B, "gates (vLLM dev)": gates(dev_init, n_test)}
    out["initial"] = {}
    for k, sel in (("all", lambda r: True), ("route A", lambda r: r["route"] == "A"), ("route B", lambda r: r["route"] == "B"),
                   ("class name", lambda r: r["answer_class"] == "name"), ("class phrase", lambda r: r["answer_class"] == "phrase")):
        rs = [r for r in test_init if sel(r)]
        out["initial"][k] = {"n": len(rs), "right": sum(r["correct"] for r in rs),
                             "unparsed+truncated": sum(r["status"] in ("unparsed", "truncated", "conflict") for r in rs)}
    ids, C, meta = dinj_table(conds, it)
    if ids:
        cls = np.array([meta[k][3] for k in ids])
        route = np.array([meta[k][2] for k in ids])
        out["dinj"] = {"name (primary)": dinj_estimates(ids, C, meta, it, cls == "name"),
                       "phrase (descriptive)": dinj_estimates(ids, C, meta, it, cls == "phrase"),
                       "route A name (descriptive)": dinj_estimates(ids, C, meta, it, (cls == "name") & (route == "A")),
                       "route B name (descriptive)": dinj_estimates(ids, C, meta, it, (cls == "name") & (route == "B"))}
    out["dnat"] = {c: dnat_estimates(conds, it, c) for c in ("name", "all")}
    out["exp2"] = {c: exp2(test_init, verbal, probes, it, c) for c in ("name", "all")}
    (ROOT / "analysis" / "out" / f"{a.out}.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")

    pc = lambda v: "—" if v is None or v[0] is None or (isinstance(v[0], float) and math.isnan(v[0])) else \
        f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"  # noqa: E731
    dp = lambda v: "—" if v is None or (isinstance(v[0], float) and math.isnan(v[0])) else \
        f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"  # noqa: E731
    L = [f"# Lore QA results, {spec['model']} {spec['split']} (generated by analysis/lore_results.py)", "",
         f"Spec `{a.spec}`; budget {B}. **Exploratory, estimation only** (docs/planning/lore_qa/prereg_addendum.md): "
         f"estimates with 95 % cluster-bootstrap CIs ({N_BOOT} resamples; clusters = item cluster). Rates in %, "
         "differences in pp. Not pooled with the math results.", ""]
    g = out["gates (vLLM dev)"]
    if g:
        L += ["## Dev gates, vLLM (addendum rules)", "",
              "- G2: " + ", ".join(f"{b}: {100 * x:.1f} %" for b, x in g["accuracy by budget"].items())
              + f"; chosen budget {g['chosen budget']} → {g['G2']}.",
              f"- G3: {100 * g['G3']['rate']:.1f} % → {g['G3']['verdict']}. G4: {g['G4']['projected wrong test items']:.0f} "
              f"projected wrong test items → {g['G4']['verdict']}.", ""]
    L += ["## Initial answers (test, chosen budget)", "", "| Subset | n | Right | Unparsed + truncated |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['right']} | {v['unparsed+truncated']} |" for k, v in out["initial"].items()]
    L.append("")
    for name, d in out.get("dinj", {}).items():
        L += [f"## D-inj-lore, {name}", "", f"Injected-wrong rows {d['n wrong rows']}, injected-right rows {d['n right rows']}.", "",
              "| Cell (F0) | W->R | R->W | Accuracy |", "|---|---|---|---|"]
        L += [f"| {c} | {pc(v['W->R'])} | {pc(v['R->W'])} | {pc(v['accuracy'])} |" for c, v in d["cells"].items()]
        L += ["", "| Contrast | on W->R (pp) | on R->W (pp) |", "|---|---|---|"]
        L += [f"| {k} | {dp(d['W->R contrasts'][k])} | {dp(d['R->W contrasts'][k])} |" for k in d["W->R contrasts"]]
        L.append("")
    for cls, d in out["dnat"].items():
        if not d:
            continue
        L += [f"## D-nat-lore, {cls} items (descriptive)", "",
              f"Items {d['n items']}, initially wrong {d['n initially wrong']}.", "",
              "| Condition | n | Accuracy | W->R (k/n) | R->W (k/n) | Truncated | Not ok |", "|---|---|---|---|---|---|---|"]
        L += [f"| {c} | {v['n']} | {pc(v['accuracy'])} | {v['W->R'][0]}/{v['W->R'][1]} | {v['R->W'][0]}/{v['R->W'][1]} | "
              f"{100 * (v['truncated'] or 0):.1f} | {100 * (v['not ok'] or 0):.1f} |" for c, v in d["conditions"].items()]
        L += ["", "| Contrast (accuracy) | pp |", "|---|---|"] + [f"| {k} | {dp(v)} |" for k, v in d["contrasts"].items()]
        L.append("")
    for cls, d in out["exp2"].items():
        if not d:
            continue
        L += [f"## Exp 2, D-nat-lore test, {cls} items", "", f"n {d['n']}, initially wrong {d['n wrong']}."]
        if "auroc" in d:
            L += ["", "| Predictor | AUROC | Minus verbal (pp) |", "|---|---|---|"]
            L += [f"| {k} | {v:.3f} | {dp(d['minus verbal'][k]) if k in d['minus verbal'] else '—'} |" for k, v in d["auroc"].items()]
        L.append("")
    (ROOT / "reports" / f"{a.out}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
