"""Robustness checks asked for by the skeptical review of the journal manuscript (post hoc, exploratory; DEVIATIONS
D19; reports/07_paper_review.md, third round). Stored outputs only; no new generation.

1. Initial answers without an answer. A truncated or unparsed a0 is scored wrong; any revision that ends with an answer
   then counts as a repair (W->R), which can be format repair rather than content repair. Per setting and condition:
   repairs whose a0 had no answer; and the main quantities recomputed on the items whose a0 gave an answer
   ("a0 answered": math status ok; MMLU-Pro status ok or fallback):
   - math: net change of every condition (CI), and F2 re-checks at 10 % / 20 % for the main rankings, re-ranked
     within the subset (vs KEEP and vs ALL, CIs);
   - MMLU-Pro: the addendum's P1-P3 via the frozen mc_results.analyse() on the subset (dev subset alike).
2. Pools. For F2 re-checks at 10 % and 20 %: repair rate among the pool's initially wrong items and damage rate among
   its initially right items, with 95 % cluster-bootstrap CIs (ratio of weighted sums, same resamples as the source
   tables), for the main rankings and the gold wrong-first reference; plus the pool's share of unanswered a0.
3. Oracle on MMLU-Pro. The gold wrong-first reference (re-check initially wrong items first, in tie-break order; uses
   gold; not a policy and not an upper bound) at 10 % / 20 %: vs KEEP and vs ALL.
4. Cost on MMLU-Pro: prompt tokens processed per item for each condition (all turns, no cache reuse), the samples'
   generated tokens alone, and their prompt tokens (5 x the a0 prompt).
Usage: python analysis/fu_ms_robustness.py [--n-boot 10000]
-> reports/fu_11_ms_robustness.md, analysis/out/fu_11_ms_robustness.json
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
import fu_recheck_vs_all as VA  # noqa: E402
import mc_results as MR  # noqa: E402
from p6_confirmatory import N_BOOT, SEED, Boot, jl, wmean  # noqa: E402

SHARES = (0.1, 0.2)
CONDS_MATH = ["F0", "F1", "F2", "F2-scratch", "F3(1.0)"]


def select(scores, tie, k):
    n = len(scores)
    m = np.zeros(n, bool)
    m[np.lexsort((tie, scores))[:int(round(k * n))]] = True
    return m


def ratio_ci(boot, num, den):
    """Ratio of weighted sums with its bootstrap CI; None when the denominator is empty (e.g. no right item in a pool)."""
    if den.sum() == 0:
        return None
    return list(boot.ci(lambda w: float((w * num).sum() / (w * den).sum()) if (w * den).sum() else float("nan")))


def pools(B, rankings, noans):
    y0, after, tie, boot = B["y0"], B["after"]["F2"], B["tie"], B["boot"]
    wrong, right = y0 == 0, y0 == 1
    out = {}
    for k in SHARES:
        row = {}
        for name in rankings:
            m = select(B["scores"]["F2"][name], tie, k)
            fix = (m & wrong & (after == 1)).astype(float)
            dam = (m & right & (after == 0)).astype(float)
            row[name] = {"repair rate in pool": ratio_ci(boot, fix, (m & wrong).astype(float)),
                         "damage rate in pool": ratio_ci(boot, dam, (m & right).astype(float)),
                         "share of pool with unanswered a0": float(noans[m].mean())}
        out[f"{int(round(100 * k))}"] = row
    out["overall"] = {"repair rate": ratio_ci(boot, (wrong & (after == 1)).astype(float), wrong.astype(float)),
                      "damage rate": ratio_ci(boot, (right & (after == 0)).astype(float), right.astype(float))}
    return out


def sub_mc(D, keep):
    ids = [i for i, k in zip(D["ids"], keep) if k]
    E = {"ids": ids, "y0": D["y0"][keep], "cat": [c for c, k in zip(D["cat"], keep) if k], "len": D["len"][keep],
         "agree": D["agree"][keep], "samp_gen": D["samp_gen"][keep], "verbal": {v: s[keep] for v, s in D["verbal"].items()},
         "task probe": D["task probe"][keep], "math probe": D["math probe"][keep],
         "after": {c: v[keep] for c, v in D["after"].items()}, "gen": {c: v[keep] for c, v in D["gen"].items()}, "dirs": D["dirs"]}
    return E


def mc_status(model, split):
    e = MR.exp_of(split)
    d = MR.latest(f"{e}/{model}/{e}-mc-initial-{split}-{model}-*")
    return {r["item_id"]: r["status"] for r in jl(d / "initial.jsonl")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    J = {}
    # ---------------- math
    B = VA.build_math(a.n_boot)
    st = {r["item_id"]: r["status"] for f in RC.INITIAL for r in jl(f) if r["split"] == "test"}
    noans = np.array([st[i] != "ok" for i in B["ids"]])
    T, y0 = B["T"], B["y0"]
    wrong = y0 == 0
    jm = {"n": len(y0), "unanswered a0": int(noans.sum()), "unanswered a0 among wrong": int((noans & wrong).sum()),
          "repairs": {}, "a0 answered": {}}
    for c in CONDS_MATH:
        aft = T["after"][c]
        rep = wrong & (aft == 1)
        jm["repairs"][c] = {"all": int(rep.sum()), "from unanswered a0": int((rep & noans).sum())}
    keep = ~noans
    bs = Boot([B["canon"][i] for i, k in zip(B["ids"], keep) if k], a.n_boot, SEED + 31)
    ci = lambda v: list(bs.ci(lambda w: wmean(v, w)))  # noqa: E731
    y0k = y0[keep]
    jm["a0 answered"]["n"] = int(keep.sum())
    jm["a0 answered"]["net change"] = {c: ci(T["after"][c][keep] - y0k) for c in CONDS_MATH}
    aft = B["after"]["F2"][keep]
    tie = B["tie"][keep]
    pol = {}
    for k in SHARES:
        row = {}
        for name in ("verbal s", "probe", "answer length", "agreement with a0"):
            m = select(B["scores"]["F2"][name][keep], tie, k)
            f = np.where(m, aft, y0k)
            row[name] = {"vs KEEP": ci(f - y0k), "vs ALL": ci(f - aft)}
        mr = select(B["scores"]["F2"]["probe"][keep], tie, k)
        mv = select(B["scores"]["F2"]["verbal s"][keep], tie, k)
        row["probe - verbal"] = ci(np.where(mr, aft, y0k) - np.where(mv, aft, y0k))
        pol[f"{int(round(100 * k))}"] = row
    jm["a0 answered"]["F2 re-check"] = pol
    jm["pools"] = pools(B, ["verbal s", "probe", "answer length", "agreement with a0", "gold wrong-first (uses gold)"], noans)
    J["math"] = jm
    # ---------------- MMLU-Pro
    canon = {r["item_id"]: r["canonical_id"] for r in jl(ROOT / "data" / "processed" / "mmlupro" / "items.jsonl")}
    for model in ("4B", "T8B"):
        B = VA.build_mc(model, a.n_boot)
        T, y0 = B["T"], B["y0"]
        st = mc_status(model, "test")
        noans = np.array([st[i] in ("truncated", "unparsed") for i in B["ids"]])
        wrong, right = y0 == 0, y0 == 1
        jo = {"n": len(y0), "unanswered a0": int(noans.sum()), "repairs": {}}
        for c in MR.CONDS:
            rep = wrong & (T["after"][c] == 1)
            jo["repairs"][c] = {"all": int(rep.sum()), "from unanswered a0": int((rep & noans).sum())}
        # P1-P3 on the answered subset, by the frozen analysis
        keep = ~noans
        Dv = MR.load(model, "dev")
        sd = mc_status(model, "dev")
        kd = np.array([sd[i] not in ("truncated", "unparsed") for i in Dv["ids"]])
        o = MR.analyse(sub_mc(T, keep), sub_mc(Dv, kd), canon, a.n_boot)
        p = o["policies"]["F2"]["budgets"]
        jo["a0 answered"] = {"n": int(keep.sum()), "P1": {c: o["conditions"][c]["net change"] for c in ("F0", "F1", "F2", "F2-scratch")},
                             "P2 AUROC task probe - verbal W0": o["task probe - verbal W0"],
                             "P2 order task probe - verbal W0": {b: p[b]["task probe"]["minus verbal W0"] for b in ("10", "20")},
                             "P3 order agreement - verbal W0 (10 %)": p["10"]["agreement with a0"]["minus verbal W0"]}
        # oracle (gold wrong-first) on the full population
        B["scores"]["F2"]["gold wrong-first (uses gold)"] = y0
        aftF2 = B["after"]["F2"]
        cif = lambda v: list(B["boot"].ci(lambda w: wmean(v, w)))  # noqa: E731
        orc = {}
        for k in SHARES:
            m = select(y0, B["tie"], k)
            f = np.where(m, aftF2, y0)
            orc[f"{int(round(100 * k))}"] = {"vs KEEP": cif(f - y0), "vs ALL": cif(f - aftF2)}
        jo["gold wrong-first"] = orc
        jo["pools"] = pools(B, ["verbal W0", "task probe", "math probe", "answer length", "agreement with a0",
                                "gold wrong-first (uses gold)"], noans)
        # cost: prompt tokens processed per item, all turns, no cache reuse
        dc = MR.latest(f"fu/{model}/fu-mc-conds-test-{model}-*")
        rows = {}
        for r in jl(dc / "rows.jsonl"):
            if r["cond"] != "B0":
                rows.setdefault(r["cond"], {})[r["item_id"]] = r["prompt_tokens"] + (r.get("turn1_prompt_tokens") or 0)
        di = MR.latest(f"fu/{model}/fu-mc-initial-test-{model}-*")
        a0p = {r["item_id"]: r["prompt_tokens"] for r in jl(di / "initial.jsonl")}
        jo["cost"] = {"prompt tokens / item": {c: float(np.mean([rows[c][i] for i in B["ids"]])) for c in MR.CONDS},
                      "a0 prompt tokens / item": float(np.mean([a0p[i] for i in B["ids"]])),
                      "samples: generated / item": float(T["samp_gen"].mean()),
                      "samples: prompt tokens / item (5 x a0 prompt)": float(5 * np.mean([a0p[i] for i in B["ids"]])),
                      "F2: generated / item": float(T["gen"]["F2"].mean())}
        J[f"mmlupro {model}"] = jo
    (ROOT / "analysis" / "out" / "fu_11_ms_robustness.json").write_text(json.dumps(J, indent=1), encoding="utf-8", newline="\n")

    pc = lambda v: f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"  # noqa: E731
    pr = lambda v: "— (no such item in the pool)" if v is None else f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"  # noqa: E731
    L = ["# Robustness checks for the journal manuscript (generated by analysis/fu_ms_robustness.py)", "",
         "**Post hoc, exploratory** (DEVIATIONS D19), asked for by the skeptical review (reports/07_paper_review.md, third round). "
         "Stored outputs only. 95 % cluster-bootstrap CIs. \"Unanswered a0\": math status not ok (truncated); MMLU-Pro status "
         "truncated or unparsed (fallback letters count as answers).", ""]
    for setting, o in J.items():
        L += [f"## {setting} (n {o['n']}; unanswered a0: {o['unanswered a0']})", "", "### Repairs whose a0 had no answer", "",
              "| Condition | Repairs (W->R) | of which from unanswered a0 |", "|---|---|---|"]
        L += [f"| {c} | {v['all']} | {v['from unanswered a0']} |" for c, v in o["repairs"].items()]
        aa = o["a0 answered"]
        L += ["", f"### On the items whose a0 gave an answer (n {aa['n']})", ""]
        if setting == "math":
            L += ["| Condition | Net change (pp) |", "|---|---|"] + [f"| {c} | {pc(v)} |" for c, v in aa["net change"].items()]
            for b, row in aa["F2 re-check"].items():
                L += ["", f"F2 re-check of {b} % (re-ranked within the subset):", "", "| Ranking | vs KEEP | vs ALL |", "|---|---|---|"]
                L += [f"| {nm} | {pc(v['vs KEEP'])} | {pc(v['vs ALL'])} |" for nm, v in row.items() if nm != "probe - verbal"]
                L.append(f"\nProbe order − verbal order: {pc(row['probe - verbal'])} pp.")
        else:
            L += ["| Quantity | Value |", "|---|---|"]
            L += [f"| P1 {c} net change | {pc(v)} |" for c, v in aa["P1"].items()]
            L += [f"| P2 AUROC task probe − verbal W0 | {pc(aa['P2 AUROC task probe - verbal W0'])} |"]
            L += [f"| P2 order task probe − verbal W0, {b} % | {pc(v)} |" for b, v in aa["P2 order task probe - verbal W0"].items()]
            L += [f"| P3 order agreement − verbal W0, 10 % | {pc(aa['P3 order agreement - verbal W0 (10 %)'])} |"]
            L += ["", "### Gold wrong-first reference (uses gold; not a policy, not an upper bound)", "",
                  "| Share | vs KEEP | vs ALL |", "|---|---|---|"]
            L += [f"| {b} % | {pc(v['vs KEEP'])} | {pc(v['vs ALL'])} |" for b, v in o["gold wrong-first"].items()]
            c = o["cost"]
            L += ["", "### Cost per item (MMLU-Pro)", "", "| Condition | Prompt tokens processed (all turns, no cache) |", "|---|---|"]
            L += [f"| {k} | {v:.0f} |" for k, v in c["prompt tokens / item"].items()]
            L += ["", f"a0 prompt {c['a0 prompt tokens / item']:.0f}; five samples: generated {c['samples: generated / item']:.0f}, "
                  f"prompt {c['samples: prompt tokens / item (5 x a0 prompt)']:.0f}; F2 generated {c['F2: generated / item']:.0f}.", ""]
        po = o["pools"]
        L += ["", "### Repair and damage rates within F2 pools", "",
              f"Overall: repair {pr(po['overall']['repair rate'])} %, damage {pr(po['overall']['damage rate'])} %.", "",
              "| Share | Ranking | Repair rate in pool (%) | Damage rate in pool (%) | Pool share with unanswered a0 |", "|---|---|---|---|---|"]
        for b in ("10", "20"):
            for nm, v in po[b].items():
                L.append(f"| {b} % | {nm} | {pr(v['repair rate in pool'])} | {pr(v['damage rate in pool'])} | "
                         f"{100 * v['share of pool with unanswered a0']:.0f} % |")
        L.append("")
    (ROOT / "reports" / "fu_11_ms_robustness.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
