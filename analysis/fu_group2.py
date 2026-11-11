"""Group 2 follow-ups (docs/planning/next_experiments/plan.md; DEVIATIONS D16). Exploratory.

1. Stream controls (src/ttt/streams.py nll, matched) next to the four frozen-run conditions (4B-streams): per condition
   and checkpoint, greedy accuracy and the truncated share (mean over the 3 order seeds, [min, max]); per trajectory,
   the share of "incorrect" verdicts used for the loss, the unlikelihood share of applied updates, and the collapse
   onset (the first step of 10 consecutive truncated a0, as analysis/p4_streams.py). Reproduction: the base evaluation
   (k = 0) of the controls run must equal the frozen run's. The trajectory is the unit; no CI, no test.
2. Structured steering controls (src/steer/structured.py): per arm, W->R on D-inj test items, R->W on the 300 D-nat
   items, answer-change, unparsed and truncated rates; main minus each control on W->R and R->W (paired by item,
   95 % cluster-bootstrap CIs, clusters = canonical problem, 10,000 resamples); the pre-check KL per direction; the
   reproduction of main and unsteered against the frozen Exp 3 run; and, for reference, the frozen run's main minus
   the mean of its 5 random arms on the same items.
Input: configs/fu3_spec_4B_test.json {streams_ref, streams_controls, steer, steer_ref}.
-> reports/fu_03_group2.md, analysis/out/fu_03_group2.json
Usage: python analysis/fu_group2.py [--spec configs/fu3_spec_4B_test.json]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from p6_confirmatory import N_BOOT, SEED, Boot, jl, wmean  # noqa: E402

RUN = 10
TRAINED = ("self", "ext", "nll", "matched")


def streams(ref_dir, new_dir):
    out = {"eval": {}, "trajectories": {}}
    base = {}
    for d in (ref_dir, new_dir):
        d = ROOT / d
        if not (d / "eval_summary.json").exists():
            continue
        summ = json.loads((d / "eval_summary.json").read_text(encoding="utf-8"))
        conds = json.loads((d / "config.json").read_text(encoding="utf-8"))["conditions"]
        base[str(d)] = summ["0"]["base"]
        trajs = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["trajectories"]
        for c in conds:
            ts = [t for t in trajs if t.split("-")[0] == c]
            out["eval"][c] = {k: {"acc_greedy": [(summ[k]["base"] if k == "0" else summ[k][t])["acc_greedy"] for t in ts],
                                  "truncated": [(lambda s: s["greedy_status"].get("truncated", 0) / s["n"])(
                                      summ[k]["base"] if k == "0" else summ[k][t]) for t in ts]}
                              for k in sorted(summ, key=int)}
        st = defaultdict(list)
        for r in jl(d / "steps.jsonl"):
            st[r["traj"]].append(r)
        for t, rs in st.items():
            rs.sort(key=lambda r: r["step"])
            tr = [r["a0_status"] == "truncated" for r in rs]
            onset = next((rs[i]["step"] for i in range(len(rs) - RUN + 1) if all(tr[i:i + RUN])), None)
            app = [r for r in rs if (r.get("update") or {}).get("applied")]
            cond = t.split("-")[0]
            out["trajectories"][t] = {
                "condition": cond, "steps": len(rs), "a0_acc": sum(r["a0_correct"] for r in rs) / len(rs),
                "incorrect verdicts used": (sum(r["verdict"] is False for r in rs) / len(rs)) if cond in TRAINED else None,
                "updates applied": len(app),
                "UL share of applied": (sum(r["verdict"] is False for r in app) / len(app)) if app and cond in TRAINED else None,
                "collapse onset": onset}
    vals = list(base.values())
    out["base eval identical across runs"] = len(vals) == 2 and vals[0] == vals[1]
    return out


def steer(spec):
    d = ROOT / spec["steer"]
    if not (d / "rows.jsonl").exists():
        return None
    rows = jl(d / "rows.jsonl")
    met = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    canon = {r["item_id"]: r["canonical_id"] for f in ("data/processed/dinj.jsonl", "data/processed/dnat.jsonl") for r in jl(ROOT / f)}
    by = defaultdict(dict)
    for r in rows:
        by[(r["data"], r["item_id"])][r["run"]] = r
    arms = met["arms"]
    keys = sorted(k for k, v in by.items() if all(a in v for a in arms))
    data = np.array([k[0] for k in keys])
    boot = Boot([canon[k[1]] for k in keys], N_BOOT, SEED)
    col = lambda a, f: np.array([f(by[k][a]) for k in keys], float)  # noqa: E731
    cor = {a: col(a, lambda r: bool(r["correct"])) for a in arms}
    dinj, dnat = data == "dinj", data == "dnat"
    ci = lambda v, m: list(boot.ci(lambda w: wmean(v, w), m)) if m.any() else None  # noqa: E731
    out = {"n dinj": int(dinj.sum()), "n dnat": int(dnat.sum()), "precheck": met["precheck"],
           "reproduction": met.get("reproduction_vs_frozen_exp3"), "arms": {}, "main minus": {}}
    for a in arms:
        out["arms"][a] = {"W->R": ci(cor[a], dinj), "R->W": ci(1 - cor[a], dnat),
                          "changed": float(col(a, lambda r: bool(r["changed"])).mean()),
                          "unparsed": float(col(a, lambda r: r["status"] == "unparsed").mean()),
                          "truncated": float(col(a, lambda r: r["finish_reason"] == "length").mean())}
    for a in arms:
        if a != "main":
            out["main minus"][a] = {"W->R": ci(cor["main"] - cor[a], dinj), "R->W": ci((1 - cor["main"]) - (1 - cor[a]), dnat)}
    ref = spec.get("steer_ref")
    if ref and (ROOT / ref).exists():
        rb = defaultdict(dict)
        for r in jl(ROOT / ref):
            rb[(r["data"], r["item_id"])][r["run"]] = bool(r["correct"])
        rk = [k for k in keys if k in rb and "main" in rb[k] and all(f"random{j}" in rb[k] for j in range(5))]
        if not rk:
            return out
        rd = np.array([rb[k]["main"] - np.mean([rb[k][f"random{j}"] for j in range(5)]) for k in rk], float)
        m = np.array([k[0] == "dinj" for k in rk])
        b2 = Boot([canon[k[1]] for k in rk], N_BOOT, SEED)
        out["frozen run: main minus mean(random), W->R"] = list(b2.ci(lambda w: wmean(rd, w), m)) if m.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/fu3_spec_4B_test.json")
    ap.add_argument("--out", default="fu_03_group2")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    spec = json.loads((ROOT / a.spec).read_text(encoding="utf-8"))
    out = {"spec": a.spec, "streams": streams(spec["streams_ref"], spec["streams_controls"]) if spec.get("streams_controls") else None,
           "steer": steer(spec) if spec.get("steer") else None}
    (ROOT / "analysis" / "out" / f"{a.out}.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    pc = lambda v: "—" if v is None else f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"  # noqa: E731
    dp = lambda v: "—" if v is None else f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"  # noqa: E731
    rg = lambda v: f"{100 * np.mean(v):.1f} [{100 * min(v):.1f}, {100 * max(v):.1f}]" if v else "—"  # noqa: E731
    L = ["# Group 2 follow-ups, 4B (generated by analysis/fu_group2.py)", "",
         f"Spec `{a.spec}`. **Exploratory** (DEVIATIONS D16).", ""]
    s = out["streams"]
    if s:
        ks = sorted(set().union(*[set(v) for v in s["eval"].values()]), key=int)
        L += ["## 1. Stream controls (trajectory = unit; mean over 3 order seeds [min, max]; no CI)", "",
              f"Base evaluation identical across the frozen run and the controls run: {s['base eval identical across runs']}.", "",
              "| Condition | " + " | ".join(f"acc k={k}" for k in ks) + " | truncated at last k |", "|---|" + "---|" * (len(ks) + 1)]
        L += [f"| {c} | " + " | ".join(rg(v[k]["acc_greedy"]) if k in v else "—" for k in ks)
              + f" | {rg(v[ks[-1]]['truncated']) if ks[-1] in v else '—'} |"
              for c, v in s["eval"].items()]
        L += ["", "| Trajectory | a0 acc | Incorrect verdicts used | Updates applied | UL share of applied | Collapse onset |",
              "|---|---|---|---|---|---|"]
        f = lambda x: "—" if x is None else f"{100 * x:.1f} %"  # noqa: E731
        L += [f"| {t} | {f(v['a0_acc'])} | {f(v['incorrect verdicts used'])} | {v['updates applied']} | "
              f"{f(v['UL share of applied'])} | {v['collapse onset'] if v['collapse onset'] is not None else 'none'} |"
              for t, v in sorted(s["trajectories"].items())]
        L.append("")
    t = out["steer"]
    if t:
        L += [f"## 2. Structured steering controls (D-inj n {t['n dinj']}, D-nat R->W n {t['n dnat']})", "",
              f"Reproduction of main and unsteered vs the frozen Exp 3 run: {t['reproduction']}.", "",
              "| Arm | W->R | R->W | Changed | Unparsed | Truncated | Pre-check KL | cos with role |", "|---|---|---|---|---|---|---|---|"]
        for arm, v in t["arms"].items():
            kl = t["precheck"]["mean_kl"].get(arm)
            cs = t["precheck"]["cos_with_role"].get(arm)
            L.append(f"| {arm} | {pc(v['W->R'])} | {pc(v['R->W'])} | {100 * v['changed']:.1f} | {100 * v['unparsed']:.1f} | "
                     f"{100 * v['truncated']:.1f} | {'—' if kl is None else f'{kl:.4f}'} | {'—' if cs is None else f'{cs:+.2f}'} |")
        L += ["", "| Main minus | on W->R (pp) | on R->W (pp) |", "|---|---|---|"]
        L += [f"| {k} | {dp(v['W->R'])} | {dp(v['R->W'])} |" for k, v in t["main minus"].items()]
        if "frozen run: main minus mean(random), W->R" in t:
            L += ["", f"Frozen run, same items: main − mean of 5 random arms on W->R: {dp(t['frozen run: main minus mean(random), W->R'])} pp."]
        L.append("")
    (ROOT / "reports" / f"{a.out}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
