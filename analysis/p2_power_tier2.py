"""Power for the Tier 2 tests H4a and H3b (design §7.4), from the dev pilots selected in analysis/out/tier2.json
(written by analysis/p2_tier2.py) -> reports/02_power_tier2_tables.md, analysis/out/power_tier2.json.

- H4a (Exp 5, parameter arm only): accuracy after the update from the model's own verdicts (self) vs after the
  update from the stratum-permuted self-verdicts (perm-self), two-sided exact McNemar. Paired outcomes per item at the
  selected (tau, lr). The tuning items over-represent initially wrong items, so the pilot patterns are taken per
  stratum (initially right / wrong) and mixed by the composition of the test pool: 'natural' (a random 400 of the
  test split, its right/wrong share from the P1 initial answers) or 'wrong first' (every test wrong item, then right
  items up to 400; the rule the pilot used to pick its tuning items). The pool composition is not fixed by the design;
  it goes into the prereg. Scenarios as in analysis/power.py: discordance x {0.5, 1, 1.5}; effect delta/2, delta, half
  the pilot estimate.
- H3b (Exp 3, main setting): d_i = W->R(steered) - mean over the 5 random directions of W->R, per D-inj item, one-sided
  paired test (normal approximation). The generic control enters (intersection-union) only if it was valid at the
  main setting on dev. The pilot distribution of d_i is rescaled in its non-zero mass and tilted to the target mean, as
  for H3a in analysis/p2_power_more.py. Test n: the D-inj test items, times the dev share of items with steerable
  content.
Level: alpha / 9 (the first Holm step when every Tier 2 hypothesis is enabled; the family has 6 to 9 members) and
alpha / 6. Usage: python analysis/p2_power_tier2.py
"""
import json
import math
from pathlib import Path

import numpy as np

from p2_power_more import tilt
from power import pattern_probs
from stats import mcnemar_exact, one_sided_paired

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "02_power_tier2_tables.md"
ALPHA, DELTA, SEED = 0.05, 0.03, 0
FAMILIES = (9, 6)
SIMS = 2000
N_EXP5 = 400
INITIAL = {"4B": "p1-initial-4B-d2ebdea", "1.7B": "p1-initial-1.7B-d2ebdea"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def h4a(model, sel, rng):
    rows = jl(ROOT / "results" / "p2" / model / sel["serve"] / "rows.jsonl")
    pick = lambda g: {r["item_id"]: (r["correct"], r["correct0"]) for r in rows if r["arm"] == "parameter"
                      and r["grounding"] == g and f"{r['tau']:g}" == sel["tau"] and f"{r['lr']:g}" == sel["lr"]}
    s, p = pick("self"), pick("perm-self")
    ids = sorted(set(s) & set(p))
    pats = {}
    for stratum in (True, False):   # correct0
        xy = np.array([(s[i][0], p[i][0]) for i in ids if s[i][1] == stratum], int).reshape(-1, 2)
        n = len(xy)
        p10 = ((xy[:, 0] == 1) & (xy[:, 1] == 0)).mean() if n else 0.0
        p01 = ((xy[:, 0] == 0) & (xy[:, 1] == 1)).mean() if n else 0.0
        p11 = ((xy[:, 0] == 1) & (xy[:, 1] == 1)).mean() if n else 0.0
        pats[stratum] = {"n": n, "p11": p11, "p10": p10, "p01": p01, "p00": 1 - p11 - p10 - p01}
    init = jl(ROOT / "results" / "p1" / model / INITIAL[model] / "initial.jsonl")
    n_right = sum(r["split"] == "test" and r["correct"] for r in init)
    n_wrong = sum(r["split"] == "test" and not r["correct"] for r in init)
    pools = {"natural": n_wrong / (n_right + n_wrong), "wrong first": min(1.0, n_wrong / N_EXP5)}
    out = {"pilot": pats, "test_right": n_right, "test_wrong": n_wrong, "pools": {}}
    for pool, w_wrong in pools.items():
        mix = {k: w_wrong * pats[False][k] + (1 - w_wrong) * pats[True][k] for k in ("p11", "p10", "p01", "p00")}
        disc, eff_pilot = mix["p10"] + mix["p01"], mix["p10"] - mix["p01"]
        share = mix["p11"] / max(1e-12, mix["p11"] + mix["p00"])
        rows_out = []
        for dm in (0.5, 1.0, 1.5):
            d = disc * dm
            for name, eff in (("δ/2", DELTA / 2), ("δ", DELTA), ("half pilot", abs(eff_pilot) / 2)):
                probs = pattern_probs(d, eff, share) if d > 0 else None
                if probs is None or eff <= 0:
                    rows_out.append({"disc": d, "scenario": name, "effect": eff, "power": None})
                    continue
                pv = np.empty(SIMS)
                pp = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
                for k in range(SIMS):
                    xy = pp[rng.choice(4, N_EXP5, p=probs)]
                    pv[k] = mcnemar_exact(xy[:, 0], xy[:, 1])
                rows_out.append({"disc": d, "scenario": name, "effect": eff,
                                 "power": {f: float((pv < ALPHA / f).mean()) for f in FAMILIES}})
        out["pools"][pool] = {"wrong_share": w_wrong, "pilot_disc": disc, "pilot_effect": eff_pilot, "rows": rows_out}
    return out


def h3b(sel, rng):
    run = ROOT / "results" / "p2" / "4B" / sel["run"]
    m = json.loads((run / "metrics.json").read_text())
    rows = [r for r in jl(run / "confirm_rows.jsonl") if r["data"] == "dinj"]
    by = {}
    for r in rows:
        by.setdefault(r["run"], {})[r["item_id"]] = int(r["correct"])
    main, rand = by[sel["main_run"]], [by[k] for k in sel["controls"]["random"]]
    ids = sorted(main)
    d = np.array([main[i] - np.mean([rr[i] for rr in rand]) for i in ids])
    generic_valid = sel["main_setting"]["generic_valid"]
    dg = np.array([main[i] - by[sel["controls"]["generic"]][i] for i in ids]) if generic_valid else None
    dinj_test = sum(r["split"] == "test" for r in jl(ROOT / "data" / "processed" / "dinj.jsonl"))
    kept = m["n_dir_items"] / (m["n_dir_items"] + m["n_dinj_excluded_no_content"])
    n = int(round(dinj_test * kept))
    values = np.round(np.arange(-5, 6) / 5, 1)
    base = np.array([np.isclose(d, v).mean() for v in values])
    zero = int(np.where(values == 0)[0][0])
    pilot = float(d.mean())
    out = {"pilot_n": len(ids), "pilot_mean_d": pilot, "generic_valid": generic_valid, "test_n": n, "rows": []}
    for dm in (0.5, 1.0, 1.5):
        nz = 1 - base[zero]
        p = base.copy()
        scale = min(dm, 1 / nz) if nz > 0 else 0
        p[np.arange(len(values)) != zero] *= scale
        p[zero] = 1 - p[np.arange(len(values)) != zero].sum()
        for name, eff in (("δ/2", DELTA / 2), ("δ", DELTA), ("half pilot", abs(pilot) / 2)):
            if eff >= (p * np.abs(values)).sum() or eff <= 0:
                out["rows"].append({"disc": float(1 - p[zero]), "scenario": name, "effect": eff, "power": None})
                continue
            w = tilt(values, p, eff)
            pv = np.empty(SIMS)
            for k in range(SIMS):
                x = rng.choice(values, n, p=w)
                pv[k] = one_sided_paired(x, np.zeros(n), 0.0, "greater")
            out["rows"].append({"disc": float(1 - p[zero]), "scenario": name, "effect": eff,
                                "power": {f: float((pv < ALPHA / f).mean()) for f in FAMILIES}})
    return out


def fmt(r):
    return " | ".join("none" if r["power"] is None else f"{r['power'][f]:.2f}" for f in FAMILIES)


def main():
    sel = json.loads((ROOT / "analysis" / "out" / "tier2.json").read_text())
    rng = np.random.default_rng(SEED)
    res = {"H4a": {}, "H3b": None}
    lines = ["# Power for H4a and H3b (generated by analysis/p2_power_tier2.py; dev pilots)", "",
             f"δ = {100 * DELTA:.0f} pp; power at α/9 and α/6 (first Holm step with all / none of the Tier 2 hypotheses "
             "enabled); 'none' = the scenario effect exceeds what the scaled discordance allows, or there is no "
             "discordance. See the script docstring for the scenarios.", ""]
    head = ["| Model | Pool | Pilot runs | Pilot discordance % | Pilot effect (pp) | Scenario discordance % | Scenario | "
            "Effect (pp) | Power α/9 | Power α/6 |", "|---|---|---|---|---|---|---|---|---|---|"]
    lines += ["## H4a (Exp 5: self vs perm-self, parameter arm; two-sided exact McNemar, n = " + str(N_EXP5) + ")", ""] + head
    for model in ("4B", "1.7B"):
        s = sel["exp5"].get(model)
        if not s:
            lines.append(f"| {model} | TODO(missing: Exp 5 pilot) | | | | | | | | |")
            continue
        r = h4a(model, s, rng)
        res["H4a"][model] = {**r, "serve_run": s["serve"], "update_repeat_identical": s["update_repeat_identical"]}
        for pool, v in r["pools"].items():
            for row in v["rows"]:
                lines.append(f"| {model} | {pool} ({100 * v['wrong_share']:.0f} % wrong) | {s['serve']} | "
                             f"{100 * v['pilot_disc']:.1f} | {100 * v['pilot_effect']:.1f} | {100 * row['disc']:.1f} | "
                             f"{row['scenario']} | {100 * row['effect']:.1f} | {fmt(row)} |")
    lines += ["", "Pilot strata (tuning items at the selected τ and lr): " + "; ".join(
        f"{m} right n = {v['pilot'][True]['n']}, wrong n = {v['pilot'][False]['n']}" for m, v in res["H4a"].items()) + ".", ""]
    e3 = sel["exp3"].get("4B")
    lines += ["## H3b (Exp 3, 4B main setting: W→R steered − mean of 5 random directions; one-sided)", ""]
    if not e3 or not e3.get("main_run"):
        lines.append("TODO(missing: Exp 3 main setting)")
    else:
        r = h3b(e3, rng)
        res["H3b"] = {**r, "run": e3["run"], "main_run": e3["main_run"]}
        lines += [f"Pilot: {e3['run']}, main setting {e3['main_run']}, {r['pilot_n']} D-inj items, mean d = "
                  f"{100 * r['pilot_mean_d']:.1f} pp; generic control valid at this setting: {r['generic_valid']} "
                  f"({'intersection-union with the generic control' if r['generic_valid'] else 'random control only'}). "
                  f"Test n = {r['test_n']}.", "",
                  "| Scenario discordance % | Scenario | Effect (pp) | Power α/9 | Power α/6 |", "|---|---|---|---|---|"]
        for row in r["rows"]:
            lines.append(f"| {100 * row['disc']:.1f} | {row['scenario']} | {100 * row['effect']:.1f} | {fmt(row)} |")
        if r["generic_valid"]:
            lines += ["", "TODO(missing: the generic-control component of the intersection-union test is not simulated)"]
    lines.append("")
    (ROOT / "analysis" / "out" / "power_tier2.json").write_text(json.dumps(res, indent=1, default=str))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
