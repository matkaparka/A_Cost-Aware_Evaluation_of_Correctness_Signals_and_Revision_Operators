"""P3 power re-run for the D-nat Tier 1 tests with the expansion (design §7.4, §9; central scenario: DECISIONS
2026-11-04) -> reports/03_power_tier1_tables.md and analysis/out/power_tier1_p3.json.

Inputs, dev split only: the P3 dev Exp 1A pilots of cloud/p3c_sequence.sh, both with F5 at the dev threshold t*.
  - tag tstar: the original dev items, strata gsm8k and math500;
  - tag ext: the expansion's dev items, stratum math_ext.
Stratum test sizes come from data/processed/dnat.jsonl and data/processed/ext/dnat_ext.jsonl; H2b's test wrong counts from
the initial-answer runs (right/wrong counts only, design §5.3).

Scopes (the population an estimand refers to):
  - original: gsm8k + math500 as sampled (the design's D-nat);
  - pooled: gsm8k + math500 + math_ext as sampled (MATH becomes the majority);
  - reweighted: all three strata, with MATH (math500 + math_ext, pooled) weighted back to its share of the original test
    pool. Estimate: the weighted mean of the two group means; test: z with the stratified standard error.
For a single group the test equals the plain paired test of analysis/stats.py.

Scenarios. A paired contrast is the item-level difference d (H1a: F2 - F1 in {-1, 0, 1}; H1d: F5 - the mean of the K = 3
F3-perm outcomes, in multiples of 1/3). Per stratum, the pilot gives the distribution of |d|; a scenario multiplies its
non-zero mass by {0.5, 1, 1.5} and gives each non-zero d the sign + with probability (1 + r) / 2. The estimand's true
effect is then e = r x (weighted mean of |d|), the same share r of discordant mass going one way in every stratum;
e is infeasible when |r| would exceed 1. Central: pilot discordance, delta/2 for equivalence, delta for superiority.
- H1a: TOST of F2 vs F1 at +-delta; effects 0, delta/2, delta.
- H1d: two-sided z on d; effects delta/2, delta, and r = 0.25, 0.5 of each scenario's discordant mass (delta is beyond the
  discordance on these models, DECISIONS 2026-11-04). The sign follows the pilot estimate.
- H1b: slope of correctness on p in {0.5, 0.7, 0.9, 1.0} under F3(p), simulated with the design's own mechanism. Nested
  masks (src/data/verifiers.py) give each item one uniform U, and its verdict is truthful at level p iff U < p. So an item
  has two outcomes, Y_c under a truthful and Y_w under an untruthful verdict, its outcome at p is Y_c if U < p else Y_w,
  and acc(1.0) - acc(0.5) = E[Y_c - Y_w] / 2. The pilot pairs (Y_c, Y_w) come from the dev items with U >= 0.5 (the
  verdict at p = 0.5 is untruthful: Y_w from F3(0.5), Y_c from F3(1.0)), a random half of the items. d = Y_c - Y_w is
  then treated as for H1a. Scenarios: trend delta/2, delta (central), the pilot's, and delta at the least discordance
  that allows it (r = 1). Test: one-sided z on the item-level OLS slope; the pre-registered GEE slope is run on the same
  draws for the original and pooled scopes at the pilot discordance (--gee-sims), to show the two agree.
- H2b: intersection-union test of analysis/p2_power_more.py on initially wrong items, n = a fifth of the scope's test
  wrong count (the probe quintile's stratum mix is not modelled). Scenarios: the pilot pattern (resampled triples); the
  central one, F2 - F0 = delta_w / 2 (half-way to the non-inferiority margin) and F3(1.0) - F0 = delta; a favourable one,
  F2 - F0 = 0 and F3(1.0) - F0 = 2 delta. In the last two, d2 = F2 - F0 and d3 = F3(1.0) - F0 keep their pilot
  discordance (signed magnitudes as for H1a) and are drawn independently. delta_w in {5, 10} pp; scopes original and
  pooled.
Levels: alpha/6 and alpha/9, the first Holm step for a family of 6 or 9 (the family is fixed at the freeze).
Also reported: whether the tstar run reproduces the P2 run on the conditions that do not depend on the threshold.
Usage: python analysis/p3_power_tier1.py [--sims N] [--gee-sims N]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from power import pattern_probs  # noqa: E402
from stats import _norm_sf, gee_slope, one_sided_paired  # noqa: E402

DELTA, ALPHA, SEED = 0.03, 0.05, 0
FAMILIES = (6, 9)
P_LEVELS = (0.5, 0.7, 0.9, 1.0)
K_PERM = 3
COMMIT = "6cd867e"
MODELS = ("4B", "1.7B")
RUNS = {m: {"orig": f"p2-exp01-dev-tstar-{m}-{COMMIT}", "ext": f"p2-exp01-dev-ext-{m}-{COMMIT}"} for m in MODELS}
P2_RUN = {m: f"p2-exp01-dev-{m}-f7e5011" for m in MODELS}
INITIAL = {m: {"orig": f"p1-initial-{m}-d2ebdea", "ext": f"p1-initial-ext-{m}-45575c2"} for m in MODELS}
STRATUM = {"gsm8k": "gsm8k", "math500": "math500", "math_full": "math_ext"}
CONDS = ["F0", "F1", "F2", "F5"] + [f"F3({p})" for p in P_LEVELS] + [f"F3-perm[{k}]" for k in range(K_PERM)]
OUT_MD = ROOT / "reports" / "03_power_tier1_tables.md"
OUT_JSON = ROOT / "analysis" / "out" / "power_tier1_p3.json"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def scopes(n_test):
    g, m5, x = n_test["gsm8k"], n_test["math500"], n_test["math_ext"]
    return {"original": [(1.0, ["gsm8k", "math500"])],
            "pooled": [(1.0, ["gsm8k", "math500", "math_ext"])],
            "reweighted": [(g / (g + m5), ["gsm8k"]), (m5 / (g + m5), ["math500", "math_ext"])]}


def omega(scope, n):
    """Each stratum's weight in the scope's estimand."""
    w = {}
    for wg, strata in scope:
        tot = sum(n[s] for s in strata)
        for s in strata:
            w[s] = wg * n[s] / tot
    return w


def z_p(est, se, margin, direction):
    if se == 0:   # as stats.one_sided_paired
        if direction == "greater":
            return 0.0 if est > margin else 1.0
        return 0.0 if est < margin else 1.0
    z = (est - margin) / se
    return _norm_sf(z) if direction == "greater" else _norm_sf(-z)


def grouped(scope, moments):
    """moments[s] = (n, sum, sum of squares) of d in stratum s -> (estimate, standard error)."""
    est, var = 0.0, 0.0
    for wg, strata in scope:
        n = sum(moments[s][0] for s in strata)
        s1 = sum(moments[s][1] for s in strata)
        s2 = sum(moments[s][2] for s in strata)
        mean = s1 / n
        v = (s2 - n * mean ** 2) / (n - 1) if n > 1 else 0.0
        est += wg * mean
        var += wg ** 2 * max(v, 0.0) / n
    return est, math.sqrt(var)


# ---------- signed-magnitude scenarios (H1a, H1d) ----------

def magnitudes(d):
    """Pilot distribution of |d| over its distinct values (rounded to thirds)."""
    a = np.round(np.abs(d) * 3) / 3
    vals = np.unique(a)
    return vals, np.array([(a == v).mean() for v in vals])


def scenario_probs(vals, p, dmult, r):
    """Signed values and probabilities: non-zero mass x dmult, sign + with probability (1 + r) / 2."""
    nz = vals > 0
    q = p.copy()
    if q[nz].sum() > 0:
        q[nz] *= min(dmult, 1 / q[nz].sum())
    q[~nz] = 1 - q[nz].sum()
    sv = np.concatenate([vals[~nz], vals[nz], -vals[nz]])
    sp = np.concatenate([q[~nz], q[nz] * (1 + r) / 2, q[nz] * (1 - r) / 2])
    return sv, sp / sp.sum(), float((q * vals).sum())


def simulate_signed(scope, n, dist, dmult, effect, test, rng, sims):
    """dist[s] = (vals, probs). Returns (feasible, power per family, r)."""
    w = omega(scope, n)
    reach = sum(w[s] * scenario_probs(*dist[s], dmult, 0.0)[2] for s in w)
    if reach == 0:
        return False, None, None
    r = effect / reach
    if abs(r) > 1 + 1e-9:
        return False, None, r
    r = float(np.clip(r, -1, 1))
    pv = np.empty(sims)
    probs = {s: scenario_probs(*dist[s], dmult, r) for s in w}
    for k in range(sims):
        mom = {}
        for s in w:
            sv, sp, _ = probs[s]
            c = rng.multinomial(n[s], sp)
            mom[s] = (n[s], float((c * sv).sum()), float((c * sv ** 2).sum()))
        est, se = grouped(scope, mom)
        pv[k] = test(est, se)
    return True, {f: float((pv < ALPHA / f).mean()) for f in FAMILIES}, r


def tost_p(est, se):
    return max(z_p(est, se, -DELTA, "greater"), z_p(est, se, DELTA, "less"))


def two_sided_p(est, se):
    return min(1.0, 2 * min(z_p(est, se, 0, "greater"), z_p(est, se, 0, "less")))


# ---------- H1b ----------

LV = np.array(P_LEVELS)
LW = (LV - LV.mean()) / ((LV - LV.mean()) ** 2).sum()   # item-level OLS slope = outcomes @ LW
# Nested masks (src/data/verifiers.py): one uniform U per item; the verdict is truthful at level p iff U < p. An item
# whose outcomes under a truthful and an untruthful verdict differ by d = Y_c - Y_w has OLS slope d x g(U), g below.
G_VALS = np.array([0.0, LW[1:].sum(), LW[2:].sum(), LW[3:].sum()])
G_PROBS = np.array([0.5, 0.2, 0.2, 0.1])                  # U in [0, .5), [.5, .7), [.7, .9), [.9, 1)


def h1b_probs(dist, dmult, r):
    """Signed item slopes d x g: values and probabilities."""
    sv, sp, _ = scenario_probs(*dist, dmult, r)
    return np.outer(sv, G_VALS).ravel(), np.outer(sp, G_PROBS).ravel()


def h1b_power(scope, n, dist, joint, dmult, trend, rng, sims, gee_sims=0):
    """trend: the estimand's acc(1.0) - acc(0.5) = E[d] / 2. Returns (feasible, result, r)."""
    w = omega(scope, n)
    reach = sum(w[s] * scenario_probs(*dist[s], dmult, 0.0)[2] for s in w)
    r = 2 * trend / reach if reach > 0 else None
    if r is None or abs(r) > 1 + 1e-9:
        return False, None, r
    r = float(np.clip(r, -1, 1))
    probs = {s: h1b_probs(dist[s], dmult, r) for s in w}
    pv = np.empty(sims)
    for k in range(sims):
        mom = {}
        for s in w:
            v, pr = probs[s]
            c = rng.multinomial(n[s], pr)
            mom[s] = (n[s], float((c * v).sum()), float((c * v ** 2).sum()))
        pv[k] = z_p(*grouped(scope, mom), 0, "greater")
    res = {"z": {f: float((pv < ALPHA / f).mean()) for f in FAMILIES}}
    if gee_sims:
        pz, pg = [], []
        for _ in range(gee_sims):
            Ys, mom = [], {}
            for s in w:
                # (Y_c, Y_w) cells in the order of analysis/power.py pattern_probs: (1,1), (1,0), (0,1), (0,0)
                D = scenario_probs(*dist[s], dmult, 0.0)[2]
                cells = pattern_probs(D, r * D, joint[s])
                k = rng.choice(4, n[s], p=cells)
                yc, yw = np.array([1, 1, 0, 0])[k], np.array([1, 0, 1, 0])[k]
                u = rng.random(n[s])
                Y = np.where(u[:, None] < LV[None, :], yc[:, None], yw[:, None]).astype(float)
                Ys.append(Y)
                c = Y @ LW
                mom[s] = (n[s], float(c.sum()), float((c ** 2).sum()))
            pz.append(z_p(*grouped(scope, mom), 0, "greater"))
            Y = np.concatenate(Ys)
            pg.append(gee_slope(Y.ravel(), np.tile(LV, len(Y)), np.repeat(np.arange(len(Y)), 4)))
        pz, pg = np.array(pz), np.array(pg)
        res["z_same_draws"] = {f: float((pz < ALPHA / f).mean()) for f in FAMILIES}
        res["gee_same_draws"] = {f: float((pg < ALPHA / f).mean()) for f in FAMILIES}
    return True, res, r


# ---------- H2b ----------

def h2b_pilot_power(n_by_s, T, dw, rng, sims):
    """The pilot pattern: resampled (F0, F2, F3(1.0)) triples of initially wrong dev items."""
    hits = {f: 0 for f in FAMILIES}
    for _ in range(sims):
        t = np.concatenate([T[s][rng.integers(0, len(T[s]), n)] for s, n in n_by_s.items()])
        p = max(one_sided_paired(t[:, 1], t[:, 0], dw, "less"), one_sided_paired(t[:, 2], t[:, 0], 0.0, "greater"))
        for f in FAMILIES:
            hits[f] += p < ALPHA / f
    return {f: hits[f] / sims for f in FAMILIES}


def h2b_true_power(n_by_s, T, gap2, gap3, dw, rng, sims):
    """H2b scenarios: d2 = F2 - F0 with mean gap2 and d3 = F3(1.0) - F0 with mean gap3, each with its pilot discordance
    per stratum (signed magnitudes as for H1a), drawn independently. Returns (feasible, power)."""
    scope = [(1.0, list(n_by_s))]
    d2 = {s: magnitudes(T[s][:, 1] - T[s][:, 0]) for s in n_by_s}
    d3 = {s: magnitudes(T[s][:, 2] - T[s][:, 0]) for s in n_by_s}
    nn = sum(n_by_s.values())
    reach2 = sum(n_by_s[s] / nn * scenario_probs(*d2[s], 1.0, 0.0)[2] for s in n_by_s)
    reach3 = sum(n_by_s[s] / nn * scenario_probs(*d3[s], 1.0, 0.0)[2] for s in n_by_s)
    if reach3 == 0 or gap3 / reach3 > 1 or (gap2 and (reach2 == 0 or gap2 / reach2 > 1)):
        return False, None
    p2 = {s: scenario_probs(*d2[s], 1.0, gap2 / reach2 if gap2 else 0.0) for s in n_by_s}
    p3 = {s: scenario_probs(*d3[s], 1.0, gap3 / reach3) for s in n_by_s}
    hits = {f: 0 for f in FAMILIES}
    for _ in range(sims):
        m2, m3 = {}, {}
        for s, n in n_by_s.items():
            for m, (v, pr, _) in ((m2, p2[s]), (m3, p3[s])):
                c = rng.multinomial(n, pr)
                m[s] = (n, float((c * v).sum()), float((c * v ** 2).sum()))
        p = max(z_p(*grouped(scope, m2), dw, "less"), z_p(*grouped(scope, m3), 0.0, "greater"))
        for f in FAMILIES:
            hits[f] += p < ALPHA / f
    return True, {f: hits[f] / sims for f in FAMILIES}


# ---------- data ----------

def load(model):
    """Per-item outcomes by condition from the two P3 pilots, with each item's stratum."""
    meta = {r["item_id"]: STRATUM[r["dataset"]] for r in jl(ROOT / "data" / "processed" / "dnat.jsonl")}
    meta.update({r["item_id"]: STRATUM[r["dataset"]] for r in jl(ROOT / "data" / "processed" / "ext" / "dnat_ext.jsonl")})
    items = defaultdict(dict)
    for tag in ("orig", "ext"):
        for r in jl(ROOT / "results" / "p2" / model / RUNS[model][tag] / "rows.jsonl"):
            if r["exp"] == "1A" and r["data"] == "dnat" and r["cond"] in CONDS:
                items[r["item_id"]][r["cond"]] = int(bool(r["correct"]))
                items[r["item_id"]]["correct0"] = int(bool(r["correct0"]))
                if r["cond"] == "F3(0.5)":
                    items[r["item_id"]]["untruthful_at_0.5"] = bool(r["verdict"]) != bool(r["correct0"])
    full = {i: v for i, v in items.items() if all(c in v for c in CONDS)}
    by_s = defaultdict(list)
    for i, v in full.items():
        by_s[meta[i]].append(v)
    return by_s, len(items) - len(full)


def reproduction(model):
    """Item-level agreement of the tstar run with the P2 run on the threshold-free conditions: generated text, and the
    label (the P2 run was scored by scorer 1.0.0 at run time, so its extracted answers can differ)."""
    key = lambda r: (r["item_id"], r["cond"])
    free = {"B1", "F0", "F1", "F2", "F2-scratch"} | {f"F3({p})" for p in P_LEVELS}
    sides = []
    for run in (P2_RUN[model], RUNS[model]["orig"]):
        d = ROOT / "results" / "p2" / model / run
        gen = {g["prompt_key"]: g["text"] for g in jl(d / "generations.jsonl")}
        sides.append({key(r): (r["correct"], gen[r["prompt_key"]]) for r in jl(d / "rows.jsonl")
                      if r["exp"] == "1A" and r["data"] == "dnat" and r["cond"] in free})
    a, b = sides
    common = set(a) & set(b)
    return {"rows": len(common), "same_text": sum(a[k][1] == b[k][1] for k in common),
            "same_correct": sum(a[k][0] == b[k][0] for k in common)}


def test_counts(model):
    """Test-split sizes per stratum, and initially wrong test items per stratum (counts only)."""
    n, wrong = defaultdict(int), defaultdict(int)
    for r in jl(ROOT / "data" / "processed" / "dnat.jsonl"):
        if r["split"] == "test":
            n[STRATUM[r["dataset"]]] += 1
    for r in jl(ROOT / "data" / "processed" / "ext" / "dnat_ext.jsonl"):
        if r["split"] == "test":
            n[STRATUM[r["dataset"]]] += 1
    for tag in ("orig", "ext"):
        for r in jl(ROOT / "results" / "p1" / model / INITIAL[model][tag] / "initial.jsonl"):
            if r["split"] == "test" and not r["correct"]:
                wrong[STRATUM[r["dataset"]]] += 1
    return dict(n), dict(wrong)


# ---------- report ----------

def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def pw(d):
    return "none | none" if d is None else " | ".join(f"{d[f]:.2f}" for f in FAMILIES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--gee-sims", type=int, default=100)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    lines = ["# P3 power re-run for the D-nat Tier 1 tests with the expansion (generated by analysis/p3_power_tier1.py)", "",
             "Dev pilots only (cloud/p3c_sequence.sh, F5 at t*). Scopes, scenarios and tests: see the script docstring. "
             f"δ = {100 * DELTA:.0f} pp; power at α/{FAMILIES[0]} | α/{FAMILIES[1]}; {a.sims} simulations "
             f"(H1b GEE check: {a.gee_sims}). 'none' = infeasible: the effect needs more one-way discordant mass than the "
             "scenario has. **Central** rows are marked.", ""]
    out = {}
    for model in MODELS:
        lines += [f"## Qwen3-{model}", ""]
        missing = [t for t in ("orig", "ext") if not (ROOT / "results" / "p2" / model / RUNS[model][t] / "rows.jsonl").exists()]
        if missing:
            lines += [f"TODO(missing: {', '.join(RUNS[model][t] for t in missing)})", ""]
            out[model] = "missing"
            continue
        by_s, incomplete = load(model)
        n, wrong = test_counts(model)
        sc = scopes(n)
        rep = reproduction(model)
        res = {"test_n": n, "test_wrong": wrong, "incomplete_items": incomplete, "reproduction": rep}
        lines += [f"Runs: `{RUNS[model]['orig']}`, `{RUNS[model]['ext']}`. Dev items with every condition: "
                  + ", ".join(f"{s} {len(v)}" for s, v in sorted(by_s.items())) + f" (incomplete: {incomplete}). "
                  "Test sizes: " + ", ".join(f"{s} {n[s]}" for s in sorted(n)) + ".", "",
                  f"Reproduction of the P2 run `{P2_RUN[model]}` on the threshold-free 1A conditions (B1, F0, F1, F2, "
                  f"F2-scratch, F3(p)): {rep['same_text']} of {rep['rows']} rows with the identical generated text, "
                  f"{rep['same_correct']} with the same label (different batch composition; VLLM_BATCH_INVARIANT=1).", ""]

        # pilot description per stratum
        lines += ["| Stratum | Dev n | Acc F1 % | Acc F2 % | F2 − F1 (pp) | F2/F1 discordance % | Acc F5 % | Acc F3-perm (mean K) % | "
                  "F5 − perm (pp) | Acc F3(0.5 / 0.7 / 0.9 / 1.0) % |", "|---|---|---|---|---|---|---|---|---|---|"]
        D = {}
        for s, v in sorted(by_s.items()):
            f1 = np.array([x["F1"] for x in v]); f2 = np.array([x["F2"] for x in v]); f5 = np.array([x["F5"] for x in v])
            perm = np.array([[x[f"F3-perm[{k}]"] for k in range(K_PERM)] for x in v]).mean(1)
            Y = np.array([[x[f"F3({p})"] for p in P_LEVELS] for x in v], float)
            T = np.array([[x["F0"], x["F2"], x["F3(1.0)"]] for x in v if not x["correct0"]], float)
            # Items with U >= 0.5 (untruthful verdict at p = 0.5): a random half whose outcome under both verdicts is seen
            # (Y_c from F3(1.0), Y_w from F3(0.5)).
            sw = np.array([[x["F3(1.0)"], x["F3(0.5)"]] for x in v if x["untruthful_at_0.5"]], float)
            D[s] = {"h1a": f2 - f1, "h1d": f5 - perm, "Y": Y, "T": T, "switch": sw}
            lines.append(f"| {s} | {len(v)} | {pct(f1.mean())} | {pct(f2.mean())} | {pct((f2 - f1).mean())} | "
                         f"{pct((f2 != f1).mean())} | {pct(f5.mean())} | {pct(perm.mean())} | {pct((f5 - perm).mean())} | "
                         + " / ".join(pct(y) for y in Y.mean(0)) + " |")
        lines.append("")

        # H1a, H1d
        for h, kind in (("h1a", "TOST F2 vs F1"), ("h1d", "two-sided F5 vs F3-perm (K-mean)")):
            dist = {s: magnitudes(D[s][h]) for s in D}
            lines += [f"**{h[:2].upper() + h[2:]}** ({kind})", "",
                      "| Scope | Test n | Pilot estimate (pp) | Pilot discordant mass (mean abs d, %) | Discordance × | Effect | "
                      f"r | Power α/{FAMILIES[0]} | Power α/{FAMILIES[1]} |", "|---|---|---|---|---|---|---|---|---|"]
            res[h] = {}
            for name, scope in sc.items():
                w = omega(scope, n)
                est = sum(w[s] * D[s][h].mean() for s in w)
                mass = sum(w[s] * np.abs(D[s][h]).mean() for s in w)
                sign = 1.0 if est >= 0 else -1.0
                if h == "h1a":
                    effects = [("0", 0.0, False), ("δ/2", DELTA / 2, True), ("δ", DELTA, False)]
                    test = tost_p
                else:
                    effects = [("δ/2", sign * DELTA / 2, False), ("δ", sign * DELTA, True),
                               ("r 0.25", sign * 0.25, False), ("r 0.5", sign * 0.5, False)]
                    test = two_sided_p
                rows = []
                for dm in (0.5, 1.0, 1.5):
                    reach = sum(w[s] * scenario_probs(*dist[s], dm, 0.0)[2] for s in w)
                    for lab, e, central in effects:
                        e = e * reach if lab.startswith("r ") else e   # a share of this scenario's discordant mass
                        ok, power, r = simulate_signed(scope, n, dist, dm, e, test, rng, a.sims)
                        rows.append({"dmult": dm, "effect_label": lab, "effect": e, "r": r, "power": power,
                                     "central": central and dm == 1.0})
                        lines.append(f"| {name} | {sum(n[s] for s in w)} | {pct(est)} | {pct(mass)} | {dm:g} | "
                                     f"{'**' + lab + ' (central)**' if central and dm == 1.0 else lab} ({pct(e)} pp) | "
                                     f"{'—' if r is None else f'{r:.2f}'} | {pw(power)} |")
                res[h][name] = {"pilot_estimate": est, "pilot_mass": mass, "rows": rows}
            lines.append("")

        # H1b
        dist = {s: magnitudes(D[s]["switch"][:, 0] - D[s]["switch"][:, 1]) for s in D}
        joint = {}
        for s in D:
            c = D[s]["switch"][D[s]["switch"][:, 0] == D[s]["switch"][:, 1]]
            joint[s] = float(c[:, 0].mean()) if len(c) else 0.5
        lines += ["**H1b** (slope on p under F3(p); verdict-switch model; z on the item-level OLS slope, GEE check on the original and pooled scopes"
                  ")", "",
                  "| Stratum | Items with U ≥ 0.5 | Y_c ≠ Y_w % | E[Y_c − Y_w] (pp) |", "|---|---|---|---|"]
        for s in sorted(D):
            sw = D[s]["switch"]
            lines.append(f"| {s} | {len(sw)} | {pct((sw[:, 0] != sw[:, 1]).mean() if len(sw) else None)} | "
                         f"{pct((sw[:, 0] - sw[:, 1]).mean() if len(sw) else None)} |")
        lines += ["", "| Scope | Test n | Pilot trend (pp): observed / switch model | Discordance × | Scenario trend | r | "
                  f"Power α/{FAMILIES[0]} | Power α/{FAMILIES[1]} | GEE vs z, same draws (α/{FAMILIES[0]}) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        res["h1b"] = {}
        for name, scope in sc.items():
            w = omega(scope, n)
            observed = 0.5 * sum(w[s] * (D[s]["Y"] @ LW).mean() for s in w)
            model_trend = 0.5 * sum(w[s] * (D[s]["switch"][:, 0] - D[s]["switch"][:, 1]).mean() for s in w)
            rows = []
            reach1 = sum(w[s] * scenario_probs(*dist[s], 1.0, 0.0)[2] for s in w)
            dm_min = 2 * DELTA / reach1 if reach1 > 0 else None
            grid = [(dm, lab, tr, central) for dm in (0.5, 1.0, 1.5)
                    for lab, tr, central in (("δ/2", DELTA / 2, False), ("δ", DELTA, True), ("pilot", model_trend, False))]
            if dm_min is not None:
                grid.append((dm_min, "δ, least discordance (r = 1)", DELTA, False))
            for dm, lab, tr, central in grid:
                gs = a.gee_sims if name in ("original", "pooled") and dm == 1.0 else 0
                ok, pr, r = h1b_power(scope, n, dist, joint, dm, tr, rng, a.sims, gs)
                rows.append({"dmult": dm, "scenario": lab, "trend": tr, "r": r, "central": central and dm == 1.0,
                             **(pr or {"z": None})})
                chk = (f"{pr['gee_same_draws'][FAMILIES[0]]:.2f} vs {pr['z_same_draws'][FAMILIES[0]]:.2f}"
                       if pr and gs else "")
                lab_s = f"**{lab} (central)**" if central and dm == 1.0 else lab
                lines.append(f"| {name} | {sum(n[s] for s in w)} | {pct(observed)} / {pct(model_trend)} | {dm:.3g} | "
                             f"{lab_s} ({pct(tr)} pp) | {'—' if r is None else f'{r:.2f}'} | "
                             f"{pw(pr['z'] if pr else None)} | {chk} |")
            res["h1b"][name] = {"observed_trend": observed, "switch_model_trend": model_trend, "rows": rows}
        lines.append("")

        # H2b
        T = {s: D[s]["T"] for s in D}
        lines += ["**H2b** (intersection-union on initially wrong items, n = a fifth of the test wrong count)", "",
                  "| Scope | n | Dev wrong items | W→R F0 / F2 / F3(1.0) % (dev, scope mix) | Scenario | δ_w | "
                  f"Power α/{FAMILIES[0]} | Power α/{FAMILIES[1]} |", "|---|---|---|---|---|---|---|---|"]
        res["h2b"] = {}
        for name, strata in (("original", ["gsm8k", "math500"]), ("pooled", ["gsm8k", "math500", "math_ext"])):
            strata = [s for s in strata if len(T[s])]   # a stratum without dev wrong items gives no pattern
            nb = {s: int(round(wrong.get(s, 0) / 5)) for s in strata}
            nn = sum(nb.values())
            wr = np.array([sum(nb[s] * T[s][:, j].mean() for s in strata) / nn for j in range(3)])
            rows = []
            for lab, g2, g3 in (("pilot pattern", None, None),
                                ("**central: F2 − F0 = δ_w/2, F3 − F0 = δ**", 0.5, DELTA),
                                ("favourable: F2 − F0 = 0, F3 − F0 = 2δ", 0.0, 2 * DELTA)):
                for dw in (0.05, 0.10):
                    if g2 is None:
                        pr = h2b_pilot_power(nb, T, dw, rng, a.sims // 2)
                    else:
                        pr = h2b_true_power(nb, T, g2 * dw, g3, dw, rng, a.sims)[1]
                    rows.append({"scenario": lab, "delta_w": dw, "power": pr})
                    lines.append(f"| {name} | {nn} | {sum(len(T[s]) for s in strata)} | {' / '.join(pct(x) for x in wr)} | "
                                 f"{lab} | {100 * dw:.0f} pp | {pw(pr)} |")
            res["h2b"][name] = {"n": nn, "wr_dev_mix": wr.tolist(), "rows": rows}
        lines.append("")
        out[model] = res
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float))
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
