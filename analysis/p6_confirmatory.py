"""Confirmatory analysis (P6), frozen with prereg.md: every family test, its estimate and CI, Holm, and the outcome
category; the estimation-only hypotheses; the secondary D-nat estimands; the pre-registered sensitivity analyses.
-> reports/<out>.md and analysis/out/<out>.json

Inputs come from a run spec (JSON) that names the result files, so the same code runs on test (P6) and, before the
freeze, on the dev pilots (a dry run of the code path, not a result):
  model, split, exp01_orig (Exp 1A on the original pool, F5 at t*), exp01_1b (Exp 1B; the same run as exp01_orig on
  test), exp01_ext (Exp 1A on the expansion, or null), initial / verbal (P1 jsonl files, original and expansion),
  probe_scores (jsonl of {item_id, p_correct}) or probe_npz + probe_acts (acts.npz files scored with the frozen probe),
  exp3_rows / exp3_texts (Exp 3 rows and texts), exp3_main (label of the main-setting arm), exp3_random (labels of the
  5 random arms).
A missing input leaves its hypotheses "not run" (family members keep their slot with p = 1, prereg §4).

Rules (prereg §3–§5, §8):
- Unit: the canonical problem. Tests use cluster-robust standard errors by canonical ID (analysis/stats.py); CIs are
  95 % percentile intervals from 10,000 resamples of canonical problems with all their items; CIs are not adjusted for
  multiplicity; Holm over the family of 6 at alpha 0.05 is computed once, here.
- D-nat estimand: the pooled pool as sampled (original + expansion). If the expansion run is absent, D-nat tests run on
  the original pool and are labelled so (prereg §10).
- H1a: TOST of F2 vs F1 at +-delta. supported: Holm p < alpha; contradicted: the CI lies entirely beyond +delta or
  -delta; else inconclusive.
- H1b: one-sided GEE slope of correctness on p (clusters = canonical ID). Effect: acc(1.0) - acc(0.5) on the fitted
  linear trend = half the mean item-level OLS slope. supported: Holm p < alpha, written "supported (effect below
  delta)" when the estimate is below delta; not supported: not significant and the CI's upper end < delta/2 (the
  powered effect, prereg §3.2); else inconclusive.
- H2a: AUROC(frozen probe P(correct)) - AUROC(s = log P(yes) - log P(no)) for a0 correctness; p = one-sided
  cluster-bootstrap p (stats.auroc_diff_boot); DeLong reported as a check, and the outcome is flagged
  "method-sensitive" when DeLong in its place would change significance after Holm. supported: Holm p < alpha; not
  supported: not significant and the CI's upper end < delta_AUC; else inconclusive.
- Chunked blocks (Exp 3): below min_n (analysis/out/min_n.json) the member keeps p = 1 and is descriptive; between min_n
  and the planned n it is labelled a pre-registered partial-sample analysis.
- H3a-role, H3a-label (non-directional, prereg §5): two-sided test on item-level d_i (D-inj, F0). supported: Holm p <
  alpha, either sign (sign reported); not supported: not significant and the CI inside (-delta, delta); else inconclusive.
- H3b: one-sided test of d_i = W->R(main) - mean W->R(5 random arms) on D-inj items. supported: Holm p < alpha; not
  supported: not significant and the CI's upper end < delta; else inconclusive.
- Estimation-only: H1d (F5 - mean of the K = 3 F3-perm), H2b (i) W->R(F2) - W->R(F0) and (ii) W->R(F3(1.0)) - W->R(F0)
  on initially wrong items in the top quintile of probe P(correct), the quintile recomputed in every resample.
- Secondary (descriptive): original-pool, per-stratum and reweighted-to-the-original-mix D-nat estimates; strict-reading
  sensitivity (H1b estimation-only, Holm over the other 5).
- Sensitivity: every family estimate and raw p with second-scorer v2 labels (src/score/second.py); D-nat and H2a without
  items whose a0 is truncated or unparsed; per-condition main-vs-v2 disagreement rates.
Usage: python analysis/p6_confirmatory.py --spec <spec.json> --out <name>
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
sys.path.insert(0, str(ROOT))
from stats import auroc, auroc_diff_boot, gee_slope, holm, one_sided_paired, tost, two_sided_paired  # noqa: E402
from src.score.second import score_v2  # noqa: E402

ALPHA, DELTA, DELTA_AUC, N_BOOT, SEED = 0.05, 0.03, 0.03, 10000, 0
P_LEVELS = (0.5, 0.7, 0.9, 1.0)
K_PERM = 3
FAMILY = ["H1a", "H1b", "H2a", "H3a-role", "H3a-label", "H3b"]
STRATUM = {"gsm8k": "gsm8k", "math500": "math500", "math_full": "math_ext"}
ORIG_MIX = {"gsm8k": 1055 / 1455, "math": 400 / 1455}          # original test pool (prereg §2)
EXP3_MIN_N = json.loads((Path(__file__).resolve().parents[1] / "analysis" / "out" / "min_n.json").read_text())["4B-exp3"]["min_n"]
LV = np.array(P_LEVELS)
LW = (LV - LV.mean()) / ((LV - LV.mean()) ** 2).sum()


def jl(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8")]


class Boot:
    """Cluster bootstrap by item multiplicities: one shared set of resampled canonical problems for every statistic."""

    def __init__(self, clusters, n_boot=N_BOOT, seed=SEED):
        _, self.cid = np.unique(np.asarray(clusters), return_inverse=True)
        g = self.cid.max() + 1
        rng = np.random.default_rng(seed)
        # cluster draw counts per resample (small integers); mapped to items when used
        self.counts = np.stack([np.bincount(rng.integers(0, g, g), minlength=g) for _ in range(n_boot)]).astype(np.uint8)

    def ci(self, f, mask=None):
        """f(multiplicity vector restricted to mask) -> statistic; returns (estimate, lo, hi)."""
        m = np.ones(len(self.cid), int) if mask is None else mask.astype(int)
        est = f(m)
        vals = np.array([f(b[self.cid].astype(int) * m) for b in self.counts])
        vals = vals[np.isfinite(vals)]
        return est, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def wmean(v, m):
    return float((m * v).sum() / m.sum()) if m.sum() else float("nan")


def verbal_s(r):
    if r.get("p_yes") is None or r.get("p_no") is None:
        return None
    return math.log(max(r["p_yes"], 1e-300)) - math.log(max(r["p_no"], 1e-300))


def rescore(text, gold, fr):
    return bool(score_v2(text, gold, finish_reason=fr)["correct"])


def load(spec):
    gold, canon, strat = {}, {}, {}
    for f in ("data/processed/dnat.jsonl", "data/processed/ext/dnat_ext.jsonl", "data/processed/dinj.jsonl"):
        for r in jl(f):
            gold[r["item_id"]], canon[r["item_id"]] = r["gold"], r["canonical_id"]
            strat[r["item_id"]] = STRATUM.get(r["dataset"], r["dataset"])
    init = {r["item_id"]: r for f in spec["initial"] for r in jl(f) if r["split"] == spec["split"]}
    verbal = {r["item_id"]: r for f in spec["verbal"] for r in jl(f) if r["split"] == spec["split"]}
    return gold, canon, strat, init, verbal


def exp01(run, data, exp, gold, v2):
    """item -> {cond: correct}; with v2, labels from the second scorer on the same texts."""
    d = Path(run)
    gen = {g["prompt_key"]: g for g in jl(d / "generations.jsonl")} if v2 else {}
    out = defaultdict(dict)
    for r in jl(d / "rows.jsonl"):
        if r["exp"] != exp or r["data"] != data or r.get("stage") == 1:
            continue
        c = r["correct"]
        if v2 and r.get("prompt_key"):
            g = gen[r["prompt_key"]]
            c = rescore(g["text"], gold[r["item_id"]], g["finish_reason"])
        out[r["item_id"]][r["cond"]] = int(bool(c))
    return out


def probe_scores(spec):
    if spec.get("probe_scores"):
        return {r["item_id"]: r["p_correct"] for r in jl(spec["probe_scores"])}
    if spec.get("probe_npz") and spec.get("probe_acts"):
        from src.probe.train import score_frozen
        out = {}
        for a in spec["probe_acts"]:
            z = np.load(ROOT / a)
            out.update(zip(z["ids"].tolist(), score_frozen(ROOT / spec["probe_npz"], z["X"]).tolist()))
        return out
    return None


def category(kind, p_holm, est, lo, hi, margin):
    if p_holm is None:
        return "not run"
    if kind == "equivalence":
        if p_holm < ALPHA:
            return "supported"
        return "contradicted" if (lo > margin or hi < -margin) else "inconclusive"
    if kind == "two-sided":
        if p_holm < ALPHA:
            return f"supported ({'positive' if est > 0 else 'negative'})"
        return "not supported" if (-margin < lo and hi < margin) else "inconclusive"
    if p_holm < ALPHA:
        return "supported"
    return "not supported" if hi < margin else "inconclusive"


def analyse(spec, v2=False, drop_bad_a0=False, boot_n=N_BOOT):
    gold, canon, strat, init, verbal = load(spec)
    res, notes = {}, []

    # ---- D-nat Exp 1A (H1a, H1b, H1d, H2b)
    runs = [spec["exp01_orig"]] + ([spec["exp01_ext"]] if spec.get("exp01_ext") else [])
    if not spec.get("exp01_ext"):
        notes.append("expansion run absent: D-nat tests on the original pool (prereg §10)")
    items = {}
    for run in runs:
        items.update(exp01(run, "dnat", "1A", gold, v2))
    need = ["F1", "F2", "F0"] + [f"F3({p})" for p in P_LEVELS] + ["F5"] + [f"F3-perm[{k}]" for k in range(K_PERM)]
    ids = sorted(i for i, v in items.items() if all(c in v for c in need))
    if drop_bad_a0:
        ids = [i for i in ids if init[i]["status"] == "ok"]
    notes.append(f"D-nat items with every 1A condition: {len(ids)} of {len(items)}")
    a = lambda c: np.array([items[i][c] for i in ids], float)
    cl = np.array([canon[i] for i in ids])
    st = np.array([strat[i] for i in ids])
    c0 = np.array([bool(rescore(init[i]["a0"], gold[i], init[i]["finish_reason"]) if v2 else init[i]["correct"]) for i in ids])
    boot = Boot(cl, boot_n)
    d1a = a("F2") - a("F1")
    est, lo, hi = boot.ci(lambda m: wmean(d1a, m))
    res["H1a"] = {"n": len(ids), "est": est, "lo": lo, "hi": hi, "p": tost(a("F2"), a("F1"), DELTA, clusters=cl),
                  "kind": "equivalence", "margin": DELTA,
                  # reported next to the TOST (prereg §0): discordance and each arm's W->R and R->W
                  "context": {"discordance": float((a("F2") != a("F1")).mean()),
                              **{f"{c} W->R": float(a(c)[~c0].mean()) if (~c0).any() else None for c in ("F1", "F2")},
                              **{f"{c} R->W": float(1 - a(c)[c0].mean()) if c0.any() else None for c in ("F1", "F2")}}}
    Y = np.stack([a(f"F3({p})") for p in P_LEVELS], 1)
    slope = Y @ LW
    est, lo, hi = boot.ci(lambda m: 0.5 * wmean(slope, m))
    res["H1b"] = {"n": len(ids), "est": est, "lo": lo, "hi": hi,
                  "p": gee_slope(Y.ravel(), np.tile(LV, len(ids)), np.repeat(cl, 4)), "kind": "one-sided", "margin": DELTA / 2}
    d1d = a("F5") - np.mean([a(f"F3-perm[{k}]") for k in range(K_PERM)], 0)
    est, lo, hi = boot.ci(lambda m: wmean(d1d, m))
    res["H1d"] = {"n": len(ids), "est": est, "lo": lo, "hi": hi, "kind": "estimation"}

    # secondary D-nat estimands
    is_m = np.isin(st, ["math500", "math_ext"])
    def reweighted(v):
        return lambda m: ORIG_MIX["gsm8k"] * wmean(v, m * (st == "gsm8k")) + ORIG_MIX["math"] * wmean(v, m * is_m)
    sec = {}
    for h, v, scale in (("H1a", d1a, 1.0), ("H1b", slope, 0.5), ("H1d", d1d, 1.0)):
        sec[h] = {"original pool": boot.ci(lambda m, v=v: scale * wmean(v, m), np.isin(st, ["gsm8k", "math500"])),
                  "reweighted to the original mix": boot.ci(lambda m, v=v: scale * reweighted(v)(m))}
        for s in ("gsm8k", "math500", "math_ext"):
            if (st == s).any():
                sec[h][s] = boot.ci(lambda m, v=v: scale * wmean(v, m), st == s)
        if (st == "math_ext").any():   # stratum heterogeneity check (prereg §2), paired over the same resamples
            orig = np.isin(st, ["gsm8k", "math500"])
            sec[h]["math_ext − original pool"] = boot.ci(
                lambda m, v=v: scale * (wmean(v, m * (st == "math_ext")) - wmean(v, m * orig)))
    res["secondary"] = sec

    # ---- H2a and H2b (need probe scores)
    ps = probe_scores(spec)
    pool = sorted(i for i in init if strat[i] in ("gsm8k", "math500", "math_ext")
                  and (spec.get("exp01_ext") or strat[i] != "math_ext"))
    if drop_bad_a0:
        pool = [i for i in pool if init[i]["status"] == "ok"]
    if ps is None:
        res["H2a"] = {"kind": "one-sided", "p": None}
        notes.append("probe scores absent: H2a and H2b not run")
    else:
        h2 = [i for i in pool if i in ps and verbal_s(verbal.get(i, {})) is not None]
        notes.append(f"H2a items with probe and verbal scores: {len(h2)} of {len(pool)} (missing excluded from both, counted)")
        yy = np.array([bool(rescore(init[i]["a0"], gold[i], init[i]["finish_reason"]) if v2 else init[i]["correct"]) for i in h2])
        sp = np.array([ps[i] for i in h2])
        sv = np.array([verbal_s(verbal[i]) for i in h2])
        r = auroc_diff_boot(yy, sp, sv, clusters=[canon[i] for i in h2], n_boot=boot_n, seed=SEED)
        from p2_power_h2a import delong_paired
        res["H2a"] = {"n": len(h2), "est": r["diff"], "lo": r["lo"], "hi": r["hi"], "p": r["p_greater"],
                      "kind": "one-sided", "margin": DELTA_AUC, "auroc_probe": auroc(yy, sp), "auroc_verbal": auroc(yy, sv),
                      "delong_p": float(delong_paired(sp, sv, yy)[0])}
        h2st = np.array([strat[i] for i in h2])
        res["secondary"]["H2a"] = {}
        for name, msk in (("original pool", np.isin(h2st, ["gsm8k", "math500"])), ("math_ext", h2st == "math_ext")):
            if msk.any() and yy[msk].any() and not yy[msk].all():
                rr = auroc_diff_boot(yy[msk], sp[msk], sv[msk], clusters=[canon[i] for i, k in zip(h2, msk) if k],
                                     n_boot=min(boot_n, 2000), seed=SEED)
                res["secondary"]["H2a"][name] = (rr["diff"], rr["lo"], rr["hi"])
        # H2b: initially wrong items, top quintile of probe P(correct), recomputed in every resample
        pw = np.array([ps.get(i, np.nan) for i in ids])
        wrong = ~c0 & np.isfinite(pw)
        def h2b(m, col):
            mm = m * wrong
            if mm.sum() < 5:
                return float("nan")
            thr = np.quantile(np.repeat(pw, mm), 0.8)
            sel = mm * (pw >= thr)
            return wmean(a(col) - a("F0"), sel)
        for part, col in (("H2b(i) F2 - F0", "F2"), ("H2b(ii) F3(1.0) - F0", "F3(1.0)")):
            est, lo, hi = boot.ci(lambda m, col=col: h2b(m, col))
            res[part] = {"n": int(wrong.sum() // 5), "est": est, "lo": lo, "hi": hi, "kind": "estimation"}

    # ---- H3a (Exp 1B, D-inj, F0)
    if spec.get("exp01_1b"):
        b = exp01(spec["exp01_1b"], "dinj", "1B", gold, v2)
        cells = ["assistant/self/F0", "assistant/other/F0", "user/self/F0", "user/other/F0"]
        bid = sorted(i for i, v in b.items() if all(c in v for c in cells))
        C = np.array([[b[i][c] for c in cells] for i in bid], float)
        bcl = np.array([canon[i] for i in bid])
        bb = Boot(bcl, boot_n, SEED + 1)
        for h, d in (("H3a-role", (C[:, 2] + C[:, 3] - C[:, 0] - C[:, 1]) / 2),
                     ("H3a-label", (C[:, 1] + C[:, 3] - C[:, 0] - C[:, 2]) / 2)):
            est, lo, hi = bb.ci(lambda m, d=d: wmean(d, m))
            res[h] = {"n": len(bid), "est": est, "lo": lo, "hi": hi, "p": two_sided_paired(d, np.zeros_like(d), clusters=bcl),
                      "kind": "two-sided", "margin": DELTA}
    else:
        for h in ("H3a-role", "H3a-label"):
            res[h] = {"kind": "two-sided", "p": None}

    # ---- H3b (Exp 3); with v2, labels from the saved texts (finish reason from the text record, else from the row status)
    if spec.get("exp3_rows") and (not v2 or spec.get("exp3_texts")):
        e3, e3nat = defaultdict(dict), defaultdict(dict)
        rows3 = jl(spec["exp3_rows"])
        texts3 = {}
        if v2:
            for t in jl(spec["exp3_texts"]):
                texts3[(t["run"], t["item_id"])] = t
        for r in rows3:
            c = r["correct"]
            if v2:
                t = texts3[(r["run"], r["item_id"])]
                fr = t.get("finish_reason") or ("length" if r["status"] == "truncated" else "stop")
                c = rescore(t["text"], gold[r["item_id"]], fr)
            (e3 if r["data"] == "dinj" else e3nat)[r["item_id"]][r["run"]] = int(bool(c))
        res["exp3_arms"] = {arm: {"W->R D-inj": float(np.mean([v[arm] for v in e3.values() if arm in v])),
                                  "R->W D-nat": (float(1 - np.mean([v[arm] for v in e3nat.values() if arm in v]))
                                                 if any(arm in v for v in e3nat.values()) else None)}
                            for arm in sorted({x for v in e3.values() for x in v})}
        arms = [spec["exp3_main"]] + spec["exp3_random"]
        eid = sorted(i for i, v in e3.items() if all(x in v for x in arms))
        d = np.array([e3[i][spec["exp3_main"]] - np.mean([e3[i][x] for x in spec["exp3_random"]]) for i in eid])
        ecl = np.array([canon[i] for i in eid])
        est, lo, hi = Boot(ecl, boot_n, SEED + 2).ci(lambda m: wmean(d, m))
        res["H3b"] = {"n": len(eid), "est": est, "lo": lo, "hi": hi,
                      "p": one_sided_paired(d, np.zeros_like(d), 0.0, "greater", clusters=ecl), "kind": "one-sided", "margin": DELTA}
        # Chunked block (prereg §10): below min_n descriptive only (p = 1, slot kept); between min_n and the planned n a
        # pre-registered partial-sample analysis. Applies to the test block only.
        if spec["split"] == "test":
            mn, planned = EXP3_MIN_N, spec.get("exp3_planned_n", 1547)
            if len(eid) < mn:
                res["H3b"].update({"p": None, "status_note": f"below min_n {mn}: descriptive only (p = 1)"})
            elif len(eid) < planned:
                res["H3b"]["status_note"] = f"pre-registered partial-sample analysis ({len(eid)} of {planned})"
    else:
        res["H3b"] = {"kind": "one-sided", "p": None}

    # ---- Holm and categories
    praw = [res[h]["p"] if res[h].get("p") is not None else 1.0 for h in FAMILY]
    for h, ph in zip(FAMILY, holm(praw)):
        r = res[h]
        r["p_holm"] = float(ph) if r.get("p") is not None else None
        r["category"] = category(r["kind"], r["p_holm"], r.get("est"), r.get("lo"), r.get("hi"), r.get("margin"))
        if r.get("status_note") and r["p_holm"] is None:
            r["category"] = "descriptive only (" + r["status_note"] + ")"
        elif r.get("status_note"):
            r["category"] += " [" + r["status_note"] + "]"
    if res["H1b"]["category"] == "supported" and res["H1b"]["est"] < DELTA:
        res["H1b"]["category"] = "supported (effect below δ)"
    if res["H2a"].get("p") is not None:
        # The outcome follows the bootstrap; it is flagged when DeLong in its place would change significance (prereg §3.1).
        alt = [res["H2a"]["delong_p"] if h == "H2a" else x for h, x in zip(FAMILY, praw)]
        if (holm(praw)[FAMILY.index("H2a")] < ALPHA) != (holm(alt)[FAMILY.index("H2a")] < ALPHA):
            res["H2a"]["category"] += " [method-sensitive: DeLong disagrees]"
    strict = [h for h in FAMILY if h != "H1b"]
    res["strict_holm"] = dict(zip(strict, holm([res[h]["p"] if res[h].get("p") is not None else 1.0 for h in strict]).tolist()))
    res["notes"] = notes
    return res


def disagreement(spec):
    """Per-condition share of rows whose label changes under the second scorer v2 (D-nat 1A and D-inj 1B F0)."""
    gold = {r["item_id"]: r["gold"] for f in ("data/processed/dnat.jsonl", "data/processed/ext/dnat_ext.jsonl",
                                              "data/processed/dinj.jsonl") for r in jl(f)}
    out = defaultdict(lambda: [0, 0])
    for run, data, exp in ((spec["exp01_orig"], "dnat", "1A"), (spec.get("exp01_ext"), "dnat", "1A"),
                           (spec.get("exp01_1b"), "dinj", "1B")):
        if not run:
            continue
        gen = {g["prompt_key"]: g for g in jl(Path(run) / "generations.jsonl")}
        for r in jl(Path(run) / "rows.jsonl"):
            if r["exp"] == exp and r["data"] == data and r.get("prompt_key") and r.get("stage") != 1:
                g = gen[r["prompt_key"]]
                out[f"{exp}/{data}/{r['cond']}"][0] += 1
                out[f"{exp}/{data}/{r['cond']}"][1] += bool(r["correct"]) != rescore(g["text"], gold[r["item_id"]], g["finish_reason"])
    return {k: v[1] / v[0] for k, v in sorted(out.items())}


def pp(x, scale=100):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{scale * x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    spec = json.loads((ROOT / a.spec).read_text(encoding="utf-8"))
    main_res = analyse(spec, boot_n=a.n_boot)
    sens_v2 = analyse(spec, v2=True, boot_n=a.n_boot)
    sens_a0 = analyse(spec, drop_bad_a0=True, boot_n=a.n_boot)
    dis = disagreement(spec)
    dry = spec["split"] != "test"
    lines = [f"# Confirmatory analysis, {spec['model']} (generated by analysis/p6_confirmatory.py)", "",
             (f"**DRY RUN on {spec['split']} data — a test of the code path, not a result.** " if dry else "") +
             f"Spec `{a.spec}`. Estimates in pp (H2a: AUROC × 100); 95 % cluster-bootstrap CIs ({a.n_boot} resamples, not "
             "multiplicity-adjusted); Holm over the family of 6. Rules: script docstring and prereg §3–§5.", ""]
    lines += [f"- {n}" for n in main_res["notes"]] + [""]
    lines += ["## Family", "", "| Hypothesis | n | Estimate | 95 % CI | p | p (Holm) | Outcome |", "|---|---|---|---|---|---|---|"]
    for h in FAMILY:
        r = main_res[h]
        lines.append(f"| {h} | {r.get('n', '—')} | {pp(r.get('est'))} | [{pp(r.get('lo'))}, {pp(r.get('hi'))}] | "
                     f"{'—' if r.get('p') is None else f'{r['p']:.4g}'} | {'—' if r.get('p_holm') is None else f'{r['p_holm']:.4g}'} | "
                     f"{r['category']} |")
    if main_res["H2a"].get("p") is not None:
        r = main_res["H2a"]
        lines += ["", f"H2a: AUROC probe {r['auroc_probe']:.4f}, verbal {r['auroc_verbal']:.4f}; DeLong one-sided p "
                      f"{r['delong_p']:.4g} (check)."]
    ctx = main_res["H1a"].get("context", {})
    lines += ["", "H1a context (prereg §0): " + ", ".join(f"{k} {pp(v)} %" for k, v in ctx.items()) + "."]
    if "exp3_arms" in main_res:
        lines += ["", "Exp 3 arms (W→R on D-inj, R→W on the D-nat R→W items; secondary arms are exploratory, compared "
                  "with unsteered only):", "", "| Arm | W→R D-inj % | R→W D-nat % |", "|---|---|---|"]
        lines += [f"| {k} | {pp(v['W->R D-inj'])} | {pp(v['R->W D-nat'])} |" for k, v in main_res["exp3_arms"].items()]
    lines += ["", "Strict reading of design §7.4 (H1b estimation-only), Holm over the other five: "
              + ", ".join(f"{h} {p:.4g}" for h, p in main_res["strict_holm"].items()) + ".", "",
              "## Estimation-only", "", "| Quantity | n | Estimate | 95 % CI |", "|---|---|---|---|"]
    for h in ("H1d", "H2b(i) F2 - F0", "H2b(ii) F3(1.0) - F0"):
        if h in main_res:
            r = main_res[h]
            lines.append(f"| {h} | {r['n']} | {pp(r['est'])} | [{pp(r['lo'])}, {pp(r['hi'])}] |")
    lines += ["", "## Secondary D-nat estimands (descriptive)", "", "| Hypothesis | Estimand | Estimate | 95 % CI |", "|---|---|---|---|"]
    for h, d in main_res["secondary"].items():
        for name, (e, lo, hi) in d.items():
            lines.append(f"| {h} | {name} | {pp(e)} | [{pp(lo)}, {pp(hi)}] |")
    lines += ["", "## Sensitivity analyses", "", "| Hypothesis | Main estimate | v2 labels: estimate | v2: p | a0 ok only: estimate | a0 ok only: p |",
              "|---|---|---|---|---|---|"]
    for h in FAMILY:
        r, s, t = main_res[h], sens_v2[h], sens_a0[h]
        f = lambda x: "—" if x.get("p") is None else f"{x['p']:.4g}"
        lines.append(f"| {h} | {pp(r.get('est'))} | {pp(s.get('est'))} | {f(s)} | {pp(t.get('est'))} | {f(t)} |")
    lines += ["", "## Scorer disagreement per condition (main vs v2, share of rows)", "", "| Condition | Share % |", "|---|---|"]
    lines += [f"| {k} | {pp(v)} |" for k, v in dis.items()]
    out = {"main": main_res, "v2": sens_v2, "a0_ok": sens_a0, "disagreement": dis, "spec": spec}
    (ROOT / "analysis" / "out" / f"{a.out}.json").write_text(json.dumps(out, indent=1, default=float))
    (ROOT / "reports" / f"{a.out}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
