"""Amendment 6 of the ARR plan (docs/planning/arr_round/plan.md): descriptive replays requested by the independent
review of the ACL v2 draft. Stored outputs only; no generation, no model, no re-extraction of any output.

Everything here is post hoc and descriptive: outside the §5 family, nothing is called significant. Intervals are the
paired cluster bootstrap of §0 (4,000 replicates, seed 20261110, selections fixed).

R1 answerless a0: (a) their share in each signal's selections at 8/10/20 %; (b) ANSWERLESS->THINK; (c) answerless
   first, then the signal among answered items, for THINK and F2; (d) the answered population: THINK/F2 ALL, selective
   THINK/F2, reach shares, AUROC.
R2 DETECT-oracle bounds at 10/20 % for F0, F2, F2-scratch, s1 and THINK.
R3 (a) HYBRID-margin-THINK; (b) ADAPTIVE-V8 on the lowest-margin items; (c) one-sample agreement as an escalation signal.
R5 development-split replication of family members 1-4 (math 8B and T8B; MMLU-Pro 4B and T8B).
R6 single-sample accuracy by sample index.
(R4 is tabulated from stored outputs by analysis/arr/acl_v2_tex.py.)

Usage: python -B analysis/arr/v2_review.py
-> analysis/out/arr/v2_review.json, v2_review_manifest.json; reports/arr_v2_review.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1a2_replays as A  # noqa: E402
import a3_dev_test as A3  # noqa: E402
import b1_samples as B1  # noqa: E402
from common import (J1, MODELS_MATH, MODELS_MC, OPS, OUT, REPO, Boot, Equivalence, dump, groups_of, guard_check,  # noqa: E402
                    manifest, pin, plan_core_hash, plan_hash, prefix_groups, rows, rows_gz, sha, tie_key, unique, winner)

SETTINGS = [f"math {m}" for m in MODELS_MATH] + [f"MMLU-Pro {m}" for m in MODELS_MC]
THINK = ("math 4B", "math 8B", "MMLU-Pro 4B")
SIGNALS = ("verbal", "probe", "length", "agreement", "margin")
GRID = range(0, 101)
FRACS = (0.5, 0.8)


# ----------------------------------------------------------------------------- loading
def load():
    a0 = json.loads(pin(OUT / "a0_common_labels.json").read_text(encoding="utf-8"))
    assert a0["plan_core_sha256"] == plan_core_hash() and not a0["guards"]
    labels = list(rows_gz(pin(OUT / "math_common_labels.jsonl.gz")))
    table = {}
    for r in rows(pin(J1 / "artifacts/math/item_evaluations.jsonl")):
        table.setdefault(r["model"], {}).setdefault(r["condition"], {})[r["item_id"]] = r
    math = A.load_math(table, labels)
    return math


def think_arrays(pops):
    man = json.loads((OUT / "c1_think_manifest.json").read_text(encoding="utf-8"))
    path = OUT / "c1_labels.jsonl.gz"
    assert sha(path) == man["outputs"]["c1_labels.jsonl.gz"], "c1_labels.jsonl.gz differs from its manifest"
    lab = {}
    for e in rows_gz(pin(path)):
        if e["source"] == "C1":
            lab.setdefault(e["setting"], {})[e["item_id"]] = e
    T = {}
    for s in THINK:
        P, L = pops[s], lab[s]
        assert set(L) == set(P["ids"]), s
        T[s] = (np.array([L[i]["correct"] for i in P["ids"]], dtype=float),
                np.array([L[i]["out_tokens"] for i in P["ids"]], dtype=float),
                np.array([L[i]["prompt_tokens"] for i in P["ids"]], dtype=float),
                np.array([L[i]["status"] == "ok" for i in P["ids"]]))
    return T


def operators(P, T=None):
    ops = {op: (P["ops"][op]["y1"], P["ops"][op]["gen"], P["ops"][op]["prompt"], P["ops"][op]["answered"]) for op in OPS}
    S = P["samples"]
    ops["s1"] = (S["correct"][:, 0], S["tok"][:, 0], S["sprompt"], S["valid"][:, 0])
    if T is not None:
        ops["THINK"] = T
    return ops


def acq(P, sig):
    if sig in ("agreement", "margin"):
        return P["samples"]["tok"].sum(axis=1)
    if sig == "one-sample agreement":
        return P["samples"]["tok"][:, 0]
    return np.zeros(P["n"])


# ----------------------------------------------------------------------------- generic pieces
def order_by(P, score, answerless_first=False):
    s = np.asarray(score, dtype=float)
    if answerless_first:
        s = np.where(P["a0_answered"], s, -np.inf)
    return np.lexsort((P["tie"], s))


def take(P, order, share):
    m = round(P["n"] * share)
    sel = np.zeros(P["n"], bool)
    sel[order[:m]] = True
    return sel


def summary(P, op, sel, boot, acq_vec=None):
    y1, gen, prompt, _ = op
    y = np.where(sel, y1, P["y0"])
    a = np.zeros(P["n"]) if acq_vec is None else acq_vec
    return {"delta": boot.ci(y - P["y0"]), "generated_tokens_per_item": float((a + sel * gen).mean()),
            "prompt_tokens_per_item": float((sel * prompt).mean()), "selected_n": int(sel.sum())}


def curve(P, op, order, acq_vec=None):
    y1, gen, _, _ = op
    n = P["n"]
    d = np.r_[0, np.cumsum((y1 - P["y0"])[order])]
    g = np.r_[0, np.cumsum(gen[order])]
    a = 0.0 if acq_vec is None else float(np.mean(acq_vec))
    return [{"share_pct": k, "delta_pp": 100 * float(d[round(n * k / 100)]) / n,
             "generated_tokens_per_item": a + float(g[round(n * k / 100)]) / n} for k in GRID]


def reach(cv, gain_pp):
    out = {}
    for f in FRACS:
        hit = None if gain_pp <= 0 else next((pt for pt in cv if pt["delta_pp"] >= f * gain_pp - 1e-9), None)
        out[f"{round(100 * f)}%"] = None if hit is None else dict(hit)
    return out


def auroc(score, y):
    """P(score of a correct a0 > score of a wrong one), equal scores counted 1/2 (higher = more likely correct)."""
    score, y = np.asarray(score, dtype=float), np.asarray(y, dtype=float)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    ranks = (start + (cnt + 1) / 2)[inv]
    n1, n0 = float(y.sum()), float(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def subset(P, mask):
    """The population restricted to mask (arrays indexed, lists filtered)."""
    idx = np.flatnonzero(mask)
    Q = {k: v for k, v in P.items() if k not in ("ids", "canonical", "source", "tie", "y0", "a0_answered", "sig", "ops",
                                                     "samples", "a0_pred", "a0_status", "a0_prompt", "y0h")}
    Q["ids"] = [P["ids"][j] for j in idx]
    Q["canonical"] = [P["canonical"][j] for j in idx]
    Q["n"] = len(idx)
    Q["tie"], Q["y0"], Q["a0_answered"] = P["tie"][idx], P["y0"][idx], P["a0_answered"][idx]
    Q["sig"] = {k: v[idx] for k, v in P["sig"].items()}
    Q["ops"] = {c: {k: v[idx] for k, v in o.items()} for c, o in P["ops"].items()}
    S = P["samples"]
    Q["samples"] = {"tok": S["tok"][idx], "correct": S["correct"][idx], "valid": S["valid"][idx],
                    "sprompt": S["sprompt"][idx]}
    return Q, idx


# ----------------------------------------------------------------------------- R1-R3, R6 on test
def r1a(P):
    un = ~P["a0_answered"]
    out = {"base_pct": 100 * float(un.mean()), "answerless_n": int(un.sum()), "by_signal": {}}
    for sig in SIGNALS:
        o = order_by(P, P["sig"][sig])
        out["by_signal"][sig] = {}
        for share in (0.08, 0.10, 0.20):
            sel = take(P, o, share)
            out["by_signal"][sig][f"{round(share * 100)}"] = {
                "answerless_pct_of_selected": 100 * float(un[sel].mean()),
                "answerless_covered_pct": 100 * float((un & sel).sum() / max(un.sum(), 1))}
    return out


def r1c(P, op, boot, gain_pp=None):
    out = {}
    for sig in SIGNALS:
        o = order_by(P, P["sig"][sig], answerless_first=True)
        a = acq(P, sig)
        cv = curve(P, op, o, a)
        e = {f"{round(q * 100)}": summary(P, op, take(P, o, q), boot, a) for q in (0.10, 0.20)}
        e["curve"] = cv
        if gain_pp is not None:
            e["reach"] = reach(cv, gain_pp)
        out[sig] = e
    return out


def r1d(P, T, boot_unused):
    mask = P["a0_answered"]
    Q, idx = subset(P, mask)
    boot = Boot(Q["canonical"])
    ops = {"F2": tuple(v[idx] for v in (P["ops"]["F2"]["y1"], P["ops"]["F2"]["gen"], P["ops"]["F2"]["prompt"], P["ops"]["F2"]["answered"]))}
    if T is not None:
        ops["THINK"] = tuple(v[idx] for v in T)
    out = {"n": Q["n"], "keep_acc_pct": 100 * float(Q["y0"].mean()),
           "auroc": {sig: auroc(Q["sig"][sig], Q["y0"]) for sig in SIGNALS}, "operators": {}}
    for name, op in ops.items():
        allsel = np.ones(Q["n"], bool)
        e = {"ALL": summary(Q, op, allsel, boot)}
        gain = e["ALL"]["delta"]["estimate"]
        for sig in SIGNALS:
            o = order_by(Q, Q["sig"][sig])
            a = acq(Q, sig)
            e[sig] = {f"{round(q * 100)}": summary(Q, op, take(Q, o, q), boot, a) for q in (0.10, 0.20)}
            if name == "THINK":
                e[sig]["reach"] = reach(curve(Q, op, o, a), gain)
        out["operators"][name] = e
    return out


def r2(P, ops, boot):
    out = {}
    o = order_by(P, P["y0"])   # wrong a0 first, then tie order: the DETECT-oracle
    for name, op in ops.items():
        out[name] = {f"{round(q * 100)}": summary(P, op, take(P, o, q), boot) for q in (0.10, 0.20)}
    return out


def r3a(P, T, pols, boot):
    S, v5 = P["samples"], pols["V5"]
    tok5, pr5 = S["tok"].sum(axis=1), 5 * S["sprompt"]
    o = order_by(P, P["sig"]["margin"])
    think_cv = curve(P, T, order_by(P, P["sig"]["verbal"]))
    out = {}
    for q in (0.10, 0.20):
        sel = take(P, o, q)
        y = np.where(sel, T[0], v5.y)
        cost = float((tok5 + sel * T[1]).mean())
        pt = min(think_cv, key=lambda p: (abs(p["generated_tokens_per_item"] - cost), p["share_pct"]))
        sel_v = take(P, order_by(P, P["sig"]["verbal"]), pt["share_pct"] / 100)
        out[f"{round(q * 100)}"] = {
            "delta": boot.ci(y - P["y0"]), "vs_V5": boot.ci(y - v5.y), "generated_tokens_per_item": cost,
            "prompt_tokens_per_item": float((pr5 + sel * T[2]).mean()), "selected_n": int(sel.sum()),
            "think_by_verbal_nearest": {"share_pct": pt["share_pct"], "generated_tokens_per_item": pt["generated_tokens_per_item"],
                                        "delta": boot.ci(np.where(sel_v, T[0], P["y0"]) - P["y0"])}}
    return out


def r3b(P, k8, boot):
    v5, v8 = k8["V5"], k8["V8"]
    o = order_by(P, P["sig"]["margin"])
    out = {}
    for q in (0.10, 0.20):
        sel = take(P, o, q)
        y = np.where(sel, v8.y, v5.y)
        out[f"{round(q * 100)}"] = {"delta": boot.ci(y - P["y0"]), "vs_V5": boot.ci(y - v5.y),
                                    "generated_tokens_per_item": float(np.where(sel, v8.gen, v5.gen).mean()),
                                    "selected_n": int(sel.sum())}
    return out


def one_sample_agreement(P):
    """1 if a0 and s1 are both valid and in one group (GATE's first-step rule), else 0."""
    return np.array([1.0 if any(0 in x and 1 in x for x in g) else 0.0 for g in P["samples"]["groups_a0"]])


def r3c(P, T, boot, gain_pp):
    agree1 = one_sample_agreement(P)
    o = np.lexsort((P["tie"], P["sig"]["verbal"], agree1))
    a = acq(P, "one-sample agreement")
    cv = curve(P, T, o, a)
    out = {f"{round(q * 100)}": summary(P, T, take(P, o, q), boot, a) for q in (0.10, 0.20)}
    out["disagree_pct"] = 100 * float((agree1 == 0).mean())
    out["reach"] = reach(cv, gain_pp)
    out["curve"] = cv
    return out


def r6(P, S8=None):
    C = S8["correct"] if S8 is not None else P["samples"]["correct"]
    acc = [100 * float(C[:, k].mean()) for k in range(C.shape[1])]
    return {"single_sample_acc_pct": acc, "min": min(acc), "max": max(acc), "keep_pct": 100 * float(P["y0"].mean())}


# ----------------------------------------------------------------------------- R5 on the development split
def dev_math(model, eq):
    """Math 8B / T8B development population from the A3-after-B5 labels (a0, operators, five samples, groups)."""
    man = json.loads((OUT / "a3_b5_manifest.json").read_text(encoding="utf-8"))
    path = OUT / "a3_b5_labels.jsonl.gz"
    assert sha(path) == man["outputs"]["a3_b5_labels.jsonl.gz"], "a3_b5_labels.jsonl.gz differs from its manifest"
    recs = sorted((r for r in rows_gz(pin(path)) if r["model"] == model), key=lambda r: r["item_id"])
    n = len(recs)
    P = {"setting": f"math {model}", "task": "math", "model": model, "ids": [r["item_id"] for r in recs], "n": n,
         "canonical": [r["canonical_id"] for r in recs], "tie": np.array([tie_key(r["item_id"]) for r in recs], dtype=float),
         "y0": np.array([r["initial"]["correct"] for r in recs], dtype=float),
         "a0_answered": np.array([r["initial"]["status"] == "ok" for r in recs]),
         "sig": {"agreement": np.array([r["agreement"] for r in recs], dtype=float)}, "ops": {}}
    for c in OPS:
        P["ops"][c] = {"y1": np.array([r[c]["correct"] for r in recs], dtype=float),
                       "gen": np.array([r[c]["gen"] for r in recs], dtype=float),
                       "prompt": np.array([r[c]["prompt"] for r in recs], dtype=float),
                       "answered": np.array([r[c]["status"] == "ok" for r in recs])}
    S = {"valid": np.zeros((n, 5), bool), "correct": np.zeros((n, 5)), "tok": np.zeros((n, 5)), "sprompt": np.zeros(n),
         "groups": [], "groups_a0": []}
    for j, r in enumerate(recs):
        cands = r["samples"]
        assert len(cands) == 5
        allc = [{"pred": r["initial"]["pred"], "status": r["initial"]["status"]}] + cands
        cache = {}

        def match(a, b, allc=allc, cache=cache):
            key = (min(a, b), max(a, b))
            if key not in cache:
                cache[key] = eq.symmetric(allc[a]["pred"], allc[b]["pred"])
            return cache[key]
        g = groups_of(cands, lambda a, b: match(a + 1, b + 1))
        assert g == r["groups"], ("stored development groups", model, r["item_id"])
        S["groups"].append(g)
        S["groups_a0"].append(groups_of(allc, match))
        for k, c in enumerate(cands):
            S["valid"][j, k] = c["status"] == "ok"
            S["correct"][j, k] = float(bool(c["correct"]))
            S["tok"][j, k] = c["out_tokens"]
    P["samples"] = S
    A.finish_votes(P)
    return P


def dev_mc(model):
    P = A3.mc_dev(model)
    S, n = P["samples"], P["n"]
    S["sprompt"] = np.zeros(n)
    # groups over (a0, samples) by letter equality, as A1 builds them for the test pool
    base = REPO / "results/v2/fu-dev" / model
    init = unique(r for r in rows(pin(base / f"fu-dev-mc-initial-dev-{model}-ef3e485" / "initial.jsonl")) if r["split"] == "dev")
    cons = unique(r for r in rows(pin(base / f"fu-dev-mc-consist-dev-{model}-ef3e485" / "consistency.jsonl")) if r["split"] == "dev")
    meta = {r["item_id"]: r for r in rows(pin(REPO / "data/processed/mmlupro/items.jsonl")) if r["item_id"] in init}

    def ok(p, i):
        return isinstance(p, str) and len(p) == 1 and "A" <= p < chr(65 + meta[i]["n_options"])
    S["groups_a0"] = []
    for i in P["ids"]:
        allc = [{"pred": init[i]["pred"], "status": "ok" if ok(init[i]["pred"], i) else "invalid"}] + \
               [{"pred": s["pred"], "status": "ok" if ok(s["pred"], i) else "invalid"} for s in cons[i]["samples"]]
        S["groups_a0"].append(groups_of(allc, lambda a, b, c=allc: c[a]["pred"] == c[b]["pred"]))
    A.finish_votes(P)
    return P


def r5(P):
    pols = A.policies(P)
    boot = Boot(P["canonical"])
    out = {"n": P["n"], "keep_acc_pct": 100 * float(P["y0"].mean()), "members": {}}
    for member, (a, b) in A.FAMILY.items():
        out["members"][str(member)] = {"a": a, "b": b, "difference": boot.ci(pols[a].y - pols[b].y),
                                       "token_difference": float(pols[a].gen.mean() - pols[b].gen.mean())}
    return out


# ----------------------------------------------------------------------------- report
def f(ci):
    return f"{ci['estimate']:+.2f} [{ci['lo']:+.2f}, {ci['hi']:+.2f}]"


def report(res):
    L = ["# ARR round, Amendment 6: descriptive replays requested by the review of the ACL v2 draft", "",
         "Generated by `analysis/arr/v2_review.py` (plan Amendment 6). Post hoc and descriptive: outside the §5 family; "
         "nothing here is called significant. Δ in pp vs KEEP with paired 95 % cluster-bootstrap intervals (4,000 "
         "replicates, seed 20261110); tokens are generated tokens per item after the shared a0.", ""]
    L += ["## R1a. Answerless a0 in each signal's selection", "",
          "| Setting | Answerless (pool) | Signal | 8 %: share of S / of answerless covered | 10 % | 20 % |", "|---|---:|---|---:|---:|---:|"]
    for s, e in res["r1a"].items():
        for sig, by in e["by_signal"].items():
            cells = [f"{by[k]['answerless_pct_of_selected']:.1f} / {by[k]['answerless_covered_pct']:.0f}" for k in ("8", "10", "20")]
            L.append(f"| {s} | {e['base_pct']:.1f} % | {sig} | " + " | ".join(cells) + " |")
    L += ["", "## R1b. ANSWERLESS->THINK", "", "| Setting | Δ vs KEEP | Gen. tok/item |", "|---|---:|---:|"]
    for s, e in res["r1b"].items():
        L.append(f"| {s} | {f(e['delta'])} | {e['generated_tokens_per_item']:.0f} |")
    L += ["", "## R1c. Answerless first, then the signal among answered items", "",
          "| Setting | Operator | Signal | 10 % | 20 % | 50 % of THINK ALL: share (tok) | 80 % |", "|---|---|---|---:|---:|---:|---:|"]
    for s, ops in res["r1c"].items():
        for op, sigs in ops.items():
            for sig, e in sigs.items():
                rr = e.get("reach", {})
                cells = [f"{f(e[k]['delta'])} ({e[k]['generated_tokens_per_item']:.0f})" for k in ("10", "20")]
                rcs = [("--" if not rr.get(k) else f"{rr[k]['share_pct']} % ({rr[k]['generated_tokens_per_item']:.0f})") for k in ("50%", "80%")] if rr else ["", ""]
                L.append(f"| {s} | {op} | {sig} | " + " | ".join(cells + rcs) + " |")
    L += ["", "## R1d. The answered population (a0 has a usable answer)", ""]
    for s, e in res["r1d"].items():
        L.append(f"**{s}** (n = {e['n']}, KEEP {e['keep_acc_pct']:.2f} %). AUROC: " +
                 ", ".join(f"{k} {v:.3f}" for k, v in e["auroc"].items()))
        L += ["", "| Operator | Signal | 10 % | 20 % | 50 % of ALL: share (tok) | 80 % |", "|---|---|---:|---:|---:|---:|"]
        for op, o in e["operators"].items():
            L.append(f"| {op} | ALL | {f(o['ALL']['delta'])} ({o['ALL']['generated_tokens_per_item']:.0f}) | | | |")
            for sig in SIGNALS:
                x = o[sig]
                rr = x.get("reach")
                rcs = [("--" if not rr or not rr.get(k) else f"{rr[k]['share_pct']} % ({rr[k]['generated_tokens_per_item']:.0f})") for k in ("50%", "80%")]
                L.append(f"| {op} | {sig} | {f(x['10']['delta'])} ({x['10']['generated_tokens_per_item']:.0f}) | "
                         f"{f(x['20']['delta'])} ({x['20']['generated_tokens_per_item']:.0f}) | " + " | ".join(rcs) + " |")
        L.append("")
    L += ["## R2. DETECT-oracle bounds (gold used; bounds, not policies)", "",
          "| Setting | Operator | 10 % | 20 % |", "|---|---|---:|---:|"]
    for s, ops in res["r2"].items():
        for op, e in ops.items():
            L.append(f"| {s} | {op} | {f(e['10']['delta'])} ({e['10']['generated_tokens_per_item']:.0f}) | "
                     f"{f(e['20']['delta'])} ({e['20']['generated_tokens_per_item']:.0f}) |")
    L += ["", "## R3a. HYBRID-margin-THINK (V5 elsewhere)", "",
          "| Setting | Share | Δ vs KEEP | − V5 | Gen. tok/item | THINK by verbal at nearest cost: share, Δ (tok) |", "|---|---:|---:|---:|---:|---:|"]
    for s, e in res["r3a"].items():
        for k in ("10", "20"):
            x = e[k]
            nb = x["think_by_verbal_nearest"]
            L.append(f"| {s} | {k} % | {f(x['delta'])} | {f(x['vs_V5'])} | {x['generated_tokens_per_item']:.0f} | "
                     f"{nb['share_pct']} %, {f(nb['delta'])} ({nb['generated_tokens_per_item']:.0f}) |")
    L += ["", "## R3b. ADAPTIVE-V8 on the lowest-margin items (V5 elsewhere)", "",
          "| Setting | Share | Δ vs KEEP | − V5 | Gen. tok/item |", "|---|---:|---:|---:|---:|"]
    for s, e in res["r3b"].items():
        for k in ("10", "20"):
            x = e[k]
            L.append(f"| {s} | {k} % | {f(x['delta'])} | {f(x['vs_V5'])} | {x['generated_tokens_per_item']:.0f} |")
    L += ["", "## R3c. One-sample agreement as an escalation signal (THINK)", "",
          "| Setting | a0 and s1 disagree | 10 % | 20 % | 50 % of THINK ALL: share (tok) | 80 % |", "|---|---:|---:|---:|---:|---:|"]
    for s, e in res["r3c"].items():
        rr = e["reach"]
        rcs = [("--" if not rr.get(k) else f"{rr[k]['share_pct']} % ({rr[k]['generated_tokens_per_item']:.0f})") for k in ("50%", "80%")]
        L.append(f"| {s} | {e['disagree_pct']:.1f} % | {f(e['10']['delta'])} ({e['10']['generated_tokens_per_item']:.0f}) | "
                 f"{f(e['20']['delta'])} ({e['20']['generated_tokens_per_item']:.0f}) | " + " | ".join(rcs) + " |")
    L += ["", "## R5. Family members 1-4 on the development split (descriptive replication)", "",
          "| Setting | n | Member | Contrast | Difference (dev) |", "|---|---:|---:|---|---:|"]
    for s, e in res["r5"].items():
        for mem, x in e["members"].items():
            L.append(f"| {s} | {e['n']} | {mem} | {x['a']} − {x['b']} | {f(x['difference'])} |")
    L += ["", "## R6. Single-sample accuracy by sample index (an alternative V1 per index)", "",
          "| Setting | KEEP % | Per index (%) | Range |", "|---|---:|---|---:|"]
    for s, e in res["r6"].items():
        L.append(f"| {s} | {e['keep_pct']:.2f} | " + ", ".join(f"{a:.2f}" for a in e["single_sample_acc_pct"]) +
                 f" | {e['min']:.2f}–{e['max']:.2f} |")
    L += ["", "Checks: " + "; ".join(res["checks"]), ""]
    (REPO / "reports/arr_v2_review.md").write_text("\n".join(L), encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- main
def main():
    res = {"plan_sha256_lf": plan_hash(), "plan_core_sha256": plan_core_hash(), "r1a": {}, "r1b": {}, "r1c": {},
           "r1d": {}, "r2": {}, "r3a": {}, "r3b": {}, "r3c": {}, "r5": {}, "r6": {}, "checks": []}
    a1 = json.loads(pin(OUT / "a1_policies.json").read_text(encoding="utf-8"))
    b1 = json.loads(pin(OUT / "b1_policies.json").read_text(encoding="utf-8"))
    c1 = json.loads(pin(OUT / "c1_think.json").read_text(encoding="utf-8"))
    au = {(x["setting"], x["signal"]): x["pooled"] for x in
          json.loads(pin(OUT / "a2_auroc.json").read_text(encoding="utf-8")) + json.loads(pin(OUT / "b1_auroc.json").read_text(encoding="utf-8"))
          if x["labels"] == "common"}
    math = load()
    eq = Equivalence()
    try:
        S8 = {}
        for m in MODELS_MATH:
            S8[m] = B1.attach(math[m], m, eq)
            assert S8[m] is not None, m
        pops = {f"math {m}": math[m] for m in MODELS_MATH}
        pops.update({f"MMLU-Pro {m}": A.load_mc(m) for m in MODELS_MC})
        T = think_arrays(pops)
        dev = {f"math {m}": dev_math(m, eq) for m in ("8B", "T8B")}
        dev.update({f"MMLU-Pro {m}": dev_mc(m) for m in MODELS_MC})
        guards = list(eq.guards)
    finally:
        eq.close()
    res["scorer_guards"] = guard_check(guards)

    for s in SETTINGS:
        P = pops[s]
        boot = Boot(P["canonical"])
        pols = A.policies(P)
        ref = (b1["a1"][s]["policies"] if s in b1["a1"] else a1["settings"][s]["policies"])
        for name in ("KEEP", "F2 ALL", "V5", "V1", "HYBRID-margin 20%"):
            assert abs(ref[name]["accuracy_pct"] - 100 * pols[name].y.mean()) < 1e-9, (s, name)
        for sig in SIGNALS:
            assert abs(auroc(P["sig"][sig], P["y0"]) - au[(s, sig)]) < 1e-9, ("AUROC parity", s, sig)
        res["checks"].append(f"{s}: KEEP, F2 ALL, V1, V5 and HYBRID-margin 20% equal the stored A1/B1 policies; "
                             "every signal's pooled AUROC equals A2/B1")
        Tz = T.get(s)
        ops = operators(P, Tz)
        res["r1a"][s] = r1a(P)
        res["r2"][s] = r2(P, ops, boot)
        a1ref = a1["settings"][s]["policies"]
        for q in ("10%", "20%"):
            assert abs(a1ref[f"REF DETECT->F2 {q}"]["accuracy_pct"] -
                       (P["y0"].mean() + res["r2"][s]["F2"][q[:-1]]["delta"]["estimate"] / 100) * 100) < 1e-9, (s, q)
        res["r1c"][s] = {"F2": r1c(P, ops["F2"], boot)}
        if Tz is not None:
            c = c1["settings"][s]
            gain = c["reach"]["think_all_gain_pp"]
            assert abs(gain - 100 * float((Tz[0] - P["y0"]).mean())) < 1e-9, s
            for q in ("10%", "20%"):
                ref_t = c["policies"][f"REF DETECT->THINK {q}"]["delta_vs_keep"]["estimate"]
                assert abs(ref_t - res["r2"][s]["THINK"][q[:-1]]["delta"]["estimate"]) < 1e-9, (s, q)
            un = ~P["a0_answered"]
            res["r1b"][s] = summary(P, Tz, un, boot)
            res["r1c"][s]["THINK"] = r1c(P, Tz, boot, gain)
            res["r3a"][s] = r3a(P, Tz, pols, boot)
            res["r3c"][s] = r3c(P, Tz, boot, gain)
            res["checks"].append(f"{s}: THINK ALL's gain and the DETECT-oracle for THINK and F2 equal C1/A1")
        res["r1d"][s] = r1d(P, Tz, boot)
        if s.startswith("math"):
            k8, _ = B1.k8_policies(P, S8[s.split()[1]])
            ref8 = b1["k8"][s]["policies"]
            for name in ("V5", "V8"):
                assert abs(ref8[name]["accuracy_pct"] - 100 * k8[name].y.mean()) < 1e-9, (s, name)
            res["r3b"][s] = r3b(P, k8, boot)
            res["r6"][s] = r6(P, S8[s.split()[1]])
        else:
            res["r6"][s] = r6(P)
    for s, P in dev.items():
        res["r5"][s] = r5(P)
    res["checks"].append("development: the stored groups over the five samples are recomputed and equal "
                         "(math 8B, T8B); MMLU-Pro groups by letter")

    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    res = clean(res)
    dump(OUT / "v2_review.json", res)
    report(res)
    manifest("v2_review", {"outputs": {"v2_review.json": sha(OUT / "v2_review.json")}})
    print("done:", {k: len(v) for k, v in res.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
