"""A1 + A2 of the ARR plan (docs/planning/arr_round/plan.md): comparator policies, decompositions, Eq. (2) terms
and stratified AUROC. Stored outputs only; no generation and no model.

Settings: math Qwen3-1.7B, 4B, 8B, T8B (common gold-blind labels from A0; historical copies kept) and MMLU-Pro
Qwen3-4B, T8B (frozen letter scorer). Five stored samples exist for 4B math and both MMLU-Pro settings.

Parity with printed ACL numbers is asserted (votes, gold-blind agreement selections, MC prefixes, ALL baselines).
Usage: python -B analysis/arr/a1a2_replays.py
-> analysis/out/arr/a1_policies.json, a2_eq2.csv, a2_curves.csv.gz, a2_auroc.json, a1a2_replays_manifest.json,
   reports/arr_a1_a2.md
"""
import csv
import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ACL, J1, MODELS_MATH, MODELS_MC, OPS, OUT, REPO, Boot, Equivalence, dump, groups_of,  # noqa: E402
                    guard_check, holm, manifest, p_floor, pfmt, pin, plan_core_hash, plan_hash, prefix_groups, rows,
                    rows_gz, sha, tie_key, unique, winner)

SHARES = (0.10, 0.20)
GRID = [i / 100 for i in range(101)]
K = 5
FAMILY = {1: ("HYBRID-margin 20%", "V5"), 2: ("VOTE+a0 k5", "V5"), 3: ("V1", "F2 ALL"), 4: ("ANSWERLESS->F2", "F2 ALL")}


class Pol:
    def __init__(self, y, gen, prompt, answered, selected=None):
        self.y = np.asarray(y, dtype=float)
        self.gen = np.asarray(gen, dtype=float)
        self.prompt = np.asarray(prompt, dtype=float)
        self.answered = np.asarray(answered, dtype=bool)
        self.selected = None if selected is None else np.asarray(selected, dtype=bool)


def select(score, tie, share):
    n = len(score)
    m = round(n * share)
    mask = np.zeros(n, dtype=bool)
    mask[np.lexsort((tie, np.asarray(score, dtype=float)))[:m]] = True
    return mask


# ----------------------------------------------------------------------------- loading
def load_math(table, labels):
    pops = {}
    for model in MODELS_MATH:
        lab = {r["item_id"]: r for r in labels if r["model"] == model}
        ids = sorted(lab)
        t = {c: table[model][c] for c in OPS}
        assert all(set(t[c]) == set(ids) for c in OPS)
        f2 = t["F2"]
        P = {"setting": f"math {model}", "task": "math", "model": model, "ids": ids, "n": len(ids),
             "canonical": [lab[i]["canonical_id"] for i in ids], "source": [lab[i]["dataset"] for i in ids],
             "tie": np.array([tie_key(i) for i in ids], dtype=float),
             "y0": np.array([lab[i]["initial"]["correct"] for i in ids], dtype=float),
             "y0h": np.array([lab[i]["initial"]["hist_correct"] for i in ids], dtype=float),
             "a0_answered": np.array([lab[i]["initial"]["status"] == "ok" for i in ids]),
             "a0_pred": [lab[i]["initial"]["pred"] for i in ids], "a0_status": [lab[i]["initial"]["status"] for i in ids],
             "a0_prompt": np.array([f2[i]["initial_prompt_tokens"] for i in ids], dtype=float),
             "sig": {"verbal": np.array([f2[i]["verbal_log_odds"] for i in ids], dtype=float),
                     "probe": np.array([f2[i]["probe_p_correct"] for i in ids], dtype=float),
                     "length": -np.array([f2[i]["initial_output_tokens"] for i in ids], dtype=float)},
             "ops": {}}
        assert np.array_equal(P["y0h"], np.array([f2[i]["y0"] for i in ids], dtype=float))
        for c in OPS:
            P["ops"][c] = {"y1": np.array([lab[i][c]["correct"] for i in ids], dtype=float),
                           "y1h": np.array([t[c][i]["y1"] for i in ids], dtype=float),
                           "gen": np.array([t[c][i]["full_output_tokens"] for i in ids], dtype=float),
                           "prompt": np.array([t[c][i]["full_prompt_tokens"] for i in ids], dtype=float),
                           "answered": np.array([lab[i][c]["status"] == "ok" for i in ids])}
            assert np.array_equal(P["ops"][c]["y1h"], np.array([lab[i][c]["hist_correct"] for i in ids], dtype=float))
        assert all(np.isfinite(v).all() for v in P["sig"].values())
        pops[model] = P
    return pops


def add_math_samples(P, eq):
    """Qwen3-4B math: five stored samples, gold-blind candidates and groups from the ACL replay (recomputed here,
    asserted equal), gold-blind agreement with a0 from the ACL same-rule sensitivity."""
    per = unique(rows(pin(ACL / "artifacts/math_vote/per_item.jsonl")))
    sr = unique(rows(pin(ACL / "artifacts/math_vote/same_rule_per_item.jsonl")))
    n = P["n"]
    S = {"valid": np.zeros((n, K), bool), "correct": np.zeros((n, K)), "tok": np.zeros((n, K)),
         "sprompt": P["a0_prompt"].copy(), "groups": [], "groups_a0": []}
    agree = np.zeros(n)
    for j, i in enumerate(P["ids"]):
        row = per[i]
        cands = row["candidates"]
        assert len(cands) == K
        a0 = {"pred": P["a0_pred"][j], "status": P["a0_status"][j]}
        assert sr[i]["initial"]["pred"] == a0["pred"] and sr[i]["initial"]["status"] == a0["status"]
        allc = [a0] + cands
        cache = {}

        def match(a, b, allc=allc, cache=cache):
            key = (min(a, b), max(a, b))
            if key not in cache:
                cache[key] = eq.symmetric(allc[a]["pred"], allc[b]["pred"])
            return cache[key]
        gs = groups_of(cands, lambda a, b: match(a + 1, b + 1))
        assert gs == row["votes"]["symbolic5"]["groups"], ("ACL group parity", i)
        S["groups"].append(gs)
        S["groups_a0"].append(groups_of(allc, match))
        for k, c in enumerate(cands):
            S["valid"][j, k] = c["status"] == "ok"
            S["correct"][j, k] = float(bool(c["correct_after_choice"]))
            S["tok"][j, k] = c["out_tokens"]
        agree[j] = sr[i]["gold_blind_agreement"]
    P["sig"]["agreement"] = agree
    P["samples"] = S
    finish_votes(P, acl_votes={i: per[i]["votes"] for i in P["ids"]})


def load_mc(model):
    items = [r for r in rows(pin(J1 / f"artifacts/mc/{model}_items.jsonl"))]
    t = {c: unique(r for r in items if r["condition"] == c) for c in OPS}
    ids = sorted(t["F2"])
    assert all(set(t[c]) == set(ids) for c in OPS)
    meta = {r["item_id"]: r for r in rows(pin(REPO / "data/processed/mmlupro/items.jsonl")) if r["item_id"] in t["F2"]}
    runs = json.loads(pin(J1 / "artifacts/mc/summary.json").read_text(encoding="utf-8"))["models"][model]["runs"]
    cons = unique(rows(pin(REPO / runs["consist"] / "consistency.jsonl")))
    f2 = t["F2"]

    def letter_ok(p, i):
        return isinstance(p, str) and len(p) == 1 and "A" <= p < chr(65 + meta[i]["n_options"])
    P = {"setting": f"MMLU-Pro {model}", "task": "mc", "model": model, "ids": ids, "n": len(ids),
         "canonical": [f2[i]["canonical_id"] for i in ids], "source": [f2[i]["category"] for i in ids],
         "tie": np.array([tie_key(i) for i in ids], dtype=float),
         "y0": np.array([f2[i]["y0"] for i in ids], dtype=float),
         "a0_answered": np.array([f2[i]["initial_pred"] is not None for i in ids]),
         "sig": {"verbal": np.array([f2[i]["verbal_W0"] for i in ids], dtype=float),
                 "probe": np.array([f2[i]["task_probe"] for i in ids], dtype=float),
                 "length": -np.array([f2[i]["initial_output_tokens"] for i in ids], dtype=float),
                 "agreement": np.array([f2[i]["agreement"] for i in ids], dtype=float)},
         "ops": {}}
    P["y0h"] = P["y0"]
    for c in OPS:
        P["ops"][c] = {"y1": np.array([t[c][i]["y1"] for i in ids], dtype=float),
                       "gen": np.array([t[c][i]["full_output_tokens"] for i in ids], dtype=float),
                       "prompt": np.array([t[c][i]["full_prompt_tokens"] for i in ids], dtype=float),
                       "answered": np.array([t[c][i]["pred"] is not None for i in ids])}
        P["ops"][c]["y1h"] = P["ops"][c]["y1"]
    n = P["n"]
    S = {"valid": np.zeros((n, K), bool), "correct": np.zeros((n, K)), "tok": np.zeros((n, K)),
         "sprompt": np.array([f2[i]["sample_prompt_tokens_inferred"] / K for i in ids], dtype=float),
         "groups": [], "groups_a0": []}
    for j, i in enumerate(ids):
        r = cons[i]
        assert r["split"] == "test" and r["a0_pred"] == f2[i]["initial_pred"] and int(r["a0_correct"]) == f2[i]["y0"]
        samples = r["samples"]
        assert len(samples) == K
        cands = [{"pred": s["pred"], "status": "ok" if letter_ok(s["pred"], i) else "invalid"} for s in samples]
        a0p = f2[i]["initial_pred"]
        allc = [{"pred": a0p, "status": "ok" if letter_ok(a0p, i) else "invalid"}] + cands
        S["groups"].append(groups_of(cands, lambda a, b, c=cands: c[a]["pred"] == c[b]["pred"]))
        S["groups_a0"].append(groups_of(allc, lambda a, b, c=allc: c[a]["pred"] == c[b]["pred"]))
        for k, s in enumerate(samples):
            S["valid"][j, k] = cands[k]["status"] == "ok"
            S["correct"][j, k] = float(S["valid"][j, k] and s["pred"] == meta[i]["gold"])
            if S["valid"][j, k]:
                assert bool(s["correct"]) == bool(S["correct"][j, k])
            S["tok"][j, k] = s["out_tokens"]
        assert S["tok"][j].sum() == f2[i]["sample_output_tokens"]
    P["samples"] = S
    finish_votes(P)
    return P


def finish_votes(P, acl_votes=None):
    """Plurality, VOTE+a0, vote margin and adaptive stopping from the stored groups."""
    S, n = P["samples"], P["n"]
    y0 = P["y0"]
    S["vk_correct"], S["vk_answered"], S["va0_correct"], S["va0_answered"] = {}, {}, {}, {}
    for k in range(1, K + 1):
        reps = [winner(prefix_groups(g, k))[0] for g in S["groups"]]
        S["vk_correct"][k] = np.array([0.0 if r is None else S["correct"][j, r] for j, r in enumerate(reps)])
        S["vk_answered"][k] = np.array([r is not None for r in reps])
        if acl_votes is not None and k in (1, 3, 5):
            acl = np.array([float(bool(acl_votes[i][f"symbolic{k}"]["correct"])) for i in P["ids"]])
            assert np.array_equal(acl, S["vk_correct"][k]), ("ACL vote parity", k)
        reps = [winner(prefix_groups(g, k + 1))[0] for g in S["groups_a0"]]
        S["va0_correct"][k] = np.array([0.0 if r is None else (y0[j] if r == 0 else S["correct"][j, r - 1])
                                        for j, r in enumerate(reps)])
        S["va0_answered"][k] = np.array([r is not None for r in reps])
    P["sig"]["margin"] = np.array([winner(g)[2] / K for g in S["groups"]], dtype=float)
    # adaptive stopping, samples drawn in stored order
    stop_y, stop_c, stop_a, stop_n = np.zeros(n), np.zeros(n), np.zeros(n, bool), np.zeros(n)
    two_y, two_c, two_a, two_n = np.zeros(n), np.zeros(n), np.zeros(n, bool), np.zeros(n)
    gate_y, gate_c, gate_a, gate_n = np.zeros(n), np.zeros(n), np.zeros(n, bool), np.zeros(n)
    rep5 = [winner(g)[0] for g in S["groups"]]
    for j, g in enumerate(S["groups"]):
        for draws in range(1, K + 1):
            sizes = sorted((len(x) for x in prefix_groups(g, draws)), reverse=True) + [0, 0]
            if sizes[0] - sizes[1] > K - draws or draws == K:
                rep = winner(prefix_groups(g, draws))[0]
                assert rep == rep5[j], ("STOP-decided must equal V5", P["setting"], j)
                stop_y[j], stop_c[j], stop_a[j], stop_n[j] = S["vk_correct"][K][j], S["tok"][j, :draws].sum(), rep is not None, draws
                break
        first_two = prefix_groups(g, 2)
        if S["valid"][j, 0] and S["valid"][j, 1] and len(first_two) == 1:
            two_y[j], two_c[j], two_a[j], two_n[j] = S["correct"][j, 0], S["tok"][j, :2].sum(), True, 2
        else:
            two_y[j], two_c[j], two_a[j], two_n[j] = S["vk_correct"][K][j], S["tok"][j].sum(), S["vk_answered"][K][j], K
        ga = S["groups_a0"][j]
        same = any(0 in x and 1 in x for x in ga)  # a0 and s1 both valid and in one group
        if same:
            gate_y[j], gate_c[j], gate_a[j], gate_n[j] = y0[j], S["tok"][j, 0], True, 1
        else:
            gate_y[j], gate_c[j], gate_a[j], gate_n[j] = S["va0_correct"][K][j], S["tok"][j].sum(), S["va0_answered"][K][j], K
    S["stop"] = (stop_y, stop_c, stop_a, stop_n)
    S["stop2"] = (two_y, two_c, two_a, two_n)
    S["gate"] = (gate_y, gate_c, gate_a, gate_n)


# ----------------------------------------------------------------------------- A1 policies
def policies(P):
    n, y0, z = P["n"], P["y0"], np.zeros(P["n"])
    pol = {"KEEP": Pol(y0, z, z, P["a0_answered"])}
    for op in OPS:
        o = P["ops"][op]
        pol[f"{op} ALL"] = Pol(o["y1"], o["gen"], o["prompt"], o["answered"])
    un = ~P["a0_answered"]
    for op in ("F2", "F2-scratch"):
        o = P["ops"][op]
        pol[f"ANSWERLESS->{op}"] = Pol(np.where(un, o["y1"], y0), np.where(un, o["gen"], 0), np.where(un, o["prompt"], 0),
                                       np.where(un, o["answered"], P["a0_answered"]), un)
    f2 = P["ops"]["F2"]
    for share in SHARES:
        tag = f"{round(share * 100)}%"
        sd = select(y0, P["tie"], share)
        pol[f"REF DETECT->F2 {tag}"] = Pol(np.where(sd, f2["y1"], y0), sd * f2["gen"], sd * f2["prompt"],
                                           np.where(sd, f2["answered"], P["a0_answered"]), sd)
        sb = select(-(f2["y1"] - y0), P["tie"], share)
        pol[f"REF BENEFIT->F2 {tag}"] = Pol(np.where(sb, f2["y1"], y0), sb * f2["gen"], sb * f2["prompt"],
                                            np.where(sb, f2["answered"], P["a0_answered"]), sb)
    if "samples" not in P:
        return pol
    S = P["samples"]
    tok5, pr5 = S["tok"].sum(axis=1), K * S["sprompt"]
    for k in (1, 3, 5):
        pol[f"V{k}"] = Pol(S["vk_correct"][k], S["tok"][:, :k].sum(axis=1), k * S["sprompt"], S["vk_answered"][k])
    pol["ANSWERLESS->s1"] = Pol(np.where(un, S["correct"][:, 0], y0), np.where(un, S["tok"][:, 0], 0),
                                np.where(un, S["sprompt"], 0), np.where(un, S["valid"][:, 0], P["a0_answered"]), un)
    for k in (3, 5):
        pol[f"VOTE+a0 k{k}"] = Pol(S["va0_correct"][k], S["tok"][:, :k].sum(axis=1), k * S["sprompt"], S["va0_answered"][k])
    v5 = pol["V5"]
    for share in SHARES:
        tag = f"{round(share * 100)}%"
        for name in ("margin", "agreement"):
            sel = select(P["sig"][name], P["tie"], share)
            label = "HYBRID-margin" if name == "margin" else "HYBRID-agree"
            pol[f"{label} {tag}"] = Pol(np.where(sel, f2["y1"], v5.y), tok5 + sel * f2["gen"], pr5 + sel * f2["prompt"],
                                        np.where(sel, f2["answered"], v5.answered), sel)
        sel = select(P["sig"]["agreement"], P["tie"], share)
        pol[f"AGREE->F2 {tag}"] = Pol(np.where(sel, f2["y1"], y0), tok5 + sel * f2["gen"], pr5 + sel * f2["prompt"],
                                      np.where(sel, f2["answered"], P["a0_answered"]), sel)
    for name, key in (("STOP-decided", "stop"), ("STOP-2", "stop2"), ("GATE-a0", "gate")):
        y, c, a, draws = S[key]
        pol[name] = Pol(y, c, draws * S["sprompt"], a)
        P.setdefault("mean_draws", {})[name] = float(draws.mean())
    pol["REF BEST(a0,V5)"] = Pol(np.maximum(y0, v5.y), tok5, pr5, v5.answered | P["a0_answered"])
    return pol


def describe(P, pol, boot):
    y0, a0a, n = P["y0"], P["a0_answered"], P["n"]
    out = {}
    for name, p in pol.items():
        comp = (y0 == 0) & ~a0a & (p.y == 1)
        cont = (y0 == 0) & a0a & (p.y == 1)
        floss = (y0 == 1) & (p.y == 0) & ~p.answered
        cdam = (y0 == 1) & (p.y == 0) & p.answered
        delta = p.y - y0
        assert abs(delta.mean() - (comp.mean() + cont.mean() - floss.mean() - cdam.mean())) < 1e-12
        out[name] = {"accuracy_pct": 100 * float(p.y.mean()), "correct": int(p.y.sum()),
                     "delta_vs_keep": boot.ci(delta), "repairs": int((comp | cont).sum()), "damage": int((floss | cdam).sum()),
                     "repairs_completion": boot.ci(comp), "repairs_content": boot.ci(cont),
                     "damage_format_loss": boot.ci(floss), "damage_content": boot.ci(cdam),
                     "generated_tokens_per_item": float(p.gen.mean()), "prompt_tokens_per_item": float(p.prompt.mean()),
                     "selected_n": None if p.selected is None else int(p.selected.sum()),
                     "final_unanswered_pct": 100 * float((~p.answered).mean())}
    return out


def contrasts(P, pol, boot):
    pairs = [(a, b, f) for f, (a, b) in FAMILY.items()]
    pairs += [("HYBRID-margin 10%", "V5", None), ("HYBRID-agree 10%", "V5", None), ("HYBRID-agree 20%", "V5", None),
              ("V5", "AGREE->F2 10%", None), ("V5", "AGREE->F2 20%", None), ("STOP-decided", "V5", None),
              ("STOP-2", "V5", None), ("GATE-a0", "V5", None), ("VOTE+a0 k3", "V3", None), ("ANSWERLESS->s1", "KEEP", None),
              ("ANSWERLESS->F2", "KEEP", None), ("ANSWERLESS->F2-scratch", "F2-scratch ALL", None),
              ("V1", "KEEP", None), ("V5", "V1", None), ("V3", "V1", None), ("V5", "V3", None),
              ("REF BEST(a0,V5)", "V5", None), ("V1", "F0 ALL", None)]
    out = []
    for a, b, fam in pairs:
        if a not in pol or b not in pol:
            continue
        d = pol[a].y - pol[b].y
        out.append({"setting": P["setting"], "a": a, "b": b, "family_member": fam, "difference": boot.ci(d),
                    "token_difference": float((pol[a].gen - pol[b].gen).mean())})
    return out


# ----------------------------------------------------------------------------- A2
def operators_for(P, label):
    ops = {}
    for op in OPS:
        o = P["ops"][op]
        ops[op] = (o["y1"] if label == "common" else o["y1h"], o["gen"])
    if "samples" in P and label == "common":
        ops["s1"] = (P["samples"]["correct"][:, 0], P["samples"]["tok"][:, 0])
    return ops


def reuses_sample(op, sig):
    """s1 is one of the five samples that agreement and margin already paid for: no second charge."""
    return op == "s1" and sig in ("agreement", "margin")


def acquisition(P, sig):
    if sig in ("agreement", "margin") and "samples" in P:
        return float(P["samples"]["tok"].sum(axis=1).mean())
    return 0.0


def eq2_table(P, boot, label):
    y0 = P["y0"] if label == "common" else P["y0h"]
    n = P["n"]
    out = []
    for op, (y1, gen) in operators_for(P, label).items():
        for sig in list(P["sig"]) + ["random", "REF detect", "REF benefit"]:
            for share in SHARES:
                m = round(n * share)
                if sig == "random":
                    sel = np.ones(n, dtype=bool)
                    w, r, d = boot.ratio_ci(1 - y0, np.ones(n)), boot.ratio_ci((1 - y0) * y1, 1 - y0), boot.ratio_ci(y0 * (1 - y1), y0)
                    delta = boot.ci((m / n) * (y1 - y0))
                    cost = (m / n) * gen.mean()
                else:
                    score = {"REF detect": y0, "REF benefit": -(y1 - y0)}.get(sig, P["sig"].get(sig))
                    sel = select(score, P["tie"], share)
                    w = boot.ratio_ci(sel * (1 - y0), sel.astype(float))
                    r = boot.ratio_ci(sel * (1 - y0) * y1, sel * (1 - y0))
                    d = boot.ratio_ci(sel * y0 * (1 - y1), sel * y0)
                    delta = boot.ci(sel * (y1 - y0))
                    cost = 0.0 if reuses_sample(op, sig) else float((sel * gen).mean())
                wr, rr, dr = w["estimate"] / 100, r["estimate"] / 100 if r["n_den"] else 0.0, d["estimate"] / 100 if d["n_den"] else 0.0
                resid = delta["estimate"] / 100 - (m / n) * (wr * rr - (1 - wr) * dr)
                assert abs(resid) < 1e-12, (P["setting"], op, sig, share, resid)
                out.append({"setting": P["setting"], "labels": label, "operator": op, "signal": sig, "share_pct": round(share * 100),
                            "selected_n": m, "w_S": w, "repair_rate": r, "damage_rate": d, "delta_pp": delta,
                            "generated_tokens_per_item": acquisition(P, sig) + cost})
    return out


def curves(P, label, pol_extra=None):
    y0 = P["y0"] if label == "common" else P["y0h"]
    n = P["n"]
    out = []
    for op, (y1, gen) in operators_for(P, label).items():
        for sig in list(P["sig"]) + ["random", "REF detect", "REF benefit"]:
            if sig == "random":
                order = np.arange(n)
            else:
                score = {"REF detect": y0, "REF benefit": -(y1 - y0)}.get(sig, P["sig"].get(sig))
                order = np.lexsort((P["tie"], np.asarray(score, dtype=float)))
            cs = {k: np.r_[0, np.cumsum(v[order])] for k, v in
                  {"d": y1 - y0, "g": gen, "w": 1 - y0, "r": (1 - y0) * y1, "dm": y0 * (1 - y1), "c": y0}.items()}
            for share in GRID:
                m = round(n * share)
                if sig == "random":
                    vals = {k: (m / n) * cs[k][-1] for k in cs}
                else:
                    vals = {k: cs[k][m] for k in cs}
                out.append({"setting": P["setting"], "labels": label, "operator": op, "signal": sig, "share_pct": round(share * 100),
                            "selected_n": m, "delta_pp": 100 * vals["d"] / n,
                            "generated_tokens_per_item": acquisition(P, sig) + (0.0 if reuses_sample(op, sig) else vals["g"] / n),
                            "w_S": vals["w"] / m if m else None, "repair_rate": vals["r"] / vals["w"] if vals["w"] else None,
                            "damage_rate": vals["dm"] / vals["c"] if vals["c"] else None})
    if label == "common" and "samples" in P:
        S, f2 = P["samples"], P["ops"]["F2"]
        v5y = S["vk_correct"][K]
        tok5 = S["tok"].sum(axis=1)
        for name in ("margin", "agreement"):
            order = np.lexsort((P["tie"], P["sig"][name]))
            cs_d = np.r_[0, np.cumsum((f2["y1"] - v5y)[order])]
            cs_g = np.r_[0, np.cumsum(f2["gen"][order])]
            for share in GRID:
                m = round(n * share)
                out.append({"setting": P["setting"], "labels": label, "operator": "HYBRID(F2 | V5)", "signal": name,
                            "share_pct": round(share * 100), "selected_n": m,
                            "delta_pp": 100 * ((v5y - y0).sum() + cs_d[m]) / n,
                            "generated_tokens_per_item": float(tok5.mean() + cs_g[m] / n),
                            "w_S": None, "repair_rate": None, "damage_rate": None})
    return out


def auroc_parts(score, y, W):
    """Weighted AUROC numerator and denominator (equal scores count 1/2) for the point (W=None) or each replicate."""
    order = np.argsort(score, kind="mergesort")
    s, yy = score[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    w = np.ones((1, len(s)), dtype=np.float64) if W is None else W[:, order].astype(np.float64)
    gp = np.add.reduceat(w * yy, starts, axis=1)
    gn = np.add.reduceat(w * (1 - yy), starts, axis=1)
    below = np.cumsum(gn, axis=1) - gn
    return (gp * (below + 0.5 * gn)).sum(axis=1), gp.sum(axis=1) * gn.sum(axis=1)


def auroc_table(P, boot, label):
    y = P["y0"] if label == "common" else P["y0h"]
    src = np.array(P["source"])
    out = []
    for sig, score in P["sig"].items():
        num, den = auroc_parts(score, y, None)
        bnum, bden = auroc_parts(score, y, boot.W)
        entry = {"setting": P["setting"], "labels": label, "signal": sig, "pooled": float(num[0] / den[0]),
                 "pooled_ci": np.quantile(bnum / bden, [0.025, 0.975]).tolist(), "per_source": {}}
        wn, wd, bwn, bwd = 0.0, 0.0, np.zeros(boot.n_boot), np.zeros(boot.n_boot)
        for s in sorted(set(P["source"])):
            idx = np.flatnonzero(src == s)
            n1, d1 = auroc_parts(score[idx], y[idx], None)
            b1, e1 = auroc_parts(score[idx], y[idx], boot.W[:, idx])
            ok = e1 > 0
            entry["per_source"][s] = {"n": int(len(idx)), "accuracy_pct": 100 * float(y[idx].mean()),
                                      "a0_unanswered_pct": 100 * float((~P["a0_answered"][idx]).mean()),
                                      "auroc": float(n1[0] / d1[0]) if d1[0] else None,
                                      "ci": np.quantile(b1[ok] / e1[ok], [0.025, 0.975]).tolist() if d1[0] else None}
            wn, wd, bwn, bwd = wn + n1[0], wd + d1[0], bwn + b1, bwd + e1
        entry["within_source"] = float(wn / wd)
        entry["within_source_ci"] = np.quantile(bwn / bwd, [0.025, 0.975]).tolist()
        out.append(entry)
    return out


# ----------------------------------------------------------------------------- parity with printed ACL numbers
def parity(math, mc, pols):
    checks = []
    base = {(r["task"], r["model"], r["policy"]): r for r in csv.DictReader(
        io.StringIO(pin(ACL / "artifacts/curves/baselines.csv").read_text(encoding="utf-8-sig")))}
    for model, P in math.items():
        for op in OPS:
            want = float(base[("math", model, f"{op} ALL")]["delta_pp"])
            got = 100 * (P["ops"][op]["y1h"] - P["y0h"]).mean()
            assert abs(want - got) < 1e-9, (model, op)
    checks.append("math ALL deltas (historical labels) equal the ACL baselines for 4 models x 3 operators")
    same = json.loads(pin(ACL / "artifacts/math_vote/same_rule_summary.json").read_text(encoding="utf-8"))
    acc = {p["policy"]: p["accuracy_pct"] for p in same["policies"]}
    p4 = pols["math 4B"]
    assert int(p4["KEEP"].y.sum()) == same["policies"][0]["correct"] == 4304
    for name, ours in (("F2_ALL", "F2 ALL"), ("F2-scratch_ALL", "F2-scratch ALL"), ("PLURALITY_SYMBOLIC_K5", "V5"),
                       ("BLIND_AGREEMENT_F2_10", "AGREE->F2 10%"), ("BLIND_AGREEMENT_F2_20", "AGREE->F2 20%")):
        assert abs(acc[name] - 100 * p4[ours].y.mean()) < 1e-9, name
    checks.append("4B math same-rule KEEP, F2/F2-scratch ALL, V5 and gold-blind agreement 10/20 % equal the ACL values")
    curves_summary = json.loads(pin(ACL / "artifacts/curves/summary.json").read_text(encoding="utf-8"))
    for row in curves_summary["prefixes"]:
        P = mc[row["model"]]
        p = pols[P["setting"]]
        assert abs(row["plurality_accuracy"] - p[f"V{row['k']}"].y.mean()) < 1e-12
        assert abs(row["plurality_generated_tokens"] - p[f"V{row['k']}"].gen.mean()) < 1e-9
        for share in (("10", "20") if row["k"] == K else ()):  # the ACL recomputes agreement from the first k samples
            ours = p[f"AGREE->F2 {share}%"]
            assert abs(row["agreement_policies"][share]["vs_KEEP"]["estimate"] - (ours.y - P["y0"]).mean()) < 1e-12
    checks.append("MMLU-Pro plurality k=1,3,5 accuracy and cost, and agreement 10/20 % gains equal the ACL prefixes")
    return checks


# ----------------------------------------------------------------------------- report
def fmt(ci):
    return f"{ci['estimate']:+.2f} [{ci['lo']:+.2f}, {ci['hi']:+.2f}]"


def report(a0, result, eq2, auroc):
    L = ["# ARR round, tier A: comparator policies, decompositions, Eq. (2) terms, AUROC",
         "", "Generated by `analysis/arr/a1a2_replays.py` (plan `docs/planning/arr_round/plan.md`, sha256 "
         f"`{result['plan_sha256_lf'][:16]}…`). Post hoc and exploratory (DEVIATIONS D20). Stored outputs only.",
         "Δ in percentage points with paired 95 % cluster-bootstrap intervals (4,000 replicates, seed 20261110).", "",
         "## A0. Common gold-blind labels (correct items, common rule vs historical)", "",
         "| Model | a0 | F0 | F2 | F2-scratch |", "|---|---:|---:|---:|---:|"]
    for model, v in a0["models"].items():
        L.append(f"| {model} | " + " | ".join(f"{v[c]['common_correct']} ({v[c]['historical_correct']})"
                                             for c in ("initial", "F0", "F2", "F2-scratch")) + " |")
    L += ["", f"Qwen3-4B parity with the ACL same-rule rows: {a0['acl_4B_parity_checks']} checks passed. "
          f"Guards: {len(a0['guards'])}.", ""]
    for setting, block in result["settings"].items():
        L += [f"## A1. {setting} (n = {block['n']})", "",
              "| Policy | Acc. % | Δ vs KEEP | Completion | Content | Format loss | Content damage | Gen. tok/item |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, p in block["policies"].items():
            L.append(f"| {name} | {p['accuracy_pct']:.2f} | {fmt(p['delta_vs_keep'])} | {p['repairs_completion']['estimate']:+.2f} | "
                     f"{p['repairs_content']['estimate']:+.2f} | {-p['damage_format_loss']['estimate']:+.2f} | "
                     f"{-p['damage_content']['estimate']:+.2f} | {p['generated_tokens_per_item']:.0f} |")
        L += ["", "| Contrast | Difference (pp) | Token difference | §5 member | p (boot) |", "|---|---:|---:|---:|---:|"]
        for c in block["contrasts"]:
            L.append(f"| {c['a']} − {c['b']} | {fmt(c['difference'])} | {c['token_difference']:+.0f} | "
                     f"{c['family_member'] or ''} | {pfmt(c['difference']['p_boot'])} |")
        L.append("")
    L += ["## §5 family members available after tier A (interim Holm; the final Holm runs over the complete family)", "",
          "| Setting | Contrast | Difference | p (boot) | Holm (interim) |", "|---|---|---:|---:|---:|"]
    for c in result["family_interim"]:
        L.append(f"| {c['setting']} | {c['a']} − {c['b']} | {fmt(c['difference'])} | {pfmt(c['p_boot'])} | {pfmt(c['holm_interim'])} |")
    L += ["", "## A2. Eq. (2) terms at 10 % and 20 % (F2; common labels for math)", "",
          "| Setting | Signal | Share | w_S | Repair rate | Damage rate | Δ | Gen. tok/item |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in eq2:
        if r["operator"] == "F2" and r["labels"] == "common":
            L.append(f"| {r['setting']} | {r['signal']} | {r['share_pct']} % | {r['w_S']['estimate']:.1f} | "
                     f"{r['repair_rate']['estimate']:.1f} [{r['repair_rate']['lo']:.1f}, {r['repair_rate']['hi']:.1f}] | "
                     f"{r['damage_rate']['estimate']:.1f} [{r['damage_rate']['lo']:.1f}, {r['damage_rate']['hi']:.1f}] | "
                     f"{fmt(r['delta_pp'])} | {r['generated_tokens_per_item']:.0f} |")
    L += ["", "All operators, signals and both label sets: `analysis/out/arr/a2_eq2.csv`; curves 0–100 %: `a2_curves.csv.gz`.", "",
          "## A2. AUROC for y0 = 1 (pooled, within-source, per source)", "",
          "| Setting | Labels | Signal | Pooled [CI] | Within-source [CI] | Per source |", "|---|---|---|---:|---:|---|"]
    for a in auroc:
        per = "; ".join(f"{s} {v['auroc']:.3f}" for s, v in a["per_source"].items() if v["auroc"] is not None) \
            if a["setting"].startswith("math") else f"{len(a['per_source'])} categories, range " \
            f"{min(v['auroc'] for v in a['per_source'].values() if v['auroc'] is not None):.3f}–" \
            f"{max(v['auroc'] for v in a['per_source'].values() if v['auroc'] is not None):.3f}"
        L.append(f"| {a['setting']} | {a['labels']} | {a['signal']} | {a['pooled']:.3f} [{a['pooled_ci'][0]:.3f}, {a['pooled_ci'][1]:.3f}] | "
                 f"{a['within_source']:.3f} [{a['within_source_ci'][0]:.3f}, {a['within_source_ci'][1]:.3f}] | {per} |")
    L += ["", "Parity checks: " + "; ".join(result["parity"]) + ".", ""]
    (REPO / "reports" / "arr_a1_a2.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- main
def main():
    plan = plan_hash()
    a0 = json.loads(pin(OUT / "a0_common_labels.json").read_text(encoding="utf-8"))
    assert a0["plan_core_sha256"] == plan_core_hash() and not a0["guards"]
    labels = list(rows_gz(pin(OUT / "math_common_labels.jsonl.gz")))
    table = {}
    for r in rows(pin(J1 / "artifacts/math/item_evaluations.jsonl")):
        table.setdefault(r["model"], {}).setdefault(r["condition"], {})[r["item_id"]] = r
    math = load_math(table, labels)
    eq = Equivalence()
    try:
        add_math_samples(math["4B"], eq)
        guards = list(eq.guards)
    finally:
        eq.close()
    # Plan Amendment 5: a guard hit is usable only when hand-checked (analysis/arr/reviewed_guards.json). Until
    # 2026-11-11 this read `assert not guards`, written before Amendment 5; the shared equivalence cache re-reports
    # every stored guard row, and later analyses stored 49 reviewed ones, so that assertion stopped any re-run.
    guard_check(guards)
    mc = {m: load_mc(m) for m in MODELS_MC}
    boot_math = Boot(math["4B"]["canonical"])
    boot_mc = Boot(mc["4B"]["canonical"])
    assert all(P["ids"] == math["4B"]["ids"] for P in math.values()) and mc["4B"]["ids"] == mc["T8B"]["ids"]
    settings = [(P, boot_math) for P in math.values()] + [(P, boot_mc) for P in mc.values()]
    pols, result = {}, {"plan_sha256_lf": plan, "plan_core_sha256": plan_core_hash(), "settings": {}}
    eq2, auroc, curve_rows, family = [], [], [], []
    for P, boot in settings:
        pol = policies(P)
        pols[P["setting"]] = pol
        cons = contrasts(P, pol, boot)
        result["settings"][P["setting"]] = {"n": P["n"], "a0_unanswered_pct": 100 * float((~P["a0_answered"]).mean()),
                                           "policies": describe(P, pol, boot), "contrasts": cons,
                                           "mean_samples_drawn": P.get("mean_draws")}
        family += [{"setting": P["setting"], "a": c["a"], "b": c["b"], "member": c["family_member"],
                    "difference": c["difference"], "p_boot": c["difference"]["p_boot"]} for c in cons if c["family_member"]]
        for label in (("common", "historical") if P["task"] == "math" else ("common",)):
            eq2 += eq2_table(P, boot, label)
            auroc += auroc_table(P, boot, label)
            curve_rows += curves(P, label)
    result["parity"] = parity(math, mc, pols)
    adj = holm({f"{f['setting']}|{f['a']}|{f['b']}": max(f["p_boot"], p_floor()) for f in family})
    for f in family:
        f["holm_interim"] = adj[f"{f['setting']}|{f['a']}|{f['b']}"]
    result["family_interim"] = family
    dump(OUT / "a1_policies.json", result)
    flat = []
    for r in eq2:
        flat.append({k: (v["estimate"] if isinstance(v, dict) else v) for k, v in r.items()} |
                    {f"{k}_lo": v["lo"] for k, v in r.items() if isinstance(v, dict)} |
                    {f"{k}_hi": v["hi"] for k, v in r.items() if isinstance(v, dict)})
    with (OUT / "a2_eq2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(flat)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(curve_rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(curve_rows)
    with (OUT / "a2_curves.csv.gz").open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue().encode("utf-8"))
    dump(OUT / "a2_auroc.json", auroc)
    report(a0, result, eq2, auroc)
    manifest("a1a2_replays", {"outputs": {p: sha(OUT / p) for p in ("a1_policies.json", "a2_eq2.csv", "a2_curves.csv.gz", "a2_auroc.json")}})
    print(json.dumps({"settings": list(result["settings"]), "eq2_rows": len(eq2), "curve_rows": len(curve_rows),
                      "auroc_rows": len(auroc), "family_interim": len(family), "parity": result["parity"]}, indent=1))


if __name__ == "__main__":
    main()
