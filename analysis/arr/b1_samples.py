"""B1 of the ARR plan (docs/planning/arr_round/plan.md §B, B1; A1/A2; Amendments 2 and 3): more samples per math model.

Plan A1: "After B1, the same definitions apply unchanged to the other math models and to k up to 8." Plan B1:
plurality k in {1, 3, 5, 8}; adaptive rules in the stored order with a draw limit K in {5, 8}.

Samples (Amendment 2; test split):
- Qwen3-1.7B, 8B, Tulu-3-8B: indices 0-4 from the B1 k5 run (stream h2a-consistency), indices 5-7 from the k3 run
  (stream h2a-consistency-ext).
- Qwen3-4B: indices 0-4 are the stored samples (the ACL replay's gold-blind candidates, reused as A1 does); indices
  5-7 from the k3 run.
A k-sample vote uses the first k samples in index order.

Labels: the common gold-blind rule (A0), in two phases. Every new sample candidate is written and hashed before any
gold is read, then scored with equivalent(pred, gold). Groups are complete-link with symmetric equivalence (ACL
math_vote_rules.md); a prefix of k samples has the prefix's groups.

Signals from the samples, as in A1:
- vote margin: the largest gold-blind group among the first five samples / 5;
- agreement with a0: the ACL gold-blind reconstruction, the mean over the first five samples of
  equivalent(sample_pred, a0_pred) on extracted predictions (also for truncated or conflicting samples).

Outputs:
- the A1/A2 analysis (a1a2_replays.py functions, unchanged) for 1.7B, 8B and T8B, with these samples;
- the k <= 8 policies for all four math models;
- for 4B, the five-sample policies are asserted equal to A1's.

Usage: python -B analysis/arr/b1_samples.py
-> analysis/out/arr/b1_extractions.jsonl.gz (frozen before gold), b1_labels.jsonl.gz, b1_policies.json, b1_eq2.csv,
   b1_curves.csv.gz, b1_auroc.json, b1_samples_manifest.json; reports/arr_b1.md
"""
import csv
import gzip
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1a2_replays as A  # noqa: E402
from common import (ACL, J1, MODELS_MATH, OUT, REPO, Boot, Equivalence, candidate, dump, groups_of, guard_check, manifest,  # noqa: E402
                    p_floor, pfmt, pin, plan_core_hash, plan_hash, prefix_groups, rows, rows_gz, sha, unique, winner,
                    write_rows_gz)

KMAX = 8
KS = (1, 3, 5, 8)
STREAMS = ((5, "h2a-consistency", range(0, 5)), (3, "h2a-consistency-ext", range(5, 8)))
NEW_MODELS = ("1.7B", "8B", "T8B")   # A1/A2 with samples for the first time


# ----------------------------------------------------------------------------- inputs
def verified(d, name):
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert sha(d / name) == man["files"][name], ("manifest checksum", d.name, name)
    return pin(d / name), man


def sample_run(model, split, k, stream, pool):
    """Every chunk of one B1 samples run, hash-checked. Returns (records by item, info) or (None, reason)."""
    exp = "b1" if split == "test" else "b1-dev"
    root = REPO / "results" / "arr" / exp / model
    pat = re.compile(rf"^{exp}-samples-{split}-k{k}-{re.escape(stream)}-t4096(?:-c(\d+))?-{re.escape(model)}-([0-9a-f]{{7}})$")
    by_commit = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        m = pat.match(d.name)
        if m and (d / "manifest.json").exists():
            by_commit.setdefault(m.group(2), {})[None if m.group(1) is None else int(m.group(1))] = d
    if not by_commit:
        return None, "no run synced"
    assert len(by_commit) == 1, ("B1 runs from more than one code commit (CLAUDE.md rule 8)", model, split, k, sorted(by_commit))
    (commit, chunks), = by_commit.items()
    recs, sizes = {}, set()
    for c, d in sorted(chunks.items(), key=lambda kv: -1 if kv[0] is None else kv[0]):
        path, man = verified(d, "samples.jsonl")
        cfg = json.loads(verified(d, "config.json")[0].read_text(encoding="utf-8"))
        assert (cfg["stage"], cfg["task"], cfg["model"], cfg["split"], cfg["k"], cfg["stream"], cfg["max_tokens"]) == \
            ("samples", "math", model, split, k, stream, 4096), d.name
        assert cfg["chunk_index"] == (c or 0) and bool(cfg["chunk_size"]) == (c is not None), d.name
        assert man["env"].get("VLLM_BATCH_INVARIANT") == "1" and man["code_commit"].startswith(commit), d.name
        sizes.add(cfg["chunk_size"])
        for r in rows(path):
            assert r["item_id"] not in recs, ("item in two chunks", model, k, r["item_id"])
            assert r["k"] == k and r["stream"] == stream and len(r["samples"]) == k
            recs[r["item_id"]] = r
    if None in chunks:
        assert len(chunks) == 1, ("a run with and without chunks", model, k)
    else:
        need = math.ceil(len(pool) / sizes.pop())
        if sorted(chunks) != list(range(need)):
            return None, f"{len(chunks)} of {need} chunks synced"
    if set(recs) != set(pool):
        return None, f"covers {len(set(recs) & set(pool))} of {len(pool)} items"
    return recs, {"run_commit": commit, "runs": [chunks[c].name for c in sorted(chunks, key=lambda c: -1 if c is None else c)]}


# ----------------------------------------------------------------------------- samples structure
def build_samples(P, cands, eq):
    """cands[j]: KMAX candidate dicts (pred, status, correct, out_tokens). Groups over the samples and over (a0, samples)."""
    n = P["n"]
    S8 = {"valid": np.zeros((n, KMAX), bool), "correct": np.zeros((n, KMAX)), "tok": np.zeros((n, KMAX)),
          "groups": [], "groups_a0": []}
    for j in range(n):
        cs = cands[j]
        assert len(cs) == KMAX
        a0 = {"pred": P["a0_pred"][j], "status": P["a0_status"][j]}
        allc = [a0] + cs
        cache = {}

        def match(a, b, allc=allc, cache=cache):
            key = (min(a, b), max(a, b))
            if key not in cache:
                cache[key] = eq.symmetric(allc[a]["pred"], allc[b]["pred"])
            return cache[key]
        S8["groups"].append(groups_of(cs, lambda a, b: match(a + 1, b + 1)))
        S8["groups_a0"].append(groups_of(allc, match))
        for k, c in enumerate(cs):
            S8["valid"][j, k] = c["status"] == "ok"
            S8["correct"][j, k] = float(bool(c["correct"]))
            S8["tok"][j, k] = c["out_tokens"]
    return S8


def five_view(P, S8, sprompt):
    """The first five samples in A1's structure, so that a1a2_replays.finish_votes and policies apply unchanged."""
    return {"valid": S8["valid"][:, :5].copy(), "correct": S8["correct"][:, :5].copy(), "tok": S8["tok"][:, :5].copy(),
            "sprompt": sprompt, "groups": [prefix_groups(g, 5) for g in S8["groups"]],
            "groups_a0": [prefix_groups(g, 6) for g in S8["groups_a0"]]}


def k8_policies(P, S8):
    """Plurality k in {1,3,5,8}, VOTE+a0 k in {3,5,8}, and the adaptive rules with a draw limit K in {5, 8}."""
    n, y0, sp = P["n"], P["y0"], P["samples"]["sprompt"]
    pol, draws_mean = {}, {}
    vk, va = {}, {}
    for k in range(1, KMAX + 1):
        reps = [winner(prefix_groups(g, k))[0] for g in S8["groups"]]
        vk[k] = (np.array([0.0 if r is None else S8["correct"][j, r] for j, r in enumerate(reps)]), np.array([r is not None for r in reps]))
        reps = [winner(prefix_groups(g, k + 1))[0] for g in S8["groups_a0"]]
        va[k] = (np.array([0.0 if r is None else (y0[j] if r == 0 else S8["correct"][j, r - 1]) for j, r in enumerate(reps)]),
                 np.array([r is not None for r in reps]))
    for k in KS:
        pol[f"V{k}"] = A.Pol(vk[k][0], S8["tok"][:, :k].sum(axis=1), k * sp, vk[k][1])
    for k in (3, 5, 8):
        pol[f"VOTE+a0 k{k}"] = A.Pol(va[k][0], S8["tok"][:, :k].sum(axis=1), k * sp, va[k][1])
    for K in (5, 8):
        stop, two, gate = [np.zeros(n), np.zeros(n), np.zeros(n, bool), np.zeros(n)], [np.zeros(n), np.zeros(n), np.zeros(n, bool), np.zeros(n)], \
            [np.zeros(n), np.zeros(n), np.zeros(n, bool), np.zeros(n)]
        repK = [winner(prefix_groups(g, K))[0] for g in S8["groups"]]
        for j, g in enumerate(S8["groups"]):
            for d in range(1, K + 1):
                sizes = sorted((len(x) for x in prefix_groups(g, d)), reverse=True) + [0, 0]
                if sizes[0] - sizes[1] > K - d or d == K:
                    rep = winner(prefix_groups(g, d))[0]
                    assert rep == repK[j], ("STOP-decided must equal V_K", P["setting"], K, j)
                    stop[0][j], stop[1][j], stop[2][j], stop[3][j] = vk[K][0][j], S8["tok"][j, :d].sum(), rep is not None, d
                    break
            if S8["valid"][j, 0] and S8["valid"][j, 1] and len(prefix_groups(g, 2)) == 1:
                two[0][j], two[1][j], two[2][j], two[3][j] = S8["correct"][j, 0], S8["tok"][j, :2].sum(), True, 2
            else:
                two[0][j], two[1][j], two[2][j], two[3][j] = vk[K][0][j], S8["tok"][j, :K].sum(), vk[K][1][j], K
            if any(0 in x and 1 in x for x in S8["groups_a0"][j]):
                gate[0][j], gate[1][j], gate[2][j], gate[3][j] = y0[j], S8["tok"][j, 0], True, 1
            else:
                gate[0][j], gate[1][j], gate[2][j], gate[3][j] = va[K][0][j], S8["tok"][j, :K].sum(), va[K][1][j], K
        for name, (y, c, a, d) in ((f"STOP-decided K{K}", stop), (f"STOP-2 K{K}", two), (f"GATE-a0 K{K}", gate)):
            pol[name] = A.Pol(y, c, d * sp, a)
            draws_mean[name] = float(d.mean())
    return pol, draws_mean


def attach(P, model, eq, labels=None, per=None):
    """Attach the samples of `model` to the setting P from this script's frozen labels (b1_labels.jsonl.gz), for use
    by other analyses (C1, B4): P["samples"] (the five-sample view), the agreement and margin signals, and the votes
    of a1a2_replays.finish_votes. Returns the eight-sample structure, or None when B1 is not complete for the model."""
    if labels is None:
        path = OUT / "b1_labels.jsonl.gz"
        if not path.exists():
            return None
        man = json.loads((OUT / "b1_samples_manifest.json").read_text(encoding="utf-8"))
        assert sha(path) == man["outputs"]["b1_labels.jsonl.gz"], "b1_labels.jsonl.gz differs from its manifest"
        if man["settings_status"].get(model) != "complete":
            return None
        pin(path)
        labels = [r for r in rows_gz(path) if r["model"] == model]
    by = {}
    for e in labels:
        by.setdefault(e["item_id"], {})[e["index"]] = e
    if model == "4B":
        per = per or unique(rows(pin(ACL / "artifacts/math_vote/per_item.jsonl")))
        A.add_math_samples(P, eq)   # A1's five stored samples, groups and agreement, unchanged
        cands = [[{"pred": c["pred"], "status": c["status"], "correct": bool(c["correct_after_choice"]),
                   "out_tokens": c["out_tokens"]} for c in per[i]["candidates"]] +
                 [{f: by[i][x][f] for f in ("pred", "status", "correct", "out_tokens")} for x in range(5, KMAX)]
                 for i in P["ids"]]
        S8 = build_samples(P, cands, eq)
        assert all(prefix_groups(g8, 5) == g5 for g8, g5 in zip(S8["groups"], P["samples"]["groups"])), "4B groups"
        assert all(prefix_groups(g8, 6) == g5 for g8, g5 in zip(S8["groups_a0"], P["samples"]["groups_a0"])), "4B a0 groups"
    else:
        cands = [[{f: by[i][x][f] for f in ("pred", "status", "correct", "out_tokens")} for x in range(KMAX)] for i in P["ids"]]
        S8 = build_samples(P, cands, eq)
        P["samples"] = five_view(P, S8, P["a0_prompt"].copy())
        P["sig"]["agreement"] = np.array([sum(eq(c["pred"], P["a0_pred"][j]) for c in cands[j][:5]) / 5 for j in range(P["n"])])
        A.finish_votes(P)
    S8["cands"] = cands
    return S8


K8_CONTRASTS = [("V8", "V5"), ("V8", "V1"), ("V5", "V1"), ("V3", "V1"), ("V1", "KEEP"), ("VOTE+a0 k8", "V8"),
                ("VOTE+a0 k5", "V5"), ("STOP-2 K8", "V8"), ("GATE-a0 K8", "V8"), ("STOP-2 K5", "V5"), ("GATE-a0 K5", "V5")]


# ----------------------------------------------------------------------------- report
def fmt(ci):
    return f"{ci['estimate']:+.2f} [{ci['lo']:+.2f}, {ci['hi']:+.2f}]"


def report(res, eq2):
    L = ["# ARR round, B1: more samples per math model", "",
         f"Generated by `analysis/arr/b1_samples.py` (plan `docs/planning/arr_round/plan.md`, core sha256 "
         f"`{res['plan_core_sha256'][:16]}…`). Post hoc and exploratory (DEVIATIONS D20); added in revision.",
         "Δ in percentage points with paired 95 % cluster-bootstrap intervals (4,000 replicates, seed 20261110). Common "
         "gold-blind labels. Costs: generated tokens per item after the shared a0; samples are charged in full.", ""]
    missing = {m: v for m, v in res["status"].items() if v != "complete"}
    if missing:
        L += ["Not yet complete: " + "; ".join(f"math {m}: TODO(missing: B1 samples, {why})" for m, why in missing.items()) + ".", ""]
    L += ["## Plurality and adaptive rules up to eight samples", "",
          "| Setting | Policy | Acc. % | Δ vs KEEP | Gen. tok/item | Mean samples drawn |", "|---|---|---:|---:|---:|---:|"]
    for s, v in res["k8"].items():
        for name, p in v["policies"].items():
            drawn = v["mean_samples_drawn"].get(name)
            L.append(f"| {s} | {name} | {p['accuracy_pct']:.2f} | {fmt(p['delta_vs_keep'])} | {p['generated_tokens_per_item']:.0f} | "
                     f"{'' if drawn is None else f'{drawn:.2f}'} |")
    L += ["", "| Setting | Contrast | Difference (pp) | Token difference |", "|---|---|---:|---:|"]
    for s, v in res["k8"].items():
        for c in v["contrasts"]:
            L.append(f"| {s} | {c['a']} − {c['b']} | {fmt(c['difference'])} | {c['token_difference']:+.0f} |")
    for s, block in res["a1"].items():
        L += ["", f"## A1 with B1 samples: {s} (n = {block['n']})", "",
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
    if res["family"]:
        L += ["", "## §5 family members added by B1 (Holm over the complete family is applied in the final family script)", "",
              "| Setting | Member | Contrast | Difference | p (boot) |", "|---|---:|---|---:|---:|"]
        for f in res["family"]:
            L.append(f"| {f['setting']} | {f['member']} | {f['a']} − {f['b']} | {fmt(f['difference'])} | {pfmt(f['p_boot'])} |")
    if eq2:
        L += ["", "## A2 Eq. (2) terms with the sample signals, at 10 % and 20 % (F2 and s1)", "",
              "| Setting | Operator | Signal | Share | w_S | Repair rate | Damage rate | Δ | Gen. tok/item |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in eq2:
            if r["operator"] in ("F2", "s1") and r["signal"] in ("agreement", "margin"):
                L.append(f"| {r['setting']} | {r['operator']} | {r['signal']} | {r['share_pct']} % | {r['w_S']['estimate']:.1f} | "
                         f"{r['repair_rate']['estimate']:.1f} | {r['damage_rate']['estimate']:.1f} | {fmt(r['delta_pp'])} | "
                         f"{r['generated_tokens_per_item']:.0f} |")
    if res["auroc"]:
        L += ["", "## AUROC of the sample signals for y0 = 1", "",
              "| Setting | Signal | Pooled [CI] | Within-source [CI] |", "|---|---|---:|---:|"]
        for a in res["auroc"]:
            if a["signal"] in ("agreement", "margin"):
                L.append(f"| {a['setting']} | {a['signal']} | {a['pooled']:.3f} [{a['pooled_ci'][0]:.3f}, {a['pooled_ci'][1]:.3f}] | "
                         f"{a['within_source']:.3f} [{a['within_source_ci'][0]:.3f}, {a['within_source_ci'][1]:.3f}] |")
    L += ["", "Checks: " + "; ".join(res["checks"]) + ".", ""]
    (REPO / "reports" / "arr_b1.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- main
def main():
    plan = plan_hash()
    a0 = json.loads(pin(OUT / "a0_common_labels.json").read_text(encoding="utf-8"))
    assert a0["plan_core_sha256"] == plan_core_hash() and not a0["guards"]
    a1 = json.loads(pin(OUT / "a1_policies.json").read_text(encoding="utf-8"))
    labels = list(rows_gz(pin(OUT / "math_common_labels.jsonl.gz")))
    table = {}
    for r in rows(pin(J1 / "artifacts/math/item_evaluations.jsonl")):
        table.setdefault(r["model"], {}).setdefault(r["condition"], {})[r["item_id"]] = r
    pops = A.load_math(table, labels)
    status, runs, checks = {}, {}, []
    for model in MODELS_MATH:
        got, why = {}, None
        for k, stream, _ in STREAMS:
            if model == "4B" and k == 5:
                continue   # the stored samples
            recs, info = sample_run(model, "test", k, stream, pops[model]["ids"])
            if recs is None:
                why = f"k{k} {stream}: {info}"
                break
            got[k] = (recs, info)
        status[model] = "complete" if why is None else why
        if why is None:
            runs[model] = got
    eq = Equivalence()
    try:
        # phase 1: candidates of every new sample, gold-blind, frozen and hashed before any gold is read
        extractions = []
        for model, got in runs.items():
            P = pops[model]
            for k, stream, idx in STREAMS:
                if k not in got:
                    continue
                recs = got[k][0]
                for j, i in enumerate(P["ids"]):
                    r = recs[i]
                    assert r["prompt_tokens"] == P["a0_prompt"][j], ("sample prompt differs from the a0 prompt", model, i)
                    for slot, s in zip(idx, r["samples"]):
                        c = candidate(s["text"], s["finish_reason"], s["out_tokens"], eq)
                        extractions.append({"model": model, "item_id": i, "index": slot, "stream": stream,
                                            **{f: c[f] for f in ("pred", "status", "raw_extraction_status", "out_tokens", "finish_reason")}})
        frozen = OUT / "b1_extractions.jsonl.gz"
        write_rows_gz(frozen, extractions)
        frozen_hash = sha(frozen)
        print(f"extractions frozen: {len(extractions)} rows, sha256 {frozen_hash[:16]}", flush=True)
        # phase 2: scoring against gold
        gold = {r["item_id"]: r["gold"] for p in (pin(REPO / "data/processed/dnat.jsonl"),
                                                  pin(REPO / "data/processed/ext/dnat_ext.jsonl")) for r in rows(p)}
        for e in extractions:
            e["correct"] = int(e["status"] == "ok" and eq(e["pred"], gold[e["item_id"]]))
        write_rows_gz(OUT / "b1_labels.jsonl.gz", extractions)
        res = {"plan_sha256_lf": plan, "plan_core_sha256": plan_core_hash(), "extractions_sha256_before_gold": frozen_hash,
               "status": status, "runs": {m: {k: v[1] for k, v in got.items()} for m, got in runs.items()},
               "k8": {}, "a1": {}, "family": [], "auroc": [], "checks": checks}
        eq2_rows, curve_rows = [], []
        per = unique(rows(pin(ACL / "artifacts/math_vote/per_item.jsonl")))
        sr = unique(rows(pin(ACL / "artifacts/math_vote/same_rule_per_item.jsonl")))
        for model in runs:
            P = pops[model]
            S8 = attach(P, model, eq, labels=[e for e in extractions if e["model"] == model], per=per)
            if model == "4B":
                agree = np.array([sum(eq(c["pred"], P["a0_pred"][j]) for c in S8["cands"][j][:5]) / 5 for j in range(P["n"])])
                assert np.array_equal(agree, np.array([sr[i]["gold_blind_agreement"] for i in P["ids"]])), "4B agreement"
                assert np.array_equal(agree, P["sig"]["agreement"])
                checks.append("Qwen3-4B: the first five of eight samples give A1's groups (with and without a0) and the "
                              "ACL gold-blind agreement exactly")
            boot = Boot(P["canonical"])
            pol = A.policies(P)
            k8, draws = k8_policies(P, S8)
            # the five-sample rules computed both ways must agree
            for name, other in (("V1", "V1"), ("V3", "V3"), ("V5", "V5"), ("VOTE+a0 k3", "VOTE+a0 k3"), ("VOTE+a0 k5", "VOTE+a0 k5"),
                                ("STOP-decided K5", "STOP-decided"), ("STOP-2 K5", "STOP-2"), ("GATE-a0 K5", "GATE-a0")):
                assert np.array_equal(k8[name].y, pol[other].y) and np.allclose(k8[name].gen, pol[other].gen), (model, name)
            if model == "4B":
                old = a1["settings"]["math 4B"]["policies"]
                for name, p in pol.items():
                    assert abs(old[name]["accuracy_pct"] - 100 * p.y.mean()) < 1e-9, ("4B differs from A1", name)
                    assert abs(old[name]["generated_tokens_per_item"] - p.gen.mean()) < 1e-9, ("4B cost differs from A1", name)
                checks.append("Qwen3-4B: every A1 policy recomputed through this script equals a1_policies.json")
            k8_all = {"KEEP": pol["KEEP"], **k8}
            res["k8"][P["setting"]] = {
                "policies": A.describe(P, k8_all, boot), "mean_samples_drawn": draws,
                "contrasts": [{"a": a, "b": b, "difference": boot.ci(k8_all[a].y - k8_all[b].y),
                               "token_difference": float((k8_all[a].gen - k8_all[b].gen).mean())} for a, b in K8_CONTRASTS],
                "sample_status": {f"s{x + 1}": {"valid_pct": 100 * float(S8["valid"][:, x].mean()),
                                                "correct_pct": 100 * float(S8["correct"][:, x].mean()),
                                                "mean_tokens": float(S8["tok"][:, x].mean())} for x in range(KMAX)}}
            if model in NEW_MODELS:
                cons = A.contrasts(P, pol, boot)
                res["a1"][P["setting"]] = {"n": P["n"], "policies": A.describe(P, pol, boot), "contrasts": cons,
                                          "mean_samples_drawn": P.get("mean_draws")}
                # member 4 (ANSWERLESS->F2 - F2 ALL) needs no samples: A1 already reports it for every model
                res["family"] += [{"setting": P["setting"], "member": c["family_member"], "a": c["a"], "b": c["b"],
                                   "difference": c["difference"], "p_boot": c["difference"]["p_boot"]}
                                  for c in cons if c["family_member"] in (1, 2, 3)]
                eq2_rows += A.eq2_table(P, boot, "common")
                curve_rows += A.curves(P, "common")
                res["auroc"] += A.auroc_table(P, boot, "common")
        guards = list(eq.guards)
    finally:
        eq.close()
    assert sha(frozen) == frozen_hash
    guards = guard_check(guards)
    if guards:
        checks.append(f"{len(guards)} scorer guard hit(s), hand-checked and scored not equivalent "
                      "(analysis/arr/reviewed_guards.json; DEVIATIONS D1; plan Amendment 5)")
    checks.append("for every model the k <= 8 code reproduces the five-sample rules of a1a2_replays.py exactly")
    dump(OUT / "b1_policies.json", res)
    if eq2_rows:
        flat = [{k: (v["estimate"] if isinstance(v, dict) else v) for k, v in r.items()} |
                {f"{k}_lo": v["lo"] for k, v in r.items() if isinstance(v, dict)} |
                {f"{k}_hi": v["hi"] for k, v in r.items() if isinstance(v, dict)} for r in eq2_rows]
        with (OUT / "b1_eq2.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(flat)
    buf = io.StringIO()
    if curve_rows:
        w = csv.DictWriter(buf, fieldnames=list(curve_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(curve_rows)
    with (OUT / "b1_curves.csv.gz").open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue().encode("utf-8"))
    dump(OUT / "b1_auroc.json", res["auroc"])
    report(res, eq2_rows)
    outs = ["b1_extractions.jsonl.gz", "b1_labels.jsonl.gz", "b1_policies.json", "b1_curves.csv.gz", "b1_auroc.json"]
    outs += ["b1_eq2.csv"] if eq2_rows else []
    manifest("b1_samples", {"outputs": {p: sha(OUT / p) for p in outs}, "settings_status": status})
    print(json.dumps({"status": status, "checks": checks,
                      "family": [(f["setting"], f["member"], round(f["difference"]["estimate"], 2)) for f in res["family"]]}, indent=1))


if __name__ == "__main__":
    main()
