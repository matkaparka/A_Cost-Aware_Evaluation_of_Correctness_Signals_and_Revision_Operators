"""P6 results (CLAUDE.md P6 step 3): for each hypothesis, its outcome category (design §3.2, prereg §5), estimate, CI
and sample status, gathered from the generated analysis files; no new statistics. The wording follows prereg §0 and
§3.4 and is chosen from the category by fixed rules below.
Inputs: analysis/out/06_confirmatory_4B.json (frozen analysis), 06_exploratory_{4B,8B,1.7B}.json, p4_4B_exp5.json,
p4_4B_channel.json, p4_4B_exp4.json, p4_4B_streams.json, p4_{4B,8B}_thinking.json, p4_{1.7B,8B}_tier1.json, runs.jsonl.
-> reports/06_results.md and analysis/out/06_results.json
Usage: python analysis/p6_results.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from stats import holm  # noqa: E402

OUT = ROOT / "analysis" / "out"
DELTA, DELTA_W = 3.0, 10.0   # pp (prereg §3, §3.4)
PLANNED = {"H1a": 5052, "H1b": 5052, "H2a": 5052, "H3a-role": 1635, "H3a-label": 1635, "H3b": 1547}   # prereg §3.1
ESTIMAND = {
    "H1a": "Δacc(F2 − F1), D-nat pooled; equivalence within ±δ (TOST)",
    "H1b": "acc(1.0) − acc(0.5) on the linear trend in p under F3(p); one-sided GEE slope",
    "H2a": "AUROC(frozen probe) − AUROC(verbal s) for a0 correctness, × 100; one-sided paired bootstrap",
    "H3a-role": "W→R(user) − W→R(assistant), D-inj, F0; two-sided",
    "H3a-label": "W→R(other) − W→R(self), D-inj, F0; two-sided",
    "H3b": "W→R(main steering) − mean W→R(5 random directions), D-inj own-error items; one-sided",
}


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(v, signed=True):
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def eci(r, scale=100):
    return f"{f(scale * r['est'])} [{f(scale * r['lo'])}, {f(scale * r['hi'])}]"


def tri(v, scale=100):
    return f"{f(scale * v[0])} [{f(scale * v[1])}, {f(scale * v[2])}]"


def rate(v):
    return f"{100 * v[0]:.1f} % [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"


def wording(h, r):
    """Permitted wording per hypothesis and category (prereg §0, §5; design §3.2-3.5)."""
    c, e, lo, hi = r["category"].split(" [")[0], 100 * r["est"], 100 * r["lo"], 100 * r["hi"]
    if h == "H1a":
        ctx = r["context"]
        w = {"supported": f"F2 − F1 lies within ±{DELTA:.0f} pp (TOST) at these revision rates",
             "contradicted": f"F2 − F1 lies outside ±{DELTA:.0f} pp",
             "inconclusive": "equivalence within ±δ is not shown"}[c]
        side = (" The CI excludes 0 (F2 above F1), and lies entirely inside ±δ." if lo > 0 else
                " The CI excludes 0 (F2 below F1), and lies entirely inside ±δ." if hi < 0 else "")
        return (w + f": discordance {100 * ctx['discordance']:.2f} %, W→R F1 {100 * ctx['F1 W->R']:.2f} % / F2 "
                f"{100 * ctx['F2 W->R']:.2f} %, R→W F1 {100 * ctx['F1 R->W']:.2f} % / F2 {100 * ctx['F2 R->W']:.2f} %." + side
                + " Not written as \"critique adds nothing\" or \"no better than 'Wait'\" (prereg §0); F2 has an extra turn, so "
                "this compares two methods.")
    if h == "H1b":
        if c.startswith("supported"):
            return ("Correctness rises with the accuracy p of the simulated verifier, which the model is not told" + (f"; the effect is below δ ({DELTA:.0f} pp)"
                    if e < DELTA else "") + ". The trend comes from the few items whose answer follows the verdict (prereg §3.2).")
        return {"not supported": f"No positive trend of δ/2 size: the CI's upper end is below {DELTA / 2:.1f} pp.",
                "inconclusive": "The trend in p is neither shown nor excluded at δ/2."}[c]
    if h == "H2a":
        base = (f"AUROC probe {r['auroc_probe']:.3f}, verbal {r['auroc_verbal']:.3f}; DeLong one-sided p "
                f"{r['delong_p']:.3g} (check{'; method-sensitive' if 'method-sensitive' in r['category'] else '; same significance'}). "
                "Linear accessibility only, not an information ceiling (prereg §0).")
        return {"supported": "The frozen dev-trained probe ranks a0 correctness better than the verbal score. " + base,
                "not supported": f"No probe advantage of {DELTA:.0f} AUROC points: the CI's upper end is below it. " + base,
                "inconclusive": "A probe advantage is neither shown nor excluded at δ_AUC. " + base}[c]
    if h.startswith("H3a"):
        what = "the user role" if h == "H3a-role" else "the 'other' label"
        ref = "the assistant role" if h == "H3a-role" else "the 'self' label"
        if c.startswith("supported"):
            sign = "raises" if e > 0 else "lowers"
            tail = ("" if e > 0 else " A negative sign is not a replication of the Self-Correction Bench blind spot (prereg §0).")
            return (f"At F0 on D-inj, {what} {sign} W→R relative to {ref}. This is a prompt-label effect; the 2 × 2 does "
                    "not recover generation history, and the blind-spot comparison is the exploratory diagonal." + tail)
        return {"not supported": f"No effect of δ size: the CI lies inside ±{DELTA:.0f} pp. Not written as \"no effect\".",
                "inconclusive": "An effect of δ size is neither shown nor excluded."}[c]
    if h == "H3b":
        neg = (f" The estimate is negative and its CI lies below 0: steering at the main setting lowered W→R relative "
               "to the random directions. The test is one-sided, so the category cannot be \"contradicted\"." if hi < 0 else "")
        return ({"supported": "Steering toward the external configuration raised W→R beyond matched random directions.",
                 "not supported": "Steering toward the external configuration did not raise W→R beyond matched random "
                                  f"directions by δ: the CI's upper end is below {DELTA:.0f} pp.",
                 "inconclusive": "An increase of δ size is neither shown nor excluded."}[c] + neg
                + " Random control only (the generic control was invalid on dev): conclusions are limited to \"beyond "
                  "random perturbation\"; a null is not support for essay §一 (prereg §0, §7).")
    return ""


def no_test(est, lo, hi, line, name):
    """prereg §3.4 wording for the estimation-only H2b parts."""
    if lo > line:
        return f"contradicted by the estimate (no test)" if name == "i" else "consistent with H2b(ii) (no test)"
    if hi < line:
        return "consistent with H2b(i) (no test)" if name == "i" else "contradicted by the estimate (no test)"
    return "inconclusive (no test)"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    conf = load("06_confirmatory_4B")
    if conf is None:
        raise SystemExit("run analysis/p6_confirmatory.py first")
    M = conf["main"]
    ex = {m: load(f"06_exploratory_{m}") for m in ("4B", "8B", "1.7B")}
    e5, ch, e4, sm = load("p4_4B_exp5"), load("p4_4B_channel"), load("p4_4B_exp4"), load("p4_4B_streams")
    th = {m: load(f"p4_{m}_thinking") for m in ("4B", "8B")}
    rep = {m: load(f"p4_{m}_tier1") for m in ("1.7B", "8B")}
    access = next(json.loads(l) for l in open(ROOT / "runs.jsonl", encoding="utf-8") if "first-test-access-4B" in l)

    res = {"family": {}, "estimation_only": {}, "exploratory": {}}
    L = ["# P6 results (generated by analysis/p6_results.py)", "",
         "Primary model Qwen3-4B, non-thinking, test split. D-nat population: GSM8K test + MATH test pooled as sampled "
         "(original pool + MATH expansion; prereg §2). Estimates in pp unless stated (H2a: AUROC difference × 100); 95 % "
         "cluster-bootstrap CIs, not adjusted for multiplicity; Holm over the fixed family of 6 at α = 0.05, computed once "
         "by the frozen `analysis/p6_confirmatory.py` ([06_confirmatory_4B.md](06_confirmatory_4B.md)).",
         f"First post-feedback test analysis: {access['start']} (`runs.jsonl` {access['run_id']}).", "",
         "## Confirmatory family (Holm over 6)", "",
         "| Hypothesis | Estimand | n (planned) | Estimate [95 % CI] | p | p (Holm) | Outcome | Sample |",
         "|---|---|---|---|---|---|---|---|"]
    for h in ("H1a", "H1b", "H2a", "H3a-role", "H3a-label", "H3b"):
        r = M[h]
        sample = "complete" if r["n"] >= PLANNED[h] else r.get("status_note", "partial")
        res["family"][h] = {k: r.get(k) for k in ("n", "est", "lo", "hi", "p", "p_holm", "category")} | {"sample": sample}
        L.append(f"| {h} | {ESTIMAND[h]} | {r['n']} ({PLANNED[h]}) | {eci(r)} | {r['p']:.3g} | {r['p_holm']:.3g} | "
                 f"**{r['category']}** | {sample} |")
    L += ["", "Wording per hypothesis (prereg §0, §5):", ""]
    for h in ("H1a", "H1b", "H2a", "H3a-role", "H3a-label", "H3b"):
        w = wording(h, M[h])
        res["family"][h]["wording"] = w
        L.append(f"- **{h} — {M[h]['category']}.** {w}")
    sh = M["strict_holm"]
    L += ["", "Sensitivity analyses ([06_confirmatory_4B.md](06_confirmatory_4B.md)):",
          "- Strict reading of design §7.4 (H1b estimation-only), Holm over the other five: "
          + ", ".join(f"{h} {p:.3g}" for h, p in sh.items()) + ".",
          "- Second-scorer (v2) labels and the drop of items whose a0 is truncated or unparsed: estimates and raw p per "
          "member in the sensitivity table; Holm-level significance under each:"]
    for name in ("v2", "a0_ok"):
        S = conf[name]
        fam = [h for h in res["family"] if not h.startswith("_")]
        adj = dict(zip(fam, holm([S[h]["p"] if S[h].get("p") is not None else 1.0 for h in fam]).tolist()))
        same = all((adj[h] < 0.05) == (M[h]["p_holm"] < 0.05) for h in adj)
        res["family"].setdefault("_sensitivity", {})[name] = {"holm": adj, "same_significance": same}
        L.append(f"  - {name}: " + ", ".join(f"{h} {adj[h]:.3g}" for h in res["family"] if h in adj)
                 + (" — every member keeps its significance status." if same else " — **a member changes significance.**"))

    # ---- estimation-only
    L += ["", "## Estimation-only (pre-registered; CI, no claim)", "", "| Quantity | n | Estimate [95 % CI] | Reading |",
          "|---|---|---|---|"]
    r = M["H1d"]
    L.append(f"| H1d: Δacc(F5 − mean F3-perm), D-nat pooled | {r['n']} | {eci(r)} | allocation of self-verdict errors; "
             "|Δacc| is bounded by the F5/F3-perm discordance (below about 1 % on dev; prereg §3.4) |")
    res["estimation_only"]["H1d"] = r
    for part, key, line in (("i", "H2b(i) F2 - F0", DELTA_W), ("ii", "H2b(ii) F3(1.0) - F0", 0.0)):
        r = M[key]
        read = no_test(100 * r["est"], 100 * r["lo"], 100 * r["hi"], line, part)
        res["estimation_only"][key] = r | {"reading": read}
        L.append(f"| {key} (initially wrong, top quintile of probe P(correct)) | {r['n']} | {eci(r)} | {read}; "
                 f"reference line {line:.0f} pp |")
    if e5:
        h4 = e5["H4a"]["est_lo_hi"]
        L.append(f"| H4a: acc(self-verdict update) − mean acc(perm-self updates), parameter arm | {e5['H4a']['n']} | "
                 f"{tri(h4)} | allocation of self-verdict errors under one update rule; not reflexive (prereg §0) |")
        res["estimation_only"]["H4a"] = e5["H4a"]

    # ---- exploratory
    L += ["", "## Exploratory (prereg §3.3; no claim)", ""]
    if ch:
        c1 = ch["all"]["1C"]
        L.append(f"- **H1c, channel** (n {ch['n_1C']}; [04_p4_4B_channel.md](04_p4_4B_channel.md)): range of accuracy "
                 f"across the three channels at F3(0.9) {tri(c1['range at F3(0.9)'])}, against F3(0.9) − F3(0.7) "
                 f"{tri(c1['F3(0.9) - F3(0.7), user'])}. Sampled and verifier-mask robustness: same file.")
    if e5:
        sp = e5["H4b"]
        L.append(f"- **H4b, spillover** ([04_p4_4B_exp5.md](04_p4_4B_exp5.md)): mean change in gold log-prob on the "
                 f"spillover set after self-verdict updates {sp['self']['spillover'][0]:+.4f} "
                 f"[{sp['self']['spillover'][1]:+.4f}, {sp['self']['spillover'][2]:+.4f}] nats; self − mean perm-k "
                 f"{sp['self - mean perm-k']['spillover'][0]:+.4f} [{sp['self - mean perm-k']['spillover'][1]:+.4f}, "
                 f"{sp['self - mean perm-k']['spillover'][2]:+.4f}].")
        d = e5["H4d"]
        L.append(f"- **H4d, self-prediction**: not measured in the base, self and random cells "
                 f"({d['self']['selfpred_no_answer']} of {e5['n_items']} self-predictions gave no answer within the 64-token "
                 "budget; [04_main_runs.md](04_main_runs.md), p4c). Only the context cell is interpretable.")
    if sm:
        k = str(max(int(x) for x in sm["eval"]["self"]))
        accs = {c: sm["eval"][c][k]["acc_greedy"] for c in ("self", "ext", "random", "context")}
        L.append(f"- **H4c, streams** (3 seeds × 200 steps; [04_p4_4B_streams.md](04_p4_4B_streams.md)): greedy dev "
                 f"accuracy at k = {k}, per seed: " + "; ".join(f"{c} " + ", ".join(f"{100 * v:.0f} %" for v in vs)
                                                             for c, vs in accs.items())
                 + f" (base {100 * sm['eval']['self']['0']['acc_greedy'][0]:.0f} %). The self streams collapse into "
                   "repetition; the post hoc description is DEVIATIONS D12. Descriptive only.")
    if e4:
        t = e4["test"]
        rep_n = {m: sum(x["replicated"] for x in v) for m, v in t["confirm"].items()}
        L.append(f"- **Exp 4, patching** (test n {t['n']}, {t['status']}; [04_4B_exp4.md](04_4B_exp4.md)): frozen "
                 "components replicated on test: " + ", ".join(f"{m} {n} of {len(t['confirm'][m])}" for m, n in rep_n.items())
                 + ". lp_gold: label positions, layers " + ", ".join(str(x["layer"]) for x in t["confirm"]["lp_gold"])
                 + "; marker: last position, layers " + ", ".join(str(x["layer"]) for x in t["confirm"]["marker"]) + ".")
    for m, t in th.items():
        if t:
            L.append(f"- **Thinking subset, {m}** (n {t['n_items']}, 2 seeds, weighted; [04_p4_{m}_thinking.md]"
                     f"(04_p4_{m}_thinking.md)): F2 − F1 in thinking mode {tri(t['contrasts']['F2 - F1'])} "
                     "(F1 affected by unclosed think blocks, DEVIATIONS D11).")
    x = ex["4B"]
    if x:
        a1 = x["1A"]["contrasts"]
        L.append(f"- **F2-scratch** ([06_exploratory_4B.md](06_exploratory_4B.md)): F2 − F2-scratch "
                 f"{tri(a1['F2 - F2-scratch'])}, F2-scratch − F1 {tri(a1['F2-scratch - F1'])}. H1b secondary: "
                 f"F3(0.5) − F0 {tri(a1['F3(0.5) - F0 (H1b secondary, predicted <= 0)'])} (predicted ≤ 0).")
        b0 = x["1B"]["D-inj F0"]
        L.append(f"- **1B diagonal and interaction** (D-inj, F0): (user, other) − (assistant, self) "
                 f"{tri(b0['diagonal (user, other) - (assistant, self)'])}; interaction "
                 f"{tri(b0['interaction (u/o - u/s) - (a/o - a/s)'])}. At F1 and F3(1.0), and on D-nat: same file.")
        if "alt_labels" in x:
            al = x["alt_labels"]
            L.append(f"- **Alternative label wording** (n {al['n']}): label contrast, alternative − main wording "
                     f"{tri(al['alternative - main'])}.")
        ts = x["tsui"]
        L.append(f"- **Tsui cell** (D-inj, n {ts['n']}): W→R internal {rate(ts['cells']['internal'])}, external "
                 f"{rate(ts['cells']['external'])}, internal + \"Wait\" {rate(ts['cells']['internal+wait'])}.")
        e0 = x["exp0"]["cells"]
        L.append(f"- **Exp 0, calibration and leakage control** ({x['exp0']['n']} items): accuracy without feedback "
                 f"{rate(e0['no-feedback']['acc'])}; gain ratio at F3(1.0) {e0['F3(1.0)']['gain ratio'][0]:.2f} "
                 f"[{e0['F3(1.0)']['gain ratio'][1]:.2f}, {e0['F3(1.0)']['gain ratio'][2]:.2f}].")
        if "H2a comparators" in x:
            hc = x["H2a comparators"]
            L.append(f"- **H2a comparators**: AUROC of minus answer length {hc['auroc']['minus answer length']:.3f}, of "
                     f"the stratum's dev accuracy {hc['auroc']['stratum dev accuracy']:.3f}; probe − answer length "
                     f"{tri(hc['probe minus']['minus answer length'])}. Per stratum and per MATH level: same file.")
    L += ["- **Not run.** H3c (role/label decodability at content tokens): no activations were cached under the 1B "
          "prompts, and computing them needs the run host. The H2a cross-fitted secondary analysis and the mean token "
          "log-prob comparator: per-token log-probs of a0 were not stored (DEVIATIONS D13)."]

    # ---- replications
    L += ["", "## Replications on 1.7B and 8B (CIs, not tests; prereg §11)", "",
          "| Quantity | 4B (confirmatory run) | 1.7B | 8B |", "|---|---|---|---|"]
    for h in ("H1a", "H1b", "H1d", "H2a", "H2b(i) F2 - F0", "H2b(ii) F3(1.0) - F0", "H3a-role", "H3a-label"):
        cells = [eci(M[h])] + [eci(rep[m]["estimates"][h]) if rep[m] and h in rep[m]["estimates"] else "–"
                               for m in ("1.7B", "8B")]
        L.append(f"| {h} | " + " | ".join(cells) + " |")
    L.append(f"| H2a AUROC probe / verbal | {M['H2a']['auroc_probe']:.3f} / {M['H2a']['auroc_verbal']:.3f} | "
             + " | ".join(f"{rep[m]['estimates']['H2a']['auroc_probe']:.3f} / {rep[m]['estimates']['H2a']['auroc_verbal']:.3f}"
                          if rep[m] else "–" for m in ("1.7B", "8B")) + " |")
    L += ["", "Per-model tables: [04_p4_1.7B_tier1.md](04_p4_1.7B_tier1.md), [04_p4_8B_tier1.md](04_p4_8B_tier1.md); "
          "exploratory replications: [06_exploratory_1.7B.md](06_exploratory_1.7B.md), "
          "[06_exploratory_8B.md](06_exploratory_8B.md)."]

    (OUT / "06_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (ROOT / "reports" / "06_results.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
