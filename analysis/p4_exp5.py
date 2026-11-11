"""Exp 5 per-block estimates on test (H4a estimation-only, prereg §3.4; H4b, H4d and the other Exp 5 quantities
exploratory, prereg §3.1). Estimates and 95 % cluster-bootstrap CIs only (prereg §8). The rules below were fixed in
DEVIATIONS D5 before this script first ran.
-> reports/04_<out>.md and analysis/out/<out>.json

- Unit: the canonical problem (data/processed/dnat.jsonl); `Boot` of analysis/p6_confirmatory.py (10,000 resamples of
  canonical problems, seed 0, one shared set of resamples for every statistic).
- H4a: greedy parameter-arm accuracy (D2), d_i = self_i - mean_k perm_k_i over the K = 3 perm-self updates, all items
  (intention-to-treat: an update with no answer span leaves the base model, prereg §6). Sensitivity (prereg §6 anomaly
  rules): without the items whose self or perm-k update has c > 100.
- Secondary: the H4a contrast on sample accuracy (mean of the 4 samples); its context-arm counterpart (the context
  prompt with the self verdict vs with each perm-k verdict) and parameter+context counterpart; the coupling x
  verdict-source interaction (parameter contrast - context contrast); each cell's accuracy and its difference from B1;
  W->R (initially wrong items) and R->W (initially right items) per cell.
- H4b: per-update spillover (mean over the spillover set of adapted - base gold log-prob, nats; D2), mean per grounding;
  the KL probe and mean |delta log-prob| alongside.
- H4d: share of items whose greedy self-prediction answer is equivalent (src/score/scorer.py `equivalent`) to the
  matching greedy re-answer (base <-> B1, self <-> parameter-self, random <-> parameter-random, context <-> context
  with the self verdict). A missing answer on either side counts as not consistent.
Usage: python analysis/p4_exp5.py --spec configs/p4_spec_4B_test.json --out p4_4B_exp5 [--n-boot 10000]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
from p6_confirmatory import N_BOOT, Boot, jl, wmean  # noqa: E402
from src.score.scorer import equivalent  # noqa: E402

K_PERM = 3
PERM = [f"perm{k}" for k in range(K_PERM)]
TRAINED = ["self", *PERM, "ext", "shuffled"]
GROUNDINGS = TRAINED + ["random"]


def pp(x):
    return "–" if x is None else f"{100 * x:+.1f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.1f} %"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    spec = json.loads((ROOT / a.spec).read_text(encoding="utf-8"))
    run = Path(spec["exp5_run"])
    rows, ups = jl(run / "rows.jsonl"), jl(run / "updates.jsonl")
    metrics = json.loads((ROOT / run / "metrics.json").read_text())
    canon = {r["item_id"]: r["canonical_id"] for r in jl("data/processed/dnat.jsonl")}

    def cell(r):
        return ("context", r["verdict"]) if r["arm"] == "context" else (r["arm"], r["grounding"])
    greedy, samples, preds, status = {}, defaultdict(list), {}, defaultdict(Counter)
    for r in rows:
        key = cell(r) + (r["item_id"],)
        status[cell(r)][r["status"]] += 1
        if r["decode"] == "greedy":
            greedy[key], preds[key] = float(r["correct"]), r["pred"]
        else:
            samples[key].append(float(r["correct"]))
    ids = sorted({r["item_id"] for r in rows if r["arm"] == "B1"})
    first = {}
    for r in rows:
        first.setdefault(r["item_id"], r["correct0"])
    c0 = np.array([first[i] for i in ids], bool)   # a0 correctness
    V = defaultdict(dict)
    for u in ups:
        V[u["item_id"]][u["grounding"]] = u["verdict"]
    boot = Boot(np.array([canon[i] for i in ids]), a.n_boot)

    G = lambda key: np.array([greedy[key + (i,)] for i in ids])
    S = lambda key: np.array([np.mean(samples[key + (i,)]) for i in ids])
    ctx = lambda g: np.array([greedy[("context", V[i][g], i)] for i in ids])   # context prompt with verdict g
    ci = lambda v, mask=None: boot.ci(lambda m: wmean(v, m), mask)

    out = {"spec": a.spec, "run": str(run), "n_items": len(ids), "n_boot": a.n_boot,
           "metrics": {k: metrics.get(k) for k in ("reset_logits_exact", "zero_step_equals_B1", "realized_norm_max_rel_err",
                                                   "anomalies", "adapters_kept", "scorer_guarded")}}

    # H4a and its sensitivity analysis
    d = G(("parameter", "self")) - np.mean([G(("parameter", p)) for p in PERM], 0)
    flagged = {u["item_id"] for u in ups if u.get("anomaly") == "c_gt_100" and u["grounding"] in ["self", *PERM]}
    keep = ~np.isin(ids, sorted(flagged))
    out["H4a"] = {"n": len(ids), "est_lo_hi": ci(d)}
    out["H4a without c > 100 items"] = {"n": int(keep.sum()), "est_lo_hi": ci(d, keep)}

    # secondary contrasts
    ds = S(("parameter", "self")) - np.mean([S(("parameter", p)) for p in PERM], 0)
    dc = ctx("self") - np.mean([ctx(p) for p in PERM], 0)
    dpc = G(("parameter+context", "self")) - np.mean([G(("parameter+context", p)) for p in PERM], 0)
    sec = {"H4a on sample accuracy (mean of 4 samples)": ds,
           "context arm: self verdict - mean perm-k verdict": dc,
           "parameter+context: self - mean perm-k": dpc,
           "coupling x verdict source: parameter contrast - context contrast": d - dc}
    out["secondary"] = {k: {"n": len(ids), "est_lo_hi": ci(v)} for k, v in sec.items()}
    for name, mask in (("initially wrong", ~c0), ("initially right", c0)):
        out["secondary"][f"H4a, {name} items"] = {"n": int(mask.sum()), "est_lo_hi": ci(d, mask)}

    # per cell
    cells = [("B1", None), ("zero-step", "zero"), ("context", True), ("context", False)] + \
            [("parameter", g) for g in GROUNDINGS] + [("parameter+context", g) for g in TRAINED]
    b1 = G(("B1", None))
    label = lambda c: "context (v = correct)" if c == ("context", True) else "context (v = incorrect)" \
        if c == ("context", False) else c[0] if c[1] is None else f"{c[0]} / {c[1]}"
    out["cells"] = {}
    for c in cells:
        y = G(c)
        out["cells"][label(c)] = {"acc": ci(y), "minus_B1": ci(y - b1), "W->R": ci(y, ~c0), "R->W": ci(1 - y, c0),
                                  "sample_acc": float(S(c).mean()) if c != ("zero-step", "zero") else None,
                                  "status": dict(status[c])}

    # H4b spillover, KL probe
    U = defaultdict(dict)
    for u in ups:
        U[u["grounding"]][u["item_id"]] = u
    out["H4b"] = {}
    for g in GROUNDINGS:
        sp = np.array([U[g][i]["spillover"] for i in ids])
        out["H4b"][g] = {"spillover": ci(sp), "kl_probe": float(np.mean([U[g][i]["kl_probe"] for i in ids])),
                         "mean_abs_dlogprob": float(np.mean([U[g][i]["mean_abs_dlogprob"] for i in ids])),
                         "anomalies": dict(Counter(U[g][i].get("anomaly") for i in ids if U[g][i].get("anomaly")))}
    sp_d = np.array([U["self"][i]["spillover"] for i in ids]) - np.mean([[U[p][i]["spillover"] for i in ids] for p in PERM], 0)
    out["H4b"]["self - mean perm-k"] = {"spillover": ci(sp_d)}

    # H4d self-prediction consistency
    match = {"base": lambda i: ("B1", None, i), "self": lambda i: ("parameter", "self", i),
             "random": lambda i: ("parameter", "random", i), "context": lambda i: ("context", V[i]["self"], i)}
    cons, missing = {}, {}
    for of, ref in match.items():
        sp_, re_ = [preds[("selfpred", of, i)] for i in ids], [preds[ref(i)] for i in ids]
        cons[of] = np.array([float(x is not None and y is not None and equivalent(x, y)) for x, y in zip(sp_, re_)])
        missing[of] = sum(x is None for x in sp_)
    out["H4d"] = {of: {"consistency": ci(v), "selfpred_no_answer": missing[of]} for of, v in cons.items()}
    for of in ("self", "random", "context"):
        out["H4d"][f"{of} - base"] = {"consistency": ci(cons[of] - cons["base"])}

    (ROOT / "analysis" / "out" / f"{a.out}.json").write_text(json.dumps(out, indent=1, default=float))

    f3 = lambda t: f"{pp(t[0])} | [{pp(t[1])}, {pp(t[2])}]"
    L = [f"# P4 Exp 5 estimates, {spec['model']} {spec['split']} (generated by analysis/p4_exp5.py)", "",
         f"Run `{run.as_posix()}`, {len(ids)} items ({int((~c0).sum())} initially wrong). Accuracies and differences in "
         f"pp, 95 % cluster-bootstrap CIs ({a.n_boot} resamples, clusters = canonical IDs). **Estimates and CIs only**: "
         "H4a is estimation-only, H4b and H4d are exploratory (prereg §3.1, §3.4, §8).", "",
         "## H4a (parameter arm, greedy): self-verdict update - mean of the K = 3 perm-self updates", "",
         "| Quantity | n | Estimate | 95 % CI |", "|---|---|---|---|",
         f"| H4a | {out['H4a']['n']} | {f3(out['H4a']['est_lo_hi'])} |",
         f"| H4a without c > 100 items (prereg §6 sensitivity) | {out['H4a without c > 100 items']['n']} | "
         f"{f3(out['H4a without c > 100 items']['est_lo_hi'])} |"]
    L += [f"| {k} | {v['n']} | {f3(v['est_lo_hi'])} |" for k, v in out["secondary"].items()]
    L += ["", "## Cells (greedy unless noted)", "",
          "| Cell | Acc | 95 % CI | − B1 (pp) | 95 % CI | W->R | R->W | Sample acc | Truncated rows |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in out["cells"].items():
        acc, dB = v["acc"], v["minus_B1"]
        L.append(f"| {k} | {pct(acc[0])} | [{pct(acc[1])}, {pct(acc[2])}] | {pp(dB[0])} | [{pp(dB[1])}, {pp(dB[2])}] | "
                 f"{pct(v['W->R'][0])} | {pct(v['R->W'][0])} | {pct(v['sample_acc'])} | {v['status'].get('truncated', 0)} |")
    L += ["", "## H4b: spillover and update diagnostics (exploratory)", "",
          "Spillover: mean over the 50 spillover items of the change in gold-answer log-prob (adapted − base, nats per "
          "token; D2), averaged over updates.", "",
          "| Grounding | Spillover | 95 % CI | Mean KL probe (nats/token) | Mean abs Δ log-prob | Anomalies |",
          "|---|---|---|---|---|---|"]
    for g, v in out["H4b"].items():
        s = v["spillover"]
        L.append(f"| {g} | {s[0]:+.4f} | [{s[1]:+.4f}, {s[2]:+.4f}] | "
                 + (f"{v['kl_probe']:.4f} | {v['mean_abs_dlogprob']:.4f} | {v['anomalies'] or '–'} |" if "kl_probe" in v else "– | – | – |"))
    L += ["", "## H4d: self-prediction consistency (exploratory)", "",
          "| Self-prediction of | Consistent | 95 % CI | No answer in the self-prediction |", "|---|---|---|---|"]
    for of, v in out["H4d"].items():
        s = v["consistency"]
        val = (f"{pct(s[0])} | [{pct(s[1])}, {pct(s[2])}]" if "-" not in of else f"{pp(s[0])} pp | [{pp(s[1])}, {pp(s[2])}]")
        L.append(f"| {of} | {val} | {v.get('selfpred_no_answer', '–')} |")
    m = out["metrics"]
    L += ["", "## Run invariants (from metrics.json)", "",
          f"- reset restores the logits: {m['reset_logits_exact']}; zero-step = B1: {pct(m['zero_step_equals_B1'])} "
          f"identical; realized-norm max relative error {m['realized_norm_max_rel_err']:.2e}.",
          f"- anomalies (updates): {m['anomalies']}; adapters kept: {m['adapters_kept']}; scorer guard: "
          f"{len(m['scorer_guarded'])} call(s)."]
    (ROOT / "reports" / f"04_{a.out}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
