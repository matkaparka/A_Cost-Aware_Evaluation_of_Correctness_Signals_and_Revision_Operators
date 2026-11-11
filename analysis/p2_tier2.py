"""P2 Tier 2 feasibility tables (design §3.1, Exp 5, Exp 3): results/p2 -> reports/02_pilot_tier2_tables.md.

Exp 5: KL per (lr, tau), the selected tau (rule in src/ttt/pilot.py), lr selection by the ext-1.0 parameter-arm gain
over B1 on the tuning items, and the four feasibility criteria. Exp 3: reachability, screening top 3, confirmation
summary, generic-control validity, alpha = 0 identity, and the main setting chosen by the pre-set rule.
Dev split only. Usage: python analysis/p2_tier2.py
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "02_pilot_tier2_tables.md"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def latest(model, kind):
    reg = {r["run_id"]: r for r in jl(ROOT / "runs.jsonl")}
    runs = [p for p in (ROOT / "results" / "p2" / model).glob(f"p2-{kind}-{model}-*")
            if (p / "manifest.json").exists() and reg.get(p.name, {}).get("status") == "done"]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def exp5(lines, model):
    pilot, serve = latest(model, "ttt-pilot"), latest(model, "ttt-serve")
    lines += [f"### Exp 5 on Qwen3-{model}", ""]
    if not pilot or not serve:
        lines += [f"TODO(missing: Exp 5 pilot or serving run for {model})", ""]
        return None
    pm, sm = json.loads((pilot / "metrics.json").read_text()), json.loads((serve / "metrics.json").read_text())
    rows = jl(serve / "rows.jsonl")
    b1 = {r["item_id"]: r["correct"] for r in rows if r["arm"] == "B1"}
    lines += [f"Tuning items: {pm['n_tune']} ({pm['n_tune_wrong']} initially wrong); KL probe items: {pm['n_kl_items']}.", "",
              "| lr / τ | Mean KL (nats/token) | ext-1.0 parameter arm acc % | B1 acc % | Gain (pp) |", "|---|---|---|---|---|"]
    gains = {}
    for key, kl in sorted(pm["mean_kl"].items()):
        lr, tau = key.split("/")
        ext = [r for r in rows if r["arm"] == "parameter" and r["grounding"] == "ext-1.0"
               and f"{r['lr']:g}" == lr and f"{r['tau']:g}" == tau]
        if not ext:
            continue
        acc = sum(r["correct"] for r in ext) / len(ext)
        base = sum(b1[r["item_id"]] for r in ext) / len(ext)
        gains[(lr, tau)] = acc - base
        lines.append(f"| {lr} / {tau} | {kl:.4f} | {pct(acc)} | {pct(base)} | {pct(acc - base)} |")
    tau_sel = f"{pm['tau_selected']:g}"
    at_tau = {lr: g for (lr, tau), g in gains.items() if tau == tau_sel}
    lr_sel = min(at_tau, key=lambda lr: (-at_tau[lr], float(lr))) if at_tau else None
    gain_sel = at_tau.get(lr_sel)
    crit = {"zero-step = B1 exactly (vLLM, greedy)": sm["zero_step_equals_B1"] == 1.0,
            "reset restores the logits": pm["reset_logits_exact"],
            "realized norm within 1% of τ": pm["realized_norm_max_rel_err"] <= 0.01,
            "ext-1.0 parameter arm > B1 on dev (selected config)": gain_sel is not None and gain_sel > 0}
    lines += ["", f"Selected τ = {tau_sel} (rule: largest τ with mean KL ≤ 0.05 in both lr), lr = {lr_sel} (ext-1.0 gain).", "",
              "| Feasibility criterion | Value | Pass |", "|---|---|---|",
              f"| zero-step = B1 exactly | {pct(sm['zero_step_equals_B1'])} % identical | {crit['zero-step = B1 exactly (vLLM, greedy)']} |",
              f"| reset restores logits | {pm['reset_logits_exact']} | {crit['reset restores the logits']} |",
              f"| realized norm | max rel. error {pm['realized_norm_max_rel_err']:.2e} | {crit['realized norm within 1% of τ']} |",
              f"| ext-1.0 parameter arm > B1 | gain {pct(gain_sel)} pp | {crit['ext-1.0 parameter arm > B1 on dev (selected config)']} |",
              # Invariant added after P2 (DECISIONS 2026-11-04): runs before the fix did not measure it.
              f"| same update twice = identical adapter (invariant) | "
              f"{'not measured (run predates the check)' if pm.get('update_repeat_identical') is None else pct(pm['update_repeat_identical']) + ' % identical'} | "
              f"{pm.get('update_repeat_identical') == 1.0} |",
              "", f"**Exp 5 feasibility: {'ENABLED' if all(crit.values()) else 'DISABLED'}** (all four criteria must pass; "
              "enabling also needs power ≥ 0.8 at freeze).", ""]
    lines += ["| Grounding (selected τ, lr) | Parameter arm acc % | Parameter + context acc % | n |", "|---|---|---|---|"]
    for g in ("ext-1.0", "self", "perm-self", "shuffled", "random"):
        sel = [r for r in rows if r.get("grounding") == g and f"{r['tau']:g}" == tau_sel and (g == "random" or f"{r['lr']:g}" == lr_sel)]
        pa = [r["correct"] for r in sel if r["arm"] == "parameter"]
        pc = [r["correct"] for r in sel if r["arm"] == "parameter+context"]
        if pa:
            lines.append(f"| {g} | {pct(sum(pa) / len(pa))} | {pct(sum(pc) / len(pc)) if pc else '—'} | {len(pa)} |")
    ctx = defaultdict(list)
    for r in rows:
        if r["arm"] == "context":
            ctx[r["verdict"]].append(r["correct"])
    lines += ["", f"Context arm (no update): verdict correct → {pct(sum(ctx[True]) / len(ctx[True]) if ctx[True] else None)} %, "
              f"verdict incorrect → {pct(sum(ctx[False]) / len(ctx[False]) if ctx[False] else None)} %; anomalies: {pm['anomalies']}.", ""]
    return {"pilot": pilot.name, "serve": serve.name, "tau": tau_sel, "lr": lr_sel, "feasible": all(crit.values()),
            "update_repeat_identical": pm.get("update_repeat_identical")}


def exp3(lines, model):
    run = latest(model, "steer-pilot")
    lines += [f"### Exp 3 on Qwen3-{model}", ""]
    if not run:
        lines += [f"TODO(missing: Exp 3 steering pilot for {model})", ""]
        return None
    m = json.loads((run / "metrics.json").read_text())
    lines += [f"Layers screened: {m['layers']}; reachable settings: {m['n_reachable']}; α = 0 reproduces the unsteered "
              f"tokens: {m['alpha0_identical']}.", f"Markers (from dev F0 revisions): correction {m['correction_markers']}, "
              f"commit {m['commit_markers']}.", "",
              "| Run | W→R on D-inj % | R→W on D-nat % | Answer-change rate on D-inj % |", "|---|---|---|---|"]
    for name, s in m["summary"].items():
        lines.append(f"| {name} | {pct(s['w2r_dinj'])} | {pct(s['r2w_dnat'])} | {pct(s['change_rate'])} |")
    lines += ["", "| Top setting | Layer | α | Proxy gain | Mean KL | W→R gain over reference (pp) | R→W rule | Generic valid |",
              "|---|---|---|---|---|---|---|---|"]
    for s in m["selection"]:
        lines.append(f"| {s['direction']} | {s['layer']} | {s['alpha']} | {s['proxy_gain']:.4f} | {s['mean_kl']:.4f} | "
                     f"{pct(s['w2r_gain_over_ref'])} | {s['r2w_increase_ok']} | {s['generic_valid']} |")
    ms = m["main_setting"]
    lines += ["", f"**Main setting:** {'none eligible' if not ms else f'{ms['direction']} at layer {ms['layer']}, α = {ms['alpha']}'}.", ""]
    k = next((k for k, s in enumerate(m["selection"]) if ms and all(s[f] == ms[f] for f in ("layer", "direction", "alpha"))), None)
    return {"run": run.name, "main_setting": ms, "main_run": None if k is None else f"top{k}:{ms['direction']}",
            "controls": None if k is None else {"generic": f"top{k}:generic", "random": [f"top{k}:random{j}" for j in range(5)]},
            "reachable": m["n_reachable"], "alpha0_identical": m["alpha0_identical"]}


def main():
    lines = ["# P2 dev pilot tables: Tier 2 feasibility (generated by analysis/p2_tier2.py; dev split only)", ""]
    sel = {"exp5": {model: exp5(lines, model) for model in ("4B", "1.7B")}, "exp3": {"4B": exp3(lines, "4B")}}
    (ROOT / "analysis" / "out" / "tier2.json").write_text(json.dumps(sel, indent=1))   # read by p2_power_tier2.py
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
