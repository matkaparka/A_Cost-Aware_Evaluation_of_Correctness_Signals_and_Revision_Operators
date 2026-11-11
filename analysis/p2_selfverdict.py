"""Self-verdict threshold on dev and its effect on H1d power (design §5.7: "0.5, or a dev-chosen threshold that
maximizes balanced accuracy", fixed in the prereg). results/p1 + results/p2 -> reports/02_selfverdict_tables.md and
analysis/out/selfverdict.json.

1. Verbal score s = log P(yes) - log P(no) on dev (the log-odds of P(yes)_norm, without float rounding near 1).
   Threshold t*: the midpoint between adjacent sorted scores that maximizes balanced accuracy (ties: the lowest).
2. Verdict table (initially right / wrong x verdict) at 0.5 and at t*.
3. H1d on the dev pilot, re-assembled: Exp 1A conditions that share a verdict value share one generation (src/exp/
   runner.py), so each item's outcome under "correct" and under "incorrect" can be read from the F3(p), F5 and
   F3-perm rows. F5(t) and its stratum permutation (dataset x initial correctness, as in §5.8) are re-assembled
   from that table. An outcome never observed for an (item, verdict) cell is drawn from its stratum's observed rate
   (counted and reported). Discordance and effect are averaged over 200 permutations, then fed into the same power
   simulation as analysis/power.py (test n = 1,455; Holm step alpha / 6).
Dev split only. Usage: python analysis/p2_selfverdict.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from power import ALPHA, DELTA, one_sided_paired, pattern_probs, simulate, tost  # noqa: E402

OUT = ROOT / "reports" / "02_selfverdict_tables.md"
PERMS, SEED, N_TEST, FAMILY = 200, 0, 1455, 6


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def score(r):
    if r["p_yes"] is None or r["p_no"] is None:
        return None
    return math.log(max(r["p_yes"], 1e-300)) - math.log(max(r["p_no"], 1e-300))


def balanced_acc(s, y, t):
    pred = s >= t                      # verdict "correct"
    tpr = (pred & y).sum() / max(1, y.sum())
    tnr = (~pred & ~y).sum() / max(1, (~y).sum())
    return (tpr + tnr) / 2


def model_tables(model, rng):
    init = {r["item_id"]: r for r in jl(ROOT / "results" / "p1" / model / f"p1-initial-{model}-d2ebdea" / "initial.jsonl")}
    verbal = [r for r in jl(ROOT / "results" / "p1" / model / f"p1-verbal-{model}-d2ebdea" / "verbal.jsonl") if r["split"] == "dev"]
    items = [(r["item_id"], r["dataset"], score(r), init[r["item_id"]]["correct"]) for r in verbal]
    scored = [it for it in items if it[2] is not None]
    s = np.array([it[2] for it in scored])
    y = np.array([it[3] for it in scored], bool)
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y, s)
    u = np.unique(s)
    cands = (u[:-1] + u[1:]) / 2
    ba = np.array([balanced_acc(s, y, t) for t in cands])
    t_star = float(cands[int(np.argmax(ba))])
    t_half = math.log(0.5) - math.log(0.5)          # P(yes)_norm >= 0.5  <=>  s >= 0

    # (item, verdict) -> outcome, from every 1A verdict condition
    run = ROOT / "results" / "p2" / model / f"p2-exp01-dev-{model}-f7e5011"
    cell, clash = {}, 0
    for r in jl(run / "rows.jsonl"):
        if r["exp"] == "1A" and r.get("verdict") is not None:
            k = (r["item_id"], bool(r["verdict"]))
            if k in cell and cell[k] != r["correct"]:
                clash += 1
            cell[k] = r["correct"]
    meta = {it[0]: (it[1], it[3]) for it in items}
    rate = {}
    for (i, v), c in cell.items():
        rate.setdefault((meta[i][0], meta[i][1], v), []).append(c)
    rate = {k: float(np.mean(v)) for k, v in rate.items()}

    def verdicts(t):
        return {it[0]: (it[2] is not None and it[2] >= t) for it in items}

    def h1d(t):
        v_self = verdicts(t)
        ids = sorted(v_self)
        missing = sum((i, v_self[i]) not in cell for i in ids)
        strata = {}
        for i in ids:
            strata.setdefault(meta[i], []).append(i)
        disc, eff = [], []
        conc_right, conc_n = 0, 0
        for _ in range(PERMS):
            v_perm = {}
            for members in strata.values():
                vs = [v_self[i] for i in members]
                rng.shuffle(vs)
                v_perm.update(zip(members, vs))

            def out(i, v):
                return cell[(i, v)] if (i, v) in cell else bool(rng.random() < rate.get((*meta[i], v), 0.0))
            x = np.array([out(i, v_self[i]) for i in ids], int)
            z = np.array([out(i, v_perm[i]) for i in ids], int)
            disc.append(((x == 1) & (z == 0)).mean() + ((x == 0) & (z == 1)).mean())
            eff.append(((x == 1) & (z == 0)).mean() - ((x == 0) & (z == 1)).mean())
            conc_right += ((x == 1) & (z == 1)).sum()
            conc_n += (x == z).sum()
        flagged = {"right": sum(not v_self[i] for i in ids if meta[i][1]), "wrong": sum(not v_self[i] for i in ids if not meta[i][1])}
        return {"threshold_s": t, "incorrect_verdicts": flagged, "missing_cells": missing, "disc": float(np.mean(disc)),
                "effect": float(np.mean(eff)), "concordant_right_share": conc_right / max(1, conc_n)}

    res = {"model": model, "n_dev": len(items), "n_unscored": len(items) - len(scored), "auroc_verbal_dev": float(auroc),
           "t_star_s": t_star, "t_star_p_yes_norm": 1 / (1 + math.exp(-t_star)), "balanced_acc_t_star": float(ba.max()),
           "balanced_acc_half": float(balanced_acc(s, y, t_half)), "cell_clashes": clash,
           "h1d": {"0.5": h1d(t_half), "t*": h1d(t_star)}}
    for key, h in res["h1d"].items():
        rows = []
        # The effect cannot exceed the discordance, so scenarios are fractions of it; the TOST column is the power of
        # an equivalence reading (|effect| < delta) of the same contrast, for comparison only.
        two_sided = lambda a, b: min(1.0, 2 * min(one_sided_paired(a, b, 0, "greater"), one_sided_paired(a, b, 0, "less")))
        for dmult in (0.5, 1.0, 1.5):
            d = max(h["disc"] * dmult, 1e-3)
            for frac in (0.25, 0.5, 1.0):
                e = frac * d
                probs = pattern_probs(d, e, h["concordant_right_share"])
                ps = simulate(two_sided, N_TEST, probs)
                pt = simulate(lambda a, b: tost(a, b, DELTA), N_TEST, probs)
                rows.append({"disc": d, "effect": e, "power_holm": float((ps < ALPHA / FAMILY).mean()),
                             "power_tost_holm": float((pt < ALPHA / FAMILY).mean())})
        h["power"] = rows
    return res


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def main():
    rng = np.random.default_rng(SEED)
    out = {m: model_tables(m, rng) for m in ("4B", "1.7B")}
    (ROOT / "analysis" / "out" / "selfverdict.json").write_text(json.dumps(out, indent=1))
    lines = ["# P2 self-verdict threshold and H1d power (generated by analysis/p2_selfverdict.py; dev split only)", "",
             "Verbal score s = log P(yes) − log P(no). Threshold t* maximizes dev balanced accuracy (design §5.7). H1d "
             "is re-assembled from the dev 1A rows (one generation per item × verdict value); unobserved cells are "
             f"imputed from their stratum's rate. Power: two-sided paired test, n = {N_TEST}, level α/{FAMILY} (first "
             f"Holm step), {PERMS} permutations.", ""]
    for m, r in out.items():
        lines += [f"## Qwen3-{m}", "",
                  f"Dev items {r['n_dev']} ({r['n_unscored']} without a verbal score); verbal AUROC on dev {r['auroc_verbal_dev']:.3f}; "
                  f"t* = {r['t_star_s']:.2f} (P(yes)_norm ≥ {r['t_star_p_yes_norm']:.6f}); balanced accuracy {r['balanced_acc_t_star']:.3f} "
                  f"at t* vs {r['balanced_acc_half']:.3f} at 0.5; outcome clashes between conditions sharing a verdict: {r['cell_clashes']}.", "",
                  "| Threshold | 'Incorrect' verdicts on right / wrong items | Imputed cells | F5 vs perm discordance % | Effect (pp) |",
                  "|---|---|---|---|---|"]
        for key, h in r["h1d"].items():
            lines.append(f"| {key} | {h['incorrect_verdicts']['right']} / {h['incorrect_verdicts']['wrong']} | {h['missing_cells']} | "
                         f"{pct(h['disc'])} | {pct(h['effect'])} |")
        lines += ["", "| Threshold | Discordance % (scenario) | True effect (pp) | Two-sided power at α/6 | TOST (±δ) power at α/6 |",
                  "|---|---|---|---|---|"]
        for key, h in r["h1d"].items():
            for row in h["power"]:
                lines.append(f"| {key} | {pct(row['disc'])} | {pct(row['effect'])} | {row['power_holm']:.2f} | {row['power_tost_holm']:.2f} |")
        lines.append("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
