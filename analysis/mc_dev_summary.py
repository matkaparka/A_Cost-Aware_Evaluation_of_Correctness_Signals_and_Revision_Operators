"""MMLU-Pro dev pilot summary for the validation addendum (docs/planning/validation_v2/plan.md §2.2; DEVIATIONS D18).
Dev only; written before any test output of the new task exists.

Per model (4B, T8B): a0 accuracy, initially wrong count and parse status, by category; per condition the net accuracy
change, W->R and R->W against a0; the share of items whose final correctness differs between F2 and F0, F1 and F0,
F3(1.0) and F0 (discordance, which bounds what the test can resolve); AUROC for a0 correctness of the verbal score W0
(and a dev-selected variant if present), the dev probe (in-sample on dev, so optimistic), the frozen math probe,
agreement with a0 and answer length. Projected precision on test: the half-width of a 95 % interval for a paired
difference in accuracy, sqrt(discordance / n_test) * 1.96 (independent items; the cluster structure widens it a
little), and for W->R among the projected wrong test items.
Inputs: the session-A run directories under results/v2/fu-dev/<model>/ (initial, consist, hf, conds).
-> reports/v2_01_mc_dev.md, analysis/out/v2_01_mc_dev.json
Usage: python analysis/mc_dev_summary.py
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import mmlupro as M  # noqa: E402

BASE = ROOT / "results" / "v2" / "fu-dev"
CONDS = ["F0", "F1", "F2", "F2-scratch", "F3(1.0)", "F3(0.5)"]


def jl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def one(pattern, model):
    hits = sorted((BASE / model).glob(pattern))
    return hits[-1] if hits else None


def auroc(y, s):
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])) if 0 < y[ok].sum() < ok.sum() else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    n_test = sum(1 for it in M.load_items("test"))
    out = {"n test items": n_test, "models": {}}
    for model in ("4B", "T8B"):
        di, dc, dh, dk = (one(p, model) for p in ("fu-dev-mc-initial-dev-*", "fu-dev-mc-conds-dev-*", "fu-dev-mc-hf-*",
                                                   "fu-dev-mc-consist-dev-*"))
        if di is None:
            out["models"][model] = None
            continue
        init = {r["item_id"]: r for r in jl(di / "initial.jsonl")}
        ids = sorted(init)
        y = np.array([bool(init[i]["correct"]) for i in ids], int)
        cat = np.array([init[i]["category"] for i in ids])
        o = {"n dev": len(ids), "a0 accuracy": float(y.mean()), "initially wrong": int((1 - y).sum()),
             "status": {s: sum(init[i]["status"] == s for i in ids) for s in ("ok", "fallback", "unparsed", "truncated")},
             "by category": {c: {"n": int((cat == c).sum()), "a0 accuracy": float(y[cat == c].mean())} for c in sorted(set(cat))},
             "mean a0 tokens": float(np.mean([init[i]["out_tokens"] for i in ids]))}
        proj_wrong = (1 - y.mean()) * n_test
        o["projected wrong test items"] = proj_wrong
        if dc is not None:
            rows = defaultdict(dict)
            for r in jl(dc / "rows.jsonl"):
                rows[r["item_id"]][r["cond"]] = r
            after = {c: np.array([bool(rows[i][c]["correct"]) for i in ids], int) for c in CONDS if all(c in rows[i] for i in ids)}
            o["conditions"] = {c: {"net change": float((a - y).mean()), "W->R": float(a[y == 0].mean()), "R->W": float(1 - a[y == 1].mean()),
                                   "not ok": float(np.mean([rows[i][c]["status"] not in ("ok", "fallback") for i in ids]))}
                               for c, a in after.items()}
            disc = {}
            for c1, c2 in (("F2", "F0"), ("F1", "F0"), ("F3(1.0)", "F0"), ("F2", "F1"), ("F2", "F2-scratch")):
                if c1 in after and c2 in after:
                    dd = float((after[c1] != after[c2]).mean())
                    dw = float((after[c1][y == 0] != after[c2][y == 0]).mean())
                    disc[f"{c1} vs {c2}"] = {"discordance": dd, "projected half-width, accuracy (pp)": 100 * 1.96 * math.sqrt(dd / n_test),
                                            "discordance among wrong": dw,
                                            "projected half-width, W->R (pp)": 100 * 1.96 * math.sqrt(dw / max(proj_wrong, 1))}
            o["discordance"] = disc
        sig = {}
        if dh is not None:
            vb = {r["item_id"]: r for r in jl(dh / "verbal.jsonl")}
            for v in [k for k in next(iter(vb.values())) if k.startswith("W")]:
                sig[f"verbal {v}"] = auroc(y, [vb[i][v]["s"] if vb[i][v]["s"] is not None else np.nan for i in ids])
            ps = {r["item_id"]: r for r in jl(dh / "probe_scores.jsonl")}
            sig["dev probe (in-sample)"] = auroc(y, [ps[i]["p_task_probe"] if ps[i]["p_task_probe"] is not None else np.nan for i in ids])
            sig["math probe (unchanged)"] = auroc(y, [ps[i]["p_math_probe"] for i in ids])
            sel = json.loads((dh / "probe_selection.json").read_text(encoding="utf-8"))["selected"]
            o["dev probe selection"] = sel
        if dk is not None:
            ag = {r["item_id"]: r for r in jl(dk / "consistency.jsonl")}
            sig["agreement with a0"] = auroc(y, [ag[i]["agreement"] for i in ids])
            o["mean sample tokens (5 samples)"] = float(np.mean([ag[i]["sample_out_tokens"] for i in ids]))
        sig["answer length (shorter = correct)"] = auroc(y, [-init[i]["out_tokens"] for i in ids])
        o["signal AUROC"] = sig
        out["models"][model] = o
    (ROOT / "analysis" / "out" / "v2_01_mc_dev.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    L = ["# MMLU-Pro dev pilot (generated by analysis/mc_dev_summary.py)", "",
         f"Dev only, for the validation addendum (DEVIATIONS D18). Test pool: {n_test} items.", ""]
    for model, o in out["models"].items():
        if o is None:
            L += [f"## {model}: no dev initial run", ""]
            continue
        L += [f"## {model}", "",
              f"Dev n {o['n dev']}, a0 accuracy {100 * o['a0 accuracy']:.1f} %, initially wrong {o['initially wrong']}, "
              f"status {o['status']}, mean a0 tokens {o['mean a0 tokens']:.0f}; projected wrong test items "
              f"{o['projected wrong test items']:.0f}.", "",
              "| Category | n | a0 accuracy |", "|---|---|---|"]
        L += [f"| {c} | {v['n']} | {100 * v['a0 accuracy']:.1f} % |" for c, v in o["by category"].items()]
        if "conditions" in o:
            L += ["", "| Condition | Net change (pp) | W->R | R->W | Not ok |", "|---|---|---|---|---|"]
            L += [f"| {c} | {100 * v['net change']:+.1f} | {100 * v['W->R']:.1f} % | {100 * v['R->W']:.1f} % | {100 * v['not ok']:.1f} % |"
                  for c, v in o["conditions"].items()]
            L += ["", "| Pair | Discordance | Projected ± (accuracy, pp) | Discordance among wrong | Projected ± (W->R, pp) |",
                  "|---|---|---|---|---|"]
            L += [f"| {p} | {100 * v['discordance']:.1f} % | {v['projected half-width, accuracy (pp)']:.1f} | "
                  f"{100 * v['discordance among wrong']:.1f} % | {v['projected half-width, W->R (pp)']:.1f} |" for p, v in o["discordance"].items()]
        L += ["", "| Signal | Dev AUROC |", "|---|---|"]
        L += [f"| {s} | {'—' if v is None else f'{v:.3f}'} |" for s, v in o["signal AUROC"].items()]
        if o.get("dev probe selection"):
            L.append(f"\nDev probe selection: {o['dev probe selection']}.")
        L.append("")
    (ROOT / "reports" / "v2_01_mc_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
