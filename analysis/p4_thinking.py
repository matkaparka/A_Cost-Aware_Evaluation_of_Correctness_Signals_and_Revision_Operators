"""Thinking-subset estimates (exploratory; design "Thinking subset"; DEVIATIONS D7, rules fixed in its addendum before
any thinking output existed). Estimates and 95 % cluster-bootstrap CIs only; nothing here is a test (prereg §3.1, §8).
-> reports/04_<out>.md and analysis/out/<out>.json

- Item outcome: the mean correctness over the 2 seeds (truncated or unparsed = wrong). Unit: the canonical problem;
  `Boot` of analysis/p6_confirmatory.py (10,000 resamples, seed 0).
- Population accuracy: weighted by stratum, w = N_stratum / n_stratum of the items run; N is read from the runs'
  subset.json (4B: a0 wrong N 149, right N 1306 in the original test pool; DEVIATIONS D8 made it per model). W->R: mean over the initially wrong items; R->W: 1 - mean over the initially right items.
- Paired with the non-thinking run on the same items (the 4B Exp 1A / 1B rows, greedy): thinking - non-thinking
  per condition (weighted).
- Contrasts within thinking mode (weighted): F2 - F1 (H1a's pair), F3(1.0) - F3(0.7) (H1b's direction), F5 - F0,
  F3(1.0) - F0; Exp 1B at F0: role = mean(assistant cells) - mean(user cells), label = mean(self) - mean(other).
- Per condition: truncated and unparsed rates, mean output tokens.
Usage: python analysis/p4_thinking.py --runs <run dir> [<run dir> ...] --nonthinking <4B exp01 run> --out p4_4B_thinking
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from p6_confirmatory import N_BOOT, Boot, jl  # noqa: E402

CONDS = ["B1", "F0", "F1", "F2", "F3(0.7)", "F3(1.0)", "F5",
         "1B:assistant/self/F0", "1B:assistant/other/F0", "1B:user/self/F0", "1B:user/other/F0"]


def pp(x):
    return "–" if x is None else f"{100 * x:+.1f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.1f} %"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--nonthinking", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--model", default="4B")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    canon = {r["item_id"]: r["canonical_id"] for r in jl("data/processed/dnat.jsonl")}
    pools = [{c0 == "True": v["N"] for c0, v in json.loads((Path(run) / "subset.json").read_text())["weights"].items()}
             for run in a.runs]
    assert all(p == pools[0] for p in pools), "runs disagree on the stratum sizes"
    N_POOL = pools[0]   # original-pool test items by a0 correctness
    acc, status, toks, c0 = defaultdict(list), defaultdict(Counter), defaultdict(list), {}
    for run in a.runs:
        for r in jl(Path(run) / "rows.jsonl"):
            if r["cond"] == "F2:turn1":
                continue
            acc[(r["cond"], r["item_id"])].append(float(r["correct"]))
            status[r["cond"]][r["status"]] += 1
            toks[r["cond"]].append(r["out_tokens"])
            c0[r["item_id"]] = r["correct0"]
    ids = sorted(c0)
    assert all(len(acc[(c, i)]) == 2 for c in CONDS for i in ids), "every item needs both seeds of every condition"
    y0 = np.array([c0[i] for i in ids], bool)
    n = {s: int((y0 == s).sum()) for s in (False, True)}
    w = np.array([N_POOL[s] / n[s] for s in y0])
    nt = defaultdict(dict)   # non-thinking correctness: cond (thinking naming) -> item -> 0/1
    for r in jl(Path(a.nonthinking) / "rows.jsonl"):
        if r["data"] == "dnat" and r["item_id"] in c0:
            nt[r["cond"] if r["exp"] == "1A" else f"1B:{r['cond']}"][r["item_id"]] = float(r["correct"])
    boot = Boot(np.array([canon[i] for i in ids]), a.n_boot)
    Y = {c: np.array([np.mean(acc[(c, i)]) for i in ids]) for c in CONDS}
    NT = {c: np.array([nt[c][i] for i in ids]) for c in CONDS}
    wm = lambda v: (lambda m: float((m * w * v).sum() / (m * w).sum()))
    sm = lambda v, s: (lambda m: float((m * v * (y0 == s)).sum() / (m * (y0 == s)).sum()))

    out = {"runs": a.runs, "nonthinking": a.nonthinking, "n_items": len(ids), "n_by_a0": {str(k): v for k, v in n.items()},
           "weights": {str(s): N_POOL[s] / n[s] for s in (False, True)}, "n_boot": a.n_boot, "cells": {}, "contrasts": {}}
    for c in CONDS:
        out["cells"][c] = {"acc": boot.ci(wm(Y[c])), "W->R": boot.ci(sm(Y[c], False)),
                           "R->W": boot.ci(lambda m, v=Y[c]: 1 - sm(v, True)(m)),
                           "minus_nonthinking": boot.ci(wm(Y[c] - NT[c])), "nonthinking_acc": wm(NT[c])(np.ones(len(ids))),
                           "status": dict(status[c]), "mean_out_tokens": float(np.mean(toks[c]))}
    role = (Y["1B:assistant/self/F0"] + Y["1B:assistant/other/F0"] - Y["1B:user/self/F0"] - Y["1B:user/other/F0"]) / 2
    label = (Y["1B:assistant/self/F0"] + Y["1B:user/self/F0"] - Y["1B:assistant/other/F0"] - Y["1B:user/other/F0"]) / 2
    for name, v in (("F2 - F1", Y["F2"] - Y["F1"]), ("F3(1.0) - F3(0.7)", Y["F3(1.0)"] - Y["F3(0.7)"]),
                    ("F5 - F0", Y["F5"] - Y["F0"]), ("F3(1.0) - F0", Y["F3(1.0)"] - Y["F0"]),
                    ("1B role at F0 (assistant - user)", role), ("1B label at F0 (self - other)", label)):
        out["contrasts"][name] = boot.ci(wm(v))
    (ROOT / "analysis" / "out" / f"{a.out}.json").write_text(json.dumps(out, indent=1, default=float))

    L = [f"# P4 thinking subset, {a.model} test (generated by analysis/p4_thinking.py)", "",
         f"Runs: {', '.join(f'`{Path(r).as_posix()}`' for r in a.runs)}; {len(ids)} items ({n[False]} with a0 wrong, "
         f"{n[True]} right; weights {N_POOL[False] / n[False]:.2f} / {N_POOL[True] / n[True]:.2f}), 2 seeds each. "
         "Exploratory: estimates and 95 % cluster-bootstrap CIs only (DEVIATIONS D7, D8). Accuracy is weighted to the "
         f"original test pool; differences in pp. Non-thinking = the greedy {a.model} Exp 1A / 1B rows on the same items "
         f"(`{Path(a.nonthinking).as_posix()}`).", "",
         "| Condition | Acc | 95 % CI | W->R | R->W | Non-thinking acc | Thinking − non-thinking | 95 % CI | Truncated | Unparsed | Mean out tokens |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in out["cells"].items():
        tot = sum(v["status"].values())
        L.append(f"| {c} | {pct(v['acc'][0])} | [{pct(v['acc'][1])}, {pct(v['acc'][2])}] | {pct(v['W->R'][0])} | "
                 f"{pct(v['R->W'][0])} | {pct(v['nonthinking_acc'])} | {pp(v['minus_nonthinking'][0])} | "
                 f"[{pp(v['minus_nonthinking'][1])}, {pp(v['minus_nonthinking'][2])}] | "
                 f"{pct(v['status'].get('truncated', 0) / tot)} | {pct(v['status'].get('unparsed', 0) / tot)} | "
                 f"{v['mean_out_tokens']:.0f} |")
    L += ["", "## Contrasts within thinking mode (weighted, pp)", "", "| Contrast | Estimate | 95 % CI |", "|---|---|---|"]
    L += [f"| {k} | {pp(e)} | [{pp(lo)}, {pp(hi)}] |" for k, (e, lo, hi) in out["contrasts"].items()]
    (ROOT / "reports" / f"04_{a.out}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
