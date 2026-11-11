"""Model-to-model differences on a common item panel (second review, review_and_suggestion.md 2026-11-08 §5 step 1:
"estimate the difference between models directly; do not read one significant and one non-significant estimate as a
difference"). Exploratory (DEVIATIONS D18); every input is an existing test run.

Per item, with the frozen definitions (analysis/p6_confirmatory.py):
- D-nat (pooled original + expansion test items that every model has): F2 - F1 (the H1a quantity), F3(1.0) - F3(0.5)
  (the endpoints of the H1b trend; not the frozen GEE slope), and F2 - a0 (the net change after a critique turn);
  the H2a quantity AUROC(probe) - AUROC(verbal s) on the items with both scores.
- D-inj at F0 (items that every model has): the role contrast (user - assistant) and the label contrast
  (other - self) of each item's four cells.
For every model the estimate; for each model pair the difference of the item-level quantities on the common items,
with a 95 % cluster-bootstrap CI that resamples the same canonical problems for both models.
Tulu-3-8B's role contrast includes a one-token template boundary difference (DEVIATIONS D15), so its role column is a
role-plus-template contrast.
-> reports/fu_06_crossmodel.md, analysis/out/fu_06_crossmodel.json
Usage: python analysis/fu_crossmodel.py [--n-boot 10000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from p6_confirmatory import N_BOOT, SEED, Boot, exp01, load, probe_scores, verbal_s, wmean  # noqa: E402

SPECS = {"Qwen3-1.7B": "configs/p4_spec_1.7B_test.json", "Qwen3-4B": "configs/p6_spec_4B_test.json",
         "Qwen3-8B": "configs/p4_spec_8B_test.json", "Tulu-3-8B": "configs/fu2_spec_T8B_test.json"}
PAIRS = [("Tulu-3-8B", "Qwen3-8B"), ("Qwen3-8B", "Qwen3-4B"), ("Qwen3-1.7B", "Qwen3-4B")]
CELLS = ["assistant/self/F0", "assistant/other/F0", "user/self/F0", "user/other/F0"]


def wauc(y, s, w):
    """AUROC with integer multiplicity weights; ties count one half."""
    order = np.argsort(s, kind="stable")
    ss, yy, ww = s[order], y[order], w[order]
    _, g = np.unique(ss, return_inverse=True)
    neg = np.bincount(g, weights=ww * (1 - yy))
    pos = np.bincount(g, weights=ww * yy)
    below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    num = (pos * (below + 0.5 * neg)).sum()
    den = pos.sum() * neg.sum()
    return num / den if den else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    M = {}
    for m, sp in SPECS.items():
        spec = json.loads((ROOT / sp).read_text(encoding="utf-8"))
        gold, canon, strat, init, verbal = load(spec)
        items = {}
        for run in [spec["exp01_orig"]] + ([spec["exp01_ext"]] if spec.get("exp01_ext") else []):
            items.update(exp01(run, "dnat", "1A", gold, False))
        b = exp01(spec["exp01_1b"], "dinj", "1B", gold, False)
        ps = probe_scores(spec)
        M[m] = {"canon": canon, "init": init, "verbal": verbal, "items": items, "b": b, "probe": ps}
    canon = M["Qwen3-4B"]["canon"]
    need = ["F1", "F2", "F3(0.5)", "F3(1.0)"]
    nat = sorted(set.intersection(*[{i for i, v in M[m]["items"].items() if all(c in v for c in need) and i in M[m]["init"]}
                                    for m in M]))
    inj = sorted(set.intersection(*[{i for i, v in M[m]["b"].items() if all(c in v for c in CELLS)} for m in M]))
    h2 = sorted(set(nat).intersection(*[{i for i in M[m]["probe"] if verbal_s(M[m]["verbal"].get(i, {})) is not None}
                                        for m in M]))
    Q = {}
    for m, d in M.items():
        it = d["items"]
        y0 = np.array([bool(d["init"][i]["correct"]) for i in nat], float)
        C = np.array([[d["b"][i][c] for c in CELLS] for i in inj], float)
        Q[m] = {"F2 - F1": np.array([it[i]["F2"] - it[i]["F1"] for i in nat], float),
                "F3(1.0) - F3(0.5)": np.array([it[i]["F3(1.0)"] - it[i]["F3(0.5)"] for i in nat], float),
                "F2 - a0": np.array([it[i]["F2"] for i in nat], float) - y0,
                "role (user - assistant), D-inj F0": (C[:, 2] + C[:, 3] - C[:, 0] - C[:, 1]) / 2,
                "label (other - self), D-inj F0": (C[:, 1] + C[:, 3] - C[:, 0] - C[:, 2]) / 2,
                "_h2": (np.array([bool(d["init"][i]["correct"]) for i in h2], float),
                        np.array([d["probe"][i] for i in h2], float),
                        np.array([verbal_s(d["verbal"][i]) for i in h2], float))}
    b_nat = Boot([canon[i] for i in nat], a.n_boot, SEED + 41)
    b_inj = Boot([canon[i] for i in inj], a.n_boot, SEED + 42)
    b_h2 = Boot([canon[i] for i in h2], a.n_boot, SEED + 43)
    out = {"n D-nat common": len(nat), "n D-inj common": len(inj), "n H2a common": len(h2), "models": {}, "pairs": {}}
    keys = ["F2 - F1", "F3(1.0) - F3(0.5)", "F2 - a0", "role (user - assistant), D-inj F0", "label (other - self), D-inj F0"]
    bt = lambda k: b_inj if "D-inj" in k else b_nat  # noqa: E731
    h2stat = lambda y, p, v, w: wauc(y, p, w) - wauc(y, v, w)  # noqa: E731
    for m, q in Q.items():
        out["models"][m] = {k: list(bt(k).ci(lambda w, v=q[k]: wmean(v, w))) for k in keys}
        y, p, v = q["_h2"]
        out["models"][m]["H2a: AUROC probe - verbal"] = list(b_h2.ci(lambda w: h2stat(y, p, v, w)))
    for m1, m2 in PAIRS:
        r = {k: list(bt(k).ci(lambda w, d=Q[m1][k] - Q[m2][k]: wmean(d, w))) for k in keys}
        y1, p1, v1 = Q[m1]["_h2"]
        y2, p2, v2 = Q[m2]["_h2"]
        r["H2a: AUROC probe - verbal"] = list(b_h2.ci(lambda w: h2stat(y1, p1, v1, w) - h2stat(y2, p2, v2, w)))
        out["pairs"][f"{m1} - {m2}"] = r
    (ROOT / "analysis" / "out" / "fu_06_crossmodel.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    pc = lambda v: f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]"  # noqa: E731
    cols = keys + ["H2a: AUROC probe - verbal"]
    L = ["# Model-to-model differences on a common item panel (generated by analysis/fu_crossmodel.py)", "",
         f"**Exploratory** (DEVIATIONS D18). Common items: D-nat {out['n D-nat common']} (pooled test), D-inj "
         f"{out['n D-inj common']}, H2a {out['n H2a common']}. pp (H2a: AUROC difference × 100); 95 % cluster-bootstrap CIs, "
         "the same canonical problems resampled for both models of a pair. F3(1.0) − F3(0.5) is the endpoint difference, not "
         "the frozen GEE slope. Tulu-3-8B's role contrast is a role-plus-template contrast (D15).", "",
         "## Per model", "", "| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    L += [f"| {m} | " + " | ".join(pc(v[k]) for k in cols) + " |" for m, v in out["models"].items()]
    L += ["", "## Differences between models (same items)", "", "| Pair | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    L += [f"| {p} | " + " | ".join(pc(v[k]) for k in cols) + " |" for p, v in out["pairs"].items()]
    (ROOT / "reports" / "fu_06_crossmodel.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
