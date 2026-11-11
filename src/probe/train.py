"""Exp 2 probe selection on dev (design §6, Exp 2; H2a): L2 logistic probe on the cached residual stream,
layer and C chosen by grouped 5-fold CV within dev (groups = canonical problem), standardization fitted inside
each training fold. The selected probe is refit on all of dev and frozen (weights saved); test items are scored
once, after the prereg freeze, by score_frozen().

Also reports the dev learning curve (CV AUROC of the selected layer/C at growing dev fractions), so P3 can apply
the pre-registered rule for enlarging dev if the curve has not plateaued.
Usage: python -m src.probe.train --acts <run>/acts.npz [<run2>/acts.npz ...] --out <dir> [--canonical <items.jsonl> ...]
(several activation files, e.g. the original dev and the P3 expansion dev, are pooled as one dev set)
"""
import argparse
import json
from pathlib import Path

import numpy as np

C_GRID = (0.001, 0.01, 0.1, 1.0)
FOLDS = 5
SEED = 0


def cv_auroc(X, y, groups, C, folds=FOLDS, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    scores = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--canonical", default=None, nargs="+", help="item tables for item -> canonical ID groups")
    args = ap.parse_args()
    zs = [np.load(a) for a in args.acts]
    X = np.concatenate([z["X"].astype(np.float32) for z in zs])
    y = np.concatenate([z["correct0"].astype(int) for z in zs])
    ids = np.concatenate([z["ids"] for z in zs])
    assert len(set(ids)) == len(ids), "an item appears in two activation files"
    root = Path(__file__).resolve().parents[2]
    canon = {json.loads(l)["item_id"]: json.loads(l)["canonical_id"]
             for f in (args.canonical or [root / "data" / "processed" / "dnat.jsonl"]) for l in open(f, encoding="utf-8")}
    groups = np.array([canon[i] for i in ids])

    grid = []
    for layer in range(X.shape[1]):
        for C in C_GRID:
            grid.append({"layer": layer, "C": C, "cv_auroc": cv_auroc(X[:, layer], y, groups, C)})
    best = max(grid, key=lambda g: (g["cv_auroc"], -g["layer"]))   # ties: the earlier layer

    rng = np.random.default_rng(SEED)
    curve = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        sel = rng.permutation(len(y))[: int(frac * len(y))]
        if len(set(y[sel])) == 2 and min(np.bincount(y[sel])) >= FOLDS:
            curve.append({"fraction": frac, "n": len(sel),
                          "cv_auroc": cv_auroc(X[sel, best["layer"]], y[sel], groups[sel], best["C"])})

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X[:, best["layer"]])
    clf = LogisticRegression(C=best["C"], max_iter=2000).fit(sc.transform(X[:, best["layer"]]), y)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "probe.npz", layer=best["layer"], C=best["C"], mean=sc.mean_, scale=sc.scale_,
             coef=clf.coef_[0], intercept=clf.intercept_[0])
    res = {"n_dev": int(len(y)), "n_wrong": int((y == 0).sum()), "selected": best, "learning_curve": curve,
           "grid": grid}
    (out / "probe_selection.json").write_text(json.dumps(res, indent=1))
    import platform
    import sklearn
    from ..runlib import code_commit, sha256_file
    (out / "manifest.json").write_text(json.dumps({
        "acts": [str(a) for a in args.acts], "acts_sha256": [sha256_file(a) for a in args.acts], "code_commit": code_commit(),
        "config": {"C_grid": C_GRID, "folds": FOLDS, "seed": SEED}, "host": platform.node(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "files": {p.name: sha256_file(p) for p in sorted(out.glob("probe*"))}}, indent=1))
    print(json.dumps({k: res[k] for k in ("n_dev", "n_wrong", "selected", "learning_curve")}, indent=1))


def score_frozen(probe_npz, X):
    """P(correct) from the frozen probe; X has shape [n, layers+1, hidden]."""
    p = np.load(probe_npz)
    h = (X[:, int(p["layer"])].astype(np.float32) - p["mean"]) / p["scale"]
    return 1 / (1 + np.exp(-(h @ p["coef"] + float(p["intercept"]))))


if __name__ == "__main__":
    main()
