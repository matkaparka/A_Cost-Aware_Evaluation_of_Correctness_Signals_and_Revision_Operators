"""Statistics invariants (design §8): each test's false-positive rate at its own null boundary over 2,000 simulations
is within nominal +/-1.5 pp (exact tests may be conservative: only the upper bound applies to them), and a known
simulated effect is recovered. Run: python -m pytest tests/test_stats.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from power import pattern_probs, simulate  # noqa: E402
from stats import (auroc, auroc_diff_boot, boot_ci, gee_slope, holm, iut, mcnemar_exact, one_sided_paired,  # noqa: E402
                   tost, two_sided_paired)

ALPHA, TOL, SIMS, N, DISC, SHARE = 0.05, 0.015, 2000, 800, 0.2, 0.8
DELTA = 0.03


def rate(test, effect, n=N, sims=SIMS, seed=1):
    ps = simulate(test, n, pattern_probs(DISC, effect, SHARE), sims=sims, seed=seed)
    return (ps < ALPHA).mean()


def test_superiority_at_zero():
    r = rate(lambda x, y: one_sided_paired(x, y, 0, "greater"), 0.0)
    assert abs(r - ALPHA) <= TOL, r


def test_mcnemar_exact_at_zero_not_anticonservative():
    r = rate(mcnemar_exact, 0.0)
    assert r <= ALPHA + TOL, r


@pytest.mark.parametrize("boundary", [DELTA, -DELTA])
def test_tost_at_margin(boundary):
    r = rate(lambda x, y: tost(x, y, DELTA), boundary, n=3000)
    assert abs(r - ALPHA) <= TOL, r


def test_noninferiority_at_margin():
    dw = 0.05
    r = rate(lambda x, y: one_sided_paired(x, y, dw, "less"), dw)
    assert abs(r - ALPHA) <= TOL, r


def test_iut_least_favourable_component():
    # Component 1 at its null boundary (effect 0, superiority), component 2 far from its null: p = max -> ~alpha.
    def test(x, y):
        return iut(one_sided_paired(x, y, 0, "greater"), 0.0)
    r = rate(test, 0.0)
    assert abs(r - ALPHA) <= TOL, r


def test_superiority_recovers_known_effect():
    r = rate(lambda x, y: one_sided_paired(x, y, 0, "greater"), 0.06, n=1000, sims=500)
    assert r > 0.9, r


def test_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(3)
    probs = pattern_probs(DISC, 0.04, SHARE)
    pats = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    cover = 0
    for s in range(200):
        xy = pats[rng.choice(4, size=600, p=probs)]
        _, lo, hi = boot_ci(xy[:, 0], xy[:, 1], n_boot=1000, seed=s)
        cover += lo <= 0.04 <= hi
    assert cover / 200 >= 0.9, cover



def test_bootstrap_clusters():
    """Singleton clusters reproduce the item bootstrap exactly; duplicated items widen the CI only when they are
    resampled as clusters (design §7.1: canonical problems with all their variants)."""
    rng = np.random.default_rng(5)
    x, y = rng.integers(0, 2, 400), rng.integers(0, 2, 400)
    assert boot_ci(x, y, n_boot=500, seed=1) == boot_ci(x, y, n_boot=500, seed=1, clusters=np.arange(400))
    xx, yy, cl = np.repeat(x, 3), np.repeat(y, 3), np.repeat(np.arange(400), 3)
    _, lo_i, hi_i = boot_ci(xx, yy, n_boot=2000, seed=2)
    _, lo_c, hi_c = boot_ci(xx, yy, n_boot=2000, seed=2, clusters=cl)
    assert (hi_c - lo_c) > 1.5 * (hi_i - lo_i)

@pytest.mark.slow
def test_gee_slope_at_zero():
    rng = np.random.default_rng(4)
    rej, sims = 0, 2000
    for _ in range(sims):
        n = 300
        u = rng.normal(0, 1.0, n)                        # item effects (clustering)
        lv = np.tile([0.5, 0.7, 0.9, 1.0], n)
        item = np.repeat(np.arange(n), 4)
        p = 1 / (1 + np.exp(-(0.5 + u[item])))           # slope 0
        y = rng.random(len(p)) < p
        rej += gee_slope(y, lv, item) < ALPHA
    assert abs(rej / sims - ALPHA) <= TOL, rej / sims


def test_holm():
    assert list(holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


def _clustered_d(rng, n_clusters, effect):
    """Item differences in {-1, 0, 1} with cluster-level dependence (2 to 10 items per canonical problem, as in scli5)."""
    sizes = rng.integers(2, 11, n_clusters)
    out, cl = [], []
    for c, m in enumerate(sizes):
        lean = rng.normal(0, 0.6)                        # shared within the cluster
        p_pos = np.clip(0.15 + effect / 2 + 0.1 * lean, 0, 1)
        p_neg = np.clip(0.15 - effect / 2 - 0.1 * lean, 0, 1)
        u = rng.random(m)
        out.append(np.where(u < p_pos, 1, np.where(u < p_pos + p_neg, -1, 0)))
        cl.append(np.full(m, c))
    return np.concatenate(out).astype(float), np.concatenate(cl)


def test_cluster_robust_paired_at_zero():
    """With dependent items inside canonical problems, the cluster-robust test keeps its size at the null boundary."""
    rng = np.random.default_rng(7)
    rej = 0
    for _ in range(SIMS):
        d, cl = _clustered_d(rng, 200, 0.0)
        rej += two_sided_paired(d, np.zeros_like(d), clusters=cl) < ALPHA
    assert abs(rej / SIMS - ALPHA) <= 2 * TOL, rej / SIMS


def test_cluster_robust_singletons_equal_item_test():
    rng = np.random.default_rng(8)
    x, y = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    a = one_sided_paired(x, y, 0.01, "greater")
    b = one_sided_paired(x, y, 0.01, "greater", clusters=np.arange(500))
    assert abs(a - b) < 0.02 and tost(x, y, 0.1) > 0 and two_sided_paired(x, y) <= 1


def test_auroc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(y, 1.0), 1)                  # rounded: many ties
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-12


def _two_scores(rng, n, mu1, mu2):
    y = rng.random(n) < 0.15                             # about 15 % wrong items, as on 4B
    return y, mu1 * y + rng.normal(0, 1, n), mu2 * y + rng.normal(0, 1, n)


def test_auroc_diff_boot_at_zero():
    """Equal true AUROC: the one-sided bootstrap test keeps its size within nominal +/-1.5 pp over 2,000 simulations
    (design §8; 400 resamples each)."""
    rng = np.random.default_rng(10)
    sims, rej = SIMS, 0
    for k in range(sims):
        y, s1, s2 = _two_scores(rng, 400, 1.2, 1.2)
        rej += auroc_diff_boot(y, s1, s2, n_boot=400, seed=k)["p_greater"] < ALPHA
    assert abs(rej / sims - ALPHA) <= TOL, rej / sims


def test_auroc_diff_boot_recovers_effect():
    rng = np.random.default_rng(11)
    sims, rej = 100, 0
    for k in range(sims):
        y, s1, s2 = _two_scores(rng, 1000, 1.6, 0.8)
        r = auroc_diff_boot(y, s1, s2, n_boot=400, seed=k, clusters=np.arange(1000) // 2)
        rej += r["p_greater"] < ALPHA and r["lo"] < r["diff"] < r["hi"]
    assert rej / sims >= 0.9, rej / sims
