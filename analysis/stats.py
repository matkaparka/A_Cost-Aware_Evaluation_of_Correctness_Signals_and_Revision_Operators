"""Confirmatory tests (design §7). Unit: the canonical problem; paired binary outcomes per item.

- mcnemar_exact: paired two-condition contrasts (two-sided, exact binomial on discordant pairs).
- one_sided_paired: superiority / non-inferiority on the paired difference (normal approximation on item differences
  d_i); with `clusters` (one canonical ID per item) the standard error is cluster-robust (design §7.1: the unit is the
  canonical problem). Used for TOST and intersection-union components.
- two_sided_paired: two-sided version (2 x the smaller one-sided p).
- tost: equivalence within +/-delta: p = max of the two one-sided p.
- gee_slope: one-sided test of a positive slope over levels (item-clustered GEE, binomial, exchangeable).
- iut: intersection-union p = max of component p.
- holm: Holm-adjusted p over a fixed family.
- boot_ci: item-clustered paired bootstrap CI of a risk difference.
- auroc / auroc_diff_boot: AUROC with ties counted 1/2, and the cluster-bootstrap CI and one-sided p of a paired AUROC
  difference (H2a, design §7.3).
"""
import math

import numpy as np


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def mcnemar_exact(x, y):
    """Two-sided exact McNemar p for paired binary vectors x, y."""
    from scipy.stats import binomtest
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    if b + c == 0:
        return 1.0
    return binomtest(b, b + c, 0.5).pvalue


def _mean_se(d, clusters=None):
    """Item-level mean of d and its standard error; cluster-robust (ratio estimator, linearized) with clusters."""
    d = np.asarray(d, float)
    n = len(d)
    m = d.mean()
    if clusters is None:
        return m, (d.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0)
    _, cid = np.unique(np.asarray(clusters), return_inverse=True)
    tot, size = np.bincount(cid, weights=d), np.bincount(cid).astype(float)
    g = len(tot)
    if g < 2:
        return m, 0.0
    resid = tot - m * size
    return m, math.sqrt(g / (g - 1) * (resid ** 2).sum()) / n


def one_sided_paired(x, y, margin=0.0, direction="greater", clusters=None):
    """H0: mean(x - y) <= margin (direction 'greater') or >= margin ('less'). Normal approximation; cluster-robust
    standard error when `clusters` is given."""
    m, se = _mean_se(np.asarray(x, float) - np.asarray(y, float), clusters)
    if se == 0:
        if direction == "greater":
            return 0.0 if m > margin else 1.0
        return 0.0 if m < margin else 1.0
    z = (m - margin) / se
    return _norm_sf(z) if direction == "greater" else _norm_sf(-z)


def two_sided_paired(x, y, clusters=None):
    return min(1.0, 2 * min(one_sided_paired(x, y, 0.0, "greater", clusters),
                            one_sided_paired(x, y, 0.0, "less", clusters)))


def tost(x, y, delta, clusters=None):
    """Equivalence of mean(x - y) within (-delta, +delta); p = max of the two one-sided p."""
    return max(one_sided_paired(x, y, -delta, "greater", clusters), one_sided_paired(x, y, delta, "less", clusters))


def gee_slope(correct, level, cluster):
    """One-sided p for a positive slope of P(correct) on level, GEE logistic clustered by item."""
    import statsmodels.api as sm
    correct = np.asarray(correct, float)
    X = sm.add_constant(np.asarray(level, float))
    if correct.min() == correct.max():
        return 1.0
    res = sm.GEE(correct, X, groups=np.asarray(cluster), family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Exchangeable()).fit()
    coef, p_two = res.params[1], res.pvalues[1]
    return p_two / 2 if coef > 0 else 1 - p_two / 2


def iut(*ps):
    return max(ps)


def holm(pvals):
    """Holm step-down adjusted p-values, same order as input."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def boot_ci(x, y, n_boot=10000, seed=0, level=0.95, clusters=None):
    """Paired bootstrap CI for mean(x) - mean(y) over items. Design §7.1 resamples canonical problems with all their
    variants: pass `clusters` (one canonical ID per item) to resample clusters; without it each item is its own
    cluster. The estimate is the item-level mean in every resample (sum over drawn clusters / their item count)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(x, float) - np.asarray(y, float)
    if clusters is None:
        idx = rng.integers(0, len(d), (n_boot, len(d)))
        stats = d[idx].mean(axis=1)
    else:
        _, cid = np.unique(np.asarray(clusters), return_inverse=True)
        sums, sizes = np.bincount(cid, weights=d), np.bincount(cid).astype(float)
        idx = rng.integers(0, len(sums), (n_boot, len(sums)))
        stats = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    a = (1 - level) / 2
    return float(d.mean()), float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def auroc(y, s):
    """AUROC of score s for binary y, ties counted 1/2 (Mann-Whitney)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    r = rankdata(np.asarray(s, float))
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_diff_boot(y, s1, s2, clusters=None, n_boot=10000, seed=0, level=0.95):
    """Paired AUROC(s1) - AUROC(s2) on the same items (H2a: s1 probe, s2 verbal).

    Resamples canonical problems with all their items (each item its own cluster without `clusters`); a resample with
    only one class is redrawn. Returns the estimate, the percentile CI at `level`, and the one-sided bootstrap p for
    H0: AUROC(s1) - AUROC(s2) <= 0, p = (1 + #{resampled difference <= 0}) / (n_boot + 1), the level at which the
    one-sided percentile interval reaches 0."""
    y, s1, s2 = np.asarray(y, bool), np.asarray(s1, float), np.asarray(s2, float)
    est = auroc(y, s1) - auroc(y, s2)
    rng = np.random.default_rng(seed)
    if clusters is None:
        members = [np.array([i]) for i in range(len(y))]
    else:
        _, cid = np.unique(np.asarray(clusters), return_inverse=True)
        order = np.argsort(cid, kind="stable")
        members = np.split(order, np.cumsum(np.bincount(cid))[:-1])
    stats = np.empty(n_boot)
    k = 0
    while k < n_boot:
        idx = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        yb = y[idx]
        if yb.all() or not yb.any():
            continue
        stats[k] = auroc(yb, s1[idx]) - auroc(yb, s2[idx])
        k += 1
    a = (1 - level) / 2
    return {"diff": float(est), "lo": float(np.quantile(stats, a)), "hi": float(np.quantile(stats, 1 - a)),
            "p_greater": float((1 + (stats <= 0).sum()) / (n_boot + 1))}
