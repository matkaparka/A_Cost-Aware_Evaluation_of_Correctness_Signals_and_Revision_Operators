"""Exp 4 component selection on dev and the confirmation rule on test (design Exp 4). Pure numpy; rules fixed in code
before any Exp 4 output existed.

Per metric m in (lp_gold, marker), on dev:
0. Selectable components. For the marker, the final block (L = n_layers - 1) is not selectable: the marker is read
   at the last prompt position, so patching the final block there reproduces the donor's marker by construction (its
   effect equals the total effect). It stays in the attribution table (attr.npz, frozen.json "not_selectable_attr",
   the analysis report) but is not in the top-N_REAL pool, so it is neither real-patched for the marker nor ranked by
   the k rule (NOT_SELECTABLE_LAST_BLOCK, `excluded_layers`). lp_gold has no excluded layer (its final-block effect is
   0, because target positions do not read the final block of prompt positions).
1. Attribution: rank every selectable (layer, group) by |mean over dev pairs of attr[m]|; the top N_REAL = 20 are
   real-patched (ties: lower layer, then earlier group). The union of the two metrics' top 20 is real-patched once;
   every forward gives both metrics.
2. Real patching: per component c, effect_i = m(R <- D at c)_i - m(R)_i; mean over dev pairs and a 95 % cluster-
   bootstrap CI (canonical problems resampled, N_BOOT = 10,000, seed 0; the resampling scheme of `Boot` in
   analysis/p6_confirmatory.py).
3. k rule: eligible = the top-20 components of m whose CI excludes 0. Ranked list: eligible first, then the others,
   each by |mean effect| (ties: attribution rank). k_m = min(K_MAX, max(K_MIN, #eligible)) with K_MAX = 5, K_MIN = 1;
   the frozen list for m is the first k_m of the ranked list. With no eligible component, the one with the largest
   |mean effect| is still confirmed (k_m = 1).
On test (exploratory, no family-wise claim): every frozen component of m is real-patched on the test pairs; it
"replicates" iff its test mean effect has the sign of its dev mean and its 95 % cluster-bootstrap CI (same scheme)
excludes 0.
"""
import numpy as np

N_REAL, K_MAX, K_MIN = 20, 5, 1
N_BOOT, SEED = 10_000, 0
NOT_SELECTABLE_LAST_BLOCK = ("marker",)   # metrics read at the last prompt position (step 0)


def excluded_layers(metric, n_layers):
    """Layers whose components are not selectable for `metric` (step 0)."""
    return [n_layers - 1] if metric in NOT_SELECTABLE_LAST_BLOCK else []


def boot_counts(clusters, n_boot=N_BOOT, seed=SEED):
    """(cluster index per item, [n_boot, n_clusters] draw counts): the scheme of analysis/p6_confirmatory.Boot."""
    _, cid = np.unique(np.asarray(clusters), return_inverse=True)
    g = cid.max() + 1
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, g, g), minlength=g) for _ in range(n_boot)])
    return cid, counts


def mean_ci(values, clusters, n_boot=N_BOOT, seed=SEED):
    """values [n, K] (items x statistics) -> (mean [K], lo [K], hi [K]); 95 % percentile CI of the item mean under the
    cluster bootstrap."""
    v = np.asarray(values, float)
    if v.ndim == 1:
        v = v[:, None]
    cid, counts = boot_counts(clusters, n_boot, seed)
    g = counts.shape[1]
    S = np.zeros((g, v.shape[1]))
    np.add.at(S, cid, v)
    N = np.bincount(cid, minlength=g).astype(float)
    bs = (counts @ S) / (counts @ N)[:, None]
    return v.mean(0), np.quantile(bs, 0.025, axis=0), np.quantile(bs, 0.975, axis=0)


def top_by_attribution(attr, groups, n=N_REAL, exclude_layers=()):
    """attr [pairs, layers, groups] for one metric -> [(layer, group, mean attr)] of the n largest |mean| among the
    selectable components (layers in exclude_layers are skipped)."""
    mu = np.asarray(attr, float).mean(0)
    order = sorted(((L, k) for L in range(mu.shape[0]) if L not in exclude_layers for k in range(mu.shape[1])),
                   key=lambda c: (-abs(mu[c]), c[0], c[1]))
    return [(int(L), groups[k], float(mu[L, k])) for L, k in order[:n]]


def choose(effects, clusters, comps, n_boot=N_BOOT, seed=SEED, exclude_layers=()):
    """effects [pairs, len(comps)] dev real-patch effects of one metric's top-N_REAL comps (attribution order).
    Returns (stats per comp, k, frozen comps). A comp in exclude_layers is an error (step 0)."""
    bad = [c for c in comps if c[0] in exclude_layers]
    if bad:
        raise ValueError(f"not selectable for this metric: {bad}")
    mu, lo, hi = mean_ci(effects, clusters, n_boot, seed)
    stats = [{"layer": c[0], "group": c[1], "attr_rank": r, "mean": float(mu[r]), "lo": float(lo[r]), "hi": float(hi[r]),
              "eligible": bool(lo[r] > 0 or hi[r] < 0)} for r, c in enumerate(comps)]
    ranked = sorted(stats, key=lambda s: (not s["eligible"], -abs(s["mean"]), s["attr_rank"]))
    n_elig = sum(s["eligible"] for s in stats)
    k = min(K_MAX, max(K_MIN, n_elig))
    return stats, k, [(s["layer"], s["group"], s["mean"]) for s in ranked[:k]]


def confirm(effects, clusters, frozen, n_boot=N_BOOT, seed=SEED):
    """effects [pairs, k] test real-patch effects of one metric's frozen comps; frozen = [(layer, group, dev mean)].
    Returns one row per comp with the test mean, CI and the replication flag."""
    mu, lo, hi = mean_ci(effects, clusters, n_boot, seed)
    out = []
    for r, (L, g, dev_mean) in enumerate(frozen):
        rep = bool(np.sign(mu[r]) == np.sign(dev_mean) and (lo[r] > 0 or hi[r] < 0))
        out.append({"layer": L, "group": g, "dev_mean": dev_mean, "test_mean": float(mu[r]), "lo": float(lo[r]),
                    "hi": float(hi[r]), "replicated": rep})
    return out
