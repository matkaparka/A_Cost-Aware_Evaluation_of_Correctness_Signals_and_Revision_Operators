"""Verifiers (design §5.8). Labels enter only here, through frozen algorithms; nothing is tuned on outcomes.

F3(p): one uniform U_i per item from the verifier stream (shared across conditions and models; masks nested
across p). The verdict is the true label if U_i < p, otherwise flipped.

F3-perm: the model's thresholded self-verdicts permuted within dataset x initial-correctness strata, K = 3 fixed
seeds. Realized class-conditional verdict counts are kept exactly; only the placement is randomized. It uses
ground-truth strata, so it is an oracle randomized control.
"""
import random
from collections import defaultdict

from .build import h

K_PERM = 3


def uniform(item_id, seed=0):
    """U_i in [0, 1) from stream verifier-v1 (seed 0 = primary mask; 1, 2 = robustness masks)."""
    return int(h("verifier-v1", seed, item_id), 16) / 16 ** 64


def f3_verdict(item_id, truly_correct, p, seed=0):
    """True if the simulated verifier says "correct"."""
    truthful = uniform(item_id, seed) < p
    return truly_correct if truthful else not truly_correct


def self_verdict(verbal_row, threshold_s=0.0):
    """The model's thresholded verbal self-verdict (design §5.7): True ("correct") iff
    s = log P(yes) - log P(no) >= threshold_s. threshold_s = 0 is P(yes)_norm >= 0.5; the prereg fixes the value
    (0, or the dev threshold maximizing balanced accuracy). Log-odds, because 4B puts P(yes)_norm within 1e-7 of 1 on
    most items and the dev threshold sits there. An item without a verbal score counts as "incorrect"."""
    import math
    py, pn = verbal_row.get("p_yes"), verbal_row.get("p_no")
    if py is None or pn is None or py <= 0:
        return False
    if pn <= 0:
        return True          # s = +inf
    return math.log(py) - math.log(pn) >= threshold_s


def permute_self_verdicts(rows, k=K_PERM):
    """rows: dicts with item_id, dataset, initially_correct, self_says_correct.
    Returns {item_id: [verdict under permutation 0..k-1]}."""
    strata = defaultdict(list)
    for r in rows:
        strata[(r["dataset"], bool(r["initially_correct"]))].append(r)
    out = defaultdict(list)
    for perm in range(k):
        for (dataset, correct), members in sorted(strata.items()):
            members = sorted(members, key=lambda r: r["item_id"])
            verdicts = [m["self_says_correct"] for m in members]
            random.Random(h("perm-v1", perm, dataset, correct)).shuffle(verdicts)
            for m, v in zip(members, verdicts):
                out[m["item_id"]].append(v)
    return dict(out)
