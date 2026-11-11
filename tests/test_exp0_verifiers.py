"""Exp 0 construction checks and the scripted binary agent truth table (design §5.5, §8), plus verifier invariants."""
from collections import Counter, defaultdict

import pytest

from src.data.exp0 import make_items
from src.data.verifiers import f3_verdict, permute_self_verdicts, uniform

ITEMS = make_items()
P_LEVELS = (0.5, 0.7, 0.9, 1.0)


def by_template():
    t = defaultdict(list)
    for r in ITEMS:
        t[r["template_id"]].append(r)
    return t


def test_counts():
    assert len(by_template()) == 200 and len(ITEMS) == 400


def test_paired_worlds_share_prompt_and_split():
    for tid, (w0, w1) in by_template().items():
        assert w0["prompt"] == w1["prompt"], tid
        assert w0["split"] == w1["split"]
        assert w0["gold_value"] != w1["gold_value"] and w0["gold_letter"] != w1["gold_letter"]
        assert w0["hidden_value"] != w1["hidden_value"]


def test_prompt_shows_both_options_and_no_world_bit():
    for r in ITEMS:
        assert f"(A) {r['options']['A']}" in r["prompt"] and f"(B) {r['options']['B']}" in r["prompt"]
        assert "world" not in r["prompt"].lower()


def test_split_fraction():
    c = Counter(r["split"] for r in ITEMS)
    assert c["dev"] == 80 and c["test"] == 320


@pytest.mark.parametrize("agent_choice", ["A", "B", "alternate"])
def test_scripted_agent_no_feedback_is_half(agent_choice):
    # One no-feedback output per template, scored in both worlds: a valid choice is right in exactly one world.
    for tid, worlds in by_template().items():
        choice = agent_choice if agent_choice != "alternate" else ("A" if hash(tid) % 2 else "B")
        assert sum(choice == w["gold_letter"] for w in worlds) == 1


@pytest.mark.parametrize("p", P_LEVELS)
def test_scripted_agent_switch_policy_equals_truthful_mask(p):
    # Switch exactly to the other option when told "incorrect": final correct <=> the verifier was truthful.
    for r in ITEMS:
        for initial in ("A", "B"):
            says_correct = f3_verdict(r["item_id"], initial == r["gold_letter"], p)
            final = initial if says_correct else ("B" if initial == "A" else "A")
            assert (final == r["gold_letter"]) == (uniform(r["item_id"]) < p)


def test_masks_nested_across_p():
    for r in ITEMS:
        truthful = [uniform(r["item_id"]) < p for p in P_LEVELS]
        assert truthful == sorted(truthful)   # once truthful at p, truthful at every larger p


def test_counterfactual_gold_changes_only_the_verdict():
    # Same item and seed, private label flipped: only the verdict changes, and it changes deterministically.
    for r in ITEMS[:50]:
        for p in P_LEVELS:
            assert f3_verdict(r["item_id"], True, p) != f3_verdict(r["item_id"], False, p)


def test_permutation_keeps_counts_and_strata():
    rows = [{"item_id": f"i{k}", "dataset": "d" + str(k % 2), "initially_correct": k % 3 == 0,
             "self_says_correct": k % 5 != 0} for k in range(300)]
    perm = permute_self_verdicts(rows)
    for k in range(3):
        for key in {(r["dataset"], r["initially_correct"]) for r in rows}:
            members = [r for r in rows if (r["dataset"], r["initially_correct"]) == key]
            assert Counter(r["self_says_correct"] for r in members) == Counter(perm[r["item_id"]][k] for r in members)
    assert any(perm[r["item_id"]][0] != r["self_says_correct"] for r in rows)


def test_self_verdict_threshold_in_log_odds():
    from src.data.verifiers import self_verdict
    # s = 0 is P(yes)_norm >= 0.5
    assert self_verdict({"p_yes": 0.6, "p_no": 0.4}) is True
    assert self_verdict({"p_yes": 0.4, "p_no": 0.6}) is False
    # 4B-like scores: P(yes)_norm within 1e-7 of 1 on both items, but the log-odds separate them
    hi, lo = {"p_yes": 0.9999, "p_no": 1e-9}, {"p_yes": 0.9999, "p_no": 1e-6}
    assert self_verdict(hi, 16.0) is True and self_verdict(lo, 16.0) is False
    # missing or degenerate scores
    assert self_verdict({"p_yes": None, "p_no": None}) is False
    assert self_verdict({"p_yes": 0.0, "p_no": 0.3}) is False
    assert self_verdict({"p_yes": 0.5, "p_no": 0.0}, 1e9) is True
