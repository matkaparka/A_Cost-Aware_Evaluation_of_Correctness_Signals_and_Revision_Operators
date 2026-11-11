"""Edge cases of the multiple-choice scorer (src/score/mc.py) and invariants of the MMLU-Pro build."""
import json
from pathlib import Path

import pytest

from src.score import mc

ITEM = {"options": ["Paris", "London", "Rome", "Berlin"], "n_options": 4, "gold": "B"}


@pytest.mark.parametrize("text,pred,status", [
    ("... so the answer is \\boxed{B}", "B", "ok"),
    ("\\boxed{(B)}", "B", "ok"),
    ("\\boxed{B. London}", "B", "ok"),
    ("\\boxed{\\text{B}}", "B", "ok"),
    ("\\boxed{b}", "B", "ok"),
    ("\\boxed{London}", "B", "ok"),
    ("first \\boxed{A} then \\boxed{C}", "C", "ok"),
    ("\\boxed{E}", None, "unparsed"),               # letter beyond the options
    ("\\boxed{London or Paris}", None, "unparsed"),
    ("The answer is (D).", "D", "fallback"),
    ("Answer: C", "C", "fallback"),
    ("no answer here", None, "unparsed"),
])
def test_extract(text, pred, status):
    assert mc.extract(text, ITEM) == (pred, status)


def test_score_truncated_and_correct():
    assert mc.score("I think", ITEM, "length") == {"status": "truncated", "pred": None, "correct": False}
    assert mc.score("\\boxed{B}", ITEM, "stop")["correct"] is True
    assert mc.score("\\boxed{A}", ITEM, "stop")["correct"] is False


def test_nested_braces():
    assert mc.last_boxed("\\boxed{\\text{(C)}}") == "\\text{(C)}"


def test_build_invariants():
    p = Path(__file__).resolve().parents[1] / "data" / "processed" / "mmlupro" / "items.jsonl"
    if not p.exists():
        pytest.skip("MMLU-Pro items not built")
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    assert len({r["item_id"] for r in rows}) == len(rows)
    split_of = {}
    for r in rows:   # every canonical stem sits in one split
        assert split_of.setdefault(r["canonical_id"], r["split"]) == r["split"]
        assert r["gold"] in mc.LETTERS[:r["n_options"]] and r["n_options"] == len(r["options"])
    for s in ("dev", "test"):
        orders = sorted(r["order"] for r in rows if r["split"] == s)
        assert orders == list(range(len(orders)))
