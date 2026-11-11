"""D-inj-lore invariants (proposal §3.3): each injected a0 scores as its variant says, a wrong value is taken from the
item's own context, and the variant is the seeded coin alone. Run: python -m pytest tests/test_lore_inj.py -q"""
import json
import re
from collections import Counter

import pytest

from src.data import lore, lore_b
from src.data.lore import OUT, load_built, unit
from src.data.lore_inj import context_text
from src.score.lore import score

INJ = OUT / "items_inj.jsonl"


@pytest.fixture(scope="module")
def data():
    if not INJ.exists():
        pytest.skip("D-inj-lore not built yet")
    passages, items_a = load_built()
    items = {it["id"]: it for it in items_a + lore_b.load_items_b()}
    rows = [json.loads(l) for l in INJ.read_text(encoding="utf-8").splitlines()]
    return passages, items, rows


def test_a0_scores_as_its_variant(data):
    _, items, rows = data
    for r in rows:
        assert score(r["a0"], items[r["base_id"]])["correct"] == (r["variant"] == "right"), r["id"]


def test_wrong_value_comes_from_the_context(data):
    passages, items, rows = data
    for r in rows:
        if r["variant"] == "wrong":
            text = context_text(items[r["base_id"]], r["budget"], passages)
            v = r["a0_value"].split()[-1] if re.match(r"^[123]E \d+$", r["a0_value"]) else r["a0_value"]
            assert re.search(re.escape(v), text, re.I), r["id"]


def test_variant_is_the_coin_and_rows_are_unique(data):
    _, _, rows = data
    assert max(Counter((r["base_id"], r["budget"]) for r in rows).values()) == 1
    for r in rows:
        assert r["variant"] == ("wrong" if unit("lore-inj", r["base_id"]) < 0.5 else "right")


def test_split_follows_the_base_item(data):
    _, items, rows = data
    assert all(r["split"] == items[r["base_id"]]["split"] for r in rows)
