"""Lore QA track A invariants (proposal §3, §5; design §8). The built files in data/processed/lore are checked
everywhere; the rebuild test needs the corpus (D:/fandom/worldbuilding) and the local tokenizer, and is skipped
without them. Run: python -m pytest tests/test_lore_build.py -q"""
import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from src.data import lore
from src.data.lore import BUDGETS, OUT, load_built, name_pattern, number_pattern, render_question, segment_text, unit

CORPUS = Path("D:/fandom/worldbuilding")
PASSAGES, ITEMS = load_built()
MANIFEST = json.loads((OUT / "build_manifest_a.json").read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_manifest_hashes_match_files():
    assert MANIFEST["passages_sha256"] == sha(OUT / "passages.jsonl")
    assert MANIFEST["items_a_sha256"] == sha(OUT / "items_a.jsonl")
    assert MANIFEST["counts"]["items"] == len(ITEMS)


def test_ids_unique_and_passage_ids_dense():
    assert len({it["id"] for it in ITEMS}) == len(ITEMS)
    assert [p["pid"] for p in PASSAGES] == list(range(len(PASSAGES)))


def test_split_is_a_function_of_the_cluster():
    splits = defaultdict(set)
    for it in ITEMS:
        splits[it["cluster"]].add(it["split"])
        assert it["split"] == ("dev" if unit("lore-split", it["cluster"]) < lore.DEV_FRACTION else "test")
    assert all(len(s) == 1 for s in splits.values())


def test_only_standard_tier_items():
    assert all(it["stage"] >= lore.TIER_STANDARD for it in ITEMS)


def test_answer_not_in_question():
    for it in ITEMS:
        for f in it["answer_forms"]:
            if not f.isdigit():
                assert not name_pattern(f).search(it["question"]), (it["id"], f)
        if it["answer_kind"] in ("relation", "string"):
            assert not name_pattern(it["gold_display"]).search(it["question"]), it["id"]


def test_final_evidence_passage_contains_the_answer():
    for it in ITEMS:
        text = PASSAGES[it["hops"][-1]["evidence_pid"]]["text"]
        if it["answer_kind"] in ("number", "year"):       # "3E 1560s" names the year 1560
            ok = any(number_pattern(f).search(text) for f in it["answer_forms"] if f.replace(",", "").isdigit())
        else:
            ok = any(re.search(r"(?<![A-Za-z0-9])" + re.escape(f) + r"(?![A-Za-z0-9])", text, re.I)
                     for f in it["answer_forms"])
        assert ok, it["id"]


@pytest.mark.parametrize("budget", BUDGETS)
def test_context_holds_evidence_and_respects_budget(budget):
    for it in ITEMS:
        ctx = it["contexts"][str(budget)]
        pids = [s["pid"] for s in ctx["segments"]]
        assert len(pids) == len(set(pids))
        for hop in it["hops"]:
            assert hop["evidence_pid"] in pids, (it["id"], budget)
        if not ctx["over_budget"]:
            assert ctx["tokens"] <= budget, (it["id"], budget)


def test_windowed_evidence_keeps_an_anchor():
    for it in ITEMS:
        for budget in BUDGETS:
            for seg in it["contexts"][str(budget)]["segments"]:
                hop = next((h for h in it["hops"] if h["evidence_pid"] == seg["pid"]), None)
                if hop is None or seg["sent"] is None:
                    continue
                text = segment_text(seg, PASSAGES)
                assert any(name_pattern(a).search(text) for a in hop["anchors"]), (it["id"], budget)


def test_gold_isolation_counterfactual():
    """Changing the private gold (and everything derived from it) leaves every rendered question byte-identical."""
    for it in ITEMS[:: max(1, len(ITEMS) // 40)]:
        fake = copy.deepcopy(it)
        fake.update(gold="COUNTERFACTUAL", gold_display="COUNTERFACTUAL", answer_forms=["COUNTERFACTUAL"])
        for hop in fake["hops"]:
            hop["anchors"] = ["COUNTERFACTUAL"]
        for budget in BUDGETS:
            assert render_question(fake, budget, PASSAGES) == render_question(it, budget, PASSAGES)
            assert "COUNTERFACTUAL" not in render_question(it, budget, PASSAGES)


def test_rendered_question_ends_with_the_question():
    it = ITEMS[0]
    q = render_question(it, BUDGETS[0], PASSAGES)
    assert q.startswith("Read the passages") and q.endswith("Question: " + it["question"])


@pytest.mark.skipif(not (CORPUS.exists() and lore.TOKENIZER.exists()), reason="needs the corpus and the local tokenizer")
def test_rebuild_is_deterministic_and_matches_committed_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ma, mb = lore.build(CORPUS, a), lore.build(CORPUS, b)
    assert ma["items_a_sha256"] == mb["items_a_sha256"] and ma["passages_sha256"] == mb["passages_sha256"]
    assert ma["items_a_sha256"] == MANIFEST["items_a_sha256"], "data/processed/lore is stale: rebuild it"
    assert ma["passages_sha256"] == MANIFEST["passages_sha256"]
