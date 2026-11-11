"""Lore QA track B invariants (proposal §3.2; design §8): every item passed the mechanical verification of
src/data/lore_b.py, and the item file matches its frozen drafts. Run: python -m pytest tests/test_lore_b.py -q"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.data import lore, lore_b
from src.data.lore import BUDGETS, OUT, load_built, render_question, unit

ITEMS_FILE = OUT / "items_b.jsonl"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def built():
    if not ITEMS_FILE.exists():
        pytest.skip("track B not built yet")
    passages, items_a = load_built()
    return passages, items_a, lore_b.load_items_b(), json.loads((OUT / "build_manifest_b.json").read_text(encoding="utf-8"))


def test_manifest_hashes(built):
    _, _, items, m = built
    assert m["items_b_sha256"] == sha(ITEMS_FILE)
    assert m["b_drafts_sha256"] == sha(OUT / "b_drafts.jsonl")
    assert m["instructions_sha256"] == sha(lore_b.INSTRUCTIONS)
    assert m["counts"]["items"] == len(items)


def test_ids_unique_and_disjoint_from_track_a(built):
    _, items_a, items, _ = built
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)) and not set(ids) & {it["id"] for it in items_a}


def test_every_item_passes_verification(built):
    passages, _, items, _ = built
    for it in items:
        d = {"question": it["question"], "answer": it["gold"]}
        assert lore_b.check(d, passages[it["hops"][0]["evidence_pid"]]) is None, it["id"]


def test_no_duplicate_questions_or_answers_per_passage(built):
    _, _, items, _ = built
    qs = [lore_b._norm(it["question"]) for it in items]
    qa = [(it["hops"][0]["evidence_pid"], lore_b._norm(it["gold"])) for it in items]
    assert len(qs) == len(set(qs)) and len(qa) == len(set(qa))


def test_not_a_duplicate_of_track_a(built):
    _, items_a, items, _ = built
    a = {(it["hops"][-1]["evidence_pid"], lore_b._norm(f)) for it in items_a for f in it["answer_forms"]}
    assert not any((it["hops"][0]["evidence_pid"], lore_b._norm(it["gold"])) in a for it in items)


def test_split_inherits_track_a_or_hashes_the_passage(built):
    passages, items_a, items, _ = built
    a_splits = {}
    for it in items_a:
        for hop in it["hops"]:
            a_splits.setdefault(hop["evidence_pid"], set()).add(it["split"])
    for it in items:
        pid = it["hops"][0]["evidence_pid"]
        if len(a_splits.get(pid, ())) == 1:
            assert it["split"] in a_splits[pid]
        else:
            assert it["cluster"] == f"passage:{passages[pid]['doc']}:{passages[pid]['start']}"
            assert it["split"] == ("dev" if unit("lore-split", it["cluster"]) < lore.DEV_FRACTION else "test")


def test_kinds_and_forms(built):
    _, _, items, _ = built
    for it in items:
        assert it["answer_kind"] in ("number", "year", "relation", "string")
        if it["answer_kind"] == "year":
            assert lore.YEAR.match(it["gold"])
        if it["answer_kind"] == "relation":
            assert it["entity"] and it["gold"] in it["answer_forms"]


@pytest.mark.parametrize("budget", BUDGETS)
def test_context_holds_evidence(built, budget):
    for it in built[2]:
        ctx = it["contexts"][str(budget)]
        assert it["hops"][0]["evidence_pid"] in [s["pid"] for s in ctx["segments"]]
        if not ctx["over_budget"]:
            assert ctx["tokens"] <= budget


def test_gold_isolation_counterfactual(built):
    passages, _, items, _ = built
    for it in items[:: max(1, len(items) // 40)]:
        fake = copy.deepcopy(it)
        fake.update(gold="COUNTERFACTUAL", gold_display="COUNTERFACTUAL", answer_forms=["COUNTERFACTUAL"])
        fake["hops"][0]["anchors"] = ["COUNTERFACTUAL"]
        for budget in BUDGETS:
            assert render_question(fake, budget, passages) == render_question(it, budget, passages)


@pytest.mark.parametrize("q,a,why", [
    ("Is Garnetport the economic capital?", "Garnetport", "yes/no question"),
    ("According to the passage, which city is the economic capital?", "Garnetport", "refers to the passage"),
    ("Which city lies beside Garnetport harbour?", "Garnetport", "answer in the question"),
    ("Which body sits in Deeproot?", "Four-Chamber Parliament", "answer not copied from the passage"),
])
def test_check_rejects(q, a, why):
    passage = {"text": "Garnetport is the economic capital of the Concordat and its largest city."}
    assert lore_b.check({"question": q, "answer": a}, passage) == why


def test_check_rejects_cloze_copies():
    passage = {"text": "The Dragon Scale serves as the Concordat's unified currency, minted in Garnetport under oversight."}
    q = "The Dragon Scale serves as the Concordat's unified currency, minted where?"
    assert lore_b.check({"question": q, "answer": "Garnetport"}, passage) == "copies the passage"
