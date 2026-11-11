"""Prompt invariants (design §6.0, §8): reviewed-solution tokens identical across conditions; the label precedes the
solution; compared wrappers length-matched within +/-3 tokens. Uses the pinned Qwen3 tokenizer and dev items only.
Run: python -m pytest tests/test_render.py -q
"""
import json
from itertools import product
from pathlib import Path

import pytest

from src.gen import render as R

ROOT = Path(__file__).resolve().parents[1]
TOK_DIR = ROOT / "data" / "tokenizers" / "qwen3-4b"
pytestmark = pytest.mark.skipif(not TOK_DIR.exists(), reason="pinned tokenizer not downloaded")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_DIR)


def dev_items(n=40):
    """Dev-only reviewed solutions: injected D-inj solutions and, if synced, 4B D-nat a0 (dev rows only)."""
    out = []
    for line in open(ROOT / "data" / "processed" / "dinj.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "dev":
            out.append((r["question"], r["solution_with_answer"]))
        if len(out) >= n // 2:
            break
    runs = sorted((ROOT / "results" / "p1" / "4B").glob("p1-initial-4B-d2ebdea"))
    if runs:
        q = {json.loads(l)["item_id"]: json.loads(l)["question"] for l in open(ROOT / "data" / "processed" / "dnat.jsonl", encoding="utf-8")}
        for line in open(runs[0] / "initial.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["split"] == "dev":
                out.append((q[r["item_id"]], r["a0"]))
            if len(out) >= n:
                break
    return out


def variants(tok, q, sol):
    for role, label, fb in product(("assistant", "user"), ("self", "other"), ("", R.VERDICT[True], R.VERDICT[False])):
        yield (role, label, fb), R.review_prompt(tok, q, sol, role=role, label=label, feedback=fb)
    yield ("assistant", "self", "F2"), R.review_prompt(tok, q, sol, history=[(R.CRITIQUE_TURN, "Step 2 looks wrong.")])
    yield ("assistant", "self", "F1"), R.review_prompt(tok, q, sol, prefill=R.TRIGGER_PREFILL)


def test_solution_tokens_identical_across_conditions(tok):
    items = dev_items()
    assert len(items) >= 20
    for q, sol in items:
        seqs = {}
        for key, r in variants(tok, q, sol):
            ids, spans = R.token_spans(tok, r)
            a, b = spans["solution"]
            seqs[key] = tuple(ids[a:b])
        assert len(set(seqs.values())) == 1, {k: len(v) for k, v in seqs.items()}


def test_label_precedes_solution(tok):
    for q, sol in dev_items(10):
        for key, r in variants(tok, q, sol):
            assert r.spans["label"][1] <= r.spans["solution"][0], key


@pytest.mark.parametrize("a,b", [
    (R.LABELS["self"], R.LABELS["other"]),
    (R.LABELS_ALT["self"], R.LABELS_ALT["other"]),
    (R.VERDICT[True], R.VERDICT[False]),
    (R.CRITIQUE_TURN, R.SCRATCH_TURN),
])
def test_wrappers_length_matched(tok, a, b):
    la, lb = (len(tok(x, add_special_tokens=False).input_ids) for x in (a, b))
    assert abs(la - lb) <= 3, (la, lb)


def test_verdict_changes_only_the_verdict(tok):
    q, sol = dev_items(2)[0]
    t1 = R.review_prompt(tok, q, sol, feedback=R.VERDICT[True]).text
    t0 = R.review_prompt(tok, q, sol, feedback=R.VERDICT[False]).text
    assert t1.replace("judged it correct.", "judged it incorrect.") == t0


def test_tsui_cells_render(tok):
    r = json.loads(open(ROOT / "data" / "raw" / "gsm8k_sc" / "rows.jsonl", encoding="utf-8").readline())
    internal = R.tsui_prompt(tok, r["messages_error_injection_in_model_bca"])
    assert internal.text.endswith(r["messages_error_injection_in_model_bca"][-1]["content"])
    external = R.tsui_prompt(tok, r["messages_error_in_user_bca"])
    assert external.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert R.tsui_prompt(tok, r["messages_error_injection_in_model_bca"], wait=True).text.endswith(" Wait")
