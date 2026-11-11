"""Thinking-subset invariants (src/exp/thinking.py; DEVIATIONS D7): every prompt equals the chat template rendered with
enable_thinking=True (the F2 critique's think block dropped as the template drops it); F1's trigger sits inside the
think block; thinking outputs are scored on the text after the think block, and an unclosed one is wrong. Uses the
pinned Qwen3 tokenizer and dev items only.
Run: python -m pytest tests/test_thinking.py -q
"""
import json
from pathlib import Path

import pytest

from src.exp.thinking import THINK_PREFILL, thinking, turn1_requests
from src.gen import render as R
from src.score.scorer import score

ROOT = Path(__file__).resolve().parents[1]
TOK_DIR = ROOT / "data" / "tokenizers" / "qwen3-4b"
pytestmark = pytest.mark.skipif(not TOK_DIR.exists(), reason="pinned tokenizer not downloaded")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_DIR)


def dev_items(n=20):
    out = []
    for line in open(ROOT / "data" / "processed" / "dinj.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "dev":
            out.append({"item_id": r["item_id"], "question": r["question"], "a0": r["solution_with_answer"],
                        "correct0": False})
        if len(out) == n:
            break
    return out


def ref(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def test_prompts_equal_thinking_template(tok):
    for r in dev_items():
        q, a0 = r["question"], r["a0"]
        head = R.PROMPT.format(q=q)
        assert thinking(R.fresh_prompt(tok, q)) == ref(tok, [{"role": "user", "content": head}])
        for role, label in (("assistant", "self"), ("assistant", "other"), ("user", "self"), ("user", "other")):
            msgs = [{"role": "user", "content": f"{head}\n\n{R.LABELS[label]}"}, {"role": role, "content": a0},
                    {"role": "user", "content": R.INSTRUCTION}]
            assert thinking(R.review_prompt(tok, q, a0, role=role, label=label)) == ref(tok, msgs)
        crit = "The answer looks right."
        msgs = [{"role": "user", "content": f"{head}\n\n{R.LABELS['self']}"}, {"role": "assistant", "content": a0},
                {"role": "user", "content": R.CRITIQUE_TURN},
                {"role": "assistant", "content": "<think>\nchecking\n</think>\n\n" + crit},
                {"role": "user", "content": R.INSTRUCTION}]
        assert thinking(R.review_prompt(tok, q, a0, history=[(R.CRITIQUE_TURN, crit)])) == ref(tok, msgs)


def test_turn1_requests(tok):
    r = dev_items(1)[0]
    reqs = {c: p for c, p, _ in turn1_requests(tok, r, False)}
    assert len(reqs) == 10 and "1B:assistant/self/F0" not in reqs   # shares the F0 generation
    assert reqs["F1"] == reqs["F0"] + THINK_PREFILL and THINK_PREFILL == "<think>\nWait,"
    assert all(p.rstrip().endswith("assistant") or p.endswith(THINK_PREFILL) for p in reqs.values())


def test_scoring_thinking_text():
    assert score("<think>\nWait, 2+3=5.\n</think>\n\nSo the answer is \\boxed{5}.", "5")["correct"]
    assert score("<think>\nWait, maybe \\boxed{5}", "5", finish_reason="stop")["status"] == "unparsed"
    assert not score("<think>\nWait, still", "5", finish_reason="length")["correct"]
