"""Invariants of the ARR-round run code (src/exp/arr_runs.py; plan docs/planning/arr_round/plan.md, Amendment 2).
CPU only: no model, no tokenizer download."""
import pytest

from src.exp import arr_runs as AR
from src.exp import thinking as T
from src.gen import h2a_extra as H
from src.gen import render as R
from src.gen.initial import PROMPT as INITIAL_PROMPT


def test_main_wording_is_the_frozen_operator_wording():
    assert AR.WORDINGS["main"] == {"F2": R.CRITIQUE_TURN, "F2-scratch": R.SCRATCH_TURN}


def test_paraphrases_are_distinct_and_withhold_the_final_answer():
    texts = [AR.WORDINGS[w][c] for w in AR.WORDINGS for c in ("F2", "F2-scratch")]
    assert len(set(texts)) == len(texts)
    for w in ("p1", "p2"):
        for c in ("F2", "F2-scratch"):
            t = AR.WORDINGS[w][c]
            assert "final answer" in t and "boxed" not in t
            assert abs(len(t) - len(AR.WORDINGS["main"][c])) <= 30   # roughly length-matched


def test_sample_aware_instruction():
    s = AR.sample_aware_instruction(["18", None, "\\frac{1}{2}", "18", "18"], "problem")
    assert s.startswith("Five other attempts at this problem ended with these final answers: 18; no final answer; ")
    assert s.endswith("Put your final answer within \\boxed{}.")
    assert "\\frac{1}{2}" in s
    with pytest.raises(AssertionError):
        AR.sample_aware_instruction(["A"] * 4, "question")


def test_prompts_and_seeds_match_the_stored_runs():
    assert R.PROMPT == INITIAL_PROMPT
    assert (AR.TEMPERATURE, AR.TOP_P, AR.TOP_K) == (H.TEMPERATURE, H.TOP_P, H.TOP_K) == (0.7, 0.8, 20)
    assert H.seed_for("dnat-gsm8k-0") == H.seed_for("dnat-gsm8k-0", "h2a-consistency")
    assert H.seed_for("dnat-gsm8k-0", "h2a-consistency-ext") != H.seed_for("dnat-gsm8k-0")
    assert H.seed_for("dnat-gsm8k-0", "a0-long") != H.seed_for("dnat-gsm8k-0")
    # C1 uses the P4 thinking seed of index 0 and its sampling settings
    assert AR.think_seed("dnat-math500-1") == int(T.h("think-seed-v1", "dnat-math500-1", 0)[:8], 16)
    assert AR.THINK_SAMPLING == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def test_budgets_match_the_runner():
    assert (AR.A0_TOKENS, AR.REVISION_TOKENS, AR.CRITIQUE_TOKENS) == (4096, 2048, 1024)


def test_mc_think_scoring_strips_the_think_block():
    item = {"item_id": "x", "gold": "B", "n_options": 4, "options": ["a", "b", "c", "d"], "question": "q"}
    its = {"x": item}
    ok = AR.score_text("mc", "<think>\nmaybe \\boxed{A}\n</think>\n\nThe answer is \\boxed{B}.", "stop", "x", {}, its)
    assert ok["correct"] and ok["pred"] == "B"
    open_block = AR.score_text("mc", "<think>\nstill thinking \\boxed{B}", "length", "x", {}, its)
    assert not open_block["correct"] and open_block["status"] == "unclosed"
    lost_tag = AR.score_text("mc", "reasoning \\boxed{A}\n</think>\n\n\\boxed{B}", "stop", "x", {}, its)
    assert lost_tag["pred"] == "B"


# ---- think-budget (plan Amendment 7) ------------------------------------------------------------------------------
def test_stop_thinking_is_the_qwen3_report_instruction():
    assert AR.STOP_THINKING == ("\n\nConsidering the limited time by the user, I have to give the solution based on "
                                "the thinking directly now.\n</think>.\n\n")
    assert AR.BUDGET_ANSWER_TOKENS == 2048
    assert AR.EXP["think-budget"] == "c1b"


def test_budget_seed_depends_on_item_and_budget():
    s = {AR.budget_seed(i, b) for i in ("dnat-gsm8k-0", "dnat-math500-1") for b in (256, 512, 1024, 2048)}
    assert len(s) == 8
    assert AR.budget_seed("dnat-gsm8k-0", 256) == int(AR.h("think-budget-v1", "dnat-gsm8k-0", 256)[:8], 16)
    assert AR.budget_seed("dnat-gsm8k-0", 256) != AR.think_seed("dnat-gsm8k-0")


def test_budget_cut():
    O, C = 1, 2   # stand-ins for the <think> and </think> token IDs
    trace = [O] + [9] * 10 + [C] + [7] * 5            # 10 thinking tokens, then the answer
    assert AR.budget_cut(trace, O, C, 10) == (10, None)            # fits exactly: reuse the stored output
    assert AR.budget_cut(trace, O, C, 4) == (10, 5)                # <think> and the first 4 thinking tokens
    assert trace[:5] == [O, 9, 9, 9, 9]
    open_only = [O] + [9] * 20                         # the C1 output stopped while thinking
    assert AR.budget_cut(open_only, O, C, 8) == (20, 9)
    assert AR.budget_cut(open_only, O, C, 20) == (20, None)
    assert AR.budget_cut([7, 7, 7], O, C, 1) == (0, None)          # no thinking at all
    lead = [5, O] + [9] * 3 + [C]                      # tokens before the opening tag are kept in the prefix
    assert AR.budget_cut(lead, O, C, 2) == (3, 4)


class _Tok:
    """Character-level stand-in tokenizer: '<think>' and '</think>' are single tokens, every other character one token."""
    SPECIAL = {"<think>": 1, "</think>": 2}

    def convert_tokens_to_ids(self, t):
        return self.SPECIAL[t]

    def _ids(self, text):
        out, i = [], 0
        while i < len(text):
            for s, k in self.SPECIAL.items():
                if text.startswith(s, i):
                    out.append(k)
                    i += len(s)
                    break
            else:
                out.append(1000 + ord(text[i]))
                i += 1
        return out

    def __call__(self, text, add_special_tokens=False):
        class R:
            pass
        r = R()
        r.input_ids = self._ids(text)
        return r

    def decode(self, ids):
        inv = {v: k for k, v in self.SPECIAL.items()}
        return "".join(inv.get(i, chr(i - 1000) if i >= 1000 else "") for i in ids)

    bos_token = None


def test_build_think_budget_reuses_or_forces(monkeypatch):
    tok = _Tok()
    monkeypatch.setattr(AR, "build_think", lambda tok, rows, task, q, its: ["PROMPT"])
    short = "<think>\nab\n</think>\n\n\\boxed{1}"     # 4 thinking tokens
    long = "<think>\nabcdefgh\n</think>\n\n\\boxed{2}"  # 10 thinking tokens
    # a stopped output's stored out_tokens counts its end-of-turn token, which its text does not hold
    traces = {"s": {"text": short, "out_tokens": len(tok._ids(short)) + 1, "prompt_tokens": 6, "finish_reason": "stop"},
              "l": {"text": long, "out_tokens": len(tok._ids(long)) + 2, "prompt_tokens": 7, "finish_reason": "stop"}}
    rows = [{"item_id": "s"}, {"item_id": "l"}]
    built, stats = AR.build_think_budget(tok, rows, "math", {}, {}, traces, budget=4)
    (p0, r0), (p1, r1) = built
    assert p0 is None and r0["reused"] and r0["think_tokens_stored"] == 4
    assert not r1["reused"] and r1["think_tokens_stored"] == 10 and r1["trace_tokens_used"] == 5
    assert r1["prefix_text"] == "<think>\nabc"
    assert p1 == tok._ids("PROMPT") + tok._ids("<think>\nabc") + tok._ids(AR.STOP_THINKING)
    assert tok._ids(AR.STOP_THINKING).count(2) == 1   # the instruction closes the think block
    assert stats["fits"] == 1 and stats["over_budget"] == 1
    assert stats["roundtrip_mismatch"] == 1           # the long trace's stored out_tokens was set off by one
    assert stats["prompt_len_mismatch"] == 1          # 'PROMPT' is 6 tokens, the stored count says 7
    assert stats["stop_tokens"] == len(tok._ids(AR.STOP_THINKING))


def test_forced_text_scores_on_the_answer_after_the_think_block():
    q = {"x": {"question": "1+1?", "gold": "2"}}
    forced = "<think>\nwe add one and one, maybe \\boxed{3}" + AR.STOP_THINKING + "The answer is \\boxed{2}."
    assert AR.score_text("math", forced, "stop", "x", q, {})["correct"]
    assert not AR.score_text("math", forced, "length", "x", q, {})["correct"]   # a forced answer cut at its limit
    item = {"item_id": "y", "gold": "B", "n_options": 4, "options": ["a", "b", "c", "d"], "question": "q"}
    forced_mc = "<think>\nperhaps \\boxed{A}" + AR.STOP_THINKING + "\\boxed{B}"
    assert AR.score_text("mc", forced_mc, "stop", "y", {}, {"y": item})["pred"] == "B"
