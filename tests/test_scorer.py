"""Hand-checked scorer edge cases (design §5.6 requires >= 50). Run: python -m pytest tests/test_scorer.py -q"""
import pytest

from src.score.scorer import equivalent, extract, extract_option, score, score_simple

# (model output, gold, expected correct, expected status)
CASES = [
    # --- extraction: \boxed ---
    (r"So the total is \boxed{18}.", "18", True, "ok"),
    (r"First \boxed{5} was wrong. Redo: the answer is \boxed{7}", "7", True, "ok"),
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", True, "ok"),
    (r"\boxed{\frac{\sqrt{3}}{2}}", r"\frac{\sqrt{3}}{2}", True, "ok"),
    (r"nested \boxed{\left( 3, \frac{\pi}{2} \right)}", r"\left( 3, \frac{\pi}{2} \right)", True, "ok"),
    (r"\fbox{42}", "42", True, "ok"),
    (r"\boxed {12}", "12", True, "ok"),
    (r"The answer is \boxed{}.", "3", False, "unparsed"),
    (r"$\boxed{18}$ dollars", "18", True, "ok"),
    (r"Step 1 ... \boxed{3}" + "\n\n" + r"Final: \boxed{3}", "3", True, "ok"),
    # --- extraction: fallbacks ---
    ("The answer is 9.", "9", True, "ok"),
    ("Therefore, the final answer is: $12$", "12", True, "ok"),
    ("Final Answer: 7", "7", True, "ok"),
    ("**Final answer:** 15", "15", True, "ok"),
    ("Janet sells 9 eggs.\n#### 18", "18", True, "ok"),
    ("I am not sure how to solve this.", "4", False, "unparsed"),
    ("", "4", False, "unparsed"),
    (r"The answer is \(5\).", "5", True, "ok"),
    # --- thinking ---
    ("<think>maybe 4? no, 5</think>\n\nSo \\boxed{5}", "5", True, "ok"),
    ("<think>long reasoning that never ends \\boxed{5}", "5", False, "unparsed"),
    ("<think>\\boxed{4}</think>\n\nThe answer is \\boxed{5}", "4", False, "ok"),
    # --- numbers ---
    (r"\boxed{18.00}", "18", True, "ok"),
    (r"\boxed{1,000}", "1000", True, "ok"),
    (r"\boxed{1000}", "1000", True, "ok"),
    (r"\boxed{-5.0}", "-5", True, "ok"),
    (r"\boxed{0.5}", r"\frac{1}{2}", True, "ok"),
    (r"\boxed{-0.75}", r"-\frac{3}{4}", True, "ok"),
    (r"\boxed{\frac{2}{4}}", r"\frac12", True, "ok"),
    (r"\boxed{2.0000001}", "2", False, "ok"),
    (r"\boxed{17}", "18", False, "ok"),
    (r"\boxed{3.14159}", r"\pi", False, "ok"),
    # --- units, currency, percent, degrees ---
    (r"\boxed{\$18}", "18", True, "ok"),
    (r"\boxed{18 \text{ dollars}}", "18", True, "ok"),
    (r"\boxed{5\text{ cm}}", "5", True, "ok"),
    (r"\boxed{30^\circ}", "30", True, "ok"),
    (r"\boxed{10\%}", "10", True, "ok"),
    # --- algebraic forms ---
    (r"\boxed{\sqrt{12}}", r"2\sqrt{3}", True, "ok"),
    (r"\boxed{\frac{1}{2}\sqrt{3}}", r"\frac{\sqrt{3}}{2}", True, "ok"),
    (r"\boxed{x = 3}", "3", True, "ok"),
    (r"\boxed{(3,\frac{\pi}{2})}", r"\left( 3, \frac{\pi}{2} \right)", True, "ok"),
    # --- intervals and sets ---
    (r"\boxed{(0, 1]}", "(0,1]", True, "ok"),
    (r"\boxed{[0,1]}", "(0,1)", False, "ok"),
    (r"\boxed{(-\infty,2)}", r"(-\infty, 2)", True, "ok"),
    (r"\boxed{\{2,1\}}", r"\{1,2\}", True, "ok"),
    (r"\boxed{3, 2}", "2, 3", True, "ok"),
    # --- text answers ---
    (r"\boxed{\text{(C)}}", "C", True, "ok"),
    (r"\boxed{Evelyn}", r"\text{Evelyn}", True, "ok"),
    # --- multiple boxes in the final paragraph ---
    (r"The answer is \boxed{3} or \boxed{4}.", "3", False, "conflict"),
    (r"$\boxed{3}$, $\boxed{4}$", "4", False, "conflict"),
    (r"Previously I got \boxed{5}, but the correct answer is \boxed{7}.", "7", True, "ok"),
    (r"My earlier answer \boxed{7} is wrong; it should be \boxed{5}.", "7", False, "ok"),
    (r"So $x = \boxed{2}$ and $x = \boxed{3}$.", "2, 3", True, "ok"),
    (r"\boxed{6} which equals \boxed{6.0}", "6", True, "ok"),
    # --- scli5-style simple tasks ---
    (r"\boxed{B}", "B", True, "ok"),
    (r"\boxed{b}", "B", True, "ok"),
    (r"The answer is 2.", "2", True, "ok"),
]


@pytest.mark.parametrize("text,gold,correct,status", CASES)
def test_score(text, gold, correct, status):
    r = score(text, gold)
    assert (r["correct"], r["status"]) == (correct, status), r


def test_case_count():
    assert len(CASES) >= 50


def test_truncated_is_wrong_even_with_answer():
    r = score(r"\boxed{18}", "18", finish_reason="length")
    assert r == {"status": "truncated", "pred": "18", "correct": False}


def test_extract_returns_last_box():
    assert extract(r"\boxed{1} ... \boxed{2}")[0] == "2"


def test_equivalent_none():
    assert equivalent(None, "1") is False


@pytest.mark.parametrize("text,expected", [
    (r"\boxed{A}", "A"), (r"\boxed{(B)}", "B"), (r"\boxed{\text{A}}", "A"), (r"\boxed{C}", None),
    ("I choose A", None), (r"\boxed{A} or \boxed{B}", None), (r"\boxed{a}", "A"),
])
def test_extract_option(text, expected):
    assert extract_option(text) == expected


@pytest.mark.parametrize("text,gold,correct", [
    (r"\boxed{18}", "18", True), (r"\boxed{18.0}", "18", True), (r"\boxed{\frac{1}{2}}", "0.5", True),
    (r"\boxed{17}", "18", False), ("no answer", "1", False),
])
def test_score_simple(text, gold, correct):
    assert score_simple(text, gold)["correct"] == correct


# ---- 1.1.0: "answer is" variants and the Tsui continuation rule (P2 hand checks) ---------------------------------

@pytest.mark.parametrize("text,pred", [
    ("The correct answer is 8.", "8"),
    ("the answer is **8**. If there's a specific context, please clarify!", "8"),
    ("The answer to **5 - 1** is **4**, not 3.", "4"),
    ("So the answer is 3.5.", "3.5"),
    ("This is the answer: 12", "12"),
    ("The answer to this question is 7", "7"),
])
def test_answer_is_variants(text, pred):
    assert extract(text) == (pred, "ok", [])


def test_answer_is_keeps_latex_fraction():
    assert extract(r"The final answer is: $\frac{1}{2}$.")[0] == r"\frac{1}{2}"


def test_last_number_only_when_asked():
    text = " So his net profit is $28.00."
    assert extract(text)[1] == "unparsed"
    assert extract(text, last_number=True) == ("28.00", "last_number", [])
    assert score(text, "28", last_number=True) == {"status": "last_number", "pred": "28.00", "correct": True}


def test_last_number_does_not_override_answer_is():
    assert extract("The answer to 5 - 1 is 4, not 3.", last_number=True)[0] == "4"


def test_last_number_empty_continuation():
    assert score("", "4", last_number=True)["status"] == "unparsed"


@pytest.mark.parametrize("text,pred", [
    ("### Final Answer: **8 pieces**.", "8"),
    ("### Final Answer: **$9.00** is what Chad still owes.", "9.00"),
    ("**Answer: 19 is larger than 6.**", "19"),
    ("The answer is 2 and 3.", "2 and 3"),
])
def test_answer_is_leading_number(text, pred):
    assert extract(text)[0] == pred


@pytest.mark.parametrize("text,pred", [
    ("**Answer: Becca has 153 cards.**", "153"),
    ("### ✅ Final Answer: The seller will give back **$500** in change.", "500"),
    ("### Final Answer:\n\nAlex charges **\\$168** for 2 weeks of tutoring.", "168"),
    ("### ✅ Final Answer:\nMaggie should set the oven to **500°F** to get the actual temperature of **520°F**.", "500"),
    ("The answer is $x^2$.", "x^2"),
    (r"The answer is $\sqrt{2}$.", r"\sqrt{2}"),
])
def test_answer_sentence_forms(text, pred):
    assert extract(text)[0] == pred


# ---- 1.1.1 guard (DEVIATIONS 2026-11-05): runs only where the guard is active (Linux; the run host's gate) ----

linux_only = pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="the guard is Linux-only")


@linux_only
@pytest.mark.parametrize("pred", [r"10^{10^{10}}", r"9^{9^{9}}", r"(x+1)^{100000}", r"10^{100000000000}"])
def test_guard_stops_non_terminating(pred):
    import time
    from src.score import scorer
    t = time.time()
    assert equivalent(pred, "5") is False
    assert time.time() - t < 2 * scorer.GUARD_CPU_S + 10   # stopped by a limit, or math-verify gave up itself


@linux_only
def test_guard_keeps_ordinary_verdicts():
    from src.score import scorer
    n = len(scorer.GUARDED)
    for text, gold, correct, status in CASES:
        s = score(text, gold)
        assert (s["correct"], s["status"]) == (correct, status), text
    assert equivalent(r"\dfrac12", "0.5") and not equivalent("3", "4") and equivalent(r"2^{10}", "1024")
    assert len(scorer.GUARDED) == n


def test_string_equality_first_and_cache():
    from src.score import scorer
    assert equivalent("1,000", "1000")
    assert equivalent(r"\frac{1}{2}", "0.5") and equivalent(r"\frac{1}{2}", "0.5")
    assert ("_symbolic", r"\frac{1}{2}", "0.5") in scorer._CACHE
