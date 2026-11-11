"""Second scorer v2 (src/score/second.py): the MATH reference normalization plus numeric equality."""
from src.score.second import equivalent_v2, score_v2, strip_string


def test_math_normalization():
    assert strip_string("\\frac43") == "\\frac{4}{3}"
    assert strip_string("\\dfrac{1}{2}") == "\\frac{1}{2}"
    assert strip_string("0.5") == "\\frac{1}{2}"
    assert strip_string("3/4") == "\\frac{3}{4}"
    assert strip_string("\\sqrt3") == "\\sqrt{3}"
    assert strip_string("x = 5") == "5"
    assert strip_string("90^\\circ") == "90"
    assert strip_string("\\$36") == "36"
    assert strip_string("12 \\text{ cm}") == "12"
    assert strip_string(".25") == "0.25"


def test_equivalence():
    assert equivalent_v2("\\frac43", "\\frac{4}{3}")
    assert equivalent_v2("18", "18\\text{ ways.}")
    assert equivalent_v2("1.50", "1.5")
    assert equivalent_v2("-\\frac{1}{4}", "-0.25")
    assert not equivalent_v2("2", "3")
    assert not equivalent_v2(None, "3")


def test_score_v2():
    assert score_v2("So the answer is \\boxed{\\frac43}.", "\\frac{4}{3}")["correct"]
    assert score_v2("\\boxed{5}", "5", finish_reason="length") == {"status": "truncated", "correct": False}
    assert score_v2("no answer here", "5")["status"] == "unparsed"
