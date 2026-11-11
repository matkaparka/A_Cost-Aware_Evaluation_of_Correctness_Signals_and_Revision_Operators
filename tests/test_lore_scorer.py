"""Hand-checked lore scorer edge cases (design §5.6 asks for >= 50). Run: python -m pytest tests/test_lore_scorer.py -q
The 1.1.0 cases come from the G3b hand check of 4B dev answers (docs/planning/lore_qa/pilot/labels.jsonl), plus guards."""
import pytest

from src.score.lore import equivalent, normalize, score, score_contains

GARNETPORT = {"answer_kind": "relation", "answer_forms": ["Garnetport", "Garnetport City"]}
SOLARI = {"answer_kind": "relation", "answer_forms": ["House Solari", "Solari"]}
PARLIAMENT = {"answer_kind": "relation",
              "answer_forms": ["Four-Chamber Parliament", "four-chamber parliament", "parliament"]}
LYS_SEAT = {"answer_kind": "relation", "answer_forms": ["Ly's Seat"]}
SENNA = {"answer_kind": "relation", "answer_forms": ["Lady Senna Prekurlind", "Senna Prekurlind"]}
DRAGON_SCALE = {"answer_kind": "relation", "answer_forms": ["Dragon Scale", "Dragon Scale currency", "the Dragon Scale"]}
MARQUIS = {"answer_kind": "string", "answer_forms": ["Marquis"]}
ROLE = {"answer_kind": "string", "answer_forms": ["Eldest son and heir"]}
ANVIL = {"answer_kind": "string", "answer_forms": ["anvil-shaped"]}
AGE = {"answer_kind": "number", "answer_forms": ["62"]}
BIG = {"answer_kind": "number", "answer_forms": ["4500000", "4,500,000"]}
YEAR = {"answer_kind": "year", "answer_forms": ["3E 1570", "1570"]}
DECADE = {"answer_kind": "year", "answer_forms": ["3E 1560", "1560"]}
# 1.1.0 (G3b): phrase and name items from the 4B dev hand check
LEN20 = {"answer_kind": "string", "answer_forms": ["twenty meters"]}
THREE = {"answer_kind": "string", "answer_forms": ["three"]}
CENT = {"answer_kind": "string", "answer_forms": ["less than two centuries"]}
MILL = {"answer_kind": "string", "answer_forms": ["two millennia"]}
PCT = {"answer_kind": "string", "answer_forms": ["seven percent"]}
ERRATIC = {"answer_kind": "string", "answer_forms": ["erratic behavior"]}
SEWERS = {"answer_kind": "string", "answer_forms": ["sewers"]}
CHAMBERS = {"answer_kind": "string", "answer_forms": ["her chambers"]}
MOON = {"answer_kind": "string", "answer_forms": ["full moon"]}
TARIFFS = {"answer_kind": "string", "answer_forms": ["special reduced tariffs"]}
NORTHWEST = {"answer_kind": "string", "answer_forms": ["northwest"]}
MINING = {"answer_kind": "string", "answer_forms": ["mining expertise"]}
BROTHER = {"answer_kind": "string", "answer_forms": ["her brother"]}
DAUGHTER = {"answer_kind": "string", "answer_forms": ["Youngest daughter"]}
MORTHAXUS = {"answer_kind": "relation", "answer_forms": ["Morthaxus"]}
MAELIS = {"answer_kind": "relation", "answer_forms": ["Maelis Thornbridge"]}
ILLEGAL = {"answer_kind": "string", "answer_forms": ["Illegal"]}
SSB = {"answer_kind": "string", "answer_forms": ["Southern South Bank"]}
ALDRIC = {"answer_kind": "relation", "answer_forms": ["Aldric Wardell"]}

# (model output, item, expected correct, expected status)
CASES = [
    # --- names: exact and normalized ---
    (r"The economic capital is \boxed{Garnetport}.", GARNETPORT, True, "ok"),
    (r"\boxed{Garnetport City}", GARNETPORT, True, "ok"),
    (r"\boxed{garnetport}", GARNETPORT, True, "ok"),
    (r"\boxed{\text{Garnetport}}", GARNETPORT, True, "ok"),
    (r"$\boxed{\textbf{Garnetport}}$", GARNETPORT, True, "ok"),
    (r"\boxed{Garnetport.}", GARNETPORT, True, "ok"),
    (r"\boxed{\"Garnetport\"}", GARNETPORT, True, "ok"),
    (r"\boxed{the city of Garnetport}", GARNETPORT, True, "ok"),
    (r"\boxed{Deeproot}", GARNETPORT, False, "ok"),
    (r"\boxed{Garnetport and Deeproot}", GARNETPORT, False, "ok"),
    (r"\boxed{not Garnetport}", GARNETPORT, False, "ok"),
    (r"\boxed{Garnet}", GARNETPORT, False, "ok"),
    (r"\boxed{House Solari}", SOLARI, True, "ok"),
    (r"\boxed{Solari}", SOLARI, True, "ok"),
    (r"\boxed{the Solari family}", SOLARI, True, "ok"),
    (r"\boxed{House Severton}", SOLARI, False, "ok"),
    (r"\boxed{\text{Solari's}}", SOLARI, True, "ok"),
    (r"\boxed{House Solari's}", SOLARI, True, "ok"),
    (r"\boxed{Severton's}", SOLARI, False, "ok"),
    (r"\boxed{Four-Chamber Parliament}", PARLIAMENT, True, "ok"),
    (r"\boxed{the four chamber parliament}", PARLIAMENT, True, "ok"),
    (r"\boxed{Four–Chamber Parliament}", PARLIAMENT, True, "ok"),
    (r"\boxed{Lords' Assembly}", PARLIAMENT, False, "ok"),
    (r"\boxed{Ly's Seat}", LYS_SEAT, True, "ok"),
    (r"\boxed{Ly’s Seat}", LYS_SEAT, True, "ok"),
    (r"\boxed{Lys Seat}", LYS_SEAT, False, "ok"),
    (r"\boxed{Senna Prekurlind}", SENNA, True, "ok"),
    (r"\boxed{Lady Senna Prekurlind, née Mereham}", SENNA, True, "ok"),        # 1.1.0: appositive after a comma
    (r"\boxed{Duchess Senna Prekurlind}", SENNA, True, "ok"),
    (r"\boxed{Kira Prekurlind}", SENNA, False, "ok"),
    (r"\boxed{the Dragon Scale}", DRAGON_SCALE, True, "ok"),
    (r"\boxed{Dragon Scales}", DRAGON_SCALE, False, "ok"),
    # --- short strings ---
    (r"\boxed{Marquis}", MARQUIS, True, "ok"),
    (r"\boxed{marquis}", MARQUIS, True, "ok"),
    (r"\boxed{Marchioness}", MARQUIS, False, "ok"),
    (r"\boxed{Count}", MARQUIS, False, "ok"),
    (r"\boxed{Eldest son and heir}", ROLE, True, "ok"),
    (r"\boxed{the eldest son and heir}", ROLE, True, "ok"),
    (r"\boxed{Eldest son}", ROLE, False, "ok"),
    (r"\boxed{heir}", ROLE, False, "ok"),
    (r"\boxed{anvil-shaped}", ANVIL, True, "ok"),
    (r"\boxed{anvil shaped}", ANVIL, True, "ok"),
    (r"\boxed{an anvil-shaped layout}", ANVIL, True, "ok"),
    (r"\boxed{spindle-shaped}", ANVIL, False, "ok"),
    # --- numbers ---
    (r"\boxed{62}", AGE, True, "ok"),
    (r"\boxed{62 years old}", AGE, True, "ok"),
    (r"\boxed{62 years}", AGE, True, "ok"),
    (r"\boxed{about 62}", AGE, True, "ok"),
    (r"\boxed{58}", AGE, False, "ok"),
    (r"\boxed{62 or 58}", AGE, False, "ok"),
    (r"\boxed{sixty-two}", AGE, True, "ok"),                                     # 1.1.0: number words
    (r"\boxed{4,500,000}", BIG, True, "ok"),
    (r"\boxed{4500000 gem}", BIG, False, "ok"),
    # --- years ---
    (r"\boxed{3E 1570}", YEAR, True, "ok"),
    (r"\boxed{1570}", YEAR, True, "ok"),
    (r"\boxed{around 3E 1570}", YEAR, True, "ok"),
    (r"\boxed{3E ~1570}", YEAR, True, "ok"),
    (r"\boxed{3E1570}", YEAR, True, "ok"),
    (r"\boxed{the year 1570}", YEAR, True, "ok"),
    (r"\boxed{2E 1570}", YEAR, False, "ok"),
    (r"\boxed{3E 1580}", YEAR, False, "ok"),
    (r"\boxed{1570 to 1580}", YEAR, False, "ok"),
    (r"\boxed{the 1560s}", DECADE, True, "ok"),
    (r"\boxed{sometime in the 3E 1560s}", DECADE, True, "ok"),
    # --- extraction and status (shared with the math scorer) ---
    ("The answer is Garnetport.", GARNETPORT, True, "ok"),
    ("Final answer: Aldous is 62 years old.", AGE, True, "ok"),
    (r"First I thought \boxed{Deeproot}, but the passage says \boxed{Garnetport}", GARNETPORT, True, "ok"),
    (r"\boxed{Garnetport} or \boxed{Deeproot}", GARNETPORT, False, "conflict"),
    (r"\boxed{Garnetport} or \boxed{Garnetport City}", GARNETPORT, True, "ok"),
    ("<think>maybe Deeproot</think>\n\n\\boxed{Garnetport}", GARNETPORT, True, "ok"),
    ("<think>never closes \\boxed{Garnetport}", GARNETPORT, False, "unparsed"),
    ("I cannot tell from the passages.", GARNETPORT, False, "unparsed"),
    (r"\boxed{}", GARNETPORT, False, "unparsed"),
    ("", GARNETPORT, False, "unparsed"),
    # --- 1.1.0 phrases: number words, units, containment, a dropped first modifier ---
    (r"\boxed{20}", LEN20, True, "ok"),
    (r"\boxed{20 meters}", LEN20, True, "ok"),
    (r"\boxed{about 20 meters}", LEN20, True, "ok"),
    (r"\boxed{20 feet}", LEN20, False, "ok"),
    (r"\boxed{25}", LEN20, False, "ok"),
    (r"\boxed{3}", THREE, True, "ok"),
    (r"\boxed{three}", THREE, True, "ok"),
    (r"\boxed{13}", THREE, False, "ok"),
    (r"\boxed{3 or 4}", THREE, False, "ok"),
    (r"\boxed{less\ than\ 200\ years}", CENT, True, "ok"),
    (r"\boxed{less than 300 years}", CENT, False, "ok"),
    (r"\boxed{2000}", MILL, True, "ok"),
    (r"\boxed{2000 years}", MILL, True, "ok"),
    (r"\boxed{1000 years}", MILL, False, "ok"),
    (r"\boxed{7\%}", PCT, True, "ok"),
    (r"\boxed{70\%}", PCT, False, "ok"),
    (r"\boxed{\text{Roland's erratic behavior}}", ERRATIC, True, "ok"),
    (r"\boxed{\text{sewers of the Middle City}}", SEWERS, True, "ok"),
    (r"\boxed{in her chambers}", CHAMBERS, True, "ok"),
    (r"\boxed{\text{During the full moon}}", MOON, True, "ok"),
    (r"\boxed{not during the full moon}", MOON, False, "ok"),
    (r"\boxed{new moon}", MOON, False, "ok"),
    (r"\boxed{\text{reduced tariffs}}", TARIFFS, True, "ok"),
    (r"\boxed{tariffs}", TARIFFS, False, "ok"),
    (r"\boxed{northwest}", NORTHWEST, True, "ok"),
    (r"\boxed{northwest and northeast}", NORTHWEST, False, "ok"),
    (r"\boxed{either northwest or northeast}", NORTHWEST, False, "ok"),
    (r"\boxed{mining expertise, particularly precious gems and magical crystals}", MINING, True, "ok"),
    (r"\boxed{Brennan Costerham}", BROTHER, False, "ok"),
    (r"\boxed{\text{Rosalind Costerham is the youngest daughter of the Costerham family}}", DAUGHTER, True, "ok"),
    (r"\boxed{\text{Scholar and potential member of the Church of Helixus}}", DAUGHTER, False, "ok"),
    # --- 1.1.0 names: titles, epithets, generic heads, appositives after a separator ---
    (r"\boxed{\text{Morthaxus the Flesh-Sculptor}}", MORTHAXUS, True, "ok"),
    (r"\boxed{Professor\ Maelis\ Thornbridge}", MAELIS, True, "ok"),
    (r"\boxed{\text{Garnetport delegation}}", GARNETPORT, True, "ok"),
    (r"\boxed{\text{Illegal; penalties vary by severity}}", ILLEGAL, True, "ok"),
    (r"\boxed{\text{Viscount Territory: Southern South Bank}}", SSB, True, "ok"),
    (r"\boxed{Aldric Wardell's son Edmund}", ALDRIC, False, "ok"),
    (r"\boxed{Deeproot, not Garnetport}", GARNETPORT, False, "ok"),
    (r"\boxed{Garnetport or Deeproot}", GARNETPORT, False, "ok"),
    (r"\boxed{Deeproot delegation}", GARNETPORT, False, "ok"),
    (r"\boxed{Morthaxus's brood}", MORTHAXUS, False, "ok"),
]


@pytest.mark.parametrize("text,item,correct,status", CASES)
def test_case(text, item, correct, status):
    r = score(text, item)
    assert (r["correct"], r["status"]) == (correct, status), r


def test_at_least_fifty_cases():
    assert len(CASES) >= 50


def test_truncated_is_incorrect_even_if_right():
    assert score(r"\boxed{Garnetport}", GARNETPORT, finish_reason="length") == {
        "status": "truncated", "pred": "Garnetport", "correct": False}


def test_none_prediction():
    assert equivalent(None, GARNETPORT["answer_forms"], "relation") is False


def test_normalize_drops_article_and_markup():
    assert normalize(r"\text{The} **Garnetport**.") == "garnetport"


@pytest.mark.parametrize("text,item,correct", [
    (r"... so the answer is \boxed{Garnetport}.", GARNETPORT, True),
    (r"\boxed{the city of Garnetport}", GARNETPORT, True),
    (r"\boxed{Deeproot}", GARNETPORT, False),
    (r"\boxed{around 3E 1570}", YEAR, True),
    ("He is 62.", AGE, True),
])
def test_second_scorer(text, item, correct):
    assert score_contains(text, item)["correct"] is correct


def test_answer_class_matches_the_builder():
    """The scorer's copy of the answer-class rule agrees with src/data/lore.py on every built item."""
    from src.data import lore, lore_b
    from src.score.lore import answer_class
    items = lore.load_built()[1] + lore_b.load_items_b()
    assert items and all(answer_class(it["answer_kind"], it["answer_forms"][0]) == it["answer_class"] for it in items)
