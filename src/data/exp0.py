"""Exp 0 hidden-state items (design §5.5): ~200 GSM8K-style templates, each stating one quantity as "either X or Y
(written on a card you cannot see)". Each template yields two paired worlds with the same visible prompt:
world 0 (hidden value X, gold g0) and world 1 (hidden value Y, gold g1), with g0 != g1 asserted.
The prompt shows both final answers as labelled options (A)/(B); the option order is seeded per template.

Randomness: numbers from stream "exp0-numbers", option order from "exp0-options", split from "split-v1",
each derived as hash(stream, id). Hidden values and seeds never appear in the rendered prompt beyond the two
stated alternatives (checked in tests/test_exp0.py).
Output: data/processed/exp0.jsonl (one row per world). Usage: python -m src.data.exp0
"""
import json
import random
from pathlib import Path

from .build import DEV_FRACTION, h

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "exp0.jsonl"
VARIANTS_PER_BASE = 10
CARD = "(the exact number is written on a card you cannot see)"

# Each base: (text with {hidden} and named slots, slot ranges, answer function of (hidden, slots), hidden range)
BASES = [
    ("A baker makes {a} loaves of bread every hour and works for either {hidden} hours {card}. How many loaves does the baker make?",
     {"a": (3, 12)}, lambda x, s: s["a"] * x, (2, 9)),
    ("Mia buys either {hidden} notebooks {card} at ${a} each. How many dollars does she spend?",
     {"a": (2, 9)}, lambda x, s: s["a"] * x, (3, 12)),
    ("A farmer has {a} cows. He buys either {hidden} more cows {card}. How many cows does he have now?",
     {"a": (10, 60)}, lambda x, s: s["a"] + x, (3, 20)),
    ("Tom had {a} marbles and gave away either {hidden} of them {card}. How many marbles does Tom have left?",
     {"a": (30, 90)}, lambda x, s: s["a"] - x, (4, 25)),
    ("A bus carries either {hidden} passengers on each trip {card} and makes {a} trips. How many passengers does it carry in total?",
     {"a": (2, 8)}, lambda x, s: s["a"] * x, (12, 45)),
    ("Lena reads {a} pages a day for either {hidden} days {card}. How many pages does she read?",
     {"a": (5, 30)}, lambda x, s: s["a"] * x, (3, 10)),
    ("A box holds either {hidden} apples {card}. Sam fills {a} boxes and eats {b} apples. How many apples are left?",
     {"a": (2, 7), "b": (1, 5)}, lambda x, s: s["a"] * x - s["b"], (8, 20)),
    ("A garden has {a} rows of plants with either {hidden} plants in each row {card}. How many plants are in the garden?",
     {"a": (3, 12)}, lambda x, s: s["a"] * x, (4, 15)),
    ("Noah saves ${a} every week. After either {hidden} weeks {card}, he spends ${b}. How many dollars does he have?",
     {"a": (5, 25), "b": (1, 20)}, lambda x, s: s["a"] * x - s["b"], (3, 10)),
    ("A class has {a} students. Either {hidden} of them {card} are absent today. How many students are present?",
     {"a": (20, 40)}, lambda x, s: s["a"] - x, (1, 9)),
    ("A shop sells pens in packs of either {hidden} {card}. Ava buys {a} packs. How many pens does she buy?",
     {"a": (2, 9)}, lambda x, s: s["a"] * x, (3, 12)),
    ("A train travels {a} miles every hour for either {hidden} hours {card}. How many miles does it travel?",
     {"a": (30, 80)}, lambda x, s: s["a"] * x, (2, 6)),
    ("Ella has {a} stickers. Her friend gives her either {hidden} times as many stickers as she has {card}. How many stickers does Ella have now?",
     {"a": (3, 15)}, lambda x, s: s["a"] + s["a"] * x, (2, 5)),
    ("A water tank holds {a} liters. Either {hidden} liters {card} are used each day for {b} days. How many liters are left?",
     {"a": (200, 500), "b": (2, 5)}, lambda x, s: s["a"] - s["b"] * x, (10, 40)),
    ("Jack runs either {hidden} laps {card} every morning and {a} laps every evening. How many laps does he run in {b} days?",
     {"a": (1, 5), "b": (2, 7)}, lambda x, s: s["b"] * (x + s["a"]), (2, 8)),
    ("A pizza is cut into either {hidden} slices {card}. A family orders {a} pizzas and eats {b} slices. How many slices are left?",
     {"a": (2, 5), "b": (3, 7)}, lambda x, s: s["a"] * x - s["b"], (6, 12)),
    ("A library gets {a} new books every month. It already has either {hidden} books {card}. How many books will it have after {b} months?",
     {"a": (10, 40), "b": (2, 6)}, lambda x, s: x + s["a"] * s["b"], (100, 400)),
    ("Ruby earns ${a} per hour and works either {hidden} hours {card} each day for {b} days. How many dollars does she earn?",
     {"a": (8, 20), "b": (2, 5)}, lambda x, s: s["a"] * x * s["b"], (3, 8)),
    ("A parking lot has {a} spaces. Either {hidden} cars {card} are parked there. How many spaces are empty?",
     {"a": (50, 120)}, lambda x, s: s["a"] - x, (10, 45)),
    ("Leo splits {a} candies equally among either {hidden} friends {card}, giving each the same whole number of candies and keeping the rest. How many candies does each friend get?",
     {"a": (40, 100)}, lambda x, s: s["a"] // x, (3, 9)),
]


def make_items():
    rows = []
    for bi, (text, slots, fn, (lo, hi)) in enumerate(BASES):
        for v in range(VARIANTS_PER_BASE):
            tid = f"e0-{bi:02d}-{v:02d}"
            rng = random.Random(h("exp0-numbers", tid))
            for _ in range(1000):
                s = {k: rng.randint(a, b) for k, (a, b) in slots.items()}
                x, y = rng.sample(range(lo, hi + 1), 2)
                g0, g1 = fn(x, s), fn(y, s)
                if g0 != g1 and g0 > 0 and g1 > 0:
                    break
            else:
                raise AssertionError(f"{tid}: no valid draw")
            assert x != y and g0 != g1
            story = text.format(hidden=f"{x} or {y}", card=CARD, **s)
            a_is_g0 = int(h("exp0-options", tid), 16) % 2 == 0
            opt_a, opt_b = (g0, g1) if a_is_g0 else (g1, g0)
            prompt = (f"{story}\n\nThe two possible final answers are:\n(A) {opt_a}\n(B) {opt_b}\n\n"
                      "Reason briefly, then put only the letter of your choice (A or B) within \\boxed{}.")
            split = None  # assigned below, per template, so both worlds share it
            for world, (hidden, gold) in enumerate([(x, g0), (y, g1)]):
                rows.append({"item_id": f"{tid}-w{world}", "template_id": tid, "world": world, "hidden_value": hidden,
                             "gold_value": gold, "gold_letter": "A" if gold == opt_a else "B", "prompt": prompt,
                             "options": {"A": opt_a, "B": opt_b}, "split": split})
    tids = sorted({r["template_id"] for r in rows}, key=lambda t: h("split-v1", t))
    n_dev = round(DEV_FRACTION * len(tids))
    split = {t: ("dev" if k < n_dev else "test") for k, t in enumerate(tids)}
    for r in rows:
        r["split"] = split[r["template_id"]]
    return rows


def main():
    rows = make_items()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"templates": len({r["template_id"] for r in rows}), "items": len(rows),
                      "dev_templates": len({r["template_id"] for r in rows if r["split"] == "dev"})}))


if __name__ == "__main__":
    main()
