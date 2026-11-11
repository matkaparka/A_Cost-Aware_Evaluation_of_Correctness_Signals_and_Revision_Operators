"""Lore QA, D-inj-lore (docs/planning/lore_qa/proposal.md §3.3): an answer attributed to the model, right or wrong by
construction, so that W->R and R->W are measurable however accurate the model's own answers are.

For every track A and track B item and every context budget, a0 is A0_TEMPLATE filled with either the gold (variant
"right") or a wrong value of the same kind that occurs in that item's own context (variant "wrong"). The variant is a
seeded coin per item (stream lore-inj), shared by all budgets. Wrong values:
- relation: another canon entity of the gold's card type named in the context (not named in the question);
- number: another number in the context, within a factor of 3 of the gold;
- year: another year of the same era in the context;
- string (track A only, where the attribute is known): another value of the same attribute named in the context.
The wrong value is the seeded choice among the candidates. An item whose coin says "wrong" but whose context has no
candidate gets no row at that budget (counted in the manifest); it is never switched to "right", so the coin alone
decides the variant. The wrong-variant rows therefore exclude items without a same-kind distractor in context.

Output: data/processed/lore/items_inj.jsonl, build_manifest_inj.json.
Usage: python -m src.data.lore_inj [--corpus D:/fandom/worldbuilding]
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.data import lore, lore_b
from src.data.lore import BUDGETS, OUT, YEAR, h, name_pattern, norm_text, segment_text, sha256_file, unit

A0_TEMPLATE = "From the passages, the answer is \\boxed{{{value}}}."
_NUM = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,])|(?<![\d.,])\d+(?![\d,])")
_YEARS = re.compile(r"\b([123])E\s*~?\s*(\d{1,4})\b")


def context_text(it, budget, passages):
    return " ".join(segment_text(s, passages) for s in it["contexts"][str(budget)]["segments"])


def wrong_candidates(it, budget, passages, ents, english, attr_values):
    text = context_text(it, budget, passages)
    gold_norm = {f.casefold() for f in it["answer_forms"]}
    kind = it["answer_kind"]
    if kind == "relation":
        gold_id = it["gold"] if it.get("route") != "B" else it.get("entity")
        gtype = ents.get(gold_id, {}).get("type")
        out = []
        for eid, forms in english.items():
            e = ents[eid]
            if eid == gold_id or e.get("status") != "canon" or e.get("type") != gtype:
                continue
            if any(f.casefold() in gold_norm for f in forms):
                continue
            if any(name_pattern(f).search(it["question"]) for f in forms):
                continue
            shown = [f for f in forms if name_pattern(f).search(text)]
            if shown:
                out.append(norm_text(str(e.get("name"))) if name_pattern(norm_text(str(e.get("name")))).search(text)
                           else sorted(shown)[0])
        return sorted(set(out))
    if kind == "number":
        g = float(it["answer_forms"][0])
        vals = {m.group(0) for m in _NUM.finditer(text)}
        return sorted(v for v in vals if (x := float(v.replace(",", ""))) != g and g / 3 <= x <= g * 3)
    if kind == "year":
        era, year = re.match(r"^([123])E (\d{1,4})$", it["answer_forms"][0]).groups()
        return sorted({f"{e}E {y}" for e, y in _YEARS.findall(text) if e == era and y != year})
    if kind == "string" and it.get("route") != "B" and len(it["attrs"]) >= 1:
        attr = it["attrs"][-1]
        return sorted(v for v in attr_values.get(attr, ()) if v.casefold() not in gold_norm
                      and re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", text, re.I))
    return []


def build(corpus):
    passages, items_a = lore.load_built()
    items_b = lore_b.load_items_b() if (OUT / "items_b.jsonl").exists() else []
    ents, _ = lore.load_entities(corpus)
    english = {eid: forms for eid, e in ents.items() if (forms := lore.surface_forms(e))}
    attr_values = defaultdict(set)                         # track A string attributes -> their values
    for it in items_a:
        if it["answer_kind"] == "string":
            attr_values[it["attrs"][-1]].add(it["gold"])
    rows, missing = [], Counter()
    for it in items_a + items_b:
        route = it.get("route", "A")
        variant = "wrong" if unit("lore-inj", it["id"]) < 0.5 else "right"
        for budget in BUDGETS:
            if variant == "right":
                value = it["gold_display"] if it["answer_kind"] != "year" else it["answer_forms"][0]
            else:
                cands = wrong_candidates(it, budget, passages, ents, english, attr_values)
                if not cands:
                    missing[f"{route}/{it['answer_kind']}/{budget}"] += 1
                    continue
                value = random.Random(int(h("lore-inj-pick", it["id"], budget)[:16], 16)).choice(cands)
            rows.append({"id": f"{it['id']}-inj-{budget}", "base_id": it["id"], "route": route, "budget": budget,
                         "variant": variant, "a0_value": value, "a0": A0_TEMPLATE.format(value=value),
                         "split": it["split"], "order": h("lore-order", it["id"], "inj")[:16]})
    with (OUT / "items_inj.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "builder": "src/data/lore_inj.py", "builder_sha256": sha256_file(__file__), "a0_template": A0_TEMPLATE,
        "items_a_sha256": sha256_file(OUT / "items_a.jsonl"),
        "items_b_sha256": sha256_file(OUT / "items_b.jsonl") if items_b else None,
        "items_inj_sha256": sha256_file(OUT / "items_inj.jsonl"),
        "counts": {"rows": len(rows),
                   "by_budget_variant": dict(sorted(Counter(f"{r['budget']}/{r['variant']}" for r in rows).items())),
                   "by_route_split": dict(sorted(Counter(f"{r['route']}/{r['split']}" for r in rows
                                                         if r["budget"] == BUDGETS[0]).items())),
                   "wrong rows without a candidate": dict(sorted(missing.items()))},
    }
    (OUT / "build_manifest_inj.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                                encoding="utf-8", newline="\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="D:/fandom/worldbuilding")
    print(json.dumps(build(Path(ap.parse_args().corpus))["counts"], indent=1))


if __name__ == "__main__":
    main()
