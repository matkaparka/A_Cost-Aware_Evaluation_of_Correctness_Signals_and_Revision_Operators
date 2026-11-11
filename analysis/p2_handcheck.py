"""P2 hand-check samples (CLAUDE.md P2.6: >= 5 examples per metric): results/p2 -> reports/02_handcheck_samples.md.

A seeded sample per metric, printed with what a reader needs to judge the label: gold, extracted answer, label, and
the end of the generated text. The judgments themselves are recorded by hand in reports/02_handcheck.md.
Exp 5: the answer-span mask of the update (which a0 tokens the loss sees). Dev split only.
Usage: python analysis/p2_handcheck.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "02_handcheck_samples.md"
SEED = 1104
PER = 5
TAIL = 500


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def tail(text):
    t = text[-TAIL:].replace("```", "'''")
    return ("…" if len(text) > TAIL else "") + t


def block(lines, title, rows, gens, gold):
    lines += [f"#### {title}", ""]
    if not rows:
        lines += ["(no rows in this category)", ""]
        return
    for k, r in enumerate(rows, 1):
        g = gens.get(r.get("prompt_key"))
        lines += [f"**{k}. {r['item_id']} · {r['cond']}** — gold `{gold(r)}`, pred `{r.get('pred') or r.get('choice')}`, "
                  f"correct0 {r['correct0']} → correct {r['correct']}, status {r['status']}", "",
                  "```", tail(g["text"]) if g else "(no generation: fixed-answer condition)", "```", ""]


def exp01(lines, model, rng):
    d = ROOT / "results" / "p2" / model / f"p2-exp01-dev-{model}-f7e5011"
    rows = jl(d / "rows.jsonl")
    gens = {g["prompt_key"]: g for g in jl(d / "generations.jsonl")}
    dnat = {r["item_id"]: r for f in ("dnat", "dinj") for r in jl(ROOT / "data" / "processed" / f"{f}.jsonl")}
    e0 = {r["item_id"]: r for r in jl(ROOT / "data" / "processed" / "exp0.jsonl")}
    gold = lambda r: e0[r["item_id"]]["gold_letter"] if r["exp"] == "0" else dnat[r["item_id"]]["gold"]
    pick = lambda rs: rng.sample(rs, min(PER, len(rs)))
    gen = [r for r in rows if r.get("prompt_key")]
    lines += [f"### Qwen3-{model}", ""]
    block(lines, "1A F0 (self-review): wrong → right", pick([r for r in gen if r["exp"] == "1A" and r["cond"] == "F0"
                                                              and not r["correct0"] and r["correct"]]), gens, gold)
    block(lines, "1A F1/F3(1.0): right → wrong", pick([r for r in gen if r["exp"] == "1A" and r["cond"] in ("F1", "F3(1.0)")
                                                       and r["correct0"] and not r["correct"]]), gens, gold)
    block(lines, "1A F3(1.0) (verifier says incorrect): still wrong", pick([r for r in gen if r["exp"] == "1A" and r["cond"] == "F3(1.0)"
                                                                           and not r["correct0"] and not r["correct"]]), gens, gold)
    block(lines, "1B user/other/F1: any row", pick([r for r in gen if r["cond"] == "user/other/F1"]), gens, gold)
    block(lines, "1A/1B truncated", pick([r for r in gen if r["exp"] in ("1A", "1B") and r["status"] == "truncated"]), gens, gold)
    block(lines, "Exp 0 F3(1.0) choices", pick([r for r in gen if r["exp"] == "0" and r["cond"] == "F3(1.0)"]), gens, gold)
    block(lines, "Exp 0 invalid choices", pick([r for r in gen if r["exp"] == "0" and r["status"] == "invalid"]), gens, gold)


def exp5_masks(lines, rng):
    from transformers import AutoTokenizer
    from src.ttt import update as U
    tok = AutoTokenizer.from_pretrained(ROOT / "data" / "tokenizers" / "qwen3-4b")
    pilot = max((ROOT / "results" / "p2" / "4B").glob("p2-ttt-pilot-4B-*"), key=lambda p: p.stat().st_mtime)
    items = sorted({a["item_id"] for a in jl(pilot / "adapters.jsonl") if a["item_id"]})
    init = {r["item_id"]: r for r in jl(ROOT / "results" / "p1" / "4B" / "p1-initial-4B-d2ebdea" / "initial.jsonl")}
    dnat = {r["item_id"]: r for r in jl(ROOT / "data" / "processed" / "dnat.jsonl")}
    lines += [f"### Exp 5 answer-span masks (Qwen3-4B, {pilot.name})", "",
              "The loss sees only the masked a0 tokens. Shown: the masked text between ⟦ ⟧ in the last 200 characters of a0.", ""]
    for k, i in enumerate(rng.sample(items, PER), 1):
        ids, mask = U.training_sequence(tok, dnat[i]["question"], init[i]["a0_token_ids"])
        if mask is None:
            lines += [f"**{k}. {i}** — no answer span (adapter left at zero)", ""]
            continue
        a0_ids, a0_mask = ids[len(ids) - len(init[i]["a0_token_ids"]):], mask[len(mask) - len(init[i]["a0_token_ids"]):]
        text = "".join(("⟦" + tok.decode([t]) + "⟧") if m else tok.decode([t]) for t, m in zip(a0_ids, a0_mask))
        lines += [f"**{k}. {i}** — gold `{dnat[i]['gold']}`, correct0 {init[i]['correct']}, masked tokens {sum(a0_mask)}", "",
                  "```", "…" + text[-200:].replace("⟧⟦", ""), "```", ""]


def text_block(lines, title, rows, texts, gold):
    lines += [f"#### {title}", ""]
    if not rows:
        lines += ["(no rows in this category)", ""]
        return
    for k, (i, r) in enumerate(rows, 1):
        lines += [f"**{k}. {r['item_id']} · {r.get('arm') or r.get('run')}"
                  f"{' · ' + str(r['grounding']) if r.get('grounding') else ''}"
                  f"{' · verdict ' + str(r['verdict']) if r.get('verdict') is not None else ''}** — gold `{gold(r)}`, "
                  f"pred `{r.get('pred')}`, correct0 {r['correct0']} → correct {r['correct']}, status {r['status']}", "",
                  "```", tail(texts[i]) if i in texts else "(no text)", "```", ""]


def exp5_texts(lines, model, sel, rng):
    """Exp 5 served generations at the selected (tau, lr): parameter arm by grounding, the context arm, random."""
    d = ROOT / "results" / "p2" / model / sel["serve"]
    if not (d / "generations.jsonl").exists():
        lines += [f"### Exp 5 texts (Qwen3-{model}, {sel['serve']})", "", "(run predates the text output)", ""]
        return
    rows = list(enumerate(jl(d / "rows.jsonl")))
    texts = {g["row"]: g["text"] for g in jl(d / "generations.jsonl")}
    dnat = {r["item_id"]: r for r in jl(ROOT / "data" / "processed" / "dnat.jsonl")}
    gold = lambda r: dnat[r["item_id"]]["gold"]
    at = lambda r: (r.get("tau") is None or f"{r['tau']:g}" == sel["tau"]) and (r.get("lr") is None or f"{r['lr']:g}" == sel["lr"])
    pick = lambda rs: rng.sample(rs, min(PER, len(rs)))
    lines += [f"### Exp 5 texts (Qwen3-{model}, {sel['serve']}, τ = {sel['tau']}, lr = {sel['lr']})", ""]
    text_block(lines, "parameter arm, self: wrong → right", pick([(i, r) for i, r in rows if r["arm"] == "parameter"
               and r.get("grounding") == "self" and at(r) and not r["correct0"] and r["correct"]]), texts, gold)
    text_block(lines, "parameter arm, self vs perm-self disagree", pick([(i, r) for i, r in rows if r["arm"] == "parameter"
               and r.get("grounding") in ("self", "perm-self") and at(r) and any(
                   r2["arm"] == "parameter" and r2.get("grounding") != r["grounding"] and r2.get("grounding") in ("self", "perm-self")
                   and r2["item_id"] == r["item_id"] and at(r2) and r2["correct"] != r["correct"] for _, r2 in rows)]), texts, gold)
    text_block(lines, "parameter + context, ext-1.0: right → wrong or wrong → wrong", pick([(i, r) for i, r in rows
               if r["arm"] == "parameter+context" and r.get("grounding") == "ext-1.0" and at(r) and not r["correct"]]), texts, gold)
    text_block(lines, "context arm (no update), verdict incorrect", pick([(i, r) for i, r in rows if r["arm"] == "context"
               and r.get("verdict") is False]), texts, gold)
    text_block(lines, "random adapter: unparsed or truncated", pick([(i, r) for i, r in rows if r.get("grounding") == "random"
               and r["status"] != "ok"]) or pick([(i, r) for i, r in rows if r.get("grounding") == "random"]), texts, gold)


def exp3_texts(lines, sel, rng):
    """Exp 3 confirmation generations: the main setting and its random controls on D-inj, R->W on D-nat."""
    d = ROOT / "results" / "p2" / "4B" / sel["run"]
    rows = list(enumerate(jl(d / "confirm_rows.jsonl")))
    texts = {g["row"]: g["text"] for g in jl(d / "confirm_texts.jsonl")}
    golds = {r["item_id"]: r["gold"] for f in ("dnat", "dinj") for r in jl(ROOT / "data" / "processed" / f"{f}.jsonl")}
    gold = lambda r: golds[r["item_id"]]
    pick = lambda rs: rng.sample(rs, min(PER, len(rs)))
    main, rand = sel["main_run"], set(sel["controls"]["random"])
    lines += [f"### Exp 3 texts (Qwen3-4B, {sel['run']}, main setting {main})", ""]
    text_block(lines, "main setting, D-inj: corrected", pick([(i, r) for i, r in rows if r["run"] == main and r["data"] == "dinj"
               and r["correct"]]), texts, gold)
    text_block(lines, "main setting, D-inj: not corrected", pick([(i, r) for i, r in rows if r["run"] == main
               and r["data"] == "dinj" and not r["correct"]]), texts, gold)
    text_block(lines, "random controls, D-inj: not corrected", pick([(i, r) for i, r in rows if r["run"] in rand
               and r["data"] == "dinj" and not r["correct"]]), texts, gold)
    text_block(lines, "unsteered, D-nat: right → wrong", pick([(i, r) for i, r in rows if r["run"] == "unsteered"
               and r["data"] == "dnat" and not r["correct"]]), texts, gold)
    text_block(lines, "any arm: unparsed or truncated", pick([(i, r) for i, r in rows if r["status"] != "ok"]), texts, gold)


def main():
    lines = ["# P2 hand-check samples (generated by analysis/p2_handcheck.py; dev split only; seed "
             f"{SEED}, {PER} per category)", "", "Judgments are in [02_handcheck.md](02_handcheck.md).", ""]
    rng = random.Random(SEED)
    for model in ("4B", "1.7B"):
        exp01(lines, model, rng)
    exp5_masks(lines, rng)
    # Runs chosen in analysis/out/tier2.json (analysis/p2_tier2.py); sampled after the sections above, so their samples
    # do not move.
    # The 4B Exp 5 texts came last (its first selected run had none), so they are sampled last; the judged 1.7B and
    # Exp 3 samples stay where they were.
    sel = json.loads((ROOT / "analysis" / "out" / "tier2.json").read_text())
    if sel["exp5"].get("1.7B"):
        exp5_texts(lines, "1.7B", sel["exp5"]["1.7B"], rng)
    if sel["exp3"].get("4B") and sel["exp3"]["4B"].get("main_run"):
        exp3_texts(lines, sel["exp3"]["4B"], rng)
    if sel["exp5"].get("4B"):
        exp5_texts(lines, "4B", sel["exp5"]["4B"], rng)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
