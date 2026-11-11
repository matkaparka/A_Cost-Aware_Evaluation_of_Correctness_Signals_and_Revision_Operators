"""Lore QA, track B (docs/planning/lore_qa/proposal.md §3.2): questions drafted by Claude from single passages, then
verified mechanically. The drafts are produced once and frozen (data/processed/lore/b_drafts.jsonl); this module never
regenerates them and never filters on any evaluated model's output.

  prepare  passages -> data/raw/lore_b/batches/batch_NN.jsonl (drafting input; rebuildable, gitignored)
  collect  data/raw/lore_b/drafts/batch_NN.jsonl -> data/processed/lore/b_drafts.jsonl (frozen)
  build    b_drafts.jsonl -> data/processed/lore/items_b.jsonl, build_manifest_b.json

Verification rules (each rejection is counted by reason in the manifest): the answer is copied from its passage, has at
most MAX_ANSWER_WORDS words and is not in the question (nor is any capitalized word of it); the question is a plain
wh-question (ends with "?", does not open like a yes/no question, never refers to "the passage"); it shares no run of
more than MAX_SHARED_RUN consecutive words with the passage; no duplicate of another draft or of a track A item; no
mentioned entity whose card carries a conflict-register tag (the track A standard tier).
Usage: python -m src.data.lore_b {prepare|collect|build} [--corpus D:/fandom/worldbuilding]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.data import lore
from src.data.lore import BUDGETS, OUT, ROOT, YEAR, build_context, h, name_pattern, norm_text, sha256_file, unit

RAW = ROOT / "data" / "raw" / "lore_b"
INSTRUCTIONS = ROOT / "docs" / "planning" / "lore_qa" / "track_b_instructions.md"
DRAFTER = "Claude Opus 5.5 subagents (Claude Code Agent tool, model=opus), one per batch, 2026-11-06"
N_BATCHES = 8
MIN_WORDS = 25
EXCLUDED_DOCS = {"events/The Grand Concordat Jubilee.txt"}   # the old Jubilee draft (conflict register D-01)
MAX_ANSWER_WORDS = 6
MAX_SHARED_RUN = 6                     # the instructions say 5; one word of slack for punctuation and hyphenation
YESNO = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "will", "would",
         "should", "shall", "may", "might", "must"}
BANNED = ("this passage", "the passage", "the text", "according to", "the author", "mentioned above", "the above")
NUMBER = re.compile(r"^\d[\d,]*(?:\.\d+)?$")


def pool(passages):
    return [p for p in passages if p["doc"] not in EXCLUDED_DOCS and not p["text"].startswith("#")
            and len(p["text"].split()) >= MIN_WORDS and p["tokens"] <= max(BUDGETS)]


def prepare():
    passages, _ = lore.load_built()
    ps = pool(passages)
    (RAW / "batches").mkdir(parents=True, exist_ok=True)
    (RAW / "drafts").mkdir(parents=True, exist_ok=True)
    size = -(-len(ps) // N_BATCHES)
    for b in range(N_BATCHES):
        with (RAW / "batches" / f"batch_{b:02d}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for p in ps[b * size:(b + 1) * size]:
                fh.write(json.dumps({"pid": p["pid"], "doc": p["doc"], "text": p["text"]}, ensure_ascii=False) + "\n")
    print(f"{len(ps)} passages in {N_BATCHES} batches of <= {size}")


def collect():
    rows, bad = [], Counter()
    for b in range(N_BATCHES):
        src = RAW / "drafts" / f"batch_{b:02d}.jsonl"
        if not src.exists():
            raise SystemExit(f"missing {src}")
        for i, line in enumerate(src.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                rows.append({"batch": b, "line": i, "pid": int(d["pid"]), "question": str(d["question"]),
                             "answer": str(d["answer"]), "answer_type": str(d.get("answer_type", ""))})
            except (ValueError, KeyError, TypeError):
                bad[f"batch_{b:02d}"] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "b_drafts.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"{len(rows)} drafts collected; malformed lines: {dict(bad)}")


# ---------------------------------------------------------------- verification

def _norm(s):
    return re.sub(r"\s+", " ", norm_text(s)).casefold()


def _words(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", _norm(s))


def longest_shared_run(a, b):
    """Longest run of consecutive words common to a and b."""
    wa, wb = _words(a), _words(b)
    best, prev = 0, [0] * (len(wb) + 1)
    for x in wa:
        cur = [0] * (len(wb) + 1)
        for j, y in enumerate(wb, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def answer_in_text(answer, text):
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(_norm(answer)) + r"(?![A-Za-z0-9])", _norm(text)) is not None


def check(d, passage):
    """The first failed rule for one draft, or None."""
    q, a = norm_text(d["question"]), norm_text(d["answer"])
    if passage is None:
        return "unknown pid"
    if not a or len(a.split()) > MAX_ANSWER_WORDS:
        return "answer empty or too long"
    if not answer_in_text(a, passage["text"]):
        return "answer not copied from the passage"
    if not q.endswith("?") or not 4 <= len(q.split()) <= 40:
        return "not a question of 4-40 words"
    if _words(q)[:1] and _words(q)[0] in YESNO:
        return "yes/no question"
    if any(b in _norm(q) for b in BANNED):
        return "refers to the passage"
    if answer_in_text(a, q) or any(len(w) >= 4 and w[0].isupper() and name_pattern(w).search(q)
                                   for w in re.findall(r"[A-Za-z][A-Za-z'-]*", a)):
        return "answer in the question"
    if longest_shared_run(q, passage["text"]) > MAX_SHARED_RUN:
        return "copies the passage"
    return None


def kind_and_forms(answer, english, ents):
    """number / year / relation (a canon English-named entity; its forms are accepted) / string."""
    a = norm_text(answer)
    if NUMBER.match(a):
        return "number", [a.replace(",", "")] + ([a] if "," in a else []), None
    m = YEAR.match(a)
    if m:
        return "year", [f"{m.group(1)}E {m.group(2)}", m.group(2)], None
    hits = sorted(eid for eid, forms in english.items()
                  if ents[eid].get("status") == "canon" and any(f.casefold() == a.casefold() for f in forms))
    if len(hits) == 1:
        e = ents[hits[0]]
        return "relation", sorted(set(english[hits[0]]) | {lore.display_name(e), a}), hits[0]
    return "string", [a], None


def build(corpus):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(lore.TOKENIZER))
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)   # noqa: E731
    passages, items_a = lore.load_built()
    ents, _ = lore.load_entities(corpus)
    english = {eid: forms for eid, e in ents.items() if (forms := lore.surface_forms(e))}
    reg = norm_text((corpus / lore.REGISTER).read_text(encoding="utf-8"))
    tagged = lore.conflict_tagged(ents)
    in_register = {eid for eid, forms in english.items() if any(len(f) >= 4 and name_pattern(f).search(reg) for f in forms)}
    ent_pats = {eid: [name_pattern(f) for f in forms if len(f) >= 4] for eid, forms in english.items()
                if ents[eid].get("status") == "canon"}
    pool_ids = {p["pid"] for p in pool(passages)}
    drafts = [json.loads(l) for l in (OUT / "b_drafts.jsonl").read_text(encoding="utf-8").splitlines()]

    a_answers = defaultdict(set)                          # final evidence pid -> normalized track A answers
    a_split = defaultdict(set)
    for it in items_a:
        pid = it["hops"][-1]["evidence_pid"]
        a_answers[pid].update(_norm(f) for f in it["answer_forms"])
        for hop in it["hops"]:
            a_split[hop["evidence_pid"]].add((it["split"], it["cluster"]))

    doc_pids = defaultdict(list)
    for p in passages:
        doc_pids[p["doc"]].append(p["pid"])
    rejected, seen_qa, seen_q, items = Counter(), set(), set(), []
    for d in drafts:
        passage = passages[d["pid"]] if d["pid"] in pool_ids else None
        why = check(d, passage)
        q, a = norm_text(d["question"]), norm_text(d["answer"])
        if why is None and ((d["pid"], _norm(a)) in seen_qa or _norm(q) in seen_q):
            why = "duplicate draft"
        if why is None and _norm(a) in a_answers.get(d["pid"], set()):
            why = "duplicate of a track A item"
        mentioned = {eid for eid, pats in ent_pats.items() if any(p.search(q) or p.search(a) for p in pats)}
        if why is None and mentioned & tagged:
            why = "mentions a conflict-tagged entity"
        if why:
            rejected[why] += 1
            continue
        seen_qa.add((d["pid"], _norm(a)))
        seen_q.add(_norm(q))
        kind, forms, entity = kind_and_forms(a, english, ents)
        inherited = a_split.get(d["pid"], set())
        if len({s for s, _ in inherited}) == 1:
            split, cluster = next(iter(inherited))
        else:
            cluster = f"passage:{passage['doc']}:{passage['start']}"
            split = "dev" if unit("lore-split", cluster) < lore.DEV_FRACTION else "test"
        anchors = sorted({a} | {f for eid in mentioned for f in english[eid]})
        it = {"id": "lore-b-" + h("lore-b-v1", _norm(q), _norm(a), d["pid"])[:12], "route": "B", "template": "passage",
              "cluster": cluster, "split": split, "question": q, "gold": a, "gold_display": a, "entity": entity,
              "answer_kind": kind, "answer_class": lore.answer_class(kind, a), "answer_forms": forms,
              "hops": [{"evidence_pid": d["pid"], "anchors": anchors}],
              "strict": not (mentioned & in_register), "draft": {"batch": d["batch"], "line": d["line"],
                                                                 "answer_type": d["answer_type"]}}
        it["order"] = h("lore-order", it["id"])[:16]
        it["contexts"] = {str(b): build_context(it, passages, doc_pids, b, ntok) for b in BUDGETS}
        items.append(it)
    items.sort(key=lambda it: it["id"])
    with (OUT / "items_b.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "builder": "src/data/lore_b.py", "builder_sha256": sha256_file(__file__),
        "instructions_sha256": sha256_file(INSTRUCTIONS), "drafter": DRAFTER,
        "b_drafts_sha256": sha256_file(OUT / "b_drafts.jsonl"), "passages_sha256": sha256_file(OUT / "passages.jsonl"),
        "items_a_sha256": sha256_file(OUT / "items_a.jsonl"), "world_git": lore.git_state(corpus / "_world"),
        "items_b_sha256": sha256_file(OUT / "items_b.jsonl"),
        "counts": {
            "pool passages": len(pool_ids), "drafts": len(drafts), "items": len(items),
            "rejected": dict(sorted(rejected.items())),
            "by_split": dict(Counter(it["split"] for it in items)),
            "by_kind": dict(Counter(it["answer_kind"] for it in items)),
            "by_class_split": dict(sorted(Counter(f"{it['answer_class']}/{it['split']}" for it in items).items())),
            "strict": sum(it["strict"] for it in items),
            "passages with an item": len({it["hops"][0]["evidence_pid"] for it in items}),
            "by_doc": dict(Counter(passages[it["hops"][0]["evidence_pid"]]["doc"] for it in items).most_common()),
            "over_budget": {str(b): sum(it["contexts"][str(b)]["over_budget"] for it in items) for b in BUDGETS},
        },
    }
    (OUT / "build_manifest_b.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    return manifest


def load_items_b():
    return [json.loads(l) for l in (OUT / "items_b.jsonl").read_text(encoding="utf-8").splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "collect", "build"])
    ap.add_argument("--corpus", default="D:/fandom/worldbuilding")
    args = ap.parse_args()
    if args.step == "prepare":
        prepare()
    elif args.step == "collect":
        collect()
    else:
        print(json.dumps(build(Path(args.corpus))["counts"], indent=1))


if __name__ == "__main__":
    main()
