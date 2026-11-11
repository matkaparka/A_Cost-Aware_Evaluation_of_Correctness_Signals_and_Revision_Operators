"""Lore QA, track A (docs/planning/lore_qa/proposal.md §3.1, §3.4, §5): in-context questions over Liam's fictional-world
corpus, with gold answers taken from the structured fields of the entity cards.

The corpus is read in place (never copied into the repository): entity cards (_world/entities, git-tracked there) and
the English Dragontooth source texts. Each fact (card, attribute, value) passes the filter funnel of proposal §3.1;
facts at the standard tier become forward, reverse and two-hop questions through the hand-written templates in ATTRS.
Contexts (evidence passage(s) plus same-document distractors, shuffled) are fixed at build time for every budget in
BUDGETS, so a rendered prompt never depends on the gold at run time.

Outputs (data/processed/lore/): passages.jsonl, items_a.jsonl, build_manifest_a.json.
Usage: python -m src.data.lore [--corpus D:/fandom/worldbuilding]
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "lore"
TOKENIZER = ROOT / "data" / "tokenizers" / "qwen3-4b" / "tokenizer.json"
DT_DIR = Path("Siawadria") / "the west" / "dragontooth"
REGISTER = Path("_review") / "冲突登记表.md"
DEV_FRACTION = 0.20
BUDGETS = (512, 1024, 2048)            # context token budgets; one is chosen on dev (proposal §7, G2)
WINDOW_SENTENCES = 1                   # sentences kept on each side of an anchor sentence in a long evidence passage
MAX_DISTRACTOR_SCAN = 40
MIN_DISTRACTOR_WORDS = 8               # headings and fragments are not used as distractors
PASSAGE_OVERHEAD_TOKENS = 4            # "[n] " and the blank line between passages
META_ATTRS = {"docs", "registry_id", "timeline_id", "place_text"}
ENTITY_ID = re.compile(r"^(place|person|polity|org|concept|event|route|group)\.[a-z0-9][a-z0-9.-]*$")
LREF = re.compile(r"L(\d+)(?:\s*[–-]\s*L?(\d+))?")
YEAR = re.compile(r"^(?:(?:around|about|approximately|circa|c\.|sometime in(?: the)?)\s*)?([123])E\s*~?\s*(\d{1,4})s?$",
                  re.I)
MAX_STRING_WORDS = 6
DEGENERATE_SHARE, DEGENERATE_MIN_N = 0.5, 5
TIER_LOOSE, TIER_STANDARD, TIER_STRICT = 7, 10, 11

# Question wording per attribute: q = forward question about {s}; np = noun phrase for {s}, used as the first hop of a
# two-hop question; rq = reverse question about the value {o} (asked only when the answer is unique); types = the card
# types the subject may have (default: any). Attributes not listed here are not asked (unclear semantics, units, or
# answers too long to match exactly).
ATTRS = {
    "age": {"q": "How old is {s}?"},
    "role": {"q": "What is the role of {s}?"},
    "title": {"q": "What title does {s} hold?"},
    "nee": {"q": "What was the maiden name of {s}?"},
    "species": {"q": "What species is {s}?"},
    "urban_form": {"q": "What is the shape of the layout of {s}?"},
    "ceiling_m": {"q": "How high, in metres, is the ceiling of {s}?"},
    "bridges": {"q": "How many bridges does {s} have?"},
    "inflow_count": {"q": "How many streams flow into {s}?"},
    "designed_by": {"q": "Who designed {s}?"},
    "floruit": {"q": "Around which year was {s} active?"},
    "arrived": {"q": "Around which year did {s} arrive?"},
    "departed": {"q": "Around which year did {s} depart?"},
    "disappeared": {"q": "Around which year did {s} disappear?"},
    "pledged_allegiance": {"q": "Around which year did {s} pledge allegiance?"},
    "head": {"q": "Who is the head of {s}?", "np": "the head of {s}", "rq": "Which house is {o} the head of?"},
    "heir": {"q": "Who is the heir of {s}?", "np": "the heir of {s}", "rq": "Which house is {o} the heir of?"},
    "seat": {"q": "What is the seat of {s}?", "np": "the seat of {s}", "rq": "Which house has its seat at {o}?"},
    "founder": {"q": "Who founded {s}?", "np": "the founder of {s}", "rq": "Which house did {o} found?"},
    "house": {"q": "Which house does {s} belong to?", "np": "the house of {s}"},
    # no reverse question: everyone born into a house qualifies, not only the card that records birth_house
    "birth_house": {"q": "Which house was {s} born into?", "np": "the house that {s} was born into"},
    "parent_body": {"q": "Which body is {s} part of?", "np": "the body that {s} is part of"},
    "org": {"q": "Which organization does {s} belong to?", "np": "the organization that {s} belongs to"},
    "leader": {"q": "Who leads {s}?", "np": "the leader of {s}", "rq": "Which organization does {o} lead?"},
    "branch_head": {"q": "Who heads {s}?", "np": "the head of {s}"},
    "clutch_brother": {"q": "Who is the clutch brother of {s}?", "np": "the clutch brother of {s}",
                       "rq": "Whose clutch brother is {o}?"},
    "tutor": {"q": "Who is the tutor of {s}?", "np": "the tutor of {s}"},
    "alma_mater": {"q": "Where was {s} educated?", "np": "the institution where {s} was educated"},
    "cult": {"q": "Which cult is associated with {s}?", "np": "the cult associated with {s}"},
    "cult_of": {"q": "To whom is {s} devoted?", "np": "the figure to whom {s} is devoted"},
    "currency": {"q": "What currency does {s} use?", "np": "the currency of {s}", "types": {"polity"}},
    "minted_at": {"q": "Where is {s} minted?", "np": "the place where {s} is minted",
                  "rq": "Which currency is minted at {o}?"},
    "mint_oversight": {"q": "Which body oversees the minting of {s}?", "np": "the body that oversees the minting of {s}",
                       "rq": "The minting of which currency does {o} oversee?"},
    "political_capital": {"q": "What is the political capital of {s}?", "np": "the political capital of {s}",
                          "rq": "Which realm has {o} as its political capital?", "types": {"polity"}},
    "economic_capital": {"q": "What is the economic capital of {s}?", "np": "the economic capital of {s}",
                         "rq": "Which realm has {o} as its economic capital?", "types": {"polity"}},
    "symbolic_capital": {"q": "What is the symbolic capital of {s}?", "np": "the symbolic capital of {s}",
                         "rq": "Which realm has {o} as its symbolic capital?", "types": {"polity"}},
    "institutional_capital": {"q": "What is the institutional capital of {s}?", "np": "the institutional capital of {s}",
                              "rq": "Which realm has {o} as its institutional capital?", "types": {"polity"}},
    "parliament": {"q": "Which body is the parliament of {s}?", "np": "the parliament of {s}",
                   "rq": "Which realm has {o} as its parliament?"},
    "highest_court": {"q": "What is the highest court of {s}?", "np": "the highest court of {s}",
                      "rq": "Which realm has {o} as its highest court?"},
    "dragon_lord": {"q": "Who is the Dragon Lord of {s}?", "np": "the Dragon Lord of {s}"},
}
Q_TEMPLATE = "Read the passages and answer the question.\n\nPassages:\n{passages}\n\nQuestion: {question}"


def h(*parts):
    """Seeded hash used for every derived random choice (as in src/data/build.py)."""
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def unit(*parts):
    return int(h(*parts)[:15], 16) / 16 ** 15


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm_text(s):
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'")).strip()


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def name_pattern(form):
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(form) + r"(?![A-Za-z0-9])")


def number_pattern(form):
    return re.compile(r"(?<![\d.,])" + re.escape(form) + r"(?![\d]|[.,]\d)")


def parse_lrefs(s):
    lines = set()
    for a, b in LREF.findall(s):
        a, b = int(a), int(b or a)
        lines.update(range(min(a, b), max(a, b) + 1))
    return lines


# ---------------------------------------------------------------- corpus

def load_docs(corpus):
    """Source texts split into passages: maximal runs of non-empty lines (the nobles file is hard-wrapped)."""
    docs, passages = {}, []
    files = sorted((corpus / DT_DIR).glob("*.txt")) + sorted((corpus / DT_DIR / "events").glob("*.txt"))
    for p in files:
        key = str(p.relative_to(corpus / DT_DIR)).replace("\\", "/")
        lines = p.read_text(encoding="utf-8-sig").splitlines()
        line_to_pid, block = {}, []
        for i, line in enumerate(lines + [""], start=1):
            if line.strip():
                block.append(i)
                continue
            if block:
                pid = len(passages)
                passages.append({"pid": pid, "doc": key, "start": block[0], "end": block[-1],
                                 "text": norm_text(" ".join(lines[j - 1] for j in block))})
                line_to_pid.update({j: pid for j in block})
                block = []
        docs[key] = {"path": p, "line_to_pid": line_to_pid, "main": "/" not in key}
    return docs, passages


def label_to_doc(label, docs):
    """Body citations use short labels ("Geography", "Southern Nobles"): the doc whose title words contain them."""
    want = set(re.findall(r"[a-z]+", label.lower()))
    hits = [k for k in docs if want <= set(re.findall(r"[a-z]+", Path(k).stem.lower()))]
    main = [k for k in hits if docs[k]["main"]]
    hits = main or hits
    return hits[0] if len(hits) == 1 else None


def load_entities(corpus):
    ents, failures = {}, []
    for p in sorted((corpus / "_world" / "entities").rglob("*.md")):
        text = p.read_text(encoding="utf-8")
        parts = text.split("\n---", 1) if text.startswith("---") else None
        try:
            fm = yaml.safe_load(parts[0][3:]) if parts else None
        except yaml.YAMLError:
            fm = None
        if not isinstance(fm, dict) or "id" not in fm:
            failures.append(str(p.relative_to(corpus)))
            continue
        fm["_body"] = parts[1] if len(parts) > 1 else ""
        ents[fm["id"]] = fm
    return ents, failures


def surface_forms(e):
    """English surface forms of an entity: name, aliases, and the bare name behind 'House '/'The '. Forms without a
    capital letter are left out: they are case variants (matching is case-insensitive where it matters) or common
    nouns ('parliament') that would match passages which never name the entity."""
    forms = set()
    for f in [e.get("name") or ""] + list(e.get("aliases") or []):
        f = norm_text(str(f))
        if not f or not is_ascii(f):
            continue
        forms.add(f)
        for prefix in ("House ", "The ", "the "):
            if f.startswith(prefix) and len(f) - len(prefix) >= 4:
                forms.add(f[len(prefix):])
    return sorted(f for f in forms if len(f) >= 3 and any(c.isupper() for c in f))


def display_name(e):
    """How a question names an entity mid-sentence: 'House X' for houses; 'the X' for organizations, concepts and
    realms, and wherever the card's own name or aliases carry the article; bare names for people and places."""
    name = norm_text(str(e.get("name") or ""))
    aliases = [norm_text(str(a)) for a in (e.get("aliases") or [])]
    house = [a for a in aliases if a.startswith("House ")]
    if e.get("type") == "polity" and house:
        return house[0]
    if name.startswith("The "):
        return "the " + name[4:]
    if e.get("type") in ("org", "concept", "polity") or f"The {name}" in aliases:
        return "the " + name
    return name


def cited_passages(e, docs):
    """Passages the card itself cites: its docs field (file + line refs) and the (Label Lnn) citations in its body."""
    pids, unmapped = set(), Counter()
    refs = (e.get("attrs") or {}).get("docs") or []
    for r in [refs] if isinstance(refs, str) else refs:
        m = re.match(r"^dragontooth-(.+?\.txt)\s+(L.*)$", str(r))
        if not m:
            continue
        key = "dragontooth-" + m.group(1)
        if key in docs:
            pids.update(docs[key]["line_to_pid"].get(n) for n in parse_lrefs(m.group(2)))
    for paren in re.findall(r"\(([^()]*)\)", e["_body"]):
        for part in paren.split(";"):
            m = re.match(r"^\s*([A-Z][A-Za-z ,'&-]*?)\s+(L\d+.*)$", part.strip())
            if not m:
                continue
            key = label_to_doc(m.group(1), docs)
            if key is None:
                unmapped[m.group(1)] += 1
                continue
            pids.update(docs[key]["line_to_pid"].get(n) for n in parse_lrefs(m.group(2)))
    pids.discard(None)
    return pids, unmapped


# ---------------------------------------------------------------- facts

def unwrap(v):
    if isinstance(v, dict) and "value" in v:
        return v["value"], True
    return v, False


def value_forms(v, ents, english):
    """(kind, answer forms, patterns, reason) for one attribute value; kind None means unusable in English."""
    if isinstance(v, bool) or v is None:
        return None, [], [], "other type"
    if isinstance(v, (int, float)):
        forms = sorted({str(v), f"{v:,}"})
        return "number", forms, [number_pattern(f) for f in forms], None
    if not isinstance(v, str):
        return None, [], [], "other type"
    if ENTITY_ID.match(v):
        o = ents.get(v)
        if o is None:
            return None, [], [], "dangling id"
        if o.get("status") != "canon":
            return None, [], [], "object not canon"
        if v not in english:
            return None, [], [], "object has no English name"
        forms = sorted(set(english[v]) | {display_name(o)})
        return "relation", forms, [name_pattern(f) for f in english[v]], None
    if not is_ascii(v):
        return None, [], [], "non-English value"
    m = YEAR.match(v)
    if m:
        return "year", [f"{m.group(1)}E {m.group(2)}", m.group(2)], [number_pattern(m.group(2))], None
    if len(v.split()) > MAX_STRING_WORDS:
        return None, [], [], "string too long for exact match"
    return "string", [v], [re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", re.I)], None


def answer_class(kind, value):
    """'name' for entities, numbers, years and capitalized names ('Marquis', 'Whisperleaf Elf'); 'phrase' for
    descriptive strings ('Eldest son and heir', 'a privileged relationship'), which exact matching may misjudge
    (proposal §7, G3b)."""
    if kind in ("relation", "number", "year"):
        return "name"
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", str(value))
    return "name" if words and all(w[0].isupper() for w in words if len(w) >= 4) else "phrase"


def leaks(answer_forms, question_forms):
    """The answer is readable off the question (e.g. house 'Solari' inside the name 'Aleksei Solari')."""
    q = " ".join(question_forms).casefold()
    return any(len(a) >= 3 and a.casefold() in q for a in answer_forms if not a.isdigit())


def find_evidence(subj_pats, ans_pats, passages, pids):
    return sorted(pid for pid in pids
                  if any(p.search(passages[pid]["text"]) for p in subj_pats)
                  and any(p.search(passages[pid]["text"]) for p in ans_pats))


MAP_ONLY_CONFLICTS = ("DM", "M")       # register sections about the maps (text vs map, map data): no effect on text QA
_TAG = re.compile(r"冲突登记\s*([A-Z]+)-\d+")


def conflict_tagged(ents):
    """Cards carrying a conflict-register tag that can bear on text answers. A tag whose register ID is in a map-only
    section (DM-*, M-*) does not count; a tag without a readable ID counts."""
    out = set()
    for eid, e in ents.items():
        for g in e.get("gaps") or []:
            g = str(g)
            if "冲突登记" in g and not all(p in MAP_ONLY_CONFLICTS for p in _TAG.findall(g) or ["?"]):
                out.add(eid)
    return out


def build_facts(ents, english, docs, passages, register_text):
    """Every non-meta attribute fact, with the furthest funnel stage it reached (proposal §3.1; counts.md §2)."""
    cited, unmapped = {}, Counter()
    for eid, e in ents.items():
        cited[eid], um = cited_passages(e, docs)
        unmapped.update(um)
    tagged = conflict_tagged(ents)
    reg = norm_text(register_text)
    in_register = {eid for eid, forms in english.items() if any(len(f) >= 4 and name_pattern(f).search(reg) for f in forms)}

    facts, lists, drops = [], [], Counter()
    for eid, e in ents.items():
        for attr, raw in (e.get("attrs") or {}).items():
            if attr in META_ATTRS:
                continue
            v, has_note = unwrap(raw)
            v = norm_text(v) if isinstance(v, str) else v
            f = {"subj": eid, "attr": attr, "has_note": has_note, "stage": 0}
            facts.append(f)
            if e.get("status") != "canon":
                continue
            f["stage"] = 1
            if eid not in english:
                continue
            f["stage"] = 2
            if isinstance(v, list):
                lists.append({"subj": eid, "attr": attr,
                              "english_members": sum(1 for x in v if isinstance(x, str) and x in english)})
                continue
            f["stage"] = 3
            kind, forms, pats, why = value_forms(v, ents, english)
            if kind is None:
                drops[why] += 1
                continue
            f.update(stage=4, kind=kind, value=v, answer_forms=forms, _pats=pats)
            if leaks(forms, english[eid]):
                continue
            f["stage"] = 5
    by_attr = defaultdict(Counter)
    for f in facts:
        if f["stage"] >= 5:
            by_attr[f["attr"]][json.dumps(f["value"], ensure_ascii=False)] += 1
    degenerate = {a for a, c in by_attr.items()
                  if sum(c.values()) >= DEGENERATE_MIN_N and max(c.values()) / sum(c.values()) > DEGENERATE_SHARE}
    subj_pats = {eid: [name_pattern(x) for x in forms] for eid, forms in english.items()}
    for f in facts:
        if f["stage"] < 5 or f["attr"] in degenerate:
            continue
        f["stage"] = 6
        f["evidence_any"] = find_evidence(subj_pats[f["subj"]], f["_pats"], passages, range(len(passages)))
        if not f["evidence_any"]:
            continue
        f["stage"] = TIER_LOOSE
        f["evidence_cited"] = find_evidence(subj_pats[f["subj"]], f["_pats"], passages, cited[f["subj"]])
        if not f["evidence_cited"]:
            continue
        f["stage"] = 8
        if f["has_note"]:
            continue
        f["stage"] = 9
        obj = f["value"] if f["kind"] == "relation" else None
        if f["subj"] in tagged or obj in tagged:
            continue
        f["stage"] = TIER_STANDARD
        if f["subj"] in in_register or obj in in_register:
            continue
        f["stage"] = TIER_STRICT
    return {"facts": facts, "lists": lists, "drops": drops, "degenerate": degenerate, "unmapped": unmapped,
            "tagged": tagged, "in_register": in_register, "cited": cited}


# ---------------------------------------------------------------- items

def _primary_leak(question, name):
    return bool(name) and name_pattern(name).search(question) is not None


def _answer_forms(forms, question):
    """Drop aliases the question itself contains ('Dragon Lord' in 'Who is the Dragon Lord of ...?')."""
    return [f for f in forms if not (len(f) >= 3 and not f.isdigit() and name_pattern(f).search(question))]


def make_items(facts, ents, english, passages, min_stage=TIER_STANDARD):
    """Forward, reverse (unique answers only) and two-hop questions from facts at min_stage or above."""
    usable = [f for f in facts if f["stage"] >= min_stage]
    holders = defaultdict(set)                            # (attr, value) -> canon cards holding it
    for eid, e in ents.items():
        if e.get("status") == "canon":
            for attr, raw in (e.get("attrs") or {}).items():
                v, _ = unwrap(raw)
                if isinstance(v, str):
                    holders[(attr, norm_text(v))].add(eid)

    def patterns(forms, kind):
        if kind in ("number", "year"):
            return [number_pattern(f) for f in forms if f.replace(",", "").isdigit()]
        flags = re.I if kind == "string" else 0
        return [re.compile(r"(?<![A-Za-z0-9])" + re.escape(f) + r"(?![A-Za-z0-9])", flags) for f in forms]

    def best(f, forms=None, kind=None):
        """Shortest evidence passage (cited ones first, ties by pid) that contains one of the given answer forms."""
        pats = patterns(forms, kind) if forms is not None else None
        for pool in (f["evidence_cited"], f["evidence_any"]):
            ok = [pid for pid in pool if pats is None or any(p.search(passages[pid]["text"]) for p in pats)]
            if ok:
                return min(ok, key=lambda pid: (passages[pid]["tokens"], pid))
        return None

    def anchors(*eids_or_forms):
        out = set()
        for x in eids_or_forms:
            out.update(english.get(x, [x]) if isinstance(x, str) else x)
        return sorted(out)

    items, dropped = [], Counter()

    def add(template, holder, attrs, question, gold, forms, kind, hops, stage, primary, final_fact):
        """The last hop's evidence is chosen after aliases borne by the question are removed from the answer forms,
        so that the passage still names the answer; an item whose evidence is lost this way is dropped."""
        if _primary_leak(question, primary):
            dropped[f"{template}: answer name in question"] += 1
            return
        forms = _answer_forms(forms, question)
        if not forms:
            dropped[f"{template}: every answer form in question"] += 1
            return
        pid = best(final_fact, forms, kind)
        if pid is None:
            dropped[f"{template}: no evidence names a remaining answer form"] += 1
            return
        hops[-1] = {"evidence_pid": pid, "anchors": hops[-1]["anchors"]}
        shown = ents[gold].get("name") if kind == "relation" else str(gold)
        items.append({"template": template, "cluster": holder, "attrs": attrs, "question": question, "gold": gold,
                      "gold_display": shown, "answer_kind": kind, "answer_class": answer_class(kind, shown),
                      "answer_forms": forms, "hops": hops, "stage": stage})

    for f in usable:
        t = ATTRS.get(f["attr"])
        if t is None:
            dropped[f"no template: {f['attr']}"] += 1
            continue
        if "types" in t and ents[f["subj"]].get("type") not in t["types"]:
            dropped[f"subject type: {f['attr']}"] += 1
            continue
        s = display_name(ents[f["subj"]])
        primary = ents[f["value"]].get("name") if f["kind"] == "relation" else (f["value"] if f["kind"] == "string" else None)
        hop = {"evidence_pid": best(f), "anchors": anchors(f["subj"], f["answer_forms"])}
        add("forward", f["subj"], [f["attr"]], t["q"].format(s=s), f["value"], f["answer_forms"], f["kind"], [dict(hop)],
            f["stage"], primary, f)
        if "rq" in t and f["kind"] == "relation" and len(holders[(f["attr"], f["value"])]) == 1:
            subj_forms = sorted(set(english[f["subj"]]) | {display_name(ents[f["subj"]])})
            add("reverse", f["subj"], [f["attr"]], t["rq"].format(o=display_name(ents[f["value"]])), f["subj"],
                subj_forms, "relation", [dict(hop)], f["stage"], ents[f["subj"]].get("name"), f)

    by_subj = defaultdict(list)
    for f in usable:
        by_subj[f["subj"]].append(f)
    for f1 in usable:
        t1 = ATTRS.get(f1["attr"])
        if f1["kind"] != "relation" or t1 is None or "np" not in t1 or \
                ("types" in t1 and ents[f1["subj"]].get("type") not in t1["types"]):
            continue
        for f2 in by_subj.get(f1["value"], []):
            t2 = ATTRS.get(f2["attr"])
            if t2 is None or (f2["kind"] == "relation" and f2["value"] == f1["subj"]) or \
                    ("types" in t2 and ents[f2["subj"]].get("type") not in t2["types"]):
                continue
            inner = t1["np"].format(s=display_name(ents[f1["subj"]]))
            primary = ents[f2["value"]].get("name") if f2["kind"] == "relation" else (f2["value"] if f2["kind"] == "string" else None)
            hops = [{"evidence_pid": best(f1), "anchors": anchors(f1["subj"], f1["value"])},
                    {"evidence_pid": best(f2), "anchors": anchors(f2["subj"], f2["answer_forms"])}]
            add("two_hop", f1["subj"], [f1["attr"], f2["attr"]], t2["q"].format(s=inner), f2["value"], f2["answer_forms"],
                f2["kind"], hops, min(f1["stage"], f2["stage"]), primary, f2)

    seen, out = set(), []
    for it in items:
        it["id"] = "lore-a-" + h("lore-a-v1", it["template"], norm_text(it["question"]).casefold(),
                                 json.dumps(it["gold"]))[:12]
        if it["id"] in seen:
            dropped["duplicate question"] += 1
            continue
        seen.add(it["id"])
        it["split"] = "dev" if unit("lore-split", it["cluster"]) < DEV_FRACTION else "test"
        it["order"] = h("lore-order", it["id"])[:16]
        it["strict"] = it["stage"] >= TIER_STRICT
        it["disjoint_evidence"] = len({hop["evidence_pid"] for hop in it["hops"]}) == len(it["hops"])
        out.append(it)
    return sorted(out, key=lambda it: it["id"]), dropped


# ---------------------------------------------------------------- contexts

_SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z\"'(\[])")


def sentences(text):
    return _SENT.split(text)


def segment_text(seg, passages):
    """A context segment: a whole passage, or sentences [a, b) of it."""
    p = passages[seg["pid"]]
    if seg.get("sent") is None:
        return p["text"]
    a, b = seg["sent"]
    return " ".join(sentences(p["text"])[a:b])


def _evidence_segment(pid, anchors, passages, budget, ntok):
    p = passages[pid]
    if p["tokens"] <= budget // 2:
        return {"pid": pid, "sent": None}, p["tokens"]
    sents = sentences(p["text"])
    pats = [name_pattern(a) if not a.isdigit() else number_pattern(a) for a in anchors]
    hits = [i for i, s in enumerate(sents) if any(q.search(s) for q in pats)] or [0]
    a, b = max(0, min(hits) - WINDOW_SENTENCES), min(len(sents), max(hits) + 1 + WINDOW_SENTENCES)
    seg = {"pid": pid, "sent": [a, b]}
    return seg, ntok(segment_text(seg, passages))


def build_context(item, passages, doc_pids, budget, ntok):
    """Evidence segment(s), then the nearest same-document passages while the budget allows, shuffled by a seeded
    stream. Depends on the gold only through the build-time anchors; the result is stored with the item."""
    segs, used, ev_pids = [], 0, []
    for hop in item["hops"]:
        pid = hop["evidence_pid"]
        if pid in ev_pids:
            continue
        seg, n = _evidence_segment(pid, hop["anchors"], passages, budget, ntok)
        segs.append(seg)
        ev_pids.append(pid)
        used += n + PASSAGE_OVERHEAD_TOKENS
    over = used > budget
    for ev in ev_pids:
        doc = passages[ev]["doc"]
        cands = sorted((pid for pid in doc_pids[doc] if pid not in ev_pids), key=lambda pid: (abs(pid - ev), pid))
        cands = [pid for pid in cands if not passages[pid]["text"].startswith("#")
                 and len(passages[pid]["text"].split()) >= MIN_DISTRACTOR_WORDS]
        for pid in cands[:MAX_DISTRACTOR_SCAN]:
            n = passages[pid]["tokens"] + PASSAGE_OVERHEAD_TOKENS
            if any(s["pid"] == pid for s in segs) or used + n > budget:
                continue
            segs.append({"pid": pid, "sent": None})
            used += n
    random.Random(int(h("lore-context", item["id"], budget)[:16], 16)).shuffle(segs)
    return {"segments": segs, "tokens": used, "over_budget": over}


def render_question(item, budget, passages):
    """The q of the shared prompt wrappers (src/gen/render.py): passages plus the question. No gold enters here."""
    ctx = item["contexts"][str(budget)]
    body = "\n\n".join(f"[{i}] {segment_text(s, passages)}" for i, s in enumerate(ctx["segments"], start=1))
    return Q_TEMPLATE.format(passages=body, question=item["question"])


# ---------------------------------------------------------------- build

def git_state(path):
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--", "entities"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": head, "entities_dirty": bool(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "entities_dirty": None}


def load_corpus(corpus, tok):
    docs, passages = load_docs(corpus)
    for p in passages:
        p["tokens"] = len(tok.encode(p["text"], add_special_tokens=False).ids)
    ents, failures = load_entities(corpus)
    english = {eid: forms for eid, e in ents.items() if (forms := surface_forms(e))}
    register_text = (corpus / REGISTER).read_text(encoding="utf-8")
    return docs, passages, ents, failures, english, register_text


def build(corpus, out_dir=OUT):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)   # noqa: E731
    docs, passages, ents, failures, english, register_text = load_corpus(corpus, tok)
    fx = build_facts(ents, english, docs, passages, register_text)
    items, dropped = make_items(fx["facts"], ents, english, passages)
    doc_pids = defaultdict(list)
    for p in passages:
        doc_pids[p["doc"]].append(p["pid"])
    for it in items:
        it["contexts"] = {str(b): build_context(it, passages, doc_pids, b, ntok) for b in BUDGETS}

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "passages.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for p in passages:
            fh.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "items_a.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "builder": "src/data/lore.py", "builder_sha256": sha256_file(__file__), "corpus": str(corpus),
        "world_git": git_state(corpus / "_world"),
        "source_sha256": {k: sha256_file(d["path"]) for k, d in docs.items()},
        "register_sha256": sha256_file(corpus / REGISTER), "tokenizer_sha256": sha256_file(TOKENIZER),
        "templates_sha256": h(json.dumps(ATTRS, sort_keys=True, default=sorted), Q_TEMPLATE),
        "frontmatter_parse_failures": failures,
        "passages_sha256": sha256_file(out_dir / "passages.jsonl"), "items_a_sha256": sha256_file(out_dir / "items_a.jsonl"),
        "counts": {
            "items": len(items),
            "by_template_split": dict(sorted(Counter(f"{it['template']}/{it['split']}" for it in items).items())),
            "by_class_split": dict(sorted(Counter(f"{it['answer_class']}/{it['split']}" for it in items).items())),
            "strict": sum(it["strict"] for it in items),
            "two_hop_disjoint_evidence": sum(it["disjoint_evidence"] for it in items if it["template"] == "two_hop"),
            "clusters": dict(Counter(("dev" if unit("lore-split", c) < DEV_FRACTION else "test")
                                     for c in {it["cluster"] for it in items})),
            "over_budget": {str(b): sum(it["contexts"][str(b)]["over_budget"] for it in items) for b in BUDGETS},
            "dropped": dict(sorted(dropped.items())),
        },
    }
    (out_dir / "build_manifest_a.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8", newline="\n")
    return manifest


def load_built(out_dir=OUT):
    passages = [json.loads(l) for l in (out_dir / "passages.jsonl").read_text(encoding="utf-8").splitlines()]
    items = [json.loads(l) for l in (out_dir / "items_a.jsonl").read_text(encoding="utf-8").splitlines()]
    return passages, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="D:/fandom/worldbuilding")
    m = build(Path(ap.parse_args().corpus))
    print(json.dumps(m["counts"], indent=1))


if __name__ == "__main__":
    main()
