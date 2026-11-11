"""Lore QA gold audit, G1 (docs/planning/lore_qa/proposal.md §7): a seeded simple random sample of AUDIT_N items from
each route (track A, track B), shown to Liam with question, gold, accepted answer forms and evidence passages only, never
any model output. Both splits are eligible, since no model output is involved.

  sheet   -> docs/planning/lore_qa/audit/audit_sheet.md (Liam writes 对 / 错 / 歧义 after each "判定：")
  tally   audit_sheet.md -> docs/planning/lore_qa/audit/audit_result.json (G1: pass when wrong + ambiguous <= MAX_BAD)

Simple random sampling within each route, so each route's share of wrong or ambiguous items estimates that route's rate
without weights.
Usage: python -m src.data.lore_audit {sheet|tally}
"""
import argparse
import json
import re
from collections import Counter

from src.data import lore, lore_b
from src.data.lore import ROOT, h, segment_text, sha256_file

AUDIT_DIR = ROOT / "docs" / "planning" / "lore_qa" / "audit"
SHEET = AUDIT_DIR / "audit_sheet.md"
AUDIT_N, MAX_BAD = 30, 3
VERDICTS = {"对": "ok", "错": "wrong", "歧义": "ambiguous"}


def sample():
    passages, items_a = lore.load_built()
    items_b = lore_b.load_items_b()
    pick = []
    for route, items in (("A", items_a), ("B", items_b)):
        ranked = sorted(items, key=lambda it: h("lore-audit-v1", it["id"]))
        pick += [(route, it) for it in ranked[:AUDIT_N]]
    order = sorted(pick, key=lambda r: h("lore-audit-order-v1", r[1]["id"]))   # routes interleaved, not grouped
    return passages, order


def sheet():
    passages, order = sample()
    L = ["# Lore QA 金答案抽查（G1）", "",
         "生成：`python -m src.data.lore_audit sheet`。题目按 seed 从 A、B 两条路线各随机抽 "
         f"{AUDIT_N} 题，混在一起排列。这里只有问题、金答案和证据段落，没有任何模型输出。", "",
         "**每题请在「判定：」后面写一个词：**", "",
         "- **对**：只看给出的证据段落，金答案是这个问题唯一正确的答案；",
         "- **错**：金答案不对，或者证据段落其实没有说出这个答案；",
         "- **歧义**：问题有不止一个合理答案、问法含糊，或者与设定的其他地方矛盾。", "",
         "需要说明的，写在「备注：」后面（中英文都行）。「可接受的答案写法」是评分时也算对的写法，如果漏了常见写法，也请备注。",
         f"标准：错和歧义合计不超过 {MAX_BAD} 题即通过（proposal §7 G1）。", "",
         "填完后告诉我，我用 `python -m src.data.lore_audit tally` 统计。", ""]
    for n, (route, it) in enumerate(order, start=1):
        L += ["---", "", f"## {n}. `{it['id']}`", "",
              f"**问题：** {it['question']}", "",
              f"**金答案：** {it['gold_display']}", "",
              f"**可接受的答案写法：** {' / '.join(it['answer_forms'])}", ""]
        for k, hop in enumerate(it["hops"], start=1):
            p = passages[hop["evidence_pid"]]
            label = f"证据 {k}" if len(it["hops"]) > 1 else "证据"
            L += [f"**{label}**（`{p['doc']}` L{p['start']}–L{p['end']}）：", "", "> " + p["text"], ""]
        L += ["判定：", "", "备注：", ""]
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    SHEET.write_text("\n".join(L), encoding="utf-8", newline="\n")
    key = {it["id"]: route for route, it in order}
    (AUDIT_DIR / "audit_key.json").write_text(json.dumps({"routes": key, "sheet_sha256": sha256_file(SHEET),
                                                          "items_a_sha256": sha256_file(lore.OUT / "items_a.jsonl"),
                                                          "items_b_sha256": sha256_file(lore.OUT / "items_b.jsonl")},
                                                         ensure_ascii=False, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    print(f"{len(order)} items -> {SHEET}")


def tally():
    key = json.loads((AUDIT_DIR / "audit_key.json").read_text(encoding="utf-8"))["routes"]
    text = SHEET.read_text(encoding="utf-8")
    blocks = re.split(r"\n## \d+\. `", text)[1:]
    rows, missing = [], []
    for b in blocks:
        iid = b.split("`", 1)[0]
        m = re.search(r"判定：[ \t]*([^\n]*)", b)
        note = re.search(r"备注：[ \t]*([^\n]*(?:\n(?!---)[^\n]*)*)", b)
        v = VERDICTS.get((m.group(1) if m else "").strip().strip("*").strip())
        if v is None:
            missing.append(iid)
        rows.append({"id": iid, "route": key.get(iid), "verdict": v, "note": (note.group(1).strip() if note else "")})
    by_route = {r: dict(Counter(x["verdict"] for x in rows if x["route"] == r)) for r in ("A", "B")}
    bad = sum(1 for x in rows if x["verdict"] in ("wrong", "ambiguous"))
    out = {"n": len(rows), "missing": missing, "by_route": by_route, "wrong_or_ambiguous": bad, "max_bad": MAX_BAD,
           "pass": not missing and bad <= MAX_BAD, "rows": rows}
    (AUDIT_DIR / "audit_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sheet", "tally"])
    {"sheet": sheet, "tally": tally}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
