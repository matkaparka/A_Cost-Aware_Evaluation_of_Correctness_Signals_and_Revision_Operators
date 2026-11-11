"""Blind human audit of scored outputs (docs/planning/next_experiments/plan.md §1 item 9; review_and_suggestion §3).

Liam reads a model output's ending and the gold answer and judges whether the final answer equals the gold. He sees no
condition name, no model, no experiment, and not the scorer's verdict. A seeded sample of SHEET_N rows from the 4B test
runs, in three strata, interleaved:
- disagree: rows where the primary scorer and the second scorer v2 disagree (where scoring is most likely wrong);
- discordant: 1A rows from items whose F2 and F1 correctness differ (the rows that drive H1a), one row per item;
- steering: Exp 3 main-arm rows whose correctness differs from the unsteered arm (the rows behind H3b).
The strata are a targeted audit: the error rate within each stratum is reported as such and is not a population rate
(review_and_suggestion §3).

  sheet   -> docs/planning/next_experiments/blind_audit/audit_sheet.md (+ audit_key.json, kept apart from the sheet)
  tally   -> docs/planning/next_experiments/blind_audit/audit_result.json and reports/fu_12_blind_audit.md: agreement
             of the reviewer's verdicts with the primary scorer, per stratum (--reviewer names who filled the sheet)
Usage: python analysis/fu_blind_audit.py {sheet|tally} [--spec configs/p6_spec_4B_test.json]
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from p6_confirmatory import jl, rescore  # noqa: E402

DIR = ROOT / "docs" / "planning" / "next_experiments" / "blind_audit"
PER_STRATUM = {"disagree": 15, "discordant": 15, "steering": 10}
TAIL_CHARS = 700
VERDICTS = {"对": True, "错": False, "看不出": None}


def h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def candidates(spec):
    items = {r["item_id"]: r for f in ("data/processed/dnat.jsonl", "data/processed/ext/dnat_ext.jsonl",
                                       "data/processed/dinj.jsonl") for r in jl(f)}
    out = defaultdict(list)
    by_item = defaultdict(dict)
    for run in (spec["exp01_orig"], spec.get("exp01_ext")):
        gen = {g["prompt_key"]: g for g in jl(Path(run) / "generations.jsonl")}
        for r in jl(Path(run) / "rows.jsonl"):
            if not r.get("prompt_key") or r.get("stage") == 1 or r["exp"] not in ("1A", "1B"):
                continue
            g = gen[r["prompt_key"]]
            row = {"item_id": r["item_id"], "text": g["text"], "finish_reason": g["finish_reason"],
                   "primary": bool(r["correct"]), "source": f"{run}|{r['exp']}|{r['data']}|{r['cond']}"}
            if row["primary"] != rescore(g["text"], items[r["item_id"]]["gold"], g["finish_reason"]):
                out["disagree"].append(row)
            if r["exp"] == "1A" and r["data"] == "dnat" and r["cond"] in ("F1", "F2"):
                by_item[r["item_id"]][r["cond"]] = row
    for iid, d in by_item.items():
        if len(d) == 2 and d["F1"]["primary"] != d["F2"]["primary"]:
            out["discordant"].append(d["F1"] if int(h("fu-audit-side", iid), 16) % 2 == 0 else d["F2"])   # one side, seeded
    texts = {(t["run"], t["item_id"]): t for t in jl(spec["exp3_texts"])}
    st = {(r["run"], r["item_id"]): r for r in jl(spec["exp3_rows"])}
    for (run, iid), r in st.items():
        if run == spec["exp3_main"] and bool(r["correct"]) != bool(st[("unsteered", iid)]["correct"]):
            t = texts[(run, iid)]
            out["steering"].append({"item_id": iid, "text": t["text"], "finish_reason": t["finish_reason"],
                                    "primary": bool(r["correct"]), "source": f"{spec['exp3_rows']}|{run}"})
    return items, out


def sheet(spec):
    items, cands = candidates(spec)
    picked = []
    for stratum, n in PER_STRATUM.items():
        ranked = sorted(cands[stratum], key=lambda r: h("fu-audit-v1", stratum, r["source"], r["item_id"]))
        picked += [{**r, "stratum": stratum} for r in ranked[:n]]
    picked.sort(key=lambda r: h("fu-audit-order-v1", r["source"], r["item_id"]))
    L = ["# 输出盲审（4B test，数学题）", "",
         "生成：`python analysis/fu_blind_audit.py sheet`。每条是模型一次输出的结尾部分和这道题的标准答案。"
         "不显示条件、实验和评分器的判定。", "",
         "**请在每条的「判定：」后写一个词：**", "",
         "- **对**：输出的最终答案与标准答案等价；",
         "- **错**：不等价，或者没有给出最终答案；",
         "- **看不出**：无法判断（例如输出被截断、答案含糊）。", "",
         "需要说明的写在「备注：」后面。填完告诉我，我用 `python analysis/fu_blind_audit.py tally` 统计。", ""]
    key = {}
    for n, r in enumerate(picked, start=1):
        it = items[r["item_id"]]
        tail = r["text"][-TAIL_CHARS:]
        L += ["---", "", f"## {n}", "", "**题目：**", "", "> " + it["question"].replace("\n", "\n> "), "",
              f"**标准答案：** `{it['gold']}`", "",
              f"**输出结尾**（{'被截断，' if r['finish_reason'] == 'length' else ''}最后 {len(tail)} 个字符）：", "",
              "```", tail, "```", "", "判定：", "", "备注：", ""]
        key[str(n)] = {k: r[k] for k in ("item_id", "stratum", "primary", "source", "finish_reason")}
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "audit_sheet.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    (DIR / "audit_key.json").write_text(json.dumps({"spec": spec["_path"], "per_stratum": PER_STRATUM,
                                                    "available": {k: len(v) for k, v in cands.items()}, "rows": key},
                                                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(picked)} rows -> {DIR / 'audit_sheet.md'}; available per stratum: { {k: len(v) for k, v in cands.items()} }")


def tally(reviewer):
    """Agreement of the reviewer's verdicts with the primary scorer, per stratum; writes audit_result.json and
    reports/fu_12_blind_audit.md (every disagreement listed with the reviewer's note). `reviewer` names who filled the
    sheet; the report states it, so an AI-filled sheet is never reported as a human audit."""
    key = json.loads((DIR / "audit_key.json").read_text(encoding="utf-8"))["rows"]
    blocks = re.split(r"\n## (\d+)\n", (DIR / "audit_sheet.md").read_text(encoding="utf-8"))[1:]
    rows, missing = [], []
    for n, body in zip(blocks[::2], blocks[1::2]):
        m = re.search(r"判定：[ \t]*([^\n]*)", body)
        word = (m.group(1) if m else "").strip().strip("*").strip()
        if word not in VERDICTS:
            missing.append(n)
            continue
        note = re.search(r"备注：[ \t]*([^\n]*)", body)
        k = key[n]
        rows.append({"n": n, "stratum": k["stratum"], "human": VERDICTS[word], "primary": k["primary"],
                     "note": note.group(1).strip() if note else ""})
    per = {}
    for s in PER_STRATUM:
        rs = [r for r in rows if r["stratum"] == s and r["human"] is not None]
        per[s] = {"judged": len(rs), "agree": sum(r["human"] == r["primary"] for r in rs),
                  "undecided": sum(1 for r in rows if r["stratum"] == s and r["human"] is None)}
    out = {"reviewer": reviewer, "missing": missing, "per_stratum": per, "rows": rows}
    (DIR / "audit_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    yes = lambda v: "right" if v else ("wrong" if v is False else "undecided")  # noqa: E731
    L = ["# Blind audit of scored math outputs (generated by analysis/fu_blind_audit.py tally)", "",
         f"**Reviewer:** {reviewer}.", "",
         "The sheet showed each output's ending and the gold answer; it hid the condition, the experiment and the scorer's "
         "verdict. The strata are targeted (rows where the two scorers disagree; F2/F1-discordant rows behind H1a; steering "
         "rows behind H3b), so agreement within a stratum is not a population error rate.", "",
         "| Stratum | Judged | Agree with the primary scorer | Undecided |", "|---|---|---|---|"]
    L += [f"| {s} | {v['judged']} | {v['agree']} | {v['undecided']} |" for s, v in per.items()]
    if missing:
        L += ["", f"Entries without a valid verdict: {', '.join(missing)}."]
    L += ["", "## Disagreements with the primary scorer", "", "| # | Stratum | Reviewer | Primary scorer | Reviewer's note |", "|---|---|---|---|---|"]
    L += [f"| {r['n']} | {r['stratum']} | {yes(r['human'])} | {yes(r['primary'])} | {r['note'].replace('|', '/')} |"
          for r in rows if r["human"] is not None and r["human"] != r["primary"]]
    (ROOT / "reports" / "fu_12_blind_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sheet", "tally"])
    ap.add_argument("--spec", default="configs/p6_spec_4B_test.json")
    ap.add_argument("--reviewer", default="the human author (Lihan Xu)", help="who filled the sheet; stated in the report")
    a = ap.parse_args()
    spec = json.loads((ROOT / a.spec).read_text(encoding="utf-8"))
    spec["_path"] = a.spec
    sheet(spec) if a.step == "sheet" else tally(a.reviewer)


if __name__ == "__main__":
    main()
