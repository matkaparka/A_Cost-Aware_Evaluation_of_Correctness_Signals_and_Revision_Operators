"""B4 selection (plan §B, B4): the 20 % of items with the lowest vote margin over five samples, gold-blind, and the
final answers of those samples, for src/exp/arr_runs.py samples-critique.

Math: gold-blind candidates and complete-link groups as analysis/arr/common.py (the A1 rules); the samples are the
first five in stored order (B1 stream "h2a-consistency", or the stored Qwen3-4B run). MMLU-Pro: the stored sample
letters, a letter valid only inside the item's option range. Margin = largest group / 5; ties by SHA-256(item_id).
Never reads a gold answer. Runs on the run host (set ARR_OUT under RESULTS_ROOT) or on the control host.
Usage: python -B analysis/arr/b4_select.py --task math --samples <samples.jsonl> --out <select.jsonl> [--share 0.2]
       python -B analysis/arr/b4_select.py --task mc --consistency <consistency.jsonl> --out <select.jsonl>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, Equivalence, candidate, groups_of, rows, sha, tie_key, winner  # noqa: E402

K = 5


def math_choices(paths, eq):
    """paths: one samples.jsonl or several (the chunks of one run); every item exactly once."""
    out = {}
    for r in (r for p in ([paths] if isinstance(paths, (str, Path)) else paths) for r in rows(p)):
        assert r["item_id"] not in out, ("item in two sample files", r["item_id"])
        samples = r["samples"][:K]
        assert len(samples) == K, r["item_id"]
        cands = [candidate(s["text"], s["finish_reason"], s["out_tokens"], eq) for s in samples]
        groups = groups_of(cands, lambda a, b, c=cands: eq.symmetric(c[a]["pred"], c[b]["pred"]))
        out[r["item_id"]] = {"answers": [c["pred"] if c["status"] == "ok" else None for c in cands],
                             "margin": winner(groups)[2] / K}
    return out


def mc_choices(path):
    meta = {r["item_id"]: r for r in rows(REPO / "data/processed/mmlupro/items.jsonl")}
    out = {}
    for r in rows(path):
        if r.get("split") not in (None, "test"):
            continue
        n_opt = meta[r["item_id"]]["n_options"]
        letters = [s["pred"] if isinstance(s["pred"], str) and len(s["pred"]) == 1 and "A" <= s["pred"] < chr(65 + n_opt)
                   else None for s in r["samples"][:K]]
        cands = [{"pred": x, "status": "ok" if x is not None else "invalid"} for x in letters]
        groups = groups_of(cands, lambda a, b, c=cands: c[a]["pred"] == c[b]["pred"])
        out[r["item_id"]] = {"answers": letters, "margin": winner(groups)[2] / K}
    return out


def select(choices, share):
    ids = sorted(choices, key=lambda i: (choices[i]["margin"], tie_key(i)))
    return ids[:round(len(ids) * share)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["math", "mc"])
    ap.add_argument("--samples", nargs="+", help="samples.jsonl of B1 (several files: the chunks of one run)")
    ap.add_argument("--consistency")
    ap.add_argument("--share", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.task == "math":
        eq = Equivalence()
        try:
            choices = math_choices(a.samples, eq)
            guards = list(eq.guards)
        finally:
            eq.close()
        source = a.samples
    else:
        choices, guards, source = mc_choices(a.consistency), [], a.consistency
    chosen = select(choices, a.share)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i in chosen:
            f.write(json.dumps({"item_id": i, **choices[i]}, ensure_ascii=False) + "\n")
    sources = source if isinstance(source, list) else [source]
    summary = {"task": a.task, "source": [str(s) for s in sources], "source_sha256": [sha(s) for s in sources], "items": len(choices),
               "share": a.share, "selected": len(chosen), "max_selected_margin": choices[chosen[-1]]["margin"] if chosen else None,
               "guards": guards, "out_sha256": sha(out)}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
