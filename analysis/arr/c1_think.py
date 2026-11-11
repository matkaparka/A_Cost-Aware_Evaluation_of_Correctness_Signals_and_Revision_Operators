"""C1 of the ARR plan (docs/planning/arr_round/plan.md §C1; Amendments 2 and 3): the thinking re-answer as an
escalation operator. Stored outputs only; no generation and no model.

Settings: math Qwen3-4B and 8B (D-nat test pool, n = 5,052) and MMLU-Pro Qwen3-4B (n = 3,457). A setting enters only
when every chunk of its C1 run is synced and hash-checked (Amendment 3: a run is complete when all its chunks are).
The others are listed as TODO(missing).

Scoring (Amendment 2): the text after the think block; an open think block or a truncation is no usable answer.
- Math: the common gold-blind rule of A0 (`common.candidate`), in two phases. Every C1 and P4 extraction is written
  and hashed before any gold is read; then each is scored with equivalent(pred, gold).
- MMLU-Pro: the frozen letter scorer (`src/score/mc.py` extract) on the post-think text, given the item without its
  gold.

Analysis (plan §C1):
- THINK ALL vs KEEP, with completion and content parts, and the cost in generated tokens (thinking included);
- selective THINK by each signal: curves over 0-100 % of items, Eq. (2) terms at 10 % and 20 %, and the share and
  cost at which each signal reaches 50 %, 80 % and 100 % of THINK ALL's gain;
- F2 and V5 at matched token cost: the nearest curve points, side by side, never interpolated;
- the DETECT-oracle (and the BENEFIT-oracle) as bounds;
- §5 members 6 and 7 (the final Holm runs over the complete family elsewhere);
- the cross-check with the stored P4 thinking outputs on the P4 subset (descriptive).
Cost: generated tokens per item after the shared a0. Reading a signal costs nothing, except agreement and margin,
which pay for their five samples.

Usage: python -B analysis/arr/c1_think.py
-> analysis/out/arr/c1_extractions.jsonl.gz (frozen before any gold), c1_labels.jsonl.gz, c1_think.json,
   c1_curves.csv.gz, c1_think_manifest.json; reports/arr_c1.md
"""
import csv
import gzip
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1a2_replays import GRID, SHARES, Pol, add_math_samples, describe, load_math, load_mc, select  # noqa: E402
import b1_samples as B1  # noqa: E402
from common import (J1, OUT, REPO, Boot, Equivalence, candidate, dump, guard_check, manifest, pfmt, pin,  # noqa: E402
                    plan_core_hash, plan_hash, rows, rows_gz, sha, unique, write_rows_gz)
from src.score import mc as MC  # noqa: E402
from src.score.scorer import strip_thinking  # noqa: E402

SETTINGS = {"math 4B": ("math", "4B"), "math 8B": ("math", "8B"), "MMLU-Pro 4B": ("mc", "4B")}
THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
MAX_TOKENS = 16384
FRACTIONS = (0.5, 0.8, 1.0)
REFS = ("REF detect", "REF benefit")
FAMILY = {6: ("THINK by probe 20%", "THINK by verbal 20%"), 7: ("THINK ALL", "KEEP")}


# ----------------------------------------------------------------------------- inputs
def verified(d, name):
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert sha(d / name) == man["files"][name], ("manifest checksum", d.name, name)
    return pin(d / name), man


def c1_run(task, model, pool):
    """Every chunk of the C1 run of (task, model), hash-checked. Returns (records by item, info) or (None, reason)."""
    root = REPO / "results" / "arr" / "c1" / model
    pat = re.compile(rf"^c1-think-{task}-test-c(\d+)-{re.escape(model)}-([0-9a-f]{{7}})$")
    by_commit = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        m = pat.match(d.name)
        if m and (d / "manifest.json").exists():
            by_commit.setdefault(m.group(2), {})[int(m.group(1))] = d
    if not by_commit:
        return None, "no chunk synced"
    assert len(by_commit) == 1, ("C1 chunks from more than one code commit (CLAUDE.md rule 8)", task, model, sorted(by_commit))
    (commit, chunks), = by_commit.items()
    recs, monitor, sizes, gpus = {}, {}, set(), set()
    for c, d in sorted(chunks.items()):
        path, man = verified(d, "generations.jsonl")
        cfg = json.loads(verified(d, "config.json")[0].read_text(encoding="utf-8"))
        assert (cfg["stage"], cfg["task"], cfg["model"], cfg["split"], cfg["chunk_index"]) == ("think", task, model, "test", c), d.name
        assert cfg["max_tokens"] == MAX_TOKENS and cfg["think_sampling"] == THINK_SAMPLING, d.name
        assert man["env"].get("VLLM_BATCH_INVARIANT") == "1" and man["code_commit"].startswith(commit), d.name
        sizes.add(cfg["chunk_size"])
        gpus.add(man["gpu"])
        for r in rows(path):
            assert r["item_id"] not in recs, ("item in two chunks", task, model, r["item_id"])
            recs[r["item_id"]] = r
        for r in rows(verified(d, "rows.jsonl")[0]):
            monitor[r["item_id"]] = bool(r["correct"])
    assert len(sizes) == 1, ("chunk sizes differ", task, model, sizes)
    need = math.ceil(len(pool) / sizes.pop())
    if sorted(chunks) != list(range(need)):
        return None, f"{len(chunks)} of {need} chunks synced"
    assert set(recs) == set(pool), ("the chunks do not cover the pool exactly", task, model)
    return recs, {"run_commit": commit, "chunks": need, "gpus": sorted(gpus), "monitor": monitor,
                  "chunk_runs": [chunks[c].name for c in sorted(chunks)]}


def p4_runs(model):
    """The stored P4 thinking outputs of condition B1 (the initial prompt, think block open), both seeds."""
    out, runs = {}, []
    for d in sorted((REPO / "results" / "main" / model).glob(f"main-thinking-test-c*-{model}-*")):
        path, man = verified(d, "generations.jsonl")
        runs.append({"run": d.name, "gpu": man["gpu"], "vllm": man["backend"]["version"]})
        for r in rows(path):
            if r["cond"] == "B1":
                key = (r["item_id"], int(r["seed"]))
                assert key not in out, ("duplicate P4 row", model, key)
                out[key] = r
    return out, runs


# ----------------------------------------------------------------------------- extraction (gold-blind) and scoring
def think_block(text):
    return "closed" if "</think>" in text else "open" if "<think>" in text else "none"


def think_body(text):
    """The text after the think block, as src/exp/arr_runs.py score_text takes it; None when the block is left open."""
    body = strip_thinking(text)
    if body is not None and "</think>" in body:   # a closing tag without its opening tag: keep the answer part
        body = body.rsplit("</think>", 1)[1].strip()
    return body


def extract(task, g, item, eq):
    """Gold-blind extraction of one thinking output. status != "ok" means no usable answer (Amendment 2)."""
    body = think_body(g["text"])
    if body is None:
        pred, status, raw = None, "unclosed", None
    elif task == "math":
        c = candidate(body, g["finish_reason"], g["out_tokens"], eq)
        pred, status, raw = c["pred"], c["status"], c["raw_extraction_status"]
    else:
        pred, raw = MC.extract(body, item)
        status = "ok" if pred is not None else "unparsed"
    if g["finish_reason"] == "length":
        status = "truncated"
    return {"pred": pred, "status": status, "raw_extraction_status": raw, "think_block": think_block(g["text"]),
            "out_tokens": int(g["out_tokens"]), "prompt_tokens": int(g["prompt_tokens"]), "finish_reason": g["finish_reason"]}


def gold_free(item):
    return {k: v for k, v in item.items() if k != "gold"}


# ----------------------------------------------------------------------------- analysis helpers
def acq_vector(P, sig):
    """Tokens a signal costs per item (plan §C1): the five samples for agreement and margin, nothing otherwise."""
    if sig in ("agreement", "margin") and "samples" in P:
        return P["samples"]["tok"].sum(axis=1)
    return np.zeros(P["n"])


def score_of(P, sig, y1):
    if sig == "REF detect":
        return P["y0"]
    if sig == "REF benefit":
        return -(y1 - P["y0"])
    return P["sig"][sig]


def ref_name(sig, op, tag):
    return f"REF {sig.split()[1].upper()}->{op} {tag}"


def selective(P, op, sig, share):
    y1, gen, prompt, answered = op
    sel = select(score_of(P, sig, y1), P["tie"], share)
    return Pol(np.where(sel, y1, P["y0"]), acq_vector(P, sig) + sel * gen, sel * prompt,
               np.where(sel, answered, P["a0_answered"]), sel)


def policies(P, T):
    y0, z = P["y0"], np.zeros(P["n"])
    f2 = P["ops"]["F2"]
    pol = {"KEEP": Pol(y0, z, z, P["a0_answered"]),
           "THINK ALL": Pol(T["y"], T["gen"], T["prompt"], T["answered"]),
           "F2 ALL": Pol(f2["y1"], f2["gen"], f2["prompt"], f2["answered"])}
    if "samples" in P:
        S = P["samples"]
        pol["V5"] = Pol(S["vk_correct"][5], S["tok"].sum(axis=1), 5 * S["sprompt"], S["vk_answered"][5])
    op = (T["y"], T["gen"], T["prompt"], T["answered"])
    for share in SHARES:
        tag = f"{round(share * 100)}%"
        for sig in P["sig"]:
            pol[f"THINK by {sig} {tag}"] = selective(P, op, sig, share)
        for sig in REFS:
            pol[ref_name(sig, "THINK", tag)] = selective(P, op, sig, share)
    return pol


def eq2(P, y1, gen, boot):
    """Eq. (2) terms of the selective operator at 10 % and 20 % (as analysis/arr/a1a2_replays.py eq2_table)."""
    y0, n = P["y0"], P["n"]
    out = []
    for sig in list(P["sig"]) + ["random"] + list(REFS):
        for share in SHARES:
            m = round(n * share)
            if sig == "random":
                w, r, d = boot.ratio_ci(1 - y0, np.ones(n)), boot.ratio_ci((1 - y0) * y1, 1 - y0), boot.ratio_ci(y0 * (1 - y1), y0)
                delta = boot.ci((m / n) * (y1 - y0))
                cost = (m / n) * float(gen.mean())
            else:
                sel = select(score_of(P, sig, y1), P["tie"], share)
                w = boot.ratio_ci(sel * (1 - y0), sel.astype(float))
                r = boot.ratio_ci(sel * (1 - y0) * y1, sel * (1 - y0))
                d = boot.ratio_ci(sel * y0 * (1 - y1), sel * y0)
                delta = boot.ci(sel * (y1 - y0))
                cost = float(acq_vector(P, sig).mean() + (sel * gen).mean())
            wr = w["estimate"] / 100
            rr = r["estimate"] / 100 if r["n_den"] else 0.0
            dr = d["estimate"] / 100 if d["n_den"] else 0.0
            resid = delta["estimate"] / 100 - (m / n) * (wr * rr - (1 - wr) * dr)
            assert abs(resid) < 1e-12, (P["setting"], sig, share, resid)
            out.append({"setting": P["setting"], "operator": "THINK", "signal": sig, "share_pct": round(share * 100),
                        "selected_n": m, "w_S": w, "repair_rate": r, "damage_rate": d, "delta_pp": delta,
                        "generated_tokens_per_item": cost})
    return out


def curve(P, y1, gen, sig):
    """Selective-operator curve over shares 0-100 % (1-point steps); random is the exact expectation."""
    y0, n = P["y0"], P["n"]
    order = np.arange(n) if sig == "random" else np.lexsort((P["tie"], np.asarray(score_of(P, sig, y1), dtype=float)))
    cs = {k: np.r_[0, np.cumsum(v[order])] for k, v in
          {"d": y1 - y0, "g": gen, "w": 1 - y0, "r": (1 - y0) * y1, "dm": y0 * (1 - y1), "c": y0}.items()}
    acq = 0.0 if sig == "random" else float(acq_vector(P, sig).mean())
    out = []
    for share in GRID:
        m = round(n * share)
        vals = {k: (m / n) * cs[k][-1] for k in cs} if sig == "random" else {k: float(cs[k][m]) for k in cs}
        out.append({"share_pct": round(share * 100), "selected_n": m, "delta_pp": 100 * vals["d"] / n,
                    "generated_tokens_per_item": acq + vals["g"] / n,
                    "w_S": vals["w"] / m if m else None, "repair_rate": vals["r"] / vals["w"] if vals["w"] else None,
                    "damage_rate": vals["dm"] / vals["c"] if vals["c"] else None})
    return out


def point_ci(P, y1, sig, share_pct, boot):
    """Δ vs KEEP with its CI at one curve point (the selection the curve uses at that share)."""
    n, m = P["n"], round(P["n"] * share_pct / 100)
    if sig == "random":
        return boot.ci((m / n) * (y1 - P["y0"]))
    return boot.ci(select(score_of(P, sig, y1), P["tie"], share_pct / 100) * (y1 - P["y0"]))


def reach(P, gain_pp, think_curves):
    """Smallest share (1-point steps) at which each signal's selective THINK reaches 50/80/100 % of THINK ALL's gain."""
    out = {"think_all_gain_pp": gain_pp, "by_signal": {}, "lowest_cost": {}}
    for sig, cv in think_curves.items():
        entry = {}
        for f in FRACTIONS:
            key = f"{round(100 * f)}%"
            hit = None if gain_pp <= 0 else next((pt for pt in cv if pt["delta_pp"] >= f * gain_pp - 1e-9), None)
            entry[key] = None if hit is None else {k: hit[k] for k in ("share_pct", "delta_pp", "generated_tokens_per_item")}
        out["by_signal"][sig] = entry
    for f in FRACTIONS:
        key = f"{round(100 * f)}%"
        ranked = sorted(((v[key]["generated_tokens_per_item"], v[key]["share_pct"], sig) for sig, v in out["by_signal"].items()
                         if v[key] is not None and not sig.startswith("REF")))
        out["lowest_cost"][key] = [{"signal": s, "share_pct": sh, "generated_tokens_per_item": c} for c, sh, s in ranked]
    if gain_pp <= 0:
        out["note"] = "THINK ALL does not gain over KEEP: no share of its gain is defined"
    return out


def nearest(cv, cost):
    return min(cv, key=lambda pt: (abs(pt["generated_tokens_per_item"] - cost), pt["share_pct"]))


def matched(P, T, boot, pols, think_curves, f2_curves):
    """F2 and V5 against THINK at matched generated-token cost: nearest curve points, side by side, no interpolation."""
    out = {"references": {}, "think_at_reference_cost": [], "f2_at_think_cost": []}
    f2 = P["ops"]["F2"]
    for name in ("F2 ALL", "V5"):
        if name not in pols:
            continue
        p = pols[name]
        cost = float(p.gen.mean())
        out["references"][name] = {"generated_tokens_per_item": cost, "delta_vs_keep": boot.ci(p.y - P["y0"])}
        for sig, cv in think_curves.items():
            pt = nearest(cv, cost)
            # agreement and margin pay for five samples before any escalation: below that cost no point exists
            out["think_at_reference_cost"].append({"reference": name, "signal": sig, **pt,
                                                   "reachable": cv[0]["generated_tokens_per_item"] <= cost + 1e-9,
                                                   "delta_ci": point_ci(P, T["y"], sig, pt["share_pct"], boot)})
    for sig, cv in think_curves.items():
        for share in SHARES:
            tpt = cv[round(share * 100)]
            fpt = nearest(f2_curves[sig], tpt["generated_tokens_per_item"])
            out["f2_at_think_cost"].append({
                "signal": sig, "think": {**tpt, "delta_ci": point_ci(P, T["y"], sig, tpt["share_pct"], boot)},
                "f2": {**fpt, "delta_ci": point_ci(P, f2["y1"], sig, fpt["share_pct"], boot)}})
    return out


def family(P, pols, boot):
    out = []
    for member, (a, b) in FAMILY.items():
        if a in pols and b in pols:
            ci = boot.ci(pols[a].y - pols[b].y)
            out.append({"setting": P["setting"], "member": member, "a": a, "b": b, "difference": ci, "p_boot": ci["p_boot"],
                        "token_difference": float((pols[a].gen - pols[b].gen).mean())})
    return out


def cross_check(model, P, T, lab_p4, p4_info):
    """C1 (seed index 0) against the stored P4 B1 thinking outputs on the P4 subset items (descriptive)."""
    idx = {i: j for j, i in enumerate(P["ids"])}
    items = sorted({i for (i, s) in lab_p4})
    assert all((i, s) in lab_p4 for i in items for s in (0, 1)), ("P4 B1 rows missing a seed", model)
    j = np.array([idx[i] for i in items])
    y_c1 = T["y"][j]
    y_s = {s: np.array([lab_p4[(i, s)]["correct"] for i in items], dtype=float) for s in (0, 1)}
    boot = Boot([P["canonical"][k] for k in j])
    same_text = sum(T["text"][idx[i]] == lab_p4[(i, 0)]["text"] for i in items)
    return {"model": model, "p4_runs": p4_info, "items": len(items), "identical_texts_seed0": int(same_text),
            "accuracy_pct": {"C1": 100 * float(y_c1.mean()), "P4 seed 0": 100 * float(y_s[0].mean()),
                             "P4 seed 1": 100 * float(y_s[1].mean())},
            "C1_minus_P4_seed0": boot.ci(y_c1 - y_s[0]), "C1_minus_P4_seed1": boot.ci(y_c1 - y_s[1]),
            "note": "P4 subset: original-pool D-nat items, equal allocation by a0 correctness; unweighted accuracy on "
                    "those items, not a pool estimate"}


def clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        return clean(v.item())
    return v


# ----------------------------------------------------------------------------- report
def fmt(ci):
    return f"{ci['estimate']:+.2f} [{ci['lo']:+.2f}, {ci['hi']:+.2f}]"


def pct(x):
    return "—" if x is None else f"{x:.1f}"


def report(res):
    L = ["# ARR round, C1: thinking re-answer as an escalation operator", "",
         f"Generated by `analysis/arr/c1_think.py` (plan `docs/planning/arr_round/plan.md`, core sha256 "
         f"`{res['plan_core_sha256'][:16]}…`). Post hoc and exploratory (DEVIATIONS D20); added in revision.",
         "Δ in percentage points vs KEEP, with paired 95 % cluster-bootstrap intervals (4,000 replicates, seed 20261110). "
         "Costs are generated tokens per item after the shared a0, thinking tokens included. The REF rows use the gold: "
         "they are bounds, not policies.", ""]
    missing = {s: v["status"] for s, v in res["settings"].items() if v["status"] != "complete"}
    if missing:
        L += ["Not yet complete: " + "; ".join(f"{s}: TODO(missing: C1 run, {why})" for s, why in missing.items()) + ".", ""]
    done = {s: v for s, v in res["settings"].items() if v["status"] == "complete"}
    L += ["## THINK ALL", "",
          "| Setting | n | KEEP % | THINK ALL % | Δ vs KEEP | Completion | Content | Format loss | Content damage | "
          "Gen. tok/item | Truncated % | Open think % |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, v in done.items():
        k, t = v["policies"]["KEEP"], v["policies"]["THINK ALL"]
        L.append(f"| {s} | {v['n']} | {k['accuracy_pct']:.2f} | {t['accuracy_pct']:.2f} | {fmt(t['delta_vs_keep'])} | "
                 f"{t['repairs_completion']['estimate']:+.2f} | {t['repairs_content']['estimate']:+.2f} | "
                 f"{-t['damage_format_loss']['estimate']:+.2f} | {-t['damage_content']['estimate']:+.2f} | "
                 f"{t['generated_tokens_per_item']:.0f} | {v['think_status']['truncated_pct']:.2f} | "
                 f"{v['think_status']['open_think_pct']:.2f} |")
    L += ["", "## Selective THINK at 10 % and 20 % (Eq. (2) terms)", "",
          "| Setting | Signal | Share | w_S | Repair rate | Damage rate | Δ vs KEEP | Gen. tok/item |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s, v in done.items():
        for r in v["eq2"]:
            L.append(f"| {s} | {r['signal']} | {r['share_pct']} % | {pct(r['w_S']['estimate'])} | {pct(r['repair_rate']['estimate'])} | "
                     f"{pct(r['damage_rate']['estimate'])} | {fmt(r['delta_pp'])} | {r['generated_tokens_per_item']:.0f} |")
    L += ["", "## Share of items needed to reach 50 %, 80 % and 100 % of THINK ALL's gain", "",
          "Each cell: share of items escalated (generated tokens per item at that point). Selection in 1-point steps; "
          "the lowest-ranked items by the signal are escalated first.", "",
          "| Setting | THINK ALL gain | Signal | 50 % | 80 % | 100 % |", "|---|---:|---|---:|---:|---:|"]
    for s, v in done.items():
        rr = v["reach"]
        for sig, e in rr["by_signal"].items():
            cells = ["not reached" if e[k] is None else f"{e[k]['share_pct']} % ({e[k]['generated_tokens_per_item']:.0f})"
                     for k in ("50%", "80%", "100%")]
            L.append(f"| {s} | {rr['think_all_gain_pp']:+.2f} pp | {sig} | " + " | ".join(cells) + " |")
        for k, ranked in rr["lowest_cost"].items():
            if ranked:
                L.append(f"| {s} | | lowest cost to {k} | {ranked[0]['signal']} ({ranked[0]['generated_tokens_per_item']:.0f}) | | |")
    L += ["", "## THINK at the token cost of F2 ALL and V5 (nearest curve points, side by side)", "",
          "| Setting | Reference | Reference Δ (tok/item) | Signal | THINK share | THINK Δ | THINK tok/item |",
          "|---|---|---:|---|---:|---:|---:|"]
    for s, v in done.items():
        refs = v["matched"]["references"]
        for r in v["matched"]["think_at_reference_cost"]:
            ref = refs[r["reference"]]
            head = f"| {s} | {r['reference']} | {fmt(ref['delta_vs_keep'])} ({ref['generated_tokens_per_item']:.0f}) | {r['signal']} | "
            if not r["reachable"]:
                L.append(head + f"— | not reachable: the signal's samples alone cost {r['generated_tokens_per_item']:.0f} | |")
                continue
            L.append(head + f"{r['share_pct']} % | {fmt(r['delta_ci'])} | {r['generated_tokens_per_item']:.0f} |")
    L += ["", "## Selective F2 at the token cost of selective THINK (same signal, nearest curve point)", "",
          "| Setting | Signal | THINK share | THINK Δ | THINK tok/item | F2 share | F2 Δ | F2 tok/item |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s, v in done.items():
        for r in v["matched"]["f2_at_think_cost"]:
            t, f = r["think"], r["f2"]
            L.append(f"| {s} | {r['signal']} | {t['share_pct']} % | {fmt(t['delta_ci'])} | {t['generated_tokens_per_item']:.0f} | "
                     f"{f['share_pct']} % | {fmt(f['delta_ci'])} | {f['generated_tokens_per_item']:.0f} |")
    L += ["", "## §5 family members 6 and 7 (Holm over the complete family is applied in the final family script)", "",
          "| Setting | Member | Contrast | Difference | Token difference | p (boot) |", "|---|---:|---|---:|---:|---:|"]
    for f in res["family"]:
        L.append(f"| {f['setting']} | {f['member']} | {f['a']} − {f['b']} | {fmt(f['difference'])} | "
                 f"{f['token_difference']:+.0f} | {pfmt(f['p_boot'])} |")
    L += ["", "## Cross-check with the stored P4 thinking outputs (descriptive)", "",
          "| Model | P4 items | Identical texts (seed 0) | C1 % | P4 seed 0 % | P4 seed 1 % | C1 − P4 seed 0 | C1 − P4 seed 1 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in res["cross_check"]:
        a = c["accuracy_pct"]
        L.append(f"| {c['model']} | {c['items']} | {c['identical_texts_seed0']} | {a['C1']:.2f} | {a['P4 seed 0']:.2f} | "
                 f"{a['P4 seed 1']:.2f} | {fmt(c['C1_minus_P4_seed0'])} | {fmt(c['C1_minus_P4_seed1'])} |")
    if res["cross_check"]:
        L += ["", res["cross_check"][0]["note"] + "."]
    L += ["", "Status counts of the thinking outputs, monitoring disagreements and run details: `analysis/out/arr/c1_think.json`; "
          "curves 0–100 % for THINK and F2: `analysis/out/arr/c1_curves.csv.gz`.", ""]
    (REPO / "reports" / "arr_c1.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- main
def main():
    plan = plan_hash()
    a0 = json.loads(pin(OUT / "a0_common_labels.json").read_text(encoding="utf-8"))
    assert a0["plan_core_sha256"] == plan_core_hash() and not a0["guards"]
    a1 = json.loads(pin(OUT / "a1_policies.json").read_text(encoding="utf-8"))
    labels = list(rows_gz(pin(OUT / "math_common_labels.jsonl.gz")))
    table = {}
    for r in rows(pin(J1 / "artifacts/math/item_evaluations.jsonl")):
        table.setdefault(r["model"], {}).setdefault(r["condition"], {})[r["item_id"]] = r
    math_pops = load_math(table, labels)
    mc_items = unique(rows(pin(REPO / "data/processed/mmlupro/items.jsonl")))
    eq = Equivalence()
    try:
        add_math_samples(math_pops["4B"], eq)
        # Qwen3-8B has no stored samples. Its margin and agreement signals and V5 come from the B1 samples
        # (plan A1: "After B1, the same definitions apply unchanged to the other math models"), through the B1
        # loader, which returns None while B1 is not complete for 8B.
        b1_8b = B1.attach(math_pops["8B"], "8B", eq) is not None
        pops = {"math 4B": math_pops["4B"], "math 8B": math_pops["8B"], "MMLU-Pro 4B": load_mc("4B")}
        runs, status = {}, {}
        for s, (task, model) in SETTINGS.items():
            recs, info = c1_run(task, model, pops[s]["ids"])
            if recs is None:
                status[s] = info
            else:
                runs[s], status[s] = (recs, info), "complete"
        # phase 1: every extraction, gold-blind, frozen and hashed before any gold is read
        extractions, p4 = [], {}
        for s, (recs, _) in runs.items():
            task, model = SETTINGS[s]
            for i in pops[s]["ids"]:
                e = extract(task, recs[i], gold_free(mc_items[i]) if task == "mc" else None, eq)
                extractions.append({"source": "C1", "setting": s, "item_id": i, **e})
            if task == "math" and model in ("4B", "8B"):
                p4[model] = p4_runs(model)
                for (i, seed), g in sorted(p4[model][0].items()):
                    extractions.append({"source": f"P4 seed {seed}", "setting": s, "item_id": i, **extract("math", g, None, eq)})
        frozen = OUT / "c1_extractions.jsonl.gz"
        write_rows_gz(frozen, extractions)
        frozen_hash = sha(frozen)
        print(f"extractions frozen: {len(extractions)} rows, sha256 {frozen_hash[:16]}", flush=True)
        # phase 2: scoring against gold
        gold = {r["item_id"]: r["gold"] for p in (pin(REPO / "data/processed/dnat.jsonl"),
                                                  pin(REPO / "data/processed/ext/dnat_ext.jsonl")) for r in rows(p)}
        for e in extractions:
            if e["status"] != "ok":
                e["correct"] = 0
            elif SETTINGS[e["setting"]][0] == "math":
                e["correct"] = int(eq(e["pred"], gold[e["item_id"]]))
            else:
                e["correct"] = int(e["pred"] == mc_items[e["item_id"]]["gold"])
        guards = list(eq.guards)
    finally:
        eq.close()
    assert sha(frozen) == frozen_hash
    guards = guard_check(guards)
    write_rows_gz(OUT / "c1_labels.jsonl.gz", extractions)

    res = {"plan_sha256_lf": plan, "plan_core_sha256": plan_core_hash(), "extractions_sha256_before_gold": frozen_hash,
           "b1_samples_attached": {"math 8B": b1_8b}, "scorer_guards": guards, "settings": {}, "family": [],
           "cross_check": []}
    curve_rows = []
    for s in SETTINGS:
        if status[s] != "complete":
            res["settings"][s] = {"status": status[s]}
            continue
        P, (recs, info) = pops[s], runs[s]
        lab = {e["item_id"]: e for e in extractions if e["setting"] == s and e["source"] == "C1"}
        T = {"y": np.array([lab[i]["correct"] for i in P["ids"]], dtype=float),
             "gen": np.array([lab[i]["out_tokens"] for i in P["ids"]], dtype=float),
             "prompt": np.array([lab[i]["prompt_tokens"] for i in P["ids"]], dtype=float),
             "answered": np.array([lab[i]["status"] == "ok" for i in P["ids"]]),
             "text": [recs[i]["text"] for i in P["ids"]]}
        boot = Boot(P["canonical"])
        pols = policies(P, T)
        # consistency with A1: KEEP, F2 ALL and V5 are the same policies there (8B's V5: B1's replay of A1)
        a1s = a1["settings"][P["setting"]]["policies"]
        if s == "math 8B" and b1_8b:
            man = json.loads((OUT / "b1_samples_manifest.json").read_text(encoding="utf-8"))
            assert sha(OUT / "b1_policies.json") == man["outputs"]["b1_policies.json"], "b1_policies.json differs from its manifest"
            a1s = {**a1s, "V5": json.loads(pin(OUT / "b1_policies.json").read_text(encoding="utf-8"))["a1"][s]["policies"]["V5"]}
        for name in ("KEEP", "F2 ALL", "V5"):
            if name in pols:
                assert name in a1s, (s, name, "no reference policy to check against")
                assert abs(a1s[name]["accuracy_pct"] - 100 * pols[name].y.mean()) < 1e-9, (s, name)
                assert abs(a1s[name]["generated_tokens_per_item"] - pols[name].gen.mean()) < 1e-9, (s, name)
        sigs = list(P["sig"]) + ["random"] + list(REFS)
        f2 = P["ops"]["F2"]
        think_curves = {sig: curve(P, T["y"], T["gen"], sig) for sig in sigs}
        f2_curves = {sig: curve(P, f2["y1"], f2["gen"], sig) for sig in sigs}
        for op, cvs in (("THINK", think_curves), ("F2", f2_curves)):
            for sig, cv in cvs.items():
                curve_rows += [{"setting": s, "operator": op, "signal": sig, **pt} for pt in cv]
        gain = 100 * float((T["y"] - P["y0"]).mean())
        st = Counter(lab[i]["status"] for i in P["ids"])
        blocks = Counter(lab[i]["think_block"] for i in P["ids"])
        res["settings"][s] = {
            "status": "complete", "n": P["n"], "run": {k: v for k, v in info.items() if k != "monitor"},
            "think_status": {"status_counts": dict(st), "think_block_counts": dict(blocks),
                             "truncated_pct": 100 * st["truncated"] / P["n"], "open_think_pct": 100 * blocks["open"] / P["n"],
                             "no_usable_answer_pct": 100 * float((~T["answered"]).mean()),
                             "mean_generated_tokens": float(T["gen"].mean()), "mean_prompt_tokens": float(T["prompt"].mean()),
                             "monitor_disagreements": int(sum(info["monitor"][i] != bool(lab[i]["correct"]) for i in P["ids"]))},
            "signals": list(P["sig"]), "policies": describe(P, pols, boot), "eq2": eq2(P, T["y"], T["gen"], boot),
            "reach": reach(P, gain, think_curves), "matched": matched(P, T, boot, pols, think_curves, f2_curves)}
        res["family"] += family(P, pols, boot)
        task, model = SETTINGS[s]
        if task == "math" and model in p4:
            lab_p4 = {(e["item_id"], int(e["source"].split()[-1])): {**e, "text": p4[model][0][(e["item_id"], int(e["source"].split()[-1]))]["text"]}
                      for e in extractions if e["setting"] == s and e["source"].startswith("P4")}
            res["cross_check"].append(cross_check(model, P, T, lab_p4, p4[model][1]))
    res = clean(res)
    dump(OUT / "c1_think.json", res)
    buf = io.StringIO()
    if curve_rows:
        w = csv.DictWriter(buf, fieldnames=list(curve_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(curve_rows)
    with (OUT / "c1_curves.csv.gz").open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue().encode("utf-8"))
    report(res)
    manifest("c1_think", {"outputs": {p: sha(OUT / p) for p in ("c1_extractions.jsonl.gz", "c1_labels.jsonl.gz",
                                                                "c1_think.json", "c1_curves.csv.gz")},
                          "settings_status": status})
    print(json.dumps({"status": status, "family": [(f["setting"], f["member"], round(f["difference"]["estimate"], 2))
                                                   for f in res["family"]]}, indent=1))


if __name__ == "__main__":
    main()
