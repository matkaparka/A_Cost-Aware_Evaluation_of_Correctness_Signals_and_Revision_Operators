"""Block 4B-exp1c-robust estimates (exploratory; prereg §3.3; src/exp/channel.py). Written before any output of the
block existed and checked on synthetic rows only (tests/test_p4_channel.py). Estimates and 95 % cluster-bootstrap
CIs only; nothing here is a test (prereg §3.1, §8: per-block analyses report estimates and CIs, no p-values).
-> reports/04_<out>.md and analysis/out/<out>.json

- Unit: the canonical problem (data/processed/dnat.jsonl); `Boot` of analysis/p6_confirmatory.py (10,000 resamples,
  seed 0), one Boot per item set, shared by every statistic on that set. Truncated or unparsed = wrong.
- Exp 1C (H1c): accuracy, W->R and R->W of the four greedy cells (F3(0.9) via tool / user / assistant, F3(0.7) via
  user) on every item with all four. The channel range = max - min of the three F3(0.9) accuracies, bootstrapped as
  a statistic (the CI is for the range); the three pairwise channel differences; Δacc(F3(0.9) - F3(0.7)) on the
  plain-user channel; and range - Δacc(0.9 - 0.7) over the same resamples. The range is >= 0 by construction and is
  biased upward by noise, so it is read with the pairwise differences.
- Sampled robustness (the 300-item subset, 3 seeds, T 0.7, top-p 0.8): each item's outcome per condition is its mean
  over the 3 seeds. Per condition: accuracy, W->R (mean over initially wrong items), R->W (1 - mean over initially
  right items). Analogues, with p6_confirmatory's estimand definitions: H1a F2 - F1; H1b half the item-level OLS
  slope of correctness on p (Y @ LW / 2); H1d F5 - mean of F3-perm[0..2]. Beside each, the greedy main-run value on the
  same items (--main, the 4B Exp 1A rows) and sampled - greedy over the same resamples. The subset is the first 300 in
  the seeded order, a random subsample of the pool, so no weights.
- Mask robustness (same subset, greedy): accuracy at each p and the H1b analogue under verifier-stream seeds 0, 1, 2;
  seed k - seed 0; and the main run's seed-0 value on the same items.
- Sensitivity: everything above again without items whose a0 tokens differ across channels (rows'
  `solution_tokens_identical` False; design §6.0 boundary rule, accepted 2026-11-06).
- Checks: all runs share one locked code commit (CLAUDE.md rule 8); no (exp, cond, item, seed) row twice; every item
  of a component has every cell; greedy texts identical to the main run's for the same prompt_key (batch invariance),
  counted when both generations files are present.
- Sample status per component: complete (planned n: the pool for Exp 1C, the subset size for the rest, from
  chunk.json), pre-registered partial (>= MIN_N, configs/blocks.yaml min_n of the block), descriptive (< MIN_N).
Usage: python analysis/p4_channel.py --runs <chunk run dirs...> --main results/main/4B/main-exp01-test-4B-a7a0579
       --out p4_4B_channel [--n-boot 10000]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from p6_confirmatory import K_PERM, LW, N_BOOT, P_LEVELS, Boot, jl, wmean  # noqa: E402

MIN_N = 150                      # configs/blocks.yaml 4B-exp1c-robust min_n
CHANNELS = ("tool", "user", "assistant")
C1 = [f"F3(0.9):{ch}" for ch in CHANNELS] + ["F3(0.7):user"]
MASK_SEEDS = (0, 1, 2)
N_SAMPLES = 3
SAMPLED = (["B0", "B1", "F0", "F1", "F2", "F2-scratch"] + [f"F3({p})" for p in P_LEVELS] + ["F5"]
           + [f"F3-perm[{k}]" for k in range(K_PERM)])
PAIRS = [("tool", "user"), ("assistant", "user"), ("tool", "assistant")]
ANALOGUES = {"H1a analogue (F2 - F1)": "H1a", "H1b analogue (half the item-level OLS slope on p)": "H1b",
             "H1d analogue (F5 - mean F3-perm)": "H1d"}


def ci(boot, f, mask=None):
    """[estimate, lo, hi] of statistic f(multiplicities), or None when the mask is empty."""
    if mask is not None and not np.asarray(mask).any():
        return None
    return [float(x) for x in boot.ci(f, mask)]


def mean_of(v):
    return lambda m: wmean(v, m)


def cell(boot, y, c0):
    return {"acc": ci(boot, mean_of(y)), "W->R": ci(boot, mean_of(y), ~c0), "R->W": ci(boot, mean_of(1 - y), c0)}


def half_slope(Y):
    """Item-level H1b estimand of p6_confirmatory: half the OLS slope of correctness on p."""
    return 0.5 * (np.stack([Y[f"F3({p})"] for p in P_LEVELS], 1) @ LW)


def analogues(Y):
    """Item-level estimands of p6_confirmatory: d1a, slope / 2, d1d (their means are the estimates)."""
    return {"H1a": Y["F2"] - Y["F1"], "H1b": half_slope(Y),
            "H1d": Y["F5"] - np.mean([Y[f"F3-perm[{k}]"] for k in range(K_PERM)], 0)}


def load_runs(runs):
    rows, flags, commits, planned = [], {}, set(), {}
    for run in runs:
        d = Path(run)
        commits.add(json.loads((d / "manifest.json").read_text())["code_commit"])
        ch = json.loads((d / "chunk.json").read_text())
        flags.update(ch["solution_tokens_identical"])
        planned = {"1C": ch["pool"], "subset": ch["subset"]}
        rows += jl(d / "rows.jsonl")
    if len(commits) != 1:
        raise SystemExit(f"runs come from different code commits {sorted(commits)} (CLAUDE.md rule 8)")
    seen = Counter((r["exp"], r["cond"], r["item_id"], r.get("sample"), r.get("mask_seed")) for r in rows)
    dup = [k for k, n in seen.items() if n > 1]
    if dup:
        raise SystemExit(f"{len(dup)} rows occur twice, e.g. {dup[0]} (overlapping chunks?)")
    return rows, flags, commits.pop(), planned


def index(rows):
    c1, sampled, mask, c0, status, toks = defaultdict(dict), defaultdict(dict), defaultdict(dict), {}, defaultdict(Counter), defaultdict(list)
    for r in rows:
        i = r["item_id"]
        c0[i] = bool(r["correct0"])
        if r["exp"] == "1C":
            c1[r["cond"]][i] = float(r["correct"])
            key = f"1C {r['cond']}"
        elif r["exp"] == "sampled":
            sampled[r["cond"]].setdefault(i, {})[r["sample"]] = float(r["correct"])
            key = f"sampled {r['cond']}"
        elif r["exp"] == "mask":
            mask[(r["mask_seed"], r["cond"])][i] = float(r["correct"])
            key = f"mask seed {r['mask_seed']} {r['cond']}"
        else:
            raise SystemExit(f"unknown exp {r['exp']!r}")
        if key:
            status[key][r["status"]] += 1
            if r.get("out_tokens") is not None:
                toks[key].append(r["out_tokens"])
    return c1, sampled, mask, c0, status, toks


def load_main(main, ids):
    out = defaultdict(dict)
    for r in jl(Path(main) / "rows.jsonl"):
        if r.get("exp") == "1A" and r.get("data") == "dnat" and r.get("stage") != 1 and r["item_id"] in ids:
            out[r["cond"]][r["item_id"]] = float(r["correct"])
    return out


def greedy_identity(runs, main):
    """Greedy generations of this block vs the main run's, for prompt_keys present in both."""
    gm = Path(main) / "generations.jsonl"
    mine = {}
    for run in runs:
        p = Path(run) / "generations.jsonl"
        if not p.exists() or not gm.exists():
            return None
        for g in jl(p):
            if g.get("sampling") == "greedy":
                mine[g["prompt_key"]] = g["text"]
    same = n = 0
    for line in open(gm, encoding="utf-8"):   # streamed: the main run's file is large
        g = json.loads(line)
        if g["prompt_key"] in mine:
            n += 1
            same += g["text"] == mine[g["prompt_key"]]
    return {"compared": n, "identical": same}


def status_of(n, planned):
    if n >= planned:
        return "complete"
    return "pre-registered partial sample" if n >= MIN_N else "descriptive only (below min_n)"


def analyse(c1, sampled, mask, c0, main, canon, ids1, ids_s, n_boot):
    res = {"n_1C": len(ids1), "n_subset": len(ids_s)}
    # ---- Exp 1C
    if ids1:
        boot = Boot(np.array([canon[i] for i in ids1]), n_boot)
        y0 = np.array([c0[i] for i in ids1])
        Y = {c: np.array([c1[c][i] for i in ids1]) for c in C1}
        A = np.stack([Y[f"F3(0.9):{ch}"] for ch in CHANNELS], 1)

        def rng(m):
            acc = (m @ A) / m.sum()
            return float(acc.max() - acc.min())
        d97 = Y["F3(0.9):user"] - Y["F3(0.7):user"]
        res["1C"] = {
            "cells": {c: cell(boot, Y[c], y0) for c in C1},
            "range at F3(0.9)": ci(boot, rng),
            "pairwise": {f"{a} - {b}": ci(boot, mean_of(Y[f"F3(0.9):{a}"] - Y[f"F3(0.9):{b}"])) for a, b in PAIRS},
            "F3(0.9) - F3(0.7), user": ci(boot, mean_of(d97)),
            "range - (F3(0.9) - F3(0.7))": ci(boot, lambda m: rng(m) - wmean(d97, m)),
        }
    # ---- sampled and mask robustness
    if ids_s:
        boot = Boot(np.array([canon[i] for i in ids_s]), n_boot)
        y0 = np.array([c0[i] for i in ids_s])
        S = {c: np.array([np.mean([sampled[c][i][s] for s in range(N_SAMPLES)]) for i in ids_s]) for c in SAMPLED}
        G = {c: np.array([main[c][i] for i in ids_s]) for c in SAMPLED}
        res["sampled"] = {"cells": {}, "analogues": {}}
        for c in SAMPLED:
            res["sampled"]["cells"][c] = {**cell(boot, S[c], y0), "greedy": cell(boot, G[c], y0),
                                          "sampled - greedy": ci(boot, mean_of(S[c] - G[c]))}
        aS, aG = analogues(S), analogues(G)
        for name, h in ANALOGUES.items():
            res["sampled"]["analogues"][name] = {"sampled": ci(boot, mean_of(aS[h])), "greedy": ci(boot, mean_of(aG[h])),
                                                 "sampled - greedy": ci(boot, mean_of(aS[h] - aG[h]))}
        M = {k: {f"F3({p})": np.array([mask[(k, f"F3({p})")][i] for i in ids_s]) for p in P_LEVELS} for k in MASK_SEEDS}
        slope = {k: half_slope(M[k]) for k in MASK_SEEDS}
        res["mask"] = {f"seed {k}": {"acc": {c: ci(boot, mean_of(v)) for c, v in M[k].items()},
                                     "H1b analogue": ci(boot, mean_of(slope[k])),
                                     **({f"seed {k} - seed 0": ci(boot, mean_of(slope[k] - slope[0]))} if k else {})}
                       for k in MASK_SEEDS}
        res["mask"]["main run (seed 0)"] = {"H1b analogue": ci(boot, mean_of(aG["H1b"]))}
    return res


def complete_ids(table, conds, label):
    ids = sorted({i for c in conds for i in table[c]})
    missing = [(c, i) for c in conds for i in ids if i not in table[c]]
    if missing:
        raise SystemExit(f"{label}: {len(missing)} missing cells, e.g. {missing[0]}")
    return ids


def pct(x):
    return "–" if x is None else f"{100 * x:.1f} %"


def pp(x):
    return "–" if x is None else f"{100 * x:+.1f}"


def ci_pct(v):
    return "–" if v is None else f"{pct(v[0])} [{pct(v[1])}, {pct(v[2])}]"


def ci_pp(v):
    return "–" if v is None else f"{pp(v[0])} [{pp(v[1])}, {pp(v[2])}]"


def report(out, a):
    L = [f"# P4 Exp 1C channel and robustness, 4B test (generated by analysis/p4_channel.py)", "",
         f"Runs: {len(a.runs)} chunk runs at commit `{out['code_commit']}`; main run `{Path(a.main).as_posix()}`. "
         "Exploratory (prereg §3.3): estimates and 95 % cluster-bootstrap CIs only "
         f"({out['n_boot']} resamples of canonical problems). Accuracies in %, differences in pp.", "",
         f"Sample status: Exp 1C {out['n_1C']} of {out['planned']['1C']} items ({out['status']['1C']}); sampled and mask "
         f"robustness {out['n_subset']} of {out['planned']['subset']} ({out['status']['subset']}); min_n {MIN_N} "
         "(configs/blocks.yaml)."]
    gi = out["greedy_identity"]
    L.append("Greedy generations identical to the main run for the same prompt: " +
             ("not checked (generations files absent)." if gi is None else f"{gi['identical']} of {gi['compared']}."))
    scopes = [("all items", out["all"])]
    if out["sensitivity"].get("same_as_all"):
        L.append("Sensitivity (without items whose a0 tokens differ across channels): no item is flagged, so it equals "
                 "the main analysis.")
    else:
        scopes.append((f"sensitivity: without the {out['flagged']} flagged item(s)", out["sensitivity"]))
    for scope, res in scopes:
        L += ["", f"## Exp 1C (H1c) — {scope}, n = {res['n_1C']}", ""]
        if "1C" in res:
            r = res["1C"]
            L += ["| Cell | Acc [95 % CI] | W->R | R->W |", "|---|---|---|---|"]
            L += [f"| {c} | {ci_pct(v['acc'])} | {ci_pct(v['W->R'])} | {ci_pct(v['R->W'])} |" for c, v in r["cells"].items()]
            L += ["", "| Contrast | Estimate [95 % CI] |", "|---|---|",
                  f"| Range of accuracy across the 3 channels at F3(0.9) | {ci_pp(r['range at F3(0.9)'])} |"]
            L += [f"| {k} at F3(0.9) | {ci_pp(v)} |" for k, v in r["pairwise"].items()]
            L += [f"| F3(0.9) − F3(0.7), plain user | {ci_pp(r['F3(0.9) - F3(0.7), user'])} |",
                  f"| Range − (F3(0.9) − F3(0.7)) | {ci_pp(r['range - (F3(0.9) - F3(0.7))'])} |", "",
                  "The range is ≥ 0 by construction and biased upward by noise; read it with the pairwise differences."]
        L += ["", f"## Sampled robustness — {scope}, n = {res['n_subset']} (3 seeds, T 0.7, top-p 0.8; item = seed mean)", ""]
        if "sampled" in res:
            s = res["sampled"]
            L += ["| Condition | Acc [95 % CI] | W->R | R->W | Greedy acc | Sampled − greedy [95 % CI] |",
                  "|---|---|---|---|---|---|"]
            L += [f"| {c} | {ci_pct(v['acc'])} | {ci_pct(v['W->R'])} | {ci_pct(v['R->W'])} | {ci_pct(v['greedy']['acc'])} | "
                  f"{ci_pp(v['sampled - greedy'])} |" for c, v in s["cells"].items()]
            L += ["", "| Estimand | Sampled [95 % CI] | Greedy, same items [95 % CI] | Sampled − greedy [95 % CI] |",
                  "|---|---|---|---|"]
            L += [f"| {k} | {ci_pp(v['sampled'])} | {ci_pp(v['greedy'])} | {ci_pp(v['sampled - greedy'])} |"
                  for k, v in s["analogues"].items()]
            m = res["mask"]
            L += ["", f"## Verifier-mask robustness — {scope}, n = {res['n_subset']} (greedy)", "",
                  "| Verifier seed | " + " | ".join(f"Acc F3({p})" for p in P_LEVELS) + " | H1b analogue [95 % CI] | "
                  "Minus seed 0 [95 % CI] |", "|---" * (len(P_LEVELS) + 3) + "|"]
            for k in MASK_SEEDS:
                v = m[f"seed {k}"]
                L.append(f"| {k} | " + " | ".join(pct(v["acc"][f"F3({p})"][0]) for p in P_LEVELS) +
                         f" | {ci_pp(v['H1b analogue'])} | {ci_pp(v.get(f'seed {k} - seed 0'))} |")
            L.append(f"| main run (seed 0) | " + " | ".join("–" for _ in P_LEVELS) +
                     f" | {ci_pp(m['main run (seed 0)']['H1b analogue'])} | – |")
    L += ["", "## Output status (all items)", "", "| Cell | n | ok | truncated | unparsed | other | Mean out tokens |",
          "|---|---|---|---|---|---|---|"]
    for k, st in out["status_counts"].items():
        n = sum(st.values())
        other = n - st.get("ok", 0) - st.get("truncated", 0) - st.get("unparsed", 0) - st.get("fixed", 0)
        L.append(f"| {k} | {n} | {st.get('ok', 0) + st.get('fixed', 0)} | {st.get('truncated', 0)} | "
                 f"{st.get('unparsed', 0)} | {other} | {out['mean_out_tokens'].get(k, float('nan')):.0f} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--json-dir", default=str(ROOT / "analysis" / "out"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    canon = {r["item_id"]: r["canonical_id"] for r in jl("data/processed/dnat.jsonl")}
    rows, flags, commit, planned = load_runs(a.runs)
    c1, sampled, mask, c0, status, toks = index(rows)
    ids1 = complete_ids(c1, C1, "Exp 1C") if c1 else []
    ids_s = complete_ids(sampled, SAMPLED, "sampled") if sampled else []
    for c in SAMPLED:
        bad = [i for i in ids_s if sorted(sampled[c][i]) != list(range(N_SAMPLES))]
        if bad:
            raise SystemExit(f"sampled {c}: item {bad[0]} lacks a seed")
    if ids_s:
        ids_m = complete_ids(mask, [(k, f"F3({p})") for k in MASK_SEEDS for p in P_LEVELS], "mask")
        if ids_m != ids_s:
            raise SystemExit("the mask-robustness items differ from the sampled-robustness items")
    main_rows = load_main(a.main, set(ids_s))
    if ids_s:
        complete_ids(main_rows, SAMPLED, "main run")
    keep = {i for i in set(ids1) | set(ids_s) if flags.get(i, True)}
    flagged = sorted(i for i in flags if not flags[i])
    out = {"runs": a.runs, "main": a.main, "code_commit": commit, "n_boot": a.n_boot, "planned": planned,
           "n_1C": len(ids1), "n_subset": len(ids_s),
           "status": {"1C": status_of(len(ids1), planned["1C"]), "subset": status_of(len(ids_s), planned["subset"])},
           "flagged": len(flagged), "flagged_items": flagged, "greedy_identity": greedy_identity(a.runs, a.main),
           "all": analyse(c1, sampled, mask, c0, main_rows, canon, ids1, ids_s, a.n_boot),
           "sensitivity": (analyse(c1, sampled, mask, c0, main_rows, canon, [i for i in ids1 if i in keep],
                                   [i for i in ids_s if i in keep], a.n_boot) if flagged else {"same_as_all": True}),
           "status_counts": {k: dict(v) for k, v in sorted(status.items())},
           "mean_out_tokens": {k: float(np.mean(v)) for k, v in sorted(toks.items())}}
    Path(a.json_dir).mkdir(parents=True, exist_ok=True)
    Path(a.reports_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.json_dir) / f"{a.out}.json").write_text(json.dumps(out, indent=1))
    L = report(out, a)
    (Path(a.reports_dir) / f"04_{a.out}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
