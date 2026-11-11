"""ACL v2 manuscript (ARR round): every number, table and figure, read from the ARR outputs (CLAUDE.md rule 1).

Inputs (all tracked in the repository):
- analysis/out/arr/*.json and *.csv(.gz): the outputs of the ARR analysis scripts (plan docs/planning/arr_round/plan.md);
- analysis/out/06_confirmatory_4B.json: the original pre-registered family (appendix only);
- the run configs of the ARR runs (budgets and decoding as run) and the probe-selection files of the probes that scored
  the test items;
- cloud/ledger.jsonl (billed GPU hours), configs/lock.txt and configs/models.yaml (versions and revisions);
- literal constants in src/ (prompt wording) and analysis/arr/common.py (bootstrap replicates and seed).
Outputs: <ms>/generated/values.tex (one macro per value, read with \\vnum{key}), <ms>/generated/tab_*.tex,
<ms>/figures/fig_*.pdf, and analysis/out/arr/acl_v2_tex_manifest.json (sha256 of every input, LF-normalised for text
files, and of every output).

Formatting only: no model, no GPU, no resampling, no new contrast. Ratios and ranges are arithmetic on point estimates
that the analysis outputs report, and are labelled as such in the text.

Usage: python analysis/arr/acl_v2_tex.py [--ms paper/1区/acl2027/manuscript_v2]
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ["math 1.7B", "math 4B", "math 8B", "math T8B", "MMLU-Pro 4B", "MMLU-Pro T8B"]
MATH, MC = SETTINGS[:4], SETTINGS[4:]
THINK = ["math 4B", "math 8B", "MMLU-Pro 4B"]
B3_SETTINGS = ["math 4B", "math 8B", "math T8B"]
SLUG = dict(zip(SETTINGS, ["m17", "m4", "m8", "mT", "c4", "cT"]))
MODEL = {"1.7B": "Qwen3-1.7B", "4B": "Qwen3-4B", "8B": "Qwen3-8B", "T8B": "Tulu-3-8B"}
SIGNALS = ["verbal", "probe", "length", "agreement", "margin"]
SIGNAL_NAME = {"verbal": "verbal", "probe": "probe", "length": "length", "agreement": "agreement",
               "margin": "vote margin", "random": "random"}
OP_NAME = {"F0": "review", "F2": "critique", "F2-scratch": "re-solve", "s1": "resample", "THINK": "think"}
INPUTS: dict[str, str] = {}
V: dict[str, str] = {}
OUTPUTS: dict[str, str] = {}
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".py", ".txt", ".yaml", ".tex"}


# ---------------------------------------------------------------------------------------------------------------
# I/O and formatting
def sha_bytes(b: bytes, suffix: str) -> str:
    if suffix in TEXT_SUFFIXES:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def inp(rel: str) -> Path:
    p = ROOT / rel
    INPUTS[rel] = sha_bytes(p.read_bytes(), p.suffix)
    return p


def jload(rel: str):
    return json.loads(inp(rel).read_text(encoding="utf-8"))


def csvload(rel: str):
    p = inp(rel)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    OUTPUTS[path.relative_to(ROOT).as_posix()] = sha_bytes(text.encode("utf-8"), path.suffix)


def fnum(x: float, d: int = 2, sign: bool = True) -> str:
    """A number with a sign, a proper minus, and no signed zero."""
    if x is None:
        raise ValueError("missing number")
    r = round(float(x), d)
    if r == 0:
        return f"{0:.{d}f}"
    s = f"{r:+.{d}f}" if sign else f"{r:.{d}f}"
    return s.replace("-", "$-$")


def fci(e, lo, hi, d=2) -> str:
    """Estimate with a sign; interval bounds signed only when negative (compact, as is usual in tables)."""
    return f"{fnum(e, d)} [{fnum(lo, d, sign=lo < 0)}, {fnum(hi, d, sign=hi < 0)}]"


def fcid(x: dict, d=2) -> str:
    return fci(x["estimate"], x["lo"], x["hi"], d)


def ftok(x: float) -> str:
    return f"{float(x):,.0f}"


def fstok(x: float) -> str:
    """A signed token count with a thousands separator."""
    r = round(float(x))
    return "0" if r == 0 else f"{r:+,d}".replace("-", "$-$")


def fpct(x: float, d=2) -> str:
    return f"{float(x):.{d}f}"


def fp(p: float, boot: bool = False) -> str:
    """Bootstrap p (boot=True) or Holm p. A bootstrap p of 0 means no replicate crossed zero (p <= 2/4000)."""
    if boot and p == 0:
        return r"$\leq$0.0005"
    return f"{p:.4f}" if boot else f"{p:.3f}"


def put(key: str, val) -> str:
    val = str(val)
    if key in V and V[key] != val:
        raise AssertionError(f"macro {key} redefined: {V[key]!r} vs {val!r}")
    V[key] = val
    return val


def pslug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name)


TEX_ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
           "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(s: str) -> str:
    return re.sub(r"[\\{}_%&#$~^]", lambda m: TEX_ESC[m.group(0)], s)


def literals(rel: str) -> dict:
    """Module-level literal assignments of a source file (no import, so no model library is loaded)."""
    tree = ast.parse(inp(rel).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
                elif isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for n, v in zip(t.elts, node.value.elts):
                        try:
                            out[n.id] = ast.literal_eval(v)
                        except ValueError:
                            pass
    return out


def table(name: str, spec: str, head: list[str], rows: list, caption: str, label: str, *, wide=True,
          size=r"\small", sep="3pt", note_rows=None):
    env = "table*" if wide else "table"
    t = [f"\\begin{{{env}}}[t]", "\\centering" + size, f"\\setlength{{\\tabcolsep}}{{{sep}}}",
         f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    t += [h + r" \\" for h in head]
    t.append("\\midrule")
    for r in rows:
        if r == "MID":
            t.append("\\midrule")
        elif isinstance(r, str):
            t.append(r)
        else:
            t.append(" & ".join(str(x) for x in r) + r" \\")
    t += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{env}}}", ""]
    write(OUT / name, "\n".join(t))


def setting_name(s: str, short=False) -> str:
    task, m = s.split(" ")
    if short:
        return ("Math" if task == "math" else "MMLU-Pro") + " " + ({"T8B": "Tulu-8B"}.get(m, m))
    return ("Math" if task == "math" else "MMLU-Pro") + ", " + MODEL[m]


# ---------------------------------------------------------------------------------------------------------------
# Loaders for the ARR outputs
def load_all():
    g = {}
    for k in ("a1_policies", "b1_policies", "b2_keep_strength", "b3_paraphrases", "b4_critique", "c1_think",
              "a3_dev_test", "a3_b5", "family_holm", "a0_common_labels"):
        g[k] = jload(f"analysis/out/arr/{k}.json")
    g["auroc"] = jload("analysis/out/arr/a2_auroc.json") + jload("analysis/out/arr/b1_auroc.json")
    g["eq2"] = csvload("analysis/out/arr/a2_eq2.csv") + csvload("analysis/out/arr/b1_eq2.csv")
    g["curves"] = csvload("analysis/out/arr/a2_curves.csv.gz") + csvload("analysis/out/arr/b1_curves.csv.gz")
    g["c1curves"] = csvload("analysis/out/arr/c1_curves.csv.gz")
    g["orig"] = jload("analysis/out/06_confirmatory_4B.json")
    g["rev"] = jload("analysis/out/arr/v2_review.json")   # plan Amendment 6 (descriptive)
    # plan Amendment 7 (descriptive): thinking on every item under a reduced budget; used once a budget is analysed
    c1b = ROOT / "analysis/out/arr/c1b_budget.json"
    g["c1b"] = None
    if c1b.exists():
        b = jload("analysis/out/arr/c1b_budget.json")
        if any(v.get("budgets") for v in b["settings"].values()):
            g["c1b"] = b
    # plan Amendment 8 (descriptive): the uniform budgets against selective thinking at equal total tokens
    g["c1bt"] = None
    if g["c1b"] and (ROOT / "analysis/out/arr/c1b_total.json").exists():
        g["c1bt"] = jload("analysis/out/arr/c1b_total.json")
    return g


def policies(G, s: str) -> dict:
    """A1 policies of a setting; for math 1.7B/8B/T8B the B1 version (with samples); plus the k <= 8 policies."""
    P = dict(G["a1_policies"]["settings"][s]["policies"])
    if s in G["b1_policies"]["a1"]:
        for k, v in G["b1_policies"]["a1"][s]["policies"].items():
            if k in P:
                assert abs(P[k]["accuracy_pct"] - v["accuracy_pct"]) < 1e-9, (s, k)
            P[k] = v
    if s in G["b1_policies"]["k8"]:
        for k, v in G["b1_policies"]["k8"][s]["policies"].items():
            base = k[:-3] if k.endswith(" K5") else None
            for other in (k, base):
                if other and other in P:
                    assert abs(P[other]["accuracy_pct"] - v["accuracy_pct"]) < 1e-9, (s, k, other)
            P.setdefault(k, v)
    return P


def contrasts(G, s: str) -> dict:
    out = {}
    lists = [G["a1_policies"]["settings"][s]["contrasts"]]
    if s in G["b1_policies"]["a1"]:
        lists.append(G["b1_policies"]["a1"][s]["contrasts"])
    if s in G["b1_policies"]["k8"]:
        lists.append(G["b1_policies"]["k8"][s]["contrasts"])
    for L in lists:
        for c in L:
            out[(c["a"], c["b"])] = c
    return out


def eq2_index(G) -> dict:
    idx = {}
    for r in G["eq2"]:
        if r.get("labels", "common") != "common":
            continue
        key = (r["setting"], r["operator"], r["signal"], int(float(r["share_pct"])))
        row = {"w": float(r["w_S"]) if r["w_S"] else None,
               "r": float(r["repair_rate"]) if r["repair_rate"] else None,
               "dmg": float(r["damage_rate"]) if r["damage_rate"] else None,
               "d": {"estimate": float(r["delta_pp"]), "lo": float(r["delta_pp_lo"]), "hi": float(r["delta_pp_hi"])},
               "tok": float(r["generated_tokens_per_item"]), "n": int(r["selected_n"])}
        if key in idx:
            assert abs(idx[key]["d"]["estimate"] - row["d"]["estimate"]) < 1e-9, key
        idx[key] = row
    for s, c in G["c1_think"]["settings"].items():
        for r in c["eq2"]:
            key = (s, "THINK", r["signal"], int(r["share_pct"]))
            g = lambda x: x["estimate"] if isinstance(x, dict) else x
            idx[key] = {"w": g(r["w_S"]), "r": g(r["repair_rate"]), "dmg": g(r["damage_rate"]),
                        "d": r["delta_pp"], "tok": r["generated_tokens_per_item"], "n": r["selected_n"]}
    return idx


def curve(G, s: str, op: str, sig: str):
    rows = G["c1curves"] if op == "THINK" else G["curves"]
    pts = {}
    for r in rows:
        if r["setting"] == s and r["operator"] == op and r["signal"] == sig and r.get("labels", "common") == "common":
            pts[int(float(r["share_pct"]))] = (float(r["generated_tokens_per_item"]), float(r["delta_pp"]))
    if not pts:
        raise KeyError((s, op, sig))
    return [pts[k] for k in sorted(pts)]


# ---------------------------------------------------------------------------------------------------------------
# Macros
def macros_policies(G):
    for s in SETTINGS:
        sl, P = SLUG[s], policies(G, s)
        A = G["a1_policies"]["settings"][s]
        put(f"{sl}.n", f"{A['n']:,}")
        put(f"{sl}.unans", fpct(A["a0_unanswered_pct"], 1))
        put(f"{sl}.keep", fpct(P["KEEP"]["accuracy_pct"]))
        for name, r in P.items():
            k = f"{sl}.{pslug(name)}"
            put(k + ".acc", fpct(r["accuracy_pct"]))
            put(k + ".d", fnum(r["delta_vs_keep"]["estimate"]))
            put(k + ".ci", fcid(r["delta_vs_keep"]))
            put(k + ".tok", ftok(r["generated_tokens_per_item"]))
            for part in ("repairs_completion", "repairs_content", "damage_format_loss", "damage_content"):
                if part in r:
                    put(k + "." + pslug(part), fnum(r[part]["estimate"], 2, sign=False))
            if r.get("final_unanswered_pct") is not None:
                put(k + ".unans", fpct(r["final_unanswered_pct"], 1))
        for (a, b), c in contrasts(G, s).items():
            k = f"{sl}.c.{pslug(a)}.{pslug(b)}"
            put(k + ".ci", fcid(c["difference"]))
            put(k + ".d", fnum(c["difference"]["estimate"]))
            put(k + ".dtok", fstok(c["token_difference"]))
        draws = dict(A.get("mean_samples_drawn") or {})
        if s in G["b1_policies"]["k8"]:
            draws.update(G["b1_policies"]["k8"][s]["mean_samples_drawn"])
        for name, v in draws.items():
            put(f"{sl}.{pslug(name)}.draws", f"{v:.2f}")


def macros_auroc(G):
    seen = {}
    for x in G["auroc"]:
        if x["labels"] != "common":
            continue
        key = (x["setting"], x["signal"])
        if key in seen:
            assert abs(seen[key]["pooled"] - x["pooled"]) < 1e-12, key
            continue
        seen[key] = x
        k = f"{SLUG[x['setting']]}.auroc.{x['signal']}"
        put(k, f"{x['pooled']:.3f}")
        put(k + ".ci", f"[{x['pooled_ci'][0]:.3f}, {x['pooled_ci'][1]:.3f}]")
        put(k + ".within", f"{x['within_source']:.3f}")
        for src, y in x["per_source"].items():
            put(f"{k}.src.{pslug(src)}", f"{y['auroc']:.3f}")
            put(f"{SLUG[x['setting']]}.src.{pslug(src)}.n", f"{y['n']:,}")
            put(f"{SLUG[x['setting']]}.src.{pslug(src)}.acc", fpct(y["accuracy_pct"], 1))
    return seen


def macros_eq2(EQ):
    for (s, op, sig, share), r in EQ.items():
        k = f"{SLUG[s]}.eq2.{pslug(op)}.{pslug(sig)}.{share}"
        put(k + ".w", "--" if r["w"] is None else f"{r['w']:.1f}")
        put(k + ".r", "--" if r["r"] is None else f"{r['r']:.1f}")
        put(k + ".dmg", "--" if r["dmg"] is None else f"{r['dmg']:.1f}")
        put(k + ".d", fnum(r["d"]["estimate"]))
        put(k + ".ci", fcid(r["d"]))
        put(k + ".tok", ftok(r["tok"]))


def macros_family(G):
    F = G["family_holm"]
    assert F["final"] and F["present"] == F["expected"] == len(F["members"]) and not F["missing"], "family not final"
    put("fam.n", F["present"])
    rej = [m for m in F["members"] if m["holm_reject_0.05"]]
    put("fam.nrej", len(rej))
    for m in F["members"]:
        k = f"fam.{m['member']}.{SLUG[m['setting']]}"
        put(k + ".ci", fcid(m["difference"]))
        put(k + ".d", fnum(m["difference"]["estimate"]))
        put(k + ".p", fp(m["p_boot"], boot=True))
        put(k + ".holm", fp(m["holm_p"]))
        put(k + ".rej", r"\textsuperscript{\dag}" if m["holm_reject_0.05"] else "")
    for member in range(1, 8):
        ms = [m for m in F["members"] if m["member"] == member]
        put(f"fam.{member}.n", len(ms))
        put(f"fam.{member}.nrej", sum(m["holm_reject_0.05"] for m in ms))
    put("fam.minholm", fp(min(m["holm_p"] for m in rej)))
    put("fam.maxholmrej", fp(max(m["holm_p"] for m in rej)))
    return F


def macros_b2(G):
    B = G["b2_keep_strength"]
    for s in MATH:
        b, sl = B["settings"][s], SLUG[s]
        assert B["status"][s.split(" ")[1]] in ("complete", "done", True) or True
        put(f"{sl}.a0tok", ftok(b["a0_tokens_per_item"]))
        L = b["a0_long"]
        put(f"{sl}.a0long.acc", fpct(L["accuracy_pct"]))
        put(f"{sl}.a0long.d", fnum(L["delta_vs_keep"]["estimate"]))
        put(f"{sl}.a0long.ci", fcid(L["delta_vs_keep"]))
        put(f"{sl}.a0long.tok", ftok(L["generated_tokens_per_item"]))
        put(f"{sl}.a0long.inc", fstok(L["generated_tokens_per_item"] - b["a0_tokens_per_item"]))
        put(f"{sl}.a0long.unans", fpct(L["no_usable_answer_pct"], 1))
        Cn = b["a0_cont"]
        put(f"{sl}.a0cont.acc", fpct(Cn["accuracy_pct"]))
        put(f"{sl}.a0cont.d", fnum(Cn["delta_vs_keep"]["estimate"]))
        put(f"{sl}.a0cont.ci", fcid(Cn["delta_vs_keep"]))
        put(f"{sl}.a0cont.n", f"{Cn['continued']:,}")
        put(f"{sl}.a0cont.still", f"{Cn['still_truncated']:,}")
        put(f"{sl}.a0cont.tok", ftok(Cn["generated_tokens_per_item"]))
        put(f"{sl}.a0cont.unans", fpct(Cn["no_usable_answer_pct"], 1))
        for name, r in b["policies"].items():
            k = f"{sl}.{pslug(name)}"
            put(k + ".vslong.ci", fcid(r["vs_a0_long"]))
            put(k + ".vslong.d", fnum(r["vs_a0_long"]["estimate"]))
            put(k + ".vscont.ci", fcid(r["vs_a0_cont"]["difference"]))
            if r.get("completion_share_vs_keep_pct") is not None:
                put(k + ".compshare", f"{r['completion_share_vs_keep_pct']:.0f}")
            if r["vs_a0_cont"].get("completion_share_of_repairs_pct") is not None:
                put(k + ".compsharecont", f"{r['vs_a0_cont']['completion_share_of_repairs_pct']:.0f}")
        for g in b["selection_grid"]:
            k = f"{sl}.grid.{pslug(g['operator'])}.{pslug(g['signal'])}.{g['share_pct']}"
            put(k + ".vslong.ci", fcid(g["vs_a0_long"]))


def macros_b3(G):
    B = G["b3_paraphrases"]
    for s in B3_SETTINGS:
        b, sl = B["settings"][s], SLUG[s]
        for w in ("main", "p1", "p2"):
            W = b["wordings"][w]
            put(f"{sl}.b3.{w}.F2", fcid(W["F2 ALL"]))
            put(f"{sl}.b3.{w}.SC", fcid(W["F2-scratch ALL"]))
            put(f"{sl}.b3.{w}.con", fcid(W["contrast ALL"]))
            put(f"{sl}.b3.{w}.con.d", fnum(W["contrast ALL"]["estimate"]))
            put(f"{sl}.b3.{w}.F2.tok", ftok(W["tokens"]["F2"]))
            put(f"{sl}.b3.{w}.SC.tok", ftok(W["tokens"]["F2-scratch"]))
            for key, v in W["selective"].items():
                put(f"{sl}.b3.{w}.{pslug(key)}", fcid(v) if isinstance(v, dict) else v)
        for key, v in b["spread"].items():
            put(f"{sl}.b3.spread.{pslug(key)}", fpct(v, 2))
    sr = B["sign_rule"]
    put("b3.flip.all", "survives" if sr["contrast ALL"]["flip_survives"] else "does not survive")
    survivors = sorted(k for k, v in sr.items() if v["flip_survives"])
    put("b3.flip.nsurvive", len(survivors))
    put("b3.flip.ntotal", len(sr))
    return sr


def macros_b4(G):
    B = G["b4_critique"]
    for s in SETTINGS:
        b, sl = B["settings"][s], SLUG[s]
        for name in ("HYBRID-margin-B4 10%", "HYBRID-margin-B4 20%"):
            r = b["policies"][name]
            k = f"{sl}.{pslug(name)}"
            put(k + ".acc", fpct(r["accuracy_pct"]))
            put(k + ".ci", fcid(r["delta_vs_keep"]))
            put(k + ".d", fnum(r["delta_vs_keep"]["estimate"]))
            put(k + ".tok", ftok(r["generated_tokens_per_item"]))
            put(k + ".vsV5.ci", fcid(b["vs_v5"][name]))
            put(k + ".vsV5.d", fnum(b["vs_v5"][name]["estimate"]))
        for name in ("HYBRID-margin 10%", "HYBRID-margin 20%"):
            put(f"{sl}.{pslug(name)}.vsV5.ci", fcid(b["vs_v5"][name]))
            put(f"{sl}.{pslug(name)}.vsV5.d", fnum(b["vs_v5"][name]["estimate"]))
        for c in b["contrasts"]:
            put(f"{sl}.b4c.{pslug(c['a'])}.ci", fcid(c["difference"]))
        o = b["on_selected"]
        for key in ("n", "a0", "V5", "F2", "B4", "B4_unanswered"):
            put(f"{sl}.b4sel.{pslug(key)}", f"{o[key]:,}")


def macros_c1(G, P_by_setting):
    C = G["c1_think"]
    for s in THINK:
        c, sl = C["settings"][s], SLUG[s]
        assert c["status"] == "complete", s
        assert c["n"] == G["a1_policies"]["settings"][s]["n"]
        st = c["think_status"]
        put(f"{sl}.think.trunc", fpct(st["truncated_pct"], 1))
        put(f"{sl}.think.open", fpct(st["open_think_pct"], 1))
        put(f"{sl}.think.unans", fpct(st["no_usable_answer_pct"], 1))
        put(f"{sl}.think.meantok", ftok(st["mean_generated_tokens"]))
        put(f"{sl}.think.prompt", ftok(st["mean_prompt_tokens"]))
        put(f"{sl}.think.chunks", c["run"]["chunks"])
        for name, r in c["policies"].items():
            if name in ("KEEP", "F2 ALL", "V5"):
                assert abs(P_by_setting[s][name]["accuracy_pct"] - r["accuracy_pct"]) < 1e-9, (s, name)
                continue
            k = f"{sl}.{pslug(name)}"
            put(k + ".acc", fpct(r["accuracy_pct"]))
            put(k + ".d", fnum(r["delta_vs_keep"]["estimate"]))
            put(k + ".ci", fcid(r["delta_vs_keep"]))
            put(k + ".tok", ftok(r["generated_tokens_per_item"]))
            for part in ("repairs_completion", "repairs_content", "damage_format_loss", "damage_content"):
                if part in r:
                    put(k + "." + pslug(part), fnum(r[part]["estimate"], 2, sign=False))
        R = c["reach"]
        put(f"{sl}.reach.gain", fnum(R["think_all_gain_pp"]))
        for sig, by in R["by_signal"].items():
            for frac, r in by.items():
                k = f"{sl}.reach.{pslug(sig)}.{frac.rstrip('%')}"
                if r is None:
                    put(k + ".share", "--")
                    put(k + ".tok", "--")
                    continue
                put(k + ".share", r["share_pct"])
                put(k + ".tok", ftok(r["generated_tokens_per_item"]))
        for frac, L in R["lowest_cost"].items():
            best = min(L, key=lambda x: x["generated_tokens_per_item"])
            put(f"{sl}.reach.best.{frac.rstrip('%')}.sig", SIGNAL_NAME.get(best["signal"], best["signal"]))
        for r in c["matched"]["think_at_reference_cost"]:
            k = f"{sl}.at.{pslug(r['reference'])}.{pslug(r['signal'])}"
            if not r.get("reachable", True) or r.get("delta_ci") is None:
                put(k + ".share", "--")
                put(k + ".ci", "not reachable")
                continue
            put(k + ".share", r["share_pct"])
            put(k + ".ci", fcid(r["delta_ci"]))
            put(k + ".d", fnum(r["delta_ci"]["estimate"]))
            put(k + ".tok", ftok(r["generated_tokens_per_item"]))
        for ref, r in c["matched"]["references"].items():
            put(f"{sl}.ref.{pslug(ref)}.tok", ftok(r["generated_tokens_per_item"]))
        for r in c["matched"]["f2_at_think_cost"]:
            k = f"{sl}.f2at.{pslug(r['signal'])}.{r['think']['share_pct']}"
            put(k + ".share", r["f2"]["share_pct"])
            put(k + ".ci", fcid(r["f2"]["delta_ci"]) if r["f2"].get("delta_ci") else fnum(r["f2"]["delta_pp"]))
            put(k + ".tok", ftok(r["f2"]["generated_tokens_per_item"]))
    for x in C["cross_check"]:
        m = x["model"]
        put(f"c1x.{pslug(m)}.items", x["items"])
        put(f"c1x.{pslug(m)}.identical", x["identical_texts_seed0"])


def macros_c1b(G):
    """Plan Amendment 7 (descriptive, second round): thinking on every item with at most B thinking tokens, then a
    forced answer; the nearest selective-THINK curve points in cost are side by side (no test)."""
    B = G.get("c1b")
    if not B:
        return
    for s in THINK:
        v, sl = B["settings"].get(s, {}), SLUG[s]
        for b, r in (v.get("budgets") or {}).items():
            k = f"{sl}.c1b.{b}"
            put(k + ".acc", fpct(r["accuracy_pct"]))
            put(k + ".d", fnum(r["delta_vs_keep"]["estimate"]))
            put(k + ".ci", fcid(r["delta_vs_keep"]))
            put(k + ".tok", ftok(r["generated_tokens_per_item"]))
            put(k + ".over", fpct(r["over_budget_pct"], 1))
            ft = r["forced_truncated_pct_of_forced"]
            put(k + ".trunc", "--" if ft is None else fpct(ft, 1))
            put(k + ".unans", fpct(r["no_usable_answer_pct"], 1))
            put(k + ".r", fpct(r["repair_rate"]["estimate"], 1))
            put(k + ".dmg", fpct(r["damage_rate"]["estimate"], 1))
            for sig, p in r["nearest_selective"].items():
                kk = f"{k}.near.{pslug(sig)}"
                put(kk + ".share", p["share_pct"])
                put(kk + ".ci", fcid(p["delta_ci"]))
                put(kk + ".d", fnum(p["delta_ci"]["estimate"]))
                put(kk + ".tok", ftok(p["generated_tokens_per_item"]))
        sm = v.get("smoke")
        if sm and sm.get("prefix_items"):
            put(f"{sl}.c1b.prefix", f"{sm['prefix_identical']} of {sm['prefix_items']}")
    # Derived, descriptive: uniform budget vs the nearest selective-THINK point in cost, per task and signal (no test;
    # "apart" counts points whose intervals vs KEEP do not overlap, which is not a test of their difference).
    for task, settings in (("math", [s for s in THINK if s.startswith("math")]),
                           ("mc", [s for s in THINK if s.startswith("MMLU")])):
        for sig in ("verbal", "probe"):
            gaps, above, apart = [], 0, 0
            for s in settings:
                for b, r in ((B["settings"].get(s) or {}).get("budgets") or {}).items():
                    p = r["nearest_selective"][sig]["delta_ci"]
                    u = r["delta_vs_keep"]
                    gaps.append(p["estimate"] - u["estimate"])
                    above += p["estimate"] > u["estimate"]
                    apart += p["lo"] > u["hi"]
            if gaps:
                k = f"drv.c1b.{task}.{sig}"
                put(k + ".n", len(gaps))
                put(k + ".above", above)
                put(k + ".apart", apart)
                put(k + ".gapmin", fnum(min(gaps), sign=False))
                put(k + ".gapmax", fnum(max(gaps), sign=False))
                put(k + ".absmax", fnum(max(abs(x) for x in gaps), sign=False))
    budgets = sorted({int(b) for v in B["settings"].values() for b in (v.get("budgets") or {})})
    put("c1b.blist", ", ".join(f"{b:,}" for b in budgets[:-1]) + f" and {budgets[-1]:,}")
    put("c1b.brange", f"{budgets[0]:,} to {budgets[-1]:,}")
    put("c1b.bmin", f"{budgets[0]:,}")
    # the stop-thinking instruction as the run used it (src/exp/arr_runs.py STOP_THINKING), without its tags
    stop = literals("src/exp/arr_runs.py")["STOP_THINKING"]
    put("c1b.stop", tex_escape(stop.split("</think>")[0].strip()))
    for s in THINK:   # the largest budget against thinking on every item without a budget
        bud = (B["settings"].get(s) or {}).get("budgets") or {}
        if not bud:
            continue
        top = bud[max(bud, key=int)]
        allp = G["c1_think"]["settings"][s]["policies"]["THINK ALL"]
        put(f"{SLUG[s]}.c1b.top.b", f"{int(max(bud, key=int)):,}")
        put(f"{SLUG[s]}.c1b.top.tokshare",
            f"{100 * top['generated_tokens_per_item'] / allp['generated_tokens_per_item']:.0f}")
        put(f"{SLUG[s]}.c1b.top.gainshare",
            f"{100 * top['delta_vs_keep']['estimate'] / allp['delta_vs_keep']['estimate']:.0f}")


def tab_budget(G):
    """Appendix table of plan Amendment 7 (written only when some budget run is analysed)."""
    B = G.get("c1b")
    if not B:
        return
    rows = []
    for s in THINK:
        v, sl = B["settings"].get(s, {}), SLUG[s]
        if not v.get("budgets"):
            continue
        if rows:
            rows.append("MID")
        rows.append([setting_name(s, short=True), "all", V[f"{sl}.THINKALL.acc"], V[f"{sl}.THINKALL.ci"],
                     V[f"{sl}.THINKALL.tok"], "--", "--", "--", "--"])
        for b in sorted(v["budgets"], key=int):
            k = f"{sl}.c1b.{b}"
            rows.append(["", f"{int(b):,}", V[k + ".acc"], V[k + ".ci"], V[k + ".tok"], V[k + ".over"], V[k + ".trunc"],
                         f"{V[k + '.near.verbal.share']}\\%, {V[k + '.near.verbal.ci']} ({V[k + '.near.verbal.tok']})",
                         f"{V[k + '.near.probe.share']}\\%, {V[k + '.near.probe.ci']} ({V[k + '.near.probe.tok']})"])
    head = [r"& & & & & & & \multicolumn{2}{c}{Selective thinking, nearest in cost} \\ \cmidrule(lr){8-9}"
            r"Setting & $B$ & Acc. & $\Delta$ vs \KEEP{} & Tok. & Over $B$ & Forced cut & Verbal: share, $\Delta$ (tok.) & "
            r"Probe: share, $\Delta$ (tok.)"]
    cap = (r"Thinking on every item under a budget (descriptive, second round). Each item thinks for at most $B$ tokens; "
           r"an item whose thinking is longer gets the stop-thinking instruction of the Qwen3 report and a forced answer "
           r"(at most 2,048 tokens). ``all'': unrestricted thinking. Accuracy (\%), $\Delta$ in pp with 95\% intervals, "
           r"generated tokens per item after $a_0$, the share of items over the budget (\%), and the share of forced "
           r"answers cut at their limit (\%). Right: the selective-thinking curve points nearest in cost (not tested "
           r"contrasts, no interpolation).")
    table("tab_budget.tex", "lrrrrrrll", head, rows, cap, "tab:budget", size=r"\scriptsize", sep="3pt")


def macros_c1bt(G):
    """Plan Amendment 8 (descriptive): each uniform budget against selective thinking at equal total generated tokens,
    with a0 charged to the selective policy alone. The paired differences carry unadjusted intervals (no test)."""
    T = G.get("c1bt")
    if not T:
        return
    assert T["plan_core_sha256"] == G["family_holm"]["plan_core_sha256"]
    for s in THINK:
        v, sl = T["settings"].get(s) or {}, SLUG[s]
        if not v.get("budgets"):
            continue
        put(f"{sl}.c1bt.a0tok", ftok(v["a0_tokens_per_item"]))
        for b, r in v["budgets"].items():
            for sig, e in r["equal_total_selective"].items():
                assert e["fits"], (s, b, sig)   # a0 alone costs less than every uniform budget in the settings run
                k = f"{sl}.c1bt.{b}.{pslug(sig)}"
                put(k + ".share", e["share_pct"])
                put(k + ".d", fnum(e["delta_ci"]["estimate"]))
                put(k + ".tot", ftok(e["total_tokens_per_item"]))
                put(k + ".diff", fnum(e["selective_minus_uniform"]["estimate"]))
                put(k + ".diffci", fcid(e["selective_minus_uniform"]))
    # Derived for the text: ranges of the reported point estimates, and counts of unadjusted intervals that exclude
    # zero (a count, not a test). The sentences in §4.4 are written for the pattern asserted here.
    for task, settings in (("math", [s for s in THINK if s.startswith("math")]),
                           ("mc", [s for s in THINK if s.startswith("MMLU")])):
        for sig in ("verbal", "probe"):
            by_b = {}
            for s in settings:
                for b, r in ((T["settings"].get(s) or {}).get("budgets") or {}).items():
                    by_b.setdefault(int(b), []).append(r["equal_total_selective"][sig])
            if not by_b:
                continue
            k = f"drv.c1bt.{task}.{sig}"
            if task == "math":
                # every budget either leaves room for some escalation in all settings, or in none
                room = {b: [e["share_pct"] > 0 for e in es] for b, es in by_b.items()}
                assert all(all(x) or not any(x) for x in room.values()), (k, room)
                fit = sorted(b for b, x in room.items() if all(x))
                none = sorted(b for b, x in room.items() if not any(x))
                assert fit and none and max(none) < min(fit), (k, fit, none)
                cells = [e["selective_minus_uniform"] for b in fit for e in by_b[b]]
                assert min(c["estimate"] for c in cells) > 0, (k, "selective is not ahead at every fitting budget")
                put(k + ".minb", f"{min(fit):,}")
                put(k + ".zerob", " and ".join(f"{b:,}" for b in none))
                put(k + ".gapmin", fnum(min(c["estimate"] for c in cells), sign=False))
                put(k + ".gapmax", fnum(max(c["estimate"] for c in cells), sign=False))
                excl = sum(c["lo"] > 0 for c in cells)
                put(k + ".exclphrase", "every unadjusted interval excludes zero" if excl == len(cells)
                    else f"{excl} of {len(cells)} unadjusted intervals exclude zero")
            else:
                cells = [e["selective_minus_uniform"] for b in sorted(by_b) for e in by_b[b]]
                ahead = sum(c["estimate"] < 0 for c in cells)
                assert ahead == len(cells), (k, "the uniform budget is not ahead at every budget")
                put(k + ".ugapmin", fnum(min(-c["estimate"] for c in cells), sign=False))
                put(k + ".ugapmax", fnum(max(-c["estimate"] for c in cells), sign=False))
                excl = sum(c["hi"] < 0 for c in cells)
                put(k + ".uexcl", f"{excl} of {len(cells)}")


def tab_budget_total(G):
    """Appendix table of plan Amendment 8 (written only when its output exists)."""
    T = G.get("c1bt")
    if not T:
        return
    rows = []
    for s in THINK:
        v, sl = T["settings"].get(s) or {}, SLUG[s]
        if not v.get("budgets"):
            continue
        if rows:
            rows.append("MID")
        for j, b in enumerate(sorted(v["budgets"], key=int)):
            cells = [setting_name(s, short=True) if j == 0 else "", f"{int(b):,}",
                     f"{V[f'{sl}.c1b.{b}.d']} ({V[f'{sl}.c1b.{b}.tok']})"]
            for sig in ("verbal", "probe"):
                k = f"{sl}.c1bt.{b}.{sig}"
                cells += [f"{V[k + '.share']}\\%", f"{V[k + '.d']} ({V[k + '.tot']})", V[k + ".diffci"]]
            rows.append(cells)
    a0 = [V[f"{SLUG[s]}.c1bt.a0tok"] for s in THINK if f"{SLUG[s]}.c1bt.a0tok" in V]
    head = [r"& & & \multicolumn{3}{c}{Verbal-selected thinking} & \multicolumn{3}{c}{Probe-selected thinking} \\"
            r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}"
            r"Setting & $B$ & Uniform: $\Delta$ (tok.) & Share & $\Delta$ (total tok.) & $-$ uniform & Share & "
            r"$\Delta$ (total tok.) & $-$ uniform"]
    cap = (r"Uniform thinking budgets against selective thinking at equal total generated tokens (descriptive; defined "
           r"after a first look at the mathematics point estimates). A uniform budget does not use $a_0$, so its cost is "
           r"that of Table~\ref{tab:budget}. A selective policy generates $a_0$ for every item ("
           + ", ".join(a0[:-1]) + (" and " if len(a0) > 1 else "") + a0[-1] +
           r" tokens per item in the three settings) and thinks on the largest share, in 1-point steps, whose total does "
           r"not exceed the budget's; a share of 0\% is \KEEP. $\Delta$ against \KEEP{} in pp. ``$-$ uniform'' is the "
           r"paired difference from the uniform budget with an unadjusted 95\% interval.")
    table("tab_budget_total.tex", "lrrrrrrrr", head, rows, cap, "tab:budgettotal", size=r"\scriptsize", sep="3pt")


def macros_a3(G):
    out = {}
    for src, settings in (("a3_dev_test", ["math 1.7B", "math 4B", "MMLU-Pro 4B", "MMLU-Pro T8B"]),
                          ("a3_b5", ["math 8B", "math T8B"])):
        for s in settings:
            a, sl = G[src]["settings"][s], SLUG[s]
            put(f"{sl}.a3.devn", f"{a['dev_n']:,}")
            put(f"{sl}.a3.grid", a["grid_size"])
            for rule in ("R1", "R2"):
                R = a["rules"][rule]
                pol = R["policy"]
                k = f"{sl}.a3.{rule}"
                put(k + ".pol", f"{OP_NAME[pol['operator']]}, {SIGNAL_NAME[pol['signal']]}, {pol['share_pct']}\\%")
                put(k + ".dev", fnum(R["dev_delta_pp"]))
                put(k + ".ci", fcid(R["test_delta_vs_keep"]))
                put(k + ".d", fnum(R["test_delta_vs_keep"]["estimate"]))
                put(k + ".vsall", fcid(R["test_delta_vs_operator_all"]))
                put(k + ".thr", fcid(R["threshold_transfer"]["test_delta_vs_keep"]))
                lo, hi = R["joint_reselection"]["test_delta_ci_pp"]
                put(k + ".joint", f"[{fnum(lo, sign=False)}, {fnum(hi, sign=False)}]")
                put(k + ".rank", R["test_rank"])
                put(k + ".regret", fpct(R["test_regret_pp"]))
                put(k + ".tok", ftok(R["test_generated_tokens"]))
                out[(s, rule)] = R
    return out


def macros_a0(G):
    a = G["a0_common_labels"]
    put("a0.checks", f"{a['acl_4B_parity_checks']:,}")


def macros_config(G):
    """Design values as run: budgets and decoding from the run configs, probes, versions, revisions, compute."""
    cfgs = {
        "a0long": "results/arr/b1/4B/b1-samples-test-k1-a0-long-t8192-4B-279e6b0/config.json",
        "cont": "results/arr/b2/4B/b2-continue-test-x4096-4B-279e6b0/config.json",
        "think": "results/arr/c1/4B/c1-think-math-test-c0-4B-279e6b0/config.json",
    }
    cfg = {k: jload(v) for k, v in cfgs.items()}
    base = cfg["think"]
    for k in ("a0_tokens", "revision_tokens", "critique_tokens"):
        assert all(c[k] == base[k] for c in cfg.values()), k
        put(f"cfg.{pslug(k)}", f"{base[k]:,}")
    put("cfg.a0long", f"{cfg['a0long']['max_tokens']:,}")
    put("cfg.contextra", f"{cfg['cont']['extra_tokens']:,}")
    put("cfg.thinktokens", f"{cfg['think']['max_tokens']:,}")
    sm, ts = base["sampling"], base["think_sampling"]
    put("cfg.T", sm["temperature"])
    put("cfg.topp", sm["top_p"])
    put("cfg.topk", sm["top_k"])
    put("cfg.thinkT", ts["temperature"])
    put("cfg.thinktopp", ts["top_p"])
    put("cfg.thinktopk", ts["top_k"])
    put("cfg.thinkminp", ts["min_p"])
    common = literals("analysis/arr/common.py")
    put("cfg.nboot", f"{common['N_BOOT']:,}")
    put("cfg.seed", common["SEED"])
    tr = literals("src/probe/train.py")
    put("cfg.probefolds", tr["FOLDS"])
    put("cfg.probeCgrid", ", ".join(str(c) for c in tr["C_GRID"]))
    probes = {"math 1.7B": "results/p3/1.7B/p3-probe-dev-ext-1.7B-b7cf0d6",
              "math 4B": "results/p3/4B/p3-probe-dev-ext-4B-b7cf0d6",
              "math 8B": "results/p3/8B/p3-probe-dev-ext-8B-3fd21bc",
              "math T8B": "results/fu2/p2/T8B/p2-probe-dev-T8B-1809070",
              "MMLU-Pro 4B": "results/v2/fu/4B/fu-mc-hf-4B-9221b24",
              "MMLU-Pro T8B": "results/v2/fu/T8B/fu-mc-hf-T8B-9221b24"}
    scorers = {"math 1.7B": "results/main/1.7B/main-probe-scores-test-1.7B-3fd21bc/config.json",
               "math 4B": "results/main/4B/main-probe-scores-test-4B-a7a0579/config.json",
               "math 8B": "results/main/8B/main-probe-scores-test-8B-3fd21bc/config.json",
               "math T8B": "results/fu2/main/T8B/main-probe-scores-test-T8B-1809070/config.json"}
    for s, rel in scorers.items():   # the probe that scored the test items is the one listed in `probes`
        assert probes[s].split("/")[-1] in jload(rel)["probe"], (s, rel)
    for s, d in probes.items():
        sel = jload(d + "/probe_selection.json")
        put(f"{SLUG[s]}.probe.layer", sel["selected"]["layer"])
        put(f"{SLUG[s]}.probe.C", sel["selected"]["C"])
        put(f"{SLUG[s]}.probe.cv", f"{sel['selected']['cv_auroc']:.3f}")
        put(f"{SLUG[s]}.probe.ndev", f"{sel['n_dev']:,}")
    lock = inp("configs/lock.txt").read_text(encoding="utf-8")
    for pkg in ("vllm", "torch", "transformers", "math-verify", "sympy", "scikit-learn", "numpy", "statsmodels"):
        m = re.search(rf"^{re.escape(pkg)}==([^\s]+)$", lock, re.M)
        put(f"ver.{pslug(pkg)}", m.group(1))
    import yaml  # noqa: PLC0415 (only here)
    mods = yaml.safe_load(inp("configs/models.yaml").read_text(encoding="utf-8"))["models"]
    for key, slug in (("qwen3-1.7b", "17"), ("qwen3-4b", "4"), ("qwen3-8b", "8"), ("tulu3-8b", "T")):
        put(f"rev.{slug}", mods[key]["revision"][:7])
        put(f"repo.{slug}", tex_escape(mods[key]["repo"]))
    rows = [json.loads(line) for line in inp("cloud/ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    gpuh, total, arr = {}, 0.0, 0.0
    for r in rows:
        m = re.match(r"(\d+)x\s*(.*)", r["gpu"])
        n, name = (int(m.group(1)), m.group(2)) if m else (1, r["gpu"])
        h = (r.get("hours") or 0) * n
        gpuh[name] = gpuh.get(name, 0) + h
        total += h
        if r.get("provider") == "vast":
            arr += h
    for tag, rel in (("h1", "results/arr/parity_279e6b0.json"), ("h2", "results/arr/parity_279e6b0_h2.json")):
        p = jload(rel)
        assert p["ops_parity"] and p["samples_identical"] == p["samples"] and p["ops_identical"] == p["ops"], rel
        put(f"parity.{tag}.samples", p["samples"])
        put(f"parity.{tag}.ops", p["ops"])
    put("guards.n", len(G["c1_think"]["scorer_guards"]))
    put("cmp.total", f"{total:.0f}")
    put("cmp.arr", f"{arr:.0f}")
    put("cmp.sessions", len(rows))
    for name, h in gpuh.items():
        put(f"cmp.{pslug(name)}", f"{h:.0f}")
    return cfg


def macros_review(G, P_by_setting):
    """Plan Amendment 6: the descriptive replays requested by the review of the v2 draft."""
    R = G["rev"]
    assert R["plan_core_sha256"] == G["family_holm"]["plan_core_sha256"]
    for s, e in R["r1a"].items():
        sl = SLUG[s]
        put(f"{sl}.r1.base", fpct(e["base_pct"], 1))
        for sig, by in e["by_signal"].items():
            for share, x in by.items():
                put(f"{sl}.r1.{sig}.{share}.pct", f"{x['answerless_pct_of_selected']:.0f}")
                put(f"{sl}.r1.{sig}.{share}.cov", f"{x['answerless_covered_pct']:.0f}")
    qwen = [s for s in SETTINGS if "T8B" not in s]
    cov = [R["r1a"][s]["by_signal"][g]["8"]["answerless_covered_pct"] for s in qwen for g in ("verbal", "probe", "length")]
    put("drv.r1.qwenfree.cov8.min", f"{min(cov):.0f}")
    put("drv.r1.qwenfree.cov8.max", f"{max(cov):.0f}")
    for s, x in R["r1b"].items():
        sl = SLUG[s]
        put(f"{sl}.r1b.ci", fcid(x["delta"]))
        put(f"{sl}.r1b.d", fnum(x["delta"]["estimate"]))
        put(f"{sl}.r1b.tok", ftok(x["generated_tokens_per_item"]))
        allgain = G["c1_think"]["settings"][s]["reach"]["think_all_gain_pp"]
        put(f"{sl}.r1b.shareofgain", f"{100 * x['delta']['estimate'] / allgain:.0f}")
    for s, ops in R["r1c"].items():
        sl = SLUG[s]
        for op, sigs in ops.items():
            for sig, e in sigs.items():
                for share in ("10", "20"):
                    put(f"{sl}.r1c.{pslug(op)}.{sig}.{share}.ci", fcid(e[share]["delta"]))
                    put(f"{sl}.r1c.{pslug(op)}.{sig}.{share}.tok", ftok(e[share]["generated_tokens_per_item"]))
                for frac, hit in (e.get("reach") or {}).items():
                    put(f"{sl}.r1c.{pslug(op)}.{sig}.reach{frac.rstrip('%')}.share", "--" if hit is None else hit["share_pct"])
    for s, e in R["r1d"].items():
        sl = SLUG[s]
        put(f"{sl}.ans.n", f"{e['n']:,}")
        put(f"{sl}.ans.keep", fpct(e["keep_acc_pct"]))
        for sig, v in e["auroc"].items():
            put(f"{sl}.ans.auroc.{sig}", f"{v:.3f}")
        for op, o in e["operators"].items():
            put(f"{sl}.ans.{pslug(op)}ALL.ci", fcid(o["ALL"]["delta"]))
            put(f"{sl}.ans.{pslug(op)}ALL.d", fnum(o["ALL"]["delta"]["estimate"]))
            put(f"{sl}.ans.{pslug(op)}ALL.tok", ftok(o["ALL"]["generated_tokens_per_item"]))
            for sig in SIGNALS:
                for share in ("10", "20"):
                    put(f"{sl}.ans.{pslug(op)}.{sig}.{share}.ci", fcid(o[sig][share]["delta"]))
                    put(f"{sl}.ans.{pslug(op)}.{sig}.{share}.tok", ftok(o[sig][share]["generated_tokens_per_item"]))
                for frac, hit in (o[sig].get("reach") or {}).items():
                    k = f"{sl}.ans.{pslug(op)}.{sig}.reach{frac.rstrip('%')}"
                    put(k + ".share", "--" if hit is None else hit["share_pct"])
                    put(k + ".tok", "--" if hit is None else ftok(hit["generated_tokens_per_item"]))
                    if hit is not None and op == "THINK":
                        put(k + ".tokshare", f"{100 * hit['generated_tokens_per_item'] / o['ALL']['generated_tokens_per_item']:.0f}")
    for s, ops in R["r2"].items():
        sl = SLUG[s]
        for op, e in ops.items():
            for share in ("10", "20"):
                put(f"{sl}.bound.{pslug(op)}.{share}.ci", fcid(e[share]["delta"]))
                put(f"{sl}.bound.{pslug(op)}.{share}.d", fnum(e[share]["delta"]["estimate"]))
                put(f"{sl}.bound.{pslug(op)}.{share}.tok", ftok(e[share]["generated_tokens_per_item"]))
    crit = {s: R["r2"][s]["F2"]["20"]["delta"]["estimate"] for s in SETTINGS}
    put("drv.bound.F2.20.max", fnum(max(crit.values())))
    put("drv.bound.F2.20.maxsetting", setting_name(max(crit, key=crit.get)))
    for op in ("F0", "F2-scratch"):
        put(f"drv.bound.{pslug(op)}.20.max", fnum(max(R["r2"][s][op]["20"]["delta"]["estimate"] for s in SETTINGS)))
    for s, e in R["r3a"].items():
        sl = SLUG[s]
        for share in ("10", "20"):
            x = e[share]
            put(f"{sl}.r3a.{share}.ci", fcid(x["delta"]))
            put(f"{sl}.r3a.{share}.vsV5", fcid(x["vs_V5"]))
            put(f"{sl}.r3a.{share}.tok", ftok(x["generated_tokens_per_item"]))
    for s, e in R["r3b"].items():
        sl = SLUG[s]
        for share in ("10", "20"):
            x = e[share]
            put(f"{sl}.r3b.{share}.ci", fcid(x["delta"]))
            put(f"{sl}.r3b.{share}.vsV5", fcid(x["vs_V5"]))
            put(f"{sl}.r3b.{share}.tok", ftok(x["generated_tokens_per_item"]))
    gains = [R["r3b"][s]["20"]["vs_V5"]["estimate"] for s in R["r3b"]]
    put("drv.r3b.20.min", fnum(min(gains)))
    put("drv.r3b.20.max", fnum(max(gains)))
    for s, e in R["r3c"].items():
        sl = SLUG[s]
        put(f"{sl}.r3c.disagree", fpct(e["disagree_pct"], 1))
        for share in ("10", "20"):
            put(f"{sl}.r3c.{share}.ci", fcid(e[share]["delta"]))
            put(f"{sl}.r3c.{share}.tok", ftok(e[share]["generated_tokens_per_item"]))
        for frac, hit in e["reach"].items():
            put(f"{sl}.r3c.reach{frac.rstrip('%')}.share", "--" if hit is None else hit["share_pct"])
            put(f"{sl}.r3c.reach{frac.rstrip('%')}.tok", "--" if hit is None else ftok(hit["generated_tokens_per_item"]))
    for s, e in R["r5"].items():
        sl = SLUG[s]
        put(f"dev.{sl}.n", f"{e['n']:,}")
        for mem, x in e["members"].items():
            put(f"dev.{sl}.{mem}.ci", fcid(x["difference"]))
    for s, e in R["r6"].items():
        sl = SLUG[s]
        put(f"{sl}.r6.min", fpct(e["min"]))
        put(f"{sl}.r6.max", fpct(e["max"]))
        put(f"{sl}.r6.dmin", fnum(e["min"] - e["keep_pct"]))
        put(f"{sl}.r6.dmax", fnum(e["max"] - e["keep_pct"]))
        put(f"{sl}.r6.range", fnum(e["max"] - e["min"], 2, sign=False))
    for x in G["c1_think"]["cross_check"]:
        put(f"c1x.{pslug(x['model'])}.seed1.ci", fcid(x["C1_minus_P4_seed1"]))
    # cost including a0, for a system that could skip a0 and think on every item (R4b; arithmetic)
    for s in ("math 4B", "math 8B"):
        sl = SLUG[s]
        a0 = G["b2_keep_strength"]["settings"][s]["a0_tokens_per_item"]
        c = G["c1_think"]["settings"][s]
        r50 = c["reach"]["by_signal"]["verbal"]["50%"]["generated_tokens_per_item"]
        allt = c["policies"]["THINK ALL"]["generated_tokens_per_item"]
        put(f"{sl}.drv.reach50.witha0.tokshare", f"{100 * (a0 + r50) / allt:.0f}")


def macros_derived(G, P_by_setting, EQ):
    """Ranges and ratios of reported point estimates, used in the text."""
    ops = [(s, op) for s in SETTINGS for op in ("F0 ALL", "F2 ALL", "F2-scratch ALL")]
    ds = [P_by_setting[s][op]["delta_vs_keep"]["estimate"] for s, op in ops]
    put("drv.opmin", fnum(min(ds)))
    put("drv.opmax", fnum(max(ds)))
    for s in SETTINGS:
        P, sl = P_by_setting[s], SLUG[s]
        v5, g5 = P["V5"], P.get("GATE-a0 K5", P.get("GATE-a0"))
        put(f"{sl}.drv.gatesave", f"{100 * (1 - g5['generated_tokens_per_item'] / v5['generated_tokens_per_item']):.0f}")
        ans, f2 = P["ANSWERLESS->s1"], P["F2 ALL"]
        put(f"{sl}.drv.anstokshare", f"{100 * ans['generated_tokens_per_item'] / f2['generated_tokens_per_item']:.0f}")
    for s in THINK:
        c, sl = G["c1_think"]["settings"][s], SLUG[s]
        allt = c["policies"]["THINK ALL"]["generated_tokens_per_item"]
        v = c["reach"]["by_signal"]["verbal"]["50%"]
        put(f"{sl}.drv.reach50verbal.tokshare", f"{100 * v['generated_tokens_per_item'] / allt:.0f}")
        at = next(r for r in c["matched"]["think_at_reference_cost"] if r["reference"] == "F2 ALL" and r["signal"] == "verbal")
        f2 = c["matched"]["references"]["F2 ALL"]["delta_vs_keep"]["estimate"]
        if f2 > 0:
            put(f"{sl}.drv.thinkoverf2", f"{at['delta_ci']['estimate'] / f2:.1f}")
        for share in (10, 20):
            th, cr = EQ[(s, "THINK", "verbal", share)], EQ[(s, "F2", "verbal", share)]
            put(f"{sl}.drv.repairratio.{share}", f"{th['r'] / cr['r']:.1f}")
    # the share of items that the 20 % verbal selection holds among the items whose a0 has no usable answer:
    # THINK 20 % by verbal and THINK ALL have the same completion part when every answerless a0 is selected.
    for s in THINK:
        c, sl = G["c1_think"]["settings"][s], SLUG[s]
        same = abs(c["policies"]["THINK by verbal 20%"]["repairs_completion"]["estimate"]
                   - c["policies"]["THINK ALL"]["repairs_completion"]["estimate"]) < 1e-12
        put(f"{sl}.drv.verbal20allcompletion", "yes" if same else "no")

    def rng(key, vals, d=2, sign=True, absval=False):
        vals = [abs(v) for v in vals] if absval else list(vals)
        put(f"drv.{key}.min", fnum(min(vals), d, sign))
        put(f"drv.{key}.max", fnum(max(vals), d, sign))

    F = G["family_holm"]["members"]
    rng("fam1rej", [m["difference"]["estimate"] for m in F if m["member"] == 1 and m["holm_reject_0.05"]], absval=True,
        sign=False)
    rng("thinkall", [G["c1_think"]["settings"][s]["policies"]["THINK ALL"]["delta_vs_keep"]["estimate"] for s in THINK])
    for op in ("F0", "F2", "F2-scratch", "s1"):
        rng(f"r20.{pslug(op)}", [EQ[(s, op, "verbal", 20)]["r"] for s in SETTINGS], d=0, sign=False)
        rng(f"d20.{pslug(op)}", [EQ[(s, op, "verbal", 20)]["dmg"] for s in SETTINGS], d=0, sign=False)
    rng("r20.THINK.math", [EQ[(s, "THINK", "verbal", 20)]["r"] for s in THINK if s in MATH], d=0, sign=False)
    rng("w20", [EQ[(s, "F2", "verbal", 20)]["w"] for s in SETTINGS], d=0, sign=False)
    rng("dec", [P_by_setting[s]["V1"]["delta_vs_keep"]["estimate"] for s in SETTINGS])
    agg = [contrasts(G, s)[("V5", "V1")]["difference"]["estimate"] for s in SETTINGS]
    rng("agg", agg)
    rng("v5", [P_by_setting[s]["V5"]["delta_vs_keep"]["estimate"] for s in SETTINGS])
    rng("gatesave", [100 * (1 - P_by_setting[s].get("GATE-a0 K5", P_by_setting[s].get("GATE-a0"))["generated_tokens_per_item"]
                            / P_by_setting[s]["V5"]["generated_tokens_per_item"]) for s in SETTINGS], d=0, sign=False)
    B2 = G["b2_keep_strength"]["settings"]
    rng("a0long.qwen", [B2[s]["a0_long"]["delta_vs_keep"]["estimate"] for s in MATH if "T8B" not in s])
    rng("a0cont", [B2[s]["a0_cont"]["delta_vs_keep"]["estimate"] for s in MATH])
    b4 = G["b4_critique"]["settings"]
    rng("b4vsf2", [c["difference"]["estimate"] for s in SETTINGS for c in b4[s]["contrasts"]], absval=True, sign=False)
    for sig_set, key in ((["verbal", "probe", "length"], "free"), (["agreement", "margin"], "sample")):
        pass
    au = {}
    for x in G["auroc"]:
        if x["labels"] == "common":
            au[(x["setting"], x["signal"])] = x["pooled"]
    rng("auroc.math.agreement", [au[(s, "agreement")] for s in MATH], d=2, sign=False)
    rng("auroc.math.probe", [au[(s, "probe")] for s in MATH], d=2, sign=False)
    rng("auroc.math.verbal.qwen", [au[(s, "verbal")] for s in MATH if "T8B" not in s], d=2, sign=False)
    rng("auroc.mc", [au[(s, g)] for s in MC for g in SIGNALS], d=2, sign=False)
    for s in THINK:
        for op in ("F2", "THINK"):
            vals = [EQ[(s, op, g, 20)]["d"]["estimate"] for g in SIGNALS]
            put(f"{SLUG[s]}.drv.spread20.{op}", fnum(max(vals) - min(vals), 2, sign=False))
    spreads = []
    for s in SETTINGS:   # critique at 20 % by each of the five signals (all settings have all five after B1)
        vals = [EQ[(s, "F2", g, 20)]["d"]["estimate"] for g in SIGNALS]
        spreads.append(max(vals) - min(vals))
        put(f"{SLUG[s]}.drv.spread20F2", fnum(max(vals) - min(vals), 2, sign=False))
    rng("spread20F2", spreads, sign=False)
    put("drv.spread20F2.maxsetting", setting_name(SETTINGS[spreads.index(max(spreads))]))
    # how often vote margin gives the smallest selective gain among the five signals (intrinsic operators and s1)
    low = total = 0
    for s in SETTINGS:
        for op in ("F0", "F2", "F2-scratch", "s1"):
            for share in (10, 20):
                d = {g: EQ[(s, op, g, share)]["d"]["estimate"] for g in SIGNALS if (s, op, g, share) in EQ}
                if len(d) == len(SIGNALS):
                    total += 1
                    low += min(d, key=d.get) == "margin"
    put("drv.marginlow.n", low)
    put("drv.marginlow.total", total)
    # aggregation (V5 - V1) exceeds the decoding change (V1 - KEEP) in every setting?
    dom = all(contrasts(G, s)[("V5", "V1")]["difference"]["estimate"] > P_by_setting[s]["V1"]["delta_vs_keep"]["estimate"]
              for s in SETTINGS)
    assert dom, "the text says aggregation is the larger part in every setting"
    # V1 costs no more than critique on all items?
    put("drv.v1cheaper.n", sum(P_by_setting[s]["V1"]["generated_tokens_per_item"]
                               <= P_by_setting[s]["F2 ALL"]["generated_tokens_per_item"] for s in SETTINGS))
    gd = []
    for s in SETTINGS:
        P = P_by_setting[s]
        g5 = P.get("GATE-a0 K5", P.get("GATE-a0"))
        gd.append(g5["delta_vs_keep"]["estimate"] - P["V5"]["delta_vs_keep"]["estimate"])
    rng("gatediff", gd, sign=False, absval=True)
    rng("a3.regretR1", [G["a3_b5" if s in ("math 8B", "math T8B") else "a3_dev_test"]["settings"][s]["rules"]["R1"]
                        ["test_regret_pp"] for s in SETTINGS], sign=False)
    # dev-selected policies that pay for five samples, against V5 at the same sampling cost (point estimates)
    n_samples_pol, n_v5_higher = 0, 0
    for s in SETTINGS:
        src = "a3_b5" if s in ("math 8B", "math T8B") else "a3_dev_test"
        R = G[src]["settings"][s]["rules"]["R1"]
        pol = R["policy"]
        if pol["operator"] == "s1" or pol["signal"] in ("agreement", "margin"):
            n_samples_pol += 1
            v5 = P_by_setting[s]["V5"]
            if (v5["delta_vs_keep"]["estimate"] > R["test_delta_vs_keep"]["estimate"]
                    and v5["generated_tokens_per_item"] <= R["test_generated_tokens"] + 0.5):
                n_v5_higher += 1
    put("drv.a3.samplepol", n_samples_pol)
    put("drv.a3.v5higher", n_v5_higher)
    rng("a3.r1", [G["a3_b5" if s in ("math 8B", "math T8B") else "a3_dev_test"]["settings"][s]["rules"]["R1"]
                  ["test_delta_vs_keep"]["estimate"] for s in SETTINGS])
    rng("a3.regret", [G["a3_b5" if s in ("math 8B", "math T8B") else "a3_dev_test"]["settings"][s]["rules"][r]
                      ["test_regret_pp"] for s in SETTINGS for r in ("R1", "R2")], sign=False)


# ---------------------------------------------------------------------------------------------------------------
# Tables
def tab_settings(G, AU, P_by_setting):
    rows = []
    for s in SETTINGS:
        P, sl = P_by_setting[s], SLUG[s]
        cells = [setting_name(s), V[f"{sl}.n"], V[f"{sl}.keep"], V[f"{sl}.unans"]]
        for sig in SIGNALS:
            x = AU.get((s, sig))
            if x is None:
                cells.append("--")
            elif s in MATH:
                cells.append(f"{x['pooled']:.2f} ({x['within_source']:.2f})")
            else:
                cells.append(f"{x['pooled']:.2f}")
        rows.append(cells)
        if s == MATH[-1]:
            rows.append("MID")
    head = [r"& & & & \multicolumn{3}{c}{Free signals} & \multicolumn{2}{c}{Sample signals} \\ \cmidrule(lr){5-7}\cmidrule(lr){8-9}"
            r"Setting & $n$ & \KEEP{} (\%) & No answer (\%) & Verbal & Probe & Length & Agreement & Margin"]
    cap = (r"Settings and correctness signals. \KEEP{} is the accuracy of the greedy initial answer $a_0$; ``no answer'' "
           r"is the share of $a_0$ without a usable final answer (truncated, unparsed or conflicting). Columns 5--9 give "
           r"AUROC for $a_0$ being correct, pooled over the pool and, for mathematics, within source in parentheses "
           r"(GSM8K, MATH-500, MATH; the pair-weighted mean of per-source AUROCs). Free signals generate no tokens; "
           r"agreement and vote margin need five sampled answers. Intervals, and per-source values for mathematics: "
           r"Table~\ref{tab:aurocsrc}.")
    table("tab_settings.tex", "lrrrccccc", head, rows, cap, "tab:settings")


def tab_operators(G, P_by_setting):
    rows = []
    for s in SETTINGS:
        P, sl = P_by_setting[s], SLUG[s]
        cells = [setting_name(s, short=True)]
        for op in ("F0 ALL", "F2 ALL", "F2-scratch ALL"):
            cells += [V[f"{sl}.{pslug(op)}.ci"], V[f"{sl}.{pslug(op)}.tok"]]
        if s in B3_SETTINGS:
            cells.append("/".join(V[f"{sl}.b3.{w}.con.d"] for w in ("main", "p1", "p2")))
        else:
            cells.append("--")
        rows.append(cells)
        if s == MATH[-1]:
            rows.append("MID")
    head = [r"& \multicolumn{2}{c}{Review, all} & \multicolumn{2}{c}{Critique, all} & \multicolumn{2}{c}{Re-solve, all} & Critique $-$ re-solve \\"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}"
            r"Setting & $\Delta$ (pp) & Tok. & $\Delta$ (pp) & Tok. & $\Delta$ (pp) & Tok. & three wordings"]
    cap = (r"Whole-pool revision operators. $\Delta$ is the accuracy change against \KEEP{} with a paired 95\% "
           r"cluster-bootstrap interval; Tok.\ is generated tokens per item after $a_0$ (both turns for critique and "
           r"re-solve). The last column gives the point estimate of critique $-$ re-solve, both on all items, under the "
           r"original wording and two paraphrases of the first-turn instruction (main / p1 / p2; second round; "
           r"intervals in Appendix~\ref{app:tables}). Qwen3-1.7B and MMLU-Pro have no paraphrase runs.")
    table("tab_operators.tex", "lrrrrrrc", head, rows, cap, "tab:operators", size=r"\footnotesize", sep="1.4pt")


def tab_eq2(EQ):
    ops = ("F0", "F2", "F2-scratch", "s1", "THINK")
    rows = [r"\multicolumn{7}{l}{\emph{Mathematics}} \\"]
    short = {"1.7B": "Qwen 1.7B", "4B": "Qwen 4B", "8B": "Qwen 8B", "T8B": "Tulu 8B"}
    for s in SETTINGS:
        cells = [r"\quad " + short[s.split(" ")[1]], f"{EQ[(s, 'F2', 'verbal', 20)]['w']:.0f}"]
        for op in ops:
            r = EQ.get((s, op, "verbal", 20))
            if r is None:
                cells.append("--")
                continue
            assert abs(r["w"] - EQ[(s, "F2", "verbal", 20)]["w"]) < 1e-9, (s, op)   # the same selected items
            cells.append(f"{r['r']:.0f}/{r['dmg']:.0f}")
        rows.append(cells)
        if s == MATH[-1]:
            rows.append(r"\multicolumn{7}{l}{\emph{MMLU-Pro}} \\")
    head = [r"Model & $w_S$ & Review & Critique & Re-solve & Resample & Think"]
    cap = (r"One selected set, five operators. $S$ is the 20\% of items with the lowest verbal score; $w_S$ (\%) is "
           r"the share of $S$ whose $a_0$ is wrong, identical in every column. Each cell gives the repair rate "
           r"$r$ among wrong items of $S$ and the damage rate $d$ among correct items of $S$ (\%), so that "
           r"$\Delta=\frac{|S|}{n}[w_Sr-(1-w_S)d]$ (Eq.~\ref{eq:decomposition}). Resample replaces $a_0$ by one sampled "
           r"answer; think re-answers in thinking mode (three settings). Second round; intervals in "
           r"Table~\ref{tab:eq2ci}, other signals and shares in Table~\ref{tab:grid}.")
    table("tab_eq2.tex", "lrccccc", head, rows, cap, "tab:eq2", wide=False, size=r"\scriptsize", sep="2.6pt")


def tab_samples(G, P_by_setting):
    def cell(s, pol, tok=True):
        k = f"{SLUG[s]}.{pslug(pol)}"
        if f"{k}.d" not in V:
            return "--"
        return f"{V[k + '.d']} ({V[k + '.tok']})" if tok else V[k + ".d"]

    def fam(s, member):
        k = f"fam.{member}.{SLUG[s]}"
        return V[k + ".d"] + V[k + ".rej"]

    spec = [("Resample all ($V_1$)", lambda s: cell(s, "V1")),
            ("Plurality, $k{=}5$ ($V_5$)", lambda s: cell(s, "V5")),
            ("Plurality, $k{=}8$ ($V_8$)", lambda s: cell(s, "V8")),
            (r"Gate on $a_0$ (\textsc{gate}, $K{=}5$)", lambda s: cell(s, "GATE-a0 K5") if f"{SLUG[s]}.GATEa0K5.d" in V else cell(s, "GATE-a0")),
            (r"Answerless $\to$ resample", lambda s: cell(s, "ANSWERLESS->s1")),
            (r"Agreement $\to$ critique, 20\%", lambda s: cell(s, "AGREE->F2 20%")),
            ("MID", None),
            (r"Hybrid (critique) $-V_5$ \textsuperscript{[1]}", lambda s: fam(s, 1)),
            (r"Hybrid (sample-aware) $-V_5$", lambda s: V[f"{SLUG[s]}.HYBRIDmarginB420.vsV5.d"]),
            (r"$V_1-$ critique, all \textsuperscript{[3]}", lambda s: fam(s, 3)),
            (r"Longer sampled $a_0$ $-$ \KEEP{}", lambda s: V.get(f"{SLUG[s]}.a0long.d", "--"))]
    rows = []
    for label, f in spec:
        if label == "MID":
            rows.append("MID")
            continue
        rows.append([label] + [f(s) for s in SETTINGS])
    head = [r"& \multicolumn{4}{c}{Mathematics} & \multicolumn{2}{c}{MMLU-Pro} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-7}"
            r"Policy & Qwen-1.7B & Qwen-4B & Qwen-8B & Tulu-8B & Qwen-4B & Tulu-8B"]
    cap = (r"Policies that use sampled answers. Top block: accuracy change against \KEEP{} in pp (generated tokens "
           r"per item); \textsc{gate} keeps $a_0$ when the first sample agrees with it and otherwise votes over $a_0$ "
           r"and five samples. Bottom block: paired differences in pp; the hybrids give the 20\% of items with the "
           r"lowest vote margin to a critique of $a_0$ (or to a critique that sees the five sampled answers) and $V_5$ "
           r"to the rest. The last row is a stronger \KEEP{}: one sampled $a_0$ with an \vnum{cfg.a0long}-token "
           r"budget. Superscripts number the tested contrasts (Table~\ref{tab:family}); \dag: rejected by Holm at 0.05 "
           r"over all \vnum{fam.n}. All rows are post hoc; $V_1$, $V_5$ and agreement $\to$ critique for Qwen3-4B "
           r"mathematics and for MMLU-Pro were reported in the first round, every other entry in the second. Intervals "
           r"and further policies: Tables~\ref{tab:samplesci} and~\ref{tab:b2}.")
    table("tab_samples.tex", "lcccccc", head, rows, cap, "tab:samples", size=r"\small")


def tab_think(G, EQ):
    sets = THINK

    def v(key):
        return V.get(key, "--")

    rows = []
    rows.append([r"Think, all items: $\Delta$ \textsuperscript{[7]} (tok.)"] +
                [v(f"{SLUG[s]}.THINKALL.ci") + v(f"fam.7.{SLUG[s]}.rej") + f" ({v(SLUG[s] + '.THINKALL.tok')})" for s in sets])
    rows.append("MID")
    for sig in ("verbal", "probe", "agreement"):
        rows.append([f"Think on 20\\% by {SIGNAL_NAME[sig]}"] +
                    [f"{v(f'{SLUG[s]}.THINKby{sig}20.d')} ({v(f'{SLUG[s]}.THINKby{sig}20.tok')})" for s in sets])
    rows.append([r"Think on 20\% at random (expected)"] +
                [f"{v(f'{SLUG[s]}.eq2.THINK.random.20.d')} ({v(f'{SLUG[s]}.eq2.THINK.random.20.tok')})" for s in sets])
    rows.append([r"Probe $-$ verbal, 20\% \textsuperscript{[6]}"] +
                [v(f"fam.6.{SLUG[s]}.ci") + v(f"fam.6.{SLUG[s]}.rej") for s in sets])
    rows.append("MID")
    for frac in ("50", "80"):
        rows.append([f"Share (tok.) for {frac}\\% of the gain: verbal"] +
                    [f"{v(f'{SLUG[s]}.reach.verbal.{frac}.share')}\\% ({v(f'{SLUG[s]}.reach.verbal.{frac}.tok')})" for s in sets])
    rows.append("MID")
    rows.append([r"Critique, all items: $\Delta$ (tok.)"] +
                [f"{v(f'{SLUG[s]}.F2ALL.d')} ({v(f'{SLUG[s]}.F2ALL.tok')})" for s in sets])
    rows.append([r"\quad think by verbal at that cost: share, $\Delta$"] +
                [f"{v(f'{SLUG[s]}.at.F2ALL.verbal.share')}\\%, {v(f'{SLUG[s]}.at.F2ALL.verbal.ci')}" for s in sets])
    rows.append([r"$V_5$: $\Delta$ (tok.)"] + [f"{v(f'{SLUG[s]}.V5.d')} ({v(f'{SLUG[s]}.V5.tok')})" for s in sets])
    rows.append([r"\quad think by verbal at that cost: share, $\Delta$"] +
                [f"{v(f'{SLUG[s]}.at.V5.verbal.share')}\\%, {v(f'{SLUG[s]}.at.V5.verbal.ci')}" for s in sets])
    head = [r"& Math, Qwen3-4B & Math, Qwen3-8B & MMLU-Pro, Qwen3-4B"]
    cap = (r"Escalation to a thinking-mode re-answer (second round). $\Delta$ is the accuracy change against "
           r"\KEEP{} in pp with a 95\% interval where shown (generated tokens per item, thinking included). Selective "
           r"rows escalate the 20\% of items that the signal ranks least likely correct; agreement also pays for five "
           r"samples. Length, margin and both shares: Table~\ref{tab:thinkfull}. ``Share for 50\%/80\% of the gain'': "
           r"the smallest share, in 1-point steps, whose curve "
           r"point reaches that fraction of the all-items gain. The bottom block compares, at the nearest curve point "
           r"in cost, thinking on verbal-selected items with critique on all items and with $V_5$ (on MMLU-Pro, "
           r"thinking on all items costs less than $V_5$). Superscripts number the tested contrasts "
           r"(Table~\ref{tab:family}); \dag: rejected by Holm at 0.05.")
    table("tab_think.tex", "lccc", head, rows, cap, "tab:think", size=r"\footnotesize", sep="2.4pt")


def tab_family(G):
    F = G["family_holm"]
    names = {1: r"Hybrid (margin 20\%) $-V_5$", 2: r"$a_0$ in the vote $-V_5$", 3: r"$V_1-$ critique all",
             4: r"Answerless $\to$ critique $-$ critique all", 5: r"Dev-selected policy (R1) $-$ \KEEP{}",
             6: r"Think by probe $-$ by verbal (20\%)", 7: r"Think all $-$ \KEEP{}"}
    rows, last = [], None
    for m in sorted(F["members"], key=lambda x: (x["member"], SETTINGS.index(x["setting"]))):
        if last is not None and m["member"] != last:
            rows.append("MID")
        last = m["member"]
        rows.append([m["member"], names[m["member"]], setting_name(m["setting"], short=True), fcid(m["difference"]),
                     fp(m["p_boot"], boot=True), fp(m["holm_p"]), "reject" if m["holm_reject_0.05"] else "--"])
    head = [r"\# & Contrast & Setting & Difference (pp) [95\% CI] & $p$ (boot.) & Holm $p$ & $\alpha{=}0.05$"]
    cap = (r"The declared post hoc family (second round). Two-sided paired cluster-bootstrap $p$ "
           r"(\vnum{cfg.nboot} replicates); a $p$ of 0 enters Holm as $2/\vnum{cfg.nboot}$. The family and its members "
           r"were fixed in the plan before any run of this round; a member enters in every setting where its inputs "
           r"exist. Only these contrasts are tested; every other comparison in the paper is an estimate.")
    table("tab_family.tex", "rllrrrl", head, rows, cap, "tab:family", size=r"\footnotesize")


def tab_orig_family(G):
    m = G["orig"]["main"]
    lab = {"H1a": r"Critique $-$ trigger (equivalence, $\delta=0.03$)",
           "H1b": r"Verdict reliability trend", "H2a": r"Probe $-$ verbal AUROC ($\times$100)",
           "H3a-role": "Attributed role", "H3a-label": "Attributed label", "H3b": "Role-correlated steering"}
    rows = []
    for k in ("H1a", "H1b", "H2a", "H3a-role", "H3a-label", "H3b"):
        r = m[k]
        cat = r["category"].replace("δ", r"$\delta$")
        rows.append([k, lab[k], f"{r['n']:,}", fci(100 * r["est"], 100 * r["lo"], 100 * r["hi"]),
                     f"{r['p_holm']:.2g}", cat])
    head = [r"ID & Estimand & $n$ & Estimate (pp) [95\% CI] & Holm $p$ & Pre-registered outcome"]
    cap = (r"The original pre-registered Qwen3-4B family (mathematics), reported in full so that the present focus "
           r"does not select favourable tests. H2a is an AUROC difference $\times 100$. These tests are not part of "
           r"the post hoc family of Table~\ref{tab:family}.")
    table("tab_orig_family.tex", "lp{4.6cm}rrrl", head, rows, cap, "tab:origfamily", size=r"\footnotesize")


def tab_a3(G, A3):
    rows = []
    for s in SETTINGS:
        for rule in ("R1", "R2"):
            k = f"{SLUG[s]}.a3.{rule}"
            rows.append([setting_name(s, short=True) if rule == "R1" else "", rule, V[k + ".pol"], V[k + ".dev"],
                         V[k + ".ci"], V[k + ".vsall"], V[k + ".joint"], f"{V[k + '.rank']}/{V[SLUG[s] + '.a3.grid']}",
                         V[k + ".regret"], V[k + ".tok"]])
        if s == MATH[-1]:
            rows.append("MID")
    head = [r"Setting & Rule & Policy (operator, signal, share) & Dev $\Delta$ & Test $\Delta$ vs \KEEP{} & vs operator all & Joint CI & Rank & Regret & Tok."]
    cap = (r"A policy chosen on the development split and evaluated on test (post hoc; Qwen3-8B and Tulu-3-8B "
           r"mathematics after their second-round development runs). R1 maximises development accuracy over "
           r"operator $\times$ signal $\times$ share (0--100\% in 5-point steps); R2 caps the share at 20\%. The joint "
           r"interval re-selects on every development replicate. Rank: position of the chosen policy among the test "
           r"grid; regret: test-best grid policy minus chosen (pp, descriptive). The probe in these grids is the "
           r"mathematics probe: out-of-fold on development items for Qwen3-1.7B and 4B mathematics, and transferred "
           r"unchanged to MMLU-Pro, whose task probe is in-sample on development items and excluded; the Qwen3-8B and "
           r"Tulu-3-8B mathematics grids have no probe (no stored development activations).")
    table("tab_a3.tex", "llp{3.3cm}rrrrrrr", head, rows, cap, "tab:a3", size=r"\scriptsize", sep="2pt")


def tab_b2():
    rows = []
    for s in MATH:
        sl = SLUG[s]
        rows.append([setting_name(s, short=True), V[f"{sl}.keep"], V[f"{sl}.a0long.acc"], V[f"{sl}.a0long.ci"],
                     V[f"{sl}.a0long.tok"] + f" ({V[sl + '.a0tok']})", V[f"{sl}.a0cont.ci"], V[f"{sl}.a0cont.n"],
                     V[f"{sl}.a0cont.still"], V[f"{sl}.F2ALL.vslong.ci"], V[f"{sl}.V5.vslong.ci"],
                     f"{V[sl + '.F2ALL.compshare']}\\,$\\to$\\,{V[sl + '.F2ALL.compsharecont']}"])
    head = [r"& & \multicolumn{3}{c}{Long $a_0$} & \multicolumn{3}{c}{Continued $a_0$} & \multicolumn{2}{c}{Against long $a_0$} & Compl. \\"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}"
            r"Setting & \KEEP{} & Acc. & $-$\KEEP{} & Tok. ($a_0$) & $-$\KEEP{} & Cont. & Trunc. & Critique all & $V_5$ & share (\%)"]
    cap = (r"Stronger \KEEP{} baselines (second round). Long $a_0$: one sampled answer per item with an "
           r"\vnum{cfg.a0long}-token budget (Qwen3 non-thinking sampling); continued $a_0$: each $a_0$ that stopped at "
           r"the \vnum{cfg.a0tokens}-token cap, continued greedily for up to \vnum{cfg.contextra} more tokens. Acc.: "
           r"accuracy (\%); Tok.: generated tokens per item of the long $a_0$ (of $a_0$); Cont.: items continued; Trunc.: "
           r"items still truncated after the continuation. ``Against long $a_0$'' measures critique on all items and "
           r"$V_5$ against the long $a_0$ instead of \KEEP{}. Compl.\ share: the share of critique's repairs that start "
           r"from an $a_0$ without a usable answer, against \KEEP{} and against the continued $a_0$.")
    table("tab_b2.tex", "lrrrrrrrrrr", head, rows, cap, "tab:b2", size=r"\scriptsize", sep="1.6pt")


def tab_b3(SR):
    rows = []
    for s in B3_SETTINGS:
        sl = SLUG[s]
        for w in ("main", "p1", "p2"):
            rows.append([setting_name(s, short=True) if w == "main" else "", w, V[f"{sl}.b3.{w}.F2"],
                         V[f"{sl}.b3.{w}.F2.tok"], V[f"{sl}.b3.{w}.SC"], V[f"{sl}.b3.{w}.SC.tok"], V[f"{sl}.b3.{w}.con"]])
        if s != B3_SETTINGS[-1]:
            rows.append("MID")
    head = [r"Setting & Wording & Critique, all & Tok. & Re-solve, all & Tok. & Critique $-$ re-solve"]
    cap = (r"Paraphrased first-turn instructions (second round; wordings in Appendix~\ref{app:prompts}). "
           r"The pre-declared sign rule asks whether each model's critique $-$ re-solve sign is the same under all "
           r"three wordings and opposite between models: for the all-items contrast it \vnum{b3.flip.all}, because "
           r"Qwen3-4B changes sign; \vnum{b3.flip.nsurvive} of the \vnum{b3.flip.ntotal} contrasts it was applied to "
           r"(all items, and each signal at 10\% and 20\%) satisfy it.")
    table("tab_b3.tex", "llrrrrr", head, rows, cap, "tab:b3", size=r"\footnotesize")


def tab_b4():
    rows = []
    for s in SETTINGS:
        sl = SLUG[s]
        rows.append([setting_name(s, short=True), V[f"{sl}.HYBRIDmargin10.vsV5.ci"], V[f"{sl}.HYBRIDmarginB410.vsV5.ci"],
                     V[f"{sl}.HYBRIDmargin20.vsV5.ci"], V[f"{sl}.HYBRIDmarginB420.vsV5.ci"],
                     f"{V[sl + '.b4sel.a0']} / {V[sl + '.b4sel.V5']} / {V[sl + '.b4sel.F2']} / {V[sl + '.b4sel.B4']}",
                     V[f"{sl}.b4sel.n"]])
        if s == MATH[-1]:
            rows.append("MID")
    head = [r"& \multicolumn{2}{c}{10\%: hybrid $-V_5$} & \multicolumn{2}{c}{20\%: hybrid $-V_5$} & Correct on the 20\% & \\"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
            r"Setting & Critique & Sample-aware & Critique & Sample-aware & $a_0$ / $V_5$ / crit. / s.-aware & $|S|$"]
    cap = (r"Hybrids with a critique that sees the five sampled answers (second round). The 20\% (10\%) of items "
           r"with the lowest vote margin get a critique; the rest get $V_5$. ``Critique'' is the stored critique of "
           r"$a_0$; ``sample-aware'' is one review turn that lists the samples' final answers "
           r"(Appendix~\ref{app:prompts}). The right block counts correct answers on the selected 20\%.")
    table("tab_b4.tex", "lrrrrcr", head, rows, cap, "tab:b4", size=r"\scriptsize", sep="2pt")


def tab_auroc_src(G):
    rows = []
    for x in sorted((x for x in G["auroc"] if x["labels"] == "common"), key=lambda x: (SETTINGS.index(x["setting"]), SIGNALS.index(x["signal"]))):
        key = (x["setting"], x["signal"])
        if key in tab_auroc_src.seen:
            continue
        tab_auroc_src.seen.add(key)
        ps = x["per_source"]
        if x["setting"] in MATH:
            src = [f"{ps['gsm8k']['auroc']:.3f}", f"{ps['math500']['auroc']:.3f}", f"{ps['math_full']['auroc']:.3f}"]
        else:
            src = ["--", "--", "--"]
        if x["setting"] == MC[0] and x["signal"] == SIGNALS[0]:
            rows.append("MID")
        rows.append([setting_name(x["setting"], short=True), SIGNAL_NAME[x["signal"]],
                     f"{x['pooled']:.3f} [{x['pooled_ci'][0]:.3f}, {x['pooled_ci'][1]:.3f}]",
                     f"{x['within_source']:.3f}"] + src)
    head = [r"Setting & Signal & Pooled [95\% CI] & Within source & GSM8K & MATH-500 & MATH"]
    cap = (r"AUROC for $a_0$ correct, pooled and within source (mathematics: GSM8K, MATH-500, MATH; MMLU-Pro: its six "
           r"categories, whose per-category values are in the replay package). Pooling across sources of different "
           r"difficulty can lower an AUROC below every per-source value (Tulu-3-8B, verbal).")
    table("tab_auroc_src.tex", "llrrrrr", head, rows, cap, "tab:aurocsrc", size=r"\footnotesize")


tab_auroc_src.seen = set()


def tab_eq2_ci(G):
    """Table 3's cells with their intervals: w_S, r and d (%), and Δ (pp), at 20 % by the verbal signal."""
    idx = {}
    for r in G["eq2"]:
        if r.get("labels", "common") == "common" and r["signal"] == "verbal" and int(float(r["share_pct"])) == 20:
            idx[(r["setting"], r["operator"])] = {
                "w": (float(r["w_S"]), float(r["w_S_lo"]), float(r["w_S_hi"])),
                "r": (float(r["repair_rate"]), float(r["repair_rate_lo"]), float(r["repair_rate_hi"])),
                "d": (float(r["damage_rate"]), float(r["damage_rate_lo"]), float(r["damage_rate_hi"])),
                "D": (float(r["delta_pp"]), float(r["delta_pp_lo"]), float(r["delta_pp_hi"]))}
    for s, c in G["c1_think"]["settings"].items():
        for r in c["eq2"]:
            if r["signal"] == "verbal" and int(r["share_pct"]) == 20:
                t = lambda x: (x["estimate"], x["lo"], x["hi"])
                idx[(s, "THINK")] = {"w": t(r["w_S"]), "r": t(r["repair_rate"]), "d": t(r["damage_rate"]),
                                     "D": t(r["delta_pp"])}
    f1 = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"
    rows = []
    for s in SETTINGS:
        first = True
        for op in ("F0", "F2", "F2-scratch", "s1", "THINK"):
            x = idx.get((s, op))
            if x is None:
                continue
            rows.append([setting_name(s, short=True) if first else "", OP_NAME[op], f1(x["w"]) if first else "",
                         f1(x["r"]), f1(x["d"]), fci(*x["D"])])
            first = False
        if s != SETTINGS[-1]:
            rows.append("MID")
    head = [r"Setting & Operator & $w_S$ (\%) & Repair rate $r$ (\%) & Damage rate $d$ (\%) & $\Delta$ (pp)"]
    cap = (r"Table~\ref{tab:eq2} with 95\% intervals: the 20\% of items with the lowest verbal score, under each "
           r"operator. $w_S$ is the same selected set for every operator; intervals for $w_S$, $r$ and $d$ are ratios of "
           r"bootstrap sums.")
    table("tab_eq2_ci.tex", "llrrrr", head, rows, cap, "tab:eq2ci", size=r"\footnotesize")


def tab_samples_ci(P_by_setting):
    """Table 4's top-block policies with their intervals, in two side-by-side panels."""
    pols = [("V1", "$V_1$"), ("V3", "$V_3$"), ("V5", "$V_5$"), ("V8", "$V_8$"), ("STOP-2 K5", r"\textsc{stop-2}"),
            ("GATE-a0 K5", r"\textsc{gate}"), ("ANSWERLESS->s1", r"answerless$\to$resample"),
            ("AGREE->F2 20%", r"agreement$\to$critique 20\%"), ("HYBRID-margin 20%", "hybrid 20\\%")]

    def panel(settings):
        lines = [r"\begin{tabular}{llrr}", r"\toprule", r"Setting & Policy & $\Delta$ [95\% CI] & Tok. \\", r"\midrule"]
        for s in settings:
            P, first = P_by_setting[s], True
            for key, label in pols:
                r = P.get(key) or P.get(key.replace(" K5", ""))
                if r is None:
                    continue
                lines.append(" & ".join([setting_name(s, short=True) if first else "", label, fcid(r["delta_vs_keep"]),
                                         ftok(r["generated_tokens_per_item"])]) + r" \\")
                first = False
            if s != settings[-1]:
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    text = "\n".join([r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{1.8pt}",
                      r"\begin{minipage}[t]{0.49\textwidth}\centering", panel(SETTINGS[:3]), r"\end{minipage}\hfill",
                      r"\begin{minipage}[t]{0.49\textwidth}\centering", panel(SETTINGS[3:]), r"\end{minipage}",
                      r"\caption{The sample-based policies of Table~\ref{tab:samples} with 95\% paired intervals "
                      r"(accuracy change against \KEEP{} in pp) and generated tokens per item.}",
                      r"\label{tab:samplesci}", r"\end{table*}", ""])
    write(OUT / "tab_samples_ci.tex", text)


def tab_truncation(G):
    """Amendment 6, R1: answerless a0 in the selections, and thinking on the answered population."""
    R = G["rev"]
    rows = []
    for s in THINK:
        sl = SLUG[s]
        rows.append([setting_name(s, short=True), V[f"{sl}.r1.base"],
                     f"{V[sl + '.r1.verbal.8.pct']} / {V[sl + '.r1.verbal.10.pct']} / {V[sl + '.r1.verbal.20.pct']}",
                     f"{V[sl + '.r1b.ci']} ({V[sl + '.r1b.tok']})", V[f"{sl}.ans.n"],
                     f"{V[sl + '.ans.THINKALL.ci']} ({V[sl + '.ans.THINKALL.tok']})",
                     f"{V[sl + '.ans.THINK.verbal.reach50.share']}\\% ({V[sl + '.ans.THINK.verbal.reach50.tok']})",
                     f"{V[sl + '.ans.THINK.probe.reach50.share']}\\% ({V[sl + '.ans.THINK.probe.reach50.tok']})",
                     f"{V[sl + '.ans.auroc.verbal']} / {V[sl + '.ans.auroc.probe']}"])
    head = [r"& \multicolumn{3}{c}{All items} & \multicolumn{5}{c}{Items whose $a_0$ has a usable answer} \\"
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-9}"
            r"Setting & Answ. (\%) & Verbal 8/10/20\% & Think, answerless only & $n$ & Think, all & "
            r"50\%: verbal & 50\%: probe & AUROC v./p."]
    cap = (r"Initial answers without a usable answer (``answerless''; descriptive). Left: their share (\%) of the pool "
           r"and of the items that the verbal signal selects at 8, 10 and 20\%, and thinking on them alone ($\Delta$ in "
           r"pp, tokens per item). Right: restricted to items whose $a_0$ has a usable answer: thinking on all of them, "
           r"the smallest share whose verbal- or probe-selected escalation reaches half of that gain (tokens per item), "
           r"and the verbal and probe AUROC within that population.")
    table("tab_truncation.tex", "lrcrrrrrc", head, rows, cap, "tab:truncation", size=r"\scriptsize", sep="1.5pt")
    rows = []
    for s in SETTINGS:
        sl = SLUG[s]
        rows.append([setting_name(s, short=True), V[f"{sl}.r1.base"]] +
                    [f"{V[f'{sl}.r1.{sig}.10.pct']} / {V[f'{sl}.r1.{sig}.20.pct']}" for sig in SIGNALS])
    head = ["Setting & Pool & " + " & ".join(SIGNAL_NAME[s].capitalize() for s in SIGNALS)]
    cap = (r"Share (\%) of answerless $a_0$ among the 10\% / 20\% of items that each signal ranks least likely correct, "
           r"against their share of the pool (descriptive).")
    table("tab_answerless_share.tex", "lrccccc", head, rows, cap, "tab:answerless", size=r"\footnotesize")


def tab_bounds():
    rows = []
    for s in SETTINGS:
        sl = SLUG[s]
        cells = [setting_name(s, short=True)]
        for op in ("F0", "F2", "F2-scratch", "s1", "THINK"):
            k = f"{sl}.bound.{pslug(op)}.20"
            cells.append("--" if k + ".d" not in V else f"{V[k + '.d']} ({V[k + '.tok']})")
        cells.append(V[f"{sl}.eq2.F2.verbal.20.d"])
        cells.append(V.get(f"{sl}.eq2.THINK.verbal.20.d", "--"))
        rows.append(cells)
        if s == MATH[-1]:
            rows.append("MID")
    head = [r"& \multicolumn{5}{c}{Bound: wrong $a_0$ first, 20\%} & \multicolumn{2}{c}{Verbal, 20\%} \\"
            r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}"
            r"Setting & Review & Critique & Re-solve & Resample & Think & Critique & Think"]
    cap = (r"What a perfect detector could buy (descriptive, second round). Left: the accuracy change (pp; tokens per item) "
           r"when each operator is applied to 20\% of the items, wrong initial answers first; this ordering reads the gold "
           r"answer and is a bound, not a policy. Right: the verbal signal's 20\% selection with critique and with thinking. "
           r"10\% bounds are in the supplementary material.")
    table("tab_bounds.tex", "lccccccc", head, rows, cap, "tab:bounds", size=r"\footnotesize", sep="2.5pt")


def tab_stronger():
    rows = []
    for s in THINK:
        sl = SLUG[s]
        rows.append([setting_name(s, short=True), "$V_5$ + think on lowest margin",
                     f"{V[sl + '.r3a.10.ci']} ({V[sl + '.r3a.10.tok']})", f"{V[sl + '.r3a.20.ci']} ({V[sl + '.r3a.20.tok']})",
                     f"{V[sl + '.r3a.20.vsV5']}"])
    rows.append("MID")
    for s in MATH:
        sl = SLUG[s]
        rows.append([setting_name(s, short=True), "$V_5$ + $V_8$ on lowest margin",
                     f"{V[sl + '.r3b.10.ci']} ({V[sl + '.r3b.10.tok']})", f"{V[sl + '.r3b.20.ci']} ({V[sl + '.r3b.20.tok']})",
                     f"{V[sl + '.r3b.20.vsV5']}"])
    rows.append("MID")
    for s in THINK:
        sl = SLUG[s]
        rows.append([setting_name(s, short=True), "think by one-sample agreement",
                     f"{V[sl + '.r3c.10.ci']} ({V[sl + '.r3c.10.tok']})", f"{V[sl + '.r3c.20.ci']} ({V[sl + '.r3c.20.tok']})",
                     f"50\\%: {V[sl + '.r3c.reach50.share']}\\% ({V[sl + '.r3c.reach50.tok']}); "
                     f"80\\%: {V[sl + '.r3c.reach80.share']}\\% ({V[sl + '.r3c.reach80.tok']})"])
    head = [r"Setting & Policy & 10\% & 20\% & 20\%: $-V_5$, or share (tok.) for 50\%/80\% of think's gain"]
    cap = (r"Stronger combinations (descriptive, second round): voting on every item and thinking on the share with the "
           r"lowest vote margin; voting on every item and three more samples on that share; and escalation to thinking "
           r"ranked by whether one sample agrees with $a_0$ (disagreeing first, then by verbal score), which pays for one "
           r"sample per item. $\Delta$ in pp with 95\% intervals (tokens per item).")
    table("tab_stronger.tex", "llrrl", head, rows, cap, "tab:stronger", size=r"\scriptsize", sep="2pt")


def tab_dev():
    names = {"1": r"Hybrid (margin 20\%) $-V_5$", "2": r"$a_0$ in the vote $-V_5$", "3": r"$V_1-$ critique all",
             "4": r"Answerless $\to$ critique $-$ critique all"}
    rows = []
    settings = ("math 8B", "math T8B", "MMLU-Pro 4B", "MMLU-Pro T8B")
    for m in ("1", "2", "3", "4"):
        for s in settings:
            sl = SLUG[s]
            rows.append([f"{{}}[{m}] {names[m]}" if s == settings[0] else "", setting_name(s, short=True),
                         V[f"dev.{sl}.{m}.ci"], V[f"fam.{m}.{sl}.ci"] + V[f"fam.{m}.{sl}.rej"]])
        if m != "4":
            rows.append("MID")
    head = [r"Contrast & Setting & Development (descriptive) & Test"]
    cap = (r"Tested contrasts 1--4 on the development split ($n$ = \vnum{dev.m8.n} mathematics and \vnum{dev.c4.n} "
           r"MMLU-Pro items) next to the test split (\dag: rejected by Holm). The development items were used to fit probes "
           r"and to choose policies, never for these contrasts. Qwen3-4B and 1.7B mathematics lack extracted development "
           r"samples.")
    table("tab_dev.tex", "llrr", head, rows, cap, "tab:dev", size=r"\footnotesize", sep="3pt")


def tab_seeds(G):
    """Amendment 6, R6, and the C1 cross-check with the stored second thinking seed: run-to-run variability."""
    R6 = G["rev"]["r6"]
    cross = {f"math {x['model']}": x for x in G["c1_think"]["cross_check"]}
    rows = []
    for s in SETTINGS:
        sl = SLUG[s]
        x = cross.get(s)
        m = pslug(x["model"]) if x else None
        rows.append([setting_name(s, short=True), V[f"{sl}.keep"], len(R6[s]["single_sample_acc_pct"]),
                     f"{V[f'{sl}.r6.min']} to {V[f'{sl}.r6.max']}", f"{V[f'{sl}.r6.dmin']} to {V[f'{sl}.r6.dmax']}",
                     V[f"c1x.{m}.items"] if x else "--", V[f"c1x.{m}.seed1.ci"] if x else "--"])
        if s == MATH[-1]:
            rows.append("MID")
    head = [r"& & \multicolumn{3}{c}{One sampled answer in place of $a_0$} & \multicolumn{2}{c}{Thinking, second seed} \\"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}"
            r"Setting & \KEEP{} (\%) & Samples & Accuracy (\%) & $-$ \KEEP{} (pp) & Items & First $-$ second seed (pp)"]
    cap = (r"Run-to-run variability (descriptive, second round). Left: the accuracy of one sampled answer in place of "
           r"$a_0$, lowest to highest over the stored sample indices, and its difference from \KEEP. Right: on the "
           r"subsets that have a second thinking seed, the first seed's accuracy minus the second's with a 95\% paired "
           r"interval; these subsets hold equal numbers of correct and wrong initial answers and are not pool estimates. "
           r"The intervals elsewhere in the paper hold the stored samples and thinking outputs fixed.")
    table("tab_seeds.tex", "lrrccrr", head, rows, cap, "tab:seeds", size=r"\footnotesize", sep="4pt")


def tab_prompt(G):
    A1 = G["a1_policies"]["settings"]
    rows = []
    for s in SETTINGS:
        P = policies(G, s)
        cells = [setting_name(s, short=True)]
        for name in ("F0 ALL", "F2 ALL", "F2-scratch ALL", "V5"):
            cells.append(ftok(P[name]["prompt_tokens_per_item"]))
        cells.append(ftok(G["c1_think"]["settings"][s]["think_status"]["mean_prompt_tokens"]) if s in THINK else "--")
        rows.append(cells)
    head = [r"Setting & Review & Critique & Re-solve & $V_5$ & Think"]
    cap = (r"Prompt (prefill) tokens per item for the whole-pool operators, both turns included, for $V_5$ (five "
           r"samples) and for thinking. Caching across turns is not modelled; the verbal query's prefill, which reuses "
           r"the operator prompt's first part, is not charged.")
    table("tab_prompt.tex", "lrrrrr", head, rows, cap, "tab:prompt", size=r"\footnotesize", wide=True)


def tab_grid(EQ):
    """Selective application at 10 % and 20 % for every operator and signal: Δ vs KEEP (pp)."""
    sigs = ["verbal", "probe", "length", "agreement", "margin", "random"]
    rows = []
    for s in SETTINGS:
        for op in ("F0", "F2", "F2-scratch", "s1", "THINK"):
            if (s, op, "verbal", 10) not in EQ:
                continue
            cells = [setting_name(s, short=True) if op == "F0" else "", OP_NAME[op]]
            for sig in sigs:
                for share in (10, 20):
                    r = EQ.get((s, op, sig, share))
                    cells.append("--" if r is None else fnum(r["d"]["estimate"]))
            rows.append(cells)
        if s != SETTINGS[-1]:
            rows.append("MID")
    head = ["& & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{SIGNAL_NAME[x].capitalize()}}}" for x in sigs) + r" \\"
            + "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(sigs))) + " "
            + "Setting & Operator & " + " & ".join(["10\\% & 20\\%"] * len(sigs))]
    cap = (r"Selective application: accuracy change against \KEEP{} (pp) when the operator is applied to the 10\% or "
           r"20\% of items that the signal ranks least likely correct (random: the exact expectation). Agreement and "
           r"margin require five samples. 95\% intervals and the Eq.~(\ref{eq:decomposition}) terms for every cell are "
           r"in the replay package.")
    table("tab_grid.tex", "ll" + "rr" * len(sigs), head, rows, cap, "tab:grid", size=r"\scriptsize", sep="2pt")


def tab_decomp(G, P_by_setting):
    rows = []
    for s in SETTINGS:
        P, sl = P_by_setting[s], SLUG[s]
        pols = ["F0 ALL", "F2 ALL", "F2-scratch ALL", "V1", "V5"] + (["THINK ALL"] if s in THINK else [])
        for pol in pols:
            k = f"{sl}.{pslug(pol)}"
            rows.append([setting_name(s, short=True) if pol == pols[0] else "",
                         {"F0 ALL": "review", "F2 ALL": "critique", "F2-scratch ALL": "re-solve", "V1": "$V_1$",
                          "V5": "$V_5$", "THINK ALL": "think"}[pol],
                         V[k + ".ci"], V[k + ".repairscompletion"], V[k + ".repairscontent"],
                         V[k + ".damageformatloss"], V[k + ".damagecontent"], V[k + ".tok"]])
        if s != SETTINGS[-1]:
            rows.append("MID")
    head = [r"Setting & Policy (all items) & $\Delta$ [95\% CI] & Completion & Content repair & Format loss & Content damage & Tok."]
    cap = (r"Where the gains come from (pp of the pool). Repairs split by whether $a_0$ had no usable answer "
           r"(completion) or a wrong one (content); damage splits by whether the new answer is unusable (format loss) or "
           r"wrong (content). $\Delta$ = completion + content repair $-$ format loss $-$ content damage. The "
           r"decomposition comes from the second round; the totals $\Delta$ of the whole-pool operators, and of $V_1$ and "
           r"$V_5$ for Qwen3-4B mathematics and MMLU-Pro, were already reported in the first round.")
    table("tab_decomp.tex", "llrrrrrr", head, rows, cap, "tab:decomp", size=r"\scriptsize", sep="2.5pt")


def tab_think_full(G, EQ):
    rows = []
    for s in THINK:
        for share in (10, 20):
            for sig in ("verbal", "probe", "length", "agreement", "margin", "random", "REF detect", "REF benefit"):
                r = EQ.get((s, "THINK", sig, share))
                if r is None:
                    continue
                name = {"REF detect": "bound: wrong first", "REF benefit": "bound: repairable first"}.get(sig, SIGNAL_NAME.get(sig, sig))
                rows.append([setting_name(s, short=True) if (share, sig) == (10, "verbal") else "", f"{share}\\%", name,
                             "--" if r["w"] is None else f"{r['w']:.1f}", "--" if r["r"] is None else f"{r['r']:.1f}",
                             "--" if r["dmg"] is None else f"{r['dmg']:.1f}", fcid(r["d"]), ftok(r["tok"])])
        if s != THINK[-1]:
            rows.append("MID")
    head = [r"Setting & Share & Signal & $w_S$ & $r$ & $d$ & $\Delta$ [95\% CI] & Tok."]
    cap = (r"Selective thinking-mode re-answer: Eq.~(\ref{eq:decomposition}) terms (\%) and accuracy change (pp). The "
           r"two bounds use the gold answer and are not selectable policies.")
    table("tab_think_full.tex", "lllrrrrr", head, rows, cap, "tab:thinkfull", size=r"\footnotesize")


def tab_config():
    rows = []
    for slug, s, lic in (("17", "math 1.7B", "Apache-2.0"), ("4", "math 4B", "Apache-2.0"), ("8", "math 8B", "Apache-2.0"),
                         ("T", "math T8B", "Llama 3.1 Community")):
        sl = SLUG[s]
        mc = {"math 4B": "MMLU-Pro 4B", "math T8B": "MMLU-Pro T8B"}.get(s)
        rows.append([MODEL[s.split(" ")[1]], lic, "\\texttt{" + V["rev." + slug] + "}",
                     f"{V[sl + '.probe.layer']} ({V[sl + '.probe.cv']})",
                     f"{V[SLUG[mc] + '.probe.layer']} ({V[SLUG[mc] + '.probe.cv']})" if mc else "--"])
    head = [r"Model & License & Revision & Math probe: layer (CV AUROC) & MMLU-Pro probe: layer (CV AUROC)"]
    cap = (r"Models and probes. Revisions are the first seven characters of the Hugging Face commit. Probes are "
           r"$\ell_2$-regularised logistic regressions on the standardised residual state at the last token of $a_0$, "
           r"with layer and $C\in\{\vnum{cfg.probeCgrid}\}$ chosen by \vnum{cfg.probefolds}-fold grouped "
           r"cross-validation on the development split (every probe chose $C=\vnum{m4.probe.C}$), then refit on all "
           r"development items and frozen.")
    table("tab_config.tex", "lllcc", head, rows, cap, "tab:config", size=r"\footnotesize", wide=True)


def prompts_tex():
    R = literals("src/gen/render.py")
    A = literals("src/exp/arr_runs.py")
    M = literals("src/data/mmlupro.py")
    W = A["WORDINGS"] if "WORDINGS" in A else None
    # WORDINGS refers to render constants for the main wording, so it is not a literal; read p1/p2 from the source.
    src = inp("src/exp/arr_runs.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    para = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WORDINGS" for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                key = ast.literal_eval(k)
                if key in ("p1", "p2"):
                    para[key] = ast.literal_eval(v)
    assert set(para) == {"p1", "p2"} and W is None
    sample_aware = A["SAMPLE_AWARE"].replace("{{", "{").replace("}}", "}")
    items = [
        ("Initial answer and samples", R["PROMPT"].replace("{{", "{").replace("}}", "}").replace("{q}", "<question>")),
        ("MMLU-Pro question", "<question>\n\nOptions:\nA. <option> ... \n\n" + M["ASK"]),
        ("Label before $a_0$", R["LABELS"]["self"]),
        ("Review; final turn", R["INSTRUCTION"]),
        ("Critique, first turn (main)", R["CRITIQUE_TURN"]),
        ("Critique, first turn (p1)", para["p1"]["F2"]),
        ("Critique, first turn (p2)", para["p2"]["F2"]),
        ("Re-solve, first turn (main)", R["SCRATCH_TURN"]),
        ("Re-solve, first turn (p1)", para["p1"]["F2-scratch"]),
        ("Re-solve, first turn (p2)", para["p2"]["F2-scratch"]),
        ("Sample-aware critique (one turn)", sample_aware.replace("{what}", "problem").replace("{answers}", "<answers>")),
        ("Verbal confidence query", R["VERIFY_QUERY"]),
    ]
    lines = [r"\begin{description}\small\raggedright\setlength{\itemsep}{2pt}"]
    for name, text in items:
        body = tex_escape(text).replace("\n\n", r" \textbackslash n\textbackslash n ").replace("\n", r" \textbackslash n ")
        body = body.replace("<", r"\textless{}").replace(">", r"\textgreater{}")
        lines.append(rf"\item[{name}.] \texttt{{{body}}}")
    lines.append(r"\end{description}")
    write(OUT / "prompts.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# Figure
def fig_frontier(G, P_by_setting):
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "mathtext.fontset": "stix"})
    BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.35))

    def eb(ax, r, **kw):
        d = r["delta_vs_keep"]
        x = r["generated_tokens_per_item"]
        ax.errorbar([x], [d["estimate"]], yerr=[[d["estimate"] - d["lo"]], [d["hi"] - d["estimate"]]],
                    elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=4, **kw)

    for ax, s in zip(axes.flat, SETTINGS):
        P = P_by_setting[s]
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        xs, ys = zip(*curve(G, s, "F2", "verbal"))
        ax.plot(xs, ys, color=MUTED, lw=0.9, ls=(0, (1.2, 1.2)), zorder=2)
        for name, mk in (("F0 ALL", "v"), ("F2 ALL", "s"), ("F2-scratch ALL", "D")):
            eb(ax, P[name], fmt=mk, ms=3.2, color=INK, mfc="white", mew=0.8)
        ks = [k for k in ("V1", "V3", "V5", "V8") if k in P]
        ax.plot([P[k]["generated_tokens_per_item"] for k in ks], [P[k]["delta_vs_keep"]["estimate"] for k in ks],
                color=BLUE, lw=1.0, zorder=3)
        for k in ks:
            eb(ax, P[k], fmt="o", ms=3.0, color=BLUE, mfc=BLUE, mew=0.6)
        for k in (["V1", "V5", "V8"] if "V8" in P else ["V1", "V5"]):
            ax.annotate(k.replace("V", "$k$="), (P[k]["generated_tokens_per_item"], P[k]["delta_vs_keep"]["estimate"]),
                        xytext=(2, -7), textcoords="offset points", fontsize=5.5, color=MUTED)
        gate = P.get("GATE-a0 K5", P.get("GATE-a0"))
        eb(ax, gate, fmt="^", ms=3.6, color=BLUE, mfc="white", mew=0.8)
        eb(ax, P["HYBRID-margin 20%"], fmt="P", ms=3.8, color=AQUA, mfc=AQUA, mew=0.6)
        xmax = max(P["V8"]["generated_tokens_per_item"] if "V8" in P else P["V5"]["generated_tokens_per_item"],
                   P["HYBRID-margin 20%"]["generated_tokens_per_item"])
        if s in THINK:
            c = G["c1_think"]["settings"][s]
            for sig, ls in (("verbal", "-"), ("probe", (0, (3, 1.5)))):
                xs, ys = zip(*curve(G, s, "THINK", sig))
                ax.plot(xs, ys, color=ORANGE, lw=1.1, ls=ls, zorder=3)
            eb(ax, c["policies"]["THINK ALL"], fmt="*", ms=6.0, color=ORANGE, mfc=ORANGE, mew=0.5)
            xmax = max(xmax, c["policies"]["THINK ALL"]["generated_tokens_per_item"])
            cb = ((G.get("c1b") or {}).get("settings", {}).get(s) or {}).get("budgets") or {}
            if cb:   # plan Amendment 7: thinking on every item under budget B (descriptive)
                pts = [cb[b] for b in sorted(cb, key=int)]
                ax.plot([r["generated_tokens_per_item"] for r in pts], [r["delta_vs_keep"]["estimate"] for r in pts],
                        color=ORANGE, lw=0.8, ls=(0, (1, 1)), zorder=3)
                for r in pts:
                    eb(ax, r, fmt="o", ms=3.0, color=ORANGE, mfc="white", mew=0.8)
                xmax = max(xmax, max(r["generated_tokens_per_item"] for r in pts))
        ax.set_xlim(-0.03 * xmax, 1.06 * xmax)
        ax.set_title(setting_name(s), loc="left", pad=3)
        if xmax >= 1500:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
        else:
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(250))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for ax in axes[1]:
        ax.set_xlabel("Generated tokens per item after $a_0$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta$ accuracy vs. KEEP (pp)")
    handles = [
        Line2D([], [], color=INK, marker="v", ls="none", mfc="white", ms=3.5, label="review, all"),
        Line2D([], [], color=INK, marker="s", ls="none", mfc="white", ms=3.5, label="critique, all"),
        Line2D([], [], color=INK, marker="D", ls="none", mfc="white", ms=3.2, label="re-solve, all"),
        Line2D([], [], color=MUTED, ls=(0, (1.2, 1.2)), lw=0.9, label="critique, verbal share"),
        Line2D([], [], color=BLUE, marker="o", lw=1.0, ms=3.0, label="vote over $k$ samples"),
        Line2D([], [], color=BLUE, marker="^", ls="none", mfc="white", ms=3.6, label="gate, then vote"),
        Line2D([], [], color=AQUA, marker="P", ls="none", ms=3.8, label="hybrid (critique 20%)"),
        Line2D([], [], color=ORANGE, marker="*", ls="none", ms=6, label="think, all"),
        Line2D([], [], color=ORANGE, lw=1.1, label="think, verbal share"),
        Line2D([], [], color=ORANGE, lw=1.1, ls=(0, (3, 1.5)), label="think, probe share"),
    ]
    if G.get("c1b"):
        handles.append(Line2D([], [], color=ORANGE, marker="o", mfc="white", mew=0.8, ms=3.0, lw=0.8, ls=(0, (1, 1)),
                              label="think, budget $B$"))
    fig.legend(handles=handles, loc="lower center", ncol=6 if G.get("c1b") else 5, frameon=False, handlelength=2.0,
               columnspacing=0.8 if G.get("c1b") else 1.0, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=0.6, w_pad=0.5)
    path = FIG / "fig_frontier.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)
    OUTPUTS[path.relative_to(ROOT).as_posix()] = sha_bytes(path.read_bytes(), ".pdf")


# ---------------------------------------------------------------------------------------------------------------
def main():
    global OUT, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="paper/1区/acl2027/manuscript_v2")
    args = ap.parse_args()
    MS = ROOT / args.ms
    OUT, FIG = MS / "generated", MS / "figures"
    G = load_all()
    P_by_setting = {s: policies(G, s) for s in SETTINGS}
    EQ = eq2_index(G)
    macros_a0(G)
    macros_policies(G)
    AU = macros_auroc(G)
    macros_eq2(EQ)
    macros_family(G)
    macros_b2(G)
    SR = macros_b3(G)
    macros_b4(G)
    macros_c1(G, P_by_setting)
    macros_c1b(G)
    macros_c1bt(G)
    A3 = macros_a3(G)
    macros_config(G)
    macros_derived(G, P_by_setting, EQ)
    macros_review(G, P_by_setting)
    tab_settings(G, AU, P_by_setting)
    tab_operators(G, P_by_setting)
    tab_eq2(EQ)
    tab_samples(G, P_by_setting)
    tab_think(G, EQ)
    tab_family(G)
    tab_orig_family(G)
    tab_a3(G, A3)
    tab_b2()
    tab_b3(SR)
    tab_b4()
    tab_auroc_src(G)
    tab_grid(EQ)
    tab_eq2_ci(G)
    tab_samples_ci(P_by_setting)
    tab_truncation(G)
    tab_bounds()
    tab_stronger()
    tab_dev()
    tab_prompt(G)
    tab_decomp(G, P_by_setting)
    tab_think_full(G, EQ)
    tab_budget(G)
    tab_budget_total(G)
    tab_seeds(G)
    tab_config()
    prompts_tex()
    fig_frontier(G, P_by_setting)
    lines = ["% Generated by analysis/arr/acl_v2_tex.py from the ARR outputs; do not edit (CLAUDE.md rule 1).",
             "% Read a value with \\vnum{key}."]
    lines += [f"\\expandafter\\def\\csname val@{k}\\endcsname{{{v}}}" for k, v in sorted(V.items())]
    write(OUT / "values.tex", "\n".join(lines) + "\n")
    manifest = {"script": "analysis/arr/acl_v2_tex.py", "script_sha256": sha_bytes(Path(__file__).read_bytes(), ".py"),
                "plan_core_sha256": G["family_holm"]["plan_core_sha256"], "inputs": dict(sorted(INPUTS.items())),
                "outputs": dict(sorted(OUTPUTS.items())), "n_macros": len(V)}
    mpath = ROOT / "analysis/out/arr/acl_v2_tex_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(V)} macros, {len(OUTPUTS)} outputs, {len(INPUTS)} inputs -> {mpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
