"""Result numbers and tables for the paper (P7; CLAUDE.md rule 1): reads the generated analysis files and the result rows,
writes LaTeX. No new statistics beyond per-condition means; every CI comes from the P4/P6 analysis files.
-> paper/generated/results.tex   key-value macros, used in the text as \\res{key} (main.tex defines \\res; a missing key
                                 prints a red marker)
   paper/generated/tab_*.tex     tables
Keys:
  c.<scope>.<H>.<field>      analysis/out/06_confirmatory_4B.json, scope main | v2 | a0_ok
  x.<model>.<path>           analysis/out/06_exploratory_<model>.json
  t.<model>.<H>.<field>      analysis/out/p4_<model>_tier1.json estimates (1.7B, 8B)
  e5.<path>, ch.<path>, e4.<path>, th.<model>.<path>, st.<path>
For a CI triple [est, lo, hi] (fractions) at <path>: <path>.d "+1.2 [+0.3, +2.1]" (pp), <path>.r "12.3 % [10.1, 14.2]",
<path>.e "+1.2", <path>.u "1.2" (×100), <path>.lo / .hi (×100, signed). Scalars: <path> as stored (3 significant
decimals for floats). Per-condition means of 1A: m.<model>.1A.<cond>.{acc,wr,rw,trunc,tok}.
Usage: python analysis/p7_results_tex.py
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from p6_confirmatory import jl, load  # noqa: E402

OUT, GEN = ROOT / "analysis" / "out", ROOT / "paper" / "generated"
SPECS = {"4B": "configs/p6_spec_4B_test.json", "1.7B": "configs/p4_spec_1.7B_test.json", "8B": "configs/p4_spec_8B_test.json"}
COND_1A = ["B0", "B1", "F0", "F1", "F2", "F2-scratch", "F3(0.5)", "F3(0.7)", "F3(0.9)", "F3(1.0)", "F3-perm[0]",
           "F3-perm[1]", "F3-perm[2]", "F5"]
K = {}


def tex_escape(s):
    s = str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    return s.replace("δ", r"$\delta$").replace("−", "-").replace("×", r"$\times$")


def num(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}".replace(",", "{,}")
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}" if abs(v) < 1 else f"{v:.2f}"


def pval(p):
    if p is None:
        return "--"
    if p >= 0.001:
        return f"{p:.3g}"
    m, e = f"{p:.1e}".split("e")
    return rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"


def is_triple(v):
    return isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def put_triple(key, v, scale=100.0, dec=1):
    e, lo, hi = (scale * x for x in v)
    K[key + ".d"] = f"{e:+.{dec}f} [{lo:+.{dec}f}, {hi:+.{dec}f}]"
    K[key + ".r"] = rf"{e:.{dec}f}\,\% [{lo:.{dec}f}, {hi:.{dec}f}]"
    K[key + ".e"] = f"{e:+.{dec}f}"
    K[key + ".u"] = f"{e:.{dec}f}"
    K[key + ".lo"], K[key + ".hi"] = f"{lo:+.{dec}f}", f"{hi:+.{dec}f}"


def flatten(prefix, obj):
    if isinstance(obj, dict):
        if {"est", "lo", "hi"} <= set(obj) and all(isinstance(obj[k], (int, float)) for k in ("est", "lo", "hi")):
            put_triple(prefix, [obj["est"], obj["lo"], obj["hi"]], dec=2 if prefix.startswith("c.") else 1)
        tri = {"est", "lo", "hi"} <= set(obj)
        for k, v in obj.items():
            if tri and k in ("est", "lo", "hi"):
                continue
            flatten(f"{prefix}.{k}", v)
            if prefix.endswith(".context") and isinstance(v, float):
                K[f"{prefix}.{k}.pct"] = f"{100 * v:.1f}" 
    elif is_triple(obj):
        if prefix.endswith("gain ratio"):
            put_triple(prefix, obj, scale=1.0, dec=2)
        else:
            put_triple(prefix, obj)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        K[prefix] = num(obj) if not prefix.endswith(("p", "p_holm", "delong_p")) else pval(obj)
    elif isinstance(obj, str) and len(obj) < 80:
        K[prefix] = tex_escape(obj)


def jload(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cond_means(model):
    spec = json.loads((ROOT / SPECS[model]).read_text(encoding="utf-8"))
    gold, canon, strat, init, verbal = load(spec)
    rows = []
    for run in [spec["exp01_orig"]] + ([spec["exp01_ext"]] if spec.get("exp01_ext") else []):
        rows += [r for r in jl(Path(run) / "rows.jsonl") if r["exp"] == "1A" and r.get("stage") != 1]
    by = defaultdict(list)
    for r in rows:
        if r["data"] == "dnat" and r["item_id"] in init:
            by[r["cond"]].append(r)
        elif r["data"] == "dinj":
            by["F4 (D-inj)"].append(r)
    out = {}
    for c, rs in by.items():
        c0 = [bool(init[r["item_id"]]["correct"]) if r["data"] == "dnat" else False for r in rs]
        ok = [bool(r["correct"]) for r in rs]
        wr = [o for o, i in zip(ok, c0) if not i]
        rw = [not o for o, i in zip(ok, c0) if i]
        toks = [r["out_tokens"] for r in rs if r.get("out_tokens") is not None]
        out[c] = {"n": len(rs), "acc": sum(ok) / len(rs), "wr": sum(wr) / len(wr) if wr else None,
                  "rw": sum(rw) / len(rw) if rw else None,
                  "trunc": sum(r.get("status") == "truncated" for r in rs) / len(rs),
                  "tok": sum(toks) / len(toks) if toks else None}
    return out


def pct(v):
    return "--" if v is None else f"{100 * v:.1f}"


def ci_pct(v, signed=False):
    """A CI triple of fractions as 'e [lo, hi]' in % (or signed pp)."""
    if v is None:
        return "--"
    f = "+.1f" if signed else ".1f"
    return f"{100 * v[0]:{f}} [{100 * v[1]:{f}}, {100 * v[2]:{f}}]"


def span(vals, fmt="{:.0f}"):
    """'a--b' over the values (or one value when they agree); None counts as 'none'."""
    if any(v is None for v in vals):
        return "none" if all(v is None for v in vals) else "mixed"
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return lo if lo == hi else f"{lo}--{hi}"


def write_table(name, lines):
    (GEN / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def followups():
    """Post-freeze follow-ups (exploratory; DEVIATIONS D14-D16): keys fu0.*, fu1.*, fu3.*, lore.* and their tables.
    Inputs: analysis/out/fu_01_reanalysis.json, fu_02_group1.json, fu_03_group2.json, lore_02_results.json, fu_04_verbal_check.json,
    06_exploratory_4B.json."""
    fu0, fu1, fu3, lore = (jload(n) for n in ("fu_01_reanalysis", "fu_02_group1", "fu_03_group2", "lore_02_results"))
    x4 = jload("06_exploratory_4B")
    flatten("fu0", fu0)
    flatten("fu1", fu1)
    flatten("fu3.steer", fu3["steer"])
    flatten("lore", lore)
    # "%" would start a comment inside \res{...}: alias such keys with "pct".
    for k in [k for k in K if k.startswith(("fu0.", "fu1.")) and "%" in k]:
        K[k.replace("%", "pct")] = K[k]
    budgets = [b for b in fu0["re-check policy"]["actions"]["F2"] if b.endswith("%")]
    K["fu0.recheck.first budget"] = budgets[0].rstrip("%")
    K["lore.dinj.name.wr.min"] = f"{100 * min(v['W->R'][0] for v in lore['dinj']['name (primary)']['cells'].values()):.1f}"
    fu4 = jload("fu_04_verbal_check")
    flatten("fu4", fu4)
    for m, d in fu4.items():
        for split, r in d.items():
            K[f"fu4.{m}.{split}.a0 pct"] = f"{100 * r['a0 accuracy']:.1f}"
    # Design constants of the follow-up runs, read from their configs and scripts (CLAUDE.md rule 1).
    cfg = lambda rel: json.loads((ROOT / rel / "config.json").read_text(encoding="utf-8"))  # noqa: E731
    cc = cfg("results/fu/4B/fu-h2a-consistency-test-4B-aac4cc6")
    K["fu.d.cons.k"], K["fu.d.cons.temp"], K["fu.d.cons.topp"], K["fu.d.cons.topk"] = (
        str(cc["k"]), str(cc["temperature"]), str(cc["top_p"]), str(cc["top_k"]))
    hc = cfg("results/fu/4B/fu-h3c-test-4B-aac4cc6")
    K["fu.d.h3c.skip"], K["fu.d.h3c.folds"], K["fu.d.h3c.C"] = str(hc["skip_first"]), str(hc["folds"]), str(hc["C"])
    sc = cfg("results/fu3/main/4B/main-streams-test-controls-4B-1809070")
    K["fu.d.streams.steps"], K["fu.d.streams.seeds"] = str(sc["steps"]), str(len(sc["seeds"]))
    m = re.search(r"assert bad <= ([0-9.]+)", (ROOT / "cloud" / "fu2_sequence.sh").read_text(encoding="utf-8"))
    K["fu.d.t8b.feas"] = f"{100 * float(m.group(1)):.0f}"
    dev = [json.loads(line) for rel in ("results/fu2/p1/T8B/p1-initial-T8B-1809070", "results/fu2/p1/T8B/p1-initial-ext-T8B-1809070")
           for line in open(ROOT / rel / "initial.jsonl", encoding="utf-8")]
    dev = [r for r in dev if r["split"] == "dev"]
    K["fu.t8b.dev.notok"] = f"{100 * sum(r['status'] != 'ok' for r in dev) / len(dev):.1f}"
    # Streams: per-seed lists (not CI triples), so no flatten.
    st = fu3["streams"]
    ks = sorted(st["eval"]["self"], key=int)
    K["fu3.st.base identical"] = "yes" if st["base eval identical across runs"] else "no"
    K["fu3.st.kmax"] = ks[-1]
    for c, v in st["eval"].items():
        last = v[ks[-1]]
        K[f"fu3.st.{c}.final"] = ", ".join(f"{100 * a:.0f}" for a in last["acc_greedy"])
        K[f"fu3.st.{c}.final.mean"] = f"{100 * sum(last['acc_greedy']) / len(last['acc_greedy']):.0f}"
        K[f"fu3.st.{c}.k25"] = ", ".join(f"{100 * a:.0f}" for a in v["25"]["acc_greedy"])
        K[f"fu3.st.{c}.trunc.mean"] = f"{100 * sum(last['truncated']) / len(last['truncated']):.0f}"
    by = defaultdict(list)
    for t, v in st["trajectories"].items():
        by[v["condition"]].append(v)
    for c, vs in by.items():
        K[f"fu3.st.{c}.onset"] = span([v["collapse onset"] for v in vs])
        K[f"fu3.st.{c}.updates"] = span([v["updates applied"] for v in vs])
        if vs[0]["incorrect verdicts used"] is not None:
            K[f"fu3.st.{c}.incorrect"] = span([100 * v["incorrect verdicts used"] for v in vs])
        if vs[0]["UL share of applied"] is not None:
            K[f"fu3.st.{c}.ul"] = span([100 * v["UL share of applied"] for v in vs])

    # ---- table: stream controls
    T = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         rf"Condition & Acc $k{{=}}25$ & Acc $k{{=}}{ks[-1]}$ & Trunc. & UL share & Onset \\", r"\midrule"]
    for c in ("self", "matched", "ext", "nll", "random", "context"):
        if c not in st["eval"]:
            continue
        a25 = st["eval"][c]["25"]["acc_greedy"]
        T.append(f"{c} & {100 * sum(a25) / len(a25):.0f} & {K[f'fu3.st.{c}.final.mean']} [{K[f'fu3.st.{c}.final']}] & "
                 f"{K[f'fu3.st.{c}.trunc.mean']} & {K.get(f'fu3.st.{c}.ul', '--')} & {K[f'fu3.st.{c}.onset']} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Persistent-stream controls (4B, {len(by['self'])} order seeds per condition, {ks[-1]} steps). Greedy "
          r"dev accuracy (\%) at the first checkpoint and at the last (mean [per seed]), truncated outputs at the last "
          r"checkpoint (\%, mean), the unlikelihood share of applied updates (\%, range over seeds) and the collapse "
          r"onset (first step of 10 consecutive truncated $a_0$). nll and matched are the follow-up controls; the other "
          r"rows are the frozen run. The base evaluation of the two runs is identical. No CI: the trajectory is the unit. "
          r"From \texttt{analysis/out/fu\_03\_group2.json}.}", r"\label{tab:fu-streams}", r"\end{table*}"]
    write_table("tab_fu_streams", T)

    # ---- table: structured steering controls
    s = fu3["steer"]
    T = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Arm & \WtoR{} & main $-$ arm & KL & cos \\", r"\midrule"]
    for a in ("unsteered", "main", "pc1", "dataset", "flip"):
        v = s["arms"][a]
        mm = ci_pct(s["main minus"][a]["W->R"], signed=True) if a in s["main minus"] else "--"
        kl = s["precheck"]["mean_kl"].get(a)
        cs = s["precheck"]["cos_with_role"].get(a)
        T.append(f"{a} & {ci_pct(v['W->R'])} & {mm} & {'--' if kl is None else f'{kl:.4f}'} & "
                 f"{'--' if cs is None else f'{cs:+.2f}'} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Structured steering controls (4B, layer and $\alpha$ of the main setting, D-inj test $n = "
          rf"{num(s['n dinj'])}$). \WtoR{{}} in \%, main minus each arm in pp (paired, 95\% cluster-bootstrap CIs). KL: mean "
          rf"next-token KL at the solution positions on the first {s['precheck']['n']} test items (pre-check); cos: cosine "
          r"with the role direction. Directions are computed on dev. pc1: first principal component of "
          r"the assistant/self states; dataset: gsm8k\_sc $-$ prm800k\_sc mean difference; flip: the role construction "
          r"with item-level random signs. Every arm has the main arm's perturbation norm. Main and unsteered reproduce the "
          rf"frozen run on {num(s['reproduction']['identical'])} of {num(s['reproduction']['compared'])} rows. From "
          r"\texttt{analysis/out/fu\_03\_group2.json}.}", r"\label{tab:fu-steer}", r"\end{table*}"]
    write_table("tab_fu_steer", T)

    # ---- table: H2a comparators (4B)
    g = fu1["H2a comparators"]
    xc = x4["H2a comparators"]
    dv = fu0["H2a dev-fitted comparator"]
    rows = [("Probe (dev-trained, frozen)", g["auroc"]["probe"], None),
            ("Verbal score $s$", g["auroc"]["verbal s"], g["probe minus"]["verbal s"]),
            ("Answer length (shorter ranks higher)", xc["auroc"]["minus answer length"], xc["probe minus"]["minus answer length"]),
            ("Stratum dev accuracy", xc["auroc"]["stratum dev accuracy"], xc["probe minus"]["stratum dev accuracy"]),
            ("Mean token log-prob of $a_0$", g["auroc"]["mean token log-prob"], g["probe minus"]["mean token log-prob"]),
            ("Min token log-prob of $a_0$", g["auroc"]["min token log-prob"], g["probe minus"]["min token log-prob"]),
            ("Agreement with $a_0$ (5 samples)", g["auroc"]["5-sample consistency"], g["probe minus"]["5-sample consistency"]),
            ("Dev-fitted: $s$, length, strata", dv["auroc"]["dev-fitted combined"], dv["probe - combined"]),
            ("Dev-fitted: + log-prob, agreement", g["dev-fitted combined"]["auroc"], g["dev-fitted combined"]["probe minus"])]
    T = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrr}", r"\toprule",
         r"Predictor of $a_0$ correctness & AUROC & Probe $-$ it \\", r"\midrule"]
    for lab, au, d in rows:
        T.append(f"{lab} & {au:.3f} & {'--' if d is None else ci_pct(d, signed=True)} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{H2a comparators (4B, pooled D-nat test, $n = {num(g['n'])}$, {num(g['n wrong'])} initially wrong). "
          r"Probe minus each predictor in AUROC points $\times 100$ (paired 95\% cluster-bootstrap CIs). The dev-fitted "
          rf"rows are logistic regressions fitted on the {num(dv['n dev'])} dev items and frozen. Agreement with $a_0$ is "
          r"the share of 5 sampled answers that the scorer judges equivalent to $a_0$ (not the samples' mutual agreement, "
          r"and not semantic entropy). Exploratory, after the confirmatory results. From "
          r"\texttt{analysis/out/fu\_02\_group1.json}, \texttt{fu\_01\_reanalysis.json} and "
          r"\texttt{06\_exploratory\_4B.json}.}", r"\label{tab:fu-h2a}", r"\end{table*}"]
    write_table("tab_fu_h2a", T)

    # ---- table: stated reliability
    sp = fu1["stated p"]
    T = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"$p$ & Acc, $p$ not stated & Acc, $p$ stated & \WtoR{}, not st. & \WtoR{}, stated \\", r"\midrule"]
    for p, v in sp["per p"].items():
        T.append(f"{p} & {ci_pct(v['unstated']['accuracy'])} & {ci_pct(v['stated']['accuracy'])} & "
                 f"{100 * v['unstated']['W->R'][0]:.1f} & {100 * v['stated']['W->R'][0]:.1f} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Stated verifier reliability (4B, pooled D-nat test, $n = {num(sp['n'])}$; \%). The stated wrapper "
          r"adds the checker's accuracy $p$ to the verdict sentence; the verdict draws are those of the frozen F3($p$) "
          rf"run ({num(sp['verdict mismatches (must be 0)'])} mismatches). 95\% cluster-bootstrap CIs. From "
          r"\texttt{analysis/out/fu\_02\_group1.json}.}", r"\label{tab:fu-stated}", r"\end{table*}"]
    write_table("tab_fu_stated", T)

    # ---- table: one factorial decomposition on D-inj (4B)
    f = fu0["D-inj factorial"]["effects"]
    T = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lr}", r"\toprule",
         r"Within-item effect on \WtoR{} & pp [95\% CI] \\", r"\midrule"]
    for k, v in f.items():
        T.append(f"{tex_escape(k)} & {ci_pct(v, signed=True)} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Role $\times$ label $\times$ feedback-regime factorial on one item set (4B, D-inj test, $n = "
          rf"{num(fu0['D-inj factorial']['n'])}$; every item initially wrong, so accuracy = \WtoR{{}}): role $\times$ label "
          r"$\times$ feedback (F0, F1, F3(1.0)) from the Exp~1B rows. Exploratory re-analysis of existing outputs. From "
          r"\texttt{analysis/out/fu\_01\_reanalysis.json}.}", r"\label{tab:fu-dinj}", r"\end{table*}"]
    write_table("tab_fu_dinj", T)

    # ---- table: lore QA (4B)
    pr = lore["dinj"]["name (primary)"]
    T = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrr}", r"\toprule",
         r"D-inj-lore, name rows, F0 & \WtoR{} & \RtoW{} \\", r"\midrule"]
    for c, v in pr["cells"].items():
        T.append(f"{tex_escape(c.removesuffix('/F0'))} & {ci_pct(v['W->R'])} & {ci_pct(v['R->W'])} \\\\")
    T.append(r"\midrule")
    for c in pr["W->R contrasts"]:
        T.append(f"{tex_escape(c)} & {ci_pct(pr['W->R contrasts'][c], signed=True)} & "
                 f"{ci_pct(pr['R->W contrasts'][c], signed=True)} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Lore QA (4B, test, source excerpts up to {num(lore['budget'])} tokens): the primary pool, injected-answer rows with "
          rf"name-class answers ({pr['n wrong rows']} injected-wrong rows for \WtoR{{}}, {pr['n right rows']} injected-right "
          r"rows for \RtoW{}). Cells in \%, contrasts in pp, 95\% cluster-bootstrap CIs. Estimation only (frozen "
          r"addendum). From \texttt{analysis/out/lore\_02\_results.json}.}", r"\label{tab:fu-lore}", r"\end{table*}"]
    write_table("tab_fu_lore", T)

    # ---- second review (D18): readout variants (fu_07), H3c regrouped (fu_08)
    fu7, fu8 = jload("fu_07_verbal_variants"), jload("fu_08_h3c_regroup")
    if fu7:
        flatten("fu7", fu7)
        for m, o in fu7.items():
            alt = [v for v in o["variants"] if v != "W0" and o["variants"][v]["auroc"] is not None]
            best = max(alt, key=lambda v: o["variants"][v]["auroc"])
            K[f"fu7.{m}.best alternative"] = best
            K[f"fu7.{m}.best alternative.auroc"] = f"{o['variants'][best]['auroc']:.3f}"
            K[f"fu7.{m}.selected"] = o["dev-selected variant"] or "none"
    if fu8:
        flatten("fu8", fu8)
        K["fu8.maxdiff"] = f"{max(v['largest per-layer |difference|'] for v in fu8['by target'].values()):.3f}"
    # ---- validation on MMLU-Pro (frozen addendum docs/planning/validation_v2/addendum.md; D18)
    v2 = jload("v2_02_mc_results")
    if v2:
        v2m = {m: o for m, o in v2["models"].items() if o}
        flatten("v2", {"models": v2m, **({"T8B - 4B": v2["T8B - 4B"]} if "T8B - 4B" in v2 else {})})
        for m, o in v2m.items():
            K[f"v2.{m}.a0 pct"] = f"{100 * o['a0 accuracy']:.1f}"
            for c, v in o["conditions"].items():
                K[f"v2.{m}.{c}.wr"], K[f"v2.{m}.{c}.rw"] = f"{100 * v['W->R']:.1f}", f"{100 * v['R->W']:.1f}"
                K[f"v2.{m}.{c}.gen"] = f"{v['generated tokens / item']:.0f}"
            for b, row in o["policies"]["F2"]["budgets"].items():
                for name, v in row.items():
                    K[f"v2.{m}.F2.{b}.{name}.gen"] = f"{v['incremental generated / item']:.0f}"
        models = list(v2m)
        T = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "rrrr" * len(models) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{m} (a$_0$ {100 * v2m[m]['a0 accuracy']:.1f}\%)}}" for m in models) + r" \\",
             "Condition & " + " & ".join(r"Net (pp) & \WtoR{} & \RtoW{} & Tokens" for _ in models) + r" \\", r"\midrule"]
        for c in v2m[models[0]]["conditions"]:
            T.append(f"{tex_escape(c)} & " + " & ".join(
                f"{ci_pct(v2m[m]['conditions'][c]['net change'], signed=True)} & {100 * v2m[m]['conditions'][c]['W->R']:.1f} & "
                f"{100 * v2m[m]['conditions'][c]['R->W']:.1f} & {v2m[m]['conditions'][c]['generated tokens / item']:.0f}" for m in models) + r" \\")
        T += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Independent validation on MMLU-Pro (six non-computational categories, test, $n = {num(v2m[models[0]]['n'])}$), under "
              r"an addendum frozen before any test output. Net accuracy change against $a_0$ in pp (95\% cluster-bootstrap CIs, "
              r"clusters = question stem), \WtoR{} and \RtoW{} in \%, generated tokens per item counting both turns of F2 and "
              r"F2-scratch. From \texttt{analysis/out/v2\_02\_mc\_results.json}.}", r"\label{tab:v2-conds}", r"\end{table*}"]
        write_table("tab_v2_conds", T)
        names = ["verbal W0", "task probe", "math probe", "answer length", "agreement with a0", "dev-fitted net gain, cheap", "random (expected)"]
        T = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "rrr" * len(models) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{m}, 10\% of items}}" for m in models) + r" \\",
             "Ranking & " + " & ".join(r"Acc. change (pp) & tokens & $-$ verbal (pp)" for _ in models) + r" \\", r"\midrule"]
        for name in names:
            cells = []
            for m in models:
                v = v2m[m]["policies"]["F2"]["budgets"]["10"].get(name)
                if v is None:
                    cells += ["--", "--", "--"]
                    continue
                mv = v.get("minus verbal W0")
                cells += [ci_pct(v["acc change"], signed=True), f"{v['incremental generated / item']:.0f}", ci_pct(mv, signed=True) if mv else "--"]
            T.append(f"{tex_escape(name)} & " + " & ".join(cells) + r" \\")
        T.append(r"\midrule")
        T.append("F2 on every item & " + " & ".join(
            f"{ci_pct(v2m[m]['policies']['F2']['every item']['acc change'], signed=True)} & "
            f"{v2m[m]['policies']['F2']['every item']['generated / item']:.0f} & --" for m in models) + r" \\")
        T += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Selective re-check with F2 on MMLU-Pro test: the 10\% of items each ranking puts first are re-checked, "
              r"$a_0$ is kept elsewhere. Accuracy change in pp (95\% cluster-bootstrap CIs), incremental generated tokens per "
              r"item (signal on every item plus F2 on the selected items), and the paired difference from the verbal order. The "
              r"task probe was selected and fitted on this task's dev split; the math probe is applied unchanged. Other shares "
              r"in \texttt{reports/v2\_02\_mc\_results.md}.}", r"\label{tab:v2-recheck}", r"\end{table*}"]
        write_table("tab_v2_recheck", T)
    # ---- second review (D18): full-cost accounting and re-check policies; model-to-model differences
    fu5, fu6 = jload("fu_05_recheck_cost"), jload("fu_06_crossmodel")
    flatten("fu5", {k: v for k, v in fu5.items() if k != "probe (frozen selection)"})
    flatten("fu6", fu6)
    K["fu5.a0.gen"] = f"{fu5['a0 cost']['generated tokens / item']:.0f}"
    for c, v in fu5["cost per condition"].items():
        K[f"fu5.cost.{c}.gen"] = f"{v['generated tokens / item']:.0f}"
        K[f"fu5.cost.{c}.rel"] = f"{v['generated, relative to a0']:.2f}"
    K["fu5.agree.gen"] = f"{fu5['signal cost']['agreement with a0 (5 samples)']['generated']:,.0f}".replace(",", "{,}")
    for b, row in fu5["policies"]["F2"]["budgets"].items():
        for name, v in row.items():
            K[f"fu5.F2.{b}.{name}.gen"] = f"{v['incremental generated / item']:.0f}"
    names = ["verbal s", "probe", "answer length", "min token log-prob", "agreement with a0", "dev-fitted correctness, cheap",
             "dev-fitted correctness, full", "dev-fitted net gain, cheap", "dev-fitted net gain, full", "random (expected)",
             "gold wrong-first (uses gold)"]
    T = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Condition (on top of $a_0$) & Accuracy change (pp) & \WtoR{} (\%) & \RtoW{} (\%) & Generated tokens / item & $\times a_0$ \\",
         r"\midrule"]
    for c, v in fu5["cost per condition"].items():
        T.append(f"{tex_escape(c)} & {ci_pct(v['acc change'], signed=True)} & {100 * v['W->R']:.1f} & {100 * v['R->W']:.1f} & "
                 f"{v['generated tokens / item']:.0f} & {v['generated, relative to a0']:.2f} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Full generation cost of each condition (4B, pooled D-nat test, $n = {num(fu5['n test'])}$): tokens "
          rf"generated per item on top of $a_0$ ({fu5['a0 cost']['generated tokens / item']:.0f} tokens), counting both turns "
          r"of F2 and F2-scratch. F3 and F5 omit the verifier's own cost. Wall-clock time per condition was not recorded. "
          r"From \texttt{analysis/out/fu\_05\_recheck\_cost.json}.}", r"\label{tab:fu-cost}", r"\end{table*}"]
    write_table("tab_fu_cost", T)
    pol = fu5["policies"]["F2"]["budgets"]
    bs = ["10", "20", "30"]
    T = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{l" + "rr" * len(bs) + "r}",
         r"\toprule", r"Ranking (re-check first) & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{b}\% of items}}" for b in bs)
         + r" & $-$ verbal at 10\% \\",
         " & " + " & ".join(r"pp & tokens" for _ in bs) + r" & pp \\", r"\midrule"]
    for name in names:
        cells = []
        for b in bs:
            v = pol[b][name]
            cells += [ci_pct(v["acc change"], signed=True), f"{v['incremental generated / item']:.0f}"]
        mv = pol["10"][name].get("minus verbal s")
        T.append(f"{tex_escape(name)} & " + " & ".join(cells) + f" & {ci_pct(mv, signed=True) if mv else '--'} \\\\")
    every = fu5["policies"]["F2"]["every item"]
    T += [r"\midrule", rf"F2 on every item & \multicolumn{{{2 * len(bs)}}}{{l}}{{{ci_pct(every['acc change'], signed=True)} pp, "
          rf"{every['generated / item']:.0f} tokens / item}} & -- \\", r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Selective re-check with F2 (4B, pooled D-nat test, $n = {num(fu5['n test'])}$): re-check the given share of "
          r"items in the order of each ranking, keep $a_0$ elsewhere. Accuracy change in pp (95\% cluster-bootstrap CIs) and "
          r"incremental generated tokens per item, counting the signal on every item and F2 on the selected items. Dev-fitted "
          r"rows are logistic models fitted on the dev split (net gain: P(fix) $-$ P(damage) under F2); dev probe features "
          r"are cross-fitted. ``Agreement with $a_0$'' is the share of 5 sampled answers equivalent to $a_0$. The gold row "
          r"re-checks wrong answers first in a fixed random order; it is not an upper bound. Exploratory, on the seen test "
          r"split. From \texttt{analysis/out/fu\_05\_recheck\_cost.json}.}", r"\label{tab:fu-recheck}", r"\end{table*}"]
    write_table("tab_fu_recheck", T)
    qs = ["F2 - F1", "F3(1.0) - F3(0.5)", "F2 - a0", "role (user - assistant), D-inj F0", "label (other - self), D-inj F0",
          "H2a: AUROC probe - verbal"]
    head = ["F2$-$F1", r"F3(1.0)$-$F3(0.5)", r"F2$-a_0$", "role", "label", "H2a"]
    T = [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{l" + "r" * len(qs) + "}",
         r"\toprule", "& " + " & ".join(head) + r" \\", r"\midrule"]
    for m, v in fu6["models"].items():
        T.append(f"{m} & " + " & ".join(ci_pct(v[q], signed=True) for q in qs) + r" \\")
    T.append(r"\midrule")
    for p, v in fu6["pairs"].items():
        T.append(f"{tex_escape(p).replace(' - ', ' $-$ ')} & " + " & ".join(ci_pct(v[q], signed=True) for q in qs) + r" \\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Model-to-model differences on common items (pooled D-nat test $n = {num(fu6['n D-nat common'])}$; D-inj "
          rf"$n = {num(fu6['n D-inj common'])}$). Upper rows: per model; lower rows: differences of the item-level quantities "
          r"between two models, with 95\% cluster-bootstrap CIs that resample the same problems for both. pp; H2a is the "
          r"AUROC difference $\times 100$. F3(1.0)$-$F3(0.5) is the endpoint difference, not the frozen slope. Role and "
          r"label are the D-inj F0 contrasts; for Tulu-3-8B the role contrast includes a template boundary token. "
          r"Exploratory. From \texttt{analysis/out/fu\_06\_crossmodel.json}.}", r"\label{tab:fu-crossmodel}", r"\end{table*}"]
    write_table("tab_fu_crossmodel", T)


def third_review():
    """Third review (DEVIATIONS D19): keys fu9.* (selective re-check vs ALL, analysis/out/fu_09_recheck_vs_all.json),
    fu10.* (MMLU-Pro scoring audit, fu_10_mc_parse_audit.json), ms.* (derived shares for the manuscript), and the
    manuscript tables tab_ms_repair, tab_ms_policy, tab_ms_parse. Arithmetic on stored estimates only."""
    fu5, v2, fu9, fu10 = (jload(n) for n in ("fu_05_recheck_cost", "v2_02_mc_results", "fu_09_recheck_vs_all", "fu_10_mc_parse_audit"))
    alias = {"math, 4B (D-nat test, seen)": "math", "MMLU-Pro, 4B (addendum test)": "mc4B", "MMLU-Pro, T8B (addendum test)": "mcT8B"}
    for long, short in alias.items():
        flatten(f"fu9.{short}", fu9[long])
        for act, r in fu9[long].items():
            if not isinstance(r, dict):
                continue
            for b, row in r["budgets"].items():
                for name, v in row.items():
                    K[f"fu9.{short}.{act}.{b}.{name}.selwrong"] = f"{100 * v['share of selected initially wrong']:.0f}"
    flatten("fu10", fu10["models"])
    add = (ROOT / "docs" / "planning" / "validation_v2" / "addendum.md").read_text(encoding="utf-8")
    K["design.addendum.date"] = re.search(r"Status: frozen (\d{4}-\d{2}-\d{2})", add).group(1)
    rv = (ROOT / "paper" / "refs_verified.md").read_text(encoding="utf-8")
    K["design.search.update"] = re.search(r"### Updated search \((\d{4}-\d{2}-\d{2})\)", rv).group(1)
    for m, o in v2["models"].items():
        f2 = o["conditions"]["F2"]
        K[f"ms.{m}.F2.breakeven"] = f"{100 * f2['R->W'] / (f2['W->R'] + f2['R->W']):.0f}"
    f2m = fu5["cost per condition"]["F2"]
    K["ms.math.F2.breakeven"] = f"{100 * f2m['R->W'] / (f2m['W->R'] + f2m['R->W']):.0f}"
    # within the selected pool: repair rate among its wrong items, damage rate among its right items
    for short, n, pol in (("math", fu5["n test"], fu5["policies"]["F2"]), ("4B", v2["models"]["4B"]["n"], v2["models"]["4B"]["policies"]["F2"]),
                          ("T8B", v2["models"]["T8B"]["n"], v2["models"]["T8B"]["policies"]["F2"])):
        for b, row in pol["budgets"].items():
            if b == "100":
                continue
            n_sel = int(round(int(b) / 100 * n))
            f9 = fu9[[k for k, v in alias.items() if v == {"math": "math", "4B": "mc4B", "T8B": "mcT8B"}[short]][0]]["F2"]["budgets"][b]
            for name, v in row.items():
                if "fixes" not in v or name not in f9:
                    continue
                nw = int(round(f9[name]["share of selected initially wrong"] * n_sel))
                K[f"ms.{short}.F2.{b}.{name}.fixrate"] = f"{100 * v['fixes'] / nw:.1f}" if nw else "--"
                K[f"ms.{short}.F2.{b}.{name}.damrate"] = f"{100 * v['damages'] / (n_sel - nw):.1f}" if n_sel > nw else "--"
    for m, o in fu10["models"].items():
        tr = o["final status among transitions"]["F2"]
        K[f"fu10.{m}.F2.rw.letter"] = str(sum(v for k, v in tr.items() if k.startswith(("R->W ok", "R->W fallback"))))
        for c in "ambc":
            K[f"fu10.{m}.F2.rw.{c}"] = str(sum(v for k, v in tr.items() if k.startswith("R->W") and k.endswith(f"({c})")))
        K[f"fu10.{m}.F2.rw.noletter"] = str(sum(int(K[f"fu10.{m}.F2.rw.{c}"]) for c in "ambc"))
        K[f"fu10.{m}.F2.rw.all"] = str(int(K[f"fu10.{m}.F2.rw.letter"]) + int(K[f"fu10.{m}.F2.rw.noletter"]))
        K[f"fu10.{m}.F2.wr.all"] = str(sum(v for k, v in tr.items() if k.startswith("W->R")))
        st = o["status"]
        K[f"fu10.{m}.F2.unparsed"] = str(st["F2"].get("unparsed", 0))
        K[f"fu10.{m}.a0.fallback"] = str(st["a0"].get("fallback", 0))
        K[f"fu10.{m}.a0.fallback.length"] = str(st["a0"].get("fallback, finish length", 0))
    # settings: (label, a0 accuracy, n, {cond: (net CI, W->R, R->W, tokens)})
    S = [("Math (Qwen3-4B; seen test split)", 1 - fu5["n wrong test"] / fu5["n test"], fu5["n test"],
          {c: (v["acc change"], v["W->R"], v["R->W"], v["generated tokens / item"]) for c, v in fu5["cost per condition"].items()})]
    for m, lab in (("4B", "MMLU-Pro (Qwen3-4B; addendum)"), ("T8B", "MMLU-Pro (Tulu-3-8B; addendum)")):
        o = v2["models"][m]
        S.append((lab, o["a0 accuracy"], o["n"], {c: (v["net change"], v["W->R"], v["R->W"], v["generated tokens / item"])
                                                  for c, v in o["conditions"].items()}))
    conds = ["F0", "F1", "F2", "F2-scratch", "F3(1.0)", "F3(0.5)"]
    T = [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Setting & Condition & Net change (pp) & \WtoR{} (\%) & \RtoW{} (\%) & Repairs (\%) & Damages (\%) & Tokens \\",
         r"\midrule"]
    for j, (lab, acc, n, d) in enumerate(S):
        short = ("math", "mc4B", "mcT8B")[j]
        K[f"ms.{short}.a0"] = f"{100 * acc:.1f}"
        for i, c in enumerate(conds):
            net, wr, rw, tok = d[c]
            rep, dam = 100 * wr * (1 - acc), 100 * rw * acc
            K[f"ms.{short}.{c}.repairs"], K[f"ms.{short}.{c}.damages"] = f"{rep:.1f}", f"{dam:.1f}"
            K[f"ms.{short}.{c}.wr"], K[f"ms.{short}.{c}.rw"] = f"{100 * wr:.1f}", f"{100 * rw:.1f}"
            T.append((rf"\multirow{{{len(conds)}}}{{*}}{{\parbox{{2.5cm}}{{{lab}, $n = {num(n)}$, $a_0$ {100 * acc:.1f}\%}}}}" if i == 0 else "")
                     + f" & {tex_escape(c)} & {ci_pct(net, signed=True)} & {100 * wr:.1f} & {100 * rw:.1f} & {rep:.1f} & {dam:.1f} & {tok:.0f} \\\\")
        if j < len(S) - 1:
            T.append(r"\midrule")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Repair and damage under each feedback condition, on one fixed $a_0$ per item. Net change against $a_0$ "
          r"(95\% cluster-bootstrap CIs); \WtoR{} is a share of the initially wrong items and \RtoW{} of the initially right "
          r"ones, so they are not subtracted directly; repairs and damages as shares of all items do add up to the net "
          r"change. Tokens: generated per item on top of $a_0$, both turns of F2 and F2-scratch; F3 omits the verifier's own "
          r"cost (it is simulated from gold). Math is the pooled D-nat test split, already seen when this analysis was "
          r"written (exploratory); MMLU-Pro is the six non-computational categories under the frozen addendum. From "
          r"\texttt{analysis/out/fu\_05\_recheck\_cost.json} and \texttt{v2\_02\_mc\_results.json}.}",
          r"\label{tab:ms-repair}", r"\end{table*}"]
    write_table("tab_ms_repair", T)

    # policies at the addendum's shares (10 %, 20 %), F2 as the action
    P = [("math", "Math, Qwen3-4B", fu5["policies"]["F2"], "verbal s", "probe"),
         ("mc4B", "MMLU-Pro, Qwen3-4B", v2["models"]["4B"]["policies"]["F2"], "verbal W0", "task probe"),
         ("mcT8B", "MMLU-Pro, Tulu-3-8B", v2["models"]["T8B"]["policies"]["F2"], "verbal W0", "task probe")]
    rows = lambda vb, pr: [("random (expected)", "random"), (vb, "verbal score"), (pr, "probe"), ("answer length", "answer length"),  # noqa: E731
                            ("agreement with a0", r"agreement with $a_0$")]
    T = [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{Re-check 10\% of items} & \multicolumn{3}{c}{Re-check 20\% of items} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Setting & Ranking & vs KEEP (pp) & vs ALL (pp) & Tokens & vs KEEP (pp) & vs ALL (pp) & Tokens \\", r"\midrule"]
    for j, (short, lab, pol, vb, pr) in enumerate(P):
        f9 = fu9[[k for k, v in alias.items() if v == short][0]]["F2"]
        T.append(rf"\multicolumn{{2}}{{l}}{{\textit{{{lab}}}: ALL (F2 on every item)}} & \multicolumn{{3}}{{l}}{{{ci_pct(pol['every item']['acc change'], signed=True)} pp, "
                 rf"{pol['every item']['generated / item']:.0f} tokens}} & & & \\")
        for key, name in rows(vb, pr):
            cells = []
            for b in ("10", "20"):
                v, w = pol["budgets"][b][key], f9["budgets"][b][key]
                cells += [ci_pct(v["acc change"], signed=True), ci_pct(w["minus ALL"], signed=True), f"{v['incremental generated / item']:.0f}"]
            T.append(f" & {name} & " + " & ".join(cells) + r" \\")
        # gold reference (re-checks wrong a0 first; uses gold; not a policy and not an upper bound)
        J11 = jload("fu_11_ms_robustness")
        cells = []
        for b in ("10", "20"):
            if short == "math":
                v, w = pol["budgets"][b]["gold wrong-first (uses gold)"], f9["budgets"][b]["gold wrong-first (uses gold)"]
                cells += [ci_pct(v["acc change"], signed=True), ci_pct(w["minus ALL"], signed=True), f"{v['incremental generated / item']:.0f}"]
            elif J11:
                g = J11[{"mc4B": "mmlupro 4B", "mcT8B": "mmlupro T8B"}[short]]["gold wrong-first"][b]
                cells += [ci_pct(g["vs KEEP"], signed=True), ci_pct(g["vs ALL"], signed=True), "--"]
            else:
                cells += ["--", "--", "--"]
        T.append(r" & \textit{gold reference} & " + " & ".join(cells) + r" \\")
        if j < len(P) - 1:
            T.append(r"\midrule")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Selective re-check with F2, replayed offline from the stored per-item outputs: the given share of items "
          r"is re-checked in the order of a ranking and $a_0$ is kept elsewhere. ``vs KEEP'' is the accuracy change against "
          r"keeping every $a_0$; ``vs ALL'' is the paired difference from re-checking every item (95\% cluster-bootstrap "
          r"CIs). Tokens: incremental generated tokens per item, counting the signal on every item and F2 on the selected "
          r"items. Probe: the frozen math probe on math, the task probe fitted on the MMLU-Pro dev split. The gold reference "
          r"re-checks initially wrong items first (it uses gold labels; it is not a policy and not an upper bound, since wrong "
          r"answers differ in how often F2 repairs them); its MMLU-Pro rows are post hoc (\texttt{fu\_11}). The 10\% and 20\% "
          r"shares are the addendum's primary shares; every other ranking and share is in \texttt{reports/fu\_05}, "
          r"\texttt{v2\_02} and \texttt{fu\_09}. The vs-ALL column is post hoc (D19). From \texttt{analysis/out/fu\_05\_recheck\_cost.json}, "
          r"\texttt{v2\_02\_mc\_results.json} and \texttt{fu\_09\_recheck\_vs\_all.json}.}", r"\label{tab:ms-policy}", r"\end{table*}"]
    write_table("tab_ms_policy", T)

    # scoring audit
    T = [r"\begin{table}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrr}", r"\toprule",
         r"Tulu-3-8B, MMLU-Pro test & Frozen & Rule v1 & Rule v2 \\", r"\midrule"]
    m = fu10["models"]["T8B"]
    for c in ("F2", "F2-scratch"):
        T.append(f"P1 {c}: net change (pp) & " + " & ".join(ci_pct(x["P1"][c]["net change"], signed=True)
                                                       for x in (m["frozen"], m["rule v1"]["primary"], m["rule v2"]["primary"])) + r" \\")
    T.append("P2 AUROC task probe $-$ verbal & " + " & ".join(ci_pct(x["P2 AUROC task probe - verbal W0"], signed=True)
                                                           for x in (m["frozen"], m["rule v1"]["primary"], m["rule v2"]["primary"])) + r" \\")
    for b in ("10", "20"):
        T.append(f"P2 probe $-$ verbal order, {b}\\% & " + " & ".join(ci_pct(x["P2 order task probe - verbal W0"][b], signed=True)
                                                                     for x in (m["frozen"], m["rule v1"]["primary"], m["rule v2"]["primary"])) + r" \\")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{MMLU-Pro scoring audit for Tulu-3-8B (post hoc, D19). The frozen scorer is primary. Rule v1 also reads a "
          r"box that starts with an option letter followed by punctuation (``H, with reservations\ldots''); rule v2 does so "
          r"only when the box names no other option letter. From \texttt{analysis/out/fu\_10\_mc\_parse\_audit.json}; "
          r"per-condition status counts in \texttt{reports/fu\_10\_mc\_parse\_audit.md}.}", r"\label{tab:ms-parse}", r"\end{table}"]
    write_table("tab_ms_parse", T)


def robustness():
    """Skeptical review of the manuscript (D19): keys fu11.* from analysis/out/fu_11_ms_robustness.json, derived cost keys
    ms.cost.*, and tables tab_ms_answered (a0-answered sensitivity), tab_ms_pools (within-pool rates with CIs)."""
    J, fu5, v2 = jload("fu_11_ms_robustness"), jload("fu_05_recheck_cost"), jload("v2_02_mc_results")
    if J is None:
        return
    short = {"math": "math", "mmlupro 4B": "mc4B", "mmlupro T8B": "mcT8B"}
    for k, s in short.items():
        flatten(f"fu11.{s}", J[k])
        r = J[k]["repairs"]["F2"]
        K[f"fu11.{s}.F2.rep.unans.pct"] = f"{100 * r['from unanswered a0'] / r['all']:.0f}"
        for b in ("10", "20"):
            for name, v in J[k]["pools"][b].items():
                K[f"fu11.{s}.pools.{b}.{name}.unans.pct"] = f"{100 * v['share of pool with unanswered a0']:.0f}"
    # total tokens per item (generated + prompt processed, all turns, no cache reuse): F2 on every item vs five samples
    c5, sc = fu5["cost per condition"]["F2"], fu5["signal cost"]["agreement with a0 (5 samples)"]
    tot = {"math": (c5["generated tokens / item"] + c5["processed prompt tokens / item"], sc["generated"] + sc["processed (no cache)"],
                    sc["generated"])}
    for k, s in (("mmlupro 4B", "mc4B"), ("mmlupro T8B", "mcT8B")):
        c = J[k]["cost"]
        tot[s] = (c["F2: generated / item"] + c["prompt tokens / item"]["F2"],
                  c["samples: generated / item"] + c["samples: prompt tokens / item (5 x a0 prompt)"], c["samples: generated / item"])
    for s, (f2, smp, gen) in tot.items():
        K[f"ms.cost.{s}.F2.total"] = f"{f2:,.0f}".replace(",", "{,}")
        K[f"ms.cost.{s}.samples.total"] = f"{smp:,.0f}".replace(",", "{,}")
        K[f"ms.cost.{s}.samples.gen"] = f"{gen:,.0f}".replace(",", "{,}")
        K[f"ms.cost.{s}.ratio.total"] = f"{smp / f2:.2f}"
    # a0-answered sensitivity table
    m, a4, aT = J["math"], J["mmlupro 4B"], J["mmlupro T8B"]
    fm = fu5["cost per condition"]
    T = [r"\begin{table}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrr}", r"\toprule",
         r"Quantity & All items & $a_0$ answered \\", r"\midrule",
         rf"\multicolumn{{3}}{{l}}{{\textit{{Math, Qwen3-4B}} ($n = {num(m['n'])}$; unanswered $a_0$: {m['unanswered a0']})}} \\"]
    for c in ("F0", "F2", "F2-scratch", "F3(1.0)"):
        T.append(f"{tex_escape(c)} net change & {ci_pct(fm[c]['acc change'], signed=True)} & {ci_pct(m['a0 answered']['net change'][c], signed=True)} \\\\")
    for b in ("10", "20"):
        T.append(f"Probe $-$ verbal order, F2 on {b}\\% & {ci_pct(fu5['policies']['F2']['budgets'][b]['probe']['minus verbal s'], signed=True)} & "
                 f"{ci_pct(m['a0 answered']['F2 re-check'][b]['probe - verbal'], signed=True)} \\\\")
    for k, o, lab in (("4B", a4, "MMLU-Pro, Qwen3-4B"), ("T8B", aT, "MMLU-Pro, Tulu-3-8B")):
        f = v2["models"][k]
        T += [r"\midrule", rf"\multicolumn{{3}}{{l}}{{\textit{{{lab}}} ($n = {num(o['n'])}$; unanswered $a_0$: {o['unanswered a0']})}} \\"]
        for c in ("F0", "F1", "F2", "F2-scratch"):
            T.append(f"P1 {tex_escape(c)} net change & {ci_pct(f['conditions'][c]['net change'], signed=True)} & "
                     f"{ci_pct(o['a0 answered']['P1'][c], signed=True)} \\\\")
        T.append(f"P2 AUROC task probe $-$ verbal & {ci_pct(f['task probe - verbal W0'], signed=True)} & "
                 f"{ci_pct(o['a0 answered']['P2 AUROC task probe - verbal W0'], signed=True)} \\\\")
        for b in ("10", "20"):
            T.append(f"P2 probe $-$ verbal order, {b}\\% & {ci_pct(f['policies']['F2']['budgets'][b]['task probe']['minus verbal W0'], signed=True)} & "
                     f"{ci_pct(o['a0 answered']['P2 order task probe - verbal W0'][b], signed=True)} \\\\")
        T.append(f"P3 agreement $-$ verbal order, 10\\% & {ci_pct(f['policies']['F2']['budgets']['10']['agreement with a0']['minus verbal W0'], signed=True)} & "
                 f"{ci_pct(o['a0 answered']['P3 order agreement - verbal W0 (10 %)'], signed=True)} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Sensitivity to initial answers without an answer (post hoc, D19). A truncated or unparsed $a_0$ is scored "
          r"wrong, so a revision that ends with an answer counts as a repair. ``$a_0$ answered'' drops those items (math: "
          r"status not ok; MMLU-Pro: truncated or unparsed; dev alike for the dev-fitted rules) and recomputes; rankings are "
          r"re-applied within the subset. pp, AUROC $\times 100$, 95\% cluster-bootstrap CIs. From "
          r"\texttt{analysis/out/fu\_11\_ms\_robustness.json}.}", r"\label{tab:ms-answered}", r"\end{table}"]
    write_table("tab_ms_answered", T)
    # within-pool rates
    names = {"math": [("verbal s", "verbal"), ("probe", "probe"), ("answer length", "length"), ("agreement with a0", "agreement"),
                      ("gold wrong-first (uses gold)", "gold reference")],
             "mc": [("verbal W0", "verbal"), ("task probe", "task probe"), ("math probe", "math probe"), ("answer length", "length"),
                    ("agreement with a0", "agreement"), ("gold wrong-first (uses gold)", "gold reference")]}
    T = [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{F2 on 10\% of items} & \multicolumn{2}{c}{F2 on 20\% of items} & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Setting & Ranking & Repair rate (\%) & Damage rate (\%) & Repair rate (\%) & Damage rate (\%) & Unanswered $a_0$ in pool (10\%) \\",
         r"\midrule"]
    rate = lambda v: "--" if v is None else f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"  # noqa: E731
    for j, (k, lab, nm) in enumerate((("math", "Math, Qwen3-4B", "math"), ("mmlupro 4B", "MMLU-Pro, Qwen3-4B", "mc"),
                                      ("mmlupro T8B", "MMLU-Pro, Tulu-3-8B", "mc"))):
        po = J[k]["pools"]
        T.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{lab}}}: all items, repair {rate(po['overall']['repair rate'])}, damage "
                 rf"{rate(po['overall']['damage rate'])}}} \\")
        for key, name in names[nm]:
            a, b = po["10"][key], po["20"][key]
            T.append(f" & {name} & {rate(a['repair rate in pool'])} & {rate(a['damage rate in pool'])} & {rate(b['repair rate in pool'])} & "
                     f"{rate(b['damage rate in pool'])} & {100 * a['share of pool with unanswered a0']:.0f}\\% \\\\")
        if j < 2:
            T.append(r"\midrule")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Repair and damage within the pools that each ranking selects for F2 (post hoc, D19): the repair rate "
          r"among the pool's initially wrong items and the damage rate among its initially right items, with 95\% "
          r"cluster-bootstrap CIs; the gold reference's pool holds no right item at these shares on math and MMLU-Pro. Last "
          r"column: share of the 10\% pool whose $a_0$ gave no answer. Descriptive; no test. From "
          r"\texttt{analysis/out/fu\_11\_ms\_robustness.json}.}", r"\label{tab:ms-pools}", r"\end{table*}"]
    write_table("tab_ms_pools", T)


def audit_keys():
    """Blind audit of math outputs (analysis/fu_blind_audit.py tally): keys audit.<stratum>.{judged,agree,undecided},
    audit.total.{judged,agree}, audit.n.disagree (reviewer disagreements with the primary scorer)."""
    p = ROOT / "docs" / "planning" / "next_experiments" / "blind_audit" / "audit_result.json"
    if not p.exists():
        return
    a = json.loads(p.read_text(encoding="utf-8"))
    for s, v in a["per_stratum"].items():
        for f in ("judged", "agree", "undecided"):
            K[f"audit.{s}.{f}"] = str(v[f])
    K["audit.total.judged"] = str(sum(v["judged"] for v in a["per_stratum"].values()))
    K["audit.total.agree"] = str(sum(v["agree"] for v in a["per_stratum"].values()))
    K["audit.n.disagree"] = str(sum(1 for r in a["rows"] if r["human"] is not None and r["human"] != r["primary"]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    GEN.mkdir(parents=True, exist_ok=True)
    conf = jload("06_confirmatory_4B")
    for scope in ("main", "v2", "a0_ok"):
        flatten(f"c.{scope}", conf[scope])
    for arm, v in conf["main"]["exp3_arms"].items():
        K[f"c.exp3.{arm}.wr"] = pct(v["W->R D-inj"])
        K[f"c.exp3.{arm}.rw"] = pct(v["R->W D-nat"])
    for m in ("4B", "1.7B", "8B", "T8B"):
        x = jload(f"06_exploratory_{m}")
        if x:
            flatten(f"x.{m}", x)
    for m in ("1.7B", "8B", "T8B"):
        t = jload(f"p4_{m}_tier1")
        if t:
            flatten(f"t.{m}", t["estimates"])
            flatten(f"t.{m}.secondary", t["secondary"])
    e5 = jload("p4_4B_exp5")
    flatten("e5.H4a", e5["H4a"]["est_lo_hi"])
    flatten("e5.H4a_noc", e5["H4a without c > 100 items"]["est_lo_hi"])
    for k, v in e5["secondary"].items():
        flatten(f"e5.sec.{k}", v["est_lo_hi"])
    flatten("e5.cells", e5["cells"])
    flatten("e5.metrics", e5["metrics"])
    flatten("e5.H4d", e5["H4d"])
    for k, v in e5["H4b"].items():
        sp = v["spillover"]
        K[f"e5.H4b.{k}"] = f"{sp[0]:+.4f} [{sp[1]:+.4f}, {sp[2]:+.4f}]"
    K["e5.n"] = num(e5["n_items"])
    flatten("ch", jload("p4_4B_channel")["all"])
    e4 = jload("p4_4B_exp4")
    t4 = e4["test"]
    K["e4.n"] = num(t4["n"])
    for metric in ("lp_gold", "marker"):
        tot = t4["total"][metric]
        K[f"e4.total.{metric}"] = f"{tot[0]:+.2f} [{tot[1]:+.2f}, {tot[2]:+.2f}]"
        comps = t4["confirm"][metric]
        K[f"e4.rep.{metric}"] = f"{sum(c['replicated'] for c in comps)} of {len(comps)}"
        K[f"e4.layers.{metric}"] = ", ".join(str(c["layer"]) for c in sorted(comps, key=lambda c: c["layer"]))
        K[f"e4.group.{metric}"] = tex_escape(sorted({c["group"] for c in comps})[0] if len({c["group"] for c in comps}) == 1 else "mixed")
        shares = [c["share_of_total"] for c in comps]
        K[f"e4.share.{metric}"] = f"{min(shares):.2f}--{max(shares):.2f}"
    for m in ("4B", "8B"):
        th = jload(f"p4_{m}_thinking")
        if th:
            flatten(f"th.{m}.contrasts", th["contrasts"])
            flatten(f"th.{m}.cells", th["cells"])
            K[f"th.{m}.n"] = num(th["n_items"])
    st = jload("p4_4B_streams")
    ks = sorted(st["eval"]["self"], key=int)
    K["st.kmax"] = ks[-1]
    es = json.loads((ROOT / "results/main/4B/main-streams-test-4B-f15b933/eval_summary.json").read_text(encoding="utf-8"))
    K["st.neval"] = num(es["0"]["base"]["n"])
    K["st.base"] = f"{100 * st['eval']['self']['0']['acc_greedy'][0]:.0f}"
    for c in ("self", "ext", "random", "context"):
        v = st["eval"][c][ks[-1]]["acc_greedy"]
        K[f"st.{c}.final"] = ", ".join(f"{100 * x:.0f}" for x in v)
        K[f"st.{c}.k25"] = ", ".join(f"{100 * x:.0f}" for x in st["eval"][c]["25"]["acc_greedy"])
    for t, p in st["post_hoc"].items():
        K[f"st.onset.{t}"] = str(p["onset_step"]) if p["onset_step"] is not None else "none"
        ub = p["updates_before"]
        K[f"st.ulright.{t}"] = str(ub.get("UL, a0 right", 0))
    on = [p["onset_step"] for t, p in st["post_hoc"].items() if t.startswith("self")]
    K["st.onset.self.range"] = f"{min(on)}--{max(on)}"
    ul = [p["updates_before"].get("UL, a0 right", 0) for t, p in st["post_hoc"].items() if t.startswith("self")]
    K["st.ulright.self.range"] = f"{min(ul)}--{max(ul)}"
    # design constants and dates not in NUMBERS_NEEDED.md
    m = re.search(r"^LAYER_STEP\s*=\s*(\d+)", (ROOT / "src/steer/pilot.py").read_text(encoding="utf-8"), re.M)
    K["design.steer.layerstep"] = m.group(1)
    ledger = [json.loads(l) for l in open(ROOT / "cloud" / "ledger.jsonl", encoding="utf-8") if l.strip()]
    K["design.cloud.lastpause"] = max(r["pause_time"] for r in ledger if r.get("pause_time"))[:10]
    # test suite at the freeze
    m = re.search(r"(\d+) passed", (ROOT / "reports" / "03_pytest_freeze.txt").read_text(encoding="utf-8", errors="replace"))
    K["tests.freeze.passed"] = m.group(1) if m else "?"

    # ---- per-condition means of 1A, every model
    means = {m: cond_means(m) for m in SPECS}
    for m, d in means.items():
        for c, v in d.items():
            for f in ("acc", "wr", "rw", "trunc"):
                K[f"m.{m}.1A.{c}.{f}"] = pct(v[f])
            K[f"m.{m}.1A.{c}.tok"] = "--" if v["tok"] is None else f"{v['tok']:.0f}"
            K[f"m.{m}.1A.{c}.n"] = num(v["n"])

    # ---- tables
    fam = ["H1a", "H1b", "H2a", "H3a-role", "H3a-label", "H3b"]
    M = conf["main"]
    T = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{lrrrrlr}", r"\toprule",
         r"Hypothesis & $n$ & Estimate [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & Outcome & v2 labels \\", r"\midrule"]
    for h in fam:
        r, v = M[h], conf["v2"][h]
        T.append(f"{h} & {num(r['n'])} & {100 * r['est']:+.2f} [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] & {pval(r['p'])} & "
                 f"{pval(r['p_holm'])} & {tex_escape(r['category'])} & {100 * v['est']:+.2f} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Confirmatory family (Qwen3-4B, non-thinking, test split; Holm over six members). Estimates in pp "
          r"(H2a: AUROC difference $\times 100$); 95\% cluster-bootstrap CIs, not adjusted for multiplicity. Every block "
          r"ran to its planned $n$. The last column is the estimate under the second scorer's labels. Generated by "
          r"\texttt{analysis/p7\_results\_tex.py} from \texttt{analysis/out/06\_confirmatory\_4B.json}.}",
          r"\label{tab:family}", r"\end{table*}"]
    (GEN / "tab_family.tex").write_text("\n".join(T) + "\n", encoding="utf-8", newline="\n")

    # secondary D-nat estimands (prereg §0, §2: shown next to every D-nat headline)
    sec = conf["main"]["secondary"]
    scopes = ["original pool", "reweighted to the original mix", "gsm8k", "math500", "math_ext", "math_ext − original pool"]
    hyps = ["H1a", "H1b", "H1d", "H2a"]
    cell = lambda v: "--" if v is None else f"{100 * v[0]:+.2f} [{100 * v[1]:+.2f}, {100 * v[2]:+.2f}]"
    T = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Scope & H1a & H1b & H1d & H2a \\", r"\midrule",
         "Pooled (confirmatory) & " + " & ".join(cell([M[h]["est"], M[h]["lo"], M[h]["hi"]]) for h in hyps) + r" \\"]
    for s in scopes:
        T.append(f"{tex_escape(s)} & " + " & ".join(cell(sec.get(h, {}).get(s)) for h in hyps) + r" \\")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Secondary D-nat estimands (4B; pp, H2a: AUROC difference $\times 100$; 95\% cluster-bootstrap CIs). "
          r"The confirmatory population pools GSM8K test, MATH-500 and the rest of the MATH test set (math\_ext) as "
          r"sampled; ``original pool'' is GSM8K + MATH-500, ``reweighted'' reweights the pooled items to the original "
          r"GSM8K/MATH mix. H1d is estimation-only. Descriptive; no tests.}",
          r"\label{tab:secondary}", r"\end{table*}"]
    (GEN / "tab_secondary.tex").write_text("\n".join(T) + "\n", encoding="utf-8", newline="\n")

    d4 = means["4B"]
    T = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Condition & Acc & \WtoR{} & \RtoW{} & Trunc. & Tokens \\", r"\midrule"]
    for c in COND_1A + ["F4 (D-inj)"]:
        if c in d4:
            v = d4[c]
            tok = "--" if v["tok"] is None else f"{v['tok']:.0f}"
            T.append(f"{tex_escape(c)} & {pct(v['acc'])} & {pct(v['wr'])} & {pct(v['rw'])} & {pct(v['trunc'])} & {tok} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Exp~1A on D-nat (4B, pooled test population, $n = {num(d4['B1']['n'])}$): accuracy, \WtoR{{}} and \RtoW{{}} "
          r"relative to $a_0$, the share of revisions truncated at the token cap (all in \%), and mean generated tokens. "
          r"B0 is $a_0$. F4 is the D-inj baseline (accuracy = \WtoR{}). Point estimates; CIs of the contrasts are in the text.}",
          r"\label{tab:exp1a}", r"\end{table}"]
    (GEN / "tab_exp1a.tex").write_text("\n".join(T) + "\n", encoding="utf-8", newline="\n")

    rows = [("H1a", "H1a"), ("H1b", "H1b"), ("H1d", "H1d"), ("H2a", "H2a"), ("H2b(i) F2 - F0", "H2b(i)"),
            ("H2b(ii) F3(1.0) - F0", "H2b(ii)"), ("H3a-role", "H3a-role"), ("H3a-label", "H3a-label")]
    reps = {m: jload(f"p4_{m}_tier1") for m in ("1.7B", "8B", "T8B")}
    T = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Quantity & Qwen3-1.7B & Qwen3-4B & Qwen3-8B & Tulu-3-8B \\", r"\midrule"]
    for key, lab in rows:
        cell = lambda r: f"{100 * r['est']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]"
        T.append(f"{lab} & {cell(reps['1.7B']['estimates'][key])} & {cell(M[key])} & {cell(reps['8B']['estimates'][key])} & "
                 f"{cell(reps['T8B']['estimates'][key])} \\\\")
    au = lambda e: f"{e['H2a']['auroc_probe']:.3f} / {e['H2a']['auroc_verbal']:.3f}"  # noqa: E731
    T.append(f"AUROC probe / verbal & {au(reps['1.7B']['estimates'])} & {au(M)} & {au(reps['8B']['estimates'])} & "
             f"{au(reps['T8B']['estimates'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Family quantities across models, same estimands and bootstrap as Table~\ref{tab:family} (pp; H2a "
          r"$\times 100$). Qwen3-4B is the confirmatory run; the other columns are replications with CIs, not tests. "
          r"Tulu-3-8B, a fine-tune of Llama~3.1, is the exploratory second family (Appendix~\ref{app:followups}); its "
          r"role cells also differ in one template boundary token, so its H3a-role is not the Qwen3 minimal pair.}",
          r"\label{tab:replication}", r"\end{table*}"]
    (GEN / "tab_replication.tex").write_text("\n".join(T) + "\n", encoding="utf-8", newline="\n")

    cells = ["B1", "context (v = correct)", "context (v = incorrect)", "parameter / self", "parameter / perm0",
             "parameter / perm1", "parameter / perm2", "parameter / ext", "parameter / shuffled", "parameter / random",
             "parameter+context / self"]
    T = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrr}", r"\toprule",
         r"Cell & Acc & $-$B1 (pp) & \WtoR{} \\", r"\midrule"]
    for c in cells:
        if c in e5["cells"]:
            v = e5["cells"][c]
            T.append(f"{tex_escape(c)} & {100 * v['acc'][0]:.1f} & {100 * v['minus_B1'][0]:+.1f} [{100 * v['minus_B1'][1]:+.1f}, "
                     f"{100 * v['minus_B1'][2]:+.1f}] & {100 * v['W->R'][0]:.1f} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Exp~5, per-item updates (4B, $n = {num(e5['n_items'])}$ test items; accuracy and \WtoR{{}} in \%). "
          r"Each item gets its own update from $\theta_0$; perm$k$ uses the stratum-permuted readout verdicts. Context "
          r"rows put $a_0$ and the verdict $v$ in the context without an update; every item is run with both values "
          r"of $v$. ``parameter+context'' combines the self update with its own verdict in the context. From "
          r"\texttt{analysis/out/p4\_4B\_exp5.json}.}", r"\label{tab:exp5}", r"\end{table}"]
    (GEN / "tab_exp5.tex").write_text("\n".join(T) + "\n", encoding="utf-8", newline="\n")

    T = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrr}", r"\toprule",
         r"Condition & 1.7B & 4B & 8B \\", r"\midrule"]
    ex = {m: jload(f"06_exploratory_{m}") for m in ("1.7B", "4B", "8B")}
    for c in ["no-feedback", "F0", "F3(0.5)", "F3(0.7)", "F3(0.9)", "F3(1.0)"]:
        T.append(f"{c} acc & " + " & ".join(f"{100 * ex[m]['exp0']['cells'][c]['acc'][0]:.1f}" for m in ("1.7B", "4B", "8B")) + r" \\")
    for c in ["F3(0.7)", "F3(0.9)", "F3(1.0)"]:
        T.append(f"{c} gain ratio & " + " & ".join(f"{ex[m]['exp0']['cells'][c]['gain ratio'][0]:.2f}" for m in ("1.7B", "4B", "8B")) + r" \\")
    T.append("valid, min over cond. & " + " & ".join(
        f"{100 * min(v['valid'][0] for v in ex[m]['exp0']['cells'].values()):.1f}" for m in ("1.7B", "4B", "8B")) + r" \\")
    T += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{Exp~0, hidden-state calibration ({num(ex['4B']['exp0']['n'])} test items in paired worlds; accuracy "
          r"and valid rate in \%, invalid = wrong). Chance is 50\%. Gain ratio $(\mathrm{acc}-0.5)/(p-0.5)$; 1 would mean "
          r"the model follows the verifier exactly. CIs in \texttt{reports/06\_exploratory\_*.md}.}",
          r"\label{tab:exp0}", r"\end{table}"]
    (GEN / "tab_exp0.tex").write_text("\n".join(T) + "\n", encoding="utf-8", newline="\n")

    followups()
    third_review()
    robustness()
    audit_keys()

    # A key with a TeX-special character cannot sit inside \csname (a "%" ends the line); those keys are left out
    # (keys with "%" have "pct" aliases).
    bad = set("%#{}\\~^$&")
    keep = sorted(k for k in K if not bad & set(k))
    L = ["% generated by analysis/p7_results_tex.py -- do not edit", f"% {len(keep)} keys ({len(K) - len(keep)} with "
         "TeX-special characters left out)"]
    for k in keep:
        L.append(rf"\expandafter\def\csname res@{k}\endcsname{{{K[k]}}}")
    (GEN / "results.tex").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(keep)} keys ({len(K) - len(keep)} left out); tables: family, exp1a, replication, exp5, exp0, fu_*")


if __name__ == "__main__":
    main()
