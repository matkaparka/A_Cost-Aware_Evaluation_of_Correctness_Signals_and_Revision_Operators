"""Neurocomputing revision: deterministic CPU-only reanalysis of stored outputs.

No generation, fitting, decoding, sampling or new experiment. The same file is
distributed as supplementary/analysis/neurocomputing_revision.py.

Usage from the repository: python -B analysis/neurocomputing_revision.py
Usage from the package: python -B analysis/neurocomputing_revision.py --manuscript ../manuscript
The copied legacy generator calls apply_revision() after its own rendering, so
its rebuild cannot restore the superseded oracle wording or mixed-cost figure.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np

SETTINGS = ("math 1.7B", "math 4B", "math 8B", "math T8B", "MMLU-Pro 4B", "MMLU-Pro T8B")
SLUGS = dict(zip(SETTINGS, ("m17", "m4", "m8", "mT", "c4", "cT")))
OPS = ("F0", "F2", "F2-scratch", "s1", "THINK")
VERSION = "neurocomputing-cpu-reanalysis-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf8", newline="\n")


def relative(path, base):
    return os.path.relpath(path, base).replace("\\", "/")


class Audit:
    def __init__(self, package, manuscript):
        self.package, self.manuscript = Path(package).resolve(), Path(manuscript).resolve()
        self.out = self.package / "analysis/out/neurocomputing"
        self.inputs, self.outputs, self.checks = {}, {}, []

    def read(self, rel):
        p = self.package / rel
        self.inputs[rel] = sha(p)
        return p

    def json(self, rel):
        return json.loads(self.read(rel).read_text(encoding="utf8"))

    def rows(self, rel):
        p = self.read(rel)
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def backup(self, path):
        path = Path(path)
        if path.is_relative_to(self.package):
            rel = path.relative_to(self.package)
        else:
            rel = Path("manuscript") / path.relative_to(self.manuscript)
        dest = self.out / "history" / rel
        if path.exists() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        return dest

    def record(self, path):
        self.outputs[relative(path, self.package)] = sha(path)


def legacy_module(package):
    p = Path(package) / "analysis/arr/acl_v2_tex.py"
    spec = importlib.util.spec_from_file_location("neuro_legacy_tables", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_threshold_function(package):
    """Execute only the actual repaired function, without importing model tooling."""
    source = (Path(package) / "analysis/arr/a3_dev_test.py").read_text(encoding="utf8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "apply_threshold")
    class Accounting:
        @staticmethod
        def reuses_sample(op, sig):
            return op == "s1" and sig in ("agreement", "margin")
    ns = {"A": Accounting, "operators": lambda p: p["operators"],
          "acquisition": lambda p, sig: p["acquisition"] if sig in ("agreement", "margin") else np.zeros(3)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "a3_dev_test.py", "exec"), ns)
    p = {"y0": np.array([0., 1., 0.]), "sig": {"agreement": np.array([0., .4, 1.]), "verbal": np.array([0., .4, 1.])},
         "acquisition": np.array([100., 200., 300.]),
         "operators": {"s1": (np.array([1., 1., 0.]), np.array([10., 20., 30.])),
                       "F2": (np.array([1., 1., 0.]), np.array([10., 20., 30.]))}}
    _, reused, n = ns["apply_threshold"](p, ("s1", "agreement", .2), .4)
    _, fresh, _ = ns["apply_threshold"](p, ("F2", "agreement", .2), .4)
    _, free, _ = ns["apply_threshold"](p, ("s1", "verbal", .2), .4)
    assert n == 2 and np.array_equal(reused, p["acquisition"])
    assert np.array_equal(fresh, np.array([110., 220., 300.]))
    assert np.array_equal(free, np.array([10., 20., 0.]))


def correct_thresholds(audit, G, generator):
    """Recompute affected cost only. Five samples are acquired for every item;
    s1 is one of them, so the corrected cost equals the recorded V5 acquisition.
    Accuracy, selections, bootstrap statistics, and the declared family do not change.
    """
    changes = []
    for stem in ("a3_dev_test", "a3_b5"):
        path = audit.package / f"analysis/out/arr/{stem}.json"
        man_path = path.with_name(stem + "_manifest.json")
        archived = audit.backup(path)
        audit.backup(man_path)
        old = json.loads(archived.read_text(encoding="utf8"))
        result = copy.deepcopy(old)
        for setting, value in result["settings"].items():
            cost = generator.policies(G, setting)["V5"]["generated_tokens_per_item"]
            for rule, rec in value["rules"].items():
                pol = rec["policy"]
                if pol["operator"] == "s1" and pol["signal"] in ("agreement", "margin"):
                    target = rec["threshold_transfer"]
                    before = target["test_generated_tokens"]
                    target["test_generated_tokens"] = cost
                    assert before >= cost and abs(rec["test_generated_tokens"] - cost) < 1e-8
                    changes.append({"file": stem + ".json", "setting": setting, "rule": rule,
                                    "old_tokens_per_item": before, "corrected_tokens_per_item": cost,
                                    "removed_duplicate_tokens_per_item": before - cost})
        dump(path, result)
        # Keep the original run's provenance in history; identify this as a cost-only
        # revision, rather than representing the original run as having this output.
        original_manifest = audit.out / "history" / man_path.relative_to(audit.package)
        manifest = json.loads(original_manifest.read_text(encoding="utf8"))
        manifest["outputs"][path.name] = sha(path)
        manifest["revision"] = {"id": VERSION, "kind": "stored-output cost correction only",
                                "original_manifest": relative(original_manifest, audit.package),
                                "original_manifest_sha256": sha(original_manifest),
                                "corrected_script_sha256": sha(audit.package / "analysis/arr/a3_dev_test.py"),
                                "analysis": "analysis/neurocomputing_revision.py"}
        dump(man_path, manifest)
        audit.record(path)
        audit.record(man_path)
        audit.inputs[relative(archived, audit.package)] = sha(archived)
    audit.checks.append("Threshold cost matches one acquisition of the five samples; accuracy, selections and intervals unchanged.")
    return changes


def a0_means(audit, G):
    result = {}
    all_math = audit.rows("analysis/out/arr/math_common_labels.jsonl.gz")
    for s in SETTINGS[:4]:
        rows = [r for r in all_math if r["model"] == s.split()[1]]
        result[s] = float(np.mean([r["initial"]["out_tokens"] for r in rows]))
        assert abs(result[s] - G["b2_keep_strength"]["settings"][s]["a0_tokens_per_item"]) < 1e-8
    for s in SETTINGS[4:]:
        rows = [r for r in audit.rows(f"paper/1区/artifacts/mc/{s.split()[1]}_items.jsonl") if r["condition"] == "F2"]
        assert len(rows) == len({r["item_id"] for r in rows})
        result[s] = float(np.mean([r["initial_output_tokens"] for r in rows]))
    for s, entry in G["c1bt"]["settings"].items():
        assert abs(result[s] - entry["a0_tokens_per_item"]) < 1e-8
    audit.checks.append("Initial-answer mean token costs reproduce B2 and all three equal-total budget settings.")
    return result


def verify_oracles(audit, data):
    """Independently recover both orderings from per-item labels, without a scorer."""
    populations = {}
    math_rows = audit.rows("analysis/out/arr/math_common_labels.jsonl.gz")
    samples = audit.rows("analysis/out/arr/b1_labels.jsonl.gz")
    first_samples = {(r["model"], r["item_id"]): r["correct"] for r in samples if r["index"] == 0}
    for r in audit.rows("paper/1区/acl2027/artifacts/math_vote/per_item.jsonl"):
        first_samples["4B", r["item_id"]] = int(r["candidates"][0]["correct_after_choice"])
    for setting in SETTINGS[:4]:
        model = setting.split()[1]
        rows = sorted((r for r in math_rows if r["model"] == model), key=lambda r:r["item_id"])
        populations[setting] = {"ids": [r["item_id"] for r in rows], "y0": [r["initial"]["correct"] for r in rows],
            "ops": {op: [r[op]["correct"] for r in rows] for op in OPS[:3]}}
        populations[setting]["ops"]["s1"] = [first_samples[model, r["item_id"]] for r in rows]
    mc_summary = audit.json("paper/1区/artifacts/mc/summary.json")
    for setting in SETTINGS[4:]:
        model = setting.split()[1]
        rows = audit.rows(f"paper/1区/artifacts/mc/{model}_items.jsonl")
        by_op = {op: sorted((r for r in rows if r["condition"] == op), key=lambda r:r["item_id"]) for op in OPS[:3]}
        base = by_op["F2"]
        populations[setting] = {"ids": [r["item_id"] for r in base], "y0": [r["y0"] for r in base],
            "ops": {op:[r["y1"] for r in rr] for op,rr in by_op.items()}}
        samples = {r["item_id"]:r for r in audit.rows(mc_summary["models"][model]["runs"]["consist"] + "/consistency.jsonl")}
        populations[setting]["ops"]["s1"] = [int(samples[r["item_id"]]["samples"][0]["correct"]) for r in base]
    thinking = audit.rows("analysis/out/arr/c1_labels.jsonl.gz")
    for setting, pop in populations.items():
        if "THINK" in data[setting]:
            labels = {r["item_id"]:r["correct"] for r in thinking if r["setting"] == setting and r["source"] == "C1"}
            pop["ops"]["THINK"] = [labels[i] for i in pop["ids"]]
        n = len(pop["ids"])
        m = round(.2*n)
        y0 = np.asarray(pop["y0"], dtype=int)
        tie = [int(hashlib.sha256(i.encode()).hexdigest()[:12],16) for i in pop["ids"]]
        wrong_order = np.lexsort((tie,y0))[:m]
        for op, ys in pop["ops"].items():
            gain = np.asarray(ys,dtype=int)-y0
            benefit = 100*float(np.sort(gain)[-m:].sum())/n
            detect = 100*float(gain[wrong_order].sum())/n
            assert abs(benefit-data[setting][op]["benefit"]["delta_pp"]) < 1e-8, (setting,op,"benefit")
            assert abs(detect-data[setting][op]["detect"]["delta_pp"]) < 1e-8, (setting,op,"detect")
    audit.checks.append("Both oracle columns independently match sorting the per-item outcomes in every available setting/operator cell.")


def oracle_table(audit, G, a0, generator):
    indexed = {}
    for row in G["curves"] + G["c1curves"]:
        if row.get("labels", "common") == "common" and int(float(row["share_pct"])) == 20 and row["signal"] in ("REF detect", "REF benefit"):
            indexed[row["setting"], row["operator"], row["signal"]] = row
    values, data = {}, {}
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Setting & Review & Critique & Re-solve & Resample & Think \\", r"\midrule"]
    for setting in SETTINGS:
        if setting == SETTINGS[4]:
            lines.append(r"\midrule")
        data[setting] = {}
        cells = [generator.setting_name(setting, short=True)]
        for op in OPS:
            if (setting, op, "REF detect") not in indexed:
                cells.append("--")
                continue
            pair = {name: indexed[setting, op, "REF " + name] for name in ("detect", "benefit")}
            ds = {name: float(row["delta_pp"]) for name, row in pair.items()}
            assert ds["benefit"] >= ds["detect"] - 1e-9
            if op == "F2":
                for name, stored in (("detect", "REF DETECT->F2 20%"), ("benefit", "REF BENEFIT->F2 20%")):
                    ref = G["a1_policies"]["settings"][setting]["policies"][stored]
                    assert abs(ds[name] - ref["delta_vs_keep"]["estimate"]) < 1e-8
            data[setting][op] = {name: {"delta_pp": ds[name], "incremental_generated_tokens": float(row["generated_tokens_per_item"]),
                                      "total_generated_tokens": a0[setting] + float(row["generated_tokens_per_item"])} for name, row in pair.items()}
            cells.append(f"{generator.fnum(ds['detect'])} / {generator.fnum(ds['benefit'])}")
            for name, val in ds.items():
                values[f"neuro.{SLUGS[setting]}.bound.{op}.20.{name}"] = generator.fnum(val, sign=False)
        lines.append(" & ".join(cells) + r" \\")
    for op in OPS:
        for name in ("detect", "benefit"):
            pairs = [(s, d[op][name]["delta_pp"]) for s, d in data.items() if op in d]
            prefix = "bound" if name == "benefit" else "detect"
            values[f"neuro.{prefix}.{op}.20.max"] = generator.fnum(max(v for _, v in pairs), sign=False)
            values[f"neuro.{prefix}.{op}.20.mathmax"] = generator.fnum(max(v for s, v in pairs if s.startswith("math")), sign=False)
    lines.extend([r"\bottomrule", r"\end{tabular}",
        r"\caption{Wrong-first reference / benefit-oracle upper bound at a 20\% selection share (accuracy change in pp; descriptive reanalysis of stored outputs). The reference orders wrong initial answers first, breaking ties by the fixed hash; it is not an upper bound. The benefit oracle orders $y_F-y_0$ from largest to smallest and attains the maximum gain among subsets of the same cardinality for this operator's stored outcomes. Both read gold labels; neither is a deployable policy. These are share-constrained, not token-constrained, comparisons.}",
        r"\label{tab:bounds}", r"\end{table*}", ""])
    target = audit.manuscript / "generated/tab_bounds.tex"
    audit.backup(target)
    target.write_text("\n".join(lines), encoding="utf8", newline="\n")
    audit.record(target)
    audit.checks.append("Benefit oracle is never below the wrong-first reference; all six critique values match the separately stored A1 policies.")
    return values, data


def frontier(audit, G, a0, generator):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # The legacy renderer changes global matplotlib defaults before calling us.
    # Reset them so standalone and automatic post-processing are byte-identical.
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .6})
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.25))
    blue, orange, green, black, grey = "#2465b3", "#d25b23", "#16856d", "#202020", "#6c6c6c"
    records = []
    def point(ax, setting, name, entry, requires_a0, **kwargs):
        delta = entry["delta_vs_keep"]
        stored = entry["generated_tokens_per_item"]
        total = stored + (a0[setting] if requires_a0 else 0)
        ax.errorbar(total, delta["estimate"], yerr=[[delta["estimate"]-delta["lo"]], [delta["hi"]-delta["estimate"]]],
                    elinewidth=.7, capsize=1.4, zorder=4, **kwargs)
        records.append({"setting": setting, "policy": name, "requires_a0": requires_a0,
                        "stored_generated_tokens": stored, "total_generated_tokens": total,
                        "delta_pp": delta["estimate"]})
        return total
    for ax, setting in zip(axes.flat, SETTINGS):
        P = generator.policies(G, setting)
        ax.axhline(0, color=grey, lw=.6)
        ax.grid(axis="y", color="#e2e5e9", lw=.5)
        ax.plot(a0[setting], 0, "x", color=black, ms=4, mew=.8, zorder=4)
        records.append({"setting": setting, "policy": "KEEP", "requires_a0": True,
                        "stored_generated_tokens": 0., "total_generated_tokens": a0[setting], "delta_pp": 0.})
        xs, ys = zip(*generator.curve(G, setting, "F2", "verbal"))
        ax.plot(np.array(xs)+a0[setting], ys, color=grey, ls=":", lw=1)
        for name, mark in (("F0 ALL", "v"), ("F2 ALL", "s"), ("F2-scratch ALL", "D")):
            point(ax, setting, name, P[name], True, fmt=mark, ms=3.5, color=black, mfc="white", mew=.8)
        names = [n for n in ("V1", "V3", "V5", "V8") if n in P]
        ax.plot([P[n]["generated_tokens_per_item"] for n in names], [P[n]["delta_vs_keep"]["estimate"] for n in names], color=blue, lw=1)
        for name in names:
            x = point(ax, setting, name, P[name], False, fmt="o", ms=3, color=blue, mfc=blue, mew=.6)
            if name in ("V1", "V5", "V8"):
                ax.annotate(name.replace("V", "k="), (x, P[name]["delta_vs_keep"]["estimate"]), xytext=(2,-8), textcoords="offset points", fontsize=6, color=grey)
        gate = P.get("GATE-a0 K5", P.get("GATE-a0"))
        point(ax, setting, "gate then vote", gate, True, fmt="^", ms=4, color=blue, mfc="white", mew=.8)
        point(ax, setting, "hybrid margin 20%", P["HYBRID-margin 20%"], True, fmt="P", ms=4, color=green, mfc=green, mew=.6)
        xmax = max(r["total_generated_tokens"] for r in records if r["setting"] == setting)
        if setting in G["c1_think"]["settings"]:
            for sig, style in (("verbal", "-"), ("probe", "--")):
                xs, ys = zip(*generator.curve(G, setting, "THINK", sig))
                ax.plot(np.array(xs)+a0[setting], ys, color=orange, lw=1, ls=style)
                xmax = max(xmax, max(xs)+a0[setting])
            point(ax, setting, "standalone think", G["c1_think"]["settings"][setting]["policies"]["THINK ALL"], False,
                  fmt="*", ms=6.5, color=orange, mfc=orange, mew=.5)
            budgets = G["c1b"]["settings"][setting]["budgets"]
            ordered = sorted(budgets, key=int)
            ax.plot([budgets[b]["generated_tokens_per_item"] for b in ordered], [budgets[b]["delta_vs_keep"]["estimate"] for b in ordered], color=orange, ls=":", lw=.9)
            for b in ordered:
                point(ax, setting, "uniform think B="+b, budgets[b], False, fmt="o", ms=3.5, color=orange, mfc="white", mew=.8)
        ax.set_xlim(0, xmax*1.07)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v/1000:g}k" if v else "0"))
        ax.set_title(generator.setting_name(setting), loc="left", pad=5)
    for ax in axes[1]:
        ax.set_xlabel("Total generated tokens per item")
    for ax in axes[:,0]:
        ax.set_ylabel("Accuracy change vs. KEEP (pp)")
    handles = [Line2D([],[],color=black,marker="x",ls="none",ms=4,label="KEEP"),
        Line2D([],[],color=black,marker="v",ls="none",mfc="white",ms=4,label="review, all"),
        Line2D([],[],color=black,marker="s",ls="none",mfc="white",ms=4,label="critique, all"),
        Line2D([],[],color=black,marker="D",ls="none",mfc="white",ms=4,label="re-solve, all"),
        Line2D([],[],color=grey,ls=":",lw=1,label="critique, verbal share"),
        Line2D([],[],color=blue,marker="o",lw=1,ms=3,label="standalone vote"),
        Line2D([],[],color=blue,marker="^",ls="none",mfc="white",ms=4,label="gate, then vote"),
        Line2D([],[],color=green,marker="P",ls="none",ms=4,label="hybrid (critique 20%)"),
        Line2D([],[],color=orange,marker="*",ls="none",ms=6,label="standalone think"),
        Line2D([],[],color=orange,lw=1,label="think, verbal share"),
        Line2D([],[],color=orange,lw=1,ls="--",label="think, probe share"),
        Line2D([],[],color=orange,marker="o",mfc="white",ms=3.5,ls=":",lw=.9,label="uniform think budget")]
    fig.legend(handles=handles,loc="lower center",ncol=4,frameon=False,handlelength=2,columnspacing=1.3,bbox_to_anchor=(.5,.005))
    fig.tight_layout(rect=(0,.17,1,1),h_pad=.9,w_pad=.7)
    pdf = audit.manuscript / "figures/fig_frontier.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    audit.backup(pdf)
    fig.savefig(pdf, metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None})
    preview = audit.out / "fig_frontier_preview.png"
    fig.savefig(preview, dpi=180)
    plt.close(fig)
    audit.record(pdf)
    audit.record(preview)
    audit.checks.append("Figure uses total generated tokens: a0-dependent pipelines pay a0; standalone voting, thinking and uniform budgets do not.")
    return records


def apply_revision(package, manuscript):
    audit = Audit(package, manuscript)
    audit.out.mkdir(parents=True, exist_ok=True)
    generator = legacy_module(audit.package)
    G = generator.load_all()
    audit.inputs.update(generator.INPUTS)
    for rel in ("analysis/arr/a3_dev_test.py", "analysis/arr/acl_v2_tex.py", "analysis/neurocomputing_revision.py"):
        audit.read(rel)
    check_threshold_function(audit.package)
    audit.checks.append("Actual repaired threshold function passes reused-sample, fresh-operator and zero-generation-signal cost checks.")
    corrections = correct_thresholds(audit, G, generator)
    a0 = a0_means(audit, G)
    values, bounds = oracle_table(audit, G, a0, generator)
    verify_oracles(audit, bounds)
    settings_path = audit.manuscript / "generated/tab_settings.tex"
    audit.backup(settings_path)
    settings_text = settings_path.read_text(encoding="utf8").replace("{Free signals}", "{No extra generation}").replace(
        "Free signals generate no tokens;", "Verbal, probe and length generate no additional tokens; verbal and probe still require a forward pass;")
    settings_path.write_text(settings_text, encoding="utf8", newline="\n")
    audit.record(settings_path)
    thinking_path = audit.manuscript / "generated/tab_think_full.tex"
    audit.backup(thinking_path)
    thinking_text = thinking_path.read_text(encoding="utf8").replace(
        "bound: wrong first", "oracle: wrong first").replace(
        "bound: repairable first", "oracle: benefit").replace(
        "The two bounds use the gold answer and are not selectable policies.",
        "The wrong-first oracle is a reference, not an upper bound. The benefit oracle maximises gain at the fixed selection share for the stored outputs. Both use gold answers and are not deployable policies.")
    thinking_path.write_text(thinking_text, encoding="utf8", newline="\n")
    audit.record(thinking_path)
    for setting, cost in a0.items():
        values[f"neuro.{SLUGS[setting]}.a0.tok"] = generator.ftok(cost)
    values["neuro.threshold.corrected_cells"] = str(len(corrections))
    macro_path = audit.manuscript / "generated/neuro_values.tex"
    macro_path.write_text("% Generated from stored outputs by analysis/neurocomputing_revision.py; do not edit.\n" +
                         "\n".join(f"\\expandafter\\def\\csname val@{k}\\endcsname{{{v}}}" for k,v in sorted(values.items())) + "\n", encoding="utf8", newline="\n")
    audit.record(macro_path)
    points = frontier(audit, G, a0, generator)
    summary = {"id": VERSION, "status": "descriptive post hoc reanalysis of existing stored outputs",
               "new_model_runs": False, "new_hypothesis_tests": False,
               "cost_scope": "generated tokens only; excludes prefill, signal forward passes, latency, energy and price",
               "oracle_definition": "For fixed operator outputs, sum the largest round(0.2*n) values of yF-y0, divided by n; wrong-first with a fixed hash is only a reference.",
               "threshold_corrections": corrections, "a0_tokens_per_item": a0, "oracle_20pct": bounds,
               "figure_points": points, "checks": audit.checks}
    summary_path = audit.out / "revision.json"
    dump(summary_path, summary)
    audit.record(summary_path)
    # Record current hashes for corrected inputs, with the originals separately archived.
    for rel in list(audit.inputs):
        path = audit.package / rel
        if path.exists():
            audit.inputs[rel] = sha(path)
    manifest = {"id": VERSION, "analysis": "analysis/neurocomputing_revision.py", "new_model_runs": False,
                "new_hypothesis_tests": False, "inputs": dict(sorted(audit.inputs.items())),
                "outputs": dict(sorted(audit.outputs.items())), "checks": audit.checks}
    dump(audit.out / "manifest.json", manifest)
    return manifest


def main():
    here = Path(__file__).resolve().parents[1]
    default_package = here if (here / "analysis/out/arr").is_dir() and (here / "analysis/arr/acl_v2_tex.py").exists() and here.name == "supplementary" else here / "paper/Neurocomputing/supplementary"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=default_package)
    parser.add_argument("--manuscript", type=Path)
    args = parser.parse_args()
    manuscript = args.manuscript or args.package.parent / "manuscript"
    manifest = apply_revision(args.package, manuscript)
    print(json.dumps({"revision": VERSION, "outputs": len(manifest["outputs"]), "checks": manifest["checks"]}, indent=1))


if __name__ == "__main__":
    main()
