"""Retrospective grouped threshold-transfer sensitivity from stored outputs only.

No model generation, model/probe fitting, rescoring, hypothesis test, interval, or p-value.
These already-inspected test records are NOT a new independent validation set.
The threshold-selection function receives signals and costs, never correctness.

Repository: python -B analysis/neurocomputing_crossfit_v2.py
Portable supplementary package: python -B analysis/neurocomputing_crossfit_v2.py
Optional: --package PATH --output PATH. Python standard library only.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path


VERSION = "neurocomputing-v2-retrospective-crossfit-v1"
FOLDS = 5
FOLD_SALT = "neurocomputing-v2-cost-only-threshold-transfer-fixed-20261111"
BUDGETS = (256, 512, 1024, 2048)
SIGNALS = ("verbal", "probe")
SETTINGS = ("math 4B", "math 8B", "MMLU-Pro 4B")
SLUG = {"math 4B": "m4", "math 8B": "m8", "MMLU-Pro 4B": "c4"}
DISPLAY = {"math 4B": "Math 4B", "math 8B": "Math 8B", "MMLU-Pro 4B": "MMLU-Pro 4B"}
ARR = "analysis/out/arr/"
MATH_ITEMS = "paper/1区/artifacts/math/item_evaluations.jsonl"
MC_ITEMS = "paper/1区/artifacts/mc/4B_items.jsonl"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def mean(values):
    values = list(values)
    assert values
    return math.fsum(values) / len(values)


def close(a, b, label):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-8), (label, a, b)


class Inputs:
    def __init__(self, package):
        self.package = package
        self.hashes = {}
        manifest = package / ARR / "c1_budget_total_manifest.json"
        self.hashes[ARR + manifest.name] = {"sha256": sha(manifest), "bytes": manifest.stat().st_size}
        self.original = json.loads(manifest.read_text(encoding="utf8"))

    def read(self, relative):
        path = self.package / relative
        digest = sha(path)
        historical = self.original["inputs"].get(relative)
        assert historical is not None, ("no historical input checksum", relative)
        assert digest == historical["sha256"], ("input differs from original total-budget analysis", relative)
        self.hashes[relative] = {"sha256": digest, "bytes": path.stat().st_size,
                                 "verified_against": ARR + "c1_budget_total_manifest.json"}
        return path

    def json(self, relative):
        return json.loads(self.read(relative).read_text(encoding="utf8"))


def unique(records, key):
    result = {}
    for row in records:
        k = key(row)
        assert k not in result, ("duplicate input row", k)
        result[k] = row
    return result


def load_populations(inputs):
    labels = unique((r for r in rows(inputs.read(ARR + "math_common_labels.jsonl.gz"))
                     if r["model"] in ("4B", "8B")), lambda r: (r["model"], r["item_id"]))
    initial = unique((r for r in rows(inputs.read(MATH_ITEMS))
                      if r["model"] in ("4B", "8B") and r["condition"] == "F2"),
                     lambda r: (r["model"], r["item_id"]))
    assert set(labels) == set(initial)
    populations = {setting: [] for setting in SETTINGS}
    for key in sorted(labels):
        lab, raw = labels[key], initial[key]
        assert lab["canonical_id"] == raw["canonical_id"]
        assert raw["split"] == "test"
        assert lab["initial"]["hist_correct"] == raw["y0"]
        assert lab["initial"]["out_tokens"] == raw["initial_output_tokens"]
        populations["math " + key[0]].append({
            "item_id": key[1], "canonical_id": lab["canonical_id"], "task": "math",
            "source": lab["dataset"], "y0": int(lab["initial"]["correct"]),
            "a0_tokens": int(raw["initial_output_tokens"]),
            "verbal": float(raw["verbal_log_odds"]), "probe": float(raw["probe_p_correct"])})
    mc = unique((r for r in rows(inputs.read(MC_ITEMS)) if r["condition"] == "F2"), lambda r: r["item_id"])
    for item_id, raw in sorted(mc.items()):
        assert raw["split"] == "test"
        populations["MMLU-Pro 4B"].append({
            "item_id": item_id, "canonical_id": raw["canonical_id"], "task": "mc",
            "source": raw["category"], "y0": int(raw["y0"]),
            "a0_tokens": int(raw["initial_output_tokens"]),
            "verbal": float(raw["verbal_W0"]), "probe": float(raw["task_probe"])})
    think = unique((r for r in rows(inputs.read(ARR + "c1_labels.jsonl.gz")) if r["source"] == "C1"),
                   lambda r: (r["setting"], r["item_id"]))
    forced = unique(rows(inputs.read(ARR + "c1b_labels.jsonl.gz")),
                    lambda r: (r["setting"], int(r["budget"]), r["item_id"]))
    assert set(think) == {(s, r["item_id"]) for s, pop in populations.items() for r in pop}
    assert set(forced) <= {(s, b, r["item_id"]) for s, pop in populations.items() for b in BUDGETS for r in pop}
    for setting, population in populations.items():
        for record in population:
            t = think[setting, record["item_id"]]
            record["think_y"] = int(t["correct"])
            record["think_tokens"] = int(t["out_tokens"])
            record["uniform"] = {}
            for budget in BUDGETS:
                key = setting, budget, record["item_id"]
                u = forced.get(key, t)
                record["uniform"][budget] = {"y": int(u["correct"]), "tokens": int(u["out_tokens"]),
                                               "forced": key in forced}
            assert record["y0"] in (0, 1) and record["think_y"] in (0, 1)
            assert record["a0_tokens"] >= 0 and record["think_tokens"] >= 0
            assert all(math.isfinite(record[signal]) for signal in SIGNALS)
            assert all(u["y"] in (0, 1) and u["tokens"] >= 0 for u in record["uniform"].values())
    return populations


def check_parity(inputs, populations):
    c1, c1b, b2 = (inputs.json(ARR + name) for name in ("c1_think.json", "c1b_budget.json", "b2_keep_strength.json"))
    assert c1b["budgets"] == list(BUDGETS) and c1b["signals"] == list(SIGNALS)
    checks = []
    for setting, population in populations.items():
        ta = c1["settings"][setting]["policies"]["THINK ALL"]
        close(100 * mean(r["think_y"] for r in population), ta["accuracy_pct"], (setting, "THINK accuracy"))
        close(mean(r["think_tokens"] for r in population), ta["generated_tokens_per_item"], (setting, "THINK cost"))
        if setting in b2["settings"]:
            close(mean(r["a0_tokens"] for r in population), b2["settings"][setting]["a0_tokens_per_item"], (setting, "a0 cost"))
        for budget in BUDGETS:
            ref = c1b["settings"][setting]["budgets"][str(budget)]
            close(100 * mean(r["uniform"][budget]["y"] for r in population), ref["accuracy_pct"], (setting, budget, "uniform accuracy"))
            close(mean(r["uniform"][budget]["tokens"] for r in population), ref["generated_tokens_per_item"], (setting, budget, "uniform cost"))
            assert sum(r["uniform"][budget]["forced"] for r in population) == ref["forced_items"]
        checks.append(f"{setting}: reconstructed full thinking and every uniform budget reproduce historical accuracy, tokens and forced-item counts.")
    return checks


def assign_folds(populations):
    tasks = collections.defaultdict(set)
    for population in populations.values():
        for record in population:
            tasks[record["task"]].add(record["canonical_id"])
    assignments = {}
    for task, groups in sorted(tasks.items()):
        ordered = sorted(groups, key=lambda group: (hashlib.sha256(
            (FOLD_SALT + "|" + task + "|" + group).encode("utf8")).hexdigest(), group))
        assignments[task] = {group: index % FOLDS for index, group in enumerate(ordered)}
    for population in populations.values():
        for record in population:
            record["fold"] = assignments[record["task"]][record["canonical_id"]]
    math_maps = [{r["item_id"]: (r["canonical_id"], r["fold"]) for r in populations[s]} for s in SETTINGS[:2]]
    assert math_maps[0] == math_maps[1], "mathematics models must share item grouping and fold assignment"
    return {"salt": FOLD_SALT, "fold_count": FOLDS,
            "rule": "Within task, sort unique canonical IDs by SHA-256(salt|task|canonical_id), then assign round-robin; models share task-level folds.",
            "assignments": assignments}


def calibrate_threshold(scores, a0_tokens, think_tokens, uniform_tokens):
    """Cost-only calibration API. No correctness values or held-out data accepted.

    Candidates are KEEP, each whole training-score tie group, and ALL. Choose the
    largest threshold fitting the training uniform total. If KEEP is infeasible,
    retain KEEP and mark infeasibility; do not conceal its budget excess.
    """
    n = len(scores)
    assert n and len(a0_tokens) == len(think_tokens) == len(uniform_tokens) == n
    base, target = sum(a0_tokens), sum(uniform_tokens)
    available = target - base
    rule = {"kind": "none", "value": None, "comparison": "score <= threshold; ties included together",
            "training_budget_feasible": available >= 0, "training_n": n,
            "training_a0_tokens_per_item": base / n, "training_uniform_tokens_per_item": target / n,
            "training_selective_tokens_per_item": base / n, "training_selected_n": 0}
    if available < 0:
        return rule
    spent, count, last = 0, 0, None
    for score, group in itertools.groupby(sorted(zip(scores, think_tokens)), key=lambda row: row[0]):
        group = list(group)
        cost = sum(row[1] for row in group)
        if spent + cost > available:
            break
        spent += cost
        count += len(group)
        last = score
    if count:
        rule.update(kind="all" if count == n else "threshold", value=None if count == n else last)
    rule.update(training_selected_n=count, training_selective_tokens_per_item=(base + spent) / n)
    return rule


def apply_threshold(score, rule):
    if rule["kind"] == "none":
        return False
    if rule["kind"] == "all":
        return True
    assert rule["kind"] == "threshold" and rule["value"] is not None
    return score <= rule["value"]


def verify_calibration(scores, a0, think, uniform, rule):
    """Independent exhaustive boundary check of the cost-only training rule."""
    boundary_costs = collections.defaultdict(int)
    boundary_counts = collections.Counter(scores)
    for score, tokens in zip(scores, think):
        boundary_costs[score] += tokens
    target, base = sum(uniform), sum(a0)
    total, count, candidates = base, 0, [(0, base)]
    for score in sorted(boundary_costs):
        total += boundary_costs[score]
        count += boundary_counts[score]
        candidates.append((count, total))
    feasible = [(count, cost) for count, cost in candidates if cost <= target]
    expected_count = max((count for count, cost in feasible), default=0)
    mask = [apply_threshold(score, rule) for score in scores]
    assert sum(mask) == rule["training_selected_n"] == expected_count
    expected_cost = base + sum(tokens for flag, tokens in zip(mask, think) if flag)
    close(expected_cost / len(scores), rule["training_selective_tokens_per_item"], "threshold training cost")
    assert rule["training_budget_feasible"] == bool(feasible)
    assert len({apply_threshold(score, rule) for score in scores if score == scores[0]}) == 1


def summarize(population, selected, budget, target_tokens):
    n = len(population)
    keep = sum(r["y0"] for r in population)
    uniform_correct = sum(r["uniform"][budget]["y"] for r in population)
    selective_correct = sum(r["think_y"] if flag else r["y0"] for r, flag in zip(population, selected))
    uni_tokens = sum(r["uniform"][budget]["tokens"] for r in population)
    sel_tokens = sum(r["a0_tokens"] + flag * r["think_tokens"] for r, flag in zip(population, selected))
    target_total = math.fsum(target_tokens)
    return {"n": n, "canonical_groups": len({r["canonical_id"] for r in population}),
            "selected_n": sum(selected), "selected_share_pct": 100 * sum(selected) / n,
            "keep_accuracy_pct": 100 * keep / n, "uniform_accuracy_pct": 100 * uniform_correct / n,
            "selective_accuracy_pct": 100 * selective_correct / n,
            "uniform_delta_vs_keep_pp": 100 * (uniform_correct - keep) / n,
            "selective_delta_vs_keep_pp": 100 * (selective_correct - keep) / n,
            "selective_minus_uniform_pp": 100 * (selective_correct - uniform_correct) / n,
            "uniform_total_tokens": uni_tokens, "selective_total_tokens": sel_tokens,
            "uniform_tokens_per_item": uni_tokens / n, "selective_tokens_per_item": sel_tokens / n,
            "cost_gap_vs_heldout_uniform_tokens_per_item": (sel_tokens - uni_tokens) / n,
            "cost_drift_vs_heldout_uniform_pct": 100 * (sel_tokens - uni_tokens) / uni_tokens,
            "overshoots_heldout_uniform": sel_tokens > uni_tokens,
            "calibration_target_tokens_per_item": target_total / n,
            "cost_gap_vs_calibration_target_tokens_per_item": (sel_tokens - target_total) / n,
            "cost_drift_vs_calibration_target_pct": 100 * (sel_tokens - target_total) / target_total}


def evaluate(populations):
    result = {}
    for setting, population in populations.items():
        entry = {"n": len(population), "canonical_groups": len({r["canonical_id"] for r in population}),
                 "fold_sizes": {str(f): sum(r["fold"] == f for r in population) for f in range(FOLDS)},
                 "budgets": {}}
        for budget in BUDGETS:
            entry["budgets"][str(budget)] = {}
            for signal in SIGNALS:
                selected = [None] * len(population)
                targets = [None] * len(population)
                folds = []
                for fold in range(FOLDS):
                    train = [r for r in population if r["fold"] != fold]
                    indices = [i for i, r in enumerate(population) if r["fold"] == fold]
                    heldout = [population[i] for i in indices]
                    assert heldout and train
                    assert {r["canonical_id"] for r in train}.isdisjoint(r["canonical_id"] for r in heldout)
                    args = ([r[signal] for r in train], [r["a0_tokens"] for r in train],
                            [r["think_tokens"] for r in train], [r["uniform"][budget]["tokens"] for r in train])
                    rule = calibrate_threshold(*args)
                    verify_calibration(*args, rule)
                    # The frozen rule is applied before any held-out correctness
                    # or cost enters the evaluation; no budget repair is performed.
                    decisions = [apply_threshold(r[signal], rule) for r in heldout]
                    for index, decision in zip(indices, decisions):
                        assert selected[index] is None
                        selected[index] = decision
                        targets[index] = rule["training_uniform_tokens_per_item"]
                    fold_summary = summarize(heldout, decisions, budget,
                                             [rule["training_uniform_tokens_per_item"]] * len(heldout))
                    folds.append({"fold": fold, "threshold_rule": rule, "heldout": fold_summary})
                assert all(isinstance(flag, bool) for flag in selected) and all(value is not None for value in targets)
                summary = summarize(population, selected, budget, targets)
                assert sum(f["heldout"]["selected_n"] for f in folds) == summary["selected_n"]
                assert sum(f["heldout"]["selective_total_tokens"] for f in folds) == summary["selective_total_tokens"]
                summary.update(folds_overshooting_heldout_uniform=sum(f["heldout"]["overshoots_heldout_uniform"] for f in folds),
                               infeasible_training_folds=sum(not f["threshold_rule"]["training_budget_feasible"] for f in folds),
                               minimum_fold_cost_drift_pct=min(f["heldout"]["cost_drift_vs_heldout_uniform_pct"] for f in folds),
                               maximum_fold_cost_drift_pct=max(f["heldout"]["cost_drift_vs_heldout_uniform_pct"] for f in folds))
                entry["budgets"][str(budget)][signal] = {"pooled_out_of_fold": summary, "folds": folds}
        result[setting] = entry
    return result


def signed(value, digits=2):
    return f"{value:+.{digits}f}" if abs(value) >= 0.5 * 10 ** (-digits) else f"{0:.{digits}f}"


def tex_minus(text):
    """Typeset a minus sign in LaTeX text mode, as the other generated tables do (formatting only)."""
    return text.replace("-", "$-$")


def render_outputs(out, result):
    values = {"cfv2.folds": str(FOLDS), "cfv2.cells": str(len(SETTINGS) * len(BUDGETS) * len(SIGNALS))}
    fields = {"gain": "selective_delta_vs_keep_pp", "gap": "selective_minus_uniform_pp",
              "share": "selected_share_pct", "tok": "selective_tokens_per_item",
              "uniformgain": "uniform_delta_vs_keep_pp", "uniformtok": "uniform_tokens_per_item",
              "drift": "cost_drift_vs_heldout_uniform_pct", "tokgap": "cost_gap_vs_heldout_uniform_tokens_per_item",
              "targetdrift": "cost_drift_vs_calibration_target_pct",
              "foldovershoot": "folds_overshooting_heldout_uniform", "infeasible": "infeasible_training_folds"}
    for setting, entry in result["settings"].items():
        for budget, signals in entry["budgets"].items():
            for signal, record in signals.items():
                row = record["pooled_out_of_fold"]
                for suffix, field in fields.items():
                    value = row[field]
                    digits = 0 if suffix in ("tok", "uniformtok", "foldovershoot", "infeasible") else 2
                    values[f"cfv2.{SLUG[setting]}.b{budget}.{signal}.{suffix}"] = tex_minus(f"{value:.{digits}f}")
    for domain, settings in (("math", SETTINGS[:2]), ("mc", SETTINGS[2:])):
        for budget_scope, budgets in (("all", BUDGETS), ("b512plus", BUDGETS[1:])):
            for signal in SIGNALS:
                subset = [result["settings"][s]["budgets"][str(b)][signal]["pooled_out_of_fold"] for s in settings for b in budgets]
                prefix = f"cfv2.{domain}.{budget_scope}.{signal}"
                for label, field in (("gap", "selective_minus_uniform_pp"), ("gain", "selective_delta_vs_keep_pp"),
                                     ("drift", "cost_drift_vs_heldout_uniform_pct")):
                    values[prefix + "." + label + "min"] = tex_minus(f"{min(r[field] for r in subset):.2f}")
                    values[prefix + "." + label + "max"] = tex_minus(f"{max(r[field] for r in subset):.2f}")
                values[prefix + ".positivegap"] = str(sum(r["selective_minus_uniform_pp"] > 0 for r in subset))
                values[prefix + ".overshoot"] = str(sum(r["overshoots_heldout_uniform"] for r in subset))
                values[prefix + ".cells"] = str(len(subset))
    lines = ["% Generated by analysis/neurocomputing_crossfit_v2.py; descriptive retrospective sensitivity."]
    lines += [f"\\expandafter\\def\\csname val@{key}\\endcsname{{{value}}}" for key, value in sorted(values.items())]
    (out / "crossfit_values.tex").write_text("\n".join(lines) + "\n", encoding="utf8", newline="\n")

    table = [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrlrrrrrr}", r"\toprule",
             r"& & & \multicolumn{3}{c}{Verbal threshold} & \multicolumn{3}{c}{Probe threshold} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Setting & $B$ & Uniform $\Delta$ (tok.) & Share & $\Delta$ [vs U] & Tok. [drift] & Share & $\Delta$ [vs U] & Tok. [drift] \\",
             r"\midrule"]
    mdrows = []
    for index, setting in enumerate(SETTINGS):
        if index:
            table.append(r"\midrule")
        for budget in BUDGETS:
            records = result["settings"][setting]["budgets"][str(budget)]
            uniform = records["verbal"]["pooled_out_of_fold"]
            cells = [DISPLAY[setting] if budget == BUDGETS[0] else "", f"{budget:,}",
                     f"{tex_minus(signed(uniform['uniform_delta_vs_keep_pp']))} ({uniform['uniform_tokens_per_item']:,.0f})"]
            for signal in SIGNALS:
                row = records[signal]["pooled_out_of_fold"]
                cells.extend((f"{row['selected_share_pct']:.2f}\\%",
                              f"{tex_minus(signed(row['selective_delta_vs_keep_pp']))} "
                              f"[{tex_minus(signed(row['selective_minus_uniform_pp']))}]",
                              f"{row['selective_tokens_per_item']:,.0f} "
                              f"[{tex_minus(signed(row['cost_drift_vs_heldout_uniform_pct']))}\\%]"))
                mdrows.append(f"| {setting} | {budget} | {signal} | {row['selected_share_pct']:.2f} | "
                              f"{signed(row['selective_delta_vs_keep_pp'])} | {signed(row['selective_minus_uniform_pp'])} | "
                              f"{row['selective_tokens_per_item']:.2f} | {row['uniform_tokens_per_item']:.2f} | "
                              f"{signed(row['cost_drift_vs_heldout_uniform_pct'])} | {row['folds_overshooting_heldout_uniform']} |")
            table.append(" & ".join(cells) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}",
                  r"\caption{Retrospective grouped five-fold threshold-transfer sensitivity on previously analysed test records; this is not independent validation. In each fold, the largest whole-score threshold fitting the calibration folds' uniform generated-token volume is selected using signals and costs only, then frozen on the held-out fold. Mathematics models share canonical-problem folds. $\Delta$: pooled out-of-fold accuracy change against \KEEP{} (pp); [vs U]: difference from uniform thinking (pp). Tokens include $a_0$ for selective policies. [Drift] is selective cost minus the held-out uniform cost, divided by that uniform cost; positive values are overshoots. No held-out budget adjustment, confidence intervals or significance tests are applied. These are actual-cost comparisons, not matched-cost accuracy claims.}",
                  r"\label{tab:crossfitv2}", r"\end{table*}"])
    (out / "tab_crossfit.tex").write_text("\n".join(table) + "\n", encoding="utf8", newline="\n")
    report = ["# Retrospective grouped threshold-transfer sensitivity", "",
              "Generated from frozen derived records by `analysis/neurocomputing_crossfit_v2.py`. "
              "No model outputs, scorer calls, probe fitting, bootstrap intervals, or hypothesis tests are added.", "",
              "This is a cost-calibration sensitivity analysis on a previously examined test pool. "
              "The analyst already knew the previous outcomes and the design is post hoc; cross-fitting cannot "
              "turn those records into a new independent validation set. The existing probes remain frozen.", "",
              "Canonical problems (question stems for MMLU-Pro) are assigned by the fixed hash rule. "
              "For each held-out fold, threshold calibration uses only the other folds' signal values, initial-answer "
              "tokens, full-thinking tokens, and uniform-budget tokens. Correctness labels never enter threshold "
              "selection. Equal-score groups are kept together. If initial answers alone exceed the training target, "
              "KEEP is retained and the fold is marked infeasible. The held-out decisions are never changed to meet "
              "a held-out token budget.", "",
              "The candidate threshold set is finer than the original whole-pool comparison's one-percentage-point "
              "share grid. Consequently, low budgets can permit a small escalation here even when that grid chose "
              "KEEP. This sensitivity changes calibration and candidate resolution together; it does not isolate "
              "the effect of splitting alone.", "",
              "The table pools one held-out decision per item. Gain is against KEEP and gap is selective minus "
              "uniform accuracy, both in percentage points. Drift is the percentage cost excess over held-out "
              "uniform thinking. Positive drift is an observed budget overshoot. Fold-level targets, thresholds, "
              "infeasibility flags and both held-out-uniform and calibration-target drift are retained in JSON.", "",
              "| Setting | B | Signal | Selected % | Gain pp | Gap pp | Selective total tok/item | Uniform tok/item | Drift % | Overshooting folds |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"] + mdrows
    report += ["", "Limitations: folds share an already selected study population; this does not remove "
               "analysis-selection bias, validate a new deployment threshold, or quantify new-generation variability. "
               "The calibration target itself is the in-fold realized uniform cost, not an externally fixed serving "
               "budget. Costs vary when the frozen threshold transfers. No cost normalization is applied to the "
               "accuracy gaps, so they must be read together with drift. Generated tokens omit scoring, prefill, "
               "caching and hardware costs. Cross-model cascades are outside this within-model sensitivity; token "
               "counts across different model sizes do not establish comparable compute costs.", "", "Checks:", ""]
    report += ["- " + check for check in result["checks"]]
    (out / "crossfit_report.md").write_text("\n".join(report) + "\n", encoding="utf8", newline="\n")


def main():
    script_root = Path(__file__).resolve().parents[1]
    repository_package = script_root / "paper/Neurocomputing_v2/supplementary"
    default_package = repository_package if repository_package.is_dir() else script_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=default_package)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    out = args.output.resolve() if args.output else package / "analysis/out/neurocomputing_v2"
    assert package.is_dir(), "use an existing supplementary data root"
    inputs = Inputs(package)
    populations = load_populations(inputs)
    checks = check_parity(inputs, populations)
    folds = assign_folds(populations)
    result = {"analysis_id": VERSION, "status": "retrospective descriptive sensitivity; not independent validation",
              "budgets": list(BUDGETS), "signals": list(SIGNALS), "fold_count": FOLDS,
              "fold_salt": FOLD_SALT,
              "protocol": {"unit": "canonical problem within task", "shared_mathematics_folds": True,
                           "threshold_selection_inputs": ["training signal scores", "training a0 token counts", "training full-thinking token counts", "training uniform-budget token counts"],
                           "threshold_selection_rule": "Largest complete score threshold whose training a0-plus-thinking tokens do not exceed training uniform tokens; all ties included together; KEEP if even a0 exceeds target, flagged infeasible.",
                           "candidate_resolution_note": "Whole-score thresholds are finer than the original one-percentage-point share grid; this sensitivity does not isolate splitting from candidate-resolution changes.",
                           "heldout_rule": "Apply frozen training threshold without reading held-out outcomes or costs; evaluate actual gain and cost afterward; never adjust to a held-out budget.",
                           "accuracy_used_for_selection": False, "heldout_cost_used_for_selection": False,
                           "probe_refitted": False, "new_model_outputs": False, "new_hypothesis_tests": False,
                           "folds_chosen_using_outcomes": False,
                           "cost_scope": "Generated tokens only; initial answers charged to selective policies, not standalone uniform thinking.",
                           "not_claimed": ["fresh independent validation", "deployment budget guarantee", "matched-cost accuracy advantage", "latency or monetary savings"]},
              "settings": evaluate(populations),
              "checks": checks + ["All input files match checksums in the historical total-budget analysis manifest.",
                                   "Every canonical group belongs to one fold; no calibration/held-out group overlap; mathematics models share identical item-level folds.",
                                   "Every training threshold passes an independent exhaustive whole-score boundary cost check.",
                                   "Every item receives exactly one out-of-fold decision; pooled selections and costs reproduce sums of held-out fold totals.",
                                   "Calibration receives no correctness labels; no selection depends on held-out token counts; all budget overshoots are retained."]}
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "crossfit.json", result)
    dump(out / "crossfit_folds.json", folds)
    render_outputs(out, result)
    output_names = ("crossfit.json", "crossfit_folds.json", "crossfit_values.tex", "tab_crossfit.tex", "crossfit_report.md")
    manifest = {"analysis_id": VERSION, "script": "analysis/neurocomputing_crossfit_v2.py", "script_sha256": sha(Path(__file__)),
                "runtime": "Python standard library; deterministic integer cost accumulation; no random sampler",
                "inputs": dict(sorted(inputs.hashes.items())), "outputs": {name: sha(out / name) for name in output_names},
                "fold_assignment_sha256": sha(out / "crossfit_folds.json"),
                "historical_plan_core_sha256": inputs.original["plan_core_sha256"],
                "historical_plan_note": "The historical plan identifies the input studies only; this new retrospective sensitivity was not preregistered.",
                "checks": result["checks"]}
    dump(out / "crossfit_manifest.json", manifest)
    print(json.dumps({"output_directory": str(out), "files": list(output_names) + ["crossfit_manifest.json"],
                      "checks_passed": len(result["checks"])}, indent=2))


if __name__ == "__main__":
    main()
