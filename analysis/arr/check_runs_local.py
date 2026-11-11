"""Control-host checks of the ARR-round run code before any GPU session. CPU only: tokenizers, no model, no GPU.

1. B4 selection parity: analysis/arr/b4_select.py on the stored Qwen3-4B math samples and on the MMLU-Pro sample
   letters selects exactly the A1 HYBRID-margin 20 % members.
2. Prompt identity: src/exp/arr_runs.py `ops` with the main wording rebuilds the stored test prompt keys of F0, F2 and
   F2-scratch (both turns; the second from the stored first-turn text) for every math model (first --n items).
3. Dry runs of every planned run (stage x model x split): every request fits its max_model_len.
Usage: CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -B analysis/arr/check_runs_local.py [--n 400]
-> analysis/out/arr/check_runs_local.json, analysis/out/arr/b4_local/*.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import J1, MODELS_MATH, OUT, REPO, Equivalence, dump, plan_core_hash, rows, rows_gz, run_dir, sha  # noqa: E402
import a1a2_replays as A  # noqa: E402
import b4_select as B4  # noqa: E402

sys.path.insert(0, str(REPO))
from src.exp import arr_runs as AR  # noqa: E402
from src.runlib import MODELS  # noqa: E402

YAML_KEY = {"1.7B": "qwen3-1.7b", "4B": "qwen3-4b", "8B": "qwen3-8b", "T8B": "tulu3-8b"}
MC_INITIAL = {m: f"results/v2/fu/{m}/fu-mc-initial-test-{m}-9221b24/initial.jsonl" for m in ("4B", "T8B")}


def tok_path(model):
    """The cached snapshot directory of the pinned revision (tokenizer files only are needed; weights may be absent)."""
    from huggingface_hub import constants
    pin = yaml.safe_load(open(REPO / "configs/models.yaml", encoding="utf-8"))["models"][YAML_KEY[model]]["revision"]
    d = Path(constants.HF_HUB_CACHE) / ("models--" + MODELS[model].replace("/", "--")) / "snapshots" / pin
    assert (d / "tokenizer_config.json").exists() and (d / "tokenizer.json").exists(), ("tokenizer files missing", d)
    return str(d)


def math_initial_files(table, model):
    runs = sorted({r["initial_run"] for r in table[model]["F2"].values()})
    return [str((run_dir(r) / "initial.jsonl").relative_to(REPO)).replace("\\", "/") for r in runs]


def check_b4(table, labels):
    out = {}
    math = A.load_math(table, labels)
    eq = Equivalence()
    try:
        P = math["4B"]
        A.add_math_samples(P, eq)
        a1 = set(np.array(P["ids"])[A.select(P["sig"]["margin"], P["tie"], 0.2)])
        choices = B4.math_choices(REPO / "results/fu/4B/fu-h2a-consistency-test-4B-aac4cc6/samples.jsonl", eq)
    finally:
        eq.close()
    sel = B4.select(choices, 0.2)
    assert set(sel) == a1, ("B4 math 4B selection differs from A1", len(set(sel) ^ a1))
    write_select(OUT / "b4_local/math_4B.jsonl", sel, choices)
    out["math 4B"] = len(sel)
    for m in ("4B", "T8B"):
        P = A.load_mc(m)
        a1 = set(np.array(P["ids"])[A.select(P["sig"]["margin"], P["tie"], 0.2)])
        runs = json.loads((J1 / "artifacts/mc/summary.json").read_text(encoding="utf-8"))["models"][m]["runs"]
        choices = B4.mc_choices(REPO / runs["consist"] / "consistency.jsonl")
        sel = B4.select(choices, 0.2)
        assert set(sel) == a1, ("B4 MMLU-Pro selection differs from A1", m)
        write_select(OUT / f"b4_local/mc_{m}.jsonl", sel, choices)
        out[f"MMLU-Pro {m}"] = len(sel)
    return out


def write_select(path, ids, choices):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for i in ids:
            f.write(json.dumps({"item_id": i, **choices[i]}, ensure_ascii=False) + "\n")


def placeholder_select(path, initial_files):
    """Length check only: every item, with a0's own prediction shown five times."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in AR.initial_rows([REPO / p for p in initial_files], "test"):
            f.write(json.dumps({"item_id": r["item_id"], "answers": [r.get("pred")] * 5}, ensure_ascii=False) + "\n")


def check_keys(table, n):
    from transformers import AutoTokenizer
    q = AR.math_items()
    out = {}
    for model in MODELS_MATH:
        tok = AutoTokenizer.from_pretrained(tok_path(model))
        t = table[model]
        ids = sorted(t["F2"])[:n]
        init = {r["item_id"]: r for p in math_initial_files(table, model) for r in rows(REPO / p) if r["split"] == "test"}
        init_rows = [init[i] for i in ids]
        gens = {}
        for run in sorted({t[c][i]["source_run"] for c in ("F0", "F2", "F2-scratch") for i in ids}):
            for g in rows(run_dir(run) / "generations.jsonl"):
                gens[g["prompt_key"]] = g
        plan1 = AR.build_ops_stage1(tok, init_rows, q, ["F0", "F2", "F2-scratch"], "main")
        out1, checked = {}, 0
        for row in plan1.rows:
            i, cond = row["item_id"], row["cond"]
            if cond == "F0":
                assert row["prompt_key"] == t["F0"][i]["final_prompt_key"], (model, i, "F0")
            else:
                assert row["prompt_key"] in gens, (model, i, cond, "first-turn key not among the stored generations")
                out1[row["prompt_key"]] = gens[row["prompt_key"]]
            checked += 1
        plan2 = AR.build_ops_stage2(tok, plan1, out1, init_rows, q, "main")
        for row in plan2.rows:
            assert row["prompt_key"] == t[row["cond"]][row["item_id"]]["final_prompt_key"], (model, row["item_id"], row["cond"])
            checked += 1
        out[model] = {"items": len(ids), "keys_checked": checked}
        del gens
    return out


def dry(stage, model, split, initial, task="math", extra=()):
    cmd = [sys.executable, "-B", "-m", "src.exp.arr_runs", stage, "--task", task, "--model", model, "--split", split,
           "--dry-run", "--tokenizer", tok_path(model)] + (["--allow-test"] if split == "test" else [])
    for f in initial:
        cmd += ["--initial", f]
    cmd += list(extra)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, (stage, model, split, extra, res.stderr[-3000:])
    return json.loads(res.stdout[res.stdout.index("{"):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA_VISIBLE_DEVICES= (no GPU on the control host)"
    table = {}
    for r in rows(J1 / "artifacts/math/item_evaluations.jsonl"):
        table.setdefault(r["model"], {}).setdefault(r["condition"], {})[r["item_id"]] = r
    labels = list(rows_gz(OUT / "math_common_labels.jsonl.gz"))
    result = {"plan_core_sha256": plan_core_hash(), "arr_runs_sha256": sha(REPO / "src/exp/arr_runs.py")}
    result["b4_selection_parity"] = check_b4(table, labels)
    print("B4 selection parity ok", result["b4_selection_parity"], flush=True)
    result["prompt_key_identity"] = check_keys(table, a.n)
    print("prompt keys ok", result["prompt_key_identity"], flush=True)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="arr_check_")   # placeholder selections (length checks only) stay out of the outputs
    runs = []
    for model in MODELS_MATH:
        ini = math_initial_files(table, model)
        for split in ("test", "dev"):
            runs.append(("samples", model, split, ini, "math", ("--k", "5")))
            runs.append(("samples", model, split, ini, "math", ("--k", "1", "--stream", "a0-long", "--max-tokens", "8192")))
            runs.append(("continue", model, split, ini, "math", ()))
        if model != "1.7B":
            for w in ("p1", "p2"):
                runs.append(("ops", model, "test", ini, "math", ("--wording", w, "--conds", "F2,F2-scratch")))
        if model in ("8B", "T8B"):
            runs.append(("ops", model, "dev", ini, "math", ("--conds", "F0,F2,F2-scratch")))
        sel = OUT / "b4_local/math_4B.jsonl" if model == "4B" else Path(tmp) / f"placeholder_math_{model}.jsonl"
        if model != "4B":
            placeholder_select(sel, ini)
        runs.append(("samples-critique", model, "test", ini, "math", ("--select", str(sel))))
        if model in ("4B", "8B"):
            runs.append(("think", model, "test", ini, "math", ()))
    for m in ("4B", "T8B"):
        runs.append(("samples-critique", m, "test", [MC_INITIAL[m]], "mc", ("--select", str(OUT / f"b4_local/mc_{m}.jsonl"))))
    runs.append(("think", "4B", "test", [MC_INITIAL["4B"]], "mc", ()))
    result["dry_runs"] = []
    for stage, model, split, ini, task, extra in runs:
        r = dry(stage, model, split, ini, task, extra)
        r["extra"] = list(extra)
        result["dry_runs"].append(r)
        print(f"dry {stage} {task} {model} {split} {' '.join(extra)}: requests {r['requests']}, worst {r['max_prompt_plus_budget']} / {r['max_model_len']}", flush=True)
    assert all(r["fits"] for r in result["dry_runs"])
    dump(OUT / "check_runs_local.json", result)
    print("all checks passed:", len(result["dry_runs"]), "dry runs")


if __name__ == "__main__":
    main()
