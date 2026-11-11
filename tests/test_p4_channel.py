"""analysis/p4_channel.py on synthetic rows only (no block output exists): rows in the format src/exp/channel.py
writes (built by its own plan functions on dev items, outcomes set by hand), checked against hand-computed
estimates; plus the run-integrity checks and one pass over the fake-engine output of channel.main().
Run: python -m pytest tests/test_p4_channel.py -q
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import p4_channel as A  # noqa: E402
from p6_confirmatory import LW  # noqa: E402

from src.exp import channel as C  # noqa: E402

TOK_DIR = ROOT / "data" / "tokenizers" / "qwen3-4b"
P1 = ROOT / "results" / "p1" / "4B"
INIT, VERB = P1 / "p1-initial-4B-d2ebdea" / "initial.jsonl", P1 / "p1-verbal-4B-d2ebdea" / "verbal.jsonl"
pytestmark = pytest.mark.skipif(not TOK_DIR.exists(), reason="pinned tokenizer not downloaded")
N_BOOT = 200
H1B = lambda ys: 0.5 * float(np.array(ys, float) @ LW)   # noqa: E731


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_DIR)


def pool12():
    """12 dev D-nat items with a synthetic a0; a0 correct for even positions."""
    out = []
    for line in open(ROOT / "data" / "processed" / "dnat.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "dev":
            k = len(out)
            out.append({**r, "a0": "So the answer is \\boxed{1}.", "correct0": k % 2 == 0,
                        "verbal": {"p_yes": 0.9 if k % 3 else 0.2, "p_no": 0.1}})
        if len(out) == 12:
            return out


def outcome(row, k):
    """Synthetic correctness by (exp, cond, seed, position k)."""
    e, c = row["exp"], row["cond"]
    if e == "1C":
        return {"F3(0.9):tool": True, "F3(0.9):user": k < 9, "F3(0.9):assistant": k < 6, "F3(0.7):user": k < 6}[c]
    if e == "mask":
        pat = {0: [0, 0, 1, 1], 1: [1, 1, 1, 1], 2: [0, 1, 1, 1]}[row["mask_seed"]]
        return bool(pat[["F3(0.5)", "F3(0.7)", "F3(0.9)", "F3(1.0)"].index(c)])
    s = row["sample"]
    return {"F2": s < 2, "F1": s == 0, "F3(0.5)": False, "F3(0.7)": False, "F3(0.9)": True, "F3(1.0)": True,
            "F5": True, "F3-perm[0]": True, "F3-perm[1]": False, "F3-perm[2]": False}.get(c, s == 1)


def main_outcome(c):
    return c != "F1"


def write_block(tmp, tok, flagged=(11,), commits=("c0ffee1", "c0ffee1"), drop=None, altered_key=True):
    pool = pool12()
    pos = {r["item_id"]: k for k, r in enumerate(pool)}
    by_id = {r["item_id"]: r for r in pool}
    subset = {r["item_id"] for r in pool[:6]}
    self_v, perm = C.verdicts(pool, 0.0)
    runs, greedy_text = [], {}
    for c, items in enumerate((pool[:6], pool[6:])):
        p1 = C.build_stage1(tok, items, subset, self_v, perm)
        out1 = {k: {"text": "critique", "finish_reason": "stop", "out_tokens": 4, "prompt_tokens": 9} for k in p1.prompts}
        p2 = C.build_stage2(tok, p1, out1, by_id)
        rows = [r for r in p1.rows + p2.rows if r.get("stage") != 1]
        for r in rows:
            k = pos[r["item_id"]]
            if r["cond"] == "B0":
                r.update({"correct": r["correct0"], "status": "fixed"})
            else:
                r.update({"correct": outcome(r, k), "status": "ok", "out_tokens": 10, "finish_reason": "stop"})
            if r["exp"] == "1C":
                r["solution_tokens_identical"] = k not in flagged
        if drop is not None and c == 1:
            rows = [r for r in rows if not (r["exp"] == "1C" and r["cond"] == drop)]
        d = tmp / f"exp1c-robust-test-c{c}"
        d.mkdir()
        (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / "chunk.json").write_text(json.dumps({"chunk_items": [r["item_id"] for r in items], "pool": 12, "subset": 6,
                                                  "solution_tokens_identical": {r["item_id"]: pos[r["item_id"]] not in flagged
                                                                                for r in items}}))
        (d / "manifest.json").write_text(json.dumps({"code_commit": commits[c]}))
        gens = [{"prompt_key": k, "sampling": "greedy" if sd is None else sd, "text": f"text {k[:8]}"}
                for k, (_, _, sd) in p1.prompts.items()]
        greedy_text.update({g["prompt_key"]: g["text"] for g in gens if g["sampling"] == "greedy"})
        (d / "generations.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gens))
        runs.append(d)
    m = tmp / "main"
    m.mkdir()
    main_rows = [{"item_id": r["item_id"], "dataset": r["dataset"], "data": "dnat", "correct0": r["correct0"], "exp": "1A",
                  "cond": c, "prompt_key": None, "correct": r["correct0"] if c == "B0" else main_outcome(c)}
                 for r in pool[:6] for c in A.SAMPLED]
    main_rows += [{**main_rows[0], "data": "dinj", "cond": "F1", "correct": True},          # other tables: ignored
                  {**main_rows[0], "cond": "F2:turn1", "stage": 1, "correct": True}]
    (m / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in main_rows))
    keys = sorted(greedy_text)
    main_gens = [{"prompt_key": k, "text": greedy_text[k] + (" changed" if altered_key and j == 0 else "")}
                 for j, k in enumerate(keys)] + [{"prompt_key": "x" * 64, "text": "unrelated"}]
    (m / "generations.jsonl").write_text("".join(json.dumps(g) + "\n" for g in main_gens))
    return runs, m, len(keys)


def run_analysis(monkeypatch, tmp, runs, main, out="p4_test_channel"):
    monkeypatch.setattr(sys, "argv", ["p4_channel.py", "--runs", *map(str, runs), "--main", str(main), "--out", out,
                                      "--n-boot", str(N_BOOT), "--reports-dir", str(tmp / "rep"),
                                      "--json-dir", str(tmp / "js")])
    A.main()
    return json.loads((tmp / "js" / f"{out}.json").read_text()), (tmp / "rep" / f"04_{out}.md").read_text(encoding="utf-8")


def est(v):
    assert v[1] - 1e-12 <= v[0] <= v[2] + 1e-12
    return v[0]


def test_estimates_on_synthetic_rows(tmp_path, monkeypatch, tok):
    runs, main, n_greedy = write_block(tmp_path, tok)
    out, rep = run_analysis(monkeypatch, tmp_path, runs, main)
    assert out["n_1C"] == 12 and out["n_subset"] == 6 and out["code_commit"] == "c0ffee1"
    assert out["status"] == {"1C": "complete", "subset": "complete"} and out["flagged"] == 1
    assert out["greedy_identity"] == {"compared": n_greedy, "identical": n_greedy - 1}
    r = out["all"]["1C"]
    assert est(r["cells"]["F3(0.9):tool"]["acc"]) == 1.0
    assert est(r["cells"]["F3(0.9):user"]["acc"]) == pytest.approx(0.75)
    assert est(r["cells"]["F3(0.9):user"]["W->R"]) == pytest.approx(4 / 6)     # wrong a0 at odd positions 1..11
    assert est(r["cells"]["F3(0.9):user"]["R->W"]) == pytest.approx(1 / 6)
    assert est(r["range at F3(0.9)"]) == pytest.approx(0.5)
    assert est(r["pairwise"]["tool - user"]) == pytest.approx(0.25)
    assert est(r["pairwise"]["assistant - user"]) == pytest.approx(-0.25)
    assert est(r["pairwise"]["tool - assistant"]) == pytest.approx(0.5)
    assert est(r["F3(0.9) - F3(0.7), user"]) == pytest.approx(0.25)
    assert est(r["range - (F3(0.9) - F3(0.7))"]) == pytest.approx(0.25)
    s = out["all"]["sampled"]
    assert est(s["cells"]["F2"]["acc"]) == pytest.approx(2 / 3) and est(s["cells"]["F1"]["acc"]) == pytest.approx(1 / 3)
    assert est(s["cells"]["B0"]["acc"]) == pytest.approx(0.5) and est(s["cells"]["B0"]["greedy"]["acc"]) == pytest.approx(0.5)
    assert est(s["cells"]["F1"]["sampled - greedy"]) == pytest.approx(1 / 3)
    an = s["analogues"]
    assert est(an["H1a analogue (F2 - F1)"]["sampled"]) == pytest.approx(1 / 3)
    assert est(an["H1a analogue (F2 - F1)"]["greedy"]) == pytest.approx(1.0)
    assert est(an["H1a analogue (F2 - F1)"]["sampled - greedy"]) == pytest.approx(-2 / 3)
    h1b = an["H1b analogue (half the item-level OLS slope on p)"]
    assert est(h1b["sampled"]) == pytest.approx(H1B([0, 0, 1, 1])) and est(h1b["greedy"]) == pytest.approx(0.0)
    assert est(an["H1d analogue (F5 - mean F3-perm)"]["sampled"]) == pytest.approx(2 / 3)
    assert est(an["H1d analogue (F5 - mean F3-perm)"]["greedy"]) == pytest.approx(0.0)
    m = out["all"]["mask"]
    assert est(m["seed 0"]["H1b analogue"]) == pytest.approx(H1B([0, 0, 1, 1]))
    assert est(m["seed 1"]["H1b analogue"]) == pytest.approx(0.0)
    assert est(m["seed 2"]["H1b analogue"]) == pytest.approx(H1B([0, 1, 1, 1]))
    assert est(m["seed 2"]["seed 2 - seed 0"]) == pytest.approx(H1B([0, 1, 0, 0]))
    assert est(m["seed 1"]["acc"]["F3(0.5)"]) == 1.0 and est(m["main run (seed 0)"]["H1b analogue"]) == pytest.approx(0.0)
    # sensitivity: position 11 (outside the subset) is flagged
    t = out["sensitivity"]
    assert t["n_1C"] == 11 and t["n_subset"] == 6
    assert est(t["1C"]["range at F3(0.9)"]) == pytest.approx(1 - 6 / 11)
    assert est(t["1C"]["cells"]["F3(0.9):user"]["acc"]) == pytest.approx(9 / 11)
    assert t["sampled"]["analogues"] == s["analogues"]
    for head in ("## Exp 1C (H1c) — all items", "## Sampled robustness — all items", "## Verifier-mask robustness",
                 "sensitivity: without the 1 flagged item(s)", "Range of accuracy across the 3 channels",
                 "## Output status"):
        assert head in rep
    assert "p-value" not in rep and " p = " not in rep


def test_no_flag_means_no_separate_sensitivity(tmp_path, monkeypatch, tok):
    runs, main, _ = write_block(tmp_path, tok, flagged=())
    out, rep = run_analysis(monkeypatch, tmp_path, runs, main)
    assert out["sensitivity"] == {"same_as_all": True} and "no item is flagged" in rep


def test_partial_sample_status(tmp_path, monkeypatch, tok):
    runs, main, _ = write_block(tmp_path, tok)
    out, _ = run_analysis(monkeypatch, tmp_path, runs[:1], main)
    assert out["n_1C"] == 6 and out["status"]["1C"] == "descriptive only (below min_n)"
    assert A.status_of(150, 300) == "pre-registered partial sample" and A.status_of(1455, 1455) == "complete"


def test_integrity_checks(tmp_path, monkeypatch, tok):
    (tmp_path / "a").mkdir()
    runs, main, _ = write_block(tmp_path / "a", tok)
    with pytest.raises(SystemExit, match="twice"):
        run_analysis(monkeypatch, tmp_path / "a", [runs[0], runs[0]], main)
    (tmp_path / "b").mkdir()
    runs, main, _ = write_block(tmp_path / "b", tok, commits=("c0ffee1", "deadbee"))
    with pytest.raises(SystemExit, match="different code commits"):
        run_analysis(monkeypatch, tmp_path / "b", runs, main)
    (tmp_path / "c").mkdir()
    runs, main, _ = write_block(tmp_path / "c", tok, drop="F3(0.7):user")
    with pytest.raises(SystemExit, match="missing cells"):
        run_analysis(monkeypatch, tmp_path / "c", runs, main)


@pytest.mark.skipif(not (INIT.exists() and VERB.exists()), reason="4B P1 files not synced")
def test_reads_channel_main_output(tmp_path, monkeypatch, tok):
    """channel.main() with a stand-in engine (dev, 2 items per chunk) -> this script, with a synthetic main run."""
    class SP:
        def __init__(self, **kw):
            self.kw = kw

    class LLM:
        def __init__(self, **kw):
            pass

        def get_tokenizer(self):
            return tok

        def generate(self, texts, params, use_tqdm=False):
            return [SimpleNamespace(outputs=[SimpleNamespace(text=" so \\boxed{0}", finish_reason="stop", token_ids=[1])],
                                    prompt_token_ids=[0]) for _ in texts]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=SP, __version__="fake"))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    runs = []

    def fake_run_dir(exp, model, kind):
        d = tmp_path / kind
        d.mkdir()
        runs.append(d)
        return f"{exp}-{kind}", d

    monkeypatch.setattr(C, "run_dir", fake_run_dir)
    monkeypatch.setattr(C, "register", lambda *a, **k: None)
    monkeypatch.setattr(C, "write_manifest", lambda d, run_id, model, tk, cfg, backend, prompts:
                        (d / "manifest.json").write_text(json.dumps({"code_commit": "fake"})))
    monkeypatch.setattr(sys, "argv", ["prog", "--model", "4B", "--split", "dev", "--initial", str(INIT), "--verbal",
                                      str(VERB), "--self-threshold-s", "16", "--chunk-index", "0", "--through", "1",
                                      "--limit", "2", "--tokenizer", str(TOK_DIR)])
    C.main()
    sub = [json.loads((d / "chunk.json").read_text())["chunk_items"] for d in runs]
    m = tmp_path / "main"
    m.mkdir()
    rows = [json.loads(l) for d in runs for l in open(d / "rows.jsonl", encoding="utf-8")]
    c0 = {r["item_id"]: r["correct0"] for r in rows}
    (m / "rows.jsonl").write_text("".join(json.dumps({"item_id": i, "data": "dnat", "exp": "1A", "cond": c,
                                                      "correct0": c0[i], "correct": False}) + "\n"
                                          for ids in sub for i in ids for c in A.SAMPLED))
    out, rep = run_analysis(monkeypatch, tmp_path, runs, m)
    assert out["n_1C"] == out["n_subset"] == 4 and out["greedy_identity"] is None
    assert out["status"] == {"1C": "descriptive only (below min_n)", "subset": "descriptive only (below min_n)"}
    assert "## Exp 1C (H1c) — all items, n = 4" in rep
