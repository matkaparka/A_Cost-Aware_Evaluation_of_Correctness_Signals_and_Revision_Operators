"""Block 4B-exp1c-robust (src/exp/channel.py, src/gen/render_channel.py): Exp 1C channel renderings, the sampled and
verifier-mask robustness plans, request structure and scoring. Pinned Qwen3 tokenizer; dev items only (the P1 file
is read and filtered to split == "dev"). No generation.
Run: python -m pytest tests/test_channel.py -q
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.verifiers import f3_verdict, permute_self_verdicts, uniform
from src.exp import channel as C
from src.exp.runner import P_LEVELS
from src.gen import render as R
from src.gen import render_channel as RC
from src.runlib import sha256_text

ROOT = Path(__file__).resolve().parents[1]
TOK_DIR = ROOT / "data" / "tokenizers" / "qwen3-4b"
P1 = ROOT / "results" / "p1" / "4B"
INIT, VERB = P1 / "p1-initial-4B-d2ebdea" / "initial.jsonl", P1 / "p1-verbal-4B-d2ebdea" / "verbal.jsonl"
NO_THINK_GEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
pytestmark = pytest.mark.skipif(not TOK_DIR.exists(), reason="pinned tokenizer not downloaded")
needs_p1 = pytest.mark.skipif(not (INIT.exists() and VERB.exists()), reason="4B P1 files not synced")
T_STAR = 15.999999163764045   # 4B t* (prereg §6; analysis/out/selfverdict.json)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_DIR)


@pytest.fixture(scope="module")
def pool():
    return C.load_pool(INIT, VERB, "dev") if INIT.exists() and VERB.exists() else []


def dev_solutions(pool, n=40):
    """Dev reviewed solutions: injected D-inj solutions and 4B D-nat dev a0."""
    out = []
    for line in open(ROOT / "data" / "processed" / "dinj.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "dev":
            out.append((r["question"], r["solution_with_answer"]))
        if len(out) >= n // 2:
            break
    return out + [(r["question"], r["a0"]) for r in pool[:n // 2]]


# ---------- renderings ----------

def test_wording_hashes():
    assert C.main_wording_sha256() == "229f9167e91e8a01ef18fa4fe6d37b79c51f69243b9a9a66988dadbc045583cf"   # prereg §1
    assert sha256_text(RC.channel_wording_json()) == "9fd6b90dba939fd4f2225fe6227cd2318cd9febbadb468596eff0c309ceb1fbc"
    assert RC.CHANNELS == ("tool", "user", "assistant") and RC.ASSISTANT_SEP == " \n\n"


def test_user_channel_is_the_f3_prompt(tok, pool):
    for q, sol in dev_solutions(pool, 10):
        for v in (True, False):
            r = RC.channel_prompt(tok, q, sol, R.VERDICT[v], "user")
            assert r.text == R.review_prompt(tok, q, sol, feedback=R.VERDICT[v]).text and r.prefill == ""


def test_tool_channel_uses_the_template_tool_format(tok, pool):
    for q, sol in dev_solutions(pool, 10):
        for v in (True, False):
            r = RC.channel_prompt(tok, q, sol, R.VERDICT[v], "tool")
            msgs = [{"role": "user", "content": f"{R.PROMPT.format(q=q)}\n\n{R.LABELS['self']}"},
                    {"role": "assistant", "content": sol}, {"role": "tool", "content": R.VERDICT[v]},
                    {"role": "user", "content": R.INSTRUCTION}]
            ref = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert r.text == ref
            tail = (sol + "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + R.VERDICT[v] + "\n</tool_response><|im_end|>\n"
                    "<|im_start|>user\n" + R.INSTRUCTION + "<|im_end|>\n" + NO_THINK_GEN)
            assert r.text.endswith(tail)
            assert "<tools>" not in r.text and "<tool_call>" not in r.text and "system" not in r.text


def test_assistant_channel_puts_the_verdict_in_the_a0_turn(tok, pool):
    for q, sol in dev_solutions(pool, 10):
        for v in (True, False):
            r = RC.channel_prompt(tok, q, sol, R.VERDICT[v], "assistant")
            tail = ("<|im_start|>assistant\n" + sol + " \n\n" + R.VERDICT[v] + "<|im_end|>\n<|im_start|>user\n"
                    + R.INSTRUCTION + "<|im_end|>\n" + NO_THINK_GEN)
            assert r.text.endswith(tail)
            assert r.text.count("<|im_start|>") == 4   # user, assistant (a0 + verdict), user, generation


def test_spans_and_order(tok, pool):
    for q, sol in dev_solutions(pool, 10):
        for ch in RC.CHANNELS:
            r = RC.channel_prompt(tok, q, sol, R.VERDICT[False], ch)
            s = r.spans
            assert s["question"][1] <= s["label"][0] < s["label"][1] <= s["solution"][0]   # label precedes solution
            assert s["solution"][1] <= s["feedback"][0] < s["feedback"][1] <= s["instruction"][0]
            assert r.text[slice(*s["feedback"])] == R.VERDICT[False]   # verdict text identical in every channel
            assert r.text[slice(*s["solution"])] == sol


def test_solution_tokens_identical_across_channels(tok, pool):
    items = dev_solutions(pool)
    assert len(items) >= 20
    for q, sol in items:
        for v in (True, False):
            seqs = set()
            for ch in RC.CHANNELS:
                ids, spans = R.token_spans(tok, RC.channel_prompt(tok, q, sol, R.VERDICT[v], ch))
                a, b = spans["solution"]
                seqs.add(tuple(ids[a:b]))
            assert len(seqs) == 1


@needs_p1
def test_solution_tokens_identical_every_dev_item(tok, pool):
    assert len(pool) == 364
    assert all(C.solution_tokens_identical(tok, r) for r in pool)


def test_boundary_flag_catches_trailing_whitespace(tok):
    r = {"item_id": "x", "correct0": True, "question": "What is 2+3?", "a0": "The sum is 5. "}
    assert not C.solution_tokens_identical(tok, r)          # a0 ending in a space merges with " \n\n"
    assert C.solution_tokens_identical(tok, {**r, "a0": "The sum is \\boxed{5}."})


def test_channel_lengths(tok, pool):
    for q, sol in dev_solutions(pool, 20):
        for v in (True, False):
            n = {ch: len(tok(RC.channel_prompt(tok, q, sol, R.VERDICT[v], ch).text, add_special_tokens=False)["input_ids"])
                 for ch in RC.CHANNELS}
            assert n["assistant"] - n["user"] == 1 and n["tool"] - n["user"] == 8


# ---------- plan ----------

@needs_p1
def test_order_subset_and_chunks(pool):
    split = json.loads((ROOT / "data" / "processed" / "split.json").read_text())["order"]["dev"]
    rank = {c: k for k, c in enumerate(split)}
    keys = [(rank[r["canonical_id"]], r["item_id"]) for r in pool]
    assert keys == sorted(keys) and all(r["split"] == "dev" for r in pool)
    chunks = [C.chunk_items(pool, c, 100) for c in range(4)]
    assert [len(c) for c in chunks] == [100, 100, 100, 64]
    assert [r["item_id"] for c in chunks for r in c] == [r["item_id"] for r in pool]
    subset = {r["item_id"] for r in pool[:300]}
    assert subset == {r["item_id"] for c in chunks[:3] for r in c}


@needs_p1
def test_stage1_structure(tok, pool):
    self_v, perm = C.verdicts(pool, T_STAR)
    subset = {r["item_id"] for r in pool[:300]}
    items = pool[298:302]                     # two in the subset, two outside
    plan = C.build_stage1(tok, items, subset, self_v, perm)
    by_item = {}
    for row in plan.rows:
        by_item.setdefault(row["item_id"], []).append(row)
    for r in items:
        rows = by_item[r["item_id"]]
        one_c = [x for x in rows if x["exp"] == "1C"]
        assert [x["cond"] for x in one_c] == [f"F3(0.9):{ch}" for ch in RC.CHANNELS] + ["F3(0.7):user"]
        assert {x["verdict"] for x in one_c[:3]} == {f3_verdict(r["item_id"], r["correct0"], 0.9)}
        ref = one_c[3]
        assert ref["channel"] == "user" and ref["verdict"] == f3_verdict(r["item_id"], r["correct0"], 0.7)
        assert (ref["prompt_key"] == one_c[1]["prompt_key"]) == (ref["verdict"] == one_c[1]["verdict"])
        assert plan.prompts[ref["prompt_key"]][0].text == R.review_prompt(tok, r["question"], r["a0"],
                                                                          feedback=R.VERDICT[ref["verdict"]]).text
        if r["item_id"] not in subset:
            assert len(rows) == 4
            continue
        assert len(rows) == 4 + 12 + 3 * 14       # 1C; mask 3 seeds x 4 p; sampled 3 x (B0 + 13 requests)
        user = one_c[1]
        m07 = next(x for x in rows if x["exp"] == "mask" and x["mask_seed"] == 0 and x["cond"] == "F3(0.7)")
        assert m07["prompt_key"] == ref["prompt_key"]                              # shared generation
        text = plan.prompts[user["prompt_key"]][0].text
        assert user["prompt_key"] == sha256_text(text + "\x00" + "" + "\x002048")   # the runner's greedy key
        m09 = next(x for x in rows if x["exp"] == "mask" and x["mask_seed"] == 0 and x["cond"] == "F3(0.9)")
        assert m09["prompt_key"] == user["prompt_key"]                             # shared generation
        sampled = [x for x in rows if x["exp"] == "sampled"]
        for s in range(C.N_SAMPLE_SEEDS):
            xs = [x for x in sampled if x["sample"] == s]
            assert {x["seed"] for x in xs} == {C.sample_seed(r["item_id"], s)}       # paired across conditions
            assert {x["cond"] for x in xs} == {"B0", "B1", "F0", "F1", "F2:turn1", "F2-scratch:turn1", "F3(0.5)",
                                               "F3(0.7)", "F3(0.9)", "F3(1.0)", "F5", "F3-perm[0]", "F3-perm[1]",
                                               "F3-perm[2]"}
            ver = [x for x in xs if "verdict" in x]
            for x in ver:
                assert plan.prompts[x["prompt_key"]][2] == x["seed"]
            for a in ver:
                for b in ver:
                    assert (a["prompt_key"] == b["prompt_key"]) == (a["verdict"] == b["verdict"])
            f = {x["cond"]: x for x in xs}
            assert [f[f"F3-perm[{k}]"]["verdict"] for k in range(3)] == perm[r["item_id"]]
            assert f["F5"]["verdict"] == self_v[r["item_id"]]
            assert f["F1"]["prompt_key"] != f["F0"]["prompt_key"]
            assert plan.prompts[f["F1"]["prompt_key"]][0].prefill == R.TRIGGER_PREFILL
            assert plan.prompts[f["F2:turn1"]["prompt_key"]][1] == 1024 and plan.prompts[f["F0"]["prompt_key"]][1] == 2048
        assert len({C.sample_seed(r["item_id"], s) for s in range(3)}) == 3
    # sampled keys never collide with greedy keys of the same prompt
    greedy = {k for k, (_, _, sd) in plan.prompts.items() if sd is None}
    assert all(row["prompt_key"] not in greedy for row in plan.rows if row["exp"] == "sampled" and row["prompt_key"])


@needs_p1
def test_mask_rows(tok, pool):
    self_v, perm = C.verdicts(pool, T_STAR)
    subset = {r["item_id"] for r in pool[:300]}
    plan = C.build_stage1(tok, pool[:100], subset, self_v, perm)
    mask = [x for x in plan.rows if x["exp"] == "mask"]
    assert len(mask) == 100 * 3 * 4
    for x in mask:
        truthful = uniform(x["item_id"], x["mask_seed"]) < float(x["cond"][3:-1])
        assert x["verdict"] == (x["correct0"] if truthful else not x["correct0"])
        if x["cond"] == "F3(1.0)":
            assert x["verdict"] == x["correct0"]
    # nested across p within each seed
    for k in C.MASK_SEEDS:
        for r in pool[:100]:
            tr = [f3_verdict(r["item_id"], r["correct0"], p, seed=k) == r["correct0"] for p in P_LEVELS]
            assert tr == sorted(tr)
    assert any(f3_verdict(r["item_id"], True, 0.5, 1) != f3_verdict(r["item_id"], True, 0.5, 0) for r in pool[:100])
    greedy_keys = {x["prompt_key"] for x in plan.rows if x["exp"] in ("mask", "1C") and x.get("channel") in (None, "user")}
    assert len(greedy_keys) <= 2 * 100        # at most one prompt per verdict value per item


@needs_p1
def test_perm_over_the_whole_pool(pool):
    self_v, perm = C.verdicts(pool, T_STAR)
    ref = permute_self_verdicts([{"item_id": r["item_id"], "dataset": r["dataset"], "initially_correct": r["correct0"],
                                  "self_says_correct": self_v[r["item_id"]]} for r in reversed(pool)])
    assert perm == ref and len(perm) == len(pool)


@needs_p1
def test_gold_isolation(tok, pool):
    """Counterfactual label (design §5.8): flipping the private label changes only verdict prompts."""
    self_v, perm = C.verdicts(pool, T_STAR)
    r = pool[0]
    subset = {r["item_id"]}
    a = C.build_stage1(tok, [r], subset, self_v, perm)
    b = C.build_stage1(tok, [{**r, "correct0": not r["correct0"], "gold": "counterfactual"}], subset, self_v, perm)
    for x, y in zip(a.rows, b.rows):
        assert (x["exp"], x["cond"], x.get("channel"), x.get("mask_seed"), x.get("sample")) == \
               (y["exp"], y["cond"], y.get("channel"), y.get("mask_seed"), y.get("sample"))
        if "verdict" not in x and x["prompt_key"] is not None:
            assert x["prompt_key"] == y["prompt_key"]
    for plan in (a, b):
        for rendered, _, _ in plan.prompts.values():
            assert "counterfactual" not in rendered.text


@needs_p1
def test_stage2_and_scoring(tok, pool):
    self_v, perm = C.verdicts(pool, T_STAR)
    items = pool[:2]
    subset = {r["item_id"] for r in items}
    plan1 = C.build_stage1(tok, items, subset, self_v, perm)
    by_id = {r["item_id"]: r for r in pool}
    out1 = {k: {"text": f"critique {k[:6]} \\boxed{{{by_id[next(x['item_id'] for x in plan1.rows if x['prompt_key'] == k)]['gold']}}}",
                "finish_reason": "stop", "out_tokens": 5, "prompt_tokens": 100} for k in plan1.prompts}
    plan2 = C.build_stage2(tok, plan1, out1, by_id)
    assert len(plan2.rows) == len(plan2.prompts) == 2 * 2 * C.N_SAMPLE_SEEDS
    t1 = {(x["item_id"], x["sample"], x["cond"].split(":")[0]): x for x in plan1.rows if x.get("stage") == 1}
    for row in plan2.rows:
        first = t1[(row["item_id"], row["sample"], row["cond"])]
        rendered, mt, sd = plan2.prompts[row["prompt_key"]]
        assert sd == first["seed"] == row["seed"] and mt == 2048
        turn = R.CRITIQUE_TURN if row["cond"] == "F2" else R.SCRATCH_TURN
        assert turn + "<|im_end|>\n<|im_start|>assistant\n" + out1[first["prompt_key"]]["text"] in rendered.text
        assert rendered.text.endswith(R.INSTRUCTION + "<|im_end|>\n" + NO_THINK_GEN)
    out2 = {k: {"text": "So \\boxed{-1}", "finish_reason": "length", "out_tokens": 9, "prompt_tokens": 200} for k in plan2.prompts}
    golds = {i: by_id[i]["gold"] for i in by_id}
    C.score_rows(plan1, out1, golds)
    C.score_rows(plan2, out2, golds)
    for row in plan1.rows:
        if row.get("stage") == 1:
            assert "correct" not in row
        elif row["cond"] == "B0":
            assert row["status"] == "fixed" and row["correct"] == row["correct0"]
        else:
            assert row["correct"] is True and row["status"] == "ok"
    assert all(row["correct"] is False for row in plan2.rows)     # truncated = wrong


def test_generate_request_parameters(tok):
    class SP:
        def __init__(self, **kw):
            self.kw = kw

    class LLM:
        def generate(self, texts, params, use_tqdm=False):
            self.texts, self.params = texts, params
            return [SimpleNamespace(outputs=[SimpleNamespace(text=" out", finish_reason="stop", token_ids=[1, 2])],
                                    prompt_token_ids=[0] * 7) for _ in texts]

    plan = C.Plan()
    g = plan.add({}, R.review_prompt(tok, "q", "a"), 2048)
    s = plan.add({}, R.review_prompt(tok, "q", "a", prefill=R.TRIGGER_PREFILL), 2048, seed=123)
    llm = LLM()
    out = C.generate(llm, plan, SP)
    kw = {k: p.kw for k, p in zip(plan.prompts, llm.params)}
    assert kw[g] == {"temperature": 0.0, "max_tokens": 2048, "seed": 0}
    assert kw[s] == {"temperature": 0.7, "top_p": 0.8, "max_tokens": 2048, "seed": 123}
    assert llm.texts[1].endswith(NO_THINK_GEN + R.TRIGGER_PREFILL)
    assert out[s]["text"] == "Wait, out" and out[g]["text"] == " out" and out[g]["prompt_tokens"] == 7


# ---------- the run-host path, with a stand-in engine ----------

@needs_p1
def test_main_end_to_end_with_fake_engine(tmp_path, monkeypatch, tok):
    """main() on dev with a fake vLLM: two chunks, one engine, every output file; then the deadline stop."""
    import sys
    import time as _time

    class SP:
        def __init__(self, **kw):
            self.kw = kw

        def __repr__(self):
            return f"SP({self.kw})"

    class LLM:
        loads = 0

        def __init__(self, **kw):
            LLM.loads += 1

        def get_tokenizer(self):
            return tok

        def generate(self, texts, params, use_tqdm=False):
            return [SimpleNamespace(outputs=[SimpleNamespace(text=" so \\boxed{0}", finish_reason="stop", token_ids=[1, 2, 3])],
                                    prompt_token_ids=[0] * 11) for _ in texts]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=SP, __version__="fake"))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    runs, events = {}, []

    def fake_run_dir(exp, model, kind):
        d = tmp_path / kind
        d.mkdir()
        runs[kind] = d
        return f"{exp}-{kind}-{model}-test", d

    monkeypatch.setattr(C, "run_dir", fake_run_dir)
    monkeypatch.setattr(C, "register", lambda run_id, status, **kw: events.append((run_id, status, kw.get("block"))))
    monkeypatch.setattr(C, "write_manifest", lambda d, run_id, model, tk, cfg, backend, prompts:
                        (d / "config.json").write_text(json.dumps(cfg)))
    base = ["prog", "--model", "4B", "--split", "dev", "--initial", str(INIT), "--verbal", str(VERB),
            "--self-threshold-s", str(T_STAR), "--chunk-index", "0", "--through", "1", "--limit", "2",
            "--tokenizer", str(TOK_DIR)]
    monkeypatch.setattr(sys, "argv", base)
    C.main()
    assert LLM.loads == 1 and sorted(runs) == ["exp1c-robust-dev-c0", "exp1c-robust-dev-c1"]
    assert [e[1] for e in events] == ["running", "done", "running", "done"] and events[0][2] == "4B-exp1c-robust"
    for c, d in enumerate(runs[k] for k in sorted(runs)):
        rows = [json.loads(l) for l in open(d / "rows.jsonl", encoding="utf-8")]
        gens = [json.loads(l) for l in open(d / "generations.jsonl", encoding="utf-8")]
        met = json.loads((d / "metrics.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        chunk = json.loads((d / "chunk.json").read_text())
        assert len(rows) == 2 * (4 + 12 + 3 * 14) and not any(r.get("stage") == 1 for r in rows)
        assert len(gens) == met["generations"] == len({g["prompt_key"] for g in gens})
        assert {r["prompt_key"] for r in rows if r["prompt_key"]} <= {g["prompt_key"] for g in gens}
        assert all(r["status"] in ("ok", "fixed") for r in rows) and met["solution_tokens_identical"] == "2/2"
        assert cfg["chunk_index"] == c and "through" not in cfg and "deadline_epoch" not in cfg
        assert cfg["wording_sha256"] == C.main_wording_sha256() and cfg["sampling"] == {"temperature": 0.7, "top_p": 0.8}
        assert len(chunk["chunk_items"]) == 2 and chunk["subset_items"] == chunk["chunk_items"]
        assert {g["sampling"] for g in gens} - {"greedy"} == {C.sample_seed(i, s) for i in chunk["chunk_items"]
                                                              for s in range(3)}
    # a deadline too close for the second chunk: only chunk 0 runs
    runs.clear()
    events.clear()
    for d in tmp_path.iterdir():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    monkeypatch.setattr(sys, "argv", base + ["--deadline-epoch", str(_time.time() + 30)])
    C.main()
    assert sorted(runs) == ["exp1c-robust-dev-c0"]
