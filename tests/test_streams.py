"""Streams (src/ttt/streams.py; H4c): item rules, the context-log rule, the increment rule (per-step norm tau, A fixed,
anomalies), the zero-step invariants, and resume = uninterrupted run. The model-level tests use a tiny random Qwen3
on CPU with the pinned tokenizer and a stub generator in place of vLLM; dev items only.
Run: python -m pytest tests/test_streams.py -q
"""
import hashlib
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.gen import render as R  # noqa: E402
from src.ttt import streams as ST  # noqa: E402
from src.ttt import update as U  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOK_DIR = ROOT / "data" / "tokenizers" / "qwen3-4b"
needs_tok = pytest.mark.skipif(not TOK_DIR.exists(), reason="pinned tokenizer not downloaded")
TAU = 0.5


# ---- items ---------------------------------------------------------------------------------------------------

def test_item_rules():
    sp = json.loads((ROOT / "data" / "processed" / "split.json").read_text())
    kl, spill = set(sp["kl_dev"]), set(sp["spillover_test"])
    S = ST.select_items("test", 200, 100, [0, 1, 2])
    meta = S["meta"]
    canon = lambda ids: {meta[i]["canonical_id"] for i in ids}
    assert len(S["pool"]) == 200 == len(set(S["pool"])) and len(S["eval"]) == 100 == len(set(S["eval"]))
    assert {meta[i]["split"] for i in S["pool"]} == {"test"} and {meta[i]["split"] for i in S["eval"]} == {"dev"}
    assert not canon(S["pool"]) & spill and not canon(S["eval"]) & kl and not set(S["pool"]) & set(S["eval"])
    # The Exp 5 per-item pool rule (src/ttt/test.py build): D-nat test items by seeded order rank, spillover excluded.
    rank = {c: k for k, c in enumerate(sp["order"]["test"])}
    exp5 = sorted((i for i, r in meta.items() if r["split"] == "test" and r["canonical_id"] not in spill),
                  key=lambda i: rank[meta[i]["canonical_id"]])
    assert S["pool"] == exp5[:200]
    for s, o in S["orders"].items():
        assert sorted(o) == sorted(S["pool"])
    assert len({tuple(o) for o in S["orders"].values()}) == 3
    assert ST.select_items("test", 200, 100, [0, 1, 2])["orders"] == S["orders"]
    D = ST.select_items("dev", 200, 100, [0])
    assert D["eval"] == S["eval"] and not set(D["pool"]) & set(D["eval"]) and not canon(D["pool"]) & kl
    small = ST.select_items("dev", 4, 2, [0])        # smoke runs never touch the full eval set
    assert small["eval"] == S["eval"][:2] and not set(small["pool"]) & set(S["eval"])


# ---- tokenizer-level rules -------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_DIR)


def entry(k, n=40):
    return {"q": f"Problem {k}: " + "how many apples " * n, "a0": f"Answer {k} " * n + "\\boxed{3}",
            "verdict": "correct" if k % 2 else "incorrect", "item_id": f"x{k}", "step": k}


@needs_tok
def test_memory_drops_oldest_first(tok):
    es = [entry(k) for k in range(6)]
    one = ST.n_tokens(tok, ST.memory_block(es[:1]))
    budget = int(2.5 * one)
    kept, dropped = ST.fit_memory(tok, es, budget)
    assert [e["item_id"] for e in kept] == ["x4", "x5"] and dropped == 4
    assert ST.n_tokens(tok, ST.memory_block(kept)) <= budget
    assert ST.n_tokens(tok, ST.memory_block(es[3:])) > budget
    assert ST.fit_memory(tok, es[:1], one - 1) == ([], 1)          # an entry larger than the budget is dropped too
    assert ST.fit_memory(tok, es[:2], 10 ** 6) == (es[:2], 0)


@needs_tok
def test_empty_log_is_the_q_only_prompt(tok):
    q = "What is 2 + 3?"
    assert ST.user_content(q, []) == R.PROMPT.format(q=q)
    assert R.render_messages(tok, [{"role": "user", "content": ST.user_content(q, [])}]) == R.fresh_prompt(tok, q).text
    full = ST.user_content(q, [entry(0)])
    assert full.startswith(ST.MEMORY_HEADER) and full.endswith(R.PROMPT.format(q=q))


def test_answer_classes():
    assert ST.n_classes(["5", "5.0", None, None]) == 2
    assert ST.n_classes(["1/2", "0.5", "3", "4"]) == 3
    assert ST.n_classes(["7", "7", "7", "7"]) == 1


# ---- tiny model ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=True)
    base = Qwen3ForCausalLM(cfg).eval()
    pm, layers, _ = U.attach(base)
    pm.eval()
    for m in layers:
        m.lora_A["default"].weight.requires_grad_(False)
    return pm, layers


def seq(tok, q="What is 2 + 3?", a0="2 + 3 = 5, so the answer is \\boxed{5}."):
    ids = tok(a0, add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    return U.training_sequence(tok, q, ids)


def logits(pm, ids):
    with torch.no_grad():
        return pm(input_ids=torch.tensor([ids])).logits


def bstate(layers):
    return [m.lora_B["default"].weight.detach().clone() for m in layers]


def astate(layers):
    return [m.lora_A["default"].weight.detach().clone() for m in layers]


@needs_tok
def test_zero_step_adapter_equals_base(tok, tiny):
    pm, layers = tiny
    U.reset(layers, 123)
    ids, _ = seq(tok)
    out = logits(pm, ids)
    with pm.disable_adapter():
        base = logits(pm, ids)
    assert torch.equal(out, base)


@needs_tok
def test_increment_norm_is_tau_and_a_is_fixed(tok, tiny):
    pm, layers = tiny
    U.reset(layers, 7)
    a0 = astate(layers)
    ids, mask = seq(tok)
    assert mask is not None
    for k, verdict in enumerate((True, False, True)):
        b_prev = bstate(layers)
        log = ST.increment(pm, layers, ids, mask, verdict, lr=1e-2, steps=4, tau=TAU)
        assert log["applied"] and abs(log["N_after"] - TAU) / TAU < 1e-4, log
        b_new = bstate(layers)
        # The per-step change of Delta W has norm tau, measured independently of the function's own log.
        assert abs(ST.increment_norm(layers, [x - y for x, y in zip(b_new, b_prev)]) - TAU) / TAU < ST.NORM_TOL
        assert all(torch.equal(x, y) for x, y in zip(astate(layers), a0))
        assert all(m.lora_B["default"].weight.grad is None for m in layers)
    total = U.global_norm(layers)
    assert TAU < total <= 3 * TAU + 1e-6     # three increments of norm tau; aligned ones add up


def delta_w(layers):
    return [m.scaling["default"] * (m.lora_B["default"].weight @ m.lora_A["default"].weight).detach() for m in layers]


@needs_tok
def test_first_increment_is_the_per_item_update(tok, tiny):
    """From B = 0 and with one optimizer step, the stream step equals the per-item rule (src/ttt/update.py update, whose
    loop is replicated here on CPU with A and B trainable): A's gradient is 0 at B = 0, and the weight decay AdamW
    applies to A is cancelled by the global rescale, so Delta W is the same."""
    pm, layers = tiny
    ids, mask = seq(tok)
    U.reset(layers, 11)
    ST.increment(pm, layers, ids, mask, True, lr=1e-2, steps=1, tau=TAU)
    dw_stream = delta_w(layers)
    U.reset(layers, 11)
    for m in layers:
        m.lora_A["default"].weight.requires_grad_(True)
    try:
        x, msk = torch.tensor([ids]), torch.tensor([mask[1:]], dtype=torch.bool)
        opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-2)
        loss = U.answer_loss(pm(input_ids=x).logits[:, :-1][msk].float(), x[:, 1:][msk], True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        U.rescale(layers, TAU)
        dw_item = delta_w(layers)
    finally:
        for m in layers:
            m.lora_A["default"].weight.requires_grad_(False)
            m.lora_A["default"].weight.grad = m.lora_B["default"].weight.grad = None
    assert all(torch.allclose(x, y, rtol=1e-4, atol=1e-9) for x, y in zip(dw_stream, dw_item))


@needs_tok
def test_no_answer_span_leaves_the_adapter(tok, tiny):
    pm, layers = tiny
    U.reset(layers, 3)
    ST.random_increment(layers, TAU, seed=5)
    before = bstate(layers)
    ids, mask = seq(tok, a0="I am not sure.")
    assert mask is None
    log = ST.increment(pm, layers, ids, mask, True, lr=1e-2, steps=4, tau=TAU)
    assert log == {"anomaly": "no_answer_span", "applied": False}
    assert all(torch.equal(x, y) for x, y in zip(bstate(layers), before))


@needs_tok
def test_random_increment_seeded(tiny):
    pm, layers = tiny
    U.reset(layers, 3)
    l1 = ST.random_increment(layers, TAU, seed=9)
    b1 = bstate(layers)
    U.reset(layers, 3)
    ST.random_increment(layers, TAU, seed=9)
    assert all(torch.equal(x, y) for x, y in zip(bstate(layers), b1))
    assert abs(l1["N_after"] - TAU) / TAU < 1e-4
    U.reset(layers, 3)
    ST.random_increment(layers, TAU, seed=10)
    assert not all(torch.equal(x, y) for x, y in zip(bstate(layers), b1))


# ---- the full loop with a stub generator: invariants and resume -------------------------------------------------

class StubGen:
    """Deterministic outputs that depend on the prompt and the trajectory's state, so that a resume which does not
    restore the state exactly changes the outputs. The base state (B = 0 or no log) gives one output per prompt."""
    needs_files = False

    def __init__(self):
        self.r = None

    def fingerprint(self, t):
        if t is None or self.r.cond[t] == "context":
            return "base"
        b = self.r.state[t]["B"]
        if all(not x.any() for x in b):
            return "base"
        return hashlib.sha256(b"".join(x.detach().cpu().numpy().tobytes() for x in b)).hexdigest()

    def generate(self, reqs):
        tok, end = self.r.tok, self.r.tok.convert_tokens_to_ids("<|im_end|>")
        res = []
        for q in reqs:
            outs = []
            for j in range(q["n"]):
                hv = int(hashlib.sha256(f"{q['prompt']}|{self.fingerprint(q['state'])}|{q['seed']}|{j}".encode())
                         .hexdigest()[:8], 16)
                text = "Let me think." if hv % 5 == 0 else f"Adding up gives {hv % 7}.\n\n\\boxed{{{hv % 7}}}"
                outs.append({"text": text, "token_ids": tok(text, add_special_tokens=False).input_ids + [end],
                             "finish_reason": "stop"})
            res.append(outs)
        return res


def make_stream(tok, tiny, d, probe, steps=4, seeds=(0, 1)):
    pm, layers = tiny
    S = ST.select_items("dev", steps, 2, list(seeds))
    cfg = {"model": "tiny", "split": "dev", "conditions": ST.CONDITIONS, "seeds": list(seeds), "steps": steps,
           "eval_every": 2, "n_eval": 2, "memory_tokens": 200, "max_tokens": 16, "n_samples": 2, "t_sample": 0.7,
           "train_steps": 2, "tau": TAU, "lr": 1e-2, "t_star": 0.0}
    gen = StubGen()
    st = ST.Stream(S=S, cfg=cfg, tok=tok, pm=pm, layers=layers, gen=gen, d=d, probe_path=probe, commit="test",
                   log=lambda m: None)
    gen.r = st
    return st


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    import numpy as np
    p = tmp_path_factory.mktemp("probe") / "probe.npz"
    rng = np.random.default_rng(0)
    np.savez(p, layer=1, C=0.001, mean=np.zeros(64), scale=np.ones(64), coef=rng.normal(size=64), intercept=0.0)
    return str(p)


def rows(path, drop=("gen_s", "hf_s")):
    return [{k: v for k, v in json.loads(l).items() if k not in drop} for l in open(path, encoding="utf-8")]


@needs_tok
def test_resume_reproduces_the_uninterrupted_run(tok, tiny, probe, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(), b.mkdir()
    st = make_stream(tok, tiny, a, probe)
    assert st.run() == "done"
    ma = st.finalize()
    assert ma["zero_step_a0_identical"] and ma["zero_step_verbal_identical"]
    assert ma["realized_increment_norm_max_rel_err"] < ST.NORM_TOL
    st = make_stream(tok, tiny, b, probe)
    assert st.run(stop_at_step=3) == "stopped"
    assert json.loads((b / "state" / "step.json").read_text())["k"] == 3
    for m in tiny[1]:                                          # a restarted process starts from scratch
        with torch.no_grad():
            m.lora_B["default"].weight.normal_()
            m.lora_A["default"].weight.normal_()
    st = make_stream(tok, tiny, b, probe)
    assert st.run() == "done"
    st.finalize()
    for name in ("steps.jsonl", "evals.jsonl", "eval_texts.jsonl", "step_texts.jsonl"):
        assert rows(a / name) == rows(b / name), name
    sa, sb = json.loads((a / "eval_summary.json").read_text()), json.loads((b / "eval_summary.json").read_text())
    assert sa == sb and sorted(sa) == ["0", "2", "4"] and list(sa["0"]) == ["base"]
    assert sorted(p.name for p in (a / "state").iterdir()) == sorted(p.name for p in (b / "state").iterdir())
    for p in (a / "state").iterdir():
        if p.suffix == ".pt":
            assert all(torch.equal(x, y) for x, y in zip(torch.load(p), torch.load(b / "state" / p.name)))
        elif p.name == "step.json":
            assert json.loads(p.read_text())["k"] == json.loads((b / "state" / p.name).read_text())["k"] == 4
        else:
            assert p.read_text(encoding="utf-8") == (b / "state" / p.name).read_text(encoding="utf-8")
    # The run actually trained, kept a log, and the conditions diverged.
    steps = rows(a / "steps.jsonl")
    assert any(r["update"] and r["update"].get("applied") for r in steps if r["condition"] in ("self", "ext"))
    assert any(r.get("memory_entries") for r in steps if r["condition"] == "context")
    last = {r["traj"]: r for r in steps if r["step"] == 3}
    assert last["self-s0"]["total_norm"] > 0 and last["random-s0"]["total_norm"] > 0
    assert sorted(p.name for p in (a / "checkpoints").iterdir()) == ["t002", "t004"]


@needs_tok
def test_resume_refuses_another_config(tok, tiny, probe, tmp_path):
    st = make_stream(tok, tiny, tmp_path, probe, steps=2, seeds=(0,))
    assert st.run(stop_at_step=1) == "stopped"
    st = make_stream(tok, tiny, tmp_path, probe, steps=2, seeds=(0,))
    st.cfg["tau"] = 2 * TAU
    with pytest.raises(SystemExit):
        st.resume()


@needs_tok
def test_zero_step_check_catches_a_nonzero_start(tok, tiny, probe, tmp_path):
    st = make_stream(tok, tiny, tmp_path, probe, steps=2, seeds=(0,))
    st.resume()
    st.evaluate(0)
    with torch.no_grad():
        st.state["self-s0"]["B"][0].add_(1.0)          # a "zero-step" adapter that is not zero
    with pytest.raises(SystemExit):
        st.step(0)


def test_summary_definitions():
    rs = [{"correct": True, "sample_correct": [True, False], "n_classes": 2, "s": 3.0, "p_yes_norm": 0.95,
           "says_correct": True, "probe": 0.8, "status": "ok", "out_tokens": 10},
          {"correct": False, "sample_correct": [False, False], "n_classes": 1, "s": -1.0, "p_yes_norm": 0.27,
           "says_correct": False, "probe": 0.6, "status": "ok", "out_tokens": 20}]
    s = ST.summarize(rs, t_star=0.0)
    assert s["acc_greedy"] == 0.5 and s["acc_samples"] == 0.25 and s["mean_classes"] == 1.5
    assert s["frac_one_class"] == 0.5 and s["mean_s"] == 1.0 and s["self_verdict_acc"] == 1.0
    assert s["agreement"] == 0.5 and s["probe_acc"] == 0.5 and s["rank_agreement"] is None   # n < 3
    assert math.isclose(s["mean_probe"], 0.7)


@needs_tok
def test_served_adapters_carry_the_trajectory_state(tok, tiny, probe, tmp_path):
    """The saved PEFT adapters vLLM serves (prepare_serving) hold the seed's A and the trajectory's own B."""
    from safetensors.torch import load_file
    pm, layers = tiny
    st = make_stream(tok, tiny, tmp_path, probe, steps=2, seeds=(0, 1))
    st.gen.needs_files = True
    st.resume()
    g = torch.Generator().manual_seed(0)
    for t in st.adapter_trajs:
        st.state[t]["B"] = [torch.randn(b.shape, generator=g) for b in st.state[t]["B"]]
    st.prepare_serving(0)
    names = {id(m): n for n, m in pm.named_modules()}
    for t in st.adapter_trajs:
        name, lid, path = st.lora(t, 0)
        assert name == f"{t}-t000" and Path(path).parent == tmp_path / "serve" / "t000"
        w = load_file(str(Path(path) / "adapter_model.safetensors"))
        for m, a, b in zip(layers, st.A[st.seed[t]], st.state[t]["B"]):
            assert torch.equal(w[names[id(m)] + ".lora_A.weight"], a)
            assert torch.equal(w[names[id(m)] + ".lora_B.weight"], b)
    assert len({st.lora(t, k)[1] for t in st.adapter_trajs for k in range(201)}) == len(st.adapter_trajs) * 201
    assert st.lora("context-s0", 0) is None
    st.prepare_serving(1)
    assert [p.name for p in (tmp_path / "serve").iterdir()] == ["t001"]


@needs_tok
def test_deadline_stop_is_clean_and_resumable(tok, tiny, probe, tmp_path):
    import time
    st = make_stream(tok, tiny, tmp_path, probe, steps=2, seeds=(0,))
    assert st.run(stop_before_epoch=time.time() - 1) == "stopped"           # nothing fits: stops before eval 0
    assert not (tmp_path / "state" / "step.json").exists() and not list((tmp_path / "evals").iterdir())
    st = make_stream(tok, tiny, tmp_path, probe, steps=2, seeds=(0,))
    assert st.run(stop_before_epoch=time.time() + 3600) == "done"
    assert json.loads((tmp_path / "state" / "step.json").read_text())["k"] == 2
