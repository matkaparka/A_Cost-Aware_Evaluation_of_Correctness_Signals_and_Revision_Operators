"""Persistent streams (H4c; exploratory, descriptive; design Exp 5 "Persistent streams", §3.6, §7.1; prereg §3.3;
configs/blocks.yaml 4B-streams, priority 11). Written after the freeze: the choices below are for a DEVIATIONS entry.

A trajectory is one evolving model state run over a 200-item sequence. At step k the state S_k answers item k
(a0, greedy), judges its own answer (the thresholded verbal verdict), and then moves to S_{k+1}:
  self     trains an increment on (q, a0), the self-verdict selecting the loss (src/ttt/update.py rule, below);
  ext      the same with the true correctness of a0 (ext-1.0);
  random   adds a matched-norm random increment; a0 and the self-verdict are logged, not used;
  context  trains nothing; appends (q, a0, self-verdict) to a log held in context. Its a0, its verdict and its
           evaluation all see the log (the log is the state, as the adapter is for the other three).
4 conditions x 3 order seeds = 12 trajectories, run in lockstep: one vLLM call generates the 12 a0 of a step, then
HF computes the 12 verdicts and the 9 adapter increments (the context log needs none).

Items (no outcome data is read; questions and gold come from data/processed/dnat.jsonl):
- Stream items, --split test: the first --steps original-pool D-nat test items in the seeded test processing order,
  spillover set excluded, i.e. the first 200 items of the Exp 5 per-item pool (prereg §3.4; src/ttt/test.py build).
- Eval items (always dev): the first N_EVAL original-pool D-nat dev items in the seeded dev processing order, KL probe
  set excluded. Disjoint from the stream items (other split, or other positions of the dev order) and from the
  spillover set (test). --split dev (smoke tests only): stream items are the dev items after the first N_EVAL.
- Order seed s: the stream items sorted by h("stream-order-v1", s, item_id); every condition of seed s uses it.

Update rule per step (trained conditions; frozen Exp 5 values: tau and lr from analysis/out/tier2.json, 4 steps):
- The adapter is LoRA rank 8 / alpha 16 on q, k, v, o (src/ttt/update.py attach). A is fixed per order seed (the
  seeded init of update.reset with seed h("stream-lora-A-v1", s)) and only B is trained: with A trained, the change
  B'A' - B_kA_k cannot be rescaled to norm tau and stay rank 8, and A's own drift scales with the adapter's size.
- A fresh AdamW (torch defaults, as update.update) on the current B for STEPS steps, on the training sequence and
  loss of update.update (answer span = last \\boxed{} of a0; NLL if v = correct, exact unlikelihood if incorrect).
- Increment dB = B' - B_k, rescaled by one global factor c = tau / N(dB), N(dB) = sqrt(sum_m ||(alpha/r) dB_m A_m||_F^2),
  so N(Delta W_{k+1} - Delta W_k) = tau exactly; the total norm N(Delta W_{k+1}) is logged.
- Anomaly rules of update.update: no answer span, NaN/Inf loss, or N = 0 -> no increment at this step (the adapter
  is unchanged), counted; c > 100 -> applied, flagged.
- random: dB ~ N(0, 1) per module (generator seed h("stream-random-v1", s, k)), then the same global rescale to tau.

Context log: entries (q, a0, verdict) in MEMORY_ENTRY wording (the Exp 5 context template's phrasing), newest last;
after each append the oldest entries are dropped until the rendered log block is <= --memory-tokens tokens (an entry
that alone exceeds the budget is dropped too). An empty log gives exactly the q-only prompt, so step 0 is the base.

Evaluation of S_k at k = 0, EVAL_EVERY, ..., steps (S_0 = the base, evaluated once and shared): per eval item, a greedy
answer and N_SAMPLES samples at T_SAMPLE (seed h("stream-sample-v1", item), shared by every trajectory and checkpoint),
max --max-tokens new tokens; the verbal P(yes) of the greedy answer (src/score/verbal.py query and candidates,
first-token shortcut, teacher-forced in HF); the frozen probe's P(correct) at the last content token of the greedy
answer in its generation context (src/probe/cache.py position, src/probe/train.py score_frozen). The context
trajectories' prompts, verbal queries and probe contexts carry their log. Definitions in summarize().

Generation: vLLM with VLLM_BATCH_INVARIANT=1, LoRA served from saved PEFT adapters (as src/ttt/test.py), max 2048
new tokens for every generation, a0 included (prereg §1, "Exp 5 served generations (all cells)"). --backend hf
replaces vLLM by HF generation (local smoke tests only). HF training runs with update.deterministic().

Invariants checked during the run (the run stops on a failure): at step 0, within each seed, every condition's a0 is
token-identical (zero adapter = base in vLLM) and its verbal score is bitwise identical (B = 0 adapter = adapter
disabled in HF); every applied increment has N within 1 % of tau.

Resume (CLAUDE.md rule 8): the state S_k (B of every adapter trajectory, every context log) is committed after each
step under state/ (step.json written last); a restarted run with the same config hash and code commit continues from
the last committed step and reproduces the uninterrupted run (no global RNG is used: every random draw is keyed by
hash). Evaluations are resumable per checkpoint. --stop-before-epoch stops cleanly at a step boundary.
Outputs: steps.jsonl, step_texts.jsonl, evals.jsonl, eval_texts.jsonl, eval_summary.json, metrics.json,
artifacts.json, manifest.json; large: checkpoints/t<k>/ (B and logs at every evaluation), state/.

Usage: VLLM_BATCH_INVARIANT=1 python -m src.ttt.streams --model 4B --split test --allow-test
       [--stop-before-epoch E] [--steps 200 --eval-every 25 --n-eval 100 --seeds 0 1 2 --conditions ...] [--tag T]
       [--dry-run] [--backend hf]
"""
import argparse
import contextlib
import json
import math
import os
import shutil
import time
from collections import Counter

from ..data.build import h
from ..data.verifiers import self_verdict
from ..gen import render as R
from .. import runlib
from ..runlib import MODELS, REPO
from ..score import scorer
from ..score.scorer import SCORER_VERSION, equivalent, score
from . import update as U
from .pilot import STEPS
from .test import N_SAMPLES, T_SAMPLE

BASE_CONDITIONS = ["self", "ext", "random", "context"]
# Follow-up controls (DEVIATIONS D16; review #8), off by default:
#   nll       an NLL update at every step, whatever the self-verdict (logged): as many updates as self, none of
#             them unlikelihood (the reference self stream judged 96.8 % of its answers incorrect);
#   matched   the reference run's realized self-stream verdict sequence of the same order seed, permuted by
#             h("stream-matched-v1", seed): the same rate of "incorrect" verdicts as the self stream, unrelated to
#             the matched stream's own correctness; the verdict selects the loss as in self.
CONDITIONS = BASE_CONDITIONS + ["nll", "matched"]
TRAINED = {"self", "ext", "nll", "matched"}
MATCHED_REF = "results/main/4B/main-streams-test-4B-f15b933/steps.jsonl"
N_SEEDS, N_STEPS, EVAL_EVERY, N_EVAL = 3, 200, 25, 100
MEMORY_TOKENS = 4096
MAX_TOKENS = 2048
PROBE_THRESHOLD = 0.5
NORM_TOL = 0.01
PROBES = {"4B": ("results/p3/4B/p3-probe-dev-ext-4B-b7cf0d6/probe.npz",
                 "a62621d9ef55fed97f3de6a16b269715fe402b2e882c4bcae7b436172c8bc833"),
          "1.7B": ("results/p3/1.7B/p3-probe-dev-ext-1.7B-b7cf0d6/probe.npz",
                   "db0c6752bfc88da58d797a954af863e7f6d862940a24c8dfc93f457c622e65d0")}
TIER2 = REPO / "analysis" / "out" / "tier2.json"
SELFVERDICT = REPO / "analysis" / "out" / "selfverdict.json"

MEMORY_HEADER = "Log of earlier problems, each with a previous attempt and its verdict:"
MEMORY_ENTRY = "Problem:\n{q}\n\nPrevious attempt:\n{a0}\n\nVerdict: the previous attempt is {verdict}."
MEMORY_SEP = "\n\n---\n\n"


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def seed_int(*parts):
    return int(h(*parts)[:8], 16)


def traj_name(cond, seed):
    return f"{cond}-s{seed}"


# ---- items ---------------------------------------------------------------------------------------------------

def select_items(split, n_steps, n_eval, seeds, root=REPO):
    """Stream items, eval items and the per-seed orders (rules in the module docstring). Reads no outcome data."""
    P = root / "data" / "processed"
    meta = {r["item_id"]: r for r in jl(P / "dnat.jsonl")}
    sp = json.loads((P / "split.json").read_text())
    rank = {s: {c: k for k, c in enumerate(sp["order"][s])} for s in ("dev", "test")}
    kl, spill = set(sp["kl_dev"]), set(sp["spillover_test"])

    def ordered(s, exclude):
        return sorted((i for i, r in meta.items() if r["split"] == s and r["canonical_id"] not in exclude),
                      key=lambda i: (rank[s][meta[i]["canonical_id"]], i))
    dev = ordered("dev", kl)
    eval_items = dev[:n_eval]
    pool = ordered("test", spill)[:n_steps] if split == "test" else dev[N_EVAL:N_EVAL + n_steps]
    if len(pool) < n_steps or len(eval_items) < n_eval:
        raise SystemExit(f"not enough items: stream {len(pool)}/{n_steps}, eval {len(eval_items)}/{n_eval}")
    orders = {s: sorted(pool, key=lambda i: h("stream-order-v1", s, i)) for s in seeds}
    return {"meta": meta, "pool": pool, "eval": eval_items, "orders": orders}


# ---- context log ---------------------------------------------------------------------------------------------

def memory_block(entries):
    return MEMORY_HEADER + "\n\n" + MEMORY_SEP.join(
        MEMORY_ENTRY.format(q=e["q"], a0=e["a0"], verdict=e["verdict"]) for e in entries)


def user_content(q, entries):
    """The user turn: the q-only prompt, preceded by the log when there is one."""
    question = R.PROMPT.format(q=q)
    return question if not entries else memory_block(entries) + MEMORY_SEP + question


def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False).input_ids)


def fit_memory(tok, entries, budget):
    """Drop the oldest entries until the log block fits the token budget. Returns (kept, n_dropped)."""
    kept = list(entries)
    while kept and n_tokens(tok, memory_block(kept)) > budget:
        kept.pop(0)
    return kept, len(entries) - len(kept)


# ---- adapter increments ----------------------------------------------------------------------------------------

def lora_b(layers):
    return [m.lora_B["default"].weight for m in layers]


def increment_norm(layers, delta):
    """N(dB) = sqrt(sum_m ||(alpha/r) dB_m A_m||_F^2) with each module's current (fixed) A."""
    import torch
    tot = 0.0
    with torch.no_grad():
        for m, dl in zip(layers, delta):
            dw = m.scaling["default"] * (dl @ m.lora_A["default"].weight)
            tot += float(dw.double().pow(2).sum())
    return math.sqrt(tot)


def _apply(bs, b0, delta, c):
    """B = B_k + c dB, module by module."""
    import torch
    with torch.no_grad():
        for b, x0, dl in zip(bs, b0, delta):
            b.copy_(x0 + c * dl)


def _restore(bs, b0):
    import torch
    with torch.no_grad():
        for b, x0 in zip(bs, b0):
            b.copy_(x0)


def increment(pm, layers, ids, mask, verdict_correct, *, lr, steps, tau):
    """One step of a trained stream on the adapter currently loaded (A fixed, B = B_k). B becomes B_k + c dB with
    N(c dB) = tau, or stays B_k on an anomaly. Returns the log dict."""
    import torch
    bs = lora_b(layers)
    b0 = [b.detach().clone() for b in bs]
    if mask is None:
        return {"anomaly": "no_answer_span", "applied": False}
    dev = bs[0].device
    x = torch.tensor([ids], device=dev)
    m = torch.tensor([mask[1:]], device=dev, dtype=torch.bool)
    opt = torch.optim.AdamW(bs, lr=lr)
    loss = None
    for _ in range(steps):
        loss = U.answer_loss(pm(input_ids=x).logits[:, :-1][m].float(), x[:, 1:][m], verdict_correct)
        if not torch.isfinite(loss):
            _restore(bs, b0)
            _clear_grads(bs)
            return {"anomaly": "nan_loss", "applied": False}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    _clear_grads(bs)
    delta = [b.detach() - x0 for b, x0 in zip(bs, b0)]
    n0 = increment_norm(layers, delta)
    if not math.isfinite(n0) or n0 == 0:
        _restore(bs, b0)
        return {"anomaly": "zero_or_nan_norm", "applied": False, "N_before": n0}
    c = tau / n0
    _apply(bs, b0, delta, c)
    n1 = increment_norm(layers, [b.detach() - x0 for b, x0 in zip(bs, b0)])
    return {"anomaly": "c_gt_100" if c > 100 else None, "applied": True, "N_before": n0, "c": c, "N_after": n1,
            "final_loss": float(loss.detach())}


def random_increment(layers, tau, seed):
    """Matched-norm random increment: Gaussian dB per module (seeded), global rescale to tau."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    bs = lora_b(layers)
    b0 = [b.detach().clone() for b in bs]
    delta = [torch.randn(b.shape, generator=g, dtype=torch.float32).to(b.device) for b in bs]
    n0 = increment_norm(layers, delta)
    c = tau / n0
    _apply(bs, b0, delta, c)
    n1 = increment_norm(layers, [b.detach() - x0 for b, x0 in zip(bs, b0)])
    return {"anomaly": None, "applied": True, "N_before": n0, "c": c, "N_after": n1}


def _clear_grads(bs):
    for b in bs:
        b.grad = None


# ---- evaluation metrics ----------------------------------------------------------------------------------------

def n_classes(preds):
    """Distinct answers among samples: extracted answers (scorer pred) are one class when equivalent() holds in either
    direction; outputs without an extracted answer (None) form one class together."""
    reps = []
    for p in preds:
        for r in reps:
            if (p is None and r is None) or (p is not None and r is not None and (equivalent(p, r) or equivalent(r, p))):
                break
        else:
            reps.append(p)
    return len(reps)


def spearman(x, y):
    import numpy as np
    from scipy.stats import rankdata
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows, t_star):
    """Per (checkpoint, trajectory) summary of eval rows. Definitions (H4c; descriptive):
    acc_greedy        mean greedy correctness (frozen scorer; unparsed/truncated = wrong)
    acc_samples       mean correctness over the N_SAMPLES samples and items
    mean_classes      mean over items of n_classes(sample answers) (1 = all samples agree, N_SAMPLES = all differ)
    frac_one_class    share of items whose samples all agree
    mean_s            mean verbal log-odds s = log P(yes) - log P(no) of the greedy answer (finite values)
    mean_p_yes_norm   mean P(yes) / (P(yes) + P(no))
    frac_says_correct share of items with s >= t* (the frozen self-verdict rule)
    self_verdict_acc  share of items where the self-verdict equals greedy correctness
    mean_probe        mean frozen-probe P(correct) at the greedy answer's last content token
    probe_acc         share of items where (P(correct) >= PROBE_THRESHOLD) equals greedy correctness
    agreement         share of items where (P(correct) >= PROBE_THRESHOLD) equals (s >= t*): probe-verbal agreement
    rank_agreement    Spearman correlation of probe P(correct) and s over items (None if either is constant)"""
    n = len(rows)
    fin = [r["s"] for r in rows if r["s"] is not None and math.isfinite(r["s"])]
    out = {"n": n,
           "acc_greedy": sum(r["correct"] for r in rows) / n,
           "acc_samples": sum(sum(r["sample_correct"]) / len(r["sample_correct"]) for r in rows) / n,
           "mean_classes": sum(r["n_classes"] for r in rows) / n,
           "frac_one_class": sum(r["n_classes"] == 1 for r in rows) / n,
           "mean_s": sum(fin) / len(fin) if fin else None,
           "mean_p_yes_norm": sum(r["p_yes_norm"] for r in rows if r["p_yes_norm"] is not None) / n,
           "frac_says_correct": sum(r["says_correct"] for r in rows) / n,
           "self_verdict_acc": sum(r["says_correct"] == r["correct"] for r in rows) / n,
           "mean_probe": sum(r["probe"] for r in rows) / n,
           "probe_acc": sum((r["probe"] >= PROBE_THRESHOLD) == r["correct"] for r in rows) / n,
           "agreement": sum((r["probe"] >= PROBE_THRESHOLD) == r["says_correct"] for r in rows) / n,
           "rank_agreement": spearman([r["probe"] for r in rows],
                                      [r["s"] if r["s"] is not None else -math.inf for r in rows]),
           "greedy_status": dict(Counter(r["status"] for r in rows)),
           "mean_out_tokens": sum(r["out_tokens"] for r in rows) / n}
    out["t_star"] = t_star
    return out


# ---- generation backends ---------------------------------------------------------------------------------------
# A request: {"prompt": text, "state": trajectory name or None (base / context), "lora": (name, int id, path) or None,
#             "n": int, "temperature": float, "max_tokens": int, "seed": int}
# An output: list of n dicts {"text", "token_ids", "finish_reason"}.

class VllmGen:
    needs_files = True

    def __init__(self, model_key, gpu_mem, max_loras, max_model_len):
        from vllm import LLM
        self.llm = LLM(model=MODELS[model_key], dtype="bfloat16", seed=0, gpu_memory_utilization=gpu_mem,
                       max_model_len=max_model_len, enable_lora=True, max_lora_rank=U.RANK, max_loras=max_loras,
                       max_cpu_loras=max(64, 4 * max_loras))

    def generate(self, reqs):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        params = [SamplingParams(n=r["n"], temperature=r["temperature"], max_tokens=r["max_tokens"], seed=r["seed"])
                  for r in reqs]
        loras = [None if r["lora"] is None else LoRARequest(*r["lora"]) for r in reqs]
        outs = self.llm.generate([r["prompt"] for r in reqs], params, lora_request=loras, use_tqdm=False)
        return [[{"text": o.text, "token_ids": list(o.token_ids), "finish_reason": o.finish_reason} for o in out.outputs]
                for out in outs]


class HfGen:
    """HF generation behind --backend hf (local smoke tests; not used for any reported number)."""
    needs_files = False

    def __init__(self, runner=None):
        self.r = runner

    def generate(self, reqs):
        import torch
        from transformers import GenerationConfig
        tok, pm = self.r.tok, self.r.pm
        stop = [t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if t is not None]
        pad = tok.pad_token_id if tok.pad_token_id is not None else stop[0]
        res = []
        for q in reqs:
            ids = torch.tensor([tok(q["prompt"], add_special_tokens=False).input_ids], device=self.r.dev)
            if q["temperature"] == 0:
                gc = GenerationConfig(do_sample=False, max_new_tokens=q["max_tokens"], num_return_sequences=1,
                                      eos_token_id=stop, pad_token_id=pad)
            else:
                torch.manual_seed(q["seed"])
                gc = GenerationConfig(do_sample=True, temperature=q["temperature"], top_k=0, top_p=1.0,
                                      max_new_tokens=q["max_tokens"], num_return_sequences=q["n"],
                                      eos_token_id=stop, pad_token_id=pad)
            with self.r.state_ctx(q["state"]), torch.no_grad():
                out = pm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=gc)
            outs = []
            for row in out:
                g = row[ids.shape[1]:].tolist()
                cut = next((k for k, t in enumerate(g) if t in stop), None)
                g = g if cut is None else g[:cut + 1]
                outs.append({"text": tok.decode(g, skip_special_tokens=True), "token_ids": g,
                             "finish_reason": "length" if cut is None else "stop"})
            res.append(outs)
        return res


# ---- the run ---------------------------------------------------------------------------------------------------

class Stream:
    def __init__(self, *, S, cfg, tok, pm, layers, gen, d, probe_path, commit, log=print):
        import torch
        self.S, self.cfg, self.tok, self.pm, self.layers, self.gen, self.d = S, cfg, tok, pm, layers, gen, d
        self.probe_path, self.commit, self.log = probe_path, commit, log
        self.meta, self.orders = S["meta"], S["orders"]
        self.dev = next(pm.parameters()).device
        self.trajs = [traj_name(c, s) for s in cfg["seeds"] for c in cfg["conditions"]]
        self.cond = {traj_name(c, s): c for s in cfg["seeds"] for c in cfg["conditions"]}
        self.seed = {traj_name(c, s): s for s in cfg["seeds"] for c in cfg["conditions"]}
        self.adapter_trajs = [t for t in self.trajs if self.cond[t] != "context"]
        self.matched = {int(s_): v for s_, v in cfg.get("matched_verdicts", {}).items()}
        for m in layers:
            m.lora_A["default"].weight.requires_grad_(False)
        self.A = {}
        for s in cfg["seeds"]:
            U.reset(layers, seed_int("stream-lora-A-v1", s))
            self.A[s] = [m.lora_A["default"].weight.detach().clone() for m in layers]
        U.zero_b(layers)
        from ..score.verbal import NO, QUERY, YES
        self.query = QUERY
        cand = {c: tok(c, add_special_tokens=False).input_ids for c in YES + NO}
        first = [v[0] for v in cand.values()]
        if not (all(len(v) == 1 for v in cand.values()) and len(set(first)) == len(first)):
            raise SystemExit("verbal first-token shortcut not valid for this tokenizer")
        self.yes_ids, self.no_ids = [cand[c][0] for c in YES], [cand[c][0] for c in NO]
        self.stop_ids = {t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None}
        self.state = {}
        self.timing = {"step_s": None, "eval_s_per_request": None}
        (d / "state").mkdir(exist_ok=True)
        (d / "steps").mkdir(exist_ok=True)
        (d / "evals").mkdir(exist_ok=True)
        self.torch = torch

    # -- adapter state --
    def zero_state(self):
        return {t: ({"memory": []} if self.cond[t] == "context" else
                    {"B": [self.torch.zeros_like(m.lora_B["default"].weight) for m in self.layers]}) for t in self.trajs}

    def load_lora(self, t):
        with self.torch.no_grad():
            for m, a, b in zip(self.layers, self.A[self.seed[t]], self.state[t]["B"]):
                m.lora_A["default"].weight.copy_(a)
                m.lora_B["default"].weight.copy_(b)

    @contextlib.contextmanager
    def state_ctx(self, t):
        """HF model in the state of trajectory t: its adapter loaded, or the adapter disabled (context, base)."""
        if t is not None and self.cond[t] != "context":
            self.load_lora(t)
            yield
        else:
            with self.pm.disable_adapter():
                yield

    def memory(self, t):
        return self.state[t]["memory"] if t is not None and self.cond[t] == "context" else []

    # -- commit / resume --
    def commit_state(self, k):
        torch, sd = self.torch, self.d / "state"
        for t in self.trajs:
            st = self.state[t]
            if "B" in st:
                torch.save([b.detach().cpu() for b in st["B"]], sd / f"B_{t}_t{k:03d}.pt")
            else:
                (sd / f"memory_{t}_t{k:03d}.json").write_text(json.dumps(st["memory"], ensure_ascii=False), encoding="utf-8")
        tmp = sd / "step.json.tmp"
        tmp.write_text(json.dumps({"k": k, "code_commit": self.commit, "config_sha256": self.cfg_sha(),
                                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        os.replace(tmp, sd / "step.json")
        for p in sd.iterdir():
            if p.name != "step.json" and not p.name.endswith(f"_t{k:03d}.pt") and not p.name.endswith(f"_t{k:03d}.json"):
                p.unlink()
        if k % self.cfg["eval_every"] == 0 or k == self.cfg["steps"]:
            cd = self.d / "checkpoints" / f"t{k:03d}"
            cd.mkdir(parents=True, exist_ok=True)
            for p in sd.iterdir():
                if p.name != "step.json":
                    shutil.copy2(p, cd / p.name)

    def cfg_sha(self):
        return runlib.sha256_text(json.dumps(self.cfg, sort_keys=True))

    def resume(self):
        """Load the last committed state; returns its step k (0 if nothing is committed)."""
        sd = self.d / "state"
        if not (sd / "step.json").exists():
            self.state = self.zero_state()
            return 0
        st = json.loads((sd / "step.json").read_text())
        if st["code_commit"] != self.commit or st["config_sha256"] != self.cfg_sha():
            raise SystemExit("state/ was written by another code commit or config; refusing to resume (rule 8)")
        k = st["k"]
        self.state = {}
        for t in self.trajs:
            if self.cond[t] == "context":
                self.state[t] = {"memory": json.loads((sd / f"memory_{t}_t{k:03d}.json").read_text(encoding="utf-8"))}
            else:
                self.state[t] = {"B": [b.to(self.dev) for b in self.torch.load(sd / f"B_{t}_t{k:03d}.pt")]}
        self.log(f"resumed at step {k}")
        return k

    # -- serving --
    def serve_dir(self, k):
        return self.d / "serve" / f"t{k:03d}"

    def prepare_serving(self, k):
        """Saved PEFT adapters of S_k for vLLM (rebuilt from the state; older ones removed)."""
        if not self.gen.needs_files:
            return
        sd = self.serve_dir(k)
        if not (sd / "done").exists():
            sd.mkdir(parents=True, exist_ok=True)
            for t in self.adapter_trajs:
                self.load_lora(t)
                self.pm.save_pretrained(str(sd / t))
            (sd / "done").write_text("")
        for p in (self.d / "serve").iterdir():
            if p.name != sd.name:
                shutil.rmtree(p, ignore_errors=True)

    def lora(self, t, k):
        if t is None or self.cond[t] == "context" or not self.gen.needs_files:
            return None
        return (f"{t}-t{k:03d}", 1 + 32 * k + self.trajs.index(t), str(self.serve_dir(k) / t))

    # -- HF scoring --
    def verbal(self, content, answer):
        msgs = [{"role": "user", "content": content}, {"role": "assistant", "content": answer},
                {"role": "user", "content": self.query}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        x = self.torch.tensor([self.tok(text, add_special_tokens=False).input_ids], device=self.dev)
        with self.torch.no_grad():
            lp = self.torch.log_softmax(self.pm(input_ids=x, logits_to_keep=1).logits[0, -1].float(), -1)
        lp = lp[self.yes_ids + self.no_ids].tolist()       # as src/score/verbal.py: fp32 log-softmax, then float64 sums
        ny = len(self.yes_ids)
        py, pn = sum(math.exp(v) for v in lp[:ny]), sum(math.exp(v) for v in lp[ny:])
        s = (math.log(py) - math.log(pn)) if py > 0 and pn > 0 else (math.inf if py > 0 else -math.inf)
        return {"p_yes": py, "p_no": pn, "s": s, "p_yes_norm": py / (py + pn) if py + pn > 0 else None,
                "says_correct": self_verdict({"p_yes": py, "p_no": pn}, self.cfg["t_star"])}

    def probe(self, prompt, answer_ids):
        """Frozen probe P(correct) at the last content token of the answer in its generation context (cache.py)."""
        import numpy as np
        from ..probe.train import score_frozen
        p_ids = self.tok(prompt, add_special_tokens=False).input_ids
        ids = p_ids + list(answer_ids)
        last = len(ids) - 1
        if answer_ids and answer_ids[-1] in self.stop_ids:
            last -= 1
        with self.torch.no_grad():
            hs = self.pm(input_ids=self.torch.tensor([ids], device=self.dev), output_hidden_states=True,
                         logits_to_keep=1).hidden_states
        X = self.torch.stack([x[0, last] for x in hs]).to(self.torch.float16).cpu().numpy()
        return float(score_frozen(self.probe_path, np.stack([X]))[0])

    def prompt(self, t, q):
        return R.render_messages(self.tok, [{"role": "user", "content": user_content(q, self.memory(t))}])

    # -- one step --
    def step(self, k):
        c = cfg = self.cfg
        t0 = time.time()
        self.prepare_serving(k)
        item = {t: self.orders[self.seed[t]][k] for t in self.trajs}
        reqs = [{"prompt": self.prompt(t, self.meta[item[t]]["question"]), "state": t, "lora": self.lora(t, k),
                 "n": 1, "temperature": 0.0, "max_tokens": cfg["max_tokens"], "seed": 0} for t in self.trajs]
        outs = [o[0] for o in self.gen.generate(reqs)]
        t1 = time.time()
        rows, texts, new_state = [], [], {}
        for t, o in zip(self.trajs, outs):
            i = item[t]
            q, gold = self.meta[i]["question"], self.meta[i]["gold"]
            sc = score(o["text"], gold, finish_reason=o["finish_reason"])
            mem = self.memory(t)
            with self.state_ctx(t):
                v = self.verbal(user_content(q, mem), o["text"])
                cond = self.cond[t]
                verdict = {"self": v["says_correct"], "context": v["says_correct"], "ext": bool(sc["correct"]),
                           "random": None, "nll": True,
                           "matched": self.matched[self.seed[t]][k] if cond == "matched" else None}[cond]
                upd, extra = None, {}
                if cond in TRAINED:
                    ids, mask = U.training_sequence(self.tok, q, o["token_ids"])
                    upd = increment(self.pm, self.layers, ids, mask, verdict, lr=c["lr"], steps=c["train_steps"],
                                    tau=c["tau"])
                elif cond == "random":
                    upd = random_increment(self.layers, c["tau"], seed_int("stream-random-v1", self.seed[t], k))
                if cond == "context":
                    entry = {"q": q, "a0": o["text"], "verdict": "correct" if verdict else "incorrect", "item_id": i,
                             "step": k}
                    kept, dropped = fit_memory(self.tok, mem + [entry], c["memory_tokens"])
                    new_state[t] = {"memory": kept}
                    extra = {"memory_entries": len(mem), "memory_tokens": n_tokens(self.tok, memory_block(mem)) if mem else 0,
                             "memory_dropped_after": dropped, "memory_items": [e["item_id"] for e in mem]}
                else:
                    new_state[t] = {"B": [b.detach().clone() for b in lora_b(self.layers)]}
                    extra = {"total_norm": U.global_norm(self.layers)}
            if upd and upd.get("applied") and abs(upd["N_after"] - c["tau"]) / c["tau"] > NORM_TOL:
                raise SystemExit(f"step {k} {t}: increment norm {upd['N_after']} not within {NORM_TOL:.0%} of tau")
            rows.append({"traj": t, "condition": self.cond[t], "seed": self.seed[t], "step": k, "item_id": i,
                         "dataset": self.meta[i]["dataset"], "a0_correct": sc["correct"], "a0_status": sc["status"],
                         "a0_pred": sc["pred"], "a0_tokens": len(o["token_ids"]), "finish_reason": o["finish_reason"],
                         "a0_sha256": runlib.sha256_text(json.dumps(o["token_ids"])), **v, "verdict": verdict,
                         "update": upd, **extra})
            texts.append({"traj": t, "step": k, "item_id": i, "text": o["text"]})
        if k == 0:
            self.check_zero_step(rows)
        t2 = time.time()
        for r in rows:
            r.update({"gen_s": t1 - t0, "hf_s": t2 - t1})
        for name, rr in ((f"t{k:03d}.jsonl", rows), (f"t{k:03d}_texts.jsonl", texts)):
            with open(self.d / "steps" / name, "w", encoding="utf-8") as f:
                for r in rr:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.state = new_state
        self.commit_state(k + 1)
        self.timing["step_s"] = time.time() - t0
        self.log(f"{time.strftime('%H:%M:%S')} step {k}: gen {t1 - t0:.1f}s, hf {t2 - t1:.1f}s; "
                 + " ".join(f"{r['traj']}:{int(bool(r['a0_correct']))}{'v' if r['says_correct'] else 'x'}" for r in rows))

    def check_zero_step(self, rows):
        """Step 0, within each seed: every condition answers the same item from the base state."""
        for s in self.cfg["seeds"]:
            rs = [r for r in rows if r["seed"] == s]
            if len({r["a0_sha256"] for r in rs}) != 1:
                raise SystemExit(f"step 0, seed {s}: zero adapter != base (a0 differs across conditions)")
            if len({repr(r["s"]) for r in rs}) != 1:
                raise SystemExit(f"step 0, seed {s}: verbal score of the B = 0 adapter != adapter disabled")

    # -- evaluation --
    def evaluate(self, k):
        done = self.d / "evals" / f"t{k:03d}.json"
        if done.exists():
            return
        cfg, t0 = self.cfg, time.time()
        trajs = [None] if k == 0 else self.trajs        # S_0 is the base for every trajectory
        self.prepare_serving(k)
        reqs, keys = [], []
        for t in trajs:
            for i in self.S["eval"]:
                p = self.prompt(t, self.meta[i]["question"])
                base = {"prompt": p, "state": t, "lora": self.lora(t, k), "max_tokens": cfg["max_tokens"]}
                reqs.append({**base, "n": 1, "temperature": 0.0, "seed": 0})
                reqs.append({**base, "n": cfg["n_samples"], "temperature": cfg["t_sample"],
                             "seed": seed_int("stream-sample-v1", i)})
                keys.append((t, i, p))
        outs = self.gen.generate(reqs)
        t1 = time.time()
        rows, texts = [], []
        for j, (t, i, p) in enumerate(keys):
            g, smp = outs[2 * j][0], outs[2 * j + 1]
            gold, q = self.meta[i]["gold"], self.meta[i]["question"]
            sg = score(g["text"], gold, finish_reason=g["finish_reason"])
            ss = [score(o["text"], gold, finish_reason=o["finish_reason"]) for o in smp]
            with self.state_ctx(t):
                v = self.verbal(user_content(q, self.memory(t)), g["text"])
                pr = self.probe(p, g["token_ids"])
            rows.append({"k": k, "traj": t or "base", "condition": self.cond[t] if t else "base",
                         "seed": self.seed[t] if t else None, "item_id": i, "dataset": self.meta[i]["dataset"],
                         "correct": sg["correct"], "status": sg["status"], "pred": sg["pred"],
                         "out_tokens": len(g["token_ids"]), "finish_reason": g["finish_reason"],
                         "sample_correct": [x["correct"] for x in ss], "sample_status": [x["status"] for x in ss],
                         "sample_pred": [x["pred"] for x in ss], "n_classes": n_classes([x["pred"] for x in ss]),
                         **{key: v[key] for key in ("p_yes", "p_no", "s", "p_yes_norm", "says_correct")}, "probe": pr,
                         "memory_entries": len(self.memory(t))})
            texts.append({"k": k, "traj": t or "base", "item_id": i, "greedy": g["text"], "samples": [o["text"] for o in smp]})
        t2 = time.time()
        summary = {}
        for t in trajs:
            name = t or "base"
            summary[name] = summarize([r for r in rows if r["traj"] == name], cfg["t_star"])
        for name, rr in ((f"t{k:03d}.jsonl", rows), (f"t{k:03d}_texts.jsonl", texts)):
            with open(self.d / "evals" / name, "w", encoding="utf-8") as f:
                for r in rr:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp = done.with_suffix(".tmp")
        tmp.write_text(json.dumps({"k": k, "summary": summary, "gen_s": t1 - t0, "hf_s": t2 - t1,
                                   "n_requests": len(reqs)}, indent=1))
        os.replace(tmp, done)
        self.timing["eval_s_per_request"] = (time.time() - t0) / len(reqs)
        self.log(f"{time.strftime('%H:%M:%S')} eval {k}: gen {t1 - t0:.0f}s, hf {t2 - t1:.0f}s; "
                 + " ".join(f"{n}:{s['acc_greedy']:.2f}" for n, s in summary.items()))

    # -- main loop --
    def run(self, stop_before_epoch=None, stop_at_step=None):
        """Returns "done" or "stopped". stop_at_step: stop before this step (tests)."""
        k = self.resume()
        n, every = self.cfg["steps"], self.cfg["eval_every"]

        def must_stop(need):
            return stop_before_epoch is not None and time.time() + need > stop_before_epoch

        def eval_need(k):
            # Eval 0 serves the base only (1/12 of a later eval's requests): scale by request count, with a margin.
            n_req = 2 * len(self.S["eval"]) * (1 if k == 0 else len(self.trajs))
            per = self.timing["eval_s_per_request"]
            return 1800 if per is None else 1.5 * per * n_req
        while k <= n:
            if stop_at_step is not None and k >= stop_at_step:
                return "stopped"
            if (k % every == 0 or k == n) and not (self.d / "evals" / f"t{k:03d}.json").exists():
                if must_stop(eval_need(k)):
                    self.log(f"stop before eval {k}: deadline")
                    return "stopped"
                self.evaluate(k)
            if k == n:
                break
            if must_stop(2 * (self.timing["step_s"] or 150)):
                self.log(f"stop before step {k}: deadline")
                return "stopped"
            self.step(k)
            k += 1
        return "done"

    def finalize(self):
        n, d = self.cfg["steps"], self.d
        steps = [json.loads(l) for k in range(n) for l in open(d / "steps" / f"t{k:03d}.jsonl", encoding="utf-8")]
        ks = sorted({0, n} | set(range(0, n + 1, self.cfg["eval_every"])))
        for out, src in (("steps.jsonl", [d / "steps" / f"t{k:03d}.jsonl" for k in range(n)]),
                         ("step_texts.jsonl", [d / "steps" / f"t{k:03d}_texts.jsonl" for k in range(n)]),
                         ("evals.jsonl", [d / "evals" / f"t{k:03d}.jsonl" for k in ks]),
                         ("eval_texts.jsonl", [d / "evals" / f"t{k:03d}_texts.jsonl" for k in ks])):
            with open(d / out, "w", encoding="utf-8") as f:
                for p in src:
                    f.write(p.read_text(encoding="utf-8"))
        ev = {k: json.loads((d / "evals" / f"t{k:03d}.json").read_text()) for k in ks}
        summary = {str(k): ev[k]["summary"] for k in ks}
        (d / "eval_summary.json").write_text(json.dumps(summary, indent=1))
        ups = [r for r in steps if r["update"]]
        applied = [r["update"] for r in ups if r["update"].get("applied")]
        final_norm = {r["traj"]: r["total_norm"] for r in steps if r["step"] == n - 1 and "total_norm" in r}
        metrics = {
            "model": self.cfg["model"], "split": self.cfg["split"], "steps": n, "trajectories": self.trajs,
            "zero_step_a0_identical": all(len({r["a0_sha256"] for r in steps if r["step"] == 0 and r["seed"] == s}) == 1
                                          for s in self.cfg["seeds"]),
            "zero_step_verbal_identical": all(len({repr(r["s"]) for r in steps if r["step"] == 0 and r["seed"] == s}) == 1
                                              for s in self.cfg["seeds"]),
            "realized_increment_norm_max_rel_err": max((abs(u["N_after"] - self.cfg["tau"]) / self.cfg["tau"]
                                                        for u in applied), default=None),
            "anomalies": {t: dict(Counter(r["update"].get("anomaly") for r in ups if r["traj"] == t
                                          and r["update"].get("anomaly"))) for t in self.adapter_trajs},
            "final_total_norm": final_norm,
            "a0_status": {t: dict(Counter(r["a0_status"] for r in steps if r["traj"] == t)) for t in self.trajs},
            "step_gen_s": sum(r["gen_s"] for r in steps if r["traj"] == self.trajs[0]),
            "step_hf_s": sum(r["hf_s"] for r in steps if r["traj"] == self.trajs[0]),
            "eval_gen_s": sum(e["gen_s"] for e in ev.values()), "eval_hf_s": sum(e["hf_s"] for e in ev.values()),
            "scorer_guarded": scorer.GUARDED}
        (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
        arts = {}
        for cd in sorted((d / "checkpoints").glob("t*")):
            for p in sorted(cd.iterdir()):
                arts[f"checkpoints/{cd.name}/{p.name}"] = runlib.sha256_file(p)
        (d / "artifacts.json").write_text(json.dumps({
            "large_artifacts_on_run_host": arts,
            "rebuild": "B of every adapter trajectory and every context log at each evaluated step; A from "
                       "update.reset(seed h('stream-lora-A-v1', s)); config.json gives every other input"}, indent=1))
        shutil.rmtree(d / "serve", ignore_errors=True)
        return metrics


# ---- entry point -----------------------------------------------------------------------------------------------

def probe_path(p):
    return p if os.path.isabs(p) else str(REPO / p)


def build_config(args, S):
    t2 = json.loads(TIER2.read_text())["exp5"][args.model]
    t_star = args.self_threshold_s if args.self_threshold_s is not None else \
        json.loads(SELFVERDICT.read_text())[args.model]["t_star_s"]
    probe = args.probe or PROBES[args.model][0]
    probe_sha = runlib.sha256_file(probe_path(probe))
    if args.probe is None and probe_sha != PROBES[args.model][1]:
        raise SystemExit(f"probe sha256 {probe_sha} is not the frozen one (prereg §6)")
    return {"model": args.model, "split": args.split, "tag": args.tag, "backend": args.backend,
            "conditions": args.conditions, "seeds": args.seeds, "steps": args.steps, "eval_every": args.eval_every,
            "n_eval": args.n_eval, "memory_tokens": args.memory_tokens, "max_tokens": args.max_tokens,
            "n_samples": N_SAMPLES, "t_sample": T_SAMPLE, "train_steps": STEPS, "tau": float(t2["tau"]),
            "lr": float(t2["lr"]), "tier2": t2, "t_star": float(t_star), "probe": probe, "probe_sha256": probe_sha,
            "probe_threshold": PROBE_THRESHOLD, "gpu_mem": args.gpu_mem, "max_loras": args.max_loras,
            "max_model_len": args.max_model_len, "scorer_version": SCORER_VERSION,
            "stream_items_sha256": runlib.sha256_text(json.dumps(S["pool"])),
            "eval_items_sha256": runlib.sha256_text(json.dumps(S["eval"])),
            "orders_sha256": runlib.sha256_text(json.dumps({str(s): o for s, o in S["orders"].items()})),
            "wording_sha256": runlib.sha256_text(json.dumps([R.PROMPT, MEMORY_HEADER, MEMORY_ENTRY, MEMORY_SEP])),
            "seeds_scheme": ["stream-order-v1", "stream-lora-A-v1", "stream-random-v1", "stream-sample-v1"]
            + (["stream-matched-v1"] if "matched" in args.conditions else []),
            **matched_config(args)}


def matched_config(args):
    """The matched condition's verdicts: per order seed, the reference run's self-stream verdicts in step order,
    permuted by h("stream-matched-v1", seed) (Python random.Random, keyed; no global RNG)."""
    if "matched" not in args.conditions:
        return {}
    import random
    ref = probe_path(args.matched_ref)
    rows = [r for r in jl(ref) if r["condition"] == "self"]
    out = {}
    for s in args.seeds:
        vs = [bool(r["verdict"]) for r in sorted((r for r in rows if r["seed"] == s), key=lambda r: r["step"])]
        if len(vs) < args.steps:
            raise SystemExit(f"matched: the reference self stream of seed {s} has {len(vs)} steps < {args.steps}")
        vs = vs[:args.steps]
        random.Random(seed_int("stream-matched-v1", s)).shuffle(vs)
        out[str(s)] = vs
    return {"matched_ref": args.matched_ref, "matched_ref_sha256": runlib.sha256_file(ref), "matched_verdicts": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    ap.add_argument("--conditions", nargs="+", default=BASE_CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--matched-ref", default=MATCHED_REF, help="matched: the reference run's steps.jsonl (its self stream)")
    ap.add_argument("--memory-tokens", type=int, default=MEMORY_TOKENS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--self-threshold-s", type=float, default=None, help="default: analysis/out/selfverdict.json t*")
    ap.add_argument("--probe", default=None, help="default: the frozen probe of prereg §6 (sha256 checked)")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--gpu-mem", type=float, default=0.6, help="vLLM gpu_memory_utilization (HF shares the GPU)")
    ap.add_argument("--max-loras", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--stop-before-epoch", type=float, default=None, help="stop cleanly at a step boundary before this")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.conditions = [c for c in CONDITIONS if c in args.conditions]
    if args.split == "test" and not args.allow_test and not args.dry_run:
        raise SystemExit("refusing the test split without --allow-test")
    S = select_items(args.split, args.steps, args.n_eval, args.seeds)
    cfg = build_config(args, S)
    if args.dry_run:
        meta = S["meta"]
        print(json.dumps({"stream_items": len(S["pool"]), "stream_datasets": Counter(meta[i]["dataset"] for i in S["pool"]),
                          "eval_items": len(S["eval"]), "eval_datasets": Counter(meta[i]["dataset"] for i in S["eval"]),
                          "trajectories": len(args.seeds) * len(args.conditions),
                          "evaluations": len(set(range(0, args.steps + 1, args.eval_every)) | {args.steps}),
                          "a0_generations": args.steps * len(args.seeds) * len(args.conditions),
                          "eval_generations": (len(set(range(args.eval_every, args.steps + 1, args.eval_every)) | {args.steps})
                                               * len(args.seeds) * len(args.conditions) + 1) * args.n_eval * (1 + N_SAMPLES),
                          "order_heads": {s: o[:3] for s, o in S["orders"].items()},
                          "config_sha256": runlib.sha256_text(json.dumps(cfg, sort_keys=True)),
                          **{k: cfg[k] for k in ("tau", "lr", "t_star", "probe_sha256", "stream_items_sha256",
                                                 "eval_items_sha256")}}, indent=1, default=str))
        return

    kind = f"streams-{args.split}" + (f"-{args.tag}" if args.tag else "")
    run_id, d = runlib.run_dir("main" if args.split == "test" else "p2", args.model, kind)
    if (d / "manifest.json").exists():
        print(f"{run_id}: already finalized")
        return
    cfg_sha = runlib.sha256_text(json.dumps(cfg, sort_keys=True))
    if (d / "config.json").exists():
        old = json.loads((d / "config.json").read_text())
        if runlib.sha256_text(json.dumps(old, sort_keys=True)) != cfg_sha:
            raise SystemExit(f"{run_id}: config differs from the one this run started with; refusing to resume")
    else:
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
    resume_k = json.loads((d / "state" / "step.json").read_text())["k"] if (d / "state" / "step.json").exists() else 0
    runlib.register(run_id, "running", block=f"{args.model}-streams", exp="streams", model=args.model,
                    config_sha256=cfg_sha, resume_step=resume_k)

    gen = None
    if args.backend == "vllm":
        runlib.require_batch_invariant()
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") == "0":
            raise SystemExit("the vLLM engine must run in its own process (HF sets deterministic algorithms here)")
        gen = VllmGen(args.model, args.gpu_mem, args.max_loras, args.max_model_len)   # before HF takes GPU memory
    U.deterministic()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    base = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation=U.ATTN)
    pm, layers, _ = U.attach(base)
    pm.eval()
    if gen is None:
        gen = HfGen()
    st = Stream(S=S, cfg=cfg, tok=tok, pm=pm, layers=layers, gen=gen, d=d, probe_path=probe_path(cfg["probe"]),
                commit=runlib.code_commit(), log=lambda m: print(m, flush=True))
    if isinstance(gen, HfGen):
        gen.r = st
    status = st.run(stop_before_epoch=args.stop_before_epoch)
    if status != "done":
        runlib.register(run_id, "stopped-budget", block=f"{args.model}-streams", exp="streams", model=args.model)
        print(json.dumps({"run_id": run_id, "status": status}))
        return
    metrics = st.finalize()
    backend = {"name": "vllm+hf-transformers/peft" if args.backend == "vllm" else "hf-transformers/peft",
               "version": transformers.__version__}
    if args.backend == "vllm":
        import vllm
        backend["vllm"] = vllm.__version__
    runlib.write_manifest(d, run_id, args.model, tok, cfg, backend,
                          {"memory_header": MEMORY_HEADER, "memory_entry": MEMORY_ENTRY, "prompt": R.PROMPT,
                           "verbal_query": st.query})
    runlib.register(run_id, "done", block=f"{args.model}-streams", exp="streams", model=args.model)
    print(json.dumps({"run_id": run_id, "status": "done",
                      **{k: metrics[k] for k in ("realized_increment_norm_max_rel_err", "anomalies", "final_total_norm")}},
                     indent=1))


if __name__ == "__main__":
    main()
