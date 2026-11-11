"""Exp 5 dev pilot, HF stage (design §6, Exp 5; §9): train adapters for the tuning grid and the feasibility
groundings; measure KL on the KL probe set; check the reset and realized-norm invariants.

Selection rules, fixed before any pilot outcome is seen (DECISIONS 2026-11-03):
- tau: the largest grid value whose mean per-token KL(base || updated) on the KL probe set is <= KAPPA;
  if none qualifies, the smallest grid value.
- lr: chosen afterwards in analysis from the vLLM stage, by the ext-1.0 parameter-arm accuracy gain over B1 on the
  tuning items (the feasibility positive control, not the H4a contrast); ties -> the smaller lr.
Groundings at the selected tau (both lr values): self (thresholded verbal verdict, --self-threshold-s), perm-self (k = 0), shuffled
(self-verdicts permuted within dataset, ignoring correctness), random (matched norm), plus one zero adapter.

Output: <run>/adapters/<name>/ (PEFT, fp32), adapters.jsonl, kl.jsonl, updates.jsonl, metrics.json.
Usage: python -m src.ttt.pilot --model 4B --initial <p1 initial.jsonl> --verbal <p1 verbal.jsonl>
"""
import argparse
import json
import random
import time
from collections import defaultdict

from ..data.build import h
from ..data.verifiers import permute_self_verdicts, self_verdict
from ..runlib import MODELS, REPO, register, run_dir, sha256_text, write_manifest
from . import update as U

LR_GRID = (1e-4, 1e-3)
TAU_GRID = (0.5, 2.0, 8.0)
KAPPA = 0.05          # nats per token
STEPS = 4
N_TUNE, N_KL_UPDATES = 60, 8
N_REPEAT = 5          # updates trained twice for the determinism invariant


def kl_on_probe_set(pm, tok, kl_items):
    """Mean per-token KL(base || adapted) over the a0 tokens of the KL probe items (teacher-forced)."""
    import torch
    tot, n = 0.0, 0
    with torch.no_grad():
        for ids, start in kl_items:
            x = torch.tensor([ids], device="cuda")
            la = torch.log_softmax(pm(input_ids=x).logits[0, start - 1:-1].float(), -1)
            with pm.disable_adapter():
                lb = torch.log_softmax(pm(input_ids=x).logits[0, start - 1:-1].float(), -1)
            tot += float((lb.exp() * (lb - la)).sum())
            n += lb.shape[0]
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--verbal", required=True)
    ap.add_argument("--self-threshold-s", type=float, default=0.0,
                    help="self-verdict threshold on log P(yes) - log P(no); 0 = P(yes)_norm >= 0.5 (design 5.7)")
    args = ap.parse_args()
    cfg = {**vars(args), "lr_grid": LR_GRID, "tau_grid": TAU_GRID, "kappa": KAPPA, "steps": STEPS,
           "n_tune": N_TUNE, "rank": U.RANK, "alpha": U.ALPHA, "targets": U.TARGETS}

    U.deterministic()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    P = REPO / "data" / "processed"
    dnat = {json.loads(l)["item_id"]: json.loads(l) for l in open(P / "dnat.jsonl", encoding="utf-8")}
    split = json.loads((P / "split.json").read_text())
    kl_ids = set(split["kl_dev"])
    init = [json.loads(l) for l in open(args.initial, encoding="utf-8")]
    verbal = {json.loads(l)["item_id"]: json.loads(l) for l in open(args.verbal, encoding="utf-8")}
    dev = [r for r in init if r["split"] == "dev"]
    pool = [r for r in dev if dnat[r["item_id"]]["canonical_id"] not in kl_ids]
    wrong = [r for r in pool if not r["correct"]]
    right = sorted((r for r in pool if r["correct"]), key=lambda r: h("ttt-tune-v1", r["item_id"]))
    tune = (wrong + right)[:N_TUNE] if len(wrong) < N_TUNE else wrong[:N_TUNE]
    kl_rows = [r for r in dev if dnat[r["item_id"]]["canonical_id"] in kl_ids]

    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    base = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda",
                                                attn_implementation=U.ATTN)
    pm, layers, params = U.attach(base)
    pm.eval()

    seqs = {r["item_id"]: U.training_sequence(tok, dnat[r["item_id"]]["question"], r["a0_token_ids"]) for r in tune + kl_rows}
    kl_items = []
    for r in kl_rows:
        ids, _ = seqs[r["item_id"]]
        kl_items.append((ids, len(ids) - len(r["a0_token_ids"])))

    run_id, d = run_dir("p2", args.model, "ttt-pilot")
    register(run_id, "running", block="ttt-pilot", exp="exp5", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    (d / "adapters").mkdir(exist_ok=True)
    adapters, kls, logs = [], [], []

    def save(name, item, grounding, lr, tau, verdict, log):
        path = d / "adapters" / name
        pm.save_pretrained(str(path))
        adapters.append({"name": name, "item_id": item, "grounding": grounding, "lr": lr, "tau": tau,
                         "verdict": verdict, "applied": log.get("applied", True), "path": str(path)})
        logs.append({"name": name, "tau": tau, **log})

    t0 = time.time()
    # Round A: the tuning grid with the truthful verdict (ext-1.0).
    kl_by_cfg = defaultdict(list)
    for lr in LR_GRID:
        for tau in TAU_GRID:
            for k, r in enumerate(tune):
                ids, mask = seqs[r["item_id"]]
                log = U.update(pm, layers, params, ids, mask, r["correct"], lr=lr, steps=STEPS, tau=tau,
                               seed=int(h("ttt-seed-v1", r["item_id"])[:8], 16))
                if k < N_KL_UPDATES and log["applied"]:
                    kl = kl_on_probe_set(pm, tok, kl_items)
                    kl_by_cfg[(lr, tau)].append(kl)
                    kls.append({"lr": lr, "tau": tau, "item_id": r["item_id"], "kl": kl})
                save(f"ext_lr{lr:g}_tau{tau:g}_{r['item_id']}", r["item_id"], "ext-1.0", lr, tau, r["correct"], log)
    mean_kl = {f"{lr:g}/{tau:g}": sum(v) / len(v) for (lr, tau), v in kl_by_cfg.items() if v}
    ok = [tau for tau in TAU_GRID if all(sum(kl_by_cfg[(lr, tau)]) / max(1, len(kl_by_cfg[(lr, tau)])) <= KAPPA
                                         for lr in LR_GRID)]
    tau_sel = max(ok) if ok else min(TAU_GRID)

    # Reset invariant: B = 0 gives exactly the base logits.
    U.reset(layers, 0)
    import torch as _t
    with _t.no_grad():
        x = _t.tensor([kl_items[0][0]], device="cuda")
        l_reset = pm(input_ids=x).logits
        with pm.disable_adapter():
            l_base = pm(input_ids=x).logits
    reset_exact = bool(_t.equal(l_reset, l_base))
    pm.save_pretrained(str(d / "adapters" / "zero"))
    adapters.append({"name": "zero", "item_id": None, "grounding": "zero", "lr": None, "tau": 0.0, "verdict": None,
                     "applied": True, "path": str(d / "adapters" / "zero")})

    # Round B: feasibility groundings at the selected tau, both lr values.
    self_v = {r["item_id"]: self_verdict(verbal[r["item_id"]], args.self_threshold_s) for r in dev}
    perm = permute_self_verdicts([{"item_id": r["item_id"], "dataset": r["dataset"], "initially_correct": r["correct"],
                                   "self_says_correct": self_v[r["item_id"]]} for r in dev])
    by_ds = defaultdict(list)
    for r in dev:
        by_ds[r["dataset"]].append(r["item_id"])
    shuffled = {}
    for ds, ids_ in by_ds.items():
        ids_ = sorted(ids_)
        vs = [self_v[i] for i in ids_]
        random.Random(h("shuffled-v1", ds)).shuffle(vs)
        shuffled.update(zip(ids_, vs))
    for lr in LR_GRID:
        for r in tune:
            ids, mask = seqs[r["item_id"]]
            seed = int(h("ttt-seed-v1", r["item_id"])[:8], 16)
            for g, v in (("self", self_v[r["item_id"]]), ("perm-self", perm[r["item_id"]][0]),
                         ("shuffled", shuffled[r["item_id"]])):
                log = U.update(pm, layers, params, ids, mask, v, lr=lr, steps=STEPS, tau=tau_sel, seed=seed)
                save(f"{g}_lr{lr:g}_tau{tau_sel:g}_{r['item_id']}", r["item_id"], g, lr, tau_sel, v, log)
    for r in tune:
        log = U.random_update(layers, tau_sel, seed=int(h("ttt-random-v1", r["item_id"])[:8], 16))
        save(f"random_tau{tau_sel:g}_{r['item_id']}", r["item_id"], "random", None, tau_sel, None, log)
    # Invariant: the same update twice gives a bitwise-identical adapter (P2: it did not, before U.deterministic()).
    repeat = []
    for r in [r for r in tune if seqs[r["item_id"]][1] is not None][:N_REPEAT]:
        ids, mask = seqs[r["item_id"]]
        seed = int(h("ttt-seed-v1", r["item_id"])[:8], 16)
        states = []
        for _ in range(2):
            U.update(pm, layers, params, ids, mask, not r["correct"], lr=LR_GRID[-1], steps=STEPS, tau=tau_sel, seed=seed)
            states.append(U.adapter_state(layers))
        repeat.append(bool(_t.equal(*states)))
    U.reset(layers, 0)

    applied = [l for l in logs if l.get("applied") and l.get("N_after") is not None and l["tau"]]
    metrics = {"model": args.model, "n_tune": len(tune), "n_tune_wrong": sum(not r["correct"] for r in tune),
               "n_kl_items": len(kl_items), "mean_kl": mean_kl, "tau_selected": tau_sel, "reset_logits_exact": reset_exact,
               "realized_norm_max_rel_err": max(abs(l["N_after"] - l["tau"]) / l["tau"] for l in applied),
               "update_repeat_identical": sum(repeat) / len(repeat) if repeat else None,
               "anomalies": dict(sorted(((k, sum(l.get("anomaly") == k for l in logs)) for k in
                                         {l.get("anomaly") for l in logs} - {None}))),
               "n_adapters": len(adapters), "wall_s": time.time() - t0, "transformers": transformers.__version__}
    for name, rows in (("adapters.jsonl", adapters), ("kl.jsonl", kls), ("updates.jsonl", logs)):
        with open(d / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers+peft", "version": transformers.__version__}, {})
    register(run_id, "done", block="ttt-pilot", exp="exp5", model=args.model)
    print(json.dumps({k: metrics[k] for k in ("n_tune", "n_tune_wrong", "mean_kl", "tau_selected", "reset_logits_exact",
                                              "realized_norm_max_rel_err", "update_repeat_identical", "anomalies", "n_adapters",
                                              "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
