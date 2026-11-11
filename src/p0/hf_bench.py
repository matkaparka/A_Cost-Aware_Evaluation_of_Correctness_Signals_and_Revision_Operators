"""P0 HF benchmark: teacher-forced scoring, activation caching, TTT LoRA updates (design Exp 5 rule),
patching forward+backward cost, and HF-vs-vLLM greedy agreement. Reads round-1 items from a vLLM run.
Usage: python -m src.p0.hf_bench --model 4B --items /home/results/p0/4B/<vllm run>/items.jsonl [--ttt 32] [--agree 24]
"""
import argparse
import json
import math
import time

import torch

from .common import MODELS, PROMPT, render, run_dir, write_manifest


def sync():
    torch.cuda.synchronize()
    return time.time()


def answer_span_mask(tok, prompt, a0):
    """Token mask over the last \\boxed{...} of a0 (the final-answer span); None if there is none."""
    i = a0.rfind("\\boxed{")
    if i < 0:
        return None, None
    enc = tok(prompt + a0, return_offsets_mapping=True, add_special_tokens=False)
    lo, hi = len(prompt) + i, len(prompt) + len(a0)
    mask = [1 if (s < hi and e > lo) else 0 for s, e in enc["offset_mapping"]]
    return torch.tensor([enc["input_ids"]]), torch.tensor([mask], dtype=torch.bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--items", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ttt", type=int, default=0)
    ap.add_argument("--save-adapters", type=int, default=0)
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--patch", type=int, default=0)
    ap.add_argument("--agree", type=int, default=0)
    ap.add_argument("--hf-gen", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()
    cfg = vars(args)

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_id, d = run_dir("p0", args.model, "hf")
    m = {"model": args.model, "transformers": transformers.__version__, "torch": torch.__version__}
    items = [json.loads(line) for line in open(args.items)][: max(args.n, args.ttt, args.agree, args.hf_gen)]

    t = time.time()
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    m["load_s"] = sync() - t
    cfg_m = model.config
    m["arch"] = {"layers": cfg_m.num_hidden_layers, "hidden": cfg_m.hidden_size,
                 "kv_heads": cfg_m.num_key_value_heads, "head_dim": getattr(cfg_m, "head_dim", None)}
    prompts = [render(tok, [{"role": "user", "content": PROMPT.format(q=it["q"])}]) for it in items]

    # 1. Teacher-forced scoring of a0 (batched, right padding) + activation caching at the last a0 token.
    seqs = [p + it["a0"] for p, it in zip(prompts, items[: args.n])]
    tok.padding_side = "right"
    ntok, t = 0, sync()
    with torch.no_grad():
        for b in range(0, len(seqs), 16):
            enc = tok(seqs[b:b + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            logits = model(**enc).logits.float()
            logp = torch.log_softmax(logits[:, :-1], -1).gather(-1, enc.input_ids[:, 1:, None])
            ntok += int(enc.attention_mask.sum())
    m["teacher_forced"] = {"n": len(seqs), "tokens": ntok, "wall_s": sync() - t}
    m["teacher_forced"]["tok_per_s"] = ntok / m["teacher_forced"]["wall_s"]

    t = sync()
    with torch.no_grad():
        for s in seqs:
            enc = tok(s, return_tensors="pt", add_special_tokens=False).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states
            vec = torch.stack([h[0, -1] for h in hs]).to(torch.bfloat16).cpu()
    m["activations"] = {"n": len(seqs), "wall_s": sync() - t, "vectors_per_item": vec.shape[0],
                        "bytes_per_position_all_layers": vec.numel() * 2}
    m["activations"]["s_per_item"] = m["activations"]["wall_s"] / len(seqs)

    # 2. HF vs vLLM greedy agreement (zero adapter == base model here), batch size 1.
    if args.agree:
        same, t = 0, sync()
        with torch.no_grad():
            for p, it in zip(prompts[: args.agree], items[: args.agree]):
                enc = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
                out = model.generate(**enc, do_sample=False, max_new_tokens=it["a0_tokens"] + 1)
                same += tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip() == it["a0"].strip()
        m["hf_vllm_agreement"] = {"n": args.agree, "identical": same, "rate": same / args.agree, "wall_s": sync() - t}

    # 2b. HF batched greedy generation (Exp 3 steering cannot run in vLLM).
    if args.hf_gen:
        tok.padding_side = "left"
        ntok, t = 0, sync()
        with torch.no_grad():
            for b in range(0, args.hf_gen, 32):
                enc = tok(prompts[b:b + 32], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                out = model.generate(**enc, do_sample=False, max_new_tokens=512, pad_token_id=tok.pad_token_id)
                ntok += int((out[:, enc.input_ids.shape[1]:] != tok.pad_token_id).sum())
        wall = sync() - t
        m["hf_generate_batched"] = {"n": args.hf_gen, "batch": 32, "max_new_tokens": 512, "output_tokens": ntok,
                                    "wall_s": wall, "output_tok_per_s": ntok / wall}
        tok.padding_side = "right"

    # 3. Patching cost proxy: one forward with grad to the embeddings + one backward of a logit difference.
    if args.patch:
        t = sync()
        for s in seqs[: args.patch]:
            enc = tok(s, return_tensors="pt", add_special_tokens=False).to("cuda")
            emb = model.get_input_embeddings()(enc.input_ids).detach().requires_grad_(True)
            out = model(inputs_embeds=emb, output_hidden_states=True)
            (out.logits[0, -1, 0] - out.logits[0, -1, 1]).backward()
        wall = sync() - t
        m["patching"] = {"n": args.patch, "wall_s": wall, "s_per_pair_fwd_bwd": wall / args.patch}

    # 4. TTT updates: LoRA r=8 on q,k,v,o, fp32 adapters on a bf16 base, fresh AdamW, k steps, global-norm rescale.
    if args.ttt:
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora import LoraLayer
        pm = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        layers = [mod for mod in pm.modules() if isinstance(mod, LoraLayer)]
        params = [p for p in pm.parameters() if p.requires_grad]
        m["ttt_params"] = sum(p.numel() for p in params)
        m["ttt_param_dtype"] = str(params[0].dtype)

        def reset(seed):
            torch.manual_seed(seed)
            for mod in layers:
                mod.reset_lora_parameters("default", True)

        def global_norm():
            return math.sqrt(sum(float((mod.scaling["default"] * (mod.lora_B["default"].weight @ mod.lora_A["default"].weight)).float().pow(2).sum())
                                 for mod in layers))

        if args.adapter_dir:
            reset(0)
            pm.save_pretrained(f"{args.adapter_dir}/zero")
        times, logs, saved = [], [], []
        for k, (p, it) in enumerate(zip(prompts[: args.ttt], items[: args.ttt])):
            ids, mask = answer_span_mask(tok, p, it["a0"])
            if ids is None:
                logs.append({"id": it["id"], "skipped": "no answer span"})
                continue
            ids, mask = ids.cuda(), mask.cuda()
            t = sync()
            reset(1000 + k)
            opt = torch.optim.AdamW(params, lr=args.lr)
            for _ in range(args.steps):
                logits = pm(input_ids=ids).logits[:, :-1].float()
                logp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
                lp = logp[mask[:, 1:]]
                if it["correct0"]:
                    loss = -lp.mean()
                else:
                    loss = -torch.log1p(-torch.exp(lp.clamp(max=-1e-6))).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                n0 = global_norm()
                c = args.tau / n0 if n0 > 0 else float("nan")
                for mod in layers:
                    mod.lora_B["default"].weight.mul_(c)
                n1 = global_norm()
            times.append(sync() - t)
            logs.append({"id": it["id"], "N_before": n0, "c": c, "N_after": n1, "span_tokens": int(mask.sum())})
            if args.adapter_dir and len(saved) < args.save_adapters:
                t = time.time()
                pm.save_pretrained(f"{args.adapter_dir}/ttt{k:03d}")
                saved.append(time.time() - t)
        # Reset check: B = 0 must give exactly the base logits.
        with torch.no_grad():
            enc = tok(seqs[0], return_tensors="pt", add_special_tokens=False).to("cuda")
            reset(0)
            l_reset = pm(**enc).logits
            with pm.disable_adapter():
                l_base = pm(**enc).logits
        m["ttt"] = {"n": len(times), "steps": args.steps, "s_per_update_mean": sum(times) / len(times),
                    "s_per_update_max": max(times), "reset_logits_equal": bool(torch.equal(l_reset, l_base)),
                    "realized_norm_max_rel_err": max(abs(x["N_after"] - args.tau) / args.tau for x in logs if "N_after" in x),
                    "save_s_mean": (sum(saved) / len(saved)) if saved else None}
        with open(d / "ttt_log.jsonl", "w") as f:
            for x in logs:
                f.write(json.dumps(x) + "\n")

    m["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    (d / "metrics.json").write_text(json.dumps(m, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__})
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
