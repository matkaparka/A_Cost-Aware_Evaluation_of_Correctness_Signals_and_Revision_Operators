"""Exp 5 update rule (design §6, Exp 5), shared by the dev pilot and the main runs.

- Training sequence: the chat-rendered (user: q, assistant: a0), the same template as the q-only re-answer.
  The verdict v is not in the sequence; it only selects the loss.
- Loss mask: the final-answer span of a0 (its last \\boxed{...}); mean over masked tokens.
  v = correct: NLL.  v = incorrect: unlikelihood -log(1 - p), computed exactly as
  -(logsumexp over the other tokens - logsumexp over all), so it keeps its gradient as p -> 1. (Before P2's fix the
  form -log1p(-exp(min(log p, -1e-6))) had zero gradient wherever p > 1 - 1e-6: no update on confident answers.)
- LoRA rank 8 on q, k, v, o (A random, seeded; B = 0); LoRA in fp32 on a bf16 base; fresh AdamW per item; k steps.
- Global norm N = sqrt(sum_m ||(alpha/r) B_m A_m||_F^2); after the k steps every B_m is multiplied by c = tau / N.
- Anomalies (frozen in advance): no answer span, N = 0, or NaN/Inf -> no update (intention-to-treat), counted;
  c > 100 -> applied and flagged.
"""
import math

from ..gen.render import PROMPT
from ..score.scorer import _boxed_spans

RANK, ALPHA = 8, 16
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTN = "sdpa"    # with deterministic() on, SDPA's backward is bitwise reproducible (checked on the laptop, 2026-11-04)


def deterministic():
    """Same inputs and seed -> bitwise-identical adapter. Call before CUDA is first used.

    P2 found that repeated updates of the same item, verdict, lr, tau and seed differed (116 of 122 on 4B, pre-rescale
    norms up to ~1.5% apart; 12 of 430 served cells changed correctness): a non-deterministic attention backward,
    amplified by AdamW, whose per-element normalization turns tiny gradient differences into full-size steps.
    """
    import os

    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def adapter_state(layers):
    """All B matrices, flattened (A is fixed by the seed), for bitwise comparison."""
    import torch
    return torch.cat([m.lora_B["default"].weight.detach().flatten() for m in layers]).clone()


def attach(model):
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer
    pm = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, target_modules=TARGETS))
    layers = [m for m in pm.modules() if isinstance(m, LoraLayer)]
    params = [p for p in pm.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in params), "LoRA parameters must be fp32"
    return pm, layers, params


def reset(layers, seed):
    """A ~ kaiming-uniform (seeded), B = 0: the adapter is exactly the zero update."""
    import torch
    torch.manual_seed(seed)
    for m in layers:
        m.reset_lora_parameters("default", True)


def zero_b(layers):
    import torch
    with torch.no_grad():
        for m in layers:
            m.lora_B["default"].weight.zero_()


def global_norm(layers):
    import torch
    tot = 0.0
    with torch.no_grad():
        for m in layers:
            dw = m.scaling["default"] * (m.lora_B["default"].weight @ m.lora_A["default"].weight)
            tot += float(dw.double().pow(2).sum())
    return math.sqrt(tot)


def rescale(layers, tau):
    import torch
    n0 = global_norm(layers)
    if not math.isfinite(n0) or n0 == 0:
        return n0, None
    c = tau / n0
    with torch.no_grad():
        for m in layers:
            m.lora_B["default"].weight.mul_(c)
    return n0, c


def training_sequence(tok, q, a0_token_ids):
    """Token IDs of (user: q, assistant: a0) with a0 as generated, and the answer-span mask over them."""
    prefix = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q)}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prefix, add_special_tokens=False).input_ids
    a0 = list(a0_token_ids)
    ids = p_ids + a0
    text_a0 = tok.decode(a0)
    boxes = _boxed_spans(text_a0)
    if not boxes:
        return ids, None
    s, e, _ = boxes[-1]                      # the last \boxed{...}, from "\boxed{" to its closing brace
    # Exact char extent of each generated token from cumulative decoding (no re-tokenization of a0).
    ends = [len(tok.decode(a0[:k + 1])) for k in range(len(a0))]
    starts = [0] + ends[:-1]
    special = set(tok.all_special_ids)
    mask = [0] * len(p_ids) + [1 if (st < e and en > s and t not in special) else 0
                               for t, st, en in zip(a0, starts, ends)]
    return ids, mask if any(mask) else None


def answer_loss(logits, targets, verdict_correct):
    """Mean loss over the answer positions. logits: [n, V] fp32 at the masked positions; targets: [n]."""
    import torch
    if verdict_correct:
        return -torch.log_softmax(logits, -1).gather(-1, targets[:, None])[:, 0].mean()
    other = logits.scatter(-1, targets[:, None], float("-inf"))
    return -(torch.logsumexp(other, -1) - torch.logsumexp(logits, -1)).mean()


def update(pm, layers, params, ids, mask, verdict_correct, *, lr, steps, tau, seed):
    """One per-item update. Returns a log dict; on an anomaly the adapter is left at zero (no update)."""
    import torch
    reset(layers, seed)
    if mask is None:
        zero_b(layers)
        return {"anomaly": "no_answer_span", "applied": False}
    x = torch.tensor([ids], device="cuda")
    m = torch.tensor([mask[1:]], device="cuda", dtype=torch.bool)
    opt = torch.optim.AdamW(params, lr=lr)
    for _ in range(steps):
        loss = answer_loss(pm(input_ids=x).logits[:, :-1][m].float(), x[:, 1:][m], verdict_correct)
        if not torch.isfinite(loss):
            zero_b(layers)
            return {"anomaly": "nan_loss", "applied": False}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    n0, c = rescale(layers, tau)
    if c is None:
        zero_b(layers)
        return {"anomaly": "zero_or_nan_norm", "applied": False, "N_before": n0}
    return {"anomaly": "c_gt_100" if c > 100 else None, "applied": True, "N_before": n0, "c": c,
            "N_after": global_norm(layers), "final_loss": float(loss)}


def random_update(layers, tau, seed):
    """Matched-norm random control: Gaussian A and B per module (seeded), then global rescale to tau."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for m in layers:
            for w in (m.lora_A["default"].weight, m.lora_B["default"].weight):
                w.copy_(torch.randn(w.shape, generator=g, dtype=torch.float32).to(w.device))
    n0, c = rescale(layers, tau)
    return {"applied": True, "N_before": n0, "c": c, "N_after": global_norm(layers)}
