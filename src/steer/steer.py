"""Exp 3 steering primitives (design §6, Exp 3), HF backend.

resid_post of block L = hidden_states[L + 1] (hidden_states[0] is the embedding output).
Intervention: add alpha * d_hat * ||h|| to resid_post of block L at the reviewed-solution positions, during the
prefill forward only (later layers at those positions and the K/V that later positions attend to change; the last
prompt position is never edited). alpha = 0 adds exactly zero, so it reproduces the unsteered run.
"""
import contextlib

import torch

SKIP_FIRST = 5          # solution content tokens: the span minus its first 5 tokens


@contextlib.contextmanager
def steer_hook(model, layer, direction, alpha, positions_per_row):
    """positions_per_row: list (batch) of LongTensors of absolute positions to edit in the prefill forward."""
    block = model.model.layers[layer]
    d_hat = direction / direction.norm()

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1 or alpha == 0:          # decode step, or the alpha = 0 identity
            return out
        h = h.clone()
        for b, pos in enumerate(positions_per_row):
            if len(pos):
                v = h[b, pos]
                d = d_hat.to(device=v.device, dtype=v.dtype)   # directions are built on the CPU
                h[b, pos] = v + alpha * d * v.float().norm(dim=-1, keepdim=True).to(v.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def solution_positions(token_span, pad=0):
    a, b = token_span
    return torch.arange(pad + a + SKIP_FIRST, pad + b)


def mean_states(model, input_ids, span, layers):
    """Mean resid_post over the solution content tokens, for each requested block (single sequence)."""
    a, b = span
    if b - a <= SKIP_FIRST:   # an empty slice would average to NaN (P2: 16 scli5 items made every direction NaN)
        raise ValueError(f"solution span {span} has no content tokens after the first {SKIP_FIRST}")
    with torch.no_grad():
        hs = model(input_ids=torch.tensor([input_ids], device=model.device), output_hidden_states=True).hidden_states
    return {L: hs[L + 1][0, a + SKIP_FIRST:b].float().mean(0).cpu() for L in layers}


def state_at(model, input_ids, pos, layers):
    with torch.no_grad():
        hs = model(input_ids=torch.tensor([input_ids], device=model.device), output_hidden_states=True).hidden_states
    return {L: hs[L + 1][0, pos].float().cpu() for L in layers}


def next_token_logprobs(model, input_ids, layer=None, direction=None, alpha=0.0, positions=None):
    x = torch.tensor([input_ids], device=model.device)
    ctx = steer_hook(model, layer, direction, alpha, [positions]) if layer is not None else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        return torch.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)


def random_directions(hidden, seeds):
    out = []
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        out.append(torch.randn(hidden, generator=g))
    return out
