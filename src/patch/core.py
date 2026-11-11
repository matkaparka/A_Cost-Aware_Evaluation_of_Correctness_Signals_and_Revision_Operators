"""Exp 4 activation-patching primitives (design Exp 4), HF backend, plain PyTorch forward hooks as in src/steer/steer.py.
Written after the freeze; the choices below are listed for DEVIATIONS before any test output exists.

Pairs. A D-inj item rendered by src/gen/render.review_prompt at F0 in two Exp 1B cells with identical solution text:
receiver R = (assistant, self), donor D = (user, other). A patch writes D's activations into R's forward, i.e. it moves
the own-error cell toward the external configuration (the sign convention of Exp 3, "positive alpha = toward external").

Components. (layer L, position group g). The activation is resid_post of block L: the output of model.model.layers[L]
(= hidden_states[L + 1] of src/steer/steer.py for L < n_layers - 1; the hook avoids the final-norm ambiguity of the last
hidden state). Position groups cover the prompt from the source label to its end, in order:
  label     the source label (render span "label")
  turn      label end -> solution start: "<|im_end|>\\n<|im_start|>{role}\\n" (holds the role token)
  sol_head  the first SKIP_FIRST = 5 solution tokens (src/steer/steer.py)
  sol_body  the rest of the solution (Exp 3's steering span)
  post      solution end -> instruction start
  instr     the revision instruction
  gen       the generation prompt after the instruction, without its last token
  last      the last prompt token (the marker proxy is read here; the target starts after it)
Tokens before the label (chat header, question) precede the first token that differs between R and D, so they have the
same states in both runs and are not a group. The target tokens are the measuring device and are never patched.
Alignment within a group (`align`): the common token prefix left-aligned, the common token suffix right-aligned, the
differing middle right-aligned; middle tokens of the longer side without a partner are not patched. For the main
labels the middle is " your own" (R) vs " another model's" (D): " your"<-" model", " own"<-"'s"; D's " another" is
unpaired. Every group except label and turn must have identical token IDs in R and D (checked in `build_pair`).

Metrics (one forward gives both; teacher-forced sequence = prompt token IDs + TARGET token IDs, tokenized apart as in
generation):
  lp_gold  sum of the log-probs of the gold-answer tokens of TARGET = "The final answer is \\boxed{<gold>}.", placed at
           the start of the revision turn (tokens overlapping the gold's characters)
  marker   logsumexp(logits[correction markers]) - logsumexp(logits[commit markers]) at the last prompt position, with
           the frozen Exp 3 markers (prereg §6: correction ['Let'], commit ['The']) = logit(Let) - logit(The)
Attribution patching (`attribute`): attr[m, L, g] = sum over the paired positions p <- q of g of
(h_D[L, q] - h_R[L, p]) . dm/dh_R[L, p], from one forward with gradients on R (inputs_embeds path) and one forward on D,
in fp32 from bf16 states. Real patching (`patch_pair`): batch-1 forwards without gradients, one per component.
Self-patch invariant: writing R's own cached states into R at every patched component at once reproduces R's metrics
within SELF_PATCH_TOL (batch-1 forwards of the same sequence: expected exactly 0).
"""
from dataclasses import dataclass, field

import torch

from ..gen import render as R
from ..steer.steer import SKIP_FIRST

RECEIVER = ("assistant", "self")
DONOR = ("user", "other")
GROUPS = ("label", "turn", "sol_head", "sol_body", "post", "instr", "gen", "last")
SAME_TOKENS = ("sol_head", "sol_body", "post", "instr", "gen", "last")
METRICS = ("lp_gold", "marker")
TARGET_HEAD, TARGET_TAIL = "The final answer is \\boxed{", "}."
MARKERS = {"correction": ["Let"], "commit": ["The"]}   # prereg §6 (results/p2/4B/p2-steer-pilot-4B-7ea8f7d/metrics.json)
SELF_PATCH_TOL = 1e-3   # nats (lp_gold) and logits (marker), absolute, per pair


class NoContent(ValueError):
    """The solution has no tokens after the first SKIP_FIRST (excluded, as in every Exp 3 stage; DECISIONS 2026-11-04)."""


@dataclass
class Side:
    ids: list            # prompt + target token IDs
    P: int               # prompt length
    gold_pos: list       # absolute positions (in ids) of the gold-answer tokens of the target
    ranges: dict         # group -> (start, end) token range in the prompt


@dataclass
class Pair:
    item_id: str
    canonical_id: str
    dataset: str
    r: Side
    d: Side
    groups: dict = field(default_factory=dict)     # group -> (receiver positions, donor positions), paired
    unpaired: dict = field(default_factory=dict)   # group -> (receiver tokens unpaired, donor tokens unpaired)


def target(tok, gold):
    """TARGET token IDs and the target-relative indices of the tokens overlapping the gold's characters."""
    text = TARGET_HEAD + gold + TARGET_TAIL
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    a, b = len(TARGET_HEAD), len(TARGET_HEAD) + len(gold)
    pos = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < b and e > a]
    if not pos:
        raise ValueError(f"empty gold span for {gold!r}")
    return enc["input_ids"], pos


def side(tok, item, cell, tgt_ids, tgt_pos):
    rr = R.review_prompt(tok, item["question"], item["solution_with_answer"], role=cell[0], label=cell[1])
    ids, sp = R.token_spans(tok, rr)
    P = len(ids)
    (la, lb), (sa, sb), (ia, ib) = sp["label"], sp["solution"], sp["instruction"]
    if sb - sa <= SKIP_FIRST:
        raise NoContent(item["item_id"])
    rng = {"label": (la, lb), "turn": (lb, sa), "sol_head": (sa, sa + SKIP_FIRST), "sol_body": (sa + SKIP_FIRST, sb),
           "post": (sb, ia), "instr": (ia, ib), "gen": (ib, P - 1), "last": (P - 1, P)}
    bounds = [rng[g] for g in GROUPS]
    if not (0 < la and all(a < b for a, b in bounds) and all(bounds[k][1] == bounds[k + 1][0] for k in range(len(GROUPS) - 1))
            and bounds[-1][1] == P):
        raise ValueError(f"{item['item_id']}: position groups not contiguous and non-empty: {rng}")
    return Side(ids=ids + tgt_ids, P=P, gold_pos=[P + i for i in tgt_pos], ranges=rng)


def align(r_ids, d_ids):
    """Pairs (i, j) of group-relative indices: common prefix left, common suffix right, differing middle right-aligned."""
    n, m = len(r_ids), len(d_ids)
    pre = 0
    while pre < min(n, m) and r_ids[pre] == d_ids[pre]:
        pre += 1
    suf = 0
    while suf < min(n, m) - pre and r_ids[n - 1 - suf] == d_ids[m - 1 - suf]:
        suf += 1
    mr, md = n - pre - suf, m - pre - suf
    k = min(mr, md)
    pairs = [(i, i) for i in range(pre)]
    pairs += [(pre + mr - k + j, pre + md - k + j) for j in range(k)]
    pairs += [(n - suf + j, m - suf + j) for j in range(suf)]
    return pairs


def build_pair(tok, item):
    tgt_ids, tgt_pos = target(tok, item["gold"])
    r, d = side(tok, item, RECEIVER, tgt_ids, tgt_pos), side(tok, item, DONOR, tgt_ids, tgt_pos)
    la = r.ranges["label"][0]
    if d.ranges["label"][0] != la or r.ids[:la] != d.ids[:la]:
        raise ValueError(f"{item['item_id']}: receiver and donor differ before the label")
    pair = Pair(item["item_id"], item["canonical_id"], item["dataset"], r, d)
    for g in GROUPS:
        (ra, rb), (da, db) = r.ranges[g], d.ranges[g]
        rt, dt = r.ids[ra:rb], d.ids[da:db]
        if g in SAME_TOKENS and rt != dt:
            raise ValueError(f"{item['item_id']}: group {g} has different tokens in receiver and donor")
        pp = align(rt, dt)
        pair.groups[g] = ([ra + i for i, _ in pp], [da + j for _, j in pp])
        pair.unpaired[g] = (len(rt) - len(pp), len(dt) - len(pp))
    return pair


def marker_ids(tok):
    out = {}
    for k, words in MARKERS.items():
        ids = []
        for w in words:
            t = tok(w, add_special_tokens=False).input_ids
            if len(t) != 1 or tok.decode(t) != w:
                raise ValueError(f"marker {w!r} is not a single token")
            ids.append(t[0])
        out[k] = ids
    if set(out["correction"]) & set(out["commit"]):
        raise ValueError("marker sets overlap")
    return out


def _h(out):
    return out[0] if isinstance(out, tuple) else out


def _wrap(out, h):
    return (h, *out[1:]) if isinstance(out, tuple) else h


def run(model, ids, *, capture=(), patches=None, grad=False):
    """One batch-1 forward of `ids`. capture: block indices whose outputs are returned ({L: [1, T, d]}; with grad=True
    they stay in the graph). patches: {L: (positions, values [n, d])} written into block L's output.
    Returns (final-norm hidden states [1, T, d], captured)."""
    layers = model.model.layers
    cache, handles = {}, []

    def cap(L):
        def hook(_m, _i, out):
            cache[L] = _h(out) if grad else _h(out).detach()
        return hook

    def put(pos, vals):
        def hook(_m, _i, out):
            h = _h(out).clone()
            h[0, pos] = vals.to(device=h.device, dtype=h.dtype)
            return _wrap(out, h)
        return hook

    try:
        for L, (pos, vals) in (patches or {}).items():
            handles.append(layers[L].register_forward_hook(put(torch.tensor(pos, dtype=torch.long, device=model.device), vals)))
        for L in capture:   # registered after the patches: a captured block output includes its patch
            handles.append(layers[L].register_forward_hook(cap(L)))
        x = torch.tensor([ids], device=model.device)
        if grad:
            emb = model.model.embed_tokens(x).detach().requires_grad_(True)
            out = model.model(inputs_embeds=emb, use_cache=False)
        else:
            with torch.no_grad():
                out = model.model(input_ids=x, use_cache=False)
    finally:
        for hd in handles:
            hd.remove()
    return out.last_hidden_state, cache


def side_metrics(model, hidden, s, mk):
    """(lp_gold, marker) as 0-dim fp32 tensors (in the graph when hidden is)."""
    dev = hidden.device
    idx = torch.tensor([s.P - 1] + [p - 1 for p in s.gold_pos], device=dev)
    logits = model.lm_head(hidden[0, idx]).float()
    lp = torch.log_softmax(logits[1:], -1)
    tgt = torch.tensor([s.ids[p] for p in s.gold_pos], device=dev)
    lp_gold = lp.gather(-1, tgt[:, None]).sum()
    marker = (torch.logsumexp(logits[0, mk["correction"]], 0) - torch.logsumexp(logits[0, mk["commit"]], 0))
    return lp_gold, marker


def attribute(model, pair, mk):
    """Attribution patching for one pair. Returns (attr [2, n_layers, n_groups] float64 numpy, receiver metrics from the
    gradient forward, donor metrics) with metrics as [lp_gold, marker] floats."""
    n_layers = len(model.model.layers)
    allL = list(range(n_layers))
    hd, cache_d = run(model, pair.d.ids, capture=allL)
    with torch.no_grad():
        md = [float(v) for v in side_metrics(model, hd, pair.d, mk)]
    hr, hs = run(model, pair.r.ids, capture=allL, grad=True)
    mr = side_metrics(model, hr, pair.r, mk)
    hs_list = [hs[L] for L in allL]
    grads = []
    for k, m in enumerate(mr):
        g = torch.autograd.grad(m, hs_list, retain_graph=k < len(mr) - 1, allow_unused=True)
        grads.append([torch.zeros_like(hs_list[L]) if x is None else x for L, x in enumerate(g)])
    dev = hr.device
    rpos = torch.tensor([p for g in GROUPS for p in pair.groups[g][0]], device=dev)
    dpos = torch.tensor([q for g in GROUPS for q in pair.groups[g][1]], device=dev)
    seg = torch.tensor([k for k, g in enumerate(GROUPS) for _ in pair.groups[g][0]], device=dev)
    out = torch.zeros(len(METRICS), n_layers, len(GROUPS), dtype=torch.float64, device=dev)
    with torch.no_grad():
        for L in allL:
            diff = cache_d[L][0, dpos].float() - hs_list[L][0, rpos].detach().float()
            for k in range(len(METRICS)):
                c = (diff * grads[k][L][0, rpos].float()).sum(-1).double()
                out[k, L].index_add_(0, seg, c)
    res = out.cpu().numpy(), [float(v.detach()) for v in mr], md
    del hs, hs_list, grads, cache_d
    return res


def patch_pair(model, pair, comps, mk):
    """Real patching of the components [(L, group)] for one pair, batch-1 forwards without gradients.
    Returns {arm: {"lp_gold", "marker", "n_pos"}} for arms base_r, base_d, self_patch and "L<L>:<group>"."""
    layers = sorted({L for L, _ in comps})
    hr, own = run(model, pair.r.ids, capture=layers)
    hd, don = run(model, pair.d.ids, capture=layers)
    out = {}
    with torch.no_grad():
        for arm, h, s in (("base_r", hr, pair.r), ("base_d", hd, pair.d)):
            out[arm] = dict(zip(METRICS, (float(v) for v in side_metrics(model, h, s, mk))), n_pos=0)
        selfp = {}
        for L, g in comps:
            selfp.setdefault(L, []).extend(pair.groups[g][0])
        sp = {L: (pos, own[L][0, torch.tensor(pos, dtype=torch.long, device=hr.device)]) for L, pos in selfp.items()}
        h, _ = run(model, pair.r.ids, patches=sp)
        out["self_patch"] = dict(zip(METRICS, (float(v) for v in side_metrics(model, h, pair.r, mk))),
                                 n_pos=sum(len(p) for p in selfp.values()))
        for L, g in comps:
            rp, dp = pair.groups[g]
            vals = don[L][0, torch.tensor(dp, device=hr.device, dtype=torch.long)]
            h, _ = run(model, pair.r.ids, patches={L: (rp, vals)})
            out[f"L{L}:{g}"] = dict(zip(METRICS, (float(v) for v in side_metrics(model, h, pair.r, mk))), n_pos=len(rp))
    return out


def self_patch_error(arms):
    """Max absolute deviation of the self-patch from the receiver baseline, per metric."""
    return {m: abs(arms["self_patch"][m] - arms["base_r"][m]) for m in METRICS}
