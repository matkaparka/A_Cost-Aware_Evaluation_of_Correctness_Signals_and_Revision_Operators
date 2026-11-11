"""Chat rendering with span bookkeeping (design §6.0).

Every condition is rendered through the model's chat template. For each rendered prompt we keep the character
spans of: question, label, solution (the reviewed a0), feedback, instruction, and critique (F2 second turn).
Token spans come from the tokenizer's offset mapping (token_spans). Rules:
- the source label always precedes the reviewed solution;
- the reviewed solution's tokens must be identical across conditions for the same item (tests/test_render.py);
- compared wrappers are length-matched within +/-3 tokens (tested).

Wording is fixed a priori and checked on dev for validity only (parse and truncation rates), never tuned on
correction outcomes. Its hash goes into every run manifest.
"""
from dataclasses import dataclass, field

PROMPT = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# Source labels (Exp 1B). Exp 1A uses the (assistant, self) cell, so 1A F0 and 1B (assistant, self, F0) coincide.
LABELS = {
    "self": "The following is your own earlier answer to this problem.",
    "other": "The following is another model's earlier answer to this problem.",
}
# Exp 1B robustness: the one alternative wrapper set, run on a 300-item subset (design Exp 1B). Fixed a priori on
# 2026-11-05, length-matched like LABELS, checked on dev for validity only (src/exp/runner.py --label-set alt).
LABELS_ALT = {
    "self": "Below is a solution to this problem that you wrote earlier.",
    "other": "Below is a solution to this problem that a different model wrote earlier.",
}

INSTRUCTION = ("Review the answer above and give your final answer. "
               "Put your final answer within \\boxed{}.")

VERDICT = {   # F3(p), F5, F3-perm share this wrapper; only the verdict word differs
    True: "Feedback: a checker reviewed the answer above and judged it correct.",
    False: "Feedback: a checker reviewed the answer above and judged it incorrect.",
}
LOCATION = {  # F4 (D-inj only): a pointer to the injected step, or "no error found"
    "step": "Feedback: a checker found an error in this step of the answer above: \"{step}\"",
    "none": "Feedback: a checker reviewed the answer above and found no error.",
}
CRITIQUE_TURN = "Review the answer above and write a critique listing any errors you find. Do not give a final answer yet."
SCRATCH_TURN = "Solve the problem again independently, step by step. Do not give a final answer yet."
TRIGGER_PREFILL = "Wait,"

# Exp 5 context arm: q + "Previous attempt: a0. Verdict: v." in one user turn (design Exp 5, 2 x 2).
CONTEXT_MEMORY = "{prompt}\n\nPrevious attempt:\n{a0}\n\nVerdict: the previous attempt is {verdict}."


def context_prompt(tok, q, a0, verdict_correct):
    question = PROMPT.format(q=q)
    content = CONTEXT_MEMORY.format(prompt=question, a0=a0, verdict="correct" if verdict_correct else "incorrect")
    text = render_messages(tok, [{"role": "user", "content": content}])
    return Rendered(text=text, spans=_locate(text, [("question", question), ("solution", a0)]))


# Exp 0: the same feedback wrappers, with an option-letter instruction.
EXP0_INSTRUCTION = "Review the choice above and give your final choice. Put only the letter (A or B) within \\boxed{}."
VERIFY_QUERY = "Is your final answer correct? Answer yes or no."


@dataclass
class Rendered:
    text: str                       # the full prompt string fed to the model (chat template applied)
    prefill: str = ""               # text appended after the generation prompt (F1 "Wait,")
    spans: dict = field(default_factory=dict)   # name -> (char_start, char_end) in text


def _locate(text, pieces):
    """Char spans of pieces, searched left to right in order (each must occur after the previous one)."""
    spans, pos = {}, 0
    for name, piece in pieces:
        if not piece:
            continue
        i = text.find(piece, pos)
        if i < 0:
            raise ValueError(f"span {name!r} not found in rendered text")
        spans[name] = (i, i + len(piece))
        pos = i + len(piece)
    return spans


def render_messages(tok, messages, *, generation=True, continue_final=False):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation and not continue_final,
                                   continue_final_message=continue_final, enable_thinking=False)


def review_prompt(tok, q, solution, *, role="assistant", label="self", feedback="", instruction=INSTRUCTION,
                  prefill="", history=(), labels=None):
    """A review-turn prompt: question, source label, reviewed solution, optional feedback, instruction.

    user[question + label] -> {role}[solution] -> (history turns) -> user[feedback + instruction]

    The role manipulation is a minimal pair: only the role of the turn that holds the solution changes
    (assistant vs user). Both versions end the solution at <|im_end|>, so its tokens are identical; a single merged
    user message would let trailing punctuation merge with the following text (found by tests/test_render.py).
    history: extra (user_text, assistant_text) turns before the final user turn (F2 / F2-scratch).
    labels: the label wording (default LABELS; LABELS_ALT for the Exp 1B robustness subset).
    """
    if role not in ("assistant", "user"):
        raise ValueError(role)
    question = PROMPT.format(q=q)
    lab = (labels or LABELS)[label]
    tail = (feedback + "\n\n" if feedback else "") + instruction
    msgs = [{"role": "user", "content": f"{question}\n\n{lab}"}, {"role": role, "content": solution}]
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": tail})
    text = render_messages(tok, msgs)
    pieces = [("question", question), ("label", lab), ("solution", solution)]
    for k, (u, a) in enumerate(history):
        pieces += [(f"history_user_{k}", u), (f"history_assistant_{k}", a)]
    pieces += [("feedback", feedback), ("instruction", instruction)]
    return Rendered(text=text, prefill=prefill, spans=_locate(text, pieces))


def fresh_prompt(tok, q):
    """B1 fresh-context: the initial-answer prompt again, a0 not in context."""
    question = PROMPT.format(q=q)
    text = render_messages(tok, [{"role": "user", "content": question}])
    return Rendered(text=text, spans=_locate(text, [("question", question)]))


def tsui_prompt(tok, messages, *, wait=False):
    """Self-Correction Bench replication cell: the dataset's own message list (design §5.4).
    Internal (last message is the assistant's erroneous reasoning): the model continues that turn.
    External (error in the user message): a new assistant turn. wait=True appends "Wait" to the continued text
    (internal) or prefills it (external)."""
    msgs = [dict(m) for m in messages]
    if msgs[-1]["role"] == "assistant":
        if wait:
            msgs[-1]["content"] = msgs[-1]["content"] + " Wait"
        text = render_messages(tok, msgs, continue_final=True)
        return Rendered(text=text, spans=_locate(text, [("solution", messages[-1]["content"])]))
    text = render_messages(tok, msgs)
    return Rendered(text=text, prefill="Wait" if wait else "", spans={})


def token_spans(tok, rendered):
    """Map char spans to token index ranges [start, end) using the offset mapping; tokens partly inside a span are
    included, so boundary merges show up as differences in the solution-token test."""
    enc = tok(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    out = {}
    for name, (a, b) in rendered.spans.items():
        idx = [i for i, (s, e) in enumerate(offs) if s < b and e > a]
        out[name] = (idx[0], idx[-1] + 1) if idx else None
    return enc["input_ids"], out
