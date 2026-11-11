"""Exp 1C channel renderings (design "Exp 1C — Channel", §3.3 H1c; prereg §3.3: exploratory). Written after the
freeze; src/gen/render.py is frozen (prereg §13) and is reused here unchanged.

At F3(0.9) the same verdict text, R.VERDICT[v] (the F3 / F5 / F3-perm wrapper), reaches the model through three
channels. Question, label, reviewed solution a0 (assistant turn, "self" label) and revision instruction are those of
the Exp 1A F3 prompt; only where the verdict sits changes:

- "user": the existing F3 rendering, R.review_prompt(..., feedback=R.VERDICT[v]):
      user[q + label] -> assistant[a0] -> user[verdict + "\\n\\n" + instruction]
- "tool": the Qwen3 chat template's own tool message (role "tool"), which the template renders as a user turn
  "<|im_start|>user\\n<tool_response>\\n{verdict}\\n</tool_response><|im_end|>" (design §4, chat-template facts):
      user[q + label] -> assistant[a0] -> tool[verdict] -> user[instruction]
  No tool definitions (no system prompt) and no tool call in the a0 turn: either would change the tokens before or
  at the end of a0. So this is the wrapper-and-position manipulation the design describes (§3.3 H1c).
- "assistant": the verdict inside the assistant turn that holds a0, after ASSISTANT_SEP:
      user[q + label] -> assistant[a0 + ASSISTANT_SEP + verdict] -> user[instruction]

Fixed a priori (2026-11-06), before any Exp 1C output existed; checked on dev for validity only (render, answer,
parse), never for which wording works better. The verdict text is byte-identical in all three channels.

ASSISTANT_SEP is " \\n\\n" (a space before the blank line), not "\\n\\n". The Qwen2/3 pre-tokenizer joins trailing
punctuation with the newlines that follow it, so "a0.\\n\\nFeedback" turns a0's last token "." into ".\\n\\n"; that
happened for 165 of the 364 4B dev a0. With " \\n\\n" (a single token) a0's tokens are identical to the other two
channels on every dev item (tests/test_channel.py), as design §6.0 requires. The exception left is an a0 that itself
ends in whitespace (a truncated a0 can), whose trailing space merges with " \\n\\n". The run checks every item and
flags exceptions (src/exp/channel.py `solution_tokens_identical`); flagged items stay in every analysis, and a
sensitivity analysis leaves them out (design §6.0: boundary differences are flagged and handled by a rule).

Length (pinned tokenizer, tests/test_channel.py): "assistant" is 1 token longer than "user"; "tool" is 8 tokens
longer, the template's own wrapper and turn boundary. The ±3-token rule for compared labels (design §6.0) cannot
apply to the tool channel: the wrapper is the manipulation.
"""
import json

from . import render as R

CHANNELS = ("tool", "user", "assistant")
ASSISTANT_SEP = " \n\n"
CHANNEL_WORDING = {   # hashed into the run config (src/exp/channel.py `channel_wording_sha256`)
    "verdict": "R.VERDICT[v], unchanged",
    "tool": "messages: user[q+label], assistant[a0], tool[R.VERDICT[v]], user[R.INSTRUCTION]; no tools, no tool call",
    "user": "R.review_prompt(tok, q, a0, feedback=R.VERDICT[v])",
    "assistant": "messages: user[q+label], assistant[a0 + ASSISTANT_SEP + R.VERDICT[v]], user[R.INSTRUCTION]",
    "assistant_sep": ASSISTANT_SEP,
}


def channel_wording_json():
    return json.dumps(CHANNEL_WORDING, sort_keys=True)


def channel_prompt(tok, q, a0, verdict_text, channel):
    """The F3 review prompt with verdict_text delivered through `channel` (one of CHANNELS). Spans: question, label,
    solution (a0), feedback (the verdict text), instruction."""
    if channel == "user":
        return R.review_prompt(tok, q, a0, feedback=verdict_text)
    question = R.PROMPT.format(q=q)
    lab = R.LABELS["self"]
    head = {"role": "user", "content": f"{question}\n\n{lab}"}
    if channel == "tool":
        msgs = [head, {"role": "assistant", "content": a0}, {"role": "tool", "content": verdict_text},
                {"role": "user", "content": R.INSTRUCTION}]
    elif channel == "assistant":
        msgs = [head, {"role": "assistant", "content": a0 + ASSISTANT_SEP + verdict_text},
                {"role": "user", "content": R.INSTRUCTION}]
    else:
        raise ValueError(channel)
    text = R.render_messages(tok, msgs)
    pieces = [("question", question), ("label", lab), ("solution", a0), ("feedback", verdict_text),
              ("instruction", R.INSTRUCTION)]
    return R.Rendered(text=text, spans=R._locate(text, pieces))
