# P1 — Data (2026-11-03)

All counts are in the generated companion [`01_data_tables.md`](01_data_tables.md) (`analysis/p1_data_card.py`, from `data/processed/` and `results/p1/`). This page holds no hand-typed numbers.

## What was done
- **Sources**, downloaded on the run host at pinned revisions (`src/data/fetch.py`): GSM8K test and MATH-500 (D-nat); gsm8k_sc, prm800k_sc and scli5 (D-inj). The D-inj revisions match the design's prefixes.
- **Canonical IDs and splits first** (`src/data/build.py`, design §5.2):
  - NFKC/whitespace/case-normalized question hashes;
  - cross-source links (GSM8K ↔ gsm8k_sc, MATH-500 ↔ prm800k_sc), all by exact normalized match;
  - near-duplicate merges (exact normalized + MinHash ≥ 0.9), every union logged;
  - a seeded 20/80 dev/test split per stratum at the canonical-ID level;
  - seeded processing orders;
  - the fixed Exp 5 spillover (test) and KL-probe (dev) sets.
- **Merges only force a shared split.** They occur only within scli5: letter tasks that differ in case, and arithmetic prompts that differ in a number. Every source row stays its own item.
- **Exp 0** (`src/data/exp0.py`): 200 templates and 400 paired-world items. The option order is seeded. Both worlds share one prompt (tested).
- **Verifiers** (`src/data/verifiers.py`): F3(p) with one uniform per item, nested across p; stratum-permuted self-verdicts (K = 3).
- **Scorer** (`src/score/scorer.py`, design §5.6): math-verify equivalence, with 72 tests (`tests/test_scorer.py`).
  - The conflict rule counts only boxes offered as alternatives; otherwise the last box is the answer. A broader rule would have scored revisions as wrong.
- **Tests** (`tests/test_exp0_verifiers.py`): the scripted binary agent truth table, nested masks, the counterfactual-label check for F3, and count preservation under permutation.
- **Initial answers a0** for all three models on every D-nat item (`src/gen/initial.py`): vLLM, `VLLM_BATCH_INVARIANT=1`, greedy, non-thinking, budget **4096**. Then the verbal P(yes) (`src/score/verbal.py`, design §5.7). Test rows were only counted.
  - The first 4B run used a 2048-token budget. On dev, every truncated MATH-500 a0 was a long unfinished solution, not a repetition loop. So the budget was raised, and the 2048 runs are marked superseded in `runs.jsonl` (their outputs are kept on the run host).
  - The first-token shortcut for the verbal verdict was verified for all three tokenizers: every candidate is a single token, and the IDs are distinct.

## Observations that feed P2 (numbers in the tables)
- **Natural errors on test are few for 4B and 8B.** This limits D-nat W→R power. The design already makes D-inj primary for H3a (§3.5). Pool expansion is decided by the P2 power simulation (§5.1, §12), never by outcomes.
- **1.7B on MATH-500 puts much of its verdict-turn probability outside the yes/no candidates** (low mean mass). The verdict prompt's behaviour on 1.7B is checked on dev in P2 before the self-verdict threshold is fixed.
- **Truncation at 4096 remains on a few dev MATH-500 items.** It is counted and scored wrong (design §5.6).

## Tsui protocol facts for the replication cell (arXiv 2507.02778v3, read 2026-11-03)
- **Internal error:** the erroneous reasoning is the final assistant turn, and the model **continues** it (no new user turn).
- **External error:** the same text is appended to the user message, and the model answers in a new assistant turn.
- **Decoding:** temperature 0, 1024 new tokens.
- **Variants:** "BCA" = before committing an answer (the `_bca` fields); "ACA" = the variant that includes the final answer.
- **Scoring:** an LLM judge (Gemini 2.5 Flash) in the paper.
- **Planned differences in our replication cell (P2):** scoring by the design §5.6 scorer, not an external LLM judge (calling a paid third-party API is a human gate, and the scorer is deterministic); Qwen3 models in non-thinking mode. "Wait" is appended to the erroneous assistant text, as in the paper; the exact whitespace is not specified there and will be fixed on dev.

## Problems
- One design detail is unspecified upstream: the exact formatting of "Wait" in Tsui's paper. It is chosen on dev and logged.
- prm800k_sc has no explicit mistake-step field, so F4 (the location pointer) is available only for gsm8k_sc (design: "where the injected step … [is] known").

## Next step
P2 dev pilots on 1.7B and 4B: the review-turn prompts and revision budget, the self-verdict threshold, probe and steering selection, the TTT update rule, feasibility decisions and the power simulation.
