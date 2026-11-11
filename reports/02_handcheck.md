# P2 hand checks (dev split only)

CLAUDE.md P2.6 asks for ≥ 5 hand-checked examples per metric. The samples are drawn by
[`analysis/p2_handcheck.py`](../analysis/p2_handcheck.py) (seed 1104, 5 per category; fewer when a category has fewer rows)
and printed in [02_handcheck_samples.md](02_handcheck_samples.md); the Tsui-cell scorer check is in
[02_scorer_check.md](02_scorer_check.md). The judgments below were made by reading each sample against its gold answer
(2026-11-04, by the agent). They do not replace the invariant tests.

## Exp 0 / 1A / 1B labels (runs p2-exp01-dev-{4B,1.7B}-f7e5011, scorer 1.1.0)

For each row: is the extracted answer the one the text commits to, and is the correct/incorrect label right against the gold?

| Category | 4B: rows, label errors | 1.7B: rows, label errors |
|---|---|---|
| 1A F0 self-review, wrong → right | 0 rows exist | 5, 0 |
| 1A F1/F3(1.0), right → wrong | 4 (all that exist), 0 | 4 (all that exist), 0 |
| 1A F3(1.0), still wrong | 5, 0 | 5, 0 |
| 1B user/other/F1, any row | 5, 0 | 5, 0 |
| 1A/1B truncated | 5, 0 | 5, 0 |
| Exp 0 F3(1.0) choices | 5, 0 | 5, 0 |
| Exp 0 invalid choices | 5, 0 | 5, 0 |

No label errors in 68 rows. Notes from reading them:

- **4B self-review never fixes a wrong answer on the dev 1A items** (the F0 wrong → right category is empty); 1.7B has
  such rows, and the five sampled are genuine fixes. Recorded as a behavior, not a scoring issue.
- **Truncations are greedy repetition loops** ("Let's define: f(x) = …" repeated until the token budget), not long
  reasoning cut short. Truncated rows count as incorrect. Two truncated rows carry a junk `pred` (e.g. "Right-hand
  side:"); it is never used, because truncation forces the label to incorrect. In the Exp 3 `changed` flag a truncated
  row counts as changed, which is the intended reading.
- **Exp 0 invalid choices are refusals to pick** (`\boxed{A} or \boxed{B}`, `\boxed{A and B}`) or repetition loops;
  `invalid` is the right label. Several F3(1.0) rows do use the verdict ("the checker said A was incorrect, so the
  answer is B"), so the verdict reaches the model even though the aggregate choice barely moves.
- **1.7B picks B in all five sampled F3(1.0) rows**, including both worlds of the same template (e0-18-01 w0 and w1).
  I first read this as a letter-position bias. The two columns added to the Exp 0 table in
  [02_pilot_exp01_tables.md](02_pilot_exp01_tables.md) show otherwise: neither model has a strong overall letter
  preference, but **both pick the same letter in both worlds of almost every template, under every feedback
  condition**, although the verdicts in the two worlds point opposite ways. So the pick is fixed per template and the
  verdict rarely moves it, which is why Exp 0 accuracy stays at chance by construction.
- Model-side oddities that the scorer handles correctly: two boxes with equal values (`\frac{100}{3}` or `33\frac{1}{3}`),
  a boxed non-answer (`\text{Insufficient information}`), compound-interest answers against GSM8K's simple-interest
  gold (106.12 vs 106).

## Exp 5 answer-span masks (run p2-ttt-pilot-4B-e506ac0)

For each sampled tuning item: does the update's loss mask cover exactly the final `\boxed{…}` of a0, and nothing else?

5 of 5 correct. The mask covers `\boxed{` through the closing brace of the **last** box (when a0 boxes the answer
twice, only the final one is masked), plus the newline that the tokenizer merges into the closing-brace token. Items
whose a0 has no box get no update, logged as `no_answer_span` (counts in
[02_pilot_tier2_tables.md](02_pilot_tier2_tables.md)).

## Exp 3 and Exp 5 generated texts (runs of the 2026-11-04 reruns)

The reruns write the texts (`confirm_texts.jsonl`, `generations.jsonl`); the samples are in
[02_handcheck_samples.md](02_handcheck_samples.md), and the runs are those chosen in `analysis/out/tier2.json`.

**Exp 3, 4B (p2-steer-pilot-4B-7ea8f7d): 21 of 21 labels correct.** Five categories: main setting, corrected or not;
random controls, not corrected; unsteered R→W on D-nat (a single row exists); unparsed or truncated.
- The only unsteered R→W row (D-nat) is a genuine error. The revision computes P(no homework) = 0.24 correctly, then
  reports its complement, 76 %, against the gold 24.
- Truncations at the 1,024-token budget come mostly from long prm800k_sc problems. They are a small share of the D-inj
  rows in every arm (count from the run's `confirm_rows.jsonl`) and score as incorrect. A truncated text can still
  carry an extracted `pred` (e.g. a heading such as `Point B: (1, 3)`); the status, not the pred, decides the label.
  `changed` then compares that pred with a0, which can only overstate the change rate. That matters only for the
  generic-control validity check, and there the generic arm would gain in the same way.

**Exp 5, 1.7B (p2-ttt-serve-1.7B-7ea8f7d, selected τ and lr): 22 of 22 labels correct.** Five categories: parameter
arm, self W→R; self vs perm-self disagreeing; parameter + context with ext-1.0 still wrong; context arm, verdict
incorrect; random adapter, unparsed (two such rows exist).
- An unboxed bold final answer (`**10,080**`) is extracted correctly.
- Behaviour, not a scoring issue, seen in the context cells: told "Verdict: incorrect", the model often keeps a0 or
  restates it. One text concludes "The previous attempt was correct". This is consistent with the parameter + context
  accuracy lying below the parameter arm in [02_pilot_tier2_tables.md](02_pilot_tier2_tables.md).
- The random adapter's failures are truncations mid-derivation, not malformed output.

**Exp 5, 4B (p2-ttt-serve-4B-ea34050, the deterministic rerun): 25 of 25 labels correct**, in the same five
categories. Notes:
- `\frac{270}{7}^\circ` against the gold `\frac{270}7\text{ degrees}` is scored correct, as it should be.
- The pattern seen on 1.7B recurs. With "Verdict: incorrect" in context, one text ends "The previous attempt was
  correct, so the final answer is indeed 6 days". In the context arm, initially right items told "incorrect" keep their
  right answer in the sampled rows.
- The random adapter's failures are truncations mid-derivation. One is on an initially right item.
