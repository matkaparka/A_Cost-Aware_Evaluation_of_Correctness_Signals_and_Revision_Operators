# Validation addendum: MMLU-Pro (second review, DEVIATIONS D18)

Status: frozen 2026-11-08, after the dev pilot (session A) was read and before any test output of this task existed. `cloud/v2_sessionB.sh` checks this line. Changes after this point go to DEVIATIONS.md.

This is an exploratory addition after the main prereg froze (prereg.md, commit 08657ca). It is outside the Holm family and does not change any confirmatory outcome. It answers the second review's request (review_and_suggestion.md, 2026-11-08, §5): an independent, publicly available non-math task with natural errors, on which we check whether the repair/damage trade-off and "a better correctness ranking does not give a better re-check choice" also hold.

## 1. What is fixed

| Item | Value |
|---|---|
| Task | TIGER-Lab/MMLU-Pro (MIT), revision b189ec765aa7ed75c8acfea42df31fdae71f97be, test split; categories law, history, philosophy, psychology, biology, health (`src/data/mmlupro.py`) |
| Items | `data/processed/mmlupro/items.jsonl`, sha256 `bb8751651d7eda5dfdf1d179acabe5a044cce2b60bc572879b59fafa7d92db61`. Dev 857, test 3,457; questions sharing a stem are one bootstrap cluster |
| Models | Qwen3-4B, and Tulu-3-8B (pinned in configs/models.yaml) |
| Prompts | the main study's wrappers (src/gen/render.py) around the question, its lettered options and "Answer with the letter of the correct option." |
| Budgets | a0 4096; revision 2048; critique / scratch turn 1024 |
| Decoding | vLLM greedy, `VLLM_BATCH_INVARIANT=1`; agreement samples: 5, T 0.7, top-p 0.8, top-k 20, seeds from the stream "mc-consistency" |
| Scorer | `src/score/mc.py` 1.0.0 (letter in the last box; option text; fallback "answer is X"), sha256 `0b384926e9d9d12cc1147c7a863cbea342efa34ff0b8c3ce022350521092489e` |
| Runner | `src/exp/mc_run.py`, sha256 `96d8c4102829364bfd92e734ba39fa55484d164b94886cd7cfd04722dce2bc04` |
| Analysis | `analysis/mc_results.py`, written before any test output, sha256 `fd42699e55cfff6d36ea13b0d4bca4f80765385a345bc49a908a1acc5e9471d4` |
| Bootstrap | 10,000 resamples, clusters = canonical stem |

## 2. Readouts and signals

- Verbal readout W0 (the main study's fixed question) is primary for both models.
- For Tulu-3-8B, a second readout is reported if the dev-only readout diagnostic on math (`src/score/verbal_variants.py`, rule fixed before any output) selected one. It is secondary and never replaces W0.
- The task probe is selected and fitted on this task's dev split by the main study's grid and applied to test. The model's frozen math probe is applied unchanged and reported separately.
- Agreement with a0 is the share of 5 samples with a0's letter. Answer length is the number of a0 tokens.

## 3. Conditions (one fixed a0 per item)

B0 (keep a0), F0, F1, F2 (critique, then revision), F2-scratch, F3(1.0), F3(0.5).

## 4. Primary comparisons (estimation; no tests)

Read from the dev pilot: `reports/v2_01_mc_dev.md` (session A, dev only) and `reports/fu_07_verbal_variants.md`.
- The readout rule selected no variant for Tulu-3-8B, so W0 is the only readout on this task.
- Dev precision from the discordance rates projects 95 % half-widths of about 0.7–1.5 pp for accuracy contrasts on the 3,457 test items, and about 0.7–2 pp for W->R contrasts.

Each comparison is estimated per model with a 95 % cluster-bootstrap CI (`analysis/mc_results.py`). There are three groups, and the reading rules are fixed here.

**P1. Repair and damage.** For F2, F1 and F0: the net accuracy change against a0, with W->R and R->W.
- Reading: "net gain" if the CI of the net change lies above zero; "net loss" if it lies below; "no net change resolved" otherwise.
- W->R and R->W are always reported next to it.
- Dev expectation, not a hypothesis test: on 4B, F2 repairs and damages at similar rates (net near zero) and F1 loses.

**P2. Does a better correctness ranking give a better re-check?** Re-check with F2 the 10 % and the 20 % of items ranked least likely correct.
- The quantity is the accuracy change under the task-probe order minus that under the verbal-W0 order, alongside AUROC(task probe) − AUROC(verbal W0).
- Reading:
  - "the ranking advantage carries over" if the order difference is above zero at both shares;
  - "not shown" if either CI includes zero;
  - "reversed" if either is below zero.
  - The AUROC difference tells whether a ranking advantage exists at all on this task.

**P3. Cost of the sampled signal.** At the 10 % share: agreement order minus verbal-W0 order (pp), with the incremental generated tokens per item of each. These sit next to the net change and tokens of F2 on every item.
- Reading: descriptive. The report states whether the sampled order's extra gain is bought at a token cost above that of re-checking every item.

**Secondary, exploratory:**
- F2 − F0, F2 − F1, F2 − F2-scratch;
- F3(1.0) and F3(0.5);
- the frozen math probe applied unchanged;
- the dev-fitted rules and the other rankings;
- re-check with F0;
- the category breakdown;
- T8B − 4B on the common items.

No outcome of this addendum enters or changes the confirmatory family.

## 5. Failure handling

- **Parse failures.** A model whose test a0 are unparsed or truncated for more than 10 % of items is reported descriptively only.
- **Probe selection.** The task probe is re-selected on the same dev data by the same code in session B; a different selection than session A's is reported as a deviation, and session B's is used.
- **Deadline.** If session B is cut by its deadline, completed blocks are reported, and blocks still missing are marked "not run". Those missing stages of this frozen plan may be completed in a later session. Nothing is re-run on test after its results have been seen.
- **No changes after freezing.** Items, prompts, scorer, conditions, readouts and `analysis/mc_results.py` are not changed. A change would need a DEVIATIONS entry and new run IDs, and its results would be labeled post hoc.
- **Already exploratory.** The test split of this task has not been used. Its role as a check is limited by the earlier test results having motivated the questions; the report says so.

## 6. Not decided here

Anything not listed above is exploratory and is labeled so in the report.
