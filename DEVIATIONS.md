# DEVIATIONS from prereg.md

prereg.md was frozen at commit 08657ca on 2026-11-05 (sha256 in STATUS.md). Design changes made before the freeze are in
DECISIONS.md.

Format: date — what changed — why — expected effect on conclusions — affected run IDs — test results seen when the
change was made.

## D1 — 2026-11-05 — scorer 1.1.1: a resource guard on math-verify; the runner writes generations before scoring

- **What.**
  - `src/score/scorer.py` 1.1.0 → 1.1.1:
    - On Linux (the run host), every math-verify call (parse + verify, and the set-gold parse) runs in a child forked by a small helper process. The helper is a plain `python -c` interpreter that has loaded math-verify once. Each child runs with a CPU-time limit of 20 s and an address-space limit of 8 GiB above the fork's own mapping, so every call starts from the same state. A 600 s wall-clock backstop exists, but the CPU limit ends a child long before it.
    - A call that exceeds a limit, or raises, counts as "math-verify does not confirm". The answer is then correct only if the normalized strings are equal; this is what 1.1.0 already did when `verify` raised. Every such call is recorded in `scorer.GUARDED`, and the runner and the Exp 3 test module write it to `metrics.json` (`scorer_guarded`).
    - Normalized string equality is now checked before math-verify. 1.1.0 returned "math-verify confirms OR the strings are equal", so the order changes no verdict.
    - Results are cached per (answer, gold).
    - On other platforms (the laptop: tests and analysis) calls run inline, without the guard.
  - `src/exp/runner.py` writes `generations.jsonl` before scoring, and records `scorer_guarded` and `score_s` in `metrics.json`.
  - `src/steer/test.py` records `scorer_guarded`.
  - `cloud/p4a_sequence.sh`:
    - its gate requires scorer 1.1.1;
    - it checks on the run host that the guard stops four non-terminating answers and keeps ordinary verdicts.
  - The step invocations are unchanged.
- **Why.**
  - The first 4B test runner (`main-exp01-test-4B-06c77b9`) generated for about 1.5 h. The process was then killed (`Killed`, exit by SIGKILL) during scoring. The watchdog did not act, and the instance's cgroup has a 112 GiB memory limit, so the out-of-memory killer is the likely cause.
  - 1.1.0 runs math-verify with its timeouts disabled, so that verdicts never depend on wall-clock time.
  - On the laptop, 1.1.0 does not finish within 20 s on `10^{10^{10}}`, `2^{2^{40}}`, `9^{9^{9}}`, `(x+1)^{100000}` or `10^{100000000000}`: the big integers or expansions grow until memory runs out.
  - The runner held all generations in memory until scoring ended, so none were kept. The answer that triggered the kill is therefore not known.
- **Expected effect on conclusions.** None beyond the guarded answers themselves.
  - For every answer on which 1.1.0 terminates within the limits, 1.1.1 gives the same verdict.
  - Check: `analysis/p4_scorer_guard_check.py` → [reports/04_scorer_guard_check.md](reports/04_scorer_guard_check.md). It runs 1.1.0 (from the freeze commit) and 1.1.1 on every stored output: P1 initial answers of all three models on dev and test, the dev runner generations, and the Exp 3 dev pilot texts. Result: no verdict, status or extracted answer differs on any of them (counts per source in the report).
  - A guarded answer is scored incorrect unless it equals the gold as a string. Under 1.1.0 such an answer had no verdict at all: the run died.
  - The number of guarded calls per run is reported with that block's results, and the guarded answers are hand-checked.
  - The CPU-time limit makes a verdict depend on the machine only for an answer whose comparison needs more than 20 s of CPU. Ordinary comparisons take milliseconds.
- **First implementation failed the gate (2026-11-05 21:58Z, before any GPU step; about $0.08).** It used a multiprocessing forkserver. multiprocessing re-runs the parent's `__main__` in each child: under the gate's `python -` that is `<stdin>`, which does not exist as a file; under `python -m src.steer.test` it would re-import torch in every child. It was replaced by the helper above. The tests and the gate now accept either outcome for the four non-terminating answers: stopped by a limit, or math-verify giving up by itself (it catches MemoryError). In both cases the verdict is "not equivalent".
- **Affected run IDs.**
  - `main-exp01-test-4B-06c77b9`: failed (killed). No outputs exist; it is marked `failed+superseded` in runs.jsonl.
  - Every later scored run uses 1.1.1.
- **Test results seen.** None. The killed run wrote nothing. The zero-difference check compares the two scorers on the P1 test initial answers, which are pre-feedback quantities, and reports only agreement counts.
- **Run-host check (2026-11-05, commit `dd0ab36`, guard active).**
  - The same zero-difference check was run on the run host: [analysis/out/scorer_guard_check_runhost.json](analysis/out/scorer_guard_check_runhost.json). No verdict differs, and the guard stopped no call.
  - The gate's guard check passed there: the four non-terminating answers were stopped by a limit, and an ordinary guarded call takes milliseconds (printed in the sequence log).
- **New hashes** (sha256 of the committed blobs at `dd0ab36`; they replace the prereg §13 entries for these files):

  | File | sha256 |
  |---|---|
  | `src/score/scorer.py` | `c4fedce34bc16e1941fc377474b1c3e579fc3d6aaf1402e9775db2ca14fdbfc3` |
  | `src/exp/runner.py` | `fc27f38e31545def1f3cd9d9a4db57a7f10f4454e353617abbb076782e4c27c2` |
  | `src/steer/test.py` | `3f8aa84f60ea1876647ba618aa3135674be900515596659a1184d32fc8417e8f` |
  | `cloud/p4a_sequence.sh` | `1a37395830728102130d589cacb027569fbf82d919ba01ed36249bb87e821f1e` |
- **Addendum (2026-11-05, before Exp 3 ran):** `cloud/p4b_sequence.sh` (Exp 3) also asserted scorer 1.1.0. Its gate now requires 1.1.1. Nothing else in it changed. New hash: in D2's table.

## D2 — 2026-11-05 — Exp 5 test module (src/ttt/test.py, cloud/p4c_sequence.sh), written after the freeze

- **What.** The prereg froze the Exp 5 pool, the update rule, τ, lr, the controls, the anomaly rules, and the status (H4a estimation-only, H4b–d exploratory). No test-split Exp 5 code existed at the freeze; the dev pilot code was `src/ttt/pilot.py` and `src/ttt/serve.py`. The new module follows them. It fixes these points, which the prereg leaves open:
  1. **H4a accuracy** is the greedy parameter-arm accuracy, as in the dev pilot and the P2 power simulation. Four samples at T = 0.7 per cell (design "greedy plus 4 samples") are generated with one seed per item, shared by every cell, and reported as a secondary accuracy.
  2. **Verdict populations.** These are the Exp 1A runner's: the original-pool D-nat test items of the P1 initial run. perm-self k = 0..2 here therefore equals F3-perm[k] in Exp 1A. Shuffled verdicts follow the dev pilot's rule (self-verdicts permuted within dataset, seed `h("shuffled-v1", dataset)`) on that population.
  3. **Spillover (H4b; the design says "gold log-prob" and gives no sequence).**
     - The sequence is the base model's own a0 for each spillover item, with its last `\boxed{}` content replaced by the gold, or `\boxed{gold}` appended when a0 has none.
     - The quantity is the mean log-prob of the tokens of that final `\boxed{gold}`, teacher-forced. This is the span the update rule trains on.
     - An update's spillover is the mean over the 50 spillover items of (adapted − base). It is computed for all 7 adapters.
  4. **Per-update diagnostics.** KL(base ‖ adapted) and mean |Δ log-prob| over the vocabulary, on the a0 tokens of the first 10 KL-probe items (canonical-ID order), not all 50: cost.
  5. **H4d self-prediction.**
     - Prompt: q followed by "State only the final answer you would give to this question. Put it within \boxed{}." The second sentence was added so the answer can be scored.
     - Cells: base, self adapter, random adapter, and the context arm with the self verdict (that prompt ends with the same instruction). Greedy, 64 tokens.
     - Each is compared with the matching greedy re-answer: B1, parameter-self, parameter-random, context-self.
  6. **Serving.** vLLM `max_loras` = 64 (the dev serve used 16) to keep batches full with 351 adapters per chunk. Under `VLLM_BATCH_INVARIANT=1` outputs do not depend on batch composition. The zero-step = B1 invariant is checked in every chunk, and the sequence stops if it fails.
  7. **Chunks and storage.** 8 chunks of 50 (blocks.yaml chunk 50, min_n 200). After serving, adapters are deleted except a seeded 5 % sample (`h("ttt-keep-v1", name)`), as the design specifies. The reset invariant is checked at the end of every training chunk.
  8. **Smoke test.** The sequence first runs 2 items through every stage (run kind `ttt-test-smoke`).
- **Why.** These are implementation choices for estimation-only and exploratory quantities. They were fixed before any Exp 5 test output existed.
- **Expected effect on conclusions.** None on the confirmatory family. H4a stays estimation-only. The spillover and self-prediction definitions bound what H4b and H4d can say.
- **Affected run IDs.** `main-ttt-test-4B-<commit>` (not yet run).
- **Test results seen.** None from Exp 5. When this entry was written, the 4B Tier 1 test runs (P4a) were still running, and no output from them had been seen.
- **Hashes** (sha256 of the committed blobs at `9c01e77`):

  | File | sha256 |
  |---|---|
  | `src/ttt/test.py` | `a6aeaa756e5921b6f322eebc660239dea3e3c40b5285044a95c53e5bdaab75ba` |
  | `cloud/p4c_sequence.sh` | `1de9aa43e78d16a7d5e5d5f45d2f2c6864bf2f181a1bd92ab59fef0298e62f17` |
  | `cloud/p4b_sequence.sh` | `7e2594129f73cc548528158f5bcab647c069d6f391ed454ca1d7ef3e66fece4c` |

## D3 — 2026-11-05 — 8B t* script and the 8B Tier 1 sequence (cloud/p4d_sequence.sh)

- **What.**
  - Prereg §11 step 1 names `analysis/p2_selfverdict.py` for the 8B t*. That script also runs an H1d power part, which needs an 8B dev runner run; none exists and the rule does not need one.
  - `analysis/p4_tstar_8B.py` therefore applies the same threshold rule using that script's own `score` and `balanced_acc`. It first reproduces the frozen 4B and 1.7B t* exactly, and stops if it cannot. It writes `analysis/out/selfverdict_8B.json` and leaves `selfverdict.json` untouched.
  - The selection is logged in runs.jsonl as `p4-tstar-8B-597994e`, before any 8B test run.
  - `cloud/p4d_sequence.sh` is the frozen p4a runner invocation with `--model 8B` and the 8B t*, then the same invocation for 1.7B (priority 5) with the frozen 1.7B t*. Its gate requires the logged 8B selection.
- **Why.** Prereg §11 requires both steps; neither had code at the freeze.
- **Expected effect on conclusions.** None. 8B results are replications with CIs.
- **Test results seen.** None from 8B. P4a (4B) was still running when this was written.

## D4 — 2026-11-05 — Exp 5 evaluation batching (src/ttt/test.py `masked_batches`), after the first test chunk was stopped

- **What.**
  - `masked_batches` builds the micro-batches for the per-update KL diagnostic and the spillover quantity (D2 items 3–4). It now also caps the padded input tokens per micro-batch (`max_tokens` = 16,384), in addition to the kept logit positions.
  - Before, the 50 spillover sequences fell into one batch padded to the longest sequence (4,218 tokens): 210,900 padded tokens for 29,048 real ones. With the cap: 4 batches, 39,789 padded tokens. The KL batches do not change.
  - Nothing else changed. The update, the adapters, the served cells and H4a do not use this function.
- **Why.** Cost.
  - The first test run (`main-ttt-test-4B-61648f6`) trained one item, i.e. 7 adapters, every ≈ 143 s. In the 2-item smoke test (2 spillover items) an item took 13 s. That rate projects to ≈ 16 GPU-h for training alone, ≈ $31 for the block, more than the session allowance ($27.71), with nothing left for the priority-4/5 blocks.
  - The run was stopped at 15:25Z during chunk 0 training (no update file written) and paused (≈ $0.7 for the session). The stop and its local profiling are in DECISIONS 2026-11-05.
- **Expected effect on conclusions.**
  - Spillover and KL are computed with a different batch layout, so bf16 values move at noise level. Local check on 1.7B, 36 spillover items, one batch vs the capped batches: mean |Δ| 0.0017 nats, max 0.023, on per-item mean log-probs around −0.31.
  - Within the run, base and adapted values share the layout. H4b stays exploratory; H4a is unaffected.
- **Affected run IDs.** `main-ttt-test-4B-61648f6` and its smoke run `p2-ttt-test-smoke-4B-61648f6` are stopped and superseded by `main-ttt-test-4B-<new commit>` and its smoke run.
- **Test results seen.**
  - From Exp 5: only the smoke run's invariants (reset exact, zero-step = B1) and the training timestamps.
  - From other blocks: the 4B Tier 1 and Exp 3 per-block estimates (reports/04_p4_4B_tier1.md, 04_p4_4B_exp3.md) had been seen. They do not involve this function.
- **Hashes.** `src/ttt/test.py` sha256 `af6bb8e7373385321603a9ac5eb54323d17d4990a9050d737754ebd70f6971de` at `1d075c6`.

## D5 — 2026-11-06 — Exp 5 estimates script (analysis/p4_exp5.py), written after the block ran and before any estimate was computed

- **What.**
  - The frozen analysis (`analysis/p6_confirmatory.py`) has no Exp 5 code. It also covers no other estimation-only Tier 2 quantity.
  - `analysis/p4_exp5.py` computes the Exp 5 quantities with the frozen bootstrap (`Boot`, 10,000 resamples of canonical problems, seed 0). It fixes these points, which the prereg and D2 leave open:
    1. **H4a.** H4a is d_i = greedy parameter-self − mean over k of greedy parameter-perm[k], over all 400 items (intention-to-treat, prereg §6).
    2. **H4a sensitivity (prereg §6, "c > 100 excluded").** An item is excluded when its self update or any of its perm-k updates has c > 100. The ext, shuffled and random updates do not enter H4a, so their flags do not exclude the item.
    3. **Secondary contrasts:**
       - the H4a contrast on sample accuracy (mean of the 4 samples);
       - the context-arm counterpart: the context prompt with the item's self verdict vs with each perm-k verdict, both served for every item;
       - the parameter+context counterpart;
       - the coupling × verdict-source interaction, i.e. the parameter contrast minus the context contrast;
       - H4a within initially wrong and initially right items.
    4. **Per cell:** accuracy, its difference from B1, W->R, R->W, and sample accuracy.
    5. **H4b.** Mean per-update spillover per grounding, and self − mean perm-k.
    6. **H4d.** Consistency means the self-prediction answer is equivalent to the matching greedy re-answer under the scorer's `equivalent`. A missing answer counts as not consistent. The matches are as in D2 item 5.
- **Why.** The prereg fixes the H4a estimand and its status (estimation-only), but no code. The other quantities are exploratory and need a fixed definition before they are computed.
- **Expected effect on conclusions.** None on the confirmatory family. H4a stays estimation-only.
- **Test results seen from Exp 5 before writing this:**
  - the run's metrics.json (invariants, anomaly counts, adapters kept);
  - row counts by arm and status;
  - anomaly flags by grounding.
  No accuracy, contrast, spillover or self-prediction value had been computed or seen.
- **Hashes.** In the commit that adds the first estimates (STATUS).

## D6 — 2026-11-06 — 1.7B Exp 2 on test (cloud/p4e_sequence.sh), missing from the p4d sequence

- **What.**
  - Block `1.7B-tier1` in configs/blocks.yaml (priority 5) is "Exp 0 + 1A + 1B (original pool) + 2". `cloud/p4d_sequence.sh` (D3) runs the 1.7B runner but not Exp 2. I noticed the gap on 2026-11-06, while p4d was running.
  - `cloud/p4e_sequence.sh` fills it, reusing the frozen p4a invocations with `--model 1.7B`:
    - `src.probe.cache` on the original-pool test items;
    - `src.probe.score` with the frozen 1.7B probe (prereg §6, sha256 `db0c6752…`, checked by the gate).
  - It runs in the next cloud session, after p4d.
- **Why.** Without it, the 1.7B H2a replication cannot be computed. The block is part of priority 5.
- **Expected effect on conclusions.** None. 1.7B results are replications with CIs, and the probe and the code are the frozen ones.
- **Test results seen.** None from 1.7B or 8B test runs. p4d was still in its first (8B) runner call.

## D7 — 2026-11-06 — 4B thinking subset on test (src/exp/thinking.py, cloud/p4f_sequence.sh), written after the freeze

- **What.**
  - The prereg fixes the thinking subset's status (exploratory), its decoding (T 0.6, top-p 0.95, top-k 20, min-p 0; budget 16,384; 2 paired seeds; prereg §1) and its place in the block order: priority 6, chunk 150, min_n 150 (prereg §10). The design names 300 items stratified by initial correctness with weights, and the conditions {B1, F0, F1, F2, F3(0.7), F3(1.0), F5} plus the Exp 1B 2 × 2 at F0.
  - No test-split code existed at the freeze; only the dev budget pilot (`src/gen/think_pilot.py`). The new module fixes these points:
    1. **a0.** a0 is the frozen P1 non-thinking initial answer. Only the revision turns are generated in thinking mode.
       - Reason 1: stratifying "by initial correctness" before any generation needs a0's correctness.
       - Reason 2: every verdict is then the one the non-thinking runs used (F3 masks, the F5 self-verdict at the 4B t*). The thinking cells are paired with the non-thinking Exp 1A / 1B cells on the same items.
       - So the contrast with the main runs is one of revision mode, holding a0 fixed.
    2. **Items.** The original-pool D-nat test items, stratified by a0 correctness with equal allocation of up to 150 per stratum, taken first-in-order in the seeded order. The pool has only 149 initially wrong items, so the subset is 149 + 150 = 299.
       - Chunk 0: 75 + 75 items. Chunk 1: the remaining 74 + 75.
       - Weights are N_stratum / n_stratum of the items actually run (N = 149 wrong, 1306 right).
    3. **Exp 1B.** The Exp 1B 2 × 2 at F0 uses the same D-nat items with their own a0, as the runner's D-nat 1B cells do. (assistant, self, F0) is the same prompt as 1A F0 and shares its generation.
    4. **Prompts.** The runner's prompts with the non-thinking generation suffix `<think>\n\n</think>\n\n` removed. This equals the chat template rendered with `enable_thinking=True` (tests/test_thinking.py, 80 prompts locally).
       - F1: the "Wait," prefill sits inside the think block (`<think>\nWait,`).
       - F2: the critique enters the second turn without its think block, as the template drops reasoning from earlier turns. An unclosed critique is passed as it is.
       - Every turn, the F2 critique included, has the 16,384-token budget.
    5. **Seeds.** Seed of (item, s) = `h("think-seed-v1", item, s)`, shared by every condition of the item.
    6. **Scoring.** The frozen scorer, which strips the think block. Truncated or unclosed = wrong, counted per condition (prereg §1).
    7. **Sequence.** A dev smoke test first (1 item per stratum, every condition). Then test chunk 0. Then chunk 1 only if chunk 0's wall time × 1.2 + 15 min ends 15 min before the session deadline. Otherwise the block stops at chunk 0 = min_n, a pre-registered partial sample (prereg §10).
- **Why.** The prereg needs this block in its priority order. Its design leaves these points open, and they must be fixed before any thinking test output exists.
- **Expected effect on conclusions.** None on the confirmatory family. The thinking subset is exploratory. Its results speak to thinking-mode revision of a fixed non-thinking a0, not to thinking-mode answers from scratch (B1 excepted).
- **Test results seen.**
  - None from thinking mode on test.
  - Seen from other blocks: the 4B Tier 1, Exp 3 and Exp 5 per-block estimates.
  - p4d (8B, 1.7B) was running, and no output from it had been seen.
- **Hashes** (sha256 at `c6ba469`): `src/exp/thinking.py` `08ceaec0778fd682f9e6aa567c3a45053e7298f0a6439fb436f9fcbfb58c6567`; `cloud/p4f_sequence.sh` `d313651abb043202016bd215eab1a21e7fd2b30159d58eb4f605b48d1b00455e`.
- **Addendum (2026-11-06, before any thinking output existed): analysis.**
  - `analysis/p4_thinking.py` computes the subset's estimates. Each item's outcome is its mean over the 2 seeds. Population accuracy is weighted by stratum (N 149 / 1306 over the n run). W->R and R->W are computed within each stratum.
  - Each cell is paired with the greedy non-thinking 4B Exp 1A / 1B cell on the same items (`main-exp01-test-4B-a7a0579`).
  - Within-thinking contrasts: F2 − F1, F3(1.0) − F3(0.7), F5 − F0, F3(1.0) − F0, and the Exp 1B role and label effects at F0.
  - All are estimates with 95 % cluster-bootstrap CIs (frozen `Boot`). The script was checked on synthetic rows only.

## D8 — 2026-11-06 — Priorities 7, 8 and 12 after the cap rise (cloud/p4g_sequence.sh); 8B probe selected after the 8B Exp 0/1 test runs

- **What.**
  - The cloud budget was raised (private decision log, 2026-11-06). Blocks 7 (8B and 1.7B Exp 1A on the expansion), 8 (8B Exp 2) and 12 (8B thinking) now run in `cloud/p4g_sequence.sh`, in that priority order.
  - **8B probe.** Prereg §11 says the 8B probe selection is "committed and logged in `runs.jsonl` before any 8B test run". The 8B Exp 0/1 test run (`main-exp01-test-8B-3e6149d`, p4d) has already happened.
    - The 8B dev activations (original + expansion) are cached on the run host and synced.
    - The probe is selected on the control host by the frozen `src/probe/train.py` (sha256 `5a66d567…`, prereg §13): the same grid, grouped CV and seed. No human or agent choice is involved.
    - It is committed and logged before any 8B Exp 2 test step; the sequence's gate checks its sha256.
  - **Expansion in Exp 2.** Block 8 in configs/blocks.yaml covers the 1455 original-pool items. Because block 7 now adds the expansion to 8B and 1.7B Exp 1A, Exp 2 also caches and scores the expansion test items, as on 4B (p4a).
    - 1.7B Exp 2 is re-run at the p4g commit, original + expansion, so that its scores come from one locked commit.
    - The new run supersedes `main-probe-scores-test-1.7B-3e5ddd4` and `main-acts-test-1.7B-3e5ddd4` (code unchanged; the reason is "expansion added").
  - **8B thinking** uses `src/exp/thinking.py` unchanged, with `--model 8B` and the 8B t*. The subset is the 8B original-pool strata (a0 wrong N 145, right N 1310; from a local dry run, which reads initial answers only). Chunk 0 has 150 items (75/75), chunk 1 has 145 (70/75). The chunk rule, the deadline rule and the smoke check are those of D7.
  - `analysis/p4_thinking.py` now reads the stratum sizes N from the runs' `subset.json` instead of a 4B constant (149 / 1306, the same values for 4B), and takes `--model`.
- **Why.** These blocks are in the manifest. They became fundable when the cap was raised.
- **Expected effect on conclusions.** None on the confirmatory family (4B).
  - 8B and 1.7B results are replications with CIs (prereg §11).
  - With the expansion, their D-nat estimands pool the expansion as on 4B. The original-pool estimates remain as a secondary scope.
  - The probe-order deviation cannot leak test information: selection reads dev activations only.
  - Thinking is exploratory.
- **Test results seen.**
  - The 8B Tier 1 per-block estimates (reports/04_p4_8B_tier1.md) and 3 hand-checked 8B Exp 0 rows.
  - The 1.7B Exp 0/1 rows (metrics only; no estimates yet).
  - None from 4B thinking test output (still running), from 8B or 1.7B expansion runs, or from 8B Exp 2.
- **Hashes** (sha256, before launch): `cloud/p4g_sequence.sh` `4cea14ff4b4394bc396f302d804210005d98236be1c6b06ebf55c13d2703fc8d`; `analysis/p4_thinking.py` `a88099a8adf2d070785c84b18f7af6a1b173da225b659d2f8bbe5854d0152e53`.

## D9 — 2026-11-06 — Blocks 9 (Exp 1C + robustness) and 10 (Exp 4) implemented after the freeze

Both blocks are exploratory (prereg §1), and both became fundable when the cap was raised (D8). Their code was written by subagents under my brief: new files only; no file in the prereg §13 hash table changed (checked by sha256); no test-split output. I reviewed it and fixed the choices below before any of their test output exists.

- **Block 9 (`src/exp/channel.py`, `src/gen/render_channel.py`, `analysis/p4_channel.py`).** 4B, original-pool D-nat test, a0 = the fixed P1 non-thinking answers. 15 chunks of 100 in the seeded order.
  - **Exp 1C (H1c), on all 1455 items, greedy.** The F3(0.9) verdict goes through three channels:
    - tool response: user[q + label], assistant[a0], tool[the unchanged F3 verdict], user[instruction], rendered by the Qwen3 template, with no tool definitions and no tool call;
    - plain user message: the main F3(0.9) rendering, regenerated in this run;
    - prior assistant turn: a0 + " \n\n" + verdict, then the next user turn holds the instruction only. The leading space keeps a0's last token unchanged; a plain "\n\n" changed it on 165 of 364 dev items.
    - Plus a plain-user F3(0.7) cell, so that H1c's comparison Δacc(0.9 − 0.7) comes from the same run.
    - a0's tokens are identical across channels on 364/364 dev and 1454/1455 test items (`solution_tokens_identical`). The one exception, a truncated a0 ending in whitespace, is kept, and a sensitivity analysis drops it.
    - Wording hashes: main `229f9167…` (= prereg), channel `9fd6b90d…`.
  - **Sampled robustness (design "Sampled robustness").**
    - Subset: the first 300 items in the seeded order (chunks 0–2).
    - Decoding: 3 seeds, T 0.7, top-p 0.8, vLLM defaults otherwise. This is the literal design wording; the Qwen card would add top-k 20.
    - Conditions: the Exp 1A conditions B0, B1, F0, F1, F2, F2-scratch, F3 × 4, F5 at the 4B t*, and F3-perm × 3. F4 is D-inj only and is excluded.
    - Seeds come from `h("robust-seed-v1", item, s)` and are paired across conditions. Conditions with identical verdicts share one generation.
  - **Mask robustness (prereg: verifier seeds 1–2 for robustness).** The same 300 items, greedy, F3 at all p levels under verifier seeds 1 and 2, with seed 0 as an in-run reference.
  - F2-scratch already ran in the main runners, so it costs nothing here.
  - **Volume:** 12,716 generations on test (dry run), against 22,100 in the manifest.
  - **Analysis.** `analysis/p4_channel.py`, frozen `Boot` (10,000 resamples, seed 0), estimates and CIs only:
    - H1c: the channel range bootstrapped as a statistic, the pairwise channel differences, Δacc(0.9 − 0.7), and range − Δacc. The report notes that the range is biased upward.
    - Sampled robustness: the H1a/H1b/H1d analogues (p6 estimand definitions), each beside the greedy main-run value on the same items.
    - Mask robustness: the H1b analogue per verifier seed.
    - The sensitivity analysis described above.
  - **min_n** stays 150 (configs/blocks.yaml). Chunks 0–2 complete every component, so a stop at or after chunk 1 is a pre-registered partial sample for Exp 1C and robustness alike.
- **Block 10 (`src/patch/`, `analysis/p4_exp4.py`).** 4B, D-inj pairs with identical solution content: receiver (assistant, self), donor (user, other), at F0. Patching runs only from donor to receiver.
  - **Components:** resid_post at every layer × 8 position groups (label, turn, first 5 solution tokens, rest of the solution, post, instruction, generation prompt, last token). The tokens before the label are identical in both cells. Alignment matches the shared prefix and suffix and right-aligns the middle.
  - **Metrics:**
    - lp_gold: the teacher-forced summed log-prob of "The final answer is \boxed{gold}." placed at the start of the revision turn. Forcing the same target in both cells keeps it comparable.
    - marker: logit("Let") − logit("The") at the last prompt position, using the prereg §6 markers.
  - **Dev stage:**
    - Attribution patching on 396 dev pairs (16 excluded, as in Exp 3).
    - Real patching of the union of the top 20 per metric by |mean attribution|. For the marker metric, the final block is not selectable, because its effect at the last position equals the total effect by construction. It is still reported in the attribution table.
    - k rule: the components whose dev 95 % cluster-bootstrap CI excludes 0 are eligible; k = min(5, max(1, #eligible)), ranked by |effect|.
    - The output is a `frozen.json` with the components, the rule, and the hashes of the patch, render and steer code. It is committed before the test stage.
  - **Test stage:** 200 pairs (2 chunks), from the 1,547 eligible. It refuses to run unless the frozen file is committed, unmodified and sha-matched, and the code hashes match.
    - A component replicates if its test mean has its dev sign and its CI excludes 0. There is no Holm correction (exploratory).
  - **Invariant:** a self-patch reproduces the baseline within 1e-3. On 1.7B dev the error is 0.0 for each of 224 components and for all of them together.
- **Tests.** `tests/test_channel.py`, `tests/test_p4_channel.py` and `tests/test_patch.py` give 33 passed on 2026-11-06. The full suite gives 188 passed, 5 skipped.
- **Expected effect on conclusions.** None on the confirmatory family. H1c and Exp 4 are exploratory. The robustness runs qualify the greedy H1 estimates without replacing them.
- **Test results seen.** As in D8, plus nothing from either block.
- **Hashes** (sha256): `src/exp/channel.py` `14f9da34…008c63`, `src/gen/render_channel.py` `eb0cac88…d5b52`, `analysis/p4_channel.py` `dca77f35…a1761`, `src/patch/core.py` `7f49e201…765c1`, `src/patch/select.py` `b8d8f386…d3559`, `src/patch/exp4.py` `ef4784c0…80965`, `analysis/p4_exp4.py` `6ada5947…f7f74`.

## D10 — 2026-11-06 — Block 11 (4B streams, H4c) implemented after the freeze; the p4h sequence

H4c is exploratory and descriptive: the unit is the trajectory, with 3 order seeds (prereg §1, design Exp 5 "Persistent streams"). A subagent wrote `src/ttt/streams.py` under my brief (new files only; no §13 file changed). I fixed these choices before any stream output existed:

- **Stream items.** The first 200 items of the Exp 5 per-item test pool (143 GSM8K, 57 MATH-500). The design names no split; this matches the Exp 5 arm. At each step the **current** model generates a0 and the self-verdict (the endogenous loop); P1 initial answers are not used.
- **Evaluation set.** The first 100 original-pool D-nat dev items in the seeded dev order, excluding the KL/spillover set (78 GSM8K / 22 MATH-500; 17 overlap the 60 Exp 5 lr-tuning items).
  - The 4B probe was trained on dev items, so probe scores on this set are in-sample for the base model. Probe–verbal agreement is descriptive only.
  - t* and the 0.5 probe cut are applied unchanged to updated models and to log-in-context prompts.
- **Seeds.**
  - order: `h("stream-order-v1", s, item)`;
  - LoRA A: `h("stream-lora-A-v1", s)`, shared across a seed's conditions;
  - random increments: `h("stream-random-v1", s, k)`;
  - samples: `h("stream-sample-v1", item)`, shared across trajectories and checkpoints.
- **Update rule (departs from the per-item arm).** A is fixed and only B is trained (fresh AdamW, 4 steps per item). The change in B is rescaled so that each step's increment ΔW_t − ΔW_{t−1} = (B_t − B_{t−1})A has norm exactly τ = 0.5 (frozen τ, lr 1e-3; frozen anomaly rules).
  - Training A as well would make the increment norm nonlinear in the rescaling.
  - The random condition draws a Gaussian change in B, rescaled to τ.
  - The total adapter norm is logged at every step.
- **Context-memory condition.** The same verdict source as the self stream; a 4,096-token log budget, oldest entries dropped first; worded like the Exp 5 context template. No parameter update.
- **Budgets.** 2,048 new tokens for every generation, a0 included (the "Exp 5 served generations" budget; the P1 a0 budget was 4,096).
- **Metrics,** every 25 steps:
  - greedy accuracy;
  - distinct answers among 4 samples (T 0.7), grouped by `equivalent`;
  - verbal s, P(yes)_norm and the share at or above t*;
  - probe ≥ 0.5 vs verbal-verdict agreement, plus Spearman.
  - Exact definitions are in `summarize()`. No per-step KL or spillover.
- **Execution.**
  - The 12 trajectories run in lockstep: one vLLM multi-LoRA call per step, with HF on the same GPU (vLLM `gpu_memory_utilization` 0.6).
  - The state is written at every step, so a stopped run resumes from its last committed step under the same run ID (code commit and config hash checked).
  - A budget stop leaves 12 equal prefixes. That is a descriptive partial result, reported by its number of steps.
  - B and the logs are kept at every evaluation checkpoint (≈ 0.7 GB, listed in `artifacts.json`).
- **Untested path.** vLLM and HF have never shared one process on the A100. The dev smoke in `cloud/p4h_sequence.sh` checks it before the test run: step-0 invariants, increment norm within 1 %, row counts.
- **Estimate.** About 6 GPU-h central, 12 worst case (subagent's estimate from the P0 and Exp 5 rates), against 16.6 in the P0 price.
- **p4h sequence.** Order: Exp 4 dev → Exp 1C/robustness → Exp 4 test → streams.
  - The Exp 4 test stage runs from a second worktree at the commit that holds the dev stage's frozen.json, announced in `/home/research_state/exp4_frozen.json`.
  - The blocks before streams fail softly, so one failure does not stop the others.
- **Expected effect on conclusions.** None on the confirmatory family. H4c is descriptive.
- **Test results seen.** As in D9.
- **Tests.** `tests/test_streams.py`: 15 passed.
- **Hashes** (sha256): `src/ttt/streams.py` `1f9d78ed…3e0b`, `cloud/p4h_sequence.sh` `ab259374…2876`.

## D11 — 2026-11-06 — Post hoc description of unclosed think blocks in the 4B thinking subset

- **What.** `analysis/p4_thinking_unclosed.py` (sha256 `abdb3647ae085cb2d8662bdf15f10242422bf0aa3617c4b5bdf9db9146e1b5c0`) was written after I had read the 4B thinking estimates (reports/04_p4_4B_thinking.md).
  - F1 in thinking mode had 15.2 % unparsed outputs; every other condition had 0 %.
  - Hand check: each of those outputs stops (end-of-turn token) inside the think block that the "Wait," prefill opens. The answer is written there, with a boxed result, and `</think>` is never emitted.
  - The script counts such outputs per condition, and how many of them hold a boxed answer that the frozen scorer would accept if the think block were read as the answer.
- **Why.** F1's thinking-mode accuracy and the F2 − F1 contrast depend on this prefill behavior. A reader needs its size to interpret them.
- **What stays primary.** The pre-specified D7 rule: an unclosed think block is unparsed, which counts as wrong. The estimates in reports/04_p4_4B_thinking.md are unchanged. This description is post hoc and is labeled so wherever it is cited.
- **Expected effect on conclusions.** None on the confirmatory family. It qualifies the exploratory F1 and F2 − F1 thinking estimates.

## D12 — 2026-11-07 — Post hoc description of the collapse of the self streams

- **What.** `analysis/p4_streams.py` (sha256 `fdc3579de34f50d1333ee061298875321c3d2081803ea5f6a261feb7c4a4808f`) tabulates the run's own summaries (`eval_summary.json`, `steps.jsonl`) as D10 specifies. Its last table is post hoc: I wrote it after the run log showed the self streams' evaluation accuracy near 0 at k = 25 and 50.
  - It defines a collapse onset: the first step of 10 consecutive truncated a0. A single truncation is not enough, because hard items truncate in every condition.
  - Before the onset, it counts the applied updates by the loss the verdict selected (NLL or unlikelihood) and by a0's true correctness.
  - It also gives the mean a0 length per 25-step bin.
- **Why.** A collapse to 0 could be a bug or an effect of the specified rule, and the reader needs to know which. Checks so far:
  - every invariant passed;
  - random increments of a larger total norm leave accuracy at the base level;
  - hand checks show the rule doing what D10 says. An a0 with a correct boxed answer and s below t* received an unlikelihood update; after the onset, outputs loop until the token cap and carry no boxed answer, so they get no further updates.
  The table describes what preceded the collapse. It does not test a mechanism.
- **Also observed, not explained.** One ext seed (s2) ends at 0 % accuracy in a different way: in late steps a0 stops early, without an answer (unparsed). This is reported as observed.
- **Expected effect on conclusions.** None on the confirmatory family. H4c stays descriptive; this qualifies how its self and ext trajectories are read.

## D13 — 2026-11-07 — Exploratory P6 analyses written after the confirmatory results; what was not run

- **What.** `analysis/p6_exploratory.py` (sha256 `0674de848671df29f0aaf0767c5ac3d64ff1ab630453b56d37a59d781564ee64`) computes the prereg §3.3 exploratory quantities that no earlier script computed: F2-scratch and its contrasts; the H1b secondary estimate F3(0.5) − F0 and the adjacent-level differences (prereg §3.1); the 1B role × label cells, interaction and diagonal at F0, F1 and F3(1.0), and on D-nat; the alternative-label subset; the Tsui cell; Exp 0 valid rate, accuracy and gain ratio; and H2a comparators (answer length; the stratum's dev accuracy; probe and verbal per stratum and per MATH level).
  - I wrote it on 2026-11-07, after reading the confirmatory output (`reports/06_confirmatory_4B.md`).
  - The definitions follow the prereg wording. Where the prereg names a quantity without a formula, the docstring states the one used:
    - the interaction is (u/o − u/s) − (a/o − a/s);
    - the diagonal is (user, other) − (assistant, self);
    - the gain ratio is computed for F3(p) with p > 0.5 only.
- **Not run, with reasons.**
  - H3c (role/label decodability at content tokens): needs activations under the 1B prompts, which were never cached. Computing them needs the run host, and no GPU budget is planned.
  - The H2a cross-fitted secondary analysis and the mean token log-prob comparator: per-token log-probs of a0 were not stored in P1.
- **`analysis/p6_results.py`** gathers the generated files into `reports/06_results.md` with no new statistics. It recomputes Holm over the sensitivity runs' raw p with the frozen `stats.holm` to show whether any member changes significance; that line is descriptive.
- **Expected effect on conclusions.** None on the confirmatory family. Every quantity here is exploratory and is labeled so.

## D14 — 2026-11-07 — Follow-up experiments after the first draft (Group 1): lore QA, stated reliability, H2a comparators, H3c

- **What.** New exploratory runs after the confirmatory analysis, on 4B. All are outside the Holm family, labeled exploratory, and use new run IDs (`fu-*`). Liam delegated the follow-ups on 2026-11-07 (DECISIONS). Plan: `docs/planning/next_experiments/plan.md` Group 1. Blocks: `configs/blocks.yaml` priority 13. Run-host sequence: `cloud/fu1_sequence.sh`.
  - Lore QA, a non-math task. Pre-registered in `docs/planning/lore_qa/prereg_addendum.md`, frozen before any lore test output; estimation only.
  - Stated reliability F3s(p), `src/exp/stated_p.py`. The verdict wrapper states p ("a checker whose judgments are right {p}% of the time"); verdicts are the frozen F3(p) draws, so results pair with the main F3(p) rows. Review #2.
  - Teacher-forced a0 log-probs and 5-sample consistency (`src/gen/h2a_extra.py`), the H2a comparators D13 could not run. Plus a dev-fitted combined model. Review #6b.
  - H3c decodability of role and label at content tokens (`src/probe/h3c.py`), not run before (D13).
- **Analysis scripts, written before any of these outputs existed:**
  - `analysis/lore_results.py` (sha256 `362f6bdb28d6155d5b9751d101c3cdcc97342eaf96a70208d71c26b1d8b212fb`);
  - `analysis/fu_group1.py` (sha256 `505c2b3be0f18045b7f6024e93a4bd00fb05abefcc1c0a5175834a8b77c3df65`), checked on stand-in data: stated identical to unstated gives exactly 0, with no verdict mismatches.
- **Not run.** The H2a cross-fitted secondary analysis. It needs the test activations, which stay on the run host, and it adds conditional CIs only.
- **Expected effect on conclusions.** None on the confirmatory family. These runs can change interpretation:
  - whether answers follow verdicts when p is stated;
  - whether the probe's advantage over verbal holds against cheaper confidence signals;
  - whether the self-correction pattern appears outside math.

## D15 — 2026-11-07 — A second model family (Tulu 3 8B), Tier 1 replication

- **What.** `allenai/Llama-3.1-Tulu-3-8B` (key `T8B`; revision `666943798adbde0b1aff34626007e26986a3c107`). It is ungated and under the Llama 3.1 license, per the HF API on 2026-11-07. It runs Tier 1 at the 8B volumes, and its results are replications with CIs, not tests.
  - Liam's delegation; `plan.md` "family2-tier1" (situation §6, review §7: "only one model family").
  - `cloud/fu2_sequence.sh` applies the prereg §11 adaptation rule by script on T8B dev, before any T8B test run: t* through `src/score/tstar.py`, which reproduces the frozen 4B, 1.7B and 8B t* exactly (tests/test_fu_tools.py), and the probe through `src/probe/train.py`. Wording, budgets, scorer and verifier streams are the 4B ones.
  - Feasibility stop: if more than 10 % of T8B's dev initial answers fail to parse, the sequence stops and family2 is reported as not run (feasibility).
  - Analysis: the existing spec-driven scripts (`analysis/p4_block_estimates.py`, `analysis/p6_exploratory.py`) with a T8B spec.
- **Choice of model.** Gated models (Llama 3.1 Instruct, Gemma 2) would need Liam to accept terms. Among ungated candidates:
  - Mistral-7B-v0.3's template rejects consecutive user turns, which the Exp 1B user cell needs;
  - OLMo 2 7B has a 4K context, too short for a0 plus revision;
  - Granite 3.1's default system prompt includes the current date.
- **Model-specific differences, measured and reported, not adjusted:**
  1. The Tulu template emits no BOS. vLLM adds one when it tokenizes a prompt; the HF paths (verbal P(yes), activations) now add it through `src/runlib.py needs_bos`. That is False for Qwen3, which has no BOS token, so Qwen runs are unchanged.
  2. A user turn ends with a newline, an assistant turn with the EOS token. For 1,688 of 2,047 D-inj items (82 %), the last token of the reviewed solution differs between the role cells (the newline merges into it); every other solution token is identical. The T8B role contrast therefore includes a one-token boundary difference that the Qwen minimal pair does not have.
- **Expected effect on conclusions.** None on the confirmatory family. The result tells whether the 4B patterns appear in another family. A difference is not evidence against the 4B results: a different model also differs in training data and post-training.

## D16 — 2026-11-07 — Group 2 follow-ups: stream controls and structured steering controls

- **What.** Two exploratory runs that explain earlier exploratory results. Neither touches the confirmatory family. Run by `cloud/fu3_group2.sh` in the second follow-up session, after family2 (`cloud/fu2_session.sh`). Liam's delegation (DECISIONS 2026-11-07).
  - **Stream controls (review #8).** `src/ttt/streams.py` gains two conditions, off by default:
    - `nll`: an NLL update at every step, whatever the self-verdict: as many updates as self, none of them unlikelihood;
    - `matched`: the frozen run's realized self-stream verdicts per order seed, permuted by `h("stream-matched-v1", seed)`. It has the same share of "incorrect" verdicts, 96.8 % over the 600 self steps, but they are unrelated to the stream's own correctness.

    Everything else is the frozen 4B-streams setting: 3 order seeds × 200 steps, the same items, evaluations, tau, lr and t*. The base evaluation (k = 0) must reproduce the frozen run.
    - I first wrote `self-nll` (self-verdicts, no update on an "incorrect" verdict). I replaced it before any run: with 97 % "incorrect" verdicts it would update on about 3 % of steps and say nothing about NLL training.
  - **Structured steering controls (review #7).** `src/steer/structured.py` runs Exp 3 at the frozen main setting (layer 18, role, α 0.2) with three control directions computed on dev:
    - the first principal component;
    - the gsm8k_sc − prm800k_sc contrast;
    - the role construction with item-level random signs.

    Plus main and unsteered, on the frozen items, order and batch layout. The per-position perturbation norm is the same for every arm. Main and unsteered must reproduce the frozen Exp 3 run token by token.
- **Analysis:** `analysis/fu_group2.py`, written before any output. `analysis/p4_streams.py` now reads the condition list from the run's config; its frozen report regenerates byte-identical.
- **Expected effect on conclusions.** None on the confirmatory family. These can change how the self-stream collapse and the H3b reversal are worded. If `matched` collapses like self and `nll` does not, the collapse follows from the volume of unlikelihood updates, not from what the self-verdict says. If the structured controls reduce W->R as much as main, the H3b drop is not specific to the role direction.

## D17 — 2026-11-08 — Post hoc analyses not logged before; a correction to D16; the follow-ups in the draft

- **What, 1: the no-GPU re-analyses (logged late).** `analysis/fu_reanalysis.py` (sha256 `6e6e0da0a68322ed0a21a2891e01dca850ed8de6df8ee99220e04b51a871eb27`, committed 2026-11-07 in `3e6dcef`) re-analyses stored 4B test outputs (`reports/fu_01_reanalysis.md`, plan §1):
  - fix and damage per 1A condition;
  - H1b split by verdict truth;
  - output status per Exp 3 arm;
  - one role × label × feedback decomposition on D-inj;
  - scorer disagreement per condition;
  - a dev-fitted H2a comparator;
  - a re-check policy at fixed budgets.

  It was written after the confirmatory and exploratory results had been seen, like D13, and it should have been logged here when it was written. It was not; I found the gap on 2026-11-08 while updating the docs.
- **What, 2: the Tulu verbal check.** `analysis/fu_verbal_check.py` (sha256 `76990a060e07853ef7491220e1fa2b02cd75da0000f1dac4b9a4e6a5d9c8c207`, `reports/fu_04_verbal_check.md`). It reports, for Tulu-3-8B and Qwen3-8B on dev and test, the AUROC of P(yes)_norm and the yes/no probability mass at the verdict position. It was written after the T8B H2a estimate (verbal AUROC near 0.5) had been seen, to tell a weak readout from a broken one.
  - Before it existed, I had computed a dev AUROC by hand on the 364 original-pool dev items and written that value into `docs/evidence_ledger.md`. That broke rule 1. The ledger now cites the script's output.
- **Correction to D16.** D16 says the `nll` stream has "as many updates as self". That holds by design only before the self stream collapses. Self applies no update once its outputs carry no answer span, so it applied far fewer updates in total (`reports/fu_03_group2.md`). `nll` updates at every step. The paper describes nll as an update at every step with no unlikelihood updates.
- **The follow-ups in the draft.** On the author's request (2026-11-08; quote omitted), every D14–D16 result is in `paper/first_draft.tex`: Section 11 ("Results V") and Appendix C, with stale statements elsewhere corrected. Numbers come from `analysis/p7_results_tex.py` and `p7_figures.py` (CLAUDE.md rule 1).
- **Expected effect on conclusions.** None on the confirmatory family. Every item here is exploratory or descriptive.

## D18 — 2026-11-08 — Second review (GPT): full-cost and re-check analyses, model-to-model differences, wording; planned validation

- **Trigger.** Second review, `review_and_suggestion.md` (2026-11-08). The author asked to organise the current experiments accordingly and to run a new validation (quote omitted). Triage and plan in `docs/planning/validation_v2/plan.md`.
- **New analyses on test outputs already seen (exploratory):**
  - `analysis/fu_recheck_cost.py` (sha256 `42a87dbfeebce4e593a3d2bff3ebb07132d04c18c826ce45e35f33f75c253307`), → `reports/fu_05_recheck_cost.md`:
    - full generation cost per condition, counting both turns of F2 and F2-scratch. The earlier "mean out tokens" column of `reports/fu_01_reanalysis.md` counted only the final turn and must not be read as the cost of F2.
    - signal costs;
    - selective re-check with F2 or F0 under budgets for 11 rankings. The dev-fitted rules use cross-fitted dev probe scores (the frozen probe's layer and C).
  - `analysis/fu_crossmodel.py` (sha256 `7019085a29956d6bf461a72499053fa7ff656c8012640329a823554180d1ee18`), → `reports/fu_06_crossmodel.md`: direct differences between models on common items, with joint cluster bootstrap.
  - Both scripts were written after the review and after the earlier test results had been seen. Their features, budgets and rankings were fixed in the scripts before their first test output. One label was changed after the first output: the "oracle" ranking became "gold wrong-first", because it is not an upper bound. That change is a label only; no number changed.
- **Wording in the paper** (review items A–G):
  - lore: name-only primary, all-items as sensitivity;
  - consistency renamed "agreement with a0";
  - Tulu's verbal readout stated narrowly, its role contrast called role-plus-template;
  - the D-inj table renamed a role × label × feedback-regime factorial, not a decomposition;
  - steering and streams claims narrowed;
  - H3c maxima called exploratory peaks;
  - the [num:] placeholders removed (the hand-check count, local GPU hours);
  - the deviation summary and the 8B timing sentence corrected.
- **Planned, not yet run** (the next cloud session):
  - H3c regrouped by canonical_id;
  - a dev-only check of Tulu's verbal readout under other wordings;
  - an independent validation block on a public non-math task.

  These will run under an addendum frozen before any of their test outputs exist.
- **Expected effect on conclusions.** None on the confirmatory family. The re-check analysis changes how H2a is used in the paper: a better correctness ranking did not give a better re-check choice.
- **D18, continued (2026-11-08): what ran and what it showed.**
  - **Session 33 (A, dev only; 2.92 h, $4.41).**
    - Readout variants on math dev (`reports/fu_07_verbal_variants.md`): the fixed rule selected no variant for T8B. Its best alternative was +0.048 AUROC, against a margin of 0.05. T8B's within-dataset AUROCs exceed its pooled AUROC.
    - H3c regrouped by canonical problem (`reports/fu_08_h3c_regroup.md`): no layer moves by more than 0.006.
    - MMLU-Pro dev pilot (`reports/v2_01_mc_dev.md`).
  - **The addendum** (`docs/planning/validation_v2/addendum.md`) was frozen after reading the dev pilot (commit `9221b24`), before any test output of the task.
  - **Session 34 (B, test; 4.41 h, $6.64).**
    - The frozen analysis, `analysis/mc_results.py` (unchanged since the freeze), gives `reports/v2_02_mc_results.md`.
    - Under the addendum's reading rules:
      - P1: F2 shows no net change resolved on 4B and a net loss on T8B; F1 is a net loss on 4B.
      - P2: the probe-order advantage over verbal order is "not shown" for both models at 10 % and 20 %.
      - P3, descriptive: agreement order adds a little over verbal order, at a token cost above F2 on every item.
  - **Pre-syncs.** Two runs of each model were synced by hand while session B was still running, to shorten the final sync; the hashes were checked as usual.
  - Every result here is exploratory and outside the Holm family.

## D19 — 2026-11-08 — Third review (GPT): selective re-check against re-checking everything, an MMLU-Pro scoring audit, repositioning of the paper

- **Trigger.** Third review, `review_and_suggestion.md` (2026-11-08, Liam's file, not committed). It asks for no new experiments: no new generations, no cloud. The author told the agent to proceed with the plan `docs/planning/final_v3/plan.md` (quote omitted).
- **New analyses on existing outputs (post hoc, exploratory; outside the Holm family and the addendum's primary comparisons):**
  - `analysis/fu_recheck_vs_all.py` (commit `b2e651b`) → `reports/fu_09_recheck_vs_all.md`.
    - What it adds: for every ranking and every budget already in `fu_05` (math, seen test split) and `v2_02` (MMLU-Pro, addendum test split), the paired difference "selective − re-check every item (ALL)", and the share of initially wrong items among the selected ones.
    - How the rankings are rebuilt: by importing the source scripts' loaders and repeating their ranking code. The script asserts that it reproduces their stored fixes, damages and point estimates for every cell, and it does.
    - Its reading rule is fixed in the script. No budget, ranking or model is selected.
  - `analysis/fu_mc_parse_audit.py` → `reports/fu_10_mc_parse_audit.md` (+ `analysis/out/fu_10_mc_parse_audit_samples.md`).
    - Status of every MMLU-Pro output under the frozen scorer, and the final-output status among W→R and R→W.
    - Outputs with no letter are put into classes.
    - A sensitivity scorer reads one class as a letter. The frozen `mc_results.analyse()` is then rerun for P1–P3.
  - **The audit rule was corrected after the first run.**
    - Rule v1 (commit `bebeb7d`, fixed before the first run, after the author had read five unparsed Tulu F2 revisions) read a box that lists several options ("A, I, G, H") as its first letter. Reading the samples showed this.
    - Rule v2 (commit `f793a71`) puts such boxes in a separate class (m), which stays without an answer, and does not read "e.g." or "i.e." as a letter.
    - Both versions are reported. The frozen scorer stays primary.
- **What they showed.**
  - **fu_09, Tulu-3-8B on MMLU-Pro with F2:** every ranking, random included, is above ALL at the addendum's 10 % and 20 % shares (at larger shares all but the transferred math probe), and none is above KEEP at any share. Re-checking fewer items only does less of a harmful action.
  - **fu_09, math:** no cheap ranking is above ALL. Only rankings that use agreement with a0 are, at shares of 20 % and more.
  - **fu_09, Qwen3-4B on MMLU-Pro:** only rankings that use agreement with a0 are above ALL.
  - **fu_10:** the frozen conclusions are unchanged under both sensitivity scorers. Tulu's F2 net loss remains a net loss, and the P2 readings do not change. Of Tulu's F2 R→W damages, about a third are revisions that do not commit to one option: boxed text, several letters, or a letter with a qualifier. For 4B, most fallback letters come from outputs that hit the token cap.
- **Paper.**
  - Repositioned as a controlled evaluation of revision decisions, following the review's §3. Title, abstract, introduction and the order of sections change.
  - The framework (Claim 7) moves from the main line to a short motivation and an appendix.
  - The confirmatory family stays complete in the main text.
  - Wording follows the review's §4.
  - Three close works (Zhang 2609.35832, Kumaran 2604.22271, Yang 2412.19513) are verified and compared. Holm, DeLong and Schuirmann are verified and cited.
- **Expected effect on conclusions.**
  - None on any confirmatory outcome or addendum reading.
  - The paper may no longer say that selective re-checking is "better" than re-checking everything where ALL itself loses accuracy. It must also report that comparison against KEEP.
  - The Tulu F2 loss is reported together with its composition.
- **D19, continued (2026-11-08): the skeptical review of the manuscript** (`reports/07_paper_review.md`, third round).
  - It asked for checks on the repair side of the format confound, for intervals on the within-pool rates, and for an
    oracle reference.
  - New post hoc script `analysis/fu_ms_robustness.py` (committed before its full run, `d231600`) →
    `reports/fu_11_ms_robustness.md`. Stored outputs only.
  - `analysis/fu_recheck_vs_all.py` was refactored into reusable builders; its output is byte-identical.
  - **What it showed:**
    - On math, 51 of F2's 127 repairs come from initial answers truncated at the token limit (scored wrong; a
      revision that finishes them counts as a repair). On the items whose a0 gave an answer, F2's math gain is about
      a third of the full-population gain, and F0's and F2-scratch's vanish. The cheap rankings' 10 % pools on math
      are 42 % truncated a0.
    - On MMLU-Pro, the F2 and F1 P1 readings are unchanged on answered items, but Qwen3-4B's small F0 gain was mostly
      completion. On answered items the Qwen3-4B P2 difference at 10 % is resolved in the probe's favor; at 20 % its
      interval touches zero.
    - The gold reference's headroom with F2 on Tulu is under a point, below F2-scratch on every item.
    - Within-pool rates now have CIs. On Tulu, the probe's pool differs from the verbal pool in repair, not damage.
  - **Expected effect on conclusions:**
    - No confirmatory outcome changes. The confirmatory family already had an a0-ok sensitivity analysis, under which
      every outcome held.
    - The manuscript's math repair statements and its probe-vs-verbal decision claim are narrowed.
    - The format-repair effect on math is reported as a finding.

## D20 — 2026-11-09 — ARR round for the ACL version (tiers A, B, C1); the historical math agreement signal reads gold structure

- **Trigger.** The readiness review of the ACL manuscript (`paper/1区/acl2027/reports/readiness_review_claude_2026-11-09.md`, with an independent ARR-style review by a subagent). The author chose the offline + tier B + upgraded-inference option (2026-11-09; quote omitted):
  - tier A, offline replays of stored outputs;
  - tier B, cheap new runs;
  - C1, thinking-mode re-answer used as an escalation operator.
- **What changes.**
  - D19's "no new experiments" is lifted for the ACL/ARR version only. The TMLR v4 and journal manuscripts are not changed.
  - All new analyses and runs are post hoc and exploratory. They are outside the Holm family and the addendum's primary comparisons.
  - Their rules are fixed in `docs/planning/arr_round/plan.md`, committed before any new test-split analysis runs. Its hash goes in STATUS.md.
  - New generation runs get new run IDs. No existing run is superseded by them.
- **Gold dependence of the historical math agreement signal.** Found during the ACL revision; not logged until now.
  - In the consistency stage, `src/gen/h2a_extra.py` (lines 115–116) extracts a0's answer and each sample's answer with `score(text, gold)`.
  - When the gold has several elements, `score` joins the final boxes into one set (`src/score/scorer.py`, lines 275–277).
  - So the stored `consistency` value (agreement with a0) can depend on the gold's structure, and the signal is not strictly gold-free.
  - Size: the ACL revision's gold-blind reconstruction (`paper/1区/acl2027/analysis/math_vote.py --same-rule`) changes 5 of 5,052 agreement scores. The 10 % and 20 % selected sets differ by 2 and 4 items.
  - MMLU-Pro agreement is not affected: `src/exp/mc_run.py` compares letters from `src/score/mc.py`, whose extraction never reads the gold.
  - Affected run: `fu-h2a-consistency-test-4B-aac4cc6`, and every result that ranks Qwen3-4B math items by agreement:
    - `fu_05`, `fu_09` and `fu_11`;
    - the manuscripts that report them: the TMLR v4 appendix, the journal manuscript and the 1区 manuscript. They use the historical signal without saying so.
  - Fix:
    - The ARR analyses use only the gold-blind reconstruction.
    - The historical outputs stay as they are: they record what was run.
    - Whichever manuscript is submitted gets a one-sentence disclosure and, where a printed number changes, the reconstructed value.
    - TMLR v3 and v4 are not edited in place: a new version goes in a new directory.
- **Expected effect on conclusions.**
  - None on the confirmatory family or the addendum, neither of which uses agreement.
  - Agreement-ranking estimates on 4B math move by at most what 2 or 4 membership changes allow. The ACL manuscript already reports the reconstruction.
- **Test results seen when the change was made:** all existing test-split results (the study is complete). No new test analysis has run yet.

## D21 — 2026-11-11 — ARR Amendment 8: uniform thinking budgets compared at equal total tokens; the MMLU-Pro reading changes

- **Trigger.** A readiness review of the submission draft `paper/1区/acl2027/final_draft` (the author asked for a readiness check of the submission draft, then told the agent to start the changes; quotes omitted). The review noticed that the Amendment 7 analysis compares each uniform thinking budget with selective thinking at the nearest cost *after a0*, although a policy that thinks on every item does not need a0.
- **What changes.**
  - Plan Amendment 8 (commit 28e7771) defines the comparison at equal total generated tokens, with a0 charged to the selective policy alone. `analysis/arr/c1_budget_total.py` (commit a73a117) computes it from stored labels. No model output is new.
  - The amendment is not blind. The review had computed the mathematics point estimates ad hoc before the amendment was written; the amendment, the new table's caption and the Limitations section say so.
  - Manuscript v2 quotes the equal-total comparison: the §4.4 paragraph on uniform budgets, one sentence each in the abstract, the introduction and the Discussion, the last clause of the Conclusion, a Limitations sentence and a new appendix table. The Amendment 7 table (nearest points after a0) stays.
- **Results** (copied from the generated `reports/arr_c1_budget_total.md`).
  - Mathematics, Qwen3-4B and 8B: at budgets of 512 tokens or more, verbal-selected thinking is 0.89 to 1.58 pp ahead of the uniform budget, and every unadjusted interval excludes zero. The after-a0 comparison had given 1.90 to 4.41 pp. At a budget of 256, a0 leaves no room to escalate any item.
  - MMLU-Pro, Qwen3-4B: the uniform budget is ahead at every budget, by 1.01 to 2.34 pp against the verbal gate and 1.48 to 2.75 pp against the probe gate. A 256-token budget gains 1.62 pp over KEEP at about a0's own cost (777 against 759 generated tokens per item).
- **Expected effect on conclusions.**
  - No tested contrast changes: the comparison is descriptive and outside the §5 family.
  - The claim that selective escalation beats a uniform budget is narrowed to mathematics and to budgets of 512 tokens or more, with a smaller margin.
  - For MMLU-Pro the earlier wording ("the two stay within 1.13 pp") is replaced: the uniform budget did better.
  - The third headline changes from "a strong operator makes cheap signals valuable" to "can make".
- **Test results seen when the change was made:** every result of the ARR round, and the ad hoc mathematics point estimates of this comparison.

## D22 — 2026-11-11 — journal revision: oracle interpretation and generation accounting

- **Authorization and timing.** Liam requested a Neurocomputing revision with as few new experiments as possible. All stored study results and the prior review were already visible. This is a retrospective correction and descriptive reanalysis, not a new preregistration.
- **What changed.** `analysis/neurocomputing_revision.py`, also shipped in the journal supplement, separates the initial-correctness oracle reference from the realised-benefit oracle bound. The latter selects the largest stored gains for a fixed selection cardinality; it is not a token-budget bound or a bound on fresh generations. The main cost figure now charges each policy for every generation it requires, including the initial answer when applicable; standalone sampling/voting and standalone thinking do not require that answer.
- **Implementation correction.** The copied `a3_dev_test.py` charged reused samples twice under threshold rules. Its correction changes token costs, not selections, accuracies, confidence intervals or the policy chosen on development data. The pre-correction code, summaries and manifests are retained under `paper/Neurocomputing/supplementary/analysis/out/neurocomputing/history/`. Numerical before/after records are generated in `revision.json`; do not infer results from this prose.
- **Expected effect on conclusions.** The original wrong-first reference must not be called a universal upper bound. The benefit bound permits more repair than that reference, especially on the knowledge task. Whole-pipeline generation accounting weakens claims based solely on incremental cost. The manuscript now explicitly limits the gate evidence to the evaluated settings and stored outputs, identifies the development-selection grid, and marks thinking-budget selection as descriptive. The contribution is a controlled comparison and diagnosis, not a new decomposition or an independently validated deployment policy.
- **Unchanged.** Original generated model outputs, correctness labels, preregistration, confirmatory family, stored hypothesis-test results and the ACL draft. The total-budget comparison from D21 is retained with its post hoc status. No model run, training or new hypothesis test was performed; no new generation run ID exists.
- **Evidence and checks.** `paper/Neurocomputing/supplementary/analysis/out/neurocomputing/manifest.json`, `revision.json`, and archived history; the build invokes the correction after baseline generation so a rebuild cannot restore the old interpretation or figure. Final scientific review: `reports/neurocomputing_final_review.md`.

## D23 — 2026-11-11 — existing cost records and retrospective threshold-transfer sensitivity

- **Trigger and status.** Author-supplied reviews requested stronger cost reporting and threshold checks; the author authorized revision without experiments. All earlier test outcomes and those reviews were already known. This is retrospective analysis, not a new preregistration. The protocol record explicitly states when it was documented.
- **Cost changes.** The journal v2 copy incorporates actual logical input counts and historical HF/P0 timing records. Exact token-prefix extraction supersedes the old nominal verbal-cache assumption. Parameterised D + alpha P comparisons preserve input, padding, KV and hidden-state assumptions; alpha is not measured latency or monetary conversion. Uniform thinking trace generation is not charged again as external input in the retained-KV scenario. Historical batch timings and study billing are not repurposed as online policy cost.
- **Threshold-transfer changes.** Canonical-problem groups define fixed folds shared by mathematics models. Thresholds use calibration-fold signals and costs only; actual held-out costs, gains and overshoots are retained without adjustment. Candidate thresholds use whole score boundaries rather than the earlier coarse share grid, so resolution and calibration change together. This does not turn the previously examined population into fresh independent validation. No new p-values are produced.
- **Expected interpretation.** The cost evidence supports a conditional resource-allocation conclusion: input processing and signal execution can change which policy fits the common budget cap and reverse an output-only advantage. Threshold replay assesses internal transfer with budget drift; it establishes no deployment guarantee. Generated reports, rather than manually transcribed estimates here, hold the numerical results.
- **Unchanged.** Original generated responses, labels, pre-registration, confirmatory family, stored test results and prior paper versions. No new generation run ID, model execution, training, hypothesis test or cloud job.
- **Evidence.** analysis/neurocomputing_cost_v2.py, analysis/neurocomputing_crossfit_v2.py and analysis/neurocomputing_figures_v2.py; journal v2 supplementary manifests and reports under analysis/out/cost_v2 and analysis/out/neurocomputing_v2. Independent reviews found and corrected equal-realised-cost language to the common cap actually used.
