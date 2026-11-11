# Pre-registration — FROZEN 2026-11-05

| | |
|---|---|
| Status | **Frozen** on 2026-11-05 after the independent pre-freeze review and its re-check ([reports/03_prefreeze_review.md](reports/03_prefreeze_review.md)). The freeze commit hash and this file's sha256 are in STATUS.md. Every later change goes to DEVIATIONS.md (§12). Departures from design v2.1: §14. |
| Date | 2026-11-05 (first draft 2026-11-04) |
| Design | `docs/research_design.md` v2.1 (2026-11-03). If this file and the design disagree on what is tested, the design wins until the freeze; after the freeze this file wins and changes go to `DEVIATIONS.md`. |
| Protocol | `CLAUDE.md` v2.1 |
| Human author | Liam (Lihan Xu). Drafted by the autonomous agent; frozen only after an independent pre-freeze review ([reports/03_prefreeze_review.md](reports/03_prefreeze_review.md)). |
| Seen so far | **Dev split:** everything (pilots, tuning, power simulations). **Test split:** the machine generated and scored initial answers a0 and computed the pre-feedback verbal P(yes) [reports/01_data.md]; the agent looked only at counts of initially right and wrong items per model × dataset [reports/01_data_tables.md], which feed the power simulations, and at the output-status counts (truncated, unparsed) that the same metrics files and data cards report [results/p1/*/*/metrics.json; reports/03_data_ext_tables.md]. Test items of the D-nat expansion: a0 and verbal P(yes) were generated and scored by the machine, with the same counts only. **No post-feedback test quantity has been computed or seen** ("Test split first post-feedback access: never" [STATUS.md]). **H2a guard:** H2a is entirely pre-feedback, and test a0 correctness and verbal P(yes) are on disk; no test AUROC, test probe score or test activation has been computed. The first test probe-scoring run and the first test AUROC computation are logged with timestamps, like the first post-feedback access (§8). |

---

## 0. Scope and wording limits (fixed)

- A frozen model receives self-critique and external feedback through the same port (context), so this study does **not** test coupling type [design §1.1].
- Every behavioral hypothesis below is an **empirical prediction motivated by Claim 7**, written so that it can fail. None is a consequence of Claim 7 [design §1.2]. No accuracy or AUROC is called a V-information measurement.
- H1d and H4a test how **self-verdict errors are allocated** relative to fixability, at fixed realized reliability. They are not tests of Claim 7 or of "external evidence" [design §3.3, §3.6]. F3-perm / perm-self is an **oracle randomized control** (it uses ground-truth strata), not a deployable verifier [design §5.8].
- The Exp 5 arm is a comparison of two ways of using the same feedback under a fixed parameter interface. The 1-bit verdict is a *minimal self-assessment signal*; the arm is **not reflexive** and is not a self-model. Conclusions are limited to this update rule [design §1.3, Exp 5].
- H3a results are prompt-label effects; a positive H3b shows that a representation correlated with role causally modulates revision, not a self-model; a null is not support for essay §一 [design §3.5].
- Linear probes measure linear accessibility, not an information ceiling. Thinking vs non-thinking is a contrast of mode and budget, not of RL training. No claims about consciousness or experience [design §1.5].
- No skeleton result is cited as a theorem; the theory-status table of design §1.4 applies.
- **H1a equivalence is relative to low revision rates.** F2 and F1 disagree on about 4.5 % of dev items, so |Δacc| cannot exceed that, and δ = 3 pp is a large share of it [reports/03_power_tier1_tables.md]. A supported H1a is reported with the discordance, W→R and R→W of both arms next to the TOST result, as "within ±δ at these revision rates". The phrases "critique adds nothing" and "critique is no better than 'Wait'" are not used.
- **"Self-verdict" (F5, perm-self, H1d, H4a) is a dev-thresholded readout of the verbal P(yes)**: "correct" iff s ≥ t*, which on 4B is P(yes)_norm ≥ 0.9999999 (§6). At a threshold of 0.5 the 4B verbal verdict calls almost every item correct [DECISIONS 2026-11-04]. The paper writes "thresholded verbal readout", not "the model's own verdict", and gives the threshold.
- **H3a sign.** A negative H3a-role or H3a-label estimate (the self- or assistant-attributed solution corrected more) is not a replication of the Self-Correction Bench blind spot and is not written as one; the diagonal comparison that replicates it is exploratory (§3.3).
- **Population.** Every D-nat headline names its population: GSM8K test + MATH test pooled as sampled (79 % MATH; §2). The original-pool, reweighted and per-stratum estimates are shown in the same table. The expansion (MATH test outside MATH-500) is the part of MATH that some public training recipes have trained on, so its contamination risk may be higher than MATH-500's; the stratum estimates and their difference are reported for that reason (§8).

## 1. Models, modes, decoding, environment

| Role | Model | Revision [configs/models.yaml] | Mode |
|---|---|---|---|
| Primary (all confirmatory tests) | Qwen/Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` | non-thinking |
| Pipeline / replication | Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | non-thinking |
| Scale replication | Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | non-thinking |

- Chat template: identical for all three, sha256 prefix `a55ee1b1660128b7` [configs/models.yaml]; full hash `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` [results/p2/4B/p2-exp01-dev-4B-f7e5011/manifest.json]. bf16 weights, no quantization [configs/models.yaml; design §4].
- Environment: Python 3.12.13, torch 2.14.0+cu130, vLLM 0.33.0, transformers 5.20.2, peft 0.21.2, nnsight 0.8.2; lockfile `configs/lock.txt` [configs/models.yaml].
- GPU: main runs on A100-SXM4-80GB; one GPU type per block, recorded in each manifest [DECISIONS 2026-11-04 (A30 entry); design §10].
- **Determinism.** All vLLM generation runs with `VLLM_BATCH_INVARIANT=1`, recorded in every manifest [DECISIONS 2026-11-03; reports/00_compute_rates.md]. Exp 5 adapter training runs with `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` [src/ttt/update.py `deterministic()`; DECISIONS 2026-11-04]. Exp 3 (HF) runs every arm, including α = 0, in one fixed batch layout [src/steer/pilot.py `generate_batch`].
- **Non-thinking decoding:** greedy (temperature 0, seed 0); vLLM engine `max_model_len` 12288 [src/exp/runner.py].

| Generation | Max new tokens | Source |
|---|---|---|
| Initial answer a0 | 4096 | [DECISIONS 2026-11-03 (a0 budget raised from 2048)] |
| Revision turn (B1, F0–F5, F3-perm, F4, Exp 1B) | 2048 | [results/p2/4B/p2-exp01-dev-4B-f7e5011/config.json `revision_tokens`; DECISIONS 2026-11-04 (corrections: revision budget)] |
| First turn of F2 / F2-scratch | 1024 | [same config, `critique_tokens`] |
| Exp 0 turns | 1024 | [same config, `exp0_tokens`] |
| Tsui replication cell | 1024 | [same config, `tsui_tokens`; reports/01_data.md] |
| Exp 3 steered generation (dev confirmation and test) | 1024 | [src/steer/pilot.py `CONFIRM_TOKENS`] |
| Exp 5 served generations (all cells) | 2048 | [results/p2/4B/p2-ttt-serve-4B-ea34050/config.json] |

Note on the revision budget: the design asks for `max_new_tokens` "frozen from the pilot (default 1024)" [design §4]. 2048 was the runner default from its first commit (218cd12), before any pilot output, and was not tuned on outcomes; it is frozen here [DECISIONS 2026-11-04 (corrections)]. Dev truncations in revision conditions were greedy repetition loops [reports/02_handcheck.md].

- **Thinking mode (exploratory only):** sampling T = 0.6, top-p 0.95, top-k 20, min-p 0, confirmed from the pinned `generation_config.json` of all three snapshots and the 4B model card (which also asks for no greedy decoding in thinking mode) [DECISIONS 2026-11-05]. Budget **16384** new tokens, chosen on 4B dev by a rule fixed before the pilot ran: the smallest of {4096, 8192, 16384} with overall truncation ≤ 5 % [reports/03_checks_tables.md §2; DECISIONS 2026-11-05]. Two paired seeds per item [design Exp thinking subset]. Truncated = unparsed = wrong, counted per condition [design §4].
- **Scorer:** `SCORER_VERSION = "1.1.0"` [src/score/scorer.py]. Rules: design §5.6 plus the conflict rule [DECISIONS 2026-11-03], math-verify without timeouts (answers > 300 chars fall back to normalized string equality) [DECISIONS 2026-11-03], and the 1.1.0 extraction changes [DECISIONS 2026-11-04]. The hash is sha256 of `src/score/scorer.py`, written as `scorer_sha256` into every run manifest [src/runlib.py `write_manifest`]; value on 2026-11-05: `40790d874fa320464b61ae2937569df455179e555678d0cbdda3fe3bb81f31e0` (re-checked at the freeze commit, §13).
- **Scorer invariant** (design §8: agreement with a second scorer ≥ 99 % on dev): **fails as written**, 97.09 % against the first second scorer and 98.44 % against v2 (the MATH reference normalization, added after the first failed) [reports/03_scorer_agreement.md]. Both second scorers share the main scorer's answer extraction, so the check covers the equivalence step only. Of the 593 v2 disagreements, 592 have the main scorer accepting and 1 rejecting; in the hand-checked random sample of 40 "accepting" disagreements every main-scorer label is right, and the 1 "rejecting" case is a strict call counted as a main-scorer error [reports/03_handcheck.md]. **The amendment was changed twice, both before the freeze** [DECISIONS 2026-11-05]: the first stated rule (≥ 50 hand-checked disagreements, all with the main scorer accepting, none wrong) fails on this evidence (40 checked from the v2 pool; one rejecting case); the rule then adopted, after the hand check, passes the invariant when the main scorer's label error located in the disagreements, at the upper 95 % bound, is below 1 % of all dev texts; it is 0.12 % [reports/03_scorer_agreement.md]. Because of this history, every confirmatory estimate is also reported with v2 labels, and per-condition disagreement rates are reported (§8).
- **HF vs vLLM** (design §6 Exp 5, §8: ≥ 98 % identical greedy outputs, else report and keep generation-based comparisons within one backend): 29.4 % identical on 4B dev, with a median per-position agreement of 99.7 % and every first mismatch at a logit gap ≤ 0.5 (bf16 near-ties) [reports/03_checks_tables.md §1]. So generation-based comparisons stay within one backend: all Exp 5 arms in vLLM, all Exp 3 arms (α = 0 included) in HF [DECISIONS 2026-11-05].
- **Prompt wording** [src/gen/render.py]: `wording_sha256 = 229f9167e91e8a01ef18fa4fe6d37b79c51f69243b9a9a66988dadbc045583cf` [results/p2/4B/p2-exp01-dev-4B-f7e5011/config.json]; the P3 runs carry the same hash [results/p2/4B/p2-exp01-dev-tstar-4B-6cd867e/config.json]. Exp 5 context template sha256 `0c0f570d8cbc60a10895a871e1992a5c0cdfd252b9b46cdc4bbd29073caa83e6` [results/p2/4B/p2-ttt-serve-4B-ea34050/config.json]. Verbal-verdict query and candidate sets [src/score/verbal.py; design §5.7]; first-token shortcut verified for all three tokenizers [reports/01_data.md].
- **Exp 1B alternative wrapper set** (design Exp 1B; fixed a priori): labels "Below is a solution to this problem that you wrote earlier." / "Below is a solution to this problem that a different model wrote earlier." (`LABELS_ALT`, 12 / 14 tokens, length-matched [tests/test_render.py]); everything else as the main set; D-inj, the first 300 test items in the seeded order, all 12 Exp 1B cells (`src/exp/runner.py --label-set alt`); `wording_sha256 = 86c71e3c9371fca5ac4858f2ffdf43b285ee9574140aa2fc6c38b10d4094e6dc`. Dev validity: no unparsed output, truncation 0.4 % vs 0.5 % for the main wording on the same items [reports/03_checks_tables.md §3]. Robustness only, descriptive.

## 2. Data

**Sources** (pinned) [reports/01_data_tables.md]: GSM8K test `openai/gsm8k@740312add88f` (1319 rows); `HuggingFaceH4/MATH-500@6e4ed1a2a79a` (500); `kenhktsui/gsm8k_sc@f9c0af7f82ec` (1313); `kenhktsui/prm800k_sc@62e7a1036e82` (448); `kenhktsui/scli5@9b159157bd1d` (286). 3866 source rows → 2047 canonical problems; every link and merge is in `data/processed/merges.jsonl` [reports/01_data_tables.md].

**Splits.** Seeded 20/80 dev/test at the canonical-ID level, stratified by problem stratum, one split for all models, seeded processing order per split [design §5.2; DECISIONS 2026-11-03; reports/01_data.md].

| Table | Dataset | Dev | Test |
|---|---|---|---|
| D-nat | gsm8k | 264 | 1055 |
| D-nat | math500 | 100 | 400 |
| D-inj | gsm8k_sc | 263 | 1050 |
| D-inj | prm800k_sc | 95 | 353 |
| D-inj | scli5 | 54 | 232 |
| Exp 0 | exp0 (paired worlds) | 80 | 320 |

[reports/01_data_tables.md]. Fixed sets: Exp 5 spillover set 50 test problems, KL probe set 50 dev problems [reports/01_data_tables.md]. Original-pool D-nat test n = 1455 (5052 pooled with the expansion, below); D-inj test n = 1635 [reports/02_power_tables.md "Test n"].

**Test outcomes seen (initial right / wrong only)** [reports/01_data_tables.md]:

| Model | gsm8k right / wrong | math500 right / wrong |
|---|---|---|
| 1.7B | 873 / 182 | 290 / 110 |
| 4B | 976 / 79 | 330 / 70 |
| 8B | 985 / 70 | 325 / 75 |

**D-nat expansion (in the confirmatory pool for 4B).** MATH test problems outside MATH-500: `EleutherAI/hendrycks_math@21a5633873b6a120296cce3e2df9d5550074f4a3`, test split, 5000 rows; 503 dropped as already in the pool; 4497 items kept (dev 900, test 3597; 4485 canonical IDs, 12 merges) [data/processed/ext/build_ext_manifest.json]. New stratum `math_ext`, same seeded 20/80 rule and its own processing order; existing pool, split and orders untouched; gold = last `\boxed{}` of the solution (agreement with MATH-500's answer field 1.0 on 500 overlapping problems) [DECISIONS 2026-11-04; build_ext_manifest.json].
- **Test outcomes seen on the expansion (initial right / wrong only):** 1.7B 2639 / 958; 4B 3000 / 597; 8B 3041 / 556 [results/p1/<model>/p1-initial-ext-<model>-45575c2/metrics.json].
- **Scope:** the expansion enters every D-nat hypothesis on 4B (H1a, H1b, H1d, H2a, H2b) [DECISIONS 2026-11-05]. It was built for H2a, and both H2a (central-scenario power 0.33 on the original pool → 0.99 pooled) and H1a (0.63 → 1.00) need it. An earlier version of the H2a table calibrated the original-pool scenarios on the pooled dev and gave 0.88; the per-scope calibration corrected it after the review [reports/03_power_tier1_tables.md; reports/03_power_h2a_tables.md]. The scope was fixed on dev power and budget only, before any post-feedback test use [design §5.1]. Replications: 8B and 1.7B run Exp 1A on the original pool at their design priority and on the expansion in separate lower-priority blocks [configs/blocks.yaml].
- **Estimand population (changed from the design):** the pooled D-nat test pool **as sampled**, 5052 items: GSM8K test (1055) + MATH-500 (400) + the expansion (3597), each item weighted equally. MATH-500 plus the expansion is the MATH test set less near-duplicates of the pool, so the population is the two standard test sets in full, and MATH is the majority (79 %). The design's D-nat ("GSM8K + MATH-500, pooled, stratified by source" [design §3.1]) remains as a **secondary, descriptive** estimand (the original pool alone), together with the per-stratum estimates (gsm8k, math500, math_ext) and the estimate reweighted to the original GSM8K / MATH mix [DECISIONS 2026-11-05]. These are shown in the main results table, not only in an appendix, with the math_ext − original-pool difference and its CI as the pre-specified stratum-heterogeneity check (descriptive).
- **The pooled estimand cannot drift.** The expansion block `4B-exp1a-ext` is atomic (§10): it is either complete or absent. If it is absent at P6, every D-nat test runs on the original pool (the design's estimand) and is labelled an **underpowered pre-registered fallback** (central-scenario power H1a 0.63, H2a 0.33); H2a then also uses the original pool, so all D-nat hypotheses always refer to one population [analysis/p6_confirmatory.py].

## 3. Hypotheses

Estimands are risk differences in percentage points on the natural item distribution, Δacc = P(initially wrong) × W→R − P(initially right) × R→W, with W→R and R→W reported separately [design §3.2]. Logistic/GEE coefficients are used only for tests. **δ = 3 pp** [design §3.2; DECISIONS 2026-11-03, confirmed by Liam 2026-11-03]. Unit: the canonical problem. All confirmatory tests: Qwen3-4B, non-thinking, test split.

### 3.1 Design of each test

| ID | Data (test n) | Contrast and estimand | Direction | Test [analysis/stats.py] |
|---|---|---|---|---|
| H1a | D-nat pooled, 5052 (§2) | Δacc(F2 − F1), whole methods (F2 has an extra turn) | equivalence within ±δ | TOST, p = max of the two one-sided paired tests at ±δ, cluster-robust standard error by canonical ID (`tost(..., clusters=)`) |
| H1b | D-nat pooled, 5052 | correctness vs p ∈ {0.5, 0.7, 0.9, 1.0} under F3(p), primary mask (verifier stream seed 0); effect size: acc(1.0) − acc(0.5) on the fitted linear trend = half the mean item-level OLS slope of correctness on p, with a cluster-bootstrap CI; adjacent-level risk differences reported | positive slope | one-sided GEE logistic on p, clustered by canonical problem, exchangeable (`gee_slope`). Secondary estimate, no test: Δacc(F3(0.5) − F0), predicted ≤ 0, two-sided CI |
| H1d | D-nat pooled, 5052 | Δacc(F5 − F3-perm), F3-perm outcome averaged over K = 3 permutations [design §5.8; src/data/verifiers.py `K_PERM`]; item-level d_i = F5_i − mean_k F3-perm[k]_i | two-sided | **estimation-only**: mean of d_i with a cluster-bootstrap CI; no confirmatory test [DECISIONS 2026-11-05] |
| H2a | D-nat pooled, 5052 (4B test right / wrong 4306 / 746 [03_power_h2a_tables]) | AUROC(frozen dev probe P(correct)) − AUROC(verbal score s = log P(yes) − log P(no)) for a0 correctness; pre-feedback position = last token of a0; ties count 1/2 | probe > verbal | `auroc_diff_boot`: paired bootstrap resampling canonical problems with all their items, 10,000 resamples; one-sided p = (1 + #{resampled difference ≤ 0}) / (10,000 + 1); 95 % percentile CI. Size at the null boundary and recovery of a known effect: tests/test_stats.py. Paired DeLong p reported as a check; Holm uses the bootstrap p; if the two disagree on significance at the Holm level, the outcome follows the bootstrap and is flagged "method-sensitive" [design §3.4, §7.3] |
| H3a-role | D-inj, 1635; feedback F0 | W→R(user) − W→R(assistant), averaged over the two label levels; d_i = (u/s + u/o − a/s − a/o)/2 | **non-directional** (the design states no sign) | two-sided paired z test on item-level d_i with a cluster-robust standard error by canonical ID (`two_sided_paired`; D-inj has scli5 problems with up to 10 items) |
| H3a-label | D-inj, 1635; F0 | W→R(other) − W→R(self), averaged over the two role levels | non-directional | as H3a-role |
| H2b | initially wrong D-nat pooled test items in the top quintile of probe P(correct), quintile recomputed in each bootstrap resample; n ≈ 149 (a fifth of 746) [03_power_tier1_tables] | (i) W→R(F2) − W→R(F0); (ii) W→R(F3(1.0)) − W→R(F0) | (i) below δ_w = 10 pp, (ii) > 0 | **estimation-only**: both differences with CIs; δ_w is a reference line, no test [DECISIONS 2026-11-05] |
| H3b | D-inj test own-error items (assistant / self / F0) with steerable content: n = 1547 (88 test items have no solution content and are excluded; counted by `src/steer/test.py --dry-run`) | W→R(main setting) − mean W→R over the 5 random directions, same layer, span and realized norm | steered > random | one-sided paired z test on item-level d_i, cluster-robust standard error by canonical ID (`one_sided_paired(..., clusters=)`). Generic control invalid at the main setting (§6, §7), not run on test. R→W under the same arms reported on 300 initially right original-pool D-nat test items: the first 300 when sorted by `h("steer-test-v1", item_id)` [src/data/build.py `h`, as for the dev confirmation set in src/steer/pilot.py] [DECISIONS 2026-11-05] |
| H4a | Exp 5 per-item pool, n = 400: the first 400 original-pool D-nat test items in the seeded order, spillover set excluded ("natural" pool) [DECISIONS 2026-11-05] | parameter arm (update, q-only context): acc(self-verdict update) − mean over the K = 3 perm-self updates | two-sided | **estimation-only**: mean item-level difference with a cluster-bootstrap CI; no confirmatory test [DECISIONS 2026-11-04, 2026-11-05] |

### 3.2 Margins, family, enabling, power, status

Central power scenario (definition fixed 2026-11-04 before any test post-feedback quantity, [DECISIONS 2026-11-04]): pilot discordance × 1; effect δ for superiority tests, δ/2 for equivalence and non-inferiority tests; for H2a, an AUROC difference of δ_AUC = 0.03. When δ exceeds what the pilot discordance allows, a hypothesis stays confirmatory only if power is ≥ 0.8 at the largest feasible effect among {δ/2, δ} at the pilot discordance; otherwise it is estimation-only [DECISIONS 2026-11-05]. Power is reported at the first Holm step, α/6 for the family of §4 (α/9 also shown in the tables). P3 values come from the dev pilots with F5 at t* and the probe re-selected on the enlarged dev [reports/03_power_tier1_tables.md; reports/03_power_h2a_tables.md].

**When these rules were set, relative to the dev results (disclosure).** All of them use dev data only, and none uses a test outcome, but each was written after the dev power tables it governs had been seen:
- The central-scenario definition (2026-11-04) came after the P2 power tables. It moved H3a-role and H3a-label from 0.79 and 0.51 (at δ/2) to 1.00 (at δ).
- δ_AUC = 0.03 (2026-11-04) came after the dev AUROCs had been seen.
- The infeasible-central rule (2026-11-05) came after the P3 tables. It keeps H1b, whose dev trend is positive, and demotes H1d, whose dev estimate is near 0. Under a strict reading of design §7.4 (no affordable n reaches 0.8 at δ), H1b would be estimation-only too. That reading is reported as a sensitivity analysis: H1b's estimate with Holm over the other five members [analysis/p6_confirmatory.py].
- H1b is confirmatory at an effect below δ (δ/2 is feasible only if nearly all discordant items move one way: r = 0.96 pooled). Its "not supported" margin is therefore δ/2, the powered effect, not δ; and a supported H1b whose estimate is below δ is written "supported (effect below δ)".

| ID | Tier | Central-scenario power (4B, α/6) | Status | Missing-data rule |
|---|---|---|---|---|
| H1a | 1 | **1.00** pooled (n 5052; discordance 4.5 %, effect 1.5 pp); 0.63 on the original pool alone, 0.76 reweighted to the original mix [reports/03_power_tier1_tables.md] | **confirmatory** (pooled estimand, §2) | §9 general |
| H1b | 1 | effect δ infeasible at the pilot discordance (the trend comes from the few items whose answer follows the verdict); at δ/2: **1.00** pooled, 0.97 original; δ at the least discordance that allows it: 1.00 [reports/03_power_tier1_tables.md] | **confirmatory** by the infeasible-central rule; "not supported" margin δ/2 | §9 general |
| H1d | 1 | no δ-scale effect is feasible at any scenario discordance (pilot discordant mass 0.6 % pooled); 0.77 pooled if half of the discordant mass went one way [reports/03_power_tier1_tables.md] | **estimation-only** by the infeasible-central rule; the bound is stated with the estimate | §9 general |
| H2a | 1 | central (Δ AUROC 0.03): **0.99** pooled (dev gap 0.009), 0.33 on the original pool alone (dev gap 0.029); at the dev gap: 0.18 pooled, 0.29 original; scenarios calibrated on each scope's own dev items [reports/03_power_h2a_tables.md]. The pooled dev gap is a third of δ_AUC, so a "not supported" or "inconclusive" outcome is likely | **confirmatory**; margin δ_AUC = 0.03 for "not supported" | §9; an item without a probe or verbal score is rerun, else excluded from both predictors and counted |
| H3a-role | 1 | 1.00 (discordance 10.9 %, δ) [reports/02_power_more_tables.md]; sensitivity: 0.79 at δ/2, 0.09 at half the pilot | **confirmatory** | §9 general |
| H3a-label | 1 | 1.00 (discordance 13.1 %, δ) [reports/02_power_more_tables.md]; sensitivity: 0.51 at δ/2, 0.22 at half the pilot | **confirmatory** | §9 general |
| H2b | 2 | ≤ 0.03 in the central scenario (F2 − F0 = δ_w/2, F3 − F0 = δ), ≤ 0.41 in a favourable one, pooled n 149 [reports/03_power_tier1_tables.md] | **estimation-only** (not enabled); δ_w = 10 pp as the reference line | §9 general |
| H3b | 2 | 1.00 at α/9 and α/6 (discordance 10.0 %, δ) [reports/02_power_tier2_tables.md]; sensitivity: 0.89–0.92 at δ/2, 0.29–0.34 at half the pilot (dev mean d = 1.5 pp) | **confirmatory (enabled), random control only** | §9; D-inj items without content tokens after the first 5 are excluded from every Exp 3 stage and counted [DECISIONS 2026-11-04] |
| H4a | 2 | 0.17 (α/9) / 0.21 (α/6) natural pool, 0.13 / 0.16 wrong-first pool, n 400 [reports/02_power_tier2_tables.md]; below 0.8 in every scenario | **estimation-only** (not enabled) [DECISIONS 2026-11-04] | §9; Exp 5 anomaly rules (§6) |

Enabling criteria [design §3.1]: a Tier 2 hypothesis is enabled iff its module passed the dev feasibility criteria **and** central-scenario power ≥ 0.8. Under §7.4 any hypothesis whose n for ≥ 0.8 power is unaffordable is estimation-only.

### 3.3 Exploratory and estimation-only analyses (no confirmatory claim)
H1c (channel, Exp 1C); H3c (role/label decodability); H4b (spillover); H4c (streams, trajectory unit, 3 order seeds, descriptive); H4d (self-prediction consistency proxy); Exp 4 (patching); the thinking subset; F2-scratch and F2 − F2-scratch; the H3a role × label interaction and the (assistant, self) vs (user, other) diagonal; H3a on D-nat and F1 / F3(1.0) as secondary feedbacks; the Tsui replication cell (scored by the design scorer, not the paper's LLM judge [DECISIONS 2026-11-03]); the two secondary Exp 3 settings, compared with unsteered only (descriptive; §14); the H2a cross-fitted secondary analysis (conditional CIs) and the comparators (answer length, mean token log-prob, dataset/difficulty model, per-dataset and per-MATH-level results); the Exp 5 context-arm counterpart of H4a (linked to H1d), ext-1.0, and the coupling × verdict-source interaction; sampled robustness (3 seeds, T 0.7, top-p 0.8, 300 items); verifier-mask robustness (2 extra mask seeds, 300 items) [design §3.1, §5.8, §6]. Replications on 1.7B and 8B and on secondary datasets are reported with CIs, not as separate confirmatory tests [design §3.1]. Exp 0 is a calibration and leakage control, reported as valid rate, accuracy and gain ratio (acc − 0.5)/(p − 0.5) for p > 0.5 [design §5.5].

### 3.4 Decisions on specific hypotheses (P3, before the freeze) [DECISIONS 2026-11-05]
- **H1d: estimation-only.** F5 and F3-perm can differ only on items whose verdict differs between arms and whose answer follows the verdict. Dev discordance stays below about 1 % even at t*, so |Δacc| is bounded at about that size on test; no δ-scale effect is feasible [DECISIONS 2026-11-04; reports/03_power_tier1_tables.md]. The report states the estimate with this bound. The 1.7B verdict hardly moves its answers at all (its F3(p) accuracy is flat in p on dev) [reports/03_power_tier1_tables.md].
- **H1d / H4a with K = 3 permutations.** The estimand averages each item's control outcome over the K = 3 permutations [design §5.8]: d_i = (self arm)_i − mean_k (perm arm)_i, a non-binary paired difference. Its mean is reported with the cluster-bootstrap CI of §8; neither enters a confirmatory test. The Exp 5 test block trains all K perm-self adapters (the pilot trained k = 0 only) [src/ttt/pilot.py].
- **H2a: confirmatory, pooled.** Verbal comparator scored as s = log P(yes) − log P(no), rank-equivalent to P(yes)_norm without the float ties near 1 that 4B produces [DECISIONS 2026-11-04]. Margin δ_AUC = 0.03 for the central scenario and for "not supported". Probe frozen in §6.
- **H2b: estimation-only**, δ_w = 10 pp as the reference line (power ≤ 0.03 in the central scenario, §3.2). δ_w was fixed in P3, not "with δ" as the design says (§14). The dev pilot runs against H2b(i): critique repairs more initially wrong items than self-review [reports/03_power_tier1_tables.md]. Wording fixed now: if the CI of W→R(F2) − W→R(F0) lies entirely above δ_w, H2b(i) is described as "contradicted by the estimate (no test)"; if it lies entirely below δ_w, "consistent with H2b(i) (no test)"; otherwise "inconclusive (no test)". H2b(ii) likewise against 0.
- **Exp 5 test pool: "natural"**, the first 400 original-pool D-nat test items in the seeded order with the spillover set excluded. It gave better H4a precision than "wrong first" in the P2 simulations and needs no sampling weights [reports/02_power_tier2_tables.md].

## 4. Fixed confirmatory family and multiplicity

- **Holm** at α = 0.05 over the family fixed at the freeze [design §3.1, §7.5]. TOST and intersection-union tests enter with p = max of their components.
- **Family (6): H1a, H1b, H2a, H3a-role, H3a-label, H3b.**
- Not members: H1d (Tier 1, estimation-only), H2b, H4a (Tier 2, estimation-only), all of §3.3. The design says the family "contains every Tier 1 hypothesis" [design §3.1] and also that an underpowered hypothesis "becomes estimation-only" [design §7.4]. Rule adopted before the freeze: an estimation-only hypothesis makes no confirmatory claim and therefore takes no family slot [DECISIONS 2026-11-04, 2026-11-05]. This affects H1d only.
- The simulated power at α/6 is the first Holm step for this family [reports/03_power_*_tables.md].
- A member that cannot be run later (budget, infrastructure), or whose block ends below its min_n, keeps its slot with p = 1. The family is never shrunk after post-feedback test outcomes have been seen [design §3.1].
- Holm is computed once, in P6, by `analysis/p6_confirmatory.py` (frozen, §8), after every family block is complete or stopped. Outcome categories are assigned only then.

## 5. Decision rules (after Holm) [design §3.2]

| Outcome | Rule |
|---|---|
| supported | significant in the predicted direction (H1a: TOST significant) |
| contradicted | significant in the opposite direction (two-sided tests only) |
| not supported | not significant, and the 95 % CI excludes effects ≥ δ in the predicted direction (only for powered tests) |
| inconclusive | neither of the above |
| estimation-only | pre-registered as such: report the CI, make no claim |
| not run | gate, feasibility or budget, with the reason |

Rules per member, as implemented in `analysis/p6_confirmatory.py`. CIs are two-sided 95 % cluster-bootstrap intervals, not adjusted for multiplicity; "significant" means the Holm-adjusted p < 0.05.
- **H1a (equivalence):** supported if the TOST is significant; contradicted if the CI of Δacc(F2 − F1) lies entirely beyond +δ or entirely beyond −δ; otherwise inconclusive.
- **H1b (one-sided):** supported if significant, written "supported (effect below δ)" when the estimate is below δ; not supported if not significant and the CI's upper end of acc(1.0) − acc(0.5) is below δ/2 (§3.2); otherwise inconclusive.
- **H2a (one-sided):** supported if significant; not supported if not significant and the CI's upper end of the AUROC difference is below δ_AUC = 0.03; otherwise inconclusive.
- **H3a-role, H3a-label (non-directional):** supported if significant in either direction, with the sign reported; not supported if not significant and the CI lies inside (−δ, δ); otherwise inconclusive. "Contradicted" does not apply.
- **H3b (one-sided):** supported if significant; not supported if not significant and the CI's upper end is below δ; otherwise inconclusive. A post-hoc narrow CI never upgrades a result to "equivalent"; "not significant" is never written as "equivalent". "Block completed" and "hypothesis supported" are separate states [CLAUDE.md rule 9]. Each claim is linked in `docs/evidence_ledger.md`.

## 6. Values frozen from dev

| Quantity | Value | Source |
|---|---|---|
| Self-verdict rule | "correct" iff s = log P(yes) − log P(no) ≥ t*; no verbal score → "incorrect" | [src/data/verifiers.py `self_verdict`; DECISIONS 2026-11-04] |
| t*, 4B (proposed: the dev threshold maximizing balanced accuracy) | t* = 15.999999163764045 (P(yes)_norm ≥ 0.9999998874647438); dev balanced accuracy 0.7717 vs 0.5595 at 0.5 | [analysis/out/selfverdict.json] |
| t*, 1.7B | t* = 15.873724267114774 (P(yes)_norm ≥ 0.999999872318173); 0.6729 vs 0.5 | [analysis/out/selfverdict.json] |
| t*, 8B | by the adaptation rule (§11) | [design §9] |
| t* scope | chosen on the original dev (P2) and applied to the whole pool, expansion included; not re-selected on the enlarged dev (only the probe was announced for re-selection) | [DECISIONS 2026-11-05] |
| F3(p) masks | stream `verifier-v1`, seed 0 primary, seeds 1–2 robustness; nested across p | [src/data/verifiers.py] |
| F3-perm / perm-self | stream `perm-v1`, K = 3, within dataset × initial-correctness strata | [src/data/verifiers.py] |
| Probe, 4B — **frozen** | re-selected on the enlarged dev (original + expansion, n 1264) by the procedure fixed in advance: layer 24, C = 0.001, grouped-CV AUROC 0.891; `probe.npz` sha256 `a62621d9ef55fed97f3de6a16b269715fe402b2e882c4bcae7b436172c8bc833` | [results/p3/4B/p3-probe-dev-ext-4B-b7cf0d6/; DECISIONS 2026-11-04, 2026-11-05] |
| Probe, 1.7B — **frozen** | layer 17, C = 0.001, 0.886 (n 1264); sha256 `db0c6752bfc88da58d797a954af863e7f6d862940a24c8dfc93f457c622e65d0` | [results/p3/1.7B/p3-probe-dev-ext-1.7B-b7cf0d6/] |
| Probe grid and procedure | all layers × C ∈ {0.001, 0.01, 0.1, 1.0}; L2 logistic; grouped 5-fold CV by canonical problem, seed 0; standardization inside folds; refit on all dev and frozen; the P2 probes on the original dev (`p2-probe-dev-*-4eb1a4b`) are superseded by these | [src/probe/train.py] |
| Exp 3 main setting (4B) | direction **role**, layer **18**, α = **0.2** (fraction of ‖h‖, toward "user"); dev: proxy gain 0.0152, mean KL 0.0117, W→R gain over the random mean 1.5 pp, R→W rule met; generic control **invalid** at this setting | [analysis/out/tier2.json; reports/02_pilot_tier2_tables.md] |
| Exp 3 secondary settings | role, layer 18, α 0.4; label, layer 6, α 0.4 | [reports/02_pilot_tier2_tables.md] |
| Exp 3 fixed elements | directions file `directions.npz` sha256 `8aaa4488419da72d1a577667bdfa434b45c4977c1ab9af7f93269d81a46621a8` (396 dev D-inj items, 16 excluded without content); random directions seeds 11–15; first 5 solution tokens skipped; prefill-only steering at reviewed-solution content tokens; reachability = mean KL > 1e-4 nats; markers correction ['Let'], commit ['The'] | [results/p2/4B/p2-steer-pilot-4B-7ea8f7d/manifest.json, config.json, metrics.json] |
| Exp 5 update rule | LoRA rank 8, lora_alpha 16, on q, k, v, o; A seeded, B = 0; LoRA fp32 on bf16 base; fresh AdamW per item; 4 steps; loss on the final-answer span of a0 (last `\boxed{}`): NLL if v = correct, unlikelihood −log(1 − p) computed exactly via logsumexp if v = incorrect; global-norm rescale of all B_m to τ | [src/ttt/update.py; src/ttt/pilot.py] |
| Exp 5, 4B | τ = 0.5, lr = 1e-3 | [analysis/out/tier2.json] |
| Exp 5, 1.7B | τ = 0.5, lr = 1e-4 | [analysis/out/tier2.json] |
| Exp 5 selection rules (as applied) | τ = largest of {0.5, 2, 8} whose mean per-token KL on the KL probe set is ≤ 0.05 nats **for every lr in the grid**; lr ∈ {1e-4, 1e-3} by the ext-1.0 parameter-arm gain over B1 on 60 dev tuning items, ties → smaller lr | [src/ttt/pilot.py; DECISIONS 2026-11-03] |
| Exp 5 controls | matched-norm random LoRA (Gaussian per module, global rescale to τ); shuffled verdict = self-verdicts permuted within dataset, ignoring correctness; zero-step = B1 | [src/ttt/update.py; DECISIONS 2026-11-03] |
| Exp 5 anomaly rules | no answer span, N = 0 or NaN/Inf → no update (intention-to-treat), counted; c > 100 → applied, flagged, excluded in a pre-registered sensitivity analysis | [src/ttt/update.py; design Exp 5] |
| Revision budget | 2048 (§1) | [DECISIONS 2026-11-04 (corrections)] |
| Thinking budget | 16384 (§1) | [reports/03_checks_tables.md §2] |

Caveats carried into the paper [DECISIONS 2026-11-04]: the 4B lr was chosen on the same 60 items that give the reported dev gain; the matched-norm random adapter also lifts dev accuracy over B1, so the ext-1.0 gain over B1 bounds the verdict's contribution from above, and verdict-specific contrasts are against random, shuffled and perm-self.

## 7. Tier 2 module status

| Module | Feasibility on dev | Power | Status |
|---|---|---|---|
| Exp 3 (H3b), 4B | reachability passes (64 reachable settings); α = 0 reproduces the unsteered tokens; generic control invalid at the main setting → random control only, conclusion limited to "beyond random perturbation" [analysis/out/tier2.json; DECISIONS 2026-11-04; design §3.5] | 1.00 central [reports/02_power_tier2_tables.md] | **enabled** |
| Exp 5 (H4a), 4B and 1.7B | all four criteria pass on both models: zero-step = B1 100 % identical; reset restores logits; realized-norm max relative error 5.30e-08 (4B), 5.71e-08 (1.7B); ext-1.0 parameter arm > B1 (gain 21.7 pp 4B, 23.3 pp 1.7B); plus the same-update-twice invariant 100 % identical [reports/02_pilot_tier2_tables.md] | < 0.8 in every scenario | **feasible, not enabled**: H4a estimation-only; H4b–d exploratory |
| H2b (reuses Exp 1A + Exp 2) | no separate module | ≤ 0.03 central (pooled) | **not enabled**: estimation-only |

The Exp 3 pilot ran on an A30 [results/p2/4B/p2-steer-pilot-4B-7ea8f7d/manifest.json]; the α = 0 and reachability invariants are re-checked on the main-run GPU before the test block [DECISIONS 2026-11-04 (A30 entry)]. On test, the generic control (invalid at the main setting on dev) is not run, and R→W is measured on 300 initially right original-pool D-nat test items in the seeded order `steer-test-v1` [DECISIONS 2026-11-05]. Block order: 4B Exp 3 (H3b, confirmatory) runs before 4B Exp 5 (H4a, estimation-only) [configs/blocks.yaml]. If the α = 0 or reachability re-check fails on the A100, the whole Exp 3 block runs on the A30, where the dev checks passed; if it fails there too, H3b is "not run" (p = 1).

**How the H3b main setting was chosen (disclosure)** [reports/02_pilot_tier2_tables.md]:
- The selection rule compares each candidate with its reference. The reference is the generic control where that control is valid, and the mean of the random directions otherwise.
- Only the first candidate (role, layer 18, α 0.4) had a valid generic control. Against it, that candidate's gain was −13.3 pp: steering lowered W→R to 79.2 % from 88.3 % unsteered.
- The winner (role, layer 18, α 0.2) was compared with the random mean, a different reference. Its gain of +1.5 pp comes from 120 dev items and is within the spread of the five random directions.
- So the main setting carries little dev evidence of an effect. A null on test is the likely outcome and is reported as such (design §3.5).

## 8. Analysis plan (common to all hypotheses)

- **CIs:** item-clustered paired bootstrap, 10,000 resamples, resampling canonical problems with all their variants and conditions [design §7.1]. Exception: streams (trajectory unit, descriptive).
- **Tests:** paired z tests with cluster-robust standard errors by canonical ID for the family's paired contrasts (H1a TOST, H3a, H3b); GEE logistic (canonical-problem clusters, exchangeable) for H1b; the cluster bootstrap for H2a; Holm over the family (§4) [design §7.2; analysis/stats.py]. Exact McNemar remains for descriptive two-condition contrasts of binary outcomes.
- **AUROC:** paired cluster bootstrap (primary, `auroc_diff_boot`), DeLong (check); the CI conditions on the frozen dev-trained probe, which is the H2a estimand [design §7.3, Exp 2].
- **Confirmatory analysis code, frozen with this file:** [analysis/p6_confirmatory.py] with the statistics in [analysis/stats.py]; its hashes are recorded at the freeze (§13). A dry run on the dev pilots exercised every code path [reports/03_p6_dryrun.md] (a test of the code, not a result). At P6 it runs on a spec naming the test runs, with no other choices left.
- **Pre-registered sensitivity analyses** (descriptive, reported next to the main estimates):
  1. every family estimate and raw p with second-scorer v2 labels;
  2. D-nat contrasts and H2a without items whose a0 is truncated or unparsed (39 of the 159 wrong math_ext dev items have a truncated a0 [results/p1/4B/p1-initial-ext-4B-45575c2/metrics.json; reports/03_data_ext_tables.md]);
  3. the strict reading of design §7.4 for H1b (§3.2);
  4. per-condition main-vs-v2 scorer disagreement rates.
- **Per-block analyses during P4** (CLAUDE.md P4) report estimates and CIs only: no Holm, no outcome category, until P6. Every post-freeze entry in DEVIATIONS.md lists which test results had been seen when it was written.
- **Reporting:** effect sizes and CIs for everything; W→R and R→W separately; generated tokens per condition (compute is post-treatment, not adjusted for); unparsed, truncated and conflicting outputs counted per condition; stratified subsets reweighted by their sampling weights; cross-model comparisons on a common item panel [design §3.2, §6, §7.8].
- **Statistics invariants:** each test's false-positive rate at its own null boundary and recovery of a known effect [design §8; tests/test_stats.py]. Latest passing run of `tests/`: 133 passed at commit `b7cf0d6` (2026-11-05, control host); at the freeze: 138 passed, 1 warning at commit `b2e0366` (control host; output in [reports/03_pytest_freeze.txt](reports/03_pytest_freeze.txt)).
- **D-nat estimand:** every D-nat estimate is computed on the pooled pool as sampled (§2); the original-pool, per-stratum and reweighted estimates are secondary and descriptive.
- **First post-feedback test analysis** is logged with a timestamp in `runs.jsonl` and STATUS.md [CLAUDE.md rule 3]. So are the first test probe-scoring run and the first test AUROC computation (H2a guard, header).
- **Pinned P4 invocations:** the 4B Tier 1 blocks run as [cloud/p4a_sequence.sh] (runner flags, t*, verbal and initial files, the frozen probe and its hash check, a gate that refuses to run before this file is frozen, and an abort on the first failed step); 4B Exp 3 runs as [cloud/p4b_sequence.sh] with [src/steer/test.py] (frozen directions with a hash check, the pre-check of §7, 100-item chunks with resume). Both were dry-run or smoke-tested before the freeze, and their hashes are recorded at the freeze.

## 9. Missing data [design §7.6]

- Nothing is dropped. Unparsed, truncated, refused or conflicting final answers count as incorrect; Exp 0 invalid outputs count as wrong; each is counted per condition.
- Infrastructure failures are rerun. If a rerun is impossible, the item is excluded from every condition of that contrast and counted.
- Module rules: Exp 3 items without solution content are excluded from every Exp 3 stage and counted [DECISIONS 2026-11-04]; Exp 5 anomalies follow §6 (intention-to-treat); an item without a verbal score gets the "incorrect" self-verdict [src/data/verifiers.py]; for H2a, an item without a probe or verbal score is rerun, else excluded from both predictors and counted.

## 10. Block order, budget stops and partial samples [design §10]

- `configs/blocks.yaml` is the single run order. Priority (cut from the bottom when the budget runs short): 4B Tier 1 (Exp 0 → Exp 1A on the original pool and on the expansion → Exp 1B with the alternative-wrapper subset → Exp 2) → 4B Tier 2 (Exp 3 → Exp 5 per-item) → 8B Exp 1A + 1B → 1.7B Tier 1 → 4B thinking → 8B and 1.7B Exp 1A on the expansion → 8B Exp 2 → Exp 1C / robustness / F2-scratch → Exp 4 → streams → 8B thinking [configs/blocks.yaml; DECISIONS 2026-11-05]. Cost estimates per block: [reports/00_compute_rates.md]; through priority 7 they sum to about $48 of the remaining allowance.
- Items are processed in the seeded order, in chunks that complete **all conditions** of the block for those items; all chunks of a block run on one GPU type.
- **Atomic blocks.** `src/exp/runner.py` generates every item of a pool in one engine call, and the Exp 2 block caches every test item. So the runner and Exp 2 blocks are atomic: chunk = min_n = planned n, and a stopped atomic block is re-run in full at its locked commit. They are cheap: priority 2 is about $7 [reports/00_compute_rates.md]. 4B-exp0, -exp1a and -exp1b are one runner call on the original pool.
- A block that reached its planned n is analysed as planned. A chunked block stopped **above** its pre-registered minimum n is a *pre-registered partial-sample analysis*, labelled as such. A block **below** it is descriptive only, and a family member in such a block keeps p = 1 (§4). A stopped block is never reported as a null.
- **Minimum n, chunk size and stop rule per block:** the `min_n`, `chunk` and `stop` fields of [configs/blocks.yaml]. min_n is computed by [analysis/p3_min_n.py] [reports/03_min_n_tables.md]:
  - atomic blocks: the planned n;
  - 4B-exp3 (H3b): the smallest n with central-scenario power ≥ 0.8 by the normal approximation from the dev sd;
  - blocks without a confirmatory test: half the planned n.
  The script checks that blocks.yaml carries these values. Since the expansion block is atomic, the pooled estimand never becomes "original pool + a prefix" (§2).
- Budget: CLOUD_CAP_USD 100 with a fixed 15 % reserve and a storage reserve [CLAUDE.md §6].

## 11. 8B adaptation rule [design §9]

No 8B tuning before the freeze. For 8B, every tuned quantity is chosen on **8B dev** by the same grid and selection rule as for 4B, by script, with no human or agent judgment, and the selection output is committed and logged in `runs.jsonl` before any 8B test run:
1. self-verdict threshold t*: the threshold on s maximizing dev balanced accuracy [analysis/p2_selfverdict.py];
2. probe layer and C: the §6 grid and grouped CV [src/probe/train.py];
3. Exp 3 and Exp 5, only if run on 8B ("rest of 8B"; no 8B Exp 3/5 block exists in [configs/blocks.yaml]): the grids and rules of [src/steer/pilot.py] and [src/ttt/pilot.py], with the same feasibility criteria;
4. budgets, wording, scorer and verifier streams: identical to 4B (not tuned per model).
8B results are replications with CIs, not confirmatory tests [design §3.1]. 8B dev mirrors 4B: step 1 (t*) on the original dev, step 2 (probe) on the original + expansion dev; the 8B expansion dev quantities already exist [results/p1/8B/p1-*-ext-8B-45575c2; DECISIONS 2026-11-05].

## 12. Deviations after the freeze

- `prereg.md` is never edited silently. Every change to hypotheses, metrics, n, δ, the family or the analysis goes into `DEVIATIONS.md` with date, what changed, why, expected effect on conclusions, and affected run IDs [CLAUDE.md rule 4; DEVIATIONS.md].
- A change that would alter a headline claim, or dropping a pre-registered hypothesis, is a human gate [CLAUDE.md §2] (decided under Liam's blanket go-ahead and logged with that label [DECISIONS 2026-11-04]).
- A bug in a reported result: fix, new run ID, old run superseded, affected reports regenerated, entry in DEVIATIONS.md [CLAUDE.md §8]. A change to scientific code means a new run ID; shards from different code versions are never mixed [CLAUDE.md rule 8].

## 13. Items resolved before the freeze, and the freeze checklist

Resolved in P3 [DECISIONS 2026-11-05 unless noted]: expansion scope and the pooled estimand (§2); H1a, H1b, H2a confirmatory, H1d and H2b estimation-only, with the rule for an infeasible central scenario (§3.2, §3.4); the K = 3 estimand for H1d / H4a (§3.4); H2a margin and verbal score form (§3.4); Exp 5 test pool (§3.4); family membership (§4); revision budget [DECISIONS 2026-11-04] and thinking settings and budget (§1); Exp 1B alternative wording (§1); probe re-selection and hashes (§6); Exp 3 test details and the block order (§7, §10); per-block minimum n, chunk size and stop rule (§10, configs/blocks.yaml); the latest passing test run (§8); 8B dev composition (§11); the scorer and HF-vs-vLLM invariants (§1).

At the freeze (code state = commit `b2e0366`):

1. Tests: 138 passed, 1 warning [reports/03_pytest_freeze.txt].
2. sha256 of every file on the confirmatory path:

| File | sha256 |
|---|---|
| `src/score/scorer.py` | `40790d874fa320464b61ae2937569df455179e555678d0cbdda3fe3bb81f31e0` |
| `src/score/second.py` | `5aa8b6e84d94b488fefda50e97c57d146deb77b652077d1dc8f5f5b4d6e3d18a` |
| `analysis/stats.py` | `4656dfa45f55593f734dba62eaf00a235c8551b2725fc61b5d5546e474dc4502` |
| `analysis/p6_confirmatory.py` | `ad9fd9274d13e62bd98c321c55b75d28da890d99bf42975479052f17b18b4b9d` |
| `analysis/p2_power_h2a.py` | `016aaa3a8dea92252369cac751ccfd381aa03d3938cabaeb48c3650c635bfe75` |
| `analysis/out/min_n.json` | `3270a50550e7bdb879f80e708c35861d57eefbf359f9c011857c6ea2e4d6495c` |
| `src/probe/train.py` | `5a66d567f0a60ac74f64f8761d1790ef18831069a775b551dee1b34e0a6f1182` |
| `src/probe/score.py` | `bf4b58d854112dda1b8c8288049c14446ccd7241722770e90ea1d59f71073a94` |
| `src/probe/cache.py` | `5d9170e67f99626d38923816471a8188d7b64a147feb0018e1bba3690c07a989` |
| `src/exp/runner.py` | `b20746841148ee4f7efc2e5e50ea6404eb2247d6b10cbef54de28a7de9c4a1fc` |
| `src/gen/render.py` | `8876e630cd7e592da50e9b3fbeaec1d95478df6e4bd5ec03debd917d29d489d5` |
| `src/data/build.py` | `69774521a20654449631f04c34e00557a6ced41db0cf269712fc1dff9b6a6d76` |
| `src/data/verifiers.py` | `3f1cefd8066ffca2d2394eab3440fb60a832d5102704f243a32f66d7e2616e64` |
| `src/steer/test.py` | `107f687f1461d3401ab3e34db5ce23f4bce8188061323cc43c3c4df7e5accf45` |
| `src/steer/pilot.py` | `328140ea7d47a242b0ddb30e479383e6ad044747a2438ec59e05f6a42bcc85cb` |
| `src/steer/steer.py` | `7dbb3aff8c1b0f779a2ba6792ce60fe33a46631107ee2a4ec42c653160fdce4e` |
| `cloud/p4a_sequence.sh` | `99bad31984e4a9b3c048389b1ba3ac292358ac6873eca7d14aac4dca8f9a1c85` |
| `cloud/p4b_sequence.sh` | `9a301beae6854759d77fe32f5aa33a52ab2283bd44fc5ab96f2e551efb2dbd0e` |
| `configs/blocks.yaml` | `105280659d756b55331ad50318f36f4f54e2e4a1009c8e15cabd24e548b200ac` |
| `configs/models.yaml` | `cbf29774e8a71a652d176fa6b7c22e396c351e7c5025b3ac2b69d6ddafa42477` |
| `configs/p6_spec_dev_dryrun.json` | `7b6ab1198f7ad378233427e0ecbfb1bfdbd764abd69a58f10182dae666a6e8bc` |

   Wording: `wording_sha256` main `229f9167e91e8a01ef18fa4fe6d37b79c51f69243b9a9a66988dadbc045583cf`, alternative set `86c71e3c9371fca5ac4858f2ffdf43b285ee9574140aa2fc6c38b10d4094e6dc`; Exp 5 context template `0c0f570d8cbc60a10895a871e1992a5c0cdfd252b9b46cdc4bbd29073caa83e6` (§1).
3. This file's commit hash and sha256: STATUS.md.

Present and current at the freeze: the related-work matrix [paper/related_work_matrix.md]; the P1 data card [reports/01_data.md] and the expansion's data card [reports/03_data_ext_tables.md]; the P3 report [reports/03_prereg.md].

## 14. Departures from design v2.1

The design wins on what is tested until the freeze (header). Every place where this file departs from it, with the reason and the timing. All use dev data only; none uses a test outcome.

| # | Design v2.1 | This file | Why | Decided (after which dev result) |
|---|---|---|---|---|
| 1 | D-nat = GSM8K + MATH-500, pooled [§3.1] | pooled with the MATH expansion, as sampled; the design's pool secondary (§2) | central-scenario power: H1a 0.63 → 1.00, H2a 0.33 → 0.99 | 2026-11-05, after the P3 power tables |
| 2 | central scenario undefined [§7.4] | pilot discordance × 1, δ (superiority), δ/2 (equivalence, non-inferiority) (§3.2) | §7.4 needs one to apply | 2026-11-04, after the P2 power tables |
| 3 | margins in pp only [§3.2] | δ_AUC = 0.03 for H2a (§3.2) | AUROC is not on the pp scale | 2026-11-04, after the dev AUROCs |
| 4 | — | infeasible-central rule (§3.2); H1b "not supported" margin δ/2 | δ unreachable at the pilot discordance for H1b and H1d | 2026-11-05, after the P3 tables; revised after the review |
| 5 | the family contains every Tier 1 hypothesis [§3.1] | estimation-only hypotheses take no slot; H1d out (§4) | §7.4 makes underpowered hypotheses estimation-only | rule 2026-11-04; applied 2026-11-05 after the P3 tables |
| 6 | scorer agreement ≥ 99 % with a second scorer [§8] | amended bound on the main scorer's error in the disagreements (§1) | the second scorers are weaker than the main one | 2026-11-05, after the check failed twice |
| 7 | HF vs vLLM: identical greedy outputs [§6 Exp 5] | teacher-forced argmax agreement; the design's fallback applies (§1) | measures HF as it is used (full-sequence forward) | 2026-11-05, before the check ran |
| 8 | revision budget "frozen from the pilot (default 1024)" [§4] | 2048 (§1) | the runner default before any pilot; not tuned | recorded 2026-11-04 |
| 9 | thinking budget default 4096 [§4] | 16384 (§1) | the dev rule, fixed before its pilot | 2026-11-05, rule before the pilot |
| 10 | unlikelihood loss with a clamp [Exp 5] | exact form via logsumexp (§6) | the clamp gave zero gradient on confident answers | 2026-11-04, during the Exp 5 pilot |
| 11 | priority: Exp 5 → Exp 3 [§10] | Exp 3 → Exp 5 (§10) | H3b is confirmatory, H4a estimation-only | 2026-11-05, after the P2 power tables |
| 12 | test: main setting + both controls [Exp 3] | the generic control (invalid on dev) not run (§7) | design §3.5 limits H3b to the random control | 2026-11-05, after the P2 Exp 3 pilot |
| 13 | δ_w "fixed with δ" [§3.4] | δ_w = 10 pp, a reference line only (§3.4) | H2b is estimation-only | 2026-11-05, after the P3 tables |
| 14 | exact McNemar for paired binary contrasts [§7.2] | cluster-robust paired z tests (§8) | the unit is the canonical problem; H3a and H3b differences are not binary | 2026-11-05, after the review |
| 15 | decision rules use "the predicted direction" [§3.2] | H3a non-directional rules (§5) | the design gives H3a no sign | 2026-11-05, after the review |
| 16 | complete paired chunks [§10] | runner and Exp 2 blocks atomic (§10) | the runner has no chunking; the blocks are cheap | 2026-11-05, after the review |
| 17 | secondary Exp 3 settings tested with Holm within the secondary set [Exp 3] | compared with unsteered only, descriptive (§3.3) | matched random controls for each would double the Exp 3 cost | 2026-11-05, after the review |
