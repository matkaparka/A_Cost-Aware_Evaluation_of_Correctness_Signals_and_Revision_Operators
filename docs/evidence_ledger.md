# Evidence ledger (CLAUDE.md rule 9; design §3.2)

This file links each claim to its hypothesis, analysis script, result table, scope, alternative explanations and permitted wording. It holds no numbers: every number is in the linked generated table.

- "Block" means the block completed at its planned n (prereg §10).
- "Outcome" is the category from design §3.2, assigned once by the frozen `analysis/p6_confirmatory.py`.
- These are separate states.
- The main table is [reports/06_results.md](../reports/06_results.md).

Common scope of every confirmatory row:
- model and run: Qwen3-4B, non-thinking, test split, greedy decoding, revision budget 2048 tokens;
- prompt templates: as frozen;
- scorer: 1.1.1.
- D-nat population: GSM8K test + MATH test pooled as sampled (79 % MATH; prereg §2).
- Replications on 1.7B and 8B are reported with CIs, not tests.

Last updated 2026-11-07 (P6).

## Confirmatory family

### H1a: critique (F2) vs a zero-information trigger (F1)
- **Block / outcome.** Complete / supported (TOST within ±δ).
- **Script → table.** `analysis/p6_confirmatory.py` → `reports/06_confirmatory_4B.md`, "Family"; context line (discordance, W→R, R→W).
- **Scope.** Whole methods: F2 has an extra generation turn.
- **Alternative explanations.**
  - Low revision rates cap |Δacc|: F1 and F2 disagree on few items (prereg §0).
  - The CI also excludes 0, with F2 slightly above F1. The equivalence holds at δ = 3 pp, not at zero.
  - Exploratory decomposition: F2 vs F2-scratch is small and positive; F2-scratch vs F1 includes 0 (`reports/06_exploratory_4B.md`).
  - In thinking mode, F2 − F1 is large on 4B, but F1 there is distorted by unclosed think blocks (D11).
  - 1.7B and 8B replications include 0.
- **Permitted wording.** "Within ±3 pp at these revision rates", with discordance, W→R and R→W of both arms. Not "critique adds nothing" or "no better than 'Wait'".

### H1b: dose-response in verifier accuracy p (p is not shown to the model)
- **Block / outcome.** Complete / supported (effect below δ).
- **Script → table.** `analysis/p6_confirmatory.py` → "Family"; strict-reading sensitivity line.
- **Alternative explanations.**
  - The trend comes from the few items whose answer follows the verdict.
  - It is below δ and was confirmatory only by the infeasible-central rule. That rule was written after the dev tables (prereg §3.2 disclosure). Under the strict reading, H1b is estimation-only, and the others' Holm results are unchanged.
  - The H1b secondary estimate F3(0.5) − F0 includes 0.
  - Exp 0 shows the model follows verifier hints only partially: gain ratio well below 1.
- **Permitted wording.** "Correctness rises with the accuracy of the simulated verifier, which the model is not told; the effect is below δ." Not "stated reliability": the verdict wrapper never states p, so H1b measures how much answers follow true rather than wrong verdicts (review, reports/07_paper_review.md #2).

### H2a: frozen probe vs verbal score for a0 correctness
- **Block / outcome.** Complete / supported; DeLong agrees.
- **Script → table.** `analysis/p6_confirmatory.py` → "Family" and the H2a line; comparators per stratum and per MATH level in `reports/06_exploratory_4B.md`.
- **Scope.** Pooled D-nat estimand. Linear probe at the last token of a0, layer and C chosen on dev.
- **Alternative explanations.**
  - The pooled advantage does not appear within any single stratum: each stratum's CI includes 0 (exploratory). The pooled AUROC also ranks items across strata, and the probe may order the strata's difficulty better than the verbal score does. The stratum's dev accuracy alone has low AUROC.
  - Answer length is a strong comparator, and the probe exceeds it.
  - The original-pool secondary estimate is positive.
  - 8B's CI includes 0; 1.7B's advantage is larger.
- **Permitted wording.**
  - "On the pooled test set, the frozen dev-trained probe ranks initial-answer correctness better than the verbal score."
  - Always add "linear accessibility, not an information ceiling", and give the per-stratum caveat.

### H3a-role: user vs assistant role of the reviewed solution
- **Block / outcome.** Complete / not supported (CI inside ±δ).
- **Script → table.** `analysis/p6_confirmatory.py` → "Family"; F1, F3(1.0) and D-nat in `reports/06_exploratory_4B.md`.
- **Alternative explanations.** 1.7B shows a negative role contrast, with the assistant role corrected more (replication, not a test).
- **Permitted wording.** "No role effect of δ size at F0." Not "no effect".

### H3a-label: "other" vs "self" label of the reviewed solution
- **Block / outcome.** Complete / supported (positive): the "other" label raises W→R.
- **Script → table.** `analysis/p6_confirmatory.py` → "Family"; exploratory contrasts in `reports/06_exploratory_4B.md`.
- **Scope.** F0 on D-inj. At F1 and F3(1.0), the label contrast includes 0 (exploratory). 1.7B is positive; 8B includes 0.
- **Alternative explanations.**
  - The label is a prompt word: "other" may simply invite a check.
  - On the 300-item alternative-wording subset, the main and the alternative wording give similar label contrasts, both with CIs that include 0.
  - The 2 × 2 does not recover generation history.
  - The diagonal (user, other) − (assistant, self) is the blind-spot comparison; it is exploratory and its CI includes 0.
- **Permitted wording.** "A prompt-label effect at F0." Not a self-model, and not by itself a replication of the Self-Correction Bench blind spot.

### H3b: steering toward the external configuration vs random directions
- **Block / outcome.** Complete / not supported.
- **Script → table.** `analysis/p6_confirmatory.py` → "Family" and the Exp 3 arms table; `reports/04_p4_4B_exp3.md`.
- **Scope.** One dev-selected setting (role, layer 18, α 0.2). Random control only: the generic control was invalid on dev.
- **Alternative explanations.**
  - The setting carried little dev evidence (prereg §7 disclosure).
  - The estimate is negative with a CI below 0: steering lowered W→R relative to random directions and to unsteered. The test is one-sided, so the category is "not supported", not "contradicted".
  - Random directions sit near unsteered, so the drop is specific to this direction at this norm.
- **Permitted wording.** "Steering at the tested setting did not raise correction beyond random perturbation; it lowered it." A null is not support for essay §一.

## Estimation-only (no claim)

| Quantity | Table | Reading (prereg §3.4 wording) |
|---|---|---|
| H1d (self-verdict placement) | 06_results.md | estimate with its discordance bound; no test; allocation of self-verdict errors, not "external evidence" |
| H2b(i) F2 − F0, confidently wrong | 06_results.md | "contradicted by the estimate (no test)": critique repairs more than self-review there |
| H2b(ii) F3(1.0) − F0 | 06_results.md | "consistent with H2b(ii) (no test)" |
| H4a (parameter arm, self vs perm-self) | 06_results.md; 04_p4_4B_exp5.md | estimate only; one update rule; not reflexive |

## Exploratory (descriptive; no claim)

| Item | Script → table | Note for wording |
|---|---|---|
| H1c channel | analysis/p4_channel.py → 04_p4_4B_channel.md | wrapper and position manipulation (Qwen3 renders tool turns as user turns) |
| Sampled and verifier-mask robustness | same | estimates move little relative to greedy and to mask seed 0 |
| Exp 4 patching | analysis/p4_exp4.py → 04_4B_exp4.md | every frozen component replicated under the dev-frozen rule; effects are not additive: each lp_gold component carries most of the total, and single marker components move the marker more than the full donor swap does |
| H4b spillover | analysis/p4_exp5.py → 04_p4_4B_exp5.md | self updates lower gold log-prob on the spillover set; self − perm includes 0 |
| H4c streams | analysis/p4_streams.py → 04_p4_4B_streams.md | self streams collapse into repetition; the post hoc table (D12) describes what preceded it and tests no mechanism; one ext seed collapses differently |
| H4d | 04_main_runs.md (p4c) | not measured in base/self/random (budget); only the context cell is interpretable |
| Thinking subset (4B, 8B) | analysis/p4_thinking.py → 04_p4_{4B,8B}_thinking.md | mode and budget contrast, not RL training; F1 distorted on 4B (D11) |
| F2-scratch, H1b secondary, 1B cells/interaction/diagonal, alt labels, Tsui cell, Exp 0, H2a comparators | analysis/p6_exploratory.py → 06_exploratory_{4B,1.7B,8B}.md | written after the confirmatory results (D13) |
| Not run | 06_results.md | H3c; H2a cross-fitted analysis and log-prob comparator (D13). H3c and the log-prob comparator were run later (D14, below) |

## Post-freeze follow-ups (exploratory; D14–D16; outside the Holm family)

Run after the first draft, from branch `lore-qa`. Estimates and CIs only; no claim enters the confirmatory results.

| Item | Script → table | Note for wording |
|---|---|---|
| Lore QA (non-math task; frozen addendum, estimation only) | analysis/lore_results.py → lore_02_results.md | D-inj-lore name rows are the primary pool (gates G2, G3b, G4); phrase rows and D-nat-lore are descriptive. W→R is high with or without critique, so the role and label contrasts are small; say "outside math, the pattern was …", not "the effect generalizes" |
| Stated reliability F3s(p) (D14) | analysis/fu_group1.py → fu_02_group1.md | stated − unstated within ±0.2 pp at every p, slope difference includes 0: stating p did not change how often answers follow verdicts. Complements the H1b wording (p is not shown) |
| H2a comparators (D14) | same | the probe beats the verbal score and both token log-prob summaries, but 5-sample consistency and a dev-fitted combination beat the probe. Say "the probe reads a0 correctness better than the verbal score"; never "the best available signal" |
| H3c (D14) | same | role and label decodable at content tokens (AUROC 1.0, layer-0 invariant holds). Decodability only; content tokens attend to the role tokens, so this does not show use |
| Family2: Tulu 3 8B Tier 1 (D15) | analysis/p4_block_estimates.py, p6_exploratory.py, fu_verbal_check.py → 04_p4_T8B_tier1.md, 06_exploratory_T8B.md, fu_04_verbal_check.md | a replication with CIs, not tests. Verbal P(yes) is near chance pooled on test and on dev. fu_04_verbal_check.md shows that yes/no carry most of the probability mass, so the candidate tokens are not the obvious problem; this does not establish that the readout means what it asks. fu_07_verbal_variants.md shows that no other wording passes the pre-set rule, and that within datasets the fixed wording carries some signal. Say "weak, and not comparable across datasets". The probe − verbal gap is large; answer length alone is close to the probe. The role contrast is always "role-plus-template". Role and label contrasts at F0 are positive, unlike Qwen3; the T8B role cell includes a one-token template boundary (D15). A difference across families is not evidence against the Qwen results |
| Stream controls nll, matched (D16) | analysis/fu_group2.py → fu_03_group2.md | matched (self's whole realized verdict sequence, permuted, including verdicts given after self degraded) collapses into repetition like self; nll (no unlikelihood) does not loop but its accuracy also falls, with wide seed spread. Say (second review, F): "a permuted verdict sequence with the same overall rate of 'incorrect' verdicts suffices to produce the repetition under this update rule, which makes unlikelihood pressure a plausible explanation; number, timing, placement and closed-loop effects are not separated" (3 seeds per condition; no test). Never call nll "stable" |
| Structured steering controls (D16) | same | main lowers W→R more than each structured control (pc1, dataset contrast, sign-flipped role) of equal norm; the frozen main/unsteered rows reproduce exactly. But main also has the largest pre-check KL, so equal norm is not equal functional perturbation. Say (second review, E): "among the tested directions of equal norm, the role direction causes the largest drop; the property that causes it is not identified". Do not say "role-specific" or "not a generic effect" |

## Second review (2026-11-08; exploratory; D17–D18)

| Item | Script → table | Note for wording |
|---|---|---|
| Full generation cost per condition | analysis/fu_recheck_cost.py → fu_05_recheck_cost.md | counts both turns of F2 and F2-scratch; F2 costs more than twice F0 and about as much as a0. Token counts only; no wall-clock time. The "mean out tokens" column of fu_01 counts only the final turn: never cite it as F2's cost |
| Selective re-check (4B math, seen test split) | same | Revised at the third review: "the probe's better ranking did not give a consistent advantage in re-check gain over the verbal order (small, and of changing sign across shares)", not "no larger gain"; "agreement with a0 ranks best but costs more than re-checking every item". Rules fitted on dev; exploratory; replayed offline from stored outputs, not "re-ran" |
| Model-to-model differences | analysis/fu_crossmodel.py → fu_06_crossmodel.md | estimate differences directly; never read one significant and one non-significant estimate as a difference. Tulu's role contrast is role-plus-template |
| Tulu readout variants (math dev) | analysis/fu_verbal_variants.py → fu_07_verbal_variants.md | the pre-set rule selected no variant; within-dataset AUROCs above the pooled one. Say "weak, and not comparable across datasets", not "Tulu cannot assess itself" or "the readout is broken" |
| H3c regrouped by canonical problem | analysis/fu_h3c_regroup.py → fu_08_h3c_regroup.md | unchanged (items barely share problems); decodability only |
| MMLU-Pro validation (frozen addendum) | analysis/mc_results.py → v2_02_mc_results.md | primary comparisons P1–P3 and their reading rules are in docs/planning/validation_v2/addendum.md; one public non-math task with lettered answers, not a general claim about non-math tasks |

## Third review (2026-11-08; post hoc, exploratory; D19)

Every row below is post hoc and outside the Holm family. The paper's main line is now the revision decision
(`paper/manuscript.tex`). The confirmatory rows above are unchanged, and they stay in the main text.

General wording rules from the third review:
- "Not shown" or "not resolved" means the interval includes zero. It never means "no difference" or "equivalent": no
  equivalence margin was set for any policy difference.
- Math re-check results are exploratory analyses of a seen split. The MMLU-Pro comparisons P1–P3 were fixed before
  their data existed. Never merge the two into "replicated".
- "Independent" means the order of our work, not the absence of the benchmark from training data.
- Task probe (fitted on MMLU-Pro dev) is a same-task result; the math probe applied unchanged is a transfer result.
  Neither is "zero-shot generalization".
- Novelty is scoped to the updated search of 2026-11-08 (`paper/related_work_matrix.md`). Do not claim as new that
  revision trades repair against damage, that gates can reduce damage, or that probe discrimination can exceed its
  decision value. These were shown by Zhang 2609.35832, Dip et al. 2606.19808, Chen et al. 2608.04355 and Yuan et al.
  2605.09502.
- Kumaran et al. 2604.22271 is not contradicted: different population (initially wrong and changed), different token
  position, different target (correctability).

| Item | Script → table | Note for wording |
|---|---|---|
| Selective vs re-check every item (ALL) | analysis/fu_recheck_vs_all.py → fu_09_recheck_vs_all.md | "Above ALL" only where the paired CI is above zero. **Above ALL is not above KEEP**: on Tulu-3-8B MMLU-Pro, every ranking (random included) is above ALL at the 10 % and 20 % shares (at larger shares all but the transferred math probe) and none above KEEP; say "re-checking fewer items does less of a harmful action", never "selective re-checking helps". On math no cheap ranking is above ALL; only agreement-based rankings are, from 20 % on |
| Within-pool repair and damage rates | generated keys `ms.*.fixrate`, `ms.*.damrate` (analysis/p7_results_tex.py) from fu_05, v2_02, fu_09 | Descriptive: pools enriched for errors also hold right answers that F2 damages far more often than average. "We did not test why"; not causal |
| MMLU-Pro scoring audit | analysis/fu_mc_parse_audit.py → fu_10_mc_parse_audit.md | The frozen scorer stays primary. About a third of Tulu's F2 damages are revisions without a single letter (boxed text, several letters, letter with a qualifier). Under both lenient scorers the F2 loss remains a loss and no P1–P3 reading changes. Rule v1 was corrected after reading its samples (v2); report both |
| F2-scratch vs F2 (MMLU-Pro) | v2_02 (secondary) | "The loss under F2 is not the cost of a second turn as such" (Tulu: F2 − F2-scratch CI below zero). Not "critique is harmful" in general |
| F3(p) | fu_05, v2_02 | An idealized reference built from gold; no verifier cost; never a deployable winner |
| a0-answered sensitivity and format repair (post hoc) | analysis/fu_ms_robustness.py → fu_11_ms_robustness.md | On math a large share of F2's repairs (and almost all of F0's and F2-scratch's) complete initial answers truncated at the token limit. Report the math F2 gain together with its answered-item value; never call the full math gain "correction" without that caveat. On MMLU-Pro, F2/F1 readings hold on answered items; 4B F0's gain does not |
| Probe vs verbal as re-check rules (revised after the manuscript's skeptical review) | v2_02, fu_05, fu_11 | Say "the probe's ranking advantage carried over into at most a small decision advantage, which the frozen comparison could not resolve (intervals allow up to about a point; on Qwen3-4B the estimates favor the probe; resolved at 10 % on answered items)". Never "the probe gives no larger gain" or "a better ranking does not give a better decision" |
| Gold-reference headroom and operator choice | fu_05, fu_11 | With F2, even re-checking known errors first gains under a point on Tulu, no more than F2-scratch on every item: "for this model, the choice of operator matters more than the choice of signal". The gold reference is not an upper bound |
| Within-pool rates with CIs | fu_11 (`tab_ms_pools`) | Fragile right answers explain selective vs ALL/KEEP; probe vs verbal on Tulu differs in repair, not damage. Descriptive, post hoc |
| Blind audit of math outputs | analysis/fu_blind_audit.py tally → reports/fu_12_blind_audit.md | Filled by a separate AI model (Codex) at Liam's request, blind to condition and scorer verdict; all verdicts agree with the primary scorer. Say "an AI-model audit", never "a human audit"; targeted strata, not a population error rate |

## Neurocomputing revision (2026-11-11; D22; existing-output reanalysis)

The journal manuscript is `paper/Neurocomputing/manuscript/`; none of these changes creates a new confirmatory result.

| Claim or correction | Script and evidence | Permitted wording and scope |
|---|---|---|
| Initial-correctness reference versus realised-benefit bound | `analysis/neurocomputing_revision.py` → supplement `analysis/out/neurocomputing/revision.json`, `out/generated/tab_bounds.tex` | A wrong-first hash ordering is a diagnostic reference. Only the sorted stored gains form the maximum for fixed cardinality and fixed operator outputs; neither a token-budget nor a population bound. |
| Complete-policy generated-token comparison | Same script → `out/figures/fig_frontier.pdf`, `manifest.json` | Charge the initial answer when required by the policy. Standalone voting and thinking bypass it. Tokens quantify generation volume; they omit signal forward-pass, prefill, latency, energy and price. |
| Threshold sample-reuse cost | Corrected supplement `analysis/arr/a3_dev_test.py`; before/after summaries in revision history | Previously generated samples are not charged again. Only cost fields change; do not describe the correction as an accuracy improvement or a newly selected policy. |
| Thinking-gate value | Existing `c1b_total.json`, main `tab_budget_total.tex` and `tab_think.tex` | Exploratory stored-output replay. The development-to-test grid does not include thinking or voting; its results do not independently validate thinking thresholds or budget comparisons. |
| Stronger initial answer | Existing `b2_keep_strength.json`, `tab_b2.tex` | The stronger initial answer is a comparator. Operators were not rerun on it, so this does not establish robustness to stronger initial decoding. |
| Research and data availability | Main Methods, AI-writing declaration, data availability, supplement README | Analysis replay is supported by included derived records. Full generation reproduction requires omitted raw outputs, weights and original environment. Review does not replace author verification of declarations. |

## Neurocomputing v2 (2026-11-11; D23; retrospective existing-record analysis)

| Claim or correction | Evidence | Permitted scope |
|---|---|---|
| Input-aware policy cost | analysis/neurocomputing_cost_v2.py; v2 supplementary cost_v2.json and extraction_manifest | D is complete generated volume, P is logical input length excluding padding; D + alpha P is a scenario index, not measured latency, FLOPs or price. |
| Verbal cache overhead | Exact token-prefix comparisons, pinned tokenizer/source hashes, derived count arrays | Chat-template rerendering breaks prefix sharing before the initial answer; do not use the earlier nominal cache assumption. Retained probe states are an optimistic conditional case, not measured marginal cost. |
| Historical timing evidence | Original run metrics/config/manifests copied into cost evidence; tab_cost_timing and tab_cost_fixture | Measured offline passes and batch throughput. P0 predates mandatory batch-invariant generation; uniform suffix timings reuse traces. Do not add heterogeneous records into full online latency. |
| Retrospective threshold transfer | analysis/neurocomputing_crossfit_v2.py; crossfit.json, folds, manifest and report | Calibration uses signals and costs only; holdout is never forced to fit its budget. Report gain together with drift. Already examined data and finer candidates prevent calling this fresh independent validation or an isolated splitting effect. |
| Budget comparison wording | Updated manuscript and generated captions | Largest feasible share has cost no greater than the comparator. Say common budget cap, not exactly equal realised costs. |
| Visual summaries | analysis/neurocomputing_figures_v2.py; figure_data.json and manifest | Existing decomposition and existing intervals are replotted; no additional test or causal control. |
