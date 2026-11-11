# Research Design v2.1 — Information, Trigger, or Attribution? Dissecting Self-Correction in Frozen Language Models, with an Operator-Level Contrast

Status: **design v2.1 (2026-11-03)**. Replaces v1 (git `ac3ec91`) and v2 (git `4f65a56`).
- v2 answered the first GPT review (`docs/reviews/2026-11-03_gpt_review.md`; response in `docs/reviews/review_response.md`).
- v2.1 answers the second (`docs/reviews/2026-11-03_gpt_v2_review.md`; response in `docs/reviews/review_response_v2.md`). Its labels M1–M11 and C1–C5 are cited in brackets below.

This document is authoritative for *what* is tested. It is not frozen: `prereg.md` (P3) fixes the concrete values.

**Companion theory documents (`theory/`):**
- *Self-Reference for Dynamic Regression* (manifesto): §6, §7, §11, §15.1, §18, §20 (Claim 7), §23
- *Dynamic Regression as an Algorithm Architecture, Not an Agentic System* (positioning note): §4 guards
- Additional background notes are not distributed with this snapshot.
- *The Coupling-Type Criterion and the Generalized Frozen-Operator Theorem* (skeleton): **missing**. See §1.4 for what it blocks.

---

## 0. Summary

When a frozen LM revises an answer after feedback, how much of the change comes from **information** about correctness, from a **trigger** to revise, and from **attribution** of the reviewed text? Does the pre-feedback internal correctness signal predict which errors get fixed? And what changes when the same 1-bit feedback is written into the weights by a test-time LoRA update instead of being placed in context?

---

## 1. What this study can and cannot test

### 1.1 The frozen side

In a frozen model, self-critique and external feedback both enter as context. Both are object-level couplings by the framework's definition, so a frozen-only study cannot test coupling type. It tests what content-level feedback does inside a frozen operator.

### 1.2 Claim 7, read for a bounded model

Claim 7 (manifesto §11.1, §20): self-reference supplies operator-arguments, not evidence. Its exact content is the data-processing inequality (DPI): a function of the model's state and input carries no more Shannon information about any variable than the state and input already do. The DPI holds everywhere. What differs between tasks is whether the bound has non-trivial *predictive* content.

**(a) On deterministic tasks the bound does not constrain accuracy.** On GSM8K and MATH the gold answer is a function of the question q, so given q, neither a verdict nor a critique adds Shannon information about the answer. A computationally bounded model can still gain *usable* information: a verdict can supply it, and so can its own computation, since re-deriving a result can expose an error. Xu et al. (2020), arXiv 2002.10689, show that computation can increase usable information for a restricted predictive family. This is used here as a **conceptual** reading only: no accuracy or AUROC in this study is a measurement of V-information [C1]. Claim 7 therefore does **not** imply that self-critique cannot raise accuracy, that a p = 0.5 verifier cannot help, or that self-updates cannot generalize. Every behavioral hypothesis in §3 is an **empirical prediction motivated by Claim 7**, written so that it can fail. None is presented as a consequence of it.

**(b) A hidden state gives the bound non-trivial predictive content** [C1]. In Exp 0 the gold answer depends on a hidden random bit that only the verifier observes. There, the DPI fixes what any no-feedback policy can achieve (§5.5). Exp 0 is a **calibration and leakage control**, not evidence for the theory.

All writing keeps two things apart: *computational help* (reorganizing what the input already determines) and *evidence* (bits about something the input does not determine).

### 1.3 The operator-level arm

Exp 5 installs the same feedback into parameters with a test-time LoRA update. It is a comparison of **two ways of using the same feedback under a fixed parameter interface** [C1]. The 1-bit verdict is a *minimal self-assessment signal*. That name does not make it a self-model in the framework's sense, and the arm is not reflexive (positioning note §4, guard 1).

**Boundary relativity (must be stated in the paper).** If the LoRA weights are folded into an extended state, "base model + fixed update rule" is again a fixed transition function on a larger state space. The object/operator distinction in this study is therefore defined relative to a fixed interface: the parameter tensors of the pretrained network as released, versus its token input. Exp 5 asks an implementation-relative question: does routing identical bits through that parameter port, rather than the token port, change behavior (generalization to other items, drift, self-prediction)? Whether the distinction survives arbitrary reparametrization is a question for the skeleton.

### 1.4 Theory status (the paper uses the weakest status listed)

| Statement | Source | Status for this study |
|---|---|---|
| Frozen-operator limitation, single model | manifesto §6.3 | candidate theorem |
| Generalized frozen-operator theorem | manifesto §7.2 ("Claim"); positioning note calls it a theorem, citing skeleton Thm 3.2 | **unverified**: skeleton missing |
| Coupling-type criterion | skeleton | **unverified**: missing |
| Self-modification blind spot | skeleton Thm 5.3 | **unverified**: missing |
| Lawvere fixed-point theorem | Lawvere 1969 | theorem under its hypotheses; applying it to LoRA-updated LMs is an interpretation, not a derivation |
| Claim 7, Shannon form | manifesto §11.1 | true as stated; its behavioral corollaries are empirical (§1.2) |
| Closure stabilizes the coupled loop | manifesto §11.2, §11.7 | conjecture |

Where the manifesto and the positioning note disagree on strength, the manifesto's "candidate" wording is used. The missing skeleton blocks **theorem wording and citations of skeleton results**. It does **not** block a conservative framework section built on operational definitions (data port vs parameter port of the released network, Claim 7, this table) [C1].

### 1.5 What the study does not claim

- Not "frozen models cannot self-correct." RL-trained reasoning models self-correct with frozen weights; that behavior was written into θ at training time (essay §四).
- Thinking vs non-thinking mode on the same weights is a contrast of **reasoning mode and compute budget**. It does not identify the causal effect of RL training.
- No claims about consciousness or experience. Linear probes measure linear accessibility, not an information ceiling.

---

## 2. Prior work and the gap

Every entry is re-verified in P7 (CLAUDE.md rule 2) with the fields `supports_this_claim / scope_and_assumptions / limitations / proof_status`. A **minimal related-work matrix** (`paper/related_work_matrix.md`) is built **before the prereg freeze**, so that a missing key control is found early [C5], and is updated in P7.

- Huang et al. (arXiv 2310.01798; ICLR 2024): intrinsic self-correction on reasoning tasks largely fails; oracle-stop setups leak labels. Empirical and task-specific.
- Kamoi et al. (TACL 2024) survey: fair-evaluation criteria, and conditions under which correction works.
- Tsui (2025), *Self-Correction Bench* (arXiv 2507.02778): average blind spot 64.5% over 14 models; "Wait" reduces it by 89.3%; RL reasoning models lack it. Data are released: `kenhktsui/scli5`, `kenhktsui/gsm8k_sc`, `kenhktsui/prm800k_sc`.
- Zhang et al. (ACL 2025): answer wavering and prompt bias in intrinsic self-correction.
- Lee et al. (arXiv 2505.11924): steering for **moral** self-correction. A methodological precedent only.
- arXiv 2607.09803, *Spectral Origins of the Self-Correction Blind Spot*: reported to have a false Lemma 2, ρ(∏J) ≤ ∏ρ(J); the counterexample is checked in `docs/reviews/review_support/`. Confirm the lemma's wording before citing, and cite only results that do not depend on it.
- Kadavath et al. 2022 (arXiv 2207.05221); Orgad et al. 2024 (arXiv 2410.02707): correctness is decodable beyond verbal reports. Decodable is not the same as usable by the model.
- Lindsey 2026 (arXiv 2601.01828), introspection: some models can tell their own outputs from prefilled text. Relevant to attribution (Exp 1B, Exp 3).
- Xu et al. 2020 (arXiv 2002.10689), usable information under computational constraints: conceptual framing only (§1.2).

**Gap (scoped).** Within the literature checked as of the matrix date, we have not found a study that (a) varies information, trigger and attribution in one design with matched prompts, (b) relates correction to the pre-feedback internal correctness signal, (c) separates chat-role from source-label effects and intervenes on the corresponding directions, and (d) contrasts context-level with parameter-level installation of identical feedback, including self-verdicts against their own placement-permuted control. A dated search cannot guarantee that this holds, so the paper states the gap with its search scope and date, and never writes "nobody has".

---

## 3. Research questions and hypotheses

### 3.1 Tiers and the confirmatory family

- **Tier 1, confirmatory.** Primary model Qwen3-4B, non-thinking, test split. H1a, H1b, H1d, H2a, H3a-role, H3a-label. The primary dataset is named per hypothesis in §3.3–3.5: D-nat (GSM8K + MATH-500, pooled, stratified by source) unless stated otherwise.
- **Tier 2, confirmatory if enabled.** H2b, H3b, H4a. A Tier 2 hypothesis is *enabled* only if, at freeze time, (i) its module passed the dev feasibility criteria and (ii) its power simulation (§7.4) reaches ≥ 0.8. Otherwise it is pre-registered as estimation-only or exploratory.
- **Family fixed at freeze** [M5]. The Holm family contains every Tier 1 hypothesis and every Tier 2 hypothesis enabled at freeze. A family member that cannot be run later (budget, infrastructure) keeps its slot with p = 1. The family is never shrunk after test-split post-feedback outcomes have been seen.
- **Exploratory.** H1c, H3c, H4b–d, Exp 4, thinking mode, F2-scratch. Replications on 1.7B, 8B and the secondary datasets are reported with CIs. They are not separate confirmatory tests.

A module that fails its dev feasibility criteria is **disabled**, not "failed": its hypotheses become exploratory or are reported as "not run (feasibility)", and **it does not block Tier 1** [M5]. An *enabled* module whose implementation invariant fails (§8) is a bug, to be fixed.

### 3.2 Estimands, margins and decision rules

All estimands are on the natural item distribution:

`Δacc = P(initially wrong) × W→R − P(initially right) × R→W`

- W→R (correction) and R→W (corruption) are reported separately.
- Stratified subsets (thinking mode, Exp 5) carry sampling weights for reweighting.
- Effects are **risk differences** in percentage points. Logistic or GEE coefficients are used only for tests, never reported as accuracy differences [M4].
- Cross-model comparisons use a common item panel.

**Margins** [M5]. δ is fixed **now, on substantive grounds**: δ = 3 pp of accuracy. The reason is that differences below 3 pp are smaller than typical prompt-wording effects in self-correction work (a judgment, to be checked against the related-work matrix before the freeze; Liam may change it before the pilot). The pilot estimates only nuisance quantities (discordance, variance), which set n. If the n needed for power ≥ 0.8 is unaffordable, the hypothesis becomes estimation-only. **δ is never changed to fit the budget.**

**Decision rules** (applied after Holm):

| Outcome | Rule |
|---|---|
| **supported** | significant in the predicted direction (for H1a: TOST significant) |
| **contradicted** | significant in the opposite direction (two-sided tests only) |
| **not supported** | not significant, and the 95% CI excludes effects ≥ δ in the predicted direction (possible only for powered tests) |
| **inconclusive** | neither of the above |
| **estimation-only** | pre-registered as such: report the CI, make no claim |
| **not run** | gate, feasibility or budget, with the reason |

A post-hoc narrow CI never upgrades a result to "equivalent". "Not significant" is never written as "equivalent".

### 3.3 RQ1 — Decomposition (Exp 1A; primary data D-nat)

- **H1a (critique vs trigger, as whole methods)** [M4]. Δacc(F2 − F1) lies within ±δ (TOST). Motivation: critique adds computation but no evidence, so its benefit beyond a zero-information trigger is predicted to be about zero. Scope: F2 uses an extra generation turn, so this compares two **methods**. It does not isolate a pure "critique content" effect. The exploratory compute-matched control F2-scratch (Exp 1A) decomposes it.
- **H1b (reliability dose-response).** Correctness increases with p ∈ {0.5, 0.7, 0.9, 1.0}. Primary test: one-sided test of a positive slope on p in an item-clustered GEE. Adjacent-level risk differences are reported. Secondary estimate: Δacc(F3(0.5) − F0), predicted ≤ 0, with a two-sided CI.
- **H1d (self-verdict placement)** [M1]. Estimand: at fixed *realized* class-conditional reliability on test, does the association between the model's own verdicts and item state matter for correction? Contrast: F5 (the model's own verdict) vs F3-perm (the same verdicts permuted within dataset × initial-correctness strata, §5.8). Two-sided test on Δacc. Interpretation: F3-perm keeps how many right and wrong items receive each verdict and randomizes *which* ones. A difference therefore measures how self-verdict errors are allocated relative to fixability. The allocation can go either way: errors may concentrate on unfixable items (self < perm) or on easily fixed ones (self > perm). H1d is **not** a test of Claim 7 or of "external evidence".
- **H1c (exploratory; channel).** Channel effect = the range of accuracy across {tool-response wrapper, plain user message, verdict inside the prior assistant turn} at p = 0.9, compared with Δacc(p 0.9 − 0.7). Qwen3 renders a tool message as a user turn wrapped in `<tool_response>`, so this is a wrapper and position manipulation.

### 3.4 RQ2 — Internal signal vs verbal report (Exp 2; primary data D-nat)

- **H2a.** On the test split, the AUROC of a **dev-trained, frozen** linear probe (pre-feedback residual stream; layer and C chosen on dev) exceeds that of the normalized verbal P(yes) (§5.7), for initial-answer correctness. Both are fixed predictors on held-out items, so a paired item bootstrap is valid. DeLong is used as a check [M6].
- **H2b (Tier 2; accessibility, not the DPI)** [M4]. In the top quintile of probe P(correct) among initially wrong test items ("confidently wrong"; the quintile is recomputed inside each bootstrap resample), an intersection-union test of two parts:
  - (i) non-inferiority bound: the upper 95% one-sided bound on W→R(F2) − W→R(F0) is < δ_w;
  - (ii) superiority: W→R(F3(1.0)) − W→R(F0) > 0.

  p_H2b = max(p_i, p_ii). δ_w is fixed on substantive grounds with δ. Interpretation limit: linear accessibility only.

### 3.5 RQ3 — Attribution and the blind spot (Exp 1B, Exp 3)

- **H3a-role and H3a-label (Tier 1; two hypotheses, two family members)** [M4]. Primary data: **D-inj test split** (every item has an error, so W→R has power; content is controlled). D-nat is a replication. Primary feedback: **F0**. F1 and F3(1.0) are secondary. In the role × label 2 × 2:
  - **H3a-role** estimand: W→R(user) − W→R(assistant), averaged over the two label levels.
  - **H3a-label** estimand: W→R(other) − W→R(self), averaged over the two role levels.

  Both are two-sided paired tests on item-level risk differences. The interaction is exploratory. The (assistant, self) vs (user, other) diagonal replicates the Self-Correction Bench blind spot. Results are **prompt-label effects**; the 2 × 2 does not recover true generation history, and the crossed cells may read as semantically odd (all are reported) [C2].
- **H3b (Tier 2; causal)** [M4, M7]. At the one dev-selected setting, steering own-error items toward the external configuration raises W→R more than **both** controls: (i) matched random directions, and (ii) the generic revision-propensity direction. Both are at the same layer, span and realized perturbation norm. Intersection-union test: p = max(p_vs_random, p_vs_generic). R→W under the same steering is reported. If the generic control fails its dev validity check (Exp 3), H3b is pre-registered with the random control only, and its conclusion is limited to "beyond random perturbation".
- **H3c (exploratory).** Role and label are linearly decodable at content tokens. Content tokens attend to role tokens, so this does not rule out role information.
- **Interpretation rules.**
  - A positive H3b shows that a representation correlated with role or label causally modulates revision. It does not show a self-model.
  - A null shows no effect at the tested setting. It is not support for essay §一.
  - The paper confronts §一 with the fact that the chat template carries explicit role tokens. The question is whether they act as a causally effective self-label.

### 3.6 RQ4 — Operator-level contrast (Exp 5)

- **H4a (Tier 2)** [M1, M4]. **Parameter arm only** (update, q-only context): accuracy after an update from the model's own verdicts vs after an update from the stratum-permuted self-verdicts (perm-self, §5.8). Two-sided test. The interpretation is the same as for H1d: allocation of self-verdict errors at fixed realized reliability. ext-1.0 is reported descriptively. The context-arm counterpart is descriptive and linked to H1d, not a second confirmatory test. Coupling × verdict-source interaction: exploratory, with its own contrast.
- **H4b (exploratory; spillover).** Parameter updates change the gold-answer log-prob on the spillover set; mean spillover after self-verdict updates ≤ 0. Zero context-arm spillover is an implementation invariant (§8).
- **H4c (exploratory; streams).** Self-confirming drift under self-verdict streams vs grounded streams vs a context-memory stream (specification in Exp 5). The unit is the trajectory, 3 order seeds: descriptive only.
- **H4d (exploratory; self-prediction).** Agreement between "the answer I will give" (separate forward pass) and the actual re-answer, after self-verdict updates, matched-norm random updates, and context feedback. These are two different prompt tasks, so this is a consistency proxy only. It does not test skeleton Thm 5.3 or any Lawvere-type result [C3].

---

## 4. Models

| Role | Model (HF id @ candidate revision) | Mode |
|---|---|---|
| Dev / pipeline | `Qwen/Qwen3-1.7B` @ `70d244cc` | non-thinking |
| Primary | `Qwen/Qwen3-4B` @ `1cfa9a72` | non-thinking (primary); thinking subset |
| Scale replication | `Qwen/Qwen3-8B` @ `b968826d` | non-thinking; thinking if budget allows |

The revisions were read from the HF API on 2026-11-03. P0 pins the full revision hashes and the tokenizer and chat-template hashes in `configs/models.yaml`. Substituting another family is a human gate. bf16 weights; no quantized model in any experiment.

**Decoding (frozen per mode in the prereg).**
- Non-thinking: greedy; `max_new_tokens` frozen from the pilot (default 1024); repetition and truncation rates recorded.
- Thinking: the sampling settings from the Qwen3 model card (expected T = 0.6, top-p 0.95, top-k 20; confirm in P0). Seeds are paired across conditions, the budget is fixed (default 4096 new tokens), and truncation rates are reported per condition. A truncated output counts as unparsed, is scored wrong, and is counted separately. It is never read as a failure to correct.
- **Determinism** (P0; `reports/00_compute_rates.md`). In vLLM's default mode, greedy outputs are not reproducible across a repeat call, a different batch composition, or a zero LoRA adapter. With `VLLM_BATCH_INVARIANT=1` they are, at about twice the generation time.
  - All vLLM generation runs with `VLLM_BATCH_INVARIANT=1`, and every manifest records it.
  - HF generation (Exp 3) is checked for the same property in P2. All its arms, including α = 0, use one fixed batch layout.

**Chat-template facts** (checked 2026-11-03 against the Qwen3-4B `tokenizer_config.json`):
- A tool message renders as `<|im_start|>user\n<tool_response>…</tool_response>`.
- `enable_thinking=False` inserts an empty think block.
- Thinking content of assistant turns before the last user turn is stripped. So in thinking mode, the "own solution" is the post-think content only.

Every claim of "same text" is checked on token sequences (§6.0).

---

## 5. Data

### 5.1 Sources (revisions pinned in P1)

- **D-nat pool:** GSM8K test (1,319) + MATH-500 (500) = 1,819 problems. At 80% test that is about **1,455 test problems** [C4]. An optional expansion with other MATH test problems is decided in P3 by the power simulation, never by outcomes.
- **D-inj:** Self-Correction Bench, as released: `kenhktsui/gsm8k_sc` @ `f9c0af7f`, `kenhktsui/prm800k_sc` @ `62e7a103`, `kenhktsui/scli5` @ `9b159157`. Rebuild only if the released data are unusable, and log why.
- **Exp 0:** generated (§5.5).
- **P0 fixtures** [C5]: GSM8K *train* and MATH *train* problems. They are outside the study pool and are never used for any reported number.

### 5.2 Canonical problem IDs and splits (first step of P1)

1. Normalize the question text (Unicode NFKC, whitespace, case), hash it, and assign a canonical problem ID.
2. Link IDs across sources: GSM8K ↔ gsm8k_sc; MATH-500 ↔ PRM800K ↔ prm800k_sc. Merge near-duplicates (normalized-exact match plus MinHash ≥ 0.9; every merge is logged).
3. Split at the canonical-ID level, seeded: dev 20%, test 80%. Every variant and every model's outputs inherit the problem's split. All models use the same split.
4. The Exp 5 spillover set and the KL probe set are drawn here, with fixed IDs (§6, Exp 5).
5. Each split gets a **seeded random processing order**, used by every block, so that a budget stop leaves a random subsample (§10).

### 5.3 What may be seen before the prereg is frozen

- **Dev split:** anything.
- **Test split:** the machine may generate and score initial answers and compute pre-feedback quantities (probe scores, verbal verdicts), which are needed for strata and gates. The agent may look only at counts of initially right and wrong items per model × dataset. No post-feedback metric or condition comparison on test before the freeze. The first test-split post-feedback analysis is logged with a timestamp in `runs.jsonl` and STATUS.md.

### 5.4 D-nat and D-inj

- **D-nat:** the model's own greedy initial answer a0 on every item, unbalanced.
- **D-inj:** "own" means *attributed to the model* (the text sits in the assistant turn), not generated by it; D-nat-own is genuinely self-generated.
- The two are reported separately and never pooled.
- P1 reproduces Tsui's published protocol exactly for one replication cell, and documents how it differs from our review-turn protocol.

### 5.5 Exp 0 hidden-state items [M2]

**Construction.**
- About 200 visible templates (GSM8K-style).
- Each states one quantity as "either X or Y (written on a card you cannot see)". X ≠ Y, and the two worlds must map to **two different final answers** g0 ≠ g1. The generator asserts both.
- The prompt instructs the model to answer with one of two **labelled options**, both shown: (A) g0, (B) g1.
- Valid output = exactly one of A/B; anything else is *invalid*, counted, and scored wrong.

**Paired worlds.** Each template yields two items with the same visible prompt, world b = 0 (gold g0) and world b = 1 (gold g1), in the same split.
- A no-feedback output is generated once and scored in both worlds. A valid output is then right in exactly one world, **deterministically**.
- Feedback conditions run separately per world, since the verdict depends on the world.

**Independent randomness.** The hidden-bit stream, the verifier stream and the sampling stream use separate seeds, each derived as hash(stream name, canonical ID). Hidden bits and seeds never appear in any rendered input; this is checked.

**Properties** (conditional on the stated constraints):
- With paired worlds, the mean no-feedback accuracy over the two worlds is exactly 0.5 × (valid rate).
- With a verifier of accuracy p judging the initial **valid** choice, and a policy that switches exactly to the other option when told "incorrect", accuracy is p.
- Invalid outputs, a third answer, or a switch to something else break these equalities.

**Reporting.** LM behavior is reported as calibration, not as an invariant:
- valid rate per condition;
- accuracy;
- gain ratio (acc − 0.5)/(p − 0.5), for p > 0.5 only, with a CI.

Accuracy/p is not used: an agent that ignores feedback scores 1 on it at p = 0.5.

### 5.6 Scorer (deterministic, versioned, hashed)

- **Extraction:** the last `\boxed{…}`; otherwise the last "The answer is …" or `####` pattern; otherwise unparsed. Exp 0 uses option-letter extraction.
- **Equivalence:** `math-verify` (or an equivalent sympy-based checker), with documented normalization of numbers, fractions, LaTeX and units, set matching for multi-answer MATH items, and intervals.
- Unparsed, truncated, refused, or multiple conflicting final answers are scored incorrect and counted per condition. They are never dropped.
- **Tests:** ≥ 50 hand-checked edge cases in `tests/test_scorer.py`; agreement with a second scorer reported on dev.

### 5.7 Verbal verdict [C5]

- The query is "Is your final answer correct? Answer yes or no.", with thinking disabled for the verdict turn.
- **Candidate sets** Y = {yes, Yes, YES} and N = {no, No, NO}, each with and without a leading space.
- **Probability of a candidate:** its full-sequence probability, teacher-forced, under the **locked tokenizer**. The first-token shortcut is allowed only if P0 verifies that every candidate is a single token, that token IDs are deduplicated, and that Y and N do not overlap.
- P(yes)_norm = P(Y) / (P(Y) + P(N)). The total mass P(Y) + P(N) is recorded.
- The self-verdict threshold (0.5, or a dev-chosen threshold that maximizes balanced accuracy) is fixed in the prereg.

### 5.8 Verifiers

- **Simulated verifier F3(p).** Per item, one uniform U_i from the verifier stream. The verdict is the true label if U_i < p, otherwise flipped. Masks are nested across p and shared across conditions and models for the same item, so all F3 comparisons are paired. It is applied to right and wrong items alike. Robustness: 2 extra mask seeds on a 300-item subset.
- **Stratum-permuted self-verdict F3-perm (and perm-self in Exp 5)** [M1].
  - Take the model's thresholded self-verdicts on the test items.
  - Within each dataset × initial-correctness stratum, permute which items receive which verdict, using K = 3 fixed permutation seeds. The outcome for each item is averaged over the K permutations.
  - This keeps the **realized** class-conditional verdict counts on test exactly, and randomizes the placement.
  - It uses ground-truth strata, so it is an **oracle randomized control**: not a verifier available at deployment, and it is named as such. Labels enter only in the isolated evaluator that builds the permutation, by a frozen algorithm. Nothing is tuned on outcomes.
- **Gold isolation, as a counterfactual-label test** [M3].
  - Allowed model inputs: q and the model-generated a0. The dataset gold is available only to the evaluator and the verifier.
  - Test: keep q, a0 and all seeds fixed, change the private gold. Then all no-feedback prompts and all TTT token targets must be byte-identical; only the pre-registered verdict or F4 pointer may change.
  - A correct a0, a number that also appears in q, or an Exp 0 option list that contains the gold are all legitimate; generating the gold is success, not leakage.

---

## 6. Experiments

### 6.0 Common prompt infrastructure

- Every condition is rendered through the model's chat template.
- For each condition, store the rendered string, the token IDs, and the token spans of the question, the source label, the reviewed solution, the feedback, and the revision instruction.
- The reviewed-solution token sequence must be identical across conditions for the same item; template-boundary differences are flagged and handled by a documented rule.
- **The source label always comes before the reviewed solution** [M7]. Under causal attention, solution-token states can only depend on a label that precedes them.
- Compared wrappers are length-matched within ±3 tokens.
- All prompt wording is chosen on dev and hashed into the run config.

### Exp 0 — Hidden-state calibration (all models; cheap)

Conditions: no-feedback answer, F0, F2, and F3(p) for p ∈ {0.5, 0.7, 0.9, 1.0}, all in paired worlds (§5.5). The hard invariant is the programmatic-agent truth table (§8); LM behavior is reported as calibration.

### Exp 1A — Feedback decomposition (frozen; role and label fixed)

The reviewed solution is a0, in the assistant turn, labeled as the model's own. Feedback comes through one fixed channel (a user turn with a fixed wrapper). All revision conditions share the revision instruction and `max_new_tokens`.

| Code | Condition |
|---|---|
| B0 keep | a0 is the final answer (no generation) |
| B1 fresh-context | answer q again with a0 not in context, same budget [C2]. With greedy decoding, B1 often equals a0; this is expected, not a failure. Resampling effects come from the sampled-robustness run |
| F0 review | instruction to review and give a final answer |
| F1 trigger | F0 + "Wait," prefilled at the start of the revision |
| F2 self-critique | the model writes a critique (separate turn, frozen budget), then revises under the F0 instruction; no "Wait" |
| F2-scratch (exploratory) [M4] | same two-turn structure and budgets as F2, but the first turn is "Solve the problem again independently" with no critique of a0. F2 − F2-scratch gives the compute-matched critique-content contrast |
| F3(p) | simulated verdict, p ∈ {0.5, 0.7, 0.9, 1.0} |
| F5 self-verdict | the model's own thresholded verbal verdict, with the F3 wrapper |
| F3-perm | the stratum-permuted self-verdicts (§5.8), with the F3 wrapper |
| F4 location | D-inj only, where the injected step and a verified-correct prefix are known: a pointer to the injected step, or "no error found" |

Natural-error items get no F4. Generated tokens per condition are reported. Compute is post-treatment, so the primary analyses do not adjust for it.

### Exp 1B — Attribution 2 × 2 (role × label)

- Role ∈ {assistant turn, user turn} × label ∈ {self ("your earlier answer"), other ("an answer written by another model")}.
- The label always precedes the solution:
  - for role = assistant, it is stated in the user message before the assistant turn;
  - for role = user, it opens the user message.
- Length-matched wrappers. Feedback ∈ {F0 (primary), F1, F3(1.0)}.
- Data: D-inj (primary) + D-nat (replication). Tsui's original protocol is the replication cell. One alternative wrapper set is run on a 300-item subset.

### Exp 1C — Channel (exploratory)

At F3(0.9): tool-response wrapper vs plain user message vs verdict inside the prior assistant turn.

### Thinking subset (exploratory)

- 4B and 8B in thinking mode, on 300 items stratified by initial correctness (with weights).
- Conditions: {B1, F0, F1, F2, F3(0.7), F3(1.0), F5}, plus the Exp 1B 2 × 2 at F0.
- Decoding per §4, with 2 paired seeds.

### Sampled robustness (non-thinking)

3 seeds, T = 0.7, top-p 0.8, on a 300-item subset, with the Exp 1A conditions.

### Exp 2 — Probing (RQ2) [M6]

- **Positions:** the last token of a0 (pre-feedback; primary) and the last token of the feedback (secondary). Residual stream, all layers.
- **Primary procedure:**
  1. Train an L2 logistic probe on the **dev split only**.
  2. Choose the layer and C by grouped CV within dev. Standardization (and PCA, if used) is fitted on dev.
  3. Freeze the probe, then score the test items once.

  The CIs condition on this trained probe, which is exactly the H2a estimand: "this probe's held-out performance".
- **Secondary:** 5-fold grouped cross-fitting inside test, with the layer and C frozen. Reported with conditional CIs only, and labelled as such. Whole-learner uncertainty is not claimed.
- **Comparators in the same table:** verbal P(yes)_norm; answer length; mean token log-prob of a0; a dataset/difficulty-only model. Results per dataset and per MATH difficulty level.
- Dev size limits the probe. The pilot reports the dev learning curve; if it has not plateaued, P3 may enlarge dev by a pre-registered rule, before any test use.

### Exp 3 — Role and label directions (RQ3) [M7]

**Directions** (from the Exp 1B renderings on dev; residual stream `resid_post` of block L, at reviewed-solution content tokens excluding the first 5):
- d_role = mean(h | user) − mean(h | assistant), averaged over both label levels with equal weight.
- d_label = mean(h | other) − mean(h | self), averaged over both role levels.

Positive α means "toward external". Polarity is logged.

**Generic revision-propensity control.**
- Under F0 on dev: mean(h | model changed its answer) − mean(h | model kept it), at the **last token of the revision instruction**, the same token position in both groups.
- **Validity check on dev:** steering along it must increase the answer-change rate. If not, the control is invalid, and H3b is limited to the random control (§3.5).

**Random control.** 5 fixed-seed random directions at the same layer and span, rescaled to the same realized perturbation norm per position.

**Intervention.**
- Add α · d̂ · ‖h‖ to `resid_post` of block L at the reviewed-solution positions, during **prefill only**. Later layers at those positions, and the K/V that later positions attend to, change; the last prompt position is not edited.
- L is restricted to blocks 2 … (n_layers − 3), so that downstream layers exist.
- **Reachability diagnostic on dev:** the intervention must change the next-token distribution at the final prompt position (KL > 0 beyond bf16 noise). A structurally unreachable setting is excluded; it is not reported as a null.
- Decode-time steering is a secondary variant.
- Implemented with HF hooks or nnsight. α = 0 reproduces the unsteered tokens exactly. All controls use the same backend.

**Selection on dev:**
1. Screen (L, α, direction ∈ {role, label}) with a single-forward-pass proxy: P(first revision token ∈ correction markers) − P(∈ commit markers).
2. Confirm the top 3 settings by full generation on dev.
3. Pick the main setting by a rule stated in advance: the largest W→R gain over the generic control, subject to the R→W increase being ≤ that of the generic control.

**Test:** the main setting plus both controls (H3b), with the other 2 settings as secondary analyses (Holm within the secondary set).

### Exp 4 — Activation patching (exploratory localization)

- **Pairs:** D-inj items in the (assistant, self) vs (user, other) cells, with identical content.
- **Metrics:** the teacher-forced log-prob of the correct final answer after revision, and the marker-proxy logit difference.
- **Procedure:**
  1. Attribution patching over (layer × position group), on dev.
  2. Real activation patching of the top 20 components, on dev.
  3. Confirm the top k (k frozen in advance) on test.
- **Invariant:** a self-patch reproduces the baseline within tolerance.

### Exp 5 — Operator-level arm (RQ4) [M8, C3, C4]

**Inputs and grounding.** Shared input per item: q, a0, and a verdict v. Grounding levels: self (the thresholded verbal verdict), perm-self (§5.8), and ext-1.0.

**2 × 2 coupling design:**

| | Context: q only | Context: q + "Previous attempt: a0. Verdict: v." |
|---|---|---|
| **No update** | B1 fresh-context | context arm |
| **LoRA update from (q, a0, v)** | parameter arm | parameter + context |

One adapter per (item, grounding level) serves both update cells. The trained adapters per item are self, perm-self (one per permutation seed), ext-1.0 and shuffled-verdict; the random adapter is generated, not trained.

**Controls:**
- a matched-norm random LoRA (q-only context);
- a shuffled verdict (v taken from another item in the same stratum);
- a zero-step update, which must equal B1.

**Update rule** (frozen in the prereg after dev tuning):
- **Training sequence:** the chat-rendered (user: q, assistant: a0), the same template as the q-only re-answer. v is **not** in the sequence; it only selects the loss.
- **Loss mask:** the final-answer span of a0, for both verdicts, averaged over the masked tokens.
  - v = correct: NLL.
  - v = incorrect: unlikelihood −log(1 − p), computed stably as −log1p(−exp(log p)), with log p clamped to ≤ −1e-6.
- These are two different algorithms; conclusions are limited to this specific update rule.
- **Model and optimizer:** LoRA rank 8 on q, k, v, o; A random (seeded), B = 0. LoRA parameters in fp32, base in bf16. A fresh AdamW per item, k steps (default 4). The learning rate is tuned on dev for 1.7B and 4B; 8B follows the §9 rule.
- **Norm.** N(ΔW) = sqrt(Σ_m ‖ΔW_m‖_F²) (global Frobenius norm over modules m), with ΔW_m = (α/r) B_m A_m. After the k steps, all B_m are multiplied by a single factor c = τ/N(ΔW), which preserves the proportions across modules. τ is chosen on dev, using the KL probe set only. Random controls are drawn per module, then globally rescaled to τ.
- **Logged per update:** N before rescaling, c, the realized N, the mean |Δlogit|, and the KL divergence on the KL probe set.
- **Anomalies** (rules frozen in advance; nothing is dropped after results are seen):
  - no extractable answer span, N = 0, or NaN/Inf: the update is not performed, the item keeps the base model (intention-to-treat), and it is counted;
  - c > 100: still applied, flagged, and analysed in a pre-registered sensitivity analysis that excludes the flagged items.
- **Per-item reset:** discard the adapter, and assert that the logits equal the base logits.

**Evaluation sets** (fixed IDs drawn at split time, §5.2):
- **KL probe set:** 50 dev problems, used for τ tuning.
- **Spillover set:** 50 test problems, removed from the Exp 5 update pool and never used for tuning.

**Backends.**
- Every Exp 5 accuracy cell, including B1 and the no-update cells, is generated with vLLM using the same configuration.
- Within vLLM, the zero-step adapter must reproduce base greedy tokens exactly.
- HF (training and teacher-forced scoring) vs vLLM agreement on dev: ≥ 98% identical greedy outputs with the zero adapter; otherwise the mismatch rate is reported, and generation-based comparisons stay within vLLM.

**Per-item protocol** (H4a, H4b): accuracy on q, greedy plus 4 samples (T = 0.7); spillover = the change in mean gold log-prob on the spillover set (teacher-forced, batched).

**Adapter storage.** Adapters are deleted after serving. Kept: the full rebuild inputs (item, verdict, seeds, config) and a seeded 5% sample of adapters. The saved dtype and size are measured in P0 (rank-8 q/k/v/o for 4B has 5.9M parameters: ≈ 11.8 MB in bf16, 23.6 MB in fp32).

**Persistent streams** (H4c; exploratory):
- 200-item sequence. At each step, a0 and the self-verdict are generated by the **current updated model** (an endogenous loop).
- Each step trains an increment with a fresh optimizer on the current adapter. τ limits the **per-step increment**, ΔW_t − ΔW_{t−1}. The total adapter norm is logged.
- Conditions: self / ext-1.0 / random / context-memory. The context-memory stream uses the same verdict source as the self stream, keeps a running (q, a0, v) log, drops the oldest entries first, and has a fixed token budget.
- Every 25 items, evaluate on 100 held-out dev problems (not the spillover set): greedy accuracy, answer diversity over 4 samples, verbal P(yes), probe–verbal agreement.
- 3 order seeds; descriptive only.

**Self-prediction** (H4d): a separate forward pass, "State only the final answer you would give to this question.", compared with the actual re-answer.

**Feasibility criteria** (dev; decide enabling, not success):
- zero-step = B1 exactly (greedy, vLLM);
- reset restores the logits;
- realized norm within 1% of τ;
- ext-1.0 parameter arm > B1 on dev.

If any fails, Exp 5 is **disabled** for confirmatory use and reported as "did not meet the feasibility criteria for this task and update rule". That outcome does **not** mean the update cannot carry information [M8].

---

## 7. Statistics

1. **Unit.** The canonical problem. CIs come from an item-clustered paired bootstrap (10,000 resamples) that resamples problems with all their variants and conditions. **Exception:** streams use the trajectory as the unit, descriptive only [C5].
2. **Tests.**
   - Paired binary two-condition contrasts: exact McNemar, with risk differences.
   - Trends and multi-factor models: GEE logistic regression, clustered by problem, exchangeable working correlation.
   - Equivalence: TOST (p = max of the two one-sided p).
   - Composite claims: intersection-union tests (p = max of the component p).
3. **AUROC differences.** Paired item bootstrap (primary); DeLong (check).
4. **Power** [M5].
   - Before the freeze, simulate each Tier 1 test and each candidate Tier 2 test by resampling the paired-outcome patterns from the 1.7B and 4B dev pilots.
   - Use **several scenarios**: the pilot discordance ±50%, and effect sizes {0, δ/2, δ} for equivalence and non-inferiority, or {δ/2, δ, the pilot estimate shrunk by 50%} for superiority.
   - Never plan on the tuned pilot effect alone.
   - Choose n for ≥ 0.8 power after Holm under the central scenario. If that is unaffordable, the hypothesis becomes estimation-only.
   - Simulated power goes into the prereg. For reference (`docs/reviews/review_support/design_checks.json`), ±2 pp TOST needs 2,141–6,423 paired items at 10–30% discordance.
5. **Multiplicity.** Holm over the family fixed at freeze (§3.1).
6. **Missing data.** Nothing is dropped; unparsed outputs count as incorrect. Infrastructure failures are rerun; if a rerun is impossible, the item is excluded from every condition of that contrast and counted.
7. **Prereg table.** One row per hypothesis: model, mode, dataset, estimand (risk difference), contrast, direction, test, unit, n, δ, family, missing-data rule, enabling criteria.
8. **Reporting.** Effect sizes and CIs for everything; decision rules from §3.2.

---

## 8. Integrity invariants (automated tests in `tests/`)

These checks apply to **enabled** modules. A failure is a bug: fix it, give the rerun a new run ID, and mark the old run superseded. A disabled module's feasibility failure is not an invariant failure (§3.1). **A hypothesis that fails is not a bug.**

| Area | Invariant |
|---|---|
| Exp 0 (programmatic) | A scripted binary agent reproduces the truth table exactly: verifier verdicts, switch-on-incorrect, option scoring, the paired-world 1-right/1-wrong property, and acc = p for the scripted switch policy over full masks. The scripted agent's verdict-ignoring variant gives a gain ratio of 0. Hidden bits and seeds are absent from rendered inputs. (LM behavior on Exp 0 is **not** an invariant [M2].) |
| Verifier | Determinism; nesting across p; keyed by canonical ID; F3-perm preserves stratum verdict counts exactly |
| Gold isolation | The counterfactual-label test (§5.8) [M3] |
| Prompts | Reviewed-solution span identical across conditions; the label precedes the solution |
| Scorer | Unit tests pass; agreement with a second scorer ≥ 99% on dev |
| Steering / patching | α = 0 reproduces the baseline; the reachability diagnostic for the chosen setting; a self-patch reproduces the baseline |
| Exp 5 | Zero-step = B1 within vLLM; reset restores logits; context-arm spillover is exactly 0; realized N = τ; HF/vLLM agreement as in Exp 5 |
| Statistics | Each test's false-positive rate at **its own null boundary**: superiority at 0; TOST at ±δ; non-inferiority at δ_w; intersection-union tests at the least-favourable component boundary (2,000 simulations, nominal ±1.5 pp); and recovery of a simulated known effect [M5] |
| Splits | No canonical ID in both dev and test; cross-source links honored; spillover and KL sets isolated |
| Sync | Every artifact in each run manifest is present on the control host with a matching sha256 [M10] |

---

## 9. Pre-registration timing

- **Freeze after:**
  - the 1.7B **and** 4B dev pilots (all tuning, including the 4B TTT learning rate and τ);
  - the feasibility decisions for the Tier 2 modules;
  - the power simulation;
  - all invariants of enabled modules passing;
  - the minimal related-work matrix.
- **8B:** no 8B tuning before the freeze. The prereg fixes an automated adaptation rule: run the same dev grid on 8B dev with the same selection criterion, then run test. The choice happens on dev, by the frozen rule, with no human or agent judgment.
- **Data:** the data rules in this document and the P1 data card are committed before any post-feedback test output is generated.

---

## 10. Block manifest, compute and budget stops [M11, C4]

**Block manifest.** `configs/blocks.yaml` is the single run order, across all models. Each block records: id, model, experiment, tier, priority, dependencies, estimated cost, minimum complete sample, and stop rule. P0 fills in the cost estimates. Priority, cut from the bottom when the budget runs short:

1. P0 benchmark; P1 data (all models' initial answers); P2 dev pilots (1.7B, 4B)
2. 4B Tier 1: Exp 0 → Exp 1A → Exp 1B → Exp 2
3. 4B enabled Tier 2: Exp 5 per-item (H4a) → Exp 3 (H3b) (H2b reuses Exp 1A/2 outputs)
4. 8B core replication: Exp 1A + Exp 1B
5. 1.7B Tier 1 replication
6. 4B thinking subset
7. 8B Exp 2
8. Exp 1C, sampled robustness, F2-scratch
9. Exp 4
10. Streams (4B)
11. The rest of 8B; 1.7B exploratory blocks

**Complete paired chunks.** Every block processes items in the seeded order (§5.2), in chunks that complete **all conditions** for those items before moving on. All chunks of a block run on one GPU type.

**Budget stops** are recorded in the manifest:
- a block that reached its planned n is analysed as planned;
- a block that stopped above its pre-registered minimum n is a *pre-registered partial-sample analysis*, labelled as such;
- a block below its minimum n is descriptive only.

A stopped block is never reported as a null.

**Estimate** = Σ over blocks of separate terms for prefill, decode, teacher-forced scoring, backward updates, model and adapter loading, refits (probes, bootstrap), and transfers, plus 30% overhead and retries. P0 measures each rate separately; there is no single tokens/s figure.

**Volumes** (unexpanded pool):
- Exp 1A: ≈ 1,455 test items × ≈ 13 generations.
- Exp 1B: (D-inj test + D-nat test) × 12 cells.
- Exp 5 per-item (≈ 400 items):
  - trained adapters per item: self, 3 perm-self, ext-1.0, shuffled, i.e. ≈ 6;
  - plus a generated random adapter;
  - each trained adapter is evaluated in 2 contexts;
  - 50-item spillover scoring per adapter;
  - 5 generations per cell.

**Prior** (A100 80GB, vLLM; to be replaced by P0): about 27 GPU-h, plus 30% ≈ 35 GPU-h ≈ $52 of compute at $1.49/h.

The instance runs on A100-80GB (Liam's choice at creation). On the A100 40GB that was recommended instead ($0.89/h; same tensor compute, ≈ 80% of the memory bandwidth), the same work is ≈ 35–44 GPU-h ≈ $31–39. The upper end assumes fully bandwidth-bound decode. Storage is extra (CLAUDE.md §6).

---

## 11. Paper outline

1. **Introduction:** performative vs structural reflection; the three confounds in "external feedback"; contributions.
2. **Framework:** operational definitions (data port vs parameter port of the released network); Claim 7 and its bounded-observer reading; the theory-status table; boundary relativity; what a frozen study can and cannot show.
3. **Related work:** from the matrix.
4. **Setup:** models, data, splits, verifiers (including the oracle permutation control), conditions, scorer, invariants.
5. **Results 0:** Exp 0 calibration.
6. **Results I:** decomposition (Exp 1A), with W→R and R→W.
7. **Results II:** internal signal vs verbal report (Exp 2).
8. **Results III:** role, label and the blind spot (Exp 1B, Exp 3, Exp 4).
9. **Results IV:** operator-level contrast (Exp 5).
10. **Discussion:** what this says, and does not say, about Claim 7; the alignment diagnostic (manifesto §17).
11. **Limitations:** scale ≤ 8B; one family; linear accessibility; minimal self-assessment signal in the TTT arm; a specific update rule; simulated and oracle-permuted verifiers; possible contamination; the artificial Exp 0; unverified theory.
12. **Ethics; LLM-use disclosure.**

---

## 12. Known risks and mitigations

| Risk | Mitigation |
|---|---|
| Too few natural errors | Unbalanced estimands; MATH; D-inj is primary for H3a; pool expansion if the power simulation calls for it |
| Role/template confound | Role × label 2 × 2, with the label before the solution; alternative-template replication; causal test with random and generic controls; token-level span checks |
| TTT instability | Fixed realized norm; anomaly rules; dev tuning; feasibility criteria; the sweep reported in the appendix |
| Single model family | Stated as a limitation; a non-Qwen ≤ 4B Exp 1A only if budget allows (new family = human gate) |
| Training-data contamination | Pairing controls problem identity, but exposure can still change how conditions respond and limits generalization. Stated as a limitation; not claimed to be harmless [C5] |
| Exp 0 is artificial | It calibrates the pipeline; no claim about natural tasks |
| Budget stop | Block manifest; random processing order; complete paired chunks; partial-sample rules |
