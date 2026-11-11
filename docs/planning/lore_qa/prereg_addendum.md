# Lore QA: pre-registration addendum

Status: frozen 2026-11-07, before any lore post-feedback output on test exists. Commit hash in STATUS.md and in the commit that adds this file. Changes after this point go to DEVIATIONS.md.

This is an exploratory addition after the main prereg froze (prereg.md, commit 08657ca). It is outside the Holm family. Every lore result is **estimation only**: estimates and 95 % cluster-bootstrap CIs, no tests, no "supported" or "contradicted" categories. Plan: [proposal.md](proposal.md). Dev pilot: [reports/lore_01_pilot.md](../../../reports/lore_01_pilot.md).

## 1. What is fixed

| Item | Value |
|---|---|
| Model | Qwen/Qwen3-4B, the revision in configs/models.yaml, bf16 |
| Backend | vLLM greedy, `VLLM_BATCH_INVARIANT=1` (generation); HF for verbal P(yes) and activations |
| Prompts | the main study's wrappers (src/gen/render.py, sha256 `8876e630…d489d5`) around the lore question (src/data/lore.py `render_question`) |
| Token budgets | a0 4096; revision 2048; critique 1024 (prereg §1) |
| Items | items_a.jsonl `afe20a80…b2595`, items_b.jsonl `c7596116…2379`, items_inj.jsonl `ae71f8bb…37030`, passages.jsonl `850d1087…79d90`. Test: 692 items; dev: 171 |
| Scorer | src/score/lore.py 1.1.0, sha256 `d980f57b5fcac6d58b27fc58583a58b692705e36387b4d6a7f1a87e68ecc9251` |
| Runner | src/exp/lore_run.py, sha256 `0fdf77db1836cc9d44fc4fcb6feb062c75a93eadb7ec159cd91ebc24eecaf36a` |
| Analysis | analysis/lore_results.py, sha256 `362f6bdb28d6155d5b9751d101c3cdcc97342eaf96a70208d71c26b1d8b212fb` |
| Bootstrap | 10,000 resamples, seed 0, clusters = the item's `cluster` field |

## 2. Gates, as applied (dev only)

From the HF dev pilot ([reports/lore_01_pilot.md](../../../reports/lore_01_pilot.md); DECISIONS 2026-11-07):

| Gate | Result | Consequence |
|---|---|---|
| G1, gold answers | passed: Liam's audit, 60 of 60 right | — |
| G2, difficulty | failed: 4B dev accuracy 90.1, 87.7 and 88.3 % at 512, 1024 and 2048 (scorer 1.1.0), above 0.85 at every budget | D-nat-lore is descriptive; D-inj-lore is the primary pool |
| G3, parsing | passed: 0 % | — |
| G3b, phrase scoring | failed: 1.0.1 missed 20 of 62. Scorer 1.1.0 was fitted on those rows; on 89 held-out phrase pairs it disagrees with a blind judge on 10.1 % [5.4, 18.1], above the 10 % bound | phrase-class items are descriptive in every pool |
| G4, size | failed: 85 projected wrong test items, below 150 | D-nat-lore is descriptive |

The run host repeats the measurement with vLLM on dev before any test run, applying these rules mechanically (cloud/fu1_sequence.sh):
- **Budget B:** the smallest budget whose 4B dev accuracy is at most 0.85. If none qualifies, the budget with the lowest dev accuracy; on a tie, the smaller.
- **G3:** at B, unparsed + truncated + conflict must be at most 5 %. If it is higher, no lore test conditions run.
- **G2 and G4:** recomputed and reported. The pool status above stands whatever they show: it was set by the dev pilot and is not re-decided on the run host.

## 3. Conditions (proposal §6)

- **Initial answers** for every test item at B. The dev rows run at all three budgets.
- **Verbal P(yes)** on a0, as in design §5.7.
- **Exp 1A core on D-nat-lore:** B0, B1, F0, F1, F2, F2-scratch, F3(p) for p ∈ {0.5, 0.7, 0.9, 1.0}, F5, F3-perm (K = 3).
  - F3 verdicts use the frozen f3_verdict stream.
  - F5's threshold t* is chosen on lore dev verbal scores by the main study's rule: the midpoint that maximizes balanced accuracy.
  - F3-perm strata are route × initial correctness.
- **Exp 1B (role × label, F0)** on D-nat-lore and on D-inj-lore at B.
- **Exp 2** on test at B:
  - (a) the frozen 4B math probe (results/p3/4B/p3-probe-dev-ext-4B-b7cf0d6/probe.npz, sha256 `a62621d9…c8bc833`), applied unchanged;
  - (b) a probe selected on lore dev by the src/probe/train.py rule (layers × C ∈ {0.001, 0.01, 0.1, 1}, 5-fold grouped CV by cluster) and refit on dev.

## 4. Estimands (analysis/lore_results.py)

1. **Primary pool: D-inj-lore, name-class rows, at B.** For each role × label cell at F0:
   - W→R on injected-wrong rows;
   - R→W on injected-right rows;
   - net accuracy.

   Then these contrasts, each paired within row, on W→R and on R→W:
   - role effect: user − assistant, averaged over labels;
   - label effect: other − self, averaged over roles;
   - interaction: (u/o − u/s) − (a/o − a/s);
   - diagonal: u/o − a/s.
2. **Descriptive:**
   - D-inj-lore phrase rows, and name rows by route;
   - D-nat-lore, name items and all items: per condition n, accuracy, W→R and R→W (k/n), truncation and not-ok rates;
   - paired accuracy contrasts F0 − B0, B1 − B0, F1 − F0, F2 − F0, F2-scratch − F0, F3(1.0) − F0, F3(0.5) − F0, F5 − F0, and F5 − mean(F3-perm).
3. **Exp 2:** AUROC for initial correctness of four scores:
   - verbal s;
   - the math probe;
   - the lore-dev probe;
   - minus answer length.

   Each score minus verbal is estimated by a paired cluster bootstrap, on name items (main) and on all items.

Reporting follows proposal §6:
- every condition with net accuracy, W→R, R→W, truncation and unparsed rates;
- routes A and B, and D-inj-lore, reported separately;
- no pooling with the math results; side-by-side discussion of direction only;
- no "one interval excludes 0, the other does not" in place of a direct comparison.

## 5. Not decided here

The paper's wording about lore results is decided at P7 under the ledger rules (docs/evidence_ledger.md), with the labels "exploratory" and "estimation only".
