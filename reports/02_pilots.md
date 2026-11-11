# P2 — Dev pilots on Qwen3-1.7B and 4B (2026-11-04)

Every number is in a generated table: [Exp 0/1](02_pilot_exp01_tables.md), [Tier 2 feasibility](02_pilot_tier2_tables.md), [self-verdict and H1d](02_selfverdict_tables.md), and the power tables ([Tier 1 basic](02_power_tables.md), [H2a](02_power_h2a_tables.md), [H1b/H3a/H2b](02_power_more_tables.md), [H4a/H3b](02_power_tier2_tables.md)). Hand checks: [judgments](02_handcheck.md), [samples](02_handcheck_samples.md). This page holds no hand-typed numbers. Dev split only; the test split has had no post-feedback access.

## What was done
- **Every experiment end-to-end at small n on dev.**
  - Exp 0/1A/1B and the Tsui cell on both models.
  - Exp 2: probe selection by grouped CV, with a learning curve.
  - Exp 3 on 4B: directions, reachability, screening, confirmation against generic and random controls.
  - Exp 5 on both models: the τ/lr grid, the feasibility criteria, every grounding arm.
- **Tuned on dev with rules fixed in advance:**
  - the revision budget;
  - the self-verdict threshold t*, as log-odds;
  - probe layer and C;
  - steering layer, α and direction;
  - TTT τ and lr.
- **Power simulation** under several scenarios for every Tier 1 and candidate Tier 2 test. The undefined "central scenario" is now defined; see DECISIONS.
- **Hand checks:** ≥ 5 examples per metric, including the Exp 3 and Exp 5 generated texts. No label errors.

## Feasibility (design §3.1; [Tier 2 table](02_pilot_tier2_tables.md))
- **Exp 5:** feasible on both models. Every criterion passes, and so does the new same-update-twice invariant (4B: the deterministic rerun on A100-80GB, p2d).
  - The matched-norm random adapter also lifts dev accuracy over B1 on the tuning items. Greedy outputs of wrong items move under any perturbation.
  - So the ext-1.0 gain over B1 is an upper bound on what the verdict adds. The informative comparisons are against the random and shuffled arms.
- **Exp 3 (4B):** feasible. Reachability passes, and α = 0 reproduces the unsteered tokens. The generic control is invalid at the main setting, so H3b is limited to the random control (design §3.5).

## Findings that shape the prereg (DECISIONS 2026-11-04)
- **H1d:** structurally bounded on these models. F5 and F3-perm can differ only where the verdict differs *and* the answer follows it.
- **H2a (4B):** underpowered at the dev effect, even with a doubled test pool. Hence the P3 expansion: MATH test problems outside MATH-500, in a new stratum (`src/data/expand.py`).
- **H4a:** underpowered at n = 400 in every scenario.
- **H2b:** zero power under the pilot pattern. On 4B dev, critique repairs more errors than self-review, the reverse of the prediction.
- **H3a, H3b:** adequate in the central scenario.
- **H1a (4B):** below 0.8 at the first Holm step in the central scenario. I first missed this; it was corrected in DECISIONS. The expansion's test items would raise its n.
- **H1b:** adequate at the pilot slope and short of 0.8 at half of it; a δ-scaled scenario is added in P3.
- **Exp 3:** a null is likely. Unsteered 4B already corrects most injected errors, and the dev gain of the main setting over random directions is about zero.
- **Exp 5 context cells:** the model often keeps a0 when told it is incorrect. This is consistent with the parameter + context cell lying below the parameter arm.
- **1.7B:** its verbal self-verdict is "correct" on every dev item, and its Exp 0 picks hardly depend on the verdict (Exp 0 table).

## Problems and fixes (each one a new run ID, the old run superseded)
- Exp 5 training was not reproducible. It is now deterministic, with an invariant test.
- Exp 3 directions were NaN: items without content gave empty spans. "0 reachable" was a bug, not a result.
- Exp 3 markers were built from too few revisions.
- Scorer 1.1.0.
- Run-registry supersede marks were lost on re-pull.
- A 4B Exp 5 run hit CUDA OOM on a 24 GB card. I had not checked P0's peak VRAM.
- The A30 runtime was misestimated. I scaled from a failed run that had generated 1 of 23 arms.

## Next (P3)
1. Pre-feedback quantities on the expansion (`cloud/p3a_sequence.sh`).
2. Probe re-selection on the enlarged dev.
3. Re-run the H2a power; fix the expansion scope.
4. Fix δ_w (H2b), the Exp 5 test pool and the H1d/H2a/H4a status.
5. Write `prereg.md`; an independent pre-freeze review; freeze.
