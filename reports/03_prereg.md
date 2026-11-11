# P3 — Freezing the pre-registration (2026-11-05)

Every number is in a generated table: [Tier 1 power with the expansion](03_power_tier1_tables.md), [H2a power](03_power_h2a_tables.md), [pre-freeze checks](03_checks_tables.md), [scorer agreement](03_scorer_agreement.md), [block costs](00_compute_rates.md). Hand checks: [03_handcheck.md](03_handcheck.md). Decisions: DECISIONS.md, 2026-11-05. Dev split only; on test, only initial right/wrong counts were looked at.

## What was done
- **D-nat expansion data.** Initial answers and verbal P(yes) for 1.7B, 4B and 8B on 4,497 MATH test problems outside MATH-500; dev activations for 4B and 1.7B.
- **Dev Exp 1A pilots** on the expansion, and on the original dev with F5 at the dev threshold t*. On the conditions where the threshold plays no part, the texts are identical to the P2 run, although the batches were composed differently.
- **Probes re-selected** on the enlarged dev, by the procedure fixed in advance.
- **Power re-runs** for H1a, H1b, H1d, H2a and H2b, on three scopes: the original pool, pooled with the expansion, and reweighted to the original mix.
- **Pre-freeze checks:**
  - HF vs vLLM agreement;
  - the thinking-mode budget, chosen by a rule fixed before its pilot;
  - validity of the Exp 1B alternative wording;
  - scorer agreement with two second scorers, with hand checks.
- **The prereg** with every decision resolved, an updated block manifest, an independent pre-freeze review with a re-check, and the freeze.

## Decisions (DECISIONS 2026-11-05)
- **Expansion and estimand.**
  - The expansion enters the 4B D-nat hypotheses.
  - The estimand becomes the pooled pool as sampled: GSM8K test + MATH test, with MATH the majority.
  - The design's original-pool estimand is kept as secondary.
  - It was built for H2a, and both H2a and H1a need it to reach adequate power.
- **Family (Holm over 6):** H1a, H1b, H2a, H3a-role, H3a-label, H3b.
- **Estimation-only:** H1d (structurally bounded), H2b, H4a.
- **New rule for an infeasible central scenario.** Added before the freeze; it decides H1b (confirmatory) and H1d (estimation-only).
- **H2a.** With the expansion it is powered for an AUROC gap of 0.03, but the pooled dev gap is far smaller. A "not supported" or "inconclusive" result is the likely outcome.

## Problems (each logged)
- **Two failed sessions from my script errors:**
  - a missing environment variable;
  - an argument value that starts with "-".

  Neither command had been run locally before launch. Rule added: every new sequence command runs locally or as a dry run first.
- **A monitor error.** The session monitor mistook a copied log line for its own end and nearly paused a running job. It was stopped in time, and it now matches only its own sequence's end line.
- **The scorer invariant fails as written.** All but one of the disagreements have the main scorer accepting an answer, and it is right on every hand-checked case. The second scorer v2 was added after v1 failed, and the amended rule was changed again after the hand check. All of this is disclosed.
- **HF and vLLM greedy outputs rarely match end to end** (bf16 near-ties). Generation-based comparisons stay within one backend.

## Review and freeze
- **First review.** An independent reviewer (a fresh subagent, read-only) found 3 blockers, 14 major and 10 minor issues, and no test-split violation ([03_prefreeze_review.md](03_prefreeze_review.md)). The blockers:
  - H3a's outcome rules were undefined;
  - H2a's test was not operationalized;
  - the scorer-amendment record was inaccurate.
- **What changed in response:**
  - cluster-robust tests;
  - frozen confirmatory analysis code, dry-run on dev ([03_p6_dryrun.md](03_p6_dryrun.md));
  - atomic runner blocks with a scripted min_n;
  - pinned P4 sequences, including new test-split code for Exp 3;
  - a table of 17 departures from the design;
  - disclosures of when each rule was set relative to the dev results.
- **One fix changed a conclusion.** H2a's original-pool power was 0.33, not 0.88: a calibration error. So H2a also needs the expansion.
- **Re-check.** The reviewer confirmed the blockers resolved. The two new major findings were closed before the freeze.
- **Freeze.** prereg.md frozen at commit 08657ca, file sha256 `ed3b2745…`; 138 tests pass. The hashes of the 21 files on the confirmatory path are recorded in prereg §13.

## Next
P4: the 4B Tier 1 blocks (priority 2 of configs/blocks.yaml), then Exp 3 and Exp 5.
