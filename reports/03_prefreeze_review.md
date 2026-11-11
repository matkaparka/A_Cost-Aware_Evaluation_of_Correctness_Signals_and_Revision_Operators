# P3 pre-freeze review (2026-11-05)

**Reviewer:** a fresh subagent that had not seen the drafting. It was told to act as a skeptical reviewer of prereg.md at commit 9300bba, to read the protocol, the design, DECISIONS and the cited evidence, and to edit nothing. Its findings are summarized below in its own order, each with my response.

**Its overall verdict:** not ready to freeze until findings 1–3 are fixed. Findings 4, 7, 8, 12 and 13 should be fixed, the rest disclosed.

**No test-split violation.** Every item in the P2/P3 outputs is a dev item, and scope decisions used only test right/wrong counts. 22 spot-checked numbers matched their sources.

| # | Severity | Finding (short) | Response | Where |
|---|---|---|---|---|
| 1 | blocker | H3a outcome categories undefined: two-sided, no predicted sign | Fixed. Non-directional rules: supported = significant either way; not supported = CI inside ±δ; no "contradicted" | prereg §3.1, §5, §14 #15 |
| 2 | blocker | H2a test not operationalized; no size test | Fixed. `stats.auroc_diff_boot` (cluster bootstrap, p rule, CI type, ties 1/2), null-boundary and known-effect tests; DeLong as a check, with a disagreement rule | analysis/stats.py; tests/test_stats.py; prereg §3.1 |
| 3 | blocker | Scorer amendment changed twice; "one-directional" false; "equivalent" overstated | Fixed. Both amendments and their timing stated; the false claims corrected; v2 labels added as a sensitivity analysis. Not done: the v2 hand-check sample stays at 40 (enlarging it after the fact would not change the record) | prereg §1; DECISIONS 2026-11-05 (corrections); 03_prereg.md |
| 4 | major | Infeasible-central rule post hoc; it keeps H1b and drops H1d | Kept, disclosed. Added: H1b "not supported" margin δ/2; "supported (effect below δ)" wording; strict reading reported as a sensitivity analysis (Holm over 5) | prereg §3.2, §5; analysis/p6_confirmatory.py |
| 5 | major | H1a equivalence nearly guaranteed by low discordance | Disclosed. Wording limit in §0: report discordance and W→R/R→W next to the TOST; banned phrasings | prereg §0 |
| 6 | major | Population change large; contamination risk in the expansion | Kept, disclosed. The population is named in every headline. Original-pool, reweighted and per-stratum estimates go in the main table. A heterogeneity check (math_ext − original) and a contamination caveat added | prereg §0, §2 |
| 7 | major | Partial expansion block lets the estimand drift | Fixed structurally. Runner blocks are atomic, so the expansion is complete or absent. If absent, every D-nat test, H2a included, uses the original pool | prereg §2, §10; configs/blocks.yaml |
| 8 | major | Departures from design v2.1 not collected | Fixed: table of 16 departures with reasons and timing | prereg §14 |
| 9 | major | Timing of the central-scenario definition and of δ_AUC not disclosed | Fixed: timing paragraph | prereg §3.2 |
| 10 | major | Scorer check covers equivalence only; no per-condition breakdown | Fixed by pre-registered sensitivity analyses: all family estimates with v2 labels, plus per-condition disagreement rates | prereg §8; analysis/p6_confirmatory.py |
| 11 | major | Truncated a0 common among expansion errors | Fixed: sensitivity analysis without truncated/unparsed a0 for D-nat contrasts and H2a | prereg §8 |
| 12 | major | min_n hand-set | Fixed: analysis/p3_min_n.py computes min_n and checks blocks.yaml | reports/03_min_n_tables.md |
| 13 | major | No frozen confirmatory analysis code; open choices (CI type, Holm timing, t* flag, R→W order, α = 0 failure) | Fixed: analysis/p6_confirmatory.py, dry-run on dev; cloud/p4a_sequence.sh pins runner flags, t*, inputs and the probe hash. R→W order uses src/data/build.py `h`; the Exp 3 GPU fallback is stated | prereg §7, §8, §10 |
| 14 | major | H2a is pre-feedback, so the post-feedback guard does not cover it | Fixed: header states no test AUROC, probe score or activation exists; first access logged | prereg header, §8 |
| 15 | major | Per-block P4 analyses expose results before later family blocks | Fixed: P4 reports estimates and CIs only; Holm and categories only at P6; deviations list the results seen | prereg §4, §8 |
| 16 | major | "Self-verdict" overstates F5 | Fixed: wording limit ("thresholded verbal readout", threshold stated) | prereg §0 |
| 17 | major | H3b main setting rests on mixed references and a noise-level gain | Disclosed in §7 | prereg §7 |
| 18 | minor | Tests ignore clustering | Fixed: cluster-robust SEs, with a size test on clustered data | analysis/stats.py; tests/test_stats.py |
| 19 | minor | H2a scenarios calibrated on pooled dev, applied to the original scope | Fixed: calibration per scope. **This changed a conclusion**: central-scenario power on the original pool is 0.33, not 0.88, so H2a needs the expansion too (pooled 0.99). Corrected in the prereg and DECISIONS | analysis/p3_power_h2a.py; reports/03_power_h2a_tables.md |
| 20 | minor | H1b GEE-vs-z check thin, original scope only | Fixed: also on the pooled scope | analysis/p3_power_tier1.py |
| 21 | minor | DECISIONS misstates H2b power | Corrected in a new entry | DECISIONS 2026-11-05 |
| 22 | minor | "Seen so far" omits test status counts | Fixed | prereg header |
| 23 | minor | Freeze checklist gaps; no expansion data card | Fixed: checklist names the matrix and data cards; analysis/p3_data_card_ext.py | prereg §13; 03_data_ext_tables.md |
| 24 | minor | "133 passed" hand-typed | Fixed at the freeze: the pytest output is saved as an artifact | reports/03_pytest_freeze.txt |
| 25 | minor | Stale "running" registry entry | Fixed: marked stopped+superseded with the reason | runs.jsonl |
| 26 | minor | Gate decisions unlabelled | Fixed: labelled (blanket go-ahead) and surfaced in STATUS | DECISIONS 2026-11-05; STATUS.md |
| 27 | minor | H2b reading of an estimation-only CI not pre-stated | Fixed: wording for CIs above, below or across δ_w | prereg §3.4 |

**What I did not change, and why.**
- **The pooled estimand, the infeasible-central rule and the family rule stay.** The reviewer judged each defensible because it rests on nuisance parameters and n. It also noted that each choice happens to keep a likely-positive hypothesis confirmatory. That point is now stated in the prereg's own text (§3.2, §14), not only in DECISIONS, and the strict reading is a sensitivity analysis.
- **The v2 hand-check sample was not enlarged.** Adding cases after the result would not repair the record. The bound uses the 40 that were checked.

## Re-check (same reviewer, at commit 0d2378b plus the uncommitted working tree)

**All three blockers resolved.** No outcome category is misassigned by the analysis code. The reviewer ran the new tests and its own size simulation: 5.3 % at α = 0.05 for the AUROC bootstrap. It reproduced min_n = 283. It confirmed that the regenerated Tier 1 values match the prereg and that the per-scope H2a recalibration is correct.

New findings, and my responses:

| # | Severity | Finding | Response |
|---|---|---|---|
| 4 | major | The committed code lacks rules the committed prereg describes ("supported (effect below δ)", "method-sensitive", the Exp 3 min_n rule) | They were in the working tree. Committed with a dry run regenerated at 10,000 resamples; the hashes are recorded at the freeze |
| 5 | major | H3b, a family member, had no test-split generation code | Written before the freeze: src/steer/test.py, with a frozen-directions hash check, the §7 pre-check, 100-item chunks and resume. Dry run on dev and test; smoke-tested locally on 6 dev items, all 9 arms (the laptop's incomplete model cache stopped only the final manifest write). Pinned in cloud/p4b_sequence.sh |
| 6 | minor | p6 omits H1a context, H3b R→W, and the v2 sensitivity for H3b | Added: discordance and per-arm W→R/R→W next to H1a; a per-arm Exp 3 table (W→R D-inj, R→W D-nat); H3b with v2 labels from the saved texts |
| 7 | minor | The freeze hash list misses files the analysis depends on | Extended to every file on the confirmatory path (prereg §13) |
| 8 | minor | p4a continues after a failed step; activation runs picked by mtime | It now aborts on the first failure, and activations are addressed by run ID (locked commit) |
| 9 | minor | No wording guard for a negative H3a sign | Added to §0 |
| 10 | minor | The no-expansion fallback keeps two weak tests confirmatory | Labelled an "underpowered pre-registered fallback", with its powers stated (§2) |
| 11 | minor | The AUROC size test was looser than design §8 | Tightened to ±1.5 pp over 2,000 simulations; it passes |

Also from the re-check: the H3b test n is 1547, not the projected 1572 (88 test D-inj items have no solution content). This is counted by the module's dry run, and blocks.yaml, min_n and the prereg are updated. The secondary Exp 3 settings are compared with unsteered only (departure 17, for cost).

The reviewer judged the document freezable once findings 4 and 5 were closed. Both are closed, so the prereg is frozen without a third review round.
