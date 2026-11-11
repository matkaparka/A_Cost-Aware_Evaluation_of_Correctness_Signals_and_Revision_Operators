# Retrospective grouped threshold-transfer sensitivity

Generated from frozen derived records by `analysis/neurocomputing_crossfit_v2.py`. No model outputs, scorer calls, probe fitting, bootstrap intervals, or hypothesis tests are added.

This is a cost-calibration sensitivity analysis on a previously examined test pool. The analyst already knew the previous outcomes and the design is post hoc; cross-fitting cannot turn those records into a new independent validation set. The existing probes remain frozen.

Canonical problems (question stems for MMLU-Pro) are assigned by the fixed hash rule. For each held-out fold, threshold calibration uses only the other folds' signal values, initial-answer tokens, full-thinking tokens, and uniform-budget tokens. Correctness labels never enter threshold selection. Equal-score groups are kept together. If initial answers alone exceed the training target, KEEP is retained and the fold is marked infeasible. The held-out decisions are never changed to meet a held-out token budget.

The candidate threshold set is finer than the original whole-pool comparison's one-percentage-point share grid. Consequently, low budgets can permit a small escalation here even when that grid chose KEEP. This sensitivity changes calibration and candidate resolution together; it does not isolate the effect of splitting alone.

The table pools one held-out decision per item. Gain is against KEEP and gap is selective minus uniform accuracy, both in percentage points. Drift is the percentage cost excess over held-out uniform thinking. Positive drift is an observed budget overshoot. Fold-level targets, thresholds, infeasibility flags and both held-out-uniform and calibration-target drift are retained in JSON.

| Setting | B | Signal | Selected % | Gain pp | Gap pp | Selective total tok/item | Uniform tok/item | Drift % | Overshooting folds |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| math 4B | 256 | verbal | 0.65 | +0.40 | +0.24 | 853.27 | 852.40 | +0.10 | 3 |
| math 4B | 256 | probe | 0.69 | +0.49 | +0.34 | 849.47 | 852.40 | -0.34 | 2 |
| math 4B | 512 | verbal | 3.11 | +2.28 | +1.37 | 1096.21 | 1099.62 | -0.31 | 2 |
| math 4B | 512 | probe | 3.01 | +2.22 | +1.31 | 1093.83 | 1099.62 | -0.53 | 2 |
| math 4B | 1024 | verbal | 8.04 | +4.43 | +1.54 | 1601.12 | 1600.36 | +0.05 | 2 |
| math 4B | 1024 | probe | 8.00 | +4.53 | +1.64 | 1599.85 | 1600.36 | -0.03 | 3 |
| math 4B | 2048 | verbal | 17.91 | +6.35 | +1.15 | 2387.44 | 2378.11 | +0.39 | 3 |
| math 4B | 2048 | probe | 17.66 | +6.29 | +1.09 | 2368.40 | 2378.11 | -0.41 | 3 |
| math 8B | 256 | verbal | 1.03 | +0.67 | +0.40 | 873.81 | 867.62 | +0.71 | 2 |
| math 8B | 256 | probe | 0.95 | +0.61 | +0.34 | 863.63 | 867.62 | -0.46 | 2 |
| math 8B | 512 | verbal | 3.25 | +2.16 | +1.25 | 1112.74 | 1120.82 | -0.72 | 3 |
| math 8B | 512 | probe | 3.44 | +2.41 | +1.50 | 1127.93 | 1120.82 | +0.63 | 3 |
| math 8B | 1024 | verbal | 8.16 | +4.16 | +1.27 | 1616.71 | 1610.59 | +0.38 | 3 |
| math 8B | 1024 | probe | 8.19 | +3.94 | +1.05 | 1626.75 | 1610.59 | +1.00 | 2 |
| math 8B | 2048 | verbal | 17.99 | +6.14 | +0.83 | 2408.33 | 2404.67 | +0.15 | 4 |
| math 8B | 2048 | probe | 18.01 | +6.47 | +1.17 | 2396.55 | 2404.67 | -0.34 | 2 |
| MMLU-Pro 4B | 256 | verbal | 0.32 | +0.06 | -1.56 | 780.54 | 776.95 | +0.46 | 2 |
| MMLU-Pro 4B | 256 | probe | 0.35 | +0.20 | -1.42 | 777.07 | 776.95 | +0.02 | 2 |
| MMLU-Pro 4B | 512 | verbal | 6.02 | +1.65 | -1.04 | 1040.92 | 1054.42 | -1.28 | 3 |
| MMLU-Pro 4B | 512 | probe | 7.00 | +1.53 | -1.16 | 1057.76 | 1054.42 | +0.32 | 2 |
| MMLU-Pro 4B | 1024 | verbal | 19.15 | +2.83 | -1.65 | 1504.87 | 1505.24 | -0.02 | 3 |
| MMLU-Pro 4B | 1024 | probe | 20.25 | +2.60 | -1.88 | 1514.33 | 1505.24 | +0.60 | 4 |
| MMLU-Pro 4B | 2048 | verbal | 35.41 | +4.02 | -2.40 | 2055.05 | 2057.70 | -0.13 | 2 |
| MMLU-Pro 4B | 2048 | probe | 38.15 | +3.62 | -2.81 | 2058.85 | 2057.70 | +0.06 | 4 |

Limitations: folds share an already selected study population; this does not remove analysis-selection bias, validate a new deployment threshold, or quantify new-generation variability. The calibration target itself is the in-fold realized uniform cost, not an externally fixed serving budget. Costs vary when the frozen threshold transfers. No cost normalization is applied to the accuracy gaps, so they must be read together with drift. Generated tokens omit scoring, prefill, caching and hardware costs. Cross-model cascades are outside this within-model sensitivity; token counts across different model sizes do not establish comparable compute costs.

Checks:

- math 4B: reconstructed full thinking and every uniform budget reproduce historical accuracy, tokens and forced-item counts.
- math 8B: reconstructed full thinking and every uniform budget reproduce historical accuracy, tokens and forced-item counts.
- MMLU-Pro 4B: reconstructed full thinking and every uniform budget reproduce historical accuracy, tokens and forced-item counts.
- All input files match checksums in the historical total-budget analysis manifest.
- Every canonical group belongs to one fold; no calibration/held-out group overlap; mathematics models share identical item-level folds.
- Every training threshold passes an independent exhaustive whole-score boundary cost check.
- Every item receives exactly one out-of-fold decision; pooled selections and costs reproduce sums of held-out fold totals.
- Calibration receives no correctness labels; no selection depends on held-out token counts; all budget overshoots are retained.
