"""Read-only design counterexamples and arithmetic; no LLM or cloud runs."""

import hashlib
import json
from math import comb, sqrt
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]

# Both verdict policies have s_R = s_W = .8. Among 100 wrong items,
# 50 are repairable only when the verdict says "incorrect"; 50 cannot be fixed.
# On 100 right items a false "incorrect" verdict corrupts the answer.
cases = []
for self_detect_easy, self_detect_hard in ((30, 50), (50, 30)):
    self_accuracy = (80 + self_detect_easy) / 200
    external_accuracy = (80 + 40) / 200
    cases.append({
        "self_detects_wrong_easy": self_detect_easy,
        "self_detects_wrong_hard": self_detect_hard,
        "external_detects_wrong_easy": 40,
        "external_detects_wrong_hard": 40,
        "s_R_both": 0.8,
        "s_W_both": (self_detect_easy + self_detect_hard) / 100,
        "self_final_accuracy": self_accuracy,
        "external_final_accuracy": external_accuracy,
        "external_minus_self_pp": 100 * (external_accuracy - self_accuracy),
    })

n = 200
pmf = [comb(n, k) * 0.5**n for k in range(n + 1)]
def quantile(probability):
    cumulative = 0
    for k, probability_mass in enumerate(pmf):
        cumulative += probability_mass
        if cumulative >= probability:
            return k
    return n

lo, hi = quantile(0.025), quantile(0.975)
false_alarm = 1 - sum(pmf[lo:hi + 1])
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
paths = ("docs/reviews/v2_review_request.md", "docs/research_design.md", "CLAUDE.md", ".gitignore")
hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
ignore = subprocess.run(
    ["git", "check-ignore", "-v", "--no-index",
     "results/example/token_ids.jsonl", "results/example/prompt_token_spans.json",
     "reports/token_usage.json"], cwd=root, text=True, capture_output=True, check=False)

# From the pinned Qwen3-4B config fetched during review:
# 36 layers, hidden 2560, q=32*128, kv=8*128, rank=8.
adapter_params = 36 * 8 * ((2560 + 4096) + 2 * (2560 + 1024) + (4096 + 2560))
data = {
    "purpose": "V2 design audit; all examples are constructed, not observed model results",
    "reviewed_commit": head,
    "reviewed_file_sha256": hashes,
    "matched_reliability_counterexamples": cases,
    "hidden_bit_examples": {
        "binary_valid_policy_no_feedback_expected_accuracy": 0.5,
        "always_refuse_policy_expected_accuracy": 0,
        "valid_binary_switch_on_incorrect_expected_accuracy": "p, if the verdict concerns that initial valid choice",
        "no_feedback_accuracy_over_p_at_p_half": 0.5 / 0.5,
        "n_for_binomial_example": n,
        "central_binomial_acceptance_counts": [lo, hi],
        "false_alarm_probability_under_correct_pipeline": false_alarm,
    },
    "git_ignore_evidence": ignore.stdout.strip(),
    "compute_and_storage_arithmetic": {
        "unexpanded_D_nat_total": 1319 + 500,
        "approx_80pct_test": (1319 + 500) * 0.8,
        "prior_hours_with_overhead": 27 * 1.3,
        "compute_USD_at_stated_1_49_per_hour": 27 * 1.3 * 1.49,
        "cap_if_15pct_reserved_no_storage": 27 * 1.3 * 1.49 / 0.85,
        "200_GB_storage_USD_per_month_at_stated_rate": 200 * 0.10,
        "adapter_parameter_count_rank8_qkvo": adapter_params,
        "adapter_MB_bf16": adapter_params * 2 / 1e6,
        "adapter_MB_fp32": adapter_params * 4 / 1e6,
    },
    "norm_example": {
        "two_module_norms": [1, 1],
        "sum_of_frobenius_norms": 2,
        "global_frobenius_norm": sqrt(2),
    },
}
target = Path(__file__).with_name("v2_design_checks.json")
target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(data, ensure_ascii=False, indent=2))
