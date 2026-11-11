"""Independent arithmetic for the design review; these are NOT model results."""

import json
from math import ceil, sqrt
from pathlib import Path
from statistics import NormalDist


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def spectral_radius_real_2x2(a):
    trace = a[0][0] + a[1][1]
    determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    discriminant = trace * trace - 4 * determinant
    if discriminant < 0:
        raise ValueError("This check only handles real eigenvalues.")
    return max(abs((trace + sqrt(discriminant)) / 2), abs((trace - sqrt(discriminant)) / 2))


z95 = NormalDist().inv_cdf(0.95)
z90 = NormalDist().inv_cdf(0.90)
precision = []
for n in (400, 800):
    for discordance in (0.1, 0.2, 0.3):
        precision.append({
            "n_independent_paired_items": n,
            "assumed_discordance": discordance,
            "normal_90pct_CI_halfwidth_pp_at_zero_effect": 100 * z95 * sqrt(discordance / n),
            "approx_n_for_80pct_TOST_power_at_zero_effect": ceil((z95 + z90) ** 2 * discordance / 0.02 ** 2),
        })

a = [[1, 1], [0, 1]]
b = [[1, 0], [1, 1]]
ab = matmul(a, b)
output = {
    "purpose": "Design arithmetic and an algebraic counterexample, not empirical LLM results",
    "equivalence_assumptions": {
        "margin": 0.02,
        "one_sided_alpha": 0.05,
        "true_paired_mean_difference": 0,
        "variance_of_item_difference": "discordance at zero mean",
        "limitations": "Normal approximation, independent items, known planning discordance, no Holm adjustment. Use pilot-based simulation for final design.",
    },
    "paired_equivalence": precision,
    "spectral_counterexample": {
        "A": a, "B": b, "AB": ab,
        "rho_A": spectral_radius_real_2x2(a),
        "rho_B": spectral_radius_real_2x2(b),
        "rho_AB": spectral_radius_real_2x2(ab),
        "claimed_upper_bound": spectral_radius_real_2x2(a) * spectral_radius_real_2x2(b),
        "inequality_holds": spectral_radius_real_2x2(ab) <= spectral_radius_real_2x2(a) * spectral_radius_real_2x2(b),
    },
    "budget_and_volume": {
        "assumed_gpu_USD_per_hour": 3.49,
        "three_hours_USD_excluding_other_costs": 3 * 3.49,
        "max_hours_with_15pct_of_11USD_reserved_excluding_other_costs": 0.85 * 11 / 3.49,
        "protocol_watchdog_hours": 11 / 3.49 - 10 / 60,
        "cloud_spillover_item_evaluations_if_300_items_x_3_updates_x_50": 300 * 3 * 50,
        "local_spillover_item_evaluations_if_400_items_x_3_updates_x_50": 400 * 3 * 50,
        "exp1_test_generations_if_two_400_item_datasets_x_20_conditions": 2 * 400 * 20,
        "nominal_4_billion_bf16_weight_GiB_excluding_all_overheads": 4e9 * 2 / 2**30,
    },
}

destination = Path(__file__).with_name("design_checks.json")
destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(output, ensure_ascii=False, indent=2))
