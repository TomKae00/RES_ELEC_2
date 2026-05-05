# task_1_3.py

from __future__ import annotations

import pandas as pd
import numpy as np

from helpers import (
    ensure_output_folders,
    prepare_scenario_data,
    subset_combined_scenarios,
    make_k_folds,
    evaluate_one_price_across_scenarios,
)

from task_1_1 import solve_one_price_offering_problem
from task_1_2 import (
    solve_two_price_offering_problem,
    evaluate_two_price_across_scenarios,
)


def main():
    ensure_output_folders()

    wind_file = "Data/scen_zone2.csv"
    price_file = "Data/DayAheadPrices.csv"

    data, combined = prepare_scenario_data(
        wind_scenario_file=wind_file,
        price_file=price_file,
        n_wind_scenarios=20,
        n_price_scenarios=20,
        n_imbalance_scenarios=4,
        deficit_probability=0.5,
        seed=42,
        price_area="DK2"
    )
    k = 16
    folds = make_k_folds(combined.scenarios, k=k, seed=42)

    results = []

    for fold_id in range(k):
        print(f"\nRunning fold {fold_id + 1}/{k}")

        in_sample_scenarios = folds[fold_id]

        out_sample_scenarios = [
            sc for i, fold in enumerate(folds)
            if i != fold_id
            for sc in fold
        ]

        in_combined = subset_combined_scenarios(combined, in_sample_scenarios)
        out_combined = subset_combined_scenarios(combined, out_sample_scenarios)

        # -------------------------
        # One-price model
        # -------------------------
        offer_one, obj_one = solve_one_price_offering_problem(data, in_combined)

        one_in_profits = evaluate_one_price_across_scenarios(
            offer_one, data, in_combined
        )
        one_out_profits = evaluate_one_price_across_scenarios(
            offer_one, data, out_combined
        )

        results.append({
            "fold": fold_id + 1,
            "scheme": "one-price",
            "optimization_objective": obj_one,
            "in_sample_expected_profit": np.mean(one_in_profits),
            "out_sample_expected_profit": np.mean(one_out_profits),
            "generalization_gap": np.mean(one_in_profits) - np.mean(one_out_profits),
        })

        # -------------------------
        # Two-price model
        # -------------------------
        offer_two, obj_two = solve_two_price_offering_problem(data, in_combined)

        two_in_profits = evaluate_two_price_across_scenarios(
            offer_two, data, in_combined
        )
        two_out_profits = evaluate_two_price_across_scenarios(
            offer_two, data, out_combined
        )

        results.append({
            "fold": fold_id + 1,
            "scheme": "two-price",
            "optimization_objective": obj_two,
            "in_sample_expected_profit": np.mean(two_in_profits),
            "out_sample_expected_profit": np.mean(two_out_profits),
            "generalization_gap": np.mean(two_in_profits) - np.mean(two_out_profits),
        })

    results_df = pd.DataFrame(results)

    summary_df = (
        results_df
        .groupby("scheme", as_index=False)
        .agg({
            "in_sample_expected_profit": "mean",
            "out_sample_expected_profit": "mean",
            "generalization_gap": "mean",
        })
    )

    results_df.to_csv("outputs/tables/task_1_3_cross_validation_results.csv", index=False)
    summary_df.to_csv("outputs/tables/task_1_3_cross_validation_summary.csv", index=False)

    print("\nCross-validation results:")
    print(results_df)

    print("\nAverage summary:")
    print(summary_df)

    print("\nSaved:")
    print(" - outputs/tables/task_1_3_cross_validation_results.csv")
    print(" - outputs/tables/task_1_3_cross_validation_summary.csv")


if __name__ == "__main__":
    main()