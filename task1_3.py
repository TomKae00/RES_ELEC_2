# task_1_3.py

from __future__ import annotations

import numpy as np
import pandas as pd

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


def evaluate_cross_validation_fold(
    data,
    combined,
    in_sample_scenarios,
    out_sample_scenarios,
    fold_id: int,
    k_folds: int,
) -> list[dict]:
    """
    Run one cross-validation fold for both one-price and two-price settlement.

    The offering model is solved on the in-sample scenarios. The resulting
    fixed day-ahead offer is then evaluated on both the in-sample scenarios
    and the out-of-sample scenarios.

    Parameters
    ----------
    data : ScenarioData
        Scenario data containing wind production, day-ahead prices, system
        imbalance indicators, and balancing prices.

    combined : CombinedScenarioSet
        Full combined scenario set used as the basis for creating scenario
        subsets.

    in_sample_scenarios : list
        Scenario tuples used to optimize the day-ahead offer in this fold.

    out_sample_scenarios : list
        Scenario tuples used only for ex-post evaluation in this fold.

    fold_id : int
        Current fold number, starting from 1.

    k_folds : int
        Total number of folds in the current cross-validation run.

    Returns
    -------
    list[dict]
        List with one result dictionary for the one-price scheme and one result
        dictionary for the two-price scheme.
    """

    fold_results = []

    in_combined = subset_combined_scenarios(
        combined=combined,
        selected_scenarios=in_sample_scenarios,
    )
    out_combined = subset_combined_scenarios(
        combined=combined,
        selected_scenarios=out_sample_scenarios,
    )

    # -------------------------
    # One-price model
    # -------------------------
    offer_one, obj_one, stats_one = solve_one_price_offering_problem(
        data=data,
        combined=in_combined,
    )

    one_in_profits = np.array(
        evaluate_one_price_across_scenarios(
            offer=offer_one,
            data=data,
            combined=in_combined,
        )
    )
    one_out_profits = np.array(
        evaluate_one_price_across_scenarios(
            offer=offer_one,
            data=data,
            combined=out_combined,
        )
    )

    fold_results.append(
        {
            "k_folds": k_folds,
            "fold": fold_id,
            "scheme": "one-price",
            "n_in_sample": len(in_sample_scenarios),
            "n_out_sample": len(out_sample_scenarios),
            "optimization_objective_eur": obj_one,
            "in_sample_expected_profit_eur": float(np.mean(one_in_profits)),
            "out_sample_expected_profit_eur": float(np.mean(one_out_profits)),
            "generalization_gap_eur": float(
                np.mean(one_in_profits) - np.mean(one_out_profits)
            ),
            "out_sample_min_profit_eur": float(np.min(one_out_profits)),
            "out_sample_max_profit_eur": float(np.max(one_out_profits)),
            "out_sample_std_profit_eur": float(np.std(one_out_profits)),
            "solve_time_s": stats_one["solve_time_s"],
            "variables": stats_one["variables"],
            "constraints": stats_one["constraints"],
        }
    )

    # -------------------------
    # Two-price model
    # -------------------------
    offer_two, obj_two, stats_two = solve_two_price_offering_problem(
        data=data,
        combined=in_combined,
    )

    two_in_profits = np.array(
        evaluate_two_price_across_scenarios(
            offer=offer_two,
            data=data,
            combined=in_combined,
        )
    )
    two_out_profits = np.array(
        evaluate_two_price_across_scenarios(
            offer=offer_two,
            data=data,
            combined=out_combined,
        )
    )

    fold_results.append(
        {
            "k_folds": k_folds,
            "fold": fold_id,
            "scheme": "two-price",
            "n_in_sample": len(in_sample_scenarios),
            "n_out_sample": len(out_sample_scenarios),
            "optimization_objective_eur": obj_two,
            "in_sample_expected_profit_eur": float(np.mean(two_in_profits)),
            "out_sample_expected_profit_eur": float(np.mean(two_out_profits)),
            "generalization_gap_eur": float(
                np.mean(two_in_profits) - np.mean(two_out_profits)
            ),
            "out_sample_min_profit_eur": float(np.min(two_out_profits)),
            "out_sample_max_profit_eur": float(np.max(two_out_profits)),
            "out_sample_std_profit_eur": float(np.std(two_out_profits)),
            "solve_time_s": stats_two["solve_time_s"],
            "variables": stats_two["variables"],
            "constraints": stats_two["constraints"],
            "both_delta_variables_active_count": stats_two[
                "both_delta_variables_active_count"
            ],
        }
    )

    return fold_results


def run_cross_validation_for_k(
    data,
    combined,
    k_folds: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run cross-validation for one selected number of folds.

    For a total of 1,600 scenarios, the number of folds determines the
    in-sample and out-of-sample sizes. For example, 8 folds correspond to
    200 in-sample and 1,400 out-of-sample scenarios per run.

    Parameters
    ----------
    data : ScenarioData
        Scenario data containing wind production, day-ahead prices, system
        imbalance indicators, and balancing prices.

    combined : CombinedScenarioSet
        Full combined scenario set.

    k_folds : int
        Number of folds used in the cross-validation run.

    seed : int, default 42
        Random seed used for shuffling scenarios before splitting them into
        folds.

    Returns
    -------
    pd.DataFrame
        Fold-level cross-validation results for both settlement schemes.
    """

    folds = make_k_folds(combined.scenarios, k=k_folds, seed=seed)

    results = []

    print("\n" + "=" * 70)
    print(f"Running cross-validation with K = {k_folds}")
    print("=" * 70)
    print(f"Total combined scenarios: {len(combined.scenarios)}")
    print(f"In-sample scenarios per fold: {len(folds[0])}")
    print(f"Out-of-sample scenarios per fold: {len(combined.scenarios) - len(folds[0])}")

    for fold_idx in range(k_folds):
        fold_id = fold_idx + 1
        print(f"\nRunning fold {fold_id}/{k_folds}")

        in_sample_scenarios = folds[fold_idx]

        out_sample_scenarios = [
            sc
            for other_fold_idx, fold in enumerate(folds)
            if other_fold_idx != fold_idx
            for sc in fold
        ]

        fold_results = evaluate_cross_validation_fold(
            data=data,
            combined=combined,
            in_sample_scenarios=in_sample_scenarios,
            out_sample_scenarios=out_sample_scenarios,
            fold_id=fold_id,
            k_folds=k_folds,
        )

        results.extend(fold_results)

    return pd.DataFrame(results)


def build_cross_validation_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact summary table from fold-level cross-validation results.

    The summary is grouped by fold count and settlement scheme. It is suitable
    for comparing the required 8-fold case with alternative in-sample sizes.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing one row per fold and settlement scheme.

    Returns
    -------
    pd.DataFrame
        Summary table with average in-sample profit, average out-of-sample
        profit, generalization gap, profit variability, and solve time.
    """

    summary_df = (
        results_df.groupby(["k_folds", "scheme"], as_index=False)
        .agg(
            n_in_sample=("n_in_sample", "first"),
            n_out_sample=("n_out_sample", "first"),
            mean_in_sample_profit_eur=(
                "in_sample_expected_profit_eur",
                "mean",
            ),
            mean_out_sample_profit_eur=(
                "out_sample_expected_profit_eur",
                "mean",
            ),
            mean_generalization_gap_eur=(
                "generalization_gap_eur",
                "mean",
            ),
            std_out_sample_profit_across_folds_eur=(
                "out_sample_expected_profit_eur",
                "std",
            ),
            mean_out_sample_std_profit_eur=(
                "out_sample_std_profit_eur",
                "mean",
            ),
            mean_solve_time_s=(
                "solve_time_s",
                "mean",
            ),
        )
    )

    return summary_df


def build_sensitivity_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact sensitivity table across different in-sample sizes.

    This table is intended for the report discussion. It compares how changing
    the number of folds, and therefore the in-sample size, affects the
    out-of-sample expected profit and generalization gap.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Cross-validation summary produced by ``build_cross_validation_summary``.

    Returns
    -------
    pd.DataFrame
        Sensitivity summary sorted by settlement scheme and in-sample size.
    """

    sensitivity_df = summary_df[
        [
            "k_folds",
            "scheme",
            "n_in_sample",
            "n_out_sample",
            "mean_in_sample_profit_eur",
            "mean_out_sample_profit_eur",
            "mean_generalization_gap_eur",
            "std_out_sample_profit_across_folds_eur",
        ]
    ].copy()

    sensitivity_df = sensitivity_df.sort_values(
        by=["scheme", "n_in_sample"],
        ascending=[True, True],
    )

    return sensitivity_df


def main():
    """
    Run the complete Task 1.3 cross-validation workflow.

    The required case is the 8-fold cross-validation with 200 in-sample and
    1,400 out-of-sample scenarios. In addition, 16-fold and 4-fold runs are
    included as a sensitivity analysis for changing the in-sample size while
    keeping the total number of scenarios fixed at 1,600.

    Returns
    -------
    None
        The function writes fold-level results, summary tables, and the
        sensitivity comparison to the ``Outputs/tables`` folder.
    """

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
        price_area="DK2",
    )

    # Required base case plus sensitivity cases.
    # K=8  -> 200 in-sample, 1400 out-of-sample
    # K=16 -> 100 in-sample, 1500 out-of-sample
    # K=4  -> 400 in-sample, 1200 out-of-sample
    fold_counts = [8, 16, 4]

    all_results = []

    for k_folds in fold_counts:
        results_k = run_cross_validation_for_k(
            data=data,
            combined=combined,
            k_folds=k_folds,
            seed=42,
        )

        results_k.to_csv(
            f"Outputs/tables/task_1_3_cross_validation_results_k{k_folds}.csv",
            index=False,
        )

        all_results.append(results_k)

    results_df = pd.concat(all_results, ignore_index=True)

    summary_df = build_cross_validation_summary(results_df)
    sensitivity_df = build_sensitivity_summary(summary_df)

    # Save complete combined results and compact summaries.
    results_df.to_csv(
        "Outputs/tables/task_1_3_cross_validation_results_all_k.csv",
        index=False,
    )
    summary_df.to_csv(
        "Outputs/tables/task_1_3_cross_validation_summary_all_k.csv",
        index=False,
    )
    sensitivity_df.to_csv(
        "Outputs/tables/task_1_3_in_sample_size_sensitivity.csv",
        index=False,
    )

    # For convenience, also save the required K=8 case separately.
    required_results_df = results_df[results_df["k_folds"] == 8].copy()
    required_summary_df = summary_df[summary_df["k_folds"] == 8].copy()

    required_results_df.to_csv(
        "Outputs/tables/task_1_3_cross_validation_results_required_k8.csv",
        index=False,
    )
    required_summary_df.to_csv(
        "Outputs/tables/task_1_3_cross_validation_summary_required_k8.csv",
        index=False,
    )

    print("\nRequired 8-fold cross-validation summary:")
    print(required_summary_df)

    print("\nIn-sample size sensitivity summary:")
    print(sensitivity_df)

    print("\nFiles saved:")
    print(" - Outputs/tables/task_1_3_cross_validation_results_required_k8.csv")
    print(" - Outputs/tables/task_1_3_cross_validation_summary_required_k8.csv")
    print(" - Outputs/tables/task_1_3_cross_validation_results_all_k.csv")
    print(" - Outputs/tables/task_1_3_cross_validation_summary_all_k.csv")
    print(" - Outputs/tables/task_1_3_in_sample_size_sensitivity.csv")
    for k_folds in fold_counts:
        print(f" - Outputs/tables/task_1_3_cross_validation_results_k{k_folds}.csv")


if __name__ == "__main__":
    main()