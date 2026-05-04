# task1.3.py

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import (
    HOURS,
    CAPACITY_MW,
    CombinedScenarioSet,
    ensure_output_folders,
    prepare_scenario_data,
)


# ============================================================
# Scenario subset helper
# ============================================================

def make_subset_combined(scenarios):
    """
    Creates a CombinedScenarioSet from a selected list of scenarios.
    All selected scenarios are assigned equal probability.
    """
    scenarios = list(scenarios)

    if len(scenarios) == 0:
        raise ValueError("Scenario subset is empty.")

    prob = 1.0 / len(scenarios)

    return CombinedScenarioSet(
        scenarios=scenarios,
        probability={sc: prob for sc in scenarios}
    )


# ============================================================
# Profit functions
# ============================================================

def scenario_profit_one_price(
    offer: np.ndarray,
    wind_realization: np.ndarray,
    da_price: np.ndarray,
    balancing_price: np.ndarray,
) -> float:
    """
    One-price profit:
        sum_t [ DA_t * offer_t + BP_t * (wind_t - offer_t) ]
    """
    return float(np.sum(
        da_price * offer
        + balancing_price * (wind_realization - offer)
    ))


def scenario_profit_two_price(
    offer: np.ndarray,
    wind_realization: np.ndarray,
    da_price: np.ndarray,
    balancing_price: np.ndarray,
    system_imbalance: np.ndarray,
) -> float:
    """
    Two-price profit:
    - If the deviation helps the system, it is settled at the DA price.
    - If the deviation harms the system, it is settled at the balancing price.
    """
    total_profit = 0.0

    for t in HOURS:
        delta = wind_realization[t] - offer[t]
        da = da_price[t]
        bp = balancing_price[t]
        si = system_imbalance[t]

        # Day-ahead revenue
        total_profit += da * offer[t]

        if si == 1:
            # System deficit:
            # Positive deviation helps the system.
            # Negative deviation harms the system.
            if delta >= 0:
                total_profit += da * delta
            else:
                total_profit += bp * delta
        else:
            # System surplus:
            # Negative deviation helps the system.
            # Positive deviation harms the system.
            if delta <= 0:
                total_profit += da * delta
            else:
                total_profit += bp * delta

    return float(total_profit)


def evaluate_expected_profit(
    offer: np.ndarray,
    data,
    combined: CombinedScenarioSet,
    scheme: str,
) -> float:
    """
    Evaluates expected profit of a fixed offer over a scenario set.
    """
    expected_profit = 0.0

    for (w_s, p_s, i_s) in combined.scenarios:
        prob = combined.probability[(w_s, p_s, i_s)]

        if scheme == "one_price":
            profit = scenario_profit_one_price(
                offer=offer,
                wind_realization=data.wind[w_s],
                da_price=data.price[p_s],
                balancing_price=data.balancing_price[(p_s, i_s)],
            )

        elif scheme == "two_price":
            profit = scenario_profit_two_price(
                offer=offer,
                wind_realization=data.wind[w_s],
                da_price=data.price[p_s],
                balancing_price=data.balancing_price[(p_s, i_s)],
                system_imbalance=data.imbalance[i_s],
            )

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        expected_profit += prob * profit

    return float(expected_profit)


# ============================================================
# Exact solvers by hourly enumeration
# ============================================================


def solve_one_price_offering_problem(data, combined: CombinedScenarioSet):
    """
    Solves the one-price offering problem exactly by hourly enumeration.
    """
    offer = np.zeros(len(HOURS))

    for t in HOURS:
        candidates = [0.0, CAPACITY_MW]

        best_offer = None
        best_value = -np.inf

        for x in candidates:
            expected_hourly_profit = 0.0

            for (w_s, p_s, i_s) in combined.scenarios:
                prob = combined.probability[(w_s, p_s, i_s)]

                wind = data.wind[w_s][t]
                da = data.price[p_s][t]
                bp = data.balancing_price[(p_s, i_s)][t]

                profit = da * x + bp * (wind - x)
                expected_hourly_profit += prob * profit

            if expected_hourly_profit > best_value + 1e-9:
                best_value = expected_hourly_profit
                best_offer = x

        offer[t] = best_offer

    expected_profit = evaluate_expected_profit(
        offer=offer,
        data=data,
        combined=combined,
        scheme="one_price",
    )

    return offer, expected_profit


def solve_two_price_offering_problem(data, combined: CombinedScenarioSet):
    """
    Solves the two-price offering problem exactly by hourly enumeration.
    """
    offer = np.zeros(len(HOURS))

    for t in HOURS:
        candidates = {0.0, CAPACITY_MW}

        # Breakpoints occur at wind production values in the in-sample scenarios.
        for (w_s, _, _) in combined.scenarios:
            candidates.add(float(data.wind[w_s][t]))

        best_offer = None
        best_value = -np.inf

        for x in sorted(candidates):
            expected_hourly_profit = 0.0

            for (w_s, p_s, i_s) in combined.scenarios:
                prob = combined.probability[(w_s, p_s, i_s)]

                wind = data.wind[w_s][t]
                da = data.price[p_s][t]
                bp = data.balancing_price[(p_s, i_s)][t]
                si = data.imbalance[i_s][t]

                delta = wind - x

                profit = da * x

                if si == 1:
                    # System deficit:
                    # Positive deviation helps, negative deviation harms.
                    if delta >= 0:
                        profit += da * delta
                    else:
                        profit += bp * delta
                else:
                    # System surplus:
                    # Negative deviation helps, positive deviation harms.
                    if delta <= 0:
                        profit += da * delta
                    else:
                        profit += bp * delta

                expected_hourly_profit += prob * profit

            if expected_hourly_profit > best_value + 1e-9:
                best_value = expected_hourly_profit
                best_offer = x

        offer[t] = best_offer

    expected_profit = evaluate_expected_profit(
        offer=offer,
        data=data,
        combined=combined,
        scheme="two_price",
    )

    return offer, expected_profit


# ============================================================
# Cross-validation
# ============================================================

def run_cross_validation(
    data,
    combined: CombinedScenarioSet,
    n_folds: int = 8,
    seed: int = 42,
):
    """
    Performs 8-fold cross-validation.

    Total scenarios: 1600
    Each fold:       200 scenarios
    In each run:
        in-sample:      200 scenarios
        out-of-sample: 1400 scenarios
    """
    all_scenarios = list(combined.scenarios)

    if len(all_scenarios) != 1600:
        raise ValueError(
            f"Task 1.3 should use exactly 1600 scenarios, but got {len(all_scenarios)}."
        )

    if len(all_scenarios) % n_folds != 0:
        raise ValueError("Number of scenarios must be divisible by number of folds.")

    rng = np.random.default_rng(seed)
    rng.shuffle(all_scenarios)

    fold_size = len(all_scenarios) // n_folds
    folds = [
        all_scenarios[i * fold_size:(i + 1) * fold_size]
        for i in range(n_folds)
    ]

    results = []
    offers = []

    for run in range(n_folds):
        print(f"\n========== Cross-validation run {run + 1}/{n_folds} ==========")

        in_sample_scenarios = folds[run]

        out_of_sample_scenarios = []
        for j in range(n_folds):
            if j != run:
                out_of_sample_scenarios.extend(folds[j])

        in_sample = make_subset_combined(in_sample_scenarios)
        out_of_sample = make_subset_combined(out_of_sample_scenarios)

        print(f"In-sample scenarios: {len(in_sample.scenarios)}")
        print(f"Out-of-sample scenarios: {len(out_of_sample.scenarios)}")

        for scheme in ["one_price", "two_price"]:
            print(f"\nSolving scheme: {scheme}")

            if scheme == "one_price":
                offer, in_sample_expected_profit = solve_one_price_offering_problem(
                    data=data,
                    combined=in_sample,
                )
            else:
                offer, in_sample_expected_profit = solve_two_price_offering_problem(
                    data=data,
                    combined=in_sample,
                )

            # Re-evaluate explicitly on both sets.
            in_eval = evaluate_expected_profit(
                offer=offer,
                data=data,
                combined=in_sample,
                scheme=scheme,
            )

            out_eval = evaluate_expected_profit(
                offer=offer,
                data=data,
                combined=out_of_sample,
                scheme=scheme,
            )

            gap = in_eval - out_eval
            gap_percent = 100.0 * gap / abs(in_eval) if abs(in_eval) > 1e-9 else np.nan

            print(f"In-sample expected profit:      {in_eval:.2f} EUR")
            print(f"Out-of-sample expected profit: {out_eval:.2f} EUR")
            print(f"Generalization gap:            {gap:.2f} EUR")

            results.append({
                "run": run + 1,
                "scheme": scheme,
                "n_in_sample": len(in_sample.scenarios),
                "n_out_of_sample": len(out_of_sample.scenarios),
                "in_sample_expected_profit_EUR": in_eval,
                "out_of_sample_expected_profit_EUR": out_eval,
                "generalization_gap_EUR": gap,
                "generalization_gap_percent": gap_percent,
            })

            for t in HOURS:
                offers.append({
                    "run": run + 1,
                    "scheme": scheme,
                    "hour": t,
                    "offer_MW": offer[t],
                })

    results_df = pd.DataFrame(results)
    offers_df = pd.DataFrame(offers)

    summary_df = (
        results_df
        .groupby("scheme", as_index=False)
        .agg(
            avg_in_sample_profit_EUR=("in_sample_expected_profit_EUR", "mean"),
            avg_out_of_sample_profit_EUR=("out_of_sample_expected_profit_EUR", "mean"),
            std_in_sample_profit_EUR=("in_sample_expected_profit_EUR", "std"),
            std_out_of_sample_profit_EUR=("out_of_sample_expected_profit_EUR", "std"),
            avg_generalization_gap_EUR=("generalization_gap_EUR", "mean"),
            avg_generalization_gap_percent=("generalization_gap_percent", "mean"),
        )
    )

    return results_df, summary_df, offers_df


# ============================================================
# Plotting
# ============================================================

def plot_cross_validation_results(results_df: pd.DataFrame, filename: str):
    plt.figure(figsize=(9, 5))

    for scheme in ["one_price", "two_price"]:
        subset = results_df[results_df["scheme"] == scheme].copy()

        plt.plot(
            subset["run"],
            subset["in_sample_expected_profit_EUR"],
            marker="o",
            label=f"{scheme} in-sample",
        )

        plt.plot(
            subset["run"],
            subset["out_of_sample_expected_profit_EUR"],
            marker="s",
            linestyle="--",
            label=f"{scheme} out-of-sample",
        )

    plt.xlabel("Cross-validation run")
    plt.ylabel("Expected profit [EUR]")
    plt.title("Task 1.3 Cross-Validation: In-Sample vs Out-of-Sample Profit")
    plt.xticks(results_df["run"].unique())
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    ensure_output_folders()

    wind_file = base_dir / "Data" / "scen_zone2.csv"
    price_file = base_dir / "Data" / "DayAheadPrices.csv"

    # Task 1.3 requires exactly 1600 scenarios:
    # 20 wind × 20 price × 4 imbalance = 1600.
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

    print(f"Wind scenarios used: {len(data.wind)}")
    print(f"Price scenarios used: {len(data.price)}")
    print(f"Imbalance scenarios used: {len(data.imbalance)}")
    print(f"Total combined scenarios: {len(combined.scenarios)}")

    results_df, summary_df, offers_df = run_cross_validation(
        data=data,
        combined=combined,
        n_folds=8,
        seed=42,
    )

    results_path = "outputs/tables/task_1_3_cross_validation_results.csv"
    summary_path = "outputs/tables/task_1_3_cross_validation_summary.csv"
    offers_path = "outputs/tables/task_1_3_cross_validation_offers.csv"
    figure_path = "outputs/figures/task_1_3_cross_validation_profit_comparison.png"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    offers_df.to_csv(offers_path, index=False)

    plot_cross_validation_results(results_df, figure_path)

    print("\n========== Task 1.3 Summary ==========")
    print(summary_df.to_string(index=False))

    print("\nFiles saved:")
    print(f" - {results_path}")
    print(f" - {summary_path}")
    print(f" - {offers_path}")
    print(f" - {figure_path}")


if __name__ == "__main__":
    main()
