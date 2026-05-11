# task_1_1.py

from __future__ import annotations

import time

from typing import List

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from helpers import (
    HOURS,
    CAPACITY_MW,
    ensure_output_folders,
    prepare_scenario_data,
    evaluate_one_price_across_scenarios,
    save_offer_to_csv,
    save_wind_scenarios_to_csv,
)

from plotting import (
    plot_hourly_offer,
    plot_profit_by_scenario,
)


def solve_one_price_offering_problem(data, combined):
    """
    Solve the stochastic offering problem under a one-price balancing scheme.

    The wind farm is modeled as a price-taking producer. It submits one
    day-ahead quantity offer for each hour of the 24-hour horizon. Since this
    offer is made before the realization of wind production, day-ahead prices,
    and system imbalance is known, the offer is a first-stage decision and is
    not scenario-indexed.

    Under the one-price settlement scheme, all imbalances are settled at the
    balancing price. Therefore, positive and negative imbalances do not need to
    be represented by separate variables in this model.

    The scenario profit is given by:

        sum_t [ DA_price_t * offer_t
                + balancing_price_t * (wind_realization_t - offer_t) ]

    Parameters
    ----------
    data : ScenarioData
        Scenario data containing wind production, day-ahead prices, system
        imbalance indicators, and balancing prices.

    combined : CombinedScenarioSet
        Combined scenario set containing tuples of wind, price, and imbalance
        scenario IDs, together with their probabilities.

    Returns
    -------
    offer_solution : np.ndarray
        Optimal hourly day-ahead offer of the wind farm [MW].

    expected_profit : float
        Optimized expected profit under the one-price scheme [EUR].

    model_stats : dict
        Dictionary containing the model name, number of variables, number of
        constraints, solve time, and objective value.
    """

    model = gp.Model("task_1_1_one_price")
    model.setParam("OutputFlag", 1)

    # First-stage decision variables:
    # one hourly day-ahead offer, independent of the realized scenario.
    offer = model.addVars(
        HOURS,
        lb=0.0,
        ub=CAPACITY_MW,
        name="offer",
    )

    expected_profit_expr = gp.LinExpr()

    for scenario in combined.scenarios:
        w_s, p_s, i_s = scenario
        probability = combined.probability[scenario]

        wind = data.wind[w_s]
        da_price = data.price[p_s]
        balancing_price = data.balancing_price[(p_s, i_s)]

        for t in HOURS:
            imbalance = wind[t] - offer[t]

            expected_profit_expr += probability * (
                da_price[t] * offer[t]
                + balancing_price[t] * imbalance
            )

    model.setObjective(expected_profit_expr, GRB.MAXIMIZE)

    start_time = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start_time

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Optimization did not find an optimal solution. "
            f"Gurobi status code: {model.Status}"
        )

    offer_solution = np.array([offer[t].X for t in HOURS])
    expected_profit = float(model.ObjVal)

    model_stats = {
        "model": "Task 1.1 one-price",
        "variables": model.NumVars,
        "constraints": model.NumConstrs,
        "solve_time_s": solve_time,
        "objective_value_eur": expected_profit,
    }

    return offer_solution, expected_profit, model_stats


def build_all_or_nothing_summary(offer: np.ndarray) -> dict:
    """
    Count the number of hours with zero, full-capacity, and interior offers.

    This diagnostic is useful for Task 1.1 because the one-price settlement
    scheme can lead to all-or-nothing bidding behavior. If the offer is close
    to either 0 MW or installed capacity in most hours, this indicates that the
    wind farm is exploiting the expected spread between day-ahead and balancing
    prices rather than offering a forecast-like production quantity.

    Parameters
    ----------
    offer : np.ndarray
        Optimal hourly day-ahead offer [MW].

    Returns
    -------
    dict
        Dictionary containing the number of hours with a zero offer, a
        full-capacity offer, and an interior offer.
    """

    zero_hours = int(np.sum(np.isclose(offer, 0.0)))
    full_hours = int(np.sum(np.isclose(offer, CAPACITY_MW)))
    interior_hours = int(len(offer) - zero_hours - full_hours)

    return {
        "zero_offer_hours": zero_hours,
        "full_capacity_offer_hours": full_hours,
        "interior_offer_hours": interior_hours,
    }


def calculate_expected_price_spread(data, combined) -> np.ndarray:
    """
    Calculate the hourly expected price spread between day-ahead and balancing prices.

    For the one-price formulation, the offer-dependent part of the objective is
    proportional to the expected spread:

        E[lambda_DA - lambda_B]

    If this value is positive in an hour, offering the upper bound is optimal.
    If it is negative, offering the lower bound is optimal.

    Parameters
    ----------
    data : ScenarioData
        Scenario data containing day-ahead prices and balancing prices.

    combined : CombinedScenarioSet
        Combined scenario set containing scenario tuples and probabilities.

    Returns
    -------
    np.ndarray
        Hourly expected spread E[lambda_DA - lambda_B] [EUR/MWh].
    """

    expected_spread = np.zeros(len(HOURS))

    for scenario in combined.scenarios:
        _, p_s, i_s = scenario
        probability = combined.probability[scenario]

        da_price = data.price[p_s]
        balancing_price = data.balancing_price[(p_s, i_s)]

        for t in HOURS:
            expected_spread[t] += probability * (
                da_price[t] - balancing_price[t]
            )

    return expected_spread


def plot_expected_price_spread_with_offer(
    expected_spread: np.ndarray,
    offer: np.ndarray,
    filename: str,
    title: str | None = None,
) -> None:
    """
    Plot the expected day-ahead minus balancing price spread and the optimal offer.
    """

    import matplotlib.pyplot as plt

    fontsize = 14
    one_price_color = "#1f77b4"
    spread_color = "lightgrey"

    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
    })

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(
        HOURS,
        expected_spread,
        width=0.75,
        alpha=0.8,
        color=spread_color,
        edgecolor="black",
        linewidth=0.6,
        label=r"$\mathbb{E}[\lambda^{\mathrm{DA}}-\lambda^{\mathrm{B}}]$",
    )

    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.5)

    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Expected price spread [EUR/MWh]")
    ax1.set_xticks(HOURS)
    ax1.set_xlim(-0.5, 23.5)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.step(
        HOURS,
        offer,
        where="mid",
        linewidth=2.2,
        color=one_price_color,
        label="One-price offer",
    )
    ax2.scatter(
        HOURS,
        offer,
        s=25,
        color=one_price_color,
    )

    ax2.set_ylabel("Day-ahead offer [MW]")
    ax2.set_ylim(-0.05 * CAPACITY_MW, 1.10 * CAPACITY_MW)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_profit_distribution(
    profits: List[float],
    filename: str,
    title: str | None = None,
) -> None:
    """
    Plot the distribution of scenario profits and the expected profit.
    """

    import matplotlib.pyplot as plt

    fontsize = 14
    one_price_color = "#1f77b4"
    expected_profit = np.mean(profits)

    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        profits,
        bins=35,
        color=one_price_color,
        edgecolor="black",
        alpha=0.4,
    )

    ax.axvline(
        expected_profit,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Expected profit: {expected_profit:,.0f} EUR",
    )

    ax.set_xlabel("Scenario profit [EUR]")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_task_summary(
    expected_profit: float,
    evaluated_mean_profit: float,
    evaluated_min_profit: float,
    evaluated_max_profit: float,
    evaluated_std_profit: float,
    all_or_nothing_summary: dict,
    filename: str,
) -> None:
    """
    Save a compact numerical summary of Task 1.1 results.

    The summary is intended for later use in the report. It includes the
    optimizer objective value, ex-post evaluated profit statistics, and the
    all-or-nothing diagnostic for the optimal day-ahead offer.

    Parameters
    ----------
    expected_profit : float
        Optimized expected profit returned by Gurobi [EUR].

    evaluated_mean_profit : float
        Mean of the ex-post scenario profit distribution [EUR].

    evaluated_min_profit : float
        Minimum scenario profit [EUR].

    evaluated_max_profit : float
        Maximum scenario profit [EUR].

    evaluated_std_profit : float
        Standard deviation of scenario profits [EUR].

    all_or_nothing_summary : dict
        Dictionary returned by ``build_all_or_nothing_summary`` containing the
        number of zero-offer, full-capacity-offer, and interior-offer hours.

    filename : str
        Path where the summary CSV file should be saved.

    Returns
    -------
    None
        The function writes a CSV file and does not return an object.
    """

    summary = {
        "expected_profit_optimizer_eur": expected_profit,
        "mean_evaluated_profit_eur": evaluated_mean_profit,
        "difference_optimizer_vs_evaluated_eur": evaluated_mean_profit - expected_profit,
        "min_scenario_profit_eur": evaluated_min_profit,
        "max_scenario_profit_eur": evaluated_max_profit,
        "std_scenario_profit_eur": evaluated_std_profit,
        **all_or_nothing_summary,
    }

    pd.DataFrame([summary]).to_csv(filename, index=False)


def main():
    """
    Run the complete Task 1.1 workflow.

    The workflow prepares the scenario data, solves the one-price stochastic
    offering problem, evaluates the optimal offer across all combined
    scenarios, saves result tables, and creates the figures needed for the
    report.

    The generated outputs include:
    - optimal hourly offer,
    - model statistics,
    - compact summary table,
    - profit distribution plot,
    - profit-by-scenario plot,
    - hourly offer plot.

    Returns
    -------
    None
        The function writes outputs to the ``outputs`` and ``data/processed``
        folders and prints a compact summary to the terminal.
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

    print("Scenario setup:")
    print(f"Wind scenarios used: {len(data.wind)}")
    print(f"Price scenarios used: {len(data.price)}")
    print(f"Imbalance scenarios used: {len(data.imbalance)}")
    print(f"Total combined scenarios: {len(combined.scenarios)}")

    offer, expected_profit, model_stats = solve_one_price_offering_problem(
        data=data,
        combined=combined,
    )

    print("\nOptimal day-ahead offers [MW]:")
    for t in HOURS:
        print(f"Hour {t:02d}: {offer[t]:.2f}")

    print(f"\nExpected profit from optimizer: {expected_profit:.2f} EUR")

    profits = np.array(
        evaluate_one_price_across_scenarios(
            offer=offer,
            data=data,
            combined=combined,
        )
    )

    evaluated_mean_profit = float(np.mean(profits))
    evaluated_min_profit = float(np.min(profits))
    evaluated_max_profit = float(np.max(profits))
    evaluated_std_profit = float(np.std(profits))

    print("\nProfit evaluation across all scenarios:")
    print(f"Mean evaluated profit: {evaluated_mean_profit:.2f} EUR")
    print(f"Minimum scenario profit: {evaluated_min_profit:.2f} EUR")
    print(f"Maximum scenario profit: {evaluated_max_profit:.2f} EUR")
    print(f"Standard deviation: {evaluated_std_profit:.2f} EUR")
    print(
        "Difference between evaluated mean and optimizer objective: "
        f"{evaluated_mean_profit - expected_profit:.6f} EUR"
    )

    all_or_nothing_summary = build_all_or_nothing_summary(offer)

    expected_spread = calculate_expected_price_spread(
        data=data,
        combined=combined,
    )

    spread_df = pd.DataFrame(
        {
            "hour": HOURS,
            "expected_da_minus_balancing_price_eur_per_mwh": expected_spread,
            "optimal_offer_MW": offer,
        }
    )
    spread_df.to_csv(
        "outputs/tables/task_1_1_expected_price_spread.csv",
        index=False,
    )

    print("\nExpected DA minus balancing price spread:")
    for t in HOURS:
        print(
            f"Hour {t:02d}: "
            f"{expected_spread[t]:.2f} EUR/MWh, "
            f"offer = {offer[t]:.2f} MW"
        )

    print("\nAll-or-nothing diagnostic:")
    print(f"Hours with 0 MW offer: {all_or_nothing_summary['zero_offer_hours']}")
    print(f"Hours with 500 MW offer: {all_or_nothing_summary['full_capacity_offer_hours']}")
    print(f"Hours with interior offer: {all_or_nothing_summary['interior_offer_hours']}")

    print("\nModel statistics:")
    print(f"Variables: {model_stats['variables']}")
    print(f"Constraints: {model_stats['constraints']}")
    print(f"Solve time: {model_stats['solve_time_s']:.4f} s")

    save_offer_to_csv(
        offer=offer,
        filename="outputs/tables/task_1_1_offer.csv",
    )

    pd.DataFrame([model_stats]).to_csv(
        "outputs/tables/task_1_1_model_stats.csv",
        index=False,
    )

    save_task_summary(
        expected_profit=expected_profit,
        evaluated_mean_profit=evaluated_mean_profit,
        evaluated_min_profit=evaluated_min_profit,
        evaluated_max_profit=evaluated_max_profit,
        evaluated_std_profit=evaluated_std_profit,
        all_or_nothing_summary=all_or_nothing_summary,
        filename="outputs/tables/task_1_1_summary.csv",
    )

    save_wind_scenarios_to_csv(
        wind_scenarios=data.wind,
        filename="data/processed/wind_scenarios_used.csv",
    )

    plot_hourly_offer(
        offer=offer,
        filename="outputs/figures/task_1_1_hourly_offer.png",
        title="Task 1.1 Optimal Hourly Offer - One-Price Scheme",
    )

    plot_profit_distribution(
        profits=profits,
        filename="outputs/figures/task_1_1_profit_distribution.png",
        title="Task 1.1 Profit Distribution - One-Price Scheme",
    )

    plot_profit_by_scenario(
        profits=profits,
        filename="outputs/figures/task_1_1_profit_by_scenario.png",
        title="Task 1.1 Profit Across Scenarios - One-Price Scheme",
    )

    plot_expected_price_spread_with_offer(
        expected_spread=expected_spread,
        offer=offer,
        filename="outputs/figures/task_1_1_expected_price_spread_with_offer.png",
        title="Task 1.1 Expected Price Spread and Optimal Offer",
    )

    print("\nFiles saved:")
    print(" - outputs/tables/task_1_1_offer.csv")
    print(" - outputs/tables/task_1_1_model_stats.csv")
    print(" - outputs/tables/task_1_1_summary.csv")
    print(" - outputs/figures/task_1_1_hourly_offer.png")
    print(" - outputs/figures/task_1_1_profit_distribution.png")
    print(" - outputs/figures/task_1_1_profit_by_scenario.png")
    print(" - outputs/tables/task_1_1_expected_price_spread.csv")
    print(" - outputs/figures/task_1_1_expected_price_spread_with_offer.png")
    print(" - data/processed/price_hourly_daily.csv")
    print(" - data/processed/wind_scenarios_used.csv")


if __name__ == "__main__":
    main()