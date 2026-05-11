# task_1_2.py

from __future__ import annotations

import time

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


def solve_two_price_offering_problem(data, combined):
    """
    Solve the stochastic offering problem under a two-price balancing scheme.

    The wind farm submits one day-ahead quantity offer for each hour. This
    offer is a first-stage decision and is therefore not scenario-indexed.
    Positive and negative imbalances are represented by two non-negative
    auxiliary variables, following the linear formulation used in the lecture
    slides.

    Under the two-price settlement scheme, desired imbalances are settled at
    the day-ahead price, while undesired imbalances are settled at the
    balancing price.

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
        Optimized expected profit under the two-price scheme [EUR].

    model_stats : dict
        Dictionary containing model size, solve time, objective value, and a
        diagnostic count for simultaneous activation of positive and negative
        imbalance variables.
    """

    model = gp.Model("task_1_2_two_price_linear")
    model.setParam("OutputFlag", 1)

    # First-stage decision:
    # one hourly day-ahead offer, independent of the realized scenario.
    offer = model.addVars(
        HOURS,
        lb=0.0,
        ub=CAPACITY_MW,
        name="offer",
    )

    # Scenario-dependent imbalance variables.
    # delta_up = positive wind imbalance / generation excess
    # delta_down = negative wind imbalance / generation deficit
    delta_up = {}
    delta_down = {}

    for sc in combined.scenarios:
        for t in HOURS:
            delta_up[(sc, t)] = model.addVar(
                lb=0.0,
                ub=CAPACITY_MW,
                name=f"delta_up_{sc}_{t}",
            )
            delta_down[(sc, t)] = model.addVar(
                lb=0.0,
                ub=CAPACITY_MW,
                name=f"delta_down_{sc}_{t}",
            )

    model.update()

    # Imbalance definition:
    # realized wind - day-ahead offer = positive imbalance - negative imbalance
    for sc in combined.scenarios:
        w_s, p_s, i_s = sc
        wind = data.wind[w_s]

        for t in HOURS:
            model.addConstr(
                delta_up[(sc, t)] - delta_down[(sc, t)]
                == wind[t] - offer[t],
                name=f"deviation_balance_{w_s}_{p_s}_{i_s}_{t}",
            )

    expected_profit_expr = gp.LinExpr()

    for sc in combined.scenarios:
        w_s, p_s, i_s = sc
        probability = combined.probability[sc]

        da_price = data.price[p_s]
        balancing_price = data.balancing_price[(p_s, i_s)]
        system_imbalance = data.imbalance[i_s]

        for t in HOURS:
            up = delta_up[(sc, t)]
            down = delta_down[(sc, t)]
            da = da_price[t]
            bp = balancing_price[t]
            si = system_imbalance[t]

            # Day-ahead revenue
            expected_profit_expr += probability * da * offer[t]

            if si == 1:
                # System deficit:
                # positive wind imbalance helps the system -> paid at DA price
                # negative wind imbalance worsens the system -> charged at BP
                expected_profit_expr += probability * (da * up - bp * down)
            else:
                # System surplus:
                # positive wind imbalance worsens the system -> paid at BP
                # negative wind imbalance helps the system -> charged at DA price
                expected_profit_expr += probability * (bp * up - da * down)

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

    # Diagnostic: check whether both split imbalance variables are positive.
    # In the intended linear formulation, this should normally be zero.
    both_active_count = 0
    tolerance = 1e-6

    for sc in combined.scenarios:
        for t in HOURS:
            if (
                delta_up[(sc, t)].X > tolerance
                and delta_down[(sc, t)].X > tolerance
            ):
                both_active_count += 1

    model_stats = {
        "model": "Task 1.2 two-price",
        "variables": model.NumVars,
        "constraints": model.NumConstrs,
        "solve_time_s": solve_time,
        "objective_value_eur": expected_profit,
        "both_delta_variables_active_count": both_active_count,
    }

    return offer_solution, expected_profit, model_stats


def scenario_profit_two_price(
    offer: np.ndarray,
    wind_realization: np.ndarray,
    da_price: np.ndarray,
    balancing_price: np.ndarray,
    system_imbalance: np.ndarray,
) -> float:
    """
    Calculate the profit of a fixed offer in one scenario under two-price settlement.

    This function is used for ex-post profit evaluation after the optimization
    has been solved. It applies the same settlement logic as the optimization
    model, but directly evaluates the realized imbalance instead of using
    optimization variables.

    Parameters
    ----------
    offer : np.ndarray
        Fixed hourly day-ahead offer [MW].

    wind_realization : np.ndarray
        Hourly realized wind production in the scenario [MW].

    da_price : np.ndarray
        Hourly day-ahead price in the scenario [EUR/MWh].

    balancing_price : np.ndarray
        Hourly balancing price in the scenario [EUR/MWh].

    system_imbalance : np.ndarray
        Hourly system imbalance indicator. A value of 1 indicates system
        deficit, while a value of 0 indicates system surplus.

    Returns
    -------
    float
        Total 24-hour profit of the fixed offer in the scenario [EUR].
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
            # System deficit
            if delta >= 0:
                # Positive wind imbalance is desired -> DA price
                total_profit += da * delta
            else:
                # Negative wind imbalance is undesired -> balancing price
                total_profit += bp * delta
        else:
            # System surplus
            if delta >= 0:
                # Positive wind imbalance is undesired -> balancing price
                total_profit += bp * delta
            else:
                # Negative wind imbalance is desired -> DA price
                total_profit += da * delta

    return float(total_profit)


def evaluate_two_price_across_scenarios(
    offer: np.ndarray,
    data,
    combined,
) -> np.ndarray:
    """
    Evaluate a fixed day-ahead offer across all combined scenarios.

    The function loops over all combined wind, price, and imbalance scenarios
    and computes the corresponding 24-hour profit under the two-price
    settlement scheme.

    Parameters
    ----------
    offer : np.ndarray
        Fixed hourly day-ahead offer [MW].

    data : ScenarioData
        Scenario data containing wind production, day-ahead prices, balancing
        prices, and system imbalance indicators.

    combined : CombinedScenarioSet
        Combined scenario set used for evaluation.

    Returns
    -------
    np.ndarray
        Array containing one total profit value per combined scenario [EUR].
    """

    profits = []

    for sc in combined.scenarios:
        w_s, p_s, i_s = sc

        profit = scenario_profit_two_price(
            offer=offer,
            wind_realization=data.wind[w_s],
            da_price=data.price[p_s],
            balancing_price=data.balancing_price[(p_s, i_s)],
            system_imbalance=data.imbalance[i_s],
        )
        profits.append(profit)

    return np.array(profits)


def make_common_profit_bins(*profit_arrays, n_bins: int = 35) -> np.ndarray:
    """
    Create common histogram bin edges for one or more profit arrays.

    Common bins ensure that individual and comparison histograms use the same
    grouping and can therefore be compared consistently.

    Parameters
    ----------
    profit_arrays : np.ndarray
        One or more arrays containing scenario profits [EUR].

    n_bins : int, default 35
        Number of bin edges.

    Returns
    -------
    np.ndarray
        Common histogram bin edges.
    """

    all_profits = np.concatenate(
        [np.asarray(profits, dtype=float) for profits in profit_arrays]
    )

    return np.linspace(
        float(np.min(all_profits)),
        float(np.max(all_profits)),
        n_bins,
    )


def plot_profit_distribution_with_bins(
    profits: np.ndarray,
    bins: np.ndarray,
    filename: str,
    title: str,
    label: str = "Scenario profit",
    color: str = "#1f77b4",
) -> None:
    """
    Plot a single profit distribution using predefined histogram bins.
    """

    import matplotlib.pyplot as plt

    fontsize = 14

    profits = np.asarray(profits, dtype=float)
    mean_profit = float(np.mean(profits))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        profits,
        bins=bins,
        color=color,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.8,
        label=label,
    )

    ax.axvline(
        mean_profit,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Expected profit: {mean_profit:,.0f} EUR",
    )

    ax.set_xlabel("Scenario profit [EUR]", fontsize=fontsize)
    ax.set_ylabel("Frequency", fontsize=fontsize)

    # Keep plot title disabled for report-style figures.
    # ax.set_title(title, fontsize=fontsize)

    ax.tick_params(axis="both", labelsize=fontsize)
    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.yaxis.get_offset_text().set_fontsize(fontsize)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_profit_distribution_comparison(
    profits_one_price: np.ndarray,
    profits_two_price: np.ndarray,
    filename: str,
    title: str = "Profit Distribution: One-Price vs Two-Price",
) -> None:
    """
    Plot one-price and two-price scenario profit distributions in one figure.
    """

    import matplotlib.pyplot as plt

    fontsize = 14
    one_price_color = "#1f77b4"
    two_price_color = "#ff7f0e"

    profits_one_price = np.asarray(profits_one_price, dtype=float)
    profits_two_price = np.asarray(profits_two_price, dtype=float)

    mean_one = float(np.mean(profits_one_price))
    mean_two = float(np.mean(profits_two_price))

    bins = make_common_profit_bins(
        profits_one_price,
        profits_two_price,
        n_bins=35,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        profits_one_price,
        bins=bins,
        color=one_price_color,
        alpha=0.40,
        edgecolor="black",
        linewidth=0.8,
        label="One-price",
    )
    ax.hist(
        profits_two_price,
        bins=bins,
        color=two_price_color,
        alpha=0.40,
        edgecolor="black",
        linewidth=0.8,
        label="Two-price",
    )

    ax.axvline(
        mean_one,
        color=one_price_color,
        linestyle="--",
        linewidth=2,
        label=f"Mean one-price: {mean_one:,.0f} EUR",
    )
    ax.axvline(
        mean_two,
        color=two_price_color,
        linestyle=":",
        linewidth=2.5,
        label=f"Mean two-price: {mean_two:,.0f} EUR",
    )

    ax.set_xlabel("Scenario profit [EUR]", fontsize=fontsize)
    ax.set_ylabel("Frequency", fontsize=fontsize)

    # Keep plot title disabled for report-style figures.
    # ax.set_title(title, fontsize=fontsize)

    ax.tick_params(axis="both", labelsize=fontsize)
    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.yaxis.get_offset_text().set_fontsize(fontsize)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_offer_comparison(
    offer_one_price: np.ndarray,
    offer_two_price: np.ndarray,
    filename: str,
    title: str = "Comparison of Optimal Hourly Offers",
) -> None:
    import matplotlib.pyplot as plt

    fontsize = 14
    one_price_color = "#1f77b4"
    two_price_color = "#ff7f0e"

    hours = np.arange(24)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.step(
        hours,
        offer_one_price,
        where="mid",
        marker="o",
        linewidth=2,
        color=one_price_color,
        label="One-price offer",
    )
    ax.step(
        hours,
        offer_two_price,
        where="mid",
        marker="s",
        linewidth=2,
        color=two_price_color,
        label="Two-price offer",
    )
    ax.axhline(
        CAPACITY_MW,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Capacity: {CAPACITY_MW:.0f} MW",
    )

    # Keep plot title disabled for report-style figures.
    # ax.set_title(title, fontsize=fontsize)

    ax.set_xlabel("Hour", fontsize=fontsize)
    ax.set_ylabel("Day-ahead offer [MW]", fontsize=fontsize)

    ax.set_xticks(hours)
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, CAPACITY_MW * 1.10)

    ax.tick_params(axis="both", labelsize=fontsize)
    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.yaxis.get_offset_text().set_fontsize(fontsize)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_task_summary(
    expected_profit: float,
    evaluated_mean_profit: float,
    evaluated_min_profit: float,
    evaluated_max_profit: float,
    evaluated_std_profit: float,
    model_stats: dict,
    filename: str,
) -> None:
    """
    Save a compact numerical summary of Task 1.2 results.

    The summary is intended for later use in the report. It includes the
    optimizer objective value, the ex-post evaluated profit statistics, and
    the diagnostic count for simultaneous activation of positive and negative
    imbalance variables.

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

    model_stats : dict
        Dictionary containing model statistics and diagnostic values.

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
        "difference_optimizer_vs_evaluated_eur": (
            evaluated_mean_profit - expected_profit
        ),
        "min_scenario_profit_eur": evaluated_min_profit,
        "max_scenario_profit_eur": evaluated_max_profit,
        "std_scenario_profit_eur": evaluated_std_profit,
        "both_delta_variables_active_count": model_stats[
            "both_delta_variables_active_count"
        ],
    }

    pd.DataFrame([summary]).to_csv(filename, index=False)


def save_profit_comparison(
    profits_one_price: np.ndarray,
    profits_two_price: np.ndarray,
    filename: str,
) -> None:
    """
    Save a compact comparison of one-price and two-price profit statistics.

    Parameters
    ----------
    profits_one_price : np.ndarray
        Scenario profits under the one-price scheme [EUR].

    profits_two_price : np.ndarray
        Scenario profits under the two-price scheme [EUR].

    filename : str
        Path where the comparison CSV file should be saved.

    Returns
    -------
    None
        The function writes a CSV file and does not return an object.
    """

    comparison_df = pd.DataFrame(
        [
            {
                "scheme": "one-price",
                "expected_profit_eur": float(np.mean(profits_one_price)),
                "min_profit_eur": float(np.min(profits_one_price)),
                "max_profit_eur": float(np.max(profits_one_price)),
                "std_profit_eur": float(np.std(profits_one_price)),
            },
            {
                "scheme": "two-price",
                "expected_profit_eur": float(np.mean(profits_two_price)),
                "min_profit_eur": float(np.min(profits_two_price)),
                "max_profit_eur": float(np.max(profits_two_price)),
                "std_profit_eur": float(np.std(profits_two_price)),
            },
        ]
    )

    comparison_df.to_csv(filename, index=False)


def main():
    """
    Run the complete Task 1.2 workflow.

    The workflow prepares the scenario data, solves the two-price stochastic
    offering problem, evaluates the optimal offer across all scenarios, saves
    result tables, and creates the figures needed for the report.

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

    offer, expected_profit, model_stats = solve_two_price_offering_problem(
        data=data,
        combined=combined,
    )

    print("\nOptimal day-ahead offers [MW]:")
    for t in HOURS:
        print(f"Hour {t:02d}: {offer[t]:.2f}")

    print(f"\nExpected profit from optimizer: {expected_profit:.2f} EUR")

    profits_two = evaluate_two_price_across_scenarios(
        offer=offer,
        data=data,
        combined=combined,
    )

    evaluated_mean_profit = float(np.mean(profits_two))
    evaluated_min_profit = float(np.min(profits_two))
    evaluated_max_profit = float(np.max(profits_two))
    evaluated_std_profit = float(np.std(profits_two))

    print("\nProfit evaluation across all scenarios:")
    print(f"Mean evaluated profit: {evaluated_mean_profit:.2f} EUR")
    print(f"Minimum scenario profit: {evaluated_min_profit:.2f} EUR")
    print(f"Maximum scenario profit: {evaluated_max_profit:.2f} EUR")
    print(f"Standard deviation: {evaluated_std_profit:.2f} EUR")
    print(
        "Difference between evaluated mean and optimizer objective: "
        f"{evaluated_mean_profit - expected_profit:.6f} EUR"
    )

    print("\nModel statistics:")
    print(f"Variables: {model_stats['variables']}")
    print(f"Constraints: {model_stats['constraints']}")
    print(f"Solve time: {model_stats['solve_time_s']:.4f} s")
    print(
        "Cases with both positive and negative imbalance active: "
        f"{model_stats['both_delta_variables_active_count']}"
    )

    save_offer_to_csv(
        offer=offer,
        filename="outputs/tables/task_1_2_offer.csv",
    )

    pd.DataFrame([model_stats]).to_csv(
        "outputs/tables/task_1_2_model_stats.csv",
        index=False,
    )

    save_task_summary(
        expected_profit=expected_profit,
        evaluated_mean_profit=evaluated_mean_profit,
        evaluated_min_profit=evaluated_min_profit,
        evaluated_max_profit=evaluated_max_profit,
        evaluated_std_profit=evaluated_std_profit,
        model_stats=model_stats,
        filename="outputs/tables/task_1_2_summary.csv",
    )

    save_wind_scenarios_to_csv(
        wind_scenarios=data.wind,
        filename="data/processed/wind_scenarios_used.csv",
    )

    plot_hourly_offer(
        offer=offer,
        filename="outputs/figures/task_1_2_hourly_offer.png",
        title="Task 1.2 Optimal Hourly Offer - Two-Price Scheme",
    )

    plot_profit_by_scenario(
        profits=profits_two,
        filename="outputs/figures/task_1_2_profit_by_scenario.png",
        title="Task 1.2 Profit Across Scenarios - Two-Price Scheme",
    )

    one_price_offer_file = "outputs/tables/task_1_1_offer.csv"

    try:
        one_price_df = pd.read_csv(one_price_offer_file)
        offer_one = one_price_df["offer_MW"].to_numpy()

        profits_one = np.array(
            evaluate_one_price_across_scenarios(
                offer=offer_one,
                data=data,
                combined=combined,
            )
        )

        # Common bins make the standalone two-price histogram and the
        # comparison histogram consistent.
        common_bins = make_common_profit_bins(
            profits_one,
            profits_two,
            n_bins=35,
        )

        plot_profit_distribution_with_bins(
            profits=profits_two,
            bins=common_bins,
            filename="outputs/figures/task_1_2_profit_distribution.png",
            title="Task 1.2 Profit Distribution - Two-Price Scheme",
            label="Two-price scenario profit",
        )

        plot_offer_comparison(
            offer_one_price=offer_one,
            offer_two_price=offer,
            filename="outputs/figures/task_1_1_vs_1_2_offer_comparison.png",
            title="Optimal Hourly Offers: One-Price vs Two-Price",
        )

        comparison_df = pd.DataFrame(
            {
                "hour": HOURS,
                "one_price_offer_MW": offer_one,
                "two_price_offer_MW": offer,
                "difference_two_minus_one_MW": offer - offer_one,
            }
        )
        comparison_df.to_csv(
            "outputs/tables/task_1_1_vs_1_2_offer_comparison.csv",
            index=False,
        )

        plot_profit_distribution_comparison(
            profits_one_price=profits_one,
            profits_two_price=profits_two,
            filename=(
                "outputs/figures/"
                "task_1_1_vs_1_2_profit_distribution_comparison.png"
            ),
            title="Profit Distribution: One-Price vs Two-Price",
        )

        save_profit_comparison(
            profits_one_price=profits_one,
            profits_two_price=profits_two,
            filename="outputs/tables/task_1_1_vs_1_2_profit_comparison.csv",
        )

    except FileNotFoundError:
        print(
            "\nTask 1.1 offer file not found. "
            "Skipping one-price vs two-price comparison plots."
        )

        # If Task 1.1 output is missing, still produce the standalone
        # two-price histogram using bins based only on two-price profits.
        two_only_bins = make_common_profit_bins(
            profits_two,
            n_bins=35,
        )
        plot_profit_distribution_with_bins(
            profits=profits_two,
            bins=two_only_bins,
            filename="outputs/figures/task_1_2_profit_distribution.png",
            title="Task 1.2 Profit Distribution - Two-Price Scheme",
            label="Two-price scenario profit",
        )

    print("\nFiles saved:")
    print(" - outputs/tables/task_1_2_offer.csv")
    print(" - outputs/tables/task_1_2_model_stats.csv")
    print(" - outputs/tables/task_1_2_summary.csv")
    print(" - outputs/tables/task_1_1_vs_1_2_offer_comparison.csv")
    print(" - outputs/tables/task_1_1_vs_1_2_profit_comparison.csv")
    print(" - outputs/figures/task_1_2_hourly_offer.png")
    print(" - outputs/figures/task_1_2_profit_distribution.png")
    print(" - outputs/figures/task_1_2_profit_by_scenario.png")
    print(" - outputs/figures/task_1_1_vs_1_2_offer_comparison.png")
    print(" - outputs/figures/task_1_1_vs_1_2_profit_distribution_comparison.png")
    print(" - data/processed/price_hourly_daily.csv")
    print(" - data/processed/wind_scenarios_used.csv")

    if model_stats["both_delta_variables_active_count"] > 0:
        print(
            "\nWarning: Some scenario-hour combinations have both positive and "
            "negative imbalance variables active. This can happen in the linear "
            "split formulation if prices create an incentive for artificial "
            "simultaneous activation. Check the selected price scenarios."
        )


if __name__ == "__main__":
    main()