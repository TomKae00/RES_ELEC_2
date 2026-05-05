# task_1_2.py

from __future__ import annotations

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from helpers import (
    HOURS,
    CAPACITY_MW,
    ensure_output_folders,
    prepare_scenario_data,
    save_offer_to_csv,
    save_wind_scenarios_to_csv,
)

from plotting import (
    plot_hourly_offer,
    plot_profit_distribution,
    plot_profit_by_scenario,
    plot_offer_comparison
)


def solve_two_price_offering_problem(data, combined):
    """
    Linear two-price stochastic offering model.
    No binary variables needed.
    """

    model = gp.Model("task_1_2_two_price_linear")
    model.setParam("OutputFlag", 1)

    # First-stage DA offer
    offer = model.addVars(HOURS, lb=0.0, ub=CAPACITY_MW, name="offer")

    # Scenario-dependent deviations
    delta_up = {}
    delta_down = {}

    for sc in combined.scenarios:
        for t in HOURS:
            delta_up[(sc, t)] = model.addVar(lb=0.0, ub=CAPACITY_MW, name=f"delta_up_{sc}_{t}")
            delta_down[(sc, t)] = model.addVar(lb=0.0, ub=CAPACITY_MW, name=f"delta_down_{sc}_{t}")

    model.update()

    # delta = wind - offer = delta_up - delta_down
    for (w_s, p_s, i_s) in combined.scenarios:
        wind = data.wind[w_s]
        sc = (w_s, p_s, i_s)

        for t in HOURS:
            model.addConstr(
                delta_up[(sc, t)] - delta_down[(sc, t)] == wind[t] - offer[t],
                name=f"deviation_balance_{w_s}_{p_s}_{i_s}_{t}"
            )

    obj = gp.LinExpr()

    for (w_s, p_s, i_s) in combined.scenarios:
        prob = combined.probability[(w_s, p_s, i_s)]
        da = data.price[p_s]
        si = data.imbalance[i_s]
        sc = (w_s, p_s, i_s)

        for t in HOURS:
            up = delta_up[(sc, t)]
            down = delta_down[(sc, t)]

            # DA revenue
            obj += prob * da[t] * offer[t]

            if si[t] == 1:
                # System deficit:
                # upward deviation desired -> DA price
                # downward deviation undesired -> 1.25 * DA
                obj += prob * (da[t] * up - 1.25 * da[t] * down)

            else:
                # System surplus:
                # upward deviation undesired -> 0.85 * DA
                # downward deviation desired -> DA price
                obj += prob * (0.85 * da[t] * up - da[t] * down)

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Optimization ended with status {model.Status}")

    offer_solution = np.array([offer[t].X for t in HOURS])
    expected_profit = model.ObjVal

    return offer_solution, expected_profit


def scenario_profit_two_price(
    offer: np.ndarray,
    wind_realization: np.ndarray,
    da_price: np.ndarray,
    balancing_price: np.ndarray,
    system_imbalance: np.ndarray
) -> float:
    total_profit = 0.0

    for t in HOURS:
        delta = wind_realization[t] - offer[t]
        da = da_price[t]
        bp = balancing_price[t]
        si = system_imbalance[t]

        total_profit += da * offer[t]

        if si == 1:
            # deficit
            if delta >= 0:
                total_profit += da * delta
            else:
                total_profit += bp * delta
        else:
            # surplus
            if delta <= 0:
                total_profit += da * delta
            else:
                total_profit += bp * delta

    return float(total_profit)


def evaluate_two_price_across_scenarios(offer, data, combined):
    profits = []

    for (w_s, p_s, i_s) in combined.scenarios:
        profit = scenario_profit_two_price(
            offer=offer,
            wind_realization=data.wind[w_s],
            da_price=data.price[p_s],
            balancing_price=data.balancing_price[(p_s, i_s)],
            system_imbalance=data.imbalance[i_s]
        )
        profits.append(profit)

    return profits


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

    print(f"Wind scenarios used: {len(data.wind)}")
    print(f"Price scenarios used: {len(data.price)}")
    print(f"Imbalance scenarios used: {len(data.imbalance)}")
    print(f"Total combined scenarios: {len(combined.scenarios)}")

    offer, expected_profit = solve_two_price_offering_problem(data, combined)

    print("\nOptimal day-ahead offers [MW]:")
    for t in HOURS:
        print(f"Hour {t:02d}: {offer[t]:.2f}")

    print(f"\nExpected profit: {expected_profit:.2f} EUR")

    profits = evaluate_two_price_across_scenarios(offer, data, combined)

    save_offer_to_csv(offer, "outputs/tables/task_1_2_offer.csv")
    save_wind_scenarios_to_csv(data.wind, "data/processed/wind_scenarios_used.csv")
    
    plot_hourly_offer(
    offer=offer,
    filename="outputs/figures/task_1_2_hourly_offer.png",
    title="Task 1.2 Optimal Hourly Offer - Two-Price Scheme"
    )

    plot_profit_distribution(
    profits=profits,
    filename="outputs/figures/task_1_2_profit_distribution.png",
    title="Task 1.2 Profit Distribution - Two-Price Scheme"
    )

    plot_profit_by_scenario(
    profits=profits,
    filename="outputs/figures/task_1_2_profit_by_scenario.png",
    title="Task 1.2 Profit Across Scenarios - Two-Price Scheme"
    )

    one_price_df = pd.read_csv("outputs/tables/task_1_1_offer.csv")
    offer_one = one_price_df["offer_MW"].to_numpy()
    plot_offer_comparison(
        offer_one_price=offer_one,
        offer_two_price=offer,   # current solution
        filename="outputs/figures/task_1_1_vs_1_2_offer_comparison.png",
        title="Optimal Hourly Offers: One-Price vs Two-Price"
    )


    print("\nFiles saved:")
    print(" - outputs/tables/task_1_2_offer.csv")
    print(" - outputs/figures/task_1_2_profit_distribution.png")


if __name__ == "__main__":
    main()