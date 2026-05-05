# task_1_1.py

from __future__ import annotations

from matplotlib.pyplot import plot
import numpy as np
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
    plot_profit_distribution,
    plot_profit_by_scenario,
)


def solve_one_price_offering_problem(data, combined):
    """
    Task 1.1:
    Offering strategy under one-price balancing scheme.

    Objective:
        maximize expected profit

    Profit in each scenario and hour:
        DA_t * offer_t + BP_t * (wind_t - offer_t)
    """

    model = gp.Model("task_1_1_one_price")
    model.setParam("OutputFlag", 1)

    # Decision variables: hourly day-ahead offer
    offer = model.addVars(HOURS, lb=0.0, ub=CAPACITY_MW, name="offer")

    obj = gp.LinExpr()

    for (w_s, p_s, i_s) in combined.scenarios:
        prob = combined.probability[(w_s, p_s, i_s)]

        wind = data.wind[w_s]
        da = data.price[p_s]
        bp = data.balancing_price[(p_s, i_s)]

        for t in HOURS:
            obj += prob * (
                da[t] * offer[t] +
                bp[t] * (wind[t] - offer[t])
            )

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError("Optimization did not find an optimal solution.")

    offer_solution = np.array([offer[t].X for t in HOURS])
    expected_profit = model.ObjVal

    return offer_solution, expected_profit


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

    offer, expected_profit = solve_one_price_offering_problem(data, combined)

    print("\nOptimal day-ahead offers [MW]:")
    for t in HOURS:
        print(f"Hour {t:02d}: {offer[t]:.2f}")

    print(f"\nExpected profit: {expected_profit:.2f} EUR")

    profits = evaluate_one_price_across_scenarios(offer, data, combined)

    save_offer_to_csv(offer, "outputs/tables/task_1_1_offer.csv")
    save_wind_scenarios_to_csv(data.wind, "data/processed/wind_scenarios_used.csv")
    
    plot_hourly_offer(
    offer=offer,
    filename="outputs/figures/task_1_1_hourly_offer.png",
    title="Task 1.1 Optimal Hourly Offer - One-Price Scheme"
    )

    plot_profit_distribution(
    profits=profits,
    filename="outputs/figures/task_1_1_profit_distribution.png",
    title="Task 1.1 Profit Distribution - One-Price Scheme"
    )

    plot_profit_by_scenario(
    profits=profits,
    filename="outputs/figures/task_1_1_profit_by_scenario.png",
    title="Task 1.1 Profit Across Scenarios - One-Price Scheme"
    )

    print("\nFiles saved:")
    print(" - outputs/tables/task_1_1_offer.csv")
    print(" - outputs/figures/task_1_1_profit_distribution.png")
    print(" - data/processed/price_hourly_daily.csv")
    print(" - data/processed/wind_scenarios_used.csv")


if __name__ == "__main__":
    main()