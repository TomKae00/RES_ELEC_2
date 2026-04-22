# task_1_2.py

from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from helpers import (
    HOURS,
    CAPACITY_MW,
    ensure_output_folders,
    prepare_scenario_data,
    save_offer_to_csv,
    plot_profit_distribution,
    save_wind_scenarios_to_csv,
)


def solve_two_price_offering_problem(data, combined):
    """
    Task 1.2:
    Offering strategy under the two-price balancing scheme.

    Two-price settlement:
    - beneficial deviation -> settles at DA price
    - harmful deviation    -> settles at balancing price
    """

    model = gp.Model("task_1_2_two_price")
    model.setParam("OutputFlag", 1)
    model.setParam("DualReductions", 0)  # helps distinguish infeasible/unbounded

    # First-stage decision variables
    offer = model.addVars(HOURS, lb=0.0, ub=CAPACITY_MW, name="offer")

    # Second-stage variables
    delta_pos = {}
    delta_neg = {}
    z = {}   # binary to enforce only one deviation side is active

    M = CAPACITY_MW  # because wind and offer are both in [0,500], deviation is in [-500,500]

    for sc in combined.scenarios:
        for t in HOURS:
            delta_pos[(sc, t)] = model.addVar(lb=0.0, ub=M, name=f"delta_pos_{sc}_{t}")
            delta_neg[(sc, t)] = model.addVar(lb=0.0, ub=M, name=f"delta_neg_{sc}_{t}")
            z[(sc, t)] = model.addVar(vtype=GRB.BINARY, name=f"z_{sc}_{t}")

    model.update()

    # Deviation split constraints
    for (w_s, p_s, i_s) in combined.scenarios:
        wind = data.wind[w_s]

        for t in HOURS:
            sc = (w_s, p_s, i_s)

            # wind - offer = positive deviation - negative deviation
            model.addConstr(
                delta_pos[(sc, t)] - delta_neg[(sc, t)] == wind[t] - offer[t],
                name=f"deviation_balance_{w_s}_{p_s}_{i_s}_{t}"
            )

            # Only one of delta_pos and delta_neg can be positive
            model.addConstr(
                delta_pos[(sc, t)] <= M * z[(sc, t)],
                name=f"pos_activation_{w_s}_{p_s}_{i_s}_{t}"
            )
            model.addConstr(
                delta_neg[(sc, t)] <= M * (1 - z[(sc, t)]),
                name=f"neg_activation_{w_s}_{p_s}_{i_s}_{t}"
            )

    # Objective
    obj = gp.LinExpr()

    for (w_s, p_s, i_s) in combined.scenarios:
        prob = combined.probability[(w_s, p_s, i_s)]

        da = data.price[p_s]
        bp = data.balancing_price[(p_s, i_s)]
        si = data.imbalance[i_s]

        for t in HOURS:
            sc = (w_s, p_s, i_s)
            dp = delta_pos[(sc, t)]
            dn = delta_neg[(sc, t)]

            # Day-ahead revenue
            obj += prob * da[t] * offer[t]

            if si[t] == 1:
                # system deficit:
                # positive deviation helpful -> DA
                # negative deviation harmful -> BP
                obj += prob * (da[t] * dp - bp[t] * dn)
            else:
                # system surplus:
                # negative deviation helpful -> DA
                # positive deviation harmful -> BP
                obj += prob * (bp[t] * dp - da[t] * dn)

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    if model.Status == GRB.UNBOUNDED:
        raise RuntimeError("Model is unbounded.")
    elif model.Status == GRB.INFEASIBLE:
        raise RuntimeError("Model is infeasible.")
    elif model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Optimization ended with status {model.Status}.")

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
        n_wind_scenarios=30,
        n_price_scenarios=30,
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
    plot_profit_distribution(
        profits=profits,
        filename="outputs/figures/task_1_2_profit_distribution.png",
        title="Task 1.2 Profit Distribution - Two-Price Scheme"
    )

    print("\nFiles saved:")
    print(" - outputs/tables/task_1_2_offer.csv")
    print(" - outputs/figures/task_1_2_profit_distribution.png")


if __name__ == "__main__":
    main()