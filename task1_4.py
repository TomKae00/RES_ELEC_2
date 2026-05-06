# task_1_4.py

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
)

from task_1_1 import (
    solve_one_price_offering_problem,
    scenario_profit_one_price,
)

from task_1_2 import (
    solve_two_price_offering_problem,
    scenario_profit_two_price,
)

from plotting import (
    plot_profit_vs_cvar,
    plot_hourly_offer,
    plot_profit_distribution,
)


# ============================================================
# Evaluation functions
# ============================================================
def evaluate_one_price_profits(offer, data, combined):
    profits = []

    for sc in combined.scenarios:
        w_s, p_s, i_s = sc

        profit = scenario_profit_one_price(
            offer=offer,
            wind_realization=data.wind[w_s],
            da_price=data.price[p_s],
            balancing_price=data.balancing_price[(p_s, i_s)],
        )

        profits.append(profit)

    return np.array(profits)


def evaluate_two_price_profits(offer, data, combined):
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


def empirical_cvar_profit(profits, alpha=0.90):
    """
    Lower-tail CVaR of profit.
    For alpha = 0.90, this is the average of the worst 10% profits.
    """

    profits = np.sort(np.asarray(profits))
    n_tail = max(1, int(np.ceil((1.0 - alpha) * len(profits))))

    return float(np.mean(profits[:n_tail]))


def profit_statistics(profits, alpha=0.90):
    return {
        "expected_profit": float(np.mean(profits)),
        "cvar_profit": empirical_cvar_profit(profits, alpha=alpha),
        "profit_std": float(np.std(profits)),
        "min_profit": float(np.min(profits)),
        "p05_profit": float(np.quantile(profits, 0.05)),
        "p10_profit": float(np.quantile(profits, 0.10)),
        "p50_profit": float(np.quantile(profits, 0.50)),
        "max_profit": float(np.max(profits)),
    }


# ============================================================
# Risk-averse one-price model
# ============================================================
def solve_risk_averse_one_price(data, combined, alpha=0.90, beta=0.10):
    model = gp.Model("risk_averse_one_price")
    model.setParam("OutputFlag", 0)

    offer = model.addVars(HOURS, lb=0.0, ub=CAPACITY_MW, name="offer")

    zeta = model.addVar(lb=-GRB.INFINITY, name="zeta")
    eta = model.addVars(combined.scenarios, lb=0.0, name="eta")

    scenario_profit = {}

    for sc in combined.scenarios:
        w_s, p_s, i_s = sc

        wind = data.wind[w_s]
        da = data.price[p_s]
        bp = data.balancing_price[(p_s, i_s)]

        profit_expr = gp.LinExpr()

        for t in HOURS:
            profit_expr += da[t] * offer[t] + bp[t] * (wind[t] - offer[t])

        scenario_profit[sc] = profit_expr

        model.addConstr(
            eta[sc] >= zeta - profit_expr,
            name=f"cvar_shortfall_{sc}",
        )

    expected_profit = gp.quicksum(
        combined.probability[sc] * scenario_profit[sc]
        for sc in combined.scenarios
    )

    cvar_profit = zeta - (1.0 / (1.0 - alpha)) * gp.quicksum(
        combined.probability[sc] * eta[sc]
        for sc in combined.scenarios
    )

    model.setObjective(
        (1.0 - beta) * expected_profit + beta * cvar_profit,
        GRB.MAXIMIZE,
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"One-price CVaR model failed with status {model.Status}")

    offer_solution = np.array([offer[t].X for t in HOURS])

    return offer_solution


# ============================================================
# Risk-averse two-price model
# ============================================================
def solve_risk_averse_two_price(data, combined, alpha=0.90, beta=0.10):
    model = gp.Model("risk_averse_two_price")
    model.setParam("OutputFlag", 0)

    offer = model.addVars(HOURS, lb=0.0, ub=CAPACITY_MW, name="offer")

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

    zeta = model.addVar(lb=-GRB.INFINITY, name="zeta")
    eta = model.addVars(combined.scenarios, lb=0.0, name="eta")

    scenario_profit = {}

    for sc in combined.scenarios:
        w_s, p_s, i_s = sc

        wind = data.wind[w_s]
        da = data.price[p_s]
        si = data.imbalance[i_s]

        profit_expr = gp.LinExpr()

        for t in HOURS:
            model.addConstr(
                delta_up[(sc, t)] - delta_down[(sc, t)] == wind[t] - offer[t],
                name=f"deviation_balance_{sc}_{t}",
            )

            profit_expr += da[t] * offer[t]

            if si[t] == 1:
                # System deficit
                # Upward deviation is beneficial
                # Downward deviation is harmful
                profit_expr += da[t] * delta_up[(sc, t)]
                profit_expr += -1.25 * da[t] * delta_down[(sc, t)]

            else:
                # System surplus
                # Upward deviation is harmful
                # Downward deviation is beneficial
                profit_expr += 0.85 * da[t] * delta_up[(sc, t)]
                profit_expr += -da[t] * delta_down[(sc, t)]

        scenario_profit[sc] = profit_expr

        model.addConstr(
            eta[sc] >= zeta - profit_expr,
            name=f"cvar_shortfall_{sc}",
        )

    expected_profit = gp.quicksum(
        combined.probability[sc] * scenario_profit[sc]
        for sc in combined.scenarios
    )

    cvar_profit = zeta - (1.0 / (1.0 - alpha)) * gp.quicksum(
        combined.probability[sc] * eta[sc]
        for sc in combined.scenarios
    )

    model.setObjective(
        (1.0 - beta) * expected_profit + beta * cvar_profit,
        GRB.MAXIMIZE,
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Two-price CVaR model failed with status {model.Status}")

    offer_solution = np.array([offer[t].X for t in HOURS])

    return offer_solution


# ============================================================
# Main
# ============================================================
def main():
    ensure_output_folders()

    wind_file = "Data/scen_zone2.csv"
    price_file = "Data/DayAheadPrices.csv"

    alpha = 0.90

    beta_values = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.20,
        0.40,
        0.60,
        0.80,
        0.95,
        1.00,
    ]

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

    results = []
    offers = []

    reference_offers = {}

    for beta in beta_values:
        print(f"\nSolving beta = {beta:.2f}")

        # ----------------------------------------------------
        # One-price
        # ----------------------------------------------------
        if beta == 0.0:
            offer_one, _ = solve_one_price_offering_problem(data, combined)
        else:
            offer_one = solve_risk_averse_one_price(
                data=data,
                combined=combined,
                alpha=alpha,
                beta=beta,
            )

        profits_one = evaluate_one_price_profits(offer_one, data, combined)
        stats_one = profit_statistics(profits_one, alpha=alpha)

        if beta == 0.0:
            reference_offers["one-price"] = offer_one.copy()

        max_offer_change_one = float(
            np.max(np.abs(offer_one - reference_offers["one-price"]))
        )

        results.append({
            "scheme": "one-price",
            "beta": beta,
            **stats_one,
            "max_offer_change_from_beta_0": max_offer_change_one,
        })

        for t in HOURS:
            offers.append({
                "scheme": "one-price",
                "beta": beta,
                "hour": t,
                "offer_MW": offer_one[t],
            })

        # ----------------------------------------------------
        # Two-price
        # ----------------------------------------------------
        if beta == 0.0:
            offer_two, _ = solve_two_price_offering_problem(data, combined)
        else:
            offer_two = solve_risk_averse_two_price(
                data=data,
                combined=combined,
                alpha=alpha,
                beta=beta,
            )

        profits_two = evaluate_two_price_profits(offer_two, data, combined)
        stats_two = profit_statistics(profits_two, alpha=alpha)

        if beta == 0.0:
            reference_offers["two-price"] = offer_two.copy()

        max_offer_change_two = float(
            np.max(np.abs(offer_two - reference_offers["two-price"]))
        )

        results.append({
            "scheme": "two-price",
            "beta": beta,
            **stats_two,
            "max_offer_change_from_beta_0": max_offer_change_two,
        })

        for t in HOURS:
            offers.append({
                "scheme": "two-price",
                "beta": beta,
                "hour": t,
                "offer_MW": offer_two[t],
            })

        # Save selected figures only to avoid too many plots
        if beta in [0.00, 0.40, 0.80, 1.00]:
            plot_hourly_offer(
                offer=offer_one,
                filename=f"outputs/figures/task_1_4_offer_one_price_beta_{beta:.2f}.png",
                title=f"One-Price Risk-Averse Offer, beta={beta:.2f}",
            )

            plot_profit_distribution(
                profits=list(profits_one),
                filename=f"outputs/figures/task_1_4_profit_dist_one_price_beta_{beta:.2f}.png",
                title=f"One-Price Profit Distribution, beta={beta:.2f}",
            )

            plot_hourly_offer(
                offer=offer_two,
                filename=f"outputs/figures/task_1_4_offer_two_price_beta_{beta:.2f}.png",
                title=f"Two-Price Risk-Averse Offer, beta={beta:.2f}",
            )

            plot_profit_distribution(
                profits=list(profits_two),
                filename=f"outputs/figures/task_1_4_profit_dist_two_price_beta_{beta:.2f}.png",
                title=f"Two-Price Profit Distribution, beta={beta:.2f}",
            )

    results_df = pd.DataFrame(results)
    offers_df = pd.DataFrame(offers)

    results_df.to_csv(
        "outputs/tables/task_1_4_risk_averse_results.csv",
        index=False,
    )

    offers_df.to_csv(
        "outputs/tables/task_1_4_risk_averse_offers.csv",
        index=False,
    )

    plot_profit_vs_cvar(
        results_df=results_df,
        filename="outputs/figures/task_1_4_expected_profit_vs_cvar.png",
        title="Expected Profit vs CVaR, alpha=0.90",
    )

    print("\nRisk-averse results:")
    print(results_df)

    print("\nSaved:")
    print(" - outputs/tables/task_1_4_risk_averse_results.csv")
    print(" - outputs/tables/task_1_4_risk_averse_offers.csv")
    print(" - outputs/figures/task_1_4_expected_profit_vs_cvar.png")


if __name__ == "__main__":
    main()