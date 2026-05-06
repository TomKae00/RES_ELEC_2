# task_1_4.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from helpers import (
    HOURS,
    CAPACITY_MW,
    ensure_output_folders,
    prepare_scenario_data,
    prepare_volatile_test_scenario_data,
    evaluate_one_price_across_scenarios,
)

from task_1_2 import evaluate_two_price_across_scenarios

from plotting import (
    plot_hourly_offer,
    plot_profit_distribution,
    plot_profit_vs_cvar_single_scheme,
)


FIG_DIR = "outputs/figures/task_1_4"
TAB_DIR = "outputs/tables/task_1_4"


def make_output_dirs():
    ensure_output_folders()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)


def empirical_cvar(profits, alpha=0.90):
    """
    Average profit in the worst (1-alpha) tail.
    """
    profits = np.sort(np.asarray(profits, dtype=float))
    n_tail = max(1, int(np.ceil((1 - alpha) * len(profits))))
    return float(np.mean(profits[:n_tail]))


def profit_summary(profits, alpha=0.90):
    profits = np.asarray(profits, dtype=float)

    return {
        "expected_profit": float(np.mean(profits)),
        "cvar_profit": empirical_cvar(profits, alpha),
        "std_profit": float(np.std(profits)),
        "min_profit": float(np.min(profits)),
        "p05_profit": float(np.percentile(profits, 5)),
        "p10_profit": float(np.percentile(profits, 10)),
        "median_profit": float(np.percentile(profits, 50)),
        "max_profit": float(np.max(profits)),
    }


# ============================================================
# One-price CVaR model
# ============================================================
def solve_risk_averse_one_price(data, combined, alpha=0.90, beta=0.0):
    """
    Risk-averse one-price offering model.

    Objective:
        maximize E[Profit] + beta * CVaR

    One-price profit:
        Profit_s = sum_t DA_t,s * p_DA_t
                 + BP_t,s * (W_t,s - p_DA_t)
    """

    model = gp.Model("task_1_4_one_price_cvar")
    model.setParam("OutputFlag", 0)

    scenarios = list(combined.scenarios)

    offer = model.addVars(
        HOURS,
        lb=0.0,
        ub=CAPACITY_MW,
        name="offer",
    )

    scenario_profit = {}
    expected_profit = gp.LinExpr()

    for s_id, sc in enumerate(scenarios):
        w_s, p_s, i_s = sc
        prob = combined.probability[sc]

        wind = data.wind[w_s]
        da = data.price[p_s]
        bp = data.balancing_price[(p_s, i_s)]

        profit = gp.LinExpr()

        for t in HOURS:
            profit += da[t] * offer[t] + bp[t] * (wind[t] - offer[t])

        scenario_profit[s_id] = profit
        expected_profit += prob * profit

    zeta = model.addVar(lb=-GRB.INFINITY, name="zeta")
    eta = model.addVars(
        range(len(scenarios)),
        lb=0.0,
        name="eta",
    )

    for s_id in range(len(scenarios)):
        model.addConstr(
            eta[s_id] >= zeta - scenario_profit[s_id],
            name=f"cvar_shortfall_{s_id}",
        )

    cvar = zeta - (1.0 / (1.0 - alpha)) * gp.quicksum(
        combined.probability[scenarios[s_id]] * eta[s_id]
        for s_id in range(len(scenarios))
    )

    model.setObjective(
    (1.0 - beta) * expected_profit + beta * cvar,
    GRB.MAXIMIZE,
)

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"One-price model ended with status {model.Status}")

    offer_solution = np.array([offer[t].X for t in HOURS])

    return offer_solution, model.ObjVal


# ============================================================
# Two-price CVaR model
# ============================================================
def solve_risk_averse_two_price(data, combined, alpha=0.90, beta=0.0):
    """
    Risk-averse two-price offering model.

    Objective:
        maximize E[Profit] + beta * CVaR

    Deviation decomposition:
        W_t,s - p_DA_t = Delta_up_t,s - Delta_down_t,s
    """

    model = gp.Model("task_1_4_two_price_cvar")
    model.setParam("OutputFlag", 0)

    scenarios = list(combined.scenarios)

    offer = model.addVars(
        HOURS,
        lb=0.0,
        ub=CAPACITY_MW,
        name="offer",
    )

    delta_up = model.addVars(
        range(len(scenarios)),
        HOURS,
        lb=0.0,
        ub=CAPACITY_MW,
        name="delta_up",
    )

    delta_down = model.addVars(
        range(len(scenarios)),
        HOURS,
        lb=0.0,
        ub=CAPACITY_MW,
        name="delta_down",
    )

    scenario_profit = {}
    expected_profit = gp.LinExpr()

    for s_id, sc in enumerate(scenarios):
        w_s, p_s, i_s = sc
        prob = combined.probability[sc]

        wind = data.wind[w_s]
        da = data.price[p_s]
        bp = data.balancing_price[(p_s, i_s)]
        si = data.imbalance[i_s]

        profit = gp.LinExpr()

        for t in HOURS:
            model.addConstr(
                delta_up[s_id, t] - delta_down[s_id, t] == wind[t] - offer[t],
                name=f"deviation_balance_{s_id}_{t}",
            )

            profit += da[t] * offer[t]

            if si[t] == 1:
                # Deficit:
                # overproduction is beneficial, underproduction is harmful
                profit += da[t] * delta_up[s_id, t]
                profit -= bp[t] * delta_down[s_id, t]
            else:
                # Surplus:
                # underproduction is beneficial, overproduction is harmful
                profit += bp[t] * delta_up[s_id, t]
                profit -= da[t] * delta_down[s_id, t]

        scenario_profit[s_id] = profit
        expected_profit += prob * profit

    zeta = model.addVar(lb=-GRB.INFINITY, name="zeta")
    eta = model.addVars(
        range(len(scenarios)),
        lb=0.0,
        name="eta",
    )

    for s_id in range(len(scenarios)):
        model.addConstr(
            eta[s_id] >= zeta - scenario_profit[s_id],
            name=f"cvar_shortfall_{s_id}",
        )

    cvar = zeta - (1.0 / (1.0 - alpha)) * gp.quicksum(
        combined.probability[scenarios[s_id]] * eta[s_id]
        for s_id in range(len(scenarios))
    )

    model.setObjective(
    (1.0 - beta) * expected_profit + beta * cvar,
    GRB.MAXIMIZE,
)

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Two-price model ended with status {model.Status}")

    offer_solution = np.array([offer[t].X for t in HOURS])

    return offer_solution, model.ObjVal


# ============================================================
# Run beta-grid
# ============================================================
def run_beta_grid(data, combined, beta_values, alpha=0.90):
    results = []
    offers = []
    profits_all = []

    for beta in beta_values:
        print(f"\nSolving beta = {beta:.2f}")

        # -------------------------
        # One-price
        # -------------------------
        offer_one, obj_one = solve_risk_averse_one_price(
            data=data,
            combined=combined,
            alpha=alpha,
            beta=beta,
        )

        profits_one = evaluate_one_price_across_scenarios(
            offer_one,
            data,
            combined,
        )

        summary_one = profit_summary(profits_one, alpha)

        results.append({
            "scheme": "one-price",
            "beta": beta,
            "optimization_objective": obj_one,
            **summary_one,
        })

        offer_row_one = {"scheme": "one-price", "beta": beta}
        offer_row_one.update({f"h{t:02d}": offer_one[t] for t in HOURS})
        offers.append(offer_row_one)

        for s_id, profit in enumerate(profits_one):
            profits_all.append({
                "scheme": "one-price",
                "beta": beta,
                "scenario_id": s_id,
                "profit": profit,
            })

        print(
            f"  One-price: "
            f"E[Profit]={summary_one['expected_profit']:,.2f}, "
            f"CVaR={summary_one['cvar_profit']:,.2f}, "
            f"Std={summary_one['std_profit']:,.2f}"
        )

        # -------------------------
        # Two-price
        # -------------------------
        offer_two, obj_two = solve_risk_averse_two_price(
            data=data,
            combined=combined,
            alpha=alpha,
            beta=beta,
        )

        profits_two = evaluate_two_price_across_scenarios(
            offer_two,
            data,
            combined,
        )

        summary_two = profit_summary(profits_two, alpha)

        results.append({
            "scheme": "two-price",
            "beta": beta,
            "optimization_objective": obj_two,
            **summary_two,
        })

        offer_row_two = {"scheme": "two-price", "beta": beta}
        offer_row_two.update({f"h{t:02d}": offer_two[t] for t in HOURS})
        offers.append(offer_row_two)

        for s_id, profit in enumerate(profits_two):
            profits_all.append({
                "scheme": "two-price",
                "beta": beta,
                "scenario_id": s_id,
                "profit": profit,
            })

        print(
            f"  Two-price: "
            f"E[Profit]={summary_two['expected_profit']:,.2f}, "
            f"CVaR={summary_two['cvar_profit']:,.2f}, "
            f"Std={summary_two['std_profit']:,.2f}"
        )

    return (
        pd.DataFrame(results),
        pd.DataFrame(offers),
        pd.DataFrame(profits_all),
    )


# ============================================================
# Figures
# ============================================================
def save_selected_plots(results_df, offers_df, profits_df):
    plot_profit_vs_cvar_single_scheme(
        results_df=results_df,
        scheme="one-price",
        filename=os.path.join(FIG_DIR, "expected_profit_vs_cvar_one_price.png"),
        title="Task 1.4 Expected Profit versus CVaR - One-Price Scheme",
    )

    plot_profit_vs_cvar_single_scheme(
        results_df=results_df,
        scheme="two-price",
        filename=os.path.join(FIG_DIR, "expected_profit_vs_cvar_two_price.png"),
        title="Task 1.4 Expected Profit versus CVaR - Two-Price Scheme",
    )

    for scheme in ["one-price", "two-price"]:
        scheme_name = scheme.replace("-", "_")

        for beta in [0.0, 1.0, 10.0, 100.0]:
            offer_row = offers_df[
                (offers_df["scheme"] == scheme)
                & (np.isclose(offers_df["beta"], beta))
            ]

            if offer_row.empty:
                continue

            offer = offer_row[
                [f"h{t:02d}" for t in HOURS]
            ].iloc[0].to_numpy(float)

            plot_hourly_offer(
                offer=offer,
                filename=os.path.join(
                    FIG_DIR,
                    f"offer_{scheme_name}_beta_{beta:.2f}.png",
                ),
                title=f"Task 1.4 Hourly Offer - {scheme}, beta={beta:.2f}",
            )

            profits = profits_df[
                (profits_df["scheme"] == scheme)
                & (np.isclose(profits_df["beta"], beta))
            ]["profit"].to_numpy(float)

            plot_profit_distribution(
                profits=profits,
                filename=os.path.join(
                    FIG_DIR,
                    f"profit_distribution_{scheme_name}_beta_{beta:.2f}.png",
                ),
                title=f"Task 1.4 Profit Distribution - {scheme}, beta={beta:.2f}",
            )


# ============================================================
# Main
# ============================================================
def main():
    make_output_dirs()

    alpha = 0.90

    beta_values = np.array([
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
])

    data, combined = prepare_scenario_data(
         wind_scenario_file="Data/scen_zone2.csv",
         price_file="Data/DayAheadPrices.csv",
         n_wind_scenarios=20,
         n_price_scenarios=20,
         n_imbalance_scenarios=4,
         deficit_probability=0.5,
         seed=42,
         price_area="DK2",
     )
    #data, combined = prepare_volatile_test_scenario_data(
     #   n_wind_scenarios=30,
      #  n_price_scenarios=30,
      #  n_imbalance_scenarios=8,
       # deficit_probability=0.5,
       # seed=42,
    #)

    print(f"Total combined scenarios: {len(combined.scenarios)}")

    results_df, offers_df, profits_df = run_beta_grid(
        data=data,
        combined=combined,
        beta_values=beta_values,
        alpha=alpha,
    )

    results_df.to_csv(
        os.path.join(TAB_DIR, "cvar_results.csv"),
        index=False,
    )

    offers_df.to_csv(
        os.path.join(TAB_DIR, "cvar_offers.csv"),
        index=False,
    )

    profits_df.to_csv(
        os.path.join(TAB_DIR, "cvar_scenario_profits.csv"),
        index=False,
    )

    save_selected_plots(results_df, offers_df, profits_df)

    print("\nSaved tables:")
    print(f" - {TAB_DIR}/cvar_results.csv")
    print(f" - {TAB_DIR}/cvar_offers.csv")
    print(f" - {TAB_DIR}/cvar_scenario_profits.csv")

    print("\nSaved figures:")
    print(f" - {FIG_DIR}/expected_profit_vs_cvar_one_price.png")
    print(f" - {FIG_DIR}/expected_profit_vs_cvar_two_price.png")
    print(f" - {FIG_DIR}/offer_<scheme>_beta_<beta>.png")
    print(f" - {FIG_DIR}/profit_distribution_<scheme>_beta_<beta>.png")


if __name__ == "__main__":
    main()