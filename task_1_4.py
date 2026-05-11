# task_1_4.py

from __future__ import annotations

import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from helpers import (
    HOURS,
    CAPACITY_MW,
    prepare_scenario_data,
    evaluate_one_price_across_scenarios,
)

from task_1_2 import evaluate_two_price_across_scenarios

from plotting import (
    plot_hourly_offer,
    plot_profit_distribution,
    plot_profit_vs_cvar_single_scheme,
)


TASK_1_OUTPUT_DIR = Path("outputs") / "task_1"
TASK_1_4_OUTPUT_DIR = TASK_1_OUTPUT_DIR / "task_1_4"


def empirical_cvar(
    profits: np.ndarray,
    alpha: float = 0.90,
) -> float:
    """
    Calculate the empirical CVaR of a profit distribution.

    The CVaR is calculated as the average profit in the worst
    ``1 - alpha`` share of scenarios.

    Parameters
    ----------
    profits : np.ndarray
        Scenario profit values [EUR].

    alpha : float, default 0.90
        Confidence level used for the CVaR calculation.

    Returns
    -------
    float
        Empirical CVaR value [EUR].
    """

    profits = np.sort(np.asarray(profits, dtype=float))
    n_tail = max(1, int(np.ceil((1.0 - alpha) * len(profits))))

    return float(np.mean(profits[:n_tail]))


def profit_summary(
    profits: np.ndarray,
    alpha: float = 0.90,
) -> dict:
    """
    Build a compact summary of a scenario profit distribution.

    Parameters
    ----------
    profits : np.ndarray
        Scenario profit values [EUR].

    alpha : float, default 0.90
        Confidence level used for the empirical CVaR calculation.

    Returns
    -------
    dict
        Summary statistics of the profit distribution.
    """

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


def solve_risk_averse_one_price(
    data,
    combined,
    alpha: float = 0.90,
    beta: float = 0.0,
) -> tuple[np.ndarray, float, dict]:
    """
    Solve the risk-averse one-price offering model.

    The model maximizes a weighted objective consisting of expected profit and
    CVaR. The day-ahead offer is a first-stage decision, while scenario profits
    are used to formulate the CVaR term.

    Parameters
    ----------
    data : ScenarioData
        Scenario data containing wind production, day-ahead prices, system
        imbalance indicators, and balancing prices.

    combined : CombinedScenarioSet
        Combined scenario set used in the stochastic optimization.

    alpha : float, default 0.90
        Confidence level for the CVaR formulation.

    beta : float, default 0.0
        Weight on the CVaR term in the objective.

    Returns
    -------
    offer_solution : np.ndarray
        Optimal hourly day-ahead offer [MW].

    objective_value : float
        Optimized objective value of the risk-averse model [EUR].

    model_stats : dict
        Model statistics including variables, constraints and solve time.
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
        probability = combined.probability[sc]

        wind = data.wind[w_s]
        da_price = data.price[p_s]
        balancing_price = data.balancing_price[(p_s, i_s)]

        profit = gp.LinExpr()

        for t in HOURS:
            profit += (
                da_price[t] * offer[t]
                + balancing_price[t] * (wind[t] - offer[t])
            )

        scenario_profit[s_id] = profit
        expected_profit += probability * profit

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

    start_time = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start_time

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"One-price risk-averse model ended with status {model.Status}"
        )

    offer_solution = np.array([offer[t].X for t in HOURS])
    objective_value = float(model.ObjVal)

    model_stats = {
        "variables": model.NumVars,
        "constraints": model.NumConstrs,
        "solve_time_s": solve_time,
        "objective_value": objective_value,
    }

    return offer_solution, objective_value, model_stats


def solve_risk_averse_two_price(
    data,
    combined,
    alpha: float = 0.90,
    beta: float = 0.0,
) -> tuple[np.ndarray, float, dict]:
    """
    Solve the risk-averse two-price offering model.

    The model maximizes a weighted objective consisting of expected profit and
    CVaR. Positive and negative imbalances are represented by non-negative
    auxiliary variables, as in the risk-neutral two-price formulation.

    Parameters
    ----------
    data : ScenarioData
        Scenario data containing wind production, day-ahead prices, system
        imbalance indicators, and balancing prices.

    combined : CombinedScenarioSet
        Combined scenario set used in the stochastic optimization.

    alpha : float, default 0.90
        Confidence level for the CVaR formulation.

    beta : float, default 0.0
        Weight on the CVaR term in the objective.

    Returns
    -------
    offer_solution : np.ndarray
        Optimal hourly day-ahead offer [MW].

    objective_value : float
        Optimized objective value of the risk-averse model [EUR].

    model_stats : dict
        Model statistics including variables, constraints and solve time.
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
        probability = combined.probability[sc]

        wind = data.wind[w_s]
        da_price = data.price[p_s]
        balancing_price = data.balancing_price[(p_s, i_s)]
        system_imbalance = data.imbalance[i_s]

        profit = gp.LinExpr()

        for t in HOURS:
            model.addConstr(
                delta_up[s_id, t] - delta_down[s_id, t] == wind[t] - offer[t],
                name=f"deviation_balance_{s_id}_{t}",
            )

            profit += da_price[t] * offer[t]

            if system_imbalance[t] == 1:
                profit += da_price[t] * delta_up[s_id, t]
                profit -= balancing_price[t] * delta_down[s_id, t]
            else:
                profit += balancing_price[t] * delta_up[s_id, t]
                profit -= da_price[t] * delta_down[s_id, t]

        scenario_profit[s_id] = profit
        expected_profit += probability * profit

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

    start_time = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start_time

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Two-price risk-averse model ended with status {model.Status}"
        )

    offer_solution = np.array([offer[t].X for t in HOURS])
    objective_value = float(model.ObjVal)

    model_stats = {
        "variables": model.NumVars,
        "constraints": model.NumConstrs,
        "solve_time_s": solve_time,
        "objective_value": objective_value,
    }

    return offer_solution, objective_value, model_stats


def run_beta_grid(
    data,
    combined,
    beta_values: np.ndarray,
    alpha: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Solve the risk-averse one-price and two-price models for all beta values.

    For each beta value, both settlement schemes are optimized. The resulting
    day-ahead offer is then evaluated across all combined scenarios to compute
    expected profit, empirical CVaR and additional profit statistics.

    Parameters
    ----------
    data : ScenarioData
        Scenario data containing wind production, day-ahead prices, system
        imbalance indicators, and balancing prices.

    combined : CombinedScenarioSet
        Combined scenario set used for optimization and evaluation.

    beta_values : np.ndarray
        Beta values used to weight the CVaR term.

    alpha : float, default 0.90
        Confidence level used for the CVaR formulation and empirical
        evaluation.

    Returns
    -------
    results_df : pd.DataFrame
        Summary results for each settlement scheme and beta value.

    offers_df : pd.DataFrame
        Optimal hourly offers for each settlement scheme and beta value.

    profits_df : pd.DataFrame
        Scenario-level profit values for each settlement scheme and beta value.
    """

    results = []
    offers = []
    profits_all = []

    for beta in beta_values:
        print(f"\nSolving beta = {beta:.2f}")

        offer_one, obj_one, stats_one = solve_risk_averse_one_price(
            data=data,
            combined=combined,
            alpha=alpha,
            beta=beta,
        )

        profits_one = np.asarray(
            evaluate_one_price_across_scenarios(
                offer=offer_one,
                data=data,
                combined=combined,
            ),
            dtype=float,
        )

        summary_one = profit_summary(profits_one, alpha)

        results.append(
            {
                "scheme": "one-price",
                "beta": beta,
                "alpha": alpha,
                "optimization_objective": obj_one,
                **summary_one,
                "variables": stats_one["variables"],
                "constraints": stats_one["constraints"],
                "solve_time_s": stats_one["solve_time_s"],
            }
        )

        offer_row_one = {"scheme": "one-price", "beta": beta}
        offer_row_one.update({f"h{t:02d}": offer_one[t] for t in HOURS})
        offers.append(offer_row_one)

        for scenario_id, profit in enumerate(profits_one):
            profits_all.append(
                {
                    "scheme": "one-price",
                    "beta": beta,
                    "scenario_id": scenario_id,
                    "profit": profit,
                }
            )

        print(
            f"  One-price: "
            f"E[Profit]={summary_one['expected_profit']:,.2f}, "
            f"CVaR={summary_one['cvar_profit']:,.2f}, "
            f"Std={summary_one['std_profit']:,.2f}, "
            f"Solve time={stats_one['solve_time_s']:.2f}s"
        )

        offer_two, obj_two, stats_two = solve_risk_averse_two_price(
            data=data,
            combined=combined,
            alpha=alpha,
            beta=beta,
        )

        profits_two = np.asarray(
            evaluate_two_price_across_scenarios(
                offer=offer_two,
                data=data,
                combined=combined,
            ),
            dtype=float,
        )

        summary_two = profit_summary(profits_two, alpha)

        results.append(
            {
                "scheme": "two-price",
                "beta": beta,
                "alpha": alpha,
                "optimization_objective": obj_two,
                **summary_two,
                "variables": stats_two["variables"],
                "constraints": stats_two["constraints"],
                "solve_time_s": stats_two["solve_time_s"],
            }
        )

        offer_row_two = {"scheme": "two-price", "beta": beta}
        offer_row_two.update({f"h{t:02d}": offer_two[t] for t in HOURS})
        offers.append(offer_row_two)

        for scenario_id, profit in enumerate(profits_two):
            profits_all.append(
                {
                    "scheme": "two-price",
                    "beta": beta,
                    "scenario_id": scenario_id,
                    "profit": profit,
                }
            )

        print(
            f"  Two-price: "
            f"E[Profit]={summary_two['expected_profit']:,.2f}, "
            f"CVaR={summary_two['cvar_profit']:,.2f}, "
            f"Std={summary_two['std_profit']:,.2f}, "
            f"Solve time={stats_two['solve_time_s']:.2f}s"
        )

    return (
        pd.DataFrame(results),
        pd.DataFrame(offers),
        pd.DataFrame(profits_all),
    )


def save_selected_plots(
    results_df: pd.DataFrame,
    offers_df: pd.DataFrame,
    profits_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """
    Create and save the selected Task 1.4 report figures.

    Parameters
    ----------
    results_df : pd.DataFrame
        Summary results for each settlement scheme and beta value.

    offers_df : pd.DataFrame
        Optimal hourly offers for each settlement scheme and beta value.

    profits_df : pd.DataFrame
        Scenario-level profit values for each settlement scheme and beta value.

    output_dir : Path
        Directory where the figures should be saved.

    Returns
    -------
    list[Path]
        List of generated figure files.
    """

    generated_files = []

    one_price_cvar_plot = output_dir / "task_1_4_expected_profit_vs_cvar_one_price.png"
    two_price_cvar_plot = output_dir / "task_1_4_expected_profit_vs_cvar_two_price.png"

    plot_profit_vs_cvar_single_scheme(
        results_df=results_df,
        scheme="one-price",
        filename=str(one_price_cvar_plot),
    )

    plot_profit_vs_cvar_single_scheme(
        results_df=results_df,
        scheme="two-price",
        filename=str(two_price_cvar_plot),
    )

    generated_files.extend(
        [
            one_price_cvar_plot,
            two_price_cvar_plot,
        ]
    )

    selected_betas = [0.0, 1.0]

    for scheme in ["one-price", "two-price"]:
        scheme_name = scheme.replace("-", "_")

        for beta in selected_betas:
            offer_row = offers_df[
                (offers_df["scheme"] == scheme)
                & (np.isclose(offers_df["beta"], beta))
            ]

            if offer_row.empty:
                continue

            offer = offer_row[
                [f"h{t:02d}" for t in HOURS]
            ].iloc[0].to_numpy(float)

            offer_plot_file = (
                output_dir
                / f"task_1_4_offer_{scheme_name}_beta_{beta:.2f}.png"
            )

            plot_hourly_offer(
                offer=offer,
                filename=str(offer_plot_file),
            )

            generated_files.append(offer_plot_file)

            profits = profits_df[
                (profits_df["scheme"] == scheme)
                & (np.isclose(profits_df["beta"], beta))
            ]["profit"].to_numpy(float)

            profit_distribution_file = (
                output_dir
                / f"task_1_4_profit_distribution_{scheme_name}_beta_{beta:.2f}.png"
            )

            plot_profit_distribution(
                profits=profits,
                filename=str(profit_distribution_file),
            )

            generated_files.append(profit_distribution_file)

    return generated_files


def main() -> None:
    """
    Run the complete Task 1.4 risk-averse offering workflow.

    The workflow solves the one-price and two-price risk-averse offering
    models for a grid of beta values, evaluates the resulting offers across
    all scenarios, saves summary tables and scenario-level profits, and creates
    the selected report figures.

    All outputs are saved in:

        outputs/task_1/task_1_4/

    Returns
    -------
    None
        The function writes output files and prints a compact summary to the
        terminal.
    """

    TASK_1_4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha = 0.90

    beta_values = np.array(
        [
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
        ]
    )

    data, combined = prepare_scenario_data(
        wind_scenario_file="data/scen_zone2.csv",
        price_file="data/DayAheadPrices.csv",
        n_wind_scenarios=20,
        n_price_scenarios=20,
        n_imbalance_scenarios=4,
        deficit_probability=0.5,
        seed=42,
        price_area="DK2",
    )

    print(f"Total combined scenarios: {len(combined.scenarios)}")

    results_df, offers_df, profits_df = run_beta_grid(
        data=data,
        combined=combined,
        beta_values=beta_values,
        alpha=alpha,
    )

    results_file = TASK_1_4_OUTPUT_DIR / "task_1_4_cvar_results.csv"
    offers_file = TASK_1_4_OUTPUT_DIR / "task_1_4_cvar_offers.csv"
    profits_file = TASK_1_4_OUTPUT_DIR / "task_1_4_cvar_scenario_profits.csv"

    results_df.to_csv(results_file, index=False)
    offers_df.to_csv(offers_file, index=False)
    profits_df.to_csv(profits_file, index=False)

    generated_files = [
        results_file,
        offers_file,
        profits_file,
    ]

    generated_files.extend(
        save_selected_plots(
            results_df=results_df,
            offers_df=offers_df,
            profits_df=profits_df,
            output_dir=TASK_1_4_OUTPUT_DIR,
        )
    )

    print("\nFiles saved:")
    for file in generated_files:
        print(f" - {file}")


if __name__ == "__main__":
    main()