# task_1_4.py

from __future__ import annotations

import os
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
    subset_combined_scenarios,
    evaluate_one_price_across_scenarios,
)

from task_1_2 import evaluate_two_price_across_scenarios

from plotting import (
    plot_hourly_offer,
    plot_profit_distribution,
    plot_profit_vs_cvar,
)


FIG_DIR = "outputs/figures/task_1_4"
TAB_DIR = "outputs/tables/task_1_4"


def make_output_dirs() -> None:
    """
    Create all output folders required for Task 1.4.

    Returns
    -------
    None
        The function creates the general output folders and the Task 1.4
        figure and table subfolders.
    """

    ensure_output_folders()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)


def validate_beta(beta: float) -> None:
    """
    Check that beta is within the normalized weighting range [0, 1].

    Parameters
    ----------
    beta : float
        Weight assigned to the CVaR term.

    Returns
    -------
    None
        Raises a ValueError if beta is outside [0, 1].
    """

    if beta < 0.0 or beta > 1.0:
        raise ValueError(
            f"beta must be between 0 and 1 for the normalized objective. "
            f"Received beta={beta}."
        )


def empirical_cvar(profits: np.ndarray, alpha: float = 0.90) -> float:
    """
    Calculate empirical CVaR of a profit distribution.

    CVaR is calculated as the average profit in the worst ``1 - alpha`` tail
    of the empirical profit distribution. For example, with ``alpha = 0.90``,
    the function averages the worst 10% of scenario profits.

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


def profit_summary(profits: np.ndarray, alpha: float = 0.90) -> dict:
    """
    Create summary statistics for a scenario profit distribution.

    Parameters
    ----------
    profits : np.ndarray
        Scenario profit values [EUR].

    alpha : float, default 0.90
        Confidence level used for the empirical CVaR calculation.

    Returns
    -------
    dict
        Dictionary containing expected profit, CVaR, standard deviation,
        minimum, selected percentiles, median, and maximum profit.
    """

    profits = np.asarray(profits, dtype=float)

    return {
        "expected_profit_eur": float(np.mean(profits)),
        "cvar_profit_eur": empirical_cvar(profits, alpha),
        "std_profit_eur": float(np.std(profits)),
        "min_profit_eur": float(np.min(profits)),
        "p05_profit_eur": float(np.percentile(profits, 5)),
        "p10_profit_eur": float(np.percentile(profits, 10)),
        "median_profit_eur": float(np.percentile(profits, 50)),
        "max_profit_eur": float(np.max(profits)),
    }


def solve_risk_averse_one_price(
    data,
    combined,
    alpha: float = 0.90,
    beta: float = 0.0,
) -> tuple[np.ndarray, float, dict]:
    """
    Solve the risk-averse one-price offering model.

    The model maximizes a normalized weighted objective consisting of expected
    profit and CVaR. The day-ahead offer is a first-stage decision and is
    therefore not scenario-indexed. Under the one-price settlement scheme, all
    imbalances are settled at the balancing price.

    Objective:
        (1 - beta) * E[profit] + beta * CVaR

    One-price scenario profit:
        sum_t [ DA_price_t * offer_t
                + balancing_price_t * (wind_realization_t - offer_t) ]

    Parameters
    ----------
    data : ScenarioData
        Scenario data containing wind production, day-ahead prices, balancing
        prices, and system imbalance indicators.

    combined : CombinedScenarioSet
        Combined scenario set used for optimization.

    alpha : float, default 0.90
        Confidence level for the CVaR term.

    beta : float, default 0.0
        Risk-aversion weight in the interval [0, 1]. ``beta = 0`` corresponds
        to the risk-neutral expected-profit model, while ``beta = 1`` maximizes
        CVaR only.

    Returns
    -------
    offer_solution : np.ndarray
        Optimal hourly day-ahead offer [MW].

    risk_adjusted_objective : float
        Optimized value of (1 - beta) * E[profit] + beta * CVaR [EUR].

    model_stats : dict
        Dictionary containing model size, solve time, objective value, beta,
        and alpha.
    """

    validate_beta(beta)

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

    # VaR variable and auxiliary shortfall variables.
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
            f"One-price CVaR model ended with status {model.Status}"
        )

    offer_solution = np.array([offer[t].X for t in HOURS])
    risk_adjusted_objective = float(model.ObjVal)

    model_stats = {
        "scheme": "one-price",
        "alpha": alpha,
        "beta": beta,
        "variables": model.NumVars,
        "constraints": model.NumConstrs,
        "solve_time_s": solve_time,
        "risk_adjusted_objective_eur": risk_adjusted_objective,
        "both_delta_variables_active_count": np.nan,
    }

    return offer_solution, risk_adjusted_objective, model_stats


def solve_risk_averse_two_price(
    data,
    combined,
    alpha: float = 0.90,
    beta: float = 0.0,
) -> tuple[np.ndarray, float, dict]:
    """
    Solve the risk-averse two-price offering model.

    The model maximizes a normalized weighted objective consisting of expected
    profit and CVaR. The day-ahead offer is a first-stage decision and is
    therefore not scenario-indexed. Positive and negative imbalances are
    represented by two non-negative auxiliary variables, following the linear
    lecture formulation.

    Objective:
        (1 - beta) * E[profit] + beta * CVaR

    Two-price settlement:
    - system deficit:
        positive wind imbalance is desired and settled at the DA price,
        negative wind imbalance is undesired and settled at the balancing price.
    - system surplus:
        positive wind imbalance is undesired and settled at the balancing price,
        negative wind imbalance is desired and settled at the DA price.

    Parameters
    ----------
    data : ScenarioData
        Scenario data containing wind production, day-ahead prices, balancing
        prices, and system imbalance indicators.

    combined : CombinedScenarioSet
        Combined scenario set used for optimization.

    alpha : float, default 0.90
        Confidence level for the CVaR term.

    beta : float, default 0.0
        Risk-aversion weight in the interval [0, 1]. ``beta = 0`` corresponds
        to the risk-neutral expected-profit model, while ``beta = 1`` maximizes
        CVaR only.

    Returns
    -------
    offer_solution : np.ndarray
        Optimal hourly day-ahead offer [MW].

    risk_adjusted_objective : float
        Optimized value of (1 - beta) * E[profit] + beta * CVaR [EUR].

    model_stats : dict
        Dictionary containing model size, solve time, objective value, beta,
        alpha, and a diagnostic count for simultaneous activation of positive
        and negative imbalance variables.
    """

    validate_beta(beta)

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
                delta_up[s_id, t] - delta_down[s_id, t]
                == wind[t] - offer[t],
                name=f"deviation_balance_{s_id}_{t}",
            )

            profit += da_price[t] * offer[t]

            if system_imbalance[t] == 1:
                # System deficit:
                # positive wind imbalance helps the system -> DA price
                # negative wind imbalance worsens the system -> balancing price
                profit += da_price[t] * delta_up[s_id, t]
                profit -= balancing_price[t] * delta_down[s_id, t]
            else:
                # System surplus:
                # positive wind imbalance worsens the system -> balancing price
                # negative wind imbalance helps the system -> DA price
                profit += balancing_price[t] * delta_up[s_id, t]
                profit -= da_price[t] * delta_down[s_id, t]

        scenario_profit[s_id] = profit
        expected_profit += probability * profit

    # VaR variable and auxiliary shortfall variables.
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
            f"Two-price CVaR model ended with status {model.Status}"
        )

    offer_solution = np.array([offer[t].X for t in HOURS])
    risk_adjusted_objective = float(model.ObjVal)

    both_active_count = 0
    tolerance = 1e-6

    for s_id in range(len(scenarios)):
        for t in HOURS:
            if (
                delta_up[s_id, t].X > tolerance
                and delta_down[s_id, t].X > tolerance
            ):
                both_active_count += 1

    model_stats = {
        "scheme": "two-price",
        "alpha": alpha,
        "beta": beta,
        "variables": model.NumVars,
        "constraints": model.NumConstrs,
        "solve_time_s": solve_time,
        "risk_adjusted_objective_eur": risk_adjusted_objective,
        "both_delta_variables_active_count": both_active_count,
    }

    return offer_solution, risk_adjusted_objective, model_stats


def run_beta_grid(
    data,
    combined,
    beta_values: np.ndarray,
    alpha: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Solve the risk-averse models for a grid of beta values.

    For each beta value, both the one-price and two-price CVaR models are
    solved. The resulting offers are evaluated across all scenarios to obtain
    expected profit, empirical CVaR, and profit-distribution statistics.

    Parameters
    ----------
    data : ScenarioData
        Scenario data used for optimization and evaluation.

    combined : CombinedScenarioSet
        Combined scenario set used for optimization and evaluation.

    beta_values : np.ndarray
        Array of risk-aversion weights in the interval [0, 1].

    alpha : float, default 0.90
        Confidence level used in the CVaR formulation.

    Returns
    -------
    results_df : pd.DataFrame
        Summary table with one row per scheme and beta value.

    offers_df : pd.DataFrame
        Hourly offer table with one row per scheme and beta value.

    profits_df : pd.DataFrame
        Scenario profit table with one row per scheme, beta value, and scenario.

    model_stats_df : pd.DataFrame
        Model statistics with one row per scheme and beta value.
    """

    results = []
    offers = []
    profits_all = []
    model_stats_all = []

    for beta in beta_values:
        validate_beta(float(beta))
        print(f"\nSolving beta = {beta:.2f}")

        # -------------------------
        # One-price model
        # -------------------------
        offer_one, obj_one, stats_one = solve_risk_averse_one_price(
            data=data,
            combined=combined,
            alpha=alpha,
            beta=float(beta),
        )

        profits_one = np.array(
            evaluate_one_price_across_scenarios(
                offer_one,
                data,
                combined,
            )
        )

        summary_one = profit_summary(profits_one, alpha)
        model_stats_all.append(stats_one)

        results.append(
            {
                "scheme": "one-price",
                "alpha": alpha,
                "beta": float(beta),
                "risk_adjusted_objective_eur": obj_one,
                **summary_one,
            }
        )

        offer_row_one = {"scheme": "one-price", "beta": float(beta)}
        offer_row_one.update({f"h{t:02d}": offer_one[t] for t in HOURS})
        offers.append(offer_row_one)

        for s_id, profit in enumerate(profits_one):
            profits_all.append(
                {
                    "scheme": "one-price",
                    "beta": float(beta),
                    "scenario_id": s_id,
                    "profit_eur": profit,
                }
            )

        print(
            f"  One-price: "
            f"E[Profit]={summary_one['expected_profit_eur']:,.2f}, "
            f"CVaR={summary_one['cvar_profit_eur']:,.2f}, "
            f"Std={summary_one['std_profit_eur']:,.2f}, "
            f"Solve time={stats_one['solve_time_s']:.3f}s"
        )

        # -------------------------
        # Two-price model
        # -------------------------
        offer_two, obj_two, stats_two = solve_risk_averse_two_price(
            data=data,
            combined=combined,
            alpha=alpha,
            beta=float(beta),
        )

        profits_two = np.array(
            evaluate_two_price_across_scenarios(
                offer_two,
                data,
                combined,
            )
        )

        summary_two = profit_summary(profits_two, alpha)
        model_stats_all.append(stats_two)

        results.append(
            {
                "scheme": "two-price",
                "alpha": alpha,
                "beta": float(beta),
                "risk_adjusted_objective_eur": obj_two,
                **summary_two,
            }
        )

        offer_row_two = {"scheme": "two-price", "beta": float(beta)}
        offer_row_two.update({f"h{t:02d}": offer_two[t] for t in HOURS})
        offers.append(offer_row_two)

        for s_id, profit in enumerate(profits_two):
            profits_all.append(
                {
                    "scheme": "two-price",
                    "beta": float(beta),
                    "scenario_id": s_id,
                    "profit_eur": profit,
                }
            )

        print(
            f"  Two-price: "
            f"E[Profit]={summary_two['expected_profit_eur']:,.2f}, "
            f"CVaR={summary_two['cvar_profit_eur']:,.2f}, "
            f"Std={summary_two['std_profit_eur']:,.2f}, "
            f"Solve time={stats_two['solve_time_s']:.3f}s"
        )

    return (
        pd.DataFrame(results),
        pd.DataFrame(offers),
        pd.DataFrame(profits_all),
        pd.DataFrame(model_stats_all),
    )


def run_in_sample_sensitivity(
    data,
    combined,
    beta_values: list[float],
    alpha: float = 0.90,
    n_in_sample: int = 200,
    n_repetitions: int = 5,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Test sensitivity of risk-averse solutions to altered in-sample scenarios.

    The assignment asks whether altering the in-sample scenarios significantly
    affects the risk-averse solutions. This function repeatedly samples
    different in-sample scenario subsets, solves both risk-averse models for
    selected beta values, and evaluates the resulting offers on the full
    scenario set.

    Parameters
    ----------
    data : ScenarioData
        Scenario data used for optimization and evaluation.

    combined : CombinedScenarioSet
        Full combined scenario set.

    beta_values : list[float]
        Selected beta values used for the sensitivity analysis.

    alpha : float, default 0.90
        Confidence level used in the CVaR formulation.

    n_in_sample : int, default 200
        Number of scenarios sampled for each in-sample optimization run.

    n_repetitions : int, default 5
        Number of random in-sample subsets.

    seed : int, default 123
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Sensitivity results with one row per scheme, beta value, and repetition.
    """

    for beta in beta_values:
        validate_beta(float(beta))

    rng = np.random.default_rng(seed)
    all_scenarios = list(combined.scenarios)
    sensitivity_results = []

    for repetition in range(1, n_repetitions + 1):
        selected_indices = rng.choice(
            len(all_scenarios),
            size=n_in_sample,
            replace=False,
        )
        selected_scenarios = [all_scenarios[i] for i in selected_indices]

        in_combined = subset_combined_scenarios(
            combined=combined,
            selected_scenarios=selected_scenarios,
        )

        print(
            f"\nIn-sample sensitivity repetition {repetition}/{n_repetitions} "
            f"with {n_in_sample} scenarios"
        )

        for beta in beta_values:
            beta = float(beta)
            print(f"  beta = {beta:.2f}")

            # One-price
            offer_one, obj_one, stats_one = solve_risk_averse_one_price(
                data=data,
                combined=in_combined,
                alpha=alpha,
                beta=beta,
            )
            profits_one_full = np.array(
                evaluate_one_price_across_scenarios(
                    offer_one,
                    data,
                    combined,
                )
            )
            summary_one = profit_summary(profits_one_full, alpha)

            sensitivity_results.append(
                {
                    "repetition": repetition,
                    "scheme": "one-price",
                    "alpha": alpha,
                    "beta": beta,
                    "n_in_sample": n_in_sample,
                    "risk_adjusted_objective_in_sample_eur": obj_one,
                    "full_sample_expected_profit_eur": summary_one[
                        "expected_profit_eur"
                    ],
                    "full_sample_cvar_profit_eur": summary_one[
                        "cvar_profit_eur"
                    ],
                    "full_sample_std_profit_eur": summary_one[
                        "std_profit_eur"
                    ],
                    "solve_time_s": stats_one["solve_time_s"],
                    "variables": stats_one["variables"],
                    "constraints": stats_one["constraints"],
                    "mean_offer_mw": float(np.mean(offer_one)),
                    "min_offer_mw": float(np.min(offer_one)),
                    "max_offer_mw": float(np.max(offer_one)),
                }
            )

            # Two-price
            offer_two, obj_two, stats_two = solve_risk_averse_two_price(
                data=data,
                combined=in_combined,
                alpha=alpha,
                beta=beta,
            )
            profits_two_full = np.array(
                evaluate_two_price_across_scenarios(
                    offer_two,
                    data,
                    combined,
                )
            )
            summary_two = profit_summary(profits_two_full, alpha)

            sensitivity_results.append(
                {
                    "repetition": repetition,
                    "scheme": "two-price",
                    "alpha": alpha,
                    "beta": beta,
                    "n_in_sample": n_in_sample,
                    "risk_adjusted_objective_in_sample_eur": obj_two,
                    "full_sample_expected_profit_eur": summary_two[
                        "expected_profit_eur"
                    ],
                    "full_sample_cvar_profit_eur": summary_two[
                        "cvar_profit_eur"
                    ],
                    "full_sample_std_profit_eur": summary_two[
                        "std_profit_eur"
                    ],
                    "solve_time_s": stats_two["solve_time_s"],
                    "variables": stats_two["variables"],
                    "constraints": stats_two["constraints"],
                    "both_delta_variables_active_count": stats_two[
                        "both_delta_variables_active_count"
                    ],
                    "mean_offer_mw": float(np.mean(offer_two)),
                    "min_offer_mw": float(np.min(offer_two)),
                    "max_offer_mw": float(np.max(offer_two)),
                }
            )

    return pd.DataFrame(sensitivity_results)


def build_in_sample_sensitivity_summary(
    sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the in-sample scenario sensitivity results.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Detailed sensitivity results from ``run_in_sample_sensitivity``.

    Returns
    -------
    pd.DataFrame
        Aggregated sensitivity table grouped by scheme and beta value.
    """

    summary_df = (
        sensitivity_df.groupby(["scheme", "beta"], as_index=False)
        .agg(
            mean_full_sample_expected_profit_eur=(
                "full_sample_expected_profit_eur",
                "mean",
            ),
            std_full_sample_expected_profit_eur=(
                "full_sample_expected_profit_eur",
                "std",
            ),
            mean_full_sample_cvar_profit_eur=(
                "full_sample_cvar_profit_eur",
                "mean",
            ),
            std_full_sample_cvar_profit_eur=(
                "full_sample_cvar_profit_eur",
                "std",
            ),
            mean_full_sample_std_profit_eur=(
                "full_sample_std_profit_eur",
                "mean",
            ),
            mean_offer_mw=("mean_offer_mw", "mean"),
            std_mean_offer_mw=("mean_offer_mw", "std"),
            mean_solve_time_s=("solve_time_s", "mean"),
        )
    )

    return summary_df


def save_selected_plots(
    results_df: pd.DataFrame,
    offers_df: pd.DataFrame,
    profits_df: pd.DataFrame,
) -> None:
    """
    Save selected Task 1.4 figures.

    The figures include the efficient-frontier-style plot of expected profit
    versus CVaR and selected offer/profit-distribution plots for representative
    beta values.

    Parameters
    ----------
    results_df : pd.DataFrame
        Summary table with expected profit and CVaR for each scheme and beta.

    offers_df : pd.DataFrame
        Hourly offer table for each scheme and beta.

    profits_df : pd.DataFrame
        Scenario profit table for each scheme and beta.

    Returns
    -------
    None
        The function saves figures to ``FIG_DIR``.
    """

    plot_profit_vs_cvar(
        results_df=results_df,
        filename=os.path.join(FIG_DIR, "expected_profit_vs_cvar.png"),
        title="Task 1.4 Expected Profit versus CVaR",
    )

    selected_betas = [0.0, 0.5, 0.8, 1.0]

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
            ]["profit_eur"].to_numpy(float)

            plot_profit_distribution(
                profits=profits,
                filename=os.path.join(
                    FIG_DIR,
                    f"profit_distribution_{scheme_name}_beta_{beta:.2f}.png",
                ),
                title=f"Task 1.4 Profit Distribution - {scheme}, beta={beta:.2f}",
            )


def main() -> None:
    """
    Run the complete Task 1.4 risk-averse offering workflow.

    The workflow solves the one-price and two-price CVaR offering models for a
    grid of beta values in [0, 1], evaluates the resulting offers across all
    scenarios, saves result tables, creates figures, and runs a small
    sensitivity analysis with altered in-sample scenario subsets.

    Returns
    -------
    None
        The function writes outputs to the Task 1.4 table and figure folders.
    """

    make_output_dirs()

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
        wind_scenario_file="Data/scen_zone2.csv",
        price_file="Data/DayAheadPrices.csv",
        n_wind_scenarios=20,
        n_price_scenarios=20,
        n_imbalance_scenarios=4,
        deficit_probability=0.5,
        seed=42,
        price_area="DK2",
    )

    print(f"Total combined scenarios: {len(combined.scenarios)}")

    results_df, offers_df, profits_df, model_stats_df = run_beta_grid(
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

    model_stats_df.to_csv(
        os.path.join(TAB_DIR, "cvar_model_stats.csv"),
        index=False,
    )

    # Sensitivity to altered in-sample scenarios.
    # Keep this smaller than the full beta grid to avoid excessive run time.
    sensitivity_betas = [0.0, 0.5, 0.8, 1.0]

    sensitivity_df = run_in_sample_sensitivity(
        data=data,
        combined=combined,
        beta_values=sensitivity_betas,
        alpha=alpha,
        n_in_sample=200,
        n_repetitions=5,
        seed=123,
    )

    sensitivity_summary_df = build_in_sample_sensitivity_summary(
        sensitivity_df
    )

    sensitivity_df.to_csv(
        os.path.join(TAB_DIR, "in_sample_sensitivity_results.csv"),
        index=False,
    )

    sensitivity_summary_df.to_csv(
        os.path.join(TAB_DIR, "in_sample_sensitivity_summary.csv"),
        index=False,
    )

    save_selected_plots(
        results_df=results_df,
        offers_df=offers_df,
        profits_df=profits_df,
    )

    print("\nSaved tables:")
    print(f" - {TAB_DIR}/cvar_results.csv")
    print(f" - {TAB_DIR}/cvar_offers.csv")
    print(f" - {TAB_DIR}/cvar_scenario_profits.csv")
    print(f" - {TAB_DIR}/cvar_model_stats.csv")
    print(f" - {TAB_DIR}/in_sample_sensitivity_results.csv")
    print(f" - {TAB_DIR}/in_sample_sensitivity_summary.csv")

    print("\nSaved figures:")
    print(f" - {FIG_DIR}/expected_profit_vs_cvar.png")
    print(f" - {FIG_DIR}/offer_<scheme>_beta_<beta>.png")
    print(f" - {FIG_DIR}/profit_distribution_<scheme>_beta_<beta>.png")

    if (
        "both_delta_variables_active_count" in model_stats_df.columns
        and model_stats_df["both_delta_variables_active_count"].fillna(0).sum() > 0
    ):
        print(
            "\nWarning: Some two-price model runs have both positive and "
            "negative imbalance variables active in at least one scenario-hour. "
            "Inspect cvar_model_stats.csv and the selected price scenarios."
        )


if __name__ == "__main__":
    main()