# task_2_1.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from helpers2 import (
    N_MINUTES,
    MIN_LOAD_KW,
    MAX_LOAD_KW,
    MAX_RAMP_KW_PER_MIN,
    N_TOTAL_PROFILES,
    N_IN_SAMPLE,
    N_OUT_SAMPLE,
    ensure_step2_output_folders,
    prepare_load_scenario_data,
    check_load_profiles,
    save_load_profiles_to_csv,
    save_reserve_availability_to_csv,
)

from plotting import (
    plot_load_profiles,
    plot_reserve_availability_envelope,
    plot_task_2_1_bid_comparison,
    plot_shortfall_distribution,
)


FIG_DIR = "outputs/figures/task_2_1"
TAB_DIR = "outputs/tables/task_2_1"


# ============================================================
# Utility functions
# ============================================================
def flatten_samples(matrix: np.ndarray) -> np.ndarray:
    """
    Converts profile x minute matrix into one vector of samples.
    """

    return np.asarray(matrix, dtype=float).flatten()


def compute_shortfalls(reserve_bid_kw: float, reserve_availability: np.ndarray) -> np.ndarray:
    """
    Shortfall is positive if the bid exceeds available reserve.

        shortfall = max(reserve_bid - available reserve, 0)
    """

    return np.maximum(reserve_bid_kw - reserve_availability, 0.0)


def evaluate_p90_requirement(reserve_bid_kw: float, reserve_availability: np.ndarray) -> dict:
    """
    Checks empirical P90 requirement.

    Requirement:
        At least 90% of minute-profile samples must have
        available reserve >= reserve bid.
    """

    shortfalls = compute_shortfalls(reserve_bid_kw, reserve_availability)

    total_samples = shortfalls.size
    violated_samples = int(np.sum(shortfalls > 1e-6))
    satisfied_samples = total_samples - violated_samples

    satisfaction_rate = satisfied_samples / total_samples
    violation_rate = violated_samples / total_samples

    return {
        "reserve_bid_kw": float(reserve_bid_kw),
        "total_samples": total_samples,
        "satisfied_samples": satisfied_samples,
        "violated_samples": violated_samples,
        "satisfaction_rate": satisfaction_rate,
        "violation_rate": violation_rate,
        "p90_requirement_met": bool(satisfaction_rate >= 0.90 - 1e-9),
        "expected_shortfall_kw": float(np.mean(shortfalls)),
        "max_shortfall_kw": float(np.max(shortfalls)),
        "p90_shortfall_kw": float(np.quantile(shortfalls, 0.90)),
        "p95_shortfall_kw": float(np.quantile(shortfalls, 0.95)),
        "p99_shortfall_kw": float(np.quantile(shortfalls, 0.99)),
    }


# ============================================================
# ALSO-X model
# ============================================================
def solve_also_x(
    reserve_availability: np.ndarray,
    epsilon: float = 0.10,
) -> dict:
    """
    ALSO-X formulation of the empirical P90 requirement.

    Decision:
        c_up = reserve bid [kW]

    Binary variable:
        y_i = 1 if sample i is allowed to violate the reserve requirement

    Model:
        maximize c_up

        c_up - F_i <= M y_i
        sum_i y_i <= floor(epsilon * N)
        y_i binary

    where:
        F_i is reserve availability in sample i.
    """

    samples = flatten_samples(reserve_availability)
    n_samples = len(samples)

    max_violations = int(np.floor(epsilon * n_samples))
   
    max_reserve = MAX_LOAD_KW - MIN_LOAD_KW
    big_m = max_reserve

    model = gp.Model("task_2_1_also_x")
    model.setParam("OutputFlag", 0)

    c_up = model.addVar(
        lb=0.0,
        ub=max_reserve,
        name="c_up",
    )

    y = model.addVars(
        range(n_samples),
        vtype=GRB.BINARY,
        name="y",
    )

    for i in range(n_samples):
        model.addConstr(
            c_up - samples[i] <= big_m * y[i],
            name=f"availability_{i}",
        )

    model.addConstr(
        gp.quicksum(y[i] for i in range(n_samples)) <= max_violations,
        name="p90_violation_limit",
    )

    model.setObjective(c_up, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"ALSO-X model ended with status {model.Status}")

    reserve_bid = c_up.X
    shortfalls = compute_shortfalls(reserve_bid, reserve_availability)

    return {
        "method": "ALSO-X",
        "reserve_bid_kw": reserve_bid,
        "objective_value": model.ObjVal,
        "max_allowed_violations": max_violations,
        "actual_violations": int(np.sum(shortfalls > 1e-6)),
        "shortfalls": shortfalls,
    }


# ============================================================
# CVaR model
# ============================================================
def solve_cvar_reserve_bid(
    reserve_availability: np.ndarray,
    epsilon: float = 0.10,
) -> dict:
    """
    CVaR approximation matching the Lecture 10 formulation.

    Model:

        max c_up

    subject to:

        c_up - F_i <= zeta_i              for all i

        (1/N) sum_i zeta_i <= (1-epsilon) * beta

        beta <= zeta_i                    for all i

        c_up >= 0

        beta <= 0

    where:
        F_i is available reserve in sample i.
    """

    samples = flatten_samples(reserve_availability)
    n_samples = len(samples)

    max_reserve = MAX_LOAD_KW - MIN_LOAD_KW

    model = gp.Model("task_2_1_cvar_lecture_formulation")
    model.setParam("OutputFlag", 0)

    # Reserve bid
    c_up = model.addVar(
        lb=0.0,
        ub=max_reserve,
        name="c_up",
    )

    # VaR-related auxiliary variable from the lecture formulation
    beta = model.addVar(
        lb=-GRB.INFINITY,
        ub=0.0,
        name="beta",
    )

    # Auxiliary variables zeta_i
    zeta = model.addVars(
        range(n_samples),
        lb=-GRB.INFINITY,
        name="zeta",
    )

    # c_up - F_i <= zeta_i
    for i in range(n_samples):
        model.addConstr(
            c_up - samples[i] <= zeta[i],
            name=f"reserve_shortfall_bound_{i}",
        )

    # beta <= zeta_i
    for i in range(n_samples):
        model.addConstr(
            beta <= zeta[i],
            name=f"beta_lower_bound_{i}",
        )

    # average zeta <= (1 - epsilon) * beta
    model.addConstr(
        (1.0 / n_samples) * gp.quicksum(zeta[i] for i in range(n_samples))
        <= (1.0 - epsilon) * beta,
        name="cvar_approximation_constraint",
    )

    model.setObjective(c_up, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"CVaR model ended with status {model.Status}")

    reserve_bid = c_up.X
    shortfalls = compute_shortfalls(reserve_bid, reserve_availability)

    return {
        "method": "CVaR",
        "reserve_bid_kw": reserve_bid,
        "objective_value": model.ObjVal,
        "beta_var": beta.X,
        "shortfalls": shortfalls,
    }

# ============================================================
# Main
# ============================================================
def main():
    ensure_step2_output_folders()

    epsilon = 0.10
    seed = 42

    scenario_data = prepare_load_scenario_data(
        n_profiles=N_TOTAL_PROFILES,
        n_in_sample=N_IN_SAMPLE,
        seed=seed,
    )

    print("Step 2 load profiles generated:")
    print(f"Total profiles: {scenario_data.load_profiles.shape[0]}")
    print(f"In-sample profiles: {scenario_data.in_sample_load.shape[0]}")
    print(f"Out-of-sample profiles: {scenario_data.out_sample_load.shape[0]}")
    print(f"Minutes per profile: {scenario_data.load_profiles.shape[1]}")

    # ------------------------------------------------------------
    # Check generated data
    # ------------------------------------------------------------
    profile_checks = check_load_profiles(scenario_data.load_profiles)

    print("\nLoad-profile checks:")
    for key, value in profile_checks.items():
        print(f"{key}: {value}")

    pd.DataFrame([profile_checks]).to_csv(
        os.path.join(TAB_DIR, "load_profile_checks.csv"),
        index=False,
    )

    # ------------------------------------------------------------
    # Save generated data
    # ------------------------------------------------------------
    save_load_profiles_to_csv(
        scenario_data.load_profiles,
        os.path.join(TAB_DIR, "all_load_profiles.csv"),
    )

    save_load_profiles_to_csv(
        scenario_data.in_sample_load,
        os.path.join(TAB_DIR, "in_sample_load_profiles.csv"),
    )

    save_load_profiles_to_csv(
        scenario_data.out_sample_load,
        os.path.join(TAB_DIR, "out_sample_load_profiles.csv"),
    )

    save_reserve_availability_to_csv(
        scenario_data.in_sample_reserve,
        os.path.join(TAB_DIR, "in_sample_reserve_availability.csv"),
    )

    # ------------------------------------------------------------
    # Solve Task 2.1 using in-sample profiles
    # ------------------------------------------------------------
    also_x_result = solve_also_x(
        reserve_availability=scenario_data.in_sample_reserve,
        epsilon=epsilon,
    )

    cvar_result = solve_cvar_reserve_bid(
        reserve_availability=scenario_data.in_sample_reserve,
        epsilon=epsilon,
    )

    # ------------------------------------------------------------
    # Evaluate in-sample P90 requirement
    # ------------------------------------------------------------
    also_x_eval = evaluate_p90_requirement(
        reserve_bid_kw=also_x_result["reserve_bid_kw"],
        reserve_availability=scenario_data.in_sample_reserve,
    )

    cvar_eval = evaluate_p90_requirement(
        reserve_bid_kw=cvar_result["reserve_bid_kw"],
        reserve_availability=scenario_data.in_sample_reserve,
    )

    results_df = pd.DataFrame([
        {
            "method": "ALSO-X",
            "reserve_bid_kw": also_x_result["reserve_bid_kw"],
            "objective_value": also_x_result["objective_value"],
            **also_x_eval,
        },
        {
            "method": "CVaR",
            "reserve_bid_kw": cvar_result["reserve_bid_kw"],
            "objective_value": cvar_result["objective_value"],
            "beta_var": cvar_result["beta_var"],
            **cvar_eval,
        },
    ])

    results_df.to_csv(
        os.path.join(TAB_DIR, "task_2_1_results.csv"),
        index=False,
    )

    print("\nTask 2.1 results:")
    print(results_df)

    # ------------------------------------------------------------
    # Save shortfalls
    # ------------------------------------------------------------
    also_x_shortfalls = compute_shortfalls(
        also_x_result["reserve_bid_kw"],
        scenario_data.in_sample_reserve,
    )

    cvar_shortfalls = compute_shortfalls(
        cvar_result["reserve_bid_kw"],
        scenario_data.in_sample_reserve,
    )

    pd.DataFrame({
        "also_x_shortfall_kw": also_x_shortfalls.flatten(),
        "cvar_shortfall_kw": cvar_shortfalls.flatten(),
    }).to_csv(
        os.path.join(TAB_DIR, "task_2_1_shortfalls_in_sample.csv"),
        index=False,
    )

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    plot_load_profiles(
        scenario_data.in_sample_load,
        os.path.join(FIG_DIR, "in_sample_load_profiles.png"),
        "Task 2.1 In-Sample Load Profiles",
    )

    plot_reserve_availability_envelope(
        scenario_data.in_sample_reserve,
        os.path.join(FIG_DIR, "in_sample_reserve_availability_envelope.png"),
        "Task 2.1 In-Sample Reserve Availability",
    )

    plot_task_2_1_bid_comparison(
        results_df,
        os.path.join(FIG_DIR, "reserve_bid_comparison.png"),
    )

    plot_shortfall_distribution(
        also_x_shortfalls,
        os.path.join(FIG_DIR, "shortfall_distribution_also_x.png"),
        "Task 2.1 In-Sample Shortfall Distribution - ALSO-X",
    )

    plot_shortfall_distribution(
        cvar_shortfalls,
        os.path.join(FIG_DIR, "shortfall_distribution_cvar.png"),
        "Task 2.1 In-Sample Shortfall Distribution - CVaR",
    )

    print("\nSaved tables:")
    print(f" - {TAB_DIR}/load_profile_checks.csv")
    print(f" - {TAB_DIR}/all_load_profiles.csv")
    print(f" - {TAB_DIR}/in_sample_load_profiles.csv")
    print(f" - {TAB_DIR}/out_sample_load_profiles.csv")
    print(f" - {TAB_DIR}/in_sample_reserve_availability.csv")
    print(f" - {TAB_DIR}/task_2_1_results.csv")
    print(f" - {TAB_DIR}/task_2_1_shortfalls_in_sample.csv")

    print("\nSaved figures:")
    print(f" - {FIG_DIR}/in_sample_load_profiles.png")
    print(f" - {FIG_DIR}/in_sample_reserve_availability_envelope.png")
    print(f" - {FIG_DIR}/reserve_bid_comparison.png")
    print(f" - {FIG_DIR}/shortfall_distribution_also_x.png")
    print(f" - {FIG_DIR}/shortfall_distribution_cvar.png")


if __name__ == "__main__":
    main()