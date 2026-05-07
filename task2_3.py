# task_2_3.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from helpers2 import (
    N_TOTAL_PROFILES,
    N_IN_SAMPLE,
    ensure_step2_output_folders,
    prepare_load_scenario_data,
)

from task2_1 import (
    solve_also_x,
    evaluate_p90_requirement,
    compute_shortfalls,
)

from plotting import (
    plot_task_2_3_threshold_vs_bid,
    plot_task_2_3_threshold_vs_shortfall,
    plot_task_2_3_tradeoff,
)


FIG_DIR = "outputs/figures/task_2_3"
TAB_DIR = "outputs/tables/task_2_3"


def make_output_dirs():
    ensure_step2_output_folders()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)


def evaluate_requirement(
    reserve_bid_kw: float,
    reserve_availability: np.ndarray,
    reliability_threshold: float,
) -> dict:
    """
    General version of the P90 evaluation.

    For example:
        reliability_threshold = 0.80 means P80
        reliability_threshold = 0.90 means P90
        reliability_threshold = 1.00 means P100
    """

    shortfalls = compute_shortfalls(
        reserve_bid_kw=reserve_bid_kw,
        reserve_availability=reserve_availability,
    )

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
        "requirement_met": bool(satisfaction_rate >= reliability_threshold - 1e-9),
        "expected_shortfall_kw": float(np.mean(shortfalls)),
        "max_shortfall_kw": float(np.max(shortfalls)),
        "p90_shortfall_kw": float(np.quantile(shortfalls, 0.90)),
        "p95_shortfall_kw": float(np.quantile(shortfalls, 0.95)),
        "p99_shortfall_kw": float(np.quantile(shortfalls, 0.99)),
    }


def main():
    make_output_dirs()

    seed = 42

    scenario_data = prepare_load_scenario_data(
        n_profiles=N_TOTAL_PROFILES,
        n_in_sample=N_IN_SAMPLE,
        seed=seed,
    )

    in_sample_reserve = scenario_data.in_sample_reserve
    out_sample_reserve = scenario_data.out_sample_reserve

    reliability_thresholds = np.array([
        0.80,
        0.85,
        0.90,
        0.95,
        0.975,
        0.99,
        1.00,
    ])

    rows = []

    for reliability in reliability_thresholds:
        epsilon = 1.0 - reliability

        print(f"\nSolving ALSO-X for reliability threshold = {reliability:.3f}")

        also_x_result = solve_also_x(
            reserve_availability=in_sample_reserve,
            epsilon=epsilon,
        )

        reserve_bid_kw = also_x_result["reserve_bid_kw"]

        in_eval = evaluate_requirement(
            reserve_bid_kw=reserve_bid_kw,
            reserve_availability=in_sample_reserve,
            reliability_threshold=reliability,
        )

        out_eval = evaluate_requirement(
            reserve_bid_kw=reserve_bid_kw,
            reserve_availability=out_sample_reserve,
            reliability_threshold=reliability,
        )

        rows.append({
            "reliability_threshold": reliability,
            "epsilon": epsilon,
            "reserve_bid_kw": reserve_bid_kw,

            "in_sample_satisfaction_rate": in_eval["satisfaction_rate"],
            "in_sample_violation_rate": in_eval["violation_rate"],
            "in_sample_expected_shortfall_kw": in_eval["expected_shortfall_kw"],
            "in_sample_max_shortfall_kw": in_eval["max_shortfall_kw"],
            "in_sample_requirement_met": in_eval["requirement_met"],

            "out_sample_satisfaction_rate": out_eval["satisfaction_rate"],
            "out_sample_violation_rate": out_eval["violation_rate"],
            "out_sample_expected_shortfall_kw": out_eval["expected_shortfall_kw"],
            "out_sample_max_shortfall_kw": out_eval["max_shortfall_kw"],
            "out_sample_requirement_met": out_eval["requirement_met"],
        })

        print(
            f"  reserve bid = {reserve_bid_kw:.2f} kW, "
            f"out-sample satisfaction = {out_eval['satisfaction_rate']:.3f}, "
            f"out-sample expected shortfall = {out_eval['expected_shortfall_kw']:.3f} kW"
        )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        os.path.join(TAB_DIR, "task_2_3_also_x_threshold_sensitivity.csv"),
        index=False,
    )

    plot_task_2_3_threshold_vs_bid(
        results_df,
        os.path.join(FIG_DIR, "threshold_vs_reserve_bid.png"),
    )

    plot_task_2_3_threshold_vs_shortfall(
        results_df,
        os.path.join(FIG_DIR, "threshold_vs_out_sample_expected_shortfall.png"),
    )

    plot_task_2_3_tradeoff(
        results_df,
        os.path.join(FIG_DIR, "threshold_tradeoff_combined.png"),
    )

    print("\nTask 2.3 results:")
    print(results_df)

    print("\nSaved tables:")
    print(f" - {TAB_DIR}/task_2_3_also_x_threshold_sensitivity.csv")

    print("\nSaved figures:")
    print(f" - {FIG_DIR}/threshold_vs_reserve_bid.png")
    print(f" - {FIG_DIR}/threshold_vs_out_sample_expected_shortfall.png")
    print(f" - {FIG_DIR}/threshold_tradeoff_combined.png")


if __name__ == "__main__":
    main()