# task_2_3.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helpers2 import (
    N_IN_SAMPLE,
    N_TOTAL_PROFILES,
    prepare_load_scenario_data,
)

from task_2_1 import (
    solve_also_x,
    compute_shortfalls,
)

from plotting import (
    plot_task_2_3_threshold_vs_bid,
    plot_task_2_3_threshold_vs_shortfall,
    plot_task_2_3_tradeoff,
)


TASK_2_OUTPUT_DIR = Path("outputs") / "task_2"
TASK_2_3_OUTPUT_DIR = TASK_2_OUTPUT_DIR / "task_2_3"


def evaluate_requirement(
    reserve_bid_kw: float,
    reserve_availability: np.ndarray,
    reliability_threshold: float,
) -> dict:
    """
    Evaluate an empirical reserve reliability requirement.

    The reliability threshold defines the required share of profile-minute
    samples for which the available reserve must be at least as large as the
    reserve bid. For example, a reliability threshold of 0.90 corresponds to a
    P90 requirement.

    Parameters
    ----------
    reserve_bid_kw : float
        Reserve bid [kW].

    reserve_availability : np.ndarray
        Available reserve values [kW].

    reliability_threshold : float
        Required satisfaction probability.

    Returns
    -------
    dict
        Evaluation statistics for the reliability requirement.
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
        "requirement_met": bool(
            satisfaction_rate >= reliability_threshold - 1e-9
        ),
        "expected_shortfall_kw": float(np.mean(shortfalls)),
        "max_shortfall_kw": float(np.max(shortfalls)),
        "p90_shortfall_kw": float(np.quantile(shortfalls, 0.90)),
        "p95_shortfall_kw": float(np.quantile(shortfalls, 0.95)),
        "p99_shortfall_kw": float(np.quantile(shortfalls, 0.99)),
    }


def main() -> None:
    """
    Run the complete Task 2.3 reliability-threshold sensitivity workflow.

    The workflow solves the ALSO-X model for different reliability thresholds,
    evaluates each resulting reserve bid on both the in-sample and
    out-of-sample load profiles, and saves the resulting sensitivity table and
    figures.

    All outputs are saved in:

        outputs/task_2/task_2_3/

    Returns
    -------
    None
        The function writes output files and prints a compact summary to the
        terminal.
    """

    TASK_2_3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed = 42

    scenario_data = prepare_load_scenario_data(
        n_profiles=N_TOTAL_PROFILES,
        n_in_sample=N_IN_SAMPLE,
        seed=seed,
    )

    in_sample_reserve = scenario_data.in_sample_reserve
    out_sample_reserve = scenario_data.out_sample_reserve

    reliability_thresholds = np.array(
        [
            0.80,
            0.85,
            0.90,
            0.95,
            0.975,
            0.99,
            1.00,
        ]
    )

    results_file = (
        TASK_2_3_OUTPUT_DIR
        / "task_2_3_also_x_threshold_sensitivity.csv"
    )
    threshold_vs_bid_plot_file = (
        TASK_2_3_OUTPUT_DIR
        / "task_2_3_threshold_vs_reserve_bid.png"
    )
    threshold_vs_shortfall_plot_file = (
        TASK_2_3_OUTPUT_DIR
        / "task_2_3_threshold_vs_out_sample_expected_shortfall.png"
    )
    tradeoff_plot_file = (
        TASK_2_3_OUTPUT_DIR
        / "task_2_3_threshold_tradeoff_combined.png"
    )

    generated_files = []

    rows = []

    for reliability in reliability_thresholds:
        epsilon = 1.0 - reliability

        print(
            f"\nSolving ALSO-X for reliability threshold = "
            f"{reliability:.3f}"
        )

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

        rows.append(
            {
                "reliability_threshold": reliability,
                "epsilon": epsilon,
                "reserve_bid_kw": reserve_bid_kw,
                "in_sample_satisfaction_rate": in_eval[
                    "satisfaction_rate"
                ],
                "in_sample_violation_rate": in_eval["violation_rate"],
                "in_sample_expected_shortfall_kw": in_eval[
                    "expected_shortfall_kw"
                ],
                "in_sample_max_shortfall_kw": in_eval["max_shortfall_kw"],
                "in_sample_requirement_met": in_eval["requirement_met"],
                "out_sample_satisfaction_rate": out_eval[
                    "satisfaction_rate"
                ],
                "out_sample_violation_rate": out_eval["violation_rate"],
                "out_sample_expected_shortfall_kw": out_eval[
                    "expected_shortfall_kw"
                ],
                "out_sample_max_shortfall_kw": out_eval["max_shortfall_kw"],
                "out_sample_requirement_met": out_eval["requirement_met"],
            }
        )

        print(
            f"  reserve bid = {reserve_bid_kw:.2f} kW, "
            f"out-of-sample satisfaction = "
            f"{out_eval['satisfaction_rate']:.3f}, "
            f"out-of-sample expected shortfall = "
            f"{out_eval['expected_shortfall_kw']:.3f} kW"
        )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        results_file,
        index=False,
    )

    generated_files.append(results_file)

    plot_task_2_3_threshold_vs_bid(
        results_df=results_df,
        filename=str(threshold_vs_bid_plot_file),
    )

    plot_task_2_3_threshold_vs_shortfall(
        results_df=results_df,
        filename=str(threshold_vs_shortfall_plot_file),
    )

    plot_task_2_3_tradeoff(
        results_df=results_df,
        filename=str(tradeoff_plot_file),
    )

    generated_files.extend(
        [
            threshold_vs_bid_plot_file,
            threshold_vs_shortfall_plot_file,
            tradeoff_plot_file,
        ]
    )

    print("\nTask 2.3 results:")
    print(results_df)

    print("\nFiles saved:")
    for file in generated_files:
        print(f" - {file}")


if __name__ == "__main__":
    main()