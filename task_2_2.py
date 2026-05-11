# task_2_2.py

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
    compute_shortfalls,
    evaluate_p90_requirement,
)

from plotting import (
    plot_task_2_2_out_sample_satisfaction,
    plot_shortfall_distribution,
)


TASK_2_OUTPUT_DIR = Path("outputs") / "task_2"
TASK_2_1_OUTPUT_DIR = TASK_2_OUTPUT_DIR / "task_2_1"
TASK_2_2_OUTPUT_DIR = TASK_2_OUTPUT_DIR / "task_2_2"

TASK_2_1_RESULTS_FILE = TASK_2_1_OUTPUT_DIR / "task_2_1_results.csv"


def main() -> None:
    """
    Run the complete Task 2.2 out-of-sample verification workflow.

    The workflow regenerates the same load-profile scenario set as in
    Task 2.1, evaluates the reserve bids from Task 2.1 on the out-of-sample
    profiles, checks the empirical P90 requirement, saves the resulting tables,
    and creates the out-of-sample verification figures.

    All outputs are saved in:

        outputs/task_2/task_2_2/

    Returns
    -------
    None
        The function writes output files and prints a compact summary to the
        terminal.
    """

    TASK_2_2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed = 42

    # Regenerate the same profiles as in Task 2.1.
    scenario_data = prepare_load_scenario_data(
        n_profiles=N_TOTAL_PROFILES,
        n_in_sample=N_IN_SAMPLE,
        seed=seed,
    )

    out_sample_reserve = scenario_data.out_sample_reserve

    print("Task 2.2 out-of-sample verification")
    print(f"Out-of-sample profiles: {out_sample_reserve.shape[0]}")
    print(f"Minutes per profile: {out_sample_reserve.shape[1]}")
    print(f"Total out-of-sample samples: {out_sample_reserve.size}")

    if not TASK_2_1_RESULTS_FILE.exists():
        raise FileNotFoundError(
            "Task 2.1 results file not found. Run task_2_1.py first. "
            f"Expected file: {TASK_2_1_RESULTS_FILE}"
        )

    task_2_1_results = pd.read_csv(TASK_2_1_RESULTS_FILE)

    # ------------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------------

    verification_file = (
        TASK_2_2_OUTPUT_DIR / "task_2_2_out_sample_verification.csv"
    )
    shortfalls_file = (
        TASK_2_2_OUTPUT_DIR / "task_2_2_out_sample_shortfalls.csv"
    )
    satisfaction_plot_file = (
        TASK_2_2_OUTPUT_DIR / "task_2_2_out_sample_satisfaction_rate.png"
    )

    generated_files = []

    # ------------------------------------------------------------------
    # Evaluate Task 2.1 reserve bids out-of-sample
    # ------------------------------------------------------------------

    verification_rows = []
    shortfall_columns = {}

    for _, row in task_2_1_results.iterrows():
        method = row["method"]
        reserve_bid_kw = float(row["reserve_bid_kw"])

        evaluation = evaluate_p90_requirement(
            reserve_bid_kw=reserve_bid_kw,
            reserve_availability=out_sample_reserve,
        )

        shortfalls = compute_shortfalls(
            reserve_bid_kw=reserve_bid_kw,
            reserve_availability=out_sample_reserve,
        )

        verification_rows.append(
            {
                "method": method,
                "reserve_bid_kw": reserve_bid_kw,
                **evaluation,
            }
        )

        method_key = method.lower().replace("-", "_")
        shortfall_columns[f"{method_key}_shortfall_kw"] = shortfalls.flatten()

        shortfall_plot_file = (
            TASK_2_2_OUTPUT_DIR
            / f"task_2_2_out_sample_shortfall_distribution_{method_key}.png"
        )

        plot_shortfall_distribution(
            shortfalls=shortfalls,
            filename=str(shortfall_plot_file),
        )

        generated_files.append(shortfall_plot_file)

    verification_df = pd.DataFrame(verification_rows)

    verification_df.to_csv(
        verification_file,
        index=False,
    )

    pd.DataFrame(shortfall_columns).to_csv(
        shortfalls_file,
        index=False,
    )

    generated_files.extend(
        [
            verification_file,
            shortfalls_file,
        ]
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    plot_task_2_2_out_sample_satisfaction(
        results_df=verification_df,
        filename=str(satisfaction_plot_file),
    )

    generated_files.append(satisfaction_plot_file)

    print("\nTask 2.2 results:")
    print(verification_df)

    print("\nFiles saved:")
    for file in generated_files:
        print(f" - {file}")


if __name__ == "__main__":
    main()