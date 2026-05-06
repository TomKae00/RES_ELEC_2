# task_2_2.py

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
    compute_shortfalls,
    evaluate_p90_requirement,
)

from plotting import (
    plot_task_2_2_out_sample_satisfaction,
    plot_shortfall_distribution,
)


FIG_DIR = "outputs/figures/task_2_2"
TAB_DIR = "outputs/tables/task_2_2"

TASK_2_1_RESULTS_PATH = "outputs/tables/task_2_1/task_2_1_results.csv"


def make_output_dirs():
    ensure_step2_output_folders()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)


def main():
    make_output_dirs()

    seed = 42

    # Regenerate the same profiles as in Task 2.1
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

    # Read reserve bids from Task 2.1
    task_2_1_results = pd.read_csv(TASK_2_1_RESULTS_PATH)

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

        verification_rows.append({
            "method": method,
            "reserve_bid_kw": reserve_bid_kw,
            **evaluation,
        })

        shortfall_columns[f"{method}_shortfall_kw"] = shortfalls.flatten()

        plot_shortfall_distribution(
            shortfalls,
            os.path.join(
                FIG_DIR,
                f"out_sample_shortfall_distribution_{method.lower().replace('-', '_')}.png",
            ),
            f"Task 2.2 Out-of-Sample Shortfall Distribution - {method}",
        )

    verification_df = pd.DataFrame(verification_rows)

    verification_df.to_csv(
        os.path.join(TAB_DIR, "task_2_2_out_sample_verification.csv"),
        index=False,
    )

    pd.DataFrame(shortfall_columns).to_csv(
        os.path.join(TAB_DIR, "task_2_2_out_sample_shortfalls.csv"),
        index=False,
    )

    plot_task_2_2_out_sample_satisfaction(
        verification_df,
        os.path.join(FIG_DIR, "out_sample_satisfaction_rate.png"),
    )

    print("\nTask 2.2 results:")
    print(verification_df)

    print("\nSaved tables:")
    print(f" - {TAB_DIR}/task_2_2_out_sample_verification.csv")
    print(f" - {TAB_DIR}/task_2_2_out_sample_shortfalls.csv")

    print("\nSaved figures:")
    print(f" - {FIG_DIR}/out_sample_satisfaction_rate.png")
    print(f" - {FIG_DIR}/out_sample_shortfall_distribution_<method>.png")


if __name__ == "__main__":
    main()