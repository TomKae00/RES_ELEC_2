# analyze_scenarios.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import (
    HOURS,
    CAPACITY_MW,
    ensure_output_folders,
    prepare_scenario_data,
    prepare_volatile_test_scenario_data)


# ============================================================
# General helpers
# ============================================================
def save_summary_statistics(values, filename, label):
    """
    Saves descriptive statistics for a 2D array:
        rows = scenarios
        columns = hours
    """

    values = np.asarray(values)

    hourly_stats = pd.DataFrame({
        "hour": HOURS,
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "min": np.min(values, axis=0),
        "p05": np.quantile(values, 0.05, axis=0),
        "p10": np.quantile(values, 0.10, axis=0),
        "p25": np.quantile(values, 0.25, axis=0),
        "p50": np.quantile(values, 0.50, axis=0),
        "p75": np.quantile(values, 0.75, axis=0),
        "p90": np.quantile(values, 0.90, axis=0),
        "p95": np.quantile(values, 0.95, axis=0),
        "max": np.max(values, axis=0),
    })

    all_values = values.flatten()

    overall_stats = pd.DataFrame([{
        "variable": label,
        "mean": np.mean(all_values),
        "std": np.std(all_values),
        "min": np.min(all_values),
        "p05": np.quantile(all_values, 0.05),
        "p10": np.quantile(all_values, 0.10),
        "p25": np.quantile(all_values, 0.25),
        "p50": np.quantile(all_values, 0.50),
        "p75": np.quantile(all_values, 0.75),
        "p90": np.quantile(all_values, 0.90),
        "p95": np.quantile(all_values, 0.95),
        "max": np.max(all_values),
    }])

    hourly_stats.to_csv(filename.replace(".csv", "_hourly.csv"), index=False)
    overall_stats.to_csv(filename.replace(".csv", "_overall.csv"), index=False)

    return hourly_stats, overall_stats


def dict_to_matrix(scenario_dict):
    """
    Converts dictionary {scenario_id: 24-hour array}
    into matrix with shape:
        n_scenarios x 24
    """

    scenario_ids = sorted(scenario_dict.keys())
    matrix = np.vstack([scenario_dict[s] for s in scenario_ids])

    return scenario_ids, matrix


# ============================================================
# Plot helpers
# ============================================================
def plot_all_scenarios(matrix, filename, title, ylabel):
    """
    Plots all scenarios as 24-hour trajectories.
    """

    hours = np.array(HOURS)

    plt.figure(figsize=(10, 6))

    for row in matrix:
        plt.plot(hours, row, marker="o", linewidth=1.0, alpha=0.55)

    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel(ylabel)
    plt.xticks(hours)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_mean_with_quantiles(matrix, filename, title, ylabel):
    """
    Plots mean, median and quantile bands.
    """

    hours = np.array(HOURS)

    mean = np.mean(matrix, axis=0)
    p05 = np.quantile(matrix, 0.05, axis=0)
    p25 = np.quantile(matrix, 0.25, axis=0)
    p50 = np.quantile(matrix, 0.50, axis=0)
    p75 = np.quantile(matrix, 0.75, axis=0)
    p95 = np.quantile(matrix, 0.95, axis=0)

    plt.figure(figsize=(10, 6))

    plt.fill_between(hours, p05, p95, alpha=0.20, label="5th-95th percentile")
    plt.fill_between(hours, p25, p75, alpha=0.35, label="25th-75th percentile")
    plt.plot(hours, mean, marker="o", linewidth=2.0, label="Mean")
    plt.plot(hours, p50, marker="s", linewidth=2.0, linestyle="--", label="Median")

    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel(ylabel)
    plt.xticks(hours)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_distribution(values, filename, title, xlabel, bins=35):
    """
    Plots distribution of all hourly values across all scenarios.
    """

    values = np.asarray(values).flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.8)
    plt.axvline(
        np.mean(values),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(values):.2f}",
    )
    plt.axvline(
        np.median(values),
        linestyle=":",
        linewidth=1.5,
        label=f"Median: {np.median(values):.2f}",
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_boxplot_by_hour(matrix, filename, title, ylabel):
    """
    Boxplot for scenario values by hour.
    """

    plt.figure(figsize=(11, 6))
    plt.boxplot(
        [matrix[:, t] for t in HOURS],
        labels=[str(t) for t in HOURS],
        showfliers=True,
    )

    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_imbalance_heatmap(imbalance_matrix, filename):
    """
    Plots binary imbalance scenarios.
    SI = 1 deficit, SI = 0 surplus.
    """

    plt.figure(figsize=(10, 4))
    plt.imshow(imbalance_matrix, aspect="auto", interpolation="nearest")

    plt.title("System Imbalance Scenarios")
    plt.xlabel("Hour")
    plt.ylabel("Imbalance scenario")
    plt.xticks(HOURS)
    plt.yticks(range(imbalance_matrix.shape[0]))
    plt.colorbar(label="SI, 1 = deficit, 0 = surplus")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_correlation_heatmap(matrix, filename, title):
    """
    Hour-to-hour correlation heatmap.
    Useful for checking whether the scenarios preserve daily structure.
    """

    corr = np.corrcoef(matrix.T)

    plt.figure(figsize=(8, 7))
    plt.imshow(corr, vmin=-1, vmax=1, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel("Hour")
    plt.xticks(HOURS)
    plt.yticks(HOURS)
    plt.colorbar(label="Correlation")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Combined scenario analysis
# ============================================================
def build_combined_scenario_dataframe(data, combined):
    """
    Creates a long dataframe for all combined scenarios.
    One row per combined scenario and hour.
    """

    rows = []

    for scenario_index, sc in enumerate(combined.scenarios):
        w_s, p_s, i_s = sc

        wind = data.wind[w_s]
        price = data.price[p_s]
        imbalance = data.imbalance[i_s]
        balancing_price = data.balancing_price[(p_s, i_s)]

        for t in HOURS:
            rows.append({
                "combined_scenario": scenario_index,
                "wind_scenario": w_s,
                "price_scenario": p_s,
                "imbalance_scenario": i_s,
                "hour": t,
                "wind_MW": wind[t],
                "day_ahead_price_EUR_per_MWh": price[t],
                "system_imbalance": imbalance[t],
                "balancing_price_EUR_per_MWh": balancing_price[t],
                "probability": combined.probability[sc],
            })

    return pd.DataFrame(rows)


def summarize_combined_scenarios(combined_df):
    """
    Saves simple checks for the combined scenario set.
    """

    summary = pd.DataFrame([{
        "number_of_combined_hourly_rows": len(combined_df),
        "number_of_combined_scenarios": combined_df["combined_scenario"].nunique(),
        "number_of_wind_scenarios": combined_df["wind_scenario"].nunique(),
        "number_of_price_scenarios": combined_df["price_scenario"].nunique(),
        "number_of_imbalance_scenarios": combined_df["imbalance_scenario"].nunique(),
        "probability_per_combined_scenario": combined_df["probability"].iloc[0],
        "sum_probabilities": (
            combined_df
            .drop_duplicates("combined_scenario")["probability"]
            .sum()
        ),
        "average_deficit_share": combined_df["system_imbalance"].mean(),
        "average_surplus_share": 1.0 - combined_df["system_imbalance"].mean(),
        "mean_wind_MW": combined_df["wind_MW"].mean(),
        "mean_day_ahead_price": combined_df["day_ahead_price_EUR_per_MWh"].mean(),
        "mean_balancing_price": combined_df["balancing_price_EUR_per_MWh"].mean(),
    }])

    return summary

def print_sample_combined_scenarios(data, combined, n_samples=10):
    """
    Prints examples of the combined scenarios that are sent to the optimization model.

    Each combined scenario is a tuple:
        (wind_scenario_id, price_scenario_id, imbalance_scenario_id)

    The model receives, for each combined scenario:
        - 24 hourly wind production values
        - 24 hourly day-ahead prices
        - 24 hourly imbalance states
        - 24 hourly balancing prices
        - one scenario probability
    """

    print("\n" + "=" * 90)
    print("SAMPLE COMBINED SCENARIOS RECEIVED BY THE MODEL")
    print("=" * 90)

    scenarios_to_print = combined.scenarios[:n_samples]

    for scenario_index, sc in enumerate(scenarios_to_print):
        w_s, p_s, i_s = sc

        wind = data.wind[w_s]
        price = data.price[p_s]
        imbalance = data.imbalance[i_s]
        balancing_price = data.balancing_price[(p_s, i_s)]
        probability = combined.probability[sc]

        print(f"\nCombined scenario {scenario_index}")
        print("-" * 90)
        print(f"Scenario tuple:              {sc}")
        print(f"Wind scenario ID:            {w_s}")
        print(f"Price scenario ID:           {p_s}")
        print(f"Imbalance scenario ID:       {i_s}")
        print(f"Scenario probability:        {probability:.6f}")

        scenario_table = pd.DataFrame({
            "hour": HOURS,
            "wind_MW": wind,
            "day_ahead_price_EUR_per_MWh": price,
            "system_imbalance": imbalance,
            "balancing_price_EUR_per_MWh": balancing_price,
        })

        print(scenario_table.to_string(index=False))

    print("\n" + "=" * 90)

# ============================================================
# Main script
# ============================================================
def main():
    ensure_output_folders()

    fig_dir = "outputs/figures/scenario_analysis"
    table_dir = "outputs/tables/scenario_analysis"

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    wind_file = "Data/scen_zone2.csv"
    price_file = "Data/DayAheadPrices.csv"

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
    #    n_wind_scenarios=30,
    #    n_price_scenarios=30,
    #    n_imbalance_scenarios=8,
    #    deficit_probability=0.5,
    #    seed=42,
    #)

    print("Scenario set created:")
    print(f"Wind scenarios: {len(data.wind)}")
    print(f"Price scenarios: {len(data.price)}")
    print(f"Imbalance scenarios: {len(data.imbalance)}")
    print(f"Combined scenarios: {len(combined.scenarios)}")

    # --------------------------------------------------------
    # Convert dictionaries to matrices
    # --------------------------------------------------------
    wind_ids, wind_matrix = dict_to_matrix(data.wind)
    price_ids, price_matrix = dict_to_matrix(data.price)
    imbalance_ids, imbalance_matrix = dict_to_matrix(data.imbalance)

    balancing_matrix = np.vstack([
        values for _, values in sorted(data.balancing_price.items())
    ])

    # --------------------------------------------------------
    # Save raw scenario matrices
    # --------------------------------------------------------
    pd.DataFrame(
        wind_matrix,
        index=[f"wind_{i}" for i in wind_ids],
        columns=[f"h{h:02d}" for h in HOURS],
    ).to_csv(f"{table_dir}/scenario_analysis_wind_matrix.csv")

    pd.DataFrame(
        price_matrix,
        index=[f"price_{i}" for i in price_ids],
        columns=[f"h{h:02d}" for h in HOURS],
    ).to_csv(f"{table_dir}/scenario_analysis_price_matrix.csv")

    pd.DataFrame(
        imbalance_matrix,
        index=[f"imbalance_{i}" for i in imbalance_ids],
        columns=[f"h{h:02d}" for h in HOURS],
    ).to_csv(f"{table_dir}/scenario_analysis_imbalance_matrix.csv")

    pd.DataFrame(
        balancing_matrix,
        columns=[f"h{h:02d}" for h in HOURS],
    ).to_csv(f"{table_dir}/scenario_analysis_balancing_price_matrix.csv", index=False)

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------
    wind_hourly_stats, wind_overall_stats = save_summary_statistics(
        wind_matrix,
        f"{table_dir}/scenario_analysis_wind_stats.csv",
        label="wind_MW",
    )

    price_hourly_stats, price_overall_stats = save_summary_statistics(
        price_matrix,
        f"{table_dir}/scenario_analysis_price_stats.csv",
        label="day_ahead_price_EUR_per_MWh",
    )

    balancing_hourly_stats, balancing_overall_stats = save_summary_statistics(
        balancing_matrix,
        f"{table_dir}/scenario_analysis_balancing_price_stats.csv",
        label="balancing_price_EUR_per_MWh",
    )

    imbalance_hourly_stats = pd.DataFrame({
        "hour": HOURS,
        "deficit_probability": np.mean(imbalance_matrix, axis=0),
        "surplus_probability": 1.0 - np.mean(imbalance_matrix, axis=0),
    })

    imbalance_overall_stats = pd.DataFrame([{
        "average_deficit_probability": np.mean(imbalance_matrix),
        "average_surplus_probability": 1.0 - np.mean(imbalance_matrix),
    }])

    imbalance_hourly_stats.to_csv(
        f"{table_dir}/scenario_analysis_imbalance_hourly_stats.csv",
        index=False,
    )

    imbalance_overall_stats.to_csv(
        f"{table_dir}/scenario_analysis_imbalance_overall_stats.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Combined scenario dataframe
    # --------------------------------------------------------
    combined_df = build_combined_scenario_dataframe(data, combined)

    combined_df.to_csv(
        f"{table_dir}/scenario_analysis_combined_scenarios_long.csv",
        index=False,
    )

    combined_summary = summarize_combined_scenarios(combined_df)

    combined_summary.to_csv(
        f"{table_dir}/scenario_analysis_combined_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plots: wind
    # --------------------------------------------------------
    plot_all_scenarios(
        wind_matrix,
        f"{fig_dir}/scenario_analysis_wind_all_scenarios.png",
        "Wind Production Scenarios",
        "Wind production [MW]",
    )

    plot_mean_with_quantiles(
        wind_matrix,
        f"{fig_dir}/scenario_analysis_wind_mean_quantiles.png",
        "Wind Production Scenario Envelope",
        "Wind production [MW]",
    )

    plot_distribution(
        wind_matrix,
        f"{fig_dir}/scenario_analysis_wind_distribution.png",
        "Distribution of Wind Production Values",
        "Wind production [MW]",
    )

    plot_boxplot_by_hour(
        wind_matrix,
        f"{fig_dir}/scenario_analysis_wind_boxplot_by_hour.png",
        "Wind Production Distribution by Hour",
        "Wind production [MW]",
    )

    plot_correlation_heatmap(
        wind_matrix,
        f"{fig_dir}/scenario_analysis_wind_hourly_correlation.png",
        "Wind Production Hour-to-Hour Correlation",
    )

    # --------------------------------------------------------
    # Plots: day-ahead prices
    # --------------------------------------------------------
    plot_all_scenarios(
        price_matrix,
        f"{fig_dir}/scenario_analysis_price_all_scenarios.png",
        "Day-Ahead Price Scenarios",
        "Day-ahead price [EUR/MWh]",
    )

    plot_mean_with_quantiles(
        price_matrix,
        f"{fig_dir}/scenario_analysis_price_mean_quantiles.png",
        "Day-Ahead Price Scenario Envelope",
        "Day-ahead price [EUR/MWh]",
    )

    plot_distribution(
        price_matrix,
        f"{fig_dir}/scenario_analysis_price_distribution.png",
        "Distribution of Day-Ahead Price Values",
        "Day-ahead price [EUR/MWh]",
    )

    plot_boxplot_by_hour(
        price_matrix,
        f"{fig_dir}/scenario_analysis_price_boxplot_by_hour.png",
        "Day-Ahead Price Distribution by Hour",
        "Day-ahead price [EUR/MWh]",
    )

    plot_correlation_heatmap(
        price_matrix,
        f"{fig_dir}/scenario_analysis_price_hourly_correlation.png",
        "Day-Ahead Price Hour-to-Hour Correlation",
    )

    # --------------------------------------------------------
    # Plots: imbalance
    # --------------------------------------------------------
    plot_imbalance_heatmap(
        imbalance_matrix,
        f"{fig_dir}/scenario_analysis_imbalance_heatmap.png",
    )

    plot_distribution(
        imbalance_matrix,
        f"{fig_dir}/scenario_analysis_imbalance_distribution.png",
        "Distribution of System Imbalance States",
        "System imbalance, 1 = deficit, 0 = surplus",
        bins=2,
    )

    # --------------------------------------------------------
    # Plots: balancing prices
    # --------------------------------------------------------
    plot_all_scenarios(
        balancing_matrix,
        f"{fig_dir}/scenario_analysis_balancing_price_all_scenarios.png",
        "Balancing Price Scenarios",
        "Balancing price [EUR/MWh]",
    )

    plot_mean_with_quantiles(
        balancing_matrix,
        f"{fig_dir}/scenario_analysis_balancing_price_mean_quantiles.png",
        "Balancing Price Scenario Envelope",
        "Balancing price [EUR/MWh]",
    )

    plot_distribution(
        balancing_matrix,
        f"{fig_dir}/scenario_analysis_balancing_price_distribution.png",
        "Distribution of Balancing Price Values",
        "Balancing price [EUR/MWh]",
    )

    plot_boxplot_by_hour(
        balancing_matrix,
        f"{fig_dir}/scenario_analysis_balancing_price_boxplot_by_hour.png",
        "Balancing Price Distribution by Hour",
        "Balancing price [EUR/MWh]",
    )

    # --------------------------------------------------------
    # Print useful summaries
    # --------------------------------------------------------
    print("\nCombined scenario summary:")
    print(combined_summary)

    print("\nWind overall statistics:")
    print(wind_overall_stats)

    print("\nDay-ahead price overall statistics:")
    print(price_overall_stats)

    print("\nBalancing price overall statistics:")
    print(balancing_overall_stats)

    print("\nImbalance overall statistics:")
    print(imbalance_overall_stats)

    print("\nSaved tables in:")
    print(f" - {table_dir}/")

    print("\nSaved figures in:")
    print(f" - {fig_dir}/")

    print_sample_combined_scenarios(
        data=data,
        combined=combined,
        n_samples=10,
    )



if __name__ == "__main__":
    main()