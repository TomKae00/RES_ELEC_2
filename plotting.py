# plotting.py

from __future__ import annotations

from typing import List

import numpy as np
import matplotlib.pyplot as plt

from helpers import CAPACITY_MW


# ============================================================
# Task 1.1 and Task 1.2
# Plot optimal day-ahead bidding strategy over the 24 hours
# ============================================================
def plot_hourly_offer(
    offer: np.ndarray,
    filename: str,
    title: str | None = None,
    capacity_mw: float = CAPACITY_MW
) -> None:
    hours = np.arange(24)

    plt.figure(figsize=(8, 5))
    plt.step(hours, offer, where="mid", marker="o", label="Optimal offer")
    plt.axhline(
        capacity_mw,
        linestyle="--",
        linewidth=1.2,
        label=f"Capacity: {capacity_mw:.0f} MW"
    )

    if title:
        plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel("Day-ahead offer [MW]")
    plt.xticks(hours)
    plt.ylim(0, capacity_mw * 1.10)
    plt.grid(True, alpha=0.3)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Task 1.1 and Task 1.2
# Plot profit distribution across all combined scenarios
# ============================================================
def plot_profit_distribution(
    profits: List[float],
    filename: str,
    title: str | None = None,
) -> None:
    expected_profit = np.mean(profits)

    plt.figure(figsize=(8, 5))
    if title:
        plt.title(title)

    plt.hist(profits, bins=35, edgecolor="black", alpha=0.8)
    plt.axvline(
        expected_profit,
        linestyle="--",
        linewidth=1.5,
        label=f"Expected profit: {expected_profit:,.0f} EUR"
    )

    plt.xlabel("Scenario profit [EUR]")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Task 1.1 and Task 1.2
# Plot realized profit for every scenario index
# ============================================================
def plot_profit_by_scenario(
    profits: List[float],
    filename: str,
) -> None:
    scenario_ids = np.arange(1, len(profits) + 1)
    expected_profit = np.mean(profits)

    plt.figure(figsize=(9, 5))
    plt.plot(
        scenario_ids,
        profits,
        marker="o",
        markersize=2,
        linewidth=0.8,
        label="Scenario profit"
    )
    plt.axhline(
        expected_profit,
        linestyle="--",
        linewidth=1.5,
        label=f"Expected profit: {expected_profit:,.0f} EUR"
    )

    plt.xlabel("Scenario index")
    plt.ylabel("Profit [EUR]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    # ============================================================
# Task 1.1 vs Task 1.2
# Compare optimal day-ahead offers from one-price and two-price models
# ============================================================
def plot_offer_comparison(
    offer_one_price: np.ndarray,
    offer_two_price: np.ndarray,
    filename: str,
) -> None:
    hours = np.arange(24)

    plt.figure(figsize=(8, 5))
    plt.step(hours, offer_one_price, where="mid", marker="o", label="One-price offer")
    plt.step(hours, offer_two_price, where="mid", marker="s", label="Two-price offer")
    plt.axhline(
        CAPACITY_MW,
        linestyle="--",
        linewidth=1.2,
        label=f"Capacity: {CAPACITY_MW:.0f} MW"
    )

    plt.xlabel("Hour")
    plt.ylabel("Day-ahead offer [MW]")
    plt.xticks(hours)
    plt.ylim(0, CAPACITY_MW * 1.10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ============================================================
# Task 1.4
# Plot expected profit versus CVaR for different beta values
# ============================================================
def plot_profit_vs_cvar_single_scheme(results_df, scheme, filename, title=None):
    import matplotlib.pyplot as plt

    df = results_df[results_df["scheme"] == scheme].copy()

    if df.empty:
        raise ValueError(f"No results found for scheme = {scheme}")

    df = df.sort_values("beta")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df["cvar_profit"],
        df["expected_profit"],
        "-o",
        label=scheme,
    )

    ax.scatter(
        df["cvar_profit"],
        df["expected_profit"],
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"β={row['beta']:.2f}",
            (row["cvar_profit"], row["expected_profit"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("CVaR [EUR]")
    ax.set_ylabel("Expected Profit [EUR]")
    #ax.set_title(title if title else f"Expected Profit vs CVaR - {scheme}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="")

    # Optional: tighter axis limits
    x = df["cvar_profit"].to_numpy()
    y = df["expected_profit"].to_numpy()

    if len(x) > 1:
        x_margin = max((x.max() - x.min()) * 0.08, 1.0)
        y_margin = max((y.max() - y.min()) * 0.08, 1.0)
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_ylim(y.min() - y_margin, y.max() + y_margin)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    

# ============================================================
# Task 1.4
# Plot selected profit distributions for risk-neutral and risk-averse cases
# ============================================================
def plot_risk_profit_distributions(
    profits_by_beta: dict,
    filename: str,
    title: str
) -> None:
    plt.figure(figsize=(8, 5))

    for beta, profits in profits_by_beta.items():
        plt.hist(
            profits,
            bins=35,
            alpha=0.45,
            label=f"$\\beta={beta}$"
        )


    plt.xlabel("Scenario profit [EUR]")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ============================================================
# Task 2.1
# Plot load profiles
# ============================================================
def plot_load_profiles(load_profiles, filename, title):
    import numpy as np
    import matplotlib.pyplot as plt

    minutes = np.arange(load_profiles.shape[1])

    plt.figure(figsize=(10, 6))

    for profile in load_profiles:
        plt.plot(minutes, profile, linewidth=0.8, alpha=0.45)

    plt.title(title)
    plt.xlabel("Minute")
    plt.ylabel("Load consumption [kW]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Task 2.1
# Reserve availability envelope
# ============================================================
def plot_reserve_availability_envelope(reserve_availability, filename, title):
    import numpy as np
    import matplotlib.pyplot as plt

    minutes = np.arange(reserve_availability.shape[1])

    mean = np.mean(reserve_availability, axis=0)
    p05 = np.quantile(reserve_availability, 0.05, axis=0)
    p10 = np.quantile(reserve_availability, 0.10, axis=0)
    p50 = np.quantile(reserve_availability, 0.50, axis=0)
    p90 = np.quantile(reserve_availability, 0.90, axis=0)
    p95 = np.quantile(reserve_availability, 0.95, axis=0)

    plt.figure(figsize=(10, 6))

    plt.fill_between(minutes, p05, p95, alpha=0.20, label="5th-95th percentile")
    plt.fill_between(minutes, p10, p90, alpha=0.30, label="10th-90th percentile")
    plt.plot(minutes, mean, linewidth=2.0, label="Mean")
    plt.plot(minutes, p50, linestyle="--", linewidth=2.0, label="Median")

    plt.title(title)
    plt.xlabel("Minute")
    plt.ylabel("Available FCR-D UP reserve [kW]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Task 2.1
# Reserve bid comparison
# ============================================================
def plot_task_2_1_bid_comparison(results_df, filename):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))

    plt.bar(
        results_df["method"],
        results_df["reserve_bid_kw"],
        edgecolor="black",
    )

    plt.title("Task 2.1 Optimal FCR-D UP Reserve Bid")
    plt.xlabel("Method")
    plt.ylabel("Reserve bid [kW]")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Task 2.1
# Shortfall distribution
# ============================================================
def plot_shortfall_distribution(shortfalls, filename, title):
    import numpy as np
    import matplotlib.pyplot as plt

    shortfalls = np.asarray(shortfalls).flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(shortfalls, bins=35, edgecolor="black", alpha=0.8)

    plt.axvline(
        np.mean(shortfalls),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean shortfall: {np.mean(shortfalls):.2f} kW",
    )

    plt.title(title)
    plt.xlabel("Reserve shortfall [kW]")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ============================================================
# Task 2.2
# Out-of-sample P90 verification
# ============================================================
def plot_task_2_2_out_sample_satisfaction(results_df, filename):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))

    plt.bar(
        results_df["method"],
        results_df["satisfaction_rate"],
        edgecolor="black",
    )

    plt.axhline(
        0.90,
        linestyle="--",
        linewidth=1.5,
        label="P90 requirement",
    )

    plt.title("Task 2.2 Out-of-Sample P90 Verification")
    plt.xlabel("Method")
    plt.ylabel("Satisfaction rate")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Task 2.3
# Combined dual-axis: reliability threshold vs reserve bid and expected shortfall
# ============================================================
def plot_task_2_3_tradeoff(results_df, filename):
    dark_blue = "#1a3a6b"
    light_blue = "#6baed6"
    dark_red = "#AE2819"
    light_red = "#e38664"

    thresholds_pct = results_df["reliability_threshold"].to_numpy() * 100

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Reliability threshold [%]")
    ax1.set_ylabel("Reserve bid [kW]", color=dark_blue)
    ax1.tick_params(axis="y", labelcolor=dark_blue)

    line1, = ax1.plot(
        thresholds_pct,
        results_df["reserve_bid_kw"],
        color=dark_blue,
        linestyle="-",
        marker="o",
        label="Reserve bid – in-sample",
    )
    line2, = ax1.plot(
        thresholds_pct,
        results_df["reserve_bid_kw"],
        color=light_blue,
        linestyle="--",
        marker="o",
        label="Reserve bid – out-of-sample",
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Expected shortfall [kW]", color=dark_red)
    ax2.tick_params(axis="y", labelcolor=dark_red)

    line3, = ax2.plot(
        thresholds_pct,
        results_df["in_sample_expected_shortfall_kw"],
        color=dark_red,
        linestyle="-",
        marker="s",
        label="Expected shortfall – in-sample",
    )
    line4, = ax2.plot(
        thresholds_pct,
        results_df["out_sample_expected_shortfall_kw"],
        color=light_red,
        linestyle="--",
        marker="s",
        label="Expected shortfall – out-of-sample",
    )

    ax1.axvline(90, linestyle="--", color="grey", linewidth=1.2, alpha=0.7)

    ax1.set_xticks([80, 85, 90, 95, 97.5, 99, 100])

    lines = [line1, line2, line3, line4]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")

    #ax1.set_title("Task 2.3 Reliability Threshold vs Reserve Bid and Expected Shortfall")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Task 2.3
# Reliability threshold vs reserve bid
# ============================================================
def plot_task_2_3_threshold_vs_bid(results_df, filename):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["reliability_threshold"],
        results_df["reserve_bid_kw"],
        marker="o",
    )

    #plt.title("Task 2.3 Reliability Requirement versus Reserve Bid")
    plt.xlabel("Reliability threshold")
    plt.ylabel("Optimal reserve bid [kW]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Task 2.3
# Reliability threshold vs out-of-sample shortfall
# ============================================================
def plot_task_2_3_threshold_vs_shortfall(results_df, filename):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["reliability_threshold"],
        results_df["out_sample_expected_shortfall_kw"],
        marker="o",
    )

    plt.title("Task 2.3 Reliability Requirement versus Expected Shortfall")
    plt.xlabel("Reliability threshold")
    plt.ylabel("Out-of-sample expected shortfall [kW]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    