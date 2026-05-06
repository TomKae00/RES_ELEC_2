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
    title: str,
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

    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel("Day-ahead offer [MW]")
    plt.xticks(hours)
    plt.ylim(0, capacity_mw * 1.10)
    plt.grid(True, alpha=0.3)
    plt.legend()
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
    title: str
) -> None:
    expected_profit = np.mean(profits)

    plt.figure(figsize=(8, 5))
    plt.hist(profits, bins=35, edgecolor="black", alpha=0.8)
    plt.axvline(
        expected_profit,
        linestyle="--",
        linewidth=1.5,
        label=f"Expected profit: {expected_profit:,.0f} EUR"
    )

    plt.title(title)
    plt.xlabel("Scenario profit [EUR]")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend()
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
    title: str
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

    plt.title(title)
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
    title: str = "Comparison of Optimal Hourly Offers"
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

    plt.title(title)
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
def plot_profit_vs_cvar(results_df, filename=None, title=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Split by scheme
    df_one = results_df[results_df["scheme"] == "one-price"]
    df_two = results_df[results_df["scheme"] == "two-price"]

    # Sort by beta
    df_one = df_one.sort_values("beta")
    df_two = df_two.sort_values("beta")

    # Plot
    ax.plot(df_one["cvar_profit"], df_one["expected_profit"], '-o', label="One-price")
    ax.plot(df_two["cvar_profit"], df_two["expected_profit"], '-s', label="Two-price")

    # Scatter for visibility
    ax.scatter(df_one["cvar_profit"], df_one["expected_profit"])
    ax.scatter(df_two["cvar_profit"], df_two["expected_profit"])

    # Annotate beta values
    for _, row in df_one.iterrows():
        ax.annotate(f"β={row['beta']:.2f}",
                    (row["cvar_profit"], row["expected_profit"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8)

    for _, row in df_two.iterrows():
        ax.annotate(f"β={row['beta']:.2f}",
                    (row["cvar_profit"], row["expected_profit"]),
                    xytext=(5, -10),
                    textcoords="offset points",
                    fontsize=8)

    # Labels
    ax.set_xlabel("CVaR [EUR]")
    ax.set_ylabel("Expected Profit [EUR]")
    ax.set_title(title if title else "Expected Profit vs CVaR")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Tight zoom (important!)
    all_cvar = list(df_one["cvar_profit"]) + list(df_two["cvar_profit"])
    all_profit = list(df_one["expected_profit"]) + list(df_two["expected_profit"])

    ax.set_xlim(min(all_cvar) * 0.98, max(all_cvar) * 1.02)
    ax.set_ylim(min(all_profit) * 0.98, max(all_profit) * 1.02)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)

    

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

    plt.title(title)
    plt.xlabel("Scenario profit [EUR]")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_profit_cdf_by_beta(
    profits_by_beta: dict,
    filename: str,
    title: str
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    for beta, profits in profits_by_beta.items():
        profits = np.sort(np.array(profits))
        cdf = np.arange(1, len(profits) + 1) / len(profits)

        plt.step(profits, cdf, where="post", label=f"$\\beta={beta}$")

    plt.title(title)
    plt.xlabel("Scenario profit [EUR]")
    plt.ylabel("Cumulative probability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    