# plotting.py

from __future__ import annotations

from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    """
    Plot expected profit against CVaR for the one-price and two-price schemes.

    Supports both old result-column names:
        expected_profit, cvar_profit

    and newer explicit result-column names:
        expected_profit_eur, cvar_profit_eur
    """

    import matplotlib.pyplot as plt

    df = results_df.copy()

    # Support both old and new column names
    if "expected_profit_eur" in df.columns:
        expected_profit_col = "expected_profit_eur"
    elif "expected_profit" in df.columns:
        expected_profit_col = "expected_profit"
    else:
        raise KeyError(
            "Could not find expected profit column. Expected either "
            "'expected_profit_eur' or 'expected_profit'."
        )

    if "cvar_profit_eur" in df.columns:
        cvar_col = "cvar_profit_eur"
    elif "cvar_profit" in df.columns:
        cvar_col = "cvar_profit"
    else:
        raise KeyError(
            "Could not find CVaR column. Expected either "
            "'cvar_profit_eur' or 'cvar_profit'."
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Split by scheme and sort by beta
    df_one = df[df["scheme"] == "one-price"].sort_values("beta")
    df_two = df[df["scheme"] == "two-price"].sort_values("beta")

    # Plot lines
    if not df_one.empty:
        ax.plot(
            df_one[cvar_col],
            df_one[expected_profit_col],
            "-o",
            label="One-price",
        )
        ax.scatter(df_one[cvar_col], df_one[expected_profit_col])

    if not df_two.empty:
        ax.plot(
            df_two[cvar_col],
            df_two[expected_profit_col],
            "-s",
            label="Two-price",
        )
        ax.scatter(df_two[cvar_col], df_two[expected_profit_col])

    # Annotate beta values
    for _, row in df_one.iterrows():
        ax.annotate(
            f"beta={row['beta']:.2f}",
            (row[cvar_col], row[expected_profit_col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    for _, row in df_two.iterrows():
        ax.annotate(
            f"beta={row['beta']:.2f}",
            (row[cvar_col], row[expected_profit_col]),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("CVaR [EUR]")
    ax.set_ylabel("Expected profit [EUR]")
    ax.set_title(title if title else "Expected Profit vs CVaR")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Tight zoom, only if data exists
    all_cvar = pd.concat([df_one[cvar_col], df_two[cvar_col]]).dropna()
    all_profit = pd.concat(
        [df_one[expected_profit_col], df_two[expected_profit_col]]
    ).dropna()

    if not all_cvar.empty and not all_profit.empty:
        cvar_min, cvar_max = all_cvar.min(), all_cvar.max()
        profit_min, profit_max = all_profit.min(), all_profit.max()

        cvar_margin = 0.02 * max(abs(cvar_max - cvar_min), 1.0)
        profit_margin = 0.02 * max(abs(profit_max - profit_min), 1.0)

        ax.set_xlim(cvar_min - cvar_margin, cvar_max + cvar_margin)
        ax.set_ylim(profit_min - profit_margin, profit_max + profit_margin)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)

    plt.close(fig)

    

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
    