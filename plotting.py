# plotting.py

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import CAPACITY_MW


# ============================================================
# Global plotting style
# ============================================================

ONE_PRICE_COLOR = "#1f77b4"
TWO_PRICE_COLOR = "#ff7f0e"
SPREAD_COLOR = "lightgrey"
MEAN_LINE_COLOR = "black"

FONT_SIZE = 14


def apply_report_style() -> None:
    """
    Apply common matplotlib font settings for report-ready figures.

    Returns
    -------
    None
        Updates matplotlib runtime configuration.
    """

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )


# ============================================================
# Task 1.1
# Expected price spread and optimal one-price offer
# ============================================================

def plot_expected_price_spread_with_offer(
    expected_spread: np.ndarray,
    offer: np.ndarray,
    filename: str,
    title: str | None = None,
) -> None:
    """
    Plot the expected day-ahead minus balancing price spread and the optimal offer.

    Parameters
    ----------
    expected_spread : np.ndarray
        Hourly expected spread E[lambda_DA - lambda_B] [EUR/MWh].

    offer : np.ndarray
        Optimal hourly day-ahead offer [MW].

    filename : str
        Path where the figure should be saved.

    title : str | None, default None
        Kept for compatibility, but not used for report-style figures.

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    expected_spread = np.asarray(expected_spread, dtype=float)
    offer = np.asarray(offer, dtype=float)

    hours = np.arange(len(expected_spread))
    capacity_mw = CAPACITY_MW

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(
        hours,
        expected_spread,
        width=0.75,
        alpha=0.8,
        color=SPREAD_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label=r"$\mathbb{E}[\lambda^{\mathrm{DA}}-\lambda^{\mathrm{B}}]$",
    )

    ax1.axhline(
        0.0,
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.5,
    )

    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Expected price spread [EUR/MWh]")
    ax1.set_xticks(hours)
    ax1.set_xlim(-0.5, len(hours) - 0.5)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()

    ax2.step(
        hours,
        offer,
        where="mid",
        linewidth=2.2,
        color=ONE_PRICE_COLOR,
        label="One-price offer",
    )

    ax2.scatter(
        hours,
        offer,
        s=25,
        color=ONE_PRICE_COLOR,
    )

    ax2.set_ylabel("Day-ahead offer [MW]")
    ax2.set_ylim(-0.05 * capacity_mw, 1.10 * capacity_mw)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Task 1.1
# One-price scenario profit distribution
# ============================================================

def plot_profit_distribution(
    profits: List[float] | np.ndarray,
    filename: str,
    title: str | None = None,
) -> None:
    """
    Plot the one-price scenario profit distribution and expected profit.

    Parameters
    ----------
    profits : list[float] | np.ndarray
        Scenario profit values [EUR].

    filename : str
        Path where the figure should be saved.

    title : str | None, default None
        Kept for compatibility, but not used for report-style figures.

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    profits = np.asarray(profits, dtype=float)
    expected_profit = float(np.mean(profits))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        profits,
        bins=35,
        color=ONE_PRICE_COLOR,
        edgecolor="black",
        alpha=0.4,
    )

    ax.axvline(
        expected_profit,
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.8,
        label=f"Expected profit: {expected_profit:,.0f} EUR",
    )

    ax.set_xlabel("Scenario profit [EUR]")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Task 1.2
# Profit distribution helpers
# ============================================================

def make_common_profit_bins(*profit_arrays: np.ndarray, n_bins: int = 35) -> np.ndarray:
    """
    Create common histogram bin edges for one or more profit arrays.

    Common bins ensure that individual and comparison histograms use the same
    grouping and can therefore be compared consistently.

    Parameters
    ----------
    profit_arrays : np.ndarray
        One or more arrays containing scenario profits [EUR].

    n_bins : int, default 35
        Number of bin edges.

    Returns
    -------
    np.ndarray
        Common histogram bin edges.
    """

    all_profits = np.concatenate(
        [np.asarray(profits, dtype=float) for profits in profit_arrays]
    )

    return np.linspace(
        float(np.min(all_profits)),
        float(np.max(all_profits)),
        n_bins,
    )


def plot_profit_distribution_with_bins(
    profits: np.ndarray,
    bins: np.ndarray,
    filename: str,
    title: str | None = None,
    label: str = "Scenario profit",
    color: str = TWO_PRICE_COLOR,
) -> None:
    """
    Plot a single profit distribution using predefined histogram bins.

    Parameters
    ----------
    profits : np.ndarray
        Scenario profit values [EUR].

    bins : np.ndarray
        Histogram bin edges.

    filename : str
        Path where the figure should be saved.

    title : str | None, default None
        Kept for compatibility, but not used for report-style figures.

    label : str, default "Scenario profit"
        Legend label for the histogram.

    color : str, default TWO_PRICE_COLOR
        Histogram color.

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    profits = np.asarray(profits, dtype=float)
    mean_profit = float(np.mean(profits))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        profits,
        bins=bins,
        color=color,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.8,
        label=label,
    )

    ax.axvline(
        mean_profit,
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"Expected profit: {mean_profit:,.0f} EUR",
    )

    ax.set_xlabel("Scenario profit [EUR]")
    ax.set_ylabel("Frequency")

    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=FONT_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_profit_distribution_comparison(
    profits_one_price: np.ndarray,
    profits_two_price: np.ndarray,
    filename: str,
    title: str | None = None,
) -> None:
    """
    Plot one-price and two-price scenario profit distributions in one figure.

    Parameters
    ----------
    profits_one_price : np.ndarray
        Scenario profits under the one-price scheme [EUR].

    profits_two_price : np.ndarray
        Scenario profits under the two-price scheme [EUR].

    filename : str
        Path where the figure should be saved.

    title : str | None, default None
        Kept for compatibility, but not used for report-style figures.

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    profits_one_price = np.asarray(profits_one_price, dtype=float)
    profits_two_price = np.asarray(profits_two_price, dtype=float)

    mean_one = float(np.mean(profits_one_price))
    mean_two = float(np.mean(profits_two_price))

    bins = make_common_profit_bins(
        profits_one_price,
        profits_two_price,
        n_bins=35,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        profits_one_price,
        bins=bins,
        color=ONE_PRICE_COLOR,
        alpha=0.40,
        edgecolor="black",
        linewidth=0.8,
        label="One-price",
    )
    ax.hist(
        profits_two_price,
        bins=bins,
        color=TWO_PRICE_COLOR,
        alpha=0.40,
        edgecolor="black",
        linewidth=0.8,
        label="Two-price",
    )

    ax.axvline(
        mean_one,
        color=ONE_PRICE_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"Mean one-price: {mean_one:,.0f} EUR",
    )
    ax.axvline(
        mean_two,
        color=TWO_PRICE_COLOR,
        linestyle=":",
        linewidth=2.5,
        label=f"Mean two-price: {mean_two:,.0f} EUR",
    )

    ax.set_xlabel("Scenario profit [EUR]")
    ax.set_ylabel("Frequency")

    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=FONT_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Task 1.1 vs Task 1.2
# Offer comparison
# ============================================================

def plot_offer_comparison(
    offer_one_price: np.ndarray,
    offer_two_price: np.ndarray,
    filename: str,
    title: str | None = None,
) -> None:
    """
    Plot the optimal hourly day-ahead offers under the one-price and two-price schemes.

    Parameters
    ----------
    offer_one_price : np.ndarray
        Optimal hourly day-ahead offer under the one-price scheme [MW].

    offer_two_price : np.ndarray
        Optimal hourly day-ahead offer under the two-price scheme [MW].

    filename : str
        Path where the figure should be saved.

    title : str | None, default None
        Kept for compatibility, but not used for report-style figures.

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    offer_one_price = np.asarray(offer_one_price, dtype=float)
    offer_two_price = np.asarray(offer_two_price, dtype=float)

    hours = np.arange(len(offer_one_price))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.step(
        hours,
        offer_one_price,
        where="mid",
        marker="o",
        linewidth=2,
        color=ONE_PRICE_COLOR,
        label="One-price offer",
    )

    ax.step(
        hours,
        offer_two_price,
        where="mid",
        marker="s",
        linewidth=2,
        color=TWO_PRICE_COLOR,
        label="Two-price offer",
    )

    ax.axhline(
        CAPACITY_MW,
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=f"Capacity: {CAPACITY_MW:.0f} MW",
    )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Day-ahead offer [MW]")

    ax.set_xticks(hours)
    ax.set_xlim(-0.5, len(hours) - 0.5)
    ax.set_ylim(0, CAPACITY_MW * 1.10)

    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=FONT_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Task 1.3
# Cross-validation plots
# ============================================================

def plot_k8_cross_validation_profits(
    results_df: pd.DataFrame,
    filename: str,
) -> None:
    """
    Plot in-sample and out-of-sample expected profits for the required K=8 case.

    The plot shows whether the fold-specific profits obtained on the scenarios
    used for optimization are close to the profits obtained on unseen scenarios.

    Parameters
    ----------
    results_df : pd.DataFrame
        Fold-level cross-validation results for all tested values of K.

    filename : str
        Path where the figure should be saved.

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    df = results_df[results_df["k_folds"] == 8].copy()

    if df.empty:
        raise ValueError("No K=8 results found in results_df.")

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_specs = {
        "one-price": {"marker": "o", "color": ONE_PRICE_COLOR},
        "two-price": {"marker": "s", "color": TWO_PRICE_COLOR},
    }

    for scheme, spec in plot_specs.items():
        scheme_df = df[df["scheme"] == scheme].sort_values("fold")

        ax.plot(
            scheme_df["fold"],
            scheme_df["in_sample_expected_profit_eur"],
            marker=spec["marker"],
            linestyle="-",
            linewidth=2,
            color=spec["color"],
            alpha=0.4,
            label=f"{scheme} in-sample",
        )

        ax.plot(
            scheme_df["fold"],
            scheme_df["out_sample_expected_profit_eur"],
            marker=spec["marker"],
            linestyle="--",
            linewidth=2,
            color=spec["color"],
            alpha=0.75,
            label=f"{scheme} out-of-sample",
        )

    ax.set_xlabel("Cross-validation fold")
    ax.set_ylabel("Expected profit [EUR]")

    ax.set_xticks(sorted(df["fold"].unique()))
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=FONT_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Legacy / optional diagnostic plots
# ============================================================

def plot_hourly_offer(
    offer: np.ndarray,
    filename: str,
    title: str | None = None,
    capacity_mw: float = CAPACITY_MW,
) -> None:
    """
    Plot a single optimal hourly day-ahead offer.

    This is mainly kept as an optional diagnostic plot. The report currently
    uses the combined comparison plots instead.

    Parameters
    ----------
    offer : np.ndarray
        Hourly day-ahead offer [MW].

    filename : str
        Path where the figure should be saved.

    title : str | None, default None
        Kept for compatibility, but not used for report-style figures.

    capacity_mw : float, default CAPACITY_MW
        Installed wind capacity [MW].

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    offer = np.asarray(offer, dtype=float)
    hours = np.arange(len(offer))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.step(
        hours,
        offer,
        where="mid",
        marker="o",
        linewidth=2,
        label="Optimal offer",
    )

    ax.axhline(
        capacity_mw,
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=f"Capacity: {capacity_mw:.0f} MW",
    )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Day-ahead offer [MW]")
    ax.set_xticks(hours)
    ax.set_xlim(-0.5, len(hours) - 0.5)
    ax.set_ylim(0, capacity_mw * 1.10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_profit_by_scenario(
    profits: List[float] | np.ndarray,
    filename: str,
    title: str | None = None,
) -> None:
    """
    Plot realized profit for every scenario index.

    This is mainly kept as an optional diagnostic plot. The report currently
    uses the profit-distribution plots instead.

    Parameters
    ----------
    profits : list[float] | np.ndarray
        Scenario profit values [EUR].

    filename : str
        Path where the figure should be saved.

    title : str | None, default None
        Kept for compatibility, but not used for report-style figures.

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    profits = np.asarray(profits, dtype=float)
    scenario_ids = np.arange(1, len(profits) + 1)
    expected_profit = float(np.mean(profits))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        scenario_ids,
        profits,
        marker="o",
        markersize=2,
        linewidth=0.8,
        label="Scenario profit",
    )

    ax.axhline(
        expected_profit,
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Expected profit: {expected_profit:,.0f} EUR",
    )

    ax.set_xlabel("Scenario index")
    ax.set_ylabel("Profit [EUR]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Task 1.4
# Expected profit versus CVaR
# ============================================================

def plot_profit_vs_cvar(results_df, filename=None, title=None):
    """
    Plot expected profit against CVaR for the one-price and two-price schemes.

    Supports both old result-column names:
        expected_profit, cvar_profit

    and newer explicit result-column names:
        expected_profit_eur, cvar_profit_eur
    """

    apply_report_style()

    df = results_df.copy()

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

    df_one = df[df["scheme"] == "one-price"].sort_values("beta")
    df_two = df[df["scheme"] == "two-price"].sort_values("beta")

    if not df_one.empty:
        ax.plot(
            df_one[cvar_col],
            df_one[expected_profit_col],
            marker="o",
            linestyle="-",
            linewidth=2,
            color=ONE_PRICE_COLOR,
            label="One-price",
        )

    if not df_two.empty:
        ax.plot(
            df_two[cvar_col],
            df_two[expected_profit_col],
            marker="s",
            linestyle="-",
            linewidth=2,
            color=TWO_PRICE_COLOR,
            label="Two-price",
        )

    for _, row in df_one.iterrows():
        ax.annotate(
            rf"$\beta={row['beta']:.2f}$",
            (row[cvar_col], row[expected_profit_col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=FONT_SIZE - 3,
        )

    for _, row in df_two.iterrows():
        ax.annotate(
            rf"$\beta={row['beta']:.2f}$",
            (row[cvar_col], row[expected_profit_col]),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=FONT_SIZE - 3,
        )

    ax.set_xlabel("CVaR [EUR]")
    ax.set_ylabel("Expected profit [EUR]")

    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

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

    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_profit_vs_cvar_single_scheme(results_df, scheme, filename, title=None):
    """
    Plot expected profit against CVaR for one settlement scheme.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results containing at least ``scheme``, ``beta``, ``cvar_profit`` and
        ``expected_profit`` columns.

    scheme : str
        Settlement scheme to plot, usually ``one-price`` or ``two-price``.

    filename : str
        Path where the figure should be saved.

    title : str | None, default None
        Kept for compatibility, but not used for report-style figures.

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    scheme_colors = {
        "one-price": ONE_PRICE_COLOR,
        "two-price": TWO_PRICE_COLOR,
    }

    color = scheme_colors.get(scheme, ONE_PRICE_COLOR)

    df = results_df[results_df["scheme"] == scheme].copy()

    if df.empty:
        raise ValueError(f"No results found for scheme = {scheme}")

    df = df.sort_values("beta")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["cvar_profit"],
        df["expected_profit"],
        marker="o",
        linestyle="-",
        linewidth=2,
        color=color,
        alpha=0.75,
        label=scheme,
    )

    for _, row in df.iterrows():
        ax.annotate(
            rf"$\beta={row['beta']:.2f}$",
            (row["cvar_profit"], row["expected_profit"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=FONT_SIZE - 3,
        )

    ax.set_xlabel("CVaR [EUR]")
    ax.set_ylabel("Expected profit [EUR]")

    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    x = df["cvar_profit"].to_numpy()
    y = df["expected_profit"].to_numpy()

    if len(x) > 1:
        x_margin = max((x.max() - x.min()) * 0.08, 1.0)
        y_margin = max((y.max() - y.min()) * 0.08, 1.0)
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_ylim(y.min() - y_margin, y.max() + y_margin)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Task 1.4
# Risk-related profit distributions
# ============================================================

def plot_risk_profit_distributions(
    profits_by_beta: dict,
    filename: str,
    title: str | None = None,
) -> None:
    """
    Plot selected profit distributions for different risk-aversion parameters.

    Parameters
    ----------
    profits_by_beta : dict
        Mapping from beta value to scenario profit array.

    filename : str
        Path where the figure should be saved.

    title : str | None, default None
        Kept for compatibility, but not used for report-style figures.

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for beta, profits in profits_by_beta.items():
        ax.hist(
            profits,
            bins=35,
            alpha=0.45,
            label=rf"$\beta={beta}$",
        )

    ax.set_xlabel("Scenario profit [EUR]")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_profit_cdf_by_beta(
    profits_by_beta: dict,
    filename: str,
    title: str | None = None,
) -> None:
    """
    Plot empirical profit CDFs for different risk-aversion parameters.

    Parameters
    ----------
    profits_by_beta : dict
        Mapping from beta value to scenario profit array.

    filename : str
        Path where the figure should be saved.

    title : str | None, default None
        Kept for compatibility, but not used for report-style figures.

    Returns
    -------
    None
        The function saves a figure and does not return an object.
    """

    apply_report_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for beta, profits in profits_by_beta.items():
        profits = np.sort(np.asarray(profits, dtype=float))
        cdf = np.arange(1, len(profits) + 1) / len(profits)

        ax.step(
            profits,
            cdf,
            where="post",
            label=rf"$\beta={beta}$",
        )

    ax.set_xlabel("Scenario profit [EUR]")
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)