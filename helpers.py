# helpers.py

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------------
# Constants
# --------------------------------
HOURS = list(range(24))
CAPACITY_MW = 500.0


# --------------------------------
# Data classes
# --------------------------------
@dataclass
class ScenarioData:
    wind: Dict[int, np.ndarray]
    price: Dict[int, np.ndarray]
    imbalance: Dict[int, np.ndarray]
    balancing_price: Dict[Tuple[int, int], np.ndarray]


@dataclass
class CombinedScenarioSet:
    scenarios: List[Tuple[int, int, int]]
    probability: Dict[Tuple[int, int, int], float]


# --------------------------------
# Folder helpers
# --------------------------------
def ensure_output_folders() -> None:
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


# --------------------------------
# Wind scenario file loader
# --------------------------------
def load_wind_scenarios_from_scenario_file(
    filepath: str,
    n_wind_scenarios: int = 30,
    n_hours: int = 24,
    capacity_mw: float = CAPACITY_MW
) -> Dict[int, np.ndarray]:
    """
    Reads a scenario file like scen_zone2.csv where:
    - columns V1, V2, ... are scenarios
    - rows are time steps
    - values are normalized wind production in [0,1]

    This function:
    - drops any END row
    - keeps the first n_hours rows
    - keeps the first n_wind_scenarios scenario columns
    - converts normalized values to MW
    """
    df = pd.read_csv(filepath)

    # Rename first unnamed column if needed
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "time_index"})

    # Remove rows like END
    df["time_index"] = df["time_index"].astype(str)
    df = df[df["time_index"].str.upper() != "END"].copy()

    # Convert time index to numeric where possible
    df["time_index_num"] = pd.to_numeric(df["time_index"], errors="coerce")
    df = df.dropna(subset=["time_index_num"]).copy()
    df = df.sort_values("time_index_num")

    # Keep first 24 rows only
    df = df.iloc[:n_hours].copy()

    scenario_cols = [c for c in df.columns if c.startswith("V")]

    if len(scenario_cols) < n_wind_scenarios:
        raise ValueError(
            f"Requested {n_wind_scenarios} wind scenarios, but only {len(scenario_cols)} are available."
        )

    scenario_cols = scenario_cols[:n_wind_scenarios]

    scenarios = {}
    for i, col in enumerate(scenario_cols):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        if len(values) != n_hours:
            raise ValueError(f"Scenario {col} does not contain {n_hours} rows.")

        # Convert per-unit to MW and clip to [0, capacity]
        values_mw = np.clip(values * capacity_mw, 0.0, capacity_mw)
        scenarios[i] = values_mw

    return scenarios


# --------------------------------
# Price cleaning
# --------------------------------
def read_price_15min_to_hourly_long(
    filepath: str,
    price_area: str = "DK2"
) -> pd.DataFrame:
    """
    Reads raw 15-min price data and converts to hourly average price.
    Returns long format:
    ['date', 'hour', 'price_EUR']
    """
    df = pd.read_csv(filepath, sep=";")

    df = df[df["PriceArea"] == price_area].copy()

    df["TimeDK"] = pd.to_datetime(df["TimeDK"], errors="coerce")

    # Convert decimal comma to decimal point
    df["DayAheadPriceEUR"] = (
        df["DayAheadPriceEUR"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df["DayAheadPriceEUR"] = pd.to_numeric(df["DayAheadPriceEUR"], errors="coerce")

    # Floor to hour and average quarter-hour observations
    df["hour_start"] = df["TimeDK"].dt.floor("h")

    hourly = (
        df.groupby("hour_start", as_index=False)["DayAheadPriceEUR"]
        .mean()
        .sort_values("hour_start")
        .rename(columns={"DayAheadPriceEUR": "price_EUR"})
    )

    hourly["date"] = hourly["hour_start"].dt.date
    hourly["hour"] = hourly["hour_start"].dt.hour

    return hourly[["date", "hour", "price_EUR"]]


def long_to_daily_matrix(df_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Converts long hourly data into:
    one row per day, columns h00 ... h23
    """
    pivot = df_long.pivot(index="date", columns="hour", values=value_col)
    pivot = pivot.reindex(columns=range(24))
    pivot = pivot.dropna()
    pivot.columns = [f"h{h:02d}" for h in range(24)]
    pivot = pivot.reset_index()
    return pivot


def load_price_scenarios_from_raw_15min(
    filepath: str,
    n_price_scenarios: int = 30,
    price_area: str = "DK2",
    use_first_n: bool = True
) -> Dict[int, np.ndarray]:
    """
    Converts raw 15-min prices to daily hourly scenarios.
    By default uses the first n complete days.
    """
    hourly_long = read_price_15min_to_hourly_long(filepath, price_area=price_area)
    daily = long_to_daily_matrix(hourly_long, value_col="price_EUR")

    daily.to_csv("data/processed/price_hourly_daily.csv", index=False)

    if len(daily) < n_price_scenarios:
        raise ValueError(
            f"Requested {n_price_scenarios} price scenarios, but only {len(daily)} complete days are available."
        )

    if use_first_n:
        selected = daily.iloc[:n_price_scenarios].copy()
    else:
        selected = daily.sample(n=n_price_scenarios, random_state=42).reset_index(drop=True)

    scenarios = {}
    hourly_cols = [f"h{h:02d}" for h in range(24)]

    for i in range(len(selected)):
        scenarios[i] = selected.iloc[i][hourly_cols].to_numpy(dtype=float)

    return scenarios


# --------------------------------
# Imbalance and balancing prices
# --------------------------------
def generate_imbalance_scenarios(
    n_scenarios: int = 4,
    deficit_probability: float = 0.5,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    SI = 1 means deficit
    SI = 0 means surplus
    """
    rng = np.random.default_rng(seed)
    scenarios = {}

    for s in range(n_scenarios):
        scenarios[s] = rng.binomial(1, deficit_probability, size=24)

    return scenarios


def compute_balancing_prices(
    price_scenarios: Dict[int, np.ndarray],
    imbalance_scenarios: Dict[int, np.ndarray],
    deficit_multiplier: float = 1.25,
    surplus_multiplier: float = 0.85
) -> Dict[Tuple[int, int], np.ndarray]:
    balancing_prices = {}

    for p_id, da in price_scenarios.items():
        for i_id, si in imbalance_scenarios.items():
            bp = np.where(si == 1, deficit_multiplier * da, surplus_multiplier * da)
            balancing_prices[(p_id, i_id)] = bp

    return balancing_prices


# --------------------------------
# Combined scenario builder
# --------------------------------
def build_combined_scenarios(
    wind_scenarios: Dict[int, np.ndarray],
    price_scenarios: Dict[int, np.ndarray],
    imbalance_scenarios: Dict[int, np.ndarray]
) -> CombinedScenarioSet:
    combined = list(product(
        wind_scenarios.keys(),
        price_scenarios.keys(),
        imbalance_scenarios.keys()
    ))

    prob = 1.0 / len(combined)
    probability = {sc: prob for sc in combined}

    return CombinedScenarioSet(
        scenarios=combined,
        probability=probability
    )


def prepare_scenario_data(
    wind_scenario_file: str,
    price_file: str,
    n_wind_scenarios: int = 20,
    n_price_scenarios: int = 20,
    n_imbalance_scenarios: int = 4,
    deficit_probability: float = 0.5,
    seed: int = 42,
    price_area: str = "DK2"
) -> Tuple[ScenarioData, CombinedScenarioSet]:
    """
    Full pipeline:
    - load wind scenarios from scen_zone2.csv
    - convert 15-min prices to hourly daily scenarios
    - generate imbalance scenarios
    - build balancing prices
    - build combined scenarios
    """
    wind = load_wind_scenarios_from_scenario_file(
        filepath=wind_scenario_file,
        n_wind_scenarios=n_wind_scenarios,
        n_hours=24,
        capacity_mw=CAPACITY_MW
    )

    price = load_price_scenarios_from_raw_15min(
        filepath=price_file,
        n_price_scenarios=n_price_scenarios,
        price_area=price_area,
        use_first_n=True
    )

    imbalance = generate_imbalance_scenarios(
        n_scenarios=n_imbalance_scenarios,
        deficit_probability=deficit_probability,
        seed=seed
    )

    balancing_price = compute_balancing_prices(price, imbalance)

    combined = build_combined_scenarios(wind, price, imbalance)

    data = ScenarioData(
        wind=wind,
        price=price,
        imbalance=imbalance,
        balancing_price=balancing_price
    )

    return data, combined


# --------------------------------
# Profit evaluation
# --------------------------------
def scenario_profit_one_price(
    offer: np.ndarray,
    wind_realization: np.ndarray,
    da_price: np.ndarray,
    balancing_price: np.ndarray
) -> float:
    """
    One-price profit:
    sum_t [ DA_t * offer_t + BP_t * (wind_t - offer_t) ]
    """
    return float(np.sum(da_price * offer + balancing_price * (wind_realization - offer)))


def evaluate_one_price_across_scenarios(
    offer: np.ndarray,
    data: ScenarioData,
    combined: CombinedScenarioSet
) -> List[float]:
    profits = []

    for (w_s, p_s, i_s) in combined.scenarios:
        profit = scenario_profit_one_price(
            offer=offer,
            wind_realization=data.wind[w_s],
            da_price=data.price[p_s],
            balancing_price=data.balancing_price[(p_s, i_s)]
        )
        profits.append(profit)

    return profits


# --------------------------------
# Output helpers
# --------------------------------
def save_offer_to_csv(offer: np.ndarray, filename: str) -> None:
    df = pd.DataFrame({
        "hour": HOURS,
        "offer_MW": offer
    })
    df.to_csv(filename, index=False)


def save_wind_scenarios_to_csv(
    wind_scenarios: Dict[int, np.ndarray],
    filename: str = "data/processed/wind_scenarios_used.csv"
) -> None:
    df = pd.DataFrame(wind_scenarios)
    df.index = [f"h{h:02d}" for h in range(24)]
    df.to_csv(filename)

def subset_combined_scenarios(combined, selected_scenarios):
    """
    Creates a new CombinedScenarioSet from a subset of combined scenarios.
    Equal probabilities are reassigned within the subset.
    """
    prob = 1.0 / len(selected_scenarios)
    probability = {sc: prob for sc in selected_scenarios}

    return CombinedScenarioSet(
        scenarios=list(selected_scenarios),
        probability=probability
    )


def make_k_folds(scenarios, k: int = 8, seed: int = 42):
    """
    Splits scenarios into k folds while keeping each scenario as a tuple.
    """
    rng = np.random.default_rng(seed)

    scenarios = list(scenarios)
    rng.shuffle(scenarios)

    folds = [[] for _ in range(k)]

    for idx, sc in enumerate(scenarios):
        folds[idx % k].append(sc)

    return folds


