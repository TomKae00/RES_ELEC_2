# helpers2.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import os
import numpy as np
import pandas as pd


# ============================================================
# Constants for Step 2
# ============================================================
MIN_LOAD_KW = 220.0
MAX_LOAD_KW = 600.0
MAX_RAMP_KW_PER_MIN = 35.0
N_MINUTES = 60

N_TOTAL_PROFILES = 300
N_IN_SAMPLE = 100
N_OUT_SAMPLE = 200


@dataclass
class LoadScenarioData:
    """
    Stores generated load profiles and reserve availability.

    load_profiles:
        shape = n_profiles x 60

    reserve_availability:
        shape = n_profiles x 60
        reserve_availability = load - MIN_LOAD_KW
    """

    load_profiles: np.ndarray
    reserve_availability: np.ndarray
    in_sample_load: np.ndarray
    out_sample_load: np.ndarray
    in_sample_reserve: np.ndarray
    out_sample_reserve: np.ndarray


# ============================================================
# Folder helpers
# ============================================================
def ensure_step2_output_folders() -> None:
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    os.makedirs("outputs/figures/task_2_1", exist_ok=True)
    os.makedirs("outputs/tables/task_2_1", exist_ok=True)

    os.makedirs("../../../Downloads/data", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


# ============================================================
# Load-profile generation
# ============================================================
def generate_single_load_profile(
    rng: np.random.Generator,
    n_minutes: int = N_MINUTES,
    min_load_kw: float = MIN_LOAD_KW,
    max_load_kw: float = MAX_LOAD_KW,
    max_ramp_kw: float = MAX_RAMP_KW_PER_MIN,
) -> np.ndarray:
    """
    Generates one stochastic load profile.

    Requirements:
        - load remains between min_load_kw and max_load_kw
        - minute-to-minute change does not exceed max_ramp_kw
    """

    profile = np.zeros(n_minutes)

    profile[0] = rng.uniform(min_load_kw, max_load_kw)

    for m in range(1, n_minutes):
        step = rng.uniform(-max_ramp_kw, max_ramp_kw)
        profile[m] = np.clip(profile[m - 1] + step, min_load_kw, max_load_kw)

    return profile


def generate_load_profiles(
    n_profiles: int = N_TOTAL_PROFILES,
    n_minutes: int = N_MINUTES,
    min_load_kw: float = MIN_LOAD_KW,
    max_load_kw: float = MAX_LOAD_KW,
    max_ramp_kw: float = MAX_RAMP_KW_PER_MIN,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Generates stochastic load profiles for Step 2.
    """

    rng = np.random.default_rng(seed)

    profiles = np.vstack([
        generate_single_load_profile(
            rng=rng,
            n_minutes=n_minutes,
            min_load_kw=min_load_kw,
            max_load_kw=max_load_kw,
            max_ramp_kw=max_ramp_kw,
        )
        for _ in range(n_profiles)
    ])

    return profiles


def split_in_sample_out_sample(
    load_profiles: np.ndarray,
    n_in_sample: int = N_IN_SAMPLE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits profiles into:
        - first n_in_sample profiles as in-sample
        - remaining profiles as out-of-sample
    """

    in_sample = load_profiles[:n_in_sample, :]
    out_sample = load_profiles[n_in_sample:, :]

    return in_sample, out_sample


def compute_reserve_availability(
    load_profiles: np.ndarray,
    min_load_kw: float = MIN_LOAD_KW,
) -> np.ndarray:
    """
    FCR-D UP reserve availability.

    Assumption:
        The flexible load can reduce consumption down to min_load_kw.

    Therefore:
        available reserve = load - min_load_kw
    """

    return np.maximum(load_profiles - min_load_kw, 0.0)


def prepare_load_scenario_data(
    n_profiles: int = N_TOTAL_PROFILES,
    n_in_sample: int = N_IN_SAMPLE,
    seed: int | None = 42,
) -> LoadScenarioData:
    """
    Full Step 2 scenario-generation pipeline.
    """

    load_profiles = generate_load_profiles(
        n_profiles=n_profiles,
        seed=seed,
    )

    reserve_availability = compute_reserve_availability(load_profiles)

    in_sample_load, out_sample_load = split_in_sample_out_sample(
        load_profiles=load_profiles,
        n_in_sample=n_in_sample,
    )

    in_sample_reserve, out_sample_reserve = split_in_sample_out_sample(
        load_profiles=reserve_availability,
        n_in_sample=n_in_sample,
    )

    return LoadScenarioData(
        load_profiles=load_profiles,
        reserve_availability=reserve_availability,
        in_sample_load=in_sample_load,
        out_sample_load=out_sample_load,
        in_sample_reserve=in_sample_reserve,
        out_sample_reserve=out_sample_reserve,
    )


# ============================================================
# Validation helpers
# ============================================================
def check_load_profiles(
    load_profiles: np.ndarray,
    min_load_kw: float = MIN_LOAD_KW,
    max_load_kw: float = MAX_LOAD_KW,
    max_ramp_kw: float = MAX_RAMP_KW_PER_MIN,
) -> dict:
    """
    Checks whether generated load profiles satisfy assignment requirements.
    """

    min_value = float(np.min(load_profiles))
    max_value = float(np.max(load_profiles))

    minute_changes = np.abs(np.diff(load_profiles, axis=1))
    max_ramp = float(np.max(minute_changes))

    return {
        "minimum_load_observed": min_value,
        "maximum_load_observed": max_value,
        "maximum_ramp_observed": max_ramp,
        "satisfies_min_load": bool(min_value >= min_load_kw - 1e-6),
        "satisfies_max_load": bool(max_value <= max_load_kw + 1e-6),
        "satisfies_ramp_limit": bool(max_ramp <= max_ramp_kw + 1e-6),
    }


def save_load_profiles_to_csv(
    load_profiles: np.ndarray,
    filename: str,
) -> None:
    """
    Saves load profiles to CSV.
    """

    df = pd.DataFrame(
        load_profiles,
        columns=[f"minute_{m:02d}" for m in range(load_profiles.shape[1])],
    )

    df.insert(0, "profile_id", range(load_profiles.shape[0]))
    df.to_csv(filename, index=False)


def save_reserve_availability_to_csv(
    reserve_availability: np.ndarray,
    filename: str,
) -> None:
    """
    Saves reserve availability profiles to CSV.
    """

    df = pd.DataFrame(
        reserve_availability,
        columns=[f"minute_{m:02d}" for m in range(reserve_availability.shape[1])],
    )

    df.insert(0, "profile_id", range(reserve_availability.shape[0]))
    df.to_csv(filename, index=False)