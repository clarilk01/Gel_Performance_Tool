"""Harvest-time economics: turn predicted batch curves into an optimal fermentation length."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OBJECTIVES = {
    "profit_rate": "Profit rate ($/h of reactor time)",
    "productivity": "Volumetric productivity (g/L/h incl. turnaround)",
    "titer": "Maximum titer (g/L)",
}


@dataclass
class Economics:
    reactor_volume_l: float = 2000.0
    product_price_per_g: float = 100.0
    operating_cost_per_h: float = 800.0
    batch_fixed_cost: float = 30000.0
    turnaround_h: float = 24.0
    # Chromatography gel wear / extra purification per gram of HCP loaded.
    resin_cost_per_g_hcp: float = 20.0


@dataclass
class Constraints:
    min_viability: float = 70.0
    max_hcp: float = 1500.0
    min_hours: float = 48.0
    max_hours: float = 240.0


def downstream_yield(hcp_mg_l):
    """Purification yield drops as HCP load on the chromatography gel rises."""
    return np.clip(0.90 - 0.00015 * np.asarray(hcp_mg_l, float), 0.30, 0.95)


def evaluate_harvest(curves: pd.DataFrame, econ: Economics, titer_col: str = "titer") -> pd.DataFrame:
    """Add economic columns for harvesting at each hour of ``curves``.

    ``curves`` needs ``hour``, a titer column (g/L) and ``hcp`` (mg/L).
    """
    out = curves.copy()
    hours = out["hour"].to_numpy(float)
    titer = out[titer_col].to_numpy(float)
    hcp = out["hcp"].to_numpy(float)
    y = downstream_yield(hcp)

    purified_g = titer * econ.reactor_volume_l * y
    revenue = purified_g * econ.product_price_per_g
    cost = (
        econ.batch_fixed_cost
        + econ.operating_cost_per_h * hours
        + econ.resin_cost_per_g_hcp * hcp * econ.reactor_volume_l / 1000.0
    )
    cycle_h = hours + econ.turnaround_h

    out["downstream_yield"] = y
    out["purified_kg"] = purified_g / 1000.0
    out["profit"] = revenue - cost
    out["profit_rate"] = out["profit"] / cycle_h
    out["productivity"] = titer * y / cycle_h
    return out


def find_optimal(evaluated: pd.DataFrame, constraints: Constraints, objective: str = "profit_rate") -> dict:
    """Pick the harvest hour that maximizes ``objective`` within the constraints.

    Returns the chosen row as a dict plus ``feasible`` (False when no hour
    satisfies every constraint, in which case the best hour inside the time
    window is returned) and ``limited_by`` naming the binding constraint.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}")
    df = evaluated
    in_window = df["hour"].between(constraints.min_hours, constraints.max_hours)
    ok_viab = df["viability"] >= constraints.min_viability
    ok_hcp = df["hcp"] <= constraints.max_hcp
    feasible = in_window & ok_viab & ok_hcp

    candidates = df[feasible] if feasible.any() else df[in_window]
    if candidates.empty:
        candidates = df
    best_idx = candidates[objective].idxmax()
    best = df.loc[best_idx].to_dict()

    # Which constraint stops us from going to the unconstrained optimum?
    unconstrained_idx = df.loc[in_window, objective].idxmax() if in_window.any() else best_idx
    limited_by = None
    if unconstrained_idx != best_idx:
        row = df.loc[unconstrained_idx]
        if row["viability"] < constraints.min_viability:
            limited_by = "viability"
        elif row["hcp"] > constraints.max_hcp:
            limited_by = "hcp"
    best["feasible"] = bool(feasible.any())
    best["limited_by"] = limited_by
    return best
