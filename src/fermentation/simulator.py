"""Synthetic fed-batch fermentation simulator.

Generates historical batch records (process set-points + in-process time
series) used to train the fermentation-length model. The kinetics are
deliberately simple but capture the trade-off that makes harvest timing
matter:

* biomass follows logistic growth towards a feed-limited plateau,
* viability collapses some hours after the plateau is reached,
* product is formed by viable cells (Luedeking-Piret) and slowly degraded,
* host-cell protein (HCP) impurities rise as cells lyse, which lowers
  downstream yield and wears the chromatography gel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Process set-points the operator controls, with the ranges seen in history.
CONDITION_RANGES = {
    "temperature": {"min": 30.0, "max": 38.0, "default": 34.0, "unit": "°C"},
    "ph": {"min": 6.6, "max": 7.4, "default": 7.0, "unit": ""},
    "dissolved_oxygen": {"min": 15.0, "max": 60.0, "default": 40.0, "unit": "%"},
    "inoculum_density": {"min": 0.2, "max": 1.0, "default": 0.5, "unit": "g/L"},
    "feed_rate": {"min": 5.0, "max": 25.0, "default": 15.0, "unit": "g/L/day"},
}
CONDITIONS = list(CONDITION_RANGES)
FEATURES = CONDITIONS + ["hour"]
TARGETS = ["titer", "viability", "hcp"]

MAX_HOURS = 240


def _kinetics(cond: dict, hidden: dict | None = None) -> dict:
    """Kinetic parameters for a set of process conditions."""
    hidden = hidden or {"mu_mult": 1.0, "xmax_mult": 1.0, "death_mult": 1.0}
    t, ph, do = cond["temperature"], cond["ph"], cond["dissolved_oxygen"]
    x0, feed = cond["inoculum_density"], cond["feed_rate"]

    f_temp = np.exp(-(((t - 34.0) / 3.5) ** 2))
    f_ph = np.exp(-(((ph - 7.0) / 0.35) ** 2))
    f_do = do / (do + 12.0)
    mu = 0.16 * f_temp * f_ph * f_do * hidden["mu_mult"]

    x_max = (18.0 + 1.2 * feed) * hidden["xmax_mult"]
    # Time to reach 95 % of the plateau, then the feed sustains the culture a while.
    t95 = np.log(19.0 * (x_max - x0) / x0) / max(mu, 1e-3)
    t_death = t95 + 0.8 * feed
    k_death = (
        0.012
        * np.exp(0.25 * (t - 34.0))
        * (1.0 + 0.5 * ((ph - 7.0) / 0.3) ** 2)
        * hidden["death_mult"]
    )
    # Specific productivity peaks at a slightly lower temperature than growth.
    beta = 0.0015 * np.exp(-(((t - 32.0) / 4.0) ** 2))
    k_deg = 0.002 * np.exp(0.2 * (t - 34.0))
    return dict(mu=mu, x0=x0, x_max=x_max, t_death=t_death, k_death=k_death,
                beta=beta, k_deg=k_deg)


def simulate_batch(
    cond: dict,
    hours: np.ndarray | None = None,
    hidden: dict | None = None,
    noise: bool = True,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate one batch; returns titer (g/L), viability (%) and HCP (mg/L) per hour.

    ``hidden`` holds batch-to-batch variation the operator cannot observe
    (raw-material lot, seed-train quality...). ``noise`` adds assay noise.
    """
    rng = rng or np.random.default_rng()
    hours = np.arange(0, MAX_HOURS + 1, 1.0) if hours is None else np.asarray(hours, float)
    k = _kinetics(cond, hidden)

    # Integrate on a fine 1 h grid, then sample the requested hours.
    grid = np.arange(0, max(hours.max(), 1.0) + 1, 1.0)
    x = k["x_max"] / (1.0 + (k["x_max"] - k["x0"]) / k["x0"] * np.exp(-k["mu"] * grid))
    viability = 97.0 * np.exp(-k["k_death"] * np.clip(grid - k["t_death"], 0, None))
    x_viable = x * viability / 100.0

    titer = np.zeros_like(grid)
    dx = np.diff(x, prepend=x[0])
    for i in range(1, len(grid)):
        protease = 0.00005 * (100.0 - viability[i])
        titer[i] = titer[i - 1] + 0.02 * dx[i] + k["beta"] * x_viable[i] - (k["k_deg"] + protease) * titer[i - 1]
    hcp = 8.0 * x + 60.0 * x * (1.0 - viability / 97.0)

    idx = np.searchsorted(grid, hours)
    out = pd.DataFrame({
        "hour": hours,
        "titer": titer[idx],
        "viability": viability[idx],
        "hcp": hcp[idx],
    })
    if noise:
        n = len(out)
        out["titer"] = np.clip(out["titer"] * rng.normal(1.0, 0.04, n), 0, None)
        out["viability"] = np.clip(out["viability"] + rng.normal(0.0, 1.0, n), 0, 100)
        out["hcp"] = np.clip(out["hcp"] * rng.normal(1.0, 0.05, n), 0, None)
    for name, value in cond.items():
        out[name] = value
    return out


def sample_conditions(rng: np.random.Generator) -> dict:
    return {name: float(rng.uniform(r["min"], r["max"])) for name, r in CONDITION_RANGES.items()}


def sample_hidden(rng: np.random.Generator) -> dict:
    return {
        "mu_mult": float(rng.normal(1.0, 0.06)),
        "xmax_mult": float(rng.normal(1.0, 0.05)),
        "death_mult": float(rng.lognormal(0.0, 0.15)),
    }


def generate_history(n_batches: int = 400, sample_every: float = 2.0, seed: int = 42):
    """Generate historical batches.

    Returns ``(batches, timeseries)``: one row per batch with its set-points
    and hidden variation, and the long-format sampled in-process data.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(0, MAX_HOURS + 1, sample_every)
    batch_rows, frames = [], []
    for batch_id in range(1, n_batches + 1):
        cond = sample_conditions(rng)
        hidden = sample_hidden(rng)
        ts = simulate_batch(cond, hours, hidden, noise=True, rng=rng)
        ts.insert(0, "batch_id", batch_id)
        frames.append(ts)
        batch_rows.append({"batch_id": batch_id, **cond, **hidden})
    timeseries = pd.concat(frames, ignore_index=True)
    return pd.DataFrame(batch_rows), timeseries[["batch_id"] + FEATURES + TARGETS]


if __name__ == "__main__":
    import pathlib

    out_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    batches, ts = generate_history()
    batches.to_csv(out_dir / "fermentation_batches.csv", index=False)
    ts.to_csv(out_dir / "fermentation_timeseries.csv", index=False)
    print(f"Wrote {len(batches)} batches / {len(ts)} samples to {out_dir}")
