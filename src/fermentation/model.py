"""Gradient-boosting model of batch trajectories (titer, viability, HCP vs. time)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from .simulator import CONDITIONS, FEATURES, MAX_HOURS


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Centered rolling mean to remove the step artefacts of tree models."""
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


class FermentationModel:
    """Predicts titer (with an 80 % band), viability and HCP for any set-points and hour."""

    def __init__(self, random_state: int = 0):
        hour_idx = FEATURES.index("hour")

        def gbm(monotonic_hour=0, **kw):
            cst = [0] * len(FEATURES)
            cst[hour_idx] = monotonic_hour
            return HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                monotonic_cst=cst, random_state=random_state, **kw,
            )

        self.models = {
            "titer": gbm(),
            "titer_lo": gbm(loss="quantile", quantile=0.1),
            "titer_hi": gbm(loss="quantile", quantile=0.9),
            "viability": gbm(monotonic_hour=-1),
            "hcp": gbm(monotonic_hour=1),
        }
        self.target_of = {"titer": "titer", "titer_lo": "titer", "titer_hi": "titer",
                          "viability": "viability", "hcp": "hcp"}
        self.fitted = False

    def fit(self, timeseries: pd.DataFrame) -> "FermentationModel":
        X = timeseries[FEATURES]
        for name, model in self.models.items():
            model.fit(X, timeseries[self.target_of[name]])
        self.fitted = True
        return self

    def predict_curves(self, conditions: dict, hours: np.ndarray | None = None) -> pd.DataFrame:
        """Predicted trajectory for one batch at the given set-points."""
        hours = np.arange(0, MAX_HOURS + 1, 1.0) if hours is None else np.asarray(hours, float)
        X = pd.DataFrame({c: np.full(len(hours), conditions[c], float) for c in CONDITIONS})
        X["hour"] = hours
        X = X[FEATURES]
        out = pd.DataFrame({"hour": hours})
        for name, model in self.models.items():
            out[name] = _smooth(model.predict(X))
        out["titer"] = out["titer"].clip(lower=0)
        out["titer_lo"] = np.minimum(out["titer_lo"].clip(lower=0), out["titer"])
        out["titer_hi"] = np.maximum(out["titer_hi"], out["titer"])
        out["viability"] = out["viability"].clip(0, 100)
        out["hcp"] = out["hcp"].clip(lower=0)
        return out

    def feature_importance(self, timeseries: pd.DataFrame, target: str = "titer",
                           n_samples: int = 4000, seed: int = 0) -> pd.Series:
        sample = timeseries.sample(min(n_samples, len(timeseries)), random_state=seed)
        result = permutation_importance(
            self.models[target], sample[FEATURES], sample[target],
            n_repeats=3, random_state=seed,
        )
        return pd.Series(result.importances_mean, index=FEATURES).sort_values(ascending=False)


def calibrate(curves: pd.DataFrame, hour: float, titer: float | None = None,
              viability: float | None = None) -> pd.DataFrame:
    """Adjust a predicted trajectory with an in-process measurement of a running batch.

    Titer is rescaled by measured/predicted at ``hour``; viability is shifted
    by the observed offset, applied progressively after the measurement.
    """
    out = curves.copy()
    at = out.iloc[(out["hour"] - hour).abs().argmin()]
    if titer is not None and at["titer"] > 1e-6:
        factor = titer / at["titer"]
        for col in ("titer", "titer_lo", "titer_hi"):
            out[col] = out[col] * factor
    if viability is not None:
        offset = viability - at["viability"]
        out["viability"] = (out["viability"] + offset).clip(0, 100)
    return out
