"""Back-test: how close is the predicted optimal length to the one seen in held-out batches?"""

from __future__ import annotations

import pandas as pd

from .model import FermentationModel
from .optimizer import Constraints, Economics, evaluate_harvest, find_optimal
from .simulator import CONDITIONS


def split_batches(timeseries: pd.DataFrame, test_fraction: float = 0.2, seed: int = 0):
    ids = pd.Series(timeseries["batch_id"].unique()).sample(frac=1.0, random_state=seed)
    n_test = max(1, int(len(ids) * test_fraction))
    test_ids = set(ids.iloc[:n_test])
    is_test = timeseries["batch_id"].isin(test_ids)
    return timeseries[~is_test], timeseries[is_test]


def backtest(model: FermentationModel, test_ts: pd.DataFrame, econ: Economics,
             constraints: Constraints, objective: str = "profit_rate") -> pd.DataFrame:
    """Compare predicted vs. observed optimal harvest hour for each held-out batch.

    The observed optimum is found on the batch's own measurements (lightly
    smoothed to remove assay noise); ``captured`` is the objective reached by
    harvesting at the predicted hour instead, on that same observed curve.
    """
    rows = []
    for batch_id, batch in test_ts.groupby("batch_id"):
        batch = batch.sort_values("hour").reset_index(drop=True)
        observed = batch[["hour"]].copy()
        for col in ("titer", "viability", "hcp"):
            observed[col] = batch[col].rolling(3, center=True, min_periods=1).mean()
        observed = evaluate_harvest(observed, econ)
        actual = find_optimal(observed, constraints, objective)

        cond = batch.iloc[0][CONDITIONS].to_dict()
        predicted_curve = evaluate_harvest(model.predict_curves(cond, batch["hour"].to_numpy()), econ)
        predicted = find_optimal(predicted_curve, constraints, objective)
        captured = observed.loc[observed["hour"] == predicted["hour"], objective].iloc[0]

        rows.append({
            "batch_id": batch_id,
            **cond,
            "actual_hours": actual["hour"],
            "predicted_hours": predicted["hour"],
            "error_hours": predicted["hour"] - actual["hour"],
            "best_objective": actual[objective],
            "captured_objective": captured,
            "objective_lost": actual[objective] - captured,
        })
    return pd.DataFrame(rows)
