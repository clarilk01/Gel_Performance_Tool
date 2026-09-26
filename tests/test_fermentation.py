import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fermentation import (  # noqa: E402
    CONDITION_RANGES,
    Constraints,
    Economics,
    FermentationModel,
    evaluate_harvest,
    find_optimal,
    generate_history,
    simulate_batch,
)
from fermentation.model import calibrate  # noqa: E402
from fermentation.validation import backtest, split_batches  # noqa: E402

DEFAULT = {k: v["default"] for k, v in CONDITION_RANGES.items()}


@pytest.fixture(scope="module")
def trained():
    _, ts = generate_history(n_batches=120, seed=1)
    train_ts, test_ts = split_batches(ts)
    return FermentationModel().fit(train_ts), test_ts


def test_simulated_batch_rises_then_declines():
    curve = simulate_batch(DEFAULT, noise=False)
    peak = curve["titer"].idxmax()
    assert 0 < peak < len(curve) - 1
    assert curve["viability"].is_monotonic_decreasing
    assert curve["hcp"].iloc[-1] > curve["hcp"].iloc[0]


def test_optimum_is_interior_and_respects_constraints():
    ev = evaluate_harvest(simulate_batch(DEFAULT, noise=False), Economics())
    cons = Constraints(min_viability=70, max_hcp=1500, min_hours=48, max_hours=240)
    best = find_optimal(ev, cons)
    assert best["feasible"]
    assert 48 < best["hour"] < 240
    assert best["viability"] >= 70 and best["hcp"] <= 1500
    # Tighter viability limit can only pull harvest earlier.
    strict = find_optimal(ev, Constraints(min_viability=95, max_hcp=1500))
    assert strict["hour"] <= best["hour"]


def test_titer_objective_ignores_costs():
    ev = evaluate_harvest(simulate_batch(DEFAULT, noise=False), Economics())
    loose = Constraints(min_viability=0, max_hcp=1e9, min_hours=0, max_hours=240)
    assert find_optimal(ev, loose, "titer")["hour"] == ev.loc[ev["titer"].idxmax(), "hour"]


def test_longer_turnaround_favours_longer_batches():
    curve = simulate_batch(DEFAULT, noise=False)
    loose = Constraints(min_viability=0, max_hcp=1e9, min_hours=0)
    short = find_optimal(evaluate_harvest(curve, Economics(turnaround_h=4)), loose)
    long = find_optimal(evaluate_harvest(curve, Economics(turnaround_h=96)), loose)
    assert long["hour"] >= short["hour"]


def test_model_predictions_are_well_formed(trained):
    model, _ = trained
    curves = model.predict_curves(DEFAULT)
    assert (curves["titer_lo"] <= curves["titer"]).all()
    assert (curves["titer"] <= curves["titer_hi"]).all()
    assert curves["viability"].between(0, 100).all()


def test_calibration_matches_measurement(trained):
    model, _ = trained
    curves = calibrate(model.predict_curves(DEFAULT), hour=72, titer=3.0, viability=80.0)
    at = curves.loc[curves["hour"] == 72].iloc[0]
    assert at["titer"] == pytest.approx(3.0)
    assert at["viability"] == pytest.approx(80.0)


def test_backtest_captures_most_of_the_value(trained):
    model, test_ts = trained
    bt = backtest(model, test_ts, Economics(), Constraints())
    assert len(bt) == test_ts["batch_id"].nunique()
    assert bt["captured_objective"].sum() / bt["best_objective"].sum() > 0.6
    assert np.isfinite(bt["error_hours"]).all()
