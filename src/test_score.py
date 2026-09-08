"""Unit tests for score helpers — future-hours generator + model dispatch."""

from datetime import UTC, datetime, timedelta
from typing import cast

import pandas as pd
import polars as pl
import pytest
from prophet import Prophet
from xgboost.sklearn import XGBRegressor

from src.score import (
    _future_hours,
    _score_prophet,
    _score_tree,
    realign_curve_by_hour,
)


def test_future_hours_returns_horizon_count() -> None:
    """24-hour horizon → 24 timestamps."""
    # Arrange + Act
    actual = _future_hours(24)
    # Assert
    assert len(actual) == 24


def test_future_hours_starts_top_of_next_hour() -> None:
    """First ts is :00 of the next hour after now."""
    # Arrange
    expected_first = datetime.now(UTC).replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=1)
    # Act
    actual = _future_hours(1)
    # Assert
    assert actual[0] == expected_first


def test_future_hours_are_evenly_hourly_spaced() -> None:
    """Each consecutive pair is exactly one hour apart."""
    # Arrange + Act
    actual = _future_hours(5)
    # Assert
    for i in range(1, 5):
        assert actual[i] - actual[i - 1] == timedelta(hours=1)


def test_score_prophet_returns_one_value_per_timestamp() -> None:
    """Prophet trained on tiny series returns N predictions for N ts."""
    # Arrange — train a tiny Prophet on synthetic data
    train_df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=60, freq="h"),
            "y": list(range(60)),
        }
    )
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        uncertainty_samples=0,
    )
    model.fit(train_df)
    future = [datetime.now(UTC) + timedelta(hours=i) for i in range(3)]
    # Act
    actual = _score_prophet(model, future)
    # Assert
    assert len(actual) == 3
    assert all(isinstance(v, float) for v in actual)


def test_realign_curve_by_hour_maps_values_by_hour_of_day() -> None:
    """A canonical curve is re-timed onto a new window, matched on UTC hour."""
    # Arrange — 24-row curve, value == hour so the mapping is easy to check
    curve = pl.DataFrame(
        {"hour_utc": list(range(24)), "usd_per_mwh": [float(h) for h in range(24)]}
    )
    target_hours = [
        datetime(2026, 9, 8, 5, tzinfo=UTC) + timedelta(hours=i) for i in range(24)
    ]
    # Act
    actual = realign_curve_by_hour(curve, target_hours)
    # Assert — forecast_for is the new window; each value came from the
    # curve row with the same hour-of-day (target 05:00 → curve hour 5 → 5.0)
    assert actual["forecast_for"].to_list() == target_hours
    assert actual["value"].to_list() == [float(t.hour) for t in target_hours]


def test_realign_curve_by_hour_rejects_incomplete_curve() -> None:
    """A curve missing a target hour-of-day is a hard error."""
    # Arrange — only 3 hours, target spans a full 24
    curve = pl.DataFrame({"hour_utc": [9, 10, 11], "usd_per_mwh": [1.0, 2.0, 3.0]})
    target_hours = [
        datetime(2026, 9, 8, 0, tzinfo=UTC) + timedelta(hours=i) for i in range(24)
    ]
    # Act / Assert
    with pytest.raises(ValueError, match="missing hours-of-day"):
        realign_curve_by_hour(curve, target_hours)


def test_realign_curve_by_hour_rejects_repeated_hour() -> None:
    """A curve with the same hour twice is a hard error."""
    # Arrange — hour 9 appears twice
    curve = pl.DataFrame(
        {"hour_utc": [*range(24), 9], "usd_per_mwh": [float(i) for i in range(25)]}
    )
    target_hours = [
        datetime(2026, 9, 8, 0, tzinfo=UTC) + timedelta(hours=i) for i in range(24)
    ]
    # Act / Assert
    with pytest.raises(ValueError, match="repeats an hour"):
        realign_curve_by_hour(curve, target_hours)


def _build_history(start: datetime, hours: int = 200) -> pl.DataFrame:
    """Synthetic hourly history with enough rows for 168h lag features.

    Includes a `load_forecast_mw` column populated with a constant
    placeholder — real ETL would join from `load_forecasts_raw`.
    """
    ts = [start + timedelta(hours=i) for i in range(hours)]
    return pl.DataFrame(
        {
            "ts": ts,
            "value": [float(i) for i in range(hours)],
            "load_forecast_mw": [10000.0 + float(i) for i in range(hours)],
        }
    )


def _train_tree(model_kind: str, history: pl.DataFrame) -> object:
    """Fit either XGBoost or LightGBM on the shared feature shape."""
    from src.models import XGBOOST_FEATURE_COLUMNS
    from src.train import create_xgboost_features

    feat = create_xgboost_features(history).drop_nulls(
        subset=["value_lag_168h", "load_forecast_mw"]
    )
    x = feat.select(XGBOOST_FEATURE_COLUMNS).to_numpy()
    y = feat["value"].to_numpy()
    if model_kind == "xgboost":
        model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    else:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(n_estimators=5, max_depth=2, random_state=42, verbose=-1)
    model.fit(x, y)
    return model


def test_score_tree_xgboost_returns_one_value_per_timestamp() -> None:
    """XGBoost predicts N future hours from time + lag + load-forecast features."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    history = _build_history(start)
    model = cast(XGBRegressor, _train_tree("xgboost", history))
    future = [start + timedelta(hours=200 + i) for i in range(3)]
    load_forecast = {ts: 10500.0 for ts in future}
    actual = _score_tree(model, future, history, load_forecast)
    assert len(actual) == 3
    assert all(isinstance(v, float) for v in actual)


def test_score_tree_lightgbm_returns_one_value_per_timestamp() -> None:
    """LightGBM predicts N future hours from time + lag + load-forecast features."""
    from lightgbm import LGBMRegressor

    start = datetime(2024, 1, 1, tzinfo=UTC)
    history = _build_history(start)
    model = cast(LGBMRegressor, _train_tree("lightgbm", history))
    future = [start + timedelta(hours=200 + i) for i in range(3)]
    load_forecast = {ts: 10500.0 for ts in future}
    actual = _score_tree(model, future, history, load_forecast)
    assert len(actual) == 3
    assert all(isinstance(v, float) for v in actual)
