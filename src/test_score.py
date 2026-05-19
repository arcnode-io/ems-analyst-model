"""Unit tests for score helpers — future-hours generator + model dispatch."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import polars as pl
from prophet import Prophet
from xgboost.sklearn import XGBRegressor

from src.score import _future_hours, _score_prophet, _score_tree


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


def test_score_tree_xgboost_returns_one_value_per_timestamp() -> None:
    """XGBoost trained on time-features returns N predictions for N ts."""
    from src.models import XGBOOST_FEATURE_COLUMNS
    from src.train import create_xgboost_features

    ts_train = [datetime(2024, 1, 1, h, tzinfo=UTC) for h in range(24)]
    train_df = pl.DataFrame({"ts": ts_train, "value": [float(h) for h in range(24)]})
    features = create_xgboost_features(train_df)
    x_train = features.select(XGBOOST_FEATURE_COLUMNS).to_numpy()
    y_train = features["value"].to_numpy()
    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(x_train, y_train)
    future = [datetime.now(UTC) + timedelta(hours=i) for i in range(3)]
    actual = _score_tree(model, future)
    assert len(actual) == 3
    assert all(isinstance(v, float) for v in actual)


def test_score_tree_lightgbm_returns_one_value_per_timestamp() -> None:
    """LightGBM trained on time-features returns N predictions for N ts."""
    from lightgbm import LGBMRegressor

    from src.models import XGBOOST_FEATURE_COLUMNS
    from src.train import create_xgboost_features

    ts_train = [datetime(2024, 1, 1, h, tzinfo=UTC) for h in range(24)]
    train_df = pl.DataFrame({"ts": ts_train, "value": [float(h) for h in range(24)]})
    features = create_xgboost_features(train_df)
    x_train = features.select(XGBOOST_FEATURE_COLUMNS).to_numpy()
    y_train = features["value"].to_numpy()
    model = LGBMRegressor(n_estimators=5, max_depth=2, random_state=42, verbose=-1)
    model.fit(x_train, y_train)
    future = [datetime.now(UTC) + timedelta(hours=i) for i in range(3)]
    actual = _score_tree(model, future)
    assert len(actual) == 3
    assert all(isinstance(v, float) for v in actual)
