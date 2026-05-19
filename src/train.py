"""Model training pipeline with champion/challenger pattern.

Three models trained per run: Prophet (seasonality), XGBoost + LightGBM
(both gradient-boosted trees on the same time-based features). Champion
is whichever has lowest holdout MAE; if all three lose to the naive
baseline (mean), system degradation is logged.
"""

import logging
import os
from datetime import datetime, timedelta

import polars as pl
import psycopg2
from lightgbm import LGBMRegressor
from prophet import Prophet
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import mean_absolute_error
from xgboost.sklearn import XGBRegressor

from src.config import Config
from src.models import (
    LIGHTGBM_LEARNING_RATE,
    LIGHTGBM_MAX_DEPTH,
    LIGHTGBM_N_ESTIMATORS,
    LIGHTGBM_RANDOM_STATE,
    MIN_LAG_HISTORY_HOURS,
    TRAIN_TEST_SPLIT_DAYS,
    XGBOOST_FEATURE_COLUMNS,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_RANDOM_STATE,
    PredictiveModels,
)


class TrainingResult(BaseModel):
    """Result of model training and evaluation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prophet_mae: float
    xgboost_mae: float
    lightgbm_mae: float
    baseline_mae: float
    champion: PredictiveModels
    prophet_model: Prophet
    xgboost_model: XGBRegressor
    lightgbm_model: LGBMRegressor


def load_timeseries_data(_config: Config) -> pl.DataFrame:
    """Load SPP target joined with day-ahead load forecast feature.

    The leakage guard `publish_time_utc <= interval_start_utc - 24h`
    keeps the load forecast realistic — only use forecasts that were
    actually available 24 hours before the target hour (DAM clearing
    horizon). `DISTINCT ON` picks the latest such publish per hour.

    LEFT JOIN tolerates missing forecast rows (first call before the
    load_forecasts_raw ETL has caught up); affected rows get
    `load_forecast_mw=NULL` and are dropped before training.
    """
    # Scope the load-forecast slice to the last 60 days so stale partial
    # backfills (from earlier ETL runs that only covered tiny windows
    # before the lookback narrowed) don't pollute the join. Outside that
    # window timeseries_data still trains on time + lag features alone
    # via the LEFT JOIN + null-drop in _featurize.
    sql = """
        WITH leakage_safe AS (
            SELECT DISTINCT ON (interval_start_utc)
                   interval_start_utc AS ts,
                   load_mw
            FROM load_forecasts_raw
            WHERE interval_start_utc >= NOW() - INTERVAL '60 days'
              AND publish_time_utc <= interval_start_utc - INTERVAL '24 hours'
            ORDER BY interval_start_utc, publish_time_utc DESC
        )
        SELECT t.ts, t.value, l.load_mw AS load_forecast_mw
        FROM timeseries_data t
        LEFT JOIN leakage_safe l ON l.ts = t.ts
        ORDER BY t.ts
    """
    with psycopg2.connect(os.environ["TIMESERIES_URL"]) as conn:
        # infer_schema_length=None → scan all rows to figure out dtypes.
        # The LEFT JOIN's load_forecast_mw column has long all-null
        # stretches (historical hours before the load-forecast ETL
        # window) — without this polars infers Null type and chokes
        # when it eventually sees a float.
        return pl.read_database(sql, connection=conn, infer_schema_length=None)


def split_train_test(
    df: pl.DataFrame, test_days: int = TRAIN_TEST_SPLIT_DAYS
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Time-ordered split — last `test_days` for holdout."""
    max_ts = df["ts"].max()
    assert isinstance(max_ts, datetime)
    split_date = max_ts - timedelta(days=test_days)  # type: ignore[operator]
    train_df = df.filter(pl.col("ts") < split_date)
    test_df = df.filter(pl.col("ts") >= split_date)
    return train_df, test_df


def train_prophet(
    train_df: pl.DataFrame, test_df: pl.DataFrame, config: Config
) -> tuple[float, Prophet]:
    """Fit Prophet on train, predict test, return (MAE, model)."""
    prophet_train = train_df.to_pandas().rename(columns={"ts": "ds", "value": "y"})
    prophet_test = test_df.to_pandas().rename(columns={"ts": "ds", "value": "y"})
    prophet_train["ds"] = prophet_train["ds"].dt.tz_localize(None)
    prophet_test["ds"] = prophet_test["ds"].dt.tz_localize(None)
    logging.info("Training Prophet model...")
    model = Prophet(
        daily_seasonality=config.prophet_daily_seasonality,
        weekly_seasonality=config.prophet_weekly_seasonality,
        yearly_seasonality=config.prophet_yearly_seasonality,
        seasonality_mode="multiplicative",
        uncertainty_samples=0,
    )
    model.fit(prophet_train)
    forecast = model.predict(prophet_test[["ds"]])
    mae: float = mean_absolute_error(
        test_df["value"].to_numpy(), forecast["yhat"].to_numpy()
    )
    return mae, model


def create_xgboost_features(df: pl.DataFrame) -> pl.DataFrame:
    """Time-derived + lagged-value features shared by XGBoost + LightGBM.

    Assumes input df is sorted by ts and hourly-spaced. Lags use polars
    shift — first 24/168 rows of the dataframe will have null lag values
    and must be dropped before training (handled by `_featurize`).
    """
    return df.with_columns(
        [
            pl.col("ts").dt.hour().alias("hour"),
            pl.col("ts").dt.day().alias("day"),
            pl.col("ts").dt.month().alias("month"),
            pl.col("ts").dt.year().alias("year"),
            pl.col("ts").dt.weekday().alias("dayofweek"),
            pl.col("ts").dt.ordinal_day().alias("dayofyear"),
            pl.col("ts").dt.week().alias("weekofyear"),
            pl.col("value").shift(24).alias("value_lag_24h"),
            pl.col("value").shift(MIN_LAG_HISTORY_HOURS).alias("value_lag_168h"),
        ]
    )


def _featurize(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add features to both splits + drop rows missing any feature.

    The test set's lags are filled because the lag values come from the
    train tail (we featurize on the concatenated set then re-split on
    timestamp). Rows missing the load forecast feature (LEFT JOIN
    misses, e.g. before the load_forecasts_raw ETL caught up to that
    historical hour) are dropped from both splits.
    """
    full = pl.concat([train_df, test_df]).sort("ts")
    full_feat = create_xgboost_features(full).drop_nulls(
        subset=["value_lag_168h", "load_forecast_mw"]
    )
    train_max_ts = train_df["ts"].max()
    train_feat = full_feat.filter(pl.col("ts") <= train_max_ts)
    test_feat = full_feat.filter(pl.col("ts") > train_max_ts)
    return train_feat, test_feat


def train_xgboost(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> tuple[float, XGBRegressor]:
    """Fit XGBoost on time features, return (MAE, model)."""
    logging.info("Training XGBoost model...")
    train_features, test_features = _featurize(train_df, test_df)
    x_train = train_features.select(XGBOOST_FEATURE_COLUMNS).to_numpy()
    y_train = train_features["value"].to_numpy()
    x_test = test_features.select(XGBOOST_FEATURE_COLUMNS).to_numpy()
    y_test = test_features["value"].to_numpy()
    model = XGBRegressor(
        n_estimators=XGBOOST_N_ESTIMATORS,
        learning_rate=XGBOOST_LEARNING_RATE,
        max_depth=XGBOOST_MAX_DEPTH,
        random_state=XGBOOST_RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    mae: float = mean_absolute_error(y_test, model.predict(x_test))
    return mae, model


def train_lightgbm(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> tuple[float, LGBMRegressor]:
    """Fit LightGBM on the same features XGBoost uses. Clean A/B comparison.

    LightGBM uses histogram-based splits + leaf-wise growth; on tabular
    data with limited features it often edges out XGBoost by a few %.
    Same hyperparams modulo defaults, so the win/loss reflects the algo,
    not tuning.
    """
    logging.info("Training LightGBM model...")
    train_features, test_features = _featurize(train_df, test_df)
    x_train = train_features.select(XGBOOST_FEATURE_COLUMNS).to_numpy()
    y_train = train_features["value"].to_numpy()
    x_test = test_features.select(XGBOOST_FEATURE_COLUMNS).to_numpy()
    y_test = test_features["value"].to_numpy()
    model = LGBMRegressor(
        n_estimators=LIGHTGBM_N_ESTIMATORS,
        learning_rate=LIGHTGBM_LEARNING_RATE,
        max_depth=LIGHTGBM_MAX_DEPTH,
        random_state=LIGHTGBM_RANDOM_STATE,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    mae: float = mean_absolute_error(y_test, model.predict(x_test))
    return mae, model


def calculate_baseline_mae(test_df: pl.DataFrame) -> float:
    """Naive baseline: predict mean of holdout. Floor any real model must beat."""
    y_true = test_df["value"].to_numpy()
    return float(mean_absolute_error(y_true, [y_true.mean()] * len(y_true)))


def get_model_mae(
    model: PredictiveModels,
    prophet_mae: float,
    xgboost_mae: float,
    lightgbm_mae: float,
) -> float:
    """Lookup MAE for a model name."""
    return {
        PredictiveModels.PROPHET: prophet_mae,
        PredictiveModels.XGBOOST: xgboost_mae,
        PredictiveModels.LIGHTGBM: lightgbm_mae,
    }[model]


def _best_challenger(
    prophet_mae: float, xgboost_mae: float, lightgbm_mae: float
) -> tuple[PredictiveModels, float]:
    """Return the model+MAE with the lowest MAE across all challengers."""
    pairs: list[tuple[PredictiveModels, float]] = [
        (PredictiveModels.PROPHET, prophet_mae),
        (PredictiveModels.XGBOOST, xgboost_mae),
        (PredictiveModels.LIGHTGBM, lightgbm_mae),
    ]
    return min(pairs, key=lambda p: p[1])


def select_champion(
    prophet_mae: float,
    xgboost_mae: float,
    lightgbm_mae: float,
    current_champion: PredictiveModels | None = None,
) -> PredictiveModels:
    """Promote whichever challenger beats the current champion.

    - First run: deploy best challenger.
    - Subsequent: if best challenger beats current champion, promote.
    - Baseline-vs-champion degradation is checked separately in
      train_models (logs + surfaces via /metrics for Grafana to alert).
    """
    challenger, challenger_mae = _best_challenger(
        prophet_mae, xgboost_mae, lightgbm_mae
    )
    if current_champion is None:
        return challenger
    champion_mae = get_model_mae(
        current_champion, prophet_mae, xgboost_mae, lightgbm_mae
    )
    if challenger_mae < champion_mae:
        return challenger
    return current_champion


def train_models(
    config: Config, current_champion: PredictiveModels | None = None
) -> TrainingResult:
    """Train Prophet + XGBoost + LightGBM, pick champion, log degradation."""
    df = load_timeseries_data(config)
    train_df, test_df = split_train_test(df)
    prophet_mae, prophet_model = train_prophet(train_df, test_df, config)
    xgboost_mae, xgboost_model = train_xgboost(train_df, test_df)
    lightgbm_mae, lightgbm_model = train_lightgbm(train_df, test_df)
    baseline_mae = calculate_baseline_mae(test_df)
    if current_champion is not None:
        champion_mae = get_model_mae(
            current_champion, prophet_mae, xgboost_mae, lightgbm_mae
        )
        if baseline_mae < champion_mae:
            logging.info(
                "System degradation: baseline %.2f < champion %.2f",
                baseline_mae,
                champion_mae,
            )
    champion = select_champion(
        prophet_mae,
        xgboost_mae,
        lightgbm_mae,
        current_champion=current_champion,
    )
    return TrainingResult(
        prophet_mae=prophet_mae,
        xgboost_mae=xgboost_mae,
        lightgbm_mae=lightgbm_mae,
        baseline_mae=baseline_mae,
        champion=champion,
        prophet_model=prophet_model,
        xgboost_model=xgboost_model,
        lightgbm_model=lightgbm_model,
    )
