"""Model training pipeline with champion/challenger pattern."""

import os
from datetime import timedelta

import polars as pl
import psycopg2
import xgboost as xgb
from prophet import Prophet
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import mean_absolute_error

from src.config import Config
from src.models import (
    TRAIN_TEST_SPLIT_DAYS,
    XGBOOST_FEATURE_COLUMNS,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_RANDOM_STATE,
    PredictiveModels,
)
from src.utils import push_model_metrics


class TrainingResult(BaseModel):
    """Result of model training and evaluation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prophet_mae: float
    xgboost_mae: float
    baseline_mae: float
    champion: PredictiveModels
    prophet_model: Prophet
    xgboost_model: xgb.XGBRegressor


def load_timeseries_data(config: Config) -> pl.DataFrame:
    """Load timeseries data from TimescaleDB.

    Args:
        config: Configuration object

    Returns:
        Timeseries DataFrame
    """
    with psycopg2.connect(
        host=config.db_host,
        database=config.db_name,
        user=config.db_user,
        password=os.environ["POSTGRES_PASSWORD"],
        port=config.db_port,
    ) as conn:
        query = "SELECT timestamp, value FROM timeseries_data ORDER BY timestamp"
        df = pl.read_database(query, connection=conn)
    return df


def split_train_test(
    df: pl.DataFrame, test_days: int = TRAIN_TEST_SPLIT_DAYS
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split data into train and test sets.

    Args:
        df: Input DataFrame
        test_days: Number of days to use for testing

    Returns:
        Tuple of (train_df, test_df)
    """
    max_timestamp = df["timestamp"].max()
    split_date = max_timestamp - timedelta(days=test_days)  # type: ignore[operator]
    train_df = df.filter(pl.col("timestamp") < split_date)
    test_df = df.filter(pl.col("timestamp") >= split_date)
    return train_df, test_df


def train_prophet(
    train_df: pl.DataFrame, test_df: pl.DataFrame, config: Config
) -> tuple[float, Prophet]:
    """Train Prophet model and evaluate.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        config: Configuration object with Prophet settings

    Returns:
        Tuple of (MAE, trained_model)
    """
    import logging

    # Convert to pandas for Prophet (Prophet requires pandas)
    prophet_train = train_df.to_pandas().rename(
        columns={"timestamp": "ds", "value": "y"}
    )
    prophet_test = test_df.to_pandas().rename(columns={"timestamp": "ds", "value": "y"})

    # Remove timezone (Prophet doesn't support timezone-aware datetimes)
    prophet_train["ds"] = prophet_train["ds"].dt.tz_localize(None)
    prophet_test["ds"] = prophet_test["ds"].dt.tz_localize(None)

    # Train model
    logging.info("Training Prophet model...")
    model = Prophet(
        daily_seasonality=config.prophet_daily_seasonality,
        weekly_seasonality=config.prophet_weekly_seasonality,
        yearly_seasonality=config.prophet_yearly_seasonality,
        seasonality_mode="multiplicative",
        uncertainty_samples=0,  # Disable uncertainty intervals for speed (doesn't affect yhat)
    )
    model.fit(prophet_train)
    logging.info("Prophet training complete")

    # Predict and evaluate
    forecast = model.predict(prophet_test[["ds"]])
    y_true = test_df["value"].to_numpy()
    y_pred = forecast["yhat"].to_numpy()

    mae: float = mean_absolute_error(y_true, y_pred)
    return mae, model


def create_xgboost_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create time-based features for XGBoost.

    Args:
        df: Input DataFrame with timestamp column

    Returns:
        DataFrame with time-based features
    """
    return df.with_columns(
        [
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.day().alias("day"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.weekday().alias("dayofweek"),
            pl.col("timestamp").dt.ordinal_day().alias("dayofyear"),
            pl.col("timestamp").dt.week().alias("weekofyear"),
        ]
    )


def train_xgboost(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> tuple[float, xgb.XGBRegressor]:
    """Train XGBoost model and evaluate.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame

    Returns:
        Tuple of (MAE, trained_model)
    """
    import logging

    logging.info("Training XGBoost model...")
    # Create features
    train_features = create_xgboost_features(train_df)
    test_features = create_xgboost_features(test_df)

    # Convert to numpy
    X_train = train_features.select(XGBOOST_FEATURE_COLUMNS).to_numpy()  # noqa: N806
    y_train = train_features["value"].to_numpy()
    X_test = test_features.select(XGBOOST_FEATURE_COLUMNS).to_numpy()  # noqa: N806
    y_test = test_features["value"].to_numpy()

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=XGBOOST_N_ESTIMATORS,
        learning_rate=XGBOOST_LEARNING_RATE,
        max_depth=XGBOOST_MAX_DEPTH,
        random_state=XGBOOST_RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae: float = mean_absolute_error(y_test, y_pred)
    logging.info("XGBoost training complete")
    return mae, model


def calculate_baseline_mae(test_df: pl.DataFrame) -> float:
    """Calculate baseline model MAE (mean prediction).

    Args:
        test_df: Test DataFrame

    Returns:
        Baseline MAE
    """
    y_true = test_df["value"].to_numpy()
    y_pred = y_true.mean()

    mae: float = mean_absolute_error(y_true, [y_pred] * len(y_true))
    return mae


def get_model_mae(
    model: PredictiveModels, prophet_mae: float, xgboost_mae: float
) -> float:
    """Get MAE for a specific model.

    Args:
        model: Model type
        prophet_mae: Prophet model MAE
        xgboost_mae: XGBoost model MAE

    Returns:
        MAE for the specified model
    """
    model_maes = {
        PredictiveModels.PROPHET: prophet_mae,
        PredictiveModels.XGBOOST: xgboost_mae,
    }
    return model_maes[model]


def _get_challenger_and_mae(
    prophet_mae: float, xgboost_mae: float
) -> tuple[PredictiveModels, float]:
    """Determine best challenger model and its MAE.

    Args:
        prophet_mae: Prophet model MAE
        xgboost_mae: XGBoost model MAE

    Returns:
        Tuple of (challenger_model, challenger_mae)
    """
    challenger = (
        PredictiveModels.PROPHET
        if prophet_mae < xgboost_mae
        else PredictiveModels.XGBOOST
    )
    challenger_mae = get_model_mae(challenger, prophet_mae, xgboost_mae)
    return challenger, challenger_mae


def select_champion(
    prophet_mae: float,
    xgboost_mae: float,
    baseline_mae: float,
    current_champion: PredictiveModels | None = None,
) -> PredictiveModels:
    """Select champion model using champion/challenger/baseline pattern.

    Logic from readme.md:
    - if challenger > champion: deploy challenger
    - elif baseline > champion: alert and deploy baseline
    - else: keep champion

    Args:
        prophet_mae: Prophet model MAE
        xgboost_mae: XGBoost model MAE
        baseline_mae: Baseline model MAE
        current_champion: Current champion model (None if first run)

    Returns:
        Champion model to deploy
    """
    # Determine best challenger (Prophet vs XGBoost)
    challenger, challenger_mae = _get_challenger_and_mae(prophet_mae, xgboost_mae)

    # If no current champion, deploy challenger
    if current_champion is None:
        return challenger

    # Get current champion MAE
    champion_mae = get_model_mae(current_champion, prophet_mae, xgboost_mae)

    # Champion/challenger/baseline logic
    if challenger_mae < champion_mae:
        # Challenger is better, deploy it
        return challenger
    elif baseline_mae < champion_mae:
        # Baseline is better than champion - system degrading!
        # Alert is sent via push_model_metrics() in train_models()
        # Keep champion (don't regress to baseline)
        return current_champion
    else:
        # Champion is still best, keep it
        return current_champion


def train_models(
    config: Config, current_champion: PredictiveModels | None = None
) -> TrainingResult:
    """Train models and select champion.

    Args:
        config: Configuration object
        current_champion: Current champion model (None if first run)

    Returns:
        Training result with MAE and champion
    """
    # Load and split data
    df = load_timeseries_data(config)
    train_df, test_df = split_train_test(df)

    # Train models
    prophet_mae, prophet_model = train_prophet(train_df, test_df, config)
    xgboost_mae, xgboost_model = train_xgboost(train_df, test_df)
    baseline_mae = calculate_baseline_mae(test_df)

    # Determine best challenger
    _challenger, challenger_mae = _get_challenger_and_mae(prophet_mae, xgboost_mae)

    # Get current champion MAE if exists
    if current_champion is not None:
        champion_mae = get_model_mae(current_champion, prophet_mae, xgboost_mae)

        # Check if system is degrading (baseline beats champion)
        if baseline_mae < champion_mae:
            # Send Grafana alert via Prometheus pushgateway
            import logging

            logging.info(
                f"System degradation detected! Baseline MAE ({baseline_mae:.2f}) < "
                f"Champion MAE ({champion_mae:.2f}). Sending alert to Grafana."
            )
            push_model_metrics(
                config.prometheus_pushgateway,
                champion_mae=champion_mae,
                challenger_mae=challenger_mae,
                champion=current_champion,
            )

    # Select champion
    champion = select_champion(
        prophet_mae, xgboost_mae, baseline_mae, current_champion=current_champion
    )

    return TrainingResult(
        prophet_mae=prophet_mae,
        xgboost_mae=xgboost_mae,
        baseline_mae=baseline_mae,
        champion=champion,
        prophet_model=prophet_model,
        xgboost_model=xgboost_model,
    )
