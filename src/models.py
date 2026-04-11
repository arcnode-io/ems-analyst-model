"""Data models and schemas for timeseries prediction pipeline."""

import enum
from datetime import datetime
from typing import Final

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class PredictiveModels(enum.StrEnum):
    """Supported predictive models."""

    PROPHET = "Prophet"
    XGBOOST = "XGBoost"


class TimeseriesDataSchema(pa.DataFrameModel):
    """Raw timeseries data schema from database."""

    timestamp: datetime = pa.Field(nullable=False)
    value: float = pa.Field(nullable=False, ge=0)

    class Config:
        strict = True
        coerce = True


class ProphetInputSchema(pa.DataFrameModel):
    """Prophet model input schema (pandas required)."""

    ds: datetime = pa.Field(nullable=False)
    y: float = pa.Field(nullable=False, ge=0)

    class Config:
        strict = True
        coerce = True


class XGBoostFeaturesSchema(pa.DataFrameModel):
    """XGBoost time-based features schema."""

    hour: int = pa.Field(nullable=False, ge=0, le=23)
    day: int = pa.Field(nullable=False, ge=1, le=31)
    month: int = pa.Field(nullable=False, ge=1, le=12)
    year: int = pa.Field(nullable=False)
    dayofweek: int = pa.Field(nullable=False, ge=0, le=6)
    dayofyear: int = pa.Field(nullable=False, ge=1, le=366)
    weekofyear: int = pa.Field(nullable=False, ge=1, le=53)

    class Config:
        strict = True
        coerce = True


class ModelMetricsSchema(pa.DataFrameModel):
    """Model evaluation metrics schema."""

    model: str = pa.Field(nullable=False)
    mae: float = pa.Field(nullable=False, ge=0)
    rmse: float = pa.Field(nullable=False, ge=0)

    class Config:
        strict = True
        coerce = True


# Type aliases for validated DataFrames
TimeseriesData = DataFrame[TimeseriesDataSchema]
XGBoostFeatures = DataFrame[XGBoostFeaturesSchema]
ModelMetrics = DataFrame[ModelMetricsSchema]


TRAIN_TEST_SPLIT_DAYS: Final[int] = 30

# XGBoost hyperparameters
XGBOOST_N_ESTIMATORS: Final[int] = 100
XGBOOST_LEARNING_RATE: Final[float] = 0.1
XGBOOST_MAX_DEPTH: Final[int] = 5
XGBOOST_RANDOM_STATE: Final[int] = 42

# XGBoost feature columns
XGBOOST_FEATURE_COLUMNS: Final[list[str]] = [
    "hour",
    "day",
    "month",
    "year",
    "dayofweek",
    "dayofyear",
    "weekofyear",
]

# Scheduling
SCHEDULE_POLL_INTERVAL_SECONDS: Final[int] = 60
